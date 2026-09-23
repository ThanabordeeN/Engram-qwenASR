#!/usr/bin/env python3
"""Measure BF16/NF4 size, peak VRAM, and end-to-end ASR latency on the same 300 clips.

Run: .venv/bin/python benchmark_step750_memory_latency.py
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Embedding4bit, Linear4bit

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb"
OUT = ROOT / "Qwen3ASR_Thai_Engram_ROCm" / "outputs"
STEP = 750
GROUP_SIZE = 32
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.bfloat16
os.chdir(ROOT)
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def exec_cells(ns: dict, indices: list[int]) -> None:
    for index in indices:
        source = "".join(nb["cells"][index].get("source", []))
        if source.strip():
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), ns)


def quantize_model(model: torch.nn.Module) -> dict[str, int]:
    counts = {"linear_modules": 0, "embedding_modules": 0}

    def linear(src: torch.nn.Linear) -> Linear4bit:
        dst = Linear4bit(
            src.in_features, src.out_features, bias=src.bias is not None,
            compute_dtype=COMPUTE_DTYPE, compress_statistics=False,
            quant_type=QUANT_TYPE, device="cpu",
        )
        dst.weight.blocksize = GROUP_SIZE
        dst.load_state_dict({key: value.detach().cpu() for key, value in src.state_dict().items()})
        dst = dst.to(src.weight.device)
        counts["linear_modules"] += 1
        return dst

    def embedding(src: torch.nn.Embedding) -> Embedding4bit:
        dst = Embedding4bit(
            src.num_embeddings, src.embedding_dim, dtype=src.weight.dtype,
            quant_type=QUANT_TYPE, device="cpu",
        )
        dst.weight.blocksize = GROUP_SIZE
        dst.padding_idx = src.padding_idx
        dst.max_norm = src.max_norm
        dst.norm_type = src.norm_type
        dst.scale_grad_by_freq = src.scale_grad_by_freq
        dst.sparse = src.sparse
        dst.load_state_dict({"weight": src.weight.detach().cpu()})
        dst = dst.to(src.weight.device)
        counts["embedding_modules"] += 1
        return dst

    def visit(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, (Linear4bit, Embedding4bit)):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, linear(child))
            elif isinstance(child, torch.nn.Embedding):
                setattr(parent, name, embedding(child))
            else:
                visit(child)

    visit(model)
    gc.collect()
    torch.cuda.empty_cache()
    return counts


def model_tensor_bytes(model: torch.nn.Module) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0

    def add(tensor: object) -> None:
        nonlocal total
        if not torch.is_tensor(tensor):
            return
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), size)
        if size and key not in seen:
            seen.add(key)
            total += size

    for tensor in list(model.parameters()) + list(model.buffers()):
        add(tensor)
    for module in model.modules():
        weight = getattr(module, "weight", None)
        state = getattr(weight, "quant_state", None)
        if state is not None:
            for name in ("absmax", "code", "offset"):
                add(getattr(state, name, None))
            nested = getattr(state, "state2", None)
            if nested is not None:
                for name in ("absmax", "code", "offset"):
                    add(getattr(nested, name, None))
    return total


def run_variant(name: str, ns: dict, records: list[dict], enabled: bool) -> tuple[dict, list[str]]:
    model = ns["model"]
    torch.cuda.synchronize()
    loaded_allocated = torch.cuda.memory_allocated()
    tensor_bytes = model_tensor_bytes(model)

    # One excluded warm-up primes kernels/caches; report latency for all 300 clips.
    ns["transcribe_record"](records[0], enabled)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    predictions: list[str] = []
    start = time.perf_counter()
    for record in ns["tqdm"](records, desc=name):
        sample_start = time.perf_counter()
        predictions.append(ns["transcribe_record"](record, enabled))
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - sample_start)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ordered = sorted(latencies)
    metrics = {
        "samples": len(records),
        "corpus_cer": float(ns["corpus_cer"]([r["text"] for r in records], predictions)),
        "mean_sample_cer": float(statistics.mean(
            ns["cer"](ns["thai_cer_norm"](r["text"]), ns["thai_cer_norm"](p))
            for r, p in zip(records, predictions)
        )),
        "model_tensor_storage_bytes": tensor_bytes,
        "allocated_after_load_bytes": loaded_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "inference_total_seconds": elapsed,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": ordered[len(ordered) // 2],
        "latency_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "throughput_samples_per_second": len(records) / elapsed,
    }
    print(f"{name}: CER={metrics['corpus_cer']:.6f}; mean={elapsed / len(records):.3f}s/clip; "
          f"peak VRAM={metrics['peak_allocated_bytes'] / 2**30:.3f} GiB; "
          f"tensor bytes={tensor_bytes / 2**30:.3f} GiB", flush=True)
    return metrics, predictions


def run_unengram(records: list[dict], expected_base: list[str], expected_nf4: list[str]) -> dict:
    ns: dict = {"__name__": "__main__"}
    exec_cells(ns, [4, 6])
    ns["RUNTIME"] = {"enabled": False}
    ns["ENGRAM_WRAPPERS"] = {}
    ns["normalize_thai_transcript"] = normalize_thai_transcript
    ns["scaling_eval_records"] = records
    exec_cells(ns, [17])

    bf16, preds = run_variant("Baseline BF16", ns, records, False)
    bf16["predictions_match_saved_eval"] = preds == expected_base
    del preds

    start = time.perf_counter()
    counts = quantize_model(ns["model"])
    quantize_seconds = time.perf_counter() - start
    nf4, preds = run_variant("Baseline NF4 W4", ns, records, False)
    nf4["quantized_modules"] = counts
    nf4["quantization_seconds"] = quantize_seconds
    nf4["predictions_match_saved_eval"] = preds == expected_nf4
    del preds
    result = {"baseline_bf16": bf16, "baseline_nf4_w4": nf4}
    ns.clear()
    del ns
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_engram(records: list[dict], expected_bf16: list[str], expected_nf4: list[str]) -> dict:
    ns: dict = {"__name__": "__main__"}
    exec_cells(ns, [4, 6, 8, 9, 10, 12])
    ns["normalize_thai_transcript"] = normalize_thai_transcript
    ns["scaling_eval_records"] = records
    exec_cells(ns, [17])
    ns["load_engram_checkpoint"](STEP)

    bf16, preds = run_variant("Step-750 Engram BF16", ns, records, True)
    bf16["predictions_match_saved_eval"] = preds == expected_bf16
    del preds

    start = time.perf_counter()
    counts = quantize_model(ns["model"])
    quantize_seconds = time.perf_counter() - start
    nf4, preds = run_variant("Step-750 Engram NF4 W4", ns, records, True)
    nf4["quantized_modules"] = counts
    nf4["quantization_seconds"] = quantize_seconds
    nf4["predictions_match_saved_eval"] = preds == expected_nf4
    del preds
    ns.clear()
    del ns
    gc.collect()
    torch.cuda.empty_cache()
    return {"engram_step750_bf16": bf16, "engram_step750_nf4_w4": nf4}


# Recover the fixed audio records once, and check the cache identifies that same set.
setup: dict = {"__name__": "__main__"}
exec_cells(setup, [4, 14, 15])
records = setup["scaling_eval_records"]
normalize_thai_transcript = setup["normalize_thai_transcript"]
base_cache = json.loads((OUT / "fixed300_base_predictions.json").read_text(encoding="utf-8"))
if len(records) != 300 or base_cache.get("source_skip") != 250_000 or base_cache["references"] != [r["text"] for r in records]:
    raise RuntimeError("The benchmark set does not match the saved fixed-300 baseline cache")
# Keep setup globals alive: the normalizer defined in that namespace references them.
expected_base_bf16 = base_cache["predictions"]
expected_base_nf4 = json.loads((OUT / "baseline_nf4_w4_predictions.json").read_text(encoding="utf-8"))["predictions"]
step750_json = json.loads((OUT / "preset_a_layer2_step750_evaluation.json").read_text(encoding="utf-8"))
expected_engram_bf16 = [row["engram_step750"] for row in step750_json["rows"]]
expected_engram_nf4 = json.loads((OUT / "step750_engram_nf4_w4_predictions.json").read_text(encoding="utf-8"))["predictions"]

print(f"Benchmarking {len(records)} clips on {torch.cuda.get_device_name(0)}; "
      f"ROCm {torch.version.hip}; bitsandbytes {bnb.__version__}", flush=True)
results = run_unengram(records, expected_base_bf16, expected_base_nf4)
results.update(run_engram(records, expected_engram_bf16, expected_engram_nf4))

summary = {
    "dataset": {"samples": 300, "source_skip": 250_000},
    "gpu": torch.cuda.get_device_name(0),
    "rocm_hip": torch.version.hip,
    "torch": torch.__version__,
    "bitsandbytes": bnb.__version__,
    "checkpoint": str(ROOT / "checkpoints" / "engram_step_000750.pt"),
    "quantization": {"method": "bitsandbytes NF4 weight-only", "bits": 4, "group_size": GROUP_SIZE,
                     "compute_dtype": str(COMPUTE_DTYPE), "double_quant": False},
    "latency_method": "end-to-end audio preprocessing + greedy ASR generation; one warm-up excluded; 300 timed clips",
    "storage_method": "unique live model parameter/buffer/quant-state tensor storage; peak VRAM is PyTorch CUDA allocator stats",
    "variants": results,
}
path = OUT / "step750_nf4_memory_latency.json"
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", path, flush=True)

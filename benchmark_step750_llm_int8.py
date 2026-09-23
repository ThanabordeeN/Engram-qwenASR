#!/usr/bin/env python3
"""Benchmark bitsandbytes LLM.int8 (PyTorch/ROCm, no GGUF) on the fixed 300 clips.
For bounded runs, evaluate --start/--limit chunks per --variant, then pass --combine.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import statistics
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Linear8bitLt
from jiwer import cer, process_characters

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb"
OUT = ROOT / "Qwen3ASR_Thai_Engram_ROCm" / "outputs"
PART_DIR = Path("/tmp/qwen3asr_llm_int8_parts")
STEP = 750
INT8_THRESHOLD = 6.0
# bitsandbytes logs this expected BF16-to-FP16 cast for every linear/token; suppress the flood.
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)
os.chdir(ROOT)
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def exec_cells(ns: dict, indices: list[int]) -> None:
    for index in indices:
        source = "".join(nb["cells"][index].get("source", []))
        if source.strip():
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), ns)


def quantize_linears(model: torch.nn.Module) -> dict[str, int]:
    counts = {"linear_modules": 0}

    def replace(src: torch.nn.Linear) -> Linear8bitLt:
        dst = Linear8bitLt(
            src.in_features, src.out_features, bias=src.bias is not None,
            has_fp16_weights=False, threshold=INT8_THRESHOLD, device="cpu",
        )
        dst.load_state_dict({key: value.detach().cpu() for key, value in src.state_dict().items()})
        dst = dst.to(src.weight.device)
        counts["linear_modules"] += 1
        return dst

    def visit(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, Linear8bitLt):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, replace(child))
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
    # Linear8bitLt keeps per-output-channel scales outside the Parameter/buffer lists.
    for module in model.modules():
        weight = getattr(module, "weight", None)
        add(getattr(weight, "CB", None))
        add(getattr(weight, "SCB", None))
        state = getattr(module, "state", None)
        add(getattr(state, "CB", None))
        add(getattr(state, "SCB", None))
    return total


def run_variant(name: str, ns: dict, records: list[dict], enabled: bool) -> tuple[dict, list[str]]:
    model = ns["model"]
    # First forward materializes Linear8bitLt scale tensors; exclude it as warm-up.
    ns["transcribe_record"](records[0], enabled)
    torch.cuda.synchronize()
    allocated_after_warmup = torch.cuda.memory_allocated()
    tensor_bytes = model_tensor_bytes(model)
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
    refs = [record["text"] for record in records]
    refs_norm = [ns["thai_cer_norm"](text) for text in refs]
    preds_norm = [ns["thai_cer_norm"](text) for text in predictions]
    aligned = process_characters(refs_norm, preds_norm)
    hits = int(aligned.hits)
    fp = int(aligned.substitutions + aligned.insertions)
    fn = int(aligned.substitutions + aligned.deletions)
    precision = hits / (hits + fp) if hits + fp else 0.0
    recall = hits / (hits + fn) if hits + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "samples": len(records),
        "corpus_cer": float(ns["corpus_cer"](refs, predictions)),
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "model_tensor_storage_bytes": tensor_bytes,
        "allocated_after_warmup_bytes": allocated_after_warmup,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "inference_total_seconds": elapsed,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": ordered[len(ordered) // 2],
        "latency_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "sample_latency_seconds": latencies,
        "throughput_samples_per_second": len(records) / elapsed,
    }
    print(f"{name}: CER={metrics['corpus_cer']:.6f}; F1={f1:.6f}; "
          f"mean={metrics['latency_mean_seconds']:.3f}s/clip; "
          f"peak VRAM={metrics['peak_allocated_bytes'] / 2**30:.3f} GiB; "
          f"tensor storage={tensor_bytes / 2**30:.3f} GiB", flush=True)
    return metrics, predictions


def load_records() -> tuple[list[dict], dict]:
    ns: dict = {"__name__": "__main__"}
    exec_cells(ns, [4, 14, 15])
    records = ns["scaling_eval_records"]
    cache = json.loads((OUT / "fixed300_base_predictions.json").read_text(encoding="utf-8"))
    if len(records) != 300 or cache.get("source_skip") != 250_000 or cache["references"] != [r["text"] for r in records]:
        raise RuntimeError("The benchmark set does not match the saved fixed-300 baseline cache")
    return records, ns


def evaluate(records: list[dict], setup: dict, engram: bool, start_index: int) -> dict:
    ns: dict = {"__name__": "__main__"}
    cells = [4, 6, 8, 9, 10, 12] if engram else [4, 6]
    exec_cells(ns, cells)
    if engram:
        ns["load_engram_checkpoint"](STEP)
    else:
        ns["RUNTIME"] = {"enabled": False}
        ns["ENGRAM_WRAPPERS"] = {}
    ns["normalize_thai_transcript"] = setup["normalize_thai_transcript"]
    ns["scaling_eval_records"] = records
    exec_cells(ns, [17])

    start = time.perf_counter()
    counts = quantize_linears(ns["model"])
    quantize_seconds = time.perf_counter() - start
    name = "Engram step 750 LLM.int8" if engram else "Baseline LLM.int8"
    metrics, predictions = run_variant(name, ns, records, engram)
    metrics["quantized_modules"] = counts
    metrics["quantization_seconds"] = quantize_seconds
    metrics["quantization"] = {
        "method": "bitsandbytes LLM.int8 / Linear8bitLt",
        "threshold": INT8_THRESHOLD,
        "has_fp16_weights": False,
        "scope": "Linear layers only; embeddings and other non-linear parameters remain in their original dtype",
    }
    result = {
        "variant": name,
        "start_index": start_index,
        "metrics": metrics,
        "references": [record["text"] for record in records],
        "predictions": predictions,
    }
    PART_DIR.mkdir(parents=True, exist_ok=True)
    part_name = "engram" if engram else "baseline"
    part_path = PART_DIR / f"{part_name}-{start_index:03d}-{len(records):03d}.json"
    part_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print("Saved part:", part_path, flush=True)
    ns.clear()
    del ns
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def combine_variant(records: list[dict], setup: dict, engram: bool) -> dict:
    part_name = "engram" if engram else "baseline"
    parts = sorted(PART_DIR.glob(f"{part_name}-*.json"), key=lambda path: json.loads(path.read_text(encoding="utf-8"))["start_index"])
    expected_start = 0
    references: list[str] = []
    predictions: list[str] = []
    latencies: list[float] = []
    part_metrics: list[dict] = []
    for path in parts:
        part = json.loads(path.read_text(encoding="utf-8"))
        start = part["start_index"]
        refs = part["references"]
        if start != expected_start or refs != [row["text"] for row in records[start:start + len(refs)]]:
            raise RuntimeError(f"Missing, overlapping, or mismatched benchmark chunk: {path}")
        if len(part["predictions"]) != len(refs):
            raise RuntimeError(f"Prediction/reference length mismatch in {path}")
        references.extend(refs)
        predictions.extend(part["predictions"])
        latencies.extend(float(x) for x in part["metrics"]["sample_latency_seconds"])
        part_metrics.append(part["metrics"])
        expected_start += len(refs)
    if expected_start != len(records):
        raise RuntimeError(f"{part_name}: only {expected_start}/{len(records)} clips are present")

    normalize = setup["normalize_thai_transcript"]
    refs_norm = [re.sub(r"\s+", "", normalize(text).lower()) for text in references]
    preds_norm = [re.sub(r"\s+", "", normalize(text).lower()) for text in predictions]
    aligned = process_characters(refs_norm, preds_norm)
    hits = int(aligned.hits)
    fp = int(aligned.substitutions + aligned.insertions)
    fn = int(aligned.substitutions + aligned.deletions)
    precision = hits / (hits + fp) if hits + fp else 0.0
    recall = hits / (hits + fn) if hits + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ordered = sorted(latencies)
    tensor_bytes = {m["model_tensor_storage_bytes"] for m in part_metrics}
    if len(tensor_bytes) != 1 or any(m["quantized_modules"] != part_metrics[0]["quantized_modules"] for m in part_metrics):
        raise RuntimeError(f"Inconsistent LLM.int8 model storage across {part_name} chunks")
    total_seconds = sum(m["inference_total_seconds"] for m in part_metrics)
    metrics = {
        "samples": len(records),
        "corpus_cer": float(cer("".join(refs_norm), "".join(preds_norm))),
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "model_tensor_storage_bytes": tensor_bytes.pop(),
        "allocated_after_warmup_bytes": max(m["allocated_after_warmup_bytes"] for m in part_metrics),
        "peak_allocated_bytes": max(m["peak_allocated_bytes"] for m in part_metrics),
        "peak_reserved_bytes": max(m["peak_reserved_bytes"] for m in part_metrics),
        "inference_total_seconds": total_seconds,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": ordered[len(ordered) // 2],
        "latency_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "sample_latency_seconds": latencies,
        "throughput_samples_per_second": len(records) / total_seconds,
        "quantized_modules": part_metrics[0]["quantized_modules"],
        "quantization_seconds": sum(m["quantization_seconds"] for m in part_metrics),
        "quantization": part_metrics[0]["quantization"],
    }
    name = "Engram step 750 LLM.int8" if engram else "Baseline LLM.int8"
    result = {"variant": name, "metrics": metrics, "references": references, "predictions": predictions}
    filename = "step750_engram_llm_int8_w8_predictions.json" if engram else "baseline_llm_int8_w8_predictions.json"
    (OUT / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Combined {name}: CER={metrics['corpus_cer']:.6f}; F1={f1:.6f}; "
          f"mean={metrics['latency_mean_seconds']:.3f}s/clip; storage={metrics['model_tensor_storage_bytes'] / 2**30:.3f} GiB")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="First offset in the fixed 300-sample set")
    parser.add_argument("--limit", type=int, default=300, help="Maximum samples in this chunk")
    parser.add_argument("--variant", choices=("baseline", "engram", "both"), default="both")
    parser.add_argument("--combine", action="store_true", help="Combine completed chunks into final results")
    args = parser.parse_args()
    all_records, setup = load_records()
    if args.combine:
        base = combine_variant(all_records, setup, engram=False)
        engram = combine_variant(all_records, setup, engram=True)
        summary = {
            "dataset": {"samples": len(all_records), "source_skip": 250_000},
            "gpu": torch.cuda.get_device_name(0),
            "rocm_hip": torch.version.hip,
            "torch": torch.__version__,
            "bitsandbytes": bnb.__version__,
            "checkpoint": str(ROOT / "checkpoints" / "engram_step_000750.pt"),
            "quantization": {
                "method": "bitsandbytes LLM.int8 (Linear8bitLt)",
                "bits": 8,
                "threshold": INT8_THRESHOLD,
                "has_fp16_weights": False,
                "scope": "Linear layers only; embeddings and other non-linear parameters remain in BF16",
                "gguf": False,
            },
            "latency_method": "end-to-end audio preprocessing + greedy ASR generation; one warm-up per chunk excluded",
            "storage_method": "unique live model parameter/buffer storage plus Linear8bitLt CB/SCB; peak VRAM is PyTorch allocator stats",
            "variants": {"baseline_llm_int8": base, "engram_step750_llm_int8": engram},
        }
        path = OUT / "step750_llm_int8_memory_latency.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved:", path, flush=True)
        return

    if args.start < 0 or args.limit < 1 or args.start >= len(all_records):
        raise ValueError("Use a valid --start and a positive --limit")
    records = all_records[args.start:args.start + args.limit]
    print(f"Benchmarking clips {args.start}–{args.start + len(records) - 1} of {len(all_records)} "
          f"on {torch.cuda.get_device_name(0)}; ROCm {torch.version.hip}; bitsandbytes {bnb.__version__}; no GGUF", flush=True)
    if args.variant in ("baseline", "both"):
        evaluate(records, setup, engram=False, start_index=args.start)
    if args.variant in ("engram", "both"):
        evaluate(records, setup, engram=True, start_index=args.start)
    setup.clear()


if __name__ == "__main__":
    main()

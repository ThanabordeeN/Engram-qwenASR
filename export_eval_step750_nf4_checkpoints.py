#!/usr/bin/env python3
"""Export BF16-baseline/NF4 and Step-750-Engram/NF4 checkpoints, then reload/evaluate them.

Run from the project root: .venv/bin/python export_eval_step750_nf4_checkpoints.py
The .pt bundles contain packed NF4 tensors, QuantState data, and every remaining
parameter/buffer. Model configuration and processor assets are exported beside them.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import statistics
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Embedding4bit, Linear4bit, Params4bit

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb"
OUTPUT_DIR = ROOT / "Qwen3ASR_Thai_Engram_ROCm" / "outputs"
EXPORT_DIR = ROOT / "Qwen3ASR_Thai_Engram_ROCm" / "exports" / "nf4"
STEP = 750
GROUP_SIZE = 32
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.bfloat16
FORMAT = "qwen3-asr-bnb-nf4-inference-v1"
os.chdir(ROOT)
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def exec_cells(ns: dict, indices: list[int]) -> None:
    for index in indices:
        source = "".join(nb["cells"][index].get("source", []))
        if source.strip():
            exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), ns)


def build_model(engram: bool, load_step750: bool = False) -> dict:
    ns: dict = {"__name__": "__main__"}
    exec_cells(ns, [4, 6])
    if engram:
        exec_cells(ns, [8, 9, 10, 12])
        if load_step750:
            ns["load_engram_checkpoint"](STEP)
    else:
        ns["RUNTIME"] = {"enabled": False}
        ns["ENGRAM_WRAPPERS"] = {}
    ns["normalize_thai_transcript"] = normalize_thai_transcript
    ns["scaling_eval_records"] = records
    exec_cells(ns, [17])
    return ns


def convert_to_nf4(model: torch.nn.Module) -> dict[str, int]:
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


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def checkpoint_payload(ns: dict, variant: str, engram: bool, counts: dict) -> dict:
    model = ns["model"]
    modules = {}
    quantized_parameter_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, (Linear4bit, Embedding4bit)):
            continue
        if module.weight.quant_state is None:
            raise RuntimeError(f"NF4 weights missing quant state: {name}")
        key = f"{name}.weight" if name else "weight"
        quantized_parameter_names.add(key)
        item = {
            "kind": "linear4bit" if isinstance(module, Linear4bit) else "embedding4bit",
            # .data is intentional: copying Params4bit itself to CPU changes its packed layout.
            "packed_weight": module.weight.data.detach().cpu().clone(),
            "quant_state": cpu_tree(module.weight.quant_state.as_dict()),
        }
        if isinstance(module, Linear4bit):
            item["compute_dtype"] = str(module.compute_dtype)
        else:
            item["output_dtype"] = str(module.dtype)
        modules[name] = item

    named_parameters = dict(model.named_parameters())
    if not quantized_parameter_names.issubset(named_parameters):
        missing = sorted(quantized_parameter_names - set(named_parameters))
        raise RuntimeError(f"Packed parameter names missing from model: {missing[:5]}")
    dense_parameters = {
        name: param.detach().cpu().clone()
        for name, param in named_parameters.items()
        if name not in quantized_parameter_names
    }
    buffers = {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()}
    engram_meta = {
        "enabled": engram,
        "checkpoint_step": STEP if engram else None,
        "config": ns.get("ENGRAM_CONFIG") if engram else None,
    }
    return {
        "metadata": {
            "format": FORMAT,
            "variant": variant,
            "base_model_id": ns["MODEL_ID"],
            "step": STEP if engram else None,
            "torch": str(torch.__version__),
            "bitsandbytes": bnb.__version__,
            "rocm_hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "quantization": {
                "method": "bitsandbytes NF4 weight-only",
                "bits": 4,
                "group_size": GROUP_SIZE,
                "compute_dtype": str(COMPUTE_DTYPE),
                "double_quant": False,
                "applied_to": "all nn.Linear and nn.Embedding weights in Qwen and Engram",
            },
            "engram": engram_meta,
            "converted_modules": counts,
            "reconstruct_with_base_model_id": True,
        },
        "quantized_modules": modules,
        "dense_parameters": dense_parameters,
        "buffers": buffers,
    }


def save_checkpoint(ns: dict, variant: str, engram: bool, counts: dict) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{variant}.pt"
    temp_path = path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload(ns, variant, engram, counts), temp_path)
    temp_path.replace(path)
    print(f"Exported {path} ({path.stat().st_size / 2**30:.3f} GiB)", flush=True)
    return path


def load_checkpoint_into_model(model: torch.nn.Module, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    if meta["format"] != FORMAT or meta["bitsandbytes"] != bnb.__version__:
        raise RuntimeError("Checkpoint format or bitsandbytes version mismatch")
    quantized_modules = dict(model.named_modules())
    if set(payload["quantized_modules"]) != {
        name for name, module in quantized_modules.items()
        if isinstance(module, (Linear4bit, Embedding4bit))
    }:
        raise RuntimeError("Checkpoint quantized-module layout does not match rebuilt architecture")

    params = dict(model.named_parameters())
    quantized_names = {f"{name}.weight" for name in payload["quantized_modules"]}
    dense = payload["dense_parameters"]
    if set(params) != set(dense) | quantized_names:
        missing = sorted(set(params) - set(dense) - quantized_names)
        extra = sorted((set(dense) | quantized_names) - set(params))
        raise RuntimeError(f"Checkpoint parameter mismatch; missing={missing[:5]} extra={extra[:5]}")
    buffers = dict(model.named_buffers())
    if set(buffers) != set(payload["buffers"]):
        raise RuntimeError("Checkpoint buffer layout does not match rebuilt architecture")

    with torch.no_grad():
        for name, tensor in dense.items():
            param = params[name]
            if tuple(param.shape) != tuple(tensor.shape):
                raise RuntimeError(f"Parameter shape mismatch: {name}")
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        for name, tensor in payload["buffers"].items():
            buffer = buffers[name]
            if tuple(buffer.shape) != tuple(tensor.shape):
                raise RuntimeError(f"Buffer shape mismatch: {name}")
            buffer.copy_(tensor.to(device=buffer.device, dtype=buffer.dtype))

        for name, item in payload["quantized_modules"].items():
            module = quantized_modules[name]
            expected_kind = "linear4bit" if isinstance(module, Linear4bit) else "embedding4bit"
            if item["kind"] != expected_kind:
                raise RuntimeError(f"Quantized module type mismatch: {name}")
            packed_weight = item["packed_weight"]
            quant_state = item["quant_state"]
            restored = Params4bit.from_prequantized(
                data=packed_weight,
                quantized_stats=quant_state,
                requires_grad=False,
                device=module.weight.device,
            )
            # Avoid a module back-reference; forward recovers state from the Param4bit.
            module.weight = restored
            module.quant_state = restored.quant_state
            if isinstance(module, Embedding4bit):
                module.dtype = getattr(torch, item["output_dtype"].removeprefix("torch."))
            else:
                module.compute_dtype = getattr(torch, item["compute_dtype"].removeprefix("torch."))

    model.eval()
    return meta


def evaluate_checkpoint(ns: dict, path: Path, variant: str, expected: list[str]) -> dict:
    meta = load_checkpoint_into_model(ns["model"], path)
    predictions = [
        ns["transcribe_record"](record, meta["engram"]["enabled"])
        for record in ns["tqdm"](records, desc=f"Checkpoint eval {variant}")
    ]
    if len(predictions) != 300:
        raise RuntimeError(f"Expected 300 predictions, got {len(predictions)}")
    match = predictions == expected
    if not match:
        raise RuntimeError(f"Reloaded {variant} predictions differ from the saved in-memory evaluation")
    refs = [record["text"] for record in records]
    sample_cers = [
        ns["cer"](ns["thai_cer_norm"](ref), ns["thai_cer_norm"](pred))
        for ref, pred in zip(refs, predictions)
    ]
    result = {
        "variant": variant,
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "samples": len(refs),
        "source_skip": 250_000,
        "corpus_cer": float(ns["corpus_cer"](refs, predictions)),
        "mean_sample_cer": float(statistics.mean(sample_cers)),
        "predictions_match_previous_eval": match,
        "gpu": torch.cuda.get_device_name(0),
        "rocm_hip": torch.version.hip,
        "quantization": meta["quantization"],
    }
    stem = "baseline_nf4_checkpoint_eval" if not meta["engram"]["enabled"] else "step750_engram_nf4_checkpoint_eval"
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps({**result, "references": refs, "predictions": predictions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (OUTPUT_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "reference", "prediction", "cer"])
        writer.writerows((i, ref, pred, float(sample_cers[i])) for i, (ref, pred) in enumerate(zip(refs, predictions)))
    print(f"Reloaded {variant}: CER={result['corpus_cer']:.6f}; checkpoint={path.stat().st_size / 2**30:.3f} GiB; predictions match={match}", flush=True)
    return result


# Rebuild the exact evaluation set once, then retain its audio arrays on the CPU.
setup: dict = {"__name__": "__main__"}
exec_cells(setup, [4, 14, 15])
records = setup["scaling_eval_records"]
normalize_thai_transcript = setup["normalize_thai_transcript"]
base_cache = json.loads((OUTPUT_DIR / "fixed300_base_predictions.json").read_text(encoding="utf-8"))
if len(records) != 300 or base_cache.get("source_skip") != 250_000 or base_cache["references"] != [r["text"] for r in records]:
    raise RuntimeError("The current records do not match the fixed 300-sample evaluation set")
expected_base = json.loads((OUTPUT_DIR / "baseline_nf4_w4_predictions.json").read_text(encoding="utf-8"))["predictions"]
expected_engram = json.loads((OUTPUT_DIR / "step750_engram_nf4_w4_predictions.json").read_text(encoding="utf-8"))["predictions"]
if not EXPORT_DIR.exists():
    EXPORT_DIR.mkdir(parents=True)

# Keep setup globals alive: the normalizer defined there closes over them.

# Export NF4 checkpoints without changing the original training checkpoint or notebook.
ns = build_model(engram=False)
# Save reusable processor/config assets next to the packed checkpoint bundles.
ns["asr"].processor.save_pretrained(EXPORT_DIR / "processor")
ns["model"].config.save_pretrained(EXPORT_DIR / "model_config")
counts = convert_to_nf4(ns["model"])
base_path = save_checkpoint(ns, "baseline_nf4_w4", False, counts)
ns.clear(); del ns; gc.collect(); torch.cuda.empty_cache()

ns = build_model(engram=True, load_step750=True)
counts = convert_to_nf4(ns["model"])
engram_path = save_checkpoint(ns, "step750_engram_nf4_w4", True, counts)
ns.clear(); del ns; gc.collect(); torch.cuda.empty_cache()

# Rebuild fresh architectures and load only the exported bundles before inference.
ns = build_model(engram=False)
convert_to_nf4(ns["model"])
base_result = evaluate_checkpoint(ns, base_path, "baseline_nf4_w4", expected_base)
ns.clear(); del ns; gc.collect(); torch.cuda.empty_cache()

ns = build_model(engram=True, load_step750=False)
convert_to_nf4(ns["model"])
engram_result = evaluate_checkpoint(ns, engram_path, "step750_engram_nf4_w4", expected_engram)
ns.clear(); del ns; gc.collect(); torch.cuda.empty_cache()

summary = {
    "format": FORMAT,
    "dataset": {"samples": 300, "source_skip": 250_000},
    "gpu": torch.cuda.get_device_name(0),
    "rocm_hip": torch.version.hip,
    "torch": torch.__version__,
    "bitsandbytes": bnb.__version__,
    "quantization": {"method": "bitsandbytes NF4 weight-only", "bits": 4, "group_size": GROUP_SIZE, "compute_dtype": str(COMPUTE_DTYPE), "double_quant": False},
    "checkpoints": {
        "baseline_nf4_w4": {"path": str(base_path), "bytes": base_path.stat().st_size, "eval": base_result},
        "step750_engram_nf4_w4": {"path": str(engram_path), "bytes": engram_path.stat().st_size, "eval": engram_result},
    },
}
summary_path = OUTPUT_DIR / "nf4_exported_checkpoint_eval_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
manifest_path = EXPORT_DIR / "manifest.json"
manifest_path.write_text(json.dumps({
    "format": FORMAT,
    "base_model_id": "Qwen/Qwen3-ASR-0.6B",
    "quantization": summary["quantization"],
    "checkpoints": {name: {"path": f"{name}.pt", "bytes": info["bytes"]} for name, info in summary["checkpoints"].items()},
    "processor_dir": "processor",
    "model_config_dir": "model_config",
    "load_note": "The .pt bundle overrides all parameters/buffers after rebuilding the architecture from the base model ID; load with export_eval_step750_nf4_checkpoints.py.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved evaluation summary:", summary_path, flush=True)
print("Saved checkpoint manifest:", manifest_path, flush=True)

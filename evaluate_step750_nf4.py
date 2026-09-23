#!/usr/bin/env python3
"""Paired BF16 vs NF4 evaluation for the best (step-750) Engram checkpoint.

Run from this project with: .venv/bin/python evaluate_step750_nf4.py
Uses the existing scaling notebook for model/data/inference definitions; does not
modify that notebook or any training checkpoint.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Embedding4bit, Linear4bit

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb"
STEP = 750
GROUP_SIZE = 32
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.bfloat16

os.chdir(ROOT)
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
ns: dict[str, object] = {"__name__": "__main__"}

# Reuse the tested setup and inference functions, but skip the notebook's
# installer cell and its multi-checkpoint evaluation cell.
for index in range(4, 19):
    cell = nb["cells"][index]
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and source.strip():
        print(f"Executing notebook cell {index}...", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), ns)

model = ns["model"]
records = ns["scaling_eval_records"]
refs = [row["text"] for row in records]
output_dir = ns["OUTPUT_DIR"]
base_cache = json.loads((output_dir / "fixed300_base_predictions.json").read_text(encoding="utf-8"))
if base_cache.get("source_skip") != 250_000 or base_cache.get("references") != refs or len(refs) != 300:
    raise RuntimeError("Fixed-set references do not match the existing 300-sample baseline cache")

# Load the best Engram checkpoint before conversion. All Linear and Embedding
# weights, in both Qwen and Engram, then use exactly the same NF4/group-32 W4
# quantizer and BF16 compute. Norms, biases, and convolution kernels stay BF16.
ns["load_engram_checkpoint"](STEP)
counts = {"linear_modules": 0, "embedding_modules": 0, "quantized_parameters": 0}


def quantize_linear(module: torch.nn.Linear) -> Linear4bit:
    target = Linear4bit(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        compute_dtype=COMPUTE_DTYPE,
        compress_statistics=False,
        quant_type=QUANT_TYPE,
        device="cpu",
    )
    target.weight.blocksize = GROUP_SIZE
    target.load_state_dict({key: value.detach().cpu() for key, value in module.state_dict().items()})
    target = target.to(module.weight.device)
    counts["linear_modules"] += 1
    counts["quantized_parameters"] += module.weight.numel()
    return target


def quantize_embedding(module: torch.nn.Embedding) -> Embedding4bit:
    target = Embedding4bit(
        module.num_embeddings,
        module.embedding_dim,
        dtype=module.weight.dtype,
        quant_type=QUANT_TYPE,
        device="cpu",
    )
    target.weight.blocksize = GROUP_SIZE
    target.padding_idx = module.padding_idx
    target.max_norm = module.max_norm
    target.norm_type = module.norm_type
    target.scale_grad_by_freq = module.scale_grad_by_freq
    target.sparse = module.sparse
    target.load_state_dict({"weight": module.weight.detach().cpu()})
    target = target.to(module.weight.device)
    counts["embedding_modules"] += 1
    counts["quantized_parameters"] += module.weight.numel()
    return target


def convert_to_nf4(parent: torch.nn.Module) -> None:
    for name, child in list(parent.named_children()):
        if isinstance(child, (Linear4bit, Embedding4bit)):
            continue
        if isinstance(child, torch.nn.Linear):
            setattr(parent, name, quantize_linear(child))
        elif isinstance(child, torch.nn.Embedding):
            setattr(parent, name, quantize_embedding(child))
        else:
            convert_to_nf4(child)


convert_to_nf4(model)
model.eval()
torch.cuda.synchronize()
print("NF4 conversion:", counts, flush=True)


def metrics(predictions: list[str]) -> dict[str, object]:
    sample_cers = [
        ns["cer"](ns["thai_cer_norm"](ref), ns["thai_cer_norm"](pred))
        for ref, pred in zip(refs, predictions)
    ]
    return {
        "samples": len(refs),
        "corpus_cer": float(ns["corpus_cer"](refs, predictions)),
        "mean_sample_cer": float(sum(sample_cers) / len(sample_cers)),
        "sample_cers": [float(value) for value in sample_cers],
    }


def save_predictions(name: str, predictions: list[str], result: dict[str, object]) -> None:
    payload = {
        "variant": name,
        "step": STEP if "engram" in name else None,
        "source_skip": 250_000,
        "references": refs,
        "predictions": predictions,
        "metrics": {key: value for key, value in result.items() if key != "sample_cers"},
        "quantization": {
            "method": "bitsandbytes NF4 weight-only",
            "bits": 4,
            "group_size": GROUP_SIZE,
            "compute_dtype": str(COMPUTE_DTYPE),
        },
    }
    (output_dir / f"{name}_predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "reference", "prediction", "cer"])
        writer.writerows((i, ref, pred, result["sample_cers"][i]) for i, (ref, pred) in enumerate(zip(refs, predictions)))


base_w4_predictions = [
    ns["transcribe_record"](record, False)
    for record in ns["tqdm"](records, desc="Baseline NF4 W4")
]
base_w4_metrics = metrics(base_w4_predictions)
save_predictions("baseline_nf4_w4", base_w4_predictions, base_w4_metrics)
print(f"Baseline NF4 corpus CER: {base_w4_metrics['corpus_cer']:.6f}", flush=True)

engram_w4_predictions = [
    ns["transcribe_record"](record, True)
    for record in ns["tqdm"](records, desc="Step-750 Engram NF4 W4")
]
engram_w4_metrics = metrics(engram_w4_predictions)
save_predictions("step750_engram_nf4_w4", engram_w4_predictions, engram_w4_metrics)
print(f"Step-750 Engram NF4 corpus CER: {engram_w4_metrics['corpus_cer']:.6f}", flush=True)

# Include the already-verified BF16 arms, creating a 2x2 paired comparison.
step750_file = output_dir / "preset_a_layer2_step750_evaluation.json"
engram_bf16_summary = json.loads(step750_file.read_text(encoding="utf-8"))["summary"]
base_bf16_summary = {
    "corpus_cer": engram_bf16_summary["base_corpus_cer"],
    "mean_sample_cer": engram_bf16_summary["base_mean_sample_cer"],
}
engram_bf16_summary_out = {
    "corpus_cer": engram_bf16_summary["engram_corpus_cer"],
    "mean_sample_cer": engram_bf16_summary["engram_mean_sample_cer"],
}
base_bf16_predictions = base_cache["predictions"]
engram_bf16_json = json.loads(step750_file.read_text(encoding="utf-8"))
engram_bf16_predictions = [row["engram_step750"] for row in engram_bf16_json["rows"]]
if len(engram_bf16_predictions) != 300:
    raise RuntimeError("Existing step-750 BF16 predictions do not contain 300 rows")

comparison = {
    "dataset": {"samples": 300, "source_skip": 250_000},
    "gpu": torch.cuda.get_device_name(0),
    "rocm_hip": torch.version.hip,
    "bitsandbytes": bnb.__version__,
    "checkpoint": str(ROOT / "checkpoints" / "engram_step_000750.pt"),
    "quantization": {
        "method": "bitsandbytes NF4 weight-only",
        "bits": 4,
        "group_size": GROUP_SIZE,
        "compute_dtype": str(COMPUTE_DTYPE),
        "double_quant": False,
        "applied_to": "all Qwen and Engram nn.Linear/nn.Embedding weights",
        "left_unquantized": "biases, normalization/convolution weights, and 1-D parameters (base BF16; Engram checkpoint FP32)",
    },
    "quantized_module_counts": counts,
    "variants": {
        "baseline_bf16_unquantized": base_bf16_summary,
        "baseline_nf4_w4": {key: base_w4_metrics[key] for key in ["samples", "corpus_cer", "mean_sample_cer"]},
        "step750_engram_bf16_unquantized": engram_bf16_summary_out,
        "step750_engram_nf4_w4": {key: engram_w4_metrics[key] for key in ["samples", "corpus_cer", "mean_sample_cer"]},
    },
    "deltas_cer_percentage_points": {
        "baseline_nf4_minus_baseline_bf16": (base_w4_metrics["corpus_cer"] - base_bf16_summary["corpus_cer"]) * 100,
        "engram_nf4_minus_engram_bf16": (engram_w4_metrics["corpus_cer"] - engram_bf16_summary_out["corpus_cer"]) * 100,
        "engram_gain_over_baseline_at_bf16": (base_bf16_summary["corpus_cer"] - engram_bf16_summary_out["corpus_cer"]) * 100,
        "engram_gain_over_baseline_at_nf4": (base_w4_metrics["corpus_cer"] - engram_w4_metrics["corpus_cer"]) * 100,
    },
    "paired_sample_counts_vs_same_precision_baseline": {
        "nf4_improved": sum(e < b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
        "nf4_tied": sum(e == b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
        "nf4_regressed": sum(e > b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
        "bf16_improved": engram_bf16_summary["improved_samples"],
        "bf16_tied": engram_bf16_summary["tied_samples"],
        "bf16_regressed": engram_bf16_summary["regressed_samples"],
    },
}
summary_path = output_dir / "step750_nf4_quantization_comparison.json"
summary_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved paired comparison:", summary_path, flush=True)

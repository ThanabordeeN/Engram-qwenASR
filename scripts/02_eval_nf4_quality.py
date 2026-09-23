#!/usr/bin/env python3
"""NF4 quality: paired BF16 vs NF4 comparison for baseline and Engram step 750.

Produces:

    results/predictions/baseline_nf4_w4_predictions.{json,csv}
    results/predictions/step750_engram_nf4_w4_predictions.{json,csv}
    results/summaries/step750_nf4_quantization_comparison.json

This is the quality arm. ``scripts/03`` measures memory and latency on the same
quantized models and checks its predictions against the files written here, so
this script must run first.

Runtime on an RX 9070 XT: roughly 10 minutes.

    .venv/bin/python scripts/02_eval_nf4_quality.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    NF4_COMPUTE_DTYPE,
    NF4_GROUP_SIZE,
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    load_cached_references,
    quantize_nf4,
    sample_cer,
)
from qwen3asr_engram import Config, PROJECT_ROOT, Session, corpus_cer, write_json  # noqa: E402


def variant_metrics(refs, predictions) -> dict:
    sample_cers = [float(sample_cer(r, p)) for r, p in zip(refs, predictions)]
    return {
        "samples": len(refs),
        "corpus_cer": float(corpus_cer(refs, predictions)),
        "mean_sample_cer": float(sum(sample_cers) / len(sample_cers)),
        "sample_cers": sample_cers,
    }


def save_predictions(name: str, refs, predictions, result: dict) -> None:
    write_json(PREDICTIONS_DIR / f"{name}_predictions.json", {
        "variant": name,
        "step": STEP if "engram" in name else None,
        "source_skip": 250_000,
        "references": refs,
        "predictions": predictions,
        "metrics": {k: v for k, v in result.items() if k != "sample_cers"},
        "quantization": {
            "method": "bitsandbytes NF4 weight-only",
            "bits": 4,
            "group_size": NF4_GROUP_SIZE,
            "compute_dtype": str(NF4_COMPUTE_DTYPE),
        },
    })
    with (PREDICTIONS_DIR / f"{name}_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "reference", "prediction", "cer"])
        writer.writerows(
            (i, ref, pred, result["sample_cers"][i])
            for i, (ref, pred) in enumerate(zip(refs, predictions))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    refs = load_cached_references()

    session = Session(cfg, engram=True)
    records = session.collect_records()
    if [r["text"] for r in records] != refs:
        raise RuntimeError("Fixed-set references do not match the frozen 300-clip cache")
    session.load_checkpoint(STEP)

    counts = quantize_nf4(session.model)
    session.model.eval()
    torch.cuda.synchronize()
    print("NF4 conversion:", counts, flush=True)

    base_w4 = [session.transcribe(record, False)
               for record in tqdm(records, desc="Baseline NF4 W4")]
    base_w4_metrics = variant_metrics(refs, base_w4)
    save_predictions("baseline_nf4_w4", refs, base_w4, base_w4_metrics)
    print(f"Baseline NF4 corpus CER: {base_w4_metrics['corpus_cer']:.6f}", flush=True)

    engram_w4 = [session.transcribe(record, True)
                 for record in tqdm(records, desc="Step-750 Engram NF4 W4")]
    engram_w4_metrics = variant_metrics(refs, engram_w4)
    save_predictions("step750_engram_nf4_w4", refs, engram_w4, engram_w4_metrics)
    print(f"Step-750 Engram NF4 corpus CER: {engram_w4_metrics['corpus_cer']:.6f}", flush=True)
    session.close()

    # Fold in the already-verified BF16 arms to form the 2x2 paired comparison.
    step750 = json.loads((SUMMARIES_DIR / "preset_a_layer2_step750_evaluation.json")
                         .read_text(encoding="utf-8"))
    base_bf16 = {
        "corpus_cer": step750["summary"]["base_corpus_cer"],
        "mean_sample_cer": step750["summary"]["base_mean_sample_cer"],
    }
    engram_bf16 = {
        "corpus_cer": step750["summary"]["engram_corpus_cer"],
        "mean_sample_cer": step750["summary"]["engram_mean_sample_cer"],
    }
    engram_bf16_predictions = [row["engram_step750"] for row in step750["rows"]]
    if len(engram_bf16_predictions) != len(refs):
        raise RuntimeError("Step-750 BF16 predictions do not cover the fixed set")
    base_bf16_predictions = json.loads(
        (PREDICTIONS_DIR / "fixed300_base_predictions.json").read_text(encoding="utf-8"))["predictions"]
    bf16_sample_cers = [sample_cer(r, p) for r, p in zip(refs, base_bf16_predictions)]
    engram_bf16_sample_cers = [sample_cer(r, p) for r, p in zip(refs, engram_bf16_predictions)]

    comparison = {
        "dataset": {"samples": len(refs), "source_skip": cfg.eval_source_skip},
        "gpu": session.environment["gpu"],
        "rocm_hip": session.environment["rocm_hip"],
        "bitsandbytes": bnb.__version__,
        "checkpoint": str(cfg.checkpoint_paths()[STEP]),
        "quantization": {
            "method": "bitsandbytes NF4 weight-only",
            "bits": 4,
            "group_size": NF4_GROUP_SIZE,
            "compute_dtype": str(NF4_COMPUTE_DTYPE),
            "double_quant": False,
            "applied_to": "all Qwen and Engram nn.Linear/nn.Embedding weights",
            "left_unquantized": ("biases, normalization/convolution weights, and 1-D parameters "
                                 "(base BF16; Engram checkpoint FP32)"),
        },
        "quantized_module_counts": counts,
        "variants": {
            "baseline_bf16_unquantized": base_bf16,
            "baseline_nf4_w4": {k: base_w4_metrics[k] for k in ("samples", "corpus_cer", "mean_sample_cer")},
            "step750_engram_bf16_unquantized": engram_bf16,
            "step750_engram_nf4_w4": {k: engram_w4_metrics[k] for k in ("samples", "corpus_cer", "mean_sample_cer")},
        },
        "deltas_cer_percentage_points": {
            "baseline_nf4_minus_baseline_bf16": (base_w4_metrics["corpus_cer"] - base_bf16["corpus_cer"]) * 100,
            "engram_nf4_minus_engram_bf16": (engram_w4_metrics["corpus_cer"] - engram_bf16["corpus_cer"]) * 100,
            "engram_gain_over_baseline_at_bf16": (base_bf16["corpus_cer"] - engram_bf16["corpus_cer"]) * 100,
            "engram_gain_over_baseline_at_nf4": (base_w4_metrics["corpus_cer"] - engram_w4_metrics["corpus_cer"]) * 100,
        },
        "paired_sample_counts_vs_same_precision_baseline": {
            "nf4_improved": sum(e < b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
            "nf4_tied": sum(e == b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
            "nf4_regressed": sum(e > b for e, b in zip(engram_w4_metrics["sample_cers"], base_w4_metrics["sample_cers"])),
            "bf16_improved": sum(e < b for e, b in zip(engram_bf16_sample_cers, bf16_sample_cers)),
            "bf16_tied": sum(e == b for e, b in zip(engram_bf16_sample_cers, bf16_sample_cers)),
            "bf16_regressed": sum(e > b for e, b in zip(engram_bf16_sample_cers, bf16_sample_cers)),
        },
    }
    write_json(SUMMARIES_DIR / "step750_nf4_quantization_comparison.json", comparison)
    print("Saved paired comparison:", SUMMARIES_DIR / "step750_nf4_quantization_comparison.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LLM.int8 (bitsandbytes Linear8bitLt) quality, memory, and latency — no GGUF.

Produces:

    results/parts/llm_int8/{baseline,engram}-<start>-<count>.json   (chunk records)
    results/predictions/baseline_llm_int8_w8_predictions.json
    results/predictions/step750_engram_llm_int8_w8_predictions.json
    results/summaries/step750_llm_int8_memory_latency.json

LLM.int8 quantizes ``nn.Linear`` layers only; embeddings and every other
parameter stay BF16. It is not a format called "NF8", and it runs directly
through PyTorch/ROCm.

The 300 clips can be run in chunks so a long run can be resumed:

    .venv/bin/python scripts/05_benchmark_llm_int8.py --start 0   --limit 75
    .venv/bin/python scripts/05_benchmark_llm_int8.py --start 75  --limit 75
    .venv/bin/python scripts/05_benchmark_llm_int8.py --start 150 --limit 75
    .venv/bin/python scripts/05_benchmark_llm_int8.py --start 225 --limit 75
    .venv/bin/python scripts/05_benchmark_llm_int8.py --combine

``--combine`` refuses to run unless the chunks cover the fixed set exactly once,
in order, with matching references.

    .venv/bin/python scripts/05_benchmark_llm_int8.py --variant both   # single pass
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import logging
import statistics
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    LLM_INT8_THRESHOLD,
    PARTS_DIR,
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    benchmark_variant,
    char_prf,
    load_cached_references,
    quantize_llm_int8,
)
from qwen3asr_engram import (  # noqa: E402
    PROJECT_ROOT,
    Config,
    Session,
    corpus_cer,
    write_json,
)

# bitsandbytes logs the expected BF16-to-FP16 cast for every linear/token; suppress the flood.
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

PART_DIR = PARTS_DIR / "llm_int8"
VARIANT_NAMES = {False: "Baseline LLM.int8", True: "Engram step 750 LLM.int8"}
OUTPUT_FILES = {False: "baseline_llm_int8_w8_predictions.json",
                True: "step750_engram_llm_int8_w8_predictions.json"}


def evaluate(cfg: Config, records, engram: bool, start_index: int) -> dict:
    session = Session(cfg, engram=engram)
    if engram:
        session.load_checkpoint(STEP)

    start = time.perf_counter()
    counts = quantize_llm_int8(session.model)
    quantize_seconds = time.perf_counter() - start

    name = VARIANT_NAMES[engram]
    metrics, predictions = benchmark_variant(session, records, engram, name,
                                             tensor_bytes_after_warmup=True)
    metrics["quantized_modules"] = counts
    metrics["quantization_seconds"] = quantize_seconds
    metrics["quantization"] = {
        "method": "bitsandbytes LLM.int8 / Linear8bitLt",
        "threshold": LLM_INT8_THRESHOLD,
        "has_fp16_weights": False,
        "scope": ("Linear layers only; embeddings and other non-linear parameters "
                  "remain in their original dtype"),
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
    session.close()
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def combine_variant(cfg: Config, records, engram: bool) -> dict:
    part_name = "engram" if engram else "baseline"
    parts = sorted(
        PART_DIR.glob(f"{part_name}-*.json"),
        key=lambda path: json.loads(path.read_text(encoding="utf-8"))["start_index"],
    )
    if not parts:
        raise RuntimeError(f"No {part_name} chunks found in {PART_DIR}")

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

    tensor_bytes = {m["model_tensor_storage_bytes"] for m in part_metrics}
    if len(tensor_bytes) != 1 or any(m["quantized_modules"] != part_metrics[0]["quantized_modules"]
                                     for m in part_metrics):
        raise RuntimeError(f"Inconsistent LLM.int8 model storage across {part_name} chunks")

    ordered = sorted(latencies)
    total_seconds = sum(m["inference_total_seconds"] for m in part_metrics)
    metrics = {
        "samples": len(records),
        "corpus_cer": float(corpus_cer(references, predictions)),
        **char_prf(references, predictions),
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
    name = VARIANT_NAMES[engram]
    write_json(PREDICTIONS_DIR / OUTPUT_FILES[engram],
               {"variant": name, "metrics": metrics,
                "references": references, "predictions": predictions})
    print(f"Combined {name}: CER={metrics['corpus_cer']:.6f}; F1={metrics['char_f1']:.6f}; "
          f"mean={metrics['latency_mean_seconds']:.3f}s/clip; "
          f"storage={metrics['model_tensor_storage_bytes'] / 2**30:.3f} GiB")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    parser.add_argument("--start", type=int, default=0, help="First offset in the fixed 300-sample set")
    parser.add_argument("--limit", type=int, default=300, help="Maximum samples in this chunk")
    parser.add_argument("--variant", choices=("baseline", "engram", "both"), default="both")
    parser.add_argument("--combine", action="store_true", help="Combine completed chunks into final results")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    load_cached_references()

    setup = Session(cfg, engram=False, verbose=False)
    all_records = setup.collect_records()
    environment = setup.environment
    setup.close()
    if [r["text"] for r in all_records] != load_cached_references():
        raise RuntimeError("The benchmark set does not match the frozen 300-clip cache")

    if args.combine:
        base = combine_variant(cfg, all_records, engram=False)
        engram = combine_variant(cfg, all_records, engram=True)
        write_json(SUMMARIES_DIR / "step750_llm_int8_memory_latency.json", {
            "dataset": {"samples": len(all_records), "source_skip": cfg.eval_source_skip},
            "gpu": environment["gpu"],
            "rocm_hip": environment["rocm_hip"],
            "torch": environment["torch"],
            "bitsandbytes": bnb.__version__,
            "checkpoint": str(cfg.checkpoint_paths()[STEP]),
            "quantization": {
                "method": "bitsandbytes LLM.int8 (Linear8bitLt)",
                "bits": 8,
                "threshold": LLM_INT8_THRESHOLD,
                "has_fp16_weights": False,
                "scope": ("Linear layers only; embeddings and other non-linear parameters "
                          "remain in BF16"),
                "gguf": False,
            },
            "latency_method": ("end-to-end audio preprocessing + greedy ASR generation; "
                               "one warm-up per chunk excluded"),
            "storage_method": ("unique live model parameter/buffer storage plus Linear8bitLt "
                               "CB/SCB; peak VRAM is PyTorch allocator stats"),
            "variants": {"baseline_llm_int8": base, "engram_step750_llm_int8": engram},
        })
        print("Saved:", SUMMARIES_DIR / "step750_llm_int8_memory_latency.json", flush=True)
        return

    if args.start < 0 or args.limit < 1 or args.start >= len(all_records):
        raise ValueError("Use a valid --start and a positive --limit")
    records = all_records[args.start:args.start + args.limit]
    print(f"Benchmarking clips {args.start}–{args.start + len(records) - 1} of {len(all_records)} "
          f"on {environment['gpu']}; ROCm {environment['rocm_hip']}; "
          f"bitsandbytes {bnb.__version__}; no GGUF", flush=True)
    if args.variant in ("baseline", "both"):
        evaluate(cfg, records, engram=False, start_index=args.start)
    if args.variant in ("engram", "both"):
        evaluate(cfg, records, engram=True, start_index=args.start)


if __name__ == "__main__":
    main()

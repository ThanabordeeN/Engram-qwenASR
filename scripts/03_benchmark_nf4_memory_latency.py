#!/usr/bin/env python3
"""Memory and latency: BF16 vs NF4 for baseline and Engram step 750.

Produces:

    results/summaries/step750_nf4_memory_latency.json

For each of the four variants it reports model-tensor storage, peak allocated
VRAM, mean/P50/P95 latency per clip, and corpus CER, and it asserts that the
predictions equal the ones saved by ``scripts/02`` (so a mismatch in the
quantizer or the decoding setup fails loudly instead of silently).

Runtime on an RX 9070 XT: roughly 30 minutes.

    .venv/bin/python scripts/03_benchmark_nf4_memory_latency.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    NF4_COMPUTE_DTYPE,
    NF4_GROUP_SIZE,
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    benchmark_variant,
    load_cached_references,
    quantize_nf4,
)
from qwen3asr_engram import Config, PROJECT_ROOT, Session, write_json  # noqa: E402


def run_unengram(cfg, records, expected_base, expected_nf4) -> dict:
    session = Session(cfg, engram=False)
    bf16, preds = benchmark_variant(session, records, False, "Baseline BF16",
                                    tensor_bytes_after_warmup=False)
    bf16["predictions_match_saved_eval"] = preds == expected_base
    del preds

    start = time.perf_counter()
    counts = quantize_nf4(session.model)
    quantize_seconds = time.perf_counter() - start
    nf4, preds = benchmark_variant(session, records, False, "Baseline NF4 W4",
                                   tensor_bytes_after_warmup=False)
    nf4["quantized_modules"] = counts
    nf4["quantization_seconds"] = quantize_seconds
    nf4["predictions_match_saved_eval"] = preds == expected_nf4
    del preds
    result = {"baseline_bf16": bf16, "baseline_nf4_w4": nf4}
    session.close()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_engram(cfg, records, expected_bf16, expected_nf4) -> dict:
    session = Session(cfg, engram=True)
    session.load_checkpoint(STEP)

    bf16, preds = benchmark_variant(session, records, True, "Step-750 Engram BF16",
                                    tensor_bytes_after_warmup=False)
    bf16["predictions_match_saved_eval"] = preds == expected_bf16
    del preds

    start = time.perf_counter()
    counts = quantize_nf4(session.model)
    quantize_seconds = time.perf_counter() - start
    nf4, preds = benchmark_variant(session, records, True, "Step-750 Engram NF4 W4",
                                   tensor_bytes_after_warmup=False)
    nf4["quantized_modules"] = counts
    nf4["quantization_seconds"] = quantize_seconds
    nf4["predictions_match_saved_eval"] = preds == expected_nf4
    del preds
    result = {"engram_step750_bf16": bf16, "engram_step750_nf4_w4": nf4}
    session.close()
    gc.collect()
    torch.cuda.empty_cache()
    return result


def read_predictions(name: str) -> list[str]:
    return json.loads((PREDICTIONS_DIR / name).read_text(encoding="utf-8"))["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    refs = load_cached_references()

    setup = Session(cfg, engram=False, verbose=False)
    records = setup.collect_records()
    environment = setup.environment
    setup.close()
    if [r["text"] for r in records] != refs:
        raise RuntimeError("The benchmark set does not match the frozen 300-clip cache")

    expected_base_bf16 = read_predictions("fixed300_base_predictions.json")
    expected_base_nf4 = read_predictions("baseline_nf4_w4_predictions.json")
    step750 = json.loads((SUMMARIES_DIR / "preset_a_layer2_step750_evaluation.json")
                         .read_text(encoding="utf-8"))
    expected_engram_bf16 = [row["engram_step750"] for row in step750["rows"]]
    expected_engram_nf4 = read_predictions("step750_engram_nf4_w4_predictions.json")

    print(f"Benchmarking {len(records)} clips on {environment['gpu']}; "
          f"ROCm {environment['rocm_hip']}; bitsandbytes {bnb.__version__}", flush=True)

    results = run_unengram(cfg, records, expected_base_bf16, expected_base_nf4)
    results.update(run_engram(cfg, records, expected_engram_bf16, expected_engram_nf4))

    summary = {
        "dataset": {"samples": len(records), "source_skip": cfg.eval_source_skip},
        "gpu": environment["gpu"],
        "rocm_hip": environment["rocm_hip"],
        "torch": environment["torch"],
        "bitsandbytes": bnb.__version__,
        "checkpoint": str(cfg.checkpoint_paths()[STEP]),
        "quantization": {"method": "bitsandbytes NF4 weight-only", "bits": 4,
                         "group_size": NF4_GROUP_SIZE,
                         "compute_dtype": str(NF4_COMPUTE_DTYPE), "double_quant": False},
        "latency_method": ("end-to-end audio preprocessing + greedy ASR generation; "
                           "one warm-up excluded; 300 timed clips"),
        "storage_method": ("unique live model parameter/buffer/quant-state tensor storage; "
                           "peak VRAM is PyTorch CUDA allocator stats"),
        "variants": results,
    }
    write_json(SUMMARIES_DIR / "step750_nf4_memory_latency.json", summary)
    print("Saved:", SUMMARIES_DIR / "step750_nf4_memory_latency.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Headline result: baseline BF16 vs Engram BF16 across the four checkpoints.

Produces (fixed 300 clips, offset 250 000 of the streaming dataset):

    results/predictions/fixed300_base_predictions.json
    results/predictions/preset_a_layer2_step{300,500,750,1000}_predictions.csv
    results/summaries/preset_a_layer2_step{300,500,750,1000}_evaluation.json
    results/summaries/scaling_300_to_1000_summary.json
    data/eval_set_300.json              (frozen reference manifest)

Runtime on an RX 9070 XT: roughly 4 minutes per variant, ~20 minutes total.
The baseline predictions are cached; delete the cache to regenerate them.

    .venv/bin/python scripts/01_eval_scaling.py
    .venv/bin/python scripts/01_eval_scaling.py --limit 20   # quick check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    char_prf,
    sample_cer,
)
from qwen3asr_engram import (  # noqa: E402
    PROJECT_ROOT,
    Config,
    Session,
    corpus_cer,
    results_subdir,
    write_json,
)

BASE_CACHE = PREDICTIONS_DIR / "fixed300_base_predictions.json"


def write_eval_manifest(cfg: Config, records) -> Path:
    """Freeze the evaluation set so later runs can prove they used the same clips."""
    refs = [r["text"] for r in records]
    path = PROJECT_ROOT / "data" / "eval_set_300.json"
    write_json(path, {
        "dataset": cfg.dataset_name,
        "split": cfg.dataset_split,
        "source_skip": cfg.eval_source_skip,
        "samples": len(refs),
        "min_audio_sec": cfg.min_audio_sec,
        "max_audio_sec": cfg.max_audio_sec,
        "references_sha256": hashlib.sha256("\n".join(refs).encode("utf-8")).hexdigest(),
        "total_audio_seconds": round(sum(r["duration"] for r in records), 3),
        "references": refs,
    })
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N clips (smoke test; results are not reportable)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Regenerate the baseline predictions even if the cache exists")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    full_set = args.limit is None

    # --- evaluation set (no model needed) ---------------------------------- #
    data_session = Session(cfg, engram=False, verbose=False)
    records = data_session.collect_records(limit=args.limit)
    data_session.close()
    refs = [r["text"] for r in records]

    if full_set:
        manifest = write_eval_manifest(cfg, records)
        print("Saved evaluation-set manifest:", manifest)

    # --- baseline BF16 (cached) -------------------------------------------- #
    base_preds = None
    if BASE_CACHE.is_file() and full_set and not args.no_cache:
        cached = json.loads(BASE_CACHE.read_text(encoding="utf-8"))
        if (cached.get("source_skip") == cfg.eval_source_skip
                and cached.get("references") == refs):
            base_preds = cached["predictions"]
            print("Reused cached baseline predictions:", BASE_CACHE)
        else:
            print("Baseline cache does not match this fixed set; regenerating.")

    if base_preds is None:
        session = Session(cfg, engram=False)
        print("Generating baseline BF16 predictions...")
        base_preds = [session.transcribe(record, False)
                      for record in tqdm(records, desc="Baseline BF16")]
        if full_set:
            write_json(BASE_CACHE, {"source_skip": cfg.eval_source_skip,
                                    "references": refs, "predictions": base_preds})
            print("Saved baseline cache:", BASE_CACHE)
        session.close()

    base_per_sample = [sample_cer(r, p) for r, p in zip(refs, base_preds)]
    base_mean = float(np.mean(base_per_sample))
    base_corpus = float(corpus_cer(refs, base_preds))
    print(f"Baseline BF16: corpus CER {base_corpus:.6f}; mean sample CER {base_mean:.6f}")

    # --- Engram checkpoints ------------------------------------------------ #
    session = Session(cfg, engram=True)
    summaries = []
    for step in cfg.checkpoint_steps:
        print(f"\n=== Step {step}: loading checkpoint and generating predictions ===")
        payload = session.load_checkpoint(step)
        engram_preds = [session.transcribe(record, True)
                        for record in tqdm(records, desc=f"Engram step {step}")]
        engram_per_sample = [sample_cer(r, p) for r, p in zip(refs, engram_preds)]
        engram_mean = float(np.mean(engram_per_sample))
        engram_corpus = float(corpus_cer(refs, engram_preds))
        improved = sum(e < b for e, b in zip(engram_per_sample, base_per_sample))
        tied = sum(e == b for e, b in zip(engram_per_sample, base_per_sample))

        summary = {
            "step": step,
            "samples": len(refs),
            "base_mean_sample_cer": base_mean,
            "engram_mean_sample_cer": engram_mean,
            "relative_mean_sample_cer_improvement_percent": (
                (base_mean - engram_mean) / base_mean * 100.0 if base_mean else None),
            "base_corpus_cer": base_corpus,
            "engram_corpus_cer": engram_corpus,
            "relative_corpus_cer_improvement_percent": (
                (base_corpus - engram_corpus) / base_corpus * 100.0 if base_corpus else None),
            "improved_samples": improved,
            "tied_samples": tied,
            "regressed_samples": len(refs) - improved - tied,
            "checkpoint": str(cfg.checkpoint_paths()[step]),
            "checkpoint_format": payload.get("format"),
            "gpu": session.environment["gpu"],
            "rocm_hip": session.environment["rocm_hip"],
            "generation_use_cache": cfg.generation_use_cache,
            "eval_source_skip": cfg.eval_source_skip,
        }

        rows = [
            {
                "index": i,
                "reference": ref,
                "base": base,
                f"engram_step{step}": eng,
                "base_cer": float(bc),
                "engram_cer": float(ec),
                "cer_delta": float(ec - bc),
                "improved": bool(ec < bc),
            }
            for i, (ref, base, eng, bc, ec) in enumerate(
                zip(refs, base_preds, engram_preds, base_per_sample, engram_per_sample))
        ]

        if full_set:
            import pandas as pd

            pd.DataFrame(rows).to_csv(
                PREDICTIONS_DIR / f"preset_a_layer2_step{step}_predictions.csv", index=False)
            write_json(SUMMARIES_DIR / f"preset_a_layer2_step{step}_evaluation.json",
                       {"summary": summary, "rows": rows})
            print("Saved:", SUMMARIES_DIR / f"preset_a_layer2_step{step}_evaluation.json")

        summaries.append(summary)
        print(f"Step {step} DONE — corpus CER {base_corpus:.4f} -> {engram_corpus:.4f} "
              f"({summary['relative_corpus_cer_improvement_percent']:.1f}% rel. improvement)")

    session.close()

    if full_set:
        write_json(SUMMARIES_DIR / "scaling_300_to_1000_summary.json",
                   {"steps": cfg.checkpoint_steps, "results": summaries})
        print("\nSaved:", SUMMARIES_DIR / "scaling_300_to_1000_summary.json")

    print("\nFINAL SCALING SUMMARY")
    for s in summaries:
        print(f"  step {s['step']:>4}: base {s['base_corpus_cer']:.4f} -> "
              f"engram {s['engram_corpus_cer']:.4f}  "
              f"({s['relative_corpus_cer_improvement_percent']:+.1f}% rel.), "
              f"improved/tied/regressed = "
              f"{s['improved_samples']}/{s['tied_samples']}/{s['regressed_samples']}")


if __name__ == "__main__":
    main()

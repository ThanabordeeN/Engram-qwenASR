#!/usr/bin/env python3
"""Reproduction test: re-run a few clips and compare against the saved artifacts.

This is the check that the code in this repository still reproduces the numbers
in the report. It re-derives every variant on the first N clips of the fixed set
and compares:

  * the decoded text against the saved per-sample predictions, and
  * model-tensor storage against the saved memory/latency summaries.

It needs a GPU and loads the model several times, so it takes a few minutes
rather than seconds. ``scripts/00_check_setup.py`` is the cheap check.

    .venv/bin/python scripts/07_verify_reproduction.py --samples 4
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    model_tensor_bytes,
    quantize_llm_int8,
    quantize_nf4,
)
from qwen3asr_engram import Config, PROJECT_ROOT, Session  # noqa: E402

logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(label)


def read_predictions(name: str) -> dict:
    return json.loads((PREDICTIONS_DIR / name).read_text(encoding="utf-8"))


def first_n(values, n: int) -> list:
    return list(values)[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    parser.add_argument("--samples", type=int, default=4, help="Clips to re-run per variant")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    n = args.samples

    base_cache = read_predictions("fixed300_base_predictions.json")
    nf4_cache = read_predictions("step750_engram_nf4_w4_predictions.json")
    int8_cache = read_predictions("step750_engram_llm_int8_w8_predictions.json")
    with (PREDICTIONS_DIR / "preset_a_layer2_step750_predictions.csv").open(encoding="utf-8-sig") as handle:
        bf16_engram_saved = [row["engram_step750"] for row in csv.DictReader(handle)]
    nf4_mem = json.loads((SUMMARIES_DIR / "step750_nf4_memory_latency.json").read_text(encoding="utf-8"))["variants"]
    int8_mem = json.loads((SUMMARIES_DIR / "step750_llm_int8_memory_latency.json").read_text(encoding="utf-8"))["variants"]

    session = Session(cfg, engram=True)
    records = session.collect_records()
    check("evaluation set matches the frozen reference list",
          [r["text"] for r in records] == base_cache["references"])

    # --- BF16 ------------------------------------------------------------------ #
    baseline = [session.transcribe(records[i], False) for i in range(n)]
    check("baseline BF16 text", baseline == first_n(base_cache["predictions"], n),
          repr(baseline[:1]))

    session.load_checkpoint(STEP)
    engram_bf16 = [session.transcribe(records[i], True) for i in range(n)]
    check("Engram BF16 step 750 text", engram_bf16 == first_n(bf16_engram_saved, n),
          repr(engram_bf16[:1]))

    # --- NF4 ------------------------------------------------------------------- #
    quantize_nf4(session.model)
    # NF4 storage is measured before the warm-up clip, as in scripts/03.
    nf4_bytes = model_tensor_bytes(session.model)
    saved_nf4_bytes = nf4_mem["engram_step750_nf4_w4"]["model_tensor_storage_bytes"]
    check("Engram NF4 tensor storage", nf4_bytes == saved_nf4_bytes,
          f"{nf4_bytes} B vs saved {saved_nf4_bytes} B")
    engram_nf4 = [session.transcribe(records[i], True) for i in range(n)]
    check("Engram NF4 text", engram_nf4 == first_n(nf4_cache["predictions"], n),
          repr(engram_nf4[:1]))
    session.close()
    torch.cuda.empty_cache()

    # --- LLM.int8 -------------------------------------------------------------- #
    session = Session(cfg, engram=True)
    session.load_checkpoint(STEP)
    quantize_llm_int8(session.model)
    session.transcribe(records[0], True)  # the first forward materializes CB/SCB
    int8_bytes = model_tensor_bytes(session.model)
    saved_int8_bytes = int8_mem["engram_step750_llm_int8"]["model_tensor_storage_bytes"]
    check("Engram LLM.int8 tensor storage", int8_bytes == saved_int8_bytes,
          f"{int8_bytes} B vs saved {saved_int8_bytes} B")
    engram_int8 = [session.transcribe(records[i], True) for i in range(n)]
    check("Engram LLM.int8 text", engram_int8 == first_n(int8_cache["predictions"], n),
          repr(engram_int8[:1]))
    session.close()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        raise SystemExit(1)
    print(f"All checks passed: {n} clips per variant reproduce the saved artifacts.")


if __name__ == "__main__":
    main()

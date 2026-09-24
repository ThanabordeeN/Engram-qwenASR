#!/usr/bin/env python3
"""Verify the environment, checkpoints, frozen evaluation set, and results inventory.

Run this first, and after copying the project to a new machine. It does not load
a model and takes a few seconds.

    .venv/bin/python scripts/00_check_setup.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import PREDICTIONS_DIR, SUMMARIES_DIR  # noqa: E402
from qwen3asr_engram import Config, PROJECT_ROOT  # noqa: E402

EXPECTED_STEPS = [300, 500, 750, 1000]
ENGRAM_HUB_REPO = "https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram"
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK  ' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


print(f"Project root: {PROJECT_ROOT}\n")

# --- config ---------------------------------------------------------------- #
cfg = Config.load(PROJECT_ROOT / "configs" / "eval_300.json")
check("configs/eval_300.json parses", True,
      f"{cfg.eval_samples} clips from offset {cfg.eval_source_skip}")

# --- checkpoints ----------------------------------------------------------- #
# The four .pt files are deliberately NOT redistributed: they live on the
# Hugging Face Hub, and the Zenodo software archive ships without them. If they
# are present, verify them against SHA256SUMS; if they are absent, say where to
# fetch them rather than failing.
sums_file = PROJECT_ROOT / "checkpoints" / "SHA256SUMS"
recorded = {}
if sums_file.is_file():
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            recorded[name.strip().lstrip("*")] = digest
missing_checkpoints = []
for step, path in cfg.checkpoint_paths().items():
    if not path.is_file():
        missing_checkpoints.append(f"checkpoints/{path.name}")
        continue
    check(f"checkpoint step {step} present", True, path.name)
    if path.name in recorded:
        ok = sha256(path) == recorded[path.name]
        check(f"checkpoint step {step} sha256", ok,
              "matches SHA256SUMS" if ok else "MISMATCH")
if missing_checkpoints:
    check("checkpoints/SHA256SUMS present", sums_file.is_file(),
          "manifest of the weights hosted elsewhere")

# --- frozen evaluation set ------------------------------------------------- #
manifest_path = PROJECT_ROOT / "data" / "eval_set_300.json"
cache_path = PREDICTIONS_DIR / "fixed300_base_predictions.json"
if manifest_path.is_file() and cache_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    refs = manifest["references"]
    digest = hashlib.sha256("\n".join(refs).encode("utf-8")).hexdigest()
    check("evaluation set size", len(refs) == cfg.eval_samples, f"{len(refs)} references")
    check("evaluation set sha256", digest == manifest["references_sha256"], digest[:16])
    check("cached base predictions match the manifest",
          cache["references"] == refs and cache.get("source_skip") == cfg.eval_source_skip)
else:
    check("frozen evaluation set manifest present", False,
          "run scripts/01_eval_scaling.py to create data/eval_set_300.json")

# --- results inventory ----------------------------------------------------- #
expected = [
    "summaries/scaling_300_to_1000_summary.json",
    "summaries/step750_nf4_memory_latency.json",
    "summaries/step750_nf4_quantization_comparison.json",
    "summaries/step750_llm_int8_memory_latency.json",
    "summaries/nf4_exported_checkpoint_eval_summary.json",
    "summaries/project_f1_scores.json",
    "tables/project_f1_results_table.csv",
    "figures/figure_1_cer_vs_tensor_storage.pdf",
    "figures/figure_2_scaling_cer_f1.pdf",
]
expected += [f"summaries/preset_a_layer2_step{step}_evaluation.json" for step in EXPECTED_STEPS]
expected += [f"predictions/preset_a_layer2_step{step}_predictions.csv" for step in EXPECTED_STEPS]
for rel in expected:
    check(f"result {rel}", (PROJECT_ROOT / "results" / rel).is_file())

# --- optional artifacts ----------------------------------------------------- #
# Not required to reproduce the numbers, so a miss is reported but does not fail.
# The reports live in the companion Zenodo record; the NF4 .pt bundles are
# regenerable with scripts/04 and are excluded from the software archive.
print("\nOptional (companion record or regenerable):")
for name in missing_checkpoints:
    print(f"  {name}: absent  — fetch from {ENGRAM_HUB_REPO}")
for lang in ("en", "th"):
    present = sum((PROJECT_ROOT / f"reports/{lang}" / f"project_technical_report_{lang}.{ext}").is_file()
                  for ext in ("md", "tex", "pdf"))
    print(f"  reports/{lang}/: {present}/3 files"
          + ("" if present == 3 else "  — see the companion technical-reports record"))
for name in ("baseline_nf4_w4.pt", "step750_engram_nf4_w4.pt"):
    present = (PROJECT_ROOT / "exports" / "nf4" / name).is_file()
    print(f"  exports/nf4/{name}: {'present' if present else 'absent'}"
          + ("" if present else "  — regenerate with scripts/04_export_eval_nf4.py"))

# --- runtime (informational) ---------------------------------------------- #
print("\nRuntime (informational):")
try:
    import torch

    print(f"  torch {torch.__version__}, HIP {torch.version.hip}, "
          f"GPU available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  NOTE: no AMD GPU visible; the benchmark scripts cannot run here.")
except Exception as exc:  # pragma: no cover - environment dependent
    print(f"  torch import failed: {exc!r}")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    raise SystemExit(1)
print("All checks passed.")

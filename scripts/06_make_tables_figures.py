#!/usr/bin/env python3
"""Report tables and figures: character-level micro-F1, CSV table, both figures.

CPU only; no model load. Every number is recomputed from the saved per-sample
predictions and cross-checked against the corpus CER recorded by the run that
produced them, so a stale or mismatched artifact fails loudly.

Produces:

    results/summaries/project_f1_scores.json
    results/tables/project_f1_results_table.csv
    results/figures/figure_1_cer_vs_tensor_storage.{pdf,png}
    results/figures/figure_2_scaling_cer_f1.{pdf,png}

    .venv/bin/python scripts/06_make_tables_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jiwer import cer, process_characters
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import FIGURES_DIR, PREDICTIONS_DIR, SUMMARIES_DIR, TABLES_DIR  # noqa: E402
from qwen3asr_engram import thai_cer_norm  # noqa: E402

STEPS = [300, 500, 750, 1000]


def score(refs: list[str], preds: list[str], saved_cer: float) -> dict:
    refs_norm = [thai_cer_norm(s) for s in refs]
    preds_norm = [thai_cer_norm(s) for s in preds]
    aligned = process_characters(refs_norm, preds_norm)
    hits = int(aligned.hits)
    substitutions = int(aligned.substitutions)
    deletions = int(aligned.deletions)
    insertions = int(aligned.insertions)
    false_positive = substitutions + insertions
    false_negative = substitutions + deletions
    precision = hits / (hits + false_positive) if hits + false_positive else 0.0
    recall = hits / (hits + false_negative) if hits + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reproduced_cer = float(cer("".join(refs_norm), "".join(preds_norm)))
    if abs(reproduced_cer - saved_cer) > 1e-10:
        raise ValueError(f"CER mismatch: recomputed={reproduced_cer}, saved={saved_cer}")
    return {
        "samples": len(refs),
        "corpus_cer": saved_cer,
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def read_summary(name: str) -> dict:
    return json.loads((SUMMARIES_DIR / name).read_text(encoding="utf-8"))


def read_predictions(name: str) -> dict:
    return json.loads((PREDICTIONS_DIR / name).read_text(encoding="utf-8"))


# --- Engram BF16 scaling arms -------------------------------------------------- #
scaling = []
refs = None
base_preds = None
for step in STEPS:
    path = f"preset_a_layer2_step{step}_evaluation.json"
    doc = read_summary(path)
    rows = doc["rows"]
    this_refs = [r["reference"] for r in rows]
    this_base = [r["base"] for r in rows]
    if refs is None:
        refs, base_preds = this_refs, this_base
    elif refs != this_refs or base_preds != this_base:
        raise ValueError(f"Evaluation set/base predictions differ in {path}")
    preds = [row[f"engram_step{step}"] for row in rows]
    meta = doc["summary"]
    scaling.append({
        "variant": f"Engram BF16 step {step}",
        "family": "Engram",
        "precision": "BF16",
        "step": step,
        **score(refs, preds, float(meta["engram_corpus_cer"])),
        "relative_cer_reduction_vs_baseline_pct": float(meta["relative_corpus_cer_improvement_percent"]),
        "improved_samples": int(meta["improved_samples"]),
        "tied_samples": int(meta["tied_samples"]),
        "regressed_samples": int(meta["regressed_samples"]),
    })

base_summary = read_summary("preset_a_layer2_step300_evaluation.json")["summary"]
base_bf16 = {
    "variant": "Baseline BF16",
    "family": "Baseline",
    "precision": "BF16",
    "step": None,
    **score(refs, base_preds, float(base_summary["base_corpus_cer"])),
    "relative_cer_reduction_vs_baseline_pct": 0.0,
    "improved_samples": None,
    "tied_samples": None,
    "regressed_samples": None,
}

# --- quantized arms ------------------------------------------------------------ #
quant_rows = []
for filename, variant, family, precision, step in [
    ("baseline_nf4_w4_predictions.json", "Baseline NF4", "Baseline", "NF4", None),
    ("step750_engram_nf4_w4_predictions.json", "Engram NF4 step 750", "Engram", "NF4", 750),
    ("baseline_llm_int8_w8_predictions.json", "Baseline LLM.int8", "Baseline", "LLM.int8", None),
    ("step750_engram_llm_int8_w8_predictions.json", "Engram LLM.int8 step 750", "Engram", "LLM.int8", 750),
]:
    doc = read_predictions(filename)
    if doc["references"] != refs:
        raise ValueError(f"Fixed references differ in {filename}")
    quant_rows.append({
        "variant": variant,
        "family": family,
        "precision": precision,
        "step": step,
        **score(refs, doc["predictions"], float(doc["metrics"]["corpus_cer"])),
        "relative_cer_reduction_vs_baseline_pct": None,
        "improved_samples": None,
        "tied_samples": None,
        "regressed_samples": None,
    })

results = [base_bf16, *scaling, *quant_rows]

# --- attach memory/latency ------------------------------------------------------ #
mem = read_summary("step750_nf4_memory_latency.json")["variants"]
int8_mem = read_summary("step750_llm_int8_memory_latency.json")["variants"]
benchmark_entries = {
    "Baseline BF16": mem["baseline_bf16"],
    "Baseline NF4": mem["baseline_nf4_w4"],
    "Engram BF16 step 750": mem["engram_step750_bf16"],
    "Engram NF4 step 750": mem["engram_step750_nf4_w4"],
    "Baseline LLM.int8": int8_mem["baseline_llm_int8"],
    "Engram LLM.int8 step 750": int8_mem["engram_step750_llm_int8"],
}
for row in results:
    entry = benchmark_entries.get(row["variant"])
    if entry:
        row["tensor_storage_gib"] = entry["model_tensor_storage_bytes"] / 2**30
        row["peak_allocated_vram_gib"] = entry["peak_allocated_bytes"] / 2**30
        row["mean_latency_seconds"] = entry["latency_mean_seconds"]
        row["p95_latency_seconds"] = entry["latency_p95_seconds"]
    else:
        row["tensor_storage_gib"] = None
        row["peak_allocated_vram_gib"] = None
        row["mean_latency_seconds"] = None
        row["p95_latency_seconds"] = None

f1_definition = (
    "Character-level micro precision/recall/F1 after the same Thai transcript normalization "
    "and whitespace removal used by the CER pipeline. JiWER character alignment counts exact "
    "matches as TP, substitutions as one FP plus one FN, insertions as FP, and deletions as FN; "
    "counts are pooled across the 300 utterances."
)
(SUMMARIES_DIR / "project_f1_scores.json").write_text(
    json.dumps({
        "dataset": {"samples": len(refs), "source_offset": 250000},
        "f1_definition": f1_definition,
        "results": results,
    }, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

columns = [
    "variant", "family", "precision", "step", "samples", "corpus_cer",
    "char_precision", "char_recall", "char_f1", "hits", "substitutions",
    "deletions", "insertions", "relative_cer_reduction_vs_baseline_pct",
    "improved_samples", "tied_samples", "regressed_samples", "tensor_storage_gib",
    "peak_allocated_vram_gib", "mean_latency_seconds", "p95_latency_seconds",
]
with (TABLES_DIR / "project_f1_results_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# --- figures -------------------------------------------------------------------- #
# Academic-style typography and colorblind-friendly colors.
plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
blue, vermillion, gray = "#0072B2", "#D55E00", "#666666"

# Figure 1: the central quality-storage trade-off narrative.
fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
plot_entries = {
    "baseline_bf16": benchmark_entries["Baseline BF16"],
    "baseline_nf4_w4": benchmark_entries["Baseline NF4"],
    "engram_step750_bf16": benchmark_entries["Engram BF16 step 750"],
    "engram_step750_nf4_w4": benchmark_entries["Engram NF4 step 750"],
    "baseline_llm_int8": benchmark_entries["Baseline LLM.int8"],
    "engram_step750_llm_int8": benchmark_entries["Engram LLM.int8 step 750"],
}
point_specs = [
    ("Baseline BF16", "baseline_bf16", blue, "s", (8, -18)),
    ("Baseline NF4", "baseline_nf4_w4", blue, "o", (8, 4)),
    ("Baseline LLM.int8", "baseline_llm_int8", blue, "^", (8, -12)),
    ("Engram BF16", "engram_step750_bf16", vermillion, "s", (8, 7)),
    ("Engram NF4", "engram_step750_nf4_w4", vermillion, "o", (8, 4)),
    ("Engram LLM.int8", "engram_step750_llm_int8", vermillion, "^", (8, -12)),
]
coords = {}
for label, key, color, marker, offset in point_specs:
    entry = plot_entries[key]
    x = entry["model_tensor_storage_bytes"] / 2**30
    y = entry["corpus_cer"] * 100
    coords[label] = (x, y)
    ax.scatter(x, y, s=78, marker=marker, color=color, edgecolor="black", linewidth=0.55, zorder=3)
    ax.annotate(f"{label}\n{y:.2f}% CER", (x, y), xytext=offset, textcoords="offset points", fontsize=8.5)
for first, second, color in [
    ("Baseline BF16", "Baseline NF4", blue),
    ("Baseline BF16", "Baseline LLM.int8", blue),
    ("Engram BF16", "Engram NF4", vermillion),
    ("Engram BF16", "Engram LLM.int8", vermillion),
]:
    ax.annotate("", xy=coords[second], xytext=coords[first],
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.15, "alpha": 0.8}, zorder=2)
ax.text(
    0.035, 0.045,
    "Engram NF4: +3.11 CER pp\nvs. original BF16 baseline",
    transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#bbbbbb", "alpha": 0.95},
)
ax.set_xlabel("Model tensor storage (GiB)")
ax.set_ylabel("Corpus CER (%) — lower is better")
ax.set_title(f"Quantization quality–storage trade-offs (N = {len(refs)})")
ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.8)
ax.set_xlim(0.43, 1.68)
ax.set_ylim(5, 33)
legend = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#777777", markeredgecolor="black", label="BF16"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#777777", markeredgecolor="black", label="NF4"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#777777", markeredgecolor="black", label="LLM.int8"),
    Line2D([0], [0], marker="o", color=blue, label="Baseline"),
    Line2D([0], [0], marker="o", color=vermillion, label="Engram, step 750"),
]
ax.legend(handles=legend, frameon=False, loc="upper right", ncol=2)
for suffix in ("pdf", "png"):
    fig.savefig(FIGURES_DIR / f"figure_1_cer_vs_tensor_storage.{suffix}", bbox_inches="tight")
plt.close(fig)

# Figure 2: scaling curves for the original BF16 Engram checkpoints.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), constrained_layout=True)
x = [row["step"] for row in scaling]
cer_values = [row["corpus_cer"] * 100 for row in scaling]
f1_values = [row["char_f1"] * 100 for row in scaling]
base_f1 = base_bf16["char_f1"] * 100
axes[0].plot(x, cer_values, color=vermillion, marker="o", linewidth=1.8, label="Engram BF16")
axes[0].axhline(base_bf16["corpus_cer"] * 100, color=blue, linestyle="--", linewidth=1.2, label="Baseline BF16")
axes[0].scatter([750], [cer_values[x.index(750)]], marker="*", s=150, color="#009E73",
                edgecolor="black", linewidth=0.4, zorder=4)
axes[0].set_ylabel("Corpus CER (%) — lower is better")
axes[0].set_title("(a) Corpus CER")
axes[0].legend(frameon=False)
axes[1].plot(x, f1_values, color=vermillion, marker="o", linewidth=1.8, label="Engram BF16")
axes[1].axhline(base_f1, color=blue, linestyle="--", linewidth=1.2, label="Baseline BF16")
axes[1].scatter([750], [f1_values[x.index(750)]], marker="*", s=150, color="#009E73",
                edgecolor="black", linewidth=0.4, zorder=4)
axes[1].set_ylabel("Character-level micro-F1 (%) — higher is better")
axes[1].set_title("(b) Character-level micro-F1")
axes[1].legend(frameon=False)
for axis in axes:
    axis.set_xlabel("Engram checkpoint step")
    axis.set_xticks(x)
    axis.grid(True, color="#dddddd", linewidth=0.65, alpha=0.8)
fig.suptitle(f"Engram checkpoint scaling on the fixed Thai ASR evaluation set (N = {len(refs)})", fontsize=11)
for suffix in ("pdf", "png"):
    fig.savefig(FIGURES_DIR / f"figure_2_scaling_cer_f1.{suffix}", bbox_inches="tight")
plt.close(fig)

print("F1 definition:", f1_definition)
for row in results:
    print(f"{row['variant']}: CER={row['corpus_cer']*100:.4f}%, P={row['char_precision']*100:.4f}%, "
          f"R={row['char_recall']*100:.4f}%, F1={row['char_f1']*100:.4f}%")
print("Saved: results/summaries/project_f1_scores.json, results/tables/project_f1_results_table.csv, "
      "results/figures/figure_1_cer_vs_tensor_storage.{pdf,png}, "
      "results/figures/figure_2_scaling_cer_f1.{pdf,png}")

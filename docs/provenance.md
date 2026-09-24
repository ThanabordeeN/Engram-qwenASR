# Provenance

Where the code came from, which script produced which file, and how to check that
nothing has been silently replaced.

## Code

`src/qwen3asr_engram.py` and `src/qwen3asr_bench.py` were extracted from the
original evaluation notebook. The code bodies were moved, not rewritten: the
notebook's side-effecting cells became `Session` methods, and the module-level
constants became `Config` fields. The Engram architecture, the injection wrapper,
the checkpoint validation, the Thai normalization, and the decoding call are
byte-for-byte the same statements.

| Source notebook | sha256 |
|---|---|
| `archive/notebooks/Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb` | `caf5e16c837aac5ab55895828e55405db47d76f21ba020c8b8fbd58fdb770258` |
| `archive/notebooks/Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval1000_Only.ipynb` | `36449cfe2a3944636a5ff21e176509bb158a231586ca8e106cc406bf7dc8b3f3` |

Cell → destination map:

| Notebook cell | Content | Destination |
|---|---|---|
| 4 | constants, seeds, checkpoint paths | `Config`, `seed_everything` |
| 6 | ROCm check, model load, outer-forward patch | `resolve_device`, `load_asr_model` |
| 8 | decoder probe | `probe_decoder`, `infer_hidden_size_for` |
| 9 | `ThaiNgramEngram` | `ThaiNgramEngram` (verbatim) |
| 10 | runtime, layer wrapper, pre-hook, `install_engram` | same names (verbatim) |
| 12 | checkpoint load + validation | `validate_checkpoint`, `Session.load_checkpoint` |
| 14 | Thai normalization, audio decode | `normalize_thai_transcript`, `decode_audio`, `prepare_record` |
| 15 | fixed-set collection | `collect_eval_records` |
| 17 | prefix text, transcribe, CER helpers | `build_prefix_text`, `Session.transcribe`, `corpus_cer` |
| 19 | scaling evaluation loop | `scripts/01_eval_scaling.py` |

The original flat scripts (`benchmark_step750_llm_int8.py`,
`benchmark_step750_memory_latency.py`, `evaluate_step750_nf4.py`,
`export_eval_step750_nf4_checkpoints.py`,
`generate_project_f1_and_figures.py`) are preserved in git history at the commit
*"Snapshot before repository restructure"*. Their replacements are the numbered
scripts; the quantizers, the memory accounting, and the timed loop moved into
`src/qwen3asr_bench.py` unchanged.

## Artifact → producing script

| Artifact | Script |
|---|---|
| `data/eval_set_300.json` | `01_eval_scaling.py` |
| `results/predictions/fixed300_base_predictions.json` | `01_eval_scaling.py` |
| `results/predictions/preset_a_layer2_step*_predictions.csv` | `01_eval_scaling.py` |
| `results/summaries/preset_a_layer2_step*_evaluation.json` | `01_eval_scaling.py` |
| `results/summaries/scaling_300_to_1000_summary.json` | `01_eval_scaling.py` |
| `results/predictions/{baseline,step750_engram}_nf4_w4_predictions.{json,csv}` | `02_eval_nf4_quality.py` |
| `results/summaries/step750_nf4_quantization_comparison.json` | `02_eval_nf4_quality.py` |
| `results/summaries/step750_nf4_memory_latency.json` | `03_benchmark_nf4_memory_latency.py` |
| `exports/nf4/*`, `results/predictions/*_nf4_checkpoint_eval.*` | `04_export_eval_nf4.py` |
| `results/summaries/nf4_exported_checkpoint_eval_summary.json` | `04_export_eval_nf4.py` |
| `results/parts/llm_int8/*`, `results/predictions/*_llm_int8_w8_predictions.json` | `05_benchmark_llm_int8.py` |
| `results/summaries/step750_llm_int8_memory_latency.json` | `05_benchmark_llm_int8.py --combine` |
| `results/summaries/project_f1_scores.json` | `06_make_tables_figures.py` |
| `results/tables/project_f1_results_table.csv` | `06_make_tables_figures.py` |
| `results/figures/figure_*.{pdf,png}` | `06_make_tables_figures.py` |
| `reports/{en,th}/*.{md,tex,pdf}` | written by hand, built with LuaLaTeX |

## Inputs (not produced by this repository)

### Engram checkpoints

Produced by the Engram training run, which is not included here. Verify with
`python scripts/00_check_setup.py`, which checks each file against
`checkpoints/SHA256SUMS`.

| File | Bytes | sha256 |
|---|---:|---|
| `engram_step_000300.pt` | 99 058 823 | `5e6ff7f953b6c072b524f86442664612752c87707d63caf31fc5f2dbc0217044` |
| `engram_step_000500.pt` | 99 060 615 | `688056992576ecdeaf588df1198e3c660c60cca26876132cc665a782be9b402c` |
| `engram_step_000750.pt` | 99 062 855 | `a4349657eedeb1664b3733d3fac80d5de4b2c27397201dc26882d91638d1dccb` |
| `engram_step_001000.pt` | 99 065 095 | `4dfad102c6d9f52371e97a9805caef8bfc2d3ab4b44c46b93eb22e619da12aea` |

Checkpoint configuration, identical across steps: Engram on decoder layer 2,
bigram and trigram tables (10 000 / 2 000 buckets), memory dimension 512, 16
heads, kernel 4 — 8 249 536 added parameters, nested under `engram_state["2"]`.

### Exported NF4 bundles

Produced by `scripts/04`, listed here because they are large and git-ignored.

| File | Bytes | sha256 |
|---|---:|---|
| `exports/nf4/baseline_nf4_w4.pt` | 598 445 625 | `4db401656c8fbbe1cf661a630bfc9a2810fa485c697e2671d5e3b932ea85d897` |
| `exports/nf4/step750_engram_nf4_w4.pt` | 603 676 797 | `60bae1f3522babe4937dc3c7dbc60df271638a6fc7fb5aac4ce9fd688c42d495` |

These two files are only reproducible against bitsandbytes `0.50.2`: the loader
refuses to load a bundle whose recorded bitsandbytes version differs from the
running one, because the packed layout of `Params4bit` is not a stable format.

## Superseded and excluded material

| Path | Status |
|---|---|
| `archive/notebooks/` | kept: the original interactive entry points; the code now lives in `src/` |
| `reports/` | kept in the working tree but **not in git**: distributed through the companion Zenodo record, where the write-ups get a DOI and a CC BY 4.0 licence of their own |
| `exports/nf4/*.pt` | git-ignored: regenerable with `scripts/04`, and loadable only against bitsandbytes 0.50.2 |
| `checkpoints/*.pt` | git-ignored: inputs from the Engram training run, tracked by `checkpoints/SHA256SUMS` |
| `hf/*.pt`, `hf/config.json` | git-ignored: generated by `scripts/08_publish_hf.py --stage` from `checkpoints/`. The delta is a strip of step 750 with the optimizer state removed, so it carries no information the checkpoints do not. |
| `reports/archive/nf4_memory_accuracy_technical_report.md` | lost during the git history rewrite that removed `reports/` from version control. It was an interim report superseded by the five-chapter reports, and nothing in the reported results depended on it. |

## How to check that nothing drifted

```bash
python scripts/00_check_setup.py                  # files, checksums, frozen set (seconds, no GPU)
python scripts/07_verify_reproduction.py --samples 4   # re-runs clips, compares to saved artifacts
python scripts/06_make_tables_figures.py          # recomputes every CER and F1 from saved predictions
python scripts/08_publish_hf.py --verify-samples 4     # Hub delta + standalone loader vs the in-repo path
```

`06` is the strongest cheap check: it recomputes all nine corpus CER values from
the per-sample predictions and raises `ValueError` if any disagrees with the
value recorded at run time. On an unchanged checkout it reproduces
`results/tables/project_f1_results_table.csv` and
`results/summaries/project_f1_scores.json` byte for byte.

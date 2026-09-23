# How to reproduce every reported number

Every number in `reports/en/` and `reports/th/` comes from a file under
`results/`, and every file under `results/` is written by one of the numbered
scripts below. Nothing is hand-edited.

## 0. Prerequisites

* AMD GPU with ROCm/HIP PyTorch. The reported run used a **Radeon RX 9070 XT**
  (`gfx1201`), ROCm/HIP **7.1.52802-9999**, PyTorch **2.9.1**, bitsandbytes
  **0.50.2**. See [`docs/environment.md`](docs/environment.md).
* Network access on the first run: the scripts stream the dataset from the
  Hugging Face Hub and download `Qwen/Qwen3-ASR-0.6B` on first use.
* The four Engram checkpoints in `checkpoints/`. They are inputs, not outputs of
  this repository — verify them with `scripts/00_check_setup.py`, which checks
  each file against `checkpoints/SHA256SUMS`.

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1 torchaudio==2.9.0
pip install -r requirements.txt
python scripts/00_check_setup.py
```

## 1. Run the pipeline in order

Each script is independent apart from the files it reads, listed per row. Timings
are for the RX 9070 XT and are the reason the order matters.

| # | Command | Reads | Writes | Time |
|---|---|---|---|---|
| 1 | `python scripts/01_eval_scaling.py` | checkpoints, dataset | `results/predictions/fixed300_base_predictions.json`, `results/predictions/preset_a_layer2_step{300,500,750,1000}_predictions.csv`, `results/summaries/preset_a_layer2_step{300,500,750,1000}_evaluation.json`, `results/summaries/scaling_300_to_1000_summary.json`, `data/eval_set_300.json` | ~20 min |
| 2 | `python scripts/02_eval_nf4_quality.py` | 1 | `results/predictions/{baseline_nf4_w4,step750_engram_nf4_w4}_predictions.{json,csv}`, `results/summaries/step750_nf4_quantization_comparison.json` | ~10 min |
| 3 | `python scripts/03_benchmark_nf4_memory_latency.py` | 1, 2 | `results/summaries/step750_nf4_memory_latency.json` | ~30 min |
| 4 | `python scripts/04_export_eval_nf4.py` | 1, 2 | `exports/nf4/*`, `results/predictions/*_nf4_checkpoint_eval.{json,csv}`, `results/summaries/nf4_exported_checkpoint_eval_summary.json` | ~40 min |
| 5 | `python scripts/05_benchmark_llm_int8.py --start 0 --limit 75` (repeat for 75, 150, 225) then `--combine` | 1 | `results/parts/llm_int8/*`, `results/predictions/*_llm_int8_w8_predictions.json`, `results/summaries/step750_llm_int8_memory_latency.json` | ~45 min |
| 6 | `python scripts/06_make_tables_figures.py` | 1–5 | `results/summaries/project_f1_scores.json`, `results/tables/project_f1_results_table.csv`, `results/figures/*` | ~10 s |
| 7 | `python scripts/07_verify_reproduction.py --samples 4` | 1–5 | nothing (test only) | ~5 min |

Step 5 can also be run in one pass with
`python scripts/05_benchmark_llm_int8.py --variant both`, which skips the chunk
files entirely. The chunked form exists because a single pass exceeds a
convenient interactive timeout; `--combine` refuses to combine unless the chunks
cover the frozen set exactly once, in order, with matching references.

Steps 1–5 are deterministic in the sense that greedy decoding
(`do_sample=False`, `num_beams=1`) with `use_cache=False` is used throughout, and
the evaluation set is fixed by offset and hash. Re-running on different hardware
or a different bitsandbytes build may still change results.

## 2. Rebuild the reports (optional)

The reports are **not** in this repository. They ship in the companion Zenodo
record as PDF plus LuaLaTeX sources in English and Thai — see
[`zenodo/METADATA.md`](zenodo/METADATA.md). Extract that archive and you get
`reports/{en,th}/` alongside `results/figures/`, which is the layout the `.tex`
files expect.

The PDFs are built with LuaLaTeX, needed for Thai script. Figures resolve through
`\graphicspath{{../../results/figures/}}`, so build from `reports/build/`:

```bash
cd reports/build
for lang in en th; do
  lualatex -interaction=nonstopmode ../$lang/project_technical_report_$lang.tex
  lualatex -interaction=nonstopmode ../$lang/project_technical_report_$lang.tex
  cp project_technical_report_$lang.pdf ../$lang/
done
```

Two passes are required so the table of contents and cross-references resolve.
The `.md` files are the readable source of the same text; the `.tex` files carry
the thesis-style chapter layout.

## 3. Which script produced which number

| Report item | Script | Artifact |
|---|---|---|
| Baseline / Engram BF16 CER, F1, scaling figure | 01, 06 | `preset_a_layer2_step*_evaluation.json`, `figure_2_scaling_cer_f1.pdf` |
| NF4 quality and the BF16→NF4 deltas | 02 | `step750_nf4_quantization_comparison.json` |
| Weights, peak VRAM, mean/P95 latency (BF16, NF4) | 03 | `step750_nf4_memory_latency.json` |
| NF4 on-disk size and reload validation | 04 | `nf4_exported_checkpoint_eval_summary.json` |
| LLM.int8 CER, F1, storage, latency | 05 | `step750_llm_int8_memory_latency.json` |
| Character precision/recall/F1 for every variant | 06 | `project_f1_scores.json`, `project_f1_results_table.csv` |
| The CER-vs-storage trade-off figure | 06 | `figure_1_cer_vs_tensor_storage.pdf` |

## 4. Definitions you need to keep straight

These are the definitions the scripts implement; changing them invalidates the
comparison against the reports.

* **Corpus CER** — character error rate over the concatenated normalized
  references and predictions, after `normalize_thai_transcript` (collapse
  whitespace, drop segmentation spaces between Thai characters) and whitespace
  removal, lowercased.
* **Character F1** — pooled character-level micro precision/recall/F1 from the
  same JiWER alignment. Exact matches are TP; a substitution is one FP plus one
  FN; an insertion is FP; a deletion is FN.
* **Model tensor storage** — bytes held by unique live model parameters and
  buffers, plus quantization side tensors (NF4 `quant_state`, LLM.int8
  `CB`/`SCB`), deduplicated by underlying storage. This is not device memory.
* **Peak allocated VRAM** — `torch.cuda.max_memory_allocated()` after a reset
  that happens *after* one excluded warm-up clip.
* **Latency** — wall-clock per clip including audio preprocessing and greedy
  generation, with `torch.cuda.synchronize()` after each clip. Model loading and
  quantization are excluded. P95 is the time within which 95 % of clips finished.

Two measurement points are deliberately preserved because the reported numbers
depend on them: NF4 tensor storage is measured **before** the warm-up clip
(603 226 560 B for Engram NF4), while LLM.int8 storage is measured **after** it,
because the first forward materializes the scale tensors (1 132 690 688 B for
Engram LLM.int8). `scripts/07` asserts both exact values.

## 5. Expected runtime and memory

* Total GPU time for the full pipeline: roughly **2.5 hours**.
* The 300 clips stay resident in host RAM as float32 audio (about 0.5 GiB), and
  the dataset stream is consumed once per script.
* Each benchmark script loads the model from scratch; peak allocated VRAM stays
  between 1.0 and 2.5 GiB depending on the variant.

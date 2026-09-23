# Thai ASR with an N-gram Engram on Qwen3-ASR-0.6B

Reproduction package for the study *"Adding a Thai N-gram Engram to Qwen3-ASR-0.6B:
quality, memory, and latency of BF16, NF4, and LLM.int8 inference on ROCm"*.

**Author:** Thanabodee Nammungkun (ธนบดี นามมุงคุณ)

The experiment adds a token-level bigram/trigram memory (**Engram**) to a frozen
Qwen3-ASR-0.6B and measures Thai transcription quality on a fixed 300-clip
held-out set, then compares three numerical precisions: **BF16**, **NF4**
(bitsandbytes, weight-only, group 32), and **LLM.int8** (bitsandbytes
`Linear8bitLt`).

---

## Headline results (300 fixed clips)

| Variant | CER (%) ↓ | Char F1 (%) ↑ | Weights (GiB) | Peak VRAM (GiB) | Mean s/clip |
|---|---:|---:|---:|---:|---:|
| Baseline, BF16 | 17.8051 | 86.3235 | 1.462 | 1.590 | 0.755 |
| **Engram step 750, BF16** | **9.0448** | **92.9302** | 1.493 | 1.623 | 0.834 |
| Baseline, NF4 | 28.7086 | 79.7589 | 0.557 | 1.024 | 1.333 |
| Engram step 750, NF4 | 20.9178 | 85.3693 | 0.562 | 1.034 | 1.556 |
| Baseline, LLM.int8 | 18.8901 | 85.3726 | 1.030 | 2.451 | 4.354 |
| Engram step 750, LLM.int8 | 10.8413 | 91.8857 | 1.055 | 2.479 | 4.055 |

Reading notes that matter:

* Engram step 750 improved CER by **8.7602 percentage points** over the baseline
  in BF16 on this slice. This is one fixed 300-clip sample, not the full dataset.
* Engram NF4 sits **3.1128 CER percentage points above** the baseline BF16.
  That is a difference in CER, **not a 3 % loss of accuracy**.
* In this ROCm/bitsandbytes setup, smaller weights did **not** mean faster
  inference: both NF4 and LLM.int8 were slower per clip than their BF16
  counterpart. Fewer bits do not guarantee higher speed.
* F1 here is **pooled character-level micro-F1** after Thai normalization and
  whitespace removal — not word-level F1, and not accuracy.

Full write-ups: [`reports/en/project_technical_report_en.pdf`](reports/en/) and
[`reports/th/project_technical_report_th.pdf`](reports/th/).

---

## Repository layout

```
├── README.md                  this file
├── REPRODUCE.md               step-by-step commands and the artifact → script map
├── requirements.txt           pinned Python dependencies
├── configs/eval_300.json      every setting of the reported run, in one place
├── data/eval_set_300.json     the frozen 300-clip evaluation set (references + hash)
│
├── src/                       the library (importable; no notebook required)
│   ├── qwen3asr_engram.py     model load, Engram architecture, dataset, inference, metrics
│   └── qwen3asr_bench.py      quantizers, memory accounting, the timed benchmark loop
│
├── scripts/                   one entry point per reported result
│   ├── 00_check_setup.py              verify environment, checkpoints, frozen set
│   ├── 01_eval_scaling.py             baseline + Engram BF16 across steps 300/500/750/1000
│   ├── 02_eval_nf4_quality.py         paired BF16 vs NF4 quality
│   ├── 03_benchmark_nf4_memory_latency.py   NF4/BF16 storage, VRAM, latency
│   ├── 04_export_eval_nf4.py          NF4 export + reload validation
│   ├── 05_benchmark_llm_int8.py       LLM.int8 quality, memory, latency (chunked)
│   ├── 06_make_tables_figures.py      F1 table and both figures (CPU only)
│   └── 07_verify_reproduction.py      re-run a few clips and compare to the saved artifacts
│
├── checkpoints/               Engram step checkpoints + SHA256SUMS
├── exports/nf4/               exported NF4 bundles + manifest + processor assets
├── results/
│   ├── predictions/           per-sample predictions (json + csv)
│   ├── summaries/             aggregate metrics and provenance json
│   ├── tables/                report tables (csv)
│   ├── figures/               report figures (pdf + png)
│   └── parts/                 resumable chunk records (git-ignored)
├── reports/
│   ├── en/, th/               Markdown, LaTeX, and built PDF per language
│   ├── build/                 LaTeX intermediate files (git-ignored)
│   └── archive/               superseded intermediate report
├── docs/
│   ├── environment.md         hardware and software record
│   └── provenance.md          notebook hash, artifact → script map, checksums
└── archive/notebooks/         the original evaluation notebooks (provenance only)
```

## Quick start

```bash
# 1. Environment (ROCm/HIP PyTorch first — see docs/environment.md)
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1 torchaudio==2.9.0
pip install -r requirements.txt

# 2. Check that the data, checkpoints, and results are all present
python scripts/00_check_setup.py

# 3. Verify that this code reproduces the saved artifacts (needs a GPU)
python scripts/07_verify_reproduction.py --samples 4

# 4. Rebuild the tables and figures from the saved predictions (CPU only)
python scripts/06_make_tables_figures.py
```

To re-run the whole study from scratch, follow [`REPRODUCE.md`](REPRODUCE.md).

## What is verified, and what is not

**Verified.** The evaluation set is frozen by hash (`data/eval_set_300.json`).
Every benchmark script re-checks its streamed clips against that list and fails
loudly if the upstream dataset changed. `scripts/07` re-runs each variant on a
few clips and confirms the decoded text and the tensor-storage accounting match
the saved artifacts byte for byte. `scripts/06` recomputes every corpus CER from
the saved predictions and raises if it disagrees with the value recorded at run
time.

**Not verified.** The results cover 300 clips from one offset of one dataset;
no confidence intervals were computed. The Engram training loop is not in this
repository; the checkpoints are inputs. Quantized GGUF inference is out of scope
for this deposit — the reported NF4 and LLM.int8 numbers come from
PyTorch/ROCm through bitsandbytes, not from llama.cpp.

## License and data

* **Code, configuration, and result artifacts:** MIT — see [`LICENSE`](LICENSE).
* **Technical reports** (`reports/`): CC BY 4.0 — see [`reports/LICENSE.md`](reports/LICENSE.md).

Data and models used, **not redistributed here**:

| Resource | Owner | Licence |
|---|---|---|
| [Porjai-Thai-voice-dataset-central](https://huggingface.co/datasets/CMKL/Porjai-Thai-voice-dataset-central) (Thai-dialect corpus) | CMKL University; Suwanbandit, Naowarat, Sangpetch, and Chuangsuwanich | CC BY-SA 4.0 |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Qwen Team | see model card |

Both are fetched at run time by the scripts. The corpus card records support from
the PMU-C grant (C10F630122), compute from the Apex cluster team, and donated
evaluation data from Wang via the Wang Data Market. The dataset paper is
Suwanbandit et al., *Thai Dialect Corpus and Transfer-based Curriculum Learning
Investigation for Dialect Automatic Speech Recognition*, INTERSPEECH 2023
([doi](https://doi.org/10.21437/Interspeech.2023-1828)).

> **Licence caveat for the Engram checkpoints.** The checkpoints in
> `checkpoints/` are trained on a CC BY-SA 4.0 corpus. Share-alike terms may
> attach to derived weights, so confirm the position with the dataset owners
> before publishing the checkpoints in a Zenodo deposit or elsewhere. The
> dataset itself is never redistributed here.

## Citation

`CITATION.cff` is machine-readable; GitHub and Zenodo both read it. The Zenodo
DOI is minted on publication and should then be added to `CITATION.cff` and to
[`zenodo/METADATA.md`](zenodo/METADATA.md).

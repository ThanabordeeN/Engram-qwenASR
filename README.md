# Thai ASR with an N-gram Engram on Qwen3-ASR-0.6B

Reproduction package for the study *"Adding a Thai N-gram Engram to Qwen3-ASR-0.6B:
quality, memory, and latency of BF16, NF4, and LLM.int8 inference on ROCm"*.

**Author:** Thanabodee Nammungkun (ธนบดี นามมุงคุณ)

**Repository:** <https://github.com/ThanabordeeN/Engram-qwenASR> ·
**Model:** <https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram>

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

Full write-ups are **not stored in this repository**. They ship in the companion
Zenodo record as PDF plus LuaLaTeX sources, in English and Thai — see
[`zenodo/METADATA.md`](zenodo/METADATA.md) for the record and its DOI.

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
│   ├── 07_verify_reproduction.py      re-run a few clips and compare to the saved artifacts
│   ├── 08_publish_hf.py               rebuild the Hub delta and verify it against the in-repo path
│   └── 09_deposit_zenodo.py           deposit both Zenodo records parsed out of zenodo/METADATA.md
│
├── hf/                        files uploaded to the Hub (README.md model card + standalone
│                              engram_loader.py are tracked; the delta and config.json are generated)
│
├── checkpoints/               Engram step checkpoints + SHA256SUMS
├── exports/nf4/               exported NF4 bundles + manifest + processor assets
├── results/
│   ├── predictions/           per-sample predictions (json + csv)
│   ├── summaries/             aggregate metrics and provenance json
│   ├── tables/                report tables (csv)
│   ├── figures/               report figures (pdf + png)
│   └── parts/                 resumable chunk records (git-ignored)
├── reports/                   NOT in git; ships in the companion Zenodo record
│   ├── en/, th/               Markdown, LaTeX, and built PDF per language
│   ├── build/                 LaTeX intermediate files (git-ignored)
│   └── archive/               superseded intermediate report
├── zenodo/
│   ├── METADATA.md            ready-to-paste deposit fields + archive sha256
│   └── build_archives.sh      builds both deposit archives
├── docs/
│   ├── environment.md         hardware and software record
│   └── provenance.md          notebook hash, artifact → script map, checksums
└── archive/notebooks/         the original evaluation notebooks (provenance only)
```

`reports/` is deliberately excluded from version control: the write-ups are
distributed through the Zenodo record, where they get a DOI and a licence of
their own (CC BY 4.0). The results they cite — every prediction, summary, table,
and figure — *are* in this repository, so the numbers stay auditable here.
`reports/` remains in the working tree; `zenodo/build_archives.sh` picks it up.

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
* **Technical reports** (in the companion Zenodo record, not here): CC BY 4.0.
* **The trained Engram delta:** CC BY-SA 4.0, published at
  [Thanabordee/Qwen3-ASR-0.6B-Thai-Engram](https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram).
  `scripts/08_publish_hf.py` rebuilds and re-verifies that artifact.

Data and models used. The **audio** is fetched at run time and never redistributed
here; the 300 reference transcripts are, because they are the frozen contract the
evaluation depends on:

| Resource | Owner | Licence | In this repo? |
|---|---|---|---|
| [Porjai-Thai-voice-dataset-central](https://huggingface.co/datasets/CMKL/Porjai-Thai-voice-dataset-central) (Thai-dialect corpus) | CMKL University; Suwanbandit, Naowarat, Sangpetch, and Chuangsuwanich | CC BY-SA 4.0 | audio: no — 300 reference transcripts: yes, in `data/eval_set_300.json` and the `results/predictions/` files |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | Qwen Team | see model card | no |

The 300 transcripts are redistributed under the corpus's CC BY-SA 4.0 terms, with
the attribution above. They are not optional: `scripts/00_check_setup.py` and every
benchmark script re-check the streamed clips against that list, so removing them
would remove the ability to tell whether the upstream corpus changed. The audio
itself stays with its provider.

Both are fetched at run time by the scripts. The corpus card records support from
the PMU-C grant (C10F630122), compute from the Apex cluster team, and donated
evaluation data from Wang via the Wang Data Market. The dataset paper is
Suwanbandit et al., *Thai Dialect Corpus and Transfer-based Curriculum Learning
Investigation for Dialect Automatic Speech Recognition*, INTERSPEECH 2023
([doi](https://doi.org/10.21437/Interspeech.2023-1828)).

> **Licence note for the Engram checkpoints.** The checkpoints in
> `checkpoints/` are trained on a CC BY-SA 4.0 corpus, so share-alike terms may
> attach to the derived weights. They are released under **CC BY-SA 4.0**, with
> the corpus credited; the delta alone is public on the
> [Hugging Face Hub](https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram).
> The audio of the corpus is never redistributed here; the 300 reference
transcripts are, under CC BY-SA 4.0 with the attribution above.

## Citation

`CITATION.cff` is machine-readable; GitHub and Zenodo both read it.

| | |
|---|---|
| Software (this repository) | [10.5281/zenodo.22933541](https://doi.org/10.5281/zenodo.22933541) |
| Preprint, English and Thai | [10.5281/zenodo.22933543](https://doi.org/10.5281/zenodo.22933543) |
| Trained Engram delta | [Thanabordee/Qwen3-ASR-0.6B-Thai-Engram](https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram) |

Cite the software DOI for the code and results, the preprint DOI for the write-up.

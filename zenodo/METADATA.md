# Zenodo deposit — Thai ASR with an N-gram Engram on Qwen3-ASR-0.6B

Prepared 2026-09-24. Two separate records, following the same split used for the
Audio-Laya deposit: software and technical reports.

| File | Record | Licence | Size |
|---|---|---|---|
| `EngramQwenASR-software-v1.0.0.zip` | Software | MIT | ~932 KB |
| `project_technical_report_en.pdf` | Publication / Preprint | CC BY 4.0 | 186 KB |
| `project_technical_report_th.pdf` | Publication / Preprint | CC BY 4.0 | 215 KB |

The preprint record carries the two PDFs **directly, not zipped**. Zenodo renders
an inline preview for a PDF — a `preview-iframe` plus a IIIF canvas — and shows
nothing at all for a zip. A preprint is read, not unpacked, so the LaTeX sources,
the Markdown, and the figures they include are not deposited; they stay in the
working tree under `reports/`.

The Engram weights are in neither archive. They are hosted on the Hugging Face
Hub at <https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram>, and
`checkpoints/SHA256SUMS` inside the software archive identifies them.

Rebuild the software archive with `./zenodo/build_archives.sh` (optionally pass a
version, e.g. `./zenodo/build_archives.sh 1.0.1`). The script still builds the
reports zip as a local copy of the LaTeX sources, but nothing uploads it.

`sha256` of the software archive:

```
f16c96029bd3c0ea614721049f7d3a7fc7cd5ae470119cdc47de2fd0de2805a1  EngramQwenASR-software-v1.0.0.zip
```

The software archive was verified by extraction into an empty directory and by
running `scripts/00_check_setup.py` there without the weights present.

`sha256` of the two deposited PDFs, so the record's bytes can be checked against
the working tree:

```
e7188d8a0ff43b3487f4129e3d706c22b003094a61cb6fa8bb4ae3cf7e6db4cb  project_technical_report_en.pdf
9217ca8862ad3216f4ab77554c2c2241930c09f720ce13c2baadd01ff360712f  project_technical_report_th.pdf
```

Note that `reports/` is **not** in the GitHub repository: it is git-ignored. It
used to be distributed through the reports record, but the preprint record now
holds only the two PDFs, so the working tree is the only copy of the LaTeX
sources and the Markdown. Keep it backed up. `zenodo/build_archives.sh` reads it
from the working tree, and `scripts/09_deposit_zenodo.py` reads the two PDFs from
it at deposit time.

---

## Read this before you publish

**1. Checkpoint licensing — decided, and the weights are not in this deposit.**
The Engram checkpoints are trained on
`CMKL/Porjai-Thai-voice-dataset-central`, which is licensed **CC BY-SA 4.0**, so
share-alike terms may attach to the derived weights.

The decision taken was to publish them under **CC BY-SA 4.0**, with the corpus
credited explicitly, and to host them **only** on the Hugging Face Hub at
<https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram>. They are
redundant here: the Hub copy is the same weights, it is already public, and
shipping them again would add about 348 MB to a deposit whose code and results
are 1.5 MB. `checkpoints/SHA256SUMS` travels in the archive so the weights stay
identified and verifiable without being duplicated.

This also removes the licence question from the deposit itself: the archive
contains no derived weights, only MIT code and the results.

**2. Publishing is permanent.** Zenodo mints a DOI on publish; a record cannot be
deleted, only superseded by a new version. Check the preview carefully.

**3. The dataset is not in either archive.** Only the frozen reference list
(`data/eval_set_300.json`) is. That is deliberate — the corpus is CC BY-SA 4.0 and
stays with its own provider.

---

## Record 1 — Software

**Upload type:** Software
**Access right:** Open Access
**Licence:** MIT License
**Version:** 1.0.0
**Language:** English
**DOI:** leave empty — Zenodo assigns it on publish

**Title**

```
Thai ASR with an N-gram Engram on Qwen3-ASR-0.6B: evaluation package for BF16, NF4, and LLM.int8 inference on ROCm
```

**Creators**

```
Nammungkun, Thanabodee
ORCID: 0009-0004-9410-9839
```

**Description** (paste as-is; Zenodo renders limited HTML, plain text is fine)

```
Evaluation package for adding a token-level bigram/trigram memory (Engram) to a
frozen Qwen3-ASR-0.6B, and for comparing three numerical precisions on Thai
speech recognition under ROCm.

Method. A Thai N-gram Engram is injected after decoder layer 2's FFN residual of
a frozen Qwen3-ASR-0.6B. Quality is measured on a fixed held-out set of 300 Thai
clips taken from source offset 250,000 of the Porjai-Thai-voice-dataset-central
streaming split, filtered to 1-8 seconds. Decoding is greedy with
max_new_tokens=192 and use_cache=False. Three precisions are compared: BF16, NF4
(bitsandbytes weight-only, group size 32, BF16 compute, double quantization
disabled, applied to Linear and Embedding modules), and LLM.int8 (bitsandbytes
Linear8bitLt, threshold 6.0, applied to Linear modules only, embeddings left in
BF16). Model-tensor storage, peak allocated VRAM, and per-clip latency are
reported separately from quality.

Reported results, with the limits that apply to each:

- BF16, 300 clips. Baseline corpus character error rate (CER) 17.8051% and
  character-level micro F1 86.3235%; Engram step 750 CER 9.0448% and F1
  92.9302%. Engram checkpoints were tested at steps 300, 500, 750, and 1000;
  step 750 was best on this slice. This is one fixed 300-clip sample, not the
  full corpus, and no confidence intervals were computed.
- NF4. Baseline CER 28.7086%, Engram step 750 CER 20.9178%. Model-tensor
  storage fell by about 62%, but peak allocated VRAM and per-clip latency both
  rose relative to BF16. Engram NF4 remained 3.1128 CER percentage points above
  the BF16 baseline. That is a difference in CER, not a 3% loss of accuracy.
- LLM.int8. Baseline CER 18.8901%, Engram step 750 CER 10.8413%. Storage fell by
  about 29% versus BF16, but peak allocated VRAM rose by about 53-54% and mean
  latency rose from 0.755 to 4.354 s/clip for the baseline and from 0.834 to
  4.055 s/clip for Engram.
- Export and reload. Both NF4 variants were exported as self-contained bundles
  and reloaded into freshly rebuilt architectures; reloaded predictions matched
  the in-memory ones on all 300 clips. The bundles load only against
  bitsandbytes 0.50.2, because the packed Params4bit layout is not a stable
  format.
- Compute. All timings are from one AMD Radeon RX 9070 XT (gfx1201) with
  ROCm/HIP 7.1.52802-9999, PyTorch 2.9.1, and bitsandbytes 0.50.2. Fewer bits did
  not mean faster inference in this setup.

What this deposit contains: the library extracted from the original evaluation
notebook, one numbered script per reported result, the frozen evaluation-set
manifest with a sha256 over its reference list, `checkpoints/SHA256SUMS` as the
manifest of the weights, and every per-sample prediction, summary table, and
figure behind the reported numbers.

The Engram weights themselves are not redistributed here. They are hosted on the
Hugging Face Hub at Thanabordee/Qwen3-ASR-0.6B-Thai-Engram, and
`checkpoints/SHA256SUMS` records what they should hash to. Nothing is withheld:
the weights are public at that address, and keeping them out of this archive
avoids duplicating 348 MB.

Reproducibility is partial, not bitwise. The dataset is streamed rather than
pinned to a revision, so the frozen reference list is the contract: each script
re-checks its streamed clips against data/eval_set_300.json and fails if the
upstream data changed. The Engram training loop is not included; the checkpoints
are inputs, fetched from the Hugging Face Hub. Quantized GGUF inference is out of
scope, and no number here comes from llama.cpp.

Checks that ship with the deposit: scripts/00_check_setup.py verifies files,
checkpoint checksums, and the frozen set without a GPU; scripts/07_verify_reproduction.py
re-runs each variant on a few clips and asserts the decoded text and the exact
tensor-storage values; scripts/06_make_tables_figures.py recomputes every corpus
CER from the saved predictions and raises if any disagrees with the value recorded
at run time.

The repository does not redistribute the Porjai-Thai-voice-dataset-central corpus
or Qwen/Qwen3-ASR-0.6B; obtain each from its own source under its own terms.
```

**Keywords**

```
automatic speech recognition
Thai speech recognition
Qwen3-ASR
n-gram memory
Engram
quantization
NF4
LLM.int8
bitsandbytes
ROCm
model compression
reproducibility
```

**Related works** — add after the reports record exists:
relation `is supplemented by`, identifier `10.5281/zenodo.<reports-id>`

**Alternate location** — the trained Engram delta alone, as a Hugging Face model
repository: <https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram>.
If Zenodo offers it, use the relation `is supplemented by`; otherwise put the URL
in the description. The Hub copy is the delta plus a standalone loader, so it is
a strict subset of this archive.

**Alternate location** — the source repository, public at
<https://github.com/ThanabordeeN/Engram-qwenASR>, related as `is supplemented by`.
Note the account name differs from the Hub one: GitHub is `ThanabordeeN`, Hugging
Face is `Thanabordee`.

---

## Record 2 — Reports

**Upload type:** Publication → Preprint
**Access right:** Open Access
**Licence:** Creative Commons Attribution 4.0 International
**Version:** 1.0.0
**Language:** English — the archive also carries a Thai edition
**DOI:** leave empty — Zenodo assigns it on publish

**Title**

```
Adding a Thai N-gram Engram to Qwen3-ASR-0.6B: effects on transcription error, memory, and latency
```

**Creators**

```
Nammungkun, Thanabodee
ORCID: 0009-0004-9410-9839
```

**Description**

```
Technical report, version 1.0.0, in English and Thai, deposited as a preprint.
Both editions are included as PDF, in full, one file per language.

The report asks whether a token-level N-gram memory (Engram) reduces Thai
transcription errors when added to a frozen Qwen3-ASR-0.6B, and what three
numerical precisions cost in quality, model-tensor storage, peak allocated VRAM,
and latency. It is written for a reader who is new to ASR terminology: the terms
that carry the argument are defined where they are used.

Contents: five chapters covering the problem and scope, the background and model
design, the methodology, the results, and the conclusions, followed by
acknowledgements, a reproducibility-artifact appendix, and the bibliography.

Findings, with the limits that apply to each:

- Engram step 750 in BF16 reduced corpus character error rate from 17.8051% to
  9.0448% on a fixed 300-clip Thai evaluation set, with character-level micro F1
  rising from 86.3235% to 92.9302%. This is one fixed sample from one offset of
  one corpus, not the full dataset, and no confidence intervals were computed.
- Quantization cost quality in this setup. NF4 raised CER to 28.7086% for the
  baseline and 20.9178% for Engram; LLM.int8 raised it to 18.8901% and 10.8413%.
  Engram NF4 remained 3.1128 CER percentage points above the BF16 baseline, which
  is a difference in CER and not a 3% loss of accuracy.
- Smaller weights were not faster. Both quantized variants ran slower per clip
  than their BF16 counterpart on this GPU and backend. NF4 reduced model-tensor
  storage by about 62% and LLM.int8 by about 29%, but both increased peak
  allocated VRAM.
- The two quantizers do not touch the same parameters: NF4 applies to Linear and
  Embedding modules, LLM.int8 to Linear modules only, leaving embeddings in BF16.
  This is a limitation of the comparison and the report says so.

The report is explicit that these are measurements of one ROCm/bitsandbytes
configuration on one GPU, not general performance claims, and that the Engram
training loop is not part of the deposit. Quantized GGUF inference is out of
scope and no number comes from llama.cpp.

The Thai speech data comes from the Porjai-Thai-voice-dataset-central corpus
(CMKL University, CC BY-SA 4.0), which is not redistributed; the 300 clips used
are identified only by offset and by a frozen reference list. The base model is
Qwen3-ASR-0.6B from the Qwen Team.
```

**Keywords**

```
automatic speech recognition
Thai speech recognition
Qwen3-ASR
Engram
n-gram memory
quantization
NF4
LLM.int8
ROCm
model compression
technical report
```

**Related works** — add after the software record exists:
relation `is supplement to`, identifier `10.5281/zenodo.<software-id>`

---

## After publishing both records — done 2026-09-24

1. **Done.** Each DOI is in the other record's **Related works**, plus the
   GitHub repository and the Hub model on both. Applied with
   `scripts/09_deposit_zenodo.py --link`, which unlocks a published record with
   `actions/edit`, writes the metadata, and re-publishes. Zenodo's docs are
   explicit that this does not affect the DOI; only *files* are frozen after
   publication. Both records stayed at version 1.0.0.
2. **Done.** The software DOI is in `CITATION.cff` and both DOIs are in
   `README.md`.
3. **Done.** The repository is public at
   <https://github.com/ThanabordeeN/Engram-qwenASR>, so both records link to it.

**Accepted drift.** The published software archive was frozen before the README's
redistribution wording was corrected and before `CITATION.cff` gained
`repository-code` and the DOIs, so it differs from the working tree in exactly
those two files. Zenodo cannot replace a file on a published record: the docs say
files "can only be edited (added, modified or deleted) after publication by
contacting support".

**Decision: leave it.** No 1.0.1, no support request. The archive's code, results,
and data match the working tree; only those two documentation files inside the zip
are older than the repository's. Cite the repository, not the archive, for the
current wording. If a 1.0.1 is ever cut for another reason, it will pick up both
files automatically.

The local `sha256` recorded above is therefore the working tree's, not the
published file's; the published one is
`57c1cc489ef2fcf854e1f0d3bf653a2b81aaa58df57aca34a92c5376d34dfff3`.

---

## What is in each archive, and what is not

**Software archive** — `README.md`, `REPRODUCE.md`, `LICENSE`, `CITATION.cff`,
`requirements.txt`, `.gitignore`, `src/`, `scripts/` (except `09`, see below),
`configs/`, `data/`, `docs/`, `hf/` (model card, standalone loader, generated
config — but not the generated delta, which `scripts/08_publish_hf.py`
rebuilds), `results/{predictions,summaries,tables,figures}/`,
`checkpoints/SHA256SUMS` and `checkpoints/latest.json` (**not** the `.pt`
weights), `archive/notebooks/`, and `exports/nf4/manifest.json`.

**Reports archive** — not deposited. The preprint record holds the two PDFs
only. `reports/{en,th}/` (Markdown, LaTeX, and PDF) and `reports/LICENSE.md`
remain in the working tree, and `./zenodo/build_archives.sh` still packages them
with `results/figures/` into `zenodo/EngramQwenASR-technical-reports-v1.0.0.zip`
as a local copy of the sources, but nothing uploads that zip. The relative
layout is preserved inside it because the `.tex` files resolve figures through
`\graphicspath{{../../results/figures/}}`.

Deliberately excluded:

| Excluded | Why |
|---|---|
| `exports/nf4/*.pt` (1.2 GB) | regenerable with `scripts/04_export_eval_nf4.py`, and only loadable against bitsandbytes 0.50.2 |
| `results/parts/` | resumable chunk scratch for the LLM.int8 benchmark, superseded once combined |
| `reports/build/` | LaTeX intermediate files |
| `reports/archive/nf4_memory_accuracy_technical_report.md` | earlier interim report, superseded by the five-chapter reports |
| `scripts/09_deposit_zenodo.py` | deposits these records, so it needs `zenodo/METADATA.md` and the two zips; `zenodo/` cannot ship inside the archive whose sha256 it records |
| `zenodo/deposits.json` | local draft state from `scripts/09` |
| `.venv/`, `.venv.cuda-backup/`, `__pycache__/` | local environments |
| the Porjai corpus, Qwen3-ASR-0.6B | third-party resources, fetched at run time |

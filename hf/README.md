---
license: cc-by-sa-4.0
language:
  - th
base_model: Qwen/Qwen3-ASR-0.6B
datasets:
  - CMKL/Porjai-Thai-voice-dataset-central
pipeline_tag: automatic-speech-recognition
tags:
  - thai
  - speech-recognition
  - qwen3-asr
  - engram
  - n-gram-memory
  - rocm
  - parameter-efficient
---

# Qwen3-ASR-0.6B + Thai N-gram Engram

A **token-level bigram/trigram memory (Engram)** injected into a frozen
`Qwen/Qwen3-ASR-0.6B`, trained for Thai speech recognition. On a fixed 300-clip
held-out set it reduced corpus character error rate from **17.8051 % to 9.0448 %**.

**This repository holds a delta, not a full model.** The base Qwen3-ASR-0.6B
weights are not duplicated here — they are fetched from the Hub. The only
trained parameters are the **8,249,344 Engram parameters** in
`engram_step_000750.pt` (32 MiB).

---

## What the Engram is

A hash-based N-gram lookup memory, not an adapter or a LoRA. It reads the
**token IDs** of the decoder input, hashes every bigram and trigram through 16
independent heads into 10,000 / 2,000 buckets, and turns the retrieved vectors
into a residual:

```
hidden = decoder_layer_2_output + Engram(hidden, input_ids)
```

It is injected after the FFN residual of **decoder layer 2 only**. During
inference the residual is applied to the last prompt position and every
generated position; the earlier prompt positions are masked out.

`key_proj`, `value_proj`, and `short_conv` are initialised to **zero**, so at
step 0 the Engram contributes exactly nothing and the model is bit-identical to
the baseline. All 785 M base parameters stay frozen — only the Engram trains.

| | Parameters | Storage |
|---|---:|---:|
| Base `Qwen/Qwen3-ASR-0.6B` (frozen, not in this repo) | 785,114,240 | 1.46 GiB (BF16) |
| **Engram, layer 2 (this repo)** | **8,249,344** | **32 MiB (fp32)** |

---

## Results

300 fixed Thai clips from source offset 250,000 of
`CMKL/Porjai-Thai-voice-dataset-central`, filtered to 1–8 seconds. Greedy
decoding, `max_new_tokens=192`, `use_cache=False`. One AMD Radeon RX 9070 XT
(gfx1201), ROCm/HIP 7.1.52802-9999, PyTorch 2.9.1, bitsandbytes 0.50.2.

| Variant | CER (%) ↓ | Char F1 (%) ↑ | Weights (GiB) | Peak VRAM (GiB) | Mean s/clip |
|---|---:|---:|---:|---:|---:|
| Baseline, BF16 | 17.8051 | 86.3235 | 1.462 | 1.590 | 0.755 |
| **Engram step 750, BF16** | **9.0448** | **92.9302** | 1.493 | 1.623 | 0.834 |
| Baseline, NF4 | 28.7086 | 79.7589 | 0.557 | 1.024 | 1.333 |
| Engram step 750, NF4 | 20.9178 | 85.3693 | 0.562 | 1.034 | 1.556 |
| Baseline, LLM.int8 | 18.8901 | 85.3726 | 1.030 | 2.451 | 4.354 |
| Engram step 750, LLM.int8 | 10.8413 | 91.8857 | 1.055 | 2.479 | 4.055 |

F1 is pooled **character-level** micro-F1 after Thai normalization and
whitespace removal — not word-level F1, and not accuracy.

The Engram checkpoints were evaluated at steps 300, 500, 750, and 1000; **750
was best on this slice** and is the one published here. Only step 750 is
released.

---

## Usage

```bash
pip install torch transformers qwen-asr huggingface_hub
```

```python
import soundfile as sf
from engram_loader import load_engram_model, transcribe

model, processor, runtime = load_engram_model()   # downloads base + this delta
audio, sr = sf.read("clip.wav")                   # 16 kHz mono, float32

print(transcribe(model, processor, audio, runtime, use_engram=True))   # Engram
print(transcribe(model, processor, audio, runtime, use_engram=False))  # baseline
```

`hf/engram_loader.py` is a self-contained port of the injection code — copy it
next to your script. Toggle `runtime["enabled"] = False` to get the frozen
baseline from the same loaded model; that is exactly how the "Baseline" rows
above were produced, so the comparison costs one flag and no extra weights.

```python
import torch
from engram_loader import load_engram_model, transcribe

# The Engram is fp32; the base model runs in BF16. Keep that split.
model, processor, runtime = load_engram_model(device="cuda", dtype=torch.bfloat16)
```

---

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B`, entirely frozen |
| Data | `CMKL/Porjai-Thai-voice-dataset-central`, `train` split, streamed |
| Audio filter | 1.0–8.0 s |
| Checkpoints released | step 750 (steps 300 / 500 / 1000 were also evaluated) |
| Seed | 42 |
| Engram layers | `[2]` |
| Buckets | bigram 10,000 / trigram 2,000 |
| Memory dim / heads | 512 / 16 |
| Short-conv kernel | 4, dilation 3 |
| Injection | after decoder layer 2, `hidden + delta` |

The training loop is **not** released — the checkpoint is an input to the
evaluation package, not an output of it.

---

## Limitations

- **One sample, one offset.** Every number is from a single fixed 300-clip slice
  at source offset 250,000. No confidence intervals were computed. This is not
  the full corpus.
- **One machine.** All timings are from one RX 9070 XT on ROCm. Fewer bits did
  **not** mean faster inference here: NF4 and LLM.int8 were both slower per clip
  than BF16.
- **Quantization hurts.** NF4 left the Engram 3.1128 CER points above the BF16
  *baseline*. That is a difference in CER, not a 3 % loss of accuracy.
- **The quantizers are not comparable to each other.** NF4 applies to Linear and
  Embedding modules; LLM.int8 applies to Linear only, leaving embeddings in BF16.
- **Bigram/trigram only.** Nothing longer than a trigram is memorized, and the
  memory is keyed on token IDs, so a tokenizer change invalidates the tables.
- **Thai only.** Trained exclusively on Thai speech; no other language was
  evaluated.

---

## Licensing — read this

The Engram checkpoints were trained on
[`CMKL/Porjai-Thai-voice-dataset-central`](https://huggingface.co/datasets/CMKL/Porjai-Thai-voice-dataset-central),
which is licensed **CC BY-SA 4.0**. These weights are therefore released under
**CC BY-SA 4.0** as well, and you must credit the corpus if you redistribute
them.

`engram_loader.py` is **MIT** licensed, separate from the weights.

The base `Qwen/Qwen3-ASR-0.6B` is **Apache-2.0** and is not redistributed here —
obtain it from its own repository under its own terms. The Porjai corpus is not
redistributed either.

---

## Citation

```bibtex
@misc{nammungkun2026engramqwenasr,
  title        = {Adding a Thai N-gram Engram to Qwen3-ASR-0.6B:
                  quality, memory, and latency of BF16, NF4, and LLM.int8
                  inference on ROCm},
  author       = {Nammungkun, Thanabodee},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Thanabordee/Qwen3-ASR-0.6B-Thai-Engram}},
  note         = {ORCID 0009-0004-9410-9839}
}
```

A companion Zenodo record carries the full evaluation package (library, one
script per reported result, every per-sample prediction, table, and figure) and
the technical report in English and Thai.

---

## ภาษาไทย

**Engram แบบ N-gram ภาษาไทยบน Qwen3-ASR-0.6B** — หน่วยความจำระดับโทเคน
(bigram/trigram) ต่อเข้ากับ decoder layer 2 ของ Qwen3-ASR-0.6B ที่แช่แข็ง weights
ไว้ทั้งหมด เทรนเฉพาะ Engram 8,249,344 พารามิเตอร์

repo นี้เก็บ **delta เท่านั้น** ไม่ได้ก็อป weights ของ base model มาไว้ — โหลด base
จาก `Qwen/Qwen3-ASR-0.6B` แยก

บนชุดทดสอบคงที่ 300 คลิป **CER ลดจาก 17.8051 % เหลือ 9.0448 %** (character F1
86.3235 % → 92.9302 %) และเปิด/ปิด Engram ได้ด้วย flag เดียวบนโมเดลที่โหลดแล้ว

ข้อจำกัด: ตัวเลขทั้งหมดมาจากคลิปชุดเดียว 300 คลิป ที่ offset 250,000 ของ corpus
ไม่ได้คำนวณ confidence interval และวัดบน RX 9070 XT + ROCm เครื่องเดียว

**สัญญาอนุญาต:** weights เทรนจากข้อมูล CC BY-SA 4.0 จึงปล่อยเป็น **CC BY-SA 4.0**
ส่วน `engram_loader.py` เป็น MIT

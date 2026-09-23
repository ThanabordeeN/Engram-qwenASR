# Native llama.cpp Engram support — experimental, not validated

**Status: does not reproduce PyTorch output. No number in the report uses this
code.** It is recorded here so the work is not lost and so nobody mistakes it for
a validated backend.

## What was attempted

Add native Engram inference to llama.cpp so the quantized GGUF models
(`Q4_K_M` and BF16) could be evaluated on the same 300 clips, with the Engram
residual applied after decoder layer 2's FFN residual, per-sequence token/value
history, and prompt-boundary masking.

The patch is at
[`patches/llama.cpp-engram-experimental.patch`](../patches/llama.cpp-engram-experimental.patch),
against upstream commit
`4e416ee7308dd6b581796f1a6241276cd5982691` (recorded in
`patches/llama.cpp-BASE_COMMIT.txt`). 715 lines across 11 files:

| File | Purpose |
|---|---|
| `conversion/qwen3vl.py` | export the 40 Engram tensors; add Qwen3-ASR PAD token metadata (`151643`) |
| `convert_hf_to_gguf.py`, `conversion/base.py` | wiring for the new tensors |
| `src/llama-arch.{h,cpp}` | `ENGRAM_*` tensor names and metadata |
| `src/models/qwen3vl.cpp` | graph-side Engram residual after layer 2 |
| `src/llama-graph.{h,cpp}`, `src/llama-context.{h,cpp}`, `src/models/models.h` | per-sequence token/value history and its plumbing |

To apply:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 4e416ee7308dd6b581796f1a6241276cd5982691
git apply /path/to/patches/llama.cpp-engram-experimental.patch
```

## What works

* The converter exports all 40 Engram tensors and each was verified equal to the
  checkpoint value.
* The GGUF files load and generate audio transcripts.
* The earlier assertion failure
  (`GGML_ASSERT(ubatch->token != nullptr && "Engram requires token IDs in the
  decoder batch")`) was resolved by feeding `<|audio_pad|>` IDs (151676) for
  audio-embedding batches and recording them in the Engram history.

## What does not work

Baseline parity holds, Engram parity does not. On one clip:

| Implementation | Output |
|---|---|
| Reference | `เสียใจด้วยตอนนี้คนสาวไม่อยู่` |
| PyTorch baseline | `สจด้วยตอนนี้คนสาวไม่อยู่` |
| llama.cpp baseline | `สจด้วยตอนนี้คนสาวไม่อยู่` |
| PyTorch Engram | `สจ๊วยด้วยตอนนี้คนสาวไม่อยู่` |
| llama.cpp Engram | `สจ๊ะใจด้วยตอนนี้คนสาวไม่อยู่` |

The baseline matching exactly while Engram diverges localizes the problem to the
Engram path, not to the audio front end or the tokenizer.

## Likely causes, in order

1. **Prompt-boundary masking.** PyTorch masks the Engram delta to positions at or
   after `inference_start = input_ids.shape[1] - 1`. The llama.cpp port masks on
   prompt boundaries reconstructed from the batch layout. The whitespace-only
   prompt differs between the two, so the masked spans differ.
2. **N-gram history contents.** PyTorch keeps the exact processed token IDs,
   including the audio placeholders inserted by the processor. The llama.cpp port
   synthesizes `<|audio_pad|>` runs; if the count or placement differs by even one
   token, every bigram and trigram index downstream shifts.
3. **Hash-seed indexing.** `hash_seeds[n, h, k]` is indexed by order and head.
   The GGUF tensor layout must match that ordering exactly; a transposed axis
   would produce plausible but wrong buckets.

## Next step

Diff the two token-ID sequences for the same clip first — everything else is
downstream of that. `conversion/qwen3vl.py` records the Engram history, and the
processor's exact prompt/audio token sequence is available from the PyTorch side
via `Session` in `src/qwen3asr_engram.py`. Only after the sequences match is it
worth looking at masking or hash indexing.

## Do not

Do not use this patch to produce Q4_K_M or GGUF numbers for the report, and do
not describe the llama.cpp path as validated. `Q4_K_M` evaluation stays blocked
until Engram output matches PyTorch on the sample above.

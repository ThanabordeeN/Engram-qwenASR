# Environment record

The reported numbers were produced on the machine below. Anything that changes
here — GPU, ROCm, PyTorch, bitsandbytes, or the dataset revision — can change the
results, so record your own values if you re-run the study.

## Hardware

| Item | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT (16 GiB), ROCm arch `gfx1201` |
| CPU | Intel Core i5-14400F |
| OS | Fedora Linux, kernel as shipped with the ROCm 7.1 install |

## Software

| Component | Version |
|---|---|
| ROCm / HIP | `7.1.52802-9999` |
| PyTorch | `2.9.1` (ROCm/HIP build; `torch.version.hip` is not `None`) |
| torchvision / torchaudio | `0.24.0` / `2.9.0` |
| bitsandbytes | `0.50.2` |
| transformers | `4.57.6` |
| qwen-asr | `0.0.6` |
| datasets | `5.0.1` |
| jiwer | `4.0.0` |
| librosa / soundfile | `1.0.0` / `0.14.0` |
| numpy / pandas / matplotlib | `2.4.6` / `3.0.5` / `3.10.8` |
| LuaLaTeX | LuaHBTeX 1.24.0 (TeX Live 2026) |

Full pinned list: [`../requirements.txt`](../requirements.txt).

Install order matters. PyTorch must come from the ROCm wheel index that matches
your ROCm version; installing a CUDA wheel first is the most common way to end up
with `torch.version.hip is None`, which the library rejects with an explicit
error rather than silently running on the wrong backend.

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1 torchaudio==2.9.0
pip install -r requirements.txt
```

## Model and dataset

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` (fetched at run time; not redistributed here) |
| Dataset | `CMKL/Porjai-Thai-voice-dataset-central`, split `train`, streamed |
| Evaluation offset | 250 000 source rows |
| Evaluation size | 300 clips passing the 1.0–8.0 s duration filter |
| Frozen set hash | see `references_sha256` in [`../data/eval_set_300.json`](../data/eval_set_300.json) |

Because the dataset is streamed rather than pinned to a revision, the frozen
reference list is the contract: every script re-checks its streamed clips against
`data/eval_set_300.json` and fails if the upstream data changed. For a
byte-stable rerun, pin a revision in `Config.dataset_name`, e.g.
`CMKL/Porjai-Thai-voice-dataset-central@<commit-sha>`.

## Decoding settings

Greedy, no sampling, no beam search, no KV cache — see
[`../configs/eval_300.json`](../configs/eval_300.json):

```json
{"max_new_tokens": 192, "generation_use_cache": false}
```

`use_cache=False` is deliberate: the Engram wrapper injects a per-layer residual
keyed on the full token history, so incremental decoding was disabled to keep the
comparison exact. This is also why the latency figures are seconds per clip
rather than tens of milliseconds.

## Quantization settings

| Method | Applied to | Settings |
|---|---|---|
| NF4 | every `nn.Linear` **and** `nn.Embedding` in Qwen and Engram | `quant_type="nf4"`, group size 32, `compute_dtype=torch.bfloat16`, double quantization off |
| LLM.int8 | every `nn.Linear` only | `Linear8bitLt`, `threshold=6.0`, `has_fp16_weights=False` |

Left unquantized in both cases: biases, RMSNorm weights, the Engram short
convolution, and 1-D parameters. For LLM.int8 the embeddings also stay BF16, so
the two methods do **not** quantize the same parameter set — the report says so
explicitly, and it is a limitation of the comparison rather than a detail.

LLM.int8 casts BF16 activations to FP16 internally before the int8 matmul;
bitsandbytes prints one line per layer about this. It is expected, and
`scripts/05` silences that specific logger so the progress output stays readable.

## Environment variables

| Variable | Effect |
|---|---|
| `ENGRAM_LOCAL_ROOT` | Override the project root that `results/` and `checkpoints/` hang off |
| `ENGRAM_LOCAL_CKPT_DIR` | Override the checkpoint directory |
| `ENGRAM_FORCE_FP16=1` | Force FP16 instead of BF16 for the Qwen weights, for ROCm builds with a BF16 problem |

## Recreating the environment from scratch

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1 torchaudio==2.9.0
pip install -r requirements.txt
python scripts/00_check_setup.py
```

`scripts/00_check_setup.py` prints the detected torch/HIP/GPU line and fails if
the ROCm runtime, the checkpoints, the frozen evaluation set, or the expected
result files are missing.

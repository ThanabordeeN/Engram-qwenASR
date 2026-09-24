#!/usr/bin/env python
"""Publish the trained Engram delta to the Hugging Face Hub.

The Engram is a *delta*, not a standalone model: the Qwen3-ASR-0.6B base weights
stay frozen and are fetched from the Hub. This script stages the delta, checks
that it still reproduces the in-repo predictions, and uploads.

    # stage hf/engram_step_000750.pt and hf/config.json
    python scripts/08_publish_hf.py --stage

    # prove the stripped checkpoint + standalone loader match the in-repo path
    python scripts/08_publish_hf.py --verify-samples 4

    # create the repo and push (add --private for a private repo)
    python scripts/08_publish_hf.py --upload

`--stage --verify --upload` in one call is fine; verification is skipped
automatically when no GPU is present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hf"))

HF_DIR = PROJECT_ROOT / "hf"
REPO_ID = "Thanabordee/Qwen3-ASR-0.6B-Thai-Engram"
STEP = 750
# Files that live in git; everything else in hf/ is generated and ignored.
SOURCE_FILES = ["README.md", "engram_loader.py"]
KEEP_KEYS = ["format", "base_model", "step", "config", "engram_state"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_checkpoint(src: Path, dst: Path) -> dict:
    """Drop optimizer/scheduler/RNG state — this is an inference artifact."""
    import torch

    payload = torch.load(src, map_location="cpu", weights_only=False)
    missing = [k for k in KEEP_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{src} is missing {missing}")
    stripped = {k: payload[k] for k in KEEP_KEYS}
    if int(stripped["step"]) != STEP:
        raise ValueError(f"{src} reports step {stripped['step']}, expected {STEP}")

    torch.save(stripped, dst)

    reloaded = torch.load(dst, map_location="cpu", weights_only=False)
    assert list(reloaded) == KEEP_KEYS, f"key drift: {list(reloaded)}"
    for layer, state in payload["engram_state"].items():
        got, want = reloaded["engram_state"][layer], state
        if set(got) != set(want):
            raise AssertionError(f"layer {layer}: state_dict keys changed")
        for name, tensor in want.items():
            if not tensor.equal(got[name]):
                raise AssertionError(f"layer {layer}.{name}: weights changed by the strip")
    if reloaded["config"] != payload["config"]:
        raise AssertionError("engram config changed by the strip")
    return stripped


def write_config(payload: dict, dst: Path) -> dict:
    cfg = payload["config"]
    out = {
        "architecture": "ThaiNgramEngram",
        "base_model_id": payload["base_model"],
        "checkpoint_step": int(payload["step"]),
        "engram_layers": cfg["layers"],
        "engram_buckets": {str(k): int(v) for k, v in cfg["buckets"].items()},
        "engram_dim": int(cfg["memory_dim"]),
        "engram_heads": int(cfg["heads"]),
        "engram_kernel": int(cfg["kernel"]),
        "seed": int(cfg["seed"]),
        "injection": "hidden + delta after the decoder layer's output, masked to the "
                     "last prompt position and all generated positions at inference",
    }
    dst.write_text(json.dumps(out, indent=2) + "\n")
    return out


def stage(step: int = STEP) -> Path:
    """Build hf/engram_step_XXXXXX.pt + hf/config.json from checkpoints/."""
    import torch  # noqa: F401  (fail early with a clear ImportError)

    src = PROJECT_ROOT / "checkpoints" / f"engram_step_{step:06d}.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = HF_DIR / src.name

    payload = strip_checkpoint(src, dst)
    cfg = write_config(payload, HF_DIR / "config.json")
    n_params = sum(v.numel() for state in payload["engram_state"].values()
                   for v in state.values() if v.dim() > 0)

    print(f"staged  {dst.relative_to(PROJECT_ROOT)}")
    print(f"        {dst.stat().st_size/2**20:.1f} MiB "
          f"(from {src.stat().st_size/2**20:.1f} MiB), sha256 {sha256(dst)}")
    print(f"        layers={cfg['engram_layers']} buckets={cfg['engram_buckets']} "
          f"dim={cfg['engram_dim']} heads={cfg['engram_heads']} (~{n_params:,} tensors' elements)")
    return dst


def verify(samples: int) -> None:
    """In-repo Session (original ckpt) vs standalone loader (stripped ckpt)."""
    import torch

    if not torch.cuda.is_available():
        print("verify: no CUDA/ROCm device — skipped")
        return

    from qwen3asr_engram import Config, Session, collect_eval_records  # noqa: E402
    import engram_loader as el  # noqa: E402

    cfg = Config()
    records = collect_eval_records(Config(**{**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__},
                                             "eval_samples": samples}))
    print(f"verify: {len(records)} clips, original checkpoint vs stripped + standalone loader")

    reference = Session(cfg, engram=True, verbose=False)
    reference.load_checkpoint(STEP)
    expected = [reference.transcribe(r, use_engram=True) for r in records]
    expected_off = [reference.transcribe(r, use_engram=False) for r in records]
    del reference
    torch.cuda.empty_cache()

    # Build the standalone path from the staged local file, so this runs pre-upload.
    from transformers import GenerationConfig
    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(cfg.model_id, dtype=torch.bfloat16, device_map=None)
    base = asr.model.to("cuda")
    el._patch_outer_forward(base)
    base.generation_config = GenerationConfig.from_model_config(base.config)
    runtime = el.make_runtime()
    ckpt = HF_DIR / f"engram_step_{STEP:06d}.pt"
    stored = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]
    engram_config = {k: stored[k] for k in ("layers", "buckets", "memory_dim", "heads", "kernel", "seed")}
    el.load_engram(base, ckpt, engram_config, runtime, "cuda")
    base.eval()
    processor = asr.processor

    got = [el.transcribe(base, processor, r["audio_array"], runtime, use_engram=True) for r in records]
    got_off = [el.transcribe(base, processor, r["audio_array"], runtime, use_engram=False) for r in records]

    bad = [i for i, (a, b) in enumerate(zip(expected, got)) if a != b]
    bad_off = [i for i, (a, b) in enumerate(zip(expected_off, got_off)) if a != b]
    if bad or bad_off:
        for i in sorted(set(bad) | set(bad_off)):
            print(f"  clip {i}\n    in-repo    on={expected[i]!r}\n               off={expected_off[i]!r}"
                  f"\n    standalone on={got[i]!r}\n               off={got_off[i]!r}")
        raise AssertionError(f"mismatch on {len(bad)}/{len(records)} engram clips, "
                             f"{len(bad_off)}/{len(records)} baseline clips")
    if expected == expected_off:
        raise AssertionError("Engram changed nothing — the checkpoint is not being applied")
    print(f"verify: OK — {len(records)}/{len(records)} clips byte-identical, "
          f"Engram path differs from baseline")


def upload(private: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    for name in SOURCE_FILES:
        if not (HF_DIR / name).is_file():
            raise FileNotFoundError(f"hf/{name} is missing")
    ckpt = HF_DIR / f"engram_step_{STEP:06d}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{ckpt} is missing — run --stage first")

    create_repo(REPO_ID, repo_type="model", private=private, exist_ok=True)
    api = HfApi()
    for name in SOURCE_FILES + ["config.json", ckpt.name]:
        path = HF_DIR / name
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=REPO_ID,
                        commit_message=f"Add {name}")
    api.create_tag(REPO_ID, tag="v1.0.0", repo_type="model")
    print(f"published https://huggingface.co/{REPO_ID}  (v1.0.0, private={private})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", action="store_true", help="build the uploadable delta + config")
    p.add_argument("--verify-samples", type=int, default=0, metavar="N",
                   help="compare in-repo vs standalone on N clips (needs a GPU)")
    p.add_argument("--upload", action="store_true", help="create the repo and push")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--clean", action="store_true", help="remove generated files from hf/")
    args = p.parse_args()

    if args.clean:
        for path in list(HF_DIR.glob("*.pt")) + [HF_DIR / "config.json"]:
            path.unlink(missing_ok=True)
            print(f"removed hf/{path.name}")
        return
    if not any([args.stage, args.verify_samples, args.upload]):
        p.print_help()
        return

    if args.stage or args.verify_samples or args.upload:
        if not (HF_DIR / f"engram_step_{STEP:06d}.pt").is_file():
            stage()
    if args.verify_samples:
        verify(args.verify_samples)
    if args.upload:
        upload(args.private)


if __name__ == "__main__":
    main()

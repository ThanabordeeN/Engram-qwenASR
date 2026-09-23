"""Qwen3-ASR + Thai N-gram Engram — evaluation library.

This module is the single source of truth for the evaluation described in
``reports/en/project_technical_report_en.pdf`` (Thai: ``reports/th/``).

Provenance: the code bodies below were extracted unchanged from the original
notebook ``Qwen3_ASR_Thai_Engram_ROCm_RX9070XT_Eval300to1000_Scaling.ipynb``
(cells 4, 6, 8, 9, 10, 12, 14, 15, 17, 19). The notebook is archived under
``archive/notebooks/`` and its sha256 is recorded in ``docs/provenance.md``.
The only change is packaging: the side-effecting notebook cells are wrapped in
``Session`` methods, and the module-level constants moved into ``Config``.

Typical use::

    from qwen3asr_engram import Config, Session

    session = Session(Config(), engram=True)
    records = session.collect_records()
    session.load_checkpoint(750)
    text = session.transcribe(records[0], use_engram=True)
"""
from __future__ import annotations

import gc
import inspect
import io
import json
import math
import os
import random
import re
import subprocess
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from datasets import Audio, load_dataset
from jiwer import cer
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig

__all__ = [
    "Config",
    "Session",
    "ThaiNgramEngram",
    "corpus_cer",
    "normalize_thai_transcript",
    "thai_cer_norm",
    "PROJECT_ROOT",
    "CKPT_DIR",
    "RESULTS_DIR",
]

PROJECT_ROOT = Path(
    os.environ.get("ENGRAM_LOCAL_ROOT", str(Path(__file__).resolve().parent.parent))
).expanduser().resolve()
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def results_subdir(name: str) -> Path:
    """Return (and create) a subdirectory of ``results/``."""
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Config:
    """Every setting of the reported run, in one place.

    Defaults reproduce the numbers in the technical report. Override fields to
    run a variant; pass ``configs/eval_300.json`` through :meth:`load` to keep
    the reported configuration pinned in version control.
    """

    model_id: str = "Qwen/Qwen3-ASR-0.6B"
    dataset_name: str = "CMKL/Porjai-Thai-voice-dataset-central"
    dataset_split: str = "train"
    seed: int = 42

    engram_layers: list = field(default_factory=lambda: [2])
    engram_buckets: dict = field(default_factory=lambda: {2: 10_000, 3: 2_000})
    engram_dim: int = 512
    engram_heads: int = 16
    engram_kernel: int = 4

    min_audio_sec: float = 1.0
    max_audio_sec: float = 8.0
    max_new_tokens: int = 192
    generation_use_cache: bool = False

    eval_source_skip: int = 250_000
    eval_samples: int = 300
    checkpoint_steps: list = field(default_factory=lambda: [300, 500, 750, 1000])

    checkpoint_dir: Path = CKPT_DIR

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown config key(s): {unknown}")
        payload["checkpoint_dir"] = Path(payload.get("checkpoint_dir", CKPT_DIR))
        payload["engram_buckets"] = {int(k): int(v) for k, v in payload["engram_buckets"].items()}
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        payload = {f: getattr(self, f) for f in self.__dataclass_fields__}
        payload["checkpoint_dir"] = str(self.checkpoint_dir)
        payload["engram_buckets"] = {str(k): v for k, v in self.engram_buckets.items()}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def checkpoint_paths(self) -> dict[int, Path]:
        return {
            step: Path(self.checkpoint_dir) / f"engram_step_{step:06d}.pt"
            for step in self.checkpoint_steps
        }

    def require_checkpoints(self) -> None:
        missing = [str(p) for p in self.checkpoint_paths().values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoint file(s):\n" + "\n".join(missing) +
                "\nSet ENGRAM_LOCAL_CKPT_DIR or place the engram_step_*.pt files in checkpoints/."
            )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    """Validate the ROCm/HIP runtime and return the compute device."""
    if torch.version.hip is None:
        raise RuntimeError(
            "This project requires ROCm/HIP PyTorch. "
            f"Detected torch={torch.__version__}, torch.version.hip={torch.version.hip!r}. "
            "Install/activate a ROCm PyTorch environment; do not use a CUDA wheel."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch is installed but no AMD GPU is visible.")
    return torch.device("cuda")  # PyTorch ROCm intentionally uses the torch.cuda API.


def resolve_dtype(device: torch.device) -> torch.dtype:
    try:
        use_bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:
        use_bf16 = True
    # Override with ENGRAM_FORCE_FP16=1 if a specific ROCm/qwen-asr build has a BF16 issue.
    if os.environ.get("ENGRAM_FORCE_FP16", "0").strip() == "1":
        use_bf16 = False
    return torch.bfloat16 if use_bf16 else torch.float16


def describe_environment(device: torch.device, dtype: torch.dtype) -> dict:
    try:
        gpu_props = torch.cuda.get_device_properties(0)
        total_vram_gb = gpu_props.total_memory / 2**30
    except Exception:
        total_vram_gb = float("nan")
    info = {
        "torch": torch.__version__,
        "rocm_hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "total_vram_gib": round(float(total_vram_gb), 2),
        "model_dtype": str(dtype),
    }
    try:
        rocminfo = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=False
        ).stdout
        info["rocminfo_arch"] = sorted(set(re.findall(r"gfx\d+", rocminfo)))[:8]
    except Exception as exc:
        info["rocminfo_arch"] = f"not detected: {exc!r}"
    return info


def load_asr_model(cfg: Config, device: torch.device, dtype: torch.dtype):
    """Load the frozen Qwen3-ASR model and patch the outer forward (cell 6)."""
    try:
        asr = Qwen3ASRModel.from_pretrained(cfg.model_id, dtype=dtype, device_map=None)
    except TypeError:
        asr = Qwen3ASRModel.from_pretrained(cfg.model_id, torch_dtype=dtype, device_map=None)

    model = asr.model.to(device)
    processor, tokenizer = asr.processor, asr.processor.tokenizer

    def patch_outer_forward(instance):
        if getattr(instance, "_thai_engram_instance_forward_patched", False):
            return
        if not hasattr(instance, "thinker") or not hasattr(instance.thinker, "forward"):
            raise RuntimeError("Incompatible qwen-asr: thinker.forward is unavailable")

        def _forward(self, input_ids=None, attention_mask=None, input_features=None,
                     feature_attention_mask=None, labels=None, **kwargs):
            # Call the module, not its .forward method, so registered thinker pre-hooks run.
            return self.thinker(input_ids=input_ids, attention_mask=attention_mask,
                input_features=input_features, feature_attention_mask=feature_attention_mask,
                labels=labels, **kwargs)

        instance.forward = types.MethodType(_forward, instance)
        instance._thai_engram_instance_forward_patched = True

    patch_outer_forward(model)
    try:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    except Exception as exc:
        print("generation_config warning:", exc)
    for p in model.parameters():
        p.requires_grad_(False)
    return asr, model, processor, tokenizer


def infer_hidden_size_for(target_model, layers) -> int:
    for obj in (layers[0], getattr(layers[0], "self_attn", None), getattr(layers[0], "mlp", None)):
        for attr in ("hidden_size", "embed_dim"):
            if obj is not None and getattr(obj, attr, None) is not None:
                return int(getattr(obj, attr))
    cfg = getattr(target_model.thinker.config, "text_config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise RuntimeError("Could not infer Qwen hidden size")


def probe_decoder(target_model):
    """Locate the text decoder stack and its hidden size (cell 8)."""
    if not hasattr(target_model, "thinker") or not hasattr(target_model.thinker, "model"):
        raise RuntimeError("Qwen thinker/model path is unavailable")
    decoder_layers = target_model.thinker.model.layers
    decoder_name = "thinker.model.layers"
    if len(decoder_layers) < 20:
        raise RuntimeError(f"Unexpected text decoder depth: {len(decoder_layers)}")
    return decoder_layers, decoder_name, infer_hidden_size_for(target_model, decoder_layers)


# --------------------------------------------------------------------------- #
# Engram architecture (cell 9)
# --------------------------------------------------------------------------- #
class ThaiNgramEngram(nn.Module):
    """Token-ID N-gram memory; it never normalizes or strips Unicode."""

    def __init__(self, hidden_size, buckets, memory_dim=512, num_heads=16,
                 kernel_size=4, pad_id=0, seed=42):
        super().__init__()
        if memory_dim % num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.buckets = {int(k): int(v) for k, v in buckets.items()}
        self.orders = sorted(self.buckets)
        self.memory_dim, self.num_heads = int(memory_dim), int(num_heads)
        self.head_dim, self.pad_id = self.memory_dim // self.num_heads, int(pad_id)
        self.tables = nn.ModuleDict()
        for n in self.orders:
            for h in range(self.num_heads):
                self.tables[f"n{n}_h{h}"] = nn.Embedding(self.buckets[n], self.head_dim)
        gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
        self.register_buffer("hash_seeds", torch.randint(
            1009, 2_000_000_000, (max(self.orders)+1, self.num_heads, max(self.orders)),
            generator=gen, dtype=torch.int64) | 1, persistent=True)
        merged = len(self.orders) * self.memory_dim
        self.key_proj, self.value_proj = nn.Linear(merged, self.hidden_size), nn.Linear(merged, self.hidden_size)
        self.key_norm = nn.RMSNorm(self.hidden_size)
        self.query_norm = nn.RMSNorm(self.hidden_size)
        dilation = max(self.orders)
        self.short_conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=kernel_size,
                                    groups=self.hidden_size, dilation=dilation,
                                    padding=(kernel_size-1)*dilation, bias=False)
        nn.init.zeros_(self.value_proj.weight); nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.short_conv.weight)

    def _shift_right(self, ids, k):
        if k == 0: return ids
        pad = torch.full((ids.shape[0], k), self.pad_id, dtype=ids.dtype, device=ids.device)
        return torch.cat([pad, ids[:, :-k]], dim=1)

    def _memory_for_order(self, ids, n):
        shifted = [self._shift_right(ids, k) for k in range(n)]
        vectors = []
        for h in range(self.num_heads):
            mixed = torch.zeros_like(ids, dtype=torch.int64)
            for k in range(n):
                mixed = torch.bitwise_xor(mixed, shifted[k].to(torch.int64) * self.hash_seeds[n, h, k])
            idx = torch.remainder(mixed, self.buckets[n]).long()
            vectors.append(self.tables[f"n{n}_h{h}"](idx))
        return torch.cat(vectors, dim=-1)

    def forward(self, hidden_states, input_ids, valid_mask=None):
        if valid_mask is not None and valid_mask.shape != input_ids.shape:
            raise RuntimeError("Engram attention mask and input_ids shapes differ")
        memory = torch.cat([self._memory_for_order(input_ids, n) for n in self.orders], dim=-1)
        if valid_mask is not None:
            memory = memory * valid_mask.to(memory.dtype).unsqueeze(-1)
        query = self.query_norm(hidden_states.detach().float())
        key = self.key_norm(self.key_proj(memory.float()))
        gate = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        value = self.value_proj(memory.float()) * torch.sigmoid(gate).unsqueeze(-1)
        conv = self.short_conv(value.transpose(1, 2))[..., :value.shape[1]].transpose(1, 2)
        return value + conv


# --------------------------------------------------------------------------- #
# Engram injection (cell 10)
# --------------------------------------------------------------------------- #
def make_runtime():
    return {"input_ids": None, "attention_mask": None, "labels": None, "enabled": True,
            "inference_start": None, "wrapper_calls": 0, "last_delta_requires_grad": None,
            "last_delta": None, "last_hidden_shape": None, "last_ids_shape": None}


def _layer_hidden(output):
    if torch.is_tensor(output): return output, None
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0], output
    raise RuntimeError(f"Unsupported decoder layer output type: {type(output)!r}")


class EngramInjectedDecoderLayer(nn.Module):
    """Real layer replacement; preserves tuple output used by Transformers."""

    def __init__(self, base_layer, engram, layer_id, runtime):
        super().__init__()
        self.base_layer, self.engram = base_layer, engram
        self.layer_id, self.runtime = int(layer_id), runtime

    def forward(self, *args, **kwargs):
        raw = self.base_layer(*args, **kwargs)
        hidden, sequence = _layer_hidden(raw)
        self.runtime["wrapper_calls"] += 1
        if not self.runtime["enabled"]: return raw
        ids, mask = self.runtime["input_ids"], self.runtime["attention_mask"]
        if not torch.is_tensor(ids):
            raise RuntimeError(f"Layer {self.layer_id}: pre-hook did not capture input_ids")
        if hidden.ndim != 3 or ids.ndim != 2 or hidden.shape[:2] != ids.shape:
            raise RuntimeError(f"Layer {self.layer_id}: hidden/IDs misaligned: hidden={tuple(hidden.shape)} ids={tuple(ids.shape)}")
        if mask is not None and mask.shape != ids.shape:
            raise RuntimeError(f"Layer {self.layer_id}: attention mask shape {tuple(mask.shape)} != ids {tuple(ids.shape)}")
        delta = self.engram(hidden, ids, valid_mask=mask)
        labels = self.runtime["labels"]
        if labels is not None:
            if labels.shape != ids.shape:
                raise RuntimeError(f"Layer {self.layer_id}: labels shape != ids shape")
            target_mask = torch.zeros_like(labels, dtype=torch.bool)
            if labels.shape[1] > 1: target_mask[:, :-1] = labels[:, 1:].ne(-100)
            delta = delta * target_mask.unsqueeze(-1)
        elif self.runtime["inference_start"] is not None:
            pos = torch.arange(hidden.shape[1], device=hidden.device)
            delta = delta * pos.ge(int(self.runtime["inference_start"])).view(1, -1, 1)
        self.runtime["last_delta"], self.runtime["last_delta_requires_grad"] = delta, bool(delta.requires_grad)
        self.runtime["last_hidden_shape"], self.runtime["last_ids_shape"] = tuple(hidden.shape), tuple(ids.shape)
        updated = hidden + delta.to(hidden.dtype)
        if sequence is None: return updated
        sequence = list(sequence); sequence[0] = updated
        return tuple(sequence) if isinstance(raw, tuple) else sequence


def _register_thinker_prehook(target_model, runtime):
    old = getattr(target_model, "_thai_engram_prehook_handle", None)
    if old is not None: old.remove()

    def _hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]): ids = args[0]
        runtime["input_ids"], runtime["attention_mask"], runtime["labels"] = ids, kwargs.get("attention_mask"), kwargs.get("labels")
        runtime["wrapper_calls"], runtime["last_delta"] = 0, None

    target_model._thai_engram_prehook_handle = target_model.thinker.register_forward_pre_hook(_hook, with_kwargs=True)


def assert_parameter_policy(target_model):
    bad = [(n, p) for n, p in target_model.named_parameters() if p.requires_grad and ".engram." not in n]
    good = [p for n, p in target_model.named_parameters() if p.requires_grad and ".engram." in n]
    if bad: raise AssertionError(f"Frozen Qwen policy violated: {bad[:2]}")
    if not good: raise AssertionError("No Engram parameters are trainable")
    return sum(p.numel() for p in good)


def install_engram(target_model, target_processor, config, runtime, device):
    layers = target_model.thinker.model.layers
    hidden_size = infer_hidden_size_for(target_model, layers)
    pad = target_processor.tokenizer.pad_token_id
    if pad is None:
        eos = target_processor.tokenizer.eos_token_id
        pad = int(eos[0] if isinstance(eos, (tuple, list)) else eos)
    _register_thinker_prehook(target_model, runtime)
    wrappers = {}
    for layer_id in config["layers"]:
        layer_id = int(layer_id)
        if not 0 <= layer_id < len(layers): raise ValueError(f"Invalid layer {layer_id}")
        base = layers[layer_id]
        if isinstance(base, EngramInjectedDecoderLayer): base = base.base_layer
        engram = ThaiNgramEngram(hidden_size, config["buckets"], config["memory_dim"],
                                 config["heads"], config["kernel"], pad,
                                 config["seed"] + layer_id * 10007).to(device)
        wrappers[layer_id] = EngramInjectedDecoderLayer(base, engram, layer_id, runtime).to(device)
        layers[layer_id] = wrappers[layer_id]
    for p in target_model.parameters(): p.requires_grad_(False)
    for wrapper in wrappers.values():
        for p in wrapper.engram.parameters(): p.requires_grad_(True)
    target_model._thai_engram_wrappers = wrappers
    return wrappers, hidden_size, assert_parameter_policy(target_model)


# --------------------------------------------------------------------------- #
# Checkpoint loading (cell 12)
# --------------------------------------------------------------------------- #
def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(payload, model_id, engram_config, hidden_size, wrappers):
    if payload.get("base_model") != model_id:
        raise ValueError(
            f"Checkpoint base model mismatch: {payload.get('base_model')} != {model_id}"
        )

    cfg = payload.get("config", {})
    expected = {
        "layers": engram_config["layers"],
        "buckets": engram_config["buckets"],
        "memory_dim": engram_config["memory_dim"],
        "heads": engram_config["heads"],
        "kernel": engram_config["kernel"],
        "hidden_size": hidden_size,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Checkpoint config mismatch for {key}: {cfg.get(key)!r} != {value!r}"
            )

    expected_states = {str(i) for i in wrappers}
    if set(payload.get("engram_state", {})) != expected_states:
        raise ValueError("Checkpoint Engram layer state does not match this architecture")


# --------------------------------------------------------------------------- #
# Dataset: the fixed 300-clip held-out set (cells 14, 15)
# --------------------------------------------------------------------------- #
THAI_CHAR = r"\u0E00-\u0E7F"


def normalize_thai_transcript(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    # Remove segmentation spaces only when both neighboring characters are Thai.
    text = re.sub(
        rf"(?<=[{THAI_CHAR}])\s+(?=[{THAI_CHAR}])",
        "",
        text,
    )
    return text.strip()


def get_record_text(ex) -> str:
    return normalize_thai_transcript(ex.get("text", ex.get("sentence", "")))


def open_stream(cfg: Config):
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
    # Avoid torchcodec dependency and decode bytes ourselves.
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print("Audio(decode=False) warning:", exc)
    return ds


def decode_audio(audio_obj, target_sr=16000):
    wav = None
    sr = None

    if isinstance(audio_obj, dict):
        if audio_obj.get("bytes") is not None:
            wav, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32")
        elif audio_obj.get("array") is not None:
            wav = np.asarray(audio_obj["array"], dtype=np.float32)
            sr = int(audio_obj.get("sampling_rate", target_sr))
        elif audio_obj.get("path"):
            wav, sr = sf.read(audio_obj["path"], dtype="float32")

    elif isinstance(audio_obj, (str, Path)):
        wav, sr = sf.read(str(audio_obj), dtype="float32")

    elif hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = samples.data
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        wav = np.asarray(data, dtype=np.float32)
        sr = int(samples.sample_rate)

    if wav is None:
        raise TypeError(f"Unsupported audio object: {type(audio_obj)}")

    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim > 1:
        # Detect [channels, samples] versus [samples, channels].
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[-1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=-1)

    wav = wav.reshape(-1)

    if int(sr) != target_sr:
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=target_sr)

    return wav.astype(np.float32)


def prepare_record(ex, cfg: Config):
    text = get_record_text(ex)
    if not text:
        return None

    wav = decode_audio(ex["audio"], target_sr=16000)
    duration = len(wav) / 16000.0

    if not (cfg.min_audio_sec <= duration <= cfg.max_audio_sec):
        return None

    return {"audio_array": wav, "text": text, "duration": duration}


def collect_eval_records(cfg: Config):
    """Collect the fixed held-out set from a fixed offset of the stream."""
    records = []
    source_seen = 0
    ds = open_stream(cfg).skip(cfg.eval_source_skip)

    for ex in ds:
        source_seen += 1
        try:
            rec = prepare_record(ex, cfg)
        except Exception:
            rec = None

        if rec is not None:
            records.append(rec)

        if len(records) >= cfg.eval_samples:
            break

    if len(records) < cfg.eval_samples:
        raise RuntimeError(
            f"Only collected {len(records)} records; wanted {cfg.eval_samples}"
        )

    print(
        "Final held-out:",
        len(records),
        "records from source offset",
        cfg.eval_source_skip,
        "(rows scanned:",
        source_seen,
        ")",
    )
    return records


# --------------------------------------------------------------------------- #
# Inference and metrics (cell 17)
# --------------------------------------------------------------------------- #
def build_prefix_text(processor, prompt="") -> str:
    messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    rendered = processor.apply_chat_template(
        [messages],
        add_generation_prompt=True,
        tokenize=False,
    )
    return rendered[0] if isinstance(rendered, (list, tuple)) else rendered


def clean_generated_asr_text(decoded, tokenizer) -> str:
    text = decoded
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    eos = tokenizer.eos_token or ""
    if eos:
        text = text.replace(eos, "")
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


THAI_CHAR_RE = r"\u0E00-\u0E7F"


def thai_cer_norm(text) -> str:
    return re.sub(r"\s+", "", normalize_thai_transcript(text).lower())


def corpus_cer(refs, preds) -> float:
    ref = "".join(thai_cer_norm(x) for x in refs)
    pred = "".join(thai_cer_norm(x) for x in preds)
    return cer(ref, pred)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
class Session:
    """Loaded model + optional Engram wrappers + inference helpers.

    Constructing a session performs the expensive work (model load, Engram
    install). Everything after that is cheap and repeatable.
    """

    def __init__(self, cfg: Config | None = None, engram: bool = False,
                 device: torch.device | None = None, verbose: bool = True):
        self.cfg = cfg or Config()
        seed_everything(self.cfg.seed)
        self.device = device or resolve_device()
        self.dtype = resolve_dtype(self.device)
        self.environment = describe_environment(self.device, self.dtype)

        self.asr, self.model, self.processor, self.tokenizer = load_asr_model(
            self.cfg, self.device, self.dtype
        )
        self.decoder_layers, self.decoder_name, self.hidden_size = probe_decoder(self.model)
        self.runtime = make_runtime()
        self.prefix_text = build_prefix_text(self.processor, "")
        self.engram_config = {
            "layers": list(self.cfg.engram_layers),
            "buckets": dict(self.cfg.engram_buckets),
            "memory_dim": self.cfg.engram_dim,
            "heads": self.cfg.engram_heads,
            "kernel": self.cfg.engram_kernel,
            "seed": self.cfg.seed,
        }
        self.wrappers: dict = {}
        self.trainable_params = 0
        if engram:
            self.cfg.require_checkpoints()
            self.wrappers, self.hidden_size, self.trainable_params = install_engram(
                self.model, self.processor, self.engram_config, self.runtime, self.device
            )
            for wrapper in self.wrappers.values():
                assert torch.count_nonzero(wrapper.engram.value_proj.weight).item() == 0
                assert torch.count_nonzero(wrapper.engram.short_conv.weight).item() == 0

        if verbose:
            print("PyTorch       :", self.environment["torch"])
            print("ROCm / HIP    :", self.environment["rocm_hip"])
            print("GPU           :", self.environment["gpu"])
            print("Qwen dtype    :", self.environment["model_dtype"])
            print("Decoder       :", self.decoder_name, len(self.decoder_layers),
                  "layers, hidden", self.hidden_size)
            print("Engram        :", "installed" if self.wrappers else "not installed",
                  f"({self.trainable_params:,} trainable params)" if self.wrappers else "")

    # -- checkpoints ------------------------------------------------------- #
    def load_checkpoint(self, step: int) -> dict:
        """Reload one checkpoint step's weights into the installed Engram wrappers."""
        if not self.wrappers:
            raise RuntimeError("Session was created with engram=False; nothing to load into.")
        path = self.cfg.checkpoint_paths()[step]
        payload = _load_checkpoint(path)
        validate_checkpoint(payload, self.cfg.model_id, self.engram_config,
                            self.hidden_size, self.wrappers)

        if int(payload.get("step", -1)) != step:
            raise ValueError(f"Expected step {step}, checkpoint reports {payload.get('step')}")

        for i, wrapper in self.wrappers.items():
            wrapper.engram.load_state_dict(payload["engram_state"][str(i)], strict=True)
            wrapper.engram.to(self.device)
            wrapper.engram.eval()

        assert_parameter_policy(self.model)
        return payload

    # -- data -------------------------------------------------------------- #
    def collect_records(self, limit: int | None = None):
        cfg = self.cfg
        if limit is not None:
            cfg = Config(**{**{f: getattr(self.cfg, f) for f in self.cfg.__dataclass_fields__},
                            "eval_samples": limit})
        return collect_eval_records(cfg)

    # -- inference --------------------------------------------------------- #
    @contextmanager
    def engram_mode(self, enabled: bool):
        old = self.runtime["enabled"]
        self.runtime["enabled"] = bool(enabled)
        try:
            yield
        finally:
            self.runtime["enabled"] = old

    @torch.no_grad()
    def transcribe(self, record, use_engram: bool = True) -> str:
        model = self.model
        model.eval()
        for wrapper in self.wrappers.values():
            wrapper.engram.eval()

        inputs = self.processor(
            text=[self.prefix_text],
            audio=[record["audio_array"]],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {
            k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }
        inputs = {
            k: (v.to(dtype=self.dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in inputs.items()
        }

        self.runtime["inference_start"] = int(inputs["input_ids"].shape[1] - 1)

        with self.engram_mode(use_engram):
            generated = model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=self.cfg.generation_use_cache,
            )

        seq = generated.sequences if hasattr(generated, "sequences") else generated
        ids = seq[:, inputs["input_ids"].shape[1]:]

        return clean_generated_asr_text(
            self.tokenizer.batch_decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0],
            self.tokenizer,
        )

    # -- reporting --------------------------------------------------------- #
    def model_tensor_storage_bytes(self) -> int:
        """Bytes held by unique live model parameters and buffers."""
        seen, total = set(), 0
        for module in self.model.modules():
            for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
                if tensor.data_ptr() in seen:
                    continue
                seen.add(tensor.data_ptr())
                total += tensor.numel() * tensor.element_size()
        return int(total)

    def describe(self) -> dict:
        return {
            "gpu": self.environment["gpu"],
            "rocm_hip": self.environment["rocm_hip"],
            "torch": self.environment["torch"],
            "dataset": self.cfg.dataset_name,
            "samples": self.cfg.eval_samples,
            "source_skip": self.cfg.eval_source_skip,
            "max_new_tokens": self.cfg.max_new_tokens,
            "use_cache": self.cfg.generation_use_cache,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def per_sample_cer(refs, preds):
    """Character error rate per sample, on the normalized Thai text."""
    return [cer(thai_cer_norm(r), thai_cer_norm(p)) for r, p in zip(refs, preds)]


def write_json(path: Path, payload) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_predictions_csv(path: Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def signatures_summary(model) -> str:
    """Human-readable signature dump used in provenance records."""
    return (
        f"thinker.forward: {inspect.signature(model.thinker.forward)}\n"
        f"model.generate:  {inspect.signature(model.generate)}"
    )

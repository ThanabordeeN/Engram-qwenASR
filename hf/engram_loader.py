"""Standalone loader for the Thai N-gram Engram on Qwen3-ASR-0.6B.

Self-contained: needs only ``torch``, ``transformers``, and ``qwen-asr``. The
code bodies are ported unchanged from ``src/qwen3asr_engram.py`` in
https://github.com/.../Engram-qwenASR so that this file can ship next to the
checkpoint on the Hub without the evaluation library.

Usage::

    from engram_loader import load_engram_model, transcribe

    model, processor, engram = load_engram_model()
    print(transcribe(model, processor, audio_array))       # Engram on
    print(transcribe(model, processor, audio_array, use_engram=False))  # baseline

The Engram is a *delta*: the Qwen3-ASR-0.6B base weights stay frozen and are
fetched from the Hub. Only ``engram_step_000750.pt`` carries trained parameters.
"""
from __future__ import annotations

import math
import re
import types

import numpy as np
import torch
import torch.nn as nn
from qwen_asr import Qwen3ASRModel
from transformers import GenerationConfig

__all__ = [
    "ThaiNgramEngram",
    "EngramInjectedDecoderLayer",
    "install_engram",
    "load_engram",
    "load_engram_model",
    "transcribe",
    "build_prefix_text",
    "clean_generated_asr_text",
]

BASE_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
HUB_REPO = "Thanabordee/Qwen3-ASR-0.6B-Thai-Engram"


# --------------------------------------------------------------------------- #
# Engram architecture — identical to src/qwen3asr_engram.py::ThaiNgramEngram
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
# Injection — identical to src/qwen3asr_engram.py::EngramInjectedDecoderLayer
# --------------------------------------------------------------------------- #
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


def make_runtime():
    return {"input_ids": None, "attention_mask": None, "labels": None, "enabled": True,
            "inference_start": None, "wrapper_calls": 0, "last_delta_requires_grad": None,
            "last_delta": None, "last_hidden_shape": None, "last_ids_shape": None}


def _register_thinker_prehook(target_model, runtime):
    old = getattr(target_model, "_thai_engram_prehook_handle", None)
    if old is not None: old.remove()

    def _hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]): ids = args[0]
        runtime["input_ids"], runtime["attention_mask"], runtime["labels"] = ids, kwargs.get("attention_mask"), kwargs.get("labels")
        runtime["wrapper_calls"], runtime["last_delta"] = 0, None

    target_model._thai_engram_prehook_handle = target_model.thinker.register_forward_pre_hook(_hook, with_kwargs=True)


def _patch_outer_forward(instance):
    """Route model(...) through thinker(...) so the pre-hook fires."""
    if getattr(instance, "_thai_engram_instance_forward_patched", False):
        return
    if not hasattr(instance, "thinker") or not hasattr(instance.thinker, "forward"):
        raise RuntimeError("Incompatible qwen-asr: thinker.forward is unavailable")

    def _forward(self, input_ids=None, attention_mask=None, input_features=None,
                 feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker(input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    instance.forward = types.MethodType(_forward, instance)
    instance._thai_engram_instance_forward_patched = True


def _infer_hidden_size(target_model, layers) -> int:
    for obj in (layers[0], getattr(layers[0], "self_attn", None), getattr(layers[0], "mlp", None)):
        for attr in ("hidden_size", "embed_dim"):
            if obj is not None and getattr(obj, attr, None) is not None:
                return int(getattr(obj, attr))
    cfg = getattr(target_model.thinker.config, "text_config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None) is not None:
        return int(cfg.hidden_size)
    raise RuntimeError("Could not infer Qwen hidden size")


def _probe_decoder(target_model):
    if not hasattr(target_model, "thinker") or not hasattr(target_model.thinker, "model"):
        raise RuntimeError("Qwen thinker/model path is unavailable")
    layers = target_model.thinker.model.layers
    if len(layers) < 20:
        raise RuntimeError(f"Unexpected text decoder depth: {len(layers)}")
    return layers, _infer_hidden_size(target_model, layers)


# --------------------------------------------------------------------------- #
# Install / load
# --------------------------------------------------------------------------- #
def install_engram(target_model, engram_config, runtime, device):
    """Wrap the configured decoder layers with a freshly initialised Engram."""
    layers, hidden_size = _probe_decoder(target_model)
    pad = getattr(target_model.thinker.config, "pad_token_id", None) or 0
    _register_thinker_prehook(target_model, runtime)
    wrappers = {}
    for layer_id in engram_config["layers"]:
        layer_id = int(layer_id)
        if not 0 <= layer_id < len(layers): raise ValueError(f"Invalid layer {layer_id}")
        base = layers[layer_id]
        if isinstance(base, EngramInjectedDecoderLayer): base = base.base_layer
        engram = ThaiNgramEngram(hidden_size, engram_config["buckets"], engram_config["memory_dim"],
                                 engram_config["heads"], engram_config["kernel"], pad,
                                 engram_config["seed"] + layer_id * 10007).to(device)
        wrappers[layer_id] = EngramInjectedDecoderLayer(base, engram, layer_id, runtime).to(device)
        layers[layer_id] = wrappers[layer_id]
    for p in target_model.parameters(): p.requires_grad_(False)
    target_model._thai_engram_wrappers = wrappers
    return wrappers, hidden_size


def load_engram(model, checkpoint_path, engram_config, runtime, device=None):
    """Install the Engram and load trained weights from a checkpoint file."""
    device = device or next(model.parameters()).device
    wrappers, hidden_size = install_engram(model, engram_config, runtime, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("base_model") != BASE_MODEL_ID:
        raise ValueError(f"Checkpoint base model mismatch: {payload.get('base_model')} != {BASE_MODEL_ID}")
    stored = payload.get("config", {})
    for key in ("layers", "buckets", "memory_dim", "heads", "kernel"):
        if stored.get(key) != engram_config.get(key):
            raise ValueError(f"Checkpoint config mismatch on {key!r}: {stored.get(key)!r} != {engram_config.get(key)!r}")
    for i, wrapper in wrappers.items():
        wrapper.engram.load_state_dict(payload["engram_state"][str(i)], strict=True)
        wrapper.engram.to(device)
        wrapper.engram.eval()
    return wrappers, payload


def load_engram_model(repo_id: str = HUB_REPO, base_model_id: str = BASE_MODEL_ID,
                      checkpoint_file: str = "engram_step_000750.pt",
                      device=None, dtype=None):
    """Load frozen Qwen3-ASR-0.6B + the trained Engram from the Hub.

    Returns ``(model, processor, runtime)``. Toggle the Engram at inference with
    ``runtime["enabled"] = False`` to reproduce the baseline.
    """
    from huggingface_hub import hf_hub_download

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)

    try:
        asr = Qwen3ASRModel.from_pretrained(base_model_id, dtype=dtype, device_map=None)
    except TypeError:
        asr = Qwen3ASRModel.from_pretrained(base_model_id, torch_dtype=dtype, device_map=None)
    model = asr.model.to(device)
    processor = asr.processor

    _patch_outer_forward(model)
    try:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    except Exception as exc:
        print("generation_config warning:", exc)

    path = hf_hub_download(repo_id, checkpoint_file)
    engram_config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    engram_config = {"layers": engram_config["layers"], "buckets": engram_config["buckets"],
                     "memory_dim": engram_config["memory_dim"], "heads": engram_config["heads"],
                     "kernel": engram_config["kernel"], "seed": engram_config["seed"]}

    runtime = make_runtime()
    load_engram(model, path, engram_config, runtime, device)
    model.eval()
    return model, processor, runtime


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_prefix_text(processor, prompt="") -> str:
    messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    rendered = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
    return rendered[0] if isinstance(rendered, (list, tuple)) else rendered


def clean_generated_asr_text(decoded, tokenizer) -> str:
    text = decoded
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    eos = tokenizer.eos_token or ""
    if eos:
        text = text.replace(eos, "")
    return re.sub(r"<\|[^>]+\|>", "", text).strip()


@torch.no_grad()
def transcribe(model, processor, audio_array, runtime=None, use_engram: bool = True,
               prefix_text: str | None = None, max_new_tokens: int = 192,
               use_cache: bool = False) -> str:
    """Greedy transcription of one 16 kHz mono waveform. Returns the raw text.

    ``audio_array`` is a 1-D float32 numpy array (what ``soundfile``/``librosa``
    return). Tensors are accepted and converted.
    """
    tokenizer = processor.tokenizer
    prefix_text = prefix_text if prefix_text is not None else build_prefix_text(processor, "")
    model.eval()

    if torch.is_tensor(audio_array):
        audio_array = audio_array.detach().float().cpu().numpy()
    audio_array = np.asarray(audio_array, dtype=np.float32).reshape(-1)

    inputs = processor(text=[prefix_text], audio=[audio_array], return_tensors="pt",
                       padding=True, truncation=False)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    inputs = {k: (v.to(dtype=dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in inputs.items()}

    old_enabled = runtime["enabled"] if runtime else None
    if runtime:
        runtime["enabled"] = bool(use_engram)
        runtime["inference_start"] = int(inputs["input_ids"].shape[1] - 1)
    try:
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   num_beams=1, use_cache=use_cache)
    finally:
        if runtime:
            runtime["enabled"] = old_enabled

    seq = generated.sequences if hasattr(generated, "sequences") else generated
    ids = seq[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)[0]
    return clean_generated_asr_text(decoded, tokenizer)


if __name__ == "__main__":
    # Smoke check: load from the Hub, run both paths, and prove the trained
    # delta is actually being applied (not still the zero-initialised one).
    import sys

    model, processor, runtime = load_engram_model()
    noise = np.random.default_rng(0).normal(0, 0.01, 16000 * 2).astype(np.float32)
    on = transcribe(model, processor, noise, runtime, use_engram=True)
    delta = runtime["last_delta"]          # captured before the baseline run resets it
    off = transcribe(model, processor, noise, runtime, use_engram=False)
    print("engram on :", repr(on))
    print("engram off:", repr(off))

    assert delta is not None, "engram path never ran"
    assert torch.isfinite(delta).all(), "delta has non-finite values"
    assert delta.abs().max().item() > 0, "delta is all zeros — checkpoint not loaded"
    print(f"delta max |v| = {delta.abs().max().item():.4f} over {tuple(delta.shape)}", file=sys.stderr)
    print("smoke check OK", file=sys.stderr)

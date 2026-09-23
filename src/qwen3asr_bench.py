"""Quantization, memory accounting, and the timed benchmark loop.

Shared by ``scripts/02``–``scripts/05``. The measurement definitions are the
ones used for the reported numbers; do not change them without re-running the
full 300-clip benchmark and updating the reports.

Definitions used in the reports:

* **model tensor storage** — bytes held by unique live model parameters and
  buffers, plus quantization side tensors (NF4 ``quant_state``, LLM.int8
  ``CB``/``SCB``). Deduplicated by underlying storage.
* **peak allocated VRAM** — ``torch.cuda.max_memory_allocated()`` after a
  ``reset_peak_memory_stats()`` that happens *after* one excluded warm-up clip.
* **latency** — wall-clock time per clip covering audio preprocessing and
  greedy generation, with ``torch.cuda.synchronize()`` after each clip.
"""
from __future__ import annotations

import gc
import math
import statistics
import time
from pathlib import Path

import torch
from bitsandbytes.nn import Embedding4bit, Linear4bit, Linear8bitLt
from jiwer import cer, process_characters
from tqdm.auto import tqdm

from qwen3asr_engram import Session, corpus_cer, results_subdir, thai_cer_norm

NF4_GROUP_SIZE = 32
NF4_QUANT_TYPE = "nf4"
NF4_COMPUTE_DTYPE = torch.bfloat16
LLM_INT8_THRESHOLD = 6.0
STEP = 750

PREDICTIONS_DIR = results_subdir("predictions")
SUMMARIES_DIR = results_subdir("summaries")
TABLES_DIR = results_subdir("tables")
FIGURES_DIR = results_subdir("figures")
PARTS_DIR = results_subdir("parts")


def load_cached_references() -> list[str]:
    """The frozen 300 references produced by scripts/01."""
    path = PREDICTIONS_DIR / "fixed300_base_predictions.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Run scripts/01_eval_scaling.py first; every other "
            "script checks its evaluation set against this cache."
        )
    return json.loads(path.read_text(encoding="utf-8"))["references"]


def assert_fixed_set(session: Session, records) -> None:
    """Fail loudly if the streamed records are not the reported 300-clip set."""
    refs = load_cached_references()
    if [r["text"] for r in records] != refs:
        raise RuntimeError(
            "The streamed evaluation set does not match the frozen 300-clip reference "
            "list. The upstream dataset revision may have changed."
        )


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #
def quantize_nf4(model: torch.nn.Module) -> dict:
    """bitsandbytes NF4 weight-only, group 32, BF16 compute, double quant off."""
    counts = {"linear_modules": 0, "embedding_modules": 0, "quantized_parameters": 0}

    def linear(src: torch.nn.Linear) -> Linear4bit:
        dst = Linear4bit(
            src.in_features, src.out_features, bias=src.bias is not None,
            compute_dtype=NF4_COMPUTE_DTYPE, compress_statistics=False,
            quant_type=NF4_QUANT_TYPE, device="cpu",
        )
        dst.weight.blocksize = NF4_GROUP_SIZE
        dst.load_state_dict({key: value.detach().cpu() for key, value in src.state_dict().items()})
        dst = dst.to(src.weight.device)
        counts["linear_modules"] += 1
        counts["quantized_parameters"] += src.weight.numel()
        return dst

    def embedding(src: torch.nn.Embedding) -> Embedding4bit:
        dst = Embedding4bit(
            src.num_embeddings, src.embedding_dim, dtype=src.weight.dtype,
            quant_type=NF4_QUANT_TYPE, device="cpu",
        )
        dst.weight.blocksize = NF4_GROUP_SIZE
        dst.padding_idx = src.padding_idx
        dst.max_norm = src.max_norm
        dst.norm_type = src.norm_type
        dst.scale_grad_by_freq = src.scale_grad_by_freq
        dst.sparse = src.sparse
        dst.load_state_dict({"weight": src.weight.detach().cpu()})
        dst = dst.to(src.weight.device)
        counts["embedding_modules"] += 1
        counts["quantized_parameters"] += src.weight.numel()
        return dst

    def visit(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, (Linear4bit, Embedding4bit)):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, linear(child))
            elif isinstance(child, torch.nn.Embedding):
                setattr(parent, name, embedding(child))
            else:
                visit(child)

    visit(model)
    gc.collect()
    torch.cuda.empty_cache()
    return counts


def quantize_llm_int8(model: torch.nn.Module) -> dict:
    """bitsandbytes LLM.int8 (Linear8bitLt); Linear layers only."""
    counts = {"linear_modules": 0}

    def replace(src: torch.nn.Linear) -> Linear8bitLt:
        dst = Linear8bitLt(
            src.in_features, src.out_features, bias=src.bias is not None,
            has_fp16_weights=False, threshold=LLM_INT8_THRESHOLD, device="cpu",
        )
        dst.load_state_dict({key: value.detach().cpu() for key, value in src.state_dict().items()})
        dst = dst.to(src.weight.device)
        counts["linear_modules"] += 1
        return dst

    def visit(parent: torch.nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, Linear8bitLt):
                continue
            if isinstance(child, torch.nn.Linear):
                setattr(parent, name, replace(child))
            else:
                visit(child)

    visit(model)
    gc.collect()
    torch.cuda.empty_cache()
    return counts


# --------------------------------------------------------------------------- #
# Memory accounting
# --------------------------------------------------------------------------- #
def model_tensor_bytes(model: torch.nn.Module) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0

    def add(tensor: object) -> None:
        nonlocal total
        if not torch.is_tensor(tensor):
            return
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        key = (str(tensor.device), storage.data_ptr(), size)
        if size and key not in seen:
            seen.add(key)
            total += size

    for tensor in list(model.parameters()) + list(model.buffers()):
        add(tensor)
    for module in model.modules():
        weight = getattr(module, "weight", None)
        # NF4 keeps absmax/code/offset outside the Parameter/buffer lists.
        state = getattr(weight, "quant_state", None)
        if state is not None:
            for name in ("absmax", "code", "offset"):
                add(getattr(state, name, None))
            nested = getattr(state, "state2", None)
            if nested is not None:
                for name in ("absmax", "code", "offset"):
                    add(getattr(nested, name, None))
        # Linear8bitLt keeps per-output-channel scales outside the Parameter/buffer lists.
        add(getattr(weight, "CB", None))
        add(getattr(weight, "SCB", None))
        module_state = getattr(module, "state", None)
        add(getattr(module_state, "CB", None))
        add(getattr(module_state, "SCB", None))
    return total


def char_prf(refs, preds) -> dict:
    """Pooled character-level precision/recall/F1 on the normalized text."""
    refs_norm = [thai_cer_norm(text) for text in refs]
    preds_norm = [thai_cer_norm(text) for text in preds]
    aligned = process_characters(refs_norm, preds_norm)
    hits = int(aligned.hits)
    fp = int(aligned.substitutions + aligned.insertions)
    fn = int(aligned.substitutions + aligned.deletions)
    precision = hits / (hits + fp) if hits + fp else 0.0
    recall = hits / (hits + fn) if hits + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": f1,
        "hits": hits,
        "substitutions": int(aligned.substitutions),
        "deletions": int(aligned.deletions),
        "insertions": int(aligned.insertions),
    }


# --------------------------------------------------------------------------- #
# Timed benchmark
# --------------------------------------------------------------------------- #
def benchmark_variant(session: Session, records, use_engram: bool, name: str,
                      tensor_bytes_after_warmup: bool) -> tuple[dict, list[str]]:
    """Time one already-loaded, already-quantized variant on the fixed clip set.

    ``tensor_bytes_after_warmup`` preserves the measurement point used by the
    original scripts: NF4 measures before the warm-up clip, LLM.int8 after it
    (the first forward materializes its scale tensors).
    """
    model = session.model
    if tensor_bytes_after_warmup:
        session.transcribe(records[0], use_engram)  # excluded warm-up
        torch.cuda.synchronize()
        allocated_after_warmup = torch.cuda.memory_allocated()
        tensor_bytes = model_tensor_bytes(model)
        allocated_after_load = None
    else:
        torch.cuda.synchronize()
        allocated_after_load = torch.cuda.memory_allocated()
        tensor_bytes = model_tensor_bytes(model)
        session.transcribe(records[0], use_engram)  # excluded warm-up
        torch.cuda.synchronize()
        allocated_after_warmup = torch.cuda.memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    predictions: list[str] = []
    start = time.perf_counter()
    for record in tqdm(records, desc=name):
        sample_start = time.perf_counter()
        predictions.append(session.transcribe(record, use_engram))
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - sample_start)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    ordered = sorted(latencies)
    refs = [record["text"] for record in records]
    metrics = {
        "samples": len(records),
        "corpus_cer": float(corpus_cer(refs, predictions)),
        "mean_sample_cer": float(statistics.mean(
            sample_cer(r, p) for r, p in zip(refs, predictions)
        )),
        **char_prf(refs, predictions),
        "model_tensor_storage_bytes": tensor_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "inference_total_seconds": elapsed,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": ordered[len(ordered) // 2],
        "latency_p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "sample_latency_seconds": latencies,
        "throughput_samples_per_second": len(records) / elapsed,
    }
    if allocated_after_warmup is not None:
        metrics["allocated_after_warmup_bytes"] = allocated_after_warmup
    if allocated_after_load is not None:
        metrics["allocated_after_load_bytes"] = allocated_after_load
    print(
        f"{name}: CER={metrics['corpus_cer']:.6f}; F1={metrics['char_f1']:.6f}; "
        f"mean={metrics['latency_mean_seconds']:.3f}s/clip; "
        f"peak VRAM={metrics['peak_allocated_bytes'] / 2**30:.3f} GiB; "
        f"tensor storage={tensor_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    return metrics, predictions


def sample_cer(ref, pred) -> float:
    return cer(thai_cer_norm(ref), thai_cer_norm(pred))


def free_session(session: Session) -> None:
    session.close()


def write_predictions_json(path: Path, references, predictions, metrics, variant: str,
                           start_index: int | None = None) -> None:
    import json

    payload = {"variant": variant, "metrics": metrics,
               "references": list(references), "predictions": list(predictions)}
    if start_index is not None:
        payload["start_index"] = start_index
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

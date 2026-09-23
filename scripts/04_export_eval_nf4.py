#!/usr/bin/env python3
"""NF4 export + reload validation.

Exports both NF4 variants as self-contained ``.pt`` bundles, then rebuilds fresh
architectures from the base model ID, reloads only the bundles, and checks that
the reloaded predictions are byte-identical to the in-memory ones from
``scripts/02``. This is the evidence that the exported files really store the
quantized weights and their scaling data.

Produces:

    exports/nf4/baseline_nf4_w4.pt
    exports/nf4/step750_engram_nf4_w4.pt
    exports/nf4/manifest.json
    exports/nf4/processor/, exports/nf4/model_config/
    results/predictions/{baseline_nf4_checkpoint_eval,step750_engram_nf4_checkpoint_eval}.{json,csv}
    results/summaries/nf4_exported_checkpoint_eval_summary.json

Runtime on an RX 9070 XT: roughly 40 minutes (four 300-clip passes plus export).

    .venv/bin/python scripts/04_export_eval_nf4.py
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
from bitsandbytes.nn import Embedding4bit, Linear4bit, Params4bit
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen3asr_bench import (  # noqa: E402
    NF4_COMPUTE_DTYPE,
    NF4_GROUP_SIZE,
    NF4_QUANT_TYPE,
    PREDICTIONS_DIR,
    STEP,
    SUMMARIES_DIR,
    load_cached_references,
    quantize_nf4,
    sample_cer,
)
from qwen3asr_engram import (  # noqa: E402
    PROJECT_ROOT,
    Config,
    Session,
    corpus_cer,
    write_json,
)

FORMAT = "qwen3-asr-bnb-nf4-inference-v1"
EXPORT_DIR = PROJECT_ROOT / "exports" / "nf4"


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def checkpoint_payload(session: Session, variant: str, engram: bool, counts: dict) -> dict:
    model = session.model
    modules = {}
    quantized_parameter_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, (Linear4bit, Embedding4bit)):
            continue
        if module.weight.quant_state is None:
            raise RuntimeError(f"NF4 weights missing quant state: {name}")
        key = f"{name}.weight" if name else "weight"
        quantized_parameter_names.add(key)
        item = {
            "kind": "linear4bit" if isinstance(module, Linear4bit) else "embedding4bit",
            # .data is intentional: copying Params4bit itself to CPU changes its packed layout.
            "packed_weight": module.weight.data.detach().cpu().clone(),
            "quant_state": cpu_tree(module.weight.quant_state.as_dict()),
        }
        if isinstance(module, Linear4bit):
            item["compute_dtype"] = str(module.compute_dtype)
        else:
            item["output_dtype"] = str(module.dtype)
        modules[name] = item

    named_parameters = dict(model.named_parameters())
    if not quantized_parameter_names.issubset(named_parameters):
        missing = sorted(quantized_parameter_names - set(named_parameters))
        raise RuntimeError(f"Packed parameter names missing from model: {missing[:5]}")
    dense_parameters = {
        name: param.detach().cpu().clone()
        for name, param in named_parameters.items()
        if name not in quantized_parameter_names
    }
    buffers = {name: tensor.detach().cpu().clone() for name, tensor in model.named_buffers()}
    engram_meta = {
        "enabled": engram,
        "checkpoint_step": STEP if engram else None,
        "config": session.engram_config if engram else None,
    }
    return {
        "metadata": {
            "format": FORMAT,
            "variant": variant,
            "base_model_id": session.cfg.model_id,
            "step": STEP if engram else None,
            "torch": str(torch.__version__),
            "bitsandbytes": bnb.__version__,
            "rocm_hip": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0),
            "quantization": {
                "method": "bitsandbytes NF4 weight-only",
                "bits": 4,
                "group_size": NF4_GROUP_SIZE,
                "compute_dtype": str(NF4_COMPUTE_DTYPE),
                "double_quant": False,
                "applied_to": "all nn.Linear and nn.Embedding weights in Qwen and Engram",
            },
            "engram": engram_meta,
            "converted_modules": counts,
            "reconstruct_with_base_model_id": True,
        },
        "quantized_modules": modules,
        "dense_parameters": dense_parameters,
        "buffers": buffers,
    }


def save_checkpoint(session: Session, variant: str, engram: bool, counts: dict) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / f"{variant}.pt"
    temp_path = path.with_suffix(".pt.tmp")
    torch.save(checkpoint_payload(session, variant, engram, counts), temp_path)
    temp_path.replace(path)
    print(f"Exported {path} ({path.stat().st_size / 2**30:.3f} GiB)", flush=True)
    return path


def load_checkpoint_into_model(session: Session, path: Path) -> dict:
    model = session.model
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    if meta["format"] != FORMAT or meta["bitsandbytes"] != bnb.__version__:
        raise RuntimeError("Checkpoint format or bitsandbytes version mismatch")
    quantized_modules = dict(model.named_modules())
    if set(payload["quantized_modules"]) != {
        name for name, module in quantized_modules.items()
        if isinstance(module, (Linear4bit, Embedding4bit))
    }:
        raise RuntimeError("Checkpoint quantized-module layout does not match rebuilt architecture")

    params = dict(model.named_parameters())
    quantized_names = {f"{name}.weight" for name in payload["quantized_modules"]}
    dense = payload["dense_parameters"]
    if set(params) != set(dense) | quantized_names:
        missing = sorted(set(params) - set(dense) - quantized_names)
        extra = sorted((set(dense) | quantized_names) - set(params))
        raise RuntimeError(f"Checkpoint parameter mismatch; missing={missing[:5]} extra={extra[:5]}")
    buffers = dict(model.named_buffers())
    if set(buffers) != set(payload["buffers"]):
        raise RuntimeError("Checkpoint buffer layout does not match rebuilt architecture")

    with torch.no_grad():
        for name, tensor in dense.items():
            param = params[name]
            if tuple(param.shape) != tuple(tensor.shape):
                raise RuntimeError(f"Parameter shape mismatch: {name}")
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        for name, tensor in payload["buffers"].items():
            buffer = buffers[name]
            if tuple(buffer.shape) != tuple(tensor.shape):
                raise RuntimeError(f"Buffer shape mismatch: {name}")
            buffer.copy_(tensor.to(device=buffer.device, dtype=buffer.dtype))

        for name, item in payload["quantized_modules"].items():
            module = quantized_modules[name]
            expected_kind = "linear4bit" if isinstance(module, Linear4bit) else "embedding4bit"
            if item["kind"] != expected_kind:
                raise RuntimeError(f"Quantized module type mismatch: {name}")
            restored = Params4bit.from_prequantized(
                data=item["packed_weight"],
                quantized_stats=item["quant_state"],
                requires_grad=False,
                device=module.weight.device,
            )
            # Avoid a module back-reference; forward recovers state from the Param4bit.
            module.weight = restored
            module.quant_state = restored.quant_state
            if isinstance(module, Embedding4bit):
                module.dtype = getattr(torch, item["output_dtype"].removeprefix("torch."))
            else:
                module.compute_dtype = getattr(torch, item["compute_dtype"].removeprefix("torch."))

    model.eval()
    return meta


def evaluate_checkpoint(session: Session, records, path: Path, variant: str,
                        expected: list[str]) -> dict:
    meta = load_checkpoint_into_model(session, path)
    predictions = [
        session.transcribe(record, meta["engram"]["enabled"])
        for record in tqdm(records, desc=f"Checkpoint eval {variant}")
    ]
    if len(predictions) != len(records):
        raise RuntimeError(f"Expected {len(records)} predictions, got {len(predictions)}")
    if predictions != expected:
        raise RuntimeError(f"Reloaded {variant} predictions differ from the saved in-memory evaluation")
    refs = [record["text"] for record in records]
    sample_cers = [float(sample_cer(ref, pred)) for ref, pred in zip(refs, predictions)]
    result = {
        "variant": variant,
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "samples": len(refs),
        "source_skip": session.cfg.eval_source_skip,
        "corpus_cer": float(corpus_cer(refs, predictions)),
        "mean_sample_cer": float(statistics.mean(sample_cers)),
        "predictions_match_previous_eval": True,
        "gpu": torch.cuda.get_device_name(0),
        "rocm_hip": torch.version.hip,
        "quantization": meta["quantization"],
    }
    stem = ("baseline_nf4_checkpoint_eval" if not meta["engram"]["enabled"]
            else "step750_engram_nf4_checkpoint_eval")
    write_json(PREDICTIONS_DIR / f"{stem}.json",
               {**result, "references": refs, "predictions": predictions})
    with (PREDICTIONS_DIR / f"{stem}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "reference", "prediction", "cer"])
        writer.writerows(
            (i, ref, pred, sample_cers[i]) for i, (ref, pred) in enumerate(zip(refs, predictions))
        )
    print(f"Reloaded {variant}: CER={result['corpus_cer']:.6f}; "
          f"checkpoint={path.stat().st_size / 2**30:.3f} GiB; predictions match=True", flush=True)
    return result


def read_predictions(name: str) -> list[str]:
    return json.loads((PREDICTIONS_DIR / name).read_text(encoding="utf-8"))["predictions"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "eval_300.json"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    refs = load_cached_references()

    setup = Session(cfg, engram=False, verbose=False)
    records = setup.collect_records()
    environment = setup.environment
    setup.close()
    if [r["text"] for r in records] != refs:
        raise RuntimeError("The current records do not match the fixed 300-sample evaluation set")

    expected_base = read_predictions("baseline_nf4_w4_predictions.json")
    expected_engram = read_predictions("step750_engram_nf4_w4_predictions.json")

    # --- export ----------------------------------------------------------------- #
    session = Session(cfg, engram=False)
    session.asr.processor.save_pretrained(EXPORT_DIR / "processor")
    session.model.config.save_pretrained(EXPORT_DIR / "model_config")
    base_counts = quantize_nf4(session.model)
    base_path = save_checkpoint(session, "baseline_nf4_w4", False, base_counts)
    session.close()
    gc.collect()
    torch.cuda.empty_cache()

    session = Session(cfg, engram=True)
    session.load_checkpoint(STEP)
    engram_counts = quantize_nf4(session.model)
    engram_path = save_checkpoint(session, "step750_engram_nf4_w4", True, engram_counts)
    session.close()
    gc.collect()
    torch.cuda.empty_cache()

    # --- rebuild from the base model ID and reload only the bundles -------------- #
    session = Session(cfg, engram=False)
    quantize_nf4(session.model)
    base_result = evaluate_checkpoint(session, records, base_path, "baseline_nf4_w4", expected_base)
    session.close()
    gc.collect()
    torch.cuda.empty_cache()

    session = Session(cfg, engram=True)
    quantize_nf4(session.model)
    engram_result = evaluate_checkpoint(session, records, engram_path, "step750_engram_nf4_w4",
                                        expected_engram)
    session.close()
    gc.collect()
    torch.cuda.empty_cache()

    summary = {
        "format": FORMAT,
        "dataset": {"samples": len(records), "source_skip": cfg.eval_source_skip},
        "gpu": environment["gpu"],
        "rocm_hip": environment["rocm_hip"],
        "torch": environment["torch"],
        "bitsandbytes": bnb.__version__,
        "quantization": {"method": "bitsandbytes NF4 weight-only", "bits": 4,
                         "group_size": NF4_GROUP_SIZE,
                         "compute_dtype": str(NF4_COMPUTE_DTYPE), "double_quant": False},
        "checkpoints": {
            "baseline_nf4_w4": {"path": str(base_path), "bytes": base_path.stat().st_size,
                                "eval": base_result},
            "step750_engram_nf4_w4": {"path": str(engram_path), "bytes": engram_path.stat().st_size,
                                      "eval": engram_result},
        },
    }
    write_json(SUMMARIES_DIR / "nf4_exported_checkpoint_eval_summary.json", summary)
    write_json(EXPORT_DIR / "manifest.json", {
        "format": FORMAT,
        "base_model_id": cfg.model_id,
        "quantization": summary["quantization"],
        "checkpoints": {name: {"path": f"{name}.pt", "bytes": info["bytes"]}
                        for name, info in summary["checkpoints"].items()},
        "processor_dir": "processor",
        "model_config_dir": "model_config",
        "load_note": ("The .pt bundle overrides all parameters/buffers after rebuilding the "
                      "architecture from the base model ID; load with scripts/04_export_eval_nf4.py."),
    }, )
    print("Saved evaluation summary:", SUMMARIES_DIR / "nf4_exported_checkpoint_eval_summary.json", flush=True)
    print("Saved checkpoint manifest:", EXPORT_DIR / "manifest.json", flush=True)


if __name__ == "__main__":
    main()

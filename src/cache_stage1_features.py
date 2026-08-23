#!/usr/bin/env python
"""Cache frozen Qwen3-VL hidden states and object embeddings for Stage-1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from mmdd_stage1.features import normalize_object_type


def _load_embedder_class(model_dir: Path):
    script = model_dir / "scripts" / "qwen3_vl_embedding.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing official Qwen embedding wrapper: {script}")
    spec = importlib.util.spec_from_file_location("_mmdd_qwen3_vl_embedding", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder


@torch.inference_mode()
def encode_inputs(embedder: Any, items: list[dict[str, Any]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return normalized final embeddings and unpooled valid hidden states."""

    conversations = [
        embedder.format_model_input(
            text=item.get("text"),
            image=item.get("image"),
            instruction=item.get("instruction"),
        )
        for item in items
    ]
    inputs = embedder._preprocess_inputs(conversations)
    inputs = {name: tensor.to(embedder.model.device) for name, tensor in inputs.items()}
    outputs = embedder.forward(inputs)
    hidden_states = outputs["last_hidden_state"]
    attention_mask = outputs["attention_mask"].bool()
    pooled = embedder._pooling_last(hidden_states, attention_mask)
    embeddings = F.normalize(pooled.float(), p=2, dim=-1)
    return [
        (embeddings[index].cpu(), hidden_states[index][attention_mask[index]].cpu())
        for index in range(hidden_states.shape[0])
    ]


def _resolve_image(record: dict[str, Any], input_dir: Path) -> str | None:
    value = record.get("image")
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = input_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return str(path)


def build_object_features(
    embedder: Any,
    record: dict[str, Any],
    *,
    input_dir: Path,
    instruction: str,
    table_part_batch_size: int,
    storage_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    object_type = normalize_object_type(str(record["object_type"]))
    item = {
        "text": record.get("text"),
        "image": _resolve_image(record, input_dir),
        "instruction": record.get("instruction", instruction),
    }
    embedding, hidden_states = encode_inputs(embedder, [item])[0]

    if object_type == "table":
        parts = record.get("table_parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(part, str) and part.strip() for part in parts):
            raise ValueError(
                f"{record.get('object_id')}: table_parts must contain schema text followed by example-row texts"
            )
        structural_tokens = []
        for start in range(0, len(parts), table_part_batch_size):
            batch = [
                {"text": part, "instruction": record.get("instruction", instruction)}
                for part in parts[start : start + table_part_batch_size]
            ]
            outputs = encode_inputs(embedder, batch)
            structural_tokens.extend(part_hidden[-1] for _, part_hidden in outputs)
        hidden_states = torch.stack(structural_tokens)

    return {
        "embedding": embedding.float(),
        "hidden_states": hidden_states.to(dtype=storage_dtype),
    }


def _completed_records(manifest: Path) -> dict[str, dict[str, Any]]:
    if not manifest.exists():
        return {}
    records = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["object_id"])] = record
    return records


def _source_fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> None:
    if args.table_part_batch_size <= 0:
        raise ValueError("--table-part-batch-size must be positive")
    input_path = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir)
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.jsonl"
    completed = _completed_records(manifest)

    model_dir = Path(args.model_dir).resolve()
    metadata = {
        "format_version": 1,
        "model_dir": str(model_dir),
        "dtype": args.dtype,
        "instruction": args.instruction,
        "table_part_batch_size": args.table_part_batch_size,
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(f"{metadata_path}: cache settings differ from this run")
    else:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending_ids = set()
    skipped = 0
    seen = set()
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in seen:
                raise ValueError(f"{input_path}:{line_number}: duplicate object_id {object_id!r}")
            seen.add(object_id)
            object_type = normalize_object_type(str(record["object_type"]))
            source_fingerprint = _source_fingerprint(record)
            if object_id in completed:
                completed_record = completed[object_id]
                completed_path = output_dir / completed_record["feature_path"]
                if not completed_path.is_file():
                    raise FileNotFoundError(f"Manifest references a missing feature file: {completed_path}")
                if completed_record.get("source_fingerprint") != source_fingerprint:
                    raise ValueError(f"{object_id}: input changed after this feature was cached")
                if completed_record.get("object_type") != object_type:
                    raise ValueError(f"{object_id}: object type changed after this feature was cached")
                skipped += 1
                continue
            pending_ids.add(object_id)

    if not pending_ids:
        print(json.dumps({"objects_written": 0, "objects_skipped": skipped, "output_dir": str(output_dir)}, indent=2))
        return

    embedder_class = _load_embedder_class(model_dir)
    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    embedder = embedder_class(model_name_or_path=str(model_dir), torch_dtype=torch_dtype)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    embedder.model.to(torch.device(device))
    embedder.model.eval()

    written = 0
    with input_path.open(encoding="utf-8") as source, manifest.open("a", encoding="utf-8") as manifest_handle:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id not in pending_ids:
                continue
            object_type = normalize_object_type(str(record["object_type"]))
            source_fingerprint = _source_fingerprint(record)
            payload = build_object_features(
                embedder,
                record,
                input_dir=input_path.parent,
                instruction=args.instruction,
                table_part_batch_size=args.table_part_batch_size,
                storage_dtype=torch_dtype,
            )
            name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
            relative_path = Path("objects") / name
            destination = output_dir / relative_path
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)
            manifest_record = {
                "object_id": object_id,
                "object_type": object_type,
                "feature_path": relative_path.as_posix(),
                "source_fingerprint": source_fingerprint,
            }
            manifest_handle.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")
            manifest_handle.flush()
            completed[object_id] = manifest_record
            written += 1
            print(json.dumps({"object_id": object_id, "written": written, "skipped": skipped}))

    print(json.dumps({"objects_written": written, "objects_skipped": skipped, "output_dir": str(output_dir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="hf_models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--table-part-batch-size", type=int, default=8)
    parser.add_argument(
        "--instruction",
        default="Represent this object for directed logical joinability discovery.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

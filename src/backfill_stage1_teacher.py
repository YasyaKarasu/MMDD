#!/usr/bin/env python3
"""Batch-generate missing Qwen Teacher hidden-state shards.

This is a narrow backfill helper for R24.  It writes only the Teacher tier;
the frozen retrieval feature cache remains the source of object identity and
embeddings.  The output directory is intentionally configurable so a caller
can place temporary shards in RAM and persist only the resulting Teacher score
cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from cache_stage1_features import (
    _load_embedder_class,
    _source_fingerprint,
    encode_inputs,
    embedding_instructions,
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_jsonl).resolve()
    output_dir = Path(args.output_dir).resolve()
    teacher_dir = output_dir / "teacher_objects"
    manifest_path = output_dir / "teacher_manifest.jsonl"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    records = _read_rows(input_path)
    if any(str(row.get("object_type")) not in {"text", "image"} for row in records):
        raise ValueError("Teacher backfill currently expects text/image records")
    completed: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        for row in _read_rows(manifest_path):
            object_id = str(row["object_id"])
            if object_id in completed:
                raise ValueError(f"Duplicate Teacher object in {manifest_path}: {object_id}")
            completed[object_id] = row
    pending = []
    skipped = 0
    for row in records:
        object_id = str(row["object_id"])
        source_fingerprint = _source_fingerprint(row)
        existing = completed.get(object_id)
        if existing is not None:
            expected = teacher_dir / Path(str(existing["teacher_feature_path"])).name
            if (
                existing.get("object_type") != str(row["object_type"])
                or existing.get("source_fingerprint") != source_fingerprint
                or not expected.is_file()
            ):
                raise ValueError(f"Existing Teacher record is not identical to input: {object_id}")
            skipped += 1
        else:
            pending.append((row, source_fingerprint))
    if not pending:
        result = {"status": "complete", "input": str(input_path), "output": str(output_dir), "written": 0, "skipped": skipped, "total": len(records)}
        print(json.dumps(result), flush=True)
        return result
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    embedder_class = _load_embedder_class(Path(args.model_dir).resolve())
    embedder = embedder_class(model_name_or_path=str(Path(args.model_dir).resolve()), torch_dtype=dtype)
    embedder.model.to(device)
    embedder.model.eval()
    written = 0
    started = time.monotonic()
    with manifest_path.open("a", encoding="utf-8") as manifest_handle:
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start : start + args.batch_size]
            rows = []
            for row, _fingerprint in chunk:
                object_type = str(row["object_type"])
                instruction, _row_instruction, _role = embedding_instructions(
                    row, object_type
                )
                rows.append(
                    {
                        "text": row.get("text"),
                        "image": row.get("image"),
                        "instruction": instruction,
                    }
                )
            with torch.inference_mode():
                outputs = encode_inputs(embedder, rows, include_hidden=True)
            for (row, source_fingerprint), (_embedding, hidden_states, _input_ids) in zip(chunk, outputs, strict=True):
                if hidden_states is None:
                    raise RuntimeError(f"No hidden states returned for {row['object_id']}")
                object_id = str(row["object_id"])
                name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
                destination = teacher_dir / name
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                torch.save({"hidden_states": hidden_states}, temporary)
                temporary.replace(destination)
                record = {
                    "object_id": object_id,
                    "object_type": str(row["object_type"]),
                    "teacher_feature_path": (Path("teacher_objects") / name).as_posix(),
                    "source_fingerprint": source_fingerprint,
                }
                manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest_handle.flush()
                completed[object_id] = record
                written += 1
            if written <= args.batch_size or written % (args.batch_size * 10) == 0 or written == len(pending):
                print(json.dumps({"status": "running", "written": written, "pending": len(pending), "elapsed_seconds": time.monotonic() - started}), flush=True)
    result = {"status": "complete", "input": str(input_path), "output": str(output_dir), "written": written, "skipped": skipped, "total": len(records), "elapsed_seconds": time.monotonic() - started}
    print(json.dumps(result), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

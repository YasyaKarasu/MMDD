#!/usr/bin/env python
"""Convert a legacy Stage-1 cache to the compact two-tier format without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from cache_stage1_features import TEACHER_MANIFEST, teacher_object_ids
from mmdd_stage1.features import FeatureStore, normalize_object_type
from mmdd_stage1.models import structural_table_pool


def _records(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate object_id {object_id!r}")
            records[object_id] = record
    return records


def _save(payload: dict[str, torch.Tensor], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _validate_completed_record(
    output_dir: Path,
    completed: dict[str, Any],
    source: dict[str, Any],
    *,
    path_field: str,
    directory: str,
    required_tensor: str,
) -> None:
    object_id = str(source["object_id"])
    expected_path = Path(directory) / (
        hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
    )
    relative_path = Path(str(completed[path_field]))
    if relative_path != expected_path:
        raise ValueError(
            f"{object_id}: cached {path_field} must be {expected_path.as_posix()}"
        )
    path = output_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Manifest references a missing feature file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{path}: feature payload is empty")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a feature mapping")
    if not isinstance(payload.get(required_tensor), torch.Tensor):
        raise ValueError(f"{path}: feature mapping has no tensor {required_tensor}")
    if normalize_object_type(str(completed["object_type"])) != normalize_object_type(
        str(source["object_type"])
    ):
        raise ValueError(f"{source['object_id']}: cached object type changed")
    source_fingerprint = source.get("source_fingerprint")
    if (
        source_fingerprint is not None
        and completed.get("source_fingerprint") != source_fingerprint
    ):
        raise ValueError(f"{source['object_id']}: cached source fingerprint changed")


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output feature directories must differ")

    source_records = _records(input_dir / "manifest.jsonl")
    if not source_records:
        raise ValueError(f"{input_dir}: feature manifest is empty")
    selected_teacher_ids = teacher_object_ids(
        [Path(value).resolve() for value in args.teacher_data],
        split=None if args.teacher_split == "all" else args.teacher_split,
    )
    missing = selected_teacher_ids - source_records.keys()
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise KeyError(f"Teacher data references objects absent from the source cache: {preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    object_dir = output_dir / "objects"
    teacher_dir = output_dir / "teacher_objects"
    object_dir.mkdir(exist_ok=True)
    if selected_teacher_ids:
        teacher_dir.mkdir(exist_ok=True)
    output_manifest = output_dir / "manifest.jsonl"
    output_teacher_manifest = output_dir / TEACHER_MANIFEST
    completed = _records(output_manifest)
    completed_teacher = _records(output_teacher_manifest)

    source_metadata_path = input_dir / "metadata.json"
    metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "format_version": 5,
            "feature_tiers": ["retrieval", "teacher"],
            "table_pooling": "prepooled_schema_rows",
        }
    )
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(f"{metadata_path}: cache settings differ from this conversion")
    else:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    store = FeatureStore.from_path(input_dir, cache_size=0)
    base_written = 0
    base_skipped = 0
    teacher_written = 0
    teacher_skipped = 0
    with (
        output_manifest.open("a", encoding="utf-8") as base_handle,
        output_teacher_manifest.open("a", encoding="utf-8") as teacher_handle,
    ):
        for object_id, source_record in source_records.items():
            completed_base = completed.get(object_id)
            if completed_base is not None:
                _validate_completed_record(
                    output_dir,
                    completed_base,
                    source_record,
                    path_field="feature_path",
                    directory="objects",
                    required_tensor="embedding",
                )
                base_skipped += 1
            needs_base = completed_base is None

            completed_hidden = completed_teacher.get(object_id)
            if object_id in selected_teacher_ids and completed_hidden is not None:
                _validate_completed_record(
                    output_dir,
                    completed_hidden,
                    source_record,
                    path_field="teacher_feature_path",
                    directory="teacher_objects",
                    required_tensor="hidden_states",
                )
                teacher_skipped += 1
            needs_teacher = object_id in selected_teacher_ids and completed_hidden is None
            if not needs_base and not needs_teacher:
                continue
            features = store.get(object_id, include_hidden=needs_teacher)
            name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
            if needs_base:
                base_payload = {"embedding": features.embedding}
                if features.row_embeddings is not None:
                    base_payload["row_embeddings"] = features.row_embeddings
                relative_path = Path("objects") / name
                _save(base_payload, output_dir / relative_path)
                record = {
                    "object_id": object_id,
                    "object_type": features.object_type,
                    "feature_path": relative_path.as_posix(),
                }
                if "source_fingerprint" in source_record:
                    record["source_fingerprint"] = source_record["source_fingerprint"]
                base_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                base_handle.flush()
                base_written += 1
            if needs_teacher:
                if features.hidden_states is None:
                    raise ValueError(
                        f"{object_id}: source cache has no Teacher hidden features"
                    )
                hidden_states = features.hidden_states
                token_groups = features.token_groups
                if features.object_type == "table":
                    if token_groups is not None:
                        hidden_states = structural_table_pool(hidden_states, token_groups)
                    token_groups = torch.arange(len(hidden_states), dtype=torch.long)
                teacher_payload = {"hidden_states": hidden_states}
                if token_groups is not None:
                    teacher_payload["token_groups"] = token_groups
                relative_path = Path("teacher_objects") / name
                _save(teacher_payload, output_dir / relative_path)
                record = {
                    "object_id": object_id,
                    "object_type": features.object_type,
                    "teacher_feature_path": relative_path.as_posix(),
                }
                if "source_fingerprint" in source_record:
                    record["source_fingerprint"] = source_record["source_fingerprint"]
                teacher_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                teacher_handle.flush()
                teacher_written += 1

    print(
        json.dumps(
            {
                "base_objects_written": base_written,
                "base_objects_skipped": base_skipped,
                "teacher_objects_written": teacher_written,
                "teacher_objects_skipped": teacher_skipped,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-data", required=True, nargs="+")
    parser.add_argument("--teacher-split", default="train")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

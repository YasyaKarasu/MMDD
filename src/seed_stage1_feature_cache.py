#!/usr/bin/env python
"""Seed a feature cache with hard-linked unchanged object types."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cache_stage1_features import (
    TEACHER_MANIFEST,
    _completed_records,
    _source_fingerprint,
)
from mmdd_stage1.features import normalize_object_type


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if not destination.samefile(source):
            raise ValueError(f"Existing feature is not linked to its source: {destination}")


def seed(
    input_jsonl: Path,
    source_cache: Path,
    output_dir: Path,
    *,
    object_types: set[str],
    teacher_object_types: set[str] | None = None,
    table_tokens_per_group: int = 1,
) -> dict[str, Any]:
    if table_tokens_per_group <= 0:
        raise ValueError("table_tokens_per_group must be positive")
    if teacher_object_types is None:
        teacher_object_types = object_types
    output_manifest = output_dir / "manifest.jsonl"
    output_teacher_manifest = output_dir / TEACHER_MANIFEST
    if output_manifest.exists() or output_teacher_manifest.exists():
        raise FileExistsError(f"Feature manifests already exist in {output_dir}")
    source_records = _completed_records(source_cache / "manifest.jsonl")
    source_teacher = _completed_records(source_cache / TEACHER_MANIFEST)
    selected_base: list[dict[str, Any]] = []
    selected_teacher: list[dict[str, Any]] = []
    seen = set()
    with input_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in seen:
                raise ValueError(f"{input_jsonl}:{line_number}: duplicate {object_id!r}")
            seen.add(object_id)
            object_type = normalize_object_type(str(record["object_type"]))
            needs_base = object_type in object_types
            needs_teacher = object_type in teacher_object_types
            if not needs_base and not needs_teacher:
                continue
            fingerprint = _source_fingerprint(record)
            if needs_base:
                source = source_records.get(object_id)
                if source is None:
                    raise KeyError(f"Source cache has no base feature for {object_id!r}")
                if source.get("source_fingerprint") != fingerprint:
                    raise ValueError(f"{object_id}: source record changed")
                relative_path = Path(str(source["feature_path"]))
                _link(source_cache / relative_path, output_dir / relative_path)
                selected_base.append(source)
            teacher = source_teacher.get(object_id) if needs_teacher else None
            if teacher is not None:
                if teacher.get("source_fingerprint") != fingerprint:
                    raise ValueError(f"{object_id}: Teacher source record changed")
                teacher_path = Path(str(teacher["teacher_feature_path"]))
                _link(source_cache / teacher_path, output_dir / teacher_path)
                selected_teacher.append(teacher)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = source_cache / "metadata.json"
    if not source_metadata.is_file():
        raise FileNotFoundError(source_metadata)
    metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
    if table_tokens_per_group > 1:
        metadata.update(
            {
                "format_version": 6,
                "table_pooling": "contiguous_mean_segments",
                "table_tokens_per_group": table_tokens_per_group,
            }
        )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in selected_base:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with output_teacher_manifest.open("w", encoding="utf-8") as handle:
        for record in selected_teacher:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "format_version": 1,
        "source_cache": str(source_cache.resolve()),
        "output_dir": str(output_dir.resolve()),
        "object_types": sorted(object_types),
        "teacher_object_types": sorted(teacher_object_types),
        "table_tokens_per_group": table_tokens_per_group,
        "base_objects_linked": len(selected_base),
        "teacher_objects_linked": len(selected_teacher),
    }
    (output_dir / "seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--object-types",
        nargs="+",
        choices=("table", "text", "image"),
        default=("text", "image"),
    )
    parser.add_argument(
        "--teacher-object-types",
        nargs="*",
        choices=("table", "text", "image"),
        help="Object types whose Teacher features should also be linked.",
    )
    parser.add_argument("--table-tokens-per-group", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed(
        args.input_jsonl.resolve(),
        args.source_cache.resolve(),
        args.output_dir.resolve(),
        object_types=set(args.object_types),
        teacher_object_types=(
            set(args.teacher_object_types)
            if args.teacher_object_types is not None
            else None
        ),
        table_tokens_per_group=args.table_tokens_per_group,
    )

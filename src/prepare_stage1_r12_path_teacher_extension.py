#!/usr/bin/env python
"""Stage only missing Teacher features for the frozen R12 C2 path pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_edge_examples, load_target_examples


def _records(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            value = str(record[key])
            if value in records:
                raise ValueError(f"{path}:{line_number}: duplicate {key} {value!r}")
            records[value] = record
    return records


def _referenced_ids(target_path: Path, edge_path: Path) -> set[str]:
    object_ids = set()
    for example in load_target_examples(target_path, split="train"):
        object_ids.add(example.query_id)
        for candidate in example.candidates:
            object_ids.add(candidate.target_id)
            object_ids.update(candidate.evidence_ids)
    for example in load_edge_examples(edge_path, split="train"):
        object_ids.add(example.query_id)
        object_ids.update(example.candidate_ids)
    return object_ids


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_path = args.target_lists.resolve()
    edge_path = args.edge_lists.resolve()
    base = args.base_features.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Teacher overlay already exists: {output}")

    referenced = _referenced_ids(target_path, edge_path)
    base_records = _records(base / "manifest.jsonl", "object_id")
    missing_base = referenced - base_records.keys()
    if missing_base:
        raise ValueError(
            "Frozen path candidates are absent from the retrieval cache: "
            f"{sorted(missing_base)[:10]}"
        )

    teacher_records: dict[str, tuple[dict[str, Any], Path]] = {}
    teacher_roots = [base, *(path.resolve() for path in args.teacher_features)]
    for teacher_root in teacher_roots:
        for object_id, record in _records(
            teacher_root / "teacher_manifest.jsonl", "object_id"
        ).items():
            if object_id in teacher_records and teacher_records[object_id][0] != record:
                raise ValueError(f"Conflicting Teacher feature record for {object_id!r}")
            teacher_records.setdefault(object_id, (record, teacher_root))

    missing_teacher = sorted(referenced - teacher_records.keys())
    canonical = _records(args.objects.resolve(), "object_id")
    missing_canonical = referenced - canonical.keys()
    if missing_canonical:
        raise ValueError(
            "Frozen path candidates are absent from the canonical object corpus: "
            f"{sorted(missing_canonical)[:10]}"
        )

    output.mkdir(parents=True)
    (output / "objects").mkdir()
    (output / "teacher_objects").mkdir()
    with (
        (output / "objects_to_encode.jsonl").open("w", encoding="utf-8") as objects,
        (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
        (output / "teacher_manifest.jsonl").open("w", encoding="utf-8") as teacher_manifest,
    ):
        for object_id in sorted(referenced):
            objects.write(json.dumps(canonical[object_id], ensure_ascii=False) + "\n")
            base_record = base_records[object_id]
            feature_path = Path(base_record["feature_path"])
            os.link(base / feature_path, output / feature_path)
            manifest.write(json.dumps(base_record, ensure_ascii=False) + "\n")
            if object_id in teacher_records:
                teacher_record, teacher_root = teacher_records[object_id]
                teacher_path = Path(teacher_record["teacher_feature_path"])
                destination = output / teacher_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(teacher_root / teacher_path, destination)
                teacher_manifest.write(
                    json.dumps(teacher_record, ensure_ascii=False) + "\n"
                )

    (output / "metadata.json").write_text(
        (base / "metadata.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output / "missing_teacher_object_ids.json", missing_teacher)
    payload = {
        "format_version": 1,
        "status": "staged",
        "referenced_objects": len(referenced),
        "existing_teacher_objects": len(referenced) - len(missing_teacher),
        "missing_teacher_objects": len(missing_teacher),
        "target_lists": str(target_path),
        "target_lists_sha256": checkpoint_fingerprint(target_path),
        "edge_lists": str(edge_path),
        "edge_lists_sha256": checkpoint_fingerprint(edge_path),
        "base_features": str(base),
        "teacher_features": [str(path) for path in teacher_roots[1:]],
        "objects": str(args.objects.resolve()),
        "objects_to_encode": str((output / "objects_to_encode.jsonl").resolve()),
        "missing_teacher_ids": str(
            (output / "missing_teacher_object_ids.json").resolve()
        ),
        "hardlink_policy": "reuse immutable retrieval and existing Teacher features",
    }
    write_json(output / "staging.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-lists", type=Path, required=True)
    parser.add_argument("--edge-lists", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--teacher-features", type=Path, nargs="*", default=[])
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())

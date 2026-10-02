#!/usr/bin/env python
"""Partition uncached Teacher objects across independent feature workers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cache_stage1_features import _completed_records, teacher_object_ids
from cache_stage1_features import normalize_object_type


def partition_pending_objects(
    objects_path: Path,
    selected_ids: set[str],
    completed_ids: set[str],
    num_shards: int,
) -> list[list[dict[str, str]]]:
    """Balance each modality independently while preserving source order."""

    pending_ids = selected_ids - completed_ids
    shards: list[list[dict[str, str]]] = [[] for _ in range(num_shards)]
    modality_counts = [Counter() for _ in range(num_shards)]
    found = set()
    with objects_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id not in pending_ids:
                continue
            object_type = normalize_object_type(str(record["object_type"]))
            shard = min(
                range(num_shards),
                key=lambda index: (
                    modality_counts[index][object_type],
                    len(shards[index]),
                    index,
                ),
            )
            shards[shard].append(
                {"object_id": object_id, "object_type": object_type}
            )
            modality_counts[shard][object_type] += 1
            found.add(object_id)

    missing = pending_ids - found
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise KeyError(f"Selected objects are absent from {objects_path}: {preview}")
    return shards


def run(args: argparse.Namespace) -> dict[str, Any]:
    objects_path = Path(args.objects).resolve()
    selected_ids = teacher_object_ids(
        [Path(value).resolve() for value in args.selected_data], split=None
    )
    completed_ids = set()
    for value in args.completed_manifests:
        completed_ids.update(_completed_records(Path(value).resolve()))
    shards = partition_pending_objects(
        objects_path,
        selected_ids,
        completed_ids,
        args.num_shards,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, records in enumerate(shards):
        output = output_dir / f"{args.prefix}_{index:02d}.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output)
        summaries.append(
            {
                "path": str(output.resolve()),
                "objects": len(records),
                "by_type": dict(sorted(Counter(r["object_type"] for r in records).items())),
            }
        )

    summary = {
        "selected_objects": len(selected_ids),
        "completed_objects": len(selected_ids & completed_ids),
        "pending_objects": sum(len(records) for records in shards),
        "shards": summaries,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--selected-data", nargs="+", required=True)
    parser.add_argument("--completed-manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="teacher_work_shard")
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())

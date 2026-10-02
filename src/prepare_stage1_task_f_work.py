#!/usr/bin/env python
"""Validate, merge, and balance enhanced Task-F table objects."""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from cache_stage1_features import normalize_object_type


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _table_work(record: dict[str, Any]) -> int:
    parts = [str(value) for value in record["table_parts"]]
    work = sum(len(value) for value in parts)
    if record.get("embedding_role") == "query" and len(parts) > 1:
        work += sum(len(parts[0]) + len(row) for row in parts[1:])
    return max(work, 1)


def prepare(
    lake_objects: list[Path],
    reference_objects: Path,
    output_dir: Path,
    *,
    num_shards: int,
    max_rows: int = 20,
    table_row_format: str = "named_cells",
    table_tokens_per_group: int = 1,
) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if max_rows <= 0 or table_tokens_per_group <= 0:
        raise ValueError("max_rows and table_tokens_per_group must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed_path = output_dir / "stage1_objects.jsonl"
    shard_paths = [output_dir / f"table_shard_{index:02d}.jsonl" for index in range(num_shards)]
    mixed_tmp = mixed_path.with_suffix(".jsonl.tmp")
    shard_tmps = [path.with_suffix(".jsonl.tmp") for path in shard_paths]
    counts = {"objects": 0, "tables": 0, "text": 0, "image": 0}
    shard_counts = [0] * num_shards
    shard_work = [0] * num_shards
    reference = _records(reference_objects)
    enhanced = itertools.chain.from_iterable(_records(path) for path in lake_objects)
    with ExitStack() as stack:
        mixed_handle = stack.enter_context(mixed_tmp.open("w", encoding="utf-8"))
        shard_handles = [
            stack.enter_context(path.open("w", encoding="utf-8"))
            for path in shard_tmps
        ]
        for enhanced_record, reference_record in itertools.zip_longest(enhanced, reference):
            if enhanced_record is None or reference_record is None:
                raise ValueError("Enhanced and reference object streams have different lengths")
            object_id = str(enhanced_record["object_id"])
            if object_id != str(reference_record["object_id"]):
                raise ValueError(f"Object order changed at {object_id!r}")
            object_type = normalize_object_type(str(enhanced_record["object_type"]))
            if object_type != normalize_object_type(str(reference_record["object_type"])):
                raise ValueError(f"{object_id}: object type changed")
            if object_type != "table" and enhanced_record != reference_record:
                raise ValueError(f"{object_id}: non-table input changed in Task F")
            line = json.dumps(enhanced_record, ensure_ascii=False) + "\n"
            mixed_handle.write(line)
            counts["objects"] += 1
            counts["tables" if object_type == "table" else object_type] += 1
            if object_type == "table":
                if len(enhanced_record.get("table_parts", [])) > max_rows + 1:
                    raise ValueError(
                        f"{object_id}: Task F retained more than {max_rows} rows"
                    )
                shard = min(range(num_shards), key=lambda index: (shard_work[index], index))
                shard_handles[shard].write(line)
                shard_counts[shard] += 1
                shard_work[shard] += _table_work(enhanced_record)
    mixed_tmp.replace(mixed_path)
    for temporary, destination in zip(shard_tmps, shard_paths):
        temporary.replace(destination)
    summary = {
        "format_version": 1,
        "max_rows": max_rows,
        "table_row_format": table_row_format,
        "table_tokens_per_group": table_tokens_per_group,
        "mixed_objects": str(mixed_path.resolve()),
        "counts": counts,
        "shards": [
            {
                "path": str(path.resolve()),
                "tables": shard_counts[index],
                "estimated_character_work": shard_work[index],
            }
            for index, path in enumerate(shard_paths)
        ],
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-objects", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-objects", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=20)
    parser.add_argument("--table-row-format", default="named_cells")
    parser.add_argument("--table-tokens-per-group", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        args.lake_objects,
        args.reference_objects,
        args.output_dir,
        num_shards=args.num_shards,
        max_rows=args.max_rows,
        table_row_format=args.table_row_format,
        table_tokens_per_group=args.table_tokens_per_group,
    )

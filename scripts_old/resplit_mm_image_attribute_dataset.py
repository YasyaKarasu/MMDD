#!/usr/bin/env python
"""Repartition an existing image-attribute dataset without splitting tables."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def select_table_groups(
    groups: dict[str, list[dict[str, Any]]],
    positive_target: int,
    negative_target: int,
) -> set[str]:
    target = (positive_target, negative_target)
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {
        (0, 0): None
    }

    for table_id in sorted(groups):
        records = groups[table_id]
        positive = sum(bool(row["extractable"]) for row in records)
        negative = len(records) - positive
        for state in list(parents):
            candidate = (state[0] + positive, state[1] + negative)
            if candidate[0] > positive_target or candidate[1] > negative_target:
                continue
            if candidate not in parents:
                parents[candidate] = (state, table_id)

    if target not in parents:
        raise ValueError(
            "cannot satisfy complete-table target "
            f"positive={positive_target}, negative={negative_target}"
        )

    selected: set[str] = set()
    state = target
    while state != (0, 0):
        parent = parents[state]
        if parent is None:
            raise RuntimeError("invalid subset-selection predecessor chain")
        state, table_id = parent
        selected.add(table_id)
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    positive = sum(bool(record["extractable"]) for record in records)
    negative = len(records) - positive
    return {"positive": positive, "negative": negative, "total": len(records)}


def resplit_dataset(
    dataset_dir: Path,
    targets: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    split_names = ("train", "val", "test")
    if set(targets) != set(split_names):
        raise ValueError("targets must define train, val, and test")

    samples = [
        record
        for split in split_names
        for record in _read_jsonl(dataset_dir / "samples" / f"{split}.jsonl")
    ]
    table_records = [
        record
        for split in split_names
        for record in _read_jsonl(dataset_dir / "tables" / f"{split}.jsonl")
    ]

    sample_ids = [str(record["sample_id"]) for record in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")
    table_map: dict[str, dict[str, Any]] = {}
    for table in table_records:
        table_id = str(table["source_table_id"])
        if table_id in table_map:
            raise ValueError(f"duplicate source table record: {table_id}")
        table_map[table_id] = table

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        table_id = str(sample["source_table_id"])
        if table_id not in table_map:
            raise ValueError(f"sample references missing source table: {table_id}")
        image_path = dataset_dir / str(sample["image_path"])
        if not image_path.is_file():
            raise ValueError(f"sample references missing image: {image_path}")
        groups[table_id].append(sample)

    if set(groups) != set(table_map):
        unused = sorted(set(table_map) - set(groups))
        raise ValueError(f"source tables without samples: {unused[:5]}")

    val_target = targets["val"]
    val_tables = select_table_groups(groups, val_target[0], val_target[1])
    remaining = {key: value for key, value in groups.items() if key not in val_tables}
    test_target = targets["test"]
    test_tables = select_table_groups(
        remaining, test_target[0], test_target[1]
    )
    train_tables = set(groups) - val_tables - test_tables
    table_sets = {
        "train": train_tables,
        "val": val_tables,
        "test": test_tables,
    }

    split_samples = {
        split: [
            sample
            for table_id in sorted(table_sets[split])
            for sample in groups[table_id]
        ]
        for split in split_names
    }
    counts = {split: _label_counts(split_samples[split]) for split in split_names}
    for split, (positive, negative) in targets.items():
        expected = {
            "positive": positive,
            "negative": negative,
            "total": positive + negative,
        }
        if counts[split] != expected:
            raise ValueError(
                f"{split} counts {counts[split]} do not match target {expected}"
            )

    before_sample_ids = set(sample_ids)
    after_sample_ids = {
        str(record["sample_id"])
        for records in split_samples.values()
        for record in records
    }
    if after_sample_ids != before_sample_ids:
        raise ValueError("sample ID set changed during resplit")
    if set().union(*table_sets.values()) != set(table_map):
        raise ValueError("table ID set changed during resplit")
    if (
        table_sets["train"] & table_sets["val"]
        or table_sets["train"] & table_sets["test"]
        or table_sets["val"] & table_sets["test"]
    ):
        raise ValueError("source table occurs in multiple splits")

    stats_path = dataset_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["selected"] = {split: counts[split]["total"] for split in split_names}
    stats["positives"] = sum(counts[split]["positive"] for split in split_names)
    stats["negatives"] = sum(counts[split]["negative"] for split in split_names)
    stats["splits"] = counts

    with tempfile.TemporaryDirectory(
        prefix=".resplit_", dir=dataset_dir
    ) as temporary_name:
        temporary = Path(temporary_name)
        for split in split_names:
            _write_jsonl(
                temporary / "samples" / f"{split}.jsonl",
                split_samples[split],
            )
            _write_jsonl(
                temporary / "tables" / f"{split}.jsonl",
                [table_map[table_id] for table_id in sorted(table_sets[split])],
            )
        stats_output = temporary / "stats.json"
        stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for split in split_names:
            os.replace(
                temporary / "samples" / f"{split}.jsonl",
                dataset_dir / "samples" / f"{split}.jsonl",
            )
            os.replace(
                temporary / "tables" / f"{split}.jsonl",
                dataset_dir / "tables" / f"{split}.jsonl",
            )
        os.replace(stats_output, stats_path)

    return {
        "counts": counts,
        "tables": {split: len(table_sets[split]) for split in split_names},
        "samples": len(samples),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--train_positive", type=int, default=1095)
    parser.add_argument("--train_negative", type=int, default=505)
    parser.add_argument("--val_positive", type=int, default=137)
    parser.add_argument("--val_negative", type=int, default=63)
    parser.add_argument("--test_positive", type=int, default=137)
    parser.add_argument("--test_negative", type=int, default=63)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = resplit_dataset(
        Path(args.dataset_dir),
        targets={
            "train": (args.train_positive, args.train_negative),
            "val": (args.val_positive, args.val_negative),
            "test": (args.test_positive, args.test_negative),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build initial Stage-1 object, edge, target/path, and corpus JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from mmdd_progress import progress

from mmdd_stage1.construction import (
    DEFAULT_MAX_CELL_CHARS,
    build_stage1_training_artifacts,
)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        total = len(records) if isinstance(records, list) else None
        for record in progress(
            records,
            total=total,
            desc=f"Write {path.name}",
            unit="record",
            leave=False,
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = build_stage1_training_artifacts(
        dataset_root,
        dataset_name=args.dataset_name or dataset_root.name,
        max_rows=args.max_rows,
        max_cell_chars=args.max_cell_chars,
        seed=args.seed,
    )
    counts = {
        name: _write_jsonl(output_dir / f"{name}.jsonl", records)
        for name, records in artifacts.items()
    }
    print(json.dumps({"output_dir": str(output_dir), "records": counts}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument(
        "--max-cell-chars",
        type=int,
        default=DEFAULT_MAX_CELL_CHARS,
        help="Maximum characters retained from each cleaned table cell.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

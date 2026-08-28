#!/usr/bin/env python
"""Derive a standalone image-to-attribute dataset from MMDD builder output."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from mmdd_progress import progress

from mmdd_dataset.utils import (
    clean_text,
    normalize,
    read_jsonl,
    stable_hash,
    values_match,
    write_json,
    write_jsonl,
)


def build(input_dir: Path, output_dir: Path, max_samples: int | None, seed: int) -> dict[str, Any]:
    tables = {
        table["source_table_id"]: table
        for table in read_jsonl(input_dir / "source_tables.jsonl")
    }
    assets = {
        asset["asset_id"]: asset
        for asset in read_jsonl(input_dir / "bridge_assets.jsonl")
        if asset["asset_type"] == "image" and Path(asset["local_path"]).is_file()
    }

    candidates: list[dict[str, Any]] = []
    for extraction in progress(
        read_jsonl(input_dir / "attribute_extractions.jsonl"),
        desc="Build image samples",
        unit="extraction",
    ):
        asset = assets.get(extraction["asset_id"])
        table = tables.get(extraction["source_table_id"])
        if asset is None or table is None:
            continue
        source_row = next(
            row for row in table["rows"] if row["row_id"] == extraction["source_row_id"]
        )
        cell = next(
            (
                cell
                for cell in source_row["cells"]
                if normalize(cell["column_name"]) == normalize(extraction["attribute_name"])
            ),
            None,
        )
        if cell is None or not clean_text(cell.get("text")):
            continue
        sample_id = "sample_" + stable_hash(
            extraction["source_table_id"],
            extraction["source_row_id"],
            extraction["asset_id"],
            extraction["attribute_name"],
        )
        candidates.append(
            {
                "sample_id": sample_id,
                "source_table_id": extraction["source_table_id"],
                "source_row_id": extraction["source_row_id"],
                "entity_id": extraction["entity_id"],
                "attribute_name": extraction["attribute_name"],
                "ground_truth_value": clean_text(cell["text"]),
                "extractable": values_match(extraction.get("value"), cell["text"]),
                "evidence": clean_text(extraction.get("evidence")),
                "asset_id": extraction["asset_id"],
            }
        )

    candidates.sort(key=lambda row: stable_hash(seed, row["sample_id"], length=40))
    if max_samples is not None:
        candidates = candidates[:max_samples]

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    copied: dict[str, str] = {}
    for sample in progress(candidates, desc="Copy sample images", unit="sample"):
        asset = assets[sample.pop("asset_id")]
        source_path = Path(asset["local_path"])
        relative_path = copied.get(asset["asset_id"])
        if relative_path is None:
            target = image_dir / f"{asset['asset_id']}{source_path.suffix or '.img'}"
            shutil.copy2(source_path, target)
            relative_path = target.relative_to(output_dir).as_posix()
            copied[asset["asset_id"]] = relative_path
        sample["image_path"] = relative_path

        draw = int(stable_hash(seed, sample["source_table_id"], length=8), 16) / 16**8
        split = "train" if draw < 0.8 else "dev" if draw < 0.9 else "test"
        by_split[split].append(sample)

    for split, records in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", records)
    stats = {
        "samples": len(candidates),
        "positive": sum(sample["extractable"] for sample in candidates),
        "negative": sum(not sample["extractable"] for sample in candidates),
        "splits": {split: len(records) for split, records in by_split.items()},
    }
    write_json(output_dir / "stats.json", stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    stats = build(Path(args.input_dir), Path(args.output_dir), args.max_samples, args.seed)
    print(f"built {stats['samples']} image-attribute samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

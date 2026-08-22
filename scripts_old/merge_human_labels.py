#!/usr/bin/env python
"""Merge filled human HITL labels without overwriting raw evidence paths."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from stage1_io import iter_jsonl, setup_logging, update_stage1_manifest, write_jsonl

ALLOWED = {2, 1, 0, -1}


def run(args: argparse.Namespace) -> None:
    setup_logging()
    stage1_dir = Path(args.stage1_dir)
    human_path = Path(args.human_labels)
    pool = {rec["path_id"]: rec for rec in iter_jsonl(stage1_dir / "hitl_pool.jsonl")}
    current: dict[str, dict[str, Any]] = {}
    existing_path = stage1_dir / "human_labeled_paths.jsonl"
    if existing_path.exists():
        current = {rec["path_id"]: rec for rec in iter_jsonl(existing_path)}
    for rec in iter_jsonl(human_path):
        path_id = rec.get("path_id")
        if path_id not in pool:
            raise ValueError(f"Unknown path_id in human labels: {path_id}")
        try:
            label = int(rec.get("label"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label for {path_id}: {rec.get('label')!r}") from exc
        if label not in ALLOWED:
            raise ValueError(f"Invalid label for {path_id}: {label}; expected one of {sorted(ALLOWED)}")
        merged = dict(pool[path_id])
        previous = current.get(path_id)
        if previous is not None:
            merged["previous_label"] = previous.get("human_label")
        merged["human_label"] = label
        merged["human_label_text"] = {2: "Direct Bridge", 1: "Indirect Bridge", 0: "Related Only", -1: "Wrong/Irrelevant"}[label]
        merged["label_source"] = "human"
        merged["annotator_notes"] = rec.get("annotator_notes", "")
        current[path_id] = merged
    records = sorted(current.values(), key=lambda item: item["path_id"])
    counts = Counter(str(item.get("human_label")) for item in records)
    out_count = write_jsonl(existing_path, records)
    update_stage1_manifest(stage1_dir, "human_labels", {"file": str(human_path), "records": out_count, "label_stats": dict(counts)})
    print(json.dumps({"records": out_count, "label_stats": dict(counts)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--human_labels", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

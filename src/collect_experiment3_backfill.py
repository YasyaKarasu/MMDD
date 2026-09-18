#!/usr/bin/env python
"""Collect the Teacher hidden-state objects Experiment 3 needs but lacks.

The train-side path bags reference evidence assets that were never encoded to
the Teacher tier (the frozen retrieval cache only ever needed their embeddings).
This writes the backfill input shards for the frozen encoder; it does not train
or retrieve anything, and it touches no frozen artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.features import FeatureStore  # noqa: E402
from run_experiment3_target_path import (  # noqa: E402
    FEATURES,
    build_schedule,
    teacher_feature_paths,
)

OBJECTS = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl"
OUT = ROOT / "work/witness_diagnostic_20260916/experiment3/teacher_backfill"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--candidates", default=None,
                        help="train-side retrieval the schedule is built from")
    args = parser.parse_args()

    from run_experiment3_target_path import TRAIN_RANKINGS
    entries = build_schedule(args.candidates or TRAIN_RANKINGS)["entries"]
    need: set[str] = set()
    for entry in entries:
        need.add(entry["query_id"])
        need.update(entry["p_train"])
        for bag in entry["bags"].values():
            need.update(bag)
    print(json.dumps({"event": "schedule_objects", "objects": len(need)}), flush=True)

    # Resolve coverage exactly the way training will: the FeatureStore's own
    # base Teacher manifest plus every configured supplement shard.
    store = FeatureStore.from_path(
        FEATURES, cache_size=1000, teacher_paths=teacher_feature_paths()
    )
    missing_evidence = sorted(
        value for value in need if value.startswith("asset_") and not store.has_teacher_features(value)
    )
    missing_other = sorted(
        value for value in need if not value.startswith("asset_") and not store.has_teacher_features(value)
    )
    print(json.dumps({"event": "missing", "evidence": len(missing_evidence),
                      "non_evidence": len(missing_other),
                      "sample_non_evidence": missing_other[:5]}), flush=True)
    if missing_other:
        raise SystemExit(
            "non-evidence objects lack Teacher features; the backfill runner only "
            "handles text/image assets, so this must be resolved upstream"
        )

    records = []
    for line in OBJECTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        object_id = str(record.get("object_id"))
        if object_id not in missing_evidence:
            continue
        if record.get("object_type") == "text" and record.get("text"):
            records.append({"object_id": object_id, "object_type": "text", "text": record["text"]})
        elif record.get("object_type") == "image" and record.get("image"):
            records.append({"object_id": object_id, "object_type": "image", "image": record["image"]})
    found = {record["object_id"] for record in records}
    unresolved = sorted(set(missing_evidence) - found)
    print(json.dumps({"event": "records", "resolved": len(records),
                      "unresolved": len(unresolved), "sample": unresolved[:5]}), flush=True)

    records.sort(key=lambda item: item["object_id"])
    for shard in range(args.shards):
        chunk = records[shard :: args.shards]
        directory = OUT / f"gpu{shard}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "inputs.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in chunk:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"event": "shard", "gpu": shard, "records": len(chunk),
                          "path": str(path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Audit real R12 candidate masks and the R11-to-R12 label corrections."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from audit_stage1_r11_protocol import _expansion_epoch, _label_summary
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_edge_examples


def run(root: Path, output_root: Path) -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    old = root / "work/stage1_optimization_r11_20260908/taskA_protocol/supervision"
    new = output_root / "taskA_correctness/supervision"
    manifest = json.loads((new / "manifest.json").read_text())
    buckets = {}
    for bucket in manifest["outputs"]:
        before = [json.loads(line) for line in (old / f"edge_lists.{bucket}.jsonl").open()]
        after = [json.loads(line) for line in (new / f"edge_lists.{bucket}.jsonl").open()]
        assert len(before) == len(after)
        changed_negatives = 0
        candidate_mismatches = []
        for index, (left, right) in enumerate(zip(before, after)):
            for key in ("query_id", "source_type", "destination_type", "candidate_ids", "positive_ids", "positive_id", "split"):
                if left[key] != right[key]:
                    candidate_mismatches.append({"list_index": index, "field": key})
            changed_negatives += sum(a == 0 and b is None for a, b in zip(left["confirmed_labels"], right["confirmed_labels"]))
        split = "dev" if bucket == "dev" else "test" if bucket == "r10_test_regression" else "train"
        examples = load_edge_examples(new / f"edge_lists.{bucket}.jsonl", split=split)
        buckets[bucket] = {
            "lists": len(examples), "candidate_mismatches": candidate_mismatches,
            "removed_unjustified_confirmed_negative_positions": changed_negatives,
            "before_dataset_names": sorted({row["dataset"] for row in before}),
            "after_dataset_names": sorted({row["dataset"] for row in after}),
            "labels": _label_summary(examples),
            "source_sha256": checkpoint_fingerprint(new / f"edge_lists.{bucket}.jsonl"),
        }
    train = load_edge_examples(new / "edge_lists.train_fit.jsonl", split="train")
    epochs = [_expansion_epoch(train, seed=13, epoch=epoch, batch_size=64, cap=256) for epoch in range(2)]
    status = "pass" if (all(row["fixed_known_positive_as_negative"] == 0 for row in epochs)
                        and not any(row["candidate_mismatches"] for row in buckets.values())) else "fail"
    calibration_counts = {
        bucket: row["labels"]["by_relation"] for bucket, row in buckets.items()
        if bucket in {"cal_fit", "cal_check"}
    }
    eligible = [relation for relation, counts in calibration_counts["cal_fit"].items()
                if counts.get("confirmed_positive", 0) and counts.get("confirmed_negative", 0)]
    payload = {
        "status": status, "buckets": buckets, "actual_mask_replay": {
            "seed": 13, "batch_size": 64, "cap": 256, "epochs": epochs,
            "legacy_total": sum(row["legacy_known_positive_as_negative"] for row in epochs),
            "fixed_total": sum(row["fixed_known_positive_as_negative"] for row in epochs),
            "implementation": "Actual scorer expansion and returned candidate/positive masks; real IDs, scalar synthetic scores",
        },
        "d3": {"confirmed_class_counts": calibration_counts,
               "eligible_relations": eligible, "new_probability_calibration_started": False,
               "status": "not_triggered_no_confirmed_negative_classes" if not eligible else "pending",
               "historical_interpretation": "Constructed-negative mapping contrast, not confirmed-negative probability calibration",
               "historical_artifact": str(root / "work/stage1_optimization_r11_20260908/taskD_controls/c2_d3_affine/metrics.json")},
        "elapsed_seconds": time.monotonic() - started,
        "code_fingerprints": {str(path.relative_to(root)): checkpoint_fingerprint(path) for path in (
            Path(__file__), root / "src/audit_stage1_r11_protocol.py", root / "src/mmdd_stage1/scoring.py",
            root / "src/mmdd_stage1/construction.py",
        )},
    }
    output = output_root / "taskA_correctness/supervision_audit.json"
    write_json(output, payload)
    with (output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "A1 and A4.4", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "command": [sys.executable, *sys.argv], "output": str(output),
                                 "elapsed_seconds": payload["elapsed_seconds"], "status": status}) + "\n")
    print(json.dumps({"status": status, "mask_replay": payload["actual_mask_replay"], "d3_status": payload["d3"]["status"]}, indent=2))
    if status != "pass":
        raise RuntimeError("The supervision audit failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output_root)

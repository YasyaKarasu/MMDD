#!/usr/bin/env python
"""Audit frozen R13 hard-candidate labels, staleness, and source composition."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from run_stage1_r13 import _output_root, _paths, freeze_plan


def _family(value: str) -> str:
    for prefix in (
        "target_",
        "dl_raw_st_table_",
        "asset_text_",
        "asset_img_",
    ):
        if value.startswith(prefix):
            return prefix.removesuffix("_")
    return "other"


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    schedule = _paths(args.root)["schedule"]
    relations: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "lists": 0,
            "positive_counts": Counter(),
            "candidate_widths": [],
            "candidate_families": Counter(),
            "confirmed_positive_labels": 0,
            "confirmed_negative_labels": 0,
            "unknown_labels": 0,
        }
    )
    global_candidates = Counter()
    steps = []
    with gzip.open(schedule, "rt", encoding="utf-8") as handle:
        for line in handle:
            batch = json.loads(line)
            steps.append(int(batch["step"]))
            for example in batch["examples"]:
                relation = f"{example['source_type']}_to_{example['destination_type']}"
                row = relations[relation]
                row["lists"] += 1
                row["positive_counts"][len(set(example["positive_ids"]))] += 1
                row["candidate_widths"].append(len(example["candidate_ids"]))
                row["candidate_families"].update(
                    _family(value) for value in example["candidate_ids"]
                )
                global_candidates.update(example["candidate_ids"])
                labels = example["confirmed_labels"]
                row["confirmed_positive_labels"] += sum(value == 1 for value in labels)
                row["confirmed_negative_labels"] += sum(value == 0 for value in labels)
                row["unknown_labels"] += sum(value is None for value in labels)
    relation_summary = {}
    for relation, row in sorted(relations.items()):
        widths = row.pop("candidate_widths")
        row["positive_counts"] = dict(sorted(row["positive_counts"].items()))
        row["candidate_families"] = dict(sorted(row["candidate_families"].items()))
        row["multi_positive_lists"] = sum(
            count for positives, count in row["positive_counts"].items() if positives > 1
        )
        row["candidate_width"] = {
            "min": min(widths),
            "median": statistics.median(widths),
            "max": max(widths),
            "mean": statistics.fmean(widths),
        }
        relation_summary[relation] = row
    candidate_quality = (
        _paths(args.root)["r12"]
        / "taskC_training/candidate_quality/summary.json"
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "schedule": {
            "path": str(schedule.resolve()),
            "sha256": checkpoint_fingerprint(schedule),
            "steps": len(steps),
            "step_min": min(steps),
            "step_max": max(steps),
        },
        "staleness": {
            "candidate_refreshes_during_training": 0,
            "mined_before_optimizer_step": 0,
            "maximum_frozen_schedule_step": max(steps),
            "C_uses_prefix_steps": 178,
            "B1_uses_full_steps": 356,
            "interpretation": (
                "Candidate IDs are intentionally frozen; this records exposure age "
                "rather than claiming a measured causal staleness effect."
            ),
        },
        "by_relation": relation_summary,
        "global_candidate_hubs": {
            "unique_ids": len(global_candidates),
            "occurrences": sum(global_candidates.values()),
            "maximum_frequency": max(global_candidates.values()),
            "top20": global_candidates.most_common(20),
        },
        "difficulty_strata_reference": {
            "definition": "base schedule versus Raw-mined hard-candidates schedule",
            "path": str(candidate_quality.resolve()),
            "sha256": checkpoint_fingerprint(candidate_quality),
            "note": (
                "The referenced exhaustive audit reports Raw/PCA/Teacher relation-macro "
                "and list-micro R@1 on both strata."
            ),
        },
        "source_bias_boundary": (
            "Candidate ID-family composition and per-ID hubs are descriptive. No source "
            "centering or candidate refresh is used for selection."
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    target = _output_root(args.root) / "taskB_diagnostics_and_kd/candidate_audit.json"
    write_json(target, payload)
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args())

#!/usr/bin/env python
"""Select the frozen R12 C2 checkpoint and bind its reusable ANN index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


ARMS = ("path_only", "path_edge")
STEPS = (0, 178, 356)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    retrieval = metrics["retrieval"]
    funnel = retrieval["evidence_funnel"]
    return {
        "valid_pool_count": int(funnel["valid_pool_count"]),
        "row_b": float(funnel["row_b"]),
        "valid_b_count": int(funnel["valid_b_count"]),
        "direct_recall@10": float(retrieval["direct"]["recall@10"]),
    }


def _order(row: dict[str, Any]) -> tuple[int, float, int, float, int]:
    metrics = row["summary"]
    return (
        metrics["valid_pool_count"],
        metrics["row_b"],
        metrics["valid_b_count"],
        metrics["direct_recall@10"],
        -row["step"],
    )


def run(output_root: Path) -> dict[str, Any]:
    task_c = output_root / "taskC_training"
    selected_dir = task_c / "selected_c2_seed13"
    selection_path = selected_dir / "selection.json"
    comparison_path = selected_dir / "comparison.json"
    if selection_path.exists() or comparison_path.exists():
        raise FileExistsError("R12 C2 selection has already been finalized")

    records = []
    for arm in ARMS:
        for step in STEPS:
            path = (
                task_c
                / "full_lake_evaluations"
                / f"r12_c2_{arm}"
                / f"step{step}"
                / "metrics.json"
            )
            metrics = _load(path)
            records.append(
                {
                    "arm": arm,
                    "step": step,
                    "metrics": str(path.resolve()),
                    "metrics_sha256": checkpoint_fingerprint(path),
                    "checkpoint": metrics["checkpoint"],
                    "checkpoint_sha256": metrics["checkpoint_sha256"],
                    "summary": _summary(metrics),
                }
            )

    ordered = sorted(records, key=_order, reverse=True)
    if _order(ordered[0]) == _order(ordered[1]):
        raise ValueError("Frozen checkpoint order does not resolve the C2 winner")
    selected = ordered[0]
    checkpoint = Path(selected["checkpoint"])
    if checkpoint_fingerprint(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint fingerprint differs from evaluation")

    index = selected_dir / "index"
    index_manifest_path = index / "manifest.json"
    index_manifest = _load(index_manifest_path)
    if index_manifest.get("student_checkpoint_sha256") != selected["checkpoint_sha256"]:
        raise ValueError("Selected index was built from a different checkpoint")

    source_selection_path = (
        task_c
        / f"c2_{selected['arm']}_seed13"
        / "student_path.pt.selection.json"
    )
    selection = _load(source_selection_path)
    if selection.get("corpus_sha256") != index_manifest.get("corpus_sha256"):
        raise ValueError("Selected index and source selection use different corpora")
    selected_metrics = _load(Path(selected["metrics"]))
    selection.update(
        {
            "best_checkpoint": str(checkpoint.resolve()),
            "best_checkpoint_sha256": selected["checkpoint_sha256"],
            "best_epoch": 0,
            "best_index": str(index.resolve()),
            "latest_index": str(index.resolve()),
            "best_metrics": selected_metrics["retrieval"],
            "primary_metric": "ValidPool_count",
            "selection_order": [
                "ValidPool_count:max",
                "RowB:max",
                "ValidB_count:max",
                "direct_R@10:max",
                "earlier_step",
            ],
            "r12_selected_arm": selected["arm"],
            "r12_selected_step": selected["step"],
            "r12_selected_from": ["path_only", "path_edge"],
            "r12_selection_metrics": selected["summary"],
            "r12_selection_metrics_path": selected["metrics"],
            "r12_selection_metrics_sha256": selected["metrics_sha256"],
            "r12_index_manifest_sha256": checkpoint_fingerprint(
                index_manifest_path
            ),
            "r12_training_gain_claimed": selected["step"] > 0,
        }
    )
    write_json(selection_path, selection)

    by_arm_step = {(row["arm"], row["step"]): row for row in records}
    comparison = {
        "format_version": 1,
        "checkpoint_order": selection["selection_order"],
        "selected": selected,
        "selected_checkpoint_note": (
            "Stage initialization selected; no C2 path-training gain is claimed"
            if selected["step"] == 0
            else "Post-initialization C2 checkpoint selected"
        ),
        "records": records,
        "path_edge_minus_path_only": {
            str(step): {
                key: (
                    by_arm_step[("path_edge", step)]["summary"][key]
                    - by_arm_step[("path_only", step)]["summary"][key]
                )
                for key in selected["summary"]
            }
            for step in STEPS
        },
        "selection": str(selection_path.resolve()),
        "selection_sha256": checkpoint_fingerprint(selection_path),
        "index_manifest": str(index_manifest_path.resolve()),
        "index_manifest_sha256": checkpoint_fingerprint(index_manifest_path),
    }
    write_json(comparison_path, comparison)
    print(json.dumps({"selected": selected, "selection": str(selection_path)}, indent=2))
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())

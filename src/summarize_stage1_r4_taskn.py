#!/usr/bin/env python
"""Summarize the conditional EntiTables hard-negative mining round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


def _read_system(path: Path, system: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["systems"][system]


def _view(result: dict[str, Any], channel: str | None = None) -> dict[str, Any]:
    values = result["metrics"]["per_query"]
    return values[channel] if channel else values


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    recall_ks: tuple[int, ...],
    iterations: int,
    seed: int,
    candidate_channel: str | None = None,
    reference_channel: str | None = None,
) -> dict[str, Any]:
    candidate_values = _view(candidate, candidate_channel)
    reference_values = _view(reference, reference_channel)
    return {
        f"recall@{k}": paired_bootstrap_delta(
            candidate_values[f"recall@{k}"],
            reference_values[f"recall@{k}"],
            iterations=iterations,
            seed=seed,
        )
        for k in recall_ks
    }


def _ci(value: dict[str, Any]) -> str:
    return (
        f"{value['mean']:+.2%} "
        f"[{value['ci_low']:+.2%}, {value['ci_high']:+.2%}]"
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _markdown(payload: dict[str, Any]) -> str:
    if payload.get("skipped"):
        return "# Task N: hard-negative mining\n\nSkipped: " + payload["reason"] + "\n"
    ks = payload["parameters"]["recall_ks"]
    rows = [
        "# Task N: EntiTables ensemble-scored hard-negative mining",
        "",
        "Candidates come from the Task-K selected Student index and target scores "
        "use the alpha=0.7 raw/Teacher z-score ensemble.",
        "",
        "| System | " + " | ".join(f"R@{k}" for k in ks) + " | R@10 delta / 95% CI |",
        "| --- | " + " | ".join("---:" for _ in ks) + " | --- |",
    ]
    for name in ("before_student", "mined_student", "before_ensemble", "mined_ensemble"):
        result = payload["systems"][name]
        metrics = result["metrics"]
        rows.append(
            f"| {result['label']} | "
            + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in ks)
            + " | "
            + (_ci(result["comparison"]["recall@10"]) if "comparison" in result else "reference")
            + " |"
        )
    rows.extend(
        [
            "",
            f"Selected mining epoch: {payload['mining']['best_epoch']}; "
            f"gate fallback: {payload['mining']['gate_unsatisfied']}.",
            "",
            "Decision: "
            + ("use the mined checkpoint." if payload["use_mined_checkpoint"] else "retain the Task-K checkpoint."),
        ]
    )
    return "\n".join(rows) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skipped_reason:
        payload = {"format_version": 1, "skipped": True, "reason": args.skipped_reason}
    else:
        taskl = json.loads(args.taskl_metrics.read_text(encoding="utf-8"))
        history = json.loads(args.history.read_text(encoding="utf-8"))
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        recall_ks = tuple(args.recall_ks)
        systems = {
            "before_student": taskl["lakes"]["entitables"]["systems"]["student"],
            "mined_student": _read_system(
                args.evaluation_dir / "student" / "metrics.json", "student"
            ),
            "before_ensemble": taskl["lakes"]["entitables"]["systems"]["student_ensemble"],
            "mined_ensemble": _read_system(
                args.evaluation_dir / "student_ensemble" / "metrics.json",
                "student_ensemble",
            ),
        }
        systems["mined_student"]["comparison"] = _comparison(
            systems["mined_student"], systems["before_student"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed, candidate_channel="fused", reference_channel="fused",
        )
        systems["mined_ensemble"]["comparison"] = _comparison(
            systems["mined_ensemble"], systems["before_ensemble"],
            recall_ks=recall_ks, iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        use_mined = int(selection["best_epoch"]) > 0 and not selection["gate_unsatisfied"]
        payload = {
            "format_version": 1,
            "skipped": False,
            "parameters": {
                "recall_ks": list(recall_ks),
                "bootstrap_iterations": args.bootstrap_iterations,
                "bootstrap_seed": args.bootstrap_seed,
                "hard_fraction": 0.5,
                "hard_learning_rate": 2e-5,
                "teacher_ensemble_alpha": 0.7,
            },
            "mining": {
                "best_epoch": int(selection["best_epoch"]),
                "gate_unsatisfied": bool(selection["gate_unsatisfied"]),
                "checkpoint": selection["best_checkpoint"],
                "checkpoint_sha256": selection["best_checkpoint_sha256"],
                "index": selection["best_index"],
                "index_bytes": _directory_bytes(Path(selection["best_index"])),
                "selection": str(args.selection.resolve()),
                "history_stop_reason": history["stop_reason"],
            },
            "systems": systems,
            "use_mined_checkpoint": use_mined,
            "final_checkpoint": (
                selection["best_checkpoint"]
                if use_mined
                else taskl["lakes"]["entitables"]["taskk_selection"]["checkpoint"]
            ),
        }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskl-metrics", type=Path)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skipped-reason")
    parser.add_argument("--recall-ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    args = parser.parse_args()
    if not args.skipped_reason and not all(
        (args.taskl_metrics, args.history, args.selection, args.evaluation_dir)
    ):
        parser.error("non-skipped summaries require Task-L, history, selection, and evaluation inputs")
    return args


if __name__ == "__main__":
    run(parse_args())

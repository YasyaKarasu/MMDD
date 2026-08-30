#!/usr/bin/env python
"""Combine the EntiTables and WDC Task-M Teacher-to-Student results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(payload: dict[str, Any]) -> str:
    rows = [
        "# Task M: per-lake Teacher-to-Student retraining",
        "",
        "Each lake uses its own retrieval-aligned lists, Teacher edge/path stages, "
        "Teacher rerank gate, and KD=0.3 Student rerun.",
        "",
        "## Teacher rerank gates",
        "",
        "| Lake | Edge lists | Path lists | Raw R@10 | Teacher R@10 | Delta | Gate (>3pt) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for lake in payload["lakes"].values():
        teacher = lake["teacher"]
        rows.append(
            f"| {lake['label']} | {lake['data']['edge_records']} | "
            f"{lake['data']['target_records']} | {teacher['raw_recall@10']:.2%} | "
            f"{teacher['reranked_recall@10']:.2%} | {teacher['delta']:+.2%} | "
            f"{'pass' if teacher['gate_pass'] else 'FAIL'} |"
        )
    rows.extend(
        [
            "",
            "## KD=0.3 Student reruns",
            "",
            "| Lake | Raw direct R@10 | Supervised direct R@10 | KD direct R@10 | "
            "KD fused R@10 | Epoch | Strict chain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for lake in payload["lakes"].values():
        references = lake["references"]
        student = lake["student_rerun"]
        rows.append(
            f"| {lake['label']} | {references['raw_direct_recall@10']:.2%} | "
            f"{references['supervised_direct_recall@10']:.2%} | "
            f"{student['direct_recall@10']:.2%} | {student['fused_recall@10']:.2%} | "
            f"{student['best_epoch']} | "
            f"{'pass' if lake['distillation_chain_established'] else 'FAIL'} |"
        )
    rows.extend(["", "## Provenance", ""])
    for lake in payload["lakes"].values():
        rows.extend(
            [
                f"### {lake['label']}",
                "",
                f"Teacher checkpoint: `{lake['teacher']['checkpoint']}`",
                f"Teacher SHA-256: `{lake['teacher']['checkpoint_sha256']}`",
                f"Student checkpoint: `{lake['student_rerun']['checkpoint']}`",
                f"Student SHA-256: `{lake['student_rerun']['checkpoint_sha256']}`",
                "",
            ]
        )
    return "\n".join(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    entitables = _read(args.entitables_metrics)
    wdc = _read(args.wdc_metrics)
    if entitables.get("lake") != "entitables" or wdc.get("lake") != "wdc":
        raise ValueError("Task-M per-lake metrics were assigned to the wrong lake")
    lakes = {"entitables": entitables, "wdc": wdc}
    payload = {
        "format_version": 2,
        "lakes": lakes,
        "all_teacher_gates_pass": all(
            lake["teacher"]["gate_pass"] for lake in lakes.values()
        ),
        "distillation_chains": {
            name: lake["distillation_chain_established"]
            for name, lake in lakes.items()
        },
        # Compatibility aliases for the EntiTables-only Task-M consumers.
        "teacher": entitables["teacher"],
        "distillation_chain_established": entitables[
            "distillation_chain_established"
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(
        _markdown(payload) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entitables-metrics", type=Path, required=True)
    parser.add_argument("--wdc-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

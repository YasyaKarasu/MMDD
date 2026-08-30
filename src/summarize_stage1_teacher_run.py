#!/usr/bin/env python
"""Summarize a rerank-gated Stage-1 Teacher training history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.selection import write_json


def _max_recall_k(metrics: dict[str, Any]) -> int:
    values = [
        int(key.split("@", 1)[1])
        for key in metrics
        if key.startswith("recall@")
    ]
    if not values:
        raise ValueError("retrieval metrics contain no recall cutoff")
    return max(values)


def _row(
    scope: str,
    raw: dict[str, Any],
    teacher: dict[str, Any],
    max_k: int,
) -> str:
    return (
        f"| {scope} | {raw['recall@10']:.2%} | {teacher['recall@10']:.2%} | "
        f"{teacher[f'recall@{max_k}']:.2%} | "
        f"{teacher[f'mrr@{max_k}']:.4f} |"
    )


def _validate_matching_direct_metrics(
    history: dict[str, Any], diagnostic: dict[str, Any], diagnostic_path: Path
) -> None:
    for ranking in ("raw_direct", "teacher_reranked"):
        if ranking not in diagnostic:
            raise ValueError(f"{diagnostic_path}: no {ranking} diagnostic")
        history_metrics = history[ranking]
        diagnostic_metrics = diagnostic[ranking]
        scopes = ["overall", *history_metrics["by_dataset"]]
        if set(history_metrics["by_dataset"]) != set(
            diagnostic_metrics["by_dataset"]
        ):
            raise ValueError(
                f"{diagnostic_path}: {ranking} dataset scopes do not match history"
            )
        for scope in scopes:
            history_scope = (
                history_metrics
                if scope == "overall"
                else history_metrics["by_dataset"][scope]
            )
            diagnostic_scope = (
                diagnostic_metrics
                if scope == "overall"
                else diagnostic_metrics["by_dataset"][scope]
            )
            if abs(
                float(history_scope["recall@10"])
                - float(diagnostic_scope["recall@10"])
            ) > 1e-12:
                raise ValueError(
                    f"{diagnostic_path}: {ranking} {scope} R@10 does not match "
                    "the best-epoch history"
                )


def run(args: argparse.Namespace) -> str:
    history_path = Path(args.history)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [
        record for record in payload["epochs"] if "dev_teacher_rerank" in record
    ]
    if not epochs:
        raise ValueError(f"{history_path}: no Teacher rerank epochs")
    best_epoch = int(payload["best_epoch"])
    best = next(record for record in epochs if int(record["epoch"]) == best_epoch)
    evaluation = best["dev_teacher_rerank"]
    diagnostic_path_value = getattr(args, "diagnostic", None)
    diagnostic = None
    evidence_to_table_pass = None
    if diagnostic_path_value:
        diagnostic_path = Path(diagnostic_path_value)
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        expected_sha256 = best.get("candidate_checkpoint_sha256")
        if (
            expected_sha256 is not None
            and diagnostic.get("teacher_checkpoint_sha256") != expected_sha256
        ):
            raise ValueError(
                f"{diagnostic_path}: diagnostic checkpoint does not match the best epoch"
            )
        _validate_matching_direct_metrics(evaluation, diagnostic, diagnostic_path)
        evaluation = diagnostic
        evidence_to_table = diagnostic.get("evidence_to_table")
        if evidence_to_table is None:
            raise ValueError(f"{diagnostic_path}: no evidence-to-table diagnostic")
        evidence_to_table_pass = (
            evidence_to_table["teacher_reranked"]["recall@10"]
            > evidence_to_table["raw_direct"]["recall@10"]
        )
    raw = evaluation["raw_direct"]
    teacher = evaluation["teacher_reranked"]
    max_k = _max_recall_k(teacher)
    dataset_passes = {
        dataset: (
            teacher["by_dataset"][dataset]["recall@10"]
            >= raw["by_dataset"][dataset]["recall@10"]
        )
        for dataset in raw["by_dataset"]
    }
    overall_pass = teacher["recall@10"] >= args.minimum_recall_at_10
    direct_pass = overall_pass and all(dataset_passes.values())
    accepted = direct_pass and evidence_to_table_pass is not False
    lines = [
        f"# {args.label}",
        "",
        f"Artifact: `{history_path.parent}`",
        "",
        f"Best epoch: {best_epoch}; dev loss: {best['dev_loss']:.4f}.",
        "",
        f"| Scope | Raw R@10 | Teacher R@10 | Teacher R@{max_k} | Teacher MRR@{max_k} |",
        "| --- | ---: | ---: | ---: | ---: |",
        _row("overall", raw, teacher, max_k),
    ]
    for dataset in raw["by_dataset"]:
        lines.append(
            _row(
                dataset,
                raw["by_dataset"][dataset],
                teacher["by_dataset"][dataset],
                max_k,
            )
        )
    lines.extend(
        [
            "",
            f"Teacher rerank acceptance (overall ≥ {args.minimum_recall_at_10:.0%} "
            f"and no dataset below raw): **{'PASS' if direct_pass else 'FAIL'}**.",
            f"Per-dataset guardrails: `{dataset_passes}`.",
            "",
        ]
    )
    if diagnostic is not None:
        evidence_to_table = diagnostic["evidence_to_table"]
        evidence_raw = evidence_to_table["raw_direct"]
        evidence_teacher = evidence_to_table["teacher_reranked"]
        evidence_max_k = _max_recall_k(evidence_teacher)
        lines.extend(
            [
                "## Evidence-to-table rerank",
                "",
                f"| Raw R@10 | Teacher R@10 | Teacher R@{evidence_max_k} | Teacher MRR@{evidence_max_k} |",
                "| ---: | ---: | ---: | ---: |",
                f"| {evidence_raw['recall@10']:.2%} | "
                f"{evidence_teacher['recall@10']:.2%} | "
                f"{evidence_teacher[f'recall@{evidence_max_k}']:.2%} | "
                f"{evidence_teacher[f'mrr@{evidence_max_k}']:.4f} |",
                "",
                "Evidence-to-table acceptance (Teacher R@10 > raw): "
                f"**{'PASS' if evidence_to_table_pass else 'FAIL'}**.",
                "",
                f"Task 7 combined acceptance: **{'PASS' if accepted else 'FAIL'}**.",
                "",
            ]
        )
    report = "\n".join(lines)
    output = Path(args.output) if args.output else history_path.parent / "RESULTS.md"
    output.write_text(report, encoding="utf-8")
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    gate_output_value = getattr(args, "gate_output", None)
    if gate_output_value:
        write_json(
            Path(gate_output_value),
            {
                "format_version": 1,
                "accepted": accepted,
                "direct_pass": direct_pass,
                "overall_pass": overall_pass,
                "dataset_passes": dataset_passes,
                "evidence_to_table_pass": evidence_to_table_pass,
                "minimum_recall_at_10": args.minimum_recall_at_10,
                "best_epoch": best_epoch,
            },
        )
    print(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--label", default="Task 7: retrieval-aligned Teacher")
    parser.add_argument("--minimum-recall-at-10", type=float, default=0.40)
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--diagnostic")
    parser.add_argument("--gate-output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

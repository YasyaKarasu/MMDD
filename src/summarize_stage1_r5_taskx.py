#!/usr/bin/env python
"""Summarize Stage-1 r5 evidence-binding and modality-balance ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta


VARIANTS = ("current", "text_only", "target_bound", "balanced")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _view(
    metrics: dict[str, Any],
    *,
    teacher_feature_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    view = {
        "fused_recall@10": float(metrics["recall@10"]),
        "direct_recall@10": float(metrics["direct"]["recall@10"]),
        "evidence_recall@10": float(metrics["evidence"]["recall@10"]),
        "coverage@10": float(metrics["positive_evidence_path_coverage@10"]),
        "fused_per_query": list(metrics["per_query"]["fused"]["recall@10"]),
    }
    if teacher_feature_coverage is not None:
        view["teacher_feature_coverage"] = teacher_feature_coverage
    return view


def _delta(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    return paired_bootstrap_delta(
        candidate["fused_per_query"],
        reference["fused_per_query"],
        iterations=iterations,
        seed=seed,
    )


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def run(args: argparse.Namespace) -> dict[str, Any]:
    task_root = Path(args.task_root)
    output_dir = Path(args.output_dir or args.task_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    distributions: dict[str, dict[str, Any]] = {}
    for lake in ("wdc", "entitables"):
        distributions[lake] = {}
        for variant in VARIANTS:
            run_dir = task_root / lake / variant
            preflight = _read(run_dir / "data" / "preflight.json")
            teacher_payload = _read(
                run_dir / "teacher_retrieval" / "metrics.json"
            )["teacher"]
            teacher = _view(
                teacher_payload["metrics"],
                teacher_feature_coverage=teacher_payload.get("feature_coverage"),
            )
            student = _view(
                _read(
                    run_dir
                    / "student_tau_0.3"
                    / "final_evaluation"
                    / "metrics.json"
                )["systems"]["student"]["metrics"]
            )
            raw_rows[(lake, variant, "teacher")] = teacher
            raw_rows[(lake, variant, "student")] = student
            distributions[lake][variant] = {
                "train_relation_counts_before": preflight[
                    "train_relation_counts_before"
                ],
                "train_relation_counts_after": preflight[
                    "train_relation_counts_after"
                ],
                "suppressed_train_relations": preflight[
                    "suppressed_train_relations"
                ],
                "evidence_concentration": preflight["evidence_concentration"],
            }

    rows = []
    for lake in ("wdc", "entitables"):
        for variant in VARIANTS:
            for model in ("teacher", "student"):
                metrics = raw_rows[(lake, variant, model)]
                current = raw_rows[(lake, "current", model)]
                rows.append(
                    {
                        "lake": lake,
                        "variant": variant,
                        "model": model,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "fused_per_query"
                        },
                        "fused_delta_vs_current": _delta(
                            metrics,
                            current,
                            iterations=args.bootstrap_iterations,
                            seed=args.bootstrap_seed,
                        ),
                        "evidence_delta_vs_current": (
                            metrics["evidence_recall@10"]
                            - current["evidence_recall@10"]
                        ),
                        "coverage_delta_vs_current": (
                            metrics["coverage@10"] - current["coverage@10"]
                        ),
                    }
                )

    def row(lake: str, variant: str, model: str) -> dict[str, Any]:
        return next(
            value
            for value in rows
            if value["lake"] == lake
            and value["variant"] == variant
            and value["model"] == model
        )

    decisions = {}
    for variant in VARIANTS[1:]:
        wdc_student = row("wdc", variant, "student")
        enti_student = row("entitables", variant, "student")
        decisions[variant] = {
            "wdc_evidence_and_coverage_gain_at_least_3pt": (
                wdc_student["evidence_delta_vs_current"] >= 0.03
                and wdc_student["coverage_delta_vs_current"] >= 0.03
            ),
            "entitables_fused_not_harmed": (
                enti_student["fused_delta_vs_current"]["ci_low"]
                >= -args.entitables_tolerance
            ),
        }
        decisions[variant]["accepted"] = all(decisions[variant].values())
    decisions["image_sparsity_resolved"] = any(
        decisions[variant]["accepted"] for variant in ("text_only", "balanced")
    )
    decisions["target_binding_resolved"] = decisions["target_bound"]["accepted"]

    payload = {
        "format_version": 1,
        "parameters": {
            "student_tau": 0.3,
            "distillation_weight": 0.3,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "entitables_tolerance": args.entitables_tolerance,
        },
        "rows": rows,
        "training_distributions": distributions,
        "decisions": decisions,
    }
    write_json(output_dir / "metrics.json", payload)
    with (output_dir / "evidence_modality_ablation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lake",
                "variant",
                "model",
                "fused_recall_at_10",
                "direct_recall_at_10",
                "evidence_recall_at_10",
                "coverage_at_10",
                "fused_delta_vs_current",
                "ci_low",
                "ci_high",
                "evidence_delta_vs_current",
                "coverage_delta_vs_current",
                "teacher_cached_hidden_fraction",
                "teacher_pooled_embedding_fallback_objects",
                "teacher_allowed_pooled_embedding_fallback_objects",
                "teacher_unexpected_pooled_embedding_fallback_objects",
            ]
        )
        for value in rows:
            delta = value["fused_delta_vs_current"]
            feature_coverage = value.get("teacher_feature_coverage")
            writer.writerow(
                [
                    value["lake"],
                    value["variant"],
                    value["model"],
                    value["fused_recall@10"],
                    value["direct_recall@10"],
                    value["evidence_recall@10"],
                    value["coverage@10"],
                    delta["mean"],
                    delta["ci_low"],
                    delta["ci_high"],
                    value["evidence_delta_vs_current"],
                    value["coverage_delta_vs_current"],
                    (
                        feature_coverage["cached_hidden_fraction"]
                        if feature_coverage is not None
                        else ""
                    ),
                    (
                        feature_coverage["pooled_embedding_fallback_objects"]
                        if feature_coverage is not None
                        else ""
                    ),
                    (
                        feature_coverage.get(
                            "allowed_pooled_embedding_fallback_objects", 0
                        )
                        if feature_coverage is not None
                        else ""
                    ),
                    (
                        feature_coverage.get(
                            "unexpected_pooled_embedding_fallback_objects",
                            feature_coverage.get(
                                "pooled_embedding_fallback_objects", 0
                            ),
                        )
                        if feature_coverage is not None
                        else ""
                    ),
                ]
            )

    lines = [
        "# Task X: evidence binding and modality balance",
        "",
        "Student KD uses tau=0.3 and distillation weight 0.3 in every variant.",
        "Teacher metrics use full zero/one-hop reranking over fixed raw ANN edge pools.",
        "",
        "| Lake | Variant | Model | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | Teacher hidden coverage | Fused delta vs current / 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for value in rows:
        delta = value["fused_delta_vs_current"]
        feature_coverage = value.get("teacher_feature_coverage")
        feature_coverage_text = (
            "n/a"
            if feature_coverage is None
            else _percent(feature_coverage["cached_hidden_fraction"])
        )
        lines.append(
            f"| {value['lake']} | {value['variant']} | {value['model']} | "
            f"{_percent(value['fused_recall@10'])} | "
            f"{_percent(value['direct_recall@10'])} | "
            f"{_percent(value['evidence_recall@10'])} | "
            f"{_percent(value['coverage@10'])} | "
            f"{feature_coverage_text} | "
            f"{_percent(delta['mean'])} "
            f"[{_percent(delta['ci_low'])}, {_percent(delta['ci_high'])}] |"
        )
    lines.extend(["", "## Decisions", ""])
    for variant in VARIANTS[1:]:
        value = decisions[variant]
        lines.append(
            f"- {variant}: WDC evidence+coverage >=3pt "
            f"**{value['wdc_evidence_and_coverage_gain_at_least_3pt']}**; "
            f"EntiTables not harmed **{value['entitables_fused_not_harmed']}**; "
            f"accepted **{value['accepted']}**."
        )
    lines.extend(
        [
            f"- Image sparsity resolved: **{decisions['image_sparsity_resolved']}**.",
            f"- Target binding resolved: **{decisions['target_binding_resolved']}**.",
            "",
            "Teacher retrieval permits pooled-embedding fallback only for the "
            "11 audited invalid image objects; all other missing hidden states "
            "remain fatal. Counts by lake and variant are recorded in "
            "`metrics.json` and the CSV.",
            "",
            "Per-relation counts and evidence top-1/top-10 concentration are in `metrics.json`.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--entitables-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

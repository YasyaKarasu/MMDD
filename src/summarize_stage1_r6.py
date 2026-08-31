#!/usr/bin/env python
"""Select Stage-1 r6 variants and assemble the final experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.protocol import validate_r6_readonly_invariants


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _student_metrics(path: Path) -> dict[str, Any]:
    return _load(path)["systems"]["student"]["metrics"]


def _r5_paths(root: Path, lake: str) -> tuple[Path, Path]:
    r5 = root / "work/stage1_optimization_r5_20260829"
    selected = _load(r5 / "task4_final/selection.json")["selected"][lake]
    evaluation = r5 / f"task4_final/final_evaluation/{lake}/metrics.json"
    return Path(selected["student_selection"]), evaluation


def _task_e_metrics(output_root: Path, lake: str, weight: float) -> dict[str, Any]:
    return _student_metrics(
        output_root
        / "taskE_modality_balance"
        / lake
        / f"relation_weight_{weight:g}"
        / "final_evaluation/metrics.json"
    )


def select_wdc_weight(root: Path, output_root: Path) -> dict[str, Any]:
    baseline_selection, baseline_evaluation = _r5_paths(root, "wdc")
    r5_metrics = _student_metrics(baseline_evaluation)
    candidates = []
    for weight in (1.0, 2.0, 4.0):
        metrics = _task_e_metrics(output_root, "wdc", weight)
        selection = (
            output_root
            / "taskE_modality_balance"
            / "wdc"
            / f"relation_weight_{weight:g}/student_path.pt.selection.json"
        )
        candidates.append(
            {
                "relation_weight": weight,
                "student_selection": str(selection.resolve()),
                "fused_recall@10": metrics["recall@10"],
                "evidence_recall@10": metrics["evidence"]["recall@10"],
                "coverage@10": metrics["positive_evidence_path_coverage@10"],
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row["evidence_recall@10"],
            row["coverage@10"],
            row["fused_recall@10"],
            -abs(row["relation_weight"] - 1.0),
        ),
    )
    payload = {
        "format_version": 1,
        "selection_rule": "evidence R@10, coverage@10, fused R@10, then lower intervention",
        "r5_manual_baseline": {
            "tau": 0.7,
            "student_selection": str(baseline_selection.resolve()),
            "fused_recall@10": r5_metrics["recall@10"],
            "evidence_recall@10": r5_metrics["evidence"]["recall@10"],
            "coverage@10": r5_metrics["positive_evidence_path_coverage@10"],
        },
        "candidates": candidates,
        "selected": selected,
    }
    _write(output_root / "taskE_modality_balance/wdc_selection.json", payload)
    return payload


def _best_fusion(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    configs = payload["systems"]["student"]["configs"]
    return max(
        configs.items(),
        key=lambda item: (
            item[1]["metrics"]["recall@10"],
            item[1]["metrics"]["positive_evidence_path_coverage@10"],
            item[1]["metrics"]["mrr@50"],
            item[0] == payload["baseline"],
        ),
    )


def _best_aggregation(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    configs = payload["systems"]["student"]["configs"]
    baseline = configs[payload["baseline"]]["metrics"]["recall@10"]
    eligible = {
        name: record
        for name, record in configs.items()
        if record["metrics"]["recall@10"] >= baseline
    }
    return max(
        eligible.items(),
        key=lambda item: (
            item[1]["metrics"]["evidence"]["recall@10"],
            item[1]["metrics"]["positive_evidence_path_coverage@10"],
            item[1]["metrics"]["recall@10"],
            item[0] == payload["baseline"],
        ),
    )


def _student_selection(
    root: Path,
    output_root: Path,
    lake: str,
    relation_weight: float,
    adaptive_tau: float,
    manual_tau: float,
) -> Path:
    if relation_weight == 1.0 and adaptive_tau == manual_tau:
        selection, _evaluation = _r5_paths(root, lake)
        return selection
    return (
        output_root
        / "taskE_modality_balance"
        / lake
        / f"relation_weight_{relation_weight:g}/student_path.pt.selection.json"
    )


def _configuration(
    root: Path, output_root: Path, task_e: dict[str, Any]
) -> dict[str, Any]:
    lakes = {}
    wdc_weight = float(task_e["selected"]["relation_weight"])
    for lake in ("entitables", "wdc"):
        task_b = _load(output_root / f"taskB_fusion_normalization/{lake}.json")
        task_d = _load(output_root / f"taskD_path_aggregation/{lake}.json")
        task_c = _load(output_root / f"taskC_adaptive_tau/{lake}.json")
        fusion_name, fusion = _best_fusion(task_b)
        aggregation_name, aggregation = _best_aggregation(task_d)
        student_selection = _student_selection(
            root,
            output_root,
            lake,
            wdc_weight,
            float(task_c["selected_tau"]),
            float(task_c["manual_r5_tau"]),
        )
        lakes[lake] = {
            "student_selection": str(student_selection.resolve()),
            "relation_weight": wdc_weight,
            "adaptive_tau": task_c["selected_tau"],
            "manual_r5_tau": task_c["manual_r5_tau"],
            "fusion": {
                "name": fusion_name,
                **next(
                    config
                    for config in task_b["configurations"]
                    if config["name"] == fusion_name
                ),
                "metrics": fusion["metrics"],
            },
            "aggregation": {
                "name": aggregation_name,
                **next(
                    config
                    for config in task_d["configurations"]
                    if config["name"] == aggregation_name
                ),
                "metrics": aggregation["metrics"],
            },
            "teacher_checkpoint": task_b["teacher_checkpoint"],
            "evidence_types": ["text", "image"],
        }
    return {
        "format_version": 1,
        "lakes": lakes,
        "global": {
            "gamma": 10,
            "gamma_evidence": 2,
            "recall_ks": [10, 20, 30, 40, 50],
            "bootstrap_iterations": 10000,
            "bootstrap_seed": 13,
            "distillation_weight": 0.3,
            "student_dim": 1024,
            "projection_frozen": True,
            "image_modality_retained": True,
        },
    }


def _final_report(
    root: Path,
    output_root: Path,
    configuration: dict[str, Any],
    task_e: dict[str, Any],
) -> None:
    task_a = {
        lake: _load(output_root / f"taskA_scale_diagnostic/{lake}.json")
        for lake in ("entitables", "wdc")
    }
    hypothesis = {
        lake: max(
            value
            for ratios in task_a[lake]["systems"]["student"][
                "opposite_edge_scale_ratios"
            ].values()
            for value in ratios.values()
            if value is not None
        )
        > 2.0
        for lake in task_a
    }
    final_metrics = {}
    final_ready = True
    for lake in ("entitables", "wdc"):
        path = output_root / f"task_final/{lake}.json"
        if path.is_file():
            final_metrics[lake] = _load(path)
        else:
            final_ready = False
    task_f1_summaries = {
        lake: output_root / f"taskF_table_representation/{lake}/summary.json"
        for lake in ("entitables", "wdc")
    }
    task_f2_zero_shot_summaries = {
        lake: output_root
        / f"taskF_table_representation/tokens_per_group_4/{lake}/summary.json"
        for lake in ("entitables", "wdc")
    }
    task_f2_summaries = {
        lake: output_root
        / f"taskF_table_representation/tokens_per_group_4/retrained/{lake}/summary.json"
        for lake in ("entitables", "wdc")
    }
    task_f1_ready = all(path.is_file() for path in task_f1_summaries.values())
    task_f2_zero_shot_ready = all(
        path.is_file() for path in task_f2_zero_shot_summaries.values()
    )
    task_f2_ready = all(path.is_file() for path in task_f2_summaries.values())
    task_f = {
        "status": "complete" if task_f1_ready and task_f2_ready else "running",
        "variants": {
            "f1_more_rows": {
                "status": "complete" if task_f1_ready else "running",
                "configuration": {
                    "max_rows": 20,
                    "table_row_format": "named_cells",
                    "table_tokens_per_group": 1,
                },
                "lakes": (
                    {lake: _load(path) for lake, path in task_f1_summaries.items()}
                    if task_f1_ready
                    else {}
                ),
            },
            "f2_more_tokens": {
                "status": "complete" if task_f2_ready else "running",
                "configuration": {
                    "max_rows": 12,
                    "table_row_format": "values",
                    "table_tokens_per_group": 4,
                },
                "lakes": (
                    {lake: _load(path) for lake, path in task_f2_summaries.items()}
                    if task_f2_ready
                    else {}
                ),
                "zero_shot_diagnostic_lakes": (
                    {
                        lake: _load(path)
                        for lake, path in task_f2_zero_shot_summaries.items()
                    }
                    if task_f2_zero_shot_ready
                    else {}
                ),
            },
        },
    }
    payload = {
        "format_version": 1,
        "configuration": configuration,
        "scale_mismatch_confirmed": hypothesis,
        "task_e": task_e,
        "final_metrics": final_metrics,
        "final_evaluation_complete": final_ready,
        "task_f": task_f,
    }
    _write(output_root / "final_metrics.json", payload)

    lines = [
        "# Stage-1 round-six results",
        "",
        "## Mechanism diagnostics",
        "",
    ]
    for lake in ("entitables", "wdc"):
        task_c = _load(output_root / f"taskC_adaptive_tau/{lake}.json")
        lines.append(
            f"- {lake}: score-scale mismatch >2x = {hypothesis[lake]}; "
            f"adaptive tau={task_c['selected_tau']} (r5 manual={task_c['manual_r5_tau']}); "
            f"fixed tau candidate pool={task_c['same_candidate_pool']}."
        )
    lines.extend(
        [
            "",
            "## Selected configuration",
            "",
            "| Lake | Fusion | Path aggregation | Image relation weight | Tau |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for lake, config in configuration["lakes"].items():
        lines.append(
            f"| {lake} | {config['fusion']['name']} | "
            f"{config['aggregation']['name']} | {config['relation_weight']:g} | "
            f"{config['adaptive_tau']:g} |"
        )
    entitables_weight = float(task_e["selected"]["relation_weight"])
    _entitables_selection, entitables_r5_evaluation = _r5_paths(root, "entitables")
    entitables_r5 = _student_metrics(entitables_r5_evaluation)
    entitables_task_e = _task_e_metrics(
        output_root, "entitables", entitables_weight
    )
    lines.extend(
        [
            "",
            "## Task E modality balance",
            "",
            "| Lake | Configuration | Fused R@10 | Evidence R@10 | Coverage@10 |",
            "| --- | --- | ---: | ---: | ---: |",
            f"| entitables | r5 manual | {entitables_r5['recall@10']:.2%} | "
            f"{entitables_r5['evidence']['recall@10']:.2%} | "
            f"{entitables_r5['positive_evidence_path_coverage@10']:.2%} |",
            f"| entitables | relation weight {entitables_weight:g} | "
            f"{entitables_task_e['recall@10']:.2%} | "
            f"{entitables_task_e['evidence']['recall@10']:.2%} | "
            f"{entitables_task_e['positive_evidence_path_coverage@10']:.2%} |",
            f"| wdc | r5 manual | "
            f"{task_e['r5_manual_baseline']['fused_recall@10']:.2%} | "
            f"{task_e['r5_manual_baseline']['evidence_recall@10']:.2%} | "
            f"{task_e['r5_manual_baseline']['coverage@10']:.2%} |",
        ]
    )
    for candidate in task_e["candidates"]:
        lines.append(
            f"| wdc | relation weight {candidate['relation_weight']:g} | "
            f"{candidate['fused_recall@10']:.2%} | "
            f"{candidate['evidence_recall@10']:.2%} | "
            f"{candidate['coverage@10']:.2%} |"
        )
    selected_task_e = task_e["selected"]
    baseline_task_e = task_e["r5_manual_baseline"]
    evidence_delta = (
        selected_task_e["evidence_recall@10"]
        - baseline_task_e["evidence_recall@10"]
    )
    coverage_delta = selected_task_e["coverage@10"] - baseline_task_e["coverage@10"]
    lines.extend(
        [
            "",
            f"WDC selected relation weight {selected_task_e['relation_weight']:g} by the "
            f"evidence-first rule: evidence R@10 changed by {evidence_delta:+.2%}, "
            f"while coverage changed by {coverage_delta:+.2%}. The full Task E "
            "evidence-plus-coverage acceptance target was not met.",
        ]
    )
    if final_ready:
        lines.extend(
            [
                "",
                "## Final main table",
                "",
                "| Lake | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | MRR@50 | Delta vs r5 / 95% CI |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for lake, final in final_metrics.items():
            metrics = final["metrics"]
            delta = final["delta_vs_r5"]["recall@10"]
            lines.append(
                f"| {lake} | {metrics['recall@10']:.2%} | "
                f"{metrics['direct']['recall@10']:.2%} | "
                f"{metrics['evidence']['recall@10']:.2%} | "
                f"{metrics['positive_evidence_path_coverage@10']:.2%} | "
                f"{metrics['mrr@50']:.4f} | {delta['mean']:.2%} "
                f"[{delta['ci_low']:.2%}, {delta['ci_high']:.2%}] |"
            )
    else:
        lines.extend(["", "Final combined evaluation is pending."])
    lines.extend(["", "## Task F table representation"])
    if task_f1_ready:
        lines.extend(
            [
                "",
                "### F1: more rows and named cells",
                "",
                "F1 increases `max_rows` from 12 to 20 and serializes rows as named cells (`column_name: value`).",
                "",
                "| Lake | Student evidence R@10 | Δ / 95% CI | Teacher rerank Δ@10 | Change / 95% CI |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for lake, summary_path in task_f1_summaries.items():
            summary = _load(summary_path)
            evidence = summary["student_evidence_recall@10"]
            evidence_delta = evidence["delta"]
            teacher = summary["teacher_rerank_delta@10"]
            teacher_delta = teacher["difference_in_differences"]
            lines.append(
                f"| {lake} | {evidence['task_f']:.2%} | "
                f"{evidence_delta['mean']:+.2%} "
                f"[{evidence_delta['ci_low']:+.2%}, {evidence_delta['ci_high']:+.2%}] | "
                f"{teacher['task_f']:+.2%} | {teacher_delta['mean']:+.2%} "
                f"[{teacher_delta['ci_low']:+.2%}, {teacher_delta['ci_high']:+.2%}] |"
            )
    else:
        lines.extend(
            [
                "",
                "F1 row-count/serialization evaluation is running.",
            ]
        )
    if task_f2_ready:
        lines.extend(
            [
                "",
                "### F2: more tokens per schema/row",
                "",
                "F2 retains the original 12-row value serialization, keeps up to four contiguous pooled tokens per schema/row group, and retrains a lake-local Teacher on that representation.",
                "",
                "| Lake | Student evidence R@10 | Δ | Teacher rerank Δ@10 | Change / 95% CI |",
                "| --- | ---: | --- | ---: | --- |",
            ]
        )
        for lake, summary_path in task_f2_summaries.items():
            summary = _load(summary_path)
            evidence = summary["student_evidence_recall@10"]
            teacher = summary["teacher_rerank_delta@10"]
            teacher_delta = teacher["difference_in_differences"]
            lines.append(
                f"| {lake} | {evidence['task_f']:.2%} | unchanged by construction | "
                f"{teacher['task_f']:+.2%} | {teacher_delta['mean']:+.2%} "
                f"[{teacher_delta['ci_low']:+.2%}, {teacher_delta['ci_high']:+.2%}] |"
            )
    else:
        lines.extend(
            [
                "",
                "F2 four-token-per-schema/row Teacher retraining and formal evaluation are running.",
            ]
        )
    if task_f2_zero_shot_ready:
        lines.extend(
            [
                "",
                "#### F2 zero-shot diagnostic (old one-token Teacher)",
                "",
                "This is retained only as a representation-shift diagnostic; it is not the formal F2 result.",
                "",
                "| Lake | Teacher rerank Δ@10 | Change / 95% CI |",
                "| --- | ---: | --- |",
            ]
        )
        for lake, summary_path in task_f2_zero_shot_summaries.items():
            summary = _load(summary_path)
            teacher = summary["teacher_rerank_delta@10"]
            teacher_delta = teacher["difference_in_differences"]
            lines.append(
                f"| {lake} | {teacher['task_f']:+.2%} | "
                f"{teacher_delta['mean']:+.2%} "
                f"[{teacher_delta['ci_low']:+.2%}, {teacher_delta['ci_high']:+.2%}] |"
            )
    lines.extend(
        [
            "",
            "All final configurations retain both text and image evidence. F2 formal results use newly retrained lake-local Teachers; confidence intervals use paired bootstrap (10,000 iterations, seed 13).",
            "",
        ]
    )
    (output_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if final_ready:
        (output_root / "FINAL.md").write_text("\n".join(lines), encoding="utf-8")


def finalize(root: Path, output_root: Path) -> dict[str, Any]:
    task_e_path = output_root / "taskE_modality_balance/wdc_selection.json"
    task_e = _load(task_e_path) if task_e_path.is_file() else select_wdc_weight(root, output_root)
    configuration = _configuration(root, output_root, task_e)
    teachers = {
        lake: values["teacher_checkpoint"]
        for lake, values in configuration["lakes"].items()
    }
    for lake in ("entitables", "wdc"):
        task_c = _load(output_root / f"taskC_adaptive_tau/{lake}.json")
        validate_r6_readonly_invariants(
            evidence_types=configuration["lakes"][lake]["evidence_types"],
            candidate_pool_fingerprints=task_c["candidate_pool_sha256_by_tau"],
            teacher_checkpoints=teachers,
        )
    _write(output_root / "selection.json", configuration)
    _final_report(root, output_root, configuration, task_e)
    return configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["select-wdc-weight", "finalize"], nargs="?", default="finalize"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r6_20260830"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.command == "select-wdc-weight":
        print(json.dumps(select_wdc_weight(repository, args.output_root), indent=2))
    else:
        print(json.dumps(finalize(repository, args.output_root), indent=2))

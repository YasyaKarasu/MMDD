#!/usr/bin/env python
"""Build the final round-3 report from the validated task artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.selection import write_json


SYSTEM_ORDER = ("raw", "student", "raw_ensemble", "student_ensemble")


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _paired_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    recall_ks: list[int],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "overall": {
            f"recall@{k}": paired_bootstrap_delta(
                candidate["per_query"][f"recall@{k}"],
                reference["per_query"][f"recall@{k}"],
                iterations=iterations,
                seed=seed,
            )
            for k in recall_ks
        },
        "by_dataset": {
            dataset: {
                f"recall@{k}": paired_bootstrap_delta(
                    candidate["by_dataset"][dataset]["per_query"][f"recall@{k}"],
                    reference["by_dataset"][dataset]["per_query"][f"recall@{k}"],
                    iterations=iterations,
                    seed=seed,
                )
                for k in recall_ks
            }
            for dataset in candidate["by_dataset"]
        },
    }


def _ci(value: dict[str, Any]) -> str:
    return f"{value['mean']:+.2%} [{value['ci_low']:+.2%}, {value['ci_high']:+.2%}]"


def _metric_row(
    label: str,
    metrics: dict[str, Any],
    recall_ks: list[int],
    comparison: dict[str, Any] | None,
) -> str:
    coverage = metrics.get("positive_evidence_path_coverage@10")
    ci = "n/a" if comparison is None else _ci(comparison["recall@10"])
    return (
        f"| {label} | "
        + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in recall_ks)
        + f" | {metrics[f'mrr@{max(recall_ks)}']:.4f} | "
        + ("n/a" if coverage is None else f"{coverage:.2%}")
        + f" | {ci} |"
    )


def run(args: argparse.Namespace) -> None:
    metrics = _load(args.metrics)
    task_e = _load(args.task_e_metrics)
    task_f = _load(args.task_f_selection)
    task_g = _load(args.task_g_selection)
    task_h = _load(args.task_h_metrics)
    selection = _load(args.student_selection)
    recall_ks = list(metrics["parameters"]["recall_ks"])
    iterations = int(metrics["parameters"]["bootstrap_iterations"])
    seed = int(metrics["parameters"]["bootstrap_seed"])
    systems = metrics["systems"]

    deployment_vs_full = _paired_comparison(
        systems["student_ensemble"]["metrics"],
        systems["raw_ensemble"]["metrics"],
        recall_ks,
        iterations=iterations,
        seed=seed,
    )
    task_e_deployment_vs_full = _paired_comparison(
        task_e["systems"]["student_ensemble"]["metrics"],
        task_e["systems"]["raw_ensemble"]["metrics"],
        list(task_e["parameters"]["recall_ks"]),
        iterations=int(task_e["parameters"]["bootstrap_iterations"]),
        seed=int(task_e["parameters"]["bootstrap_seed"]),
    )

    raw_index_bytes = _directory_bytes(Path(selection["raw_embedding_index"]))
    student_index_bytes = _directory_bytes(Path(selection["best_index"]))
    index_ratio = raw_index_bytes / student_index_bytes
    summary = {
        "format_version": 1,
        "final_metrics": str(Path(args.metrics).resolve()),
        "deployment_vs_full_ensemble": deployment_vs_full,
        "task_e_deployment_vs_full_ensemble": task_e_deployment_vs_full,
        "gamma_star": task_f["gamma_star"],
        "recall_depth_bottleneck": task_f["recall_depth_bottleneck"],
        "gamma_evidence_star": task_g["gamma_evidence_star"],
        "kd_direction_closed": task_h["kd_direction_closed"],
        "final_student_checkpoint": task_h["final_student_checkpoint"],
        "final_student_checkpoint_sha256": task_h["final_student_checkpoint_sha256"],
        "index_bytes": {
            "raw": raw_index_bytes,
            "student": student_index_bytes,
            "raw_to_student_ratio": index_ratio,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "summary.json", summary)

    header = (
        "| System | "
        + " | ".join(f"R@{k}" for k in recall_ks)
        + f" | MRR@{max(recall_ks)} | Coverage@10 | R@10 delta / paired 95% CI vs matching raw |"
    )
    divider = "| --- | " + " | ".join("---:" for _ in recall_ks) + " | ---: | ---: | --- |"
    lines = [
        "# Stage-1 optimization round 3: final",
        "",
        "## Decision",
        "",
        f"Final Student remains the R2 epoch-0 checkpoint because neither ensemble-KD weight passed the WDC CI gate after training. The deployed two-stage system is Student ANN retrieval plus the alpha=0.7 frozen-cosine/Teacher ensemble reranker, with gamma={task_f['gamma_star']} and gamma_e={task_g['gamma_evidence_star']}.",
        "",
        "Teacher-ensemble rows rerank direct Q-to-T candidate pools, so evidence-path coverage is not applicable to those two systems.",
        "",
        "## Overall",
        "",
        header,
        divider,
    ]
    for name in SYSTEM_ORDER:
        result = systems[name]
        comparison = result.get("vs_raw", {}).get("overall")
        lines.append(_metric_row(result["label"], result["metrics"], recall_ks, comparison))

    gap = deployment_vs_full["overall"]["recall@10"]
    e_gap = task_e_deployment_vs_full["overall"]["recall@10"]
    lines += [
        "",
        f"At the final gamma, Student+ensemble is {_ci(gap)} versus raw+ensemble at R@10 (point gap {gap['mean']:+.2%}). At Task E's gamma=4 baseline the corresponding gap was {_ci(e_gap)}.",
        "",
        "## By dataset",
        "",
    ]
    for dataset in systems["raw"]["metrics"]["by_dataset"]:
        lines += [f"### {dataset}", "", header, divider]
        for name in SYSTEM_ORDER:
            result = systems[name]
            comparison = result.get("vs_raw", {}).get("by_dataset", {}).get(dataset)
            lines.append(
                _metric_row(
                    result["label"],
                    result["metrics"]["by_dataset"][dataset],
                    recall_ks,
                    comparison,
                )
            )
        dataset_gap = deployment_vs_full["by_dataset"][dataset]["recall@10"]
        lines += [
            "",
            f"Student+ensemble vs raw+ensemble R@10: {_ci(dataset_gap)}.",
            "",
        ]

    lines += [
        "## Sensitivity choices",
        "",
        f"- gamma*={task_f['gamma_star']}. The ensemble R@50 curve is still not saturated: gamma=8 to 10 gains {task_f['final_step_gains']['student_ensemble.recall@50']:+.2%}; retrieval depth remains a documented bottleneck. Source: `../taskF_gamma_sweep/RESULTS.md`; plot: `../taskF_gamma_sweep/gamma_sweep.png`.",
        f"- gamma_e*={task_g['gamma_evidence_star']}. It is within 0.5 points of maximum coverage; gamma_e=4 adds only 0.26 points while increasing CPU latency and lowering evidence R@10. Source: `../taskG_evidence_sweep/RESULTS.md`; plot: `../taskG_evidence_sweep/evidence_sweep.png`.",
        "- Ensemble-KD weights 0.3 and 1.0 both failed the WDC CI gate; the KD direction is closed for this formulation. Source: `../taskH_ensemble_kd/RESULTS.md`.",
        "",
        "## Timing and index size",
        "",
        "Timings are wall-clock CPU measurements over all five independently requested k values. Teacher score caches are independent across systems and reused only across k values within one system.",
        "",
        "| System | Total (s) | ms/query/k |",
        "| --- | ---: | ---: |",
    ]
    for name in SYSTEM_ORDER:
        result = systems[name]
        timing = result["timing"]
        lines.append(
            f"| {result['label']} | {timing['total_seconds']:.1f} | "
            f"{timing['average_seconds_per_query_per_k'] * 1000:.2f} |"
        )
    lines += [
        "",
        f"Raw index: {raw_index_bytes / 2**30:.2f} GiB; Student index: {student_index_bytes / 2**30:.2f} GiB; raw/Student size ratio: {index_ratio:.2f}x.",
        "",
        f"Hardware: device={metrics['hardware']['device']}; CPU={metrics['hardware']['cpu']}; torch={metrics['hardware']['torch']}; CUDA available to the evaluation process={metrics['hardware']['cuda_available']}.",
        "",
        "## Final configuration",
        "",
        f"- recall_ks={recall_ks}; MRR cutoff={max(recall_ks)}.",
        f"- Direct retrieval budget=gamma*k with gamma={task_f['gamma_star']}; evidence and evidence-to-target budgets=gamma_e*k with gamma_e={task_g['gamma_evidence_star']}.",
        "- Fusion=weighted RRF; direct weight=1.0; evidence weight=0.05; RRF k=60.",
        f"- Evidence aggregation={metrics['parameters']['evidence_aggregation']}; top-k={metrics['parameters']['evidence_top_k']}.",
        f"- Teacher ensemble alpha={metrics['parameters']['teacher_alpha']}; Teacher batch size={metrics['parameters']['teacher_batch_size']}.",
        "- Gate=paired bootstrap lower 95% CI >= -0.02 on WDC direct R@10; iterations=10,000; seed=13.",
        "- Student=fresh PCA-1024 epoch-0, frozen projections, identity-initialized relations; no hard-negative mining.",
        "",
        "## Provenance",
        "",
        f"- Student checkpoint: `{metrics['student_checkpoint']}`",
        f"- Student SHA-256: `{metrics['student_checkpoint_sha256']}`",
        f"- Teacher checkpoint: `{metrics['teacher_checkpoint']}`",
        f"- Teacher SHA-256: `{metrics['teacher_checkpoint_sha256']}`",
        f"- Corpus: `{metrics['corpus']}`",
        f"- Corpus SHA-256: `{metrics['corpus_sha256']}`",
        "",
        "Hard-negative mining remains frozen. Although Student+ensemble exceeds 45% overall R@10, the R3 plan requires Task H to succeed before mining restarts, and Task H failed.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--task-e-metrics", required=True)
    parser.add_argument("--task-f-selection", required=True)
    parser.add_argument("--task-g-selection", required=True)
    parser.add_argument("--task-h-metrics", required=True)
    parser.add_argument("--student-selection", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

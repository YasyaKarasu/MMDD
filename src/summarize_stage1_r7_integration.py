#!/usr/bin/env python
"""Build the auditable r7 two-lake integration table from saved artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


R5_ROOT = Path("work/stage1_optimization_r5_20260829")
EPOCH0_Q_CONFIG = "weighted_rrf_e0.05__logsumexp_edges_none"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _metric_values(metrics: dict[str, Any], *, coverage_key: str) -> dict[str, Any]:
    return {
        **{f"recall@{k}": float(metrics[f"recall@{k}"]) for k in (10, 20, 30, 40, 50)},
        "mrr@50": float(metrics["mrr@50"]),
        "coverage@10": (
            None if metrics.get(coverage_key) is None else float(metrics[coverage_key])
        ),
    }


def _ci_values(
    comparison: dict[str, Any] | None,
    *,
    baseline: str,
) -> dict[str, Any]:
    if comparison is None:
        return {
            "ci_baseline": baseline,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
        }
    return {
        "ci_baseline": baseline,
        "delta": float(comparison["delta_mean"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
    }


def _r5_rows(
    r5: dict[str, Any],
    lake: str,
    *,
    source_path: Path,
) -> list[dict[str, Any]]:
    labels = {
        "Raw embedding": (
            "Raw embedding",
            "reference",
            "raw embedding / weighted RRF e=0.05",
        ),
        "Supervised Student": (
            "r5 supervised Student",
            "raw embedding",
            "r5 supervised full-rank relation / weighted RRF e=0.05",
        ),
        "KD Student (final)": (
            "r5 KD Student (final)",
            "raw embedding",
            "r5 KD full-rank relation / weighted RRF e=0.05",
        ),
        "Student + online reranker": (
            "r5 Student + online reranker"
            if lake == "entitables"
            else "r5 Student + online reranker (no-op)",
            "raw embedding",
            "r5 KD + online teacher reranker"
            if lake == "entitables"
            else "r5 KD / online reranker explicitly disabled",
        ),
    }
    rows = []
    for source in r5["lakes"][lake]["rows"]:
        name = str(source["name"])
        output_name, baseline, configuration = labels[name]
        rows.append(
            {
                "lake": lake,
                "system": output_name,
                "configuration": configuration,
                **_metric_values(source, coverage_key="coverage@10"),
                **_ci_values(
                    None if name == "Raw embedding" else source["vs_raw_fused_recall@10"],
                    baseline=baseline,
                ),
                "source": str(source_path),
            }
        )
    return rows


def _task_p_summary(task_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = task_root / "taskP_lowrank/k_sensitivity.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["lake"], int(row["rank"])): row
            for row in csv.DictReader(handle)
            if float(row["anchor_weight"]) == 0.0
        }


def _epoch0_row(
    task_root: Path,
    r5: dict[str, Any],
    lake: str,
) -> dict[str, Any]:
    q_path = (
        task_root
        / f"taskQ_joint_selection/{lake}/lowrank_k_16_mu_0/metrics.json"
    )
    q_payload = _load(q_path)
    q_metrics = q_payload["metrics"][EPOCH0_Q_CONFIG]["metrics"]
    raw_per_query = r5["lakes"][lake]["rows"][0]["per_query_recall@10"]
    ci = paired_bootstrap_delta(
        q_metrics["per_query"]["recall@10"],
        raw_per_query,
        iterations=10_000,
        seed=13,
    )
    if lake == "entitables":
        metric_path = (
            task_root
            / "taskP_lowrank/entitables/k_16/mu_0/"
            "final_evaluation_epoch0_full/metrics.json"
        )
        metrics = _load(metric_path)["metrics"]
    else:
        metric_path = (
            task_root
            / "taskP_lowrank/wdc/k_16/mu_0/final_evaluation/metrics.json"
        )
        metrics = _load(metric_path)["systems"]["student"]["metrics"]
    if abs(float(metrics["recall@10"]) - float(q_metrics["recall@10"])) > 1e-12:
        raise ValueError(f"{lake}: epoch-0 R@10 disagrees across formal artifacts")
    return {
        "lake": lake,
        "system": "Low-rank Student epoch-0",
        "configuration": "k=16, μ=0 / weighted RRF e=0.05 + logsumexp",
        **_metric_values(metrics, coverage_key="positive_evidence_path_coverage@10"),
        **_ci_values(ci, baseline="raw embedding"),
        "source": f"{metric_path}; R@10 CI inputs: {q_path}",
    }


def _task_p_row(
    task_root: Path,
    summaries: dict[tuple[str, int], dict[str, str]],
    lake: str,
) -> dict[str, Any]:
    rank = 64 if lake == "entitables" else 256
    summary = summaries[(lake, rank)]
    metric_path = (
        task_root
        / f"taskP_lowrank/{lake}/k_{rank}/mu_0/final_evaluation/metrics.json"
    )
    metrics = _load(metric_path)["systems"]["student"]["metrics"]
    if abs(float(metrics["recall@10"]) - float(summary["fused_recall@10"])) > 1e-12:
        raise ValueError(f"{lake}: Task-P summary disagrees with final evaluation")
    ci = {
        "delta_mean": float(summary["fused_delta_vs_raw"]),
        "ci_low": float(summary["fused_ci_low_vs_raw"]),
        "ci_high": float(summary["fused_ci_high_vs_raw"]),
    }
    return {
        "lake": lake,
        "system": "Best r7 Task P low-rank",
        "configuration": (
            f"k={rank}, μ=0, epoch={summary['epoch']} / "
            "weighted RRF e=0.05 + logsumexp"
        ),
        **_metric_values(metrics, coverage_key="positive_evidence_path_coverage@10"),
        **_ci_values(ci, baseline="raw embedding"),
        "source": f"{metric_path}; CI: {task_root / 'taskP_lowrank/k_sensitivity.csv'}",
    }


def _task_q_row(task_root: Path, lake: str) -> dict[str, Any]:
    if lake == "entitables":
        q_path = (
            task_root
            / "taskQ_joint_selection/entitables/lowrank_k_256_mu_0/"
            "selected_epoch_002/metrics.json"
        )
        checkpoint = "k=256, μ=0, epoch=2"
    else:
        q_path = task_root / "taskQ_joint_selection/wdc/lowrank_k_256_mu_0/metrics.json"
        checkpoint = "k=256, μ=0, epoch=1"
    payload = _load(q_path)
    selected = str(payload["selected"])
    record = payload["metrics"][selected]
    return {
        "lake": lake,
        "system": "Best evaluated r7 Task Q",
        "configuration": f"{checkpoint} / {selected}",
        **_metric_values(
            record["metrics"],
            coverage_key="positive_evidence_path_coverage@10",
        ),
        **_ci_values(record["delta_vs_r5"], baseline="r5 KD Student (final)"),
        "source": str(q_path),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _ci_text(row: dict[str, Any]) -> str:
    if row["delta"] is None:
        return "reference"
    return (
        f"vs {row['ci_baseline']}: {_percent(row['delta'])} "
        f"[{_percent(row['ci_low'])}, {_percent(row['ci_high'])}]"
    )


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage-1 r7 two-lake integration table (gate not met)",
        "",
        "This is the final auditable r7 summary table, but not a passing `FINAL.md`: "
        "EntiTables remains below the r5 fused anchor, while WDC only matches it.",
        "",
        "Recall and coverage are percentages. MRR@50 is reported on the 0–1 scale. "
        "All intervals are paired bootstrap 95% CIs with 10,000 iterations and seed 13.",
        "",
    ]
    for lake in ("entitables", "wdc"):
        lines.extend(
            [
                f"## {lake}",
                "",
                "| System | Configuration | R@10 | R@20 | R@30 | R@40 | R@50 | MRR@50 | Coverage@10 | Paired R@10 comparison |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in (item for item in rows if item["lake"] == lake):
            lines.append(
                f"| {row['system']} | `{row['configuration']}` | "
                f"{_percent(row['recall@10'])} | {_percent(row['recall@20'])} | "
                f"{_percent(row['recall@30'])} | {_percent(row['recall@40'])} | "
                f"{_percent(row['recall@50'])} | {row['mrr@50']:.4f} | "
                f"{_percent(row['coverage@10'])} | {_ci_text(row)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation and audit notes",
            "",
            "- The r5 online reranker is a genuine additional direct reranking system on EntiTables; on WDC it is an explicit no-op and duplicates the r5 KD row.",
            "- Task Q covers four retrieval-side combinations (2 fusion × 2 aggregation). Relation weight and KD tau remain fixed by the checkpoint, so this is not the planned full 16-way joint sweep.",
            "- Epoch-0 rows use the formal full-recall evaluation for displayed metrics. Their R@10 CI is recomputed from the matching Task-Q weighted-RRF/logsumexp per-query vector and the r5 raw per-query vector; the displayed R@10 point estimate is asserted identical before writing.",
            "- Task-P intervals come from the training-time same-run paired raw comparison. Task-Q intervals are versus the r5 KD final row, as stored in each Task-Q artifact.",
            "- Independently built HNSW indices showed candidate-pool variability. Therefore paired query-level intervals remain useful uncertainty summaries, but independently rebuilt runs must not be described as strict same-candidate-pool comparisons.",
            "",
            "Machine-readable values and source paths are in `FINAL_TABLE.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task_root = args.output_root
    r5 = _load(args.r5_root / "final_metrics.json")
    summaries = _task_p_summary(task_root)
    rows: list[dict[str, Any]] = []
    for lake in ("entitables", "wdc"):
        r5_rows = _r5_rows(
            r5,
            lake,
            source_path=args.r5_root / "final_metrics.json",
        )
        rows.extend(
            [
                r5_rows[0],
                _epoch0_row(task_root, r5, lake),
                *r5_rows[1:],
                _task_p_row(task_root, summaries, lake),
                _task_q_row(task_root, lake),
            ]
        )
    output_dir = task_root / "taskS_integration"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "FINAL_TABLE.csv", rows)
    _write_markdown(output_dir / "FINAL_TABLE.md", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r7_20260831"),
    )
    parser.add_argument("--r5-root", type=Path, default=R5_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

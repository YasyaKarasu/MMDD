#!/usr/bin/env python
"""Summarize the Stage-1 r5 KD-target teacher-alpha ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta


TAUS = (0.0, 0.3, 0.7, 1.0)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "fused_recall@10": float(metrics["recall@10"]),
        "direct_recall@10": float(metrics["direct"]["recall@10"]),
        "evidence_recall@10": float(metrics["evidence"]["recall@10"]),
        "coverage@10": float(metrics["positive_evidence_path_coverage@10"]),
        "fused_per_query": list(metrics["per_query"]["fused"]["recall@10"]),
        "direct_per_query": list(metrics["per_query"]["direct"]["recall@10"]),
    }


def _evaluation_metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return _metric_view(payload["systems"]["student"]["metrics"])


def _history_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    epochs = []
    for record in payload["epochs"]:
        retrieval = record.get("dev_retrieval", {})
        epochs.append(
            {
                "epoch": int(record["epoch"]),
                "relation_drift_table_to_table": float(
                    record.get("relation_drift", {}).get("table_to_table", 0.0)
                ),
                "fused_recall@10": retrieval.get("recall@10"),
                "direct_recall@10": retrieval.get("direct", {}).get("recall@10"),
                "evidence_recall@10": retrieval.get("evidence", {}).get(
                    "recall@10"
                ),
                "gate": record.get("per_dataset_gate"),
            }
        )
    return {
        "best_epoch": int(payload["best_epoch"]),
        "gate_unsatisfied": bool(payload.get("gate_unsatisfied", False)),
        "epochs": epochs,
    }


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


def _run_paths(task_root: Path, r4_root: Path, lake: str, tau: float) -> tuple[Path, Path]:
    if tau == 0.7:
        run = (
            r4_root / "taskM_entitables_teacher" / "student_kd0.3"
            if lake == "entitables"
            else r4_root
            / "taskM_entitables_teacher"
            / "per_lake"
            / "wdc"
            / "student_kd0.3"
        )
    else:
        run = task_root / lake / f"tau_{tau:.1f}"
    return (
        run / "final_evaluation" / "metrics.json",
        run / "student_path.pt.history.json",
    )


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1296, 792
    left, right, top, bottom = 150, 55, 65, 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    styles = {
        "entitables": ((31, 119, 180, 255), "circle"),
        "wdc": ((214, 39, 40, 255), "square"),
    }
    bounds = [
        100.0 * row["delta_vs_raw"][key]
        for row in rows
        for key in ("ci_low", "ci_high")
    ]
    y_min = min(0.0, min(bounds))
    y_max = max(0.0, max(bounds))
    padding = max(1.0, 0.08 * (y_max - y_min or 1.0))
    y_min -= padding
    y_max += padding

    def x_position(tau: float) -> float:
        return left + tau * plot_width

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        y = y_position(value)
        draw.line((left, y, width - right, y), fill=(190, 190, 190, 100), width=1)
        draw.text((left - 12, y), f"{value:.1f}", fill="#333333", anchor="rm")
    zero_y = y_position(0.0)
    draw.line((left, zero_y, width - right, zero_y), fill="#444444", width=2)
    draw.line((left, top, left, height - bottom), fill="#222222", width=2)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="#222222",
        width=2,
    )
    for tau in TAUS:
        x = x_position(tau)
        draw.line((x, height - bottom, x, height - bottom + 8), fill="#222222", width=2)
        draw.text(
            (x, height - bottom + 16), f"{tau:.1f}", fill="#222222", anchor="ma"
        )

    for lake in ("entitables", "wdc"):
        selected = [row for row in rows if row["lake"] == lake]
        color, marker = styles[lake]
        upper = [
            (
                x_position(row["tau"]),
                y_position(100.0 * row["delta_vs_raw"]["ci_high"]),
            )
            for row in selected
        ]
        lower = [
            (
                x_position(row["tau"]),
                y_position(100.0 * row["delta_vs_raw"]["ci_low"]),
            )
            for row in reversed(selected)
        ]
        draw.polygon(upper + lower, fill=(*color[:3], 35))
        points = [
            (
                x_position(row["tau"]),
                y_position(100.0 * row["delta_vs_raw"]["mean"]),
            )
            for row in selected
        ]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            if marker == "circle":
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
            else:
                draw.rectangle((x - 7, y - 7, x + 7, y + 7), fill=color)

    draw.text(
        (width / 2, 24),
        "KD target Teacher coefficient attribution",
        fill="#111111",
        anchor="ma",
    )
    draw.text(
        (width / 2, height - 42),
        "KD target Teacher coefficient tau",
        fill="#222222",
        anchor="ma",
    )
    draw.text(
        (left, top - 16),
        "Fused R@10 delta vs raw (percentage points)",
        fill="#222222",
        anchor="la",
    )
    legend_y = top + 18
    for offset, (lake, label) in enumerate(
        (("entitables", "EntiTables"), ("wdc", "WDC"))
    ):
        color, _marker = styles[lake]
        x = width - right - 230 + offset * 120
        draw.line((x, legend_y, x + 30, legend_y), fill=color, width=5)
        draw.text(
            (x + 38, legend_y),
            label,
            fill="#222222",
            anchor="lm",
        )
    image.save(path)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def run(args: argparse.Namespace) -> dict[str, Any]:
    task_root = Path(args.task_root)
    r4_root = Path(args.r4_root)
    output_dir = Path(args.output_dir or args.task_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    taskk = _read_json(r4_root / "taskK_per_lake_training" / "metrics.json")
    taskj = _read_json(r4_root / "taskJ_per_lake_baselines" / "metrics.json")
    rows = []
    mechanisms = {}
    for lake in ("entitables", "wdc"):
        raw = _metric_view(taskj["lakes"][lake]["systems"]["raw"]["metrics"])
        supervised = _metric_view(
            taskk["runs"][f"{lake}_supervised"]["final_evaluation"]["metrics"]
        )
        by_tau = {}
        for tau in TAUS:
            evaluation_path, history_path = _run_paths(task_root, r4_root, lake, tau)
            metrics = _evaluation_metrics(evaluation_path)
            history = _history_summary(history_path)
            row = {
                "lake": lake,
                "tau": tau,
                **{
                    key: value
                    for key, value in metrics.items()
                    if not key.endswith("_per_query")
                },
                "best_epoch": history["best_epoch"],
                "gate_unsatisfied": history["gate_unsatisfied"],
                "delta_vs_raw": _delta(
                    metrics,
                    raw,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                ),
                "delta_vs_supervised": _delta(
                    metrics,
                    supervised,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed,
                ),
                "epochs": history["epochs"],
            }
            rows.append(row)
            by_tau[tau] = metrics
        mechanisms[lake] = {
            "cosine_anchoring_vs_supervised": _delta(
                by_tau[0.0],
                supervised,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            "teacher_residual_at_tau_0.7_vs_cosine": _delta(
                by_tau[0.7],
                by_tau[0.0],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            "deployed_kd_vs_supervised": _delta(
                by_tau[0.7],
                supervised,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            "pure_teacher_vs_cosine": _delta(
                by_tau[1.0],
                by_tau[0.0],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
        }

    payload = {
        "format_version": 1,
        "parameters": {
            "taus": list(TAUS),
            "distillation_weight": 0.3,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "primary_metric": "fused recall@10",
        },
        "rows": rows,
        "mechanism_decomposition": mechanisms,
    }
    write_json(output_dir / "metrics.json", payload)
    with (output_dir / "tau_gain_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lake",
                "tau",
                "fused_recall_at_10",
                "direct_recall_at_10",
                "evidence_recall_at_10",
                "coverage_at_10",
                "delta_vs_raw",
                "ci_low",
                "ci_high",
                "best_epoch",
                "gate_unsatisfied",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["lake"],
                    row["tau"],
                    row["fused_recall@10"],
                    row["direct_recall@10"],
                    row["evidence_recall@10"],
                    row["coverage@10"],
                    row["delta_vs_raw"]["mean"],
                    row["delta_vs_raw"]["ci_low"],
                    row["delta_vs_raw"]["ci_high"],
                    row["best_epoch"],
                    row["gate_unsatisfied"],
                ]
            )
    _plot(rows, output_dir / "tau_gain_curve.png")

    lines = [
        "# Task 1: KD target tau attribution",
        "",
        "Primary metric: fused R@10 with weighted RRF (evidence weight 0.05).",
        "",
        "| Lake | tau | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 | Delta vs raw / 95% CI | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        delta = row["delta_vs_raw"]
        lines.append(
            f"| {row['lake']} | {row['tau']:.1f} | "
            f"{_percent(row['fused_recall@10'])} | "
            f"{_percent(row['direct_recall@10'])} | "
            f"{_percent(row['evidence_recall@10'])} | "
            f"{_percent(row['coverage@10'])} | "
            f"{_percent(delta['mean'])} "
            f"[{_percent(delta['ci_low'])}, {_percent(delta['ci_high'])}] | "
            f"{row['best_epoch']} |"
        )
    lines.extend(["", "## Mechanism decomposition", ""])
    for lake, values in mechanisms.items():
        lines.append(f"### {lake}")
        lines.append("")
        for label, delta in values.items():
            lines.append(
                f"- {label}: {_percent(delta['mean'])} "
                f"[{_percent(delta['ci_low'])}, {_percent(delta['ci_high'])}]"
            )
        lines.append("")
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--r4-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

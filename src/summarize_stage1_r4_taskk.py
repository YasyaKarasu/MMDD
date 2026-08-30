#!/usr/bin/env python
"""Summarize per-lake Task-K supervised and ensemble-KD training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _run_spec(value: str) -> tuple[str, str, str, float, Path]:
    name, separator, raw_values = value.partition("=")
    values = raw_values.split(",") if separator else []
    if not name or len(values) != 4:
        raise argparse.ArgumentTypeError(
            "run must use NAME=LABEL,LAKE,KD_WEIGHT,RUN_DIR"
        )
    return name, values[0], values[1], float(values[2]), Path(values[3])


def _epoch_metrics(record: dict[str, Any]) -> dict[str, Any]:
    retrieval = record["dev_retrieval"]
    gate_rows = record["gate"].get("per_dataset", [])
    gate = gate_rows[0] if gate_rows else None
    return {
        "epoch": int(record["epoch"]),
        "direct_recall@10": float(retrieval["direct"]["recall@10"]),
        "evidence_recall@10": float(retrieval["evidence"]["recall@10"]),
        "fused_recall@10": float(retrieval["recall@10"]),
        "direct_recall@50": float(retrieval["direct"]["recall@50"]),
        "fused_recall@50": float(retrieval["recall@50"]),
        "relation_drift_table_to_table": float(
            record["relation_drift"]["table_to_table"]
        ),
        "eligible": bool(record["gate"]["eligible"]),
        "direct_recall@10_vs_raw": gate["bootstrap"] if gate else None,
    }


def _summarize_run(
    spec: tuple[str, str, str, float, Path],
    taskj: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    name, label, lake, kd_weight, directory = spec
    history = json.loads(
        (directory / "student_path.pt.history.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        (directory / "student_path.pt.selection.json").read_text(encoding="utf-8")
    )
    final_evaluation = json.loads(
        (directory / "final_evaluation" / "metrics.json").read_text(encoding="utf-8")
    )["systems"]["student"]
    epochs = [_epoch_metrics(record) for record in history["epochs"]]
    selected = next(
        row for row in epochs if row["epoch"] == int(selection["best_epoch"])
    )
    raw = taskj["lakes"][lake]["systems"]["raw"]["metrics"]
    raw_metrics = {
        "direct_recall@10": float(raw["direct"]["recall@10"]),
        "fused_recall@10": float(raw["recall@10"]),
    }
    training_raw = history["best_metrics"]["raw_embedding"]
    if abs(training_raw["direct"]["recall@10"] - raw_metrics["direct_recall@10"]) > 1e-12:
        raise ValueError(f"{name}: training and Task-J raw direct baselines differ")
    return name, {
        "label": label,
        "lake": lake,
        "kd_weight": kd_weight,
        "directory": str(directory.resolve()),
        "best_epoch": int(selection["best_epoch"]),
        "stop_reason": selection["stop_reason"],
        "gate_unsatisfied": bool(selection["gate_unsatisfied"]),
        "best_checkpoint": selection["best_checkpoint"],
        "best_checkpoint_sha256": selection["best_checkpoint_sha256"],
        "raw": raw_metrics,
        "selected": selected,
        "final_evaluation": final_evaluation,
        "epochs": epochs,
    }


def _ci(value: dict[str, Any] | None) -> str:
    if value is None:
        return "n/a"
    return (
        f"{value['mean']:+.2%} "
        f"[{value['ci_low']:+.2%}, {value['ci_high']:+.2%}]"
    )


def _decisions(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    entitables = [run for run in runs.values() if run["lake"] == "entitables"]
    supervised = next(run for run in entitables if run["kd_weight"] == 0)
    best_kd = max(
        (run for run in entitables if run["kd_weight"] > 0),
        key=lambda run: run["selected"]["direct_recall@10"],
    )
    raw_direct = supervised["raw"]["direct_recall@10"]
    chain = (
        best_kd["selected"]["direct_recall@10"]
        > supervised["selected"]["direct_recall@10"]
        > raw_direct
    )
    wdc = [run for run in runs.values() if run["lake"] == "wdc"]
    return {
        "entitables_distillation_chain": chain,
        "entitables_best_kd_label": best_kd["label"],
        "entitables_best_kd_checkpoint": best_kd["best_checkpoint"],
        "wdc_all_gates_satisfied": all(not run["gate_unsatisfied"] for run in wdc),
        "task_m_required": (
            not chain
            and best_kd["selected"]["direct_recall@10"]
            - supervised["selected"]["direct_recall@10"]
            < 0.02
        ),
    }


def _select_by_lake(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for lake in sorted({run["lake"] for run in runs.values()}):
        candidates = [
            (name, run)
            for name, run in runs.items()
            if run["lake"] == lake and not run["gate_unsatisfied"]
        ]
        if not candidates:
            candidates = [
                (name, run) for name, run in runs.items() if run["lake"] == lake
            ]
        name, best = max(
            candidates,
            key=lambda item: item[1]["selected"]["fused_recall@10"],
        )
        selected[lake] = {
            "run": name,
            "label": best["label"],
            "best_epoch": best["best_epoch"],
            "checkpoint": best["best_checkpoint"],
            "checkpoint_sha256": best["best_checkpoint_sha256"],
            "selection": str(
                Path(best["directory"]) / "student_path.pt.selection.json"
            ),
            "gate_unsatisfied": best["gate_unsatisfied"],
            "fused_recall@10": best["selected"]["fused_recall@10"],
        }
    return selected


def _markdown(payload: dict[str, Any]) -> str:
    rows = [
        "# Task K: per-lake Student training and ensemble-KD",
        "",
        "All runs use a fresh lake-specific PCA-1024 initialization, frozen "
        "projection, path-only in-batch training, mu=0.1, and a paired lake-local "
        "CI gate against raw direct retrieval.",
        "",
        "| Run | KD | Raw direct R@10 | Selected direct R@10 | Selected fused R@10 | "
        "Selected fused R@50 | Epoch | Direct delta / 95% CI | Gate | R(table,table) drift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for run in payload["runs"].values():
        selected = run["selected"]
        rows.append(
            f"| {run['label']} | {run['kd_weight']:.1f} | "
            f"{run['raw']['direct_recall@10']:.2%} | "
            f"{selected['direct_recall@10']:.2%} | "
            f"{selected['fused_recall@10']:.2%} | "
            f"{selected['fused_recall@50']:.2%} | {run['best_epoch']} | "
            f"{_ci(selected['direct_recall@10_vs_raw'])} | "
            f"{'fallback' if run['gate_unsatisfied'] else 'pass'} | "
            f"{selected['relation_drift_table_to_table']:.4f} |"
        )
    rows.extend(
        [
            "",
            "## Final full-k evaluation",
            "",
            "| Run | R@10 | R@20 | R@30 | R@40 | R@50 | MRR@50 | Coverage@10 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in payload["runs"].values():
        metrics = run["final_evaluation"]["metrics"]
        rows.append(
            f"| {run['label']} | {metrics['recall@10']:.2%} | "
            f"{metrics['recall@20']:.2%} | {metrics['recall@30']:.2%} | "
            f"{metrics['recall@40']:.2%} | {metrics['recall@50']:.2%} | "
            f"{metrics['mrr@50']:.4f} | "
            f"{metrics['positive_evidence_path_coverage@10']:.2%} |"
        )
    rows.extend(["", "## Epoch trajectories", ""])
    for run in payload["runs"].values():
        rows.extend(
            [
                f"### {run['label']}",
                "",
                "| Epoch | Direct R@10 | Evidence R@10 | Fused R@10 | "
                "Direct delta / 95% CI | Eligible | R(table,table) drift |",
                "| ---: | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for epoch in run["epochs"]:
            rows.append(
                f"| {epoch['epoch']} | {epoch['direct_recall@10']:.2%} | "
                f"{epoch['evidence_recall@10']:.2%} | "
                f"{epoch['fused_recall@10']:.2%} | "
                f"{_ci(epoch['direct_recall@10_vs_raw'])} | "
                f"{'yes' if epoch['eligible'] else 'no'} | "
                f"{epoch['relation_drift_table_to_table']:.4f} |"
            )
        rows.append("")
    decision = payload["decisions"]
    rows.extend(
        [
            "## Per-lake selection",
            "",
            "| Lake | Selected run | Epoch | Fused R@10 | Gate |",
            "| --- | --- | ---: | ---: | --- |",
            *[
                f"| {lake} | {selected['label']} | {selected['best_epoch']} | "
                f"{selected['fused_recall@10']:.2%} | "
                f"{'fallback' if selected['gate_unsatisfied'] else 'pass'} |"
                for lake, selected in payload["selected_by_lake"].items()
            ],
            "",
            "## Decision",
            "",
            "- EntiTables distillation chain: "
            + ("established." if decision["entitables_distillation_chain"] else "not established."),
            f"- Best EntiTables KD run: {decision['entitables_best_kd_label']}.",
            "- WDC lake-local gates: "
            + ("all satisfied." if decision["wdc_all_gates_satisfied"] else "at least one fallback."),
            "- Task M: "
            + ("required by the decision tree." if decision["task_m_required"] else "not triggered by Task K."),
        ]
    )
    return "\n".join(rows) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    taskj = json.loads(args.taskj_metrics.read_text(encoding="utf-8"))
    runs = dict(_summarize_run(spec, taskj) for spec in args.run)
    payload = {
        "format_version": 1,
        "runs": runs,
        "selected_by_lake": _select_by_lake(runs),
        "decisions": _decisions(runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "RESULTS.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskj-metrics", type=Path, required=True)
    parser.add_argument("--run", type=_run_spec, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

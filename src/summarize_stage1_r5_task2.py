#!/usr/bin/env python
"""Summarize the Stage-1 r5 WDC mixed-negative Teacher ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    return paired_bootstrap_delta(
        candidate["per_query"]["recall@10"],
        reference["per_query"]["recall@10"],
        iterations=iterations,
        seed=seed,
    )


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _read(Path(args.baseline_diagnostic))
    mixed = _read(Path(args.mixed_diagnostic))
    if baseline["corpus_sha256"] != mixed["corpus_sha256"]:
        raise ValueError("Teacher diagnostics do not use the same corpus")
    if baseline["raw_top_k"] != mixed["raw_top_k"]:
        raise ValueError("Teacher diagnostics do not use the same raw candidate width")
    raw = mixed["raw_direct"]
    baseline_teacher = baseline["teacher_reranked"]
    mixed_teacher = mixed["teacher_reranked"]
    baseline_vs_raw = _comparison(
        baseline_teacher,
        raw,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    mixed_vs_raw = _comparison(
        mixed_teacher,
        raw,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    mixed_vs_baseline = _comparison(
        mixed_teacher,
        baseline_teacher,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    gate_threshold = float(raw["recall@10"]) - args.raw_tolerance
    gate_pass = float(mixed_teacher["recall@10"]) >= gate_threshold
    payload: dict[str, Any] = {
        "format_version": 1,
        "same_candidate_pool": {
            "corpus_sha256": mixed["corpus_sha256"],
            "raw_top_k": mixed["raw_top_k"],
            "queries": mixed["queries"],
        },
        "raw_recall@10": float(raw["recall@10"]),
        "baseline_wdc_negative_teacher": {
            "recall@10": float(baseline_teacher["recall@10"]),
            "vs_raw": baseline_vs_raw,
            "checkpoint": baseline["teacher_checkpoint"],
            "checkpoint_sha256": baseline["teacher_checkpoint_sha256"],
        },
        "mixed_negative_teacher": {
            "recall@10": float(mixed_teacher["recall@10"]),
            "vs_raw": mixed_vs_raw,
            "vs_baseline_teacher": mixed_vs_baseline,
            "checkpoint": mixed["teacher_checkpoint"],
            "checkpoint_sha256": mixed["teacher_checkpoint_sha256"],
        },
        "gate": {
            "criterion": f"teacher recall@10 >= raw recall@10 - {args.raw_tolerance:g}",
            "threshold": gate_threshold,
            "passed": gate_pass,
            "decision": "teacher_repair_succeeded" if gate_pass else "teacher_no_op_negative_asset",
        },
    }
    write_json(output_dir / "metrics.json", payload)
    with (output_dir / "same_pool_teacher_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["system", "recall_at_10", "delta_vs_raw", "ci_low", "ci_high"]
        )
        writer.writerow(["raw", raw["recall@10"], 0.0, 0.0, 0.0])
        for label, metrics, delta in (
            ("wdc_negative_teacher", baseline_teacher, baseline_vs_raw),
            ("mixed_negative_teacher", mixed_teacher, mixed_vs_raw),
        ):
            writer.writerow(
                [
                    label,
                    metrics["recall@10"],
                    delta["mean"],
                    delta["ci_low"],
                    delta["ci_high"],
                ]
            )

    lines = [
        "# Task 2: WDC Teacher negative-source ablation",
        "",
        "All rows rerank the same WDC-local raw top-100 candidate pool.",
        "",
        "| System | R@10 | Delta vs raw / 95% CI |",
        "| --- | ---: | --- |",
        f"| Raw | {_percent(raw['recall@10'])} | reference |",
        "| WDC-negative Teacher | "
        f"{_percent(baseline_teacher['recall@10'])} | "
        f"{_percent(baseline_vs_raw['mean'])} "
        f"[{_percent(baseline_vs_raw['ci_low'])}, "
        f"{_percent(baseline_vs_raw['ci_high'])}] |",
        "| Mixed-negative Teacher | "
        f"{_percent(mixed_teacher['recall@10'])} | "
        f"{_percent(mixed_vs_raw['mean'])} "
        f"[{_percent(mixed_vs_raw['ci_low'])}, "
        f"{_percent(mixed_vs_raw['ci_high'])}] |",
        "",
        "Mixed-negative minus WDC-negative Teacher: "
        f"{_percent(mixed_vs_baseline['mean'])} "
        f"[{_percent(mixed_vs_baseline['ci_low'])}, "
        f"{_percent(mixed_vs_baseline['ci_high'])}].",
        "",
        f"Gate ({payload['gate']['criterion']}): **{gate_pass}**.",
        f"Decision: **{payload['gate']['decision']}**.",
        "",
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-diagnostic", required=True)
    parser.add_argument("--mixed-diagnostic", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-tolerance", type=float, default=0.03)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

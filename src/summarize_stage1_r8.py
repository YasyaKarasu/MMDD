#!/usr/bin/env python
"""Integrate Stage-1 r8 results and write the final per-lake report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


def final_choice(r2: dict[str, Any]) -> str:
    """Choose whether r8 replaces the r5 retrieval configuration for one lake."""

    selected = r2["metrics"][r2["selected"]]
    if not (
        selected["r5_point_anchor_satisfied"]
        and selected["r5_ci_gate_satisfied"]
    ):
        return "r5"
    if r2["lake"] == "entitables" and not r2["entitables_double_win"]:
        return "r5"
    return "r8"


def _per_query_recall(metrics: dict[str, Any]) -> list[float]:
    per_query = metrics["per_query"]
    if "fused" in per_query:
        return per_query["fused"]["recall@10"]
    return per_query["recall@10"]


def _delta_vs_raw(metrics: dict[str, Any], raw: dict[str, Any]) -> dict[str, float]:
    return paired_bootstrap_delta(
        _per_query_recall(metrics),
        _per_query_recall(raw),
        iterations=10_000,
        seed=13,
    )


def _size_gib(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 2**30


def _metric_row(
    label: str,
    metrics: dict[str, Any],
    raw: dict[str, Any],
    *,
    coverage: bool,
) -> dict[str, Any]:
    return {
        "label": label,
        **{f"recall@{k}": float(metrics[f"recall@{k}"]) for k in (10, 20, 30, 40, 50)},
        "mrr@50": float(metrics["mrr@50"]),
        "coverage@10": (
            float(metrics["positive_evidence_path_coverage@10"])
            if coverage
            else None
        ),
        "delta_vs_raw": None if metrics is raw else _delta_vs_raw(metrics, raw),
    }


def _table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            "| System | R@10 | R@20 | R@30 | R@40 | R@50 | MRR@50 | Coverage@10 | Δ R@10 vs raw / 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        delta = row["delta_vs_raw"]
        delta_text = (
            "reference"
            if delta is None
            else f"{delta['delta_mean']:+.2%} [{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}]"
        )
        coverage = (
            "n/a" if row["coverage@10"] is None else f"{row['coverage@10']:.2%}"
        )
        lines.append(
            f"| {row['label']} | {row['recall@10']:.2%} | "
            f"{row['recall@20']:.2%} | {row['recall@30']:.2%} | "
            f"{row['recall@40']:.2%} | {row['recall@50']:.2%} | "
            f"{row['mrr@50']:.4f} | {coverage} | {delta_text} |"
        )


def _lake_payload(
    root: Path,
    output_root: Path,
    lake: str,
) -> dict[str, Any]:
    r2_path = output_root / f"taskR2_full_r_task_q/{lake}/metrics.json"
    r2 = json.loads(r2_path.read_text(encoding="utf-8"))
    r5_path = (
        root
        / f"work/stage1_optimization_r5_20260829/task4_final/final_evaluation/{lake}/metrics.json"
    )
    r5 = json.loads(r5_path.read_text(encoding="utf-8"))
    raw = r5["systems"]["raw"]["metrics"]
    choice = final_choice(r2)
    selected_r2 = r2["metrics"][r2["selected"]]["metrics"]
    student = r5["systems"]["student"]["metrics"] if choice == "r5" else selected_r2
    if lake == "entitables":
        reranker = r5["systems"]["student_ensemble"]["metrics"]
        reranker_label = "Student + online reranker"
    else:
        reranker = student
        reranker_label = "Student + online reranker (no-op)"
    selection = json.loads(
        (
            root
            / "work/stage1_optimization_r5_20260829/task4_final/selection.json"
        ).read_text(encoding="utf-8")
    )["selected"][lake]
    raw_index = (
        root
        / f"work/stage1_optimization_r4_20260829/taskJ_per_lake_baselines/epoch0/{lake}/raw_index"
    )
    student_selection = json.loads(
        Path(selection["student_selection"]).read_text(encoding="utf-8")
    )
    return {
        "lake": lake,
        "choice": choice,
        "r2": r2,
        "r2_path": r2_path,
        "r5_path": r5_path,
        "raw": raw,
        "student": student,
        "reranker": reranker,
        "reranker_label": reranker_label,
        "rows": [
            _metric_row("Raw embedding", raw, raw, coverage=True),
            _metric_row("Full-R KD Student (final)", student, raw, coverage=True),
            _metric_row(reranker_label, reranker, raw, coverage=lake == "wdc"),
        ],
        "selection": selection,
        "raw_index_gib": _size_gib(raw_index),
        "student_index_gib": _size_gib(Path(student_selection["best_index"])),
        "r5_timing": {
            "raw": r5["systems"]["raw"]["timing"][
                "average_seconds_per_query_per_k"
            ],
            "student": r5["systems"]["student"]["timing"][
                "average_seconds_per_query_per_k"
            ],
            "reranker": (
                r5["systems"]["student_ensemble"]["timing"][
                    "average_seconds_per_query_per_k"
                ]
                if lake == "entitables"
                else r5["systems"]["student"]["timing"][
                    "average_seconds_per_query_per_k"
                ]
            ),
        },
    }


def _final_markdown(lakes: dict[str, dict[str, Any]], r1: dict[str, Any]) -> str:
    enti = lakes["entitables"]
    wdc = lakes["wdc"]
    lines = [
        "# Stage-1 round 8: final per-lake configuration",
        "",
        "## Decision",
        "",
        "- EntiTables retains the r5 default. The best r8 full-R recombination "
        "reaches 37.95% fused R@10 but 9.81% evidence R@10, narrowly missing "
        "the prespecified 10% double-win gate.",
        "- WDC adopts the retrieval-only `weighted_rrf_e0.05 + "
        "topk_sum_edges_none` recombination on the unchanged r5 full-R "
        "checkpoint. It reaches 65.35% fused R@10, +0.99 points versus r5 "
        "with paired 95% CI [0.00, 2.48] points.",
        "- No new training, Teacher change, feature change, or index rebuild is "
        "part of the r8 final choice.",
        "",
    ]
    for lake, payload in (("EntiTables", enti), ("WDC", wdc)):
        lines.extend([f"## {lake}", ""])
        _table(lines, payload["rows"])
        r2 = payload["r2"]
        selected = r2["metrics"][r2["selected"]]["metrics"]
        if lake == "EntiTables":
            config = "weighted_rrf_e0.05 + logsumexp_edges_none (r5 retained)"
        else:
            config = r2["selected"].replace("__", " + ")
        lines.extend(
            [
                "",
                f"- Retrieval configuration: `{config}`; relation parameterization: "
                f"full-R; tau={payload['selection']['tau']:.1f}.",
                f"- Teacher role: {payload['selection']['online_teacher_provenance']}.",
                f"- Raw index: {payload['raw_index_gib']:.3f} GiB; Student index: "
                f"{payload['student_index_gib']:.3f} GiB.",
                f"- Existing r5 single-configuration latency (raw/student/reranker): "
                f"{payload['r5_timing']['raw'] * 1000:.2f}/"
                f"{payload['r5_timing']['student'] * 1000:.2f}/"
                f"{payload['r5_timing']['reranker'] * 1000:.2f} ms/query/k. "
                f"The r8 four-configuration sweep took "
                f"{r2['timing']['seconds_per_query_per_k'] * 1000:.2f} ms/query/k "
                "and is not presented as deployment latency.",
                f"- R2 best evaluated candidate (including non-adopted candidates): "
                f"fused={selected['recall@10']:.2%}, direct="
                f"{selected['direct']['recall@10']:.2%}, evidence="
                f"{selected['evidence']['recall@10']:.2%}, coverage="
                f"{selected['positive_evidence_path_coverage@10']:.2%}.",
                "",
            ]
        )

    curve = {str(row["requested_k"]): row["recall"] for row in r1["curve"]}
    q = r1["rank_quantiles"]
    lines.extend(
        [
            "## WDC saturation decision",
            "",
            f"Exact brute-force raw direct recall in the current "
            f"{r1['table_corpus_size']:,}-table pool is R@50={curve['50']:.2%}, "
            f"R@100={curve['100']:.2%}, R@200={curve['200']:.2%}, and "
            f"R@500={curve['500']:.2%}. Positive-target ranks have median "
            f"{q['median']:.1f}, P75 {q['p75']:.1f}, and P95 {q['p95']:.1f}.",
            "",
            "This falls into a gap between the plan's two explicit rules: R@50 "
            "is below the 85% saturation gate, while R@100 and R@200 are not "
            "below 80%. The defensible conclusion is an intermediate current-pool "
            "curve with substantial deep-retrieval headroom, not saturation of the "
            "full WebTable space. No scale-up experiment was launched because Task "
            "R3 has no implementation section or experimental specification in the "
            "r8 plan.",
            "",
            "## Efficiency and retained negative results",
            "",
            "The paper/default Students remain full-R (9,437,184 relation "
            "parameters). The r7 residual low-rank form uses `18,432 × k` relation "
            "parameters—32× fewer at k=16, 8× fewer at k=64, and 2× fewer at "
            "k=256—but stays an efficiency ablation because it misses the EntiTables "
            "quality anchor.",
            "",
            "Retained negative/mechanism results: low-rank does not remove WDC "
            "drift; KD temperatures {0.5, 2, 4} do not beat T=1; larger table token "
            "budgets do not improve the formal result; and the score-scale mismatch "
            "hypothesis is not supported. WDC online Teacher reranking remains a no-op.",
            "",
            "## Evaluation protocol",
            "",
            "`recall_ks={10,20,30,40,50}`, gamma=10, gamma_e=2, text+image "
            "evidence, paired bootstrap with 10,000 iterations and seed 13, and "
            "lake CI tolerance 0.02. R2's four configurations share one candidate "
            "pool per query/k within each Student evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root
    r1_path = output_root / "taskR1_saturation_curve/metrics.json"
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))
    lakes = {
        lake: _lake_payload(root, output_root, lake)
        for lake in ("entitables", "wdc")
    }
    final = _final_markdown(lakes, r1)
    integration = output_root / "taskR4_integration"
    integration.mkdir(parents=True, exist_ok=True)
    (integration / "FINAL.md").write_text(final, encoding="utf-8")
    (output_root / "FINAL.md").write_text(final, encoding="utf-8")

    selected_summary = {
        lake: {
            "choice": payload["choice"],
            "configuration": (
                "weighted_rrf_e0.05__logsumexp_edges_none"
                if lake == "entitables"
                else payload["r2"]["selected"]
            ),
            "student_checkpoint": payload["r2"]["student_checkpoint"],
            "student_checkpoint_sha256": payload["r2"][
                "student_checkpoint_sha256"
            ],
            "student_metrics": payload["student"],
        }
        for lake, payload in lakes.items()
    }
    summary = {
        "format_version": 1,
        "r1_source": str(r1_path.resolve()),
        "r1_decision": r1["decision"],
        "r2_sources": {
            lake: str(payload["r2_path"].resolve()) for lake, payload in lakes.items()
        },
        "final": selected_summary,
        "task_r3": {
            "executed": False,
            "reason": "referenced by the plan but no Task R3 specification is present",
        },
    }
    (integration / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    task_r3 = output_root / "taskR3_not_specified"
    task_r3.mkdir(parents=True, exist_ok=True)
    (task_r3 / "RESULTS.md").write_text(
        "# Stage-1 r8 Task R3\n\n"
        "Status: **not executed**.\n\n"
        "The r8 plan references Task R3 in the R1 decision rule and output "
        "requirements, but contains no Task R3 implementation section: no "
        "candidate-pool source, target scale, data construction rule, command, "
        "comparison protocol, compute budget, or acceptance threshold is defined. "
        "R1 also lands in the plan's threshold gap (`R@50 < 85%`, while both "
        "`R@100` and `R@200` are above 80%). Starting a scale-up intervention "
        "would therefore invent both the experiment and its trigger. No scale-up "
        "training or corpus mutation was performed.\n",
        encoding="utf-8",
    )
    results = [
        "# Stage-1 round 8 results",
        "",
        f"- R1: `{r1['decision']}` in the current {r1['table_corpus_size']:,}-table "
        "pool; R@50=79.70%, R@200=85.15%, R@1000=97.52%.",
        "- R2 EntiTables: best fused 37.95%, evidence 9.81%; strict 10% "
        "evidence gate failed, so retain r5.",
        "- R2 WDC: selected fused 65.35%, evidence 43.56%; both point and CI "
        "gates pass, so adopt the retrieval-only recombination.",
        "- R3: not executed because the plan references it but does not define its "
        "inputs, intervention, scale, or acceptance rule.",
        "- R4: report integration completed; no new training was required.",
        "",
        "See `taskR1_saturation_curve/`, `taskR2_full_r_task_q/`, and `FINAL.md` "
        "for detailed auditable artifacts.",
        "",
    ]
    results_text = "\n".join(results)
    (integration / "RESULTS.md").write_text(results_text, encoding="utf-8")
    (output_root / "RESULTS.md").write_text(results_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_optimization_r8_20260831"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python
"""Build the final Stage-1 round-4 report from completed task artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.significance import paired_bootstrap_delta


def _view(result: dict[str, Any], channel: str | None = None) -> dict[str, Any]:
    values = result["metrics"]["per_query"]
    return values[channel] if channel else values


def _comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    iterations: int,
    seed: int,
    candidate_channel: str | None = None,
    reference_channel: str | None = None,
) -> dict[str, Any]:
    candidate_values = _view(candidate, candidate_channel)
    reference_values = _view(reference, reference_channel)
    return paired_bootstrap_delta(
        candidate_values["recall@10"],
        reference_values["recall@10"],
        iterations=iterations,
        seed=seed,
    )


def _ci(value: dict[str, Any]) -> str:
    return (
        f"{value['mean']:+.2%} "
        f"[{value['ci_low']:+.2%}, {value['ci_high']:+.2%}]"
    )


def _run_for(
    taskk: dict[str, Any], lake: str, *, kd: bool
) -> tuple[str, dict[str, Any]]:
    candidates = [
        (name, run)
        for name, run in taskk["runs"].items()
        if run["lake"] == lake and (run["kd_weight"] > 0) == kd
    ]
    if not candidates:
        raise ValueError(f"Task K has no {'KD' if kd else 'supervised'} run for {lake}")
    return max(
        candidates,
        key=lambda item: item[1]["selected"]["direct_recall@10"],
    )


def _effective_teacher(
    lake: str,
    taskj_lake: dict[str, Any],
    taskm: dict[str, Any] | None,
) -> dict[str, Any]:
    taskm_lake = None
    if taskm is not None:
        taskm_lake = taskm.get("lakes", {}).get(lake)
        if taskm_lake is None and lake == "entitables":
            taskm_lake = taskm
    if taskm_lake is not None and taskm_lake["teacher"]["gate_pass"]:
        label = taskm_lake.get("label", "EntiTables")
        return {
            "checkpoint": taskm_lake["teacher"]["checkpoint"],
            "checkpoint_sha256": taskm_lake["teacher"]["checkpoint_sha256"],
            "source": f"Task M {label}-only Teacher",
        }
    provenance = taskj_lake["provenance"]
    return {
        "checkpoint": provenance["teacher_checkpoint"],
        "checkpoint_sha256": provenance["teacher_checkpoint_sha256"],
        "source": "Task J mixed Teacher",
    }


def _final_rows(
    lake: str,
    taskj: dict[str, Any],
    taskk: dict[str, Any],
    taskl: dict[str, Any],
    taskn: dict[str, Any],
    *,
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    taskj_lake = taskj["lakes"][lake]
    taskl_lake = taskl["lakes"][lake]
    _supervised_name, supervised = _run_for(taskk, lake, kd=False)
    _kd_name, kd = _run_for(taskk, lake, kd=True)
    raw = taskj_lake["systems"]["raw"]
    raw_ensemble = taskl_lake["systems"]["raw_ensemble"]
    rows = [
        {"name": "Raw embedding", "result": raw, "comparison": None},
        {
            "name": "Student PCA-1024 epoch 0",
            "result": taskj_lake["systems"]["student"],
            "comparison": _comparison(
                taskj_lake["systems"]["student"], raw, iterations=iterations,
                seed=seed, candidate_channel="fused", reference_channel="fused",
            ),
        },
        {
            "name": "Supervised Student",
            "result": supervised["final_evaluation"],
            "comparison": _comparison(
                supervised["final_evaluation"], raw, iterations=iterations,
                seed=seed, candidate_channel="fused", reference_channel="fused",
            ),
        },
        {
            "name": kd["label"],
            "result": kd["final_evaluation"],
            "comparison": _comparison(
                kd["final_evaluation"], raw, iterations=iterations,
                seed=seed, candidate_channel="fused", reference_channel="fused",
            ),
        },
        {
            "name": "Raw + Teacher ensemble",
            "result": raw_ensemble,
            "comparison": _comparison(
                raw_ensemble, raw,
                iterations=iterations, seed=seed, reference_channel="direct",
            ),
        },
        {
            "name": "Task-K selected Student",
            "result": taskl_lake["systems"]["student"],
            "comparison": _comparison(
                taskl_lake["systems"]["student"], raw, iterations=iterations,
                seed=seed, candidate_channel="fused", reference_channel="fused",
            ),
        },
        {
            "name": "Task-K selected Student + Teacher ensemble",
            "result": taskl_lake["systems"]["student_ensemble"],
            "comparison": _comparison(
                taskl_lake["systems"]["student_ensemble"], raw,
                iterations=iterations, seed=seed, reference_channel="direct",
            ),
        },
    ]
    final_checkpoint = taskk["selected_by_lake"][lake]["checkpoint"]
    final_checkpoint_sha256 = taskk["selected_by_lake"][lake]["checkpoint_sha256"]
    final_system = taskl_lake["systems"]["student_ensemble"]
    final_index_bytes = int(taskl_lake["index_bytes"]["student"])
    if lake == "entitables" and not taskn.get("skipped", True):
        rows.extend(
            [
                {
                    "name": "Mined Student",
                    "result": taskn["systems"]["mined_student"],
                    "comparison": _comparison(
                        taskn["systems"]["mined_student"], raw,
                        iterations=iterations, seed=seed,
                        candidate_channel="fused", reference_channel="fused",
                    ),
                },
                {
                    "name": "Mined Student + Teacher ensemble",
                    "result": taskn["systems"]["mined_ensemble"],
                    "comparison": _comparison(
                        taskn["systems"]["mined_ensemble"], raw,
                        iterations=iterations, seed=seed, reference_channel="direct",
                    ),
                },
            ]
        )
        if taskn["use_mined_checkpoint"]:
            final_checkpoint = taskn["mining"]["checkpoint"]
            final_checkpoint_sha256 = taskn["mining"]["checkpoint_sha256"]
            final_system = taskn["systems"]["mined_ensemble"]
            final_index_bytes = int(taskn["mining"]["index_bytes"])
    deployment = _comparison(
        final_system,
        raw_ensemble,
        iterations=iterations,
        seed=seed,
    )
    return rows, {
        "checkpoint": final_checkpoint,
        "checkpoint_sha256": final_checkpoint_sha256,
        "index_bytes": final_index_bytes,
        "ensemble_seconds_per_query_per_k": final_system["timing"][
            "average_seconds_per_query_per_k"
        ],
        "deployment_gap": deployment,
    }


def _markdown(payload: dict[str, Any]) -> str:
    ks = payload["parameters"]["recall_ks"]
    rows = [
        "# Stage-1 round 4: per-lake final",
        "",
        "The framework is shared, but PCA, Student relations, corpus, and ANN index "
        "are instantiated independently for each data lake.",
    ]
    for lake in payload["lakes"].values():
        rows.extend(
            [
                "",
                f"## {lake['label']}",
                "",
                "| System | " + " | ".join(f"R@{k}" for k in ks)
                + " | MRR@50 | Coverage@10 | R@10 delta / 95% CI |",
                "| --- | " + " | ".join("---:" for _ in ks)
                + " | ---: | ---: | --- |",
            ]
        )
        for row in lake["rows"]:
            metrics = row["result"]["metrics"]
            coverage = metrics.get("positive_evidence_path_coverage@10")
            rows.append(
                f"| {row['name']} | "
                + " | ".join(f"{metrics[f'recall@{k}']:.2%}" for k in ks)
                + f" | {metrics['mrr@50']:.4f} | "
                + (f"{coverage:.2%}" if coverage is not None else "n/a")
                + " | "
                + (_ci(row["comparison"]) if row["comparison"] else "reference")
                + " |"
            )
        rows.extend(
            [
                "",
                "Final Student+ensemble minus raw+ensemble at R@10: "
                + _ci(lake["final"]["deployment_gap"])
                + ".",
                "",
                f"Final Student checkpoint: `{lake['final']['checkpoint']}`",
                f"Student SHA-256: `{lake['final']['checkpoint_sha256']}`",
                f"Teacher source: {lake['teacher']['source']}",
                f"Teacher checkpoint: `{lake['teacher']['checkpoint']}`",
                f"Teacher SHA-256: `{lake['teacher']['checkpoint_sha256']}`",
                f"Corpus SHA-256: `{lake['corpus_sha256']}`",
            ]
        )
    rows.extend(
        [
            "",
            "## Offline footprint and latency",
            "",
            "| Lake | Raw index GiB | Final Student index GiB | Ratio | "
            "Raw ms/query/k | Student+ensemble ms/query/k |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lake in payload["lakes"].values():
        footprint = lake["footprint"]
        rows.append(
            f"| {lake['label']} | {footprint['raw_bytes'] / 2**30:.3f} | "
            f"{footprint['student_bytes'] / 2**30:.3f} | {footprint['ratio']:.2f}x | "
            f"{footprint['raw_seconds_per_query_per_k'] * 1000:.2f} | "
            f"{footprint['ensemble_seconds_per_query_per_k'] * 1000:.2f} |"
        )
    rows.extend(
        [
            "",
            "## Decisions and defaults",
            "",
            f"- EntiTables distillation chain: {'pass' if payload['decisions']['entitables_distillation_chain'] else 'fail'}.",
            f"- Both Task-L deployment CI gates: {'pass' if payload['decisions']['taskl_all_lakes_pass'] else 'fail'}.",
            f"- Task M: {payload['decisions']['task_m_status']}.",
            f"- Task N: {payload['decisions']['task_n_status']}.",
            "- PCA dimension=1024; projections frozen; anchor mu=0.1; "
            "in-batch max negatives=256.",
            "- gamma=10; gamma_e=2; weighted RRF direct=1.0/evidence=0.05; "
            "evidence aggregation=logsumexp/top4.",
            "- Teacher ensemble alpha=0.7; paired bootstrap iterations=10,000; seed=13; "
            "lake-local gate tolerance=0.02.",
            "- Gamma evidence: `../stage1_optimization_r3_20260829/taskF_gamma_sweep/gamma_sweep.png` "
            "and `../stage1_optimization_r3_20260829/taskF_gamma_sweep/gamma_latency.png`.",
            "- Shared-model relation-drift motivation: "
            "`relation_drift_motivation.png` (source: "
            "`../stage1_optimization_r3_20260829/taskH_ensemble_kd/RESULTS.md`).",
        ]
    )
    return "\n".join(rows) + "\n"


def _results_index(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage-1 round-4 task index",
            "",
            "- Task J: `taskJ_per_lake_baselines/RESULTS.md`",
            "- Task K: `taskK_per_lake_training/RESULTS.md`",
            "- Task L: `taskL_end_to_end/RESULTS.md`",
            f"- Task M: {payload['decisions']['task_m_status']}",
            "- Task N: `taskN_hard_negative_mining/RESULTS.md`",
            "- Final: `FINAL.md`",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    taskj = json.loads(args.taskj_metrics.read_text(encoding="utf-8"))
    taskk = json.loads(args.taskk_metrics.read_text(encoding="utf-8"))
    taskl = json.loads(args.taskl_metrics.read_text(encoding="utf-8"))
    taskn = json.loads(args.taskn_metrics.read_text(encoding="utf-8"))
    taskm = (
        json.loads(args.taskm_metrics.read_text(encoding="utf-8"))
        if args.taskm_metrics is not None
        else None
    )
    task_m_required = taskk["decisions"]["task_m_required"] or not taskl["all_lakes_pass"]
    if task_m_required and taskm is None:
        raise ValueError("Task M is required by the decision tree but no result was provided")
    if taskm is None:
        task_m_status = "not triggered"
        entitables_chain = taskk["decisions"]["entitables_distillation_chain"]
    else:
        taskm_lakes = taskm.get("lakes", {"entitables": taskm})
        teacher_gates = {
            name: bool(lake["teacher"]["gate_pass"])
            for name, lake in taskm_lakes.items()
        }
        strict_chains = {
            name: bool(lake["distillation_chain_established"])
            for name, lake in taskm_lakes.items()
        }
        entitables_chain = (
            teacher_gates["entitables"] and strict_chains["entitables"]
        )
        gate_status = ", ".join(
            f"{name}={'pass' if passed else 'fail'}"
            for name, passed in teacher_gates.items()
        )
        chain_status = ", ".join(
            f"{name}={'pass' if passed else 'fail'}"
            for name, passed in strict_chains.items()
        )
        task_m_status = (
            f"executed; Teacher gates {gate_status}; strict chains {chain_status}; "
            "see `taskM_entitables_teacher/RESULTS.md`"
        )
    lakes: dict[str, Any] = {}
    for name, label in (("entitables", "EntiTables"), ("wdc", "WDC")):
        rows, final = _final_rows(
            name, taskj, taskk, taskl, taskn,
            iterations=args.bootstrap_iterations, seed=args.bootstrap_seed,
        )
        taskl_lake = taskl["lakes"][name]
        lakes[name] = {
            "label": label,
            "rows": rows,
            "final": final,
            "teacher": _effective_teacher(name, taskj["lakes"][name], taskm),
            "corpus_sha256": taskj["lakes"][name]["provenance"]["corpus_sha256"],
            "footprint": {
                "raw_bytes": taskl_lake["index_bytes"]["raw"],
                "student_bytes": final["index_bytes"],
                "ratio": taskl_lake["index_bytes"]["raw"] / final["index_bytes"],
                "raw_seconds_per_query_per_k": taskl_lake["systems"]["raw"]["timing"][
                    "average_seconds_per_query_per_k"
                ],
                "ensemble_seconds_per_query_per_k": final[
                    "ensemble_seconds_per_query_per_k"
                ],
            },
        }
    payload = {
        "format_version": 1,
        "parameters": {
            "recall_ks": [10, 20, 30, 40, 50],
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "lakes": lakes,
        "decisions": {
            "entitables_distillation_chain": entitables_chain,
            "taskl_all_lakes_pass": taskl["all_lakes_pass"],
            "task_m_status": task_m_status,
            "task_n_status": (
                "skipped: " + taskn["reason"]
                if taskn.get("skipped")
                else ("mined checkpoint selected" if taskn["use_mined_checkpoint"] else "mining ran; Task-K checkpoint retained")
            ),
        },
        "teachers_by_lake": {
            name: lake["teacher"] for name, lake in lakes.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "FINAL.md").write_text(_markdown(payload), encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(_results_index(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskj-metrics", type=Path, required=True)
    parser.add_argument("--taskk-metrics", type=Path, required=True)
    parser.add_argument("--taskl-metrics", type=Path, required=True)
    parser.add_argument("--taskn-metrics", type=Path, required=True)
    parser.add_argument("--taskm-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

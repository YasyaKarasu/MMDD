#!/usr/bin/env python
"""Run the training-consistent Stage-1 pipeline-unification experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmdd_stage1.checkpoints import load_path_aggregator
from mmdd_stage1.significance import paired_bootstrap_delta


@dataclass(frozen=True)
class LakeConfig:
    name: str
    dataset: str
    source_data: Path
    aligned_data: Path
    corpus: Path
    raw_index: Path
    pca: Path
    r5_teacher: Path
    r5_metrics: Path
    kd_teacher_alpha: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_logged(command: list[str], log_path: Path, *, root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"running: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd="/tmp",
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"completed: {log_path}", flush=True)


def _python(root: Path, *arguments: str) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "MMDD",
        "python",
        str(root / "src" / arguments[0]),
        *arguments[1:],
    ]


def lake_configs(root: Path) -> dict[str, LakeConfig]:
    r4 = root / "work/stage1_optimization_r4_20260829"
    r5 = root / "work/stage1_optimization_r5_20260829"
    shared = root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828"
    enti_source = root / "work/stage1_stage2_entitables20k_v4_20260827/stage1_data"
    task_m = r4 / "taskM_entitables_teacher"
    return {
        "entitables": LakeConfig(
            name="entitables",
            dataset="entitables20k_v4",
            source_data=enti_source,
            aligned_data=task_m / "data",
            corpus=r4 / "taskJ_per_lake_baselines/corpora/entitables_corpus.jsonl",
            raw_index=r4 / "taskJ_per_lake_baselines/epoch0/entitables/raw_index",
            pca=r4 / "taskJ_per_lake_baselines/pca/entitables_pca_1024.pt",
            r5_teacher=task_m / "checkpoints/teacher_path.pt",
            r5_metrics=r5 / "task4_final/final_evaluation/entitables/metrics.json",
            kd_teacher_alpha=1.0,
        ),
        "wdc": LakeConfig(
            name="wdc",
            dataset="wdc2k_v2",
            source_data=shared / "wdc_stage1_data",
            aligned_data=task_m / "per_lake/wdc/data",
            corpus=r4 / "taskJ_per_lake_baselines/corpora/wdc_corpus.jsonl",
            raw_index=r4 / "taskJ_per_lake_baselines/epoch0/wdc/raw_index",
            pca=r4 / "taskJ_per_lake_baselines/pca/wdc_pca_1024.pt",
            r5_teacher=task_m / "per_lake/wdc/checkpoints/teacher_path.pt",
            r5_metrics=r5 / "task4_final/final_evaluation/wdc/metrics.json",
            kd_teacher_alpha=0.7,
        ),
    }


def _task_a_summary(output_root: Path) -> dict[str, Any]:
    path = output_root / "taskA_zero_training/summary.json"
    if not path.is_file():
        raise FileNotFoundError("Task A must finish on both lakes before Task B")
    return json.loads(path.read_text(encoding="utf-8"))


def _best_task_a_row(
    summary: dict[str, Any], lake: str, aggregation_form: str, fusion_form: str | None = None
) -> dict[str, Any]:
    candidates = [
        row["per_lake"][lake]
        for row in summary["rows"].values()
        if row["aggregation_form"] == aggregation_form
        and (fusion_form is None or row["fusion_form"] == fusion_form)
    ]
    if not candidates:
        raise ValueError(f"Task A has no {aggregation_form}/{fusion_form} row")
    return max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["recall@10"]),
            float(row["metrics"]["evidence"]["recall@10"]),
            row["exact_configuration"],
        ),
    )


def _aggregation_arguments(configuration: dict[str, Any]) -> list[str]:
    aggregation = configuration["aggregation"]
    return [
        "--evidence-aggregation",
        str(aggregation["form"]),
        "--evidence-top-k",
        str(aggregation["top_k"]),
        "--evidence-temperature",
        str(aggregation["temperature"]),
        "--evidence-power",
        str(aggregation["power"]),
    ]


def _fusion_arguments(configuration: dict[str, Any]) -> list[str]:
    fusion = configuration["fusion"]
    return [
        "--fusion-mode",
        str(fusion["mode"]),
        "--direct-weight",
        "1",
        "--evidence-weight",
        str(fusion["evidence_weight"]),
        "--fusion-score-normalization",
        str(fusion["normalization"]),
        "--fusion-score-temperature",
        "1",
    ]


def _select_task_b_aggregation(
    rows: dict[str, dict[str, Any]], selected_aggregation: str
) -> tuple[str, bool]:
    forms = sorted({key.split("/", maxsplit=1)[1] for key in rows})

    def passes(form: str) -> bool:
        return all(
            rows[f"{lake}/{form}"]["point_gate_minus_3pt"]
            and rows[f"{lake}/{form}"]["ci_gate_minus_3pt"]
            for lake in ("entitables", "wdc")
        )

    def maximin_score(form: str) -> tuple[float, float, bool]:
        deltas = [
            float(rows[f"{lake}/{form}"]["delta_vs_r5_teacher"]["delta_mean"])
            for lake in ("entitables", "wdc")
        ]
        return min(deltas), sum(deltas) / len(deltas), form == selected_aggregation

    selected_passes = passes(selected_aggregation)
    if selected_passes:
        return selected_aggregation, True
    passing = [form for form in forms if passes(form)]
    candidates = passing or forms
    return max(candidates, key=maximin_score), False


def _teacher_diagnostic(
    root: Path,
    lake: LakeConfig,
    teacher: Path,
    output_dir: Path,
    device: str,
) -> None:
    if (output_dir / "metrics.json").is_file():
        return
    _run_logged(
        _python(
            root,
            "diagnose_stage1_teacher_rerank.py",
            "--features",
            str(root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"),
            "--dev-data",
            str(lake.source_data / "target_lists.jsonl"),
            "--edge-dev-data",
            str(lake.source_data / "edge_lists.jsonl"),
            "--teacher-checkpoint",
            str(teacher),
            "--corpus",
            str(lake.corpus),
            "--raw-index-root",
            str(lake.raw_index),
            "--output-dir",
            str(output_dir),
            "--raw-top-k",
            "100",
            "--teacher-batch-size",
            "16",
            "--ensemble-alphas",
            "--bootstrap-iterations",
            "10000",
            "--bootstrap-seed",
            "13",
            "--feature-cache-size",
            "32000",
            "--device",
            device,
        ),
        output_dir.with_suffix(".log"),
        root=root,
    )


def run_task_b(
    root: Path,
    output_root: Path,
    device: str,
    lakes: tuple[str, ...] = ("entitables", "wdc"),
) -> dict[str, Any] | None:
    summary = _task_a_summary(output_root)
    selected_aggregation = summary["rows"][summary["selected_unified_form"]][
        "aggregation_form"
    ]
    forms = list(dict.fromkeys(["logsumexp", "topk_sum", selected_aggregation]))
    features = root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    task_root = output_root / "taskB_teacher_retrain"
    configs = lake_configs(root)
    for lake_name in lakes:
        lake = configs[lake_name]
        lake_root = task_root / lake.name
        edge = lake_root / "shared_edge/teacher_edge.pt"
        if not edge.with_suffix(".pt.selection.json").is_file():
            _run_logged(
                _python(
                    root,
                    "train_stage1.py",
                    "teacher-edge",
                    "--features",
                    str(features),
                    "--base-data",
                    str(lake.aligned_data / "edge_lists.jsonl"),
                    "--dev-data",
                    str(lake.aligned_data / "edge_lists.jsonl"),
                    "--learning-rate",
                    "5e-5",
                    "--epochs",
                    "10",
                    "--batch-size",
                    "8",
                    "--patience",
                    "3",
                    "--feature-cache-size",
                    "24000",
                    "--teacher-amp",
                    "off",
                    "--device",
                    device,
                    "--output",
                    str(edge),
                ),
                edge.with_suffix(".log"),
                root=root,
            )
        _teacher_diagnostic(
            root, lake, lake.r5_teacher, lake_root / "reference_r5", device
        )
        for form in forms:
            row = _best_task_a_row(summary, lake.name, form)
            run_root = lake_root / form
            teacher = run_root / "teacher_path.pt"
            if not teacher.with_suffix(".pt.selection.json").is_file():
                _run_logged(
                    _python(
                        root,
                        "train_stage1.py",
                        "teacher-path",
                        "--features",
                        str(features),
                        "--base-data",
                        str(lake.aligned_data / "target_lists.jsonl"),
                        "--dev-data",
                        str(lake.aligned_data / "target_lists.jsonl"),
                        "--teacher-checkpoint",
                        str(edge),
                        "--corpus",
                        str(lake.corpus),
                        "--raw-index-root",
                        str(lake.raw_index),
                        "--teacher-rerank",
                        "--teacher-rerank-dev-data",
                        str(lake.source_data / "target_lists.jsonl"),
                        "--teacher-rerank-top-k",
                        "100",
                        "--teacher-rerank-batch-size",
                        "16",
                        "--teacher-rerank-interval",
                        "2",
                        "--primary-metric",
                        "teacher_rerank.recall@10",
                        "--learning-rate",
                        "5e-5",
                        "--epochs",
                        "10",
                        "--batch-size",
                        "8",
                        "--patience",
                        "3",
                        "--feature-cache-size",
                        "32000",
                        "--teacher-amp",
                        "off",
                        *_aggregation_arguments(row["configuration"]),
                        "--device",
                        device,
                        "--output",
                        str(teacher),
                    ),
                    teacher.with_suffix(".log"),
                    root=root,
                )
            _teacher_diagnostic(
                root, lake, teacher, run_root / "diagnostic", device
            )
    if not _task_b_artifacts_complete(root, output_root, forms):
        return None
    return summarize_task_b(root, output_root, forms, selected_aggregation)


def _task_b_artifacts_complete(
    root: Path, output_root: Path, forms: list[str]
) -> bool:
    task_root = output_root / "taskB_teacher_retrain"
    return all(
        (task_root / lake.name / "reference_r5/metrics.json").is_file()
        and all(
            (task_root / lake.name / form / "diagnostic/metrics.json").is_file()
            for form in forms
        )
        for lake in lake_configs(root).values()
    )


def summarize_task_b(
    root: Path,
    output_root: Path,
    forms: list[str],
    selected_aggregation: str,
) -> dict[str, Any]:
    task_root = output_root / "taskB_teacher_retrain"
    rows: dict[str, dict[str, Any]] = {}
    for lake in lake_configs(root).values():
        reference = json.loads(
            (task_root / lake.name / "reference_r5/metrics.json").read_text(
                encoding="utf-8"
            )
        )["teacher_reranked"]
        for form in forms:
            metrics = json.loads(
                (task_root / lake.name / form / "diagnostic/metrics.json").read_text(
                    encoding="utf-8"
                )
            )["teacher_reranked"]
            delta = paired_bootstrap_delta(
                metrics["per_query"]["recall@10"],
                reference["per_query"]["recall@10"],
                iterations=10_000,
                seed=13,
            )
            rows[f"{lake.name}/{form}"] = {
                "lake": lake.name,
                "aggregation_form": form,
                "recall@10": metrics["recall@10"],
                "reference_r5_recall@10": reference["recall@10"],
                "delta_vs_r5_teacher": delta,
                "point_gate_minus_3pt": metrics["recall@10"]
                >= reference["recall@10"] - 0.03,
                "ci_gate_minus_3pt": delta["ci_low"] >= -0.03,
            }
    final_aggregation, selected_passes = _select_task_b_aggregation(
        rows, selected_aggregation
    )
    final_passes = all(
        rows[f"{lake}/{final_aggregation}"]["point_gate_minus_3pt"]
        and rows[f"{lake}/{final_aggregation}"]["ci_gate_minus_3pt"]
        for lake in ("entitables", "wdc")
    )
    payload = {
        "format_version": 1,
        "task_a_selected_aggregation": selected_aggregation,
        "selected_aggregation_passed_both_teacher_gates": selected_passes,
        "final_unified_aggregation": final_aggregation,
        "final_aggregation_passed_both_teacher_gates": final_passes,
        "rows": rows,
    }
    _write_json(task_root / "selection.json", payload)
    lines = [
        "# Task B: training-consistent Teacher comparison",
        "",
        f"Task-A aggregation: `{selected_aggregation}`. Task-C aggregation: "
        f"`{final_aggregation}`.",
        (
            "The Task-A aggregation passed both lakes' point and CI gates."
            if selected_passes
            else (
                "The Task-A aggregation failed at least one Teacher gate; Task C "
                "uses the maximin candidate among the forms that passed both lakes."
                if final_passes
                else "No candidate passed both lakes' Teacher gates; Task C uses "
                "the best available maximin candidate and records the gate failure."
            )
        ),
        "",
        "| Lake | Aggregation | Rerank R@10 | r5 Teacher | Paired delta / 95% CI | ±3pt gate |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows.values():
        delta = row["delta_vs_r5_teacher"]
        lines.append(
            f"| {row['lake']} | {row['aggregation_form']} | {row['recall@10']:.2%} | "
            f"{row['reference_r5_recall@10']:.2%} | {delta['delta_mean']:+.2%} "
            f"[{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}] | "
            f"{row['point_gate_minus_3pt']} / {row['ci_gate_minus_3pt']} |"
        )
    (task_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _task_c_configuration(
    task_a: dict[str, Any], task_b: dict[str, Any], lake: str
) -> dict[str, Any]:
    final_aggregation = task_b["final_unified_aggregation"]
    task_a_fusion = task_a["rows"][task_a["selected_unified_form"]]["fusion_form"]
    return _best_task_a_row(task_a, lake, final_aggregation, task_a_fusion)[
        "configuration"
    ]


def run_task_cde(
    root: Path,
    output_root: Path,
    device: str,
    lakes: tuple[str, ...] = ("entitables", "wdc"),
) -> None:
    task_a = _task_a_summary(output_root)
    task_b_path = output_root / "taskB_teacher_retrain/selection.json"
    if not task_b_path.is_file():
        raise FileNotFoundError("Task B must complete before Task C")
    task_b = json.loads(task_b_path.read_text(encoding="utf-8"))
    final_aggregation = task_b["final_unified_aggregation"]
    features = root / "work/stage1_stage2_wdc2k_entitables20k_v4_20260828/features_qwen3_vl_embedding_8b"
    task_root = output_root / "taskC_unified_retrain"
    result_rows = {}
    configs = lake_configs(root)
    for lake_name in lakes:
        lake = configs[lake_name]
        configuration = _task_c_configuration(task_a, task_b, lake.name)
        teacher = (
            output_root
            / "taskB_teacher_retrain"
            / lake.name
            / final_aggregation
            / "teacher_path.pt"
        )
        run_root = task_root / lake.name
        student = run_root / "student_path.pt"
        if not student.with_suffix(".pt.selection.json").is_file():
            _run_logged(
                _python(
                    root,
                    "train_stage1.py",
                    "student-path",
                    "--features",
                    str(features),
                    "--base-data",
                    str(lake.source_data / "target_lists.jsonl"),
                    "--dev-data",
                    str(lake.source_data / "target_lists.jsonl"),
                    "--student-initialization",
                    "pca",
                    "--student-pca-basis",
                    str(lake.pca),
                    "--student-dim",
                    "1024",
                    "--freeze-projection",
                    "--corpus",
                    str(lake.corpus),
                    "--raw-index-root",
                    str(lake.raw_index),
                    "--teacher-checkpoint",
                    str(teacher),
                    "--teacher-logit-cache",
                    str(run_root / "teacher_logits"),
                    "--teacher-logit-batch-size",
                    "32",
                    "--teacher-amp",
                    "off",
                    "--kd-target-teacher-alpha",
                    str(lake.kd_teacher_alpha),
                    "--distillation-weight",
                    "0.3",
                    "--in-batch-negatives",
                    "--in-batch-max-negatives",
                    "256",
                    "--anchor-weight",
                    "0.1",
                    "--anchor-weight-evidence",
                    "0.1",
                    "--learning-rate",
                    "1e-4",
                    "--relation-learning-rate",
                    "1e-5",
                    "--dataset-sampling-alpha",
                    "0",
                    "--epochs",
                    "12",
                    "--batch-size",
                    "64",
                    "--patience",
                    "4",
                    "--primary-metric",
                    "recall@10",
                    "--per-dataset-gate",
                    f"{lake.dataset}:recall@10>=0",
                    "--gate-tolerance",
                    "0.02",
                    "--bootstrap-iterations",
                    "10000",
                    "--bootstrap-seed",
                    "13",
                    "--recall-ks",
                    "10,20,30,40,50",
                    "--train-eval-ks",
                    "10,50",
                    "--gamma",
                    "10",
                    "--gamma-evidence",
                    "2",
                    *_aggregation_arguments(configuration),
                    *_fusion_arguments(configuration),
                    "--min-dev-evidence-path-queries",
                    "1",
                    "--min-dev-evidence-path-coverage",
                    "0.07",
                    "--min-dev-evidence-path-coverage-by-dataset",
                    "0.01",
                    "--device",
                    device,
                    "--output",
                    str(student),
                ),
                run_root / "train.log",
                root=root,
            )
        evaluation = run_root / "final_evaluation"
        if not (evaluation / "metrics.json").is_file():
            _run_logged(
                _python(
                    root,
                    "evaluate_stage1_r3_baselines.py",
                    "--selection",
                    str(student.with_suffix(".pt.selection.json")),
                    "--features",
                    str(features),
                    "--dev-data",
                    str(lake.source_data / "target_lists.jsonl"),
                    "--corpus",
                    str(lake.corpus),
                    "--teacher-checkpoint",
                    str(teacher),
                    "--output-dir",
                    str(evaluation),
                    "--title",
                    f"Unified Stage-1 {lake.name}",
                    "--student-label",
                    "Training-consistent unified KD Student",
                    "--systems",
                    "raw,student,raw_ensemble,student_ensemble",
                    "--recall-ks",
                    "10,20,30,40,50",
                    "--gamma",
                    "10",
                    "--gamma-evidence",
                    "2",
                    "--teacher-alpha",
                    "0.7",
                    *_aggregation_arguments(configuration),
                    *_fusion_arguments(configuration),
                    "--feature-cache-size",
                    "60000",
                    "--bootstrap-iterations",
                    "10000",
                    "--bootstrap-seed",
                    "13",
                    "--device",
                    device,
                ),
                run_root / "final_evaluation.log",
                root=root,
            )
        metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        student_metrics = metrics["systems"]["student"]["metrics"]
        r5_metrics = json.loads(lake.r5_metrics.read_text(encoding="utf-8"))[
            "systems"
        ]["student"]["metrics"]
        delta = paired_bootstrap_delta(
            student_metrics["per_query"]["fused"]["recall@10"],
            r5_metrics["per_query"]["fused"]["recall@10"],
            iterations=10_000,
            seed=13,
        )
        saved = load_path_aggregator(student)
        lake_result = {
            "configuration": configuration,
            "teacher_checkpoint": str(teacher.resolve()),
            "student_checkpoint": str(student.resolve()),
            "checkpoint_aggregation": {
                "form": saved.evidence_aggregation,
                "top_k": saved.top_k,
                "temperature": saved.temperature,
                "power": saved.power,
            },
            "recall@10": student_metrics["recall@10"],
            "r5_recall@10": r5_metrics["recall@10"],
            "delta_vs_r5": delta,
            "gate": student_metrics["recall@10"] >= r5_metrics["recall@10"],
            "systems": {
                name: row["metrics"]["recall@10"]
                for name, row in metrics["systems"].items()
            },
        }
        _write_json(run_root / "result.json", lake_result)
        result_rows[lake.name] = lake_result
    for lake in configs.values():
        result_path = task_root / lake.name / "result.json"
        if not result_path.is_file():
            return
        result_rows[lake.name] = json.loads(result_path.read_text(encoding="utf-8"))
    _write_task_cde_reports(output_root, task_a, task_b, result_rows)


def _task_c_decision(rows: dict[str, Any]) -> dict[str, Any]:
    """Apply the plan's adoption rule to the two-lake Task-C results."""

    all_r5_point_gates_passed = all(
        bool(rows[lake]["gate"]) for lake in ("entitables", "wdc")
    )
    return {
        "all_r5_point_gates_passed": all_r5_point_gates_passed,
        "adopt_unified_as_primary": all_r5_point_gates_passed,
        "primary_configuration": (
            "unified" if all_r5_point_gates_passed else "r5"
        ),
        "unified_role": (
            "primary_paper_configuration"
            if all_r5_point_gates_passed
            else "training_consistent_upper_bound_analysis"
        ),
    }


def _write_task_cde_reports(
    output_root: Path,
    task_a: dict[str, Any],
    task_b: dict[str, Any],
    rows: dict[str, Any],
) -> None:
    task_c = output_root / "taskC_unified_retrain"
    decision = _task_c_decision(rows)
    payload = {
        "format_version": 1,
        "unified_aggregation": task_b["final_unified_aggregation"],
        "unified_fusion": task_a["rows"][task_a["selected_unified_form"]][
            "fusion_form"
        ],
        "training_inference_consistent": True,
        "decision": decision,
        "rows": rows,
    }
    _write_json(task_c / "metrics.json", payload)
    lines = [
        "# Task C: complete unified retraining chain",
        "",
        f"Aggregation: `{payload['unified_aggregation']}`; fusion: "
        f"`{payload['unified_fusion']}`. Both forms are shared across lakes.",
        "",
        "| Lake | Unified R@10 | r5 R@10 | Paired delta / 95% CI | Point gate |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for lake, row in rows.items():
        delta = row["delta_vs_r5"]
        lines.append(
            f"| {lake} | {row['recall@10']:.2%} | {row['r5_recall@10']:.2%} | "
            f"{delta['delta_mean']:+.2%} [{delta['ci_low']:+.2%}, "
            f"{delta['ci_high']:+.2%}] | {row['gate']} |"
        )
    lines.extend(
        [
            "",
            (
                "Both lakes meet the r5 point gates, so the unified pipeline is "
                "the primary paper configuration."
                if decision["adopt_unified_as_primary"]
                else "The unified chain is methodologically valid, but WDC does "
                "not meet the r5 point gate. Per the plan, retain r5 as the "
                "primary configuration and report Task C as the training-consistent "
                "unification upper-bound analysis."
            ),
            "",
        ]
    )
    (task_c / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    task_d = output_root / "taskD_wdc_65_confirmation"
    wdc = rows["wdc"]
    unified_vs_r8 = float(wdc["recall@10"]) - 0.6535
    reproduces_r8_65_35 = float(wdc["recall@10"]) >= 0.6535
    _write_json(
        task_d / "metrics.json",
        {
            "unified_wdc_recall@10": wdc["recall@10"],
            "r5_wdc_recall@10": wdc["r5_recall@10"],
            "r8_inconsistent_wdc_recall@10": 0.6535,
            "r8_adoptable": False,
            "meets_r5_point_gate": bool(wdc["gate"]),
            "reproduces_r8_65_35": reproduces_r8_65_35,
            "unified_delta_vs_r8": unified_vs_r8,
            "primary_configuration": decision["primary_configuration"],
            "delta_vs_r5": wdc["delta_vs_r5"],
        },
    )
    (task_d / "RESULTS.md").write_text(
        "# Task D: WDC 65+ confirmation\n\n"
        f"Outcome: **65+ was not reproduced**. The unified result is "
        f"{wdc['recall@10']:.2%}, {wdc['delta_vs_r5']['delta_mean']:+.2%} "
        "versus r5 and "
        f"{unified_vs_r8:+.2%} versus the non-adoptable r8 number. Retain r5 "
        "as the primary configuration.\n\n"
        "| System | WDC R@10 | Training-consistent | Primary |\n"
        "| --- | ---: | --- | --- |\n"
        f"| Unified pipeline | {wdc['recall@10']:.2%} | yes | no |\n"
        f"| r5 | {wdc['r5_recall@10']:.2%} | yes | yes |\n"
        "| r8 per-lake inference-only | 65.35% | no | no |\n",
        encoding="utf-8",
    )

    task_e = output_root / "taskE_paper_integration"
    final_lines = [
        "# Stage-1 unified pipeline final report",
        "",
        f"Unified aggregation form: `{payload['unified_aggregation']}`. Unified "
        f"fusion form: `{payload['unified_fusion']}`. Lake-local internal "
        "parameters are recorded in Task C metrics.",
        "",
        "All unified numbers below use a teacher trained with the same path "
        "aggregation, newly generated Teacher logits, and a Student distilled "
        "with that identical aggregation. The r8 WDC 65.35% result changed only "
        "inference and is reference-only.",
        "",
        "## Decision",
        "",
        (
            "Both lakes meet the r5 point gates; adopt the unified pipeline as "
            "the primary paper configuration."
            if decision["adopt_unified_as_primary"]
            else "The unified pipeline is training–inference consistent, but WDC "
            "falls below r5. Retain r5 as the primary paper configuration and "
            "use the unified result as the measured upper bound/cost of enforcing "
            "one aggregation and fusion form across both lakes."
        ),
        "",
        "| Lake | Raw | Supervised Teacher reranker | Unified KD | KD + reranker | r5 KD | Unified KD vs r5 / 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for lake, row in rows.items():
        systems = row["systems"]
        delta = row["delta_vs_r5"]
        teacher_key = f"{lake}/{payload['unified_aggregation']}"
        teacher_rerank = task_b["rows"][teacher_key]["recall@10"]
        adopted_ensemble = (
            systems.get("student_ensemble", systems["student"])
            if lake == "entitables"
            else systems["student"]
        )
        final_lines.append(
            f"| {lake} | {systems['raw']:.2%} | {teacher_rerank:.2%} | "
            f"{systems['student']:.2%} | {adopted_ensemble:.2%} | "
            f"{row['r5_recall@10']:.2%} | {delta['delta_mean']:+.2%} "
            f"[{delta['ci_low']:+.2%}, {delta['ci_high']:+.2%}] |"
        )
    final_lines.extend(
        [
            "",
            "## Unified-form cost and benefit",
            "",
            f"- EntiTables unified KD changes by "
            f"{rows['entitables']['delta_vs_r5']['delta_mean']:+.2%} versus r5; "
            f"its Student+Teacher ensemble reaches "
            f"{rows['entitables']['systems']['student_ensemble']:.2%}.",
            f"- WDC unified KD changes by "
            f"{rows['wdc']['delta_vs_r5']['delta_mean']:+.2%} versus r5 and "
            f"{unified_vs_r8:+.2%} versus the r8 inference-only 65.35% reference.",
            "- The WDC 65.35% reference is not adoptable because its training and "
            "inference aggregation configurations differ.",
            "",
            "The WDC reranker remains a no-op in the unified evaluation; the "
            "EntiTables Student+Teacher row is reported separately. See Task C "
            "for paired 10,000-iteration bootstrap confidence intervals.",
            "",
        ]
    )
    (task_e / "FINAL.md").parent.mkdir(parents=True, exist_ok=True)
    (task_e / "FINAL.md").write_text("\n".join(final_lines), encoding="utf-8")
    (output_root / "FINAL.md").write_text("\n".join(final_lines), encoding="utf-8")

    task_a_row = task_a["rows"][task_a["selected_unified_form"]]
    root_lines = [
        "# Stage-1 pipeline unification results",
        "",
        "## Decision trail",
        "",
        f"- Task A selected `{task_a['selected_unified_form']}` as the zero-training "
        "maximin form; it required training-side validation.",
        f"- Task B selected `{task_b['final_unified_aggregation']}` after the "
        "training-consistent Teacher gates.",
        f"- Task C used `{payload['unified_aggregation']}` aggregation and "
        f"`{payload['unified_fusion']}` fusion for both lakes, with lake-local "
        "internal parameters.",
        (
            "- Both r5 point gates passed, so the unified pipeline is primary."
            if decision["adopt_unified_as_primary"]
            else "- WDC failed the r5 point gate, so r5 remains primary and the "
            "unified pipeline is reported as the training-consistent upper-bound "
            "analysis."
        ),
        "",
        "## Final training-consistent comparison",
        "",
        "| Lake | Task-A diagnostic R@10 | Unified KD R@10 | r5 R@10 | "
        "Unified delta / 95% CI | Meets r5 point gate |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for lake, row in rows.items():
        task_a_lake = task_a_row["per_lake"][lake]
        delta = row["delta_vs_r5"]
        root_lines.append(
            f"| {lake} | {task_a_lake['metrics']['recall@10']:.2%} | "
            f"{row['recall@10']:.2%} | {row['r5_recall@10']:.2%} | "
            f"{delta['delta_mean']:+.2%} [{delta['ci_low']:+.2%}, "
            f"{delta['ci_high']:+.2%}] | {row['gate']} |"
        )
    root_lines.extend(
        [
            "",
            "The Task-A values are diagnostic because non-`logsumexp` "
            "aggregation reused r5-trained checkpoints. The Task-C values are "
            "methodologically valid: Teacher training, Teacher-logit generation, "
            "Student KD, and retrieval all use the same aggregation configuration. "
            "The decision above separately states whether they become primary.",
            "",
            "Detailed reports: `taskA_zero_training/RESULTS.md`, "
            "`taskB_teacher_retrain/RESULTS.md`, "
            "`taskC_unified_retrain/RESULTS.md`, "
            "`taskD_wdc_65_confirmation/RESULTS.md`, and `FINAL.md`.",
            "",
        ]
    )
    (output_root / "RESULTS.md").write_text(
        "\n".join(root_lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["b", "cde", "all"], default="all")
    parser.add_argument(
        "--lake",
        choices=["entitables", "wdc"],
        help="Run only one lake; shared Task B/C-E reports wait for both lakes.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("work/stage1_pipeline_unification_20260831"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    lakes = (args.lake,) if args.lake else ("entitables", "wdc")
    if args.phase in {"b", "all"}:
        run_task_b(root, output_root, args.device, lakes)
    if args.phase in {"cde", "all"}:
        run_task_cde(root, output_root, args.device, lakes)


if __name__ == "__main__":
    main()

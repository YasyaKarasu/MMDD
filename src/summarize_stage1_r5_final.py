#!/usr/bin/env python
"""Build the final Stage-1 r5 paper-ready table and decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmdd_stage1.evaluation import DEFAULT_RECALL_KS
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta


RECALL_KS = DEFAULT_RECALL_KS


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _per_query(metrics: dict[str, Any], kind: str) -> list[float]:
    if kind == "zero_one_hop_weighted_rrf":
        return list(metrics["per_query"]["fused"]["recall@10"])
    return list(metrics["per_query"]["recall@10"])


def _delta(
    candidate: list[float],
    reference: list[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    return paired_bootstrap_delta(
        candidate, reference, iterations=iterations, seed=seed
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _checkpoint_sha256(path_value: str | None) -> str | None:
    return (
        checkpoint_fingerprint(Path(path_value))
        if path_value is not None
        else None
    )


def _row(
    name: str,
    result: dict[str, Any],
    raw_per_query: list[float],
    *,
    iterations: int,
    seed: int,
    online_no_op: bool = False,
) -> dict[str, Any]:
    metrics = result["metrics"]
    kind = result["kind"]
    per_query = _per_query(metrics, kind)
    return {
        "name": name,
        "kind": kind,
        **{f"recall@{k}": float(metrics[f"recall@{k}"]) for k in RECALL_KS},
        "mrr@50": float(metrics["mrr@50"]),
        "coverage@10": (
            float(metrics["positive_evidence_path_coverage@10"])
            if "positive_evidence_path_coverage@10" in metrics
            else None
        ),
        "vs_raw_fused_recall@10": _delta(
            per_query,
            raw_per_query,
            iterations=iterations,
            seed=seed,
        ),
        "per_query_recall@10": per_query,
        "online_no_op": online_no_op,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    selection = _read(Path(args.selection))
    evaluations = {
        "entitables": _read(Path(args.entitables_evaluation)),
        "wdc": _read(Path(args.wdc_evaluation)),
    }
    task1 = _read(Path(args.task1_metrics))
    task2 = _read(Path(args.task2_metrics))
    taskx = _read(Path(args.taskx_metrics))
    r4_root = Path(args.r4_root)
    taskj = _read(r4_root / "taskJ_per_lake_baselines" / "metrics.json")
    taskk = _read(r4_root / "taskK_per_lake_training" / "metrics.json")
    lakes = {}
    teacher_provenance = {}
    for lake in ("entitables", "wdc"):
        systems = evaluations[lake]["systems"]
        raw = systems["raw"]
        student = systems["student"]
        raw_per_query = _per_query(raw["metrics"], raw["kind"])
        supervised_result = taskk["runs"][f"{lake}_supervised"][
            "final_evaluation"
        ]
        rows = [
            _row(
                "Raw embedding",
                raw,
                raw_per_query,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            _row(
                "Supervised Student",
                supervised_result,
                raw_per_query,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            _row(
                "KD Student (final)",
                student,
                raw_per_query,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
        ]
        student_per_query = _per_query(student["metrics"], student["kind"])
        online_gate = None
        deployment = student
        online_no_op = False
        if "student_ensemble" in systems:
            ensemble = systems["student_ensemble"]
            ensemble_per_query = _per_query(ensemble["metrics"], ensemble["kind"])
            online_gate = _delta(
                ensemble_per_query,
                student_per_query,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            )
            accepted = lake == "entitables" or (
                online_gate["mean"] > 0
                and online_gate["ci_low"] >= -args.gate_tolerance
            )
            if accepted:
                deployment = ensemble
            else:
                online_no_op = True
        else:
            online_no_op = True
        rows.append(
            _row(
                "Student + online reranker",
                deployment,
                raw_per_query,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
                online_no_op=online_no_op,
            )
        )
        supervised_per_query = _per_query(
            supervised_result["metrics"], supervised_result["kind"]
        )
        distillation_chain = _delta(
            student_per_query,
            supervised_per_query,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        )
        selected = selection["selected"][lake]
        student_selection = _read(Path(selected["student_selection"]))
        raw_index = Path(student_selection["raw_embedding_index"])
        student_index = Path(student_selection["best_index"])
        timing = {
            name: result["timing"]["average_seconds_per_query_per_k"]
            for name, result in systems.items()
            if "timing" in result
        }
        lakes[lake] = {
            "rows": rows,
            "distillation_chain": {
                "definition": "KD fused recall@10 >= supervised fused recall@10",
                "passed_point_estimate": distillation_chain["mean"] >= 0,
                "delta": distillation_chain,
            },
            "online_reranker_gate": online_gate,
            "online_reranker_no_op": online_no_op,
            "selected_student": selected,
            "pca_explained_variance_ratio": taskj["lakes"][lake][
                "pca_explained_variance_ratio"
            ],
            "footprint": {
                "raw_index_bytes": _directory_bytes(raw_index),
                "student_index_bytes": _directory_bytes(student_index),
            },
            "timing_seconds_per_query_per_k": timing,
            "corpus_sha256": evaluations[lake]["corpus_sha256"],
        }
        teacher_provenance[lake] = {
            "kd_source": selected["kd_teacher_label"],
            "kd_teacher_checkpoint": selected["kd_teacher_checkpoint"],
            "kd_teacher_checkpoint_sha256": _checkpoint_sha256(
                selected["kd_teacher_checkpoint"]
            ),
            "online_source": selected["online_teacher_provenance"],
            "online_teacher_candidate": selected["online_teacher_candidate"],
            "online_teacher_candidate_sha256": _checkpoint_sha256(
                selected["online_teacher_candidate"]
            ),
            "online_decision": "no-op" if online_no_op else "enabled",
        }

    payload = {
        "format_version": 1,
        "primary_metric": "fused recall@10",
        "lakes": lakes,
        "teacher_provenance": teacher_provenance,
        "mechanism_decomposition": task1["mechanism_decomposition"],
        "task2_gate": task2["gate"],
        "taskx_decisions": taskx["decisions"],
        "defaults": {
            "kd_target_teacher_alpha": {
                lake: selection["selected"][lake]["tau"]
                for lake in ("entitables", "wdc")
            },
            "distillation_weight": 0.3,
            "pca_dimension": 1024,
            "projection_frozen": True,
            "anchor_mu": 0.1,
            "in_batch_max_negatives": 256,
            "gamma": 10,
            "gamma_evidence": 2,
            "fusion": "weighted_rrf",
            "direct_weight": 1.0,
            "evidence_weight": 0.05,
            "evidence_aggregation": "logsumexp",
            "evidence_top_k": 4,
            "gate_tolerance": args.gate_tolerance,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "final_metrics.json", payload)

    lines = [
        "# Stage-1 round 5: final per-lake configuration",
        "",
        "Primary model-selection and distillation-chain metric: fused R@10 with weighted RRF.",
        "",
    ]
    for lake in ("entitables", "wdc"):
        label = "EntiTables" if lake == "entitables" else "WDC"
        lines.extend(
            [
                f"## {label}",
                "",
                "| System | R@10 | R@20 | R@30 | R@40 | R@50 | MRR@50 | Coverage@10 | R@10 delta vs raw / 95% CI |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in lakes[lake]["rows"]:
            delta = row["vs_raw_fused_recall@10"]
            coverage = (
                "n/a" if row["coverage@10"] is None else _percent(row["coverage@10"])
            )
            delta_text = (
                "reference"
                if row["name"] == "Raw embedding"
                else f"{_percent(delta['mean'])} "
                f"[{_percent(delta['ci_low'])}, {_percent(delta['ci_high'])}]"
            )
            suffix = " (no-op)" if row["online_no_op"] else ""
            lines.append(
                f"| {row['name']}{suffix} | "
                + " | ".join(_percent(row[f"recall@{k}"]) for k in RECALL_KS)
                + f" | {row['mrr@50']:.4f} | {coverage} | {delta_text} |"
            )
        chain = lakes[lake]["distillation_chain"]
        chain_delta = chain["delta"]
        lines.extend(
            [
                "",
                "Distillation chain (KD fused R@10 minus supervised fused R@10): "
                f"{_percent(chain_delta['mean'])} "
                f"[{_percent(chain_delta['ci_low'])}, {_percent(chain_delta['ci_high'])}]; "
                f"point-estimate pass **{chain['passed_point_estimate']}**.",
                "",
                f"- KD source: {teacher_provenance[lake]['kd_source']}; "
                f"checkpoint `{teacher_provenance[lake]['kd_teacher_checkpoint']}`.",
                f"- Online reranker: {teacher_provenance[lake]['online_decision']}; "
                f"provenance: {teacher_provenance[lake]['online_source']}.",
                f"- PCA-1024 explained variance: "
                f"{lakes[lake]['pca_explained_variance_ratio']:.2%}.",
                f"- Raw index: {lakes[lake]['footprint']['raw_index_bytes'] / 2**30:.3f} GiB; "
                f"Student index: {lakes[lake]['footprint']['student_index_bytes'] / 2**30:.3f} GiB.",
                "- Latency (mean per query per evaluated k): "
                f"raw={1_000 * lakes[lake]['timing_seconds_per_query_per_k']['raw']:.2f} ms; "
                f"Student={1_000 * lakes[lake]['timing_seconds_per_query_per_k']['student']:.2f} ms; "
                "Student + online reranker="
                f"{1_000 * lakes[lake]['timing_seconds_per_query_per_k'].get('student_ensemble', lakes[lake]['timing_seconds_per_query_per_k']['student']):.2f} ms"
                f"{' (no-op)' if lakes[lake]['online_reranker_no_op'] else ''}.",
                "",
            ]
        )

    lines.extend(["## Mechanism attribution", ""])
    for lake, values in task1["mechanism_decomposition"].items():
        anchor = values["cosine_anchoring_vs_supervised"]
        teacher = values["teacher_residual_at_tau_0.7_vs_cosine"]
        lines.append(
            f"- {lake}: pure-cosine anchoring vs supervised "
            f"{_percent(anchor['mean'])} "
            f"[{_percent(anchor['ci_low'])}, {_percent(anchor['ci_high'])}]; "
            f"Teacher residual at tau=0.7 vs pure cosine "
            f"{_percent(teacher['mean'])} "
            f"[{_percent(teacher['ci_low'])}, {_percent(teacher['ci_high'])}]."
        )
    lines.extend(
        [
            "",
            "The full tau curve is in `task1_kd_tau_ablation/tau_gain_curve.csv` and `.png`.",
            "",
            "## WDC Teacher and evidence decisions",
            "",
            f"- Mixed-negative Teacher gate: **{task2['gate']['passed']}**; "
            f"decision: `{task2['gate']['decision']}`.",
            f"- Target-bound evidence accepted: **{taskx['decisions']['target_binding_resolved']}**.",
            f"- Image sparsity resolved by text-only/balanced treatment: "
            f"**{taskx['decisions']['image_sparsity_resolved']}**.",
            "",
            "## Final defaults and provenance",
            "",
            f"- tau: EntiTables={selection['selected']['entitables']['tau']}; "
            f"WDC={selection['selected']['wdc']['tau']}; distillation weight=0.3.",
            "- PCA-1024, frozen projection, anchor mu=0.1, in-batch maximum=256.",
            "- gamma=10, gamma_e=2, weighted RRF direct=1/evidence=0.05, logsumexp/top4.",
            "- Paired bootstrap iterations=10,000, seed=13; lake CI tolerance=0.02.",
            "",
        ]
    )
    for lake in ("entitables", "wdc"):
        selected = selection["selected"][lake]
        provenance = teacher_provenance[lake]
        lines.append(
            f"- {lake} Student SHA-256: `{selected['student_checkpoint_sha256']}`; "
            f"corpus SHA-256: `{lakes[lake]['corpus_sha256']}`."
        )
        lines.append(
            f"- {lake} KD Teacher SHA-256: "
            f"`{provenance['kd_teacher_checkpoint_sha256'] or 'n/a'}`; "
            f"online Teacher SHA-256: "
            f"`{provenance['online_teacher_candidate_sha256'] or 'n/a'}`."
        )
    lines.append("")
    (output_dir / "FINAL.md").write_text("\n".join(lines), encoding="utf-8")

    ledger = [
        "# Stage-1 round 5 experiment ledger",
        "",
        "## Task 1: KD target attribution",
        "",
        f"- Selected tau: EntiTables={selection['selected']['entitables']['tau']}; "
        f"WDC={selection['selected']['wdc']['tau']}.",
    ]
    for lake, values in task1["mechanism_decomposition"].items():
        anchor = values["cosine_anchoring_vs_supervised"]
        teacher = values["teacher_residual_at_tau_0.7_vs_cosine"]
        ledger.append(
            f"- {lake}: cosine anchoring vs supervised "
            f"{_percent(anchor['mean'])}; Teacher residual at tau=0.7 "
            f"{_percent(teacher['mean'])}."
        )
    ledger.extend(
        [
            "- Detail: `task1_kd_tau_ablation/RESULTS.md`, "
            "`tau_gain_curve.csv`, and `tau_gain_curve.png`.",
            "",
            "## Task 2: WDC mixed-negative Teacher",
            "",
            f"- Gate passed: **{task2['gate']['passed']}**; "
            f"decision: `{task2['gate']['decision']}`.",
            "- Detail: `task2_wdc_mixed_negative_teacher/RESULTS.md` and "
            "`same_pool_teacher_comparison.csv`.",
            "",
            "## Task X: evidence binding and modality balance",
            "",
        ]
    )
    for variant in ("text_only", "target_bound", "balanced"):
        ledger.append(
            f"- {variant} accepted: "
            f"**{taskx['decisions'][variant]['accepted']}**."
        )
    ledger.extend(
        [
            "- Detail: `taskX_evidence_modality_ablation/RESULTS.md` and "
            "`evidence_modality_ablation.csv`.",
            "",
            "## Task 4: final per-lake evaluation",
            "",
        ]
    )
    for lake in ("entitables", "wdc"):
        kd_row = next(
            row for row in lakes[lake]["rows"] if row["name"] == "KD Student (final)"
        )
        deployment_row = lakes[lake]["rows"][-1]
        ledger.append(
            f"- {lake}: final KD Student R@10 "
            f"{_percent(kd_row['recall@10'])}; deployed R@10 "
            f"{_percent(deployment_row['recall@10'])}"
            f"{' (online no-op)' if deployment_row['online_no_op'] else ''}."
        )
    ledger.extend(
        [
            "- Complete paper-ready table and provenance: `FINAL.md` and "
            "`final_metrics.json`.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text(
        "\n".join(ledger), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--entitables-evaluation", required=True)
    parser.add_argument("--wdc-evaluation", required=True)
    parser.add_argument("--task1-metrics", required=True)
    parser.add_argument("--task2-metrics", required=True)
    parser.add_argument("--taskx-metrics", required=True)
    parser.add_argument("--r4-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

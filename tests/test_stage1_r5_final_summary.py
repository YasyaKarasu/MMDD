from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import summarize_stage1_r5_final


def _path_metrics(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    return {
        **{f"recall@{k}": mean for k in (10, 20, 30, 40, 50)},
        "mrr@50": mean / 2,
        "direct": {f"recall@{k}": mean for k in (10, 20, 30, 40, 50)},
        "evidence": {f"recall@{k}": mean / 2 for k in (10, 20, 30, 40, 50)},
        "positive_evidence_path_coverage@10": mean / 4,
        "per_query": {"fused": {"recall@10": values}},
    }


def _ensemble_metrics(values: list[float]) -> dict:
    mean = sum(values) / len(values)
    return {
        **{f"recall@{k}": mean for k in (10, 20, 30, 40, 50)},
        "mrr@50": mean / 2,
        "per_query": {"recall@10": values},
    }


def _system(metrics: dict, kind: str) -> dict:
    return {
        "kind": kind,
        "metrics": metrics,
        "timing": {"average_seconds_per_query_per_k": 0.01},
    }


def test_final_summary_uses_fused_distillation_chain_and_wdc_no_op(
    tmp_path: Path,
) -> None:
    r4 = tmp_path / "r4"
    (r4 / "taskJ_per_lake_baselines").mkdir(parents=True)
    (r4 / "taskK_per_lake_training").mkdir()
    raw_values = [1.0, 0.0, 1.0, 0.0]
    supervised_values = [1.0, 1.0, 0.0, 0.0]
    student_values = [1.0, 1.0, 1.0, 0.0]
    (r4 / "taskJ_per_lake_baselines" / "metrics.json").write_text(
        json.dumps(
            {
                "lakes": {
                    lake: {"pca_explained_variance_ratio": 0.9}
                    for lake in ("entitables", "wdc")
                }
            }
        ),
        encoding="utf-8",
    )
    supervised = _system(
        _path_metrics(supervised_values), "zero_one_hop_weighted_rrf"
    )
    (r4 / "taskK_per_lake_training" / "metrics.json").write_text(
        json.dumps(
            {
                "runs": {
                    f"{lake}_supervised": {"final_evaluation": supervised}
                    for lake in ("entitables", "wdc")
                }
            }
        ),
        encoding="utf-8",
    )

    selected = {}
    evaluations = {}
    for lake in ("entitables", "wdc"):
        kd_teacher = tmp_path / lake / "kd_teacher.pt"
        online_teacher = tmp_path / lake / "online_teacher.pt"
        kd_teacher.parent.mkdir(parents=True)
        kd_teacher.write_bytes(b"kd teacher")
        online_teacher.write_bytes(b"online teacher")
        raw_index = tmp_path / lake / "raw_index"
        student_index = tmp_path / lake / "student_index"
        raw_index.mkdir(parents=True)
        student_index.mkdir()
        (raw_index / "table.hnsw").write_bytes(b"raw")
        (student_index / "table.hnsw").write_bytes(b"student")
        student_selection = tmp_path / lake / "student.selection.json"
        student_selection.write_text(
            json.dumps(
                {
                    "raw_embedding_index": str(raw_index),
                    "best_index": str(student_index),
                }
            ),
            encoding="utf-8",
        )
        selected[lake] = {
            "tau": 0.7,
            "student_selection": str(student_selection),
            "student_checkpoint_sha256": f"{lake}-student-sha",
            "kd_teacher_label": "lake Teacher",
            "kd_teacher_checkpoint": str(kd_teacher),
            "online_teacher_provenance": "online Teacher",
            "online_teacher_candidate": (
                str(online_teacher) if lake == "entitables" else None
            ),
        }
        ensemble_values = (
            [1.0, 1.0, 1.0, 1.0]
            if lake == "entitables"
            else [0.0, 0.0, 1.0, 0.0]
        )
        evaluations[lake] = {
            "corpus_sha256": f"{lake}-corpus-sha",
            "systems": {
                "raw": _system(
                    _path_metrics(raw_values), "zero_one_hop_weighted_rrf"
                ),
                "student": _system(
                    _path_metrics(student_values), "zero_one_hop_weighted_rrf"
                ),
                "student_ensemble": _system(
                    _ensemble_metrics(ensemble_values), "direct_teacher_ensemble"
                ),
            },
        }
        (tmp_path / f"{lake}_evaluation.json").write_text(
            json.dumps(evaluations[lake]), encoding="utf-8"
        )

    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"selected": selected}), encoding="utf-8")
    task1 = tmp_path / "task1.json"
    task1.write_text(
        json.dumps(
            {
                "mechanism_decomposition": {
                    lake: {
                        "cosine_anchoring_vs_supervised": {
                            "mean": 0.1,
                            "ci_low": 0.0,
                            "ci_high": 0.2,
                        },
                        "teacher_residual_at_tau_0.7_vs_cosine": {
                            "mean": 0.05,
                            "ci_low": -0.05,
                            "ci_high": 0.15,
                        },
                    }
                    for lake in ("entitables", "wdc")
                }
            }
        ),
        encoding="utf-8",
    )
    task2 = tmp_path / "task2.json"
    task2.write_text(
        json.dumps(
            {"gate": {"passed": False, "decision": "teacher_no_op_negative_asset"}}
        ),
        encoding="utf-8",
    )
    taskx = tmp_path / "taskx.json"
    taskx.write_text(
        json.dumps(
                {
                    "decisions": {
                        "text_only": {"accepted": False},
                        "target_bound": {"accepted": False},
                        "balanced": {"accepted": False},
                        "target_binding_resolved": False,
                        "image_sparsity_resolved": True,
                    }
            }
        ),
        encoding="utf-8",
    )

    payload = summarize_stage1_r5_final.run(
        argparse.Namespace(
            selection=str(selection_path),
            entitables_evaluation=str(tmp_path / "entitables_evaluation.json"),
            wdc_evaluation=str(tmp_path / "wdc_evaluation.json"),
            task1_metrics=str(task1),
            task2_metrics=str(task2),
            taskx_metrics=str(taskx),
            r4_root=str(r4),
            output_dir=str(tmp_path / "output"),
            gate_tolerance=0.02,
            bootstrap_iterations=100,
            bootstrap_seed=13,
        )
    )

    assert payload["lakes"]["entitables"]["distillation_chain"][
        "passed_point_estimate"
    ] is True
    assert payload["lakes"]["wdc"]["online_reranker_no_op"] is True
    final = (tmp_path / "output" / "FINAL.md").read_text()
    assert "Student + online reranker (no-op)" in final
    assert "raw=10.00 ms; Student=10.00 ms" in final
    assert "KD Teacher SHA-256" in final
    assert "online Teacher SHA-256: `n/a`" in final
    assert "distillation chain: fail" not in final.lower()
    ledger = (tmp_path / "output" / "RESULTS.md").read_text()
    assert "Task 1: KD target attribution" in ledger
    assert "Task 2: WDC mixed-negative Teacher" in ledger
    assert "Task X: evidence binding and modality balance" in ledger
    assert "Task 4: final per-lake evaluation" in ledger

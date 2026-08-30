from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_taskm import run
from summarize_stage1_r4_taskm_combined import run as run_combined


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_taskm_summary_records_teacher_gate_and_strict_chain(tmp_path: Path) -> None:
    taskk = _write(
        tmp_path / "taskk.json",
        {
            "decisions": {
                "task_m_required": True,
                "entitables_distillation_chain": False,
            },
            "runs": {
                "entitables_supervised": {
                    "raw": {
                        "direct_recall@10": 0.31,
                        "fused_recall@10": 0.30,
                    },
                    "selected": {
                        "direct_recall@10": 0.37,
                        "fused_recall@10": 0.37,
                    },
                    "best_epoch": 8,
                }
            },
        },
    )
    preflight = _write(
        tmp_path / "preflight.json",
        {"edge_records": 10, "target_records": 5, "missing_teacher_objects": 0},
    )
    teacher_selection = _write(
        tmp_path / "teacher.selection.json",
        {
            "best_checkpoint": "/teacher.pt",
            "best_checkpoint_sha256": "teacher-sha",
            "best_epoch": 4,
        },
    )
    diagnostic = _write(
        tmp_path / "diagnostic.json",
        {
            "raw_direct": {"recall@10": 0.31},
            "teacher_reranked": {"recall@10": 0.36},
        },
    )
    history = _write(
        tmp_path / "student" / "student_path.pt.history.json",
        {
            "epochs": [
                {
                    "epoch": 3,
                    "dev_retrieval": {
                        "recall@10": 0.38,
                        "direct": {"recall@10": 0.38},
                    },
                    "gate": {
                        "per_dataset": [
                            {
                                "bootstrap": {
                                    "delta_mean": 0.07,
                                    "ci95_low": 0.04,
                                    "ci95_high": 0.09,
                                }
                            }
                        ]
                    },
                }
            ]
        },
    )
    selection = _write(
        tmp_path / "student" / "student_path.pt.selection.json",
        {
            "best_epoch": 3,
            "best_checkpoint": "/student.pt",
            "best_checkpoint_sha256": "student-sha",
            "gate_unsatisfied": False,
        },
    )
    evaluation = _write(
        tmp_path / "evaluation.json",
        {"systems": {"student": {"metrics": {"recall@10": 0.38}}}},
    )
    output = tmp_path / "output"

    result = run(
        argparse.Namespace(
            taskk_metrics=taskk,
            preflight=preflight,
            teacher_selection=teacher_selection,
            diagnostic=diagnostic,
            student_history=history,
            student_selection=selection,
            student_evaluation=evaluation,
            output_dir=output,
            run_name="entitables_kd0.3_taskm",
            teacher_required_delta=0.03,
            teacher_ensemble_alpha=0.7,
        )
    )

    assert result["teacher"]["gate_pass"] is True
    assert result["distillation_chain_established"] is True
    assert result["student_rerun"]["checkpoint_sha256"] == "student-sha"
    assert (output / "RESULTS.md").is_file()


def test_taskm_combined_summary_preserves_both_lakes(tmp_path: Path) -> None:
    def lake_payload(name: str, label: str, gate: bool, chain: bool) -> dict:
        return {
            "lake": name,
            "label": label,
            "data": {"edge_records": 10, "target_records": 5},
            "teacher": {
                "checkpoint": f"/{name}-teacher.pt",
                "checkpoint_sha256": f"{name}-teacher-sha",
                "raw_recall@10": 0.3,
                "reranked_recall@10": 0.35,
                "delta": 0.05,
                "gate_pass": gate,
            },
            "references": {
                "raw_direct_recall@10": 0.3,
                "supervised_direct_recall@10": 0.32,
            },
            "student_rerun": {
                "checkpoint": f"/{name}-student.pt",
                "checkpoint_sha256": f"{name}-student-sha",
                "direct_recall@10": 0.34,
                "fused_recall@10": 0.35,
                "best_epoch": 3,
            },
            "distillation_chain_established": chain,
        }

    entitables = _write(
        tmp_path / "entitables.json",
        lake_payload("entitables", "EntiTables", True, True),
    )
    wdc = _write(
        tmp_path / "wdc.json",
        lake_payload("wdc", "WDC", False, False),
    )

    result = run_combined(
        argparse.Namespace(
            entitables_metrics=entitables,
            wdc_metrics=wdc,
            output_dir=tmp_path / "combined",
        )
    )

    assert set(result["lakes"]) == {"entitables", "wdc"}
    assert result["all_teacher_gates_pass"] is False
    assert result["teacher"] == result["lakes"]["entitables"]["teacher"]

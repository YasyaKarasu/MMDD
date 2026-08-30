from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import summarize_stage1_r5_task1


def _metrics(values: list[float]) -> dict:
    recall = sum(values) / len(values)
    return {
        "recall@10": recall,
        "direct": {"recall@10": recall},
        "evidence": {"recall@10": recall / 2},
        "positive_evidence_path_coverage@10": recall / 4,
        "per_query": {
            "fused": {"recall@10": values},
            "direct": {"recall@10": values},
        },
    }


def _write_run(path: Path, values: list[float], best_epoch: int) -> None:
    (path / "final_evaluation").mkdir(parents=True)
    (path / "final_evaluation" / "metrics.json").write_text(
        json.dumps({"systems": {"student": {"metrics": _metrics(values)}}}),
        encoding="utf-8",
    )
    (path / "student_path.pt.history.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "gate_unsatisfied": False,
                "epochs": [
                    {
                        "epoch": best_epoch,
                        "relation_drift": {"table_to_table": 1.5},
                        "dev_retrieval": {
                            "recall@10": sum(values) / len(values),
                            "direct": {"recall@10": sum(values) / len(values)},
                            "evidence": {"recall@10": 0.1},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_task1_summary_writes_curve_and_mechanism_split(tmp_path: Path) -> None:
    r4 = tmp_path / "r4"
    task = tmp_path / "task"
    (r4 / "taskJ_per_lake_baselines").mkdir(parents=True)
    (r4 / "taskK_per_lake_training").mkdir()
    raw = [0.0, 1.0, 0.0, 1.0]
    supervised = [1.0, 1.0, 0.0, 1.0]
    (r4 / "taskJ_per_lake_baselines" / "metrics.json").write_text(
        json.dumps(
            {
                "lakes": {
                    lake: {"systems": {"raw": {"metrics": _metrics(raw)}}}
                    for lake in ("entitables", "wdc")
                }
            }
        ),
        encoding="utf-8",
    )
    (r4 / "taskK_per_lake_training" / "metrics.json").write_text(
        json.dumps(
            {
                "runs": {
                    f"{lake}_supervised": {
                        "final_evaluation": {"metrics": _metrics(supervised)}
                    }
                    for lake in ("entitables", "wdc")
                }
            }
        ),
        encoding="utf-8",
    )
    for lake in ("entitables", "wdc"):
        for tau, values in (
            (0.0, supervised),
            (0.3, [1.0, 1.0, 1.0, 1.0]),
            (1.0, raw),
        ):
            _write_run(task / lake / f"tau_{tau:.1f}", values, 2)
    _write_run(
        r4 / "taskM_entitables_teacher" / "student_kd0.3",
        supervised,
        3,
    )
    _write_run(
        r4
        / "taskM_entitables_teacher"
        / "per_lake"
        / "wdc"
        / "student_kd0.3",
        supervised,
        4,
    )

    payload = summarize_stage1_r5_task1.run(
        argparse.Namespace(
            task_root=str(task),
            r4_root=str(r4),
            output_dir=None,
            bootstrap_iterations=100,
            bootstrap_seed=13,
        )
    )

    assert len(payload["rows"]) == 8
    assert payload["mechanism_decomposition"]["wdc"][
        "cosine_anchoring_vs_supervised"
    ]["mean"] == 0.0
    assert (task / "tau_gain_curve.csv").is_file()
    assert (task / "tau_gain_curve.png").stat().st_size > 0
    assert "Mechanism decomposition" in (task / "RESULTS.md").read_text()

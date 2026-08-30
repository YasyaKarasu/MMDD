from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import select_stage1_r5_final


def _selection(run: Path, score: float) -> None:
    run.mkdir(parents=True)
    checkpoint = run / "student_path.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "student_path.pt.selection.json").write_text(
        json.dumps(
            {
                "best_checkpoint": str(checkpoint),
                "best_checkpoint_sha256": f"sha-{score}",
            }
        ),
        encoding="utf-8",
    )


def test_final_selection_prefers_best_fused_tau_and_records_two_teachers(
    tmp_path: Path,
) -> None:
    task1 = tmp_path / "task1"
    task2 = tmp_path / "task2"
    r4 = tmp_path / "r4"
    rows = []
    for lake in ("entitables", "wdc"):
        for tau, score in ((0.0, 0.5), (0.3, 0.6), (0.7, 0.6), (1.0, 0.55)):
            run = select_stage1_r5_final._task1_run(task1, r4, lake, tau)
            _selection(run, score)
            rows.append(
                {"lake": lake, "tau": tau, "fused_recall@10": score}
            )
    for teacher in (
        r4 / "taskM_entitables_teacher" / "checkpoints" / "teacher_path.pt",
        r4
        / "taskM_entitables_teacher"
        / "per_lake"
        / "wdc"
        / "checkpoints"
        / "teacher_path.pt",
    ):
        teacher.parent.mkdir(parents=True, exist_ok=True)
        teacher.write_bytes(b"teacher")
    task1_metrics = tmp_path / "task1.json"
    task1_metrics.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    task2.mkdir()
    task2_metrics = tmp_path / "task2.json"
    task2_metrics.write_text(
        json.dumps({"gate": {"passed": False}}), encoding="utf-8"
    )

    payload = select_stage1_r5_final.run(
        argparse.Namespace(
            task1_metrics=str(task1_metrics),
            task1_root=str(task1),
            task2_metrics=str(task2_metrics),
            task2_root=str(task2),
            r4_root=str(r4),
            output=str(tmp_path / "selection.json"),
        )
    )

    assert payload["selected"]["entitables"]["tau"] == 0.7
    assert payload["selected"]["wdc"]["online_teacher_candidate"] is None
    assert payload["selected"]["entitables"]["kd_teacher_checkpoint"] != payload[
        "selected"
    ]["entitables"]["online_teacher_candidate"] or payload["selected"][
        "entitables"
    ]["kd_teacher_label"] == "Task-M lake-local Teacher"

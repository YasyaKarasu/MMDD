from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import run_stage1_r7_task_q


def _args(tmp_path: Path, run_name: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=tmp_path,
        lake="entitables",
        rank=16,
        anchor_weight=0.0,
        run_name=run_name,
    )


def test_task_q_output_dir_isolated_by_run_name(tmp_path: Path) -> None:
    base = run_stage1_r7_task_q._output_dir(_args(tmp_path, None))
    recovered = run_stage1_r7_task_q._output_dir(
        _args(tmp_path, "epoch_001_recovered")
    )

    assert base == tmp_path / "taskQ_joint_selection/entitables/lowrank_k_16_mu_0"
    assert recovered == base / "epoch_001_recovered"


@pytest.mark.parametrize("run_name", ["nested/name", "", ".", ".."])
def test_task_q_rejects_non_component_run_name(run_name: str) -> None:
    with pytest.raises(ValueError, match="one non-empty path component"):
        run_stage1_r7_task_q._validate_run_name(run_name)


def test_task_q_accepts_simple_run_name() -> None:
    run_stage1_r7_task_q._validate_run_name("epoch_001_recovered")

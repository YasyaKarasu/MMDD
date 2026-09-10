from __future__ import annotations

import pytest
import run_stage1_r16


def test_paired_group_bootstrap_preserves_query_macro_weighting() -> None:
    rows = [
        {"source_table_id": "a", "left": 1.0, "right": 0.0},
        {"source_table_id": "a", "left": 1.0, "right": 0.0},
        {"source_table_id": "b", "left": 0.0, "right": 1.0},
    ]
    result = run_stage1_r16.paired_group_bootstrap(
        rows, "left", "right", iterations=200, seed=7
    )
    assert result["observed_delta"] == pytest.approx(1 / 3)
    assert result["source_groups"] == 2
    assert result["queries"] == 3


def test_recall_is_query_macro_fraction() -> None:
    assert run_stage1_r16._recall(["a", "b"], ["x", "b", "a"], 2) == 0.5
    assert run_stage1_r16._recall(["a", "b"], ["x", "b", "a"], 3) == 1.0


def test_summary_keeps_implicit_and_explicit_separate() -> None:
    rows = []
    for kind, value in (("implicit", 1.0), ("explicit", 0.0)):
        row = {"query_kind": kind, "source_table_id": kind}
        for k in (10, 20, 50):
            row[f"arm_recall@{k}"] = value
        rows.append(row)
    summary = run_stage1_r16._summary(rows, "arm")
    assert summary["all"]["recall@10"] == 0.5
    assert summary["implicit"]["recall@10"] == 1.0
    assert summary["explicit"]["recall@10"] == 0.0

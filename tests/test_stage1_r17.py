from __future__ import annotations

import run_stage1_r17


def test_select_replay_samples_freezes_all_three_strata() -> None:
    rows = []
    known = set()
    for source_index in range(12):
        for destination_index in range(12):
            row = {
                "source_id": f"q{source_index}",
                "destination_id": f"t{destination_index}",
                "saved_score": float(12 - destination_index),
            }
            rows.append(row)
            if destination_index == 0:
                known.add((row["source_id"], row["destination_id"]))
    selected = run_stage1_r17.select_replay_samples(
        rows, known, count=30, seed=3
    )
    assert len(selected) == 30
    assert {row["sample_stratum"] for row in selected} == {
        "known",
        "high_unlabeled",
        "ordinary",
    }


def test_comparison_reports_exact_order_and_error() -> None:
    result = run_stage1_r17._comparison([3.0, 2.0, 1.0], [3.0, 2.0, 1.000001])
    assert result["max_absolute_error"] < 1e-5
    assert result["full_order_consistent"] is True
    assert result["top10_set_consistent"] is True

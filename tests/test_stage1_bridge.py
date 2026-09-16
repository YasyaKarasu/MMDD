"""Semantic checks for the cumulative B13-to-modern bridge controls."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_stage1_r12_candidates import apply_positive_closure
from run_stage1_bridge import _corr, _rank, bridge_lineage


def test_positive_closure_reuses_base_order_and_evicts_only_tail_negatives():
    row = {
        "query_id": "q",
        "source_type": "table",
        "destination_type": "text",
        "candidate_ids": ["present", "n1", "n2"],
        "positive_ids": ["present"],
        "confirmed_labels": [1, None, 0],
    }
    key = ("q", "table", "text")
    closed, audit = apply_positive_closure([row], {key: {"present", "missing"}}, cap=3)
    assert closed[0]["candidate_ids"] == ["missing", "present", "n1"]
    assert closed[0]["positive_ids"] == ["missing", "present"]
    assert closed[0]["confirmed_labels"] == [None, 1, None]
    assert audit == {"lists": 1, "lists_modified": 1, "positives_inserted": 1, "negatives_evicted": 1}


def test_positive_closure_does_not_mark_absent_positive_as_negative():
    row = {
        "query_id": "q",
        "source_type": "table",
        "destination_type": "image",
        "candidate_ids": ["p", "n"],
        "positive_ids": ["p"],
        "confirmed_labels": [1, 0],
    }
    key = ("q", "table", "image")
    closed, _ = apply_positive_closure([row], {key: {"p", "outside"}}, cap=2)
    assert closed[0]["candidate_ids"] == ["outside", "p"]
    assert closed[0]["confirmed_labels"] == [None, 1]
    assert "n" not in closed[0]["positive_ids"]


def test_rank_and_correlations_cover_ties_without_scipy():
    assert _rank([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    assert _corr([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert _corr([1.0, 1.0], [2.0, 3.0]) is None


def test_bridge_lineage_does_not_claim_planned_stages_are_executed():
    inputs = {"resolved": {"b0_verdict": {"exists": True, "sha256": "b0"}}}
    lineage = bridge_lineage(inputs, {"status": "available", "gpus": ["0"]})
    status = {row["stage"]: row["status"] for row in lineage["stages"]}
    assert status["B0"] == "ready"
    assert status["B1"] in {"planned", "completed", "in_progress"}
    assert status["B2"] in {"planned", "completed", "in_progress"}
    assert status["B8"] == "not_triggered"
    assert all(status[name] in {"planned", "completed", "in_progress", "not_triggered"}
               for name in status if name not in {"B0", "B1"})

from __future__ import annotations

import pytest
import torch

from diagnose_stage1_r30 import OUT, checkpoint_spec
from diagnose_stage1_r30_auxiliary import aggregate_cosine, distribution, spectral_norm_estimate
from finalize_stage1_r30 import source_bootstrap, target_recall


def test_r30_target_recall_uses_all_positive_targets_and_deduplicates_hits() -> None:
    assert target_recall(["a", "a", "x"], ["a", "b"], 3) == 0.5


def test_r30_source_bootstrap_preserves_the_query_weighted_estimand() -> None:
    rows = [
        {"source_table_id": "one-query", "delta": 1.0},
        {"source_table_id": "three-queries", "delta": 0.0},
        {"source_table_id": "three-queries", "delta": 0.0},
        {"source_table_id": "three-queries", "delta": 0.0},
    ]
    low, high = source_bootstrap(rows)
    assert low <= 0.25 <= high
    assert high <= 1.0


def test_r30_c2_relation_checkpoint_spec_uses_selected_checkpoint_and_index() -> None:
    checkpoint, index = checkpoint_spec(29, "C2-F-P178")
    assert checkpoint == OUT / "C2-CHECK/F-P/seed29/checkpoints/step_000178.pt"
    assert index == OUT / "indexes/C2-F-P178_s29"


def test_r30_auxiliary_summary_and_spectral_estimate() -> None:
    summary = distribution([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    matrix = torch.diag(torch.tensor([3.0, 1.0]))
    assert spectral_norm_estimate(matrix) == pytest.approx(3.0, rel=1e-5)


def test_r30_aggregate_gradient_cosine_treats_missing_coordinates_as_zero() -> None:
    names = ["projections.table.weight", "projections.text.weight"]
    left = [torch.tensor([1.0]), None]
    right = [torch.tensor([1.0]), torch.tensor([1.0])]
    assert aggregate_cosine(left, right, names, "P") == pytest.approx(1 / 2**0.5)

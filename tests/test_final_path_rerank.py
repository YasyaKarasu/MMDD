from __future__ import annotations

import math

import pytest

from evaluate_final_path_rerank import (
    _teacher_pair_job_counts,
    logsumexp,
    rank_scores,
    source_bootstrap,
    target_recall,
)


def test_target_recall_counts_all_positive_targets() -> None:
    assert target_recall(["a", "x", "a"], ["a", "b"], 3) == 0.5


def test_path_ranking_never_returns_no_path_targets() -> None:
    c100 = ["no-a", "path-low", "no-b", "path-high"]
    scores = {"no-a": None, "path-low": -10.0, "no-b": None, "path-high": 2.0}
    assert rank_scores(c100, scores) == ["path-high", "path-low"]


def test_path_ranking_rejects_membership_changes() -> None:
    with pytest.raises(ValueError, match="exactly equal"):
        rank_scores(["a", "b"], {"a": 1.0})


def test_logsumexp_keeps_all_path_occurrences() -> None:
    assert logsumexp([0.0, 0.0]) == pytest.approx(math.log(2.0))
    assert logsumexp([0.0]) == pytest.approx(0.0)


def test_source_bootstrap_reports_paired_wins_losses_and_ties() -> None:
    result = source_bootstrap([1.0, -1.0, 0.0], ["s1", "s1", "s2"], replicates=100)
    assert result["wins"] == result["losses"] == result["ties"] == 1
    assert result["queries"] == 3
    assert result["source_groups"] == 2


def test_teacher_execution_distinguishes_consumed_and_new_scores() -> None:
    execution = {
        "status": "complete",
        "required_pairs": 12,
        "cached_pairs": 12,
        "new_pairs": 0,
    }
    assert _teacher_pair_job_counts(execution) == {
        "teacher_pair_scores_consumed": 12,
        "teacher_pair_scores_computed_this_run": 0,
    }

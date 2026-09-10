"""Synthetic tests for R15 estimands and deployed witness responsibility."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_stage1_r15_interaction import (
    CONTRASTS, bootstrap_columns, candidate_source, greedy_responsibility,
    raw_lse_responsibility,
)


def path(evidence_id: str, score: float, modality: str = "text") -> dict:
    return {"kind": "evidence", "evidence_id": evidence_id,
            "evidence_type": modality, "path_score": score}


def test_lse_responsibility_uses_real_temperature_and_all_paths():
    paths = [path("known", 0), *[path(f"unknown{n}", 2) for n in range(5)]]
    assert raw_lse_responsibility(paths, {"known"}, temperature=2) == pytest.approx(1 / (1 + 5 * np.e))
    assert raw_lse_responsibility([], {"known"}, temperature=1) is None


def test_deployed_responsibility_follows_greedy_row_gains_not_path_softmax():
    paths = [path("unknown", 0), path("known", 0)]
    result = greedy_responsibility(paths, ["unknown", "known"],
                                   {"unknown": [1, 0], "known": [0.5, 0.5]}, {"known"})
    assert result["score"] == pytest.approx(0.375)
    assert result["known_mass"] == pytest.approx(1 / 3)
    assert raw_lse_responsibility(paths, {"known"}, temperature=1) == 0.5


def test_bootstrap_preserves_query_macro_not_equal_source_macro():
    values = np.asarray([[1.0], [0.0], [0.0]])
    result = bootstrap_columns(values, ["a", "b", "b"], iterations=1000, seed=13)[0]
    assert result["point_delta"] == pytest.approx(1 / 3)
    assert result["source_groups"] == 2
    assert (result["win_queries"], result["loss_queries"], result["tie_queries"]) == (1, 0, 2)


def test_interaction_algebra_and_candidate_source():
    assert {arm: CONTRASTS["I_N"].get(arm, 0) - CONTRASTS["I_L"].get(arm, 0)
            for arm in CONTRASTS["I_N_minus_L"]} == CONTRASTS["I_N_minus_L"]
    source = candidate_source([{"kind": "direct", "path_score": 1}, path("i", 1, "image")])
    assert source["source"] == "both"
    assert source["evidence_modalities"] == ["image"]
    assert candidate_source([])["source"] == "absent"

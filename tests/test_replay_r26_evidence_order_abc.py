"""Behavioral tests for the independent frozen R26 three-arm replay."""
from copy import deepcopy
import math

import pytest

from replay_r26_evidence_order_abc import ARMS, lse, replay_arms


def path(score: float) -> dict:
    return {"kind": "evidence", "path_score": score}


def test_three_orders_use_fixed_membership_and_witnesses():
    row = {"D100_ANN": [{"target_id": "direct", "direct_score": 1.}],
           "E_paths": [
               {"target_id": "x", "evidence_score": .9, "retained_paths": [path(0.)]},
               {"target_id": "y", "evidence_score": .1, "retained_paths": [path(2.)]}],
           "E_pre_retention": [
               {"target_id": "x", "paths": [path(0.), path(5.)]},
               {"target_id": "y", "paths": [path(2.)]},
               {"target_id": "dropped", "paths": [path(100.)]}]}
    before = deepcopy(row)
    result = replay_arms(row, {"direct": 0., "x": 1., "y": 2.})
    assert result[ARMS[0]]["E"] == ["x", "y"]
    assert result[ARMS[1]]["E"] == ["y", "x"]
    assert result[ARMS[2]]["E"] == ["x", "y"]
    assert all(set(a["Equal"]) == {"direct", "x", "y"} for a in result.values())
    assert row == before


def test_teacher_only_reranks_the_new_c100():
    direct = [{"target_id": f"d{i:03d}", "direct_score": 100.-i} for i in range(100)]
    row = {"D100_ANN": direct, "E_paths": [], "E_pre_retention": []}
    scores = {r["target_id"]: 0. for r in direct}
    for i in range(100):
        t = f"e{i:03d}"
        row["E_paths"].append({"target_id": t, "evidence_score": 100.-i, "retained_paths": [path(float(i))]})
        row["E_pre_retention"].append({"target_id": t, "paths": [path(float(i))]})
        scores[t] = 1000. if i == 99 else 0.
    result = replay_arms(row, scores)
    assert "e099" not in result[ARMS[0]]["T0"]
    assert result[ARMS[1]]["T0"][0] == "e099"
    assert all(len(a["C100"]) == len(a["T0"]) == 100 for a in result.values())


def test_lse_preserves_multiplicity_and_is_stable():
    assert lse([path(1000.), path(1000.)]) == pytest.approx(1000. + math.log(2))
    with pytest.raises(ValueError):
        lse([])


def test_empty_e_channel_preserves_direct_ranking():
    row = {"D100_ANN": [{"target_id": "z", "direct_score": 2.}, {"target_id": "a", "direct_score": 1.}],
           "E_paths": [], "E_pre_retention": []}
    result = replay_arms(row, {"z": 0., "a": 0.})
    assert all(a["Equal"] == ["z", "a"] and a["T0"] == ["a", "z"] for a in result.values())

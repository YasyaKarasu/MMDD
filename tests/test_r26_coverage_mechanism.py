"""Controlled examples for the frozen R26 inference interventions."""
import math

import pytest

from replay_r26_coverage_mechanism import coverage, select, target_arms, rank_scores


def path(eid, score):
    return {"kind": "evidence", "evidence_id": eid, "path_score": score}


def test_greedy_selects_complementary_rows_over_redundant_top_scores():
    paths = [path("a", 3), path("b", 2.9), path("c", 2.8), path("d", 2.7), path("e", 2.6)]
    support = {p["evidence_id"]: [1., 0.] for p in paths}
    support["e"] = [0., 1.]
    chosen = select(paths, support, weighted=True)
    assert [p["evidence_id"] for p in chosen] == ["a", "e"]
    assert coverage(chosen, support) > coverage(paths[:4], support)
    assert len(chosen) < 4  # No useful extra support: matches historical early stop.


def test_no_path_is_invariant_to_path_scores_including_screening():
    paths = [path(f"e{i:02}", 30-i) for i in range(22)]
    support = {p["evidence_id"]: [1., 0.] for p in paths}
    support["e21"] = [0., 1.]
    keys = {p["evidence_id"]: p["evidence_id"] for p in paths}
    def evaluate(values):
        pool = sorted(values, key=lambda p: (-p["path_score"], p["evidence_id"]))[:20]
        chosen = select(pool, support, weighted=True)
        after = {"selected_evidence_ids": [p["evidence_id"] for p in chosen],
                 "evidence_score": coverage(chosen, support), "retained_paths": chosen}
        return target_arms({"paths": values}, after, support, keys)
    original = evaluate(paths)
    changed = evaluate([dict(p, path_score=-p["path_score"]) for p in paths])
    assert original[0]["greedy_no_path"] == changed[0]["greedy_no_path"] == 1.
    assert original[1]["no_path"] == changed[1]["no_path"]
    assert original[0]["greedy_no_path_same_pool"] == .5


def test_fixed_no_weight_and_lme_remove_only_intended_factors():
    paths = [path("a", math.log(3)), path("b", math.log(3))]
    support = {"a": [1., 0.], "b": [0., 1.]}
    after = {"selected_evidence_ids": ["a", "b"], "retained_paths": paths, "evidence_score": .75}
    scores, _, _ = target_arms({"paths": paths}, after, support, {"a": "a", "b": "b"})
    assert scores["fixed_greedy_no_weight"] == 1.
    assert scores["greedy_lse"] == pytest.approx(math.log(6))
    assert scores["greedy_lme"] == pytest.approx(math.log(3))
    assert scores["greedy_max"] == pytest.approx(math.log(3))


def test_budget_expansion_rescues_excluded_teacher_positive_and_keeps_ties():
    row = {"D100_ANN": [{"target_id": "a", "direct_score": 1.}, {"target_id": "b", "direct_score": .9}]}
    evidence = {"c": .9, "d": .8}
    teacher = {"a": 0., "b": 1., "c": 1., "d": 5.}
    small = rank_scores(row, evidence, teacher, 2)
    full = rank_scores(row, evidence, teacher, None)
    assert "d" not in small["C"]
    assert full["T0"] == ["d", "b", "c", "a"]
    assert set(small["C"]) <= set(full["C"])

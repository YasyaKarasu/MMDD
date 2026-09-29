import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.calibration_cohort import choose_cohort


def data():
    pop = [{"query_id": f"q{g}_{i}", "source_group": str(g), "split": "train"}
           for g in range(20) for i in range(1 + g % 3)]
    part = {r["query_id"]: "fit" if int(r["source_group"]) < 15 else "inner_val" for r in pop}
    return pop, part


def test_nested_cohorts_keep_whole_groups_and_fixed_validation():
    pop, part = data()
    a = choose_cohort(pop, part, 5, 3)
    b = choose_cohort(pop[::-1], part, 15, 3)
    assert set(a["query_ids"]) < set(b["query_ids"])
    assert a["source_groups"]["inner_val"] == b["source_groups"]["inner_val"]
    assert not set(b["source_groups"]["fit"]) & set(b["source_groups"]["inner_val"])
    for r in pop:
        if r["source_group"] in a["source_groups"][part[r["query_id"]]]:
            assert r["query_id"] in a["query_ids"]
    assert a == choose_cohort(pop[::-1], part, 5, 3)


def test_rejects_leakage_and_unavailable_budgets():
    pop, part = data()
    with pytest.raises(ValueError, match="Not enough"):
        choose_cohort(pop, part, 100, 3)
    pop[0]["split"] = "dev"
    with pytest.raises(ValueError, match="train"):
        choose_cohort(pop, part, 5, 3)
    pop, part = data()
    part["q1_0"] = "inner_val"
    with pytest.raises(ValueError, match="crosses"):
        choose_cohort(pop, part, 5, 3)

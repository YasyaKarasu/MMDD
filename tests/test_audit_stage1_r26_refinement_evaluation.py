from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from audit_stage1_r26_refinement_evaluation import recall, redistill_decision


def gate_rows():
    return [{"seed":seed,"Tnew_minus_Told_R10":{"overall":.01,"implicit":0.},"lists_with_new_competitors":1} for seed in (13,29)]


def test_endpoint_recall_is_macro_positive_recall_not_hit():
    assert recall(["p1","p1","n"],["p1","p2"],3) == .5


def test_redistill_missing_seed_is_unassessable():
    assert redistill_decision(gate_rows()[:1]) == "unassessable"
    assert redistill_decision(gate_rows()) == "triggered"


@pytest.mark.parametrize("field,value",[("overall",0.),("overall",-.01),("implicit",-.01)])
def test_redistill_requires_both_seed_metric_conditions(field,value):
    rows = gate_rows()
    rows[1]["Tnew_minus_Told_R10"][field] = value
    assert redistill_decision(rows) == "not_triggered"


def test_redistill_requires_actual_new_competitors():
    rows = gate_rows()
    rows[0]["lists_with_new_competitors"] = 0
    assert redistill_decision(rows) == "not_triggered"

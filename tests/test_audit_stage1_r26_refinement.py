"""Synthetic checks for actual-history verification, without model execution."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from audit_stage1_r26_refinement import audit_history


def history_fixture():
    examples = [{"query_id":"q1","source_type":"table","destination_type":"table","candidate_ids":["t1","t2"]},
                {"query_id":"q2","source_type":"table","destination_type":"image","candidate_ids":["i1"]}]
    order = [{"epoch":1,"step":1,"source_rows":[0,1]}, {"epoch":2,"step":2,"source_rows":[1,0]}]
    history = [{**r,"query_ids":[examples[i]["query_id"] for i in r["source_rows"]],
                "relations":[f"{examples[i]['source_type']}->{examples[i]['destination_type']}" for i in r["source_rows"]],
                "pair_slots":3,"loss":.2,"gradient_norms":{"weight":1.,"zero":0.}} for r in order]
    return history, order, examples


def test_refinement_audit_reconstructs_actual_counts():
    actual = audit_history(*history_fixture())
    assert actual["pair_slots"] == 6
    assert actual["relations"] == {"table->table":2,"table->image":2}
    assert actual["gradient_present_steps"] == {"weight":2,"zero":2}
    assert actual["nonzero_gradient_steps"] == {"weight":2,"zero":0}


@pytest.mark.parametrize("field,value",[("step",9),("query_ids",["wrong","q2"]),
                                      ("pair_slots",2),("loss",float("nan")),
                                      ("gradient_norms",{"weight":float("inf")})])
def test_refinement_audit_rejects_corrupted_actual_history(field, value):
    history, order, examples = history_fixture()
    history[0][field] = value
    with pytest.raises(ValueError):
        audit_history(history, order, examples)


def test_refinement_audit_rejects_short_budget_and_duplicate_list_coverage():
    history, order, examples = history_fixture()
    with pytest.raises(ValueError, match="full frozen update"):
        audit_history(history[:-1],order,examples)
    order[0]["source_rows"] = [0,0]
    history[0] = {**deepcopy(order[0]),"query_ids":["q1","q1"],
                  "relations":["table->table"]*2,"pair_slots":4,"loss":.2,"gradient_norms":{"weight":1.}}
    with pytest.raises(ValueError, match="every relation list"):
        audit_history(history,order,examples)

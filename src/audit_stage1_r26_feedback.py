"""Verify every train-fit hard32 against its actual own universe and fixed T0 scores."""
from __future__ import annotations

from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

from mmdd_stage1.r26_feedback import feedback_gate
from prepare_stage1_r26 import OUT, ROOT, file_record
from prepare_stage1_r26_refinement import T0
from run_stage1_r21 import paths, read_rows
from run_stage1_r25 import _json, sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run() -> dict:
    population_path = OUT / "common/feedback_queries.jsonl"
    frozen = {r["query_id"]:r for r in read_rows(population_path)}
    registry_path = paths(ROOT)["train"]
    known = defaultdict(set)
    for row in read_rows(registry_path):
        if row["source_type"] == row["destination_type"] == "table":
            known[row["query_id"]].update(row.get("positive_ids",[]))
            if row.get("positive_id"):
                known[row["query_id"]].add(row["positive_id"])
    gate_path = OUT / "feedback/REFINEMENT_GATE_FROZEN.json"
    gate = json.loads(gate_path.read_text())
    names = ["B13",*gate["selected_generators"],*[f"R26-O-SUP/seed{s}/step178" for s in (13,29)]]
    old,results,changed,dev_metrics = {},[],{},{}
    for name in names:
        directory = OUT / "feedback" / name
        receipt_path = directory / "MINING_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text())
        require(receipt["queries"] == len(frozen) == 11390,"Incomplete mining population")
        require(receipt["teacher"]["sha256"] == sha256(T0),"Hardness uses a different Teacher")
        require(receipt["registry"]["sha256"] == sha256(registry_path),"Mining registry changed")
        for key in ("lists","natural_retrieval"):
            require(sha256(Path(receipt[key]["path"])) == receipt[key]["sha256"],f"Mining source changed: {key}")
        natural_receipt = json.loads(Path(receipt["natural_retrieval"]["path"]).read_text())
        natural_path = Path(natural_receipt["rankings"]["path"])
        require(sha256(natural_path) == natural_receipt["rankings"]["sha256"],"Actual natural pool changed")
        pairs = {} if name == "B13" else {r["query_id"]:r for r in read_rows(directory / "H_old_H_new_comparison.jsonl.gz")}
        seen,unique_counts,member_changes = set(),[],0
        for raw,mined in itertools.zip_longest(read_rows(natural_path),read_rows(Path(receipt["lists"]["path"]))):
            require(raw is not None and mined is not None,"Natural/mining row count differs")
            q = raw["query_id"]
            require(q == mined["query_id"] and q not in seen and q in frozen,"Mining order/population differs")
            seen.add(q)
            positives = set(raw["positive_target_ids"]) | known[q]
            d,e,u = set(raw["rankings"]["D100_ANN"]),set(raw["E_target_ids"]),set(raw["U"])
            require(set(mined["D"]) == d and set(mined["E"]) == e and set(mined["U"]) == u,"Hard list uses a different generator universe")
            require(set(mined["train_known_positive_closure"]) == positives,"Positive closure differs")
            require(set(mined["training_injected_positive_ids"]) == positives-u,"Natural/injected positives conflated")
            scores = mined["T0_scores"]
            require(set(scores) == u|positives and all(math.isfinite(v) for v in scores.values()),"Missing/nonfinite actual T0 scores")
            hard = sorted(u-positives,key=lambda t:(-scores[t],t))[:32]
            require(mined["hard32"] == hard and mined["hard_shortfall"] == 32-len(hard),"Hard32 is not fixed T0 top32 legal negatives")
            require(mined["training_list"] == sorted(positives)+hard,"Training list differs")
            truth = set(frozen[q]["positive_target_ids"])
            require(set(mined["natural_hit_ids"]) == truth & u,"Natural hits differ")
            require(math.isclose(mined["EO_ANN_recall"],len(truth & (e-d))/len(truth),abs_tol=1e-12),"Implicit EO formula differs")
            if name == "B13":
                old[q] = {"hard":hard,"EO":mined["EO_ANN_recall"]}
            else:
                pair = pairs[q]
                a,b = set(hard),set(old[q]["hard"])
                require(pair["hard_membership_changed"] == (a!=b),"Hard member comparison differs")
                require(pair["hard_order_changed"] == (hard!=old[q]["hard"]),"Hard order comparison differs")
                require(pair["new_unique_negatives"] == sorted(a-b),"New competitor identities differ")
                require(math.isclose(pair["jaccard"],len(a&b)/len(a|b),abs_tol=1e-12),"Hard Jaccard differs")
                require(math.isclose(pair["new_unique_negative_fraction"],len(a-b)/len(a),abs_tol=1e-12),"Novel hard fraction differs")
                require(pair["old_EO"] == old[q]["EO"] and pair["new_EO"] == mined["EO_ANN_recall"],"EO comparison differs")
                member_changes += a!=b
                unique_counts.append(len(a-b))
        require(seen == frozen.keys() and (not pairs or pairs.keys() == frozen.keys()),"Incomplete feedback comparison")
        if name != "B13":
            changed[name] = member_changes
        dev = json.loads((OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json").read_text())
        dev_metrics[name] = dev["metrics"]
        result = {"generator":name,"queries":len(seen),"hard_membership_changed":member_changes if name != "B13" else None,
                  "new_competitor_slots":sum(unique_counts),"mining_receipt":file_record(receipt_path),
                  "execution_status":"ran","scientific_validity":"valid"}
        results.append(result)
        print(json.dumps({"feedback_invariants_verified":name,"queries":len(seen)}),flush=True)
    measured = feedback_gate(dev_metrics,changed)
    require(measured["status"] == gate["status"] and measured.get("selected_generators") == gate.get("selected_generators"),"Frozen priority/health/membership gate does not reproduce")
    result = {"execution_status":"ran","scientific_validity":"valid","test_ids":["F01","F02"],"generators":results,
              "gate_recomputed":measured,"frozen_gate":file_record(gate_path),"population":file_record(population_path),
              "registry":file_record(registry_path),"teacher":file_record(T0),"code":file_record(Path(__file__)),
              "scope":"All5 generator x11390 train-fit universes, closure, actual saved T0 score ordering, natural/injected distinction, H comparisons, and measured gate. Does not claim all T0 pairs were numerically recomputed."}
    _json(OUT / "acceptance/FEEDBACK_AUDIT.json",result)
    return {"generators":len(results),"queries":len(results)*len(frozen),"gate":measured["status"]}


if __name__ == "__main__":
    print(json.dumps(run()))

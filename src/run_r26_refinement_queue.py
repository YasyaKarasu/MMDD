"""Measure the fixed-priority feedback gate, then run its single continuation round."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

from mmdd_stage1.r26_feedback import feedback_gate, PRIORITY
from prepare_stage1_r26 import ROOT, OUT, file_record
from prepare_stage1_r26_refinement import freeze_protocol
from run_r26_followups import await_artifact, execute
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json, sha256


def gate_inputs() -> tuple[dict,list[dict],dict[str,int]]:
    metrics, evidence, changed = {}, [], {}
    names = ["B13", *[f"R26-{arm}/seed{seed}/step178" for arm in PRIORITY for seed in (13,29)]]
    for name in names:
        own = OUT / "rankings" / name / "RETRIEVAL_RECEIPT.json"
        if own.exists():
            data = json.loads(own.read_text())
            metrics[name] = data["metrics"]
            evidence.append({"generator":name,"retrieval":file_record(own)})
        comparison = OUT / "feedback" / name / "H_old_H_new_comparison.jsonl.gz"
        if comparison.exists():
            rows = list(read_rows(comparison))
            frozen = {r["query_id"] for r in read_rows(OUT / "common/feedback_queries.jsonl")}
            if len(rows) != len(frozen) or {r["query_id"] for r in rows} != frozen:
                raise ValueError("Incomplete actual hard-list comparison")
            changed[name] = sum(r["hard_membership_changed"] for r in rows)
    return metrics, evidence, changed


def freeze_evaluation(gate: dict) -> None:
    names = ["B13", *gate["selected_generators"]]
    inputs = []
    for name in names:
        path = OUT / "rankings" / name
        receipt = json.loads((path / "RETRIEVAL_RECEIPT.json").read_text())
        if sha256(path / "rankings.jsonl.gz") != receipt["rankings"]["sha256"]:
            raise ValueError("Feedback generator's dev ranking changed")
        inputs.append({"generator":name,"receipt":file_record(path / "RETRIEVAL_RECEIPT.json"),
                       "rankings":file_record(path / "rankings.jsonl.gz")})
    protocol = {"version":"R26_refinement_evaluation_v1", "pools":inputs,
                "population":file_record(OUT / "common/dev_queries.jsonl"),
                "K":[10,20,30,40,50], "seeds":[13,29], "arms":["Tcont","Told","Tnew"],
                "primary_redistill_gate_pool":"Each paired seed's selected new generator, Equal BT100; identical pool for Told and Tnew",
                "secondary_controls":"B13 Equal BT100 and Direct100; selected generator Direct100; natural U/M offline diagnostic",
                "statistics":"10,000 source-cluster bootstrap draws, seed260914; average model seeds per query before paired inference",
                "redistill_rule":"Tnew minus Told overall R10 strictly positive and implicit R10 nonnegative in BOTH paired seeds, plus actual newly added hard competitors",
                "no_selection":"Only complete10536-update endpoint; one continuation recipe, no grid or best-checkpoint selection"}
    path = OUT / "feedback/REFINEMENT_EVALUATION_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Frozen refinement evaluation changed")
    if not path.exists():
        _json(path, protocol)


def run() -> dict:
    freeze_protocol()
    gate_path = OUT / "feedback/REFINEMENT_GATE_FROZEN.json"
    while not gate_path.exists():
        metrics, evidence, changed = gate_inputs()
        decision = feedback_gate(metrics, changed)
        _json(OUT / "feedback/REFINEMENT_GATE_CURRENT.json",{**decision,"inputs":evidence})
        eligible = decision.get("eligible_for_mining", [])
        if decision["status"] == "unassessable" and not eligible:
            print(json.dumps({"gate":decision["reason"],"last_arm":decision["candidates"][-1]["arm"] if decision["candidates"] else None}),flush=True)
            time.sleep(20)
            continue
        if eligible:
            for name in eligible:
                if not (OUT / "train_retrieval/feedback" / name / "RETRIEVAL_RECEIPT.json").exists():
                    execute("retrieve_stage1_r26_train.py",["--generator",name,"--population","feedback","--device","cuda:1"],
                            "feedback_selected_pool_"+name.replace("/","_")+".log")
            # Original controls and selected-new mining each keep one writer.
            # Per-query copying reuses completed B13 scores unchanged.
            await_artifact(OUT / "feedback/PRIORITY_CACHE_COPY.json")
            missing = [n for n in eligible if not (OUT / "feedback" / n / "H_old_H_new_comparison.jsonl.gz").exists()]
            if missing:
                execute("mine_stage1_r26_feedback.py",["--device","cpu",
                        "--cache-path",str(OUT / "feedback/train_T0_pairs_priority.sqlite"),
                        "--diagnostics-name","PRIORITY_HARD_MEMBERSHIP_DIAGNOSTICS.json",
                        *[a for n in missing for a in ("--generator",n)]],
                        "feedback_priority_mining.log")
            continue
        decision.update({"execution_status":"ran","scientific_validity":"valid","inputs":evidence,
                         "protocol":file_record(OUT / "feedback/REFINEMENT_PROTOCOL.json"),
                         "actual_hard_comparisons":[file_record(OUT / "feedback" / n / "H_old_H_new_comparison.jsonl.gz") for n in changed],
                         "code":file_record(Path(__file__))})
        _json(gate_path,decision)
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "triggered":
        return gate
    freeze_evaluation(gate)
    execute("prepare_stage1_r26_refinement.py",[],"refinement_prepare.log")
    children = []
    # Each small Teacher fits alongside the other two arms on the same4090.
    # All arms within a seed share a GPU and identical list order.
    for seed,device in ((13,"cuda:0"),(29,"cuda:1")):
        for arm in ("Tcont","Told","Tnew"):
            log = (OUT / "logs" / f"refinement_{arm}_{seed}.log").open("w")
            child = subprocess.Popen([sys.executable,str(ROOT / "src/train_stage1_r26_refinement.py"),
                                      "--arm",arm,"--seed",str(seed),"--device",device],cwd="/tmp",stdout=log,stderr=subprocess.STDOUT)
            children.append((arm,seed,child,log))
    failures = []
    for arm,seed,child,log in children:
        code = child.wait()
        log.close()
        if code:
            failures.append({"arm":arm,"seed":seed,"code":code})
    if failures:
        _json(OUT / "feedback/refinement/TRAINING_FAILURES.json",failures)
        raise RuntimeError("Refinement job failed; preserve and inspect its log")
    result = {"execution_status":"ran","scientific_validity":"valid",
              "jobs":[file_record(OUT / "feedback/refinement" / arm / f"seed{seed}" / "TRAINING_RECEIPT.json")
                      for arm,seed,_,_ in children], "gate":file_record(gate_path)}
    _json(OUT / "feedback/REFINEMENT_TRAINING_RECEIPT.json",result)
    execute("evaluate_stage1_r26_refinement.py",["--device","cuda:1"],"refinement_evaluation.log")
    return result


if __name__ == "__main__":
    print(json.dumps(run()))

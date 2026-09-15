"""Freeze one Teacher-refinement recipe and materialize its actual five-relation lists."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.r26_feedback import PRIORITY,augment_teacher_lists
from prepare_stage1_r26 import ROOT,OUT,file_record,stable_sha
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r25 import _json
from train_stage1_r26 import known_edges

R22 = ROOT / "work/stage1_optimization_r22_20260911"
T0 = R22 / "fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"


def freeze_protocol() -> dict:
    protocol = {"version":"R26_one_refinement_v1","priority":PRIORITY,"parent":file_record(T0),
        "base_lists":file_record(R22 / "manifests/full_natural.jsonl"),
        "historical_hard":file_record(R22 / "fresh_lineage/S0_mining/seed13/hard_negatives.jsonl.gz"),
        "historical_T0_recipe":file_record(R22 / "fresh_lineage/T1-B/seed13/config.json"),
        "registry":file_record(paths(ROOT)["train"]),"teacher_seeds":[13,29],"arms":["Tcont","Told","Tnew"],
        "optimizer":{"kind":"fresh AdamW from the identical fixed T0 weights","lr":5e-5,"weight_decay":.01},
        "batch_lists":8,"microbatch_lists":2,"epochs":2,"updates_per_epoch":5268,"total_updates":10536,
        "shuffle":"random.Random(260914+seed), shared order per seed across all three arms; sequential epoch shuffles",
        "hard_budget":32,"list_budget":42143,"QT_lists":12041,"QT_query_population":11390,
        "list_rule":"Preserve all historical five-relation lists and their multiplicities. Append this arm's hard32 only to QT candidates; apply identical full train-known positive closure within candidate membership in every arm.",
        "Tcont":"Original T0 training lists: full_natural plus original S0_mining hard32, with common closure protection",
        "Told":"Same base five-relation lists plus historical B13 generator's current T0 hard32",
        "Tnew":"Same base five-relation lists plus selected healthy new generator's current T0 hard32; paired Student seed",
        "evaluation":"Same frozen own pools per generator for Tcont/Told/Tnew; query-macro R10/20/30/40/50, paired source bootstrap; no best-checkpoint selection",
        "initialization":"No fresh Teacher initializer; all trainable components continue the same T0, fresh optimizer in all controls",
        "budget_rationale":"Reuse actual T0 two-epoch/batch8/microbatch2/LR5e-5/wd.01 recipe; one fixed refinement only, no grid"}
    path = OUT / "feedback/REFINEMENT_PROTOCOL.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise ValueError("Frozen Teacher refinement recipe changed")
    if not path.exists():
        _json(path,protocol)
    return protocol


def prepare(freeze_only: bool) -> dict:
    protocol = freeze_protocol()
    if freeze_only:
        return {"frozen_protocol":str(OUT / "feedback/REFINEMENT_PROTOCOL.json"),"total_updates_per_teacher":protocol["total_updates"]}
    gate_path = OUT / "feedback/REFINEMENT_GATE_FROZEN.json"
    gate = json.loads(gate_path.read_text())
    if gate["status"] != "triggered":
        raise ValueError("Teacher refinement requires the real triggered feedback gate")
    base = load_edge_examples(R22 / "manifests/full_natural.jsonl",split="train")
    if len(base) != protocol["list_budget"]:
        raise ValueError("Teacher base list count changed")
    population = {r["query_id"] for r in read_rows(OUT / "common/feedback_queries.jsonl")}
    if {e.query_id for e in base if e.source_type == e.destination_type == "table"} != population:
        raise ValueError("Teacher QT population differs from full train-fit mining")
    historical = {r["query_id"]:r["hard_candidate_ids"] for r in read_rows(R22 / "fresh_lineage/S0_mining/seed13/hard_negatives.jsonl.gz")}
    known = known_edges()
    directory = OUT / "feedback/refinement"
    receipts = []
    for seed,name in zip((13,29),gate["selected_generators"]):
        rng = random.Random(260914+seed)
        order = list(range(len(base)))
        batches = []
        for epoch in (1,2):
            rng.shuffle(order)
            batches.extend({"epoch":epoch,"source_rows":order[i:i+8]} for i in range(0,len(order),8))
        order_path = directory / f"order_seed{seed}.jsonl"
        for step,row in enumerate(batches,1):
            row["step"] = step
        write_rows(order_path,batches)
        for arm,hard_path in (("Tcont",None),("Told",OUT / "feedback/B13/hard_lists.jsonl.gz"),
                              ("Tnew",OUT / "feedback" / name / "hard_lists.jsonl.gz")):
            hard = historical if hard_path is None else {r["query_id"]:r["hard32"] for r in read_rows(hard_path)}
            if not population.issubset(hard):
                raise ValueError("Missing real training hard list")
            examples,audit = augment_teacher_lists(base,hard,known)
            records = []
            for index,example in enumerate(examples):
                record = asdict(example)
                record["positive_id"] = example.candidate_ids[example.positive_index] if example.positive_index >= 0 else None
                record["source_list_index"] = index
                # Absent cached logits must be omitted for the production loader.
                records.append({k:v for k,v in record.items() if v is not None})
            path = directory / arm / f"seed{seed}" / "lists.jsonl"
            write_rows(path,records)
            if load_edge_examples(path,split="train") != examples:
                raise ValueError("Serialized refinement lists changed actual training examples")
            non_qt_changed = sum(a.candidate_ids != b.candidate_ids for a,b in zip(base,examples) if a.source_type != "table" or a.destination_type != "table")
            if non_qt_changed:
                raise ValueError("Hard mining changed a non-QT candidate list")
            result = {**audit,"list_count":len(examples),"arm":arm,"seed":seed,"generator":name if arm == "Tnew" else "B13" if arm == "Told" else "historical_S0_mining",
                "protocol":file_record(OUT / "feedback/REFINEMENT_PROTOCOL.json"),"gate":file_record(gate_path),"lists":file_record(path),
                "order":file_record(order_path),"hard_source":file_record(hard_path) if hard_path else protocol["historical_hard"],
                "logical_list_sha":stable_sha(records),"relations":dict(Counter(f"{e.source_type}->{e.destination_type}" for e in examples)),
                "non_QT_candidate_lists_changed":non_qt_changed,"code":file_record(Path(__file__))}
            _json(path.parent / "LISTS_RECEIPT.json",result)
            receipts.append(file_record(path.parent / "LISTS_RECEIPT.json"))
    _json(directory / "PREPARATION_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid","jobs":receipts})
    return {"prepared_jobs":len(receipts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only",action="store_true")
    print(json.dumps(prepare(parser.parse_args().freeze_only)))

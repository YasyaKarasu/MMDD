"""Independent replay invariants, evidence attribution and cell-length control."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mmdd_dataset.utils import values_match
from mmdd_stage2.join_diagnostic import digest, read_rows, recall, spearman, write_json
from run_r26_joinability_diagnostic import OUT, R26
from analyze_r26_joinability_diagnostic import Scores, write_csv


def audit(out: Path) -> None:
    torch.set_num_threads(4)
    plan = json.loads((out / "prepared.json").read_text())
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    assert digest(plan) == protocol["prepared_sha"]
    summary = json.loads((out / "SUMMARY.json").read_text())
    trained_audit = []
    for seed in (13,29):
        fitted = torch.load(out / f"projection_seed{seed}.pt",map_location="cpu",weights_only=False)
        initial = torch.load(out / f"projection_random_seed{seed}.pt",map_location="cpu",weights_only=False)
        assert fitted["epoch"] >= 1 and initial["epoch"] == 0
        delta = float((fitted["state_dict"]["projection.weight"]-initial["state_dict"]["projection.weight"]).norm())
        assert delta > 0
        trained_audit.append({"seed":seed,"epoch":fitted["epoch"],"parameter_delta_norm":delta})
    ranks = {(r["generator"],r["query_id"],r["condition"],r["arm"]):r for r in read_rows(out / "replay_rankings.jsonl")}
    source_sets = {k:set(v) for k,v in protocol["source_groups"].items()}
    assert not source_sets["train"] & source_sets["calibration"]
    assert not source_sets["train"] & source_sets["eval"]
    assert not source_sets["calibration"] & source_sets["eval"]
    gold = {(e["query_id"],e["target_id"]):e for e in plan["examples"] if e["split"] == "eval"}
    audits, evidence_rows, details = [], [], []
    for generator, prepared in plan["pilots"].items():
        original = read_rows(R26 / f"stage2/pilot/{generator}/results.jsonl")
        assert len(original) == len(prepared) == 192
        assert read_rows(R26 / f"stage2/inputs/{generator}/retrieval.jsonl") == plan["inputs"][generator]
        original_map = {(r["query_id"],r["condition"]):r for r in original}
        no_e = {(r["query_id"],c["target_id"]):c for r in original if r["condition"] == "NoE-fill" for c in r["diagnostic_candidates"]}
        current = {}
        for row in prepared:
            original_row = original_map[row["query_id"],row["condition"]]
            clone = json.loads(json.dumps(row))
            for c in clone["diagnostic_candidates"]:
                c.pop("recovered_key",None)
            assert clone == original_row
            qid,condition = row["query_id"],row["condition"]
            baseline = next(r for r in plan["inputs"][generator] if r["query_id"] == qid)
            pool = {c["target_id"] for c in baseline["results"]}
            assert ranks[generator,qid,condition,"A"]["ranking"] == original_row["ranking"]
            for arm in ("A","B","C13","C29"):
                rr = ranks[generator,qid,condition,arm]
                assert set(rr["ranking"]) == (pool if row["status"] == "ran" else set())
                # The original T0 rank breaks ties in every B/C implementation.
                assert len(rr["ranking"]) == len(set(rr["ranking"]))
            current[qid,condition] = row
            for c in row["diagnostic_candidates"]:
                key = (qid,c["target_id"])
                if key not in gold or not c.get("recovered_key"):
                    continue
                e = gold[key]
                correct_column = f"target:{c['target_id']}:{c['selection']['column_index']}" == e["gold"]
                vals = plan["columns"][c["recovered_key"]]["values"]
                truths = plan["columns"][e["anchor"]]["values"]
                assert len(vals) == len(truths)
                no_values = [r["value"] for r in no_e[key]["branches"]["evidence"]["rows"]]
                correct = [correct_column and values_match(v,t) for v,t in zip(vals,truths)]
                added_correct = [ok and not values_match(n,t) for ok,n,t in zip(correct,no_values,truths)]
                details.append({"generator":generator,"condition":condition,"query_id":qid,"target_id":c["target_id"],"kind":e["kind"],
                    "correct_column":correct_column,"rows":len(vals),"nonempty":sum(bool(v.strip()) for v in vals),
                    "source_correct_gold_attribute_cells":sum(correct),"additional_correct_vs_NoE":sum(added_correct)})
        for condition in ("Real-crop","Real-crop+original","NoE-fill"):
            for arm in ("A","B","C13","C29"):
                records = []
                for (qid,cond),row in current.items():
                    if cond != condition:
                        continue
                    rr = ranks[generator,qid,condition,arm]
                    for tid in rr["ranking"][:9]:
                        if rr["branches"].get(tid) != "evidence":
                            continue
                        d = next((d for d in details if d["generator"] == generator and d["condition"] == condition and d["query_id"] == qid and d["target_id"] == tid),None)
                        records.append(d)
                known = [d for d in records if d is not None]
                evidence_rows.append({"generator":generator,"condition":condition,"arm":arm,"evidence_top9":len(records),
                    "qrel_positive_E_top9":len(known),"correct_gold_column_E_top9":sum(d["correct_column"] for d in known),
                    "source_correct_gold_cells_E_top9":sum(d["source_correct_gold_attribute_cells"] for d in known),
                    "additional_correct_vs_NoE_E_top9":sum(d["additional_correct_vs_NoE"] for d in known),
                    "qrel_positive_E_with_any_additional_correct_cell":sum(d["additional_correct_vs_NoE"] > 0 for d in known)})
        audits.append({"generator":generator,"original_rows":len(original),"failed_rows":sum(r["status"] != "ran" for r in original),"unchanged_inputs":True,"unchanged_generated_values":True,"unchanged_selections":True,"replay_pool_identity":True})
    # Direct100 is an offline attribution label only. Read its original R26 ranking once.
    strict_rows = []
    with (out / "table_width.csv").open() as handle:
        widths = list(csv.DictReader(handle))
    width_lookup = {(r["query_id"],r["target_id"],r["arm"]):r for r in widths}
    strict_sets = {}
    for generator in plan["pilots"]:
        original_ranks = read_rows(R26 / f"rankings/{generator}/rankings.jsonl.gz")
        direct = {r["query_id"]:set(r["rankings"]["D100_ANN"]) for r in original_ranks if r["query_id"] in plan["populations"]["eval"]}
        strict = {(qid,tid) for qid,tid in gold if plan["population"][qid]["query_kind"] == "implicit" and tid not in direct[qid]}
        strict_sets[generator] = strict
        for arm in ("A","B","C13","C29"):
            vals = [width_lookup[q,t,arm] for q,t in strict]
            strict_rows.append({"generator":generator,"arm":arm,"strict_evidence_only_gt_pairs":len(vals),
                "hypothetical_direct_accept_rate":float(np.mean([r["accepted"] == "True" for r in vals]))})
    # Source-wise recall attribution does not equate branch switching with evidence causality.
    attribution = []
    for generator in plan["pilots"]:
        for condition in ("Real-crop","Real-crop+original","NoE-fill"):
            qids = plan["populations"]["eval"]
            for arm in ("A","B","C13","C29"):
                hits, total, available, input_hits = 0,0,0,0
                for q,t in strict_sets[generator]:
                    total += 1
                    hits += t in ranks[generator,q,condition,arm]["ranking"][:9]
                    initial = ranks[generator,q,condition,"T0"]["ranking"]
                    available += t in initial
                    input_hits += t in initial[:9]
                attribution.append({"generator":generator,"condition":condition,"arm":arm,"strict_pairs":total,"strict_in_C18":available,"strict_input_R9":input_hits,"strict_hits_R9":hits})
    write_csv(out / "evidence_attribution.csv",evidence_rows)
    write_csv(out / "gold_attribute_recovery.csv",details)
    write_csv(out / "strict_evidence_only.csv",strict_rows)
    write_csv(out / "strict_replay.csv",attribution)
    # Length control: nested target-cell subsets for the same A-selected column pair.
    scorer = Scores(out,plan)
    arows = [r for r in widths if r["arm"] == "A"]
    length_rows = []
    rng = np.random.default_rng(260915)
    for kind in ("implicit","explicit"):
        for positive in ("True","False"):
            chosen = sorted([r for r in arows if r["kind"] == kind and r["positive"] == positive],key=lambda r:digest([r["query_id"],r["target_id"]]))[:64]
            for r in chosen:
                qkeys = plan["tables"]["query"][r["query_id"]]
                tkeys = plan["tables"]["target"][r["target_id"]]
                qk,tk = max(((q,t) for q in qkeys for t in tkeys),key=lambda pair:scorer.scalar(*pair,"A"))
                q = plan["columns"][qk]["values"]
                t = [v for v in plan["columns"][tk]["values"] if v.strip()]
                if not q or not t:
                    continue
                cos = scorer.cell_vectors[[scorer.cell_index[v] for v in q]] @ scorer.cell_vectors[[scorer.cell_index[v] for v in t]].T
                exact = np.array([[bool(v.strip()) and values_match(v,w) for w in t] for v in q])
                for draw in range(8):
                    order = rng.permutation(len(t))
                    for budget in sorted({min(b,len(t)) for b in (1,5,10,20,50,len(t))}):
                        scores = cos[:,order[:budget]].amax(1).numpy()
                        matches = exact[:,order[:budget]].any(1)
                        coverage = np.mean([bool(v.strip()) and (ex or s >= .8) for v,ex,s in zip(q,matches,scores)])
                        length_rows.append({"query_id":r["query_id"],"target_id":r["target_id"],"kind":kind,"positive":positive == "True",
                            "target_rows":len(t),"budget":budget,"draw":draw,"coverage":float(coverage),"raw_mean_max_cosine":float(scores.mean())})
    write_csv(out / "target_length_subsampling.csv",length_rows)
    calibration_tpr = {}
    cal = [e for e in plan["examples"] if e["split"] == "calibration" and e["kind"] == "explicit"]
    for arm in ("A","B","C13","C29"):
        accepted = [scorer.cell(e["anchor"],e["gold"])["coverage"] >= .6 if arm == "A" else scorer.cosine(e["anchor"],e["gold"],arm) >= summary["thresholds"][arm] for e in cal]
        calibration_tpr[arm] = {"pairs":len(cal),"acceptance":float(np.mean(accepted))}
    validation = {"audit":audits,"trained_projections":trained_audit,"calibration_explicit_tpr":calibration_tpr,"prepared_hash_valid":True,"source_split_disjoint":True,"replay_entries":len(ranks),
        "strict_evidence_only":strict_rows,"strict_replay":attribution,"evidence_attribution":evidence_rows,
        "length_control_records":len(length_rows),"limitations":["Direct100 exclusion is a retrieval-reachability label, not proof of no direct join.","Source-correct added cells vs NoE are not proof of grounding in supplied evidence."]}
    write_json(out / "VALIDATION.json",validation)
    print(json.dumps({"audits":audits,"replay_entries":len(ranks),"length_records":len(length_rows)}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    args = parser.parse_args()
    audit(args.output)

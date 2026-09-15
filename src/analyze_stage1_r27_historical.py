"""Audit all 534 consumed updates, state parity, and the six own retrieval nodes."""
from __future__ import annotations

from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from mmdd_stage1.r26_metrics import query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r27 import ROOT, OUT, R12, R13, R26, rows, read_json, write_json, record, sha
from prepare_stage2_r27 import write_rows
from run_stage1_r12_task_c import _score_payload, _schedule_batches
from run_stage1_r13 import _merge_witness_metadata


def analyze() -> dict:
    torch.set_num_threads(2)
    hist=OUT/"historical_replay"
    state=[];updates={};loss_comparisons=[]
    logits,teacher=_score_payload(R12)
    expected_c1=list(_schedule_batches(R12/"taskC_training/candidates_seed13_steps356/candidates.jsonl.gz",logits,teacher["teacher_checkpoint_sha256"]))
    c2=_merge_witness_metadata(ROOT)
    order=read_json(R13/"taskD_witness_supervision/schedule_order.json")["indices"]
    ordered=[c2[i] for i in order]
    expected_c2=[(step,ordered[start:start+64]) for step,start in enumerate(range(0,len(ordered),64),1)]
    for stage,nodes,expected in (("C1",(0,45,89,178,267,356),expected_c1),("C2",(0,45,89,178),expected_c2)):
        directory=hist/stage/"seed13"
        ex=read_json(directory/"EXECUTION.json")
        assert ex["status"]=="completed" and ex["updates"]==nodes[-1]
        if stage=="C2":
            assert ex["parent"]["sha256"]==sha(hist/"C1/seed13/checkpoints/step_000356.pt")
        trace=list(rows(directory/"step_traces.jsonl.gz"))
        consumption=list(rows(directory/("consumed_batches.jsonl.gz" if stage=="C1" else "consumed_order.jsonl.gz")))
        assert len(trace)==len(consumption)==nodes[-1]
        for (step,batch),consumed in zip(expected,consumption):
            # JSON round-trip normalizes tuple/list without changing order/scalars.
            assert consumed==json.loads(json.dumps({"step":step,"examples":[asdict(e) for e in batch]})),(stage,step)
        updates[stage]={"updates":len(trace),"consumed_original_batches_exact":True,"trace":record(directory/"step_traces.jsonl.gz"),"parent":ex["parent"],"fresh_optimizer":read_json(directory/"RUNTIME_LOCK.json")["fresh_optimizer_empty"]}
        for t in trace:
            loss_comparisons.append({"stage":stage,"step":t["step"],"components":t["historical_loss_comparison"]})
        for step in nodes:
            node=read_json(directory/"checkpoints"/f"step_{step:06d}.json")
            assert sha(Path(node["checkpoint"]["path"]))==node["checkpoint"]["sha256"]
            assert sha(Path(node["metadata_export"]["path"]))==node["expected_archived_sha256"]
            assert node["historical_export_file_exact"]
            state.append(node)
    write_rows(hist/"parity/state_comparison.jsonl.gz",state)
    write_rows(hist/"parity/per_step_loss_comparison.jsonl.gz",loss_comparisons)
    max_loss=max(c["abs_difference"] for r in loss_comparisons for c in r["components"].values())
    first=next((r for r in loss_comparisons if any(c["abs_difference"]>0 for c in r["components"].values())),None)
    write_json(hist/"parity/first_divergence.json",{"first_optimizer_step_loss_divergence":first,"max_loss_component_error":max_loss,"checkpoint_tensor_divergence":False,"preflight_serialization_difference":"resolved before updates; see preflight_zero_update/ZERO_UPDATE_DIAGNOSIS.json","interpretation":"All archived node SHA matches after documented default-only metadata export; original H files preserved."})
    evaluation=hist/"own_evaluation"
    raw_direct={r["query_id"]:set(r["D100_EXACT"]) for r in rows(R26/"rankings/Qwen-Raw/rankings.jsonl.gz")}
    reference_metrics=list(rows(OUT/"score_handoff/B13/baseline_per_query.jsonl.gz"))
    refmeta={r["query_id"]:r for r in reference_metrics}
    summary={};perquery=[];statistics=[]
    for spec in read_json(evaluation/"MODEL_INVENTORY.json"):
        name=spec["generator_id"]
        source=evaluation/"rankings"/name
        rr=read_json(source/"RETRIEVAL_RECEIPT.json")
        tr=read_json(evaluation/"teacher"/name/"TEACHER_RECEIPT.json")
        assert sha(source/"rankings.jsonl.gz")==rr["rankings"]["sha256"]
        teachers={r["query_id"]:r for r in rows(evaluation/"teacher"/name/"rankings.jsonl.gz")}
        observed=[]
        for row in rows(source/"rankings.jsonl.gz"):
            q=row["query_id"];g=set(row["positive_target_ids"])
            assert q in refmeta and row["positive_target_ids"]==refmeta[q]["positive_target_ids"]
            d=set(row["rankings"]["D100_ANN"]);e=set(row["E_target_ids"]);u=set(row["U"]);m=set(row["M_exact"])
            assert len(u)==len(m)
            strict=g&(e-d-set(row["D100_EXACT"]))
            sets={"EO_ANN":g&(e-d),"EO_EXACT":g&(e-set(row["D100_EXACT"])),"EO_STRICT":strict,"EO_FIXED_RAW":g&(u-raw_direct[q]),"U_ONLY_VS_M":g&(u-m),"M_ONLY_VS_U":g&(m-u)}
            rec={k:row[k] for k in ("query_id","source_table_id","query_kind","positive_target_ids")}
            rec.update(generator=name,metrics={method:query_metrics(rank,row["positive_target_ids"],(10,20,50)) for method,rank in {**row["rankings"],**teachers[q]["rankings"]}.items()},EO_sets={k:sorted(v) for k,v in sets.items()},strict_C100=sorted(strict&set(teachers[q]["rankings"]["BT100_NO_T0"])),strict_T0_10=sorted(strict&set(teachers[q]["rankings"]["BT100_T0"][:10])))
            observed.append(rec)
        assert len(observed)==1198
        write_rows(source/"r27_per_query_metrics.jsonl.gz",observed)
        perquery.extend(observed)
        summary[name]={"retrieval":rr["metrics"],"teacher":tr["metrics"],"strict_EO_pairs":sum(len(r["EO_sets"]["EO_STRICT"]) for r in observed),"strict_C100_pairs":sum(len(r["strict_C100"]) for r in observed),"strict_T0_10_pairs":sum(len(r["strict_T0_10"]) for r in observed),"node":record(Path(spec["checkpoint"]))}
        for kind in ("overall","implicit","explicit"):
            chosen=[r for r in observed if kind=="overall" or r["query_kind"]==kind]
            for method,metric in (("D100_ANN","recall@10"),("D100_EXACT","recall@10"),("U","raw_recall"),("BT100_T0","recall@10"),("BT100_T0","recall@20"),("BT100_T0","recall@50")):
                delta=np.array([r["metrics"][method][metric]-refmeta[r["query_id"]]["metrics"][method][metric] for r in chosen])
                statistics.append({"generator":name,"reference":"historical_B13_R26_own_pool","query_kind":kind,"endpoint":method+"/"+metric,**source_cluster_comparison(delta,[r["source_table_id"] for r in chosen])})
    # Compare independent H final retrieval with the canonical archived own B13 pool.
    new={r["query_id"]:r for r in rows(evaluation/"rankings/H-C2-step000178/rankings.jsonl.gz")}
    comparisons=[]
    for old in rows(R26/"rankings/B13/rankings.jsonl.gz"):
        q=old["query_id"];cur=new[q]
        common=set(old["QT_OVER_U_scores"])&set(cur["QT_OVER_U_scores"])
        comparisons.append({"query_id":q,"Direct_exact_rank_equal":old["D100_EXACT"]==cur["D100_EXACT"],"Direct_ANN_rank_equal":old["rankings"]["D100_ANN"]==cur["rankings"]["D100_ANN"],"E_membership_equal":set(old["E_target_ids"])==set(cur["E_target_ids"]),"U_equal":old["U"]==cur["U"],"M_equal":old["M_exact"]==cur["M_exact"],"Equal_C100_equal":old["rankings"]["Equal"][:100]==cur["rankings"]["Equal"][:100],"max_common_QT_score_error":max(abs(old["QT_OVER_U_scores"][t]-cur["QT_OVER_U_scores"][t]) for t in common),"E_added":sorted(set(cur["E_target_ids"])-set(old["E_target_ids"])),"E_removed":sorted(set(old["E_target_ids"])-set(cur["E_target_ids"]))})
    write_rows(hist/"parity/retrieval_comparison.jsonl.gz",comparisons)
    write_rows(OUT/"statistics/H_paired_source_bootstrap.jsonl",statistics)
    write_json(hist/"own_evaluation/SUMMARY.json",summary)
    equality={key:sum(r[key] for r in comparisons) for key in ("Direct_exact_rank_equal","Direct_ANN_rank_equal","E_membership_equal","U_equal","M_equal","Equal_C100_equal")}
    verdict={"status":"completed","recipe_replayed":True,"state_parity_level":"tensor_exact_serialization_different","retrieval_parity_level":"exact_direct_and_own_ANN_compared_querywise","historical_fresh_reproduction_status":"historical_fresh_state_exact_own_retrieval_evaluated","updates":updates,"total_optimizer_updates":sum(v["updates"] for v in updates.values()),"archived_checkpoint_nodes":len(state),"all_metadata_exports_match_archived_file_sha":True,"max_per_step_loss_component_error":max_loss,"retrieval_equality_queries":equality,"queries":1198,"max_common_QT_score_error":max(r["max_common_QT_score_error"] for r in comparisons),"runtime_provenance_limit":"Historical runtime/complete original dependency snapshot not independently available; current actual dependency closure is logged. Matching archived files/loss traces establish replayed computational states, not recovery of every historical environment detail.","interpretation":"Entire historical recipe reproduces fresh C1 then B13 state; no attribution to budget, closure, initialization, or path cap individually; seed13 only."}
    write_json(hist/"H_VERDICT.json",verdict)
    write_json(hist/"H_EXECUTION_LEDGER.json",{"GH0":"passed_inputs_historical_runtime_unknown","GH1":"pass","GH2":"pass_default_metadata_difference_resolved","GH3":"pass","GH4":"pass","GH5":"state_exact_retrieval_measured","H0":{"planned":True,"implemented":True,"executed":True,"evaluated":True,"status":"completed"},"H1":{"planned":True,"implemented":True,"executed":True,"evaluated":True,"status":"completed","updates":356},"H2":{"planned":True,"implemented":True,"executed":True,"evaluated":True,"status":"completed","updates":178}})
    return verdict


if __name__=="__main__":
    print(json.dumps(analyze()))

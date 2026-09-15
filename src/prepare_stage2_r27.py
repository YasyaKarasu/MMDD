"""Lock the strict-EO panel and replay the original C18 results without generation."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from mmdd_stage1.r26_metrics import population_metrics, query_metrics
from prepare_stage1_r27 import ROOT, OUT, R26, read_json, write_json, record, rows, stable_sha


def write_rows(path: Path, records) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,"wt") if path.suffix==".gz" else path.open("w")) as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


def prepare() -> dict:
    baseline=read_json(OUT/"score_handoff/B13/BASELINE_REPLAY.json")
    assert baseline["status"]=="pass" and baseline["counts"]["EO_STRICT"]==207
    frame=[]
    retrieval={}
    for row in rows(R26/"rankings/B13/rankings.jsonl.gz"):
        direct=set(row["rankings"]["D100_ANN"])|set(row["D100_EXACT"])
        strict=set(row["positive_target_ids"])& (set(row["E_target_ids"])-direct)
        if strict:
            retrieval[row["query_id"]]=row
        for t in sorted(strict):
            frame.append({"query_id":row["query_id"],"target_id":t,"source_table_id":row["source_table_id"],"query_kind":row["query_kind"],"in_A0_C100":t in row["rankings"]["Equal"][:100],"sampling_hash":hashlib.sha256((row["query_id"]+"|"+t+"|R27-witness").encode()).hexdigest()})
    assert len(frame)==207
    selected=[]
    for admitted in (True,False):
        used=set()
        for row in sorted((r for r in frame if r["in_A0_C100"]==admitted),key=lambda r:r["sampling_hash"]):
            if row["source_table_id"] in used:
                row["exclusion_reason"]="duplicate_source_in_stratum"
            elif len(used)==16:
                row["exclusion_reason"]="beyond_fixed_hash_budget"
            else:
                used.add(row["source_table_id"])
                row["selected"]=True
                selected.append(row)
        assert len(used)==16
    panel=OUT/"witness_panel"
    lock={"status":"locked_before_generation","sampling_rule":"within each admission stratum, ascending sha256(query_id|target_id|R27-witness), at most one per source per stratum","cases":selected,"case_count":len(selected),"first_request_max":256,"physical_attempt_max":512,"conditions":["Real-crop","NoE-fill"],"original_retrieval":record(R26/"rankings/B13/rankings.jsonl.gz"),"A1_used_for_selection":False,"diagnostic_only_no_population_recall":True}
    write_json(panel/"B_PANEL_LOCK.json",lock)
    write_rows(panel/"sampling_frame.jsonl.gz",frame)
    frozen=[]
    for case in selected:
        row=retrieval[case["query_id"]]
        e=next(r for r in row["E_paths"] if r["target_id"]==case["target_id"])
        frozen.append({"query_id":case["query_id"],"target_id":case["target_id"],"retained_evidence":e,"D_ids":row["rankings"]["D100_ANN"],"A0_C100":row["rankings"]["Equal"][:100],"QT_scores":row["QT_OVER_U_scores"],"diagnostic_outside_deployment_queue":not case["in_A0_C100"],"own_retrieval_identity":row["candidate_pool_id"]})
    write_rows(panel/"frozen_case_retrieval.jsonl.gz",frozen)
    population={r["query_id"]:r for r in rows(R26/"common/dev_queries.jsonl")}
    legacy={}
    for generator in ("Qwen-Raw","B13"):
        src=R26/"stage2/pilot"/generator
        inputs=list(rows(R26/"stage2/inputs"/generator/"retrieval.jsonl"))
        results=list(rows(src/"results.jsonl"))
        assert len(inputs)==64 and len(results)==192
        byq={r["query_id"]:r for r in inputs}
        comparisons=[]
        branch=[]
        for row in results:
            source=byq[row["query_id"]]
            assert len(source["results"])==18 and source["input_pool_sha"]==row["input_pool_sha"]
            ranking=[c["target_id"] for c in source["results"]]
            truth=population[row["query_id"]]["positive_target_ids"]
            if row["status"]!="ran":
                assert row["ranking"]==[]
            elif set(row["ranking"])!=set(ranking):
                raise ValueError("C18 membership changed")
            before=query_metrics(ranking,truth,(1,3,5,7,9))
            after=query_metrics(row["ranking"],truth,(1,3,5,7,9))
            comparisons.append({"query_id":row["query_id"],"source_table_id":population[row["query_id"]]["source_table_id"],"query_kind":population[row["query_id"]]["query_kind"],"condition":row["condition"],"status":row["status"],"T0_ranking":ranking,"Stage2_ranking":row["ranking"],"T0_metrics":before,"Stage2_metrics":after,"deltas":{k:after[k]-v for k,v in before.items()},"input_pool_sha":row["input_pool_sha"]})
            for c in row["diagnostic_candidates"]:
                branch.append({"query_id":row["query_id"],"condition":row["condition"],"candidate":c,"opportunity":row["opportunities"].get(c["target_id"]),"original_T0_rank":ranking.index(c["target_id"])+1,"final_rank":row["ranking"].index(c["target_id"])+1 if c["target_id"] in row["ranking"] else None})
        dest=OUT/"stage2_legacy_replay"/generator
        write_rows(dest/"per_query_before_after.jsonl.gz",comparisons)
        write_rows(dest/"branch_opportunity_audit.jsonl.gz",branch)
        deleted=R26/"stage2/cell_audit"/generator/"all_confirmed_cell_deletion.jsonl.gz"
        deletion=list(rows(deleted))
        assert len(deletion)==64
        write_rows(dest/"value_deletion.jsonl.gz",deletion)
        cells=list(rows(R26/"stage2/cell_audit"/generator/"all_added_cells_with_independent_truth.jsonl"))
        write_rows(dest/"source_truth_cells.jsonl.gz",cells)
        means={condition:{kind:{method:{metric:sum(c[method][metric] for c in comparisons if c["condition"]==condition and (kind=="overall" or c["query_kind"]==kind))/sum(c["condition"]==condition and (kind=="overall" or c["query_kind"]==kind) for c in comparisons) for metric in comparisons[0][method]} for method in ("T0_metrics","Stage2_metrics")} for kind in ("overall","implicit","explicit")} for condition in ("Real-crop","Real-crop+original","NoE-fill")}
        legacy[generator]={"status":"completed","queries":64,"new_generation_requests":0,"candidate_budget":18,"metrics":means,"source_confirmed_new_cells":sum(c["correctness"]=="confirmed_exact" for c in cells),"source_results":record(src/"results.jsonl"),"source_deletion":record(deleted),"deletion_reuse":"Archived actual semantic recomputation; not new generation or newly computed semantic vectors"}
        write_json(dest/"EXECUTION.json",legacy[generator])
    assert abs(legacy["B13"]["metrics"]["Real-crop"]["overall"]["T0_metrics"]["recall@9"]-.390625)<1e-10
    assert abs(legacy["B13"]["metrics"]["Real-crop"]["overall"]["Stage2_metrics"]["recall@9"]-.3229166666666667)<1e-10
    return {"panel_cases":len(selected),"legacy":{k:{"queries":v["queries"],"source_confirmed_new_cells":v["source_confirmed_new_cells"]} for k,v in legacy.items()}}


if __name__=="__main__":
    print(json.dumps(prepare()))

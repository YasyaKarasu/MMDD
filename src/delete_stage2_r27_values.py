"""Delete independently correct new cells and recompute the fixed semantic backend."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json

import torch

from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import column_values, load_stage2_index
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.r26_generation import R26QwenBackend
from audit_stage2_r26_engineering import DATASET
from prepare_stage1_r27 import ROOT, OUT, R26, rows, record, read_json, write_json
from prepare_stage2_r27 import write_rows
from run_stage2_r27_panel import visible_table

PANEL=OUT/"witness_panel"


def rank(scores: dict, stage1: list[str]) -> list[str]:
    return sorted(stage1,key=lambda t:(-scores[t]["coverage"],-scores[t]["mean_similarity"],stage1.index(t)))


@torch.inference_mode()
def run() -> dict:
    torch.set_num_threads(2)
    delete=list(rows(PANEL/"deletable_cells.jsonl.gz"))
    bypair=defaultdict(set)
    for c in delete:
        bypair[c["query_id"],c["target_id"]].add(c["row_id"])
    if not bypair:
        receipt={"status":"not_triggered","reason":"no_independently_confirmed_new_cells","truth_audit":record(PANEL/"TRUTH_AUDIT.json")}
        write_json(PANEL/"B2_EXECUTION.json",receipt)
        return receipt
    runtime={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"inputs_sanitized.jsonl.gz")}
    real={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"Real.jsonl.gz")}
    frozen={(r["query_id"],r["target_id"]):r for r in rows(PANEL/"frozen_case_retrieval.jsonl.gz")}
    teacher={r["query_id"]:r["teacher_scores"] for r in rows(R26/"teacher/B13/rankings.jsonl.gz") if r["query_id"] in {q for q,t in bypair}}
    target_ids={t for key in bypair for t in frozen[key]["A0_C100"]}
    objects=load_stage2_index(DATASET,query_ids=set(),target_ids=target_ids,evidence_ids=set())
    targets={t:visible_table(table) for t,table in objects.targets.items()}
    model_dir=ROOT/"hf_models/Qwen3.5-9B"
    backend=R26QwenBackend(model_dir,device="cuda:1",dtype="bf16")
    scorer_path=ROOT/"work/stage1_optimization_r25_final_20260914/stage2/r25_b13_column_scorer.pt"
    scorer=load_candidate_scorer(scorer_path,torch.device("cpu"),expected_model_dir=model_dir).to(backend.device)
    verifier=Stage2Verifier(backend,scorer)
    output=[]
    for key,removed in bypair.items():
        source=real[key];case=runtime[key];q,t=key
        values=[c["value"] for c in source["cells"]]
        target_values=column_values(case["target"],source["selection"]["column_index"])
        before=verifier._semantic_check(values,target_values)
        error=max(abs(before.coverage-source["verification"]["coverage"]),abs(before.mean_similarity-source["verification"]["mean_similarity"]))
        assert error<=1e-6,(key,error)
        after_values=["" if c["row_id"] in removed else c["value"] for c in source["cells"]]
        # Calls embed_texts again on the changed list; no old cell vectors/scores reused.
        after=verifier._semantic_check(after_values,target_values)
        pool=frozen[key]["A0_C100"]
        diagnostic_pool=list(pool) if t in pool else [*pool,t]
        stage1=sorted(diagnostic_pool,key=lambda target:(-teacher[q][target],target))
        direct_ids=[target for target in pool if target in frozen[key]["D_ids"]]
        direct=verifier.verify_direct(case["query"],targets,direct_ids)
        zero={"coverage":0.,"mean_similarity":0.}
        before_scores={target:dict(zero) for target in stage1}
        direct_records={d.target_id:asdict(d) for d in direct}
        for d in direct:
            before_scores[d.target_id]=asdict(d.semantic_joinability)
        assert t not in direct_ids,"strict-EO target must not have a Direct100 branch"
        before_scores[t]=asdict(before)
        after_scores=deepcopy(before_scores);after_scores[t]=asdict(after)
        before_rank=rank(before_scores,stage1);after_rank=rank(after_scores,stage1)
        assert set(before_rank)==set(after_rank)==set(stage1)
        row={"query_id":q,"target_id":t,"deleted_row_ids":sorted(removed),"deleted_cell_ids":[c["cell_id"] for c in delete if (c["query_id"],c["target_id"])==key],"before_values":values,"after_values":after_values,"before_verification":asdict(before),"after_verification":asdict(after),"baseline_max_error":error,"direct_branch_for_strict_target":None,"before_final_branch":"evidence","after_final_branch":"evidence","before_rank":before_rank,"after_rank":after_rank,"target_rank_before":before_rank.index(t)+1,"target_rank_after":after_rank.index(t)+1,"fixed_competitor_direct_verification":direct_records,"before_scores":before_scores,"after_scores":after_scores,"diagnostic_outside_deployment_queue":case["diagnostic_outside_deployment_queue"],"rank_scope":"local-pair evidence intervention against frozen Direct verification in original A0 C100; outside-queue target appended only to this diagnostic comparator, never a deployed pool; no other candidate recovery is generated","coverage_upper_bound":source["coverage_upper_bound"],"competitors_with_direct_coverage_1":sum(d.semantic_joinability.coverage==1. for d in direct),"changed_semantic_score":before.coverage!=after.coverage or before.mean_similarity!=after.mean_similarity}
        output.append(row)
        write_rows(PANEL/"value_deletion.jsonl.gz",output)
        print(json.dumps({"query_id":q,"deleted_cells":len(removed),"coverage_before":before.coverage,"coverage_after":after.coverage,"rank_before":row["target_rank_before"],"rank_after":row["target_rank_after"]}),flush=True)
    receipt={"status":"completed","planned":True,"implemented":True,"executed":True,"evaluated":True,"pairs":len(output),"deleted_cells":len(delete),"score_changed_pairs":sum(r["changed_semantic_score"] for r in output),"rank_changed_pairs":sum(r["target_rank_before"]!=r["target_rank_after"] for r in output),"new_generation_requests":len(backend.generation_records),"backend_config":record(model_dir/"config.json"),"raw":record(PANEL/"value_deletion.jsonl.gz"),"rank_scope":"local diagnostic; not full Stage2 execution or population recall"}
    assert receipt["new_generation_requests"]==0
    write_json(PANEL/"B2_EXECUTION.json",receipt)
    return receipt


if __name__=="__main__":
    print(json.dumps(run()))

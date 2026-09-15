"""Export row support and score distributions for independently inspectable A arms."""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from prepare_stage1_r27 import ROOT, OUT, rows, read_json, write_json, record
from run_stage1_r11_task_e import _row_strengths, _sigmoid


def run() -> dict:
    torch.set_num_threads(2)
    store=FeatureStore.from_path(ROOT/"work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",cache_size=100000)
    results=[]
    for spec in read_json(ROOT/"mmdd_r26_review/R27_INPUT_LOCK.json")["models"]:
        name=spec["generator_id"];dest=OUT/"score_handoff"/name
        counts=Counter();size=[];sigmoids=[];coverage=[]
        with gzip.open(dest/"row_support.jsonl.gz","wt") as f:
            for row in rows(dest/"raw_candidates_paths.jsonl.gz"):
                evidence=sorted({p["evidence_id"] for t in row["E_paths"] for p in t["retained_paths"]})
                support=_row_strengths(row["query_id"],evidence,store)
                targets=[]
                for target in row["E_paths"]:
                    paths=target["retained_paths"]
                    pvalues=[_sigmoid(p["path_score"]) for p in paths]
                    current=[max(_sigmoid(p["path_score"])*s for p in paths for s in [support[p["evidence_id"]][ri]]) for ri in range(len(next(iter(support.values()))))]
                    score=sum(current)/len(current)
                    assert abs(score-target["evidence_score"])<=1e-8
                    size.append(len(paths));sigmoids.extend(pvalues);coverage.append(score)
                    for p in paths:counts[p["evidence_type"]]+=1
                    targets.append({"target_id":target["target_id"],"selected_ids":target["selected_evidence_ids"],"row_coverage":current,"D1_coverage_recomputed":score,"path_count":len(paths),"sigmoid_path_quantiles":np.quantile(pvalues,[0,.25,.5,.75,1]).tolist()})
                f.write(json.dumps({"query_id":row["query_id"],"row_support_by_evidence":support,"targets":targets})+"\n")
                counts["queries"]+=1
        diagnostic=read_json(dest/"diagnostics.json")
        diagnostic["retained_scalar_distributions"]={"counts":counts,"path_count_quantiles":np.quantile(size,[0,.5,.95,1]).tolist(),"sigmoid_path_quantiles":np.quantile(sigmoids,[0,.25,.5,.75,.95,1]).tolist(),"D1_coverage_quantiles":np.quantile(coverage,[0,.25,.5,.75,.95,1]).tolist(),"row_support_file":record(dest/"row_support.jsonl.gz"),"same_path_vs_D1_reconstruction":"all targets reproduced within 1e-8"}
        write_json(dest/"diagnostics.json",diagnostic)
        results.append(name);print(json.dumps({"row_support_complete":name}),flush=True)
    return {"models":len(results)}


if __name__=="__main__":
    print(json.dumps(run()))

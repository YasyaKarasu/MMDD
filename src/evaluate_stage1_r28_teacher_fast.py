"""Batch all targets of one query so QE and compression are shared across its full pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import evaluate_stage1_r28_teacher as evaluation
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.r28_objectives import coverage_scores
from mmdd_stage1.scoring import score_target_batch
from prepare_stage1_r27 import record, rows, write_json
from prepare_stage1_r28 import OUT, inputs, feature_store
from run_stage1_r19 import load_r19_checkpoint


@torch.inference_mode()
def score_example(model, example, store, device) -> dict:
    raw = score_target_batch(model,[example],store,device,PathAggregator("logsumexp",4,path_combination="sum"))
    cov = coverage_scores(raw,[example],store)
    direct = raw.direct.logits[0].cpu().tolist()
    lse = raw.evidence.logits[0].cpu().tolist()
    coverage = cov.evidence.logits[0].cpu().tolist()
    result = {"D":{},"E-LSE":{},"E-COV":{},"path_logits":{}}
    for i,c in enumerate(example.candidates):
        result["D"][c.target_id] = direct[i]
        if c.evidence_ids:
            result["E-LSE"][c.target_id] = lse[i]
            result["E-COV"][c.target_id] = coverage[i]
            result["path_logits"][c.target_id] = raw.path_logits[0][i].cpu().tolist()
    return result


def benchmark(device_name: str) -> dict:
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    model = load_r19_checkpoint(Path(inputs()["teacher_parent"]["path"]),device)[3].eval()
    store = feature_store(True)
    records = []
    for i,r in enumerate(rows(Path(inputs()["historical_b13_rankings"]["path"]))):
        if i in (0,1,2):
            records.append(r)
        if i == 2:
            break
    checks = []
    for ex in evaluation.frozen_examples(records):
        started = time.monotonic()
        old = evaluation.score_example(model,ex,store,device)
        chunked_seconds = time.monotonic()-started
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        new = score_example(model,ex,store,device)
        seconds = time.monotonic()-started
        differences = {}
        for view in ("D","E-LSE","E-COV"):
            assert old[view].keys() == new[view].keys()
            differences[view] = max(abs(old[view][t]-new[view][t]) for t in old[view])
            assert differences[view] <= 1e-5
            assert sorted(old[view],key=lambda t:(-old[view][t],t)) == sorted(new[view],key=lambda t:(-new[view][t],t))
        checks.append({"query_id":ex.query_id,"targets":len(ex.candidates),"max_score_delta":differences,
                       "chunked_seconds":chunked_seconds,"full_query_seconds":seconds,
                       "full_query_peak_allocated_bytes":torch.cuda.max_memory_allocated(device)})
    result = {"status":"pass","checks":checks,"reference":record(Path(evaluation.__file__)),"runner":record(Path(__file__)),
              "interpretation":"same pair scorer and bags; query-wide deduplication avoids repeated QE and compression; warm cache timing"}
    write_json(OUT / "TEACHER_EVALUATION_BATCH_AUDIT.json",result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--seed",type=int,default=13)
    parser.add_argument("--epoch",type=float,default=0)
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    if args.benchmark:
        result = benchmark(args.device)
    else:
        assert json.loads((OUT / "TEACHER_EVALUATION_BATCH_AUDIT.json").read_text())["status"] == "pass"
        evaluation.score_example = score_example
        result = evaluation.run(args.arm,args.seed,args.epoch,args.device)
        gid = "T0" if args.arm == "T0" else f"{args.arm}/seed{args.seed}/epoch{args.epoch:g}"
        write_json(OUT / "teacher/evaluation" / gid / "EXECUTION_WRAPPER.json",
                   {"runner":record(Path(__file__)),"batch_audit":record(OUT / "TEACHER_EVALUATION_BATCH_AUDIT.json")})
    print(json.dumps(result))

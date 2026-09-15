"""Frozen-B13 Teacher trajectories with separate QT, Evidence-LSE and Evidence-COV views."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.r28_objectives import coverage_scores
from mmdd_stage1.scoring import score_target_batch
from prepare_stage1_r27 import record, rows, sha, stable_sha, write_json
from prepare_stage1_r28 import ROOT, OUT, inputs, feature_store
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import write_rows


def frozen_examples(records: list[dict]) -> list[TargetExample]:
    result = []
    for r in records:
        by_target = {c["target_id"]: tuple(c["selected_evidence_ids"]) for c in r["E_paths"]}
        # GT labels are omitted from inference wiring. This function fixes only membership.
        result.append(TargetExample(r["query_id"], tuple(TargetCandidate(t, by_target.get(t, ()))
            for t in r["U"]), 0, 0))
    return result


def shuffle_bundles(examples: list[TargetExample], groups: dict[str,str], types: dict[str,str]) -> tuple[list, list]:
    """Sample donors by modality composition and other source group, without qrels."""
    pools = defaultdict(lambda:defaultdict(list))
    def key(c):
        return tuple(sorted(types[e] for e in c.evidence_ids))
    for ex in examples:
        for c in ex.candidates:
            if c.evidence_ids:
                pools[key(c)][groups[ex.query_id]].append((ex.query_id,c))
    rng = random.Random(280915)
    group_cache, result, receipts = {}, [], []
    for ex in examples:
        candidates = []
        for c in ex.candidates:
            if not c.evidence_ids:
                candidates.append(c)
                continue
            sig, group = key(c), groups[ex.query_id]
            group_cache.setdefault((sig,group), sorted(g for g in pools[sig] if g != group))
            eligible = group_cache[(sig,group)]
            if not eligible:
                raise ValueError("No cross-source donor with identical modality counts")
            donor_group = rng.choice(eligible)
            donor_q, donor = rng.choice(pools[sig][donor_group])
            candidates.append(replace(c,evidence_ids=donor.evidence_ids))
            receipts.append({"query_id":ex.query_id,"target_id":c.target_id,"source_group":group,
                "donor_query_id":donor_q,"donor_target_id":donor.target_id,"donor_source_group":donor_group,
                "modality_composition":sig,"evidence_ids":donor.evidence_ids})
        result.append(replace(ex,candidates=tuple(candidates)))
    return result, receipts


@torch.inference_mode()
def score_example(model, example: TargetExample, store, device: torch.device) -> dict:
    result = {"D":{}, "E-LSE":{}, "E-COV":{}, "path_logits":{}}
    for start in range(0,len(example.candidates),32):
        chunk = replace(example,candidates=example.candidates[start:start+32])
        raw = score_target_batch(model,[chunk],store,device,PathAggregator("logsumexp",4,path_combination="sum"))
        cov = coverage_scores(raw,[chunk],store)
        for i,c in enumerate(chunk.candidates):
            result["D"][c.target_id] = float(raw.direct.logits[0,i])
            if c.evidence_ids:
                result["E-LSE"][c.target_id] = float(raw.evidence.logits[0,i])
                result["E-COV"][c.target_id] = float(cov.evidence.logits[0,i])
                result["path_logits"][c.target_id] = raw.path_logits[0][i].cpu().tolist()
    return result


def metric_rows(meta: dict, scores: dict, model_id: str, condition: str) -> list[dict]:
    truth = set(meta["positive_target_ids"])
    strict = truth & (set(meta["E_target_ids"]) - (set(meta["rankings"]["D100_ANN"]) | set(meta["D100_EXACT"])))
    result = []
    for budget in (100,150,200,"Full-U"):
        pool = set(meta["U"] if budget == "Full-U" else meta["rankings"]["Equal"][:budget])
        for view in ("D","E-LSE","E-COV"):
            ranking = sorted(pool & scores[view].keys(), key=lambda t:(-scores[view][t],t))
            rec = {"model_id":model_id,"condition":condition,"query_id":meta["query_id"],
                "source_table_id":meta["source_table_id"],"query_kind":meta["query_kind"],
                "budget":budget,"view":view,"candidate_count":len(pool),"rankable_count":len(ranking),
                "raw_recall":len(truth & pool)/len(truth),"strict_EO_total":len(strict),
                "strict_EO_admitted":len(strict & pool),"ranking":ranking,
                "recall":{str(k):len(truth & set(ranking[:k]))/len(truth) for k in (10,20,50)},
                "strict_EO_hits":{str(k):len(strict & set(ranking[:k])) for k in (10,20,50)},
                "strict_EO_recall":{str(k):len(strict & set(ranking[:k]))/len(strict) if strict else None for k in (10,20,50)}}
            result.append(rec)
    return result


def run(arm: str, seed: int, epoch: float, device_name: str) -> dict:
    torch.set_num_threads(2)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    if arm == "T0":
        checkpoint = Path(inputs()["teacher_parent"]["path"])
        model_id = "T0"
    else:
        assert epoch in (.5,1,2,3,5)
        step = int(epoch*1424)
        checkpoint = OUT / "teacher" / arm / f"seed{seed}/checkpoints/step_{step:06d}.pt"
        assert checkpoint.with_suffix(".json").is_file(), "Checkpoint not yet fully written"
        model_id = f"{arm}/seed{seed}/epoch{epoch:g}"
    directory = OUT / "teacher/evaluation" / model_id
    if (directory / "EVALUATION_RECEIPT.json").exists():
        old = json.loads((directory / "EVALUATION_RECEIPT.json").read_text())
        assert old["checkpoint"]["sha256"] == sha(checkpoint)
        return {"model_id":model_id,"status":"verified_completed"}
    model = load_r19_checkpoint(checkpoint,device)[3].eval()
    store = feature_store(True)
    records = list(rows(Path(inputs()["historical_b13_rankings"]["path"])))
    examples = frozen_examples(records)
    groups = {r["query_id"]:r["source_table_id"] for r in records}
    shuffled = None
    if epoch == 5 or arm == "T0":
        evidence = {e for ex in examples for c in ex.candidates for e in c.evidence_ids}
        types = {e:store.embedding_features(e).object_type for e in evidence}
        shuffled, donors = shuffle_bundles(examples,groups,types)
        write_rows(OUT / "teacher/evidence_shuffle" / model_id / "donors.jsonl.gz",donors)
    directory.mkdir(parents=True,exist_ok=True)
    started, output, scalar_rows, max_d_delta = time.monotonic(), [], [], 0.
    for index,(meta,example) in enumerate(zip(records,examples)):
        real = score_example(model,example,store,device)
        output.extend(metric_rows(meta,real,model_id,"Real"))
        scalar_rows.append({"query_id":example.query_id,"condition":"Real",**real})
        if shuffled is not None:
            changed = score_example(model,shuffled[index],store,device)
            assert [c.target_id for c in example.candidates] == [c.target_id for c in shuffled[index].candidates]
            assert real["D"].keys() == changed["D"].keys()
            delta = max(abs(real["D"][t]-changed["D"][t]) for t in real["D"])
            max_d_delta = max(max_d_delta,delta)
            assert delta <= 1e-6, "Evidence shuffle changed QT logits"
            assert sorted(real["D"],key=lambda t:(-real["D"][t],t)) == sorted(changed["D"],key=lambda t:(-changed["D"][t],t))
            output.extend(metric_rows(meta,changed,model_id,"Shuffled"))
            scalar_rows.append({"query_id":example.query_id,"condition":"Shuffled",**changed})
        if index % 50 == 0:
            print(json.dumps({"model_id":model_id,"queries":index+1,"elapsed":time.monotonic()-started}),flush=True)
    write_rows(directory / "per_query.jsonl.gz",output)
    write_rows(directory / "scores.jsonl.gz",scalar_rows)
    result = {"model_id":model_id,"epoch":epoch,"arm":arm,"seed":seed,"status":"completed",
        "checkpoint":record(checkpoint),"candidate_membership":inputs()["historical_b13_rankings"],
        "evidence_bundle_policy":"frozen historical selected_evidence_ids; all paths within this retained bag",
        "views":["D","E-LSE","E-COV"],"primary":"D","shuffle":shuffled is not None,"shuffle_max_QT_delta":max_d_delta,
        "per_query":record(directory / "per_query.jsonl.gz"),"scores":record(directory / "scores.jsonl.gz"),
        "elapsed_seconds":time.monotonic()-started,"code":record(Path(__file__))}
    write_json(directory / "EVALUATION_RECEIPT.json",result)
    return {"model_id":model_id,"status":"completed","seconds":result["elapsed_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",required=True)
    parser.add_argument("--seed",type=int,default=13)
    parser.add_argument("--epoch",type=float,default=0)
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.arm,args.seed,args.epoch,args.device)))

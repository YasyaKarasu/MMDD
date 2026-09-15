"""Train-fit T0 hard32 mining on independently retrieved old/new universes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_teacher import TeacherPairCache
from prepare_stage1_r26 import ROOT,OUT,file_record,stable_sha
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import paths,read_rows,write_rows
from run_stage1_r25 import _json,_r25_teacher_feature_paths


@torch.inference_mode()
def run(generators: list[str],device_name: str,cache_path: Path | None = None,
        diagnostics_name: str = "HARD_MEMBERSHIP_DIAGNOSTICS.json") -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    ps = paths(ROOT)
    checkpoint = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
    _,_,_,teacher,_ = load_r19_checkpoint(checkpoint,device)
    teacher.eval()
    store = FeatureStore.from_path(ps["features"],cache_size=30000,cache_bytes=8*1024**3,teacher_paths=_r25_teacher_feature_paths(ROOT))
    directory = OUT / "feedback"
    directory.mkdir(parents=True,exist_ok=True)
    namespace = stable_sha({"teacher":file_record(checkpoint),"semantics":"R19GlobalTeacher raw QT","cache_code":file_record(ROOT / "src/mmdd_stage1/r26_teacher.py")})
    cache_path = cache_path or directory / "train_T0_pairs.sqlite"
    cache = TeacherPairCache(cache_path,namespace,teacher,store,device)
    known = defaultdict(set)
    for row in read_rows(ps["train"]):
        if row["source_type"] == row["destination_type"] == "table":
            known[row["query_id"]].update(row.get("positive_ids",[]))
            if row.get("positive_id"):
                known[row["query_id"]].add(row["positive_id"])
    frozen = {r["query_id"]:r for r in read_rows(OUT / "common/feedback_queries.jsonl")}
    all_hard = {}
    for generator in generators:
        source = OUT / "train_retrieval/feedback" / generator
        receipt = source / "RETRIEVAL_RECEIPT.json"
        if not receipt.exists():
            raise ValueError(f"Missing real train-fit own retrieval: {generator}")
        output = []
        started = time.monotonic()
        for row in read_rows(source / "rankings.jsonl.gz"):
            q = row["query_id"]
            positive = set(row["positive_target_ids"]) | known[q]
            targets = sorted(set(row["U"]) | positive)
            scores,cost = cache.score(q,targets)
            negative = sorted(set(row["U"])-positive,key=lambda t:(-scores[t],t))
            hard = negative[:32]
            d,e = set(row["rankings"]["D100_ANN"]),set(row["E_target_ids"])
            truth = set(frozen[q]["positive_target_ids"])
            output.append({"query_id":q,"generator_id":generator,"query_kind":row["query_kind"],"candidate_pool_id":row["candidate_pool_id"],
                "D":sorted(d),"E":sorted(e),"U":row["U"],"train_known_positive_closure":sorted(positive),
                "natural_hit_ids":sorted(truth & set(row["U"])),"training_injected_positive_ids":sorted(positive-set(row["U"])),
                "natural_raw_recall":len(truth & set(row["U"]))/len(truth),"EO_ANN_recall":len(truth & (e-d))/len(truth),
                "T0_scores":scores,"hard32":hard,"hard_shortfall":32-len(hard),
                "training_list":sorted(positive)+hard,"cost":cost})
            if len(output) % 500 == 0:
                print(json.dumps({"generator":generator,"mined":len(output),"queries":len(frozen)}),flush=True)
        if {r["query_id"] for r in output} != frozen.keys():
            raise ValueError("Feedback query population mismatch")
        destination = directory / generator
        destination.mkdir(parents=True,exist_ok=True)
        write_rows(destination / "hard_lists.jsonl.gz",output)
        _json(destination / "MINING_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid","queries":len(output),
            "natural_retrieval":file_record(receipt),"teacher":file_record(checkpoint),"namespace":namespace,"registry":file_record(ps["train"]),
            "lists":file_record(destination / "hard_lists.jsonl.gz"),"elapsed_seconds":time.monotonic()-started,
            "cache_path":str(cache_path),"code":file_record(Path(__file__)),
            "hard_shortfall_lists":sum(r["hard_shortfall"]>0 for r in output),"positive_mislabeled_as_hard":sum(len(set(r["hard32"]) & set(r["train_known_positive_closure"])) for r in output)})
        all_hard[generator] = {r["query_id"]:r for r in output}
    if "B13" not in all_hard:
        all_hard["B13"] = {r["query_id"]:r for r in read_rows(directory / "B13/hard_lists.jsonl.gz")}
    comparisons = []
    for generator,new in all_hard.items():
        if generator == "B13":
            continue
        pairs = []
        for q,current in new.items():
            old = all_hard["B13"][q]
            a,b = set(current["hard32"]),set(old["hard32"])
            pairs.append({"query_id":q,"query_kind":current["query_kind"],"hard_membership_changed":a!=b,
                "hard_order_changed":current["hard32"] != old["hard32"],"jaccard":len(a&b)/len(a|b) if a|b else 1.,
                "new_unique_negative_fraction":len(a-b)/len(a) if a else 0.,"new_unique_negatives":sorted(a-b),
                "universe_changed":set(current["U"]) != set(old["U"]),"old_EO":old["EO_ANN_recall"],"new_EO":current["EO_ANN_recall"]})
        write_rows(directory / generator / "H_old_H_new_comparison.jsonl.gz",pairs)
        comparisons.append({"generator":generator,"queries":len(pairs),"hard_membership_diff_lists":sum(r["hard_membership_changed"] for r in pairs),
            "hard_order_diff_lists":sum(r["hard_order_changed"] for r in pairs),"universe_diff_lists":sum(r["universe_changed"] for r in pairs),
            "mean_jaccard":sum(r["jaccard"] for r in pairs)/len(pairs),"new_unique_negative_fraction":sum(r["new_unique_negative_fraction"] for r in pairs)/len(pairs),
            "implicit_eo_admission":{name:sum(r[name+"_EO"] for r in pairs if r["query_kind"] == "implicit")/sum(r["query_kind"] == "implicit" for r in pairs) for name in ("old","new")}})
    _json(directory / diagnostics_name,comparisons)
    cache.db.close()
    return {"comparisons":comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",action="append",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--cache-path",type=Path)
    parser.add_argument("--diagnostics-name",default="HARD_MEMBERSHIP_DIAGNOSTICS.json")
    args = parser.parse_args()
    print(json.dumps(run(args.generator,args.device,args.cache_path,args.diagnostics_name)))

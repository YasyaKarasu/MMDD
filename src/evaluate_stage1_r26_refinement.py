"""Compare Tcont/Told/Tnew on frozen old/new pools and measure the redistill gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import population_metrics, query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from mmdd_stage1.r26_teacher import rerank_pools
from mmdd_stage1.teacher_rerank import _teacher_scores
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r21 import paths, read_rows, write_rows
from run_stage1_r25 import _json, _r25_teacher_feature_paths, sha256

KS = (10,20,30,40,50)
ARMS = ("Tcont","Told","Tnew")


@torch.inference_mode()
def run(device_name: str) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    protocol_path = OUT / "feedback/REFINEMENT_EVALUATION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    gate = json.loads((OUT / "feedback/REFINEMENT_GATE_FROZEN.json").read_text())
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    own = {}
    for spec in protocol["pools"]:
        path = Path(spec["rankings"]["path"])
        if sha256(path) != spec["rankings"]["sha256"]:
            raise ValueError("Frozen refinement evaluation pool changed")
        rows = {r["query_id"]:r for r in read_rows(path)}
        if rows.keys() != population.keys() or any(rows[q]["positive_target_ids"] != m["positive_target_ids"] for q,m in population.items()):
            raise ValueError("Refinement evaluation population/qrels mismatch")
        own[spec["generator"]] = rows
    store = FeatureStore.from_path(paths(ROOT)["features"], cache_size=30000, cache_bytes=4*1024**3,
                                   teacher_paths=_r25_teacher_feature_paths(ROOT))
    outputs, summaries, receipts = {}, {}, []
    for seed,name in zip((13,29),gate["selected_generators"]):
        for arm in ARMS:
            job = OUT / "feedback/refinement" / arm / f"seed{seed}"
            training_path = job / "TRAINING_RECEIPT.json"
            training = json.loads(training_path.read_text())
            checkpoint = Path(training["final_checkpoint"]["path"])
            if sha256(checkpoint) != training["final_checkpoint"]["sha256"] or training["optimizer_updates"] != 10536:
                raise ValueError("Refinement endpoint differs from the fixed full budget")
            destination = job / "evaluation"
            destination.mkdir(exist_ok=True)
            signature = {"training":file_record(training_path), "checkpoint":file_record(checkpoint),
                         "protocol":file_record(protocol_path), "code":file_record(Path(__file__)), "device":device_name}
            receipt_path = destination / "EVALUATION_RECEIPT.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["signature"] != signature or sha256(destination / "rankings.jsonl.gz") != receipt["rankings"]["sha256"]:
                    raise ValueError("Refinement evaluation resume identity differs")
                rows = list(read_rows(destination / "rankings.jsonl.gz"))
            else:
                _,_,_,teacher,_ = load_r19_checkpoint(checkpoint, device)
                teacher.eval()
                compression = teacher.new_compression_cache()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                rows, pairs = [], 0
                started = time.monotonic()
                for index,(q,meta) in enumerate(population.items(),1):
                    sources = {"B13":own["B13"][q], "selected_new":own[name][q]}
                    targets = sorted({t for row in sources.values() for key in ("U","M_exact") for t in row[key]})
                    values = _teacher_scores(teacher,q,targets,store,device,batch_size=64,compression_cache=compression)
                    scores = dict(zip(targets,values))
                    pairs += len(scores)
                    for pool,row in sources.items():
                        rankings = {key.replace("T0","TEACHER"):value for key,value in rerank_pools(row,scores).items()}
                        rows.append({**meta,"pool":pool,"generator_id":"B13" if pool == "B13" else name,
                                     "arm":arm,"seed":seed,"candidate_pool_id":row["candidate_pool_id"],
                                     "rankings":rankings,"scores":{t:scores[t] for t in sorted(set(row["U"])|set(row["M_exact"]))}})
                    if index % 100 == 0:
                        print(json.dumps({"arm":arm,"seed":seed,"queries":index,"actual_pairs":pairs}),flush=True)
                write_rows(destination / "rankings.jsonl.gz",rows)
                receipt = {"execution_status":"ran","scientific_validity":"valid","signature":signature,
                           "rankings":file_record(destination / "rankings.jsonl.gz"), "actual_pairs":pairs,
                           "elapsed_seconds":time.monotonic()-started,
                           "peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                           "scope":"Actual QT scores on frozen own U/M; BT100 and D100 are subsets. Offline full-pool cost, not online100 latency."}
                _json(receipt_path,receipt)
                del compression,teacher
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            receipts.append(file_record(receipt_path))
            for pool in ("B13","selected_new"):
                subset = {r["query_id"]:r for r in rows if r["pool"] == pool}
                outputs[arm,seed,pool] = subset
                summary = {}
                for kind in ("overall","implicit","explicit"):
                    truth = {q:r["positive_target_ids"] for q,r in population.items() if kind == "overall" or r["query_kind"] == kind}
                    summary[kind] = {method:population_metrics({q:r["rankings"][method] for q,r in subset.items()},truth,KS)
                                     for method in next(iter(subset.values()))["rankings"]}
                summaries[f"{arm}/seed{seed}/{pool}"] = summary
    comparisons = []
    for pool in ("B13","selected_new"):
        for new,old in (("Told","Tcont"),("Tnew","Told")):
            for kind in ("overall","implicit","explicit"):
                qs = [q for q,r in population.items() if kind == "overall" or r["query_kind"] == kind]
                for method in ("BT100_TEACHER","D100_TEACHER","U_OFFLINE_TEACHER","M_OFFLINE_TEACHER"):
                    for k in KS:
                        deltas = []
                        for seed in (13,29):
                            values = []
                            for q in qs:
                                truth = population[q]["positive_target_ids"]
                                values.append(query_metrics(outputs[new,seed,pool][q]["rankings"][method],truth,(k,))[f"recall@{k}"]-
                                              query_metrics(outputs[old,seed,pool][q]["rankings"][method],truth,(k,))[f"recall@{k}"])
                            deltas.append(values)
                        comparisons.append({"pool":pool,"new":new,"old":old,"kind":kind,"method":method,"K":k,
                                            "seed_deltas":{str(s):float(np.mean(v)) for s,v in zip((13,29),deltas)},
                                            **source_cluster_comparison(np.mean(deltas,axis=0),[population[q]["source_table_id"] for q in qs])})
    gates = []
    for seed,name in zip((13,29),gate["selected_generators"]):
        delta = {kind:summaries[f"Tnew/seed{seed}/selected_new"][kind]["BT100_TEACHER"]["recall@10"]-
                      summaries[f"Told/seed{seed}/selected_new"][kind]["BT100_TEACHER"]["recall@10"] for kind in ("overall","implicit")}
        hard_path = OUT / "feedback" / name / "H_old_H_new_comparison.jsonl.gz"
        novel = sum(bool(r["new_unique_negatives"]) for r in read_rows(hard_path))
        gates.append({"seed":seed,"generator":name,"Tnew_minus_Told_R10":delta,"lists_with_new_competitors":novel,
                      "pass":delta["overall"]>0 and delta["implicit"]>=-1e-12 and novel>0,"hard_comparison":file_record(hard_path)})
    result = {"execution_status":"ran","scientific_validity":"valid","K":KS,"metrics":summaries,
              "comparisons":comparisons,"inputs":receipts,"redistill_gate":{"status":"triggered" if all(r["pass"] for r in gates) else "not_triggered","seeds":gates}}
    directory = OUT / "feedback"
    _json(directory / "REFINEMENT_EVALUATION.json",result)
    _json(directory / "REDISTILL_GATE_FROZEN.json",{**result["redistill_gate"],"evaluation":file_record(directory / "REFINEMENT_EVALUATION.json"),
                                                  "protocol":file_record(protocol_path)})
    lines = ["# R26 Teacher refinement", "", "All values are Stage1 query-macro Recall (%), full10536-update endpoints. Each seed uses the same selected-new Equal C100 pool for all three Teachers.", "",
             "| Teacher | Seed | R10 | R20 | R30 | R40 | R50 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed in (13,29):
            m = summaries[f"{arm}/seed{seed}/selected_new"]["overall"]["BT100_TEACHER"]
            lines.append(f"| {arm} | {seed} | "+" | ".join(f"{100*m[f'recall@{k}']:.3f}" for k in KS)+" |")
    lines.extend(["", "Redistillation budget gate: **"+result["redistill_gate"]["status"]+"**. Full B13/new U/M and D100 controls, three query groups and paired source-bootstrap results are in `REFINEMENT_EVALUATION.json`.",
                  "", "This is exploratory dev analysis. QT Teacher scores do not establish path grounding; Tnew−Told isolates the generator after the common label-closure repair and fixed continuation recipe."])
    (directory / "REFINEMENT_RESULTS.md").write_text("\n".join(lines)+"\n")
    return {"teachers":len(receipts),"comparisons":len(comparisons),"redistill_gate":result["redistill_gate"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:1")
    print(json.dumps(run(parser.parse_args().device)))

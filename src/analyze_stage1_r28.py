"""Summarize R28 preregistered endpoints, seed directions, and source-paired trajectories."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r27 import record, rows, write_json
from prepare_stage1_r28 import ROOT, OUT, FAMILIES, inputs
from evaluate_stage1_r28_student import OWN
from run_stage1_r21 import write_rows


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]) if records else ["status"])
        writer.writeheader()
        writer.writerows(records)


def paired_queries(left: dict, right: dict, population: dict, seeds: tuple[int,...],
                   kind: str) -> tuple[np.ndarray,list[str]]:
    """Complete paired seed means within each query, followed by source clustering."""
    qids = sorted(q for q,m in population.items() if kind == "overall" or m["query_kind"] == kind)
    if any((s,q) not in side and (0,q) not in side for s in seeds for q in qids for side in (left,right)):
        raise ValueError("Missing query/seed in a registered paired comparison")
    def value(side, seed, query):
        return side[seed,query] if (seed,query) in side else side[0,query]
    deltas = np.array([np.mean([value(left,s,q)-value(right,s,q) for s in seeds]) for q in qids])
    return deltas,[population[q]["source_table_id"] for q in qids]


def collect() -> tuple[list[dict],list[dict]]:
    output, missing = [], []
    historical = {r["query_id"]: r for r in rows(Path(inputs()["historical_b13_rankings"]["path"]))}
    # Teacher epoch0 is the one frozen T0 control; all six epoch0 checkpoint receipts still exist.
    specs = [("T0",0,0,"T0")]
    specs += [(a,s,e,f"{a}/seed{s}/epoch{e:g}") for a in FAMILIES if a.startswith("T-")
              for s in (13,29) for e in (.5,1,2,3,5)]
    for arm,seed,epoch,gid in specs:
        path = OUT / "teacher/evaluation" / gid / "EVALUATION_RECEIPT.json"
        if not path.exists():
            missing.append({"kind":"teacher","model_id":gid,"missing":str(path)})
            continue
        for r in rows(path.parent / "per_query.jsonl.gz"):
            meta = historical[r["query_id"]]
            truth = set(meta["positive_target_ids"])
            strict = truth & (set(meta["E_target_ids"]) - (set(meta["rankings"]["D100_ANN"]) | set(meta["D100_EXACT"])))
            metrics = {"RawRecall":r["raw_recall"],"candidate_count":r["candidate_count"],
                       "rankable_count":r["rankable_count"], "rankable_RawRecall":len(truth & set(r["ranking"])) / len(truth),
                       "EO_STRICT_total":r["strict_EO_total"],"EO_STRICT_admitted":r["strict_EO_admitted"]}
            metrics.update({f"R@{k}":r["recall"][str(k)] for k in (10,20,50)})
            metrics.update({f"EO_STRICT_hits@{k}":r["strict_EO_hits"][str(k)] for k in (10,20,50)})
            for k in (10, 20, 50):
                top = set(r["ranking"][:k])
                metrics[f"strict_contribution@{k}"] = len(strict & top) / len(truth)
                metrics[f"non_strict_contribution@{k}"] = len((truth - strict) & top) / len(truth)
            output.append({"section":"teacher","arm":arm,"seed":seed,"epoch":epoch,
                           "budget":str(r["budget"]),"view":r["view"],"condition":r["condition"],
                           **{k:r[k] for k in ("query_id","source_table_id","query_kind")},"metrics":metrics})
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        gid = spec["generator_id"]
        path = OWN / "rankings" / gid / "R28_EVALUATION_RECEIPT.json"
        if not path.exists():
            missing.append({"kind":"student","model_id":gid,"missing":str(path)})
            continue
        funnel = {r["query_id"]:r for r in rows(path.parent / "eo_strict_funnel.jsonl.gz")}
        teacher = {r["query_id"]:r for r in rows(OWN / "teacher" / gid / "rankings.jsonl.gz")}
        for r in rows(path.parent / "rankings.jsonl.gz"):
            truth,q = set(r["positive_target_ids"]),r["query_id"]
            rankings = {**r["rankings"],**teacher[q]["rankings"]}
            for view,rank in rankings.items():
                fixed,own = set(funnel[q]["fixed_EO_STRICT"]),set(funnel[q]["own_EO_STRICT"])
                metrics = {"RawRecall":len(truth & set(rank))/len(truth),"candidate_count":len(rank),
                           "fixed_EO_total":len(fixed),"own_EO_total":len(own),
                           "fixed_EO_admitted":len(fixed & set(rank)),"own_EO_admitted":len(own & set(rank))}
                for k in (10,20,50):
                    top = set(rank[:k])
                    metrics.update({f"R@{k}":len(truth & top)/len(truth),
                                    f"fixed_EO_hits@{k}":len(fixed & top),f"own_EO_hits@{k}":len(own & top)})
                output.append({"section":"student","arm":spec["arm"],"seed":spec["seed"],"epoch":spec["epoch"],
                    "budget":"own","view":view,"condition":"Real",
                    **{k:r[k] for k in ("query_id","source_table_id","query_kind")},"metrics":metrics})
    # Historical references only: no new baseline training or own index reconstruction.
    for name in ("Qwen-Raw","B13"):
        base = ROOT / "work/stage1_optimization_r26_20260914"
        path = base / "rankings" / name / "rankings.jsonl.gz"
        tpath = base / "teacher" / name / "rankings.jsonl.gz"
        if not path.exists() or not tpath.exists():
            missing.append({"kind":"baseline","model_id":name,"missing":"historical rankings or T0 rankings"})
            continue
        teacher = {r["query_id"]:r for r in rows(tpath)}
        for r in rows(path):
            truth = set(r["positive_target_ids"])
            rankings = {k:v for k,v in r["rankings"].items() if k in ("D100_ANN","D100_EXACT","U","E_ONLY","M_EXACT","Equal")}
            rankings.update(teacher[r["query_id"]]["rankings"])
            for view,rank in rankings.items():
                metrics = {"RawRecall":len(truth & set(rank))/len(truth), "candidate_count":len(rank),
                           **{f"R@{k}":len(truth & set(rank[:k]))/len(truth) for k in (10,20,50)}}
                output.append({"section":"historical_reference","arm":name,"seed":0,"epoch":0,
                               "budget":"historical_own","view":view,"condition":"Real",
                               **{k:r[k] for k in ("query_id","source_table_id","query_kind")},"metrics":metrics})
    return output,missing


def summarize(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    dimensions = ("section","arm","seed","epoch","budget","view","condition")
    family = defaultdict(dict)
    for r in records:
        if r["seed"] in (13,29):
            key = tuple(r[k] for k in ("section","arm","epoch","budget","view","condition","query_id"))
            family[key][r["seed"]] = r
    averaged = []
    for pair in family.values():
        if set(pair) == {13,29}:
            a,b = pair[13],pair[29]
            averaged.append({**a,"seed":"13+29","metrics":{k:(a["metrics"][k]+b["metrics"][k])/2 for k in a["metrics"]}})
    for r in [*records,*averaged]:
        key = tuple(r[k] for k in dimensions)
        grouped[key,"overall"].append(r)
        grouped[key,r["query_kind"]].append(r)
    result = []
    for (key,kind),rs in grouped.items():
        for metric in rs[0]["metrics"]:
            result.append({**dict(zip(dimensions,key)),"kind":kind,"metric":metric,
                           "value":float(np.mean([r["metrics"][metric] for r in rs])),"queries":len(rs),
                           "total":sum(r["metrics"][metric] for r in rs) if "EO" in metric else None})
    return result


def comparisons(records: list[dict], population: dict) -> list[dict]:
    values = defaultdict(dict)
    for r in records:
        if r["section"] == "teacher" and r["budget"] != "Full-U":
            continue
        if r["section"] == "student" and r["view"] not in ("U","U_OFFLINE_T0","E_ONLY","D100_ANN","D100_EXACT"):
            continue
        for metric,value in r["metrics"].items():
            if metric in ("candidate_count", "rankable_count", "rankable_RawRecall") or metric.endswith("total"):
                continue
            values[r["section"],r["arm"],r["epoch"],r["view"],r["condition"],metric][r["seed"],r["query_id"]] = value
    result = []
    def add(label, left_key, right_key):
        left,right = values.get(left_key),values.get(right_key)
        if not left or not right:
            return
        for seeds in ((13,),(29,),(13,29)):
            if not all(any(s in (0,seed) for s,q in left) and any(s in (0,seed) for s,q in right) for seed in seeds):
                continue
            for kind in ("overall","implicit","explicit"):
                delta,sources = paired_queries(left,right,population,seeds,kind)
                result.append({"comparison":label,"left":list(left_key),"right":list(right_key),
                               "seeds":list(seeds),"kind":kind,
                               **source_cluster_comparison(delta,sources,replicates=10000,seed=280915)})
    for section,edge,lse,cov in (("teacher","T-EDGE-CONT","T-PATH-SPLIT-LSE","T-PATH-SPLIT-COV"),
                                ("student","S-EDGE-LONG","S-PATH-LONG","S-COV-LONG")):
        endpoints = [k for k in values if k[0] == section and k[1] == lse and k[2] == 5 and k[4] == "Real"]
        for key in endpoints:
            _,_,_,view,condition,metric = key
            for name,a,b in (("Path vs Edge",lse,edge),("COV vs LSE",cov,lse),("COV vs Edge",cov,edge)):
                add(name,(section,a,5,view,condition,metric),(section,b,5,view,condition,metric))
        for arm in (edge,lse,cov):
            bases = [k for k in values if k[0] == section and k[1] == arm and k[2] == 1 and k[4] == "Real"]
            for key in bases:
                for epoch in (2,3,5):
                    add(f"Epoch{epoch} vs Epoch1",(*key[:2],epoch,*key[3:]),key)
            if section == "teacher":
                for key in [k for k in values if k[0] == section and k[1] == arm and k[2] == 5 and k[4] == "Real"]:
                    add("Real vs Shuffled",key,(*key[:4],"Shuffled",key[-1]))
                    add("Epoch5 vs fixed T0",key,(section,"T0",0,*key[3:]))
    # T0 has one fixed checkpoint, not two fabricated seeds.
    for key in [k for k in values if k[1] == "T0" and k[4] == "Real"]:
        other = (*key[:4],"Shuffled",key[-1])
        if other not in values:
            continue
        for kind in ("overall","implicit","explicit"):
            delta,sources = paired_queries(values[key],values[other],population,(0,),kind)
            result.append({"comparison":"T0 Real vs Shuffled","left":list(key),"right":list(other),"seeds":[0],"kind":kind,
                           **source_cluster_comparison(delta,sources,replicates=10000,seed=280915)})
    return result


def retention_summary(summary: list[dict]) -> list[dict]:
    """Conditional pair retention, with explicit denominators separate from query Recall."""
    dimensions = ("section", "arm", "seed", "epoch", "budget", "view", "condition", "kind")
    grouped = defaultdict(dict)
    for r in summary:
        if "EO" in r["metric"]:
            grouped[tuple(r[k] for k in dimensions)][r["metric"]] = r
    output = []
    for key, metrics in grouped.items():
        for prefix in ("EO_STRICT", "fixed_EO", "own_EO"):
            total = metrics.get(prefix + "_total")
            if total is None:
                continue
            admitted = metrics.get(prefix + "_admitted")
            for k in (10, 20, 50):
                hits = metrics[prefix + f"_hits@{k}"]["total"]
                denominator = total["total"]
                admission = admitted["total"] if admitted else None
                output.append({**dict(zip(dimensions, key)), "EO_definition": prefix, "K": k,
                    "EO_pairs": denominator, "admitted_pairs": admission, "topK_hits": hits,
                    "admission_fraction": admission / denominator if denominator and admission is not None else None,
                    "retention_over_all_EO": hits / denominator if denominator else None,
                    "retention_over_admitted_EO": hits / admission if admission else None,
                    "queries": total["queries"],
                    "denominator_note": "pair-pooled conditional retention; family counts average seeds"})
    return output


def cost_summary() -> list[dict]:
    """Wall time under concurrent execution; cache reuse is reported explicitly."""
    output = []
    def append(phase, arm, seed, epoch, seconds, receipt, *, updates=None, pairs=None,
               new_pairs=None, cached_pairs=None, peak=None):
        output.append({"phase": phase, "arm": arm, "seed": seed, "epoch": epoch,
            "elapsed_seconds": seconds, "updates": updates, "requested_pairs": pairs,
            "new_pairs": new_pairs, "cached_pairs": cached_pairs, "peak_allocated_bytes": peak,
            "receipt": str(receipt), "scope": "wall time with concurrent jobs; not isolated online latency"})
    for arm in FAMILIES:
        kind = "student" if arm.startswith("S-") else "teacher"
        for seed in (13, 29):
            path = OUT / kind / arm / f"seed{seed}/EXECUTION.json"
            r = json.loads(path.read_text())
            if r["status"] == "completed":
                append(kind + "_training", arm, seed, 5, r["elapsed_seconds"], path,
                       updates=r["updates"], peak=r["peak_allocated_bytes"])
    for path in sorted((OUT / "teacher/evaluation").glob("**/EVALUATION_RECEIPT.json")):
        r = json.loads(path.read_text())
        append("teacher_evaluation", r["arm"], r["seed"], r["epoch"], r["elapsed_seconds"], path)
    for spec in json.loads((OWN / "MODEL_INVENTORY.json").read_text()):
        gid = spec["generator_id"]
        for phase, branch, filename in (("own_retrieval", "rankings", "RETRIEVAL_RECEIPT.json"),
                                         ("own_frozen_T0", "teacher", "TEACHER_RECEIPT.json")):
            path = OWN / branch / gid / filename
            if not path.exists():
                continue
            c = json.loads(path.read_text())["cost"]
            append(phase, spec["arm"], spec["seed"], spec["epoch"],
                c["total_seconds"] if phase == "own_retrieval" else c["elapsed_seconds"], path,
                pairs=c.get("requested_pairs"), new_pairs=c.get("new_pairs"),
                cached_pairs=c.get("cached_pairs"), peak=c.get("peak_allocated_bytes"))
    return output


def run(bootstrap: bool) -> dict:
    records,missing = collect()
    summary = summarize(records)
    write_rows(OUT / "statistics/per_query.jsonl.gz",records)
    write_csv(OUT / "statistics/main_table.csv",summary)
    write_csv(OUT / "statistics/eo_admission_retention.csv", retention_summary(summary))
    write_csv(OUT / "statistics/costs.csv", cost_summary())
    for section in ("teacher","student"):
        write_csv(OUT / section / "trajectory_metrics.csv",[r for r in summary if r["section"] == section])
    population = {r["query_id"]:r for r in rows(ROOT / "work/stage1_optimization_r26_20260914/common/dev_queries.jsonl")}
    stats = comparisons(records,population) if bootstrap else []
    if bootstrap:
        write_rows(OUT / "statistics/source_group_bootstrap.jsonl",stats)
        write_rows(OUT / "teacher/pairwise_bootstrap.jsonl",[r for r in stats if r["left"][0] == "teacher"])
        write_rows(OUT / "statistics/win_loss_tie.jsonl",[{k:r[k] for k in ("comparison","left","right","seeds","kind","wins","losses","ties")} for r in stats])
    write_rows(OUT / "student/eo_strict_funnel.jsonl.gz",(r for p in sorted((OWN / "rankings").glob("*/*/*/eo_strict_funnel.jsonl.gz")) for r in rows(p)))
    result = {"status":"partial" if missing else "all_evaluations_present_pending_completion_audit",
              "missing":missing,"per_query_rows":len(records),"bootstrap_comparisons":len(stats)}
    write_json(OUT / "ANALYSIS_STATUS.json",result)
    return {"status":result["status"],"missing_nodes":len(missing),"rows":len(records),"comparisons":len(stats)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",action="store_true")
    print(json.dumps(run(parser.parse_args().bootstrap)))

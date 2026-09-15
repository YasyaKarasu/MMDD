"""Run full-U Equal/conf/column on actual R26 generator paths and frozen columns."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from mmdd_stage1.r26_column import ColumnSimilarity, QueryBalancedCDF
from mmdd_stage1.r26_metrics import fuse_channels,population_metrics
from mmdd_stage1.teacher_rerank import _average_ranks
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json,sha256


def distribution(values: list[float]) -> dict:
    return {"n":len(values),"mean":float(np.mean(values)),"p10":float(np.quantile(values,.1)),"p50":float(np.quantile(values,.5)),
            "p90":float(np.quantile(values,.9)),"fraction_lt_0.1":float(np.mean(np.array(values)<.1)),
            "fraction_gt_0.9":float(np.mean(np.array(values)>.9))} if values else {"n":0}


def auc(values: list[float],labels: list[bool]) -> float | None:
    n = sum(labels)
    if n == 0 or n == len(labels):
        return None
    ranks = _average_ranks(values)
    return (sum(r for r,l in zip(ranks,labels) if l)-n*(n+1)/2)/(n*(len(labels)-n))


def run(device_name: str, generators: list[str] | None = None) -> dict:
    torch.set_num_threads(4)
    root = OUT / "fusion"
    columns_path = root / "columns/columns.pt"
    receipt = root / "columns/COLUMN_RECEIPT.json"
    if not receipt.exists():
        raise ValueError("Column generation and numerical reuse audit must finish first")
    vectors = torch.load(columns_path,map_location="cpu",weights_only=False)["vectors"]
    scorer = ColumnSimilarity(vectors,torch.device(device_name))
    train_path = OUT / "train_retrieval/column256/Qwen-Raw/rankings.jsonl.gz"
    reference = []
    cdf_receipt = root / "COLUMN_CDF_RECEIPT.json"
    if cdf_receipt.exists():
        previous = json.loads(cdf_receipt.read_text())
        for key,path in (("train_own_retrieval",train_path),("columns",columns_path),
                         ("column_audit",receipt),("reference",root / "column_reference_C.jsonl.gz")):
            if previous[key]["sha256"] != sha256(path):
                raise ValueError(f"Frozen column CDF input changed: {key}")
        reference = list(read_rows(root / "column_reference_C.jsonl.gz"))
    else:
        for row in read_rows(train_path):
            values = scorer.scores(row["query_id"],row["U"])
            if any(v is None for v in values.values()):
                raise ValueError("Reference CDF has missing columns; finish natural-U column coverage")
            reference.append({"query_id":row["query_id"],"candidate_pool_id":row["candidate_pool_id"],"C":values})
    if len(reference) != 256:
        raise ValueError("Frozen reference must preserve all256 selected train queries")
    cdf = QueryBalancedCDF([list(r["C"].values()) for r in reference])
    if not cdf_receipt.exists():
        write_rows(root / "column_reference_C.jsonl.gz",reference)
        _json(cdf_receipt,{"train_own_retrieval":file_record(train_path),"columns":file_record(columns_path),
            "column_audit":file_record(receipt),"reference":file_record(root / "column_reference_C.jsonl.gz"),
            "queries":256,"weight":"1/256 per query; uniform over its natural U; no positive injection or dev fitting"})
    raw_exact = {r["query_id"]:set(r["D100_EXACT"]) for r in read_rows(OUT / "rankings/Qwen-Raw/rankings.jsonl.gz")}
    completed = []
    for entry in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = entry["generator_id"]
        if generators is not None and name not in generators:
            continue
        source = OUT / "rankings" / name
        retrieval_receipt = source / "RETRIEVAL_RECEIPT.json"
        if not retrieval_receipt.exists():
            continue
        output,seconds = [],[]
        for row in read_rows(source / "rankings.jsonl.gz"):
            started = time.monotonic()
            similarities = scorer.scores(row["query_id"],row["U"])
            alphas = {t:cdf.alpha(value) for t,value in similarities.items()}
            fusion = fuse_channels(row["D100_ANN"],row["E_paths"],alphas)
            if set(fusion["rankings"]["Column"]) != set(row["U"]):
                raise ValueError("Column promotion dropped union candidates")
            seconds.append(time.monotonic()-started)
            direct = row["rankings"]["D100_ANN"]
            c_q = [similarities[t] for t in direct[:10] if similarities[t] is not None]
            output.append({k:row[k] for k in ("query_id","query_kind","source_table_id","positive_target_ids","candidate_pool_id","EO_ANN","EO_EXACT","U_ONLY_VS_M")})
            output[-1].update({"generator_id":name,"rankings":{**fusion["rankings"],"Direct-only":direct},"scores":fusion["scores"],
                "confidence_alpha":fusion["confidence_alpha"],"column_alpha":alphas,"C":similarities,"C_q":max(c_q) if c_q else None,
                "RAW_EXACT_D100_OUTSIDE":sorted(set(row["positive_target_ids"]) & (set(row["U"])-raw_exact[row["query_id"]]))})
        destination = root / name
        destination.mkdir(parents=True,exist_ok=True)
        write_rows(destination / "rankings.jsonl.gz",output)
        summary = {}
        for kind in ("overall","implicit","explicit"):
            rows = [r for r in output if kind == "overall" or r["query_kind"] == kind]
            qrels = {r["query_id"]:r["positive_target_ids"] for r in rows}
            scores = {method:population_metrics({r["query_id"]:r["rankings"][method] for r in rows},qrels,(10,20,50))
                      for method in output[0]["rankings"]}
            summary[kind] = {"metrics":scores,"queries":len(rows),"confidence_query_alpha":distribution([r["confidence_alpha"] for r in rows]),
                "column_pair_alpha":distribution([a for r in rows for a in r["column_alpha"].values()]),
                "column_query_mean_alpha":distribution([float(np.mean(list(r["column_alpha"].values()))) for r in rows]),
                "C_q":distribution([r["C_q"] for r in rows if r["C_q"] is not None]),
                "C_candidate":distribution([c for r in rows for c in r["C"].values() if c is not None]),
                "missing_pairs":sum(c is None for r in rows for c in r["C"].values()),
                "outside_pair_alpha":{s:distribution([r["column_alpha"][t] for r in rows for t in r[s]]) for s in ("EO_ANN","EO_EXACT","U_ONLY_VS_M","RAW_EXACT_D100_OUTSIDE")}}
        cqueries = [r for r in output if r["C_q"] is not None]
        pairs = [(c,t in r["positive_target_ids"]) for r in output for t,c in r["C"].items() if c is not None]
        _json(destination / "COLUMN_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid",
            "source":file_record(retrieval_receipt),"CDF":file_record(root / "COLUMN_CDF_RECEIPT.json"),"rankings":file_record(destination / "rankings.jsonl.gz"),
            "metrics_and_diagnostics":summary,"low_Cq_predicts_implicit_auc":auc([-r["C_q"] for r in cqueries],[r["query_kind"] == "implicit" for r in cqueries]),
            "high_C_predicts_qrel_positive_pair_auc":auc([c for c,l in pairs],[l for c,l in pairs]),
            "grouping_scope":"implicit/explicit are frozen query kinds; candidate distributions repeat their query kind, not independently labeled pair kind. Pair AUC contrasts qrel-positive against unlabeled candidates, not confirmed negatives.",
            "column_type_policy":"All visible columns compatible; no reliable CTA. Generic names/ID columns can dominate max; no fitted gate or threshold.",
            "cost":{"device":device_name,"query_seconds_p50":float(np.quantile(seconds,.5)),"query_seconds_p95":float(np.quantile(seconds,.95)),
                    "scope":"Offline scoring with cross-model identical-pair C reuse; not claimed as cold online latency"}})
        completed.append(name)
        print(json.dumps({"column_complete":name}),flush=True)
    return {"completed":completed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--generator",action="append")
    args = parser.parse_args()
    print(json.dumps(run(args.device,args.generator)))

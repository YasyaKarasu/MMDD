"""User extension: B13 -> fixed T0 reranking -> Equal/conf/column fusion."""
from __future__ import annotations

import argparse
import json

import torch

from mmdd_stage1.r26_column import ColumnSimilarity,QueryBalancedCDF
from mmdd_stage1.r26_metrics import fuse_channels,population_metrics
from prepare_stage1_r26 import ROOT,OUT,file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json

KS = (1,3,5,7,9,10,20,50)


def teacher_then_fusion(row: dict, teacher_scores: dict[str,float], pool: list[str], column_alpha: dict | None) -> dict:
    """T0 reorders each channel's membership; its E scores are QT, not path scores."""
    members = set(pool)
    original_d = [r for r in row["D100_ANN"] if r["target_id"] in members]
    original_e = [r for r in row["E_paths"] if r["target_id"] in members]
    teacher_d = sorted([{"target_id":r["target_id"],"direct_score":teacher_scores[r["target_id"]]} for r in original_d],
                       key=lambda r:(-r["direct_score"],r["target_id"]))
    teacher_e = sorted([{"target_id":r["target_id"],"teacher_QT_score":teacher_scores[r["target_id"]]} for r in original_e],
                       key=lambda r:(-r["teacher_QT_score"],r["target_id"]))
    variants = {"B13_original_channels":fuse_channels(original_d,original_e,column_alpha),
                "T0_D_original_E":fuse_channels(teacher_d,original_e,column_alpha),
                "T0_D_T0_E":fuse_channels(teacher_d,teacher_e,column_alpha)}
    rankings = {f"{variant}/{method}":rank for variant,result in variants.items() for method,rank in result["rankings"].items()}
    rankings["T0_only"] = sorted(pool,key=lambda t:(-teacher_scores[t],t))
    rankings["B13_QT_only"] = sorted(pool,key=lambda t:(-row["QT_OVER_U_scores"][t],t))
    if any(set(rank) != members for rank in rankings.values()):
        raise ValueError("Teacher/fusion comparison changed candidate pool")
    return {"rankings":rankings,"fusion_scores":{k:v["scores"] for k,v in variants.items()},
            "confidence_alpha":{k:v["confidence_alpha"] for k,v in variants.items()},
            "D_ids_before_T0":[r["target_id"] for r in original_d],"E_ids_before_T0":[r["target_id"] for r in original_e],
            "D_ids_after_T0":[r["target_id"] for r in teacher_d],"E_ids_after_T0":[r["target_id"] for r in teacher_e]}


def run(without_column: bool) -> dict:
    torch.set_num_threads(4)
    teacher_path = OUT / "teacher/B13/rankings.jsonl.gz"
    teacher_receipt = OUT / "teacher/B13/TEACHER_RECEIPT.json"
    if not teacher_receipt.exists():
        raise ValueError("Complete real B13 own-pool T0 scoring first")
    teacher = {r["query_id"]:r for r in read_rows(teacher_path)}
    column = cdf = None
    if not without_column:
        if not (OUT / "fusion/columns/COLUMN_RECEIPT.json").exists():
            raise ValueError("Column vectors must pass the reencoding audit")
        payload = torch.load(OUT / "fusion/columns/columns.pt",map_location="cpu",weights_only=False)
        column = ColumnSimilarity(payload["vectors"],torch.device("cpu"))
        reference = []
        for row in read_rows(OUT / "train_retrieval/column256/Qwen-Raw/rankings.jsonl.gz"):
            values = list(column.scores(row["query_id"],row["U"]).values())
            if any(v is None for v in values):
                raise ValueError("Missing natural train CDF columns")
            reference.append(values)
        cdf = QueryBalancedCDF(reference)
    outputs = []
    for row in read_rows(OUT / "rankings/B13/rankings.jsonl.gz"):
        scores = teacher[row["query_id"]]["teacher_scores"]
        alpha = {t:cdf.alpha(c) for t,c in column.scores(row["query_id"],row["U"]).items()} if column else None
        for pool_name,pool in (("natural_U_offline",row["U"]),("Equal_BT100",row["rankings"]["Equal"][:100])):
            result = teacher_then_fusion(row,scores,pool,alpha)
            outputs.append({key:row[key] for key in ("query_id","query_kind","source_table_id","positive_target_ids","candidate_pool_id")})
            outputs[-1].update({"pool_protocol":pool_name,"pool":pool,"teacher_QT_scores":{t:scores[t] for t in pool},
                               "column_alpha":{t:alpha[t] for t in pool} if alpha else None,**result})
    directory = OUT / "teacher_fusion/B13"
    directory.mkdir(parents=True,exist_ok=True)
    write_rows(directory / "rankings.jsonl.gz",outputs)
    metrics = {}
    for pool in ("natural_U_offline","Equal_BT100"):
        metrics[pool] = {}
        for kind in ("overall","implicit","explicit"):
            rows = [r for r in outputs if r["pool_protocol"] == pool and (kind == "overall" or r["query_kind"] == kind)]
            qrels = {r["query_id"]:r["positive_target_ids"] for r in rows}
            metrics[pool][kind] = {method:population_metrics({r["query_id"]:r["rankings"][method] for r in rows},qrels,KS)
                                  for method in rows[0]["rankings"]}
    receipt = {"execution_status":"ran","scientific_validity":"valid_for_executed_methods","column_status":"pending" if without_column else "ran",
        "request":"Add B13+Teacher Recall and fusion after Teacher reranking",
        "metrics":metrics,"rankings":file_record(directory / "rankings.jsonl.gz"),
        "own_retrieval":file_record(OUT / "rankings/B13/RETRIEVAL_RECEIPT.json"),"teacher":file_record(teacher_receipt),
        "script":file_record(ROOT / "src/evaluate_r26_b13_teacher_fusion.py"),
        "semantics":"Each pool is fixed before T0. T0_D_original_E isolates reranking Direct; T0_D_T0_E reorders both memberships with QT Teacher. Teacher-ranked E is not called original evidence/path rank. Confidence uses that variant's actual Direct scores. Column uses the same frozen natural train CDF.",
        "cost":"Natural U is offline; BT100 requires at most100 distinct T0 pairs/query, scores reused across channels and fusion variants. No further generation."}
    _json(directory / "TEACHER_FUSION_RECEIPT.json",receipt)
    return {"rows":len(outputs),"column_status":receipt["column_status"],"metrics":metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-column",action="store_true",help="Produce interim Equal/conf results while column encoding runs")
    print(json.dumps(run(parser.parse_args().without_column)))

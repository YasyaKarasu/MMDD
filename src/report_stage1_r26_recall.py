"""Recompute the user's Stage1 R10/20/30/40/50 from saved full rankings."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from mmdd_stage1.r26_metrics import population_metrics
from prepare_stage1_r26 import OUT,file_record
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json,sha256

KS = (10,20,30,40,50)


def summarize(rows: list[dict], population: dict[str,dict]) -> dict:
    if len(rows) != len(population) or {r["query_id"] for r in rows} != population.keys():
        raise ValueError("Stage1 reporting must preserve the full frozen query population")
    for row in rows:
        if set(row["positive_target_ids"]) != set(population[row["query_id"]]["positive_target_ids"]):
            raise ValueError("Stage1 reporting qrels changed")
    # U itself is an unordered admission set serialized by ID, not a scorer.
    methods = sorted({method for row in rows for method in row["rankings"] if method != "U"})
    summary = {}
    for kind in ("overall","implicit","explicit"):
        qrels = {q:r["positive_target_ids"] for q,r in population.items() if kind == "overall" or r["query_kind"] == kind}
        if not qrels:
            continue
        summary[kind] = {method:population_metrics({r["query_id"]:r["rankings"].get(method,[]) for r in rows},qrels,KS) for method in methods}
    return summary


def run() -> dict:
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    destination = OUT / "statistics/stage1_recall"
    destination.mkdir(parents=True,exist_ok=True)
    results,flat,missing = [],[],[]
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = canonical = spec["generator_id"]
        alias_path = OUT / "rankings" / name / "ALIAS_RECEIPT.json"
        if alias_path.exists():
            canonical = json.loads(alias_path.read_text())["canonical_generator"]
        for module,receipt_name in (("rankings","RETRIEVAL_RECEIPT.json"),("fusion","COLUMN_RECEIPT.json"),("teacher","TEACHER_RECEIPT.json")):
            directory = OUT / module / canonical
            receipt_path = directory / receipt_name
            if not receipt_path.exists():
                missing.append({"generator":name,"module":module})
                continue
            receipt = json.loads(receipt_path.read_text())
            path = directory / "rankings.jsonl.gz"
            if sha256(path) != receipt["rankings"]["sha256"]:
                raise ValueError(f"Stage1 source ranks changed: {path}")
            metrics = summarize(list(read_rows(path)),population)
            results.append({"generator":name,"canonical_generator":canonical,"module":module,"metrics":metrics,
                            "receipt":file_record(receipt_path),"rankings":file_record(path)})
            for kind,methods in metrics.items():
                for method,values in methods.items():
                    flat.append({"generator":name,"canonical_generator":canonical,"module":module,"query_kind":kind,"scorer":method,
                                 **{f"recall@{k}":values[f"recall@{k}"] for k in KS}})
        print(json.dumps({"stage1_recall_reported":name}),flush=True)
    _json(destination / "metrics.json",results)
    with (destination / "metrics.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=["generator","canonical_generator","module","query_kind","scorer",*(f"recall@{k}" for k in KS)])
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# R26 Stage1 Recall", "", "Stage1：Recall@10/20/30/40/50；Stage2：Recall@1/3/5/7/9，候选重排预算18。", "",
             "以下从已保存的完整排名逐 query 复算，未插值或重跑模型。原训练门使用的 R10 指标文件保留；本目录是用户指定 K 的报告层。", "",
             "主表：各模型自己的 U，按 QT 排序，整体 query-macro Recall。全部分桶、融合和 Teacher 结果见 metrics.csv/json。", "",
             "| Generator | R10 | R20 | R30 | R40 | R50 |", "|---|---:|---:|---:|---:|---:|"]
    for row in flat:
        if row["module"] != "rankings" or row["query_kind"] != "overall" or row["scorer"] != "QT_OVER_U":
            continue
        if row["generator"] != row["canonical_generator"]:
            continue
        name = row["generator"]
        if "/step" in name and not (name.endswith("/step178") or name.endswith("/step659")):
            continue
        lines.append("| "+name+" | "+" | ".join(f"{100*row[f'recall@{k}']:.2f}%" for k in KS)+" |")
    lines += ["",f"仍缺 {len(missing)} 个模块输出；此报告覆盖已完成部分，不代表完整 R26 已完成。" if missing else "本报告覆盖当前 inventory 的全部模块；完整实验完成性仍以总合同审计为准。"]
    (destination / "RESULTS.md").write_text("\n".join(lines)+"\n")
    _json(destination / "REPORT_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid_for_available_models","stage1_ks":KS,
          "source_results":len(results),"missing":missing,"population":file_record(OUT / "common/dev_queries.jsonl"),
          "metrics":file_record(destination / "metrics.json"),"csv":file_record(destination / "metrics.csv"),"report":file_record(destination / "RESULTS.md"),
          "code":file_record(Path(__file__))})
    _json(OUT / "REPORTING_PROTOCOL.json",{"authority":"User correction: Stage1 K={10,20,30,40,50}; small K only for Stage2 final output",
          "stage1_recall_ks":KS,"stage2_recall_ks":[1,3,5,7,9],"stage2_candidate_budget":18,
          "scope":"Reporting-only override; candidate pools, model runs, frozen training gates and Stage2 budgets unchanged."})
    return {"results":len(results),"missing":len(missing),"stage1_ks":KS}


if __name__ == "__main__":
    print(json.dumps(run()))

"""Recompute R26 endpoint/trajectory, EO retention, budget gate and paired CIs."""
from __future__ import annotations

import json
import numpy as np

from mmdd_stage1.r26_metrics import query_metrics
from mmdd_stage1.r26_statistics import loss_extension_gate, source_cluster_comparison
from prepare_stage1_r26 import ROOT, OUT, file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json


def run() -> dict:
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    metrics, evidence, per_query, admissions = {}, [], {}, []
    raw_path = OUT / "rankings/Qwen-Raw/rankings.jsonl.gz"
    raw_exact = {r["query_id"]:set(r["D100_EXACT"]) for r in read_rows(raw_path)}
    for item in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        generator = item["generator_id"]
        directory = OUT / "rankings" / generator
        receipt = directory / "RETRIEVAL_RECEIPT.json"
        if not receipt.exists():
            alias = directory / "ALIAS_RECEIPT.json"
            if not alias.exists():
                continue
            directory = OUT / "rankings" / json.loads(alias.read_text())["canonical_generator"]
            receipt = directory / "RETRIEVAL_RECEIPT.json"
        metrics[generator] = json.loads((directory / "metrics.json").read_text())
        evidence.append({"generator_id":generator,"receipt":file_record(receipt)})
        records = {}
        for row in read_rows(directory / "rankings.jsonl.gz"):
            q = row["query_id"]
            if row["source_table_id"] != population[q]["source_table_id"]:
                raise ValueError("Source identity changed")
            truth = set(row["positive_target_ids"])
            sets = {name:set(row[name]) for name in ("EO_ANN","EO_EXACT","U_ONLY_VS_M")}
            sets["RAW_EXACT_D100_OUTSIDE"] = truth & (set(row["U"])-raw_exact[q])
            values = {f"{method}/{key}":value for method,ranking in row["rankings"].items()
                      for key,value in query_metrics(ranking,truth,(10,20,30,40,50)).items()}
            values.update({name:len(targets)/len(truth) for name,targets in sets.items()})
            records[q] = {"query_id":q,"source_table_id":row["source_table_id"],"query_kind":row["query_kind"],"values":values}
            admissions.append({"generator_id":generator,"query_id":q,"source_table_id":row["source_table_id"],"query_kind":row["query_kind"],
                "positive_target_ids":sorted(truth),"sets":{name:sorted(targets) for name,targets in sets.items()},
                "scorer_retention":{method:{name:{str(k):sorted(targets & set(ranking[:k])) for k in (10,20,30,40,50)} for name,targets in sets.items()}
                                    for method,ranking in row["rankings"].items()},
                "witness_source":str(directory / "rankings.jsonl.gz"),"witness_correctness":"unknown unless independently labeled"})
        if records.keys() != population.keys():
            raise ValueError("Evaluation population differs from frozen denominator")
        per_query[generator] = records
    destination = OUT / "statistics"
    destination.mkdir(parents=True,exist_ok=True)
    write_rows(destination / "EO_sets_and_scorer_retention.jsonl.gz",admissions)
    write_rows(destination / "per_query_metrics.jsonl.gz",[{"generator_id":g,**r} for g,rows in per_query.items() for r in rows.values()])
    _json(OUT / "acceptance/LOSS_EXTENSION_GATE.json",{**loss_extension_gate(metrics),"inputs":evidence})
    # Each comparison is fixed in advance; no best checkpoint or best seed selection.
    comparisons = [
        ("order_native","R26-O-NATIVE","R25-B13-FULL"),
        ("order_sup","R26-O-SUP","R25-SPLIT-SUP"),
        ("path_vs_edge","R26-O-SUP","R26-E-GRAPH"),
        ("R25_added_QTKD","R25-SPLIT-QTKD","R25-SPLIT-SUP"),
        ("R25_added_U","R25-SPLIT-U","R25-SPLIT-SUP"),
        ("R25_U_added_QTKD","R25-SPLIT-UQTKD","R25-SPLIT-U"),
        ("R25_LSE_vs_split","R25-LSE-QTKD","R25-SPLIT-QTKD"),
        ("R26_added_QTKD","R26-O-QTKD","R26-O-SUP"),
        ("R26_added_U","R26-O-U","R26-O-SUP"),
        ("R26_U_added_QTKD","R26-O-UQTKD","R26-O-U"),
        ("R26_QTKD_added_U","R26-O-UQTKD","R26-O-QTKD"),
        ("R26_LSE_vs_split","R26-O-LSE-QTKD","R26-O-QTKD")]
    outputs = []
    metric_names = ["QT_OVER_U/recall@10","QT_OVER_U/recall@20","QT_OVER_U/recall@30","QT_OVER_U/recall@40","QT_OVER_U/recall@50",
                    "U/raw_recall","D100_ANN/raw_recall","D100_EXACT/recall@10","EO_ANN","EO_EXACT","U_ONLY_VS_M",
                    "Equal/recall@10","Conf/recall@10"]
    resolved = [(name,[f"{left}/seed{s}/step178" for s in (13,29)],
                 [f"{right}/seed{s}/step178" for s in (13,29)]) for name,left,right in comparisons]
    endpoint_families = sorted({g.split("/seed")[0] for g in per_query if g.endswith("/step178")})
    for family in endpoint_families:
        for baseline in ("Qwen-Raw","B13"):
            # Repeating one baseline ID aligns seed-specific deltas; it never
            # invents a second raw/B13 model or doubles the query sample size.
            resolved.append((family+"_vs_"+baseline,[f"{family}/seed{s}/step178" for s in (13,29)],[baseline,baseline]))
        for end,start in ((89,0),(178,89),(178,0)):
            resolved.append((family+f"_trajectory_{end}_minus_{start}",
                             [f"{family}/seed{s}/step{end}" for s in (13,29)],
                             [f"{family}/seed{s}/step{start}" for s in (13,29)]))
    for name,left_ids,right_ids in resolved:
        if any(g not in per_query for g in left_ids+right_ids):
            continue
        for kind in ("overall","implicit","explicit"):
            ids = [q for q,r in population.items() if kind == "overall" or r["query_kind"] == kind]
            for metric in metric_names:
                deltas = [np.array([per_query[a][q]["values"][metric]-per_query[b][q]["values"][metric] for q in ids]) for a,b in zip(left_ids,right_ids)]
                stat = source_cluster_comparison(np.mean(deltas,axis=0),[population[q]["source_table_id"] for q in ids])
                outputs.append({"comparison":name,"left":left_ids,"right":right_ids,"kind":kind,"metric":metric,
                                "seed_deltas":dict(zip(("13","29"),(float(v.mean()) for v in deltas))),**stat})
    write_rows(destination / "paired_source_bootstrap.jsonl",outputs)
    _json(destination / "STAGE1_ANALYSIS_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid_for_available_models",
        "generators":list(metrics),"inputs":evidence,"comparisons":len(outputs),"population":file_record(OUT / "common/dev_queries.jsonl"),
        "code":file_record(ROOT / "src/analyze_stage1_r26.py"),"interpretation":"Exploratory dev analysis; endpoint178, paired seeds averaged per query before source-cluster resampling"})
    return {"generators":len(metrics),"comparisons":len(outputs),"loss_extension_gate":loss_extension_gate(metrics)}


if __name__ == "__main__":
    # The extension-evaluation and downstream-fusion queues both request this
    # analysis. Serialize their writes to the same statistics artifacts.
    import fcntl
    with (OUT / "statistics/.stage1_analysis.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        print(json.dumps(run()))

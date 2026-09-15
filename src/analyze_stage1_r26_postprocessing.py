"""Inspect frozen column gates and the added-C2-KD by online-T0 comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmdd_stage1.r26_metrics import query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r26 import OUT,file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json,sha256


def verify_rows(directory: Path, receipt_name: str, population: dict) -> tuple[list[dict],dict]:
    receipt = json.loads((directory / receipt_name).read_text())
    path = directory / "rankings.jsonl.gz"
    if sha256(path) != receipt["rankings"]["sha256"]:
        raise ValueError(f"Postprocessing ranks changed: {path}")
    rows = list(read_rows(path))
    if len(rows) != len(population) or {r["query_id"] for r in rows} != population.keys():
        raise ValueError("Postprocessing lost frozen queries")
    for row in rows:
        truth = population[row["query_id"]]
        if (set(row["positive_target_ids"]) != set(truth["positive_target_ids"])
                or row["source_table_id"] != truth["source_table_id"]):
            raise ValueError("Postprocessing qrels/source population differs")
    return rows,receipt


def bin_index(value: float | None, cuts: list[float]) -> int:
    return -1 if value is None else int(np.searchsorted(cuts,value,side="right"))


def run() -> dict:
    population = {r["query_id"]:r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    destination = OUT / "statistics/postprocessing"
    destination.mkdir(parents=True,exist_ok=True)
    bins,teacher_cells,teacher_comparisons,inputs,missing = [],[],[],[],[]
    teacher_queries = {}
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        # Aliases have no independent observations and are registered separately.
        if (OUT / "rankings" / name / "ALIAS_RECEIPT.json").exists():
            continue
        fusion = OUT / "fusion" / name
        if (fusion / "COLUMN_RECEIPT.json").exists():
            rows,receipt = verify_rows(fusion,"COLUMN_RECEIPT.json",population)
            if sha256(Path(receipt["CDF"]["path"])) != receipt["CDF"]["sha256"]:
                raise ValueError("Column diagnostics must use the originally frozen CDF")
            inputs.append(file_record(fusion / "COLUMN_RECEIPT.json"))
            for kind in ("overall","implicit","explicit"):
                subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
                for field,cuts in (("C_q",[.2,.4,.6,.8]),("confidence_alpha",[.1,.3,.5,.7,.9])):
                    for index in range(-1,len(cuts)+1):
                        selected = [r for r in subset if bin_index(r[field],cuts) == index]
                        means = {method:float(np.mean([query_metrics(r["rankings"][method],r["positive_target_ids"],(10,))["recall@10"] for r in selected]))
                                 for method in ("Direct-only","Equal","Conf","Column")} if selected else {}
                        bins.append({"generator":name,"query_kind":kind,"unit":"query","variable":field,
                                     "cuts":cuts,"bin":index,"count":len(selected),"R10":means,
                                     "delta_vs_equal":{m:v-means["Equal"] for m,v in means.items()},
                                     "delta_vs_direct":{m:v-means["Direct-only"] for m,v in means.items()}})
                # Candidate bins inherit their query kind. They do not have an
                # independent implicit/explicit label or confirmed-negative GT.
                for field,cuts in (("C",[.2,.4,.6,.8]),("column_alpha",[.1,.3,.5,.7,.9])):
                    counts = {i:[0,0,0] for i in range(-1,len(cuts)+1)}
                    for row in subset:
                        positive,eo = set(row["positive_target_ids"]),set(row["EO_ANN"])
                        for target,value in row[field].items():
                            cell = counts[bin_index(value,cuts)]
                            cell[0] += 1
                            cell[1] += int(target in positive)
                            cell[2] += int(target in eo)
                    for index,(count,positive,eo) in counts.items():
                        bins.append({"generator":name,"query_kind":kind,"unit":"candidate_pair","variable":field,
                                     "cuts":cuts,"bin":index,"count":count,"qrel_positive_pairs":positive,"EO_ANN_positive_pairs":eo})
        else:
            missing.append({"generator":name,"module":"column"})
        directory = OUT / "teacher" / name
        if not (directory / "TEACHER_RECEIPT.json").exists():
            missing.append({"generator":name,"module":"teacher"})
            continue
        rows,_ = verify_rows(directory,"TEACHER_RECEIPT.json",population)
        inputs.append(file_record(directory / "TEACHER_RECEIPT.json"))
        teacher_queries[name] = {r["query_id"]:r for r in rows}
        for kind in ("overall","implicit","explicit"):
            subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
            for method in rows[0]["rankings"]:
                values = [query_metrics(r["rankings"][method],r["positive_target_ids"],(10,20,30,40,50)) for r in subset]
                teacher_cells.append({"generator":name,"query_kind":kind,"scorer":method,"queries":len(subset),
                                      "metrics":{key:float(np.mean([v[key] for v in values])) for key in values[0]}})
            for pool,left,right in (("BT100","BT100_T0","BT100_NO_T0"),
                                    ("D100","D100_T0","D100_NO_T0"),
                                    ("U_vs_M_OFFLINE","U_OFFLINE_T0","M_OFFLINE_T0")):
                for k in (10,20,30,40,50):
                    delta = np.array([query_metrics(r["rankings"][left],r["positive_target_ids"],(k,))[f"recall@{k}"]-
                                      query_metrics(r["rankings"][right],r["positive_target_ids"],(k,))[f"recall@{k}"] for r in subset])
                    teacher_comparisons.append({"comparison":"T0_candidate_composition_offline" if pool == "U_vs_M_OFFLINE" else "online_T0_same_pool","generator":name,"query_kind":kind,"pool":pool,"k":k,
                                               **source_cluster_comparison(delta,[r["source_table_id"] for r in subset])})
    # Fixed arm comparisons, averaging the two seed deltas per query before
    # resampling sources. Every model uses its own Equal prequeue in both cells.
    factorial = []
    for version,prefix in (("R25","SPLIT-"),("R26","O-")):
        for control,kd in (("SUP","QTKD"),("U","UQTKD")):
            left = [f"{version}-{prefix}{kd}/seed{s}/step178" for s in (13,29)]
            right = [f"{version}-{prefix}{control}/seed{s}/step178" for s in (13,29)]
            if any(name not in teacher_queries for name in left+right):
                factorial.append({"version":version,"control":control,"kd":kd,"status":"unassessable","missing":[n for n in left+right if n not in teacher_queries]})
                continue
            for kind in ("overall","implicit","explicit"):
                ids = [q for q,r in population.items() if kind == "overall" or r["query_kind"] == kind]
                for k in (10,20,30,40,50):
                    deltas = {}
                    for method in ("BT100_NO_T0","BT100_T0"):
                        def value(name,q):
                            row = teacher_queries[name][q]
                            return query_metrics(row["rankings"][method],row["positive_target_ids"],(k,))[f"recall@{k}"]
                        deltas[method] = [np.array([value(a,q)-value(b,q) for q in ids]) for a,b in zip(left,right)]
                    deltas["KD_by_online_T0_interaction"] = [a-b for a,b in zip(deltas["BT100_T0"],deltas["BT100_NO_T0"])]
                    for method,values in deltas.items():
                        factorial.append({"version":version,"control":control,"kd":kd,"status":"ran","left":left,"right":right,
                                          "query_kind":kind,"k":k,"scorer":method,"seed_deltas":dict(zip(("13","29"),(float(x.mean()) for x in values))),
                                          **source_cluster_comparison(np.mean(values,axis=0),[population[q]["source_table_id"] for q in ids])})
    for name,rows in (("fusion_bins",bins),("teacher_cells",teacher_cells),("teacher_same_pool_bootstrap",teacher_comparisons),("teacher_2x2",factorial)):
        write_rows(destination / (name+".jsonl"),rows)
    result = {"execution_status":"ran","scientific_validity":"valid_for_available_models","missing":missing,"inputs":inputs,
              "bin_rule":"Fixed cuts before this analysis; left-closed/right-open, final upper tail included; bin -1 is missing. No fitted gate.",
              "pair_semantics":"Implicit/explicit are query labels; qrel positives versus unlabeled candidates, no pair-type or confirmed-negative claims.",
              "factorial_scope":"Own-generator Equal BT100 per cell; +T0 changes only its ranking. KD arm contrasts include own-pool composition changes; both arms share a Teacher-trained C1.",
              "code":file_record(Path(__file__)),"files":{p.name:file_record(p) for p in destination.glob("*.jsonl")}}
    _json(destination / "POSTPROCESSING_RECEIPT.json",result)
    return {"fusion_bins":len(bins),"teacher_models":len(teacher_queries),"factorial_records":len(factorial),"missing":len(missing)}


if __name__ == "__main__":
    print(json.dumps(run()))

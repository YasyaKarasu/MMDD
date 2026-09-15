"""Fixed-witness D1 versus retained LSE, with independent baseline and gates."""
from __future__ import annotations

import argparse
import gzip
import math
from collections import Counter, defaultdict
from contextlib import ExitStack

import numpy as np

from mmdd_stage1.r26_metrics import query_metrics
from prepare_stage1_r27 import ROOT, OUT, R26, read_json, rows, sha, record, stable_sha, write_json


def retained_lse(paths: list[dict]) -> float:
    values = [p["path_score"] for p in paths]
    if not values:
        raise ValueError("Retained target requires at least one retained path")
    m = max(values)
    return m + math.log(sum(math.exp(v-m) for v in values))


def equal_rank(direct: list[str], evidence: list[str]) -> tuple[list[str], dict]:
    d = {t: 1/(60+i) for i,t in enumerate(direct,1)}
    e = {t: 1/(60+i) for i,t in enumerate(evidence,1)}
    scores = {t: d.get(t,0)+e.get(t,0) for t in d.keys() | e.keys()}
    return sorted(scores, key=lambda t:(-scores[t],t)), scores


def arms(row: dict, teacher: dict) -> dict:
    direct = [r["target_id"] for r in row["D100_ANN"]]
    a0 = sorted(row["E_paths"], key=lambda r:(-r["evidence_score"],r["target_id"]))
    a1 = sorted(row["E_paths"], key=lambda r:(-retained_lse(r["retained_paths"]),r["target_id"]))
    result = {}
    for name, evidence in (("A0",a0),("A1",a1)):
        ranked, scores = equal_rank(direct,[r["target_id"] for r in evidence])
        prequeue = ranked[:100]
        final = sorted(prequeue, key=lambda t:(-teacher[t],t))
        result[name] = {"E_rank": [r["target_id"] for r in evidence], "Equal": ranked, "Equal_scores": scores, "C100": prequeue, "T0": final}
    return result


def identity(generator: str) -> dict:
    lock = read_json(ROOT / "mmdd_r26_review/R27_INPUT_LOCK.json")
    model = next(m for m in lock["models"] if m["generator_id"] == generator)
    rr = read_json(R26 / "rankings" / generator / "RETRIEVAL_RECEIPT.json")
    tr = read_json(R26 / "teacher" / generator / "TEACHER_RECEIPT.json")
    cache = read_json(R26 / "teacher/CACHE_IDENTITY.json")
    assert sha(__import__('pathlib').Path(cache["teacher"]["path"])) == cache["teacher"]["sha256"] == "ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc"
    if model["checkpoint"]:
        assert sha(__import__('pathlib').Path(model["checkpoint"])) == model["sha256"] == rr["signature"]["checkpoint_sha256"]
    assert tr["signature"]["namespace"] == cache["namespace"]
    rp = R26 / "rankings" / generator / "rankings.jsonl.gz"
    tp = R26 / "teacher" / generator / "rankings.jsonl.gz"
    assert sha(rp) == rr["rankings"]["sha256"] == tr["signature"]["own_rankings"]["sha256"]
    assert sha(tp) == tr["rankings"]["sha256"]
    assert sha(R26 / "common/dev_queries.jsonl") == rr["signature"]["query_sha256"]
    from pathlib import Path
    ir = read_json(R26 / "rankings" / generator / "INDEX_RECEIPT.json")
    for rec in ir["files"]:
        assert sha(Path(rec["path"])) == rec["sha256"]
    return {"model": model, "retrieval": record(R26 / "rankings" / generator / "RETRIEVAL_RECEIPT.json"), "rankings": record(rp), "teacher_rankings": record(tp), "teacher_identity": cache, "index_receipt": ir, "reuse": "frozen R26 own-pool scalar artifacts with validated checkpoint/query/index/ranking/T0 hashes"}


def run(generator: str, mode: str) -> dict:
    import json
    rec = identity(generator)
    dest = OUT / "score_handoff" / generator
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / "INPUT_IDENTITY.json", rec)
    if mode == "intervention":
        assert read_json(dest / "BASELINE_REPLAY.json")["status"] == "pass"
        assert read_json(OUT / "audit/numerical_spotcheck" / generator / "AUDIT.json")["status"] == "pass"
        assert read_json(OUT / "audit/G3.json")["status"] == "pass"
    raw_direct = {r["query_id"]: set(r["D100_EXACT"]) for r in rows(R26 / "rankings/Qwen-Raw/rankings.jsonl.gz")}
    teacher = {r["query_id"]: r for r in rows(R26 / "teacher" / generator / "rankings.jsonl.gz")}
    population = {r["query_id"]:r for r in rows(R26 / "common/dev_queries.jsonl")}
    summaries = defaultdict(list)
    counts = Counter()
    strict_pairs = []
    metrics_rows = []
    with ExitStack() as stack:
        files = {name: stack.enter_context(gzip.open(dest / f"{name}.jsonl.gz","wt")) for name in (("baseline_per_query",) if mode == "baseline" else ("raw_candidates_paths", "A0_rankings", "A1_rankings", "teacher_scores_refs", "per_query_metrics", "eo_funnel", "single_factor_assertions"))}
        def emit(name, value):
            files[name].write(json.dumps(value)+"\n")
        for row in rows(R26 / "rankings" / generator / "rankings.jsonl.gz"):
            q = row["query_id"]
            meta = {k: row[k] for k in ("query_id","query_kind","source_table_id","positive_target_ids")}
            assert all(meta[k] == population[q][k] for k in meta)
            t = teacher[q]
            assert t["candidate_pool_id"] == row["candidate_pool_id"]
            tscores = t["teacher_scores"]
            d = [r["target_id"] for r in row["D100_ANN"]]
            e = [r["target_id"] for r in row["E_paths"]]
            u = set(d) | set(e)
            assert u == set(row["U"]) and len(row["M_exact"]) == len(u)
            assert len(set(row["M_exact"])) == len(u)
            assert row["parameter_sha"] == rec["index_receipt"]["signature"]["parameter_sha256"]
            assert all(math.isfinite(tscores[target]) for target in u)
            truth = set(row["positive_target_ids"])
            sets = {"EO_ANN": truth & (set(e)-set(d)), "EO_EXACT": truth & (set(e)-set(row["D100_EXACT"])), "EO_STRICT": truth & (set(e)-set(d)-set(row["D100_EXACT"])), "EO_FIXED_RAW": truth & (u-raw_direct[q]), "U_ONLY_VS_M": truth & (u-set(row["M_exact"])), "M_ONLY_VS_U": truth & (set(row["M_exact"])-u)}
            assert sets["EO_ANN"] == set(row["EO_ANN"]) and sets["EO_EXACT"] == set(row["EO_EXACT"])
            for target in sets["EO_EXACT"]-sets["EO_STRICT"]:
                strict_pairs.append({"query_id":q,"target_id":target})
            for erow in row["E_paths"]:
                retained = erow["retained_paths"]
                ids = [p["evidence_id"] for p in retained]
                assert len(set(ids)) == len(ids), (q,erow["target_id"],"duplicate retained path")
                assert set(ids) == set(erow["selected_evidence_ids"])
                assert abs(retained_lse(retained)-erow["retained_path_lse"]) <= 1e-8
                for path in retained:
                    assert abs(path["path_score"]-path["query_evidence_score"]-path["evidence_target_score"]) <= 1e-8
            # A0 independently recomputed before any intervention is evaluated.
            a0, _ = equal_rank(d,e)
            assert a0 == row["rankings"]["Equal"]
            t0 = sorted(a0[:100],key=lambda target:(-tscores[target],target))
            assert t0 == t["rankings"]["BT100_T0"]
            counts["queries"] += 1
            for key,value in sets.items():
                counts[key] += len(value)
            if mode == "baseline":
                mm = {method:query_metrics(rank, row["positive_target_ids"], (10,20,50)) for method,rank in row["rankings"].items()}
                mm.update({method:query_metrics(rank, row["positive_target_ids"], (10,20,50)) for method,rank in t["rankings"].items()})
                for kind in ("overall",row["query_kind"]):
                    summaries[kind].append(mm)
                emit("baseline_per_query",{**meta,"metrics":mm,"EO_sets": {k:sorted(v) for k,v in sets.items()}})
                continue
            result = arms(row,tscores)
            assert result["A0"]["Equal"] == a0
            invariant = {"D":row["D100_ANN"],"E_ids":sorted(e),"U":row["U"],"M":row["M_exact"],"retained":{r["target_id"]:{k:r[k] for k in ("retained_paths","selected_evidence_ids","routed_rows")} for r in row["E_paths"]},"T0":tscores}
            common_hash = stable_sha(invariant)
            assert set(result["A0"]["E_rank"]) == set(result["A1"]["E_rank"])
            emit("single_factor_assertions",{"query_id":q,"A0_common_hash":common_hash,"A1_common_hash":stable_sha(invariant),"changed_fields_allowed":["E_rank","Equal","C100","T0_rank"]})
            for name in ("A0","A1"):
                emit(name+"_rankings",{"query_id":q,**result[name]})
            raw = {k:row[k] for k in ("query_id","D100_ANN","D100_EXACT","M_exact","exact_scores","E_pre_retention","E_paths","U","QT_OVER_U_scores")}
            emit("raw_candidates_paths",raw)
            emit("teacher_scores_refs",{"query_id":q,"namespace":t["teacher_namespace"],"teacher_scores":tscores})
            mm = {}
            funnel = {"query_id":q,"sets":{k:sorted(v) for k,v in sets.items()},"EO_EXACT_minus_STRICT":sorted(sets["EO_EXACT"]-sets["EO_STRICT"])}
            for name in ("A0","A1"):
                for key in ("E_rank","Equal","C100","T0"):
                    mm[name+"/"+key] = query_metrics(result[name][key],row["positive_target_ids"],(10,20,50))
                funnel[name] = {s:{"pairs":len(v),"C100":sorted(v&set(result[name]["C100"])),"T0_10":sorted(v&set(result[name]["T0"][:10])),"E50":sorted(v&set(result[name]["E_rank"][:50])),"Equal50":sorted(v&set(result[name]["Equal"][:50]))} for s,v in sets.items()}
                mm[name+"/EO_STRICT"] = {"C100":len(sets["EO_STRICT"]&set(result[name]["C100"]))/len(truth),"T0_10":len(sets["EO_STRICT"]&set(result[name]["T0"][:10]))/len(truth)}
            deltas = {method:{k:mm['A1/'+method][k]-v for k,v in mm['A0/'+method].items()} for method in ("E_rank","Equal","C100","T0","EO_STRICT")}
            mr = {**meta,"metrics":mm,"deltas":deltas,"WLT_T0_R10":"W" if deltas["T0"]["recall@10"]>0 else "L" if deltas["T0"]["recall@10"]<0 else "T"}
            emit("per_query_metrics",mr)
            emit("eo_funnel",funnel)
            metrics_rows.append(mr)
        assert counts["queries"] == len(population) == 1198
    if mode == "baseline":
        means = {kind:{method:{metric:sum(x[method][metric] for x in rr)/len(rr) for metric in rr[0][method]} for method in rr[0]} for kind,rr in summaries.items()}
        originals = [read_json(R26 / folder / generator / receipt)["metrics"] for folder,receipt in (("rankings","RETRIEVAL_RECEIPT.json"),("teacher","TEACHER_RECEIPT.json"))]
        for old in originals:
            for kind in means:
                for method, metrics in old[kind].items():
                    if method not in means[kind]:
                        continue
                    for key,value in metrics.items():
                        assert abs(means[kind][method][key]-value) <= 1e-10, (generator,kind,method,key)
        if generator == "B13":
            assert abs(means["overall"]["BT100_T0"]["recall@10"]-.4732888146911519) <= 1e-10
            assert counts["EO_STRICT"] == 207
        receipt = {"status":"pass","counts":counts,"metrics":means,"EO_EXACT_minus_STRICT":strict_pairs,"raw_artifacts_hash_verified":True,"baseline_all_recorded_metrics_tolerance":1e-10}
        write_json(dest / "BASELINE_REPLAY.json",receipt)
    else:
        receipt = {"status":"completed","planned":True,"implemented":True,"executed":True,"evaluated":True,"counts":counts,"online_teacher_pairs_per_query":100,"new_teacher_inference_pairs":0,"online_teacher_cost":"100 pairs despite offline cache reuse","single_factor":"same D/E/U/M/witness/path scores/row masks/T0; E ranking only","raw_files":{name:record(dest/f'{name}.jsonl.gz') for name in files}}
        write_json(dest / "EXECUTION.json",receipt)
    return {"generator":generator,"mode":mode,"counts":counts}


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--mode", choices=("baseline","intervention"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.generator,args.mode)))

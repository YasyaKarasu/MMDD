"""Reconstruct pool/scorer invariants from every canonical R26 raw ranking."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from prepare_stage1_r26 import OUT, ROOT, file_record, stable_sha
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json, sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def lse(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v-top) for v in values))


def ranked(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda target: (-scores[target], target))


def audit_row(row: dict, meta: dict, signature: dict, tables: set[str]) -> dict:
    """Check saved behavior independently of the production fusion helpers."""
    for key in ("query_id", "query_kind", "source_table_id", "positive_target_ids"):
        require(row[key] == meta[key], f"Population/qrels differ: {key}")
    require(row["generator_id"] == signature["generator_id"], "Wrong generator")
    require(row["parameter_sha"] == signature["parameter_sha256"], "Wrong parameter binding")
    require(row["index_id"] == stable_sha(signature), "Wrong index identity")
    require(row["retrieval_protocol_id"] == stable_sha(signature["protocol"]), "Wrong retrieval protocol")
    d = [r["target_id"] for r in row["D100_ANN"]]
    e = [r["target_id"] for r in row["E_paths"]]
    pre = {r["target_id"]: r for r in row["E_pre_retention"]}
    union = set(d) | set(e)
    require(len(d) == 100 and len(set(d)) == 100, "Direct budget/duplicates")
    require(len(e) == len(set(e)), "Duplicate E targets")
    require(row["U"] == sorted(union) and union <= tables, "U differs from own D union E/lake")
    require(row["U_pre_retention"] == sorted(set(d) | pre.keys()), "Pre-retention union differs")
    require(row["E_target_ids"] == e, "E membership differs from retained paths")
    require(row["candidate_pool_id"] == stable_sha({"q":row["query_id"], "D":d, "E":e}), "Pool identity differs")
    m, exact_scores = row["M_exact"], row["exact_scores"]
    require(len(m) == len(union) == len(exact_scores) and len(m) == len(set(m)), "M budget/duplicates")
    require(set(m) <= tables and m[:100] == row["D100_EXACT"], "M is not current exact prefix")
    require(all(a >= b for a,b in zip(exact_scores,exact_scores[1:])), "M exact scores unsorted")
    ranks = row["rankings"]
    require(ranks["D100_ANN"] == d and ranks["D100_EXACT"] == m[:100] and ranks["M_EXACT"] == m, "Scorer pool mismatch")
    qt = row["QT_OVER_U_scores"]
    require(set(qt) == union and ranks["QT_OVER_U"] == ranked(qt), "QT-over-U binding differs")
    require(ranks["U"] == ranked(qt), "Historical U rank alias differs")
    require(e == ranked({r["target_id"]:r["evidence_score"] for r in row["E_paths"]}), "E rank differs from D1 coverage")
    require(ranks["E_ONLY"] == e, "E-only scorer binding differs")
    paths, retained = 0, 0
    for target, before in pre.items():
        evidence_paths = [p for p in before["paths"] if p["kind"] == "evidence"]
        require(bool(evidence_paths), "E target lacks actual evidence")
        for path in evidence_paths:
            require(math.isclose(path["path_score"], path["query_evidence_score"]+path["evidence_target_score"], abs_tol=1e-10), "Path is not actual QE+ET")
        require(math.isclose(before["evidence_score"], lse([p["path_score"] for p in evidence_paths]), abs_tol=1e-10), "Natural E LSE omitted paths")
        paths += len(evidence_paths)
    for after in row["E_paths"]:
        before = pre[after["target_id"]]
        selected = after["selected_evidence_ids"]
        require(0 < len(selected) <= 4 and len(set(selected)) == len(selected), "D1 retained budget differs")
        require(after["paths"] == before["paths"], "D1 modified pre-retention paths")
        expected = [p for p in before["paths"] if p.get("evidence_id") in selected]
        require(expected == after["retained_paths"] and {p["evidence_id"] for p in expected} == set(selected), "Retained paths not selected original paths")
        require(set(after["routed_rows"]) == set(selected), "Routed-row identities differ")
        require(math.isclose(after["retained_path_lse"], lse([p["path_score"] for p in expected]), abs_tol=1e-10), "Retained LSE differs")
        retained += len(expected)
    natural = {t:lse([s,pre[t]["evidence_score"]]) if t in pre else s for t,s in qt.items()}
    require(all(math.isclose(natural[t],row["PATH_FUSED_LSE_scores"][t],abs_tol=1e-10) for t in union), "Natural fused score differs")
    require(ranks["PATH_FUSED_LSE"] == ranked(natural), "Natural fused rank differs")
    ds = [r["direct_score"] for r in row["D100_ANN"]]
    require(all(a >= b for a,b in zip(ds,ds[1:])), "Direct scores unsorted")
    alpha = 1-min(1.,max(0.,(ds[0]-ds[1])/(ds[0]-ds[-1]+1e-8)))
    dr = {t:1/(60+i) for i,t in enumerate(d,1)}
    er = {t:1/(60+i) for i,t in enumerate(e,1)}
    require(math.isclose(alpha,row["fusion"]["confidence_alpha"],abs_tol=1e-12), "Confidence used wrong score space")
    for name,weight in (("Equal",1.),("Conf",alpha)):
        scores = {t:dr.get(t,0)+weight*er.get(t,0) for t in union}
        require(scores == row["fusion"]["scores"][name], f"{name} score binding differs")
        require(ranks[name] == row["fusion"]["rankings"][name] == ranked(scores), f"{name} rank binding differs")
    truth = set(meta["positive_target_ids"])
    for key, expected in (("EO_ANN",truth & (set(e)-set(d))), ("EO_EXACT",truth & (set(e)-set(m[:100]))), ("U_ONLY_VS_M",truth & (union-set(m)))):
        require(row[key] == sorted(expected), f"{key} admission differs")
    return {"queries":1, "union_pairs":len(union), "pre_paths":paths, "retained_paths":retained}


def run(generators: list[str] | None = None) -> dict:
    population_path = OUT / "common/dev_queries.jsonl"
    population = {r["query_id"]:r for r in read_rows(population_path)}
    destination = OUT / "acceptance/raw_rankings"
    destination.mkdir(parents=True,exist_ok=True)
    results, pending, aliases = [], [], []
    source = file_record(Path(__file__))
    for spec in json.loads((OUT / "MODEL_INVENTORY.json").read_text()):
        name = spec["generator_id"]
        if generators and name not in generators:
            continue
        directory = OUT / "rankings" / name
        if (directory / "ALIAS_RECEIPT.json").exists():
            aliases.append(file_record(directory / "ALIAS_RECEIPT.json"))
            continue
        receipt_path = directory / "RETRIEVAL_RECEIPT.json"
        if not receipt_path.exists():
            pending.append(name)
            continue
        receipt = json.loads(receipt_path.read_text())
        rank_path = Path(receipt["rankings"]["path"])
        signature = receipt["signature"]
        require(sha256(rank_path) == receipt["rankings"]["sha256"], f"Ranking hash changed: {name}")
        require(signature["query_sha256"] == sha256(population_path), "Population hash changed")
        for filename,digest in signature["code_sha256"].items():
            require(sha256(ROOT / "src" / filename) == digest, f"Active evaluator source changed: {filename}")
        index_path = directory / "INDEX_RECEIPT.json"
        index = json.loads(index_path.read_text())
        require(index["signature"] == signature, "Index/checkpoint signature differs")
        ids_record = next(r for r in index["files"] if r["path"].endswith("/table_ids.json"))
        require(sha256(Path(ids_record["path"])) == ids_record["sha256"], "Lake IDs changed")
        tables = set(json.loads(Path(ids_record["path"]).read_text()))
        require(len(tables) == 22886 and not tables.intersection(population), "Lake count/self-exclusion differs")
        counts, seen = Counter(), set()
        for row in read_rows(rank_path):
            query = row["query_id"]
            require(query not in seen, "Duplicate query")
            try:
                counts.update(audit_row(row,population[query],signature,tables))
            except (ValueError,KeyError) as exc:
                raise ValueError(f"{name}/{query}: {exc}") from exc
            seen.add(query)
        require(seen == population.keys(), "Incomplete full query population")
        result = {"generator":name,"execution_status":"ran","scientific_validity":"valid",
                  "observed":dict(counts),"rankings":receipt["rankings"],"retrieval_receipt":file_record(receipt_path),
                  "index_receipt":file_record(index_path),"population":file_record(population_path),"code":source}
        _json(destination / name / "AUDIT.json",result)
        results.append(result)
        print(json.dumps({"rank_invariants_verified":name,**dict(counts)}),flush=True)
    result = {"execution_status":"ran","scientific_validity":"valid_for_available_canonical_models",
              "verified_at":datetime.now(timezone.utc).isoformat(),"test_ids":["G02","G03","G04"],
              "models":results,"pending":pending,"aliases":aliases,"source":source,
              "scope":"All saved canonical queries: pool, own signature/index binding, path QE+ET/LSE, D1 retained provenance, exact prefix cardinality, QT/Equal/Conf/natural-LSE ranking, EO formulas. Independent numerical full-lake exact replay and D1 row-strength replay are separate checks; this audit does not claim to recompute them."}
    _json(destination / ("AUDIT.json" if generators is None else "SELECTED_AUDIT.json"),result)
    return {"canonical_models":len(results),"pending":len(pending),"queries":sum(r["observed"]["queries"] for r in results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator",action="append")
    args = parser.parse_args()
    print(json.dumps(run(args.generator)))

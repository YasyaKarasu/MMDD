"""Recompute Teacher endpoint metrics and redistillation gates from raw ranks."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from statistics import mean

from package_stage1_r26 import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_optimization_r26_20260914"
KS = (10,20,30,40,50)


def recall(ranking: list[str], positives: list[str], k: int) -> float:
    return len(set(ranking[:k]) & set(positives)) / len(set(positives))


def redistill_decision(seed_rows: list[dict]) -> str:
    """Missing paired endpoints are unassessable, never negative evidence."""
    if len(seed_rows) != 2 or {r["seed"] for r in seed_rows} != {13,29}:
        return "unassessable"
    return "triggered" if all(r["Tnew_minus_Told_R10"]["overall"] > 0
        and r["Tnew_minus_Told_R10"]["implicit"] >= -1e-12
        and r["lists_with_new_competitors"] > 0 for r in seed_rows) else "not_triggered"


def run() -> dict:
    verified = {}

    def check(record):
        path = Path(record["path"])
        if str(path) not in verified:
            verified[str(path)] = digest(path)
        if verified[str(path)] != record["sha256"]:
            raise ValueError(f"Evaluation input changed: {path}")
        return path

    def rows(path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path,"rt") as stream:
            return [json.loads(line) for line in stream]

    result_path = OUT / "feedback/REFINEMENT_EVALUATION.json"
    result = json.loads(result_path.read_text())
    protocol_path = OUT / "feedback/REFINEMENT_EVALUATION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    gate = json.loads((OUT / "feedback/REFINEMENT_GATE_FROZEN.json").read_text())
    population = {r["query_id"]:r for r in rows(check(protocol["population"]))}
    if len(population) != 1198 or result["K"] != list(KS):
        raise ValueError("Teacher evaluation population or Stage1 K changed")
    own = {p["generator"]:{r["query_id"]:r for r in rows(check(p["rankings"]))} for p in protocol["pools"]}
    observed, values = {}, {}
    for seed, generator in zip((13,29),gate["selected_generators"]):
        for arm in ("Tcont","Told","Tnew"):
            path = OUT / f"feedback/refinement/{arm}/seed{seed}/evaluation/EVALUATION_RECEIPT.json"
            receipt = json.loads(path.read_text())
            for key in ("training","checkpoint","protocol","code"):
                check(receipt["signature"][key])
            data = rows(check(receipt["rankings"]))
            if len(data) != 2*len(population):
                raise ValueError("Incomplete B13/new evaluation rows")
            for row in data:
                pool, q = row["pool"],row["query_id"]
                name = "B13" if pool == "B13" else generator
                if pool not in ("B13","selected_new") or row["generator_id"] != name:
                    raise ValueError("Teacher evaluation pool assignment differs")
                key = arm,seed,pool,q
                if key in observed or row["arm"] != arm or row["seed"] != seed:
                    raise ValueError("Duplicate or misassigned Teacher evaluation row")
                observed[key] = row
                truth = population[q]["positive_target_ids"]
                if row["positive_target_ids"] != truth or row["source_table_id"] != population[q]["source_table_id"]:
                    raise ValueError("Evaluation qrels/source cluster changed")
                base = own[name][q]
                if row["candidate_pool_id"] != base["candidate_pool_id"]:
                    raise ValueError("Teacher control changed the candidate pool")
                scores = row["scores"]
                if set(scores) != set(base["U"]) | set(base["M_exact"]) or not all(math.isfinite(v) for v in scores.values()):
                    raise ValueError("Incomplete or invalid actual Teacher scores")
                pools = {"BT100":base["rankings"]["Equal"][:100],"D100":base["rankings"]["D100_ANN"],
                         "U_OFFLINE":base["U"],"M_OFFLINE":base["M_exact"]}
                expected = {"BT100_NO_TEACHER":pools["BT100"],"D100_NO_TEACHER":pools["D100"]}
                expected.update({p+"_TEACHER":sorted(targets,key=lambda t:(-scores[t],t)) for p,targets in pools.items()})
                if expected != row["rankings"]:
                    raise ValueError("Saved Teacher ranks differ from actual scores/frozen pools")
                for method,ranking in expected.items():
                    for k in KS:
                        values[arm,seed,pool,q,method,k] = recall(ranking,truth,k)
            for pool in ("B13","selected_new"):
                summary = result["metrics"][f"{arm}/seed{seed}/{pool}"]
                if set(summary) != {"overall","implicit","explicit"} or any(set(methods) != set(expected) for methods in summary.values()):
                    raise ValueError("Missing endpoint query groups or scorers")
                for kind,methods in summary.items():
                    qs = [q for q,r in population.items() if kind == "overall" or r["query_kind"] == kind]
                    for method,metrics in methods.items():
                        for k in KS:
                            actual = mean(values[arm,seed,pool,q,method,k] for q in qs)
                            if not math.isclose(actual,metrics[f"recall@{k}"],abs_tol=1e-12):
                                raise ValueError("Reported endpoint Recall differs from raw ranks")
            print(json.dumps({"audited_evaluation":arm,"seed":seed}),flush=True)
    comparison_keys = {(r["pool"],r["new"],r["old"],r["kind"],r["method"],r["K"]) for r in result["comparisons"]}
    expected_keys = {(pool,new,old,kind,method,k) for pool in ("B13","selected_new")
                     for new,old in (("Told","Tcont"),("Tnew","Told")) for kind in ("overall","implicit","explicit")
                     for method in ("BT100_TEACHER","D100_TEACHER","U_OFFLINE_TEACHER","M_OFFLINE_TEACHER") for k in KS}
    if len(result["comparisons"]) != 240 or comparison_keys != expected_keys:
        raise ValueError("Missing fixed pool/arm/kind/scorer/K comparisons")
    for comparison in result["comparisons"]:
        qs = [q for q,r in population.items() if comparison["kind"] == "overall" or r["query_kind"] == comparison["kind"]]
        deltas = {}
        for seed in (13,29):
            deltas[str(seed)] = mean(values[comparison["new"],seed,comparison["pool"],q,comparison["method"],comparison["K"]]-
                                     values[comparison["old"],seed,comparison["pool"],q,comparison["method"],comparison["K"]] for q in qs)
        if any(not math.isclose(v,comparison["seed_deltas"][s],abs_tol=1e-12) for s,v in deltas.items()) or not math.isclose(mean(deltas.values()),comparison["mean_delta"],abs_tol=1e-12):
            raise ValueError("Paired contrast differs from raw ranks")
    seeds = []
    for seed,generator in zip((13,29),gate["selected_generators"]):
        differences = {}
        for kind in ("overall","implicit"):
            qs = [q for q,r in population.items() if kind == "overall" or r["query_kind"] == kind]
            differences[kind] = mean(values["Tnew",seed,"selected_new",q,"BT100_TEACHER",10]-values["Told",seed,"selected_new",q,"BT100_TEACHER",10] for q in qs)
        source = OUT / "feedback" / generator / "H_old_H_new_comparison.jsonl.gz"
        novel = sum(bool(r["new_unique_negatives"]) for r in rows(source))
        seeds.append({"seed":seed,"Tnew_minus_Told_R10":differences,"lists_with_new_competitors":novel})
    decision = redistill_decision(seeds)
    frozen = json.loads((OUT / "feedback/REDISTILL_GATE_FROZEN.json").read_text())
    check(frozen["evaluation"])
    check(frozen["protocol"])
    if decision != frozen["status"] or decision != result["redistill_gate"]["status"]:
        raise ValueError("Frozen redistillation gate differs from actual endpoint evidence")
    for actual, reported in zip(seeds,frozen["seeds"]):
        check(reported["hard_comparison"])
        if actual["seed"] != reported["seed"] or actual["lists_with_new_competitors"] != reported["lists_with_new_competitors"] or any(
                not math.isclose(v,reported["Tnew_minus_Told_R10"][kind],abs_tol=1e-12) for kind,v in actual["Tnew_minus_Told_R10"].items()):
            raise ValueError("Reported per-seed gate measurements differ")
    audit = {"execution_status":"ran","scientific_validity":"valid","actual_rows":len(observed),
             "paired_contrasts":240,"redistill_gate":decision,"seeds":seeds,"verified_artifact_hashes":verified,
             "evaluation":{"path":str(result_path),"sha256":digest(result_path)},
             "code":{"path":str(Path(__file__)),"sha256":digest(Path(__file__))},
             "scope":"All six Teachers, both frozen pools, all1198queries: actual score-to-rank reconstruction, allK/group macro Recall and paired means, independent actual hard-competitor gate. No new neural inference or independent bootstrap resampling."}
    (OUT / "acceptance/REFINEMENT_EVALUATION_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    return {"actual_rows":len(observed),"redistill_gate":decision}


if __name__ == "__main__":
    print(json.dumps(run()))

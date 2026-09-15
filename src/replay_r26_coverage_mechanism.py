"""B13 frozen coverage mechanism and candidate-budget replay; no training."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import fuse_channels, query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from mmdd_stage1.row_support import greedy_row_bundle, load_evidence_content_keys
from replay_r26_evidence_order_abc import read_rows, read_json, write_json, record, stable_sha, lse
from run_stage1_r11_task_e import _deduplicate, _evidence_paths, _row_strengths, _sigmoid

BASE = "greedy_coverage"
ARMS = (BASE, "greedy_lse", "top4_coverage", "top4_lse", "fixed_greedy_no_weight",
        "greedy_no_path_same_pool", "greedy_no_path", "top_matched_coverage",
        "greedy_lme", "greedy_max", "pre_lse", "pre_lme", "pre_max")
BUDGETS = (50, 100, 150, 200, None)
KS = (10, 20, 50)


def coverage(paths: list[dict], support: dict[str, list[float]], *, weighted: bool = True) -> float:
    """Mean cross-row max support, with the historical shifted/clipped cosine."""
    rows = len(support[paths[0]["evidence_id"]])
    return sum(max((_sigmoid(p["path_score"]) if weighted else 1.0) * support[p["evidence_id"]][r]
                   for p in paths) for r in range(rows)) / rows


def select(paths: list[dict], support: dict[str, list[float]], *, weighted: bool) -> list[dict]:
    selected, _ = greedy_row_bundle(
        [{"evidence_id": p["evidence_id"], "quality": _sigmoid(p["path_score"]) if weighted else 1.0}
         for p in paths], row_support=support, budget=4, threshold=0.0)
    by_id = {p["evidence_id"]: p for p in paths}
    return [by_id[e] for e in selected]


def target_arms(before: dict, after: dict, support: dict, content_keys: dict) -> tuple[dict, dict, dict]:
    paths = _evidence_paths(before["paths"])
    pool = _deduplicate(paths, content_keys)[:20]
    greedy = select(pool, support, weighted=True)
    assert [p["evidence_id"] for p in greedy] == after["selected_evidence_ids"]
    assert math.isclose(coverage(greedy, support), after["evidence_score"], abs_tol=1e-12)
    top4 = pool[:4]
    # A genuinely score-free selection: content representative chosen by ID,
    # no score-based Top20 truncation. Retrieval itself remains frozen/learned.
    independent = {}
    for p in sorted(paths, key=lambda p: p["evidence_id"]):
        independent.setdefault(content_keys[p["evidence_id"]], p)
    no_path = select(list(independent.values()), support, weighted=False)
    same_pool = select(pool, support, weighted=False)
    retained = after["retained_paths"]
    scores = {
        BASE: after["evidence_score"], "greedy_lse": lse(retained),
        "top4_coverage": coverage(top4, support), "top4_lse": lse(top4),
        "fixed_greedy_no_weight": coverage(greedy, support, weighted=False),
        "greedy_no_path_same_pool": coverage(same_pool, support, weighted=False),
        "greedy_no_path": coverage(no_path, support, weighted=False),
        "top_matched_coverage": coverage(pool[:len(greedy)], support),
        "greedy_lme": lse(retained) - math.log(len(retained)),
        "greedy_max": max(p["path_score"] for p in retained),
        "pre_lse": lse(paths), "pre_lme": lse(paths) - math.log(len(paths)),
        "pre_max": max(p["path_score"] for p in paths),
    }
    bundles = {"greedy": greedy, "top4": top4, "no_path_same_pool": same_pool,
               "no_path": no_path, "top_matched": pool[:len(greedy)]}
    counts = {"targets": 1, "pre_paths": len(paths), "eligible_paths": len(pool),
              "top20_truncated": len(_deduplicate(paths, content_keys)) > 20,
              "deduplicated_paths": len(paths) - len(_deduplicate(paths, content_keys)),
              "greedy_top4_same_set": {p["evidence_id"] for p in greedy} == {p["evidence_id"] for p in top4}}
    for name, chosen in bundles.items():
        routing = {p["evidence_id"]: max(range(len(support[p["evidence_id"]])),
                                       key=lambda r: (support[p["evidence_id"]][r], -r)) for p in chosen}
        counts[name + "/paths"] = len(chosen)
        counts[name + "/routed_rows"] = len(set(routing.values()))
        counts[name + "/coverage_gain_over_best_single"] = coverage(chosen, support) - max(coverage([p], support) for p in chosen)
    return scores, {k: [p["evidence_id"] for p in v] for k, v in bundles.items()}, counts


def rank_scores(row: dict, scores: dict[str, float], teacher: dict, budget: int | None = 100) -> dict:
    evidence = sorted(scores, key=lambda t: (-scores[t], t))
    equal = fuse_channels(row["D100_ANN"], [{"target_id": t} for t in evidence])["rankings"]["Equal"]
    candidates = equal if budget is None else equal[:budget]
    return {"E": evidence, "Equal": equal, "C": candidates,
            "T0": sorted(candidates, key=lambda t: (-teacher[t], t))}


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for kind in ("overall", "implicit", "explicit"):
        subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
        arms = {}
        for arm in subset[0]["metrics"]:
            arms[arm] = {
                "metrics": {stage: {m: float(np.mean([r["metrics"][arm][stage][m] for r in subset]))
                                     for m in subset[0]["metrics"][arm][stage]}
                            for stage in subset[0]["metrics"][arm]},
                "strict_hits": {m: sum(r["strict_hits"][arm][m] for r in subset)
                                for m in subset[0]["strict_hits"][arm]},
                "teacher_pairs": sum(r["pairs"][arm] for r in subset),
                "mean_pairs": float(np.mean([r["pairs"][arm] for r in subset])),
            }
        summary[kind] = {"queries": len(subset), "strict_pairs": sum(r["strict_pairs"] for r in subset), "arms": arms}
    return summary


def comparisons(rows: list[dict]) -> list[dict]:
    contrasts = [(BASE, other) for other in ("top4_coverage", "fixed_greedy_no_weight", "greedy_no_path",
                 "greedy_no_path_same_pool", "top_matched_coverage", "greedy_lme", "greedy_max", "pre_lme", "pre_max")]
    contrasts += [("top4_coverage", "top4_lse"), (BASE, "greedy_lse")]
    contrasts += [("C" + str(b), "C100") for b in (50, 150, 200)] + [("Full-U", "C100")]
    result = []
    for kind in ("overall", "implicit", "explicit"):
        subset = [r for r in rows if kind == "overall" or r["query_kind"] == kind]
        sources = [r["source_table_id"] for r in subset]
        for new, old in contrasts:
            for stage, metric in (("E", "recall@10"), ("E", "recall@50"), ("C", "raw_recall"),
                                  ("T0", "recall@10"), ("T0", "recall@20"), ("T0", "recall@50")):
                delta = np.array([r["metrics"][new][stage][metric] - r["metrics"][old][stage][metric] for r in subset])
                result.append({"kind": kind, "new": new, "old": old, "stage": stage, "metric": metric,
                               **source_cluster_comparison(delta, sources)})
    return result


@torch.inference_mode()
def run(root: Path, output: Path) -> dict:
    torch.set_num_threads(2)
    started = time.monotonic()
    source = root / "work/stage1_optimization_r26_20260914"
    abc = root / "work/r26_evidence_order_abc_20260915"
    rp, tp = (source / p / "B13/rankings.jsonl.gz" for p in ("rankings", "teacher"))
    rr, tr = read_json(rp.with_name("RETRIEVAL_RECEIPT.json")), read_json(tp.with_name("TEACHER_RECEIPT.json"))
    features = root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
    key_path = root / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
    inputs = {"retrieval": record(rp), "teacher": record(tp), "retrieval_receipt": record(rp.with_name("RETRIEVAL_RECEIPT.json")),
              "teacher_receipt": record(tp.with_name("TEACHER_RECEIPT.json")), "features_manifest": record(features / "manifest.jsonl"),
              "content_keys": record(key_path), "population": record(source / "common/dev_queries.jsonl"),
              "abc_results": record(abc / "models/B13/RESULTS.json")}
    assert inputs["retrieval"]["sha256"] == rr["rankings"]["sha256"] == tr["signature"]["own_rankings"]["sha256"]
    assert inputs["teacher"]["sha256"] == tr["rankings"]["sha256"]
    assert inputs["features_manifest"]["sha256"] == rr["signature"]["feature_manifest_sha256"]
    assert inputs["population"]["sha256"] == rr["signature"]["query_sha256"]
    dest = output / "replay"
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "INPUTS.json", inputs)
    write_json(output / "PROTOCOL.json", {
        "generator": "B13", "arms": ARMS, "budgets": [50, 100, 150, 200, "Full-U"],
        "fixed": ["checkpoint", "pre-retention paths and scores", "E target membership", "Direct100 order", "U", "T0 QT scores", "RRF k=60"],
        "primary": ["greedy_coverage versus top4_coverage", "greedy_coverage versus fixed_greedy_no_weight",
                    "greedy_coverage versus greedy_no_path", "C200 and Full-U versus C100"],
        "selection": "2x2 uses identical content-deduplicated Path Top20 and at most4; greedy stops at nonpositive gain; matched-count control included",
        "no_path": "fixed_greedy_no_weight removes weight only; same_pool also removes greedy quality/tie dependence but retains Path-based screening; greedy_no_path uses ID-based content representatives and no Path Top20, so no path score after frozen retrieval",
        "row_similarity": "clip((frozen Qwen row dot evidence + 1)/2,0,1); continuous similarity, not truth-labeled attribute coverage",
        "P2": "LSE/LME/Max on identical greedy-retained sets, plus identical pre-retention sets; diagnostic only",
        "stats": "1198 frozen dev queries; overall/implicit/explicit query-macro recall; source-cluster paired bootstrap 10000, seed260914; exploratory multiple contrasts, no multiplicity correction",
        "strict_EO": "fixed G intersect E minus union(D_ANN100,D_EXACT100), expected207 query-target pairs",
        "cost": "exact online pair counts from replay; replay walltime is NOT Teacher inference latency; separate GPU benchmark",
        "training_gate": "Continue coverage-aligned training only if full coverage shows independent learned-score value, including strict EO; no training in this run",
        "stage2": "deferred by user; no generation or attribute correctness claims",
        "teacher_namespace": tr["signature"]["namespace"], "inputs": inputs,
    })
    teacher = {r["query_id"]: r for r in read_rows(tp)}
    population = {r["query_id"]: r for r in read_rows(source / "common/dev_queries.jsonl")}
    assert teacher.keys() == population.keys()
    store = FeatureStore.from_path(features, cache_size=20000, cache_bytes=1024**3)
    content_keys, _ = load_evidence_content_keys(key_path)
    rows, seen, counts = [], set(), Counter()
    quality, path_count, cov_scores, lse_scores = [], [], [], []
    with gzip.open(dest / "rankings.jsonl.gz", "wt", compresslevel=1) as ranks_out, \
         gzip.open(dest / "witnesses.jsonl.gz", "wt", compresslevel=1) as witness_out, \
         gzip.open(dest / "strict_evidence_only_pairs.jsonl.gz", "wt", compresslevel=1) as strict_out:
        for row in read_rows(rp):
            q = row["query_id"]
            assert q not in seen
            seen.add(q)
            meta = population[q]
            assert all(row[k] == v for k, v in meta.items())
            t = teacher[q]
            assert t["candidate_pool_id"] == row["candidate_pool_id"]
            assert t["teacher_namespace"] == tr["signature"]["namespace"]
            ts = t["teacher_scores"]
            assert all(math.isfinite(ts[tid]) for tid in row["U"])
            truth = set(row["positive_target_ids"])
            strict = truth & (set(row["E_target_ids"]) - set(row["rankings"]["D100_ANN"]) - set(row["D100_EXACT"]))
            eids = sorted({p["evidence_id"] for e in row["E_pre_retention"] for p in e["paths"] if p["kind"] == "evidence"})
            support = _row_strengths(q, eids, store)
            original = {e["target_id"]: e for e in row["E_pre_retention"]}
            scores = {a: {} for a in ARMS}
            witnesses = {}
            for after in row["E_paths"]:
                tid = after["target_id"]
                values, bundles, stats = target_arms(original[tid], after, support, content_keys)
                for arm, value in values.items():
                    scores[arm][tid] = value
                witnesses[tid] = bundles
                for group in ("all", "positive" if tid in truth else "unlabeled", "strict" if tid in strict else "non_strict"):
                    counts.update({group + "/" + k: v for k, v in stats.items()})
                quality.extend(_sigmoid(p["path_score"]) for p in original[tid]["paths"] if p["kind"] == "evidence")
                path_count.append(stats["pre_paths"])
                cov_scores.append(values[BASE])
                lse_scores.append(values["pre_lse"])
            arms = {a: rank_scores(row, s, ts) for a, s in scores.items()}
            assert arms[BASE]["E"] == row["rankings"]["E_ONLY"]
            assert arms[BASE]["Equal"] == row["rankings"]["Equal"]
            assert arms[BASE]["T0"] == t["rankings"]["BT100_T0"]
            for budget in BUDGETS:
                name = "Full-U" if budget is None else f"C{budget}"
                arms[name] = rank_scores(row, scores[BASE], ts, budget)
            assert arms["Full-U"]["T0"] == t["rankings"]["U_OFFLINE_T0"]
            assert all(set(a["E"]) == set(row["E_target_ids"]) and set(a["Equal"]) == set(row["U"]) for a in arms.values())
            metrics = {name: {stage: query_metrics(rank, list(truth), KS) for stage, rank in a.items()} for name, a in arms.items()}
            hits = {name: {"E@50": len(strict & set(a["E"][:50])), "C": len(strict & set(a["C"])),
                          **{f"T0@{k}": len(strict & set(a["T0"][:k])) for k in KS}} for name, a in arms.items()}
            rows.append({**meta, "metrics": metrics, "strict_pairs": len(strict), "strict_hits": hits,
                         "pairs": {name: len(a["C"]) for name, a in arms.items()}})
            ranks_out.write(json.dumps({**meta, "arms": arms, "E_scores": scores, "candidate_pool_id": row["candidate_pool_id"]}) + "\n")
            witness_out.write(json.dumps({"query_id": q, "row_support": support, "bundles": witnesses}) + "\n")
            for tid in sorted(strict):
                strict_out.write(json.dumps({**meta, "target_id": tid, "bundles": witnesses[tid],
                    "ranks": {name: {stage: rank.index(tid)+1 if tid in rank else None for stage, rank in a.items()}
                              for name, a in arms.items()}, "E_scores": {a: s[tid] for a, s in scores.items()}}) + "\n")
            if len(rows) % 100 == 0:
                print(json.dumps({"queries": len(rows), "total": len(population), "seconds": time.monotonic()-started}), flush=True)
    assert seen == population.keys() and len(rows) == 1198
    assert sum(r["strict_pairs"] for r in rows) == 207
    with gzip.open(dest / "per_query.jsonl.gz", "wt", compresslevel=1) as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    for kind in ("overall", "implicit", "explicit"):
        for new, old, receipt in ((BASE, "E_ONLY", rr),):
            assert all(math.isclose(summary[kind]["arms"][new]["metrics"]["E"][k], v, abs_tol=1e-12)
                       for k, v in receipt["metrics"][kind][old].items())
    result = {"summary": summary, "comparisons": comparisons(rows), "counts": dict(counts),
              "diagnostics": {"path_sigmoid_quantiles_0_10_50_90_100": np.quantile(quality, [0,.1,.5,.9,1]).tolist(),
                              "pre_path_count_quantiles": np.quantile(path_count, [0,.1,.5,.9,1]).tolist(),
                              "pearson_path_count_pre_lse": float(np.corrcoef(path_count, lse_scores)[0,1]),
                              "pearson_path_count_coverage": float(np.corrcoef(path_count, cov_scores)[0,1])},
              "baseline_reproduced_all_queries": True, "full_U_reproduced_all_queries": True,
              "elapsed_replay_seconds": time.monotonic()-started, "new_teacher_inference_pairs": 0}
    write_json(dest / "RESULTS.json", result)
    write_json(dest / "OUTPUTS.json", {p.name: record(p) for p in dest.iterdir() if p.is_file()})
    print(json.dumps({"completed": str(dest), "seconds": result["elapsed_replay_seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)

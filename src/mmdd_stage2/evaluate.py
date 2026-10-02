"""Ranking metrics with source-group paired bootstrap.

Gold = every qrel with rel > 0. A query is ``implicit`` when all its gold reasons are
``model_recoverable_join_column`` (the attribute must be recovered), ``explicit`` when all are
``explicit_visible_join_column``, otherwise ``mixed`` (counted in ``overall`` only).
Recall@k = |gold in top k| / |gold|; nDCG uses binary gains.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mmdd_dataset.wdc_runtime import iter_dataset_artifact

from .common import iter_jsonl, read_json

POLICIES = ("STAGE1", "BRIDGE_RRF60", "VISIBLE_IDF_RRF60", "BIDF_RRF60")


def metrics(order: list[str], gold: set[str], cutoffs: list[int]) -> dict[str, float]:
    result = {}
    for k in cutoffs:
        hits = len(set(order[:k]) & gold)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        result[f"R{k}"] = hits / len(gold)
        result[f"P{k}"] = hits / k
        result[f"NDCG{k}"] = sum(1 / math.log2(i + 2) for i, t in enumerate(order[:k]) if t in gold) / ideal
    return result


def bootstrap(groups: list[str], deltas: list[float], replicates: int, seed: int) -> dict[str, float]:
    """95% interval of the mean delta, resampling source groups (all their queries together)."""
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, delta in zip(groups, deltas):
        by_group[group].append(delta)
    sums = np.array([[sum(by_group[g]), len(by_group[g])] for g in sorted(by_group)])
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 250):
        draw = sums[rng.integers(0, len(sums), size=(min(250, replicates - start), len(sums)))].sum(1)
        means.extend(draw[:, 0] / draw[:, 1])
    d = np.asarray(deltas)
    return {"delta": float(d.mean()), "CI_low": float(np.quantile(means, 0.025)),
            "CI_high": float(np.quantile(means, 0.975)),
            "W": int((d > 1e-12).sum()), "L": int((d < -1e-12).sum()), "T": int((abs(d) <= 1e-12).sum())}


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(config: dict[str, Any], run: Path) -> None:
    ev = config["evaluation"]
    cutoffs = ev["cutoffs"]
    gold: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    for qrel in iter_dataset_artifact(Path(config["paths"]["dataset_root"]), "qrels"):
        if qrel.get("split") in ("dev", "test") and float(qrel.get("rel", 0)) > 0:
            gold[str(qrel["query_table_id"])].add(str(qrel["target_table_id"]))
            reasons[str(qrel["query_table_id"])].add(qrel["reason"])

    def kind(query_id: str) -> str:
        if reasons[query_id] == {"model_recoverable_join_column"}:
            return "implicit"
        return "explicit" if reasons[query_id] == {"explicit_visible_join_column"} else "mixed"

    per_query = []
    for split in ("dev", "test"):
        groups = {q["query_id"]: q["source_group"] for q in read_json(run / "population" / f"{split}.json")}
        for record in iter_jsonl(run / "scores" / f"{split}.jsonl"):
            query_id = record["query_id"]
            if not gold[query_id]:
                continue
            for policy in POLICIES:
                per_query.append({"query_id": query_id, "split": split, "kind": kind(query_id),
                                  "source_group": groups[query_id], "policy": policy,
                                  "recovery_status": record["recovery_status"],
                                  "recovered_rows": record["recovered_rows"],
                                  **metrics(record["rankings"][policy], gold[query_id], cutoffs)})

    lookup = {(r["query_id"], r["policy"]): r for r in per_query}
    summary, contrasts = [], []
    for split in ("dev", "test"):
        for population in ("overall", "implicit", "explicit"):
            queries = sorted({r["query_id"] for r in per_query if r["split"] == split
                              and (population == "overall" or r["kind"] == population)})
            if not queries:
                continue
            for policy in POLICIES:
                rows = [lookup[q, policy] for q in queries]
                summary.append({"split": split, "population": population, "policy": policy, "queries": len(rows),
                                **{key: float(np.mean([r[key] for r in rows]))
                                   for k in cutoffs for key in (f"R{k}", f"P{k}", f"NDCG{k}")}})
            for method, reference in (("BIDF_RRF60", "STAGE1"), ("BRIDGE_RRF60", "STAGE1"),
                                      ("VISIBLE_IDF_RRF60", "STAGE1"), ("BIDF_RRF60", "VISIBLE_IDF_RRF60")):
                for key in [f"{m}{k}" for k in (10, 20) for m in ("R", "NDCG")]:
                    deltas = [lookup[q, method][key] - lookup[q, reference][key] for q in queries]
                    contrasts.append({"split": split, "population": population, "method": method,
                                      "reference": reference, "metric": key,
                                      **bootstrap([lookup[q, method]["source_group"] for q in queries], deltas,
                                                  ev["bootstrap_replicates"], ev["bootstrap_seed"])})
    out = run / "evaluation"
    _csv(out / "PER_QUERY.csv", per_query)
    _csv(out / "METRICS.csv", summary)
    _csv(out / "CONTRASTS.csv", contrasts)
    for row in summary:
        if row["population"] == "overall":
            print(json.dumps({k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in row.items()
                              if k in ("split", "policy", "queries", "R5", "R10", "R20", "NDCG10")}), flush=True)

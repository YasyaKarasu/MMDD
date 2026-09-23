"""Metrics, strict cohorts, statistics (SPEC 14.1, 14.4, 14.5).

Every metric is computed from saved target-ID rankings and the full GT set
``G[q]`` (denominator always |G[q]|; queries without any positive candidate
are never dropped).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

KS = (10, 20, 30, 40, 50)


def recall_at(gold: Sequence[str], ranking: Sequence[str], k: int) -> float:
    g = set(gold)
    if not g:
        raise ValueError("empty gold")
    return len(g & set(ranking[:k])) / len(g)


def oracle_at(gold: Sequence[str], candidates: Sequence[str], k: int) -> float:
    g = set(gold)
    return min(k, len(g & set(candidates))) / len(g)


def coverage(gold: Sequence[str], candidates: Sequence[str]) -> float:
    g = set(gold)
    return len(g & set(candidates)) / len(g)


def query_metric(gold: Mapping[str, Sequence[str]], rankings: Mapping[str, Sequence[str]], k: int) -> dict[str, float]:
    return {q: recall_at(gold[q], rankings.get(q, ()), k) for q in gold}


def grouped(gold: Mapping[str, Sequence[str]], kinds: Mapping[str, str], rankings: Mapping[str, Sequence[str]],
            *, pools: Mapping[str, Sequence[str]] | None = None, ks: Sequence[int] = KS) -> dict:
    """query-macro R@K for overall/implicit/explicit, CR@50 and Oracle@K on the pool."""
    groups = {"overall": list(gold)}
    for label in ("implicit", "explicit", "mixed"):
        members = [q for q in gold if kinds.get(q) == label]
        if members:
            groups[label] = members
    out: dict[str, dict] = {}
    for name, members in groups.items():
        row: dict[str, float] = {"queries": len(members)}
        for k in ks:
            row[f"R{k}"] = float(np.mean([recall_at(gold[q], rankings.get(q, ()), k) for q in members]))
        row["CR50"] = row["R50"]
        if pools is not None:
            row["coverage"] = float(np.mean([coverage(gold[q], pools.get(q, ())) for q in members]))
            for k in (10, 50):
                row[f"Oracle{k}"] = float(np.mean([oracle_at(gold[q], pools.get(q, ()), k) for q in members]))
        out[name] = row
    return out


def wlt(gold: Mapping[str, Sequence[str]], a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]],
        k: int = 10, tol: float = 1e-12) -> dict[str, int]:
    win = loss = tie = 0
    for q in gold:
        d = recall_at(gold[q], a.get(q, ()), k) - recall_at(gold[q], b.get(q, ()), k)
        if d > tol:
            win += 1
        elif d < -tol:
            loss += 1
        else:
            tie += 1
    return {"win": win, "loss": loss, "tie": tie}


def paired_bootstrap(gold: Mapping[str, Sequence[str]], a: Mapping[str, Sequence[str]],
                     b: Mapping[str, Sequence[str]], groups: Mapping[str, str], *, k: int = 10,
                     replicates: int = 10000, seed: int = 20260922, queries: Sequence[str] | None = None) -> dict:
    """Source-group paired bootstrap of the query-macro R@K difference (SPEC 14.5).

    Each replicate resamples source groups with replacement, keeps every query
    of a drawn group and macro-averages over the drawn queries.  The point
    estimate is the observed mean difference, not the bootstrap mean.
    """
    members = list(queries) if queries is not None else list(gold)
    if not members:
        return {"queries": 0}
    da = np.array([recall_at(gold[q], a.get(q, ()), k) for q in members])
    db = np.array([recall_at(gold[q], b.get(q, ()), k) for q in members])
    diff = da - db
    by_group: dict[str, list[int]] = {}
    for i, q in enumerate(members):
        by_group.setdefault(groups.get(q, q), []).append(i)
    keys = sorted(by_group)
    index_lists = [np.array(by_group[g], dtype=np.int64) for g in keys]
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates)
    for r in range(replicates):
        picks = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([index_lists[p] for p in picks])
        deltas[r] = float(diff[idx].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"k": k, "queries": len(members), "source_groups": len(keys), "point_estimate": float(diff.mean()),
            "ci95": [float(lo), float(hi)], "bootstrap_mean": float(deltas.mean()), "replicates": replicates,
            "seed": seed, **wlt({q: gold[q] for q in members}, a, b, k)}


# ----------------------------------------------------------------- strict ----


def strict_own(gold: Mapping[str, Sequence[str]], evidence: Mapping[str, Sequence[str]],
               d_ann: Mapping[str, Sequence[str]], d_exact: Mapping[str, Sequence[str]],
               pools: Mapping[str, Mapping[str, Sequence[str]]], final: Mapping[str, Mapping[str, Sequence[str]]],
               *, ks: Sequence[int] = (10, 50)) -> dict:
    """SPEC 14.4 own strict EO: eligible = G \\ (D100_ANN u D100_exact); admitted = eligible ∩ Evidence.

    Returns full target-ID evidence per query plus aggregate retention counts.
    """
    per_query = {}
    eligible_total = admitted_total = 0
    retained = {name: 0 for name in pools}
    tops = {f"{name}@{k}": 0 for name in final for k in ks}
    for q in gold:
        eligible = set(gold[q]) - (set(d_ann.get(q, ())) | set(d_exact.get(q, ())))
        admitted = eligible & set(evidence.get(q, ()))
        eligible_total += len(eligible)
        admitted_total += len(admitted)
        row = {"eligible": sorted(eligible), "admitted": sorted(admitted), "pools": {}, "final": {}}
        for name, pool in pools.items():
            kept = admitted & set(pool.get(q, ()))
            retained[name] += len(kept)
            row["pools"][name] = sorted(kept)
        for name, ranking in final.items():
            for k in ks:
                kept = admitted & set(ranking.get(q, ())[:k])
                tops[f"{name}@{k}"] += len(kept)
                row["final"][f"{name}@{k}"] = sorted(kept)
        per_query[q] = row
    return {"eligible_pairs": eligible_total, "admitted_pairs": admitted_total,
            "retained": retained, "top": tops, "per_query": per_query}


def strict_fixed_cohort(gold: Mapping[str, Sequence[str]], direct_sets: Sequence[Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    """Paired cohort: G minus every compared model's ANN/exact Direct100 finds."""
    out: dict[str, list[str]] = {}
    for q in gold:
        found: set[str] = set()
        for d in direct_sets:
            found |= set(d.get(q, ()))
        rest = sorted(set(gold[q]) - found)
        if rest:
            out[q] = rest
    return out

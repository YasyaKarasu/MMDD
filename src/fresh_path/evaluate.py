"""Evaluation contract for FRESH-PATH v2.1 (SPEC 11, 12).

Builds the required system rows: own retrieval per Student, a fixed Teacher
pool (Raw-QT / T_EDGE-QT / T_QT / T_PATH-f0 / T_PATH-Real / T_PATH-E-swap),
the conditional second-hop probe, strict evidence-only cohorts, grouped
metrics and the source-group bootstrap.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from . import inputs
from .retrieval import _lse
from .score import ObjectBank


def query_recall(gold: Sequence[str], ranking: Sequence[str], k: int) -> tuple[int, int]:
    hits = len(set(ranking[:k]) & set(gold))
    return hits, len(gold)


def macro_recall(gold: Mapping[str, Sequence[str]], rankings: Mapping[str, Sequence[str]], k: int) -> float:
    if not gold:
        raise ValueError("empty gold population")
    total = 0.0
    for q, targets in gold.items():
        hits, population = query_recall(targets, rankings.get(q, ()), k)
        total += hits / population
    return total / len(gold)


def grouped_metrics(gold: Mapping[str, Sequence[str]], rankings: Mapping[str, Sequence[str]],
                    kinds: Mapping[str, str], *, ks: Sequence[int] = (10, 20, 50)) -> dict:
    groups = {"overall": list(gold)}
    for label in ("implicit", "explicit", "mixed", "unknown"):
        members = [q for q in gold if kinds.get(q) == label]
        if members:
            groups[label] = members
    out: dict[str, dict[str, float]] = {}
    for name, members in groups.items():
        sub = {q: gold[q] for q in members}
        row = {}
        for k in ks:
            row[f"R{k}"] = macro_recall(sub, rankings, k)
        row["CR50"] = macro_recall(sub, rankings, 50)
        row["queries"] = len(members)
        out[name] = row
    return out


# ----------------------------------------------------------- teacher rerank ---


def rerank_zero_hop(teacher, bank: ObjectBank, query_id: str, candidates: Sequence[str], device,
                    *, batch: int = 32) -> dict[str, float]:
    from .train_teacher import pair_logits

    if not candidates:
        return {}
    scores = pair_logits(teacher, bank, query_id, list(candidates), device, batch=batch)
    return {t: float(s) for t, s in zip(candidates, scores)}


def rerank_path(teacher, bank: ObjectBank, query_id: str, candidates: Sequence[str],
                paths: Mapping[str, Sequence], device, *, batch: int = 32,
                swapped: Mapping[str, str] | None = None) -> dict[str, float]:
    """T_PATH-Real: LSE(zero hop, QET over the retained natural paths)."""
    from .train_teacher import pair_logits, triplet_pairs

    if not candidates:
        return {}
    zero = pair_logits(teacher, bank, query_id, list(candidates), device, batch=batch)
    flat: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for t in candidates:
        slots = list(paths.get(t, []))
        counts[t] = len(slots)
        for slot in slots:
            eid = slot[0]
            if swapped is not None:
                eid = swapped.get(eid, eid)
            flat.append((eid, t))
    qet = triplet_pairs(teacher, bank, query_id, flat, device, batch=batch)
    cursor = 0
    out: dict[str, float] = {}
    for i, t in enumerate(candidates):
        n = counts[t]
        if n:
            out[t] = float(torch.logsumexp(torch.cat([zero[i].reshape(1), qet[cursor : cursor + n]]), 0))
            cursor += n
        else:
            out[t] = float(zero[i])
    return out


def e_swap_map(content_key: Mapping[str, str], modality: Mapping[str, str]) -> dict[str, str]:
    """Fixed non-identity cyclic permutation inside each modality by content key."""
    by_modality: dict[str, dict[str, str]] = {}
    for eid, key in content_key.items():
        by_modality.setdefault(modality.get(eid, "text"), {})[key] = min(
            by_modality.setdefault(modality.get(eid, "text"), {}).get(key, eid), eid,
            key=lambda x: x.encode("utf-8"),
        )
    out: dict[str, str] = {}
    for mod, keys in by_modality.items():
        ordered = sorted(keys.values(), key=lambda x: x.encode("utf-8"))
        if len(ordered) < 2:
            continue
        donors = ordered[1:] + ordered[:1]
        for src, dst in zip(ordered, donors):
            out[src] = dst
    return out


def apply_swap(paths: Mapping[str, Sequence], swap: Mapping[str, str]) -> dict[str, list[list[object]]]:
    return {t: [[swap.get(slot[0], slot[0]), slot[1]] for slot in slots] for t, slots in paths.items()}


# ------------------------------------------------------------------ probe -----


def probe_et(model, bank: ObjectBank, legal: Sequence[str], pairs: Sequence[tuple[str, str, Sequence[str]]],
             device, *, ks: Sequence[int] = (10, 20), modality: Mapping[str, str] | None = None) -> dict:
    """SPEC 12.3: rank the whole static target space for fixed dev (q, e) pairs."""
    from .train_student import adapter_vector, target_keys

    keys = target_keys(model, bank, list(legal), device)
    rows = []
    pairs_by_query: dict[str, list[str]] = {}
    for qid, eid, positive in pairs:
        if not positive:
            continue
        v = adapter_vector(model, bank, qid, eid, device)
        scores = (v @ keys.T).float().cpu()
        order = sorted(range(len(legal)), key=lambda i: (-float(scores[i]), legal[i].encode("utf-8")))
        rank_of = {t: i + 1 for i, t in enumerate(legal[i] for i in order)}
        ranks = [rank_of[t] for t in positive if t in rank_of]
        if not ranks:
            continue
        top1 = legal[order[0]]
        pairs_by_query.setdefault(top1, []).append(qid)
        rows.append({"query_id": qid, "evidence_id": eid, "ranks": ranks, "top1": top1})
    if not rows:
        return {"pairs": 0}
    result: dict[str, object] = {"pairs": len(rows)}
    for k in ks:
        result[f"R{k}"] = float(np.mean([sum(1 for r in row["ranks"] if r <= k) / len(row["ranks"]) for row in rows]))
    result["MRR"] = float(np.mean([np.mean([1.0 / r for r in row["ranks"]]) for row in rows]))
    result["median_positive_rank"] = float(np.median([r for row in rows for r in row["ranks"]]))
    most_common = max(pairs_by_query, key=lambda t: len(pairs_by_query[t]))
    result["common_top1"] = most_common
    result["common_top1_share"] = len(pairs_by_query[most_common]) / len(rows)
    if modality:
        for label in ("text", "image"):
            subset = [row for row in rows if modality.get(row["evidence_id"]) == label]
            if subset:
                result[f"R10_{label}"] = float(np.mean(
                    [sum(1 for r in row["ranks"] if r <= 10) / len(row["ranks"]) for row in subset]))
                result[f"pairs_{label}"] = len(subset)
    return result


# ------------------------------------------------------------------ strict ----


def strict_eo(gold: Sequence[str], evidence: Sequence[str], d_ann: Sequence[str], d_exact: Sequence[str]) -> set[str]:
    return set(gold) & set(evidence) - (set(d_ann) | set(d_exact))


def strict_report(queue: Mapping[str, Sequence[str]], evidence: Mapping[str, Sequence[str]],
                  d_ann: Mapping[str, Sequence[str]], d_exact: Mapping[str, Sequence[str]],
                  final: Mapping[str, Sequence[str]], *, ks: Sequence[int] = (10, 50)) -> dict:
    total = 0
    entered_u = 0                      # was aliased to `top`: dict += int -> TypeError
    top = {k: 0 for k in ks}
    for q, gold in queue.items():
        z = strict_eo(gold, evidence.get(q, ()), d_ann.get(q, ()), d_exact.get(q, ()))
        total += len(z)
        if z & set(evidence.get(q, ())):
            entered_u += 1
        for k in ks:
            if z & set(final.get(q, ())[:k]):
                top[k] += 1
    return {"strict_pairs": total, "entered_evidence": entered_u,
            "top_hits": {f"top{k}": v for k, v in top.items()}}


# -------------------------------------------------------------- statistics ----


def wlt(gold: Mapping[str, Sequence[str]], a: Mapping[str, Sequence[str]], b: Mapping[str, Sequence[str]],
        k: int = 10, tol: float = 1e-12) -> tuple[int, int, int]:
    win = loss = tie = 0
    for q, targets in gold.items():
        pa = len(set(a.get(q, ())[:k]) & set(targets)) / len(targets)
        pb = len(set(b.get(q, ())[:k]) & set(targets)) / len(targets)
        if pa - pb > tol:
            win += 1
        elif pb - pa > tol:
            loss += 1
        else:
            tie += 1
    return win, loss, tie


def bootstrap_delta(gold: Mapping[str, Sequence[str]], a: Mapping[str, Sequence[str]],
                    b: Mapping[str, Sequence[str]], groups: Mapping[str, str], k: int, *,
                    replicates: int = 10000, seed: int = 20260920) -> dict:
    by_group: dict[str, list[str]] = {}
    for q in gold:
        by_group.setdefault(groups.get(q, q), []).append(q)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        picks = rng.integers(0, len(keys), size=len(keys))
        members = [q for i in picks for q in by_group[keys[i]]]
        da = np.mean([len(set(a.get(q, ())[:k]) & set(gold[q])) / len(gold[q]) for q in members])
        db = np.mean([len(set(b.get(q, ())[:k]) & set(gold[q])) / len(gold[q]) for q in members])
        deltas.append(da - db)
    arr = np.array(deltas)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)), "replicates": replicates}

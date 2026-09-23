"""Dev/test evaluation (SPEC 14): system table, Teacher reranks on P0-P3 and Full-U,
E-swap, MatchedDirectM, strict cohorts, bootstrap, latency.

Teacher scores are computed once per (q,t) / (q,e,t) on the largest pool
(U) and restricted to the fixed C150 policy afterwards, as SPEC 14.2 allows;
the formal C150 latency is measured separately with fresh forwards.
"""
from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from . import metrics
from .data import Labels, utf8_sorted
from .io import sha256_file, write_json
from .pools import PoolRecord
from .teacher import pair_logits, triplet_logits


def rank_by(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda t: (-scores[t], t.encode("utf-8")))


class TeacherScorer:
    """Memoised frozen-Teacher scoring; every score is a real forward once."""

    def __init__(self, model, bank, device: str, *, name: str, pair_chunk: int = 256, triplet_chunk: int = 128) -> None:
        self.model = model.eval()
        self.bank = bank
        self.device = device
        self.name = name
        self.pair_chunk = pair_chunk
        self.triplet_chunk = triplet_chunk
        self.pairs: dict[tuple[str, str], float] = {}
        self.triplets: dict[tuple[str, str, str], float] = {}
        self.forwards = {"pairs": 0, "triplets": 0}

    @torch.no_grad()
    def f0(self, qid: str, targets: Sequence[str]) -> dict[str, float]:
        missing = [t for t in targets if (qid, t) not in self.pairs]
        if missing:
            values = pair_logits(self.model, self.bank, qid, missing, self.device, chunk=self.pair_chunk)
            for t, v in zip(missing, values.tolist()):
                self.pairs[(qid, t)] = v
            self.forwards["pairs"] += len(missing)
        return {t: self.pairs[(qid, t)] for t in targets}

    @torch.no_grad()
    def qet(self, qid: str, slots: Sequence[tuple[str, str]]) -> dict[tuple[str, str], float]:
        missing = [(e, t) for e, t in dict.fromkeys(slots) if (qid, e, t) not in self.triplets]
        if missing:
            values = triplet_logits(self.model, self.bank, qid, missing, self.device, chunk=self.triplet_chunk)
            for (e, t), v in zip(missing, values.tolist()):
                self.triplets[(qid, e, t)] = v
            self.forwards["triplets"] += len(missing)
        return {(e, t): self.triplets[(qid, e, t)] for e, t in slots}


def swap_map(evidence_ids: Sequence[str], content_key: Mapping[str, str],
             modality: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Fixed donor rings; formal evaluation keeps text and image in separate rings."""
    if modality is not None:
        unknown = {e for e in evidence_ids if modality.get(e) not in ("text", "image")}
        if unknown:
            raise ValueError(f"swap donors require text/image modality for {len(unknown)} evidence IDs")
        result = {}
        for kind in ("text", "image"):
            ring = [e for e in evidence_ids if modality[e] == kind]
            result.update(swap_map(ring, content_key))
        return result
    ring = utf8_sorted(set(evidence_ids))
    out: dict[str, str | None] = {}
    for i, e in enumerate(ring):
        donor = None
        for step in range(1, len(ring)):
            candidate = ring[(i + step) % len(ring)]
            if content_key.get(candidate, candidate) != content_key.get(e, e):
                donor = candidate
                break
        out[e] = donor
    return out


def lse(values: Sequence[float]) -> float:
    m = max(values)
    return float(m + np.log(np.sum(np.exp(np.asarray(values) - m))))


def teacher_readouts(scorer: TeacherScorer, pools: Mapping[str, PoolRecord], *, generator_id: str,
                     with_paths: bool, swap: Mapping[str, str | None] | None, queries: Sequence[str],
                     exclude_policies: Sequence[str] = (), log=print) -> dict:
    """Score every U member once, then materialize the fixed P0-P3 policies."""
    rankings: dict[str, dict[str, list[str]]] = {}
    logits: dict[str, dict] = {}
    swap_slots = swap_replaced = 0
    fullu_extra_slots = fullu_extra_replaced = 0
    started = time.time()
    for n, qid in enumerate(queries, 1):
        pool = pools[qid]
        pool.validate(expected_generator=generator_id, query_id=qid)
        universe = list(pool.U)
        f0 = scorer.f0(qid, universe)
        record: dict = {"f0": f0, "qt_scores_all_U": dict(f0), "origin": dict(pool.origin)}
        variants = {"f0": f0}
        if with_paths:
            slots = [(e, t) for t in universe for e in pool.path_bag(t)]
            real = scorer.qet(qid, slots)
            real_scores = {}
            structural = {}
            for t in universe:
                bag = pool.path_bag(t)
                real_scores[t] = lse([f0[t], *[real[(e, t)] for e in bag]]) if bag else f0[t]
                structural[t] = f0[t] + math.log1p(len(bag))
            variants["Real"] = real_scores
            variants["Struct"] = structural
            record["paths"] = {t: [[e, real[(e, t)]] for e in pool.path_bag(t)] for t in universe if pool.path_bag(t)}
            if swap is not None:
                swapped_slots, mapping = [], {}
                for e, t in slots:
                    donor = swap.get(e)
                    if donor is not None:
                        swapped_slots.append((donor, t))
                        mapping[(e, t)] = donor
                    else:
                        swapped_slots.append((e, t))
                        mapping[(e, t)] = e
                sw = scorer.qet(qid, swapped_slots)
                swap_scores = {}
                for t in universe:
                    bag = pool.path_bag(t)
                    swap_scores[t] = lse([f0[t], *[sw[(mapping[(e, t)], t)] for e in bag]]) if bag else f0[t]
                for e, t in slots:
                    if t in pool.C150:
                        swap_slots += 1
                        swap_replaced += mapping[(e, t)] != e
                    else:
                        fullu_extra_slots += 1
                        fullu_extra_replaced += mapping[(e, t)] != e
                variants["Swap"] = swap_scores
                record["swap_paths"] = {t: [[mapping[(e, t)], sw[(mapping[(e, t)], t)]] for e in pool.path_bag(t)]
                                        for t in universe if pool.path_bag(t)}
        for variant, scores in variants.items():
            policies = pool.policies or {"P3_K50_C150_QTALL": pool.C150}
            for policy_id, ids in policies.items():
                if policy_id in exclude_policies:
                    continue
                rankings.setdefault(f"{variant}|{policy_id}", {})[qid] = rank_by({t: scores[t] for t in ids})
            # A compact alias is useful to old report readers; its value is
            # exactly P3 and never a 100-item budget.
            rankings.setdefault(f"{variant}|C150", {})[qid] = rank_by({t: scores[t] for t in pool.C150})
            d100 = {t: scores[t] for t in pool.direct_ids}
            rankings.setdefault(f"{variant}|FullU", {})[qid] = rank_by(scores)
            rankings.setdefault(f"{variant}|D100", {})[qid] = rank_by(d100)
        logits[qid] = record
        if n % 200 == 0:
            log({"event": "teacher_readout", "teacher": scorer.name, "generator": generator_id, "done": n,
                 "total": len(queries), "elapsed": round(time.time() - started, 1), **scorer.forwards})
    return {"rankings": rankings, "logits": logits,
            "swap": {"slots": swap_slots, "replaced": swap_replaced,
                     "fullu_extra_slots": fullu_extra_slots, "fullu_extra_replaced": fullu_extra_replaced,
                     "effective": (swap_replaced / swap_slots) if swap_slots else None}}


def matched_direct_m(scorer: TeacherScorer, retriever, pools: Mapping[str, PoolRecord], *, queries: Sequence[str]) -> dict:
    """SPEC 14.2 MatchedDirectM: own Direct top M=|U_q| under the same T_QT vs U."""
    rankings: dict[str, list[str]] = {}
    sizes = []
    efs = []
    cost = {"teacher_pairs": 0}
    for qid in queries:
        pool = pools[qid]
        m = len(pool.U)
        hits, ef = retriever.direct_m(qid, m)
        ids = [t for t, _ in hits]
        before = scorer.forwards["pairs"]
        scores = scorer.f0(qid, ids)
        cost["teacher_pairs"] += scorer.forwards["pairs"] - before
        rankings[qid] = rank_by(scores)
        sizes.append(m)
        efs.append(ef)
    return {"rankings": rankings, "mean_M": float(np.mean(sizes)), "max_M": max(sizes), "ef_used": sorted(set(efs)), **cost}


def save_rankings(path: Path, rankings: Mapping[str, Mapping[str, Sequence[str]]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({k: {q: list(v) for q, v in rows.items()} for k, rows in rankings.items()}, handle)
    return sha256_file(path)


def save_json_gz(path: Path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return sha256_file(path)


def system_table(gold: Mapping[str, Sequence[str]], kinds: Mapping[str, str],
                 rankings: Mapping[str, Mapping[str, Sequence[str]]],
                 pools_for: Mapping[str, Mapping[str, Sequence[str]]]) -> dict[str, dict]:
    table = {}
    for system, rows in rankings.items():
        table[system] = metrics.grouped(gold, kinds, rows, pools=pools_for.get(system))
    return table


@torch.no_grad()
def measure_latency(scorer_factory: Callable[[], TeacherScorer], pools: Mapping[str, PoolRecord], *,
                    queries: Sequence[str], with_paths: bool, device: str) -> dict:
    """Exclusive-card per-query C150 latency with fresh (uncached) forwards."""
    times = []
    for qid in queries:
        scorer = scorer_factory()
        pool = pools[qid]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        f0 = scorer.f0(qid, pool.C150)
        if with_paths:
            slots = [(e, t) for t in pool.C150 for e in pool.path_bag(t)]
            scorer.qet(qid, slots)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - started)
    arr = np.array(times)
    return {"queries": len(times), "p50_ms": float(np.percentile(arr, 50) * 1000),
            "p95_ms": float(np.percentile(arr, 95) * 1000), "mean_ms": float(arr.mean() * 1000),
            "with_paths": with_paths, "mode": "exclusive_gpu_fresh_forward_C150"}

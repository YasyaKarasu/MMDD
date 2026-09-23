"""Candidate pools: D1 natural-path retention, RRF admission and PoolRecord.

Shared by the raw (exact ``z``) generator and every Student's own HNSW
retriever, so the budget rules of SPEC 5.2-5.5 are implemented once.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .data import utf8_sorted

DIRECT_K = 100
FIRST_HOP_K = 20
SECOND_HOP_K = 50
TOP_L = 20
EVIDENCE_BUDGET = 4
RRF_K = 60
CANDIDATE_BUDGET = 150
# Kept as a read-only compatibility alias for old result readers.  Production
# construction uses ``CANDIDATE_BUDGET``/``C150`` and never treats 100 as the
# candidate budget.
C100 = CANDIDATE_BUDGET


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def lse(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


@dataclass(frozen=True)
class PathEntry:
    evidence_id: str
    first_score: float
    second_score: float
    modality: str

    @property
    def raw_path_score(self) -> float:
        return self.first_score + self.second_score


@dataclass
class PoolRecord:
    """SPEC 5.5: every pool carries its generator identity and full paths."""

    split: str
    query_id: str
    generator_id: str
    model_sha: str
    target_index_sha: str
    evidence_index_sha: str
    direct: list[tuple[str, float]]
    direct_exact: list[str]
    first_hop: dict[str, list[tuple[str, float]]]
    evidence: list[tuple[str, float]]
    U: list[str]
    C150: list[str]
    pre_paths: dict[str, list[PathEntry]]
    retained_paths: dict[str, list[str]]
    retained_coverage: dict[str, float]
    qt_scores_all_U: dict[str, float] = field(default_factory=dict)
    qt_rank_all_U: dict[str, int] = field(default_factory=dict)
    e_rank: dict[str, int] = field(default_factory=dict)
    admission_scores: dict[str, float] = field(default_factory=dict)
    origin: dict[str, str] = field(default_factory=dict)
    candidate_budget: int = CANDIDATE_BUDGET
    policies: dict[str, list[str]] = field(default_factory=dict)
    retrieval_meta: dict = field(default_factory=dict)

    @property
    def C100(self) -> list[str]:
        """Deprecated reader alias; v3.1's primary pool is C150."""
        return self.C150

    @property
    def direct_ids(self) -> list[str]:
        return [t for t, _ in self.direct]

    @property
    def evidence_ids(self) -> list[str]:
        return [t for t, _ in self.evidence]

    def path_bag(self, target: str) -> list[str]:
        return list(self.retained_paths.get(target, []))

    def arrivals(self, target: str) -> list[str]:
        return utf8_sorted({p.evidence_id for p in self.pre_paths.get(target, [])})

    def validate(self, *, expected_generator: str, expected_model_sha: str | None = None,
                 query_id: str | None = None) -> None:
        if self.generator_id != expected_generator:
            raise ValueError(f"foreign generator paths: {self.generator_id!r} != {expected_generator!r}")
        if expected_model_sha is not None and self.model_sha != expected_model_sha:
            raise ValueError("foreign checkpoint paths")
        if query_id is not None and self.query_id != query_id:
            raise ValueError("path pool query does not match")
        if not self.model_sha or not self.target_index_sha:
            raise ValueError("pool identity missing")
        if len(set(self.C150)) != len(self.C150) or len(set(self.U)) != len(self.U):
            raise ValueError("duplicate candidates")
        if len(self.C150) > self.candidate_budget:
            raise ValueError("candidate budget exceeded")
        u = set(self.U)
        if self.origin and set(self.origin) != u:
            raise ValueError("candidate origin does not cover exactly U")
        if any(t not in u for t in self.retained_paths):
            raise ValueError("path pool contains foreign target")
        for bag in self.retained_paths.values():
            if len(bag) != len(set(bag)):
                raise ValueError("path pool requires canonical evidence dedup")

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["pre_paths"] = {t: [asdict(p) for p in v] for t, v in self.pre_paths.items()}
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "PoolRecord":
        payload = dict(payload)
        # Legacy serialized C100; accepting it is useful for read-only migration,
        # but such a record is never admitted to a v3.1 training stage.
        if "C150" not in payload and "C100" in payload:
            payload["C150"] = payload.pop("C100")
        # A raw/train process started before the final v3.1 schema patch may
        # have serialized the pool without the new lineage fields.  Migrate
        # only that current-run record shape; callers still validate the run
        # generator/model identity before using it for training.
        direct_ids = [str(t) for t, _ in payload.get("direct", ())]
        evidence_ids = {str(t) for t, _ in payload.get("evidence", ())}
        u_ids = [str(t) for t in payload.get("U", ())]
        if "origin" not in payload:
            payload["origin"] = {t: ("D100" if t in direct_ids else "E_channel") for t in u_ids}
        payload.setdefault("candidate_budget", CANDIDATE_BUDGET)
        payload.setdefault("qt_scores_all_U", {})
        payload.setdefault("qt_rank_all_U", {})
        payload.setdefault("e_rank", {})
        payload.setdefault("admission_scores", {})
        payload.setdefault("policies", {})
        payload["pre_paths"] = {t: [PathEntry(**p) for p in v] for t, v in payload["pre_paths"].items()}
        payload["direct"] = [tuple(x) for x in payload["direct"]]
        payload["evidence"] = [tuple(x) for x in payload["evidence"]]
        payload["first_hop"] = {k: [tuple(x) for x in v] for k, v in payload["first_hop"].items()}
        return cls(**payload)


# ------------------------------------------------------------- D1 retention ---


def row_support(rows: np.ndarray, z_e: np.ndarray) -> np.ndarray:
    """b_ie = clip((row_i . z_e + 1) / 2, 0, 1); rows (n,d), z_e (d,) or (m,d)."""
    affinity = rows @ (z_e.T if z_e.ndim == 2 else z_e)
    return np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)


def d1_retain(entries: Sequence[PathEntry], support: Mapping[str, np.ndarray], *,
              content_key: Mapping[str, str] | None = None, top_l: int = TOP_L,
              budget: int = EVIDENCE_BUDGET) -> tuple[list[str], float]:
    """SPEC 5.3 greedy soft row coverage; returns (selected evidence, C(S))."""
    if not entries:
        return [], 0.0
    best: dict[str, PathEntry] = {}
    for entry in entries:
        key = content_key.get(entry.evidence_id, entry.evidence_id) if content_key else entry.evidence_id
        current = best.get(key)
        if current is None or (-entry.raw_path_score, entry.evidence_id.encode("utf-8")) < (
                -current.raw_path_score, current.evidence_id.encode("utf-8")):
            best[key] = entry
    candidates = sorted(best.values(), key=lambda p: (-p.raw_path_score, p.evidence_id.encode("utf-8")))[:top_l]
    if not candidates:
        return [], 0.0
    n_rows = len(support[candidates[0].evidence_id])
    if n_rows == 0:
        raise ValueError("query rows required for D1 retention")
    current = np.zeros(n_rows, dtype=np.float64)
    selected: list[str] = []
    remaining = list(candidates)
    for _ in range(min(budget, len(remaining))):
        options = []
        for entry in remaining:
            a = sigmoid(entry.raw_path_score)
            updated = np.maximum(current, a * np.asarray(support[entry.evidence_id], dtype=np.float64))
            gain = float(updated.mean() - current.mean())
            options.append((gain, a, entry.evidence_id, updated))
        gain, _a, evidence_id, updated = min(options, key=lambda o: (-o[0], -o[1], o[2].encode("utf-8")))
        if gain <= 0.0:
            break
        selected.append(evidence_id)
        current = updated
        remaining = [e for e in remaining if e.evidence_id != evidence_id]
    return selected, float(current.mean())


def equal_rrf(direct: Sequence[str], evidence: Sequence[str], *, k: int = RRF_K,
              budget: int = CANDIDATE_BUDGET) -> list[str]:
    score: dict[str, float] = {}
    for channel in (direct, evidence):
        if len(set(channel)) != len(channel):
            raise ValueError("RRF channels must not contain duplicates")
        for rank, item in enumerate(channel, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda t: (-score[t], t.encode("utf-8")))[:budget]


def rank_within_u(scores: Mapping[str, float]) -> dict[str, int]:
    """Return 1-based QT ranks over exactly U, with UTF-8 tie breaking."""
    ordered = sorted(scores, key=lambda t: (-float(scores[t]), t.encode("utf-8")))
    return {target: rank for rank, target in enumerate(ordered, 1)}


def admit_policies(*, direct: Sequence[str], evidence: Sequence[str], qt_scores: Mapping[str, float],
                   budget: int = CANDIDATE_BUDGET) -> tuple[list[str], dict[str, list[str]], dict[str, float], dict[str, int], dict[str, int]]:
    """Build the fixed P0/P1/P2/P3 policies.

    P3 is the only training/selection policy: QT is scored for every member of
    U and its rank is fused with the D1 evidence rank by equal RRF (k=60).
    The other policies are read-only controls and deliberately re-use no P3
    QT rank or candidate list.
    """
    direct = list(dict.fromkeys(direct))
    evidence = list(dict.fromkeys(evidence))
    u = list(dict.fromkeys([*direct, *evidence]))
    missing = set(u) - set(qt_scores)
    if missing:
        raise ValueError(f"QTALL score missing for U members: {sorted(missing)!r}")
    qt_rank = rank_within_u({target: float(qt_scores[target]) for target in u})
    e_rank = {target: rank for rank, target in enumerate(evidence, 1)}

    p0 = equal_rrf(direct, evidence, budget=100)
    p1 = equal_rrf(direct, evidence, budget=budget)
    p2 = equal_rrf(direct, evidence, budget=budget)
    admission = {target: 1.0 / (RRF_K + qt_rank[target])
                 + (1.0 / (RRF_K + e_rank[target]) if target in e_rank else 0.0)
                 for target in u}
    p3 = sorted(u, key=lambda t: (-admission[t], t.encode("utf-8")))[:budget]
    return p3, {
        "P0_K20_C100_OLD": p0,
        "P1_K20_C150_OLD": p1,
        "P2_K50_C150_OLD": p2,
        "P3_K50_C150_QTALL": p3,
    }, admission, qt_rank, e_rank


def assemble_pool(
    *,
    split: str,
    query_id: str,
    generator_id: str,
    model_sha: str,
    target_index_sha: str,
    evidence_index_sha: str,
    direct: list[tuple[str, float]],
    direct_exact: list[str],
    first_hop: dict[str, list[tuple[str, float]]],
    second_hop: Callable[[str, str], list[tuple[str, float]]],
    query_rows: np.ndarray,
    evidence_z: Callable[[Sequence[str]], np.ndarray],
    content_key: Mapping[str, str] | None,
    retrieval_meta: dict | None = None,
    qt_scores: Mapping[str, float] | None = None,
    candidate_budget: int = CANDIDATE_BUDGET,
) -> PoolRecord:
    """Build one PoolRecord from Direct hits and first-hop hits (SPEC 5.2-5.4)."""
    arrivals: dict[str, list[PathEntry]] = {}
    evidence_ids: list[str] = []
    for modality in ("text", "image"):
        for evidence_id, first_score in first_hop.get(modality, []):
            evidence_ids.append(evidence_id)
            for target, second_score in second_hop(evidence_id, modality):
                arrivals.setdefault(target, []).append(
                    PathEntry(evidence_id, float(first_score), float(second_score), modality))
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("first-hop evidence must be canonical and unique")
    support: dict[str, np.ndarray] = {}
    if evidence_ids:
        ez = evidence_z(evidence_ids)
        matrix = row_support(query_rows, ez)  # (rows, n_e)
        support = {e: matrix[:, i] for i, e in enumerate(evidence_ids)}
    retained: dict[str, list[str]] = {}
    coverage: dict[str, float] = {}
    for target in utf8_sorted(arrivals):
        arrivals[target].sort(key=lambda p: (-p.raw_path_score, p.evidence_id.encode("utf-8")))
        selected, value = d1_retain(arrivals[target], support, content_key=content_key)
        if selected:
            retained[target] = selected
            coverage[target] = value
    evidence_ranked = sorted(coverage.items(), key=lambda p: (-p[1], p[0].encode("utf-8")))
    direct_ids = [t for t, _ in direct]
    evidence_targets = [t for t, _ in evidence_ranked]
    U = utf8_sorted(set(direct_ids) | set(evidence_targets))
    if qt_scores is None:
        # Callers must provide a score for every U item.  The fallback is only
        # valid when the direct scores already cover all U (e.g. an empty
        # evidence channel); E-only candidates must never receive a fake vote.
        direct_score = dict(direct)
        if set(U) - set(direct_score):
            raise ValueError("qt_scores is required for evidence-only U members")
        qt_scores = {t: direct_score[t] for t in U}
    p3, policies, admission, qt_rank, e_rank = admit_policies(
        direct=direct_ids, evidence=evidence_targets, qt_scores=qt_scores, budget=candidate_budget)
    origin = {target: ("D100" if target in set(direct_ids) else "E_channel") for target in U}
    return PoolRecord(
        split=split, query_id=query_id, generator_id=generator_id, model_sha=model_sha,
        target_index_sha=target_index_sha, evidence_index_sha=evidence_index_sha,
        direct=[(t, float(s)) for t, s in direct], direct_exact=list(direct_exact),
        first_hop={m: [(e, float(s)) for e, s in first_hop.get(m, [])] for m in ("text", "image")},
        evidence=[(t, float(s)) for t, s in evidence_ranked],
        U=U,
        C150=p3,
        pre_paths=arrivals, retained_paths=retained, retained_coverage=coverage,
        qt_scores_all_U={t: float(qt_scores[t]) for t in U}, qt_rank_all_U=qt_rank,
        e_rank=e_rank, admission_scores=admission, candidate_budget=candidate_budget,
        origin=origin,
        policies=policies,
        retrieval_meta=dict(retrieval_meta or {}),
    )


def coverage_at(gold: Sequence[str], candidates: Sequence[str]) -> float:
    g = set(gold)
    return len(g & set(candidates)) / len(g) if g else 0.0


def pool_summary(pools: Mapping[str, PoolRecord], gold: Mapping[str, Sequence[str]]) -> dict:
    keys = ("D100", "E", "U", "C150")
    total = {k: 0.0 for k in keys}
    sizes = {k: 0 for k in keys}
    n = 0
    for qid, pool in pools.items():
        g = gold.get(qid)
        if not g:
            continue
        n += 1
        total["D100"] += coverage_at(g, pool.direct_ids)
        total["E"] += coverage_at(g, pool.evidence_ids)
        total["U"] += coverage_at(g, pool.U)
        total["C150"] += coverage_at(g, pool.C150)
        sizes["D100"] += len(pool.direct)
        sizes["E"] += len(pool.evidence)
        sizes["U"] += len(pool.U)
        sizes["C150"] += len(pool.C150)
    if n == 0:
        return {"queries": 0}
    return {"queries": n, "coverage": {k: total[k] / n for k in keys}, "mean_size": {k: sizes[k] / n for k in keys}}

"""HNSW indexing, exact search, D1 retention, and P3 admission for CLEAN-QET v4.0."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import hnswlib
import numpy as np
import torch
from torch import Tensor

from .config import Paths
from .data import utf8_sorted
from .features import ZStore
from .labels import Labels

CANDIDATE_BUDGET = 150
DIRECT_K = 100
FIRST_HOP_K = 20
SECOND_HOP_K = 50
EVIDENCE_BUDGET = 4
TOP_L = 16
RRF_K = 60


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def stable_topk(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


@dataclass(frozen=True)
class PathEntry:
    evidence_id: str
    modality: str
    first_score: float
    second_score: float

    @property
    def raw_path_score(self) -> float:
        return self.first_score + self.second_score


@dataclass
class PoolRecord:
    split: str
    query_id: str
    generator_id: str
    direct: list[tuple[str, float]]
    direct_exact: list[str]
    first_hop: dict[str, list[tuple[str, float]]]
    pre_paths: dict[str, list[PathEntry]]
    retained_paths: dict[str, list[str]]
    retained_coverage: dict[str, float]
    U: list[str]
    C150: list[str]
    qt_scores_all_U: dict[str, float]
    admission_scores: dict[str, float]
    candidate_budget: int = CANDIDATE_BUDGET


def row_support(rows: np.ndarray, z_e: np.ndarray) -> np.ndarray:
    """b_ie = clip((row_i . z_e + 1) / 2, 0, 1); rows (n,d), z_e (d,) or (m,d)."""
    affinity = rows @ (z_e.T if z_e.ndim == 2 else z_e)
    return np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)


def d1_retain(
    entries: Sequence[PathEntry],
    support: Mapping[str, np.ndarray],
    *,
    content_key: Optional[Mapping[str, str]] = None,
    top_l: int = TOP_L,
    budget: int = EVIDENCE_BUDGET,
) -> tuple[list[str], float]:
    """SPEC 8.1 & v3.1 audited D1 greedy soft row coverage."""
    if not entries:
        return [], 0.0
    best: dict[str, PathEntry] = {}
    for entry in entries:
        key = content_key.get(entry.evidence_id, entry.evidence_id) if content_key else entry.evidence_id
        current = best.get(key)
        if current is None or (-entry.raw_path_score, entry.evidence_id.encode("utf-8")) < (
            -current.raw_path_score,
            current.evidence_id.encode("utf-8"),
        ):
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


def p3_admission(
    qt_scores: dict[str, float],
    evidence_order: Sequence[str],
    budget: int = CANDIDATE_BUDGET,
    constant: int = RRF_K,
) -> tuple[list[str], dict[str, float]]:
    """P3 admission contract from reference_contracts.py & SPEC 8.2."""
    if budget <= 0 or constant < 0:
        raise ValueError("invalid admission parameters")
    if len(evidence_order) != len(set(evidence_order)):
        raise ValueError("duplicate evidence-channel target")
    if any(t not in qt_scores for t in evidence_order):
        raise ValueError("all evidence targets need real QT scores")

    qt_order = sorted(
        qt_scores,
        key=lambda t: (-qt_scores[t], t.encode("utf-8")),
    )
    q_rank = {t: i + 1 for i, t in enumerate(qt_order)}
    e_rank = {t: i + 1 for i, t in enumerate(evidence_order)}

    score = {
        t: 1.0 / (constant + q_rank[t])
        + (1.0 / (constant + e_rank[t]) if t in e_rank else 0.0)
        for t in qt_order
    }
    admitted = sorted(
        qt_order,
        key=lambda t: (-score[t], t.encode("utf-8")),
    )[:budget]
    return admitted, score


class HNSWIndex:
    """HNSW index adhering to SPEC 8.3."""

    def __init__(
        self,
        vectors: np.ndarray,  # (N, dim) float32
        ids: Sequence[str],
        dim: int = 1024,
        m: int = 32,
        ef_construction: int = 200,
        ef_search_floor: int = 256,
        seed: int = 13,
    ) -> None:
        self.dim = dim
        self.ids = list(ids)
        self.ef_search_floor = ef_search_floor
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.index.init_index(max_elements=len(ids), ef_construction=ef_construction, M=m, random_seed=seed)
        self.index.set_num_threads(1)
        # Insertion in UTF-8 object ID order
        sorted_indices = sorted(range(len(ids)), key=lambda i: ids[i].encode("utf-8"))
        sorted_vectors = vectors[sorted_indices]
        self.index.add_items(sorted_vectors, np.asarray(sorted_indices, dtype=np.int64))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        self.index.set_ef(max(self.ef_search_floor, k))
        labels, distances = self.index.knn_query(query_vector.reshape(1, -1), k=k)
        # hnswlib inner product space returns 1 - inner_product
        scores = 1.0 - distances[0]
        hits = [(self.ids[int(lbl)], float(scores[i])) for i, lbl in enumerate(labels[0])]
        # Sort by score desc, tie break UTF-8 asc
        return sorted(hits, key=lambda x: (-x[1], x[0].encode("utf-8")))

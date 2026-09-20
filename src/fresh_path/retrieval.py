"""This run's own HNSW retrieval, retention and admission (SPEC 5, 11).

Indexes are built only from the calling model's own final target vectors.
``ef_search`` follows the shared depth rule; results are re-scored with the
exact inner product before any ranking is reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .candidates import utf8_sorted

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_SEED = 20260920
HNSW_THREADS = 1


@dataclass
class Index:
    index: object
    ids: list[str]
    vectors: np.ndarray

    def search(self, query: np.ndarray, k: int, *, ef_min: int = 200) -> list[tuple[str, float]]:
        import hnswlib

        k = min(k, len(self.ids))
        if k == 0:
            return []
        ef = max(ef_min, k)
        target = k
        while True:
            self.index.set_ef(ef)
            labels, distances = self.index.knn_query(query.reshape(1, -1).astype(np.float32), k=target)
            got = labels[0]
            if len(got) >= min(k, len(self.ids)):
                break
            if target >= len(self.ids):
                break
            target = min(len(self.ids), max(target * 2, target + 32))
            ef = max(ef_min, target)
        ids = [self.ids[int(i)] for i in labels[0]]
        # restore the inner product, then re-score exactly and tie-break by id
        scores = [float(query.astype(np.float32) @ self.vectors[int(i)]) for i in labels[0]]
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i].encode("utf-8")))
        return [(ids[i], scores[i]) for i in order[:k]]


def build_index(vectors: np.ndarray, ids: Sequence[str], *, ef_construction: int = HNSW_EF_CONSTRUCTION) -> Index:
    import hnswlib

    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    index = hnswlib.Index(space="ip", dim=vectors.shape[1])
    index.init_index(max_elements=len(ids), ef_construction=ef_construction, M=HNSW_M, random_seed=HNSW_SEED)
    index.set_num_threads(HNSW_THREADS)
    index.add_items(vectors, np.arange(len(ids)))
    return Index(index=index, ids=list(ids), vectors=vectors)


def equal_rrf(direct: Sequence[str], evidence: Sequence[str], *, k: int = 60, budget: int = 100) -> list[str]:
    score: dict[str, float] = {}
    for channel in (direct, evidence):
        for rank, item in enumerate(channel, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda t: (-score[t], t.encode("utf-8")))[:budget]


def retain_paths(paths: dict[str, list[tuple[float, str]]], *, content_key: dict[str, str] | None,
                 budget: int = 4) -> dict[str, list[list[object]]]:
    """Deduplicate exact content and keep at most ``budget`` natural paths per target."""
    out: dict[str, list[list[object]]] = {}
    for target, entries in paths.items():
        best: dict[str, tuple[float, str]] = {}
        for score, eid in entries:
            key = (content_key or {}).get(eid, eid)
            current = best.get(key)
            if current is None or (-score, eid) < (-current[0], current[1]):
                best[key] = (score, eid)
        ordered = sorted(best.values(), key=lambda p: (-p[0], p[1].encode("utf-8")))[:budget]
        out[target] = [[e, s] for s, e in ordered]
    return out


class OwnRetriever:
    """Per-model Direct / first-hop / second-hop retrieval over its own vectors."""

    def __init__(self, model, bank, legal: Sequence[str], text_ids: Sequence[str], image_ids: Sequence[str],
                 *, device: str = "cpu", chunk: int = 4096) -> None:
        self.model = model
        self.bank = bank
        self.device = torch.device(device)
        self.legal = list(legal)
        self.text_ids = list(text_ids)
        self.image_ids = list(image_ids)
        self.target_vectors = self._vectors("table", self.legal, chunk)
        self.text_vectors = self._vectors("text", self.text_ids, chunk)
        self.image_vectors = self._vectors("image", self.image_ids, chunk)
        self.target_index = build_index(self.target_vectors, self.legal)
        self.text_index = build_index(self.text_vectors, self.text_ids)
        self.image_index = build_index(self.image_vectors, self.image_ids)
        self._first_hop: dict[tuple[str, str], list[tuple[str, float]]] = {}

    @torch.no_grad()
    def _vectors(self, kind: str, ids: Sequence[str], chunk: int) -> np.ndarray:
        out = []
        for start in range(0, len(ids), chunk):
            group = ids[start : start + chunk]
            u = self.model.u(kind, self.bank.z_many(list(group)).to(self.device))
            out.append(u.float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    @torch.no_grad()
    def _query_vector(self, qid: str | None, relation: str, *, evidence_id: str | None = None,
                      kind: str = "table") -> np.ndarray:
        if evidence_id is None:
            if not qid:
                raise ValueError("query id required when no evidence id is given")
            u = self.model.u("table", self.bank.z(qid).to(self.device))
            vec = u @ self.model.relations[relation]
        else:
            ue = self.model.u(kind, self.bank.z(evidence_id).to(self.device))
            vec = ue @ self.model.relations[relation]
        return vec.float().cpu().numpy()

    def direct(self, qid: str, k: int = 100) -> list[tuple[str, float]]:
        return self.target_index.search(self._query_vector(qid, "QT"), k)

    def first_hop(self, qid: str, modality: str, k: int = 20) -> list[tuple[str, float]]:
        key = (qid, modality)
        if key not in self._first_hop:
            index = self.text_index if modality == "text" else self.image_index
            self._first_hop[key] = index.search(self._query_vector(qid, f"Q_to_{modality}"), k)
        return self._first_hop[key]

    def second_hop(self, evidence_id: str, modality: str, k: int = 20) -> list[tuple[str, float]]:
        return self.target_index.search(
            self._query_vector(None, f"{modality}_to_T", evidence_id=evidence_id, kind=modality), k)

    def two_way(self, qid: str, *, direct_k: int = 100, evidence_k: int = 20, per_evidence: int = 20,
                retained: int = 4, content_key: dict[str, str] | None = None) -> dict:
        direct = self.direct(qid, direct_k)
        evidence_lists = {}
        path_map: dict[str, list[tuple[float, str]]] = {}
        for modality in ("text", "image"):
            hits = self.first_hop(qid, modality, evidence_k)
            evidence_lists[modality] = hits
            for eid, first_score in hits:
                for tid, second in self.second_hop(eid, modality, per_evidence):
                    path_map.setdefault(tid, []).append((first_score + second, eid))
        paths = retain_paths(path_map, content_key=content_key, budget=retained)
        evidence_scored = sorted(
            ((t, _lse([s for _, s in lst])) for t, lst in paths.items()),
            key=lambda p: (-p[1], p[0].encode("utf-8")),
        )
        evidence_ids = [t for t, _ in evidence_scored]
        c100 = equal_rrf([t for t, _ in direct], evidence_ids, budget=direct_k)
        return {
            "direct": direct,
            "first_hop": evidence_lists,
            "paths": paths,
            "evidence": evidence_ids,
            "U": utf8_sorted(set([t for t, _ in direct]) | set(evidence_ids)),
            "C100": c100,
        }


def _lse(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    return float(torch.logsumexp(torch.tensor(list(values), dtype=torch.float64), 0))


def direct_only(model, bank, legal: Sequence[str], query_ids: Sequence[str], *, device: str = "cpu",
                chunk: int = 4096) -> dict[str, list[tuple[str, float]]]:
    """QT-only online pool: own Direct100 = C100, no Q->E / E->T at all."""
    vectors = []
    for start in range(0, len(legal), chunk):
        group = legal[start : start + chunk]
        with torch.no_grad():
            u = model.keys(bank.z_many(list(group)).to(device))
        vectors.append(u.float().cpu().numpy())
    index = build_index(np.concatenate(vectors, 0), legal)
    out = {}
    for qid in query_ids:
        with torch.no_grad():
            q = model.query(bank.z(qid).to(device)).float().cpu().numpy()
        out[qid] = index.search(q, 100)
    return out

"""Exact and ANN retrieval, candidate admission, and Teacher re-ranking.

Spec section 7.1 (raw retrieval), 10.1 (indexes) and 10.2 (the deterministic
per-query pipeline).  The final score is always the shared Teacher's J value;
no path logits are added, LSE'd or RRF-fused.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import ConfigError
from .reference import round_robin
from .timing import Timing

TOP_K_DEFAULTS = {
    "direct_k": 100,
    "evidence_per_modality": 10,
    "second_hop_k": 20,
    "train_target_pool": 128,
    "train_evidence_pool": 128,
    "train_second_hop_pool": 128,
    "refreshed_pool": 128,
}


def exact_topk(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
    *,
    query_batch: int = 64,
    corpus_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunked float32 inner-product top-k over a (unit-normalized) corpus.

    Returns per-query indices and scores, each sorted by descending score and
    then by ascending corpus index.  No Q x whole-lake matrix is ever built.
    """
    if query.ndim == 1:
        query = query[None, :]
    if query.shape[1] != corpus.shape[1]:
        raise ConfigError("query and corpus dimensions differ")
    total = corpus.shape[0]
    k = int(min(k, total))
    index_out = np.zeros((query.shape[0], k), dtype=np.int64)
    score_out = np.zeros((query.shape[0], k), dtype=np.float32)
    for start in range(0, query.shape[0], query_batch):
        block = query[start : start + query_batch].astype(np.float32, copy=False)
        best_scores = np.full((block.shape[0], 0), -np.inf, dtype=np.float32)
        best_index = np.zeros((block.shape[0], 0), dtype=np.int64)
        for chunk_start in range(0, total, corpus_chunk):
            chunk = corpus[chunk_start : chunk_start + corpus_chunk].astype(np.float32, copy=False)
            scores = block @ chunk.T
            offset = np.arange(chunk_start, chunk_start + chunk.shape[0], dtype=np.int64)
            merged_scores = np.concatenate([best_scores, scores], axis=1)
            merged_index = np.concatenate(
                [best_index, np.broadcast_to(offset, scores.shape)], axis=1
            )
            order = np.lexsort((merged_index, -merged_scores), axis=1)[:, :k]
            best_scores = np.take_along_axis(merged_scores, order, axis=1)
            best_index = np.take_along_axis(merged_index, order, axis=1)
        if not np.isfinite(best_scores).all():
            raise ConfigError(
                "non-finite retrieval score encountered; NaN/Inf is an implementation "
                "error and is never replaced with 0"
            )
        index_out[start : start + block.shape[0]] = best_index
        score_out[start : start + block.shape[0]] = best_scores
    return index_out, score_out


def rank_of(ids: Sequence[str], target: str) -> int | None:
    for position, value in enumerate(ids):
        if value == target:
            return position
    return None


def interleave_text_image(
    text_ids: Sequence[str], image_ids: Sequence[str], per_modality: int, limit: int = 20
) -> list[str]:
    """Spec 10.2 step 4: text1, image1, text2, image2, ... with a hard per-modality cap."""
    text = list(text_ids[:per_modality])
    image = list(image_ids[:per_modality])
    out: list[str] = []
    seen: set[str] = set()
    for position in range(per_modality):
        for stream in (text, image):
            if position < len(stream):
                value = stream[position]
                if value not in seen:
                    seen.add(value)
                    out.append(value)
        if len(out) >= limit:
            break
    return out[:limit]


def retrieve_query(
    *,
    query_id: str,
    rank_target: Callable[[np.ndarray, int], list[str]],
    rank_text: Callable[[np.ndarray, int], list[str]],
    rank_image: Callable[[np.ndarray, int], list[str]],
    condition: Callable[[int], np.ndarray],
    u_D: np.ndarray,
    u_E: np.ndarray,
    retrieval: dict[str, Any],
    bundle: Sequence[str] | None = None,
) -> dict[str, Any]:
    """The deterministic per-query pipeline of spec 10.2 steps 2-8.

    ``rank_*`` perform one top-k retrieval against the relevant index (exact or
    ANN, already re-sorted by the true inner product) and return candidate IDs.
    ``condition(i)`` yields the conditional query vector for natural evidence i.
    ``bundle`` reuses an already computed step-4 bundle, which keeps the exact
    and ANN diagnostics on the *same* natural B_Q.
    """
    direct_k = int(retrieval["direct_k"])
    per_modality = int(retrieval["evidence_per_modality"])
    second_hop_k = int(retrieval["second_hop_k"])
    budget = int(retrieval["candidate_budget"])

    d100 = rank_target(u_D, direct_k)
    e_text = rank_text(u_E, per_modality)
    e_image = rank_image(u_E, per_modality)
    b_q = interleave_text_image(e_text, e_image, per_modality) if bundle is None else list(bundle)

    lists: dict[str, list[str]] = {}
    for position, evidence_id in enumerate(b_q):
        lists[evidence_id] = rank_target(condition(position), second_hop_k)

    streams = list(lists.values())
    r_e = round_robin(streams) if streams else []
    union = list(dict.fromkeys(list(d100) + r_e))
    c100 = round_robin([d100, r_e], budget) if r_e else list(d100[:budget])
    arrival: dict[str, list[str]] = {}
    for evidence_id, ranked in lists.items():
        for target in ranked:
            arrival.setdefault(target, []).append(evidence_id)
    return {
        "query_id": query_id,
        "D100": d100,
        "E_text": e_text,
        "E_image": e_image,
        "B_Q": b_q,
        "L_E": lists,
        "R_E": r_e,
        "U": union,
        "C100": c100,
        "arrival_evidence_by_target": arrival,
    }


# --------------------------------------------------------------------------
# ANN (hnswlib) — spec section 10.1
# --------------------------------------------------------------------------


class AnnIndex:
    """Thin deterministic wrapper over hnswlib's inner-product index."""

    def __init__(
        self,
        keys: np.ndarray,
        ann: dict[str, Any],
        seed: int,
        label: str = "index",
        timing: Timing | None = None,
    ):
        import hnswlib

        self.timing = timing if timing is not None else Timing()
        self.label = label
        self.keys = np.ascontiguousarray(keys.astype(np.float32))
        self.dim = self.keys.shape[1]
        self.config = dict(ann)
        with self.timing.stage("index_init", index=label):
            self.index = hnswlib.Index(space=str(ann["space"]), dim=self.dim)
            self.index.init_index(
                max_elements=self.keys.shape[0],
                ef_construction=int(ann["ef_construction"]),
                M=int(ann["M"]),
                random_seed=int(seed),
            )
            self.index.set_num_threads(int(ann["construction_threads"]))
        labels = np.arange(self.keys.shape[0], dtype=np.int64)
        with self.timing.stage("index_add_items", index=label):
            self.index.add_items(self.keys, labels)
        self.index.set_num_threads(int(ann["query_threads"]))
        self.index.set_ef(int(ann["ef_search"]))
        self.bytes = int(self.keys.shape[0]) * self.dim * 4

    def query(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = int(min(k, self.keys.shape[0]))
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        with self.timing.stage("ann_knn_query", index=self.label):
            labels, distances = self.index.knn_query(vectors, k=k)
        # hnswlib returns distance = 1 - dot for space=ip; recompute the real dot.
        return labels, distances


def ann_order(
    index: AnnIndex, vector: np.ndarray, k: int, corpus_ids: Sequence[str]
) -> list[str]:
    labels, _ = index.query(vector[None, :], k)
    with index.timing.stage("ann_rescore", index=index.label):
        ids = [corpus_ids[i] for i in labels[0]]
        scores = index.keys[labels[0]] @ vector.astype(np.float32)
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    out: list[str] = []
    seen: set[str] = set()
    for i in order:
        if ids[i] not in seen:
            seen.add(ids[i])
            out.append(ids[i])
    return out


def nn_fidelity(
    ann_ids: Sequence[str], exact_ids: Sequence[str], k: int
) -> float:
    """|ANN_k ∩ Exact_k| / k.  Not a GT recall (spec 12.5)."""
    left = list(ann_ids[:k])
    right = set(exact_ids[:k])
    if not left:
        return 0.0
    return len([i for i in left if i in right]) / float(min(k, len(left)))

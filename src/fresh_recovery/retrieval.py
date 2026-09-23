"""Own HNSW retrieval for a Student (SPEC 5.1-5.5), plus exact Direct and MatchedDirectM.

Indexes are built from the calling model's own final vectors: target key
``u_t``, relation query ``u_a @ R``.  HNSW inner product, M=32,
efConstruction=200, seed = model seed, single-threaded UTF-8 insertion order,
efSearch = max(256, requested K), returned set re-scored with exact inner
product before truncation.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import Labels, RowStore, utf8_sorted
from .io import sha256_json
from .pools import CANDIDATE_BUDGET, DIRECT_K, FIRST_HOP_K, SECOND_HOP_K, PoolRecord, assemble_pool

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
EF_FLOOR = 256


@dataclass
class Index:
    ids: list[str]
    vectors: np.ndarray
    index: object
    sha256: str
    build_seconds: float

    def search(self, query: np.ndarray, k: int) -> tuple[list[tuple[str, float]], int]:
        k = min(k, len(self.ids))
        if k == 0:
            return [], 0
        ef = max(EF_FLOOR, k)
        self.index.set_ef(ef)
        labels, _ = self.index.knn_query(query.reshape(1, -1).astype(np.float32), k=k)
        got = [int(i) for i in labels[0]]
        scores = self.vectors[got] @ query.astype(np.float32)
        order = sorted(range(len(got)), key=lambda i: (-float(scores[i]), self.ids[got[i]].encode("utf-8")))
        return [(self.ids[got[i]], float(scores[i])) for i in order[:k]], ef


def build_index(vectors: np.ndarray, ids: Sequence[str], *, seed: int) -> Index:
    import hnswlib

    started = time.time()
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    index = hnswlib.Index(space="ip", dim=vectors.shape[1])
    index.init_index(max_elements=len(ids), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M, random_seed=seed)
    index.set_num_threads(1)
    index.add_items(vectors, np.arange(len(ids)), num_threads=1)
    digest = hashlib.sha256()
    digest.update(vectors.tobytes())
    digest.update("\n".join(ids).encode("utf-8"))
    return Index(ids=list(ids), vectors=vectors, index=index, sha256=digest.hexdigest(),
                 build_seconds=time.time() - started)


class OwnRetriever:
    """Direct / first-hop / second-hop retrieval on one Student's own indexes."""

    def __init__(self, model, bank, labels: Labels, rows: RowStore, *, device: str, seed: int,
                 generator_id: str, model_sha: str, qt_only: bool = False, chunk: int = 4096) -> None:
        self.model = model
        self.bank = bank
        self.labels = labels
        self.rows = rows
        self.device = torch.device(device)
        self.seed = seed
        self.generator_id = generator_id
        self.model_sha = model_sha
        self.qt_only = qt_only
        self.chunk = chunk
        self.legal = list(labels.legal)
        vectors = {"table": self._vectors("table", self.legal)}
        if not qt_only:
            vectors["text"] = self._vectors("text", labels.canonical_text)
            vectors["image"] = self._vectors("image", labels.canonical_image)
        self.indexes: dict[str, Index] = {}
        threads = []
        for kind, ids in (("table", self.legal), ("text", labels.canonical_text), ("image", labels.canonical_image)):
            if kind not in vectors:
                continue
            thread = threading.Thread(target=self._build, args=(kind, vectors[kind], ids))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.target_vectors_gpu = torch.from_numpy(self.indexes["table"].vectors).to(self.device)
        self.target_index_sha = self.indexes["table"].sha256
        self.evidence_index_sha = sha256_json({k: self.indexes[k].sha256 for k in ("text", "image") if k in self.indexes})
        self.meta = {"hnsw": {"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION, "seed": seed, "threads": 1,
                              "insertion": "UTF-8 id order", "ef_floor": EF_FLOOR, "exact_rescore": True},
                     "index_sha256": {k: v.sha256 for k, v in self.indexes.items()},
                     "build_seconds": {k: v.build_seconds for k, v in self.indexes.items()},
                     "generator_id": generator_id, "model_sha": model_sha}

    def _build(self, kind: str, vectors: np.ndarray, ids: Sequence[str]) -> None:
        self.indexes[kind] = build_index(vectors, ids, seed=self.seed)

    @torch.no_grad()
    def _vectors(self, kind: str, ids: Sequence[str]) -> np.ndarray:
        out = []
        for start in range(0, len(ids), self.chunk):
            group = ids[start : start + self.chunk]
            out.append(self.model.u(kind, self.bank.z_many(list(group)).to(self.device)).float().cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)

    @torch.no_grad()
    def query_vector(self, relation: str, anchor_id: str) -> np.ndarray:
        return self.model.query_vector(relation, self.bank.z(anchor_id).to(self.device)).float().cpu().numpy()

    def direct(self, qid: str, k: int = DIRECT_K) -> tuple[list[tuple[str, float]], int]:
        return self.indexes["table"].search(self.query_vector("QT", qid), k)

    @torch.no_grad()
    def direct_exact(self, qid: str, k: int = DIRECT_K) -> list[tuple[str, float]]:
        q = torch.from_numpy(self.query_vector("QT", qid)).to(self.device)
        scores = self.target_vectors_gpu @ q
        top = torch.argsort(scores, descending=True, stable=True)[:k]
        return [(self.legal[int(i)], float(scores[int(i)])) for i in top]

    def first_hop(self, qid: str, modality: str, k: int = FIRST_HOP_K) -> list[tuple[str, float]]:
        hits, _ = self.indexes[modality].search(self.query_vector(f"Q_{modality}", qid), k)
        return hits

    def second_hop(self, evidence_id: str, modality: str, k: int = SECOND_HOP_K) -> list[tuple[str, float]]:
        hits, _ = self.indexes["table"].search(self.query_vector(f"{modality}_T", evidence_id), k)
        return hits

    def pool(self, split: str, qid: str) -> PoolRecord:
        direct, ef = self.direct(qid)
        exact = [t for t, _ in self.direct_exact(qid)]
        if self.qt_only:
            first_hop = {"text": [], "image": []}
        else:
            first_hop = {m: self.first_hop(qid, m) for m in ("text", "image")}
        qvec = torch.from_numpy(self.query_vector("QT", qid)).to(self.device)
        all_scores = (self.target_vectors_gpu @ qvec).float().detach().cpu().tolist()
        qt_scores = {target: float(score) for target, score in zip(self.legal, all_scores)}
        return assemble_pool(
            split=split, query_id=qid, generator_id=self.generator_id, model_sha=self.model_sha,
            target_index_sha=self.target_index_sha, evidence_index_sha=self.evidence_index_sha,
            direct=direct, direct_exact=exact, first_hop=first_hop,
            second_hop=self.second_hop, query_rows=self.rows.get(qid),
            evidence_z=lambda ids: self.bank.z_many(list(ids)).float().cpu().numpy(), content_key=None,
            qt_scores=qt_scores, candidate_budget=CANDIDATE_BUDGET,
            retrieval_meta={"ef_direct": ef, "mode": "hnsw_ip_exact_rescore", **self.meta["hnsw"]},
        )

    def direct_m(self, qid: str, m: int) -> tuple[list[tuple[str, float]], int]:
        """MatchedDirectM: own Direct with K = M (ef = max(256, M))."""
        return self.direct(qid, m)


def student_identity(payload: dict, path: Path) -> str:
    return f"{payload['stage']}:{payload.get('snapshot', payload.get('epoch'))}:{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]}"

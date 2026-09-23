"""Raw (frozen ``z``) retrieval: reservoirs, two-way pools, ET reservoirs (SPEC 6.1, 5).

All scores are exact inner products of the approved pure ``z``; the only
learned object is nothing.  Tie rule everywhere: score descending, UTF-8 ID
ascending (ids are kept UTF-8 sorted so a stable descending sort applies it).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .config import Paths
from .data import Labels, RowStore, ZStore, load_labels, load_row_store, load_z, split_query_ids, utf8_sorted
from .io import sha256_file, sha256_json, write_json
from .pools import (CANDIDATE_BUDGET, DIRECT_K, FIRST_HOP_K, SECOND_HOP_K, PoolRecord, assemble_pool,
                    coverage_at, pool_summary)

RESERVOIR = 256


def stable_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


class RawIndex:
    """Exact-score libraries over the legal targets and canonical evidence."""

    def __init__(self, z: ZStore, labels: Labels, device: str) -> None:
        self.device = torch.device(device)
        self.z = z
        self.targets = list(labels.legal)
        self.text = list(labels.canonical_text)
        self.image = list(labels.canonical_image)
        self.z_targets = z.rows(self.targets).to(self.device)
        self.z_text = z.rows(self.text).to(self.device)
        self.z_image = z.rows(self.image).to(self.device)
        self._position = {t: i for i, t in enumerate(self.targets)}
        self.target_index_sha = sha256_json({"ids": self.targets, "z": z.sha256})
        self.evidence_index_sha = sha256_json({"text": self.text, "image": self.image, "z": z.sha256})

    def position(self, target: str) -> int:
        return self._position[target]

    def vector(self, object_id: str) -> torch.Tensor:
        return self.z.z[self.z.index[object_id]].to(self.device)

    def target_scores(self, vector: torch.Tensor) -> torch.Tensor:
        return self.z_targets @ vector

    def topk_targets(self, vector: torch.Tensor, k: int) -> list[tuple[str, float]]:
        scores = self.target_scores(vector)
        top = stable_topk(scores, k)
        return [(self.targets[int(i)], float(scores[int(i)])) for i in top]

    def topk_evidence(self, vector: torch.Tensor, modality: str, k: int) -> list[tuple[str, float]]:
        library = self.text if modality == "text" else self.image
        z = self.z_text if modality == "text" else self.z_image
        scores = z @ vector
        top = stable_topk(scores, k)
        return [(library[int(i)], float(scores[int(i)])) for i in top]

    def second_hops(self, evidence_ids: Sequence[str], k: int) -> dict[str, list[tuple[str, float]]]:
        if not evidence_ids:
            return {}
        ez = self.z.rows(evidence_ids).to(self.device)
        scores = ez @ self.z_targets.T  # (n_e, n_t)
        out: dict[str, list[tuple[str, float]]] = {}
        for row, evidence_id in enumerate(evidence_ids):
            top = stable_topk(scores[row], k)
            out[evidence_id] = [(self.targets[int(i)], float(scores[row, int(i)])) for i in top]
        return out


def raw_query_pool(index: RawIndex, rows: RowStore, split: str, query_id: str, *,
                   model_sha: str) -> tuple[PoolRecord, dict]:
    qz = index.vector(query_id)
    target_scores = index.target_scores(qz)
    d256 = stable_topk(target_scores, RESERVOIR)
    qt_top256 = [index.targets[int(i)] for i in d256]
    direct = [(index.targets[int(i)], float(target_scores[int(i)])) for i in d256[:DIRECT_K]]
    first_hop = {}
    reservoirs = {}
    for modality in ("text", "image"):
        hits = index.topk_evidence(qz, modality, RESERVOIR)
        reservoirs[modality] = [e for e, _ in hits]
        first_hop[modality] = hits[:FIRST_HOP_K]
    evidence_ids = [e for m in ("text", "image") for e, _ in first_hop[m]]
    second = index.second_hops(evidence_ids, SECOND_HOP_K)
    pool = assemble_pool(
        split=split, query_id=query_id, generator_id="raw", model_sha=model_sha,
        target_index_sha=index.target_index_sha, evidence_index_sha=index.evidence_index_sha,
        direct=direct, direct_exact=[t for t, _ in direct], first_hop=first_hop,
        second_hop=lambda e, m: second[e], query_rows=rows.get(query_id),
        evidence_z=lambda ids: index.z.rows(ids).numpy(), content_key=None,
        qt_scores={index.targets[int(i)]: float(target_scores[int(i)]) for i in range(len(index.targets))},
        retrieval_meta={"mode": "exact_inner_product", "direct_k": DIRECT_K, "first_hop_k": FIRST_HOP_K,
                        "second_hop_k": SECOND_HOP_K},
    )
    # QT reservoir: raw QT256 u raw U, re-ranked by the raw QT score only.
    union = list(dict.fromkeys([*qt_top256, *pool.U]))
    positions = torch.tensor([index.position(t) for t in union], dtype=torch.long)
    union_scores = target_scores[positions.to(index.device)].cpu().tolist()
    order = sorted(range(len(union)), key=lambda i: (-union_scores[i], union[i].encode("utf-8")))
    reservoir = {
        "qt_top256": qt_top256,
        "qt_reservoir": [union[i] for i in order],
        "qe_reservoir": reservoirs,
    }
    return pool, reservoir


def save_pools(path: Path, pools: dict[str, PoolRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump({q: p.to_json() for q, p in pools.items()}, handle, protocol=4)
    tmp.replace(path)
    return sha256_file(path)


def load_pools(path: Path) -> dict[str, PoolRecord]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return {q: PoolRecord.from_json(p) for q, p in payload.items()}


def build_raw(paths: Paths, *, split: str, device: str = "cuda:0", log=print) -> dict:
    labels = load_labels(paths)
    z = load_z(paths)
    rows = load_row_store(paths)
    index = RawIndex(z, labels, device)
    z_sha = json.loads((paths.work_dir / "pca" / "PCA_RUN_CONTRACT.json").read_text())["z_sha256"]
    model_sha = f"pure_z:{z_sha}"
    if split == "train":
        queries = labels.query_ids
        gold = {q: labels.queries[q]["G"] for q in queries}
    else:
        # Candidate generation is label-blind.  Existing dev/test qrels are
        # not opened by this producer at all; the evaluator owns GT access.
        queries = split_query_ids(paths, split)
        gold = {}
    started = time.time()
    pools: dict[str, PoolRecord] = {}
    reservoirs: dict[str, dict] = {}
    with torch.no_grad():
        for n, qid in enumerate(queries, 1):
            pool, reservoir = raw_query_pool(index, rows, split, qid, model_sha=model_sha)
            pools[qid] = pool
            reservoirs[qid] = reservoir
            if n % 500 == 0:
                log(json.dumps({"event": "raw_progress", "split": split, "done": n, "total": len(queries),
                                "elapsed": round(time.time() - started, 1)}))
    out_dir = paths.work_dir / "raw" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    pools_sha = save_pools(out_dir / "pools.pkl", pools)
    reservoir_path = out_dir / "reservoirs.pkl"
    with reservoir_path.open("wb") as handle:
        pickle.dump(reservoirs, handle, protocol=4)
    summary = (pool_summary(pools, gold) if split == "train"
               else {"queries": len(pools), "label_free": True, "coverage_deferred": "evaluation"})
    et_summary = None
    if split == "train":
        et_summary = build_et_reservoir(paths, index, labels, log=log)
    report = {
        "split": split, "queries": len(queries), "generator_id": "raw", "model_sha": model_sha,
        "target_index_sha": index.target_index_sha, "evidence_index_sha": index.evidence_index_sha,
        "budgets": {"direct": DIRECT_K, "first_hop": FIRST_HOP_K, "second_hop": SECOND_HOP_K,
                    "reservoir": RESERVOIR, "candidate_budget": CANDIDATE_BUDGET,
                    "retention": "D1_soft_row_coverage", "admission": "QTALL_equal_RRF_D1"},
        "summary": summary, "pools_sha256": pools_sha,
        "reservoirs_sha256": sha256_file(reservoir_path), "elapsed_seconds": time.time() - started,
        "et_reservoir": et_summary,
    }
    write_json(out_dir / "RAW_REPORT.json", report)
    return report


def build_et_reservoir(paths: Paths, index: RawIndex, labels: Labels, *, log=print) -> dict:
    anchors = utf8_sorted(labels.epos)
    out: dict[str, list[str]] = {}
    with torch.no_grad():
        for start in range(0, len(anchors), 256):
            group = anchors[start : start + 256]
            hops = index.second_hops(group, RESERVOIR)
            for e in group:
                out[e] = [t for t, _ in hops[e]]
    path = paths.work_dir / "raw" / "et_reservoir.pkl"
    with path.open("wb") as handle:
        pickle.dump(out, handle, protocol=4)
    log(json.dumps({"event": "et_reservoir", "anchors": len(out)}))
    return {"anchors": len(out), "sha256": sha256_file(path)}


def load_reservoirs(paths: Paths, split: str = "train") -> dict[str, dict]:
    with (paths.work_dir / "raw" / split / "reservoirs.pkl").open("rb") as handle:
        return pickle.load(handle)


def load_et_reservoir(paths: Paths) -> dict[str, list[str]]:
    with (paths.work_dir / "raw" / "et_reservoir.pkl").open("rb") as handle:
        return pickle.load(handle)

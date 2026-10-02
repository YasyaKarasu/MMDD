"""Two-hop candidate retrieval for Stage-1 CQET: HNSW indices, D1 retention, P3 admission.

``build_pools`` runs the formal retrieval of one split under one scoring space: the frozen
Qwen ``z`` (Raw, ``student=None``) or a Student's projected relation vectors. Per query it
takes the direct ANN top-150 targets, the top-20 text and image evidence, the top-50 targets
of every evidence (second hop), keeps at most four evidence per target by greedy row coverage
(D1), and admits the C150 Teacher pool by reciprocal-rank fusion of the QT order with the
D1-coverage order (P3). Every ANN hit is checked against the exact bilinear score.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import hnswlib
import numpy as np
import torch
from torch import Tensor

from .data import utf8_sorted
from .features import RowStore, ZStore
from .labels import Labels
from .models import NativeStudent, QTStudent, model_state_sha

CANDIDATE_BUDGET = 150
DIRECT_K = 100
FIRST_HOP_K = 20
SECOND_HOP_K = 50
EVIDENCE_BUDGET = 4
TOP_L = 16
RRF_K = 60
TRAINING_EXACT_K = 128


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def stable_topk(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


def stable_topk_rows(scores: Tensor, k: int) -> Tensor:
    """Row-wise ``stable_topk`` of a (rows, n) score matrix, returned on the host."""
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, : min(k, scores.shape[1])].cpu()


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
    D150: list[tuple[str, float]] = field(default_factory=list)
    MatchedDirectC: list[tuple[str, float]] = field(default_factory=list)
    MatchedDirectU: list[tuple[str, float]] = field(default_factory=list)
    d1_scores: dict[str, float] = field(default_factory=dict)
    qt_ranks: dict[str, int] = field(default_factory=dict)
    d1_ranks: dict[str, int] = field(default_factory=dict)
    object_vector_hash: str = ""
    index_hash: str = ""
    score_space: str = "bilinear_ip"
    ann_exact_overlap: dict[str, float] = field(default_factory=dict)
    training_exact: dict[str, object] = field(default_factory=dict)
    d1_trace: dict[str, list[dict[str, object]]] = field(default_factory=dict)
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
    """SPEC 8.1 D1 greedy soft row coverage: ``(retained evidence ids, coverage)``."""
    selected, coverage, _trace = d1_retain_with_trace(
        entries, support, content_key=content_key, top_l=top_l, budget=budget
    )
    return selected, coverage


def d1_retain_with_trace(
    entries: Sequence[PathEntry],
    support: Mapping[str, np.ndarray],
    *,
    content_key: Optional[Mapping[str, str]] = None,
    top_l: int = TOP_L,
    budget: int = EVIDENCE_BUDGET,
) -> tuple[list[str], float, list[dict[str, object]]]:
    """D1 retention plus per-candidate gain/row-support audit data.

    One path per canonical content keeps the best raw score; the ``top_l`` best paths compete.
    Each step adds the candidate with the largest coverage gain (ties: larger sigmoid score,
    then smaller id) until ``budget`` paths are kept or no candidate still adds coverage.
    """
    if not entries:
        return [], 0.0, []
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
    ids = [entry.evidence_id for entry in candidates]
    rows = np.stack([np.asarray(support[e], dtype=np.float64) for e in ids])
    if rows.shape[1] == 0:
        raise ValueError("query rows required for D1 retention")
    activation = np.array([sigmoid(entry.raw_path_score) for entry in candidates])
    weighted = activation[:, None] * rows
    current = np.zeros(rows.shape[1], dtype=np.float64)
    remaining = list(range(len(ids)))
    selected: list[str] = []
    gain = np.full(len(ids), np.nan)
    step = [None] * len(ids)
    for _ in range(min(budget, len(ids))):
        updated = np.maximum(current[None, :], weighted[remaining])
        gains = updated.mean(axis=1) - current.mean()
        pick = min(range(len(remaining)), key=lambda k: (-gains[k], -activation[remaining[k]], ids[remaining[k]].encode("utf-8")))
        if gains[pick] <= 0.0:
            break
        i = remaining[pick]
        selected.append(ids[i])
        step[i], gain[i], current = len(selected), float(gains[pick]), updated[pick]
        remaining.remove(i)
    if remaining:
        gain[remaining] = np.maximum(current[None, :], weighted[remaining]).mean(axis=1) - current.mean()
    trace = [
        {"evidence_id": ids[i], "path_raw_score": candidates[i].raw_path_score,
         "row_support_mean": float(rows[i].mean()), "selected_step": step[i], "marginal_gain": float(gain[i])}
        for i in range(len(ids))
    ]
    return selected, float(current.mean()), trace


def p3_admission(
    qt_scores: dict[str, float],
    evidence_order: Sequence[str],
    budget: int = CANDIDATE_BUDGET,
    constant: int = RRF_K,
) -> tuple[list[str], dict[str, float]]:
    """P3 admission (SPEC 8.2): RRF of the QT order with the D1-coverage order of evidence targets."""
    if budget <= 0 or constant < 0:
        raise ValueError("invalid admission parameters")
    if len(evidence_order) != len(set(evidence_order)):
        raise ValueError("duplicate evidence-channel target")
    if any(t not in qt_scores for t in evidence_order):
        raise ValueError("all evidence targets need real QT scores")

    qt_order = sorted(qt_scores, key=lambda t: (-qt_scores[t], t.encode("utf-8")))
    q_rank = {t: i + 1 for i, t in enumerate(qt_order)}
    e_rank = {t: i + 1 for i, t in enumerate(evidence_order)}
    score = {
        t: 1.0 / (constant + q_rank[t]) + (1.0 / (constant + e_rank[t]) if t in e_rank else 0.0)
        for t in qt_order
    }
    admitted = sorted(qt_order, key=lambda t: (-score[t], t.encode("utf-8")))[:budget]
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
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(ids), dim):
            raise ValueError("HNSW vectors/IDs/dimension do not align")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate HNSW object ID")
        if not len(ids):
            raise ValueError("empty HNSW library")
        self.dim = dim
        self.ids = list(ids)
        self.ef_search_floor = ef_search_floor
        self.seed = seed
        self.vector_hash = hashlib.sha256(memoryview(np.ascontiguousarray(vectors)).cast("B")).hexdigest()
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.index.init_index(max_elements=len(ids), ef_construction=ef_construction, M=m, random_seed=seed)
        self.index.set_num_threads(1)
        # Insertion in UTF-8 object ID order
        sorted_indices = sorted(range(len(ids)), key=lambda i: ids[i].encode("utf-8"))
        self.index.add_items(vectors[sorted_indices], np.asarray(sorted_indices, dtype=np.int64))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.shape != (self.dim,):
            raise ValueError(f"HNSW query must have shape ({self.dim},)")
        k = min(int(k), len(self.ids))
        if k <= 0:
            return []
        self.index.set_ef(max(self.ef_search_floor, k))
        labels, distances = self.index.knn_query(query_vector.reshape(1, -1), k=k)
        # hnswlib inner product space returns 1 - inner_product
        scores = 1.0 - distances[0]
        hits = [(self.ids[int(lbl)], float(scores[i])) for i, lbl in enumerate(labels[0])]
        return sorted(hits, key=lambda x: (-x[1], x[0].encode("utf-8")))

    def save(self, path: Path) -> dict[str, object]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))
        meta = {
            "dim": self.dim,
            "ids": self.ids,
            "ef_search_floor": self.ef_search_floor,
            "seed": self.seed,
            "vector_hash": self.vector_hash,
            "index_sha256": _sha256_file(path),
            "score_space": "bilinear_ip",
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return meta

    @classmethod
    def load(cls, path: Path) -> "HNSWIndex":
        path = Path(path)
        meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if _sha256_file(path) != meta["index_sha256"]:
            raise ValueError("HNSW index hash mismatch")
        obj = cls.__new__(cls)
        obj.dim = int(meta["dim"])
        obj.ids = [str(x) for x in meta["ids"]]
        obj.ef_search_floor = int(meta["ef_search_floor"])
        obj.seed = int(meta["seed"])
        obj.vector_hash = str(meta["vector_hash"])
        obj.index = hnswlib.Index(space="ip", dim=obj.dim)
        obj.index.load_index(str(path), max_elements=len(obj.ids))
        obj.index.set_num_threads(1)
        return obj


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class _Library:
    """One right-hand object library of a relation: its index vectors (device and host), HNSW index
    and exact-score audit."""

    def __init__(self, relation: str, ids: Sequence[str], vectors: Tensor, *, seed: int,
                 index_path: Optional[Path], reuse_path: Optional[Path]) -> None:
        self.relation = relation
        self.ids = list(ids)
        self.position = {object_id: i for i, object_id in enumerate(self.ids)}
        self.vectors = vectors
        self.host = vectors.cpu().numpy().astype(np.float32)
        if reuse_path is None:
            self.index = HNSWIndex(self.host, self.ids, dim=vectors.shape[1], seed=seed)
        else:
            self.index = HNSWIndex.load(reuse_path)
            vector_hash = hashlib.sha256(memoryview(np.ascontiguousarray(self.host)).cast("B")).hexdigest()
            if self.index.ids != self.ids or self.index.seed != seed or self.index.vector_hash != vector_hash:
                raise ValueError(f"reused Raw HNSW index does not match current objects: {reuse_path}")
        self.vector_hash = self.index.vector_hash
        self.meta = self.index.save(index_path) if index_path is not None else None

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Checked ANN search: every returned score must equal the exact inner product."""
        hits = self.index.search(query, k)
        if hits:
            exact = self.host[[self.position[object_id] for object_id, _ in hits]] @ query
            scores = np.array([score for _, score in hits])
            if not np.allclose(scores, exact, rtol=1e-4, atol=1e-5):
                bad = int(np.argmax(~np.isclose(scores, exact, rtol=1e-4, atol=1e-5)))
                raise AssertionError(f"ANN score mismatch for {hits[bad][0]}: {scores[bad]} != {exact[bad]}")
        return hits


def _relation_vectors(student, relation: str, z: Tensor) -> Tensor:
    return z if student is None else student.index_vectors(relation, z)


def _relation_query(student, relation: str, z: Tensor) -> Tensor:
    return z if student is None else student.ann_query(relation, z)


def build_pools(
    z_store: ZStore,
    row_store: Optional[RowStore],
    query_ids: Sequence[str],
    labels: Labels,
    split: str | Mapping[str, str],
    *,
    student: NativeStudent | QTStudent | None,
    generator_id: str,
    hnsw_seed: int,
    device: str = "cuda:0",
    index_dir: Optional[Path] = None,
    reuse_index_dir: Optional[Path] = None,
    training_exact: bool = False,
) -> dict[str, PoolRecord]:
    """Formal two-hop retrieval of ``query_ids`` (see module docstring).

    ``student=None`` scores every relation with the frozen ``z`` (Raw); a ``QTStudent`` has no
    evidence relations and admits its direct top-150. ``index_dir`` saves the three HNSW
    indices, ``reuse_index_dir`` reloads same-seed indices built earlier (their ids, seed and
    vector hashes must match). ``training_exact`` adds the exact top-128 QT / QE lists the
    Teacher training lists are built from.
    """
    dev = torch.device(device)
    if student is not None:
        student.to(dev).eval()
    qt_only = isinstance(student, QTStudent)
    relations = {"QT": labels.legal_targets} if qt_only else {
        "QT": labels.legal_targets, "Q_text": labels.canonical_text, "Q_image": labels.canonical_image,
    }
    index_names = {"QT": "targets.hnsw", "Q_text": "text.hnsw", "Q_image": "image.hnsw"}
    with torch.no_grad():
        libraries = {
            relation: _Library(
                relation, ids, _relation_vectors(student, relation, z_store.rows(ids).to(dev)), seed=hnsw_seed,
                index_path=None if index_dir is None else index_dir / index_names[relation],
                reuse_path=None if reuse_index_dir is None else reuse_index_dir / index_names[relation],
            )
            for relation, ids in relations.items()
        }
    targets = libraries["QT"]
    index_meta = {
        f"{name}_vector_hash": libraries[relation].vector_hash if relation in libraries else None
        for name, relation in (("target", "QT"), ("text", "Q_text"), ("image", "Q_image"))
    }
    index_meta.update({
        "model_state_hash": None if student is None else model_state_sha(student),
        "score_space": "bilinear_ip",
        **{relation: library.meta for relation, library in libraries.items() if library.meta is not None},
    })
    index_hash = hashlib.sha256(json.dumps(index_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    object_vector_hash = hashlib.sha256("".join(l.vector_hash for l in libraries.values()).encode("ascii")).hexdigest()
    # HNSW search is deterministic for a fixed index and query, so each distinct evidence
    # object needs one checked second-hop search per split.
    second_hop_cache: dict[str, list[tuple[str, float]]] = {}
    pools: dict[str, PoolRecord] = {}

    with torch.no_grad():
        for qid in query_ids:
            zq = z_store.vector(qid).to(dev)
            queries = {relation: _relation_query(student, relation, zq) for relation in relations}
            qt_query = queries["QT"].cpu().numpy().astype(np.float32)
            direct_150 = targets.search(qt_query, CANDIDATE_BUDGET)
            direct_100 = direct_150[:DIRECT_K]
            exact = {relation: libraries[relation].vectors @ queries[relation] for relation in relations}

            first_hop: dict[str, list[tuple[str, float]]] = {"text": [], "image": []}
            pre_paths: dict[str, list[PathEntry]] = defaultdict(list)
            et_overlaps: list[float] = []
            if not qt_only:
                for modality in ("text", "image"):
                    first_hop[modality] = libraries[f"Q_{modality}"].search(
                        queries[f"Q_{modality}"].cpu().numpy().astype(np.float32), FIRST_HOP_K)
                evidence_items = [(m, e, s) for m in ("text", "image") for e, s in first_hop[m]]
                if evidence_items:
                    et_queries = torch.stack([
                        _relation_query(student, f"{m}_T", z_store.vector(e).to(dev)) for m, e, _ in evidence_items
                    ])
                    exact_second = stable_topk_rows(et_queries @ targets.vectors.T, SECOND_HOP_K)
                    et_host = et_queries.cpu().numpy().astype(np.float32)
                for row, (modality, evidence_id, first_score) in enumerate(evidence_items):
                    second_hits = second_hop_cache.get(evidence_id)
                    if second_hits is None:
                        second_hits = second_hop_cache[evidence_id] = targets.search(et_host[row], SECOND_HOP_K)
                    exact_ids = {targets.ids[int(i)] for i in exact_second[row]}
                    et_overlaps.append(len({t for t, _ in second_hits} & exact_ids) / len(second_hits))
                    for target_id, second_score in second_hits:
                        pre_paths[target_id].append(PathEntry(evidence_id, modality, first_score, second_score))
            evidence_targets = utf8_sorted(pre_paths)

            # D1 retention of at most EVIDENCE_BUDGET evidence per two-hop target.
            path_evidence = utf8_sorted({p.evidence_id for paths in pre_paths.values() for p in paths})
            support_map = {}
            if path_evidence:
                q_rows = row_store.get(qid) if row_store is not None else np.zeros((1, z_store.dim), dtype=np.float32)
                support = row_support(q_rows, z_store.rows(path_evidence).numpy())
                support_map = {e: support[:, i] for i, e in enumerate(path_evidence)}
            retained_paths, retained_coverage, d1_trace = {}, {}, {}
            for target_id in evidence_targets:
                retained_paths[target_id], retained_coverage[target_id], d1_trace[target_id] = d1_retain_with_trace(
                    pre_paths[target_id], support_map, content_key=labels.canonical_map,
                )
            evidence_order = sorted(
                (t for t in evidence_targets if retained_paths[t]),
                key=lambda t: (-retained_coverage[t], t.encode("utf-8")),
            )

            # U = direct top-100 plus every two-hop target; P3 admits C150 from U.
            u_list = list(dict.fromkeys([t for t, _ in direct_100] + evidence_targets))
            u_scores = exact["QT"][[targets.position[t] for t in u_list]].tolist()
            qt_scores_all_u = dict(zip(u_list, u_scores))
            if qt_only:
                c150 = [t for t, _ in direct_150]
                adm_scores = {t: 1.0 / (RRF_K + i) for i, t in enumerate(c150, 1)}
            else:
                c150, adm_scores = p3_admission(qt_scores_all_u, evidence_order)
            qt_order = sorted(qt_scores_all_u, key=lambda t: (-qt_scores_all_u[t], t.encode("utf-8")))
            matched_c = targets.search(qt_query, min(CANDIDATE_BUDGET, len(u_list)))
            matched_u = targets.search(qt_query, len(u_list))

            # Exact audits of the ANN hops.
            top = {relation: stable_topk(exact[relation], max(CANDIDATE_BUDGET, len(u_list))) for relation in relations}
            direct_exact = [targets.ids[int(i)] for i in top["QT"]]
            overlap = {"QT_D100": len({t for t, _ in direct_100} & set(direct_exact[:DIRECT_K])) / max(1, len(direct_100))}
            if not qt_only:
                for modality in ("text", "image"):
                    library = libraries[f"Q_{modality}"]
                    exact_20 = {library.ids[int(i)] for i in top[f"Q_{modality}"][:FIRST_HOP_K]}
                    hits = first_hop[modality]
                    overlap[f"Q_{modality}"] = len({e for e, _ in hits} & exact_20) / max(1, len(hits))
                overlap["ET_mean"] = float(np.mean(et_overlaps)) if et_overlaps else 0.0

            pools[qid] = PoolRecord(
                split=split[qid] if isinstance(split, Mapping) else split,
                query_id=qid,
                generator_id=generator_id,
                direct=direct_100,
                direct_exact=direct_exact,
                first_hop=first_hop,
                pre_paths=pre_paths,
                retained_paths=retained_paths,
                retained_coverage=retained_coverage,
                U=u_list,
                C150=c150,
                qt_scores_all_U=qt_scores_all_u,
                admission_scores=adm_scores,
                D150=direct_150,
                MatchedDirectC=matched_c,
                MatchedDirectU=matched_u,
                d1_scores=retained_coverage,
                qt_ranks={t: j + 1 for j, t in enumerate(qt_order)},
                d1_ranks={t: j + 1 for j, t in enumerate(evidence_order)},
                object_vector_hash=object_vector_hash,
                index_hash=index_hash,
                ann_exact_overlap=overlap,
                training_exact={
                    "RawQT128": direct_exact[:TRAINING_EXACT_K],
                    "RawQE128": {m: [libraries[f"Q_{m}"].ids[int(i)] for i in top[f"Q_{m}"][:TRAINING_EXACT_K]]
                                 for m in ("text", "image")},
                } if training_exact else {},
                d1_trace=d1_trace,
            )
    return pools

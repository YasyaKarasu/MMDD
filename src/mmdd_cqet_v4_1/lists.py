"""Candidate list builders and witness competitor extraction for CLEAN-QET v4.0."""
from __future__ import annotations

import hashlib
import heapq
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Collection, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import Paths
from .data import iter_jsonl, read_json, sha256_file, utf8_sorted, write_json, write_jsonl_gz
from .features import RowStore, ZStore
from .labels import Labels
from .models import NativeStudent
from .retrieval import (
    CANDIDATE_BUDGET,
    DIRECT_K,
    EVIDENCE_BUDGET,
    FIRST_HOP_K,
    SECOND_HOP_K,
    HNSWIndex,
    PathEntry,
    PoolRecord,
    d1_retain,
    d1_retain_with_trace,
    p3_admission,
    row_support,
)


def local_rng(*parts: object) -> random.Random:
    """SPEC 17.1: Deterministic local RNG."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def hash_order(items: Sequence[str], namespace: str, seed: int, anchor_id: str) -> list[str]:
    """Protocol sampling order: SHA256(namespace|seed|anchor|object)."""
    return sorted(
        set(items),
        key=lambda object_id: (
            hashlib.sha256(
                f"{namespace}|{seed}|{anchor_id}|{object_id}".encode("utf-8")
            ).digest(),
            object_id.encode("utf-8"),
        ),
    )


class HashOrderLibrary:
    """First k of ``hash_order(library - exclude)`` without sorting the whole library."""

    def __init__(self, ids: Sequence[str]):
        self.ids = list(dict.fromkeys(ids))
        self.encoded = [object_id.encode("utf-8") for object_id in self.ids]

    def first(
        self,
        namespace: str,
        seed: int,
        anchor_id: str,
        k: int,
        exclude: Collection[str] = (),
    ) -> list[str]:
        if k <= 0:
            return []
        prefix = f"{namespace}|{seed}|{anchor_id}|".encode("utf-8")
        digests = [h.digest() for h in map(hashlib.sha256, [prefix + e for e in self.encoded])]
        # The first k legal IDs lie within the first k + |exclude| of the unfiltered order.
        head = heapq.nsmallest(k + len(exclude), zip(digests, self.encoded, range(len(self.ids))))
        return [self.ids[i] for _, _, i in head if self.ids[i] not in exclude][:k]


_HASH_LIBRARIES: dict[int, tuple[Sequence[str], HashOrderLibrary]] = {}


def _hash_library(ids: Sequence[str]) -> HashOrderLibrary:
    cached = _HASH_LIBRARIES.get(id(ids))
    if cached is None or cached[0] is not ids:
        cached = (ids, HashOrderLibrary(ids))
        _HASH_LIBRARIES[id(ids)] = cached
    return cached[1]


def stable_topk(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


def _load_matching_index(path: Path, ids: Sequence[str], vectors: np.ndarray, seed: int) -> HNSWIndex:
    index = HNSWIndex.load(path)
    vector_hash = hashlib.sha256(memoryview(np.ascontiguousarray(vectors)).cast("B")).hexdigest()
    if index.ids != list(ids) or index.seed != seed or index.vector_hash != vector_hash:
        raise ValueError(f"reused Raw HNSW index does not match current objects: {path}")
    index._vectors = np.asarray(vectors, dtype=np.float32)
    return index


def build_raw_pools_split(
    z_store: ZStore,
    row_store: Optional[RowStore],
    query_ids: Sequence[str],
    labels: Labels,
    split: str | Mapping[str, str],
    device: str = "cuda:0",
    batch_size: int = 512,
    hnsw_seed: int = 13,
    index_dir: Optional[Path] = None,
    reuse_index_dir: Optional[Path] = None,
) -> dict[str, PoolRecord]:
    """Build formal Raw pools with HNSW on all three relation families.

    ``reuse_index_dir`` loads the same-seed Raw indices saved by an earlier split
    instead of rebuilding them; their IDs, seed and vector hashes must match.
    """
    dev = torch.device(device)
    targets = list(labels.legal_targets)
    text_lib = list(labels.canonical_text)
    image_lib = list(labels.canonical_image)
    z_targets_np = z_store.rows(targets).numpy().astype(np.float32, copy=False)
    z_text_np = z_store.rows(text_lib).numpy().astype(np.float32, copy=False)
    z_image_np = z_store.rows(image_lib).numpy().astype(np.float32, copy=False)
    if reuse_index_dir is None:
        target_index = HNSWIndex(z_targets_np, targets, dim=z_store.dim, seed=hnsw_seed)
        text_index = HNSWIndex(z_text_np, text_lib, dim=z_store.dim, seed=hnsw_seed)
        image_index = HNSWIndex(z_image_np, image_lib, dim=z_store.dim, seed=hnsw_seed)
    else:
        target_index, text_index, image_index = (
            _load_matching_index(reuse_index_dir / name, ids, vectors, hnsw_seed)
            for name, ids, vectors in (
                ("targets.hnsw", targets, z_targets_np),
                ("text.hnsw", text_lib, z_text_np),
                ("image.hnsw", image_lib, z_image_np),
            )
        )
    index_meta: dict[str, object] = {
        "target_vector_hash": target_index.vector_hash,
        "text_vector_hash": text_index.vector_hash,
        "image_vector_hash": image_index.vector_hash,
        "score_space": "bilinear_ip",
        "generator": "Raw_unit_Qwen",
    }
    if index_dir is not None:
        index_meta["QT"] = target_index.save(index_dir / "targets.hnsw")
        index_meta["Q_text"] = text_index.save(index_dir / "text.hnsw")
        index_meta["Q_image"] = image_index.save(index_dir / "image.hnsw")
    index_hash = hashlib.sha256(
        json.dumps(index_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    object_vector_hash = hashlib.sha256(
        (target_index.vector_hash + text_index.vector_hash + image_index.vector_hash).encode("ascii")
    ).hexdigest()

    positions = {
        id(ids): {object_id: i for i, object_id in enumerate(ids)}
        for ids in (targets, text_lib, image_lib)
    }

    def check_scores(query: np.ndarray, hits: Sequence[tuple[str, float]], ids: Sequence[str], vectors: np.ndarray) -> None:
        pos = positions[id(ids)]
        for object_id, score in hits:
            expected = float(query @ vectors[pos[object_id]])
            if not np.isclose(score, expected, rtol=1e-4, atol=1e-5):
                raise AssertionError(f"Raw ANN score mismatch for {object_id}: {score} != {expected}")

    pools: dict[str, PoolRecord] = {}
    z_targets_gpu = torch.from_numpy(z_targets_np).to(dev)
    z_text_gpu = torch.from_numpy(z_text_np).to(dev)
    z_image_gpu = torch.from_numpy(z_image_np).to(dev)
    # HNSW search is deterministic for a fixed index and query vector, so each
    # distinct evidence object needs one checked second-hop search per split.
    second_hop_cache: dict[str, list[tuple[str, float]]] = {}

    for qid in query_ids:
            zq_np = z_store.vector(qid).numpy().astype(np.float32, copy=False)
            direct_150 = target_index.search(zq_np, CANDIDATE_BUDGET)
            direct_100 = direct_150[:DIRECT_K]
            first_text = text_index.search(zq_np, FIRST_HOP_K)
            first_image = image_index.search(zq_np, FIRST_HOP_K)
            check_scores(zq_np, direct_150, targets, z_targets_np)
            check_scores(zq_np, first_text, text_lib, z_text_np)
            check_scores(zq_np, first_image, image_lib, z_image_np)
            first_hop = {"text": first_text, "image": first_image}

            # Pre-paths grouping by target
            pre_paths: dict[str, list[PathEntry]] = defaultdict(list)
            et_overlaps: list[float] = []
            evidence_items = [
                (modality, evidence_id, first_score)
                for modality in ("text", "image")
                for evidence_id, first_score in first_hop[modality]
            ]
            exact_second_matrix = z_store.rows([item[1] for item in evidence_items]).to(dev) @ z_targets_gpu.T
            exact_second_top = torch.argsort(
                exact_second_matrix, dim=1, descending=True, stable=True
            )[:, :SECOND_HOP_K].cpu()
            for evidence_row, (m, eid, f_score) in enumerate(evidence_items):
                second_hits = second_hop_cache.get(eid)
                if second_hits is None:
                    ze_np = z_store.vector(eid).numpy().astype(np.float32, copy=False)
                    second_hits = target_index.search(ze_np, SECOND_HOP_K)
                    check_scores(ze_np, second_hits, targets, z_targets_np)
                    second_hop_cache[eid] = second_hits
                exact_second_ids = {targets[int(x)] for x in exact_second_top[evidence_row]}
                et_overlaps.append(len({t for t, _ in second_hits} & exact_second_ids) / len(second_hits))
                for tid, s_score in second_hits:
                    pre_paths[tid].append(
                        PathEntry(evidence_id=eid, modality=m, first_score=f_score, second_score=s_score)
                    )
            evidence_targets = utf8_sorted(pre_paths)

            # D1 Retention
            q_rows = row_store.get(qid) if row_store is not None else np.zeros((1, 4096), dtype=np.float32)
            retained_paths: dict[str, list[str]] = {}
            retained_coverage: dict[str, float] = {}
            d1_trace: dict[str, list[dict[str, object]]] = {}

            all_path_evidence = utf8_sorted({p.evidence_id for paths in pre_paths.values() for p in paths})
            if all_path_evidence:
                z_ev_np = z_store.rows(all_path_evidence).numpy()
                affinity = q_rows @ z_ev_np.T
                support_matrix = np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)
                support_map = {
                    eid: support_matrix[:, e_idx]
                    for e_idx, eid in enumerate(all_path_evidence)
                }
            else:
                support_map = {}

            for tid in evidence_targets:
                selected_e, cov, trace = d1_retain_with_trace(
                    pre_paths[tid],
                    support_map,
                    content_key=labels.canonical_map,
                    budget=EVIDENCE_BUDGET,
                )
                retained_paths[tid] = selected_e
                retained_coverage[tid] = cov
                d1_trace[tid] = trace

            evidence_order = sorted(
                (t for t in evidence_targets if retained_paths.get(t)),
                key=lambda t: (-retained_coverage[t], t.encode("utf-8")),
            )

            # Full U = Direct100 union all two-hop targets.
            u_list = list(dict.fromkeys([t for t, _ in direct_100] + evidence_targets))

            # Compute QT scores for all t in U
            u_positions = [z_store.index[t] for t in u_list]
            z_u = z_store.z[u_positions].to(dev)
            zq = torch.from_numpy(zq_np).to(dev)
            u_qt_scores_tensor = z_u @ zq
            qt_scores_all_u = {t: float(u_qt_scores_tensor[idx]) for idx, t in enumerate(u_list)}

            # P3 Admission
            admitted, adm_scores = p3_admission(qt_scores_all_u, evidence_order, budget=CANDIDATE_BUDGET, constant=60)
            qt_order = sorted(qt_scores_all_u, key=lambda t: (-qt_scores_all_u[t], t.encode("utf-8")))
            qt_ranks = {t: j + 1 for j, t in enumerate(qt_order)}
            d1_ranks = {t: j + 1 for j, t in enumerate(evidence_order)}
            matched_c = target_index.search(zq_np, min(CANDIDATE_BUDGET, len(u_list)))
            matched_u = target_index.search(zq_np, len(u_list))

            qt_exact_scores = z_targets_gpu @ zq
            text_exact_scores = z_text_gpu @ zq
            image_exact_scores = z_image_gpu @ zq
            qt128_idx = stable_topk(qt_exact_scores, 128)
            text128_idx = stable_topk(text_exact_scores, 128)
            image128_idx = stable_topk(image_exact_scores, 128)
            direct_exact_idx = stable_topk(qt_exact_scores, max(CANDIDATE_BUDGET, len(u_list)))
            direct_exact = [targets[int(x)] for x in direct_exact_idx]
            qt_overlap = len({t for t, _ in direct_100} & set(direct_exact[:DIRECT_K])) / max(1, len(direct_100))
            text_exact20 = {text_lib[int(x)] for x in text128_idx[:FIRST_HOP_K]}
            image_exact20 = {image_lib[int(x)] for x in image128_idx[:FIRST_HOP_K]}

            pools[qid] = PoolRecord(
                split=split[qid] if isinstance(split, Mapping) else split,
                query_id=qid,
                generator_id="raw",
                direct=direct_100,
                direct_exact=direct_exact,
                first_hop=first_hop,
                pre_paths=pre_paths,
                retained_paths=retained_paths,
                retained_coverage=retained_coverage,
                U=u_list,
                C150=admitted,
                qt_scores_all_U=qt_scores_all_u,
                admission_scores=adm_scores,
                D150=direct_150,
                MatchedDirectC=matched_c,
                MatchedDirectU=matched_u,
                d1_scores=retained_coverage,
                qt_ranks=qt_ranks,
                d1_ranks=d1_ranks,
                object_vector_hash=object_vector_hash,
                index_hash=index_hash,
                ann_exact_overlap={
                    "QT_D100": qt_overlap,
                    "Q_text": len({e for e, _ in first_text} & text_exact20) / max(1, len(first_text)),
                    "Q_image": len({e for e, _ in first_image} & image_exact20) / max(1, len(first_image)),
                    "ET_mean": float(np.mean(et_overlaps)) if et_overlaps else 0.0,
                },
                training_exact={
                    "RawQT128": [targets[int(x)] for x in qt128_idx],
                    "RawQE128": {
                        "text": [text_lib[int(x)] for x in text128_idx],
                        "image": [image_lib[int(x)] for x in image128_idx],
                    },
                },
                d1_trace=d1_trace,
            )
    return pools


def build_raw_et128_exact(
    z_store: ZStore,
    labels: Labels,
    device: str = "cuda:0",
    batch_size: int = 128,
    anchors: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Independent exact RawET128 for every train witness evidence anchor."""
    dev = torch.device(device)
    targets = list(labels.legal_targets)
    anchors = utf8_sorted(anchors if anchors is not None else labels.epos)
    z_targets = z_store.rows(targets).to(dev)
    result: dict[str, list[str]] = {}
    with torch.no_grad():
        for start in range(0, len(anchors), batch_size):
            batch_ids = anchors[start : start + batch_size]
            scores = z_store.rows(batch_ids).to(dev) @ z_targets.T
            top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :128].cpu()
            for row, evidence_id in enumerate(batch_ids):
                result[evidence_id] = [targets[int(i)] for i in top[row]]
    return result


def build_ta_records(
    query_id: str,
    raw_pool: PoolRecord,
    labels: Labels,
    seed: int,
    raw_et128: Mapping[str, Sequence[str]],
) -> dict:
    """Build candidate lists for T_A (SPEC 11.1 - 11.3, 12)."""
    q_entry = labels.queries[query_id]
    g_targets = set(q_entry["G"])
    # 1. QT: RawQT128 union RawC150 union G_q union U32_T
    raw_qt128 = list(raw_pool.training_exact["RawQT128"])
    raw_c150 = list(raw_pool.C150)
    u32_t = _hash_library(labels.legal_targets).first("UNIFORM_T", seed, query_id, 32, g_targets)
    qt_candidates = utf8_sorted(set(raw_qt128) | set(raw_c150) | g_targets | set(u32_t))

    # 2. QE: text and image
    qe_lists: dict[str, list[str]] = {}
    for m in ("text", "image"):
        qpos_m = q_entry["Qpos"][m]
        raw_qe128 = list(raw_pool.training_exact["RawQE128"][m])
        u32_e = _hash_library(labels.library(f"Q_{m}")).first(
            "UNIFORM_E", seed, f"{query_id}|{m}", 32, set(qpos_m)
        )
        c_qe = utf8_sorted(set(qpos_m) | set(raw_qe128) | set(u32_e))
        qe_lists[m] = c_qe

    # 3. Conditional ET: for each positive witness (q, e) -> targets
    # C_QET(q, e) = RawET128(e) union RawC150(q) union G_q
    qet_lists: list[dict] = []
    w_map = q_entry["W"]  # target -> list of witness assets
    all_witnesses = {e for assets in w_map.values() for e in assets}

    for eid in sorted(all_witnesses, key=lambda x: x.encode("utf-8")):
        pos_targets = [tid for tid, assets in w_map.items() if eid in assets]
        # Ignore targets: other known gold targets for this query or other targets for e
        ignore_targets = (g_targets - set(pos_targets)) | set(labels.epos.get(eid, ())) - set(pos_targets)
        c_qet = utf8_sorted(set(list(raw_et128[eid])[:128]) | set(raw_c150) | g_targets)
        qet_lists.append(
            {
                "evidence_id": eid,
                "evidence_kind": labels.modality.get(eid, "text"),
                "positives": utf8_sorted(pos_targets),
                "ignore": utf8_sorted(ignore_targets),
                "candidates": c_qet,
            }
        )

    # 4. Support records: for each (q, t, m), up to 8 same-modality competitors
    # Exclude Protect(q, t)
    support_records: list[dict] = []
    for tid in sorted(w_map, key=lambda x: x.encode("utf-8")):
        protect = labels.protect_set(query_id, tid)
        for m in ("text", "image"):
            pos_m = [e for e in w_map[tid] if labels.modality.get(e) == m]
            if not pos_m:
                continue
            # Collect competitors:
            # 1. Natural bag in raw_pool
            bag_comps = [e for e in raw_pool.retained_paths.get(tid, []) if labels.modality.get(e) == m and e not in protect]
            # 2. Raw QE Top 128
            bag_set = set(bag_comps)
            qe_comps = [e for e in raw_pool.training_exact["RawQE128"][m] if e not in protect and e not in bag_set]
            # 3. Uniform sample
            needed = max(0, 8 - len(bag_comps) - len(qe_comps))
            sample_m = _hash_library(labels.library(f"Q_{m}")).first(
                "UNIFORM_E", seed, f"{query_id}|{tid}|{m}", needed,
                protect | set(bag_comps) | set(qe_comps),
            )
            chosen_bag = bag_comps[:8]
            chosen_qe = qe_comps[: max(0, 8 - len(chosen_bag))]
            chosen_uniform = sample_m[: max(0, 8 - len(chosen_bag) - len(chosen_qe))]
            competitors = chosen_bag + chosen_qe + chosen_uniform
            if competitors:
                support_records.append(
                    {
                        "target_id": tid,
                        "fixed_QT": [query_id, tid],
                        "modality": m,
                        "positives": pos_m,
                        "competitors": competitors,
                        "competitor_sources": {
                            "natural_bag": chosen_bag,
                            "RawQE128": chosen_qe,
                            "uniform": chosen_uniform,
                        },
                        "protect": utf8_sorted(protect),
                        "screening": {
                            "natural_bag_protected": sum(
                                e in protect for e in raw_pool.retained_paths.get(tid, [])
                                if labels.modality.get(e) == m
                            ),
                            "RawQE128_protected": sum(
                                e in protect for e in raw_pool.training_exact["RawQE128"][m]
                            ),
                            "shortfall": max(0, 8 - len(competitors)),
                        },
                    }
                )

    return {
        "query_id": query_id,
        "qt_candidates": qt_candidates,
        "qe_candidates": qe_lists,
        "qet_lists": qet_lists,
        "support_records": support_records,
    }


def build_tb_records(
    query_id: str,
    raw_pool: PoolRecord,
    labels: Labels,
    seed: int,
) -> dict:
    """Build candidate lists for T_B (SPEC 11.4 - 11.5, 13).
    
    C^B(q) = RawU(q) union RawDirect150(q) union G_q union U32_T(q).
    """
    q_entry = labels.queries[query_id]
    g_targets = set(q_entry["G"])
    raw_u = list(raw_pool.U)
    raw_d150 = [t for t, _ in raw_pool.D150]
    u32_t = _hash_library(labels.legal_targets).first("UNIFORM_T", seed, query_id, 32, g_targets)

    cb = utf8_sorted(set(raw_u) | set(raw_d150) | g_targets | set(u32_t))

    # Natural bag for each target in cb from raw_pool
    natural_bags = {t: list(raw_pool.retained_paths.get(t, [])) for t in cb}

    # Support records for positive witnesses
    w_map = q_entry["W"]
    support_records: list[dict] = []
    for tid in sorted(w_map, key=lambda x: x.encode("utf-8")):
        protect = labels.protect_set(query_id, tid)
        for m in ("text", "image"):
            pos_m = [e for e in w_map[tid] if labels.modality.get(e) == m]
            if not pos_m:
                continue
            bag_comps = [e for e in raw_pool.retained_paths.get(tid, []) if labels.modality.get(e) == m and e not in protect]
            bag_set = set(bag_comps)
            qe_comps = [e for e in raw_pool.training_exact["RawQE128"][m] if e not in protect and e not in bag_set]
            needed = max(0, 8 - len(bag_comps) - len(qe_comps))
            sample_m = _hash_library(labels.library(f"Q_{m}")).first(
                "UNIFORM_E", seed, f"{query_id}|{tid}|{m}", needed,
                protect | set(bag_comps) | set(qe_comps),
            )
            chosen_bag = bag_comps[:8]
            chosen_qe = qe_comps[: max(0, 8 - len(chosen_bag))]
            chosen_uniform = sample_m[: max(0, 8 - len(chosen_bag) - len(chosen_qe))]
            competitors = chosen_bag + chosen_qe + chosen_uniform
            if competitors:
                support_records.append(
                    {
                        "target_id": tid,
                        "fixed_QT": [query_id, tid],
                        "modality": m,
                        "positives": pos_m,
                        "competitors": competitors,
                        "competitor_sources": {
                            "natural_bag": chosen_bag,
                            "RawQE128": chosen_qe,
                            "uniform": chosen_uniform,
                        },
                        "protect": utf8_sorted(protect),
                        "screening": {
                            "natural_bag_protected": sum(
                                e in protect for e in raw_pool.retained_paths.get(tid, [])
                                if labels.modality.get(e) == m
                            ),
                            "RawQE128_protected": sum(
                                e in protect for e in raw_pool.training_exact["RawQE128"][m]
                            ),
                            "shortfall": max(0, 8 - len(competitors)),
                        },
                    }
                )

    return {
        "query_id": query_id,
        "targets": cb,
        "positives": list(g_targets),
        "natural_bags": natural_bags,
        "support_records": support_records,
    }


def build_c1_edge_lists(
    labels: Labels,
    raw_pools: Mapping[str, PoolRecord],
    z_store: ZStore,
    seed: int,
    raw_et128: Mapping[str, Sequence[str]],
    device: str = "cuda:0",
) -> list[dict]:
    """Build Student C1 edge lists: 32 hard + 31 uniform competitors (SPEC 15.1)."""
    edge_lists: list[dict] = []
    dev = torch.device(device)
    for record in labels.edge_anchors:
        item_id = record["item_id"]
        rel = record["relation"]
        anchor = record["anchor_id"]
        positives = list(record["positive_ids"])
        pset = set(positives)
        library = labels.library(rel)
        # 1. 32 Hard competitors
        if rel == "QT":
            pool = raw_pools.get(anchor)
            hard_pool = [t for t in pool.training_exact["RawQT128"] if t not in pset] if pool else []
        elif rel in ("Q_text", "Q_image"):
            pool = raw_pools.get(anchor)
            m = rel.split("_")[1]
            hard_pool = [e for e in pool.training_exact["RawQE128"][m] if e not in pset] if pool else []
        else:
            hard_pool = [t for t in raw_et128[anchor] if t not in pset]

        if len(hard_pool) < 32:
            library_vectors = z_store.rows(library).to(dev)
            anchor_vector = z_store.vector(anchor).to(dev)
            exact = library_vectors @ anchor_vector
            full_order = stable_topk(exact, len(library))
            hard_pool = [library[int(i)] for i in full_order if library[int(i)] not in pset]

        hard_32 = hard_pool[:32]

        # 2. 31 Uniform competitors
        exclude = pset | set(hard_32)
        namespace = "UNIFORM_T" if rel in ("QT", "text_T", "image_T") else "UNIFORM_E"
        uniform_31 = _hash_library(library).first(namespace, seed, f"{rel}|{anchor}", 31, exclude)

        candidates = utf8_sorted(set(positives) | set(hard_32) | set(uniform_31))

        edge_lists.append(
            {
                "item_id": item_id,
                "relation": rel,
                "anchor_id": anchor,
                "positives": positives,
                "candidates": candidates,
                "hard": hard_32,
                "uniform": uniform_31,
                "hard_shortfall": max(0, 32 - len(hard_32)),
                "uniform_shortfall": max(0, 31 - len(uniform_31)),
            }
        )

    return edge_lists


def build_c2_shared_graph(
    raw_pools: Mapping[str, PoolRecord],
    c1_pools: Mapping[str, PoolRecord],
    selected_c1: NativeStudent,
    z_store: ZStore,
    row_store: RowStore,
    labels: Labels,
    query_ids: Sequence[str],
    prepaths_path: Path,
    device: str = "cuda:0",
) -> list[dict]:
    """Merge complete Raw/C1 prepaths, rescore with C1, then run one shared D1.

    Returns the C2 records. The per-path audit rows (~3k per query, >30 GiB as
    Python objects on the full train split) are streamed to ``prepaths_path``
    query by query instead of being returned.
    """
    dev = torch.device(device)
    selected_c1.to(dev).eval()
    records: list[dict] = []
    started = time.time()

    def audit_rows():
        for query_id in query_ids:
            raw_pool = raw_pools[query_id]
            c1_pool = c1_pools[query_id]
            positives = set(labels.queries[query_id]["G"])
            targets = utf8_sorted(set(raw_pool.U) | set(c1_pool.U) | positives)

            merged: dict[tuple[str, str], dict[str, object]] = {}
            for source, pool in (("Raw", raw_pool), ("NativeC1", c1_pool)):
                for target_id, paths in pool.pre_paths.items():
                    for path in paths:
                        evidence_id = labels.canonical_map[path.evidence_id]
                        key = (target_id, evidence_id)
                        item = merged.setdefault(
                            key,
                            {
                                "target_id": target_id,
                                "evidence_id": evidence_id,
                                "modality": labels.modality[evidence_id],
                                "sources": set(),
                            },
                        )
                        item["sources"].add(source)

            evidence_ids = utf8_sorted({e for _, e in merged})
            target_pos = {target_id: i for i, target_id in enumerate(targets)}
            evidence_pos = {evidence_id: i for i, evidence_id in enumerate(evidence_ids)}
            uq = selected_c1.u("table", z_store.vector(query_id).to(dev))
            ut = selected_c1.u("table", z_store.rows(targets).to(dev))
            qe = torch.empty(len(evidence_ids), device=dev)
            projected_et = torch.empty(len(evidence_ids), selected_c1.dim, device=dev)
            for modality in ("text", "image"):
                ids = [e for e in evidence_ids if labels.modality[e] == modality]
                if not ids:
                    continue
                positions = torch.tensor([evidence_pos[e] for e in ids], device=dev)
                ue = selected_c1.u(modality, z_store.rows(ids).to(dev))
                qe[positions] = (uq @ selected_c1.R[f"Q_{modality}"] * ue).sum(dim=-1)
                projected_et[positions] = ue @ selected_c1.R[f"{modality}_T"]

            by_target: dict[str, list[PathEntry]] = defaultdict(list)
            rescored: dict[tuple[str, str], float] = {}
            for (target_id, evidence_id), item in merged.items():
                e_i = evidence_pos[evidence_id]
                t_i = target_pos[target_id]
                first = float(qe[e_i])
                second = float((projected_et[e_i] * ut[t_i]).sum())
                entry = PathEntry(evidence_id, str(item["modality"]), first, second)
                by_target[target_id].append(entry)
                rescored[(target_id, evidence_id)] = entry.raw_path_score

            q_rows = row_store.get(query_id)
            if evidence_ids:
                affinity = q_rows @ z_store.rows(evidence_ids).numpy().T
                supports = np.clip((affinity + 1.0) / 2.0, 0.0, 1.0)
                support_map = {e: supports[:, evidence_pos[e]] for e in evidence_ids}
            else:
                support_map = {}

            retained: dict[str, list[str]] = {}
            d1_scores: dict[str, float] = {}
            d1_trace_by_target: dict[str, list[dict[str, object]]] = {}
            for target_id in targets:
                selected, d1, trace = d1_retain_with_trace(
                    by_target.get(target_id, []),
                    support_map,
                    content_key=labels.canonical_map,
                    top_l=16,
                    budget=4,
                )
                retained[target_id] = selected
                d1_scores[target_id] = d1
                for item in trace:
                    item["target_id"] = target_id
                d1_trace_by_target[target_id] = trace

            for (target_id, evidence_id), item in sorted(
                merged.items(), key=lambda pair: (pair[0][0].encode("utf-8"), pair[0][1].encode("utf-8"))
            ):
                yield {
                    "schema_version": "4.1.0",
                    "query_id": query_id,
                    "target_id": target_id,
                    "evidence_id": evidence_id,
                    "modality": item["modality"],
                    "sources": utf8_sorted(item["sources"]),
                    "c1_path_score": rescored[(target_id, evidence_id)],
                    "retained": evidence_id in retained[target_id],
                    "d1_target_score": d1_scores[target_id],
                    "d1_trace": next(
                        (trace_item for trace_item in d1_trace_by_target[target_id]
                         if trace_item["evidence_id"] == evidence_id),
                        None,
                    ),
                }

            evidence_targets = [t for t in targets if retained[t]]
            records.append(
                {
                    "record_id": query_id,
                    "query_id": query_id,
                    "targets": targets,
                    "positives": utf8_sorted(positives),
                    "natural_bags": retained,
                    "evidence_targets": evidence_targets,
                    "evidence_positive_mask": [t in positives for t in evidence_targets],
                    "d1_scores": d1_scores,
                    "source_graph": "RawU_union_selectedNativeC1U_union_G",
                }
            )
            if len(records) % 500 == 0 or len(records) == len(query_ids):
                print(
                    f"[C2 shared graph] {len(records)}/{len(query_ids)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    with torch.no_grad():
        write_jsonl_gz(prepaths_path, audit_rows())
    return records

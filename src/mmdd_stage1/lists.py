"""Training-list builders for Stage-1 CQET: Raw pools, T_A / T_B records, C1 edge lists, C2 graph."""
from __future__ import annotations

import hashlib
import heapq
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Collection, Mapping, Optional, Sequence

import torch

from . import SCHEMA_VERSION
from .data import utf8_sorted, write_jsonl_gz
from .features import RowStore, ZStore
from .labels import Labels
from .models import NativeStudent
from .retrieval import (
    PathEntry,
    PoolRecord,
    build_pools,
    d1_retain_with_trace,
    row_support,
    stable_topk,
)

HARD_COMPETITORS = 32
UNIFORM_COMPETITORS = 31
SUPPORT_COMPETITORS = 8


def local_rng(*parts: object) -> random.Random:
    """SPEC 17.1: Deterministic local RNG."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def hash_order(items: Sequence[str], namespace: str, seed: int, anchor_id: str) -> list[str]:
    """Protocol sampling order: SHA256(namespace|seed|anchor|object)."""
    return sorted(
        set(items),
        key=lambda object_id: (
            hashlib.sha256(f"{namespace}|{seed}|{anchor_id}|{object_id}".encode("utf-8")).digest(),
            object_id.encode("utf-8"),
        ),
    )


class HashOrderLibrary:
    """First k of ``hash_order(library - exclude)`` without sorting the whole library."""

    def __init__(self, ids: Sequence[str]):
        self.ids = list(dict.fromkeys(ids))
        self.encoded = [object_id.encode("utf-8") for object_id in self.ids]

    def first(self, namespace: str, seed: int, anchor_id: str, k: int, exclude: Collection[str] = ()) -> list[str]:
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


def build_raw_pools_split(
    z_store: ZStore,
    row_store: Optional[RowStore],
    query_ids: Sequence[str],
    labels: Labels,
    split: str | Mapping[str, str],
    device: str = "cuda:0",
    hnsw_seed: int = 13,
    index_dir: Optional[Path] = None,
    reuse_index_dir: Optional[Path] = None,
) -> dict[str, PoolRecord]:
    """Raw pools: ``build_pools`` in the frozen Qwen space, with the exact top-128 training lists."""
    return build_pools(
        z_store, row_store, query_ids, labels, split, student=None, generator_id="raw", hnsw_seed=hnsw_seed,
        device=device, index_dir=index_dir, reuse_index_dir=reuse_index_dir, training_exact=True,
    )


def build_raw_et128_exact(
    z_store: ZStore,
    labels: Labels,
    device: str = "cuda:0",
    batch_size: int = 128,
    anchors: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Exact top-128 targets of every train witness evidence (default: all of ``labels.epos``)."""
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


def _support_records(query_id: str, raw_pool: PoolRecord, labels: Labels, seed: int) -> list[dict]:
    """Per (gold target, modality) with witnesses: up to 8 same-modality competitors outside
    Protect(q, t), taken from the natural bag, then RawQE128, then a uniform sample."""
    w_map = labels.queries[query_id]["W"]
    records: list[dict] = []
    for tid in sorted(w_map, key=lambda x: x.encode("utf-8")):
        protect = labels.protect_set(query_id, tid)
        bag = raw_pool.retained_paths.get(tid, [])
        for m in ("text", "image"):
            pos_m = [e for e in w_map[tid] if labels.modality.get(e) == m]
            if not pos_m:
                continue
            bag_m = [e for e in bag if labels.modality.get(e) == m]
            bag_comps = [e for e in bag_m if e not in protect]
            qe_m = raw_pool.training_exact["RawQE128"][m]
            qe_comps = [e for e in qe_m if e not in protect and e not in bag_comps]
            needed = max(0, SUPPORT_COMPETITORS - len(bag_comps) - len(qe_comps))
            sample_m = _hash_library(labels.library(f"Q_{m}")).first(
                "UNIFORM_E", seed, f"{query_id}|{tid}|{m}", needed, protect | set(bag_comps) | set(qe_comps),
            )
            chosen_bag = bag_comps[:SUPPORT_COMPETITORS]
            chosen_qe = qe_comps[: max(0, SUPPORT_COMPETITORS - len(chosen_bag))]
            chosen_uniform = sample_m[: max(0, SUPPORT_COMPETITORS - len(chosen_bag) - len(chosen_qe))]
            competitors = chosen_bag + chosen_qe + chosen_uniform
            if competitors:
                records.append({
                    "target_id": tid,
                    "fixed_QT": [query_id, tid],
                    "modality": m,
                    "positives": pos_m,
                    "competitors": competitors,
                    "competitor_sources": {"natural_bag": chosen_bag, "RawQE128": chosen_qe, "uniform": chosen_uniform},
                    "protect": utf8_sorted(protect),
                    "screening": {
                        "natural_bag_protected": sum(e in protect for e in bag_m),
                        "RawQE128_protected": sum(e in protect for e in qe_m),
                        "shortfall": max(0, SUPPORT_COMPETITORS - len(competitors)),
                    },
                })
    return records


def build_ta_records(
    query_id: str,
    raw_pool: PoolRecord,
    labels: Labels,
    seed: int,
    raw_et128: Mapping[str, Sequence[str]],
) -> dict:
    """T_A lists (SPEC 11.1-11.3, 12): QT, QE per modality, conditional QET per witness, support."""
    q_entry = labels.queries[query_id]
    g_targets = set(q_entry["G"])
    raw_c150 = list(raw_pool.C150)
    u32_t = _hash_library(labels.legal_targets).first("UNIFORM_T", seed, query_id, 32, g_targets)
    qt_candidates = utf8_sorted(set(raw_pool.training_exact["RawQT128"]) | set(raw_c150) | g_targets | set(u32_t))

    qe_lists: dict[str, list[str]] = {}
    for m in ("text", "image"):
        qpos_m = q_entry["Qpos"][m]
        u32_e = _hash_library(labels.library(f"Q_{m}")).first("UNIFORM_E", seed, f"{query_id}|{m}", 32, set(qpos_m))
        qe_lists[m] = utf8_sorted(set(qpos_m) | set(raw_pool.training_exact["RawQE128"][m]) | set(u32_e))

    # C_QET(q, e) = RawET128(e) | RawC150(q) | G_q; other gold targets of q and other targets of e are ignored.
    w_map = q_entry["W"]
    qet_lists: list[dict] = []
    for eid in utf8_sorted({e for assets in w_map.values() for e in assets}):
        pos_targets = [tid for tid, assets in w_map.items() if eid in assets]
        ignore_targets = (g_targets | set(labels.epos.get(eid, ()))) - set(pos_targets)
        qet_lists.append({
            "evidence_id": eid,
            "evidence_kind": labels.modality.get(eid, "text"),
            "positives": utf8_sorted(pos_targets),
            "ignore": utf8_sorted(ignore_targets),
            "candidates": utf8_sorted(set(list(raw_et128[eid])[:128]) | set(raw_c150) | g_targets),
        })
    return {
        "query_id": query_id,
        "qt_candidates": qt_candidates,
        "qe_candidates": qe_lists,
        "qet_lists": qet_lists,
        "support_records": _support_records(query_id, raw_pool, labels, seed),
    }


def build_tb_records(
    query_id: str,
    raw_pool: PoolRecord,
    labels: Labels,
    seed: int,
) -> dict:
    """T_B list (SPEC 11.4-11.5, 13): C^B(q) = RawU | RawDirect150 | G_q | U32_T with natural bags."""
    q_entry = labels.queries[query_id]
    g_targets = set(q_entry["G"])
    u32_t = _hash_library(labels.legal_targets).first("UNIFORM_T", seed, query_id, 32, g_targets)
    cb = utf8_sorted(set(raw_pool.U) | {t for t, _ in raw_pool.D150} | g_targets | set(u32_t))
    return {
        "query_id": query_id,
        "targets": cb,
        "positives": utf8_sorted(g_targets),
        "natural_bags": {t: list(raw_pool.retained_paths.get(t, [])) for t in cb},
        "support_records": _support_records(query_id, raw_pool, labels, seed),
    }


def build_c1_edge_lists(
    labels: Labels,
    raw_pools: Mapping[str, PoolRecord],
    z_store: ZStore,
    seed: int,
    raw_et128: Mapping[str, Sequence[str]],
    device: str = "cuda:0",
) -> list[dict]:
    """Student C1 edge lists (SPEC 15.1): positives + 32 hard + 31 uniform competitors per anchor."""
    edge_lists: list[dict] = []
    dev = torch.device(device)
    for record in labels.edge_anchors:
        rel, anchor = record["relation"], record["anchor_id"]
        positives = list(record["positive_ids"])
        pset = set(positives)
        library = labels.library(rel)
        pool = raw_pools.get(anchor)
        if rel == "QT":
            hard_pool = [t for t in pool.training_exact["RawQT128"] if t not in pset] if pool else []
        elif rel in ("Q_text", "Q_image"):
            hard_pool = [e for e in pool.training_exact["RawQE128"][rel.split("_")[1]] if e not in pset] if pool else []
        else:
            hard_pool = [t for t in raw_et128[anchor] if t not in pset]
        if len(hard_pool) < HARD_COMPETITORS:
            exact = z_store.rows(library).to(dev) @ z_store.vector(anchor).to(dev)
            hard_pool = [library[int(i)] for i in stable_topk(exact, len(library)) if library[int(i)] not in pset]
        hard = hard_pool[:HARD_COMPETITORS]
        namespace = "UNIFORM_T" if rel in ("QT", "text_T", "image_T") else "UNIFORM_E"
        uniform = _hash_library(library).first(namespace, seed, f"{rel}|{anchor}", UNIFORM_COMPETITORS, pset | set(hard))
        edge_lists.append({
            "item_id": record["item_id"],
            "relation": rel,
            "anchor_id": anchor,
            "positives": positives,
            "candidates": utf8_sorted(set(positives) | set(hard) | set(uniform)),
            "hard": hard,
            "uniform": uniform,
            "hard_shortfall": max(0, HARD_COMPETITORS - len(hard)),
            "uniform_shortfall": max(0, UNIFORM_COMPETITORS - len(uniform)),
        })
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
    """Merge complete Raw/C1 prepaths, rescore with the selected C1, then run one shared D1.

    Returns the C2 records. The per-path audit rows (~3k per query, >30 GiB as Python objects
    on the full train split) are streamed to ``prepaths_path`` query by query instead of being
    returned.
    """
    dev = torch.device(device)
    selected_c1.to(dev).eval()
    records: list[dict] = []
    started = time.time()

    def audit_rows():
        for query_id in query_ids:
            raw_pool, c1_pool = raw_pools[query_id], c1_pools[query_id]
            positives = set(labels.queries[query_id]["G"])
            targets = utf8_sorted(set(raw_pool.U) | set(c1_pool.U) | positives)
            sources: dict[tuple[str, str], set[str]] = defaultdict(set)
            for source, pool in (("Raw", raw_pool), ("NativeC1", c1_pool)):
                for target_id, paths in pool.pre_paths.items():
                    for path in paths:
                        sources[(target_id, labels.canonical_map[path.evidence_id])].add(source)
            merged = sorted(sources, key=lambda pair: (pair[0].encode("utf-8"), pair[1].encode("utf-8")))
            evidence_ids = utf8_sorted({e for _, e in merged})
            target_pos = {target_id: i for i, target_id in enumerate(targets)}
            evidence_pos = {evidence_id: i for i, evidence_id in enumerate(evidence_ids)}

            # Rescore every merged (target, evidence) path with the selected C1 Student, batched.
            uq = selected_c1.u("table", z_store.vector(query_id).to(dev))
            ut = selected_c1.u("table", z_store.rows(targets).to(dev))
            qe = torch.empty(len(evidence_ids), device=dev)
            projected_et = torch.empty(len(evidence_ids), selected_c1.dim, device=dev)
            for modality in ("text", "image"):
                ids = [e for e in evidence_ids if labels.modality[e] == modality]
                if ids:
                    positions = torch.tensor([evidence_pos[e] for e in ids], device=dev)
                    ue = selected_c1.u(modality, z_store.rows(ids).to(dev))
                    qe[positions] = (uq @ selected_c1.R[f"Q_{modality}"] * ue).sum(dim=-1)
                    projected_et[positions] = ue @ selected_c1.R[f"{modality}_T"]
            e_index = torch.tensor([evidence_pos[e] for _, e in merged], device=dev)
            t_index = torch.tensor([target_pos[t] for t, _ in merged], device=dev)
            first = qe[e_index].tolist()
            second = (projected_et[e_index] * ut[t_index]).sum(dim=-1).tolist()
            by_target: dict[str, list[PathEntry]] = defaultdict(list)
            rescored: dict[tuple[str, str], float] = {}
            for (target_id, evidence_id), first_score, second_score in zip(merged, first, second):
                entry = PathEntry(evidence_id, labels.modality[evidence_id], first_score, second_score)
                by_target[target_id].append(entry)
                rescored[(target_id, evidence_id)] = entry.raw_path_score

            support_map = {}
            if evidence_ids:
                support = row_support(row_store.get(query_id), z_store.rows(evidence_ids).numpy())
                support_map = {e: support[:, evidence_pos[e]] for e in evidence_ids}
            retained: dict[str, list[str]] = {}
            d1_scores: dict[str, float] = {}
            trace_by_pair: dict[tuple[str, str], dict] = {}
            for target_id in targets:
                retained[target_id], d1_scores[target_id], trace = d1_retain_with_trace(
                    by_target.get(target_id, []), support_map, content_key=labels.canonical_map,
                )
                for item in trace:
                    trace_by_pair[(target_id, item["evidence_id"])] = {**item, "target_id": target_id}

            for target_id, evidence_id in merged:
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "query_id": query_id,
                    "target_id": target_id,
                    "evidence_id": evidence_id,
                    "modality": labels.modality[evidence_id],
                    "sources": utf8_sorted(sources[(target_id, evidence_id)]),
                    "c1_path_score": rescored[(target_id, evidence_id)],
                    "retained": evidence_id in retained[target_id],
                    "d1_target_score": d1_scores[target_id],
                    "d1_trace": trace_by_pair.get((target_id, evidence_id)),
                }

            evidence_targets = [t for t in targets if retained[t]]
            records.append({
                "record_id": query_id,
                "query_id": query_id,
                "targets": targets,
                "positives": utf8_sorted(positives),
                "natural_bags": retained,
                "evidence_targets": evidence_targets,
                "evidence_positive_mask": [t in positives for t in evidence_targets],
                "d1_scores": d1_scores,
                "source_graph": "RawU_union_selectedNativeC1U_union_G",
            })
            if len(records) % 500 == 0 or len(records) == len(query_ids):
                print(f"[C2 shared graph] {len(records)}/{len(query_ids)} elapsed={time.time() - started:.1f}s", flush=True)

    with torch.no_grad():
        write_jsonl_gz(prepaths_path, audit_rows())
    return records

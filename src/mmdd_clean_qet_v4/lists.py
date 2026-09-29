"""Candidate list builders and witness competitor extraction for CLEAN-QET v4.0."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import Paths
from .data import iter_jsonl, read_json, sha256_file, utf8_sorted, write_json
from .features import RowStore, ZStore
from .labels import Labels
from .retrieval import (
    CANDIDATE_BUDGET,
    DIRECT_K,
    EVIDENCE_BUDGET,
    FIRST_HOP_K,
    SECOND_HOP_K,
    PathEntry,
    PoolRecord,
    d1_retain,
    p3_admission,
    row_support,
)


def local_rng(*parts: object) -> random.Random:
    """SPEC 17.1: Deterministic local RNG."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def stable_topk(scores: Tensor, k: int) -> Tensor:
    k = min(k, scores.numel())
    return torch.argsort(scores, descending=True, stable=True)[:k]


def build_raw_pools_split(
    z_store: ZStore,
    row_store: Optional[RowStore],
    query_ids: Sequence[str],
    labels: Labels,
    split: str,
    device: str = "cuda:0",
    batch_size: int = 512,
) -> dict[str, PoolRecord]:
    """Compute exact inner product raw candidate pools for query_ids."""
    dev = torch.device(device)
    targets = list(labels.legal_targets)
    z_targets = z_store.rows(targets).to(dev)  # (N_target, 4096)

    text_lib = list(labels.canonical_text)
    image_lib = list(labels.canonical_image)
    z_text = z_store.rows(text_lib).to(dev)
    z_image = z_store.rows(image_lib).to(dev)

    pools: dict[str, PoolRecord] = {}

    for b_start in range(0, len(query_ids), batch_size):
        b_qids = query_ids[b_start : b_start + batch_size]
        z_queries = z_store.rows(b_qids).to(dev)  # (B, 4096)

        # 1. Direct scores: (B, N_target)
        qt_matrix = z_queries @ z_targets.T
        # 2. First hop text: (B, N_text)
        q_text_matrix = z_queries @ z_text.T
        # 3. First hop image: (B, N_image)
        q_image_matrix = z_queries @ z_image.T

        # Collect all first-hop evidence across this batch
        batch_first_text = []
        batch_first_image = []
        for i in range(len(b_qids)):
            t_scores = q_text_matrix[i]
            t_top_idx = stable_topk(t_scores, 128)[:FIRST_HOP_K]
            batch_first_text.append([(text_lib[int(idx)], float(t_scores[idx])) for idx in t_top_idx])

            img_scores = q_image_matrix[i]
            img_top_idx = stable_topk(img_scores, 128)[:FIRST_HOP_K]
            batch_first_image.append([(image_lib[int(idx)], float(img_scores[idx])) for idx in img_top_idx])

        # Batch second hop for all unique evidence items in this batch
        all_b_evidence = list(dict.fromkeys(
            e for i in range(len(b_qids))
            for e, _ in (batch_first_text[i] + batch_first_image[i])
        ))
        second_hits: dict[str, list[tuple[str, float]]] = {}
        if all_b_evidence:
            z_ev_batch = z_store.rows(all_b_evidence).to(dev)
            sec_matrix = z_ev_batch @ z_targets.T  # (N_unique_e, N_target)
            sec_top_val, sec_top_idx = torch.topk(sec_matrix, SECOND_HOP_K, dim=1)
            sec_top_val_cpu = sec_top_val.cpu().numpy()
            sec_top_idx_cpu = sec_top_idx.cpu().numpy()
            for e_idx, eid in enumerate(all_b_evidence):
                second_hits[eid] = [
                    (targets[int(sec_top_idx_cpu[e_idx, k])], float(sec_top_val_cpu[e_idx, k]))
                    for k in range(SECOND_HOP_K)
                ]

        for i, qid in enumerate(b_qids):
            # Direct top 100
            d_scores = qt_matrix[i]
            d_top_idx = stable_topk(d_scores, 256)
            direct_100 = [(targets[int(idx)], float(d_scores[idx])) for idx in d_top_idx[:DIRECT_K]]
            direct_exact = [t for t, _ in direct_100]

            first_text = batch_first_text[i]
            first_image = batch_first_image[i]
            first_hop = {"text": first_text, "image": first_image}

            # Pre-paths grouping by target
            pre_paths: dict[str, list[PathEntry]] = defaultdict(list)
            for m in ("text", "image"):
                for eid, f_score in first_hop[m]:
                    for tid, s_score in second_hits.get(eid, []):
                        pre_paths[tid].append(
                            PathEntry(evidence_id=eid, modality=m, first_score=f_score, second_score=s_score)
                        )

            # Evidence targets order: sorted by best raw path score
            evidence_targets = sorted(
                pre_paths.keys(),
                key=lambda tid: (-max(p.raw_path_score for p in pre_paths[tid]), tid.encode("utf-8")),
            )

            # D1 Retention
            q_rows = row_store.get(qid) if row_store is not None else np.zeros((1, 4096), dtype=np.float32)
            retained_paths: dict[str, list[str]] = {}
            retained_coverage: dict[str, float] = {}

            all_path_evidence = list({p.evidence_id for paths in pre_paths.values() for p in paths})
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
                selected_e, cov = d1_retain(
                    pre_paths[tid],
                    support_map,
                    content_key=labels.canonical_map,
                    budget=EVIDENCE_BUDGET,
                )
                retained_paths[tid] = selected_e
                retained_coverage[tid] = cov

            # Full U = Direct100 union Evidence_targets
            u_list = list(dict.fromkeys([t for t, _ in direct_100] + evidence_targets))

            # Compute QT scores for all t in U
            u_positions = [z_store.index[t] for t in u_list]
            z_u = z_store.z[u_positions].to(dev)
            u_qt_scores_tensor = z_u @ z_queries[i]
            qt_scores_all_u = {t: float(u_qt_scores_tensor[idx]) for idx, t in enumerate(u_list)}

            # P3 Admission
            admitted, adm_scores = p3_admission(
                qt_scores_all_u, evidence_targets, budget=CANDIDATE_BUDGET, constant=60
            )

            pools[qid] = PoolRecord(
                split=split,
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
            )

    return pools


def build_ta_records(
    query_id: str,
    raw_pool: PoolRecord,
    labels: Labels,
    seed: int,
) -> dict:
    """Build candidate lists for T_A (SPEC 11.1 - 11.3, 12)."""
    q_entry = labels.queries[query_id]
    g_targets = set(q_entry["G"])
    rng = local_rng("TA", seed, query_id)

    # 1. QT: RawQT128 union RawC150 union G_q union U32_T
    raw_qt128 = [t for t, _ in raw_pool.direct[:128]]
    raw_c150 = list(raw_pool.C150)
    legal_remaining = [t for t in labels.legal_targets if t not in g_targets]
    u32_t = rng.sample(legal_remaining, min(32, len(legal_remaining)))
    qt_candidates = list(dict.fromkeys(raw_qt128 + raw_c150 + list(g_targets) + u32_t))
    rng.shuffle(qt_candidates)

    # 2. QE: text and image
    qe_lists: dict[str, list[str]] = {}
    for m in ("text", "image"):
        qpos_m = q_entry["Qpos"][m]
        raw_qe128 = [e for e, _ in raw_pool.first_hop.get(m, [])[:128]]
        lib_m = [e for e in labels.library(f"Q_{m}") if e not in set(qpos_m)]
        u32_e = rng.sample(lib_m, min(32, len(lib_m)))
        c_qe = list(dict.fromkeys(qpos_m + raw_qe128 + u32_e))
        rng.shuffle(c_qe)
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
        c_qet = list(dict.fromkeys(raw_c150 + list(g_targets)))
        rng.shuffle(c_qet)
        qet_lists.append(
            {
                "evidence_id": eid,
                "evidence_kind": labels.modality.get(eid, "text"),
                "positives": pos_targets,
                "ignore": list(ignore_targets),
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
            qe_comps = [e for e, _ in raw_pool.first_hop.get(m, []) if e not in protect and e not in set(bag_comps)]
            # 3. Uniform sample
            lib_m = [e for e in labels.library(f"Q_{m}") if e not in protect and e not in set(bag_comps) and e not in set(qe_comps)]
            needed = max(0, 8 - len(bag_comps) - len(qe_comps))
            sample_m = rng.sample(lib_m, min(needed, len(lib_m)))
            competitors = (bag_comps + qe_comps + sample_m)[:8]
            if competitors:
                support_records.append(
                    {
                        "target_id": tid,
                        "modality": m,
                        "positives": pos_m,
                        "competitors": competitors,
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
    rng = local_rng("TB_SHARED", seed, query_id)

    raw_u = list(raw_pool.U)
    raw_d150 = [t for t, _ in raw_pool.direct[:150]]
    legal_remaining = [t for t in labels.legal_targets if t not in g_targets]
    u32_t = rng.sample(legal_remaining, min(32, len(legal_remaining)))

    cb = list(dict.fromkeys(raw_u + raw_d150 + list(g_targets) + u32_t))
    rng.shuffle(cb)

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
            qe_comps = [e for e, _ in raw_pool.first_hop.get(m, []) if e not in protect and e not in set(bag_comps)]
            lib_m = [e for e in labels.library(f"Q_{m}") if e not in protect and e not in set(bag_comps) and e not in set(qe_comps)]
            needed = max(0, 8 - len(bag_comps) - len(qe_comps))
            sample_m = rng.sample(lib_m, min(needed, len(lib_m)))
            competitors = (bag_comps + qe_comps + sample_m)[:8]
            if competitors:
                support_records.append(
                    {
                        "target_id": tid,
                        "modality": m,
                        "positives": pos_m,
                        "competitors": competitors,
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
    device: str = "cuda:0",
) -> list[dict]:
    """Build Student C1 edge lists: 32 hard + 31 uniform competitors (SPEC 15.1)."""
    edge_lists: list[dict] = []
    dev = torch.device(device)
    targets = list(labels.legal_targets)
    z_targets = z_store.rows(targets).to(dev)

    for record in labels.edge_anchors:
        item_id = record["item_id"]
        rel = record["relation"]
        anchor = record["anchor_id"]
        positives = list(record["positive_ids"])
        pset = set(positives)
        library = labels.library(rel)
        rng = local_rng("STUDENT_C1", seed, rel, anchor)

        # 1. 32 Hard competitors
        if rel == "QT":
            pool = raw_pools.get(anchor)
            hard_pool = [t for t, _ in pool.direct if t not in pset] if pool else []
        elif rel in ("Q_text", "Q_image"):
            pool = raw_pools.get(anchor)
            m = rel.split("_")[1]
            hard_pool = [e for e, _ in pool.first_hop.get(m, []) if e not in pset] if pool else []
        else:
            # text_T or image_T: exact inner product from anchor evidence to all targets
            ez = z_store.vector(anchor).to(dev)
            scores = z_targets @ ez
            top_idx = stable_topk(scores, 64)
            hard_pool = [targets[int(idx)] for idx in top_idx if targets[int(idx)] not in pset]

        hard_32 = hard_pool[:32]

        # 2. 31 Uniform competitors
        exclude = pset | set(hard_32)
        remaining = [x for x in library if x not in exclude]
        uniform_31 = rng.sample(remaining, min(31, len(remaining)))

        candidates = list(dict.fromkeys(positives + hard_32 + uniform_31))
        rng.shuffle(candidates)

        edge_lists.append(
            {
                "item_id": item_id,
                "relation": rel,
                "anchor_id": anchor,
                "positives": positives,
                "candidates": candidates,
            }
        )

    return edge_lists

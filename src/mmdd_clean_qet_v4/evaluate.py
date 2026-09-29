"""Evaluation, Teacher reranking, E-swap, bootstrap statistics, and decision checks for CLEAN-QET v4.0."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .data import utf8_sorted
from .features import ObjectBank, RowStore, ZStore
from .labels import Labels
from .losses import aggregate_paths
from .models import FreshPathTeacher, NativeStudent, QTStudent
from .retrieval import (
    CANDIDATE_BUDGET,
    DIRECT_K,
    FIRST_HOP_K,
    SECOND_HOP_K,
    HNSWIndex,
    PathEntry,
    PoolRecord,
    d1_retain,
    p3_admission,
    row_support,
    stable_topk,
)


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & gold) / len(gold)


def coverage_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(ranked[:k])
    return 1.0 if bool(topk & gold) else 0.0


# ------------------------------------------------------------- Student Eval ---


def evaluate_student_retrieval(
    student: NativeStudent | QTStudent,
    z_store: ZStore,
    row_store: Optional[RowStore],
    query_ids: Sequence[str],
    labels: Labels,
    split: str,
    device: str = "cuda:0",
    hnsw_seed: int = 13,
) -> dict[str, PoolRecord]:
    """Run full student retrieval (Direct ANN/exact, first/second hop, D1, P3)."""
    dev = torch.device(device)
    student.to(dev)
    student.eval()

    is_qt_only = isinstance(student, QTStudent)

    # 1. Build HNSW index for lake targets using Student's table projection
    targets = list(labels.legal_targets)
    with torch.no_grad():
        z_targets = z_store.rows(targets).to(dev)
        if is_qt_only:
            u_targets = student.u(z_targets).cpu().numpy().astype(np.float32)
        else:
            u_targets = student.u("table", z_targets).cpu().numpy().astype(np.float32)

    hnsw_idx = HNSWIndex(u_targets, targets, dim=student.dim, seed=hnsw_seed)

    text_lib = list(labels.canonical_text)
    image_lib = list(labels.canonical_image)

    with torch.no_grad():
        if not is_qt_only:
            z_text = z_store.rows(text_lib).to(dev)
            z_image = z_store.rows(image_lib).to(dev)
            u_text = student.u("text", z_text)
            u_image = student.u("image", z_image)
            u_targets_t = torch.from_numpy(u_targets).to(dev)

    pools: dict[str, PoolRecord] = {}

    for qid in query_ids:
        with torch.no_grad():
            zq = z_store.vector(qid).to(dev)
            if is_qt_only:
                uq = student.query_vector(zq).cpu().numpy().astype(np.float32)
            else:
                uq_t = student.query_vector(zq)
                uq = uq_t.cpu().numpy().astype(np.float32)

            # Direct ANN 100
            direct_ann = hnsw_idx.search(uq, DIRECT_K)

            # Direct exact 100
            if is_qt_only:
                exact_scores = (student.u(zq) @ student.R_QT * torch.from_numpy(u_targets).to(dev)).sum(dim=-1)
            else:
                exact_scores = (student.u("table", zq) @ student.R["QT"] * u_targets_t).sum(dim=-1)
            top_exact_idx = stable_topk(exact_scores, DIRECT_K)
            direct_exact = [targets[int(i)] for i in top_exact_idx]

            first_hop = {"text": [], "image": []}
            pre_paths: dict[str, list[PathEntry]] = defaultdict(list)
            evidence_targets: list[str] = []

            if not is_qt_only:
                # First hop text & image
                # s_S(q, e) = u_q @ R @ u_e
                sq_text = (uq_t @ student.R["Q_text"] * u_text).sum(dim=-1)
                top_text_idx = stable_topk(sq_text, FIRST_HOP_K)
                first_text = [(text_lib[int(i)], float(sq_text[i])) for i in top_text_idx]

                sq_image = (uq_t @ student.R["Q_image"] * u_image).sum(dim=-1)
                top_image_idx = stable_topk(sq_image, FIRST_HOP_K)
                first_image = [(image_lib[int(i)], float(sq_image[i])) for i in top_image_idx]

                first_hop = {"text": first_text, "image": first_image}
                ev_items = [("text", eid, sc) for eid, sc in first_text] + [("image", eid, sc) for eid, sc in first_image]

                # Second hop
                for m, eid, f_sc in ev_items:
                    ze = z_store.vector(eid).to(dev)
                    ue = student.u(m, ze)
                    r_rel = student.R["text_T"] if m == "text" else student.R["image_T"]
                    s_sec = (ue @ r_rel * u_targets_t).sum(dim=-1)
                    top_sec_idx = stable_topk(s_sec, SECOND_HOP_K)
                    for s_i in top_sec_idx:
                        tid = targets[int(s_i)]
                        s_sc = float(s_sec[s_i])
                        pre_paths[tid].append(PathEntry(evidence_id=eid, modality=m, first_score=f_sc, second_score=s_sc))

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
                support_mat = row_support(q_rows, z_ev_np)
                support_map = {
                    eid: support_mat[:, e_idx]
                    for e_idx, eid in enumerate(all_path_evidence)
                }
            else:
                support_map = {}

            for tid in evidence_targets:
                sel_e, cov = d1_retain(pre_paths[tid], support_map, content_key=labels.canonical_map, budget=4)
                retained_paths[tid] = sel_e
                retained_coverage[tid] = cov

            # Full U
            u_list = list(dict.fromkeys([t for t, _ in direct_ann] + evidence_targets))

            # Compute Student QT scores for all U
            z_u = z_store.rows(u_list).to(dev)
            if is_qt_only:
                u_qt_scores_t = (student.u(zq) @ student.R_QT * student.u(z_u)).sum(dim=-1)
            else:
                u_qt_scores_t = (student.u("table", zq) @ student.R["QT"] * student.u("table", z_u)).sum(dim=-1)
            qt_scores_all_u = {t: float(u_qt_scores_t[idx]) for idx, t in enumerate(u_list)}

            # P3 Admission
            c150, adm_scores = p3_admission(qt_scores_all_u, evidence_targets, budget=CANDIDATE_BUDGET, constant=60)

            pools[qid] = PoolRecord(
                split=split,
                query_id=qid,
                generator_id="student",
                direct=direct_ann,
                direct_exact=direct_exact,
                first_hop=first_hop,
                pre_paths=pre_paths,
                retained_paths=retained_paths,
                retained_coverage=retained_coverage,
                U=u_list,
                C150=c150,
                qt_scores_all_U=qt_scores_all_u,
                admission_scores=adm_scores,
            )

    return pools


# ------------------------------------------------------------- Teacher Eval ---


def build_evidence_swap_ring(canonical_evidence_ids: Sequence[str]) -> dict[str, str]:
    """SPEC 20.1: Deterministic canonical evidence swap ring."""
    sorted_e = utf8_sorted(set(canonical_evidence_ids))
    n = len(sorted_e)
    if n <= 1:
        return {}
    return {sorted_e[i]: sorted_e[(i + 1) % n] for i in range(n)}


def evaluate_teacher_rerank(
    teacher_path: FreshPathTeacher,
    teacher_qt: FreshPathTeacher,
    bank: ObjectBank,
    pools: Mapping[str, PoolRecord],
    split_gt: Mapping[str, dict],
    labels: Labels,
    device: str = "cuda:0",
) -> dict[str, dict[str, list[str]]]:
    """Rerank Student C150 candidates under 5 policies (SPEC 19.2):
    1. T_B_QT (f0)
    2. T_B_PATH f0
    3. T_B_PATH Real
    4. T_B_PATH Swap
    5. f0 + log(1 + bag_size) control
    """
    dev = torch.device(device)
    teacher_path.to(dev)
    teacher_path.eval()
    teacher_qt.to(dev)
    teacher_qt.eval()
    bank.attach_device(dev)

    # Build canonical swap rings for text and image
    text_swap = build_evidence_swap_ring(labels.canonical_text)
    image_swap = build_evidence_swap_ring(labels.canonical_image)

    rankings: dict[str, dict[str, list[str]]] = {}
    total_pools = len(pools)
    t_start = time.time()

    with torch.no_grad():
        for q_idx, (qid, pool) in enumerate(pools.items()):
            c150 = list(pool.C150)
            if not c150:
                rankings[qid] = {pol: [] for pol in ("TB_QT", "PATH_f0", "PATH_Real", "PATH_Swap", "PATH_logbag")}
                continue

            # Build paths for real and swap
            real_paths = [(t_idx, eid) for t_idx, t in enumerate(c150) for eid in pool.retained_paths.get(t, [])]
            swap_paths = []
            for t_idx, t in enumerate(c150):
                for eid in pool.retained_paths.get(t, []):
                    ekind = bank.kind(eid)
                    ring = text_swap if ekind == "text" else image_swap
                    donor = ring.get(eid, eid)
                    swap_paths.append((t_idx, donor))

            # Fetch all tokens in single transfer
            all_obj_ids = list(dict.fromkeys([qid] + c150 + [p[1] for p in real_paths] + [p[1] for p in swap_paths]))
            tok_map = dict(zip(all_obj_ids, bank.tokens_many(all_obj_ids)))

            zq = bank.z(qid)
            cq = tok_map[qid]
            zt_matrix = bank.z_many(c150)
            ct_list = [tok_map[t] for t in c150]

            # 1. Score f0 with T_B_QT
            f0_qt, _ = teacher_qt.score_query_lists((zq, cq), (zt_matrix, ct_list), {}, [], chunk=1024)

            # 2. Score f0 and natural paths with T_B_PATH
            real_e_map = {eid: (bank.kind(eid), bank.z(eid), tok_map[eid]) for eid in set(p[1] for p in real_paths)}
            f0_path, sc_real = teacher_path.score_query_lists((zq, cq), (zt_matrix, ct_list), real_e_map, real_paths, chunk=1024)

            # 3. Score swap paths with T_B_PATH
            swap_e_map = {donor: (bank.kind(donor), bank.z(donor), tok_map[donor]) for donor in set(p[1] for p in swap_paths)}
            _, sc_swap = teacher_path.score_query_lists((zq, cq), (zt_matrix, ct_list), swap_e_map, swap_paths, chunk=1024)

            # 4. Aggregate
            if real_paths:
                t_idx_real = torch.tensor([p[0] for p in real_paths], dtype=torch.long, device=dev)
                real_t = aggregate_paths(f0_path, sc_real, t_idx_real)
            else:
                real_t = f0_path

            if swap_paths:
                t_idx_swap = torch.tensor([p[0] for p in swap_paths], dtype=torch.long, device=dev)
                swap_t = aggregate_paths(f0_path, sc_swap, t_idx_swap)
            else:
                swap_t = f0_path

            # f0 + log(1 + bag_size)
            logbag_offsets = torch.tensor([math.log(1.0 + len(pool.retained_paths.get(t, []))) for t in c150], device=dev)
            logbag_t = f0_path + logbag_offsets

            def sort_candidates(scores_tensor: Tensor) -> list[str]:
                order = sorted(range(len(c150)), key=lambda i: (-float(scores_tensor[i]), c150[i].encode("utf-8")))
                return [c150[i] for i in order]

            rankings[qid] = {
                "TB_QT": sort_candidates(f0_qt),
                "PATH_f0": sort_candidates(f0_path),
                "PATH_Real": sort_candidates(real_t),
                "PATH_Swap": sort_candidates(swap_t),
                "PATH_logbag": sort_candidates(logbag_t),
            }

            if (q_idx + 1) % 200 == 0 or (q_idx + 1) == total_pools:
                el = time.time() - t_start
                print(f"  [Teacher Rerank] {q_idx + 1}/{total_pools} queries reranked ({el:.1f}s)", flush=True)

    return rankings


# ----------------------------------------------------------------- Bootstrap ---


def paired_bootstrap(
    deltas_by_query: Mapping[str, float],
    query_to_group: Mapping[str, str],
    replicates: int = 10000,
    seed: int = 13,
) -> dict[str, Any]:
    """10,000 source-group paired bootstrap replicates (SPEC 21.3)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for qid, delta in deltas_by_query.items():
        grp = query_to_group.get(qid, qid)
        groups[grp].append(delta)

    group_keys = list(groups.keys())
    group_means = np.asarray([np.mean(groups[k]) for k in group_keys], dtype=np.float64)
    n_groups = len(group_keys)

    if n_groups == 0:
        return {"mean_delta_pp": 0.0, "ci_95": [0.0, 0.0], "wlt": (0, 0, 0)}

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_groups, size=(replicates, n_groups))
    boot_means = group_means[boot_indices].mean(axis=1) * 100.0  # percentage points

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    mean_pp = float(np.mean(boot_means))

    # W/L/T counts
    w = sum(1 for d in deltas_by_query.values() if d > 1e-6)
    l = sum(1 for d in deltas_by_query.values() if d < -1e-6)
    t = sum(1 for d in deltas_by_query.values() if abs(d) <= 1e-6)

    return {
        "mean_delta_pp": mean_pp,
        "ci_95": [ci_lower, ci_upper],
        "wlt": [w, l, t],
    }

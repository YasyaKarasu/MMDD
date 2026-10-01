"""Evaluation, Teacher reranking, E-swap, bootstrap statistics, and decision checks for Stage-1 CQET."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from . import SCHEMA_VERSION
from .execution_layout import TEACHER_INFERENCE_CHUNK
from .artifacts import pool_identity
from .data import sha256_file, utf8_sorted, write_json, write_jsonl_gz
from .features import ObjectBank, RowStore, ZStore
from .labels import Labels
from .losses import aggregate_cqet, aggregate_corrected_lse
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
    d1_retain_with_trace,
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
    """Query-level target coverage, not hit rate."""
    if not gold:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & gold) / len(gold)


def hit_rate_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return float(bool(set(ranked[:k]) & gold))


def oracle_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return min(k, len(set(ranked) & gold)) / len(gold)


def _tensor_state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


_POSITION_MAPS: dict[int, tuple[Sequence[str], dict[str, int]]] = {}


def _positions(ids: Sequence[str]) -> dict[str, int]:
    cached = _POSITION_MAPS.get(id(ids))
    if cached is None or cached[0] is not ids:
        cached = (ids, {object_id: i for i, object_id in enumerate(ids)})
        _POSITION_MAPS[id(ids)] = cached
    return cached[1]


def _check_ann_scores(
    query: np.ndarray,
    hits: Sequence[tuple[str, float]],
    ids: Sequence[str],
    vectors: np.ndarray,
) -> None:
    positions = _positions(ids)
    for object_id, ann_score in hits:
        exact = float(query @ vectors[positions[object_id]])
        if not np.isclose(ann_score, exact, rtol=1e-4, atol=1e-5):
            raise AssertionError(
                f"ANN score mismatch for {object_id}: returned={ann_score}, bilinear={exact}"
            )


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
    generator_id: str = "student",
    index_dir: Optional[str] = None,
) -> dict[str, PoolRecord]:
    """Run full student retrieval (Direct ANN/exact, first/second hop, D1, P3)."""
    dev = torch.device(device)
    student.to(dev)
    student.eval()

    is_qt_only = isinstance(student, QTStudent)

    # Build right-object indices. Every formal relation query is transformed by R.
    targets = list(labels.legal_targets)
    with torch.no_grad():
        z_targets = z_store.rows(targets).to(dev)
        if is_qt_only:
            u_targets = student.index_vectors(z_targets).cpu().numpy().astype(np.float32)
        else:
            u_targets = student.index_vectors("QT", z_targets).cpu().numpy().astype(np.float32)

    hnsw_idx = HNSWIndex(u_targets, targets, dim=student.dim, seed=hnsw_seed)

    text_lib = list(labels.canonical_text)
    image_lib = list(labels.canonical_image)
    text_idx = None
    image_idx = None
    u_text_np = np.empty((0, student.dim), dtype=np.float32)
    u_image_np = np.empty((0, student.dim), dtype=np.float32)

    with torch.no_grad():
        if not is_qt_only:
            z_text = z_store.rows(text_lib).to(dev)
            z_image = z_store.rows(image_lib).to(dev)
            u_text_np = student.index_vectors("Q_text", z_text).cpu().numpy().astype(np.float32)
            u_image_np = student.index_vectors("Q_image", z_image).cpu().numpy().astype(np.float32)
            text_idx = HNSWIndex(u_text_np, text_lib, dim=student.dim, seed=hnsw_seed)
            image_idx = HNSWIndex(u_image_np, image_lib, dim=student.dim, seed=hnsw_seed)

    index_meta: dict[str, object] = {
        "target_vector_hash": hnsw_idx.vector_hash,
        "text_vector_hash": text_idx.vector_hash if text_idx else None,
        "image_vector_hash": image_idx.vector_hash if image_idx else None,
        "model_state_hash": _tensor_state_hash(student),
        "score_space": "bilinear_ip",
    }
    if index_dir is not None:
        root = Path(index_dir)
        index_meta["QT"] = hnsw_idx.save(root / "targets.hnsw")
        if text_idx is not None and image_idx is not None:
            index_meta["Q_text"] = text_idx.save(root / "text.hnsw")
            index_meta["Q_image"] = image_idx.save(root / "image.hnsw")
    index_hash = hashlib.sha256(
        json.dumps(index_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    object_vector_hash = hashlib.sha256(
        (hnsw_idx.vector_hash + (text_idx.vector_hash if text_idx else "") +
         (image_idx.vector_hash if image_idx else "")).encode("ascii")
    ).hexdigest()

    pools: dict[str, PoolRecord] = {}
    # Fixed model and index: each distinct evidence object needs one checked second-hop search.
    second_hop_cache: dict[str, list[tuple[str, float]]] = {}

    for qid in query_ids:
        with torch.no_grad():
            zq = z_store.vector(qid).to(dev)
            if is_qt_only:
                qt_query = student.ann_query(zq).cpu().numpy().astype(np.float32)
            else:
                qt_query = student.ann_query("QT", zq).cpu().numpy().astype(np.float32)

            # Direct ANN 100 plus the fixed, independently exported D150.
            direct_150 = hnsw_idx.search(qt_query, CANDIDATE_BUDGET)
            _check_ann_scores(qt_query, direct_150, targets, u_targets)
            direct_ann = direct_150[:DIRECT_K]

            # Exact QT audit; expanded after U is known below.
            if is_qt_only:
                exact_scores = student.ann_query(zq) @ torch.from_numpy(u_targets).to(dev).T
            else:
                exact_scores = student.ann_query("QT", zq) @ torch.from_numpy(u_targets).to(dev).T

            first_hop = {"text": [], "image": []}
            pre_paths: dict[str, list[PathEntry]] = defaultdict(list)
            evidence_targets: list[str] = []

            if not is_qt_only:
                text_query = student.ann_query("Q_text", zq).cpu().numpy().astype(np.float32)
                image_query = student.ann_query("Q_image", zq).cpu().numpy().astype(np.float32)
                first_text = text_idx.search(text_query, FIRST_HOP_K)
                first_image = image_idx.search(image_query, FIRST_HOP_K)
                _check_ann_scores(text_query, first_text, text_lib, u_text_np)
                _check_ann_scores(image_query, first_image, image_lib, u_image_np)

                first_hop = {"text": first_text, "image": first_image}
                ev_items = [("text", eid, sc) for eid, sc in first_text] + [("image", eid, sc) for eid, sc in first_image]

                # Second hop
                for m, eid, f_sc in ev_items:
                    second = second_hop_cache.get(eid)
                    if second is None:
                        ze = z_store.vector(eid).to(dev)
                        relation = "text_T" if m == "text" else "image_T"
                        et_query = student.ann_query(relation, ze).cpu().numpy().astype(np.float32)
                        second = hnsw_idx.search(et_query, SECOND_HOP_K)
                        _check_ann_scores(et_query, second, targets, u_targets)
                        second_hop_cache[eid] = second
                    for tid, s_sc in second:
                        pre_paths[tid].append(PathEntry(evidence_id=eid, modality=m, first_score=f_sc, second_score=s_sc))

                evidence_targets = utf8_sorted(pre_paths)

            # D1 Retention
            q_rows = row_store.get(qid) if row_store is not None else np.zeros((1, 4096), dtype=np.float32)
            retained_paths: dict[str, list[str]] = {}
            retained_coverage: dict[str, float] = {}
            d1_trace: dict[str, list[dict[str, object]]] = {}

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
                sel_e, cov, trace = d1_retain_with_trace(
                    pre_paths[tid], support_map, content_key=labels.canonical_map, budget=4
                )
                retained_paths[tid] = sel_e
                retained_coverage[tid] = cov
                d1_trace[tid] = trace

            evidence_order = sorted(
                (t for t in evidence_targets if retained_paths.get(t)),
                key=lambda t: (-retained_coverage[t], t.encode("utf-8")),
            )

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
            if is_qt_only:
                c150 = [t for t, _ in direct_150]
                adm_scores = {t: 1.0 / (60 + i) for i, t in enumerate(c150, 1)}
            else:
                c150, adm_scores = p3_admission(qt_scores_all_u, evidence_order, budget=CANDIDATE_BUDGET, constant=60)

            qt_order = sorted(qt_scores_all_u, key=lambda t: (-qt_scores_all_u[t], t.encode("utf-8")))
            qt_ranks = {t: i + 1 for i, t in enumerate(qt_order)}
            d1_ranks = {t: i + 1 for i, t in enumerate(evidence_order)}
            matched_c = hnsw_idx.search(qt_query, min(CANDIDATE_BUDGET, len(u_list)))
            matched_u = hnsw_idx.search(qt_query, len(u_list))
            _check_ann_scores(qt_query, matched_u, targets, u_targets)
            top_exact_idx = stable_topk(exact_scores, max(CANDIDATE_BUDGET, len(u_list)))
            direct_exact = [targets[int(i)] for i in top_exact_idx]
            overlap = len(set(t for t, _ in direct_ann) & set(direct_exact[:DIRECT_K])) / max(1, len(direct_ann))

            pools[qid] = PoolRecord(
                split=split,
                query_id=qid,
                generator_id=generator_id,
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
                D150=direct_150,
                MatchedDirectC=matched_c,
                MatchedDirectU=matched_u,
                d1_scores=retained_coverage,
                qt_ranks=qt_ranks,
                d1_ranks=d1_ranks,
                object_vector_hash=object_vector_hash,
                index_hash=index_hash,
                ann_exact_overlap={"QT_D100": overlap},
                d1_trace=d1_trace,
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
                real_t = aggregate_cqet(f0_path, sc_real, t_idx_real)
            else:
                real_t = f0_path

            if swap_paths:
                t_idx_swap = torch.tensor([p[0] for p in swap_paths], dtype=torch.long, device=dev)
                swap_t = aggregate_cqet(f0_path, sc_swap, t_idx_swap)
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


def evaluate_teacher_matrix(
    teachers: Mapping[str, FreshPathTeacher],
    bank: ObjectBank,
    pools: Mapping[str, PoolRecord],
    labels: Labels,
    *,
    seed: int,
    generator: str,
    split: str,
    output_dir: Path,
    device: str = "cuda:0",
    pool_kind: str = "C150",
    direct_only: bool = False,
    split_gt: Optional[Mapping[str, dict]] = None,
    teacher_modes: Optional[Mapping[str, str]] = None,
) -> dict[str, dict[str, dict[str, dict[str, list]]]]:
    """Score the preregistered same-pool Teacher/view matrix and export raw logits."""
    dev = torch.device(device)
    text_swap = build_evidence_swap_ring(labels.canonical_text)
    image_swap = build_evidence_swap_ring(labels.canonical_image)
    result: dict[str, dict[str, dict[str, dict[str, list]]]] = defaultdict(lambda: defaultdict(dict))
    ranking_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    logit_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    swap_paths_total = 0
    swap_witness_collisions = 0

    if not direct_only:
        for query_id in utf8_sorted(pools):
            pool = pools[query_id]
            for target in pool.C150:
                for evidence_id in pool.retained_paths.get(target, ()):
                    ring = text_swap if labels.modality[evidence_id] == "text" else image_swap
                    donor = ring.get(evidence_id, evidence_id)
                    swap_paths_total += 1
                    if split_gt is not None and donor in set(
                        split_gt.get(query_id, {}).get("W", {}).get(target, ())
                    ):
                        swap_witness_collisions += 1

    for teacher_name, teacher in teachers.items():
        mode = (
            teacher_modes[teacher_name]
            if teacher_modes is not None
            else ("qt" if teacher_name == "TB_QT" else
                  "lse" if teacher_name == "TB_LSE" else "cqet")
        )
        if mode not in {"cqet", "lse", "qt"}:
            raise ValueError(f"invalid Teacher evaluation mode for {teacher_name}: {mode}")
        teacher.to(dev).eval()
        # Scoring reads bank tokens, so the bank must follow the teacher onto `dev`.
        # Training stages attach it as a side effect; a resumed run may skip them all.
        bank.attach_device(dev)
        teacher_hash = _tensor_state_hash(teacher)
        with torch.no_grad():
            for query_id in utf8_sorted(pools):
                pool = pools[query_id]
                targets = list(pool.C150)
                if not targets:
                    continue
                bags = pool.retained_paths
                real_paths = [(i, e) for i, target in enumerate(targets) for e in bags.get(target, ())]
                swap_paths = []
                for i, target in enumerate(targets):
                    for evidence_id in bags.get(target, ()):
                        ring = text_swap if labels.modality[evidence_id] == "text" else image_swap
                        donor = ring.get(evidence_id, evidence_id)
                        swap_paths.append((i, donor))
                all_ids = list(dict.fromkeys(
                    [query_id, *targets, *(e for _, e in real_paths), *(e for _, e in swap_paths)]
                ))
                token_map = dict(zip(all_ids, bank.tokens_many(all_ids)))
                q = (bank.z(query_id), token_map[query_id])
                target_values = (bank.z_many(targets), [token_map[t] for t in targets])

                if mode == "qt":
                    f0, _ = teacher.score_query_lists(q, target_values, {}, [], chunk=TEACHER_INFERENCE_CHUNK)
                    views = {"Direct": (f0, [], torch.empty(0, device=dev))}
                elif direct_only:
                    f0, _ = teacher.score_query_lists(q, target_values, {}, [], chunk=TEACHER_INFERENCE_CHUNK)
                    views = {"f0": (f0, [], torch.empty(0, device=dev))}
                else:
                    real_evidence = {
                        evidence_id: (labels.modality[evidence_id], bank.z(evidence_id), token_map[evidence_id])
                        for evidence_id in dict.fromkeys(e for _, e in real_paths)
                    }
                    f0, real_logits = teacher.score_query_lists(
                        q, target_values, real_evidence, real_paths, chunk=TEACHER_INFERENCE_CHUNK
                    )
                    if swap_paths:
                        swap_evidence = {
                            evidence_id: (labels.modality[evidence_id], bank.z(evidence_id), token_map[evidence_id])
                            for evidence_id in dict.fromkeys(e for _, e in swap_paths)
                        }
                        _swap_f0, swap_logits = teacher.score_query_lists(
                            q, target_values, swap_evidence, swap_paths, chunk=TEACHER_INFERENCE_CHUNK
                        )
                    else:
                        swap_logits = torch.empty(0, device=dev)
                    target_index_real = torch.tensor([i for i, _ in real_paths], dtype=torch.long, device=dev)
                    target_index_swap = torch.tensor([i for i, _ in swap_paths], dtype=torch.long, device=dev)
                    aggregator = aggregate_cqet if mode == "cqet" else aggregate_corrected_lse
                    real = aggregator(f0, real_logits, target_index_real)
                    swap = aggregator(f0, swap_logits, target_index_swap)
                    views = {
                        "f0": (f0, [], torch.empty(0, device=dev)),
                        "Real": (real, real_paths, real_logits),
                        "Swap": (swap, swap_paths, swap_logits),
                    }

                for view, (scores, paths, path_logits) in views.items():
                    order = sorted(
                        range(len(targets)),
                        key=lambda i: (-float(scores[i]), targets[i].encode("utf-8")),
                    )
                    ranked_ids = [targets[i] for i in order]
                    ranked_scores = [float(scores[i]) for i in order]
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        "query_id": query_id,
                        "split": split,
                        "seed": seed,
                        "generator": generator,
                        "pool_kind": pool_kind,
                        "pool_id": pool_identity(pool),
                        "teacher": teacher_name,
                        "teacher_state_hash": teacher_hash,
                        "view": view,
                        "target_ids": ranked_ids,
                        "scores": ranked_scores,
                    }
                    ranking_rows[(teacher_name, view)].append(row)
                    result[teacher_name][view][query_id] = {
                        "target_ids": ranked_ids,
                        "scores": ranked_scores,
                    }

                    path_by_target: dict[int, list[dict]] = defaultdict(list)
                    for slot, ((target_i, evidence_id), logit) in enumerate(zip(paths, path_logits)):
                        path_by_target[target_i].append(
                            {
                                "slot": slot,
                                "evidence_id": evidence_id,
                                "modality": labels.modality[evidence_id],
                                "raw_QET": float(logit),
                                "natural": view == "Real",
                                "swap": view == "Swap",
                            }
                        )
                    for target_i, target_id in enumerate(targets):
                        logit_rows[(teacher_name, view)].append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "query_id": query_id,
                                "target_id": target_id,
                                "split": split,
                                "seed": seed,
                                "generator": generator,
                                "teacher": teacher_name,
                                "teacher_state_hash": teacher_hash,
                                "view": view,
                                "f0": float(f0[target_i]),
                                "paths": path_by_target.get(target_i, []),
                                "aggregated_score": float(scores[target_i]),
                            }
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key in sorted(ranking_rows):
        teacher_name, view = key
        rankings_path = output_dir / f"rankings.{teacher_name}.{view}.jsonl.gz"
        logits_path = output_dir / f"logits.{teacher_name}.{view}.jsonl.gz"
        write_jsonl_gz(rankings_path, ranking_rows[key])
        write_jsonl_gz(logits_path, logit_rows[key])
        manifest[f"rankings.{teacher_name}.{view}"] = {
            "path": rankings_path.name,
            "sha256": sha256_file(rankings_path),
        }
        manifest[f"logits.{teacher_name}.{view}"] = {
            "path": logits_path.name,
            "sha256": sha256_file(logits_path),
        }
    write_json(output_dir / "TEACHER_MATRIX_MANIFEST.json", manifest)
    write_json(
        output_dir / "SWAP_COLLISIONS.json",
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "generator": generator,
            "pool_kind": pool_kind,
            "swap_paths": swap_paths_total,
            "known_witness_collisions": swap_witness_collisions,
            "collision_rate": (
                swap_witness_collisions / swap_paths_total if swap_paths_total else None
            ),
            "formal_swap_mapping_changed_by_gt": False,
        },
    )
    return {teacher: dict(views) for teacher, views in result.items()}


# ----------------------------------------------------------------- Bootstrap ---


def paired_bootstrap(
    deltas_by_query: Mapping[str, float],
    query_to_group: Mapping[str, str],
    replicates: int = 10000,
    seed: int = 20260925,
) -> dict[str, Any]:
    """10,000 source-group paired bootstrap replicates (SPEC 21.3)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for qid in utf8_sorted(deltas_by_query):
        if qid not in query_to_group or not query_to_group[qid]:
            raise ValueError(f"{qid}: missing source group")
        groups[str(query_to_group[qid])].append(float(deltas_by_query[qid]))

    group_keys = utf8_sorted(groups)
    n_groups = len(group_keys)

    if n_groups == 0:
        return {"mean_delta_pp": None, "ci_95": [None, None], "wlt": [0, 0, 0], "queries": 0}

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_groups, size=(replicates, n_groups))
    boot_means = np.empty(replicates, dtype=np.float64)
    for replicate, selected in enumerate(boot_indices):
        sampled = [value for group_index in selected for value in groups[group_keys[int(group_index)]]]
        boot_means[replicate] = np.mean(sampled) * 100.0

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    mean_pp = float(np.mean(list(deltas_by_query.values())) * 100.0)

    # W/L/T counts
    w = sum(1 for d in deltas_by_query.values() if d > 1e-12)
    l = sum(1 for d in deltas_by_query.values() if d < -1e-12)
    t = sum(1 for d in deltas_by_query.values() if abs(d) <= 1e-12)

    return {
        "mean_delta_pp": mean_pp,
        "ci_95": [ci_lower, ci_upper],
        "wlt": [w, l, t],
        "queries": len(deltas_by_query),
        "groups": n_groups,
        "replicates": replicates,
        "seed": seed,
        "sampling_indices_sha256": hashlib.sha256(boot_indices.tobytes()).hexdigest(),
        "aggregation": "source_group_resample_then_query_weighted_mean",
    }

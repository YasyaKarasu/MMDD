"""Student retrieval pools, the Teacher reranking matrix with evidence swap, and the paired bootstrap."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from . import SCHEMA_VERSION
from .artifacts import pool_identity
from .data import sha256_file, utf8_sorted, write_json, write_jsonl_gz
from .features import ObjectBank, RowStore, ZStore
from .labels import Labels
from .losses import aggregate_cqet, aggregate_corrected_lse
from .models import FreshPathTeacher, NativeStudent, QTStudent, model_state_sha
from .retrieval import PoolRecord, build_pools


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
    index_dir: Optional[str | Path] = None,
    search: str = "hnsw",
) -> dict[str, PoolRecord]:
    """Student pools: ``build_pools`` in the Student's projected relation spaces."""
    return build_pools(
        z_store, row_store, query_ids, labels, split, student=student, generator_id=generator_id,
        hnsw_seed=hnsw_seed, device=device, index_dir=None if index_dir is None else Path(index_dir),
        search=search,
    )


# ------------------------------------------------------------- Teacher Eval ---


def build_evidence_swap_ring(canonical_evidence_ids: Sequence[str]) -> dict[str, str]:
    """SPEC 20.1: Deterministic canonical evidence swap ring."""
    sorted_e = utf8_sorted(set(canonical_evidence_ids))
    n = len(sorted_e)
    if n <= 1:
        return {}
    return {sorted_e[i]: sorted_e[(i + 1) % n] for i in range(n)}


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
        teacher_hash = model_state_sha(teacher)
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
                    f0, _ = teacher.score_query_lists(q, target_values, {}, [])
                    views = {"Direct": (f0, [], torch.empty(0, device=dev))}
                elif direct_only:
                    f0, _ = teacher.score_query_lists(q, target_values, {}, [])
                    views = {"f0": (f0, [], torch.empty(0, device=dev))}
                else:
                    real_evidence = {
                        evidence_id: (labels.modality[evidence_id], bank.z(evidence_id), token_map[evidence_id])
                        for evidence_id in dict.fromkeys(e for _, e in real_paths)
                    }
                    f0, real_logits = teacher.score_query_lists(q, target_values, real_evidence, real_paths)
                    if swap_paths:
                        swap_evidence = {
                            evidence_id: (labels.modality[evidence_id], bank.z(evidence_id), token_map[evidence_id])
                            for evidence_id in dict.fromkeys(e for _, e in swap_paths)
                        }
                        _swap_f0, swap_logits = teacher.score_query_lists(q, target_values, swap_evidence, swap_paths)
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

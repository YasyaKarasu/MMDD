"""Prepare and run the R17 fixed-80 Stage-2 evidence mechanism probe."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import fuse_ranked_channels
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage2.checkpoints import (
    load_candidate_scorer,
    load_candidate_scorer_metadata,
)
from mmdd_stage2.data import load_stage2_index, local_column_index
from mmdd_stage2.pipeline import CandidateResult, Stage2Verifier, joinability_sort_key
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.r12_task_f import _no_evidence_recovery, _row_equivalence
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import EvidenceBundle
from run_stage1_r11_task_e import select_evidence


SEED = 170915


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    if path.suffix == ".gz":
        handle = opener(path, "rt", encoding="utf-8")
    else:
        handle = opener(path, encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _paths(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r11 = root / "work/stage1_optimization_r11_20260908"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r13 = root / "work/stage1_optimization_r13_20260909"
    r16 = root / "work/stage1_optimization_r16_20260910"
    return {
        "fixed80": root / "mmdd_r16_review/fixed80_case_records.jsonl.gz",
        "path_pool": r13
        / "taskD_witness_supervision/p_s_target_only/evaluation_step178/path_pool.jsonl.gz",
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "content_keys": r10 / "taskB_g5/evidence_content_keys.jsonl",
        "objects": r10 / "stage1_data/stage1_objects.jsonl",
        "edge_train": r12
        / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "edge_dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "r16_plan": r16 / "PLAN_FROZEN.json",
        "r16_qt_scores": r16 / "teacher_pair_scores_qt.jsonl.gz",
        "r16_candidate_pools": r16 / "candidate_pools.jsonl.gz",
        "scorer": r12 / "taskF_end_to_end/column_scorer/seed_13/candidate.pt",
        "scorer_metrics": r12 / "taskF_end_to_end/column_scorer/cal_check_metrics.json",
        "model": root / "hf_models/Qwen3.5-9B",
        "dataset": root
        / (
            "output_mm_joinability_entitables_20000_retry100_rounds5_"
            "qwen35_final_survivor_context_gaussian_v9"
        ),
    }


def _output(root: Path) -> Path:
    return root / "work/stage1_optimization_r17_20260910"


def _content_length(record: dict[str, Any]) -> int:
    if str(record.get("object_type")) == "text":
        return max(1, len(str(record.get("text") or "")))
    path = Path(str(record.get("image") or ""))
    return max(1, path.stat().st_size if path.is_file() else 1)


def choose_wrong_evidence(
    original_ids: Sequence[str],
    *,
    target_id: str,
    query_id: str,
    object_metadata: dict[str, dict[str, Any]],
    verified_wrong_by_target: dict[str, dict[str, list[str]]],
    seed: int = SEED,
) -> tuple[list[str], str]:
    """Match verified-wrong evidence by modality, count, and input-size proxy."""

    if not original_ids:
        return [], "no_original_evidence"
    used = set(original_ids)
    selected = []
    for original_id in original_ids:
        original = object_metadata[original_id]
        modality = str(original["object_type"])
        candidates = [
            candidate
            for candidate in verified_wrong_by_target.get(target_id, {}).get(modality, [])
            if candidate not in used and candidate in object_metadata
        ]
        if not candidates:
            return [], "unavailable_verified_wrong"
        original_length = _content_length(original)
        donor = min(
            candidates,
            key=lambda candidate: (
                abs(
                    math.log(_content_length(object_metadata[candidate]))
                    - math.log(original_length)
                ),
                hashlib.sha256(
                    f"{seed}\0{query_id}\0{target_id}\0{original_id}\0{candidate}".encode()
                ).digest(),
            ),
        )
        selected.append(donor)
        used.add(donor)
    return selected, "verified_wrong"


def _selected_evidence(
    query_id: str,
    paths: Sequence[dict[str, Any]],
    *,
    store: FeatureStore,
    content_keys: dict[str, str],
    support_cache: dict[str, list[float]],
) -> tuple[list[str], float | None]:
    return select_evidence(
        "e2_row_coverage",
        paths,
        query_id=query_id,
        store=store,
        content_keys=content_keys,
        top_l=20,
        budget=4,
        support_cache=support_cache,
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root)
    output = _output(args.root)
    output.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    fixed = list(_read_jsonl(paths["fixed80"]))
    if len(fixed) != 80:
        raise ValueError("R17 P5 fixed queue must contain 80 opportunities")
    fixed_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    c50_by_query = {}
    source_by_query = {}
    for row in fixed:
        query_id = str(row["query_id"])
        fixed_by_query[query_id].append(row)
        c50 = [str(value) for value in row["E1_c50_ids"]]
        if len(c50) != 50 or len(set(c50)) != 50:
            raise ValueError(f"{query_id}: E1 C50 is not 50 unique targets")
        if query_id in c50_by_query and c50_by_query[query_id] != c50:
            raise ValueError(f"{query_id}: fixed opportunities disagree on E1 C50")
        c50_by_query[query_id] = c50
        source_by_query[query_id] = str(row["source_table_id"])
    if len(fixed_by_query) != 77 or len(set(source_by_query.values())) != 74:
        raise ValueError("R17 P5 fixed queue identity changed")
    selected_pool = {
        str(row["query_id"]): row
        for row in _read_jsonl(paths["path_pool"])
        if str(row["query_id"]) in fixed_by_query
    }
    if set(selected_pool) != set(fixed_by_query):
        raise ValueError("Frozen path pool misses a fixed-80 query")
    teacher_qt: dict[str, dict[str, float]] = defaultdict(dict)
    for row in _read_jsonl(paths["r16_qt_scores"]):
        query_id = str(row["query_id"])
        if query_id in fixed_by_query:
            teacher_qt[query_id][str(row["target_id"])] = float(
                row["teacher_qt_score"]
            )
    candidate_sources = {
        str(row["query_id"]): row
        for row in _read_jsonl(paths["r16_candidate_pools"])
        if str(row["query_id"]) in fixed_by_query
    }
    if set(candidate_sources) != set(fixed_by_query):
        raise ValueError("R16 candidate-source manifest misses a fixed-80 query")
    store = FeatureStore.from_path(paths["features"], cache_size=120_000)
    content_keys, _content_keys_sha256 = load_evidence_content_keys(
        paths["content_keys"]
    )
    assignments = []
    assignment_by_pair = {}
    evidence_ids = set()
    for query_id in sorted(fixed_by_query):
        pool = selected_pool[query_id]
        support_cache: dict[str, list[float]] = {}
        all_needed_targets = {str(value) for value in pool["paths_by_target"]}
        channel_rows = []
        for target_id in sorted(all_needed_targets):
            paths_for_target = [
                dict(path) for path in pool["paths_by_target"].get(target_id, [])
            ]
            selected, evidence_score = _selected_evidence(
                query_id,
                paths_for_target,
                store=store,
                content_keys=content_keys,
                support_cache=support_cache,
            )
            assignment_by_pair[(query_id, target_id)] = selected
            channel_rows.append(
                {
                    "target_id": target_id,
                    "direct_score": teacher_qt[query_id][target_id],
                    "evidence_score": evidence_score,
                    "selected_evidence_ids": selected,
                    "paths": paths_for_target,
                }
            )
        direct = sorted(
            channel_rows,
            key=lambda row: (-float(row["direct_score"]), str(row["target_id"])),
        )
        evidence = sorted(
            (row for row in channel_rows if row["evidence_score"] is not None),
            key=lambda row: (-float(row["evidence_score"]), str(row["target_id"])),
        )
        fused = fuse_ranked_channels(direct, evidence, rrf_k=60)
        frozen_c50 = c50_by_query[query_id]
        if [str(row["target_id"]) for row in fused[:50]] != frozen_c50:
            raise ValueError(f"{query_id}: reconstructed E1 C50 changed")
        channel_by_target = {str(row["target_id"]): row for row in channel_rows}
        fused_by_target = {str(row["target_id"]): row for row in fused[:50]}
        source = candidate_sources[query_id]
        source_sets = {
            "in_ann_direct100": set(source["ann_direct100_ids"]),
            "in_exact_direct100": set(source["exact_direct100_ids"]),
            "in_matched_direct_M": set(source["matched_direct_candidate_ids"]),
            "in_evidence_union": set(source["evidence_candidate_ids"]),
            "in_natural_union": set(source["natural_candidate_ids"]),
        }
        for target_id in frozen_c50:
            channel = channel_by_target[target_id]
            paths_for_target = channel["paths"]
            selected = channel["selected_evidence_ids"]
            evidence_score = channel["evidence_score"]
            evidence_ids.update(selected)
            selected_paths = {
                str(path["evidence_id"]): path
                for path in paths_for_target
                if path.get("kind") == "evidence"
                and str(path["evidence_id"]) in selected
            }
            assignments.append(
                {
                    "query_id": query_id,
                    "source_table_id": source_by_query[query_id],
                    "target_id": target_id,
                    "stage1_rank": c50_by_query[query_id].index(target_id) + 1,
                    "stage1_score": float(fused_by_target[target_id]["score"]),
                    "selected_evidence_ids": selected,
                    "evidence_score": evidence_score,
                    "evidence_modalities": [
                        "image" if value.startswith("asset_img_") else "text"
                        for value in selected
                    ],
                    "evidence_content_hashes": [content_keys[value] for value in selected],
                    "selected_paths": [selected_paths[value] for value in selected],
                    "has_direct_path": any(
                        path.get("kind") == "direct" for path in paths_for_target
                    ),
                    **{
                        name: target_id in values
                        for name, values in source_sets.items()
                    },
                }
            )
    if len(assignments) != 77 * 50:
        raise ValueError("R17 P5 evidence assignment must contain 3,850 units")
    for row in fixed:
        pair = (str(row["query_id"]), str(row["target_id"]))
        if assignment_by_pair[pair] != list(row["E1_selected_evidence_ids"]):
            raise ValueError(f"Frozen E1 evidence reconstruction changed for {pair}")

    object_metadata = {
        str(row["object_id"]): row
        for row in _read_jsonl(paths["objects"])
        if str(row.get("object_type")) in {"text", "image"}
    }
    verified_wrong: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label_path in (paths["edge_train"], paths["edge_dev"]):
        for row in _read_jsonl(label_path):
            if str(row["source_type"]) not in {"text", "image"} or str(
                row["destination_type"]
            ) != "table":
                continue
            evidence_id = str(row["query_id"])
            for target_id, label in zip(
                row["candidate_ids"], row.get("confirmed_labels") or [], strict=True
            ):
                if label == 0:
                    verified_wrong[str(target_id)][str(row["source_type"])].append(
                        evidence_id
                    )
    wrong_rows = []
    wrong_evidence_ids = set()
    for row in assignments:
        wrong_ids, status = choose_wrong_evidence(
            row["selected_evidence_ids"],
            target_id=str(row["target_id"]),
            query_id=str(row["query_id"]),
            object_metadata=object_metadata,
            verified_wrong_by_target=verified_wrong,
        )
        wrong_evidence_ids.update(wrong_ids)
        wrong_rows.append(
            {
                "query_id": row["query_id"],
                "target_id": row["target_id"],
                "original_evidence_ids": row["selected_evidence_ids"],
                "wrong_evidence_ids": wrong_ids,
                "wrong_evidence_modalities": [
                    str(object_metadata[value]["object_type"]) for value in wrong_ids
                ],
                "wrong_evidence_content_hashes": [
                    content_keys[value] for value in wrong_ids
                ],
                "status": status,
                "modality_matched": (
                    [
                        object_metadata[value]["object_type"]
                        for value in row["selected_evidence_ids"]
                    ]
                    == [object_metadata[value]["object_type"] for value in wrong_ids]
                    if status == "verified_wrong"
                    else None
                ),
                "count_matched": (
                    len(row["selected_evidence_ids"]) == len(wrong_ids)
                    if status == "verified_wrong"
                    else None
                ),
                "original_length_proxy": [
                    _content_length(object_metadata[value])
                    for value in row["selected_evidence_ids"]
                ],
                "wrong_length_proxy": [
                    _content_length(object_metadata[value]) for value in wrong_ids
                ],
            }
        )
    _write_jsonl_gz(output / "STAGE2_EVIDENCE_ASSIGNMENT.jsonl.gz", assignments)
    _write_jsonl_gz(output / "STAGE2_WRONG_EVIDENCE_MAP.jsonl.gz", wrong_rows)
    scorer_metadata = load_candidate_scorer_metadata(paths["scorer"])
    fixed_pairs = {
        (str(row["query_id"]), str(row["target_id"])) for row in fixed
    }
    fixed_wrong = [
        row
        for row in wrong_rows
        if (str(row["query_id"]), str(row["target_id"])) in fixed_pairs
    ]
    pool_payload = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": _now(),
        "fixed_opportunities": 80,
        "queries": 77,
        "source_groups": 74,
        "candidate_units_per_condition": 3850,
        "conditions": ["original", "no_evidence", "wrong_evidence"],
        "target_pool": "R16 E1 Top50",
        "top_k_evidence": 4,
        "recovery_budget": 10,
        "full_rank_rule": (
            "verified candidates by descending coverage, descending mean similarity, "
            "then Stage1 rank; unverified candidates append in Stage1 order"
        ),
        "unknown_wrong_policy": "not_run; never substitute unknown as verified wrong",
        "verified_wrong_label_sources": [
            "R12 corrected train_fit",
            "R12 corrected dev",
        ],
        "execution": {
            "dtype": "bfloat16",
            "focus_start_layer": 14,
            "max_text_evidence_tokens": 1024,
            "text_overlap_tokens": 128,
            "max_span_tokens": 192,
            "roi_candidates": 4,
            "embedding_batch_size": 64,
            "max_embedding_tokens": 128,
            "similarity_batch_size": 1024,
            "similarity_threshold": 0.8,
            "min_row_coverage": 0.6,
            "column_permutation_seed": int(scorer_metadata["column_permutation_seed"]),
            "column_rejection_threshold": float(scorer_metadata["rejection_threshold"]),
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(
                    path / "config.json" if name == "model" else path
                ),
            }
            for name, path in paths.items()
            if path.is_file() or name == "model"
        },
        "features": {
            "path": str(paths["features"].resolve()),
            "manifest_sha256": checkpoint_fingerprint(
                paths["features"] / "manifest.jsonl"
            ),
        },
        "artifacts": {
            "assignment_sha256": checkpoint_fingerprint(
                output / "STAGE2_EVIDENCE_ASSIGNMENT.jsonl.gz"
            ),
            "wrong_map_sha256": checkpoint_fingerprint(
                output / "STAGE2_WRONG_EVIDENCE_MAP.jsonl.gz"
            ),
        },
        "wrong_evidence_coverage": {
            status: sum(row["status"] == status for row in wrong_rows)
            for status in sorted({row["status"] for row in wrong_rows})
        },
        "fixed80_wrong_evidence_coverage": {
            status: sum(row["status"] == status for row in fixed_wrong)
            for status in sorted({row["status"] for row in fixed_wrong})
        },
        "unique_original_evidence": len(evidence_ids),
        "unique_wrong_evidence": len(wrong_evidence_ids),
        "runner_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "STAGE2_FIXED80_POOL.json", pool_payload)
    print(json.dumps(pool_payload, indent=2))
    return pool_payload


def _condition_paths(
    output: Path,
    condition: str,
    shard_index: int,
    num_shards: int,
    *,
    run_tag: str | None = None,
) -> tuple[Path, Path]:
    run_name = (
        "wrong_evidence_r12_corrected"
        if condition == "wrong_evidence"
        else condition
    )
    if run_tag:
        run_name = f"{run_name}_{run_tag}"
    base = output / "p5_runs" / f"{run_name}_{shard_index:03d}_of_{num_shards:03d}"
    return base.with_suffix(".jsonl"), base.with_suffix(".manifest.json")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _load_assignments(output: Path) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(output / "STAGE2_EVIDENCE_ASSIGNMENT.jsonl.gz"):
        by_query[str(row["query_id"])].append(row)
    for query_id, rows in by_query.items():
        rows.sort(key=lambda row: int(row["stage1_rank"]))
        if len(rows) != 50 or [int(row["stage1_rank"]) for row in rows] != list(
            range(1, 51)
        ):
            raise ValueError(f"{query_id}: Stage2 assignment is not a complete C50")
    return dict(by_query)


def _load_wrong_map(output: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["query_id"]), str(row["target_id"])): row
        for row in _read_jsonl(output / "STAGE2_WRONG_EVIDENCE_MAP.jsonl.gz")
    }


def _load_fixed_truth(
    paths: dict[str, Path], fixed_pairs: set[tuple[str, str]]
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    qrels = {
        (str(row["query_table_id"]), str(row["target_table_id"])): row
        for row in iter_dataset_artifact(paths["dataset"], "qrels")
        if (str(row["query_table_id"]), str(row["target_table_id"]))
        in fixed_pairs
    }
    recovery_values: dict[tuple[str, str, int], dict[str, Any]] = {}
    recoverable = {
        pair
        for pair, row in qrels.items()
        if row.get("reason") == "model_recoverable_join_column"
    }
    for row in iter_dataset_artifact(paths["dataset"], "evidence_recoveries"):
        pair = (str(row["query_table_id"]), str(row["target_table_id"]))
        if pair not in recoverable:
            continue
        key = (*pair, int(row["query_row_id"]))
        value = str(row["recovered_attribute"].get("value") or "").strip()
        evidence_id = str(row["evidence"]["asset_id"])
        existing = recovery_values.setdefault(
            key, {"value": value, "evidence_ids": []}
        )
        if existing["value"] != value:
            raise ValueError(f"Conflicting recovery truth for {key}")
        if evidence_id not in existing["evidence_ids"]:
            existing["evidence_ids"].append(evidence_id)
    return qrels, recovery_values


def _supports(fixed: dict[str, Any]) -> dict[str, set[int]]:
    return {
        str(evidence_id): {int(row_id) for row_id in row_ids}
        for evidence_id, row_ids in fixed.get("known_rows_by_evidence", {}).items()
    }


def _run_query(
    verifier: Stage2Verifier,
    *,
    condition: str,
    query_id: str,
    assignments: list[dict[str, Any]],
    wrong_map: dict[tuple[str, str], dict[str, Any]],
    objects: Any,
    fixed_by_pair: dict[tuple[str, str], dict[str, Any]],
    qrels: dict[tuple[str, str], dict[str, Any]],
    recovery_values: dict[tuple[str, str, int], dict[str, Any]],
    recovery_budget: int,
) -> dict[str, Any]:
    query = objects.queries[query_id]
    evidence_ids_by_target = {}
    for assignment in assignments:
        target_id = str(assignment["target_id"])
        if condition == "original":
            evidence_ids = tuple(str(value) for value in assignment["selected_evidence_ids"])
        elif condition == "no_evidence":
            evidence_ids = ()
        else:
            evidence_ids = tuple(
                str(value)
                for value in wrong_map[(query_id, target_id)]["wrong_evidence_ids"]
            )
        evidence_ids_by_target[target_id] = evidence_ids
    bundles = [
        EvidenceBundle(
            str(row["target_id"]),
            float(row["stage1_score"]),
            evidence_ids_by_target[str(row["target_id"])],
        )
        for row in assignments
    ]
    scores = verifier.score_candidates(
        query, bundles, objects.targets, objects.evidence
    )
    selected_ids = {
        score.selection.target_id
        for score in sorted(
            (score for score in scores if score.accepted),
            key=lambda score: -score.recovery_priority,
        )[:recovery_budget]
    }
    direct_ids = [
        str(row["target_id"]) for row in assignments if row["has_direct_path"]
    ]
    direct = {
        result.target_id: result
        for result in verifier.verify_direct(query, objects.targets, direct_ids)
    }
    recovered = {}
    for bundle, score in zip(bundles, scores, strict=True):
        if bundle.target_id not in selected_ids:
            continue
        target = objects.targets[bundle.target_id]
        recovered[bundle.target_id] = (
            verifier.recover_candidate(
                query, bundle, score.selection, objects.targets, objects.evidence
            )
            if bundle.evidence_ids
            else _no_evidence_recovery(verifier, query, target, score.selection)
        )
    candidates = []
    assignment_by_target = {
        str(row["target_id"]): row for row in assignments
    }
    for row, bundle, score in zip(assignments, bundles, scores, strict=True):
        target_id = str(row["target_id"])
        candidates.append(
            CandidateResult(
                target_id=target_id,
                stage1_rank=int(row["stage1_rank"]),
                stage1_score=float(row["stage1_score"]),
                table_prior=bundle.retrieval_score,
                bundle=bundle,
                scores=score,
                selected_for_recovery=target_id in selected_ids,
                direct=direct.get(target_id),
                evidence=recovered.get(target_id),
            )
        )
    verified = sorted(
        (candidate for candidate in candidates if candidate.semantic_joinability is not None),
        key=lambda candidate: joinability_sort_key(
            candidate.semantic_joinability, candidate.stage1_rank
        ),
    )
    unverified = sorted(
        (candidate for candidate in candidates if candidate.semantic_joinability is None),
        key=lambda candidate: candidate.stage1_rank,
    )
    candidate_rows = []
    for final_rank, candidate in enumerate([*verified, *unverified], 1):
        candidate = replace(candidate, rerank_rank=final_rank)
        result = candidate.to_dict()
        target_id = candidate.target_id
        assignment = assignment_by_target[target_id]
        fixed = fixed_by_pair.get((query_id, target_id))
        result.update(
            {
                "query_id": query_id,
                "source_table_id": assignment["source_table_id"],
                "condition": condition,
                "candidate_source": {
                    name: bool(assignment[name])
                    for name in (
                        "in_ann_direct100",
                        "in_exact_direct100",
                        "in_matched_direct_M",
                        "in_evidence_union",
                        "in_natural_union",
                    )
                },
                "delivered_evidence_ids": list(
                    evidence_ids_by_target[target_id]
                ),
                "delivered_evidence_modalities": [
                    "image" if value.startswith("asset_img_") else "text"
                    for value in evidence_ids_by_target[target_id]
                ],
                "evidence_content_hashes": (
                    assignment["evidence_content_hashes"]
                    if condition == "original"
                    else []
                ),
                "fixed80_opportunity": fixed is not None,
                "error": None,
                "not_run_reason": None,
            }
        )
        if fixed is not None:
            qrel = qrels.get((query_id, target_id))
            gold_column = (
                local_column_index(
                    objects.targets[target_id],
                    int(qrel["join_attribute"]["source_column_index"]),
                )
                if qrel and qrel.get("reason") == "model_recoverable_join_column"
                else None
            )
            selection = result.get("selection")
            correct_attribute = bool(
                gold_column is not None
                and result.get("column_accepted")
                and selection
                and int(selection["column_index"]) == gold_column
            )
            verification = recovered.get(target_id)
            row_metrics = (
                _row_equivalence(
                    verifier.backend,
                    verification.rows,
                    query_id=query_id,
                    target_id=target_id,
                    truth=recovery_values,
                    supports=_supports(fixed),
                    threshold=verifier.similarity_threshold,
                )
                if verification is not None
                else []
            )
            result["fixed80_evaluation"] = {
                "known_witness_ids": fixed["known_witness_ids"],
                "supported_query_rows": fixed["retained_supported_rows"],
                "gold_column_index": gold_column,
                "recovered_attribute": selection,
                "correct_attribute": correct_attribute,
                "row_metrics": row_metrics,
                "correct_value": any(
                    row["model_value_correct"] for row in row_metrics
                ),
                "evidence_supported_correct_value": any(
                    row["correct_value_recovery"] for row in row_metrics
                ),
                "correct_join": bool(
                    result.get("verification")
                    and result["verification"]["joinable"]
                    and correct_attribute
                    and any(row["model_value_correct"] for row in row_metrics)
                ),
                "final_top10": final_rank <= 10,
            }
        candidate_rows.append(result)
    return {
        "query_id": query_id,
        "source_table_id": assignments[0]["source_table_id"],
        "condition": condition,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
    }


def run_condition(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root)
    output = _output(args.root)
    frozen = json.loads(
        (output / "STAGE2_FIXED80_POOL.json").read_text(encoding="utf-8")
    )
    assignments_by_query = _load_assignments(output)
    query_ids = sorted(assignments_by_query)
    shard_queries = query_ids[args.shard_index :: args.num_shards]
    if args.reverse:
        shard_queries = list(reversed(shard_queries))
    output_path, manifest_path = _condition_paths(
        output,
        args.condition,
        args.shard_index,
        args.num_shards,
        run_tag=args.run_tag,
    )
    completed = {
        str(row["query_id"])
        for row in _read_jsonl(output_path)
    } if output_path.is_file() else set()
    wrong_map = _load_wrong_map(output)
    started = time.monotonic()
    if args.condition == "wrong_evidence":
        eligible = all(
            wrong_map[(query_id, str(row["target_id"]))]["status"]
            in {"verified_wrong", "no_original_evidence"}
            for query_id in shard_queries
            for row in assignments_by_query[query_id]
        )
        if not eligible:
            for query_id in shard_queries:
                if query_id in completed:
                    continue
                candidates = []
                for row in assignments_by_query[query_id]:
                    wrong = wrong_map[(query_id, str(row["target_id"]))]
                    candidates.append(
                        {
                            "query_id": query_id,
                            "source_table_id": row["source_table_id"],
                            "target_id": row["target_id"],
                            "condition": args.condition,
                            "stage1_rank": row["stage1_rank"],
                            "stage1_score": row["stage1_score"],
                            "delivered_evidence_ids": [],
                            "fixed80_opportunity": False,
                            "status": "not_run",
                            "error": None,
                            "not_run_reason": (
                                "condition_pool_incomplete_verified_wrong_coverage"
                            ),
                            "wrong_evidence_status": wrong["status"],
                            "rerank_rank": None,
                        }
                    )
                _append_jsonl(
                    output_path,
                    {
                        "query_id": query_id,
                        "source_table_id": assignments_by_query[query_id][0]["source_table_id"],
                        "condition": args.condition,
                        "candidate_count": 50,
                        "candidates": candidates,
                    },
                )
            payload = {
                "format_version": 1,
                "status": "not_run_insufficient_verified_wrong_coverage",
                "condition": args.condition,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "queries": len(shard_queries),
                "candidate_units": 50 * len(shard_queries),
                "result": str(output_path.resolve()),
                "result_sha256": checkpoint_fingerprint(output_path),
                "elapsed_seconds": time.monotonic() - started,
                "completed_at_utc": _now(),
            }
            write_json(manifest_path, payload)
            print(json.dumps(payload, indent=2))
            return payload

    fixed_rows = list(_read_jsonl(paths["fixed80"]))
    fixed_by_pair = {
        (str(row["query_id"]), str(row["target_id"])): row for row in fixed_rows
    }
    fixed_pairs = set(fixed_by_pair)
    qrels, recovery_values = _load_fixed_truth(paths, fixed_pairs)
    target_ids = {
        str(row["target_id"])
        for query_id in shard_queries
        for row in assignments_by_query[query_id]
    }
    evidence_ids = {
        str(value)
        for query_id in shard_queries
        for row in assignments_by_query[query_id]
        for value in (
            row["selected_evidence_ids"] if args.condition == "original" else []
        )
    }
    objects = load_stage2_index(
        paths["dataset"],
        query_ids=set(shard_queries),
        target_ids=target_ids,
        evidence_ids=evidence_ids,
    )
    execution = frozen["execution"]
    scorer = load_candidate_scorer(
        paths["scorer"], torch.device("cpu"), expected_model_dir=paths["model"]
    )
    backend = QwenStage2Backend(
        paths["model"],
        device=args.device,
        dtype="bf16",
        focus_start_layer=int(execution["focus_start_layer"]),
        max_text_evidence_tokens=int(execution["max_text_evidence_tokens"]),
        text_overlap_tokens=int(execution["text_overlap_tokens"]),
        max_span_tokens=int(execution["max_span_tokens"]),
        roi_candidates=int(execution["roi_candidates"]),
        embedding_batch_size=int(execution["embedding_batch_size"]),
        max_embedding_tokens=int(execution["max_embedding_tokens"]),
    )
    scorer.to(backend.device)
    verifier = Stage2Verifier(
        backend,
        scorer,
        evidence_router=SimilarityEvidenceRouter(
            FeatureStore.from_path(paths["features"], cache_size=120_000)
        ),
        similarity_threshold=float(execution["similarity_threshold"]),
        min_row_coverage=float(execution["min_row_coverage"]),
        similarity_batch_size=int(execution["similarity_batch_size"]),
        column_permutation_seed=int(execution["column_permutation_seed"]),
        column_rejection_threshold=float(execution["column_rejection_threshold"]),
    )
    errors = 0
    for query_id in shard_queries:
        if query_id in completed:
            continue
        query_started = time.monotonic()
        try:
            row = _run_query(
                verifier,
                condition=args.condition,
                query_id=query_id,
                assignments=assignments_by_query[query_id],
                wrong_map=wrong_map,
                objects=objects,
                fixed_by_pair=fixed_by_pair,
                qrels=qrels,
                recovery_values=recovery_values,
                recovery_budget=int(frozen["recovery_budget"]),
            )
            row["elapsed_seconds"] = time.monotonic() - query_started
        except Exception as error:
            errors += 1
            row = {
                "query_id": query_id,
                "source_table_id": assignments_by_query[query_id][0]["source_table_id"],
                "condition": args.condition,
                "candidate_count": 50,
                "candidates": [
                    {
                        "query_id": query_id,
                        "source_table_id": item["source_table_id"],
                        "target_id": item["target_id"],
                        "condition": args.condition,
                        "stage1_rank": item["stage1_rank"],
                        "stage1_score": item["stage1_score"],
                        "status": "not_run",
                        "error": f"{type(error).__name__}: {error}",
                        "not_run_reason": "query_execution_failure",
                        "rerank_rank": None,
                    }
                    for item in assignments_by_query[query_id]
                ],
                "elapsed_seconds": time.monotonic() - query_started,
            }
        _append_jsonl(output_path, row)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "query_id": query_id,
                    "completed": len(completed) + 1,
                    "shard_queries": len(shard_queries),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "error": bool(row["candidates"][0].get("error")),
                }
            ),
            flush=True,
        )
        completed.add(query_id)
    payload = {
        "format_version": 1,
        "status": "complete" if errors == 0 else "complete_with_errors",
        "condition": args.condition,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "queries": len(shard_queries),
        "candidate_units": 50 * len(shard_queries),
        "query_errors_this_invocation": errors,
        "result": str(output_path.resolve()),
        "result_sha256": checkpoint_fingerprint(output_path),
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "runner_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))
    return payload


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_source_bootstrap(
    rows: Sequence[dict[str, Any]], key: str, *, iterations: int = 10_000
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_table_id"])].append(
            float(row["original"][key]) - float(row["no_evidence"][key])
        )
    groups = sorted(grouped)
    rng = random.Random(SEED)
    draws = []
    for _ in range(iterations):
        selected = [groups[rng.randrange(len(groups))] for _group in groups]
        draws.append(
            sum(sum(grouped[group]) for group in selected)
            / sum(len(grouped[group]) for group in selected)
        )
    observed = statistics.fmean(
        float(row["original"][key]) - float(row["no_evidence"][key])
        for row in rows
    )
    return {
        "unit": "source_table_id",
        "source_groups": len(groups),
        "opportunities": len(rows),
        "iterations": iterations,
        "seed": SEED,
        "observed_delta": observed,
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def _candidate_outcomes(candidate: dict[str, Any]) -> dict[str, Any]:
    evaluation = candidate.get("fixed80_evaluation") or {}
    evidence_branch = candidate.get("branches", {}).get("evidence", {})
    row_metrics = evaluation.get("row_metrics") or []
    correct_attribute = bool(evaluation.get("correct_attribute"))
    correct_value = any(bool(row.get("model_value_correct")) for row in row_metrics)
    correct_join = bool(
        correct_attribute
        and correct_value
        and evidence_branch.get("verification")
        and evidence_branch["verification"].get("joinable")
    )
    return {
        "stage2_called": candidate.get("status") != "not_run",
        "selected_for_recovery": bool(candidate.get("selected_for_recovery")),
        "correct_attribute": correct_attribute,
        "correct_value": correct_value,
        "evidence_supported_correct_value": any(
            bool(row.get("correct_value_recovery")) for row in row_metrics
        ),
        "correct_value_rows": sum(
            bool(row.get("model_value_correct")) for row in row_metrics
        ),
        "recoverable_rows": sum(bool(row.get("truth_available")) for row in row_metrics),
        "generated_rows": len(row_metrics),
        "nonempty_generated_rows": sum(
            bool(row.get("generated_value")) for row in row_metrics
        ),
        "correct_join": correct_join,
        "final_top10": bool(candidate.get("rerank_rank") and candidate["rerank_rank"] <= 10),
        "final_rank": candidate.get("rerank_rank"),
        "stage1_rank": candidate.get("stage1_rank"),
        "rank_improved": bool(
            candidate.get("rerank_rank")
            and int(candidate["rerank_rank"]) < int(candidate["stage1_rank"])
        ),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args.root)
    output = _output(args.root)
    frozen = json.loads(
        (output / "STAGE2_FIXED80_POOL.json").read_text(encoding="utf-8")
    )
    verified_wrong_units = int(
        frozen["wrong_evidence_coverage"].get("verified_wrong", 0)
    )
    fixed_verified_wrong = int(
        frozen["fixed80_wrong_evidence_coverage"].get("verified_wrong", 0)
    )
    assignments_by_query = _load_assignments(output)
    assignment_by_pair = {
        (query_id, str(row["target_id"])): row
        for query_id, rows in assignments_by_query.items()
        for row in rows
    }
    fixed = list(_read_jsonl(paths["fixed80"]))
    fixed_by_pair = {
        (str(row["query_id"]), str(row["target_id"])): row for row in fixed
    }
    condition_candidates: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    artifact_names = {
        "original": "STAGE2_ORIGINAL.jsonl.gz",
        "no_evidence": "STAGE2_NO_EVIDENCE.jsonl.gz",
        "wrong_evidence": "STAGE2_WRONG_EVIDENCE.jsonl.gz",
    }
    for condition, artifact_name in artifact_names.items():
        input_path, manifest_path = _condition_paths(output, condition, 0, 1)
        if not input_path.is_file():
            raise FileNotFoundError(f"Incomplete P5 condition: {condition}")
        input_paths = [input_path]
        input_paths.extend(
            sorted(
                path
                for path in (output / "p5_runs").glob(
                    f"{condition}_tail_000_of_001.jsonl"
                )
                if path != input_path
            )
        )
        query_by_id = {}
        for path in input_paths:
            for row in _read_jsonl(path):
                query_by_id.setdefault(str(row["query_id"]), row)
        query_rows = list(query_by_id.values())
        if len(query_rows) != 77:
            raise ValueError(f"{condition}: expected 77 query records")
        flattened = []
        by_pair = {}
        for query_row in query_rows:
            query_id = str(query_row["query_id"])
            if len(query_row["candidates"]) != 50:
                raise ValueError(f"{condition}/{query_id}: expected 50 candidates")
            expected = {
                str(row["target_id"]) for row in assignments_by_query[query_id]
            }
            observed = {str(row["target_id"]) for row in query_row["candidates"]}
            if expected != observed:
                raise ValueError(f"{condition}/{query_id}: C50 identity changed")
            for candidate in query_row["candidates"]:
                target_id = str(candidate["target_id"])
                pair = (query_id, target_id)
                assignment = assignment_by_pair[pair]
                candidate["fixed80_opportunity"] = pair in fixed_by_pair
                candidate["candidate_source"] = {
                    name: bool(assignment[name])
                    for name in (
                        "in_ann_direct100",
                        "in_exact_direct100",
                        "in_matched_direct_M",
                        "in_evidence_union",
                        "in_natural_union",
                    )
                }
                source = candidate["candidate_source"]
                source.update(
                    {
                        "union_only": bool(
                            source["in_natural_union"]
                            and not source["in_matched_direct_M"]
                        ),
                        "matched_only": bool(
                            source["in_matched_direct_M"]
                            and not source["in_natural_union"]
                        ),
                        "both": bool(
                            source["in_natural_union"]
                            and source["in_matched_direct_M"]
                        ),
                        "neither": bool(
                            not source["in_natural_union"]
                            and not source["in_matched_direct_M"]
                        ),
                    }
                )
                candidate.setdefault("delivered_evidence_modalities", [])
                candidate.setdefault("evidence_content_hashes", [])
                verification = candidate.get("verification")
                candidate["final_score"] = (
                    {
                        "coverage": verification["coverage"],
                        "mean_similarity": verification["mean_similarity"],
                    }
                    if verification is not None
                    else None
                )
                candidate["supported_query_rows"] = (
                    fixed_by_pair[pair]["retained_supported_rows"]
                    if pair in fixed_by_pair and condition == "original"
                    else []
                    if pair in fixed_by_pair
                    else None
                )
                by_pair[pair] = candidate
                flattened.append(candidate)
        if len(flattened) != 3850 or len(by_pair) != 3850:
            raise ValueError(f"{condition}: candidate unit count changed")
        _write_jsonl_gz(output / artifact_name, flattened)
        condition_candidates[condition] = by_pair

    execution_errors = {
        condition: sum(
            bool(candidate.get("error")) for candidate in candidates.values()
        )
        for condition, candidates in condition_candidates.items()
    }

    funnel_rows = []
    paired_rows = []
    for record in fixed:
        query_id = str(record["query_id"])
        target_id = str(record["target_id"])
        pair = (query_id, target_id)
        in_c50 = pair in assignment_by_pair
        selected = set(record["E1_selected_evidence_ids"])
        known = set(record["known_witness_ids"])
        modalities = {
            "image" if value.startswith("asset_img_") else "text"
            for value in selected
        }
        modality_group = (
            "none"
            if not modalities
            else "mixed"
            if len(modalities) > 1
            else f"{next(iter(modalities))}_only"
        )
        condition_results = {}
        for condition in artifact_names:
            candidate = condition_candidates[condition].get(pair)
            condition_results[condition] = (
                _candidate_outcomes(candidate)
                if candidate is not None
                else {
                    "stage2_called": False,
                    "selected_for_recovery": False,
                    "correct_attribute": False,
                    "correct_value": False,
                    "evidence_supported_correct_value": False,
                    "correct_value_rows": 0,
                    "recoverable_rows": 0,
                    "generated_rows": 0,
                    "nonempty_generated_rows": 0,
                    "correct_join": False,
                    "final_top10": False,
                    "final_rank": None,
                    "stage1_rank": None,
                    "rank_improved": False,
                    "not_run_reason": "upstream_omission_not_in_E1_C50",
                }
            )
        funnel_rows.append(
            {
                "query_id": query_id,
                "source_table_id": record["source_table_id"],
                "target_id": target_id,
                "evidence_introduced_target": bool(
                    record["introduced_by_natural_evidence"]
                ),
                "target_in_E1_C50": in_c50,
                "selected_evidence_contains_known_witness": bool(selected & known),
                "evidence_modality_group": modality_group,
                "outside_exact_direct100": not bool(record["in_exact_direct100"]),
                "outside_matched_direct_M": not bool(record["in_matched_direct_m"]),
                "conditions": condition_results,
            }
        )
        if in_c50:
            paired_rows.append(
                {
                    "query_id": query_id,
                    "source_table_id": record["source_table_id"],
                    "target_id": target_id,
                    "original": condition_results["original"],
                    "no_evidence": condition_results["no_evidence"],
                }
            )
    funnel = {
        "format_version": 1,
        "fixed_opportunities": 80,
        "source_groups": len({str(row["source_table_id"]) for row in fixed}),
        "stages": {
            "evidence_introduced_target": sum(
                row["evidence_introduced_target"] for row in funnel_rows
            ),
            "target_in_E1_C50": sum(row["target_in_E1_C50"] for row in funnel_rows),
            "selected_evidence_contains_known_witness": sum(
                row["target_in_E1_C50"]
                and row["selected_evidence_contains_known_witness"]
                for row in funnel_rows
            ),
        },
        "conditions": {
            condition: {
                key: sum(row["conditions"][condition][key] for row in funnel_rows)
                for key in (
                    "stage2_called",
                    "selected_for_recovery",
                    "correct_attribute",
                    "correct_value",
                    "evidence_supported_correct_value",
                    "correct_join",
                    "final_top10",
                    "rank_improved",
                    "generated_rows",
                    "nonempty_generated_rows",
                )
            }
            for condition in artifact_names
        },
        "subsets": {
            subset: {
                "opportunities": len(rows),
                "in_E1_C50": sum(row["target_in_E1_C50"] for row in rows),
                "original_correct_join": sum(
                    row["conditions"]["original"]["correct_join"] for row in rows
                ),
                "no_evidence_correct_join": sum(
                    row["conditions"]["no_evidence"]["correct_join"] for row in rows
                ),
            }
            for subset, rows in {
                "outside_matched_direct_M": [
                    row for row in funnel_rows if row["outside_matched_direct_M"]
                ],
                "text_only": [
                    row
                    for row in funnel_rows
                    if row["evidence_modality_group"] == "text_only"
                ],
                "image_only": [
                    row
                    for row in funnel_rows
                    if row["evidence_modality_group"] == "image_only"
                ],
                "mixed": [
                    row
                    for row in funnel_rows
                    if row["evidence_modality_group"] == "mixed"
                ],
            }.items()
        },
        "per_opportunity": funnel_rows,
    }
    write_json(output / "STAGE2_FUNNEL.json", funnel)
    bootstrap = {
        key: _paired_source_bootstrap(paired_rows, key)
        for key in (
            "correct_attribute",
            "correct_value",
            "correct_join",
            "final_top10",
        )
    }
    paired_wlt = {
        key: {
            "original_wins": sum(
                float(row["original"][key]) > float(row["no_evidence"][key])
                for row in paired_rows
            ),
            "no_evidence_wins": sum(
                float(row["original"][key]) < float(row["no_evidence"][key])
                for row in paired_rows
            ),
            "ties": sum(
                float(row["original"][key]) == float(row["no_evidence"][key])
                for row in paired_rows
            ),
        }
        for key in (
            "correct_attribute",
            "correct_value",
            "correct_join",
            "final_top10",
        )
    }
    strict_cases = [
        row
        for row in paired_rows
        if row["original"]["correct_join"]
        and not row["no_evidence"]["correct_join"]
    ]
    counterfactual = {
        "format_version": 1,
        "status": "partial_wrong_evidence_not_identifiable",
        "paired_original_no_evidence_opportunities": len(paired_rows),
        "wrong_evidence_full_pool_executed": False,
        "execution_errors": execution_errors,
        "wrong_evidence_reason": (
            f"Only {verified_wrong_units}/3850 candidate units and "
            f"{fixed_verified_wrong}/59 in-C50 fixed opportunities have count- and "
            "modality-matched verified-wrong donors under R12 corrected labels; "
            "unknowns were not relabeled."
        ),
        "bootstrap_original_minus_no_evidence": bootstrap,
        "paired_win_loss_tie": paired_wlt,
        "original_correct_join_no_evidence_failure": len(strict_cases),
        "candidate_mechanism_cases": [
            {
                "query_id": row["query_id"],
                "source_table_id": row["source_table_id"],
                "target_id": row["target_id"],
                "original": row["original"],
                "no_evidence": row["no_evidence"],
            }
            for row in strict_cases
        ],
    }
    write_json(output / "STAGE2_COUNTERFACTUAL_ANALYSIS.json", counterfactual)
    smoke_manifest = json.loads(
        (output / "p5_runs/original_000_of_077.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    code_manifest_path = output / "CODE_HASH_MANIFEST.json"
    code_manifest = json.loads(code_manifest_path.read_text(encoding="utf-8"))
    code_manifest["r17_code"] = {
        "path": str((args.root / "src/run_stage1_r17.py").resolve()),
        "sha256": checkpoint_fingerprint(args.root / "src/run_stage1_r17.py"),
    }
    code_manifest["r17_p5"] = {
        "path": str(Path(__file__).resolve()),
        "execution_runner_sha256": smoke_manifest["runner_sha256"],
        "finalizer_sha256": checkpoint_fingerprint(Path(__file__)),
        "note": (
            "The dual-GPU processes used the same runner verified by the pre-run "
            "smoke test; finalize/report helpers were appended while they ran."
        ),
    }
    write_json(code_manifest_path, code_manifest)
    input_manifest_path = output / "INPUT_MANIFEST.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    for name in ("edge_train", "edge_dev"):
        path = paths[name]
        input_manifest["inputs"][name] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_fingerprint(path),
            "bytes": path.stat().st_size,
        }
    input_manifest["label_semantics"] = (
        "R12 corrected supervision; R11 withdrawn zeros are not verified negatives"
    )
    write_json(input_manifest_path, input_manifest)
    p0 = json.loads((output / "P0_GATE.json").read_text(encoding="utf-8"))
    original_counts = funnel["conditions"]["original"]
    no_evidence_counts = funnel["conditions"]["no_evidence"]
    report = f"""# Stage1 Optimization R17 Results

Completed at `{_now()}` against the frozen R16 inputs.

## Decision summary

- P0-A/B/C passed. Current and historical five-relation Teacher logits replayed with maximum absolute errors `{p0['teacher_function_replay_max_error']:.3g}` and `{p0['historical_function_replay_max_error']:.3g}`.
- P0-D passed lineage/cache checks with an image-data sensitivity: `{p0['corrupt_or_unsafe_image_count']}` raw image inputs are corrupt or unsafe to decode.
- P0-E is blocked because all five relations have no verified negative labels under the R12-corrected supervision. Unknown candidates were not relabeled as negatives.
- Therefore P1–P4 were not triggered. This follows the R17 correctness gate; it is not a negative structural result for those unrun arms.
- P5 ran the frozen E1 Top-50 pool for Original and NoEvidence. WrongEvidence was not run as a full condition because R12-corrected verified-wrong coverage is only {verified_wrong_units}/3850 candidate units and {fixed_verified_wrong}/59 in-C50 fixed opportunities.

## Why some experiments were not run

### P0-E tiny verified-list overfit: blocked by unavailable labels

R17 requires a small memorization test before any Teacher-method comparison: 16 train lists for each of table→table, table→text, table→image, text→table, and image→table, with both known positives and explicitly verified negatives. The plan forbids treating an unlabelled candidate as a negative.

The first inventory exposed `0` values in the older R11 reverse-edge files. Those values could not be used: R12's correctness pass explicitly withdrew them, and the R12-corrected train supervision contains zero verified-negative lists and zero verified-negative candidates for all five relations. The authoritative audit is `TINY_OVERFIT_RESULTS.json`; it records the corrected source path/hash and the per-relation counts.

P0-A through P0-D establish that saved logits are reproducible, relation-direction interfaces are stable, and current feature/cache lineage is coherent. They do not answer whether the current loss, labels, gradients, and optimizer can memorize a tiny deterministic five-relation task. Skipping P0-E would therefore bypass the exact correctness gate that R17 introduced. No synthetic negatives, cross-query assumptions, or withdrawn R11 labels were substituted.

### P1–P4: paused by the P0 gate, not evaluated and not failed

- **P1 T-edge versus T-path frozen checkpoints** was not run because the R17 plan permits Teacher-method experiments only after all P0 conditions are satisfied. Consequently there are no R17 P1 scores, ranks, or checkpoint winner.
- **P2 global representation residual** depends on P1 showing that both frozen checkpoints remain weak. Because P1 was not authorized, P2 was not triggered; no residual model was trained.
- **P3 relation-specific heads** is conditional on P2 being partial or negative. Neither P2 nor P3 was run, so no claim about relation-head usefulness is supported.
- **P4 natural-candidate training** is conditional on the preceding correctness and minimal-structure tests. It was not run, and there is no R17 evidence for or against the candidate-distribution hypothesis.

These statuses must not be summarized as “P1–P4 failed” or “the proposed structures did not improve recall.” They are unobserved interventions stopped by protocol.

### P5 WrongEvidence: full counterfactual not identifiable

P5 is orthogonal to P1–P4, so its Original and NoEvidence arms were executed on the frozen R16 E1 C50. The WrongEvidence arm has an additional constraint: every condition must keep the same 50 targets, and each replacement must be known not to support that query-target pair while matching evidence modality and count as closely as possible.

Under R12-corrected labels, verified-wrong donor coverage is {verified_wrong_units}/3850 candidate units and {fixed_verified_wrong}/59 in-C50 fixed positive opportunities. Running only an easier subset would change the target pool and normalization; filling the rest with unknown evidence would violate the plan. Therefore all 3,850 WrongEvidence records are preserved with `status=not_run` and `not_run_reason=condition_pool_incomplete_verified_wrong_coverage`. P5 is correctly marked partial, and no three-way evidence-necessity claim is made.

### P6 latency: not triggered without a viable Teacher

P6 measures the deployable Student-only versus Student+Teacher system selected by P1–P4. Since the gate produced no authorized viable Teacher, running a new latency benchmark would not correspond to a selected R17 system. P6 is recorded as `not_triggered_no_viable_teacher` rather than zero cost or failure.

## What is needed to resume the stopped stages

1. Produce authoritative train-only verified negatives for at least 16 lists in each of the five relations, without reusing dev/test groups or relabelling unknowns.
2. Freeze and run P0-E, saving loss, positive margins, relation-wise R@1, gradient norms, and parameter-update norms.
3. If P0 then passes, execute P1 T-edge versus T-path. Trigger P2, P3, and P4 only through their documented sequential stop rules.
4. To complete P5 WrongEvidence, construct a pre-frozen donor map with verified non-support coverage for the complete fixed C50 condition; otherwise retain the current two-arm result as partial.
5. Run P6 only after one Teacher is selected as viable by the preceding gates.

## Fixed-80 Stage2 funnel

- Frozen opportunities: 80 across 77 queries and 74 source groups.
- Target retained in E1 C50: {funnel['stages']['target_in_E1_C50']}/80.
- Retained target with a selected known witness: {funnel['stages']['selected_evidence_contains_known_witness']}/80.
- Original: selected for recovery {original_counts['selected_for_recovery']}, correct attribute {original_counts['correct_attribute']}, correct value {original_counts['correct_value']}, evidence-supported correct value {original_counts['evidence_supported_correct_value']}, non-empty generated rows {original_counts['nonempty_generated_rows']}/{original_counts['generated_rows']}, correct materialized join {original_counts['correct_join']}, final Top10 {original_counts['final_top10']}.
- NoEvidence: selected for recovery {no_evidence_counts['selected_for_recovery']}, correct attribute {no_evidence_counts['correct_attribute']}, correct value {no_evidence_counts['correct_value']}, non-empty generated rows {no_evidence_counts['nonempty_generated_rows']}/{no_evidence_counts['generated_rows']}, correct materialized join {no_evidence_counts['correct_join']}, final Top10 {no_evidence_counts['final_top10']}.

`correct_join` requires the recovered evidence branch itself to be joinable, the selected attribute to match the qrel, and at least one recovered row value to match evaluator-side truth. A joinable direct branch alone is not counted as evidence materialization.

## Interpretation boundary

P0 supports score-function reproducibility and interface consistency, but does not authorize a Teacher architecture claim because the verified-list overfit test could not be constructed. P5 is a partial two-condition mechanism probe; no three-way Original/No/Wrong necessity claim is allowed without a complete verified-wrong pool.
"""
    report_path = output / "README.md"
    temporary_report = report_path.with_suffix(".md.tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    completion = {
        "format_version": 1,
        "completed_at_utc": _now(),
        "P0": p0["status"],
        "P1": "not_triggered_by_P0_gate",
        "P2": "not_triggered_by_P0_gate",
        "P3": "not_triggered_by_P0_gate",
        "P4": "not_triggered_by_P0_gate",
        "P5": (
            "failed_execution"
            if execution_errors["original"] or execution_errors["no_evidence"]
            else "partial_wrong_evidence_coverage"
        ),
        "P6": "not_triggered_no_viable_teacher",
        "stage_rationales": {
            "P0": (
                "A-D passed, but P0-E could not be constructed because R12 "
                "corrected supervision has zero verified negatives in all five relations."
            ),
            "P1": "Not authorized because P0-E remains unresolved.",
            "P2": "Not triggered because P1 was not run and produced no weak-checkpoint result.",
            "P3": "Not triggered because its prerequisite P2 result does not exist.",
            "P4": "Not triggered because the sequential correctness/structure gates were not cleared.",
            "P5": (
                "Original and NoEvidence completed; WrongEvidence was not runnable "
                "because R12-corrected verified-wrong donor coverage is zero."
            ),
            "P6": "Not triggered because no viable R17 Teacher was selected.",
        },
        "artifacts": {
            name: checkpoint_fingerprint(output / name)
            for name in [
                *artifact_names.values(),
                "STAGE2_FUNNEL.json",
                "STAGE2_COUNTERFACTUAL_ANALYSIS.json",
                "README.md",
            ]
        },
        "runner_sha256": checkpoint_fingerprint(Path(__file__)),
        "condition_execution_runner_sha256": smoke_manifest["runner_sha256"],
    }
    write_json(output / "COMPLETION_AUDIT.json", completion)
    print(json.dumps(completion, indent=2))
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("finalize")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--condition",
        choices=("original", "no_evidence", "wrong_evidence"),
        required=True,
    )
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--reverse", action="store_true")
    run_parser.add_argument("--run-tag")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == "run" and not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "prepare":
        prepare(arguments)
    elif arguments.command == "run":
        run_condition(arguments)
    elif arguments.command == "finalize":
        finalize(arguments)

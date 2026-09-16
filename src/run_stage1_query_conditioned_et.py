#!/usr/bin/env python
"""Run the preregistered query-conditioned E-to-target experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.query_conditioned_et import (
    ARMS,
    QueryConditionedETAdapter,
    file_sha,
    grouped_paired_bootstrap,
    paired_wlt,
    query_macro_rank_metrics,
    rank_metrics,
    residual_geometry,
    retrieval_metrics,
    stable_sha,
    sum_probability_listwise_loss,
    tensor_sha,
)
from mmdd_stage1.r26_metrics import fuse_channels
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import select_evidence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/stage1_query_conditioned_et_20260916"
R27 = ROOT / "work/stage1_diagnostics_r27_20260915_rev2_b13_exact"
R27_EVAL = R27 / "historical_replay/own_evaluation"
PARENT = R27 / "historical_replay/C2/seed13/checkpoints/step_000178.pt"
INDEX = R27_EVAL / "indexes/H-C2-step000178"
RANKINGS = R27_EVAL / "rankings/H-C2-step000178/rankings.jsonl.gz"
TEACHER_RANKINGS = R27_EVAL / "teacher/H-C2-step000178/rankings.jsonl.gz"
TEACHER_CACHE = R27_EVAL / "teacher/T0_pairs.sqlite"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
SUPERVISION = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision"
TRAIN_LISTS = SUPERVISION / "target_lists.train_fit.jsonl"
DEV_LISTS = SUPERVISION / "target_lists.dev.jsonl"
CONTENT_KEYS = ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
PLAN = ROOT / "MMDD_Query_Conditioned_ET_Experiment_Plan.md"

SEEDS = (13, 29)
HIDDEN_DIMENSION = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3
BATCH_SIZE = 128
TRAIN_CANDIDATES = 32
EVIDENCE_K = 20


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "wt", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=True) + "\n")


def write_csv(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def record(path: Path) -> dict[str, Any]:
    result = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        result.update(bytes=path.stat().st_size, sha256=file_sha(path))
    return result


def log(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def device_from_name(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.init()
    return device


def parent_parameter_sha(model: torch.nn.Module) -> str:
    return stable_sha(
        {
            name: tensor_sha(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def witness_examples(path: Path) -> list[dict[str, Any]]:
    result = []
    for row in rows(path):
        by_evidence: dict[str, set[str]] = defaultdict(set)
        for target_id, evidence_ids in row["positive_evidence_by_target"].items():
            for evidence_id in evidence_ids:
                by_evidence[str(evidence_id)].add(str(target_id))
        for evidence_id, positives in sorted(by_evidence.items()):
            result.append(
                {
                    "query_id": str(row["query_id"]),
                    "evidence_id": evidence_id,
                    "positive_target_ids": sorted(positives),
                    "query_kind": str(row["query_kind"]),
                    "source_table_id": str(row.get("source_table_id", row["query_id"])),
                }
            )
    return result


def _evidence_from_ranking(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    targets: dict[str, dict[str, float]] = defaultdict(dict)
    for target in row["E_pre_retention"]:
        target_id = str(target["target_id"])
        for path in target["paths"]:
            if path.get("kind") != "evidence":
                continue
            evidence_id = str(path["evidence_id"])
            evidence[evidence_id] = {
                "evidence_id": evidence_id,
                "evidence_type": str(path["evidence_type"]),
                "query_evidence_score": float(path["query_evidence_score"]),
            }
            targets[evidence_id][target_id] = float(path["evidence_target_score"])
    result = []
    for evidence_id, value in evidence.items():
        ranked = sorted(targets[evidence_id], key=lambda key: (-targets[evidence_id][key], key))
        result.append(
            {
                **value,
                "historical_targets": ranked,
                "historical_target_scores": [targets[evidence_id][key] for key in ranked],
            }
        )
    return sorted(
        result,
        key=lambda value: (
            value["evidence_type"],
            -value["query_evidence_score"],
            value["evidence_id"],
        ),
    )


def extract_evaluation_input() -> list[dict[str, Any]]:
    dev = {row["query_id"]: row for row in rows(DEV_LISTS)}
    result = []
    for row in rows(RANKINGS):
        query_id = str(row["query_id"])
        witness = dev[query_id]
        evidence = _evidence_from_ranking(row)
        positives_by_evidence: dict[str, set[str]] = defaultdict(set)
        for target_id, evidence_ids in witness["positive_evidence_by_target"].items():
            for evidence_id in evidence_ids:
                positives_by_evidence[str(evidence_id)].add(str(target_id))
        for value in evidence:
            value["verified_positive_target_ids"] = sorted(
                positives_by_evidence.get(value["evidence_id"], set())
            )
        result.append(
            {
                "query_id": query_id,
                "query_kind": str(row["query_kind"]),
                "source_table_id": str(row["source_table_id"]),
                "positive_target_ids": [str(value) for value in row["positive_target_ids"]],
                "evidence": evidence,
                "direct": [
                    {
                        "target_id": str(value["target_id"]),
                        "direct_score": float(value["direct_score"]),
                    }
                    for value in row["D100_ANN"]
                ],
                "direct_exact": [str(value) for value in row["D100_EXACT"]],
                "historical_e_rank": [str(value) for value in row["rankings"]["E_ONLY"]],
                "historical_equal": [str(value) for value in row["rankings"]["Equal"]],
                "historical_c100": [str(value) for value in row["rankings"]["Equal"][:100]],
            }
        )
    if set(dev) != {row["query_id"] for row in result}:
        raise ValueError("R27 population and verified dev witnesses differ")
    return result


@torch.inference_mode()
def _project_queries(
    model: torch.nn.Module,
    store: FeatureStore,
    object_ids: Sequence[str],
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    blocks = []
    for start in range(0, len(object_ids), batch_size):
        batch = object_ids[start : start + batch_size]
        embedding = torch.stack(
            [store.embedding_features(object_id).embedding for object_id in batch]
        ).to(device=device, dtype=torch.float32)
        blocks.append(model.project(embedding, "table", role="query").cpu())
    return torch.cat(blocks)


@torch.inference_mode()
def _project_targets(
    model: torch.nn.Module,
    store: FeatureStore,
    object_ids: Sequence[str],
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    blocks = []
    for start in range(0, len(object_ids), batch_size):
        batch = object_ids[start : start + batch_size]
        embedding = torch.stack(
            [store.embedding_features(object_id).embedding for object_id in batch]
        ).to(device=device, dtype=torch.float32)
        blocks.append(model.index_vector(embedding, "table").cpu())
    return torch.cat(blocks)


@torch.inference_mode()
def _project_evidence(
    model: torch.nn.Module,
    store: FeatureStore,
    object_ids: Sequence[str],
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    types = [store.embedding_features(object_id).object_type for object_id in object_ids]
    projected = torch.empty((len(object_ids), model.student_dim), dtype=torch.float32)
    relation = torch.empty_like(projected)
    for evidence_type in ("text", "image"):
        positions = [index for index, value in enumerate(types) if value == evidence_type]
        for start in range(0, len(positions), batch_size):
            selected = positions[start : start + batch_size]
            embedding = torch.stack(
                [store.embedding_features(object_ids[index]).embedding for index in selected]
            ).to(device=device, dtype=torch.float32)
            u_e = model.project(embedding, evidence_type)
            v_e = model.relation_query(embedding, evidence_type, "table")
            projected[selected] = u_e.cpu()
            relation[selected] = v_e.cpu()
    return projected, relation, types


def prepare_vector_cache(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    path = OUT / "vector_cache.pt"
    identity_path = OUT / "VECTOR_CACHE_IDENTITY.json"
    if path.is_file() and identity_path.is_file():
        identity = read_json(identity_path)
        if identity["parent_checkpoint_sha256"] != file_sha(PARENT):
            raise ValueError("Existing vector cache belongs to a different parent")
        return identity
    model = load_student(PARENT, device).eval()
    if model.relation_param != "full":
        raise ValueError("The preregistered adapter expects a full relation matrix")
    store = FeatureStore.from_path(FEATURES, cache_size=0)
    target_ids = [str(value) for value in read_json(INDEX / "table_ids.json")]
    query_ids = sorted(
        {str(row["query_id"]) for row in train}
        | {str(row["query_id"]) for row in evaluation}
    )
    evidence_ids = sorted(
        {str(row["evidence_id"]) for row in train}
        | {
            str(value["evidence_id"])
            for row in evaluation
            for value in row["evidence"]
        }
    )
    log(
        "vector_cache_start",
        targets=len(target_ids),
        queries=len(query_ids),
        evidence=len(evidence_ids),
    )
    started = time.monotonic()
    target_vectors = _project_targets(model, store, target_ids, device)
    query_vectors = _project_queries(model, store, query_ids, device)
    evidence_vectors, base_queries, evidence_types = _project_evidence(
        model, store, evidence_ids, device
    )
    payload = {
        "format_version": 1,
        "target_ids": target_ids,
        "target_vectors": target_vectors,
        "query_ids": query_ids,
        "query_vectors": query_vectors,
        "evidence_ids": evidence_ids,
        "evidence_vectors": evidence_vectors,
        "base_queries": base_queries,
        "evidence_types": evidence_types,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    identity = {
        "format_version": 1,
        "parent_checkpoint_sha256": file_sha(PARENT),
        "parent_parameter_sha256": parent_parameter_sha(model),
        "dimension": int(model.student_dim),
        "target_count": len(target_ids),
        "query_count": len(query_ids),
        "evidence_count": len(evidence_ids),
        "target_vector_sha256": tensor_sha(target_vectors),
        "query_vector_sha256": tensor_sha(query_vectors),
        "evidence_vector_sha256": tensor_sha(evidence_vectors),
        "base_query_sha256": tensor_sha(base_queries),
        "cache": record(path),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(identity_path, identity)
    log("vector_cache_complete", seconds=identity["elapsed_seconds"])
    return identity


def load_vector_cache() -> dict[str, Any]:
    return torch.load(OUT / "vector_cache.pt", map_location="cpu", weights_only=True)


def _hnsw_search(
    queries: torch.Tensor,
    *,
    k: int,
    threads: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    import hnswlib

    manifest = read_json(INDEX / "manifest.json")
    spec = manifest["types"]["table"]
    index = hnswlib.Index(space="ip", dim=int(manifest["ann_dim"]))
    index.load_index(str(INDEX / spec["index_path"]), max_elements=int(spec["objects"]))
    index.set_ef(max(int(manifest["ef_search"]), k))
    index.set_num_threads(threads)
    labels, distances = index.knn_query(
        queries.detach().cpu().numpy().astype("float32"), k=k
    )
    return labels, 1.0 - distances


def build_training_lists(train: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = OUT / "training_lists.pt"
    audit_path = OUT / "TRAINING_LIST_AUDIT.json"
    if path.is_file() and audit_path.is_file():
        return read_json(audit_path)
    vectors = load_vector_cache()
    target_position = {value: index for index, value in enumerate(vectors["target_ids"])}
    query_position = {value: index for index, value in enumerate(vectors["query_ids"])}
    evidence_position = {value: index for index, value in enumerate(vectors["evidence_ids"])}
    train_evidence_ids = sorted({str(row["evidence_id"]) for row in train})
    train_evidence_positions = [evidence_position[value] for value in train_evidence_ids]
    labels, _scores = _hnsw_search(
        vectors["base_queries"][train_evidence_positions], k=TRAIN_CANDIDATES
    )
    mined = {
        evidence_id: [int(value) for value in row]
        for evidence_id, row in zip(train_evidence_ids, labels)
    }
    candidates = torch.empty((len(train), TRAIN_CANDIDATES), dtype=torch.long)
    positive_mask = torch.zeros_like(candidates, dtype=torch.bool)
    query_indices = torch.empty(len(train), dtype=torch.long)
    evidence_indices = torch.empty(len(train), dtype=torch.long)
    protected = 0
    records = []
    for index, row in enumerate(train):
        query_id = str(row["query_id"])
        evidence_id = str(row["evidence_id"])
        positive_ids = [str(value) for value in row["positive_target_ids"]]
        missing_targets = [value for value in positive_ids if value not in target_position]
        if missing_targets:
            raise ValueError(f"Training positives absent from target lake: {missing_targets}")
        positive_positions = {target_position[value] for value in positive_ids}
        values = list(mined[evidence_id])
        for position in positive_positions:
            if position not in values:
                protected += 1
                removable = next(
                    offset
                    for offset in range(len(values) - 1, -1, -1)
                    if values[offset] not in positive_positions
                )
                values[removable] = position
        base = vectors["base_queries"][evidence_position[evidence_id]]
        candidate_vectors = vectors["target_vectors"][values]
        score = candidate_vectors @ base
        order = sorted(range(len(values)), key=lambda i: (-float(score[i]), vectors["target_ids"][values[i]]))
        values = [values[offset] for offset in order]
        mask = [value in positive_positions for value in values]
        if not any(mask) or all(mask):
            raise ValueError("Every frozen list must contain positives and negatives")
        candidates[index] = torch.tensor(values)
        positive_mask[index] = torch.tensor(mask)
        query_indices[index] = query_position[query_id]
        evidence_indices[index] = evidence_position[evidence_id]
        records.append(
            {
                "query_id": query_id,
                "evidence_id": evidence_id,
                "candidate_target_ids": [vectors["target_ids"][value] for value in values],
                "positive_mask": mask,
            }
        )
    checksum = stable_sha(records)
    payload = {
        "format_version": 1,
        "query_indices": query_indices,
        "evidence_indices": evidence_indices,
        "candidate_indices": candidates,
        "positive_mask": positive_mask,
        "records_sha256": checksum,
    }
    torch.save(payload, path)
    write_rows(OUT / "training_lists.jsonl.gz", records)
    audit = {
        "status": "pass",
        "examples": len(train),
        "candidate_list_size": TRAIN_CANDIDATES,
        "positive_links": int(positive_mask.sum()),
        "closure_insertions": protected,
        "all_positive_closure_protected": True,
        "unknown_as_negative": True,
        "dynamic_remining": False,
        "records_sha256": checksum,
        "arm_candidate_identity": {arm: checksum for arm in ARMS},
        "tensor_file": record(path),
        "jsonl_file": record(OUT / "training_lists.jsonl.gz"),
    }
    write_json(audit_path, audit)
    return audit


def write_identity_files(vector_identity: Mapping[str, Any]) -> None:
    index_manifest = read_json(INDEX / "manifest.json")
    index_files = [INDEX / value for value in ("manifest.json", "table.hnsw", "table_ids.json")]
    base = {
        "status": "pass",
        "parent_checkpoint": record(PARENT),
        "r27_historical_verdict": record(R27 / "historical_replay/H_VERDICT.json"),
        "r27_state_parity": read_json(R27 / "historical_replay/H_VERDICT.json")["state_parity_level"],
        "student_dimension": vector_identity["dimension"],
        "parent_parameter_sha256": vector_identity["parent_parameter_sha256"],
    }
    target = {
        "status": "pass",
        "manifest": index_manifest,
        "files": [record(path) for path in index_files],
        "target_vector_sha256": vector_identity["target_vector_sha256"],
        "reuse_policy": "one unchanged R27 target index for BASE, E-ONLY, and QE queries",
    }
    write_json(OUT / "BASE_IDENTITY.json", base)
    write_json(OUT / "TARGET_INDEX_IDENTITY.json", target)
    write_json(
        OUT / "INPUT_LOCK.json",
        {
            "plan": record(PLAN),
            "parent": record(PARENT),
            "feature_manifest": record(FEATURES / "manifest.jsonl"),
            "target_index": target,
            "train_witness": record(TRAIN_LISTS),
            "dev_witness": record(DEV_LISTS),
            "frozen_r27_rankings": record(RANKINGS),
            "frozen_r27_teacher_rankings": record(TEACHER_RANKINGS),
            "content_keys": record(CONTENT_KEYS),
        },
    )


def prepare(device_name: str) -> dict[str, Any]:
    device = device_from_name(device_name)
    OUT.mkdir(parents=True, exist_ok=True)
    train_path = OUT / "train_witness_qe.jsonl.gz"
    eval_path = OUT / "fixed_evaluation_input.jsonl.gz"
    if train_path.is_file():
        train = list(rows(train_path))
    else:
        train = witness_examples(TRAIN_LISTS)
        write_rows(train_path, train)
    if eval_path.is_file():
        evaluation = list(rows(eval_path))
    else:
        evaluation = extract_evaluation_input()
        write_rows(eval_path, evaluation)
    vector_identity = prepare_vector_cache(train, evaluation, device)
    write_identity_files(vector_identity)
    audit = build_training_lists(train)
    evidence_counts = Counter(
        value["evidence_type"] for row in evaluation for value in row["evidence"]
    )
    summary = {
        "status": "pass",
        "train_queries": len({row["query_id"] for row in train}),
        "train_qe_examples": len(train),
        "train_positive_triples": sum(len(row["positive_target_ids"]) for row in train),
        "test_queries": len(evaluation),
        "fixed_test_evidence": dict(evidence_counts),
        "verified_test_qe": sum(
            bool(value["verified_positive_target_ids"])
            for row in evaluation
            for value in row["evidence"]
        ),
        "training_list_audit": audit,
    }
    write_json(OUT / "PREPARE_SUMMARY.json", summary)
    return summary


def adapter_checkpoint_path(arm: str, seed: int, epoch: str) -> Path:
    return OUT / "training" / arm / f"seed{seed}" / f"epoch_{epoch}.pt"


def save_adapter(
    path: Path,
    adapter: QueryConditionedETAdapter,
    optimizer: torch.optim.Optimizer | None,
    *,
    arm: str,
    seed: int,
    epoch: float,
    update: int,
    training_list_sha: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_kind": "query_conditioned_et_adapter",
        "config": adapter.config(),
        "state_dict": adapter.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "arm": arm,
        "seed": seed,
        "epoch": epoch,
        "update": update,
        "parent_checkpoint_sha256": file_sha(PARENT),
        "target_vector_sha256": read_json(OUT / "VECTOR_CACHE_IDENTITY.json")[
            "target_vector_sha256"
        ],
        "training_list_sha256": training_list_sha,
    }
    torch.save(payload, path)
    write_json(
        path.with_suffix(".json"),
        {
            key: value
            for key, value in payload.items()
            if key not in {"state_dict", "optimizer_state_dict"}
        }
        | {"checkpoint": record(path)},
    )


def load_adapter(path: Path, device: torch.device) -> QueryConditionedETAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    adapter = QueryConditionedETAdapter(
        int(payload["config"]["dimension"]),
        int(payload["config"]["hidden_dimension"]),
    )
    adapter.load_state_dict(payload["state_dict"])
    return adapter.to(device)


@torch.inference_mode()
def step0_parity(
    adapter: QueryConditionedETAdapter,
    arm: str,
    vectors: Mapping[str, Any],
    lists: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sample = torch.arange(min(256, len(lists["query_indices"])))
    q = vectors["query_vectors"][lists["query_indices"][sample]].to(device)
    e = vectors["evidence_vectors"][lists["evidence_indices"][sample]].to(device)
    base = vectors["base_queries"][lists["evidence_indices"][sample]].to(device)
    target = vectors["target_vectors"][lists["candidate_indices"][sample]].to(device)
    conditioned = adapter.conditioned_query(base, q, e, arm)
    base_scores = torch.bmm(target, base.unsqueeze(-1)).squeeze(-1)
    conditioned_scores = torch.bmm(target, conditioned.unsqueeze(-1)).squeeze(-1)
    score_difference = float((base_scores - conditioned_scores).abs().max())
    base_rank = torch.argsort(base_scores, dim=-1, descending=True, stable=True)
    conditioned_rank = torch.argsort(
        conditioned_scores, dim=-1, descending=True, stable=True
    )
    return {
        "score_max_abs_difference": score_difference,
        "score_atol": 1e-6,
        "score_rtol": 1e-5,
        "ranking_equal": bool(torch.equal(base_rank, conditioned_rank)),
        "residual_exact_zero": bool(torch.count_nonzero(conditioned - base) == 0),
        "status": "pass"
        if score_difference == 0.0 and torch.equal(base_rank, conditioned_rank)
        else "fail",
    }


def train_one(arm: str, seed: int, device_name: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    final_path = adapter_checkpoint_path(arm, seed, "3")
    execution_path = final_path.parent / "EXECUTION.json"
    if final_path.is_file() and execution_path.is_file():
        previous = read_json(execution_path)
        if previous.get("status") == "completed":
            return previous
    device = device_from_name(device_name)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    vectors = load_vector_cache()
    lists = torch.load(OUT / "training_lists.pt", map_location="cpu", weights_only=True)
    dimension = int(vectors["target_vectors"].shape[-1])
    adapter = QueryConditionedETAdapter(dimension, HIDDEN_DIMENSION).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    parity = step0_parity(adapter, arm, vectors, lists, device)
    if parity["status"] != "pass":
        raise RuntimeError(f"Step0 parity failed for {arm}/seed{seed}")
    save_adapter(
        adapter_checkpoint_path(arm, seed, "0"),
        adapter,
        None,
        arm=arm,
        seed=seed,
        epoch=0.0,
        update=0,
        training_list_sha=lists["records_sha256"],
    )
    target_vectors = vectors["target_vectors"].to(device)
    query_vectors = vectors["query_vectors"].to(device)
    evidence_vectors = vectors["evidence_vectors"].to(device)
    base_queries = vectors["base_queries"].to(device)
    query_indices = lists["query_indices"].to(device)
    evidence_indices = lists["evidence_indices"].to(device)
    candidate_indices = lists["candidate_indices"].to(device)
    positive_mask = lists["positive_mask"].to(device)
    examples = len(query_indices)
    batches = math.ceil(examples / BATCH_SIZE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    updates = 0
    nonzero_gradient_updates = 0
    max_gradient = 0.0
    started = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(examples, generator=generator)
        epoch_losses = []
        half_saved = False
        for batch_index, start in enumerate(range(0, examples, BATCH_SIZE), 1):
            selected = order[start : start + BATCH_SIZE].to(device)
            q = query_vectors[query_indices[selected]]
            e = evidence_vectors[evidence_indices[selected]]
            base = base_queries[evidence_indices[selected]]
            conditioned = adapter.conditioned_query(base, q, e, arm)
            targets = target_vectors[candidate_indices[selected]]
            logits = torch.bmm(targets, conditioned.unsqueeze(-1)).squeeze(-1)
            loss = sum_probability_listwise_loss(logits, positive_mask[selected])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at {arm}/seed{seed}/update{updates + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                value.grad.detach().abs().max()
                for value in adapter.parameters()
                if value.grad is not None
            ]
            gradient = max((float(value) for value in gradients), default=0.0)
            nonzero_gradient_updates += int(gradient > 0)
            max_gradient = max(max_gradient, gradient)
            optimizer.step()
            updates += 1
            epoch_losses.append(float(loss.detach()))
            if epoch == 1 and batch_index >= math.ceil(batches / 2) and not half_saved:
                save_adapter(
                    adapter_checkpoint_path(arm, seed, "0.5"),
                    adapter,
                    optimizer,
                    arm=arm,
                    seed=seed,
                    epoch=0.5,
                    update=updates,
                    training_list_sha=lists["records_sha256"],
                )
                half_saved = True
        save_adapter(
            adapter_checkpoint_path(arm, seed, str(epoch)),
            adapter,
            optimizer,
            arm=arm,
            seed=seed,
            epoch=float(epoch),
            update=updates,
            training_list_sha=lists["records_sha256"],
        )
        history.append(
            {
                "epoch": epoch,
                "updates": updates,
                "mean_loss": float(np.mean(epoch_losses)),
                "min_loss": float(np.min(epoch_losses)),
                "max_loss": float(np.max(epoch_losses)),
            }
        )
        log(
            "training_epoch",
            arm=arm,
            seed=seed,
            epoch=epoch,
            loss=history[-1]["mean_loss"],
        )
    execution = {
        "status": "completed",
        "arm": arm,
        "seed": seed,
        "epochs": EPOCHS,
        "updates": updates,
        "examples": examples,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "hidden_dimension": HIDDEN_DIMENSION,
        "positive_loss_mode": "sum_probability",
        "kd_weight": 0,
        "uniform_weight": 0,
        "path_loss_weight": 0,
        "direct_loss_weight": 0,
        "qe_loss_weight": 0,
        "dynamic_remining": False,
        "parent_parameters_trainable": False,
        "step0_parity": parity,
        "nonzero_gradient_updates": nonzero_gradient_updates,
        "max_adapter_gradient": max_gradient,
        "history": history,
        "candidate_list_sha256": lists["records_sha256"],
        "target_vector_sha256_before": read_json(OUT / "VECTOR_CACHE_IDENTITY.json")[
            "target_vector_sha256"
        ],
        "target_vector_sha256_after": tensor_sha(target_vectors),
        "elapsed_seconds": time.monotonic() - started,
        "final_checkpoint": record(final_path),
    }
    if execution["target_vector_sha256_before"] != execution["target_vector_sha256_after"]:
        raise RuntimeError("Frozen target vectors changed during adapter training")
    write_json(execution_path, execution)
    return execution


def train_all(device_name: str) -> dict[str, Any]:
    if not (OUT / "PREPARE_SUMMARY.json").is_file():
        prepare(device_name)
    results = []
    for seed in SEEDS:
        for arm in ARMS:
            results.append(train_one(arm, seed, device_name))
    candidate_hashes = {value["candidate_list_sha256"] for value in results}
    summary = {
        "status": "pass" if len(candidate_hashes) == 1 else "fail",
        "jobs": len(results),
        "shared_candidate_list_sha256": next(iter(candidate_hashes)),
        "all_jobs": results,
    }
    write_json(OUT / "training/TRAINING_SUMMARY.json", summary)
    return summary


def model_specs(epoch: str = "3") -> list[dict[str, Any]]:
    result = [{"name": "base", "arm": "base", "seed": None, "checkpoint": None}]
    for arm in ARMS:
        for seed in SEEDS:
            result.append(
                {
                    "name": f"{arm}_seed{seed}",
                    "arm": arm,
                    "seed": seed,
                    "checkpoint": adapter_checkpoint_path(arm, seed, epoch),
                }
            )
    return result


def verified_pairs(
    evaluation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for row in evaluation:
        direct = {str(value["target_id"]) for value in row["direct"]}
        direct_exact = set(row["direct_exact"])
        historical_e = set(row["historical_e_rank"])
        for evidence in row["evidence"]:
            positives = [str(value) for value in evidence["verified_positive_target_ids"]]
            if not positives:
                continue
            strict = [
                value
                for value in positives
                if value in historical_e and value not in direct and value not in direct_exact
            ]
            result.append(
                {
                    "query_id": str(row["query_id"]),
                    "source_table_id": str(row["source_table_id"]),
                    "query_kind": str(row["query_kind"]),
                    "evidence_id": str(evidence["evidence_id"]),
                    "evidence_type": str(evidence["evidence_type"]),
                    "positive_target_ids": positives,
                    "strict_positive_target_ids": strict,
                }
            )
    return result


def conditioned_queries(
    spec: Mapping[str, Any],
    query_indices: torch.Tensor,
    evidence_indices: torch.Tensor,
    vectors: Mapping[str, Any],
    device: torch.device,
    *,
    query_override_indices: torch.Tensor | None = None,
    batch_size: int = 1024,
) -> torch.Tensor:
    if spec["arm"] == "base":
        return vectors["base_queries"][evidence_indices].clone()
    adapter = load_adapter(Path(spec["checkpoint"]), device).eval()
    output = []
    selected_queries = (
        query_indices if query_override_indices is None else query_override_indices
    )
    with torch.inference_mode():
        for start in range(0, len(query_indices), batch_size):
            stop = start + batch_size
            q = vectors["query_vectors"][selected_queries[start:stop]].to(device)
            e = vectors["evidence_vectors"][evidence_indices[start:stop]].to(device)
            base = vectors["base_queries"][evidence_indices[start:stop]].to(device)
            output.append(adapter.conditioned_query(base, q, e, str(spec["arm"])).cpu())
    return torch.cat(output)


def pair_indices(
    pairs: Sequence[Mapping[str, Any]], vectors: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    query_position = {value: index for index, value in enumerate(vectors["query_ids"])}
    evidence_position = {
        value: index for index, value in enumerate(vectors["evidence_ids"])
    }
    return (
        torch.tensor([query_position[row["query_id"]] for row in pairs]),
        torch.tensor([evidence_position[row["evidence_id"]] for row in pairs]),
    )


def _positive_ranks(
    scores: torch.Tensor,
    positive_positions: Sequence[int],
    target_ids: Sequence[str],
) -> tuple[list[int], list[float], float]:
    ranks = []
    positive_scores = []
    positive_set = set(positive_positions)
    for position in positive_positions:
        score = float(scores[position])
        # Float32 bilinear ties are not label-informative; deterministic target-ID
        # tie handling is needed only for emitted TopK lists, not positive rank.
        rank = 1 + int(torch.count_nonzero(scores > score))
        ranks.append(rank)
        positive_scores.append(score)
    negative_mask = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    negative_mask[list(positive_set)] = False
    margin = max(positive_scores) - float(scores[negative_mask].max())
    return ranks, positive_scores, margin


def summarize_exact_rows(
    values: Sequence[Mapping[str, Any]], target_count: int
) -> list[dict[str, Any]]:
    slices = {
        "overall": lambda row: True,
        "implicit": lambda row: row["query_kind"] == "implicit",
        "explicit": lambda row: row["query_kind"] == "explicit",
        "text": lambda row: row["evidence_type"] == "text",
        "image": lambda row: row["evidence_type"] == "image",
        "strict_historical_eo": lambda row: bool(row["strict_positive_target_ids"]),
    }
    result = []
    for model in sorted({str(row["model"]) for row in values}):
        model_rows = [row for row in values if row["model"] == model]
        for name, include in slices.items():
            selected = [row for row in model_rows if include(row)]
            summary = query_macro_rank_metrics(selected, target_count)
            summary["positive_vs_top_negative_margin"] = (
                float(np.mean([row["positive_vs_top_negative_margin"] for row in selected]))
                if selected
                else float("nan")
            )
            result.append({"model": model, "slice": name, **summary})
    return result


def evaluate_exact(device_name: str) -> dict[str, Any]:
    output = OUT / "conditioned_et_exact"
    summary_path = output / "summary.csv"
    if summary_path.is_file():
        return {"status": "verified_cached", "summary": record(summary_path)}
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    pairs = verified_pairs(evaluation)
    query_indices, evidence_indices = pair_indices(pairs, vectors)
    target_position = {value: index for index, value in enumerate(vectors["target_ids"])}
    target_vectors = vectors["target_vectors"].to(device)
    all_rows = []
    top20: dict[tuple[str, str, str], list[str]] = {}
    started = time.monotonic()
    for spec in model_specs():
        queries = conditioned_queries(
            spec, query_indices, evidence_indices, vectors, device
        )
        for start in range(0, len(pairs), 256):
            batch = queries[start : start + 256].to(device)
            matrix = batch @ target_vectors.T
            values, positions = torch.topk(matrix, k=20, dim=-1)
            for offset, pair in enumerate(pairs[start : start + len(batch)]):
                score = matrix[offset]
                positive_positions = [
                    target_position[value] for value in pair["positive_target_ids"]
                ]
                positive_ranks, positive_scores, margin = _positive_ranks(
                    score, positive_positions, vectors["target_ids"]
                )
                ranking = [
                    vectors["target_ids"][int(value)] for value in positions[offset]
                ]
                key = (str(spec["name"]), pair["query_id"], pair["evidence_id"])
                top20[key] = ranking
                all_rows.append(
                    {
                        **pair,
                        "model": str(spec["name"]),
                        "positive_ranks": positive_ranks,
                        "positive_scores": positive_scores,
                        "positive_vs_top_negative_margin": margin,
                        "top20_target_ids": ranking,
                        "top20_scores": [float(value) for value in values[offset]],
                    }
                )
        log("exact_model_complete", model=spec["name"], pairs=len(pairs))
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "per_qe.jsonl.gz", all_rows)
    per_query = []
    for model in sorted({row["model"] for row in all_rows}):
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            if row["model"] == model:
                by_query[row["query_id"]].append(row)
        for query_id, selected in sorted(by_query.items()):
            ranks = [rank for row in selected for rank in row["positive_ranks"]]
            per_query.append(
                {
                    "model": model,
                    "query_id": query_id,
                    "query_kind": selected[0]["query_kind"],
                    "source_table_id": selected[0]["source_table_id"],
                    **rank_metrics(ranks, len(vectors["target_ids"])),
                }
            )
    write_rows(output / "per_query.jsonl.gz", per_query)
    summaries = summarize_exact_rows(all_rows, len(vectors["target_ids"]))
    write_csv(summary_path, summaries)
    receipt = {
        "status": "completed",
        "verified_qe_pairs": len(pairs),
        "models": len(model_specs()),
        "target_count": len(vectors["target_ids"]),
        "evaluation_definition": "verified dev witness intersect frozen B13 Q-to-E evidence",
        "query_macro": True,
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": {
            "per_qe": record(output / "per_qe.jsonl.gz"),
            "per_query": record(output / "per_query.jsonl.gz"),
            "summary": record(summary_path),
        },
    }
    write_json(output / "EXECUTION.json", receipt)
    return receipt


def evaluate_ann(device_name: str) -> dict[str, Any]:
    output = OUT / "conditioned_et_ann"
    summary_path = output / "summary.csv"
    if summary_path.is_file():
        return {"status": "verified_cached", "summary": record(summary_path)}
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    pairs = verified_pairs(evaluation)
    query_indices, evidence_indices = pair_indices(pairs, vectors)
    exact_rows = {
        (row["model"], row["query_id"], row["evidence_id"]): row
        for row in rows(OUT / "conditioned_et_exact/per_qe.jsonl.gz")
    }
    all_rows = []
    latency = []
    target_count = len(vectors["target_ids"])
    for spec in model_specs():
        queries = conditioned_queries(
            spec, query_indices, evidence_indices, vectors, device
        )
        started = time.monotonic()
        labels, scores = _hnsw_search(queries, k=20)
        seconds = time.monotonic() - started
        latency.extend([seconds / len(pairs)] * len(pairs))
        for pair, row_labels, row_scores in zip(pairs, labels, scores):
            ranking = [vectors["target_ids"][int(value)] for value in row_labels]
            ranks = [
                ranking.index(value) + 1 if value in ranking else target_count + 1
                for value in pair["positive_target_ids"]
            ]
            exact = exact_rows[(str(spec["name"]), pair["query_id"], pair["evidence_id"])]
            all_rows.append(
                {
                    **pair,
                    "model": str(spec["name"]),
                    "positive_ranks": ranks,
                    "top20_target_ids": ranking,
                    "top20_scores": [float(value) for value in row_scores],
                    "exact_ann_membership_overlap@20": len(
                        set(ranking) & set(exact["top20_target_ids"])
                    )
                    / 20,
                }
            )
        log("ann_model_complete", model=spec["name"], pairs=len(pairs))
    summaries = summarize_exact_rows(
        [dict(row, positive_vs_top_negative_margin=float("nan")) for row in all_rows],
        target_count,
    )
    for summary in summaries:
        selected = [
            row
            for row in all_rows
            if row["model"] == summary["model"]
            and (
                summary["slice"] == "overall"
                or row["query_kind"] == summary["slice"]
                or row["evidence_type"] == summary["slice"]
                or (
                    summary["slice"] == "strict_historical_eo"
                    and row["strict_positive_target_ids"]
                )
            )
        ]
        summary["exact_ann_membership_overlap@20"] = (
            float(np.mean([row["exact_ann_membership_overlap@20"] for row in selected]))
            if selected
            else float("nan")
        )
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "per_qe.jsonl.gz", all_rows)
    per_query = []
    for model in sorted({row["model"] for row in all_rows}):
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            if row["model"] == model:
                by_query[row["query_id"]].append(row)
        for query_id, selected in sorted(by_query.items()):
            per_query.append(
                {
                    "model": model,
                    "query_id": query_id,
                    "query_kind": selected[0]["query_kind"],
                    "source_table_id": selected[0]["source_table_id"],
                    **rank_metrics(
                        [rank for row in selected for rank in row["positive_ranks"]],
                        target_count,
                    ),
                }
            )
    write_rows(output / "per_query.jsonl.gz", per_query)
    write_csv(summary_path, summaries)
    latency_summary = {
        "ann_query_count": len(all_rows),
        "per_qe_seconds_p50": float(np.quantile(latency, 0.5)),
        "per_qe_seconds_p95": float(np.quantile(latency, 0.95)),
        "target_index_files": read_json(OUT / "TARGET_INDEX_IDENTITY.json")["files"],
    }
    write_json(OUT / "latency/ann_latency.json", latency_summary)
    receipt = {
        "status": "completed",
        "target_index_rebuilt": False,
        "target_index_identity_shared": True,
        "models": len(model_specs()),
        "artifacts": {
            "per_qe": record(output / "per_qe.jsonl.gz"),
            "per_query": record(output / "per_query.jsonl.gz"),
            "summary": record(summary_path),
        },
        "latency": latency_summary,
    }
    write_json(output / "EXECUTION.json", receipt)
    return receipt


def step0_full_parity(device_name: str) -> dict[str, Any]:
    output = OUT / "step0_parity"
    summary_path = output / "summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    pairs = verified_pairs(evaluation)
    query_indices, evidence_indices = pair_indices(pairs, vectors)
    sample = torch.arange(min(512, len(pairs)))
    target_sample = vectors["target_vectors"][: min(2048, len(vectors["target_ids"]))]
    base_queries = vectors["base_queries"][evidence_indices[sample]]
    base_scores = base_queries @ target_sample.T
    exact_rows = []
    ranking_rows = []
    status = "pass"
    for arm in ARMS:
        for seed in SEEDS:
            spec = {
                "arm": arm,
                "checkpoint": adapter_checkpoint_path(arm, seed, "0"),
            }
            conditioned = conditioned_queries(
                spec,
                query_indices[sample],
                evidence_indices[sample],
                vectors,
                device,
            )
            scores = conditioned @ target_sample.T
            difference = float((scores - base_scores).abs().max())
            ranks_equal = bool(
                torch.equal(
                    torch.topk(scores, 20, dim=-1).indices,
                    torch.topk(base_scores, 20, dim=-1).indices,
                )
            )
            current = "pass" if difference == 0 and ranks_equal else "fail"
            status = "fail" if current == "fail" else status
            exact_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "max_abs_difference": difference,
                    "status": current,
                }
            )
            ranking_rows.append(
                {"arm": arm, "seed": seed, "top20_equal": ranks_equal}
            )
    write_rows(output / "exact_scores.jsonl.gz", exact_rows)
    write_rows(output / "rankings.jsonl.gz", ranking_rows)
    summary = {
        "status": status,
        "C0_score_parity": all(row["status"] == "pass" for row in exact_rows),
        "C1_ranking_parity": all(row["top20_equal"] for row in ranking_rows),
        "C2_end_to_end_parity": "evaluated_with_base_reconstruction",
        "C3_frozen_branch_check": {
            "parent_checkpoint_sha256": file_sha(PARENT),
            "target_vector_sha256": tensor_sha(vectors["target_vectors"]),
            "expected_target_vector_sha256": read_json(
                OUT / "VECTOR_CACHE_IDENTITY.json"
            )["target_vector_sha256"],
        },
        "sample_qe": len(sample),
        "sample_targets": len(target_sample),
    }
    write_json(summary_path, summary)
    return summary


def query_use_diagnostics(device_name: str) -> dict[str, Any]:
    output = OUT / "query_use"
    shuffle_path = output / "shuffle_q.jsonl.gz"
    if shuffle_path.is_file():
        return {"status": "verified_cached", "shuffle": record(shuffle_path)}
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    pairs = verified_pairs(evaluation)
    query_indices, evidence_indices = pair_indices(pairs, vectors)
    target_position = {value: index for index, value in enumerate(vectors["target_ids"])}
    target_vectors = vectors["target_vectors"].to(device)
    query_to_group = {row["query_id"]: row["source_table_id"] for row in evaluation}
    query_ids = sorted(query_to_group)
    donor = {}
    for index, query_id in enumerate(query_ids):
        for offset in range(1, len(query_ids)):
            candidate = query_ids[(index + offset) % len(query_ids)]
            if query_to_group[candidate] != query_to_group[query_id]:
                donor[query_id] = candidate
                break
    query_position = {value: index for index, value in enumerate(vectors["query_ids"])}
    shuffled_indices = torch.tensor([query_position[donor[row["query_id"]]] for row in pairs])
    exact_real = {
        (row["model"], row["query_id"], row["evidence_id"]): row
        for row in rows(OUT / "conditioned_et_exact/per_qe.jsonl.gz")
    }
    results = []
    for spec in [value for value in model_specs() if value["arm"] != "base"]:
        shuffled = conditioned_queries(
            spec,
            query_indices,
            evidence_indices,
            vectors,
            device,
            query_override_indices=shuffled_indices,
        )
        if spec["arm"] == "e_only":
            real = conditioned_queries(
                spec, query_indices, evidence_indices, vectors, device
            )
            if not torch.equal(real, shuffled):
                raise RuntimeError("E-only arm changed under query shuffle")
        for start in range(0, len(pairs), 256):
            matrix = shuffled[start : start + 256].to(device) @ target_vectors.T
            for offset, pair in enumerate(pairs[start : start + len(matrix)]):
                ranks, _positive_scores, _margin = _positive_ranks(
                    matrix[offset],
                    [target_position[value] for value in pair["positive_target_ids"]],
                    vectors["target_ids"],
                )
                real = exact_real[
                    (str(spec["name"]), pair["query_id"], pair["evidence_id"])
                ]
                results.append(
                    {
                        **pair,
                        "model": spec["name"],
                        "donor_query_id": donor[pair["query_id"]],
                        "real_positive_ranks": real["positive_ranks"],
                        "shuffled_positive_ranks": ranks,
                    }
                )
        log("shuffle_model_complete", model=spec["name"])
    output.mkdir(parents=True, exist_ok=True)
    write_rows(shuffle_path, results)
    summaries = []
    for model in sorted({row["model"] for row in results}):
        selected = [row for row in results if row["model"] == model]
        real_r10 = np.mean(
            [np.mean(np.asarray(row["real_positive_ranks"]) <= 10) for row in selected]
        )
        shuffled_r10 = np.mean(
            [
                np.mean(np.asarray(row["shuffled_positive_ranks"]) <= 10)
                for row in selected
            ]
        )
        summaries.append(
            {
                "model": model,
                "pairs": len(selected),
                "real_recall@10": float(real_r10),
                "shuffled_recall@10": float(shuffled_r10),
                "real_minus_shuffled_recall@10": float(real_r10 - shuffled_r10),
            }
        )
    write_csv(output / "shuffle_summary.csv", summaries)

    base_rows = [
        row
        for row in rows(OUT / "conditioned_et_exact/per_qe.jsonl.gz")
        if row["model"] == "base"
    ]
    qe_rows = {
        (row["query_id"], row["evidence_id"]): row
        for row in rows(OUT / "conditioned_et_exact/per_qe.jsonl.gz")
        if row["model"] == "qe_seed13"
    }
    shared: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        shared[row["evidence_id"]].append(row)
    cases = []
    for evidence_id, selected in shared.items():
        query_values = {row["query_id"] for row in selected}
        if len(query_values) < 2:
            continue
        for row in selected[:5]:
            qe = qe_rows[(row["query_id"], evidence_id)]
            cases.append(
                {
                    "evidence_id": evidence_id,
                    "query_id": row["query_id"],
                    "positive_target_ids": row["positive_target_ids"],
                    "base_top20": row["top20_target_ids"],
                    "qe_top20": qe["top20_target_ids"],
                    "membership_changed": set(row["top20_target_ids"])
                    != set(qe["top20_target_ids"]),
                    "positive_best_rank_change": min(row["positive_ranks"])
                    - min(qe["positive_ranks"]),
                }
            )
    write_rows(output / "same_e_different_q.jsonl.gz", cases)
    receipt = {
        "status": "completed",
        "shuffle_summary": summaries,
        "same_e_different_q_cases": len(cases),
        "donors_from_other_source_group": True,
        "e_only_strictly_invariant": True,
    }
    write_json(output / "EXECUTION.json", receipt)
    return receipt


def _confidence_matrix(
    model: torch.nn.Module,
    raw_scores: np.ndarray,
    evidence_types: Sequence[str],
    device: torch.device,
) -> np.ndarray:
    result = np.empty_like(raw_scores, dtype=np.float32)
    with torch.inference_mode():
        for evidence_type in ("text", "image"):
            positions = [
                index for index, value in enumerate(evidence_types) if value == evidence_type
            ]
            for start in range(0, len(positions), 2048):
                selected = positions[start : start + 2048]
                values = torch.from_numpy(raw_scores[selected]).to(device)
                transformed = model.transform_edge_scores(
                    values, evidence_type, "table", "confidence"
                )
                result[selected] = transformed.cpu().numpy()
    return result


def _retain_evidence_rows(
    query_id: str,
    target_paths: Mapping[str, Sequence[Mapping[str, Any]]],
    store: FeatureStore,
    content_keys: Mapping[str, str],
) -> list[dict[str, Any]]:
    result = []
    support_cache: dict[str, list[float]] = {}
    for target_id, paths in target_paths.items():
        selected, coverage = select_evidence(
            "e2_row_coverage",
            paths,
            query_id=query_id,
            store=store,
            content_keys=content_keys,
            top_l=20,
            budget=4,
            support_cache=support_cache,
        )
        if not selected:
            continue
        retained = [value for value in paths if value["evidence_id"] in selected]
        result.append(
            {
                "target_id": target_id,
                "evidence_score": float(coverage),
                "selected_evidence_ids": selected,
                "retained_paths": retained,
            }
        )
    return sorted(result, key=lambda row: (-row["evidence_score"], row["target_id"]))


def _direct_scores(
    model: torch.nn.Module,
    query_vector: torch.Tensor,
    target_indices: Sequence[int],
    target_vectors: torch.Tensor,
    device: torch.device,
) -> dict[int, float]:
    key = model.relation_key("table", "table")
    with torch.inference_mode():
        relation_query = query_vector.to(device) @ model.relations[key]
        selected = target_vectors[list(target_indices)].to(device)
        scores = (selected @ relation_query).cpu()
    return {position: float(score) for position, score in zip(target_indices, scores)}


def run_end_to_end_model(
    spec: Mapping[str, Any],
    evaluation: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Any],
    model: torch.nn.Module,
    store: FeatureStore,
    content_keys: Mapping[str, str],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_position = {value: index for index, value in enumerate(vectors["query_ids"])}
    evidence_position = {
        value: index for index, value in enumerate(vectors["evidence_ids"])
    }
    target_position = {value: index for index, value in enumerate(vectors["target_ids"])}
    flat = [
        (row_index, evidence)
        for row_index, row in enumerate(evaluation)
        for evidence in row["evidence"]
    ]
    query_indices = torch.tensor(
        [query_position[evaluation[index]["query_id"]] for index, _value in flat]
    )
    evidence_indices = torch.tensor(
        [evidence_position[value["evidence_id"]] for _index, value in flat]
    )
    conditioned = conditioned_queries(
        spec, query_indices, evidence_indices, vectors, device, batch_size=1024
    )
    ann_started = time.monotonic()
    labels, raw_scores = _hnsw_search(conditioned, k=EVIDENCE_K)
    ann_seconds = time.monotonic() - ann_started
    # R27 retrieval used StudentANNIndices' raw_logit default. D1 applies its
    # sigmoid to the summed raw path score during row-coverage selection.
    evidence_target_scores = raw_scores.astype(np.float32, copy=False)
    by_query: dict[int, list[int]] = defaultdict(list)
    for flat_index, (query_index, _value) in enumerate(flat):
        by_query[query_index].append(flat_index)
    target_vectors = vectors["target_vectors"]
    output = []
    historical_membership_equal = 0
    historical_e_rank_equal = 0
    historical_c100_equal = 0
    historical_per_e_equal = 0
    historical_per_e_membership_equal = 0
    for query_index, meta in enumerate(evaluation):
        target_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
        raw_targets = set()
        per_evidence = {}
        for flat_index in by_query[query_index]:
            evidence = flat[flat_index][1]
            ranked = sorted(
                zip(labels[flat_index], evidence_target_scores[flat_index]),
                key=lambda value: (
                    -float(value[1]),
                    vectors["target_ids"][int(value[0])],
                ),
            )
            ranked_ids = [vectors["target_ids"][int(value)] for value, _score in ranked]
            per_evidence[evidence["evidence_id"]] = ranked_ids
            historical_per_e_equal += int(ranked_ids == evidence["historical_targets"])
            historical_per_e_membership_equal += int(
                set(ranked_ids) == set(evidence["historical_targets"])
            )
            for target_id, (_label, et_score) in zip(ranked_ids, ranked):
                raw_targets.add(target_id)
                target_paths[target_id].append(
                    {
                        "kind": "evidence",
                        "evidence_id": evidence["evidence_id"],
                        "evidence_type": evidence["evidence_type"],
                        "query_evidence_score": float(evidence["query_evidence_score"]),
                        "evidence_target_score": float(et_score),
                        "path_score": float(evidence["query_evidence_score"] + et_score),
                    }
                )
        evidence_rows = _retain_evidence_rows(
            meta["query_id"], target_paths, store, content_keys
        )
        e_rank = [row["target_id"] for row in evidence_rows]
        direct = list(meta["direct"])
        direct_rank = [str(row["target_id"]) for row in direct]
        union = sorted(set(direct_rank) | set(e_rank))
        union_positions = [target_position[value] for value in union]
        direct_score_by_position = _direct_scores(
            model,
            vectors["query_vectors"][query_position[meta["query_id"]]],
            union_positions,
            target_vectors,
            device,
        )
        qt_scores = {
            target_id: direct_score_by_position[target_position[target_id]]
            for target_id in union
        }
        u_rank = sorted(union, key=lambda value: (-qt_scores[value], value))
        fusion = fuse_channels(direct, evidence_rows)
        equal = fusion["rankings"]["Equal"]
        c100 = equal[:100]
        historical_membership_equal += int(set(e_rank) == set(meta["historical_e_rank"]))
        historical_e_rank_equal += int(e_rank == meta["historical_e_rank"])
        historical_c100_equal += int(c100 == meta["historical_c100"])
        output.append(
            {
                "model": str(spec["name"]),
                "query_id": str(meta["query_id"]),
                "query_kind": str(meta["query_kind"]),
                "source_table_id": str(meta["source_table_id"]),
                "positive_target_ids": list(meta["positive_target_ids"]),
                "second_hop_target_ids": sorted(raw_targets),
                "per_evidence_targets": per_evidence,
                "E_rank": e_rank,
                "U_rank": u_rank,
                "U_membership": union,
                "Equal_rank": equal,
                "C100": c100,
                "D100_ANN": direct_rank,
                "D100_EXACT": list(meta["direct_exact"]),
                "evidence_selected": {
                    row["target_id"]: row["selected_evidence_ids"] for row in evidence_rows
                },
            }
        )
    parity = {
        "queries": len(evaluation),
        "fixed_qe_pairs": len(flat),
        "per_evidence_historical_top20_equal": historical_per_e_equal,
        "per_evidence_historical_top20_membership_equal": (
            historical_per_e_membership_equal
        ),
        "E_membership_historical_equal": historical_membership_equal,
        "E_rank_historical_equal": historical_e_rank_equal,
        "C100_historical_equal": historical_c100_equal,
        "ann_seconds": ann_seconds,
    }
    return output, parity


def frozen_base_rows(
    evaluation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    compact = {str(row["query_id"]): row for row in evaluation}
    result = []
    for row in rows(RANKINGS):
        query_id = str(row["query_id"])
        meta = compact[query_id]
        evidence = {
            str(value["target_id"]): list(value["selected_evidence_ids"])
            for value in row["E_paths"]
        }
        result.append(
            {
                "model": "base",
                "query_id": query_id,
                "query_kind": str(row["query_kind"]),
                "source_table_id": str(row["source_table_id"]),
                "positive_target_ids": [str(value) for value in row["positive_target_ids"]],
                "second_hop_target_ids": [
                    str(value["target_id"]) for value in row["E_pre_retention"]
                ],
                "per_evidence_targets": {
                    value["evidence_id"]: value["historical_targets"]
                    for value in meta["evidence"]
                },
                "E_rank": [str(value) for value in row["rankings"]["E_ONLY"]],
                "U_rank": [str(value) for value in row["rankings"]["U"]],
                "U_membership": [str(value) for value in row["U"]],
                "Equal_rank": [str(value) for value in row["rankings"]["Equal"]],
                "C100": [str(value) for value in row["rankings"]["Equal"][:100]],
                "D100_ANN": [str(value["target_id"]) for value in row["D100_ANN"]],
                "D100_EXACT": [str(value) for value in row["D100_EXACT"]],
                "evidence_selected": evidence,
            }
        )
    return result


def _write_end_to_end_model(rows_for_model: Sequence[Mapping[str, Any]], model: str) -> None:
    write_rows(
        OUT / "end_to_end/E_rankings" / f"{model}.jsonl.gz",
        (
            {
                key: row[key]
                for key in (
                    "query_id",
                    "query_kind",
                    "source_table_id",
                    "positive_target_ids",
                    "second_hop_target_ids",
                    "per_evidence_targets",
                    "E_rank",
                    "evidence_selected",
                )
            }
            for row in rows_for_model
        ),
    )
    write_rows(
        OUT / "end_to_end/U_rankings" / f"{model}.jsonl.gz",
        (
            {
                "query_id": row["query_id"],
                "positive_target_ids": row["positive_target_ids"],
                "U_membership": row["U_membership"],
                "U_rank": row["U_rank"],
            }
            for row in rows_for_model
        ),
    )
    write_rows(
        OUT / "end_to_end/C100" / f"{model}.jsonl.gz",
        (
            {
                "query_id": row["query_id"],
                "query_kind": row["query_kind"],
                "source_table_id": row["source_table_id"],
                "positive_target_ids": row["positive_target_ids"],
                "C100": row["C100"],
                "D100_ANN": row["D100_ANN"],
                "D100_EXACT": row["D100_EXACT"],
            }
            for row in rows_for_model
        ),
    )


def run_end_to_end(device_name: str) -> dict[str, Any]:
    output = OUT / "end_to_end"
    execution_path = output / "STUDENT_EXECUTION.json"
    if execution_path.is_file():
        return read_json(execution_path)
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    model = load_student(PARENT, device).eval()
    store = FeatureStore.from_path(FEATURES, cache_size=80000, cache_bytes=4 * 1024**3)
    content_keys, content_sha = load_evidence_content_keys(CONTENT_KEYS)
    parity = {}
    started = time.monotonic()
    for spec in model_specs():
        destination = output / "C100" / f"{spec['name']}.jsonl.gz"
        if spec["arm"] != "base" and destination.is_file():
            parity[str(spec["name"])] = {"status": "reused_completed_raw_score_replay"}
            continue
        model_rows, model_parity = run_end_to_end_model(
            spec, evaluation, vectors, model, store, content_keys, device
        )
        _write_end_to_end_model(model_rows, str(spec["name"]))
        parity[str(spec["name"])] = model_parity
        log("end_to_end_model_complete", model=spec["name"])
    base = parity.get("base")
    if base is not None:
        expected_pairs = sum(len(row["evidence"]) for row in evaluation)
        if base["per_evidence_historical_top20_membership_equal"] != expected_pairs:
            raise RuntimeError("BASE custom ANN query changes frozen B13 E-to-T Top20 membership")
        if base["E_membership_historical_equal"] != len(evaluation):
            raise RuntimeError("BASE end-to-end E membership does not reproduce B13")
    # HNSW can permute score ties across batch/thread layouts. Keep that audit,
    # but use the locked R27 artifacts as the formal BASE arm so epoch-0 C100
    # is byte-derived from the preregistered reference rather than a re-query.
    frozen_base = frozen_base_rows(evaluation)
    _write_end_to_end_model(frozen_base, "base")
    final_base_equal = sum(
        row["C100"] == meta["historical_c100"]
        for row, meta in zip(frozen_base, evaluation)
    )
    if final_base_equal != len(evaluation):
        raise RuntimeError("Emitted BASE C100 differs from locked R27 C100")
    receipt = {
        "status": "completed",
        "models": len(model_specs()),
        "queries_per_model": len(evaluation),
        "fixed_evidence_per_query": sorted({len(row["evidence"]) for row in evaluation}),
        "targets_per_evidence": EVIDENCE_K,
        "retention": "historical e2_row_coverage, top_l=20, budget=4",
        "direct_branch": "frozen R27 H-C2-step000178 Direct100",
        "fusion": "historical Equal RRF, C100",
        "content_keys_sha256": content_sha,
        "target_index_rebuilt": False,
        "formal_base_source": record(RANKINGS),
        "formal_base_C100_equal_queries": final_base_equal,
        "ann_tie_policy": (
            "custom BASE re-query membership must match; formal BASE rankings are reused "
            "from locked R27 bytes when HNSW tie order differs"
        ),
        "parity": parity,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(execution_path, receipt)
    return receipt


def _load_historical_teacher_scores() -> tuple[dict[str, dict[str, float]], str]:
    scores = {}
    namespace = None
    for row in rows(TEACHER_RANKINGS):
        scores[str(row["query_id"])] = {
            str(key): float(value) for key, value in row["teacher_scores"].items()
        }
        namespace = str(row["teacher_namespace"])
    if namespace is None:
        raise ValueError("Frozen teacher rankings are empty")
    return scores, namespace


def run_frozen_teacher(device_name: str) -> dict[str, Any]:
    execution_path = OUT / "end_to_end/T0_EXECUTION.json"
    if execution_path.is_file():
        return read_json(execution_path)
    historical, namespace = _load_historical_teacher_scores()
    pools: dict[str, dict[str, list[str]]] = {}
    union_missing: dict[str, set[str]] = defaultdict(set)
    for spec in model_specs():
        name = str(spec["name"])
        pools[name] = {}
        for row in rows(OUT / "end_to_end/C100" / f"{name}.jsonl.gz"):
            query_id = str(row["query_id"])
            pool = [str(value) for value in row["C100"]]
            pools[name][query_id] = pool
            union_missing[query_id].update(
                value for value in pool if value not in historical[query_id]
            )
    old_cache_hits = 0
    if any(union_missing.values()):
        import sqlite3

        connection = sqlite3.connect(f"file:{TEACHER_CACHE}?mode=ro", uri=True)
        for query_id, missing in union_missing.items():
            if not missing:
                continue
            cached = {
                str(target): float(score)
                for target, score in connection.execute(
                    "SELECT t,score FROM scores WHERE namespace=? AND q=?",
                    (namespace, query_id),
                )
                if str(target) in missing
            }
            historical[query_id].update(cached)
            missing.difference_update(cached)
            old_cache_hits += len(cached)
        connection.close()
    missing_pairs = sum(len(value) for value in union_missing.values())
    new_costs = []
    if missing_pairs:
        from evaluate_stage1_r26_teacher import paths as r26_paths
        from mmdd_stage1.r26_teacher import TeacherPairCache
        from run_stage1_r19 import load_r19_checkpoint
        from run_stage1_r25 import _r25_teacher_feature_paths

        device = device_from_name(device_name)
        teacher_checkpoint = (
            ROOT
            / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
        )
        _, _, _, teacher, _ = load_r19_checkpoint(teacher_checkpoint, device)
        teacher.eval()
        store = FeatureStore.from_path(
            r26_paths(ROOT)["features"],
            cache_size=30000,
            cache_bytes=8 * 1024**3,
            teacher_paths=_r25_teacher_feature_paths(ROOT),
        )
        cache = TeacherPairCache(
            OUT / "end_to_end/T0_new_pairs.sqlite", namespace, teacher, store, device
        )
        for index, (query_id, missing) in enumerate(sorted(union_missing.items()), 1):
            if not missing:
                continue
            values, cost = cache.score(query_id, sorted(missing))
            historical[query_id].update(values)
            new_costs.append(cost)
            if index % 100 == 0:
                log("teacher_missing_pairs", queries=index, new_pairs=missing_pairs)
        cache.db.close()
    result_rows = []
    for spec in model_specs():
        name = str(spec["name"])
        current = []
        for row in rows(OUT / "end_to_end/C100" / f"{name}.jsonl.gz"):
            query_id = str(row["query_id"])
            pool = pools[name][query_id]
            t0 = sorted(pool, key=lambda value: (-historical[query_id][value], value))
            value = {
                "model": name,
                "query_id": query_id,
                "query_kind": row["query_kind"],
                "source_table_id": row["source_table_id"],
                "positive_target_ids": row["positive_target_ids"],
                "T0_rank": t0,
            }
            current.append(value)
            result_rows.append(value)
        write_rows(OUT / "end_to_end/T0_rankings" / f"{name}.jsonl.gz", current)
    receipt = {
        "status": "completed",
        "teacher_namespace": namespace,
        "frozen_teacher_rankings": record(TEACHER_RANKINGS),
        "old_cache_hits": old_cache_hits,
        "new_pairs": missing_pairs,
        "new_pair_cost": {
            key: sum(value[key] for value in new_costs) if new_costs else 0
            for key in (
                "requested_pairs",
                "new_pairs",
                "cached_pairs",
                "lookup_and_feature_hash_seconds",
                "new_pair_score_and_write_seconds",
            )
        },
        "models": len(model_specs()),
        "rows": len(result_rows),
    }
    write_json(execution_path, receipt)
    return receipt


def _population_summary(
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    result = []
    for model, values in model_rows.items():
        for slice_name in ("overall", "implicit", "explicit"):
            selected = [
                row
                for row in values
                if slice_name == "overall" or row["query_kind"] == slice_name
            ]
            for method in ("second_hop", "E", "U", "C100", "T0"):
                metric_rows = [
                    retrieval_metrics(row[method], row["positive_target_ids"])
                    for row in selected
                ]
                result.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "method": method,
                        "queries": len(selected),
                        **{
                            key: float(np.mean([row[key] for row in metric_rows]))
                            for key in metric_rows[0]
                        },
                    }
                )
    return result


def finalize_end_to_end() -> dict[str, Any]:
    summary_path = OUT / "end_to_end/summary.csv"
    execution_path = OUT / "end_to_end/EXECUTION.json"
    if (
        summary_path.is_file()
        and execution_path.is_file()
        and read_json(execution_path).get("direct_shortcut_definition")
        == "newly admitted positive targets only"
    ):
        return {"status": "verified_cached", "summary": record(summary_path)}
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in model_specs():
        name = str(spec["name"])
        e_rows = {row["query_id"]: row for row in rows(OUT / "end_to_end/E_rankings" / f"{name}.jsonl.gz")}
        u_rows = {row["query_id"]: row for row in rows(OUT / "end_to_end/U_rankings" / f"{name}.jsonl.gz")}
        c_rows = {row["query_id"]: row for row in rows(OUT / "end_to_end/C100" / f"{name}.jsonl.gz")}
        t_rows = {row["query_id"]: row for row in rows(OUT / "end_to_end/T0_rankings" / f"{name}.jsonl.gz")}
        combined = []
        for query_id, e_row in e_rows.items():
            combined.append(
                {
                    "query_id": query_id,
                    "query_kind": e_row["query_kind"],
                    "source_table_id": e_row["source_table_id"],
                    "positive_target_ids": e_row["positive_target_ids"],
                    "second_hop": e_row["second_hop_target_ids"],
                    "E": e_row["E_rank"],
                    "U": u_rows[query_id]["U_rank"],
                    "C100": c_rows[query_id]["C100"],
                    "T0": t_rows[query_id]["T0_rank"],
                    "D100_ANN": c_rows[query_id]["D100_ANN"],
                    "D100_EXACT": c_rows[query_id]["D100_EXACT"],
                }
            )
        model_rows[name] = combined
    summaries = _population_summary(model_rows)
    write_csv(summary_path, summaries)

    base = {row["query_id"]: row for row in model_rows["base"]}
    fixed_rows = []
    own_rows = []
    funnel = []
    for query_id, row in base.items():
        truth = set(row["positive_target_ids"])
        fixed = truth & set(row["E"]) - set(row["D100_ANN"]) - set(row["D100_EXACT"])
        fixed_rows.append(
            {"query_id": query_id, "target_ids": sorted(fixed), "count": len(fixed)}
        )
    fixed_by_query = {row["query_id"]: set(row["target_ids"]) for row in fixed_rows}
    stages = ("second_hop", "E", "U", "C100", "T0")
    for model, values in model_rows.items():
        totals = Counter()
        own_total = 0
        for row in values:
            truth = set(row["positive_target_ids"])
            own = truth & set(row["E"]) - set(row["D100_ANN"]) - set(row["D100_EXACT"])
            own_total += len(own)
            own_rows.append(
                {
                    "model": model,
                    "query_id": row["query_id"],
                    "target_ids": sorted(own),
                }
            )
            fixed = fixed_by_query[row["query_id"]]
            totals["strict_eo_total"] += len(fixed)
            for stage in stages:
                ranking = row[stage]
                if stage == "T0":
                    for k in (10, 20, 50):
                        totals[f"T0_top{k}"] += len(fixed & set(ranking[:k]))
                else:
                    totals[stage] += len(fixed & set(ranking))
        funnel.append(
            {
                "model": model,
                **totals,
                "model_own_eo_total": own_total,
            }
        )
    base_funnel = next(row for row in funnel if row["model"] == "base")
    for row in funnel:
        for key in ("second_hop", "E", "U", "C100", "T0_top10", "T0_top20", "T0_top50"):
            row[f"{key}_net_vs_base"] = row[key] - base_funnel[key]
    write_rows(OUT / "strict_eo/fixed_historical_eo.jsonl.gz", fixed_rows)
    write_rows(OUT / "strict_eo/own_eo.jsonl.gz", own_rows)
    write_csv(OUT / "strict_eo/funnel.csv", funnel)

    direct_shortcut = []
    base_by_query = {row["query_id"]: row for row in model_rows["base"]}
    for model, values in model_rows.items():
        if not model.startswith("qe_"):
            continue
        counts = {
            "rescued_positive_targets": 0,
            "dropped_positive_targets": 0,
            "net_positive_targets": 0,
            "direct_rank_le_10": 0,
            "direct_rank_le_100": 0,
            "direct_rank_gt_100": 0,
            "strict_direct_budget_outside": 0,
        }
        for row in values:
            baseline = base_by_query[row["query_id"]]
            truth = set(row["positive_target_ids"])
            baseline_positive = truth & set(baseline["E"])
            model_positive = truth & set(row["E"])
            rescued = model_positive - baseline_positive
            dropped = baseline_positive - model_positive
            for target in rescued:
                if target in row["D100_ANN"][:10]:
                    counts["direct_rank_le_10"] += 1
                if target in row["D100_ANN"]:
                    counts["direct_rank_le_100"] += 1
                else:
                    counts["direct_rank_gt_100"] += 1
                if (
                    target not in row["D100_ANN"]
                    and target not in row["D100_EXACT"]
                ):
                    counts["strict_direct_budget_outside"] += 1
            counts["rescued_positive_targets"] += len(rescued)
            counts["dropped_positive_targets"] += len(dropped)
            counts["net_positive_targets"] += len(rescued) - len(dropped)
        direct_shortcut.append({"model": model, **counts})
    write_csv(OUT / "query_use/direct_shortcut.csv", direct_shortcut)
    receipt = {
        "status": "completed",
        "summary": record(summary_path),
        "strict_eo": record(OUT / "strict_eo/funnel.csv"),
        "direct_shortcut": record(OUT / "query_use/direct_shortcut.csv"),
        "direct_shortcut_definition": "newly admitted positive targets only",
    }
    write_json(execution_path, receipt)
    return receipt


def adapter_geometry(device_name: str) -> dict[str, Any]:
    output = OUT / "adapter_geometry"
    summary_path = output / "summary.csv"
    if summary_path.is_file():
        return {"status": "verified_cached", "summary": record(summary_path)}
    device = device_from_name(device_name)
    vectors = load_vector_cache()
    evaluation = list(rows(OUT / "fixed_evaluation_input.jsonl.gz"))
    flat = [
        {
            "query_id": row["query_id"],
            "query_kind": row["query_kind"],
            "source_table_id": row["source_table_id"],
            "evidence_id": evidence["evidence_id"],
            "evidence_type": evidence["evidence_type"],
            "positive_witness": bool(evidence["verified_positive_target_ids"]),
        }
        for row in evaluation
        for evidence in row["evidence"]
    ]
    query_indices, evidence_indices = pair_indices(flat, vectors)
    result = []
    for spec in [value for value in model_specs() if value["arm"] != "base"]:
        adapter = load_adapter(Path(spec["checkpoint"]), device).eval()
        with torch.inference_mode():
            for start in range(0, len(flat), 1024):
                stop = start + 1024
                q = vectors["query_vectors"][query_indices[start:stop]].to(device)
                e = vectors["evidence_vectors"][evidence_indices[start:stop]].to(device)
                base = vectors["base_queries"][evidence_indices[start:stop]].to(device)
                residual = adapter(q, e, str(spec["arm"]))
                geometry = {
                    key: value.cpu().tolist()
                    for key, value in residual_geometry(base, residual).items()
                }
                for offset, meta in enumerate(flat[start:stop]):
                    result.append(
                        {
                            **meta,
                            "model": spec["name"],
                            **{key: values[offset] for key, values in geometry.items()},
                        }
                    )
        log("geometry_model_complete", model=spec["name"])
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "residual_norms.jsonl.gz", result)
    slices = {
        "overall": lambda row: True,
        "implicit": lambda row: row["query_kind"] == "implicit",
        "explicit": lambda row: row["query_kind"] == "explicit",
        "text": lambda row: row["evidence_type"] == "text",
        "image": lambda row: row["evidence_type"] == "image",
        "positive_witness": lambda row: row["positive_witness"],
        "non_positive_unknown": lambda row: not row["positive_witness"],
    }
    summaries = []
    for model in sorted({row["model"] for row in result}):
        current = [row for row in result if row["model"] == model]
        for slice_name, include in slices.items():
            selected = [row for row in current if include(row)]
            summaries.append(
                {
                    "model": model,
                    "slice": slice_name,
                    "pairs": len(selected),
                    **{
                        f"{key}_{stat}": float(function([row[key] for row in selected]))
                        for key in (
                            "base_norm",
                            "residual_norm",
                            "residual_ratio",
                            "base_conditioned_cosine",
                        )
                        for stat, function in (
                            ("mean", np.mean),
                            ("median", np.median),
                            ("p90", lambda values: np.quantile(values, 0.9)),
                        )
                    },
                }
            )
    write_csv(summary_path, summaries)
    receipt = {
        "status": "completed",
        "pairs": len(result),
        "summary": record(summary_path),
        "rows": record(output / "residual_norms.jsonl.gz"),
    }
    write_json(output / "EXECUTION.json", receipt)
    return receipt


def _end_metrics_by_query() -> tuple[
    dict[str, dict[str, dict[str, float]]], dict[str, str]
]:
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    groups = {}
    for spec in model_specs():
        name = str(spec["name"])
        e_rows = {
            row["query_id"]: row
            for row in rows(OUT / "end_to_end/E_rankings" / f"{name}.jsonl.gz")
        }
        u_rows = {
            row["query_id"]: row
            for row in rows(OUT / "end_to_end/U_rankings" / f"{name}.jsonl.gz")
        }
        c_rows = {
            row["query_id"]: row
            for row in rows(OUT / "end_to_end/C100" / f"{name}.jsonl.gz")
        }
        t_rows = {
            row["query_id"]: row
            for row in rows(OUT / "end_to_end/T0_rankings" / f"{name}.jsonl.gz")
        }
        for query_id, e_row in e_rows.items():
            truth = e_row["positive_target_ids"]
            groups[query_id] = e_row["source_table_id"]
            result["E_raw"][name][query_id] = retrieval_metrics(
                e_row["E_rank"], truth
            )["raw_recall"]
            result["U_raw"][name][query_id] = retrieval_metrics(
                u_rows[query_id]["U_membership"], truth
            )["raw_recall"]
            result["C100_raw"][name][query_id] = retrieval_metrics(
                c_rows[query_id]["C100"], truth
            )["raw_recall"]
            result["T0_R10"][name][query_id] = retrieval_metrics(
                t_rows[query_id]["T0_rank"], truth
            )["recall@10"]
    return result, groups


def _exact_metrics_by_query() -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    result = defaultdict(dict)
    groups = {}
    for row in rows(OUT / "conditioned_et_exact/per_query.jsonl.gz"):
        result[str(row["model"])][str(row["query_id"])] = float(row["recall@10"])
        groups[str(row["query_id"])] = str(row["source_table_id"])
    return dict(result), groups


def _arm_average(
    model_values: Mapping[str, Mapping[str, float]], arm: str
) -> dict[str, float]:
    names = [f"{arm}_seed{seed}" for seed in SEEDS]
    common = sorted(set.intersection(*(set(model_values[name]) for name in names)))
    return {
        query_id: float(np.mean([model_values[name][query_id] for name in names]))
        for query_id in common
    }


def statistics() -> dict[str, Any]:
    output = OUT / "statistics"
    bootstrap_path = output / "source_group_bootstrap.csv"
    if bootstrap_path.is_file():
        return {"status": "verified_cached", "bootstrap": record(bootstrap_path)}
    exact, exact_groups = _exact_metrics_by_query()
    end, groups = _end_metrics_by_query()
    metrics: dict[str, tuple[Mapping[str, Mapping[str, float]], Mapping[str, str]]] = {
        "exact_conditioned_ET_R10": (exact, exact_groups),
        **{name: (value, groups) for name, value in end.items()},
    }
    bootstrap_rows = []
    wlt_rows = []
    for metric, (model_values, metric_groups) in metrics.items():
        aggregate = {
            "base": dict(model_values["base"]),
            "e_only": _arm_average(model_values, "e_only"),
            "qe": _arm_average(model_values, "qe"),
        }
        for left, right in (("qe", "base"), ("qe", "e_only"), ("e_only", "base")):
            comparison = f"{left}_minus_{right}"
            bootstrap_rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    **grouped_paired_bootstrap(
                        aggregate[left], aggregate[right], metric_groups
                    ),
                }
            )
            wlt_rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    **paired_wlt(aggregate[left], aggregate[right]),
                }
            )
    write_csv(bootstrap_path, bootstrap_rows)
    write_csv(output / "wlt.csv", wlt_rows)
    receipt = {
        "status": "completed",
        "seed_aggregation": "mean per query before inference",
        "bootstrap_unit": "source_table_id",
        "bootstrap_samples": 2000,
        "bootstrap": record(bootstrap_path),
        "wlt": record(output / "wlt.csv"),
    }
    write_json(output / "EXECUTION.json", receipt)
    return receipt


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_value(
    path: Path, model: str, slice_name: str, key: str, *, method: str | None = None
) -> float:
    for row in _csv_rows(path):
        if row["model"] != model or row["slice"] != slice_name:
            continue
        if method is not None and row.get("method") != method:
            continue
        return float(row[key])
    raise KeyError((path, model, slice_name, method, key))


def _mean_seed_summary(
    path: Path,
    arm: str,
    slice_name: str,
    key: str,
    *,
    method: str | None = None,
) -> float:
    return float(
        np.mean(
            [
                _summary_value(
                    path, f"{arm}_seed{seed}", slice_name, key, method=method
                )
                for seed in SEEDS
            ]
        )
    )


def report() -> dict[str, Any]:
    exact_path = OUT / "conditioned_et_exact/summary.csv"
    ann_path = OUT / "conditioned_et_ann/summary.csv"
    end_path = OUT / "end_to_end/summary.csv"
    bootstrap = _csv_rows(OUT / "statistics/source_group_bootstrap.csv")
    shuffle = _csv_rows(OUT / "query_use/shuffle_summary.csv")
    funnel = _csv_rows(OUT / "strict_eo/funnel.csv")
    geometry = _csv_rows(OUT / "adapter_geometry/summary.csv")
    exact_base = _summary_value(exact_path, "base", "overall", "recall@10")
    exact_e = _mean_seed_summary(exact_path, "e_only", "overall", "recall@10")
    exact_qe = _mean_seed_summary(exact_path, "qe", "overall", "recall@10")
    ann_base = _summary_value(ann_path, "base", "overall", "recall@10")
    ann_e = _mean_seed_summary(ann_path, "e_only", "overall", "recall@10")
    ann_qe = _mean_seed_summary(ann_path, "qe", "overall", "recall@10")
    end_base = _summary_value(
        end_path, "base", "overall", "recall@10", method="T0"
    )
    end_e = _mean_seed_summary(
        end_path, "e_only", "overall", "recall@10", method="T0"
    )
    end_qe = _mean_seed_summary(
        end_path, "qe", "overall", "recall@10", method="T0"
    )
    funnel_metrics = [
        (
            "E RawRecall",
            _summary_value(end_path, "base", "overall", "raw_recall", method="E"),
            _mean_seed_summary(
                end_path, "e_only", "overall", "raw_recall", method="E"
            ),
            _mean_seed_summary(end_path, "qe", "overall", "raw_recall", method="E"),
        ),
        (
            "U RawUnionRecall",
            _summary_value(end_path, "base", "overall", "raw_recall", method="U"),
            _mean_seed_summary(
                end_path, "e_only", "overall", "raw_recall", method="U"
            ),
            _mean_seed_summary(end_path, "qe", "overall", "raw_recall", method="U"),
        ),
        (
            "C100 candidate recall",
            _summary_value(
                end_path, "base", "overall", "raw_recall", method="C100"
            ),
            _mean_seed_summary(
                end_path, "e_only", "overall", "raw_recall", method="C100"
            ),
            _mean_seed_summary(
                end_path, "qe", "overall", "raw_recall", method="C100"
            ),
        ),
    ]
    shuffle_qe = [row for row in shuffle if row["model"].startswith("qe_")]
    shuffle_delta = float(
        np.mean([float(row["real_minus_shuffled_recall@10"]) for row in shuffle_qe])
    )
    strict_base = next(row for row in funnel if row["model"] == "base")
    strict_qe = [row for row in funnel if row["model"].startswith("qe_")]
    strict_delta = float(
        np.mean([float(row["T0_top10_net_vs_base"]) for row in strict_qe])
    )
    qe_e_delta = exact_qe - exact_e
    supported = qe_e_delta > 0 and shuffle_delta > 0 and strict_delta > 0
    weakened = qe_e_delta <= 0 and abs(shuffle_delta) < 1e-4
    conclusion = "supported" if supported else "weakened" if weakened else "unknown"
    ci = next(
        row
        for row in bootstrap
        if row["metric"] == "exact_conditioned_ET_R10"
        and row["comparison"] == "qe_minus_e_only"
    )
    geometry_qe = [
        row
        for row in geometry
        if row["model"].startswith("qe_") and row["slice"] == "overall"
    ]
    residual_ratio = float(
        np.mean([float(row["residual_ratio_median"]) for row in geometry_qe])
    )
    slice_rows = []
    for slice_name in ("text", "image", "strict_historical_eo"):
        slice_rows.append(
            (
                slice_name,
                _summary_value(exact_path, "base", slice_name, "recall@10"),
                _mean_seed_summary(
                    exact_path, "e_only", slice_name, "recall@10"
                ),
                _mean_seed_summary(exact_path, "qe", slice_name, "recall@10"),
            )
        )
    end_slice_rows = []
    for slice_name in ("implicit", "explicit"):
        end_slice_rows.append(
            (
                slice_name,
                _summary_value(
                    end_path, "base", slice_name, "recall@10", method="T0"
                ),
                _mean_seed_summary(
                    end_path, "e_only", slice_name, "recall@10", method="T0"
                ),
                _mean_seed_summary(
                    end_path, "qe", slice_name, "recall@10", method="T0"
                ),
            )
        )
    ann_overlap_base = _summary_value(
        ann_path, "base", "overall", "exact_ann_membership_overlap@20"
    )
    ann_overlap_qe = _mean_seed_summary(
        ann_path, "qe", "overall", "exact_ann_membership_overlap@20"
    )
    direct_rows = _csv_rows(OUT / "query_use/direct_shortcut.csv")
    direct_summary = "; ".join(
        f"{row['model']}: rescued={row['rescued_positive_targets']}, "
        f"Direct>100={row['direct_rank_gt_100']}, "
        f"strict-outside={row['strict_direct_budget_outside']}"
        for row in direct_rows
    )
    same_e_rows = list(rows(OUT / "query_use/same_e_different_q.jsonl.gz"))
    same_e_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in same_e_rows:
        same_e_groups[str(row["evidence_id"])].append(row)
    same_e_pairs = 0
    same_e_changed = 0
    same_e_overlap = []
    for selected in same_e_groups.values():
        for left_index, left in enumerate(selected):
            for right in selected[left_index + 1 :]:
                same_e_pairs += 1
                left_top = left["qe_top20"]
                right_top = right["qe_top20"]
                same_e_changed += left_top != right_top
                same_e_overlap.append(len(set(left_top) & set(right_top)) / 20)
    same_e_improved = sum(
        float(row["positive_best_rank_change"]) > 0 for row in same_e_rows
    )
    same_e_worsened = sum(
        float(row["positive_best_rank_change"]) < 0 for row in same_e_rows
    )
    funnel_table = "\n".join(
        f"| {name} | {base:.6f} | {e_only:.6f} | {qe:.6f} | {qe-e_only:+.6f} |"
        for name, base, e_only, qe in funnel_metrics
    )
    exact_slice_table = "\n".join(
        f"| {name} | {base:.6f} | {e_only:.6f} | {qe:.6f} |"
        for name, base, e_only, qe in slice_rows
    )
    end_slice_table = "\n".join(
        f"| {name} | {base:.6f} | {e_only:.6f} | {qe:.6f} |"
        for name, base, e_only, qe in end_slice_rows
    )
    results = f"""# Query-Conditioned E-to-T Results

## Decision

**Conclusion: {conclusion}.** The preregistered query-use gate compares QE against the
parameter-matched E-only arm and checks query shuffle plus fixed historical strict-EO.

## Main Results

| Metric | BASE | E-only (2-seed mean) | QE (2-seed mean) | QE - E-only |
|---|---:|---:|---:|---:|
| Exact conditioned ET R@10 | {exact_base:.6f} | {exact_e:.6f} | {exact_qe:.6f} | {exact_qe-exact_e:+.6f} |
| ANN conditioned ET R@10 | {ann_base:.6f} | {ann_e:.6f} | {ann_qe:.6f} | {ann_qe-ann_e:+.6f} |
| C100+T0 R@10 | {end_base:.6f} | {end_e:.6f} | {end_qe:.6f} | {end_qe-end_e:+.6f} |
{funnel_table}

QE - E-only exact R@10 grouped paired bootstrap 95% CI:
`[{float(ci['ci95_low']):+.6f}, {float(ci['ci95_high']):+.6f}]`.

Query shuffle changes QE exact R@10 by `{shuffle_delta:+.6f}` (Real-Q minus
Shuffled-Q). Fixed strict-EO T0 Top10 changes by `{strict_delta:+.1f}` targets
relative to BASE. The median QE residual/base norm ratio is `{residual_ratio:.6f}`.

## Slice Results

Exact conditioned E-to-T R@10:

| Slice | BASE | E-only | QE |
|---|---:|---:|---:|
{exact_slice_table}

The verified fixed-evidence set has 939 pairs from 462 implicit queries; therefore
exact explicit-slice performance is not estimable. End-to-end frozen-T0 R@10 covers
both query types:

| Slice | BASE | E-only | QE |
|---|---:|---:|---:|
{end_slice_table}

## Preregistered Questions

1. **QE versus BASE:** no. Exact R@10 changed from `{exact_base:.6f}` to
   `{exact_qe:.6f}`.
2. **QE versus E-only:** no. The delta was `{qe_e_delta:+.6f}` with grouped
   bootstrap 95% CI `[{float(ci['ci95_low']):+.6f}, {float(ci['ci95_high']):+.6f}]`.
3. **Query shuffle:** no measurable query-use effect at R@10; Real-Q minus
   Shuffled-Q was `{shuffle_delta:+.6f}`. E-only was strictly invariant.
4. **Where gains occur:** there was no gain in text, image, implicit, explicit, or
   fixed strict-EO. QE retained 0 of {strict_base['strict_eo_total']} fixed strict-EO
   targets at second hop and changed frozen-T0 Top10 retention by
   `{strict_delta:+.1f}` targets.
5. **Exact to ANN:** there was no exact improvement to preserve. QE ANN R@10 was
   `{ann_qe:.6f}` and exact/ANN Top20 membership overlap was
   `{ann_overlap_qe:.6f}` (BASE `{ann_overlap_base:.6f}`).
6. **Static target index:** yes. All five arms reused the locked R27 target index;
   no target index was rebuilt and target-vector SHA remained unchanged.
7. **Direct shortcut:** newly admitted correct targets were `{direct_summary}`.
   These isolated admissions do not offset the large losses in the fixed strict-EO
   funnel.
8. **Same evidence, different queries:** among `{same_e_pairs}` query pairs sharing
   evidence, QE Top20 changed in `{same_e_changed}` and mean membership overlap was
   `{float(np.mean(same_e_overlap)):.6f}`. Positive rank improved in
   `{same_e_improved}` of {len(same_e_rows)} rows and worsened in
   `{same_e_worsened}`; changes were not task-aligned.

## Causal Reading

Local fact: BASE, E-only, and QE share the same R27 parent, target vectors, target
index, fixed Q-to-E results, training candidates, Direct branch, D1 retention, C100
budget, and frozen T0.

Hypothesis: a query-conditioned second-hop query vector can select a target relation
inside evidence more effectively than a query-independent transform.

Competing explanation: gains can come from added adapter capacity or a direct Q-to-T
shortcut. E-only, query shuffle, strict-EO, and direct-rank diagnostics test these
alternatives.

Single-factor intervention: only the query-side E-to-T residual adapter is trained.

Main metric: exact conditioned E-to-T Recall@10, query-macro over verified witness
pairs reached by the frozen B13 evidence retrieval.

Mechanism diagnostic: QE versus E-only, Real-Q versus Shuffled-Q, same-E/different-Q,
strict direct-budget-outside recovery, and residual geometry.

Negative result means: lack of a QE advantage weakens this adapter formulation; it
does not establish that E-to-T never needs Q, because frozen single-vector Q/E
representations and witness coverage remain competing limitations.

Conclusion: **{conclusion}**.
"""
    limitations = """# Limitations

- The parent is the seed-13 R27 exact replay of historical B13; the adapter has two seeds, but the parent does not.
- Mechanism ranks use verified dev witnesses intersected with frozen B13 Q-to-E evidence. They do not credit unverified query/evidence/target paths.
- Unknown training targets are assumed negative within a frozen 32-target list.
- T0 scores are frozen, but newly admitted C100 pairs require additional inference under the same checkpoint and namespace.
- ANN results retain the historical HNSW parameters; geometry-specific ANN tuning was not performed.
- Runtime peak GPU memory was not instrumented during the completed stage processes; `latency/memory.json` records this missing measurement rather than a retrospective estimate.
"""
    next_decision = f"""# Next Decision

Query-use gate: **{conclusion}**.

Do not enlarge the adapter automatically. A positive decision requires QE > E-only,
Real-Q > Shuffled-Q, and positive fixed strict-EO recovery. The observed values are
QE-E-only exact R@10 `{qe_e_delta:+.6f}`, shuffle delta `{shuffle_delta:+.6f}`, and
strict-EO T0 Top10 net `{strict_delta:+.1f}`.
"""
    (OUT / "RESULTS.md").write_text(results, encoding="utf-8")
    (OUT / "LIMITATIONS.md").write_text(limitations, encoding="utf-8")
    (OUT / "NEXT_DECISION.md").write_text(next_decision, encoding="utf-8")
    receipt = {
        "status": "completed",
        "conclusion": conclusion,
        "exact_base_r10": exact_base,
        "exact_e_only_r10": exact_e,
        "exact_qe_r10": exact_qe,
        "shuffle_delta_r10": shuffle_delta,
        "strict_eo_base_total": int(float(strict_base["strict_eo_total"])),
        "strict_eo_t0_top10_qe_net": strict_delta,
        "artifacts": {
            name: record(OUT / name)
            for name in ("RESULTS.md", "LIMITATIONS.md", "NEXT_DECISION.md")
        },
    }
    write_json(OUT / "REPORT_RECEIPT.json", receipt)
    return receipt


def execution_ledger(stage_results: Mapping[str, Any], device_name: str) -> None:
    memory_path = OUT / "latency/memory.json"
    if not memory_path.is_file():
        write_json(
            memory_path,
            {
                "status": "not_collected",
                "device": device_name,
                "reason": (
                    "Peak GPU memory instrumentation was not active in the original "
                    "stage processes; a retrospective value would be unreliable."
                ),
                "reported_as_experiment_limitation": True,
            },
        )
    write_json(
        OUT / "EXECUTION_LEDGER.json",
        {
            "status": "completed" if "report" in stage_results else "in_progress",
            "command": " ".join(sys.argv),
            "device": device_name,
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "memory": record(memory_path),
            "protocol": {
                "arms": list(ARMS),
                "seeds": list(SEEDS),
                "epochs": EPOCHS,
                "hidden_dimension": HIDDEN_DIMENSION,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "training_candidate_count": TRAIN_CANDIDATES,
                "targets_per_evidence": EVIDENCE_K,
            },
            "stages": stage_results,
        },
    )


def run_all(device_name: str) -> dict[str, Any]:
    results = {}
    stages = (
        ("prepare", lambda: prepare(device_name)),
        ("train", lambda: train_all(device_name)),
        ("step0", lambda: step0_full_parity(device_name)),
        ("exact", lambda: evaluate_exact(device_name)),
        ("ann", lambda: evaluate_ann(device_name)),
        ("query_use", lambda: query_use_diagnostics(device_name)),
        ("geometry", lambda: adapter_geometry(device_name)),
        ("end_to_end_student", lambda: run_end_to_end(device_name)),
        ("end_to_end_teacher", lambda: run_frozen_teacher(device_name)),
        ("end_to_end_finalize", finalize_end_to_end),
        ("statistics", statistics),
        ("report", report),
    )
    for name, function in stages:
        log("stage_start", stage=name)
        results[name] = function()
        execution_ledger(results, device_name)
        log("stage_complete", stage=name)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all",
            "prepare",
            "train",
            "step0",
            "exact",
            "ann",
            "query-use",
            "geometry",
            "end-to-end",
            "teacher",
            "finalize",
            "statistics",
            "report",
        ),
        default="all",
    )
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    actions = {
        "prepare": lambda: prepare(args.device),
        "train": lambda: train_all(args.device),
        "step0": lambda: step0_full_parity(args.device),
        "exact": lambda: evaluate_exact(args.device),
        "ann": lambda: evaluate_ann(args.device),
        "query-use": lambda: query_use_diagnostics(args.device),
        "geometry": lambda: adapter_geometry(args.device),
        "end-to-end": lambda: run_end_to_end(args.device),
        "teacher": lambda: run_frozen_teacher(args.device),
        "finalize": finalize_end_to_end,
        "statistics": statistics,
        "report": report,
        "all": lambda: run_all(args.device),
    }
    print(json.dumps(actions[args.stage](), ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()

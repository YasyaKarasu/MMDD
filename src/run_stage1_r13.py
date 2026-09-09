#!/usr/bin/env python
"""Run the frozen R13 Stage-1 role-projection experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel, split_table_projection
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    fuse_ranked_channels,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
from mmdd_stage1.training import (
    _student_edge_losses,
    _student_path_losses,
    _target_teacher_scores,
    checkpoint,
    student_gradient_norms,
    student_projection_drift,
    student_projection_references,
    student_relation_drift,
)
from mmdd_stage1.witness_supervision import witness_auxiliary_loss
from run_stage1_r12_task_c import (
    _initialize_student,
    _ranking_scores,
    _save_checkpoint,
    _schedule_batches,
    _score_payload,
    _teacher_list_scores,
)
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import (
    _accumulate,
    _empty,
    _finalize,
    _target_channels,
)


CHECKPOINT_STEPS = (0, 45, 89, 178)
KD_CHECKPOINT_STEPS = (0, 178, 356)


def _paths(root: Path) -> dict[str, Path]:
    r12 = root / "work/stage1_optimization_r12_20260908"
    return {
        "plan": root / "stage1_optimization_r13_plan_20260909_revised.md",
        "r12": r12,
        "s0": r12
        / "taskC_training/c_candidates_seed13/checkpoints/step_000356.pt",
        "schedule": r12
        / "taskC_training/candidates_seed13_steps356/candidates.jsonl.gz",
        "schedule_manifest": r12
        / "taskC_training/candidates_seed13_steps356/manifest.json",
        "teacher_manifest": r12 / "taskC_training/teacher_pair_scores/manifest.json",
        "features": root
        / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b",
        "dev_edges": r12
        / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "train_targets": r12
        / "taskA_correctness/supervision/target_lists.train_fit.jsonl",
        "dev_targets": r12
        / "taskA_correctness/supervision/target_lists.dev.jsonl",
        "test_targets": r12
        / "taskA_correctness/supervision/target_lists.r10_test_regression.jsonl",
        "corpus": root
        / "work/stage1_optimization_r10_20260907/stage1_data/stage1_corpus.jsonl",
        "path_hard": r12 / "taskC_training/c2_candidates_seed13/path_hard.jsonl",
        "path_hard_metadata": r12
        / "taskC_training/c2_candidates_seed13/path_hard.jsonl.metadata.json",
        "evidence_content_keys": root
        / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl",
    }


def _output_root(root: Path) -> Path:
    return root / "work/stage1_optimization_r13_20260909"


def freeze_plan(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    output = _output_root(root)
    plan_path = output / "PLAN_FROZEN.json"
    if plan_path.is_file():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        if payload["plan_sha256"] != checkpoint_fingerprint(paths["plan"]):
            raise ValueError("Frozen plan hash differs from the revised R13 plan")
        return payload
    for name, path in paths.items():
        if name not in {"r12", "features"} and not path.is_file():
            raise FileNotFoundError(path)
    s0_manifest = json.loads(
        (
            paths["r12"]
            / "taskC_training/c_candidates_seed13/manifest.json"
        ).read_text(encoding="utf-8")
    )
    s0_hash = checkpoint_fingerprint(paths["s0"])
    if s0_hash != s0_manifest["checkpoints"]["356"]["checkpoint_sha256"]:
        raise ValueError("S0 checkpoint fingerprint differs from its R12 manifest")
    schedule_manifest = json.loads(paths["schedule_manifest"].read_text(encoding="utf-8"))
    if checkpoint_fingerprint(paths["schedule"]) != schedule_manifest["arms"][
        "candidates"
    ]["schedule_sha256"]:
        raise ValueError("Frozen hard-candidate schedule fingerprint mismatch")
    teacher_manifest = json.loads(paths["teacher_manifest"].read_text(encoding="utf-8"))
    teacher_scores = Path(teacher_manifest["scores"])
    if checkpoint_fingerprint(teacher_scores) != teacher_manifest["scores_sha256"]:
        raise ValueError("Frozen Teacher scores fingerprint mismatch")
    output.mkdir(parents=True, exist_ok=True)
    for directory in (
        "taskA_stage1_protocol",
        "taskB_diagnostics_and_kd",
        "taskC_role_projection",
        "taskD_witness_supervision",
        "taskE_conditional_extension",
        "taskF_stage2_deferred",
        "statistics",
    ):
        (output / directory).mkdir(exist_ok=True)
    payload = {
        "format_version": 1,
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(paths["plan"].resolve()),
        "plan_sha256": checkpoint_fingerprint(paths["plan"]),
        "protocol": {
            "primary_metric": "full_dev_macro_target_recall@10",
            "ranking": "D1 retention + F1 union-direct",
            "sensitivity_ranking": "equal union-RRF k=60",
            "retrieval_budget": {
                "direct_k": 100,
                "evidence_k_per_modality": 20,
                "targets_per_evidence": 20,
                "retention_top_l": 20,
                "retention_budget": 4,
                "delivery_n": 50,
                "recall_k": [10, 20, 50],
            },
            "seed": 13,
            "batch_size": 64,
            "first_stage_updates": 178,
            "optimizer": "fresh AdamW",
            "projection_lr": 1e-6,
            "relation_lr": 1e-5,
            "weight_decay": 0.01,
            "kd_weight": 0.3,
            "kd_temperature": 1.0,
            "edge_ranking": "10*sigmoid(raw_logit)",
            "bce_weight": 0.0,
            "anchor_weight_all": 0.1,
            "anchor_weight_evidence": 0.1,
        },
        "inputs": {
            key: {
                "path": str(path.resolve()),
                "sha256": checkpoint_fingerprint(path),
            }
            for key, path in paths.items()
            if key not in {"r12", "features"}
        },
        "feature_manifest": {
            "path": str((paths["features"] / "manifest.jsonl").resolve()),
            "sha256": checkpoint_fingerprint(paths["features"] / "manifest.jsonl"),
        },
        "teacher_scores": {
            "path": str(teacher_scores.resolve()),
            "sha256": teacher_manifest["scores_sha256"],
            "teacher_checkpoint_sha256": teacher_manifest[
                "teacher_checkpoint_sha256"
            ],
        },
        "s0": {
            "id": "c1_candidates356",
            "path": str(paths["s0"].resolve()),
            "sha256": s0_hash,
        },
    }
    write_json(plan_path, payload)
    (output / "runs.jsonl").touch()
    return payload


def _optimizer(model: StudentJoinabilityModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": model.relation_parameters(), "lr": 1e-5},
            {"params": model.projections.parameters(), "lr": 1e-6},
        ],
        weight_decay=0.01,
    )


def _merge_witness_metadata(root: Path):
    paths = _paths(root)
    base = load_target_examples(paths["train_targets"], split="train")
    hard = load_target_examples(paths["path_hard"], split="train")
    by_query = {example.query_id: example for example in base}
    if len(by_query) != len(base):
        raise ValueError("Train-fit witness source has duplicate query IDs")
    merged = []
    for example in hard:
        witness = by_query.get(example.query_id)
        if witness is None:
            raise ValueError(f"Path-hard query lacks train-fit metadata: {example.query_id}")
        if set(example.positive_target_ids) != set(witness.positive_target_ids):
            raise ValueError(f"Positive targets differ for {example.query_id}")
        merged.append(
            replace(
                example,
                positive_evidence_by_target=witness.positive_evidence_by_target,
                positive_evidence_rows_by_target=(
                    witness.positive_evidence_rows_by_target
                ),
                query_row_count=witness.query_row_count,
                query_kind=witness.query_kind,
            )
        )
    return merged


def freeze_path_schedule(root: Path) -> dict[str, Any]:
    plan = freeze_plan(root)
    output = _output_root(root) / "taskD_witness_supervision"
    manifest_path = output / "schedule_manifest.json"
    order_path = output / "schedule_order.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "frozen":
            return payload
        raise ValueError("Existing path schedule manifest is incomplete")
    examples = _merge_witness_metadata(root)
    order = list(range(len(examples)))
    random.Random(13).shuffle(order)
    batches = [order[start : start + 64] for start in range(0, len(order), 64)]
    if len(batches) != 178:
        raise ValueError(f"Expected 178 path batches, found {len(batches)}")
    structural = {
        "queries": len(examples),
        "queries_with_witness_metadata": 0,
        "queries_with_eligible_negative_paths": 0,
        "queries_with_witness_candidate_intersection": 0,
        "positive_pairs_with_intersection": 0,
        "row_groups_with_intersection": 0,
        "supported_path_ids": 0,
    }
    for example in examples:
        positives = set(example.positive_target_ids)
        witness_rows = example.positive_evidence_rows_by_target or {}
        witness_ids = example.positive_evidence_by_target or {}
        structural["queries_with_witness_metadata"] += int(
            bool(witness_rows or witness_ids)
        )
        structural["queries_with_eligible_negative_paths"] += int(
            any(
                candidate.target_id not in positives and candidate.evidence_ids
                for candidate in example.candidates
            )
        )
        query_intersection = False
        for candidate in example.candidates:
            if candidate.target_id not in positives:
                continue
            available = set(candidate.evidence_ids)
            pair_intersection = False
            rows = witness_rows.get(candidate.target_id, {})
            for evidence_id, row_ids in rows.items():
                if evidence_id in available:
                    pair_intersection = True
                    structural["supported_path_ids"] += 1
                    structural["row_groups_with_intersection"] += len(row_ids)
            if not rows:
                overlap = available & set(witness_ids.get(candidate.target_id, ()))
                if overlap:
                    pair_intersection = True
                    structural["supported_path_ids"] += len(overlap)
                    structural["row_groups_with_intersection"] += 1
            structural["positive_pairs_with_intersection"] += int(pair_intersection)
            query_intersection = query_intersection or pair_intersection
        structural["queries_with_witness_candidate_intersection"] += int(
            query_intersection
        )
    write_json(order_path, {"seed": 13, "indices": order})
    paths = _paths(root)
    payload = {
        "format_version": 1,
        "status": "frozen",
        "plan_sha256": plan["plan_sha256"],
        "seed": 13,
        "batch_size": 64,
        "batches": len(batches),
        "batch_sizes": [len(batch) for batch in batches],
        "order": str(order_path.resolve()),
        "order_sha256": checkpoint_fingerprint(order_path),
        "path_hard": {
            "path": str(paths["path_hard"].resolve()),
            "sha256": checkpoint_fingerprint(paths["path_hard"]),
            "metadata_sha256": checkpoint_fingerprint(
                paths["path_hard_metadata"]
            ),
        },
        "witness_source": plan["inputs"]["train_targets"],
        "witness_scope": "any_known_witness_by_row",
        "attribute_scope_available": False,
        "structural_eligibility": structural,
    }
    write_json(manifest_path, payload)
    return payload


def _path_schedule(root: Path):
    manifest = freeze_path_schedule(root)
    examples = _merge_witness_metadata(root)
    order = json.loads(Path(manifest["order"]).read_text(encoding="utf-8"))[
        "indices"
    ]
    ordered = [examples[index] for index in order]
    return [ordered[start : start + 64] for start in range(0, len(ordered), 64)]


@torch.inference_mode()
def _fixed_path_metrics(scores, examples) -> dict[str, Any]:
    result = {}
    for name, channel in (("direct", scores.direct), ("evidence", scores.evidence)):
        eligible = 0
        hits = 0
        for row, example in enumerate(examples):
            width = len(example.candidates)
            mask = channel.candidate_mask[row, :width]
            if name == "evidence" and not bool(mask.any()):
                continue
            eligible += 1
            values = channel.logits[row, :width].masked_fill(~mask, -torch.inf)
            hits += int(bool(channel.positive_mask[row, values.argmax()]))
        result[name] = {
            "eligible_lists": eligible,
            "hits@1": hits,
            "recall@1": hits / eligible if eligible else None,
        }
    return result


def _save_path_checkpoint(
    output: Path,
    step: int,
    model: StudentJoinabilityModel,
    aggregator: PathAggregator,
    fixed_batch,
    store: FeatureStore,
    device: torch.device,
    gradient_norms,
    witness_stats,
) -> dict[str, Any]:
    path = output / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(model, "student-path", aggregator), path)
    model.eval()
    scores = score_target_batch(model, fixed_batch, store, device, aggregator)
    payload = {
        "optimizer_updates": step,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(path),
        "fixed_training_batch": _fixed_path_metrics(scores, fixed_batch),
        "projection_drift_from_pca": student_projection_drift(model),
        "projection_drift_from_stage_start": student_projection_drift(
            model, reference="stage_start"
        ),
        "projection_references": student_projection_references(model),
        "relation_drift": student_relation_drift(model),
        "last_gradient_norms": gradient_norms,
        "last_witness_stats": witness_stats,
    }
    write_json(path.with_suffix(".json"), payload)
    return payload


def train_path_arm(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    schedule_manifest = freeze_path_schedule(args.root)
    arm_id = "p_s_target_only" if args.arm == "path_control" else "p_w_witness"
    output = _output_root(args.root) / "taskD_witness_supervision" / arm_id
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete path arm: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    model = _model(args.root, "shared", device)
    optimizer = _optimizer(model)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    batches = _path_schedule(args.root)
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        paths["evidence_content_keys"]
    )
    started = time.monotonic()
    checkpoints = {
        0: _save_path_checkpoint(
            output,
            0,
            model,
            aggregator,
            batches[0],
            store,
            device,
            None,
            None,
        )
    }
    history = []
    last_gradient_norms = None
    last_witness_stats = None
    aggregate_witness_stats: dict[str, int] = {}
    for step, batch in enumerate(batches, 1):
        model.train()
        student_scores = score_target_batch(
            model, batch, store, device, aggregator
        )
        teacher_scores = _target_teacher_scores(batch, device)
        objective = _student_path_losses(
            model,
            student_scores,
            teacher_scores,
            None,
            temperature=1.0,
            distillation_weight=0.3,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            distillation_rows=None,
            positive_loss_mode="sum_probability",
        )
        witness_loss, witness_stats = witness_auxiliary_loss(
            batch, student_scores, content_keys=content_keys
        )
        loss = objective["loss"] + (
            0.1 * witness_loss if args.arm == "path_witness" else 0.0
        )
        optimizer.zero_grad()
        loss.backward()
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        last_witness_stats = witness_stats
        for key, value in witness_stats.items():
            if isinstance(value, int):
                aggregate_witness_stats[key] = aggregate_witness_stats.get(key, 0) + value
        history.append(
            {
                "optimizer_updates": step,
                "loss": float(loss.detach()),
                "base_path_loss": float(objective["loss"].detach()),
                "supervised_loss": float(objective["supervised_loss"].detach()),
                "distillation_loss": float(objective["distillation_loss"].detach()),
                "anchor_loss": float(objective["anchor_loss"].detach()),
                "witness_loss": float(witness_loss.detach()),
                "weighted_witness_loss": float(
                    (0.1 * witness_loss).detach()
                    if args.arm == "path_witness"
                    else witness_loss.new_zeros(())
                ),
                "witness_stats": witness_stats,
            }
        )
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = _save_path_checkpoint(
                output,
                step,
                model,
                aggregator,
                batches[0],
                store,
                device,
                last_gradient_norms,
                last_witness_stats,
            )
        if step % 25 == 0 or step in CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": arm_id,
                        "step": step,
                        "loss": history[-1]["loss"],
                        "witness_loss": history[-1]["witness_loss"],
                        "eligible_witness_queries": witness_stats[
                            "eligible_queries"
                        ],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm_id,
        "objective": (
            "L_path" if args.arm == "path_control" else "L_path + 0.1 L_W"
        ),
        "unique_algorithm_change": (
            "matched target-only path control"
            if args.arm == "path_control"
            else "positive witness bag auxiliary loss only"
        ),
        "parent_checkpoint": plan["s0"],
        "plan_sha256": plan["plan_sha256"],
        "schedule_manifest": {
            "path": str(
                (_output_root(args.root) / "taskD_witness_supervision/schedule_manifest.json").resolve()
            ),
            "sha256": checkpoint_fingerprint(
                _output_root(args.root)
                / "taskD_witness_supervision/schedule_manifest.json"
            ),
        },
        "schedule": schedule_manifest,
        "content_keys_sha256": content_keys_sha256,
        "optimizer_updates": 178,
        "checkpoints": checkpoints,
        "aggregate_witness_stats": aggregate_witness_stats,
        "history": history,
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "D witness supervision",
                    "arm": arm_id,
                    "status": "complete",
                    "output": str(manifest_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "arm": arm_id}, indent=2))
    return payload


@torch.inference_mode()
def verify_role_migration(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    output = _output_root(args.root) / "taskA_stage1_protocol"
    target = output / "role_migration_step0.json"
    if target.is_file():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("status") == "pass":
            return previous
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    shared = _model(args.root, "shared", device).eval()
    split = split_table_projection(shared).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    scores, teacher_manifest = _score_payload(paths["r12"])
    batches = list(
        _schedule_batches(
            paths["schedule"],
            scores,
            str(teacher_manifest["teacher_checkpoint_sha256"]),
        )
    )[:8]
    edge_examples = [example for _step, batch in batches for example in batch]
    relation_counts: dict[str, int] = {}
    max_pair_difference = 0.0
    max_ann_ip_difference = 0.0
    exact_rank_mismatches = 0
    for start in range(0, len(edge_examples), 64):
        batch = edge_examples[start : start + 64]
        shared_scores = score_edge_batch(shared, batch, store, device).logits
        split_scores = score_edge_batch(split, batch, store, device).logits
        max_pair_difference = max(
            max_pair_difference,
            float((shared_scores - split_scores).abs().max()),
        )
        for row, example in enumerate(batch):
            relation = f"{example.source_type}_to_{example.destination_type}"
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            width = len(example.candidate_ids)
            exact_rank_mismatches += int(
                not torch.equal(
                    shared_scores[row, :width].argsort(descending=True),
                    split_scores[row, :width].argsort(descending=True),
                )
            )
            source = store.embedding_features(example.query_id)
            source_embedding = source.embedding.to(device)
            query = split.relation_query(
                source_embedding,
                source.object_type,
                example.destination_type,
                source_role="query" if source.object_type == "table" else None,
            )
            for candidate_id in example.candidate_ids:
                destination = store.embedding_features(candidate_id)
                index_vector = split.index_vector(
                    destination.embedding.to(device),
                    destination.object_type,
                    destination_role=(
                        "target" if destination.object_type == "table" else None
                    ),
                )
                direct = split.score_embeddings(
                    source_embedding,
                    source.object_type,
                    destination.embedding.to(device),
                    destination.object_type,
                )
                max_ann_ip_difference = max(
                    max_ann_ip_difference, float((query @ index_vector - direct).abs())
                )
    required_relations = {
        "table_to_table",
        "table_to_text",
        "table_to_image",
        "text_to_table",
        "image_to_table",
    }
    missing_relations = sorted(required_relations - relation_counts.keys())
    target_examples = load_target_examples(paths["train_targets"], split="train")[:64]
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    shared_paths = score_target_batch(
        shared, target_examples, store, device, aggregator
    )
    split_paths = score_target_batch(split, target_examples, store, device, aggregator)
    path_direct_difference = float(
        (shared_paths.direct.logits - split_paths.direct.logits).abs().max()
    )
    path_evidence_difference = float(
        (shared_paths.evidence.logits - split_paths.evidence.logits).abs().max()
    )
    passed = (
        not missing_relations
        and max_pair_difference <= 1e-6
        and max_ann_ip_difference <= 1e-5
        and exact_rank_mismatches == 0
        and path_direct_difference <= 1e-6
        and path_evidence_difference <= 1e-6
    )
    payload = {
        "format_version": 1,
        "status": "pass" if passed else "fail",
        "plan_sha256": plan["plan_sha256"],
        "s0_sha256": plan["s0"]["sha256"],
        "edge_examples": len(edge_examples),
        "path_examples": len(target_examples),
        "relation_counts": dict(sorted(relation_counts.items())),
        "missing_relations": missing_relations,
        "max_shared_split_pair_score_abs_difference": max_pair_difference,
        "max_split_ann_ip_direct_score_abs_difference": max_ann_ip_difference,
        "exact_candidate_rank_mismatches": exact_rank_mismatches,
        "max_path_direct_abs_difference": path_direct_difference,
        "max_path_evidence_abs_difference": path_evidence_difference,
        "float32_atol": 1e-6,
        "device": args.device,
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(target, payload)
    if not passed:
        raise RuntimeError("R13 role migration step-0 verification failed")
    print(json.dumps(payload, indent=2))
    return payload


def _model(root: Path, arm: str, device: torch.device) -> StudentJoinabilityModel:
    shared = load_student(_paths(root)["s0"], device)
    if shared.projection_mode != "shared":
        raise ValueError("R13 S0 must use the historical shared projection")
    if arm == "split":
        return split_table_projection(shared)
    shared.reset_projection_anchors()
    return shared


class _DirectScorer:
    def __init__(
        self, model: StudentJoinabilityModel, store: FeatureStore, device: torch.device
    ) -> None:
        self.model = model
        self.store = store
        self.device = device
        self.scored_pairs = 0

    @torch.inference_mode()
    def score(
        self, query_id: str, target_ids: list[str], *, batch_size: int
    ) -> dict[str, float]:
        self.scored_pairs += len(target_ids)
        query = self.store.embedding_features(query_id).for_scoring(
            self.device, include_hidden=False
        )
        result = {}
        for start in range(0, len(target_ids), batch_size):
            ids = target_ids[start : start + batch_size]
            targets = [
                self.store.embedding_features(target_id).for_scoring(
                    self.device, include_hidden=False
                )
                for target_id in ids
            ]
            values = self.model.score_pairs_in_space(
                [query] * len(targets), targets, "raw_logit"
            )
            result.update(
                (target_id, float(value))
                for target_id, value in zip(ids, values.detach().cpu())
            )
        return result


def _paths_by_target(result: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    paths = {}
    for channel in ("direct", "evidence"):
        for row in result[channel]:
            paths[str(row["target_id"])] = row["paths"]
    return paths


def _evaluation_spec(root: Path, arm: str) -> tuple[str, Path, int]:
    if arm == "eval_s0":
        return "s0", _paths(root)["s0"], 0
    if arm in {"eval_kd_on", "eval_kd_off"}:
        arm_id = (
            "b1_kd_on_hard356" if arm == "eval_kd_on" else "b1_kd_off_hard356"
        )
        return (
            arm_id,
            _output_root(root)
            / f"taskB_diagnostics_and_kd/{arm_id}/checkpoints/step_000356.pt",
            356,
        )
    if arm in {"eval_path_control", "eval_path_witness"}:
        arm_id = (
            "p_s_target_only"
            if arm == "eval_path_control"
            else "p_w_witness"
        )
        return (
            arm_id,
            _output_root(root)
            / f"taskD_witness_supervision/{arm_id}/checkpoints/step_000178.pt",
            178,
        )
    role = "shared" if arm == "eval_shared" else "split"
    arm_id = "c_s_shared" if role == "shared" else "c_r_split"
    return (
        arm_id,
        _output_root(root)
        / f"taskC_role_projection/{arm_id}/checkpoints/step_000178.pt",
        178,
    )


def _recall_record(
    record: dict[str, Any],
    f1: list[dict[str, Any]],
    rrf: list[dict[str, Any]],
    pure: list[dict[str, Any]],
) -> dict[str, Any]:
    positives = {str(value) for value in record["positive_target_ids"]}
    evidence_ids = {
        str(path["evidence_id"])
        for paths in record["paths_by_target"].values()
        for path in paths
        if path.get("kind") == "evidence" and path.get("evidence_id") is not None
    }
    result = {
        "query_id": record["query_id"],
        "query_kind": record.get("query_kind"),
        "positive_target_ids": sorted(positives),
        "positive_denominator": len(positives),
        "union_unique_targets": len(f1),
        # One Q->T vector, one Q->E vector for each configured modality, and one
        # E->T vector for every evidence object actually returned by Q->E.
        "search_vectors": 1 + 2 + len(evidence_ids),
        "rankings": {},
    }
    for name, ranking in (("f1_union_direct", f1), ("union_rrf_equal", rrf), ("pure_direct100", pure)):
        unique_ranking = []
        seen_targets = set()
        for row in ranking:
            target_id = str(row["target_id"])
            if target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            unique_ranking.append(row)
        by_k = {}
        for k in (10, 20, 50):
            ids = [str(row["target_id"]) for row in unique_ranking[:k]]
            hits = sorted(positives & set(ids))
            by_k[str(k)] = {
                "target_ids": ids,
                "hit_ids": hits,
                "numerator": len(hits),
                "denominator": len(positives),
                "recall": len(hits) / len(positives),
            }
        result["rankings"][name] = by_k
    return result


def evaluate_stage1(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    arm_id, checkpoint_path, step = _evaluation_spec(args.root, args.arm)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    task_directory = (
        "taskA_stage1_protocol"
        if arm_id == "s0"
        else "taskB_diagnostics_and_kd"
        if arm_id.startswith("b1_")
        else "taskD_witness_supervision"
        if arm_id.startswith("p_")
        else "taskC_role_projection"
    )
    evaluation_name = (
        f"evaluation_step{step}"
        if args.evaluation_split == "dev"
        else f"evaluation_{args.evaluation_split}_step{step}"
    )
    output = (
        _output_root(args.root)
        / task_directory
        / arm_id
        / evaluation_name
    )
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    model = load_student(checkpoint_path, device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    target_path = (
        paths["dev_targets"]
        if args.evaluation_split == "dev"
        else paths["test_targets"]
    )
    target_split = "dev" if args.evaluation_split == "dev" else "test"
    examples = load_target_examples(target_path, split=target_split)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    preload_started = time.monotonic()
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(example.query_id for example in examples),
        ]
    )
    preload_seconds = time.monotonic() - preload_started
    index_dir = output / "index"
    started = time.monotonic()
    index_started = time.monotonic()
    index_manifest = build_indices(
        model,
        store,
        ids_by_type,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=plan["inputs"]["corpus"]["sha256"],
        batch_size=4096,
        num_threads=args.index_threads,
    )
    index_seconds = time.monotonic() - index_started
    indices = StudentANNIndices(
        model,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=plan["inputs"]["corpus"]["sha256"],
        score_space="raw_logit",
    )
    content_keys, content_keys_sha256 = load_evidence_content_keys(
        args.root
        / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
    )
    scorer = _DirectScorer(model, store, device)
    intervention_stats = empty_intervention_stats()
    f1_accumulator = _empty()
    rrf_accumulator = _empty()
    pure_accumulator = _empty()
    ranking_path = output / "rankings.jsonl.gz"
    path_pool_path = output / "path_pool.jsonl.gz"
    processing_latencies = []
    search_vectors_per_query_max = 0
    retrieval_started = time.monotonic()
    with gzip.open(ranking_path, "wt", encoding="utf-8") as ranking_handle, gzip.open(
        path_pool_path, "wt", encoding="utf-8"
    ) as pool_handle:
        for start in range(0, len(examples), args.query_batch_size):
            batch = examples[start : start + args.query_batch_size]
            batch_started = time.monotonic()
            detailed = retrieve_zero_one_hop_detailed_many(
                [example.query_id for example in batch],
                indices,
                k=50,
                direct_k=100,
                evidence_k=20,
                targets_per_evidence=20,
                evidence_types=("text", "image"),
                evidence_aggregation="logsumexp",
                evidence_top_k=4,
                evidence_temperature=1.0,
                path_combination="sum",
                fusion_mode="rrf",
                query_batch_size=args.query_batch_size,
            )
            batch_retrieval_seconds = time.monotonic() - batch_started
            for example, retrieved in zip(batch, detailed):
                record = {
                    "query_id": example.query_id,
                    "dataset": example.dataset,
                    "split": example.split,
                    "positive_target_ids": list(example.positive_target_ids),
                    "positive_evidence_by_target": {
                        key: list(value)
                        for key, value in (example.positive_evidence_by_target or {}).items()
                    },
                    "positive_evidence_rows_by_target": {
                        target_id: {
                            evidence_id: list(rows)
                            for evidence_id, rows in evidence_rows.items()
                        }
                        for target_id, evidence_rows in (
                            example.positive_evidence_rows_by_target or {}
                        ).items()
                    },
                    "query_row_count": example.query_row_count,
                    "query_kind": example.query_kind,
                    "paths_by_target": _paths_by_target(retrieved),
                }
                processing_started = time.monotonic()
                direct, evidence = _target_channels(
                    record,
                    retention="e2_row_coverage",
                    scorer=scorer,
                    store=store,
                    content_keys=content_keys,
                    top_l=20,
                    evidence_budget=4,
                    pair_batch_size=256,
                    intervention="original_mixed",
                    intervention_stats=intervention_stats,
                )
                f1 = direct
                rrf = fuse_ranked_channels(
                    direct, evidence, rrf_k=60, fusion_mode="rrf"
                )
                pure = [row for row in direct if row["original_direct_member"]]
                rankings = {k: f1 for k in (10, 20, 50)}
                rrf_rankings = {k: rrf for k in (10, 20, 50)}
                pure_rankings = {k: pure for k in (10, 20, 50)}
                own_d100 = {
                    str(row["target_id"])
                    for row in direct
                    if row["original_direct_member"]
                }
                _accumulate(f1_accumulator, record, rankings, rankings, own_d100)
                _accumulate(
                    rrf_accumulator, record, rrf_rankings, rankings, own_d100
                )
                _accumulate(
                    pure_accumulator, record, pure_rankings, rankings, own_d100
                )
                recall_record = _recall_record(record, f1, rrf, pure)
                search_vectors_per_query_max = max(
                    search_vectors_per_query_max,
                    int(recall_record["search_vectors"]),
                )
                ranking_handle.write(json.dumps(recall_record) + "\n")
                pool_handle.write(json.dumps(record) + "\n")
                processing_latencies.append(
                    batch_retrieval_seconds / len(batch)
                    + time.monotonic()
                    - processing_started
                )
    retrieval_seconds = time.monotonic() - retrieval_started
    f1_metrics = _finalize(f1_accumulator)
    rrf_metrics = _finalize(rrf_accumulator)
    pure_metrics = _finalize(pure_accumulator)
    latency_values = sorted(processing_latencies)
    index_bytes = sum(
        value.stat().st_size for value in index_dir.iterdir() if value.is_file()
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm_id,
        "step": step,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "projection_mode": model.projection_mode,
        "evaluation_split": args.evaluation_split,
        "target_lists": {
            "path": str(target_path.resolve()),
            "sha256": checkpoint_fingerprint(target_path),
        },
        "plan_sha256": plan["plan_sha256"],
        "protocol": plan["protocol"],
        "primary": f1_metrics,
        "sensitivity_equal_union_rrf": rrf_metrics,
        "pure_direct100": pure_metrics,
        "candidate_recall@50": f1_metrics["recall@50"],
        "stage1_recall@50": f1_metrics["recall@50"],
        "rankings": {
            "path": str(ranking_path.resolve()),
            "sha256": checkpoint_fingerprint(ranking_path),
        },
        "path_pool": {
            "path": str(path_pool_path.resolve()),
            "sha256": checkpoint_fingerprint(path_pool_path),
        },
        "index_manifest": index_manifest,
        "content_keys_sha256": content_keys_sha256,
        "cost": {
            "feature_preload_seconds": preload_seconds,
            "index_build_seconds": index_seconds,
            "index_bytes": index_bytes,
            "retrieval_and_ranking_seconds": retrieval_seconds,
            "elapsed_seconds_excluding_preload": time.monotonic() - started,
            "direct_supplement_pairs": scorer.scored_pairs,
            "queries": len(examples),
            "search_vectors_per_query_max": search_vectors_per_query_max,
            "online_amortized_batch_size": args.query_batch_size,
            "online_seconds_p50": statistics.median(latency_values),
            "online_seconds_p95": latency_values[
                min(len(latency_values) - 1, int(0.95 * len(latency_values)))
            ],
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "index_threads": args.index_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(metrics_path, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "A/C natural Stage-1 evaluation",
                    "arm": arm_id,
                    "status": "complete",
                    "output": str(metrics_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": arm_id,
                "R@10": f1_metrics["recall@10"],
                "R@20": f1_metrics["recall@20"],
                "R@50": f1_metrics["recall@50"],
            },
            indent=2,
        )
    )
    return payload


def train_role_arm(args: argparse.Namespace) -> dict[str, Any]:
    plan = freeze_plan(args.root)
    output = _output_root(args.root)
    arm_id = "c_s_shared" if args.arm == "shared" else "c_r_split"
    arm_dir = output / "taskC_role_projection" / arm_id
    manifest_path = arm_dir / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete R13 arm: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    scores, teacher_manifest = _score_payload(paths["r12"])
    schedule = list(
        _schedule_batches(
            paths["schedule"],
            scores,
            str(teacher_manifest["teacher_checkpoint_sha256"]),
        )
    )
    if len(schedule) < 178 or schedule[0][0] != 1:
        raise ValueError("R13 requires the first 178 frozen R12 candidate batches")
    schedule = schedule[:178]
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    model = _model(args.root, args.arm, device)
    optimizer = _optimizer(model)
    dev_edges = load_edge_examples(paths["dev_edges"], split="dev")
    fixed_batch = schedule[0][1]
    started = time.monotonic()
    checkpoints = {
        0: _save_checkpoint(
            arm_dir, 0, model, dev_edges, fixed_batch, store, device, None
        )
    }
    history = []
    last_gradient_norms = None
    for expected_step, (registered_step, examples) in enumerate(schedule, 1):
        if registered_step != expected_step:
            raise ValueError("Frozen R13 edge schedule order is not contiguous")
        model.train()
        raw = score_edge_batch(
            model, examples, store, device, student_score_space="raw_logit"
        )
        teacher = _teacher_list_scores(examples, raw, device)
        objective = _student_edge_losses(
            model,
            examples,
            raw,
            teacher,
            _ranking_scores(raw),
            None,
            ranking_weight=1.0,
            temperature=1.0,
            distillation_weight=0.3,
            edge_bce_weight=0.0,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            positive_loss_mode="sum_probability",
        )
        optimizer.zero_grad()
        objective["loss"].backward()
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        history.append(
            {
                "optimizer_updates": expected_step,
                "loss": float(objective["loss"].detach()),
                "supervised_loss": float(objective["supervised_loss"].detach()),
                "distillation_loss": float(
                    objective["distillation_loss"].detach()
                ),
                "anchor_loss": float(objective["anchor_loss"].detach()),
                "weighted_anchor_loss": float(
                    objective["weighted_anchor_loss"].detach()
                ),
            }
        )
        if expected_step in CHECKPOINT_STEPS:
            checkpoints[expected_step] = _save_checkpoint(
                arm_dir,
                expected_step,
                model,
                dev_edges,
                fixed_batch,
                store,
                device,
                last_gradient_norms,
            )
        if expected_step % 25 == 0 or expected_step in CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": arm_id,
                        "step": expected_step,
                        "loss": history[-1]["loss"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if set(checkpoints) != set(CHECKPOINT_STEPS):
        raise RuntimeError("R13 did not save every registered role checkpoint")
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm_id,
        "projection_mode": model.projection_mode,
        "unique_algorithm_change": (
            "same-objective shared-projection continuation control"
            if args.arm == "shared"
            else "split only the runtime query/target table projection"
        ),
        "parent_checkpoint": plan["s0"],
        "plan_sha256": plan["plan_sha256"],
        "schedule": plan["inputs"]["schedule"],
        "teacher_scores": plan["teacher_scores"],
        "optimizer_updates": 178,
        "batch_size": 64,
        "optimizer": plan["protocol"],
        "checkpoints": checkpoints,
        "history": history,
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    with (output / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "C role projection",
                    "arm": arm_id,
                    "status": "complete",
                    "output": str(manifest_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "arm": arm_id}, indent=2))
    return payload


def train_kd_arm(args: argparse.Namespace) -> dict[str, Any]:
    """Replay the frozen hard-candidate experiment with KD as the only switch."""

    plan = freeze_plan(args.root)
    kd_weight = 0.3 if args.arm == "kd_on" else 0.0
    arm_id = f"b1_{args.arm}_hard356"
    output = _output_root(args.root) / "taskB_diagnostics_and_kd" / arm_id
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps({"status": "retained", "arm": arm_id}))
            return payload
        raise FileExistsError(f"Inspect incomplete KD arm: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    scores, teacher_manifest = _score_payload(paths["r12"])
    schedule = list(
        _schedule_batches(
            paths["schedule"],
            scores,
            str(teacher_manifest["teacher_checkpoint_sha256"]),
        )
    )
    if len(schedule) != 356 or schedule[0][0] != 1:
        raise ValueError("B1 requires all 356 frozen hard-candidate batches")
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    model = _initialize_student(args.root, device)
    optimizer = _optimizer(model)
    dev_edges = load_edge_examples(paths["dev_edges"], split="dev")
    fixed_batch = schedule[0][1]
    started = time.monotonic()
    checkpoints = {
        0: _save_checkpoint(
            output, 0, model, dev_edges, fixed_batch, store, device, None
        )
    }
    history = []
    last_gradient_norms = None
    for expected_step, (registered_step, examples) in enumerate(schedule, 1):
        if registered_step != expected_step:
            raise ValueError("Frozen hard-candidate schedule order is not contiguous")
        model.train()
        raw = score_edge_batch(
            model, examples, store, device, student_score_space="raw_logit"
        )
        teacher = (
            _teacher_list_scores(examples, raw, device)
            if kd_weight > 0
            else None
        )
        objective = _student_edge_losses(
            model,
            examples,
            raw,
            teacher,
            _ranking_scores(raw),
            None,
            ranking_weight=1.0,
            temperature=1.0,
            distillation_weight=kd_weight,
            edge_bce_weight=0.0,
            anchor_weight=0.1,
            anchor_weight_evidence=0.1,
            positive_loss_mode="sum_probability",
        )
        optimizer.zero_grad()
        objective["loss"].backward()
        last_gradient_norms = student_gradient_norms(model)
        optimizer.step()
        history.append(
            {
                "optimizer_updates": expected_step,
                "loss": float(objective["loss"].detach()),
                "supervised_loss": float(objective["supervised_loss"].detach()),
                "distillation_loss": (
                    float(objective["distillation_loss"].detach())
                    if kd_weight > 0
                    else None
                ),
                "weighted_anchor_loss": float(
                    objective["weighted_anchor_loss"].detach()
                ),
            }
        )
        if expected_step in KD_CHECKPOINT_STEPS:
            checkpoints[expected_step] = _save_checkpoint(
                output,
                expected_step,
                model,
                dev_edges,
                fixed_batch,
                store,
                device,
                last_gradient_norms,
            )
        if expected_step % 50 == 0 or expected_step in KD_CHECKPOINT_STEPS:
            print(
                json.dumps(
                    {
                        "arm": arm_id,
                        "step": expected_step,
                        "loss": history[-1]["loss"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    if set(checkpoints) != set(KD_CHECKPOINT_STEPS):
        raise RuntimeError("B1 did not save every registered checkpoint")
    payload = {
        "format_version": 1,
        "status": "complete",
        "arm": arm_id,
        "unique_algorithm_change": f"KD coefficient {kd_weight}",
        "initialization": "same PCA-1024 and identity relations",
        "parent_checkpoint": None,
        "plan_sha256": plan["plan_sha256"],
        "schedule": plan["inputs"]["schedule"],
        "teacher_scores": plan["teacher_scores"],
        "teacher_scores_used_in_loss": kd_weight > 0,
        "optimizer_updates": 356,
        "batch_size": 64,
        "optimizer": {
            **plan["protocol"],
            "first_stage_updates": 356,
            "kd_weight": kd_weight,
        },
        "checkpoints": checkpoints,
        "history": history,
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "new_teacher_inference": 0,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(manifest_path, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "B1 matched hard-candidate KD ablation",
                    "arm": arm_id,
                    "status": "complete",
                    "output": str(manifest_path.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "arm": arm_id}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "freeze",
            "verify",
            "shared",
            "split",
            "eval_s0",
            "eval_shared",
            "eval_split",
            "eval_path_control",
            "eval_path_witness",
            "freeze_path",
            "path_control",
            "path_witness",
            "kd_on",
            "kd_off",
            "eval_kd_on",
            "eval_kd_off",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--index-threads", type=int, default=12)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--evaluation-split",
        choices=("dev", "r10_test_regression"),
        default="dev",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.arm == "freeze":
        print(json.dumps(freeze_plan(arguments.root), indent=2))
    elif arguments.arm == "verify":
        verify_role_migration(arguments)
    elif arguments.arm == "freeze_path":
        print(json.dumps(freeze_path_schedule(arguments.root), indent=2))
    elif arguments.arm in {"path_control", "path_witness"}:
        train_path_arm(arguments)
    elif arguments.arm in {"kd_on", "kd_off"}:
        train_kd_arm(arguments)
    elif arguments.arm.startswith("eval_"):
        evaluate_stage1(arguments)
    else:
        train_role_arm(arguments)

#!/usr/bin/env python
"""Run the frozen R13 one-step interference and exact-ET diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel, split_table_projection
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import load_corpus_ids
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
from mmdd_stage1.training import (
    _student_edge_losses,
    _student_path_losses,
    _target_teacher_scores,
    student_gradient_norms,
)
from run_stage1_r12_task_c import (
    _ranking_scores,
    _schedule_batches,
    _score_payload,
    _teacher_list_scores,
)
from run_stage1_r13 import (
    _fixed_path_metrics,
    _model,
    _optimizer,
    _output_root,
    _path_schedule,
    _paths,
    freeze_plan,
)


RELATION_GROUPS = {
    "QT": {"table_to_table"},
    "QE": {"table_to_text", "table_to_image"},
    "ET": {"text_to_table", "image_to_table"},
}


def _relation(example: Any) -> str:
    return f"{example.source_type}_to_{example.destination_type}"


@torch.inference_mode()
def _edge_metrics(
    model: StudentJoinabilityModel,
    examples: list[Any],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    totals: dict[str, dict[str, float]] = {}
    for start in range(0, len(examples), 64):
        batch = examples[start : start + 64]
        scored = score_edge_batch(model, batch, store, device)
        scores = scored.logits
        for row, example in enumerate(batch):
            relation = _relation(example)
            item = totals.setdefault(
                relation, {"lists": 0, "hits@1": 0, "margin_sum": 0.0}
            )
            width = len(example.candidate_ids)
            values = scores[row, :width]
            positive = scored.positive_mask[row, :width]
            negative = ~positive
            margin = values[positive].mean() - values[negative].max()
            item["lists"] += 1
            item["hits@1"] += int(bool(positive[values.argmax()]))
            item["margin_sum"] += float(margin)
    return {
        relation: {
            "lists": int(item["lists"]),
            "recall@1": item["hits@1"] / item["lists"],
            "mean_positive_margin": item["margin_sum"] / item["lists"],
        }
        for relation, item in sorted(totals.items())
    }


@torch.inference_mode()
def _score_distribution(
    model: StudentJoinabilityModel,
    examples: list[Any],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    scores = score_edge_batch(model, examples, store, device).logits
    result = {}
    for name, relations in {"all": set().union(*RELATION_GROUPS.values()), **RELATION_GROUPS}.items():
        values = torch.cat(
            [
                scores[row, : len(example.candidate_ids)]
                for row, example in enumerate(examples)
                if _relation(example) in relations
            ]
        )
        sigmoid = values.sigmoid()
        derivative = 10.0 * sigmoid * (1.0 - sigmoid)
        result[name] = {
            "values": int(values.numel()),
            "raw_quantiles": [
                float(value)
                for value in torch.quantile(
                    values.float(), values.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0])
                )
            ],
            "sigmoid_quantiles": [
                float(value)
                for value in torch.quantile(
                    sigmoid.float(), sigmoid.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0])
                )
            ],
            "ranking_derivative_quantiles": [
                float(value)
                for value in torch.quantile(
                    derivative.float(),
                    derivative.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]),
                )
            ],
        }
    return result


def _delta_norms(
    before: dict[str, torch.Tensor], model: StudentJoinabilityModel
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        if name.startswith("projections.table_query"):
            group = "P_query"
        elif name.startswith("projections.table_target"):
            group = "P_target"
        elif name.startswith("projections.table"):
            group = "P_table"
        elif name.startswith(("projections.text", "projections.image")):
            group = "P_text_image"
        elif name.startswith(("relations.", "relation_as.", "relation_bs.")):
            group = "R"
        else:
            continue
        groups.setdefault(group, []).append(
            float((parameter.detach() - before[name]).double().square().sum())
        )
    return {
        key: math.sqrt(sum(values)) for key, values in sorted(groups.items())
    }


def _one_step(
    base: StudentJoinabilityModel,
    loss_builder: Callable[[StudentJoinabilityModel], torch.Tensor],
) -> tuple[StudentJoinabilityModel, dict[str, Any]]:
    model = copy.deepcopy(base)
    optimizer = _optimizer(model)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.train()
    loss = loss_builder(model)
    optimizer.zero_grad()
    loss.backward()
    gradients = student_gradient_norms(model)
    optimizer.step()
    return model, {
        "loss": float(loss.detach()),
        "gradient_norms": gradients,
        "actual_adamw_delta_norms": _delta_norms(before, model),
    }


def _copy_parameter_block(
    base: StudentJoinabilityModel,
    updated: StudentJoinabilityModel,
    prefixes: tuple[str, ...],
) -> StudentJoinabilityModel:
    result = copy.deepcopy(base)
    source = dict(updated.named_parameters())
    with torch.no_grad():
        for name, parameter in result.named_parameters():
            if name.startswith(prefixes):
                parameter.copy_(source[name])
    return result


def _edge_loss_builder(
    examples: list[Any],
    store: FeatureStore,
    device: torch.device,
    mode: str,
) -> Callable[[StudentJoinabilityModel], torch.Tensor]:
    def build(model: StudentJoinabilityModel) -> torch.Tensor:
        scored = score_edge_batch(model, examples, store, device)
        teacher = _teacher_list_scores(examples, scored, device) if mode != "ranking" else None
        objective = _student_edge_losses(
            model,
            examples,
            scored,
            teacher,
            _ranking_scores(scored),
            None,
            ranking_weight=0.0 if mode == "kd" else 1.0,
            temperature=1.0,
            distillation_weight=0.3 if mode != "ranking" else 0.0,
            edge_bce_weight=0.0,
            anchor_weight=0.1 if mode == "combined" else 0.0,
            anchor_weight_evidence=0.1 if mode == "combined" else 0.0,
            positive_loss_mode="sum_probability",
        )
        return objective["loss"]

    return build


def _path_loss_builder(
    examples: list[Any],
    store: FeatureStore,
    device: torch.device,
    branch: str,
) -> Callable[[StudentJoinabilityModel], torch.Tensor]:
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")

    def build(model: StudentJoinabilityModel) -> torch.Tensor:
        scores = score_target_batch(model, examples, store, device, aggregator)
        teacher = _target_teacher_scores(examples, device)
        objective = _student_path_losses(
            model,
            scores,
            teacher,
            None,
            temperature=1.0,
            distillation_weight=0.3,
            anchor_weight=0.0,
            anchor_weight_evidence=0.0,
            distillation_rows=None,
            positive_loss_mode="sum_probability",
        )
        return (
            objective[f"{branch}_supervised_loss"]
            + 0.3 * objective[f"{branch}_distillation_loss"]
        )

    return build


def _outside_original_rowspace(
    delta: torch.Tensor, original: torch.Tensor
) -> dict[str, float]:
    inside = (delta @ original.T) @ original
    outside = delta - inside
    total = float(delta.double().norm())
    outside_norm = float(outside.double().norm())
    return {
        "delta_frobenius": total,
        "outside_s0_rowspace_frobenius": outside_norm,
        "outside_fraction": outside_norm / total if total else 0.0,
    }


@torch.inference_mode()
def _exact_et(
    models: dict[str, StudentJoinabilityModel],
    evidence_pairs: list[tuple[str, str]],
    target_ids: list[str],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    target_embeddings = torch.stack(
        [store.embedding_features(target_id).embedding for target_id in target_ids]
    )
    result = {}
    for name, model in models.items():
        model.eval()
        projected = []
        for start in range(0, len(target_ids), 2048):
            projected.append(
                model.index_vector(
                    target_embeddings[start : start + 2048].to(device),
                    "table",
                    destination_role="target",
                )
            )
        targets = torch.cat(projected)
        ranks = []
        top_targets = Counter()
        details = []
        for evidence_id, positive_target in evidence_pairs:
            features = store.embedding_features(evidence_id)
            query = model.relation_query(
                features.embedding.to(device), features.object_type, "table"
            )
            scores = query @ targets.T
            positive_index = target_index[positive_target]
            rank = 1 + int((scores > scores[positive_index]).sum())
            ranks.append(rank)
            top = torch.topk(scores, k=min(20, len(target_ids))).indices.tolist()
            top_targets.update(target_ids[index] for index in top)
            details.append(
                {
                    "evidence_id": evidence_id,
                    "positive_target_id": positive_target,
                    "exact_rank": rank,
                }
            )
        result[name] = {
            "pairs": len(ranks),
            "recall@20": sum(rank <= 20 for rank in ranks) / len(ranks),
            "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
            "median_rank": sorted(ranks)[len(ranks) // 2],
            "top20_hub_max_frequency": max(top_targets.values(), default=0),
            "top20_hub_max_share": max(top_targets.values(), default=0) / len(ranks),
            "details": details,
        }
        del targets, projected
        torch.cuda.empty_cache()
    return result


def _historical_one_step_diagnostics(
    base: StudentJoinabilityModel,
    update_edges: list[Any],
    holdout_edges: list[Any],
    update_paths: list[Any],
    holdout_paths: list[Any],
    store: FeatureStore,
    device: torch.device,
) -> dict[str, Any]:
    """Apply the current R13 objectives to a historical checkpoint copy."""

    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    with torch.inference_mode():
        baseline_path = _fixed_path_metrics(
            score_target_batch(base, holdout_paths, store, device, aggregator),
            holdout_paths,
        )
    result: dict[str, Any] = {
        "baseline_holdout_edge": _edge_metrics(base, holdout_edges, store, device),
        "baseline_holdout_path": baseline_path,
        "score_distribution": _score_distribution(base, update_edges, store, device),
        "updates": {},
    }
    combined, info = _one_step(
        base, _edge_loss_builder(update_edges, store, device, "combined")
    )
    info["holdout_edge"] = _edge_metrics(combined, holdout_edges, store, device)
    result["updates"]["edge_combined"] = info
    for block, prefixes in {
        "delta_P_table_only": ("projections.table",),
        "delta_P_text_image_only": ("projections.text", "projections.image"),
        "delta_R_only": ("relations.", "relation_as.", "relation_bs."),
    }.items():
        replaced = _copy_parameter_block(base, combined, prefixes)
        result["updates"][block] = {
            "holdout_edge": _edge_metrics(replaced, holdout_edges, store, device)
        }
        del replaced
    del combined
    for group, relations in RELATION_GROUPS.items():
        batch = [row for row in update_edges if _relation(row) in relations]
        updated, info = _one_step(
            base, _edge_loss_builder(batch, store, device, "combined")
        )
        info["lists"] = len(batch)
        info["holdout_edge"] = _edge_metrics(
            updated, holdout_edges, store, device
        )
        result["updates"][f"edge_{group}"] = info
        del updated
    for mode in ("ranking", "kd"):
        updated, info = _one_step(
            base, _edge_loss_builder(update_edges, store, device, mode)
        )
        info["holdout_edge"] = _edge_metrics(
            updated, holdout_edges, store, device
        )
        result["updates"][f"edge_{mode}"] = info
        del updated
    for branch in ("direct", "evidence"):
        updated, info = _one_step(
            base, _path_loss_builder(update_paths, store, device, branch)
        )
        with torch.inference_mode():
            info["holdout_path"] = _fixed_path_metrics(
                score_target_batch(
                    updated, holdout_paths, store, device, aggregator
                ),
                holdout_paths,
            )
        result["updates"][f"path_{branch}"] = info
        del updated
    return result


def _round_robin_source_sample(
    values: list[Any],
    source_group: Callable[[Any], str],
    identity: Callable[[Any], str],
    count: int,
) -> list[Any]:
    unique = {}
    for value in values:
        unique.setdefault(identity(value), value)
    groups: dict[str, list[Any]] = {}
    for value in unique.values():
        groups.setdefault(source_group(value), []).append(value)
    for group, rows in groups.items():
        rows.sort(
            key=lambda value: hashlib.sha256(
                f"13:{group}:{identity(value)}".encode()
            ).digest()
        )
    group_order = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"13:{group}".encode()).digest(),
    )
    selected = []
    depth = 0
    while len(selected) < count:
        added = 0
        for group in group_order:
            if depth < len(groups[group]):
                selected.append(groups[group][depth])
                added += 1
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} unique records for {count} diagnostics")
    return selected


def _diagnostic_batches(
    root: Path,
    edge_rows: list[Any],
    path_rows: list[Any],
) -> tuple[list[list[Any]], list[list[Any]], dict[str, Any]]:
    supervision_manifest = json.loads(
        (
            _paths(root)["r12"]
            / "taskA_correctness/supervision/manifest.json"
        ).read_text(encoding="utf-8")
    )
    source_by_query = {
        str(row["table_id"]): str(row["source_table_id"])
        for row in iter_dataset_artifact(
            Path(supervision_manifest["dataset_root"]), "query_tables"
        )
    }
    evidence_source = {}
    for example in path_rows:
        group = source_by_query.get(example.query_id, example.query_id)
        for evidence_ids in (example.positive_evidence_by_target or {}).values():
            for evidence_id in evidence_ids:
                evidence_source.setdefault(evidence_id, group)

    def edge_group(example: Any) -> str:
        return (
            source_by_query.get(example.query_id, example.query_id)
            if example.source_type == "table"
            else evidence_source.get(example.query_id, example.query_id)
        )

    def edge_identity(example: Any) -> str:
        return "|".join(
            (
                example.query_id,
                example.source_type,
                example.destination_type,
                *example.candidate_ids,
            )
        )

    by_relation = {
        relation: [value for value in edge_rows if _relation(value) == relation]
        for relation in sorted({_relation(value) for value in edge_rows})
    }
    relation_order = sorted(by_relation)
    selected_by_relation = {
        relation: _round_robin_source_sample(
            by_relation[relation],
            edge_group,
            edge_identity,
            103 if index < 2 else 102,
        )
        for index, relation in enumerate(relation_order)
    }
    selected_edges = []
    depth = 0
    while len(selected_edges) < 512:
        for relation in relation_order:
            rows = selected_by_relation[relation]
            if depth < len(rows):
                selected_edges.append(rows[depth])
        depth += 1
    selected_paths = _round_robin_source_sample(
        path_rows,
        lambda example: source_by_query.get(example.query_id, example.query_id),
        lambda example: example.query_id,
        8 * 64,
    )
    manifest = {
        "policy": "seed13 SHA256 order with source-group round-robin",
        "edge_source_groups": len({edge_group(value) for value in selected_edges}),
        "path_source_groups": len(
            {
                source_by_query.get(value.query_id, value.query_id)
                for value in selected_paths
            }
        ),
        "edge_identities_sha256": hashlib.sha256(
            "\n".join(edge_identity(value) for value in selected_edges).encode()
        ).hexdigest(),
        "path_query_ids_sha256": hashlib.sha256(
            "\n".join(value.query_id for value in selected_paths).encode()
        ).hexdigest(),
        "edge_records": [
            {
                "query_id": value.query_id,
                "relation": _relation(value),
                "source_group": edge_group(value),
            }
            for value in selected_edges
        ],
        "path_records": [
            {
                "query_id": value.query_id,
                "source_group": source_by_query.get(value.query_id, value.query_id),
            }
            for value in selected_paths
        ],
    }
    return (
        [selected_edges[start : start + 64] for start in range(0, 512, 64)],
        [selected_paths[start : start + 64] for start in range(0, 512, 64)],
        manifest,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    plan = freeze_plan(args.root)
    output = _output_root(args.root) / "taskB_diagnostics_and_kd"
    target = output / "b0_one_step_and_exact.json"
    if target.is_file():
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("format_version", 1) >= 3:
            print(json.dumps({"status": "retained", "output": str(target)}))
            return payload
    torch.manual_seed(13)
    torch.cuda.manual_seed_all(13)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = _paths(args.root)
    store = FeatureStore.from_path(paths["features"], cache_size=80_000)
    scores, teacher_manifest = _score_payload(paths["r12"])
    all_edge_batches = list(
        _schedule_batches(
            paths["schedule"],
            scores,
            str(teacher_manifest["teacher_checkpoint_sha256"]),
        )
    )
    all_edges = [row for _step, batch in all_edge_batches for row in batch]
    all_paths = _path_schedule(args.root)
    edge_batches, path_batches, diagnostic_manifest = _diagnostic_batches(
        args.root,
        all_edges,
        [row for batch in all_paths for row in batch],
    )
    update_edges = edge_batches[0]
    holdout_edges = [row for batch in edge_batches[1:] for row in batch]
    update_paths = path_batches[0]
    holdout_paths = [row for batch in path_batches[1:] for row in batch]
    base = _model(args.root, "shared", device).eval()
    base_edge = _edge_metrics(base, holdout_edges, store, device)
    aggregator = PathAggregator("logsumexp", 4, path_combination="sum")
    with torch.inference_mode():
        base_path = _fixed_path_metrics(
            score_target_batch(base, holdout_paths, store, device, aggregator),
            holdout_paths,
        )
    one_step: dict[str, Any] = {}
    combined, info = _one_step(
        base, _edge_loss_builder(update_edges, store, device, "combined")
    )
    info["holdout_edge"] = _edge_metrics(combined, holdout_edges, store, device)
    one_step["edge_combined"] = info
    for block, prefixes in {
        "delta_P_table_only": ("projections.table",),
        "delta_P_text_image_only": ("projections.text", "projections.image"),
        "delta_R_only": ("relations.", "relation_as.", "relation_bs."),
    }.items():
        model = _copy_parameter_block(base, combined, prefixes)
        one_step[block] = {
            "holdout_edge": _edge_metrics(model, holdout_edges, store, device)
        }
        del model
    for group, relations in RELATION_GROUPS.items():
        batch = [row for row in update_edges if _relation(row) in relations]
        model, info = _one_step(
            base, _edge_loss_builder(batch, store, device, "combined")
        )
        info["lists"] = len(batch)
        info["holdout_edge"] = _edge_metrics(model, holdout_edges, store, device)
        one_step[f"edge_{group}"] = info
        del model
    for mode in ("ranking", "kd"):
        model, info = _one_step(
            base, _edge_loss_builder(update_edges, store, device, mode)
        )
        info["holdout_edge"] = _edge_metrics(model, holdout_edges, store, device)
        one_step[f"edge_{mode}"] = info
        del model
    for branch in ("direct", "evidence"):
        model, info = _one_step(
            base, _path_loss_builder(update_paths, store, device, branch)
        )
        with torch.inference_mode():
            info["holdout_path"] = _fixed_path_metrics(
                score_target_batch(model, holdout_paths, store, device, aggregator),
                holdout_paths,
            )
        one_step[f"path_{branch}"] = info
        del model
    split = split_table_projection(base).to(device)
    split_updates = {}
    for group in ("QE", "ET"):
        batch = [row for row in update_edges if _relation(row) in RELATION_GROUPS[group]]
        model, info = _one_step(
            split, _edge_loss_builder(batch, store, device, "combined")
        )
        info["holdout_edge"] = _edge_metrics(model, holdout_edges, store, device)
        split_updates[group] = info
        del model
    role_endpoint = load_student(
        _output_root(args.root)
        / "taskC_role_projection/c_r_split/checkpoints/step_000178.pt",
        device,
    )
    original = base.projections["table"].weight.detach()
    role_subspace = {
        role: _outside_original_rowspace(
            role_endpoint.projections[f"table_{role}"].weight.detach() - original,
            original,
        )
        for role in ("query", "target")
    }
    candidate_counts = Counter(
        candidate
        for batch in edge_batches
        for example in batch
        for candidate in example.candidate_ids
    )
    evidence_pairs = []
    seen = set()
    for example in holdout_paths:
        positive_targets = set(example.positive_target_ids)
        for candidate in example.candidates:
            if candidate.target_id not in positive_targets:
                continue
            known = set(
                (example.positive_evidence_by_target or {}).get(
                    candidate.target_id, ()
                )
            )
            for evidence_id in candidate.evidence_ids:
                pair = (evidence_id, candidate.target_id)
                if evidence_id in known and pair not in seen:
                    evidence_pairs.append(pair)
                    seen.add(pair)
                    if len(evidence_pairs) == 16:
                        break
            if len(evidence_pairs) == 16:
                break
        if len(evidence_pairs) == 16:
            break
    if not evidence_pairs:
        raise ValueError("Fixed path batches contain no labelled ET pair")
    target_ids = load_corpus_ids(paths["corpus"], store)["table"]
    exact_models = {
        "s0": base,
        "historical_edge_r11_epoch2": load_student(
            args.root
            / "work/stage1_optimization_r11_20260908/taskC_clean/c2_long/student_edge.last.pt",
            device,
        ),
        "historical_path_step356": load_student(
            paths["r12"]
            / "taskC_training/c2_path_only_seed13/student_path.steps/step_000356.pt",
            device,
        ),
        "c_r_split_step178": role_endpoint,
    }
    repaired_anchor_models = []
    for name, model in exact_models.items():
        if not name.startswith("historical_"):
            continue
        if not bool(torch.isfinite(model.initial_projection_weights).all()):
            with torch.no_grad():
                model.initial_projection_weights.copy_(
                    base.initial_projection_weights
                )
            repaired_anchor_models.append(name)
    exact_et = _exact_et(
        exact_models, evidence_pairs, target_ids, store, device
    )
    historical_one_step = {
        name: _historical_one_step_diagnostics(
            model,
            update_edges,
            holdout_edges,
            update_paths,
            holdout_paths,
            store,
            device,
        )
        for name, model in exact_models.items()
        if name.startswith("historical_")
    }
    candidate_quality = (
        paths["r12"] / "taskC_training/candidate_quality/summary.json"
    )
    payload = {
        "format_version": 3,
        "status": "complete",
        "identity": "diagnostic temporary one-step copies; no formal training checkpoint",
        "plan_sha256": plan["plan_sha256"],
        "fixed_batches": {
            "edge_batches": 8,
            "edge_update_lists": len(update_edges),
            "edge_holdout_lists": len(holdout_edges),
            "path_batches": 8,
            "path_update_queries": len(update_paths),
            "path_holdout_queries": len(holdout_paths),
            "edge_schedule_sha256": plan["inputs"]["schedule"]["sha256"],
            "source_group_hash_lock": diagnostic_manifest,
        },
        "baseline_holdout_edge": base_edge,
        "baseline_holdout_path": base_path,
        "score_distribution": _score_distribution(
            base, update_edges, store, device
        ),
        "one_step_actual_fresh_adamw": one_step,
        "historical_degraded_current_objective_one_step": historical_one_step,
        "historical_anchor_reference_compatibility": {
            "models_repaired": repaired_anchor_models,
            "reference": "verified S0 full-chain PCA reference",
            "changes_model_scores": False,
            "reason": (
                "Older checkpoint containers predate the persisted full-chain "
                "projection-reference buffer required by the current R13 anchor."
            ),
        },
        "split_role_isolation": split_updates,
        "trained_split_role_subspace": role_subspace,
        "fixed_e_all_target_exact": exact_et,
        "candidate_staleness_and_hubs": {
            "scope": "first eight frozen hard-candidate batches",
            "unique_candidate_ids": len(candidate_counts),
            "candidate_occurrences": sum(candidate_counts.values()),
            "max_candidate_frequency": max(candidate_counts.values()),
            "max_candidate_share": max(candidate_counts.values())
            / sum(candidate_counts.values()),
            "top20": candidate_counts.most_common(20),
        },
        "same_hard_candidate_teacher_audit": {
            "path": str(candidate_quality.resolve()),
            "sha256": checkpoint_fingerprint(candidate_quality),
            "interpretation": (
                "Existing exhaustive 22,784-list audit: Teacher macro R@1 is "
                "0.1238 versus PCA 0.0896 on Raw-mined hard candidates, but "
                "Teacher is worse than PCA on the complete/base lists; the advantage "
                "is conditional on candidate difficulty and not universal."
            ),
        },
        "limitations": {
            "source_bias": (
                "Target-frequency hubs are descriptive. No independently labelled "
                "source-bias intervention is used for model selection."
            ),
            "attribute_scope": (
                "The materialized R13 witness source lacks an attribute field; exact "
                "ET uses known positive evidence-target pairs without claiming a full "
                "attribute-specific negative audit."
            ),
        },
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
            "formal_optimizer_updates": 0,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(target, payload)
    with (_output_root(args.root) / "runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(
                {
                    "task": "B0 one-step and exact diagnostics",
                    "status": "complete",
                    "output": str(target.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())

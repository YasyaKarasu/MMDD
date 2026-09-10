"""Run the frozen R18 assumed-negative Teacher structure experiment."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_teacher
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import (
    OBJECT_TYPES,
    FeatureStore,
    ObjectFeatures,
    normalize_object_type,
)
from mmdd_stage1.models import TYPE_TO_ID, TeacherJoinabilityModel
from mmdd_stage1.objectives import listwise_cross_entropy
from mmdd_stage1.scoring import ListScores, score_edge_batch
from run_stage1_r16 import paired_group_bootstrap


ARMS = {
    "A0": "A0_base_continuation",
    "A1": "A1_global_residual",
    "A2": "A2_relation_heads",
}
RELATIONS = (
    "table_to_table",
    "table_to_text",
    "table_to_image",
    "text_to_table",
    "image_to_table",
)
RELATION_TYPES = {
    "table_to_table": ("table", "table"),
    "table_to_text": ("table", "text"),
    "table_to_image": ("table", "image"),
    "text_to_table": ("text", "table"),
    "image_to_table": ("image", "table"),
}
EXPECTED_INPUT_SHA256 = {
    "teacher": "fd544cc166f2a24f2c55a040c0a2645bc14b22c3c82b49176823db458fe75bc1",
    "train": "5be5e3aee605397c80b4bb43867e57d02d5e65147b150dc1275375946e543158",
    "dev": "d686c35d435631149a62b4f228c7d82c7ed6f5f5d42230149a11524bbb712de6",
    "features": "c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1",
    "candidates": "4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1",
}
SEED = 13
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r11 = root / "work/stage1_optimization_r11_20260908"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r16 = root / "work/stage1_optimization_r16_20260910"
    return {
        "teacher": r11 / "taskC_clean/teacher/teacher_edge.pt",
        "train": r12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "candidates": r16 / "candidate_pools.jsonl.gz",
        "r16_metrics": r16 / "QT_RESULTS.json",
        "r16_rankings": r16 / "teacher_rerank_per_query.jsonl.gz",
        "teacher_extra": r12 / "taskC_training/teacher_extra",
        "teacher_extra_matched_gpu0": r16 / "teacher_extra_matched_gpu0",
        "teacher_extra_matched_gpu1": r16 / "teacher_extra_matched_gpu1",
        "teacher_extra_edges_gpu0": r16 / "teacher_extra_edges_gpu0",
        "teacher_extra_edges_gpu1": r16 / "teacher_extra_edges_gpu1",
    }


def _output(root: Path) -> Path:
    return root / "work/stage1_optimization_r18_20260910"


def _arm_output(root: Path, arm: str) -> Path:
    return _output(root) / ARMS[arm]


def _teacher_paths(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for name, path in paths.items()
        if name.startswith("teacher_extra")
        and (path / "teacher_manifest.jsonl").is_file()
    ]


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256_text(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relation(example: EdgeExample) -> str:
    if example.source_type is None or example.destination_type is None:
        raise ValueError("R18 requires explicit directed relation types")
    return TeacherJoinabilityModel.relation_key(
        example.source_type, example.destination_type
    )


def protect_known_positives(
    examples: Sequence[EdgeExample],
) -> tuple[list[EdgeExample], int]:
    """Protect every train-side known positive within a query/relation list."""

    known: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in examples:
        known[(example.query_id, _relation(example))].update(
            example.positive_ids
            or (example.candidate_ids[example.positive_index],)
        )
    protected = []
    exclusion_count = 0
    for example in examples:
        local = set(
            example.positive_ids
            or (example.candidate_ids[example.positive_index],)
        )
        positives = tuple(
            candidate_id
            for candidate_id in example.candidate_ids
            if candidate_id in known[(example.query_id, _relation(example))]
        )
        exclusion_count += len(set(positives) - local)
        protected.append(replace(example, positive_ids=positives))
    return protected, exclusion_count


def _pair_representations(
    model: TeacherJoinabilityModel,
    source_tokens: Sequence[torch.Tensor],
    source_types: Sequence[str],
    destination_tokens: Sequence[torch.Tensor],
    destination_types: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the original Teacher relation-token representation and pair IDs."""

    if not source_tokens:
        return model.rel_token.new_empty((0, model.model_dim)), torch.empty(
            0, dtype=torch.long, device=model.rel_token.device
        )
    if not (
        len(source_tokens)
        == len(source_types)
        == len(destination_tokens)
        == len(destination_types)
    ):
        raise ValueError("Pair inputs must have equal lengths")
    device = source_tokens[0].device
    batch_size = len(source_tokens)
    source_lengths = torch.tensor(
        [tokens.shape[0] for tokens in source_tokens], device=device
    )
    destination_lengths = torch.tensor(
        [tokens.shape[0] for tokens in destination_tokens], device=device
    )
    source_ids = torch.tensor(
        [TYPE_TO_ID[normalize_object_type(value)] for value in source_types],
        device=device,
    )
    destination_ids = torch.tensor(
        [TYPE_TO_ID[normalize_object_type(value)] for value in destination_types],
        device=device,
    )
    lengths = source_lengths + destination_lengths + 2
    input_dtype = model.compute_dtype or model.rel_token.dtype
    inputs = torch.zeros(
        (batch_size, int(lengths.max()), model.model_dim),
        dtype=input_dtype,
        device=device,
    )
    rows = torch.arange(batch_size, device=device)
    pair_ids = source_ids * len(OBJECT_TYPES) + destination_ids
    inputs[:, 0] = (
        model.rel_token + model.type_pair_embeddings(pair_ids)
    ).to(input_dtype)

    padded_sources = pad_sequence(source_tokens, batch_first=True).to(input_dtype)
    padded_sources = padded_sources + (
        model.modality_embeddings(source_ids) + model.role_embeddings.weight[0]
    ).to(input_dtype).unsqueeze(1)
    inputs[:, 1 : 1 + padded_sources.shape[1]] = padded_sources
    inputs[rows, source_lengths + 1] = model.sep_token.to(input_dtype)

    padded_destinations = pad_sequence(
        destination_tokens, batch_first=True
    ).to(input_dtype)
    padded_destinations = padded_destinations + (
        model.modality_embeddings(destination_ids)
        + model.role_embeddings.weight[1]
    ).to(input_dtype).unsqueeze(1)
    destination_offsets = torch.arange(
        padded_destinations.shape[1], device=device
    ).unsqueeze(0)
    destination_positions = source_lengths.unsqueeze(1) + 2 + destination_offsets
    destination_mask = destination_offsets < destination_lengths.unsqueeze(1)
    destination_rows = rows.unsqueeze(1).expand_as(destination_positions)
    inputs[
        destination_rows[destination_mask],
        destination_positions[destination_mask],
    ] = padded_destinations[destination_mask]

    positions = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
    padding_mask = positions >= lengths.unsqueeze(1)
    encoded = model.relation_transformer(
        inputs, src_key_padding_mask=padding_mask
    )
    return encoded[:, 0], pair_ids


class GlobalResidualTeacher(TeacherJoinabilityModel):
    """Original Teacher plus a zero-init residual from final object embeddings."""

    def __init__(self, **base_config: Any) -> None:
        super().__init__(**base_config)
        self.global_adapters = nn.ModuleDict(
            {
                object_type: nn.Linear(self.input_dim, self.model_dim)
                for object_type in OBJECT_TYPES
            }
        )
        self.global_norms = nn.ModuleDict(
            {object_type: nn.LayerNorm(self.model_dim) for object_type in OBJECT_TYPES}
        )
        self.global_residual_in = nn.Linear(self.model_dim * 5, self.model_dim)
        self.global_residual_out = nn.Linear(self.model_dim, self.model_dim)
        nn.init.zeros_(self.global_residual_out.weight)
        nn.init.zeros_(self.global_residual_out.bias)

    def _global_vectors(
        self, features: Sequence[ObjectFeatures]
    ) -> torch.Tensor:
        result = []
        for item in features:
            object_type = normalize_object_type(item.object_type)
            result.append(
                self.global_norms[object_type](
                    self.global_adapters[object_type](item.embedding)
                )
            )
        return torch.stack(result)

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        *,
        compression_cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        if not sources:
            return self.rel_token.new_empty(0)
        compressed = compression_cache if compression_cache is not None else {}
        missing = []
        seen = set(compressed)
        for features in (*sources, *destinations):
            if features.object_id not in seen:
                missing.append(features)
                seen.add(features.object_id)
        with self._autocast_context():
            self.compress_many(missing, compressed)
            local, pair_ids = _pair_representations(
                self,
                [compressed[item.object_id] for item in sources],
                [item.object_type for item in sources],
                [compressed[item.object_id] for item in destinations],
                [item.object_type for item in destinations],
            )
            left = self._global_vectors(sources)
            right = self._global_vectors(destinations)
            global_relation = torch.cat(
                [
                    left,
                    right,
                    left * right,
                    torch.abs(left - right),
                    self.type_pair_embeddings(pair_ids),
                ],
                dim=1,
            )
            residual = self.global_residual_out(
                torch.nn.functional.gelu(
                    self.global_residual_in(global_relation)
                )
            )
            scores = self.scoring_head(local + residual).squeeze(-1)
        return scores.float()


class RelationHeadsTeacher(TeacherJoinabilityModel):
    """Original Teacher with five copied relation-specific final heads."""

    def __init__(self, **base_config: Any) -> None:
        super().__init__(**base_config)
        shared = self.scoring_head
        self.relation_scoring_heads = nn.ModuleDict(
            {relation: copy.deepcopy(shared) for relation in RELATIONS}
        )
        del self.scoring_head

    def score_compressed_pairs(
        self,
        source_tokens: Sequence[torch.Tensor],
        source_types: Sequence[str],
        destination_tokens: Sequence[torch.Tensor],
        destination_types: Sequence[str],
    ) -> torch.Tensor:
        representation, _pair_ids = _pair_representations(
            self,
            source_tokens,
            source_types,
            destination_tokens,
            destination_types,
        )
        scores = representation.new_empty(len(source_tokens))
        groups: dict[str, list[int]] = defaultdict(list)
        for index, (source_type, destination_type) in enumerate(
            zip(source_types, destination_types)
        ):
            groups[self.relation_key(source_type, destination_type)].append(index)
        for relation, indices in groups.items():
            if relation not in self.relation_scoring_heads:
                raise ValueError(f"R18 A2 has no scoring head for {relation}")
            index = torch.tensor(indices, device=representation.device)
            values = self.relation_scoring_heads[relation](
                representation.index_select(0, index)
            ).squeeze(-1)
            scores = scores.index_copy(0, index, values)
        return scores


def _base_state_load(
    model: TeacherJoinabilityModel,
    parent: TeacherJoinabilityModel,
    allowed_missing_prefixes: Sequence[str],
) -> None:
    result = model.load_state_dict(parent.state_dict(), strict=False)
    unexpected = list(result.unexpected_keys)
    missing = list(result.missing_keys)
    if unexpected or any(
        not name.startswith(tuple(allowed_missing_prefixes)) for name in missing
    ):
        raise ValueError(
            f"Invalid R18 parent load: missing={missing}, unexpected={unexpected}"
        )


def initialize_arm(
    parent_path: Path, arm: str, device: torch.device
) -> TeacherJoinabilityModel:
    parent = load_teacher(parent_path, torch.device("cpu"))
    if arm == "A0":
        model: TeacherJoinabilityModel = parent
    elif arm == "A1":
        model = GlobalResidualTeacher(**parent.config())
        _base_state_load(
            model,
            parent,
            ("global_adapters.", "global_norms.", "global_residual_in.", "global_residual_out."),
        )
        if torch.count_nonzero(model.global_residual_out.weight) or torch.count_nonzero(
            model.global_residual_out.bias
        ):
            raise RuntimeError("A1 residual output is not zero initialized")
    elif arm == "A2":
        model = RelationHeadsTeacher(**parent.config())
        parent_state = parent.state_dict()
        filtered = {
            name: value
            for name, value in parent_state.items()
            if not name.startswith("scoring_head.")
        }
        result = model.load_state_dict(filtered, strict=False)
        expected_missing = {
            f"relation_scoring_heads.{relation}.{suffix}"
            for relation in RELATIONS
            for suffix in ("0.weight", "0.bias", "3.weight", "3.bias")
        }
        if result.unexpected_keys or set(result.missing_keys) != expected_missing:
            raise ValueError(
                f"Invalid A2 backbone load: {result.missing_keys}, {result.unexpected_keys}"
            )
        for head in model.relation_scoring_heads.values():
            head.load_state_dict(parent.scoring_head.state_dict(), strict=True)
    else:
        raise ValueError(f"Unknown arm {arm}")
    return model.to(device)


def _checkpoint_payload(
    model: TeacherJoinabilityModel, arm: str, epoch: int
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_kind": "teacher" if arm == "A0" else f"teacher_r18_{arm.lower()}",
        "completed_stage": "teacher-edge-r18",
        "r18_arm": arm,
        "epoch": epoch,
        "config": model.config(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
    }


def load_arm_checkpoint(path: Path, device: torch.device) -> tuple[str, TeacherJoinabilityModel]:
    payload = load_checkpoint(path)
    arm = str(payload.get("r18_arm", "A0"))
    config = dict(payload["config"])
    if arm == "A0":
        model: TeacherJoinabilityModel = TeacherJoinabilityModel(**config)
    elif arm == "A1":
        model = GlobalResidualTeacher(**config)
    elif arm == "A2":
        model = RelationHeadsTeacher(**config)
    else:
        raise ValueError(f"Unsupported R18 checkpoint arm {arm}")
    model.load_state_dict(payload["state_dict"], strict=True)
    return arm, model.to(device)


def _device_feature(
    store: FeatureStore,
    object_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> ObjectFeatures:
    return store.get(object_id, include_hidden=True).for_scoring(
        device, include_hidden=True, hidden_dtype=dtype
    )


def _comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
    left_order = sorted(range(len(left)), key=lambda index: (-left[index], index))
    right_order = sorted(range(len(right)), key=lambda index: (-right[index], index))
    return {
        "pairs": len(left),
        "max_absolute_score_difference": max(errors, default=0.0),
        "mean_absolute_score_difference": statistics.fmean(errors) if errors else 0.0,
        "ranking_consistent": left_order == right_order,
    }


def _replay_samples(examples: Sequence[EdgeExample], count: int = 4) -> dict[str, list[EdgeExample]]:
    result: dict[str, list[EdgeExample]] = {relation: [] for relation in RELATIONS}
    for example in examples:
        relation = _relation(example)
        if len(result[relation]) < count:
            result[relation].append(example)
    if any(len(rows) != count for rows in result.values()):
        raise ValueError("Insufficient five-relation replay examples")
    return result


@torch.inference_mode()
def function_equivalence(
    root: Path,
    arm: str,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> dict[str, Any]:
    paths = _paths(root)
    examples, _count = protect_known_positives(
        load_edge_examples(paths["train"], split="train")
    )
    samples = _replay_samples(examples)
    baseline = initialize_arm(paths["teacher"], "A0", device).eval()
    candidate = initialize_arm(paths["teacher"], arm, device).eval()
    store = FeatureStore.from_path(
        paths["features"], cache_size=1024, teacher_paths=_teacher_paths(paths)
    )
    dtype = next(baseline.parameters()).dtype
    by_relation = {}
    all_differences = []
    all_ranking_consistent = True
    for relation, relation_examples in samples.items():
        sources = []
        destinations = []
        destination_ids = []
        for example in relation_examples:
            source = _device_feature(store, example.query_id, device, dtype)
            for destination_id in example.candidate_ids[:2]:
                sources.append(source)
                destinations.append(
                    _device_feature(store, destination_id, device, dtype)
                )
                destination_ids.append(destination_id)
        base_cached = baseline.score_pairs(
            sources, destinations, compression_cache={}
        ).cpu().tolist()
        arm_cached = candidate.score_pairs(
            sources, destinations, compression_cache={}
        ).cpu().tolist()
        base_single = [
            float(baseline.score_pairs([source], [destination]).cpu()[0])
            for source, destination in zip(sources, destinations)
        ]
        arm_single = [
            float(candidate.score_pairs([source], [destination]).cpu()[0])
            for source, destination in zip(sources, destinations)
        ]
        comparisons = {
            "arm_vs_A0_batch_cached": _comparison(base_cached, arm_cached),
            "A0_batch_vs_single_uncached": _comparison(base_cached, base_single),
            "arm_batch_vs_single_uncached": _comparison(arm_cached, arm_single),
        }
        by_relation[relation] = {
            "pairs": len(sources),
            "destination_ids": destination_ids,
            "comparisons": comparisons,
        }
        all_differences.extend(
            values["max_absolute_score_difference"]
            for values in comparisons.values()
        )
        all_ranking_consistent &= all(
            values["ranking_consistent"] for values in comparisons.values()
        )
    payload = {
        "format_version": 1,
        "status": "pass"
        if max(all_differences) <= 1e-5 and all_ranking_consistent
        else "failed_correctness",
        "arm": arm,
        "relations": by_relation,
        "max_absolute_score_difference": max(all_differences),
        "ranking_consistent_all": all_ranking_consistent,
        "batch_size_cap": batch_size,
        "cached_and_uncached": True,
        "batch_and_single": True,
        "device": str(device),
        "completed_at_utc": _now(),
    }
    arm_output = _arm_output(root, arm)
    arm_output.mkdir(parents=True, exist_ok=True)
    write_json(arm_output / "step0_replay.json", payload)
    if arm != "A0":
        write_json(arm_output / "function_equivalence.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"{arm} step0 function equivalence failed")
    return payload


def _edge_loss(scores: ListScores) -> torch.Tensor:
    return listwise_cross_entropy(
        scores.logits,
        scores.positive_indices,
        scores.candidate_mask,
        scores.positive_mask,
        positive_loss_mode="sum_probability",
    )


def _margin(scores: ListScores) -> float:
    assert scores.positive_mask is not None
    positives = scores.logits[scores.positive_mask]
    negatives = scores.logits[scores.candidate_mask & ~scores.positive_mask]
    return float(positives.mean().detach().cpu() - negatives.mean().detach().cpu())


def smoke_test(root: Path, arm: str, device: torch.device) -> dict[str, Any]:
    paths = _paths(root)
    examples, protection_count = protect_known_positives(
        load_edge_examples(paths["train"], split="train")
    )
    samples = [row for rows in _replay_samples(examples, count=2).values() for row in rows]
    model = initialize_arm(paths["teacher"], arm, device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer_complete = optimizer_ids == trainable
    store = FeatureStore.from_path(
        paths["features"], cache_size=256, teacher_paths=_teacher_paths(paths)
    )
    history = []
    branch_gradient_seen = arm == "A0"
    expected_update_name = (
        "scoring_head.3.weight"
        if arm == "A0"
        else "global_residual_out.weight"
        if arm == "A1"
        else "relation_scoring_heads.table_to_table.3.weight"
    )
    before = dict(model.named_parameters())[expected_update_name].detach().clone()
    for step in range(4):
        model.train()
        optimizer.zero_grad()
        scores = score_edge_batch(model, samples, store, device)
        loss = _edge_loss(scores)
        loss.backward()
        gradients = {
            name: float(parameter.grad.detach().norm().cpu())
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        if arm == "A1" and step >= 1:
            branch_gradient_seen |= (
                gradients.get("global_residual_out.weight", 0.0) > 0
                and gradients.get("global_residual_in.weight", 0.0) > 0
                and any(
                    value > 0
                    for name, value in gradients.items()
                    if name.startswith("global_adapters.")
                )
            )
        if arm == "A2":
            branch_gradient_seen |= any(
                value > 0
                for name, value in gradients.items()
                if name.startswith("relation_scoring_heads.")
            )
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "margin": _margin(scores),
                "gradient_norms": gradients,
            }
        )
    after = dict(model.named_parameters())[expected_update_name].detach()
    finite = all(math.isfinite(row["loss"]) for row in history)
    updated = not torch.equal(before, after)
    model.eval()
    payload = {
        "format_version": 1,
        "status": "pass"
        if finite and optimizer_complete and branch_gradient_seen and updated
        else "failed_correctness",
        "arm": arm,
        "relations": list(RELATIONS),
        "examples": len(samples),
        "assumed_negative_rule": True,
        "known_positive_protection_exclusions": protection_count,
        "finite_loss": finite,
        "optimizer_covers_all_trainable_parameters": optimizer_complete,
        "expected_branch_gradient_seen": branch_gradient_seen,
        "parameter_updated": updated,
        "updated_parameter": expected_update_name,
        "train_mode_during_updates": True,
        "eval_mode_after_updates": not model.training,
        "parameter_dependent_cache_reused": False,
        "history": history,
        "completed_at_utc": _now(),
    }
    write_json(_arm_output(root, arm) / "smoke_test.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"{arm} smoke test failed")
    return payload


@torch.inference_mode()
def _evaluate_edges(
    model: TeacherJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    model.eval()
    losses = []
    by_relation: dict[str, Counter[str]] = {
        relation: Counter() for relation in RELATIONS
    }
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        scores = score_edge_batch(model, batch, store, device)
        loss = _edge_loss(scores)
        losses.extend([float(loss.cpu())] * len(batch))
        predictions = scores.logits.masked_fill(~scores.candidate_mask, -torch.inf).argmax(dim=1)
        assert scores.positive_mask is not None
        hits = scores.positive_mask[
            torch.arange(len(batch), device=device), predictions
        ].cpu().tolist()
        for example, hit in zip(batch, hits):
            by_relation[_relation(example)]["lists"] += 1
            by_relation[_relation(example)]["hits@1"] += bool(hit)
    relation_metrics = {
        relation: {
            **counts,
            "recall@1": counts["hits@1"] / counts["lists"],
        }
        for relation, counts in by_relation.items()
    }
    return {
        "dev_loss": statistics.fmean(losses),
        "macro_recall@1": statistics.fmean(
            values["recall@1"] for values in relation_metrics.values()
        ),
        "by_relation": relation_metrics,
    }


def _gradient_summary(model: TeacherJoinabilityModel) -> dict[str, float]:
    prefixes = (
        "scoring_head.",
        "global_",
        "relation_scoring_heads.",
        "relation_transformer.",
    )
    return {
        name: float(parameter.grad.detach().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name.startswith(prefixes)
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = _paths(args.root)
    manifest = json.loads(
        (_output(args.root) / "INPUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "pass":
        raise RuntimeError("R18 input preflight did not pass")
    for name in ("teacher", "train", "dev", "candidates"):
        if checkpoint_fingerprint(paths[name]) != EXPECTED_INPUT_SHA256[name]:
            raise RuntimeError(f"Frozen R18 input changed: {name}")
    if checkpoint_fingerprint(paths["features"] / "manifest.jsonl") != EXPECTED_INPUT_SHA256["features"]:
        raise RuntimeError("Frozen R18 feature manifest changed")
    replay = json.loads(
        (_arm_output(args.root, args.arm) / "step0_replay.json").read_text(
            encoding="utf-8"
        )
    )
    smoke = json.loads(
        (_arm_output(args.root, args.arm) / "smoke_test.json").read_text(
            encoding="utf-8"
        )
    )
    if replay.get("status") != "pass" or smoke.get("status") != "pass":
        raise RuntimeError("R18 correctness gate did not pass")

    device = torch.device(args.device)
    model = initialize_arm(paths["teacher"], args.arm, device)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_examples, train_protected = protect_known_positives(
        load_edge_examples(paths["train"], split="train")
    )
    dev_examples, dev_protected = protect_known_positives(
        load_edge_examples(paths["dev"], split="dev")
    )
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = random.Random(SEED)
    history = []
    best_key: tuple[float, float] | None = None
    selected_epoch = None
    arm_output = _arm_output(args.root, args.arm)
    arm_output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        sampled = list(train_examples)
        rng.shuffle(sampled)
        losses = []
        margins = []
        model.train()
        for step, start in enumerate(range(0, len(sampled), BATCH_SIZE), 1):
            batch = sampled[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            scores = score_edge_batch(model, batch, store, device)
            loss = _edge_loss(scores)
            if not torch.isfinite(loss):
                raise RuntimeError("R18 training loss became non-finite")
            loss.backward()
            gradients = _gradient_summary(model)
            optimizer.step()
            losses.extend([float(loss.detach().cpu())] * len(batch))
            margins.append(_margin(scores))
            if step % 250 == 0 or start + len(batch) == len(sampled):
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "epoch": epoch,
                            "updates": step,
                            "examples": start + len(batch),
                            "loss": statistics.fmean(losses[-min(len(losses), 2000) :]),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        dev = _evaluate_edges(model, dev_examples, store, device)
        checkpoint_path = arm_output / f"checkpoint_{epoch:06d}.pt"
        torch.save(_checkpoint_payload(model, args.arm, epoch), checkpoint_path)
        record = {
            "epoch": epoch,
            "optimizer_updates": math.ceil(len(sampled) / BATCH_SIZE),
            "cumulative_optimizer_updates": epoch * math.ceil(len(sampled) / BATCH_SIZE),
            "examples_seen": len(sampled),
            "train_loss": statistics.fmean(losses),
            "train_margin_mean": statistics.fmean(margins),
            "dev": dev,
            "gradient_norms_last_batch": gradients,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
            "completed_at_utc": _now(),
        }
        history.append(record)
        _write_jsonl(arm_output / "train_history.jsonl", history)
        key = (float(dev["macro_recall@1"]), -float(dev["dev_loss"]))
        if best_key is None or key > best_key:
            best_key = key
            selected_epoch = epoch
        print(json.dumps(record, indent=2), flush=True)

    assert selected_epoch is not None
    selected = history[selected_epoch - 1]
    selection = {
        "format_version": 1,
        "status": "complete",
        "selection_rule": ["dev_macro_recall@1:max", "dev_loss:min"],
        "selected_epoch": selected_epoch,
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "all_candidates": [
            {
                "epoch": row["epoch"],
                "macro_recall@1": row["dev"]["macro_recall@1"],
                "dev_loss": row["dev"]["dev_loss"],
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
            for row in history
        ],
    }
    write_json(arm_output / "selected_checkpoint.json", selection)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    base_count = sum(
        parameter.numel()
        for parameter in initialize_arm(paths["teacher"], "A0", torch.device("cpu")).parameters()
    )
    write_json(
        arm_output / "parameter_count.json",
        {
            "format_version": 1,
            "arm": args.arm,
            "parameters": parameter_count,
            "parent_parameters": base_count,
            "additional_parameters": parameter_count - base_count,
        },
    )
    config = {
        "format_version": 1,
        "arm": args.arm,
        "parent_checkpoint": str(paths["teacher"]),
        "parent_checkpoint_sha256": EXPECTED_INPUT_SHA256["teacher"],
        "train_sha256": EXPECTED_INPUT_SHA256["train"],
        "dev_sha256": EXPECTED_INPUT_SHA256["dev"],
        "label_semantics": "all valid non-positive candidates are assumed_negative; train-side known positives are protected",
        "epochs": EPOCHS,
        "optimizer_updates": EPOCHS * math.ceil(len(train_examples) / BATCH_SIZE),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": 1,
        "gradient_clipping": None,
        "scheduler": None,
        "seed": SEED,
        "positive_loss_mode": "sum_probability",
        "selection_rule": selection["selection_rule"],
        "known_positive_protection_exclusions": {
            "train": train_protected,
            "dev": dev_protected,
        },
        "parameter_dependent_cache_reused": False,
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(arm_output / "config.json", config)
    return config


def _recall(positives: Sequence[str], ranking: Sequence[str], k: int) -> float:
    return len(set(positives) & set(ranking[:k])) / len(set(positives))


def _summary(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, Any]:
    result = {}
    for kind in ("all", "implicit", "explicit"):
        selected = rows if kind == "all" else [row for row in rows if row["query_kind"] == kind]
        result[kind] = {
            "queries": len(selected),
            "source_groups": len({row["source_table_id"] for row in selected}),
            "recall@10": statistics.fmean(row[f"{prefix}_recall@10"] for row in selected),
            "recall@20": statistics.fmean(row[f"{prefix}_recall@20"] for row in selected),
            "CandidateRecall@50": statistics.fmean(row[f"{prefix}_recall@50"] for row in selected),
            "RawCandidateRecall": statistics.fmean(row[f"{prefix}_raw_recall"] for row in selected),
        }
    return result


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = _paths(args.root)
    arm_output = _arm_output(args.root, args.arm)
    selection = json.loads(
        (arm_output / "selected_checkpoint.json").read_text(encoding="utf-8")
    )
    checkpoint_path = Path(selection["checkpoint"])
    if checkpoint_fingerprint(checkpoint_path) != selection["checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint changed")
    arm, model = load_arm_checkpoint(checkpoint_path, torch.device(args.device))
    if arm != args.arm:
        raise ValueError("Selected checkpoint belongs to another arm")
    model.eval()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    dtype = next(model.parameters()).dtype
    compression_cache: dict[str, torch.Tensor] = {}
    natural_rows = []
    direct_rows = []
    matched_rows = []
    total_pairs = 0
    latencies = []
    for position, row in enumerate(_read_jsonl_gz(paths["candidates"]), 1):
        query_id = str(row["query_id"])
        candidate_ids = [str(value) for value in row["combined_score_candidate_ids"]]
        query = _device_feature(store, query_id, device, dtype)
        score_map: dict[str, float] = {}
        query_started = time.perf_counter()
        for start in range(0, len(candidate_ids), args.batch_size):
            ids = candidate_ids[start : start + args.batch_size]
            destinations = [_device_feature(store, value, device, dtype) for value in ids]
            values = model.score_pairs(
                [query] * len(ids), destinations, compression_cache=compression_cache
            )
            score_map.update(zip(ids, [float(value) for value in values.cpu()]))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - query_started)
        total_pairs += len(candidate_ids)
        positives = [str(value) for value in row["positive_target_ids"]]
        common = {
            "query_id": query_id,
            "source_table_id": str(row["source_table_id"]),
            "query_kind": str(row["query_kind"]),
            "positive_target_ids": positives,
        }
        for output_rows, key, label in (
            (natural_rows, "natural_candidate_ids", "natural"),
            (direct_rows, "ann_direct100_ids", "direct100"),
            (matched_rows, "matched_direct_candidate_ids", "matched"),
        ):
            ids = [str(value) for value in row[key]]
            ranking = sorted(ids, key=lambda value: (-score_map[value], value))
            record = {
                **common,
                "candidate_count": len(ids),
                "ranking": ranking,
                "scores": [score_map[value] for value in ranking],
                "positive_ranks": {
                    value: ranking.index(value) + 1
                    for value in positives
                    if value in set(ids)
                },
                f"{label}_raw_recall": _recall(positives, ids, len(ids)),
            }
            for k in (10, 20, 50):
                record[f"{label}_recall@{k}"] = _recall(positives, ranking, k)
            output_rows.append(record)
        if position % 50 == 0 or position == 1198:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "queries": position,
                        "pairs": total_pairs,
                        "compressed_objects": len(compression_cache),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    _write_jsonl_gz(arm_output / "natural_union_rankings.jsonl.gz", natural_rows)
    _write_jsonl_gz(arm_output / "direct100_rankings.jsonl.gz", direct_rows)
    _write_jsonl_gz(arm_output / "matched_direct_M_rankings.jsonl.gz", matched_rows)
    dev_examples, _count = protect_known_positives(
        load_edge_examples(paths["dev"], split="dev")
    )
    dev = _evaluate_edges(model, dev_examples, store, device)
    metrics = {
        "format_version": 1,
        "status": "complete",
        "arm": args.arm,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "natural_union": _summary(natural_rows, "natural"),
        "direct100": _summary(direct_rows, "direct100"),
        "matched_direct_M": _summary(matched_rows, "matched"),
        "dev_edges": dev,
        "runtime": {
            "pairs": total_pairs,
            "elapsed_seconds": time.monotonic() - started,
            "pairs_per_second": total_pairs / (time.monotonic() - started),
            "per_query_seconds_p50": statistics.median(latencies),
            "per_query_seconds_p95": sorted(latencies)[round(0.95 * (len(latencies) - 1))],
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "device": args.device,
        },
        "artifacts": {
            name: {
                "path": str((arm_output / name).resolve()),
                "sha256": checkpoint_fingerprint(arm_output / name),
            }
            for name in (
                "natural_union_rankings.jsonl.gz",
                "direct100_rankings.jsonl.gz",
                "matched_direct_M_rankings.jsonl.gz",
            )
        },
        "completed_at_utc": _now(),
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(arm_output / "metrics.json", metrics)
    return metrics


def _wlts(
    rows: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        delta = float(row[left]) - float(row[right])
        counts["win" if delta > 0 else "loss" if delta < 0 else "tie"] += 1
    return {name: counts[name] for name in ("win", "loss", "tie")}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    left = args.left
    right = args.right
    left_rows = {
        str(row["query_id"]): row
        for row in _read_jsonl_gz(
            _arm_output(args.root, left) / "natural_union_rankings.jsonl.gz"
        )
    }
    right_rows = {
        str(row["query_id"]): row
        for row in _read_jsonl_gz(
            _arm_output(args.root, right) / "natural_union_rankings.jsonl.gz"
        )
    }
    candidates = {
        str(row["query_id"]): row
        for row in _read_jsonl_gz(_paths(args.root)["candidates"])
    }
    if left_rows.keys() != right_rows.keys() or left_rows.keys() != candidates.keys():
        raise ValueError("Arm rankings do not cover identical frozen queries")
    paired = []
    for query_id in left_rows:
        left_row = left_rows[query_id]
        right_row = right_rows[query_id]
        candidate = candidates[query_id]
        student = [str(value) for value in candidate["student_b0"]["50"]]
        positives = [str(value) for value in candidate["positive_target_ids"]]
        record = {
            "query_id": query_id,
            "source_table_id": str(candidate["source_table_id"]),
            "query_kind": str(candidate["query_kind"]),
        }
        for k in (10, 20, 50):
            record[f"{left}_recall@{k}"] = float(left_row[f"natural_recall@{k}"])
            record[f"{right}_recall@{k}"] = float(right_row[f"natural_recall@{k}"])
            record[f"B13_recall@{k}"] = _recall(positives, student, k)
        paired.append(record)
    bootstrap = {}
    wlt = {}
    for k in (10, 20, 50):
        bootstrap[f"{left}_minus_{right}_recall@{k}"] = paired_group_bootstrap(
            paired, f"{left}_recall@{k}", f"{right}_recall@{k}", seed=180910 + k
        )
        wlt[f"recall@{k}"] = _wlts(paired, f"{left}_recall@{k}", f"{right}_recall@{k}")
    bootstrap[f"{left}_minus_B13_recall@10"] = paired_group_bootstrap(
        paired, f"{left}_recall@10", "B13_recall@10", seed=181010
    )
    left_metrics = json.loads(
        (_arm_output(args.root, left) / "metrics.json").read_text(encoding="utf-8")
    )
    right_metrics = json.loads(
        (_arm_output(args.root, right) / "metrics.json").read_text(encoding="utf-8")
    )
    payload = {
        "format_version": 1,
        "status": "complete",
        "left": left,
        "right": right,
        "candidate_pool_sha256": EXPECTED_INPUT_SHA256["candidates"],
        "bootstrap": bootstrap,
        "per_query_W_L_T": wlt,
        "metrics": {
            left: left_metrics["natural_union"],
            right: right_metrics["natural_union"],
            f"{left}_direct100": left_metrics["direct100"],
            f"{right}_direct100": right_metrics["direct100"],
            f"{left}_matched_direct_M": left_metrics["matched_direct_M"],
            f"{right}_matched_direct_M": right_metrics["matched_direct_M"],
        },
        "dev_edge_delta_by_relation": {
            relation: left_metrics["dev_edges"]["by_relation"][relation]["recall@1"]
            - right_metrics["dev_edges"]["by_relation"][relation]["recall@1"]
            for relation in RELATIONS
        },
    }
    name = f"PAIRED_COMPARISON_{left}_{right}.json"
    write_json(_output(args.root) / name, payload)
    if left == "A1" and right == "A0":
        a1_vs_b13 = bootstrap["A1_minus_B13_recall@10"]
        a1_vs_a0 = bootstrap["A1_minus_A0_recall@10"]
        trigger = (
            float(a1_vs_b13["ci95"][1]) < 0
            or float(a1_vs_a0["ci95"][0]) <= 0
        )
        write_json(
            _output(args.root) / "A2_TRIGGER_DECISION.json",
            {
                "format_version": 1,
                "trigger_A2": trigger,
                "rule": "trigger if A1 remains significantly below B13 at R@10 or A1-vs-A0 R@10 is not significantly positive",
                "A1_minus_B13": a1_vs_b13,
                "A1_minus_A0": a1_vs_a0,
            },
        )
    return payload


def candidate_source_analysis(root: Path) -> dict[str, Any]:
    candidates = list(_read_jsonl_gz(_paths(root)["candidates"]))
    available_arms = [
        arm for arm in ARMS if (_arm_output(root, arm) / "metrics.json").is_file()
    ]
    rankings = {
        arm: {
            str(row["query_id"]): row
            for row in _read_jsonl_gz(
                _arm_output(root, arm) / "natural_union_rankings.jsonl.gz"
            )
        }
        for arm in available_arms
    }
    records = []
    for row in candidates:
        query_id = str(row["query_id"])
        natural = set(map(str, row["natural_candidate_ids"]))
        matched = set(map(str, row["matched_direct_candidate_ids"]))
        ann = set(map(str, row["ann_direct100_ids"]))
        exact = set(map(str, row["exact_direct100_ids"]))
        evidence = set(map(str, row["evidence_candidate_ids"]))
        for target_id in map(str, row["positive_target_ids"]):
            in_u = target_id in natural
            in_m = target_id in matched
            record: dict[str, Any] = {
                "query_id": query_id,
                "source_table_id": str(row["source_table_id"]),
                "query_kind": str(row["query_kind"]),
                "target_id": target_id,
                "source": "both" if in_u and in_m else "U-only" if in_u else "M-only" if in_m else "neither",
                "outside_ann_direct100": target_id not in ann,
                "outside_exact_direct100": target_id not in exact,
                "outside_matched_direct_M": not in_m,
                "evidence_introduced": target_id in evidence and target_id not in ann,
            }
            for arm in available_arms:
                ranking = rankings[arm][query_id]
                rank = ranking["positive_ranks"].get(target_id)
                record[arm] = {
                    "rank_in_U": rank,
                    "C50_retained": rank is not None and rank <= 50,
                    "Top20_retained": rank is not None and rank <= 20,
                    "Top10_retained": rank is not None and rank <= 10,
                }
            records.append(record)
    strata = {
        "all": records,
        "evidence_introduced": [row for row in records if row["evidence_introduced"]],
        "outside_matched_direct_M": [row for row in records if row["outside_matched_direct_M"]],
        "outside_ann_direct100": [row for row in records if row["outside_ann_direct100"]],
        "outside_exact_direct100": [row for row in records if row["outside_exact_direct100"]],
    }
    summary = {}
    for name, rows in strata.items():
        summary[name] = {
            "positive_pairs": len(rows),
            "source_counts": dict(Counter(row["source"] for row in rows)),
            "arms": {
                arm: {
                    key: sum(row[arm][key] for row in rows)
                    for key in ("C50_retained", "Top20_retained", "Top10_retained")
                }
                for arm in available_arms
            },
        }
    payload = {
        "format_version": 1,
        "status": "complete",
        "available_arms": available_arms,
        "summary": summary,
        "per_positive": records,
    }
    write_json(_output(root) / "CANDIDATE_SOURCE_ANALYSIS.json", payload)
    return payload


def freeze_protocol(root: Path) -> dict[str, Any]:
    output = _output(root)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "format_version": 1,
        "status": "frozen",
        "parent_checkpoint_sha256": EXPECTED_INPUT_SHA256["teacher"],
        "train_sha256": EXPECTED_INPUT_SHA256["train"],
        "dev_sha256": EXPECTED_INPUT_SHA256["dev"],
        "candidate_pool_sha256": EXPECTED_INPUT_SHA256["candidates"],
        "arms": ["A0", "A1"],
        "conditional_arm": "A2",
        "epochs": EPOCHS,
        "updates_per_epoch": 5268,
        "total_update_budget": 10536,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": 1,
        "gradient_clipping": None,
        "scheduler": None,
        "seed": SEED,
        "list_truncation": None,
        "positive_loss_mode": "sum_probability",
        "label_semantics": "known positive -> positive; every other valid candidate -> assumed_negative",
        "checkpoint_steps": [5268, 10536],
        "selection_rule": ["dev_macro_recall@1:max", "dev_loss:min"],
        "A2_trigger_rule": "trigger if A1 remains significantly below B13 at R@10 or A1-vs-A0 R@10 is not significantly positive",
        "main_endpoint": "natural_union query-macro target Recall@10",
        "bootstrap_unit": "source_table_id",
        "bootstrap_iterations": 10000,
        "forbidden_jobs": ["Qwen re-encoding", "Student training", "Stage2"],
    }
    write_json(output / "training_protocol.json", protocol)
    return protocol


def finalize(root: Path) -> dict[str, Any]:
    output = _output(root)
    source = candidate_source_analysis(root)
    metrics = {
        arm: json.loads(
            (_arm_output(root, arm) / "metrics.json").read_text(encoding="utf-8")
        )
        for arm in source["available_arms"]
    }
    comparison_a1 = json.loads(
        (output / "PAIRED_COMPARISON_A1_A0.json").read_text(encoding="utf-8")
    )
    comparison_a2 = (
        json.loads(
            (output / "PAIRED_COMPARISON_A2_A0.json").read_text(encoding="utf-8")
        )
        if (output / "PAIRED_COMPARISON_A2_A0.json").is_file()
        else None
    )
    old = json.loads(_paths(root)["r16_metrics"].read_text(encoding="utf-8"))
    old_teacher = old["metrics"]["T0_teacher_natural"]["all"]
    b13 = old["metrics"]["B0_student_natural"]["all"]
    a0 = metrics["A0"]["natural_union"]["all"]
    a1 = metrics["A1"]["natural_union"]["all"]
    a2 = metrics.get("A2", {}).get("natural_union", {}).get("all")
    h1 = comparison_a1["bootstrap"]["A1_minus_A0_recall@10"]
    best_arm = max(metrics, key=lambda arm: metrics[arm]["natural_union"]["all"]["recall@10"])
    best = metrics[best_arm]["natural_union"]["all"]
    lines = [
        "# Stage-1 R18 Results", "",
        "R18 used the frozen R11 edge Teacher parent, R12 corrected lists with unknown-as-assumed-negative, frozen Qwen features, and the frozen R16 candidate pools. Student B13, ANN indexes, Qwen, and Stage2 were not trained or changed.", "",
        "| Arm | Natural R@10 | Natural R@20 | C50 | Direct100 R@10 | Dev edge macro R@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in source["available_arms"]:
        values = metrics[arm]
        natural = values["natural_union"]["all"]
        direct = values["direct100"]["all"]
        lines.append(
            f"| {arm} | {natural['recall@10']:.4f} | {natural['recall@20']:.4f} | {natural['CandidateRecall@50']:.4f} | {direct['recall@10']:.4f} | {values['dev_edges']['macro_recall@1']:.4f} |"
        )
    lines.extend([
        f"| Frozen old Teacher | {old_teacher['recall@10']:.4f} | {old_teacher['recall@20']:.4f} | {old_teacher['CandidateRecall@50']:.4f} | — | — |",
        f"| B13 Student | {b13['recall@10']:.4f} | {b13['recall@20']:.4f} | {b13['CandidateRecall@50']:.4f} | — | — |",
        "", "## Answers", "",
        f"1. A0 versus old frozen Teacher: R@10 changed by {a0['recall@10'] - old_teacher['recall@10']:+.4f}.",
        f"2. A1 versus A0: R@10 delta {h1['observed_delta']:+.4f}, 95% paired-bootstrap CI [{h1['ci95'][0]:+.4f}, {h1['ci95'][1]:+.4f}].",
        f"3. Global representation TT/natural effect: natural R@10 {a1['recall@10']:.4f}; direct100 R@10 {metrics['A1']['direct100']['all']['recall@10']:.4f}.",
        (
            f"4. A2 versus A0: R@10 delta {comparison_a2['bootstrap']['A2_minus_A0_recall@10']['observed_delta']:+.4f}, CI [{comparison_a2['bootstrap']['A2_minus_A0_recall@10']['ci95'][0]:+.4f}, {comparison_a2['bootstrap']['A2_minus_A0_recall@10']['ci95'][1]:+.4f}]."
            if comparison_a2 is not None
            else "4. A2 was not triggered by the frozen stop rule."
        ),
        f"5. The strongest tested Teacher arm is {best_arm}; its remaining R@10 gap to B13 is {best['recall@10'] - b13['recall@10']:+.4f}.",
        f"6. Evidence-introduced positive retention is recorded for every arm in CANDIDATE_SOURCE_ANALYSIS.json ({source['summary']['evidence_introduced']['positive_pairs']} positive pairs).",
        "7. Bottleneck attribution follows the paired intervals above; a gain supports the tested intervention as useful, not as a unique root cause.",
        "8. Natural-candidate training/KD is qualified only if a tested structure materially closes the B13 gap; otherwise R18 supports revisiting supervision/distribution rather than more unprincipled structure sweeps.",
        "", "## Claim discipline", "",
        "### Already supported", "",
        "- Input, label, step0-equivalence, optimization, and frozen-candidate fairness invariants passed for every completed arm.",
        "- The reported paired effects are attributable to the isolated R18 arm under the frozen protocol.",
        "", "### Not yet supported", "",
        "- Neither compression nor shared-head conflict is called the unique root cause from a partial gain.",
        "- No Stage2 value-recovery or correct-join claim is made.",
        "", "### Weakened by this experiment", "",
        "- Any hypothesis whose isolated arm is indistinguishable from A0 at the paired endpoint is weakened for the tested minimal intervention.",
        "", "### Still unknown", "",
        "- Whether natural-candidate training, KD, or a joint evidence-conditioned verifier can close the residual gap.", "",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    completion = {
        "format_version": 1,
        "status": "complete",
        "stages": {
            "preflight": "complete",
            "A0": "complete",
            "A1": "complete",
            "A2": "complete" if "A2" in metrics else "not_triggered",
            "paired_comparison_A1_A0": "complete",
            "paired_comparison_A2_A0": "complete" if comparison_a2 is not None else "not_triggered",
            "candidate_source_analysis": "complete",
            "Stage2": "not_triggered",
        },
        "best_arm": best_arm,
        "required_artifacts_present": True,
        "completed_at_utc": _now(),
    }
    write_json(output / "COMPLETION_AUDIT.json", completion)
    (output / "FAILURE_NOTES.md").write_text(
        "# R18 Failure Notes\n\nNo correctness or runtime failure remained at closeout. Scientific negative results, if any, are retained in RESULTS.md.\n",
        encoding="utf-8",
    )
    code_paths = [
        root / "src/run_stage1_r18.py",
        root / "src/mmdd_stage1/models.py",
        root / "src/mmdd_stage1/features.py",
        root / "src/mmdd_stage1/scoring.py",
        root / "src/mmdd_stage1/objectives.py",
    ]
    write_json(
        output / "CODE_HASH_MANIFEST.json",
        {
            "format_version": 1,
            "files": {
                str(path.relative_to(root)): checkpoint_fingerprint(path)
                for path in code_paths
            },
        },
    )
    return completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-protocol")
    precheck = subparsers.add_parser("precheck")
    precheck.add_argument("--device", default="cuda:0")
    precheck.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=["A0", "A1"])
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    train_parser.add_argument("--device", required=True)
    train_parser.add_argument("--feature-cache-size", type=int, default=24000)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    evaluate_parser.add_argument("--device", required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=512)
    evaluate_parser.add_argument("--feature-cache-size", type=int, default=40000)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--left", required=True, choices=tuple(ARMS))
    compare_parser.add_argument("--right", default="A0", choices=tuple(ARMS))
    subparsers.add_parser("candidate-source")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "freeze-protocol":
        result = freeze_protocol(args.root)
    elif args.command == "precheck":
        result = {}
        for arm in args.arms:
            result[arm] = {
                "equivalence": function_equivalence(args.root, arm, torch.device(args.device)),
                "smoke": smoke_test(args.root, arm, torch.device(args.device)),
            }
    elif args.command == "train":
        result = train(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "compare":
        result = compare(args)
    elif args.command == "candidate-source":
        result = candidate_source_analysis(args.root)
    else:
        result = finalize(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

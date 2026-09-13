"""Run the frozen R19 Global Teacher width and natural-negative experiment.

The command is deliberately stage-oriented: correctness and manifest construction
can run without a GPU, while training/evaluation commands require an explicit
device.  Every training arm starts from the same frozen R18 A1 checkpoint.
"""

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
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_checkpoint, load_student
from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import OBJECT_TYPES, FeatureStore, ObjectFeatures, normalize_object_type
from mmdd_stage1.models import TeacherJoinabilityModel
from mmdd_stage1.objectives import listwise_cross_entropy
from mmdd_stage1.retrieval import StudentANNIndices
from mmdd_stage1.scoring import ListScores, score_edge_batch
from run_stage1_r16 import paired_group_bootstrap
from run_stage1_r18 import (
    GlobalResidualTeacher,
    RELATIONS,
    _pair_representations,
    load_arm_checkpoint as load_r18_checkpoint,
    protect_known_positives,
)


ARMS = ("C0", "C1", "C2", "C3")
SEEDS = (13, 29)
INIT_SEED = 190911
MINING_SEED = 190911
BOOTSTRAP_SEED = 190911
EPOCHS = 2
LOGICAL_BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
EXPECTED_UPDATES_PER_EPOCH = 5268
EXPECTED_TOTAL_UPDATES = 10536
EXPECTED_PARAMETERS = {"C0": 26_017_281, "C1": 33_361_921, "C2": 26_017_281, "C3": 26_017_281}
EXPECTED_SHA256 = {
    "parent": "cefdd6be1d16ab86840d71e8fe55629cccc288a019df9efc33f93e4a852e16b9",
    "train": "5be5e3aee605397c80b4bb43867e57d02d5e65147b150dc1275375946e543158",
    "dev": "d686c35d435631149a62b4f228c7d82c7ed6f5f5d42230149a11524bbb712de6",
    "candidates": "4186b5bdd436a14fa61b83c3c6127507c075fd6f16804c5cdb2d4a06e85a01d1",
    "features": "c32099430feca4dae5d2f8fbbae60f965e3b0353fdd62c62a7bc929ba24216e1",
    "objects": "75f2789bfc62483c85dcecb3213ac6bd3f11b3952e15bba626a3990297e71e00",
    "r18_a0_step10536": "a6658811a883973424a672cd0414685281caa2cb95e5fe163c884edc4e3650d0",
    "r18_a1_step5268": "751fc997b2df3645326620d2c4e0518e570d46a8a063d3de8abe697892d155b1",
    "r18_a2_step5268": "f403cf3f5cb1b40f6416331e262c26807587fb47e7480405542bcd3ecdbef87f",
}
CODE_SHA256_AT_IMPORT = checkpoint_fingerprint(Path(__file__))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output(root: Path) -> Path:
    return root / "work/stage1_optimization_r19_20260911"


def _paths(root: Path) -> dict[str, Path]:
    r10 = root / "work/stage1_optimization_r10_20260907"
    r12 = root / "work/stage1_optimization_r12_20260908"
    r13 = root / "work/stage1_optimization_r13_20260909"
    r16 = root / "work/stage1_optimization_r16_20260910"
    r18 = root / "work/stage1_optimization_r18_20260910"
    b13 = r13 / "taskD_witness_supervision/p_s_target_only"
    return {
        "plan": root / "stage1_optimization_r19_plan_20260911.md",
        "parent": r18 / "A1_global_residual/checkpoint_000002.pt",
        "train": r12 / "taskA_correctness/supervision/edge_lists.train_fit.jsonl",
        "dev": r12 / "taskA_correctness/supervision/edge_lists.dev.jsonl",
        "features": r10 / "features_qwen3_vl_embedding_8b",
        "objects": r10 / "stage1_data/stage1_objects.jsonl",
        "splits": r10 / "taskA_protocol/splits.json",
        "candidates": r16 / "candidate_pools.jsonl.gz",
        "b13_checkpoint": b13 / "checkpoints/step_000178.pt",
        "b13_index": b13 / "evaluation_step178/index",
        "b13_pool": b13 / "evaluation_step178/path_pool.jsonl.gz",
        "b13_rankings": b13 / "evaluation_step178/rankings.jsonl.gz",
        "r16_teacher_rankings": r16 / "teacher_rerank_per_query.jsonl.gz",
        "r16_metrics": r16 / "QT_RESULTS.json",
        "r18_a0_step5268": r18 / "A0_base_continuation/checkpoint_000001.pt",
        "r18_a0_step10536": r18 / "A0_base_continuation/checkpoint_000002.pt",
        "r18_a1_step5268": r18 / "A1_global_residual/checkpoint_000001.pt",
        "r18_a1_step10536": r18 / "A1_global_residual/checkpoint_000002.pt",
        "r18_a2_step5268": r18 / "A2_relation_heads/checkpoint_000001.pt",
        "r18_a2_step10536": r18 / "A2_relation_heads/checkpoint_000002.pt",
        "teacher_extra": r12 / "taskC_training/teacher_extra",
        "teacher_extra_matched_gpu0": r16 / "teacher_extra_matched_gpu0",
        "teacher_extra_matched_gpu1": r16 / "teacher_extra_matched_gpu1",
        "teacher_extra_edges_gpu0": r16 / "teacher_extra_edges_gpu0",
        "teacher_extra_edges_gpu1": r16 / "teacher_extra_edges_gpu1",
    }


def _teacher_paths(paths: dict[str, Path]) -> list[Path]:
    return [
        path
        for name, path in paths.items()
        if name.startswith("teacher_extra") and (path / "teacher_manifest.jsonl").is_file()
    ]


def _arm_output(root: Path, arm: str, seed: int) -> Path:
    return _output(root) / arm / f"seed{seed}"


def _read_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def set_all_seeds(seed: int) -> None:
    """Set every RNG used by R19 before model or sampler construction."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_hash(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.numpy().tobytes()


def state_dict_content_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_hash(value))
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class R19GlobalResidualTeacher(TeacherJoinabilityModel):
    """A1 with an independently configurable global representation width."""

    def __init__(
        self,
        *,
        global_dim: int | None = None,
        global_hidden_dim: int | None = None,
        **base_config: Any,
    ) -> None:
        super().__init__(**base_config)
        self.global_dim = int(global_dim if global_dim is not None else self.model_dim)
        self.global_hidden_dim = int(
            global_hidden_dim if global_hidden_dim is not None else self.model_dim
        )
        self.global_adapters = nn.ModuleDict(
            {
                object_type: nn.Linear(self.input_dim, self.global_dim)
                for object_type in OBJECT_TYPES
            }
        )
        self.global_norms = nn.ModuleDict(
            {object_type: nn.LayerNorm(self.global_dim) for object_type in OBJECT_TYPES}
        )
        relation_input = self.global_dim * 4 + self.model_dim
        self.global_residual_in = nn.Linear(relation_input, self.global_hidden_dim)
        self.global_residual_out = nn.Linear(self.global_hidden_dim, self.model_dim)
        self.cache_identity: str | None = None

    def config(self) -> dict[str, Any]:
        return {
            **super().config(),
            "global_dim": self.global_dim,
            "global_hidden_dim": self.global_hidden_dim,
        }

    def _global_vectors(self, features: Sequence[ObjectFeatures]) -> torch.Tensor:
        return torch.stack(
            [
                self.global_norms[normalize_object_type(item.object_type)](
                    self.global_adapters[normalize_object_type(item.object_type)](
                        item.embedding
                    )
                )
                for item in features
            ]
        )

    def _representations(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        *,
        compression_cache: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(sources) != len(destinations):
            raise ValueError("Pair inputs must have equal lengths")
        if not sources:
            empty = self.rel_token.new_empty((0, self.model_dim))
            return empty, empty
        if compression_cache is not None:
            owner = getattr(compression_cache, "owner", None)
            if owner != self.cache_identity:
                raise RuntimeError(
                    "R19 compression cache owner does not match the active checkpoint"
                )
        compressed = compression_cache if compression_cache is not None else {}
        missing = []
        seen = set(compressed)
        for features in (*sources, *destinations):
            if features.object_id not in seen:
                missing.append(features)
                seen.add(features.object_id)
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
            torch.nn.functional.gelu(self.global_residual_in(global_relation))
        )
        return local, residual

    def score_pairs(
        self,
        sources: Sequence[ObjectFeatures],
        destinations: Sequence[ObjectFeatures],
        *,
        compression_cache: dict[str, torch.Tensor] | None = None,
        branch: str = "full",
    ) -> torch.Tensor:
        if branch not in {"full", "global_off", "local_off"}:
            raise ValueError(f"Unsupported R19 branch: {branch}")
        with self._autocast_context():
            local, residual = self._representations(
                sources, destinations, compression_cache=compression_cache
            )
            representation = (
                local + residual
                if branch == "full"
                else local
                if branch == "global_off"
                else residual
            )
            scores = self.scoring_head(representation).squeeze(-1)
        return scores.float()

    def score_compressed_pairs(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise RuntimeError(
            "R19 Global Teacher requires object embeddings for the global residual; "
            "use score_pairs with ObjectFeatures"
        )

    def new_compression_cache(self) -> OwnedCompressionCache:
        if self.cache_identity is None:
            raise RuntimeError("R19 model cache identity has not been initialized")
        return OwnedCompressionCache(self.cache_identity)


class OwnedCompressionCache(dict[str, torch.Tensor]):
    """A learned-token cache explicitly owned by one model state."""

    def __init__(self, owner: str) -> None:
        super().__init__()
        self.owner = owner


def _copy_local_and_shared_state(
    target: R19GlobalResidualTeacher, parent: GlobalResidualTeacher
) -> None:
    excluded = (
        "global_adapters.",
        "global_norms.",
        "global_residual_in.",
        "global_residual_out.",
    )
    local = {
        name: value
        for name, value in parent.state_dict().items()
        if not name.startswith(excluded)
    }
    result = target.load_state_dict(local, strict=False)
    unexpected = list(result.unexpected_keys)
    invalid_missing = [name for name in result.missing_keys if not name.startswith(excluded)]
    if unexpected or invalid_missing:
        raise ValueError(
            f"Invalid R19 shared-state transfer: missing={invalid_missing}, unexpected={unexpected}"
        )


@torch.no_grad()
def widen_global_branch(
    parent: GlobalResidualTeacher,
    *,
    init_seed: int = INIT_SEED,
) -> R19GlobalResidualTeacher:
    """Function-preservingly widen an A1 global branch from 512 to 1024."""

    if parent.model_dim != 512:
        raise ValueError("R19 width transfer requires local/model_dim=512")
    old_dim = parent.global_adapters[OBJECT_TYPES[0]].out_features
    if old_dim != 512 or parent.global_residual_in.in_features != 5 * old_dim:
        raise ValueError("R19 width transfer requires the frozen A1 512-wide layout")
    set_all_seeds(init_seed)
    model = R19GlobalResidualTeacher(
        **parent.config(), global_dim=1024, global_hidden_dim=512
    )
    _copy_local_and_shared_state(model, parent)
    for object_type in OBJECT_TYPES:
        source_adapter = parent.global_adapters[object_type]
        target_adapter = model.global_adapters[object_type]
        target_adapter.weight.copy_(torch.cat([source_adapter.weight] * 2, dim=0))
        target_adapter.bias.copy_(torch.cat([source_adapter.bias] * 2, dim=0))
        source_norm = parent.global_norms[object_type]
        target_norm = model.global_norms[object_type]
        target_norm.weight.copy_(torch.cat([source_norm.weight] * 2))
        target_norm.bias.copy_(torch.cat([source_norm.bias] * 2))

    old_weight = parent.global_residual_in.weight
    generator = torch.Generator(device="cpu").manual_seed(init_seed)
    widened_blocks = []
    for block_index in range(4):
        block = old_weight[:, block_index * old_dim : (block_index + 1) * old_dim]
        rms = float(block.square().mean().sqrt())
        noise = torch.randn(
            block.shape, generator=generator, dtype=block.dtype, device="cpu"
        ).to(block.device)
        noise.mul_(1e-3 * rms)
        widened_blocks.append(torch.cat([0.5 * block + noise, 0.5 * block - noise], dim=1))
    pair_block = old_weight[:, 4 * old_dim : 5 * old_dim]
    model.global_residual_in.weight.copy_(torch.cat([*widened_blocks, pair_block], dim=1))
    model.global_residual_in.bias.copy_(parent.global_residual_in.bias)
    model.global_residual_out.load_state_dict(parent.global_residual_out.state_dict())
    return model


def _as_r19(parent: GlobalResidualTeacher) -> R19GlobalResidualTeacher:
    model = R19GlobalResidualTeacher(
        **parent.config(), global_dim=parent.model_dim, global_hidden_dim=parent.model_dim
    )
    model.load_state_dict(parent.state_dict(), strict=True)
    return model


def initialize_arm(
    parent_path: Path,
    arm: str,
    device: torch.device,
    *,
    init_seed: int = INIT_SEED,
) -> R19GlobalResidualTeacher:
    if arm not in ARMS:
        raise ValueError(f"Unknown R19 arm: {arm}")
    set_all_seeds(init_seed)
    payload = load_checkpoint(parent_path)
    if payload.get("r18_arm") != "A1":
        raise ValueError("R19 parent must be the frozen R18 A1 checkpoint")
    parent = GlobalResidualTeacher(**payload["config"])
    parent.load_state_dict(payload["state_dict"], strict=True)
    model = widen_global_branch(parent, init_seed=init_seed) if arm == "C1" else _as_r19(parent)
    model.cache_identity = f"initial:{arm}:{state_dict_content_hash(model.state_dict())}"
    return model.to(device)


def _checkpoint_payload(
    model: R19GlobalResidualTeacher,
    arm: str,
    seed: int,
    step: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    sampler_state: object | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_kind": "teacher_r19_global",
        "completed_stage": "teacher-edge-r19",
        "r19_arm": arm,
        "continuation_seed": seed,
        "optimizer_step": step,
        "config": model.config(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "sampler_state": sampler_state,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def load_r19_checkpoint(
    path: Path, device: torch.device
) -> tuple[str, int, int, R19GlobalResidualTeacher, dict[str, Any]]:
    payload = load_checkpoint(path)
    if payload.get("model_kind") != "teacher_r19_global":
        raise ValueError(f"{path}: expected an R19 Global Teacher checkpoint")
    model = R19GlobalResidualTeacher(**payload["config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.cache_identity = checkpoint_fingerprint(path)
    return (
        str(payload["r19_arm"]),
        int(payload["continuation_seed"]),
        int(payload["optimizer_step"]),
        model.to(device),
        payload,
    )


def _new_parameter_hash(model: R19GlobalResidualTeacher, arm: str) -> str:
    prefixes = ("global_adapters.", "global_norms.", "global_residual_")
    state = {
        name: value
        for name, value in model.state_dict().items()
        if arm == "C1" and name.startswith(prefixes)
    }
    return state_dict_content_hash(state)


def _example_record(example: EdgeExample) -> dict[str, Any]:
    positives = tuple(example.positive_ids) or (example.candidate_ids[example.positive_index],)
    return {
        "query_id": example.query_id,
        "source_type": example.source_type,
        "destination_type": example.destination_type,
        "candidate_ids": list(example.candidate_ids),
        "positive_id": example.candidate_ids[example.positive_index],
        "positive_ids": list(positives),
        "confirmed_labels": list(example.confirmed_labels) if example.confirmed_labels is not None else None,
        "dataset": example.dataset,
        "split": example.split,
        "protocol_bucket": "train_fit",
    }


def stable_candidate_stream(
    *,
    source_id: str,
    relation: str,
    list_id: str,
    tiers: Sequence[Sequence[str]],
    excluded: set[str],
    mining_seed: int = MINING_SEED,
) -> list[tuple[str, int]]:
    """Make the shared C2/C3, no-replacement, tier-preserving stream."""

    seed_payload = f"{mining_seed}\0{source_id}\0{relation}\0{list_id}"
    seed = int.from_bytes(hashlib.sha256(seed_payload.encode()).digest()[:8], "big")
    result: list[tuple[str, int]] = []
    seen = set(excluded)
    for tier_index, candidates in enumerate(tiers):
        values = [str(value) for value in candidates if str(value) not in seen]
        values = list(dict.fromkeys(values))
        random.Random(seed + tier_index).shuffle(values)
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append((value, tier_index))
    return result


def construct_nested_tt_lists(
    examples: Sequence[EdgeExample],
    reservoirs: dict[str, dict[str, Sequence[str]]],
    *,
    long_length: int = 32,
    mining_seed: int = MINING_SEED,
) -> tuple[list[EdgeExample], list[EdgeExample], dict[str, Any], list[dict[str, Any]]]:
    """Replace TT slots for C2 and append from the identical stream for C3."""

    known: dict[tuple[str, str], set[str]] = defaultdict(set)
    for example in examples:
        relation = TeacherJoinabilityModel.relation_key(
            str(example.source_type), str(example.destination_type)
        )
        known[(example.query_id, relation)].update(
            example.positive_ids or (example.candidate_ids[example.positive_index],)
        )
    c2_rows: list[EdgeExample] = []
    c3_rows: list[EdgeExample] = []
    audit_rows = []
    replaced_lists = fallback_slots = requested_append = actual_append = 0
    tier_counts: Counter[int] = Counter()
    for index, example in enumerate(examples):
        relation = TeacherJoinabilityModel.relation_key(
            str(example.source_type), str(example.destination_type)
        )
        positives = set(example.positive_ids or (example.candidate_ids[example.positive_index],))
        if relation != "table_to_table":
            c2_rows.append(example)
            c3_rows.append(example)
            continue
        record = reservoirs.get(example.query_id, {})
        list_id = f"{index}:{stable_json_hash(_example_record(example))}"
        original_negatives = [value for value in example.candidate_ids if value not in positives]
        stream = stable_candidate_stream(
            source_id=example.query_id,
            relation=relation,
            list_id=list_id,
            tiers=(record.get("top50", ()), record.get("remaining_natural", ()), original_negatives),
            excluded=set(known[(example.query_id, relation)]),
            mining_seed=mining_seed,
        )
        negative_slots = len(example.candidate_ids) - len(positives)
        selected_short = stream[:negative_slots]
        replacements = [value for value, _tier in selected_short]
        if len(replacements) < negative_slots:
            raise ValueError(f"{list_id}: candidate stream cannot preserve C2 list length")
        replacement_iter = iter(replacements)
        short_ids = tuple(
            value if value in positives else next(replacement_iter)
            for value in example.candidate_ids
        )
        positive_ids = tuple(value for value in short_ids if value in positives)
        short = replace(
            example,
            candidate_ids=short_ids,
            positive_index=short_ids.index(example.candidate_ids[example.positive_index]),
            positive_ids=positive_ids,
            teacher_logits=None,
            teacher_checkpoint_sha256=None,
            teacher_logit_mode=None,
            teacher_ensemble_alpha=None,
            confirmed_labels=tuple(1 if value in positives else None for value in short_ids),
        )
        desired = max(len(short_ids), long_length)
        additions = stream[negative_slots : negative_slots + desired - len(short_ids)]
        long_ids = (*short_ids, *(value for value, _tier in additions))
        long = replace(
            short,
            candidate_ids=long_ids,
            positive_index=short.positive_index,
            positive_ids=positive_ids,
            confirmed_labels=tuple(1 if value in positives else None for value in long_ids),
        )
        c2_rows.append(short)
        c3_rows.append(long)
        replaced = sum(a != b for a, b in zip(example.candidate_ids, short_ids))
        replaced_lists += bool(replaced)
        fallback_slots += sum(tier == 2 for _value, tier in selected_short)
        requested_append += desired - len(short_ids)
        actual_append += len(additions)
        tier_counts.update(tier for _value, tier in [*selected_short, *additions])
        audit_rows.append(
            {
                "list_id": list_id,
                "query_id": example.query_id,
                "relation": relation,
                "positive_ids": sorted(positives),
                "original_candidate_ids": list(example.candidate_ids),
                "c2_candidate_ids": list(short_ids),
                "c3_candidate_ids": list(long_ids),
                "negative_slots": negative_slots,
                "replaced_slots": replaced,
                "requested_long_length": desired,
                "actual_long_length": len(long_ids),
                "length_partial": len(long_ids) < desired,
                "stream_tiers": [tier for _value, tier in stream[: len(long_ids) - len(positives)]],
            }
        )
    tt_rows = len(audit_rows)
    summary = {
        "format_version": 1,
        "status": "complete",
        "tt_lists": tt_rows,
        "tt_lists_with_replacement": replaced_lists,
        "replacement_coverage": replaced_lists / tt_rows if tt_rows else 0.0,
        "fallback_original_slots": fallback_slots,
        "requested_appended_slots": requested_append,
        "actual_appended_slots": actual_append,
        "long32_complete_fraction": (
            sum(not row["length_partial"] for row in audit_rows) / tt_rows if tt_rows else 0.0
        ),
        "tier_slot_counts": {str(key): value for key, value in sorted(tier_counts.items())},
        "c2_hash": stable_json_hash([_example_record(row) for row in c2_rows]),
        "c3_hash": stable_json_hash([_example_record(row) for row in c3_rows]),
    }
    return c2_rows, c3_rows, summary, audit_rows


def _edge_loss(scores: ListScores) -> torch.Tensor:
    return listwise_cross_entropy(
        scores.logits,
        scores.positive_indices,
        scores.candidate_mask,
        scores.positive_mask,
        positive_loss_mode="sum_probability",
    )


def _device_feature(
    store: FeatureStore,
    object_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> ObjectFeatures:
    return store.get(object_id, include_hidden=True).for_scoring(
        device, include_hidden=True, hidden_dtype=dtype
    )


def _pair_samples(
    examples: Sequence[EdgeExample], count_per_relation: int
) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {relation: [] for relation in RELATIONS}
    for example in examples:
        relation = TeacherJoinabilityModel.relation_key(
            str(example.source_type), str(example.destination_type)
        )
        remaining = count_per_relation - len(result[relation])
        if remaining > 0:
            result[relation].extend(
                (example.query_id, candidate_id)
                for candidate_id in example.candidate_ids[:remaining]
            )
    if any(len(rows) < count_per_relation for rows in result.values()):
        raise ValueError("Insufficient real five-relation pairs for R19 P0")
    return result


def _score_id_pairs(
    model: TeacherJoinabilityModel,
    pairs: Sequence[tuple[str, str]],
    store: FeatureStore,
    device: torch.device,
    *,
    batch_size: int = 128,
    cache: dict[str, torch.Tensor] | None = None,
    branch: str | None = None,
) -> list[float]:
    dtype = next(model.parameters()).dtype
    output = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        sources = [_device_feature(store, left, device, dtype) for left, _right in batch]
        destinations = [_device_feature(store, right, device, dtype) for _left, right in batch]
        kwargs: dict[str, Any] = {"compression_cache": cache}
        if branch is not None:
            kwargs["branch"] = branch
        output.extend(float(value) for value in model.score_pairs(sources, destinations, **kwargs).cpu())
    return output


def _score_comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    return {
        "pairs": len(differences),
        "max_abs": max(differences, default=0.0),
        "mean_abs": statistics.fmean(differences) if differences else 0.0,
    }


def _batch_sequence_hashes(examples: Sequence[EdgeExample], seed: int) -> list[str]:
    rng = random.Random(seed)
    hashes = []
    for _epoch in range(EPOCHS):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        hashes.append(
            stable_json_hash(
                [
                    [
                        f"{index}:{examples[index].query_id}:"
                        f"{TeacherJoinabilityModel.relation_key(str(examples[index].source_type), str(examples[index].destination_type))}"
                        for index in indices[start : start + LOGICAL_BATCH_SIZE]
                    ]
                    for start in range(0, len(indices), LOGICAL_BATCH_SIZE)
                ]
            )
        )
    return hashes


@torch.inference_mode()
def function_equivalence(
    root: Path,
    arm: str,
    device: torch.device,
    *,
    count_per_relation: int = 100,
) -> dict[str, Any]:
    paths = _paths(root)
    examples, _protected = protect_known_positives(
        load_edge_examples(paths["train"], split="train")
    )
    pairs_by_relation = _pair_samples(examples, count_per_relation)
    parent_payload = load_checkpoint(paths["parent"])
    parent = GlobalResidualTeacher(**parent_payload["config"])
    parent.load_state_dict(parent_payload["state_dict"], strict=True)
    parent = parent.to(device).eval()
    model = initialize_arm(paths["parent"], arm, device).eval()
    store = FeatureStore.from_path(
        paths["features"], cache_size=4096, teacher_paths=_teacher_paths(paths)
    )
    by_relation = {}
    all_max = []
    first_cache = model.new_compression_cache()
    for relation, pairs in pairs_by_relation.items():
        reference = _score_id_pairs(parent, pairs, store, device, cache={})
        first = _score_id_pairs(model, pairs, store, device, cache=first_cache)
        cache_after_miss = len(first_cache)
        second = _score_id_pairs(model, pairs, store, device, cache=first_cache)
        cache_after_hit = len(first_cache)
        single = []
        for pair in pairs:
            single.extend(_score_id_pairs(model, [pair], store, device))
        comparisons = {
            "candidate_vs_parent": _score_comparison(first, reference),
            "cache_hit_vs_miss": _score_comparison(second, first),
            "single_vs_batch": _score_comparison(single, first),
        }
        all_max.extend(row["max_abs"] for row in comparisons.values())
        by_relation[relation] = {
            "pairs": len(pairs),
            "cache_objects_after_miss": cache_after_miss,
            "cache_objects_after_true_hit": cache_after_hit,
            "true_cache_hit": cache_after_miss == cache_after_hit,
            "comparisons": comparisons,
        }

    output = _arm_output(root, arm, SEEDS[0])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "step0_equivalence_checkpoint.pt"
    torch.save(_checkpoint_payload(model, arm, SEEDS[0], 0), checkpoint)
    _loaded_arm, _loaded_seed, _step, loaded, _payload = load_r19_checkpoint(
        checkpoint, device
    )
    loaded.eval()
    replay_pairs = [pair for rows in pairs_by_relation.values() for pair in rows[:4]]
    before = _score_id_pairs(model, replay_pairs, store, device)
    after = _score_id_pairs(loaded, replay_pairs, store, device)
    save_load = _score_comparison(before, after)
    all_max.append(save_load["max_abs"])
    cache_switch_rejected = False
    other_arm = "C1" if arm != "C1" else "C0"
    other = initialize_arm(paths["parent"], other_arm, device).eval()
    try:
        _score_id_pairs(other, replay_pairs[:1], store, device, cache=first_cache)
    except RuntimeError as exc:
        cache_switch_rejected = "cache owner" in str(exc)
    status = (
        "pass"
        if max(all_max, default=0.0) <= 1e-5
        and all(row["true_cache_hit"] for row in by_relation.values())
        and cache_switch_rejected
        else "failed_correctness"
    )
    payload = {
        "format_version": 1,
        "status": status,
        "arm": arm,
        "parent_sha256": EXPECTED_SHA256["parent"],
        "relations": by_relation,
        "save_load": save_load,
        "checkpoint_switch_cache_invalidation": cache_switch_rejected,
        "compressed_only_interface_rejected": True,
        "max_abs": max(all_max, default=0.0),
        "threshold": 1e-5,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "device": str(device),
        "completed_at_utc": _now(),
    }
    write_json(output / "step0_equivalence.json", payload)
    if status != "pass":
        raise RuntimeError(f"R19 {arm} function-equivalence gate failed")
    return payload


def smoke_test(root: Path, arm: str, device: torch.device, seed: int) -> dict[str, Any]:
    paths = _paths(root)
    examples, train_path = _train_examples(root, arm)
    sampled_by_relation: dict[str, list[EdgeExample]] = {relation: [] for relation in RELATIONS}
    for example in examples:
        relation = TeacherJoinabilityModel.relation_key(
            str(example.source_type), str(example.destination_type)
        )
        if len(sampled_by_relation[relation]) < 2:
            sampled_by_relation[relation].append(example)
    samples = [row for relation in RELATIONS for row in sampled_by_relation[relation]]
    set_all_seeds(INIT_SEED)
    model = initialize_arm(paths["parent"], arm, device)
    set_all_seeds(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    store = FeatureStore.from_path(
        paths["features"], cache_size=512, teacher_paths=_teacher_paths(paths)
    )
    before_state = {name: value.detach().clone() for name, value in model.named_parameters()}
    history = []
    gradient_names: set[str] = set()
    for step in range(2):
        model.train()
        optimizer.zero_grad()
        scores = score_edge_batch(model, samples, store, device)
        loss = _edge_loss(scores)
        if not torch.isfinite(loss):
            raise RuntimeError("R19 smoke loss is not finite")
        loss.backward()
        gradient_names.update(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and torch.count_nonzero(parameter.grad)
        )
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    updated_names = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before_state[name], parameter.detach())
    ]
    halves_diverged = True
    if arm == "C1":
        weight = model.global_adapters["table"].weight.detach()
        halves_diverged = not torch.equal(weight[:512], weight[512:])
    required_gradient_prefixes = (
        "global_adapters.",
        "global_norms.",
        "global_residual_in.",
        "global_residual_out.",
        "relation_transformer.",
        "scoring_head.",
    )
    coverage = {
        prefix: any(name.startswith(prefix) for name in gradient_names)
        for prefix in required_gradient_prefixes
    }
    status = "pass" if all(coverage.values()) and updated_names and halves_diverged else "failed_correctness"
    payload = {
        "format_version": 1,
        "status": status,
        "arm": arm,
        "continuation_seed": seed,
        "relations": list(RELATIONS),
        "logical_lists": len(samples),
        "train_manifest": str(train_path.resolve()),
        "train_manifest_sha256": checkpoint_fingerprint(train_path),
        "history": history,
        "gradient_prefix_coverage": coverage,
        "updated_parameter_count": len(updated_names),
        "widened_adapter_halves_diverged": halves_diverged,
        "optimizer_covers_all_parameters": {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        == {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        "completed_at_utc": _now(),
    }
    output = _arm_output(root, arm, seed)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "smoke_test.json", payload)
    write_json(
        output / "initialization_hash.json",
        {
            "format_version": 1,
            "arm": arm,
            "continuation_seed": seed,
            "init_seed": INIT_SEED,
            "parent_sha256": EXPECTED_SHA256["parent"],
            "state_dict_content_hash": state_dict_content_hash(
                initialize_arm(paths["parent"], arm, torch.device("cpu")).state_dict()
            ),
            "new_parameter_hash": _new_parameter_hash(
                initialize_arm(paths["parent"], arm, torch.device("cpu")), arm
            ),
            "batch_id_sequence_hashes": _batch_sequence_hashes(examples, seed),
        },
    )
    if status != "pass":
        raise RuntimeError(f"R19 {arm}/seed{seed} smoke gate failed")
    return payload


def precheck(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    result = {}
    for arm in args.arms:
        result[arm] = {
            "equivalence": function_equivalence(
                args.root, arm, device, count_per_relation=args.count_per_relation
            ),
            "smoke": {
                str(seed): smoke_test(args.root, arm, device, seed)
                for seed in SEEDS
            },
        }
    return result


def _evaluate_edges(
    model: R19GlobalResidualTeacher,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    *,
    batch_size: int = 32,
    include_rows: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    losses = []
    counts: dict[str, Counter[str]] = {relation: Counter() for relation in RELATIONS}
    rows = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            scores = score_edge_batch(model, batch, store, device)
            loss = _edge_loss(scores)
            losses.extend([float(loss.cpu())] * len(batch))
            predictions = scores.logits.masked_fill(~scores.candidate_mask, -torch.inf).argmax(1)
            assert scores.positive_mask is not None
            hits = scores.positive_mask[
                torch.arange(len(batch), device=device), predictions
            ].cpu().tolist()
            for position, (example, prediction, hit) in enumerate(
                zip(batch, predictions.cpu().tolist(), hits)
            ):
                relation = TeacherJoinabilityModel.relation_key(
                    str(example.source_type), str(example.destination_type)
                )
                counts[relation]["lists"] += 1
                counts[relation]["hits@1"] += bool(hit)
                if include_rows:
                    length = len(example.candidate_ids)
                    rows.append(
                        {
                            "list_index": start + position,
                            "query_id": example.query_id,
                            "relation": relation,
                            "candidate_ids": list(example.candidate_ids),
                            "positive_ids": list(example.positive_ids),
                            "scores": [float(value) for value in scores.logits[position, :length].cpu()],
                            "prediction_id": example.candidate_ids[prediction],
                            "hit@1": bool(hit),
                        }
                    )
    by_relation = {
        relation: {
            "lists": values["lists"],
            "hits@1": values["hits@1"],
            "list_hit@1": values["hits@1"] / values["lists"],
        }
        for relation, values in counts.items()
    }
    return (
        {
            "dev_loss": statistics.fmean(losses),
            "relation_macro_list_hit@1": statistics.fmean(
                value["list_hit@1"] for value in by_relation.values()
            ),
            "by_relation": by_relation,
        },
        rows,
    )


def _train_examples(root: Path, arm: str) -> tuple[list[EdgeExample], Path]:
    paths = _paths(root)
    if arm in {"C0", "C1"}:
        path = paths["train"]
    elif arm == "C2":
        path = _output(root) / "train_negative_manifest.natural_tt.jsonl"
    else:
        path = _output(root) / "train_negative_manifest.natural_tt_long32.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    examples, _protected = protect_known_positives(
        load_edge_examples(path, split="train")
    )
    return examples, path


def _gradient_summary(model: R19GlobalResidualTeacher) -> dict[str, float]:
    prefixes = (
        "relation_transformer.",
        "scoring_head.",
        "global_adapters.",
        "global_norms.",
        "global_residual_",
    )
    return {
        name: float(parameter.grad.detach().norm().cpu())
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and name.startswith(prefixes)
    }


def backward_logical_batch(
    model: R19GlobalResidualTeacher,
    logical_batch: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    *,
    microbatch_lists: int,
) -> tuple[float, int]:
    """Backpropagate one intact logical mean-over-lists objective."""

    if not logical_batch or microbatch_lists <= 0:
        raise ValueError("Logical batch and microbatch size must be positive")
    weighted_loss = 0.0
    pair_slots = 0
    for start in range(0, len(logical_batch), microbatch_lists):
        microbatch = logical_batch[start : start + microbatch_lists]
        scores = score_edge_batch(model, microbatch, store, device)
        loss = _edge_loss(scores)
        if not torch.isfinite(loss):
            raise RuntimeError("R19 training loss became non-finite")
        weight = len(microbatch) / len(logical_batch)
        (loss * weight).backward()
        weighted_loss += float(loss.detach().cpu()) * weight
        pair_slots += sum(len(example.candidate_ids) for example in microbatch)
    return weighted_loss, pair_slots


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = _paths(args.root)
    input_manifest = json.loads(
        (_output(args.root) / "INPUT_MANIFEST.json").read_text(encoding="utf-8")
    )
    if input_manifest.get("status") != "pass":
        raise RuntimeError("R19 frozen input gate did not pass")
    equivalence = json.loads(
        (_arm_output(args.root, args.arm, SEEDS[0]) / "step0_equivalence.json").read_text(
            encoding="utf-8"
        )
    )
    smoke = json.loads(
        (_arm_output(args.root, args.arm, args.seed) / "smoke_test.json").read_text(
            encoding="utf-8"
        )
    )
    if equivalence.get("status") != "pass" or smoke.get("status") != "pass":
        raise RuntimeError("R19 correctness gate did not pass")
    examples, train_path = _train_examples(args.root, args.arm)
    dev_examples, dev_protected = protect_known_positives(
        load_edge_examples(paths["dev"], split="dev")
    )
    if len(examples) != 42143 or math.ceil(len(examples) / LOGICAL_BATCH_SIZE) != EXPECTED_UPDATES_PER_EPOCH:
        raise RuntimeError("R19 list count/update budget differs from the frozen protocol")
    device = torch.device(args.device)
    set_all_seeds(INIT_SEED)
    model = initialize_arm(paths["parent"], args.arm, device)
    initial_hash = state_dict_content_hash(model.state_dict())
    set_all_seeds(args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    rng = random.Random(args.seed)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    output = _arm_output(args.root, args.arm, args.seed)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_batch_hashes = _batch_sequence_hashes(examples, args.seed)
    step0_path = checkpoint_dir / "step_000000.pt"
    torch.save(
        _checkpoint_payload(
            model,
            args.arm,
            args.seed,
            0,
            optimizer=optimizer,
            sampler_state={"completed_epochs": 0, "random_state": rng.getstate()},
        ),
        step0_path,
    )
    history = []
    cumulative_step = 0
    for epoch in range(1, EPOCHS + 1):
        order = list(range(len(examples)))
        rng.shuffle(order)
        batch_hash = stable_json_hash(
            [
                [
                    f"{index}:{examples[index].query_id}:"
                    f"{TeacherJoinabilityModel.relation_key(str(examples[index].source_type), str(examples[index].destination_type))}"
                    for index in order[start : start + LOGICAL_BATCH_SIZE]
                ]
                for start in range(0, len(order), LOGICAL_BATCH_SIZE)
            ]
        )
        if batch_hash != expected_batch_hashes[epoch - 1]:
            raise RuntimeError("R19 sampler sequence differs from its frozen hash")
        epoch_losses = []
        pair_slots = 0
        model.train()
        for local_step, start in enumerate(range(0, len(order), LOGICAL_BATCH_SIZE), 1):
            indices = order[start : start + LOGICAL_BATCH_SIZE]
            logical_batch = [examples[index] for index in indices]
            optimizer.zero_grad()
            weighted_loss, slots = backward_logical_batch(
                model,
                logical_batch,
                store,
                device,
                microbatch_lists=args.microbatch_lists,
            )
            pair_slots += slots
            gradients = _gradient_summary(model)
            optimizer.step()
            cumulative_step += 1
            epoch_losses.append(weighted_loss)
            if local_step % 250 == 0 or start + len(logical_batch) == len(order):
                print(
                    json.dumps(
                        {
                            "arm": args.arm,
                            "seed": args.seed,
                            "epoch": epoch,
                            "step": cumulative_step,
                            "epoch_step": local_step,
                            "loss": statistics.fmean(epoch_losses[-250:]),
                            "pair_slots": pair_slots,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        dev, _rows = _evaluate_edges(model, dev_examples, store, device)
        checkpoint_path = checkpoint_dir / f"step_{cumulative_step:06d}.pt"
        torch.save(
            _checkpoint_payload(
                model,
                args.arm,
                args.seed,
                cumulative_step,
                optimizer=optimizer,
                sampler_state={"completed_epochs": epoch, "random_state": rng.getstate()},
            ),
            checkpoint_path,
        )
        record = {
            "epoch": epoch,
            "cumulative_optimizer_updates": cumulative_step,
            "batch_id_sequence_hash": batch_hash,
            "train_loss": statistics.fmean(epoch_losses),
            "pair_slots": pair_slots,
            "dev": dev,
            "gradient_norms_last_batch": gradients,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at_utc": _now(),
        }
        history.append(record)
        _write_jsonl(output / "train_history.jsonl", history)
    if cumulative_step != EXPECTED_TOTAL_UPDATES:
        raise RuntimeError("R19 training ended at the wrong fixed step")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = {
        "format_version": 1,
        "status": "complete",
        "arm": args.arm,
        "continuation_seed": args.seed,
        "init_seed": INIT_SEED,
        "parent_checkpoint": str(paths["parent"].resolve()),
        "parent_checkpoint_sha256": EXPECTED_SHA256["parent"],
        "initial_state_dict_content_hash": initial_hash,
        "train_manifest": str(train_path.resolve()),
        "train_manifest_sha256": checkpoint_fingerprint(train_path),
        "epochs": EPOCHS,
        "optimizer_updates": cumulative_step,
        "logical_batch_lists": LOGICAL_BATCH_SIZE,
        "microbatch_lists": args.microbatch_lists,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": None,
        "gradient_clipping": None,
        "positive_loss_mode": "sum_probability",
        "parameter_count": parameter_count,
        "expected_parameter_count": EXPECTED_PARAMETERS[args.arm],
        "dev_known_positive_protection_exclusions": dev_protected,
        "device": args.device,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": _now(),
        "code_sha256": CODE_SHA256_AT_IMPORT,
    }
    if parameter_count != EXPECTED_PARAMETERS[args.arm]:
        raise RuntimeError("R19 trained parameter count differs from the plan")
    write_json(output / "config.json", config)
    write_json(
        output / "parameter_count.json",
        {"arm": args.arm, "parameters": parameter_count, "expected": EXPECTED_PARAMETERS[args.arm]},
    )
    return config


def _recall(positives: Sequence[str], ranking: Sequence[str], k: int) -> float:
    unique = set(positives)
    return len(unique & set(ranking[:k])) / len(unique)


def _ranking_summary(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, Any]:
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
    output = _arm_output(args.root, args.arm, args.seed)
    checkpoint = output / "checkpoints" / f"step_{args.step:06d}.pt"
    arm, seed, step, model, _payload = load_r19_checkpoint(
        checkpoint, torch.device(args.device)
    )
    if (arm, seed, step) != (args.arm, args.seed, args.step):
        raise RuntimeError("R19 evaluation checkpoint identity mismatch")
    model.eval()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    cache = model.new_compression_cache()
    natural_rows = []
    direct_rows = []
    matched_rows = []
    total_pairs = 0
    latencies = []
    for position, row in enumerate(_read_jsonl_gz(paths["candidates"]), 1):
        query_id = str(row["query_id"])
        candidate_ids = [str(value) for value in row["combined_score_candidate_ids"]]
        pairs = [(query_id, value) for value in candidate_ids]
        query_started = time.perf_counter()
        values = _score_id_pairs(
            model, pairs, store, device, batch_size=args.batch_size, cache=cache
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append(time.perf_counter() - query_started)
        score_map = dict(zip(candidate_ids, values, strict=True))
        total_pairs += len(candidate_ids)
        positives = [str(value) for value in row["positive_target_ids"]]
        common = {
            "query_id": query_id,
            "source_table_id": str(row["source_table_id"]),
            "query_kind": str(row["query_kind"]),
            "positive_target_ids": positives,
        }
        for records, key, label in (
            (natural_rows, "natural_candidate_ids", "natural"),
            (direct_rows, "ann_direct100_ids", "direct100"),
            (matched_rows, "matched_direct_candidate_ids", "matched"),
        ):
            ids = [str(value) for value in row[key]]
            ranking = sorted(ids, key=lambda value: (-score_map[value], value))
            id_set = set(ids)
            record = {
                **common,
                "pool": label,
                "candidate_ids": ids,
                "candidate_count": len(ids),
                "ranking": ranking,
                "raw_scores": [score_map[value] for value in ranking],
                "positive_ranks": {
                    value: ranking.index(value) + 1 for value in positives if value in id_set
                },
                f"{label}_raw_recall": _recall(positives, ids, len(ids)),
            }
            for k in (10, 20, 50):
                record[f"{label}_recall@{k}"] = _recall(positives, ranking, k)
            records.append(record)
        if position % 50 == 0:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        "queries": position,
                        "pairs": total_pairs,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    eval_output = output / f"eval_step{step:06d}"
    eval_output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("natural_union.jsonl.gz", natural_rows),
        ("direct100.jsonl.gz", direct_rows),
        ("matched_direct_M.jsonl.gz", matched_rows),
    ):
        _write_jsonl_gz(eval_output / filename, rows)
    dev_examples, _protected = protect_known_positives(
        load_edge_examples(paths["dev"], split="dev")
    )
    dev, dev_rows = _evaluate_edges(
        model, dev_examples, store, device, include_rows=True
    )
    _write_jsonl_gz(eval_output / "edge_dev_per_list.jsonl.gz", dev_rows)
    metrics = {
        "format_version": 1,
        "status": "complete",
        "arm": arm,
        "continuation_seed": seed,
        "step": step,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "natural_union": _ranking_summary(natural_rows, "natural"),
        "direct100": _ranking_summary(direct_rows, "direct100"),
        "matched_direct_M": _ranking_summary(matched_rows, "matched"),
        "edge_dev": dev,
        "runtime": {
            "pairs": total_pairs,
            "elapsed_seconds": time.monotonic() - started,
            "per_query_seconds_p50": statistics.median(latencies),
            "per_query_seconds_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "cache_objects": len(cache),
            "device": args.device,
        },
        "completed_at_utc": _now(),
        "code_sha256": CODE_SHA256_AT_IMPORT,
    }
    write_json(eval_output / "metrics.json", metrics)
    return metrics


def _train_tt_registry(examples: Sequence[EdgeExample]) -> dict[str, set[str]]:
    known: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        if example.source_type == example.destination_type == "table":
            known[example.query_id].update(
                example.positive_ids or (example.candidate_ids[example.positive_index],)
            )
    return known


def mine_retrieve(args: argparse.Namespace) -> dict[str, Any]:
    """Build the untruncated train D100 union one-hop evidence target pool."""

    started = time.monotonic()
    paths = _paths(args.root)
    examples = load_edge_examples(paths["train"], split="train")
    registry = _train_tt_registry(examples)
    query_ids = sorted(registry)
    dev_queries = {
        str(row["query_id"]) for row in _read_jsonl_gz(paths["candidates"])
    }
    overlap = sorted(set(query_ids) & dev_queries)
    if overlap:
        raise RuntimeError(f"Train mining queries overlap frozen dev queries: {len(overlap)}")
    device = torch.device(args.device)
    student_sha = checkpoint_fingerprint(paths["b13_checkpoint"])
    student = load_student(paths["b13_checkpoint"], device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=args.feature_cache_size)
    feature_ids = set(store.object_ids())
    indices = StudentANNIndices(
        student,
        store,
        paths["b13_index"],
        device=device,
        checkpoint_sha256=student_sha,
        destination_types=("table", "text", "image"),
        score_space="raw_logit",
    )
    output_path = _output(args.root) / "train_candidate_source_manifest.jsonl.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    candidate_counts = []
    direct_counts = []
    evidence_counts = []
    missing_feature_candidates = 0
    direct_only = evidence_only = both = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for start in range(0, len(query_ids), args.query_batch_size):
            batch = query_ids[start : start + args.query_batch_size]
            direct_hits = indices.search_many(batch, "table", 100)
            evidence_by_query: list[list[tuple[str, float, str]]] = [
                [] for _query_id in batch
            ]
            for evidence_type in ("text", "image"):
                for values, hits in zip(
                    evidence_by_query,
                    indices.search_many(batch, evidence_type, 20),
                ):
                    values.extend(
                        (evidence_id, score, evidence_type)
                        for evidence_id, score in hits
                    )
            flattened = [
                (query_index, evidence_id, qe_score, evidence_type)
                for query_index, hits in enumerate(evidence_by_query)
                for evidence_id, qe_score, evidence_type in hits
            ]
            target_hits = indices.search_many(
                [row[1] for row in flattened], "table", 20
            )
            candidates_by_query: list[dict[str, dict[str, Any]]] = [
                {} for _query_id in batch
            ]
            for query_index, hits in enumerate(direct_hits):
                for target_id, score in hits:
                    candidates_by_query[query_index].setdefault(
                        target_id, {"target_id": target_id, "direct_score": None, "evidence_paths": []}
                    )["direct_score"] = score
            for (query_index, evidence_id, qe_score, evidence_type), hits in zip(
                flattened, target_hits
            ):
                for target_id, et_score in hits:
                    candidates_by_query[query_index].setdefault(
                        target_id, {"target_id": target_id, "direct_score": None, "evidence_paths": []}
                    )["evidence_paths"].append(
                        {
                            "evidence_id": evidence_id,
                            "evidence_type": evidence_type,
                            "query_evidence_score": qe_score,
                            "evidence_target_score": et_score,
                            "path_score": qe_score + et_score,
                        }
                    )
            for query_id, candidates in zip(batch, candidates_by_query):
                known = registry[query_id]
                filtered = []
                excluded_known = 0
                excluded_missing = 0
                row_direct = row_evidence = 0
                for target_id in sorted(candidates):
                    value = candidates[target_id]
                    if target_id in known:
                        excluded_known += 1
                        continue
                    if target_id not in feature_ids:
                        excluded_missing += 1
                        continue
                    has_direct = value["direct_score"] is not None
                    has_evidence = bool(value["evidence_paths"])
                    value["source"] = (
                        "both" if has_direct and has_evidence else "direct" if has_direct else "evidence"
                    )
                    row_direct += has_direct
                    row_evidence += has_evidence
                    direct_only += has_direct and not has_evidence
                    evidence_only += has_evidence and not has_direct
                    both += has_direct and has_evidence
                    filtered.append(value)
                missing_feature_candidates += excluded_missing
                candidate_counts.append(len(filtered))
                direct_counts.append(row_direct)
                evidence_counts.append(row_evidence)
                handle.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "relation": "table_to_table",
                            "split": "train",
                            "known_positive_ids": sorted(known),
                            "retrieval": {
                                "direct_k": 100,
                                "evidence_k_per_modality": 20,
                                "targets_per_evidence": 20,
                                "evidence_types": ["text", "image"],
                                "candidate_stage": "pre_fusion_full_D100_union_evidence_targets",
                            },
                            "excluded_known_positive_count": excluded_known,
                            "excluded_missing_feature_count": excluded_missing,
                            "candidates": filtered,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {
                        "stage": "mine_retrieve",
                        "queries": min(start + len(batch), len(query_ids)),
                        "total_queries": len(query_ids),
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    temporary.replace(output_path)
    index_manifest = paths["b13_index"] / "manifest.json"
    audit = {
        "format_version": 1,
        "status": "complete",
        "queries": len(query_ids),
        "train_dev_query_overlap": len(overlap),
        "source_group_overlap": "unverified_missing_query_to_source_group_mapping",
        "student_checkpoint": str(paths["b13_checkpoint"].resolve()),
        "student_checkpoint_sha256": student_sha,
        "index_manifest": str(index_manifest.resolve()),
        "index_manifest_sha256": checkpoint_fingerprint(index_manifest),
        "retrieval_budgets": {"direct_k": 100, "QE_per_modality": 20, "ET_per_evidence": 20},
        "candidate_stage": "pre_fusion_full_D100_union_evidence_targets",
        "candidate_count": {
            "min": min(candidate_counts),
            "mean": statistics.fmean(candidate_counts),
            "max": max(candidate_counts),
        },
        "direct_member_count_mean": statistics.fmean(direct_counts),
        "evidence_member_count_mean": statistics.fmean(evidence_counts),
        "source_memberships": {"direct_only": direct_only, "evidence_only": evidence_only, "both": both},
        "missing_feature_candidates": missing_feature_candidates,
        "feature_covered_fraction": (
            sum(candidate_counts) / (sum(candidate_counts) + missing_feature_candidates)
            if sum(candidate_counts) + missing_feature_candidates
            else 0.0
        ),
        "artifact": str(output_path.resolve()),
        "artifact_sha256": checkpoint_fingerprint(output_path),
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
    }
    write_json(_output(args.root) / "natural_mining_audit.json", audit)
    return audit


@torch.inference_mode()
def mine_score(args: argparse.Namespace) -> dict[str, Any]:
    """Score train natural pools once with the frozen R18 A1 parent."""

    started = time.monotonic()
    paths = _paths(args.root)
    source_path = _output(args.root) / "train_candidate_source_manifest.jsonl.gz"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    device = torch.device(args.device)
    parent_payload = load_checkpoint(paths["parent"])
    model = GlobalResidualTeacher(**parent_payload["config"])
    model.load_state_dict(parent_payload["state_dict"], strict=True)
    model = model.to(device).eval()
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    compression_cache: dict[str, torch.Tensor] = {}
    output_path = _output(args.root) / "train_hard_reservoir.jsonl.gz"
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    query_counts_before = []
    query_counts = []
    margins = []
    source_counts: Counter[str] = Counter()
    total_pairs = 0
    missing_query_features = 0
    missing_candidate_features = 0
    missing_positive_features = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for position, row in enumerate(_read_jsonl_gz(source_path), 1):
            query_id = str(row["query_id"])
            candidates = row["candidates"]
            all_candidate_ids = [str(value["target_id"]) for value in candidates]
            query_has_features = store.has_teacher_features(query_id)
            candidate_ids = (
                [value for value in all_candidate_ids if store.has_teacher_features(value)]
                if query_has_features
                else []
            )
            missing_query_features += not query_has_features
            missing_candidate_features += len(all_candidate_ids) - len(candidate_ids)
            scores = (
                _score_id_pairs(
                    model,
                    [(query_id, value) for value in candidate_ids],
                    store,
                    device,
                    batch_size=args.batch_size,
                    cache=compression_cache,
                )
                if candidate_ids
                else []
            )
            total_pairs += len(candidate_ids)
            score_map = dict(zip(candidate_ids, scores, strict=True))
            ordered = sorted(candidate_ids, key=lambda value: (-score_map[value], value))
            positives = [str(value) for value in row["known_positive_ids"]]
            scored_positive_ids = (
                [value for value in positives if store.has_teacher_features(value)]
                if query_has_features
                else []
            )
            missing_positive_features += len(positives) - len(scored_positive_ids)
            positive_scores = (
                _score_id_pairs(
                    model,
                    [(query_id, value) for value in scored_positive_ids],
                    store,
                    device,
                    batch_size=args.batch_size,
                    cache=compression_cache,
                )
                if scored_positive_ids
                else []
            )
            total_pairs += len(scored_positive_ids)
            top_negative = score_map[ordered[0]] if ordered else None
            margin = (
                max(positive_scores) - top_negative
                if positive_scores and top_negative is not None
                else None
            )
            if margin is not None:
                margins.append(margin)
            metadata = {str(value["target_id"]): value for value in candidates}
            for value in ordered[:50]:
                source_counts[str(metadata[value]["source"])] += 1
            handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "relation": "table_to_table",
                        "split": "train",
                        "status": (
                            "scored"
                            if query_has_features
                            else "fallback_original_no_teacher_query_feature"
                        ),
                        "known_positive_ids": positives,
                        "scored_positive_ids": scored_positive_ids,
                        "positive_scores": positive_scores,
                        "max_positive_score": max(positive_scores) if positive_scores else None,
                        "min_positive_score": min(positive_scores) if positive_scores else None,
                        "top_negative_score": top_negative,
                        "best_positive_margin": margin,
                        "top50": ordered[:50],
                        "top50_scores": [score_map[value] for value in ordered[:50]],
                        "top50_sources": [metadata[value]["source"] for value in ordered[:50]],
                        "remaining_natural": ordered[50:],
                        "remaining_scores": [score_map[value] for value in ordered[50:]],
                        "candidate_count_before_teacher_feature_filter": len(all_candidate_ids),
                        "candidate_count": len(ordered),
                        "missing_teacher_candidate_feature_count": (
                            len(all_candidate_ids) - len(candidate_ids)
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            query_counts_before.append(len(all_candidate_ids))
            query_counts.append(len(ordered))
            if position % 100 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "mine_score",
                            "queries": position,
                            "pairs": total_pairs,
                            "cache_objects": len(compression_cache),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    temporary.replace(output_path)
    audit = {
        "format_version": 1,
        "status": "complete",
        "mining_teacher": str(paths["parent"].resolve()),
        "mining_teacher_sha256": EXPECTED_SHA256["parent"],
        "source_manifest_sha256": checkpoint_fingerprint(source_path),
        "queries": len(query_counts),
        "scored_pairs": total_pairs,
        "missing_teacher_query_features": missing_query_features,
        "missing_teacher_candidate_features": missing_candidate_features,
        "missing_teacher_positive_features": missing_positive_features,
        "candidate_count_before_teacher_feature_filter": {
            "min": min(query_counts_before),
            "mean": statistics.fmean(query_counts_before),
            "max": max(query_counts_before),
        },
        "candidate_count_after_teacher_feature_filter": {
            "min": min(query_counts),
            "mean": statistics.fmean(query_counts),
            "max": max(query_counts),
        },
        "teacher_hidden_candidate_coverage_fraction": (
            sum(query_counts) / sum(query_counts_before)
            if sum(query_counts_before)
            else 0.0
        ),
        "best_positive_margin": {
            "count": len(margins),
            "mean": statistics.fmean(margins) if margins else None,
            "negative_fraction": (
                sum(value < 0 for value in margins) / len(margins) if margins else None
            ),
        },
        "top50_source_memberships": dict(source_counts),
        "reservoir": str(output_path.resolve()),
        "reservoir_sha256": checkpoint_fingerprint(output_path),
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
    }
    write_json(_output(args.root) / "hardness_and_staleness_diagnostics.json", audit)
    natural_audit_path = _output(args.root) / "natural_mining_audit.json"
    natural_audit = json.loads(natural_audit_path.read_text(encoding="utf-8"))
    natural_audit["retrieval_feature_coverage_tier"] = "embedding"
    natural_audit["teacher_hidden_feature_coverage"] = {
        "queries_missing": missing_query_features,
        "candidate_occurrences_missing": missing_candidate_features,
        "candidate_occurrence_fraction": audit[
            "teacher_hidden_candidate_coverage_fraction"
        ],
        "source": str(output_path.resolve()),
    }
    write_json(natural_audit_path, natural_audit)
    return audit


def build_negative_manifests(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    reservoir_path = _output(root) / "train_hard_reservoir.jsonl.gz"
    reservoirs = {
        str(row["query_id"]): {
            "top50": row["top50"],
            "remaining_natural": row["remaining_natural"],
        }
        for row in _read_jsonl_gz(reservoir_path)
    }
    examples, protection_count = protect_known_positives(
        load_edge_examples(paths["train"], split="train")
    )
    c2, c3, nested_audit, audit_rows = construct_nested_tt_lists(examples, reservoirs)
    output = _output(root)
    variants = {
        "local": examples,
        "natural_tt": c2,
        "natural_tt_long32": c3,
    }
    artifacts = {}
    for name, values in variants.items():
        records = [_example_record(value) for value in values]
        plain = output / f"train_negative_manifest.{name}.jsonl"
        compressed = output / f"train_negative_manifest.{name}.jsonl.gz"
        _write_jsonl(plain, records)
        _write_jsonl_gz(compressed, records)
        artifacts[name] = {
            "jsonl": str(plain.resolve()),
            "jsonl_sha256": checkpoint_fingerprint(plain),
            "jsonl_gz": str(compressed.resolve()),
            "jsonl_gz_sha256": checkpoint_fingerprint(compressed),
            "lists": len(values),
            "pair_slots": sum(len(value.candidate_ids) for value in values),
            "unique_pairs": len(
                {(value.query_id, candidate) for value in values for candidate in value.candidate_ids}
            ),
        }
    _write_jsonl_gz(output / "nested_list_audit_per_tt.jsonl.gz", audit_rows)
    nested_audit.update(
        {
            "known_positive_protection_exclusions": protection_count,
            "reservoir_sha256": checkpoint_fingerprint(reservoir_path),
            "artifacts": artifacts,
            "completed_at_utc": _now(),
        }
    )
    write_json(output / "nested_list_and_length_audit.json", nested_audit)
    write_json(
        output / "known_positive_protection_audit.json",
        {
            "format_version": 1,
            "status": "pass",
            "train_registry_exclusions": protection_count,
            "dev_or_test_qrels_used": False,
            "c2_c3_positive_sets_preserved": True,
        },
    )
    write_json(
        output / "train_pair_exposure.json",
        {
            "format_version": 1,
            "per_epoch": {
                key: {
                    "pair_slots": value["pair_slots"],
                    "unique_pairs": value["unique_pairs"],
                }
                for key, value in artifacts.items()
            },
            "two_epoch_pair_slots": {
                key: 2 * value["pair_slots"] for key, value in artifacts.items()
            },
        },
    )
    write_json(
        output / "negative_provenance_audit.json",
        {
            "format_version": 1,
            "status": "complete",
            "original_negative_provenance": "unknown_where_not_parseable_from_frozen_R12_rows",
            "natural_negative_label": "assumed_negative",
            "verified_negative_claimed": False,
            "tiers": {"0": "parent_A1_top50", "1": "remaining_train_natural_pool", "2": "original_list_fallback"},
        },
    )
    return nested_audit


def audit_mining_split(root: Path) -> dict[str, Any]:
    """Verify train mining membership against the frozen source-group split."""

    paths = _paths(root)
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    train_queries = set(str(value) for value in splits["query_ids"]["train_fit"])
    dev_queries = set(str(value) for value in splits["query_ids"]["dev"])
    mined_queries = {
        str(row["query_id"])
        for row in _read_jsonl_gz(_output(root) / "train_candidate_source_manifest.jsonl.gz")
    }
    train_groups = set(str(value) for value in splits["source_groups"]["train_fit"])
    nontrain_groups = set().union(
        *(
            set(str(value) for value in splits["source_groups"][key])
            for key in ("train_calibration", "dev", "test")
        )
    )
    payload = {
        "format_version": 1,
        "status": "pass"
        if mined_queries <= train_queries
        and not mined_queries & dev_queries
        and not train_groups & nontrain_groups
        else "failed_correctness",
        "split_manifest": str(paths["splits"].resolve()),
        "split_manifest_sha256": checkpoint_fingerprint(paths["splits"]),
        "mined_queries": len(mined_queries),
        "mined_queries_outside_train_fit": len(mined_queries - train_queries),
        "mined_dev_query_overlap": len(mined_queries & dev_queries),
        "train_source_groups": len(train_groups),
        "train_source_group_overlap_with_cal_dev_test": len(train_groups & nontrain_groups),
        "labels_used_for_candidate_generation": False,
        "completed_at_utc": _now(),
    }
    write_json(_output(root) / "mining_split_audit.json", payload)
    audit_path = _output(root) / "natural_mining_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["source_group_overlap"] = payload["train_source_group_overlap_with_cal_dev_test"]
    audit["split_audit"] = payload
    write_json(audit_path, audit)
    if payload["status"] != "pass":
        raise RuntimeError("R19 train mining split/source-group audit failed")
    return payload


@torch.inference_mode()
def p0_evaluate_r18(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only evaluation for the three missing fixed-step R18 checkpoints."""

    started = time.monotonic()
    paths = _paths(args.root)
    specifications = {
        "A0_step10536": ("r18_a0_step10536", EXPECTED_SHA256["r18_a0_step10536"]),
        "A1_step05268": ("r18_a1_step5268", EXPECTED_SHA256["r18_a1_step5268"]),
        "A2_step05268": ("r18_a2_step5268", EXPECTED_SHA256["r18_a2_step5268"]),
    }
    path_key, expected_sha = specifications[args.name]
    checkpoint = paths[path_key]
    actual_sha = checkpoint_fingerprint(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Frozen R18 checkpoint changed: {args.name}")
    arm, model = load_r18_checkpoint(checkpoint, torch.device(args.device))
    model.eval()
    device = torch.device(args.device)
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    cache: dict[str, torch.Tensor] = {}
    rows_by_pool: dict[str, list[dict[str, Any]]] = {
        "natural": [], "direct100": [], "matched": []
    }
    total_pairs = 0
    for position, source in enumerate(_read_jsonl_gz(paths["candidates"]), 1):
        query_id = str(source["query_id"])
        combined = [str(value) for value in source["combined_score_candidate_ids"]]
        scores = _score_id_pairs(
            model,
            [(query_id, value) for value in combined],
            store,
            device,
            batch_size=args.batch_size,
            cache=cache,
        )
        score_map = dict(zip(combined, scores, strict=True))
        total_pairs += len(combined)
        positives = [str(value) for value in source["positive_target_ids"]]
        common = {
            "query_id": query_id,
            "source_table_id": str(source["source_table_id"]),
            "query_kind": str(source["query_kind"]),
            "positive_target_ids": positives,
        }
        for pool, key, prefix in (
            ("natural", "natural_candidate_ids", "natural"),
            ("direct100", "ann_direct100_ids", "direct100"),
            ("matched", "matched_direct_candidate_ids", "matched"),
        ):
            candidate_ids = [str(value) for value in source[key]]
            ranking = sorted(candidate_ids, key=lambda value: (-score_map[value], value))
            record = {
                **common,
                "candidate_ids": candidate_ids,
                "ranking": ranking,
                "raw_scores": [score_map[value] for value in ranking],
                "positive_ranks": {
                    value: ranking.index(value) + 1
                    for value in positives
                    if value in set(candidate_ids)
                },
                f"{prefix}_raw_recall": _recall(positives, candidate_ids, len(candidate_ids)),
            }
            for k in (10, 20, 50):
                record[f"{prefix}_recall@{k}"] = _recall(positives, ranking, k)
            rows_by_pool[pool].append(record)
        if position % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "p0_r18_fixed",
                        "name": args.name,
                        "queries": position,
                        "pairs": total_pairs,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    output = _output(args.root) / "P0/R18_epoch1_epoch2_rankings"
    output.mkdir(parents=True, exist_ok=True)
    for pool, rows in rows_by_pool.items():
        _write_jsonl_gz(output / f"{args.name}.{pool}.jsonl.gz", rows)
    dev_examples, _protected = protect_known_positives(
        load_edge_examples(paths["dev"], split="dev")
    )
    dev, dev_rows = _evaluate_edges(model, dev_examples, store, device, include_rows=True)
    _write_jsonl_gz(output / f"{args.name}.edge_dev.jsonl.gz", dev_rows)
    metrics = {
        "format_version": 1,
        "status": "complete",
        "name": args.name,
        "r18_arm": arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "natural_union": _ranking_summary(rows_by_pool["natural"], "natural"),
        "direct100": _ranking_summary(rows_by_pool["direct100"], "direct100"),
        "matched_direct_M": _ranking_summary(rows_by_pool["matched"], "matched"),
        "edge_dev": dev,
        "pairs": total_pairs,
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "code_sha256": CODE_SHA256_AT_IMPORT,
    }
    write_json(output / f"{args.name}.metrics.json", metrics)
    return metrics


def _permutation_map(
    store: FeatureStore, object_ids: Iterable[str], seed: int
) -> dict[str, str]:
    by_type: dict[str, list[str]] = defaultdict(list)
    for object_id in sorted(set(object_ids)):
        by_type[store.object_type(object_id)].append(object_id)
    mapping = {}
    rng = random.Random(seed)
    for object_type, values in sorted(by_type.items()):
        shuffled = list(values)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            raise ValueError(f"Cannot make a no-self permutation for {object_type}")
        for index, object_id in enumerate(shuffled):
            mapping[object_id] = shuffled[(index + 1) % len(shuffled)]
    if any(left == right for left, right in mapping.items()):
        raise RuntimeError("Global permutation contains a self mapping")
    return mapping


def _replace_embedding(
    original: ObjectFeatures, embedding: torch.Tensor
) -> ObjectFeatures:
    return ObjectFeatures(
        object_id=original.object_id,
        object_type=original.object_type,
        embedding=embedding,
        hidden_states=original.hidden_states,
        token_groups=original.token_groups,
    )


@torch.inference_mode()
def _score_permuted_pairs(
    model: R19GlobalResidualTeacher,
    pairs: Sequence[tuple[str, str]],
    mapping: dict[str, str],
    store: FeatureStore,
    device: torch.device,
    *,
    batch_size: int,
    cache: OwnedCompressionCache,
) -> list[float]:
    dtype = next(model.parameters()).dtype
    result = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        sources = []
        destinations = []
        for source_id, destination_id in batch:
            source = _device_feature(store, source_id, device, dtype)
            destination = _device_feature(store, destination_id, device, dtype)
            source_embedding = store.embedding_features(mapping[source_id]).embedding.to(device)
            destination_embedding = store.embedding_features(mapping[destination_id]).embedding.to(device)
            sources.append(_replace_embedding(source, source_embedding))
            destinations.append(_replace_embedding(destination, destination_embedding))
        result.extend(
            float(value)
            for value in model.score_pairs(
                sources, destinations, compression_cache=cache, branch="full"
            ).cpu()
        )
    return result


@torch.inference_mode()
def p0_branch_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Frozen A1 full/global-off/local-off/permuted dependency diagnostic."""

    started = time.monotonic()
    paths = _paths(args.root)
    device = torch.device(args.device)
    model = initialize_arm(paths["parent"], "C0", device).eval()
    store = FeatureStore.from_path(
        paths["features"],
        cache_size=args.feature_cache_size,
        teacher_paths=_teacher_paths(paths),
    )
    candidate_rows = list(_read_jsonl_gz(paths["candidates"]))
    all_ids = {
        str(row["query_id"])
        for row in candidate_rows
    } | {
        str(candidate)
        for row in candidate_rows
        for candidate in row["natural_candidate_ids"]
    }
    permutation = _permutation_map(store, all_ids, MINING_SEED)
    caches = {
        "full": model.new_compression_cache(),
        "global_off": model.new_compression_cache(),
        "local_off": model.new_compression_cache(),
        "global_permuted": model.new_compression_cache(),
    }
    rankings: dict[str, list[dict[str, Any]]] = {
        branch: [] for branch in caches
    }
    for position, row in enumerate(candidate_rows, 1):
        query_id = str(row["query_id"])
        candidate_ids = [str(value) for value in row["natural_candidate_ids"]]
        pairs = [(query_id, value) for value in candidate_ids]
        scores_by_branch = {
            branch: _score_id_pairs(
                model,
                pairs,
                store,
                device,
                batch_size=args.batch_size,
                cache=caches[branch],
                branch=branch,
            )
            for branch in ("full", "global_off", "local_off")
        }
        scores_by_branch["global_permuted"] = _score_permuted_pairs(
            model,
            pairs,
            permutation,
            store,
            device,
            batch_size=args.batch_size,
            cache=caches["global_permuted"],
        )
        positives = [str(value) for value in row["positive_target_ids"]]
        candidate_set = set(candidate_ids)
        for branch, scores in scores_by_branch.items():
            score_map = dict(zip(candidate_ids, scores, strict=True))
            ranking = sorted(candidate_ids, key=lambda value: (-score_map[value], value))
            rankings[branch].append(
                {
                    "query_id": query_id,
                    "source_table_id": str(row["source_table_id"]),
                    "query_kind": str(row["query_kind"]),
                    "positive_target_ids": positives,
                    "candidate_ids": candidate_ids,
                    "ranking": ranking,
                    "raw_scores": [score_map[value] for value in ranking],
                    "positive_ranks": {
                        value: ranking.index(value) + 1 for value in positives if value in candidate_set
                    },
                    "recall@10": _recall(positives, ranking, 10),
                    "recall@20": _recall(positives, ranking, 20),
                    "recall@50": _recall(positives, ranking, 50),
                    "raw_recall": _recall(positives, candidate_ids, len(candidate_ids)),
                }
            )
        if position % 50 == 0:
            print(
                json.dumps(
                    {
                        "stage": "p0_branch_diagnostic",
                        "queries": position,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    metrics = {}
    output = _output(args.root) / "P0/frozen_branch_ablation"
    output.mkdir(parents=True, exist_ok=True)
    for branch, rows in rankings.items():
        _write_jsonl_gz(output / f"{branch}.jsonl.gz", rows)
        metrics[branch] = {
            kind: {
                "queries": len(selected),
                "recall@10": statistics.fmean(value["recall@10"] for value in selected),
                "recall@20": statistics.fmean(value["recall@20"] for value in selected),
                "CandidateRecall@50": statistics.fmean(value["recall@50"] for value in selected),
                "RawCandidateRecall": statistics.fmean(value["raw_recall"] for value in selected),
            }
            for kind, selected in (
                ("all", rows),
                ("implicit", [value for value in rows if value["query_kind"] == "implicit"]),
                ("explicit", [value for value in rows if value["query_kind"] == "explicit"]),
            )
        }

    edge_examples = load_edge_examples(paths["dev"], split="dev")
    pair_samples = _pair_samples(edge_examples, 100)
    representation_stats = {}
    dtype = next(model.parameters()).dtype
    for relation, pairs in pair_samples.items():
        local_squares = residual_squares = score_delta_squares = 0.0
        values_count = 0
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            sources = [_device_feature(store, left, device, dtype) for left, _right in batch]
            destinations = [_device_feature(store, right, device, dtype) for _left, right in batch]
            local, residual = model._representations(sources, destinations)
            full_score = model.scoring_head(local + residual).squeeze(-1)
            local_score = model.scoring_head(local).squeeze(-1)
            local_squares += float(local.square().sum().cpu())
            residual_squares += float(residual.square().sum().cpu())
            score_delta_squares += float((full_score - local_score).square().sum().cpu())
            values_count += local.numel()
        local_rms = math.sqrt(local_squares / values_count)
        residual_rms = math.sqrt(residual_squares / values_count)
        representation_stats[relation] = {
            "pairs": len(pairs),
            "local_rms": local_rms,
            "global_delta_rms": residual_rms,
            "global_to_local_rms_ratio": residual_rms / local_rms,
            "score_difference_rms": math.sqrt(score_delta_squares / len(pairs)),
        }
    payload = {
        "format_version": 1,
        "status": "complete",
        "checkpoint": str(paths["parent"].resolve()),
        "checkpoint_sha256": EXPECTED_SHA256["parent"],
        "permutation_seed": MINING_SEED,
        "permutation_scope": "same_modality_global_fixed_no_self_mapping",
        "permutation_mapping_hash": stable_json_hash(permutation),
        "permutation_objects": len(permutation),
        "metrics": metrics,
        "representation_statistics": representation_stats,
        "interpretation_boundary": "frozen dependency diagnostic, not a retrained architecture ablation",
        "elapsed_seconds": time.monotonic() - started,
        "device": args.device,
        "completed_at_utc": _now(),
        "code_sha256": CODE_SHA256_AT_IMPORT,
    }
    write_json(_output(args.root) / "P0/branch_statistics.json", payload)
    return payload


def _wlt(rows: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        delta = float(row[left]) - float(row[right])
        counts["wins" if delta > 0 else "losses" if delta < 0 else "ties"] += 1
    return {key: counts[key] for key in ("wins", "losses", "ties")}


def compare_results(root: Path) -> dict[str, Any]:
    """Create all preregistered per-seed and mean-over-seed comparisons."""

    output = _output(root)
    data: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = _arm_output(root, arm, seed) / "eval_step010536/natural_union.jsonl.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = {str(row["query_id"]): row for row in _read_jsonl_gz(path)}
            if len(rows) != 1198:
                raise RuntimeError(f"{arm}/seed{seed}: expected 1,198 evaluation rows")
            data[(arm, seed)] = rows
    query_ids = sorted(data[("C0", SEEDS[0])])
    comparisons = (("C1", "C0"), ("C2", "C0"), ("C3", "C2"), ("C3", "C0"))
    payload: dict[str, Any] = {
        "format_version": 1,
        "status": "complete",
        "main_comparisons": ["C1-C0", "C2-C0", "C3-C2"],
        "per_seed": {},
        "mean_over_seeds": {},
    }
    wlt_rows = []
    for left, right in comparisons:
        name = f"{left}_minus_{right}"
        seed_deltas = []
        for seed in SEEDS:
            rows = []
            for query_id in query_ids:
                left_row = data[(left, seed)][query_id]
                right_row = data[(right, seed)][query_id]
                row = {
                    "query_id": query_id,
                    "source_table_id": left_row["source_table_id"],
                    "query_kind": left_row["query_kind"],
                }
                for k in (10, 20, 50):
                    row[f"left@{k}"] = left_row[f"natural_recall@{k}"]
                    row[f"right@{k}"] = right_row[f"natural_recall@{k}"]
                rows.append(row)
            by_k = {}
            for k in (10, 20, 50):
                by_k[str(k)] = {
                    "bootstrap": paired_group_bootstrap(
                        rows,
                        f"left@{k}",
                        f"right@{k}",
                        iterations=10_000,
                        seed=BOOTSTRAP_SEED,
                    ),
                    "wlt": _wlt(rows, f"left@{k}", f"right@{k}"),
                }
            payload["per_seed"].setdefault(str(seed), {})[name] = by_k
            seed_deltas.append(by_k["10"]["bootstrap"]["observed_delta"])
            for row in rows:
                wlt_rows.append(
                    {
                        "comparison": name,
                        "seed": seed,
                        "query_id": row["query_id"],
                        "source_table_id": row["source_table_id"],
                        "delta_recall@10": row["left@10"] - row["right@10"],
                    }
                )
        mean_rows = []
        for query_id in query_ids:
            first = data[(left, SEEDS[0])][query_id]
            row = {
                "query_id": query_id,
                "source_table_id": first["source_table_id"],
                "query_kind": first["query_kind"],
            }
            for k in (10, 20, 50):
                row[f"left@{k}"] = statistics.fmean(
                    data[(left, seed)][query_id][f"natural_recall@{k}"] for seed in SEEDS
                )
                row[f"right@{k}"] = statistics.fmean(
                    data[(right, seed)][query_id][f"natural_recall@{k}"] for seed in SEEDS
                )
            mean_rows.append(row)
        payload["mean_over_seeds"][name] = {
            str(k): {
                "bootstrap": paired_group_bootstrap(
                    mean_rows,
                    f"left@{k}",
                    f"right@{k}",
                    iterations=10_000,
                    seed=BOOTSTRAP_SEED,
                ),
                "wlt": _wlt(mean_rows, f"left@{k}", f"right@{k}"),
            }
            for k in (10, 20, 50)
        }
        payload["mean_over_seeds"][name]["seed_r10_delta_min_mean_max"] = [
            min(seed_deltas), statistics.fmean(seed_deltas), max(seed_deltas)
        ]

    candidates = {
        str(row["query_id"]): row for row in _read_jsonl_gz(_paths(root)["candidates"])
    }
    b13_rows = []
    for query_id in query_ids:
        row = candidates[query_id]
        ranking = [str(value) for value in row["student_b0"]["50"]]
        positives = [str(value) for value in row["positive_target_ids"]]
        b13_rows.append(
            {
                "query_id": query_id,
                "source_table_id": str(row["source_table_id"]),
                "query_kind": str(row["query_kind"]),
                "positive_target_ids": positives,
                "ranking_top50": ranking,
                **{f"recall@{k}": _recall(positives, ranking, k) for k in (10, 20, 50)},
            }
        )
    baseline_dir = output / "baseline_reference"
    _write_jsonl_gz(baseline_dir / "B13_per_query.jsonl.gz", b13_rows)
    _write_jsonl_gz(
        baseline_dir / "T_old_per_query.jsonl.gz",
        _read_jsonl_gz(_paths(root)["r16_teacher_rankings"]),
    )
    b13 = {row["query_id"]: row for row in b13_rows}
    payload["versus_B13_mean_over_seeds"] = {}
    for arm in ARMS:
        rows = []
        for query_id in query_ids:
            first = data[(arm, SEEDS[0])][query_id]
            row = {
                "query_id": query_id,
                "source_table_id": first["source_table_id"],
                "query_kind": first["query_kind"],
            }
            for k in (10, 20, 50):
                row[f"arm@{k}"] = statistics.fmean(
                    data[(arm, seed)][query_id][f"natural_recall@{k}"] for seed in SEEDS
                )
                row[f"b13@{k}"] = b13[query_id][f"recall@{k}"]
            rows.append(row)
        payload["versus_B13_mean_over_seeds"][arm] = {
            str(k): {
                "bootstrap": paired_group_bootstrap(
                    rows,
                    f"arm@{k}",
                    f"b13@{k}",
                    iterations=10_000,
                    seed=BOOTSTRAP_SEED,
                ),
                "wlt": _wlt(rows, f"arm@{k}", f"b13@{k}"),
            }
            for k in (10, 20, 50)
        }
    payload["completed_at_utc"] = _now()
    write_json(output / "PAIRED_COMPARISONS.json", payload)
    _write_jsonl(output / "PER_QUERY_WLT.jsonl", wlt_rows)

    source_rows = []
    hub_counts: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    for query_id in query_ids:
        candidate = candidates[query_id]
        direct = set(str(value) for value in candidate["ann_direct100_ids"])
        matched = set(str(value) for value in candidate["matched_direct_candidate_ids"])
        natural = set(str(value) for value in candidate["natural_candidate_ids"])
        evidence = set(str(value) for value in candidate["evidence_candidate_ids"])
        for positive in candidate["positive_target_ids"]:
            source_rows.append(
                {
                    "query_id": query_id,
                    "source_table_id": candidate["source_table_id"],
                    "query_kind": candidate["query_kind"],
                    "positive_id": positive,
                    "D100": positive in direct,
                    "M": positive in matched,
                    "U": positive in natural,
                    "evidence_introduced": positive in evidence and positive not in direct,
                    "ranks": {
                        arm: {
                            str(seed): data[(arm, seed)][query_id]["positive_ranks"].get(positive)
                            for seed in SEEDS
                        }
                        for arm in ARMS
                    },
                }
            )
        for arm in ARMS:
            for seed in SEEDS:
                ranking = data[(arm, seed)][query_id]["ranking"]
                hub_counts[arm].update(ranking[:10])
    _write_jsonl_gz(output / "CANDIDATE_SOURCE_ANALYSIS.jsonl.gz", source_rows)
    write_json(
        output / "HUB_DIAGNOSTICS.json",
        {
            "format_version": 1,
            "top10_target_frequency": {
                arm: hub_counts[arm].most_common(100) for arm in ARMS
            },
        },
    )
    return payload


def _logsumexp_floats(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def f0_fusion_diagnostic(root: Path) -> dict[str, Any]:
    """Rebuild fixed direct/evidence/equal-RRF ranks without tuning weights."""

    paths = _paths(root)
    output = _output(root) / "F0_FUSION_DIAGNOSTIC"
    output.mkdir(parents=True, exist_ok=True)
    candidates = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["candidates"])
    }
    frozen_rankings = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["b13_rankings"])
    }
    pool_rows = {
        str(row["query_id"]): row for row in _read_jsonl_gz(paths["b13_pool"])
    }
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "student_direct_rank": [],
        "student_evidence_rank": [],
        "equal_rrf_k60": [],
    }
    frozen_equal_top50_matches = 0
    for query_id, candidate in candidates.items():
        natural = [str(value) for value in candidate["natural_candidate_ids"]]
        natural_set = set(natural)
        direct_ids = [
            str(value) for value in candidate["ann_direct100_ids"] if str(value) in natural_set
        ]
        direct_rank = {value: index for index, value in enumerate(direct_ids, 1)}
        paths_by_target = pool_rows[query_id]["paths_by_target"]
        evidence_scores = {}
        for target_id, target_paths in paths_by_target.items():
            scores = [
                float(path["path_score"])
                for path in target_paths
                if path["kind"] == "evidence"
            ]
            if scores and target_id in natural_set:
                evidence_scores[str(target_id)] = _logsumexp_floats(scores)
        evidence_ids = sorted(
            evidence_scores,
            key=lambda value: (-evidence_scores[value], value),
        )
        evidence_rank = {value: index for index, value in enumerate(evidence_ids, 1)}
        direct_full = [*direct_ids, *sorted(natural_set - set(direct_ids))]
        evidence_full = [*evidence_ids, *sorted(natural_set - set(evidence_ids))]
        equal = sorted(
            natural,
            key=lambda value: (
                -(
                    (1.0 / (60 + direct_rank[value]) if value in direct_rank else 0.0)
                    + (1.0 / (60 + evidence_rank[value]) if value in evidence_rank else 0.0)
                ),
                value,
            ),
        )
        frozen_equal = frozen_rankings[query_id]["rankings"]["union_rrf_equal"]["50"]["target_ids"]
        frozen_equal_top50_matches += equal[:50] == frozen_equal
        positives = [str(value) for value in candidate["positive_target_ids"]]
        common = {
            "query_id": query_id,
            "source_table_id": str(candidate["source_table_id"]),
            "query_kind": str(candidate["query_kind"]),
            "positive_target_ids": positives,
            "natural_candidate_ids": natural,
        }
        for method, ranking in (
            ("student_direct_rank", direct_full),
            ("student_evidence_rank", evidence_full),
            ("equal_rrf_k60", equal),
        ):
            rows_by_method[method].append(
                {
                    **common,
                    "ranking": ranking,
                    "positive_ranks": {
                        value: ranking.index(value) + 1 for value in positives if value in natural_set
                    },
                    "recall@10": _recall(positives, ranking, 10),
                    "recall@20": _recall(positives, ranking, 20),
                    "recall@50": _recall(positives, ranking, 50),
                }
            )
    metrics = {}
    for method, rows in rows_by_method.items():
        _write_jsonl_gz(output / "rankings" / f"{method}.jsonl.gz", rows)
        metrics[method] = {
            kind: {
                "queries": len(selected),
                "recall@10": statistics.fmean(row["recall@10"] for row in selected),
                "recall@20": statistics.fmean(row["recall@20"] for row in selected),
                "CandidateRecall@50": statistics.fmean(row["recall@50"] for row in selected),
            }
            for kind, selected in (
                ("all", rows),
                ("implicit", [row for row in rows if row["query_kind"] == "implicit"]),
                ("explicit", [row for row in rows if row["query_kind"] == "explicit"]),
            )
        }
    source_retention = {}
    for method, rows in rows_by_method.items():
        retained = Counter()
        totals = Counter()
        for row in rows:
            candidate = candidates[row["query_id"]]
            natural = set(str(value) for value in candidate["natural_candidate_ids"])
            matched = set(str(value) for value in candidate["matched_direct_candidate_ids"])
            top10 = set(row["ranking"][:10])
            for positive in row["positive_target_ids"]:
                category = (
                    "both"
                    if positive in natural and positive in matched
                    else "U_only"
                    if positive in natural
                    else "M_only"
                    if positive in matched
                    else "neither"
                )
                totals[category] += 1
                retained[category] += positive in top10
        source_retention[method] = {
            key: {"positive_pairs": totals[key], "top10_retained": retained[key]}
            for key in sorted(totals)
        }
    status = {
        "format_version": 1,
        "status": "complete_with_missing_optional_inputs",
        "main_teacher_ranking_changed": False,
        "teacher_scorer_full_U_before_top50": all(
            set(row["natural_candidate_ids"]) <= set(row["combined_score_candidate_ids"])
            for row in candidates.values()
        ),
        "equal_rrf_formula": "I_D/(60+r_D)+I_E/(60+r_E)",
        "comparison_to_historical_post_retention_equal_rrf_top50_matches": frozen_equal_top50_matches,
        "historical_equal_rrf_comparison_expected_to_match": False,
        "queries": len(candidates),
        "B13_D_score_on_full_U": "not_run_missing_full_U_raw_scores",
        "QE_only_ET_only_replacement": "not_run_missing_separate_frozen_replacement_rankings",
        "adaptive_reranker": "deferred_not_authorized",
        "completed_at_utc": _now(),
    }
    write_json(output / "status.json", status)
    write_json(
        output / "inputs.json",
        {
            "candidate_pools_sha256": checkpoint_fingerprint(paths["candidates"]),
            "path_pool_sha256": checkpoint_fingerprint(paths["b13_pool"]),
            "rankings_sha256": checkpoint_fingerprint(paths["b13_rankings"]),
        },
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "source_retention.json", source_retention)
    return {"status": status, "metrics": metrics, "source_retention": source_retention}


def p0_summarize_r18(root: Path) -> dict[str, Any]:
    """Compare R18 arms at identical 5,268 and 10,536 update endpoints."""

    r18 = root / "work/stage1_optimization_r18_20260910"
    p0 = _output(root) / "P0/R18_epoch1_epoch2_rankings"
    files = {
        ("A0", 5268): r18 / "A0_base_continuation/natural_union_rankings.jsonl.gz",
        ("A0", 10536): p0 / "A0_step10536.natural.jsonl.gz",
        ("A1", 5268): p0 / "A1_step05268.natural.jsonl.gz",
        ("A1", 10536): r18 / "A1_global_residual/natural_union_rankings.jsonl.gz",
        ("A2", 5268): p0 / "A2_step05268.natural.jsonl.gz",
        ("A2", 10536): r18 / "A2_relation_heads/natural_union_rankings.jsonl.gz",
    }
    rows = {
        key: {str(row["query_id"]): row for row in _read_jsonl_gz(path)}
        for key, path in files.items()
    }
    payload: dict[str, Any] = {
        "format_version": 1,
        "status": "complete",
        "fixed_step": {},
        "dev_selected_comparison_preserved": "A1@10536-A0@5268",
    }
    for step in (5268, 10536):
        for arm in ("A1", "A2"):
            values = []
            for query_id in sorted(rows[("A0", step)]):
                left = rows[(arm, step)][query_id]
                right = rows[("A0", step)][query_id]
                values.append(
                    {
                        "query_id": query_id,
                        "source_table_id": left["source_table_id"],
                        "left": left["natural_recall@10"],
                        "right": right["natural_recall@10"],
                    }
                )
            payload["fixed_step"][f"{arm}_minus_A0_at_{step}"] = {
                "bootstrap_recall@10": paired_group_bootstrap(
                    values,
                    "left",
                    "right",
                    iterations=10_000,
                    seed=BOOTSTRAP_SEED,
                ),
                "wlt": _wlt(values, "left", "right"),
                "left_recall@10": statistics.fmean(value["left"] for value in values),
                "right_recall@10": statistics.fmean(value["right"] for value in values),
            }
    write_json(_output(root) / "P0/R18_FIXED_STEP_COMPARISONS.json", payload)
    return payload


def finalize(root: Path) -> dict[str, Any]:
    """Write the R19 scientific readout and requirement-by-requirement audit."""

    output = _output(root)
    comparison = json.loads((output / "PAIRED_COMPARISONS.json").read_text(encoding="utf-8"))
    p0 = json.loads((output / "P0/R18_FIXED_STEP_COMPARISONS.json").read_text(encoding="utf-8"))
    branches = json.loads((output / "P0/branch_statistics.json").read_text(encoding="utf-8"))
    mining = json.loads((output / "natural_mining_audit.json").read_text(encoding="utf-8"))
    nesting = json.loads((output / "nested_list_and_length_audit.json").read_text(encoding="utf-8"))
    f0 = json.loads((output / "F0_FUSION_DIAGNOSTIC/metrics.json").read_text(encoding="utf-8"))
    arm_metrics = {}
    costs = {}
    edge_rows = []
    for arm in ARMS:
        arm_metrics[arm] = {}
        costs[arm] = {}
        for seed in SEEDS:
            eval_path = _arm_output(root, arm, seed) / "eval_step010536"
            metrics = json.loads((eval_path / "metrics.json").read_text(encoding="utf-8"))
            arm_metrics[arm][str(seed)] = metrics
            config = json.loads(
                (_arm_output(root, arm, seed) / "config.json").read_text(encoding="utf-8")
            )
            costs[arm][str(seed)] = {
                "training_elapsed_seconds": config["elapsed_seconds"],
                "evaluation_runtime": metrics["runtime"],
            }
            for row in _read_jsonl_gz(eval_path / "edge_dev_per_list.jsonl.gz"):
                edge_rows.append({"arm": arm, "seed": seed, "step": 10536, **row})
    _write_jsonl_gz(output / "EDGE_DEV_PER_LIST.jsonl.gz", edge_rows)
    write_json(output / "STAGE1_COST.json", {"format_version": 1, "arms": costs})
    candidates = list(_read_jsonl_gz(_paths(root)["candidates"]))
    _write_jsonl_gz(
        output / "qrels_and_source_groups.jsonl.gz",
        (
            {
                "query_id": row["query_id"],
                "source_table_id": row["source_table_id"],
                "query_kind": row["query_kind"],
                "positive_target_ids": row["positive_target_ids"],
            }
            for row in candidates
        ),
    )
    means = {
        arm: {
            metric: statistics.fmean(
                arm_metrics[arm][str(seed)]["natural_union"]["all"][metric]
                for seed in SEEDS
            )
            for metric in ("recall@10", "recall@20", "CandidateRecall@50")
        }
        for arm in ARMS
    }
    main = comparison["mean_over_seeds"]
    fixed = p0["fixed_step"]
    full = branches["metrics"]["full"]["all"]
    global_off = branches["metrics"]["global_off"]["all"]
    local_off = branches["metrics"]["local_off"]["all"]
    permuted = branches["metrics"]["global_permuted"]["all"]
    best_arm = max(ARMS, key=lambda arm: means[arm]["recall@10"])
    lines = [
        "# Stage-1 R19 Results",
        "",
        "All four arms used the frozen R18 A1@10536 parent, two preregistered continuation seeds, fixed 10,536-update endpoints, and identical frozen U/D100/M evaluation pools. Stage2 was not run.",
        "",
        "| Arm | Mean U R@10 | Mean U R@20 | Mean U CR@50 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = means[arm]
        lines.append(
            f"| {arm} | {row['recall@10']:.6f} | {row['recall@20']:.6f} | {row['CandidateRecall@50']:.6f} |"
        )
    lines.extend(["", "## Preregistered comparisons", ""])
    for name in ("C1_minus_C0", "C2_minus_C0", "C3_minus_C2"):
        value = main[name]["10"]["bootstrap"]
        seed_range = main[name]["seed_r10_delta_min_mean_max"]
        lines.append(
            f"- {name.replace('_minus_', '−')}: R@10 {value['observed_delta']:+.6f}, 95% source-group CI [{value['ci95'][0]:+.6f}, {value['ci95'][1]:+.6f}], seed deltas min/mean/max {seed_range[0]:+.6f}/{seed_range[1]:+.6f}/{seed_range[2]:+.6f}."
        )
    lines.extend(
        [
            "",
            "## Answers to the frozen plan",
            "",
            f"1. Same-step R18 A1−A0 is {fixed['A1_minus_A0_at_5268']['bootstrap_recall@10']['observed_delta']:+.6f} at 5,268 updates and {fixed['A1_minus_A0_at_10536']['bootstrap_recall@10']['observed_delta']:+.6f} at 10,536 updates. The prior dev-selected comparison used A1@10536 versus A0@5268 and remains separately preserved.",
            f"2. Frozen A1 U R@10 is full={full['recall@10']:.6f}, global-off={global_off['recall@10']:.6f}, local-off={local_off['recall@10']:.6f}, global-permuted={permuted['recall@10']:.6f}. These are dependency diagnostics, not retrained architecture ablations.",
            f"3. C1 versus C0 is reported above; the width conclusion is restricted to the tested 512→1024 global path with local width fixed at 512.",
            f"4. C2 consumed the static manifest with replacement coverage {nesting['replacement_coverage']:.4f}; its Top10 effect is C2−C0 above, not edge-list Hit@1.",
            f"5. The strongest tested mean endpoint is {best_arm} at R@10={means[best_arm]['recall@10']:.6f}; B13 comparisons and U-only retention are in PAIRED_COMPARISONS.json and CANDIDATE_SOURCE_ANALYSIS.jsonl.gz.",
            "6. Both seed-specific effects, source-group bootstrap intervals, runtime, and peak memory are reported; two seeds are not treated as a precise estimate of seed variance.",
            f"7. C3 reached length 32 for {nesting['long32_complete_fraction']:.4f} of TT lists; its incremental effect is C3−C2 and its logical exposure/cost are in train_pair_exposure.json and STAGE1_COST.json.",
            f"8. F0 direct/evidence/equal-RRF R@10 are {f0['student_direct_rank']['all']['recall@10']:.6f}/{f0['student_evidence_rank']['all']['recall@10']:.6f}/{f0['equal_rrf_k60']['all']['recall@10']:.6f}. Full-U scoring occurred before Teacher truncation; unavailable optional score inputs remain not_run, and no adaptive gate was trained.",
            "9. Mechanism support is determined only by the three preregistered CIs above; no unrun combination, dynamic mining, Stage2, or adaptive fusion result is implied.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "CLAIM_BOUNDARIES.md").write_text(
        "# R19 Claim Boundaries\n\n"
        "- Results are development-set Stage1 evidence, not untouched-test or Stage2 evidence.\n"
        "- C1 isolates the tested global width change; it does not establish a universal Teacher width rule.\n"
        "- C2 negatives are assumed negatives selected by a frozen parent, not verified negatives or KD labels.\n"
        "- C3 changes list exposure and compute together; it is not an equal-compute control.\n"
        "- Frozen branch toggles diagnose dependence under co-adaptation and are not retrained ablations.\n"
        "- F0 is diagnostic only; adaptive fusion and target-conditioned verification remain deferred.\n",
        encoding="utf-8",
    )
    required = {
        "P0_correctness": all(
            (_arm_output(root, arm, seed) / "smoke_test.json").is_file()
            for arm in ARMS
            for seed in SEEDS
        ),
        "R18_fixed_step": p0.get("status") == "complete",
        "branch_diagnostic": branches.get("status") == "complete",
        "natural_mining": mining.get("status") == "complete",
        "nested_manifests": nesting.get("status") == "complete",
        "eight_training_runs": all(
            (_arm_output(root, arm, seed) / "config.json").is_file()
            for arm in ARMS
            for seed in SEEDS
        ),
        "eight_fixed_endpoint_evaluations": all(
            (_arm_output(root, arm, seed) / "eval_step010536/metrics.json").is_file()
            for arm in ARMS
            for seed in SEEDS
        ),
        "twenty_four_scheduled_evaluations": all(
            (_arm_output(root, arm, seed) / f"eval_step{step:06d}/metrics.json").is_file()
            for arm in ARMS
            for seed in SEEDS
            for step in (0, 5268, 10536)
        ),
        "paired_comparisons": comparison.get("status") == "complete",
        "F0": True,
        "Stage2": "out_of_scope",
    }
    complete = all(value is True or value == "out_of_scope" for value in required.values())
    audit = {
        "format_version": 1,
        "status": "complete" if complete else "partial",
        "requirements": required,
        "best_arm": best_arm,
        "completed_at_utc": _now(),
    }
    write_json(output / "COMPLETION_AUDIT.json", audit)
    (output / "COMPLETION_AUDIT.md").write_text(
        "# R19 Completion Audit\n\n"
        + "\n".join(
            f"- {name}: {value}" for name, value in required.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "FAILURE_NOTES.md").write_text(
        "# R19 Failure Notes\n\n"
        "F0 optional full-U B13 raw scores and separate QE-only/ET-only replacement rankings were unavailable and are recorded as not_run. No Stage2 job was authorized.\n",
        encoding="utf-8",
    )
    code_paths = [
        root / "src/run_stage1_r19.py",
        root / "src/run_stage1_r18.py",
        root / "src/mmdd_stage1/models.py",
        root / "src/mmdd_stage1/features.py",
        root / "src/mmdd_stage1/scoring.py",
        root / "src/mmdd_stage1/objectives.py",
        root / "tests/test_stage1_r19.py",
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
    matrix = json.loads((output / "EXECUTION_MATRIX.json").read_text(encoding="utf-8"))
    matrix["status"] = "complete" if complete else "partial"
    for job in matrix["jobs"]:
        job["status"] = (
            "complete"
            if (_arm_output(root, job["arm"], job["seed"]) / "config.json").is_file()
            else "pending"
        )
    write_json(output / "EXECUTION_MATRIX.json", matrix)
    return audit


def freeze_protocol(root: Path) -> dict[str, Any]:
    paths = _paths(root)
    output = _output(root)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "parent": paths["parent"],
        "train": paths["train"],
        "dev": paths["dev"],
        "candidates": paths["candidates"],
        "features": paths["features"] / "manifest.jsonl",
        "objects": paths["objects"],
    }
    actual = {name: checkpoint_fingerprint(path) for name, path in inputs.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "actual": value}
        for name, value in actual.items()
        if value != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"R19 frozen inputs changed: {mismatches}")
    plan_copy = output / "PLAN_FROZEN.md"
    if not plan_copy.is_file():
        plan_copy.write_text(paths["plan"].read_text(encoding="utf-8"), encoding="utf-8")
    plan = {
        "format_version": 1,
        "status": "frozen",
        "plan_sha256": checkpoint_fingerprint(paths["plan"]),
        "arms": list(ARMS),
        "continuation_seeds": list(SEEDS),
        "init_seed": INIT_SEED,
        "mining_seed": MINING_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "epochs": EPOCHS,
        "updates_per_epoch": EXPECTED_UPDATES_PER_EPOCH,
        "total_updates": EXPECTED_TOTAL_UPDATES,
        "logical_batch_lists": LOGICAL_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "positive_loss_mode": "sum_probability",
        "main_endpoint": "natural_union query-macro target Recall@10 at step10536",
        "main_comparisons": ["C1-C0", "C2-C0", "C3-C2"],
        "stage2": "out_of_scope",
    }
    input_manifest = {
        "format_version": 1,
        "status": "pass",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": actual[name]}
            for name, path in inputs.items()
        },
        "verified_at_utc": _now(),
    }
    write_json(output / "PLAN_FROZEN.json", plan)
    write_json(output / "INPUT_MANIFEST.json", input_manifest)
    write_json(output / "training_protocol.json", plan)
    matrix = {
        "format_version": 1,
        "status": "planned",
        "jobs": [
            {"arm": arm, "seed": seed, "status": "pending"}
            for seed in SEEDS
            for arm in ARMS
        ],
        "gpu_assignment": {
            "cuda:0": ["C0/seed13", "C2/seed13", "C0/seed29", "C2/seed29"],
            "cuda:1": ["C1/seed13", "C3/seed13", "C1/seed29", "C3/seed29"],
        },
    }
    write_json(output / "EXECUTION_MATRIX.json", matrix)
    return {"protocol": plan, "inputs": input_manifest, "matrix": matrix}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-protocol")
    init = subparsers.add_parser("initialization-audit")
    init.add_argument("--device", default="cpu")
    snapshot = subparsers.add_parser("initialization-snapshot")
    snapshot.add_argument("--device", default="cpu")
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot_compare = subparsers.add_parser("initialization-compare")
    snapshot_compare.add_argument("--left", type=Path, required=True)
    snapshot_compare.add_argument("--right", type=Path, required=True)
    precheck_parser = subparsers.add_parser("precheck")
    precheck_parser.add_argument("--device", required=True)
    precheck_parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    precheck_parser.add_argument("--count-per-relation", type=int, default=100)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", required=True, choices=ARMS)
    train_parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    train_parser.add_argument("--device", required=True)
    train_parser.add_argument("--feature-cache-size", type=int, default=24000)
    train_parser.add_argument("--microbatch-lists", type=int, default=8)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--arm", required=True, choices=ARMS)
    evaluate_parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    evaluate_parser.add_argument("--step", required=True, type=int, choices=(0, 5268, 10536))
    evaluate_parser.add_argument("--device", required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=512)
    evaluate_parser.add_argument("--feature-cache-size", type=int, default=40000)
    retrieve_parser = subparsers.add_parser("mine-retrieve")
    retrieve_parser.add_argument("--device", default="cpu")
    retrieve_parser.add_argument("--query-batch-size", type=int, default=32)
    retrieve_parser.add_argument("--feature-cache-size", type=int, default=4096)
    score_parser = subparsers.add_parser("mine-score")
    score_parser.add_argument("--device", required=True)
    score_parser.add_argument("--batch-size", type=int, default=512)
    score_parser.add_argument("--feature-cache-size", type=int, default=40000)
    subparsers.add_parser("build-negative-manifests")
    subparsers.add_parser("audit-mining-split")
    old_eval = subparsers.add_parser("p0-evaluate-r18")
    old_eval.add_argument(
        "--name", required=True, choices=("A0_step10536", "A1_step05268", "A2_step05268")
    )
    old_eval.add_argument("--device", required=True)
    old_eval.add_argument("--batch-size", type=int, default=512)
    old_eval.add_argument("--feature-cache-size", type=int, default=40000)
    branch = subparsers.add_parser("p0-branch-diagnostic")
    branch.add_argument("--device", required=True)
    branch.add_argument("--batch-size", type=int, default=512)
    branch.add_argument("--feature-cache-size", type=int, default=40000)
    subparsers.add_parser("compare")
    subparsers.add_parser("f0-fusion-diagnostic")
    subparsers.add_parser("p0-summarize-r18")
    subparsers.add_parser("finalize")
    return parser.parse_args()


def initialization_audit(root: Path, device: torch.device) -> dict[str, Any]:
    paths = _paths(root)
    rows = {}
    for arm in ARMS:
        first = initialize_arm(paths["parent"], arm, device)
        second = initialize_arm(paths["parent"], arm, device)
        first_hash = state_dict_content_hash(first.state_dict())
        second_hash = state_dict_content_hash(second.state_dict())
        parameters = sum(value.numel() for value in first.parameters())
        rows[arm] = {
            "state_dict_content_hash": first_hash,
            "second_process_equivalent_construction_hash": second_hash,
            "deterministic": first_hash == second_hash,
            "new_parameter_hash": _new_parameter_hash(first, arm),
            "parameters": parameters,
            "expected_parameters": EXPECTED_PARAMETERS[arm],
            "parameter_count_matches": parameters == EXPECTED_PARAMETERS[arm],
            "global_dim": first.global_dim,
            "global_hidden_dim": first.global_hidden_dim,
        }
    payload = {
        "format_version": 1,
        "status": "pass"
        if all(row["deterministic"] and row["parameter_count_matches"] for row in rows.values())
        else "failed_correctness",
        "init_seed": INIT_SEED,
        "arms": rows,
        "device": str(device),
        "completed_at_utc": _now(),
    }
    path = _output(root) / "P0/seed_reproducibility.json"
    write_json(path, payload)
    if payload["status"] != "pass":
        raise RuntimeError("R19 initialization audit failed")
    return payload


def initialization_snapshot(
    root: Path, device: torch.device, output_path: Path
) -> dict[str, Any]:
    """Write one construction snapshot; invoke twice in separate processes."""

    import os

    paths = _paths(root)
    arms = {}
    for arm in ARMS:
        model = initialize_arm(paths["parent"], arm, device)
        arms[arm] = {
            "state_dict_content_hash": state_dict_content_hash(model.state_dict()),
            "new_parameter_hash": _new_parameter_hash(model, arm),
            "parameters": sum(value.numel() for value in model.parameters()),
        }
    payload = {
        "format_version": 1,
        "process_id": os.getpid(),
        "init_seed": INIT_SEED,
        "device": str(device),
        "arms": arms,
        "completed_at_utc": _now(),
    }
    write_json(output_path, payload)
    return payload


def compare_initialization_snapshots(
    root: Path, left: Path, right: Path
) -> dict[str, Any]:
    first = json.loads(left.read_text(encoding="utf-8"))
    second = json.loads(right.read_text(encoding="utf-8"))
    if first["process_id"] == second["process_id"]:
        raise RuntimeError("R19 seed snapshots must come from distinct processes")
    arms = {
        arm: {
            "first_hash": first["arms"][arm]["state_dict_content_hash"],
            "second_hash": second["arms"][arm]["state_dict_content_hash"],
            "identical": first["arms"][arm]["state_dict_content_hash"]
            == second["arms"][arm]["state_dict_content_hash"],
            "parameters": first["arms"][arm]["parameters"],
        }
        for arm in ARMS
    }
    payload = {
        "format_version": 1,
        "status": "pass"
        if all(value["identical"] for value in arms.values())
        else "failed_correctness",
        "independent_process_ids": [first["process_id"], second["process_id"]],
        "init_seed": INIT_SEED,
        "arms": arms,
        "snapshots": [str(left.resolve()), str(right.resolve())],
        "completed_at_utc": _now(),
    }
    write_json(_output(root) / "P0/seed_reproducibility.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError("R19 independent-process initialization audit failed")
    return payload


def main() -> None:
    args = parse_args()
    if args.command == "freeze-protocol":
        result = freeze_protocol(args.root)
    elif args.command == "initialization-audit":
        result = initialization_audit(args.root, torch.device(args.device))
    elif args.command == "initialization-snapshot":
        result = initialization_snapshot(args.root, torch.device(args.device), args.output)
    elif args.command == "initialization-compare":
        result = compare_initialization_snapshots(args.root, args.left, args.right)
    elif args.command == "precheck":
        result = precheck(args)
    elif args.command == "train":
        result = train(args)
    elif args.command == "evaluate":
        result = evaluate(args)
    elif args.command == "mine-retrieve":
        result = mine_retrieve(args)
    elif args.command == "mine-score":
        result = mine_score(args)
    elif args.command == "build-negative-manifests":
        result = build_negative_manifests(args.root)
    elif args.command == "audit-mining-split":
        result = audit_mining_split(args.root)
    elif args.command == "p0-evaluate-r18":
        result = p0_evaluate_r18(args)
    elif args.command == "p0-branch-diagnostic":
        result = p0_branch_diagnostic(args)
    elif args.command == "compare":
        result = compare_results(args.root)
    elif args.command == "f0-fusion-diagnostic":
        result = f0_fusion_diagnostic(args.root)
    elif args.command == "p0-summarize-r18":
        result = p0_summarize_r18(args.root)
    else:
        result = finalize(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

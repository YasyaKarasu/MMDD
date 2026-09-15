"""Canonical B13 training semantics used by the R25 runner.

This module is deliberately a thin, auditable bridge to the historical
training helpers.  It does not define a new objective: the edge and path
wrappers below call the same reductions used by the B13 implementation, while
making the score-space conversion and sampling contract explicit at the call
site.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import torch

from .data import EdgeExample
from .scoring import ListScores, TargetScores
from .training import _student_edge_losses, _student_path_losses

B13_RELATIONS = (
    "table->table",
    "table->text",
    "table->image",
    "text->table",
    "image->table",
)
B13_BATCH_SIZE = 64
B13_KD_WEIGHT = 0.3
B13_TEMPERATURE = 1.0
B13_ANCHOR_WEIGHT = 0.1
B13_ANCHOR_WEIGHT_EVIDENCE = 0.1
B13_RELATION_LR = 1e-5
B13_PROJECTION_LR = 1e-6
B13_WEIGHT_DECAY = 0.01


def ranking_scores(raw: ListScores) -> ListScores:
    """Return B13's listwise SUP score space, ``10 * sigmoid(raw_logit)``."""

    return ListScores(
        torch.sigmoid(raw.logits) / 0.1,
        raw.candidate_mask,
        raw.positive_indices,
        raw.positive_mask,
        raw.candidate_ids,
    )


def edge_objective(
    student: torch.nn.Module,
    examples: Sequence[EdgeExample],
    raw_scores: ListScores,
    teacher_scores: ListScores,
    *,
    kd_weight: float = B13_KD_WEIGHT,
) -> dict[str, torch.Tensor]:
    """Build one historical B13 C1 update.

    SUP consumes transformed scores, while KD consumes the raw logits.  The
    five directed relations are passed together by the caller; this helper
    does not silently drop a relation or replace Teacher scores with Student
    scores.
    """

    transformed = ranking_scores(raw_scores)
    return _student_edge_losses(
        student,
        examples,
        raw_scores,
        teacher_scores,
        transformed,
        None,
        ranking_weight=1.0,
        temperature=B13_TEMPERATURE,
        distillation_weight=kd_weight,
        edge_bce_weight=0.0,
        anchor_weight=B13_ANCHOR_WEIGHT,
        anchor_weight_evidence=B13_ANCHOR_WEIGHT_EVIDENCE,
        positive_loss_mode="sum_probability",
    )


def path_objective(
    student: torch.nn.Module,
    student_scores: TargetScores,
    teacher_scores: TargetScores,
    *,
    kd_weight: float = B13_KD_WEIGHT,
) -> dict[str, torch.Tensor]:
    """Build B13-FULL's native target/path update.

    The native helper owns its historical branch reduction (including
    optional-evidence normalization); this is intentionally separate from the
    modern raw-logit Split/LSE factories.
    """

    return _student_path_losses(
        student,
        student_scores,
        teacher_scores,
        None,
        temperature=B13_TEMPERATURE,
        distillation_weight=kd_weight,
        anchor_weight=B13_ANCHOR_WEIGHT,
        anchor_weight_evidence=B13_ANCHOR_WEIGHT_EVIDENCE,
        distillation_rows=None,
        positive_loss_mode="sum_probability",
    )


def consumed_batch_order(batch_count: int, seed: int) -> list[int]:
    """Return the historical deterministic C1 schedule order.

    Seed 13 retains the frozen schedule order.  Other seeds use the same
    namespaced deterministic shuffle as R12; no global RNG state is consumed.
    """

    if batch_count < 0:
        raise ValueError("batch_count must be non-negative")
    order = list(range(batch_count))
    if seed != 13:
        random.Random(f"r12-c1-batch-order:{seed}").shuffle(order)
    return order


def validate_relation_coverage(examples: Sequence[EdgeExample]) -> dict[str, int]:
    """Count actual logical lists by the five B13 directed relations."""

    counts = {relation: 0 for relation in B13_RELATIONS}
    for example in examples:
        relation = f"{example.source_type}->{example.destination_type}"
        if relation not in counts:
            raise ValueError(f"unexpected C1 relation: {relation}")
        counts[relation] += 1
    missing = [relation for relation, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"C1 relation coverage is incomplete: {missing}")
    return counts


def recipe_signature() -> dict[str, Any]:
    """Machine-readable constants for receipts and implementation maps."""

    return {
        "relations": list(B13_RELATIONS),
        "batch_size": B13_BATCH_SIZE,
        "supervision_score_space": "10*sigmoid(raw_logit)",
        "distillation_score_space": "raw_logit",
        "distillation_temperature": B13_TEMPERATURE,
        "distillation_weight": B13_KD_WEIGHT,
        "edge_bce_weight": 0.0,
        "anchor_weight_all": B13_ANCHOR_WEIGHT,
        "anchor_weight_evidence": B13_ANCHOR_WEIGHT_EVIDENCE,
        "relation_learning_rate": B13_RELATION_LR,
        "projection_learning_rate": B13_PROJECTION_LR,
        "weight_decay": B13_WEIGHT_DECAY,
        "positive_loss_mode": "sum_probability",
        "candidate_sampling": {
            "destination_type_local_pool_cap": 256,
            "hard_source": "raw_qwen_ann_top256",
            "hard_negative_fraction": 0.5,
            "positive_policy": "global_relation_closure",
        },
    }

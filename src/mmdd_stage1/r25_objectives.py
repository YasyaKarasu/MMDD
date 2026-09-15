"""Explicit R25 C1/C2 objective factories.

The R25 contract distinguishes edge continuation from path training and keeps
the four Split arms as a true 2x2 ablation.  This module is intentionally
tensor-only so the semantics can be tested without loading a model or the
large feature lake.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .objectives import distillation_kl, listwise_cross_entropy
from .scoring import ListScores, TargetScores


@dataclass(frozen=True)
class R25ObjectiveConfig:
    """Fixed coefficients from the R25 execution contract."""

    kd_weight: float = 0.3
    uniform_weight: float = 0.3
    anchor_weight: float = 0.1
    anchor_weight_evidence: float = 0.1
    temperature: float = 1.0


def _optional_list_loss(scores: ListScores) -> tuple[torch.Tensor, int]:
    """Listwise CE with query-denominator semantics and an active-row count."""

    positive = scores.positive_mask
    if positive is None:
        positive = torch.zeros_like(scores.candidate_mask)
        rows = torch.arange(scores.logits.shape[0], device=scores.logits.device)
        valid_index = scores.positive_indices.clamp_min(0)
        positive[rows, valid_index] = True
    positive = positive & scores.candidate_mask
    active = positive.any(dim=-1) & (scores.candidate_mask & ~positive).any(dim=-1)
    if not active.any():
        return scores.logits.sum() * 0.0, 0
    # Compute on the selected rows but preserve the contract's explicit
    # denominator: inactive rows contribute zero to the batch mean.
    value = listwise_cross_entropy(
        scores.logits[active],
        scores.positive_indices[active],
        scores.candidate_mask[active],
        positive[active],
    )
    return value * (active.sum().to(value.dtype) / scores.logits.shape[0]), int(active.sum())


def _optional_kl(student: ListScores, teacher: ListScores | None, temperature: float) -> tuple[torch.Tensor, int]:
    if teacher is None:
        return student.logits.sum() * 0.0, 0
    mask = student.candidate_mask & teacher.candidate_mask
    positive = student.positive_mask
    if positive is None:
        positive = torch.zeros_like(mask)
        rows = torch.arange(mask.shape[0], device=mask.device)
        positive[rows, student.positive_indices.clamp_min(0)] = True
    positive = positive & mask
    active = mask.any(dim=-1) & positive.any(dim=-1) & (mask & ~positive).any(dim=-1)
    if not active.any():
        return student.logits.sum() * 0.0, 0
    value = distillation_kl(student.logits[active], teacher.logits[active], mask[active], temperature)
    return value * (active.sum().to(value.dtype) / student.logits.shape[0]), int(active.sum())


def uniform_kl(scores: ListScores, *, relation_keys: list[str] | tuple[str, ...]) -> tuple[torch.Tensor, int]:
    """KL(uniform valid candidates || student) for non-TT auxiliary edges.

    Unknown candidates remain valid negatives for this stabilization term, but
    no positive labels are introduced.  TT rows are deliberately excluded.
    """

    if len(relation_keys) != scores.logits.shape[0]:
        raise ValueError("relation_keys must align with score rows")
    active = torch.tensor([r != "table->table" for r in relation_keys], device=scores.logits.device)
    active &= scores.candidate_mask.any(dim=-1)
    if not active.any():
        return scores.logits.sum() * 0.0, 0
    mask = scores.candidate_mask[active]
    logits = scores.logits[active]
    uniform = torch.zeros_like(logits)
    value = distillation_kl(logits, uniform, mask, 1.0)
    return value * (active.sum().to(value.dtype) / scores.logits.shape[0]), int(active.sum())


def edge_continuation_loss(
    qe: ListScores,
    et: ListScores,
    *,
    config: R25ObjectiveConfig = R25ObjectiveConfig(),
) -> dict[str, torch.Tensor | int]:
    """True EDGE-CONT objective; no target/path loss is evaluated."""

    qe_loss, qe_active = _optional_list_loss(qe)
    et_loss, et_active = _optional_list_loss(et)
    anchor = qe.logits.sum() * 0.0
    return {
        "loss": qe_loss * 0.5 + et_loss * 0.5 + anchor,
        "qe_supervised_loss": qe_loss,
        "et_supervised_loss": et_loss,
        "path_supervised_loss": qe.logits.sum() * 0.0,
        "qe_active": qe_active,
        "et_active": et_active,
    }


def split_objective(
    scores: TargetScores,
    *,
    arm: str,
    teacher_qt: TargetScores | None = None,
    uniform_scores: ListScores | None = None,
    uniform_relations: list[str] | tuple[str, ...] | None = None,
    config: R25ObjectiveConfig = R25ObjectiveConfig(),
    anchor_loss: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int | str]:
    """Build one of the six modern C2 objectives or the LSE arm.

    ``teacher_qt`` must contain the same QT target logits for both D and E;
    callers should construct its evidence channel by masking the direct QT
    list, never by reading a legacy native-E cache.
    """

    allowed = {"SPLIT-SUP", "SPLIT-QTKD", "SPLIT-U", "SPLIT-UQTKD", "LSE-QTKD"}
    if arm not in allowed:
        raise ValueError(f"unsupported split arm: {arm}")
    if teacher_qt is not None:
        overlap = teacher_qt.direct.candidate_mask & teacher_qt.evidence.candidate_mask
        if overlap.any() and not torch.equal(
            teacher_qt.direct.logits[overlap], teacher_qt.evidence.logits[overlap]
        ):
            raise ValueError("R25 QT-KD requires identical T0(Q,T) values in D and E")
    d_sup, d_active = _optional_list_loss(scores.direct)
    e_sup, e_active = _optional_list_loss(scores.evidence)
    d_kd, d_kd_active = _optional_kl(scores.direct, None if teacher_qt is None else teacher_qt.direct, config.temperature)
    e_kd, e_kd_active = _optional_kl(scores.evidence, None if teacher_qt is None else teacher_qt.evidence, config.temperature)
    u = scores.direct.logits.sum() * 0.0
    u_active = 0
    if uniform_scores is not None:
        if uniform_relations is None:
            raise ValueError("uniform_relations is required with uniform_scores")
        u, u_active = uniform_kl(uniform_scores, relation_keys=uniform_relations)
    if arm in {"SPLIT-SUP", "SPLIT-U", "SPLIT-QTKD", "SPLIT-UQTKD"}:
        loss = d_sup + e_sup
        if arm in {"SPLIT-QTKD", "SPLIT-UQTKD"}:
            loss = loss + config.kd_weight * (d_kd + e_kd)
        if arm in {"SPLIT-U", "SPLIT-UQTKD"}:
            loss = loss + config.uniform_weight * u
        family = "split"
    else:
        # F is computed from the same Student D/E scores and the same QT
        # target ordering.  Invalid evidence is -inf, not a synthetic zero.
        evidence = scores.evidence.logits.masked_fill(~scores.evidence.candidate_mask, -torch.inf)
        fused = torch.logsumexp(torch.stack((scores.direct.logits, evidence), dim=-1), dim=-1)
        fused = torch.where(scores.evidence.candidate_mask.any(dim=-1, keepdim=True), fused, scores.direct.logits)
        fused_scores = ListScores(fused, scores.direct.candidate_mask, scores.direct.positive_indices, scores.direct.positive_mask)
        f_sup, f_active = _optional_list_loss(fused_scores)
        f_kd, f_kd_active = _optional_kl(fused_scores, None if teacher_qt is None else teacher_qt.direct, config.temperature)
        branch_units = d_active + e_active
        scale = branch_units / f_active if f_active else 0.0
        f_sup = f_sup * scale
        f_kd = f_kd * scale
        loss = f_sup + config.kd_weight * f_kd
        family = "lse"
        d_kd_active = e_kd_active = f_kd_active
    if anchor_loss is None:
        anchor_loss = scores.direct.logits.sum() * 0.0
    loss = loss + anchor_loss
    return {
        "loss": loss,
        "direct_supervised_loss": d_sup,
        "evidence_supervised_loss": e_sup,
        "direct_kd_loss": d_kd,
        "evidence_kd_loss": e_kd,
        "uniform_loss": u,
        "weighted_anchor_loss": anchor_loss,
        "direct_active": d_active,
        "evidence_active": e_active,
        "direct_kd_active": d_kd_active,
        "evidence_kd_active": e_kd_active,
        "uniform_active": u_active,
        "objective_family": family,
    }

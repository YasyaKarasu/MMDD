"""Stage-1 CQET loss functions and mathematical contracts."""
from __future__ import annotations

from typing import Optional, Sequence
import torch
import torch.nn.functional as F
from torch import Tensor


def rank_mass_loss(
    scores: Tensor,
    positive_mask: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """Multi-positive total-probability loss; missing supervision returns None.
    
    R(s; P, C) = LSE_{x in C}(s_x) - LSE_{p in P}(s_p).
    """
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if positive_mask.shape != scores.shape or positive_mask.dtype != torch.bool:
        raise ValueError("invalid positive_mask")
    if valid_mask is None:
        valid_mask = torch.ones_like(positive_mask)
    if valid_mask.shape != scores.shape or valid_mask.dtype != torch.bool:
        raise ValueError("invalid valid_mask")
    if torch.any(positive_mask & ~valid_mask):
        raise ValueError("positive outside valid candidate set")

    positives = positive_mask & valid_mask
    negatives = valid_mask & ~positive_mask
    if not bool(positives.any()) or not bool(negatives.any()):
        return None
    return torch.logsumexp(scores[valid_mask], 0) - torch.logsumexp(
        scores[positives], 0
    )


def positive_average_pair_loss(
    positive_scores: Tensor,
    competitor_scores: Tensor,
) -> Optional[Tensor]:
    """Same-modality support loss, without f0 reference or detached current score."""
    if positive_scores.ndim != 1 or competitor_scores.ndim != 1:
        raise ValueError("one-dimensional scores required")
    if positive_scores.numel() == 0 or competitor_scores.numel() == 0:
        return None

    differences = competitor_scores[None, :] - positive_scores[:, None]
    zero = torch.zeros(
        (positive_scores.numel(), 1),
        dtype=positive_scores.dtype,
        device=positive_scores.device,
    )
    return torch.logsumexp(torch.cat((zero, differences), dim=1), dim=1).mean()


def _validate_path_inputs(
    f0: Tensor,
    path_scores: Tensor,
    path_target_index: Tensor,
) -> None:
    if f0.ndim != 1 or path_scores.ndim != 1:
        raise ValueError("one-dimensional score arrays required")
    if path_target_index.shape != path_scores.shape:
        raise ValueError("path index/score length mismatch")
    if path_target_index.dtype != torch.long:
        raise ValueError("path_target_index must be int64")
    if path_target_index.numel():
        if int(path_target_index.min()) < 0:
            raise ValueError("negative target index")
        if int(path_target_index.max()) >= f0.numel():
            raise ValueError("target index outside candidate list")



def _path_slots(
    f0: Tensor,
    path_scores: Tensor,
    path_target_index: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return (path-only slots, per-target counts) independent of input order."""
    _validate_path_inputs(f0, path_scores, path_target_index)
    n_targets = f0.numel()
    counts = torch.bincount(path_target_index, minlength=n_targets)
    if path_scores.numel() == 0:
        return f0.new_empty((n_targets, 0)), counts

    max_count = int(counts.max().item())
    sorted_target_idx, perm = torch.sort(path_target_index, stable=True)
    sorted_scores = path_scores[perm]
    starts = torch.cumsum(counts, dim=0) - counts
    within_group_slot = torch.arange(sorted_target_idx.numel(), device=f0.device) - starts[sorted_target_idx]

    slots = f0.new_full((n_targets, max_count), float("-inf"))
    slots[sorted_target_idx, within_group_slot] = sorted_scores
    return slots, counts


def aggregate_cqet(
    f0: Tensor,
    path_scores: Tensor,
    path_target_index: Tensor,
) -> Tensor:
    """V4.1 CQET: nonempty bags use path-only LME; empty bags use f0."""
    slots, counts = _path_slots(f0, path_scores, path_target_index)
    if slots.shape[1] == 0:
        return f0
    nonempty = counts > 0
    path_lme = torch.logsumexp(slots, dim=1) - counts.clamp_min(1).to(f0.dtype).log()
    return torch.where(nonempty, path_lme, f0)


def aggregate_corrected_lse(
    f0: Tensor,
    path_scores: Tensor,
    path_target_index: Tensor,
) -> Tensor:
    """Corrected-LSE control: every target aggregates f0 together with paths."""
    slots, _counts = _path_slots(f0, path_scores, path_target_index)
    if slots.shape[1] == 0:
        return f0
    return torch.logsumexp(torch.cat((f0[:, None], slots), dim=1), dim=1)


def aggregate_paths(
    f0: Tensor,
    path_scores: Tensor,
    path_target_index: Tensor,
) -> Tensor:
    """Compatibility name for the corrected-LSE control, never the CQET arm."""
    return aggregate_corrected_lse(f0, path_scores, path_target_index)


def global_features(
    left: Tensor,
    right: Tensor,
    pair_embedding: Tensor,
    evidence: Optional[Tensor] = None,
    evidence_type_embedding: Optional[Tensor] = None,
) -> Tensor:
    """Return exactly 11h features."""
    if left.shape != right.shape or left.shape != pair_embedding.shape:
        raise ValueError("left/right/type-pair shapes must match")

    base = torch.cat(
        (left, right, left * right, (left - right).abs(), pair_embedding),
        dim=-1,
    )

    if evidence is None:
        if evidence_type_embedding is not None:
            raise ValueError("no evidence type embedding allowed for empty E")
        extension = torch.zeros(
            (*left.shape[:-1], 6 * left.shape[-1]),
            dtype=left.dtype,
            device=left.device,
        )
    else:
        if evidence.shape != left.shape:
            raise ValueError("invalid evidence shape")
        if evidence_type_embedding is None:
            raise ValueError("nonempty evidence requires type embedding")
        if evidence_type_embedding.shape != left.shape:
            raise ValueError("invalid evidence type embedding")
        extension = torch.cat(
            (
                evidence,
                left * evidence,
                (left - evidence).abs(),
                evidence * right,
                (evidence - right).abs(),
                evidence_type_embedding,
            ),
            dim=-1,
        )
    return torch.cat((base, extension), dim=-1)


def hierarchical_support_mean(
    targets: Sequence[Sequence[Tensor]],
) -> Optional[Tensor]:
    """
    Each target contains its valid modality scalar losses.
    A modality scalar must already average its positive witnesses.
    """
    target_means = [
        torch.stack(list(modality_losses)).mean()
        for modality_losses in targets
        if len(modality_losses) > 0
    ]
    return torch.stack(target_means).mean() if target_means else None


def hierarchical_relation_mean(
    relations: Sequence[Sequence[Tensor]],
) -> Optional[Tensor]:
    """Mean valid lists within each relation, then mean active relations."""
    relation_means = [torch.stack(list(losses)).mean() for losses in relations if losses]
    return torch.stack(relation_means).mean() if relation_means else None


def list_kl_divergence(
    student_scores: Tensor,
    teacher_scores: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """KL divergence D_KL(softmax(teacher_scores) || softmax(student_scores)).

    Both inputs are final logits. Callers apply the asymmetric scaling themselves:
    student logits are already multiplied by ``logit_scale`` and teacher logits are
    divided by ``kd_temperature`` (see ``train.train_student_c2``).
    """
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("student and teacher scores must match in shape")
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != student_scores.shape:
            raise ValueError("invalid valid_mask for KL")
        if not bool(valid_mask.any()):
            return None
        student_scores = student_scores[valid_mask]
        teacher_scores = teacher_scores[valid_mask]

    if student_scores.numel() <= 1:
        return None

    if not student_scores.requires_grad:
        raise RuntimeError("student KL scores must remain grad-enabled")
    t_log_p = F.log_softmax(teacher_scores.detach(), dim=0)
    s_log_p = F.log_softmax(student_scores, dim=0)
    # KL = sum(p_T * (log_p_T - log_p_S))
    p_t = torch.exp(t_log_p)
    return (p_t * (t_log_p - s_log_p)).sum()

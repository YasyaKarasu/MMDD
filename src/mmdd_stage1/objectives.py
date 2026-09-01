"""Path aggregation and listwise objectives for Stage-1 training."""

from __future__ import annotations

import torch
from torch import nn


PATH_AGGREGATIONS = {
    "comb_mnz",
    "logsumexp",
    "max",
    "power_mean",
    "softmax_weighted_mean",
    "topk_mean",
    "topk_sum",
}


class PathAggregator(nn.Module):
    """Aggregate Q->evidence->target paths within the evidence channel."""

    def __init__(
        self,
        evidence_aggregation: str = "logsumexp",
        top_k: int = 4,
        *,
        temperature: float = 1.0,
        power: float = 2.0,
    ) -> None:
        super().__init__()
        if evidence_aggregation not in PATH_AGGREGATIONS:
            choices = ", ".join(sorted(PATH_AGGREGATIONS))
            raise ValueError(f"evidence_aggregation must be one of: {choices}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if power <= 0:
            raise ValueError("power must be positive")
        self.evidence_aggregation = evidence_aggregation
        self.top_k = top_k
        self.temperature = float(temperature)
        self.power = float(power)

    def forward(
        self,
        query_evidence_scores: torch.Tensor,
        evidence_target_scores: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_evidence_scores.shape != evidence_target_scores.shape:
            raise ValueError("The two evidence-edge score tensors must have equal shapes")
        if evidence_mask.shape != query_evidence_scores.shape:
            raise ValueError("evidence_mask must match evidence-edge scores")
        if query_evidence_scores.shape[-1] == 0:
            return query_evidence_scores.new_zeros(query_evidence_scores.shape[:2])

        path_scores = query_evidence_scores + evidence_target_scores
        masked_paths = path_scores.masked_fill(~evidence_mask, -torch.inf)
        has_evidence = evidence_mask.any(dim=-1)

        if self.evidence_aggregation == "logsumexp":
            safe_paths = masked_paths.masked_fill(~has_evidence.unsqueeze(-1), 0.0)
            evidence_scores = torch.logsumexp(safe_paths, dim=-1)
        elif self.evidence_aggregation == "max":
            evidence_scores = masked_paths.max(dim=-1).values
        elif self.evidence_aggregation in {"topk_mean", "topk_sum"}:
            count = min(self.top_k, path_scores.shape[-1])
            values = torch.topk(masked_paths, k=count, dim=-1).values
            valid = torch.isfinite(values)
            evidence_scores = values.masked_fill(~valid, 0.0).sum(dim=-1)
            if self.evidence_aggregation == "topk_mean":
                evidence_scores = evidence_scores / valid.sum(dim=-1).clamp_min(1)
        elif self.evidence_aggregation == "softmax_weighted_mean":
            safe_paths = masked_paths.masked_fill(~has_evidence.unsqueeze(-1), 0.0)
            weights = torch.softmax(safe_paths / self.temperature, dim=-1)
            weights = weights.masked_fill(~evidence_mask, 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).tiny
            )
            evidence_scores = (weights * safe_paths).sum(dim=-1)
        elif self.evidence_aggregation == "power_mean":
            safe_paths = masked_paths.masked_fill(~evidence_mask, torch.inf)
            minimum = safe_paths.min(dim=-1).values
            minimum = torch.where(has_evidence, minimum, torch.zeros_like(minimum))
            shifted = (path_scores - minimum.unsqueeze(-1)).clamp_min(0.0)
            shifted = shifted.masked_fill(~evidence_mask, 0.0)
            count = evidence_mask.sum(dim=-1).clamp_min(1)
            evidence_scores = (
                shifted.pow(self.power).sum(dim=-1) / count
            ).pow(1.0 / self.power) + minimum
        else:
            valid_paths = path_scores.masked_fill(~evidence_mask, 0.0)
            evidence_scores = (
                valid_paths.sum(dim=-1) * evidence_mask.sum(dim=-1)
            )

        return torch.where(has_evidence, evidence_scores, torch.zeros_like(evidence_scores))


def positive_indices_to_mask(
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Expand one positive index per list into a boolean candidate mask."""

    if positive_indices.shape != (candidate_mask.shape[0],):
        raise ValueError("positive_indices must have shape [batch]")
    rows = torch.arange(candidate_mask.shape[0], device=candidate_mask.device)
    if not torch.all(candidate_mask[rows, positive_indices]):
        raise ValueError("Every positive index must identify a valid candidate")
    positive_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    positive_mask[rows, positive_indices] = True
    return positive_mask


def _resolve_positive_mask(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor | None,
) -> torch.Tensor:
    if logits.shape != candidate_mask.shape:
        raise ValueError("candidate_mask must match logits")
    if positive_mask is None:
        return positive_indices_to_mask(positive_indices, candidate_mask)
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must match logits")
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if torch.any(positive_mask & ~candidate_mask):
        raise ValueError("positive_mask must identify only valid candidates")
    return positive_mask


def listwise_cross_entropy(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log probability assigned to all positives in each list."""

    positive_mask = _resolve_positive_mask(
        logits, positive_indices, candidate_mask, positive_mask
    )
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("Every candidate list must contain at least one positive")
    normalizers = torch.logsumexp(
        logits.masked_fill(~candidate_mask, -torch.inf), dim=-1
    )
    positive_mass = torch.logsumexp(
        logits.masked_fill(~positive_mask, -torch.inf), dim=-1
    )
    return (normalizers - positive_mass).mean()


def distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(P_teacher || P_student) over each candidate list."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape or student_logits.shape != candidate_mask.shape:
        raise ValueError("Student, Teacher, and candidate mask tensors must have equal shapes")
    masked_student = student_logits.masked_fill(~candidate_mask, -torch.inf) / temperature
    masked_teacher = teacher_logits.detach().masked_fill(~candidate_mask, -torch.inf) / temperature
    teacher_probabilities = torch.softmax(masked_teacher, dim=-1)
    student_log_probabilities = torch.log_softmax(masked_student, dim=-1)
    teacher_log_probabilities = torch.log_softmax(masked_teacher, dim=-1)
    per_candidate = teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)
    per_candidate = per_candidate.masked_fill(~candidate_mask, 0.0)
    return per_candidate.sum(dim=-1).mean() * temperature**2


def _usable_list_rows(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_mask = _resolve_positive_mask(
        logits, positive_indices, candidate_mask, positive_mask
    )
    has_positive = positive_mask.any(dim=-1)
    has_negative = (candidate_mask & ~positive_mask).any(dim=-1)
    return has_positive & has_negative


def optional_listwise_cross_entropy(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Listwise CE over rows with a positive and at least one negative."""

    usable = _usable_list_rows(
        logits, positive_indices, candidate_mask, positive_mask
    )
    if not usable.any().item():
        return logits.sum() * 0.0
    return listwise_cross_entropy(
        logits[usable],
        positive_indices[usable],
        candidate_mask[usable],
        None if positive_mask is None else positive_mask[usable],
    )


def optional_distillation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Listwise distillation over rows with a usable supervised channel list."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and Teacher logits must have equal shapes")
    usable = _usable_list_rows(
        student_logits, positive_indices, candidate_mask, positive_mask
    )
    if not usable.any().item():
        return student_logits.sum() * 0.0
    return distillation_kl(
        student_logits[usable],
        teacher_logits[usable],
        candidate_mask[usable],
        temperature,
    )

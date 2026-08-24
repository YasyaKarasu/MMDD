"""Path aggregation and listwise objectives for Stage-1 training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PathAggregator(nn.Module):
    """Combine direct and Q->evidence->target paths into target scores."""

    def __init__(self, evidence_aggregation: str = "logsumexp", top_k: int = 4) -> None:
        super().__init__()
        if evidence_aggregation not in {"logsumexp", "topk_mean", "topk_sum"}:
            raise ValueError("evidence_aggregation must be logsumexp, topk_mean, or topk_sum")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.evidence_aggregation = evidence_aggregation
        self.top_k = top_k

    def forward(
        self,
        direct_scores: torch.Tensor,
        query_evidence_scores: torch.Tensor,
        evidence_target_scores: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_evidence_scores.shape != evidence_target_scores.shape:
            raise ValueError("The two evidence-edge score tensors must have equal shapes")
        if evidence_mask.shape != query_evidence_scores.shape:
            raise ValueError("evidence_mask must match evidence-edge scores")
        if direct_scores.shape != query_evidence_scores.shape[:2]:
            raise ValueError("direct_scores must match the batch and target dimensions")
        if query_evidence_scores.shape[-1] == 0:
            return direct_scores

        path_scores = query_evidence_scores + evidence_target_scores
        masked_paths = path_scores.masked_fill(~evidence_mask, -torch.inf)
        has_evidence = evidence_mask.any(dim=-1)

        if self.evidence_aggregation == "logsumexp":
            safe_paths = masked_paths.masked_fill(~has_evidence.unsqueeze(-1), 0.0)
            evidence_scores = torch.logsumexp(safe_paths, dim=-1)
        else:
            count = min(self.top_k, path_scores.shape[-1])
            values = torch.topk(masked_paths, k=count, dim=-1).values
            valid = torch.isfinite(values)
            evidence_scores = values.masked_fill(~valid, 0.0).sum(dim=-1)
            if self.evidence_aggregation == "topk_mean":
                evidence_scores = evidence_scores / valid.sum(dim=-1).clamp_min(1)

        return torch.where(has_evidence, torch.logaddexp(direct_scores, evidence_scores), direct_scores)


def listwise_cross_entropy(
    logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy for one positive candidate in each variable-length list."""

    if logits.shape != candidate_mask.shape:
        raise ValueError("candidate_mask must match logits")
    if positive_indices.shape != (logits.shape[0],):
        raise ValueError("positive_indices must have shape [batch]")
    rows = torch.arange(logits.shape[0], device=logits.device)
    if not torch.all(candidate_mask[rows, positive_indices]):
        raise ValueError("Every positive index must identify a valid candidate")
    return F.cross_entropy(logits.masked_fill(~candidate_mask, -torch.inf), positive_indices)


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


def student_path_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    positive_indices: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
    distillation_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supervised = listwise_cross_entropy(student_logits, positive_indices, candidate_mask)
    distillation = distillation_kl(student_logits, teacher_logits, candidate_mask, temperature)
    total = supervised + distillation_weight * distillation
    return total, supervised, distillation

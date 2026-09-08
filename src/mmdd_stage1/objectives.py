"""Path aggregation and listwise objectives for Stage-1 training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .row_support import (
    IsotonicModel,
    embedding_content_key,
    greedy_row_bundle_tensor,
    load_evidence_content_keys,
    load_row_support_models,
)


PATH_AGGREGATIONS = {
    "comb_mnz",
    "logmeanexp",
    "logsumexp",
    "max",
    "power_mean",
    "softmax_weighted_mean",
    "topk_logmeanexp",
    "topk_logsumexp",
    "topk_mean",
    "topk_sum",
    "fixed_power_mean",
    "greedy_row_support",
}
RAW_EDGE_SCORE_PATH_AGGREGATIONS = PATH_AGGREGATIONS - {
    "fixed_power_mean",
    "greedy_row_support",
}
PATH_COMBINATIONS = {"sum", "min", "product"}
POSITIVE_LOSS_MODES = {"mean_log_probability", "sum_probability"}


class PathAggregator(nn.Module):
    """Aggregate Q->evidence->target paths within the evidence channel."""

    def __init__(
        self,
        evidence_aggregation: str = "logsumexp",
        top_k: int = 4,
        *,
        temperature: float = 1.0,
        power: float = 2.0,
        path_combination: str = "sum",
        threshold: float = 0.0,
        target_temperature: float = 1.0,
        row_support_model: str | Path | None = None,
        row_support_model_sha256: str | None = None,
        row_support_top_l: int = 20,
        evidence_content_keys: str | Path | None = None,
        evidence_content_keys_sha256: str | None = None,
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
        if path_combination not in PATH_COMBINATIONS:
            choices = ", ".join(sorted(PATH_COMBINATIONS))
            raise ValueError(f"path_combination must be one of: {choices}")
        if not 0 <= threshold < 1:
            raise ValueError("threshold must be in [0, 1)")
        if target_temperature <= 0:
            raise ValueError("target_temperature must be positive")
        if evidence_aggregation == "fixed_power_mean" and power < 1:
            raise ValueError("fixed_power_mean power must be at least one")
        if row_support_top_l <= 0:
            raise ValueError("row_support_top_l must be positive")
        self.evidence_aggregation = evidence_aggregation
        self.top_k = top_k
        self.temperature = float(temperature)
        self.power = float(power)
        self.path_combination = path_combination
        self.threshold = float(threshold)
        self.target_temperature = float(target_temperature)
        self.row_support_top_l = int(row_support_top_l)
        self.row_support_model = (
            str(Path(row_support_model).resolve())
            if row_support_model is not None
            else None
        )
        self.row_support_models: dict[str, IsotonicModel] = {}
        self.row_support_model_sha256: str | None = None
        if self.row_support_model is not None:
            (
                self.row_support_models,
                self.row_support_model_sha256,
            ) = load_row_support_models(
                Path(self.row_support_model),
                expected_sha256=row_support_model_sha256,
            )
        elif row_support_model_sha256 is not None:
            raise ValueError("row_support_model_sha256 requires row_support_model")
        self.evidence_content_keys = (
            str(Path(evidence_content_keys).resolve())
            if evidence_content_keys is not None
            else None
        )
        self.evidence_content_key_by_id: dict[str, str] = {}
        self.evidence_content_keys_sha256: str | None = None
        if self.evidence_content_keys is not None:
            (
                self.evidence_content_key_by_id,
                self.evidence_content_keys_sha256,
            ) = load_evidence_content_keys(
                Path(self.evidence_content_keys),
                expected_sha256=evidence_content_keys_sha256,
            )
        elif evidence_content_keys_sha256 is not None:
            raise ValueError(
                "evidence_content_keys_sha256 requires evidence_content_keys"
            )

    def config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "evidence_aggregation": self.evidence_aggregation,
            "evidence_top_k": self.top_k,
            "evidence_temperature": self.temperature,
            "evidence_power": self.power,
            "path_combination": self.path_combination,
            "evidence_threshold": self.threshold,
            "evidence_target_temperature": self.target_temperature,
        }
        if self.evidence_aggregation == "greedy_row_support":
            result.update(
                {
                    "row_support_model": self.row_support_model,
                    "row_support_model_sha256": self.row_support_model_sha256,
                    "row_support_top_l": self.row_support_top_l,
                    "evidence_content_keys": self.evidence_content_keys,
                    "evidence_content_keys_sha256": (
                        self.evidence_content_keys_sha256
                    ),
                }
            )
        return result

    def content_key(self, object_id: str, embedding: torch.Tensor) -> str:
        """Return an exact-content key, with legacy embedding fallback."""

        if self.evidence_content_key_by_id:
            try:
                return self.evidence_content_key_by_id[object_id]
            except KeyError as exc:
                raise KeyError(
                    f"Evidence content-key manifest has no object {object_id!r}"
                ) from exc
        return embedding_content_key(embedding)

    def forward(
        self,
        query_evidence_scores: torch.Tensor,
        evidence_target_scores: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        row_support: torch.Tensor | None = None,
        content_groups: torch.Tensor | None = None,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query_evidence_scores.shape != evidence_target_scores.shape:
            raise ValueError("The two evidence-edge score tensors must have equal shapes")
        if evidence_mask.shape != query_evidence_scores.shape:
            raise ValueError("evidence_mask must match evidence-edge scores")
        if query_evidence_scores.shape[-1] == 0:
            return query_evidence_scores.new_zeros(query_evidence_scores.shape[:2])
        if self.evidence_aggregation == "greedy_row_support":
            if row_support is None:
                raise ValueError("greedy_row_support requires row_support")
            expected = (*query_evidence_scores.shape, row_support.shape[-1])
            if row_support.shape != expected:
                raise ValueError(
                    "row_support must extend the evidence score shape with rows"
                )
            if content_groups is not None and content_groups.shape != evidence_mask.shape:
                raise ValueError("content_groups must match evidence scores")
            if row_mask is not None and row_mask.shape != row_support.shape[:-2] + (
                row_support.shape[-1],
            ):
                raise ValueError("row_mask must match batch/candidate and row dimensions")

        if self.path_combination == "sum":
            path_scores = query_evidence_scores + evidence_target_scores
        elif self.path_combination == "min":
            path_scores = torch.minimum(
                query_evidence_scores, evidence_target_scores
            )
        else:
            path_scores = query_evidence_scores * evidence_target_scores
        masked_paths = path_scores.masked_fill(~evidence_mask, -torch.inf)
        has_evidence = evidence_mask.any(dim=-1)

        if self.evidence_aggregation == "greedy_row_support":
            flat_scores = path_scores.reshape(-1, path_scores.shape[-1])
            flat_support = row_support.reshape(
                -1, row_support.shape[-2], row_support.shape[-1]
            )
            flat_mask = evidence_mask.reshape(-1, evidence_mask.shape[-1])
            flat_groups = (
                None
                if content_groups is None
                else content_groups.reshape(-1, content_groups.shape[-1])
            )
            flat_row_mask = (
                None
                if row_mask is None
                else row_mask.reshape(-1, row_mask.shape[-1])
            )
            evidence_scores = torch.stack(
                [
                    greedy_row_bundle_tensor(
                        flat_scores[index],
                        flat_support[index],
                        flat_mask[index],
                        budget=self.top_k,
                        top_l=self.row_support_top_l,
                        threshold=self.threshold,
                        content_groups=(
                            None if flat_groups is None else flat_groups[index]
                        ),
                        row_mask=(
                            None if flat_row_mask is None else flat_row_mask[index]
                        ),
                    )[1]
                    for index in range(flat_scores.shape[0])
                ]
            ).reshape(path_scores.shape[:-1])
        elif self.evidence_aggregation in {
            "logmeanexp",
            "logsumexp",
            "topk_logmeanexp",
            "topk_logsumexp",
        }:
            selected_paths = masked_paths
            selected_mask = evidence_mask
            if self.evidence_aggregation.startswith("topk_"):
                count = min(self.top_k, path_scores.shape[-1])
                selected_paths, indices = torch.topk(
                    masked_paths, k=count, dim=-1
                )
                selected_mask = evidence_mask.gather(-1, indices)
            selected_has_evidence = selected_mask.any(dim=-1)
            scaled_paths = selected_paths / self.temperature
            safe_paths = scaled_paths.masked_fill(
                ~selected_has_evidence.unsqueeze(-1), 0.0
            )
            evidence_scores = self.temperature * torch.logsumexp(
                safe_paths, dim=-1
            )
            if "logmeanexp" in self.evidence_aggregation:
                selected_count = selected_mask.sum(dim=-1).clamp_min(1)
                evidence_scores = evidence_scores - self.temperature * torch.log(
                    selected_count.to(evidence_scores.dtype)
                )
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
        elif self.evidence_aggregation == "fixed_power_mean":
            count = min(self.top_k, path_scores.shape[-1])
            values, indices = torch.topk(masked_paths, k=count, dim=-1)
            valid = evidence_mask.gather(-1, indices)
            values = values.masked_fill(~valid, 0.0)
            strengths = ((values - self.threshold) / (1.0 - self.threshold)).clamp(
                min=0.0, max=1.0
            )
            mean_power = strengths.pow(self.power).sum(dim=-1) / self.top_k
            positive = mean_power > 0
            safe_mean = mean_power.clamp_min(torch.finfo(mean_power.dtype).tiny)
            powered = safe_mean.pow(1.0 / self.power)
            evidence_scores = torch.where(positive, powered, mean_power)
        else:
            valid_paths = path_scores.masked_fill(~evidence_mask, 0.0)
            evidence_scores = (
                valid_paths.sum(dim=-1) * evidence_mask.sum(dim=-1)
            )

        evidence_scores = evidence_scores / self.target_temperature
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
    *,
    positive_loss_mode: str = "sum_probability",
) -> torch.Tensor:
    """Return a multi-positive listwise cross-entropy objective."""

    if positive_loss_mode not in POSITIVE_LOSS_MODES:
        choices = ", ".join(sorted(POSITIVE_LOSS_MODES))
        raise ValueError(f"positive_loss_mode must be one of: {choices}")

    positive_mask = _resolve_positive_mask(
        logits, positive_indices, candidate_mask, positive_mask
    )
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("Every candidate list must contain at least one positive")
    normalizers = torch.logsumexp(
        logits.masked_fill(~candidate_mask, -torch.inf), dim=-1
    )
    if positive_loss_mode == "sum_probability":
        positive_scores = torch.logsumexp(
            logits.masked_fill(~positive_mask, -torch.inf), dim=-1
        )
    else:
        positive_scores = (
            logits.masked_fill(~positive_mask, 0.0).sum(dim=-1)
            / positive_mask.sum(dim=-1)
        )
    return (normalizers - positive_scores).mean()


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
    *,
    positive_loss_mode: str = "sum_probability",
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
        positive_loss_mode=positive_loss_mode,
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


def relation_macro_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    confirmed_mask: torch.Tensor,
    relation_keys: Sequence[str],
) -> torch.Tensor:
    """Average confirmed-label BCE within lists, then equally across relations."""

    if logits.shape != labels.shape or logits.shape != confirmed_mask.shape:
        raise ValueError("Logits, labels, and confirmed mask must have equal shapes")
    if len(relation_keys) != logits.shape[0]:
        raise ValueError("relation_keys must contain one value per candidate list")
    if torch.any((labels[confirmed_mask] != 0) & (labels[confirmed_mask] != 1)):
        raise ValueError("Confirmed BCE labels must be binary")

    per_candidate = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(dtype=logits.dtype),
        reduction="none",
    )
    per_relation: dict[str, list[torch.Tensor]] = {}
    for row, relation_key in enumerate(relation_keys):
        selected = confirmed_mask[row]
        if selected.any().item():
            per_relation.setdefault(relation_key, []).append(
                per_candidate[row, selected].mean()
            )
    if not per_relation:
        return logits.sum() * 0.0
    relation_losses = [
        torch.stack(list_losses).mean()
        for _relation_key, list_losses in sorted(per_relation.items())
    ]
    return torch.stack(relation_losses).mean()

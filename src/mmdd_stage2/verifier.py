"""Core scoring and evidence-localization algorithms for Stage 2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from mmdd_dataset.utils import values_match


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    evidence_type: str
    path_score: float
    weight: float


@dataclass(frozen=True)
class EvidenceBundle:
    target_id: str
    retrieval_score: float
    evidence: tuple[EvidenceRef, ...]
    paths: tuple[dict[str, Any], ...]

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)


@dataclass(frozen=True)
class RegionOfInterest:
    box: tuple[float, float, float, float]
    relevance: float


@dataclass(frozen=True)
class SemanticJoinability:
    joinable: bool
    coverage: float
    mean_similarity: float
    similarities: tuple[float, ...]
    target_indices: tuple[int, ...]


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def build_evidence_bundles(
    retrieval_results: Sequence[dict[str, Any]],
    *,
    top_k_evidence: int,
) -> list[EvidenceBundle]:
    """Group one-hop paths by target and weight unique evidence with LogSumExp."""

    if top_k_evidence < 0:
        raise ValueError("top_k_evidence must be non-negative")
    bundles = []
    for result in retrieval_results:
        paths = [
            path
            for path in result.get("paths", [])
            if path.get("kind") == "evidence" and path.get("evidence_id") is not None
        ]
        if not paths:
            continue
        paths.sort(key=lambda path: float(path["path_score"]), reverse=True)
        scores_by_evidence: dict[str, list[float]] = {}
        type_by_evidence: dict[str, str] = {}
        for path in paths:
            evidence_id = str(path["evidence_id"])
            scores_by_evidence.setdefault(evidence_id, []).append(float(path["path_score"]))
            type_by_evidence.setdefault(evidence_id, str(path.get("evidence_type", "")))
        ranked = sorted(
            (
                (evidence_id, type_by_evidence[evidence_id], _logsumexp(scores))
                for evidence_id, scores in scores_by_evidence.items()
            ),
            key=lambda item: item[2],
            reverse=True,
        )[:top_k_evidence]
        if not ranked:
            continue
        weights = torch.softmax(torch.tensor([item[2] for item in ranked]), dim=0).tolist()
        bundles.append(
            EvidenceBundle(
                target_id=str(result["target_id"]),
                retrieval_score=_logsumexp([float(path["path_score"]) for path in paths]),
                evidence=tuple(
                    EvidenceRef(evidence_id, evidence_type, score, float(weight))
                    for (evidence_id, evidence_type, score), weight in zip(ranked, weights)
                ),
                paths=tuple(paths),
            )
        )
    return bundles


class CandidateColumnScorer(nn.Module):
    """RATA reader head over each candidate column's boundary states."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(hidden_dim * 2, 1)

    def forward(self, open_states: torch.Tensor, close_states: torch.Tensor) -> torch.Tensor:
        if open_states.shape != close_states.shape:
            raise ValueError("Opening and closing marker states must have equal shapes")
        return self.weight(torch.cat([open_states, close_states], dim=-1)).squeeze(-1)


def joint_candidate_probabilities(
    retrieval_scores: torch.Tensor,
    column_logits: torch.Tensor,
    target_mask: torch.Tensor,
    column_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute softmax(r_T) * rho_(T,c) from the verifier formulation."""

    if retrieval_scores.shape != target_mask.shape:
        raise ValueError("target_mask must match retrieval_scores")
    if column_logits.shape != column_mask.shape or column_logits.shape[:2] != retrieval_scores.shape:
        raise ValueError("Column tensors must have shape [batch, targets, columns]")
    if not torch.all(target_mask.any(dim=-1)):
        raise ValueError("Every batch item must have at least one valid target")
    if not torch.all(column_mask.any(dim=-1) | ~target_mask):
        raise ValueError("Every valid target must have at least one candidate column")
    table_probabilities = torch.softmax(retrieval_scores.masked_fill(~target_mask, -torch.inf), dim=-1)
    column_probabilities = torch.softmax(column_logits.masked_fill(~column_mask, -torch.inf), dim=-1)
    column_probabilities = column_probabilities.masked_fill(~column_mask, 0.0)
    joint = table_probabilities.unsqueeze(-1) * column_probabilities
    joint = joint.masked_fill(~column_mask | ~target_mask.unsqueeze(-1), 0.0)
    return table_probabilities, column_probabilities, joint


def build_row_evidence_query(
    row: Mapping[str, Any],
    *,
    entity_column: str,
    attribute_name: str,
) -> dict[str, str]:
    if entity_column not in row:
        raise ValueError(f"Row has no entity column {entity_column!r}")
    serialized_row = " | ".join(f"{name}={value}" for name, value in row.items() if value is not None)
    return {
        "entity_anchor": str(row[entity_column]),
        "attribute_name": attribute_name,
        "serialized_row": serialized_row,
    }


def _relevance(query_states: torch.Tensor, evidence_states: torch.Tensor) -> torch.Tensor:
    if query_states.ndim != 2 or evidence_states.ndim != 2:
        raise ValueError("Query and evidence states must have shape [tokens, hidden]")
    if query_states.shape[1] != evidence_states.shape[1] or query_states.shape[0] == 0 or evidence_states.shape[0] == 0:
        raise ValueError("Hidden dimensions must match and token sequences cannot be empty")
    query = F.normalize(query_states.mean(dim=0).float(), dim=0)
    evidence = F.normalize(evidence_states.float(), dim=-1)
    return torch.softmax(evidence @ query, dim=0)


def joint_relevance(
    entity_states: torch.Tensor,
    attribute_states: torch.Tensor,
    evidence_states: torch.Tensor,
) -> torch.Tensor:
    """Multiply normalized entity and attribute relevance maps for one layer."""

    combined = _relevance(entity_states, evidence_states) * _relevance(attribute_states, evidence_states)
    return combined / combined.sum().clamp_min(torch.finfo(combined.dtype).tiny)


def focus_relevance(
    value_layers: Sequence[torch.Tensor],
    entity_indices: torch.Tensor,
    attribute_indices: torch.Tensor,
    evidence_indices: torch.Tensor,
) -> torch.Tensor:
    """Aggregate FOCUS-style value-feature maps over later MLLM layers."""

    if not value_layers:
        raise ValueError("FOCUS requires at least one value-feature layer")
    maps = [
        joint_relevance(
            layer.index_select(0, entity_indices),
            layer.index_select(0, attribute_indices),
            layer.index_select(0, evidence_indices),
        )
        for layer in value_layers
    ]
    relevance = torch.stack(maps).mean(dim=0)
    return relevance / relevance.sum().clamp_min(torch.finfo(relevance.dtype).tiny)


def gaussian_smooth(relevance_map: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if relevance_map.ndim != 2 or relevance_map.numel() == 0:
        raise ValueError("relevance_map must be a non-empty matrix")
    if sigma <= 0:
        return relevance_map
    radius = max(1, math.ceil(2 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=relevance_map.device, dtype=relevance_map.dtype)
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    kernel = torch.outer(kernel, kernel)
    kernel /= kernel.sum()
    smoothed = F.conv2d(
        relevance_map.reshape(1, 1, *relevance_map.shape),
        kernel.reshape(1, 1, *kernel.shape),
        padding=radius,
    ).reshape_as(relevance_map)
    return smoothed / smoothed.sum().clamp_min(torch.finfo(smoothed.dtype).tiny)


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def propose_image_regions(
    relevance_map: torch.Tensor,
    image_size: tuple[int, int],
    *,
    anchors: int = 8,
    min_anchor_distance: float = 2.0,
    min_side: int = 2,
    max_side: int | None = None,
    expansion_threshold: float = 1.5,
    nms_threshold: float = 0.2,
) -> list[RegionOfInterest]:
    """Generate, expand, rank, and de-duplicate FOCUS regions of interest."""

    relevance_map = gaussian_smooth(relevance_map)
    height, width = relevance_map.shape
    image_width, image_height = image_size
    max_side = max_side or max(height, width)
    chosen: list[tuple[int, int]] = []
    for flat_index in torch.argsort(relevance_map.flatten(), descending=True).tolist():
        row, column = divmod(flat_index, width)
        if all(math.hypot(row - other_row, column - other_column) >= min_anchor_distance for other_row, other_column in chosen):
            chosen.append((row, column))
        if len(chosen) >= anchors:
            break

    proposals = []
    threshold = float(relevance_map.mean()) * expansion_threshold
    for row, column in chosen:
        side = min_side
        best = (column, row, column + 1, row + 1)
        while side <= max_side:
            half = side / 2
            left = max(0, math.floor(column + 0.5 - half))
            top = max(0, math.floor(row + 0.5 - half))
            right = min(width, math.ceil(column + 0.5 + half))
            bottom = min(height, math.ceil(row + 0.5 + half))
            if float(relevance_map[top:bottom, left:right].mean()) < threshold and side > min_side:
                break
            best = (left, top, right, bottom)
            side += 2
        left, top, right, bottom = best
        box = (
            image_width * left / width,
            image_height * top / height,
            image_width * right / width,
            image_height * bottom / height,
        )
        proposals.append(RegionOfInterest(box=box, relevance=float(relevance_map[row, column])))

    retained: list[RegionOfInterest] = []
    for proposal in sorted(proposals, key=lambda item: item.relevance, reverse=True):
        if all(_iou(proposal.box, item.box) <= nms_threshold for item in retained):
            retained.append(proposal)
    return retained


def best_text_span(relevance: torch.Tensor, max_span_tokens: int) -> tuple[int, int]:
    """Return a coherent span around the highest-relevance evidence token."""

    if relevance.ndim != 1 or relevance.shape[0] == 0:
        raise ValueError("relevance must be a non-empty vector")
    if max_span_tokens <= 0:
        raise ValueError("max_span_tokens must be positive")
    limit = min(max_span_tokens, relevance.shape[0])
    start = int(relevance.argmax())
    end = start + 1
    threshold = float(relevance.mean())
    while end - start < limit:
        left_score = float(relevance[start - 1]) if start else -math.inf
        right_score = float(relevance[end]) if end < relevance.shape[0] else -math.inf
        if max(left_score, right_score) < threshold and end - start >= max(1, limit // 2):
            break
        if left_score >= right_score:
            start -= 1
        else:
            end += 1
    return start, end


def best_image_region(
    relevance: torch.Tensor,
    patch_boxes: torch.Tensor,
    *,
    top_fraction: float = 0.2,
) -> tuple[float, float, float, float]:
    """Return the union box of the most relevant image patches."""

    if relevance.ndim != 1 or patch_boxes.shape != (relevance.shape[0], 4):
        raise ValueError("patch_boxes must have shape [evidence tokens, 4]")
    if relevance.shape[0] == 0:
        raise ValueError("relevance must be non-empty")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    count = max(1, math.ceil(relevance.shape[0] * top_fraction))
    selected = patch_boxes[torch.topk(relevance, k=count).indices]
    return (
        float(selected[:, 0].min()),
        float(selected[:, 1].min()),
        float(selected[:, 2].max()),
        float(selected[:, 3].max()),
    )


def semantic_joinability(
    query_values: Sequence[str],
    target_values: Sequence[str],
    *,
    query_embeddings: torch.Tensor | None = None,
    target_embeddings: torch.Tensor | None = None,
    similarity_threshold: float = 0.8,
    min_coverage: float = 0.6,
) -> SemanticJoinability:
    """Check whether augmented query values are semantically contained in a target column."""

    if not query_values or not target_values:
        return SemanticJoinability(False, 0.0, 0.0, (), ())
    if (query_embeddings is None) != (target_embeddings is None):
        raise ValueError("query_embeddings and target_embeddings must be supplied together")
    if query_embeddings is not None:
        if query_embeddings.shape[0] != len(query_values) or target_embeddings.shape[0] != len(target_values):
            raise ValueError("Embedding rows must match their values")
        similarities = F.normalize(query_embeddings.float(), dim=-1) @ F.normalize(target_embeddings.float(), dim=-1).T
    else:
        similarities = torch.zeros((len(query_values), len(target_values)))

    best_scores = []
    best_indices = []
    for query_index, query_value in enumerate(query_values):
        if not str(query_value).strip():
            best_scores.append(0.0)
            best_indices.append(-1)
            continue
        exact = next((index for index, value in enumerate(target_values) if values_match(query_value, value)), None)
        if exact is not None:
            best_scores.append(1.0)
            best_indices.append(exact)
        else:
            score, index = similarities[query_index].max(dim=0)
            best_scores.append(float(score))
            best_indices.append(int(index))
    coverage = sum(score >= similarity_threshold for score in best_scores) / len(best_scores)
    return SemanticJoinability(
        joinable=coverage >= min_coverage,
        coverage=coverage,
        mean_similarity=sum(best_scores) / len(best_scores),
        similarities=tuple(best_scores),
        target_indices=tuple(best_indices),
    )

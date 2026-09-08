"""Core scoring and evidence-localization algorithms for Stage 2."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from mmdd_dataset.utils import values_match
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceBundle:
    target_id: str
    retrieval_score: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegionOfInterest:
    box: tuple[float, float, float, float]
    relevance: float


@dataclass(frozen=True)
class SemanticJoinability:
    joinable: bool
    coverage: float
    mean_similarity: float


def build_evidence_bundles(
    retrieval_results: Sequence[dict[str, Any]],
    *,
    top_k_evidence: int,
) -> list[EvidenceBundle]:
    """Build evidence-only bundles in Stage-1's compact path order."""

    if top_k_evidence < 0:
        raise ValueError("top_k_evidence must be non-negative")
    bundles = []
    for result in retrieval_results:
        evidence_ids = tuple(
            str(path["evidence_id"])
            for path in result.get("paths", [])
            if path.get("kind") == "evidence" and path.get("evidence_id") is not None
        )[:top_k_evidence]
        if not evidence_ids:
            continue
        evidence_score = result.get("evidence_score")
        if evidence_score is None:
            raise ValueError(
                f"{result['target_id']}: retrieval result has evidence paths but no evidence_score; "
                "regenerate it with the current Stage-1 retriever"
            )
        table_score = result.get("stage2_table_score", evidence_score)
        bundles.append(
            EvidenceBundle(
                target_id=str(result["target_id"]),
                retrieval_score=float(table_score),
                evidence_ids=evidence_ids,
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
    column_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute softmax(r_T) * rho_(T,c) from the verifier formulation."""

    if retrieval_scores.ndim != 1:
        raise ValueError("retrieval_scores must have shape [targets]")
    if (
        column_logits.ndim != 2
        or column_logits.shape != column_mask.shape
        or column_logits.shape[0] != retrieval_scores.shape[0]
    ):
        raise ValueError("Column tensors must have shape [targets, columns]")
    if not torch.all(column_mask.any(dim=-1)):
        raise ValueError("Every target must have at least one candidate column")
    table_probabilities = torch.softmax(retrieval_scores, dim=-1)
    column_probabilities = torch.softmax(
        column_logits.masked_fill(~column_mask, -torch.inf), dim=-1
    )
    column_probabilities = column_probabilities.masked_fill(~column_mask, 0.0)
    return table_probabilities.unsqueeze(-1) * column_probabilities


def _relevance_logits(
    query_states: torch.Tensor,
    evidence_states: torch.Tensor,
) -> torch.Tensor:
    if query_states.ndim != 2 or evidence_states.ndim != 2:
        raise ValueError("Query and evidence states must have shape [tokens, hidden]")
    if query_states.shape[1] != evidence_states.shape[1] or query_states.shape[0] == 0 or evidence_states.shape[0] == 0:
        raise ValueError("Hidden dimensions must match and token sequences cannot be empty")
    query = F.normalize(query_states.mean(dim=0).float(), dim=0)
    evidence = F.normalize(evidence_states.float(), dim=-1)
    return evidence @ query


def joint_relevance_logits(
    row_anchor_states: torch.Tensor,
    attribute_states: torch.Tensor,
    evidence_states: torch.Tensor,
) -> torch.Tensor:
    """Return unnormalized joint row-anchor and attribute relevance."""

    return _relevance_logits(
        row_anchor_states, evidence_states
    ) + _relevance_logits(attribute_states, evidence_states)


def focus_relevance_from_logits(
    layer_logits: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Normalize joint logits per layer and average the resulting maps."""

    if not layer_logits:
        raise ValueError("FOCUS requires at least one value-feature layer")
    shape = layer_logits[0].shape
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("FOCUS layer logits must be non-empty vectors")
    if any(logits.shape != shape for logits in layer_logits):
        raise ValueError("FOCUS layer logits must share one token axis")
    return torch.stack(
        [torch.softmax(logits.float(), dim=0) for logits in layer_logits]
    ).mean(dim=0)


def focus_relevance(
    value_layers: Sequence[torch.Tensor],
    row_anchor_indices: torch.Tensor,
    attribute_indices: torch.Tensor,
    evidence_indices: torch.Tensor,
) -> torch.Tensor:
    """Aggregate FOCUS-style value-feature maps over later MLLM layers."""

    logits = [
        joint_relevance_logits(
            layer.index_select(0, row_anchor_indices),
            layer.index_select(0, attribute_indices),
            layer.index_select(0, evidence_indices),
        )
        for layer in value_layers
    ]
    return focus_relevance_from_logits(logits)


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


def semantic_joinability(
    query_values: Sequence[str],
    target_values: Sequence[str],
    *,
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    similarity_threshold: float = 0.8,
    min_coverage: float = 0.6,
    similarity_batch_size: int = 1024,
) -> SemanticJoinability:
    """Check whether augmented query values are semantically contained in a target column."""

    if similarity_batch_size <= 0:
        raise ValueError("similarity_batch_size must be positive")
    if not query_values or not target_values:
        return SemanticJoinability(False, 0.0, 0.0)
    if (
        query_embeddings.shape[0] != len(query_values)
        or target_embeddings.shape[0] != len(target_values)
    ):
        raise ValueError("Embedding rows must match their values")
    query_vectors = F.normalize(query_embeddings.float(), dim=-1)
    target_vectors = F.normalize(target_embeddings.float(), dim=-1)
    semantic_scores = torch.empty(len(query_values), dtype=torch.float32)
    for query_start in range(0, len(query_values), similarity_batch_size):
        query_batch = query_vectors[
            query_start : query_start + similarity_batch_size
        ]
        batch_scores = torch.full(
            (query_batch.shape[0],),
            -torch.inf,
            device=query_batch.device,
        )
        for target_start in range(0, len(target_values), similarity_batch_size):
            target_batch = target_vectors[
                target_start : target_start + similarity_batch_size
            ]
            batch_scores = torch.maximum(
                batch_scores,
                (query_batch @ target_batch.T).max(dim=1).values,
            )
        semantic_scores[
            query_start : query_start + query_batch.shape[0]
        ] = batch_scores.cpu()

    best_scores = []
    for query_index, query_value in enumerate(query_values):
        if not str(query_value).strip():
            best_scores.append(0.0)
            continue
        if any(values_match(query_value, value) for value in target_values):
            best_scores.append(1.0)
        else:
            best_scores.append(float(semantic_scores[query_index]))
    coverage = sum(
        bool(str(value).strip()) and score >= similarity_threshold
        for value, score in zip(query_values, best_scores, strict=True)
    ) / len(best_scores)
    return SemanticJoinability(
        joinable=coverage >= min_coverage,
        coverage=coverage,
        mean_similarity=sum(best_scores) / len(best_scores),
    )

"""Monotonic row-support calibration and fixed-budget evidence selection."""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .artifacts import checkpoint_fingerprint


@dataclass(frozen=True)
class IsotonicModel:
    """Piecewise-constant nondecreasing probability map."""

    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.values):
            raise ValueError("Isotonic model requires equally sized non-empty knots")
        if any(
            right <= left
            for left, right in zip(self.upper_bounds, self.upper_bounds[1:])
        ):
            raise ValueError("Isotonic upper bounds must be strictly increasing")
        if any(not 0.0 <= value <= 1.0 for value in self.values):
            raise ValueError("Isotonic values must lie in [0, 1]")
        if any(
            right < left for left, right in zip(self.values, self.values[1:])
        ):
            raise ValueError("Isotonic values must be nondecreasing")

    def predict(self, score: float) -> float:
        index = bisect.bisect_left(self.upper_bounds, float(score))
        index = min(index, len(self.values) - 1)
        return self.values[index]

    def to_json(self) -> dict[str, list[float]]:
        return {
            "upper_bounds": list(self.upper_bounds),
            "values": list(self.values),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> IsotonicModel:
        return cls(
            upper_bounds=tuple(float(value) for value in payload["upper_bounds"]),
            values=tuple(float(value) for value in payload["values"]),
        )


def load_row_support_models(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, IsotonicModel], str]:
    """Load the frozen modality-specific maps used by G5."""

    path = path.resolve()
    sha256 = checkpoint_fingerprint(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"{path}: row-support model fingerprint differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ValueError(f"{path}: invalid row-support model")
    models = {
        str(evidence_type): IsotonicModel.from_json(model)
        for evidence_type, model in payload["models"].items()
    }
    if not {"text", "image"} <= models.keys():
        raise ValueError(f"{path}: row-support model must cover text and image")
    return models, sha256


def load_evidence_content_keys(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, str], str]:
    """Load exact-content keys used to prevent duplicate G5 votes."""

    path = path.resolve()
    sha256 = checkpoint_fingerprint(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(f"{path}: evidence content-key fingerprint differs")
    keys: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            object_id = str(record.get("object_id", ""))
            content_key = str(record.get("content_key", ""))
            if not object_id or not content_key:
                raise ValueError(
                    f"{path}:{line_number}: object_id and content_key are required"
                )
            if object_id in keys:
                raise ValueError(f"{path}:{line_number}: duplicate object_id {object_id!r}")
            keys[object_id] = content_key
    if not keys:
        raise ValueError(f"{path}: evidence content-key manifest is empty")
    return keys, sha256


def embedding_content_key(embedding: torch.Tensor) -> str:
    """Return a stable key that collapses byte-identical frozen embeddings."""

    values = embedding.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def predict_row_support(
    row_embeddings: torch.Tensor,
    evidence_embedding: torch.Tensor,
    model: IsotonicModel,
) -> list[float]:
    """Apply a frozen isotonic map to raw row/evidence cosine similarities."""

    if row_embeddings.ndim != 2 or evidence_embedding.ndim != 1:
        raise ValueError("Row and evidence embeddings must have shapes [rows,D] and [D]")
    if row_embeddings.shape[1] != evidence_embedding.shape[0]:
        raise ValueError("Row and evidence embedding dimensions must match")
    similarities = torch.mv(
        row_embeddings.detach().cpu().float(),
        evidence_embedding.detach().cpu().float(),
    )
    return [model.predict(float(value)) for value in similarities]


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> IsotonicModel:
    """Fit a weighted pool-adjacent-violators model after merging tied scores."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("Scores and labels must be equally sized and non-empty")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Isotonic labels must be binary")

    tied: list[list[float]] = []
    for score, label in sorted(zip(scores, labels), key=lambda pair: pair[0]):
        score = float(score)
        if tied and score == tied[-1][0]:
            tied[-1][1] += 1.0
            tied[-1][2] += float(label)
        else:
            tied.append([score, 1.0, float(label)])

    blocks: list[list[float]] = []
    for upper, weight, positive in tied:
        blocks.append([upper, weight, positive])
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            if left[2] / left[1] <= right[2] / right[1]:
                break
            blocks[-2:] = [
                [right[0], left[1] + right[1], left[2] + right[2]]
            ]

    return IsotonicModel(
        upper_bounds=tuple(block[0] for block in blocks),
        values=tuple(block[2] / block[1] for block in blocks),
    )


def threshold_strength(score: float, threshold: float) -> float:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("Threshold must lie in [0, 1)")
    return min(1.0, max(0.0, (float(score) - threshold) / (1.0 - threshold)))


def greedy_row_bundle(
    candidates: Sequence[dict[str, object]],
    *,
    row_support: dict[str, Sequence[float]],
    budget: int,
    threshold: float,
) -> tuple[list[str], float]:
    """Choose evidence by deterministic marginal row-coverage gain."""

    if budget <= 0:
        raise ValueError("Evidence budget must be positive")
    if not candidates:
        return [], 0.0
    row_count = len(next(iter(row_support.values()))) if row_support else 0
    if row_count == 0:
        return [], 0.0
    if any(len(values) != row_count for values in row_support.values()):
        raise ValueError("All row-support vectors must have the same length")

    remaining = list(candidates)
    current = [0.0] * row_count
    selected: list[str] = []
    for _step in range(min(budget, len(remaining))):
        choices = []
        for candidate in remaining:
            evidence_id = str(candidate["evidence_id"])
            quality = threshold_strength(float(candidate["quality"]), threshold)
            support = row_support[evidence_id]
            updated = [
                max(value, quality * float(probability))
                for value, probability in zip(current, support)
            ]
            score = sum(updated) / row_count
            gain = score - sum(current) / row_count
            choices.append(
                (
                    gain,
                    float(candidate["quality"]),
                    evidence_id,
                    updated,
                )
            )
        gain, _quality, evidence_id, updated = min(
            choices,
            key=lambda value: (-value[0], -value[1], value[2]),
        )
        if gain <= 0.0:
            break
        selected.append(evidence_id)
        current = updated
        remaining = [
            candidate
            for candidate in remaining
            if str(candidate["evidence_id"]) != evidence_id
        ]
    return selected, sum(current) / row_count


def greedy_row_bundle_tensor(
    qualities: torch.Tensor,
    row_support: torch.Tensor,
    mask: torch.Tensor,
    *,
    budget: int,
    top_l: int,
    threshold: float,
    content_groups: torch.Tensor | None = None,
    row_mask: torch.Tensor | None = None,
) -> tuple[list[int], torch.Tensor]:
    """Tensor G5 selection with gradients through the selected path qualities.

    Top-L and greedy bundle choices are discrete, like ``torch.topk`` in G3.
    Choices are made from detached scores, while the final max-coverage score
    retains gradients through each selected path quality.
    """

    if qualities.ndim != 1 or mask.shape != qualities.shape:
        raise ValueError("qualities and mask must be aligned one-dimensional tensors")
    if row_support.ndim != 2 or row_support.shape[0] != qualities.shape[0]:
        raise ValueError("row_support must have shape [evidence, rows]")
    if budget <= 0 or top_l <= 0:
        raise ValueError("G5 budgets must be positive")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("Threshold must lie in [0, 1)")
    if content_groups is not None and content_groups.shape != qualities.shape:
        raise ValueError("content_groups must align with qualities")
    if row_mask is None:
        row_mask = torch.ones(
            row_support.shape[1], dtype=torch.bool, device=row_support.device
        )
    if row_mask.ndim != 1 or row_mask.shape[0] != row_support.shape[1]:
        raise ValueError("row_mask must align with the row-support dimension")
    if not bool(row_mask.any()):
        return [], qualities.sum() * 0.0

    valid_indices = mask.nonzero(as_tuple=True)[0].tolist()
    if not valid_indices:
        return [], qualities.sum() * 0.0
    groups = (
        content_groups.detach().cpu().tolist()
        if content_groups is not None
        else list(range(len(qualities)))
    )
    best_by_group: dict[int, int] = {}
    for index in valid_indices:
        group = int(groups[index])
        previous = best_by_group.get(group)
        if previous is None or (
            -float(qualities[index].detach()), index
        ) < (-float(qualities[previous].detach()), previous):
            best_by_group[group] = index
    candidates = sorted(
        best_by_group.values(),
        key=lambda index: (-float(qualities[index].detach()), index),
    )[:top_l]

    current = qualities.new_zeros(int(row_mask.sum()))
    selected: list[int] = []
    remaining = list(candidates)
    for _step in range(min(budget, len(remaining))):
        choices = []
        for index in remaining:
            strength = ((qualities[index] - threshold) / (1.0 - threshold)).clamp(
                min=0.0, max=1.0
            )
            support = row_support[index, row_mask].to(qualities.dtype)
            updated = torch.maximum(current, strength * support)
            choices.append((updated.mean(), index, updated))
        score, index, updated = min(
            choices,
            key=lambda value: (
                -float(value[0].detach()),
                -float(qualities[value[1]].detach()),
                value[1],
            ),
        )
        if float(score.detach()) <= float(current.mean().detach()):
            break
        selected.append(index)
        current = updated
        remaining.remove(index)
    return selected, current.mean()

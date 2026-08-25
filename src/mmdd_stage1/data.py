"""JSONL training records for edge lists and target/path lists."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .features import normalize_object_type


@dataclass(frozen=True)
class EdgeExample:
    query_id: str
    candidate_ids: tuple[str, ...]
    positive_index: int
    dataset: str = "default"
    split: str | None = None
    teacher_logits: tuple[float, ...] | None = None
    teacher_checkpoint_sha256: str | None = None
    source_type: str | None = None
    destination_type: str | None = None
    edge_kind: str | None = None


@dataclass(frozen=True)
class TargetCandidate:
    target_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TeacherScoreConfig:
    evidence_aggregation: str
    evidence_top_k: int


@dataclass(frozen=True)
class TargetExample:
    query_id: str
    candidates: tuple[TargetCandidate, ...]
    positive_index: int
    dataset: str = "default"
    split: str | None = None
    teacher_logits: tuple[float, ...] | None = None
    positive_target_ids: tuple[str, ...] = ()
    teacher_score_config: TeacherScoreConfig | None = None
    teacher_checkpoint_sha256: str | None = None


def _records(path: Path, split: str | None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must contain a JSON object")
            if split is None or record.get("split") == split:
                yield line_number, record


def _positive_index(
    path: Path,
    line_number: int,
    record: dict[str, Any],
    candidate_ids: list[str],
    positive_id_key: str,
) -> int:
    if positive_id_key in record:
        positive_id = str(record[positive_id_key])
        try:
            return candidate_ids.index(positive_id)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: positive ID is not present in candidates") from exc
    if "positive_index" not in record:
        raise ValueError(f"{path}:{line_number}: missing {positive_id_key} or positive_index")
    index = int(record["positive_index"])
    if not 0 <= index < len(candidate_ids):
        raise ValueError(f"{path}:{line_number}: positive_index is outside the candidate list")
    return index


def _teacher_logits(
    path: Path,
    line_number: int,
    record: dict[str, Any],
    candidate_count: int,
) -> tuple[float, ...] | None:
    values = record.get("teacher_logits")
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != candidate_count:
        raise ValueError(f"{path}:{line_number}: teacher_logits must align with candidates")
    logits = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in logits):
        raise ValueError(f"{path}:{line_number}: teacher_logits must be finite")
    return logits


def load_edge_examples(
    path: Path,
    *,
    split: str | None = "train",
    dataset_name: str | None = None,
) -> list[EdgeExample]:
    examples = []
    for line_number, record in _records(path, split):
        query_id = str(record["query_id"])
        candidate_ids = [str(value) for value in record["candidate_ids"]]
        if len(candidate_ids) < 2:
            raise ValueError(f"{path}:{line_number}: edge candidate list must contain at least two objects")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"{path}:{line_number}: candidate_ids contains duplicates")
        positive_index = _positive_index(path, line_number, record, candidate_ids, "positive_id")
        examples.append(
            EdgeExample(
                query_id,
                tuple(candidate_ids),
                positive_index,
                dataset=str(record.get("dataset") or dataset_name or "default"),
                split=record.get("split"),
                teacher_logits=_teacher_logits(path, line_number, record, len(candidate_ids)),
                teacher_checkpoint_sha256=record.get("teacher_checkpoint_sha256"),
                source_type=(
                    normalize_object_type(str(record["source_type"]))
                    if record.get("source_type") is not None
                    else None
                ),
                destination_type=(
                    normalize_object_type(str(record["destination_type"]))
                    if record.get("destination_type") is not None
                    else None
                ),
                edge_kind=record.get("edge_kind"),
            )
        )
    if not examples:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"{path}: no edge examples{suffix}")
    return examples


def load_target_examples(
    path: Path,
    *,
    split: str | None = "train",
    max_evidence: int = 8,
    dataset_name: str | None = None,
) -> list[TargetExample]:
    if max_evidence < 0:
        raise ValueError("max_evidence must be non-negative")
    examples = []
    for line_number, record in _records(path, split):
        query_id = str(record["query_id"])
        raw_candidates = record["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError(f"{path}:{line_number}: target candidate list must contain at least two targets")
        candidates = []
        evidence_was_truncated = False
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError(f"{path}:{line_number}: each target candidate must be an object")
            raw_evidence_ids = raw_candidate.get("evidence_ids", [])
            if not isinstance(raw_evidence_ids, list):
                raise ValueError(f"{path}:{line_number}: evidence_ids must be a list")
            evidence_was_truncated |= len(raw_evidence_ids) > max_evidence
            evidence_ids = tuple(str(value) for value in raw_evidence_ids[:max_evidence])
            candidates.append(TargetCandidate(str(raw_candidate["target_id"]), evidence_ids))
        target_ids = [candidate.target_id for candidate in candidates]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"{path}:{line_number}: target candidates contain duplicates")
        positive_index = _positive_index(path, line_number, record, target_ids, "positive_target_id")
        positive_target_ids = tuple(str(value) for value in record.get("positive_target_ids", []))
        designated_positive = target_ids[positive_index]
        if not positive_target_ids:
            positive_target_ids = (designated_positive,)
        elif designated_positive not in positive_target_ids:
            raise ValueError(f"{path}:{line_number}: positive_target_ids omits the designated positive")
        teacher_logits = _teacher_logits(path, line_number, record, len(candidates))
        if teacher_logits is not None and evidence_was_truncated:
            raise ValueError(
                f"{path}:{line_number}: cannot reuse teacher_logits after truncating candidate evidence"
            )
        raw_score_config = record.get("teacher_score_config")
        teacher_score_config = None
        if raw_score_config is not None:
            if not isinstance(raw_score_config, dict):
                raise ValueError(f"{path}:{line_number}: teacher_score_config must be an object")
            teacher_score_config = TeacherScoreConfig(
                evidence_aggregation=str(raw_score_config["evidence_aggregation"]),
                evidence_top_k=int(raw_score_config["evidence_top_k"]),
            )
            if teacher_score_config.evidence_aggregation not in {"logsumexp", "topk_mean", "topk_sum"}:
                raise ValueError(f"{path}:{line_number}: invalid Teacher evidence aggregation")
            if teacher_score_config.evidence_top_k <= 0:
                raise ValueError(f"{path}:{line_number}: Teacher evidence_top_k must be positive")
        examples.append(
            TargetExample(
                query_id,
                tuple(candidates),
                positive_index,
                dataset=str(record.get("dataset") or dataset_name or "default"),
                split=record.get("split"),
                teacher_logits=teacher_logits,
                positive_target_ids=positive_target_ids,
                teacher_score_config=teacher_score_config,
                teacher_checkpoint_sha256=record.get("teacher_checkpoint_sha256"),
            )
        )
    if not examples:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"{path}: no target examples{suffix}")
    return examples

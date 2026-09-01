"""JSONL training records for edge lists and target/path lists."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mmdd_progress import progress

from .features import normalize_object_type
from .objectives import PATH_AGGREGATIONS


@dataclass(frozen=True)
class EdgeExample:
    query_id: str
    candidate_ids: tuple[str, ...]
    positive_index: int
    dataset: str = "default"
    split: str | None = None
    teacher_logits: tuple[float, ...] | None = None
    teacher_checkpoint_sha256: str | None = None
    teacher_logit_mode: str | None = None
    teacher_ensemble_alpha: float | None = None
    source_type: str | None = None
    destination_type: str | None = None


@dataclass(frozen=True)
class TargetCandidate:
    target_id: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TeacherScoreConfig:
    evidence_aggregation: str
    evidence_top_k: int
    evidence_temperature: float = 1.0
    evidence_power: float = 2.0


@dataclass(frozen=True)
class TargetExample:
    query_id: str
    candidates: tuple[TargetCandidate, ...]
    direct_positive_index: int
    evidence_positive_index: int
    dataset: str = "default"
    split: str | None = None
    teacher_direct_logits: tuple[float, ...] | None = None
    teacher_evidence_logits: tuple[float, ...] | None = None
    positive_target_ids: tuple[str, ...] = ()
    teacher_score_config: TeacherScoreConfig | None = None
    teacher_checkpoint_sha256: str | None = None
    teacher_logit_mode: str | None = None
    teacher_ensemble_alpha: float | None = None


def _records(path: Path, split: str | None) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        lines = progress(
            handle,
            desc=f"Load {path.name}",
            unit="record",
            leave=False,
        )
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must contain a JSON object")
            if split is None or record.get("split") == split:
                yield line_number, record


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be a JSON object")
    return metadata


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
    key: str = "teacher_logits",
) -> tuple[float, ...] | None:
    values = record.get(key)
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != candidate_count:
        raise ValueError(f"{path}:{line_number}: {key} must align with candidates")
    logits = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in logits):
        raise ValueError(f"{path}:{line_number}: {key} must be finite")
    return logits


def load_edge_examples(
    path: Path,
    *,
    split: str | None = "train",
    dataset_name: str | None = None,
) -> list[EdgeExample]:
    metadata = _metadata(path)
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
                teacher_checkpoint_sha256=(
                    record.get("teacher_checkpoint_sha256")
                    or metadata.get("teacher_checkpoint_sha256")
                ),
                teacher_logit_mode=(
                    record.get("teacher_logit_mode")
                    or metadata.get("teacher_edge_logit_mode")
                    or metadata.get("teacher_logit_mode")
                ),
                teacher_ensemble_alpha=(
                    record.get("teacher_ensemble_alpha")
                    if record.get("teacher_ensemble_alpha") is not None
                    else metadata.get("teacher_edge_ensemble_alpha")
                ),
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
    dataset_name: str | None = None,
) -> list[TargetExample]:
    metadata = _metadata(path)
    examples = []
    for line_number, record in _records(path, split):
        query_id = str(record["query_id"])
        raw_candidates = record["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise ValueError(f"{path}:{line_number}: target candidate list must contain at least two targets")
        candidates = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                raise ValueError(f"{path}:{line_number}: each target candidate must be an object")
            raw_evidence_ids = raw_candidate.get("evidence_ids", [])
            if not isinstance(raw_evidence_ids, list):
                raise ValueError(f"{path}:{line_number}: evidence_ids must be a list")
            evidence_ids = tuple(str(value) for value in raw_evidence_ids)
            candidates.append(TargetCandidate(str(raw_candidate["target_id"]), evidence_ids))
        target_ids = [candidate.target_id for candidate in candidates]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"{path}:{line_number}: target candidates contain duplicates")
        required_positive_keys = {
            "direct_positive_target_id",
            "evidence_positive_target_id",
        }
        if "positive_target_id" in record or not required_positive_keys <= record.keys():
            raise ValueError(
                f"{path}:{line_number}: target lists require separate "
                "direct_positive_target_id and evidence_positive_target_id; regenerate this file"
            )
        direct_positive_index = _positive_index(
            path, line_number, record, target_ids, "direct_positive_target_id"
        )
        evidence_positive_index = _positive_index(
            path, line_number, record, target_ids, "evidence_positive_target_id"
        )
        positive_target_ids = tuple(str(value) for value in record.get("positive_target_ids", []))
        designated_positives = {
            target_ids[direct_positive_index],
            target_ids[evidence_positive_index],
        }
        if not positive_target_ids:
            positive_target_ids = tuple(
                dict.fromkeys(
                    target_ids[index]
                    for index in (direct_positive_index, evidence_positive_index)
                )
            )
        elif not designated_positives <= set(positive_target_ids):
            raise ValueError(f"{path}:{line_number}: positive_target_ids omits a designated positive")
        if len(set(positive_target_ids)) != len(positive_target_ids):
            raise ValueError(f"{path}:{line_number}: positive_target_ids contains duplicates")
        missing_positive_candidates = set(positive_target_ids) - set(target_ids)
        if missing_positive_candidates:
            raise ValueError(
                f"{path}:{line_number}: positive_target_ids references missing candidates: "
                + ", ".join(sorted(missing_positive_candidates))
            )
        if "teacher_logits" in record:
            raise ValueError(
                f"{path}:{line_number}: merged target teacher_logits are obsolete; "
                "regenerate separate teacher_direct_logits and teacher_evidence_logits"
            )
        has_direct_logits = "teacher_direct_logits" in record
        has_evidence_logits = "teacher_evidence_logits" in record
        if has_direct_logits != has_evidence_logits:
            raise ValueError(
                f"{path}:{line_number}: cached target scores require both "
                "teacher_direct_logits and teacher_evidence_logits"
            )
        teacher_direct_logits = _teacher_logits(
            path, line_number, record, len(candidates), "teacher_direct_logits"
        )
        teacher_evidence_logits = _teacher_logits(
            path, line_number, record, len(candidates), "teacher_evidence_logits"
        )
        raw_score_config = record.get("teacher_score_config")
        if raw_score_config is None and {
            "evidence_aggregation",
            "evidence_top_k",
        } <= metadata.keys():
            raw_score_config = {
                "evidence_aggregation": metadata["evidence_aggregation"],
                "evidence_top_k": metadata["evidence_top_k"],
                "evidence_temperature": metadata.get("evidence_temperature", 1.0),
                "evidence_power": metadata.get("evidence_power", 2.0),
            }
        teacher_score_config = None
        if raw_score_config is not None:
            if not isinstance(raw_score_config, dict):
                raise ValueError(f"{path}:{line_number}: teacher_score_config must be an object")
            teacher_score_config = TeacherScoreConfig(
                evidence_aggregation=str(raw_score_config["evidence_aggregation"]),
                evidence_top_k=int(raw_score_config["evidence_top_k"]),
                evidence_temperature=float(
                    raw_score_config.get("evidence_temperature", 1.0)
                ),
                evidence_power=float(raw_score_config.get("evidence_power", 2.0)),
            )
            if teacher_score_config.evidence_aggregation not in PATH_AGGREGATIONS:
                raise ValueError(f"{path}:{line_number}: invalid Teacher evidence aggregation")
            if teacher_score_config.evidence_top_k <= 0:
                raise ValueError(f"{path}:{line_number}: Teacher evidence_top_k must be positive")
            if teacher_score_config.evidence_temperature <= 0:
                raise ValueError(
                    f"{path}:{line_number}: Teacher evidence_temperature must be positive"
                )
            if teacher_score_config.evidence_power <= 0:
                raise ValueError(
                    f"{path}:{line_number}: Teacher evidence_power must be positive"
                )
        examples.append(
            TargetExample(
                query_id,
                tuple(candidates),
                direct_positive_index,
                evidence_positive_index,
                dataset=str(record.get("dataset") or dataset_name or "default"),
                split=record.get("split"),
                teacher_direct_logits=teacher_direct_logits,
                teacher_evidence_logits=teacher_evidence_logits,
                positive_target_ids=positive_target_ids,
                teacher_score_config=teacher_score_config,
                teacher_checkpoint_sha256=(
                    record.get("teacher_checkpoint_sha256")
                    or metadata.get("teacher_checkpoint_sha256")
                ),
                teacher_logit_mode=(
                    record.get("teacher_logit_mode")
                    or metadata.get("teacher_target_logit_mode")
                    or metadata.get("teacher_logit_mode")
                ),
                teacher_ensemble_alpha=(
                    record.get("teacher_ensemble_alpha")
                    if record.get("teacher_ensemble_alpha") is not None
                    else metadata.get("teacher_target_ensemble_alpha")
                ),
            )
        )
    if not examples:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"{path}: no target examples{suffix}")
    return examples

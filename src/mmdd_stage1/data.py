"""JSONL training records for edge lists and target/path lists."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mmdd_progress import progress

from .features import normalize_object_type
from .objectives import PATH_AGGREGATIONS, PATH_COMBINATIONS


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
    positive_ids: tuple[str, ...] = ()
    confirmed_labels: tuple[int | None, ...] | None = None


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
    path_combination: str = "sum"
    evidence_threshold: float = 0.0
    evidence_target_temperature: float = 1.0
    row_support_model: str | None = None
    row_support_model_sha256: str | None = None
    row_support_top_l: int = 20
    evidence_content_keys: str | None = None
    evidence_content_keys_sha256: str | None = None


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


def _edge_positive_ids(
    path: Path,
    line_number: int,
    record: dict[str, Any],
    candidate_ids: list[str],
    positive_index: int,
) -> tuple[str, ...]:
    values = record.get("positive_ids")
    if values is None:
        return (candidate_ids[positive_index],)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path}:{line_number}: positive_ids must be a non-empty list")
    positive_ids = tuple(str(value) for value in values)
    if len(set(positive_ids)) != len(positive_ids):
        raise ValueError(f"{path}:{line_number}: positive_ids contains duplicates")
    missing = set(positive_ids) - set(candidate_ids)
    if missing:
        raise ValueError(
            f"{path}:{line_number}: positive_ids references missing candidates: "
            + ", ".join(sorted(missing))
        )
    if candidate_ids[positive_index] not in positive_ids:
        raise ValueError(
            f"{path}:{line_number}: positive_ids omits the designated positive"
        )
    return positive_ids


def _confirmed_edge_labels(
    path: Path,
    line_number: int,
    record: dict[str, Any],
    candidate_ids: list[str],
) -> tuple[int | None, ...] | None:
    values = record.get("confirmed_labels")
    positive_values = record.get("confirmed_positive_ids")
    negative_values = record.get("confirmed_negative_ids")
    if values is not None and (positive_values is not None or negative_values is not None):
        raise ValueError(
            f"{path}:{line_number}: use confirmed_labels or confirmed positive/negative IDs, not both"
        )
    if values is not None:
        if not isinstance(values, list) or len(values) != len(candidate_ids):
            raise ValueError(
                f"{path}:{line_number}: confirmed_labels must align with candidates"
            )
        labels = []
        for value in values:
            if value is None:
                labels.append(None)
            elif isinstance(value, (bool, int)) and int(value) in {0, 1}:
                labels.append(int(value))
            else:
                raise ValueError(
                    f"{path}:{line_number}: confirmed_labels entries must be 0, 1, or null"
                )
        return tuple(labels)
    if positive_values is None and negative_values is None:
        return None
    if positive_values is not None and not isinstance(positive_values, list):
        raise ValueError(
            f"{path}:{line_number}: confirmed_positive_ids must be a list"
        )
    if negative_values is not None and not isinstance(negative_values, list):
        raise ValueError(
            f"{path}:{line_number}: confirmed_negative_ids must be a list"
        )
    confirmed_positive_ids = {str(value) for value in positive_values or []}
    confirmed_negative_ids = {str(value) for value in negative_values or []}
    overlap = confirmed_positive_ids & confirmed_negative_ids
    if overlap:
        raise ValueError(
            f"{path}:{line_number}: confirmed positive and negative IDs overlap: "
            + ", ".join(sorted(overlap))
        )
    missing = (confirmed_positive_ids | confirmed_negative_ids) - set(candidate_ids)
    if missing:
        raise ValueError(
            f"{path}:{line_number}: confirmed labels reference missing candidates: "
            + ", ".join(sorted(missing))
        )
    return tuple(
        1
        if candidate_id in confirmed_positive_ids
        else 0
        if candidate_id in confirmed_negative_ids
        else None
        for candidate_id in candidate_ids
    )


def _provenance_value(
    record: dict[str, Any],
    metadata: dict[str, Any],
    record_key: str,
    *metadata_keys: str,
    fallback_on_falsy: bool,
) -> Any:
    sources_and_keys = (
        (record, record_key),
        *((metadata, key) for key in metadata_keys),
    )
    if fallback_on_falsy:
        for source, key in sources_and_keys[:-1]:
            value = source.get(key)
            if value:
                return value
        source, key = sources_and_keys[-1]
        return source.get(key)
    for source, key in sources_and_keys:
        value = source.get(key)
        if value is not None:
            return value
    return None


def _resolve_positive_targets(
    path: Path,
    line_number: int,
    record: dict[str, Any],
    target_ids: list[str],
) -> tuple[int, int, tuple[str, ...]]:
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
    positive_target_ids = tuple(
        str(value) for value in record.get("positive_target_ids", [])
    )
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
        raise ValueError(
            f"{path}:{line_number}: positive_target_ids omits a designated positive"
        )
    if len(set(positive_target_ids)) != len(positive_target_ids):
        raise ValueError(
            f"{path}:{line_number}: positive_target_ids contains duplicates"
        )
    missing_positive_candidates = set(positive_target_ids) - set(target_ids)
    if missing_positive_candidates:
        raise ValueError(
            f"{path}:{line_number}: positive_target_ids references missing candidates: "
            + ", ".join(sorted(missing_positive_candidates))
        )
    return direct_positive_index, evidence_positive_index, positive_target_ids


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
        positive_ids = _edge_positive_ids(
            path, line_number, record, candidate_ids, positive_index
        )
        examples.append(
            EdgeExample(
                query_id,
                tuple(candidate_ids),
                positive_index,
                dataset=str(record.get("dataset") or dataset_name or "default"),
                split=record.get("split"),
                teacher_logits=_teacher_logits(path, line_number, record, len(candidate_ids)),
                teacher_checkpoint_sha256=_provenance_value(
                    record,
                    metadata,
                    "teacher_checkpoint_sha256",
                    "teacher_checkpoint_sha256",
                    fallback_on_falsy=True,
                ),
                teacher_logit_mode=_provenance_value(
                    record,
                    metadata,
                    "teacher_logit_mode",
                    "teacher_edge_logit_mode",
                    "teacher_logit_mode",
                    fallback_on_falsy=True,
                ),
                teacher_ensemble_alpha=_provenance_value(
                    record,
                    metadata,
                    "teacher_ensemble_alpha",
                    "teacher_edge_ensemble_alpha",
                    fallback_on_falsy=False,
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
                positive_ids=positive_ids,
                confirmed_labels=_confirmed_edge_labels(
                    path, line_number, record, candidate_ids
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
        (
            direct_positive_index,
            evidence_positive_index,
            positive_target_ids,
        ) = _resolve_positive_targets(path, line_number, record, target_ids)
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
        if raw_score_config is None and metadata.get("training_score_config") is not None:
            raw_score_config = metadata["training_score_config"]
        if raw_score_config is None and {
            "evidence_aggregation",
            "evidence_top_k",
        } <= metadata.keys():
            raw_score_config = {
                "evidence_aggregation": metadata["evidence_aggregation"],
                "evidence_top_k": metadata["evidence_top_k"],
                "evidence_temperature": metadata.get("evidence_temperature", 1.0),
                "evidence_power": metadata.get("evidence_power", 2.0),
                "path_combination": metadata.get("path_combination", "sum"),
                "evidence_threshold": metadata.get("evidence_threshold", 0.0),
                "evidence_target_temperature": metadata.get(
                    "evidence_target_temperature", 1.0
                ),
                "row_support_model": metadata.get("row_support_model"),
                "row_support_model_sha256": metadata.get(
                    "row_support_model_sha256"
                ),
                "row_support_top_l": metadata.get("row_support_top_l", 20),
                "evidence_content_keys": metadata.get("evidence_content_keys"),
                "evidence_content_keys_sha256": metadata.get(
                    "evidence_content_keys_sha256"
                ),
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
                path_combination=str(raw_score_config.get("path_combination", "sum")),
                evidence_threshold=float(
                    raw_score_config.get("evidence_threshold", 0.0)
                ),
                evidence_target_temperature=float(
                    raw_score_config.get("evidence_target_temperature", 1.0)
                ),
                row_support_model=(
                    str(raw_score_config["row_support_model"])
                    if raw_score_config.get("row_support_model") is not None
                    else None
                ),
                row_support_model_sha256=(
                    str(raw_score_config["row_support_model_sha256"])
                    if raw_score_config.get("row_support_model_sha256") is not None
                    else None
                ),
                row_support_top_l=int(raw_score_config.get("row_support_top_l", 20)),
                evidence_content_keys=(
                    str(raw_score_config["evidence_content_keys"])
                    if raw_score_config.get("evidence_content_keys") is not None
                    else None
                ),
                evidence_content_keys_sha256=(
                    str(raw_score_config["evidence_content_keys_sha256"])
                    if raw_score_config.get("evidence_content_keys_sha256")
                    is not None
                    else None
                ),
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
            if teacher_score_config.path_combination not in PATH_COMBINATIONS:
                raise ValueError(f"{path}:{line_number}: invalid Teacher path combination")
            if not 0 <= teacher_score_config.evidence_threshold < 1:
                raise ValueError(
                    f"{path}:{line_number}: Teacher evidence_threshold must be in [0, 1)"
                )
            if teacher_score_config.evidence_target_temperature <= 0:
                raise ValueError(
                    f"{path}:{line_number}: Teacher evidence_target_temperature "
                    "must be positive"
                )
            if teacher_score_config.row_support_top_l <= 0:
                raise ValueError(
                    f"{path}:{line_number}: Teacher row_support_top_l must be positive"
                )
            if (
                teacher_score_config.evidence_aggregation == "greedy_row_support"
                and (
                    teacher_score_config.row_support_model is None
                    or teacher_score_config.row_support_model_sha256 is None
                    or teacher_score_config.evidence_content_keys is None
                    or teacher_score_config.evidence_content_keys_sha256 is None
                )
            ):
                raise ValueError(
                    f"{path}:{line_number}: G5 Teacher scores require row-support "
                    "and exact-content provenance"
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
                teacher_checkpoint_sha256=_provenance_value(
                    record,
                    metadata,
                    "teacher_checkpoint_sha256",
                    "teacher_checkpoint_sha256",
                    fallback_on_falsy=True,
                ),
                teacher_logit_mode=_provenance_value(
                    record,
                    metadata,
                    "teacher_logit_mode",
                    "teacher_target_logit_mode",
                    "teacher_logit_mode",
                    fallback_on_falsy=True,
                ),
                teacher_ensemble_alpha=_provenance_value(
                    record,
                    metadata,
                    "teacher_ensemble_alpha",
                    "teacher_target_ensemble_alpha",
                    fallback_on_falsy=False,
                ),
            )
        )
    if not examples:
        suffix = f" for split {split!r}" if split is not None else ""
        raise ValueError(f"{path}: no target examples{suffix}")
    return examples

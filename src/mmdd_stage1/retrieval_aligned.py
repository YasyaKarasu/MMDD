"""Build retrieval-aligned Teacher lists from frozen raw ANN neighbors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .features import normalize_object_type


class RawSearch(Protocol):
    def search(
        self, source_id: str, destination_type: str, k: int
    ) -> list[tuple[str, float]]: ...


def _raw_negatives(
    indices: RawSearch,
    query_id: str,
    destination_type: str,
    *,
    excluded: set[str],
    count: int,
) -> list[str]:
    if count <= 0:
        return []
    hits = indices.search(query_id, destination_type, max(100, count * 4))
    selected = []
    for object_id, _score in hits:
        if object_id in excluded:
            continue
        selected.append(object_id)
        excluded.add(object_id)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"{query_id}: raw {destination_type} index supplied only "
            f"{len(selected)} of {count} requested hard negatives"
        )
    return selected


def align_edge_record(
    record: dict[str, Any],
    indices: RawSearch,
    *,
    list_width: int,
    handcrafted_negatives: int,
    unavailable_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Keep a small handcrafted quota and fill an edge list from raw ANN."""

    if list_width < 2 or handcrafted_negatives < 0:
        raise ValueError("list_width must be at least 2 and quotas non-negative")
    query_id = str(record["query_id"])
    positive_id = str(record["positive_id"])
    destination_type = normalize_object_type(str(record["destination_type"]))
    original = [str(value) for value in record["candidate_ids"]]
    if positive_id not in original:
        raise ValueError(f"{query_id}: edge positive is absent from candidates")
    unavailable_ids = unavailable_ids or set()
    negatives = [
        value
        for value in original
        if value != positive_id and value not in unavailable_ids
    ]
    candidate_ids = [positive_id, *negatives[:handcrafted_negatives]]
    if len(candidate_ids) > list_width:
        raise ValueError("positive and handcrafted quotas exceed list_width")
    excluded = {query_id, *candidate_ids, *unavailable_ids}
    candidate_ids.extend(
        _raw_negatives(
            indices,
            query_id,
            destination_type,
            excluded=excluded,
            count=list_width - len(candidate_ids),
        )
    )
    return {
        "query_id": query_id,
        "source_type": normalize_object_type(str(record["source_type"])),
        "positive_id": positive_id,
        "candidate_ids": candidate_ids,
        "destination_type": destination_type,
        "dataset": str(record.get("dataset", "default")),
        "split": str(record["split"]),
    }


def _limited_evidence(
    evidence_ids: Sequence[str],
    object_type: Callable[[str], str],
    max_per_type: int,
    evidence_types: Sequence[str] = ("text", "image"),
) -> list[str]:
    counts = {evidence_type: 0 for evidence_type in evidence_types}
    selected = []
    for evidence_id in evidence_ids:
        evidence_type = normalize_object_type(object_type(evidence_id))
        if evidence_type not in counts or counts[evidence_type] >= max_per_type:
            continue
        selected.append(evidence_id)
        counts[evidence_type] += 1
    return selected


def align_target_record(
    record: dict[str, Any],
    indices: RawSearch,
    object_type: Callable[[str], str],
    *,
    list_width: int,
    handcrafted_negatives: int,
    evidence_per_type: int,
    unavailable_ids: set[str] | None = None,
    evidence_binding: str = "query-hard",
    target_evidence: dict[str, Sequence[str]] | None = None,
    evidence_types: Sequence[str] = ("text", "image"),
) -> dict[str, Any]:
    """Build a wide direct list and bind evidence to newly mined ANN targets."""

    if list_width < 2 or handcrafted_negatives < 0 or evidence_per_type < 0:
        raise ValueError("list_width must be at least 2 and quotas non-negative")
    if evidence_binding not in {"query-hard", "target-bound"}:
        raise ValueError("evidence_binding must be query-hard or target-bound")
    evidence_types = tuple(dict.fromkeys(evidence_types))
    if not evidence_types or any(
        evidence_type not in {"text", "image"}
        for evidence_type in evidence_types
    ):
        raise ValueError("evidence_types must contain text and/or image")
    unavailable_ids = unavailable_ids or set()
    query_id = str(record["query_id"])
    direct_positive = str(record["direct_positive_target_id"])
    evidence_positive = str(record["evidence_positive_target_id"])
    positive_ids = list(
        dict.fromkeys(
            [
                *[str(value) for value in record.get("positive_target_ids", [])],
                direct_positive,
                evidence_positive,
            ]
        )
    )
    original_candidates = {
        str(candidate["target_id"]): [
            str(value) for value in candidate.get("evidence_ids", [])
        ]
        for candidate in record["candidates"]
    }
    if direct_positive not in original_candidates or evidence_positive not in original_candidates:
        raise ValueError(f"{query_id}: designated target positive is absent")

    original_negatives = [
        target_id
        for target_id in original_candidates
        if target_id not in set(positive_ids) and target_id not in unavailable_ids
    ]
    target_ids = [*positive_ids, *original_negatives[:handcrafted_negatives]]
    if len(target_ids) > list_width:
        raise ValueError("positive and handcrafted quotas exceed list_width")
    excluded = {query_id, *target_ids, *unavailable_ids}
    target_ids.extend(
        _raw_negatives(
            indices,
            query_id,
            "table",
            excluded=excluded,
            count=list_width - len(target_ids),
        )
    )

    hard_evidence = []
    for evidence_type in evidence_types:
        if evidence_per_type:
            hard_evidence.extend(
                _raw_negatives(
                    indices,
                    query_id,
                    evidence_type,
                    excluded={query_id, *unavailable_ids},
                    count=evidence_per_type,
                )
            )
    candidates = []
    for target_id in target_ids:
        if target_id in original_candidates:
            evidence_ids = _limited_evidence(
                original_candidates[target_id],
                object_type,
                evidence_per_type,
                evidence_types,
            )
        elif evidence_binding == "target-bound":
            evidence_ids = _limited_evidence(
                (target_evidence or {}).get(target_id, ()),
                object_type,
                evidence_per_type,
                evidence_types,
            )
            present_types = {
                normalize_object_type(object_type(evidence_id))
                for evidence_id in evidence_ids
            }
            for evidence_type in evidence_types:
                if evidence_type in present_types or not evidence_per_type:
                    continue
                evidence_ids.extend(
                    _raw_negatives(
                        indices,
                        target_id,
                        evidence_type,
                        excluded={target_id, *unavailable_ids, *evidence_ids},
                        count=evidence_per_type,
                    )
                )
        else:
            evidence_ids = hard_evidence
        candidates.append(
            {"target_id": target_id, "evidence_ids": list(evidence_ids)}
        )
    return {
        "query_id": query_id,
        "direct_positive_target_id": direct_positive,
        "evidence_positive_target_id": evidence_positive,
        "positive_target_ids": positive_ids,
        "candidates": candidates,
        "dataset": str(record.get("dataset", "default")),
        "split": str(record["split"]),
    }

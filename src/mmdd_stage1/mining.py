"""Hard-negative refresh from the current Student retrieval distribution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .data import EdgeExample, TargetCandidate, TargetExample
from .features import FeatureStore
from .models import TeacherJoinabilityModel
from .objectives import PathAggregator
from .retrieval import StudentANNIndices
from .scoring import score_edge_batch, score_target_batch


@dataclass(frozen=True)
class HardPath:
    evidence_id: str
    target_id: str
    score: float


@dataclass(frozen=True)
class HardCandidateSet:
    """Target/path candidates plus independently mined Q->E negatives."""

    target_example: TargetExample
    evidence_negative_ids: tuple[str, ...]


def _known_positive_target_ids(example: TargetExample) -> tuple[str, ...]:
    direct_positive = example.candidates[example.direct_positive_index]
    evidence_positive = example.candidates[example.evidence_positive_index]
    return example.positive_target_ids or tuple(
        dict.fromkeys((direct_positive.target_id, evidence_positive.target_id))
    )


def build_hard_candidate_set(
    example: TargetExample,
    hard_target_ids: Sequence[str],
    hard_evidence_ids: Sequence[str],
    hard_paths: Sequence[HardPath],
    *,
    hard_targets_per_query: int,
) -> HardCandidateSet:
    """Merge the three independently ranked hard-negative pools."""

    if hard_targets_per_query <= 0:
        raise ValueError("hard_targets_per_query must be positive")

    direct_positive = example.candidates[example.direct_positive_index]
    evidence_positive = example.candidates[example.evidence_positive_index]
    known_positive_ids = _known_positive_target_ids(example)
    known_positives = set(known_positive_ids)

    selected_target_ids = []
    for target_id in hard_target_ids:
        target_id = str(target_id)
        if (
            target_id == example.query_id
            or target_id in known_positives
            or target_id in selected_target_ids
        ):
            continue
        selected_target_ids.append(target_id)
        if len(selected_target_ids) >= hard_targets_per_query:
            break

    path_evidence_by_target: dict[str, list[str]] = {}
    for path in hard_paths:
        if path.target_id == example.query_id or path.target_id in known_positives:
            continue
        evidence_ids = path_evidence_by_target.setdefault(path.target_id, [])
        if path.evidence_id not in evidence_ids:
            evidence_ids.append(path.evidence_id)

    selected_ids = set(selected_target_ids) | set(path_evidence_by_target)
    fallback_candidates = []
    for candidate in example.candidates:
        if len(selected_target_ids) + len(fallback_candidates) >= hard_targets_per_query:
            break
        if candidate.target_id in known_positives or candidate.target_id in selected_ids:
            continue
        fallback_candidates.append(candidate)
        selected_ids.add(candidate.target_id)

    negative_target_ids = tuple(
        dict.fromkeys(
            (
                *selected_target_ids,
                *path_evidence_by_target,
                *(candidate.target_id for candidate in fallback_candidates),
            )
        )
    )
    fallback_by_id = {candidate.target_id: candidate for candidate in fallback_candidates}
    candidates = list(dict.fromkeys((direct_positive, evidence_positive)))
    for target_id in negative_target_ids:
        if target_id in path_evidence_by_target:
            evidence_ids = tuple(path_evidence_by_target[target_id])
        elif target_id in fallback_by_id:
            evidence_ids = fallback_by_id[target_id].evidence_ids
        else:
            evidence_ids = ()
        candidates.append(TargetCandidate(target_id, evidence_ids))

    if len(candidates) < 2:
        raise ValueError(f"{example.query_id}: mining and existing data produced no negative target")

    target_example = TargetExample(
        query_id=example.query_id,
        candidates=tuple(candidates),
        direct_positive_index=candidates.index(direct_positive),
        evidence_positive_index=candidates.index(evidence_positive),
        dataset=example.dataset,
        split=example.split,
        positive_target_ids=known_positive_ids,
    )
    positive_evidence_ids = set(evidence_positive.evidence_ids)
    evidence_negative_ids = tuple(
        dict.fromkeys(
            evidence_id
            for evidence_id in hard_evidence_ids
            if evidence_id not in positive_evidence_ids
        )
    )
    return HardCandidateSet(target_example, evidence_negative_ids)


def _edge_examples_for_candidate_set(
    target_example: TargetExample,
    store: FeatureStore,
    evidence_negative_ids: Sequence[str] = (),
) -> list[EdgeExample]:
    """Expand one mined target list into its directly supervised path edges."""

    evidence_positive = target_example.candidates[target_example.evidence_positive_index]
    negative_target_ids = tuple(
        candidate.target_id
        for index, candidate in enumerate(target_example.candidates)
        if index != target_example.evidence_positive_index
    )
    positive_evidence_ids = tuple(dict.fromkeys(evidence_positive.evidence_ids))
    negative_evidence_by_type: dict[str, list[str]] = {}
    for evidence_id in evidence_negative_ids:
        evidence_type = store.get(evidence_id, include_hidden=False).object_type
        negative_evidence_by_type.setdefault(evidence_type, []).append(evidence_id)

    examples = [
        EdgeExample(
            query_id=target_example.query_id,
            candidate_ids=tuple(candidate.target_id for candidate in target_example.candidates),
            positive_index=target_example.direct_positive_index,
            dataset=target_example.dataset,
            split=target_example.split,
            source_type="table",
            destination_type="table",
        )
    ]
    for evidence_id in positive_evidence_ids:
        evidence_type = store.get(evidence_id, include_hidden=False).object_type
        negative_evidence_ids = negative_evidence_by_type.get(evidence_type, [])
        if negative_evidence_ids:
            examples.append(
                EdgeExample(
                    query_id=target_example.query_id,
                    candidate_ids=(evidence_id, *negative_evidence_ids),
                    positive_index=0,
                    dataset=target_example.dataset,
                    split=target_example.split,
                    source_type="table",
                    destination_type=evidence_type,
                )
            )
        examples.append(
            EdgeExample(
                query_id=evidence_id,
                candidate_ids=(evidence_positive.target_id, *negative_target_ids),
                positive_index=0,
                dataset=target_example.dataset,
                split=target_example.split,
                source_type=evidence_type,
                destination_type="table",
            )
        )
    return examples


def _target_record(example: TargetExample) -> dict[str, Any]:
    record = {
        "query_id": example.query_id,
        "direct_positive_target_id": example.candidates[
            example.direct_positive_index
        ].target_id,
        "evidence_positive_target_id": example.candidates[
            example.evidence_positive_index
        ].target_id,
        "positive_target_ids": list(example.positive_target_ids),
        "candidates": [
            {
                "target_id": candidate.target_id,
                "evidence_ids": list(candidate.evidence_ids),
            }
            for candidate in example.candidates
        ],
        "dataset": example.dataset,
    }
    if example.split is not None:
        record["split"] = example.split
    return record


def _edge_record(example: EdgeExample) -> dict[str, Any]:
    record = {
        "query_id": example.query_id,
        "source_type": example.source_type,
        "positive_id": example.candidate_ids[example.positive_index],
        "candidate_ids": list(example.candidate_ids),
        "destination_type": example.destination_type,
        "dataset": example.dataset,
    }
    if example.split is not None:
        record["split"] = example.split
    return record


def hard_candidate_records(
    candidate_sets: Sequence[HardCandidateSet],
    store: FeatureStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Serialize mined candidates without Teacher logits for tier supplementation."""

    target_records = []
    edge_records = []
    for item in candidate_sets:
        example = item.target_example
        target_records.append(_target_record(example))
        for edge in _edge_examples_for_candidate_set(
            example, store, item.evidence_negative_ids
        ):
            edge_records.append(_edge_record(edge))
    return target_records, edge_records


def retrieve_hard_candidate_sets(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices,
    *,
    hard_targets_per_query: int,
    hard_evidence_per_type: int,
    hard_paths_per_query: int,
    direct_k: int,
    evidence_k: int,
    targets_per_evidence: int,
    evidence_types: tuple[str, ...] = ("text", "image"),
) -> list[HardCandidateSet]:
    if hard_targets_per_query <= 0:
        raise ValueError("hard_targets_per_query must be positive")
    if hard_evidence_per_type < 0 or hard_paths_per_query < 0:
        raise ValueError("Hard-evidence and hard-path sizes must be non-negative")
    if min(direct_k, evidence_k, targets_per_evidence) < 0:
        raise ValueError("ANN search sizes must be non-negative")

    candidate_sets = []
    for example in examples:
        known_positives = set(_known_positive_target_ids(example))
        hard_target_ids = [
            target_id
            for target_id, _score in indices.search(example.query_id, "table", direct_k)
        ]

        evidence_positive = example.candidates[example.evidence_positive_index]
        positive_evidence_ids = set(evidence_positive.evidence_ids)
        hard_evidence_ids = []
        hard_paths = []
        for evidence_type in dict.fromkeys(evidence_types):
            evidence_hits = indices.search(example.query_id, evidence_type, evidence_k)
            evidence_negatives_for_type = 0
            for evidence_id, query_evidence_score in evidence_hits:
                if (
                    evidence_id not in positive_evidence_ids
                    and evidence_negatives_for_type < hard_evidence_per_type
                ):
                    hard_evidence_ids.append(evidence_id)
                    evidence_negatives_for_type += 1

                for target_id, evidence_target_score in indices.search(
                    evidence_id, "table", targets_per_evidence
                ):
                    if target_id == example.query_id or target_id in known_positives:
                        continue
                    hard_paths.append(
                        HardPath(
                            evidence_id,
                            target_id,
                            float(query_evidence_score) + float(evidence_target_score),
                        )
                    )

        hard_paths.sort(key=lambda path: (-path.score, path.target_id, path.evidence_id))
        candidate_sets.append(
            build_hard_candidate_set(
                example,
                hard_target_ids,
                hard_evidence_ids,
                hard_paths[:hard_paths_per_query],
                hard_targets_per_query=hard_targets_per_query,
            )
        )
    return candidate_sets


@torch.no_grad()
def score_hard_candidate_sets(
    candidate_sets: Sequence[HardCandidateSet],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    aggregator: PathAggregator,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    teacher.eval()
    target_records = []
    edge_records = []
    for start in range(0, len(candidate_sets), batch_size):
        mined_batch = candidate_sets[start : start + batch_size]
        batch = [item.target_example for item in mined_batch]
        edge_examples_by_item = [
            _edge_examples_for_candidate_set(
                item.target_example,
                store,
                item.evidence_negative_ids,
            )
            for item in mined_batch
        ]
        edge_examples = [
            example for item_examples in edge_examples_by_item for example in item_examples
        ]
        teacher_targets = score_target_batch(teacher, batch, store, device, aggregator)
        teacher_edges = score_edge_batch(teacher, edge_examples, store, device)
        edge_offset = 0
        for index, (item, item_edge_examples) in enumerate(zip(batch, edge_examples_by_item)):
            target_count = len(item.candidates)
            target_record = _target_record(item)
            target_record.update(
                teacher_direct_logits=teacher_targets.direct.logits[
                    index, :target_count
                ].cpu().tolist(),
                teacher_evidence_logits=teacher_targets.evidence.logits[
                    index, :target_count
                ].cpu().tolist(),
            )
            target_records.append(target_record)
            for edge_example in item_edge_examples:
                edge_count = len(edge_example.candidate_ids)
                edge_record = _edge_record(edge_example)
                edge_record["teacher_logits"] = teacher_edges.logits[
                    edge_offset, :edge_count
                ].cpu().tolist()
                edge_records.append(edge_record)
                edge_offset += 1
    return target_records, edge_records

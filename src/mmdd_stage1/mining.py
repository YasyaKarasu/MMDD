"""Hard-negative refresh from the current Student retrieval distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .data import EdgeExample, TargetCandidate, TargetExample
from .features import FeatureStore
from .models import TeacherJoinabilityModel
from .objectives import PathAggregator
from .retrieval import StudentANNIndices, retrieve_zero_one_hop
from .scoring import score_edge_batch, score_target_batch


@dataclass(frozen=True)
class HardCandidateSet:
    target_example: TargetExample


def _path_evidence_ids(result: dict[str, Any], max_evidence: int) -> tuple[str, ...]:
    evidence_ids = []
    evidence_paths = [path for path in result.get("paths", []) if path.get("evidence_id") is not None]
    for path in sorted(evidence_paths, key=lambda item: float(item["path_score"]), reverse=True):
        evidence_id = str(path["evidence_id"])
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
        if len(evidence_ids) >= max_evidence:
            break
    return tuple(evidence_ids)


def build_hard_candidate_set(
    example: TargetExample,
    retrieval_results: Sequence[dict[str, Any]],
    *,
    hard_targets_per_query: int,
    max_evidence_per_target: int,
) -> HardCandidateSet:
    """Exclude all GT targets and retain the highest-ranked Student errors."""

    if hard_targets_per_query <= 0:
        raise ValueError("hard_targets_per_query must be positive")
    if max_evidence_per_target < 0:
        raise ValueError("max_evidence_per_target must be non-negative")
    positive = example.candidates[example.positive_index]
    known_positive_ids = example.positive_target_ids or (positive.target_id,)
    known_positives = set(known_positive_ids)
    selected_results = []
    selected_ids = set()
    for result in retrieval_results:
        target_id = str(result["target_id"])
        if target_id == example.query_id or target_id in known_positives or target_id in selected_ids:
            continue
        selected_results.append(result)
        selected_ids.add(target_id)
        if len(selected_results) >= hard_targets_per_query:
            break

    original_negatives = [
        candidate
        for candidate in example.candidates
        if candidate.target_id not in known_positives and candidate.target_id not in selected_ids
    ]
    candidates = [positive]
    for result in selected_results:
        target_id = str(result["target_id"])
        evidence_ids = _path_evidence_ids(result, max_evidence_per_target)
        candidates.append(TargetCandidate(target_id, evidence_ids))

    for candidate in original_negatives:
        if len(candidates) - 1 >= hard_targets_per_query:
            break
        candidates.append(candidate)
    if len(candidates) < 2:
        raise ValueError(f"{example.query_id}: retrieval and existing data produced no negative target")

    target_example = TargetExample(
        query_id=example.query_id,
        candidates=tuple(candidates),
        positive_index=0,
        dataset=example.dataset,
        split=example.split,
        positive_target_ids=known_positive_ids,
    )
    return HardCandidateSet(target_example)


def _edge_examples_for_candidate_set(
    candidate_set: HardCandidateSet,
    store: FeatureStore,
) -> list[EdgeExample]:
    """Expand one mined target list into its directly supervised path edges."""

    target_example = candidate_set.target_example
    positive = target_example.candidates[target_example.positive_index]
    negative_candidates = [
        candidate
        for index, candidate in enumerate(target_example.candidates)
        if index != target_example.positive_index
    ]
    negative_target_ids = tuple(candidate.target_id for candidate in negative_candidates)
    positive_evidence_ids = tuple(dict.fromkeys(positive.evidence_ids))
    positive_evidence_set = set(positive_evidence_ids)
    negative_evidence_by_type: dict[str, list[str]] = {}
    for candidate in negative_candidates:
        for evidence_id in candidate.evidence_ids:
            if evidence_id in positive_evidence_set:
                continue
            evidence_type = store.get(evidence_id).object_type
            values = negative_evidence_by_type.setdefault(evidence_type, [])
            if evidence_id not in values:
                values.append(evidence_id)

    examples = [
        EdgeExample(
            query_id=target_example.query_id,
            candidate_ids=tuple(candidate.target_id for candidate in target_example.candidates),
            positive_index=target_example.positive_index,
            dataset=target_example.dataset,
            split=target_example.split,
            source_type="table",
            destination_type="table",
        )
    ]
    for evidence_id in positive_evidence_ids:
        evidence_type = store.get(evidence_id).object_type
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
                candidate_ids=(positive.target_id, *negative_target_ids),
                positive_index=0,
                dataset=target_example.dataset,
                split=target_example.split,
                source_type=evidence_type,
                destination_type="table",
            )
        )
    return examples


def retrieve_hard_candidate_sets(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices,
    *,
    hard_targets_per_query: int,
    max_evidence_per_target: int,
    retrieval_k: int,
    direct_k: int,
    evidence_k: int,
    targets_per_evidence: int,
    evidence_types: tuple[str, ...] = ("text", "image"),
    evidence_aggregation: str = "logsumexp",
    evidence_top_k: int = 4,
    rrf_k: int = 60,
) -> list[HardCandidateSet]:
    candidate_sets = []
    for example in examples:
        results = retrieve_zero_one_hop(
            example.query_id,
            indices,
            direct_k=direct_k,
            evidence_k=evidence_k,
            targets_per_evidence=targets_per_evidence,
            result_k=retrieval_k,
            evidence_types=evidence_types,
            evidence_aggregation=evidence_aggregation,
            evidence_top_k=evidence_top_k,
            rrf_k=rrf_k,
        )
        candidate_sets.append(
            build_hard_candidate_set(
                example,
                results,
                hard_targets_per_query=hard_targets_per_query,
                max_evidence_per_target=max_evidence_per_target,
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
        batch = candidate_sets[start : start + batch_size]
        target_examples = [item.target_example for item in batch]
        edge_examples_by_item = [
            _edge_examples_for_candidate_set(item, store) for item in batch
        ]
        edge_examples = [
            example for item_examples in edge_examples_by_item for example in item_examples
        ]
        teacher_targets = score_target_batch(teacher, target_examples, store, device, aggregator)
        teacher_edges = score_edge_batch(teacher, edge_examples, store, device)
        edge_offset = 0
        for index, (item, item_edge_examples) in enumerate(zip(batch, edge_examples_by_item)):
            target_count = len(item.target_example.candidates)
            target_record = {
                "query_id": item.target_example.query_id,
                "positive_target_id": item.target_example.candidates[0].target_id,
                "positive_target_ids": list(item.target_example.positive_target_ids),
                "candidates": [
                    {
                        "target_id": candidate.target_id,
                        "evidence_ids": list(candidate.evidence_ids),
                    }
                    for candidate in item.target_example.candidates
                ],
                "teacher_logits": teacher_targets.logits[index, :target_count].cpu().tolist(),
                "dataset": item.target_example.dataset,
            }
            if item.target_example.split is not None:
                target_record["split"] = item.target_example.split
            target_records.append(target_record)
            for edge_example in item_edge_examples:
                edge_count = len(edge_example.candidate_ids)
                edge_record = {
                    "query_id": edge_example.query_id,
                    "source_type": edge_example.source_type,
                    "positive_id": edge_example.candidate_ids[edge_example.positive_index],
                    "candidate_ids": list(edge_example.candidate_ids),
                    "destination_type": edge_example.destination_type,
                    "teacher_logits": teacher_edges.logits[
                        edge_offset, :edge_count
                    ].cpu().tolist(),
                    "dataset": edge_example.dataset,
                }
                if edge_example.split is not None:
                    edge_record["split"] = edge_example.split
                edge_records.append(edge_record)
                edge_offset += 1
    return target_records, edge_records

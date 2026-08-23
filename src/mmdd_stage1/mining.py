"""Hard-negative refresh from the current Student retrieval distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .data import EdgeExample, TargetCandidate, TargetExample
from .features import FeatureStore
from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import PathAggregator
from .retrieval import StudentANNIndices, retrieve_zero_one_hop
from .scoring import score_edge_batch, score_target_batch


@dataclass(frozen=True)
class HardCandidateSet:
    target_example: TargetExample
    edge_example: EdgeExample
    target_candidates: tuple[dict[str, Any], ...]
    hard_target_ids: tuple[str, ...]
    hard_evidence_ids: tuple[str, ...]
    hard_path_count: int


def _path_evidence_ids(result: dict[str, Any], max_evidence: int) -> tuple[str, ...]:
    evidence_ids = []
    for path in sorted(result.get("paths", []), key=lambda item: float(item["path_score"]), reverse=True):
        evidence_id = path.get("evidence_id")
        if evidence_id is not None and evidence_id not in evidence_ids:
            evidence_ids.append(str(evidence_id))
        if len(evidence_ids) >= max_evidence:
            break
    return tuple(evidence_ids)


def build_hard_candidate_set(
    example: TargetExample,
    retrieval_results: Sequence[dict[str, Any]],
    *,
    hard_targets_per_query: int,
    hard_evidence_per_query: int,
    max_evidence_per_target: int,
) -> HardCandidateSet:
    """Exclude all GT targets and retain the highest-ranked Student errors."""

    if hard_targets_per_query <= 0:
        raise ValueError("hard_targets_per_query must be positive")
    if hard_evidence_per_query < 0 or max_evidence_per_target < 0:
        raise ValueError("Evidence limits must be non-negative")
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
    candidate_records = [
        {
            "target_id": positive.target_id,
            "evidence_ids": list(positive.evidence_ids),
            "negative_source": None,
        }
    ]
    candidates = [positive]
    hard_evidence_ids = []
    hard_path_count = 0
    for result in selected_results:
        target_id = str(result["target_id"])
        evidence_ids = _path_evidence_ids(result, max_evidence_per_target)
        hard_path_count += sum(path.get("kind") == "evidence" for path in result.get("paths", []))
        for evidence_id in evidence_ids:
            if evidence_id not in hard_evidence_ids and len(hard_evidence_ids) < hard_evidence_per_query:
                hard_evidence_ids.append(evidence_id)
        candidates.append(TargetCandidate(target_id, evidence_ids))
        candidate_records.append(
            {
                "target_id": target_id,
                "evidence_ids": list(evidence_ids),
                "negative_source": "student_ann",
                "student_retrieval_score": float(result["score"]),
                "retrieval_paths": result.get("paths", []),
            }
        )

    for candidate in original_negatives:
        if len(candidates) - 1 >= hard_targets_per_query:
            break
        candidates.append(candidate)
        candidate_records.append(
            {
                "target_id": candidate.target_id,
                "evidence_ids": list(candidate.evidence_ids),
                "negative_source": "fallback_existing",
            }
        )
    if len(candidates) < 2:
        raise ValueError(f"{example.query_id}: retrieval and existing data produced no negative target")

    edge_candidate_ids = [positive.target_id]
    for object_id in (*hard_evidence_ids, *(candidate.target_id for candidate in candidates[1:])):
        if object_id not in edge_candidate_ids:
            edge_candidate_ids.append(object_id)
    target_example = TargetExample(
        query_id=example.query_id,
        candidates=tuple(candidates),
        positive_index=0,
        dataset=example.dataset,
        split=example.split,
        positive_target_ids=known_positive_ids,
    )
    edge_example = EdgeExample(
        query_id=example.query_id,
        candidate_ids=tuple(edge_candidate_ids),
        positive_index=0,
        dataset=example.dataset,
        split=example.split,
    )
    return HardCandidateSet(
        target_example=target_example,
        edge_example=edge_example,
        target_candidates=tuple(candidate_records),
        hard_target_ids=tuple(str(result["target_id"]) for result in selected_results),
        hard_evidence_ids=tuple(hard_evidence_ids),
        hard_path_count=hard_path_count,
    )


def retrieve_hard_candidate_sets(
    examples: Sequence[TargetExample],
    indices: StudentANNIndices,
    *,
    hard_targets_per_query: int,
    hard_evidence_per_query: int,
    max_evidence_per_target: int,
    retrieval_k: int,
    direct_k: int,
    evidence_k: int,
    targets_per_evidence: int,
    evidence_types: tuple[str, ...] = ("text", "image"),
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
        )
        candidate_sets.append(
            build_hard_candidate_set(
                example,
                results,
                hard_targets_per_query=hard_targets_per_query,
                hard_evidence_per_query=hard_evidence_per_query,
                max_evidence_per_target=max_evidence_per_target,
            )
        )
    return candidate_sets


@torch.no_grad()
def score_hard_candidate_sets(
    candidate_sets: Sequence[HardCandidateSet],
    teacher: TeacherJoinabilityModel,
    student: StudentJoinabilityModel,
    store: FeatureStore,
    aggregator: PathAggregator,
    *,
    device: torch.device,
    batch_size: int,
    mining_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    teacher.eval()
    student.eval()
    target_records = []
    edge_records = []
    for start in range(0, len(candidate_sets), batch_size):
        batch = candidate_sets[start : start + batch_size]
        target_examples = [item.target_example for item in batch]
        edge_examples = [item.edge_example for item in batch]
        teacher_targets = score_target_batch(teacher, target_examples, store, device, aggregator)
        student_targets = score_target_batch(student, target_examples, store, device, aggregator)
        teacher_edges = score_edge_batch(teacher, edge_examples, store, device)
        student_edges = score_edge_batch(student, edge_examples, store, device)
        for index, item in enumerate(batch):
            target_count = len(item.target_example.candidates)
            edge_count = len(item.edge_example.candidate_ids)
            metadata = {
                "hard_target_ids": list(item.hard_target_ids),
                "hard_evidence_ids": list(item.hard_evidence_ids),
                "hard_path_count": item.hard_path_count,
                **(mining_metadata or {}),
            }
            target_record = {
                "query_id": item.target_example.query_id,
                "positive_target_id": item.target_example.candidates[0].target_id,
                "positive_target_ids": list(item.target_example.positive_target_ids),
                "candidates": list(item.target_candidates),
                "teacher_logits": teacher_targets.logits[index, :target_count].cpu().tolist(),
                "student_logits": student_targets.logits[index, :target_count].cpu().tolist(),
                "teacher_score_config": {
                    "evidence_aggregation": aggregator.evidence_aggregation,
                    "evidence_top_k": aggregator.top_k,
                },
                "dataset": item.target_example.dataset,
                "mining": metadata,
            }
            edge_record = {
                "query_id": item.edge_example.query_id,
                "positive_id": item.edge_example.candidate_ids[0],
                "candidate_ids": list(item.edge_example.candidate_ids),
                "teacher_logits": teacher_edges.logits[index, :edge_count].cpu().tolist(),
                "student_logits": student_edges.logits[index, :edge_count].cpu().tolist(),
                "dataset": item.edge_example.dataset,
                "mining": metadata,
            }
            teacher_sha256 = metadata.get("teacher_checkpoint_sha256")
            if teacher_sha256 is not None:
                target_record["teacher_checkpoint_sha256"] = teacher_sha256
                edge_record["teacher_checkpoint_sha256"] = teacher_sha256
            if item.target_example.split is not None:
                target_record["split"] = item.target_example.split
                edge_record["split"] = item.target_example.split
            target_records.append(target_record)
            edge_records.append(edge_record)
    return target_records, edge_records

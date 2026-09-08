"""Hard-negative refresh from the current Student retrieval distribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from mmdd_progress import progress

from .data import EdgeExample, TargetCandidate, TargetExample
from .features import FeatureStore
from .models import TeacherJoinabilityModel
from .objectives import PathAggregator
from .retrieval import StudentANNIndices
from .scoring import score_edge_batch, score_target_batch
from .teacher_logits import (
    _ensemble_list_scores,
    _raw_edge_scores,
    _score_target_ensemble_logits,
)


@dataclass(frozen=True)
class HardPath:
    evidence_id: str
    target_id: str
    score: float


@dataclass(frozen=True)
class HardCandidateSet:
    """Merged candidates plus provenance for independently mined edge pools."""

    target_example: TargetExample
    evidence_negative_ids: tuple[str, ...]
    pool_counts: tuple[tuple[str, int], ...] = ()
    direct_target_negative_ids: tuple[str, ...] = ()
    evidence_target_negative_ids: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _known_positive_target_ids(example: TargetExample) -> tuple[str, ...]:
    direct_positive = example.candidates[example.direct_positive_index]
    evidence_positive = example.candidates[example.evidence_positive_index]
    return example.positive_target_ids or tuple(
        dict.fromkeys((direct_positive.target_id, evidence_positive.target_id))
    )


def _known_positive_candidates(example: TargetExample) -> tuple[TargetCandidate, ...]:
    positive_ids = _known_positive_target_ids(example)
    candidates_by_id = {
        candidate.target_id: candidate
        for candidate in example.candidates
        if candidate.target_id in positive_ids
    }
    missing = set(positive_ids) - set(candidates_by_id)
    if missing:
        raise ValueError(
            f"{example.query_id}: positive targets are absent from candidates: "
            f"{sorted(missing)}"
        )
    return tuple(candidates_by_id[target_id] for target_id in positive_ids)


def build_hard_candidate_set(
    example: TargetExample,
    hard_target_ids: Sequence[str],
    hard_evidence_ids: Sequence[str],
    hard_paths: Sequence[HardPath],
    *,
    hard_targets_per_query: int,
    hard_evidence_target_ids: Mapping[str, Sequence[str]] | None = None,
) -> HardCandidateSet:
    """Merge independently ranked Q-T, Q-E, E-T, and complete-path pools."""

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

    evidence_target_ids_by_evidence: list[tuple[str, tuple[str, ...]]] = []
    evidence_path_by_target: dict[str, list[str]] = {}
    for evidence_id, target_ids in (hard_evidence_target_ids or {}).items():
        selected = []
        for target_id in target_ids:
            target_id = str(target_id)
            if (
                target_id == example.query_id
                or target_id in known_positives
                or target_id in selected
            ):
                continue
            selected.append(target_id)
            evidence_ids = evidence_path_by_target.setdefault(target_id, [])
            if evidence_id not in evidence_ids:
                evidence_ids.append(str(evidence_id))
        evidence_target_ids_by_evidence.append((str(evidence_id), tuple(selected)))

    path_evidence_by_target: dict[str, list[str]] = {}
    for path in hard_paths:
        if path.target_id == example.query_id or path.target_id in known_positives:
            continue
        evidence_ids = path_evidence_by_target.setdefault(path.target_id, [])
        if path.evidence_id not in evidence_ids:
            evidence_ids.append(path.evidence_id)

    selected_ids = (
        set(selected_target_ids)
        | set(evidence_path_by_target)
        | set(path_evidence_by_target)
    )
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
                *evidence_path_by_target,
                *path_evidence_by_target,
                *(candidate.target_id for candidate in fallback_candidates),
            )
        )
    )
    fallback_by_id = {candidate.target_id: candidate for candidate in fallback_candidates}
    candidates = list(_known_positive_candidates(example))
    for target_id in negative_target_ids:
        if target_id in evidence_path_by_target or target_id in path_evidence_by_target:
            evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *evidence_path_by_target.get(target_id, ()),
                        *path_evidence_by_target.get(target_id, ()),
                    )
                )
            )
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
    base_negative_ids = {
        candidate.target_id
        for candidate in example.candidates
        if candidate.target_id not in known_positives
    }
    direct_target_set = set(selected_target_ids)
    evidence_target_set = set(evidence_path_by_target)
    path_target_set = set(path_evidence_by_target)
    pool_counts = [
        ("direct_target_candidates", len(selected_target_ids)),
        ("evidence_candidates", len(evidence_negative_ids)),
    ]
    if hard_evidence_target_ids:
        pool_counts.extend(
            [
                (
                    "evidence_target_candidates",
                    sum(len(target_ids) for _, target_ids in evidence_target_ids_by_evidence),
                ),
                ("evidence_target_unique_targets", len(evidence_target_set)),
                (
                    "direct_evidence_target_overlap",
                    len(direct_target_set & evidence_target_set),
                ),
                (
                    "evidence_target_path_overlap",
                    len(evidence_target_set & path_target_set),
                ),
            ]
        )
    pool_counts.extend(
        [
            ("path_target_candidates", len(path_evidence_by_target)),
            ("fallback_target_candidates", len(fallback_candidates)),
            ("direct_path_overlap", len(direct_target_set & path_target_set)),
            (
                "base_negative_overlap",
                len(set(negative_target_ids) & base_negative_ids),
            ),
            ("merged_negative_targets", len(negative_target_ids)),
        ]
    )
    return HardCandidateSet(
        target_example,
        evidence_negative_ids,
        tuple(pool_counts),
        tuple(
            (
                *selected_target_ids,
                *(candidate.target_id for candidate in fallback_candidates),
            )
        ),
        tuple(evidence_target_ids_by_evidence),
    )


def summarize_hard_candidate_sets(
    candidate_sets: Sequence[HardCandidateSet],
) -> dict[str, int | float]:
    """Aggregate the mining pools and their target-level deduplication."""

    totals: dict[str, int] = {}
    for candidate_set in candidate_sets:
        for key, value in candidate_set.pool_counts:
            totals[key] = totals.get(key, 0) + value
    raw_target_candidates = sum(
        totals.get(key, 0)
        for key in (
            "direct_target_candidates",
            "evidence_target_candidates",
            "path_target_candidates",
            "fallback_target_candidates",
        )
    )
    merged = totals.get("merged_negative_targets", 0)
    totals.update(
        {
            "queries": len(candidate_sets),
            "raw_target_candidates": raw_target_candidates,
            "target_candidates_removed_by_dedup": raw_target_candidates - merged,
        }
    )
    return {
        **totals,
        "target_pool_dedup_rate": (
            (raw_target_candidates - merged) / raw_target_candidates
            if raw_target_candidates
            else 0.0
        ),
        "merged_targets_overlapping_base_rate": (
            totals.get("base_negative_overlap", 0) / merged if merged else 0.0
        ),
    }


def _edge_examples_for_candidate_set(
    target_example: TargetExample,
    store: FeatureStore,
    evidence_negative_ids: Sequence[str] = (),
    direct_target_negative_ids: Sequence[str] = (),
    evidence_target_negative_ids: Sequence[tuple[str, Sequence[str]]] = (),
) -> list[EdgeExample]:
    """Expand one mined target list into its directly supervised path edges."""

    evidence_positive = target_example.candidates[target_example.evidence_positive_index]
    known_positives = set(_known_positive_target_ids(target_example))
    all_negative_target_ids = tuple(
        candidate.target_id
        for candidate in target_example.candidates
        if candidate.target_id not in known_positives
    )
    direct_negative_ids = tuple(direct_target_negative_ids) or all_negative_target_ids
    evidence_target_map = {
        evidence_id: tuple(target_ids)
        for evidence_id, target_ids in evidence_target_negative_ids
    }
    positive_evidence_ids = tuple(dict.fromkeys(evidence_positive.evidence_ids))
    negative_evidence_by_type: dict[str, list[str]] = {}
    for evidence_id in evidence_negative_ids:
        evidence_type = store.embedding_features(evidence_id).object_type
        negative_evidence_by_type.setdefault(evidence_type, []).append(evidence_id)

    examples = [
        EdgeExample(
            query_id=target_example.query_id,
            candidate_ids=(
                target_example.candidates[
                    target_example.direct_positive_index
                ].target_id,
                *(
                    target_id
                    for target_id in _known_positive_target_ids(target_example)
                    if target_id
                    != target_example.candidates[
                        target_example.direct_positive_index
                    ].target_id
                ),
                *direct_negative_ids,
            ),
            positive_index=0,
            dataset=target_example.dataset,
            split=target_example.split,
            source_type="table",
            destination_type="table",
            positive_ids=_known_positive_target_ids(target_example),
        )
    ]
    for evidence_id in positive_evidence_ids:
        evidence_type = store.embedding_features(evidence_id).object_type
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
                    positive_ids=(evidence_id,),
                    confirmed_labels=(1, *([None] * len(negative_evidence_ids))),
                )
            )
        evidence_target_negatives = evidence_target_map.get(
            evidence_id, all_negative_target_ids
        )
        if evidence_target_negatives:
            examples.append(
                EdgeExample(
                    query_id=evidence_id,
                    candidate_ids=(
                        evidence_positive.target_id,
                        *evidence_target_negatives,
                    ),
                    positive_index=0,
                    dataset=target_example.dataset,
                    split=target_example.split,
                    source_type=evidence_type,
                    destination_type="table",
                    positive_ids=(evidence_positive.target_id,),
                    confirmed_labels=(
                        1,
                        *([None] * len(evidence_target_negatives)),
                    ),
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
        "positive_ids": list(
            example.positive_ids
            or (example.candidate_ids[example.positive_index],)
        ),
        "candidate_ids": list(example.candidate_ids),
        "destination_type": example.destination_type,
        "dataset": example.dataset,
    }
    if example.split is not None:
        record["split"] = example.split
    if example.confirmed_labels is not None:
        record["confirmed_labels"] = list(example.confirmed_labels)
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
            example,
            store,
            item.evidence_negative_ids,
            item.direct_target_negative_ids,
            item.evidence_target_negative_ids,
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
    hard_targets_per_positive_evidence: int = 0,
    direct_k: int,
    evidence_k: int,
    targets_per_evidence: int,
    evidence_types: tuple[str, ...] = ("text", "image"),
) -> list[HardCandidateSet]:
    if hard_targets_per_query <= 0:
        raise ValueError("hard_targets_per_query must be positive")
    if min(
        hard_evidence_per_type,
        hard_paths_per_query,
        hard_targets_per_positive_evidence,
    ) < 0:
        raise ValueError("Hard-evidence, E-T, and hard-path sizes must be non-negative")
    if min(direct_k, evidence_k, targets_per_evidence) < 0:
        raise ValueError("ANN search sizes must be non-negative")

    candidate_sets = []
    for example in progress(
        examples, desc="Mine hard negatives", unit="query", leave=False
    ):
        known_positives = set(_known_positive_target_ids(example))
        hard_target_ids = [
            target_id
            for target_id, _score in indices.search(example.query_id, "table", direct_k)
        ]

        evidence_positive = example.candidates[example.evidence_positive_index]
        positive_evidence_ids = set(evidence_positive.evidence_ids)
        evidence_target_ids: dict[str, list[str]] = {}
        if hard_targets_per_positive_evidence:
            positive_ids = tuple(dict.fromkeys(evidence_positive.evidence_ids))
            positive_target_hits = indices.search_many(
                positive_ids,
                "table",
                targets_per_evidence,
            )
            for evidence_id, hits in zip(
                positive_ids, positive_target_hits, strict=True
            ):
                selected = []
                for target_id, _score in hits:
                    if (
                        target_id == example.query_id
                        or target_id in known_positives
                        or target_id in selected
                    ):
                        continue
                    selected.append(target_id)
                    if len(selected) == hard_targets_per_positive_evidence:
                        break
                evidence_target_ids[evidence_id] = selected
        hard_evidence_ids = []
        hard_paths = []
        all_evidence_hits = []
        for evidence_type in dict.fromkeys(evidence_types):
            evidence_hits = indices.search(example.query_id, evidence_type, evidence_k)
            evidence_negatives_for_type = 0
            for evidence_id, query_evidence_score in evidence_hits:
                all_evidence_hits.append((evidence_id, query_evidence_score))
                if (
                    evidence_id not in positive_evidence_ids
                    and evidence_negatives_for_type < hard_evidence_per_type
                ):
                    hard_evidence_ids.append(evidence_id)
                    evidence_negatives_for_type += 1

        target_hits = indices.search_many(
            [evidence_id for evidence_id, _score in all_evidence_hits],
            "table",
            targets_per_evidence,
        )
        for (evidence_id, query_evidence_score), evidence_targets in zip(
            all_evidence_hits, target_hits
        ):
            for target_id, evidence_target_score in evidence_targets:
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
                hard_evidence_target_ids=evidence_target_ids,
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
    ensemble_alpha: float | None = None,
    teacher_score_space: str = "raw_logit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ensemble_alpha is not None and teacher_score_space != "raw_logit":
        raise ValueError("Teacher/raw ensembles require raw Teacher logits")
    teacher.eval()
    target_records = []
    edge_records = []
    starts = range(0, len(candidate_sets), batch_size)
    for start in progress(
        starts,
        total=len(starts),
        desc="Teacher hard-negative scoring",
        unit="batch",
        leave=False,
    ):
        mined_batch = candidate_sets[start : start + batch_size]
        batch = [item.target_example for item in mined_batch]
        edge_examples_by_item = [
            _edge_examples_for_candidate_set(
                item.target_example,
                store,
                item.evidence_negative_ids,
                item.direct_target_negative_ids,
                item.evidence_target_negative_ids,
            )
            for item in mined_batch
        ]
        edge_examples = [
            example for item_examples in edge_examples_by_item for example in item_examples
        ]
        teacher_targets = (
            _score_target_ensemble_logits(
                batch, teacher, store, device, aggregator, ensemble_alpha
            )
            if ensemble_alpha is not None
            else score_target_batch(
                teacher,
                batch,
                store,
                device,
                aggregator,
                student_score_space=teacher_score_space,
            )
        )
        teacher_edges = score_edge_batch(
            teacher,
            edge_examples,
            store,
            device,
            student_score_space=teacher_score_space,
        )
        edge_scores = (
            _ensemble_list_scores(
                _raw_edge_scores(edge_examples, store, device),
                teacher_edges,
                ensemble_alpha,
            )
            if ensemble_alpha is not None
            else teacher_edges
        )
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
                teacher_score_config=aggregator.config(),
            )
            if ensemble_alpha is not None:
                target_record.update(
                    teacher_logit_mode="ensemble",
                    teacher_ensemble_alpha=ensemble_alpha,
                )
            elif teacher_score_space != "raw_logit":
                target_record["teacher_logit_mode"] = (
                    f"teacher_{teacher_score_space}"
                )
            target_records.append(target_record)
            for edge_example in item_edge_examples:
                edge_count = len(edge_example.candidate_ids)
                edge_record = _edge_record(edge_example)
                edge_record["teacher_logits"] = edge_scores.logits[
                    edge_offset, :edge_count
                ].cpu().tolist()
                if ensemble_alpha is not None:
                    edge_record["teacher_logit_mode"] = "ensemble"
                    edge_record["teacher_ensemble_alpha"] = ensemble_alpha
                elif teacher_score_space != "raw_logit":
                    edge_record["teacher_logit_mode"] = (
                        f"teacher_{teacher_score_space}"
                    )
                edge_records.append(edge_record)
                edge_offset += 1
    return target_records, edge_records


@torch.no_grad()
def score_pending_hard_examples(
    target_examples: Sequence[TargetExample],
    edge_examples: Sequence[EdgeExample],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    aggregator: PathAggregator,
    *,
    device: torch.device,
    batch_size: int,
    ensemble_alpha: float | None = None,
    teacher_score_space: str = "raw_logit",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score persisted hard candidates without repeating ANN retrieval."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ensemble_alpha is not None and teacher_score_space != "raw_logit":
        raise ValueError("Teacher/raw ensembles require raw Teacher logits")
    teacher.eval()

    target_records = []
    target_starts = range(0, len(target_examples), batch_size)
    for start in progress(
        target_starts,
        total=len(target_starts),
        desc="Teacher hard-target scoring",
        unit="batch",
        leave=False,
    ):
        batch = target_examples[start : start + batch_size]
        scores = (
            _score_target_ensemble_logits(
                batch, teacher, store, device, aggregator, ensemble_alpha
            )
            if ensemble_alpha is not None
            else score_target_batch(
                teacher,
                batch,
                store,
                device,
                aggregator,
                student_score_space=teacher_score_space,
            )
        )
        for index, example in enumerate(batch):
            candidate_count = len(example.candidates)
            record = _target_record(example)
            record.update(
                teacher_direct_logits=scores.direct.logits[
                    index, :candidate_count
                ].cpu().tolist(),
                teacher_evidence_logits=scores.evidence.logits[
                    index, :candidate_count
                ].cpu().tolist(),
                teacher_score_config=aggregator.config(),
            )
            if ensemble_alpha is not None:
                record.update(
                    teacher_logit_mode="ensemble",
                    teacher_ensemble_alpha=ensemble_alpha,
                )
            elif teacher_score_space != "raw_logit":
                record["teacher_logit_mode"] = f"teacher_{teacher_score_space}"
            target_records.append(record)

    # One mined target produces roughly four directly supervised edge lists.
    # Preserve the effective batch size of score_hard_candidate_sets.
    edge_batch_size = batch_size * 4
    edge_records = []
    edge_starts = range(0, len(edge_examples), edge_batch_size)
    for start in progress(
        edge_starts,
        total=len(edge_starts),
        desc="Teacher hard-edge scoring",
        unit="batch",
        leave=False,
    ):
        batch = edge_examples[start : start + edge_batch_size]
        teacher_scores = score_edge_batch(
            teacher,
            batch,
            store,
            device,
            student_score_space=teacher_score_space,
        )
        scores = (
            _ensemble_list_scores(
                _raw_edge_scores(batch, store, device),
                teacher_scores,
                ensemble_alpha,
            )
            if ensemble_alpha is not None
            else teacher_scores
        )
        for index, example in enumerate(batch):
            candidate_count = len(example.candidate_ids)
            record = _edge_record(example)
            record["teacher_logits"] = scores.logits[
                index, :candidate_count
            ].cpu().tolist()
            if ensemble_alpha is not None:
                record["teacher_logit_mode"] = "ensemble"
                record["teacher_ensemble_alpha"] = ensemble_alpha
            elif teacher_score_space != "raw_logit":
                record["teacher_logit_mode"] = f"teacher_{teacher_score_space}"
            edge_records.append(record)

    return target_records, edge_records

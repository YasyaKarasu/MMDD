"""Batch scoring for variable edge and target/path candidate lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import EdgeExample, TargetExample
from .features import FeatureStore, ObjectFeatures, normalize_object_type
from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import PathAggregator

JoinabilityModel = TeacherJoinabilityModel | StudentJoinabilityModel


@dataclass(frozen=True)
class ListScores:
    logits: torch.Tensor
    candidate_mask: torch.Tensor
    positive_indices: torch.Tensor


def _device_features(
    object_id: str,
    store: FeatureStore,
    cache: dict[str, ObjectFeatures],
    device: torch.device,
    include_hidden: bool,
) -> ObjectFeatures:
    if object_id not in cache:
        cache[object_id] = store.get(object_id).to(device, include_hidden=include_hidden)
    return cache[object_id]


def _mask(lengths: Sequence[int], width: int, device: torch.device) -> torch.Tensor:
    length_tensor = torch.tensor(lengths, device=device)
    return torch.arange(width, device=device).unsqueeze(0) < length_tensor.unsqueeze(1)


def score_edge_batch(
    model: JoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
) -> ListScores:
    include_hidden = isinstance(model, TeacherJoinabilityModel)
    feature_cache: dict[str, ObjectFeatures] = {}
    sources = []
    destinations = []
    lengths = []
    for example in examples:
        query = _device_features(example.query_id, store, feature_cache, device, include_hidden)
        source_type = (
            normalize_object_type(example.source_type) if example.source_type is not None else None
        )
        destination_type = (
            normalize_object_type(example.destination_type)
            if example.destination_type is not None
            else None
        )
        if source_type is not None and query.object_type != source_type:
            raise ValueError(
                f"{example.query_id}: declared source_type {example.source_type!r} "
                f"does not match cached type {query.object_type!r}"
            )
        lengths.append(len(example.candidate_ids))
        for candidate_id in example.candidate_ids:
            destination = _device_features(candidate_id, store, feature_cache, device, include_hidden)
            if (
                destination_type is not None
                and destination.object_type != destination_type
            ):
                raise ValueError(
                    f"{candidate_id}: declared destination_type {example.destination_type!r} "
                    f"does not match cached type {destination.object_type!r}"
                )
            sources.append(query)
            destinations.append(destination)
    flat_scores = model.score_pairs(sources, destinations)
    rows = pad_sequence(list(flat_scores.split(lengths)), batch_first=True, padding_value=0.0)
    candidate_mask = _mask(lengths, rows.shape[1], device)
    positive_indices = torch.tensor([example.positive_index for example in examples], device=device)
    return ListScores(rows, candidate_mask, positive_indices)


def score_target_batch(
    model: JoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    device: torch.device,
    aggregator: PathAggregator,
) -> ListScores:
    include_hidden = isinstance(model, TeacherJoinabilityModel)
    feature_cache: dict[str, ObjectFeatures] = {}
    direct_sources = []
    direct_destinations = []
    evidence_sources = []
    evidence_destinations = []
    target_sources = []
    target_destinations = []
    evidence_lengths = []

    for example in examples:
        query = _device_features(example.query_id, store, feature_cache, device, include_hidden)
        for candidate in example.candidates:
            target = _device_features(candidate.target_id, store, feature_cache, device, include_hidden)
            direct_sources.append(query)
            direct_destinations.append(target)
            evidence_lengths.append(len(candidate.evidence_ids))
            for evidence_id in candidate.evidence_ids:
                evidence = _device_features(evidence_id, store, feature_cache, device, include_hidden)
                evidence_sources.append(query)
                evidence_destinations.append(evidence)
                target_sources.append(evidence)
                target_destinations.append(target)

    direct_scores = model.score_pairs(direct_sources, direct_destinations)
    if evidence_sources:
        query_evidence_scores = model.score_pairs(evidence_sources, evidence_destinations)
        evidence_target_scores = model.score_pairs(target_sources, target_destinations)
    else:
        query_evidence_scores = direct_scores.new_empty(0)
        evidence_target_scores = direct_scores.new_empty(0)

    target_scores = []
    evidence_offset = 0
    for direct_score, evidence_count in zip(direct_scores, evidence_lengths):
        if evidence_count:
            query_evidence = query_evidence_scores[evidence_offset : evidence_offset + evidence_count]
            evidence_target = evidence_target_scores[evidence_offset : evidence_offset + evidence_count]
            evidence_offset += evidence_count
            combined = aggregator(
                direct_score.reshape(1, 1),
                query_evidence.reshape(1, 1, evidence_count),
                evidence_target.reshape(1, 1, evidence_count),
                torch.ones((1, 1, evidence_count), dtype=torch.bool, device=device),
            ).squeeze()
        else:
            combined = direct_score
        target_scores.append(combined)

    candidate_lengths = [len(example.candidates) for example in examples]
    rows = pad_sequence(list(torch.stack(target_scores).split(candidate_lengths)), batch_first=True, padding_value=0.0)
    candidate_mask = _mask(candidate_lengths, rows.shape[1], device)
    positive_indices = torch.tensor([example.positive_index for example in examples], device=device)
    return ListScores(rows, candidate_mask, positive_indices)

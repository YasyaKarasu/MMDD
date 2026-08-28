"""Batch scoring for variable edge and target/path candidate lists."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class TargetScores:
    direct: ListScores
    evidence: ListScores


def _device_features(
    object_id: str,
    store: FeatureStore,
    cache: dict[str, ObjectFeatures],
    device: torch.device,
    include_hidden: bool,
) -> ObjectFeatures:
    if object_id not in cache:
        if include_hidden:
            cache[object_id] = store.get(
                object_id, include_hidden=True
            ).for_scoring(device, include_hidden=True)
        else:
            cache[object_id] = store.embedding_features(object_id)
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
) -> TargetScores:
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

    if isinstance(model, TeacherJoinabilityModel):
        compression_cache: dict[str, torch.Tensor] = {}
        direct_scores = model.score_pairs(
            direct_sources,
            direct_destinations,
            compression_cache=compression_cache,
        )
        if evidence_sources:
            query_evidence_scores = model.score_pairs(
                evidence_sources,
                evidence_destinations,
                compression_cache=compression_cache,
            )
            evidence_target_edge_scores = model.score_pairs(
                target_sources,
                target_destinations,
                compression_cache=compression_cache,
            )
        else:
            query_evidence_scores = direct_scores.new_empty(0)
            evidence_target_edge_scores = direct_scores.new_empty(0)
    else:
        evidence_count = len(evidence_sources)
        scores = model.score_pairs(
            [*direct_sources, *evidence_sources, *target_sources],
            [*direct_destinations, *evidence_destinations, *target_destinations],
        )
        direct_scores, query_evidence_scores, evidence_target_edge_scores = (
            scores.split((len(direct_sources), evidence_count, evidence_count))
        )

    query_evidence_rows = pad_sequence(
        list(query_evidence_scores.split(evidence_lengths)),
        batch_first=True,
        padding_value=0.0,
    )
    evidence_target_rows = pad_sequence(
        list(evidence_target_edge_scores.split(evidence_lengths)),
        batch_first=True,
        padding_value=0.0,
    )
    evidence_path_mask = _mask(
        evidence_lengths, query_evidence_rows.shape[1], device
    )
    evidence_scores = aggregator(
        query_evidence_rows.unsqueeze(0),
        evidence_target_rows.unsqueeze(0),
        evidence_path_mask.unsqueeze(0),
    ).squeeze(0)

    candidate_lengths = [len(example.candidates) for example in examples]
    direct_rows = pad_sequence(
        list(direct_scores.split(candidate_lengths)), batch_first=True, padding_value=0.0
    )
    evidence_rows = pad_sequence(
        list(evidence_scores.split(candidate_lengths)),
        batch_first=True,
        padding_value=0.0,
    )
    candidate_mask = _mask(candidate_lengths, direct_rows.shape[1], device)
    evidence_mask = pad_sequence(
        [
            torch.tensor(
                [bool(candidate.evidence_ids) for candidate in example.candidates],
                dtype=torch.bool,
                device=device,
            )
            for example in examples
        ],
        batch_first=True,
        padding_value=False,
    )
    direct_positive_indices = torch.tensor(
        [example.direct_positive_index for example in examples], device=device
    )
    evidence_positive_indices = torch.tensor(
        [example.evidence_positive_index for example in examples], device=device
    )
    return TargetScores(
        direct=ListScores(direct_rows, candidate_mask, direct_positive_indices),
        evidence=ListScores(evidence_rows, evidence_mask, evidence_positive_indices),
    )

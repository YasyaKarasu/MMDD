"""Batch scoring for variable edge and target/path candidate lists."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import EdgeExample, TargetExample
from .features import FeatureStore, ObjectFeatures, normalize_object_type
from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import PathAggregator
from .row_support import predict_row_support

JoinabilityModel = TeacherJoinabilityModel | StudentJoinabilityModel
EdgePositiveKey = tuple[str, str, str]


@dataclass(frozen=True)
class ListScores:
    logits: torch.Tensor
    candidate_mask: torch.Tensor
    positive_indices: torch.Tensor
    positive_mask: torch.Tensor | None = None
    candidate_ids: tuple[tuple[str, ...], ...] | None = None

    def select(self, row_mask: torch.Tensor) -> ListScores:
        """Return the selected batch rows for every score field."""

        return ListScores(
            logits=self.logits[row_mask],
            candidate_mask=self.candidate_mask[row_mask],
            positive_indices=self.positive_indices[row_mask],
            positive_mask=(
                None
                if self.positive_mask is None
                else self.positive_mask[row_mask]
            ),
            candidate_ids=(
                None if self.candidate_ids is None else tuple(
                    ids for ids, selected in zip(
                        self.candidate_ids, row_mask.detach().cpu().tolist()
                    ) if selected
                )
            ),
        )


@dataclass(frozen=True)
class TargetScores:
    direct: ListScores
    evidence: ListScores
    path_logits: tuple[tuple[torch.Tensor, ...], ...] | None = None


def _device_features(
    object_id: str,
    store: FeatureStore,
    cache: dict[str, ObjectFeatures],
    device: torch.device,
    include_hidden: bool,
    hidden_dtype: torch.dtype | None = None,
) -> ObjectFeatures:
    if object_id not in cache:
        if include_hidden:
            cache[object_id] = store.get(
                object_id, include_hidden=True
            ).for_scoring(
                device, include_hidden=True, hidden_dtype=hidden_dtype
            )
        else:
            cache[object_id] = store.embedding_features(object_id)
    return cache[object_id]


def _score_teacher_pairs(
    model: TeacherJoinabilityModel,
    sources: Sequence[ObjectFeatures],
    destinations: Sequence[ObjectFeatures],
    compression_cache: dict[str, torch.Tensor] | None = None,
    score_space: str = "raw_logit",
) -> torch.Tensor:
    """Score unique directed object pairs and gather repeated occurrences."""

    unique_sources = []
    unique_destinations = []
    pair_indices: dict[tuple[str, str], int] = {}
    inverse = []
    for source, destination in zip(sources, destinations):
        pair = (source.object_id, destination.object_id)
        if pair not in pair_indices:
            pair_indices[pair] = len(unique_sources)
            unique_sources.append(source)
            unique_destinations.append(destination)
        inverse.append(pair_indices[pair])
    scores = model.score_pairs(
        unique_sources,
        unique_destinations,
        compression_cache=compression_cache,
    )
    scores = model.transform_pair_scores(
        scores,
        [features.object_type for features in unique_sources],
        [features.object_type for features in unique_destinations],
        score_space,
    )
    if len(unique_sources) == len(sources):
        return scores
    return scores.index_select(
        0, torch.tensor(inverse, dtype=torch.long, device=scores.device)
    )


def _mask(lengths: Sequence[int], width: int, device: torch.device) -> torch.Tensor:
    length_tensor = torch.tensor(lengths, device=device)
    return torch.arange(width, device=device).unsqueeze(0) < length_tensor.unsqueeze(1)


def _candidate_positive_mask(
    candidate_rows: Sequence[Sequence[str]],
    positive_id_sets: Sequence[set[str]],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if len(candidate_rows) != len(positive_id_sets):
        raise ValueError("Candidate rows and positive ID sets must have equal lengths")
    mask = pad_sequence(
        [
            torch.tensor(
                [candidate_id in positive_ids for candidate_id in candidate_ids],
                dtype=torch.bool,
                device=device,
            )
            for candidate_ids, positive_ids in zip(candidate_rows, positive_id_sets)
        ],
        batch_first=True,
        padding_value=False,
    )
    if mask.shape != (len(candidate_rows), width):
        raise ValueError("Positive mask width must match the scored candidate lists")
    return mask


def _target_positive_id_set(
    example: TargetExample, fallback_index: int
) -> set[str]:
    return set(example.positive_target_ids) or {
        example.candidates[fallback_index].target_id
    }


def _edge_positive_id_set(example: EdgeExample) -> set[str]:
    return set(example.positive_ids) or {
        example.candidate_ids[example.positive_index]
    }


def edge_positive_key(example: EdgeExample) -> EdgePositiveKey:
    """Identify one directed edge neighborhood without crossing relation types."""

    return (
        example.query_id,
        normalize_object_type(example.source_type) if example.source_type else "",
        normalize_object_type(example.destination_type)
        if example.destination_type
        else "",
    )


def global_edge_positive_ids(
    examples: Sequence[EdgeExample],
) -> dict[EdgePositiveKey, frozenset[str]]:
    """Collect every materialized positive neighbor for each training source."""

    neighbors: dict[EdgePositiveKey, set[str]] = defaultdict(set)
    for example in examples:
        neighbors[edge_positive_key(example)].update(_edge_positive_id_set(example))
    return {key: frozenset(values) for key, values in neighbors.items()}


def edge_positive_mask(
    examples: Sequence[EdgeExample],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the complete positive mask for edge candidate lists."""

    return _candidate_positive_mask(
        [example.candidate_ids for example in examples],
        [_edge_positive_id_set(example) for example in examples],
        width,
        device,
    )


def edge_confirmed_label_tensors(
    examples: Sequence[EdgeExample],
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aligned binary labels and a mask that excludes unknown candidates."""

    label_rows = []
    confirmed_rows = []
    for example in examples:
        labels = example.confirmed_labels
        if labels is not None and len(labels) != len(example.candidate_ids):
            raise ValueError("confirmed_labels must align with edge candidates")
        labels = labels or (None,) * len(example.candidate_ids)
        label_rows.append(
            torch.tensor(
                [0.0 if value is None else float(value) for value in labels],
                dtype=torch.float32,
                device=device,
            )
        )
        confirmed_rows.append(
            torch.tensor(
                [value is not None for value in labels],
                dtype=torch.bool,
                device=device,
            )
        )
    label_tensor = pad_sequence(
        label_rows, batch_first=True, padding_value=0.0
    )
    confirmed_mask = pad_sequence(
        confirmed_rows, batch_first=True, padding_value=False
    )
    if label_tensor.shape[0] != len(examples) or label_tensor.shape[1] > width:
        raise ValueError(
            "Confirmed-label width cannot exceed the scored candidate lists"
        )
    if label_tensor.shape[1] < width:
        padding = width - label_tensor.shape[1]
        label_tensor = torch.cat(
            [label_tensor, label_tensor.new_zeros((len(examples), padding))],
            dim=1,
        )
        confirmed_mask = torch.cat(
            [
                confirmed_mask,
                confirmed_mask.new_zeros((len(examples), padding)),
            ],
            dim=1,
        )
    return label_tensor, confirmed_mask


def target_positive_mask(
    examples: Sequence[TargetExample],
    width: int,
    device: torch.device,
    *,
    channel: str,
) -> torch.Tensor:
    """Build the complete positive mask for a target-list channel."""

    if channel not in {"direct", "evidence"}:
        raise ValueError("channel must be 'direct' or 'evidence'")
    candidate_rows = [
        [candidate.target_id for candidate in example.candidates]
        for example in examples
    ]
    fallback_indices = [
        example.direct_positive_index
        if channel == "direct"
        else example.evidence_positive_index
        for example in examples
    ]
    return _candidate_positive_mask(
        candidate_rows,
        [
            _target_positive_id_set(example, fallback_index)
            for example, fallback_index in zip(examples, fallback_indices)
        ],
        width,
        device,
    )


def score_edge_batch(
    model: JoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    *,
    student_score_space: str = "raw_logit",
) -> ListScores:
    include_hidden = isinstance(model, TeacherJoinabilityModel)
    hidden_dtype = (
        model.compute_dtype or next(model.parameters()).dtype
        if include_hidden
        else None
    )
    feature_cache: dict[str, ObjectFeatures] = {}
    sources = []
    destinations = []
    lengths = []
    for example in examples:
        query = _device_features(
            example.query_id,
            store,
            feature_cache,
            device,
            include_hidden,
            hidden_dtype,
        )
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
            destination = _device_features(
                candidate_id,
                store,
                feature_cache,
                device,
                include_hidden,
                hidden_dtype,
            )
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
    flat_scores = (
        _score_teacher_pairs(
            model,
            sources,
            destinations,
            score_space=student_score_space,
        )
        if isinstance(model, TeacherJoinabilityModel)
        else model.score_pairs_in_space(
            sources, destinations, student_score_space
        )
    )
    rows = pad_sequence(list(flat_scores.split(lengths)), batch_first=True, padding_value=0.0)
    candidate_mask = _mask(lengths, rows.shape[1], device)
    positive_indices = torch.tensor([example.positive_index for example in examples], device=device)
    return ListScores(
        rows,
        candidate_mask,
        positive_indices,
        edge_positive_mask(examples, rows.shape[1], device),
    )


def _expanded_candidate_ids(
    original_ids: Sequence[str],
    pool_ids: Sequence[str],
    positive_ids: set[str],
    max_negatives: int,
    rng: random.Random,
) -> list[str]:
    original = list(original_ids)
    excluded = set(original) | positive_ids
    extras = [object_id for object_id in pool_ids if object_id not in excluded]
    if len(extras) > max_negatives:
        selected = set(rng.sample(range(len(extras)), max_negatives))
        extras = [value for index, value in enumerate(extras) if index in selected]
    return [*original, *extras]


def _score_student_candidate_rows(
    student: StudentJoinabilityModel,
    query_features: Sequence[ObjectFeatures],
    candidate_rows: Sequence[Sequence[str]],
    candidate_features: dict[str, ObjectFeatures],
    positive_indices: Sequence[int],
    device: torch.device,
    positive_id_sets: Sequence[set[str]] | None = None,
    student_score_space: str = "raw_logit",
) -> ListScores:
    parameter = next(student.parameters())
    score_rows: list[torch.Tensor | None] = [None] * len(query_features)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (query, candidate_ids) in enumerate(
        zip(query_features, candidate_rows)
    ):
        destination_types = {candidate_features[value].object_type for value in candidate_ids}
        if len(destination_types) != 1:
            raise ValueError(
                "In-batch candidate expansion requires one destination type per list"
            )
        groups[(query.object_type, next(iter(destination_types)))].append(index)

    for (source_type, destination_type), row_indices in groups.items():
        pooled_ids = list(
            dict.fromkeys(
                candidate_id
                for row_index in row_indices
                for candidate_id in candidate_rows[row_index]
            )
        )
        query_embeddings = torch.stack(
            [query_features[index].embedding for index in row_indices]
        ).to(device=parameter.device, dtype=torch.float32)
        candidate_embeddings = torch.stack(
            [candidate_features[object_id].embedding for object_id in pooled_ids]
        ).to(device=parameter.device, dtype=torch.float32)
        score_matrix = student.score_embedding_matrix_in_space(
            query_embeddings,
            source_type,
            candidate_embeddings,
            destination_type,
            student_score_space,
        )
        column_by_id = {object_id: column for column, object_id in enumerate(pooled_ids)}
        for matrix_row, row_index in enumerate(row_indices):
            columns = torch.tensor(
                [column_by_id[value] for value in candidate_rows[row_index]],
                device=score_matrix.device,
            )
            score_rows[row_index] = score_matrix[matrix_row].index_select(0, columns)

    rows = [row for row in score_rows if row is not None]
    if len(rows) != len(candidate_rows):
        raise RuntimeError("Every in-batch candidate row must be scored")
    logits = pad_sequence(rows, batch_first=True, padding_value=0.0)
    lengths = [len(row) for row in candidate_rows]
    positive_mask = (
        _candidate_positive_mask(
            candidate_rows, positive_id_sets, logits.shape[1], device
        )
        if positive_id_sets is not None
        else None
    )
    return ListScores(
        logits,
        _mask(lengths, logits.shape[1], device),
        torch.tensor(positive_indices, device=device),
        positive_mask,
        tuple(tuple(row) for row in candidate_rows),
    )


def score_edge_batch_in_batch(
    student: StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    *,
    max_negatives: int = 256,
    rng: random.Random | None = None,
    student_score_space: str = "raw_logit",
    known_positive_ids: Mapping[EdgePositiveKey, Set[str]] | None = None,
    use_global_positive_mask: bool = True,
    sampling_seed: int | None = None,
    sampling_context: str = "",
    expansion_audit: dict[str, int] | None = None,
) -> ListScores:
    """Score each edge list against same-type candidates pooled from the batch."""

    if max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    rng = rng or random.Random(0)
    cache: dict[str, ObjectFeatures] = {}
    queries = []
    destination_pools: dict[str, list[str]] = defaultdict(list)
    destination_types = []
    for example in examples:
        query = _device_features(example.query_id, store, cache, device, False)
        if example.source_type is not None and query.object_type != normalize_object_type(
            example.source_type
        ):
            raise ValueError(
                f"{example.query_id}: declared source_type {example.source_type!r} "
                f"does not match cached type {query.object_type!r}"
            )
        candidates = [
            _device_features(candidate_id, store, cache, device, False)
            for candidate_id in example.candidate_ids
        ]
        types = {candidate.object_type for candidate in candidates}
        if len(types) != 1:
            raise ValueError(
                "In-batch candidate expansion requires one destination type per edge list"
            )
        destination_type = next(iter(types))
        if example.destination_type is not None and destination_type != normalize_object_type(
            example.destination_type
        ):
            raise ValueError(
                f"{example.query_id}: declared destination_type "
                f"{example.destination_type!r} does not match cached candidates"
            )
        queries.append(query)
        destination_types.append(destination_type)
        destination_pools[destination_type].extend(example.candidate_ids)

    destination_pools = {
        key: list(dict.fromkeys(values)) for key, values in destination_pools.items()
    }
    candidate_rows = []
    positive_id_sets = []
    for example, destination_type in zip(examples, destination_types):
        local_positive_ids = _edge_positive_id_set(example)
        all_positive_ids = set(
            (known_positive_ids or {}).get(
                edge_positive_key(example), local_positive_ids
            )
        ) | local_positive_ids
        pool_ids = destination_pools[destination_type]
        extra_known_positives = (
            set(pool_ids) - set(example.candidate_ids)
        ) & all_positive_ids
        positive_ids = (
            all_positive_ids if use_global_positive_mask else local_positive_ids
        )
        row_rng = rng
        if sampling_seed is not None:
            list_id = ":".join(
                (
                    example.query_id,
                    example.source_type or "",
                    example.destination_type or "",
                    example.candidate_ids[example.positive_index],
                )
            )
            row_rng = random.Random(
                f"{sampling_seed}:{sampling_context}:{list_id}"
            )
        assert row_rng is not None
        if expansion_audit is not None:
            expansion_audit["lists"] = expansion_audit.get("lists", 0) + 1
            expansion_audit["known_positive_candidates"] = (
                expansion_audit.get("known_positive_candidates", 0)
                + len(extra_known_positives)
            )
            expansion_audit["affected_lists"] = (
                expansion_audit.get("affected_lists", 0)
                + int(bool(extra_known_positives))
            )
        positive_id_sets.append(positive_ids)
        candidate_row = _expanded_candidate_ids(
            example.candidate_ids,
            pool_ids,
            positive_ids,
            max_negatives,
            row_rng,
        )
        candidate_rows.append(candidate_row)
        if expansion_audit is not None:
            expansion_audit["added_negatives"] = (
                expansion_audit.get("added_negatives", 0)
                + len(candidate_row)
                - len(example.candidate_ids)
            )
    scores = _score_student_candidate_rows(
        student,
        queries,
        candidate_rows,
        cache,
        [example.positive_index for example in examples],
        device,
        positive_id_sets,
        student_score_space=student_score_space,
    )
    if expansion_audit is not None:
        errors = 0
        for index, (example, row) in enumerate(zip(examples, candidate_rows)):
            known = set((known_positive_ids or {}).get(
                edge_positive_key(example), _edge_positive_id_set(example)
            ))
            errors += sum(
                candidate_id in known
                and bool(scores.candidate_mask[index, column])
                and not bool(scores.positive_mask[index, column])
                for column, candidate_id in enumerate(row)
            )
        expansion_audit["known_positive_as_negative"] = (
            expansion_audit.get("known_positive_as_negative", 0) + errors
        )
    return scores


def restrict_list_scores(
    scores: ListScores, lengths: Sequence[int], device: torch.device
) -> ListScores:
    """Restrict expanded rows back to their original candidate prefixes."""

    width = max(lengths)
    return ListScores(
        scores.logits[:, :width],
        _mask(lengths, width, device),
        scores.positive_indices,
        None if scores.positive_mask is None else scores.positive_mask[:, :width],
    )


def score_target_batch(
    model: JoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    device: torch.device,
    aggregator: PathAggregator,
    relation_loss_weights: dict[str, float] | None = None,
    student_score_space: str = "raw_logit",
) -> TargetScores:
    include_hidden = isinstance(model, TeacherJoinabilityModel)
    hidden_dtype = (
        model.compute_dtype or next(model.parameters()).dtype
        if include_hidden
        else None
    )
    feature_cache: dict[str, ObjectFeatures] = {}
    direct_sources = []
    direct_destinations = []
    evidence_sources = []
    evidence_destinations = []
    target_sources = []
    target_destinations = []
    evidence_lengths = []
    query_evidence_relation_keys = []
    evidence_target_relation_keys = []
    use_row_support = aggregator.evidence_aggregation == "greedy_row_support"
    flat_row_support: list[torch.Tensor] = []
    flat_content_groups: list[int] = []
    candidate_row_counts: list[int] = []
    content_key_cache: dict[str, str] = {}
    row_support_cache: dict[tuple[str, str], torch.Tensor] = {}

    for example in examples:
        query = _device_features(
            example.query_id,
            store,
            feature_cache,
            device,
            include_hidden,
            hidden_dtype,
        )
        raw_query = store.embedding_features(example.query_id) if use_row_support else None
        if use_row_support and raw_query.row_embeddings is None:
            raise ValueError(f"{example.query_id}: G5 requires cached row embeddings")
        for candidate in example.candidates:
            target = _device_features(
                candidate.target_id,
                store,
                feature_cache,
                device,
                include_hidden,
                hidden_dtype,
            )
            direct_sources.append(query)
            direct_destinations.append(target)
            evidence_lengths.append(len(candidate.evidence_ids))
            candidate_row_counts.append(
                0 if raw_query is None else int(raw_query.row_embeddings.shape[0])
            )
            content_groups: dict[str, int] = {}
            for evidence_id in sorted(candidate.evidence_ids):
                evidence = _device_features(
                    evidence_id,
                    store,
                    feature_cache,
                    device,
                    include_hidden,
                    hidden_dtype,
                )
                evidence_sources.append(query)
                evidence_destinations.append(evidence)
                target_sources.append(evidence)
                target_destinations.append(target)
                query_evidence_relation_keys.append(
                    f"{query.object_type}_to_{evidence.object_type}"
                )
                evidence_target_relation_keys.append(
                    f"{evidence.object_type}_to_{target.object_type}"
                )
                if use_row_support:
                    raw_evidence = store.embedding_features(evidence_id)
                    try:
                        support_model = aggregator.row_support_models[
                            raw_evidence.object_type
                        ]
                    except KeyError as exc:
                        raise ValueError(
                            f"G5 has no row-support model for {raw_evidence.object_type}"
                        ) from exc
                    support_key = (example.query_id, evidence_id)
                    support = row_support_cache.get(support_key)
                    if support is None:
                        assert raw_query is not None
                        assert raw_query.row_embeddings is not None
                        support = torch.tensor(
                            predict_row_support(
                                raw_query.row_embeddings,
                                raw_evidence.embedding,
                                support_model,
                            ),
                            dtype=torch.float32,
                        )
                        row_support_cache[support_key] = support
                    flat_row_support.append(support)
                    content_key = content_key_cache.get(evidence_id)
                    if content_key is None:
                        content_key = aggregator.content_key(
                            evidence_id, raw_evidence.embedding
                        )
                        content_key_cache[evidence_id] = content_key
                    flat_content_groups.append(
                        content_groups.setdefault(content_key, len(content_groups))
                    )

    if isinstance(model, TeacherJoinabilityModel):
        compression_cache: dict[str, torch.Tensor] = {}
        direct_scores = _score_teacher_pairs(
            model,
            direct_sources,
            direct_destinations,
            compression_cache,
            student_score_space,
        )
        if evidence_sources:
            query_evidence_scores = _score_teacher_pairs(
                model,
                evidence_sources,
                evidence_destinations,
                compression_cache,
                student_score_space,
            )
            evidence_target_edge_scores = _score_teacher_pairs(
                model,
                target_sources,
                target_destinations,
                compression_cache,
                student_score_space,
            )
        else:
            query_evidence_scores = direct_scores.new_empty(0)
            evidence_target_edge_scores = direct_scores.new_empty(0)
    else:
        evidence_count = len(evidence_sources)
        scores = model.score_pairs_in_space(
            [*direct_sources, *evidence_sources, *target_sources],
            [*direct_destinations, *evidence_destinations, *target_destinations],
            student_score_space,
        )
        direct_scores, query_evidence_scores, evidence_target_edge_scores = (
            scores.split((len(direct_sources), evidence_count, evidence_count))
        )

    if relation_loss_weights and not isinstance(model, TeacherJoinabilityModel):
        if any(value <= 0 for value in relation_loss_weights.values()):
            raise ValueError("relation loss weights must be positive")

        def scale_gradient(
            values: torch.Tensor, relation_keys: Sequence[str]
        ) -> torch.Tensor:
            weights = values.new_tensor(
                [relation_loss_weights.get(key, 1.0) for key in relation_keys]
            )
            return values.detach() + weights * (values - values.detach())

        query_evidence_scores = scale_gradient(
            query_evidence_scores, query_evidence_relation_keys
        )
        evidence_target_edge_scores = scale_gradient(
            evidence_target_edge_scores, evidence_target_relation_keys
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
    aggregation_kwargs = {}
    if use_row_support:
        max_paths = query_evidence_rows.shape[1]
        max_rows = max(candidate_row_counts)
        support_rows = []
        content_group_rows = []
        row_masks = []
        offset = 0
        for evidence_count, row_count in zip(
            evidence_lengths, candidate_row_counts
        ):
            support = torch.zeros(
                (max_paths, max_rows), dtype=torch.float32, device=device
            )
            groups = torch.full(
                (max_paths,), -1, dtype=torch.long, device=device
            )
            if evidence_count:
                values = torch.stack(
                    flat_row_support[offset : offset + evidence_count]
                ).to(device)
                support[:evidence_count, :row_count] = values
                groups[:evidence_count] = torch.tensor(
                    flat_content_groups[offset : offset + evidence_count],
                    dtype=torch.long,
                    device=device,
                )
            support_rows.append(support)
            content_group_rows.append(groups)
            row_masks.append(
                torch.arange(max_rows, device=device) < row_count
            )
            offset += evidence_count
        aggregation_kwargs = {
            "row_support": torch.stack(support_rows).unsqueeze(0),
            "content_groups": torch.stack(content_group_rows).unsqueeze(0),
            "row_mask": torch.stack(row_masks).unsqueeze(0),
        }
    evidence_scores = aggregator(
        query_evidence_rows.unsqueeze(0),
        evidence_target_rows.unsqueeze(0),
        evidence_path_mask.unsqueeze(0),
        **aggregation_kwargs,
    ).squeeze(0)

    raw_path_rows = query_evidence_rows + evidence_target_rows
    path_logits = []
    candidate_offset = 0
    for example in examples:
        example_rows = []
        for candidate in example.candidates:
            evidence_count = evidence_lengths[candidate_offset]
            example_rows.append(
                raw_path_rows[candidate_offset, :evidence_count]
            )
            candidate_offset += 1
        path_logits.append(tuple(example_rows))

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
    direct_positive_mask = target_positive_mask(
        examples,
        direct_rows.shape[1],
        device,
        channel="direct",
    )
    evidence_positive_mask = target_positive_mask(
        examples,
        evidence_rows.shape[1],
        device,
        channel="evidence",
    ) & evidence_mask
    return TargetScores(
        direct=ListScores(
            direct_rows,
            candidate_mask,
            direct_positive_indices,
            direct_positive_mask,
        ),
        evidence=ListScores(
            evidence_rows,
            evidence_mask,
            evidence_positive_indices,
            evidence_positive_mask,
        ),
        path_logits=tuple(path_logits),
    )


def score_target_direct_batch_in_batch(
    student: StudentJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    device: torch.device,
    *,
    max_negatives: int = 256,
    rng: random.Random | None = None,
    student_score_space: str = "raw_logit",
) -> ListScores:
    """Expand direct target lists with other table candidates from the batch."""

    if max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    rng = rng or random.Random(0)
    cache: dict[str, ObjectFeatures] = {}
    queries = [
        _device_features(example.query_id, store, cache, device, False)
        for example in examples
    ]
    all_target_ids = list(
        dict.fromkeys(
            candidate.target_id
            for example in examples
            for candidate in example.candidates
        )
    )
    for target_id in all_target_ids:
        target = _device_features(target_id, store, cache, device, False)
        if target.object_type != "table":
            raise ValueError(
                "Direct in-batch candidate expansion requires table targets"
            )

    candidate_rows = []
    positive_id_sets = []
    for example in examples:
        original_ids = [candidate.target_id for candidate in example.candidates]
        positive_ids = _target_positive_id_set(
            example, example.direct_positive_index
        )
        positive_id_sets.append(positive_ids)
        candidate_rows.append(
            _expanded_candidate_ids(
                original_ids,
                all_target_ids,
                positive_ids,
                max_negatives,
                rng,
            )
        )
    return _score_student_candidate_rows(
        student,
        queries,
        candidate_rows,
        cache,
        [example.direct_positive_index for example in examples],
        device,
        positive_id_sets,
        student_score_space,
    )

"""Persistent frozen-Teacher logits for Student training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F

from .data import (
    EdgeExample,
    TargetExample,
    TeacherScoreConfig,
)
from .features import FeatureStore
from .models import TeacherJoinabilityModel
from .objectives import PathAggregator
from .scoring import score_edge_batch, score_target_batch
from .scoring import ListScores, TargetScores
from .teacher_rerank import ensemble_scores

TrainingExample = EdgeExample | TargetExample
ENSEMBLE_TARGET_VERSION = 2


def _raw_target_edges(
    examples: Sequence[TargetExample],
    store: FeatureStore,
    device: torch.device,
) -> tuple[ListScores, list[list[torch.Tensor]]]:
    """Return frozen-cosine Q-to-T lists and per-candidate E-to-T edges."""

    direct_rows = []
    evidence_edges = []
    for example in examples:
        query = store.embedding_features(example.query_id).embedding.to(device)
        direct_values = []
        example_edges = []
        for candidate in example.candidates:
            target = store.embedding_features(candidate.target_id).embedding.to(device)
            direct_values.append(F.cosine_similarity(query, target, dim=0))
            example_edges.append(
                torch.stack(
                    [
                        F.cosine_similarity(
                            store.embedding_features(evidence_id).embedding.to(device),
                            target,
                            dim=0,
                        )
                        for evidence_id in candidate.evidence_ids
                    ]
                )
                if candidate.evidence_ids
                else query.new_empty(0)
            )
        direct_rows.append(torch.stack(direct_values))
        evidence_edges.append(example_edges)

    direct_logits = pad_sequence(direct_rows, batch_first=True, padding_value=0.0)
    candidate_mask = torch.arange(
        direct_logits.shape[1], device=device
    ).unsqueeze(0) < torch.tensor(
        [len(example.candidates) for example in examples], device=device
    ).unsqueeze(1)
    return (
        ListScores(
            direct_logits,
            candidate_mask,
            torch.tensor(
                [example.direct_positive_index for example in examples], device=device
            ),
        ),
        evidence_edges,
    )


def _teacher_target_edges(
    examples: Sequence[TargetExample],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    device: torch.device,
) -> tuple[ListScores, list[list[torch.Tensor]]]:
    direct_examples = [
        EdgeExample(
            example.query_id,
            tuple(candidate.target_id for candidate in example.candidates),
            example.direct_positive_index,
            dataset=example.dataset,
            split=example.split,
            destination_type="table",
        )
        for example in examples
    ]
    direct = score_edge_batch(teacher, direct_examples, store, device)
    edge_examples = [
        EdgeExample(
            evidence_id,
            (candidate.target_id,),
            0,
            dataset=example.dataset,
            split=example.split,
            destination_type="table",
        )
        for example in examples
        for candidate in example.candidates
        for evidence_id in candidate.evidence_ids
    ]
    flat_scores = (
        score_edge_batch(teacher, edge_examples, store, device).logits[:, 0]
        if edge_examples
        else direct.logits.new_empty(0)
    )
    offset = 0
    evidence_edges = []
    for example in examples:
        example_edges = []
        for candidate in example.candidates:
            count = len(candidate.evidence_ids)
            example_edges.append(flat_scores[offset : offset + count])
            offset += count
        evidence_edges.append(example_edges)
    return direct, evidence_edges


def _ensemble_list_scores(
    raw: ListScores, teacher: ListScores, alpha: float
) -> ListScores:
    logits = teacher.logits.new_zeros(teacher.logits.shape)
    for row in range(logits.shape[0]):
        valid = teacher.candidate_mask[row]
        indices = valid.nonzero(as_tuple=True)[0]
        combined = ensemble_scores(
            raw.logits[row, indices].detach().cpu().tolist(),
            teacher.logits[row, indices].detach().cpu().tolist(),
            alpha,
        )
        logits[row, indices] = torch.tensor(
            combined, dtype=logits.dtype, device=logits.device
        )
    return ListScores(logits, teacher.candidate_mask, teacher.positive_indices)


def _ensemble_evidence_scores(
    raw_edges: list[list[torch.Tensor]],
    teacher_edges: list[list[torch.Tensor]],
    examples: Sequence[TargetExample],
    aggregator: PathAggregator,
    alpha: float,
) -> ListScores:
    rows = []
    masks = []
    for raw_example, teacher_example in zip(raw_edges, teacher_edges):
        raw_flat = torch.cat(raw_example)
        teacher_flat = torch.cat(teacher_example)
        combined = teacher_flat.new_tensor(
            ensemble_scores(
                raw_flat.detach().cpu().tolist(),
                teacher_flat.detach().cpu().tolist(),
                alpha,
            )
        )
        offset = 0
        values = []
        valid = []
        for raw_candidate in raw_example:
            count = len(raw_candidate)
            valid.append(count > 0)
            if count == 0:
                values.append(combined.new_zeros(()))
                continue
            edges = combined[offset : offset + count].reshape(1, 1, count)
            values.append(
                aggregator(
                    torch.zeros_like(edges),
                    edges,
                    torch.ones_like(edges, dtype=torch.bool),
                ).reshape(())
            )
            offset += count
        rows.append(torch.stack(values))
        masks.append(torch.tensor(valid, dtype=torch.bool, device=combined.device))
    logits = pad_sequence(rows, batch_first=True, padding_value=0.0)
    return ListScores(
        logits,
        pad_sequence(masks, batch_first=True, padding_value=False),
        torch.tensor(
            [example.evidence_positive_index for example in examples],
            device=logits.device,
        ),
    )


def _score_target_ensemble_logits(
    examples: Sequence[TargetExample],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    device: torch.device,
    aggregator: PathAggregator,
    alpha: float,
) -> TargetScores:
    teacher_direct, teacher_edges = _teacher_target_edges(
        examples, teacher, store, device
    )
    raw_direct, raw_edges = _raw_target_edges(examples, store, device)
    return TargetScores(
        direct=_ensemble_list_scores(raw_direct, teacher_direct, alpha),
        evidence=_ensemble_evidence_scores(
            raw_edges, teacher_edges, examples, aggregator, alpha
        ),
    )


def _example_record(example: TrainingExample) -> dict[str, Any]:
    if isinstance(example, EdgeExample):
        return {
            "query_id": example.query_id,
            "candidate_ids": example.candidate_ids,
            "positive_index": example.positive_index,
            "source_type": example.source_type,
            "destination_type": example.destination_type,
            "dataset": example.dataset,
            "split": example.split,
        }
    return {
        "query_id": example.query_id,
        "candidates": [
            (candidate.target_id, candidate.evidence_ids)
            for candidate in example.candidates
        ],
        "direct_positive_index": example.direct_positive_index,
        "evidence_positive_index": example.evidence_positive_index,
        "dataset": example.dataset,
        "split": example.split,
    }


def examples_fingerprint(examples: Sequence[TrainingExample]) -> str:
    payload = json.dumps(
        [_example_record(example) for example in examples],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(
    cache_dir: Path,
    examples: Sequence[TrainingExample],
    teacher_sha256: str,
    aggregator: PathAggregator | None,
    ensemble_alpha: float | None,
) -> Path:
    kind = "edge" if isinstance(examples[0], EdgeExample) else "target"
    aggregation = (
        ""
        if aggregator is None
        else f"-{aggregator.evidence_aggregation}-k{aggregator.top_k}"
    )
    ensemble = (
        ""
        if ensemble_alpha is None
        else f"-ensemble-edge-v{ENSEMBLE_TARGET_VERSION}-a{ensemble_alpha:.12g}"
    )
    name = f"{kind}{aggregation}{ensemble}-{examples_fingerprint(examples)}.pt"
    return cache_dir / teacher_sha256 / name


def _validate_ensemble_alpha(ensemble_alpha: float | None) -> None:
    if ensemble_alpha is not None and not 0 <= ensemble_alpha <= 1:
        raise ValueError("ensemble_alpha must be in [0, 1]")


def _logit_mode_matches(
    example: TrainingExample, ensemble_alpha: float | None
) -> bool:
    if ensemble_alpha is None:
        return example.teacher_logit_mode in {None, "teacher"}
    return (
        example.teacher_logit_mode == "ensemble"
        and example.teacher_ensemble_alpha == ensemble_alpha
    )


def has_teacher_logits(
    examples: Sequence[TrainingExample],
    teacher_sha256: str | None = None,
    aggregator: PathAggregator | None = None,
    ensemble_alpha: float | None = None,
) -> bool:
    _validate_ensemble_alpha(ensemble_alpha)
    score_config = (
        TeacherScoreConfig(aggregator.evidence_aggregation, aggregator.top_k)
        if aggregator is not None
        else None
    )
    return all(
        (
            (
                example.teacher_logits is not None
                if isinstance(example, EdgeExample)
                else example.teacher_direct_logits is not None
                and example.teacher_evidence_logits is not None
                and (
                    score_config is None
                    or example.teacher_score_config == score_config
                )
            )
            and (
                teacher_sha256 is None
                or example.teacher_checkpoint_sha256 == teacher_sha256
            )
            and _logit_mode_matches(example, ensemble_alpha)
        )
        for example in examples
    )


def load_teacher_logits(
    examples: Sequence[TrainingExample],
    cache_dir: Path,
    teacher_sha256: str,
    aggregator: PathAggregator | None = None,
    ensemble_alpha: float | None = None,
) -> tuple[list[TrainingExample], Path, bool]:
    _validate_ensemble_alpha(ensemble_alpha)
    path = _cache_path(
        cache_dir, examples, teacher_sha256, aggregator, ensemble_alpha
    )
    if not path.is_file():
        return list(examples), path, False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "format_version": 1,
        "teacher_checkpoint_sha256": teacher_sha256,
        "examples_sha256": examples_fingerprint(examples),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"{path}: Teacher logit cache metadata does not match")
    cached_mode = payload.get("teacher_logit_mode", "teacher")
    cached_alpha = payload.get("teacher_ensemble_alpha")
    expected_mode = "ensemble" if ensemble_alpha is not None else "teacher"
    if cached_mode != expected_mode or cached_alpha != ensemble_alpha:
        raise ValueError(f"{path}: Teacher logit cache scoring mode does not match")
    if (
        ensemble_alpha is not None
        and payload.get("ensemble_target_version") != ENSEMBLE_TARGET_VERSION
    ):
        raise ValueError(f"{path}: ensemble target version does not match")

    if isinstance(examples[0], EdgeExample):
        logits = payload.get("teacher_logits")
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape[0] != len(examples)
            or any(
                logits.shape[1] < len(example.candidate_ids)
                for example in examples
            )
        ):
            raise ValueError(f"{path}: Teacher edge logits do not align with examples")
        loaded = [
            replace(
                example,
                teacher_logits=tuple(
                    float(value)
                    for value in logits[row, : len(example.candidate_ids)].tolist()
                ),
                teacher_checkpoint_sha256=teacher_sha256,
                teacher_logit_mode=cached_mode,
                teacher_ensemble_alpha=ensemble_alpha,
            )
            for row, example in enumerate(examples)
        ]
    else:
        assert aggregator is not None
        direct = payload.get("teacher_direct_logits")
        evidence = payload.get("teacher_evidence_logits")
        if (
            not isinstance(direct, torch.Tensor)
            or not isinstance(evidence, torch.Tensor)
            or direct.ndim != 2
            or evidence.ndim != 2
            or direct.shape[0] != len(examples)
            or evidence.shape[0] != len(examples)
            or payload.get("evidence_aggregation")
            != aggregator.evidence_aggregation
            or payload.get("evidence_top_k") != aggregator.top_k
            or any(
                direct.shape[1] < len(example.candidates)
                or evidence.shape[1] < len(example.candidates)
                for example in examples
            )
        ):
            raise ValueError(f"{path}: Teacher target logits do not align with examples")
        config = TeacherScoreConfig(
            aggregator.evidence_aggregation, aggregator.top_k
        )
        loaded = [
            replace(
                example,
                teacher_direct_logits=tuple(
                    float(value)
                    for value in direct[row, : len(example.candidates)].tolist()
                ),
                teacher_evidence_logits=tuple(
                    float(value)
                    for value in evidence[
                        row, : len(example.candidates)
                    ].tolist()
                ),
                teacher_score_config=config,
                teacher_checkpoint_sha256=teacher_sha256,
                teacher_logit_mode=cached_mode,
                teacher_ensemble_alpha=ensemble_alpha,
            )
            for row, example in enumerate(examples)
        ]
    return loaded, path, True


@torch.no_grad()
def score_and_cache_teacher_logits(
    examples: Sequence[TrainingExample],
    teacher: TeacherJoinabilityModel,
    store: FeatureStore,
    cache_dir: Path,
    teacher_sha256: str,
    *,
    device: torch.device,
    batch_size: int,
    aggregator: PathAggregator | None = None,
    ensemble_alpha: float | None = None,
) -> tuple[list[TrainingExample], Path]:
    _validate_ensemble_alpha(ensemble_alpha)
    if ensemble_alpha is not None and any(
        isinstance(example, EdgeExample) for example in examples
    ):
        raise ValueError("ensemble logits are only supported for target/path examples")
    result = list(examples)
    missing = [
        index
        for index, example in enumerate(result)
        if not has_teacher_logits(
            [example], teacher_sha256, aggregator, ensemble_alpha
        )
    ]
    teacher.eval()
    starts = range(0, len(missing), batch_size)
    for start in progress(
        starts,
        total=len(starts),
        desc="Teacher logits",
        unit="batch",
        leave=False,
    ):
        indices = missing[start : start + batch_size]
        batch = [result[index] for index in indices]
        if isinstance(batch[0], EdgeExample):
            scores = score_edge_batch(teacher, batch, store, device)
            for row, index in enumerate(indices):
                example = result[index]
                assert isinstance(example, EdgeExample)
                count = len(example.candidate_ids)
                logits = tuple(
                    float(value) for value in scores.logits[row, :count].cpu().tolist()
                )
                result[index] = replace(
                    example,
                    teacher_logits=logits,
                    teacher_checkpoint_sha256=teacher_sha256,
                    teacher_logit_mode="teacher",
                )
        else:
            assert aggregator is not None
            scores = (
                _score_target_ensemble_logits(
                    batch, teacher, store, device, aggregator, ensemble_alpha
                )
                if ensemble_alpha is not None
                else score_target_batch(teacher, batch, store, device, aggregator)
            )
            for row, index in enumerate(indices):
                example = result[index]
                assert isinstance(example, TargetExample)
                count = len(example.candidates)
                result[index] = replace(
                    example,
                    teacher_direct_logits=tuple(
                        float(value)
                        for value in scores.direct.logits[row, :count].cpu().tolist()
                    ),
                    teacher_evidence_logits=tuple(
                        float(value)
                        for value in scores.evidence.logits[row, :count].cpu().tolist()
                    ),
                    teacher_score_config=TeacherScoreConfig(
                        aggregator.evidence_aggregation, aggregator.top_k
                    ),
                    teacher_checkpoint_sha256=teacher_sha256,
                    teacher_logit_mode=("ensemble" if ensemble_alpha is not None else "teacher"),
                    teacher_ensemble_alpha=ensemble_alpha,
                )

    path = _cache_path(
        cache_dir, result, teacher_sha256, aggregator, ensemble_alpha
    )
    payload: dict[str, Any] = {
        "format_version": 1,
        "teacher_checkpoint_sha256": teacher_sha256,
        "examples_sha256": examples_fingerprint(result),
        "teacher_logit_mode": "ensemble" if ensemble_alpha is not None else "teacher",
        "teacher_ensemble_alpha": ensemble_alpha,
    }
    if ensemble_alpha is not None:
        payload["ensemble_target_version"] = ENSEMBLE_TARGET_VERSION
    if isinstance(result[0], EdgeExample):
        payload["teacher_logits"] = pad_sequence(
            [torch.tensor(example.teacher_logits) for example in result],
            batch_first=True,
        ).float()
    else:
        assert aggregator is not None
        payload["evidence_aggregation"] = aggregator.evidence_aggregation
        payload["evidence_top_k"] = aggregator.top_k
        payload["teacher_direct_logits"] = pad_sequence(
            [torch.tensor(example.teacher_direct_logits) for example in result],
            batch_first=True,
        ).float()
        payload["teacher_evidence_logits"] = pad_sequence(
            [torch.tensor(example.teacher_evidence_logits) for example in result],
            batch_first=True,
        ).float()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return result, path

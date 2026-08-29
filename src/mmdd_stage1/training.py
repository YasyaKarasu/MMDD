"""The four explicit Teacher/Student training stages."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch
from mmdd_progress import progress
from torch.nn.utils.rnn import pad_sequence

from .data import EdgeExample, TargetExample
from .features import OBJECT_TYPES, FeatureStore
from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import (
    PathAggregator,
    distillation_kl,
    listwise_cross_entropy,
    optional_distillation_kl,
    optional_listwise_cross_entropy,
)
from .scoring import (
    ListScores,
    TargetScores,
    restrict_list_scores,
    score_edge_batch,
    score_edge_batch_in_batch,
    score_target_batch,
    score_target_direct_batch_in_batch,
)

Example = TypeVar("Example", EdgeExample, TargetExample)
Model = TypeVar("Model", TeacherJoinabilityModel, StudentJoinabilityModel)
EpochCallback = Callable[[int, Model, dict[str, Any]], bool]
LOSS_REFRESH_STEPS = 100


def sample_balanced_epoch(
    examples: Sequence[Example],
    rng: random.Random,
    dataset_sampling_alpha: float,
) -> list[Example]:
    """Sample one epoch with dataset mass proportional to n_d ** alpha."""

    if not 0 <= dataset_sampling_alpha <= 1:
        raise ValueError("dataset_sampling_alpha must be between 0 and 1")
    if not examples:
        raise ValueError("Cannot sample an empty training set")
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.dataset].append(example)
    if len(groups) == 1 or dataset_sampling_alpha == 1:
        sampled = list(examples)
        rng.shuffle(sampled)
        return sampled

    datasets = sorted(groups)
    weights = {dataset: len(groups[dataset]) ** dataset_sampling_alpha for dataset in datasets}
    total_weight = sum(weights.values())
    exact = {dataset: len(examples) * weights[dataset] / total_weight for dataset in datasets}
    quotas = {dataset: int(exact[dataset]) for dataset in datasets}
    remaining = len(examples) - sum(quotas.values())
    by_fraction = sorted(datasets, key=lambda dataset: (exact[dataset] - quotas[dataset], dataset), reverse=True)
    for dataset in by_fraction[:remaining]:
        quotas[dataset] += 1

    sampled = []
    for dataset in datasets:
        pool = groups[dataset]
        needed = quotas[dataset]
        while needed:
            cycle = list(pool)
            rng.shuffle(cycle)
            take = min(needed, len(cycle))
            sampled.extend(cycle[:take])
            needed -= take
    rng.shuffle(sampled)
    return sampled


def _sample_balanced_count(
    examples: Sequence[Example],
    count: int,
    rng: random.Random,
    dataset_sampling_alpha: float,
) -> list[Example]:
    if count == 0:
        return []
    if not examples:
        raise ValueError("Cannot sample a non-zero count from an empty training set")
    groups: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        groups[example.dataset].append(example)
    datasets = sorted(groups)
    weights = {
        dataset: len(groups[dataset]) ** dataset_sampling_alpha
        for dataset in datasets
    }
    total_weight = sum(weights.values())
    exact = {
        dataset: count * weights[dataset] / total_weight for dataset in datasets
    }
    quotas = {dataset: int(exact[dataset]) for dataset in datasets}
    remaining = count - sum(quotas.values())
    by_fraction = sorted(
        datasets,
        key=lambda dataset: (exact[dataset] - quotas[dataset], dataset),
        reverse=True,
    )
    for dataset in by_fraction[:remaining]:
        quotas[dataset] += 1

    sampled = []
    for dataset in datasets:
        needed = quotas[dataset]
        while needed:
            cycle = list(groups[dataset])
            rng.shuffle(cycle)
            take = min(needed, len(cycle))
            sampled.extend(cycle[:take])
            needed -= take
    return sampled


def sample_mixed_epoch(
    base_examples: Sequence[Example],
    hard_examples: Sequence[Example],
    rng: random.Random,
    *,
    hard_fraction: float,
    dataset_sampling_alpha: float,
) -> tuple[list[Example], dict[str, int]]:
    """Sample an auditable base/hard mixture, keeping one base pass per epoch."""

    if not base_examples:
        raise ValueError("Base training data cannot be empty")
    if not 0 <= hard_fraction < 1:
        raise ValueError("hard_fraction must be in [0, 1)")
    if not 0 <= dataset_sampling_alpha <= 1:
        raise ValueError("dataset_sampling_alpha must be between 0 and 1")

    base_count = len(base_examples)
    hard_count = 0
    if hard_examples and hard_fraction > 0:
        hard_count = round(base_count * hard_fraction / (1.0 - hard_fraction))
        hard_count = max(1, hard_count)
    sampled = [
        *_sample_balanced_count(
            base_examples, base_count, rng, dataset_sampling_alpha
        ),
        *_sample_balanced_count(
            hard_examples, hard_count, rng, dataset_sampling_alpha
        ),
    ]
    rng.shuffle(sampled)
    return sampled, {"base": base_count, "hard": hard_count}


def _batches(examples: Sequence[Example], batch_size: int) -> list[list[Example]]:
    return [list(examples[start : start + batch_size]) for start in range(0, len(examples), batch_size)]


def oversample_student_edges(
    examples: Sequence[EdgeExample],
    factors: dict[str, int],
    rng: random.Random,
) -> list[EdgeExample]:
    """Repeat selected directed edge types before Student batching."""

    sampled = list(examples)
    for example in examples:
        if example.source_type is None or example.destination_type is None:
            continue
        key = f"{example.source_type}_{example.destination_type}"
        sampled.extend([example] * (factors.get(key, 1) - 1))
    rng.shuffle(sampled)
    return sampled


def _list_scores(
    rows: list[torch.Tensor],
    positive_indices: torch.Tensor,
    device: torch.device,
) -> ListScores:
    logits = pad_sequence(rows, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(row) for row in rows], device=device)
    candidate_mask = torch.arange(logits.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return ListScores(logits, candidate_mask, positive_indices)


def _edge_teacher_scores(
    examples: Sequence[EdgeExample],
    device: torch.device,
) -> ListScores:
    if any(example.teacher_logits is None for example in examples):
        raise ValueError("Student training requires cached Teacher logits")
    rows = [
        torch.tensor(example.teacher_logits, dtype=torch.float32, device=device)
        for example in examples
    ]
    positive_indices = torch.tensor(
        [example.positive_index for example in examples], device=device
    )
    return _list_scores(rows, positive_indices, device)


def _target_teacher_scores(
    examples: Sequence[TargetExample],
    device: torch.device,
) -> TargetScores:
    if any(
        example.teacher_direct_logits is None
        or example.teacher_evidence_logits is None
        for example in examples
    ):
        raise ValueError("Student training requires cached Teacher logits")

    direct_positive_indices = torch.tensor(
        [example.direct_positive_index for example in examples], device=device
    )
    evidence_positive_indices = torch.tensor(
        [example.evidence_positive_index for example in examples], device=device
    )
    direct = _list_scores(
        [
            torch.tensor(
                example.teacher_direct_logits, dtype=torch.float32, device=device
            )
            for example in examples
        ],
        direct_positive_indices,
        device,
    )
    evidence_logits = pad_sequence(
        [
            torch.tensor(
                example.teacher_evidence_logits, dtype=torch.float32, device=device
            )
            for example in examples
        ],
        batch_first=True,
        padding_value=0.0,
    )
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
    return TargetScores(
        direct=direct,
        evidence=ListScores(
            evidence_logits, evidence_mask, evidence_positive_indices
        ),
    )


def _path_supervised_losses(scores: TargetScores) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direct = listwise_cross_entropy(
        scores.direct.logits,
        scores.direct.positive_indices,
        scores.direct.candidate_mask,
    )
    evidence = optional_listwise_cross_entropy(
        scores.evidence.logits,
        scores.evidence.positive_indices,
        scores.evidence.candidate_mask,
    )
    return direct + evidence, direct, evidence


def _path_distillation_losses(
    student: TargetScores,
    teacher: TargetScores,
    temperature: float,
    row_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if row_mask is not None:
        if row_mask.shape != (student.direct.logits.shape[0],):
            raise ValueError("Distillation row mask must match the batch")
        if not row_mask.any().item():
            zero = student.direct.logits.sum() * 0.0
            return zero, zero, zero
        student = TargetScores(
            direct=ListScores(
                student.direct.logits[row_mask],
                student.direct.candidate_mask[row_mask],
                student.direct.positive_indices[row_mask],
            ),
            evidence=ListScores(
                student.evidence.logits[row_mask],
                student.evidence.candidate_mask[row_mask],
                student.evidence.positive_indices[row_mask],
            ),
        )
        teacher = TargetScores(
            direct=ListScores(
                teacher.direct.logits[row_mask],
                teacher.direct.candidate_mask[row_mask],
                teacher.direct.positive_indices[row_mask],
            ),
            evidence=ListScores(
                teacher.evidence.logits[row_mask],
                teacher.evidence.candidate_mask[row_mask],
                teacher.evidence.positive_indices[row_mask],
            ),
        )
    direct = distillation_kl(
        student.direct.logits,
        teacher.direct.logits,
        student.direct.candidate_mask,
        temperature,
    )
    evidence = optional_distillation_kl(
        student.evidence.logits,
        teacher.evidence.logits,
        student.evidence.positive_indices,
        student.evidence.candidate_mask,
        temperature,
    )
    return direct + evidence, direct, evidence


def _epoch_record(
    epoch: int,
    loss_values: dict[str, float],
    sampled: Sequence[Example],
    source_samples: dict[str, int],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        **loss_values,
        "dataset_samples": dict(sorted(Counter(example.dataset for example in sampled).items())),
        "source_samples": source_samples,
    }


def _optimize(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def student_anchor_loss(student: StudentJoinabilityModel) -> torch.Tensor:
    """Return the scale-normalized distance from raw-retrieval geometry."""

    parameter = next(student.parameters())
    identity = torch.eye(
        student.student_dim, device=parameter.device, dtype=parameter.dtype
    )
    relation_anchor = sum(
        (relation - identity).square().sum() / (student.student_dim**2)
        for relation in student.relations.values()
    )
    if student.freeze_projections:
        return relation_anchor
    projection_anchor = sum(
        (student.projections[object_type].weight - initial).square().sum()
        / (student.input_dim * student.student_dim)
        for object_type, initial in zip(
            OBJECT_TYPES, student.initial_projection_weights
        )
    )
    return relation_anchor + projection_anchor


def student_evidence_anchor_loss(
    student: StudentJoinabilityModel,
) -> torch.Tensor:
    """Return the anchor term for the four table/evidence relations."""

    parameter = next(student.parameters())
    identity = torch.eye(
        student.student_dim, device=parameter.device, dtype=parameter.dtype
    )
    evidence_keys = {
        student.relation_key("table", "text"),
        student.relation_key("text", "table"),
        student.relation_key("table", "image"),
        student.relation_key("image", "table"),
    }
    return sum(
        (student.relations[key] - identity).square().sum()
        / (student.student_dim**2)
        for key in evidence_keys
    )


def student_relation_drift(
    student: StudentJoinabilityModel,
) -> dict[str, float]:
    """Measure the Frobenius distance from identity for every relation."""

    parameter = next(student.parameters())
    identity = torch.eye(
        student.student_dim, device=parameter.device, dtype=parameter.dtype
    )
    with torch.no_grad():
        return {
            key: float(torch.linalg.vector_norm(relation - identity).cpu())
            for key, relation in sorted(student.relations.items())
        }


def _anchor_losses(
    student: StudentJoinabilityModel,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    evidence_weight = (
        anchor_weight
        if anchor_weight_evidence is None
        else anchor_weight_evidence
    )
    if anchor_weight == 0 and evidence_weight == 0:
        zero = next(student.parameters()).new_zeros(())
        return zero, zero
    anchor = student_anchor_loss(student)
    if evidence_weight == anchor_weight:
        return anchor, anchor_weight * anchor
    evidence_anchor = student_evidence_anchor_loss(student)
    weighted = (
        anchor_weight * anchor
        + (evidence_weight - anchor_weight) * evidence_anchor
    )
    return anchor, weighted


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _loss_refresh_due(step: int, step_count: int) -> bool:
    return step % LOSS_REFRESH_STEPS == 0 or step == step_count


def _flush_loss_values(
    pending: list[torch.Tensor], values: list[float]
) -> None:
    values.extend(torch.stack(pending).cpu().tolist())
    pending.clear()


def _finish_epoch(
    history: list[dict[str, Any]],
    record: dict[str, Any],
    model: Model,
    callback: EpochCallback[Model] | None,
) -> bool:
    history.append(record)
    return bool(callback and callback(int(record["epoch"]), model, record))


@torch.no_grad()
def _teacher_edge_objective(
    model: TeacherJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        scores = score_edge_batch(model, batch, store, device)
        losses.append(
            float(
                listwise_cross_entropy(
                    scores.logits, scores.positive_indices, scores.candidate_mask
                )
            )
        )
    return _mean(losses)


@torch.no_grad()
def _teacher_path_objective(
    model: TeacherJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    aggregator: PathAggregator,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses = []
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        scores = score_target_batch(model, batch, store, device, aggregator)
        loss, _direct, _evidence = _path_supervised_losses(scores)
        losses.append(float(loss))
    return _mean(losses)


@torch.no_grad()
def _student_edge_objective(
    student: StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    batch_size: int,
    temperature: float,
    distillation_weight: float,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
    in_batch_negatives: bool,
    in_batch_max_negatives: int,
) -> float:
    student.eval()
    losses = []
    rng = random.Random(0)
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        teacher_scores = (
            _edge_teacher_scores(batch, device)
            if distillation_weight > 0
            else None
        )
        if in_batch_negatives:
            expanded_scores = score_edge_batch_in_batch(
                student,
                batch,
                store,
                device,
                max_negatives=in_batch_max_negatives,
                rng=rng,
            )
            student_scores = restrict_list_scores(
                expanded_scores,
                [len(example.candidate_ids) for example in batch],
                device,
            )
            supervised = listwise_cross_entropy(
                expanded_scores.logits,
                expanded_scores.positive_indices,
                expanded_scores.candidate_mask,
            )
        else:
            student_scores = score_edge_batch(student, batch, store, device)
            supervised = student_scores.logits.new_zeros(())
        distillation = (
            distillation_kl(
                student_scores.logits,
                teacher_scores.logits,
                student_scores.candidate_mask,
                temperature,
            )
            if teacher_scores is not None
            else student_scores.logits.new_zeros(())
        )
        _anchor, weighted_anchor = _anchor_losses(
            student, anchor_weight, anchor_weight_evidence
        )
        losses.append(
            float(supervised + distillation_weight * distillation + weighted_anchor)
        )
    return _mean(losses)


def _validate_cached_path_aggregation(
    examples: Sequence[TargetExample], aggregator: PathAggregator
) -> None:
    for example in examples:
        config = example.teacher_score_config
        if config is not None and (
            config.evidence_aggregation != aggregator.evidence_aggregation
            or config.evidence_top_k != aggregator.top_k
        ):
            raise ValueError(
                "Cached Teacher logits use a different evidence aggregation configuration"
            )


@torch.no_grad()
def _student_path_objective(
    student: StudentJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    aggregator: PathAggregator,
    device: torch.device,
    batch_size: int,
    temperature: float,
    distillation_weight: float,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
    distillation_datasets: set[str] | None,
    in_batch_negatives: bool,
    in_batch_max_negatives: int,
) -> float:
    student.eval()
    losses = []
    rng = random.Random(0)
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        teacher_scores = (
            _target_teacher_scores(batch, device)
            if distillation_weight > 0
            else None
        )
        student_scores = score_target_batch(
            student, batch, store, device, aggregator
        )
        supervised, direct, evidence = _path_supervised_losses(student_scores)
        if in_batch_negatives:
            expanded_direct = score_target_direct_batch_in_batch(
                student,
                batch,
                store,
                device,
                max_negatives=in_batch_max_negatives,
                rng=rng,
            )
            direct = listwise_cross_entropy(
                expanded_direct.logits,
                expanded_direct.positive_indices,
                expanded_direct.candidate_mask,
            )
            supervised = direct + evidence
        distillation_rows = (
            torch.tensor(
                [example.dataset in distillation_datasets for example in batch],
                dtype=torch.bool,
                device=device,
            )
            if distillation_datasets is not None
            else None
        )
        distillation = (
            _path_distillation_losses(
                student_scores,
                teacher_scores,
                temperature,
                distillation_rows,
            )[0]
            if teacher_scores is not None
            else supervised.new_zeros(())
        )
        _anchor, weighted_anchor = _anchor_losses(
            student, anchor_weight, anchor_weight_evidence
        )
        losses.append(
            float(supervised + distillation_weight * distillation + weighted_anchor)
        )
    return _mean(losses)


def train_teacher_edges(
    model: TeacherJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[EdgeExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[EdgeExample] = (),
    epoch_callback: EpochCallback[TeacherJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    epoch_bar = progress(range(epochs), desc="Teacher edge", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        losses = []
        pending_losses = []
        sampled, source_samples = sample_mixed_epoch(
            examples,
            hard_examples,
            rng,
            hard_fraction=hard_fraction,
            dataset_sampling_alpha=dataset_sampling_alpha,
        )
        batches = _batches(sampled, batch_size)
        batch_bar = progress(
            batches,
            desc=f"Epoch {epoch + 1}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(batch_bar, 1):
            scores = score_edge_batch(model, batch, store, device)
            loss = listwise_cross_entropy(scores.logits, scores.positive_indices, scores.candidate_mask)
            _optimize(loss, optimizer)
            pending_losses.append(loss.detach())
            if _loss_refresh_due(step, len(batches)):
                _flush_loss_values(pending_losses, losses)
                batch_bar.set_postfix(loss=f"{_mean(losses):.4f}")
        train_loss = _mean(losses)
        values = {"loss": train_loss, "train_loss": train_loss}
        if dev_examples:
            values["dev_loss"] = _teacher_edge_objective(
                model, dev_examples, store, device, batch_size
            )
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(epoch + 1, values, sampled, source_samples)
        if _finish_epoch(history, record, model, epoch_callback):
            break
    return history


def train_teacher_paths(
    model: TeacherJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    aggregator: PathAggregator,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[TargetExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[TargetExample] = (),
    epoch_callback: EpochCallback[TeacherJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    epoch_bar = progress(range(epochs), desc="Teacher path", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        losses = []
        direct_losses = []
        evidence_losses = []
        pending_losses = []
        pending_direct_losses = []
        pending_evidence_losses = []
        sampled, source_samples = sample_mixed_epoch(
            examples,
            hard_examples,
            rng,
            hard_fraction=hard_fraction,
            dataset_sampling_alpha=dataset_sampling_alpha,
        )
        batches = _batches(sampled, batch_size)
        batch_bar = progress(
            batches,
            desc=f"Epoch {epoch + 1}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(batch_bar, 1):
            scores = score_target_batch(model, batch, store, device, aggregator)
            loss, direct_loss, evidence_loss = _path_supervised_losses(scores)
            _optimize(loss, optimizer)
            pending_losses.append(loss.detach())
            pending_direct_losses.append(direct_loss.detach())
            pending_evidence_losses.append(evidence_loss.detach())
            if _loss_refresh_due(step, len(batches)):
                _flush_loss_values(pending_losses, losses)
                _flush_loss_values(pending_direct_losses, direct_losses)
                _flush_loss_values(pending_evidence_losses, evidence_losses)
                batch_bar.set_postfix(loss=f"{_mean(losses):.4f}")
        train_loss = _mean(losses)
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "direct_loss": _mean(direct_losses),
            "evidence_loss": _mean(evidence_losses),
        }
        if dev_examples:
            values["dev_loss"] = _teacher_path_objective(
                model, dev_examples, store, aggregator, device, batch_size
            )
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(epoch + 1, values, sampled, source_samples)
        if _finish_epoch(history, record, model, epoch_callback):
            break
    return history


def train_student_edges(
    student: StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    temperature: float,
    distillation_weight: float = 1.0,
    anchor_weight: float = 0.0,
    anchor_weight_evidence: float | None = None,
    in_batch_negatives: bool = False,
    in_batch_max_negatives: int = 256,
    edge_type_oversample: dict[str, int] | None = None,
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[EdgeExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[EdgeExample] = (),
    epoch_callback: EpochCallback[StudentJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    epoch_bar = progress(range(epochs), desc="Student edge", unit="epoch")
    for epoch in epoch_bar:
        student.train()
        losses = []
        supervised_losses = []
        distillation_losses = []
        anchor_losses = []
        weighted_anchor_losses = []
        pending_losses = []
        pending_supervised_losses = []
        pending_distillation_losses = []
        pending_anchor_losses = []
        pending_weighted_anchor_losses = []
        sampled, source_samples = sample_mixed_epoch(
            examples,
            hard_examples,
            rng,
            hard_fraction=hard_fraction,
            dataset_sampling_alpha=dataset_sampling_alpha,
        )
        if edge_type_oversample:
            original_count = len(sampled)
            sampled = oversample_student_edges(
                sampled, edge_type_oversample, rng
            )
            source_samples["oversampled"] = len(sampled) - original_count
        batches = _batches(sampled, batch_size)
        batch_bar = progress(
            batches,
            desc=f"Epoch {epoch + 1}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(batch_bar, 1):
            teacher_scores = (
                _edge_teacher_scores(batch, device)
                if distillation_weight > 0
                else None
            )
            if in_batch_negatives:
                expanded_scores = score_edge_batch_in_batch(
                    student,
                    batch,
                    store,
                    device,
                    max_negatives=in_batch_max_negatives,
                    rng=rng,
                )
                student_scores = restrict_list_scores(
                    expanded_scores,
                    [len(example.candidate_ids) for example in batch],
                    device,
                )
                supervised = listwise_cross_entropy(
                    expanded_scores.logits,
                    expanded_scores.positive_indices,
                    expanded_scores.candidate_mask,
                )
            else:
                student_scores = score_edge_batch(student, batch, store, device)
                supervised = student_scores.logits.new_zeros(())
            distillation = (
                distillation_kl(
                    student_scores.logits,
                    teacher_scores.logits,
                    student_scores.candidate_mask,
                    temperature,
                )
                if teacher_scores is not None
                else student_scores.logits.new_zeros(())
            )
            anchor, weighted_anchor = _anchor_losses(
                student, anchor_weight, anchor_weight_evidence
            )
            total = (
                supervised
                + distillation_weight * distillation
                + weighted_anchor
            )
            _optimize(total, optimizer)
            pending_losses.append(total.detach())
            pending_supervised_losses.append(supervised.detach())
            pending_distillation_losses.append(distillation.detach())
            pending_anchor_losses.append(anchor.detach())
            pending_weighted_anchor_losses.append(weighted_anchor.detach())
            if _loss_refresh_due(step, len(batches)):
                _flush_loss_values(pending_losses, losses)
                _flush_loss_values(pending_supervised_losses, supervised_losses)
                _flush_loss_values(
                    pending_distillation_losses, distillation_losses
                )
                _flush_loss_values(pending_anchor_losses, anchor_losses)
                _flush_loss_values(
                    pending_weighted_anchor_losses, weighted_anchor_losses
                )
                batch_bar.set_postfix(loss=f"{_mean(losses):.4f}")
        train_loss = _mean(losses)
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "supervised_loss": _mean(supervised_losses),
            "distillation_loss": _mean(distillation_losses),
            "anchor_loss": _mean(anchor_losses),
            "weighted_anchor_loss": _mean(weighted_anchor_losses),
        }
        if dev_examples:
            values["dev_loss"] = _student_edge_objective(
                student,
                dev_examples,
                store,
                device,
                batch_size,
                temperature,
                distillation_weight,
                anchor_weight,
                anchor_weight_evidence,
                in_batch_negatives,
                in_batch_max_negatives,
            )
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(epoch + 1, values, sampled, source_samples)
        if _finish_epoch(history, record, student, epoch_callback):
            break
    return history


def train_student_paths(
    student: StudentJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    aggregator: PathAggregator,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    temperature: float,
    distillation_weight: float,
    anchor_weight: float = 0.0,
    anchor_weight_evidence: float | None = None,
    distillation_datasets: set[str] | None = None,
    in_batch_negatives: bool = False,
    in_batch_max_negatives: int = 256,
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[TargetExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[TargetExample] = (),
    epoch_callback: EpochCallback[StudentJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    _validate_cached_path_aggregation(
        [*examples, *hard_examples, *dev_examples], aggregator
    )
    epoch_bar = progress(range(epochs), desc="Student path", unit="epoch")
    for epoch in epoch_bar:
        student.train()
        totals = []
        supervised_losses = []
        distillation_losses = []
        direct_supervised_losses = []
        evidence_supervised_losses = []
        direct_distillation_losses = []
        evidence_distillation_losses = []
        anchor_losses = []
        weighted_anchor_losses = []
        pending_totals = []
        pending_supervised_losses = []
        pending_distillation_losses = []
        pending_direct_supervised_losses = []
        pending_evidence_supervised_losses = []
        pending_direct_distillation_losses = []
        pending_evidence_distillation_losses = []
        pending_anchor_losses = []
        pending_weighted_anchor_losses = []
        sampled, source_samples = sample_mixed_epoch(
            examples,
            hard_examples,
            rng,
            hard_fraction=hard_fraction,
            dataset_sampling_alpha=dataset_sampling_alpha,
        )
        batches = _batches(sampled, batch_size)
        batch_bar = progress(
            batches,
            desc=f"Epoch {epoch + 1}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(batch_bar, 1):
            teacher_scores = (
                _target_teacher_scores(batch, device)
                if distillation_weight > 0
                else None
            )
            student_scores = score_target_batch(student, batch, store, device, aggregator)
            supervised, direct_supervised, evidence_supervised = _path_supervised_losses(
                student_scores
            )
            if in_batch_negatives:
                expanded_direct = score_target_direct_batch_in_batch(
                    student,
                    batch,
                    store,
                    device,
                    max_negatives=in_batch_max_negatives,
                    rng=rng,
                )
                direct_supervised = listwise_cross_entropy(
                    expanded_direct.logits,
                    expanded_direct.positive_indices,
                    expanded_direct.candidate_mask,
                )
                supervised = direct_supervised + evidence_supervised
            if teacher_scores is not None:
                distillation_rows = (
                    torch.tensor(
                        [
                            example.dataset in distillation_datasets
                            for example in batch
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    if distillation_datasets is not None
                    else None
                )
                distillation, direct_distillation, evidence_distillation = (
                    _path_distillation_losses(
                        student_scores,
                        teacher_scores,
                        temperature,
                        distillation_rows,
                    )
                )
            else:
                distillation = supervised.new_zeros(())
                direct_distillation = supervised.new_zeros(())
                evidence_distillation = supervised.new_zeros(())
            anchor, weighted_anchor = _anchor_losses(
                student, anchor_weight, anchor_weight_evidence
            )
            total = (
                supervised
                + distillation_weight * distillation
                + weighted_anchor
            )
            _optimize(total, optimizer)
            pending_totals.append(total.detach())
            pending_supervised_losses.append(supervised.detach())
            pending_distillation_losses.append(distillation.detach())
            pending_direct_supervised_losses.append(direct_supervised.detach())
            pending_evidence_supervised_losses.append(evidence_supervised.detach())
            pending_direct_distillation_losses.append(direct_distillation.detach())
            pending_evidence_distillation_losses.append(evidence_distillation.detach())
            pending_anchor_losses.append(anchor.detach())
            pending_weighted_anchor_losses.append(weighted_anchor.detach())
            if _loss_refresh_due(step, len(batches)):
                _flush_loss_values(pending_totals, totals)
                _flush_loss_values(pending_supervised_losses, supervised_losses)
                _flush_loss_values(pending_distillation_losses, distillation_losses)
                _flush_loss_values(
                    pending_direct_supervised_losses, direct_supervised_losses
                )
                _flush_loss_values(
                    pending_evidence_supervised_losses, evidence_supervised_losses
                )
                _flush_loss_values(
                    pending_direct_distillation_losses,
                    direct_distillation_losses,
                )
                _flush_loss_values(
                    pending_evidence_distillation_losses,
                    evidence_distillation_losses,
                )
                _flush_loss_values(pending_anchor_losses, anchor_losses)
                _flush_loss_values(
                    pending_weighted_anchor_losses, weighted_anchor_losses
                )
                batch_bar.set_postfix(loss=f"{_mean(totals):.4f}")
        train_loss = _mean(totals)
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "supervised_loss": _mean(supervised_losses),
            "distillation_loss": _mean(distillation_losses),
            "direct_supervised_loss": _mean(direct_supervised_losses),
            "evidence_supervised_loss": _mean(evidence_supervised_losses),
            "direct_distillation_loss": _mean(direct_distillation_losses),
            "evidence_distillation_loss": _mean(evidence_distillation_losses),
            "anchor_loss": _mean(anchor_losses),
            "weighted_anchor_loss": _mean(weighted_anchor_losses),
        }
        if dev_examples:
            values["dev_loss"] = _student_path_objective(
                student,
                dev_examples,
                store,
                aggregator,
                device,
                batch_size,
                temperature,
                distillation_weight,
                anchor_weight,
                anchor_weight_evidence,
                distillation_datasets,
                in_batch_negatives,
                in_batch_max_negatives,
            )
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(epoch + 1, values, sampled, source_samples)
        if _finish_epoch(history, record, student, epoch_callback):
            break
    return history


def checkpoint(
    model: TeacherJoinabilityModel | StudentJoinabilityModel,
    stage: str,
    aggregator: PathAggregator | None = None,
) -> dict[str, Any]:
    model_kind = "teacher" if isinstance(model, TeacherJoinabilityModel) else "student"
    payload = {
        "format_version": 1,
        "model_kind": model_kind,
        "completed_stage": stage,
        "config": model.config(),
        "state_dict": model.state_dict(),
    }
    if aggregator is not None:
        payload["path_aggregation"] = {
            "evidence_aggregation": aggregator.evidence_aggregation,
            "evidence_top_k": aggregator.top_k,
        }
    return payload

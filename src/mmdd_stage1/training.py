"""The four explicit Teacher/Student training stages."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any, TypeVar

import torch
from torch.nn.utils.rnn import pad_sequence

from .data import EdgeExample, TargetExample
from .features import FeatureStore
from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import PathAggregator, distillation_kl, listwise_cross_entropy, student_path_loss
from .scoring import ListScores, score_edge_batch, score_target_batch

Example = TypeVar("Example", EdgeExample, TargetExample)


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


def _batches(examples: Sequence[Example], batch_size: int) -> list[list[Example]]:
    return [list(examples[start : start + batch_size]) for start in range(0, len(examples), batch_size)]


def _cached_teacher_scores(
    examples: Sequence[EdgeExample] | Sequence[TargetExample],
    device: torch.device,
) -> ListScores | None:
    if not all(example.teacher_logits is not None for example in examples):
        return None
    rows = [torch.tensor(example.teacher_logits, dtype=torch.float32, device=device) for example in examples]
    logits = pad_sequence(rows, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(row) for row in rows], device=device)
    candidate_mask = torch.arange(logits.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    positive_indices = torch.tensor([example.positive_index for example in examples], device=device)
    return ListScores(logits, candidate_mask, positive_indices)


def _epoch_record(epoch: int, loss_values: dict[str, float], sampled: Sequence[Example]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        **loss_values,
        "dataset_samples": dict(sorted(Counter(example.dataset for example in sampled).items())),
    }


def _optimize(loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    model.train()
    for epoch in range(epochs):
        losses = []
        sampled = sample_balanced_epoch(examples, rng, dataset_sampling_alpha)
        for batch in _batches(sampled, batch_size):
            scores = score_edge_batch(model, batch, store, device)
            loss = listwise_cross_entropy(scores.logits, scores.positive_indices, scores.candidate_mask)
            _optimize(loss, optimizer)
            losses.append(float(loss.detach()))
        history.append(_epoch_record(epoch + 1, {"loss": sum(losses) / len(losses)}, sampled))
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
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    model.train()
    for epoch in range(epochs):
        losses = []
        sampled = sample_balanced_epoch(examples, rng, dataset_sampling_alpha)
        for batch in _batches(sampled, batch_size):
            scores = score_target_batch(model, batch, store, device, aggregator)
            loss = listwise_cross_entropy(scores.logits, scores.positive_indices, scores.candidate_mask)
            _optimize(loss, optimizer)
            losses.append(float(loss.detach()))
        history.append(_epoch_record(epoch + 1, {"loss": sum(losses) / len(losses)}, sampled))
    return history


def train_student_edges(
    student: StudentJoinabilityModel,
    teacher: TeacherJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
    temperature: float,
    dataset_sampling_alpha: float = 0.0,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    teacher.eval()
    teacher.requires_grad_(False)
    student.train()
    for epoch in range(epochs):
        losses = []
        sampled = sample_balanced_epoch(examples, rng, dataset_sampling_alpha)
        for batch in _batches(sampled, batch_size):
            teacher_scores = _cached_teacher_scores(batch, device)
            if teacher_scores is None:
                with torch.no_grad():
                    teacher_scores = score_edge_batch(teacher, batch, store, device)
            student_scores = score_edge_batch(student, batch, store, device)
            loss = distillation_kl(
                student_scores.logits,
                teacher_scores.logits,
                student_scores.candidate_mask,
                temperature,
            )
            _optimize(loss, optimizer)
            losses.append(float(loss.detach()))
        history.append(_epoch_record(epoch + 1, {"loss": sum(losses) / len(losses)}, sampled))
    return history


def train_student_paths(
    student: StudentJoinabilityModel,
    teacher: TeacherJoinabilityModel,
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
    dataset_sampling_alpha: float = 0.0,
) -> list[dict[str, Any]]:
    history = []
    rng = random.Random(seed)
    teacher.eval()
    teacher.requires_grad_(False)
    student.train()
    for epoch in range(epochs):
        totals = []
        supervised_losses = []
        distillation_losses = []
        sampled = sample_balanced_epoch(examples, rng, dataset_sampling_alpha)
        for batch in _batches(sampled, batch_size):
            teacher_scores = _cached_teacher_scores(batch, device)
            if teacher_scores is None:
                with torch.no_grad():
                    teacher_scores = score_target_batch(teacher, batch, store, device, aggregator)
            else:
                for example in batch:
                    config = example.teacher_score_config
                    if config is not None and (
                        config.evidence_aggregation != aggregator.evidence_aggregation
                        or config.evidence_top_k != aggregator.top_k
                    ):
                        raise ValueError("Cached Teacher logits use a different evidence aggregation configuration")
            student_scores = score_target_batch(student, batch, store, device, aggregator)
            total, supervised, distillation = student_path_loss(
                student_scores.logits,
                teacher_scores.logits,
                student_scores.positive_indices,
                student_scores.candidate_mask,
                temperature,
                distillation_weight,
            )
            _optimize(total, optimizer)
            totals.append(float(total.detach()))
            supervised_losses.append(float(supervised.detach()))
            distillation_losses.append(float(distillation.detach()))
        history.append(
            _epoch_record(
                epoch + 1,
                {
                    "loss": sum(totals) / len(totals),
                    "supervised_loss": sum(supervised_losses) / len(supervised_losses),
                    "distillation_loss": sum(distillation_losses) / len(distillation_losses),
                },
                sampled,
            )
        )
    return history


def checkpoint(model: TeacherJoinabilityModel | StudentJoinabilityModel, stage: str) -> dict[str, Any]:
    model_kind = "teacher" if isinstance(model, TeacherJoinabilityModel) else "student"
    return {
        "format_version": 1,
        "model_kind": model_kind,
        "completed_stage": stage,
        "config": model.config(),
        "state_dict": model.state_dict(),
    }

"""The four explicit Teacher/Student training stages."""

from __future__ import annotations

import math
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
    relation_macro_binary_cross_entropy_with_logits,
)
from .scoring import (
    ListScores,
    TargetScores,
    edge_confirmed_label_tensors,
    edge_positive_mask,
    global_edge_positive_ids,
    restrict_list_scores,
    score_edge_batch,
    score_edge_batch_in_batch,
    score_target_batch,
    score_target_direct_batch_in_batch,
    target_positive_mask,
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
    if len({example.dataset for example in examples}) == 1 or dataset_sampling_alpha == 1:
        sampled = list(examples)
        rng.shuffle(sampled)
        return sampled
    sampled = _sample_balanced_count(
        examples,
        len(examples),
        rng,
        dataset_sampling_alpha,
    )
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
    positive_mask: torch.Tensor | None = None,
) -> ListScores:
    logits = pad_sequence(rows, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(row) for row in rows], device=device)
    candidate_mask = torch.arange(logits.shape[1], device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return ListScores(logits, candidate_mask, positive_indices, positive_mask)


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
    scores = _list_scores(rows, positive_indices, device)
    return ListScores(
        scores.logits,
        scores.candidate_mask,
        scores.positive_indices,
        edge_positive_mask(examples, scores.logits.shape[1], device),
    )


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
    positive_mask = target_positive_mask(
        examples,
        direct.logits.shape[1],
        device,
        channel="direct",
    )
    evidence_positive_mask = target_positive_mask(
        examples,
        evidence_logits.shape[1],
        device,
        channel="evidence",
    ) & evidence_mask
    direct = ListScores(
        direct.logits,
        direct.candidate_mask,
        direct.positive_indices,
        positive_mask,
    )
    return TargetScores(
        direct=direct,
        evidence=ListScores(
            evidence_logits,
            evidence_mask,
            evidence_positive_indices,
            evidence_positive_mask,
        ),
    )


def _path_supervised_losses(
    scores: TargetScores,
    *,
    positive_loss_mode: str = "sum_probability",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    direct = listwise_cross_entropy(
        scores.direct.logits,
        scores.direct.positive_indices,
        scores.direct.candidate_mask,
        scores.direct.positive_mask,
        positive_loss_mode=positive_loss_mode,
    )
    evidence = optional_listwise_cross_entropy(
        scores.evidence.logits,
        scores.evidence.positive_indices,
        scores.evidence.candidate_mask,
        scores.evidence.positive_mask,
        positive_loss_mode=positive_loss_mode,
    )
    return direct + evidence, direct, evidence


def _teacher_edge_loss(
    scores: ListScores,
    *,
    positive_loss_mode: str = "sum_probability",
) -> torch.Tensor:
    return listwise_cross_entropy(
        scores.logits,
        scores.positive_indices,
        scores.candidate_mask,
        scores.positive_mask,
        positive_loss_mode=positive_loss_mode,
    )


def _teacher_edge_losses(
    examples: Sequence[EdgeExample],
    ranking_scores: ListScores,
    confidence_scores: ListScores | None,
    edge_bce_weight: float,
    *,
    positive_loss_mode: str = "sum_probability",
) -> dict[str, torch.Tensor]:
    ranking = _teacher_edge_loss(
        ranking_scores, positive_loss_mode=positive_loss_mode
    )
    absolute = ranking.new_zeros(())
    if edge_bce_weight > 0:
        if confidence_scores is None:
            raise ValueError("Teacher edge BCE requires confidence-logit scores")
        labels, confirmed_mask = edge_confirmed_label_tensors(
            examples,
            confidence_scores.logits.shape[1],
            confidence_scores.logits.device,
        )
        absolute = relation_macro_binary_cross_entropy_with_logits(
            confidence_scores.logits,
            labels,
            confirmed_mask,
            [_edge_relation_key(example) for example in examples],
        )
    return {
        "loss": ranking + edge_bce_weight * absolute,
        "ranking_loss": ranking,
        "absolute_loss": absolute,
    }


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
            direct=student.direct.select(row_mask),
            evidence=student.evidence.select(row_mask),
        )
        teacher = TargetScores(
            direct=teacher.direct.select(row_mask),
            evidence=teacher.evidence.select(row_mask),
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
        student.evidence.positive_mask,
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

    relation_anchor = sum(
        student.relation_residual_squared_norm(key) / (student.student_dim**2)
        for key in _student_relation_keys(student)
    )
    if student.freeze_projections:
        return relation_anchor
    if not bool(torch.isfinite(student.initial_projection_weights).all()):
        raise ValueError("Checkpoint has no full-chain projection reference")
    terms = {
        key: (student.projections[key].weight - initial).square().sum()
        / (student.input_dim * student.student_dim)
        for key, initial in zip(
            student.projection_keys, student.initial_projection_weights
        )
    }
    if student.projection_mode == "split":
        projection_anchor = (
            0.5 * (terms["table_query"] + terms["table_target"])
            + terms["text"]
            + terms["image"]
        )
    else:
        projection_anchor = sum(terms.values())
    return relation_anchor + projection_anchor


def student_evidence_anchor_loss(
    student: StudentJoinabilityModel,
) -> torch.Tensor:
    """Return the anchor term for the four table/evidence relations."""

    evidence_keys = {
        student.relation_key("table", "text"),
        student.relation_key("text", "table"),
        student.relation_key("table", "image"),
        student.relation_key("image", "table"),
    }
    return sum(
        student.relation_residual_squared_norm(key) / (student.student_dim**2)
        for key in evidence_keys
    )


def student_relation_drift(
    student: StudentJoinabilityModel,
) -> dict[str, float]:
    """Measure the Frobenius distance from identity for every relation."""

    with torch.no_grad():
        return {
            key: float(student.relation_residual_squared_norm(key).sqrt().cpu())
            for key in _student_relation_keys(student)
        }


def student_projection_drift(
    student: StudentJoinabilityModel,
    *,
    reference: str = "pca",
) -> dict[str, float | None]:
    """Measure normalized projection drift from the initialization basis."""

    normalizer = math.sqrt(student.input_dim * student.student_dim)
    if reference not in {"pca", "stage_start"}:
        raise ValueError("Projection reference must be pca or stage_start")
    anchors = (
        student.initial_projection_weights
        if reference == "pca"
        else student.stage_initial_projection_weights
    )
    with torch.no_grad():
        return {
            key: float(
                torch.linalg.vector_norm(
                    student.projections[key].weight - initial
                ).cpu()
                / normalizer
            ) if bool(torch.isfinite(initial).all()) else None
            for key, initial in zip(
                student.projection_keys, anchors
            )
        }


def student_projection_references(student: StudentJoinabilityModel) -> dict[str, Any]:
    """Fingerprint the two persisted references without conflating their roles."""

    import hashlib

    result: dict[str, Any] = {
        "origin": student.projection_reference_origin,
        "regularizer_reference": "P_PCA",
        "relation_reference": "identity",
    }
    for name, tensor in (
        ("P_PCA", student.initial_projection_weights),
        ("P_stage_start", student.stage_initial_projection_weights),
    ):
        values = tensor.detach().cpu().contiguous()
        available = bool(torch.isfinite(values).all())
        result[name] = {
            "available": available,
            "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest()
            if available else None,
        }
    return result


def student_gradient_norms(student: StudentJoinabilityModel) -> dict[str, float | None]:
    return {
        name: float(parameter.grad.detach().norm().cpu()) if parameter.grad is not None else None
        for name, parameter in student.named_parameters()
        if name.startswith((
            "projections.",
            "projection_residual_inputs.",
            "projection_residual_outputs.",
            "relations.",
            "relation_as.",
            "relation_bs.",
        ))
    }


def _student_relation_keys(student: StudentJoinabilityModel) -> list[str]:
    if student.relation_param == "full":
        return sorted(student.relations)
    return sorted(student.relation_as)


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


def _student_edge_losses(
    student: StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    student_scores: ListScores,
    teacher_scores: ListScores | None,
    supervised_scores: ListScores | None,
    confidence_scores: ListScores | None,
    *,
    ranking_weight: float,
    temperature: float,
    distillation_weight: float,
    edge_bce_weight: float,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
    positive_loss_mode: str = "sum_probability",
) -> dict[str, torch.Tensor]:
    if ranking_weight < 0:
        raise ValueError("ranking_weight must be non-negative")
    supervised = (
        _teacher_edge_loss(
            supervised_scores, positive_loss_mode=positive_loss_mode
        )
        if supervised_scores is not None
        else student_scores.logits.new_zeros(())
    )
    if teacher_scores is None:
        distillation = student_scores.logits.new_zeros(())
    else:
        # Teacher targets may be present only for a relation subset (for
        # example TT rows in the fresh-lineage runner).  Intersect the masks
        # and remove rows with no valid Teacher candidates before softmax;
        # passing an all-False row to distillation_kl would produce NaNs.
        kd_mask = student_scores.candidate_mask & teacher_scores.candidate_mask
        usable = kd_mask.any(dim=-1)
        if usable.any().item():
            distillation = distillation_kl(
                student_scores.logits[usable],
                teacher_scores.logits[usable],
                kd_mask[usable],
                temperature,
            )
        else:
            distillation = student_scores.logits.new_zeros(())
    absolute = student_scores.logits.new_zeros(())
    if edge_bce_weight > 0:
        if confidence_scores is None:
            raise ValueError("Edge BCE requires confidence-logit scores")
        labels, confirmed_mask = edge_confirmed_label_tensors(
            examples,
            confidence_scores.logits.shape[1],
            confidence_scores.logits.device,
        )
        relation_keys = [_edge_relation_key(example) for example in examples]
        absolute = relation_macro_binary_cross_entropy_with_logits(
            confidence_scores.logits,
            labels,
            confirmed_mask,
            relation_keys,
        )
    anchor, weighted_anchor = _anchor_losses(
        student, anchor_weight, anchor_weight_evidence
    )
    return {
        "loss": (
            ranking_weight * supervised
            + distillation_weight * distillation
            + edge_bce_weight * absolute
            + weighted_anchor
        ),
        "supervised_loss": supervised,
        "weighted_supervised_loss": ranking_weight * supervised,
        "distillation_loss": distillation,
        "absolute_loss": absolute,
        "anchor_loss": anchor,
        "weighted_anchor_loss": weighted_anchor,
    }


def _edge_relation_key(example: EdgeExample) -> str:
    if example.source_type is None or example.destination_type is None:
        return "unspecified"
    return StudentJoinabilityModel.relation_key(
        example.source_type, example.destination_type
    )


def _recall_at_one(scores: ListScores) -> tuple[int, int]:
    """Count lists whose highest-scoring candidate is one of the positives."""

    positive_mask = scores.positive_mask
    if positive_mask is None:
        positive_mask = torch.zeros_like(scores.candidate_mask)
        positive_mask.scatter_(1, scores.positive_indices.unsqueeze(1), True)
    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        return 0, 0
    winners = scores.logits.masked_fill(~scores.candidate_mask, -torch.inf).argmax(dim=1)
    hits = positive_mask.gather(1, winners.unsqueeze(1)).squeeze(1)
    return int(hits[valid_rows].sum().item()), int(valid_rows.sum().item())


@torch.no_grad()
def _edge_list_metrics(
    model: TeacherJoinabilityModel | StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    store: FeatureStore,
    device: torch.device,
    batch_size: int,
    score_space: str,
) -> dict[str, Any]:
    """Evaluate list R@1 per directed relation and its equal-weight macro mean."""

    model.eval()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for batch in progress(
        _batches(examples, batch_size), desc="Dev R@1", unit="batch", leave=False
    ):
        batch_scores = score_edge_batch(
            model, batch, store, device, student_score_space=score_space
        )
        for relation in sorted({_edge_relation_key(example) for example in batch}):
            row_mask = torch.tensor(
                [_edge_relation_key(example) == relation for example in batch],
                dtype=torch.bool,
                device=device,
            )
            hits, lists = _recall_at_one(batch_scores.select(row_mask))
            counts[relation][0] += hits
            counts[relation][1] += lists
    by_relation = {
        relation: {
            "hits@1": hits,
            "lists": lists,
            "recall@1": hits / lists if lists else 0.0,
        }
        for relation, (hits, lists) in sorted(counts.items())
    }
    recalls = [values["recall@1"] for values in by_relation.values()]
    return {
        "relations": len(by_relation),
        "macro_recall@1": _mean(recalls),
        "by_relation": by_relation,
    }


@torch.no_grad()
def _target_list_metrics(
    model: TeacherJoinabilityModel | StudentJoinabilityModel,
    examples: Sequence[TargetExample],
    store: FeatureStore,
    aggregator: PathAggregator,
    device: torch.device,
    batch_size: int,
    score_space: str,
) -> dict[str, Any]:
    """Evaluate direct and evidence target-list R@1 on fixed candidates."""

    model.eval()
    direct_hits = direct_lists = evidence_hits = evidence_lists = 0
    for batch in progress(
        _batches(examples, batch_size), desc="Dev target R@1", unit="batch", leave=False
    ):
        batch_scores = score_target_batch(
            model,
            batch,
            store,
            device,
            aggregator,
            student_score_space=score_space,
        )
        hits, lists = _recall_at_one(batch_scores.direct)
        direct_hits += hits
        direct_lists += lists
        hits, lists = _recall_at_one(batch_scores.evidence)
        evidence_hits += hits
        evidence_lists += lists
    return {
        "direct": {
            "hits@1": direct_hits,
            "lists": direct_lists,
            "recall@1": direct_hits / direct_lists if direct_lists else 0.0,
        },
        "evidence": {
            "hits@1": evidence_hits,
            "lists": evidence_lists,
            "recall@1": evidence_hits / evidence_lists if evidence_lists else 0.0,
        },
    }


def confirmed_edge_label_summary(
    examples: Sequence[EdgeExample],
) -> dict[str, Any]:
    """Summarize confirmed positives/negatives without treating unknowns as zero."""

    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    lists_with_confirmed = 0
    for example in examples:
        labels = example.confirmed_labels or (None,) * len(example.candidate_ids)
        if len(labels) != len(example.candidate_ids):
            raise ValueError("confirmed_labels must align with edge candidates")
        relation = _edge_relation_key(example)
        if any(label is not None for label in labels):
            lists_with_confirmed += 1
        for label in labels:
            name = "unknown" if label is None else "positive" if label == 1 else "negative"
            relation_counts[relation][name] += 1
            totals[name] += 1
    return {
        "candidates": sum(totals.values()),
        "confirmed": totals["positive"] + totals["negative"],
        "positive": totals["positive"],
        "negative": totals["negative"],
        "unknown": totals["unknown"],
        "lists": len(examples),
        "lists_with_confirmed": lists_with_confirmed,
        "by_relation": {
            relation: dict(sorted(counts.items()))
            for relation, counts in sorted(relation_counts.items())
        },
    }


def _edge_losses_by_relation(
    student: StudentJoinabilityModel,
    examples: Sequence[EdgeExample],
    student_scores: ListScores,
    teacher_scores: ListScores | None,
    supervised_scores: ListScores | None,
    confidence_scores: ListScores | None,
    *,
    ranking_weight: float = 1.0,
    temperature: float,
    distillation_weight: float,
    edge_bce_weight: float,
    positive_loss_mode: str = "sum_probability",
) -> dict[str, dict[str, int | torch.Tensor]]:
    """Return diagnostic edge losses without adding a second anchor term."""

    relations = [_edge_relation_key(example) for example in examples]
    result: dict[str, dict[str, int | torch.Tensor]] = {}
    for relation in sorted(set(relations)):
        indices = [
            index for index, value in enumerate(relations) if value == relation
        ]
        row_mask = torch.tensor(
            [value == relation for value in relations],
            dtype=torch.bool,
            device=student_scores.logits.device,
        )
        relation_examples = [examples[index] for index in indices]
        losses = _student_edge_losses(
            student,
            relation_examples,
            student_scores.select(row_mask),
            teacher_scores.select(row_mask) if teacher_scores is not None else None,
            (
                supervised_scores.select(row_mask)
                if supervised_scores is not None
                else None
            ),
            (
                confidence_scores.select(row_mask)
                if confidence_scores is not None
                else None
            ),
            ranking_weight=ranking_weight,
            temperature=temperature,
            distillation_weight=distillation_weight,
            edge_bce_weight=edge_bce_weight,
            anchor_weight=0.0,
            anchor_weight_evidence=0.0,
            positive_loss_mode=positive_loss_mode,
        )
        confirmed = sum(
            label is not None
            for example in relation_examples
            for label in (
                example.confirmed_labels
                or (None,) * len(example.candidate_ids)
            )
        )
        result[relation] = {
            "lists": len(relation_examples),
            "confirmed_candidates": confirmed,
            **losses,
        }
    return result


class _EdgeRelationLossTracker:
    """Accumulate relation diagnostics without synchronizing every GPU batch."""

    _NAMES = (
        "loss",
        "supervised_loss",
        "distillation_loss",
        "absolute_loss",
    )

    def __init__(self) -> None:
        self.lists: Counter[str] = Counter()
        self.confirmed: Counter[str] = Counter()
        self.weights: dict[str, Counter[str]] = defaultdict(Counter)
        self.pending: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.sums: dict[str, Counter[str]] = defaultdict(Counter)

    def add(self, values: dict[str, dict[str, int | torch.Tensor]]) -> None:
        for relation, relation_values in values.items():
            lists = int(relation_values["lists"])
            confirmed = int(relation_values["confirmed_candidates"])
            self.lists[relation] += lists
            self.confirmed[relation] += confirmed
            for name in self._NAMES:
                weight = confirmed if name == "absolute_loss" else lists
                if weight == 0:
                    continue
                value = relation_values[name]
                assert isinstance(value, torch.Tensor)
                self.pending[relation][name].append(value.detach() * weight)
                self.weights[relation][name] += weight

    def flush(self) -> None:
        for relation, by_name in self.pending.items():
            for name, pending in by_name.items():
                if pending:
                    self.sums[relation][name] += float(
                        torch.stack(pending).sum().cpu()
                    )
                    pending.clear()

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        self.flush()
        return {
            relation: {
                "lists": self.lists[relation],
                "confirmed_candidates": self.confirmed[relation],
                **{
                    name: (
                        self.sums[relation][name]
                        / self.weights[relation][name]
                        if self.weights[relation][name]
                        else None
                    )
                    for name in self._NAMES
                },
            }
            for relation in sorted(self.lists)
        }


def _student_path_losses(
    student: StudentJoinabilityModel,
    student_scores: TargetScores,
    teacher_scores: TargetScores | None,
    expanded_direct_scores: ListScores | None,
    *,
    temperature: float,
    distillation_weight: float,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
    distillation_rows: torch.Tensor | None,
    positive_loss_mode: str = "sum_probability",
    evidence_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if evidence_loss_weight < 0:
        raise ValueError("evidence_loss_weight must be non-negative")
    supervised, direct_supervised, evidence_supervised = _path_supervised_losses(
        student_scores,
        positive_loss_mode=positive_loss_mode,
    )
    if expanded_direct_scores is not None:
        direct_supervised = _teacher_edge_loss(
            expanded_direct_scores,
            positive_loss_mode=positive_loss_mode,
        )
        supervised = direct_supervised + evidence_supervised
    if teacher_scores is not None:
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
    direct_loss = direct_supervised + distillation_weight * direct_distillation
    evidence_loss = evidence_supervised + distillation_weight * evidence_distillation
    return {
        "loss": direct_loss + evidence_loss_weight * evidence_loss + weighted_anchor,
        "supervised_loss": supervised,
        "distillation_loss": distillation,
        "direct_supervised_loss": direct_supervised,
        "evidence_supervised_loss": evidence_supervised,
        "direct_distillation_loss": direct_distillation,
        "evidence_distillation_loss": evidence_distillation,
        "anchor_loss": anchor,
        "weighted_anchor_loss": weighted_anchor,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _loss_refresh_due(step: int, step_count: int) -> bool:
    return step % LOSS_REFRESH_STEPS == 0 or step == step_count


def _flush_loss_values(
    pending: list[torch.Tensor], values: list[float]
) -> None:
    values.extend(torch.stack(pending).cpu().tolist())
    pending.clear()


class _LossTracker:
    """Named batch-loss bookkeeping with periodic pending-buffer flushes."""

    def __init__(self, names: Sequence[str]) -> None:
        self.values: dict[str, list[float]] = {name: [] for name in names}
        self._pending: dict[str, list[torch.Tensor]] = {name: [] for name in names}

    def add(self, values: dict[str, torch.Tensor]) -> None:
        for name, value in values.items():
            self._pending[name].append(value.detach())

    def flush(self) -> None:
        for name, pending in self._pending.items():
            if pending:
                self.values[name].extend(torch.stack(pending).cpu().tolist())
                pending.clear()

    def mean(self, name: str) -> float:
        return _mean(self.values[name])


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
    score_space: str,
    edge_bce_weight: float,
    positive_loss_mode: str = "sum_probability",
) -> float:
    model.eval()
    losses = []
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        scores = score_edge_batch(
            model,
            batch,
            store,
            device,
            student_score_space=score_space,
        )
        confidence_scores = (
            score_edge_batch(
                model,
                batch,
                store,
                device,
                student_score_space="confidence_logit",
            )
            if edge_bce_weight > 0
            else None
        )
        losses.append(
            float(
                _teacher_edge_losses(
                    batch,
                    scores,
                    confidence_scores,
                    edge_bce_weight,
                    positive_loss_mode=positive_loss_mode,
                )["loss"]
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
    score_space: str,
    positive_loss_mode: str = "sum_probability",
) -> float:
    model.eval()
    losses = []
    for batch in progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    ):
        scores = score_target_batch(
            model,
            batch,
            store,
            device,
            aggregator,
            student_score_space=score_space,
        )
        loss, _direct, _evidence = _path_supervised_losses(
            scores, positive_loss_mode=positive_loss_mode
        )
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
    edge_bce_weight: float,
    anchor_weight: float,
    anchor_weight_evidence: float | None,
    in_batch_negatives: bool,
    in_batch_max_negatives: int,
    student_score_space: str,
    ranking_weight: float,
    ranking_score_space: str,
    ranking_temperature: float,
    use_global_positive_mask: bool,
    positive_loss_mode: str = "sum_probability",
) -> float:
    student.eval()
    losses = []
    known_positives = global_edge_positive_ids(examples)
    for batch_index, batch in enumerate(progress(
        _batches(examples, batch_size), desc="Dev", unit="batch", leave=False
    )):
        teacher_scores = (
            _edge_teacher_scores(batch, device)
            if distillation_weight > 0
            else None
        )
        if in_batch_negatives:
            supervised_scores = score_edge_batch_in_batch(
                student,
                batch,
                store,
                device,
                max_negatives=in_batch_max_negatives,
                student_score_space=ranking_score_space,
                known_positive_ids=known_positives,
                use_global_positive_mask=use_global_positive_mask,
                sampling_seed=0,
                sampling_context=f"dev:{batch_index}",
            )
        else:
            supervised_scores = score_edge_batch(
                student,
                batch,
                store,
                device,
                student_score_space=ranking_score_space,
            )
        if ranking_temperature != 1.0:
            supervised_scores = ListScores(
                supervised_scores.logits / ranking_temperature,
                supervised_scores.candidate_mask,
                supervised_scores.positive_indices,
                supervised_scores.positive_mask,
            )
        student_scores = score_edge_batch(
            student,
            batch,
            store,
            device,
            student_score_space=student_score_space,
        )
        confidence_scores = (
            score_edge_batch(
                student,
                batch,
                store,
                device,
                student_score_space="confidence_logit",
            )
            if edge_bce_weight > 0
            else None
        )
        objective = _student_edge_losses(
            student,
            batch,
            student_scores,
            teacher_scores,
            supervised_scores,
            confidence_scores,
            ranking_weight=ranking_weight,
            temperature=temperature,
            distillation_weight=distillation_weight,
            edge_bce_weight=edge_bce_weight,
            anchor_weight=anchor_weight,
            anchor_weight_evidence=anchor_weight_evidence,
            positive_loss_mode=positive_loss_mode,
        )
        losses.append(float(objective["loss"]))
    return _mean(losses)


def _validate_cached_path_aggregation(
    examples: Sequence[TargetExample], aggregator: PathAggregator
) -> None:
    for example in examples:
        config = example.teacher_score_config
        if config is not None and (
            config.evidence_aggregation != aggregator.evidence_aggregation
            or config.evidence_top_k != aggregator.top_k
            or config.evidence_temperature != aggregator.temperature
            or config.evidence_power != aggregator.power
            or config.path_combination != aggregator.path_combination
            or config.evidence_threshold != aggregator.threshold
            or config.evidence_target_temperature
            != aggregator.target_temperature
            or config.row_support_model != aggregator.row_support_model
            or config.row_support_model_sha256
            != aggregator.row_support_model_sha256
            or config.row_support_top_l != aggregator.row_support_top_l
            or config.evidence_content_keys != aggregator.evidence_content_keys
            or config.evidence_content_keys_sha256
            != aggregator.evidence_content_keys_sha256
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
    student_score_space: str,
    positive_loss_mode: str = "sum_probability",
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
            student,
            batch,
            store,
            device,
            aggregator,
            student_score_space=student_score_space,
        )
        expanded_direct_scores = None
        if in_batch_negatives:
            expanded_direct_scores = score_target_direct_batch_in_batch(
                student,
                batch,
                store,
                device,
                max_negatives=in_batch_max_negatives,
                rng=rng,
                student_score_space=student_score_space,
            )
        distillation_rows = (
            torch.tensor(
                [example.dataset in distillation_datasets for example in batch],
                dtype=torch.bool,
                device=device,
            )
            if distillation_datasets is not None
            else None
        )
        objective = _student_path_losses(
            student,
            student_scores,
            teacher_scores,
            expanded_direct_scores,
            temperature=temperature,
            distillation_weight=distillation_weight,
            anchor_weight=anchor_weight,
            anchor_weight_evidence=anchor_weight_evidence,
            distillation_rows=distillation_rows,
            positive_loss_mode=positive_loss_mode,
        )
        losses.append(float(objective["loss"]))
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
    score_space: str = "raw_logit",
    edge_bce_weight: float = 0.0,
    positive_loss_mode: str = "sum_probability",
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[EdgeExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[EdgeExample] = (),
    epoch_callback: EpochCallback[TeacherJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    if edge_bce_weight < 0:
        raise ValueError("edge_bce_weight must be non-negative")
    if edge_bce_weight > 0 and not model.confidence_transform:
        raise ValueError("Teacher edge BCE requires the confidence transform")
    history = []
    rng = random.Random(seed)
    epoch_bar = progress(range(epochs), desc="Teacher edge", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        tracker = _LossTracker(("loss", "ranking_loss", "absolute_loss"))
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
            scores = score_edge_batch(
                model,
                batch,
                store,
                device,
                student_score_space=score_space,
            )
            confidence_scores = (
                score_edge_batch(
                    model,
                    batch,
                    store,
                    device,
                    student_score_space="confidence_logit",
                )
                if edge_bce_weight > 0
                else None
            )
            objective = _teacher_edge_losses(
                batch,
                scores,
                confidence_scores,
                edge_bce_weight,
                positive_loss_mode=positive_loss_mode,
            )
            _optimize(objective["loss"], optimizer)
            tracker.add(objective)
            if _loss_refresh_due(step, len(batches)):
                tracker.flush()
                batch_bar.set_postfix(loss=f"{tracker.mean('loss'):.4f}")
        train_loss = tracker.mean("loss")
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "optimizer_updates": len(batches),
            "examples_seen": len(sampled),
            "ranking_loss": tracker.mean("ranking_loss"),
            "absolute_loss": tracker.mean("absolute_loss"),
            "confirmed_labels": confirmed_edge_label_summary(sampled),
        }
        if dev_examples:
            values["dev_loss"] = _teacher_edge_objective(
                model,
                dev_examples,
                store,
                device,
                batch_size,
                score_space,
                edge_bce_weight,
                positive_loss_mode,
            )
            values["dev_edge"] = _edge_list_metrics(
                model,
                dev_examples,
                store,
                device,
                batch_size,
                score_space,
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
    score_space: str = "raw_logit",
    positive_loss_mode: str = "sum_probability",
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
            scores = score_target_batch(
                model,
                batch,
                store,
                device,
                aggregator,
                student_score_space=score_space,
            )
            loss, direct_loss, evidence_loss = _path_supervised_losses(
                scores, positive_loss_mode=positive_loss_mode
            )
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
            "optimizer_updates": len(batches),
            "examples_seen": len(sampled),
            "direct_loss": _mean(direct_losses),
            "evidence_loss": _mean(evidence_losses),
        }
        if dev_examples:
            values["dev_loss"] = _teacher_path_objective(
                model,
                dev_examples,
                store,
                aggregator,
                device,
                batch_size,
                score_space,
                positive_loss_mode,
            )
            values["dev_target_lists"] = _target_list_metrics(
                model,
                dev_examples,
                store,
                aggregator,
                device,
                batch_size,
                score_space,
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
    edge_bce_weight: float = 0.0,
    anchor_weight: float = 0.0,
    anchor_weight_evidence: float | None = None,
    in_batch_negatives: bool = False,
    in_batch_max_negatives: int = 256,
    edge_type_oversample: dict[str, int] | None = None,
    student_score_space: str = "raw_logit",
    ranking_weight: float = 1.0,
    ranking_score_space: str = "raw_logit",
    ranking_temperature: float = 1.0,
    use_global_positive_mask: bool = True,
    max_optimizer_updates: int | None = None,
    positive_loss_mode: str = "sum_probability",
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[EdgeExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[EdgeExample] = (),
    epoch_callback: EpochCallback[StudentJoinabilityModel] | None = None,
    eval_epoch_zero: bool = False,
    checkpoint_steps: Sequence[int] = (),
    step_callback: EpochCallback[StudentJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    if min(edge_bce_weight, ranking_weight) < 0:
        raise ValueError("edge_bce_weight and ranking_weight must be non-negative")
    if ranking_temperature <= 0:
        raise ValueError("ranking_temperature must be positive")
    if max_optimizer_updates is not None and max_optimizer_updates <= 0:
        raise ValueError("max_optimizer_updates must be positive")
    if edge_bce_weight > 0 and not student.confidence_transform:
        raise ValueError("Edge BCE requires the Student confidence transform")
    history = []
    rng = random.Random(seed)
    known_positives = global_edge_positive_ids([*examples, *hard_examples])
    optimizer_updates = 0
    def evaluate(record: dict[str, Any]) -> None:
        if not dev_examples:
            return
        record["dev_loss"] = _student_edge_objective(
            student, dev_examples, store, device, batch_size, temperature,
            distillation_weight, edge_bce_weight, anchor_weight,
            anchor_weight_evidence, in_batch_negatives, in_batch_max_negatives,
            student_score_space, ranking_weight, ranking_score_space,
            ranking_temperature, use_global_positive_mask, positive_loss_mode,
        )
        record["dev_edge"] = _edge_list_metrics(
            student, dev_examples, store, device, batch_size, ranking_score_space,
        )

    if eval_epoch_zero:
        record = {"epoch": 0, "training_state": "initial", "optimizer_updates_total": 0,
                  "dataset_samples": {}, "source_samples": {"base": 0, "hard": 0}}
        evaluate(record)
        if _finish_epoch(history, record, student, epoch_callback):
            return history
    epoch_bar = progress(range(epochs), desc="Student edge", unit="epoch")
    for epoch in epoch_bar:
        student.train()
        tracker = _LossTracker(
            (
                "loss",
                "supervised_loss",
                "weighted_supervised_loss",
                "distillation_loss",
                "absolute_loss",
                "anchor_loss",
                "weighted_anchor_loss",
            )
        )
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
        processed_examples: list[EdgeExample] = []
        expansion_audit: dict[str, int] = {}
        epoch_start_updates = optimizer_updates
        for step, batch in enumerate(batch_bar, 1):
            if (
                max_optimizer_updates is not None
                and optimizer_updates >= max_optimizer_updates
            ):
                break
            processed_examples.extend(batch)
            teacher_scores = (
                _edge_teacher_scores(batch, device)
                if distillation_weight > 0
                else None
            )
            if in_batch_negatives:
                supervised_scores = score_edge_batch_in_batch(
                    student,
                    batch,
                    store,
                    device,
                    max_negatives=in_batch_max_negatives,
                    student_score_space=ranking_score_space,
                    known_positive_ids=known_positives,
                    use_global_positive_mask=use_global_positive_mask,
                    sampling_seed=seed,
                    sampling_context=f"epoch={epoch}:step={step}",
                    expansion_audit=expansion_audit,
                )
            else:
                supervised_scores = score_edge_batch(
                    student,
                    batch,
                    store,
                    device,
                    student_score_space=ranking_score_space,
                )
            if ranking_temperature != 1.0:
                supervised_scores = ListScores(
                    supervised_scores.logits / ranking_temperature,
                    supervised_scores.candidate_mask,
                    supervised_scores.positive_indices,
                    supervised_scores.positive_mask,
                )
            student_scores = score_edge_batch(
                student,
                batch,
                store,
                device,
                student_score_space=student_score_space,
            )
            confidence_scores = (
                score_edge_batch(
                    student,
                    batch,
                    store,
                    device,
                    student_score_space="confidence_logit",
                )
                if edge_bce_weight > 0
                else None
            )
            objective = _student_edge_losses(
                student,
                batch,
                student_scores,
                teacher_scores,
                supervised_scores,
                confidence_scores,
                ranking_weight=ranking_weight,
                temperature=temperature,
                distillation_weight=distillation_weight,
                edge_bce_weight=edge_bce_weight,
                anchor_weight=anchor_weight,
                anchor_weight_evidence=anchor_weight_evidence,
                positive_loss_mode=positive_loss_mode,
            )
            _optimize(objective["loss"], optimizer)
            optimizer_updates += 1
            tracker.add(objective)
            if optimizer_updates in checkpoint_steps and step_callback is not None:
                step_record = {"epoch": epoch + 1, "optimizer_updates_total": optimizer_updates,
                               "loss": float(objective["loss"].detach()),
                               "in_batch_expansion": dict(expansion_audit)}
                evaluate(step_record)
                if step_callback(optimizer_updates, student, step_record):
                    raise RuntimeError("Step callbacks must not silently truncate the registered update budget")
                student.train()
            if _loss_refresh_due(step, len(batches)):
                tracker.flush()
                batch_bar.set_postfix(loss=f"{tracker.mean('loss'):.4f}")
        tracker.flush()
        train_loss = tracker.mean("loss")
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "optimizer_updates": optimizer_updates - epoch_start_updates,
            "optimizer_updates_total": optimizer_updates,
            "optimizer_update_budget": max_optimizer_updates,
            "optimizer_update_budget_exhausted": (
                max_optimizer_updates is not None
                and optimizer_updates >= max_optimizer_updates
            ),
            "examples_seen": len(processed_examples),
            "supervised_loss": tracker.mean("supervised_loss"),
            "weighted_supervised_loss": tracker.mean(
                "weighted_supervised_loss"
            ),
            "distillation_loss": tracker.mean("distillation_loss"),
            "absolute_loss": tracker.mean("absolute_loss"),
            "confirmed_labels": confirmed_edge_label_summary(sampled),
            "anchor_loss": tracker.mean("anchor_loss"),
            "weighted_anchor_loss": tracker.mean("weighted_anchor_loss"),
            "in_batch_expansion": expansion_audit,
        }
        evaluate(values)
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(
            epoch + 1, values, processed_examples, source_samples
        )
        if _finish_epoch(history, record, student, epoch_callback):
            break
        if values["optimizer_update_budget_exhausted"]:
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
    use_global_positive_mask: bool = True,
    relation_loss_weights: dict[str, float] | None = None,
    student_score_space: str = "raw_logit",
    positive_loss_mode: str = "sum_probability",
    continuous_edge_examples: Sequence[EdgeExample] = (),
    continuous_edge_dev_examples: Sequence[EdgeExample] = (),
    continuous_edge_weight: float = 1.0,
    continuous_edge_bce_weight: float = 0.0,
    continuous_edge_batch_size: int | None = None,
    continuous_edge_ranking_score_space: str | None = None,
    continuous_edge_ranking_temperature: float = 1.0,
    max_optimizer_updates: int | None = None,
    dataset_sampling_alpha: float = 0.0,
    hard_examples: Sequence[TargetExample] = (),
    hard_fraction: float = 0.5,
    dev_examples: Sequence[TargetExample] = (),
    epoch_callback: EpochCallback[StudentJoinabilityModel] | None = None,
    checkpoint_steps: Sequence[int] = (),
    step_callback: EpochCallback[StudentJoinabilityModel] | None = None,
) -> list[dict[str, Any]]:
    if continuous_edge_weight < 0:
        raise ValueError("continuous_edge_weight must be non-negative")
    if continuous_edge_bce_weight < 0:
        raise ValueError("continuous_edge_bce_weight must be non-negative")
    if continuous_edge_ranking_temperature <= 0:
        raise ValueError("continuous_edge_ranking_temperature must be positive")
    if continuous_edge_bce_weight > 0 and not student.confidence_transform:
        raise ValueError(
            "Continuous edge BCE requires the Student confidence transform"
        )
    if continuous_edge_dev_examples and not continuous_edge_examples:
        raise ValueError(
            "Continuous edge dev data requires continuous edge training data"
        )
    edge_batch_size = continuous_edge_batch_size or batch_size
    edge_ranking_score_space = (
        continuous_edge_ranking_score_space or student_score_space
    )
    if edge_batch_size <= 0:
        raise ValueError("continuous_edge_batch_size must be positive")
    continuous_known_positives = global_edge_positive_ids(
        continuous_edge_examples
    )
    if max_optimizer_updates is not None and max_optimizer_updates <= 0:
        raise ValueError("max_optimizer_updates must be positive")
    history = []
    rng = random.Random(seed)
    continuous_edge_rng = random.Random(f"{seed}:continuous_edge")
    optimizer_updates = 0
    _validate_cached_path_aggregation(
        [*examples, *hard_examples, *dev_examples], aggregator
    )
    epoch_bar = progress(range(epochs), desc="Student path", unit="epoch")
    for epoch in epoch_bar:
        student.train()
        tracker_names = [
            "loss",
            "supervised_loss",
            "distillation_loss",
            "direct_supervised_loss",
            "evidence_supervised_loss",
            "direct_distillation_loss",
            "evidence_distillation_loss",
            "anchor_loss",
            "weighted_anchor_loss",
        ]
        if continuous_edge_examples:
            tracker_names.extend(
                [
                    "path_loss",
                    "continuous_edge_loss",
                    "continuous_edge_supervised_loss",
                    "continuous_edge_distillation_loss",
                    "continuous_edge_absolute_loss",
                ]
            )
        tracker = _LossTracker(tracker_names)
        edge_relation_tracker = _EdgeRelationLossTracker()
        sampled, source_samples = sample_mixed_epoch(
            examples,
            hard_examples,
            rng,
            hard_fraction=hard_fraction,
            dataset_sampling_alpha=dataset_sampling_alpha,
        )
        batches = _batches(sampled, batch_size)
        if max_optimizer_updates is not None:
            remaining_updates = max_optimizer_updates - optimizer_updates
            if remaining_updates <= 0:
                break
            if len(batches) > remaining_updates:
                batches = batches[:remaining_updates]
                sampled = [example for batch in batches for example in batch]
                hard_ids = {id(example) for example in hard_examples}
                hard_count = sum(id(example) in hard_ids for example in sampled)
                source_samples = {
                    "base": len(sampled) - hard_count,
                    "hard": hard_count,
                }
        edge_sampled: list[EdgeExample] = []
        edge_batches: list[list[EdgeExample]] = []
        if continuous_edge_examples:
            edge_sampled = _sample_balanced_count(
                continuous_edge_examples,
                len(batches) * edge_batch_size,
                continuous_edge_rng,
                dataset_sampling_alpha,
            )
            continuous_edge_rng.shuffle(edge_sampled)
            edge_batches = _batches(edge_sampled, edge_batch_size)
            if len(edge_batches) != len(batches):
                raise RuntimeError(
                    "Every path batch must have exactly one continuous edge batch"
                )
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
            student_scores = score_target_batch(
                student,
                batch,
                store,
                device,
                aggregator,
                relation_loss_weights=relation_loss_weights,
                student_score_space=student_score_space,
            )
            expanded_direct_scores = None
            if in_batch_negatives:
                expanded_direct_scores = score_target_direct_batch_in_batch(
                    student,
                    batch,
                    store,
                    device,
                    max_negatives=in_batch_max_negatives,
                    rng=rng,
                    student_score_space=student_score_space,
                )
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
            path_objective = _student_path_losses(
                student,
                student_scores,
                teacher_scores,
                expanded_direct_scores,
                temperature=temperature,
                distillation_weight=distillation_weight,
                anchor_weight=anchor_weight,
                anchor_weight_evidence=anchor_weight_evidence,
                distillation_rows=distillation_rows,
                positive_loss_mode=positive_loss_mode,
            )
            tracked_objective = dict(path_objective)
            total_loss = path_objective["loss"]
            if continuous_edge_examples:
                edge_batch = edge_batches[step - 1]
                edge_teacher_scores = (
                    _edge_teacher_scores(edge_batch, device)
                    if distillation_weight > 0
                    else None
                )
                if in_batch_negatives:
                    edge_sampling_context = (
                        f"path_epoch={epoch}:step={step}:continuous_edge"
                    )
                    edge_expanded_scores = score_edge_batch_in_batch(
                        student,
                        edge_batch,
                        store,
                        device,
                        max_negatives=in_batch_max_negatives,
                        rng=rng,
                        student_score_space=student_score_space,
                        known_positive_ids=continuous_known_positives,
                        use_global_positive_mask=use_global_positive_mask,
                        sampling_seed=seed,
                        sampling_context=edge_sampling_context,
                    )
                    edge_student_scores = restrict_list_scores(
                        edge_expanded_scores,
                        [len(example.candidate_ids) for example in edge_batch],
                        device,
                    )
                    edge_supervised_scores = (
                        edge_expanded_scores
                        if edge_ranking_score_space == student_score_space
                        else score_edge_batch_in_batch(
                            student,
                            edge_batch,
                            store,
                            device,
                            max_negatives=in_batch_max_negatives,
                            student_score_space=edge_ranking_score_space,
                            known_positive_ids=continuous_known_positives,
                            use_global_positive_mask=use_global_positive_mask,
                            sampling_seed=seed,
                            sampling_context=edge_sampling_context,
                        )
                    )
                else:
                    edge_student_scores = score_edge_batch(
                        student,
                        edge_batch,
                        store,
                        device,
                        student_score_space=student_score_space,
                    )
                    edge_supervised_scores = (
                        edge_student_scores
                        if edge_ranking_score_space == student_score_space
                        else score_edge_batch(
                            student,
                            edge_batch,
                            store,
                            device,
                            student_score_space=edge_ranking_score_space,
                        )
                    )
                if continuous_edge_ranking_temperature != 1.0:
                    edge_supervised_scores = ListScores(
                        edge_supervised_scores.logits
                        / continuous_edge_ranking_temperature,
                        edge_supervised_scores.candidate_mask,
                        edge_supervised_scores.positive_indices,
                        edge_supervised_scores.positive_mask,
                        edge_supervised_scores.candidate_ids,
                    )
                edge_confidence_scores = (
                    score_edge_batch(
                        student,
                        edge_batch,
                        store,
                        device,
                        student_score_space="confidence_logit",
                    )
                    if continuous_edge_bce_weight > 0
                    else None
                )
                edge_objective = _student_edge_losses(
                    student,
                    edge_batch,
                    edge_student_scores,
                    edge_teacher_scores,
                    edge_supervised_scores,
                    edge_confidence_scores,
                    ranking_weight=1.0,
                    temperature=temperature,
                    distillation_weight=distillation_weight,
                    edge_bce_weight=continuous_edge_bce_weight,
                    anchor_weight=0.0,
                    anchor_weight_evidence=0.0,
                    positive_loss_mode=positive_loss_mode,
                )
                with torch.no_grad():
                    edge_relation_tracker.add(
                        _edge_losses_by_relation(
                            student,
                            edge_batch,
                            edge_student_scores,
                            edge_teacher_scores,
                            edge_supervised_scores,
                            edge_confidence_scores,
                            ranking_weight=1.0,
                            temperature=temperature,
                            distillation_weight=distillation_weight,
                            edge_bce_weight=continuous_edge_bce_weight,
                            positive_loss_mode=positive_loss_mode,
                        )
                    )
                total_loss = (
                    path_objective["loss"]
                    + continuous_edge_weight * edge_objective["loss"]
                )
                tracked_objective.update(
                    {
                        "loss": total_loss,
                        "path_loss": path_objective["loss"],
                        "continuous_edge_loss": edge_objective["loss"],
                        "continuous_edge_supervised_loss": edge_objective[
                            "supervised_loss"
                        ],
                        "continuous_edge_distillation_loss": edge_objective[
                            "distillation_loss"
                        ],
                        "continuous_edge_absolute_loss": edge_objective[
                            "absolute_loss"
                        ],
                    }
                )
            _optimize(total_loss, optimizer)
            optimizer_updates += 1
            tracker.add(tracked_objective)
            if optimizer_updates in checkpoint_steps and step_callback is not None:
                step_record = {"epoch": epoch + 1, "optimizer_updates_total": optimizer_updates,
                               "loss": float(total_loss.detach())}
                if step_callback(optimizer_updates, student, step_record):
                    raise RuntimeError("Step callbacks must not silently truncate the registered update budget")
                student.train()
            if _loss_refresh_due(step, len(batches)):
                tracker.flush()
                edge_relation_tracker.flush()
                batch_bar.set_postfix(loss=f"{tracker.mean('loss'):.4f}")
        train_loss = tracker.mean("loss")
        values = {
            "loss": train_loss,
            "train_loss": train_loss,
            "optimizer_updates": len(batches),
            "cumulative_optimizer_updates": optimizer_updates,
            "optimizer_update_budget": max_optimizer_updates,
            "optimizer_update_budget_exhausted": (
                max_optimizer_updates is not None
                and optimizer_updates >= max_optimizer_updates
            ),
            "examples_seen": len(sampled),
            "supervised_loss": tracker.mean("supervised_loss"),
            "distillation_loss": tracker.mean("distillation_loss"),
            "direct_supervised_loss": tracker.mean("direct_supervised_loss"),
            "evidence_supervised_loss": tracker.mean("evidence_supervised_loss"),
            "direct_distillation_loss": tracker.mean("direct_distillation_loss"),
            "evidence_distillation_loss": tracker.mean(
                "evidence_distillation_loss"
            ),
            "anchor_loss": tracker.mean("anchor_loss"),
            "weighted_anchor_loss": tracker.mean("weighted_anchor_loss"),
        }
        if continuous_edge_examples:
            values.update(
                {
                    "path_loss": tracker.mean("path_loss"),
                    "path_batches": len(batches),
                    "continuous_edge_loss": tracker.mean(
                        "continuous_edge_loss"
                    ),
                    "continuous_edge_supervised_loss": tracker.mean(
                        "continuous_edge_supervised_loss"
                    ),
                    "continuous_edge_distillation_loss": tracker.mean(
                        "continuous_edge_distillation_loss"
                    ),
                    "continuous_edge_absolute_loss": tracker.mean(
                        "continuous_edge_absolute_loss"
                    ),
                    "continuous_edge_batches": len(edge_batches),
                    "continuous_edge_examples_seen": len(edge_sampled),
                    "additional_optimizer_updates": 0,
                    "continuous_edge_participation": (
                        confirmed_edge_label_summary(edge_sampled)
                    ),
                    "continuous_edge_loss_by_relation": (
                        edge_relation_tracker.summary()
                    ),
                }
            )
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
                student_score_space,
                positive_loss_mode,
            )
        if continuous_edge_dev_examples:
            values["dev_continuous_edge_loss"] = _student_edge_objective(
                student,
                continuous_edge_dev_examples,
                store,
                device,
                edge_batch_size,
                temperature,
                distillation_weight,
                continuous_edge_bce_weight,
                0.0,
                0.0,
                in_batch_negatives,
                in_batch_max_negatives,
                student_score_space,
                1.0,
                edge_ranking_score_space,
                continuous_edge_ranking_temperature,
                use_global_positive_mask,
                positive_loss_mode,
            )
            if "dev_loss" in values:
                values["dev_joint_loss"] = (
                    values["dev_loss"]
                    + continuous_edge_weight
                    * values["dev_continuous_edge_loss"]
                )
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            **({"dev": f"{values['dev_loss']:.4f}"} if "dev_loss" in values else {}),
        )
        record = _epoch_record(epoch + 1, values, sampled, source_samples)
        should_stop = _finish_epoch(history, record, student, epoch_callback)
        if should_stop or values["optimizer_update_budget_exhausted"]:
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
        payload["path_aggregation"] = aggregator.config()
    if isinstance(model, StudentJoinabilityModel):
        payload["projection_references"] = student_projection_references(model)
    return payload

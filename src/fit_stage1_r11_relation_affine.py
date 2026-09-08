#!/usr/bin/env python
"""Fit R11 D3 positive-slope relation maps on cal-fit edge labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from mmdd_progress import progress

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import (
    EdgeExample,
    TargetExample,
    load_edge_examples,
    load_target_examples,
)
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.scoring import score_edge_batch


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relation(example: EdgeExample) -> str:
    return f"{example.source_type}_to_{example.destination_type}"


def _batches(values: Sequence[EdgeExample], size: int) -> list[Sequence[EdgeExample]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _scored_labels(
    examples: Sequence[EdgeExample],
    student: torch.nn.Module,
    store: FeatureStore,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, tuple[list[float], list[int]]], bool]:
    scores: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    ranking_unchanged = True
    for batch in progress(
        _batches(examples, batch_size), desc="Score affine labels", unit="batch"
    ):
        with torch.inference_mode():
            values = score_edge_batch(
                student,
                batch,
                store,
                device,
                student_score_space="raw_logit",
            ).logits.detach().cpu()
        for row, example in zip(values, batch):
            relation = _relation(example)
            if example.confirmed_labels is None:
                continue
            raw = row[: len(example.candidate_ids)].tolist()
            for score, label in zip(raw, example.confirmed_labels):
                if label is not None:
                    scores[relation].append(float(score))
                    labels[relation].append(int(label))
            ranking_unchanged &= len(raw) == len(example.candidate_ids)
    return {
        relation: (scores[relation], labels[relation])
        for relation in sorted(set(scores) | set(labels))
    }, ranking_unchanged


def _fit(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    raw = torch.tensor(scores, dtype=torch.float64)
    expected = torch.tensor(labels, dtype=torch.float64)
    alpha = torch.tensor(
        math.log(math.expm1(1.0 - 1e-6)), dtype=torch.float64, requires_grad=True
    )
    bias = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [alpha, bias], lr=1.0, max_iter=100, tolerance_grad=1e-10
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = F.softplus(alpha) + 1e-6
        loss = F.binary_cross_entropy_with_logits(scale * raw + bias, expected)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(F.softplus(alpha).detach() + 1e-6), float(bias.detach())


def _metrics(
    scores: Sequence[float], labels: Sequence[int], scale: float, bias: float
) -> dict[str, float | int | None]:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not scores:
        return {
            "pairs": 0,
            "positive": 0,
            "negative": 0,
            "brier": None,
            "nll": None,
        }
    logits = torch.tensor(scores, dtype=torch.float64) * scale + bias
    expected = torch.tensor(labels, dtype=torch.float64)
    probability = torch.sigmoid(logits)
    return {
        "pairs": len(labels),
        "positive": positives,
        "negative": negatives,
        "brier": (
            float((probability - expected).square().mean())
            if positives and negatives
            else None
        ),
        "nll": (
            float(F.binary_cross_entropy_with_logits(logits, expected))
            if positives and negatives
            else None
        ),
    }


def _ranking_invariant(
    examples: Sequence[EdgeExample],
    scored: dict[str, tuple[list[float], list[int]]],
    models: dict[str, dict[str, Any]],
) -> bool:
    offsets = defaultdict(int)
    for example in examples:
        relation = _relation(example)
        confirmed = [
            label for label in (example.confirmed_labels or ()) if label is not None
        ]
        count = len(confirmed)
        raw = scored.get(relation, ([], []))[0][
            offsets[relation] : offsets[relation] + count
        ]
        offsets[relation] += count
        model = models.get(relation, {"scale": 1.0, "bias": 0.0})
        calibrated = [model["scale"] * value + model["bias"] for value in raw]
        raw_order = sorted(range(len(raw)), key=lambda index: (-raw[index], index))
        calibrated_order = sorted(
            range(len(raw)), key=lambda index: (-calibrated[index], index)
        )
        if raw_order != calibrated_order:
            return False
    return True


def _mapped_score(
    score: float,
    source_type: str,
    destination_type: str,
    models: dict[str, dict[str, Any]],
) -> float:
    model = models.get(
        f"{source_type}_to_{destination_type}",
        {"scale": 1.0, "bias": 0.0},
    )
    return float(model["scale"]) * score + float(model["bias"])


def _top_id(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return min(scores, key=lambda key: (-scores[key], key))


def _rrf_top(
    direct: dict[str, float], evidence: dict[str, float], *, rrf_k: int = 60
) -> str | None:
    direct_order = sorted(direct, key=lambda key: (-direct[key], key))
    evidence_order = sorted(evidence, key=lambda key: (-evidence[key], key))
    direct_rank = {key: rank for rank, key in enumerate(direct_order, 1)}
    evidence_rank = {key: rank for rank, key in enumerate(evidence_order, 1)}
    fused = {
        key: (
            (1.0 / (rrf_k + direct_rank[key]) if key in direct_rank else 0.0)
            + (1.0 / (rrf_k + evidence_rank[key]) if key in evidence_rank else 0.0)
        )
        for key in set(direct_rank) | set(evidence_rank)
    }
    return _top_id(fused)


@torch.no_grad()
def _path_ranking_metrics(
    examples: Sequence[TargetExample],
    student: torch.nn.Module,
    store: FeatureStore,
    device: torch.device,
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    counts = {
        "lists": len(examples),
        "evidence_lists": 0,
        "raw_direct_hits@1": 0,
        "affine_direct_hits@1": 0,
        "raw_evidence_hits@1": 0,
        "affine_evidence_hits@1": 0,
        "raw_equal_rrf_hits@1": 0,
        "affine_equal_rrf_hits@1": 0,
        "direct_top1_switches": 0,
        "evidence_top1_switches": 0,
        "equal_rrf_top1_switches": 0,
        "evidence_top_path_modality_switches": 0,
    }
    modality_counts = {
        "raw": defaultdict(int),
        "affine": defaultdict(int),
    }
    for example in progress(
        examples, desc="Evaluate affine path ranking", unit="list"
    ):
        query = store.embedding_features(example.query_id).for_scoring(
            device, include_hidden=False
        )
        targets = {
            candidate.target_id: store.embedding_features(
                candidate.target_id
            ).for_scoring(device, include_hidden=False)
            for candidate in example.candidates
        }
        direct_sources = [query] * len(example.candidates)
        direct_destinations = [targets[candidate.target_id] for candidate in example.candidates]
        path_rows = [
            (candidate.target_id, evidence_id)
            for candidate in example.candidates
            for evidence_id in sorted(candidate.evidence_ids)
        ]
        evidence = {
            evidence_id: store.embedding_features(evidence_id).for_scoring(
                device, include_hidden=False
            )
            for _, evidence_id in path_rows
        }
        sources = [
            *direct_sources,
            *(query for _ in path_rows),
            *(evidence[evidence_id] for _, evidence_id in path_rows),
        ]
        destinations = [
            *direct_destinations,
            *(evidence[evidence_id] for _, evidence_id in path_rows),
            *(targets[target_id] for target_id, _ in path_rows),
        ]
        values = student.score_pairs_in_space(
            sources, destinations, "raw_logit"
        ).detach().cpu().tolist()
        direct_count = len(example.candidates)
        path_count = len(path_rows)
        direct_values = values[:direct_count]
        query_evidence = values[direct_count : direct_count + path_count]
        evidence_target = values[direct_count + path_count :]
        raw_direct = {
            candidate.target_id: float(score)
            for candidate, score in zip(example.candidates, direct_values)
        }
        affine_direct = {
            target_id: _mapped_score(score, "table", "table", models)
            for target_id, score in raw_direct.items()
        }
        raw_paths: dict[str, list[float]] = defaultdict(list)
        affine_paths: dict[str, list[float]] = defaultdict(list)
        raw_best_path: tuple[float, str] | None = None
        affine_best_path: tuple[float, str] | None = None
        for (target_id, evidence_id), first, second in zip(
            path_rows, query_evidence, evidence_target
        ):
            evidence_type = evidence[evidence_id].object_type
            raw_score = float(first) + float(second)
            affine_score = _mapped_score(
                float(first), "table", evidence_type, models
            ) + _mapped_score(float(second), evidence_type, "table", models)
            raw_paths[target_id].append(raw_score)
            affine_paths[target_id].append(affine_score)
            if raw_best_path is None or raw_score > raw_best_path[0]:
                raw_best_path = (raw_score, evidence_type)
            if affine_best_path is None or affine_score > affine_best_path[0]:
                affine_best_path = (affine_score, evidence_type)
        raw_evidence = {
            target_id: float(torch.logsumexp(torch.tensor(scores), dim=0))
            for target_id, scores in raw_paths.items()
        }
        affine_evidence = {
            target_id: float(torch.logsumexp(torch.tensor(scores), dim=0))
            for target_id, scores in affine_paths.items()
        }
        positives = set(example.positive_target_ids)
        raw_direct_top = _top_id(raw_direct)
        affine_direct_top = _top_id(affine_direct)
        raw_evidence_top = _top_id(raw_evidence)
        affine_evidence_top = _top_id(affine_evidence)
        raw_fused_top = _rrf_top(raw_direct, raw_evidence)
        affine_fused_top = _rrf_top(affine_direct, affine_evidence)
        counts["raw_direct_hits@1"] += int(raw_direct_top in positives)
        counts["affine_direct_hits@1"] += int(affine_direct_top in positives)
        counts["raw_equal_rrf_hits@1"] += int(raw_fused_top in positives)
        counts["affine_equal_rrf_hits@1"] += int(affine_fused_top in positives)
        counts["direct_top1_switches"] += int(raw_direct_top != affine_direct_top)
        counts["equal_rrf_top1_switches"] += int(raw_fused_top != affine_fused_top)
        if raw_evidence:
            counts["evidence_lists"] += 1
            counts["raw_evidence_hits@1"] += int(raw_evidence_top in positives)
            counts["affine_evidence_hits@1"] += int(
                affine_evidence_top in positives
            )
            counts["evidence_top1_switches"] += int(
                raw_evidence_top != affine_evidence_top
            )
        if raw_best_path is not None and affine_best_path is not None:
            modality_counts["raw"][raw_best_path[1]] += 1
            modality_counts["affine"][affine_best_path[1]] += 1
            counts["evidence_top_path_modality_switches"] += int(
                raw_best_path[1] != affine_best_path[1]
            )
    lists = int(counts["lists"])
    evidence_lists = int(counts["evidence_lists"])
    return {
        **counts,
        "raw_direct_recall@1": counts["raw_direct_hits@1"] / lists,
        "affine_direct_recall@1": counts["affine_direct_hits@1"] / lists,
        "raw_evidence_recall@1": (
            counts["raw_evidence_hits@1"] / evidence_lists
            if evidence_lists
            else 0.0
        ),
        "affine_evidence_recall@1": (
            counts["affine_evidence_hits@1"] / evidence_lists
            if evidence_lists
            else 0.0
        ),
        "raw_equal_rrf_recall@1": counts["raw_equal_rrf_hits@1"] / lists,
        "affine_equal_rrf_recall@1": (
            counts["affine_equal_rrf_hits@1"] / lists
        ),
        "top_path_modality": {
            regime: dict(sorted(values.items()))
            for regime, values in modality_counts.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    student = load_student(checkpoint, device).eval()
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    fit_examples = load_edge_examples(Path(args.cal_fit), split="train")
    check_examples = load_edge_examples(Path(args.cal_check), split="train")
    target_check_examples = load_target_examples(
        Path(args.target_cal_check), split="train"
    )
    fit, fit_shapes_ok = _scored_labels(
        fit_examples, student, store, device, args.batch_size
    )
    check, check_shapes_ok = _scored_labels(
        check_examples, student, store, device, args.batch_size
    )
    models = {}
    diagnostics = {}
    for relation in sorted(set(fit) | set(check)):
        fit_scores, fit_labels = fit.get(relation, ([], []))
        check_scores, check_labels = check.get(relation, ([], []))
        has_fit_classes = bool(sum(fit_labels)) and sum(fit_labels) < len(fit_labels)
        if has_fit_classes:
            scale, bias = _fit(fit_scores, fit_labels)
            status = "fitted"
        else:
            scale, bias = 1.0, 0.0
            status = "identity_insufficient_fit_classes"
        models[relation] = {
            "scale": scale,
            "bias": bias,
            "status": status,
        }
        diagnostics[relation] = {
            "cal_fit": _metrics(fit_scores, fit_labels, scale, bias),
            "cal_check": _metrics(check_scores, check_labels, scale, bias),
            "cal_check_reportable": (
                bool(sum(check_labels)) and sum(check_labels) < len(check_labels)
            ),
        }
    ranking_unchanged = (
        fit_shapes_ok
        and check_shapes_ok
        and _ranking_invariant(fit_examples, fit, models)
        and _ranking_invariant(check_examples, check, models)
        and all(float(model["scale"]) > 0 for model in models.values())
    )
    payload = {
        "format_version": 1,
        "kind": "r11_relation_monotonic_affine",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "fit_split": "cal_fit",
        "check_split": "cal_check",
        "unknown_label_policy": "excluded",
        "models": models,
        "diagnostics": diagnostics,
        "same_relation_ranking_unchanged": ranking_unchanged,
        "target_cal_check": str(Path(args.target_cal_check).resolve()),
        "target_cal_check_sha256": checkpoint_fingerprint(
            Path(args.target_cal_check)
        ),
        "path_ranking": _path_ranking_metrics(
            target_check_examples, student, store, device, models
        ),
    }
    _write_json(Path(args.output), payload)
    print(json.dumps({"status": "pass", "output": args.output}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cal-fit", required=True)
    parser.add_argument("--cal-check", required=True)
    parser.add_argument("--target-cal-check", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.feature_cache_size) <= 0:
        parser.error("Batch and cache sizes must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())

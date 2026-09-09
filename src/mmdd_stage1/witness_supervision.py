"""Train-only positive witness supervision for the R13 path objective."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .data import TargetExample
from .scoring import TargetScores


def logmeanexp(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("logmeanexp requires a non-empty vector")
    return torch.logsumexp(values, dim=0) - math.log(values.numel())


def witness_auxiliary_loss(
    examples: Sequence[TargetExample],
    scores: TargetScores,
    *,
    content_keys: Mapping[str, str] | None = None,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Return the hierarchical R13 W loss without changing deployment scores."""

    if scores.path_logits is None or len(scores.path_logits) != len(examples):
        raise ValueError("W requires aligned per-path logits")
    content_keys = content_keys or {}
    query_losses = []
    stats: dict[str, int | str] = {
        "label_scope": "any_known_witness_by_row",
        "queries": len(examples),
        "eligible_queries": 0,
        "eligible_pairs": 0,
        "eligible_rows": 0,
        "supported_path_ids": 0,
        "supported_content_groups": 0,
        "negative_targets": 0,
        "skipped_no_negative_paths": 0,
        "skipped_no_witness_metadata": 0,
        "skipped_no_candidate_intersection": 0,
    }
    for example, example_paths in zip(examples, scores.path_logits):
        positives = set(example.positive_target_ids)
        negative_bags = [
            logmeanexp(path_values)
            for candidate, path_values in zip(example.candidates, example_paths)
            if candidate.target_id not in positives and path_values.numel()
        ]
        if not negative_bags:
            stats["skipped_no_negative_paths"] += 1
            continue
        bq = logmeanexp(torch.stack(negative_bags))
        stats["negative_targets"] += len(negative_bags)
        witness_rows = example.positive_evidence_rows_by_target or {}
        witness_ids = example.positive_evidence_by_target or {}
        target_losses = []
        had_metadata = bool(witness_rows or witness_ids)
        for candidate, path_values in zip(example.candidates, example_paths):
            target_id = candidate.target_id
            if target_id not in positives or not path_values.numel():
                continue
            evidence_ids = tuple(sorted(candidate.evidence_ids))
            score_by_id = {
                evidence_id: path_values[index]
                for index, evidence_id in enumerate(evidence_ids)
            }
            by_row: dict[int, set[str]] = defaultdict(set)
            for evidence_id, rows in witness_rows.get(target_id, {}).items():
                for row in rows:
                    by_row[int(row)].add(str(evidence_id))
            if not by_row and target_id in witness_ids:
                by_row[-1].update(str(value) for value in witness_ids[target_id])
            row_losses = []
            for supported_ids in by_row.values():
                available = sorted(supported_ids & score_by_id.keys())
                if not available:
                    stats["skipped_no_candidate_intersection"] += 1
                    continue
                scores_by_content: dict[str, list[torch.Tensor]] = defaultdict(list)
                for evidence_id in available:
                    scores_by_content[
                        content_keys.get(evidence_id, evidence_id)
                    ].append(score_by_id[evidence_id])
                unique_scores = torch.stack(
                    [torch.stack(values).max() for values in scores_by_content.values()]
                )
                aw = logmeanexp(unique_scores)
                row_losses.append(torch.nn.functional.softplus(bq - aw))
                stats["eligible_rows"] += 1
                stats["supported_path_ids"] += len(available)
                stats["supported_content_groups"] += len(scores_by_content)
            if row_losses:
                target_losses.append(torch.stack(row_losses).mean())
                stats["eligible_pairs"] += 1
        if target_losses:
            query_losses.append(torch.stack(target_losses).mean())
            stats["eligible_queries"] += 1
        elif not had_metadata:
            stats["skipped_no_witness_metadata"] += 1
    if not query_losses:
        return scores.direct.logits.sum() * 0.0, stats
    return torch.stack(query_losses).mean(), stats

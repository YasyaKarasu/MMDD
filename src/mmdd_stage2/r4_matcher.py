"""S2-R4 matching contract: EXACT_NORMALIZED and a frozen SEMANTIC replay, both returning
the actual target row ids they matched.

One-to-many is legitimate and is never collapsed to an argmax row. A value appearing
somewhere in the target is not, by itself, a supported join: the value matcher runs after
generation and may read the target column, while the generator never can.

The semantic path reuses the locked backend and the project's frozen thresholds. It
returns row ids but keeps the original value-matching semantics exactly, and the
compatibility check proves that by reproducing ``coverage`` and ``mean_similarity``.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .r4_metrics import numeric_equal, normalize, strict_equal

EXACT_NORMALIZED_VERSION = 'r4-exact-normalized-v1'
SEMANTIC_FROZEN_VERSION = 'semantic-frozen-similarity0.8-coverage0.6'
SIMILARITY_THRESHOLD = 0.8
MIN_ROW_COVERAGE = 0.6
BUILD_COVERAGE_DIAGNOSTIC = 0.4


def column_values(table: dict[str, Any], column_index: int) -> list[str]:
    from mmdd_stage2.data import column_values as _column_values

    return list(_column_values(table, column_index))


def exact_normalized_matches(predicted: str, target_values: Sequence[str]) -> list[int]:
    """Every target row whose normalized cell equals the normalized prediction.

    Substring matching is explicitly not used: a name, a number and a date are different
    types and a containment test would silently join them.
    """
    if predicted is None:
        return []
    key = normalize(predicted)
    if not key:
        return []
    return [index for index, value in enumerate(target_values) if normalize(value) == key]


def exact_matches_report(values: list[dict], target_values: Sequence[str]) -> dict:
    rows: dict[int, list[str]] = {}
    per_value = []
    for item in values:
        matched = exact_normalized_matches(item.get('value'), target_values)
        per_value.append({'value': item.get('value'), 'matched_row_ids': matched,
                          'support_level': item.get('support_level')})
        for row_id in matched:
            rows.setdefault(row_id, []).append(str(item.get('value')))
    return {
        'matcher': EXACT_NORMALIZED_VERSION,
        'values_matched': sum(bool(v['matched_row_ids']) for v in per_value),
        'values_total': len(per_value),
        'matched_row_ids': sorted(rows),
        'row_to_values': {str(k): v for k, v in sorted(rows.items())},
        'per_value': per_value,
        'one_to_many_preserved': True,
    }


@torch.inference_mode()
def semantic_frozen_matches(values: Sequence[str], target_values: Sequence[str], backend,
                            *, similarity_threshold: float = SIMILARITY_THRESHOLD,
                            min_row_coverage: float = MIN_ROW_COVERAGE) -> dict:
    """Return matched row ids while preserving the original coverage/mean_similarity semantics."""
    if not values or not target_values:
        return {'matcher': SEMANTIC_FROZEN_VERSION, 'coverage': 0.0, 'mean_similarity': 0.0,
                'joinable': False, 'per_value': [], 'matched_row_ids': []}
    embeddings = backend.embed_texts([*values, *target_values])
    query_vectors = torch.nn.functional.normalize(embeddings[:len(values)].float(), dim=-1)
    target_vectors = torch.nn.functional.normalize(embeddings[len(values):].float(), dim=-1)
    similarity = query_vectors @ target_vectors.T

    per_value = []
    best_scores = []
    for index, value in enumerate(values):
        text = str(value).strip()
        if not text:
            best_scores.append(0.0)
            per_value.append({'value': value, 'matched_row_ids': [], 'best_similarity': 0.0})
            continue
        if any(strict_equal(value, target) or normalize(value) == normalize(target)
               for target in target_values):
            # The original function rewards an exact target match with the maximum score.
            best_scores.append(1.0)
        else:
            best_scores.append(float(similarity[index].max()))
        matched = [row for row in range(len(target_values))
                   if float(similarity[index, row]) >= similarity_threshold]
        per_value.append({'value': value, 'matched_row_ids': matched,
                          'best_similarity': float(similarity[index].max())})
    coverage = sum(bool(str(value).strip()) and score >= similarity_threshold
                   for value, score in zip(values, best_scores, strict=True)) / len(best_scores)
    return {
        'matcher': SEMANTIC_FROZEN_VERSION,
        'similarity_threshold': similarity_threshold,
        'min_row_coverage': min_row_coverage,
        'coverage': coverage,
        'mean_similarity': sum(best_scores) / len(best_scores),
        'joinable': coverage >= min_row_coverage,
        'per_value': per_value,
        'matched_row_ids': sorted({row for item in per_value for row in item['matched_row_ids']}),
        'note': 'coverage is the production criterion; 0.4 is only a build-coverage diagnostic',
    }


def compatibility_check(values: Sequence[str], target_values: Sequence[str], backend) -> dict:
    """The row-id matcher must reproduce the original coverage and mean_similarity."""
    from .verifier import semantic_joinability

    embeddings = backend.embed_texts([*values, *target_values])
    original = semantic_joinability(
        list(values), list(target_values),
        query_embeddings=embeddings[:len(values)], target_embeddings=embeddings[len(values):],
        similarity_threshold=SIMILARITY_THRESHOLD, min_coverage=MIN_ROW_COVERAGE)
    extended = semantic_frozen_matches(values, target_values, backend)
    return {
        'coverage_delta': abs(original.coverage - extended['coverage']),
        'mean_similarity_delta': abs(original.mean_similarity - extended['mean_similarity']),
        'joinable_agree': original.joinable == extended['joinable'],
        'reproduced': (abs(original.coverage - extended['coverage']) < 1e-9
                       and abs(original.mean_similarity - extended['mean_similarity']) < 1e-9
                       and original.joinable == extended['joinable']),
    }


def claim_coverage(rows: list[dict], field: str, *, denominator: str = 'all_visible_query_rows',
                   threshold: float = MIN_ROW_COVERAGE) -> dict:
    """Claim coverage always divides by every visible query row, not by successful rows."""
    total = len(rows)
    matched = sum(bool(row.get(field)) for row in rows)
    return {
        'denominator': denominator,
        'rows_total': total,
        'rows_matched': matched,
        'coverage': (matched / total) if total else None,
        'primary_threshold': MIN_ROW_COVERAGE,
        'build_diagnostic_threshold': BUILD_COVERAGE_DIAGNOSTIC,
        'reached_primary_threshold': bool(total) and (matched / total) >= threshold,
    }

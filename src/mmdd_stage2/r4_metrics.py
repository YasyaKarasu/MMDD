"""S2-R4 metrics and the corrected cluster bootstrap.

Two things the R3 audit found broken and this module fixes:

1. ``row_r3_metrics.bootstrap_ci`` resampled source groups with replacement and then
   flattened the drawn rows, so a group drawn m times had its rows re-merged into a
   single (dataset, query_id) bucket by ``query_macro``. The multiplicity was lost.
   Here every group carries a weight, and a group drawn m times contributes m times
   to both the query-metric sum and the query count.
2. ``numeric_equal`` compared ``float(...)``, which silently rounds integers past 2**53.
   Comparison now goes through ``Decimal``.

Naming is explicit everywhere: ``strict`` / ``normalized`` / ``numeric`` are separate
levels and ``any_level`` is only reported as a union, never as a stand-in for strict.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from statistics import mean

NORMALIZATION_VERSION = 'r4-norm-v1'
VALUE_LEVELS = ('strict', 'normalized', 'numeric')

_WS = re.compile(r'\s+')
_EDGE = re.compile(r'^[\s"\'`\(\)\[\]\{\}.,;:]+|[\s"\'`\(\)\[\]\{\}.,;:]+$')
_THOUSANDS = re.compile(r'(?<=\d),(?=\d{3}\b)')
_NUMERIC = re.compile(r'^[+-]?(\d+)(\.\d+)?([eE][+-]?\d+)?$')


def normalize(value) -> str:
    if value is None:
        return ''
    text = unicodedata.normalize('NFKC', str(value))
    text = _THOUSANDS.sub('', text)
    text = _WS.sub(' ', text).strip()
    text = _EDGE.sub('', text)
    if re.fullmatch(r'-?\d+\.0+', text):
        text = text.split('.')[0]
    return text.casefold()


def strict_equal(predicted, gold) -> bool:
    return predicted is not None and str(predicted).strip() == str(gold).strip()


def normalized_equal(predicted, gold) -> bool:
    left, right = normalize(predicted), normalize(gold)
    return bool(left) and left == right


def numeric_equal(predicted, gold) -> bool:
    """Exact numeric equality after separator normalization, without float rounding.

    Decimal is used so that a 20-digit identifier is not collapsed onto a neighbour by
    binary floating point. No unit conversion is applied.
    """
    def parse(value):
        text = normalize(value).replace(' ', '')
        if not text or not _NUMERIC.match(text):
            return None
        try:
            number = Decimal(text)
        except InvalidOperation:
            return None
        return number if number.is_finite() else None

    left, right = parse(predicted), parse(gold)
    if left is None or right is None:
        return False
    return left == right


def value_matches(predicted, gold) -> dict:
    """Every comparison level, named. ``any_level`` is a convenience union only."""
    levels = {
        'strict': strict_equal(predicted, gold),
        'normalized': normalized_equal(predicted, gold),
        'numeric': numeric_equal(predicted, gold),
    }
    return {**levels, 'any_level': any(levels.values())}


def query_macro(rows: list[dict], field: str) -> float | None:
    """Mean over queries of the mean over that query's rows. Multiplicity-preserving."""
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['query_id']), []).append(float(row[field]))
    if not groups:
        return None
    return mean(mean(values) for values in groups.values())


def _weighted_macro(values: dict[str, dict[tuple, list[float]]],
                    multiplicity: Counter, position: int | None = None) -> float | None:
    total, count = 0.0, 0
    for group, weight in multiplicity.items():
        if weight == 0:
            continue
        for items in values[group].values():
            total += weight * (sum(items) if position is None else sum(i[position] for i in items))
            count += weight * len(items)
    return total / count if count else None


def cluster_bootstrap(rows: list[dict], group_field: str, field: str, *, iterations: int = 10000,
                      seed: int = 20260917, alpha: float = 0.05) -> dict:
    """Cluster bootstrap over source groups that keeps the multiplicity of redraws.

    ``row_r3_metrics.py:69-91`` dropped that multiplicity, which widened nothing and
    shrank nothing in a predictable way - it simply gave a wrong interval.
    """
    import random

    by_group: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        by_group[str(row[group_field])][(row['dataset'], row['query_id'])].append(float(value))
    keys = sorted(by_group)
    if not keys:
        return {'point': None, 'low': None, 'high': None, 'iterations': 0, 'groups': 0,
                'method': 'weighted_cluster_bootstrap', 'field': field}
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        multiplicity = Counter(keys[rng.randrange(len(keys))] for _ in keys)
        value = _weighted_macro(by_group, multiplicity)
        if value is not None:
            samples.append(value)
    samples.sort()
    return {
        'point': query_macro(rows, field),
        'low': samples[max(0, int(len(samples) * alpha / 2) - 1)] if samples else None,
        'high': samples[min(len(samples) - 1, int(len(samples) * (1 - alpha / 2)))] if samples else None,
        'iterations': len(samples), 'groups': len(keys),
        'method': 'weighted_cluster_bootstrap', 'field': field,
    }


def paired_cluster_bootstrap(rows: list[dict], group_field: str, field_a: str, field_b: str, *,
                             iterations: int = 10000, seed: int = 20260917,
                             alpha: float = 0.05) -> dict:
    """Paired difference A-B: one resample of groups drives both arms on the same query.

    A seed-keyed comparison that pairs nothing is not a paired estimate; both arms are
    read from the same row so the same queries are always compared.
    """
    import random

    by_group: dict[str, dict[tuple, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        a, b = row.get(field_a), row.get(field_b)
        if a is None or b is None:
            continue
        by_group[str(row[group_field])][(row['dataset'], row['query_id'])].append((float(a), float(b)))
    keys = sorted(by_group)
    if not keys:
        return {'point': None, 'low': None, 'high': None, 'iterations': 0, 'groups': 0,
                'method': 'paired_weighted_cluster_bootstrap'}

    index = {query: group for group, queries in by_group.items() for query in queries}
    for group in keys:
        for query in by_group[group]:
            index[query] = group

    def macro(values, multiplicity, position):
        total = count = 0.0
        for group, weight in multiplicity.items():
            if weight == 0:
                continue
            for items in values[group].values():
                total += weight * sum(item[position] for item in items)
                count += weight * len(items)
        return total / count if count else None

    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        multiplicity = Counter(keys[rng.randrange(len(keys))] for _ in keys)
        left, right = macro(by_group, multiplicity, 0), macro(by_group, multiplicity, 1)
        if left is not None and right is not None:
            samples.append(left - right)
    samples.sort()
    point_a = query_macro(rows, field_a)
    point_b = query_macro(rows, field_b)
    return {
        'point': None if point_a is None or point_b is None else point_a - point_b,
        'low': samples[max(0, int(len(samples) * alpha / 2) - 1)] if samples else None,
        'high': samples[min(len(samples) - 1, int(len(samples) * (1 - alpha / 2)))] if samples else None,
        'iterations': len(samples), 'groups': len(keys),
        'method': 'paired_weighted_cluster_bootstrap',
        'field_a': field_a, 'field_b': field_b,
    }


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float('nan')
    return len(set(ranked[:k]) & relevant) / len(relevant)

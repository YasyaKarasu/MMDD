"""Value normalization and row-level metrics for the R3 pilot.

Normalization is fixed here, before any pilot output is scored, and is reported
alongside strict equality so that no semantic threshold can turn a mismatch into a match.
"""
from __future__ import annotations

import math
import re
import unicodedata
from statistics import mean

NORMALIZATION_VERSION = 'r3-norm-v1'

_WS = re.compile(r'\s+')
_EDGE = re.compile(r'^[\s"\'`\(\)\[\]\{\}.,;:]+|[\s"\'`\(\)\[\]\{\}.,;:]+$')
_THOUSANDS = re.compile(r'(?<=\d),(?=\d{3}\b)')


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
    """Exact numeric equality after separator normalization. No unit conversion is applied."""
    try:
        left = float(normalize(predicted).replace(' ', ''))
        right = float(normalize(gold).replace(' ', ''))
    except (TypeError, ValueError):
        return False
    return math.isfinite(left) and math.isfinite(right) and left == right


def value_matches(predicted, gold) -> dict:
    """Report every comparison level; `any_level` is only a convenience union."""
    return {'strict': strict_equal(predicted, gold),
            'normalized': normalized_equal(predicted, gold),
            'numeric': numeric_equal(predicted, gold),
            'any_level': bool(strict_equal(predicted, gold) or normalized_equal(predicted, gold)
                              or numeric_equal(predicted, gold))}


def query_macro(rows: list[dict], field: str) -> float | None:
    if not rows:
        return None
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['query_id']), []).append(float(row[field]))
    return mean(mean(values) for values in groups.values())


def bootstrap_ci(rows: list[dict], group_field: str, field: str, *, iterations: int = 2000,
                 seed: int = 20260916, alpha: float = 0.05) -> dict:
    """Cluster bootstrap over source groups; query-macro is recomputed inside each resample."""
    import random
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[group_field]), []).append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        drawn = [groups[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [row for bucket in drawn for row in bucket]
        value = query_macro(flat, field)
        if value is not None:
            samples.append(value)
    samples.sort()
    if not samples:
        return {'point': None, 'low': None, 'high': None, 'iterations': 0, 'groups': len(keys)}
    low = samples[max(0, int(len(samples) * alpha / 2) - 1)]
    high = samples[min(len(samples) - 1, int(len(samples) * (1 - alpha / 2)))]
    return {'point': query_macro(rows, field), 'low': low, 'high': high,
            'iterations': len(samples), 'groups': len(keys)}

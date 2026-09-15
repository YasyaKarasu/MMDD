"""Query-macro recall and real D/E fusion shared by R26 evaluation stages."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def query_metrics(ranking: Sequence[str], positives: Sequence[str], ks: Sequence[int]) -> dict[str, float]:
    """De-duplicate before TopK; absent/failed rankings contribute zero."""
    truth = set(positives)
    if not truth:
        raise ValueError("Queries without qrels must be excluded when freezing the population")
    unique = list(dict.fromkeys(ranking))
    result = {"raw_recall": len(truth.intersection(unique)) / len(truth)}
    for k in ks:
        hits = len(truth.intersection(unique[:k]))
        result[f"recall@{k}"] = hits / len(truth)
        result[f"any_hit_at_{k}"] = float(hits > 0)
    return result


def population_metrics(rankings: Mapping[str, Sequence[str]], qrels: Mapping[str, Sequence[str]],
                       ks: Sequence[int]) -> dict[str, float]:
    rows = [query_metrics(rankings.get(q, []), truth, ks) for q, truth in qrels.items()]
    if not rows:
        raise ValueError("Empty frozen query population")
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def fuse_channels(direct: Sequence[dict], evidence: Sequence[dict] | None,
                  column_alpha: Mapping[str, float] | None = None) -> dict:
    """RRF over true Direct100 and E path ranks; QT-over-U is never an input."""
    if evidence is None:
        raise ValueError("missing_evidence_channel")
    d = {str(row["target_id"]): 1 / (60 + rank) for rank, row in enumerate(direct, 1)}
    e = {str(row["target_id"]): 1 / (60 + rank) for rank, row in enumerate(evidence, 1)}
    scores = [float(row["direct_score"]) for row in direct]
    alpha = 1.0 if len(scores) < 2 else 1 - min(1.0, max(0.0, (scores[0] - scores[1]) / (scores[0] - scores[-1] + 1e-8)))
    union = sorted(d.keys() | e.keys())
    maps = {
        "Equal": {t: d.get(t, 0) + e.get(t, 0) for t in union},
        "Conf": {t: d.get(t, 0) + alpha * e.get(t, 0) for t in union},
    }
    if column_alpha is not None:
        maps["Column"] = {t: d.get(t, 0) + column_alpha.get(t, 1.0) * e.get(t, 0) for t in union}
    return {"rankings": {name: sorted(values, key=lambda t: (-values[t], t)) for name, values in maps.items()},
            "scores": maps, "confidence_alpha": alpha,
            "confidence_reason": "fewer_than_two_candidates" if len(scores) < 2 else "direct_raw_margin"}


def fused_lse(qt_scores: Mapping[str, float], evidence: Sequence[dict]) -> dict[str, float]:
    e = {str(row["target_id"]): float(row["evidence_score"]) for row in evidence}
    return {t: (max(d, e[t]) + math.log1p(math.exp(-abs(d - e[t])))) if t in e else d
            for t, d in qt_scores.items()}

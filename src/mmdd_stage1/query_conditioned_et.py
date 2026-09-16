"""Query-conditioned second-hop adapter and experiment metrics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("e_only", "qe")


class QueryConditionedETAdapter(nn.Module):
    """Residual MLP that changes only the E-to-target query vector."""

    def __init__(self, dimension: int, hidden_dimension: int = 256) -> None:
        super().__init__()
        if dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.dimension = int(dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.input = nn.Linear(4 * dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, dimension)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def features(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        arm: str,
    ) -> torch.Tensor:
        if arm not in ARMS:
            raise ValueError(f"Unknown adapter arm: {arm}")
        if query.shape != evidence.shape or query.shape[-1] != self.dimension:
            raise ValueError("Query and evidence vectors must have equal adapter dimensions")
        if arm == "e_only":
            zero = torch.zeros_like(query)
            return torch.cat((zero, evidence, zero, zero), dim=-1)
        return torch.cat(
            (query, evidence, query * evidence, (query - evidence).abs()), dim=-1
        )

    def forward(
        self,
        query: torch.Tensor,
        evidence: torch.Tensor,
        arm: str,
    ) -> torch.Tensor:
        return self.output(F.gelu(self.input(self.features(query, evidence, arm))))

    def conditioned_query(
        self,
        base_query: torch.Tensor,
        query: torch.Tensor,
        evidence: torch.Tensor,
        arm: str,
    ) -> torch.Tensor:
        return base_query + self(query, evidence, arm)

    def config(self) -> dict[str, int]:
        return {
            "dimension": self.dimension,
            "hidden_dimension": self.hidden_dimension,
            "input_dimension": 4 * self.dimension,
        }


def sum_probability_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive listwise loss used by the Stage-1 Student."""

    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("Logits and positive mask must be equal rank-2 tensors")
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("Every candidate list must contain a positive")
    positive = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=-1)
    return (torch.logsumexp(logits, dim=-1) - positive).mean()


def rank_metrics(ranks: Sequence[int], candidate_count: int) -> dict[str, float]:
    """Summarize one-indexed positive ranks for a fixed evaluation slice."""

    if not ranks:
        return {
            "pairs": 0,
            "recall@1": float("nan"),
            "recall@5": float("nan"),
            "recall@10": float("nan"),
            "recall@20": float("nan"),
            "mrr": float("nan"),
            "positive_median_rank": float("nan"),
            "positive_p90_rank": float("nan"),
        }
    values = np.asarray(ranks, dtype=np.float64)
    return {
        "pairs": int(len(ranks)),
        **{f"recall@{k}": float(np.mean(values <= k)) for k in (1, 5, 10, 20)},
        "mrr": float(np.mean(1.0 / values)),
        "positive_median_rank": float(np.median(values)),
        "positive_p90_rank": float(np.quantile(values, 0.9)),
        "candidate_count": int(candidate_count),
    }


def query_macro_rank_metrics(
    rows: Sequence[Mapping[str, Any]],
    candidate_count: int,
) -> dict[str, float]:
    """Average rank metrics within query before averaging across queries."""

    by_query: dict[str, list[int]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).extend(
            int(value) for value in row["positive_ranks"]
        )
    if not by_query:
        return rank_metrics([], candidate_count)
    per_query = [rank_metrics(values, candidate_count) for values in by_query.values()]
    keys = [
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "mrr",
        "positive_median_rank",
        "positive_p90_rank",
    ]
    return {
        "queries": len(per_query),
        "pairs": len(rows),
        **{key: float(np.mean([row[key] for row in per_query])) for key in keys},
        "candidate_count": int(candidate_count),
    }


def retrieval_metrics(
    ranking: Sequence[str], positives: Sequence[str], ks: Sequence[int] = (10, 20, 50)
) -> dict[str, float]:
    truth = set(positives)
    if not truth:
        raise ValueError("Retrieval metrics require at least one positive")
    unique = list(dict.fromkeys(ranking))
    result = {"raw_recall": len(truth.intersection(unique)) / len(truth)}
    for k in ks:
        result[f"recall@{k}"] = len(truth.intersection(unique[:k])) / len(truth)
    return result


def stable_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def paired_wlt(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, int]:
    common = sorted(set(left) & set(right))
    deltas = [float(left[key]) - float(right[key]) for key in common]
    return {
        "queries": len(common),
        "wins": sum(value > 0 for value in deltas),
        "losses": sum(value < 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
    }


def grouped_paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    samples: int = 2000,
    seed: int = 20260916,
) -> dict[str, float | int]:
    """Paired percentile bootstrap that resamples source groups."""

    common = sorted(set(left) & set(right) & set(groups))
    grouped: dict[str, list[float]] = {}
    for query_id in common:
        grouped.setdefault(groups[query_id], []).append(
            float(left[query_id]) - float(right[query_id])
        )
    group_ids = sorted(grouped)
    if not group_ids:
        raise ValueError("Bootstrap requires common grouped queries")
    observed = float(np.mean([value for values in grouped.values() for value in values]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(group_ids, size=len(group_ids), replace=True)
        values = [value for group_id in selected for value in grouped[str(group_id)]]
        draws[index] = np.mean(values)
    return {
        "queries": len(common),
        "source_groups": len(group_ids),
        "samples": int(samples),
        "delta": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def residual_geometry(
    base: torch.Tensor, residual: torch.Tensor
) -> dict[str, torch.Tensor]:
    base_norm = torch.linalg.vector_norm(base, dim=-1)
    residual_norm = torch.linalg.vector_norm(residual, dim=-1)
    conditioned = base + residual
    cosine = F.cosine_similarity(base, conditioned, dim=-1)
    return {
        "base_norm": base_norm,
        "residual_norm": residual_norm,
        "residual_ratio": residual_norm / base_norm.clamp_min(1e-12),
        "base_conditioned_cosine": cosine,
    }


def mean_finite(values: Sequence[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan")

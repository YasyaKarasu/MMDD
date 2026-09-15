"""Small, explicit scoring helpers for the R26 verifier diagnostic."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def column_text(name: str, values: list[str]) -> str:
    return json.dumps({"column_name": name, "visible_cells": values[:5]}, ensure_ascii=False)


class ColumnProjection(nn.Module):
    def __init__(self, input_dim: int = 4096, output_dim: int = 256) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(inputs.float()), dim=-1)


def multiple_positive_loss(scores: torch.Tensor, positives: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Known repeated positives share numerator; excluded negatives leave denominator."""
    return (torch.logsumexp(scores.masked_fill(~allowed, -torch.inf), dim=1)
            - torch.logsumexp(scores.masked_fill(~positives, -torch.inf), dim=1)).mean()


def rank_of_positive(scores: list[float], positive: list[bool]) -> float:
    """Expected rank within a tie, avoiding gold-column/index ordering advantage."""
    best = max(s for s, p in zip(scores, positive) if p)
    greater = sum(s > best + 1e-7 for s in scores)
    tied = sum(abs(s - best) <= 1e-7 for s in scores)
    tied_positive = sum(p and abs(s - best) <= 1e-7 for s, p in zip(scores, positive))
    return greater + (tied + 1) / (tied_positive + 1)


def column_metrics(scores: list[float], positive: list[bool]) -> dict:
    best = max(s for s, p in zip(scores, positive) if p)
    greater = sum(s > best + 1e-7 for s in scores)
    tied = sum(abs(s - best) <= 1e-7 for s in scores)
    tied_positive = sum(p and abs(s - best) <= 1e-7 for s, p in zip(scores, positive))
    # Exact expected reciprocal rank of the first positive in a random tie order.
    survival, mrr = 1., 0.
    for offset in range(tied - tied_positive + 1):
        probability = survival * tied_positive / (tied - offset)
        mrr += probability / (greater + offset + 1)
        survival *= (tied - offset - tied_positive) / (tied - offset)
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    auc = float(np.mean([float(a > b + 1e-7) + .5 * float(abs(a-b) <= 1e-7) for a in pos for b in neg])) if neg else None
    return {"top1": tied_positive / tied if greater == 0 else 0., "mrr": mrr, "auc": auc}


def recall(ranking: list[str], positives: list[str], k: int) -> float:
    return len(set(ranking[:k]) & set(positives)) / len(set(positives))


def spearman(left: list[float], right: list[float]) -> float | None:
    def ranks(values: list[float]) -> np.ndarray:
        array = np.asarray(values)
        unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        return (ends - (counts - 1) / 2)[inverse]
    if len(set(left)) <= 1 or len(set(right)) <= 1:
        return None
    return float(np.corrcoef(ranks(left), ranks(right))[0,1])

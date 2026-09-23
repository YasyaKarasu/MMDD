"""Loss primitives used by every v3 stage (SPEC 7, 8.2, 9.3, 11.4).

* ``rank_ce``: multi-positive listwise CE ``LSE_A s - LSE_P s`` over the allowed set.
* ``kd_loss``: ``KL(softmax(t/τ) || softmax(s/τ)) τ²`` over the allowed set (τ = 1 in v3).
* ``sup_transform``: the C1 SUP score space ``10 * sigmoid(raw)``.
* ``student_anchor``: ``Σ_m ||P_m - P0||²/(4096·1024) + Σ_r ||R_r - I||²/1024²``.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor


def rank_ce(logits: Tensor, positive: Tensor, allowed: Tensor | None = None, *,
            check_finite: bool = False) -> Tensor | None:
    if logits.ndim != 1 or positive.shape != logits.shape or positive.dtype != torch.bool:
        raise ValueError("rank_ce expects aligned 1-D logits and a boolean positive mask")
    a = torch.ones_like(positive) if allowed is None else allowed
    if a.shape != positive.shape or a.dtype != torch.bool:
        raise ValueError("allowed mask must align with logits")
    if bool((positive & ~a).any()):
        raise ValueError("positive must be inside the allowed set")
    if not bool(positive.any()) or not bool((a & ~positive).any()):
        return None
    if check_finite and not bool(torch.isfinite(logits[a]).all()):
        raise ValueError("non-finite allowed logit")
    return torch.logsumexp(logits[a], 0) - torch.logsumexp(logits[positive], 0)


def kd_loss(student: Tensor, teacher: Tensor, allowed: Tensor, temperature: float = 1.0) -> Tensor:
    if temperature <= 0 or student.ndim != 1 or teacher.shape != student.shape or allowed.shape != student.shape:
        raise ValueError("KD arrays/order must align and temperature must be positive")
    if not bool(allowed.any()):
        raise ValueError("empty KD comparison set")
    t = teacher.detach()[allowed] / temperature
    s = student[allowed] / temperature
    log_t = torch.log_softmax(t, dim=0)
    log_s = torch.log_softmax(s, dim=0)
    return temperature ** 2 * torch.sum(log_t.exp() * (log_t - log_s))


def sup_transform(raw: Tensor) -> Tensor:
    return 10.0 * torch.sigmoid(raw)


def student_anchor(projections: Mapping[str, Tensor], relations: Mapping[str, Tensor], basis: Tensor) -> Tensor:
    terms = []
    for p in projections.values():
        if p.shape != basis.shape:
            raise ValueError("P/P0 shape mismatch")
        terms.append((p - basis).square().mean())
    for r in relations.values():
        if r.ndim != 2 or r.shape[0] != r.shape[1]:
            raise ValueError("square R required")
        terms.append((r - torch.eye(r.shape[0], device=r.device, dtype=r.dtype)).square().mean())
    if not terms:
        raise ValueError("no trainable task parameters")
    return torch.stack(terms).sum()


def lse_paths(zero_hop: Tensor, path_scores: Tensor, counts: list[int]) -> Tensor:
    """Per target LSE of the zero hop and its path scores (counts[i] slots each)."""
    out, cursor = [], 0
    for i, n in enumerate(counts):
        if n:
            out.append(torch.logsumexp(torch.cat([zero_hop[i].reshape(1), path_scores[cursor : cursor + n]]), 0))
            cursor += n
        else:
            out.append(zero_hop[i])
    if cursor != path_scores.numel():
        raise ValueError("path slot accounting mismatch")
    return torch.stack(out)

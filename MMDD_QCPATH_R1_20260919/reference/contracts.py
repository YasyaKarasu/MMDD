"""QCPATH-R1 mathematical contracts, not the MMDD training pipeline.

All tensors in these small helpers are synthetic-test-friendly. Production
integration must retain the same equations, label scopes and reductions.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ContractError(ValueError):
    """Invalid input or a violation of the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def namespace_seed(namespace: str) -> int:
    """One hash per sampling namespace, never a full-corpus hash sort."""
    return int.from_bytes(hashlib.sha256(namespace.encode('utf-8')).digest()[:8], 'big')


class BoundedAdapter(nn.Module):
    """Exactly Linear(4d,256) -> GELU -> Linear(256,d), with bounded delta."""

    def __init__(self, dim: int, mode: str, rho: float = 0.5) -> None:
        super().__init__()
        _require(dim > 0, 'dim must be positive')
        _require(mode in {'EONLY', 'QE'}, 'unknown adapter mode')
        _require(rho == 0.5, 'QCPATH-R1 fixes rho=0.5')
        self.dim, self.mode, self.rho = dim, mode, rho
        self.hidden = nn.Linear(4 * dim, 256)
        self.output = nn.Linear(256, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, uq: Tensor, ue: Tensor, base_query: Tensor) -> Tensor:
        _require(uq.shape == ue.shape == base_query.shape, 'vector shape mismatch')
        _require(uq.ndim == 2 and uq.shape[1] == self.dim, 'expected [batch,d]')
        if self.mode == 'QE':
            h = torch.cat([uq, ue, uq * ue, (uq - ue).abs()], dim=-1)
        else:
            zero = torch.zeros_like(ue)
            h = torch.cat([zero, ue, zero, zero], dim=-1)
        delta = self.output(F.gelu(self.hidden(h)))
        return bounded_query(base_query, delta, self.rho)


def bounded_query(base: Tensor, delta: Tensor, rho: float = 0.5,
                  eps: float = 1e-12) -> Tensor:
    """Clip delta's norm relative to base. Do NOT detach the clipping factor.

    Float64 is allowed in mathematical tests; production uses float32.
    No normalization is applied after addition.
    """
    _require(base.shape == delta.shape and base.ndim == 2, 'expected matching [B,d]')
    _require(rho == 0.5, 'rho cannot be tuned in this protocol')
    _require(bool(torch.isfinite(base).all() and torch.isfinite(delta).all()),
             'nonfinite vectors')
    norm_b = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
    _require(bool((norm_b > eps).all()), 'zero or near-zero base query')
    norm_d = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    factor = (rho * norm_b / norm_d.clamp_min(eps)).clamp(max=1.0)
    return base + factor * delta


def build_masks(target_ids: Sequence[str], legal: set[str],
                positives: set[str], query_positives: set[str],
                evidence_positives: set[str]) -> tuple[Tensor, Tensor, Tensor]:
    """Return P, I, N for one (q,e). No dev/test labels belong here."""
    _require(len(set(target_ids)) == len(target_ids), 'duplicate target ID')
    universe = set(target_ids)
    _require(legal <= universe, 'legal target lacks an index vector')
    _require(positives <= legal, 'positive is not legal')
    _require(positives <= query_positives, 'conditional positive missing from G[q]')
    ignored = ((query_positives | evidence_positives) - positives) & legal
    competitors = legal - positives - ignored
    p = torch.tensor([t in positives for t in target_ids], dtype=torch.bool)
    i = torch.tensor([t in ignored for t in target_ids], dtype=torch.bool)
    n = torch.tensor([t in competitors for t in target_ids], dtype=torch.bool)
    return p, i, n


def validate_loss_masks(logits: Tensor, p: Tensor, n: Tensor) -> None:
    _require(logits.ndim == 2 and logits.shape == p.shape == n.shape,
             'logits/P/N must have matching [B,N] shape')
    _require(p.dtype == torch.bool and n.dtype == torch.bool, 'masks must be boolean')
    _require(not bool((p & n).any()), 'positive/competitor overlap')
    _require(bool(p.any(dim=1).all()), 'inactive empty P must be handled before loss')
    _require(bool(n.any(dim=1).all()), 'inactive empty N must be handled before loss')
    _require(bool(torch.isfinite(logits).all()), 'nonfinite raw logits')


def sum_probability_losses(logits: Tensor, p: Tensor, n: Tensor) -> Tensor:
    """Per-item LSE(P union N)-LSE(P); ignore entries have zero gradient."""
    validate_loss_masks(logits, p, n)
    all_lse = torch.logsumexp(logits.masked_fill(~(p | n), -torch.inf), dim=1)
    pos_lse = torch.logsumexp(logits.masked_fill(~p, -torch.inf), dim=1)
    return all_lse - pos_lse


def full_target_losses(queries: Tensor, targets: Tensor, p: Tensor, n: Tensor,
                       chunk_size: int | None = None) -> Tensor:
    """Exact full-target denominator, with optional numerically stable chunking.

    The chunked path first computes detached global maxima, then differentiable
    sums over ALL blocks. Detaching these maxima is correct for LSE gradients;
    detaching logits or clipping factors would not be correct. Separate positive
    maxima avoid underflow when all positives are far below the top competitor.
    Production should prefer the dense matmul when it fits.
    """
    _require(queries.ndim == targets.ndim == 2, 'expected matrices')
    _require(queries.shape[1] == targets.shape[1], 'embedding dimension mismatch')
    b, nt = queries.shape[0], targets.shape[0]
    _require(p.shape == n.shape == (b, nt), 'mask dimension mismatch')
    _require(p.dtype == n.dtype == torch.bool, 'masks must be boolean')
    _require(not bool((p & n).any()), 'overlapping masks')
    _require(bool(p.any(1).all()) and bool(n.any(1).all()), 'inactive loss row')
    _require(bool(torch.isfinite(queries).all() and torch.isfinite(targets).all()),
             'nonfinite vectors')
    if chunk_size is None:
        return sum_probability_losses(queries @ targets.T, p, n)
    _require(chunk_size > 0, 'chunk_size must be positive')
    allowed = p | n
    max_a = queries.new_full((b,), -torch.inf)
    max_p = queries.new_full((b,), -torch.inf)
    with torch.no_grad():
        for start in range(0, nt, chunk_size):
            stop = min(start + chunk_size, nt)
            scores = queries @ targets[start:stop].T
            max_a = torch.maximum(max_a, scores.masked_fill(~allowed[:, start:stop],
                                  -torch.inf).max(1).values)
            max_p = torch.maximum(max_p, scores.masked_fill(~p[:, start:stop],
                                  -torch.inf).max(1).values)
    total_a = queries.new_zeros(b)
    total_p = queries.new_zeros(b)
    for start in range(0, nt, chunk_size):
        stop = min(start + chunk_size, nt)
        scores = queries @ targets[start:stop].T
        a_block = scores.masked_fill(~allowed[:, start:stop], -torch.inf)
        p_block = scores.masked_fill(~p[:, start:stop], -torch.inf)
        total_a = total_a + torch.exp(a_block - max_a[:, None]).sum(1)
        total_p = total_p + torch.exp(p_block - max_p[:, None]).sum(1)
    return (max_a + total_a.log()) - (max_p + total_p.log())


def weighted_microbatch_loss(per_item: Tensor, weights: Tensor,
                             logical_batch_size: int) -> Tensor:
    """Call once per physical microbatch; backward and sum over the logical batch."""
    _require(per_item.shape == weights.shape and per_item.ndim == 1,
             'weights/per_item shape mismatch')
    _require(logical_batch_size > 0, 'logical batch must not be empty')
    return (weights * per_item).sum() / logical_batch_size


def query_macro_weights(query_ids: Sequence[str]) -> Tensor:
    _require(len(query_ids) > 0, 'empty training population')
    counts: dict[str, int] = {}
    for q in query_ids:
        counts[q] = counts.get(q, 0) + 1
    n, nq = len(query_ids), len(counts)
    return torch.tensor([n / (nq * counts[q]) for q in query_ids], dtype=torch.float64)


def round_robin_order(pairs: Sequence[tuple[str, str]], seed: int,
                      epoch: int) -> list[int]:
    """Every unique (q,e) once. Return indices into caller's canonical pair table."""
    _require(len(set(pairs)) == len(pairs), 'duplicate (q,e)')
    rng = random.Random(namespace_seed(f'QCPATH-R1|A-order|{seed}|{epoch}'))
    groups: dict[str, list[int]] = {}
    for idx, (_q, e) in enumerate(pairs):
        groups.setdefault(e, []).append(idx)
    eids = sorted(groups)
    rng.shuffle(eids)
    for e in eids:
        groups[e].sort(key=lambda idx: pairs[idx][0])
        rng.shuffle(groups[e])
    result: list[int] = []
    for position in range(max((len(v) for v in groups.values()), default=0)):
        for e in eids:
            if position < len(groups[e]):
                result.append(groups[e][position])
    return result


def path_lse(zero_hop: Tensor, evidence_scores: Tensor,
             multiplicities: Sequence[int]) -> Tensor:
    """One target. The unique zero-hop path is always present, exactly once."""
    _require(zero_hop.numel() == 1 and evidence_scores.ndim == 1, 'invalid path shapes')
    _require(len(multiplicities) == evidence_scores.numel(), 'multiplicity mismatch')
    _require(all(type(m) is int and m > 0 for m in multiplicities), 'm must be positive int')
    _require(bool(torch.isfinite(zero_hop).all() and torch.isfinite(evidence_scores).all()),
             'nonfinite path score')
    if not multiplicities:
        return zero_hop.reshape(())
    ms = torch.tensor(multiplicities, dtype=evidence_scores.dtype,
                      device=evidence_scores.device)
    return torch.logsumexp(torch.cat([zero_hop.reshape(1), evidence_scores + ms.log()]), 0)


def support_contrast(zero_hop: Tensor, witnessed: Tensor) -> Tensor:
    """For QT control, pass the SAME forward scalar twice, not two dropout draws."""
    return F.softplus(1.0 + zero_hop - witnessed)


def strict_cohort(gold: set[str], evidence_base: set[str],
                  direct_ann100: set[str], direct_exact100: set[str]) -> set[str]:
    return gold & evidence_base - (direct_ann100 | direct_exact100)


def recall_at_k(gold: set[str], ranking: Sequence[str], k: int) -> float:
    _require(len(gold) > 0, 'empty gold set')
    _require(k > 0 and len(set(ranking)) == len(ranking), 'invalid K or duplicate ranking')
    return len(gold & set(ranking[:k])) / len(gold)


def stable_rank(ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    _require(len(ids) == len(scores) and len(set(ids)) == len(ids), 'invalid ranking inputs')
    _require(all(torch.isfinite(torch.tensor(v)).item() for v in scores), 'nonfinite score')
    return [t for t, _ in sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))]


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: Mapping[str, bool]


def _validate_metrics(values: Mapping[str, float], keys: Sequence[str]) -> None:
    for key in keys:
        _require(key in values, f'missing gate metric: {key}')
        value = values[key]
        _require(isinstance(value, (float, int)) and math.isfinite(value)
                 and 0.0 <= value <= 1.0, f'invalid gate metric: {key}')


def _ge(left: float, right: float) -> bool:
    return left + 1e-12 >= right


def gate_a(base: Mapping[str, float], eonly: Mapping[str, float],
           qe: Mapping[str, float], complete: bool = True) -> GateResult:
    _validate_metrics(base, ['et_r10', 'u_implicit', 'u_overall', 'u_explicit', 'c100'])
    _validate_metrics(eonly, ['et_r10'])
    _validate_metrics(qe, ['et_r10', 'u_implicit', 'u_overall', 'u_explicit', 'c100'])
    checks = {
        'complete': complete,
        'et_vs_base': _ge(qe['et_r10'] - base['et_r10'], 0.010),
        'et_vs_eonly': _ge(qe['et_r10'] - eonly['et_r10'], 0.005),
        'u_implicit_gain': _ge(qe['u_implicit'] - base['u_implicit'], 0.005),
        'u_overall_guard': _ge(qe['u_overall'] - base['u_overall'], -0.0025),
        'u_explicit_guard': _ge(qe['u_explicit'] - base['u_explicit'], -0.005),
        'c100_guard': _ge(qe['c100'] - base['c100'], -0.005),
    }
    return GateResult(all(checks.values()), checks)


def gate_b(t0: Mapping[str, float], qt_cont: Mapping[str, float],
           path: Mapping[str, float], swap_overall: float,
           complete: bool = True) -> GateResult:
    for metrics in (t0, qt_cont, path):
        _validate_metrics(metrics, ['overall', 'implicit', 'strict'])
    _validate_metrics({'swap': swap_overall}, ['swap'])
    checks = {
        'complete': complete,
        'overall_near_qt': _ge(path['overall'], max(t0['overall'], qt_cont['overall']) - 0.005),
        'implicit_guard': _ge(path['implicit'], max(t0['implicit'], qt_cont['implicit'])),
        'strict_gain': _ge(path['strict'], max(t0['strict'], qt_cont['strict']) + 0.010),
        'e_content_gain': _ge(path['overall'] - swap_overall, 0.005),
    }
    return GateResult(all(checks.values()), checks)

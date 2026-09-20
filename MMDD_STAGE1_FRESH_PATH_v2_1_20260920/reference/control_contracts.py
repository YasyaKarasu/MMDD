"""CPU reference for v2.1 controls and scheduling contracts.

This file does not launch GPU jobs or implement the complete experiment.
All GPU capacity/throughput numbers used in the tests are synthetic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math
import torch
from torch import Tensor, nn
from contracts import rank_loss, target_path_lse


class QTOnlyStudent(nn.Module):
    """Only table projection and QT relation; no evidence branch/adapter."""
    def __init__(self, basis: Tensor):
        super().__init__()
        if basis.ndim != 2 or not torch.isfinite(basis).all():
            raise ValueError('finite two-dimensional PCA basis required')
        self.P_table = nn.Parameter(basis.detach().clone())
        self.R_QT = nn.Parameter(torch.eye(len(basis), dtype=basis.dtype, device=basis.device))

    def keys(self, z_target: Tensor) -> Tensor:
        return z_target @ self.P_table.T

    def query(self, z_query: Tensor) -> Tensor:
        return (z_query @ self.P_table.T) @ self.R_QT

    def forward(self, z_query: Tensor, z_target: Tensor) -> Tensor:
        return self.query(z_query) @ self.keys(z_target).T


def full_qt_loss(model: QTOnlyStudent, z_query: Tensor, z_target: Tensor,
                 positive: Tensor, allowed: Tensor, chunk: int) -> Tensor | None:
    """Trainable target-side P; NOT the frozen-key C2 assumption.

Reference autograd retains chunk graphs. Production may checkpoint/recompute
but must preserve gradients through both the query and target projections.
"""
    if z_query.ndim != 1 or z_target.ndim != 2 or chunk <= 0:
        raise ValueError('one query, a target matrix and positive chunk required')
    if positive.shape != z_target.shape[:1] or allowed.shape != positive.shape:
        raise ValueError('mask shape mismatch')
    if positive.dtype != torch.bool or allowed.dtype != torch.bool:
        raise ValueError('Boolean masks required')
    if torch.any(positive & ~allowed):
        raise ValueError('positive excluded')
    if not positive.any().item() or not (allowed & ~positive).any().item():
        return None
    q = model.query(z_query)
    alls, poss = [], []
    for start in range(0, len(z_target), chunk):
        stop = min(start + chunk, len(z_target))
        scores = q @ model.keys(z_target[start:stop]).T
        a, p = allowed[start:stop], positive[start:stop]
        if not torch.isfinite(scores[a]).all():
            raise ValueError('non-finite scores')
        if a.any().item():
            alls.append(torch.logsumexp(scores[a], 0))
        if p.any().item():
            poss.append(torch.logsumexp(scores[p], 0))
    return torch.logsumexp(torch.stack(alls), 0)-torch.logsumexp(torch.stack(poss), 0)


def native_et_score(z_e: Tensor, z_target: Tensor, p_e: Tensor,
                    p_t: Tensor, r_et: Tensor) -> Tensor:
    """No q argument: old-style directed bilinear ET, without an adapter."""
    return ((z_e @ p_e.T) @ r_et) @ (z_target @ p_t.T).T


def strict_direct_pool(direct: Sequence[str], *, evidence_rankings: Any = None,
                       budget: int = 100) -> list[str]:
    if evidence_rankings is not None:
        raise ValueError('strict Direct-only cannot receive evidence rankings')
    if len(direct) != len(set(direct)) or budget < 0:
        raise ValueError('unique Direct ranking and valid budget required')
    return list(direct[:budget])


def validate_qt_label_payload(payload: Mapping[str, Any]) -> None:
    allowed = {'query_id', 'positive_target_ids', 'legal_target_ids', 'source_group'}
    if set(payload)-allowed:
        raise ValueError('QT-only payload includes unapproved annotation fields')
    if not {'query_id', 'positive_target_ids', 'legal_target_ids'} <= set(payload):
        raise ValueError('missing required QT labels')
    if not set(payload['positive_target_ids']) <= set(payload['legal_target_ids']):
        raise ValueError('positive target outside legal universe')


def score_cache_key(run: str, branch: str, checkpoint: str, q: str,
                    evidence: str | None, candidate_ids: Sequence[str],
                    mask_fingerprint: str, view: str) -> tuple[Any, ...]:
    if not run or branch not in {'T_EDGE', 'T_QT', 'T_PATH'} or not checkpoint:
        raise ValueError('explicit Teacher identity required')
    return (run, branch, checkpoint, q, evidence, tuple(candidate_ids), mask_fingerprint, view)


def validate_dag(stages: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate only task edges; data readiness is checked by ready_stages."""
    ids = [s['stage'] for s in stages]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate stage')
    by_id = {s['stage']:s for s in stages}
    visited, visiting, ordered = set(), set(), []
    def visit(x: str):
        if x in visiting:
            raise ValueError('cyclic stage dependency')
        if x in visited:
            return
        if x not in by_id:
            raise ValueError('external or missing task stage')
        visiting.add(x)
        s = by_id[x]
        for parent in s['weight_parents'] + s['read_only_model_dependencies']:
            visit(parent)
        visiting.remove(x); visited.add(x); ordered.append(x)
    for x in ids:
        visit(x)
    return ordered


def ready_stages(stages: Sequence[Mapping[str, Any]], committed: Iterable[str],
                 sealed_data: Iterable[str], running: Iterable[str] = ()) -> list[str]:
    validate_dag(stages)
    done, data, active = set(committed), set(sealed_data), set(running)
    return [s['stage'] for s in sorted(stages, key=lambda s:s['priority'])
            if s['stage'] not in done|active
            and set(s['weight_parents'] + s['read_only_model_dependencies']) <= done
            and set(s['data_dependencies']) <= data]


@dataclass(frozen=True)
class MemoryProfile:
    framework_reserved_gib: float
    observed_process_gib: float | None = None
    def reserve(self) -> float:
        values=[self.framework_reserved_gib]
        if self.observed_process_gib is not None:
            values.append(self.observed_process_gib)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid memory observation')
        return max(values)+1.0


def can_colocate(profiles: Sequence[MemoryProfile], *, device_total_gib: float,
                 external_used_gib: float = 0, parity_passed: bool,
                 serial_fixed_work_seconds: float, concurrent_fixed_work_seconds: float) -> bool:
    numbers=(device_total_gib, external_used_gib, serial_fixed_work_seconds, concurrent_fixed_work_seconds)
    if any(not math.isfinite(x) for x in numbers) or device_total_gib<=0 or external_used_gib<0:
        raise ValueError('invalid device/timing observations')
    if serial_fixed_work_seconds<=0 or concurrent_fixed_work_seconds<=0:
        raise ValueError('positive fixed-work timings required')
    if not profiles or len(profiles)>2 or not parity_passed:
        return False
    reserved=external_used_gib+sum(p.reserve() for p in profiles)
    capacity_ok=reserved<=0.85*device_total_gib and device_total_gib-reserved>=3.0
    return capacity_ok and serial_fixed_work_seconds/concurrent_fixed_work_seconds >= 1.05

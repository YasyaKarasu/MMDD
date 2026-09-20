"""Stage losses and reductions for FRESH-PATH v2.1 (SPEC 7, 9, 10).

The rank/full-denominator/KD primitives live in :mod:`fresh_path.contracts`
(ported verbatim from the package reference so the CPU reference tests apply).
This module only composes them into the stage objectives.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .contracts import kd_loss, rank_loss, streamed_full_loss, target_path_lse

__all__ = [
    "kd_loss",
    "rank_loss",
    "streamed_full_loss",
    "target_path_lse",
    "mean_active",
    "relation_macro",
    "support_margin",
    "aggregate_paths",
    "teacher_edge_loss",
    "teacher_path_loss",
    "tqt_loss",
    "native_loss",
    "c2_loss",
    "logical_batch_scale",
]


def mean_active(values: Iterable[Tensor | None]) -> Tensor | None:
    """Mean over the non-inactive entries; None when every entry is inactive."""
    live = [v for v in values if v is not None]
    if not live:
        return None
    return torch.stack(live).mean()


def relation_macro(per_relation: Mapping[str, Tensor | None]) -> Tensor | None:
    """SPEC 3.4: average inside a relation first, then equal-weight relations."""
    return mean_active(per_relation.values())


def support_margin(zero_hop: Tensor, with_evidence: Tensor, margin: float = 1.0) -> Tensor:
    """L_support = softplus(margin + f(q,empty,t*) - f(q,e*,t*)) (SPEC 7.3)."""
    return F.softplus(margin + zero_hop - with_evidence)


def aggregate_paths(zero_hop: Tensor, path_scores: Sequence[Tensor]) -> Tensor:
    """S = LSE(f0, {path scores}); one scalar per target."""
    if not path_scores:
        return zero_hop
    return torch.logsumexp(torch.stack([zero_hop, *path_scores]), 0)


def teacher_edge_loss(per_relation: Mapping[str, Tensor | None]) -> Tensor | None:
    return relation_macro(per_relation)


def teacher_path_loss(
    target_loss: Tensor | None,
    edge_new: Tensor | None,
    conditional: Tensor | None,
    support: Tensor | None,
    *,
    edge_weight: float = 0.5,
    conditional_weight: float = 0.5,
    support_weight: float = 0.2,
) -> Tensor | None:
    """L_T = mean_views(L_target) + w_e L_edge + w_c mean_e L_cond + w_s L_support."""
    terms: list[Tensor] = []
    if target_loss is not None:
        terms.append(target_loss)
    if edge_new is not None and edge_weight:
        terms.append(edge_weight * edge_new)
    if conditional is not None and conditional_weight:
        terms.append(conditional_weight * conditional)
    if support is not None and support_weight:
        terms.append(support_weight * support)
    if not terms:
        return None
    return torch.stack(terms).sum()


def tqt_loss(target_loss: Tensor | None, edge_new: Tensor | None, *, edge_weight: float = 0.5) -> Tensor | None:
    """L_TQT = L_rank(f0; G, C\\G) + 0.5 L_edge,new (SPEC 7.5). No QET/support/path terms."""
    terms: list[Tensor] = []
    if target_loss is not None:
        terms.append(target_loss)
    if edge_new is not None and edge_weight:
        terms.append(edge_weight * edge_new)
    if not terms:
        return None
    return torch.stack(terms).sum()


def native_loss(
    direct_sup: Tensor | None,
    direct_kd: Tensor | None,
    path_sup: Tensor | None,
    path_kd: Tensor | None,
    *,
    direct_weight: float = 0.5,
    path_weight: float = 0.5,
) -> Tensor | None:
    """SPEC 9.6: 1/2 [rank + KD] on the direct QT list plus 1/2 mean_views[rank + KD] on paths."""
    direct = None
    if direct_sup is not None or direct_kd is not None:
        direct = torch.stack([t for t in (direct_sup, direct_kd) if t is not None]).sum()
    path = None
    if path_sup is not None or path_kd is not None:
        path = torch.stack([t for t in (path_sup, path_kd) if t is not None]).sum()
    terms = []
    if direct is not None:
        terms.append(direct_weight * direct)
    if path is not None:
        terms.append(path_weight * path)
    if not terms:
        return None
    return torch.stack(terms).sum()


def c2_loss(
    full: Tensor | None,
    cond_kd: Tensor | None,
    target_sup: Tensor | None,
    target_kd: Tensor | None,
    *,
    kd_weight: float = 1.0,
    target_weight: float = 0.5,
    target_kd_weight: float = 0.5,
) -> Tensor | None:
    """SPEC 9.3 L_C2.  All inputs are already reduced over e / views."""
    terms: list[Tensor] = []
    if full is not None:
        terms.append(full)
    if cond_kd is not None and kd_weight:
        terms.append(kd_weight * cond_kd)
    if target_sup is not None and target_weight:
        terms.append(target_weight * target_sup)
    if target_kd is not None and target_kd_weight:
        terms.append(target_kd_weight * kd_weight * target_kd)
    if not terms:
        return None
    return torch.stack(terms).sum()


def logical_batch_scale(active_queries: int) -> float:
    """SPEC 10.2: accumulate per-query loss / true active count of the logical batch."""
    if active_queries <= 0:
        raise ValueError("logical batch must contain at least one active query")
    return 1.0 / active_queries

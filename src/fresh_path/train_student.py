"""Student stages (SPEC 8 and 9).

Main line: S_SUP_C1 / S_KD_C1 train P/R; the conditional C2 stages train only
the adapter.  KD-NATIVE trains all P/R without an adapter, and the QT-only
Students train only P_table / R_QT with no evidence branch at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from . import config
from .contracts import kd_loss, rank_loss, streamed_full_loss
from .inputs import TrainLabels
from .losses import mean_active
from .score import ObjectBank
from .teacher_cache import FrozenTeacherLogitCache
from .train_teacher import PATH_BLOCK, rank_active


# ------------------------------------------------------------ vector helpers ---


def project(model, bank: ObjectBank, kind: str, object_ids: Sequence[str], device) -> torch.Tensor:
    z = bank.z_many(list(object_ids)).to(device)
    return model.u(kind, z)


def target_keys(model, bank: ObjectBank, object_ids: Sequence[str], device) -> torch.Tensor:
    return project(model, bank, "table", object_ids, device)


def relation_matrix(model, name: str) -> torch.Tensor:
    return model.relations[name]


def qt_logits(model, bank: ObjectBank, query_id: str, targets: Sequence[str], device) -> torch.Tensor:
    uq = model.u("table", bank.z(query_id).to(device))
    keys = target_keys(model, bank, targets, device)
    return (uq @ model.relations["QT"]) @ keys.T


def first_hop_logits(model, bank: ObjectBank, query_id: str, modality: str, evidences: Sequence[str], device) -> torch.Tensor:
    uq = model.u("table", bank.z(query_id).to(device))
    ue = project(model, bank, modality, evidences, device)
    return (uq @ model.relations[f"Q_to_{modality}"]) @ ue.T


def et_logits(model, bank: ObjectBank, evidence_id: str, targets: Sequence[str], device) -> torch.Tensor:
    kind = bank.kind(evidence_id)
    ue = model.u(kind, bank.z(evidence_id).to(device))
    keys = target_keys(model, bank, targets, device)
    return (ue @ model.relations[f"{kind}_to_T"]) @ keys.T


def masks(ids: Sequence[str], positives: set[str], ignore: set[str], device):
    p = torch.tensor([i in positives for i in ids], dtype=torch.bool, device=device)
    allowed = torch.tensor([i not in ignore for i in ids], dtype=torch.bool, device=device)
    return p, allowed


# --------------------------------------------------------------- C1 stages ---


def edge_losses(model, teacher, bank: ObjectBank, labels: TrainLabels, query_id: str, groups: dict,
                edge_positives: dict, device, *, kd: bool,
                batch: int = 128, teacher_cache: FrozenTeacherLogitCache | None = None,
                encode_cache: dict | None = None) -> dict[str, torch.Tensor | None]:
    by_relation: dict[str, list[torch.Tensor | None]] = {}
    for key, candidates in groups.items():
        if not candidates:
            continue
        positives = set(edge_positives[query_id][key])
        p, allowed = masks(candidates, positives, set(), device)
        if key == "QT":
            relation = "QT"
            logits = qt_logits(model, bank, query_id, candidates, device)
        elif key.startswith("Q_"):
            relation = key
            logits = first_hop_logits(model, bank, query_id, key[2:], candidates, device)
        else:
            asset = key[2:]
            relation = f"E_{bank.kind(asset)}"
            logits = et_logits(model, bank, asset, candidates, device)
        loss = rank_loss(logits, p, allowed, active=rank_active(candidates, positives))
        if kd:
            from .train_teacher import pair_logits

            anchor = query_id if not key.startswith("E_") else key[2:]
            t_logits = pair_logits(
                teacher, bank, anchor, candidates, device, batch=batch,
                teacher_cache=teacher_cache, encode_cache=encode_cache,
                view=f"edge:{key}",
            ).detach()
            kd_term = kd_loss(logits, t_logits, allowed, temperature=2.0)
            loss = kd_term if loss is None else loss + kd_term
        by_relation.setdefault(relation, []).append(loss)
    return {relation: mean_active(values) for relation, values in by_relation.items()}


def train_c1(model, teacher, bank: ObjectBank, labels: TrainLabels, edge: dict, edge_positives: dict, *,
             seed: int, epochs: int, lr: float, logical_batch: int, device: str, out_dir: Path,
             protocol_path: Path, kd: bool, optimizer=None, start_epoch: int = 1,
             counters: dict | None = None,
             batch: int = 128, teacher_cache: FrozenTeacherLogitCache | None = None,
             log=print) -> dict:
    from .train_teacher import _accumulate, _epoch_data_state, _save, _save_initial, query_order

    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    stage = "S_KD_C1" if kd else "S_SUP_C1"
    model.train()
    counters = dict(counters or {"updates": 0, "active_queries": 0, "lists": 0})
    _save_initial(model, optimizer, out_dir, seed, stage, protocol_path)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch, namespace="order")
        buffer: list[torch.Tensor] = []
        pending = 0
        encode_cache: dict = {}
        for qid in order:
            losses = edge_losses(model, teacher, bank, labels, qid, edge[qid], edge_positives, device, kd=kd,
                                 batch=batch, teacher_cache=teacher_cache,
                                 encode_cache=encode_cache)
            total = mean_active(losses.values())
            if total is None:
                continue
            buffer.append(total)
            pending += 1
            counters["lists"] += sum(1 for v in losses.values() if v is not None)
            if pending == logical_batch:
                _accumulate(buffer, pending, optimizer)
                counters["updates"] += 1
                counters["active_queries"] += pending
                buffer, pending, encode_cache = [], 0, {}
        if pending:
            _accumulate(buffer, pending, optimizer)
            counters["updates"] += 1
            counters["active_queries"] += pending
        log(json.dumps({"event": "epoch", "stage": stage, "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, stage, epoch, protocol_path, counters,
              data_state=_epoch_data_state(order))
    if teacher_cache is not None:
        teacher_cache.save()
    return counters


# ------------------------------------------------------- conditional C2 ------


def adapter_vector(model, bank: ObjectBank, query_id: str, asset: str, device, *, read_q: bool = True) -> torch.Tensor:
    kind = bank.kind(asset)
    uq = model.u("table", bank.z(query_id).to(device))
    ue = model.u(kind, bank.z(asset).to(device))
    base = ue @ model.relations[f"{kind}_to_T"]
    return model.adapter(uq, ue, base, read_q=read_q)


def _target_index(ids: Sequence[str]) -> dict[str, int]:
    return {t: i for i, t in enumerate(ids)}


@dataclass(frozen=True)
class C2ConditionIndex:
    """Model-independent registry for one ``(q,e)`` C2 condition."""

    asset: str
    candidates: tuple[str, ...]
    positives: frozenset[str]
    ignore: frozenset[str]
    candidate_indices: tuple[int, ...]
    positive_mask: tuple[bool, ...]
    allowed_mask: tuple[bool, ...]
    full_positive_mask: tuple[bool, ...]
    full_allowed_mask: tuple[bool, ...]
    active: bool
    full_active: bool
    full_chunk_activity: tuple[tuple[bool, bool], ...]
    natural_paths: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def paths(self) -> dict[str, tuple[str, ...]]:
        return dict(self.natural_paths)


@dataclass(frozen=True)
class C2QueryIndex:
    graph: tuple[str, ...]
    graph_positive: frozenset[str]
    target_active: bool
    natural_paths: tuple[tuple[str, tuple[str, ...]], ...]
    conditions: tuple[C2ConditionIndex, ...]

    @property
    def paths(self) -> dict[str, tuple[str, ...]]:
        return dict(self.natural_paths)


def _chunk_flags(allowed: Sequence[bool], positive: Sequence[bool], chunk: int) -> tuple[tuple[bool, bool], ...]:
    return tuple(
        (any(allowed[start:start + chunk]), any(positive[start:start + chunk]))
        for start in range(0, len(allowed), chunk)
    )


def build_c2_static_index(labels: TrainLabels, raw: dict, graph: dict, *, full_chunk: int = 4096,
                          conditional_registry=None) -> dict[str, C2QueryIndex]:
    """Precompute all label/raw-derived C2 lists, masks and path maps.

    Nothing in this registry depends on model parameters.  It is therefore
    safe to reuse for all C2 epochs and all paired SUP/KD arms; only device
    tensors and frozen target keys are materialized by the training call.
    """
    from .train_teacher import conditional_list, paths_for_targets

    legal = tuple(raw["legal"])
    legal_index = _target_index(legal)
    out: dict[str, C2QueryIndex] = {}
    for query_id in sorted(graph, key=lambda x: x.encode("utf-8")):
        targets = tuple(graph[query_id])
        entry = labels.queries[query_id]
        g = frozenset(entry["G"])
        target_active = bool(g & set(targets)) and bool(set(targets) - g)
        natural = paths_for_targets(raw, query_id, targets, None)
        conditions: list[C2ConditionIndex] = []
        assets = sorted({a for ev in entry["W"].values() for a in ev}, key=lambda x: x.encode("utf-8"))
        for asset in assets:
            candidates, p_ids, ignore = conditional_list(
                labels, raw, query_id, asset, registry=conditional_registry,
            )
            if not candidates or not p_ids:
                continue
            candidate_ids = tuple(candidates)
            pset, iset = frozenset(p_ids), frozenset(ignore)
            p_mask = tuple(t in pset for t in candidate_ids)
            allowed_mask = tuple(t not in iset for t in candidate_ids)
            full_positive = tuple(t in pset for t in legal)
            full_allowed = tuple(t not in iset for t in legal)
            conditions.append(C2ConditionIndex(
                asset=asset,
                candidates=candidate_ids,
                positives=pset,
                ignore=iset,
                candidate_indices=tuple(legal_index[t] for t in candidate_ids),
                positive_mask=p_mask,
                allowed_mask=allowed_mask,
                full_positive_mask=full_positive,
                full_allowed_mask=full_allowed,
                active=any(p_mask) and any(a and not p for a, p in zip(allowed_mask, p_mask)),
                full_active=any(full_positive) and any(a and not p for a, p in zip(full_allowed, full_positive)),
                full_chunk_activity=_chunk_flags(full_allowed, full_positive, full_chunk),
                natural_paths=tuple((t, tuple(natural[t])) for t in targets),
            ))
        out[query_id] = C2QueryIndex(
            graph=targets,
            graph_positive=g,
            target_active=target_active,
            natural_paths=tuple((t, tuple(natural[t])) for t in targets),
            conditions=tuple(conditions),
        )
    return out


def conditional_c2_query(model, teacher, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str,
                         graph: Sequence[str], keys_full: torch.Tensor, legal_index: dict[str, int],
                         device, *, kd: bool, read_q: bool = True, full_chunk: int = 4096,
                         batch: int = 32, anchor=None,
                         teacher_cache: FrozenTeacherLogitCache | None = None,
                         encode_cache: dict | None = None,
                         static_index: C2QueryIndex | None = None,
                         conditional_registry=None) -> tuple[dict, dict]:
    """One query's conditional C2 terms (all reducible tensors)."""
    from .train_teacher import conditional_list, paths_for_targets, pair_logits, triplet_pairs

    entry = labels.queries[query_id]
    legal = tuple(raw["legal"])
    if static_index is None:
        static_index = build_c2_static_index(
            labels, raw, {query_id: list(graph)}, full_chunk=full_chunk,
            conditional_registry=conditional_registry,
        )[query_id]
    full_terms: list[torch.Tensor] = []
    kd_terms: list[torch.Tensor] = []
    for condition in static_index.conditions:
        asset = condition.asset
        candidates = condition.candidates
        if not condition.active and not condition.full_active:
            continue
        v_qe = adapter_vector(model, bank, query_id, asset, device, read_q=read_q)
        legal_mask = torch.tensor(condition.full_allowed_mask, dtype=torch.bool, device=device)
        pos_mask = torch.tensor(condition.full_positive_mask, dtype=torch.bool, device=device)
        full = streamed_full_loss(
            v_qe, keys_full, pos_mask, legal_mask, full_chunk,
            active=condition.full_active, chunk_activity=condition.full_chunk_activity,
            validate_finite=False, validate_masks=False,
        )
        if full is not None:
            full_terms.append(full)
        if kd and condition.active:
            cand_keys = keys_full[torch.tensor(condition.candidate_indices, device=device)]
            s_logits = v_qe @ cand_keys.T
            p, allowed = masks(candidates, set(condition.positives), set(condition.ignore), device)
            t_logits = triplet_pairs(
                teacher, bank, query_id, [(asset, t) for t in candidates], device,
                batch=batch, teacher_cache=teacher_cache, mask=allowed.tolist(),
                view=f"conditional:{asset}", encode_cache=encode_cache,
            )
            kd_terms.append(kd_loss(s_logits, t_logits.detach(), allowed, temperature=2.0))

    # target-path terms
    views = [None]
    if anchor is not None:
        views.append(anchor[1])
    sup_targets, kd_targets = [], []
    const_qt = qt_logits(model, bank, query_id, graph, device)
    gi = _target_index(graph)
    for augmented_e in views:
        per_target = static_index.paths
        if augmented_e is not None:
            per_target = {
                t: tuple(dict.fromkeys([*per_target[t], augmented_e])) for t in graph
            }
        flat = [(e, t) for t in graph for e in per_target[t]]
        path_scores = _conditional_path_scores(model, bank, query_id, flat, device, read_q=read_q)
        stacked = []
        cursor = 0
        for i, t in enumerate(graph):
            n = len(per_target[t])
            if n:
                stacked.append(torch.logsumexp(torch.cat([const_qt[i].reshape(1), path_scores[cursor : cursor + n]]), 0))
                cursor += n
            else:
                stacked.append(const_qt[i])
        S = torch.stack(stacked)
        p = torch.tensor([t in static_index.graph_positive for t in graph], dtype=torch.bool, device=device)
        allowed = torch.ones(len(graph), dtype=torch.bool, device=device)
        sup_targets.append(rank_loss(
            S, p, allowed, active=static_index.target_active, validate_finite=False
        ))
        if kd:
            t_scores = teacher_target_view(
                teacher, bank, raw, query_id, graph, augmented_e, device,
                batch=PATH_BLOCK, teacher_cache=teacher_cache, encode_cache=encode_cache,
                path_map=per_target,
            )
            kd_targets.append(kd_loss(S, t_scores.detach(), allowed, temperature=2.0))
    return (
        {"full": mean_active(full_terms), "cond_kd": mean_active(kd_terms) if kd else None},
        {"target_sup": mean_active(sup_targets), "target_kd": mean_active(kd_targets) if kd else None},
    )


def _grouped_path_scores(
    flat: Sequence[tuple[str, str]],
    score_group,
    *,
    device,
) -> torch.Tensor:
    """Score grouped paths and restore the caller's original slot order.

    Grouping avoids repeating the query/evidence encoding, but concatenating
    group outputs directly changes ``[(e1,t1),(e2,t1),...]`` into evidence-
    major order.  The inverse permutation below is a differentiable gather,
    so both values and gradients remain aligned with the original slots.
    """
    if not flat:
        return torch.zeros(0, device=device)
    grouped: dict[str, list[tuple[int, str]]] = {}
    for slot, (evidence_id, target_id) in enumerate(flat):
        grouped.setdefault(evidence_id, []).append((slot, target_id))
    values: list[torch.Tensor] = []
    positions: list[int] = []
    for evidence_id, items in grouped.items():
        scores = score_group(evidence_id, [target for _, target in items]).reshape(-1)
        if scores.numel() != len(items):
            raise ValueError("grouped path scorer returned the wrong number of slots")
        values.append(scores)
        positions.extend(slot for slot, _ in items)
    grouped_values = torch.cat(values)
    inverse = torch.argsort(torch.tensor(positions, dtype=torch.long, device=grouped_values.device))
    return grouped_values[inverse]


def _conditional_path_scores(
    model,
    bank: ObjectBank,
    query_id: str,
    flat: Sequence[tuple[str, str]],
    device,
    *,
    read_q: bool = True,
) -> torch.Tensor:
    if not flat:
        return torch.zeros(0, device=device)

    def score_group(evidence_id: str, targets: Sequence[str]) -> torch.Tensor:
        v_qe = adapter_vector(model, bank, query_id, evidence_id, device, read_q=read_q)
        keys = target_keys(model, bank, targets, device)
        first = first_hop_logits(model, bank, query_id, bank.kind(evidence_id), [evidence_id], device)[0]
        return (v_qe @ keys.T).reshape(-1) + first

    return _grouped_path_scores(flat, score_group, device=device)


def teacher_target_view(teacher, bank: ObjectBank, raw: dict, query_id: str, graph: Sequence[str],
                        augmented_e: str | None, device, *, batch: int = 32,
                        teacher_cache: FrozenTeacherLogitCache | None = None,
                        encode_cache: dict | None = None,
                        path_map: dict[str, Sequence[str]] | None = None) -> torch.Tensor:
    from .train_teacher import pair_logits, paths_for_targets, triplet_pairs

    f0 = pair_logits(
        teacher, bank, query_id, graph, device, batch=batch, teacher_cache=teacher_cache,
        encode_cache=encode_cache, view="target:natural" if augmented_e is None else "target:augmented",
    )
    per_target = paths_for_targets(raw, query_id, graph, augmented_e) if path_map is None else path_map
    flat = [(e, t) for t in graph for e in per_target[t]]
    scores = triplet_pairs(
        teacher, bank, query_id, flat, device, batch=batch, teacher_cache=teacher_cache,
        encode_cache=encode_cache,
        view="target_paths:natural" if augmented_e is None else f"target_paths:augmented:{augmented_e}",
    )
    stacked = []
    cursor = 0
    for i, t in enumerate(graph):
        n = len(per_target[t])
        if n:
            stacked.append(torch.logsumexp(torch.cat([f0[i].reshape(1), scores[cursor : cursor + n]]), 0))
            cursor += n
        else:
            stacked.append(f0[i])
    return torch.stack(stacked)


def train_conditional_c2(model, teacher, bank: ObjectBank, labels: TrainLabels, raw: dict, graph: dict, *,
                         seed: int, epochs: int, lr: float, logical_batch: int, device: str,
                         out_dir: Path, protocol_path: Path, kd: bool, eonly: bool,
                         parent_dirs: list[Path], full_chunk: int = 4096, batch: int = 32,
                         optimizer=None, start_epoch: int = 1,
                         counters: dict | None = None,
                         teacher_cache: FrozenTeacherLogitCache | None = None,
                         conditional_registry=None,
                         anchor_registry=None,
                         log=print) -> dict:
    from .train_teacher import _accumulate, _epoch_data_state, _save, _save_initial, build_anchor_registry, query_order

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.adapter.parameters():
        param.requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    stage = ("S_KD" if kd else "S_SUP") + ("_EONLY_C2" if eonly else "_QE_C2")
    legal = raw["legal"]
    keys_full = target_keys(model, bank, legal, device).detach()
    legal_index = _target_index(legal)
    static_index = build_c2_static_index(
        labels, raw, graph, full_chunk=full_chunk, conditional_registry=conditional_registry,
    )
    anchor_registry = anchor_registry or build_anchor_registry(labels, epochs)
    model.eval()
    counters = dict(counters or {"updates": 0, "active_queries": 0, "items": 0, "zero_trainable_path": 0})
    _save_initial(model, optimizer, out_dir, seed, stage, protocol_path, parent_dirs)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch, namespace="order")
        buffer: list[torch.Tensor] = []
        pending = 0
        encode_cache: dict = {}
        for qid in order:
            anchor = anchor_registry[epoch][qid]
            terms, target_terms = conditional_c2_query(
                model, teacher, bank, labels, raw, qid, graph[qid], keys_full, legal_index, device,
                kd=kd, read_q=not eonly, full_chunk=full_chunk, batch=batch, anchor=anchor,
                teacher_cache=teacher_cache, encode_cache=encode_cache,
                static_index=static_index[qid], conditional_registry=conditional_registry,
            )
            total = _c2_total(terms, target_terms, kd=kd)
            if total is None:
                counters["zero_trainable_path"] += 1
                continue
            buffer.append(total)
            pending += 1
            counters["items"] += len(graph[qid])
            if pending == logical_batch:
                _accumulate(buffer, pending, optimizer)
                counters["updates"] += 1
                counters["active_queries"] += pending
                buffer, pending, encode_cache = [], 0, {}
        if pending:
            _accumulate(buffer, pending, optimizer)
            counters["updates"] += 1
            counters["active_queries"] += pending
        log(json.dumps({"event": "epoch", "stage": stage, "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, stage, epoch, protocol_path, counters,
              {"_parent_dirs": [str(p) for p in parent_dirs]}, data_state=_epoch_data_state(order))
    if teacher_cache is not None:
        teacher_cache.save()
    return counters


def _c2_total(terms: dict, target_terms: dict, *, kd: bool) -> torch.Tensor | None:
    parts = []
    if terms["full"] is not None:
        parts.append(terms["full"])
    if kd and terms["cond_kd"] is not None:
        parts.append(terms["cond_kd"])
    if target_terms["target_sup"] is not None:
        parts.append(0.5 * target_terms["target_sup"])
    if kd and target_terms["target_kd"] is not None:
        parts.append(0.5 * target_terms["target_kd"])
    if not parts:
        return None
    return torch.stack(parts).sum()


# ------------------------------------------------------------ KD-NATIVE ------


def native_query_loss(model, teacher_qt, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str,
                      graph: Sequence[str], device, anchor, *, batch: int = 32,
                      teacher_cache: FrozenTeacherLogitCache | None = None,
                      encode_cache: dict | None = None) -> torch.Tensor | None:
    """SPEC 9.6: old-style unconditioned ET, teacher = T_QT pair outputs only."""
    from .train_teacher import pair_logits, paths_for_targets

    entry = labels.queries[query_id]
    positives = set(entry["G"])
    # direct QT list: rank + KD
    s_direct = qt_logits(model, bank, query_id, graph, device)
    p = torch.tensor([t in positives for t in graph], dtype=torch.bool, device=device)
    allowed = torch.ones(len(graph), dtype=torch.bool, device=device)
    active = rank_active(graph, positives)
    sup_direct = rank_loss(s_direct, p, allowed, active=active)
    t_direct = pair_logits(
        teacher_qt, bank, query_id, graph, device, batch=batch,
        teacher_cache=teacher_cache, encode_cache=encode_cache, view="native:direct",
    ).detach()
    kd_direct = kd_loss(s_direct, t_direct, allowed, temperature=2.0)

    terms = []
    if sup_direct is not None:
        terms.append(sup_direct)
    terms.append(kd_direct)
    direct_total = torch.stack(terms).sum()

    # path views: student LSE of s_N(q,t) and s_N(q,e)+s_N(e,t)
    views = [None] + ([anchor[1]] if anchor is not None else [])
    path_sup, path_kd = [], []
    for augmented_e in views:
        per_target = paths_for_targets(raw, query_id, graph, augmented_e)
        flat = [(e, t) for t in graph for e in per_target[t]]
        scores = _native_path_scores(model, bank, query_id, flat, device)
        zero = qt_logits(model, bank, query_id, graph, device)
        stacked, cursor = [], 0
        for i, t in enumerate(graph):
            n = len(per_target[t])
            if n:
                stacked.append(torch.logsumexp(torch.cat([zero[i].reshape(1), scores[cursor : cursor + n]]), 0))
                cursor += n
            else:
                stacked.append(zero[i])
        S = torch.stack(stacked)
        path_sup.append(rank_loss(S, p, allowed, active=active))
        t_scores = native_teacher_paths(
            teacher_qt, bank, raw, query_id, graph, augmented_e, device, batch=batch,
            teacher_cache=teacher_cache, encode_cache=encode_cache,
        )
        path_kd.append(kd_loss(S, t_scores.detach(), allowed, temperature=2.0))
    path_total = None
    sup_path = mean_active(path_sup)
    kd_path = mean_active(path_kd)
    parts = [x for x in (sup_path, kd_path) if x is not None]
    if parts:
        path_total = torch.stack(parts).sum()
    out = [0.5 * direct_total]
    if path_total is not None:
        out.append(0.5 * path_total)
    return torch.stack(out).sum()


def _native_path_scores(model, bank: ObjectBank, query_id: str, flat: Sequence[tuple[str, str]], device) -> torch.Tensor:
    if not flat:
        return torch.zeros(0, device=device)
    def score_group(evidence_id: str, targets: Sequence[str]) -> torch.Tensor:
        kind = bank.kind(evidence_id)
        ue = model.u(kind, bank.z(evidence_id).to(device))
        v_e = ue @ model.relations[f"{kind}_to_T"]
        keys = target_keys(model, bank, targets, device)
        first = first_hop_logits(model, bank, query_id, kind, [evidence_id], device)[0]
        return (v_e @ keys.T).reshape(-1) + first

    return _grouped_path_scores(flat, score_group, device=device)


def native_teacher_paths(teacher_qt, bank: ObjectBank, raw: dict, query_id: str, graph: Sequence[str],
                         augmented_e: str | None, device, *, batch: int = 32,
                         teacher_cache: FrozenTeacherLogitCache | None = None,
                         encode_cache: dict | None = None) -> torch.Tensor:
    from .train_teacher import pair_logits, paths_for_targets

    per_target = paths_for_targets(raw, query_id, graph, augmented_e)
    f0 = pair_logits(
        teacher_qt, bank, query_id, graph, device, batch=batch,
        teacher_cache=teacher_cache, encode_cache=encode_cache, view="native:target",
    )
    assets = sorted({e for lst in per_target.values() for e in lst}, key=lambda x: x.encode("utf-8"))
    # All q->e scores share the same query and frozen Teacher.  One batched
    # forward replaces one launch per evidence while preserving asset order.
    qe_scores = pair_logits(
        teacher_qt, bank, query_id, assets, device, batch=batch,
        teacher_cache=teacher_cache, encode_cache=encode_cache, view="native:first_hop",
    )
    f_qe = {e: qe_scores[i] for i, e in enumerate(assets)}
    f_et: dict[str, dict[str, torch.Tensor]] = {}
    for e in assets:
        targets = sorted({t for t, lst in per_target.items() if e in lst}, key=lambda x: x.encode("utf-8"))
        scores = pair_logits(
            teacher_qt, bank, e, targets, device, batch=batch,
            teacher_cache=teacher_cache, encode_cache=encode_cache,
            view=f"native:second_hop:{e}",
        )
        f_et[e] = {t: scores[i] for i, t in enumerate(targets)}
    stacked = []
    for i, t in enumerate(graph):
        lst = per_target[t]
        if not lst:
            stacked.append(f0[i])
            continue
        paths = torch.stack([f_qe[e] + f_et[e][t] for e in lst])
        stacked.append(torch.logsumexp(torch.cat([f0[i].reshape(1), paths]), 0))
    return torch.stack(stacked)


def train_native(model, teacher_qt, bank: ObjectBank, labels: TrainLabels, raw: dict, graph: dict, *,
                 seed: int, epochs: int, lr: float, logical_batch: int, device: str, out_dir: Path,
                 protocol_path: Path, parent_dirs: list[Path], optimizer=None, start_epoch: int = 1,
                 counters: dict | None = None,
                 batch: int = 32, teacher_cache: FrozenTeacherLogitCache | None = None,
                 anchor_registry=None,
                 log=print) -> dict:
    from .train_teacher import (_epoch_data_state, _finish_batch, _save, _save_initial,
                                backward_scaled, build_anchor_registry, query_order)

    for param in model.parameters():
        param.requires_grad_(True)
    if getattr(model, "adapter", None) is not None:
        raise ValueError("KD-NATIVE must not instantiate a conditional adapter")
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    counters = dict(counters or {"updates": 0, "active_queries": 0, "items": 0})
    anchor_registry = anchor_registry or build_anchor_registry(labels, epochs)
    _save_initial(model, optimizer, out_dir, seed, "S_KD_NATIVE_C2", protocol_path, parent_dirs)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch, namespace="order")
        for start in range(0, len(order), logical_batch):
            batch_q = order[start : start + logical_batch]
            active = len(batch_q)
            done = 0
            encode_cache: dict = {}
            for qid in batch_q:
                anchor = anchor_registry[epoch][qid]
                loss = native_query_loss(
                    model, teacher_qt, bank, labels, raw, qid, graph[qid], device, anchor,
                    batch=batch, teacher_cache=teacher_cache, encode_cache=encode_cache,
                )
                if loss is None:
                    continue
                backward_scaled(loss, active)
                done += 1
                counters["items"] += len(graph[qid])
            if done:
                _finish_batch(optimizer)
                counters["updates"] += 1
                counters["active_queries"] += done
        log(json.dumps({"event": "epoch", "stage": "S_KD_NATIVE_C2", "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, "S_KD_NATIVE_C2", epoch, protocol_path, counters,
              {"_parent_dirs": [str(p) for p in parent_dirs]}, data_state=_epoch_data_state(order))
    if teacher_cache is not None:
        teacher_cache.save()
    return counters


# ------------------------------------------------------------- QT-only -------


def qt_c1_list(labels: TrainLabels, raw: dict, query_id: str, *, hard: int = 32, random_n: int = 32):
    from .candidates import _sample_random, utf8_sorted

    entry = labels.queries[query_id]
    positives = utf8_sorted(entry["G"])
    pset = set(positives)
    hard_ids = [t for t in raw["qt_top256"][query_id] if t not in pset][:hard]
    pool = [t for t in raw["legal"] if t not in pset and t not in set(hard_ids)]
    rand = _sample_random(pool, set(), random_n, config.namespace("qt_c1", query_id))
    return list(dict.fromkeys([*positives, *hard_ids, *rand]))


def qt_c2_kd_list(teacher_qt, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str, device,
                  *, hard: int = 64, random_n: int = 32, batch: int = 32,
                  teacher_cache: FrozenTeacherLogitCache | None = None,
                  encode_cache: dict | None = None) -> list[str]:
    """All G + 64 T_QT-ranked non-positive raw-QT-Top256 candidates + 32 random (SPEC 9.7)."""
    from .candidates import _sample_random, utf8_sorted
    from .train_teacher import pair_logits

    entry = labels.queries[query_id]
    positives = utf8_sorted(entry["G"])
    pset = set(positives)
    non_positive = [t for t in raw["qt_top256"][query_id] if t not in pset]
    scores = pair_logits(
        teacher_qt, bank, query_id, non_positive, device, batch=batch,
        teacher_cache=teacher_cache, encode_cache=encode_cache, view="qt_c2_hard",
    )
    score_values = scores.detach().cpu().tolist()
    order = sorted(range(len(non_positive)), key=lambda i: (-score_values[i], non_positive[i].encode("utf-8")))
    hard_ids = [non_positive[i] for i in order[:hard]]
    pool = [t for t in raw["legal"] if t not in pset and t not in set(hard_ids)]
    rand = _sample_random(pool, set(), random_n, config.namespace("qt_c2kd", query_id))
    return list(dict.fromkeys([*positives, *hard_ids, *rand]))


@dataclass(frozen=True)
class QTListIndex:
    candidates: tuple[str, ...]
    teacher_logits: torch.Tensor | None


def build_qt_static_registry(teacher_qt, bank: ObjectBank, labels: TrainLabels, raw: dict, *,
                             stage: str, device, batch: int = 128,
                             teacher_cache: FrozenTeacherLogitCache | None = None,
                             log=print) -> dict[str, QTListIndex]:
    """Freeze QT candidate lists and all T_QT logits before epoch training."""
    if not stage.endswith("_C1") and "KD" not in stage:
        return {}
    if "KD" in stage:
        teacher_qt.eval()
    registry: dict[str, QTListIndex] = {}
    query_ids = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    with torch.no_grad():
        for number, query_id in enumerate(query_ids, 1):
            encode_cache: dict = {}
            if stage.endswith("_C1"):
                candidates = qt_c1_list(labels, raw, query_id)
                view = "qt_c1"
            else:
                candidates = qt_c2_kd_list(
                    teacher_qt, bank, labels, raw, query_id, device, batch=batch,
                    teacher_cache=teacher_cache, encode_cache=encode_cache,
                )
                view = "qt_c2_kd"
            teacher_logits = None
            if "KD" in stage:
                from .train_teacher import pair_logits

                teacher_logits = pair_logits(
                    teacher_qt, bank, query_id, candidates, device, batch=batch,
                    teacher_cache=teacher_cache, encode_cache=encode_cache, view=view,
                ).detach().to(dtype=torch.float32, device="cpu")
            registry[query_id] = QTListIndex(tuple(candidates), teacher_logits)
            if log and number % 250 == 0:
                log(json.dumps({"event": "static_registry", "stage": stage, "queries": number}))
    if teacher_cache is not None:
        teacher_cache.save()
    return registry


def streamed_full_trainable(model, bank: ObjectBank, query_id: str, positives: set[str], legal: Sequence[str],
                            device, *, chunk: int = 4096) -> torch.Tensor | None:
    """Full-denominator loss whose target keys follow the current P (trainable)."""
    if not rank_active(legal, positives):
        return None
    q = model.query(bank.z(query_id).to(device))
    legal_tensor = torch.tensor([t in positives for t in legal], dtype=torch.bool, device=device)
    alls, poss = [], []
    for start in range(0, len(legal), chunk):
        group = legal[start : start + chunk]
        keys = model.keys(bank.z_many(list(group)).to(device))
        scores = q @ keys.T
        p = legal_tensor[start : start + chunk]
        alls.append(torch.logsumexp(scores, 0))
        if any(t in positives for t in group):
            poss.append(torch.logsumexp(scores[p], 0))
    return torch.logsumexp(torch.stack(alls), 0) - torch.logsumexp(torch.stack(poss), 0)


def train_qt_student(model, teacher_qt, bank: ObjectBank, labels: TrainLabels, raw: dict, *,
                     seed: int, stage: str, epochs: int, lr: float, logical_batch: int, device: str,
                     out_dir: Path, protocol_path: Path, parent_dirs: list[Path], optimizer=None,
                     start_epoch: int = 1, counters: dict | None = None, batch: int = 128, full_chunk: int = 4096,
                     teacher_cache: FrozenTeacherLogitCache | None = None, log=print) -> dict:
    from .train_teacher import _epoch_data_state, _finish_batch, _save, _save_initial, backward_scaled, query_order

    checkpoint_stage = stage if stage.startswith("S_") else f"S_{stage}"
    for param in model.parameters():
        param.requires_grad_(True)
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    counters = dict(counters or {"updates": 0, "active_queries": 0, "items": 0})
    qt_registry = build_qt_static_registry(
        teacher_qt, bank, labels, raw, stage=stage, device=device, batch=batch,
        teacher_cache=teacher_cache, log=log,
    )
    _save_initial(model, optimizer, out_dir, seed, checkpoint_stage, protocol_path, parent_dirs)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch, namespace="order")
        for batch_start in range(0, len(order), logical_batch):
            batch_q = order[batch_start : batch_start + logical_batch]
            active = len(batch_q)
            done = 0
            for qid in batch_q:
                entry = labels.queries[qid]
                positives = set(entry["G"])
                if stage.endswith("_C1"):
                    prepared = qt_registry[qid]
                    candidates = list(prepared.candidates)
                    logits = model(bank.z(qid).to(device), bank.z_many(candidates).to(device))
                    p = torch.tensor([t in positives for t in candidates], dtype=torch.bool, device=device)
                    allowed = torch.ones(len(candidates), dtype=torch.bool, device=device)
                    loss = rank_loss(logits, p, allowed, active=rank_active(candidates, positives))
                    if "KD" in stage:
                        t_logits = prepared.teacher_logits.to(device)
                        kd_term = kd_loss(logits, t_logits, allowed, temperature=2.0)
                        loss = kd_term if loss is None else loss + kd_term
                else:
                    loss = streamed_full_trainable(model, bank, qid, positives, raw["legal"], device, chunk=full_chunk)
                    if loss is not None and "KD" in stage:
                        prepared = qt_registry[qid]
                        candidates = list(prepared.candidates)
                        logits = model(bank.z(qid).to(device), bank.z_many(candidates).to(device))
                        allowed = torch.ones(len(candidates), dtype=torch.bool, device=device)
                        t_logits = prepared.teacher_logits.to(device)
                        loss = loss + kd_loss(logits, t_logits, allowed, temperature=2.0)
                if loss is None:
                    continue
                backward_scaled(loss, active)
                done += 1
                counters["items"] += 1
            if done:
                _finish_batch(optimizer)
                counters["updates"] += 1
                counters["active_queries"] += done
            if log and counters["updates"] and counters["updates"] % 50 == 0:
                log(json.dumps({"event": "progress", "stage": checkpoint_stage, "updates": counters["updates"],
                                "active_queries": counters["active_queries"]}))
        log(json.dumps({"event": "epoch", "stage": checkpoint_stage, "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, checkpoint_stage, epoch, protocol_path, counters,
              {"_parent_dirs": [str(p) for p in parent_dirs]}, data_state=_epoch_data_state(order))
    if teacher_cache is not None:
        teacher_cache.save()
    return counters

"""Teacher stages: T_EDGE, one hard refresh, then T_PATH and T_QT (SPEC 7).

Both branches start from the same T_EDGE endpoint tensors with their own fresh
AdamW.  T_QT never calls the QET forward and always uses all of ``G`` as its
positive set.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import config, lineage
from .contracts import local_rng, rank_loss
from .inputs import TrainLabels
from .losses import mean_active, support_margin
from .score import ObjectBank
from .teacher_cache import FrozenTeacherLogitCache

# SPEC 7.4: the path forward block is capped at 8 and must be the largest value
# that fits memory; list batching (edges, conditional KD) is a separate knob.
PATH_BLOCK = 8


# ------------------------------------------------------------------ helpers ---


def masks(ids: Sequence[str], positives: set[str], ignore: set[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.tensor([i in positives for i in ids], dtype=torch.bool, device=device)
    allowed = torch.tensor([i not in ignore for i in ids], dtype=torch.bool, device=device)
    return p, allowed


def rank_active(ids: Sequence[str], positives: set[str], ignore: set[str] | None = None) -> bool:
    """CPU membership check for static candidate lists.

    The caller already has these Python IDs while constructing a list.  Passing
    the result to ``rank_loss`` avoids synchronising CUDA once for each list
    merely to discover whether it has a positive and a negative.
    """
    ignored = ignore or set()
    return bool(positives) and any(item not in positives and item not in ignored for item in ids)


def _epoch_data_state(order: Sequence[str]) -> dict[str, object]:
    encoded = "\n".join(order).encode("utf-8")
    return {
        "order": list(order),
        "order_cursor": len(order),
        "order_hash": hashlib.sha256(encoded).hexdigest(),
    }


def query_order(labels: TrainLabels, seed: int, epoch: int, namespace: str = "order") -> list[str]:
    qids = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    rng = local_rng(config.namespace(namespace, "teacher", seed, epoch))
    rng.shuffle(qids)
    return qids


def anchor_choice(labels: TrainLabels, query_id: str, epoch: int) -> tuple[str, str] | None:
    """One labelled (e*, t*) anchor per query per epoch (SPEC 7.3/10.3)."""
    annotations = sorted(
        ((t, e) for t, evidence in labels.queries[query_id]["W"].items() for e in evidence),
        key=lambda p: (p[0].encode("utf-8"), p[1].encode("utf-8")),
    )
    if not annotations:
        return None
    offset = local_rng(config.namespace("anchor", query_id)).randrange(len(annotations))
    return annotations[(offset + epoch - 1) % len(annotations)]


def _pair_logits_online(model, bank: ObjectBank, anchor_id: str, dest_ids: Sequence[str], device, *,
                        batch: int = 128, encode_cache: dict | None = None) -> torch.Tensor:
    kind = bank.kind(anchor_id)
    az = bank.z(anchor_id).to(device)
    ac = bank.tokens(anchor_id).to(device)
    dz = bank.z_many(list(dest_ids)).to(device)
    cache = encode_cache if encode_cache is not None else {}
    out = []
    for start in range(0, len(dest_ids), batch):
        group = list(dest_ids[start : start + batch])
        pairs = [
            (kind, az, ac, bank.kind(d), dz[start + i], bank.tokens(d).to(device))
            for i, d in enumerate(group)
        ]
        out.append(model.score_pairs(pairs, cache=cache,
                                     cache_keys=[(anchor_id, d) for d in group]))
    if not out:
        return torch.zeros(0, device=device)
    return torch.cat(out)


def pair_logits(model, bank: ObjectBank, anchor_id: str, dest_ids: Sequence[str], device, *,
                batch: int = 128, teacher_cache: FrozenTeacherLogitCache | None = None,
                encode_cache: dict | None = None, view: str = "pair") -> torch.Tensor:
    """Pair scores, optionally served from a frozen-Teacher logits cache."""
    if teacher_cache is None:
        return _pair_logits_online(model, bank, anchor_id, dest_ids, device,
                                   batch=batch, encode_cache=encode_cache)
    ids = list(dest_ids)
    return teacher_cache.get_or_compute(
        kind="pair", query_id=anchor_id, evidence_id=None, target_id=tuple(ids),
        candidate_ids=ids, mask=None, view=view, device=device,
        compute=lambda: _pair_logits_online(model, bank, anchor_id, ids, device,
                                            batch=batch, encode_cache=encode_cache),
    )


def _triplet_logits_online(model, bank: ObjectBank, query_id: str, evidence_id: str,
                           dest_ids: Sequence[str], device, *, batch: int = 128,
                           encode_cache: dict | None = None) -> torch.Tensor:
    qz = bank.z(query_id).to(device)
    qc = bank.tokens(query_id).to(device)
    ek = bank.kind(evidence_id)
    ez = bank.z(evidence_id).to(device)
    ec = bank.tokens(evidence_id).to(device)
    dz = bank.z_many(list(dest_ids)).to(device)
    cache = encode_cache if encode_cache is not None else {}
    out = []
    for start in range(0, len(dest_ids), batch):
        group = list(dest_ids[start : start + batch])
        triplets = [
            ("table", qz, qc, ek, ez, ec, bank.kind(d), dz[start + i], bank.tokens(d).to(device))
            for i, d in enumerate(group)
        ]
        out.append(model.score_triplets(
            triplets, cache=cache,
            cache_keys=[(query_id, evidence_id, d) for d in group],
        ))
    if not out:
        return torch.zeros(0, device=device)
    return torch.cat(out)


def triplet_logits(model, bank: ObjectBank, query_id: str, evidence_id: str,
                   dest_ids: Sequence[str], device, *, batch: int = 128,
                   teacher_cache: FrozenTeacherLogitCache | None = None,
                   encode_cache: dict | None = None, view: str = "qet") -> torch.Tensor:
    ids = list(dest_ids)
    if teacher_cache is None:
        return _triplet_logits_online(model, bank, query_id, evidence_id, ids, device,
                                      batch=batch, encode_cache=encode_cache)
    return teacher_cache.get_or_compute(
        kind="qet", query_id=query_id, evidence_id=evidence_id, target_id=None,
        candidate_ids=ids, mask=None, view=view, device=device,
        compute=lambda: _triplet_logits_online(
            model, bank, query_id, evidence_id, ids, device,
            batch=batch, encode_cache=encode_cache,
        ),
    )


def path_lse(zero_hop: torch.Tensor, path_scores: Sequence[torch.Tensor]) -> torch.Tensor:
    if not path_scores:
        return zero_hop
    return torch.logsumexp(torch.stack([zero_hop, *path_scores]), 0)


# --------------------------------------------------------------- edge stage ---


def edge_relation_losses(model, bank: ObjectBank, query_id: str, groups: dict, edge_positives: dict,
                         labels: TrainLabels, device, *, batch: int = 128,
                         encode_cache: dict | None = None) -> dict[str, torch.Tensor | None]:
    by_relation: dict[str, list[torch.Tensor | None]] = {}
    for key, candidates in groups.items():
        if not candidates:
            continue
        positives = set(edge_positives[query_id][key])
        if key in ("QT", "Q_text", "Q_image"):
            relation = key
            logits = pair_logits(model, bank, query_id, candidates, device, batch=batch,
                                 encode_cache=encode_cache)
        else:
            relation = "ET"
            asset = key[2:]
            logits = pair_logits(model, bank, asset, candidates, device, batch=batch,
                                 encode_cache=encode_cache)
        p, allowed = masks(candidates, positives, set(), device)
        by_relation.setdefault(relation, []).append(rank_loss(
            logits, p, allowed, active=rank_active(candidates, positives),
            validate_finite=False, validate_masks=False,
        ))
    return {relation: mean_active(values) for relation, values in by_relation.items()}


def train_edge(model, bank: ObjectBank, labels: TrainLabels, edge: dict, edge_positives: dict, *,
               seed: int, epochs: int, lr: float, logical_batch: int, device: str,
               out_dir: Path, protocol_path: Path, optimizer: torch.optim.Optimizer | None = None,
               start_epoch: int = 1, counters: dict | None = None, batch: int = 128, log=print) -> dict:
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    _stage_name = "T_EDGE"
    counters = dict(counters or {"updates": 0, "active_queries": 0, "relations": 0})
    _save_initial(model, optimizer, out_dir, seed, _stage_name, protocol_path)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        buffer: list[torch.Tensor] = []
        encode_cache: dict = {}
        for qid in order:
            # One stable-ID encode memo is shared by all relations in the
            # optimizer step; it is cleared after the step so no trainable
            # graph crosses an update boundary.
            losses = edge_relation_losses(model, bank, qid, edge[qid], edge_positives, labels, device,
                                         batch=batch, encode_cache=encode_cache)
            total = mean_active(losses.values())
            if total is None:
                continue
            _require_finite(total, _stage_name, qid)
            buffer.append(total)
            pending += 1
            counters["relations"] += sum(1 for v in losses.values() if v is not None)
            if pending == logical_batch:
                _accumulate(buffer, pending, optimizer)
                counters["updates"] += 1
                counters["active_queries"] += pending
                buffer, pending, encode_cache = [], 0, {}
                if counters["updates"] % 50 == 0:
                    log(json.dumps({"event": "progress", "stage": _stage_name, "epoch": epoch,
                                    "updates": counters["updates"],
                                    "active_queries": counters["active_queries"],
                                    "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)
                                    if torch.cuda.is_available() else None,
                                    "allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 2)
                                    if torch.cuda.is_available() else None}))
        if pending:
            _accumulate(buffer, pending, optimizer)
            counters["updates"] += 1
            counters["active_queries"] += pending
        log(json.dumps({"event": "epoch", "stage": "T_EDGE", "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, "T_EDGE", epoch, protocol_path, counters,
              data_state=_epoch_data_state(order))
    return counters


def _accumulate(buffer: list[torch.Tensor], active: int, optimizer) -> None:
    total = torch.stack(buffer).sum() / active
    total.backward()
    _finish_batch(optimizer)


def _finish_batch(optimizer) -> None:
    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _require_finite(loss: torch.Tensor, stage: str, query_id: str) -> None:
    """One correctness synchronisation per active query, rather than per list."""
    if not torch.isfinite(loss).all().item():
        raise ValueError(f"{stage}: non-finite query loss for {query_id}")


def backward_scaled(loss: torch.Tensor, active: int) -> None:
    """Per-item backward used where target keys are trainable.

    Retaining the autograd graph of a whole logical batch would save one
    target-matrix slice per chunk per query (hundreds of MB each); SPEC 10.1
    explicitly allows recompute, so the accumulation is done by summing
    gradients of ``loss / active`` immediately, then one clip+step per batch.
    """
    (loss / active).backward()


def _checkpoint_payload(model, optimizer, *, seed: int, stage: str, protocol_path: Path,
                        parent_dirs: list[Path], epoch: int | None = None,
                        counters: dict | None = None, data_state: dict | None = None) -> dict:
    meta = lineage.stage_manifest(None, seed, stage, protocol_path, parent_dirs)
    payload = {
        **meta,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": lineage.capture_rng_state(),
    }
    if epoch is not None:
        payload.update({"epoch": epoch, "counters": dict(counters or {}), "data_state": data_state or {}})
    return payload


def _save_initial(model, optimizer, out_dir: Path, seed: int, stage: str, protocol_path: Path,
                  parent_dirs: list[Path] | None = None) -> Path:
    """Persist the deterministic fresh start before the first optimizer update."""
    path = Path(out_dir) / "init.pt"
    if path.exists():
        return path
    payload = _checkpoint_payload(
        model, optimizer, seed=seed, stage=stage, protocol_path=protocol_path,
        parent_dirs=list(parent_dirs or []),
    )
    lineage.atomic_save(path, payload)
    return path


def _save(model, optimizer, out_dir: Path, seed: int, stage: str, epoch: int, protocol_path: Path, counters: dict,
          extra: dict | None = None, data_state: dict | None = None) -> Path:
    extra = dict(extra or {})
    parent_dirs = [Path(p) for p in extra.pop("_parent_dirs", [])]
    payload = _checkpoint_payload(
        model, optimizer, seed=seed, stage=stage, protocol_path=protocol_path,
        parent_dirs=parent_dirs, epoch=epoch, counters=counters, data_state=data_state,
    )
    payload.update(extra)
    path = Path(out_dir) / f"epoch{epoch}.pt"
    artifact_id = lineage.atomic_save(path, payload)
    lineage.atomic_link(path, Path(out_dir) / "checkpoint.pt")
    lineage.StageReceipt(
        stage=stage, model_seed=seed, epoch=epoch, updates=counters.get("updates", 0),
        active_queries=counters.get("active_queries", 0), items=counters.get("items", 0),
        lists=counters.get("lists", 0), parent_artifact_ids=payload["parent_artifact_ids"],
        notes={"counters": counters, "checkpoint_sha256": artifact_id, "data_state": data_state or {}},
    ).write(Path(out_dir) / "receipt.json")
    return path


# ------------------------------------------------------- target graph build ---


def build_train_graph(model, bank: ObjectBank, labels: TrainLabels, raw: dict, *, device: str,
                      direct: int = 100, extra_hard: int = 32, batch: int = 32,
                      log=None) -> dict[str, list[str]]:
    """SPEC 5.5 common target graph: raw C100 u G u top32 T_EDGE QT non-positive of raw U."""
    graph: dict[str, list[str]] = {}
    order_all = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    with torch.no_grad():
        for n, qid in enumerate(order_all):
            entry = labels.queries[qid]
            encode_cache: dict = {}
            admission = raw["admission"][qid]
            pooled = list(dict.fromkeys([*admission["C100"], *entry["G"]]))
            pool_set = set(pooled)
            positives = set(entry["G"])
            candidates = [t for t in admission["U"] if t not in positives]
            if candidates:
                scores = pair_logits(model, bank, qid, candidates, device, batch=batch,
                                     encode_cache=encode_cache)
                score_values = scores.detach().cpu().tolist()
                order = sorted(range(len(candidates)),
                               key=lambda i: (-score_values[i], candidates[i].encode("utf-8")))
                for i in order[:extra_hard]:
                    tid = candidates[i]
                    if tid not in pool_set:
                        pooled.append(tid)
                        pool_set.add(tid)
            graph[qid] = pooled
            if log and n % 1000 == 0:
                log(json.dumps({"event": "graph_progress", "done": n, "total": len(order_all)}))
    return graph


def refreshed_hard(model, bank: ObjectBank, labels: TrainLabels, raw: dict, *, device: str,
                   batch: int = 32, log=None) -> dict[tuple[str, str], list[str]]:
    """One authorised hard refresh (SPEC 5.5); pure inference, no autograd graph."""
    out: dict[tuple[str, str], list[str]] = {}
    order_all = sorted(labels.queries, key=lambda x: x.encode("utf-8"))
    with torch.no_grad():
        for n, qid in enumerate(order_all):
            entry = labels.queries[qid]
            # Object encodings contain no dropout, so the cache is safe for all
            # independently-scored reservoirs of this query.
            encode_cache: dict = {}
            specs = [("QT", raw["qt_reservoir"][qid])]
            for modality in ("text", "image"):
                if entry["Qpos"][modality]:
                    specs.append((f"Q_{modality}", raw["qe_reservoir"][qid][modality]))
            for asset in sorted({a for ev in entry["W"].values() for a in ev}, key=lambda x: x.encode("utf-8")):
                if labels.epos.get(asset) and raw["et_reservoir"].get(asset):
                    specs.append((f"E_{asset}", raw["et_reservoir"][asset]))
            for key, reservoir in specs:
                anchor = qid if not key.startswith("E_") else key[2:]
                scores = pair_logits(model, bank, anchor, reservoir, device, batch=batch,
                                     encode_cache=encode_cache)
                score_values = scores.detach().cpu().tolist()
                order = sorted(range(len(reservoir)),
                               key=lambda i: (-score_values[i], reservoir[i].encode("utf-8")))
                out[(qid, key)] = [reservoir[i] for i in order]
            if log and n % 500 == 0:
                log(json.dumps({"event": "refresh_progress", "done": n, "total": len(order_all)}))
    return out


# --------------------------------------------------- conditional small list ---


@dataclass(frozen=True)
class ConditionalList:
    candidates: tuple[str, ...]
    positives: tuple[str, ...]
    ignore: tuple[str, ...]


def _build_conditional_list(labels: TrainLabels, raw: dict, query_id: str, asset: str, *,
                            hard: int, random_n: int) -> ConditionalList:
    from .candidates import _sample_random, utf8_sorted

    entry = labels.queries[query_id]
    p = utf8_sorted([t for t, evidence in entry["W"].items() if asset in evidence])
    p_set = set(p)
    legal = raw["legal"]
    ignore_set = (set(entry["G"]) | set(labels.epos.get(asset, []))) & set(legal)
    ignore_set -= p_set
    ignore = utf8_sorted(ignore_set)
    reservoir = raw["et_reservoir"].get(asset, [])
    hard_ids = [t for t in reservoir if t not in p_set and t not in ignore_set][:hard]
    hard_set = set(hard_ids)
    pool = [t for t in legal if t not in p_set and t not in ignore_set and t not in hard_set]
    rand = _sample_random(pool, set(), random_n, config.namespace("cond", query_id, asset))
    return ConditionalList(tuple(dict.fromkeys([*p, *hard_ids, *rand])), tuple(p), tuple(ignore))


def build_conditional_registry(labels: TrainLabels, raw: dict, *, hard: int = 64,
                               random_n: int = 32) -> dict[str, dict[str, ConditionalList]]:
    """Precompute all static ``(q,e)`` conditional lists after the edge refresh."""
    registry: dict[str, dict[str, ConditionalList]] = {}
    for query_id, entry in labels.queries.items():
        assets = sorted({asset for evidence in entry["W"].values() for asset in evidence},
                        key=lambda x: x.encode("utf-8"))
        registry[query_id] = {
            asset: _build_conditional_list(labels, raw, query_id, asset, hard=hard, random_n=random_n)
            for asset in assets
        }
    return registry


def conditional_list(labels: TrainLabels, raw: dict, query_id: str, asset: str, *,
                     hard: int = 64, random_n: int = 32,
                     registry: Mapping[str, Mapping[str, ConditionalList]] | None = None) -> tuple[list[str], list[str], list[str]]:
    """The shared small QET list: all P, 64 raw-hard, 32 random (SPEC 7.3/5.4)."""
    prepared = registry.get(query_id, {}).get(asset) if registry is not None else None
    prepared = prepared or _build_conditional_list(labels, raw, query_id, asset, hard=hard, random_n=random_n)
    return list(prepared.candidates), list(prepared.positives), list(prepared.ignore)


def paths_for_targets(raw: dict, query_id: str, targets: Sequence[str], augmented_e: str | None):
    """Per target: the natural retained e list, optionally plus the shared augmented e."""
    stored = raw["admission"][query_id]["paths"]
    out: dict[str, list[str]] = {}
    for t in targets:
        lst = [p[0] for p in stored.get(t, [])]
        if augmented_e is not None:
            lst = list(dict.fromkeys([*lst, augmented_e]))
        out[t] = lst
    return out


def _triplet_pairs_online(model, bank: ObjectBank, query_id: str, pairs: Sequence[tuple[str, str]], device,
                          *, batch: int = 128, encode_cache: dict | None = None) -> torch.Tensor:
    """Differentiable QET scores for (e, t) pairs, in the given order."""
    if not pairs:
        return torch.zeros(0, device=device)
    qz = bank.z(query_id).to(device)
    qc = bank.tokens(query_id).to(device)
    assets = list(dict.fromkeys(e for e, _ in pairs))
    targets = list(dict.fromkeys(t for _, t in pairs))
    target_index = {t: i for i, t in enumerate(targets)}
    tz = bank.z_many(targets).to(device)
    cache = encode_cache if encode_cache is not None else {}
    out = []
    for start in range(0, len(pairs), batch):
        group = list(pairs[start : start + batch])
        triplets = [
            ("table", qz, qc, bank.kind(e), bank.z(e).to(device), bank.tokens(e).to(device),
             bank.kind(t), tz[target_index[t]], bank.tokens(t).to(device))
            for e, t in group
        ]
        out.append(model.score_triplets(
            triplets, cache=cache,
            cache_keys=[(query_id, e, t) for e, t in group],
        ))
    return torch.cat(out)


def triplet_pairs(model, bank: ObjectBank, query_id: str, pairs: Sequence[tuple[str, str]], device,
                  *, batch: int = 128, teacher_cache: FrozenTeacherLogitCache | None = None,
                  mask: Sequence[bool] | None = None, view: str = "qet",
                  encode_cache: dict | None = None) -> torch.Tensor:
    """Differentiable QET scores, optionally served from a frozen cache."""
    if not pairs:
        return torch.zeros(0, device=device)
    pair_ids = [f"{e}\x00{t}" for e, t in pairs]
    if teacher_cache is None:
        return _triplet_pairs_online(model, bank, query_id, pairs, device,
                                     batch=batch, encode_cache=encode_cache)
    return teacher_cache.get_or_compute(
        kind="qet_pairs", query_id=query_id,
        evidence_id=tuple(e for e, _ in pairs), target_id=tuple(t for _, t in pairs),
        candidate_ids=pair_ids, mask=mask, view=view, device=device,
        compute=lambda: _triplet_pairs_online(
            model, bank, query_id, pairs, device,
            batch=batch, encode_cache=encode_cache,
        ),
    )


def target_view_scores(model, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str,
                       graph: Sequence[str], augmented_e: str | None, device,
                       *, batch: int = 32, encode_cache: dict | None = None) -> torch.Tensor:
    """S^v(q,t) = LSE(f0(q,t), {f(q,e,t) : e in B^v(q,t)}) for every target."""
    f0 = pair_logits(model, bank, query_id, graph, device, batch=batch, encode_cache=encode_cache)
    per_target_paths = paths_for_targets(raw, query_id, graph, augmented_e)
    flat = [(e, t) for t in graph for e in per_target_paths[t]]
    scores = triplet_pairs(model, bank, query_id, flat, device, batch=batch, encode_cache=encode_cache)
    cursor = 0
    stacked = []
    for i, t in enumerate(graph):
        n = len(per_target_paths[t])
        if n:
            stacked.append(torch.logsumexp(torch.cat([f0[i].reshape(1), scores[cursor : cursor + n]]), 0))
            cursor += n
        else:
            stacked.append(f0[i])
    return torch.stack(stacked)


def augmented_scores_from_natural(model, bank: ObjectBank, raw: dict, query_id: str,
                                  graph: Sequence[str], natural: torch.Tensor, evidence_id: str,
                                  device, *, batch: int = PATH_BLOCK,
                                  encode_cache: dict | None = None) -> torch.Tensor:
    """Evaluation-only algebraic shortcut for an augmented view.

    The augmented view only adds the single evidence ``e*`` to every target, so
    ``S_aug(q,t) = LSE(S_nat(q,t), f(q,e*,t))``.  It is valid with a frozen
    deterministic model, but training must use ``target_view_scores`` for each
    view so dropout consumes the same random draws as the specified full path
    forwards.
    """
    natural_paths = paths_for_targets(raw, query_id, graph, None)
    pending = [t for t in graph if evidence_id not in natural_paths[t]]
    if not pending:
        return natural
    extra = triplet_pairs(model, bank, query_id, [(evidence_id, t) for t in pending], device,
                          batch=batch, encode_cache=encode_cache)
    position = {t: i for i, t in enumerate(pending)}
    parts = []
    for i, t in enumerate(graph):
        j = position.get(t)
        if j is None:
            parts.append(natural[i])
        else:
            parts.append(torch.logsumexp(torch.stack([natural[i], extra[j]]), 0))
    return torch.stack(parts)


def target_view_loss(model, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str,
                     graph: Sequence[str], positive: set[str], augmented_e: str | None, device,
                     *, batch: int = 32) -> torch.Tensor | None:
    S = target_view_scores(model, bank, labels, raw, query_id, graph, augmented_e, device, batch=batch)
    p = torch.tensor([t in positive for t in graph], dtype=torch.bool, device=device)
    allowed = torch.ones(len(graph), dtype=torch.bool, device=device)
    return rank_loss(S, p, allowed)


def conditional_loss(model, bank: ObjectBank, labels: TrainLabels, raw: dict, query_id: str,
                     device, *, hard: int = 64, random_n: int = 32, batch: int = 32,
                     encode_cache: dict | None = None,
                     conditional_registry: Mapping[str, Mapping[str, ConditionalList]] | None = None) -> torch.Tensor | None:
    entry = labels.queries[query_id]
    assets = sorted({a for ev in entry["W"].values() for a in ev}, key=lambda x: x.encode("utf-8"))
    losses = []
    for asset in assets:
        candidates, p_ids, ignore = conditional_list(
            labels, raw, query_id, asset, hard=hard, random_n=random_n,
            registry=conditional_registry,
        )
        if not candidates or not p_ids:
            continue
        p_set, ignore_set = set(p_ids), set(ignore)
        logits = triplet_logits(model, bank, query_id, asset, candidates, device,
                                batch=batch, encode_cache=encode_cache)
        p, allowed = masks(candidates, p_set, ignore_set, device)
        losses.append(rank_loss(
            logits, p, allowed, active=rank_active(candidates, p_set, ignore_set),
            validate_finite=False, validate_masks=False,
        ))
    return mean_active(losses)


def support_loss(model, bank: ObjectBank, labels: TrainLabels, query_id: str, anchor, device,
                 *, margin: float = 1.0, encode_cache: dict | None = None) -> torch.Tensor | None:
    if anchor is None:
        return None
    t_star, e_star = anchor
    if t_star in set(labels.queries[query_id]["D"]):
        return None
    zero = pair_logits(model, bank, query_id, [t_star], device, batch=1,
                       encode_cache=encode_cache)[0]
    with_e = triplet_pairs(model, bank, query_id, [(e_star, t_star)], device, batch=1,
                           encode_cache=encode_cache)[0]
    return support_margin(zero, with_e, margin)


def train_path(model, bank: ObjectBank, labels: TrainLabels, edge: dict, edge_positives: dict,
               graph: dict, raw: dict, *, seed: int, epochs: int, lr: float, logical_batch: int,
               device: str, out_dir: Path, protocol_path: Path, parent_dirs: list[Path],
               optimizer: torch.optim.Optimizer | None = None, start_epoch: int = 1,
               edge_weight: float = 0.5, conditional_weight: float = 0.5, support_weight: float = 0.2,
               batch: int = 32, counters: dict | None = None,
               conditional_registry: Mapping[str, Mapping[str, ConditionalList]] | None = None,
               log=print) -> dict:
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    _stage_name = "T_PATH"
    counters = dict(counters or {"updates": 0, "active_queries": 0, "items": 0})
    conditional_registry = conditional_registry or build_conditional_registry(labels, raw)
    _save_initial(model, optimizer, out_dir, seed, _stage_name, protocol_path, parent_dirs)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch)
        buffer: list[torch.Tensor] = []
        pending = 0
        encode_cache: dict = {}
        for qid in order:
            entry = labels.queries[qid]
            anchor = anchor_choice(labels, qid, epoch)
            targets = graph[qid]
            positive = set(entry["G"]) & set(targets)
            p_mask = torch.tensor([t in positive for t in targets], dtype=torch.bool, device=device)
            allowed = torch.ones(len(targets), dtype=torch.bool, device=device)
            natural_scores = target_view_scores(model, bank, labels, raw, qid, targets, None, device,
                                                batch=PATH_BLOCK, encode_cache=encode_cache)
            active = rank_active(targets, positive)
            natural = rank_loss(natural_scores, p_mask, allowed, active=active,
                                validate_finite=False, validate_masks=False)
            augmented = None
            if anchor is not None:
                # Training must not reuse Natural logits: each view needs its
                # own Transformer forward and dropout realization.
                aug_scores = target_view_scores(model, bank, labels, raw, qid, targets, anchor[1], device,
                                                batch=PATH_BLOCK, encode_cache=encode_cache)
                augmented = rank_loss(aug_scores, p_mask, allowed, active=active,
                                      validate_finite=False, validate_masks=False)
            target_loss = mean_active([natural, augmented])
            edges = edge_relation_losses(model, bank, qid, edge[qid], edge_positives, labels, device,
                                        batch=batch, encode_cache=encode_cache)
            edge_loss = mean_active(edges.values())
            cond = conditional_loss(model, bank, labels, raw, qid, device, batch=batch,
                                    encode_cache=encode_cache, conditional_registry=conditional_registry)
            support = support_loss(model, bank, labels, qid, anchor, device,
                                   encode_cache=encode_cache)
            terms = []
            if target_loss is not None:
                terms.append(target_loss)
            if edge_loss is not None:
                terms.append(edge_weight * edge_loss)
            if cond is not None:
                terms.append(conditional_weight * cond)
            if support is not None:
                terms.append(support_weight * support)
            if not terms:
                continue
            total = torch.stack(terms).sum()
            _require_finite(total, _stage_name, qid)
            buffer.append(total)
            pending += 1
            counters["items"] += len(graph[qid])
            if pending == logical_batch:
                _accumulate(buffer, pending, optimizer)
                counters["updates"] += 1
                counters["active_queries"] += pending
                buffer, pending, encode_cache = [], 0, {}
                if counters["updates"] % 50 == 0:
                    log(json.dumps({"event": "progress", "stage": _stage_name, "epoch": epoch,
                                    "updates": counters["updates"],
                                    "active_queries": counters["active_queries"],
                                    "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)
                                    if torch.cuda.is_available() else None,
                                    "allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 2)
                                    if torch.cuda.is_available() else None}))
        if pending:
            _accumulate(buffer, pending, optimizer)
            counters["updates"] += 1
            counters["active_queries"] += pending
        log(json.dumps({"event": "epoch", "stage": "T_PATH", "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, "T_PATH", epoch, protocol_path, counters,
              {"_parent_dirs": [str(p) for p in parent_dirs]}, data_state=_epoch_data_state(order))
    return counters


def train_qt(model, bank: ObjectBank, labels: TrainLabels, edge: dict, edge_positives: dict,
             graph: dict, *, seed: int, epochs: int, lr: float, logical_batch: int, device: str,
             out_dir: Path, protocol_path: Path, parent_dirs: list[Path],
             optimizer: torch.optim.Optimizer | None = None, start_epoch: int = 1,
             edge_weight: float = 0.5, batch: int = 32, counters: dict | None = None, log=print) -> dict:
    """T_QT: f0-only target ranking on all G plus five-relation pair replay."""
    _stage_name = "T_QT"
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    model.train()
    counters = dict(counters or {"updates": 0, "active_queries": 0, "items": 0})
    _save_initial(model, optimizer, out_dir, seed, _stage_name, protocol_path, parent_dirs)
    for epoch in range(start_epoch, epochs + 1):
        order = query_order(labels, seed, epoch)
        buffer: list[torch.Tensor] = []
        pending = 0
        encode_cache: dict = {}
        for qid in order:
            entry = labels.queries[qid]
            candidates = graph[qid]
            logits = pair_logits(model, bank, qid, candidates, device, batch=batch,
                                 encode_cache=encode_cache)
            positives = set(entry["G"])
            p = torch.tensor([t in positives for t in candidates], dtype=torch.bool, device=device)
            allowed = torch.ones(len(candidates), dtype=torch.bool, device=device)
            target_loss = rank_loss(logits, p, allowed, active=rank_active(candidates, positives),
                                    validate_finite=False, validate_masks=False)
            edges = edge_relation_losses(model, bank, qid, edge[qid], edge_positives, labels, device,
                                        batch=batch, encode_cache=encode_cache)
            edge_loss = mean_active(edges.values())
            terms = []
            if target_loss is not None:
                terms.append(target_loss)
            if edge_loss is not None:
                terms.append(edge_weight * edge_loss)
            if not terms:
                continue
            total = torch.stack(terms).sum()
            _require_finite(total, _stage_name, qid)
            buffer.append(total)
            pending += 1
            counters["items"] += len(candidates)
            if pending == logical_batch:
                _accumulate(buffer, pending, optimizer)
                counters["updates"] += 1
                counters["active_queries"] += pending
                buffer, pending, encode_cache = [], 0, {}
                if counters["updates"] % 50 == 0:
                    log(json.dumps({"event": "progress", "stage": _stage_name, "epoch": epoch,
                                    "updates": counters["updates"],
                                    "active_queries": counters["active_queries"],
                                    "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)
                                    if torch.cuda.is_available() else None,
                                    "allocated_GiB": round(torch.cuda.memory_allocated() / 2**30, 2)
                                    if torch.cuda.is_available() else None}))
        if pending:
            _accumulate(buffer, pending, optimizer)
            counters["updates"] += 1
            counters["active_queries"] += pending
        log(json.dumps({"event": "epoch", "stage": "T_QT", "epoch": epoch, **counters}))
        _save(model, optimizer, out_dir, seed, "T_QT", epoch, protocol_path, counters,
              {"_parent_dirs": [str(p) for p in parent_dirs]}, data_state=_epoch_data_state(order))
    return counters

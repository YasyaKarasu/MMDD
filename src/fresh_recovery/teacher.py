"""Teacher stages: T_INIT, T_BOOT / T_QT list training, T_PATH (SPEC 7, 10.2, 11).

The architecture is ``fresh_path.models.FreshPathTeacher`` (SPEC 7); every
training loop here is new v3 code.  Pair and triplet scorers return logits in
the caller's order; grouping for encode reuse never reorders the output.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from fresh_path.models import FreshPathTeacher

from . import runlog
from .config import Paths
from .data import Labels, local_rng, seed_int, utf8_sorted
from .io import sha256_json, write_json
from .losses import kd_loss, rank_ce

TEACHER_KWARGS = dict(input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
                      text_slots=16, image_slots=24, dropout=0.1)
PAIR_CHUNK = 256
TRIPLET_CHUNK = 128
LIST_LR = 5e-5
LIST_LOGICAL = 8
LIST_MICRO = 2
PATH_LR = 1e-5
PATH_LOGICAL = 8
PATH_MICRO = 2
PATH_FROZEN_PREFIXES = ("adapters.", "poolers.", "globals.", "table_kind.", "modality.")


def make_teacher(seed: int) -> FreshPathTeacher:
    """T_INIT: deterministic fresh initialisation of every task parameter."""
    torch.manual_seed(seed_int("init", "teacher", seed))
    return FreshPathTeacher(**TEACHER_KWARGS).float()


def adamw(params, lr: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


# ----------------------------------------------------------------- scoring ---


def pair_logits(model, bank, anchor_id: str, dest_ids: Sequence[str], device, *,
                chunk: int = PAIR_CHUNK, cache: dict | None = None) -> torch.Tensor:
    """f(anchor, dest) for every dest, in the given order (batched object encoding).

    ``fastscore.pair_logits_batched`` reproduces ``FreshPathTeacher.score_pairs``
    with the objects of each chunk encoded in one pass; the differential test
    in ``integration.probe_fast_scoring`` compares values and gradients with
    ``pair_logits_reference`` below.
    """
    from .fastscore import pair_logits_batched

    return pair_logits_batched(model, bank, anchor_id, list(dest_ids), device, chunk=chunk)


def pair_logits_reference(model, bank, anchor_id: str, dest_ids: Sequence[str], device, *,
                          chunk: int = PAIR_CHUNK, cache: dict | None = None) -> torch.Tensor:
    """Per-pair reference loop over ``FreshPathTeacher.score_pairs`` (kept for the differential test)."""
    dest_ids = list(dest_ids)
    if not dest_ids:
        return torch.zeros(0, device=device)
    a_kind = bank.kind(anchor_id)
    az = bank.z(anchor_id).to(device)
    ac = bank.tokens(anchor_id).to(device)
    dz = bank.z_many(dest_ids).to(device)
    cache = cache if cache is not None else {}
    out = []
    for start in range(0, len(dest_ids), chunk):
        group = dest_ids[start : start + chunk]
        pairs = [(a_kind, az, ac, bank.kind(d), dz[start + i], bank.tokens(d).to(device))
                 for i, d in enumerate(group)]
        out.append(model.score_pairs(pairs, cache=cache, cache_keys=[(anchor_id, d) for d in group]))
    return torch.cat(out)


def triplet_logits(model, bank, query_id: str, pairs: Sequence[tuple[str, str]], device, *,
                   chunk: int = TRIPLET_CHUNK, cache: dict | None = None) -> torch.Tensor:
    """f(q, e, t) for every (e, t) slot, in the given slot order (batched object encoding)."""
    from .fastscore import triplet_logits_batched

    return triplet_logits_batched(model, bank, query_id, list(pairs), device, chunk=chunk)


def triplet_logits_reference(model, bank, query_id: str, pairs: Sequence[tuple[str, str]], device, *,
                             chunk: int = TRIPLET_CHUNK, cache: dict | None = None) -> torch.Tensor:
    """Per-slot reference loop over ``FreshPathTeacher.score_triplets`` (kept for the differential test)."""
    pairs = list(pairs)
    if not pairs:
        return torch.zeros(0, device=device)
    qz = bank.z(query_id).to(device)
    qc = bank.tokens(query_id).to(device)
    cache = cache if cache is not None else {}
    out = []
    for start in range(0, len(pairs), chunk):
        group = pairs[start : start + chunk]
        triplets = [("table", qz, qc, bank.kind(e), bank.z(e).to(device), bank.tokens(e).to(device),
                     "table", bank.z(t).to(device), bank.tokens(t).to(device)) for e, t in group]
        out.append(model.score_triplets(triplets, cache=cache, cache_keys=[(query_id, e, t) for e, t in group]))
    return torch.cat(out)


class LogitStore:
    """Frozen-Teacher logits keyed by teacher sha + kind + anchor + ordered candidates."""

    def __init__(self, path: Path | None, *, teacher_sha: str, stage: str) -> None:
        self.path = Path(path) if path is not None else None
        self.teacher_sha = teacher_sha
        self.stage = stage
        self.entries: dict[str, torch.Tensor] = {}
        if self.path is not None and self.path.exists():
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
            if payload.get("teacher_sha") != teacher_sha:
                raise ValueError(f"{self.path}: logit store belongs to another Teacher")
            self.entries = payload["entries"]

    @staticmethod
    def key(kind: str, anchor: str, candidates: Sequence[str], extra: str = "") -> str:
        digest = hashlib.sha256(("\n".join(candidates)).encode("utf-8")).hexdigest()
        return f"{kind}|{anchor}|{extra}|{len(candidates)}|{digest}"

    def get(self, key: str) -> torch.Tensor | None:
        return self.entries.get(key)

    def put(self, key: str, values: torch.Tensor) -> None:
        self.entries[key] = values.detach().to(dtype=torch.float32, device="cpu").contiguous()

    def save(self) -> str | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({"teacher_sha": self.teacher_sha, "stage": self.stage, "entries": self.entries}, tmp)
        tmp.replace(self.path)
        return runlog.sha256_file(self.path)


# ------------------------------------------------------------ list training ---


def list_loss(model, bank, row: dict, device, *, cache: dict | None = None) -> torch.Tensor | None:
    logits = pair_logits(model, bank, row["anchor_id"], row["candidates"], device, cache=cache)
    positive = torch.tensor([c in set(row["positives"]) for c in row["candidates"]], dtype=torch.bool, device=device)
    allowed = torch.tensor([c not in set(row.get("ignore", ())) for c in row["candidates"]], dtype=torch.bool, device=device)
    return rank_ce(logits, positive, allowed)


def epoch_order(item_ids: Sequence[str], *, root: str, stage: str, seed: int, epoch: int, scope: str = "") -> list[str]:
    order = utf8_sorted(item_ids)
    local_rng(root, stage, seed, epoch, scope).shuffle(order)
    return order


def matched_final_order(item_ids: Sequence[str], *, root: str, seed: int, epoch: int) -> list[str]:
    """Both final Teacher arms must consume identical logical query batches."""
    return epoch_order(item_ids, root=root, stage="MATCHED_FINAL_TEACHERS", seed=seed, epoch=epoch)


def train_lists(model, bank, lists: Mapping[str, dict], *, paths: Paths, seed: int, stage: str,
                stage_dir: Path, root: str, epochs: int, parents: dict[str, str], device: str,
                inputs: dict, lr: float = LIST_LR, logical: int = LIST_LOGICAL, micro: int = LIST_MICRO,
                epoch_hook: Callable[[int], dict] | None = None, log=print) -> dict:
    """T_BOOT / T_QT: raw-logit multi-positive CE, lists equally weighted (SPEC 7)."""
    runlog.enforce_precision()
    model = model.to(device)
    optimizer = adamw(model.parameters(), lr)
    active = [k for k, row in lists.items() if row["active"]]
    counters = {"updates": 0, "lists_consumed": 0, "epochs_done": 0, "loss_sum": 0.0}
    init_sha = runlog.state_sha(model.state_dict())
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={**inputs, "active_lists": len(active), "total_lists": len(lists)},
                   initial_state_sha=init_sha, optimizer_state="empty_AdamW",
                   config={"epochs": epochs, "lr": lr, "logical_lists": logical, "micro_lists": micro,
                           "loss": "raw_logit_multi_positive_CE_list_equal_weight", "clip_norm": 1.0,
                           "weight_decay": 0.01, "pair_chunk": PAIR_CHUNK})
    runlog.save_checkpoint(stage_dir / "init.pt", model=model, optimizer=optimizer, stage=stage, seed=seed,
                           paths=paths, parents=parents, counters=counters)
    started = time.time()
    per_epoch = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = epoch_order(active, root=root, stage=stage, seed=seed, epoch=epoch)
        epoch_loss, epoch_lists = 0.0, 0
        for start in range(0, len(order), logical):
            batch = order[start : start + logical]
            n = len(batch)
            optimizer.zero_grad(set_to_none=True)
            for m_start in range(0, n, micro):
                cache: dict = {}
                total = None
                for item_id in batch[m_start : m_start + micro]:
                    loss = list_loss(model, bank, lists[item_id], device, cache=cache)
                    if loss is None:
                        raise ValueError(f"{item_id}: inactive list reached training")
                    total = loss if total is None else total + loss
                    epoch_loss += float(loss.detach())
                (total / n).backward()
                del cache, total
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            counters["updates"] += 1
            counters["lists_consumed"] += n
            epoch_lists += n
            if counters["updates"] % 100 == 0:
                log({"event": "progress", "stage": stage, "epoch": epoch, "updates": counters["updates"],
                     "mean_loss": round(epoch_loss / max(epoch_lists, 1), 4),
                     "elapsed": round(time.time() - started, 1),
                     "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)})
        counters["epochs_done"] = epoch
        counters["loss_sum"] += epoch_loss
        record = {"epoch": epoch, "updates": counters["updates"], "lists": epoch_lists,
                  "mean_loss": epoch_loss / max(epoch_lists, 1), "elapsed": time.time() - started,
                  "order_sha256": hashlib.sha256("\n".join(order).encode()).hexdigest()}
        sha = runlog.save_checkpoint(stage_dir / f"epoch{epoch}.pt", model=model, optimizer=optimizer, stage=stage,
                                     seed=seed, paths=paths, parents=parents, counters=counters,
                                     extra={"epoch": epoch, "epoch_record": record})
        record["checkpoint_sha256"] = sha
        if epoch_hook is not None:
            model.eval()
            record["dev"] = epoch_hook(epoch)
        per_epoch.append(record)
        write_json(stage_dir / "EPOCHS.json", per_epoch)
        log({"event": "epoch", "stage": stage, **{k: v for k, v in record.items() if k != "order_sha256"}})
    final = stage_dir / f"epoch{epochs}.pt"
    link = stage_dir / "checkpoint.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(final.name)
    outputs = {f"epoch{e}.pt": per_epoch[e - 1]["checkpoint_sha256"] for e in range(1, epochs + 1)}
    runlog.post_run(stage_dir, status="COMPLETE", counters=counters, outputs=outputs,
                    notes={"epochs": per_epoch, "final_state_sha256": runlog.state_sha(model.state_dict())})
    return {"counters": counters, "epochs": per_epoch, "final_sha256": outputs[f"epoch{epochs}.pt"]}


# ------------------------------------------------------------------ T_PATH ---


def path_trainable(model) -> list[torch.nn.Parameter]:
    params = []
    for name, param in model.named_parameters():
        frozen = name.startswith(PATH_FROZEN_PREFIXES)
        param.requires_grad_(not frozen)
        if not frozen:
            params.append(param)
    return params


def view_scores(model, bank, query_id: str, targets: Sequence[str], bags: Mapping[str, Sequence[str]],
                device, *, cache: dict | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """S(q,t) = LSE(f0, {f(q,e,t)}) per target; returns (S, f0, flat_triplet_scores)."""
    f0 = pair_logits(model, bank, query_id, targets, device, cache=cache)
    flat = [(e, t) for t in targets for e in bags.get(t, ())]
    f = triplet_logits(model, bank, query_id, flat, device, cache=cache)
    stacked, cursor = [], 0
    for i, t in enumerate(targets):
        n = len(bags.get(t, ()))
        if n:
            stacked.append(torch.logsumexp(torch.cat([f0[i].reshape(1), f[cursor : cursor + n]]), 0))
            cursor += n
        else:
            stacked.append(f0[i])
    if cursor != len(flat):
        raise AssertionError("path slot accounting mismatch")
    return torch.stack(stacked), f0, f


def train_path(model, bank, items: Mapping[str, dict], *, paths: Paths, seed: int, stage_dir: Path, root: str,
               parents: dict[str, str], device: str, inputs: dict, anchor_logits: Mapping[str, torch.Tensor],
               epochs: int = 1, lr: float = PATH_LR, logical: int = PATH_LOGICAL, micro: int = PATH_MICRO,
               target_weight: float = 1.0, cond_weight: float = 0.5, anchor_weight: float = 0.3,
               snapshot_hook: Callable[[str], dict] | None = None, log=print) -> dict:
    """v3.1 T_PATH: natural target CE + conditional/witness/gap support + f0 KL.

    ``items[q]`` target bags are natural-only.  Verified witness records live in
    ``support_records`` and never alter those bags.
    ``anchor_logits[q]`` = frozen T_QT f0 over C_q (same order).
    """
    stage = "T_PATH"
    runlog.enforce_precision()
    model = model.to(device)
    params = path_trainable(model)
    optimizer = adamw(params, lr)
    order_all = [q for q in utf8_sorted(items) if items[q]["active"]]
    counters = {"updates": 0, "queries_consumed": 0, "conditional_queries": 0,
                "witness_queries": 0, "gap_queries": 0, "loss_target": 0.0,
                "loss_cond": 0.0, "loss_witness": 0.0, "loss_gap": 0.0, "loss_anchor": 0.0}
    init_sha = runlog.state_sha(model.state_dict())
    total_updates = math.ceil(len(order_all) / logical) * epochs
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={**inputs, "active_queries": len(order_all), "queries": len(items),
                           "planned_updates": total_updates,
                           "query_order_sha256": sha256_json(matched_final_order(order_all, root=root, seed=seed, epoch=1))},
                   initial_state_sha=init_sha, optimizer_state="empty_AdamW",
                   config={"epochs": epochs, "lr": lr, "logical_queries": logical, "micro_queries": micro,
                           "trainable": [n for n, p in model.named_parameters() if p.requires_grad],
                           "frozen_prefixes": list(PATH_FROZEN_PREFIXES), "target_weight": target_weight,
                           "conditional_weight": cond_weight, "witness_weight": 0.5,
                           "witness_margin": 0.5, "witness_competitors_max": 8,
                           "gap_weight": 0.1, "gap_beta": 1.0,
                           "f0_KL_anchor_weight": anchor_weight,
                           "presence_margin_weight": 0.0, "teacher_target_GT_augmentation": False,
                           "clip_norm": 1.0, "weight_decay": 0.01,
                           "pair_chunk": PAIR_CHUNK, "triplet_chunk": TRIPLET_CHUNK})
    runlog.save_checkpoint(stage_dir / "init.pt", model=model, optimizer=optimizer, stage=stage, seed=seed,
                           paths=paths, parents=parents, counters=counters)
    # Capture the detached, same-run T_QT initial relative gaps before the
    # first optimizer update.  This is deliberately label agnostic.
    initial_gap: dict[str, dict[str, torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for qid, item in items.items():
            if not item["active"]:
                continue
            _, f0_init, path_init = view_scores(model, bank, qid, item["targets"], item["natural"], device)
            gaps: dict[str, torch.Tensor] = {}
            cursor = 0
            for i, target in enumerate(item["targets"]):
                bag = item["natural"].get(target, ())
                if bag:
                    gaps[target] = (path_init[cursor:cursor + len(bag)].mean() - f0_init[i]).detach().cpu()
                    cursor += len(bag)
            initial_gap[qid] = gaps
    started = time.time()
    snapshots = {math.ceil(total_updates * f): f"frac{int(f * 100):03d}" for f in (0.5, 1.0)}
    snapshot_records = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = matched_final_order(order_all, root=root, seed=seed, epoch=epoch)
        for start in range(0, len(order), logical):
            batch = order[start : start + logical]
            n = len(batch)
            n_cond = sum(1 for q in batch if items[q]["conditions"])
            n_wit = sum(1 for q in batch if items[q].get("support_records"))
            n_gap = sum(1 for q in batch if initial_gap.get(q))
            optimizer.zero_grad(set_to_none=True)
            for q in batch:
                item = items[q]
                targets = item["targets"]
                positive = torch.tensor([t in set(item["positives"]) for t in targets], dtype=torch.bool, device=device)
                allowed = torch.ones(len(targets), dtype=torch.bool, device=device)
                teacher_f0 = anchor_logits[q].to(device)
                cache: dict = {}
                S, f0, flat_paths = view_scores(model, bank, q, targets, item["natural"], device, cache=cache)
                loss = rank_ce(S, positive, allowed)
                if loss is None:
                    raise ValueError(f"{q}: inactive natural target view reached training")
                anchor = kd_loss(f0, teacher_f0, allowed, temperature=1.0)
                term = (target_weight * loss + anchor_weight * anchor) / n
                counters["loss_target"] += float(loss.detach())
                counters["loss_anchor"] += float(anchor.detach())
                # The label-free gap anchor reuses the natural path scores
                # from this same view.  Keep that graph alive until the gap
                # term has contributed, then let the gap backward release it.
                term.backward(retain_graph=bool(initial_gap.get(q)))
                del cache, S, loss, anchor, term
                if item["conditions"]:
                    # L_cond = mean over this query's witness conditions; each condition is
                    # backwarded separately (linear), so only one conditional graph is resident.
                    k = len(item["conditions"])
                    cond_value = 0.0
                    for condition in item["conditions"]:
                        cands = condition["candidates"]
                        logits = triplet_logits(model, bank, q, [(condition["asset"], t) for t in cands], device)
                        p = torch.tensor([t in set(condition["positives"]) for t in cands], dtype=torch.bool, device=device)
                        a = torch.tensor(condition["allowed"], dtype=torch.bool, device=device)
                        value = rank_ce(logits, p, a)
                        if value is None:
                            raise ValueError(f"{q}/{condition['asset']}: inactive conditional list reached training")
                        cond_value += float(value.detach()) / k
                        (cond_weight * value / (k * n_cond)).backward()
                        del logits, value
                    counters["loss_cond"] += cond_value
                    counters["conditional_queries"] += 1
                if item.get("support_records"):
                    witness_terms = []
                    for record in item["support_records"]:
                        positives = record["positive_evidence"]
                        competitors = record.get("competitors", [])[:8]
                        f0_w = pair_logits(model, bank, q, [record["target_id"]], device)[0]
                        ids = [(asset, record["target_id"]) for asset in [*positives, *competitors]]
                        scores = triplet_logits(model, bank, q, ids, device)
                        n_pos = len(positives)
                        for pos in range(n_pos):
                            pos_score = scores[pos]
                            terms = [torch.zeros((), device=device), 0.5 + f0_w.detach() - pos_score]
                            terms.extend(0.5 + scores[n_pos + j] - pos_score for j in range(len(competitors)))
                            witness_terms.append(torch.logsumexp(torch.stack(terms), 0))
                    if witness_terms:
                        witness = torch.stack(witness_terms).mean()
                        (0.5 * witness / max(n_wit, 1)).backward()
                        counters["loss_witness"] += float(witness.detach())
                        counters["witness_queries"] += 1
                if initial_gap.get(q):
                    gap_terms = []
                    cursor = 0
                    for i, target in enumerate(targets):
                        bag = item["natural"].get(target, ())
                        if not bag:
                            continue
                        current = flat_paths[cursor:cursor + len(bag)].mean() - f0.detach()[i]
                        reference = initial_gap[q][target].to(device)
                        gap_terms.append(F.smooth_l1_loss(current, reference, beta=1.0))
                        cursor += len(bag)
                    if gap_terms:
                        gap = torch.stack(gap_terms).mean()
                        (0.1 * gap / max(n_gap, 1)).backward()
                        counters["loss_gap"] += float(gap.detach())
                        counters["gap_queries"] += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            counters["updates"] += 1
            counters["queries_consumed"] += n
            if counters["updates"] % 25 == 0:
                done = counters["queries_consumed"]
                log({"event": "progress", "stage": stage, "updates": counters["updates"], "queries": done,
                     "mean_target": round(counters["loss_target"] / done, 4),
                     "mean_cond": round(counters["loss_cond"] / max(counters["conditional_queries"], 1), 4),
                     "mean_anchor": round(counters["loss_anchor"] / done, 4),
                     "elapsed": round(time.time() - started, 1),
                     "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / 2**30, 2)})
            if counters["updates"] in snapshots:
                tag = snapshots[counters["updates"]]
                sha = runlog.save_checkpoint(stage_dir / f"{tag}.pt", model=model, optimizer=optimizer, stage=stage,
                                             seed=seed, paths=paths, parents=parents, counters=counters,
                                             extra={"snapshot": tag, "epoch": epoch})
                record = {"snapshot": tag, "updates": counters["updates"], "checkpoint_sha256": sha,
                          "elapsed": time.time() - started, "counters": dict(counters)}
                if snapshot_hook is not None:
                    model.eval()
                    record["dev"] = snapshot_hook(tag)
                    model.train()
                snapshot_records.append(record)
                write_json(stage_dir / "SNAPSHOTS.json", snapshot_records)
                log({"event": "snapshot", "stage": stage, "tag": tag, "updates": counters["updates"]})
    final = stage_dir / "frac100.pt"
    link = stage_dir / "checkpoint.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(final.name)
    outputs = {r["snapshot"] + ".pt": r["checkpoint_sha256"] for r in snapshot_records}
    runlog.post_run(stage_dir, status="COMPLETE", counters=counters, outputs=outputs,
                    notes={"snapshots": snapshot_records, "final_state_sha256": runlog.state_sha(model.state_dict()),
                           "selection_rule": "fixed end (frac100); half kept for trajectory only"})
    return {"counters": counters, "snapshots": snapshot_records}


def train_qt_cont(model, bank, items: Mapping[str, dict], *, paths: Paths, seed: int, stage_dir: Path,
                  root: str, parents: dict[str, str], device: str, inputs: dict,
                  anchor_logits: Mapping[str, torch.Tensor], epochs: int = 1, lr: float = PATH_LR,
                  logical: int = PATH_LOGICAL, micro: int = PATH_MICRO, log=print) -> dict:
    """Matched QT continuation from the same T_QT state and target table.

    The function intentionally consumes exactly the same ``items[q][targets]``
    and order schedule as :func:`train_path`, while never reading bags or
    support records.  It is therefore a real control, not a renamed T_QT run.
    """
    runlog.enforce_precision()
    stage = "T_QT_CONT"
    model = model.to(device)
    params = path_trainable(model)
    optimizer = adamw(params, lr)
    order_all = [q for q in utf8_sorted(items) if items[q]["active"]]
    total_updates = math.ceil(len(order_all) / logical) * epochs
    snapshots = {max(1, math.ceil(total_updates * f)): tag for f, tag in ((0.5, "half"), (1.0, "end"))}
    counters = {"updates": 0, "queries_consumed": 0, "loss_target": 0.0, "loss_anchor": 0.0}
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={**inputs, "active_queries": len(order_all), "planned_updates": total_updates,
                           "query_order_sha256": sha256_json(matched_final_order(order_all, root=root, seed=seed, epoch=1))},
                   initial_state_sha=runlog.state_sha(model.state_dict()), optimizer_state="empty_AdamW",
                   config={"epochs": epochs, "lr": lr, "logical_queries": logical, "micro_queries": micro,
                           "loss": "target_CE_plus_same_f0_KL_only", "f0_KL_anchor_weight": 0.3,
                           "uses_support_records": False, "uses_natural_bags": False})
    runlog.save_checkpoint(stage_dir / "init.pt", model=model, optimizer=optimizer, stage=stage, seed=seed,
                           paths=paths, parents=parents, counters=counters)
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = matched_final_order(order_all, root=root, seed=seed, epoch=epoch)
        for start in range(0, len(order), logical):
            batch = order[start:start + logical]
            optimizer.zero_grad(set_to_none=True)
            for qid in batch:
                item = items[qid]
                targets = item["targets"]
                logits = pair_logits(model, bank, qid, targets, device)
                positive = torch.tensor([t in set(item["positives"]) for t in targets], dtype=torch.bool, device=device)
                allowed = torch.ones(len(targets), dtype=torch.bool, device=device)
                target_loss = rank_ce(logits, positive, allowed)
                if target_loss is None:
                    continue
                ref = anchor_logits[qid].to(device)
                anchor = kd_loss(logits, ref, allowed, temperature=1.0)
                ((target_loss + 0.3 * anchor) / len(batch)).backward()
                counters["loss_target"] += float(target_loss.detach())
                counters["loss_anchor"] += float(anchor.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            counters["updates"] += 1
            counters["queries_consumed"] += len(batch)
            if counters["updates"] in snapshots:
                tag = snapshots[counters["updates"]]
                sha = runlog.save_checkpoint(stage_dir / f"{tag}.pt", model=model, optimizer=optimizer, stage=stage,
                                             seed=seed, paths=paths, parents=parents, counters=counters,
                                             extra={"snapshot": tag, "epoch": epoch})
                records.append({"snapshot": tag, "updates": counters["updates"], "checkpoint_sha256": sha,
                                "state_sha256": runlog.state_sha(model.state_dict())})
                write_json(stage_dir / "SNAPSHOTS.json", records)
    final = stage_dir / "end.pt"
    link = stage_dir / "checkpoint.pt"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(final.name)
    runlog.post_run(stage_dir, status="COMPLETE", counters=counters,
                    outputs={r["snapshot"] + ".pt": r["checkpoint_sha256"] for r in records},
                    notes={"snapshots": records, "target_table_sha256": inputs.get("target_table_sha256")})
    return {"counters": counters, "snapshots": records}


def load_teacher(path: Path, device: str, *, expect_stage: str | None = None) -> tuple[FreshPathTeacher, dict]:
    payload = runlog.load_checkpoint(path, expect_stage=expect_stage)
    model = FreshPathTeacher(**TEACHER_KWARGS).float()
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload

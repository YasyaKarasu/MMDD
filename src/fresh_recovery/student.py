"""Students: native P/R Student and QT-only Student, C1 list training, C2 graph training.

SPEC 8: three projections initialised from this run's P0, five directed
relations initialised to identity, no bias/adapter/normalisation.  SPEC 9:
C2 keeps the same P/R trainable with separate Direct and Evidence
distributions; every path slot is addressed by explicit (e_index, t_index)
tensors so no grouping can permute scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from . import runlog
from .config import Paths
from .data import local_rng, utf8_sorted
from .io import write_json
from .losses import kd_loss, rank_ce, student_anchor, sup_transform

KINDS = ("table", "text", "image")
RELATION_KINDS = {"QT": ("table", "table"), "Q_text": ("table", "text"), "Q_image": ("table", "image"),
                  "text_T": ("text", "table"), "image_T": ("image", "table")}
LIST_RELATION = {"QT": "QT", "Q_text": "Q_text", "Q_image": "Q_image", "text_T": "text_T", "image_T": "image_T"}
P_LR = 1e-6
R_LR = 1e-5
ANCHOR_WEIGHT = 0.1
KD_WEIGHT = 0.3
KD_TEMPERATURE = 1.0
C1_LOGICAL = 64
C1_MICRO = 16
C1_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
C2_LOGICAL = 64
C2_MICRO = 8
# C2 is capped at two epochs and selection evaluates all four cumulative
# fractions required by v3.1.  The paired arms consume the same snapshot
# positions; there is no early stopping or extra grid.
C2_FRACTIONS = (0.5, 1.0, 1.5, 2.0)


class NativeStudent(nn.Module):
    def __init__(self, basis: Tensor) -> None:
        super().__init__()
        dim = basis.shape[0]
        self.dim = dim
        self.projections = nn.ParameterDict({k: nn.Parameter(basis.detach().clone().float()) for k in KINDS})
        self.relations = nn.ParameterDict({r: nn.Parameter(torch.eye(dim)) for r in RELATION_KINDS})
        self.register_buffer("basis", basis.detach().clone().float(), persistent=False)

    def u(self, kind: str, z: Tensor) -> Tensor:
        return z @ self.projections[kind].T

    def query_vector(self, relation: str, z_anchor: Tensor) -> Tensor:
        kind_a, _ = RELATION_KINDS[relation]
        return self.u(kind_a, z_anchor) @ self.relations[relation]

    def score(self, relation: str, z_anchor: Tensor, z_dest: Tensor) -> Tensor:
        _, kind_b = RELATION_KINDS[relation]
        return self.query_vector(relation, z_anchor) @ self.u(kind_b, z_dest).T

    def anchor(self) -> Tensor:
        return student_anchor(dict(self.projections.items()), dict(self.relations.items()), self.basis)

    def param_groups(self) -> list[dict]:
        return [{"params": list(self.projections.parameters()), "lr": P_LR, "name": "P"},
                {"params": list(self.relations.parameters()), "lr": R_LR, "name": "R"}]


class QTStudent(nn.Module):
    """QT-only control: P_table and R_QT only (SPEC 6.3, 9.4)."""

    def __init__(self, basis: Tensor) -> None:
        super().__init__()
        dim = basis.shape[0]
        self.dim = dim
        self.projections = nn.ParameterDict({"table": nn.Parameter(basis.detach().clone().float())})
        self.relations = nn.ParameterDict({"QT": nn.Parameter(torch.eye(dim))})
        self.register_buffer("basis", basis.detach().clone().float(), persistent=False)

    def u(self, kind: str, z: Tensor) -> Tensor:
        if kind != "table":
            raise ValueError("QT-only Student has no evidence projection")
        return z @ self.projections["table"].T

    def query_vector(self, relation: str, z_anchor: Tensor) -> Tensor:
        if relation != "QT":
            raise ValueError("QT-only Student has only R_QT")
        return self.u("table", z_anchor) @ self.relations["QT"]

    def score(self, relation: str, z_anchor: Tensor, z_dest: Tensor) -> Tensor:
        return self.query_vector(relation, z_anchor) @ self.u("table", z_dest).T

    def anchor(self) -> Tensor:
        return student_anchor(dict(self.projections.items()), dict(self.relations.items()), self.basis)

    def param_groups(self) -> list[dict]:
        return [{"params": list(self.projections.parameters()), "lr": P_LR, "name": "P"},
                {"params": list(self.relations.parameters()), "lr": R_LR, "name": "R"}]


def make_student(basis: Tensor, *, qt_only: bool) -> nn.Module:
    return QTStudent(basis) if qt_only else NativeStudent(basis)


def load_student(path: Path, basis: Tensor, device: str, *, expect_stage: str | None = None):
    payload = runlog.load_checkpoint(path, expect_stage=expect_stage)
    qt_only = bool(payload.get("qt_only", False))
    model = make_student(basis, qt_only=qt_only)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


def adamw(model) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.param_groups(), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def list_relation(row: dict) -> str:
    return LIST_RELATION[row["relation"]]


# ------------------------------------------------------------------- C1 ----


def c1_list_loss(model, bank, row: dict, device, *, kd: bool) -> tuple[Tensor | None, dict]:
    relation = list_relation(row)
    z_anchor = bank.z(row["anchor_id"]).to(device)
    z_dest = bank.z_many(row["candidates"]).to(device)
    raw = model.score(relation, z_anchor, z_dest)
    pset = set(row["positives"])
    positive = torch.tensor([c in pset for c in row["candidates"]], dtype=torch.bool, device=device)
    allowed = torch.tensor([c not in set(row.get("ignore", ())) for c in row["candidates"]], dtype=torch.bool, device=device)
    sup = rank_ce(sup_transform(raw), positive, allowed)
    if sup is None:
        return None, {}
    parts = {"sup": float(sup.detach())}
    loss = sup
    if kd:
        teacher = torch.tensor(row["teacher_logits"], dtype=torch.float32, device=device)
        if teacher.shape != raw.shape:
            raise ValueError(f"{row['item_id']}: teacher logits misaligned")
        kd_term = kd_loss(raw, teacher, allowed, temperature=KD_TEMPERATURE)
        loss = loss + KD_WEIGHT * kd_term
        parts["kd"] = float(kd_term.detach())
    return loss, parts


def snapshot_steps(total_updates: int, fractions: Sequence[float]) -> dict[int, str]:
    out: dict[int, str] = {}
    for f in fractions:
        step = max(1, math.ceil(total_updates * f))
        out.setdefault(step, f"frac{int(round(f * 100)):03d}")
    return out


def _link(stage_dir: Path, name: str, target: str) -> None:
    link = stage_dir / name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def train_c1(model, bank, lists: Mapping[str, dict], *, paths: Paths, seed: int, stage: str, stage_dir: Path,
             root: str, scope: str, kd: bool, parents: dict[str, str], device: str, inputs: dict,
             snapshot_hook: Callable[[str, nn.Module], dict] | None = None, epochs: int = 1,
             logical: int = C1_LOGICAL, micro: int = C1_MICRO, log=print) -> dict:
    """SPEC 8.2-8.3: 64 edge lists per update, SUP 10·sigmoid CE (+0.3 raw KD), anchor once."""
    runlog.enforce_precision()
    model = model.to(device)
    optimizer = adamw(model)
    active = [k for k, row in lists.items() if row["active"]]
    total_updates = math.ceil(len(active) / logical) * epochs
    snapshots = snapshot_steps(total_updates, C1_FRACTIONS)
    counters = {"updates": 0, "lists_consumed": 0, "loss_sup": 0.0, "loss_kd": 0.0, "anchor": 0.0,
                "by_relation": {}}
    init_sha = runlog.state_sha(model.state_dict())
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={**inputs, "active_lists": len(active), "total_lists": len(lists),
                           "planned_updates": total_updates, "snapshots": snapshots},
                   initial_state_sha=init_sha, optimizer_state="empty_AdamW",
                   config={"epochs": epochs, "logical_lists": logical, "micro_lists": micro, "P_lr": P_LR, "R_lr": R_LR,
                           "sup": "CE(10*sigmoid(raw))", "kd": f"{KD_WEIGHT}*KL(softmax(T)||softmax(S)) tau={KD_TEMPERATURE} raw" if kd else None,
                           "anchor_weight": ANCHOR_WEIGHT, "anchor_reference": "same_run_P0_and_identity",
                           "clip_norm": 1.0, "weight_decay": 0.01,
                           "param_groups": [{"name": g["name"], "lr": g["lr"], "tensors": len(g["params"])} for g in optimizer.param_groups]})
    runlog.save_checkpoint(stage_dir / "init.pt", model=model, optimizer=optimizer, stage=stage, seed=seed,
                           paths=paths, parents=parents, counters=counters, extra={"qt_only": isinstance(model, QTStudent)})
    started = time.time()
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = utf8_sorted(active)
        local_rng(root, "C1", seed, epoch, scope).shuffle(order)
        for start in range(0, len(order), logical):
            batch = order[start : start + logical]
            n = len(batch)
            optimizer.zero_grad(set_to_none=True)
            for m_start in range(0, n, micro):
                total = None
                for item_id in batch[m_start : m_start + micro]:
                    row = lists[item_id]
                    loss, parts = c1_list_loss(model, bank, row, device, kd=kd)
                    if loss is None:
                        raise ValueError(f"{item_id}: inactive list reached training")
                    total = loss if total is None else total + loss
                    counters["loss_sup"] += parts["sup"]
                    counters["loss_kd"] += parts.get("kd", 0.0)
                    rel = counters["by_relation"].setdefault(row["relation"], 0)
                    counters["by_relation"][row["relation"]] = rel + 1
                (total / n).backward()
            anchor = model.anchor()
            (ANCHOR_WEIGHT * anchor).backward()
            counters["anchor"] += float(anchor.detach())
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            counters["updates"] += 1
            counters["lists_consumed"] += n
            if counters["updates"] % 20 == 0:
                log({"event": "progress", "stage": stage, "updates": counters["updates"], "of": total_updates,
                     "mean_sup": round(counters["loss_sup"] / counters["lists_consumed"], 4),
                     "mean_kd": round(counters["loss_kd"] / counters["lists_consumed"], 4),
                     "anchor": round(float(anchor.detach()), 6), "elapsed": round(time.time() - started, 1)})
            if counters["updates"] in snapshots:
                tag = snapshots[counters["updates"]]
                sha = runlog.save_checkpoint(stage_dir / f"{tag}.pt", model=model, optimizer=optimizer, stage=stage,
                                             seed=seed, paths=paths, parents=parents, counters=counters,
                                             extra={"snapshot": tag, "epoch": epoch, "qt_only": isinstance(model, QTStudent)})
                record = {"snapshot": tag, "updates": counters["updates"], "checkpoint_sha256": sha,
                          "state_sha256": runlog.state_sha(model.state_dict()),
                          "elapsed": time.time() - started, "counters": json.loads(json.dumps(counters))}
                if snapshot_hook is not None:
                    model.eval()
                    record["dev"] = snapshot_hook(tag, model)
                    model.train()
                records.append(record)
                write_json(stage_dir / "SNAPSHOTS.json", records)
                log({"event": "snapshot", "stage": stage, "tag": tag, "updates": counters["updates"],
                     "dev": record.get("dev", {}).get("summary")})
    _link(stage_dir, "checkpoint.pt", "frac100.pt")
    runlog.post_run(stage_dir, status="COMPLETE", counters=json.loads(json.dumps(counters)),
                    outputs={r["snapshot"] + ".pt": r["checkpoint_sha256"] for r in records},
                    notes={"snapshots": records, "final_state_sha256": runlog.state_sha(model.state_dict())})
    return {"counters": counters, "snapshots": records}


# ------------------------------------------------------------------- C2 ----


def c2_query_terms(model, bank, item: dict, device, *, kd: bool, teacher: dict | None) -> dict:
    """One query's D/E student terms (SPEC 9.2-9.3).

    ``item``: targets, d_positive, evidence [e ids], views [{"e_index","t_index","counts",
    "e_positive","e_allowed","active"}].  ``teacher``: {"D": Tensor, "QE": Tensor, "ET": {view: Tensor}}.
    """
    targets = item["targets"]
    z_q = bank.z(item["query_id"]).to(device)
    z_t = bank.z_many(targets).to(device)
    u_q = model.u("table", z_q)
    u_t = model.u("table", z_t)
    d_student = (u_q @ model.relations["QT"]) @ u_t.T
    d_pos = torch.tensor(item["d_positive"], dtype=torch.bool, device=device)
    d_allowed = torch.ones(len(targets), dtype=torch.bool, device=device)
    out = {"d_sup": rank_ce(d_student, d_pos, d_allowed), "d_kd": None, "e_terms": []}
    if out["d_sup"] is None:
        raise ValueError(f"{item['query_id']}: inactive Direct list reached C2")
    if kd:
        out["d_kd"] = kd_loss(d_student, teacher["D"].to(device), d_allowed, temperature=KD_TEMPERATURE)
    evidence = item["evidence"]
    if not evidence or not any(v["active"] for v in item["views"]):
        return out
    first = torch.empty(len(evidence), device=device)
    second = torch.empty(len(evidence), len(targets), device=device)
    for kind in ("text", "image"):
        idx = [i for i, e in enumerate(evidence) if bank.kind(e) == kind]
        if not idx:
            continue
        z_e = bank.z_many([evidence[i] for i in idx]).to(device)
        u_e = model.u(kind, z_e)
        rows = torch.tensor(idx, dtype=torch.long, device=device)
        first[rows] = (u_q @ model.relations[f"Q_{kind}"]) @ u_e.T
        second[rows] = (u_e @ model.relations[f"{kind}_T"]) @ u_t.T
    for view_no, view in enumerate(item["views"]):
        if not view["active"]:
            continue
        e_index = torch.tensor(view["e_index"], dtype=torch.long, device=device)
        t_index = torch.tensor(view["t_index"], dtype=torch.long, device=device)
        flat = first[e_index] + second[e_index, t_index]
        e_student = segment_lse(flat, t_index, len(targets), view["counts"])
        e_pos = torch.tensor(view["e_positive"], dtype=torch.bool, device=device)
        e_allowed = torch.tensor(view["e_allowed"], dtype=torch.bool, device=device)
        sup = rank_ce(e_student, e_pos, e_allowed)
        if sup is None:
            raise ValueError(f"{item['query_id']}: view {view_no} marked active but inactive")
        term = {"sup": sup, "kd": None}
        if kd:
            t_flat = teacher["ET"][view_no].to(device)
            e_teacher = segment_lse(t_flat, t_index, len(targets), view["counts"])
            term["kd"] = kd_loss(e_student, e_teacher, e_allowed, temperature=KD_TEMPERATURE)
        out["e_terms"].append(term)
    return out


def segment_lse(flat: Tensor, t_index: Tensor, n_targets: int, counts: Sequence[int]) -> Tensor:
    """LSE of the flat path scores grouped by their explicit target index."""
    width = max(counts) if counts else 0
    if width == 0:
        return torch.full((n_targets,), float("-inf"), device=flat.device)
    padded = torch.full((n_targets, width), float("-inf"), device=flat.device, dtype=flat.dtype)
    offsets = torch.zeros(n_targets, dtype=torch.long, device=flat.device)
    counts_t = torch.tensor(counts, dtype=torch.long, device=flat.device)
    starts = torch.cumsum(counts_t, 0) - counts_t
    slot_in_target = torch.arange(flat.numel(), device=flat.device) - starts[t_index]
    padded[t_index, slot_in_target] = flat
    del offsets
    has = counts_t > 0
    out = torch.full((n_targets,), float("-inf"), device=flat.device, dtype=flat.dtype)
    out[has] = torch.logsumexp(padded[has], dim=1)
    return out


def train_c2(model, bank, items: Mapping[str, dict], *, paths: Paths, seed: int, stage: str, stage_dir: Path,
             root: str, scope: str, kd: bool, teacher_logits: Mapping[str, dict] | None, parents: dict[str, str],
             device: str, inputs: dict, snapshot_hook: Callable[[str, nn.Module], dict] | None = None,
             epochs: int = 2, logical: int = C2_LOGICAL, micro: int = C2_MICRO, log=print) -> dict:
    """SPEC 9.3 / 9.4: mean_q CE(D) + mean_{valid q} mean_views CE(E) + 0.1 A (+0.3 KD terms)."""
    runlog.enforce_precision()
    model = model.to(device)
    optimizer = adamw(model)
    order_all = [q for q in utf8_sorted(items) if items[q]["active"]]
    # C2 fractions are cumulative positions across the two-epoch run: 0.5
    # and 1.0 are the first epoch checkpoints, while 1.5 and 2.0 are in the
    # second epoch.  Compute them from one epoch's update count so all four
    # required positions are reachable.
    epoch_updates = math.ceil(len(order_all) / logical)
    total_updates = epoch_updates * epochs
    snapshots = snapshot_steps(epoch_updates, C2_FRACTIONS)
    counters = {"updates": 0, "queries_consumed": 0, "e_queries": 0, "loss_d_sup": 0.0, "loss_d_kd": 0.0,
                "loss_e_sup": 0.0, "loss_e_kd": 0.0, "anchor": 0.0}
    init_sha = runlog.state_sha(model.state_dict())
    runlog.pre_run(stage_dir, stage=stage, seed=seed, paths=paths, parents=parents,
                   inputs={**inputs, "active_queries": len(order_all), "queries": len(items),
                           "planned_updates": total_updates, "snapshots": snapshots,
                           "e_valid_queries": sum(1 for q in order_all if any(v["active"] for v in items[q]["views"]))},
                   initial_state_sha=init_sha, optimizer_state="empty_AdamW",
                   config={"epochs": epochs, "logical_queries": logical, "micro_queries": micro, "P_lr": P_LR, "R_lr": R_LR,
                           "loss": "mean_q CE(D) + mean_{valid q} mean_views CE(E) + 0.1 A" + (" + 0.3[KL(D)+KL(E)]" if kd else ""),
                           "temperature": KD_TEMPERATURE, "anchor_reference": "same_run_P0_and_identity (not reset)",
                           "clip_norm": 1.0, "weight_decay": 0.01})
    runlog.save_checkpoint(stage_dir / "init.pt", model=model, optimizer=optimizer, stage=stage, seed=seed,
                           paths=paths, parents=parents, counters=counters, extra={"qt_only": isinstance(model, QTStudent)})
    started = time.time()
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = utf8_sorted(order_all)
        local_rng(root, "C2", seed, epoch, scope).shuffle(order)
        for start in range(0, len(order), logical):
            batch = order[start : start + logical]
            n = len(batch)
            n_e = sum(1 for q in batch if any(v["active"] for v in items[q]["views"]))
            optimizer.zero_grad(set_to_none=True)
            for m_start in range(0, n, micro):
                total = None
                for q in batch[m_start : m_start + micro]:
                    terms = c2_query_terms(model, bank, items[q], device, kd=kd,
                                           teacher=teacher_logits[q] if kd else None)
                    part = terms["d_sup"] / n
                    counters["loss_d_sup"] += float(terms["d_sup"].detach())
                    if kd:
                        part = part + KD_WEIGHT * terms["d_kd"] / n
                        counters["loss_d_kd"] += float(terms["d_kd"].detach())
                    if terms["e_terms"]:
                        views = torch.stack([t["sup"] for t in terms["e_terms"]]).mean()
                        part = part + views / n_e
                        counters["loss_e_sup"] += float(views.detach())
                        counters["e_queries"] += 1
                        if kd:
                            kd_views = torch.stack([t["kd"] for t in terms["e_terms"]]).mean()
                            part = part + KD_WEIGHT * kd_views / n_e
                            counters["loss_e_kd"] += float(kd_views.detach())
                    total = part if total is None else total + part
                total.backward()
            anchor = model.anchor()
            (ANCHOR_WEIGHT * anchor).backward()
            counters["anchor"] += float(anchor.detach())
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            counters["updates"] += 1
            counters["queries_consumed"] += n
            if counters["updates"] % 10 == 0:
                log({"event": "progress", "stage": stage, "updates": counters["updates"], "of": total_updates,
                     "mean_d_sup": round(counters["loss_d_sup"] / counters["queries_consumed"], 4),
                     "mean_e_sup": round(counters["loss_e_sup"] / max(counters["e_queries"], 1), 4),
                     "mean_d_kd": round(counters["loss_d_kd"] / counters["queries_consumed"], 4),
                     "mean_e_kd": round(counters["loss_e_kd"] / max(counters["e_queries"], 1), 4),
                     "anchor": round(float(anchor.detach()), 6), "elapsed": round(time.time() - started, 1)})
            if counters["updates"] in snapshots:
                tag = snapshots[counters["updates"]]
                sha = runlog.save_checkpoint(stage_dir / f"{tag}.pt", model=model, optimizer=optimizer, stage=stage,
                                             seed=seed, paths=paths, parents=parents, counters=counters,
                                             extra={"snapshot": tag, "epoch": epoch, "qt_only": isinstance(model, QTStudent)})
                record = {"snapshot": tag, "updates": counters["updates"], "checkpoint_sha256": sha,
                          "state_sha256": runlog.state_sha(model.state_dict()),
                          "elapsed": time.time() - started, "counters": json.loads(json.dumps(counters))}
                if snapshot_hook is not None:
                    model.eval()
                    record["dev"] = snapshot_hook(tag, model)
                    model.train()
                records.append(record)
                write_json(stage_dir / "SNAPSHOTS.json", records)
                log({"event": "snapshot", "stage": stage, "tag": tag, "updates": counters["updates"],
                     "dev": record.get("dev", {}).get("summary")})
    _link(stage_dir, "checkpoint.pt", "frac100.pt")
    runlog.post_run(stage_dir, status="COMPLETE", counters=json.loads(json.dumps(counters)),
                    outputs={r["snapshot"] + ".pt": r["checkpoint_sha256"] for r in records},
                    notes={"snapshots": records, "final_state_sha256": runlog.state_sha(model.state_dict())})
    return {"counters": counters, "snapshots": records}


# --------------------------------------------------------- C2 graph items ----


def build_c2_item(query_id: str, *, targets: Sequence[str], gold: Sequence[str], witness: Mapping[str, Sequence[str]],
                  natural_bags: Mapping[str, Sequence[str]], augmented_e: str | None) -> dict:
    """Model-independent C2 item: fixed target order, explicit slot indices, view masks."""
    targets = list(targets)
    t_pos = {t: i for i, t in enumerate(targets)}
    g = set(gold)
    views_bags: list[Mapping[str, Sequence[str]]] = [natural_bags]
    if augmented_e is not None:
        views_bags.append({t: list(dict.fromkeys([*natural_bags.get(t, ()), augmented_e])) for t in targets})
    evidence = utf8_sorted({e for bags in views_bags for lst in bags.values() for e in lst})
    e_pos = {e: i for i, e in enumerate(evidence)}
    views = []
    for bags in views_bags:
        e_index, t_index, counts = [], [], []
        positive, allowed = [], []
        for t in targets:
            bag = list(bags.get(t, ()))
            if len(bag) != len(set(bag)):
                raise ValueError(f"{query_id}/{t}: duplicate evidence in bag")
            counts.append(len(bag))
            for e in bag:
                e_index.append(e_pos[e])
                t_index.append(t_pos[t])
            supported = bool(set(bag) & set(witness.get(t, ())))
            has_path = bool(bag)
            positive.append(has_path and t in g and supported)
            allowed.append(has_path and (t not in g or supported))
        active = any(positive) and any(a and not p for a, p in zip(allowed, positive))
        views.append({"e_index": e_index, "t_index": t_index, "counts": counts,
                      "e_positive": positive, "e_allowed": allowed, "active": active,
                      "bags": {t: list(bags.get(t, ())) for t in targets if bags.get(t)}})
    d_positive = [t in g for t in targets]
    d_active = any(d_positive) and any(not p for p in d_positive)
    return {"query_id": query_id, "targets": targets, "d_positive": d_positive, "evidence": evidence,
            "views": views, "augmented_e": augmented_e, "active": d_active,
            "sha256": hashlib.sha256(json.dumps({"t": targets, "e": evidence, "v": [(v["e_index"], v["t_index"]) for v in views]}).encode()).hexdigest()}

"""Training loops for Teacher (T_A, T_B) and Student (C1, C2) for CLEAN-QET v4.0."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from .config import Paths
from .data import utf8_sorted, write_json
from .features import ObjectBank, ZStore
from .labels import Labels
from .losses import (
    aggregate_paths,
    hierarchical_support_mean,
    list_kl_divergence,
    positive_average_pair_loss,
    rank_mass_loss,
)
from .models import FreshPathTeacher, NativeStudent, QTStudent


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    extra: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ T_A Train ---


def train_ta(
    model: FreshPathTeacher,
    bank: ObjectBank,
    ta_records: list[dict],
    labels: Labels,
    device: str = "cuda:0",
    epochs: int = 2,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    logical_batch: int = 8,
    support_weight: float = 0.2,
    save_dir: Optional[Path] = None,
    log_interval: int = 50,
) -> Path:
    dev = torch.device(device)
    model.to(dev)
    bank.attach_device(dev)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

    if save_dir:
        save_checkpoint(save_dir / "init.pt", model, optimizer, {"stage": "T_A", "step": 0})

    total_queries = len(ta_records)
    total_steps = math.ceil(total_queries / logical_batch) * epochs
    updates = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        queries_done = 0
        optimizer.zero_grad(set_to_none=True)

        for start in range(0, total_queries, logical_batch):
            batch = ta_records[start : start + logical_batch]
            n_batch = len(batch)
            batch_loss = 0.0

            for record in batch:
                qid = record["query_id"]
                zq = bank.z(qid)
                cq = bank.tokens(qid)
                cache: dict = {}

                relation_losses: list[Tensor] = []

                # 1. QT Loss
                qt_cands = record["qt_candidates"]
                pos_qt = set(labels.queries[qid]["G"])
                if pos_qt and any(c not in pos_qt for c in qt_cands):
                    pairs = [("table", zq, cq, "table", bank.z(t), bank.tokens(t)) for t in qt_cands]
                    keys = [((qid, 0), (t, 1)) for t in qt_cands]
                    scores = model.score_pairs(pairs, cache=cache, cache_keys=keys)
                    mask = torch.tensor([t in pos_qt for t in qt_cands], dtype=torch.bool, device=dev)
                    l_qt = rank_mass_loss(scores, mask)
                    if l_qt is not None:
                        relation_losses.append(l_qt)

                # 2. QE Loss (text and image)
                for m in ("text", "image"):
                    qe_cands = record["qe_candidates"].get(m, [])
                    pos_qe = set(labels.queries[qid]["Qpos"].get(m, []))
                    if pos_qe and any(c not in pos_qe for c in qe_cands):
                        pairs = [("table", zq, cq, m, bank.z(e), bank.tokens(e)) for e in qe_cands]
                        keys = [((qid, 0), (e, 1)) for e in qe_cands]
                        scores = model.score_pairs(pairs, cache=cache, cache_keys=keys)
                        mask = torch.tensor([e in pos_qe for e in qe_cands], dtype=torch.bool, device=dev)
                        l_qe = rank_mass_loss(scores, mask)
                        if l_qe is not None:
                            relation_losses.append(l_qe)

                # 3. Conditional QET Loss
                for qet_item in record.get("qet_lists", []):
                    eid = qet_item["evidence_id"]
                    ekind = qet_item["evidence_kind"]
                    cands = qet_item["candidates"]
                    pos_qet = set(qet_item["positives"])
                    ign_qet = set(qet_item.get("ignore", []))
                    valid_cands = [t for t in cands if t not in ign_qet]
                    if pos_qet and any(c not in pos_qet for c in valid_cands):
                        ze = bank.z(eid)
                        ce = bank.tokens(eid)
                        triplets = [("table", zq, cq, ekind, ze, ce, "table", bank.z(t), bank.tokens(t)) for t in valid_cands]
                        keys = [((qid, 0), (eid, 2), (t, 1)) for t in valid_cands]
                        scores = model.score_triplets(triplets, cache=cache, cache_keys=keys)
                        mask = torch.tensor([t in pos_qet for t in valid_cands], dtype=torch.bool, device=dev)
                        l_qet = rank_mass_loss(scores, mask)
                        if l_qet is not None:
                            relation_losses.append(l_qet)

                # 4. Support Witness Content Loss A(q)
                target_modality_losses: list[list[Tensor]] = []
                for sup in record.get("support_records", []):
                    tid = sup["target_id"]
                    m = sup["modality"]
                    pos_e = sup["positives"]
                    comp_e = sup["competitors"]
                    if not pos_e or not comp_e:
                        continue
                    zt = bank.z(tid)
                    ct = bank.tokens(tid)
                    triplets_pos = [("table", zq, cq, m, bank.z(e), bank.tokens(e), "table", zt, ct) for e in pos_e]
                    triplets_comp = [("table", zq, cq, m, bank.z(e), bank.tokens(e), "table", zt, ct) for e in comp_e]
                    keys_pos = [((qid, 0), (e, 2), (tid, 1)) for e in pos_e]
                    keys_comp = [((qid, 0), (e, 2), (tid, 1)) for e in comp_e]
                    pos_scores = model.score_triplets(triplets_pos, cache=cache, cache_keys=keys_pos)
                    comp_scores = model.score_triplets(triplets_comp, cache=cache, cache_keys=keys_comp)
                    mod_loss = positive_average_pair_loss(pos_scores, comp_scores)
                    if mod_loss is not None:
                        target_modality_losses.append([mod_loss])

                a_loss = hierarchical_support_mean(target_modality_losses)

                # L_A(q) = Mean(relations) + 0.2 * A(q)
                loss_q = None
                if relation_losses:
                    loss_q = torch.stack(relation_losses).mean()
                if a_loss is not None:
                    loss_q = a_loss * support_weight if loss_q is None else loss_q + support_weight * a_loss

                if loss_q is not None:
                    (loss_q / n_batch).backward()
                    batch_loss += float(loss_q.detach())

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            queries_done += n_batch
            epoch_loss += batch_loss
            elapsed = time.time() - t_start
            eta = (elapsed / max(updates, 1)) * (total_steps - updates)
            step_loss = batch_loss
            print(f"[T_A] Epoch {epoch}/{epochs} Step {updates}/{total_steps} loss={step_loss:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

            if save_dir:
                prog = {
                    "stage": "T_A",
                    "epoch": epoch,
                    "step": updates,
                    "total_steps": total_steps,
                    "queries_done": queries_done,
                    "loss": round(step_loss, 5),
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(eta, 1),
                    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                tmp_p = save_dir / "PROGRESS.json.tmp"
                with open(tmp_p, "w", encoding="utf-8") as f:
                    json.dump(prog, f, indent=2)
                tmp_p.replace(save_dir / "PROGRESS.json")

        if save_dir:
            save_checkpoint(save_dir / f"epoch{epoch}.pt", model, optimizer, {"stage": "T_A", "epoch": epoch, "loss": epoch_loss / max(queries_done, 1)})

    final_ckpt = save_dir / f"epoch{epochs}.pt" if save_dir else Path("epoch2.pt")
    return final_ckpt


# ------------------------------------------------------------------ T_B Train ---


def train_tb(
    model: FreshPathTeacher,
    bank: ObjectBank,
    tb_records: list[dict],
    device: str = "cuda:0",
    mode: str = "path",  # "path" or "qt"
    epochs: int = 1,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    logical_batch: int = 8,
    save_dir: Optional[Path] = None,
) -> Path:
    dev = torch.device(device)
    model.to(dev)
    bank.attach_device(dev)

    trainable_params = model.set_tb_trainable()
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

    if save_dir:
        save_checkpoint(save_dir / "init.pt", model, optimizer, {"stage": f"T_B_{mode.upper()}", "step": 0})

    total_queries = len(tb_records)
    total_steps = math.ceil(total_queries / logical_batch) * epochs
    half_step = total_steps // 2
    step = 0

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for start in range(0, total_queries, logical_batch):
            batch = tb_records[start : start + logical_batch]
            n_batch = len(batch)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            for record in batch:
                qid = record["query_id"]
                zq = bank.z(qid)
                targets = record["targets"]
                positives = set(record["positives"])
                pos_mask = torch.tensor([t in positives for t in targets], dtype=torch.bool, device=dev)

                if mode == "qt":
                    # T_B_QT: L_D only
                    all_ids = list(dict.fromkeys([qid] + targets))
                    tok_map = dict(zip(all_ids, bank.tokens_many(all_ids)))
                    cq = tok_map[qid]
                    ct_list = [tok_map[t] for t in targets]
                    zt_matrix = bank.z_many(targets)
                    f0_scores, _ = model.score_query_lists((zq, cq), (zt_matrix, ct_list), {}, [], chunk=1024)
                    l_d = rank_mass_loss(f0_scores, pos_mask)
                    total_q = l_d
                else:
                    # T_B_PATH: 0.5 L_D + 0.5 L_F + 0.2 A
                    paths_ = [(k, e) for k, t in enumerate(targets) for e in record["natural_bags"].get(t, [])]
                    e_ids = list(dict.fromkeys(e for _, e in paths_))
                    all_ids = list(dict.fromkeys([qid] + targets + e_ids))
                    tok_map = dict(zip(all_ids, bank.tokens_many(all_ids)))
                    cq = tok_map[qid]
                    ct_list = [tok_map[t] for t in targets]
                    zt_matrix = bank.z_many(targets)
                    ev_map = {e: (bank.kind(e), bank.z(e), tok_map[e]) for e in e_ids}
                    f0_scores, path_scores = model.score_query_lists((zq, cq), (zt_matrix, ct_list), ev_map, paths_, chunk=1024)
                    l_d = rank_mass_loss(f0_scores, pos_mask)

                    if paths_:
                        target_idx_t = torch.tensor([k for k, _ in paths_], dtype=torch.long, device=dev)
                        agg_f = aggregate_paths(f0_scores, path_scores, target_idx_t)
                    else:
                        agg_f = f0_scores
                    l_f = rank_mass_loss(agg_f, pos_mask)

                    # Support witness loss A(q)
                    target_modality_losses: list[list[Tensor]] = []
                    cache: dict = {}
                    for sup in record.get("support_records", []):
                        tid = sup["target_id"]
                        m = sup["modality"]
                        pos_e = sup["positives"]
                        comp_e = sup["competitors"]
                        if not pos_e or not comp_e:
                            continue
                        zt = bank.z(tid)
                        ct = bank.tokens(tid)
                        triplets_pos = [("table", zq, cq, m, bank.z(e), bank.tokens(e), "table", zt, ct) for e in pos_e]
                        triplets_comp = [("table", zq, cq, m, bank.z(e), bank.tokens(e), "table", zt, ct) for e in comp_e]
                        keys_pos = [((qid, 0), (e, 2), (tid, 1)) for e in pos_e]
                        keys_comp = [((qid, 0), (e, 2), (tid, 1)) for e in comp_e]
                        pos_s = model.score_triplets(triplets_pos, cache=cache, cache_keys=keys_pos)
                        comp_s = model.score_triplets(triplets_comp, cache=cache, cache_keys=keys_comp)
                        m_loss = positive_average_pair_loss(pos_s, comp_s)
                        if m_loss is not None:
                            target_modality_losses.append([m_loss])

                    l_a = hierarchical_support_mean(target_modality_losses)

                    total_q = None
                    if l_d is not None and l_f is not None:
                        total_q = 0.5 * l_d + 0.5 * l_f
                    elif l_d is not None:
                        total_q = l_d
                    elif l_f is not None:
                        total_q = l_f

                    if l_a is not None:
                        total_q = 0.2 * l_a if total_q is None else total_q + 0.2 * l_a

                if total_q is not None:
                    (total_q / n_batch).backward()
                    batch_loss += float(total_q.detach())

            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            step += 1
            elapsed = time.time() - t_start
            eta = (elapsed / max(step, 1)) * (total_steps - step)
            step_loss = batch_loss
            print(f"[T_B_{mode.upper()}] Step {step}/{total_steps} loss={step_loss:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

            if save_dir:
                prog = {
                    "stage": f"T_B_{mode.upper()}",
                    "step": step,
                    "total_steps": total_steps,
                    "queries_done": min(start + logical_batch, total_queries) * epoch,
                    "loss": round(step_loss, 5),
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(eta, 1),
                    "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                tmp_p = save_dir / "PROGRESS.json.tmp"
                with open(tmp_p, "w", encoding="utf-8") as f:
                    json.dump(prog, f, indent=2)
                tmp_p.replace(save_dir / "PROGRESS.json")

            if save_dir and step == half_step:
                save_checkpoint(save_dir / "half.pt", model, optimizer, {"stage": f"T_B_{mode.upper()}", "step": step})

    final_ckpt = save_dir / "end.pt" if save_dir else Path("end.pt")
    if save_dir:
        save_checkpoint(final_ckpt, model, optimizer, {"stage": f"T_B_{mode.upper()}", "step": step})
    return final_ckpt


# ------------------------------------------------------------- Student Train ---


def train_student_c1(
    student: NativeStudent | QTStudent,
    edge_lists: list[dict],
    teacher: Optional[FreshPathTeacher],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_KD",
    logical_batch: int = 64,
    lr_p: float = 1e-6,
    lr_r: float = 1e-5,
    save_dir: Optional[Path] = None,
) -> dict[float, Path]:
    dev = torch.device(device)
    student.to(dev)
    if teacher is not None:
        teacher.to(dev)
        teacher.eval()

    optimizer = AdamW(student.param_groups(lr_p, lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    total_lists = len(edge_lists)
    total_steps = math.ceil(total_lists / logical_batch)
    checkpoints_fractions = [0.25, 0.5, 0.75, 1.0]
    checkpoint_steps = {math.ceil(total_steps * f): f for f in checkpoints_fractions}
    saved_ckpts: dict[float, Path] = {}

    is_kd = "KD" in arm
    is_qt_only = isinstance(student, QTStudent)

    step = 0
    t_start = time.time()
    student.train()

    for start in range(0, total_lists, logical_batch):
        batch = edge_lists[start : start + logical_batch]
        n_batch = len(batch)
        optimizer.zero_grad(set_to_none=True)

        batch_loss = 0.0
        for item in batch:
            rel = item["relation"]
            anchor = item["anchor_id"]
            positives = set(item["positives"])
            cands = item["candidates"]

            if not positives or not cands:
                continue

            za = bank.z(anchor)
            a_kind = "table" if rel in ("QT", "Q_text", "Q_image") else ("text" if rel == "text_T" else "image")
            b_kind = "table" if rel in ("QT", "text_T", "image_T") else ("text" if rel == "Q_text" else "image")

            # Compute Student scores
            if is_qt_only:
                if rel != "QT":
                    continue
                zb_matrix = bank.z_many(cands)
                scores_s = (student.u(za) @ student.R_QT * student.u(zb_matrix)).sum(dim=-1)
            else:
                zb_matrix = bank.z_many(cands)
                ua = student.u(a_kind, za)
                ub = student.u(b_kind, zb_matrix)
                r_matrix = student.R[rel]
                scores_s = (ua @ r_matrix * ub).sum(dim=-1)

            pos_mask = torch.tensor([c in positives for c in cands], dtype=torch.bool, device=dev)
            # SUP: R(10 * sigmoid(scores_s))
            sup_scores = 10.0 * torch.sigmoid(scores_s)
            l_sup = rank_mass_loss(sup_scores, pos_mask)

            # KD: raw logits KL, temperature=1.0, only for eligible relations QT, Q_text, Q_image
            l_kd = None
            if is_kd and teacher is not None and rel in ("QT", "Q_text", "Q_image"):
                with torch.no_grad():
                    ca = bank.tokens(anchor)
                    pairs = [(a_kind, za, ca, b_kind, bank.z(c), bank.tokens(c)) for c in cands]
                    keys = [((anchor, 0), (c, 1)) for c in cands]
                    scores_t = teacher.score_pairs(pairs, cache_keys=keys)
                l_kd = list_kl_divergence(scores_s, scores_t, temperature=1.0)

            loss_list = None
            if l_sup is not None:
                loss_list = l_sup
            if l_kd is not None:
                loss_list = 0.3 * l_kd if loss_list is None else loss_list + 0.3 * l_kd

            if loss_list is not None:
                anchor_loss = 0.1 * student.anchor_loss()
                total_loss = (loss_list + anchor_loss) / n_batch
                total_loss.backward()
                batch_loss += float(total_loss.detach())

        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step += 1
        elapsed = time.time() - t_start
        eta = (elapsed / max(step, 1)) * (total_steps - step)
        step_loss = batch_loss
        print(f"[{arm}_C1] Step {step}/{total_steps} loss={step_loss:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        if save_dir:
            prog = {
                "stage": f"STUDENT_{arm}_C1",
                "step": step,
                "total_steps": total_steps,
                "queries_done": min(start + logical_batch, total_lists),
                "loss": round(step_loss, 5),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp_p = save_dir / "PROGRESS.json.tmp"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(prog, f, indent=2)
            tmp_p.replace(save_dir / "PROGRESS.json")

        if step in checkpoint_steps and save_dir:
            f = checkpoint_steps[step]
            ckpt_path = save_dir / f"snapshot_frac{int(f*100):03d}.pt"
            save_checkpoint(ckpt_path, student, optimizer, {"stage": "STUDENT_C1", "fraction": f, "step": step})
            saved_ckpts[f] = ckpt_path

    if save_dir and 1.0 not in saved_ckpts:
        ckpt_path = save_dir / "snapshot_frac100.pt"
        save_checkpoint(ckpt_path, student, optimizer, {"stage": "STUDENT_C1", "fraction": 1.0, "step": step})
        saved_ckpts[1.0] = ckpt_path

    return saved_ckpts


def train_student_c2(
    student: NativeStudent | QTStudent,
    c2_records: list[dict],
    teacher: Optional[FreshPathTeacher],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_KD",
    logical_batch: int = 64,
    lr_p: float = 1e-6,
    lr_r: float = 1e-5,
    save_dir: Optional[Path] = None,
) -> dict[float, Path]:
    dev = torch.device(device)
    student.to(dev)
    if teacher is not None:
        teacher.to(dev)
        teacher.eval()

    optimizer = AdamW(student.param_groups(lr_p, lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    total_queries = len(c2_records)
    total_steps = math.ceil(total_queries / logical_batch)
    checkpoints_fractions = [0.25, 0.5, 0.75, 1.0]
    checkpoint_steps = {math.ceil(total_steps * f): f for f in checkpoints_fractions}
    saved_ckpts: dict[float, Path] = {}

    is_kd = "KD" in arm
    is_qt_only = isinstance(student, QTStudent)

    step = 0
    start_step = 0
    t_start = time.time()
    student.train()

    if save_dir:
        for f in [0.75, 0.5, 0.25]:
            ckpt_f = save_dir / f"snapshot_frac{int(f*100):03d}.pt"
            if ckpt_f.exists():
                try:
                    payload = torch.load(ckpt_f, map_location=dev)
                    student.load_state_dict(payload["model"])
                    if payload.get("optimizer") is not None:
                        optimizer.load_state_dict(payload["optimizer"])
                    rng = payload.get("rng")
                    if rng:
                        if rng.get("torch") is not None:
                            torch.set_rng_state(rng["torch"].cpu())
                        if rng.get("cuda") is not None and torch.cuda.is_available():
                            torch.cuda.set_rng_state(rng["cuda"].cpu())
                    step = payload.get("extra", {}).get("step", math.ceil(total_steps * f))
                    start_step = step
                    for prev_f in checkpoints_fractions:
                        prev_p = save_dir / f"snapshot_frac{int(prev_f*100):03d}.pt"
                        if prev_p.exists() and prev_f <= f:
                            saved_ckpts[prev_f] = prev_p
                    print(f"Resuming {arm}_C2 from {ckpt_f.name} at step {step}/{total_steps}", flush=True)
                    break
                except Exception as e:
                    print(f"Warning: could not resume from {ckpt_f}: {e}", flush=True)

    for start in range(start_step * logical_batch, total_queries, logical_batch):
        batch = c2_records[start : start + logical_batch]
        n_batch = len(batch)
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0

        for record in batch:
            qid = record["query_id"]
            zq = bank.z(qid)
            targets = record["targets"]
            positives = set(record["positives"])

            if not targets or not positives:
                continue

            pos_mask = torch.tensor([t in positives for t in targets], dtype=torch.bool, device=dev)
            bags = record.get("natural_bags", {})
            bt = [(k, t) for k, t in enumerate(targets) if bags.get(t)]
            paths = [(row, k, e, slot) for row, (k, t) in enumerate(bt) for slot, e in enumerate(bags[t])]
            e_ids = list(dict.fromkeys(e for _, _, e, _ in paths))

            # 1. Direct Student scores d_S
            zt_matrix = bank.z_many(targets)
            if is_qt_only:
                d_s = (student.u(zq) @ student.R_QT * student.u(zt_matrix)).sum(dim=-1)
            else:
                u_q = student.u("table", zq)
                u_t = student.u("table", zt_matrix)
                d_s = (u_q @ student.R["QT"] * u_t).sum(dim=-1)

            l_dir_sup = rank_mass_loss(d_s, pos_mask)

            # Direct KD and Evidence KD via Teacher
            l_dir_kd = None
            l_ev_kd = None
            e_t = None

            if is_kd and teacher is not None:
                with torch.no_grad():
                    all_ids = list(dict.fromkeys([qid] + targets + e_ids))
                    tok_map = dict(zip(all_ids, bank.tokens_many(all_ids)))
                    cq = tok_map[qid]
                    ct_list = [tok_map[t] for t in targets]
                    ev_map = {e: (bank.kind(e), bank.z(e), tok_map[e]) for e in e_ids}
                    t_paths = [(k, e) for _, k, e, _ in paths]
                    d_t, f_t = teacher.score_query_lists((zq, cq), (zt_matrix, ct_list), ev_map, t_paths, chunk=1024)
                    l_dir_kd = list_kl_divergence(d_s, d_t, temperature=1.0)
                    if bt:
                        max_bag = max(len(bags[t]) for _, t in bt)
                        bag_t = f_t.new_full((len(bt), max_bag), float("-inf"))
                        bag_t[torch.tensor([r for r, _, _, _ in paths], device=dev, dtype=torch.long),
                              torch.tensor([s for _, _, _, s in paths], device=dev, dtype=torch.long)] = f_t
                        e_t = torch.logsumexp(bag_t, dim=1)

            # 2. Evidence channel for Native Student
            l_ev_sup = None
            if not is_qt_only and bt:
                s_qe = torch.empty(len(e_ids), device=dev)
                proj_eT = torch.empty(len(e_ids), student.dim, device=dev)
                for kind in ("text", "image"):
                    loc = [j for j, e in enumerate(e_ids) if bank.kind(e) == kind]
                    if not loc:
                        continue
                    loc_t = torch.tensor(loc, device=dev, dtype=torch.long)
                    u_e = student.u(kind, bank.z_many([e_ids[j] for j in loc]))
                    s_qe = s_qe.index_put((loc_t,), (u_q @ student.R[f"Q_{kind}"] * u_e).sum(dim=-1))
                    proj_eT = proj_eT.index_put((loc_t,), u_e @ student.R[f"{kind}_T"])

                e_pos = {e: j for j, e in enumerate(e_ids)}
                e_idx = torch.tensor([e_pos[e] for _, _, e, _ in paths], device=dev, dtype=torch.long)
                t_idx = torch.tensor([k for _, k, _, _ in paths], device=dev, dtype=torch.long)
                path_scores = s_qe[e_idx] + (proj_eT[e_idx] * u_t[t_idx]).sum(dim=-1)

                max_bag = max(len(bags[t]) for _, t in bt)
                bag_s = path_scores.new_full((len(bt), max_bag), float("-inf"))
                bag_s = bag_s.index_put((torch.tensor([r for r, _, _, _ in paths], device=dev, dtype=torch.long),
                                         torch.tensor([s for _, _, _, s in paths], device=dev, dtype=torch.long)), path_scores)
                e_s = torch.logsumexp(bag_s, dim=1)
                ev_pos_mask = torch.tensor([t in positives for _, t in bt], dtype=torch.bool, device=dev)
                l_ev_sup = rank_mass_loss(e_s, ev_pos_mask)
                if is_kd and e_t is not None:
                    l_ev_kd = list_kl_divergence(e_s, e_t, temperature=1.0)

            # Combine C2 losses:
            # L_q = R(d_S) + 1_E R(e_S) + 0.3 * [KL(d_T, d_S) + 1_E KL(e_T, e_S)] + 0.1 * L_anchor
            loss_q = None
            if l_dir_sup is not None:
                loss_q = l_dir_sup
            if l_ev_sup is not None:
                loss_q = l_ev_sup if loss_q is None else loss_q + l_ev_sup
            if l_dir_kd is not None:
                loss_q = 0.3 * l_dir_kd if loss_q is None else loss_q + 0.3 * l_dir_kd
            if l_ev_kd is not None:
                loss_q = 0.3 * l_ev_kd if loss_q is None else loss_q + 0.3 * l_ev_kd

            if loss_q is not None:
                anchor_loss = 0.1 * student.anchor_loss()
                total_loss = (loss_q + anchor_loss) / n_batch
                total_loss.backward()
                batch_loss += float(total_loss.detach())

        nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step += 1
        elapsed = time.time() - t_start
        steps_done = max(step - start_step, 1)
        eta = (elapsed / steps_done) * (total_steps - step)
        step_loss = batch_loss
        print(f"[{arm}_C2] Step {step}/{total_steps} loss={step_loss:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

        if save_dir:
            prog = {
                "stage": f"STUDENT_{arm}_C2",
                "step": step,
                "total_steps": total_steps,
                "queries_done": min(start + logical_batch, total_queries),
                "loss": round(step_loss, 5),
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1),
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp_p = save_dir / "PROGRESS.json.tmp"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(prog, f, indent=2)
            tmp_p.replace(save_dir / "PROGRESS.json")

        if step in checkpoint_steps and save_dir:
            f = checkpoint_steps[step]
            ckpt_path = save_dir / f"snapshot_frac{int(f*100):03d}.pt"
            save_checkpoint(ckpt_path, student, optimizer, {"stage": "STUDENT_C2", "fraction": f, "step": step})
            saved_ckpts[f] = ckpt_path

    if save_dir and 1.0 not in saved_ckpts:
        ckpt_path = save_dir / "snapshot_frac100.pt"
        save_checkpoint(ckpt_path, student, optimizer, {"stage": "STUDENT_C2", "fraction": 1.0, "step": step})
        saved_ckpts[1.0] = ckpt_path

    return saved_ckpts

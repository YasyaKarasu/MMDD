"""Packet assembly, Teacher/Student losses and the two training trajectories.

Spec sections 7.3, 8 and 9.  All loss mathematics comes from ``reference``;
this module only decides which candidate IDs enter each list and how batches of
singleton logits are reduced.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from . import models, reference
from .config import ConfigError
from .models import (
    MODE_J,
    MODE_P,
    ObjectBank,
    TeacherBatch,
    build_optimizer,
    build_student,
    build_teacher,
    lr_multiplier,
    set_seed,
)
from .timing import Timing
from .sampling import (
    B_PACKET,
    C_PACKET,
    D_PACKET,
    E_PACKET,
    MakeList,
    NegativeList,
    augmented_bundle,
    select_witness_anchor,
    support_positive_set,
    use_natural_bundle,
)
from .util import SamplingCorpus, log_line, stable_digest, write_json, write_jsonl

PACKET_CHUNK = 8
CALIBRATION_CHUNK = 8


# --------------------------------------------------------------------------
# rank containers
# --------------------------------------------------------------------------


class RankTable:
    """Ranked candidate ids with their real float32 scores, in stored order."""

    def __init__(self, splits: dict[str, list[tuple[str, float]]] | None = None) -> None:
        self.splits = splits or {}

    def get(self, key: str) -> list[tuple[str, float]]:
        return self.splits.get(key, [])

    def ids(self, key: str) -> list[str]:
        return [value for value, _ in self.splits.get(key, [])]

    def to_rows(self) -> list[dict[str, Any]]:
        return [
            {"key": key, "ranked_ids": [v for v, _ in ranked],
             "scores": [s for _, s in ranked]}
            for key, ranked in sorted(self.splits.items())
        ]

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> "RankTable":
        return cls(
            {
                str(row["key"]): list(zip(row["ranked_ids"], row["scores"]))
                for row in rows
            }
        )


# --------------------------------------------------------------------------
# packet assembly
# --------------------------------------------------------------------------


class PacketBuilder:
    """Builds the D/E/C/B packets of spec 7.3 for one query and epoch."""

    def __init__(
        self,
        *,
        gt: dict[str, Any],
        corpora: dict[str, Sequence[str]],
        bundle_limit: int = 20,
        hard_pool_top_n: int = 128,
    ) -> None:
        self.gt = gt
        self.corpora = corpora
        self.bundle_limit = bundle_limit
        self.hard_pool_top_n = hard_pool_top_n
        self.modality: dict[str, str] = {}
        for evidence_id in corpora["evidence_text"]:
            self.modality[evidence_id] = "text"
        for evidence_id in corpora["evidence_image"]:
            self.modality[evidence_id] = "image"
        # Byte-ordered corpus indexes, derived once here rather than once per
        # packet: encoding and sorting a few hundred thousand ids costs far more
        # than the sampling it feeds.
        self._corpora: dict[str, SamplingCorpus] = {}

    def preload_corpora(self, corpora: dict[str, "SamplingCorpus"]) -> None:
        """Adopt pre-built corpus indexes, so a worker does not rebuild them."""
        self._corpora = dict(corpora)

    def corpus(self, destination: str) -> "SamplingCorpus":
        """Byte-ordered index for one destination, derived at most once."""
        cached = getattr(self, "_corpora", None)
        if cached is None:
            cached = {}
            self._corpora = cached
        if destination not in cached:
            cached[destination] = SamplingCorpus.build(self.corpora[destination])
        return cached[destination]

    # -- per-query supervision state ---------------------------------------

    def state(self, split: str, query_id: str) -> dict[str, Any] | None:
        population = self.gt[split]["population"]
        for row in population:
            if row["query_id"] == query_id:
                return row
        return None

    def population(self, split: str) -> list[dict[str, Any]]:
        return list(self.gt[split]["population"])

    def positive_sets(self, row: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
        return (
            set(row["direct_target_ids"]),
            set(row["implicit_target_ids"]),
            set(row["positive_target_ids"]),
        )

    def witnesses(self, row: dict[str, Any]) -> dict[str, set[str]]:
        return {k: set(v) for k, v in row["witnesses"].items()}

    def witness_union(self, row: dict[str, Any]) -> list[str]:
        out: set[str] = set()
        for values in row["witnesses"].values():
            out.update(values)
        return sorted(out, key=lambda v: v.encode("utf-8"))

    def natural_bundle(self, q_rank: dict[str, Any], per_modality: int) -> list[str]:
        return retrieve_interleave(
            q_rank["text_ids"], q_rank["image_ids"], per_modality, self.bundle_limit
        )

    # -- packet construction -----------------------------------------------

    def build(
        self,
        *,
        split: str,
        row: dict[str, Any],
        epoch: int,
        phase: str,
        sampling_arm: str,
        q_rank: dict[str, Any] | None,
        anchor_rank: dict[str, list[tuple[str, float]]],
        rank_tables: dict[str, RankTable],
        per_modality: int,
        include_bundle: bool,
    ) -> dict[str, Any]:
        query_id = row["query_id"]
        direct, implicit, all_positive = self.positive_sets(row)
        witnesses = self.witnesses(row)
        make = MakeList(
            phase=phase, sampling_arm=sampling_arm, corpus_universe=self.corpora,
            corpus_index=self._corpora,
        )
        packets: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, Any] = {"query_id": query_id, "epoch": epoch, "packets": {}}

        def add(name: str, packet: dict[str, Any]) -> None:
            packets[name] = packet
            diagnostics["packets"][name] = {
                "positives": len(packet["list"].positive_ids),
                "negatives": len(packet["list"].negative_ids),
                "list_length": len(packet["list"]),
                "provenance": packet["list"].provenance_counts(),
                "mode": packet["mode"],
                "context_size": len(packet["context"]) if packet.get("context") else 0,
            }

        # --- D packet: potential retrieval over all targets
        d_list = make.build(
            packet=D_PACKET,
            epoch=epoch,
            query_id=query_id,
            anchor=None,
            destination="target",
            positives=sorted(all_positive),
            excluded=[],
            hard_rank=rank_tables["D"].get(query_id),
        )
        add(D_PACKET, {
            "list": d_list,
            "mode": "P",
            "candidates": d_list.ordered_ids,
            "context": [],
        })

        # --- E packet: potential retrieval over all canonical evidence
        e_hard = merge_modality_ranks(
            rank_tables["E_text"].get(query_id),
            rank_tables["E_image"].get(query_id),
            self.hard_pool_top_n,
        )
        e_list = make.build(
            packet=E_PACKET,
            epoch=epoch,
            query_id=query_id,
            anchor=None,
            destination="evidence",
            positives=self.witness_union(row),
            excluded=[],
            hard_rank=e_hard,
        )
        add(E_PACKET, {
            "list": e_list,
            "mode": "P",
            "candidates": e_list.ordered_ids,
            "context": [],
        })

        # --- C packet: one witness anchor per epoch
        anchor = select_witness_anchor(self.witness_union(row), query_id, epoch)
        diagnostics["witness_anchor"] = anchor
        if anchor is not None:
            c_positives = support_positive_set(
                direct=direct, implicit=implicit, witnesses=witnesses, context={anchor}
            )
            c_excluded = all_positive - c_positives
            c_hard = merge_modality_ranks(
                anchor_rank.get(anchor, []),
                rank_tables["D"].get(query_id),
                self.hard_pool_top_n,
            )
            c_list = make.build(
                packet=C_PACKET,
                epoch=epoch,
                query_id=query_id,
                anchor=anchor,
                destination="target",
                positives=sorted(c_positives),
                excluded=c_excluded,
                hard_rank=c_hard,
            )
            add(C_PACKET, {
                "list": c_list,
                "mode": "J",
                "candidates": c_list.ordered_ids,
                "context": [anchor],
            })

        # --- B packet: same target list ids as D, context varies by epoch
        if include_bundle:
            natural = self.natural_bundle(q_rank or {}, per_modality) if q_rank else []
            if q_rank is None:
                natural = []
            if use_natural_bundle(epoch, query_id) or anchor is None:
                bundle = list(natural)
                view = "natural"
            else:
                bundle = augmented_bundle(natural, anchor, modality=self.modality)
                view = "witness_augmented"
            b_positives = support_positive_set(
                direct=direct, implicit=implicit, witnesses=witnesses, context=set(bundle)
            )
            b_excluded = all_positive - b_positives
            add(B_PACKET, {
                "list": d_list,
                "mode": "J",
                "candidates": d_list.ordered_ids,
                "context": bundle,
                "view": view,
                "positive_ids": sorted(b_positives),
                "excluded_ids": sorted(b_excluded),
            })
            diagnostics["packets"][B_PACKET]["positive_ids"] = sorted(b_positives)
        return {"packets": packets, "diagnostics": diagnostics}


def retrieve_interleave(
    text_ids: Sequence[str], image_ids: Sequence[str], per_modality: int, limit: int
) -> list[str]:
    from .retrieve import interleave_text_image

    return interleave_text_image(text_ids, image_ids, per_modality, limit)


def merge_modality_ranks(
    left: Sequence[tuple[str, float]] | None,
    right: Sequence[tuple[str, float]] | None,
    limit: int,
) -> list[tuple[str, float]]:
    """Alternate two already ranked lists by position, never comparing raw scores
    across modalities (spec 7.3 E packet)."""
    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    left = list(left or [])
    right = list(right or [])
    for position in range(max(len(left), len(right))):
        for stream in (left, right):
            if position < len(stream):
                value, score = stream[position]
                if value not in seen:
                    seen.add(value)
                    out.append((value, score))
        if len(out) >= limit:
            break
    return out[:limit]


class PackedTeacher:
    """Batches Teacher forwards across *rows of one candidate each*.

    A Teacher call returns exactly one logit per row, because it reads the [REL]
    position.  So every (query, candidate, context) triple becomes its own row of
    the forward pass, while rows from different packets share the same call.  The
    attention mask keeps rows independent, so no candidate ever sees another.
    """

    def __init__(self, batch: TeacherBatch, teacher: reference.UnifiedTeacher, chunk: int) -> None:
        self.batch = batch
        self.teacher = teacher
        self.chunk = max(1, int(chunk))
        self.entries: list[tuple[str, list[str], list[str], int]] = []

    def add(self, query_id: str, candidates: Sequence[str], context: Sequence[str], mode: int) -> int:
        self.entries.append((query_id, list(candidates), list(context), int(mode)))
        return len(self.entries) - 1

    def run(self, max_batch: int = 8) -> list[torch.Tensor]:
        """Score every queued entry and return one [n_candidates] tensor per entry."""
        rows: list[tuple[int, str, str, list[str], int]] = []
        for entry_index, (query_id, candidates, context, mode) in enumerate(self.entries):
            for candidate in candidates:
                rows.append((entry_index, query_id, candidate, context, mode))
        if not rows:
            return [torch.zeros(0, device=self.batch.device) for _ in self.entries]
        outputs: list[torch.Tensor | None] = [None] * len(rows)
        buckets: dict[int, list[int]] = {}
        for index, (_entry, _query, _candidate, context, _mode) in enumerate(rows):
            buckets.setdefault(1 + 1 + len(context), []).append(index)
        for width in sorted(buckets, reverse=True):
            group = buckets[width]
            for start in range(0, len(group), max_batch):
                block = group[start : start + max_batch]
                queries = [rows[i][1] for i in block]
                candidates = [[rows[i][2]] for i in block]
                contexts = [rows[i][3] for i in block]
                modes = {rows[i][4] for i in block}
                if len(modes) != 1:
                    raise ConfigError("a packed Teacher batch must share one mode")
                mode = modes.pop()
                built = self.batch.build(queries, candidates, contexts, mode)
                built["mode"] = torch.full(
                    (len(block),), mode, dtype=torch.long, device=self.batch.device
                )
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=str(self.batch.device).startswith("cuda"),
                ):
                    logits = self.teacher(**built)
                for offset, row_index in enumerate(block):
                    outputs[row_index] = logits[offset]
        if any(value is None for value in outputs):
            raise ConfigError("internal error: a packed Teacher row was never scored")
        grouped: list[list[torch.Tensor]] = [[] for _ in self.entries]
        for row, (entry_index, *_rest) in enumerate(rows):
            grouped[entry_index].append(outputs[row])  # type: ignore[arg-type]
        return [torch.stack(values) for values in grouped]


def list_scores(scores: torch.Tensor, packet: dict[str, Any]) -> torch.Tensor:
    """Reorder flat per-candidate scores into the packet's list order."""
    flat = torch.atleast_1d(scores)
    index = {value: i for i, value in enumerate(packet["candidates"])}
    return torch.stack([flat[index[value]] for value in packet["list"].ordered_ids])


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def teacher_rank_term(
    scores: torch.Tensor,
    packet: dict[str, Any],
) -> torch.Tensor | None:
    negative_list: NegativeList = packet["list"]
    positive_ids = packet.get("positive_ids", negative_list.positive_ids)
    if not positive_ids or not negative_list.negative_ids:
        return None
    index = {value: i for i, value in enumerate(packet["candidates"])}
    positive_mask = torch.zeros_like(scores, dtype=torch.bool)
    negative_mask = torch.zeros_like(scores, dtype=torch.bool)
    positive_mask[[index[v] for v in positive_ids]] = True
    negative_mask[[index[v] for v in negative_list.negative_ids]] = True
    if (positive_mask & negative_mask).any():
        raise ConfigError("a known positive target entered the negative competition")
    return reference.rank_loss(scores, positive_mask, negative_mask)


def calibration_loss(
    *,
    batch: TeacherBatch,
    teacher: reference.UnifiedTeacher,
    query_id: str,
    direct: set[str],
    implicit: set[str],
    witnesses: dict[str, set[str]],
    anchor: str | None,
    device: torch.device | str,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Spec 8.2.  Three softplus contrasts, averaged per class then across classes.

    A: softplus(-J(Q,T;empty)) for T in D_Q, target 1.
    B: softplus( J(Q,T;empty)) for T in I_Q, target 0.
    C: softplus(-J(Q,T;{E}))  for the epoch's witness anchor, target 1.
    """
    classes: list[torch.Tensor] = []
    diagnostics: dict[str, Any] = {"class_sizes": {}}

    def contrast(targets: list[str], context: list[str], sign: float) -> torch.Tensor | None:
        if not targets:
            return None
        logits = batch.score(
            teacher,
            [query_id] * len(targets),
            [[t] for t in targets],
            [list(context) for _ in targets],
            MODE_J,
            chunk=CALIBRATION_CHUNK,
        )
        stacked = torch.cat(logits)
        return torch.nn.functional.softplus(sign * stacked).mean()

    direct_term = contrast(sorted(direct), [], -1.0)
    if direct_term is not None:
        classes.append(direct_term)
        diagnostics["class_sizes"]["direct_empty"] = len(direct)
    implicit_term = contrast(sorted(implicit), [], 1.0)
    if implicit_term is not None:
        classes.append(implicit_term)
        diagnostics["class_sizes"]["implicit_empty"] = len(implicit)
    if anchor is not None:
        supported = sorted(
            t for t in implicit if witnesses.get(t, set()) & {anchor}
        )
        anchor_term = contrast(supported, [anchor], -1.0)
        if anchor_term is not None:
            classes.append(anchor_term)
            diagnostics["class_sizes"]["implicit_with_anchor"] = len(supported)
    if not classes:
        return None, diagnostics
    return torch.stack(classes).mean(), diagnostics


# --------------------------------------------------------------------------
# the Teacher trajectory
# --------------------------------------------------------------------------


class TeacherTrainer:
    def __init__(
        self,
        *,
        resolved: dict[str, Any],
        bank: ObjectBank,
        builder: PacketBuilder,
        rank_tables: dict[str, RankTable],
        anchor_rank: dict[str, list[tuple[str, float]]],
        output_dir: Path,
        device: str,
        limit_queries: int | None = None,
        max_epochs: int | None = None,
        prefetch_workers: int = 1,
    ) -> None:
        self.resolved = resolved
        self.bank = bank
        self.builder = builder
        self.rank_tables = rank_tables
        self.anchor_rank = anchor_rank
        self.output_dir = Path(output_dir)
        self.device = device
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.teacher = build_teacher(resolved["teacher"]).to(device)
        self.batch = TeacherBatch(bank, device=device)
        self.optimizer = build_optimizer(self.teacher, resolved["optimizer"], float(resolved["teacher"]["lr"]))
        teacher_config = resolved["teacher"]
        self.epochs = int(teacher_config["epochs"])
        if max_epochs:
            self.epochs = min(self.epochs, int(max_epochs))
        self.bundle_epochs = set(int(e) for e in teacher_config["bundle_epochs"])
        self.refresh_after = int(teacher_config["hard_refresh_after_epoch"])
        self.effective_batch = int(teacher_config["effective_query_batch"])
        self.chunk = int(resolved["retrieval"]["teacher_target_chunk"])
        self.per_modality = int(resolved["retrieval"]["evidence_per_modality"])
        self.calibration_weight = float(teacher_config["calibration_weight"])
        self.queries = [row["query_id"] for row in builder.population("train")]
        if limit_queries:
            self.queries = self.queries[: int(limit_queries)]
        if max_epochs:
            self.epochs = min(self.epochs, int(max_epochs))
        self.total_steps = self.epochs * max(1, math.ceil(len(self.queries) / self.effective_batch))
        self.optimizer_config = resolved["optimizer"]
        self.history: list[dict[str, Any]] = []
        self.timing = Timing()

    # -- one query ---------------------------------------------------------

    def query_loss(
        self, query_id: str, epoch: int, prefetched: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        row = self.builder.state("train", query_id)
        assert row is not None
        direct, implicit, all_positive = self.builder.positive_sets(row)
        witnesses = self.builder.witnesses(row)
        if prefetched is not None:
            built = prefetched
        else:
            with self.timing.stage("packet_build"):
                built = self.builder.build(
                    split="train",
                    row=row,
                    epoch=epoch,
                    phase="teacher",
                    sampling_arm="T",
                    q_rank=None,
                    anchor_rank=self.anchor_rank,
                    rank_tables=self.rank_tables,
                    per_modality=self.per_modality,
                    include_bundle=epoch in self.bundle_epochs,
                )
        terms: list[tuple[str, torch.Tensor]] = []
        skipped: list[str] = []
        for name in (D_PACKET, E_PACKET, C_PACKET, B_PACKET):
            packet = built["packets"].get(name)
            if packet is None:
                skipped.append(f"{name}:not_scheduled")
                continue
            with self.timing.stage("teacher_forward", packet=name):
                logits = self.batch.score(
                    self.teacher,
                    [query_id],
                    [packet["candidates"]],
                    [packet["context"]] if packet["mode"] == "J" else [[]],
                    MODE_J if packet["mode"] == "J" else MODE_P,
                    chunk=self.chunk,
                )[0]
            term = teacher_rank_term(logits, packet)
            if term is None:
                skipped.append(f"{name}:empty_positive_or_negative")
                continue
            terms.append((name, term))
            overlap = len(
                set(packet["list"].negative_ids)
                & set(packet.get("positive_ids", packet["list"].positive_ids))
            )
            if overlap:
                raise ConfigError(
                    f"{name}/{query_id}: {overlap} known positive target(s) entered "
                    "the negative competition"
                )
        anchor = built["diagnostics"].get("witness_anchor")
        with self.timing.stage("calibration_forward"):
            calibration, cal_diag = calibration_loss(
                batch=self.batch,
                teacher=self.teacher,
                query_id=query_id,
                direct=direct,
                implicit=implicit,
                witnesses=witnesses,
                anchor=anchor,
                device=self.device,
            )
        if not terms and calibration is None:
            return {"query_id": query_id, "loss": None, "skipped": skipped, "diagnostics": built["diagnostics"]}
        base = torch.stack([t for _, t in terms]).mean() if terms else None
        total = base
        if calibration is not None:
            total = calibration * self.calibration_weight if total is None else total + self.calibration_weight * calibration
        return {
            "query_id": query_id,
            "loss": total,
            "base": base,
            "calibration": calibration,
            "terms": {name: float(value.detach()) for name, value in terms},
            "skipped": skipped,
            "calibration_diagnostics": cal_diag,
            "diagnostics": built["diagnostics"],
        }

    # -- the schedule ------------------------------------------------------

    def train(self) -> dict[str, Any]:
        from .sampling import query_order_namespace
        from .util import stable_order

        started = time.time()
        step = 0
        for epoch in range(1, self.epochs + 1):
            order = stable_order(self.queries, query_order_namespace("T", epoch))
            pending: list[torch.Tensor] = []
            pending_diag: list[dict[str, Any]] = []
            self.teacher.train()
            include_bundle = epoch in self.bundle_epochs
            requests = (
                PacketRequest(
                    split="train",
                    row=self.builder.state("train", query_id),
                    epoch=epoch,
                    phase="teacher",
                    sampling_arm="T",
                    q_rank=None,
                    per_modality=self.per_modality,
                    include_bundle=include_bundle,
                )
                for query_id in order
            )
            prefetcher = PacketPrefetcher(builder=self.builder, workers=self.prefetch_workers)
            with self.timing.stage("packet_wait"):
                for query_id, prefetched in zip(
                    order,
                    prefetcher.stream(
                        requests, anchor_rank=self.anchor_rank, rank_tables=self.rank_tables
                    ),
                ):
                    result = self.query_loss(query_id, epoch, prefetched=prefetched)
                    if result["loss"] is not None:
                        pending.append(result["loss"])
                    pending_diag.append(result)
                    if len(pending) >= self.effective_batch:
                        step = self._step(pending, step)
                        pending = []
            if pending:
                step = self._step(pending, step)
            losses = [r for r in pending_diag if r["loss"] is not None]
            record = {
                "epoch": epoch,
                "queries": len(order),
                "queries_with_loss": len(losses),
                "timing": self._epoch_timing(epoch),
                "mean_loss": float(np.mean([float(r["loss"].detach()) for r in losses])) if losses else None,
                "terms": {
                    name: float(np.mean([r["terms"][name] for r in losses if name in r["terms"]]))
                    for name in (D_PACKET, E_PACKET, C_PACKET, B_PACKET)
                    if any(name in r["terms"] for r in losses)
                },
                "skip_counts": _count_skips(pending_diag),
                "step": step,
                "elapsed_seconds": time.time() - started,
            }
            self.history.append(record)
            write_json(self.output_dir / f"epoch_{epoch:02d}.json", record)
            log_line(
                f"teacher epoch {epoch}: loss={record['mean_loss']} "
                f"terms={record['terms']} ({record['elapsed_seconds']:.0f}s)"
            )
            # Keep a per-epoch checkpoint so the pre-registered dev selection can
            # be applied to every candidate epoch, not only the last one.
            torch.save(
                {"model": self.teacher.state_dict(), "epoch": epoch, "arm": "T"},
                self.output_dir / f"epoch_{epoch:02d}.pt",
            )
            if epoch == self.refresh_after:
                self.refresh_hard_ranks()
        return {"epochs": self.history, "steps": step, "elapsed_seconds": time.time() - started}

    def _epoch_timing(self, epoch: int) -> dict[str, Any]:
        """Per-epoch attribution; the accumulator resets so epochs stay separate."""
        report = self.timing.report(name=f"teacher_epoch_{epoch:02d}")
        self.timing = Timing()
        return report

    def _step(self, losses: list[torch.Tensor], step: int) -> int:
        step += 1
        multiplier = lr_multiplier(
            step,
            self.total_steps,
            float(self.optimizer_config["warmup_fraction"]),
            float(self.optimizer_config["end_lr_fraction"]),
        )
        base_lr = float(self.resolved["teacher"]["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * multiplier
        with self.timing.stage("optimizer_step"):
            total = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.teacher.parameters(), float(self.optimizer_config["clip_grad_norm"])
            )
            self.optimizer.step()
        return step

    def refresh_hard_ranks(self) -> dict[str, Any]:
        """Spec 8.3: one in-run refresh after epoch 2, using the current Teacher."""
        started = time.time()
        self.teacher.eval()
        refreshed: dict[str, dict[str, list[tuple[str, float]]]] = {
            "D": {}, "E_text": {}, "E_image": {}, "anchor": {}
        }
        with torch.no_grad():
            for row in self.builder.population("train"):
                query_id = row["query_id"]
                pool_targets = _dedupe(
                    self.rank_tables["D"].ids(query_id)
                    + [v for v, _ in self.anchor_hard_pool(query_id)]
                )
                if pool_targets:
                    scores = torch.cat(
                        self.batch.score(
                            self.teacher, [query_id], [pool_targets], None, MODE_P,
                            chunk=self.chunk,
                        )
                    )
                    refreshed["D"][query_id] = _ranked(pool_targets, scores)
                pool_text = self.rank_tables["E_text"].ids(query_id)
                pool_image = self.rank_tables["E_image"].ids(query_id)
                if pool_text:
                    scores = torch.cat(
                        self.batch.score(self.teacher, [query_id], [pool_text], None, MODE_P,
                                         chunk=self.chunk)
                    )
                    refreshed["E_text"][query_id] = _ranked(pool_text, scores)
                if pool_image:
                    scores = torch.cat(
                        self.batch.score(self.teacher, [query_id], [pool_image], None, MODE_P,
                                         chunk=self.chunk)
                    )
                    refreshed["E_image"][query_id] = _ranked(pool_image, scores)
        payload = {
            key: RankTable(value).to_rows() for key, value in refreshed.items()
        }
        write_jsonl(self.output_dir / "mining_epoch2" / "refreshed_ranks.jsonl", (
            {"table": table, **row} for table, rows in payload.items() for row in rows
        ))
        self.rank_tables["D"] = RankTable(refreshed["D"])
        self.rank_tables["E_text"] = RankTable(refreshed["E_text"])
        self.rank_tables["E_image"] = RankTable(refreshed["E_image"])
        write_json(
            self.output_dir / "mining_epoch2" / "meta.json",
            {
                "refreshed_queries": {k: len(v) for k, v in refreshed.items()},
                "elapsed_seconds": time.time() - started,
                "source": "current in-run Teacher, mode=P",
            },
        )
        log_line(f"teacher: refreshed hard ranks after epoch 2 ({time.time() - started:.0f}s)")
        return {"refreshed": {k: len(v) for k, v in refreshed.items()}}

    def anchor_hard_pool(self, query_id: str) -> list[tuple[str, float]]:
        return self.rank_tables["D"].get(query_id)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ranked(ids: Sequence[str], scores: torch.Tensor) -> list[tuple[str, float]]:
    pairs = [(value, float(score)) for value, score in zip(ids, scores)]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _count_skips(results: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for reason in result.get("skipped", []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


# --------------------------------------------------------------------------
# the Student trajectories
# --------------------------------------------------------------------------


class StudentTrainer:
    """Spec section 9: SUP and SUP+KD from one shared fresh initialisation."""

    def __init__(
        self,
        *,
        resolved: dict[str, Any],
        bank: ObjectBank,
        builder: PacketBuilder,
        rank_tables: dict[str, RankTable],
        anchor_rank: dict[str, list[tuple[str, float]]],
        output_dir: Path,
        arm: str,
        init_state: dict[str, torch.Tensor],
        device: str,
        limit_queries: int | None = None,
        max_epochs: int | None = None,
        prefetch_workers: int = 1,
    ) -> None:
        if arm not in ("SUP", "KD"):
            raise ConfigError(f"unknown Student arm: {arm}")
        self.resolved = resolved
        self.bank = bank
        self.builder = builder
        self.rank_tables = rank_tables
        self.anchor_rank = anchor_rank
        self.output_dir = Path(output_dir)
        self.arm = arm
        self.device = device
        self.student = build_student(resolved["student"]).to(device)
        # Both arms start from the identical fresh initialisation, loaded into
        # separate parameter storage (spec 3.1: never share mutable parameters).
        self.student.load_state_dict({k: v.clone() for k, v in init_state.items()})
        student_config = resolved["student"]
        self.optimizer = build_optimizer(
            self.student, resolved["optimizer"], float(student_config["lr"])
        )
        self.epochs = int(student_config["epochs"])
        self.effective_batch = int(student_config["effective_query_batch"])
        # The Teacher forward is packed per this many queries; the optimizer step
        # still happens once per ``effective_query_batch`` queries.
        self.block_queries = min(self.effective_batch, 4)
        self.kd_temperature = float(student_config["kd_temperature"])
        self.kd_weight = float(student_config["kd_weight"])
        self.per_modality = int(resolved["retrieval"]["evidence_per_modality"])
        self.queries = [row["query_id"] for row in builder.population("train")]
        if limit_queries:
            self.queries = self.queries[: int(limit_queries)]
        if max_epochs:
            self.epochs = min(self.epochs, int(max_epochs))
        self.total_steps = self.epochs * max(1, math.ceil(len(self.queries) / self.effective_batch))
        self.optimizer_config = resolved["optimizer"]
        self.history: list[dict[str, Any]] = []
        self.timing = Timing()
        self.object_ids = list(bank.object_ids)
        self.prefetch_workers = max(1, int(prefetch_workers))
        self.metadata: dict[str, Any] = {"arm": arm, "epochs": []}

    # -- keys --------------------------------------------------------------

    def object_keys(self) -> np.ndarray:
        """Spec 9.2 item 1: recompute every object key with the current parameters."""
        with self.timing.stage("object_key_bank"):
            return self._object_keys_inner()

    def _object_keys_inner(self) -> np.ndarray:
        self.student.eval()
        keys = np.zeros((len(self.object_ids), self.student.d), dtype=np.float32)
        with torch.no_grad():
            step = 2048
            for start in range(0, len(self.object_ids), step):
                block = self.object_ids[start : start + step]
                cache, valid, modality, kind = self.bank.keys(block, self.device)
                keys[start : start + step] = (
                    self.student.encode(cache, valid, modality, kind).float().cpu().numpy()
                )
        return keys

    def key_index(self) -> dict[str, Any]:
        """Per-path projected corpus keys for this epoch (spec 9.2, 10.1).

        Each object's Student key is computed once and reused for all four
        projections: re-encoding the identical object set per path would not
        change a single stored vector.
        """
        corpora = self.builder.corpora
        base = self.object_keys()
        index = {object_id: i for i, object_id in enumerate(self.object_ids)}
        target_ids = list(corpora["target"])
        text_ids = list(corpora["evidence_text"])
        image_ids = list(corpora["evidence_image"])
        base_tensor = torch.from_numpy(base)

        def project(ids: list[str], linear: torch.nn.Module) -> np.ndarray:
            vectors = base_tensor[[index[o] for o in ids]]
            return reference.unit(vectors @ linear.weight.detach().cpu().T).numpy()

        return {
            "target_ids": target_ids,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "target_position": index,
            # C path: the target index key is base_e(nu_T); Eq. (14) conditions only
            # the query vector, never the target key or the target index.
            "target_keys_C": torch.from_numpy(
                reference.unit(
                    base_tensor[[index[o] for o in target_ids]]
                    @ self.student.base_e.weight.detach().cpu().T
                ).numpy()
            ).to(self.device),
            "D_keys": project(target_ids, self.student.direct),
            "E_text_keys": project(text_ids, self.student.evidence),
            "E_image_keys": project(image_ids, self.student.evidence),
        }

    # -- one query ---------------------------------------------------------

    def prepare_query(
        self,
        query_id: str,
        epoch: int,
        q_rank: dict[str, Any] | None,
        prefetched: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Phase 1: build packets (no forward), so KD rows can be packed by width."""
        row = self.builder.state("train", query_id)
        assert row is not None
        if prefetched is not None:
            built = prefetched
        else:
            with self.timing.stage("packet_build"):
                built = self.builder.build(
                    split="train",
                    row=row,
                    epoch=epoch,
                    phase="student",
                    sampling_arm="S",
                    q_rank=q_rank,
                    anchor_rank=self.anchor_rank,
                    rank_tables=self.rank_tables,
                    per_modality=self.per_modality,
                    include_bundle=False,
                )
        return {"query_id": query_id, "epoch": epoch, "built": built}

    def finalize_query(self, prepared: dict[str, Any], kd_scores: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Phase 2: Student rank loss (plus KD when the arm asks for it)."""
        query_id = prepared["query_id"]
        terms: list[torch.Tensor] = []
        diagnostics: dict[str, Any] = {
            "query_id": query_id, "epoch": prepared["epoch"], "packets": {}
        }
        skipped: list[str] = []
        for name in (D_PACKET, E_PACKET, C_PACKET):
            packet = prepared["built"]["packets"].get(name)
            if packet is None:
                skipped.append(f"{name}:not_scheduled")
                continue
            with self.timing.stage("student_forward", packet=name):
                scores = self.score_packet(query_id, packet)
            with self.timing.stage("rank_loss", packet=name):
                term = teacher_rank_term(scores, packet)
            entry = {
                "positives": len(packet["list"].positive_ids),
                "negatives": len(packet["list"].negative_ids),
                "provenance": packet["list"].provenance_counts(),
                "known_positive_overlap": len(
                    set(packet["list"].negative_ids) & set(packet["list"].positive_ids)
                ),
            }
            if term is None:
                skipped.append(f"{name}:empty_positive_or_negative")
                diagnostics["packets"][name] = {**entry, "skipped": True}
                continue
            terms.append(term)
            entry["rank_loss"] = float(term.detach())
            if self.arm == "KD":
                teacher_scores = kd_scores.get(name)
                if teacher_scores is None:
                    raise ConfigError(
                        f"{name}/{query_id}: the KD arm requires frozen Teacher logits"
                    )
                valid_mask = torch.zeros_like(scores, dtype=torch.bool)
                index = {value: i for i, value in enumerate(packet["list"].ordered_ids)}
                for value in packet["list"].positive_ids:
                    valid_mask[index[value]] = True
                for value in packet["list"].negative_ids:
                    valid_mask[index[value]] = True
                with self.timing.stage("kd_loss", packet=name):
                    kd = reference.kd_loss(
                        scores, teacher_scores, valid_mask, self.kd_temperature
                    )
                terms.append(kd * self.kd_weight)
                probability = torch.softmax(
                    teacher_scores[valid_mask].detach().float() / self.kd_temperature, 0
                )
                entry["kd"] = float(kd.detach())
                entry["kd_entropy"] = float(
                    -(probability * probability.clamp_min(1e-12).log()).sum()
                )
                entry["kd_valid"] = int(valid_mask.sum())
            diagnostics["packets"][name] = entry
        if not terms:
            return {"query_id": query_id, "loss": None, "skipped": skipped, "diagnostics": diagnostics}
        return {
            "query_id": query_id,
            "loss": torch.stack(terms).mean(),
            "skipped": skipped,
            "diagnostics": diagnostics,
        }

    def score_packet(self, query_id: str, packet: dict[str, Any]) -> torch.Tensor:
        """Spec Eq. (15) for one packet.

        Every object in the list is re-encoded with the **current** parameters,
        so the whole path from the raw cache through pooling, projection and the
        conditional query is differentiable.  Eq. (14) conditions only the query
        vector, so the candidate keys go through the shared pooling plus the
        target-side projection and never see the evidence.
        """
        candidates = packet["candidates"]
        objects = [query_id] + list(candidates)
        if packet["context"]:
            objects = objects + [packet["context"][0]]
        cache, valid, modality, kind = self.bank.keys(objects, self.device)
        vectors = self.student.encode(cache, valid, modality, kind)
        if packet["context"]:
            query_vector = self.student.query_next(vectors[0], vectors[-1])
        elif packet["mode"] == "P":
            query_vector = self.student.query_direct(vectors[0])
        else:
            query_vector = self.student.query_evidence(vectors[0])
        candidate_keys = self.student.base_e(vectors[1 : 1 + len(candidates)])
        return self.student.logits(query_vector, candidate_keys)

    # -- mining refresh ----------------------------------------------------

    def refresh_mining(self, epoch: int, keys: dict[str, Any]) -> dict[str, Any]:
        """Spec 9.2: exact top-128 mining with the previous epoch's *last* model."""
        started = time.time()
        refreshed: dict[str, dict[str, list[tuple[str, float]]]] = {
            "D": {}, "E_text": {}, "E_image": {}, "anchor": {}
        }
        self.student.eval()
        target_ids = keys["target_ids"]
        text_ids = keys["text_ids"]
        image_ids = keys["image_ids"]
        rows = self.builder.population("train")
        query_ids = [row["query_id"] for row in rows]
        base = self._base_vectors(query_ids)
        with self.timing.stage("mining_project_queries"), torch.no_grad():
            d_queries = torch.from_numpy(
                _unit_np(base @ self.student.direct.weight.detach().cpu().numpy().T)
            ).numpy()
            e_queries = torch.from_numpy(
                _unit_np(base @ self.student.evidence.weight.detach().cpu().numpy().T)
            ).numpy()
        with self.timing.stage("mining_exact_topk"):
            d_scores = d_queries @ keys["D_keys"].T
            text_scores = e_queries @ keys["E_text_keys"].T
            image_scores = e_queries @ keys["E_image_keys"].T
        for i, row in enumerate(rows):
            query_id = row["query_id"]
            positives = set(row["positive_target_ids"])
            refreshed["D"][query_id] = _top_pairs(target_ids, d_scores[i], 128, positives)
            refreshed["E_text"][query_id] = _top_pairs(text_ids, text_scores[i], 128, set())
            refreshed["E_image"][query_id] = _top_pairs(image_ids, image_scores[i], 128, set())
        payload = {key: RankTable(value).to_rows() for key, value in refreshed.items()}
        out_dir = self.output_dir / f"lists_epoch{epoch:02d}"
        write_jsonl(
            out_dir / "refreshed_ranks.jsonl",
            ({"table": table, **row} for table, rows_ in payload.items() for row in rows_),
        )
        self.rank_tables["D"] = RankTable(refreshed["D"])
        self.rank_tables["E_text"] = RankTable(refreshed["E_text"])
        self.rank_tables["E_image"] = RankTable(refreshed["E_image"])
        meta = {
            "epoch": epoch,
            "generator": f"{self.arm} epoch {epoch - 1} last parameters",
            "elapsed_seconds": time.time() - started,
            "corpus_sizes": {
                "target": len(target_ids), "text": len(text_ids), "image": len(image_ids)
            },
            "note": "hard-negative ids only; no loss is computed against these keys",
        }
        write_json(out_dir / "meta.json", meta)
        log_line(f"{self.arm}: refreshed mining for epoch {epoch} ({time.time() - started:.0f}s)")
        return meta

    def _base_vectors(self, query_ids: Sequence[str]) -> np.ndarray:
        self.student.eval()
        out = np.zeros((len(query_ids), self.student.d), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(query_ids), 512):
                block = list(query_ids[start : start + 512])
                cache, valid, modality, kind = self.bank.keys(block, self.device)
                out[start : start + 512] = (
                    self.student.encode(cache, valid, modality, kind).float().cpu().numpy()
                )
        return out

    # -- the schedule ------------------------------------------------------

    def train(
        self,
        *,
        teacher: reference.UnifiedTeacher | None,
        teacher_batch: TeacherBatch | None,
        query_rankings: dict[str, dict[str, Any]],
        dev_ranker: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        from .sampling import query_order_namespace
        from .util import stable_order

        started = time.time()
        step = 0
        chunk = int(self.resolved["retrieval"]["teacher_target_chunk"])
        for epoch in range(1, self.epochs + 1):
            keys = self.key_index()
            order = stable_order(self.queries, query_order_namespace("S", epoch))
            self.student.train()
            self.student.requires_grad_(True)
            pending: list[torch.Tensor] = []
            diagnostics: list[dict[str, Any]] = []
            prepared_block: list[dict[str, Any]] = []
            # Packet construction is the bulk of a Student step and needs no model,
            # so it runs in worker processes while the GPU works on earlier queries.
            # Results are consumed strictly in query order.
            requests = (
                PacketRequest(
                    split="train",
                    row=self.builder.state("train", query_id),
                    epoch=epoch,
                    phase="student",
                    sampling_arm="S",
                    q_rank=query_rankings.get(query_id),
                    per_modality=self.per_modality,
                    include_bundle=False,
                )
                for query_id in order
            )
            prefetcher = PacketPrefetcher(
                builder=self.builder, workers=self.prefetch_workers
            )
            built = prefetcher.stream(
                requests, anchor_rank=self.anchor_rank, rank_tables=self.rank_tables
            )
            for query_id, prefetched in zip(order, built):
                prepared = self.prepare_query(
                    query_id, epoch, query_rankings.get(query_id), prefetched=prefetched
                )
                prepared_block.append(prepared)
                if len(prepared_block) < self.block_queries:
                    continue
                result = self._run_block(prepared_block, teacher, teacher_batch, chunk)
                diagnostics.extend(result.pop("records"))
                pending.extend(result["losses"])
                prepared_block = []
                if len(pending) >= self.effective_batch:
                    step = self._step(pending[: self.effective_batch], step)
                    pending = pending[self.effective_batch :]
            if prepared_block:
                result = self._run_block(prepared_block, teacher, teacher_batch, chunk)
                diagnostics.extend(result.pop("records"))
                pending.extend(result["losses"])
            if pending:
                # Fewer queries than one effective batch: mean over the real count.
                step = self._step(pending, step)
            losses = [r for r in diagnostics if r["loss"] is not None]
            record = {
                "epoch": epoch,
                "arm": self.arm,
                "queries": len(order),
                "queries_with_loss": len(losses),
                "mean_loss": float(np.mean([float(r["loss"].detach()) for r in losses])) if losses else None,
                "skip_counts": _count_skips(diagnostics),
                "step": step,
                "elapsed_seconds": time.time() - started,
                "timing": self._epoch_timing(epoch),
                "packet_terms": _aggregate_packet_terms(diagnostics),
            }
            self.history.append(record)
            write_json(self.output_dir / f"epoch_{epoch:02d}.json", record)
            log_line(
                f"{self.arm} epoch {epoch}: loss={record['mean_loss']} "
                f"steps={record['step']} ({record['elapsed_seconds']:.0f}s)"
            )
            # Keep the per-epoch checkpoint: dev selection needs every candidate
            # epoch, and next-epoch mining is driven by *last*, never by best.
            torch.save(
                {"model": self.student.state_dict(), "epoch": epoch, "arm": self.arm},
                self.output_dir / f"epoch_{epoch:02d}.pt",
            )
            if dev_ranker is not None:
                evaluation = dev_ranker(self.student, keys)
                record["dev"] = evaluation
                write_json(self.output_dir / f"epoch_{epoch:02d}.dev.json", evaluation)
                log_line(
                    f"{self.arm} epoch {epoch} dev R@10={evaluation.get('overall_R10')} "
                    f"implicit={evaluation.get('implicit_R10')}"
                )
            if epoch < self.epochs:
                self.refresh_mining(epoch + 1, keys)
        return {"epochs": self.history, "steps": step, "elapsed_seconds": time.time() - started}

    def _run_block(
        self,
        prepared_block: list[dict[str, Any]],
        teacher: reference.UnifiedTeacher | None,
        teacher_batch: TeacherBatch | None,
        chunk: int,
    ) -> dict[str, Any]:
        """One effective batch: student rank losses and packed Teacher KD logits.

        The Teacher forward is grouped by sequence width across the whole batch,
        so its cost is amortised without changing any row's score.  Teacher rows
        are scored under ``no_grad``: the KD target is a constant (spec 9.1).
        """
        kd_scores: list[dict[str, torch.Tensor]] = [{} for _ in prepared_block]
        if self.arm == "KD":
            if teacher is None or teacher_batch is None:
                raise ConfigError("the KD arm requires the frozen Teacher")
            packed = PackedTeacher(teacher_batch, teacher, chunk)
            slots: list[tuple[int, str, int]] = []
            for index, prepared in enumerate(prepared_block):
                for name in (D_PACKET, E_PACKET, C_PACKET):
                    packet = prepared["built"]["packets"].get(name)
                    if packet is None:
                        continue
                    mode = MODE_J if packet["mode"] == "J" else MODE_P
                    slots.append(
                        (
                            index,
                            name,
                            packed.add(prepared["query_id"], packet["candidates"],
                                       packet["context"], mode),
                        )
                    )
            with self.timing.stage("teacher_forward_kd"), torch.no_grad():
                flat = packed.run(max_batch=8)
            for index, name, row in slots:
                packet = prepared_block[index]["built"]["packets"][name]
                kd_scores[index][name] = list_scores(flat[row], packet)
        records = []
        for index, prepared in enumerate(prepared_block):
            records.append(self.finalize_query(prepared, kd_scores[index]))
        return {"losses": [r["loss"] for r in records if r["loss"] is not None], "records": records}

    def _epoch_timing(self, epoch: int) -> dict[str, Any]:
        report = self.timing.report(name=f"{self.arm}_epoch_{epoch:02d}")
        self.timing = Timing()
        return report

    def _step(self, losses: list[torch.Tensor], step: int) -> int:
        """One optimizer step per ``effective_query_batch`` queries (spec 9.3)."""
        step += 1
        multiplier = lr_multiplier(
            step,
            self.total_steps,
            float(self.optimizer_config["warmup_fraction"]),
            float(self.optimizer_config["end_lr_fraction"]),
        )
        base_lr = float(self.resolved["student"]["lr"])
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * multiplier
        with self.timing.stage("optimizer_step"):
            total = torch.stack(losses).mean()
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), float(self.optimizer_config["clip_grad_norm"])
            )
            self.optimizer.step()
        return step

    def refresh_hard_ranks(self) -> dict[str, Any]:
        """Spec 8.3: one in-run refresh after epoch 2, using the current Teacher."""
        started = time.time()
        self.teacher.eval()
        refreshed: dict[str, dict[str, list[tuple[str, float]]]] = {
            "D": {}, "E_text": {}, "E_image": {}, "anchor": {}
        }
        with torch.no_grad():
            for row in self.builder.population("train"):
                query_id = row["query_id"]
                pool_targets = _dedupe(
                    self.rank_tables["D"].ids(query_id)
                    + [v for v, _ in self.anchor_hard_pool(query_id)]
                )
                if pool_targets:
                    scores = torch.cat(
                        self.batch.score(
                            self.teacher, [query_id], [pool_targets], None, MODE_P,
                            chunk=self.chunk,
                        )
                    )
                    refreshed["D"][query_id] = _ranked(pool_targets, scores)
                pool_text = self.rank_tables["E_text"].ids(query_id)
                pool_image = self.rank_tables["E_image"].ids(query_id)
                if pool_text:
                    scores = torch.cat(
                        self.batch.score(self.teacher, [query_id], [pool_text], None, MODE_P,
                                         chunk=self.chunk)
                    )
                    refreshed["E_text"][query_id] = _ranked(pool_text, scores)
                if pool_image:
                    scores = torch.cat(
                        self.batch.score(self.teacher, [query_id], [pool_image], None, MODE_P,
                                         chunk=self.chunk)
                    )
                    refreshed["E_image"][query_id] = _ranked(pool_image, scores)
        payload = {
            key: RankTable(value).to_rows() for key, value in refreshed.items()
        }
        write_jsonl(self.output_dir / "mining_epoch2" / "refreshed_ranks.jsonl", (
            {"table": table, **row} for table, rows in payload.items() for row in rows
        ))
        self.rank_tables["D"] = RankTable(refreshed["D"])
        self.rank_tables["E_text"] = RankTable(refreshed["E_text"])
        self.rank_tables["E_image"] = RankTable(refreshed["E_image"])
        write_json(
            self.output_dir / "mining_epoch2" / "meta.json",
            {
                "refreshed_queries": {k: len(v) for k, v in refreshed.items()},
                "elapsed_seconds": time.time() - started,
                "source": "current in-run Teacher, mode=P",
            },
        )
        log_line(f"teacher: refreshed hard ranks after epoch 2 ({time.time() - started:.0f}s)")
        return {"refreshed": {k: len(v) for k, v in refreshed.items()}}

    def anchor_hard_pool(self, query_id: str) -> list[tuple[str, float]]:
        return self.rank_tables["D"].get(query_id)


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _ranked(ids: Sequence[str], scores: torch.Tensor) -> list[tuple[str, float]]:
    pairs = [(value, float(score)) for value, score in zip(ids, scores)]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs


def _count_skips(results: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for reason in result.get("skipped", []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


# --------------------------------------------------------------------------
# the Student trajectories
# --------------------------------------------------------------------------


def _unit_np(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms < 1e-6).any():
        raise ConfigError(
            "non-finite or degenerate Student vector before unit normalization"
        )
    return values / norms


def _aggregate_packet_terms(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in (D_PACKET, E_PACKET, C_PACKET):
        entries = [
            r["diagnostics"]["packets"][name]
            for r in diagnostics
            if name in r.get("diagnostics", {}).get("packets", {})
        ]
        if not entries:
            continue
        present = [e for e in entries if "rank_loss" in e]
        out[name] = {
            "packets": len(entries),
            "packets_with_loss": len(present),
            "mean_rank_loss": float(np.mean([e["rank_loss"] for e in present])) if present else None,
            "mean_negatives": float(np.mean([e["negatives"] for e in entries])),
            "mean_positives": float(np.mean([e["positives"] for e in entries])),
            # Counted from the packet construction itself: every negative id is
            # checked against the task's known positives before the loss runs.
            "known_positive_as_negative": int(
                sum(e.get("known_positive_overlap", 0) for e in entries)
            ),
            "provenance_totals": _sum_provenance(entries),
        }
        if any("kd" in e for e in present):
            kd_entries = [e for e in present if "kd" in e]
            out[name]["mean_kd"] = float(np.mean([e["kd"] for e in kd_entries]))
            out[name]["mean_kd_entropy"] = float(
                np.mean([e["kd_entropy"] for e in kd_entries])
            )
    return out


def _sum_provenance(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for entry in entries:
        for key, value in entry.get("provenance", {}).items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _top_pairs(
    ids: Sequence[str], scores: np.ndarray, k: int, excluded: set[str]
) -> list[tuple[str, float]]:
    pairs = [
        (value, float(scores[i]))
        for i, value in enumerate(ids)
        if value not in excluded
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs[:k]


# --------------------------------------------------------------------------
# bounded CPU prefetch of packet construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PacketRequest:
    """One unit of packet-construction work, small enough to pickle per task."""

    split: str
    row: dict[str, Any]
    epoch: int
    phase: str
    sampling_arm: str
    q_rank: dict[str, Any] | None
    per_modality: int
    include_bundle: bool


_PREFETCH_BUILDER: "PacketBuilder | None" = None
_PREFETCH_ANCHOR_RANK: dict[str, list[tuple[str, float]]] = {}
_PREFETCH_RANK_TABLES: dict[str, "RankTable"] = {}


def _prefetch_init(
    builder: "PacketBuilder",
    corpora: dict[str, "SamplingCorpus"],
    anchor_rank: dict[str, list[tuple[str, float]]],
    rank_tables: dict[str, "RankTable"],
) -> None:
    """Worker start-up: adopt everything that is constant for the epoch.

    Ranking tables and the corpus index are identical in every worker and are far
    larger than the per-task descriptor, so they are handed over once here instead
    of being shipped with each request.  No model, no GPU tensor and no torch RNG
    state crosses into a worker.
    """
    global _PREFETCH_BUILDER, _PREFETCH_ANCHOR_RANK, _PREFETCH_RANK_TABLES
    _PREFETCH_BUILDER = builder
    _PREFETCH_ANCHOR_RANK = anchor_rank
    _PREFETCH_RANK_TABLES = rank_tables
    builder.preload_corpora(corpora)


def _prefetch_run(request: PacketRequest) -> dict[str, Any]:
    if _PREFETCH_BUILDER is None:
        raise RuntimeError("prefetch worker was not initialised")
    anchor_rank = _PREFETCH_ANCHOR_RANK
    rank_tables = _PREFETCH_RANK_TABLES
    built = _PREFETCH_BUILDER.build(
        split=request.split,
        row=request.row,
        epoch=request.epoch,
        phase=request.phase,
        sampling_arm=request.sampling_arm,
        q_rank=request.q_rank,
        anchor_rank=anchor_rank,
        rank_tables=rank_tables,
        per_modality=request.per_modality,
        include_bundle=request.include_bundle,
    )
    return _compact_packets(built)


def _compact_packets(built: dict[str, Any]) -> dict[str, Any]:
    """Reduce a build result to plain data, so the pool never pickles a module."""
    packets = {}
    for name, packet in built["packets"].items():
        negative_list = packet["list"]
        packets[name] = {
            "positive_ids": list(negative_list.positive_ids),
            "negative_ids": list(negative_list.negative_ids),
            "ordered_ids": list(negative_list.ordered_ids),
            "provenance": dict(negative_list.provenance),
            "mode": packet["mode"],
            "candidates": list(packet["candidates"]),
            "context": list(packet["context"]),
            "view": packet.get("view"),
            "positive_ids_override": packet.get("positive_ids"),
        }
    return {"packets": packets, "diagnostics": built["diagnostics"]}


def _expand_packets(compact: dict[str, Any]) -> dict[str, Any]:
    packets = {}
    for name, data in compact["packets"].items():
        packets[name] = {
            "list": NegativeList(
                positive_ids=data["positive_ids"],
                negative_ids=data["negative_ids"],
                ordered_ids=data["ordered_ids"],
                provenance=data["provenance"],
            ),
            "mode": data["mode"],
            "candidates": data["candidates"],
            "context": data["context"],
            **({"view": data["view"]} if data.get("view") else {}),
            **({"positive_ids": data["positive_ids_override"]}
               if data.get("positive_ids_override") else {}),
        }
    return {"packets": packets, "diagnostics": compact["diagnostics"]}


class PacketPrefetcher:
    """Computes packet construction on CPU ahead of the GPU step, in input order.

    Sampling is about two thirds of a training step and needs no model at all, so
    it runs in worker processes while the GPU works on earlier queries.  Results are
    consumed strictly in the order they were requested: a worker may finish out of
    order, but training never reorders queries, skips one, or substitutes another.
    With one worker, or on a machine with a single usable core, the same generator
    degrades to running inline.
    """

    def __init__(
        self,
        *,
        builder: "PacketBuilder",
        workers: int = 4,
        max_pending: int = 16,
    ) -> None:
        self.builder = builder
        self.workers = max(1, int(workers))
        self.max_pending = max(1, int(max_pending))
        self._pool = None

    def _start(
        self,
        corpora: dict[str, "SamplingCorpus"],
        anchor_rank: dict[str, list[tuple[str, float]]],
        rank_tables: dict[str, "RankTable"],
    ) -> None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_prefetch_init,
            initargs=(self.builder, corpora, anchor_rank, rank_tables),
        )

    def stream(
        self,
        requests: Iterable[PacketRequest],
        *,
        anchor_rank: dict[str, list[tuple[str, float]]],
        rank_tables: dict[str, "RankTable"],
    ) -> Iterator[dict[str, Any]]:
        """Yield build results in request order, computing ahead by ``max_pending``."""
        corpora = {
            destination: self.builder.corpus(destination)
            for destination in ("target", "evidence")
            if destination in self.builder.corpora
        }
        if self.workers == 1:
            # Inline: no pool, no worker globals, identical code path to a direct
            # build, so a machine with one usable core still trains correctly.
            for request in requests:
                yield _expand_packets(
                    _compact_packets(
                        self.builder.build(
                            split=request.split,
                            row=request.row,
                            epoch=request.epoch,
                            phase=request.phase,
                            sampling_arm=request.sampling_arm,
                            q_rank=request.q_rank,
                            anchor_rank=anchor_rank,
                            rank_tables=rank_tables,
                            per_modality=request.per_modality,
                            include_bundle=request.include_bundle,
                        )
                    )
                )
            return
        from collections import deque

        self._start(corpora, anchor_rank, rank_tables)
        try:
            iterator = iter(requests)
            pending: deque = deque()
            for _ in range(self.max_pending):
                request = next(iterator, None)
                if request is None:
                    break
                pending.append(self._pool.submit(_prefetch_run, request))
            while pending:
                compact = pending.popleft().result()
                request = next(iterator, None)
                if request is not None:
                    pending.append(self._pool.submit(_prefetch_run, request))
                yield _expand_packets(compact)
        finally:
            self.close()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

"""Teacher and Student training kernels.

Student recipe (V4.2). The student scores are bilinear forms on unit-norm Qwen features, so
raw scores live in a cosine-sized range (std ~0.06). Softmax losses on such scores are flat
(loss ~ log N), the KD target at temperature 1 is one-hot, and the only cheap way for the
optimiser to lower the loss is to inflate the common component of R, which destroys global
retrieval. The recipe therefore (1) multiplies student scores by ``logit_scale`` before any
softmax, (2) divides teacher logits by ``kd_temperature`` so the KD target keeps its ranking
information, (3) adds ``random_negatives`` uniformly drawn targets to every C2 direct list so
the global geometry is anchored, and (4) logs the top two singular values of ``R_QT - I`` so a
rank-one drift is visible in the training log.

Extensions, each a recipe switch so it can be ablated: a cosine learning-rate decay over the
stage; a per-list z-score of the teacher logits before the temperature; ``teacher_scored_negatives``
fixes the random negatives per query so the frozen teacher can score them and KD covers the same
list as SUP; ``kd_top_k`` restricts the KL to the teacher's top-k and only ranks the tail below
it; ``evidence_random_negatives`` gives that many random negatives a one-path bag with a random
evidence object, so the evidence list (Q-E and E-T relations) is anchored the same way.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from . import SCHEMA_VERSION
from .features import ObjectBank
from .labels import Labels
from .losses import (
    aggregate_corrected_lse,
    aggregate_cqet,
    hierarchical_relation_mean,
    hierarchical_support_mean,
    positive_average_pair_loss,
    rank_mass_loss,
    top_k_list_kd,
)
from .data import json_identity, utf8_sorted
from .models import RELATION_KINDS, TEACHER_INFERENCE_CHUNK, FreshPathTeacher, NativeStudent, QTStudent, SegmentCache, model_state_sha

# Teacher candidate chunk: the relation transformer batch, halved on OOM (whole logical batch replayed).
TEACHER_CHUNK_LADDER = (256, 128, 64, 32, 16, 8, 4, 2, 1)
# Student query microbatch, likewise halved on OOM.
STUDENT_MICROBATCH = 64
TEACHER_LAYOUT_REVISION = "v4_3_pooled_gather_inference1024_20261003"


def teacher_numerical_layout() -> dict:
    """Execution layout recorded in Teacher checkpoints and stage receipts (no model/loss effect)."""
    return {
        "logical_batch": 8,
        "candidate_chunk_ladder": list(TEACHER_CHUNK_LADDER),
        "teacher_scorer": "single_graph_first_query_cache_v1",
        "fallback": "same_batch_same_chunk_two_pass_then_halve",
        "retry_scope": "whole_logical_batch_before_optimizer_step",
        "initial_mode_each_batch": "single_graph",
        "inference_candidate_chunk": TEACHER_INFERENCE_CHUNK,
        "numerical_layout_revision": TEACHER_LAYOUT_REVISION,
    }


@dataclass(frozen=True)
class StudentRecipe:
    """Student optimisation and distillation settings; built from ``protocol["student"]``."""

    lr_p: float = 1e-4
    lr_r: float = 1e-3
    logit_scale: float = 20.0
    kd_weight: float = 1.0
    kd_temperature: float = 10.0
    random_negatives: int = 256
    anchor_weight: float = 0.0
    clip_norm: float = 1.0
    lr_schedule: str = "constant"  # or "cosine": lr * (1 + cos(pi * step / steps)) / 2 over the stage
    kd_normalization: str = "temperature"  # or "zscore": standardise each teacher list, then / temperature
    kd_top_k: int = 0  # > 0: KL on the teacher's top-k only, plus rank-mass of that set over the rest
    teacher_scored_negatives: bool = False  # fixed per-query negatives, scored by the teacher, inside KD
    evidence_random_negatives: int = 0  # random negatives that get a one-path bag with random evidence

    @classmethod
    def from_protocol(cls, protocol: Mapping, stage: str = "C2") -> "StudentRecipe":
        """Recipe of a Student stage: ``student.P_lr/R_lr`` unless the stage block (``student.C1`` /
        ``student.C2``) carries its own ``P_lr`` / ``R_lr``; everything else is shared."""
        student = protocol["student"]
        block = student.get(stage, {})
        return cls(
            lr_p=float(block.get("P_lr", student["P_lr"])),
            lr_r=float(block.get("R_lr", student["R_lr"])),
            logit_scale=float(student["logit_scale"]),
            kd_weight=float(student["kd_weight"]),
            kd_temperature=float(student["temperature"]),
            random_negatives=int(student["random_negatives"]),
            anchor_weight=float(student["anchor_weight"]),
            clip_norm=float(protocol.get("numerics", {}).get("grad_clip", 1.0)),
            lr_schedule=str(student["lr_schedule"]),
            kd_normalization=str(student["kd_normalization"]),
            kd_top_k=int(student["kd_top_k"]),
            teacher_scored_negatives=bool(student["teacher_scored_negatives"]),
            evidence_random_negatives=int(student["evidence_random_negatives"]),
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def lr_factor(self, step: int, total_steps: int) -> float:
        """Multiplier of the base learning rates for optimizer update ``step`` (0-based)."""
        if self.lr_schedule == "constant":
            return 1.0
        if self.lr_schedule == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
        raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")

    def kd_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Optional[Tensor]:
        """KD of one list: ``student_logits`` are already scaled by ``logit_scale``; the teacher
        list is optionally z-scored, divided by ``kd_temperature``, then matched in full or
        top-k focused (``losses.top_k_list_kd``)."""
        target = teacher_logits
        if self.kd_normalization == "zscore":
            target = (target - target.mean()) / target.std(unbiased=False).clamp_min(1e-6)
        elif self.kd_normalization != "temperature":
            raise ValueError(f"unknown kd_normalization {self.kd_normalization!r}")
        return top_k_list_kd(student_logits, target / self.kd_temperature, self.kd_top_k)


def enforce_task_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    if set(payload) != {"python", "numpy", "torch_cpu", "torch_cuda_all"}:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    cuda_state = payload["torch_cuda_all"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint requires CUDA RNG restoration")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])


def save_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, extra: Optional[dict] = None) -> None:
    """Full checkpoint: model, optimizer, RNG streams and ``extra`` plus the model state hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": _rng_state(),
        "extra": {**(extra or {}), "model_state_sha256": model_state_sha(model)},
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_training_checkpoint(path: Path, model: nn.Module, optimizer: AdamW, *, expected_stage: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    if extra.get("stage") != expected_stage:
        raise ValueError(f"resume stage mismatch: {extra.get('stage')} != {expected_stage}")
    if payload.get("optimizer") is None:
        raise ValueError("resume checkpoint has no optimizer state")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_state(payload["rng"])
    if model_state_sha(model) != extra.get("model_state_sha256"):
        raise ValueError("resume model state hash mismatch")
    return extra


def _record_id(row: dict) -> str:
    return str(row.get("record_id") or row.get("item_id") or row["query_id"])


def _hash_order(records: Sequence[dict], namespace: str, seed: int, epoch: int) -> list[dict]:
    """Protocol shuffle: records sorted by SHA256(namespace|seed|epoch|record id)."""
    return sorted(
        records,
        key=lambda row: (
            hashlib.sha256(f"{namespace}|{seed}|{epoch}|{_record_id(row)}".encode("utf-8")).digest(),
            _record_id(row).encode("utf-8"),
        ),
    )


def _order_sha(records: Sequence[dict]) -> str:
    return hashlib.sha256("\n".join(_record_id(row) for row in records).encode("utf-8")).hexdigest()


def _student_schedule(records: Sequence[dict], namespace: str, seed: int,
                      epochs: int, logical_batch: int) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """Shuffle each epoch separately and keep its final partial batch separate."""
    if epochs < 1 or logical_batch < 1:
        raise ValueError("epochs and logical_batch must be positive")
    ordered, batches = [], []
    for epoch in range(1, epochs + 1):
        offset = len(ordered)
        ordered.extend(_hash_order(records, namespace, seed, epoch))
        batches.extend((epoch, offset + start, offset + min(start + logical_batch, len(records)))
                       for start in range(0, len(records), logical_batch))
    return ordered, batches


def _student_snapshot_steps(total_steps: int, epochs: int) -> dict[int, list[float]]:
    """Step -> training fractions snapshotted there: quarters of one epoch, else epoch ends."""
    steps: dict[int, list[float]] = defaultdict(list)
    if epochs == 1:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            steps[math.ceil(total_steps * fraction)].append(fraction)
    else:
        for epoch in range(1, epochs + 1):
            steps[(total_steps // epochs) * epoch].append(epoch / epochs)
    return steps


def _snapshot_name(fraction: float, epoch: int, epochs: int) -> str:
    return f"snapshot_frac{int(fraction * 100):03d}.pt" if epochs == 1 else f"snapshot_epoch{epoch:03d}.pt"


def _log(path: Optional[Path], row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema_version": SCHEMA_VERSION, **row}, sort_keys=True) + "\n")


def _gpu_peaks() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"gpu_peak_allocated_bytes": 0, "gpu_peak_reserved_bytes": 0}
    return {
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _student_stats(student: NativeStudent | QTStudent) -> dict[str, float]:
    """P/R parameter norms and the top two singular values of ``R_QT - I`` (sigma1 >> sigma2 is
    the rank-one drift signature), read back from the device in one transfer."""
    with torch.no_grad():
        if isinstance(student, QTStudent):
            p_values, r_values, r_qt = [student.P_table], [student.R_QT], student.R_QT
        else:
            p_values, r_values, r_qt = list(student.P.values()), list(student.R.values()), student.R["QT"]
        p_norm = torch.sqrt(sum(torch.sum(value * value) for value in p_values))
        r_norm = torch.sqrt(sum(torch.sum(value * value) for value in r_values))
        sigma = torch.linalg.svdvals(r_qt - torch.eye(r_qt.shape[0], device=r_qt.device))[:2]
        values = torch.stack([p_norm, r_norm, sigma[0], sigma[1]]).tolist()
    return dict(zip(("P_parameter_norm", "R_parameter_norm", "sigma1_R_QT_minus_I", "sigma2_R_QT_minus_I"), values))


def _random_negative_ids(
    pool: Sequence[str], exclude: set[str], count: int, namespace: str, seed: int, epoch: int, query_id: str,
) -> list[str]:
    """Uniform negatives from ``pool``; deterministic per (seed, epoch, query) so resume replays them."""
    if count <= 0:
        return []
    digest = hashlib.sha256(f"{namespace}|{seed}|{epoch}|{query_id}|random_negatives".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    chosen: list[str] = []
    seen = set(exclude)
    while len(chosen) < count and len(seen) < len(pool):
        candidate = pool[rng.randrange(len(pool))]
        if candidate not in seen:
            seen.add(candidate)
            chosen.append(candidate)
    return chosen


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


# ------------------------------------------------------------------- Teacher --

def _torch_rng_state() -> tuple[Tensor, Optional[list[Tensor]]]:
    return (
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def _restore_torch_rng_state(state: tuple[Tensor, Optional[list[Tensor]]]) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


class TeacherListScorer:
    """Scores complete candidate lists of one query against a shared encoding cache.

    Rows are ``(kind, z, content)`` per slot (two slots for pairs, three for triplets) with a
    cache key per slot; an object is encoded once per query and reused by every list that
    touches it. Scores are computed in ``chunk``-sized relation batches, the OOM knob.

    ``single_graph`` keeps one autograd graph per query. ``two_pass`` (the OOM fallback) scores
    under ``no_grad`` and returns detached leaves; ``backward`` then re-encodes with grad,
    replays each chunk's dropout stream and back-propagates the chunk's adjoint, so only one
    chunk's activations are live at a time while the loss keeps the complete list.
    """

    def __init__(self, model: FreshPathTeacher, chunk: int, *, mode: str = "two_pass") -> None:
        if mode not in {"single_graph", "two_pass"}:
            raise ValueError(f"invalid Teacher backward mode: {mode}")
        if chunk not in TEACHER_CHUNK_LADDER:
            raise ValueError(f"invalid Teacher candidate chunk: {chunk}")
        self.model = model
        self.chunk = chunk
        self.mode = mode
        self.cache = SegmentCache()
        self.calls: list[tuple[str, list[tuple], list[tuple], tuple, Tensor]] = []
        self.scores: list[Tensor] = []

    def encode(self, rows: Sequence[tuple], keys: Sequence[tuple]) -> None:
        """Encode every object of ``rows`` not yet cached, batched per kind, tagged with the role
        its cache key carries (0 query, 1 target, 2 evidence)."""
        grouped: dict[str, list[tuple]] = defaultdict(list)
        pending: set[tuple] = set()
        for row, key in zip(rows, keys):
            for slot in range(len(row) // 3):
                ref = (row[3 * slot], key[slot])
                if ref not in self.cache and ref not in pending:
                    pending.add(ref)
                    grouped[ref[0]].append((ref, key[slot][1], row[3 * slot + 1], row[3 * slot + 2]))
        for kind, items in grouped.items():
            segments, lengths, globals_ = self.model.encode_many(
                kind, torch.stack([z for _, _, z, _ in items]), [c for _, _, _, c in items]
            )
            for role in sorted({role for _, role, _, _ in items}):
                positions = [i for i, item in enumerate(items) if item[1] == role]
                index = torch.tensor(positions, device=segments.device)
                self.cache.add(
                    self.model.tag(kind, role, segments[index], globals_[index]),
                    [lengths[i] + (role == 2) for i in positions], globals_[index], [items[i][0] for i in positions],
                )

    @staticmethod
    def _refs(rows: Sequence[tuple], keys: Sequence[tuple]) -> list[tuple]:
        return [tuple((row[3 * slot], key[slot]) for slot in range(len(row) // 3)) for row, key in zip(rows, keys)]

    def _score(self, kind: str, rows: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        if len(rows) != len(keys):
            raise ValueError("scorer rows/cache keys must have the same length")
        score = {"pairs": self.model.score_pairs, "triplets": self.model.score_triplets,
                 "path_pairs": self.model.score_path_pairs}[kind]
        if not rows:
            return torch.empty(0, device=next(self.model.parameters()).device)
        if self.mode == "single_graph":
            self.encode(rows, keys)
            result = torch.cat([
                score(self.cache, self._refs(rows[start : start + self.chunk], keys[start : start + self.chunk]))
                for start in range(0, len(rows), self.chunk)
            ])
            self.scores.append(result)
            return result
        with torch.no_grad():
            self.encode(rows, keys)
        outputs = []
        for start in range(0, len(rows), self.chunk):
            chunk_rows, chunk_keys = list(rows[start : start + self.chunk]), list(keys[start : start + self.chunk])
            rng = _torch_rng_state()
            with torch.no_grad():
                leaf = score(self.cache, self._refs(chunk_rows, chunk_keys)).detach().requires_grad_(True)
            self.calls.append((kind, chunk_rows, chunk_keys, rng, leaf))
            outputs.append(leaf)
        result = torch.cat(outputs)
        self.scores.append(result)
        return result

    def score_pairs(self, pairs: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        return self._score("pairs", pairs, keys)

    def score_triplets(self, triplets: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        return self._score("triplets", triplets, keys)

    def score_path_pairs(self, pairs: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        return self._score("path_pairs", pairs, keys)

    def backward(self, loss: Tensor, *, scale: float) -> None:
        if not self.scores:
            raise RuntimeError("Teacher list scorer has no scored lists")
        if self.mode == "single_graph":
            try:
                (loss * scale).backward()
            finally:
                self.cache = SegmentCache()
                self.scores.clear()
            return
        forward_end_rng = _torch_rng_state()
        adjoints = torch.autograd.grad(loss, [call[-1] for call in self.calls])
        self.cache = SegmentCache()
        for _kind, rows, keys, _rng, _leaf in self.calls:
            self.encode(rows, keys)
        for index, ((kind, rows, keys, rng, _leaf), adjoint) in enumerate(zip(self.calls, adjoints)):
            _restore_torch_rng_state(rng)
            score = {"pairs": self.model.score_pairs, "triplets": self.model.score_triplets,
                     "path_pairs": self.model.score_path_pairs}[kind]
            # Chunks share the encoder outputs, so every chunk but the last must retain them.
            score(self.cache, self._refs(rows, keys)).backward(
                adjoint * scale, retain_graph=index + 1 < len(self.calls)
            )
        _restore_torch_rng_state(forward_end_rng)
        self.calls.clear()
        self.cache = SegmentCache()
        self.scores.clear()


def path_scores(
    scorer: TeacherListScorer,
    bank: ObjectBank,
    query_id: str,
    query_tokens: Tensor,
    pairs: Sequence[tuple[str, str]],
    tokens: Mapping[str, Tensor],
    base: Optional[Tensor] = None,
) -> Tensor:
    """Path scores ``(N,)`` of the ``(evidence, target)`` pairs of ``query_id``.

    ``triplet`` Teachers score the QET triplet directly. ``pairwise_residual`` Teachers return
    ``base + path(q, e) + path(e, t)`` where ``base`` is f0(q, t) per pair; callers that only rank
    evidence for one fixed (q, t) pass ``base=None`` and get the evidence-dependent residual.
    """
    model = scorer.model
    zq = bank.z(query_id)
    if model.path_mode == "triplet":
        return scorer.score_triplets(
            [("table", zq, query_tokens, bank.kind(e), bank.z(e), tokens[e], "table", bank.z(t), tokens[t]) for e, t in pairs],
            [((query_id, 0), (e, 2), (t, 1)) for e, t in pairs],
        )
    evidence = list(dict.fromkeys(e for e, _ in pairs))
    position = {e: i for i, e in enumerate(evidence)}
    qe = scorer.score_path_pairs(
        [("table", zq, query_tokens, bank.kind(e), bank.z(e), tokens[e]) for e in evidence],
        [((query_id, 0), (e, 2)) for e in evidence],
    )
    et = scorer.score_path_pairs(
        [(bank.kind(e), bank.z(e), tokens[e], "table", bank.z(t), tokens[t]) for e, t in pairs],
        [((e, 2), (t, 1)) for e, t in pairs],
    )
    delta = qe[torch.tensor([position[e] for e, _ in pairs], dtype=torch.long, device=qe.device)] + et
    return delta if base is None else base + delta


def _support_loss(
    bank: ObjectBank,
    query_id: str,
    records: Sequence[dict],
    query_tokens: Tensor,
    scorer: TeacherListScorer,
    tokens: Mapping[str, Tensor],
) -> Optional[Tensor]:
    """Same-modality witness-vs-competitor loss per (target, modality), averaged per target."""
    by_target: dict[str, list[Tensor]] = defaultdict(list)
    for row in records:
        target_id = row["target_id"]
        positives, competitors = list(row["positives"]), list(row["competitors"])
        if not positives or not competitors:
            continue
        scores = [path_scores(scorer, bank, query_id, query_tokens, [(e, target_id) for e in ids], tokens)
                  for ids in (positives, competitors)]
        loss = positive_average_pair_loss(*scores)
        if loss is not None:
            by_target[target_id].append(loss)
    return hierarchical_support_mean([by_target[t] for t in sorted(by_target, key=lambda x: x.encode("utf-8"))])


def _witness_target_loss(
    scorer: TeacherListScorer,
    bank: ObjectBank,
    query_id: str,
    query_tokens: Tensor,
    qet_lists: Sequence[dict],
    tokens: Mapping[str, Tensor],
    f0_of: Mapping[str, Tensor],
    dev: torch.device,
) -> dict[str, list[Tensor]]:
    """Rank-mass of the path score ``(q, witness, t)`` over each witness list's candidate targets,
    keyed by ``QET_<modality>``; ``f0_of`` maps every candidate to its direct score."""
    by_relation: dict[str, list[Tensor]] = defaultdict(list)
    for item in qet_lists:
        evidence_id, candidates = item["evidence_id"], list(item["candidates"])
        ignore, positives = set(item.get("ignore", ())), set(item["positives"])
        base = torch.stack([f0_of[t] for t in candidates]) if scorer.model.path_mode != "triplet" else None
        scores = path_scores(scorer, bank, query_id, query_tokens, [(evidence_id, t) for t in candidates], tokens, base)
        loss = rank_mass_loss(scores, torch.tensor([t in positives for t in candidates], device=dev),
                              torch.tensor([t not in ignore for t in candidates], device=dev))
        if loss is not None:
            by_relation[f"QET_{item['evidence_kind']}"].append(loss)
    return by_relation


def _support_object_ids(records: Sequence[dict]) -> list[str]:
    return list(dict.fromkeys(
        object_id for record in records
        for object_id in [record["target_id"], *record["positives"], *record["competitors"]]
    ))


def _ta_query_backward(model, bank, row, labels, dev, candidate_chunk,
                       backward_mode, scale, support_weight=0.2) -> dict:
    """T_A: rank-mass loss per relation list (QT, Q_text, Q_image, QET per witness) plus support."""
    qid = row["query_id"]
    zq = bank.z(qid)
    scorer = TeacherListScorer(model, candidate_chunk, mode=backward_mode)
    support_records = row.get("support_records", [])
    qet_lists = row.get("qet_lists", [])
    needed = [qid, *row["qt_candidates"], *row["qe_candidates"]["text"], *row["qe_candidates"]["image"]]
    for item in qet_lists:
        needed += [item["evidence_id"], *item["candidates"]]
    needed += _support_object_ids(support_records)
    needed = list(dict.fromkeys(needed))
    tokens = dict(zip(needed, bank.tokens_many(needed)))

    by_relation: dict[str, list[Tensor]] = {r: [] for r in ("QT", "Q_text", "Q_image", "QET_text", "QET_image")}
    qt = list(row["qt_candidates"])
    # pairwise_residual path scores need f0 of every QET candidate: score the union once.
    scored = qt if model.path_mode == "triplet" else list(dict.fromkeys([*qt, *(t for item in qet_lists for t in item["candidates"])]))
    f0_all = scorer.score_pairs([("table", zq, tokens[qid], "table", bank.z(t), tokens[t]) for t in scored],
                                [((qid, 0), (t, 1)) for t in scored])
    f0_of = dict(zip(scored, f0_all))
    loss = rank_mass_loss(f0_all[: len(qt)], torch.tensor([t in set(labels.queries[qid]["G"]) for t in qt], device=dev))
    if loss is not None:
        by_relation["QT"].append(loss)
    for modality in ("text", "image"):
        positives = set(labels.queries[qid]["Qpos"][modality])
        if positives:
            candidates = list(row["qe_candidates"][modality])
            scores = scorer.score_pairs([("table", zq, tokens[qid], modality, bank.z(e), tokens[e]) for e in candidates],
                                        [((qid, 0), (e, 1)) for e in candidates])
            loss = rank_mass_loss(scores, torch.tensor([e in positives for e in candidates], device=dev))
            if loss is not None:
                by_relation[f"Q_{modality}"].append(loss)
    for relation, losses in _witness_target_loss(scorer, bank, qid, tokens[qid], qet_lists, tokens, f0_of, dev).items():
        by_relation[relation].extend(losses)

    relation_loss = hierarchical_relation_mean(list(by_relation.values()))
    support = _support_loss(bank, qid, support_records, tokens[qid], scorer, tokens)
    query_loss = relation_loss
    if support is not None:
        query_loss = support_weight * support if query_loss is None else query_loss + support_weight * support
    if query_loss is None:
        raise RuntimeError(f"TA query {qid} has no active loss")
    scorer.backward(query_loss, scale=scale)
    return {
        "loss": float(query_loss.detach()),
        "relation": None if relation_loss is None else float(relation_loss.detach()),
        "support": None if support is None else float(support.detach()),
        "active_relations": sum(bool(values) for values in by_relation.values()),
    }


def _tb_query_backward(model, bank, row, dev, candidate_chunk, backward_mode, scale, mode, *,
                       direct_weight=0.5, aggregate_weight=0.5, support_weight=0.2,
                       path_loss_scope="all", witness_target_weight=0.0) -> dict:
    """T_B: direct rank-mass on the shared candidate list, plus (cqet/lse) the aggregated-path
    rank-mass over natural bags (over every target, or ``path_loss_scope="bagged"``: only the
    targets that have a bag, so bag presence itself carries no supervision), the support loss and,
    with ``witness_target_weight``, the witness-conditioned target lists of ``row["qet_lists"]``."""
    qid = row["query_id"]
    targets = list(row["targets"])
    positives = set(row["positives"])
    pos_mask = torch.tensor([t in positives for t in targets], device=dev)
    zq = bank.z(qid)
    qet_lists = row.get("qet_lists", []) if mode != "qt" and witness_target_weight else []
    all_ids = [qid, *targets]
    if mode != "qt":
        all_ids += [e for t in targets for e in row["natural_bags"].get(t, ())]
        all_ids += _support_object_ids(row.get("support_records", []))  # tokens only; never in a ranking bag
        all_ids += [x for item in qet_lists for x in (item["evidence_id"], *item["candidates"])]
    all_ids = list(dict.fromkeys(all_ids))
    tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
    cq = tokens[qid]
    scorer = TeacherListScorer(model, candidate_chunk, mode=backward_mode)
    scored = targets if model.path_mode == "triplet" else list(dict.fromkeys(
        [*targets, *(t for item in qet_lists for t in item["candidates"])]))
    f0_all = scorer.score_pairs(
        [("table", zq, cq, "table", bank.z(t), tokens[t]) for t in scored],
        [((qid, 0), (t, 1)) for t in scored],
    )
    f0 = f0_all[: len(targets)]
    direct = rank_mass_loss(f0, pos_mask)
    path_loss = support = witness = None
    if mode == "qt":
        query_loss = direct
    else:
        paths = [(i, e) for i, t in enumerate(targets) for e in row["natural_bags"].get(t, ())]
        target_index = torch.tensor([i for i, _ in paths], dtype=torch.long, device=dev)
        scores = path_scores(scorer, bank, qid, cq, [(e, targets[i]) for i, e in paths], tokens,
                             None if model.path_mode == "triplet" else f0[target_index])
        aggregate = aggregate_cqet if mode == "cqet" else aggregate_corrected_lse
        bagged = torch.bincount(target_index, minlength=len(targets)) > 0 if path_loss_scope == "bagged" else None
        path_loss = rank_mass_loss(aggregate(f0, scores, target_index), pos_mask, bagged)
        support = _support_loss(bank, qid, row.get("support_records", []), cq, scorer, tokens)
        if qet_lists:
            witness = hierarchical_relation_mean(list(_witness_target_loss(
                scorer, bank, qid, cq, qet_lists, tokens, dict(zip(scored, f0_all)), dev).values()))
        terms = [weight * loss for weight, loss in (
            (direct_weight, direct), (aggregate_weight, path_loss), (support_weight, support),
            (witness_target_weight, witness)) if loss is not None]
        query_loss = sum(terms[1:], terms[0]) if terms else None
    if query_loss is None:
        raise RuntimeError(f"TB_{mode.upper()} query {qid} has no active loss")
    scorer.backward(query_loss, scale=scale)
    return {
        "loss": float(query_loss.detach()),
        "direct": None if direct is None else float(direct.detach()),
        "path": None if path_loss is None else float(path_loss.detach()),
        "support": None if support is None else float(support.detach()),
        "witness_target": None if witness is None else float(witness.detach()),
    }


def _run_teacher_logical_batch(optimizer, batch, query_fn, *, candidate_chunk: int,
                               initial_mode: str = "single_graph") -> tuple:
    """Accumulate all query gradients transactionally; NEVER update optimizer here.

    A failed attempt discards ALL gradients from the logical batch, restores Python/NumPy/
    CPU/CUDA RNG and replays every query. Single-graph failure first retries two-pass at the
    SAME chunk; only then reduces the chunk.
    """
    if not batch:
        raise ValueError("empty logical Teacher batch")
    if initial_mode not in {"single_graph", "two_pass"}:
        raise ValueError(initial_mode)
    if candidate_chunk not in TEACHER_CHUNK_LADDER:
        raise ValueError(candidate_chunk)
    batch_rng = _rng_state()
    backward_mode = initial_mode
    events = []
    while True:
        optimizer.zero_grad(set_to_none=True)
        metrics = []
        try:
            for row in batch:
                result = query_fn(row, candidate_chunk, backward_mode, 1.0 / len(batch))
                if not math.isfinite(float(result["loss"])):
                    raise FloatingPointError("nonfinite Teacher loss; no optimizer step executed")
                metrics.append(result)
            return metrics, candidate_chunk, backward_mode, events
        except BaseException as error:
            if not _is_cuda_oom(error):
                optimizer.zero_grad(set_to_none=True)
                raise
            # Do not run the retry while an exception traceback still owns the
            # failed forward/backward's tensors. empty_cache alone cannot free them.
            events.append({"mode": backward_mode, "chunk": candidate_chunk,
                           "completed_queries_discarded": len(metrics), "error": type(error).__name__})
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        _restore_rng_state(batch_rng)
        if backward_mode == "single_graph":
            backward_mode = "two_pass"
        elif candidate_chunk > 1:
            candidate_chunk = TEACHER_CHUNK_LADDER[TEACHER_CHUNK_LADDER.index(candidate_chunk) + 1]
        else:
            raise RuntimeError("BLOCKED_RESOURCE: Teacher two-pass chunk 1 OOM; no optimizer update")


def _train_teacher(
    model: FreshPathTeacher,
    trainable: list[nn.Parameter],
    records: list[dict],
    query_fn: Callable,
    *,
    stage: str,
    namespace: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    logical_batch: int,
    seed: int,
    save_dir: Path,
    snapshot: Callable[[int, int, int], Optional[str]],
    metadata: Optional[dict],
    log_path: Optional[Path],
    log_extra: dict,
) -> None:
    """Shared Teacher loop: hash-ordered epochs, transactional logical batches, clip, step, log.

    ``snapshot(epoch, step, total_steps)`` names the checkpoint to save after a step (or None);
    ``query_fn(row, chunk, mode, scale)`` scores one query, back-propagates and returns its
    metrics, whose non-``loss`` keys are averaged over the queries that produced them.
    """
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    total_steps = math.ceil(len(records) / logical_batch) * epochs
    base_meta = {**(metadata or {}), "numerical_layout": teacher_numerical_layout()}
    save_checkpoint(save_dir / "init.pt", model, optimizer,
                    {**base_meta, "stage": stage, "epoch": 0, "logical_step": 0, "next_record_cursor": 0})
    logical_step = 0
    started = time.time()
    candidate_chunk = TEACHER_CHUNK_LADDER[0]
    for epoch in range(1, epochs + 1):
        ordered = _hash_order(records, namespace, seed, epoch)
        model.train()
        for start in range(0, len(ordered), logical_batch):
            batch = ordered[start : start + logical_batch]
            metrics, candidate_chunk, backward_mode, oom_events = _run_teacher_logical_batch(
                optimizer, batch, query_fn, candidate_chunk=candidate_chunk,
            )
            batch_loss = sum(m["loss"] for m in metrics) / len(batch)
            grad_norm = float(nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
            optimizer.step()
            logical_step += 1
            components = {}
            for key in metrics[0]:
                if key == "loss":
                    continue
                values = [m[key] for m in metrics if m[key] is not None]
                components[key] = float(np.mean(values)) if values else None
                components[f"{key}_denominator"] = len(values)
            _log(log_path, {
                "stage": stage, "epoch": epoch, "step": logical_step,
                "record_ids": [row["query_id"] for row in batch],
                "batch_queries": len(batch), "loss": batch_loss,
                "grad_norm_preclip": grad_norm, "grad_norm_postclip": min(grad_norm, 1.0), "clip_norm": 1.0,
                "lr": lr, "teacher_candidate_chunk": candidate_chunk, "teacher_backward_mode": backward_mode,
                "numerical_layout_revision": TEACHER_LAYOUT_REVISION, "oom_events": oom_events,
                **components, **log_extra, "elapsed_seconds": time.time() - started, **_gpu_peaks(),
            })
            print(f"[{stage}] epoch={epoch}/{epochs} step={logical_step}/{total_steps} "
                  f"loss={batch_loss:.5f} elapsed={time.time()-started:.1f}s", flush=True)
            name = snapshot(epoch, logical_step, total_steps)
            if name is not None:
                end_of_epoch = start + logical_batch >= len(ordered)
                save_checkpoint(save_dir / name, model, optimizer, {
                    **base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                    "next_record_cursor": 0 if end_of_epoch else start + logical_batch,
                    "order_sha256": _order_sha(ordered), "teacher_candidate_chunk": candidate_chunk,
                })


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
    save_dir: Path = Path("TA"),
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """T_A over all parameters; checkpoints ``init.pt`` and ``epoch<n>.pt``; returns the last epoch."""
    enforce_task_numerics()
    dev = torch.device(device)
    model.to(dev)
    bank.attach_device(dev)
    steps_per_epoch = math.ceil(len(ta_records) / logical_batch)
    _train_teacher(
        model, list(model.parameters()), ta_records,
        lambda row, chunk, mode, scale: _ta_query_backward(model, bank, row, labels, dev, chunk, mode, scale, support_weight),
        stage="TA", namespace="TA", epochs=epochs, lr=lr, weight_decay=weight_decay,
        logical_batch=logical_batch, seed=seed, save_dir=save_dir,
        snapshot=lambda epoch, step, total: f"epoch{epoch}.pt" if step % steps_per_epoch == 0 else None,
        metadata=metadata, log_path=log_path, log_extra={},
    )
    return save_dir / f"epoch{epochs}.pt"


def train_tb(
    model: FreshPathTeacher,
    bank: ObjectBank,
    tb_records: list[dict],
    device: str = "cuda:0",
    mode: str = "cqet",
    epochs: int = 1,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    logical_batch: int = 8,
    direct_weight: float = 0.5,
    aggregate_weight: float = 0.5,
    support_weight: float = 0.2,
    path_loss_scope: str = "all",
    witness_target_weight: float = 0.0,
    save_dir: Path = Path("TB"),
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> Path:
    """T_B (cqet / lse / qt) over the relation head only; checkpoints ``init/half/end.pt``."""
    if mode not in {"cqet", "lse", "qt"}:
        raise ValueError("TB mode must be cqet, lse, or qt")
    if path_loss_scope not in {"all", "bagged"}:
        raise ValueError("path_loss_scope must be all or bagged")
    enforce_task_numerics()
    dev = torch.device(device)
    model.to(dev)
    bank.attach_device(dev)
    total_steps = math.ceil(len(tb_records) / logical_batch) * epochs
    half_step = math.ceil(0.5 * total_steps)
    _train_teacher(
        model, model.set_tb_trainable(), tb_records,
        lambda row, chunk, mode_, scale: _tb_query_backward(
            model, bank, row, dev, chunk, mode_, scale, mode, direct_weight=direct_weight,
            aggregate_weight=aggregate_weight, support_weight=support_weight,
            path_loss_scope=path_loss_scope, witness_target_weight=witness_target_weight),
        stage=f"TB_{mode.upper()}", namespace="TB_SHARED", epochs=epochs, lr=lr, weight_decay=weight_decay,
        logical_batch=logical_batch, seed=seed, save_dir=save_dir,
        snapshot=lambda epoch, step, total: "half.pt" if step == half_step else "end.pt" if step == total else None,
        metadata=metadata, log_path=log_path,
        log_extra={"aggregation": mode, "path_mode": model.path_mode, "path_loss_scope": path_loss_scope,
                   "witness_target_weight": witness_target_weight},
    )
    return save_dir / "end.pt"


# ------------------------------------------------------------------- Student --

def _student_scores(student: NativeStudent | QTStudent, bank: ObjectBank, relation: str,
                    anchor: Tensor, candidates: Sequence[str]) -> Tensor:
    """Raw bilinear scores of ``anchor`` against ``candidates`` under one relation."""
    zb = bank.z_many(candidates)
    if isinstance(student, QTStudent):
        return student.score(anchor, zb)
    left_kind, right_kind = RELATION_KINDS[relation]
    return student.score(left_kind, anchor, right_kind, zb)


def _student_c1_batch_scores(
    student: NativeStudent | QTStudent, bank: ObjectBank, rows: Sequence[dict],
) -> list[Tensor]:
    """Project each distinct object once per microbatch; keep every edge list's order.

    Projections have no dropout. Reusing their graph sums all uses' gradients into the same
    trainable P, as in C2; no representation survives an optimizer step.
    """
    positions: dict[str, dict[str, int]] = {}
    for row in rows:
        left, right = RELATION_KINDS[row["relation"]]
        for kind, ids in ((left, [row["anchor_id"]]), (right, row["candidates"])):
            index = positions.setdefault(kind, {})
            for object_id in ids:
                if object_id not in index:
                    index[object_id] = len(index)
    projected = {}
    for kind, index in positions.items():
        z = bank.z_many(list(index))
        projected[kind] = student.u(z) if isinstance(student, QTStudent) else student.u(kind, z)
    scores = []
    for row in rows:
        relation = row["relation"]
        left, right = RELATION_KINDS[relation]
        anchor = projected[left][positions[left][row["anchor_id"]]]
        index = torch.tensor([positions[right][c] for c in row["candidates"]],
                             dtype=torch.long, device=anchor.device)
        candidates = projected[right][index]
        matrix = student.R_QT if isinstance(student, QTStudent) else student.R[relation]
        scores.append((anchor @ matrix * candidates).sum(dim=-1))
    return scores


def _recipe_log(recipe: StudentRecipe, **overrides) -> dict:
    """Recipe fields as written into stage logs and checkpoints (``P_lr``/``R_lr`` naming)."""
    values = {key: value for key, value in recipe.as_dict().items() if key not in ("lr_p", "lr_r")}
    return {"P_lr": recipe.lr_p, "R_lr": recipe.lr_r, **values, **overrides}


def _set_scheduled_lr(optimizer: AdamW, recipe: StudentRecipe, step: int, total_steps: int) -> float:
    """Set the P/R group learning rates for update ``step`` (0-based); a pure function of the
    step, so a resumed run follows the same schedule without scheduler state."""
    factor = recipe.lr_factor(step, total_steps)
    for group, base in zip(optimizer.param_groups, (recipe.lr_p, recipe.lr_r)):
        group["lr"] = base * factor
    return factor


def _student_step(
    student: NativeStudent | QTStudent,
    optimizer: AdamW,
    batch: Sequence[dict],
    micro_losses: Callable[[Sequence[dict]], tuple[list[Tensor], dict[str, list[Tensor]]]],
    *,
    anchor_weight: float,
    stage: str,
) -> tuple[int, float, dict[str, list[float]], float]:
    """One optimizer step over ``batch`` with the OOM microbatch ladder.

    ``micro_losses(rows)`` returns each row's loss and the microbatch's named loss components;
    a CUDA OOM restores the RNG, halves the query microbatch and replays the whole batch.
    Returns the microbatch used, the mean row loss, the per-component values and the weighted
    anchor loss.
    """
    batch_rng = _rng_state()
    microbatch = STUDENT_MICROBATCH
    while True:
        optimizer.zero_grad(set_to_none=True)
        totals: list[Tensor] = []
        components: dict[str, list[Tensor]] = defaultdict(list)
        try:
            for micro_start in range(0, len(batch), microbatch):
                losses, parts = micro_losses(batch[micro_start : micro_start + microbatch])
                for name, values in parts.items():
                    components[name].extend(value.detach() for value in values)
                micro_sum = torch.stack(losses).sum()
                totals.append(micro_sum.detach())
                (micro_sum / len(batch)).backward()
            anchor = anchor_weight * student.anchor_loss()
            if anchor_weight:
                anchor.backward()
            break
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            if microbatch == 1:
                raise RuntimeError(f"BLOCKED_RESOURCE: {stage} query microbatch 1 OOM") from error
            optimizer.zero_grad(set_to_none=True)
            _restore_rng_state(batch_rng)
            torch.cuda.empty_cache()
            microbatch //= 2
    mean_loss = float(torch.stack(totals).sum()) / len(batch)
    values = {name: torch.stack(items).tolist() for name, items in components.items()}
    return microbatch, mean_loss, values, float(anchor.detach())


def train_student_c1(
    student: NativeStudent | QTStudent,
    edge_lists: list[dict],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_SUP",
    logical_batch: int = 64,
    recipe: StudentRecipe = StudentRecipe(),
    save_dir: Optional[Path] = None,
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
    epochs: int = 1,
) -> dict[float, Path]:
    """C1: rank-mass loss over each edge list's candidates (all five relations for the Native
    Student, QT only for the QT Student). Returns the snapshot checkpoints by training fraction."""
    if "KD" in arm:
        raise ValueError("C1 has no KD branch")
    enforce_task_numerics()
    dev = torch.device(device)
    student.to(dev).train()
    bank.attach_device(dev)
    is_qt = isinstance(student, QTStudent)
    valid_lists = [row for row in edge_lists if not is_qt or row["relation"] == "QT"]
    ordered, batches = _student_schedule(valid_lists, "C1_QT" if is_qt else "C1_NATIVE", seed, epochs, logical_batch)
    optimizer = AdamW(student.param_groups(recipe.lr_p, recipe.lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total_steps = len(batches)
    fraction_steps = _student_snapshot_steps(total_steps, epochs)
    stage = "QT_C1_SUP" if is_qt else "NATIVE_C1_SUP"
    recipe_log = _recipe_log(recipe)
    base_meta = {**(metadata or {}), "epochs": epochs, "logical_batch": logical_batch, **recipe_log}

    def edge_losses(rows: Sequence[dict]) -> tuple[list[Tensor], dict[str, list[Tensor]]]:
        losses = []
        for row, raw in zip(rows, _student_c1_batch_scores(student, bank, rows)):
            candidates = list(row["candidates"])
            positives = set(row["positives"])
            loss = rank_mass_loss(recipe.logit_scale * raw, torch.tensor([c in positives for c in candidates], device=dev))
            if loss is None:
                raise RuntimeError(f"{stage}: prevalidated active list became inactive")
            losses.append(loss)
        return losses, {}

    saved: dict[float, Path] = {}
    if save_dir:
        saved[0.0] = save_dir / "snapshot_frac000.pt"
        save_checkpoint(saved[0.0], student, optimizer,
                        {**base_meta, "stage": stage, "epoch": 0, "logical_step": 0, "next_record_cursor": 0,
                         "order_sha256": _order_sha(ordered), "student_query_microbatch": STUDENT_MICROBATCH})
    logical_step = 0
    started = time.time()
    for epoch, start, end in batches:
        batch = ordered[start:end]
        active_batch = []
        for row in batch:
            positives = set(row["positives"])
            if any(c in positives for c in row["candidates"]) and any(c not in positives for c in row["candidates"]):
                active_batch.append(row)
        if not active_batch:
            continue
        microbatch, rank_loss, _components, anchor_value = _student_step(
            student, optimizer, active_batch, edge_losses, anchor_weight=recipe.anchor_weight, stage=stage,
        )
        grad_norm = float(nn.utils.clip_grad_norm_(student.parameters(), recipe.clip_norm))
        lr_factor = _set_scheduled_lr(optimizer, recipe, logical_step, total_steps)
        optimizer.step()
        logical_step += 1
        _log(log_path, {
            "stage": stage, "epoch": epoch, "step": logical_step,
            "record_ids": [row["item_id"] for row in batch],
            "active_lists": len(active_batch), "batch_lists": len(batch),
            "loss": rank_loss + anchor_value, "rank_mass_loss": rank_loss, "anchor_loss_weighted": anchor_value,
            "grad_norm_preclip": grad_norm, "grad_norm_postclip": min(grad_norm, recipe.clip_norm),
            **recipe_log, "lr_factor": lr_factor, "student_query_microbatch": microbatch,
            "elapsed_seconds": time.time() - started, **_student_stats(student), **_gpu_peaks(),
        })
        if logical_step in fraction_steps and save_dir:
            for fraction in fraction_steps[logical_step]:
                path = save_dir / _snapshot_name(fraction, epoch, epochs)
                save_checkpoint(path, student, optimizer, {
                    **base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                    "next_record_cursor": end, "order_sha256": _order_sha(ordered), "fraction": fraction,
                    "student_query_microbatch": microbatch,
                })
                saved[fraction] = path
    if logical_step != total_steps:
        raise RuntimeError(f"{stage}: inactive batch changed registered step count")
    return saved


def _student_c2_batch_scores(
    student: NativeStudent | QTStudent,
    bank: ObjectBank,
    rows: Sequence[dict],
    logit_scale: float = 1.0,
) -> list[tuple[Tensor, Optional[Tensor], list[tuple[int, str]]]]:
    """Scaled student logits for a batch of C2 records; every distinct object is projected once.

    Per record ``(direct, evidence, bag_targets)``: ``direct`` is one logit per target,
    ``evidence`` one CQET-aggregated logit per target with a non-empty bag (None if no bags).
    Every bilinear score is multiplied by ``logit_scale``; a path logit is the sum of the scaled
    Q-E and E-T scores, gathered with index tensors rather than a per-path Python loop.
    """
    tables = list(dict.fromkeys(i for row in rows for i in (row["query_id"], *row["targets"])))
    table_pos = {object_id: i for i, object_id in enumerate(tables)}
    if isinstance(student, QTStudent):
        u = student.u(bank.z_many(tables))
        result = []
        for row in rows:
            index = torch.tensor([table_pos[t] for t in row["targets"]], dtype=torch.long, device=u.device)
            uq = u[table_pos[row["query_id"]]]
            result.append((logit_scale * (uq @ student.R_QT * u[index]).sum(dim=-1), None, []))
        return result
    u_table = student.u("table", bank.z_many(tables))
    dev = u_table.device
    evidence = list(dict.fromkeys(
        e for row in rows for t in row["targets"] for e in row.get("natural_bags", {}).get(t, ())
    ))
    modalities = [m for m in ("text", "image") if any(bank.kind(e) == m for e in evidence)]
    by_modality = {m: [e for e in evidence if bank.kind(e) == m] for m in modalities}
    evidence_pos = {e: i for i, e in enumerate(e for m in modalities for e in by_modality[m])}
    if evidence:
        u_parts = [student.u(m, bank.z_many(by_modality[m])) for m in modalities]
        u_evidence = torch.cat(u_parts)
        et_projected = torch.cat([part @ student.R[f"{m}_T"] for m, part in zip(modalities, u_parts)])
        evidence_relation = torch.tensor(
            [k for k, m in enumerate(modalities) for _ in by_modality[m]], dtype=torch.long, device=dev,
        )
    result = []
    for row in rows:
        targets = row["targets"]
        uq = u_table[table_pos[row["query_id"]]]
        ut = u_table[torch.tensor([table_pos[t] for t in targets], dtype=torch.long, device=dev)]
        direct = logit_scale * (uq @ student.R["QT"] * ut).sum(dim=-1)
        bags = row.get("natural_bags", {})
        bag_targets = [(i, t) for i, t in enumerate(targets) if bags.get(t)]
        flat = [(target_i, e) for target_i, target in bag_targets for e in bags[target]]
        if not flat:
            result.append((direct, None, bag_targets))
            continue
        path_evidence = torch.tensor([evidence_pos[e] for _, e in flat], dtype=torch.long, device=dev)
        path_target = torch.tensor([target_i for target_i, _ in flat], dtype=torch.long, device=dev)
        uq_relations = torch.stack([uq @ student.R[f"Q_{m}"] for m in modalities])
        qe = (uq_relations[evidence_relation[path_evidence]] * u_evidence[path_evidence]).sum(dim=-1)
        path_scores = logit_scale * (qe + (et_projected[path_evidence] * ut[path_target]).sum(dim=-1))
        evidence_all_targets = aggregate_cqet(direct, path_scores, path_target)
        bag_index = torch.tensor([target_i for target_i, _ in bag_targets], dtype=torch.long, device=dev)
        result.append((direct, evidence_all_targets[bag_index], bag_targets))
    return result


def _student_c2_scores(
    student: NativeStudent | QTStudent,
    bank: ObjectBank,
    row: dict,
    logit_scale: float = 1.0,
) -> tuple[Tensor, Optional[Tensor], list[tuple[int, str]]]:
    """``_student_c2_batch_scores`` of a single C2 record."""
    return _student_c2_batch_scores(student, bank, [row], logit_scale)[0]


def c2_training_row(
    row: dict,
    recipe: StudentRecipe,
    negative_pool: Sequence[str],
    evidence_pool: Sequence[str],
    seed: int,
    epoch: int,
) -> dict:
    """The C2 record a Student trains on in ``epoch``.

    Targets are the shared-graph targets followed by ``random_negatives`` uniform draws from
    ``negative_pool``; the first ``evidence_random_negatives`` of those draws get a one-path
    bag with a uniform draw from ``evidence_pool``, so they also enter the evidence list.
    Draws are keyed by (seed, epoch, query), or by (seed, query) alone under
    ``teacher_scored_negatives`` so that the frozen teacher can score the same list once.
    """
    epoch = 0 if recipe.teacher_scored_negatives else epoch
    query_id = row["query_id"]
    bags = row.get("natural_bags", {})
    negatives = _random_negative_ids(
        negative_pool, set(row["targets"]) | set(row["positives"]), recipe.random_negatives,
        "C2_SHARED", seed, epoch, query_id,
    )
    if not negatives:
        return row
    evidence = _random_negative_ids(
        evidence_pool, {e for t in row["targets"] for e in bags.get(t, ())},
        min(recipe.evidence_random_negatives, len(negatives)), "C2_SHARED_EVIDENCE", seed, epoch, query_id,
    )
    return {
        **row,
        "targets": [*row["targets"], *negatives],
        "natural_bags": {**bags, **{t: [e] for t, e in zip(negatives, evidence)}},
    }


def c2_teacher_rows(
    records: Sequence[dict],
    recipe: StudentRecipe,
    negative_pool: Sequence[str],
    evidence_pool: Sequence[str],
    seed: int,
) -> list[dict]:
    """The lists the frozen teacher scores for KD: the shared-graph records, or under
    ``teacher_scored_negatives`` the fixed extended records every C2 arm trains on."""
    if not recipe.teacher_scored_negatives:
        return list(records)
    return [c2_training_row(row, recipe, negative_pool, evidence_pool, seed, epoch=0) for row in records]


def _scored_list_sha(row: dict) -> str:
    """Identity of the direct and evidence lists of a C2 record (targets and their bags, in order)."""
    bags = row.get("natural_bags", {})
    return json_identity([list(row["targets"]), [list(bags.get(t, ())) for t in row["targets"]]])


def build_teacher_logits_cache(
    teacher: FreshPathTeacher,
    bank: ObjectBank,
    c2_records: Sequence[dict],
    *,
    device: str = "cuda:0",
) -> dict[str, dict]:
    """Frozen TB_CQET logits per C2 record (CPU float32), aligned with ``_student_c2_scores``:
    ``{"direct": one logit per target, "evidence": one aggregated logit per bag target or None,
    "list_sha256": identity of the scored lists}``."""
    dev = torch.device(device)
    teacher.to(dev).eval()
    teacher.requires_grad_(False)
    bank.attach_device(dev)
    result = {}
    started = time.time()
    with torch.no_grad():
        for i, row in enumerate(c2_records, 1):
            qid, targets, bags = row["query_id"], list(row["targets"]), row.get("natural_bags", {})
            paths = [(k, e) for k, t in enumerate(targets) for e in bags.get(t, ())]
            evidence_ids = list(dict.fromkeys(e for _, e in paths))
            all_ids = list(dict.fromkeys([qid, *targets, *evidence_ids]))
            tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
            direct, path_scores = teacher.score_query_lists(
                (bank.z(qid), tokens[qid]), (bank.z_many(targets), [tokens[t] for t in targets]),
                {e: (bank.kind(e), bank.z(e), tokens[e]) for e in evidence_ids}, paths,
            )
            evidence = None
            if paths:
                target_index = torch.tensor([k for k, _ in paths], dtype=torch.long, device=dev)
                bag_index = torch.tensor([k for k, t in enumerate(targets) if bags.get(t)], dtype=torch.long, device=dev)
                evidence = aggregate_cqet(direct, path_scores, target_index)[bag_index].cpu().float()
            result[qid] = {"direct": direct.cpu().float(), "evidence": evidence, "list_sha256": _scored_list_sha(row)}
            if i % 100 == 0 or i == len(c2_records):
                print(f"[TB_CQET C2 logits] {i}/{len(c2_records)} elapsed={time.time()-started:.1f}s", flush=True)
    return result


def train_student_c2(
    student: NativeStudent | QTStudent,
    c2_records: list[dict],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_KD",
    logical_batch: int = 64,
    recipe: StudentRecipe = StudentRecipe(),
    negative_pool: Optional[Sequence[str]] = None,
    evidence_pool: Optional[Sequence[str]] = None,
    save_dir: Optional[Path] = None,
    seed: int = 13,
    expected_parent_hash: Optional[str] = None,
    metadata: Optional[dict] = None,
    resume_from: Optional[Path] = None,
    teacher_logits: Optional[Mapping[str, dict]] = None,
    max_updates: Optional[int] = None,
    log_path: Optional[Path] = None,
    epochs: int = 1,
) -> dict[float, Path]:
    """C2 training on the shared graph.

    Each record is extended by ``c2_training_row`` (random negative targets, some with a
    random-evidence bag). Per query the loss is ``SUP_direct + SUP_evidence + kd_weight *
    (KD_direct + KD_evidence)``; SUP covers the extended lists, KD covers the prefix of each
    list the teacher scored (``teacher_logits`` from ``build_teacher_logits_cache`` over
    ``c2_teacher_rows``): the shared-graph targets, or the whole extended list under
    ``teacher_scored_negatives``. ``recipe.kd_loss`` sets temperature, normalisation and top-k.
    ``negative_pool`` defaults to every target of ``c2_records``, ``evidence_pool`` to every
    evidence object in their bags.
    """
    is_qt = isinstance(student, QTStudent)
    is_kd = arm == "NATIVE_KD"
    if is_kd and teacher_logits is None:
        raise ValueError("NATIVE_C2_KD requires the TB_CQET logits cache")
    if is_qt and (is_kd or teacher_logits is not None):
        raise ValueError("QT-only C2 is SUP-only and independent of CQET")
    enforce_task_numerics()
    dev = torch.device(device)
    student.to(dev).train()
    bank.attach_device(dev)
    parent_hash = model_state_sha(student)
    if expected_parent_hash is None or parent_hash != expected_parent_hash:
        raise ValueError(f"C2 step0 state {parent_hash} != selected C1 {expected_parent_hash}")
    if negative_pool is None:
        negative_pool = utf8_sorted({t for row in c2_records for t in row["targets"]})
    if evidence_pool is None:
        evidence_pool = utf8_sorted({e for row in c2_records for bag in row.get("natural_bags", {}).values() for e in bag})
    negative_pool, evidence_pool = list(negative_pool), list(evidence_pool)
    if is_kd:
        for row in c2_teacher_rows(c2_records, recipe, negative_pool, evidence_pool, seed):
            if teacher_logits[row["query_id"]]["list_sha256"] != _scored_list_sha(row):
                raise ValueError(f"{row['query_id']}: teacher logits were scored on a different C2 list")
        teacher_logits = {
            q: {"direct": entry["direct"].to(dev), "evidence": None if entry["evidence"] is None else entry["evidence"].to(dev)}
            for q, entry in teacher_logits.items()
        }

    ordered, batches = _student_schedule(c2_records, "C2_SHARED", seed, epochs, logical_batch)
    order_sha = _order_sha(ordered)
    optimizer = AdamW(student.param_groups(recipe.lr_p, recipe.lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    stage = "QT_C2_SUP" if is_qt else ("NATIVE_C2_KD" if is_kd else "NATIVE_C2_SUP")
    total_steps = len(batches)
    fraction_steps = _student_snapshot_steps(total_steps, epochs)
    recipe_log = _recipe_log(
        recipe, kd_weight=recipe.kd_weight if is_kd else 0.0,
        negative_pool_size=len(negative_pool), evidence_pool_size=len(evidence_pool),
    )
    base_meta = {**(metadata or {}), "epochs": epochs, "logical_batch": logical_batch, **recipe_log}

    def query_losses(rows: Sequence[dict], epoch: int) -> tuple[list[Tensor], dict[str, list[Tensor]]]:
        extended = [c2_training_row(row, recipe, negative_pool, evidence_pool, seed, epoch) for row in rows]
        scored = _student_c2_batch_scores(student, bank, extended, recipe.logit_scale)
        losses: list[Tensor] = []
        parts: dict[str, list[Tensor]] = defaultdict(list)
        for row, (direct_s, evidence_s, bag_targets) in zip(extended, scored):
            qid = row["query_id"]
            positives = set(row["positives"])
            direct_sup = rank_mass_loss(direct_s, torch.tensor([t in positives for t in row["targets"]], device=dev))
            if direct_sup is None:
                raise ValueError(f"{qid}: C2 Direct supervision is inactive")
            total = direct_sup
            parts["direct_sup"].append(direct_sup)
            evidence_sup = None
            if evidence_s is not None:
                evidence_sup = rank_mass_loss(evidence_s, torch.tensor([t in positives for _, t in bag_targets], device=dev))
                if evidence_sup is not None:
                    total = total + evidence_sup
                    parts["evidence_sup"].append(evidence_sup)
            if is_kd:
                entry = teacher_logits[qid]
                direct_kd = recipe.kd_loss(direct_s[: entry["direct"].numel()], entry["direct"])
                if direct_kd is None:
                    raise RuntimeError(f"{qid}: Direct KD list is too short")
                total = total + recipe.kd_weight * direct_kd
                parts["direct_kd"].append(direct_kd)
                if evidence_sup is not None and entry["evidence"] is not None:
                    evidence_kd = recipe.kd_loss(evidence_s[: entry["evidence"].numel()], entry["evidence"])
                    if evidence_kd is not None:
                        total = total + recipe.kd_weight * evidence_kd
                        parts["evidence_kd"].append(evidence_kd)
            losses.append(total)
        return losses, parts

    saved: dict[float, Path] = {}
    logical_step = 0
    next_cursor = 0
    started = time.time()
    if resume_from is not None:
        extra = load_training_checkpoint(resume_from, student, optimizer, expected_stage=stage)
        if extra.get("order_sha256") != order_sha:
            raise ValueError("resume materialized query order changed")
        if extra.get("logical_batch", logical_batch) != logical_batch:
            raise ValueError("resume logical batch changed")
        logical_step = int(extra["logical_step"])
        next_cursor = int(extra["next_record_cursor"])
    elif save_dir:
        saved[0.0] = save_dir / "snapshot_frac000.pt"
        save_checkpoint(saved[0.0], student, optimizer, {
            **base_meta, "stage": stage, "epoch": 0, "logical_step": 0, "next_record_cursor": 0,
            "order_sha256": order_sha, "parent_state_sha256": parent_hash,
        })

    for epoch, start, end in batches:
        if start < next_cursor:
            continue
        if max_updates is not None and logical_step >= max_updates:
            break
        batch = ordered[start:end]
        microbatch, mean_loss, components, anchor_value = _student_step(
            student, optimizer, batch, lambda rows: query_losses(rows, epoch),
            anchor_weight=recipe.anchor_weight, stage=stage,
        )
        grad_norm = float(nn.utils.clip_grad_norm_(student.parameters(), recipe.clip_norm))
        lr_factor = _set_scheduled_lr(optimizer, recipe, logical_step, total_steps)
        optimizer.step()
        logical_step += 1
        next_cursor = end
        _log(log_path, {
            "stage": stage, "epoch": epoch, "step": logical_step,
            "record_ids": [row["query_id"] for row in batch], "batch_queries": len(batch),
            "loss": mean_loss + anchor_value, "anchor_loss_weighted": anchor_value,
            "grad_norm_preclip": grad_norm, "grad_norm_postclip": min(grad_norm, recipe.clip_norm),
            **{f"{name}_loss": float(np.mean(components[name])) if name in components else None
               for name in ("direct_sup", "evidence_sup", "direct_kd", "evidence_kd")},
            **{f"{name}_denominator": len(components.get(name, ()))
               for name in ("direct_sup", "evidence_sup", "direct_kd", "evidence_kd")},
            **recipe_log, "lr_factor": lr_factor, "parent_state_sha256": parent_hash,
            "student_query_microbatch": microbatch,
            "elapsed_seconds": time.time() - started, **_student_stats(student), **_gpu_peaks(),
        })
        if logical_step in fraction_steps and save_dir:
            for fraction in fraction_steps[logical_step]:
                path = save_dir / _snapshot_name(fraction, epoch, epochs)
                save_checkpoint(path, student, optimizer, {
                    **base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                    "next_record_cursor": next_cursor, "order_sha256": order_sha,
                    "parent_state_sha256": parent_hash, "fraction": fraction, "student_query_microbatch": microbatch,
                })
                saved[fraction] = path
    if max_updates is None and logical_step != total_steps:
        raise RuntimeError(f"{stage}: completed {logical_step} steps, expected {total_steps}")
    return saved

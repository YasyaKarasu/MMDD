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
"""
from __future__ import annotations

import hashlib
import gc
import traceback
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

from . import SCHEMA_VERSION
from .execution_layout import (
    TEACHER_CHUNK_LADDER, TEACHER_INITIAL_CHUNK, TEACHER_INFERENCE_CHUNK,
    TEACHER_LAYOUT_REVISION, teacher_numerical_layout,
)
from .features import ObjectBank
from .labels import Labels
from .losses import (
    aggregate_corrected_lse,
    aggregate_cqet,
    hierarchical_relation_mean,
    hierarchical_support_mean,
    list_kl_divergence,
    positive_average_pair_loss,
    rank_mass_loss,
)
from .models import FreshPathTeacher, NativeStudent, QTStudent


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

    @classmethod
    def from_protocol(cls, protocol: Mapping) -> "StudentRecipe":
        student = protocol["student"]
        return cls(
            lr_p=float(student["P_lr"]),
            lr_r=float(student["R_lr"]),
            logit_scale=float(student["logit_scale"]),
            kd_weight=float(student["kd_weight"]),
            kd_temperature=float(student["temperature"]),
            random_negatives=int(student["random_negatives"]),
            anchor_weight=float(student["anchor_weight"]),
            clip_norm=float(protocol.get("numerics", {}).get("grad_clip", 1.0)),
        )

    def as_dict(self) -> dict:
        return asdict(self)


def enforce_task_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def state_sha(state: Mapping[str, Tensor]) -> str:
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(array.dtype).encode("ascii"))
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def model_state_sha(model: nn.Module) -> str:
    return state_sha(model.state_dict())


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(payload: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(payload) != required:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"].cpu())
    cuda_state = payload["torch_cuda_all"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint requires CUDA RNG restoration")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[AdamW] = None,
    extra: Optional[dict] = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(extra or {})
    metadata["model_state_sha256"] = model_state_sha(model)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "rng": _rng_state(),
        "extra": metadata,
    }
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    *,
    expected_stage: str,
) -> dict:
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


def _hash_order(records: Sequence[dict], namespace: str, seed: int, epoch: int) -> list[dict]:
    def record_id(row: dict) -> str:
        return str(row.get("record_id") or row.get("item_id") or row["query_id"])

    return sorted(
        records,
        key=lambda row: (
            hashlib.sha256(
                f"{namespace}|{seed}|{epoch}|{record_id(row)}".encode("utf-8")
            ).digest(),
            record_id(row).encode("utf-8"),
        ),
    )


def _order_sha(records: Sequence[dict]) -> str:
    ids = [str(row.get("record_id") or row.get("item_id") or row["query_id"]) for row in records]
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


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
    steps: dict[int, list[float]] = defaultdict(list)
    if epochs == 1:
        for fraction in (0.25, 0.5, 0.75, 1.0):
            steps[math.ceil(total_steps * fraction)].append(fraction)
    else:
        for epoch in range(1, epochs + 1):
            steps[(total_steps // epochs) * epoch].append(epoch / epochs)
    return steps


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


def _student_parameter_norms(student: NativeStudent | QTStudent) -> dict[str, float]:
    with torch.no_grad():
        if isinstance(student, QTStudent):
            p_values = [student.P_table]
            r_values = [student.R_QT]
        else:
            p_values = list(student.P.values())
            r_values = list(student.R.values())
        p_norm = torch.sqrt(sum(torch.sum(value * value) for value in p_values))
        r_norm = torch.sqrt(sum(torch.sum(value * value) for value in r_values))
    return {"P_parameter_norm": float(p_norm), "R_parameter_norm": float(r_norm)}


def _student_drift_spectrum(student: NativeStudent | QTStudent) -> dict[str, float]:
    """Top two singular values of R_QT - I: sigma1 >> sigma2 is the rank-one drift signature."""
    with torch.no_grad():
        r_qt = student.R_QT if isinstance(student, QTStudent) else student.R["QT"]
        values = torch.linalg.svdvals(r_qt - torch.eye(r_qt.shape[0], device=r_qt.device))
    return {"sigma1_R_QT_minus_I": float(values[0]), "sigma2_R_QT_minus_I": float(values[1])}


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


def _scorer_objects(
    rows: Sequence[tuple], keys: Sequence[tuple]
) -> list[tuple[str, tuple, Tensor, Tensor]]:
    """(kind, cache_key, z, content) for every object a scored row touches."""
    objects: list[tuple[str, tuple, Tensor, Tensor]] = []
    for row, key in zip(rows, keys):
        if key is None:
            continue
        for slot in range(len(row) // 3):
            objects.append((row[3 * slot], key[slot], row[3 * slot + 1], row[3 * slot + 2]))
    return objects


def _torch_rng_state() -> tuple[Tensor, Optional[list[Tensor]]]:
    return (
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else None,
    )


def _restore_torch_rng_state(state: tuple[Tensor, Optional[list[Tensor]]]) -> None:
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)


class TeacherListScorer:
    """Two-pass full-list score VJP that encodes every object once per pass.

    Pass 1 scores each list under ``no_grad`` and returns detached leaves, so the caller
    builds the loss on the complete score vectors; ``backward`` then takes its adjoint.
    Pass 2 re-encodes into a fresh grad-enabled cache, replays each chunk's dropout, and
    back-propagates that chunk's adjoint. Parameter gradients accumulate across chunks
    while the loss denominator stays the complete list and only one chunk's activations
    are live at a time, so the chunk size remains the OOM knob.

    This keeps the two-pass structure and dropout replay of the previous scorer but drops
    two kinds of repeated work that dominated the stage's wall clock: the cache is no
    longer rebuilt per chunk (each object was re-encoded once per chunk, and objects
    shared between relation lists -- the query itself, and any evidence/target pair
    appearing in several lists -- were re-encoded once per list), and objects are encoded
    through ``encode_many`` instead of one ``encode_one`` call each.
    """

    def __init__(self, model: FreshPathTeacher, chunk: int, *, mode: str = "two_pass") -> None:
        if mode not in {"single_graph", "two_pass"}:
            raise ValueError(f"invalid Teacher backward mode: {mode}")
        self.mode = mode
        if chunk not in TEACHER_CHUNK_LADDER:
            raise ValueError(f"invalid Teacher candidate chunk: {chunk}")
        self.model = model
        self.chunk = chunk
        self.cache: dict = {}
        self.calls: list[tuple[str, list[tuple], list[tuple],
                              tuple[Tensor, Optional[list[Tensor]]], Tensor]] = []
        self.scores: list[Tensor] = []

    def encode(self, rows: Sequence[tuple], keys: Sequence[tuple]) -> None:
        """Materialise every object in ``rows`` that is not already in the cache."""
        grouped: dict[str, list[tuple]] = defaultdict(list)
        seen: set[tuple] = set()
        for kind, object_key, z, content in _scorer_objects(rows, keys):
            if (kind, object_key) in self.cache or (kind, object_key) in seen:
                continue
            seen.add((kind, object_key))
            grouped[kind].append((object_key, z, content))
        for kind, items in grouped.items():
            segments, lengths, globals_ = self.model.encode_many(
                kind, torch.stack([z for _, z, _ in items]), [c for _, _, c in items]
            )
            for position, (object_key, _, _) in enumerate(items):
                pooled = (
                    segments[position, : lengths[position]]
                    if kind == "table"
                    else segments[position, 1:]
                )
                self.cache[(kind, object_key)] = (pooled, globals_[position])

    def _score(self, kind: str, rows: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        if len(rows) != len(keys):
            raise ValueError("scorer rows/cache keys must have the same length")
        if kind not in {"pairs", "triplets"}:
            raise ValueError(f"invalid score kind: {kind}")
        if self.mode == "single_graph":
            # This cache is grad-enabled and lives for this query only. It MUST
            # NOT be detached or survive an optimizer update during TA.
            self.encode(rows, keys)
            score = self.model.score_pairs if kind == "pairs" else self.model.score_triplets
            outputs = [
                score(rows[start:start+self.chunk], cache=self.cache,
                      cache_keys=keys[start:start+self.chunk])
                for start in range(0, len(rows), self.chunk)
            ]
            if not outputs:
                return torch.empty(0, device=next(self.model.parameters()).device)
            result = torch.cat(outputs)
            self.scores.append(result)
            return result
        # Pass 1 never back-propagates, so encoding here must not build a graph that
        # would stay alive until backward(); pass 2 re-encodes with grad enabled.
        with torch.no_grad():
            self.encode(rows, keys)
        score = self.model.score_pairs if kind == "pairs" else self.model.score_triplets
        outputs = []
        for start in range(0, len(rows), self.chunk):
            chunk_rows = list(rows[start : start + self.chunk])
            chunk_keys = list(keys[start : start + self.chunk])
            rng = _torch_rng_state()
            with torch.no_grad():
                result = score(chunk_rows, cache=self.cache, cache_keys=chunk_keys)
            leaf = result.detach().requires_grad_(True)
            self.calls.append((kind, chunk_rows, chunk_keys, rng, leaf))
            outputs.append(leaf)
        if not outputs:
            return torch.empty(0, device=next(self.model.parameters()).device)
        result = torch.cat(outputs)
        self.scores.append(result)
        return result

    def score_pairs(self, pairs: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        return self._score("pairs", pairs, keys)

    def score_triplets(self, triplets: Sequence[tuple], keys: Sequence[tuple]) -> Tensor:
        return self._score("triplets", triplets, keys)

    def backward(self, loss: Tensor, *, scale: float) -> None:
        if self.mode == "single_graph":
            if not self.scores:
                raise RuntimeError("Teacher list scorer has no scored lists")
            try:
                (loss * scale).backward()
            finally:
                self.cache.clear()
                self.scores.clear()
            return
        forward_end_rng = _torch_rng_state()
        leaves = [call[-1] for call in self.calls]
        if not leaves:
            raise RuntimeError("Teacher list scorer has no scored leaves")
        adjoints = torch.autograd.grad(loss, leaves)
        cache: dict = {}
        self.cache = cache
        for _kind, rows, keys, _rng, _leaf in self.calls:
            self.encode(rows, keys)
        for index, ((kind, rows, keys, rng, _leaf), adjoint) in enumerate(zip(self.calls, adjoints)):
            _restore_torch_rng_state(rng)
            score = self.model.score_pairs if kind == "pairs" else self.model.score_triplets
            # Chunks share the encoder outputs, so every chunk but the last must retain them.
            score(rows, cache=cache, cache_keys=keys).backward(
                adjoint * scale, retain_graph=index + 1 < len(self.calls)
            )
        _restore_torch_rng_state(forward_end_rng)
        self.calls.clear()
        self.cache.clear()
        self.scores.clear()


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _support_loss(
    bank: ObjectBank,
    query_id: str,
    records: Sequence[dict],
    query_tokens: Tensor,
    scorer: TeacherListScorer,
    tokens: Mapping[str, Tensor],
) -> Optional[Tensor]:
    by_target: dict[str, list[Tensor]] = defaultdict(list)
    zq = bank.z(query_id)
    for row in records:
        target_id = row["target_id"]
        modality = row["modality"]
        positives = list(row["positives"])
        competitors = list(row["competitors"])
        if not positives or not competitors:
            continue
        zt, ct = bank.z(target_id), tokens[target_id]
        pos = [
            ("table", zq, query_tokens, modality, bank.z(e), tokens[e], "table", zt, ct)
            for e in positives
        ]
        neg = [
            ("table", zq, query_tokens, modality, bank.z(e), tokens[e], "table", zt, ct)
            for e in competitors
        ]
        pos_keys = [((query_id, 0), (e, 2), (target_id, 1)) for e in positives]
        neg_keys = [((query_id, 0), (e, 2), (target_id, 1)) for e in competitors]
        pos_scores = scorer.score_triplets(pos, pos_keys)
        neg_scores = scorer.score_triplets(neg, neg_keys)
        loss = positive_average_pair_loss(pos_scores, neg_scores)
        if loss is not None:
            by_target[target_id].append(loss)
    target_losses = [by_target[t] for t in sorted(by_target, key=lambda x: x.encode("utf-8"))]
    return hierarchical_support_mean(target_losses)


def _support_object_ids(records: Sequence[dict]) -> list[str]:
    return list(dict.fromkeys(
        object_id for record in records
        for object_id in [record["target_id"], *record["positives"], *record["competitors"]]
    ))


def _ta_query_backward(model, bank, row, labels, dev, candidate_chunk,
                       backward_mode, scale, support_weight=0.2) -> dict:
    relation_values: list[float] = []
    support_values: list[float] = []
    active_relation_counts: list[int] = []
    loss_values: list[float] = []
    qid = row["query_id"]
    zq = bank.z(qid)
    scorer = TeacherListScorer(model, candidate_chunk, mode=backward_mode)
    by_relation: dict[str, list[Tensor]] = {
        "QT": [], "Q_text": [], "Q_image": [],
        "QET_text": [], "QET_image": [],
    }
    support_records = row.get("support_records", [])

    # One host gather per query for every content tensor it needs.
    needed = [qid, *row["qt_candidates"]]
    for modality in ("text", "image"):
        needed.extend(row["qe_candidates"][modality])
    for item in row.get("qet_lists", []):
        needed.append(item["evidence_id"])
        needed.extend(item["candidates"])
    for record in support_records:
        needed.append(record["target_id"])
        needed.extend(record["positives"])
        needed.extend(record["competitors"])
    needed = list(dict.fromkeys(needed))
    tokens = dict(zip(needed, bank.tokens_many(needed)))

    # (relation, kind, scored rows, cache keys, positives, valid mask)
    lists: list[tuple[str, str, list[tuple], list[tuple],
                       set[str], Optional[Tensor]]] = []

    qt = list(row["qt_candidates"])
    lists.append((
        "QT", "pairs",
        [("table", zq, tokens[qid], "table", bank.z(t), tokens[t]) for t in qt],
        [((qid, 0), (t, 1)) for t in qt],
        set(labels.queries[qid]["G"]), None,
    ))

    for modality in ("text", "image"):
        candidates = list(row["qe_candidates"][modality])
        positives = set(labels.queries[qid]["Qpos"][modality])
        if not positives:
            continue
        lists.append((
            f"Q_{modality}", "pairs",
            [("table", zq, tokens[qid], modality, bank.z(e), tokens[e])
             for e in candidates],
            [((qid, 0), (e, 1)) for e in candidates],
            positives, None,
        ))

    for item in row.get("qet_lists", []):
        evidence_id = item["evidence_id"]
        modality = item["evidence_kind"]
        candidates = list(item["candidates"])
        ignore = set(item.get("ignore", ()))
        lists.append((
            f"QET_{modality}", "triplets",
            [("table", zq, tokens[qid], modality, bank.z(evidence_id),
              tokens[evidence_id], "table", bank.z(t), tokens[t])
             for t in candidates],
            [((qid, 0), (evidence_id, 2), (t, 1)) for t in candidates],
            set(item["positives"]),
            torch.tensor([t not in ignore for t in candidates], device=dev),
        ))

    for relation, kind, rows_, keys, positives, valid in lists:
        scores = (
            scorer.score_triplets(rows_, keys) if kind == "triplets"
            else scorer.score_pairs(rows_, keys)
        )
        loss = rank_mass_loss(
            scores,
            torch.tensor([key[-1][0] in positives for key in keys], device=dev),
            valid,
        )
        if loss is not None:
            by_relation[relation].append(loss)

    relation_loss = hierarchical_relation_mean(list(by_relation.values()))
    support = _support_loss(
        bank, qid, support_records, tokens[qid], scorer, tokens
    )
    query_loss = relation_loss
    if relation_loss is not None:
        relation_values.append(float(relation_loss.detach()))
        active_relation_counts.append(sum(bool(values) for values in by_relation.values()))
    if support is not None:
        support_values.append(float(support.detach()))
        query_loss = support_weight * support if query_loss is None else query_loss + support_weight * support
    if query_loss is None:
        raise RuntimeError(f"TA query {qid} has no active loss")
    loss_values.append(float(query_loss.detach()))
    scorer.backward(query_loss, scale=scale)
    return {"loss": loss_values[0],
            "relation": relation_values[0] if relation_values else None,
            "support": support_values[0] if support_values else None,
            "active_relations": active_relation_counts[0] if active_relation_counts else 0}


def _tb_query_backward(model, bank, row, dev, candidate_chunk,
                       backward_mode, scale, mode, *,
                       direct_weight=0.5, aggregate_weight=0.5, support_weight=0.2) -> dict:
    loss_values: list[float] = []
    direct_values: list[float] = []
    path_values: list[float] = []
    support_values: list[float] = []
    stage = f"TB_{mode.upper()}"
    qid = row["query_id"]
    targets = list(row["targets"])
    positives = set(row["positives"])
    pos_mask = torch.tensor([t in positives for t in targets], device=dev)
    zq = bank.z(qid)
    all_ids = [qid, *targets]
    if mode != "qt":
        all_ids.extend(e for t in targets for e in row["natural_bags"].get(t, ()))
        # Auxiliary W/competitors need tokens, but never enter natural ranking bags.
        all_ids.extend(_support_object_ids(row.get("support_records", [])))
    all_ids = list(dict.fromkeys(all_ids))
    tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
    cq = tokens[qid]
    scorer = TeacherListScorer(model, candidate_chunk, mode=backward_mode)
    pairs = [
        ("table", zq, cq, "table", bank.z(target), tokens[target])
        for target in targets
    ]
    pair_keys = [((qid, 0), (target, 1)) for target in targets]
    f0 = scorer.score_pairs(pairs, pair_keys)
    direct = rank_mass_loss(f0, pos_mask)
    if direct is not None:
        direct_values.append(float(direct.detach()))
    if mode == "qt":
        query_loss = direct
    else:
        paths = [
            (i, e) for i, t in enumerate(targets)
            for e in row["natural_bags"].get(t, ())
        ]
        trips = [
            ("table", zq, cq, bank.kind(evidence_id), bank.z(evidence_id),
             tokens[evidence_id], "table", bank.z(targets[target_i]),
             tokens[targets[target_i]])
            for target_i, evidence_id in paths
        ]
        trip_keys = [
            ((qid, 0), (evidence_id, 2), (targets[target_i], 1))
            for target_i, evidence_id in paths
        ]
        path_scores = scorer.score_triplets(trips, trip_keys)
        target_index = torch.tensor(
            [i for i, _ in paths], dtype=torch.long, device=dev
        )
        aggregated = (
            aggregate_cqet(f0, path_scores, target_index)
            if mode == "cqet"
            else aggregate_corrected_lse(f0, path_scores, target_index)
        )
        path_loss = rank_mass_loss(aggregated, pos_mask)
        if path_loss is not None:
            path_values.append(float(path_loss.detach()))
        query_loss = None
        if direct is not None:
            query_loss = direct_weight * direct
        if path_loss is not None:
            weighted = aggregate_weight * path_loss
            query_loss = weighted if query_loss is None else query_loss + weighted
        support = _support_loss(
            bank, qid, row.get("support_records", []), cq, scorer, tokens
        )
        if support is not None:
            support_values.append(float(support.detach()))
            weighted = support_weight * support
            query_loss = weighted if query_loss is None else query_loss + weighted
    if query_loss is None:
        raise RuntimeError(f"{stage} query {qid} has no active loss")
    loss_values.append(float(query_loss.detach()))
    scorer.backward(query_loss, scale=scale)
    return {"loss": loss_values[0],
            "direct": direct_values[0] if direct_values else None,
            "path": path_values[0] if path_values else None,
            "support": support_values[0] if support_values else None}


def _run_teacher_logical_batch(optimizer, batch, query_fn, *, candidate_chunk: int,
                               initial_mode: str = "single_graph") -> tuple:
    """Accumulate all query gradients transactionally; NEVER update optimizer here.

    A failed attempt discards ALL gradients from the logical batch, unwinds query
    frames, restores Python/NumPy/CPU/CUDA RNG and replays every query. Single-graph
    failure first retries two-pass at the SAME chunk; only then reduces the chunk.
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
        failed = False
        try:
            for row in batch:
                result = query_fn(row, candidate_chunk, backward_mode, 1.0 / len(batch))
                if not math.isfinite(float(result["loss"])):
                    raise FloatingPointError("nonfinite Teacher loss; no optimizer step executed")
                metrics.append(result)
        except BaseException as error:
            if not _is_cuda_oom(error):
                optimizer.zero_grad(set_to_none=True)
                raise
            # Do not run the retry while an exception traceback still owns the
            # failed forward/backward's tensors. empty_cache alone cannot free them.
            events.append({"mode": backward_mode, "chunk": candidate_chunk,
                           "completed_queries_discarded": len(metrics),
                           "error": type(error).__name__})
            traceback.clear_frames(error.__traceback__)
            error.__traceback__ = None
            failed = True
        if not failed:
            return metrics, candidate_chunk, backward_mode, events
        metrics.clear()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _restore_rng_state(batch_rng)
        if backward_mode == "single_graph":
            backward_mode = "two_pass"
        elif candidate_chunk > 1:
            candidate_chunk = TEACHER_CHUNK_LADDER[
                TEACHER_CHUNK_LADDER.index(candidate_chunk) + 1
            ]
        else:
            raise RuntimeError("BLOCKED_RESOURCE: Teacher two-pass chunk 1 OOM; no optimizer update")


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
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> Path:
    enforce_task_numerics()
    dev = torch.device(device)
    model.to(dev).train()
    bank.attach_device(dev)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                      betas=(0.9, 0.999), eps=1e-8)
    total_steps = math.ceil(len(ta_records) / logical_batch) * epochs
    base_meta = {**dict(metadata or {}), "numerical_layout": teacher_numerical_layout()}
    if save_dir:
        save_checkpoint(save_dir / "init.pt", model, optimizer,
                        {**base_meta, "stage": "TA", "epoch": 0, "logical_step": 0,
                         "next_record_cursor": 0})
    logical_step = 0
    started = time.time()
    candidate_chunk = TEACHER_INITIAL_CHUNK
    for epoch in range(1, epochs + 1):
        ordered = _hash_order(ta_records, "TA", seed, epoch)
        for start in range(0, len(ordered), logical_batch):
            batch = ordered[start : start + logical_batch]
            metrics, candidate_chunk, backward_mode, oom_events = _run_teacher_logical_batch(
                optimizer, batch,
                lambda row, chunk, mode_, scale: _ta_query_backward(
                    model, bank, row, labels, dev, chunk, mode_, scale, support_weight),
                candidate_chunk=candidate_chunk,
            )
            loss_values = [x["loss"] for x in metrics]
            relation_values = [x["relation"] for x in metrics if x["relation"] is not None]
            support_values = [x["support"] for x in metrics if x["support"] is not None]
            active_relation_counts = [x["active_relations"] for x in metrics]

            batch_loss_value = sum(loss_values) / len(batch)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            logical_step += 1
            grad_norm_value = float(grad_norm)
            _log(log_path, {
                "stage": "TA", "epoch": epoch, "step": logical_step,
                "record_ids": [row["query_id"] for row in batch],
                "active_queries": len(batch), "batch_queries": len(batch),
                "loss": batch_loss_value, "grad_norm_preclip": grad_norm_value,
                "grad_norm_postclip": min(grad_norm_value, 1.0),
                "clip_norm": 1.0, "lr": lr,
                "teacher_candidate_chunk": candidate_chunk,
                "teacher_backward_mode": backward_mode,
                "numerical_layout_revision": TEACHER_LAYOUT_REVISION,
                "oom_retries": len(oom_events),
                "oom_events": oom_events,
                "loss_denominator_active_queries": len(batch),
                "relation_loss": float(np.mean(relation_values)) if relation_values else None,
                "support_loss_unweighted": float(np.mean(support_values)) if support_values else None,
                "relation_query_denominator": len(relation_values),
                "support_query_denominator": len(support_values),
                "active_relations_total": sum(active_relation_counts),
                "elapsed_seconds": time.time() - started,
                **_gpu_peaks(),
            })
            print(
                f"[TA] epoch={epoch}/{epochs} step={logical_step}/{total_steps} "
                f"loss={batch_loss_value:.5f} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        if save_dir:
            save_checkpoint(
                save_dir / f"epoch{epoch}.pt", model, optimizer,
                {**base_meta, "stage": "TA", "epoch": epoch, "logical_step": logical_step,
                 "next_record_cursor": 0, "order_sha256": _order_sha(ordered),
                 "teacher_candidate_chunk": candidate_chunk},
            )
    return save_dir / f"epoch{epochs}.pt" if save_dir else Path(f"epoch{epochs}.pt")


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
    save_dir: Optional[Path] = None,
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
) -> Path:
    if mode not in {"cqet", "lse", "qt"}:
        raise ValueError("TB mode must be cqet, lse, or qt")
    enforce_task_numerics()
    dev = torch.device(device)
    model.to(dev)
    bank.attach_device(dev)
    trainable = model.set_tb_trainable()
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay,
                      betas=(0.9, 0.999), eps=1e-8)
    total_steps = math.ceil(len(tb_records) / logical_batch) * epochs
    half_step = math.ceil(0.5 * total_steps)
    stage = f"TB_{mode.upper()}"
    base_meta = {**dict(metadata or {}), "numerical_layout": teacher_numerical_layout()}
    if save_dir:
        save_checkpoint(save_dir / "init.pt", model, optimizer,
                        {**base_meta, "stage": stage, "epoch": 0, "logical_step": 0,
                         "next_record_cursor": 0})
    logical_step = 0
    started = time.time()
    candidate_chunk = TEACHER_INITIAL_CHUNK
    for epoch in range(1, epochs + 1):
        ordered = _hash_order(tb_records, "TB_SHARED", seed, epoch)
        model.train()
        for start in range(0, len(ordered), logical_batch):
            batch = ordered[start : start + logical_batch]
            metrics, candidate_chunk, backward_mode, oom_events = _run_teacher_logical_batch(
                optimizer, batch,
                lambda row, chunk, mode_, scale: _tb_query_backward(
                    model, bank, row, dev, chunk, mode_, scale, mode,
                    direct_weight=direct_weight, aggregate_weight=aggregate_weight,
                    support_weight=support_weight),
                candidate_chunk=candidate_chunk,
            )
            loss_values = [x["loss"] for x in metrics]
            direct_values = [x["direct"] for x in metrics if x["direct"] is not None]
            path_values = [x["path"] for x in metrics if x["path"] is not None]
            support_values = [x["support"] for x in metrics if x["support"] is not None]

            batch_loss_value = sum(loss_values) / len(batch)
            grad_norm = nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            logical_step += 1
            grad_norm_value = float(grad_norm)
            _log(log_path, {
                "stage": stage, "epoch": epoch, "step": logical_step,
                "record_ids": [row["query_id"] for row in batch],
                "active_queries": len(batch), "batch_queries": len(batch),
                "loss": batch_loss_value, "grad_norm_preclip": grad_norm_value,
                "grad_norm_postclip": min(grad_norm_value, 1.0),
                "clip_norm": 1.0, "lr": lr, "aggregation": mode,
                "teacher_candidate_chunk": candidate_chunk,
                "teacher_backward_mode": backward_mode,
                "numerical_layout_revision": TEACHER_LAYOUT_REVISION,
                "oom_retries": len(oom_events),
                "oom_events": oom_events,
                "loss_denominator_active_queries": len(batch),
                "direct_rank_loss": float(np.mean(direct_values)) if direct_values else None,
                "aggregate_rank_loss": float(np.mean(path_values)) if path_values else None,
                "support_loss_unweighted": float(np.mean(support_values)) if support_values else None,
                "direct_loss_denominator": len(direct_values),
                "aggregate_loss_denominator": len(path_values),
                "support_loss_denominator": len(support_values),
                "elapsed_seconds": time.time() - started,
                **_gpu_peaks(),
            })
            print(
                f"[{stage}] step={logical_step}/{total_steps} loss={batch_loss_value:.5f} "
                f"elapsed={time.time()-started:.1f}s", flush=True,
            )
            if save_dir and logical_step == half_step:
                save_checkpoint(
                    save_dir / "half.pt", model, optimizer,
                    {**base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                     "next_record_cursor": min(start + logical_batch, len(ordered)),
                     "order_sha256": _order_sha(ordered),
                     "teacher_candidate_chunk": candidate_chunk},
                )
    if save_dir:
        save_checkpoint(
            save_dir / "end.pt", model, optimizer,
            {**base_meta, "stage": stage, "epoch": epochs, "logical_step": logical_step,
             "next_record_cursor": 0, "order_sha256": _order_sha(ordered),
             "teacher_candidate_chunk": candidate_chunk},
        )
    return save_dir / "end.pt" if save_dir else Path("end.pt")


def train_student_c1(
    student: NativeStudent | QTStudent,
    edge_lists: list[dict],
    teacher: Optional[FreshPathTeacher],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_SUP",
    logical_batch: int = 64,
    lr_p: float = StudentRecipe.lr_p,
    lr_r: float = StudentRecipe.lr_r,
    logit_scale: float = StudentRecipe.logit_scale,
    anchor_weight: float = StudentRecipe.anchor_weight,
    clip_norm: float = StudentRecipe.clip_norm,
    save_dir: Optional[Path] = None,
    seed: int = 13,
    metadata: Optional[dict] = None,
    log_path: Optional[Path] = None,
    epochs: int = 1,
) -> dict[float, Path]:
    if teacher is not None or "KD" in arm:
        raise ValueError("C1 has no KD branch")
    enforce_task_numerics()
    dev = torch.device(device)
    student.to(dev).train()
    is_qt = isinstance(student, QTStudent)
    valid_lists = [row for row in edge_lists if not is_qt or row["relation"] == "QT"]
    ordered, batches = _student_schedule(valid_lists, "C1_QT" if is_qt else "C1_NATIVE", seed, epochs, logical_batch)
    optimizer = AdamW(student.param_groups(lr_p, lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    total_steps = len(batches)
    fraction_steps = _student_snapshot_steps(total_steps, epochs)
    stage = "QT_C1_SUP" if is_qt else "NATIVE_C1_SUP"
    base_meta = {**dict(metadata or {}), "epochs": epochs, "logical_batch": logical_batch,
                 "P_lr": lr_p, "R_lr": lr_r, "logit_scale": logit_scale, "anchor_weight": anchor_weight}
    saved: dict[float, Path] = {}
    if save_dir:
        path = save_dir / "snapshot_frac000.pt"
        save_checkpoint(path, student, optimizer,
                        {**base_meta, "stage": stage, "epoch": 0, "logical_step": 0,
                         "next_record_cursor": 0, "order_sha256": _order_sha(ordered),
                         "student_query_microbatch": 64})
        saved[0.0] = path
    logical_step = 0
    query_microbatch = 64
    started = time.time()
    for epoch, start, end in batches:
        batch = ordered[start:end]
        active_batch = [
            row for row in batch
            if any(candidate in set(row["positives"]) for candidate in row["candidates"])
            and any(candidate not in set(row["positives"]) for candidate in row["candidates"])
        ]
        if not active_batch:
            continue
        batch_rng = _rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            rank_loss_sum = 0.0
            active_lists = 0
            try:
                for micro_start in range(0, len(active_batch), query_microbatch):
                    micro = active_batch[micro_start : micro_start + query_microbatch]
                    micro_losses: list[Tensor] = []
                    for row in micro:
                        relation = row["relation"]
                        anchor = row["anchor_id"]
                        candidates = list(row["candidates"])
                        positives = set(row["positives"])
                        za, zb = bank.z(anchor), bank.z_many(candidates)
                        if is_qt:
                            raw = student.score(za, zb)
                        else:
                            left_kind = "table" if relation.startswith("Q") else relation.removesuffix("_T")
                            right_kind = (
                                "table" if relation == "QT" or relation.endswith("_T")
                                else relation.removeprefix("Q_")
                            )
                            raw = student.score(left_kind, za, right_kind, zb)
                        mask = torch.tensor([c in positives for c in candidates], device=dev)
                        loss = rank_mass_loss(logit_scale * raw, mask)
                        if loss is None:
                            raise RuntimeError(f"{stage}: prevalidated active list became inactive")
                        micro_losses.append(loss)
                    micro_sum = torch.stack(micro_losses).sum()
                    rank_loss_sum += float(micro_sum.detach())
                    active_lists += len(micro_losses)
                    (micro_sum / len(active_batch)).backward()
                anchor_loss = student.anchor_loss()
                if anchor_weight:
                    (anchor_weight * anchor_loss).backward()
                break
            except BaseException as error:
                if not _is_cuda_oom(error) or query_microbatch == 1:
                    if _is_cuda_oom(error):
                        raise RuntimeError(
                            f"BLOCKED_RESOURCE: {stage} query microbatch 1 OOM"
                        ) from error
                    raise
                optimizer.zero_grad(set_to_none=True)
                _restore_rng_state(batch_rng)
                torch.cuda.empty_cache()
                query_microbatch //= 2

        rank_loss_value = rank_loss_sum / active_lists
        total_value = rank_loss_value + anchor_weight * float(anchor_loss.detach())
        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        logical_step += 1
        grad_norm_value = float(grad_norm)
        _log(log_path, {
            "stage": stage, "epoch": epoch, "step": logical_step,
            "record_ids": [row["item_id"] for row in batch],
            "active_lists": active_lists, "batch_lists": len(batch),
            "loss": total_value, "grad_norm_preclip": grad_norm_value,
            "grad_norm_postclip": min(grad_norm_value, clip_norm),
            "rank_mass_loss": rank_loss_value,
            "anchor_loss_unweighted": float(anchor_loss.detach()),
            "rank_loss_denominator_active_lists": active_lists,
            "clip_norm": clip_norm, "P_lr": lr_p, "R_lr": lr_r, "logit_scale": logit_scale,
            "anchor_weight": anchor_weight, "student_query_microbatch": query_microbatch,
            "elapsed_seconds": time.time() - started,
            **_student_parameter_norms(student),
            **_student_drift_spectrum(student),
            **_gpu_peaks(),
        })
        if logical_step in fraction_steps and save_dir:
            for fraction in fraction_steps[logical_step]:
                path = save_dir / (f"snapshot_frac{int(fraction * 100):03d}.pt" if epochs == 1
                                   else f"snapshot_epoch{epoch:03d}.pt")
                save_checkpoint(
                    path, student, optimizer,
                    {**base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                     "next_record_cursor": end,
                     "order_sha256": _order_sha(ordered), "fraction": fraction,
                     "student_query_microbatch": query_microbatch},
                )
                saved[fraction] = path
    if logical_step != total_steps:
        raise RuntimeError(f"{stage}: inactive batch changed registered step count")
    return saved


def _student_c2_scores(
    student: NativeStudent | QTStudent,
    bank: ObjectBank,
    row: dict,
) -> tuple[Tensor, Optional[Tensor], list[tuple[int, str]], list[str]]:
    qid = row["query_id"]
    targets = list(row["targets"])
def _student_c2_scores(
    student: NativeStudent | QTStudent,
    bank: ObjectBank,
    row: dict,
    logit_scale: float = 1.0,
) -> tuple[Tensor, Optional[Tensor], list[tuple[int, str]], list[str]]:
    """Scaled student logits for one C2 record.

    Returns ``(direct, evidence, bag_targets, evidence_ids)``: ``direct`` is one logit per
    target, ``evidence`` one CQET-aggregated logit per target with a non-empty bag (None if
    no bags). Every bilinear score is multiplied by ``logit_scale``; a path logit is the sum
    of the scaled Q-E and E-T scores. Path scores are gathered with index tensors rather than
    a per-path Python loop, which is what made the original C2 step launch-bound.
    """
    qid = row["query_id"]
    targets = list(row["targets"])
    zq, zt = bank.z(qid), bank.z_many(targets)
    if isinstance(student, QTStudent):
        return logit_scale * student.score(zq, zt), None, [], []
    uq = student.u("table", zq)
    ut = student.u("table", zt)
    direct = logit_scale * (uq @ student.R["QT"] * ut).sum(dim=-1)
    bags = row.get("natural_bags", {})
    bag_targets = [(i, t) for i, t in enumerate(targets) if bags.get(t)]
    flat = [(target_i, evidence_id) for target_i, target in bag_targets for evidence_id in bags[target]]
    if not flat:
        return direct, None, bag_targets, []
    unique = list(dict.fromkeys(e for _, e in flat))
    by_modality = {m: [e for e in unique if bank.kind(e) == m] for m in ("text", "image")}
    evidence_ids = by_modality["text"] + by_modality["image"]
    qe_parts, et_parts = [], []
    for modality in ("text", "image"):
        ids = by_modality[modality]
        if not ids:
            continue
        ue = student.u(modality, bank.z_many(ids))
        qe_parts.append((uq @ student.R[f"Q_{modality}"] * ue).sum(dim=-1))
        et_parts.append(ue @ student.R[f"{modality}_T"])
    qe = torch.cat(qe_parts)
    et_projected = torch.cat(et_parts)
    position = {e: i for i, e in enumerate(evidence_ids)}
    path_evidence = torch.tensor([position[e] for _, e in flat], dtype=torch.long, device=direct.device)
    path_target = torch.tensor([target_i for target_i, _ in flat], dtype=torch.long, device=direct.device)
    path_scores = logit_scale * (qe[path_evidence] + (et_projected[path_evidence] * ut[path_target]).sum(dim=-1))
    evidence_all_targets = aggregate_cqet(direct, path_scores, path_target)
    bag_index = torch.tensor([target_i for target_i, _ in bag_targets], dtype=torch.long, device=direct.device)
    return direct, evidence_all_targets[bag_index], bag_targets, evidence_ids


def _teacher_c2_scores(
    teacher: FreshPathTeacher,
    bank: ObjectBank,
    row: dict,
    device: torch.device,
) -> tuple[Tensor, Optional[Tensor]]:
    qid = row["query_id"]
    targets = list(row["targets"])
    bags = row.get("natural_bags", {})
    paths = [(i, e) for i, t in enumerate(targets) for e in bags.get(t, ())]
    evidence_ids = list(dict.fromkeys(e for _, e in paths))
    all_ids = list(dict.fromkeys([qid, *targets, *evidence_ids]))
    tokens = dict(zip(all_ids, bank.tokens_many(all_ids)))
    evidence_map = {e: (bank.kind(e), bank.z(e), tokens[e]) for e in evidence_ids}
    with torch.no_grad():
        direct, path_scores = teacher.score_query_lists(
            (bank.z(qid), tokens[qid]),
            (bank.z_many(targets), [tokens[t] for t in targets]),
            evidence_map,
            paths,
            chunk=TEACHER_INFERENCE_CHUNK,
        )
        if not paths:
            return direct.detach(), None
        target_index = torch.tensor([i for i, _ in paths], dtype=torch.long, device=device)
        aggregate = aggregate_cqet(direct, path_scores, target_index)
        bag_targets = [i for i, t in enumerate(targets) if bags.get(t)]
        evidence = torch.stack([aggregate[i] for i in bag_targets])
    return direct.detach(), evidence.detach()


def build_teacher_logits_cache(
    teacher: FreshPathTeacher,
    bank: ObjectBank,
    c2_records: Sequence[dict],
    *,
    device: str = "cuda:0",
) -> dict[str, tuple[Tensor, Optional[Tensor]]]:
    """Freeze aligned TB_CQET Direct/E logits for the materialized C2 graph."""
    dev = torch.device(device)
    teacher.to(dev).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    result = {}
    started = time.time()
    for i, row in enumerate(c2_records, 1):
        direct, evidence = _teacher_c2_scores(teacher, bank, row, dev)
        result[row["query_id"]] = (
            direct.to(device="cpu", dtype=torch.float32).contiguous(),
            evidence.to(device="cpu", dtype=torch.float32).contiguous() if evidence is not None else None,
        )
        if i % 100 == 0 or i == len(c2_records):
            print(
                f"[TB_CQET C2 logits] {i}/{len(c2_records)} elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    return result


def train_student_c2(
    student: NativeStudent | QTStudent,
    c2_records: list[dict],
    teacher: Optional[FreshPathTeacher],
    bank: ObjectBank,
    device: str = "cuda:0",
    arm: str = "NATIVE_KD",
    logical_batch: int = 64,
    lr_p: float = StudentRecipe.lr_p,
    lr_r: float = StudentRecipe.lr_r,
    logit_scale: float = StudentRecipe.logit_scale,
    kd_weight: float = StudentRecipe.kd_weight,
    kd_temperature: float = StudentRecipe.kd_temperature,
    random_negatives: int = StudentRecipe.random_negatives,
    negative_pool: Optional[Sequence[str]] = None,
    anchor_weight: float = StudentRecipe.anchor_weight,
    clip_norm: float = StudentRecipe.clip_norm,
    save_dir: Optional[Path] = None,
    seed: int = 13,
    expected_parent_hash: Optional[str] = None,
    metadata: Optional[dict] = None,
    resume_from: Optional[Path] = None,
    teacher_logits: Optional[Mapping[str, tuple[Tensor, Optional[Tensor]]]] = None,
    max_updates: Optional[int] = None,
    log_path: Optional[Path] = None,
    epochs: int = 1,
) -> dict[float, Path]:
    """C2 training on the shared graph.

    Per query the loss is ``SUP_direct + SUP_evidence + kd_weight * (KD_direct + KD_evidence)``.
    The direct SUP list is the record's targets plus ``random_negatives`` uniform draws from
    ``negative_pool`` (default: every target that appears in ``c2_records``); the evidence
    list and both KD lists cover the record's targets only, because teacher logits exist only
    there. KD is ``KL(softmax(teacher / kd_temperature) || softmax(student))`` with the
    student logits already multiplied by ``logit_scale``.
    """
    is_qt = isinstance(student, QTStudent)
    is_kd = arm == "NATIVE_KD"
    if is_kd and teacher is None and teacher_logits is None:
        raise ValueError("NATIVE_C2_KD requires TB_CQET end logits or model")
    if is_qt and (is_kd or teacher is not None or teacher_logits is not None):
        raise ValueError("QT-only C2 is SUP-only and independent of CQET")
    enforce_task_numerics()
    dev = torch.device(device)
    student.to(dev).train()
    bank.attach_device(dev)
    parent_hash = model_state_sha(student)
    if expected_parent_hash is None or parent_hash != expected_parent_hash:
        raise ValueError(f"C2 step0 state {parent_hash} != selected C1 {expected_parent_hash}")
    if teacher is not None:
        teacher.to(dev).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    if negative_pool is None:
        negative_pool = sorted({t for row in c2_records for t in row["targets"]}, key=lambda x: x.encode("utf-8"))
    negative_pool = list(negative_pool)

    ordered, batches = _student_schedule(c2_records, "C2_SHARED", seed, epochs, logical_batch)
    order_sha = _order_sha(ordered)
    optimizer = AdamW(student.param_groups(lr_p, lr_r), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    stage = "QT_C2_SUP" if is_qt else ("NATIVE_C2_KD" if is_kd else "NATIVE_C2_SUP")
    total_steps = len(batches)
    fraction_steps = _student_snapshot_steps(total_steps, epochs)
    recipe = {
        "P_lr": lr_p, "R_lr": lr_r, "logit_scale": logit_scale,
        "kd_weight": kd_weight if is_kd else 0.0, "kd_temperature": kd_temperature,
        "random_negatives": random_negatives, "negative_pool_size": len(negative_pool),
        "anchor_weight": anchor_weight, "clip_norm": clip_norm,
    }
    base_meta = {**dict(metadata or {}), "epochs": epochs, "logical_batch": logical_batch, **recipe}
    saved: dict[float, Path] = {}
    logical_step = 0
    next_cursor = 0
    query_microbatch = 64
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
        path = save_dir / "snapshot_frac000.pt"
        save_checkpoint(
            path, student, optimizer,
            {**base_meta, "stage": stage, "epoch": 0, "logical_step": 0,
             "next_record_cursor": 0, "order_sha256": order_sha,
             "parent_state_sha256": parent_hash},
        )
        saved[0.0] = path

    for epoch, start, end in batches:
        if start < next_cursor:
            continue
        if max_updates is not None and logical_step >= max_updates:
            break
        batch = ordered[start:end]
        batch_rng = _rng_state()
        while True:
            optimizer.zero_grad(set_to_none=True)
            query_loss_sum = 0.0
            component_values: dict[str, list[float]] = {
                "direct_sup": [], "evidence_sup": [],
                "direct_kd": [], "evidence_kd": [],
            }
            try:
                for micro_start in range(0, len(batch), query_microbatch):
                    micro = batch[micro_start : micro_start + query_microbatch]
                    query_losses: list[Tensor] = []
                    for row in micro:
                        qid = row["query_id"]
                        targets = list(row["targets"])
                        positives = set(row["positives"])
                        direct_s, evidence_s, bag_targets, _ = _student_c2_scores(student, bank, row, logit_scale)
                        direct_positive = torch.tensor([t in positives for t in targets], device=dev)
                        negative_ids = _random_negative_ids(
                            negative_pool, set(targets) | positives, random_negatives,
                            "C2_SHARED", seed, epoch, qid,
                        )
                        if negative_ids:
                            zq, zn = bank.z(qid), bank.z_many(negative_ids)
                            random_s = logit_scale * (
                                student.score(zq, zn) if is_qt else student.score("table", zq, "table", zn)
                            )
                            sup_scores = torch.cat([direct_s, random_s])
                            sup_positive = torch.cat([direct_positive, torch.zeros(len(negative_ids), dtype=torch.bool, device=dev)])
                        else:
                            sup_scores, sup_positive = direct_s, direct_positive
                        direct_sup = rank_mass_loss(sup_scores, sup_positive)
                        if direct_sup is None:
                            raise ValueError(f"{qid}: C2 Direct supervision is inactive")
                        component_values["direct_sup"].append(float(direct_sup.detach()))
                        query_loss = direct_sup

                        evidence_sup = None
                        if evidence_s is not None:
                            evidence_positive = torch.tensor(
                                [t in positives for _, t in bag_targets], device=dev
                            )
                            evidence_sup = rank_mass_loss(evidence_s, evidence_positive)
                            if evidence_sup is not None:
                                component_values["evidence_sup"].append(float(evidence_sup.detach()))
                                query_loss = query_loss + evidence_sup

                        if is_kd:
                            if teacher_logits is not None and qid in teacher_logits:
                                direct_t, evidence_t = teacher_logits[qid]
                                direct_t = direct_t.to(dev)
                                evidence_t = evidence_t.to(dev) if evidence_t is not None else None
                            else:
                                direct_t, evidence_t = _teacher_c2_scores(teacher, bank, row, dev)
                            direct_kd = list_kl_divergence(direct_s, direct_t / kd_temperature)
                            if direct_kd is None or not direct_kd.requires_grad:
                                raise RuntimeError(f"{qid}: Direct KD did not produce a Student graph")
                            component_values["direct_kd"].append(float(direct_kd.detach()))
                            query_loss = query_loss + kd_weight * direct_kd
                            if evidence_sup is not None:
                                if evidence_s is None or evidence_t is None:
                                    raise RuntimeError(
                                        f"{qid}: active evidence SUP has no aligned Teacher logits"
                                    )
                                evidence_kd = list_kl_divergence(evidence_s, evidence_t / kd_temperature)
                                if evidence_kd is None or not evidence_kd.requires_grad:
                                    raise RuntimeError(
                                        f"{qid}: evidence KD did not produce a Student graph"
                                    )
                                component_values["evidence_kd"].append(float(evidence_kd.detach()))
                                query_loss = query_loss + kd_weight * evidence_kd
                        query_losses.append(query_loss)
                    micro_sum = torch.stack(query_losses).sum()
                    query_loss_sum += float(micro_sum.detach())
                    (micro_sum / len(batch)).backward()
                anchor = anchor_weight * student.anchor_loss()
                anchor_value = float(anchor.detach())
                if anchor_weight:
                    anchor.backward()
                break
            except BaseException as error:
                if not _is_cuda_oom(error) or query_microbatch == 1:
                    if _is_cuda_oom(error):
                        raise RuntimeError(
                            f"BLOCKED_RESOURCE: {stage} query microbatch 1 OOM"
                        ) from error
                    raise
                optimizer.zero_grad(set_to_none=True)
                _restore_rng_state(batch_rng)
                torch.cuda.empty_cache()
                query_microbatch //= 2

        total_value = query_loss_sum / len(batch) + anchor_value
        grad_norm = nn.utils.clip_grad_norm_(student.parameters(), clip_norm)
        optimizer.step()
        logical_step += 1
        next_cursor = end
        grad_norm_value = float(grad_norm)
        _log(log_path, {
            "stage": stage, "epoch": epoch, "step": logical_step,
            "record_ids": [row["query_id"] for row in batch],
            "active_queries": len(batch), "batch_queries": len(batch),
            "loss": total_value, "grad_norm_preclip": grad_norm_value,
            "grad_norm_postclip": min(grad_norm_value, clip_norm),
            **{
                f"{name}_loss": (float(np.mean(values)) if values else None)
                for name, values in component_values.items()
            },
            **{
                f"{name}_denominator": len(values)
                for name, values in component_values.items()
            },
            "anchor_loss_weighted": anchor_value,
            **recipe,
            "parent_state_sha256": parent_hash,
            "student_query_microbatch": query_microbatch,
            "loss_denominator_active_queries": len(batch),
            "elapsed_seconds": time.time() - started,
            **_student_parameter_norms(student),
            **_student_drift_spectrum(student),
            **_gpu_peaks(),
        })
        if teacher is not None and any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Teacher acquired gradients during C2 KD")
        if logical_step in fraction_steps and save_dir:
            for fraction in fraction_steps[logical_step]:
                path = save_dir / (f"snapshot_frac{int(fraction * 100):03d}.pt" if epochs == 1
                                   else f"snapshot_epoch{epoch:03d}.pt")
                save_checkpoint(
                    path, student, optimizer,
                    {**base_meta, "stage": stage, "epoch": epoch, "logical_step": logical_step,
                     "next_record_cursor": next_cursor, "order_sha256": order_sha,
                     "parent_state_sha256": parent_hash, "fraction": fraction,
                     "student_query_microbatch": query_microbatch},
                )
                saved[fraction] = path
    if max_updates is None and logical_step != total_steps:
        raise RuntimeError(f"{stage}: completed {logical_step} steps, expected {total_steps}")
    return saved

"""Model construction and batched calls for the shared Teacher and Student.

Spec sections 5, 6, 8.4 and 9.  Everything here delegates to ``reference`` so
the production path cannot drift from the reference equations.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from . import reference
from .cache import MODALITY_IMAGE, MODALITY_TABLE, MODALITY_TEXT, TEXT_KINDS, TABLE_KINDS, IMAGE_KINDS_FULL
from .config import ConfigError

SLOT_COUNT = 9
ROLE_QUERY = 0
ROLE_CANDIDATE = 1
ROLE_EVIDENCE = 2
MODE_P = 0
MODE_J = 1

MODALITY_ID = {"table": MODALITY_TABLE, "text": MODALITY_TEXT, "image": MODALITY_IMAGE}
KIND_BY_MODALITY = {
    "table": TABLE_KINDS,
    "text": TEXT_KINDS,
    "image": IMAGE_KINDS_FULL,
}


def build_teacher(config: dict[str, Any]) -> reference.UnifiedTeacher:
    return reference.UnifiedTeacher(
        input_dim=int(config["input_dim"]),
        d=int(config["dimension"]),
        heads=int(config["heads"]),
        ffn=int(config["ffn_dimension"]),
        kinds=int(config["slot_kind_count"]),
    )


def build_student(config: dict[str, Any]) -> reference.CompactStudent:
    return reference.CompactStudent(
        input_dim=int(config["input_dim"]),
        d=int(config["dimension"]),
        rank=int(config["interaction_rank"]),
        kinds=int(config["slot_kind_count"]),
    )


def set_seed(seed: int) -> None:
    """Reset Python/NumPy/Torch RNGs; dataloader workers are always 0."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def no_decay_parameters(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Split names into decay / no-decay groups (spec section 8.3)."""
    markers = ("bias", "norm", "modality", "kind", "role", "task", "rel", "pool_query")
    decay: list[str] = []
    no_decay: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if parameter.ndim <= 1 or any(marker in lowered for marker in markers):
            no_decay.append(name)
        else:
            decay.append(name)
    return decay, no_decay


def build_optimizer(
    model: torch.nn.Module, optimizer_config: dict[str, Any], lr: float
) -> torch.optim.AdamW:
    decay, no_decay = no_decay_parameters(model)
    groups = [
        {"params": [p for n, p in model.named_parameters() if n in set(decay)], "weight_decay": float(optimizer_config["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if n in set(no_decay)], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=lr,
        betas=tuple(float(b) for b in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )


def lr_multiplier(step: int, total_steps: int, warmup_fraction: float, end_fraction: float) -> float:
    """Spec Eq. in section 8.4 (the factor applied to the base learning rate)."""
    if total_steps <= 0:
        raise ConfigError("total_steps must be positive")
    warmup = max(1, int(np.ceil(warmup_fraction * total_steps)))
    if step <= warmup:
        return step / warmup
    if total_steps == warmup:
        return 1.0
    import math

    progress = (step - warmup) / (total_steps - warmup)
    return end_fraction + (1.0 - end_fraction) * (1.0 + math.cos(math.pi * progress)) / 2.0


class ObjectBank:
    """Compact frozen features for the whole object set, in canonical order."""

    def __init__(
        self,
        object_ids: list[str],
        modality_id: np.ndarray,
        kind_ids: np.ndarray,
        z: np.ndarray,
        summary: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        self.object_ids = list(object_ids)
        self.position = {o: i for i, o in enumerate(self.object_ids)}
        self.modality_id = modality_id
        self.kind_ids = kind_ids
        self.z = z
        self.summary = summary
        self.mask = mask

    @classmethod
    def load(cls, manifest_rows: Sequence[dict[str, Any]], shards_root, dim: int, slots: int) -> "ObjectBank":
        from .cache import load_shard_vectors

        rows = list(manifest_rows)
        ids = [r["object_id"] for r in rows]
        modality_id = np.array([int(r["modality_id"]) for r in rows], dtype=np.int64)
        kind_ids = np.array([list(r["kind_ids"]) for r in rows], dtype=np.int64)
        z, summary, mask = load_shard_vectors(shards_root, rows, dim, slots)
        return cls(ids, modality_id, kind_ids, z, summary, mask)

    def slots(self, object_id: str, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, int]:
        index = self.position[object_id]
        cache = torch.from_numpy(
            np.concatenate(
                [self.z[index][None, :], self.summary[index].astype(np.float32)], axis=0
            )
        ).to(device)
        valid = torch.from_numpy(self.mask[index].astype(bool)).to(device)
        return cache, valid, int(self.modality_id[index])

    def keys(self, object_ids: Sequence[str], device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Raw (z, C, mask, modality) tensors for a set of objects."""
        index = np.array([self.position[o] for o in object_ids], dtype=np.int64)
        cache = torch.from_numpy(
            np.concatenate(
                [self.z[index][:, None, :], self.summary[index].astype(np.float32)], axis=1
            )
        ).to(device)
        valid = torch.from_numpy(self.mask[index].astype(bool)).to(device)
        modality = torch.from_numpy(self.modality_id[index]).to(device)
        kind = torch.from_numpy(self.kind_ids[index]).to(device)
        return cache, valid, modality, kind


class TeacherBatch:
    """Builds Teacher inputs: one [B, O, 9, D] tensor plus validity/role/kind."""

    def __init__(self, bank: ObjectBank, device: torch.device | str = "cpu") -> None:
        self.bank = bank
        self.device = device

    def build(
        self,
        queries: Sequence[str],
        candidates: Sequence[Sequence[str]],
        evidences: Sequence[Sequence[str]] | None,
        mode: int,
    ) -> dict[str, torch.Tensor]:
        """One row per query; row i scores ``candidates[i]`` under context ``evidences[i]``.

        ``candidates[i]`` is a flat list of object ids in the caller's own order.
        ``evidences[i]`` is already canonical-ordered (spec 6) or empty for mode P.
        """
        batch = len(queries)
        if len(candidates) != batch:
            raise ConfigError("candidate list must align with the query list")
        if mode == MODE_P and evidences is not None and any(evidences):
            raise ConfigError("mode=P must be called with an empty context")
        width = max(
            (1 + len(c) + (len(evidences[i]) if evidences else 0)
             for i, c in enumerate(candidates)),
            default=1,
        )
        dim = self.bank.z.shape[1]
        cache = torch.zeros(batch, width, SLOT_COUNT, dim, dtype=torch.float32)
        valid = torch.zeros(batch, width, SLOT_COUNT, dtype=torch.bool)
        modality = torch.zeros(batch, width, dtype=torch.long)
        role = torch.zeros(batch, width, dtype=torch.long)
        kind = torch.zeros(batch, width, SLOT_COUNT, dtype=torch.long)
        for i in range(batch):
            self._fill(cache[i, 0], valid[i, 0], modality[i, 0], role[i, 0], kind[i, 0],
                       queries[i], ROLE_QUERY)
            for j, candidate_id in enumerate(candidates[i]):
                self._fill(cache[i, 1 + j], valid[i, 1 + j], modality[i, 1 + j],
                           role[i, 1 + j], kind[i, 1 + j], candidate_id, ROLE_CANDIDATE)
            if evidences:
                for j, evidence_id in enumerate(evidences[i]):
                    self._fill(cache[i, 1 + len(candidates[i]) + j],
                               valid[i, 1 + len(candidates[i]) + j],
                               modality[i, 1 + len(candidates[i]) + j],
                               role[i, 1 + len(candidates[i]) + j],
                               kind[i, 1 + len(candidates[i]) + j],
                               evidence_id, ROLE_EVIDENCE)
        return {
            "cache": cache.to(self.device),
            "valid": valid.to(self.device),
            "modality": modality.to(self.device),
            "role": role.to(self.device),
            "kind": kind.to(self.device),
        }

    def score(
        self,
        teacher: reference.UnifiedTeacher,
        queries: Sequence[str],
        candidates: Sequence[Sequence[str]],
        evidences: Sequence[Sequence[str]] | None,
        mode: int,
        chunk: int | None = 8,
    ) -> list[torch.Tensor]:
        """Spec Eq. (1).  Chunking splits the candidate axis only.

        All candidate logits for a list are concatenated before the list loss is
        computed, so neither the softmax denominator nor the negative competition
        changes with the chunk size.
        """
        batch = len(queries)
        if batch == 0:
            return []
        if len(candidates) != batch:
            raise ConfigError("candidate list must align with the query list")
        span = max(len(c) for c in candidates) if candidates else 0
        if span == 0:
            empties = [torch.zeros(0, device=self.device) for _ in range(batch)]
            return empties
        if chunk is None or chunk <= 0:
            chunk = span
        scores: list[list[torch.Tensor | None]] = [[None] * len(candidates[i]) for i in range(batch)]
        for start in range(0, span, chunk):
            stop = min(start + chunk, span)
            block_queries: list[str] = []
            block_candidates: list[str] = []
            block_evidences: list[list[str]] = []
            slots: list[tuple[int, int]] = []
            for i in range(batch):
                for j in range(start, min(stop, len(candidates[i]))):
                    block_queries.append(queries[i])
                    block_candidates.append(candidates[i][j])
                    block_evidences.append(list(evidences[i]) if evidences else [])
                    slots.append((i, j))
            if not block_queries:
                continue
            built = self.build(block_queries, [[c] for c in block_candidates],
                               block_evidences, mode)
            built["mode"] = torch.full(
                (len(block_queries),), mode, dtype=torch.long, device=self.device
            )
            logits = teacher(**built)
            for (i, j), value in zip(slots, logits):
                scores[i][j] = value
        output: list[torch.Tensor] = []
        for row in scores:
            if any(value is None for value in row):
                raise ConfigError("internal error: a candidate logit was never computed")
            output.append(torch.stack([value for value in row]) if row else torch.zeros(0, device=self.device))
        return output

    def _fill(
        self,
        cache: torch.Tensor,
        valid: torch.Tensor,
        modality: torch.Tensor,
        role: torch.Tensor,
        kind: torch.Tensor,
        object_id: str,
        role_id: int,
    ) -> None:
        index = self.bank.position[object_id]
        slots = torch.from_numpy(
            np.concatenate(
                [self.bank.z[index][None, :], self.bank.summary[index].astype(np.float32)],
                axis=0,
            )
        )
        cache.copy_(slots)
        valid.copy_(torch.from_numpy(self.bank.mask[index].astype(bool)))
        modality.fill_(int(self.bank.modality_id[index]))
        role.fill_(role_id)
        kind.copy_(torch.from_numpy(self.bank.kind_ids[index]))


def student_object_keys(
    student: reference.CompactStudent,
    bank: ObjectBank,
    object_ids: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    """Spec Eq. (11)-(12): one key per object."""
    cache, valid, modality, kind = bank.keys(list(object_ids), device)
    return student.encode(cache, valid, modality, kind)


@torch.no_grad()
def project_corpus_keys(
    student: reference.CompactStudent,
    bank: ObjectBank,
    object_ids: Sequence[str],
    path: str,
    device: torch.device | str,
    block: int = 4096,
) -> np.ndarray:
    """Return one raw, unit object key per destination object.

    Relation transforms belong exclusively to the query side.  ``path`` remains
    in the API to make call sites explicit, but all three indexes store ``nu_x``.
    """
    student.eval()
    out = np.zeros((len(object_ids), student.d), dtype=np.float32)
    if path not in {"D", "E", "C"}:
        raise ConfigError(f"unknown Student index path: {path}")
    for start in range(0, len(object_ids), block):
        chunk = list(object_ids[start : start + block])
        vector = student_object_keys(student, bank, chunk, device)
        out[start : start + block] = reference.unit(vector).cpu().numpy()
    return out


@torch.no_grad()
def conditional_keys(
    student: reference.CompactStudent,
    bank: ObjectBank,
    query_ids: Sequence[str],
    evidence_ids: Sequence[str],
    device: torch.device | str,
) -> np.ndarray:
    """The Q-conditioned second-hop query of spec Eq. (14), batched."""
    student.eval()
    q = student_object_keys(student, bank, list(query_ids), device)
    e = student_object_keys(student, bank, list(evidence_ids), device)
    return student.query_next(q, e).cpu().numpy()

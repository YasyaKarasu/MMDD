"""Frozen Teacher-logit cache used by Student distillation stages.

The cache deliberately stores only detached scalar logits.  Learned Teacher
representations (or any trainable Student/Teacher tensor) never enter it.  A
cache key carries every piece of context that can change a distillation
target: branch/checkpoint hash, protocol hash, query/evidence/target IDs,
ordered candidate IDs, mask, and view name.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import torch

from . import lineage


def _mask_tuple(mask: Sequence[bool] | None, n: int) -> tuple[bool, ...]:
    if mask is None:
        return (True,) * n
    out = tuple(bool(x) for x in mask)
    if len(out) != n:
        raise ValueError("Teacher-logit cache mask must align with candidate IDs")
    return out


@dataclass
class FrozenTeacherLogitCache:
    """Small in-memory cache with optional atomic persistence.

    Values are kept on CPU in float32 and copied to the caller's device on a
    hit.  ``teacher_hash`` and ``protocol_hash`` are mandatory even for an
    in-memory cache so a cache cannot accidentally be reused across Teacher
    branches or protocol revisions.
    """

    teacher_branch: str
    teacher_hash: str
    protocol_hash: str
    path: Path | None = None
    entries: dict[str, dict] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @classmethod
    def for_model(cls, model, protocol_path: Path, *, teacher_branch: str,
                  path: Path | None = None):
        return cls(
            teacher_branch=str(teacher_branch),
            teacher_hash=lineage.tensor_state_hash(model.state_dict()),
            protocol_hash=lineage.protocol_hash(protocol_path),
            path=Path(path) if path is not None else None,
        )

    def _metadata(
        self,
        *,
        kind: str,
        query_id,
        evidence_id,
        target_id,
        candidate_ids: Sequence[str],
        mask: Sequence[bool] | None,
        view: str,
    ) -> dict:
        candidates = tuple(str(x) for x in candidate_ids)
        def field(value):
            if value is None:
                return None
            if isinstance(value, (tuple, list)):
                return [str(x) for x in value]
            return str(value)

        return {
            "teacher_branch": self.teacher_branch,
            "teacher_hash": self.teacher_hash,
            "protocol_hash": self.protocol_hash,
            "kind": str(kind),
            "q": field(query_id),
            "e": field(evidence_id),
            "t": field(target_id),
            "candidate_ids": candidates,
            "mask": _mask_tuple(mask, len(candidates)),
            "view": str(view),
        }

    @staticmethod
    def _digest(metadata: dict) -> str:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_or_compute(
        self,
        *,
        kind: str,
        query_id,
        evidence_id,
        target_id,
        candidate_ids: Sequence[str],
        mask: Sequence[bool] | None,
        view: str,
        device,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        metadata = self._metadata(
            kind=kind,
            query_id=query_id,
            evidence_id=evidence_id,
            target_id=target_id,
            candidate_ids=candidate_ids,
            mask=mask,
            view=view,
        )
        digest = self._digest(metadata)
        record = self.entries.get(digest)
        if record is not None:
            self.hits += 1
            return record["values"].to(device)
        self.misses += 1
        # The caller only supplies a frozen/eval Teacher.  Detaching here is a
        # final guard against retaining a computation graph in the cache.
        values = compute().detach().to(dtype=torch.float32, device="cpu").contiguous()
        if values.ndim != 1 or values.shape[0] != len(candidate_ids):
            raise ValueError("Teacher-logit cache value does not align with candidates")
        self.entries[digest] = {"metadata": metadata, "values": values}
        return values.to(device)

    def save(self, path: Path | None = None) -> None:
        destination = Path(path) if path is not None else self.path
        if destination is None:
            return
        payload = {
            "format": 2,
            "teacher_branch": self.teacher_branch,
            "teacher_hash": self.teacher_hash,
            "protocol_hash": self.protocol_hash,
            "entries": self.entries,
        }
        lineage.atomic_save(destination, payload)

    @classmethod
    def load(cls, path: Path, *, teacher_branch: str, teacher_hash: str,
             protocol_hash: str):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("format") != 2:
            raise ValueError(f"unsupported Teacher-logit cache format: {path}")
        if (payload.get("teacher_branch") != teacher_branch
                or payload.get("teacher_hash") != teacher_hash
                or payload.get("protocol_hash") != protocol_hash):
            raise ValueError(f"Teacher-logit cache metadata mismatch: {path}")
        return cls(teacher_branch, teacher_hash, protocol_hash, Path(path),
                   dict(payload.get("entries", {})))

    def load_existing(self) -> "FrozenTeacherLogitCache":
        if self.path is None or not self.path.exists():
            return self
        loaded = self.load(
            self.path,
            teacher_branch=self.teacher_branch,
            teacher_hash=self.teacher_hash,
            protocol_hash=self.protocol_hash,
        )
        self.entries = loaded.entries
        return self

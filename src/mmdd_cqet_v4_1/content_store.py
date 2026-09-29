"""Vendored reader for the locked pure-content chunk format."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def chunk_stem(chunk_dir: Path, index: int) -> Path:
    return Path(chunk_dir) / f"chunk_{index:06d}"


def chunk_files(chunk_dir: Path, index: int) -> dict[str, Path]:
    stem = chunk_stem(chunk_dir, index)
    return {name: Path(f"{stem}.{name}.npy") for name in ("ids", "types", "lens", "tokens")}


class ContentStore:
    """Random-access reader for immutable pure-content token chunks."""

    def __init__(self, root: Path, *, lru_bytes: int = 0) -> None:
        self.root = Path(root)
        payload = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.ids: list[str] = payload["ids"]
        self.types: list[str] = payload["types"]
        self.chunks = payload["chunks"]
        self.rows = payload["rows"]
        self.index = {oid: i for i, oid in enumerate(self.ids)}
        self.chunk_dir = self.root / "chunks"
        self._open: dict[int, dict[str, np.ndarray]] = {}
        self._lru: dict[str, torch.Tensor] = {}
        self._lru_used = 0
        self._lru_limit = lru_bytes

    def has(self, object_id: str) -> bool:
        return object_id in self.index

    def __len__(self) -> int:
        return len(self.ids)

    def _chunk(self, index: int) -> dict[str, np.ndarray]:
        if index not in self._open:
            files = chunk_files(self.chunk_dir, index)
            lens = np.load(files["lens"], allow_pickle=False)
            self._open[index] = {
                "ids": np.load(files["ids"], allow_pickle=False),
                "lens": lens,
                "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                "tokens": np.load(files["tokens"], mmap_mode="r", allow_pickle=False),
            }
            if len(self._open) > 8:
                self._open.pop(next(iter(self._open)))
        return self._open[index]

    def get(self, object_id: str) -> torch.Tensor:
        cached = self._lru.pop(object_id, None)
        if cached is not None:
            self._lru[object_id] = cached
            return cached
        i = self.index[object_id]
        chunk = self._chunk(int(self.chunks[i]))
        row = int(self.rows[i])
        start = int(chunk["offsets"][row])
        length = int(chunk["lens"][row])
        tensor = torch.from_numpy(np.array(chunk["tokens"][start : start + length], copy=True))
        if self._lru_limit:
            nbytes = tensor.numel() * tensor.element_size()
            while self._lru and self._lru_used + nbytes > self._lru_limit:
                victim = self._lru.pop(next(iter(self._lru)))
                self._lru_used -= victim.numel() * victim.element_size()
            if nbytes <= self._lru_limit:
                self._lru[object_id] = tensor
                self._lru_used += nbytes
        return tensor

    def ids_by_type(self, object_type: str) -> list[str]:
        return [oid for oid, kind in zip(self.ids, self.types) if kind == object_type]

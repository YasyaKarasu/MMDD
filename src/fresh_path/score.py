"""Object access and batched Teacher scoring.

Bridges the frozen pure-feature layer (:mod:`fresh_path.features`) and the
fresh Teacher/Student models.  Nothing here trains; it only assembles model
inputs for the exact objects a stage asks for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch

from .candidates import RawStore


class ObjectBank:
    """z + content tokens for every object, with the run-wide CPU LRU budget.

    For training stages the caller may attach a CUDA device: z then lives on the
    GPU (one transfer for the whole run) and recently used token blocks are kept
    in a bounded device-side LRU.  Both are pure caches -- they change no maths,
    only how many tiny host-to-device copies the loop has to issue.
    """

    def __init__(self, z_store: RawStore, content, *, lru_bytes: int = 8 * 2**30,
                 device: str | None = None, gpu_token_bytes: int = 1 * 2**30) -> None:
        self.z_store = z_store
        self.content = content
        # Plain dicts used as LRU: re-inserting a key on a hit moves it to the
        # end in O(1), and the running byte counters avoid the previous
        # O(cache size) `sum(numel)` scan on every miss.
        self._lru: dict[str, torch.Tensor] = {}
        self._lru_used = 0
        self._lru_limit = lru_bytes
        self._device: torch.device | None = None
        self._z_gpu: torch.Tensor | None = None
        self._gpu_tokens: dict[str, torch.Tensor] = {}
        self._gpu_token_bytes = gpu_token_bytes
        self._gpu_token_used = 0
        if device is not None:
            self.attach_device(device)

    def attach_device(self, device: str | torch.device) -> None:
        self._device = torch.device(device)
        if self._z_gpu is None:
            self._z_gpu = self.z_store.z.to(self._device)

    def kind(self, object_id: str) -> str:
        return self.z_store.types[self.z_store.index[object_id]]

    def z(self, object_id: str) -> torch.Tensor:
        if self._z_gpu is not None:
            return self._z_gpu[self.z_store.index[object_id]]
        return self.z_store.z[self.z_store.index[object_id]]

    def z_many(self, object_ids: Sequence[str]) -> torch.Tensor:
        index = torch.tensor([self.z_store.index[i] for i in object_ids], dtype=torch.long)
        if self._z_gpu is not None:
            return self._z_gpu[index.to(self._device)]
        return self.z_store.z[index]

    def tokens(self, object_id: str) -> torch.Tensor:
        if self._device is not None:
            cached = self._gpu_tokens.pop(object_id, None)
            if cached is not None:                     # hit: move to the end (LRU)
                self._gpu_tokens[object_id] = cached
                return cached
            # ContentStore keeps the contract's float16 tokens on CPU.  Move
            # those exact bits first, then expand on the device; this yields the
            # same float32 values as the prior CPU-side expansion.
            tensor = self.content.get(object_id).to(self._device).to(torch.float32)
            nbytes = tensor.numel() * tensor.element_size()
            while self._gpu_tokens and self._gpu_token_used + nbytes > self._gpu_token_bytes:
                victim = next(iter(self._gpu_tokens))
                self._gpu_token_used -= self._gpu_tokens.pop(victim).numel() * 4
            if nbytes <= self._gpu_token_bytes:
                self._gpu_tokens[object_id] = tensor
                self._gpu_token_used += nbytes
            return tensor
        cached = self._lru.pop(object_id, None)
        if cached is not None:
            self._lru[object_id] = cached
            return cached
        tensor = self.content.get(object_id).to(torch.float32)
        nbytes = tensor.numel() * 4
        while self._lru and self._lru_used + nbytes > self._lru_limit:
            victim = next(iter(self._lru))
            self._lru_used -= self._lru.pop(victim).numel() * 4
        self._lru[object_id] = tensor
        self._lru_used += nbytes
        return tensor

    def has(self, object_id: str) -> bool:
        return object_id in self.z_store.index and self.content.has(object_id)


def chunks(items: Sequence, size: int) -> Iterator[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class TeacherScorer:
    """Scores text-free pair and QET inputs on one frozen Teacher branch."""

    def __init__(self, model, bank: ObjectBank, *, device: str = "cuda:0", path_batch: int = 32) -> None:
        self.model = model
        self.bank = bank
        self.device = torch.device(device)
        self.path_batch = path_batch

    @torch.no_grad()
    def pair_scores(self, anchor_id: str, destination_ids: Sequence[str], *, batch: int | None = None) -> list[float]:
        """f(anchor, EMPTY, dest) for every destination, in the given order."""
        batch = batch or self.path_batch
        out: list[float] = []
        an_kind = self.bank.kind(anchor_id)
        az = self.bank.z(anchor_id).to(self.device)
        ac = self.bank.tokens(anchor_id).to(self.device)
        for group in chunks(list(destination_ids), batch):
            pairs = []
            for did in group:
                dk = self.bank.kind(did)
                pairs.append((an_kind, az, ac, dk, self.bank.z(did).to(self.device),
                              self.bank.tokens(did).to(self.device)))
            scores = self.model.score_pairs(pairs)
            out.extend(float(x) for x in scores)
        return out

    @torch.no_grad()
    def qet_scores(self, query_id: str, evidence_id: str, destination_ids: Sequence[str], *,
                   batch: int | None = None) -> list[float]:
        batch = batch or self.path_batch
        out: list[float] = []
        qz = self.bank.z(query_id).to(self.device)
        qc = self.bank.tokens(query_id).to(self.device)
        ez = self.bank.z(evidence_id).to(self.device)
        ec = self.bank.tokens(evidence_id).to(self.device)
        ek = self.bank.kind(evidence_id)
        for group in chunks(list(destination_ids), batch):
            triplets = []
            for did in group:
                dk = self.bank.kind(did)
                triplets.append(("table", qz, qc, ek, ez, ec, dk,
                                 self.bank.z(did).to(self.device), self.bank.tokens(did).to(self.device)))
            out.extend(float(x) for x in self.model.score_triplets(triplets))
        return out

    @torch.no_grad()
    def zero_hop(self, query_id: str, destination_ids: Sequence[str], *, batch: int | None = None) -> list[float]:
        return self.pair_scores(query_id, destination_ids, batch=batch)

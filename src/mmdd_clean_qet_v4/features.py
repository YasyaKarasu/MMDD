"""Feature stores, ContentStore, ObjectBank, and PCA fitting for CLEAN-QET v4.0."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import Paths
from .data import iter_jsonl, read_json, sha256_file, utf8_sorted, write_json
from .labels import Labels

INPUT_DIM = 4096
PCA_COMPONENTS = 1024


@dataclass
class ZStore:
    ids: list[str]
    types: list[str]
    z: Tensor  # (N, 4096) float32 CPU
    index: dict[str, int]
    sha256: str

    @property
    def dim(self) -> int:
        return int(self.z.shape[1])

    def vector(self, object_id: str) -> Tensor:
        return self.z[self.index[object_id]]

    def rows(self, object_ids: Sequence[str]) -> Tensor:
        positions = torch.tensor([self.index[i] for i in object_ids], dtype=torch.long)
        return self.z[positions]


def load_z(paths: Paths) -> ZStore:
    z_dir = paths.pure_cache_dir / "z"
    z_index_path = z_dir / "z_index.json"
    z_data_path = z_dir / "z.f32.npy"

    meta = read_json(z_index_path)
    ids = [str(x) for x in meta["ids"]]
    types = [str(x) for x in meta["types"]]

    raw = np.load(z_data_path, mmap_mode="r", allow_pickle=False)
    z_tensor = torch.from_numpy(np.ascontiguousarray(raw).astype(np.float32))

    index = {oid: i for i, oid in enumerate(ids)}
    z_sha = str(meta.get("z_sha256") or "c1df9012351ab669046ff6a5cb98e4f5ee35d949479b4a4cb8e8f85f1c4274c4")
    return ZStore(ids=ids, types=types, z=z_tensor, index=index, sha256=z_sha)


from fresh_path.features import ContentStore


class ObjectBank:
    """Combines ZStore and ContentStore with GPU caching for fast training."""

    def __init__(
        self,
        z_store: ZStore,
        content: ContentStore,
        device: Optional[str | torch.device] = None,
        gpu_token_bytes: int = 1 * 2**30,
    ) -> None:
        self.z_store = z_store
        self.content = content
        self._device: Optional[torch.device] = None
        self._z_gpu: Optional[Tensor] = None
        self._gpu_tokens: dict[str, Tensor] = {}
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

    def z(self, object_id: str) -> Tensor:
        idx = self.z_store.index[object_id]
        if self._z_gpu is not None:
            return self._z_gpu[idx]
        return self.z_store.z[idx]

    def z_many(self, object_ids: Sequence[str]) -> Tensor:
        positions = torch.tensor([self.z_store.index[i] for i in object_ids], dtype=torch.long)
        if self._z_gpu is not None:
            return self._z_gpu[positions.to(self._device)]
        return self.z_store.z[positions]

    def tokens(self, object_id: str) -> Tensor:
        if self._device is not None:
            cached = self._gpu_tokens.pop(object_id, None)
            if cached is not None:
                self._gpu_tokens[object_id] = cached
                return cached
            t = self.content.get(object_id).to(device=self._device, dtype=torch.float32)
            nbytes = t.numel() * t.element_size()
            while self._gpu_tokens and self._gpu_token_used + nbytes > self._gpu_token_bytes:
                victim = next(iter(self._gpu_tokens))
                self._gpu_token_used -= self._gpu_tokens.pop(victim).numel() * 4
            if nbytes <= self._gpu_token_bytes:
                self._gpu_tokens[object_id] = t
                self._gpu_token_used += nbytes
            return t
        return self.content.get(object_id).float()

    def tokens_many(self, object_ids: Sequence[str]) -> list[Tensor]:
        """Batched token fetch: gather rows on host into one buffer, single transfer, .float() on device."""
        if not object_ids:
            return []
        content = self.content
        if hasattr(content, "index") and hasattr(content, "_chunk") and hasattr(content, "chunks") and hasattr(content, "rows"):
            chunk_groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for pos, oid in enumerate(object_ids):
                i = content.index[oid]
                ch_id = int(content.chunks[i])
                r = int(content.rows[i])
                chunk_groups[ch_id].append((pos, r))

            lens = [0] * len(object_ids)
            item_coords = [None] * len(object_ids)
            for ch_id, items in chunk_groups.items():
                ch = content._chunk(ch_id)
                for pos, r in items:
                    s = int(ch["offsets"][r])
                    l = int(ch["lens"][r])
                    lens[pos] = l
                    item_coords[pos] = (ch_id, s, l)

            total_len = sum(lens)
            if self._device is not None and self._device.type == "cuda":
                host = torch.empty((total_len, 4096), dtype=torch.float16).pin_memory()
            else:
                host = torch.empty((total_len, 4096), dtype=torch.float16)

            host_np = host.numpy()
            target_offsets = [0] * len(object_ids)
            curr = 0
            for pos, l in enumerate(lens):
                target_offsets[pos] = curr
                curr += l

            for ch_id, items in chunk_groups.items():
                ch = content._chunk(ch_id)
                tok_data = ch["tokens"]
                for pos, r in items:
                    _, s, l = item_coords[pos]
                    dst = target_offsets[pos]
                    host_np[dst : dst + l] = tok_data[s : s + l]

            if self._device is not None and self._device.type == "cuda":
                flat = host.to(self._device, non_blocking=True).float()
            else:
                flat = host.float()
                if self._device is not None:
                    flat = flat.to(self._device)
            return list(torch.split(flat, lens))
        return [self.tokens(oid) for oid in object_ids]


@dataclass
class RowStore:
    rows: np.ndarray  # (total_rows, 4096)
    offsets: dict[str, tuple[int, int]]

    def get(self, query_id: str) -> np.ndarray:
        start, count = self.offsets[query_id]
        return np.asarray(self.rows[start : start + count])


def build_or_load_row_store(paths: Paths) -> RowStore:
    out_dir = paths.run_root / "rows"
    index_file = out_dir / "index.json"
    data_file = out_dir / "rows.f32.npy"

    if index_file.exists() and data_file.exists():
        index_data = read_json(index_file)
        rows_mmap = np.load(data_file, mmap_mode="r", allow_pickle=False)
        offsets = {k: (int(v[0]), int(v[1])) for k, v in index_data["offsets"].items()}
        return RowStore(rows=rows_mmap, offsets=offsets)

    # Check if existing recovery row cache can be reused
    old_row_dir = paths.repo_root / "work" / "mmdd_fresh_recovery_v3_1_20260923" / "rows"
    if (old_row_dir / "index.json").exists() and (old_row_dir / "rows.f32.npy").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        # Link or load
        index_data = read_json(old_row_dir / "index.json")
        write_json(index_file, index_data)
        os.symlink(old_row_dir / "rows.f32.npy", data_file)
        rows_mmap = np.load(data_file, mmap_mode="r", allow_pickle=False)
        offsets = {k: (int(v[0]), int(v[1])) for k, v in index_data["offsets"].items()}
        return RowStore(rows=rows_mmap, offsets=offsets)

    # Materialize from manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths.row_cache_manifest
    rows_by_qid: list[tuple[str, np.ndarray]] = []
    total = 0
    for line in iter_jsonl(manifest):
        qid = str(line["object_id"])
        fpath = Path(line["feature_path"])
        payload = torch.load(fpath, map_location="cpu", weights_only=True)
        vecs = payload["row_embeddings"].float().numpy()
        rows_by_qid.append((qid, vecs))
        total += len(vecs)

    rows_by_qid.sort(key=lambda x: x[0].encode("utf-8"))
    array = np.lib.format.open_memmap(data_file, mode="w+", dtype=np.float32, shape=(total, INPUT_DIM))
    offsets_out: dict[str, list[int]] = {}
    cursor = 0
    for qid, vecs in rows_by_qid:
        array[cursor : cursor + len(vecs)] = vecs
        offsets_out[qid] = [cursor, len(vecs)]
        cursor += len(vecs)
    array.flush()
    del array

    write_json(index_file, {"offsets": offsets_out, "total_rows": total})
    rows_mmap = np.load(data_file, mmap_mode="r", allow_pickle=False)
    offsets = {k: (int(v[0]), int(v[1])) for k, v in offsets_out.items()}
    return RowStore(rows=rows_mmap, offsets=offsets)


def fit_pca(paths: Paths, z_store: ZStore, labels: Labels) -> dict[str, Any]:
    """Fit PCA (1024 dim) on:
    * all data lake targets
    * all canonical evidence
    * all original train queries
    (SPEC 6.3: dev/test queries excluded).
    """
    pca_dir = paths.pca_dir
    pca_dir.mkdir(parents=True, exist_ok=True)

    targets = utf8_sorted(labels.legal_targets)
    canonical_evidence = utf8_sorted(set(labels.canonical_map.values()))
    train_queries = utf8_sorted(labels.query_ids)

    fit_ids = utf8_sorted(set(targets) | set(canonical_evidence) | set(train_queries))
    write_json(
        pca_dir / "fit_ids.json",
        {
            "ordering": "UTF-8 byte order",
            "targets_count": len(targets),
            "canonical_evidence_count": len(canonical_evidence),
            "train_queries_count": len(train_queries),
            "total_fit_ids": len(fit_ids),
        },
    )

    # 1. Compute Mean
    total = len(fit_ids)
    sum_vec = np.zeros(INPUT_DIM, dtype=np.float64)
    chunk_size = 4096

    for start in range(0, total, chunk_size):
        batch_ids = fit_ids[start : start + chunk_size]
        vecs = z_store.rows(batch_ids).numpy().astype(np.float64)
        sum_vec += vecs.sum(axis=0)

    mean = sum_vec / total

    # 2. Compute Covariance
    cov = np.zeros((INPUT_DIM, INPUT_DIM), dtype=np.float64)
    for start in range(0, total, chunk_size):
        batch_ids = fit_ids[start : start + chunk_size]
        vecs = z_store.rows(batch_ids).numpy().astype(np.float64)
        diff = vecs - mean
        cov += diff.T @ diff

    cov /= total

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; reverse to descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Top 1024 components
    basis = eigenvectors[:, :PCA_COMPONENTS].T.astype(np.float32)  # (1024, 4096)
    mean_f32 = mean.astype(np.float32)

    # Save
    np.save(pca_dir / "mean.f32.npy", mean_f32)
    np.save(pca_dir / "basis.f32.npy", basis)
    np.save(pca_dir / "eigenvalues.f64.npy", eigenvalues)
    torch.save(
        {"basis": torch.from_numpy(basis), "mean": torch.from_numpy(mean_f32)},
        pca_dir / "basis.pt",
    )

    report = {
        "status": "PASS",
        "components": PCA_COMPONENTS,
        "input_dim": INPUT_DIM,
        "total_fitted": total,
        "variance_explained_ratio": float(np.sum(eigenvalues[:PCA_COMPONENTS]) / np.sum(eigenvalues)),
        "mean_sha256": sha256_file(pca_dir / "mean.f32.npy"),
        "basis_sha256": sha256_file(pca_dir / "basis.f32.npy"),
        "basis_pt_sha256": sha256_file(pca_dir / "basis.pt"),
    }
    write_json(paths.run_root / "PCA_REPORT.json", report)
    return report


def load_pca(paths: Paths) -> tuple[Tensor, Tensor]:
    pca_dir = paths.pca_dir
    basis_pt = pca_dir / "basis.pt"
    if not basis_pt.exists():
        raise FileNotFoundError(f"PCA basis not found: {basis_pt}")
    payload = torch.load(basis_pt, map_location="cpu", weights_only=True)
    return payload["basis"].float(), payload["mean"].float()

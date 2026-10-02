"""Frozen feature stores (z, content tokens, query rows), the ObjectBank, and PCA fitting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .config import Paths
from .content import EMBED_DIM, ContentStore
from .data import iter_jsonl, read_json, sha256_file, utf8_sorted, write_json
from .labels import Labels

INPUT_DIM = EMBED_DIM
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
    if raw.shape != (len(ids), INPUT_DIM) or raw.dtype != np.float32:
        raise ValueError(f"unexpected frozen z array: shape={raw.shape}, dtype={raw.dtype}")
    z_tensor = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True, order="C"))
    norms = torch.linalg.vector_norm(z_tensor, dim=1)
    if bool((norms == 0).any()):
        raise ValueError(f"zero frozen feature vector: {ids[int(torch.nonzero(norms == 0)[0])]}")
    z_tensor.div_(norms[:, None])

    index = {oid: i for i, oid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("duplicate object IDs in frozen z index")
    cache_rows = [
        row for row in iter_jsonl(paths.run_root / "CACHE_MANIFEST.jsonl")
        if row["path"] == "z/z.f32.npy"
    ]
    if len(cache_rows) != 1:
        raise RuntimeError("locked cache manifest has no unique z/z.f32.npy identity")
    identity = {
        "input_sha256": cache_rows[0]["sha256"],
        "preprocessing": "float32_l2_unit_rows_v1",
        "shape": list(z_tensor.shape),
    }
    z_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(
        paths.run_root / "Z_UNIT_IDENTITY.json",
        {**identity, "identity_sha256": z_sha,
         "input_norm_min": float(norms.min()), "input_norm_max": float(norms.max())},
    )
    return ZStore(ids=ids, types=types, z=z_tensor, index=index, sha256=z_sha)


class ObjectBank:
    """Retrieval ``z`` (device-resident) plus content tokens fetched per batch from the ContentStore."""

    def __init__(self, z_store: ZStore, content: ContentStore, device: Optional[str | torch.device] = None) -> None:
        self.z_store = z_store
        self.content = content
        self._device: Optional[torch.device] = None
        self._z_gpu: Optional[Tensor] = None
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

    def tokens_many(self, object_ids: Sequence[str]) -> list[Tensor]:
        """Content tokens of ``object_ids`` as float32 device tensors, gathered in one host buffer and one transfer."""
        if not object_ids:
            return []
        positions, lengths = self.content.locate(object_ids)
        cuda = self._device is not None and self._device.type == "cuda"
        # pin_memory=True draws from torch's cached host allocator, so repeated batches do not
        # pay cudaHostAlloc again; the float16 -> float32 expansion happens on the device.
        host = torch.empty((int(lengths.sum()), INPUT_DIM), dtype=torch.float16, pin_memory=cuda)
        self.content.copy_rows(positions, host.numpy())
        flat = host.to(self._device, non_blocking=cuda) if self._device is not None else host
        return list(torch.split(flat.float(), lengths.tolist()))


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

    # Materialize only from the locked pure row-feature manifest.
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = paths.row_cache_manifest
    required_query_ids = {
        str(row["table_id"])
        for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl")
    }
    rows_by_qid: list[tuple[str, np.ndarray]] = []
    seen_query_ids: set[str] = set()
    total = 0
    for line in iter_jsonl(manifest):
        qid = str(line["object_id"])
        if qid not in required_query_ids:
            continue
        if str(line.get("object_type")) != "table":
            raise ValueError(f"{qid}: query row-cache manifest entry is not a table")
        if qid in seen_query_ids:
            raise ValueError(f"{qid}: duplicate query row-cache manifest entry")
        fpath = paths.row_cache_manifest.parent / str(line["feature_path"])
        payload = torch.load(fpath, map_location="cpu", weights_only=True)
        if "row_embeddings" not in payload:
            raise KeyError(f"{qid}: table payload has no row_embeddings")
        vecs_t = payload["row_embeddings"].float()
        if vecs_t.ndim != 2 or vecs_t.shape[1] != INPUT_DIM or not len(vecs_t):
            raise ValueError(f"{qid}: invalid row embedding shape {tuple(vecs_t.shape)}")
        norms = torch.linalg.vector_norm(vecs_t, dim=1)
        if bool((norms == 0).any()):
            raise ValueError(f"{qid}: zero row embedding")
        vecs = (vecs_t / norms[:, None]).numpy()
        rows_by_qid.append((qid, vecs))
        seen_query_ids.add(qid)
        total += len(vecs)

    missing = utf8_sorted(required_query_ids - seen_query_ids)
    if missing:
        raise FileNotFoundError(f"row cache missing {len(missing)} query tables; first={missing[:5]}")

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

    write_json(
        index_file,
        {
            "offsets": offsets_out,
            "total_rows": total,
            "query_count": len(rows_by_qid),
            "preprocessing": "float32_l2_unit_rows_v1",
            "source_manifest_sha256": sha256_file(paths.row_cache_manifest),
            "rows_sha256": sha256_file(data_file),
        },
    )
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
            "ids": fit_ids,
            "z_unit_identity_sha256": z_store.sha256,
        },
    )

    total = len(fit_ids)
    sum_vec = np.zeros(INPUT_DIM, dtype=np.float64)
    chunk_size = 4096

    for start in range(0, total, chunk_size):
        batch_ids = fit_ids[start : start + chunk_size]
        vecs = z_store.rows(batch_ids).numpy().astype(np.float64)
        sum_vec += vecs.sum(axis=0)

    mean = sum_vec / total

    cov = np.zeros((INPUT_DIM, INPUT_DIM), dtype=np.float64)
    for start in range(0, total, chunk_size):
        batch_ids = fit_ids[start : start + chunk_size]
        vecs = z_store.rows(batch_ids).numpy().astype(np.float64)
        diff = vecs - mean
        cov += diff.T @ diff

    cov /= total

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis64 = eigenvectors[:, :PCA_COMPONENTS].T
    for row in basis64:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    basis = basis64.astype(np.float32)  # (1024, 4096)
    mean_f32 = mean.astype(np.float32)

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
        "fit_ids_sha256": sha256_file(pca_dir / "fit_ids.json"),
        "sign_rule": "largest_absolute_component_positive",
        "covariance_denominator": total,
    }
    write_json(paths.run_root / "PCA_REPORT.json", report)
    write_json(paths.pca_dir / "PCA_REPORT.json", report)
    return report


def load_pca(paths: Paths) -> tuple[Tensor, Tensor]:
    pca_dir = paths.pca_dir
    basis_pt = pca_dir / "basis.pt"
    if not basis_pt.exists():
        raise FileNotFoundError(f"PCA basis not found: {basis_pt}")
    payload = torch.load(basis_pt, map_location="cpu", weights_only=True)
    return payload["basis"].float(), payload["mean"].float()

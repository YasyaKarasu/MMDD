"""This run's Student PCA basis (SPEC 8.1).

Fit on the z of every distinct lake target and evidence object, without any
query object and without any relevance label.  float64 chunked mean and
centred covariance; deterministic CPU ``eigh``; the basis is applied as ``Wz``
without subtracting the mean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def fit_scope(labels, content_keys: dict[str, str] | None) -> list[str]:
    """Lake targets plus evidence objects, identical content counted once."""
    ids = list(labels.legal)
    seen: set[str] = set()
    canon: dict[str, str] = {}
    if content_keys:
        groups: dict[str, list[str]] = {}
        for oid, key in content_keys.items():
            groups.setdefault(key, []).append(oid)
        for members in groups.values():
            first = sorted(members, key=lambda x: x.encode("utf-8"))[0]
            for m in members:
                canon[m] = first
    for oid in content_keys or {}:
        c = canon.get(oid, oid)
        if c in seen:
            continue
        seen.add(c)
        ids.append(c)
    return sorted(set(ids), key=lambda x: x.encode("utf-8"))


def compute_pca(z_path: Path, z_ids: Sequence[str], fit_ids: Sequence[str], out_path: Path, *,
                components: int = 1024, chunk: int = 8192) -> dict:
    index = {oid: i for i, oid in enumerate(z_ids)}
    rows = np.array([index[i] for i in fit_ids], dtype=np.int64)
    z = np.load(Path(z_path), mmap_mode="r")
    n = len(rows)
    mean = np.zeros(z.shape[1], dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(z[rows[start : start + chunk]], dtype=np.float64)
        mean += block.sum(0)
    mean /= n
    cov = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(z[rows[start : start + chunk]], dtype=np.float64) - mean
        cov += block.T @ block
    cov /= n
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals, kind="stable")[:components]
    basis = eigvecs[:, order].T.copy()
    vals = eigvals[order]
    for i in range(len(basis)):
        vector = basis[i]
        j = int(np.argmax(np.abs(vector)))
        if vector[j] < 0:
            basis[i] = -vector
    weight = torch.from_numpy(basis).to(torch.float32)
    payload = {
        "basis": weight,
        "eigenvalues": torch.from_numpy(vals).to(torch.float32),
        "mean": torch.from_numpy(mean).to(torch.float32),
        "fit_objects": len(fit_ids),
        "components": components,
        "explained_variance_ratio": float(vals.sum() / eigvals.sum()),
        "applied_as": "Wz_without_centering",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return {
        "components": components,
        "fit_objects": len(fit_ids),
        "basis_shape": list(weight.shape),
        "orthonormal_max_deviation": float((weight @ weight.T - torch.eye(components)).abs().max()),
        "explained_variance_ratio": payload["explained_variance_ratio"],
    }

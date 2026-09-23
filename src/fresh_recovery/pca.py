from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Paths
from .io import iter_jsonl, read_jsonl_gz, sha256_file, sha256_json, write_json

INPUT_DIM = 4096
COMPONENTS = 1024
CHUNK_SIZE = 8192


def _save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _fit_ids(paths: Paths) -> tuple[list[str], list[str], list[str]]:
    targets = []
    for part in sorted((paths.dataset_root / "data_lake_tables").glob("part-*.jsonl")):
        targets.extend(str(row["table_id"]) for row in iter_jsonl(part))
    canonical_evidence = sorted(
        {str(row["canonical_id"]) for row in read_jsonl_gz(paths.work_dir / "CONTENT_ALIASES.jsonl.gz")},
        key=lambda value: value.encode("utf-8"),
    )
    targets = sorted(set(targets), key=lambda value: value.encode("utf-8"))
    if set(targets) & set(canonical_evidence):
        raise ValueError("target and canonical-evidence IDs overlap")
    fit_ids = sorted([*targets, *canonical_evidence], key=lambda value: value.encode("utf-8"))
    return targets, canonical_evidence, fit_ids


def _valid_npy(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    if not path.is_file():
        return False
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    return value.shape == shape and value.dtype == dtype and bool(np.isfinite(value).all())


def fit_pca(paths: Paths) -> dict[str, Any]:
    """Fit the protocol PCA from targets and canonical evidence only."""

    pure_source_path = paths.work_dir / "PURE_FEATURE_SOURCE.json"
    pure_source = json.loads(pure_source_path.read_text(encoding="utf-8"))
    if pure_source.get("status") != "PASS":
        raise RuntimeError("pure frozen feature source must pass before PCA")
    aliases_summary_path = paths.work_dir / "CONTENT_ALIASES.json"
    aliases_summary = json.loads(aliases_summary_path.read_text(encoding="utf-8"))
    z_path = Path(str(pure_source["z"])).resolve()
    z_index_path = Path(str(pure_source["z_index"])).resolve()
    z_index = json.loads(z_index_path.read_text(encoding="utf-8"))
    z_ids = [str(value) for value in z_index["ids"]]
    z_types = [str(value) for value in z_index["types"]]
    if len(z_ids) != len(set(z_ids)):
        raise ValueError("pure z index contains duplicate IDs")
    z = np.load(z_path, mmap_mode="r", allow_pickle=False)
    if z.shape != (len(z_ids), INPUT_DIM) or z.dtype != np.float32:
        raise ValueError(f"invalid pure z matrix: shape={z.shape}, dtype={z.dtype}")

    targets, canonical_evidence, fit_ids = _fit_ids(paths)
    if len(canonical_evidence) != int(aliases_summary["canonical_objects"]):
        raise ValueError("canonical evidence count differs from CONTENT_ALIASES summary")
    index = {object_id: row for row, object_id in enumerate(z_ids)}
    missing = [object_id for object_id in fit_ids if object_id not in index]
    if missing:
        raise KeyError(f"PCA fit IDs missing from pure z: {missing[:10]}")
    target_type_errors = [object_id for object_id in targets if z_types[index[object_id]] != "table"]
    evidence_type_errors = [
        object_id
        for object_id in canonical_evidence
        if z_types[index[object_id]] not in {"text", "image"}
    ]
    if target_type_errors or evidence_type_errors:
        raise ValueError(
            f"PCA fit type mismatch: targets={target_type_errors[:5]}, evidence={evidence_type_errors[:5]}"
        )
    rows = np.asarray([index[object_id] for object_id in fit_ids], dtype=np.int64)

    out_dir = paths.work_dir / "pca"
    partial_dir = out_dir / "covariance_partials"
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_ids_path = out_dir / "fit_ids.json"
    write_json(
        fit_ids_path,
        {
            "ordering": "UTF-8 byte order",
            "target_count": len(targets),
            "canonical_evidence_count": len(canonical_evidence),
            "fit_ids": fit_ids,
        },
    )
    run_contract = {
        "format_version": 1,
        "algorithm": "two_pass_float64_centered_population_covariance_then_CPU_eigh",
        "input_dim": INPUT_DIM,
        "components": COMPONENTS,
        "chunk_size": CHUNK_SIZE,
        "fit_objects": len(fit_ids),
        "fit_ids_sha256": sha256_file(fit_ids_path),
        "z_sha256": sha256_file(z_path),
        "z_index_sha256": sha256_file(z_index_path),
        "content_aliases_sha256": sha256_file(paths.work_dir / "CONTENT_ALIASES.jsonl.gz"),
        "queries_in_fit": 0,
        "relevance_labels_read": False,
        "covariance_divisor": len(fit_ids),
    }
    run_contract_path = out_dir / "PCA_RUN_CONTRACT.json"
    if run_contract_path.exists():
        existing = json.loads(run_contract_path.read_text(encoding="utf-8"))
        if existing != run_contract:
            raise RuntimeError("existing PCA partials belong to a different input contract")
    else:
        write_json(run_contract_path, run_contract)

    started = time.time()
    mean_path = out_dir / "mean.f64.npy"
    if _valid_npy(mean_path, (INPUT_DIM,), np.dtype(np.float64)):
        mean = np.asarray(np.load(mean_path, allow_pickle=False), dtype=np.float64)
        print(json.dumps({"event": "pca_mean_reused", "path": str(mean_path)}), flush=True)
    else:
        mean_sum = np.zeros(INPUT_DIM, dtype=np.float64)
        for chunk_index, start in enumerate(range(0, len(rows), CHUNK_SIZE)):
            block = np.asarray(z[rows[start : start + CHUNK_SIZE]], dtype=np.float64)
            mean_sum += block.sum(axis=0, dtype=np.float64)
            print(
                json.dumps(
                    {
                        "event": "pca_mean_chunk",
                        "chunk": chunk_index,
                        "objects_done": min(start + CHUNK_SIZE, len(rows)),
                        "objects_total": len(rows),
                    }
                ),
                flush=True,
            )
        mean = mean_sum / len(rows)
        _save_npy(mean_path, mean)

    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_paths = []
    for chunk_index, start in enumerate(range(0, len(rows), CHUNK_SIZE)):
        partial_path = partial_dir / f"part_{chunk_index:04d}.f64.npy"
        partial_paths.append(partial_path)
        if _valid_npy(partial_path, (INPUT_DIM, INPUT_DIM), np.dtype(np.float64)):
            print(json.dumps({"event": "pca_covariance_partial_reused", "chunk": chunk_index}), flush=True)
            continue
        block = np.asarray(z[rows[start : start + CHUNK_SIZE]], dtype=np.float64)
        block -= mean
        partial = block.T @ block
        _save_npy(partial_path, partial)
        print(
            json.dumps(
                {
                    "event": "pca_covariance_chunk",
                    "chunk": chunk_index,
                    "objects_done": min(start + CHUNK_SIZE, len(rows)),
                    "objects_total": len(rows),
                }
            ),
            flush=True,
        )

    covariance = np.zeros((INPUT_DIM, INPUT_DIM), dtype=np.float64)
    for partial_path in partial_paths:
        covariance += np.load(partial_path, mmap_mode="r", allow_pickle=False)
    covariance /= len(rows)
    covariance_path = out_dir / "covariance.f64.npy"
    _save_npy(covariance_path, covariance)
    print(json.dumps({"event": "pca_eigh_start", "dimension": INPUT_DIM}), flush=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis = eigenvectors[:, :COMPONENTS].T.copy()
    for row in basis:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    basis_f32 = basis.astype(np.float32)
    eigenvalues_path = out_dir / "eigenvalues.f64.npy"
    basis_path = out_dir / "P0.f32.npy"
    _save_npy(eigenvalues_path, eigenvalues)
    _save_npy(basis_path, basis_f32)

    payload_path = out_dir / "basis.pt"
    temporary = payload_path.with_name(f".{payload_path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "format_version": 1,
            "basis": torch.from_numpy(basis_f32.copy()),
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "eigenvalues": torch.from_numpy(eigenvalues[:COMPONENTS].astype(np.float32)),
            "fit_objects": len(fit_ids),
            "components": COMPONENTS,
            "applied_as": "Pz_without_centering_or_post_normalizing",
        },
        temporary,
    )
    temporary.replace(payload_path)

    gram64_error = float(np.max(np.abs(basis @ basis.T - np.eye(COMPONENTS))))
    gram32 = basis_f32 @ basis_f32.T
    gram32_error = float(np.max(np.abs(gram32 - np.eye(COMPONENTS, dtype=np.float32))))
    pivots = np.argmax(np.abs(basis), axis=1)
    sign_violations = int(sum(basis[row, pivot] < 0 for row, pivot in enumerate(pivots)))
    total_variance = float(eigenvalues.sum())
    report = {
        "status": "PASS",
        "scope": "all legal lake targets plus all canonical evidence; no queries or relevance labels",
        "fit_objects": len(fit_ids),
        "fit_targets": len(targets),
        "fit_canonical_evidence": len(canonical_evidence),
        "input_dim": INPUT_DIM,
        "components": COMPONENTS,
        "accumulator_dtype": "float64",
        "solver": "numpy.linalg.eigh on CPU",
        "covariance": "centered population covariance divided by N",
        "application": "u=Pz without subtracting saved mean and without post-projection normalization",
        "explained_variance_ratio": float(eigenvalues[:COMPONENTS].sum() / total_variance),
        "smallest_eigenvalue": float(eigenvalues[-1]),
        "orthonormal_max_deviation_float64": gram64_error,
        "orthonormal_max_deviation_float32": gram32_error,
        "sign_violations": sign_violations,
        "elapsed_seconds": time.time() - started,
        "artifacts": {
            "fit_ids": {"path": str(fit_ids_path), "sha256": sha256_file(fit_ids_path)},
            "mean": {"path": str(mean_path), "sha256": sha256_file(mean_path)},
            "covariance": {"path": str(covariance_path), "sha256": sha256_file(covariance_path)},
            "eigenvalues": {"path": str(eigenvalues_path), "sha256": sha256_file(eigenvalues_path)},
            "basis_npy": {"path": str(basis_path), "sha256": sha256_file(basis_path)},
            "basis_pt": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
            "pure_feature_source": {"path": str(pure_source_path), "sha256": sha256_file(pure_source_path)},
        },
        "run_contract_sha256": sha256_json(run_contract),
    }
    if (
        not np.isfinite(basis_f32).all()
        or total_variance <= 0
        or gram32_error > 1e-4
        or sign_violations
    ):
        report["status"] = "FAIL"
    write_json(out_dir / "PCA_REPORT.json", report)
    write_json(paths.work_dir / "PCA_REPORT.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("PCA numerical acceptance failed")
    return report

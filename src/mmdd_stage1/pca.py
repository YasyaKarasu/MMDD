"""PCA projection artifacts for near-raw low-dimensional Student initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from mmdd_progress import progress

PCA_FORMAT_VERSION = 1
PCA_SPECTRUM_FORMAT_VERSION = 1


def compute_pca_spectrum(
    embeddings: torch.Tensor,
    max_components: int,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top directions and the complete centered covariance spectrum."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            "embeddings must have shape [objects, dim] with at least two objects"
        )
    if not 0 < max_components <= min(embeddings.shape[0] - 1, embeddings.shape[1]):
        raise ValueError("max_components must not exceed the covariance rank bound")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")

    mean = embeddings.float().mean(dim=0)
    device_mean = mean.to(device)
    scatter = torch.zeros(
        embeddings.shape[1], embeddings.shape[1], device=device, dtype=torch.float32
    )
    with torch.no_grad():
        starts = range(0, embeddings.shape[0], batch_size)
        for start in progress(
            starts,
            total=len(starts),
            desc="Accumulate PCA covariance",
            unit="batch",
            leave=False,
        ):
            batch = embeddings[start : start + batch_size].to(
                device=device, dtype=torch.float32
            )
            centered = batch - device_mean
            scatter.addmm_(centered.T, centered)

        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        eigenvalues = eigenvalues.flip(0).clamp_min_(0)
        eigenvectors = eigenvectors.flip(1)
        total_variance = eigenvalues.sum()
        if total_variance <= 0:
            raise ValueError("embeddings must have non-zero variance")
        explained_variance_ratio = eigenvalues.cumsum(0) / total_variance
        projection = eigenvectors[:, :max_components].T.contiguous()

    result = (
        projection.cpu(),
        mean,
        eigenvalues.cpu(),
        explained_variance_ratio.cpu(),
    )
    del scatter, eigenvectors, device_mean
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def compute_pca_projection(
    embeddings: torch.Tensor,
    student_dim: int,
    *,
    device: torch.device,
    oversampling: int = 32,
    iterations: int = 2,
    seed: int = 13,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return top principal directions as a row-orthogonal projection."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError(
            "embeddings must have shape [objects, dim] with at least two objects"
        )
    if not 0 < student_dim <= min(embeddings.shape):
        raise ValueError("student_dim must not exceed the embedding matrix rank bound")
    if oversampling < 0 or iterations < 0:
        raise ValueError("oversampling and iterations must be non-negative")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    q = min(student_dim + oversampling, min(embeddings.shape))
    device_embeddings = embeddings.to(device=device, dtype=torch.float32)
    left, singular_values, components = torch.pca_lowrank(
        device_embeddings,
        q=q,
        center=True,
        niter=iterations,
    )
    orthonormal_components, _triangular = torch.linalg.qr(
        components[:, :student_dim], mode="reduced"
    )
    projection = orthonormal_components.T.contiguous().cpu()
    retained_variance = singular_values[:student_dim].square().sum().cpu()
    total_variance = embeddings.float().var(dim=0, correction=1).sum()
    if total_variance <= 0:
        raise ValueError("embeddings must have non-zero variance")
    explained_variance_ratio = float(
        retained_variance / ((embeddings.shape[0] - 1) * total_variance)
    )
    mean = embeddings.float().mean(dim=0)
    del left, singular_values, components, device_embeddings
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return projection, mean, explained_variance_ratio


def load_pca_projection(
    path: Path,
    *,
    input_dim: int,
    student_dim: int,
) -> torch.Tensor:
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != PCA_FORMAT_VERSION
    ):
        raise ValueError(f"{path}: unsupported PCA projection artifact")
    is_spectrum = payload.get("artifact_kind") == "stage1_pca_spectrum"
    if payload.get("input_dim") != input_dim or (
        is_spectrum
        and int(payload.get("max_components", 0)) < student_dim
    ) or (
        not is_spectrum and payload.get("student_dim") != student_dim
    ):
        raise ValueError(f"{path}: PCA projection dimensions do not match the Student")
    projection = payload.get("projection")
    if not isinstance(projection, torch.Tensor):
        raise ValueError(f"{path}: PCA artifact has no projection tensor")
    projection = projection[:student_dim].float() if is_spectrum else projection.float()
    if (
        projection.shape != (student_dim, input_dim)
        or not torch.isfinite(projection).all()
    ):
        raise ValueError(f"{path}: invalid PCA projection tensor")
    gram = projection @ projection.T
    if not torch.allclose(gram, torch.eye(student_dim), atol=1e-4, rtol=1e-4):
        raise ValueError(f"{path}: PCA projection rows are not orthonormal")
    return projection

"""Checkpoint helpers for the trainable RATA candidate-column head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .verifier import CandidateColumnScorer


def save_candidate_scorer(path: Path, scorer: CandidateColumnScorer, *, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "hidden_dim": scorer.hidden_dim,
            "input_dim": scorer.input_dim,
            "head_type": scorer.head_type,
            "reader_layout_version": (metadata or {}).get("reader_layout_version", "header_markers_v0"),
            "state_dict": scorer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


def load_candidate_scorer(
    path: Path,
    device: torch.device,
    *,
    expected_model_dir: Path | None = None,
    expected_reader_layout: str | None = None,
) -> CandidateColumnScorer:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError(f"{path}: unsupported Stage-2 checkpoint")
    if expected_model_dir is not None:
        checkpoint_model_dir = payload.get("metadata", {}).get("model_dir")
        if checkpoint_model_dir is None or Path(checkpoint_model_dir).resolve() != expected_model_dir.resolve():
            raise ValueError(
                f"{path}: candidate scorer was not trained with {expected_model_dir}; "
                "retrain it for the selected Stage-2 backbone"
            )
    layout = payload.get("reader_layout_version", "header_markers_v0")
    if expected_reader_layout is not None and expected_reader_layout != layout:
        raise ValueError(f"{path}: checkpoint/cache reader layout mismatch")
    scorer = CandidateColumnScorer(int(payload["hidden_dim"]), head_type=payload.get("head_type", "linear")).to(device)
    if payload.get("input_dim", scorer.input_dim) != scorer.input_dim:
        raise ValueError(f"{path}: inconsistent scorer input dimension")
    scorer.reader_layout_version = layout
    scorer.load_state_dict(payload["state_dict"])
    return scorer


def load_candidate_scorer_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") not in {1, 2}:
        raise ValueError(f"{path}: unsupported Stage-2 checkpoint")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: invalid Stage-2 checkpoint metadata")
    return metadata

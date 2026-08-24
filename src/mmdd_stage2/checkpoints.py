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
            "format_version": 1,
            "hidden_dim": scorer.weight.in_features // 2,
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
) -> CandidateColumnScorer:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        payload = torch.load(path, map_location=device)
    if payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Stage-2 checkpoint")
    if expected_model_dir is not None:
        checkpoint_model_dir = payload.get("metadata", {}).get("model_dir")
        if checkpoint_model_dir is None or Path(checkpoint_model_dir).resolve() != expected_model_dir.resolve():
            raise ValueError(
                f"{path}: candidate scorer was not trained with {expected_model_dir}; "
                "retrain it for the selected Stage-2 backbone"
            )
    scorer = CandidateColumnScorer(int(payload["hidden_dim"])).to(device)
    scorer.load_state_dict(payload["state_dict"])
    return scorer

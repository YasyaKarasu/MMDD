"""Versioned checkpoint I/O shared by training and retrieval entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import StudentJoinabilityModel, TeacherJoinabilityModel
from .objectives import PATH_AGGREGATIONS, PathAggregator


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Stage-1 checkpoint")
    return payload


def load_teacher(
    path: Path,
    device: torch.device,
    *,
    table_tokens_per_group: int | None = None,
) -> TeacherJoinabilityModel:
    payload = load_checkpoint(path)
    if payload.get("model_kind") != "teacher":
        raise ValueError(f"{path}: expected a Teacher checkpoint")
    model = TeacherJoinabilityModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    if table_tokens_per_group is not None:
        if table_tokens_per_group <= 0:
            raise ValueError("table_tokens_per_group must be positive")
        model.table_tokens_per_group = table_tokens_per_group
    return model.to(device)


def load_student(path: Path, device: torch.device) -> StudentJoinabilityModel:
    payload = load_checkpoint(path)
    if payload.get("model_kind") != "student":
        raise ValueError(f"{path}: expected a Student checkpoint")
    config = dict(payload["config"])
    if config.get("initialization") == "pca":
        input_dim = int(config["input_dim"])
        student_dim = int(config["student_dim"])
        placeholder = torch.zeros(student_dim, input_dim)
        placeholder[:, :student_dim] = torch.eye(student_dim)
        config["initialization_basis"] = placeholder
    model = StudentJoinabilityModel(**config)
    model.load_state_dict(payload["state_dict"])
    model.reset_projection_anchors()
    return model.to(device)


def load_path_aggregator(path: Path) -> PathAggregator:
    """Load the complete path aggregation configuration from a checkpoint."""

    config = load_checkpoint(path).get("path_aggregation", {})
    try:
        return PathAggregator(
            str(config.get("evidence_aggregation", "logsumexp")),
            int(config.get("evidence_top_k", 4)),
            temperature=float(config.get("evidence_temperature", 1.0)),
            power=float(config.get("evidence_power", 2.0)),
            path_combination=str(config.get("path_combination", "sum")),
            threshold=float(config.get("evidence_threshold", 0.0)),
            target_temperature=float(
                config.get("evidence_target_temperature", 1.0)
            ),
            row_support_model=config.get("row_support_model"),
            row_support_model_sha256=config.get("row_support_model_sha256"),
            row_support_top_l=int(config.get("row_support_top_l", 20)),
            evidence_content_keys=config.get("evidence_content_keys"),
            evidence_content_keys_sha256=config.get(
                "evidence_content_keys_sha256"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid path aggregation configuration") from exc

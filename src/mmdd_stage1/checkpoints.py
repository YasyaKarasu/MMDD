"""Versioned checkpoint I/O shared by training and retrieval entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import StudentJoinabilityModel, TeacherJoinabilityModel


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"{path}: unsupported Stage-1 checkpoint")
    return payload


def load_teacher(path: Path, device: torch.device) -> TeacherJoinabilityModel:
    payload = load_checkpoint(path)
    if payload.get("model_kind") != "teacher":
        raise ValueError(f"{path}: expected a Teacher checkpoint")
    model = TeacherJoinabilityModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
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
    return model.to(device)


def load_path_aggregation(path: Path) -> tuple[str, int]:
    config = load_checkpoint(path).get("path_aggregation", {})
    aggregation = str(config.get("evidence_aggregation", "logsumexp"))
    top_k = int(config.get("evidence_top_k", 4))
    if aggregation not in {"logsumexp", "topk_mean", "topk_sum"} or top_k <= 0:
        raise ValueError(f"{path}: invalid path aggregation configuration")
    return aggregation, top_k

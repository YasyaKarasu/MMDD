"""Versioned checkpoint I/O shared by training and retrieval entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .models import StudentJoinabilityModel, TeacherJoinabilityModel


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch.
        payload = torch.load(path, map_location="cpu")
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
    model = StudentJoinabilityModel(**payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device)

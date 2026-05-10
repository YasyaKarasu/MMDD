"""Helpers for clearing Stage-1 training outputs before a forced retrain."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from stage1_io import load_json, write_json

TRAINING_FILE_PATTERNS = [
    "teacher_scores.jsonl",
    "train_pairs.jsonl",
    "train_pairs_round_*.jsonl",
    "train_pairs_final_round_*.jsonl",
    "student_train_pairs.jsonl",
    "student_train_groups.jsonl",
    "hitl_selected_round_*.jsonl",
    "human_labels_template_round_*.jsonl",
    "human_labels_filled_round_*.jsonl",
    "hitl_round_*_annotation_status.json",
]

TRAINING_DIR_PATTERNS = [
    "teacher",
    "teacher_round_*",
    "teacher_final",
    "student",
]

TRAINING_MANIFEST_KEYS = {
    "teacher_training_data",
    "teacher",
    "student",
    "stage1_train_models",
    "hitl_training_loop",
}


def _remove_path(path: Path, removed: list[str]) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        removed.append(str(path))
    elif path.exists():
        path.unlink()
        removed.append(str(path))


def clear_training_outputs(stage1_dir: Path, student_dir: Path | None = None, reset_human_labels: bool = False) -> list[str]:
    """Remove generated teacher/student/HITL training outputs while keeping prepared data."""
    stage1_dir = stage1_dir.resolve()
    removed: list[str] = []

    for pattern in TRAINING_FILE_PATTERNS:
        for path in stage1_dir.glob(pattern):
            _remove_path(path, removed)
    for pattern in TRAINING_DIR_PATTERNS:
        for path in stage1_dir.glob(pattern):
            _remove_path(path, removed)

    if student_dir is not None:
        student_path = student_dir.resolve()
        if student_path != stage1_dir / "student" and stage1_dir in student_path.parents:
            _remove_path(student_path, removed)

    if reset_human_labels:
        _remove_path(stage1_dir / "human_labeled_paths.jsonl", removed)

    _clear_training_manifest_sections(stage1_dir, reset_human_labels)
    return removed


def _clear_training_manifest_sections(stage1_dir: Path, reset_human_labels: bool) -> None:
    manifest_path = stage1_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest: dict[str, Any] = load_json(manifest_path)
    for key in list(manifest):
        if key in TRAINING_MANIFEST_KEYS or key.startswith("hitl_round_"):
            manifest.pop(key, None)
    if reset_human_labels:
        manifest.pop("human_labels", None)
    write_json(manifest_path, manifest)

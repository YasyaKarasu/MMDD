"""Shared path contract for the WDC 200K builder and dynamic runner."""

from __future__ import annotations

from pathlib import Path


DEFAULT_WORK_DIR_NAME = "work_wdc_200k"
DEFAULT_MIN_FREE_DISK_BYTES = 1_000_000_000


def resolve_work_dir(
    output_dir: Path,
    work_dir: str | Path | None = None,
) -> Path:
    output = Path(output_dir).resolve()
    if work_dir:
        return Path(work_dir).resolve()
    return (output.parent / DEFAULT_WORK_DIR_NAME).resolve()


def required_runtime_dir(
    output_dir: Path,
    work_dir: str | Path | None = None,
) -> Path:
    return (resolve_work_dir(output_dir, work_dir) / "runtime").resolve()


__all__ = [
    "DEFAULT_MIN_FREE_DISK_BYTES",
    "DEFAULT_WORK_DIR_NAME",
    "required_runtime_dir",
    "resolve_work_dir",
]

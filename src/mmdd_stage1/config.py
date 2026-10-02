"""Protocol loading and path resolution.

Everything machine-specific lives in the protocol file: the ``paths`` block names the
dataset, backbone, feature caches and run root, and ``hardware`` names the GPU the run is
pinned to. Nothing in this package hardcodes a dataset, a cache directory or a GPU UUID.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from . import EXPERIMENT_ID, VERSION

REQUIRED_PATH_KEYS = ("dataset_root", "backbone_dir", "pure_cache_dir", "row_cache_manifest", "run_root")
STAGE_ORDER = [
    "TA", "TB_CQET", "TB_LSE", "TB_QT", "NATIVE_C1_SUP",
    "QT_C1_SUP", "NATIVE_C2_SUP", "NATIVE_C2_KD", "QT_C2_SUP",
]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    dataset_root: Path
    backbone_dir: Path
    pure_cache_dir: Path
    row_cache_manifest: Path
    protocol_path: Path
    run_root: Path
    # Optional: only the preflight lock / feature-provenance probe need these.
    upstream_cache_dir: Path | None = None
    upstream_data_dir: Path | None = None
    package_dir: Path | None = None
    # GPU the run is pinned to; None means "not pinned" (CPU tests, offline analysis).
    gpu_uuid: str | None = None
    gpu_physical_index: int | None = None

    @property
    def labels_dir(self) -> Path:
        return self.run_root / "labels"

    @property
    def pca_dir(self) -> Path:
        return self.run_root / "pca"

    def seed_dir(self, seed: int) -> Path:
        return self.run_root / f"seed{seed}"


def load_protocol(protocol_path: Path) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(data)
    return data


def validate_protocol(p: Mapping[str, Any]) -> None:
    if p.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Unexpected experiment_id: {p.get('experiment_id')}")
    if p.get("version") != VERSION:
        raise ValueError(f"Unexpected version: {p.get('version')} (this package is {VERSION})")
    hw = p.get("hardware", {})
    if not isinstance(hw.get("physical_index"), int) or hw["physical_index"] < 0:
        raise ValueError("hardware.physical_index must be a non-negative integer")
    if not str(hw.get("uuid", "")).startswith("GPU-"):
        raise ValueError("hardware.uuid must be the full GPU UUID (GPU-...)")
    if not hw.get("model"):
        raise ValueError("hardware.model must name the GPU model")
    if hw.get("gpu_processes") != 1 or hw.get("ddp") is not False:
        raise ValueError("the pipeline runs as one non-DDP GPU process")
    if hw.get("max_cpu_workers") != 4 or hw.get("max_prefetch_units") != 16:
        raise ValueError("CPU worker/prefetch limits changed")
    seeds = p.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) for s in seeds):
        raise ValueError("seeds must be a non-empty list of integers")
    if p.get("stage_order") != STAGE_ORDER or p.get("stages_per_seed") != 9:
        raise ValueError("nine-stage order changed")
    if p.get("max_registered_stages") != 9 * len(seeds):
        raise ValueError("max_registered_stages must be 9 * len(seeds)")
    if p.get("historical_training_dependencies") is not False:
        raise ValueError("historical training dependencies are forbidden")
    if p.get("fallback_on_research_failure") is not False:
        raise ValueError("research-failure fallback is forbidden")
    retrieval = p.get("retrieval", {})
    locked_budgets = {
        "direct_k": 100,
        "first_text_k": 20,
        "first_image_k": 20,
        "second_k": 50,
        "teacher_budget": 150,
        "prepath_top_l": 16,
        "retained_max": 4,
        "rrf_constant": 60,
    }
    for key, expected in locked_budgets.items():
        if retrieval.get(key) != expected:
            raise ValueError(f"retrieval.{key} must be {expected}")
    if retrieval.get("formal_hops") != "all_ANN":
        raise ValueError("all formal hops must use ANN")
    numerics = p.get("numerics", {})
    if numerics.get("task_dtype") != "float32" or numerics.get("AMP") is not False:
        raise ValueError("task numerics must remain FP32 without AMP")
    student = p.get("student", {})
    for key in ("P_lr", "R_lr", "logit_scale", "kd_weight", "temperature", "random_negatives", "anchor_weight"):
        if not isinstance(student.get(key), (int, float)):
            raise ValueError(f"student.{key} must be a number")
    if student["logit_scale"] <= 0 or student["temperature"] <= 0:
        raise ValueError("student.logit_scale and student.temperature must be positive")
    if student.get("lr_schedule") not in ("constant", "cosine"):
        raise ValueError("student.lr_schedule must be constant or cosine")
    if student.get("kd_normalization") not in ("temperature", "zscore"):
        raise ValueError("student.kd_normalization must be temperature or zscore")
    if not isinstance(student.get("teacher_scored_negatives"), bool):
        raise ValueError("student.teacher_scored_negatives must be a boolean")
    for key in ("kd_top_k", "evidence_random_negatives"):
        if not isinstance(student.get(key), int) or student[key] < 0:
            raise ValueError(f"student.{key} must be a non-negative integer")
    if student["evidence_random_negatives"] > student["random_negatives"]:
        raise ValueError("student.evidence_random_negatives draws from the random negatives and cannot exceed them")
    configured = p.get("paths")
    if not isinstance(configured, dict) or any(k not in configured for k in REQUIRED_PATH_KEYS):
        missing = sorted(set(REQUIRED_PATH_KEYS) - set(configured or {}))
        raise ValueError(f"protocol paths block is missing {missing}")


def _resolve(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return (path if path.is_absolute() else repo_root / path).resolve()


def resolve_default_paths(protocol_path: Path, run_root: Path) -> Paths:
    """Bind the protocol's ``paths`` and ``hardware`` blocks; relative paths are repo-relative."""
    repo_root = Path(__file__).resolve().parents[2]
    protocol = load_protocol(protocol_path)
    configured = protocol["paths"]
    expected_run_root = _resolve(repo_root, configured["run_root"])
    requested_run_root = Path(run_root).resolve()
    if requested_run_root != expected_run_root:
        raise ValueError(f"run root is fixed by protocol: {expected_run_root}")
    hw = protocol["hardware"]
    return Paths(
        repo_root=repo_root,
        dataset_root=_resolve(repo_root, configured["dataset_root"]),
        backbone_dir=_resolve(repo_root, configured["backbone_dir"]),
        pure_cache_dir=_resolve(repo_root, configured["pure_cache_dir"]),
        row_cache_manifest=_resolve(repo_root, configured["row_cache_manifest"]),
        protocol_path=Path(protocol_path).resolve(),
        run_root=requested_run_root,
        upstream_cache_dir=_resolve(repo_root, configured.get("upstream_cache_dir")),
        upstream_data_dir=_resolve(repo_root, configured.get("upstream_data_dir")),
        package_dir=_resolve(repo_root, configured.get("package_dir")),
        gpu_uuid=str(hw["uuid"]),
        gpu_physical_index=int(hw["physical_index"]),
    )

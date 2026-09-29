"""Protocol loading, configuration dataclasses, and path resolution for CLEAN-QET v4.0."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from . import EXPERIMENT_ID, VERSION


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    dataset_root: Path
    backbone_dir: Path
    pure_cache_dir: Path
    row_cache_manifest: Path
    protocol_path: Path
    run_root: Path

    @property
    def labels_dir(self) -> Path:
        return self.run_root / "labels"

    @property
    def pca_dir(self) -> Path:
        return self.run_root / "pca"

    @property
    def raw_dir(self) -> Path:
        return self.run_root / "raw"

    @property
    def lists_dir(self) -> Path:
        return self.run_root / "lists"

    @property
    def source_snapshots_dir(self) -> Path:
        return self.run_root / "source_snapshots"

    def seed_dir(self, seed: int) -> Path:
        return self.run_root / f"seed{seed}"


def load_protocol(protocol_path: Path) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"protocol file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_protocol(data)
    return data


def validate_protocol(p: dict[str, Any]) -> None:
    if p.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"Unexpected experiment_id: {p.get('experiment_id')}")
    if p.get("version") != VERSION:
        raise ValueError(f"Unexpected version: {p.get('version')}")
    hw = p.get("hardware", {})
    if hw.get("physical_gpu_index") != 0 or hw.get("visible_gpu_count") != 1:
        raise ValueError("Hardware must be physical GPU 0 only, visible_gpu_count=1")
    if p.get("seeds") != [13, 29]:
        raise ValueError("Seeds must be [13, 29]")
    if p.get("maximum_formal_training_stages") != 22:
        raise ValueError("maximum_formal_training_stages must be 22")
    teacher = p.get("teacher", {})
    if teacher.get("gap_loss") is not False:
        raise ValueError("gap_loss must be False in v4.0")
    if teacher.get("current_f0_stop_gradient_reference") is not False:
        raise ValueError("current_f0_stop_gradient_reference must be False in v4.0")


def resolve_default_paths(protocol_path: Path, run_root: Path) -> Paths:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
    backbone_dir = repo_root / "hf_models" / "Qwen3-VL-Embedding-8B"
    pure_cache_dir = repo_root / "work" / "mmdd_stage1_fresh_path_v2_1_20260920" / "features"
    row_cache_manifest = repo_root / "work" / "stage1_optimization_r10_20260907" / "features_qwen3_vl_embedding_8b" / "manifest.jsonl"

    p = Paths(
        repo_root=repo_root,
        dataset_root=dataset_root,
        backbone_dir=backbone_dir,
        pure_cache_dir=pure_cache_dir,
        row_cache_manifest=row_cache_manifest,
        protocol_path=Path(protocol_path).resolve(),
        run_root=Path(run_root).resolve(),
    )
    return p

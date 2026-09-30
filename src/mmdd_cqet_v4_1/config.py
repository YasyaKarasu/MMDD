"""Protocol loading and fixed path resolution for the V4.1 run."""
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
    if hw.get("physical_index") != 0:
        raise ValueError("hardware.physical_index must be 0")
    if hw.get("uuid") != "GPU-3d43b1bc-b727-456f-2b9f-e3c3b69eb725":
        raise ValueError("unexpected physical GPU0 UUID")
    if hw.get("gpu_processes") != 1 or hw.get("ddp") is not False:
        raise ValueError("V4.1 requires one non-DDP GPU process")
    if hw.get("max_cpu_workers") != 4 or hw.get("max_prefetch_units") != 16:
        raise ValueError("V4.1 CPU worker/prefetch limits changed")
    if p.get("seeds") not in ([13, 29], [13]):
        raise ValueError("Seeds must be [13, 29] or [13]")
    expected_order = [
        "TA", "TB_CQET", "TB_LSE", "TB_QT", "NATIVE_C1_SUP",
        "QT_C1_SUP", "NATIVE_C2_SUP", "NATIVE_C2_KD", "QT_C2_SUP",
    ]
    if p.get("stage_order") != expected_order or p.get("stages_per_seed") != 9:
        raise ValueError("V4.1 nine-stage order changed")
    if p.get("max_registered_stages") not in (18, 9):
        raise ValueError("max_registered_stages must be 18 or 9")
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


def resolve_default_paths(protocol_path: Path, run_root: Path) -> Paths:
    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = repo_root / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
    backbone_dir = repo_root / "hf_models" / "Qwen3-VL-Embedding-8B"
    pure_cache_dir = repo_root / "work" / "mmdd_stage1_fresh_path_v2_1_20260920" / "features"
    row_cache_manifest = repo_root / "work" / "stage1_optimization_r10_20260907" / "features_qwen3_vl_embedding_8b" / "manifest.jsonl"

    expected_run_root = (repo_root / "work" / "mmdd_stage1_v4_1_correctness_locked").resolve()
    requested_run_root = Path(run_root).resolve()
    if requested_run_root != expected_run_root:
        raise ValueError(f"run root is fixed by protocol: {expected_run_root}")
    p = Paths(
        repo_root=repo_root,
        dataset_root=dataset_root,
        backbone_dir=backbone_dir,
        pure_cache_dir=pure_cache_dir,
        row_cache_manifest=row_cache_manifest,
        protocol_path=Path(protocol_path).resolve(),
        run_root=requested_run_root,
    )
    return p

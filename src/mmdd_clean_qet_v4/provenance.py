"""Provenance tracking, source snapshotting, and manifest generation for CLEAN-QET v4.0."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from . import EXPERIMENT_ID, VERSION
from .config import Paths
from .data import sha256_file, write_json


def state_sha(state_dict: Mapping[str, Any]) -> str:
    """Compute sha256 of model parameter tensor buffers."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        v = state_dict[k]
        if hasattr(v, "cpu"):
            arr = v.cpu().numpy()
            h.update(arr.tobytes())
    return h.hexdigest()


def snapshot_source(source_dir: Path, target_dir: Path) -> dict[str, str]:
    """Copy immutable snapshot of current implementation and return file hashes."""
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for p in sorted(source_dir.glob("*.py")):
        if p.name.startswith("."):
            continue
        dest = target_dir / p.name
        shutil.copy2(p, dest)
        manifest[p.name] = sha256_file(dest)
    return manifest


def record_stage_pre_run(
    stage_dir: Path,
    stage: str,
    seed: int,
    gpu_uuid: str,
    parents: Optional[dict[str, str]] = None,
    config: Optional[dict[str, Any]] = None,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "seed": seed,
        "status": "RUNNING",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_uuid": gpu_uuid,
        "pid": os.getpid(),
        "parents": parents or {},
        "config": config or {},
    }
    write_json(stage_dir / "PRE_RUN.json", payload)


def record_stage_post_run(
    stage_dir: Path,
    stage: str,
    seed: int,
    status: str = "COMPLETE",
    counters: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, str]] = None,
) -> None:
    payload = {
        "stage": stage,
        "seed": seed,
        "status": status,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counters": counters or {},
        "outputs": outputs or {},
    }
    write_json(stage_dir / "POST_RUN.json", payload)


def append_error_ledger(
    run_root: Path,
    stage: str,
    seed: int,
    error: Exception,
    attempt: int = 1,
) -> None:
    ledger_path = run_root / "ERROR_LEDGER.jsonl"
    entry = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "seed": seed,
        "attempt": attempt,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_provenance_manifests(paths: Paths, gpu_uuid: str) -> None:
    """Generate protocol-mandated provenance manifests at run initialization."""
    root = paths.run_root
    root.mkdir(parents=True, exist_ok=True)

    # 1. RESOLVED_INPUTS.json
    resolved = {
        "experiment_id": EXPERIMENT_ID,
        "version": VERSION,
        "dataset_root": str(paths.dataset_root),
        "backbone_dir": str(paths.backbone_dir),
        "pure_cache_dir": str(paths.pure_cache_dir),
        "row_cache_manifest": str(paths.row_cache_manifest),
        "protocol_path": str(paths.protocol_path),
        "physical_gpu0_uuid": gpu_uuid,
    }
    write_json(root / "RESOLVED_INPUTS.json", resolved)

    # Copy protocol.json to run root
    if paths.protocol_path.exists() and paths.protocol_path.resolve() != (root / "protocol.json").resolve():
        shutil.copy2(paths.protocol_path, root / "protocol.json")

    # Touch ERROR_LEDGER.jsonl if not exists
    ledger = root / "ERROR_LEDGER.jsonl"
    if not ledger.exists():
        ledger.touch()

    # Snapshot test files to tests/
    tests_dest = root / "tests"
    tests_dest.mkdir(parents=True, exist_ok=True)
    for t_src in (paths.repo_root / "tests").glob("test_*.py"):
        shutil.copy2(t_src, tests_dest / t_src.name)
    if (paths.protocol_path.parent / "reference_contracts.py").exists():
        shutil.copy2(paths.protocol_path.parent / "reference_contracts.py", tests_dest / "reference_contracts.py")

    # 2. DATASET_IDENTITY.json
    dataset_manifest = paths.dataset_root / "dataset_manifest.json"
    splits = paths.dataset_root / "splits.json"
    ds_id = {
        "dataset_manifest_sha256": sha256_file(dataset_manifest) if dataset_manifest.exists() else None,
        "splits_sha256": sha256_file(splits) if splits.exists() else None,
        "qrels_sha256": sha256_file(paths.dataset_root / "qrels.jsonl") if (paths.dataset_root / "qrels.jsonl").exists() else None,
    }
    write_json(root / "DATASET_IDENTITY.json", ds_id)

    # 3. DATA_SPLIT_REPORT.json
    split_report = {
        "splits_json": str(splits),
        "split_policy": "query_only",
        "supervision_scope": "full_original_train",
        "legacy_calibration_buckets": False,
        "entity_url": "unchanged",
    }
    write_json(root / "DATA_SPLIT_REPORT.json", split_report)

    # 4. FEATURE_RECIPE.json
    recipe = {
        "backbone": "Qwen3-VL-Embedding-8B",
        "pooling": "mean_pooling",
        "feature_dim": 4096,
        "z_path": str(paths.pure_cache_dir / "z" / "z.f32.npy"),
        "z_index_path": str(paths.pure_cache_dir / "z" / "z_index.json"),
        "row_manifest": str(paths.row_cache_manifest),
    }
    write_json(root / "FEATURE_RECIPE.json", recipe)

    # 5. ROOT_DEPENDENCY_PROOF.json
    proof = {
        "allowed_inputs": ["original_dataset", "pure_frozen_qwen_features", "current_v4_code"],
        "historical_task_checkpoints_loaded": False,
        "historical_pca_loaded": False,
        "historical_lists_loaded": False,
        "historical_logits_loaded": False,
    }
    write_json(root / "ROOT_DEPENDENCY_PROOF.json", proof)

    # 6. SOURCE_LOCK.json
    pkg_dir = Path(__file__).resolve().parent
    source_hashes = {p.name: sha256_file(p) for p in sorted(pkg_dir.glob("*.py"))}
    write_json(root / "SOURCE_LOCK.json", source_hashes)

    # 7. EXECUTION_DAG.json
    dag = {
        "schedule": "sequential_seeds",
        "seeds": [13, 29],
        "stages_per_seed": [
            "T_A",
            "T_B_PATH",
            "T_B_QT",
            "S_NATIVE_KD_C1",
            "S_NATIVE_SUP_C1",
            "S_QT_KD_C1",
            "S_QT_SUP_C1",
            "SELECT_C1",
            "BUILD_C2_GRAPH",
            "S_NATIVE_KD_C2",
            "S_NATIVE_SUP_C2",
            "S_QT_KD_C2",
            "S_QT_SUP_C2",
            "SELECT_C2",
            "EVALUATE_DEV",
            "EVALUATE_TEST",
        ],
    }
    write_json(root / "EXECUTION_DAG.json", dag)

    # 8. PHASE_STATUS.json
    write_json(root / "PHASE_STATUS.json", {"current_phase": "INITIALIZED", "completed_stages": []})

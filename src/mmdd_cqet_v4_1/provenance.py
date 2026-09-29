"""Immutable stage receipts and source/output identity for V4.1."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from . import EXPERIMENT_ID, VERSION
from .config import Paths
from .data import iter_jsonl, sha256_file, write_json
from .train import state_sha

SOURCE_AMENDMENTS = "SOURCE_AMENDMENTS.jsonl"


def json_sha(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_files(paths: Paths) -> list[Path]:
    files = sorted((paths.repo_root / "src" / "mmdd_cqet_v4_1").glob("*.py"))
    files.extend([
        paths.repo_root / "scripts" / "run_v4_1_gpu0.sh",
        paths.repo_root / "tests" / "test_mmdd_cqet_v4_1.py",
        paths.repo_root / "tests" / "conftest.py",
        paths.repo_root / "src" / "cache_stage1_features.py",
        paths.repo_root / "hf_models" / "Qwen3-VL-Embedding-8B" / "scripts" / "qwen3_vl_embedding.py",
    ])
    files.extend(sorted((paths.repo_root / "src" / "fresh_path").glob("*.py")))
    package_root = paths.repo_root / "audit" / "MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE"
    files.extend(sorted((package_root / "tools").glob("*.py")))
    files.extend(sorted((package_root / "tests").glob("*.py")))
    files.append(package_root / "next_round" / "reference_contracts.py")
    return sorted(set(files), key=lambda path: str(path).encode("utf-8"))


def source_manifest(paths: Paths) -> list[dict[str, object]]:
    files = source_files(paths)
    return [
        {
            "path": str(path.relative_to(paths.repo_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "executed_source",
        }
        for path in files
    ]


def source_identity(paths: Paths) -> str:
    return json_sha(source_manifest(paths))


def record_source_amendment(
    paths: Paths, *, amendment_id: str, carried_stages: Sequence[str], reason: str,
) -> dict[str, Any]:
    """Declare that completed ``carried_stages`` stay valid under the current source.

    The previous source is read from SOURCE_TREE_MANIFEST.jsonl, so this must run
    after the code edit and before ``prepare`` rewrites that manifest.
    """
    previous = list(iter_jsonl(paths.run_root / "SOURCE_TREE_MANIFEST.jsonl"))
    current = source_manifest(paths)
    old = {row["path"]: row["sha256"] for row in previous}
    new = {row["path"]: row["sha256"] for row in current}
    changed = {
        path: {"from_sha256": old.get(path), "to_sha256": new.get(path)}
        for path in sorted(old.keys() | new.keys())
        if old.get(path) != new.get(path)
    }
    if not changed:
        raise RuntimeError(
            "SOURCE_TREE_MANIFEST.jsonl already describes the current source; "
            "record the amendment before prepare"
        )
    row = {
        "schema_version": VERSION,
        "amendment_id": amendment_id,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "from_source_identity_sha256": json_sha(previous),
        "to_source_identity_sha256": json_sha(current),
        "changed_paths": changed,
        "carried_stages": list(carried_stages),
        "reason": reason,
    }
    row["receipt_sha256"] = json_sha(row)
    with (paths.run_root / SOURCE_AMENDMENTS).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def carried_source_identities(paths: Paths, stage: str) -> set[str]:
    """Current source plus earlier ones from which every later amendment carries ``stage``."""
    accepted = {source_identity(paths)}
    ledger = paths.run_root / SOURCE_AMENDMENTS
    for row in reversed(list(iter_jsonl(ledger)) if ledger.exists() else []):
        if row["to_source_identity_sha256"] in accepted and stage in row["carried_stages"]:
            accepted.add(row["from_source_identity_sha256"])
    return accepted


def assert_declared_project_imports(paths: Paths) -> None:
    declared = {path.resolve() for path in source_files(paths)}
    undeclared = []
    for module in tuple(sys.modules.values()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        raw_path = Path(raw)
        # Extension namespaces such as torch.ops use relative placeholder
        # names (for example, ``_ops.py``) that are not importable files.
        if not raw_path.is_absolute():
            continue
        path = raw_path.resolve()
        if paths.repo_root in path.parents and path.suffix == ".py" and path not in declared:
            undeclared.append(str(path.relative_to(paths.repo_root)))
    if undeclared:
        raise RuntimeError(f"undeclared project-local execution imports: {sorted(set(undeclared))}")


def _next_attempt(stage_dir: Path) -> str:
    existing = list(stage_dir.glob("PRE_RUN.attempt_*.json"))
    return f"attempt_{len(existing) + 1:03d}"


def record_stage_pre_run(
    stage_dir: Path,
    stage: str,
    seed: int,
    gpu_uuid: str,
    *,
    physical_index: int = 0,
    paths: Optional[Paths] = None,
    parents: Optional[dict[str, str]] = None,
    config: Optional[dict[str, Any]] = None,
    inputs: Optional[dict[str, Any]] = None,
    lists: Optional[dict[str, str]] = None,
    command: Optional[Sequence[str]] = None,
) -> str:
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt = _next_attempt(stage_dir)
    source_sha = source_identity(paths) if paths is not None else None
    payload = {
        "schema_version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt,
        "stage": stage,
        "seed": seed,
        "status": "RUNNING",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": {
            "physical_index": physical_index,
            "uuid": gpu_uuid,
            "mapped_device": "cuda:0",
            "visible_device_count": torch.cuda.device_count(),
        },
        "pid": os.getpid(),
        "parents": parents or {},
        "config": config or {},
        "inputs": inputs or {},
        "training_lists": lists or {},
        "source_identity_sha256": source_sha,
        "protocol_sha256": sha256_file(paths.protocol_path) if paths is not None else None,
        "feature_identity_sha256": (
            json.loads((paths.run_root / "CACHE_IDENTITY.json").read_text())["identity_sha256"]
            if paths is not None else None
        ),
        "command": list(command or sys.argv),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    payload["receipt_sha256"] = json_sha(payload)
    path = stage_dir / f"PRE_RUN.{attempt}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, payload)
    return attempt


def record_stage_post_run(
    stage_dir: Path,
    stage: str,
    seed: int,
    *,
    attempt_id: str,
    status: str,
    counters: Optional[dict[str, Any]] = None,
    outputs: Optional[dict[str, str]] = None,
    timing: Optional[dict[str, Any]] = None,
    impact: Optional[str] = None,
) -> None:
    pre = stage_dir / f"PRE_RUN.{attempt_id}.json"
    if not pre.exists():
        raise FileNotFoundError(f"missing PRE receipt: {pre}")
    verified_outputs = {}
    for name, value in (outputs or {}).items():
        path = Path(value)
        if path.is_file():
            verified_outputs[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            verified_outputs[name] = value
    payload = {
        "schema_version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt_id,
        "stage": stage,
        "seed": seed,
        "status": status,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counters": counters or {},
        "outputs": verified_outputs,
        "timing": timing or {},
        "impact": impact,
        "pre_run_sha256": sha256_file(pre),
    }
    payload["receipt_sha256"] = json_sha(payload)
    path = stage_dir / f"POST_RUN.{attempt_id}.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path, payload)


def append_error_ledger(
    run_root: Path,
    stage: str,
    seed: int,
    error: Exception,
    *,
    attempt_id: str,
    impact: str,
    status: str = "FAILED_RETRYABLE",
) -> None:
    entry = {
        "schema_version": VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attempt_id": attempt_id,
        "stage": stage,
        "seed": seed,
        "classification": "runtime_failure",
        "error_type": error.__class__.__name__,
        "error": str(error),
        "impact": impact,
        "status": status,
    }
    with (run_root / "ERROR_LEDGER.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def generate_provenance_manifests(paths: Paths, gpu_uuid: str) -> None:
    manifest = source_manifest(paths)
    source_path = paths.run_root / "SOURCE_TREE_MANIFEST.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        paths.run_root / "EXECUTION_DAG.json",
        {
            "schema_version": VERSION,
            "seed_order": [13, 29],
            "stage_order": [
                "TA", "TB_CQET", "TB_LSE", "TB_QT", "NATIVE_C1_SUP",
                "QT_C1_SUP", "NATIVE_C2_SUP", "NATIVE_C2_KD", "QT_C2_SUP",
            ],
            "parents": {
                "TA": "fresh_init",
                "TB_CQET": "TA_epoch2",
                "TB_LSE": "TA_epoch2",
                "TB_QT": "TA_epoch2",
                "NATIVE_C1_SUP": "run_PCA_identity",
                "QT_C1_SUP": "run_PCA_identity",
                "NATIVE_C2_SUP": "selected_NATIVE_C1_SUP",
                "NATIVE_C2_KD": "selected_NATIVE_C1_SUP",
                "QT_C2_SUP": "selected_QT_C1_SUP",
            },
            "gpu_uuid": gpu_uuid,
            "source_identity_sha256": source_identity(paths),
        },
    )

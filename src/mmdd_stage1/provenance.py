"""Immutable stage receipts and source/output identity for Stage-1 CQET."""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from . import EXPERIMENT_ID, VERSION
from .config import STAGE_ORDER, Paths
from .data import iter_jsonl, json_identity, sha256_file, write_json

SOURCE_AMENDMENTS = "SOURCE_AMENDMENTS.jsonl"


def source_files(paths: Paths) -> list[Path]:
    """Every project file a Stage-1 run executes: the package, entrypoint, encoder and CPU contracts."""
    root = paths.repo_root
    files = sorted((root / "src" / "mmdd_stage1").glob("*.py"))
    files.extend([
        root / "src" / "run_stage1.py",
        root / "src" / "cache_stage1_features.py",
        root / "src" / "mmdd_progress.py",
        root / "tests" / "test_stage1_cqet.py",
        root / "tests" / "test_stage1_reference_contracts.py",
        root / "tests" / "stage1_reference_contracts.py",
        root / "tests" / "conftest.py",
    ])
    backbone_script = paths.backbone_dir / "scripts" / "qwen3_vl_embedding.py"
    if backbone_script.is_file():
        files.append(backbone_script)
    return sorted(set(files), key=lambda path: str(path).encode("utf-8"))


def source_manifest(paths: Paths) -> list[dict[str, object]]:
    files = source_files(paths)
    return [
        {
            "path": str(path.relative_to(paths.repo_root)) if path.is_relative_to(paths.repo_root) else str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": "executed_source",
        }
        for path in files
    ]


def source_identity(paths: Paths) -> str:
    return json_identity(source_manifest(paths))


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
        "from_source_identity_sha256": json_identity(previous),
        "to_source_identity_sha256": json_identity(current),
        "changed_paths": changed,
        "carried_stages": list(carried_stages),
        "reason": reason,
    }
    row["receipt_sha256"] = json_identity(row)
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
    paths: Paths,
    *,
    parents: dict[str, str],
    config: dict[str, Any],
    inputs: dict[str, Any],
    lists: dict[str, str],
) -> str:
    """Write the immutable PRE_RUN receipt of a new attempt and return its id."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    attempt = _next_attempt(stage_dir)
    payload = {
        "schema_version": VERSION,
        "experiment_id": EXPERIMENT_ID,
        "attempt_id": attempt,
        "stage": stage,
        "seed": seed,
        "status": "RUNNING",
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": {
            "physical_index": 0,
            "uuid": paths.gpu_uuid,
            "mapped_device": "cuda:0",
            "visible_device_count": torch.cuda.device_count(),
        },
        "pid": os.getpid(),
        "parents": parents,
        "config": config,
        "inputs": inputs,
        "training_lists": lists,
        "source_identity_sha256": source_identity(paths),
        "protocol_sha256": sha256_file(paths.protocol_path),
        "feature_identity_sha256": json.loads((paths.run_root / "CACHE_IDENTITY.json").read_text())["identity_sha256"],
        "command": list(sys.argv),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    payload["receipt_sha256"] = json_identity(payload)
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
    payload["receipt_sha256"] = json_identity(payload)
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


def generate_provenance_manifests(paths: Paths, gpu_uuid: str, seeds: Sequence[int] = (13,)) -> None:
    manifest = source_manifest(paths)
    source_path = paths.run_root / "SOURCE_TREE_MANIFEST.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        paths.run_root / "EXECUTION_DAG.json",
        {
            "schema_version": VERSION,
            "seed_order": list(seeds),
            "stage_order": list(STAGE_ORDER),
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


def verify_file_manifest(root: Path, manifest: Path, check_stages: bool = False) -> dict[str, Any]:
    """Re-hash every file in a delivery ledger. This does NOT validate model/metric semantics."""
    root = Path(root).resolve()
    checked: set[str] = set()
    for number, row in enumerate(iter_jsonl(manifest), 1):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path at {number}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError(f"path escapes root: {relative}")
        if str(relative) in checked:
            raise ValueError(f"duplicate entry: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"hash/size mismatch: {relative}")
        checked.add(str(relative))
    if not checked:
        raise ValueError("empty manifest")
    if check_stages:
        protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
        for seed in protocol["seeds"]:
            for stage in protocol["stage_order"]:
                posts = list((root / f"seed{seed}" / stage).glob("POST_RUN.*.json"))
                if not any(json.loads(p.read_text()).get("status") == "SUCCESS" for p in posts):
                    raise ValueError(f"{seed}/{stage} has no SUCCESS POST receipt")
    return {"status": "PASS_FILE_INTEGRITY_ONLY", "checked_files": len(checked), "stages_checked": check_stages}

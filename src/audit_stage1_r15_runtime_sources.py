#!/usr/bin/env python
"""Audit pre-run CPython cache evidence for R15's local import closure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import marshal
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import CodeType
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_equivalent(left: CodeType, right: CodeType) -> bool:
    """Compare executable code recursively without marshal reference-table noise."""

    attributes = (
        "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
        "co_stacksize", "co_flags", "co_code", "co_names", "co_varnames",
        "co_filename", "co_name", "co_firstlineno", "co_lnotab", "co_linetable",
        "co_exceptiontable", "co_freevars", "co_cellvars", "co_qualname",
    )
    for name in attributes:
        if hasattr(left, name) and getattr(left, name) != getattr(right, name):
            return False
    if len(left.co_consts) != len(right.co_consts):
        return False
    for first, second in zip(left.co_consts, right.co_consts):
        if isinstance(first, CodeType) or isinstance(second, CodeType):
            if not (isinstance(first, CodeType) and isinstance(second, CodeType)
                    and code_equivalent(first, second)):
                return False
        elif type(first) is not type(second) or first != second:
            return False
    return True


def local_module(root: Path, name: str) -> Path | None:
    source_root = root / "src"
    stem = source_root.joinpath(*name.split("."))
    module = stem.with_suffix(".py")
    package = stem / "__init__.py"
    return module if module.is_file() else package if package.is_file() else None


def imported_modules(path: Path, module_name: str) -> set[str]:
    """Return syntactically referenced imports, including function-local imports."""

    package = module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]
    result = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                if not package:
                    continue
                name = importlib.util.resolve_name("." * node.level + name, package)
            if name:
                result.add(name)
    return result


def local_import_closure(root: Path, entrypoint: Path) -> dict[str, Path]:
    pending = sorted(imported_modules(entrypoint, "run_stage1_r15"))
    found: dict[str, Path] = {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = local_module(root, name)
        if path is None:
            continue
        found[name] = path
        pending.extend(imported_modules(path, name) - found.keys())
    return found


def cache_path(source: Path) -> Path:
    return source.parent / "__pycache__" / f"{source.stem}.cpython-310.pyc"


def cache_record(source: Path, training_started_at: datetime) -> dict[str, Any]:
    pyc = cache_path(source)
    payload = pyc.read_bytes()
    if payload[:4] != importlib.util.MAGIC_NUMBER:
        raise ValueError(f"Unexpected CPython cache version: {pyc}")
    flags, source_mtime, source_size = struct.unpack("<III", payload[4:16])
    if flags != 0:
        raise ValueError(f"Expected timestamp-based CPython cache: {pyc}")
    cached_code = marshal.loads(payload[16:])
    if not isinstance(cached_code, CodeType):
        raise ValueError(f"CPython cache does not contain a code object: {pyc}")
    compiled_code = compile(
        source.read_text(encoding="utf-8"), cached_code.co_filename, "exec",
        dont_inherit=True,
    )
    stat = source.stat()
    pyc_stat = pyc.stat()
    evidence = {
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_bytes": stat.st_size,
        "source_mtime_epoch_seconds": int(stat.st_mtime),
        "pyc": str(pyc.resolve()),
        "pyc_sha256": sha256_file(pyc),
        "pyc_mtime_utc": datetime.fromtimestamp(pyc_stat.st_mtime, timezone.utc).isoformat(),
        "header_source_mtime_matches": source_mtime == int(stat.st_mtime),
        "header_source_size_matches": source_size == stat.st_size,
        "pyc_created_before_training_start": pyc_stat.st_mtime <= training_started_at.timestamp(),
        "cached_bytecode_matches_current_source": code_equivalent(
            cached_code, compiled_code
        ),
        "bytecode_payload_sha256": sha256_bytes(payload[16:]),
        "cached_code_filename": cached_code.co_filename,
    }
    evidence["passed"] = all(
        evidence[key]
        for key in (
            "header_source_mtime_matches",
            "header_source_size_matches",
            "pyc_created_before_training_start",
            "cached_bytecode_matches_current_source",
        )
    )
    return evidence


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / "work/stage1_optimization_r15_20260909"
    entrypoint = output / "source_snapshot/historical/run_stage1_r15.py"
    manifests = [
        output / f"stageI_interaction/{arm}_seed13/manifest.json"
        for arm in ("l_eoff", "n_eoff")
    ]
    run_windows = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        completed = datetime.fromisoformat(manifest["completed_at_utc"])
        started = completed - timedelta(seconds=manifest["cost"]["elapsed_seconds"])
        run_windows.append({
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
            "started_at_utc_derived": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "elapsed_seconds": manifest["cost"]["elapsed_seconds"],
            "entrypoint_sha256": manifest["code_sha256"],
        })
    training_started_at = min(
        datetime.fromisoformat(row["started_at_utc_derived"]) for row in run_windows
    )
    modules = local_import_closure(root, entrypoint)
    records = [
        {"module": name, **cache_record(path, training_started_at)}
        for name, path in sorted(modules.items())
    ]
    r14_entrypoint = output / "source_snapshot/historical/run_stage1_r14.py"
    r14_manifest_paths = [
        root / "work/stage1_optimization_r14_20260909/stage1_B_branch_ablation/b_d_e_loss_off_seed13/manifest.json",
        root / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity/m_l_linear_residual_seed13/manifest.json",
        root / "work/stage1_optimization_r14_20260909/stage1_M_projection_capacity/m_n_gelu_residual_seed13/manifest.json",
    ]
    r14_run_windows = []
    for path in r14_manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        completed = datetime.fromisoformat(manifest["completed_at_utc"])
        started = completed - timedelta(seconds=manifest["cost"]["elapsed_seconds"])
        r14_run_windows.append({
            "arm": manifest["arm"],
            "manifest": str(path.resolve()),
            "manifest_sha256": sha256_file(path),
            "started_at_utc_derived": started.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "elapsed_seconds": manifest["cost"]["elapsed_seconds"],
            "entrypoint_sha256": manifest["code_sha256"],
        })
    r14_started_at = min(
        datetime.fromisoformat(row["started_at_utc_derived"])
        for row in r14_run_windows
    )
    r14_modules = local_import_closure(root, r14_entrypoint)
    r14_records = [
        {"module": name, **cache_record(path, r14_started_at)}
        for name, path in sorted(r14_modules.items())
    ]
    r14_pre_run = [row["module"] for row in r14_records if row["passed"]]
    r14_post_run = [row["module"] for row in r14_records if not row["passed"]]
    entrypoint_sha256 = sha256_file(entrypoint)
    r14_entrypoint_sha256 = sha256_file(r14_entrypoint)
    passed = (
        bool(records)
        and all(row["passed"] for row in records)
        and all(row["entrypoint_sha256"] == entrypoint_sha256 for row in run_windows)
    )
    return {
        "format_version": 1,
        "status": "supporting_cache_evidence_complete_not_process_trace" if passed else "failed",
        "passed": passed,
        "scope": "R15 L/N-Eoff local Python import closure only; no claim about R14 training-time imports or third-party packages",
        "entrypoint": {
            "path": str(entrypoint.resolve()),
            "sha256": entrypoint_sha256,
            "manifest_hashes_match": all(
                row["entrypoint_sha256"] == entrypoint_sha256 for row in run_windows
            ),
        },
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "run_windows": run_windows,
        "training_start_bound_utc": training_started_at.isoformat(),
        "local_modules": records,
        "local_module_count": len(records),
        "all_timestamp_size_headers_match": all(
            row["header_source_mtime_matches"] and row["header_source_size_matches"]
            for row in records
        ),
        "all_caches_precede_training_start": all(
            row["pyc_created_before_training_start"] for row in records
        ),
        "all_cached_bytecode_matches_current_source": all(
            row["cached_bytecode_matches_current_source"] for row in records
        ),
        "R14_seed13_dependency_cache_evidence": {
            "status": "partial_pre_run_cache_evidence",
            "scope": "The three R14 seed13 arms reused by R15; entrypoint identity is separately manifest-bound",
            "entrypoint": {
                "path": str(r14_entrypoint.resolve()),
                "sha256": r14_entrypoint_sha256,
                "manifest_hashes_match": all(
                    row["entrypoint_sha256"] == r14_entrypoint_sha256
                    for row in r14_run_windows
                ),
            },
            "run_windows": r14_run_windows,
            "training_start_bound_utc": r14_started_at.isoformat(),
            "local_modules": r14_records,
            "local_module_count": len(r14_records),
            "pre_run_cache_supported_count": len(r14_pre_run),
            "pre_run_cache_supported_modules": r14_pre_run,
            "post_run_cache_only_count": len(r14_post_run),
            "post_run_cache_only_modules": r14_post_run,
            "all_cached_bytecode_matches_current_source": all(
                row["cached_bytecode_matches_current_source"] for row in r14_records
            ),
            "historical_imported_module_runtime_identity_fully_verified": False,
        },
        "limitations": [
            "CPython caches are contemporaneous filesystem evidence, not a process-bound import trace.",
            "Timestamp-based cache validation does not cryptographically bind the source content by itself; exact marshaled bytecode equality supplies the content check.",
            "For R14 seed13, 15 local modules have matching pre-run caches; models.py and training.py only have post-run caches and remain unverified at execution time.",
            "This does not recover third-party package builds, masks, process-bound import traces, or Adam moment tensors.",
        ],
        "historical_imported_module_runtime_identity_fully_verified": False,
        "new_student_updates": 0,
        "new_teacher_inferences": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    destination = args.root.resolve() / "work/stage1_optimization_r15_20260909/RUNTIME_SOURCE_CACHE_AUDIT.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "modules": result["local_module_count"]}))

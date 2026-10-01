"""End-to-end V4.1 preparation, smoke, formal DAG, and frozen evaluation."""
from __future__ import annotations

import copy
import dataclasses
import gc
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .execution_layout import teacher_numerical_layout
from . import EXPERIMENT_ID, VERSION
from .artifacts import json_identity, load_pool_bundle, save_pool_bundle, save_training_records
from .config import STAGE_ORDER, Paths, load_protocol, resolve_default_paths
from .content_store import ContentStore
from .data import (
    build_content_aliases,
    iter_jsonl,
    load_split_gt,
    read_json,
    sha256_file,
    split_query_ids,
    utf8_sorted,
    write_json,
    write_jsonl,
)
from .evaluate import evaluate_student_retrieval, evaluate_teacher_matrix
from .retrieval import PoolRecord
from .features import ObjectBank, RowStore, ZStore, build_or_load_row_store, fit_pca, load_pca, load_z
from .labels import Labels, build_labels, export_eval_labels, load_labels
from .lists import (
    build_c1_edge_lists,
    build_c2_shared_graph,
    build_raw_et128_exact,
    build_raw_pools_split,
    build_ta_records,
    build_tb_records,
)
from .metrics import bootstrap_contrast, candidate_metrics, evaluate_matrix, export_funnels
from .models import FreshPathTeacher, NativeStudent, QTStudent
from .probes import student_gradient_probe, teacher_content_probe, teacher_gradient_probe
from .provenance import (
    append_error_ledger,
    assert_declared_project_imports,
    carried_source_identities,
    generate_provenance_manifests,
    record_source_amendment,
    record_stage_post_run,
    record_stage_pre_run,
    source_identity,
)
from .train import (
    StudentRecipe,
    _hash_order,
    _order_sha,
    build_teacher_logits_cache,
    model_state_sha,
    state_sha,
    train_student_c1,
    train_student_c2,
    train_ta,
    train_tb,
)

STAGES = list(STAGE_ORDER)


@dataclass
class Runtime:
    paths: Paths
    protocol: dict[str, Any]
    labels: Labels
    z_store: ZStore
    row_store: RowStore
    bank: ObjectBank
    pca_basis: torch.Tensor
    pca_mean: torch.Tensor


def _set_seed(seed: int, namespace: str) -> None:
    digest = hashlib.sha256(f"{seed}|{namespace}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _hashed_ids(ids: Sequence[str], count: int) -> list[str]:
    return sorted(ids, key=lambda value: (hashlib.sha256(value.encode()).digest(), value.encode()))[:count]


def _gpu_guard(paths: Paths) -> None:
    """The process must see exactly the GPU the protocol pins (set CUDA_VISIBLE_DEVICES to its UUID)."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal/smoke execution requires exactly one visible CUDA device")
    props = torch.cuda.get_device_properties(0)
    uuid = str(props.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    if paths.gpu_uuid is None or uuid.lower() != paths.gpu_uuid.lower():
        raise RuntimeError(f"BLOCKED_GPU_IDENTITY: cuda:0={uuid} {props.name}, protocol pins {paths.gpu_uuid}")


def _protocol_paths(protocol_path: Path, run_root: Path) -> tuple[dict[str, Any], Paths]:
    protocol = load_protocol(protocol_path)
    paths = resolve_default_paths(protocol_path, run_root)
    return protocol, paths


def _phase(paths: Paths, status: str, *, formal_training_started: bool, detail: dict | None = None) -> None:
    write_json(
        paths.run_root / "PHASE_STATUS.json",
        {
            "schema_version": VERSION,
            "status": status,
            "formal_training_started": formal_training_started,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(detail or {}),
        },
    )


def amend_source(
    protocol_path: Path,
    run_root: Path,
    *,
    amendment_id: str,
    carried_stages: Sequence[str],
    reason: str,
) -> None:
    """Carry completed stages across an execution-only source change; run before prepare."""
    _protocol, paths = _protocol_paths(protocol_path, run_root)
    row = record_source_amendment(
        paths, amendment_id=amendment_id, carried_stages=carried_stages, reason=reason,
    )
    print(json.dumps(row, indent=2, ensure_ascii=False))


def prepare(protocol_path: Path, run_root: Path) -> None:
    protocol, paths = _protocol_paths(protocol_path, run_root)
    preflight = read_json(paths.run_root / "FROZEN_RECIPE_LOCK.json")
    if preflight.get("status") != "PASS":
        raise RuntimeError("BLOCKED_FEATURE_PROVENANCE")
    generate_provenance_manifests(paths, paths.gpu_uuid, seeds=protocol["seeds"])
    canonical = build_content_aliases(paths)
    if not (paths.labels_dir / "label_stats.json").exists():
        build_labels(paths, canonical)
    labels = load_labels(paths)
    z_store = load_z(paths)
    build_or_load_row_store(paths)
    pca_report = paths.run_root / "PCA_REPORT.json"
    if not pca_report.exists():
        fit_pca(paths, z_store, labels)
    else:
        report = read_json(pca_report)
        if report.get("status") != "PASS":
            raise RuntimeError("existing current-run PCA is incomplete")
        if sha256_file(paths.pca_dir / "basis.f32.npy") != report["basis_sha256"]:
            raise RuntimeError("current-run PCA basis hash mismatch")
        if not (paths.pca_dir / "PCA_REPORT.json").exists():
            write_json(paths.pca_dir / "PCA_REPORT.json", report)
    identity = read_json(paths.run_root / "RUN_IDENTITY.json")
    identity["status"] = "PASS_READY_FOR_SMOKE"
    identity["source_identity_sha256"] = source_identity(paths)
    identity["z_unit_identity_sha256"] = z_store.sha256
    identity["pca_report_sha256"] = sha256_file(pca_report)
    identity["identity_sha256"] = json_identity({k: v for k, v in identity.items() if k != "identity_sha256"})
    write_json(paths.run_root / "RUN_IDENTITY.json", identity)
    _phase(paths, "PASS_READY_FOR_VALIDATION", formal_training_started=False)


def validate(protocol_path: Path, run_root: Path) -> None:
    """Run isolated CPU contracts and bind the real prepared adapter to the source hash."""
    _protocol, paths = _protocol_paths(protocol_path, run_root)
    phase = read_json(paths.run_root / "PHASE_STATUS.json")
    if phase.get("status") != "PASS_READY_FOR_VALIDATION":
        raise RuntimeError("prepare must complete before CPU validation")
    test_paths = [
        paths.repo_root / "tests" / "test_mmdd_cqet_v4_1.py",
        paths.repo_root / "audit" / "MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE" / "tests",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(paths.repo_root / "src")
    command = [sys.executable, "-m", "pytest", *(str(path) for path in test_paths), "-q"]
    completed = subprocess.run(
        command, cwd="/tmp", env=environment, text=True, capture_output=True, check=False,
    )
    validation_dir = paths.run_root / "tests" / "integration"
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "pytest.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    labels = load_labels(paths)
    exposure = list(iter_jsonl(paths.labels_dir / "training_exposure.jsonl.gz"))
    feature_probe_path = paths.run_root / "tests" / "real_tensor_probes" / "feature_provenance.json"
    feature_probe = read_json(feature_probe_path)
    receipt = {
        "schema_version": VERSION,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "source_identity_sha256": source_identity(paths),
        "pytest_command": command,
        "pytest_returncode": completed.returncode,
        "pytest_output_sha256": sha256_file(validation_dir / "pytest.txt"),
        "label_stats": labels.stats,
        "train_exposure_rows": len(exposure),
        "train_eligible_rows": sum(bool(row["eligible"]) for row in exposure),
        "alias_archive_sha256": sha256_file(paths.run_root / "CONTENT_ALIASES.jsonl.gz"),
        "feature_probe_sha256": sha256_file(feature_probe_path),
        "feature_probe_status": feature_probe.get("status"),
        "real_adapter_checks": {
            "all_eligible_queries_loaded": sum(bool(row["eligible"]) for row in exposure) == len(labels.queries),
            "canonical_text_nonempty": bool(labels.canonical_text),
            "canonical_image_nonempty": bool(labels.canonical_image),
            "legal_targets_nonempty": bool(labels.legal_targets),
            "test_labels_absent_before_global_freeze": not (
                paths.run_root / "eval_labels" / "test" / "qrels.jsonl"
            ).exists(),
        },
    }
    if receipt["feature_probe_status"] != "PASS" or not all(receipt["real_adapter_checks"].values()):
        receipt["status"] = "FAIL"
    write_json(validation_dir / "VALIDATION_RECEIPT.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("CPU reference/adapter integration validation failed")
    _phase(paths, "PASS_READY_FOR_SMOKE", formal_training_started=False,
           detail={"validation_receipt_sha256": sha256_file(validation_dir / "VALIDATION_RECEIPT.json")})


def load_runtime(protocol_path: Path, run_root: Path) -> Runtime:
    protocol, paths = _protocol_paths(protocol_path, run_root)
    labels = load_labels(paths)
    z_store = load_z(paths)
    row_store = build_or_load_row_store(paths)
    basis, mean = load_pca(paths)
    content = ContentStore(paths.pure_cache_dir / "content", lru_bytes=8 * 2**30)
    bank = ObjectBank(z_store, content)
    return Runtime(paths, protocol, labels, z_store, row_store, bank, basis, mean)


def _teacher() -> FreshPathTeacher:
    return FreshPathTeacher(
        input_dim=4096, width=512, heads=8, layers=3, ffn=2048,
        text_slots=16, image_slots=24, dropout=0.1,
    )


def _load_teacher(path: Path, device: str = "cuda:0") -> FreshPathTeacher:
    model = _teacher()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def _load_native(path: Path, rt: Runtime, device: str = "cuda:0") -> NativeStudent:
    model = NativeStudent(rt.pca_basis, rt.pca_mean)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def _load_qt(path: Path, rt: Runtime, device: str = "cuda:0") -> QTStudent:
    model = QTStudent(rt.pca_basis, rt.pca_mean)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def _checkpoint_state(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return state_sha(payload["model"])


def _stage_paths(seed_dir: Path, stage: str, attempt: str) -> tuple[Path, Path]:
    stage_dir = seed_dir / stage
    checkpoints = stage_dir / "attempts" / attempt / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    return stage_dir, checkpoints


def _publish_checkpoints(stage_dir: Path, checkpoints: Path) -> None:
    alias = stage_dir / "checkpoints"
    if alias.exists() or alias.is_symlink():
        raise FileExistsError(f"checkpoint alias already exists: {alias}")
    alias.symlink_to(checkpoints.relative_to(stage_dir))


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _completed_stage_result(paths: Paths, stage_dir: Path, stage: str) -> Any | None:
    alias = stage_dir / "checkpoints"
    if not (alias.exists() or alias.is_symlink()):
        return None
    posts = sorted(stage_dir.glob("POST_RUN.attempt_*.json"))
    post_payloads = [read_json(path) for path in posts]
    successful = [payload for payload in post_payloads if payload.get("status") == "SUCCESS"]
    if not successful:
        raise RuntimeError(f"{stage}: published checkpoints lack a SUCCESS receipt")
    latest = successful[-1]
    pre = read_json(stage_dir / f"PRE_RUN.{latest['attempt_id']}.json")
    if pre.get("source_identity_sha256") not in carried_source_identities(paths, stage):
        raise RuntimeError(
            f"{stage}: completed source differs; carry it over with `amend-source` "
            "before prepare, or invalidate it explicitly"
        )
    for value in latest["outputs"].values():
        if not isinstance(value, dict) or "path" not in value:
            continue
        path = Path(value["path"])
        if not path.exists() or sha256_file(path) != value["sha256"]:
            raise RuntimeError(f"{stage}: completed output hash mismatch: {path}")
    if stage == "TA":
        return alias / "epoch2.pt"
    if stage.startswith("TB_"):
        return alias / "end.pt"
    return {
        int(path.stem.removeprefix("snapshot_frac")) / 100: path
        for path in sorted(alias.glob("snapshot_frac*.pt"))
    }


def _run_stage(
    rt: Runtime,
    seed: int,
    stage: str,
    *,
    parents: dict[str, str],
    config: dict[str, Any],
    inputs: dict[str, Any],
    lists: dict[str, str],
    action,
) -> Any:
    seed_dir = rt.paths.seed_dir(seed)
    stage_dir = seed_dir / stage
    completed = _completed_stage_result(rt.paths, stage_dir, stage)
    if completed is not None:
        return completed
    attempt = record_stage_pre_run(
        stage_dir, stage, seed, rt.paths.gpu_uuid, paths=rt.paths, parents=parents,
        config=config, inputs=inputs, lists=lists,
    )
    checkpoints = stage_dir / "attempts" / attempt / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    log_path = stage_dir / f"train.{attempt}.jsonl"
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cuda_start.record()
    try:
        result = action(checkpoints, log_path, attempt)
        cuda_end.record()
        torch.cuda.synchronize()
        cuda_seconds = cuda_start.elapsed_time(cuda_end) / 1000.0
        log_rows = list(iter_jsonl(log_path))
        timing_row = {
            "schema_version": VERSION,
            "seed": seed,
            "stage": stage,
            "attempt_id": attempt,
            "pid": os.getpid(),
            "gpu_uuid": rt.paths.gpu_uuid,
            "mapped_device": "cuda:0",
            "wall_seconds": time.time() - started,
            "cuda_event_seconds": cuda_seconds,
            "optimizer_steps": len(log_rows),
            "exposure_units": sum(
                int(row.get("batch_queries", row.get("batch_lists", 0))) for row in log_rows
            ),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "synchronization": "event_end_then_torch.cuda.synchronize",
        }
        timing_dir = seed_dir / "timing"
        _append_jsonl(timing_dir / "stage_wall.jsonl", timing_row)
        _append_jsonl(timing_dir / "gpu_events.jsonl", timing_row)
        _append_jsonl(timing_dir / "peak_memory.jsonl", timing_row)
        _append_jsonl(
            timing_dir / "throughput.jsonl",
            {
                **timing_row,
                "exposure_units_per_wall_second": (
                    timing_row["exposure_units"] / timing_row["wall_seconds"]
                    if timing_row["wall_seconds"] else None
                ),
            },
        )
        _publish_checkpoints(stage_dir, checkpoints)
        outputs = {path.name: str(path) for path in sorted(checkpoints.glob("*.pt"))}
        outputs["train_log"] = str(log_path)
        record_stage_post_run(
            stage_dir, stage, seed, attempt_id=attempt, status="SUCCESS",
            counters={
                "checkpoint_count": len(list(checkpoints.glob("*.pt"))),
                "optimizer_steps": timing_row["optimizer_steps"],
                "exposure_units": timing_row["exposure_units"],
            },
            outputs=outputs,
            timing={
                "wall_seconds": timing_row["wall_seconds"],
                "cuda_event_seconds": cuda_seconds,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            impact="formal training stage",
        )
        return result
    except Exception as error:
        append_error_ledger(
            rt.paths.run_root, stage, seed, error, attempt_id=attempt,
            impact="stage and all descendants are invalid until this same fixed stage succeeds",
        )
        record_stage_post_run(
            stage_dir, stage, seed, attempt_id=attempt, status="FAILED",
            timing={"wall_seconds": time.time() - started},
            impact="no descendant may consume this attempt",
        )
        raise


def _split_map(paths: Paths, query_ids: Sequence[str]) -> dict[str, str]:
    mapping = {}
    for split in ("train", "dev", "test"):
        for query_id in split_query_ids(paths, split):
            if query_id in query_ids:
                mapping[query_id] = split
    return mapping


def _candidate_summary(pools: Mapping[str, Any], gt: Mapping[str, dict]) -> dict[str, float]:
    rows = [candidate_metrics(pools[q], set(gt[q]["G"])) for q in utf8_sorted(pools)]
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def _gini_from_counts(counts: Mapping[str, int]) -> float | None:
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if not len(values) or values.sum() == 0:
        return None
    values.sort()
    n = len(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) / (n * values.sum())) - (n + 1) / n)


def _retrieval_diagnostics(pools: Mapping[str, Any]) -> dict:
    top1: dict[str, dict[str, int]] = {relation: {} for relation in (
        "QT", "Q_text", "Q_image", "text_T", "image_T",
    )}
    top10: dict[str, dict[str, int]] = {relation: {} for relation in top1}

    def add(relation: str, ranked: Sequence[str]) -> None:
        if ranked:
            top1[relation][ranked[0]] = top1[relation].get(ranked[0], 0) + 1
        for object_id in ranked[:10]:
            top10[relation][object_id] = top10[relation].get(object_id, 0) + 1

    bag_sizes = []
    score_stds = []
    overlaps: dict[str, list[float]] = {}
    for pool in pools.values():
        add("QT", [target for target, _score in pool.direct])
        add("Q_text", [evidence for evidence, _score in pool.first_hop.get("text", ())])
        add("Q_image", [evidence for evidence, _score in pool.first_hop.get("image", ())])
        by_evidence: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for target_id, paths in pool.pre_paths.items():
            for path in paths:
                by_evidence.setdefault((path.modality, path.evidence_id), []).append(
                    (target_id, path.second_score)
                )
        for (modality, _evidence), values in by_evidence.items():
            ranked = [target for target, _score in sorted(
                values, key=lambda item: (-item[1], item[0].encode("utf-8"))
            )]
            add(f"{modality}_T", ranked)
        bag_sizes.extend(len(values) for values in pool.retained_paths.values())
        if pool.qt_scores_all_U:
            score_stds.append(float(np.std(list(pool.qt_scores_all_U.values()))))
        for relation, value in pool.ann_exact_overlap.items():
            overlaps.setdefault(relation, []).append(float(value))
    return {
        "queries": len(pools),
        "mean_bag_size": float(np.mean(bag_sizes)) if bag_sizes else 0.0,
        "max_bag_size": max(bag_sizes, default=0),
        "mean_query_QT_score_std": float(np.mean(score_stds)) if score_stds else None,
        "ann_exact_overlap": {
            relation: float(np.mean(values)) for relation, values in overlaps.items()
        },
        "hub": {
            relation: {
                "top1_max_frequency": max(top1[relation].values(), default=0),
                "top10_max_frequency": max(top10[relation].values(), default=0),
                "top1_gini": _gini_from_counts(top1[relation]),
                "top10_gini": _gini_from_counts(top10[relation]),
            }
            for relation in top1
        },
    }


def _student_drift(model: NativeStudent | QTStudent) -> dict[str, float]:
    with torch.no_grad():
        if isinstance(model, QTStudent):
            return {
                "P_table_relative_to_PCA": float(
                    torch.linalg.vector_norm(model.P_table - model.pca_basis)
                    / torch.linalg.vector_norm(model.pca_basis)
                ),
                "R_QT_relative_to_identity": float(
                    torch.linalg.vector_norm(model.R_QT - torch.eye(model.dim, device=model.R_QT.device))
                    / torch.linalg.vector_norm(torch.eye(model.dim, device=model.R_QT.device))
                ),
            }
        result = {}
        for kind, parameter in model.P.items():
            result[f"P_{kind}_relative_to_PCA"] = float(
                torch.linalg.vector_norm(parameter - model.pca_basis)
                / torch.linalg.vector_norm(model.pca_basis)
            )
        identity = torch.eye(model.dim, device=next(model.parameters()).device)
        for relation, parameter in model.R.items():
            result[f"R_{relation}_relative_to_identity"] = float(
                torch.linalg.vector_norm(parameter - identity) / torch.linalg.vector_norm(identity)
            )
        return result


def _relative_model_drift(model: torch.nn.Module, parent_state: Mapping[str, torch.Tensor]) -> float:
    numerator = 0.0
    denominator = 0.0
    parameter_names = {name for name, _parameter in model.named_parameters()}
    for name, value in model.state_dict().items():
        if name not in parameter_names:
            continue
        parent = parent_state[name].to(value.device)
        numerator += float(torch.sum((value - parent) ** 2))
        denominator += float(torch.sum(parent ** 2))
    return float(np.sqrt(numerator / denominator)) if denominator else 0.0


def _student_point_funnels(
    rt: Runtime,
    seed: int,
    stage: str,
    fraction: float,
    branch: str,
    pools: Mapping[str, Any],
    dev_gt: Mapping[str, dict],
    teacher: FreshPathTeacher,
    *,
    qt: bool,
) -> dict[str, Any]:
    teacher_name = "TB_QT" if qt else "TB_CQET"
    mode = "qt" if qt else "cqet"
    point_name = f"frac{int(fraction * 100):03d}.{branch}"
    output_dir = (
        rt.paths.seed_dir(seed) / "eval" / "dev" / "trajectories" / "student"
        / stage / point_name
    )
    matrix = evaluate_teacher_matrix(
        {teacher_name: teacher}, rt.bank, pools, rt.labels,
        seed=seed, generator=f"{stage}.{point_name}", split="dev",
        output_dir=output_dir, pool_kind="C150", split_gt=dev_gt,
        teacher_modes={teacher_name: mode},
    )
    metrics = evaluate_matrix(pools, matrix, dev_gt, output_dir)
    view = "Direct" if qt else "Real"
    export_funnels(
        pools, matrix[teacher_name][view], dev_gt, rt.labels,
        output_dir / "funnels", seed=seed, generator=f"{stage}.{point_name}",
    )
    artifacts = {}
    for name in (
        "rankings.%s.%s.jsonl.gz" % (teacher_name, view),
        "logits.%s.%s.jsonl.gz" % (teacher_name, view),
        "funnels/strict_EO.jsonl.gz",
        "funnels/strict_EO_SUMMARY.json",
        "funnels/witness.jsonl.gz",
        "funnels/witness_pairs.jsonl.gz",
    ):
        path = output_dir / name
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    result = {
        "teacher": teacher_name,
        "teacher_mode": mode,
        "view": view,
        "metrics": metrics["teacher"][teacher_name][view],
        "artifacts": artifacts,
    }
    del matrix
    torch.cuda.empty_cache()
    return result


def _better(candidate: tuple[float, ...], best: tuple[float, ...] | None, tolerance: float = 1e-12) -> bool:
    if best is None:
        return True
    for left, right in zip(candidate, best):
        if left > right + tolerance:
            return True
        if left < right - tolerance:
            return False
    return False


def _select_c1(
    rt: Runtime,
    seed: int,
    checkpoints: Mapping[float, Path],
    dev_ids: Sequence[str],
    dev_gt: Mapping[str, dict],
    *,
    qt: bool,
    trajectory_teacher_checkpoint: Path,
) -> dict:
    points = []
    best_key = None
    best_fraction = None
    trajectory_teacher = _load_teacher(trajectory_teacher_checkpoint)
    for fraction, checkpoint in sorted(checkpoints.items()):
        model = _load_qt(checkpoint, rt) if qt else _load_native(checkpoint, rt)
        pools = evaluate_student_retrieval(
            model, rt.z_store, rt.row_store, dev_ids, rt.labels, "dev",
            hnsw_seed=seed, generator_id="QT_C1" if qt else "Native_C1",
        )
        summary = _candidate_summary(pools, dev_gt)
        key = (
            (summary["Direct_ANN_R10"], -fraction)
            if qt else
            (summary["C150_target_coverage"], summary["U_target_coverage"],
             summary["Direct_ANN_R10"], -fraction)
        )
        points.append({
            "fraction": fraction,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_state_sha256": model_state_sha(model),
            "metrics": summary,
            "diagnostics": _retrieval_diagnostics(pools),
            "parameter_drift": _student_drift(model),
            "selection_key": list(key),
            "strict_witness_trajectory": _student_point_funnels(
                rt, seed, "QT_C1_SUP" if qt else "NATIVE_C1_SUP",
                fraction, "SUP", pools, dev_gt, trajectory_teacher, qt=qt,
            ),
        })
        if _better(key, best_key):
            best_key = key
            best_fraction = fraction
        del pools, model
        torch.cuda.empty_cache()
    del trajectory_teacher
    torch.cuda.empty_cache()
    selected = next(point for point in points if point["fraction"] == best_fraction)
    return {
        "schema_version": VERSION,
        "selection_owner": "QT_C1_SUP" if qt else "NATIVE_C1_SUP",
        "rule": "Direct_ANN_R10_overall_then_earlier" if qt else
                "CR_C150_overall_then_RawUnionRecall_overall_then_Direct_ANN_R10_then_earlier",
        "points": points,
        "selected_fraction": best_fraction,
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_state_sha256": selected["model_state_sha256"],
    }


def _select_c2(
    rt: Runtime,
    seed: int,
    sup_checkpoints: Mapping[float, Path],
    kd_checkpoints: Mapping[float, Path] | None,
    dev_ids: Sequence[str],
    dev_gt: Mapping[str, dict],
    *,
    qt: bool,
    parent_checkpoint: Path,
    trajectory_teacher_checkpoint: Path,
) -> dict:
    points = []
    best_key = None
    best_fraction = None
    parent_state = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)["model"]
    trajectory_teacher = _load_teacher(trajectory_teacher_checkpoint)
    for fraction, checkpoint in sorted(sup_checkpoints.items()):
        sup_model = _load_qt(checkpoint, rt) if qt else _load_native(checkpoint, rt)
        sup_pools = evaluate_student_retrieval(
            sup_model, rt.z_store, rt.row_store, dev_ids, rt.labels, "dev",
            hnsw_seed=seed, generator_id="QT_C2_SUP" if qt else "Native_C2_SUP",
        )
        sup_metrics = _candidate_summary(sup_pools, dev_gt)
        key = (
            (sup_metrics["Direct_ANN_R10"], -fraction)
            if qt else
            (sup_metrics["C150_target_coverage"], sup_metrics["U_target_coverage"],
             sup_metrics["Direct_ANN_R10"], -fraction)
        )
        point = {
            "fraction": fraction,
            "SUP_checkpoint": str(checkpoint),
            "SUP_checkpoint_sha256": sha256_file(checkpoint),
            "SUP_state_sha256": model_state_sha(sup_model),
            "SUP_metrics": sup_metrics,
            "SUP_diagnostics": _retrieval_diagnostics(sup_pools),
            "SUP_parameter_drift": _student_drift(sup_model),
            "SUP_relative_to_selected_C1": _relative_model_drift(sup_model, parent_state),
            "selection_key": list(key),
            "SUP_strict_witness_trajectory": _student_point_funnels(
                rt, seed, "QT_C2_SUP" if qt else "NATIVE_C2_SUP",
                fraction, "SUP", sup_pools, dev_gt, trajectory_teacher, qt=qt,
            ),
        }
        if kd_checkpoints is not None:
            kd_model = _load_native(kd_checkpoints[fraction], rt)
            kd_pools = evaluate_student_retrieval(
                kd_model, rt.z_store, rt.row_store, dev_ids, rt.labels, "dev",
                hnsw_seed=seed, generator_id="Native_C2_KD",
            )
            point.update({
                "KD_checkpoint": str(kd_checkpoints[fraction]),
                "KD_checkpoint_sha256": sha256_file(kd_checkpoints[fraction]),
                "KD_state_sha256": model_state_sha(kd_model),
                "KD_metrics": _candidate_summary(kd_pools, dev_gt),
                "KD_diagnostics": _retrieval_diagnostics(kd_pools),
                "KD_parameter_drift": _student_drift(kd_model),
                "KD_relative_to_selected_C1": _relative_model_drift(kd_model, parent_state),
                "KD_strict_witness_trajectory": _student_point_funnels(
                    rt, seed, "NATIVE_C2_KD", fraction, "KD", kd_pools,
                    dev_gt, trajectory_teacher, qt=False,
                ),
            })
            del kd_model, kd_pools
        points.append(point)
        if _better(key, best_key):
            best_key = key
            best_fraction = fraction
        del sup_model, sup_pools
        torch.cuda.empty_cache()
    del trajectory_teacher
    torch.cuda.empty_cache()
    selected = next(point for point in points if point["fraction"] == best_fraction)
    result = {
        "schema_version": VERSION,
        "selection_owner": "QT_C2_SUP" if qt else "NATIVE_C2_SUP",
        "points": points,
        "selected_fraction": best_fraction,
        "SUP_checkpoint": selected["SUP_checkpoint"],
        "SUP_checkpoint_sha256": selected["SUP_checkpoint_sha256"],
        "SUP_state_sha256": selected["SUP_state_sha256"],
    }
    if kd_checkpoints is not None:
        result.update({
            "KD_uses_SUP_fraction": True,
            "KD_checkpoint": selected["KD_checkpoint"],
            "KD_checkpoint_sha256": selected["KD_checkpoint_sha256"],
            "KD_state_sha256": selected["KD_state_sha256"],
        })
    return result


def _attach_student_gradient_probes(
    rt: Runtime,
    seed: int,
    c2_records: Sequence[dict],
    teacher_logits: Mapping[str, tuple[torch.Tensor, torch.Tensor | None]],
    native_c1_selection: dict,
    qt_c1_selection: dict,
    native_c2_selection: dict,
    qt_c2_selection: dict,
    recipe: StudentRecipe,
) -> dict[str, Any]:
    probe_ids = _hashed_ids([row["query_id"] for row in c2_records], 128)
    by_query = {row["query_id"]: row for row in c2_records}
    probe_rows = [by_query[query_id] for query_id in probe_ids]
    probe_logits = {query_id: teacher_logits[query_id] for query_id in probe_ids}
    summary: dict[str, Any] = {
        "schema_version": VERSION,
        "seed": seed,
        "fixed_train_probe_query_ids": probe_ids,
        "probe_identity_sha256": json_identity(probe_ids),
        "points": {},
    }

    def attach(point: dict, key: str, checkpoint_key: str, *, qt: bool) -> None:
        checkpoint = Path(point[checkpoint_key])
        model = _load_qt(checkpoint, rt) if qt else _load_native(checkpoint, rt)
        probe = student_gradient_probe(
            model, rt.bank, probe_rows, None if qt else probe_logits, recipe=recipe,
        )
        probe.update({
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "state_sha256": model_state_sha(model),
        })
        point[key] = probe
        summary["points"][f"{key}:{checkpoint.name}:{probe['state_sha256']}"] = probe
        del model
        torch.cuda.empty_cache()

    for point in native_c1_selection["points"]:
        attach(point, "gradient_probe", "checkpoint", qt=False)
    for point in qt_c1_selection["points"]:
        attach(point, "gradient_probe", "checkpoint", qt=True)
    for point in native_c2_selection["points"]:
        attach(point, "SUP_gradient_probe", "SUP_checkpoint", qt=False)
        attach(point, "KD_gradient_probe", "KD_checkpoint", qt=False)
    for point in qt_c2_selection["points"]:
        attach(point, "SUP_gradient_probe", "SUP_checkpoint", qt=True)
    return summary


def smoke(protocol_path: Path, run_root: Path) -> None:
    rt = load_runtime(protocol_path, run_root)
    _gpu_guard(rt.paths)
    if read_json(rt.paths.run_root / "PHASE_STATUS.json")["status"] != "PASS_READY_FOR_SMOKE":
        raise RuntimeError("prepare and CPU integration gates must pass before smoke")
    validation = read_json(rt.paths.run_root / "tests" / "integration" / "VALIDATION_RECEIPT.json")
    if validation.get("status") != "PASS" or validation.get("source_identity_sha256") != source_identity(rt.paths):
        raise RuntimeError("CPU validation does not match the current execution source")
    assert_declared_project_imports(rt.paths)
    smoke_parent = rt.paths.run_root / "tests" / "smoke"
    attempt_number = len(list(smoke_parent.glob("attempt_*"))) + 1
    smoke_root = smoke_parent / f"attempt_{attempt_number:03d}"
    smoke_root.mkdir(parents=True)
    train_ids = _hashed_ids(rt.labels.query_ids, 8)
    split_map = {query_id: "train" for query_id in train_ids}
    raw = build_raw_pools_split(
        rt.z_store, rt.row_store, train_ids, rt.labels, split_map,
        hnsw_seed=13, device="cuda:0",
    )
    evidence_anchors = utf8_sorted({
        evidence
        for query_id in train_ids
        for values in rt.labels.queries[query_id]["W"].values()
        for evidence in values
    })
    raw_et = build_raw_et128_exact(rt.z_store, rt.labels, anchors=evidence_anchors)
    ta_records = [build_ta_records(q, raw[q], rt.labels, 13, raw_et) for q in train_ids]
    tb_records = [build_tb_records(q, raw[q], rt.labels, 13) for q in train_ids]
    subset_anchors = [
        row for row in rt.labels.edge_anchors
        if row["anchor_id"] in set(train_ids) | set(evidence_anchors)
    ]
    smoke_labels = dataclasses.replace(rt.labels, edge_anchors=subset_anchors)
    edge_lists = build_c1_edge_lists(smoke_labels, raw, rt.z_store, 13, raw_et)

    teacher_ta = rt.protocol["teacher"]["TA"]
    teacher_tb = rt.protocol["teacher"]["TB"]
    recipe = StudentRecipe.from_protocol(rt.protocol)
    student_kwargs = dict(
        lr_p=recipe.lr_p, lr_r=recipe.lr_r, logit_scale=recipe.logit_scale,
        anchor_weight=recipe.anchor_weight, clip_norm=recipe.clip_norm,
    )
    _set_seed(13, "SMOKE_TA")
    ta_model = _teacher()
    ta_end = train_ta(
        ta_model, rt.bank, ta_records, rt.labels,
        epochs=int(teacher_ta["epochs"]), lr=float(teacher_ta["lr"]),
        weight_decay=float(teacher_ta["wd"]), logical_batch=int(teacher_ta["logical_batch_queries"]),
        support_weight=float(teacher_ta["support_weight"]),
        save_dir=smoke_root / "TA", seed=13,
    )
    tb_models = {}
    for mode in ("cqet", "lse", "qt"):
        _set_seed(13, "SMOKE_TB_SHARED")
        model = _load_teacher(ta_end)
        train_tb(
            model, rt.bank, tb_records, mode=mode,
            epochs=int(teacher_tb["epochs"]), lr=float(teacher_tb["lr"]),
            weight_decay=float(teacher_tb["wd"]), logical_batch=int(teacher_tb["logical_batch_queries"]),
            direct_weight=float(teacher_tb["path_direct_weight"]),
            aggregate_weight=float(teacher_tb["path_aggregate_weight"]),
            support_weight=float(teacher_tb["support_weight"]),
            save_dir=smoke_root / f"TB_{mode.upper()}", seed=13,
        )
        tb_models[mode] = model

    native = NativeStudent(rt.pca_basis, rt.pca_mean)
    native_points = train_student_c1(
        native, edge_lists, None, rt.bank, arm="NATIVE_SUP", logical_batch=64,
        save_dir=smoke_root / "NATIVE_C1_SUP", seed=13, **student_kwargs,
    )
    qt = QTStudent(rt.pca_basis, rt.pca_mean)
    train_student_c1(
        qt, edge_lists, None, rt.bank, arm="QT_SUP", logical_batch=64,
        save_dir=smoke_root / "QT_C1_SUP", seed=13, **student_kwargs,
    )
    native_parent = native_points[1.0]
    native_selected = _load_native(native_parent, rt)
    c1_pools = evaluate_student_retrieval(
        native_selected, rt.z_store, rt.row_store, train_ids, rt.labels, "train",
        hnsw_seed=13, generator_id="smoke_native_c1",
    )
    c2_prepaths = smoke_root / "C2_SHARED_PREPATHS.jsonl.gz"
    c2_records = build_c2_shared_graph(
        raw, c1_pools, native_selected, rt.z_store, rt.row_store, rt.labels, train_ids,
        c2_prepaths,
    )
    parent_hash = model_state_sha(native_selected)
    kd_model = _load_native(native_parent, rt)
    train_student_c2(
        kd_model, c2_records, tb_models["cqet"], rt.bank, arm="NATIVE_KD",
        logical_batch=64, save_dir=smoke_root / "NATIVE_C2_KD", seed=13,
        expected_parent_hash=parent_hash, max_updates=1, **student_kwargs,
        kd_weight=recipe.kd_weight, kd_temperature=recipe.kd_temperature,
        random_negatives=recipe.random_negatives, negative_pool=rt.labels.legal_targets,
    )
    result = {
        "schema_version": VERSION,
        "status": "PASS",
        "train_queries": len(train_ids),
        "optimizer_updates": 8,
        "limits": {"max_train_queries": 16, "max_updates": 8},
        "source_identity_sha256": source_identity(rt.paths),
        "smoke_as_formal_parent": False,
        "query_ids": train_ids,
        "c2_parent_state_sha256": parent_hash,
        "c2_step0_state_sha256": _checkpoint_state(smoke_root / "NATIVE_C2_KD" / "snapshot_frac000.pt"),
        "c2_graph_sha256": json_identity(list(iter_jsonl(c2_prepaths))),
    }
    if result["c2_parent_state_sha256"] != result["c2_step0_state_sha256"]:
        raise AssertionError("smoke C2 parent/step0 mismatch")
    write_json(smoke_root / "SMOKE_RESULT.json", result)
    write_json(rt.paths.run_root / "tests" / "smoke" / "LATEST.json", result)
    _phase(rt.paths, "PASS_READY_FOR_FORMAL", formal_training_started=False,
           detail={"smoke_result": str(smoke_root / "SMOKE_RESULT.json")})


def _materialize_raw_train_dev(
    rt: Runtime,
    seed: int,
    raw_et: Mapping[str, Sequence[str]],
    dev_ids: Sequence[str],
):
    seed_dir = rt.paths.seed_dir(seed)
    train_ids = rt.labels.query_ids
    train_bundle = seed_dir / "training_records" / "raw_train"
    dev_bundle = seed_dir / "eval" / "dev" / "raw"
    if (train_bundle / "POOL_MANIFEST.json").exists():
        raw_train = _load_raw_bundle(train_bundle, train_ids, seed)
    else:
        raw_train = build_raw_pools_split(
            rt.z_store, rt.row_store, train_ids, rt.labels, "train", hnsw_seed=seed,
            index_dir=train_bundle / "indices",
        )
        save_pool_bundle(train_bundle, raw_train, rt.labels, seed=seed, generator="raw")
    if (dev_bundle / "POOL_MANIFEST.json").exists():
        raw_dev = _load_raw_bundle(dev_bundle, dev_ids, seed)
    else:
        raw_dev = build_raw_pools_split(
            rt.z_store, rt.row_store, dev_ids, rt.labels, "dev", hnsw_seed=seed,
            reuse_index_dir=train_bundle / "indices",
        )
        save_pool_bundle(dev_bundle, raw_dev, rt.labels, seed=seed, generator="raw")
    return raw_train, raw_dev


def _load_raw_bundle(directory: Path, query_ids: Sequence[str], seed: int) -> dict[str, PoolRecord]:
    """Reload a completed same-seed Raw bundle; Raw pools do not depend on any trained model."""
    pools = load_pool_bundle(directory)
    if list(pools) != list(query_ids):
        raise RuntimeError(f"Raw bundle query set/order differs from current split: {directory}")
    with gzip.open(directory / "pools.jsonl.gz", "rt", encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    if first["seed"] != seed or first["generator"] != "raw":
        raise RuntimeError(f"Raw bundle seed/generator mismatch: {directory}")
    return pools


def train_seed(rt: Runtime, seed: int, raw_et: Mapping[str, Sequence[str]]) -> dict:
    seed_dir = rt.paths.seed_dir(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(seed_dir / "execution_dag.json", {"seed": seed, "stage_order": STAGES})
    train_ids = rt.labels.query_ids
    dev_gt = export_eval_labels(rt.paths, rt.labels.canonical_map, "dev")
    dev_ids = utf8_sorted(dev_gt)
    raw_train, raw_dev = _materialize_raw_train_dev(rt, seed, raw_et, dev_ids)
    # D1 traces are read only by save_pool_bundle and the Raw train bundle is on disk;
    # dropping them frees ~8 GiB of the ~25 GiB Raw train pools held until the C2 graph.
    for pool in raw_train.values():
        pool.d1_trace = {}
    training_dir = seed_dir / "training_records"
    ta_gz = training_dir / "TA.jsonl.gz"
    tb_gz = training_dir / "TB_SHARED.jsonl.gz"
    c1_native_gz = training_dir / "C1_NATIVE.jsonl.gz"
    c1_qt_gz = training_dir / "C1_QT.jsonl.gz"
    if ta_gz.exists() and tb_gz.exists() and c1_native_gz.exists() and c1_qt_gz.exists():
        ta_records = list(iter_jsonl(ta_gz))
        tb_records = list(iter_jsonl(tb_gz))
        edge_lists = list(iter_jsonl(c1_native_gz))
        hashes = {
            "TA": sha256_file(ta_gz),
            "TB_SHARED": sha256_file(tb_gz),
            "C1_NATIVE": sha256_file(c1_native_gz),
            "C1_QT": sha256_file(c1_qt_gz),
        }
    else:
        ta_records = [build_ta_records(q, raw_train[q], rt.labels, seed, raw_et) for q in train_ids]
        tb_records = [build_tb_records(q, raw_train[q], rt.labels, seed) for q in train_ids]
        edge_lists = build_c1_edge_lists(rt.labels, raw_train, rt.z_store, seed, raw_et)
        hashes = {
            "TA": save_training_records(ta_gz, ta_records),
            "TB_SHARED": save_training_records(tb_gz, tb_records),
            "C1_NATIVE": save_training_records(c1_native_gz, edge_lists),
            "C1_QT": save_training_records(
                c1_qt_gz, [row for row in edge_lists if row["relation"] == "QT"]
            ),
        }
    common_inputs = {
        "dataset_identity": read_json(rt.paths.run_root / "DATASET_IDENTITY.json")["identity_sha256"],
        "cache_identity": read_json(rt.paths.run_root / "CACHE_IDENTITY.json")["identity_sha256"],
        "pca_report": sha256_file(rt.paths.run_root / "PCA_REPORT.json"),
        "raw_train_pool_manifest": sha256_file(training_dir / "raw_train" / "POOL_MANIFEST.json"),
        "raw_dev_pool_manifest": sha256_file(seed_dir / "eval" / "dev" / "raw" / "POOL_MANIFEST.json"),
    }

    teacher_ta = rt.protocol["teacher"]["TA"]
    teacher_tb = rt.protocol["teacher"]["TB"]
    recipe = StudentRecipe.from_protocol(rt.protocol)
    c1_epochs = int(rt.protocol["student"]["C1"]["epochs"])
    c1_batch = int(rt.protocol["student"]["C1"]["logical_batch_edge_lists"])
    c2_epochs = int(rt.protocol["student"]["C2"]["epochs"])
    c2_batch = int(rt.protocol["student"]["C2"]["logical_batch_queries"])
    student_kwargs = dict(
        lr_p=recipe.lr_p, lr_r=recipe.lr_r, logit_scale=recipe.logit_scale,
        anchor_weight=recipe.anchor_weight, clip_norm=recipe.clip_norm,
    )

    _set_seed(seed, "TA")
    ta_model = _teacher()
    ta_batch = int(teacher_ta["logical_batch_queries"])
    ta_config = {
        **teacher_ta,
        "optimizer_steps": int(teacher_ta["epochs"]) * ((len(ta_records) + ta_batch - 1) // ta_batch),
        "order_sha256_by_epoch": {
            str(epoch): _order_sha(_hash_order(ta_records, "TA", seed, epoch))
            for epoch in range(1, int(teacher_ta["epochs"]) + 1)
        },
        "numerical_layout": teacher_numerical_layout(),
    }
    ta_ckpt = _run_stage(
        rt, seed, "TA", parents={"fresh_init_state": model_state_sha(ta_model)},
        config=ta_config, inputs=common_inputs, lists={"TA": hashes["TA"]},
        action=lambda ckpts, log, attempt: train_ta(
            ta_model, rt.bank, ta_records, rt.labels, save_dir=ckpts,
            epochs=int(teacher_ta["epochs"]), lr=float(teacher_ta["lr"]),
            weight_decay=float(teacher_ta["wd"]), logical_batch=ta_batch,
            support_weight=float(teacher_ta["support_weight"]),
            seed=seed, metadata={"attempt": attempt, **common_inputs}, log_path=log,
        ),
    )
    ta_hash = _checkpoint_state(ta_ckpt)

    tb_ckpts = {}
    tb_batch = int(teacher_tb["logical_batch_queries"])
    tb_config = {
        **teacher_tb,
        "optimizer_steps": int(teacher_tb["epochs"]) * ((len(tb_records) + tb_batch - 1) // tb_batch),
        "order_sha256": _order_sha(_hash_order(tb_records, "TB_SHARED", seed, 1)),
        "numerical_layout": teacher_numerical_layout(),
    }
    for stage, mode in (("TB_CQET", "cqet"), ("TB_LSE", "lse"), ("TB_QT", "qt")):
        _set_seed(seed, "TB_SHARED")
        model = _load_teacher(ta_ckpt)
        if model_state_sha(model) != ta_hash:
            raise AssertionError(f"{stage} did not load TA epoch2")
        tb_ckpts[stage] = _run_stage(
            rt, seed, stage,
            parents={"TA_epoch2": sha256_file(ta_ckpt), "TA_state": ta_hash},
            config={**tb_config, "mode": mode},
            inputs=common_inputs, lists={"TB_SHARED": hashes["TB_SHARED"]},
            action=lambda ckpts, log, attempt, model=model, mode=mode: train_tb(
                model, rt.bank, tb_records, mode=mode, save_dir=ckpts,
                epochs=int(teacher_tb["epochs"]), lr=float(teacher_tb["lr"]),
                weight_decay=float(teacher_tb["wd"]), logical_batch=tb_batch,
                direct_weight=float(teacher_tb["path_direct_weight"]),
                aggregate_weight=float(teacher_tb["path_aggregate_weight"]),
                support_weight=float(teacher_tb["support_weight"]),
                seed=seed, metadata={"attempt": attempt, **common_inputs}, log_path=log,
            ),
        )

    native_c1 = NativeStudent(rt.pca_basis, rt.pca_mean)
    c1_native_order = _hash_order(edge_lists, "C1_NATIVE", seed, 1)
    c1_qt_records = [row for row in edge_lists if row["relation"] == "QT"]
    c1_qt_order = _hash_order(c1_qt_records, "C1_QT", seed, 1)
    c1_layout = {"logical_batch": c1_batch, "query_microbatch_ladder": [64, 32, 16, 8, 4, 2, 1]}
    native_c1_points = _run_stage(
        rt, seed, "NATIVE_C1_SUP",
        parents={"PCA": sha256_file(rt.paths.pca_dir / "basis.pt")},
        config={
            **rt.protocol["student"]["C1"], **recipe.as_dict(),
            "optimizer_steps": c1_epochs * ((len(c1_native_order) + c1_batch - 1) // c1_batch),
            "order_sha256": _order_sha(c1_native_order),
            "numerical_layout": c1_layout,
        }, inputs=common_inputs,
        lists={"C1_NATIVE": hashes["C1_NATIVE"]},
        action=lambda ckpts, log, attempt: train_student_c1(
            native_c1, edge_lists, None, rt.bank, arm="NATIVE_SUP", save_dir=ckpts,
            epochs=c1_epochs, logical_batch=c1_batch, **student_kwargs,
            seed=seed, metadata={"attempt": attempt, **common_inputs}, log_path=log,
        ),
    )
    qt_c1 = QTStudent(rt.pca_basis, rt.pca_mean)
    qt_c1_points = _run_stage(
        rt, seed, "QT_C1_SUP",
        parents={"PCA": sha256_file(rt.paths.pca_dir / "basis.pt")},
        config={
            **rt.protocol["student"]["C1"], **recipe.as_dict(),
            "optimizer_steps": c1_epochs * ((len(c1_qt_order) + c1_batch - 1) // c1_batch),
            "order_sha256": _order_sha(c1_qt_order),
            "numerical_layout": c1_layout,
        }, inputs=common_inputs,
        lists={"C1_QT": hashes["C1_QT"]},
        action=lambda ckpts, log, attempt: train_student_c1(
            qt_c1, edge_lists, None, rt.bank, arm="QT_SUP", save_dir=ckpts,
            epochs=c1_epochs, logical_batch=c1_batch, **student_kwargs,
            seed=seed, metadata={"attempt": attempt, **common_inputs}, log_path=log,
        ),
    )
    selections = seed_dir / "selections"
    selections.mkdir(exist_ok=True)
    native_c1_selection = _select_c1(
        rt, seed, native_c1_points, dev_ids, dev_gt, qt=False,
        trajectory_teacher_checkpoint=tb_ckpts["TB_CQET"],
    )
    qt_c1_selection = _select_c1(
        rt, seed, qt_c1_points, dev_ids, dev_gt, qt=True,
        trajectory_teacher_checkpoint=tb_ckpts["TB_QT"],
    )
    write_json(selections / "NATIVE_C1.json", native_c1_selection)
    write_json(selections / "QT_C1.json", qt_c1_selection)

    selected_native_c1 = _load_native(Path(native_c1_selection["selected_checkpoint"]), rt)
    c1_train_pools = evaluate_student_retrieval(
        selected_native_c1, rt.z_store, rt.row_store, train_ids, rt.labels, "train",
        hnsw_seed=seed, generator_id="selected_Native_C1",
        index_dir=str(training_dir / "native_c1_train_indices"),
    )
    save_pool_bundle(training_dir / "native_c1_train", c1_train_pools, rt.labels,
                     seed=seed, generator="selected_native_c1")
    c2_records = build_c2_shared_graph(
        raw_train, c1_train_pools, selected_native_c1,
        rt.z_store, rt.row_store, rt.labels, train_ids,
        training_dir / "C2_SHARED_PREPATHS.jsonl.gz",
    )
    # Neither train pool dict is read again; release them before the C2 stages.
    del raw_train, c1_train_pools
    hashes["C2_SHARED"] = save_training_records(training_dir / "C2_SHARED.jsonl.gz", c2_records)
    graph_hash = hashes["C2_SHARED"]
    parent_hash = native_c1_selection["selected_state_sha256"]
    c2_order = _hash_order(c2_records, "C2_SHARED", seed, 1)
    c2_config = {
        **rt.protocol["student"]["C2"], **recipe.as_dict(),
        "optimizer_steps": c2_epochs * ((len(c2_order) + c2_batch - 1) // c2_batch),
        "order_sha256": _order_sha(c2_order),
        "negative_pool": "legal_targets",
        "numerical_layout": {"logical_batch": c2_batch, "query_microbatch_ladder": [64, 32, 16, 8, 4, 2, 1]},
    }
    c2_kwargs = dict(
        **student_kwargs, kd_weight=recipe.kd_weight, kd_temperature=recipe.kd_temperature,
        random_negatives=recipe.random_negatives, negative_pool=rt.labels.legal_targets,
        epochs=c2_epochs, logical_batch=c2_batch,
    )

    native_sup = _load_native(Path(native_c1_selection["selected_checkpoint"]), rt)
    native_sup_points = _run_stage(
        rt, seed, "NATIVE_C2_SUP",
        parents={"selected_NATIVE_C1_SUP": native_c1_selection["selected_checkpoint_sha256"],
                 "parent_state": parent_hash},
        config=c2_config, inputs={**common_inputs, "graph_hash": graph_hash},
        lists={"C2_SHARED": hashes["C2_SHARED"]},
        action=lambda ckpts, log, attempt: train_student_c2(
            native_sup, c2_records, None, rt.bank, arm="NATIVE_SUP", save_dir=ckpts,
            seed=seed, expected_parent_hash=parent_hash, **c2_kwargs,
            metadata={"attempt": attempt, "graph_hash": graph_hash, **common_inputs}, log_path=log,
        ),
    )

    cqet_teacher = _load_teacher(tb_ckpts["TB_CQET"])
    logits = build_teacher_logits_cache(cqet_teacher, rt.bank, c2_records)
    logits_dir = seed_dir / "teacher_logits_cache"
    logits_dir.mkdir(exist_ok=True)
    logits_path = logits_dir / "scores.pt"
    torch.save(logits, logits_path)
    logits_identity = {
        "schema_version": VERSION,
        "teacher": "TB_CQET",
        "teacher_checkpoint": str(tb_ckpts["TB_CQET"]),
        "teacher_checkpoint_sha256": sha256_file(tb_ckpts["TB_CQET"]),
        "teacher_state_sha256": model_state_sha(cqet_teacher),
        "graph_sha256": graph_hash,
        "training_list_sha256": hashes["C2_SHARED"],
        "feature_identity": common_inputs["cache_identity"],
        "scoring_source_identity": source_identity(rt.paths),
        "scores_sha256": sha256_file(logits_path),
    }
    write_json(logits_dir / "identity.json", logits_identity)

    native_kd = _load_native(Path(native_c1_selection["selected_checkpoint"]), rt)
    native_kd_points = _run_stage(
        rt, seed, "NATIVE_C2_KD",
        parents={
            "selected_NATIVE_C1_SUP": native_c1_selection["selected_checkpoint_sha256"],
            "parent_state": parent_hash,
            "TB_CQET_end": sha256_file(tb_ckpts["TB_CQET"]),
        },
        config=c2_config,
        inputs={**common_inputs, "graph_hash": graph_hash, "teacher_logits": logits_identity},
        lists={"C2_SHARED": hashes["C2_SHARED"]},
        action=lambda ckpts, log, attempt: train_student_c2(
            native_kd, c2_records, cqet_teacher, rt.bank, arm="NATIVE_KD", save_dir=ckpts,
            seed=seed, expected_parent_hash=parent_hash, teacher_logits=logits, **c2_kwargs,
            metadata={"attempt": attempt, "graph_hash": graph_hash, **common_inputs}, log_path=log,
        ),
    )

    qt_parent_hash = qt_c1_selection["selected_state_sha256"]
    qt_c2 = _load_qt(Path(qt_c1_selection["selected_checkpoint"]), rt)
    qt_c2_points = _run_stage(
        rt, seed, "QT_C2_SUP",
        parents={"selected_QT_C1_SUP": qt_c1_selection["selected_checkpoint_sha256"],
                 "parent_state": qt_parent_hash},
        config=c2_config, inputs={**common_inputs, "graph_hash": graph_hash},
        lists={"C2_SHARED": hashes["C2_SHARED"]},
        action=lambda ckpts, log, attempt: train_student_c2(
            qt_c2, c2_records, None, rt.bank, arm="QT_SUP", save_dir=ckpts,
            seed=seed, expected_parent_hash=qt_parent_hash, **c2_kwargs,
            metadata={"attempt": attempt, "graph_hash": graph_hash, **common_inputs}, log_path=log,
        ),
    )

    native_c2_selection = _select_c2(
        rt, seed, native_sup_points, native_kd_points, dev_ids, dev_gt, qt=False,
        parent_checkpoint=Path(native_c1_selection["selected_checkpoint"]),
        trajectory_teacher_checkpoint=tb_ckpts["TB_CQET"],
    )
    qt_c2_selection = _select_c2(
        rt, seed, qt_c2_points, None, dev_ids, dev_gt, qt=True,
        parent_checkpoint=Path(qt_c1_selection["selected_checkpoint"]),
        trajectory_teacher_checkpoint=tb_ckpts["TB_QT"],
    )
    gradient_summary = _attach_student_gradient_probes(
        rt, seed, c2_records, logits,
        native_c1_selection, qt_c1_selection, native_c2_selection, qt_c2_selection,
        recipe=recipe,
    )
    write_json(selections / "NATIVE_C1.json", native_c1_selection)
    write_json(selections / "QT_C1.json", qt_c1_selection)
    write_json(selections / "NATIVE_C2_COMMON.json", native_c2_selection)
    write_json(selections / "QT_C2.json", qt_c2_selection)
    gradient_path = (
        seed_dir / "eval" / "dev" / "trajectories" / "student" / "GRADIENT_PROBES.json"
    )
    write_json(gradient_path, gradient_summary)
    student_trajectory = {
        "schema_version": VERSION,
        "seed": seed,
        "split": "dev",
        "query_ids_sha256": json_identity(dev_ids),
        "query_count": len(dev_ids),
        "NATIVE_C1_SUP": native_c1_selection["points"],
        "QT_C1_SUP": qt_c1_selection["points"],
        "NATIVE_C2_COMMON": native_c2_selection["points"],
        "QT_C2_SUP": qt_c2_selection["points"],
        "gradient_probes_sha256": sha256_file(gradient_path),
    }
    student_trajectory_path = (
        seed_dir / "eval" / "dev" / "trajectories" / "student" / "TRAJECTORY_SUMMARY.json"
    )
    write_json(student_trajectory_path, student_trajectory)
    teacher_trajectory = _teacher_trajectory(rt, seed, raw_dev, dev_gt, tb_records)
    freeze = {
        "schema_version": VERSION,
        "seed": seed,
        "status": "FROZEN_BEFORE_TEST",
        "NATIVE_C1": native_c1_selection,
        "QT_C1": qt_c1_selection,
        "NATIVE_C2_COMMON": native_c2_selection,
        "QT_C2": qt_c2_selection,
        "teacher_trajectory_sha256": json_identity(teacher_trajectory),
        "student_trajectory_sha256": sha256_file(student_trajectory_path),
        "test_qrels_read": False,
    }
    freeze["freeze_sha256"] = json_identity(freeze)
    write_json(seed_dir / "SELECTION_FREEZE.json", freeze)
    del raw_dev, c2_records, logits
    gc.collect()
    torch.cuda.empty_cache()
    return freeze


def _independent_verify(
    rt: Runtime,
    split: str,
    generator_dir: Path,
    metrics: dict,
    *,
    allow_subset: bool = False,
) -> dict:
    script = rt.paths.repo_root / "audit" / "MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE" / "tools" / "independent_metrics.py"
    query_path = rt.paths.run_root / "eval_labels" / split / "queries.jsonl"
    qrel_path = rt.paths.run_root / "eval_labels" / split / "qrels.jsonl"
    results = {}
    for ranking in sorted(generator_dir.glob("rankings.*.jsonl.gz")):
        name = ranking.name.removeprefix("rankings.").removesuffix(".jsonl.gz")
        out = generator_dir / "independent" / name
        command = [
            sys.executable, str(script), "--queries", str(query_path), "--qrels", str(qrel_path),
            "--rankings", str(ranking), "--out", str(out),
        ]
        if allow_subset:
            command.append("--allow-subset")
        subprocess.run(
            command,
            cwd="/tmp", check=True, stdout=subprocess.DEVNULL,
        )
        summary = read_json(out / "summary.json")
        teacher, view = name.split(".", 1)
        for segment, values in summary["segments"].items():
            expected = metrics["teacher"][teacher][view][segment]
            for k in (10, 20, 30, 40, 50):
                actual = values[f"R{k}"]
                server = expected[f"R@{k}"]
                if actual is None and server is None:
                    continue
                if abs(actual - server) > 1e-10:
                    raise AssertionError(f"independent metric mismatch: {name}/{segment}/R{k}")
        results[name] = sha256_file(out / "summary.json")
    write_json(generator_dir / "INDEPENDENT_VERIFICATION.json", results)
    return results


def _teacher_trajectory(
    rt: Runtime,
    seed: int,
    raw_dev: Mapping[str, Any],
    dev_gt: Mapping[str, dict],
    tb_records: Sequence[dict],
) -> dict:
    probe_ids = _hashed_ids(list(raw_dev), 128)
    probe_pools = {query_id: raw_dev[query_id] for query_id in probe_ids}
    train_probe_ids = _hashed_ids([row["query_id"] for row in tb_records], 128)
    train_by_query = {row["query_id"]: row for row in tb_records}
    gradient_rows = [train_by_query[query_id] for query_id in train_probe_ids]
    checkpoint_specs = [
        ("TA_init", "TA", "init.pt", "cqet"),
        ("TA_epoch1", "TA", "epoch1.pt", "cqet"),
        ("TA_epoch2", "TA", "epoch2.pt", "cqet"),
        ("TB_CQET_init", "TB_CQET", "init.pt", "cqet"),
        ("TB_CQET_half", "TB_CQET", "half.pt", "cqet"),
        ("TB_CQET_end", "TB_CQET", "end.pt", "cqet"),
        ("TB_LSE_init", "TB_LSE", "init.pt", "lse"),
        ("TB_LSE_half", "TB_LSE", "half.pt", "lse"),
        ("TB_LSE_end", "TB_LSE", "end.pt", "lse"),
        ("TB_QT_init", "TB_QT", "init.pt", "qt"),
        ("TB_QT_half", "TB_QT", "half.pt", "qt"),
        ("TB_QT_end", "TB_QT", "end.pt", "qt"),
    ]
    root = rt.paths.seed_dir(seed) / "eval" / "dev" / "trajectories" / "teacher"
    result = {
        "schema_version": VERSION,
        "seed": seed,
        "probe_query_ids": probe_ids,
        "probe_identity_sha256": json_identity(probe_ids),
        "fixed_train_probe_query_ids": train_probe_ids,
        "train_probe_identity_sha256": json_identity(train_probe_ids),
        "pool_generator": "raw",
        "points": {},
    }
    for name, stage, filename, mode in checkpoint_specs:
        checkpoint = rt.paths.seed_dir(seed) / stage / "checkpoints" / filename
        teacher = _load_teacher(checkpoint)
        directory = root / name
        matrix = evaluate_teacher_matrix(
            {name: teacher}, rt.bank, probe_pools, rt.labels,
            seed=seed, generator="raw.trajectory", split="dev", output_dir=directory,
            pool_kind="C150", split_gt=dev_gt, teacher_modes={name: mode},
        )
        metrics = evaluate_matrix(probe_pools, matrix, dev_gt, directory)
        independent = _independent_verify(rt, "dev", directory, metrics, allow_subset=True)
        content_probe = teacher_content_probe(
            teacher, rt.bank, probe_pools, rt.labels, dev_gt, matrix,
            teacher_name=name, mode=mode, seed=seed, checkpoint=checkpoint,
            output_dir=directory,
        )
        gradient_probe = teacher_gradient_probe(
            teacher, rt.bank, gradient_rows, mode=mode, stage=stage,
        )
        result["points"][name] = {
            "stage": stage,
            "mode": mode,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "state_sha256": model_state_sha(teacher),
            "metrics": metrics["teacher"][name],
            "independent": independent,
            "content_probe": content_probe,
            "gradient_probe": gradient_probe,
        }
        del teacher, matrix
        torch.cuda.empty_cache()
    write_json(root / "TRAJECTORY_SUMMARY.json", result)
    return result


def _pool_target_view(
    pools: Mapping[str, Any],
    *,
    target_lists: Mapping[str, Sequence[str]],
    generator: str,
    keep_bags: bool,
) -> dict[str, Any]:
    result = {}
    for query_id in utf8_sorted(pools):
        pool = pools[query_id]
        targets = list(target_lists[query_id])
        result[query_id] = dataclasses.replace(
            pool,
            generator_id=generator,
            C150=targets,
            U=targets if not keep_bags else pool.U,
            pre_paths=pool.pre_paths if keep_bags else {},
            retained_paths=pool.retained_paths if keep_bags else {},
            retained_coverage=pool.retained_coverage if keep_bags else {},
            d1_scores=pool.d1_scores if keep_bags else {},
            d1_ranks=pool.d1_ranks if keep_bags else {},
        )
    return result


def evaluate_seed(rt: Runtime, seed: int, global_freeze_hash: str) -> dict:
    seed_dir = rt.paths.seed_dir(seed)
    freeze = read_json(seed_dir / "SELECTION_FREEZE.json")
    selections = freeze
    teachers = {
        "TB_CQET": _load_teacher(seed_dir / "TB_CQET" / "checkpoints" / "end.pt"),
        "TB_LSE": _load_teacher(seed_dir / "TB_LSE" / "checkpoints" / "end.pt"),
        "TB_QT": _load_teacher(seed_dir / "TB_QT" / "checkpoints" / "end.pt"),
    }
    models = {
        "native_sup": _load_native(Path(selections["NATIVE_C2_COMMON"]["SUP_checkpoint"]), rt),
        "native_kd": _load_native(Path(selections["NATIVE_C2_COMMON"]["KD_checkpoint"]), rt),
        "qt_sup": _load_qt(Path(selections["QT_C2"]["SUP_checkpoint"]), rt),
    }
    result = {"seed": seed, "global_freeze_sha256": global_freeze_hash, "splits": {}}
    for split in ("dev", "test"):
        global_freeze = read_json(rt.paths.run_root / "GLOBAL_SELECTION_FREEZE.json")
        if global_freeze.get("freeze_sha256") != global_freeze_hash:
            raise RuntimeError("global selection freeze identity changed before evaluation")
        if split == "test" and global_freeze.get("status") != "ALL_SELECTIONS_FROZEN_BEFORE_TEST":
            raise RuntimeError("test firewall opened before all seed selections were frozen")
        gt = export_eval_labels(rt.paths, rt.labels.canonical_map, split)
        query_ids = utf8_sorted(gt)
        split_root = seed_dir / "eval" / split
        raw_dir = split_root / "raw"
        if (raw_dir / "POOL_MANIFEST.json").exists():
            raw_pools = load_pool_bundle(raw_dir)
        else:
            raw_pools = build_raw_pools_split(
                rt.z_store, rt.row_store, query_ids, rt.labels, split,
                hnsw_seed=seed, index_dir=raw_dir / "indices" if split == "test" else None,
            )
            save_pool_bundle(raw_dir, raw_pools, rt.labels, seed=seed, generator="raw")
        pools_by_generator = {"raw": raw_pools}
        for generator, model in models.items():
            directory = split_root / generator
            pools = evaluate_student_retrieval(
                model, rt.z_store, rt.row_store, query_ids, rt.labels, split,
                hnsw_seed=seed, generator_id=generator,
                index_dir=str(directory / "indices"),
            )
            save_pool_bundle(directory, pools, rt.labels, seed=seed, generator=generator)
            pools_by_generator[generator] = pools

        split_result = {}
        for generator, pools in pools_by_generator.items():
            directory = split_root / generator
            matrix = evaluate_teacher_matrix(
                teachers, rt.bank, pools, rt.labels, seed=seed, generator=generator,
                split=split, output_dir=directory, split_gt=gt,
            )
            metrics = evaluate_matrix(pools, matrix, gt, directory)
            primary = (
                matrix["TB_QT"]["Direct"] if generator == "qt_sup"
                else matrix["TB_CQET"]["Real"]
            )
            export_funnels(
                pools, primary, gt, rt.labels, split_root / "funnels" / generator,
                seed=seed, generator=generator,
            )
            _independent_verify(rt, split, directory, metrics)
            split_result[generator] = metrics

            if generator != "qt_sup":
                direct_results = {}
                for direct_name, field in (
                    ("MatchedDirectC", "MatchedDirectC"),
                    ("MatchedDirectU", "MatchedDirectU"),
                ):
                    targets = {
                        q: [target for target, _score in getattr(pools[q], field)]
                        for q in pools
                    }
                    direct_pools = _pool_target_view(
                        pools,
                        target_lists=targets,
                        generator=f"{generator}.{direct_name}",
                        keep_bags=False,
                    )
                    direct_dir = directory / "direct_baselines" / direct_name
                    direct_matrix = evaluate_teacher_matrix(
                        {"TB_QT": teachers["TB_QT"], "TB_CQET": teachers["TB_CQET"]},
                        rt.bank,
                        direct_pools,
                        rt.labels,
                        seed=seed,
                        generator=f"{generator}.{direct_name}",
                        split=split,
                        output_dir=direct_dir,
                        pool_kind=direct_name,
                        direct_only=True,
                        split_gt=gt,
                    )
                    direct_metrics = evaluate_matrix(direct_pools, direct_matrix, gt, direct_dir)
                    _independent_verify(rt, split, direct_dir, direct_metrics)
                    direct_results[direct_name] = direct_metrics
                split_result[generator]["direct_baselines"] = direct_results

        main_pools = pools_by_generator["native_kd"]
        admission_order = {
            q: sorted(
                main_pools[q].admission_scores,
                key=lambda target: (
                    -main_pools[q].admission_scores[target], target.encode("utf-8")
                ),
            )
            for q in main_pools
        }
        curve_results = {}
        for budget in (100, 200):
            targets = {q: admission_order[q][:budget] for q in admission_order}
            curve_pools = _pool_target_view(
                main_pools,
                target_lists=targets,
                generator=f"native_kd.C{budget}",
                keep_bags=True,
            )
            curve_dir = split_root / "native_kd" / "budget_curves" / f"C{budget}"
            curve_matrix = evaluate_teacher_matrix(
                {"TB_CQET": teachers["TB_CQET"]}, rt.bank, curve_pools, rt.labels,
                seed=seed, generator=f"native_kd.C{budget}", split=split,
                output_dir=curve_dir, pool_kind=f"C{budget}", split_gt=gt,
            )
            curve_metrics = evaluate_matrix(curve_pools, curve_matrix, gt, curve_dir)
            _independent_verify(rt, split, curve_dir, curve_metrics)
            curve_results[f"C{budget}"] = curve_metrics
        split_result["native_kd"]["budget_curves"] = curve_results

        if split == "dev":
            probe_ids = _hashed_ids(query_ids, 128)
            full_u_targets = {q: main_pools[q].U for q in probe_ids}
            full_u_pools = _pool_target_view(
                {q: main_pools[q] for q in probe_ids},
                target_lists=full_u_targets,
                generator="native_kd.FullU_dev128",
                keep_bags=True,
            )
            full_u_dir = split_root / "trajectories" / "FullU_dev128"
            full_u_matrix = evaluate_teacher_matrix(
                teachers, rt.bank, full_u_pools, rt.labels, seed=seed,
                generator="native_kd.FullU_dev128", split=split,
                output_dir=full_u_dir, pool_kind="FullU", split_gt=gt,
            )
            full_u_metrics = evaluate_matrix(full_u_pools, full_u_matrix, gt, full_u_dir)
            _independent_verify(rt, split, full_u_dir, full_u_metrics, allow_subset=True)
            split_result["native_kd"]["FullU_dev128"] = full_u_metrics

        implicit = [q for q in query_ids if gt[q]["kind"] == "implicit"]
        kd = split_result["native_kd"]["per_query"]
        sup = split_result["native_sup"]["per_query"]
        raw = split_result["raw"]["per_query"]
        contrasts = {
            "content_CQET_Real_minus_Swap_implicit": bootstrap_contrast(
                {q: kd[q]["TB_CQET.Real.R@10"] for q in implicit},
                {q: kd[q]["TB_CQET.Swap.R@10"] for q in implicit}, gt,
            ),
            "CQET_Real_minus_TB_QT_overall": bootstrap_contrast(
                {q: kd[q]["TB_CQET.Real.R@10"] for q in query_ids},
                {q: kd[q]["TB_QT.Direct.R@10"] for q in query_ids}, gt,
            ),
            "CQET_Real_minus_LSE_Real_overall": bootstrap_contrast(
                {q: kd[q]["TB_CQET.Real.R@10"] for q in query_ids},
                {q: kd[q]["TB_LSE.Real.R@10"] for q in query_ids}, gt,
            ),
            "KD_minus_SUP_same_TB_CQET_overall": bootstrap_contrast(
                {q: kd[q]["TB_CQET.Real.R@10"] for q in query_ids},
                {q: sup[q]["TB_CQET.Real.R@10"] for q in query_ids}, gt,
            ),
        }
        bootstrap_dir = split_root / "bootstrap"
        bootstrap_dir.mkdir(exist_ok=True)
        write_json(bootstrap_dir / "results.json", contrasts)
        write_jsonl(
            bootstrap_dir / "group_map.jsonl",
            ({"query_id": q, "source_group": gt[q]["source_group"]} for q in query_ids),
        )
        split_result["contrasts"] = contrasts
        result["splits"][split] = split_result
        write_json(split_root / "SUMMARY.json", split_result)
    return result


def _checkpoint_manifest(seed_dir: Path) -> None:
    rows = []
    for path in sorted(seed_dir.glob("*/attempts/*/checkpoints/*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.append({
            "schema_version": VERSION,
            "path": str(path.relative_to(seed_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "state_sha256": state_sha(payload["model"]),
            "has_optimizer": payload.get("optimizer") is not None,
            "rng_keys": sorted(payload.get("rng", {})),
            "stage": payload.get("extra", {}).get("stage"),
            "parent_state_sha256": payload.get("extra", {}).get("parent_state_sha256"),
        })
    write_jsonl(seed_dir / "checkpoint_manifest.jsonl", rows)


def _reports(rt: Runtime, seed_results: Mapping[int, dict]) -> dict:
    reports = rt.paths.run_root / "reports"
    reports.mkdir(exist_ok=True)
    rows = []
    for seed, result in seed_results.items():
        for split, split_result in result["splits"].items():
            kd_candidate = split_result["native_kd"]["candidate"]["overall"]
            sup_candidate = split_result["native_sup"]["candidate"]["overall"]
            contrasts = split_result["contrasts"]
            rows.append({
                "seed": seed,
                "split": split,
                "candidate_gain_pp": 100 * (
                    kd_candidate["C150_target_coverage"] - kd_candidate["MatchedDirectC_target_coverage"]
                ),
                "content_implicit_pp": contrasts["content_CQET_Real_minus_Swap_implicit"]["mean_delta_pp"],
                "teacher_vs_qt_pp": contrasts["CQET_Real_minus_TB_QT_overall"]["mean_delta_pp"],
                "cqet_vs_lse_pp": contrasts["CQET_Real_minus_LSE_Real_overall"]["mean_delta_pp"],
                "kd_pp": contrasts["KD_minus_SUP_same_TB_CQET_overall"]["mean_delta_pp"],
                "kd_coverage_delta_pp": 100 * (
                    kd_candidate["C150_target_coverage"] - sup_candidate["C150_target_coverage"]
                ),
            })
    dev = [row for row in rows if row["split"] == "dev"]
    test = [row for row in rows if row["split"] == "test"]
    candidate_pass = all(row["candidate_gain_pp"] >= 0 for row in rows) and all(
        np.mean([r["candidate_gain_pp"] for r in rows if r["split"] == split]) >= 1.0
        for split in ("dev", "test")
    )
    content_pass = (
        all(row["content_implicit_pp"] >= 0 for row in rows)
        and np.mean([r["content_implicit_pp"] for r in dev]) >= 0.5
        and np.mean([r["content_implicit_pp"] for r in test]) > 0
        and all(row["teacher_vs_qt_pp"] >= -0.5 for row in rows)
        and np.mean([r["cqet_vs_lse_pp"] for r in dev]) >= 0
        and np.mean([r["cqet_vs_lse_pp"] for r in test]) >= 0
    )
    kd_pass = (
        all(row["kd_pp"] >= 0 for row in rows)
        and np.mean([r["kd_pp"] for r in dev]) >= 0.3
        and np.mean([r["kd_pp"] for r in test]) > 0
        and all(row["kd_coverage_delta_pp"] >= -0.5 for row in rows)
    )
    decision = {
        "schema_version": VERSION,
        "correctness": "PASS",
        "candidate_gain": {"pass": bool(candidate_pass), "rows": rows},
        "teacher_E_content_gain": {"pass": bool(content_pass)},
        "KD_gain": {"pass": bool(kd_pass)},
        "historical_comparability": "NOT_CAUSALLY_COMPARABLE_TO_V4_0_DUE_TO_CORRECTNESS_FIXES",
        "historical_reference": "HISTORICAL_REFERENCE_UNAVAILABLE",
    }
    write_json(reports / "DECISION.json", decision)
    (reports / "DECISION.md").write_text(
        "# Preregistered decision\n\n"
        f"- Correctness: {decision['correctness']}\n"
        f"- Candidate gain: {'PASS' if candidate_pass else 'FAIL'}\n"
        f"- Teacher E-content gain: {'PASS' if content_pass else 'FAIL'}\n"
        f"- C2 KD gain: {'PASS' if kd_pass else 'FAIL'}\n"
        "- Historical comparison: not causally comparable to V4.0.\n",
        encoding="utf-8",
    )
    lines = [
        "# MMDD Stage1 V4.1 results",
        "",
        f"- Correctness: {decision['correctness']}",
        f"- Candidate gain gate: {'PASS' if candidate_pass else 'FAIL'}",
        f"- Teacher E-content gate: {'PASS' if content_pass else 'FAIL'}",
        f"- KD gate: {'PASS' if kd_pass else 'FAIL'}",
        "- Historical comparison: V4.0 is not a single-factor causal baseline because V4.1 repairs multiple execution semantics.",
        "",
        "If only the candidate gate passes, the supported conclusion is candidate expansion only; the Path-content and KD claims are not retained.",
        "",
        "| seed | split | candidate pp | E-content pp | Teacher-vs-QT pp | CQET-vs-LSE pp | KD pp | KD coverage pp |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['split']} | {row['candidate_gain_pp']:.4f} | "
            f"{row['content_implicit_pp']:.4f} | {row['teacher_vs_qt_pp']:.4f} | "
            f"{row['cqet_vs_lse_pp']:.4f} | {row['kd_pp']:.4f} | {row['kd_coverage_delta_pp']:.4f} |"
        )
    (reports / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "AUDIT_LIMITATIONS.md").write_text(
        "# Audit limitations\n\nThe test split is a historically exposed regression set, not an unseen holdout. "
        "No compatible predeclared historical-reference manifest was available.\n",
        encoding="utf-8",
    )
    return decision


def _manifest_role(relative: Path) -> str:
    value = relative.as_posix()
    if "checkpoints/" in value and value.endswith(".pt"):
        return "full_training_checkpoint"
    if value.startswith("src_snapshots/"):
        return "immutable_source_snapshot"
    if value.startswith("eval_labels/") or "/rankings." in value or "/logits." in value:
        return "independent_recompute_input"
    if "training_records/" in value or value.startswith("labels/"):
        return "materialized_training_input"
    if value.startswith("reports/"):
        return "report"
    if value.startswith("tests/"):
        return "validation"
    return "run_artifact"


def _write_file_manifest(run_root: Path) -> None:
    rows = []
    for path in sorted(run_root.rglob("*"), key=lambda item: str(item).encode("utf-8")):
        if not path.is_file() or path.name == "FILE_MANIFEST.jsonl" or ".tmp." in path.name:
            continue
        relative = path.relative_to(run_root)
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "role": _manifest_role(relative),
        })
    write_jsonl(run_root / "FILE_MANIFEST.jsonl", rows)


def _write_delivery_readme(rt: Runtime, decision: dict, phase_status: str) -> None:
    candidate = decision["candidate_gain"]["pass"]
    content = decision["teacher_E_content_gain"]["pass"]
    kd = decision["KD_gain"]["pass"]
    seeds = tuple(rt.protocol.get("seeds", [13]))
    text = (
        "# MMDD Stage1 V4.1 delivery\n\n"
        f"Status: `{phase_status}`. Seeds {list(seeds)} ({len(seeds) * 9} registered stages) completed on the locked GPU.\n\n"
        "All registered checkpoints are full checkpoints containing model, optimizer, RNG, cursor, order, "
        "parent, source, protocol, and feature identities. Test evaluation began only after "
        "`GLOBAL_SELECTION_FREEZE.json` was written. Metrics were recomputed from exported qrels and raw "
        "rankings by the independent audit tool.\n\n"
        f"- Correctness: {decision['correctness']}\n"
        f"- Candidate gain gate: {'PASS' if candidate else 'FAIL'}\n"
        f"- Teacher E-content gate: {'PASS' if content else 'FAIL'}\n"
        f"- C2 KD gate: {'PASS' if kd else 'FAIL'}\n"
        "- Historical comparison: V4.0 is not a causal single-factor baseline; no historical checkpoint "
        "was used to fill or train this run.\n\n"
        "Pure frozen feature bytes are referenced by the locked recipe and complete shard manifest rather "
        "than duplicated into this run directory. The test split is historically exposed and is reported "
        "as a regression set, not an unseen holdout.\n"
    )
    (rt.paths.run_root / "README.md").write_text(text, encoding="utf-8")


def _verify_delivery_manifest(rt: Runtime) -> None:
    verifier = (
        rt.paths.repo_root / "audit" / "MMDD_S1_V4_AUDIT_AND_V4_1_PACKAGE"
        / "tools" / "verify_file_manifest.py"
    )
    subprocess.run(
        [sys.executable, str(verifier), "--root", str(rt.paths.run_root),
         "--manifest", str(rt.paths.run_root / "FILE_MANIFEST.jsonl"), "--check-stages"],
        cwd="/tmp", check=True, stdout=subprocess.DEVNULL,
    )


def run_formal(protocol_path: Path, run_root: Path) -> None:
    rt = load_runtime(protocol_path, run_root)
    _gpu_guard(rt.paths)
    assert_declared_project_imports(rt.paths)
    smoke_result = read_json(rt.paths.run_root / "tests" / "smoke" / "LATEST.json")
    if smoke_result.get("status") != "PASS" or smoke_result["source_identity_sha256"] != source_identity(rt.paths):
        raise RuntimeError("current source has not passed the restricted GPU smoke")
    seeds = tuple(rt.protocol.get("seeds", [13]))
    _phase(rt.paths, "FORMAL_TRAINING", formal_training_started=True,
           detail={"seed_order": list(seeds)})
    raw_et_path = rt.paths.run_root / "training_shared" / "RawET128.pt"
    raw_et_path.parent.mkdir(exist_ok=True)
    if raw_et_path.exists():
        raw_et = torch.load(raw_et_path, map_location="cpu", weights_only=False)
    else:
        raw_et = build_raw_et128_exact(rt.z_store, rt.labels)
        torch.save(raw_et, raw_et_path)
    freezes = {}
    for seed in seeds:
        freezes[seed] = train_seed(rt, seed, raw_et)
        _checkpoint_manifest(rt.paths.seed_dir(seed))
    global_freeze = {
        "schema_version": VERSION,
        "status": "ALL_SELECTIONS_FROZEN_BEFORE_TEST",
        "seeds": {str(seed): freezes[seed]["freeze_sha256"] for seed in seeds},
        "test_qrels_read": False,
    }
    global_freeze["freeze_sha256"] = json_identity(global_freeze)
    write_json(rt.paths.run_root / "GLOBAL_SELECTION_FREEZE.json", global_freeze)
    _phase(rt.paths, "FROZEN_EVALUATION", formal_training_started=True,
           detail={"global_freeze_sha256": global_freeze["freeze_sha256"]})
    seed_results = {
        seed: evaluate_seed(rt, seed, global_freeze["freeze_sha256"])
        for seed in seeds
    }
    for seed, result in seed_results.items():
        write_json(rt.paths.seed_dir(seed) / "FINAL_EVALUATION.json", result)
        _checkpoint_manifest(rt.paths.seed_dir(seed))
    decision = _reports(rt, seed_results)
    performance_pass = all((
        decision["candidate_gain"]["pass"],
        decision["teacher_E_content_gain"]["pass"],
        decision["KD_gain"]["pass"],
    ))
    phase_status = "COMPLETE_PASS" if performance_pass else "COMPLETE_PERFORMANCE_GATES_FAILED"
    _phase(rt.paths, phase_status, formal_training_started=True,
           detail={"seeds_completed": list(seeds), "test_evaluated_after_global_freeze": True,
                   "delivery_success": performance_pass})
    _write_delivery_readme(rt, decision, phase_status)
    _write_file_manifest(rt.paths.run_root)
    try:
        _verify_delivery_manifest(rt)
    except Exception:
        _phase(rt.paths, "COMPLETE_DELIVERY_INTEGRITY_FAILED", formal_training_started=True,
               detail={"seeds_completed": list(seeds), "test_evaluated_after_global_freeze": True,
                       "delivery_success": False})
        _write_file_manifest(rt.paths.run_root)
        raise

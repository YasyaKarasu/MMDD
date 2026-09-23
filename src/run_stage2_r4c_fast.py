#!/usr/bin/env python3
"""Execute the dev-only, label-blind MMDD S2-R4c FAST visual pilot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from mmdd_stage2.r4_common import digest, file_hash, read_jsonl, write_json
from mmdd_stage2.r4c_fast_metrics import component_summary, decide_gate, score_records, summarize_arm
from mmdd_stage2.r4c_fast_recovery import ARMS, FastCropRecoveryBackend, generation_input_hash
from mmdd_stage2.r4c_fast_report import write_pilot_report
from mmdd_stage2.r4c_fast_sampling import (
    lock_natural,
    smoke_units,
    unit_shard,
    write_deterministic_jsonl_gz,
)
from mmdd_stage2.r4c_fast_types import FastUnit
from mmdd_stage2.r4c_fast_views import make_evidence_views, save_crop

REPO = Path("/home/oycy/MMDD")
DEFAULT_MODEL = REPO / "hf_models/Qwen3.5-9B"
DEFAULT_R3 = REPO / "work/S2_COL_R3_ROW/ARTIFACTS"
DEFAULT_R4 = REPO / "work/S2_R4"
DEFAULT_OUT = REPO / "work/S2_R4C_FAST"
SOURCE_FILES = (
    "src/mmdd_stage2/r4c_fast_types.py",
    "src/mmdd_stage2/r4c_fast_sampling.py",
    "src/mmdd_stage2/r4c_fast_localizer.py",
    "src/mmdd_stage2/r4c_fast_views.py",
    "src/mmdd_stage2/r4c_fast_recovery.py",
    "src/mmdd_stage2/r4c_fast_metrics.py",
    "src/mmdd_stage2/r4c_fast_report.py",
    "src/run_stage2_r4c_fast.py",
)


def _objects(r4: Path) -> dict[str, Any]:
    with gzip.open(r4 / "READER_OBJECTS.dev.jsonl.gz", "rt", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _source_hash() -> str:
    return digest({name: file_hash(REPO / name) for name in SOURCE_FILES})


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "NOT_A_GIT_REPOSITORY"


def _required_paths(r4: Path, r3: Path, model: Path) -> list[Path]:
    return [
        model / "config.json",
        r3 / "ROW_GT.dev.jsonl",
        r4 / "CANDIDATE_LOCK.dev.jsonl.gz",
        r4 / "READER_OBJECTS.dev.jsonl.gz",
        r4 / "RECOVERY_SCHEDULES/S1_JointGlobal.dev.json",
    ]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    paths = _required_paths(args.r4, args.r3, args.model)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("BLOCKED_MISSING_REQUIRED_INPUT:" + ",".join(missing))
    from transformers import AutoConfig, AutoProcessor

    AutoConfig.from_pretrained(args.model, local_files_only=True)
    AutoProcessor.from_pretrained(args.model, local_files_only=True)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"BLOCKED_GPU_COUNT:{torch.cuda.device_count()}")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if gpu_names != ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4090"]:
        raise RuntimeError(f"BLOCKED_GPU_MODEL:{gpu_names}")
    if str(args.device) != "cuda:1":
        raise RuntimeError("BLOCKED_DEVICE_CONTRACT:standalone requires cuda:1")

    objects = _objects(args.r4)
    schedule = json.loads((args.r4 / "RECOVERY_SCHEDULES/S1_JointGlobal.dev.json").read_text())
    query_ids = {record["query_id"] for record in schedule["records"]}
    lock_by_query = {
        row["query_id"]: row for row in read_jsonl(args.r4 / "CANDIDATE_LOCK.dev.jsonl.gz")
    }
    referenced_images = []
    for query_id in query_ids:
        for target in lock_by_query[query_id]["targets"]:
            for asset_id in target["retained_evidence_ids"]:
                item = objects["evidence"][asset_id]
                if item.get("asset_type") == "image":
                    referenced_images.append(asset_id)
    path_failures = []
    for asset_id in sorted(set(referenced_images)):
        path = Path(str(objects["evidence"][asset_id].get("local_path", "")))
        if not path.is_file():
            path_failures.append({"asset_id": asset_id, "path": str(path)})
    if path_failures:
        raise RuntimeError(f"BLOCKED_IMAGE_PATH:{len(path_failures)} schedule images")

    manifest = {
        "status": "PASS",
        "split": "dev",
        "test_opened": False,
        "git_commit": _git_commit(),
        "source_code_hash": _source_hash(),
        "device": str(args.device),
        "gpu_names": gpu_names,
        "files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_hash(path)}
            for path in paths
        ],
        "schedule_image_assets": len(set(referenced_images)),
        "schedule_image_path_failures": 0,
        "note": "Preflight resolves every scheduled image local_path; candidate construction separately decodes every eligible focus image.",
    }
    write_json(args.out / "INPUT_MANIFEST.json", manifest)
    write_json(args.out / "PREFLIGHT.json", manifest)
    return manifest


def _load_units(path: Path) -> list[FastUnit]:
    return [FastUnit.from_dict(row) for row in read_jsonl(path)]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _unit_record_base(unit: FastUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "query_id": unit.query_id,
        "target_id": unit.target_id,
        "column_id": unit.column_id,
        "column_name": unit.column_name,
        "query_row_id": unit.query_row_id,
        "source_group": unit.source_group,
        "evidence_ids_natural": list(unit.evidence_ids),
        "focus_image_id": unit.focus_image_id,
    }


def _localizer_key(
    backend: FastCropRecoveryBackend, unit: FastUnit, evidence: dict[str, Any]
) -> str:
    layers = backend.resolve_layers()
    return digest({
        "model_config_hash": backend.model_config_hash,
        "source_code_hash": _source_hash(),
        "row": [list(cell) for cell in unit.cells],
        "column_name": unit.column_name,
        "focus_image_file_sha256": file_hash(Path(str(evidence["local_path"]))),
        "resolved_layers": list(layers),
        "localizer_formula_version": backend.formula_version,
    })


def _run_localizer(
    backend: FastCropRecoveryBackend,
    unit: FastUnit,
    objects: dict[str, Any],
    out: Path,
    *,
    save_heatmaps: bool,
) -> dict[str, Any]:
    evidence = objects["evidence"][unit.focus_image_id]
    key = _localizer_key(backend, unit, evidence)
    cache_path = out / "CACHE/LOCALIZER" / f"{key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if all(
            result.get("crop_fallback")
            or Path(str(result.get("crop_path", ""))).is_file()
            for result in cached["crops"].values()
        ):
            return cached
    forward = backend.localizer_forward(unit.row_dict(), unit.column_name, evidence)
    maps = backend.build_maps(forward)
    crops = {}
    source_path = Path(str(evidence["local_path"]))
    for arm in ("V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"):
        result = backend.crop_for_arm(maps, arm, forward.image_size)
        if not result["crop_fallback"]:
            crop_path, crop_sha = save_crop(
                source_path, tuple(result["pixel_box"]), out / "CROPS"
            )
            result.update({"crop_path": str(crop_path), "crop_view_sha256": crop_sha})
        crops[arm] = result
    value = {
        "localizer_cache_key": key,
        "formula_version": backend.formula_version,
        "metadata": maps.metadata,
        "crops": crops,
    }
    if save_heatmaps:
        path = out / "SMOKE_HEATMAPS" / f"{unit.unit_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "row": maps.row,
            "attribute_maps": maps.attribute_maps,
            "raea": maps.raea,
            "consensus": maps.consensus,
            "joint_raea": maps.joint_raea,
            "joint_consensus": maps.joint_consensus,
        }, path)
        value["smoke_heatmap_path"] = str(path)
    _atomic_json(cache_path, value)
    return value


def _error_record(
    unit: FastUnit, arm: str, status: str, reason: str, localizer: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        **_unit_record_base(unit),
        "arm": arm,
        "generation_input_hash": None,
        "localizer_cache_key": (localizer or {}).get("localizer_cache_key"),
        "crop_view_sha256": None,
        "crop_box": None,
        "crop_fallback": False,
        "status": status,
        "value": None,
        "evidence_ids": [],
        "parse_error": reason,
        "finish_reason": "not_run",
        "elapsed_localizer_seconds": float((localizer or {}).get("metadata", {}).get("elapsed_localizer_seconds", 0.0)),
        "elapsed_generation_seconds": 0.0,
        "peak_gpu_memory_bytes": 0,
        "views": [],
    }


def _run_arm(
    backend: FastCropRecoveryBackend,
    unit: FastUnit,
    objects: dict[str, Any],
    arm: str,
    out: Path,
    localizer: dict[str, Any] | None,
) -> dict[str, Any]:
    crop = localizer["crops"][arm] if localizer is not None and arm in localizer["crops"] else None
    sources = make_evidence_views(unit, objects, arm, crop)
    generation_hash = generation_input_hash(
        unit, arm, sources, model_config_hash=backend.model_config_hash
    )
    generation_cache = out / "CACHE/GENERATION" / f"{generation_hash}.json"
    if generation_cache.is_file():
        output = json.loads(generation_cache.read_text())
    else:
        output = backend.recover(unit, sources, arm)
        _atomic_json(generation_cache, output)
    record = {
        **_unit_record_base(unit),
        **output,
        "arm": arm,
        "generation_input_hash": generation_hash,
        "localizer_cache_key": localizer.get("localizer_cache_key") if localizer else None,
        "crop_view_sha256": crop.get("crop_view_sha256") if crop else None,
        "crop_box": crop.get("pixel_box") if crop else None,
        "crop_fallback": bool(crop and crop.get("crop_fallback")),
        "crop_fallback_reason": crop.get("fallback_reason") if crop else None,
        "crop_area_ratio": crop.get("area_ratio") if crop else None,
        "elapsed_localizer_seconds": (
            float(localizer["metadata"]["elapsed_localizer_seconds"])
            if localizer is not None else 0.0
        ),
    }
    record["peak_gpu_memory_bytes"] = max(
        int(record.get("peak_gpu_memory_bytes", 0)),
        int((localizer or {}).get("metadata", {}).get("peak_localizer_gpu_memory_bytes", 0)),
    )
    return record


def _process_unit(
    backend: FastCropRecoveryBackend,
    unit: FastUnit,
    objects: dict[str, Any],
    arms: tuple[str, ...],
    out: Path,
    *,
    smoke: bool,
) -> list[dict[str, Any]]:
    raw_folder = out / "CACHE/RAW" / unit.unit_id
    raw_folder.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    needs_localizer = any(arm in {"V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"} for arm in arms)
    localizer = None
    localizer_error = None

    def run_one(arm: str) -> None:
        raw_path = raw_folder / f"{arm}.json"
        if raw_path.is_file():
            records.append(json.loads(raw_path.read_text()))
            return
        if arm in {"V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"} and localizer_error:
            record = _error_record(unit, arm, "LOCALIZER_ERROR", localizer_error)
        else:
            try:
                record = _run_arm(backend, unit, objects, arm, out, localizer)
            except torch.OutOfMemoryError as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                record = _error_record(unit, arm, "GENERATION_OOM", str(error), localizer)
            except Exception as error:
                record = _error_record(
                    unit, arm, "GENERATION_ERROR", f"{type(error).__name__}:{error}", localizer
                )
        _atomic_json(raw_path, record)
        records.append(record)

    for arm in arms:
        if arm in {"V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"}:
            continue
        run_one(arm)
    if needs_localizer:
        try:
            localizer = _run_localizer(backend, unit, objects, out, save_heatmaps=smoke)
        except Exception as error:  # Recorded as a paired crop failure; V0/V1 may still run.
            localizer_error = f"{type(error).__name__}:{error}"
    for arm in arms:
        if arm in {"V2_RAEA_DUAL", "V3_CONSENSUS_DUAL"}:
            run_one(arm)
    return records


def _consolidate(out: Path, units: list[FastUnit], arms: tuple[str, ...], name: str) -> Path:
    records = []
    for unit in units:
        for arm in arms:
            path = out / "CACHE/RAW" / unit.unit_id / f"{arm}.json"
            if path.is_file():
                records.append(json.loads(path.read_text()))
    records.sort(key=lambda row: (row["unit_id"], row["arm"]))
    keys = [(row["unit_id"], row["arm"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate unit/arm raw records")
    path = out / "RAW" / name
    write_deterministic_jsonl_gz(path, records)
    return path


def _backend(args: argparse.Namespace) -> FastCropRecoveryBackend:
    backend = FastCropRecoveryBackend(
        args.model, device=args.device, dtype="bf16", max_new_tokens=256
    )
    backend.resolve_layers()
    write_json(args.out / "LOCALIZER_CONFIG.json", backend.localizer_config)
    return backend


def run_pilot(args: argparse.Namespace, *, smoke_only: bool) -> dict[str, Any]:
    lock_receipt = json.loads((args.out / "PILOT_LOCK_RECEIPT.json").read_text())
    if lock_receipt["status"] != "LOCKED":
        raise RuntimeError("BLOCKED_TOO_FEW_LABEL_BLIND_UNITS")
    units = _load_units(args.out / "PILOT_LOCK.dev.jsonl.gz")
    if file_hash(args.out / "PILOT_LOCK.dev.jsonl.gz") != lock_receipt["lock_sha256"]:
        raise RuntimeError("BLOCKED_LOCK_HASH_MISMATCH")
    selected = smoke_units(units) if smoke_only else units
    if smoke_only and (len({unit.source_group for unit in selected}) != 8 or len(selected) > 16):
        raise RuntimeError("BLOCKED_SMOKE_SAMPLE_CONTRACT")
    selected = [unit for unit in selected if unit_shard(unit.unit_id, args.shards) == args.shard]
    if args.shards != 1 or args.shard != 0:
        raise RuntimeError("BLOCKED_DEVICE_CONTRACT:standalone uses one GPU1 process")
    backend = _backend(args)
    objects = _objects(args.r4)
    for position, unit in enumerate(selected, 1):
        _process_unit(backend, unit, objects, ARMS, args.out, smoke=smoke_only)
        print(f"{'smoke' if smoke_only else 'pilot'} {position}/{len(selected)} {unit.unit_id[:12]}", flush=True)
    path = _consolidate(args.out, selected, ARMS, "smoke.jsonl.gz" if smoke_only else "pilot.jsonl.gz")
    receipt = {
        "status": "executed",
        "stage": "FAST-SMOKE" if smoke_only else "FAST-PILOT",
        "units": len(selected),
        "arms": list(ARMS),
        "raw_path": str(path),
        "device": args.device,
        "source_groups": len({unit.source_group for unit in selected}),
        "lock_sha256": lock_receipt["lock_sha256"],
    }
    if smoke_only:
        rows = read_jsonl(path)
        by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
        checks = {
            "four_arms_all_units": all(len(by_arm[arm]) == len(selected) for arm in ARMS),
            "v0_v1_parse_at_least_7": all(
                sum(row["status"] not in {"PARSE_ERROR", "GENERATION_ERROR", "GENERATION_OOM"} for row in by_arm[arm]) >= 7
                for arm in ("V0_FULL_BASE", "V1_FULL_HIGHRES")
            ),
            "v2_v3_localizer_at_least_7": all(
                sum(row["status"] != "LOCALIZER_ERROR" for row in by_arm[arm]) >= 7
                for arm in ("V2_RAEA_DUAL", "V3_CONSENSUS_DUAL")
            ),
            "no_crop_evidence_ids": all(
                all("crop" not in label.casefold() and "view" not in label.casefold() for label in row.get("evidence_ids", []))
                for row in rows
            ),
            "all_generation_hashes_present": all(
                row.get("generation_input_hash") is not None
                for row in rows if row["status"] != "LOCALIZER_ERROR"
            ),
            "heatmaps_saved": all((args.out / "SMOKE_HEATMAPS" / f"{unit.unit_id}.pt").is_file() for unit in selected),
        }
        receipt["checks"] = checks
        receipt["status"] = "PASS" if all(checks.values()) else "FAIL"
        write_json(args.out / "FAST_SMOKE_RECEIPT.json", receipt)
    else:
        write_json(args.out / "FAST_PILOT_RECEIPT.json", receipt)
    return receipt


def score_pilot(args: argparse.Namespace, *, stopped_early: bool = False) -> dict[str, Any]:
    units = _load_units(args.out / "PILOT_LOCK.dev.jsonl.gz")
    if stopped_early:
        complete = [unit for unit in units if all(
            (args.out / "CACHE/RAW" / unit.unit_id / f"{arm}.json").is_file()
            for arm in ARMS
        )]
        raw = read_jsonl(_consolidate(args.out, complete, ARMS, "pilot.partial.jsonl.gz"))
    else:
        raw = read_jsonl(args.out / "RAW/pilot.jsonl.gz")
    if not stopped_early and len(raw) != len(units) * len(ARMS):
        raise RuntimeError("BLOCKED_INCOMPLETE_PILOT")
    gt_index: dict[tuple[str, str, int, int], dict[str, set[Any]]] = {}
    for row in read_jsonl(args.r3 / "ROW_GT.dev.jsonl"):
        key = (str(row["query_id"]), str(row["target_id"]), int(row["query_row_id"]), int(row["local_column_index"]))
        entry = gt_index.setdefault(key, {"gold": set(), "rows": set(), "witness": set()})
        entry["gold"].add(str(row["gold_value_raw"]))
        entry["rows"].update(int(item) for item in row.get("target_row_ids", []))
        entry["witness"].update(str(item) for item in row.get("witness_evidence_ids", []))
    evaluation = []
    for unit in units:
        entry = gt_index.get((unit.query_id, unit.target_id, unit.query_row_id, unit.column_id))
        if entry:
            evaluation.append({
                "unit_id": unit.unit_id,
                "gold_values": sorted(entry["gold"]),
                "gold_target_row_ids": sorted(entry["rows"]),
                "witness_evidence_ids": sorted(entry["witness"]),
            })
    write_deterministic_jsonl_gz(args.out / "PILOT_EVAL_SIDECAR.dev.jsonl.gz", evaluation)
    scored = score_records(raw, evaluation)
    write_deterministic_jsonl_gz(args.out / "SCORED/pilot.jsonl.gz", scored)
    summary = component_summary(scored)
    summary["all_units"] = len(units)
    summary["evaluable_units"] = len(evaluation)
    summary["executed_units"] = len({row["unit_id"] for row in raw})
    summary["evaluable_executed_units"] = len({row["unit_id"] for row in scored})
    summary["status"] = "stopped_insufficient_gt" if stopped_early else "executed"
    summary["all_source_groups"] = len({unit.source_group for unit in units})
    runtime_rows = score_records(raw, [
        {"unit_id": unit.unit_id, "gold_values": [], "witness_evidence_ids": []}
        for unit in units
    ])
    runtime_fields = {
        "units", "median_total_elapsed_seconds", "p95_total_elapsed_seconds",
        "median_input_pixels", "median_image_tokens", "median_prompt_tokens",
        "median_peak_gpu_memory_bytes", "crop_fallback",
    }
    summary["all_units_runtime"] = {
        arm: {key: value for key, value in summarize_arm(
            [row for row in runtime_rows if row["arm"] == arm]
        ).items() if key in runtime_fields}
        for arm in ARMS
    }
    summary["raw_status_counts"] = {
        arm: {status: sum(row["arm"] == arm and row["status"] == status for row in raw)
              for status in sorted({row["status"] for row in raw})}
        for arm in ARMS
    }
    write_json(args.out / "PILOT_SCORE.json", summary)
    return summary


def gate(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads((args.out / "PILOT_SCORE.json").read_text())
    decision = decide_gate(summary)
    write_json(args.out / "GATE_DECISION.json", decision)
    return decision


def report(args: argparse.Namespace, blocker: str | None = None) -> None:
    def load_json(name: str) -> dict[str, Any] | None:
        path = args.out / name
        return json.loads(path.read_text()) if path.is_file() else None

    pre = load_json("PREFLIGHT.json") or {"status": "planned"}
    lock = load_json("PILOT_LOCK_RECEIPT.json") or {"status": "planned"}
    smoke = load_json("FAST_SMOKE_RECEIPT.json")
    summary = load_json("PILOT_SCORE.json")
    decision = load_json("GATE_DECISION.json")
    write_pilot_report(
        args.out / "PILOT_REPORT.zh-CN.md",
        preflight=pre,
        lock=lock,
        smoke=smoke,
        summary=summary,
        gate=decision,
        blocker=blocker,
    )


def write_receipt(args: argparse.Namespace, statuses: dict[str, Any]) -> None:
    write_json(args.out / "IMPLEMENTATION_RECEIPT.json", {
        "date": "2026-09-23",
        "source_code_hash": _source_hash(),
        "git_commit": _git_commit(),
        "files": list(SOURCE_FILES),
        "tests": statuses.get("tests", "not_recorded"),
        "stages": statuses,
        "scope": {"split": "dev", "full_dev": False, "test": "planned_after_all_dev_selection"},
    })


def blocker_report(args: argparse.Namespace, error: BaseException) -> None:
    text = (
        "# S2-R4c FAST Blocker Report\n\n"
        f"日期：2026-09-23\n\n"
        f"状态：BLOCKED\n\n"
        f"失败阶段：`{args.phase}`\n\n"
        f"异常：`{type(error).__name__}: {error}`\n\n"
        "## 真实调用与环境\n\n"
        f"- device: `{args.device}`\n"
        f"- model: `{args.model}`\n"
        f"- torch.cuda.is_available: `{torch.cuda.is_available()}`\n"
        f"- torch.cuda.device_count: `{torch.cuda.device_count()}`\n\n"
        "## Traceback\n\n```text\n"
        + traceback.format_exc()
        + "\n```\n\n"
        "## 最小修复建议\n\n"
        "保持 Adapt-v1、prompt、四臂和 gate 不变；只修复上面失败的实际 API/shape 合同后，从当前缓存继续。"
        "不得替换模型、层规则、阈值或输入 artifact。\n"
    )
    (args.out / "BLOCKER_REPORT.md").write_text(text, encoding="utf-8")
    report(args, blocker=f"`{type(error).__name__}: {error}`。详见 `BLOCKER_REPORT.md`。")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("phase", choices=(
        "preflight", "lock-pilot", "smoke", "pilot", "score-pilot", "finalize-stopped", "gate", "report",
    ))
    result.add_argument("--out", type=Path, default=DEFAULT_OUT)
    result.add_argument("--r4", type=Path, default=DEFAULT_R4)
    result.add_argument("--r3", type=Path, default=DEFAULT_R3)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--device", default="cuda:1")
    result.add_argument("--shard", type=int, default=0)
    result.add_argument("--shards", type=int, default=1)
    return result


def main() -> None:
    args = parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "preflight":
            result = preflight(args)
        elif args.phase == "lock-pilot":
            result = lock_natural(args.r4, args.out)
            write_json(args.out / "PILOT_LOCK_RECEIPT.json", result)
        elif args.phase == "smoke":
            result = run_pilot(args, smoke_only=True)
        elif args.phase == "pilot":
            smoke = json.loads((args.out / "FAST_SMOKE_RECEIPT.json").read_text())
            if smoke["status"] != "PASS":
                raise RuntimeError("BLOCKED_SMOKE_FAILED")
            result = run_pilot(args, smoke_only=False)
        elif args.phase == "score-pilot":
            result = score_pilot(args)
        elif args.phase == "finalize-stopped":
            result = score_pilot(args, stopped_early=True)
        elif args.phase == "gate":
            result = gate(args)
        else:
            report(args)
            result = {"status": "written", "path": str(args.out / "PILOT_REPORT.zh-CN.md")}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        blocker_report(args, error)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()

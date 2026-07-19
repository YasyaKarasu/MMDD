#!/usr/bin/env python
"""Validate immutable WDC URL ETA gate evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from build_wdc200k_mm_joinability_dataset import PipelineConfig, ProgressReporter
from wdc200k_eta import URL_TELEMETRY_SCHEMA_VERSION


@dataclass(frozen=True)
class RestoredProgress:
    path: Path
    sha256: str
    stage_telemetry: dict[str, dict[str, Any]]


class GateValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def restore_progress_read_only(path: Path) -> RestoredProgress:
    """Strictly restore progress from an isolated copy of captured bytes."""
    try:
        resolved = Path(path).resolve()
    except OSError as error:
        raise GateValidationError(
            "INPUT_READ_FAILED", "unable to resolve progress input"
        ) from error
    if not resolved.exists():
        raise GateValidationError("INPUT_NOT_FOUND", "progress input not found")
    if not resolved.is_file():
        raise GateValidationError("INPUT_NOT_FILE", "progress input is not a file")
    try:
        captured = resolved.read_bytes()
    except OSError as error:
        raise GateValidationError(
            "INPUT_READ_FAILED", "unable to read progress input"
        ) from error

    digest = hashlib.sha256(captured).hexdigest()
    try:
        with TemporaryDirectory(prefix="wdc-eta-gate-") as temporary:
            root = Path(temporary)
            work_dir = root / "work"
            work_dir.mkdir()
            (work_dir / "progress.json").write_bytes(captured)
            config = PipelineConfig(
                input_dir=root / "input",
                output_dir=root / "output",
                work_dir=work_dir,
                cache_dir=root / "cache",
                resume=True,
            )
            reporter = ProgressReporter(config)
            telemetry = json.loads(
                json.dumps(
                    reporter._stage_telemetry,
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
    except (OSError, TypeError, ValueError) as error:
        raise GateValidationError(
            "STRICT_RESTORE_FAILED", "progress strict restore failed"
        ) from error
    return RestoredProgress(
        path=resolved,
        sha256=digest,
        stage_telemetry=telemetry,
    )


def _require_native_v2(
    restored: RestoredProgress,
    *,
    require_complete: bool,
) -> None:
    required_stages = ("pages", "images")
    if require_complete:
        for stage in required_stages:
            if stage not in restored.stage_telemetry:
                raise GateValidationError(
                    "MISSING_URL_STAGE", f"{stage} stage is missing"
                )

    for stage, telemetry in restored.stage_telemetry.items():
        if telemetry.get("telemetry_schema_version") != (
            URL_TELEMETRY_SCHEMA_VERSION
        ) or any(
            sample.get("telemetry_schema_version")
            != URL_TELEMETRY_SCHEMA_VERSION
            for sample in telemetry.get("samples", [])
        ):
            raise GateValidationError(
                "NON_NATIVE_V2", f"{stage} stage is not native v2"
            )
        if require_complete and stage in required_stages and (
            telemetry.get("completed_at") is None
            or telemetry.get("completed_units")
            != telemetry.get("total_units")
        ):
            raise GateValidationError(
                "INCOMPLETE_URL_STAGE", f"{stage} stage is incomplete"
            )


def _input_identity(restored: RestoredProgress) -> dict[str, str]:
    return {"path": str(restored.path), "sha256": restored.sha256}


def _enumerate_stage(
    stage: str,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    completed_at = float(telemetry["completed_at"])
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for position, sample in enumerate(telemetry["samples"]):
        if sample["completed_units"] * 2 < sample["total_units"]:
            continue
        actual = completed_at - float(sample["timestamp"])
        predicted = sample["predicted_remaining_seconds"]
        reason = None
        if predicted is None:
            reason = "PREDICTION_MISSING"
        elif float(predicted) <= 0.0:
            reason = "PREDICTION_NON_POSITIVE"
        elif actual <= 0.0:
            reason = "ACTUAL_REMAINING_NON_POSITIVE"

        record = {
            **sample,
            "stage": stage,
            "sample_position": position,
            "actual_remaining_seconds": actual,
        }
        if reason is not None:
            excluded.append(
                {
                    **record,
                    "symmetric_factor": None,
                    "exclusion_reason": reason,
                }
            )
            continue
        predicted_value = float(predicted)
        factor = max(
            predicted_value / actual,
            actual / predicted_value,
        )
        eligible.append({**record, "symmetric_factor": factor})

    eligible.sort(
        key=lambda row: (-row["symmetric_factor"], row["sample_position"])
    )
    maximum = eligible[0]["symmetric_factor"] if eligible else None
    derived = (
        len(eligible),
        len(excluded),
        maximum,
    )
    restored = (
        telemetry["eligible_final_half_samples"],
        telemetry["excluded_final_half_samples"],
        telemetry["max_symmetric_eta_factor"],
    )
    if derived != restored:
        raise GateValidationError(
            "STRICT_RESTORE_FAILED",
            f"{stage} ETA summary differs from sample enumeration",
        )
    return {
        "telemetry_schema_version": telemetry["telemetry_schema_version"],
        "completed_at": telemetry["completed_at"],
        "eligible_final_half_samples": len(eligible),
        "excluded_final_half_samples": len(excluded),
        "max_symmetric_eta_factor": maximum,
        "eligible_samples": eligible,
        "excluded_samples": excluded,
        "worst_record": eligible[0] if eligible else None,
        "worst_reason": None if eligible else "ZERO_ELIGIBLE_SAMPLES",
        "resume_prefix": None,
    }


def validate_eta_gate(
    progress_path: Path,
    before_resume_path: Path | None = None,
) -> dict[str, Any]:
    progress = restore_progress_read_only(progress_path)
    _require_native_v2(progress, require_complete=True)
    before_resume = None
    if before_resume_path is not None:
        before_resume = restore_progress_read_only(before_resume_path)
        _require_native_v2(before_resume, require_complete=False)

    return {
        "status": "ok",
        "progress": _input_identity(progress),
        "before_resume": (
            None if before_resume is None else _input_identity(before_resume)
        ),
        "stages": {
            stage: _enumerate_stage(stage, telemetry)
            for stage, telemetry in sorted(progress.stage_telemetry.items())
            if stage in {"pages", "images"}
        },
    }

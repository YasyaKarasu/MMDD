from __future__ import annotations

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_wdc200k_mm_joinability_dataset as pipeline_module  # noqa: E402
from build_wdc200k_mm_joinability_dataset import (  # noqa: E402
    PipelineConfig,
    ProgressReporter,
)
from validate_wdc_eta_gate import (  # noqa: E402
    GateValidationError,
    validate_eta_gate,
)
from wdc200k_eta import UrlProgressSnapshot  # noqa: E402


URL_TELEMETRY_V2 = "wdc200k-url-telemetry-v2"


def _config(root: Path) -> PipelineConfig:
    return PipelineConfig(
        input_dir=root / "input",
        output_dir=root / "output",
        work_dir=root / "work",
        cache_dir=root / "cache",
        min_free_disk_bytes=0,
        resume=True,
    )


def _snapshot(
    *,
    epoch: str,
    completed: int,
    total: int,
    elapsed: float,
    baseline: int = 0,
) -> UrlProgressSnapshot:
    transport = [0] * 64
    commit = [0] * 32
    completed_in_epoch = completed - baseline
    transport[0] = completed_in_epoch
    commit[0] = completed_in_epoch
    return UrlProgressSnapshot(
        execution_epoch=epoch,
        baseline_completed=baseline,
        completed_durable=completed,
        total=total,
        local_buffered_not_started=0,
        in_flight_jobs=0,
        physical_in_flight=0,
        finished_not_durable=0,
        unobserved_nonlocal=total - completed,
        deadline_seconds=8.0,
        effective_concurrency=4,
        epoch_elapsed_seconds=elapsed,
        transport_event_histogram=tuple(transport),
        active_censor_histogram=(0,) * 64,
        commit_event_histogram=tuple(commit),
        transport_overflow_events=0,
        active_overflow_censors=0,
        commit_overflow_events=0,
    )


def _native_v2_progress(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete_images: bool = True,
) -> Path:
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 1_000.0)
    reporter = ProgressReporter(_config(root))
    for stage in ("pages", "images"):
        final = 4 if stage == "pages" or complete_images else 2
        reporter.update(
            stage=stage,
            url_snapshot=_snapshot(
                epoch=f"{stage}-epoch",
                completed=0,
                total=4,
                elapsed=0.0,
            ),
        )
        reporter.update(
            url_snapshot=_snapshot(
                epoch=f"{stage}-epoch",
                completed=2,
                total=4,
                elapsed=2.0,
            )
        )
        if final == 4:
            reporter.update(
                url_snapshot=_snapshot(
                    epoch=f"{stage}-epoch",
                    completed=4,
                    total=4,
                    elapsed=4.0,
                )
            )
    reporter.publish()
    return reporter.path


def _progress_with_page_snapshots(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    page_snapshots: list[UrlProgressSnapshot],
) -> Path:
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 2_000.0)
    config = _config(root)
    reporter = ProgressReporter(config)
    active_epoch: str | None = None
    for index, snapshot in enumerate(page_snapshots):
        if active_epoch is not None and snapshot.execution_epoch != active_epoch:
            reporter.publish()
            reporter = ProgressReporter(config)
        reporter.update(
            stage=(
                "pages"
                if index == 0 or snapshot.execution_epoch != active_epoch
                else None
            ),
            url_snapshot=snapshot,
        )
        active_epoch = snapshot.execution_epoch
    reporter.update(
        stage="images",
        url_snapshot=_snapshot(
            epoch="images-epoch", completed=0, total=1, elapsed=0.0
        ),
    )
    reporter.update(
        url_snapshot=_snapshot(
            epoch="images-epoch", completed=1, total=1, elapsed=1.0
        )
    )
    reporter.publish()
    return reporter.path


def _progress_with_tied_factors(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    samples = [
        _snapshot(epoch="pages-epoch", completed=0, total=12, elapsed=0.0),
        _snapshot(epoch="pages-epoch", completed=6, total=12, elapsed=4.0),
        _snapshot(epoch="pages-epoch", completed=8, total=12, elapsed=6.0),
        _snapshot(
            epoch="pages-epoch",
            completed=10,
            total=12,
            elapsed=20.0 / 3.0,
        ),
        _snapshot(epoch="pages-epoch", completed=12, total=12, elapsed=12.0),
    ]
    return _progress_with_page_snapshots(root, monkeypatch, samples)


def _progress_with_all_exclusions(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_eligible: bool,
) -> Path:
    samples = [
        _snapshot(epoch="old", completed=0, total=10, elapsed=0.0),
        _snapshot(epoch="old", completed=4, total=10, elapsed=4.0),
        _snapshot(
            epoch="resumed",
            baseline=5,
            completed=5,
            total=10,
            elapsed=0.0,
        ),
    ]
    if include_eligible:
        samples.append(
            _snapshot(
                epoch="resumed",
                baseline=5,
                completed=6,
                total=10,
                elapsed=2.0,
            )
        )
    samples.extend(
        [
            _snapshot(
                epoch="resumed",
                baseline=5,
                completed=8,
                total=10,
                elapsed=10.0,
            ),
            _snapshot(
                epoch="resumed",
                baseline=5,
                completed=10,
                total=10,
                elapsed=10.0,
            ),
        ]
    )
    return _progress_with_page_snapshots(root, monkeypatch, samples)


def _payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_v1_progress(root: Path) -> Path:
    path = root / "progress-v1.json"
    stages: dict[str, Any] = {}
    for stage, basis in (("pages", "page_urls"), ("images", "image_urls")):
        stages[stage] = {
            "rate_basis": basis,
            "samples": [
                {
                    "timestamp": 10.0,
                    "completed_units": 2,
                    "total_units": 4,
                    "rate": 1.0,
                    "rolling_rate": 1.0,
                    "predicted_remaining_seconds": 2.0,
                },
                {
                    "timestamp": 12.0,
                    "completed_units": 4,
                    "total_units": 4,
                    "rate": 1.0,
                    "rolling_rate": 1.0,
                    "predicted_remaining_seconds": 0.0,
                },
            ],
            "completed_units": 4,
            "total_units": 4,
            "completed_at": 12.0,
            "eligible_final_half_samples": 1,
            "excluded_final_half_samples": 1,
            "max_symmetric_eta_factor": 1.0,
        }
    _write_payload(path, {"stage_telemetry": stages})
    return path


def test_validate_eta_gate_restores_native_v2_without_writing_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _native_v2_progress(tmp_path / "final", monkeypatch)
    before = _native_v2_progress(tmp_path / "before", monkeypatch)
    before_properties = {
        path: (
            path.read_bytes(),
            path.stat().st_mtime_ns,
            stat.S_IMODE(path.stat().st_mode),
        )
        for path in (progress, before)
    }

    result = validate_eta_gate(progress, before)

    assert result["status"] == "ok"
    assert set(result["stages"]) == {"pages", "images"}
    for path, (original_bytes, original_mtime, original_mode) in (
        before_properties.items()
    ):
        assert path.read_bytes() == original_bytes
        assert path.stat().st_mtime_ns == original_mtime
        assert stat.S_IMODE(path.stat().st_mode) == original_mode


def test_validate_eta_gate_rejects_completed_v1(tmp_path: Path) -> None:
    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(_completed_v1_progress(tmp_path))

    assert caught.value.code == "NON_NATIVE_V2"


def test_validate_eta_gate_rejects_missing_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _native_v2_progress(tmp_path, monkeypatch)
    payload = _payload(progress)
    payload["stage_telemetry"].pop("images")
    _write_payload(progress, payload)

    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(progress)

    assert caught.value.code == "MISSING_URL_STAGE"


def test_validate_eta_gate_rejects_incomplete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _native_v2_progress(
        tmp_path, monkeypatch, complete_images=False
    )

    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(progress)

    assert caught.value.code == "INCOMPLETE_URL_STAGE"


def test_validate_eta_gate_rejects_malformed_histogram(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _native_v2_progress(tmp_path, monkeypatch)
    payload = _payload(progress)
    payload["stage_telemetry"]["pages"]["samples"][-1][
        "histogram_blob"
    ] = "not-base64"
    _write_payload(progress, payload)

    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(progress)

    assert caught.value.code == "STRICT_RESTORE_FAILED"


def test_validate_eta_gate_rejects_mixed_v1_v2_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _native_v2_progress(tmp_path, monkeypatch)
    payload = _payload(progress)
    pages = payload["stage_telemetry"]["pages"]
    first_timestamp = pages["samples"][0]["timestamp"]
    pages["samples"].insert(
        0,
        {
            "timestamp": first_timestamp - 1.0,
            "completed_units": 0,
            "total_units": 4,
            "rate": 0.0,
            "rolling_rate": 0.0,
            "predicted_remaining_seconds": None,
        },
    )
    _write_payload(progress, payload)

    with pytest.raises(GateValidationError) as caught:
        validate_eta_gate(progress)

    assert caught.value.code == "NON_NATIVE_V2"


def test_eligible_records_sort_factor_desc_then_position_asc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = validate_eta_gate(
        _progress_with_tied_factors(tmp_path, monkeypatch)
    )["stages"]["pages"]["eligible_samples"]

    assert [row["sample_position"] for row in records] == [3, 1, 2]
    assert [row["symmetric_factor"] for row in records] == pytest.approx(
        [4.0, 2.0, 2.0]
    )
    assert validate_eta_gate(
        _progress_with_tied_factors(tmp_path / "again", monkeypatch)
    )["stages"]["pages"]["worst_record"]["sample_position"] == 3


def test_final_half_samples_are_enumerated_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress_with_all_exclusions(
        tmp_path, monkeypatch, include_eligible=True
    )
    result = validate_eta_gate(progress)["stages"]["pages"]
    raw_samples = _payload(progress)["stage_telemetry"]["pages"]["samples"]
    expected = {
        position
        for position, sample in enumerate(raw_samples)
        if sample["completed_units"] * 2 >= sample["total_units"]
    }
    eligible = {
        row["sample_position"] for row in result["eligible_samples"]
    }
    excluded = {
        row["sample_position"] for row in result["excluded_samples"]
    }

    assert eligible.isdisjoint(excluded)
    assert eligible | excluded == expected


def test_exclusion_reasons_follow_fixed_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = validate_eta_gate(
        _progress_with_all_exclusions(
            tmp_path, monkeypatch, include_eligible=True
        )
    )["stages"]["pages"]

    assert [row["exclusion_reason"] for row in stage["excluded_samples"]] == [
        "PREDICTION_MISSING",
        "ACTUAL_REMAINING_NON_POSITIVE",
        "PREDICTION_NON_POSITIVE",
    ]


def test_zero_eligible_has_explicit_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = validate_eta_gate(
        _progress_with_all_exclusions(
            tmp_path, monkeypatch, include_eligible=False
        )
    )["stages"]["pages"]

    assert stage["eligible_final_half_samples"] == 0
    assert stage["max_symmetric_eta_factor"] is None
    assert stage["worst_record"] is None
    assert stage["worst_reason"] == "ZERO_ELIGIBLE_SAMPLES"

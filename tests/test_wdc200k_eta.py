from __future__ import annotations

import base64
import inspect
import json
import math
import sys
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wdc200k_eta import (  # noqa: E402
    UrlEtaEstimate,
    UrlProgressSnapshot,
    decode_histogram_blob,
    encode_histogram_blob,
    estimate_url_eta,
    fixed_bin,
)


PAGE_FIXTURE = ROOT / "tests/fixtures/wdc_gate100_page_eta_events.json"
IMAGE_FIXTURE = ROOT / "tests/fixtures/wdc_gate100_image_eta_events.json"
UINT64_MAX = 2**64 - 1


def _histogram(size: int, values: dict[int, int] | None = None) -> tuple[int, ...]:
    result = [0] * size
    for index, count in (values or {}).items():
        result[index] = count
    return tuple(result)


def _snapshot(**overrides: object) -> UrlProgressSnapshot:
    values: dict[str, object] = {
        "execution_epoch": "epoch-1",
        "baseline_completed": 0,
        "completed_durable": 1,
        "total": 2,
        "local_buffered_not_started": 1,
        "in_flight_jobs": 0,
        "physical_in_flight": 0,
        "finished_not_durable": 0,
        "unobserved_nonlocal": 0,
        "deadline_seconds": 8.0,
        "effective_concurrency": 128,
        "epoch_elapsed_seconds": 1.0,
        "transport_event_histogram": _histogram(64, {0: 1}),
        "active_censor_histogram": _histogram(64),
        "commit_event_histogram": _histogram(32, {0: 1}),
        "transport_overflow_events": 0,
        "active_overflow_censors": 0,
        "commit_overflow_events": 0,
    }
    values.update(overrides)
    return UrlProgressSnapshot(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ReplaySample:
    completed_units: int
    elapsed_seconds: float
    predicted_remaining_seconds: float | None
    actual_remaining_seconds: float
    symmetric_factor: float | None
    used_future_state: bool


@dataclass(frozen=True)
class ReplayResult:
    samples: tuple[ReplaySample, ...]
    max_symmetric_factor: float
    worst_completed_units: int


def _fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event_histogram(
    values: list[float], horizon: float, bin_count: int
) -> tuple[tuple[int, ...], int]:
    counts = [0] * bin_count
    overflow = 0
    for value in values:
        index, is_overflow = fixed_bin(value, horizon, bin_count)
        counts[index] += 1
        overflow += int(is_overflow)
    return tuple(counts), overflow


def _replay(path: Path) -> ReplayResult:
    fixture = _fixture(path)
    total = int(fixture["total"])
    deadline = float(fixture["deadline_seconds"])
    concurrency = int(fixture["effective_concurrency"])
    baseline = int(fixture["baseline_completed"])
    events = list(fixture["events"])
    final_elapsed = max(
        float(item["elapsed_seconds"])
        for item in events
        if item["kind"] == "durable"
    )
    starts: dict[int, float] = {}
    finishes: dict[int, float] = {}
    durable: dict[int, float] = {}
    samples: list[ReplaySample] = []

    for prefix_length, event in enumerate(events, start=1):
        kind = str(event["kind"])
        ordinal = int(event["attempt_ordinal"])
        elapsed = float(event["elapsed_seconds"])
        if kind == "start":
            starts[ordinal] = elapsed
        elif kind == "finish":
            finishes[ordinal] = elapsed
        elif kind == "durable":
            durable[ordinal] = elapsed
        else:  # pragma: no cover - protected by the frozen fixture test
            raise AssertionError(kind)

        if kind != "durable":
            continue

        transport_histogram, transport_overflow = _event_histogram(
            [finishes[item] - starts[item] for item in finishes], deadline, 64
        )
        active = set(starts) - set(finishes)
        censor_histogram, censor_overflow = _event_histogram(
            [elapsed - starts[item] for item in active], deadline, 64
        )
        commit_horizon = min(2.0, deadline)
        commit_histogram, commit_overflow = _event_histogram(
            [durable_elapsed - finishes[item]
             for item, durable_elapsed in durable.items()],
            commit_horizon,
            32,
        )
        finished_not_durable = len(finishes) - len(durable)
        snapshot = UrlProgressSnapshot(
            execution_epoch=f"gate-{fixture['stage']}",
            baseline_completed=baseline,
            completed_durable=baseline + len(durable),
            total=total,
            local_buffered_not_started=total - len(starts),
            in_flight_jobs=len(active),
            physical_in_flight=len(active),
            finished_not_durable=finished_not_durable,
            unobserved_nonlocal=0,
            deadline_seconds=deadline,
            effective_concurrency=concurrency,
            epoch_elapsed_seconds=elapsed,
            transport_event_histogram=transport_histogram,
            active_censor_histogram=censor_histogram,
            commit_event_histogram=commit_histogram,
            transport_overflow_events=transport_overflow,
            active_overflow_censors=censor_overflow,
            commit_overflow_events=commit_overflow,
        )
        estimate = estimate_url_eta(snapshot)
        actual = final_elapsed - elapsed
        predicted = estimate.predicted_remaining_seconds
        eligible = 2 * len(durable) >= total and len(durable) < total
        factor = None
        if eligible and predicted is not None and predicted > 0.0 and actual > 0.0:
            factor = max(predicted / actual, actual / predicted)
        samples.append(
            ReplaySample(
                completed_units=len(durable),
                elapsed_seconds=elapsed,
                predicted_remaining_seconds=predicted,
                actual_remaining_seconds=actual,
                symmetric_factor=factor,
                used_future_state=any(
                    float(candidate["elapsed_seconds"]) > elapsed
                    for candidate in events[:prefix_length]
                ),
            )
        )

    eligible_samples = [sample for sample in samples if sample.symmetric_factor]
    worst = max(eligible_samples, key=lambda sample: sample.symmetric_factor or 0.0)
    return ReplayResult(
        samples=tuple(samples),
        max_symmetric_factor=worst.symmetric_factor or 0.0,
        worst_completed_units=worst.completed_units,
    )


def _legacy_replay(path: Path) -> tuple[float, int]:
    fixture = _fixture(path)
    total = int(fixture["total"])
    durable_times = sorted(
        float(item["elapsed_seconds"])
        for item in fixture["events"]
        if item["kind"] == "durable"
    )
    completed_at = durable_times[-1]
    factors: list[tuple[float, int]] = []
    for completed, elapsed in enumerate(durable_times, start=1):
        if 2 * completed < total or completed == total:
            continue
        cumulative_rate = completed / elapsed
        predicted = (total - completed) / cumulative_rate
        actual = completed_at - elapsed
        factors.append((max(predicted / actual, actual / predicted), completed))
    return max(factors)


def test_gate_event_fixtures_are_aggregate_and_frozen() -> None:
    for path, expected_count in ((PAGE_FIXTURE, 100), (IMAGE_FIXTURE, 135)):
        payload = path.read_bytes()
        assert b"url" not in payload.lower()
        assert b"payload" not in payload.lower()
        fixture = json.loads(payload)
        assert set(fixture) == {
            "schema_version",
            "stage",
            "total",
            "deadline_seconds",
            "effective_concurrency",
            "baseline_completed",
            "events",
        }
        events = fixture["events"]
        assert sorted({item["attempt_ordinal"] for item in events}) == list(
            range(expected_count)
        )
        assert all(set(item) == {"kind", "attempt_ordinal", "elapsed_seconds"}
                   for item in events)
        assert sum(item["kind"] == "start" for item in events) == expected_count
        assert sum(item["kind"] == "finish" for item in events) == expected_count
        assert sum(item["kind"] == "durable" for item in events) == expected_count


def test_frozen_fixtures_reproduce_legacy_gate_factors() -> None:
    assert _legacy_replay(PAGE_FIXTURE) == pytest.approx(
        (2.879517510143001, 73)
    )
    assert _legacy_replay(IMAGE_FIXTURE) == pytest.approx(
        (1.8537476708755196, 117)
    )


def test_gate_prefix_replay_passes_without_future_state() -> None:
    page = _replay(PAGE_FIXTURE)
    image = _replay(IMAGE_FIXTURE)

    assert page.max_symmetric_factor == pytest.approx(1.8978047370910645)
    assert page.worst_completed_units == 82
    assert image.max_symmetric_factor <= 1.8537476708755196
    assert all(
        sample.used_future_state is False
        for sample in page.samples + image.samples
    )


def test_snapshot_and_estimate_have_the_exact_public_fields() -> None:
    assert [field.name for field in fields(UrlProgressSnapshot)] == [
        "execution_epoch", "baseline_completed", "completed_durable", "total",
        "local_buffered_not_started", "in_flight_jobs", "physical_in_flight",
        "finished_not_durable", "unobserved_nonlocal", "deadline_seconds",
        "effective_concurrency", "epoch_elapsed_seconds",
        "transport_event_histogram", "active_censor_histogram",
        "commit_event_histogram", "transport_overflow_events",
        "active_overflow_censors", "commit_overflow_events",
    ]
    assert [field.name for field in fields(UrlEtaEstimate)] == [
        "durable_rate", "rate_eta", "queue_eta", "inflight_eta", "commit_eta",
        "overflow_eta", "predicted_remaining_seconds", "fallback",
    ]
    snapshot = _snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.total = 3  # type: ignore[misc]
    assert tuple(inspect.signature(estimate_url_eta).parameters) == ("snapshot",)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, (0, False)),
        (8.0 / 64.0, (1, False)),
        (8.0, (63, True)),
        (9.0, (63, True)),
    ],
)
def test_fixed_bin_exact_edges(value: float, expected: tuple[int, bool]) -> None:
    assert fixed_bin(value, 8.0, 64) == expected


def test_km_risk_set_rmst_and_maturity_boundary_hand_calculation() -> None:
    events = _histogram(64, {0: 1, 22: 1})
    young = _snapshot(
        total=4,
        completed_durable=1,
        local_buffered_not_started=1,
        in_flight_jobs=1,
        physical_in_flight=1,
        finished_not_durable=0,
        unobserved_nonlocal=1,
        deadline_seconds=64.0,
        transport_event_histogram=events,
        active_censor_histogram=_histogram(64, {21: 1}),
        epoch_elapsed_seconds=1.0,
    )
    assert estimate_url_eta(young).inflight_eta is None
    mature = replace(
        young,
        active_censor_histogram=_histogram(64, {22: 1}),
    )
    assert estimate_url_eta(mature).inflight_eta == pytest.approx(21.5)


def test_overflow_uses_two_durable_intervals_and_zero_capped_rmst() -> None:
    snapshot = _snapshot(
        completed_durable=4,
        total=5,
        local_buffered_not_started=0,
        in_flight_jobs=1,
        physical_in_flight=1,
        epoch_elapsed_seconds=2.0,
        transport_event_histogram=_histogram(64, {0: 4}),
        active_censor_histogram=_histogram(64, {63: 1}),
        active_overflow_censors=1,
        commit_event_histogram=_histogram(32, {0: 4}),
    )
    estimate = estimate_url_eta(snapshot)
    assert estimate.durable_rate == pytest.approx(2.0)
    assert estimate.inflight_eta == 0.0
    assert estimate.overflow_eta == pytest.approx(1.0)
    assert estimate.predicted_remaining_seconds == pytest.approx(1.0)
    assert estimate.fallback == "active_overflow_continuity"


def test_young_censor_keeps_rate_authoritative() -> None:
    snapshot = _snapshot(
        completed_durable=2,
        total=4,
        local_buffered_not_started=1,
        in_flight_jobs=1,
        physical_in_flight=1,
        transport_event_histogram=_histogram(64, {0: 2}),
        active_censor_histogram=_histogram(64, {21: 1}),
        commit_event_histogram=_histogram(32, {0: 2}),
        epoch_elapsed_seconds=2.0,
    )
    estimate = estimate_url_eta(snapshot)
    assert estimate.rate_eta == pytest.approx(2.0)
    assert estimate.inflight_eta is None
    assert estimate.predicted_remaining_seconds == pytest.approx(2.0)
    assert estimate.fallback == "young_active_censor"


def test_commit_midpoint_mean_and_queue_are_diagnostics() -> None:
    snapshot = _snapshot(
        completed_durable=2,
        total=10,
        local_buffered_not_started=2,
        in_flight_jobs=2,
        physical_in_flight=0,
        finished_not_durable=2,
        unobserved_nonlocal=2,
        epoch_elapsed_seconds=1.0,
        transport_event_histogram=_histogram(64, {0: 2}),
        commit_event_histogram=_histogram(32, {0: 1, 31: 1}),
    )
    estimate = estimate_url_eta(snapshot)
    assert estimate.durable_rate == pytest.approx(2.0)
    assert estimate.queue_eta == pytest.approx(2.0)
    assert estimate.commit_eta == pytest.approx(2.0)
    assert estimate.rate_eta == pytest.approx(4.0)
    assert estimate.predicted_remaining_seconds == pytest.approx(4.0)


def test_commit_mean_falls_back_to_one_bin_midpoint() -> None:
    estimate = estimate_url_eta(
        _snapshot(
            completed_durable=1,
            total=2,
            local_buffered_not_started=0,
            finished_not_durable=1,
            commit_event_histogram=_histogram(32),
        )
    )
    assert estimate.commit_eta == pytest.approx(2.0 / 32.0)


def test_no_rate_and_no_transport_events_returns_null_eta() -> None:
    estimate = estimate_url_eta(
        _snapshot(
            completed_durable=0,
            total=1,
            local_buffered_not_started=1,
            epoch_elapsed_seconds=0.0,
            transport_event_histogram=_histogram(64),
            commit_event_histogram=_histogram(32),
        )
    )
    assert estimate.durable_rate is None
    assert estimate.predicted_remaining_seconds is None
    assert estimate.fallback == "no_durable_rate_or_transport_events"


def test_empty_km_risk_set_keeps_available_rate_authoritative() -> None:
    estimate = estimate_url_eta(
        _snapshot(
            transport_event_histogram=_histogram(64),
            commit_event_histogram=_histogram(32),
        )
    )
    assert estimate.rate_eta == pytest.approx(1.0)
    assert estimate.inflight_eta is None
    assert estimate.predicted_remaining_seconds == pytest.approx(1.0)
    assert estimate.fallback == "empty_km_risk_set"


def test_completed_snapshot_has_zero_remaining_eta() -> None:
    estimate = estimate_url_eta(
        _snapshot(
            completed_durable=2,
            total=2,
            local_buffered_not_started=0,
            epoch_elapsed_seconds=2.0,
            transport_event_histogram=_histogram(64, {0: 2}),
            commit_event_histogram=_histogram(32, {0: 2}),
        )
    )
    assert estimate.predicted_remaining_seconds == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"total": 3},
        {"physical_in_flight": 1},
        {"active_censor_histogram": _histogram(64, {0: 1})},
        {"transport_event_histogram": _histogram(63)},
        {"active_censor_histogram": _histogram(65)},
        {"commit_event_histogram": _histogram(31)},
        {"baseline_completed": 2},
        {"transport_overflow_events": 2},
        {"execution_epoch": ""},
    ],
)
def test_snapshot_rejects_invalid_topology_and_schema(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _snapshot(**overrides)


def test_histogram_codec_is_canonical_fixed_width_uint64() -> None:
    snapshot = _snapshot(
        transport_event_histogram=(UINT64_MAX,) * 64,
        active_censor_histogram=_histogram(64),
        commit_event_histogram=(UINT64_MAX,) * 32,
        transport_overflow_events=UINT64_MAX,
        commit_overflow_events=UINT64_MAX,
    )
    blob = encode_histogram_blob(snapshot)
    assert len(base64.b64decode(blob)) == 1_280
    transport, active, commit = decode_histogram_blob(blob)
    assert transport == (UINT64_MAX,) * 64
    assert active == _histogram(64)
    assert commit == (UINT64_MAX,) * 32
    assert encode_histogram_blob(snapshot) == blob


@pytest.mark.parametrize("value", [-1, 2**64])
def test_snapshot_rejects_histogram_values_outside_uint64(value: int) -> None:
    values = list(_histogram(64))
    values[0] = value
    with pytest.raises(ValueError):
        _snapshot(transport_event_histogram=tuple(values))


@pytest.mark.parametrize(
    "blob",
    [
        "not-base64!",
        base64.b64encode(b"short").decode("ascii"),
        base64.b64encode(bytes(1_281)).decode("ascii"),
        base64.b64encode(bytes(1_280)).decode("ascii") + "\n",
    ],
)
def test_histogram_decoder_rejects_invalid_or_noncanonical_blob(blob: str) -> None:
    with pytest.raises(ValueError):
        decode_histogram_blob(blob)


@pytest.mark.parametrize(
    "overrides",
    [
        {"deadline_seconds": math.nan},
        {"deadline_seconds": math.inf},
        {"epoch_elapsed_seconds": math.nan},
        {"epoch_elapsed_seconds": math.inf},
    ],
)
def test_snapshot_rejects_non_finite_scalars(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _snapshot(**overrides)


def test_estimator_rejects_non_finite_derived_rate() -> None:
    snapshot = _snapshot(epoch_elapsed_seconds=5e-324)
    with pytest.raises(ValueError):
        estimate_url_eta(snapshot)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_fixed_bin_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        fixed_bin(value, 8.0, 64)

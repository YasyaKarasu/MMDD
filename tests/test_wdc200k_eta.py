from __future__ import annotations

import base64
import inspect
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, fields, replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from wdc200k_eta import (  # noqa: E402
    DurableUrlCounts,
    UrlEtaEstimate,
    UrlProgressTracker,
    UrlProgressSnapshot,
    decode_histogram_blob,
    encode_histogram_blob,
    estimate_url_eta,
    fixed_bin,
)
import wdc200k_eta as eta_module  # noqa: E402


PAGE_FIXTURE = ROOT / "tests/fixtures/wdc_gate100_page_eta_events.json"
IMAGE_FIXTURE = ROOT / "tests/fixtures/wdc_gate100_image_eta_events.json"
UINT64_MAX = 2**64 - 1


class FakeMonotonic:
    def __init__(self) -> None:
        self._value = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


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


def _eligible_symmetric_factor(
    *,
    completed: int,
    total: int,
    predicted: float | None,
    actual: float,
) -> float | None:
    eligible = 2 * completed >= total and completed < total
    if not eligible:
        return None
    assert predicted is not None, "eligible sample has no ETA prediction"
    assert math.isfinite(predicted), "eligible ETA prediction is not finite"
    assert predicted > 0.0, "eligible ETA prediction is not positive"
    assert math.isfinite(actual), "eligible actual remaining time is not finite"
    assert actual > 0.0, "eligible actual remaining time is not positive"
    return max(predicted / actual, actual / predicted)


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
        factor = _eligible_symmetric_factor(
            completed=len(durable),
            total=total,
            predicted=predicted,
            actual=actual,
        )
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

    eligible_samples = [
        sample
        for sample in samples
        if 2 * sample.completed_units >= total
        and sample.completed_units < total
    ]
    assert all(sample.symmetric_factor is not None for sample in eligible_samples)
    worst = max(eligible_samples, key=lambda sample: float(sample.symmetric_factor))
    assert worst.symmetric_factor is not None
    return ReplayResult(
        samples=tuple(samples),
        max_symmetric_factor=worst.symmetric_factor,
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
    assert sum(sample.symmetric_factor is not None for sample in page.samples) == 50
    assert sum(sample.symmetric_factor is not None for sample in image.samples) == 67
    assert all(
        sample.used_future_state is False
        for sample in page.samples + image.samples
    )


@pytest.mark.parametrize(
    ("predicted", "actual"),
    [
        (None, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (math.nan, 1.0),
        (math.inf, 1.0),
        (1.0, 0.0),
        (1.0, math.nan),
        (1.0, math.inf),
    ],
)
def test_eligible_factor_rejects_non_positive_or_non_finite_samples(
    predicted: float | None,
    actual: float,
) -> None:
    with pytest.raises(AssertionError):
        _eligible_symmetric_factor(
            completed=50,
            total=100,
            predicted=predicted,
            actual=actual,
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


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "transport_event_histogram": _histogram(64, {0: 1}),
            "transport_overflow_events": 1,
        },
        {
            "local_buffered_not_started": 0,
            "in_flight_jobs": 1,
            "physical_in_flight": 1,
            "active_censor_histogram": _histogram(64, {0: 1}),
            "active_overflow_censors": 1,
        },
        {
            "commit_event_histogram": _histogram(32, {0: 1}),
            "commit_overflow_events": 1,
        },
    ],
    ids=("transport", "active-censor", "commit"),
)
def test_snapshot_rejects_overflow_not_represented_in_last_bin(
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


def test_v2_estimator_metadata_is_integer_rational_and_deadline_bound() -> None:
    assert eta_module.URL_TELEMETRY_SCHEMA_VERSION == (
        "wdc200k-url-telemetry-v2"
    )
    assert eta_module.url_estimator_metadata(8.0) == {
        "transport_bins": 64,
        "active_censor_bins": 64,
        "commit_bins": 32,
        "maturity_numerator": 1,
        "maturity_denominator": 3,
        "deadline_seconds": 8.0,
    }
    with pytest.raises(ValueError):
        eta_module.url_estimator_metadata(float("nan"))


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


def _counts(
    *, completed: int = 0, pending: int = 0, leased: int = 0, total: int = 1
) -> DurableUrlCounts:
    return DurableUrlCounts(
        completed=completed,
        pending=pending,
        leased=leased,
        total=total,
    )


def test_tracker_captures_exact_mutable_censors_and_monotonic_histograms() -> None:
    clock = FakeMonotonic()
    tracker = UrlProgressTracker(
        total=1,
        deadline_seconds=8.0,
        effective_concurrency=1,
        execution_epoch="epoch-race",
        baseline_completed=0,
        monotonic=clock,
    )
    tracker.buffered(1)
    tracker.job_submitted("job")
    tracker.physical_started("job")

    clock.advance(1.0)
    young = tracker.snapshot(_counts(leased=1))
    clock.advance(2.0)
    older = tracker.snapshot(_counts(leased=1))
    assert sum(young.active_censor_histogram) == 1
    assert sum(older.active_censor_histogram) == 1
    assert young.active_censor_histogram != older.active_censor_histogram
    assert young.epoch_elapsed_seconds == 1.0
    assert older.epoch_elapsed_seconds == 3.0

    tracker.physical_finished("job")
    tracker.future_finished("job")
    clock.advance(0.25)
    tracker.durable_completed("job")
    completed = tracker.snapshot(_counts(completed=1))
    assert sum(completed.transport_event_histogram) == 1
    assert sum(completed.commit_event_histogram) == 1
    assert completed.active_censor_histogram == (0,) * 64
    assert completed.completed_durable == completed.total == 1


def test_tracker_interleaves_128_workers_with_atomic_topology_snapshots() -> None:
    total = 128
    clock = FakeMonotonic()
    tracker = UrlProgressTracker(
        total=total,
        deadline_seconds=4.0,
        effective_concurrency=total,
        execution_epoch="epoch-128",
        baseline_completed=0,
        monotonic=clock,
    )
    tracker.buffered(total)
    all_started = threading.Barrier(total + 1)
    release_finishes = threading.Barrier(total + 1)
    all_futures_finished = threading.Barrier(total + 1)
    release_commits = threading.Barrier(total + 1)

    def run(index: int) -> None:
        job_id = f"job-{index}"
        tracker.job_submitted(job_id)
        tracker.physical_started(job_id)
        all_started.wait(timeout=5)
        release_finishes.wait(timeout=5)
        tracker.physical_finished(job_id)
        tracker.future_finished(job_id)
        all_futures_finished.wait(timeout=5)
        release_commits.wait(timeout=5)
        tracker.durable_completed(job_id)

    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = [pool.submit(run, index) for index in range(total)]
        all_started.wait(timeout=5)
        clock.advance(1.0)
        active = tracker.snapshot(_counts(leased=total, total=total))
        assert active.local_buffered_not_started == 0
        assert active.in_flight_jobs == total
        assert active.physical_in_flight == total
        assert sum(active.active_censor_histogram) == total
        assert active.physical_in_flight <= active.in_flight_jobs

        release_finishes.wait(timeout=5)
        all_futures_finished.wait(timeout=5)
        clock.advance(0.5)
        finished = tracker.snapshot(_counts(leased=total, total=total))
        assert finished.in_flight_jobs == 0
        assert finished.finished_not_durable == total
        assert sum(finished.transport_event_histogram) == total
        assert all(
            after >= before
            for before, after in zip(
                active.transport_event_histogram,
                finished.transport_event_histogram,
            )
        )

        release_commits.wait(timeout=5)
        for future in futures:
            future.result(timeout=5)

    durable = tracker.snapshot(_counts(completed=total, total=total))
    assert durable.completed_durable == total
    assert durable.finished_not_durable == 0
    assert sum(durable.commit_event_histogram) == total
    assert (
        durable.completed_durable
        + durable.local_buffered_not_started
        + durable.in_flight_jobs
        + durable.finished_not_durable
        + durable.unobserved_nonlocal
        == total
    )


def test_tracker_external_durable_completion_reduces_unobserved_residual() -> None:
    tracker = UrlProgressTracker(
        total=3,
        deadline_seconds=8.0,
        effective_concurrency=1,
        execution_epoch="epoch-external",
        baseline_completed=1,
        monotonic=lambda: 10.0,
    )
    initial = tracker.snapshot(
        _counts(completed=1, pending=1, leased=1, total=3)
    )
    external = tracker.snapshot(
        _counts(completed=2, pending=1, leased=0, total=3)
    )
    assert initial.unobserved_nonlocal == 2
    assert external.unobserved_nonlocal == 1


def test_tracker_rejects_unknown_duplicate_and_out_of_order_transitions() -> None:
    clock = FakeMonotonic()
    tracker = UrlProgressTracker(
        total=2,
        deadline_seconds=8.0,
        effective_concurrency=1,
        execution_epoch="epoch-errors",
        baseline_completed=0,
        monotonic=clock,
    )
    tracker.buffered(2)
    tracker.job_submitted("known")
    with pytest.raises(ValueError, match="buffer"):
        tracker.buffered(1)
    with pytest.raises(ValueError, match="duplicate|already"):
        tracker.job_submitted("known")
    with pytest.raises(ValueError, match="unknown"):
        tracker.physical_started("missing")
    tracker.physical_started("known")
    with pytest.raises(ValueError, match="duplicate|already"):
        tracker.physical_started("known")
    with pytest.raises(ValueError, match="physical"):
        tracker.future_finished("known")
    tracker.physical_finished("known")
    with pytest.raises(ValueError, match="physical|active"):
        tracker.physical_finished("known")
    tracker.future_finished("known")
    with pytest.raises(ValueError, match="duplicate|already"):
        tracker.future_finished("known")
    tracker.durable_completed("known")
    with pytest.raises(ValueError, match="unknown|durable"):
        tracker.durable_completed("known")


def test_tracker_rejects_buffer_and_histogram_uint64_overflow() -> None:
    clock = FakeMonotonic()
    tracker = UrlProgressTracker(
        total=1,
        deadline_seconds=1.0,
        effective_concurrency=1,
        execution_epoch="epoch-overflow",
        baseline_completed=0,
        monotonic=clock,
    )
    with pytest.raises(ValueError, match="buffer"):
        tracker.buffered(2)
    tracker.buffered(1)
    tracker.job_submitted("job")
    tracker.physical_started("job")
    clock.advance(1.0)
    tracker._transport_event_histogram[-1] = UINT64_MAX
    with pytest.raises(OverflowError, match="uint64"):
        tracker.physical_finished("job")

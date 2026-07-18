from __future__ import annotations

import base64
import binascii
from bisect import bisect_right
import math
import struct
import threading
from dataclasses import dataclass
from typing import Callable


UINT64_MAX = 2**64 - 1
TRANSPORT_BIN_COUNT = 64
COMMIT_BIN_COUNT = 32
MAX_URL_COMPLETION_PUBLICATIONS = 224
MAX_URL_EXECUTION_EPOCHS = 32
MAX_URL_STAGE_SAMPLES = (
    MAX_URL_COMPLETION_PUBLICATIONS + MAX_URL_EXECUTION_EPOCHS
)
MATURITY_FRACTION = 1.0 / 3.0
URL_TELEMETRY_SCHEMA_VERSION = "wdc200k-url-telemetry-v2"
_HISTOGRAM_STRUCT = struct.Struct("<" + "Q" * 160)


def _require_uint64(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or value > UINT64_MAX:
        raise ValueError(f"{name} must be in the uint64 range")
    return value


def _require_finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_histogram(
    histogram: tuple[int, ...], expected_length: int, name: str
) -> None:
    if not isinstance(histogram, tuple) or len(histogram) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} bins")
    for index, value in enumerate(histogram):
        _require_uint64(value, f"{name}[{index}]")


def url_estimator_metadata(deadline_seconds: float) -> dict[str, int | float]:
    """Return deterministic manifest metadata for the fixed v2 estimator."""
    deadline = _require_finite(
        deadline_seconds, "deadline_seconds", positive=True
    )
    return {
        "transport_bins": TRANSPORT_BIN_COUNT,
        "active_censor_bins": TRANSPORT_BIN_COUNT,
        "commit_bins": COMMIT_BIN_COUNT,
        "maturity_numerator": 1,
        "maturity_denominator": 3,
        "deadline_seconds": deadline,
    }


@dataclass(frozen=True)
class UrlProgressSnapshot:
    execution_epoch: str
    baseline_completed: int
    completed_durable: int
    total: int
    local_buffered_not_started: int
    in_flight_jobs: int
    physical_in_flight: int
    finished_not_durable: int
    unobserved_nonlocal: int
    deadline_seconds: float
    effective_concurrency: int
    epoch_elapsed_seconds: float
    transport_event_histogram: tuple[int, ...]
    active_censor_histogram: tuple[int, ...]
    commit_event_histogram: tuple[int, ...]
    transport_overflow_events: int
    active_overflow_censors: int
    commit_overflow_events: int

    def __post_init__(self) -> None:
        if not isinstance(self.execution_epoch, str) or not self.execution_epoch:
            raise ValueError("execution_epoch must be a non-empty string")

        counter_names = (
            "baseline_completed",
            "completed_durable",
            "total",
            "local_buffered_not_started",
            "in_flight_jobs",
            "physical_in_flight",
            "finished_not_durable",
            "unobserved_nonlocal",
            "effective_concurrency",
            "transport_overflow_events",
            "active_overflow_censors",
            "commit_overflow_events",
        )
        for name in counter_names:
            _require_uint64(getattr(self, name), name)

        if self.effective_concurrency == 0:
            raise ValueError("effective_concurrency must be positive")
        _require_finite(self.deadline_seconds, "deadline_seconds", positive=True)
        elapsed = _require_finite(
            self.epoch_elapsed_seconds, "epoch_elapsed_seconds"
        )
        if elapsed < 0.0:
            raise ValueError("epoch_elapsed_seconds must be non-negative")

        _validate_histogram(
            self.transport_event_histogram,
            TRANSPORT_BIN_COUNT,
            "transport_event_histogram",
        )
        _validate_histogram(
            self.active_censor_histogram,
            TRANSPORT_BIN_COUNT,
            "active_censor_histogram",
        )
        _validate_histogram(
            self.commit_event_histogram,
            COMMIT_BIN_COUNT,
            "commit_event_histogram",
        )

        if not self.baseline_completed <= self.completed_durable <= self.total:
            raise ValueError(
                "baseline_completed must not exceed completed_durable or total"
            )
        topology_total = (
            self.completed_durable
            + self.local_buffered_not_started
            + self.in_flight_jobs
            + self.finished_not_durable
            + self.unobserved_nonlocal
        )
        if topology_total != self.total:
            raise ValueError("URL job topology does not sum to total")
        if self.physical_in_flight > self.in_flight_jobs:
            raise ValueError("physical_in_flight exceeds in_flight_jobs")
        if sum(self.active_censor_histogram) != self.physical_in_flight:
            raise ValueError(
                "active_censor_histogram does not match physical_in_flight"
            )
        overflow_pairs = (
            (
                self.transport_overflow_events,
                self.transport_event_histogram,
                "transport_overflow_events",
            ),
            (
                self.active_overflow_censors,
                self.active_censor_histogram,
                "active_overflow_censors",
            ),
            (
                self.commit_overflow_events,
                self.commit_event_histogram,
                "commit_overflow_events",
            ),
        )
        for overflow, histogram, name in overflow_pairs:
            if overflow > histogram[-1]:
                raise ValueError(f"{name} exceeds its final histogram bin")


@dataclass(frozen=True)
class DurableUrlCounts:
    completed: int
    pending: int
    leased: int
    total: int

    def __post_init__(self) -> None:
        for name in ("completed", "pending", "leased", "total"):
            _require_uint64(getattr(self, name), name)
        if self.completed + self.pending + self.leased > self.total:
            raise ValueError("durable URL counts exceed total")


class UrlProgressTracker:
    """Bounded local scheduler state and fixed-width latency aggregates."""

    def __init__(
        self,
        total: int,
        deadline_seconds: float,
        effective_concurrency: int,
        execution_epoch: str,
        baseline_completed: int,
        monotonic: Callable[[], float],
    ) -> None:
        self._total = _require_uint64(total, "total")
        self._baseline_completed = _require_uint64(
            baseline_completed, "baseline_completed"
        )
        if self._baseline_completed > self._total:
            raise ValueError("baseline_completed exceeds total")
        self._deadline_seconds = _require_finite(
            deadline_seconds, "deadline_seconds", positive=True
        )
        self._effective_concurrency = _require_uint64(
            effective_concurrency, "effective_concurrency"
        )
        if self._effective_concurrency == 0:
            raise ValueError("effective_concurrency must be positive")
        if not isinstance(execution_epoch, str) or not execution_epoch:
            raise ValueError("execution_epoch must be a non-empty string")
        if not callable(monotonic):
            raise ValueError("monotonic must be callable")
        self._execution_epoch = execution_epoch
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._epoch_started_monotonic = _require_finite(
            monotonic(), "monotonic"
        )
        self._buffered = 0
        self._submitted: set[str] = set()
        self._physical_starts: dict[str, float] = {}
        self._finished: dict[str, float] = {}
        self._transport_event_histogram = [0] * TRANSPORT_BIN_COUNT
        self._commit_event_histogram = [0] * COMMIT_BIN_COUNT
        self._transport_overflow_events = 0
        self._commit_overflow_events = 0

    @staticmethod
    def _job_id(job_id: str) -> str:
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        return job_id

    @staticmethod
    def _delta(delta: int) -> int:
        if isinstance(delta, bool) or not isinstance(delta, int) or delta <= 0:
            raise ValueError("buffer delta must be a positive integer")
        return delta

    @staticmethod
    def _increment(values: list[int], index: int) -> None:
        if values[index] == UINT64_MAX:
            raise OverflowError("histogram counter exceeds uint64")
        values[index] += 1

    def _now_locked(self) -> float:
        return _require_finite(self._monotonic(), "monotonic")

    def buffered(self, delta: int) -> None:
        count = self._delta(delta)
        with self._lock:
            local_total = (
                self._buffered + len(self._submitted) + len(self._finished)
            )
            if local_total + count > self._total:
                raise ValueError("buffered jobs exceed total")
            self._buffered += count

    def buffered_durable(self, delta: int = 1) -> None:
        count = self._delta(delta)
        with self._lock:
            if count > self._buffered:
                raise ValueError("buffered durable jobs exceed buffer")
            self._buffered -= count

    def job_submitted(self, job_id: str) -> None:
        identity = self._job_id(job_id)
        with self._lock:
            if identity in self._submitted or identity in self._finished:
                raise ValueError(f"job {identity!r} is already tracked")
            if self._buffered == 0:
                raise ValueError("job submission has no buffered job")
            self._buffered -= 1
            self._submitted.add(identity)

    def physical_started(self, job_id: str) -> None:
        identity = self._job_id(job_id)
        with self._lock:
            if identity not in self._submitted:
                raise ValueError(f"unknown submitted job {identity!r}")
            if identity in self._physical_starts:
                raise ValueError(f"physical job {identity!r} already started")
            self._physical_starts[identity] = self._now_locked()

    def physical_finished(self, job_id: str) -> None:
        identity = self._job_id(job_id)
        with self._lock:
            if identity not in self._submitted:
                raise ValueError(f"unknown submitted job {identity!r}")
            if identity not in self._physical_starts:
                raise ValueError(f"physical job {identity!r} is not active")
            now = self._now_locked()
            duration = max(0.0, now - self._physical_starts[identity])
            index, overflow = fixed_bin(
                duration, self._deadline_seconds, TRANSPORT_BIN_COUNT
            )
            self._increment(self._transport_event_histogram, index)
            if overflow:
                if self._transport_overflow_events == UINT64_MAX:
                    self._transport_event_histogram[index] -= 1
                    raise OverflowError("transport overflow exceeds uint64")
                self._transport_overflow_events += 1
            del self._physical_starts[identity]

    def future_finished(self, job_id: str) -> None:
        identity = self._job_id(job_id)
        with self._lock:
            if identity in self._physical_starts:
                raise ValueError("physical job must finish before its future")
            if identity in self._finished:
                raise ValueError(f"job {identity!r} future already finished")
            if identity not in self._submitted:
                raise ValueError(f"unknown submitted job {identity!r}")
            self._submitted.remove(identity)
            self._finished[identity] = self._now_locked()

    def durable_completed(self, job_id: str) -> None:
        identity = self._job_id(job_id)
        with self._lock:
            if identity not in self._finished:
                raise ValueError(f"unknown finished job {identity!r} for durable")
            now = self._now_locked()
            duration = max(0.0, now - self._finished[identity])
            horizon = min(2.0, self._deadline_seconds)
            index, overflow = fixed_bin(duration, horizon, COMMIT_BIN_COUNT)
            self._increment(self._commit_event_histogram, index)
            if overflow:
                if self._commit_overflow_events == UINT64_MAX:
                    self._commit_event_histogram[index] -= 1
                    raise OverflowError("commit overflow exceeds uint64")
                self._commit_overflow_events += 1
            del self._finished[identity]

    def snapshot(self, refresh: DurableUrlCounts) -> UrlProgressSnapshot:
        if not isinstance(refresh, DurableUrlCounts):
            raise ValueError("refresh must be DurableUrlCounts")
        if refresh.total != self._total:
            raise ValueError("durable refresh total does not match tracker total")
        if refresh.completed < self._baseline_completed:
            raise ValueError("durable completion count precedes baseline")
        with self._lock:
            captured = self._now_locked()
            elapsed = captured - self._epoch_started_monotonic
            if elapsed < 0.0:
                raise ValueError("monotonic clock moved backwards")
            active_histogram = [0] * TRANSPORT_BIN_COUNT
            active_overflow = 0
            for started in self._physical_starts.values():
                index, overflow = fixed_bin(
                    max(0.0, captured - started),
                    self._deadline_seconds,
                    TRANSPORT_BIN_COUNT,
                )
                if active_histogram[index] == UINT64_MAX:
                    raise OverflowError("active censor counter exceeds uint64")
                active_histogram[index] += 1
                active_overflow += int(overflow)
            buffered = self._buffered
            in_flight = len(self._submitted)
            physical = len(self._physical_starts)
            finished = len(self._finished)
            local = buffered + in_flight + finished
            if refresh.completed + local > self._total:
                raise ValueError("local and durable URL topology exceeds total")
            unobserved = self._total - refresh.completed - local
            transport_histogram = tuple(self._transport_event_histogram)
            commit_histogram = tuple(self._commit_event_histogram)
            transport_overflow = self._transport_overflow_events
            commit_overflow = self._commit_overflow_events
        return UrlProgressSnapshot(
            execution_epoch=self._execution_epoch,
            baseline_completed=self._baseline_completed,
            completed_durable=refresh.completed,
            total=self._total,
            local_buffered_not_started=buffered,
            in_flight_jobs=in_flight,
            physical_in_flight=physical,
            finished_not_durable=finished,
            unobserved_nonlocal=unobserved,
            deadline_seconds=self._deadline_seconds,
            effective_concurrency=self._effective_concurrency,
            epoch_elapsed_seconds=elapsed,
            transport_event_histogram=transport_histogram,
            active_censor_histogram=tuple(active_histogram),
            commit_event_histogram=commit_histogram,
            transport_overflow_events=transport_overflow,
            active_overflow_censors=active_overflow,
            commit_overflow_events=commit_overflow,
        )


@dataclass(frozen=True)
class UrlEtaEstimate:
    durable_rate: float | None
    rate_eta: float | None
    queue_eta: float | None
    inflight_eta: float | None
    commit_eta: float
    overflow_eta: float | None
    predicted_remaining_seconds: float | None
    fallback: str | None


def fixed_bin(
    value: float, horizon: float, bin_count: int = TRANSPORT_BIN_COUNT
) -> tuple[int, bool]:
    numeric_value = _require_finite(value, "value")
    numeric_horizon = _require_finite(horizon, "horizon", positive=True)
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValueError("bin_count must be a positive integer")
    overflow = numeric_value >= numeric_horizon
    if overflow:
        return bin_count - 1, True
    edges = tuple(
        index * numeric_horizon / bin_count
        for index in range(bin_count + 1)
    )
    index = bisect_right(edges, max(0.0, numeric_value)) - 1
    return min(bin_count - 1, index), False


def completion_publication_interval(
    total: int,
    requested_interval: int | None = None,
) -> int:
    """Return the bounded production interval for absolute completions."""
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError("total must be a non-negative integer")
    if requested_interval is not None and (
        isinstance(requested_interval, bool)
        or not isinstance(requested_interval, int)
        or requested_interval <= 0
    ):
        raise ValueError("requested_interval must be a positive integer")
    production = max(
        1,
        (total + MAX_URL_COMPLETION_PUBLICATIONS - 1)
        // MAX_URL_COMPLETION_PUBLICATIONS,
    )
    return max(production, requested_interval or production)


def next_completion_milestone(completed: int, interval: int) -> int:
    """Return the first stage-global absolute milestone after completed."""
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
    ):
        raise ValueError("completed must be a non-negative integer")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
    ):
        raise ValueError("interval must be a positive integer")
    return (completed // interval + 1) * interval


def encode_histogram_blob(snapshot: UrlProgressSnapshot) -> str:
    values = (
        *snapshot.transport_event_histogram,
        *snapshot.active_censor_histogram,
        *snapshot.commit_event_histogram,
    )
    return base64.b64encode(_HISTOGRAM_STRUCT.pack(*values)).decode("ascii")


def decode_histogram_blob(
    blob: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if not isinstance(blob, str):
        raise ValueError("histogram blob must be an ASCII string")
    try:
        encoded = blob.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ValueError("histogram blob is not canonical base64") from error
    if len(decoded) != _HISTOGRAM_STRUCT.size:
        raise ValueError(
            f"histogram blob must decode to exactly {_HISTOGRAM_STRUCT.size} bytes"
        )
    if base64.b64encode(decoded) != encoded:
        raise ValueError("histogram blob is not canonical base64")
    values = _HISTOGRAM_STRUCT.unpack(decoded)
    return values[:64], values[64:128], values[128:]


def _kaplan_meier_survival(
    events: tuple[int, ...], censors: tuple[int, ...]
) -> tuple[float, ...]:
    risk = [0] * TRANSPORT_BIN_COUNT
    running = 0
    for index in range(TRANSPORT_BIN_COUNT - 1, -1, -1):
        running += events[index] + censors[index]
        risk[index] = running

    survival = [1.0]
    for index in range(TRANSPORT_BIN_COUNT):
        current = survival[-1]
        if risk[index] > 0:
            current *= 1.0 - events[index] / risk[index]
        survival.append(current)
    return tuple(survival)


def _remaining_rmst(
    age_bin: int,
    survival: tuple[float, ...],
    horizon: float,
) -> float:
    width = horizon / TRANSPORT_BIN_COUNT
    denominator = max(survival[age_bin], 1e-12)
    remaining = sum(
        width * survival[index] / denominator
        for index in range(age_bin, TRANSPORT_BIN_COUNT)
    )
    upper_bound = horizon - age_bin * width
    return min(upper_bound, max(0.0, remaining))


def _histogram_midpoint_mean(
    histogram: tuple[int, ...], horizon: float
) -> float:
    total = sum(histogram)
    if total == 0:
        return horizon / len(histogram)
    width = horizon / len(histogram)
    weighted = sum(
        count * ((index + 0.5) * width)
        for index, count in enumerate(histogram)
    )
    return weighted / total


def estimate_url_eta(snapshot: UrlProgressSnapshot) -> UrlEtaEstimate:
    completed_in_epoch = snapshot.completed_durable - snapshot.baseline_completed
    durable_rate = None
    if completed_in_epoch > 0 and snapshot.epoch_elapsed_seconds > 0.0:
        durable_rate = completed_in_epoch / snapshot.epoch_elapsed_seconds
        if not math.isfinite(durable_rate):
            raise ValueError("durable_rate is not finite")

    remaining_jobs = snapshot.total - snapshot.completed_durable
    rate_eta = remaining_jobs / durable_rate if durable_rate is not None else None
    queued_jobs = (
        snapshot.local_buffered_not_started + snapshot.unobserved_nonlocal
    )
    queue_eta = queued_jobs / durable_rate if durable_rate is not None else None

    commit_horizon = min(2.0, snapshot.deadline_seconds)
    commit_mean = _histogram_midpoint_mean(
        snapshot.commit_event_histogram, commit_horizon
    )
    commit_eta = snapshot.finished_not_durable * commit_mean

    inflight_eta = None
    overflow_eta = None
    fallback = None
    active_bins = [
        index
        for index, count in enumerate(snapshot.active_censor_histogram)
        if count
    ]
    if active_bins:
        oldest_bin = max(active_bins)
        oldest_lower_edge = (
            oldest_bin * snapshot.deadline_seconds / TRANSPORT_BIN_COUNT
        )
        if snapshot.active_overflow_censors > 0:
            inflight_eta = 0.0
            if durable_rate is not None:
                overflow_eta = 2.0 / durable_rate
            fallback = "active_overflow_continuity"
        elif oldest_lower_edge >= snapshot.deadline_seconds * MATURITY_FRACTION:
            survival = _kaplan_meier_survival(
                snapshot.transport_event_histogram,
                snapshot.active_censor_histogram,
            )
            inflight_eta = _remaining_rmst(
                oldest_bin, survival, snapshot.deadline_seconds
            )
        else:
            fallback = "young_active_censor"
    elif sum(snapshot.transport_event_histogram) == 0:
        fallback = "empty_km_risk_set"

    for name, component in (
        ("rate_eta", rate_eta),
        ("queue_eta", queue_eta),
        ("inflight_eta", inflight_eta),
        ("commit_eta", commit_eta),
        ("overflow_eta", overflow_eta),
    ):
        if component is not None and not math.isfinite(component):
            raise ValueError(f"{name} is not finite")

    if snapshot.completed_durable == snapshot.total:
        return UrlEtaEstimate(
            durable_rate=durable_rate,
            rate_eta=0.0 if durable_rate is not None else None,
            queue_eta=0.0 if durable_rate is not None else None,
            inflight_eta=inflight_eta,
            commit_eta=commit_eta,
            overflow_eta=overflow_eta,
            predicted_remaining_seconds=0.0,
            fallback=fallback,
        )

    if durable_rate is None and sum(snapshot.transport_event_histogram) == 0:
        return UrlEtaEstimate(
            durable_rate=None,
            rate_eta=None,
            queue_eta=None,
            inflight_eta=inflight_eta,
            commit_eta=commit_eta,
            overflow_eta=None,
            predicted_remaining_seconds=None,
            fallback="no_durable_rate_or_transport_events",
        )

    available_components = [
        component
        for component in (rate_eta, inflight_eta, commit_eta, overflow_eta)
        if component is not None
    ]
    predicted = max(available_components) if available_components else None
    if predicted is not None and not math.isfinite(predicted):
        raise ValueError("ETA estimate is not finite")
    return UrlEtaEstimate(
        durable_rate=durable_rate,
        rate_eta=rate_eta,
        queue_eta=queue_eta,
        inflight_eta=inflight_eta,
        commit_eta=commit_eta,
        overflow_eta=overflow_eta,
        predicted_remaining_seconds=predicted,
        fallback=fallback,
    )

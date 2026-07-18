from __future__ import annotations

import base64
import binascii
import math
import struct
from dataclasses import dataclass


UINT64_MAX = 2**64 - 1
TRANSPORT_BIN_COUNT = 64
COMMIT_BIN_COUNT = 32
MATURITY_FRACTION = 1.0 / 3.0
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
    index = math.floor(
        bin_count * max(0.0, numeric_value) / numeric_horizon
    )
    return min(bin_count - 1, index), False


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

from __future__ import annotations

import email.utils
import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class MediaPolicyConfig:
    workers: int = 2
    max_mbps: float = 24.0
    max_retries: int = 5
    retry_base_seconds: float = 5.0
    retry_max_seconds: float = 60.0
    chunk_bytes: int = 128 * 1024

    def validate(self) -> "MediaPolicyConfig":
        if not 1 <= self.workers <= 2:
            raise ValueError("media workers must be in the range 1..2")
        if not 0.0 < self.max_mbps <= 25.0:
            raise ValueError("media max_mbps must be in the range (0, 25]")
        if self.max_retries < 0:
            raise ValueError("media max_retries must be non-negative")
        if self.retry_base_seconds < 0 or self.retry_max_seconds < 0:
            raise ValueError("media retry delays must be non-negative")
        if self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError(
                "media retry_max_seconds must be >= retry_base_seconds"
            )
        if self.chunk_bytes <= 0:
            raise ValueError("media chunk_bytes must be positive")
        return self


@dataclass
class MediaStats:
    _counters: dict[str, int | float] = field(default_factory=dict, init=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)

    def increment(self, name: str, amount: int | float = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def summary(self) -> dict[str, int | float]:
        with self._lock:
            return dict(self._counters)


def parse_retry_after(
    value: str | None, *, now: datetime | None = None
) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (parsed - current).total_seconds())


class MediaBandwidthLimiter:
    def __init__(
        self,
        *,
        max_mbps: float,
        capacity_bytes: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            normalized_max_mbps = float(max_mbps)
        except (TypeError, ValueError) as error:
            raise ValueError("media max_mbps must be finite and positive") from error
        if not math.isfinite(normalized_max_mbps) or normalized_max_mbps <= 0:
            raise ValueError("media max_mbps must be finite and positive")

        try:
            normalized_capacity = float(capacity_bytes)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "media capacity_bytes must be finite and positive"
            ) from error
        if not math.isfinite(normalized_capacity) or normalized_capacity <= 0:
            raise ValueError("media capacity_bytes must be finite and positive")

        self.rate_bytes_per_second = normalized_max_mbps * 1_000_000 / 8
        self.capacity_bytes = normalized_capacity
        self._tokens = normalized_capacity
        self._updated_at = clock()
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def acquire(self, byte_count: int) -> float:
        waited = 0.0
        remaining = float(byte_count)
        while remaining > 0:
            with self._lock:
                now = self._clock()
                elapsed = max(0.0, now - self._updated_at)
                self._tokens = min(
                    self.capacity_bytes,
                    self._tokens + elapsed * self.rate_bytes_per_second,
                )
                self._updated_at = now
                granted = min(remaining, self._tokens)
                self._tokens -= granted
                remaining -= granted
                delay = (
                    0.0
                    if remaining <= 0
                    else min(remaining, self.capacity_bytes)
                    / self.rate_bytes_per_second
                )
            if delay > 0:
                self._sleep(delay)
                waited += delay
        return waited


class MediaCooldown:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._blocked_until = 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def extend(self, delay_seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(
                self._blocked_until, self._clock() + delay_seconds
            )

    def wait(self) -> float:
        waited = 0.0
        while True:
            with self._lock:
                delay = self._blocked_until - self._clock()
            if delay <= 0:
                return waited
            self._sleep(delay)
            waited += delay


class MediaHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        headers: Mapping[str, Any],
        body_excerpt: str,
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers)
        self.body_excerpt = body_excerpt
        message = f"HTTP {status_code}"
        if body_excerpt:
            message = f"{message}: {body_excerpt}"
        super().__init__(message)


class NonImageMediaError(Exception):
    def __init__(self, content_type: str) -> None:
        self.content_type = " ".join(str(content_type).split())
        super().__init__(self.content_type)


@dataclass(frozen=True)
class RetryDecision:
    retryable: bool
    shared_cooldown: bool
    delay_seconds: float
    reason: str


def _retry_after_header(headers: Mapping[str, Any]) -> str | None:
    for name, value in headers.items():
        if str(name).lower() == "retry-after":
            return str(value)
    return None


def _backoff_delay(
    *,
    attempt: int,
    config: MediaPolicyConfig,
    jitter: float,
) -> float:
    return min(
        config.retry_base_seconds * 2**attempt,
        config.retry_max_seconds,
    ) + jitter


def _is_connection_or_timeout(error: Exception) -> bool:
    network_exception_names = {
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectTimeout",
        "ProxyError",
        "ReadTimeout",
        "SSLError",
        "Timeout",
        "TimeoutError",
        "URLError",
    }
    return any(
        error_type.__name__ in network_exception_names
        for error_type in type(error).__mro__
    )


def classify_media_failure(
    error: Exception,
    *,
    attempt: int,
    config: MediaPolicyConfig,
    jitter: float,
) -> RetryDecision:
    if isinstance(error, MediaHTTPError):
        body = error.body_excerpt.lower()
        if "thumbnail steps" in body or "standard thumbnail size" in body:
            return RetryDecision(False, False, 0.0, "thumbnail_configuration")

        retry_after = parse_retry_after(_retry_after_header(error.headers))
        if error.status_code == 429:
            delay = retry_after
            if delay is None:
                delay = _backoff_delay(
                    attempt=attempt,
                    config=config,
                    jitter=jitter,
                )
            return RetryDecision(True, True, delay, "http_429")

        if error.status_code == 503 and retry_after is not None:
            return RetryDecision(True, True, retry_after, "http_503_retry_after")

        if error.status_code in {408, 500, 502, 503, 504}:
            delay = _backoff_delay(
                attempt=attempt,
                config=config,
                jitter=jitter,
            )
            return RetryDecision(
                True, False, delay, f"http_{error.status_code}"
            )

        if 400 <= error.status_code < 500:
            return RetryDecision(
                False, False, 0.0, f"http_{error.status_code}_terminal"
            )

        return RetryDecision(False, False, 0.0, "http_terminal")

    if _is_connection_or_timeout(error):
        delay = _backoff_delay(
            attempt=attempt,
            config=config,
            jitter=jitter,
        )
        return RetryDecision(True, False, delay, "connection_or_timeout")

    return RetryDecision(False, False, 0.0, "terminal_error")


class MediaFailureRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._count = 0

    def reset(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")
            self._count = 0

    def record(
        self,
        context: Mapping[str, Any],
        error: Exception,
        attempts: int,
        retry_delays: list[float],
    ) -> None:
        headers = getattr(error, "headers", {})
        retry_after_present = (
            bool(_retry_after_header(headers))
            if isinstance(headers, Mapping)
            else False
        )
        record = {
            **context,
            "timestamp": time.time(),
            "attempts": attempts,
            "retry_delays": list(retry_delays),
            "status_code": getattr(error, "status_code", None),
            "exception_class": type(error).__name__,
            "retry_after_present": retry_after_present,
            "error": str(error)[:500],
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


@dataclass
class _Flight:
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class WikimediaMediaDownloader:
    def __init__(
        self,
        *,
        config: MediaPolicyConfig,
        failure_recorder: Any,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = lambda: 0.0,
    ) -> None:
        self.config = config.validate()
        self.stats = MediaStats()
        self.bandwidth = MediaBandwidthLimiter(
            max_mbps=config.max_mbps,
            capacity_bytes=config.chunk_bytes,
            clock=clock,
            sleep=sleep,
        )
        self.cooldown = MediaCooldown(clock=clock, sleep=sleep)
        self.failure_recorder = failure_recorder
        self._media_slots = threading.BoundedSemaphore(config.workers)
        self._jitter = jitter
        self._sleep = sleep
        self._flights: dict[str, _Flight] = {}
        self._flights_lock = threading.Lock()

    def _get_or_create_flight(self, cache_key: str) -> tuple[_Flight, bool]:
        with self._flights_lock:
            existing = self._flights.get(cache_key)
            if existing is not None:
                return existing, False
            created = _Flight()
            self._flights[cache_key] = created
            return created, True

    def run(
        self,
        *,
        cache_key: str,
        attempt: Callable[[MediaBandwidthLimiter], dict[str, Any]],
        failure_context: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        flight, leader = self._get_or_create_flight(cache_key)
        if not leader:
            self.stats.increment("singleflight_followers")
            flight.done.wait()
            return flight.result

        delays: list[float] = []
        try:
            for attempt_index in range(self.config.max_retries + 1):
                cooldown_wait = self.cooldown.wait()
                self.stats.increment("cooldown_wait_seconds", cooldown_wait)
                attempt_error: Exception | None = None
                decision: RetryDecision | None = None
                should_retry = False
                with self._media_slots:
                    cooldown_wait = self.cooldown.wait()
                    self.stats.increment("cooldown_wait_seconds", cooldown_wait)
                    try:
                        result = attempt(self.bandwidth)
                    except Exception as error:
                        attempt_error = error
                        decision = classify_media_failure(
                            error,
                            attempt=attempt_index,
                            config=self.config,
                            jitter=self._jitter(),
                        )
                        should_retry = (
                            decision.retryable
                            and attempt_index < self.config.max_retries
                        )
                        if should_retry and decision.shared_cooldown:
                            self.cooldown.extend(decision.delay_seconds)
                            self.stats.increment("shared_cooldown_events")
                if attempt_error is not None:
                    assert decision is not None
                    if not should_retry:
                        self.failure_recorder.record(
                            failure_context,
                            attempt_error,
                            attempt_index + 1,
                            delays,
                        )
                        self.stats.increment("terminal_failures")
                        flight.result = None
                        return None
                    delays.append(decision.delay_seconds)
                    self.stats.increment("retry_attempts")
                    if not decision.shared_cooldown:
                        self._sleep(decision.delay_seconds)
                    continue
                flight.result = result
                self.stats.increment("successful_downloads")
                return result
            return None
        finally:
            flight.done.set()

    def summary(self) -> dict[str, int | float]:
        summary = self.stats.summary()
        summary.setdefault("shared_cooldown_events", 0)
        summary["failure_records"] = self.failure_recorder.count
        return summary

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import wikimedia_media

from wikimedia_media import (
    MediaBandwidthLimiter,
    MediaCooldown,
    MediaHTTPError,
    MediaPolicyConfig,
    MediaStats,
    NonImageMediaError,
    classify_media_failure,
    parse_retry_after,
)


class _DiscardingFailureRecorder:
    count = 0

    def record(self, context, error, attempts, retry_delays):
        self.count += 1


def make_downloader(tmp_path, **overrides):
    values = {
        "workers": 2,
        "max_mbps": 24.0,
        "max_retries": 0,
        "retry_base_seconds": 0.0,
        "retry_max_seconds": 0.0,
        "chunk_bytes": 128 * 1024,
    }
    values.update(overrides)
    recorder = wikimedia_media.MediaFailureRecorder(tmp_path / "failures.jsonl")
    recorder.reset()
    return wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(**values),
        failure_recorder=recorder,
        jitter=lambda: 0.0,
    )


def test_media_policy_config_rejects_values_outside_wikimedia_limits():
    MediaPolicyConfig(workers=2, max_mbps=24.0).validate()
    with pytest.raises(ValueError, match="workers"):
        MediaPolicyConfig(workers=3).validate()
    with pytest.raises(ValueError, match="max_mbps"):
        MediaPolicyConfig(max_mbps=25.01).validate()


def test_parse_retry_after_supports_seconds_and_http_date():
    now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
    assert parse_retry_after("7", now=now) == 7.0
    assert parse_retry_after("Mon, 13 Jul 2026 12:00:11 GMT", now=now) == 11.0
    assert parse_retry_after("invalid", now=now) is None


def test_media_stats_accumulates_counters_and_returns_a_snapshot():
    stats = MediaStats()
    stats.increment("downloaded")
    stats.increment("bytes", 2.5)

    summary = stats.summary()

    assert summary == {"downloaded": 1, "bytes": 2.5}
    summary["downloaded"] = 99
    assert stats.summary()["downloaded"] == 1


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ControlledTime:
    def __init__(self):
        self.now = 0.0
        self.sleeping = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    def monotonic(self):
        with self._lock:
            return self.now

    def sleep(self, seconds):
        self.sleeping.set()
        assert self.release.wait(timeout=2)
        with self._lock:
            self.now += seconds


def test_bandwidth_limiter_paces_aggregate_bytes_across_callers():
    fake = FakeTime()
    limiter = MediaBandwidthLimiter(
        max_mbps=8.0,
        capacity_bytes=100_000,
        clock=fake.monotonic,
        sleep=fake.sleep,
    )
    limiter.acquire(100_000)
    limiter.acquire(100_000)
    assert sum(fake.sleeps) == pytest.approx(0.1)


def test_bandwidth_limiter_paces_requests_larger_than_bucket_capacity():
    fake = FakeTime()
    limiter = MediaBandwidthLimiter(
        max_mbps=8.0,
        capacity_bytes=100_000,
        clock=fake.monotonic,
        sleep=fake.sleep,
    )

    waited = limiter.acquire(300_000)

    assert waited == pytest.approx(0.2)
    assert sum(fake.sleeps) == pytest.approx(0.2)


@pytest.mark.parametrize("max_mbps", [0.0, -1.0, float("inf"), float("nan")])
def test_bandwidth_limiter_rejects_non_positive_or_non_finite_rate(max_mbps):
    with pytest.raises(ValueError, match="max_mbps"):
        MediaBandwidthLimiter(max_mbps=max_mbps, capacity_bytes=100_000)


@pytest.mark.parametrize(
    "capacity_bytes", [0, -1, float("inf"), float("nan")]
)
def test_bandwidth_limiter_rejects_non_positive_or_non_finite_capacity(
    capacity_bytes,
):
    with pytest.raises(ValueError, match="capacity_bytes"):
        MediaBandwidthLimiter(max_mbps=8.0, capacity_bytes=capacity_bytes)


def test_shared_cooldown_uses_latest_deadline():
    fake = FakeTime()
    cooldown = MediaCooldown(clock=fake.monotonic, sleep=fake.sleep)
    cooldown.extend(4.0)
    cooldown.extend(7.0)
    cooldown.wait()
    assert sum(fake.sleeps) == pytest.approx(7.0)


def test_failure_classifier_distinguishes_shared_and_per_item_retries():
    config = MediaPolicyConfig(retry_base_seconds=1.0, retry_max_seconds=3.0)

    rate_limited = classify_media_failure(
        MediaHTTPError(429, {"Retry-After": "9"}, "request limit"),
        attempt=0,
        config=config,
        jitter=0.25,
    )
    assert rate_limited.retryable is True
    assert rate_limited.shared_cooldown is True
    assert rate_limited.delay_seconds == 9.0

    unavailable = classify_media_failure(
        MediaHTTPError(503, {}, "unavailable"),
        attempt=2,
        config=config,
        jitter=0.25,
    )
    assert unavailable.retryable is True
    assert unavailable.shared_cooldown is False
    assert unavailable.delay_seconds == 5.25

    connection = classify_media_failure(
        ConnectionError("reset"),
        attempt=2,
        config=config,
        jitter=0.25,
    )
    assert connection.retryable is True
    assert connection.shared_cooldown is False
    assert connection.delay_seconds == 3.25


@pytest.mark.parametrize(
    ("status_code", "expected_shared"),
    [(429, True), (503, False)],
)
def test_429_and_503_without_retry_after_wait_at_least_five_seconds(
    status_code,
    expected_shared,
):
    config = MediaPolicyConfig(
        retry_base_seconds=0.0,
        retry_max_seconds=0.0,
    )

    decision = classify_media_failure(
        MediaHTTPError(status_code, {}, "server busy"),
        attempt=0,
        config=config,
        jitter=0.0,
    )

    assert decision.retryable is True
    assert decision.shared_cooldown is expected_shared
    assert decision.delay_seconds == 5.0


def test_five_second_floor_does_not_apply_to_other_per_item_backoff():
    decision = classify_media_failure(
        ConnectionError("reset"),
        attempt=0,
        config=MediaPolicyConfig(
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        jitter=0.0,
    )

    assert decision.delay_seconds == 0.0


def test_failure_classifier_treats_configuration_and_other_4xx_as_terminal():
    config = MediaPolicyConfig()

    thumbnail = classify_media_failure(
        MediaHTTPError(429, {"Retry-After": "10"}, "Use thumbnail steps"),
        attempt=0,
        config=config,
        jitter=0.0,
    )
    assert thumbnail.retryable is False
    assert thumbnail.shared_cooldown is False
    assert thumbnail.delay_seconds == 0.0

    missing = classify_media_failure(
        MediaHTTPError(404, {}, "not found"),
        attempt=0,
        config=config,
        jitter=0.0,
    )
    assert missing.retryable is False
    assert missing.shared_cooldown is False
    assert missing.delay_seconds == 0.0


def test_non_image_media_error_message_is_only_sanitized_content_type():
    error = NonImageMediaError("  text/html\r\n charset=utf-8  ")
    assert str(error) == "text/html charset=utf-8"


def test_connection_error_retries_only_current_media(tmp_path):
    calls = []

    def attempt(limiter):
        calls.append("attempt")
        if len(calls) == 1:
            raise ConnectionError("reset")
        return {"local_path": str(tmp_path / "ok.jpg"), "bytes": 3}

    downloader = make_downloader(
        tmp_path,
        max_retries=1,
        retry_base_seconds=0,
    )
    result = downloader.run(
        cache_key="url-a",
        attempt=attempt,
        failure_context={"url": "a"},
    )

    assert result["bytes"] == 3
    assert calls == ["attempt", "attempt"]
    assert downloader.summary()["shared_cooldown_events"] == 0


def test_retry_exhaustion_records_failure_without_negative_cache(tmp_path):
    downloader = make_downloader(
        tmp_path,
        max_retries=1,
        retry_base_seconds=0,
    )

    result = downloader.run(
        cache_key="bad-url",
        attempt=lambda limiter: (_ for _ in ()).throw(TimeoutError("slow")),
        failure_context={
            "url": "https://upload.wikimedia.org/bad.jpg",
            "asset_id": "a1",
        },
    )

    assert result is None
    records = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl").read_text().splitlines()
    ]
    assert records[0]["attempts"] == 2
    assert records[0]["asset_id"] == "a1"
    assert records[0]["exception_class"] == "TimeoutError"
    assert downloader.summary()["failure_records"] == 1


@pytest.mark.parametrize("status_code", [429, 503])
def test_retry_after_extends_shared_cooldown_after_budget_is_exhausted(
    tmp_path,
    status_code,
):
    fake = FakeTime()
    downloader = wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(
            max_retries=0,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        failure_recorder=_DiscardingFailureRecorder(),
        clock=fake.monotonic,
        sleep=fake.sleep,
        jitter=lambda: 0.0,
    )

    result = downloader.run(
        cache_key=f"status-{status_code}",
        attempt=lambda limiter: (_ for _ in ()).throw(
            MediaHTTPError(status_code, {"Retry-After": "7"}, "server busy")
        ),
        failure_context={"url": "rate-limited"},
    )

    assert result is None
    assert downloader.cooldown.wait() == 7.0
    summary = downloader.summary()
    assert summary["shared_cooldown_events"] == 1
    assert summary[f"http_{status_code}_responses"] == 1


def test_http_response_counts_are_per_attempt_without_duplicates(tmp_path):
    fake = FakeTime()
    downloader = wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(
            max_retries=2,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        failure_recorder=_DiscardingFailureRecorder(),
        clock=fake.monotonic,
        sleep=fake.sleep,
        jitter=lambda: 0.0,
    )
    errors = [
        MediaHTTPError(429, {}, "request limit"),
        MediaHTTPError(503, {}, "unavailable"),
    ]

    def attempt(limiter):
        if errors:
            raise errors.pop(0)
        return {"local_path": "ok.jpg", "bytes": 3}

    result = downloader.run(
        cache_key="eventual-success",
        attempt=attempt,
        failure_context={"url": "eventual-success"},
    )

    assert result["bytes"] == 3
    summary = downloader.summary()
    assert summary["http_429_responses"] == 1
    assert summary["http_503_responses"] == 1
    assert summary["retry_attempts"] == 2


def test_downloader_summary_always_exposes_all_manifest_metrics(tmp_path):
    downloader = make_downloader(tmp_path)

    assert downloader.summary() == {
        "legacy_cache_hits": 0,
        "shared_cache_hits": 0,
        "singleflight_followers": 0,
        "successful_downloads": 0,
        "downloaded_bytes": 0,
        "retry_attempts": 0,
        "http_429_responses": 0,
        "http_503_responses": 0,
        "bandwidth_wait_seconds": 0.0,
        "cooldown_wait_seconds": 0.0,
        "shared_cooldown_events": 0,
        "terminal_failures": 0,
        "failure_records": 0,
    }


def test_shared_cooldown_blocks_a_second_new_attempt(tmp_path):
    fake = ControlledTime()
    first_calls = 0
    second_started = threading.Event()

    def first_attempt(limiter):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            raise MediaHTTPError(429, {"Retry-After": "5"}, "request limit")
        return {"local_path": "first.jpg", "bytes": 3}

    def second_attempt(limiter):
        second_started.set()
        return {"local_path": "second.jpg", "bytes": 4}

    downloader = wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(
            workers=2,
            max_mbps=24.0,
            max_retries=1,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        failure_recorder=_DiscardingFailureRecorder(),
        clock=fake.monotonic,
        sleep=fake.sleep,
        jitter=lambda: 0.0,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            downloader.run,
            cache_key="first",
            attempt=first_attempt,
            failure_context={"url": "first"},
        )
        assert fake.sleeping.wait(timeout=2)
        second = pool.submit(
            downloader.run,
            cache_key="second",
            attempt=second_attempt,
            failure_context={"url": "second"},
        )
        assert not second_started.wait(timeout=0.05)
        fake.release.set()
        assert first.result(timeout=2)["bytes"] == 3
        assert second.result(timeout=2)["bytes"] == 4


def test_queued_waiter_rechecks_cooldown_before_starting_request(
    tmp_path,
    monkeypatch,
):
    fake = ControlledTime()
    first_started = threading.Event()
    allow_rate_limit = threading.Event()
    queued_checked_cooldown = threading.Event()
    classification_started = threading.Event()
    allow_classification = threading.Event()
    queued_attempt_started = threading.Event()
    first_calls = 0
    results = {}
    errors = []

    downloader = wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(
            workers=1,
            max_mbps=24.0,
            max_retries=1,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        failure_recorder=_DiscardingFailureRecorder(),
        clock=fake.monotonic,
        sleep=fake.sleep,
        jitter=lambda: 0.0,
    )

    original_wait = downloader.cooldown.wait

    def observed_wait():
        if threading.current_thread().name == "queued-worker":
            queued_checked_cooldown.set()
        return original_wait()

    downloader.cooldown.wait = observed_wait

    original_classifier = wikimedia_media.classify_media_failure

    def blocking_classifier(*args, **kwargs):
        classification_started.set()
        assert allow_classification.wait(timeout=2)
        return original_classifier(*args, **kwargs)

    monkeypatch.setattr(
        wikimedia_media,
        "classify_media_failure",
        blocking_classifier,
    )

    def rate_limited_attempt(limiter):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            first_started.set()
            assert allow_rate_limit.wait(timeout=2)
            raise MediaHTTPError(429, {"Retry-After": "5"}, "request limit")
        return {"local_path": "retried.jpg", "bytes": 3}

    def queued_attempt(limiter):
        queued_attempt_started.set()
        return {"local_path": "queued.jpg", "bytes": 4}

    def run(name, attempt):
        try:
            results[name] = downloader.run(
                cache_key=name,
                attempt=attempt,
                failure_context={"url": name},
            )
        except Exception as error:
            errors.append(error)

    rate_limited = threading.Thread(
        target=run,
        args=("rate-limited", rate_limited_attempt),
        name="rate-limited-worker",
    )
    queued = threading.Thread(
        target=run,
        args=("queued", queued_attempt),
        name="queued-worker",
    )
    rate_limited.start()
    assert first_started.wait(timeout=2)
    queued.start()
    assert queued_checked_cooldown.wait(timeout=2)
    allow_rate_limit.set()
    assert classification_started.wait(timeout=2)

    started_before_cooldown_was_published = queued_attempt_started.wait(
        timeout=0.1
    )
    allow_classification.set()
    if not started_before_cooldown_was_published:
        assert fake.sleeping.wait(timeout=2)
        assert not queued_attempt_started.is_set()
    fake.release.set()
    rate_limited.join(timeout=2)
    queued.join(timeout=2)

    assert not rate_limited.is_alive()
    assert not queued.is_alive()
    assert errors == []
    assert results["rate-limited"]["bytes"] == 3
    assert results["queued"]["bytes"] == 4
    assert started_before_cooldown_was_published is False


def test_shared_cooldown_does_not_cancel_active_success(tmp_path):
    fake = ControlledTime()
    active_started = threading.Event()
    allow_active_finish = threading.Event()
    rate_limited_calls = 0

    def active_attempt(limiter):
        active_started.set()
        assert allow_active_finish.wait(timeout=2)
        return {"local_path": "active.jpg", "bytes": 7}

    def rate_limited_attempt(limiter):
        nonlocal rate_limited_calls
        rate_limited_calls += 1
        if rate_limited_calls == 1:
            raise MediaHTTPError(429, {"Retry-After": "5"}, "request limit")
        return {"local_path": "retried.jpg", "bytes": 2}

    downloader = wikimedia_media.WikimediaMediaDownloader(
        config=MediaPolicyConfig(
            workers=2,
            max_mbps=24.0,
            max_retries=1,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        failure_recorder=_DiscardingFailureRecorder(),
        clock=fake.monotonic,
        sleep=fake.sleep,
        jitter=lambda: 0.0,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(
            downloader.run,
            cache_key="active",
            attempt=active_attempt,
            failure_context={"url": "active"},
        )
        assert active_started.wait(timeout=2)
        rate_limited = pool.submit(
            downloader.run,
            cache_key="rate-limited",
            attempt=rate_limited_attempt,
            failure_context={"url": "rate-limited"},
        )
        assert fake.sleeping.wait(timeout=2)
        allow_active_finish.set()
        assert active.result(timeout=2)["bytes"] == 7
        fake.release.set()
        assert rate_limited.result(timeout=2)["bytes"] == 2


def test_singleflight_runs_one_attempt_for_duplicate_key(tmp_path):
    calls = 0
    lock = threading.Lock()

    def attempt(limiter):
        nonlocal calls
        with lock:
            calls += 1
        return {"local_path": "shared.jpg", "bytes": 3}

    downloader = make_downloader(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: downloader.run(
                    cache_key="same",
                    attempt=attempt,
                    failure_context={"url": "same"},
                ),
                range(2),
            )
        )

    assert calls == 1
    assert results[0] == results[1]
    assert downloader.summary()["singleflight_followers"] == 1


def test_terminal_failure_is_skipped_this_run_but_retryable_next_run(tmp_path):
    calls = 0

    def attempt(limiter):
        nonlocal calls
        calls += 1
        raise MediaHTTPError(404, {}, "not found")

    downloader = make_downloader(tmp_path)
    assert (
        downloader.run(
            cache_key="missing",
            attempt=attempt,
            failure_context={"url": "missing"},
        )
        is None
    )
    assert (
        downloader.run(
            cache_key="missing",
            attempt=attempt,
            failure_context={"url": "missing"},
        )
        is None
    )
    assert calls == 1

    next_run = make_downloader(tmp_path)
    assert (
        next_run.run(
            cache_key="missing",
            attempt=attempt,
            failure_context={"url": "missing"},
        )
        is None
    )
    assert calls == 2

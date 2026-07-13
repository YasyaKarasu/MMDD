import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

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
    assert unavailable.delay_seconds == 3.25

    connection = classify_media_failure(
        ConnectionError("reset"),
        attempt=2,
        config=config,
        jitter=0.25,
    )
    assert connection.retryable is True
    assert connection.shared_cooldown is False
    assert connection.delay_seconds == 3.25


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

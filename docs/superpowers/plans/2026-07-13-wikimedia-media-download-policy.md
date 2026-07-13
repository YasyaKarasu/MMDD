# Wikimedia Media Download Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Wikimedia image downloads policy-compliant, retryable, resumable, bandwidth-bounded, and URL-deduplicated without invalidating existing image or model caches.

**Architecture:** Keep the Action API limiter in `WikipediaClient` and add a focused `wikimedia_media.py` component for aggregate media concurrency, bandwidth pacing, shared server cooldowns, retry classification, single-flight coordination, failure recording, and counters. `WikipediaClient.download_image()` remains the compatibility boundary for URL selection and SVG finalization, while both dataset builders supply validated media configuration and publish downloader statistics.

**Tech Stack:** Python 3.10+, `requests`, `threading`, `time.monotonic`, `pathlib.Path`, JSONL, `pytest`.

## Global Constraints

- Preserve every non-empty legacy image file named from an entity-specific `asset_id`.
- Action API traffic remains serial and below five requests per second; `--sleep` affects only that path.
- Media concurrency is configurable only in the range `1..2` and defaults to `2`.
- Aggregate media bandwidth defaults to `24.0` Mbps and may not exceed `25.0` Mbps.
- The standard requested thumbnail width is `960` pixels.
- A request-volume 429 shares its `Retry-After` cooldown across media workers, but an existing HTTP 200 stream may finish.
- Per-item connection failures do not pause unrelated media items.
- Exhausted failures are recorded and skipped for this run, with no persistent negative cache, so the next run retries them.
- Existing dirty changes in `scripts/build_mm_table_dataset.py`, `scripts/build_mm_joinability_dataset.py`, and `tests/test_stage1_pipeline.py` belong to the user and must be preserved.
- Tests make no live Wikimedia requests and use `tmp_path`, fake responses, fake clocks, and fake sleepers.

---

### Task 1: Media policy primitives

**Files:**
- Create: `scripts/wikimedia_media.py`
- Create: `tests/test_wikimedia_media.py`

**Interfaces:**
- Produces: `MediaPolicyConfig`, `MediaStats`, `MediaBandwidthLimiter`, `MediaCooldown`, `parse_retry_after()`, `classify_media_failure()`, `MediaHTTPError`, and `RetryDecision`.
- Consumes: only Python standard-library modules in this task.

- [ ] **Step 1: Write failing configuration and Retry-After tests**

Add these tests to `tests/test_wikimedia_media.py`:

```python
from datetime import datetime, timezone

import pytest

from wikimedia_media import MediaPolicyConfig, parse_retry_after


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
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py::test_media_policy_config_rejects_values_outside_wikimedia_limits tests/test_wikimedia_media.py::test_parse_retry_after_supports_seconds_and_http_date -q
```

Expected: collection fails because `wikimedia_media` does not exist.

- [ ] **Step 3: Implement validated configuration and Retry-After parsing**

Create `scripts/wikimedia_media.py` with these public definitions:

```python
from __future__ import annotations

import email.utils
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
            raise ValueError("media retry_max_seconds must be >= retry_base_seconds")
        if self.chunk_bytes <= 0:
            raise ValueError("media chunk_bytes must be positive")
        return self


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
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
```

Also define `MediaStats` as a lock-protected counter with `increment(name, amount=1)` and `summary() -> dict[str, int | float]`.

- [ ] **Step 4: Write failing aggregate-bandwidth and cooldown tests**

Append tests that inject a fake monotonic clock and sleeper:

```python
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
```

- [ ] **Step 5: Implement the token bucket and shared cooldown**

Implement:

```python
class MediaBandwidthLimiter:
    def __init__(self, *, max_mbps, capacity_bytes, clock=time.monotonic, sleep=time.sleep):
        self.rate_bytes_per_second = max_mbps * 1_000_000 / 8
        self.capacity_bytes = float(capacity_bytes)
        self._tokens = float(capacity_bytes)
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
                delay = 0.0 if remaining <= 0 else remaining / self.rate_bytes_per_second
            if delay > 0:
                self._sleep(delay)
                waited += delay
        return waited


class MediaCooldown:
    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self._blocked_until = 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    def extend(self, delay_seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + delay_seconds)

    def wait(self) -> float:
        waited = 0.0
        while True:
            with self._lock:
                delay = self._blocked_until - self._clock()
            if delay <= 0:
                return waited
            self._sleep(delay)
            waited += delay
```

Implement `MediaHTTPError(status_code, headers, body_excerpt)`, a frozen
`RetryDecision(retryable, shared_cooldown, delay_seconds, reason)`, and
`classify_media_failure(error, attempt, config, jitter)` with these exact rules:

- request-volume 429: retryable, shared cooldown;
- body containing `thumbnail steps` or `standard thumbnail size`: terminal;
- 503 with `Retry-After`: retryable, shared cooldown;
- 408/500/502/503/504 and connection/timeout exceptions: per-item retry;
- other 4xx: terminal;
- missing `Retry-After`: `min(base * 2**attempt, max) + jitter`, with a minimum five seconds for 429/503 in production configuration.

Also define `NonImageMediaError(content_type)` as a terminal exception whose
message contains only the sanitized content type.

- [ ] **Step 6: Run Task 1 tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add scripts/wikimedia_media.py tests/test_wikimedia_media.py
git commit -m "Add Wikimedia media policy primitives"
```

---

### Task 2: Retry coordinator, single-flight, and failure records

**Files:**
- Modify: `scripts/wikimedia_media.py`
- Modify: `tests/test_wikimedia_media.py`

**Interfaces:**
- Consumes: Task 1's `MediaPolicyConfig`, limiter, cooldown, failure classifier, and stats.
- Produces: `WikimediaMediaDownloader.run(cache_key, attempt, failure_context) -> dict[str, Any] | None`, `MediaFailureRecorder`, and `WikimediaMediaDownloader.summary()`.

- [ ] **Step 1: Write failing retry-scope tests**

Add fake-attempt tests with two threads and synchronization events. The 429 test must assert that a second new attempt starts only after the first attempt's shared cooldown expires; the active-success test must assert that an already-running success is not cancelled. Add this per-item test:

```python
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
    recorder = MediaFailureRecorder(tmp_path / "failures.jsonl")
    recorder.reset()
    return WikimediaMediaDownloader(
        config=MediaPolicyConfig(**values),
        failure_recorder=recorder,
        jitter=lambda: 0.0,
    )


def test_connection_error_retries_only_current_media(tmp_path):
    calls = []

    def attempt(limiter):
        calls.append("attempt")
        if len(calls) == 1:
            raise ConnectionError("reset")
        return {"local_path": str(tmp_path / "ok.jpg"), "bytes": 3}

    downloader = make_downloader(tmp_path, max_retries=1, retry_base_seconds=0)
    result = downloader.run(cache_key="url-a", attempt=attempt, failure_context={"url": "a"})
    assert result["bytes"] == 3
    assert calls == ["attempt", "attempt"]
    assert downloader.summary()["shared_cooldown_events"] == 0
```

- [ ] **Step 2: Run retry tests and verify failure**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py -k 'connection_error or shared_cooldown or active_success' -q
```

Expected: failure because `WikimediaMediaDownloader` is not implemented.

- [ ] **Step 3: Implement bounded retry orchestration**

Add:

```python
class WikimediaMediaDownloader:
    def __init__(
        self,
        *,
        config: MediaPolicyConfig,
        failure_recorder: "MediaFailureRecorder",
        clock=time.monotonic,
        sleep=time.sleep,
        jitter=lambda: 0.0,
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

    def _get_or_create_flight(self, cache_key: str) -> tuple["_Flight", bool]:
        with self._flights_lock:
            existing = self._flights.get(cache_key)
            if existing is not None:
                return existing, False
            created = _Flight()
            self._flights[cache_key] = created
            return created, True

    def run(self, *, cache_key, attempt, failure_context):
        flight, leader = self._get_or_create_flight(cache_key)
        if not leader:
            self.stats.increment("singleflight_followers")
            flight.done.wait()
            return flight.result
        delays = []
        try:
            for attempt_index in range(self.config.max_retries + 1):
                cooldown_wait = self.cooldown.wait()
                self.stats.increment("cooldown_wait_seconds", cooldown_wait)
                try:
                    with self._media_slots:
                        result = attempt(self.bandwidth)
                except Exception as error:
                    decision = classify_media_failure(
                        error,
                        attempt=attempt_index,
                        config=self.config,
                        jitter=self._jitter(),
                    )
                    if not decision.retryable or attempt_index >= self.config.max_retries:
                        self.failure_recorder.record(failure_context, error, attempt_index + 1, delays)
                        self.stats.increment("terminal_failures")
                        flight.result = None
                        return None
                    delays.append(decision.delay_seconds)
                    self.stats.increment("retry_attempts")
                    if decision.shared_cooldown:
                        self.cooldown.extend(decision.delay_seconds)
                        self.stats.increment("shared_cooldown_events")
                    else:
                        self._sleep(decision.delay_seconds)
                    continue
                flight.result = result
                self.stats.increment("successful_downloads")
                return result
        finally:
            flight.done.set()
```

Define the flight state as:

```python
@dataclass
class _Flight:
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
```

Keep completed flights in memory until process exit so a terminal failure
cannot be retried by another entity during the same run.

- [ ] **Step 4: Write failing failure-recorder and single-flight tests**

Add tests asserting:

```python
def test_retry_exhaustion_records_failure_without_negative_cache(tmp_path):
    downloader = make_downloader(tmp_path, max_retries=1, retry_base_seconds=0)
    result = downloader.run(
        cache_key="bad-url",
        attempt=lambda limiter: (_ for _ in ()).throw(TimeoutError("slow")),
        failure_context={"url": "https://upload.wikimedia.org/bad.jpg", "asset_id": "a1"},
    )
    assert result is None
    records = [json.loads(line) for line in (tmp_path / "failures.jsonl").read_text().splitlines()]
    assert records[0]["attempts"] == 2
    assert records[0]["asset_id"] == "a1"


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
        results = list(pool.map(lambda _: downloader.run(
            cache_key="same", attempt=attempt, failure_context={"url": "same"}
        ), range(2)))
    assert calls == 1
    assert results[0] == results[1]
```

- [ ] **Step 5: Implement thread-safe JSONL failure recording and summary**

`MediaFailureRecorder(path)` opens no persistent handle. Its `reset()` creates
the parent directory and writes an empty UTF-8 file once at build startup.
`record()` appends under a lock with `json.dumps(..., ensure_ascii=False)`,
including `timestamp`, `attempts`, `retry_delays`, `status_code` or
`exception_class`, and an error excerpt truncated to 500 characters.

```python
class MediaFailureRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._count = 0

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._count = 0

    def record(self, context, error, attempts, retry_delays) -> None:
        record = {
            **context,
            "timestamp": time.time(),
            "attempts": attempts,
            "retry_delays": list(retry_delays),
            "status_code": getattr(error, "status_code", None),
            "exception_class": type(error).__name__,
            "retry_after_present": bool(
                getattr(error, "headers", {}).get("Retry-After")
            ),
            "error": str(error)[:500],
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count
```

Implement `summary()` as `self.stats.summary()` plus failure-recorder count.
Never store response headers other than the boolean `retry_after_present`.

- [ ] **Step 6: Run Task 2 tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py -q
```

Expected: all policy, retry, single-flight, and failure-record tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add scripts/wikimedia_media.py tests/test_wikimedia_media.py
git commit -m "Add retryable Wikimedia media coordinator"
```

---

### Task 3: Integrate URL-keyed caching and streamed media attempts

**Files:**
- Modify: `scripts/build_mm_table_dataset.py:978-1345`
- Modify: `tests/test_stage1_pipeline.py:1126-1390`
- Modify: `tests/test_wikimedia_media.py`

**Interfaces:**
- Consumes: `WikimediaMediaDownloader.run()` and its shared limiter.
- Produces: backward-compatible `WikipediaClient.download_image(imageinfo, asset_id) -> dict[str, Any] | None`, deterministic `_shared_media_path()`, and `_download_image_attempt()`.

- [ ] **Step 1: Write failing legacy-cache, URL-cache, and standard-thumbnail tests**

Preserve the existing thumbnail tests and add:

```python
def jpeg_info(name):
    return {
        "file_title": f"File:{name}",
        "url": f"https://upload.wikimedia.org/original/{name}",
        "thumburl": f"https://upload.wikimedia.org/thumb/{name}/960px-{name}",
        "mime": "image/jpeg",
        "mediatype": "BITMAP",
        "width": 1200,
        "height": 800,
        "thumbwidth": 960,
        "thumbheight": 640,
    }


class SessionThatFailsOnGet:
    headers = {}

    def get(self, *args, **kwargs):
        raise AssertionError("cache hit must not access the network")


class FakeStreamingResponse:
    def __init__(self, status_code, body, headers):
        self.status_code = status_code
        self.body = body
        self.headers = headers
        self.text = body.decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, chunk_size):
        yield self.body


class SuccessfulImageSession:
    def __init__(self, body):
        self.headers = {}
        self.body = body
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return FakeStreamingResponse(200, self.body, {"Content-Type": "image/jpeg"})


def make_wikipedia_client(tmp_path):
    recorder = MediaFailureRecorder(tmp_path / "failures.jsonl")
    recorder.reset()
    return WikipediaClient(
        cache_dir=tmp_path / "metadata",
        image_output_dir=tmp_path / "images",
        output_dir=tmp_path,
        sleep=0,
        user_agent="DatasetBot/1.0 (mailto:test@example.com)",
        media_config=MediaPolicyConfig(retry_base_seconds=0, retry_max_seconds=0),
        media_failure_recorder=recorder,
    )


def test_download_image_reuses_legacy_asset_file(tmp_path):
    client = make_wikipedia_client(tmp_path)
    legacy = tmp_path / "images" / "asset_alpha.jpg"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    client.session = SessionThatFailsOnGet()
    record = client.download_image(jpeg_info("Alpha.jpg"), "asset_alpha")
    assert Path(record["local_path"]) == legacy
    assert record["downloaded"] is False


def test_download_image_reuses_url_keyed_file_for_distinct_assets(tmp_path):
    client = make_wikipedia_client(tmp_path)
    session = SuccessfulImageSession(b"image")
    client.session = session
    first = client.download_image(jpeg_info("Shared.jpg"), "asset_a")
    second = client.download_image(jpeg_info("Shared.jpg"), "asset_b")
    assert first["local_path"] == second["local_path"]
    assert session.calls == 1
```

Update the imageinfo test to assert `iiurlwidth == 960` rather than `768`.

- [ ] **Step 2: Run integration tests and verify failure**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -k 'wikipedia_download_image or imageinfo_requests_thumbnail' -q
```

Expected: URL-cache test fails and thumbnail width still differs.

- [ ] **Step 3: Wire the downloader into `WikipediaClient`**

Change `WikipediaClient.__init__` to accept optional keyword-only arguments:

```python
def __init__(
    self,
    cache_dir: Path,
    image_output_dir: Path,
    output_dir: Path,
    sleep: float,
    user_agent: str,
    *,
    media_config: MediaPolicyConfig | None = None,
    media_failure_recorder: MediaFailureRecorder | None = None,
) -> None:
```

Construct one `MediaFailureRecorder` and one `WikimediaMediaDownloader` per
client. If the recorder argument is omitted, create one targeting
`output_dir / "media_download_failures.jsonl"`. Tests that omit the new
arguments receive validated defaults. Do not reset the failure file in the
constructor; the builder does that once before network work.

Set `DEFAULT_IMAGE_THUMB_WIDTH = 960`.

Implement deterministic paths:

```python
def _shared_media_path(self, url: str, extension: str) -> Path:
    media_key = stable_hash(url, extension, length=24)
    return self.image_output_dir / f"media_{media_key}{extension}"
```

`download_image()` lookup order is legacy non-empty final path, shared
non-empty URL path, then `media_downloader.run()`. The cache key is the exact
selected URL plus output extension. Empty files are removed before retry.

- [ ] **Step 4: Implement one streamed attempt with bandwidth accounting**

Extract the current request body into `_download_image_attempt()` and replace
`response.raise_for_status()` with explicit classification:

```python
if response.status_code >= 400:
    body_excerpt = clean_text(response.text)[:500]
    raise MediaHTTPError(response.status_code, dict(response.headers), body_excerpt)
content_type = response.headers.get("Content-Type", "")
if content_type and not content_type.lower().startswith("image/"):
    raise NonImageMediaError(content_type)
```

Stream with the shared limiter:

```python
iterator = response.iter_content(chunk_size=self.media_config.chunk_bytes)
for chunk in iterator:
    if not chunk:
        continue
    waited = limiter.acquire(len(chunk))
    self.media_downloader.stats.increment("bandwidth_wait_seconds", waited)
    handle.write(chunk)
    digest.update(chunk)
    total_bytes += len(chunk)
```

Although `requests` may buffer a small amount before yielding, the configured
128 KiB chunk and 24 Mbps rate keep the ten-second aggregate below 25 Mbps.
Only the leader creates or promotes temporary files. Preserve the existing SVG
conversion path and atomically replace the shared final path after conversion.

Return `file_download_record(shared_path, output_dir, downloaded=True)` with
the existing `download_url`, `download_kind`, SHA-256, byte, and SVG source
fields.

- [ ] **Step 5: Add retry and cleanup integration tests**

Add fake response sequences `[429 Retry-After: 0, 200]`, `[503, 200]`, and
`[404]`. Assert two calls for retryable responses, one call for 404, and no
`.tmp` or `.svg.tmp` files after terminal failure. Add a response whose body
contains `Use thumbnail steps` and assert it is not retried.

- [ ] **Step 6: Run Task 3 tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py tests/test_stage1_pipeline.py -k 'media or wikipedia_download_image or imageinfo' -q
```

Expected: all focused integration tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add scripts/wikimedia_media.py scripts/build_mm_table_dataset.py tests/test_wikimedia_media.py tests/test_stage1_pipeline.py
git commit -m "Integrate policy-compliant Wikimedia downloads"
```

---

### Task 4: Builder configuration, progress, and manifest metrics

**Files:**
- Modify: `scripts/build_mm_table_dataset.py:1594-1754,2080-2180,2230-2280`
- Modify: `scripts/build_mm_joinability_dataset.py:2070-2420,2430-2508`
- Modify: `tests/test_stage1_pipeline.py`

**Interfaces:**
- Consumes: `MediaPolicyConfig` and `WikimediaMediaDownloader.summary()`.
- Produces: validated CLI flags, current-run `media_download_failures.jsonl`, separate media progress, and `wikimedia_media` manifest summaries in both builders.

- [ ] **Step 1: Write failing CLI validation tests**

Add parameterized tests for both parser functions:

```python
@pytest.mark.parametrize("parser", [mm_table_dataset.parse_args, join_dataset.parse_args])
def test_media_cli_defaults_follow_wikimedia_policy(parser):
    args = parser(["--input_dir", "in", "--output_dir", "out"])
    assert args.media_download_workers == 2
    assert args.media_max_mbps == 24.0
    assert args.media_max_retries == 5
    assert args.media_chunk_bytes == 131072
```

Add build-level tests asserting `workers=3` and `max_mbps=25.1` raise a clear
`ValueError` before constructing a network client.

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -k 'media_cli' -q
```

Expected: failure because the parser namespaces lack media options.

- [ ] **Step 3: Add media CLI flags to both builders**

Add the six options from the design with identical defaults and help text to
both parsers. Build a helper:

```python
def media_policy_config_from_args(args: argparse.Namespace) -> MediaPolicyConfig:
    return MediaPolicyConfig(
        workers=args.media_download_workers,
        max_mbps=args.media_max_mbps,
        max_retries=args.media_max_retries,
        retry_base_seconds=args.media_retry_base_seconds,
        retry_max_seconds=args.media_retry_max_seconds,
        chunk_bytes=args.media_chunk_bytes,
    ).validate()
```

The dynamic vLLM wrapper already forwards unknown builder arguments, so it
needs no parser duplication.

- [ ] **Step 4: Reset failure output once and expose downloader stats**

Before building bridge assets, set:

```python
media_failure_path = output_dir / "media_download_failures.jsonl"
failure_recorder = MediaFailureRecorder(media_failure_path)
failure_recorder.reset()
```

Pass the validated config as `media_config` and the recorder as
`media_failure_recorder` into `WikipediaClient`. Add
`wikipedia_client.media_summary()` and include its result under
`manifest["wikimedia_media"]`. Preserve the existing `api_failures` field for
backward compatibility but stop treating it as an Action-API-only count in
documentation.

- [ ] **Step 5: Separate entity and media progress**

Keep `Fetching Wikipedia assets` for entity and metadata traversal. Wrap the
pending media-future resolution loop with a second progress bar:

```python
pending_iterator = pending_image_downloads
if tqdm is not None:
    pending_iterator = tqdm(
        pending_image_downloads,
        total=len(pending_image_downloads),
        desc="Resolving Wikimedia media",
        unit="asset",
        dynamic_ncols=True,
    )
for future, entity, asset_id, imageinfo in pending_iterator:
    ...
```

The downloader's single-flight layer supplies the true network/cache counters;
the media progress bar counts asset resolutions and must not be labeled as
network downloads.

- [ ] **Step 6: Add User-Agent warning and Action API maxlag**

When Wikipedia access is enabled and the built-in placeholder User-Agent is
used, log one warning requesting project/operator contact information without
printing the configured value. Add `maxlag=5` to both Action API query parameter
dictionaries. Preserve the current environment and CLI override tests.

- [ ] **Step 7: Run builder and manifest tests**

Run:

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -k 'wikipedia or media or manifest' -q
```

Expected: all focused builder, progress, cache, and manifest tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add scripts/build_mm_table_dataset.py scripts/build_mm_joinability_dataset.py tests/test_stage1_pipeline.py
git commit -m "Expose Wikimedia media controls and metrics"
```

---

### Task 5: Full regression verification and cache-safety audit

**Files:**
- Modify only if verification exposes a defect: `scripts/wikimedia_media.py`, `scripts/build_mm_table_dataset.py`, `scripts/build_mm_joinability_dataset.py`, `tests/test_wikimedia_media.py`, `tests/test_stage1_pipeline.py`

**Interfaces:**
- Consumes: all earlier tasks.
- Produces: a verified implementation with no live downloads and no cache mutation during tests.

- [ ] **Step 1: Run focused media tests from a clean process**

```bash
conda run -n MMDD python -m pytest tests/test_wikimedia_media.py -q
```

Expected: all tests pass with no network access.

- [ ] **Step 2: Run the repository regression suite**

```bash
conda run -n MMDD python -m pytest tests/test_stage1_pipeline.py -q
```

Expected: all Stage-1 tests pass.

- [ ] **Step 3: Run adjacent multimodal tests**

```bash
conda run -n MMDD python -m pytest tests/test_mm_joinability_extraction.py tests/test_mm_joinability_dynamic_vllm.py -q
```

Expected: all adjacent joinability and dynamic-wrapper tests pass.

- [ ] **Step 4: Run static and CLI smoke checks**

```bash
conda run -n MMDD python -m py_compile scripts/wikimedia_media.py scripts/build_mm_table_dataset.py scripts/build_mm_joinability_dataset.py scripts/run_mm_joinability_dynamic_vllm.py
conda run -n MMDD python scripts/build_mm_joinability_dataset.py --help
conda run -n MMDD python scripts/build_mm_table_dataset.py --help
```

Expected: compilation succeeds and both help pages list the six media options.

- [ ] **Step 5: Verify existing cache remains untouched**

Record before and after counts without hashing 53 GB of data:

```bash
find cache/mm_joinability/images -maxdepth 1 -type f | wc -l
find cache/mm_joinability/images -maxdepth 1 -type f -name '*.tmp' | wc -l
```

Expected: test execution does not change the existing file count and leaves
zero temporary files.

- [ ] **Step 6: Review the final diff for accidental scope expansion**

```bash
git diff --check
git status --short
git diff -- scripts/wikimedia_media.py scripts/build_mm_table_dataset.py scripts/build_mm_joinability_dataset.py tests/test_wikimedia_media.py tests/test_stage1_pipeline.py
```

Expected: no whitespace errors, no generated outputs, no cache files, and no
unrelated changes.

- [ ] **Step 7: Commit verification-only fixes if any were required**

If Step 1-6 required code corrections, commit only those corrections:

```bash
git add scripts/wikimedia_media.py scripts/build_mm_table_dataset.py scripts/build_mm_joinability_dataset.py tests/test_wikimedia_media.py tests/test_stage1_pipeline.py
git commit -m "Fix Wikimedia media regression findings"
```

If no correction was required, do not create an empty commit.

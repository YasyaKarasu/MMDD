from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import socket
import sqlite3
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_wdc_mm_joinability_dataset as wdc_builder  # noqa: E402
import wdc200k_fetch as fetch_module  # noqa: E402
from build_wdc_mm_joinability_dataset import WdcWebClient  # noqa: E402
from wdc200k_fetch import (  # noqa: E402
    FetchPolicy,
    PageOutcomeStore,
    fetch_unique_pages,
    iter_finalized_page_refs,
    iter_page_fanout,
    iter_page_outcomes,
    validate_complete_page_fetch,
)
from wdc200k_io import SqliteJobStore  # noqa: E402
from wdc200k_io import AtomicJsonlShard  # noqa: E402
from stage1_io import stable_hash  # noqa: E402
from wdc200k_eta import UrlProgressSnapshot  # noqa: E402


def page_ref(entity_id: str, url: str) -> dict[str, str]:
    return {
        "entity_id": entity_id,
        "page_url": url,
        "url_key": hashlib.sha256(url.encode("utf-8")).hexdigest(),
    }


def png_bytes(color: tuple[int, int, int] = (20, 40, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _open_outcome_store_process(
    path: str,
    results: Any,
) -> None:
    try:
        store = PageOutcomeStore(Path(path))
        results.put(("ok", store.counts("legacy-policy")))
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))


class CountingTransport:
    def __init__(
        self,
        outcomes: dict[str, dict[str, Any] | BaseException | None],
    ) -> None:
        self.outcomes = outcomes
        self.network_policy_fingerprint = "wdc-web-v1"
        self.calls: list[str] = []
        self.cached: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict[str, Any] | None:
        assert deadline_seconds > 0
        assert max_retries == 0
        with self.lock:
            self.calls.append(url)
        outcome = self.outcomes[url]
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is None:
            self.cached[url] = {
                "status": "terminal",
                "page_url": url,
                "error_class": "fetch_failed",
            }
            return None
        result = {
            "page_url": url,
            "final_url": url,
            "text": str(outcome.get("text", "")),
            "image_urls": list(outcome.get("image_urls", [])),
        }
        self.cached[url] = {"status": "success", **result}
        return result

    def cached_page_outcome(self, url: str) -> dict[str, Any] | None:
        return self.cached.get(url)


def test_duplicate_page_urls_make_one_physical_request(tmp_path: Path) -> None:
    url = "https://e.test/a"
    transport = CountingTransport({url: {"text": "hello"}})

    result = fetch_unique_pages(
        page_refs=[page_ref("e1", url), page_ref("e2", url)],
        store=SqliteJobStore(tmp_path / "pages.sqlite3"),
        transport=transport,
        policy=FetchPolicy(retries=0, deadline_seconds=8),
    )

    assert transport.calls == [url]
    assert result.unique == 1
    assert result.success == 1
    assert result.terminal == 0
    assert result.transport_attempt_summary["transport_attempts"] == 1
    assert (
        result.transport_attempt_summary["duplicate_physical_requests"] == 0
    )
    assert result.transport_attempt_summary["terminal_replays"] == 0
    assert (
        result.transport_attempt_summary["unfinished_transport_attempts"] == 0
    )
    assert result.transport_attempt_summary["blocked_durable_replays"] == 0


def test_page_transport_attempts_survive_resume_without_replay(
    tmp_path: Path,
) -> None:
    url = "https://e.test/resume"
    transport = CountingTransport({url: {"text": "hello"}})
    jobs = SqliteJobStore(tmp_path / "pages.sqlite3")
    policy = FetchPolicy()

    first = fetch_unique_pages(
        [page_ref("e1", url)],
        jobs,
        transport,
        policy,
    )
    second = fetch_unique_pages(
        [page_ref("e1", url)],
        jobs,
        transport,
        policy,
    )

    assert transport.calls == [url]
    assert second.transport_attempt_summary == first.transport_attempt_summary
    assert second.transport_attempt_summary["records"] == 1
    assert len(second.transport_attempt_summary["digest"]) == 64


def test_page_progress_callback_starts_from_durable_baseline_and_is_bounded(
    tmp_path: Path,
) -> None:
    urls = [f"https://e.test/progress-{index}" for index in range(5)]
    transport = CountingTransport(
        {url: {"text": str(index)} for index, url in enumerate(urls)}
    )
    jobs = SqliteJobStore(tmp_path / "pages.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    policy = FetchPolicy(global_concurrency=1, per_host_concurrency=1)

    fetch_unique_pages(
        [page_ref("e0", urls[0])],
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
    )
    updates: list[UrlProgressSnapshot] = []
    result = fetch_unique_pages(
        [page_ref(f"e{index}", url) for index, url in enumerate(urls)],
        jobs,
        transport,
        policy,
        outcomes_path=outcomes_path,
        progress_callback=updates.append,
        progress_callback_every=2,
    )

    assert result.complete
    assert [snapshot.completed_durable for snapshot in updates] == [1, 3, 5]
    assert all(snapshot.total == 5 for snapshot in updates)
    assert len({snapshot.execution_epoch for snapshot in updates}) == 1


def test_page_tracker_distinguishes_physical_cache_and_suppressed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = "https://a.test/physical"
    failed = "https://b.test/failed"
    cached = "https://c.test/cached"
    suppressed = "https://d.test/suppressed"
    refs = [
        page_ref("physical", physical),
        page_ref("failed", failed),
        page_ref("cached", cached),
        page_ref("suppressed", suppressed),
    ]
    policy = FetchPolicy(global_concurrency=4, per_host_concurrency=1)
    outcomes_path = tmp_path / "outcomes.sqlite3"
    outcome_store = PageOutcomeStore(outcomes_path)
    for url in (cached, suppressed):
        outcome_store.put(
            policy.fingerprint,
            hashlib.sha256(url.encode("utf-8")).hexdigest(),
            url,
            {"status": "success", "text": url, "image_urls": []},
        )
    suppressed_key = hashlib.sha256(suppressed.encode("utf-8")).hexdigest()
    original_get = fetch_module.PageOutcomeStore.get
    get_calls = 0

    def hide_suppressed_once(self, fingerprint: str, url_key: str):
        nonlocal get_calls
        if url_key == suppressed_key:
            get_calls += 1
            if get_calls == 1:
                return None
        return original_get(self, fingerprint, url_key)

    monkeypatch.setattr(
        fetch_module.PageOutcomeStore,
        "get",
        hide_suppressed_once,
    )
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_pages(
        refs,
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        CountingTransport(
            {
                physical: {"text": "ok"},
                failed: TimeoutError("deadline"),
                suppressed: {"text": "must not run"},
            }
        ),
        policy,
        outcomes_path=outcomes_path,
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )

    assert result.complete
    assert snapshots[0].completed_durable == 0
    final = snapshots[-1]
    assert final.completed_durable == final.total == 4
    assert sum(final.transport_event_histogram) == 2
    assert sum(final.commit_event_histogram) == 3
    assert final.physical_in_flight == 0
    assert final.in_flight_jobs == 0
    assert final.finished_not_durable == 0


def test_page_claim_batch_is_buffered_before_synchronous_cache_callback(
    tmp_path: Path,
) -> None:
    urls = ["https://e.test/claim-a", "https://e.test/claim-b"]
    cached = min(
        urls,
        key=lambda url: hashlib.sha256(url.encode("utf-8")).hexdigest(),
    )
    physical = next(url for url in urls if url != cached)
    policy = FetchPolicy(global_concurrency=1, per_host_concurrency=1)
    outcomes_path = tmp_path / "outcomes.sqlite3"
    PageOutcomeStore(outcomes_path).put(
        policy.fingerprint,
        hashlib.sha256(cached.encode("utf-8")).hexdigest(),
        cached,
        {"status": "success", "text": "cached", "image_urls": []},
    )
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_pages(
        [page_ref(str(index), url) for index, url in enumerate(urls)],
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        CountingTransport({physical: {"text": "physical"}}),
        policy,
        outcomes_path=outcomes_path,
        claim_buffer=2,
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )

    assert result.complete
    cached_snapshot = next(
        item for item in snapshots if item.completed_durable == 1
    )
    assert cached_snapshot.local_buffered_not_started == 1
    assert cached_snapshot.unobserved_nonlocal == 0


def test_page_first_completed_batch_moves_all_futures_before_serial_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(2)

    class BatchTransport(CountingTransport):
        def fetch_page(self, url: str, **kwargs: Any) -> dict[str, Any]:
            barrier.wait(timeout=5)
            return super().fetch_page(url, **kwargs)

    urls = ["https://a.test/batch", "https://b.test/batch"]
    real_wait = fetch_module.wait

    def wait_for_whole_batch(fs, *, return_when):
        assert return_when is fetch_module.FIRST_COMPLETED
        done, pending = real_wait(fs)
        return done, pending

    monkeypatch.setattr(fetch_module, "wait", wait_for_whole_batch)
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_pages(
        [page_ref(str(index), url) for index, url in enumerate(urls)],
        SqliteJobStore(tmp_path / "jobs.sqlite3"),
        BatchTransport({url: {"text": url} for url in urls}),
        FetchPolicy(global_concurrency=2, per_host_concurrency=1),
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )

    assert result.complete
    first_durable = next(
        snapshot for snapshot in snapshots if snapshot.completed_durable == 1
    )
    assert first_durable.in_flight_jobs == 0
    assert first_durable.finished_not_durable == 1


def test_page_external_execution_completion_is_visible_in_final_snapshot(
    tmp_path: Path,
) -> None:
    url = "https://e.test/external"
    ref = page_ref("external", url)
    policy = FetchPolicy(global_concurrency=1, per_host_concurrency=1)
    kind = fetch_module._job_kind(policy.fingerprint)
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    jobs.enqueue(
        kind,
        f"{policy.fingerprint}:{ref['url_key']}",
        {**ref, "host": "e.test"},
    )
    foreign = jobs.claim(kind, 1, "foreign", lease_seconds=60)[0]
    outcomes_path = tmp_path / "outcomes.sqlite3"
    outcome_store = PageOutcomeStore(outcomes_path)

    def finish_elsewhere() -> None:
        time.sleep(0.03)
        outcome_store.put(
            policy.fingerprint,
            ref["url_key"],
            url,
            {"status": "success", "text": "external", "image_urls": []},
        )
        jobs.finish(
            foreign.job_id,
            "success",
            {"url_key": ref["url_key"]},
            owner="foreign",
            lease_id=foreign.lease_id,
        )

    thread = threading.Thread(target=finish_elsewhere)
    thread.start()
    snapshots: list[UrlProgressSnapshot] = []
    result = fetch_unique_pages(
        [ref],
        jobs,
        CountingTransport({url: {"text": "must not run"}}),
        policy,
        outcomes_path=outcomes_path,
        max_wait_seconds=0.3,
        poll_interval_seconds=0.01,
        progress_callback=snapshots.append,
        progress_callback_every=1,
    )
    thread.join(timeout=2)

    assert result.complete
    assert [item.completed_durable for item in snapshots] == [0, 1]
    assert snapshots[0].unobserved_nonlocal == 1
    assert snapshots[-1].unobserved_nonlocal == 0
    assert sum(snapshots[-1].transport_event_histogram) == 0


def test_page_url_refresh_uses_kind_status_and_outcome_key_indexes(
    tmp_path: Path,
) -> None:
    jobs = SqliteJobStore(tmp_path / "jobs.sqlite3")
    outcomes_path = tmp_path / "outcomes.sqlite3"
    PageOutcomeStore(outcomes_path)
    kind = "wdc200k-page:policy"
    jobs.enqueue(kind, "job", {"url_key": "a" * 64})
    counts = fetch_module._refresh_page_url_counts(
        jobs, kind, outcomes_path, "policy"
    )
    assert (counts.completed, counts.pending, counts.leased, counts.total) == (
        0,
        1,
        0,
        1,
    )
    with jobs._connect() as connection:
        connection.execute(
            "ATTACH DATABASE ? AS page_progress_outcomes",
            (str(outcomes_path),),
        )
        status_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT status, COUNT(*) FROM jobs "
            "WHERE kind = ? GROUP BY status",
            (kind,),
        ).fetchall()
        completed_plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM jobs AS job "
            "JOIN page_progress_outcomes.page_outcomes AS outcome "
            "ON outcome.policy_fingerprint = ? "
            "AND outcome.url_key = json_extract(job.payload_json, '$.url_key') "
            "WHERE job.kind = ? AND job.status IN ('success', 'terminal') "
            "AND outcome.status IN ('success', 'terminal')",
            ("policy", kind),
        ).fetchall()
    details = [str(row[3]) for row in status_plan + completed_plan]
    assert not any("SCAN jobs" in detail for detail in details)
    assert any("jobs_kind_status" in detail for detail in details)
    assert any(
        "sqlite_autoindex_page_outcomes_1" in detail for detail in details
    )


def test_page_url_refresh_million_rows_keeps_python_peak_below_16_mib(
    tmp_path: Path,
) -> None:
    jobs = SqliteJobStore(tmp_path / "million-jobs.sqlite3")
    outcomes_path = tmp_path / "million-outcomes.sqlite3"
    PageOutcomeStore(outcomes_path)
    kind = "wdc200k-page:million"
    with jobs._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            WITH digits(d) AS (
                VALUES (0),(1),(2),(3),(4),(5),(6),(7),(8),(9)
            )
            INSERT INTO jobs (
                job_id, kind, payload_json, status, updated_at
            )
            SELECT printf('bulk-%07d',
                    a.d + 10*b.d + 100*c.d + 1000*d.d
                    + 10000*e.d + 100000*f.d),
                   ?, '{"url_key":"missing"}', 'pending', 0.0
            FROM digits AS a CROSS JOIN digits AS b
            CROSS JOIN digits AS c CROSS JOIN digits AS d
            CROSS JOIN digits AS e CROSS JOIN digits AS f
            """,
            (kind,),
        )
        connection.commit()

    tracemalloc.start()
    counts = fetch_module._refresh_page_url_counts(
        jobs, kind, outcomes_path, "policy"
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert counts.total == counts.pending == 1_000_000
    assert peak < 16 * 1024 * 1024


def test_page_transport_summary_is_scoped_to_current_job_store(
    tmp_path: Path,
) -> None:
    first_url = "https://e.test/first"
    unrelated_url = "https://other.test/unrelated"
    outcomes_path = tmp_path / "shared-outcomes.sqlite3"
    policy = FetchPolicy()
    transport = CountingTransport(
        {
            first_url: {"text": "first"},
            unrelated_url: {"text": "unrelated"},
        }
    )

    first = fetch_unique_pages(
        [page_ref("e1", first_url)],
        SqliteJobStore(tmp_path / "first-jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=outcomes_path,
    )
    fetch_unique_pages(
        [page_ref("e2", unrelated_url)],
        SqliteJobStore(tmp_path / "other-jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=outcomes_path,
    )
    resumed = fetch_unique_pages(
        [page_ref("e1", first_url)],
        SqliteJobStore(tmp_path / "first-jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=outcomes_path,
    )

    assert resumed.transport_attempt_summary == (
        first.transport_attempt_summary
    )
    assert resumed.transport_attempt_summary["transport_attempts"] == 1


def test_page_transport_attempt_summary_counts_anomalies(
    tmp_path: Path,
) -> None:
    store = PageOutcomeStore(tmp_path / "outcomes.sqlite3")
    policy = "policy"
    url = "https://e.test/anomaly"
    url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()

    first = store.begin_transport_attempt(
        execution_id="execution-one",
        policy_fingerprint=policy,
        url_key=url_key,
        url=url,
        baseline_outcome_status=None,
        suppressed=False,
        now=1.0,
    )
    store.finish_transport_attempt(first, final_status="success", now=2.0)
    store.put(
        policy,
        url_key,
        url,
        {"status": "terminal", "error_class": "prior_terminal"},
    )
    second = store.begin_transport_attempt(
        execution_id="execution-two",
        policy_fingerprint=policy,
        url_key=url_key,
        url=url,
        baseline_outcome_status=None,
        suppressed=False,
        now=time.time() + 1.0,
    )
    store.finish_transport_attempt(
        second,
        final_status="terminal",
        now=time.time() + 2.0,
    )
    store.begin_transport_attempt(
        execution_id="execution-three",
        policy_fingerprint=policy,
        url_key=url_key,
        url=url,
        baseline_outcome_status="terminal",
        suppressed=True,
        now=5.0,
    )
    store.begin_transport_attempt(
        execution_id="execution-four",
        policy_fingerprint=policy,
        url_key=hashlib.sha256(b"https://e.test/crash").hexdigest(),
        url="https://e.test/crash",
        baseline_outcome_status=None,
        suppressed=False,
        now=6.0,
    )

    summary = store.transport_attempt_summary(policy)
    assert summary["records"] == 4
    assert summary["transport_attempts"] == 3
    assert summary["duplicate_physical_requests"] == 1
    assert summary["terminal_replays"] == 1
    assert summary["unfinished_transport_attempts"] == 1
    assert summary["blocked_durable_replays"] == 1


def test_page_transport_attempt_start_is_disk_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    PageOutcomeStore(path)

    def reject(_path: Path, estimated_bytes: int = 0) -> None:
        if estimated_bytes:
            raise OSError("transport reserve exhausted")

    guarded = PageOutcomeStore(
        path,
        write_tracker=fetch_module.GuardedWriteTracker(
            path,
            reject,
            interval_bytes=1,
        ),
    )
    with pytest.raises(OSError, match="transport reserve"):
        guarded.begin_transport_attempt(
            execution_id="execution",
            policy_fingerprint="policy",
            url_key="a" * 64,
            url="https://e.test/guard",
            baseline_outcome_status=None,
            suppressed=False,
        )
    assert PageOutcomeStore(path).transport_attempt_summary("policy")[
        "records"
    ] == 0


def test_page_outcome_fanout_streams_every_entity_reference(
    tmp_path: Path,
) -> None:
    url = "https://e.test/shared"
    transport = CountingTransport({url: {"text": "shared"}})
    result = fetch_unique_pages(
        page_refs=[
            {
                **page_ref("e1", url),
                "source_table_id": "t1",
                "row_id": 1,
            },
            {
                **page_ref("e2", url),
                "source_table_id": "t2",
                "row_id": 7,
            },
        ],
        store=SqliteJobStore(tmp_path / "pages.sqlite3"),
        transport=transport,
        policy=FetchPolicy(),
    )

    fanout = list(
        iter_page_fanout(
            result.outcomes_path,
            result.policy_fingerprint,
        )
    )
    assert transport.calls == [url]
    assert sorted(
        (row["entity_id"], row["row_id"]) for row in fanout
    ) == [
        ("e1", 1),
        ("e2", 7),
    ]
    assert len({row["payload_sha256"] for row in fanout}) == 1
    assert all(row["status"] == "success" for row in fanout)


def test_terminal_page_failure_is_not_replayed_on_resume(
    tmp_path: Path,
) -> None:
    url = "https://e.test/a"
    transport = CountingTransport({url: TimeoutError()})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")

    fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(),
    )
    resumed = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(),
    )

    assert transport.calls == [url]
    assert resumed.terminal == 1
    outcomes = list(iter_page_outcomes(resumed.outcomes_path, resumed.policy_fingerprint))
    assert outcomes[0]["status"] == "terminal"
    assert outcomes[0]["error_class"] == "TimeoutError"
    assert "error" not in outcomes[0]


def test_cache_write_before_job_finish_is_repaired_without_request(
    tmp_path: Path,
) -> None:
    url = "https://e.test/crash"
    transport = CountingTransport({url: {"text": "cached before crash"}})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")
    crashed = False

    def crash_after_cache(_outcome: dict[str, Any]) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated process crash"):
        fetch_unique_pages(
            [page_ref("e1", url)],
            store,
            transport,
            FetchPolicy(),
            after_cache_write=crash_after_cache,
            lease_seconds=-1,
        )

    resumed = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(),
    )

    assert transport.calls == [url]
    assert resumed.success == 1
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status FROM jobs"
        ).fetchone() == ("success",)


def test_positive_ttl_crash_is_reclaimed_by_immediate_resume(
    tmp_path: Path,
) -> None:
    url = "https://e.test/positive-ttl"
    transport = CountingTransport({url: {"text": "durable"}})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")

    with pytest.raises(RuntimeError, match="crash after cache"):
        fetch_unique_pages(
            [page_ref("e1", url)],
            store,
            transport,
            FetchPolicy(global_concurrency=1),
            lease_seconds=0.05,
            after_cache_write=lambda _outcome: (_ for _ in ()).throw(
                RuntimeError("crash after cache")
            ),
        )

    resumed = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(global_concurrency=1),
        max_wait_seconds=0.5,
        poll_interval_seconds=0.01,
    )

    assert resumed.complete is True
    assert resumed.leased == 0
    assert transport.calls == [url]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute(
            "SELECT status FROM jobs"
        ).fetchone() == ("success",)


def test_expired_lease_cannot_finish_but_reclaim_repairs_from_cache(
    tmp_path: Path,
) -> None:
    url = "https://e.test/lease"
    transport = CountingTransport({url: {"text": "one request"}})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")

    with pytest.raises(RuntimeError, match="no active lease"):
        fetch_unique_pages(
            [page_ref("e1", url)],
            store,
            transport,
            FetchPolicy(),
            lease_seconds=-1,
        )

    result = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(),
    )
    assert transport.calls == [url]
    assert result.success == 1


def test_policy_change_has_distinct_jobs_and_outcomes(tmp_path: Path) -> None:
    url = "https://e.test/versioned"
    transport = CountingTransport({url: {"text": "ok"}})
    store = SqliteJobStore(tmp_path / "pages.sqlite3")

    first = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(policy_version="page-v1"),
    )
    second = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(policy_version="page-v2"),
    )

    assert first.policy_fingerprint != second.policy_fingerprint
    assert transport.calls == [url, url]
    assert len(list(iter_page_outcomes(second.outcomes_path))) == 2


def test_outcome_counts_are_transactional_idempotent_and_constant_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = PageOutcomeStore(tmp_path / "outcomes.sqlite3")
    policy = "p1"
    key = "a" * 64
    success = {
        "status": "success",
        "final_url": "https://e.test/a",
        "text": "ok",
        "image_urls": [],
    }

    store.put(policy, key, "https://e.test/a", success)
    store.put(policy, key, "https://e.test/a", success)
    assert store.counts(policy) == (1, 0)
    with pytest.raises(ValueError, match="conflicting immutable"):
        store.put(
            policy,
            key,
            "https://e.test/a",
            {"status": "terminal", "error_class": "TimeoutError"},
        )
    assert store.counts(policy) == (1, 0)

    statements: list[str] = []
    original_connect = store._connect

    def traced_connect() -> sqlite3.Connection:
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(store, "_connect", traced_connect)
    assert store.counts(policy) == (1, 0)
    count_sql = " ".join(statements).casefold()
    assert "policy_outcome_counts" in count_sql
    assert "group by" not in count_sql


def test_page_outcome_commit_guard_rolls_back_and_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    PageOutcomeStore(path)
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 3:
                raise OSError("page outcome commit reserve exhausted")

    tracker = fetch_module.GuardedWriteTracker(
        path,
        reject_commit,
        interval_bytes=1,
    )
    guarded = PageOutcomeStore(path, write_tracker=tracker)
    outcome = {
        "status": "success",
        "final_url": "https://e.test/a",
        "text": "ok",
        "image_urls": [],
    }
    with pytest.raises(OSError, match="page outcome commit reserve"):
        guarded.put("policy", "a" * 64, "https://e.test/a", outcome)

    resumed = PageOutcomeStore(path)
    assert resumed.counts("policy") == (0, 0)
    resumed.put("policy", "a" * 64, "https://e.test/a", outcome)
    assert resumed.counts("policy") == (1, 0)


def test_page_outcome_initialization_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outcomes.sqlite3"
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("page init reserve exhausted")

    tracker = fetch_module.GuardedWriteTracker(path, reject_commit)
    with pytest.raises(OSError, match="page init reserve"):
        PageOutcomeStore(path, write_tracker=tracker)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE name = 'page_outcomes'"
        ).fetchone() == (0,)


def test_legacy_outcome_schema_migrates_once_across_processes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-outcomes.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE page_outcomes (
                policy_fingerprint TEXT NOT NULL,
                url_key TEXT NOT NULL,
                page_url TEXT NOT NULL,
                status TEXT NOT NULL,
                final_url TEXT,
                text TEXT,
                image_urls_json TEXT,
                error_class TEXT,
                http_status INTEGER,
                payload_sha256 TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (policy_fingerprint, url_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE page_references (
                policy_fingerprint TEXT NOT NULL,
                url_key TEXT NOT NULL,
                reference_key TEXT NOT NULL,
                PRIMARY KEY (
                    policy_fingerprint, url_key, reference_key
                )
            )
            """
        )
        connection.execute(
            """
            INSERT INTO page_outcomes VALUES (
                'legacy-policy', ?, 'https://e.test/legacy',
                'success', 'https://e.test/legacy', 'legacy', '[]',
                NULL, NULL, ?, 1.0
            )
            """,
            ("a" * 64, "b" * 64),
        )

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_open_outcome_store_process,
            args=(str(path), results),
        )
        for _index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert all(process.exitcode == 0 for process in processes)
    assert [results.get(timeout=1) for _index in processes] == [
        ("ok", (1, 0))
    ] * 4
    store = PageOutcomeStore(path)
    assert store.counts("legacy-policy") == (1, 0)
    assert len(list(store.iter("legacy-policy"))) == 1
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(page_references)"
            )
        }
        assert {
            "entity_id",
            "source_table_id",
            "row_id_json",
        }.issubset(columns)
        assert connection.execute(
            """
            SELECT success, terminal, total
            FROM policy_outcome_counts
            WHERE policy_fingerprint = 'legacy-policy'
            """
        ).fetchone() == (1, 0, 1)


def test_full_progress_query_uses_only_indexed_leased_status_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    kind = "wdc200k-page:policy"
    for index in range(20):
        job_store.enqueue(kind, f"job-{index}", {"index": index})
    job_store.claim(kind, limit=2, owner="other", lease_seconds=60)
    outcomes = PageOutcomeStore(tmp_path / "outcomes.sqlite3")
    statements: list[str] = []
    original_connect = job_store._connect

    def traced_connect() -> sqlite3.Connection:
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(job_store, "_connect", traced_connect)
    fetch_module._publish_progress(
        outcomes,
        store=job_store,
        kind=kind,
        policy_fingerprint="policy",
        unique=20,
        progress_path=tmp_path / "progress.json",
        inflight=0,
    )

    sql = " ".join(statements).casefold()
    assert "group by" not in sql
    assert "status = 'leased'" in sql
    with original_connect() as connection:
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT COUNT(*) FROM jobs
            WHERE kind = ? AND status = 'leased'
            """,
            (kind,),
        ).fetchall()
    assert any(
        "jobs_kind_status" in str(row[3])
        and "kind=?" in str(row[3])
        and "status=?" in str(row[3])
        for row in plan
    )


def test_fetch_rejects_transport_network_policy_mismatch_before_work(
    tmp_path: Path,
) -> None:
    url = "https://e.test/policy"
    session = _NoNetworkSession()
    client = WdcWebClient(
        tmp_path / "web",
        session=session,
        host_delay=0,
        network_policy_version="stage-v1",
    )

    with pytest.raises(ValueError, match="network policy"):
        fetch_unique_pages(
            [page_ref("e1", url)],
            SqliteJobStore(tmp_path / "pages.sqlite3"),
            client,
            FetchPolicy(network_policy_fingerprint="stage-v2"),
        )

    assert session.calls == 0
    with sqlite3.connect(tmp_path / "pages.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone() == (0,)


class ConcurrencyTransport(CountingTransport):
    def __init__(self, delay: float = 0.02) -> None:
        super().__init__({})
        self.delay = delay
        self.active = 0
        self.maximum_global = 0
        self.active_by_host: dict[str, int] = {}
        self.maximum_by_host: dict[str, int] = {}
        self.started_hosts: list[str] = []

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict[str, Any]:
        host = url.split("/", 3)[2]
        with self.lock:
            self.calls.append(url)
            self.started_hosts.append(host)
            self.active += 1
            self.maximum_global = max(self.maximum_global, self.active)
            self.active_by_host[host] = self.active_by_host.get(host, 0) + 1
            self.maximum_by_host[host] = max(
                self.maximum_by_host.get(host, 0),
                self.active_by_host[host],
            )
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
            self.active_by_host[host] -= 1
        result = {
            "status": "success",
            "page_url": url,
            "final_url": url,
            "text": "ok",
            "image_urls": [],
        }
        self.cached[url] = result
        return result


class BlockingTransport(CountingTransport):
    def __init__(self, url: str) -> None:
        super().__init__({url: {"text": "eventually"}})
        self.started = threading.Event()
        self.release = threading.Event()

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict[str, Any] | None:
        self.started.set()
        assert self.release.wait(timeout=2)
        return super().fetch_page(
            url,
            deadline_seconds=deadline_seconds,
            max_retries=max_retries,
        )


def test_second_worker_reports_active_lease_as_incomplete(
    tmp_path: Path,
) -> None:
    url = "https://e.test/active"
    transport = BlockingTransport(url)
    store = SqliteJobStore(tmp_path / "pages.sqlite3")
    first_results: list[Any] = []
    first = threading.Thread(
        target=lambda: first_results.append(
            fetch_unique_pages(
                [page_ref("e1", url)],
                store,
                transport,
                FetchPolicy(global_concurrency=1),
            )
        )
    )
    first.start()
    assert transport.started.wait(timeout=1)

    concurrent = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        transport,
        FetchPolicy(global_concurrency=1),
        max_wait_seconds=0,
    )
    progress = json.loads(concurrent.progress_path.read_text(encoding="utf-8"))

    assert concurrent.complete is False
    assert concurrent.remaining == 1
    assert concurrent.leased == 1
    assert concurrent.inflight == 1
    assert progress["complete"] is False
    assert progress["remaining"] == 1
    assert progress["leased"] == 1
    assert progress["inflight"] == 1

    transport.release.set()
    first.join(timeout=2)
    assert not first.is_alive()
    assert first_results[0].complete is True
    final_progress = json.loads(
        concurrent.progress_path.read_text(encoding="utf-8")
    )
    assert final_progress["complete"] is True
    assert final_progress["remaining"] == 0
    assert not list(tmp_path.glob(".*.tmp"))


def test_scheduler_enforces_host_and_global_limits_without_starvation(
    tmp_path: Path,
) -> None:
    refs = [
        page_ref(f"a{index}", f"https://a.test/{index}")
        for index in range(100)
    ]
    refs.extend(
        page_ref(f"b{index}", f"https://b.test/{index}")
        for index in range(3)
    )
    transport = ConcurrencyTransport()

    result = fetch_unique_pages(
        refs,
        SqliteJobStore(tmp_path / "pages.sqlite3"),
        transport,
        FetchPolicy(global_concurrency=4, per_host_concurrency=2),
    )

    assert result.success == 103
    assert transport.maximum_global <= 4
    assert max(transport.maximum_by_host.values()) <= 2
    assert "b.test" in transport.started_hosts[:8]


def test_scheduler_keeps_claimed_and_inflight_work_bounded(
    tmp_path: Path,
) -> None:
    refs = [
        page_ref(str(index), f"https://h{index % 7}.test/{index}")
        for index in range(200)
    ]
    result = fetch_unique_pages(
        refs,
        SqliteJobStore(tmp_path / "pages.sqlite3"),
        ConcurrencyTransport(delay=0),
        FetchPolicy(global_concurrency=5, per_host_concurrency=2),
        claim_buffer=15,
    )

    assert result.maximum_inflight <= 5
    assert result.maximum_claimed <= 15
    assert result.maximum_host_limiters <= 15


def test_failure_and_progress_snapshots_are_durable_and_sanitized(
    tmp_path: Path,
) -> None:
    url = "https://e.test/private?token=secret"
    result = fetch_unique_pages(
        [page_ref("e1", url)],
        SqliteJobStore(tmp_path / "pages.sqlite3"),
        CountingTransport({url: RuntimeError("password=hunter2")}),
        FetchPolicy(),
        failure_path=tmp_path / "failures.jsonl",
        progress_path=tmp_path / "progress.json",
    )

    failure = json.loads(
        (tmp_path / "failures.jsonl").read_text(encoding="utf-8")
    )
    progress = json.loads(
        (tmp_path / "progress.json").read_text(encoding="utf-8")
    )
    assert failure["page_url"] == "https://e.test/private"
    assert failure["error_class"] == "RuntimeError"
    assert "hunter2" not in json.dumps(failure)
    assert progress["unique"] == 1
    assert progress["terminal"] == 1
    assert progress["inflight"] == 0
    assert result.failure_path == tmp_path / "failures.jsonl"


def test_complete_page_fetch_validator_binds_jobs_refs_and_snapshots(
    tmp_path: Path,
) -> None:
    success_url = "https://e.test/success"
    failure_url = "https://e.test/failure?secret=hidden"
    refs = [
        page_ref("e1", success_url),
        page_ref("e2", failure_url),
    ]
    result = fetch_unique_pages(
        refs,
        SqliteJobStore(tmp_path / "pages.sqlite3"),
        CountingTransport(
            {
                success_url: {
                    "text": "page text",
                    "image_urls": [],
                },
                failure_url: TimeoutError("private failure"),
            }
        ),
        FetchPolicy(),
    )

    guarded_paths: list[Path] = []
    snapshot = validate_complete_page_fetch(
        result,
        refs,
        validation_database=tmp_path / "validate.sqlite3",
        pre_write_guard=lambda path, _size=0: guarded_paths.append(
            Path(path)
        ),
    )

    assert snapshot["unique"] == 2
    assert snapshot["success"] == 1
    assert snapshot["terminal"] == 1
    assert len(snapshot["identity"]) == 64
    assert snapshot["failure_records"] == 1
    assert result.outcomes_path in guarded_paths
    assert tmp_path / "validate.sqlite3" in guarded_paths

    with pytest.raises(ValueError, match="reference"):
        validate_complete_page_fetch(
            result,
            refs[:1],
            validation_database=tmp_path / "foreign.sqlite3",
        )

    with pytest.raises(ValueError, match="job store"):
        validate_complete_page_fetch(
            replace(
                result,
                job_store_path=tmp_path / "missing.sqlite3",
            ),
            refs,
            validation_database=tmp_path / "missing-validate.sqlite3",
        )

    with sqlite3.connect(result.job_store_path) as connection:
        connection.execute(
            """
            UPDATE jobs SET payload_json = '{"forged":true}'
            WHERE kind = ?
            """,
            (result.job_kind,),
        )
    with pytest.raises(ValueError, match="job store membership"):
        validate_complete_page_fetch(
            result,
            refs,
            validation_database=tmp_path / "forged-job.sqlite3",
        )


def test_page_validation_recovers_after_mid_batch_guard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs = [
        page_ref(str(index), f"https://e.test/{index}")
        for index in range(20)
    ]
    result = fetch_unique_pages(
        refs,
        SqliteJobStore(tmp_path / "pages.sqlite3"),
        CountingTransport(
            {
                record["page_url"]: {"text": "page", "image_urls": []}
                for record in refs
            }
        ),
        FetchPolicy(),
    )
    validation_database = tmp_path / "validate.sqlite3"
    positive_calls = 0
    monkeypatch.setattr(
        fetch_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    def interrupt(path: Path, estimated_bytes: int = 0) -> None:
        nonlocal positive_calls
        assert Path(path) == validation_database
        if estimated_bytes > 0:
            positive_calls += 1
            if positive_calls == 2:
                raise RuntimeError("synthetic validation reserve exhausted")

    with pytest.raises(RuntimeError, match="validation reserve"):
        validate_complete_page_fetch(
            result,
            refs,
            validation_database=validation_database,
            pre_write_guard=interrupt,
        )

    snapshot = validate_complete_page_fetch(
        result,
        refs,
        validation_database=validation_database,
    )
    assert positive_calls == 2
    assert snapshot["unique"] == len(refs)


def test_wdc_client_exposes_read_only_cached_page_outcome(
    tmp_path: Path,
) -> None:
    client = WdcWebClient(tmp_path, session=_NoNetworkSession(), host_delay=0)
    expected = {
        "page_url": "https://e.test/page",
        "final_url": "https://e.test/final",
        "text": "hello",
        "image_urls": ["https://e.test/a.png"],
    }
    client._store_page(expected)

    outcome = client.cached_page_outcome(expected["page_url"])

    assert outcome == {
        "status": "success",
        **expected,
        "policy_fingerprint": "wdc-web-v1",
    }


def test_wdc_page_success_cache_is_scoped_to_network_policy(
    tmp_path: Path,
) -> None:
    page_url = "https://e.test/page"
    first = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
        network_policy_version="page-v1",
    )
    first._store_page(
        {
            "page_url": page_url,
            "final_url": page_url,
            "text": "old policy",
            "image_urls": [],
        }
    )
    changed = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
        network_policy_version="page-v2",
    )

    assert changed.cached_page_outcome(page_url) is None


def test_wdc_page_cache_preserves_v1_v2_v1_results(
    tmp_path: Path,
) -> None:
    page_url = "https://e.test/multi-policy"
    v1 = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
        network_policy_version="page-v1",
    )
    v2 = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
        network_policy_version="page-v2",
    )
    v1._store_page(
        {
            "page_url": page_url,
            "final_url": page_url,
            "text": "v1",
            "image_urls": [],
        }
    )
    v2._store_page(
        {
            "page_url": page_url,
            "final_url": page_url,
            "text": "v2",
            "image_urls": [],
        }
    )

    assert v1.cached_page_outcome(page_url)["text"] == "v1"
    assert v2.cached_page_outcome(page_url)["text"] == "v2"
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM page_cache WHERE page_url = ?",
            (page_url,),
        ).fetchone() == (2,)


def test_legacy_single_key_page_cache_migrates_without_losing_row(
    tmp_path: Path,
) -> None:
    database = tmp_path / "wdc_web.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE page_cache (
                page_url TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                final_url TEXT,
                text TEXT,
                image_urls_json TEXT,
                http_status INTEGER,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO page_cache VALUES (
                'https://e.test/legacy', 'success',
                'https://e.test/legacy', 'legacy', '[]', 200, NULL, 1.0
            )
            """
        )

    client = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
    )

    assert client.cached_page_outcome("https://e.test/legacy")["text"] == "legacy"
    with sqlite3.connect(database) as connection:
        primary_key = [
            row[1]
            for row in connection.execute("PRAGMA table_info(page_cache)")
            if row[5]
        ]
    assert primary_key == ["page_url", "policy_fingerprint"]


def test_concurrent_page_writes_in_different_policies_do_not_overwrite(
    tmp_path: Path,
) -> None:
    url = "https://e.test/concurrent-policy"
    clients = [
        WdcWebClient(
            tmp_path,
            session=_NoNetworkSession(),
            host_delay=0,
            network_policy_version=policy,
        )
        for policy in ("p1", "p2")
    ]
    barrier = threading.Barrier(2)

    def write(client: WdcWebClient, text: str) -> None:
        barrier.wait(timeout=1)
        client._store_page(
            {
                "page_url": url,
                "final_url": url,
                "text": text,
                "image_urls": [],
            }
        )

    threads = [
        threading.Thread(target=write, args=(clients[index], f"p{index + 1}"))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert clients[0].cached_page_outcome(url)["text"] == "p1"
    assert clients[1].cached_page_outcome(url)["text"] == "p2"


class _NoNetworkSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls = 0

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("network must not be used")


class _InvalidImageResponse:
    status_code = 200
    url = "https://i.test/broken.png"
    encoding = "utf-8"
    headers = {"Content-Type": "image/png"}

    def __enter__(self) -> "_InvalidImageResponse":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def iter_content(self, chunk_size: int) -> Any:
        del chunk_size
        yield b"not a raster"


class _PageResponse:
    status_code = 200
    encoding = "utf-8"
    headers = {"Content-Type": "text/html"}

    def __init__(self, url: str, text: str) -> None:
        self.url = url
        self.body = f"<p>{text}</p>".encode()

    def __enter__(self) -> "_PageResponse":
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def close(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        del chunk_size
        yield self.body


class _PageSession(_NoNetworkSession):
    def __init__(self, response: _PageResponse) -> None:
        super().__init__()
        self.response = response

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.response


class _OneResponseSession(_NoNetworkSession):
    def __init__(self) -> None:
        super().__init__()
        self.response = _InvalidImageResponse()

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.response


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class _RedirectResponse:
    status_code = 302
    url = "https://e.test/start"
    headers = {"Location": "/next"}

    def close(self) -> None:
        return None


class _RedirectSession(_NoNetworkSession):
    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return _RedirectResponse()


def test_redirect_host_delay_is_truncated_by_absolute_deadline(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    session = _RedirectSession()
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=1.0,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
        max_retries=0,
    )

    assert client.fetch_page(
        "https://e.test/start",
        deadline_seconds=0.05,
        max_retries=0,
    ) is None
    assert session.calls == 1
    assert clock.sleeps == [pytest.approx(0.05)]


def test_real_wdc_stage_cache_round_trip_v1_v2_v1(
    tmp_path: Path,
) -> None:
    url = "https://e.test/stage-policy"
    store = SqliteJobStore(tmp_path / "pages.sqlite3")
    sessions = {
        "v1": _PageSession(_PageResponse(url, "version one")),
        "v2": _PageSession(_PageResponse(url, "version two")),
    }
    results = {}
    for version in ("v1", "v2"):
        client = WdcWebClient(
            tmp_path / "web",
            session=sessions[version],
            host_delay=0,
            network_policy_version=version,
        )
        results[version] = fetch_unique_pages(
            [page_ref("e1", url)],
            store,
            client,
            FetchPolicy(network_policy_fingerprint=version),
        )
    offline = _NoNetworkSession()
    v1_again = WdcWebClient(
        tmp_path / "web",
        session=offline,
        host_delay=0,
        network_policy_version="v1",
    )
    resumed = fetch_unique_pages(
        [page_ref("e1", url)],
        store,
        v1_again,
        FetchPolicy(network_policy_fingerprint="v1"),
    )

    assert sessions["v1"].calls == 1
    assert sessions["v2"].calls == 1
    assert offline.calls == 0
    assert resumed.complete is True
    assert {
        outcome["text"]
        for outcome in iter_page_outcomes(results["v1"].outcomes_path)
    } == {"version one", "version two"}


def test_invalid_image_is_negative_cached_for_policy(tmp_path: Path) -> None:
    session = _NoNetworkSession()
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        network_policy_version="image-v1",
    )

    assert client.download_image(
        "not-a-url",
        page_url="https://e.test/page",
        source="wdc_page_image",
        entity_id="e1",
    ) is None
    assert client.cached_image_outcome("not-a-url") == {
        "status": "terminal",
        "image_url": "not-a-url",
        "error_class": "invalid_image_url",
        "policy_fingerprint": "image-v1",
    }
    assert client.download_image(
        "not-a-url",
        page_url="https://e.test/page",
        source="wdc_page_image",
        entity_id="e2",
    ) is None
    assert session.calls == 0


def test_failed_image_download_is_negative_cached_until_policy_changes(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/broken.png"
    first_session = _OneResponseSession()
    first = WdcWebClient(
        tmp_path,
        session=first_session,
        host_delay=0,
        network_policy_version="image-v1",
    )
    assert first.download_image(
        image_url,
        page_url="https://e.test/page",
        source="wdc_page_image",
        entity_id="e1",
    ) is None
    assert first_session.calls == 1
    assert first.cached_image_outcome(image_url)["status"] == "terminal"

    offline = _NoNetworkSession()
    same_policy = WdcWebClient(
        tmp_path,
        session=offline,
        host_delay=0,
        network_policy_version="image-v1",
    )
    assert same_policy.download_image(
        image_url,
        page_url="https://e.test/page",
        source="wdc_page_image",
        entity_id="e2",
    ) is None
    assert offline.calls == 0

    refreshed_session = _OneResponseSession()
    refreshed = WdcWebClient(
        tmp_path,
        session=refreshed_session,
        host_delay=0,
        network_policy_version="image-v2",
    )
    assert refreshed.download_image(
        image_url,
        page_url="https://e.test/page",
        source="wdc_page_image",
        entity_id="e3",
    ) is None
    assert refreshed_session.calls == 1


@pytest.mark.parametrize("damage", ["missing", "sha_mismatch", "invalid_raster"])
def test_cached_image_success_validates_file_without_network_or_mutation(
    tmp_path: Path,
    damage: str,
) -> None:
    client = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
    )
    client.image_dir.mkdir()
    path = client.image_dir / "cached.png"
    path.write_bytes(png_bytes())
    image_url = "https://i.test/cached.png"
    client._store_image_index(
        original_url=image_url,
        final_url=image_url,
        path=path,
        width=64,
        height=48,
        mime_type="image/png",
    )
    if damage == "missing":
        path.unlink()
    elif damage == "sha_mismatch":
        path.write_bytes(png_bytes((200, 10, 10)))
    else:
        path.write_bytes(b"not a raster")

    assert client.cached_image_outcome(image_url) is None
    with sqlite3.connect(client.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_cache WHERE original_url = ?",
            (image_url,),
        ).fetchone() == (1,)


def test_cached_image_race_with_file_deletion_returns_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
    )
    client.image_dir.mkdir()
    path = client.image_dir / "racy.png"
    path.write_bytes(png_bytes())
    image_url = "https://i.test/racy.png"
    client._store_image_index(
        original_url=image_url,
        final_url=image_url,
        path=path,
        width=64,
        height=48,
        mime_type="image/png",
    )

    def validate_then_delete(_path: Path) -> tuple[int, int, str, str]:
        path.unlink()
        return 64, 48, "image/png", ".png"

    monkeypatch.setattr(client, "_validated_raster", validate_then_delete)

    assert client.cached_image_outcome(image_url) is None


def test_image_success_cache_keeps_independent_policy_rows(
    tmp_path: Path,
) -> None:
    image_url = "https://i.test/policy.png"
    outcomes: dict[str, dict[str, Any] | None] = {}
    for index, policy in enumerate(("image-v1", "image-v2")):
        client = WdcWebClient(
            tmp_path,
            session=_NoNetworkSession(),
            host_delay=0,
            network_policy_version=policy,
        )
        client.image_dir.mkdir(exist_ok=True)
        path = client.image_dir / f"{policy}.png"
        path.write_bytes(png_bytes((20 + index, 40, 60)))
        client._store_image_index(
            original_url=image_url,
            final_url=image_url,
            path=path,
            width=64,
            height=48,
            mime_type="image/png",
        )
        outcomes[policy] = client.cached_image_outcome(image_url)

    assert outcomes["image-v1"]["file_name"] == "image-v1.png"
    assert outcomes["image-v2"]["file_name"] == "image-v2.png"
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_cache WHERE original_url = ?",
            (image_url,),
        ).fetchone() == (2,)


def test_legacy_single_key_image_cache_migrates_with_verified_asset(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "wdc_images"
    image_dir.mkdir()
    path = image_dir / "legacy.png"
    body = png_bytes()
    path.write_bytes(body)
    database = tmp_path / "wdc_web.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE image_cache (
                original_url TEXT PRIMARY KEY,
                final_url TEXT NOT NULL,
                file_name TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO image_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "https://i.test/legacy.png",
                "https://i.test/legacy.png",
                path.name,
                hashlib.sha256(body).hexdigest(),
                64,
                48,
                "image/png",
                1.0,
            ),
        )

    client = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
    )

    assert client.cached_image_outcome(
        "https://i.test/legacy.png"
    )["status"] == "success"
    with sqlite3.connect(database) as connection:
        primary_key = [
            row[1]
            for row in connection.execute("PRAGMA table_info(image_cache)")
            if row[5]
        ]
    assert primary_key == ["original_url", "policy_fingerprint"]


def test_image_success_and_negative_are_mutually_exclusive_per_policy(
    tmp_path: Path,
) -> None:
    client = WdcWebClient(
        tmp_path,
        session=_NoNetworkSession(),
        host_delay=0,
        network_policy_version="image-policy",
    )
    client.image_dir.mkdir()
    path = client.image_dir / "valid.png"
    path.write_bytes(png_bytes())
    image_url = "https://i.test/state.png"

    client._store_image_failure(image_url, error_class="TimeoutError")
    client._store_image_index(
        original_url=image_url,
        final_url=image_url,
        path=path,
        width=64,
        height=48,
        mime_type="image/png",
    )
    client._store_image_failure(image_url, error_class="late_failure")

    assert client.cached_image_outcome(image_url)["status"] == "success"
    with sqlite3.connect(client.database_path) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM image_failure_cache
            WHERE original_url = ? AND policy_fingerprint = ?
            """,
            (image_url, "image-policy"),
        ).fetchone() == (0,)


def test_finalized_page_refs_reject_incomplete_global_barrier(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "stage_manifests" / "validated-selection-global.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_validated_selection",
                "complete": False,
                "completed_shards": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="barrier is incomplete"):
        list(iter_finalized_page_refs(tmp_path, manifest))


def test_finalized_page_refs_are_bound_to_structural_manifests(
    tmp_path: Path,
) -> None:
    ref = page_ref("e1", "https://e.test/page")
    page_writer = AtomicJsonlShard(
        tmp_path / "page_refs" / "part-00000.jsonl"
    )
    page_writer.write(ref)
    page_completed = asdict(page_writer.commit())
    page_completed["path"] = "page_refs/part-00000.jsonl"

    selection_writer = AtomicJsonlShard(
        tmp_path / "selection" / "validated-selected-tables.jsonl"
    )
    selection_writer.write({"source_table_id": "t1"})
    selection_completed = asdict(selection_writer.commit())
    selection_completed["path"] = (
        "selection/validated-selected-tables.jsonl"
    )

    structural_manifest = (
        tmp_path / "stage_manifests" / "structural-00000.json"
    )
    structural_manifest.parent.mkdir(parents=True)
    structural_manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_structural",
                "complete": True,
                "completed_shards": [
                    page_completed,
                    {
                        "path": "selection/validated-00000.jsonl",
                        "records": 1,
                        "bytes": 0,
                        "sha256": "0" * 64,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(structural_manifest.read_bytes()).hexdigest()
    barrier = (
        tmp_path / "stage_manifests" / "validated-selection-global.json"
    )
    barrier.write_text(
        json.dumps(
            {
                "stage": "wdc200k_validated_selection",
                "complete": True,
                "input_fingerprint": stable_hash(
                    "wdc200k-structural-v2",
                    f"{structural_manifest.resolve().as_posix()}:{digest}",
                    length=40,
                ),
                "completed_shards": [selection_completed],
            }
        ),
        encoding="utf-8",
    )

    assert list(iter_finalized_page_refs(tmp_path, barrier)) == [ref]


@pytest.mark.parametrize("slow_part", ["headers", "body"])
def test_wdc_transport_deadline_covers_headers_and_body(
    tmp_path: Path,
    slow_part: str,
) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def serve() -> None:
        try:
            connection, _address = server.accept()
            with connection:
                connection.recv(4096)
                if slow_part == "headers":
                    payload = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
                else:
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/html\r\n"
                        b"Content-Length: 20\r\n\r\n"
                    )
                    payload = b"<p>slow response</p>"
                for byte in payload:
                    try:
                        connection.sendall(bytes([byte]))
                    except OSError:
                        break
                    time.sleep(0.01)
        finally:
            server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def local_pinned_request(url: str, **kwargs: Any) -> Any:
        return wdc_builder._pinned_http_get(
            url,
            pinned_ip="127.0.0.1",
            server_hostname=str(kwargs["server_hostname"]),
            port=int(kwargs["port"]),
            headers=dict(kwargs["headers"]),
            timeout=kwargs["timeout"],
            deadline=float(kwargs["deadline"]),
            monotonic_fn=kwargs["monotonic_fn"],
        )

    client = WdcWebClient(
        tmp_path,
        host_delay=0,
        resolve_host_fn=lambda _host: ["93.184.216.34"],
        pinned_request_fn=local_pinned_request,
    )
    started = time.monotonic()

    assert client.fetch_page(
        f"http://deadline.test:{port}/{slow_part}",
        deadline_seconds=0.08,
        max_retries=0,
    ) is None
    assert time.monotonic() - started < 0.4
    thread.join(timeout=1)
    assert not thread.is_alive()

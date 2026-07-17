from __future__ import annotations

import hashlib
import json
import socket
import sqlite3
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_wdc_mm_joinability_dataset as wdc_builder  # noqa: E402
from build_wdc_mm_joinability_dataset import WdcWebClient  # noqa: E402
from wdc200k_fetch import (  # noqa: E402
    FetchPolicy,
    fetch_unique_pages,
    iter_finalized_page_refs,
    iter_page_outcomes,
)
from wdc200k_io import SqliteJobStore  # noqa: E402
from wdc200k_io import AtomicJsonlShard  # noqa: E402
from stage1_io import stable_hash  # noqa: E402


def page_ref(entity_id: str, url: str) -> dict[str, str]:
    return {
        "entity_id": entity_id,
        "page_url": url,
        "url_key": hashlib.sha256(url.encode("utf-8")).hexdigest(),
    }


class CountingTransport:
    def __init__(
        self,
        outcomes: dict[str, dict[str, Any] | BaseException | None],
    ) -> None:
        self.outcomes = outcomes
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


class _OneResponseSession(_NoNetworkSession):
    def __init__(self) -> None:
        super().__init__()
        self.response = _InvalidImageResponse()

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        return self.response


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

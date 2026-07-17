import gzip
import hashlib
import io
import json
import socket
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import dns.resolver
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_wdc_mm_joinability_dataset as wdc_builder
from build_wdc_mm_joinability_dataset import (
    WdcWebClient,
    extract_html_assets,
    extract_image_urls,
    read_wdc_table,
)
from stage1_io import get_cell, get_cell_text, stable_hash


def write_gzip_rows(tmp_path: Path, rows: list[Any], name: str = "Thing_host.json.gz") -> Path:
    path = tmp_path / "Thing" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row)
            else:
                handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    return path


def png_bytes(
    width: int = 64,
    height: int = 48,
    color: tuple[int, int, int] = (20, 40, 60),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status_code: int = 200,
        url: str = "https://example.test/final",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = body
        self.status_code = status_code
        self.url = url
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset : offset + chunk_size]

    def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeWdcAssetClient:
    def __init__(
        self,
        *,
        page: dict[str, Any] | None | Exception,
        image_outcomes: dict[str, str | None | Exception],
    ) -> None:
        self.page = page
        self.image_outcomes = image_outcomes
        self.fetch_page_calls: list[str] = []
        self.download_image_calls: list[tuple[str, dict[str, str]]] = []

    def fetch_page(self, page_url: str) -> dict[str, Any] | None:
        self.fetch_page_calls.append(page_url)
        if isinstance(self.page, Exception):
            raise self.page
        return self.page

    def download_image(self, image_url: str, **kwargs: str) -> dict[str, Any] | None:
        self.download_image_calls.append((image_url, kwargs))
        outcome = self.image_outcomes.get(image_url)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return {
            "asset_id": f"asset_img_{stable_hash(kwargs['entity_id'], kwargs['source'], image_url)}",
            "entity_id": kwargs["entity_id"],
            "asset_type": "image",
            "source": kwargs["source"],
            "image_url": image_url,
            "original_url": image_url,
            "page_url": kwargs["page_url"],
            "sha256": outcome,
        }


def wdc_asset_entity() -> dict[str, Any]:
    return {
        "entity_id": "ent_alpha",
        "wiki_title": "wdc_alpha",
        "display_texts": ["Alpha"],
        "context_terms": ["bridge evidence"],
        "appears_in": [{"column_name": "name"}],
        "page_url": "https://x.test/entity",
        "image_urls": [
            "https://img.test/direct-1.jpg",
            "https://img.test/direct-2.jpg",
        ],
    }


def test_direct_images_take_quota_but_page_is_always_fetched():
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/final",
            "text": "Alpha has useful bridge evidence on its entity page.",
            "image_urls": [
                "https://img.test/page-1.jpg",
                "https://img.test/page-2.jpg",
            ],
        },
        image_outcomes={
            "https://img.test/direct-1.jpg": "sha-direct",
            "https://img.test/direct-2.jpg": None,
            "https://img.test/page-1.jpg": "sha-page",
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(),
        client,
        2,
        800,
        120,
        3,
    )

    assert client.fetch_page_calls == ["https://x.test/entity"]
    images = [record for record in records if record["asset_type"] == "image"]
    assert [record["source"] for record in images] == [
        "wdc_image_column",
        "wdc_page_image",
    ]
    assert len(images) == 2
    assert any(record["asset_type"] == "text" for record in records)
    text_record = next(record for record in records if record["asset_type"] == "text")
    assert text_record["page_url"] == "https://x.test/entity"
    assert text_record["final_url"] == "https://x.test/final"


def test_failed_direct_images_leave_full_quota_for_page_images():
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/entity",
            "text": "Alpha page text.",
            "image_urls": [
                "https://img.test/page-1.jpg",
                "https://img.test/page-2.jpg",
            ],
        },
        image_outcomes={
            "https://img.test/direct-1.jpg": None,
            "https://img.test/direct-2.jpg": None,
            "https://img.test/page-1.jpg": "sha-page-1",
            "https://img.test/page-2.jpg": "sha-page-2",
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(), client, 2
    )

    images = [record for record in records if record["asset_type"] == "image"]
    assert [record["source"] for record in images] == [
        "wdc_page_image",
        "wdc_page_image",
    ]
    assert [call[0] for call in client.download_image_calls] == [
        "https://img.test/direct-1.jpg",
        "https://img.test/direct-2.jpg",
        "https://img.test/page-1.jpg",
        "https://img.test/page-2.jpg",
    ]


def test_full_direct_quota_still_fetches_page_text_without_page_image_downloads():
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/entity",
            "text": "Alpha page text remains mandatory.",
            "image_urls": ["https://img.test/page-unused.jpg"],
        },
        image_outcomes={
            "https://img.test/direct-1.jpg": "sha-direct-1",
            "https://img.test/direct-2.jpg": "sha-direct-2",
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(), client, 2
    )

    assert client.fetch_page_calls == ["https://x.test/entity"]
    assert [call[0] for call in client.download_image_calls] == [
        "https://img.test/direct-1.jpg",
        "https://img.test/direct-2.jpg",
    ]
    assert any(record["asset_type"] == "text" for record in records)


def test_zero_image_quota_still_fetches_and_emits_page_text():
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/entity",
            "text": "Alpha page text is emitted with a zero image quota.",
            "image_urls": ["https://img.test/page-unused.jpg"],
        },
        image_outcomes={},
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(), client, 0
    )

    assert client.fetch_page_calls == ["https://x.test/entity"]
    assert client.download_image_calls == []
    assert [record["asset_type"] for record in records] == ["text"]


def test_page_text_uses_unlimited_split_then_exact_relevance_limit(monkeypatch):
    calls: dict[str, Any] = {}

    def fake_split(content, *, max_chars, min_chars, max_chunks):
        calls["split"] = (content, max_chars, min_chars, max_chunks)
        return ["zero", "one Alpha", "two", "three bridge evidence"]

    def fake_select(chunks, entity, max_chunks):
        calls["select"] = (chunks, entity, max_chunks)
        return [(1, chunks[1], 4.0), (3, chunks[3], 2.0)]

    monkeypatch.setattr(wdc_builder, "split_text_asset_content", fake_split)
    monkeypatch.setattr(wdc_builder, "select_relevant_text_chunks", fake_select)
    entity = wdc_asset_entity()
    client = FakeWdcAssetClient(
        page={"final_url": "https://x.test/final", "text": "raw page", "image_urls": []},
        image_outcomes={},
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        entity,
        client,
        max_images_per_entity=0,
        text_asset_chunk_chars=91,
        min_text_asset_chunk_chars=17,
        max_text_asset_chunks_per_entity=2,
    )

    assert calls["split"] == ("raw page", 91, 17, 0)
    assert calls["select"] == (
        ["zero", "one Alpha", "two", "three bridge evidence"],
        entity,
        2,
    )
    assert [record["text_chunk_index"] for record in records] == [1, 3]
    assert {record["text_chunk_count"] for record in records} == {4}
    assert {record["selected_text_chunk_count"] for record in records} == {2}


def test_page_text_chunk_defaults_remain_800_120_and_3(monkeypatch):
    calls: dict[str, Any] = {}

    def fake_split(content, *, max_chars, min_chars, max_chunks):
        calls["split"] = (max_chars, min_chars, max_chunks)
        return ["text"]

    def fake_select(chunks, entity, max_chunks):
        calls["select_limit"] = max_chunks
        return [(0, chunks[0], 0.0)]

    monkeypatch.setattr(wdc_builder, "split_text_asset_content", fake_split)
    monkeypatch.setattr(wdc_builder, "select_relevant_text_chunks", fake_select)
    client = FakeWdcAssetClient(
        page={"final_url": "https://x.test/entity", "text": "raw", "image_urls": []},
        image_outcomes={},
    )

    wdc_builder.build_wdc_bridge_assets_for_entity(wdc_asset_entity(), client, 0)

    assert calls == {"split": (800, 120, 0), "select_limit": 3}


def test_image_candidates_are_deduplicated_by_url_and_returned_sha():
    entity = wdc_asset_entity()
    entity["image_urls"] = [
        "https://img.test/direct-1.jpg",
        "https://img.test/direct-1.jpg",
        "https://img.test/direct-same-sha.jpg",
    ]
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/entity",
            "text": "Alpha page text.",
            "image_urls": [
                "https://img.test/direct-1.jpg",
                "https://img.test/page-same-sha.jpg",
                "https://img.test/page-unique-1.jpg",
                "https://img.test/page-unique-2.jpg",
            ],
        },
        image_outcomes={
            "https://img.test/direct-1.jpg": "sha-shared",
            "https://img.test/direct-same-sha.jpg": "sha-shared",
            "https://img.test/page-same-sha.jpg": "sha-shared",
            "https://img.test/page-unique-1.jpg": "sha-page-1",
            "https://img.test/page-unique-2.jpg": "sha-page-2",
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(entity, client, 3)

    images = [record for record in records if record["asset_type"] == "image"]
    assert [record["image_url"] for record in images] == [
        "https://img.test/direct-1.jpg",
        "https://img.test/page-unique-1.jpg",
        "https://img.test/page-unique-2.jpg",
    ]
    assert [call[0] for call in client.download_image_calls] == [
        "https://img.test/direct-1.jpg",
        "https://img.test/direct-same-sha.jpg",
        "https://img.test/page-same-sha.jpg",
        "https://img.test/page-unique-1.jpg",
        "https://img.test/page-unique-2.jpg",
    ]


def test_individual_download_exceptions_do_not_abort_remaining_candidates():
    client = FakeWdcAssetClient(
        page={
            "final_url": "https://x.test/entity",
            "text": "Alpha page text.",
            "image_urls": [
                "https://img.test/page-error.jpg",
                "https://img.test/page-ok.jpg",
            ],
        },
        image_outcomes={
            "https://img.test/direct-1.jpg": RuntimeError("direct failed"),
            "https://img.test/direct-2.jpg": "sha-direct",
            "https://img.test/page-error.jpg": RuntimeError("page image failed"),
            "https://img.test/page-ok.jpg": "sha-page",
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(), client, 2
    )

    images = [record for record in records if record["asset_type"] == "image"]
    assert [record["source"] for record in images] == [
        "wdc_image_column",
        "wdc_page_image",
    ]


def test_fetch_page_exception_isolated_while_direct_images_are_still_attempted():
    client = FakeWdcAssetClient(
        page=RuntimeError("page failed"),
        image_outcomes={
            "https://img.test/direct-1.jpg": "sha-direct",
            "https://img.test/direct-2.jpg": None,
        },
    )

    records = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(), client, 1
    )

    assert client.fetch_page_calls == ["https://x.test/entity"]
    assert [record["asset_type"] for record in records] == ["image"]
    assert records[0]["source"] == "wdc_image_column"


def test_page_text_records_keep_bridge_contract_and_stable_ids():
    page = {
        "final_url": "https://x.test/final",
        "text": "Alpha has stable webpage text bridge evidence.",
        "image_urls": [],
    }

    first = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(),
        FakeWdcAssetClient(page=page, image_outcomes={}),
        0,
    )
    second = wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(),
        FakeWdcAssetClient(page=page, image_outcomes={}),
        0,
    )

    assert [record["asset_id"] for record in first] == [
        record["asset_id"] for record in second
    ]
    assert first[0] == {
        "asset_id": first[0]["asset_id"],
        "source_asset_id": first[0]["source_asset_id"],
        "entity_id": "ent_alpha",
        "entity_wiki_title": "wdc_alpha",
        "asset_type": "text",
        "content": "Alpha has stable webpage text bridge evidence.",
        "text_chunk_index": 0,
        "text_chunk_count": 1,
        "selected_text_chunk_count": 1,
        "text_chunk_relevance_score": 6.0,
            "source": "wdc_page_text_chunk",
            "url": "https://x.test/final",
            "page_url": "https://x.test/entity",
            "final_url": "https://x.test/final",
        }


def test_extract_image_urls_recurses_resolves_and_deduplicates():
    value = {
        "contentUrl": "/a.jpg",
        "nested": [
            "https://cdn.test/b.png",
            {"url": "/a.jpg", "ignored": "data:image/png;base64,abc"},
        ],
    }

    assert extract_image_urls(value, "https://example.test/path/page") == [
        "https://example.test/a.jpg",
        "https://cdn.test/b.png",
    ]


def test_extract_image_urls_skips_malformed_urls_without_losing_valid_urls():
    assert extract_image_urls(
        ["http://[", "https://cdn.test/valid.jpg"],
        "https://example.test/page",
    ) == ["https://cdn.test/valid.jpg"]


def test_extract_html_assets_keeps_content_and_resolves_images():
    text, images = extract_html_assets(
        '<html><head><title>Entity page</title>'
        '<meta name="description" content="A useful summary.">'
        '<meta property="og:image" content="/hero.jpg"></head>'
        '<body><nav>menu</nav><h1>Entity A</h1><p>Useful description.</p>'
        '<img data-src="photo.png"><script>ignored()</script></body></html>',
        "https://example.test/path/page",
    )

    assert "Entity page" in text
    assert "A useful summary." in text
    assert "Entity A" in text
    assert "Useful description." in text
    assert "menu" not in text
    assert "ignored" not in text
    assert images == [
        "https://example.test/hero.jpg",
        "https://example.test/path/photo.png",
    ]


def test_extract_html_assets_keeps_lazy_image_when_src_is_data_url():
    _text, images = extract_html_assets(
        '<img src="data:image/gif;base64,AAAA" data-src="real.jpg">',
        "https://example.test/path/page",
    )

    assert images == ["https://example.test/path/real.jpg"]


def test_fetch_page_reuses_sqlite_cache_across_clients(tmp_path):
    page_url = "https://example.test/page"
    response = FakeResponse(
        b'<html><h1>Cached entity</h1><img src="/photo.jpg"></html>',
        url="https://example.test/final/page",
    )
    first_client = WdcWebClient(
        tmp_path,
        session=FakeSession([response]),
        host_delay=0,
    )

    first = first_client.fetch_page(page_url)

    assert first == {
        "page_url": page_url,
        "final_url": "https://example.test/final/page",
        "text": "Cached entity",
        "image_urls": ["https://example.test/photo.jpg"],
    }
    offline_session = FakeSession([AssertionError("network must not be used")])
    second_client = WdcWebClient(tmp_path, session=offline_session, host_delay=0)
    assert second_client.fetch_page(page_url) == first
    assert offline_session.calls == []
    assert (tmp_path / "wdc_web.sqlite3").is_file()


def test_web_client_requests_identity_content_encoding(tmp_path):
    session = FakeSession([])

    WdcWebClient(tmp_path, session=session, host_delay=0)

    assert session.headers["Accept-Encoding"] == "identity"


def test_fetch_page_rejects_and_records_unexpected_content_encoding(tmp_path):
    failures: list[dict[str, Any]] = []
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    gzip.compress(b"<p>must not parse compressed bytes</p>"),
                    headers={
                        "Content-Type": "text/html; charset=utf-8",
                        "Content-Encoding": "gzip",
                    },
                )
            ]
        ),
        host_delay=0,
        max_retries=0,
        web_failure_callback=failures.append,
    )

    assert client.fetch_page("https://example.test/compressed") is None
    assert failures[-1]["status"] == "terminal"
    assert "unsupported_content_encoding:gzip" in failures[-1]["error"]


def test_page_deadline_includes_initial_dns_resolution(tmp_path):
    clock = FakeClock()
    session = FakeSession([FakeResponse(b"<p>too late</p>")])

    def slow_resolver(_host: str) -> list[str]:
        clock.now += 0.6
        return ["93.184.216.34"]

    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=0,
        max_response_seconds=0.5,
        monotonic_fn=clock.monotonic,
        resolve_host_fn=slow_resolver,
    )

    assert client.fetch_page("https://example.com/slow-dns") is None
    assert session.calls == []


def test_production_dns_timeout_does_not_fall_back_to_blocked_getaddrinfo(
    tmp_path,
    monkeypatch,
):
    blackhole = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    blackhole.bind(("127.0.0.1", 0))
    blackhole.settimeout(1.0)
    dns_port = int(blackhole.getsockname()[1])
    dns_query_received = threading.Event()
    stop_blackhole = threading.Event()

    def discard_dns_query() -> None:
        try:
            try:
                blackhole.recvfrom(4096)
            except TimeoutError:
                return
            else:
                dns_query_received.set()
                stop_blackhole.wait(timeout=1.0)
        finally:
            blackhole.close()

    blackhole_thread = threading.Thread(target=discard_dns_query, daemon=True)
    blackhole_thread.start()
    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = ["127.0.0.1"]
    resolver.port = dns_port
    resolver.timeout = 5.0
    monkeypatch.setattr(dns.resolver, "Resolver", lambda: resolver)

    stdlib_started = threading.Event()
    release_stdlib = threading.Event()

    def blocked_getaddrinfo(*_args: Any, **_kwargs: Any) -> list[Any]:
        stdlib_started.set()
        release_stdlib.wait(timeout=1.0)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", blocked_getaddrinfo)
    session = FakeSession([FakeResponse(b"<p>must not be fetched</p>")])
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=0,
        max_response_seconds=0.1,
    )
    completed = threading.Event()

    def fetch() -> None:
        try:
            client.fetch_page("https://blocked-dns.example/entity")
        finally:
            completed.set()

    worker = threading.Thread(target=fetch)
    worker.start()
    try:
        assert completed.wait(timeout=0.5), "DNS resolution exceeded its deadline"
        assert dns_query_received.is_set()
        assert not stdlib_started.is_set()
        assert session.calls == []
    finally:
        release_stdlib.set()
        stop_blackhole.set()
        worker.join(timeout=1.0)
        blackhole_thread.join(timeout=1.0)


def test_pinned_response_deadline_interrupts_blocking_drip_read():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def serve_drip() -> None:
        try:
            connection, _address = server.accept()
            with connection:
                connection.recv(4096)
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/html\r\n"
                    b"Content-Length: 40\r\n\r\n"
                )
                for _index in range(40):
                    try:
                        connection.sendall(b"x")
                    except OSError:
                        break
                    time.sleep(0.02)
        finally:
            server.close()

    thread = threading.Thread(target=serve_drip, daemon=True)
    thread.start()
    started = time.monotonic()
    response = wdc_builder._pinned_http_get(
        f"http://localhost:{port}/drip",
        pinned_ip="127.0.0.1",
        server_hostname="localhost",
        port=port,
        headers={"Accept-Encoding": "identity"},
        timeout=(0.5, 0.5),
        deadline=started + 0.1,
        monotonic_fn=time.monotonic,
    )

    with response, pytest.raises(TimeoutError):
        list(response.iter_content(chunk_size=64 * 1024))
    assert time.monotonic() - started < 0.5
    thread.join(timeout=1.0)


def test_pinned_response_deadline_interrupts_dripping_headers():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def serve_drip() -> None:
        try:
            connection, _address = server.accept()
            with connection:
                connection.recv(4096)
                for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n":
                    try:
                        connection.sendall(bytes([byte]))
                    except OSError:
                        break
                    time.sleep(0.02)
        finally:
            server.close()

    thread = threading.Thread(target=serve_drip, daemon=True)
    thread.start()
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        wdc_builder._pinned_http_get(
            f"http://localhost:{port}/drip-headers",
            pinned_ip="127.0.0.1",
            server_hostname="localhost",
            port=port,
            headers={"Accept-Encoding": "identity"},
            timeout=(0.5, 0.5),
            deadline=started + 0.1,
            monotonic_fn=time.monotonic,
        )
    assert time.monotonic() - started < 0.5
    thread.join(timeout=1.0)


def test_fetch_page_caches_terminal_failure_without_raising(tmp_path):
    page_url = "https://example.test/missing"
    first_client = WdcWebClient(
        tmp_path,
        session=FakeSession([FakeResponse(b"missing", status_code=404)]),
        host_delay=0,
    )

    assert first_client.fetch_page(page_url) is None

    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT status, http_status FROM page_cache WHERE page_url = ?",
            (page_url,),
        ).fetchone() == ("terminal", 404)
    offline_session = FakeSession([AssertionError("terminal failure must be cached")])
    second_client = WdcWebClient(tmp_path, session=offline_session, host_delay=0)
    assert second_client.fetch_page(page_url) is None
    assert offline_session.calls == []


def test_page_failure_store_does_not_downgrade_existing_success(tmp_path):
    page_url = "https://example.test/race"
    offline_session = FakeSession([AssertionError("success cache must be retained")])
    client = WdcWebClient(
        tmp_path,
        session=offline_session,
        host_delay=0,
        max_retries=0,
    )
    expected = {
        "page_url": page_url,
        "final_url": "https://example.test/final",
        "text": "Successful extraction",
        "image_urls": ["https://example.test/image.png"],
    }
    client._store_page(expected)

    client._store_page_failure(
        page_url,
        status="retryable",
        http_status=503,
        error="late concurrent failure",
    )

    assert client.fetch_page(page_url) == expected
    assert offline_session.calls == []
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT status, text FROM page_cache WHERE page_url = ?",
            (page_url,),
        ).fetchone() == ("success", "Successful extraction")


def test_fetch_page_records_retryable_failure_and_recovers_on_next_client(tmp_path):
    page_url = "https://example.test/temporary"
    failed_client = WdcWebClient(
        tmp_path,
        session=FakeSession([TimeoutError("timed out")]),
        host_delay=0,
        max_retries=0,
    )

    assert failed_client.fetch_page(page_url) is None
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT status, http_status FROM page_cache WHERE page_url = ?",
            (page_url,),
        ).fetchone() == ("retryable", None)

    recovered_client = WdcWebClient(
        tmp_path,
        session=FakeSession([FakeResponse(b"<p>Recovered page.</p>")]),
        host_delay=0,
    )
    assert recovered_client.fetch_page(page_url)["text"] == "Recovered page."
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT status FROM page_cache WHERE page_url = ?",
            (page_url,),
        ).fetchone() == ("success",)


def test_fetch_page_retries_retryable_status_with_injected_sleep(tmp_path):
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(b"busy", status_code=503),
            FakeResponse(b"<p>Available now.</p>"),
        ]
    )
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=1,
        retry_base_seconds=0.25,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    assert client.fetch_page("https://example.test/retry")["text"] == "Available now."
    assert len(session.calls) == 2
    assert clock.sleeps == [0.25]


def test_fetch_page_enforces_host_delay_with_injected_clock(tmp_path):
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(b"<p>First.</p>"),
            FakeResponse(b"<p>Second.</p>"),
        ]
    )
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=1.5,
        max_retries=0,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    assert client.fetch_page("https://example.test/first") is not None
    assert client.fetch_page("https://example.test/second") is not None

    assert clock.sleeps == [1.5]


def test_download_image_rejects_oversized_stream_and_removes_temporary_file(tmp_path):
    session = FakeSession(
        [
            FakeResponse(
                b"x" * 32,
                url="https://cdn.test/large.png",
                headers={"Content-Type": "image/png"},
            )
        ]
    )
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=0,
        max_image_bytes=16,
    )

    assert client.download_image(
        "https://cdn.test/large.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_1",
    ) is None
    assert session.calls[0][1]["stream"] is True
    image_dir = tmp_path / "wdc_images"
    assert not image_dir.exists() or list(image_dir.iterdir()) == []


def test_download_image_validates_raster_and_reuses_verified_cache(tmp_path):
    image_url = "https://cdn.test/photo"
    body = png_bytes()
    first_client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    body,
                    url="https://media.test/final.png",
                    headers={"Content-Type": "image/png"},
                )
            ]
        ),
        host_delay=0,
        max_retries=0,
    )

    first = first_client.download_image(
        image_url,
        page_url="https://example.test/page",
        source="wdc_image_column",
        entity_id="ent_1",
    )

    assert first is not None
    assert first["asset_type"] == "image"
    assert first["entity_id"] == "ent_1"
    assert first["source"] == "wdc_image_column"
    assert first["original_url"] == image_url
    assert first["final_url"] == "https://media.test/final.png"
    assert first["page_url"] == "https://example.test/page"
    assert first["bytes"] == len(body)
    assert first["sha256"] == hashlib.sha256(body).hexdigest()
    assert first["width"] == 64
    assert first["height"] == 48
    assert first["mime_type"] == "image/png"
    assert first["downloaded"] is True
    assert Path(first["local_path"]).is_file()
    assert first["relative_path"].startswith("wdc_images/")

    offline_session = FakeSession([AssertionError("verified image cache must be reused")])
    second_client = WdcWebClient(tmp_path, session=offline_session, host_delay=0)
    second = second_client.download_image(
        image_url,
        page_url="https://example.test/other",
        source="wdc_page_image",
        entity_id="ent_2",
    )
    assert second is not None
    assert second["downloaded"] is False
    assert second["local_path"] == first["local_path"]
    assert second["entity_id"] == "ent_2"
    assert second["final_url"] == first["final_url"]
    assert offline_session.calls == []


def test_download_image_replaces_cache_file_when_sha_does_not_match_index(tmp_path):
    image_url = "https://cdn.test/tampered.png"
    first_client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [FakeResponse(png_bytes(), headers={"Content-Type": "image/png"})]
        ),
        host_delay=0,
        max_retries=0,
    )
    first = first_client.download_image(
        image_url,
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_tampered",
    )
    assert first is not None
    Path(first["local_path"]).write_bytes(png_bytes(color=(200, 10, 10)))

    replacement = png_bytes(color=(10, 200, 10))
    replacement_session = FakeSession(
        [FakeResponse(replacement, headers={"Content-Type": "image/png"})]
    )
    second_client = WdcWebClient(
        tmp_path,
        session=replacement_session,
        host_delay=0,
        max_retries=0,
    )
    second = second_client.download_image(
        image_url,
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_tampered",
    )

    assert second is not None
    assert second["downloaded"] is True
    assert second["sha256"] == hashlib.sha256(replacement).hexdigest()
    assert Path(second["local_path"]).read_bytes() == replacement
    assert len(replacement_session.calls) == 1


def test_download_image_rejects_svg_without_rasterizing(tmp_path):
    svg_body = b'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"></svg>'
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    svg_body,
                    url="https://cdn.test/final.svg",
                    headers={"Content-Type": "image/svg+xml"},
                )
            ]
        ),
        host_delay=0,
        max_retries=0,
    )

    record = client.download_image(
        "https://cdn.test/vector.svg",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_svg",
    )

    assert record is None
    assert not list((tmp_path / "wdc_images").glob("*.tmp"))


def test_download_image_rejects_raster_over_pixel_limit_before_load(tmp_path):
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [FakeResponse(png_bytes(100, 100), headers={"Content-Type": "image/png"})]
        ),
        host_delay=0,
        max_retries=0,
        max_image_pixels=5_000,
    )

    assert client.download_image(
        "https://cdn.test/too-many-pixels.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_pixels",
    ) is None


@pytest.mark.parametrize(("width", "height"), [(16, 16), (1000, 32)])
def test_download_image_rejects_tiny_or_extreme_rasters(tmp_path, width, height):
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                FakeResponse(
                    png_bytes(width, height),
                    headers={"Content-Type": "image/png"},
                )
            ]
        ),
        host_delay=0,
        max_retries=0,
    )

    assert client.download_image(
        f"https://cdn.test/{width}x{height}.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_dimensions",
    ) is None
    assert not list((tmp_path / "wdc_images").glob("image_*"))


def test_download_image_retries_with_injected_sleep(tmp_path):
    clock = FakeClock()
    session = FakeSession(
        [
            FakeResponse(b"busy", status_code=503),
            FakeResponse(png_bytes(), headers={"Content-Type": "image/png"}),
        ]
    )
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=1,
        retry_base_seconds=0.5,
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )

    assert client.download_image(
        "https://cdn.test/retry.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_retry",
    ) is not None
    assert len(session.calls) == 2
    assert clock.sleeps == [0.5]


def test_download_image_shares_duplicate_content_from_different_urls(tmp_path):
    body = png_bytes()
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                FakeResponse(body, headers={"Content-Type": "image/png"}),
                FakeResponse(body, headers={"Content-Type": "image/png"}),
            ]
        ),
        host_delay=0,
        max_retries=0,
    )

    first = client.download_image(
        "https://cdn.test/first.png",
        page_url="https://example.test/page",
        source="wdc_image_column",
        entity_id="ent_duplicate",
    )
    duplicate = client.download_image(
        "https://cdn.test/second.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_duplicate",
    )

    assert first is not None
    assert duplicate is not None
    assert first["local_path"] == duplicate["local_path"]
    assert first["sha256"] == duplicate["sha256"]
    assert first["original_url"] == "https://cdn.test/first.png"
    assert duplicate["original_url"] == "https://cdn.test/second.png"
    assert duplicate["final_url"] == "https://example.test/final"
    assert len(list((tmp_path / "wdc_images").glob("image_*"))) == 1
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT original_url, file_name, sha256
            FROM image_cache
            ORDER BY original_url
            """
        ).fetchall()
        failures = connection.execute(
            "SELECT COUNT(*) FROM image_failure_cache"
        ).fetchone()[0]
    assert len(rows) == 2
    assert len({row[1] for row in rows}) == 1
    assert len({row[2] for row in rows}) == 1
    assert failures == 0


def test_download_image_orphan_sha_conflict_is_cleaned_without_raising(tmp_path):
    body = png_bytes()
    first_url = "https://cdn.test/indexed.png"
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [FakeResponse(body, headers={"Content-Type": "image/png"})]
        ),
        host_delay=0,
        max_retries=0,
    )
    assert client.download_image(
        first_url,
        page_url="https://example.test/page",
        source="wdc_image_column",
        entity_id="ent_indexed",
    ) is not None

    orphan_url = "https://cdn.test/orphan.png"
    orphan_path = (
        tmp_path
        / "wdc_images"
        / f"image_{stable_hash(orphan_url, length=24)}.png"
    )
    orphan_path.write_bytes(body)
    offline_session = FakeSession([AssertionError("orphan recovery must not fetch")])
    recovering_client = WdcWebClient(
        tmp_path,
        session=offline_session,
        host_delay=0,
        max_retries=0,
    )

    recovered = recovering_client.download_image(
        orphan_url,
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_orphan",
    )
    assert recovered is not None
    assert recovered["original_url"] == orphan_url
    assert recovered["local_path"] == str(
        tmp_path
        / "wdc_images"
        / client.cached_image_outcome(first_url)["file_name"]
    )
    assert not orphan_path.exists()
    assert offline_session.calls == []


def test_duplicate_content_is_shared_across_policy_namespaces(tmp_path):
    body = png_bytes()
    records = []
    for policy, image_url in (
        ("image-v1", "https://cdn.test/one.png"),
        ("image-v2", "https://cdn.test/two.png"),
    ):
        client = WdcWebClient(
            tmp_path,
            session=FakeSession(
                [
                    FakeResponse(
                        body,
                        headers={"Content-Type": "image/png"},
                    )
                ]
            ),
            host_delay=0,
            max_retries=0,
            network_policy_version=policy,
        )
        records.append(
            client.download_image(
                image_url,
                page_url="https://example.test/page",
                source="wdc_page_image",
                entity_id=policy,
            )
        )

    assert all(record is not None for record in records)
    assert records[0]["local_path"] == records[1]["local_path"]
    assert len(list((tmp_path / "wdc_images").glob("image_*"))) == 1
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_cache"
        ).fetchone() == (2,)


def test_concurrent_different_urls_with_same_content_share_one_file(
    tmp_path,
):
    body = png_bytes()
    barrier = threading.Barrier(2)

    class BarrierResponse(FakeResponse):
        def iter_content(self, chunk_size: int):
            barrier.wait(timeout=5)
            yield from super().iter_content(chunk_size)

    class ConcurrentSession(FakeSession):
        def __init__(self):
            super().__init__([])
            self.lock = threading.Lock()

        def get(self, url: str, **kwargs):
            with self.lock:
                self.calls.append((url, kwargs))
            return BarrierResponse(
                body,
                url=url + "?final=1",
                headers={"Content-Type": "image/png"},
            )

    client = WdcWebClient(
        tmp_path,
        session=ConcurrentSession(),
        host_delay=0,
        max_retries=0,
    )
    start = threading.Barrier(2)

    def download(image_url: str):
        start.wait(timeout=5)
        return client.download_image(
            image_url,
            page_url="https://example.test/page",
            source="wdc_page_image",
            entity_id=image_url,
        )

    urls = [
        "https://cdn.test/concurrent-a.png",
        "https://cdn.test/concurrent-b.png",
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(download, urls))

    assert all(record is not None for record in records)
    assert records[0]["local_path"] == records[1]["local_path"]
    assert len(list((tmp_path / "wdc_images").glob("image_*"))) == 1
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM image_cache"
        ).fetchone() == (2,)


def test_download_image_same_url_singleflight_keeps_file_index_and_records_consistent(
    tmp_path,
    monkeypatch,
):
    first_body = png_bytes(64, 48, color=(200, 10, 10))
    second_body = png_bytes(96, 64, color=(10, 200, 10))
    second_request_started = threading.Event()

    class FirstResponse(FakeResponse):
        def iter_content(self, chunk_size: int):
            second_request_started.wait(timeout=0.5)
            yield self.body

    class DivergentSession(FakeSession):
        def __init__(self):
            super().__init__([])
            self.call_lock = threading.Lock()

        def get(self, url: str, **kwargs):
            with self.call_lock:
                call_index = len(self.calls)
                self.calls.append((url, kwargs))
            if call_index == 0:
                return FirstResponse(
                    first_body,
                    url="https://media.test/first.png",
                    headers={"Content-Type": "image/png"},
                )
            second_request_started.set()
            return FakeResponse(
                second_body,
                url="https://media.test/second.png",
                headers={"Content-Type": "image/png"},
            )

    session = DivergentSession()
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=0,
    )
    original_store = client._store_image_index
    store_state_lock = threading.Lock()
    first_store_waiting = threading.Event()
    second_store_finished = threading.Event()
    store_calls = 0

    def coordinated_store(**kwargs):
        nonlocal store_calls
        with store_state_lock:
            store_calls += 1
            call_number = store_calls
        if call_number == 1:
            first_store_waiting.set()
            second_store_finished.wait(timeout=0.5)
            return original_store(**kwargs)
        first_store_waiting.wait(timeout=0.5)
        result = original_store(**kwargs)
        second_store_finished.set()
        return result

    monkeypatch.setattr(client, "_store_image_index", coordinated_store)
    start = threading.Barrier(2)

    def download(entity_id: str):
        start.wait(timeout=5)
        return client.download_image(
            "https://cdn.test/concurrent.png",
            page_url="https://example.test/page",
            source="wdc_page_image",
            entity_id=entity_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, ["ent_a", "ent_b"]))

    assert all(record is not None for record in results)
    final_path = Path(results[0]["local_path"])
    final_sha = hashlib.sha256(final_path.read_bytes()).hexdigest()
    with Image.open(final_path) as image:
        final_size = image.size
    for record in results:
        assert record["sha256"] == final_sha
        assert (record["width"], record["height"]) == final_size
        assert record["final_url"] == "https://media.test/first.png"
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        indexed = connection.execute(
            """
            SELECT final_url, sha256, width, height
            FROM image_cache WHERE original_url = ?
            """,
            ("https://cdn.test/concurrent.png",),
        ).fetchone()
    assert indexed == (
        "https://media.test/first.png",
        final_sha,
        final_size[0],
        final_size[1],
    )
    assert len(session.calls) == 1


def test_download_image_different_lock_stripes_can_run_concurrently(tmp_path):
    first_url = "https://cdn.test/parallel-a.png"
    first_stripe = int(stable_hash(first_url, length=8), 16) % 256
    second_url = next(
        candidate
        for index in range(100)
        if (
            candidate := f"https://cdn.test/parallel-{index}.png"
        ) != first_url
        and int(stable_hash(candidate, length=8), 16) % 256 != first_stripe
    )
    overlapped: list[bool] = []
    barrier = threading.Barrier(2, action=lambda: overlapped.append(True))

    class ConcurrentResponse(FakeResponse):
        def iter_content(self, chunk_size: int):
            barrier.wait(timeout=0.5)
            yield self.body

    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [
                ConcurrentResponse(
                    png_bytes(64, 48),
                    headers={"Content-Type": "image/png"},
                ),
                ConcurrentResponse(
                    png_bytes(80, 60, color=(80, 100, 120)),
                    headers={"Content-Type": "image/png"},
                ),
            ]
        ),
        host_delay=0,
        max_retries=0,
    )

    def download(url: str):
        return client.download_image(
            url,
            page_url="https://example.test/page",
            source="wdc_page_image",
            entity_id=f"ent_{stable_hash(url, length=8)}",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(download, [first_url, second_url]))

    assert all(record is not None for record in results)
    assert overlapped == [True]


def test_read_wdc_table_removes_image_and_preserves_nested_values(tmp_path):
    path = write_gzip_rows(
        tmp_path,
        [
            {
                "row_id": 7,
                "name": "A",
                "geo": {"lat": "1"},
                "image": ["/a.jpg"],
                "page_url": "https://x.test/p",
            },
            {
                "row_id": 8,
                "name": "B",
                "geo": {"lat": "2"},
                "image": "https://cdn.test/b.jpg",
                "page_url": "https://x.test/q",
            },
        ],
    )

    result = read_wdc_table(path, tmp_path, min_rows=2, min_cols=2)

    assert result.source_table is not None
    assert [column["column_name"] for column in result.source_table["columns"]] == [
        "name",
        "geo",
        "page_url",
    ]
    assert get_cell_text(result.source_table["rows"][0], 1) == '{"lat":"1"}'
    assert result.source_table["metadata"]["candidate_entity_columns"] == [0]
    assert len(result.source_table["metadata"]["column_profiles"]) == 3

    first_entity = result.entities[0]
    first_cell = get_cell(result.source_table["rows"][0], 0)
    assert first_cell["wiki_title"] == first_entity["wiki_title"]
    assert first_entity.keys() == {
        "entity_id",
        "wiki_title",
        "display_texts",
        "context_terms",
        "appears_in",
        "page_url",
        "image_urls",
    }
    assert result.image_urls_by_entity[first_entity["entity_id"]] == ["https://x.test/a.jpg"]
    assert first_entity["image_urls"] == ["https://x.test/a.jpg"]


def test_read_wdc_table_isolates_malformed_rows_and_honors_row_cap(tmp_path):
    path = write_gzip_rows(
        tmp_path,
        [
            "not-json",
            {"row_id": 3, "title": "First", "page_url": "https://x.test/shared"},
            ["not", "an", "object"],
            {
                "row_id": 4,
                "title": "Second",
                "late_column": "kept",
                "page_url": "https://x.test/shared",
            },
            {"row_id": 5, "title": "Not retained", "after_cap": "excluded"},
        ],
    )

    result = read_wdc_table(path, tmp_path, min_rows=2, min_cols=2, max_rows=2)

    assert result.source_table is not None
    assert result.malformed_rows == 2
    assert result.source_table["num_rows"] == 2
    assert [column["column_name"] for column in result.source_table["columns"]] == [
        "title",
        "page_url",
        "late_column",
    ]
    assert result.entities[0]["entity_id"] != result.entities[1]["entity_id"]


def test_read_wdc_table_isolates_invalid_utf8_lines(tmp_path):
    path = tmp_path / "Thing" / "Thing_host.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wb") as handle:
        handle.write(b"\xff\xfe\n")
        handle.write(b'{"name":"valid","page_url":"https://x.test/p"}\n')

    result = read_wdc_table(path, tmp_path, min_rows=1, min_cols=2)

    assert result.source_table is not None
    assert result.malformed_rows == 1


def test_read_wdc_table_rejects_tables_below_thresholds(tmp_path):
    path = write_gzip_rows(tmp_path, [{"name": "Only row", "image": "/only.jpg"}])

    too_few_rows = read_wdc_table(path, tmp_path, min_rows=2, min_cols=1)
    too_few_cols = read_wdc_table(path, tmp_path, min_rows=1, min_cols=2)

    assert too_few_rows.source_table is None
    assert too_few_rows.skip_reason is not None
    assert too_few_cols.source_table is None
    assert too_few_cols.skip_reason is not None


def _read_sharded_records(output_dir: Path, artifact: str) -> list[dict[str, Any]]:
    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    return [
        json.loads(line)
        for shard in manifest["artifacts"][artifact]["shards"]
        for line in (output_dir / shard["path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_build_dataset_small_wdc_end_to_end(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    rows = [
        {
            "row_id": index,
            "name": f"Entity {index}",
            "category": f"Category {index}",
            "detail": f"Detail {index}",
            "region": f"Region {index % 2}",
            "page_url": f"https://pages.test/entity-{index}",
            "image": [f"https://images.test/direct-{index}.png"],
        }
        for index in range(6)
    ]
    write_gzip_rows(input_dir, rows)

    class FakeSharedWebClient:
        def __init__(self) -> None:
            self.fetch_page_calls: list[str] = []
            self.download_image_calls: list[tuple[str, str]] = []

        def fetch_page(self, page_url: str) -> dict[str, Any]:
            self.fetch_page_calls.append(page_url)
            entity_number = page_url.rsplit("-", 1)[-1]
            chunks = [
                f"Entity {entity_number} has useful bridge evidence in chunk {chunk}."
                for chunk in range(3)
            ]
            return {
                "final_url": page_url,
                "text": "\n\n".join(chunks),
                "image_urls": [
                    f"https://images.test/page-{entity_number}-0.png",
                    f"https://images.test/page-{entity_number}-1.png",
                ],
            }

        def download_image(
            self,
            image_url: str,
            *,
            page_url: str,
            source: str,
            entity_id: str,
        ) -> dict[str, Any]:
            self.download_image_calls.append((image_url, source))
            return {
                "asset_id": f"asset_img_{stable_hash(entity_id, source, image_url)}",
                "entity_id": entity_id,
                "asset_type": "image",
                "source": source,
                "image_url": image_url,
                "page_url": page_url,
                "sha256": stable_hash(image_url, length=64),
            }

    class FakeExtractor:
        def extract(
            self,
            asset: dict[str, Any],
            entity: dict[str, Any],
            candidate_attribute_names: list[str],
        ) -> dict[str, Any]:
            entity_number = int(entity["cell_text"].rsplit(" ", 1)[-1])
            attributes = []
            if entity_number < 3:
                attributes.append(
                    {
                        "name": "category",
                        "value": f"Category {entity_number}",
                        "evidence": "visible category value",
                        "connection_evidence": "visible entity name",
                    }
                )
            return {"attributes": attributes, "raw_response": "", "error": ""}

    shared_client = FakeSharedWebClient()
    web_factory_calls: list[dict[str, Any]] = []

    def web_client_factory(**kwargs: Any) -> FakeSharedWebClient:
        web_factory_calls.append(kwargs)
        return shared_client

    args = wdc_builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--max_source_tables",
            "1",
            "--max_images_per_entity",
            "2",
            "--text_asset_chunk_chars",
            "80",
            "--min_text_asset_chunk_chars",
            "1",
            "--max_text_asset_chunks_per_entity",
            "3",
            "--min_recovered_value_ratio",
            "0.6",
            "--min_recovery_denominator",
            "2",
            "--web_workers",
            "3",
            "--records_per_shard",
            "2",
            "--no_model_progress",
        ]
    )

    stats = wdc_builder.build_dataset(
        args,
        web_client_factory=web_client_factory,
        extractor_factory=lambda _args: FakeExtractor(),
    )

    assert len(web_factory_calls) == 1
    assert len(shared_client.fetch_page_calls) == 6
    assert [source for _url, source in shared_client.download_image_calls].count(
        "wdc_image_column"
    ) == 6
    assert [source for _url, source in shared_client.download_image_calls].count(
        "wdc_page_image"
    ) == 6

    sources = _read_sharded_records(output_dir, "source_tables")
    queries = _read_sharded_records(output_dir, "query_tables")
    targets = _read_sharded_records(output_dir, "data_lake_tables")
    recoveries = _read_sharded_records(output_dir, "evidence_recoveries")
    qrels = [
        json.loads(line)
        for line in (output_dir / "qrels.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(sources) == len(queries) == len(targets) == len(qrels) == 1
    assert recoveries
    query = queries[0]
    target = targets[0]
    assert len(query["rows"]) == len(target["rows"]) == 5
    assert query["source_row_indices"] == target["source_row_indices"]
    assert qrels[0]["query_table_id"] == query["table_id"]
    assert qrels[0]["target_table_id"] == target["table_id"]
    assert query["provenance"]["builder"] == "build_wdc_mm_joinability_dataset.py"
    assert target["provenance"]["builder"] == "build_wdc_mm_joinability_dataset.py"
    for table in [*sources, *queries, *targets]:
        assert "image" not in [column["column_name"] for column in table["columns"]]
        assert '"column_name": "image"' not in json.dumps(table)

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    persisted_stats = json.loads((output_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["source_tables"] == stats["query_tables"] == stats["qrels"] == 1
    assert persisted_stats == stats
    assert manifest["source_corpus"] == "WDC Schema.org Table Corpus 2023"
    assert manifest["web_cache"]["database"].endswith("wdc_web.sqlite3")
    assert manifest["artifacts"]["bridge_assets"]["total_records"] == 30


def test_web_failure_callback_is_best_effort_and_structured(tmp_path):
    failures: list[dict[str, Any]] = []

    def failing_callback(record: dict[str, Any]) -> None:
        failures.append(record)
        raise RuntimeError("logging must not abort fetching")

    client = WdcWebClient(
        tmp_path,
        session=FakeSession([FakeResponse(b"missing", status_code=404)]),
        host_delay=0,
        max_retries=0,
        web_failure_callback=failing_callback,
    )

    assert client.fetch_page("https://example.test/missing") is None
    assert failures == [
        {
            "failure_type": "web_fetch_failure",
            "page_url": "https://example.test/missing",
            "status": "terminal",
            "http_status": 404,
            "error": "HTTP 404",
        }
    ]


def test_wdc_file_discovery_rotates_classes_without_path_rglob(tmp_path, monkeypatch):
    for class_name in ("Alpha", "Beta"):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for index in range(3):
            (class_dir / f"{class_name}_{index}.json.gz").touch()

    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Path.rglob may materialize scandir entries on Python 3.10")
        ),
    )
    iterator = wdc_builder.iter_wdc_gzip_paths(tmp_path)

    assert [next(iterator).parent.name for _ in range(4)] == [
        "Alpha",
        "Beta",
        "Alpha",
        "Beta",
    ]


def test_web_host_delay_does_not_block_a_different_host(tmp_path):
    sleeping = threading.Event()
    release_sleep = threading.Event()
    other_host_done = threading.Event()

    def blocking_sleep(_seconds: float) -> None:
        sleeping.set()
        release_sleep.wait(timeout=2)

    client = WdcWebClient(
        tmp_path,
        session=FakeSession([]),
        host_delay=1,
        sleep_fn=blocking_sleep,
        monotonic_fn=lambda: 0.0,
    )
    client._last_request_by_host["slow.test"] = 0.0
    slow_thread = threading.Thread(
        target=client._wait_for_host, args=("https://slow.test/a",)
    )
    other_thread = threading.Thread(
        target=lambda: (
            client._wait_for_host("https://other.test/b"),
            other_host_done.set(),
        )
    )
    slow_thread.start()
    assert sleeping.wait(timeout=1)
    other_thread.start()
    try:
        assert other_host_done.wait(timeout=0.2)
    finally:
        release_sleep.set()
        slow_thread.join(timeout=2)
        other_thread.join(timeout=2)


def test_entity_fairness_lookahead_interleaves_contiguous_hosts():
    entities = [
        {"entity_id": f"a-{index}", "page_url": f"https://a.test/{index}"}
        for index in range(3)
    ] + [
        {"entity_id": f"b-{index}", "page_url": f"https://b.test/{index}"}
        for index in range(3)
    ]

    ordered = list(wdc_builder.iter_fair_entities(entities, lookahead=6))

    assert [urlsplit(item["page_url"]).netloc for item in ordered[:4]] == [
        "a.test",
        "b.test",
        "a.test",
        "b.test",
    ]


def test_entity_fairness_tolerates_malformed_page_url():
    entities = [
        {"entity_id": "bad", "page_url": "http://["},
        {"entity_id": "good", "page_url": "https://good.test/page"},
    ]

    assert list(wdc_builder.iter_fair_entities(entities, lookahead=2)) == entities


def test_dynamic_runner_flags_parse_and_zero_pending_tasks_write_markers(tmp_path):
    start_marker = tmp_path / "runtime" / "start.json"
    ready_marker = tmp_path / "runtime" / "ready.json"
    text_done_marker = tmp_path / "runtime" / "text-done.json"
    image_done_marker = tmp_path / "runtime" / "image-done.json"
    text_endpoints = tmp_path / "runtime" / "text-endpoints.txt"
    image_endpoints = tmp_path / "runtime" / "image-endpoints.txt"
    args = wdc_builder.parse_args(
        [
            "--input_dir",
            str(tmp_path / "empty-input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--text_model_base_url",
            "http://127.0.0.1:8101/v1",
            "--text_model_base_urls_file",
            str(text_endpoints),
            "--text_model_name",
            "fake-text",
            "--image_model_base_url",
            "http://127.0.0.1:8100/v1",
            "--image_model_base_urls_file",
            str(image_endpoints),
            "--image_model_name",
            "fake-image",
            "--precompute_model_cache",
            "--model_start_marker",
            str(start_marker),
            "--model_ready_marker",
            str(ready_marker),
            "--model_text_done_marker",
            str(text_done_marker),
            "--model_image_done_marker",
            str(image_done_marker),
            "--text_model_workers",
            "2",
            "--image_model_workers",
            "3",
            "--no_model_progress",
        ]
    )
    Path(args.input_dir).mkdir()
    extractor_calls: list[bool] = []

    class UnusedClient:
        pass

    stats = wdc_builder.build_dataset(
        args,
        web_client_factory=lambda **_kwargs: UnusedClient(),
        extractor_factory=lambda _args: extractor_calls.append(True),
    )

    assert stats["source_tables"] == 0
    assert extractor_calls == []
    assert not ready_marker.exists()
    assert json.loads(start_marker.read_text(encoding="utf-8"))[
        "text_task_count"
    ] == 0
    assert json.loads(start_marker.read_text(encoding="utf-8"))[
        "image_task_count"
    ] == 0
    assert json.loads(text_done_marker.read_text(encoding="utf-8"))[
        "model_kind"
    ] == "text"
    assert json.loads(image_done_marker.read_text(encoding="utf-8"))[
        "model_kind"
    ] == "image"


def test_media_failure_callback_is_best_effort_and_structured(tmp_path):
    failures: list[dict[str, Any]] = []

    def failing_callback(record: dict[str, Any]) -> None:
        failures.append(record)
        raise RuntimeError("logging must not abort image handling")

    client = WdcWebClient(
        tmp_path,
        session=FakeSession([]),
        host_delay=0,
        media_failure_callback=failing_callback,
    )

    assert client.download_image(
        "not-an-image-url",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_failure",
    ) is None
    assert failures == [
        {
            "failure_type": "media_download_failure",
            "entity_id": "ent_failure",
            "page_url": "https://example.test/page",
            "image_url": "not-an-image-url",
            "source": "wdc_page_image",
            "error": "invalid_image_url",
        }
    ]


def test_cached_terminal_page_failure_is_reported_in_each_run(tmp_path):
    page_url = "https://example.test/permanent-missing"
    first_failures: list[dict[str, Any]] = []
    first = WdcWebClient(
        tmp_path,
        session=FakeSession([FakeResponse(b"missing", status_code=404)]),
        host_delay=0,
        max_retries=0,
        web_failure_callback=first_failures.append,
    )
    assert first.fetch_page(page_url) is None

    cached_failures: list[dict[str, Any]] = []
    second_session = FakeSession([])
    second = WdcWebClient(
        tmp_path,
        session=second_session,
        host_delay=0,
        max_retries=0,
        web_failure_callback=cached_failures.append,
    )

    assert second.fetch_page(page_url) is None
    assert second_session.calls == []
    assert cached_failures == first_failures


def test_bridge_asset_helper_records_swallowed_fetch_and_download_exceptions():
    client = FakeWdcAssetClient(
        page=RuntimeError("page exploded"),
        image_outcomes={
            "https://img.test/direct-1.jpg": RuntimeError("image exploded"),
            "https://img.test/direct-2.jpg": None,
        },
    )
    web_failures: list[dict[str, Any]] = []
    media_failures: list[dict[str, Any]] = []

    wdc_builder.build_wdc_bridge_assets_for_entity(
        wdc_asset_entity(),
        client,
        1,
        web_failure_callback=web_failures.append,
        media_failure_callback=media_failures.append,
    )

    assert web_failures[0]["failure_type"] == "web_fetch_failure"
    assert "page exploded" in web_failures[0]["error"]
    assert media_failures[0]["failure_type"] == "media_download_failure"
    assert media_failures[0]["image_url"] == "https://img.test/direct-1.jpg"
    assert "image exploded" in media_failures[0]["error"]


def test_build_writes_swallowed_client_exceptions_to_failure_jsonl(tmp_path):
    input_dir = tmp_path / "input"
    write_gzip_rows(
        input_dir,
        [
            {
                "row_id": index,
                "name": f"Entity {index}",
                "detail": f"Detail {index}",
                "page_url": f"https://pages.test/{index}",
                "image": f"https://images.test/{index}.png",
            }
            for index in range(2)
        ],
    )
    client = FakeWdcAssetClient(
        page=RuntimeError("page exploded"),
        image_outcomes={
            "https://images.test/0.png": RuntimeError("image zero exploded"),
            "https://images.test/1.png": RuntimeError("image one exploded"),
        },
    )
    output_dir = tmp_path / "output"
    args = wdc_builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--max_source_tables",
            "1",
            "--web_workers",
            "1",
            "--no_model_progress",
        ]
    )

    wdc_builder.build_dataset(
        args,
        web_client_factory=lambda **_kwargs: client,
        extractor_factory=lambda _args: object(),
    )

    web_records = [
        json.loads(line)
        for line in (output_dir / "web_fetch_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    media_records = [
        json.loads(line)
        for line in (output_dir / "media_download_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(web_records) == 2
    assert all("page exploded" in record["error"] for record in web_records)
    assert len(media_records) == 2
    assert all("exploded" in record["error"] for record in media_records)


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://127.0.0.1/private",
        "http://169.254.169.254/latest/meta-data",
        "http://user:password@example.com/secret",
    ],
)
def test_web_client_rejects_unsafe_or_userinfo_urls_before_request(tmp_path, unsafe_url):
    failures: list[dict[str, Any]] = []
    session = FakeSession([])
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        web_failure_callback=failures.append,
    )

    assert client.fetch_page(unsafe_url) is None
    assert session.calls == []
    assert failures and failures[0]["status"] == "terminal"
    assert failures[0]["error"].startswith("unsafe_url:")


def test_web_client_validates_each_redirect_hop_against_ssrf(tmp_path):
    failures: list[dict[str, Any]] = []
    session = FakeSession(
        [
            FakeResponse(
                b"redirect",
                status_code=302,
                url="https://public.test/start",
                headers={"Location": "http://127.0.0.1/internal"},
            )
        ]
    )
    client = WdcWebClient(
        tmp_path,
        session=session,
        host_delay=0,
        max_retries=0,
        resolve_host_fn=lambda _host: ["93.184.216.34"],
        web_failure_callback=failures.append,
    )

    assert client.fetch_page("https://public.test/start") is None
    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False
    assert failures[0]["error"].startswith("unsafe_redirect:")


def test_production_transport_pins_first_validated_dns_answer(tmp_path):
    resolver_calls: list[str] = []
    pinned_calls: list[tuple[str, str]] = []

    def rebinding_resolver(host: str) -> list[str]:
        resolver_calls.append(host)
        return ["93.184.216.34"] if len(resolver_calls) == 1 else ["127.0.0.1"]

    def pinned_request(url: str, *, pinned_ip: str, server_hostname: str, **_kwargs):
        pinned_calls.append((pinned_ip, server_hostname))
        return FakeResponse(b"<p>Pinned public response.</p>", url=url)

    client = WdcWebClient(
        tmp_path,
        host_delay=0,
        max_retries=0,
        resolve_host_fn=rebinding_resolver,
        pinned_request_fn=pinned_request,
    )

    assert client.fetch_page("https://rebind.test/page") is not None
    assert resolver_calls == ["rebind.test"]
    assert pinned_calls == [("93.184.216.34", "rebind.test")]


def test_total_image_byte_quota_stops_cache_growth(tmp_path):
    body = png_bytes()
    failures: list[dict[str, Any]] = []
    client = WdcWebClient(
        tmp_path,
        session=FakeSession(
            [FakeResponse(body, headers={"Content-Type": "image/png"})]
        ),
        host_delay=0,
        max_retries=0,
        max_total_image_bytes=len(body) - 1,
        media_failure_callback=failures.append,
    )

    assert client.download_image(
        "https://cdn.test/quota.png",
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_quota",
    ) is None
    assert failures[-1]["error"] == "download_or_validation_failed"
    assert not list((tmp_path / "wdc_images").glob("image_*"))


def test_unbounded_input_requires_explicit_opt_in(tmp_path):
    base = ["--input_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]

    assert wdc_builder.parse_args(base).max_rows_per_source_table == 100
    with pytest.raises(SystemExit):
        wdc_builder.parse_args([*base, "--max_rows_per_source_table", "0"])
    allowed = wdc_builder.parse_args(
        [*base, "--max_rows_per_source_table", "0", "--allow_unbounded"]
    )
    assert allowed.allow_unbounded is True


def test_legacy_image_cache_bytes_are_backfilled_from_actual_files(tmp_path):
    image_dir = tmp_path / "wdc_images"
    image_dir.mkdir()
    body = png_bytes()
    image_path = image_dir / "image_legacy.png"
    image_path.write_bytes(body)
    database_path = tmp_path / "wdc_web.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE image_cache (
                original_url TEXT PRIMARY KEY, final_url TEXT NOT NULL,
                file_name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
                width INTEGER NOT NULL, height INTEGER NOT NULL,
                mime_type TEXT NOT NULL, updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO image_cache VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "https://cdn.test/legacy.png",
                "https://cdn.test/legacy.png",
                image_path.name,
                hashlib.sha256(body).hexdigest(),
                64,
                48,
                "image/png",
                0.0,
            ),
        )

    client = WdcWebClient(tmp_path, session=FakeSession([]), host_delay=0)

    with sqlite3.connect(database_path) as connection:
        indexed_bytes = connection.execute(
            "SELECT bytes FROM image_cache WHERE original_url = ?",
            ("https://cdn.test/legacy.png",),
        ).fetchone()[0]
    assert indexed_bytes == len(body)
    assert client._image_bytes_total == len(body)


def test_orphan_image_recovery_counts_bytes_against_total_quota(tmp_path):
    image_url = "https://cdn.test/orphan-counted.png"
    image_dir = tmp_path / "wdc_images"
    image_dir.mkdir()
    body = png_bytes()
    orphan = image_dir / f"image_{stable_hash(image_url, length=24)}.png"
    orphan.write_bytes(body)
    client = WdcWebClient(
        tmp_path,
        session=FakeSession([]),
        host_delay=0,
        max_total_image_bytes=len(body),
    )

    assert client.download_image(
        image_url,
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_orphan_counted",
    ) is not None
    assert client._image_bytes_total == len(body)


def test_page_body_obeys_total_web_cache_byte_cap(tmp_path):
    client = WdcWebClient(
        tmp_path,
        session=FakeSession([FakeResponse(b"<p>body exceeds quota</p>")]),
        host_delay=0,
        max_retries=0,
        max_total_cache_bytes=5,
    )

    assert client.fetch_page("https://example.test/quota") is None
    with sqlite3.connect(tmp_path / "wdc_web.sqlite3") as connection:
        assert connection.execute(
            "SELECT status FROM page_cache WHERE page_url = ?",
            ("https://example.test/quota",),
        ).fetchone() != ("success",)


def test_page_stream_obeys_total_response_deadline(tmp_path):
    clock = FakeClock()

    class DripResponse(FakeResponse):
        def iter_content(self, chunk_size: int):
            for chunk in (b"<p>", b"slow</p>"):
                clock.now += 0.4
                yield chunk

    client = WdcWebClient(
        tmp_path,
        session=FakeSession([DripResponse(b"")]),
        host_delay=0,
        max_retries=0,
        max_response_seconds=0.5,
        monotonic_fn=clock.monotonic,
    )

    assert client.fetch_page("https://example.test/drip") is None


def test_safe_defaults_and_scan_cap_bound_rejected_file_walk(tmp_path):
    input_dir = tmp_path / "input"
    for index in range(5):
        write_gzip_rows(
            input_dir,
            [{"name": f"Only {index}", "page_url": f"https://x.test/{index}"}],
            name=f"Thing_{index}.json.gz",
        )
    args = wdc_builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(tmp_path / "output"),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--max_scanned_files",
            "2",
            "--no_model_progress",
        ]
    )
    assert args.max_source_tables == 100
    assert args.max_rows_per_source_table == 100
    assert args.max_images_per_entity == 2

    stats = wdc_builder.build_dataset(
        args,
        web_client_factory=lambda **_kwargs: FakeWdcAssetClient(
            page=None, image_outcomes={}
        ),
        extractor_factory=lambda _args: object(),
    )

    assert stats["scanned_files"] == 2
    assert stats["processed_tables"] == 2
    assert stats["scanned_file_cap_reached"] is True


def test_stats_and_logs_show_row_truncation_safety_config(tmp_path, caplog):
    input_dir = tmp_path / "input"
    write_gzip_rows(
        input_dir,
        [
            {
                "name": f"Entity {index}",
                "detail": f"Detail {index}",
                "page_url": f"https://x.test/{index}",
            }
            for index in range(3)
        ],
    )
    args = wdc_builder.parse_args(
        [
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(tmp_path / "output"),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--max_source_tables",
            "1",
            "--max_rows_per_source_table",
            "2",
            "--max_scanned_files",
            "1",
            "--no_model_progress",
        ]
    )

    with caplog.at_level("INFO"):
        stats = wdc_builder.build_dataset(
            args,
            web_client_factory=lambda **_kwargs: FakeWdcAssetClient(
                page=None, image_outcomes={}
            ),
            extractor_factory=lambda _args: object(),
        )

    assert stats["row_capped_source_tables"] == 1
    assert stats["safety_config"]["max_rows_per_source_table"] == 2
    assert "WDC safety limits" in caplog.text

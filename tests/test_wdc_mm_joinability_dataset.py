import gzip
import hashlib
import io
import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
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


def test_download_image_rasterizes_svg_with_repository_helper(tmp_path, monkeypatch):
    svg_body = b'<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"></svg>'

    def fake_rasterize(svg_path, png_path, imageinfo):
        assert svg_path.read_bytes() == svg_body
        assert imageinfo["mime"] == "image/svg+xml"
        png_path.write_bytes(png_bytes(80, 60))
        return True, "fake"

    monkeypatch.setattr(wdc_builder, "rasterize_svg_to_png", fake_rasterize)
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

    assert record is not None
    assert record["mime_type"] == "image/png"
    assert record["converted_from"] == "image/svg+xml"
    assert record["width"] == 80
    assert record["height"] == 60
    assert not list((tmp_path / "wdc_images").glob("*.tmp"))


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


def test_download_image_rejects_duplicate_content_from_different_urls(tmp_path):
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
    assert duplicate is None
    assert len(list((tmp_path / "wdc_images").glob("image_*"))) == 1


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

    assert recovering_client.download_image(
        orphan_url,
        page_url="https://example.test/page",
        source="wdc_page_image",
        entity_id="ent_orphan",
    ) is None
    assert not orphan_path.exists()
    assert offline_session.calls == []


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

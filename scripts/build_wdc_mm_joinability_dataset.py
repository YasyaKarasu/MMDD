#!/usr/bin/env python
"""Adapt WDC Schema.org gzip JSONL host tables to the internal table schema."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from PIL import Image

from build_mm_table_dataset import (
    rasterize_svg_to_png,
    select_relevant_text_chunks,
    split_text_asset_content,
)
from stage1_io import clean_text, column_profiles, is_numeric_text, stable_hash

try:
    import requests
except ImportError:  # pragma: no cover - requests is a declared runtime dependency.
    requests = None  # type: ignore[assignment]


ENTITY_COLUMN_PRIORITY = ("name", "headline", "title", "identifier", "page_url")
EXCLUDED_COLUMNS = {"row_id", "image"}


@dataclass
class WdcTableResult:
    source_table: dict[str, Any] | None
    entities: list[dict[str, Any]]
    image_urls_by_entity: dict[str, list[str]]
    skip_reason: str | None
    malformed_rows: int


def _looks_like_relative_url(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../"))
        or "/" in value
        or "." in value
    )


def extract_image_urls(value: Any, base_url: str = "") -> list[str]:
    """Return first-seen HTTP(S) URLs found recursively in an image value."""
    urls: list[str] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if not isinstance(item, str):
            return

        candidate = clean_text(item)
        if not candidate or any(char.isspace() for char in candidate):
            return
        try:
            parsed_candidate = urlsplit(candidate)
        except ValueError:
            return
        if not parsed_candidate.scheme and not _looks_like_relative_url(candidate):
            return
        try:
            resolved = urljoin(base_url, candidate)
            parsed = urlsplit(resolved)
        except ValueError:
            return
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return
        if resolved not in seen:
            seen.add(resolved)
            urls.append(resolved)

    visit(value)
    return urls


class _AssetHTMLParser(HTMLParser):
    _SKIP_TAGS = {
        "script",
        "style",
        "template",
        "svg",
        "nav",
        "footer",
        "form",
        "noscript",
    }
    _TEXT_TAGS = {"title", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    _IMAGE_ATTRIBUTES = ("src", "data-src", "data-lazy-src", "data-original")
    _DESCRIPTION_METADATA = {
        "description",
        "og:description",
        "twitter:description",
    }
    _IMAGE_METADATA = {"og:image", "og:image:url", "twitter:image", "twitter:image:src"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.text_depth = 0
        self.text_parts: list[str] = []
        self.metadata_text: list[str] = []
        self.image_candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value or "" for name, value in attrs}
        if tag in self._SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self._TEXT_TAGS:
            self.text_depth += 1
        if tag == "meta":
            metadata_name = clean_text(
                attributes.get("property") or attributes.get("name")
            ).casefold()
            content = clean_text(attributes.get("content"))
            if content and metadata_name in self._DESCRIPTION_METADATA:
                self.metadata_text.append(content)
            if content and metadata_name in self._IMAGE_METADATA:
                self.image_candidates.append(content)
        elif tag == "img":
            for attribute in self._IMAGE_ATTRIBUTES:
                candidate = clean_text(attributes.get(attribute))
                if candidate:
                    self.image_candidates.append(candidate)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if not self.skip_depth and tag in self._TEXT_TAGS and self.text_depth:
            self.text_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip_depth and self.text_depth:
            text = clean_text(data)
            if text:
                self.text_parts.append(text)


def extract_html_assets(html_text: str, base_url: str) -> tuple[str, list[str]]:
    """Extract useful visible text and first-seen HTTP(S) image URLs from HTML."""
    parser = _AssetHTMLParser()
    parser.feed(html_text)
    parser.close()
    text = "\n".join([*parser.metadata_text, *parser.text_parts])
    images = extract_image_urls(parser.image_candidates, base_url)
    return text, images


class WdcWebClient:
    """Bounded generic HTTP client backed by a durable SQLite page cache."""

    _RETRYABLE_HTTP_STATUSES = {408, 425, 429}
    _IMAGE_LOCK_STRIPES = 256

    def __init__(
        self,
        cache_dir: Path,
        *,
        session: Any | None = None,
        user_agent: str = "MMDD-WDC-DatasetBuilder/0.1 (research dataset construction)",
        connect_timeout: float = 10.0,
        read_timeout: float = 30.0,
        max_retries: int = 2,
        retry_base_seconds: float = 0.25,
        max_page_bytes: int = 2_000_000,
        max_image_bytes: int = 10_000_000,
        min_image_side: int = 32,
        max_image_aspect_ratio: float = 20.0,
        host_delay: float = 0.5,
        sleep_fn: Any = time.sleep,
        monotonic_fn: Any = time.monotonic,
    ) -> None:
        if session is None and requests is None:
            raise RuntimeError("requests is required for WDC web asset fetching")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.cache_dir / "wdc_web.sqlite3"
        self.image_dir = self.cache_dir / "wdc_images"
        self.session = session if session is not None else requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.timeout = (max(0.01, connect_timeout), max(0.01, read_timeout))
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.max_page_bytes = max(1, max_page_bytes)
        self.max_image_bytes = max(1, max_image_bytes)
        self.min_image_side = max(1, min_image_side)
        self.max_image_aspect_ratio = max(1.0, max_image_aspect_ratio)
        self.host_delay = max(0.0, host_delay)
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self._host_lock = threading.Lock()
        self._last_request_by_host: dict[str, float] = {}
        self._image_locks = tuple(
            threading.Lock() for _ in range(self._IMAGE_LOCK_STRIPES)
        )
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS page_cache (
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
                CREATE TABLE IF NOT EXISTS image_cache (
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

    def _wait_for_host(self, url: str) -> None:
        host = urlsplit(url).netloc.casefold()
        with self._host_lock:
            now = float(self.monotonic_fn())
            last_request = self._last_request_by_host.get(host)
            if last_request is not None:
                wait_seconds = self.host_delay - (now - last_request)
                if wait_seconds > 0:
                    self.sleep_fn(wait_seconds)
                    now = float(self.monotonic_fn())
            self._last_request_by_host[host] = now

    def _cached_page(self, page_url: str) -> tuple[str | None, dict[str, Any] | None]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM page_cache WHERE page_url = ?",
                (page_url,),
            ).fetchone()
        if row is None:
            return None, None
        if row["status"] != "success":
            return str(row["status"]), None
        try:
            image_urls = json.loads(row["image_urls_json"] or "[]")
        except json.JSONDecodeError:
            return None, None
        if not isinstance(image_urls, list):
            return None, None
        return "success", {
                "page_url": page_url,
                "final_url": row["final_url"] or page_url,
                "text": row["text"] or "",
                "image_urls": image_urls,
            }

    def _store_page(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO page_cache (
                    page_url, status, final_url, text, image_urls_json,
                    http_status, error, updated_at
                ) VALUES (?, 'success', ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(page_url) DO UPDATE SET
                    status = excluded.status,
                    final_url = excluded.final_url,
                    text = excluded.text,
                    image_urls_json = excluded.image_urls_json,
                    http_status = excluded.http_status,
                    error = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["page_url"],
                    payload["final_url"],
                    payload["text"],
                    json.dumps(payload["image_urls"], ensure_ascii=False),
                    200,
                    time.time(),
                ),
            )

    def _store_page_failure(
        self,
        page_url: str,
        *,
        status: str,
        http_status: int | None,
        error: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO page_cache (
                    page_url, status, final_url, text, image_urls_json,
                    http_status, error, updated_at
                ) VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?)
                ON CONFLICT(page_url) DO UPDATE SET
                    status = excluded.status,
                    final_url = NULL,
                    text = NULL,
                    image_urls_json = NULL,
                    http_status = excluded.http_status,
                    error = excluded.error,
                    updated_at = excluded.updated_at
                WHERE page_cache.status != 'success'
                """,
                (page_url, status, http_status, error[:1000], time.time()),
            )

    def fetch_page(self, page_url: str) -> dict[str, Any] | None:
        """Fetch, extract, and cache one page without retaining response HTML."""
        page_url = clean_text(page_url)
        if not extract_image_urls(page_url):
            return None
        cache_status, cached = self._cached_page(page_url)
        if cache_status == "success":
            return cached
        if cache_status == "terminal":
            return None

        last_error = "request failed"
        last_http_status: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._wait_for_host(page_url)
                with self.session.get(page_url, stream=True, timeout=self.timeout) as response:
                    http_status = int(response.status_code)
                    if http_status >= 400:
                        retryable = (
                            http_status in self._RETRYABLE_HTTP_STATUSES
                            or http_status >= 500
                        )
                        if retryable and attempt < self.max_retries:
                            self.sleep_fn(self.retry_base_seconds * (2**attempt))
                            continue
                        self._store_page_failure(
                            page_url,
                            status="retryable" if retryable else "terminal",
                            http_status=http_status,
                            error=f"HTTP {http_status}",
                        )
                        return None
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        if len(body) + len(chunk) > self.max_page_bytes:
                            self._store_page_failure(
                                page_url,
                                status="terminal",
                                http_status=http_status,
                                error="page response exceeded byte limit",
                            )
                            return None
                        body.extend(chunk)
                    encoding = getattr(response, "encoding", None) or "utf-8"
                    html_text = bytes(body).decode(encoding, errors="replace")
                    final_url = clean_text(getattr(response, "url", None)) or page_url
                    text, image_urls = extract_html_assets(html_text, final_url)
                    payload = {
                        "page_url": page_url,
                        "final_url": final_url,
                        "text": text,
                        "image_urls": image_urls,
                    }
                    self._store_page(payload)
                    return payload
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_http_status = None
                if attempt < self.max_retries:
                    self.sleep_fn(self.retry_base_seconds * (2**attempt))
                    continue
                self._store_page_failure(
                    page_url,
                    status="retryable",
                    http_status=last_http_status,
                    error=last_error,
                )
                return None

        self._store_page_failure(
            page_url,
            status="retryable",
            http_status=last_http_status,
            error=last_error,
        )
        return None

    def download_image(
        self,
        image_url: str,
        *,
        page_url: str,
        source: str,
        entity_id: str,
    ) -> dict[str, Any] | None:
        """Download one image candidate within the configured byte limit."""
        image_url = clean_text(image_url)
        if not extract_image_urls(image_url):
            return None
        lock_index = int(stable_hash(image_url, length=8), 16) % len(self._image_locks)
        with self._image_locks[lock_index]:
            return self._download_image_locked(
                image_url,
                page_url=page_url,
                source=source,
                entity_id=entity_id,
            )

    def _download_image_locked(
        self,
        image_url: str,
        *,
        page_url: str,
        source: str,
        entity_id: str,
    ) -> dict[str, Any] | None:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        image_key = stable_hash(image_url, length=24)
        attempt_key = uuid.uuid4().hex
        temporary_path = self.image_dir / f".{image_key}.{attempt_key}.download.tmp"
        converted_path = self.image_dir / f".{image_key}.{attempt_key}.raster.tmp.png"

        with self._connect() as connection:
            cached_row = connection.execute(
                "SELECT * FROM image_cache WHERE original_url = ?",
                (image_url,),
            ).fetchone()
        if cached_row is not None:
            cached_path = self.image_dir / cached_row["file_name"]
            raster = self._validated_raster(cached_path)
            sha_matches = (
                raster is not None
                and self._sha256_path(cached_path) == cached_row["sha256"]
            )
            if raster is not None and sha_matches:
                width, height, mime_type, _extension = raster
                return self._image_record(
                    cached_path,
                    original_url=image_url,
                    final_url=cached_row["final_url"],
                    page_url=page_url,
                    source=source,
                    entity_id=entity_id,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                    downloaded=False,
                    converted_from=(
                        "image/svg+xml"
                        if urlsplit(image_url).path.casefold().endswith(".svg")
                        else None
                    ),
                )
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM image_cache WHERE original_url = ?",
                    (image_url,),
                )
            try:
                cached_path.unlink(missing_ok=True)
            except OSError:
                pass

        for cached_path in sorted(self.image_dir.glob(f"image_{image_key}.*")):
            raster = self._validated_raster(cached_path)
            if raster is None:
                try:
                    cached_path.unlink()
                except OSError:
                    pass
                continue
            width, height, mime_type, _extension = raster
            try:
                self._store_image_index(
                    original_url=image_url,
                    final_url=image_url,
                    path=cached_path,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                )
            except sqlite3.IntegrityError:
                try:
                    cached_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None
            return self._image_record(
                cached_path,
                original_url=image_url,
                final_url=image_url,
                page_url=page_url,
                source=source,
                entity_id=entity_id,
                width=width,
                height=height,
                mime_type=mime_type,
                downloaded=False,
                converted_from=(
                    "image/svg+xml"
                    if urlsplit(image_url).path.casefold().endswith(".svg")
                    else None
                ),
            )

        for attempt in range(self.max_retries + 1):
            try:
                temporary_path.unlink(missing_ok=True)
                converted_path.unlink(missing_ok=True)
                self._wait_for_host(image_url)
                with self.session.get(image_url, stream=True, timeout=self.timeout) as response:
                    http_status = int(response.status_code)
                    if http_status >= 400:
                        retryable = (
                            http_status in self._RETRYABLE_HTTP_STATUSES
                            or http_status >= 500
                        )
                        if retryable and attempt < self.max_retries:
                            self.sleep_fn(self.retry_base_seconds * (2**attempt))
                            continue
                        return None
                    content_type = clean_text(response.headers.get("Content-Type")).casefold()
                    if content_type and not content_type.startswith("image/"):
                        return None
                    total_bytes = 0
                    with temporary_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            total_bytes += len(chunk)
                            if total_bytes > self.max_image_bytes:
                                return None
                            handle.write(chunk)
                    final_url = clean_text(getattr(response, "url", None)) or image_url

                source_mime_type = content_type.split(";", 1)[0]
                source_is_svg = (
                    source_mime_type == "image/svg+xml"
                    or urlsplit(final_url).path.casefold().endswith(".svg")
                )
                raster_path = temporary_path
                if source_is_svg:
                    converted, _reason = rasterize_svg_to_png(
                        temporary_path,
                        converted_path,
                        {"mime": "image/svg+xml", "url": final_url},
                    )
                    if not converted:
                        return None
                    raster_path = converted_path

                raster = self._validated_raster(raster_path)
                if raster is None:
                    return None
                width, height, mime_type, extension = raster
                digest = self._sha256_path(raster_path)
                with self._connect() as connection:
                    duplicate = connection.execute(
                        "SELECT original_url FROM image_cache WHERE sha256 = ?",
                        (digest,),
                    ).fetchone()
                if duplicate is not None and duplicate["original_url"] != image_url:
                    return None
                image_path = self.image_dir / f"image_{image_key}{extension}"
                raster_path.replace(image_path)
                try:
                    self._store_image_index(
                        original_url=image_url,
                        final_url=final_url,
                        path=image_path,
                        width=width,
                        height=height,
                        mime_type=mime_type,
                    )
                except sqlite3.IntegrityError:
                    image_path.unlink(missing_ok=True)
                    return None
                return self._image_record(
                    image_path,
                    original_url=image_url,
                    final_url=final_url,
                    page_url=page_url,
                    source=source,
                    entity_id=entity_id,
                    width=width,
                    height=height,
                    mime_type=mime_type,
                    downloaded=True,
                    converted_from="image/svg+xml" if source_is_svg else None,
                )
            except Exception:
                if attempt < self.max_retries:
                    self.sleep_fn(self.retry_base_seconds * (2**attempt))
                    continue
                return None
            finally:
                for cleanup_path in (temporary_path, converted_path):
                    try:
                        cleanup_path.unlink(missing_ok=True)
                    except OSError:
                        pass
        return None

    def _validated_raster(
        self,
        path: Path,
    ) -> tuple[int, int, str, str] | None:
        try:
            with Image.open(path) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
            with Image.open(path) as image:
                image.load()
        except Exception:
            return None
        if min(width, height) < self.min_image_side:
            return None
        if max(width, height) / min(width, height) > self.max_image_aspect_ratio:
            return None
        extensions = {
            "BMP": ".bmp",
            "GIF": ".gif",
            "JPEG": ".jpg",
            "PNG": ".png",
            "TIFF": ".tiff",
            "WEBP": ".webp",
        }
        extension = extensions.get(image_format)
        mime_type = Image.MIME.get(image_format)
        if extension is None or mime_type is None:
            return None
        return width, height, mime_type, extension

    @staticmethod
    def _sha256_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _store_image_index(
        self,
        *,
        original_url: str,
        final_url: str,
        path: Path,
        width: int,
        height: int,
        mime_type: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO image_cache (
                    original_url, final_url, file_name, sha256,
                    width, height, mime_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(original_url) DO UPDATE SET
                    final_url = excluded.final_url,
                    file_name = excluded.file_name,
                    sha256 = excluded.sha256,
                    width = excluded.width,
                    height = excluded.height,
                    mime_type = excluded.mime_type,
                    updated_at = excluded.updated_at
                """,
                (
                    original_url,
                    final_url,
                    path.name,
                    self._sha256_path(path),
                    width,
                    height,
                    mime_type,
                    time.time(),
                ),
            )

    def _image_record(
        self,
        path: Path,
        *,
        original_url: str,
        final_url: str,
        page_url: str,
        source: str,
        entity_id: str,
        width: int,
        height: int,
        mime_type: str,
        downloaded: bool,
        converted_from: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "asset_id": f"asset_img_{stable_hash(entity_id, source, original_url, length=20)}",
            "entity_id": entity_id,
            "asset_type": "image",
            "source": source,
            "image_url": original_url,
            "original_url": original_url,
            "final_url": final_url,
            "page_url": page_url,
            "local_path": str(path),
            "relative_path": path.relative_to(self.cache_dir).as_posix(),
            "file_name": path.name,
            "bytes": path.stat().st_size,
            "sha256": self._sha256_path(path),
            "width": width,
            "height": height,
            "mime_type": mime_type,
            "downloaded": downloaded,
        }
        if converted_from:
            record["converted_from"] = converted_from
        return record


def build_wdc_bridge_assets_for_entity(
    entity: dict[str, Any],
    client: WdcWebClient,
    max_images_per_entity: int,
    text_asset_chunk_chars: int = 800,
    min_text_asset_chunk_chars: int = 120,
    max_text_asset_chunks_per_entity: int = 3,
) -> list[dict[str, Any]]:
    """Build webpage text and direct-first image assets for one WDC entity."""
    page_url = clean_text(entity.get("page_url"))
    try:
        fetched_page = client.fetch_page(page_url)
    except Exception:
        fetched_page = None
    page = fetched_page if isinstance(fetched_page, dict) else None
    records: list[dict[str, Any]] = []

    if page:
        text_chunks = split_text_asset_content(
            page.get("text"),
            max_chars=text_asset_chunk_chars,
            min_chars=min_text_asset_chunk_chars,
            max_chunks=0,
        )
        selected_text_chunks = select_relevant_text_chunks(
            text_chunks,
            entity,
            max_text_asset_chunks_per_entity,
        )
        source_asset_id = f"asset_text_{stable_hash(entity['entity_id'], 'wdc_page_text')}"
        for chunk_index, chunk, chunk_score in selected_text_chunks:
            records.append(
                {
                    "asset_id": f"{source_asset_id}_{chunk_index:03d}",
                    "source_asset_id": source_asset_id,
                    "entity_id": entity["entity_id"],
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "text",
                    "content": chunk,
                    "text_chunk_index": chunk_index,
                    "text_chunk_count": len(text_chunks),
                    "selected_text_chunk_count": len(selected_text_chunks),
                    "text_chunk_relevance_score": round(chunk_score, 6),
                    "source": "wdc_page_text_chunk",
                    "url": clean_text(page.get("final_url")) or page_url,
                }
            )

    image_quota = max(0, int(max_images_per_entity))
    kept = 0
    seen_image_urls: set[str] = set()
    seen_image_hashes: set[str] = set()
    image_sources = [
        (entity.get("image_urls") or [], "wdc_image_column"),
        ((page or {}).get("image_urls") or [], "wdc_page_image"),
    ]
    for image_urls, source in image_sources:
        for image_url in image_urls:
            if kept >= image_quota:
                break
            normalized_url = clean_text(image_url)
            if not normalized_url or normalized_url in seen_image_urls:
                continue
            seen_image_urls.add(normalized_url)
            try:
                image_record = client.download_image(
                    normalized_url,
                    page_url=page_url,
                    source=source,
                    entity_id=entity["entity_id"],
                )
            except Exception:
                continue
            if not isinstance(image_record, dict):
                continue
            image_hash = clean_text(image_record.get("sha256"))
            if image_hash and image_hash in seen_image_hashes:
                continue
            if image_hash:
                seen_image_hashes.add(image_hash)
            records.append(image_record)
            kept += 1
    return records


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return clean_text(value)


def _relative_source(path: Path, input_root: Path) -> str:
    try:
        relative = path.relative_to(input_root)
    except ValueError:
        relative = Path(path.name)
    return relative.as_posix()


def _schema_class(relative_source: str) -> str:
    parts = Path(relative_source).parts
    if len(parts) > 1:
        return parts[0]
    filename = parts[0] if parts else relative_source
    return filename.split("_", 1)[0].removesuffix(".json.gz")


def _read_rows(path: Path, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_rows = 0
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed_rows += 1
                continue
            if not isinstance(row, dict):
                malformed_rows += 1
                continue
            rows.append(row)
    return rows, malformed_rows


def _first_seen_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name in EXCLUDED_COLUMNS or name in seen:
                continue
            seen.add(name)
            columns.append(name)
    return columns


def _entity_column(columns: list[str]) -> int | None:
    for name in ENTITY_COLUMN_PRIORITY:
        if name in columns:
            return columns.index(name)
    return None


def _source_row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def _context_terms(cells: list[dict[str, Any]], entity_column: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        if cell["column_index"] == entity_column:
            continue
        text = clean_text(cell.get("text"))
        name = clean_text(cell.get("column_name"))
        for term in (text if text and not is_numeric_text(text) else "", name):
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _empty_result(reason: str, malformed_rows: int) -> WdcTableResult:
    return WdcTableResult(
        source_table=None,
        entities=[],
        image_urls_by_entity={},
        skip_reason=reason,
        malformed_rows=malformed_rows,
    )


def read_wdc_table(
    path: Path,
    input_root: Path,
    min_rows: int,
    min_cols: int,
    max_rows: int = 0,
) -> WdcTableResult:
    """Read one WDC gzip host table and adapt it to the internal table contract."""
    raw_rows, malformed_rows = _read_rows(path, max_rows)
    if len(raw_rows) < min_rows:
        return _empty_result("too_few_rows", malformed_rows)

    column_names = _first_seen_columns(raw_rows)
    if len(column_names) < min_cols:
        return _empty_result("too_few_columns", malformed_rows)

    entity_column = _entity_column(column_names)
    if entity_column is None:
        return _empty_result("missing_entity_column", malformed_rows)

    relative_source = _relative_source(path, input_root)
    schema_class = _schema_class(relative_source)
    source_table_id = f"st_wdc_{stable_hash(schema_class, relative_source, length=16)}"
    columns = [
        {"column_index": index, "column_name": name, "is_numeric_column": False}
        for index, name in enumerate(column_names)
    ]
    rows: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    image_urls_by_entity: dict[str, list[str]] = {}

    for fallback, raw_row in enumerate(raw_rows):
        source_row_id = _source_row_id(raw_row, fallback)
        display_text = _cell_text(raw_row.get(column_names[entity_column]))
        page_url = clean_text(raw_row.get("page_url"))
        entity_key = f"wdc_{stable_hash(schema_class, relative_source, source_row_id, page_url, display_text, length=20)}"
        entity_id = f"ent_{stable_hash(entity_key, length=16)}"
        image_urls = extract_image_urls(raw_row.get("image"), page_url)

        cells: list[dict[str, Any]] = []
        for column_index, column_name in enumerate(column_names):
            raw_value = raw_row.get(column_name)
            is_entity = column_index == entity_column
            cells.append(
                {
                    "column_index": column_index,
                    "column_name": column_name,
                    "raw": raw_value,
                    "text": _cell_text(raw_value),
                    "wiki_title": entity_key if is_entity else None,
                    "has_wiki_link": is_entity,
                }
            )

        appears_in = {
            "source_table_id": source_table_id,
            "query_view_id": None,
            "row_id": source_row_id,
            "column_index": entity_column,
            "column_name": column_names[entity_column],
        }
        entities.append(
            {
                "entity_id": entity_id,
                "wiki_title": entity_key,
                "display_texts": [display_text] if display_text else [],
                "context_terms": _context_terms(cells, entity_column),
                "appears_in": [appears_in],
                "page_url": page_url,
                "image_urls": image_urls,
            }
        )
        image_urls_by_entity[entity_id] = image_urls
        rows.append({"row_id": source_row_id, "cells": cells})

    source_table: dict[str, Any] = {
        "source_table_id": source_table_id,
        "source_file": relative_source,
        "page_title": schema_class,
        "caption": "",
        "section_title": "",
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": columns,
        "rows": rows,
        "metadata": {"candidate_entity_columns": [entity_column], "column_profiles": []},
    }
    profiles = column_profiles(source_table)
    source_table["metadata"]["column_profiles"] = [
        {
            "column_index": column["column_index"],
            "column_name": column["column_name"],
            **profiles[column["column_index"]],
        }
        for column in columns
    ]
    for column in columns:
        profile = profiles[column["column_index"]]
        column["is_numeric_column"] = float(profile.get("numeric_ratio", 0.0)) >= 0.8

    return WdcTableResult(
        source_table=source_table,
        entities=entities,
        image_urls_by_entity=image_urls_by_entity,
        skip_reason=None,
        malformed_rows=malformed_rows,
    )

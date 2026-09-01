#!/usr/bin/env python
"""Adapt WDC Schema.org gzip JSONL host tables to the internal table schema."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import ipaddress
import json
import logging
import os
import re
import shutil
import socket
import ssl
import sqlite3
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urljoin, urlsplit

import dns.exception
import dns.resolver
from PIL import Image

import build_mm_joinability_dataset as join_builder
from build_mm_table_dataset import (
    ShardedJsonlWriter,
    iter_jsonl_records,
    normalize_title,
    select_relevant_text_chunks,
    split_text_asset_content,
    write_jsonl_record,
    write_table_asset_links_from_jsonl,
)
from stage1_io import (
    clean_text,
    column_profiles,
    is_numeric_text,
    setup_logging,
    stable_hash,
    write_json,
    write_jsonl,
)

ENTITY_COLUMN_PRIORITY = ("name", "headline", "title", "identifier", "page_url")
EXCLUDED_COLUMNS = {"row_id", "image"}
DEFAULT_CACHE_DIR = Path("cache") / "wdc_mm_joinability"


@dataclass(frozen=True)
class _ValidatedTarget:
    url: str
    host: str
    port: int
    pinned_ip: str


@dataclass(frozen=True)
class _HttpProxy:
    host: str
    port: int


def _parse_http_proxy(value: str | None) -> _HttpProxy | None:
    proxy_url = clean_text(value)
    if not proxy_url:
        return None
    try:
        parsed = urlsplit(proxy_url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"malformed HTTP proxy URL: {exc}") from exc
    if (
        parsed.scheme.casefold() != "http"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "proxy_url must be an http://host[:port] URL without credentials"
        )
    return _HttpProxy(host=host, port=int(port or 80))


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect TCP to a vetted IP while retaining hostname TLS verification."""

    def __init__(
        self,
        pinned_ip: str,
        port: int,
        *,
        server_hostname: str,
        timeout: float,
    ) -> None:
        super().__init__(
            pinned_ip,
            port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._verified_server_hostname = server_hostname

    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        assert self.sock is not None
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self._verified_server_hostname,
        )


class _PinnedProxyHTTPSConnection(http.client.HTTPConnection):
    """CONNECT to a vetted target IP, then verify TLS for its hostname."""

    def __init__(
        self,
        proxy: _HttpProxy,
        *,
        pinned_ip: str,
        target_port: int,
        server_hostname: str,
        timeout: float,
    ) -> None:
        super().__init__(proxy.host, proxy.port, timeout=timeout)
        tunnel_host = (
            f"[{pinned_ip}]" if ipaddress.ip_address(pinned_ip).version == 6 else pinned_ip
        )
        self.set_tunnel(tunnel_host, target_port)
        self._verified_server_hostname = server_hostname
        self._context = ssl.create_default_context()

    def connect(self) -> None:
        super().connect()
        assert self.sock is not None
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self._verified_server_hostname,
        )


class _PinnedResponse:
    def __init__(
        self,
        response: http.client.HTTPResponse,
        connection: http.client.HTTPConnection,
        url: str,
        *,
        deadline: float,
        monotonic_fn: Callable[[], float],
        deadline_timer: threading.Timer,
    ) -> None:
        self._response = response
        self._connection = connection
        self.status_code = int(response.status)
        self.headers = response.headers
        self.url = url
        self._deadline = deadline
        self._monotonic_fn = monotonic_fn
        self._deadline_timer = deadline_timer
        content_type = clean_text(response.headers.get("Content-Type"))
        self.encoding = "utf-8"
        if "charset=" in content_type.casefold():
            self.encoding = content_type.rsplit("charset=", 1)[-1].split(";", 1)[0].strip()

    def __enter__(self) -> "_PinnedResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        read_size = max(1, min(int(chunk_size), 16 * 1024))
        while True:
            remaining = self._deadline - float(self._monotonic_fn())
            if remaining <= 0:
                raise TimeoutError("response deadline exceeded")
            if self._connection.sock is not None:
                self._connection.sock.settimeout(max(0.001, remaining))
            try:
                chunk = self._response.read1(read_size)
            except OSError as exc:
                if float(self._monotonic_fn()) >= self._deadline:
                    raise TimeoutError("response deadline exceeded") from exc
                raise
            if float(self._monotonic_fn()) > self._deadline:
                raise TimeoutError("response deadline exceeded")
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        self._deadline_timer.cancel()
        try:
            self._response.close()
        finally:
            self._connection.close()


def _pinned_http_get(
    url: str,
    *,
    pinned_ip: str,
    server_hostname: str,
    port: int,
    headers: dict[str, str],
    timeout: tuple[float, float],
    deadline: float,
    monotonic_fn: Callable[[], float],
    proxy_url: str | None = None,
    **_kwargs: Any,
) -> _PinnedResponse:
    parsed = urlsplit(url)
    remaining = deadline - float(monotonic_fn())
    if remaining <= 0:
        raise TimeoutError("response deadline exceeded")
    connect_timeout = min(float(timeout[0]), remaining)
    proxy = _parse_http_proxy(proxy_url)
    connection: http.client.HTTPConnection
    if proxy is not None and parsed.scheme.casefold() == "https":
        connection = _PinnedProxyHTTPSConnection(
            proxy,
            pinned_ip=pinned_ip,
            target_port=port,
            server_hostname=server_hostname,
            timeout=connect_timeout,
        )
    elif proxy is not None:
        connection = http.client.HTTPConnection(
            proxy.host,
            proxy.port,
            timeout=connect_timeout,
        )
        tunnel_host = (
            f"[{pinned_ip}]" if ipaddress.ip_address(pinned_ip).version == 6 else pinned_ip
        )
        connection.set_tunnel(tunnel_host, port)
    elif parsed.scheme.casefold() == "https":
        connection = _PinnedHTTPSConnection(
            pinned_ip,
            port,
            server_hostname=server_hostname,
            timeout=connect_timeout,
        )
    else:
        connection = http.client.HTTPConnection(
            pinned_ip,
            port,
            timeout=connect_timeout,
        )
    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    default_port = 443 if parsed.scheme.casefold() == "https" else 80
    host_name = f"[{server_hostname}]" if ":" in server_hostname else server_hostname
    host_header = host_name if port == default_port else f"{host_name}:{port}"
    deadline_expired = threading.Event()

    def abort_at_deadline() -> None:
        deadline_expired.set()
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    deadline_timer = threading.Timer(
        max(0.0, deadline - float(monotonic_fn())),
        abort_at_deadline,
    )
    deadline_timer.daemon = True
    deadline_timer.start()
    try:
        connection.request(
            "GET",
            request_target,
            headers={**headers, "Host": host_header},
        )
        remaining = deadline - float(monotonic_fn())
        if remaining <= 0:
            raise TimeoutError("response deadline exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(min(float(timeout[1]), remaining))
        response = connection.getresponse()
        if deadline_expired.is_set() or float(monotonic_fn()) >= deadline:
            raise TimeoutError("response deadline exceeded")
    except BaseException as exc:
        deadline_timer.cancel()
        connection.close()
        if deadline_expired.is_set() or float(monotonic_fn()) >= deadline:
            raise TimeoutError("response deadline exceeded") from exc
        raise
    return _PinnedResponse(
        response,
        connection,
        url,
        deadline=deadline,
        monotonic_fn=monotonic_fn,
        deadline_timer=deadline_timer,
    )


@dataclass
class WdcTableResult:
    source_table: dict[str, Any] | None
    entities: list[dict[str, Any]]
    image_urls_by_entity: dict[str, list[str]]
    skip_reason: str | None
    malformed_rows: int
    rows_truncated: bool = False


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


def _cache_error_class(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "fetch_failed"
    candidate = text.split(":", 1)[0].replace(" ", "_")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,99}", candidate):
        return candidate
    return "fetch_failed"


class WdcWebClient:
    """Bounded generic HTTP client backed by a durable SQLite page cache."""

    _RETRYABLE_HTTP_STATUSES = {408, 425, 429}
    _IMAGE_LOCK_STRIPES = 256

    @property
    def network_policy_fingerprint(self) -> str:
        """Fingerprint of the network/cache namespace used by this client."""
        return self.network_policy_version

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
        max_image_pixels: int = 25_000_000,
        max_total_image_bytes: int = 100_000_000_000,
        max_total_cache_bytes: int = 120_000_000_000,
        min_free_disk_bytes: int = 1_000_000_000,
        max_response_seconds: float = 120.0,
        min_image_side: int = 32,
        max_image_aspect_ratio: float = 20.0,
        max_redirects: int = 3,
        host_delay: float = 0.5,
        sleep_fn: Any = time.sleep,
        monotonic_fn: Any = time.monotonic,
        resolve_host_fn: Callable[[str], Iterable[str]] | None = None,
        pinned_request_fn: Callable[..., Any] | None = None,
        proxy_url: str | None = None,
        web_failure_callback: Callable[[dict[str, Any]], None] | None = None,
        media_failure_callback: Callable[[dict[str, Any]], None] | None = None,
        network_policy_version: str = "wdc-web-v1",
        pre_write_guard: Callable[[Path, int], None] | None = None,
        write_tracker: Any | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.database_path = self.cache_dir / "wdc_web.sqlite3"
        self.min_free_disk_bytes = max(0, int(min_free_disk_bytes))
        if pre_write_guard is None:
            reserve_bytes = self.min_free_disk_bytes

            def pre_write_guard(path: Path, estimated_bytes: int) -> None:
                probe = Path(path).resolve()
                while not probe.exists() and probe != probe.parent:
                    probe = probe.parent
                free = int(shutil.disk_usage(probe).free)
                required = reserve_bytes + max(0, int(estimated_bytes))
                if free < required:
                    raise OSError(
                        "insufficient disk for WDC web cache target "
                        f"{path}: free={free}, required={required}"
                    )

        if write_tracker is None:
            try:
                from wdc200k_io import GuardedWriteTracker
            except ModuleNotFoundError:
                from scripts_old.wdc200k_io import GuardedWriteTracker
            write_tracker = GuardedWriteTracker(
                self.database_path,
                pre_write_guard,
            )
        elif Path(write_tracker.path).resolve() != (
            self.database_path.resolve()
        ):
            raise ValueError(
                "web cache write tracker targets a different database"
            )
        self._write_tracker = write_tracker
        self._tracker_type = type(write_tracker)
        self._pre_write_guard = (
            pre_write_guard
            if pre_write_guard is not None
            else getattr(write_tracker, "guard", None)
        )
        self._write_tracker.before_write(64 * 1024)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir = self.cache_dir / "wdc_images"
        self._image_dir_tracker = self._tracker_type(
            self.image_dir,
            self._pre_write_guard,
        )
        self.session = session
        self._session_injected = session is not None
        self._session_headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "identity",
        }
        if self.session is not None:
            self.session.headers.update(self._session_headers)
        self.timeout = (max(0.01, connect_timeout), max(0.01, read_timeout))
        self.max_retries = max(0, max_retries)
        self.retry_base_seconds = max(0.0, retry_base_seconds)
        self.max_page_bytes = max(1, max_page_bytes)
        self.max_image_bytes = max(1, max_image_bytes)
        self.max_image_pixels = max(1, int(max_image_pixels))
        self.max_total_image_bytes = max(1, int(max_total_image_bytes))
        self.max_total_cache_bytes = max(1, int(max_total_cache_bytes))
        self.max_response_seconds = max(0.01, float(max_response_seconds))
        self.min_image_side = max(1, min_image_side)
        self.max_image_aspect_ratio = max(1.0, max_image_aspect_ratio)
        self.max_redirects = max(0, int(max_redirects))
        self.host_delay = max(0.0, host_delay)
        self.sleep_fn = sleep_fn
        self.monotonic_fn = monotonic_fn
        self.web_failure_callback = web_failure_callback
        self.media_failure_callback = media_failure_callback
        self.network_policy_version = clean_text(network_policy_version) or "wdc-web-v1"
        self.resolve_host_fn = resolve_host_fn
        self.pinned_request_fn = pinned_request_fn
        self.proxy_url = clean_text(proxy_url) or None
        _parse_http_proxy(self.proxy_url)
        self._host_map_lock = threading.Lock()
        self._host_locks: dict[str, threading.Lock] = {}
        self._last_request_by_host: dict[str, float] = {}
        self._image_quota_lock = threading.Lock()
        self._image_bytes_total = 0
        self._page_bytes_total = 0
        self._image_bytes_in_flight = 0
        self._cache_bytes_in_flight = 0
        self._image_locks = tuple(
            threading.Lock() for _ in range(self._IMAGE_LOCK_STRIPES)
        )
        self._image_content_locks = tuple(
            threading.Lock() for _ in range(self._IMAGE_LOCK_STRIPES)
        )
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _image_write_tracker(self, path: Path) -> Any:
        return self._tracker_type(Path(path), self._pre_write_guard)

    def _initialize_database(self) -> None:
        existing_bytes = (
            self.database_path.stat().st_size
            if self.database_path.is_file()
            else 0
        )
        self._write_tracker.before_write(
            max(64 * 1024, existing_bytes * 2)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._migrate_page_cache(connection)
            self._migrate_image_cache(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_failure_cache (
                    original_url TEXT NOT NULL,
                    policy_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_class TEXT NOT NULL,
                    http_status INTEGER,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (original_url, policy_fingerprint)
                )
                """
            )
            for row in connection.execute(
                """
                SELECT original_url, policy_fingerprint, file_name
                FROM image_cache WHERE bytes <= 0
                """
            ):
                path = self.image_dir / str(row["file_name"])
                try:
                    actual_bytes = path.stat().st_size
                except OSError:
                    actual_bytes = 0
                connection.execute(
                    """
                    UPDATE image_cache SET bytes = ?
                    WHERE original_url = ? AND policy_fingerprint = ?
                    """,
                    (
                        actual_bytes,
                        row["original_url"],
                        row["policy_fingerprint"],
                    ),
                )
            for row in connection.execute(
                """
                SELECT page_url, policy_fingerprint, text, image_urls_json
                FROM page_cache WHERE status = 'success' AND body_bytes <= 0
                """
            ):
                estimated_bytes = len((row["text"] or "").encode("utf-8")) + len(
                    (row["image_urls_json"] or "").encode("utf-8")
                )
                connection.execute(
                    """
                    UPDATE page_cache SET body_bytes = ?
                    WHERE page_url = ? AND policy_fingerprint = ?
                    """,
                    (
                        estimated_bytes,
                        row["page_url"],
                        row["policy_fingerprint"],
                    ),
                )
            self._image_bytes_total = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(bytes), 0)
                    FROM (
                        SELECT file_name, MAX(bytes) AS bytes
                        FROM image_cache
                        GROUP BY file_name
                    )
                    """
                ).fetchone()[0]
            )
            self._page_bytes_total = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(body_bytes), 0)
                    FROM page_cache WHERE status = 'success'
                    """
                ).fetchone()[0]
            )
            self._write_tracker.before_commit(0)
            connection.commit()

    @staticmethod
    def _table_columns(
        connection: sqlite3.Connection,
        table: str,
    ) -> dict[str, sqlite3.Row]:
        return {
            str(row["name"]): row
            for row in connection.execute(f"PRAGMA table_info({table})")
        }

    @staticmethod
    def _create_page_cache(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE page_cache (
                page_url TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                final_url TEXT,
                text TEXT,
                image_urls_json TEXT,
                http_status INTEGER,
                error TEXT,
                body_bytes INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (page_url, policy_fingerprint)
            )
            """
        )

    @classmethod
    def _migrate_page_cache(cls, connection: sqlite3.Connection) -> None:
        columns = cls._table_columns(connection, "page_cache")
        if not columns:
            cls._create_page_cache(connection)
            return
        primary_key = [
            name
            for name, row in sorted(
                columns.items(),
                key=lambda item: int(item[1]["pk"] or 0),
            )
            if int(row["pk"] or 0)
        ]
        if primary_key == ["page_url", "policy_fingerprint"]:
            return
        connection.execute("ALTER TABLE page_cache RENAME TO page_cache_legacy")
        cls._create_page_cache(connection)
        body_expression = "body_bytes" if "body_bytes" in columns else "0"
        policy_expression = (
            "policy_fingerprint"
            if "policy_fingerprint" in columns
            else "'wdc-web-v1'"
        )
        connection.execute(
            f"""
            INSERT INTO page_cache (
                page_url, policy_fingerprint, status, final_url, text,
                image_urls_json, http_status, error, body_bytes, updated_at
            )
            SELECT
                page_url, {policy_expression}, status, final_url, text,
                image_urls_json, http_status, error, {body_expression},
                updated_at
            FROM page_cache_legacy
            """
        )
        connection.execute("DROP TABLE page_cache_legacy")

    @staticmethod
    def _create_image_cache(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE image_cache (
                original_url TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                final_url TEXT NOT NULL,
                file_name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                bytes INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (original_url, policy_fingerprint)
            )
            """
        )

    @classmethod
    def _migrate_image_cache(cls, connection: sqlite3.Connection) -> None:
        columns = cls._table_columns(connection, "image_cache")
        if not columns:
            cls._create_image_cache(connection)
            return
        primary_key = [
            name
            for name, row in sorted(
                columns.items(),
                key=lambda item: int(item[1]["pk"] or 0),
            )
            if int(row["pk"] or 0)
        ]
        unique_sha_indexes = []
        for index in connection.execute("PRAGMA index_list(image_cache)"):
            if not int(index["unique"] or 0):
                continue
            index_name = str(index["name"])
            columns_in_index = [
                str(row["name"])
                for row in connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                )
            ]
            if "sha256" in columns_in_index:
                unique_sha_indexes.append(index_name)
        if (
            primary_key == ["original_url", "policy_fingerprint"]
            and not unique_sha_indexes
        ):
            return
        connection.execute(
            "ALTER TABLE image_cache RENAME TO image_cache_legacy"
        )
        cls._create_image_cache(connection)
        bytes_expression = "bytes" if "bytes" in columns else "0"
        policy_expression = (
            "policy_fingerprint"
            if "policy_fingerprint" in columns
            else "'wdc-web-v1'"
        )
        connection.execute(
            f"""
            INSERT INTO image_cache (
                original_url, policy_fingerprint, final_url, file_name,
                sha256, width, height, mime_type, bytes, updated_at
            )
            SELECT
                original_url, {policy_expression}, final_url, file_name,
                sha256, width, height, mime_type, {bytes_expression},
                updated_at
            FROM image_cache_legacy
            """
        )
        connection.execute("DROP TABLE image_cache_legacy")

    def _reserve_cache_bytes(self, amount: int, *, image: bool) -> bool:
        amount = max(0, int(amount))
        with self._image_quota_lock:
            if image and (
                self._image_bytes_total + self._image_bytes_in_flight + amount
                > self.max_total_image_bytes
            ):
                return False
            if (
                self._image_bytes_total
                + self._page_bytes_total
                + self._cache_bytes_in_flight
                + amount
                > self.max_total_cache_bytes
            ):
                return False
            self._cache_bytes_in_flight += amount
            if image:
                self._image_bytes_in_flight += amount
            return True

    def _release_cache_bytes(self, amount: int, *, image: bool) -> None:
        amount = max(0, int(amount))
        with self._image_quota_lock:
            self._cache_bytes_in_flight = max(0, self._cache_bytes_in_flight - amount)
            if image:
                self._image_bytes_in_flight = max(
                    0, self._image_bytes_in_flight - amount
                )

    def _resolved_addresses(self, host: str, *, deadline: float) -> list[str]:
        if float(self.monotonic_fn()) >= deadline:
            raise TimeoutError("dns deadline exceeded")
        try:
            return [str(ipaddress.ip_address(host))]
        except ValueError:
            pass
        if self.resolve_host_fn is not None:
            addresses = [clean_text(value) for value in self.resolve_host_fn(host)]
            if float(self.monotonic_fn()) >= deadline:
                raise TimeoutError("dns deadline exceeded")
            return addresses
        if self._session_injected and host.endswith(".test"):
            # RFC 2606 test hosts are non-routable; allow injected fake sessions
            # without depending on external DNS in unit tests.
            return ["93.184.216.34"]
        resolver = dns.resolver.Resolver()
        addresses: set[str] = set()
        for record_type in ("A", "AAAA"):
            remaining = deadline - float(self.monotonic_fn())
            if remaining <= 0:
                raise TimeoutError("dns deadline exceeded")
            try:
                answer = resolver.resolve(
                    host,
                    record_type,
                    lifetime=remaining,
                    search=False,
                )
            except dns.resolver.NXDOMAIN:
                return []
            except dns.resolver.NoAnswer:
                continue
            except (dns.resolver.LifetimeTimeout, dns.exception.Timeout) as exc:
                raise TimeoutError("dns deadline exceeded") from exc
            except dns.resolver.NoNameservers as exc:
                raise ValueError("dns_no_nameservers") from exc
            addresses.update(clean_text(value) for value in answer)
        return sorted(addresses)

    def _validate_target(self, value: str, *, deadline: float) -> _ValidatedTarget:
        url = clean_text(value)
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise ValueError(f"malformed:{exc}") from exc
        if parsed.scheme.casefold() not in {"http", "https"} or not host:
            raise ValueError("scheme_or_host")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("userinfo")
        try:
            normalized_host = host.encode("idna").decode("ascii").casefold().rstrip(".")
        except UnicodeError as exc:
            raise ValueError("invalid_hostname") from exc
        addresses = self._resolved_addresses(normalized_host, deadline=deadline)
        if not addresses:
            raise ValueError("unresolved_host")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise ValueError(f"invalid_resolved_address:{address}") from exc
            if not ip.is_global:
                raise ValueError(f"non_global_address:{address}")
        default_port = 443 if parsed.scheme.casefold() == "https" else 80
        return _ValidatedTarget(
            url=url,
            host=normalized_host,
            port=int(parsed.port or default_port),
            pinned_ip=addresses[0],
        )

    def _validate_url(self, value: str, *, deadline: float) -> str:
        return self._validate_target(value, deadline=deadline).url

    def _request_with_redirects(
        self,
        url: str,
        *,
        initial_target: _ValidatedTarget | None = None,
        deadline: float,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        current_target = initial_target or self._validate_target(url, deadline=deadline)
        for redirect_index in range(self.max_redirects + 1):
            if float(self.monotonic_fn()) >= deadline:
                raise TimeoutError("response deadline exceeded")
            current_url = current_target.url
            self._wait_for_host(current_url, deadline=deadline)
            if float(self.monotonic_fn()) >= deadline:
                raise TimeoutError("response deadline exceeded")
            response = self._session_get(
                current_url,
                pinned_target=current_target,
                deadline=deadline,
                allow_redirects=False,
                **kwargs,
            )
            status = int(response.status_code)
            if status not in {301, 302, 303, 307, 308}:
                response_url = clean_text(getattr(response, "url", None)) or current_url
                effective_url = current_url
                if response_url != current_url:
                    try:
                        effective_url = self._validate_url(response_url, deadline=deadline)
                    except Exception:
                        response.close()
                        raise
                return response, effective_url
            location = clean_text(response.headers.get("Location"))
            response.close()
            if not location:
                raise ValueError("redirect_without_location")
            next_url = urljoin(current_url, location)
            try:
                current_target = self._validate_target(next_url, deadline=deadline)
            except Exception as exc:
                raise ValueError(f"unsafe_redirect:{exc}") from exc
            if float(self.monotonic_fn()) >= deadline:
                raise TimeoutError("response deadline exceeded")
            if redirect_index >= self.max_redirects:
                raise ValueError("too_many_redirects")
        raise ValueError("too_many_redirects")

    def _wait_for_host(
        self,
        url: str,
        *,
        deadline: float | None = None,
    ) -> None:
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or parsed.netloc).casefold().rstrip(".")
        except ValueError:
            host = stable_hash(url, length=16)
        with self._host_map_lock:
            host_lock = self._host_locks.setdefault(host, threading.Lock())
        with host_lock:
            now = float(self.monotonic_fn())
            last_request = self._last_request_by_host.get(host)
            if last_request is not None:
                wait_seconds = self.host_delay - (now - last_request)
                if wait_seconds > 0:
                    remaining = (
                        wait_seconds
                        if deadline is None
                        else max(0.0, deadline - now)
                    )
                    self.sleep_fn(min(wait_seconds, remaining))
                    now = float(self.monotonic_fn())
            self._last_request_by_host[host] = now

    def _session_get(
        self,
        url: str,
        *,
        pinned_target: _ValidatedTarget | None = None,
        deadline: float,
        **kwargs: Any,
    ) -> Any:
        if self.session is not None:
            return self.session.get(url, **kwargs)
        target = pinned_target or self._validate_target(url, deadline=deadline)
        request_fn = self.pinned_request_fn or _pinned_http_get
        transport_kwargs = dict(kwargs)
        transport_kwargs.pop("timeout", None)
        return request_fn(
            url,
            pinned_ip=target.pinned_ip,
            server_hostname=target.host,
            port=target.port,
            headers=self._session_headers,
            timeout=self.timeout,
            deadline=deadline,
            monotonic_fn=self.monotonic_fn,
            proxy_url=self.proxy_url,
            **transport_kwargs,
        )

    @staticmethod
    def _unsupported_content_encoding(response: Any) -> str | None:
        content_encoding = clean_text(response.headers.get("Content-Encoding")).casefold()
        if not content_encoding or content_encoding == "identity":
            return None
        return content_encoding

    def _cached_page(self, page_url: str) -> tuple[str | None, dict[str, Any] | None]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM page_cache
                WHERE page_url = ? AND policy_fingerprint = ?
                """,
                (page_url, self.network_policy_version),
            ).fetchone()
        if row is None:
            return None, None
        if row["status"] != "success":
            return str(row["status"]), {
                "failure_type": "web_fetch_failure",
                "page_url": page_url,
                "status": str(row["status"]),
                "http_status": row["http_status"],
                "error": clean_text(row["error"]),
            }
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

    def cached_page_outcome(self, page_url: str) -> dict[str, Any] | None:
        """Return the current durable page outcome without making a request."""
        page_url = clean_text(page_url)
        status, payload = self._cached_page(page_url)
        if status is None or not isinstance(payload, dict):
            return None
        if status == "success":
            return {
                "status": "success",
                **payload,
                "policy_fingerprint": self.network_policy_version,
            }
        return {
            "status": status,
            "page_url": page_url,
            "error_class": _cache_error_class(payload.get("error")),
            "http_status": payload.get("http_status"),
            "policy_fingerprint": self.network_policy_version,
        }

    def cached_image_outcome(self, image_url: str) -> dict[str, Any] | None:
        """Return a durable image success/failure without network or mutation."""
        image_url = clean_text(image_url)
        with self._connect() as connection:
            success = connection.execute(
                """
                SELECT * FROM image_cache
                WHERE original_url = ? AND policy_fingerprint = ?
                """,
                (image_url, self.network_policy_version),
            ).fetchone()
            failure = connection.execute(
                """
                SELECT *
                FROM image_failure_cache
                WHERE original_url = ? AND policy_fingerprint = ?
                """,
                (image_url, self.network_policy_version),
            ).fetchone()
        if success is not None:
            try:
                candidate_path = (
                    self.image_dir / str(success["file_name"])
                ).resolve()
                image_root = self.image_dir.resolve()
                if not candidate_path.is_relative_to(image_root):
                    return None
                raster = self._validated_raster(candidate_path)
                if (
                    raster is None
                    or self._sha256_path(candidate_path)
                    != str(success["sha256"])
                    or raster[0] != int(success["width"])
                    or raster[1] != int(success["height"])
                    or raster[2] != str(success["mime_type"])
                    or candidate_path.stat().st_size
                    != int(success["bytes"])
                ):
                    return None
            except (OSError, RuntimeError):
                return None
            return {
                "status": "success",
                "image_url": image_url,
                "final_url": str(success["final_url"]),
                "file_name": str(success["file_name"]),
                "sha256": str(success["sha256"]),
                "width": int(success["width"]),
                "height": int(success["height"]),
                "mime_type": str(success["mime_type"]),
                "bytes": int(success["bytes"]),
                "policy_fingerprint": self.network_policy_version,
            }
        if failure is None:
            return None
        return {
            "status": str(failure["status"]),
            "image_url": image_url,
            "error_class": str(failure["error_class"]),
            "policy_fingerprint": self.network_policy_version,
            **(
                {}
                if failure["http_status"] is None
                else {"http_status": int(failure["http_status"])}
            ),
        }

    def _store_image_failure(
        self,
        image_url: str,
        *,
        error_class: str,
        http_status: int | None = None,
    ) -> None:
        self._write_tracker.before_write(
            16 * 1024
            + 2 * len(image_url.encode("utf-8"))
            + 2 * len(error_class.encode("utf-8"))
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                """
                SELECT 1 FROM image_cache
                WHERE original_url = ? AND policy_fingerprint = ?
                """,
                (image_url, self.network_policy_version),
            ).fetchone() is not None:
                return
            connection.execute(
                """
                INSERT INTO image_failure_cache (
                    original_url, policy_fingerprint, status,
                    error_class, http_status, updated_at
                ) VALUES (?, ?, 'terminal', ?, ?, ?)
                ON CONFLICT(original_url, policy_fingerprint) DO UPDATE SET
                    status = 'terminal',
                    error_class = excluded.error_class,
                    http_status = excluded.http_status,
                    updated_at = excluded.updated_at
                """,
                (
                    image_url,
                    self.network_policy_version,
                    clean_text(error_class)[:200] or "download_or_validation_failed",
                    http_status,
                    time.time(),
                ),
            )
            self._write_tracker.before_commit(0)
            connection.commit()

    def _store_page(self, payload: dict[str, Any]) -> None:
        encoded_images = json.dumps(
            payload["image_urls"],
            ensure_ascii=False,
        )
        self._write_tracker.before_write(
            16 * 1024
            + 2 * len(str(payload["page_url"]).encode("utf-8"))
            + 2 * len(str(payload["text"]).encode("utf-8"))
            + 2 * len(encoded_images.encode("utf-8"))
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO page_cache (
                    page_url, status, final_url, text, image_urls_json,
                    http_status, error, body_bytes, policy_fingerprint,
                    updated_at
                ) VALUES (?, 'success', ?, ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(page_url, policy_fingerprint) DO UPDATE SET
                    status = excluded.status,
                    final_url = excluded.final_url,
                    text = excluded.text,
                    image_urls_json = excluded.image_urls_json,
                    http_status = excluded.http_status,
                    error = NULL,
                    body_bytes = excluded.body_bytes,
                    policy_fingerprint = excluded.policy_fingerprint,
                    updated_at = excluded.updated_at
                """,
                (
                    payload["page_url"],
                    payload["final_url"],
                    payload["text"],
                    encoded_images,
                    200,
                    int(payload.get("body_bytes", 0)),
                    self.network_policy_version,
                    time.time(),
                ),
            )
            self._write_tracker.before_commit(0)
            connection.commit()

    def _store_page_failure(
        self,
        page_url: str,
        *,
        status: str,
        http_status: int | None,
        error: str,
    ) -> None:
        self._write_tracker.before_write(
            16 * 1024
            + 2 * len(page_url.encode("utf-8"))
            + 2 * len(error.encode("utf-8"))
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO page_cache (
                    page_url, status, final_url, text, image_urls_json,
                    http_status, error, policy_fingerprint, updated_at
                ) VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?, ?)
                ON CONFLICT(page_url, policy_fingerprint) DO UPDATE SET
                    status = excluded.status,
                    final_url = NULL,
                    text = NULL,
                    image_urls_json = NULL,
                    http_status = excluded.http_status,
                    error = excluded.error,
                    policy_fingerprint = excluded.policy_fingerprint,
                    updated_at = excluded.updated_at
                WHERE page_cache.status != 'success'
                   OR page_cache.policy_fingerprint
                      != excluded.policy_fingerprint
                """,
                (
                    page_url,
                    status,
                    http_status,
                    error[:1000],
                    self.network_policy_version,
                    time.time(),
                ),
            )
            self._write_tracker.before_commit(0)
            connection.commit()
        self._report_failure(
            self.web_failure_callback,
            {
                "failure_type": "web_fetch_failure",
                "page_url": page_url,
                "status": status,
                "http_status": http_status,
                "error": error[:1000],
            },
        )

    @staticmethod
    def _report_failure(
        callback: Callable[[dict[str, Any]], None] | None,
        record: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            callback(record)
        except Exception:
            logging.exception("WDC failure callback raised; continuing")

    def fetch_page(
        self,
        page_url: str,
        *,
        deadline_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch, extract, and cache one page without retaining response HTML."""
        response_seconds = (
            self.max_response_seconds
            if deadline_seconds is None
            else max(0.01, float(deadline_seconds))
        )
        request_retries = (
            self.max_retries
            if max_retries is None
            else max(0, int(max_retries))
        )
        deadline = float(self.monotonic_fn()) + response_seconds
        page_url = clean_text(page_url)
        if not extract_image_urls(page_url):
            self._store_page_failure(
                page_url,
                status="terminal",
                http_status=None,
                error="unsafe_url:invalid_url",
            )
            return None
        try:
            initial_target = self._validate_target(page_url, deadline=deadline)
            page_url = initial_target.url
        except Exception as exc:
            self._store_page_failure(
                page_url,
                status="terminal",
                http_status=None,
                error=f"unsafe_url:{exc}",
            )
            return None
        if float(self.monotonic_fn()) >= deadline:
            self._store_page_failure(
                page_url,
                status="terminal",
                http_status=None,
                error="page response deadline exceeded",
            )
            return None
        cache_status, cached = self._cached_page(page_url)
        if cache_status == "success":
            return cached
        if cache_status == "terminal":
            if isinstance(cached, dict):
                self._report_failure(self.web_failure_callback, cached)
            return None

        last_error = "request failed"
        last_http_status: int | None = None
        for attempt in range(request_retries + 1):
            reserved_bytes = 0
            try:
                response, final_url = self._request_with_redirects(
                    page_url,
                    initial_target=initial_target,
                    deadline=deadline,
                    stream=True,
                    timeout=self.timeout,
                )
                with response:
                    if float(self.monotonic_fn()) >= deadline:
                        self._store_page_failure(
                            page_url,
                            status="terminal",
                            http_status=None,
                            error="page response deadline exceeded",
                        )
                        return None
                    http_status = int(response.status_code)
                    if http_status >= 400:
                        retryable = (
                            http_status in self._RETRYABLE_HTTP_STATUSES
                            or http_status >= 500
                        )
                        if retryable and attempt < request_retries:
                            self.sleep_fn(self.retry_base_seconds * (2**attempt))
                            continue
                        self._store_page_failure(
                            page_url,
                            status="retryable" if retryable else "terminal",
                            http_status=http_status,
                            error=f"HTTP {http_status}",
                        )
                        return None
                    content_encoding = self._unsupported_content_encoding(response)
                    if content_encoding is not None:
                        self._store_page_failure(
                            page_url,
                            status="terminal",
                            http_status=http_status,
                            error=f"unsupported_content_encoding:{content_encoding}",
                        )
                        return None
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        if (
                            float(self.monotonic_fn()) >= deadline
                        ):
                            self._store_page_failure(
                                page_url,
                                status="terminal",
                                http_status=http_status,
                                error="page response deadline exceeded",
                            )
                            return None
                        if len(body) + len(chunk) > self.max_page_bytes:
                            self._store_page_failure(
                                page_url,
                                status="terminal",
                                http_status=http_status,
                                error="page response exceeded byte limit",
                            )
                            return None
                        if not self._reserve_cache_bytes(len(chunk), image=False):
                            self._store_page_failure(
                                page_url,
                                status="terminal",
                                http_status=http_status,
                                error="page cache or disk byte quota exceeded",
                            )
                            return None
                        reserved_bytes += len(chunk)
                        body.extend(chunk)
                    encoding = getattr(response, "encoding", None) or "utf-8"
                    html_text = bytes(body).decode(encoding, errors="replace")
                    text, image_urls = extract_html_assets(html_text, final_url)
                    payload = {
                        "page_url": page_url,
                        "final_url": final_url,
                        "text": text,
                        "image_urls": image_urls,
                        "body_bytes": len(body),
                    }
                    self._store_page(payload)
                    with self._image_quota_lock:
                        self._page_bytes_total += len(body)
                    payload.pop("body_bytes", None)
                    return payload
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_http_status = None
                if "deadline exceeded" in str(exc):
                    self._store_page_failure(
                        page_url,
                        status="terminal",
                        http_status=None,
                        error="page response deadline exceeded",
                    )
                    return None
                if attempt < request_retries:
                    self.sleep_fn(self.retry_base_seconds * (2**attempt))
                    continue
                status = "terminal" if "unsafe_redirect:" in str(exc) else "retryable"
                error = str(exc) if "unsafe_redirect:" in str(exc) else last_error
                self._store_page_failure(
                    page_url,
                    status=status,
                    http_status=last_http_status,
                    error=error,
                )
                return None
            finally:
                if reserved_bytes:
                    self._release_cache_bytes(reserved_bytes, image=False)

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
        deadline = float(self.monotonic_fn()) + self.max_response_seconds
        image_url = clean_text(image_url)
        cached_outcome = self.cached_image_outcome(image_url)
        if (
            isinstance(cached_outcome, dict)
            and cached_outcome.get("status") == "terminal"
        ):
            self._report_failure(
                self.media_failure_callback,
                {
                    "failure_type": "media_download_failure",
                    "entity_id": entity_id,
                    "page_url": page_url,
                    "image_url": image_url,
                    "source": source,
                    "error": cached_outcome.get("error_class")
                    or "download_or_validation_failed",
                },
            )
            return None
        if not extract_image_urls(image_url):
            self._store_image_failure(
                image_url,
                error_class="invalid_image_url",
            )
            self._report_failure(
                self.media_failure_callback,
                {
                    "failure_type": "media_download_failure",
                    "entity_id": entity_id,
                    "page_url": page_url,
                    "image_url": image_url,
                    "source": source,
                    "error": "invalid_image_url",
                },
            )
            return None
        if float(self.monotonic_fn()) >= deadline:
            self._store_image_failure(
                image_url,
                error_class="image_response_deadline_exceeded",
            )
            self._report_failure(
                self.media_failure_callback,
                {
                    "failure_type": "media_download_failure",
                    "entity_id": entity_id,
                    "page_url": page_url,
                    "image_url": image_url,
                    "source": source,
                    "error": "image response deadline exceeded",
                },
            )
            return None
        try:
            initial_target = self._validate_target(image_url, deadline=deadline)
            image_url = initial_target.url
        except Exception as exc:
            self._store_image_failure(
                image_url,
                error_class=f"unsafe_url:{type(exc).__name__}",
            )
            self._report_failure(
                self.media_failure_callback,
                {
                    "failure_type": "media_download_failure",
                    "entity_id": entity_id,
                    "page_url": page_url,
                    "image_url": image_url,
                    "source": source,
                    "error": f"unsafe_url:{exc}",
                },
            )
            return None
        try:
            if urlsplit(image_url).path.casefold().endswith(".svg"):
                self._store_image_failure(
                    image_url,
                    error_class="svg_rejected",
                )
                self._report_failure(
                    self.media_failure_callback,
                    {
                        "failure_type": "media_download_failure",
                        "entity_id": entity_id,
                        "page_url": page_url,
                        "image_url": image_url,
                        "source": source,
                        "error": "svg_rejected",
                    },
                )
                return None
        except ValueError:
            return None
        lock_index = int(stable_hash(image_url, length=8), 16) % len(self._image_locks)
        with self._image_locks[lock_index]:
            record = self._download_image_locked(
                image_url,
                initial_target=initial_target,
                deadline=deadline,
                page_url=page_url,
                source=source,
                entity_id=entity_id,
            )
        if record is None:
            self._store_image_failure(
                image_url,
                error_class="download_or_validation_failed",
            )
            self._report_failure(
                self.media_failure_callback,
                {
                    "failure_type": "media_download_failure",
                    "entity_id": entity_id,
                    "page_url": page_url,
                    "image_url": image_url,
                    "source": source,
                    "error": "download_or_validation_failed",
                },
            )
        return record

    def _download_image_locked(
        self,
        image_url: str,
        *,
        initial_target: _ValidatedTarget,
        deadline: float,
        page_url: str,
        source: str,
        entity_id: str,
    ) -> dict[str, Any] | None:
        self._image_dir_tracker.before_commit(0)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        image_key = (
            stable_hash(image_url, length=24)
            if self.network_policy_version == "wdc-web-v1"
            else stable_hash(
                image_url,
                self.network_policy_version,
                length=24,
            )
        )
        attempt_key = uuid.uuid4().hex
        temporary_path = self.image_dir / f".{image_key}.{attempt_key}.download.tmp"
        temporary_tracker = self._image_write_tracker(temporary_path)

        with self._connect() as connection:
            cached_row = connection.execute(
                """
                SELECT * FROM image_cache
                WHERE original_url = ? AND policy_fingerprint = ?
                """,
                (image_url, self.network_policy_version),
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
                )
            self._invalidate_image_cache_entry(
                original_url=image_url,
                policy_fingerprint=self.network_policy_version,
                file_name=str(cached_row["file_name"]),
            )

        for cached_path in sorted(self.image_dir.glob(f"image_{image_key}.*")):
            raster = self._validated_raster(cached_path)
            if raster is None:
                try:
                    cached_path.unlink()
                except OSError:
                    pass
                continue
            width, height, mime_type, _extension = raster
            existing_bytes = cached_path.stat().st_size
            digest = self._sha256_path(cached_path)
            try:
                with self._connect() as connection:
                    duplicate = connection.execute(
                        """
                        SELECT final_url, file_name
                        FROM image_cache
                        WHERE sha256 = ?
                        ORDER BY updated_at, original_url
                        LIMIT 1
                        """,
                        (digest,),
                    ).fetchone()
                if duplicate is not None:
                    shared_path = (
                        self.image_dir / str(duplicate["file_name"])
                    )
                    shared_raster = self._validated_raster(shared_path)
                    if (
                        shared_raster is not None
                        and self._sha256_path(shared_path) == digest
                    ):
                        cached_path.unlink(missing_ok=True)
                        width, height, mime_type, _extension = shared_raster
                        self._store_image_index(
                            original_url=image_url,
                            final_url=image_url,
                            path=shared_path,
                            width=width,
                            height=height,
                            mime_type=mime_type,
                        )
                        return self._image_record(
                            shared_path,
                            original_url=image_url,
                            final_url=image_url,
                            page_url=page_url,
                            source=source,
                            entity_id=entity_id,
                            width=width,
                            height=height,
                            mime_type=mime_type,
                            downloaded=False,
                        )
                with self._image_quota_lock:
                    if (
                        self._image_bytes_total + existing_bytes
                        > self.max_total_image_bytes
                        or self._image_bytes_total
                        + self._page_bytes_total
                        + existing_bytes
                        > self.max_total_cache_bytes
                    ):
                        cached_path.unlink(missing_ok=True)
                        return None
                    content_path = (
                        self.image_dir
                        / f"image_{digest}{_extension}"
                    )
                    if cached_path != content_path:
                        try:
                            self._image_write_tracker(
                                content_path
                            ).before_commit(0)
                            os.link(cached_path, content_path)
                        except FileExistsError:
                            if self._sha256_path(content_path) != digest:
                                return None
                        cached_path.unlink(missing_ok=True)
                    self._store_image_index(
                        original_url=image_url,
                        final_url=image_url,
                        path=content_path,
                        width=width,
                        height=height,
                        mime_type=mime_type,
                    )
                    self._image_bytes_total += existing_bytes
            except sqlite3.IntegrityError:
                try:
                    cached_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return None
            return self._image_record(
                content_path,
                original_url=image_url,
                final_url=image_url,
                page_url=page_url,
                source=source,
                entity_id=entity_id,
                width=width,
                height=height,
                mime_type=mime_type,
                downloaded=False,
            )

        for attempt in range(self.max_retries + 1):
            reserved_bytes = 0
            try:
                temporary_path.unlink(missing_ok=True)
                if float(self.monotonic_fn()) >= deadline:
                    return None
                response, final_url = self._request_with_redirects(
                    image_url,
                    initial_target=initial_target,
                    deadline=deadline,
                    stream=True,
                    timeout=self.timeout,
                )
                with response:
                    if float(self.monotonic_fn()) >= deadline:
                        return None
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
                    content_encoding = self._unsupported_content_encoding(response)
                    if content_encoding is not None:
                        self._report_failure(
                            self.media_failure_callback,
                            {
                                "failure_type": "media_download_failure",
                                "entity_id": entity_id,
                                "page_url": page_url,
                                "image_url": image_url,
                                "source": source,
                                "error": (
                                    "unsupported_content_encoding:"
                                    f"{content_encoding}"
                                ),
                            },
                        )
                        return None
                    content_type = clean_text(response.headers.get("Content-Type")).casefold()
                    if content_type and not content_type.startswith("image/"):
                        return None
                    if content_type.split(";", 1)[0] == "image/svg+xml":
                        return None
                    total_bytes = 0
                    with temporary_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            if (
                                float(self.monotonic_fn()) >= deadline
                            ):
                                return None
                            total_bytes += len(chunk)
                            if total_bytes > self.max_image_bytes:
                                return None
                            if not self._reserve_cache_bytes(len(chunk), image=True):
                                return None
                            reserved_bytes += len(chunk)
                            temporary_tracker.before_write(len(chunk))
                            handle.write(chunk)

                source_mime_type = content_type.split(";", 1)[0]
                if source_mime_type == "image/svg+xml":
                    return None
                try:
                    if urlsplit(final_url).path.casefold().endswith(".svg"):
                        return None
                except ValueError:
                    return None
                raster_path = temporary_path

                raster = self._validated_raster(raster_path)
                if raster is None:
                    return None
                width, height, mime_type, extension = raster
                digest = self._sha256_path(raster_path)
                image_path = self.image_dir / f"image_{digest}{extension}"
                content_lock = self._image_content_locks[
                    int(digest[:8], 16) % len(self._image_content_locks)
                ]
                with content_lock:
                    created = False
                    if image_path.exists():
                        if (
                            self._validated_raster(image_path) is None
                            or self._sha256_path(image_path) != digest
                        ):
                            return None
                    else:
                        try:
                            self._image_write_tracker(
                                image_path
                            ).before_commit(0)
                            os.link(raster_path, image_path)
                            created = True
                        except FileExistsError:
                            if (
                                self._validated_raster(image_path) is None
                                or self._sha256_path(image_path) != digest
                            ):
                                return None
                    self._store_image_index(
                        original_url=image_url,
                        final_url=final_url,
                        path=image_path,
                        width=width,
                        height=height,
                        mime_type=mime_type,
                    )
                    if created:
                        with self._image_quota_lock:
                            self._image_bytes_total += (
                                image_path.stat().st_size
                            )
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
                )
            except Exception as exc:
                if "deadline exceeded" in str(exc):
                    self._report_failure(
                        self.media_failure_callback,
                        {
                            "failure_type": "media_download_failure",
                            "entity_id": entity_id,
                            "page_url": page_url,
                            "image_url": image_url,
                            "source": source,
                            "error": "image response deadline exceeded",
                        },
                    )
                    return None
                if attempt < self.max_retries:
                    self.sleep_fn(self.retry_base_seconds * (2**attempt))
                    continue
                return None
            finally:
                if reserved_bytes:
                    self._release_cache_bytes(reserved_bytes, image=True)
                for cleanup_path in (temporary_path,):
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
                if width <= 0 or height <= 0 or width * height > self.max_image_pixels:
                    return None
                image.verify()
            with Image.open(path) as image:
                if image.width * image.height > self.max_image_pixels:
                    return None
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

    def _invalidate_image_cache_entry(
        self,
        *,
        original_url: str,
        policy_fingerprint: str,
        file_name: str,
    ) -> None:
        """Remove a stale alias without deleting content used by valid aliases."""
        digest_match = re.fullmatch(
            r"image_([0-9a-f]{64})\.[A-Za-z0-9]+",
            file_name,
        )
        lock_key = (
            digest_match.group(1)
            if digest_match is not None
            else stable_hash(file_name, length=64)
        )
        content_lock = self._image_content_locks[
            int(lock_key[:8], 16) % len(self._image_content_locks)
        ]
        path = self.image_dir / file_name
        self._write_tracker.before_write(64 * 1024)
        with content_lock:
            raster = self._validated_raster(path)
            actual_sha256 = (
                self._sha256_path(path)
                if raster is not None
                else None
            )
            remove_file = False
            removed_bytes = 0
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """
                    SELECT sha256
                    FROM image_cache
                    WHERE original_url = ? AND policy_fingerprint = ?
                      AND file_name = ?
                    """,
                    (original_url, policy_fingerprint, file_name),
                ).fetchone()
                if current is None:
                    return
                removed_bytes = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(bytes), 0)
                        FROM image_cache
                        WHERE file_name = ?
                        """,
                        (file_name,),
                    ).fetchone()[0]
                )
                if actual_sha256 is None:
                    connection.execute(
                        "DELETE FROM image_cache WHERE file_name = ?",
                        (file_name,),
                    )
                    remove_file = True
                elif str(current["sha256"]) != actual_sha256:
                    connection.execute(
                        """
                        DELETE FROM image_cache
                        WHERE file_name = ? AND sha256 != ?
                        """,
                        (file_name, actual_sha256),
                    )
                    valid_references = int(
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM image_cache
                            WHERE file_name = ? AND sha256 = ?
                            """,
                            (file_name, actual_sha256),
                        ).fetchone()[0]
                    )
                    remove_file = valid_references == 0
                    if remove_file:
                        connection.execute(
                            "DELETE FROM image_cache WHERE file_name = ?",
                            (file_name,),
                        )
                else:
                    return
                self._write_tracker.before_commit(0)
                connection.commit()
            if remove_file:
                try:
                    path.unlink(missing_ok=True)
                finally:
                    with self._image_quota_lock:
                        self._image_bytes_total = max(
                            0,
                            self._image_bytes_total - removed_bytes,
                        )

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
        digest = self._sha256_path(path)
        size = path.stat().st_size
        self._write_tracker.before_write(
            16 * 1024
            + 2 * len(original_url.encode("utf-8"))
            + 2 * len(final_url.encode("utf-8"))
            + 2 * len(path.name.encode("utf-8"))
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO image_cache (
                    original_url, final_url, file_name, sha256,
                    width, height, mime_type, bytes, policy_fingerprint,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(original_url, policy_fingerprint) DO UPDATE SET
                    final_url = excluded.final_url,
                    file_name = excluded.file_name,
                    sha256 = excluded.sha256,
                    width = excluded.width,
                    height = excluded.height,
                    mime_type = excluded.mime_type,
                    bytes = excluded.bytes,
                    policy_fingerprint = excluded.policy_fingerprint,
                    updated_at = excluded.updated_at
                """,
                (
                    original_url,
                    final_url,
                    path.name,
                    digest,
                    width,
                    height,
                    mime_type,
                    size,
                    self.network_policy_version,
                    time.time(),
                ),
            )
            connection.execute(
                """
                DELETE FROM image_failure_cache
                WHERE original_url = ? AND policy_fingerprint = ?
                """,
                (original_url, self.network_policy_version),
            )
            self._write_tracker.before_commit(0)
            connection.commit()

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
        return record


def build_wdc_bridge_assets_for_entity(
    entity: dict[str, Any],
    client: WdcWebClient,
    max_images_per_entity: int,
    text_asset_chunk_chars: int = 800,
    min_text_asset_chunk_chars: int = 120,
    max_text_asset_chunks_per_entity: int = 3,
    *,
    web_failure_callback: Callable[[dict[str, Any]], None] | None = None,
    media_failure_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Build webpage text and direct-first image assets for one WDC entity."""
    page_url = clean_text(entity.get("page_url"))
    try:
        fetched_page = client.fetch_page(page_url)
    except Exception as exc:
        WdcWebClient._report_failure(
            web_failure_callback,
            {
                "failure_type": "web_fetch_failure",
                "entity_id": entity.get("entity_id"),
                "page_url": page_url,
                "status": "client_exception",
                "http_status": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
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
                    "page_url": page_url,
                    "final_url": clean_text(page.get("final_url")) or page_url,
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
            except Exception as exc:
                WdcWebClient._report_failure(
                    media_failure_callback,
                    {
                        "failure_type": "media_download_failure",
                        "entity_id": entity.get("entity_id"),
                        "page_url": page_url,
                        "image_url": normalized_url,
                        "source": source,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
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


def _iter_scandir_gzip_files(root: Path, *, recursive: bool) -> Iterator[Path]:
    """Walk with closeable scandir iterators and no per-directory file list."""
    stack: list[Any] = []
    try:
        stack.append(os.scandir(root))
        while stack:
            iterator = stack[-1]
            try:
                entry = next(iterator)
            except StopIteration:
                iterator.close()
                stack.pop()
                continue
            try:
                if recursive and entry.is_dir(follow_symlinks=False):
                    stack.append(os.scandir(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(
                    ".json.gz"
                ):
                    yield Path(entry.path)
            except OSError:
                continue
    finally:
        for iterator in reversed(stack):
            try:
                iterator.close()
            except Exception:
                pass


def iter_wdc_gzip_paths(input_root: Path) -> Iterator[Path]:
    """Yield gzip host tables lazily, rotating across schema.org classes."""
    input_root = Path(input_root)
    class_dirs: list[Path] = []
    with os.scandir(input_root) as entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    class_dirs.append(Path(entry.path))
            except OSError:
                continue
    class_dirs.sort(key=lambda path: path.name)
    iterators: list[Iterator[Path]] = [
        _iter_scandir_gzip_files(input_root, recursive=False),
        *(
            _iter_scandir_gzip_files(class_dir, recursive=True)
            for class_dir in class_dirs
        ),
    ]
    active = list(iterators)
    try:
        while active:
            next_active: list[Iterator[Path]] = []
            for iterator in active:
                try:
                    yield next(iterator)
                except StopIteration:
                    continue
                next_active.append(iterator)
            active = next_active
    finally:
        for iterator in iterators:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


def iter_wdc_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every JSON-object row from a WDC gzip table without a row cap."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _read_rows(path: Path, max_rows: int) -> tuple[list[dict[str, Any]], int, bool]:
    rows: list[dict[str, Any]] = []
    malformed_rows = 0
    rows_truncated = False
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
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
            if max_rows > 0 and len(rows) >= max_rows:
                rows_truncated = True
                break
            rows.append(row)
    return rows, malformed_rows, rows_truncated


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
        rows_truncated=False,
    )


def _adapt_wdc_rows(
    raw_rows: list[dict[str, Any]],
    path: Path,
    input_root: Path,
    min_rows: int,
    min_cols: int,
    *,
    malformed_rows: int = 0,
    rows_truncated: bool = False,
) -> WdcTableResult:
    """Adapt already-read WDC rows to the internal table contract."""
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
        "provenance_builder": "build_wdc_mm_joinability_dataset.py",
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
        rows_truncated=rows_truncated,
    )


def read_wdc_table(
    path: Path,
    input_root: Path,
    min_rows: int,
    min_cols: int,
    max_rows: int = 0,
) -> WdcTableResult:
    """Read one WDC gzip host table and adapt it to the internal table contract."""
    raw_rows, malformed_rows, rows_truncated = _read_rows(path, max_rows)
    return _adapt_wdc_rows(
        raw_rows,
        path,
        input_root,
        min_rows,
        min_cols,
        malformed_rows=malformed_rows,
        rows_truncated=rows_truncated,
    )


class _FailureJsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.count = 0

    def reset(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.count = 0

    def record(self, record: dict[str, Any]) -> None:
        try:
            payload = dict(record)
            payload.setdefault("timestamp", time.time())
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.count += 1
        except Exception:
            logging.exception("Could not write WDC failure record to %s", self.path)


def _ensure_free_space(path: Path, minimum_free_bytes: int) -> None:
    required = max(0, int(minimum_free_bytes))
    if required <= 0:
        return
    free = int(shutil.disk_usage(path).free)
    if free < required:
        raise RuntimeError(
            f"insufficient disk space under {path}: free={free} required={required}"
        )


def iter_fair_entities(
    entities: Iterable[dict[str, Any]],
    *,
    lookahead: int = 256,
) -> Iterator[dict[str, Any]]:
    """Interleave hosts inside a bounded lookahead buffer."""
    iterator = iter(entities)
    lookahead = max(1, int(lookahead))
    while True:
        groups: dict[str, deque[dict[str, Any]]] = {}
        for _ in range(lookahead):
            try:
                entity = next(iterator)
            except StopIteration:
                break
            try:
                parsed = urlsplit(clean_text(entity.get("page_url")))
                host = (parsed.hostname or parsed.netloc).casefold().rstrip(".")
            except ValueError:
                host = ""
            key = host or clean_text(entity.get("entity_id"))
            groups.setdefault(key, deque()).append(entity)
        if not groups:
            return
        active = list(groups)
        while active:
            next_active: list[str] = []
            for key in active:
                group = groups[key]
                yield group.popleft()
                if group:
                    next_active.append(key)
            active = next_active


def _build_wdc_assets_parallel(
    *,
    entities: Iterable[dict[str, Any]],
    client: WdcWebClient,
    asset_writer: ShardedJsonlWriter,
    max_entities: int | None,
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    workers: int,
    max_in_flight: int,
    fairness_lookahead: int,
    flush_every_records: int,
    failure_recorder: _FailureJsonlRecorder,
    media_failure_recorder: _FailureJsonlRecorder,
) -> tuple[dict[str, list[str]], int, int]:
    """Fetch entity assets with one shared client and bounded future batches."""
    workers = max(1, int(workers or 1))
    max_in_flight = max(workers, int(max_in_flight or workers * 2))
    entity_to_assets: dict[str, list[str]] = defaultdict(list)
    text_asset_count = 0
    image_asset_count = 0
    written_assets = 0
    selected_count = 0
    iterator = iter(iter_fair_entities(entities, lookahead=fairness_lookahead))

    def fetch(entity: dict[str, Any]) -> list[dict[str, Any]]:
        return build_wdc_bridge_assets_for_entity(
            entity,
            client,
            max_images_per_entity,
            text_asset_chunk_chars,
            min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity,
            web_failure_callback=failure_recorder.record,
            media_failure_callback=media_failure_recorder.record,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        exhausted = False
        while not exhausted:
            batch: list[tuple[int, dict[str, Any]]] = []
            while len(batch) < max_in_flight:
                if max_entities is not None and max_entities > 0 and selected_count >= max_entities:
                    exhausted = True
                    break
                try:
                    entity = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                batch.append((selected_count, entity))
                selected_count += 1
            if not batch:
                break

            future_to_item = {
                pool.submit(fetch, entity): (index, entity)
                for index, entity in batch
            }
            completed: dict[int, list[dict[str, Any]]] = {}
            for future in as_completed(future_to_item):
                index, entity = future_to_item[future]
                try:
                    completed[index] = future.result()
                except Exception as exc:
                    failure_recorder.record(
                        {
                            "failure_type": "web_fetch_failure",
                            "entity_id": entity.get("entity_id"),
                            "page_url": entity.get("page_url"),
                            "status": "worker_error",
                            "http_status": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    completed[index] = []

            for index, _entity in batch:
                for record in completed.get(index, []):
                    write_jsonl_record(asset_writer, record)
                    entity_to_assets[record["entity_id"]].append(record["asset_id"])
                    if record.get("asset_type") == "image":
                        image_asset_count += 1
                    else:
                        text_asset_count += 1
                    written_assets += 1
                    if flush_every_records > 0 and written_assets % flush_every_records == 0:
                        asset_writer.flush()
    asset_writer.flush()
    return entity_to_assets, text_asset_count, image_asset_count


def _new_web_client(
    args: argparse.Namespace,
    *,
    cache_dir: Path,
    web_failure_recorder: _FailureJsonlRecorder,
    media_failure_recorder: _FailureJsonlRecorder,
    factory: Callable[..., WdcWebClient] | None,
) -> WdcWebClient:
    constructor: Callable[..., WdcWebClient] = factory or WdcWebClient
    return constructor(
        cache_dir=cache_dir,
        user_agent=args.web_user_agent,
        connect_timeout=args.web_connect_timeout,
        read_timeout=args.web_read_timeout,
        max_retries=args.web_max_retries,
        retry_base_seconds=args.web_retry_base_seconds,
        max_page_bytes=args.web_max_page_bytes,
        max_image_bytes=args.web_max_image_bytes,
        max_image_pixels=args.web_max_image_pixels,
        max_total_image_bytes=args.web_max_total_image_bytes,
        max_total_cache_bytes=args.web_max_total_cache_bytes,
        min_free_disk_bytes=args.min_free_disk_bytes,
        max_response_seconds=args.web_max_response_seconds,
        min_image_side=args.web_min_image_side,
        max_image_aspect_ratio=args.web_max_image_aspect_ratio,
        max_redirects=args.web_max_redirects,
        host_delay=args.web_host_delay,
        web_failure_callback=web_failure_recorder.record,
        media_failure_callback=media_failure_recorder.record,
    )


def _new_extractor(
    args: argparse.Namespace,
    factory: Callable[..., Any] | None,
) -> Any:
    return factory(args) if factory is not None else join_builder.LocalAttributeExtractor(args)


def build_dataset(
    args: argparse.Namespace,
    *,
    web_client_factory: Callable[..., WdcWebClient] | None = None,
    extractor_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Build the WDC multimodal joinability dataset using the shared query core."""
    if not getattr(args, "allow_unbounded", False) and (
        int(args.max_source_tables) <= 0
        or int(args.max_scanned_files) <= 0
        or int(args.max_rows_per_source_table) <= 0
    ):
        raise ValueError(
            "unbounded WDC input requires --allow_unbounded; use positive source/row caps"
        )
    args.query_rows_per_table = join_builder.configured_query_rows_per_table(args)
    args.max_train_query_row_views_per_join = (
        join_builder.configured_max_train_query_row_views_per_join(args)
    )
    args.explicit_join_fallback_mode = (
        join_builder.configured_explicit_join_fallback_mode(args)
    )
    args.explicit_join_fallback_ratio = (
        join_builder.configured_explicit_join_fallback_ratio(args)
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ensure_free_space(output_dir, args.min_free_disk_bytes)
    if cache_dir != output_dir:
        _ensure_free_space(cache_dir, args.min_free_disk_bytes)
    logging.info(
        "WDC safety limits: max_scanned_files=%d max_source_tables=%d "
        "max_rows_per_source_table=%d max_images_per_entity=%d "
        "max_total_cache_bytes=%d min_free_disk_bytes=%d allow_unbounded=%s",
        args.max_scanned_files,
        args.max_source_tables,
        args.max_rows_per_source_table,
        args.max_images_per_entity,
        args.web_max_total_cache_bytes,
        args.min_free_disk_bytes,
        args.allow_unbounded,
    )

    web_failures = _FailureJsonlRecorder(output_dir / "web_fetch_failures.jsonl")
    media_failures = _FailureJsonlRecorder(output_dir / "media_download_failures.jsonl")
    web_failures.reset()
    media_failures.reset()
    model_errors_path = clean_text(getattr(args, "model_attribute_errors_path", ""))
    if not model_errors_path:
        model_errors_path = str(output_dir / "model_attribute_errors.jsonl")
        args.model_attribute_errors_path = model_errors_path
    model_error_file = Path(model_errors_path)
    model_error_file.parent.mkdir(parents=True, exist_ok=True)
    model_error_file.write_text("", encoding="utf-8")

    records_per_shard = max(1, int(args.records_per_shard))
    flush_every = max(1, int(args.flush_every_records))
    source_writer = ShardedJsonlWriter(output_dir / "source_tables", records_per_shard)
    entities_writer = ShardedJsonlWriter(output_dir / "entities", records_per_shard)
    source_split_records: list[dict[str, str]] = []
    wiki_to_entity_id: dict[str, str] = {}
    skip_reasons: Counter[str] = Counter()
    processed_tables = 0
    scanned_files = 0
    scanned_file_cap_reached = False
    source_table_cap_reached = False
    skipped_tables = 0
    malformed_rows = 0
    source_table_count = 0
    entity_count = 0
    row_capped_source_tables = 0

    with source_writer as source_handle, entities_writer as entity_handle:
        for gzip_path in iter_wdc_gzip_paths(input_dir):
            if args.max_scanned_files > 0 and scanned_files >= args.max_scanned_files:
                scanned_file_cap_reached = True
                break
            if args.max_source_tables > 0 and source_table_count >= args.max_source_tables:
                source_table_cap_reached = True
                break
            scanned_files += 1
            processed_tables += 1
            try:
                result = read_wdc_table(
                    gzip_path,
                    input_dir,
                    args.min_rows,
                    args.min_cols,
                    args.max_rows_per_source_table,
                )
            except Exception as exc:
                skipped_tables += 1
                skip_reasons["gzip_read_error"] += 1
                logging.warning("Skipping WDC host table %s: %s", gzip_path, exc)
                continue
            malformed_rows += result.malformed_rows
            if result.rows_truncated:
                row_capped_source_tables += 1
            if result.source_table is None:
                skipped_tables += 1
                skip_reasons[result.skip_reason or "unknown"] += 1
                continue
            source_table = result.source_table
            if (
                join_builder.choose_entity_column(
                    source_table,
                    min_linked_rows=args.query_rows_per_table,
                )
                is None
            ):
                skipped_tables += 1
                skip_reasons["too_few_candidate_entity_rows"] += 1
                continue
            write_jsonl_record(source_handle, source_table)
            source_table_count += 1
            source_split_records.append(
                {
                    "source_table_id": source_table["source_table_id"],
                    "page_title": clean_text(source_table.get("page_title")),
                }
            )
            for entity in result.entities:
                write_jsonl_record(entity_handle, entity)
                entity_count += 1
                wiki_title = clean_text(entity.get("wiki_title"))
                entity_id = clean_text(entity.get("entity_id"))
                if wiki_title and entity_id:
                    wiki_to_entity_id[wiki_title] = entity_id
                    wiki_to_entity_id[normalize_title(wiki_title)] = entity_id
            if source_table_count % flush_every == 0:
                source_handle.flush()
                entity_handle.flush()
    logging.info(
        "WDC input scan finished: scanned_files=%d accepted_source_tables=%d "
        "row_capped_source_tables=%d scan_cap_reached=%s source_cap_reached=%s",
        scanned_files,
        source_table_count,
        row_capped_source_tables,
        scanned_file_cap_reached,
        source_table_cap_reached,
    )

    web_client = _new_web_client(
        args,
        cache_dir=cache_dir,
        web_failure_recorder=web_failures,
        media_failure_recorder=media_failures,
        factory=web_client_factory,
    )
    bridge_assets_writer = ShardedJsonlWriter(output_dir / "bridge_assets", records_per_shard)
    with bridge_assets_writer as asset_handle:
        entity_to_assets, text_asset_count, image_asset_count = _build_wdc_assets_parallel(
            entities=iter_jsonl_records(entities_writer.paths()),
            client=web_client,
            asset_writer=asset_handle,
            max_entities=args.max_entities,
            max_images_per_entity=args.max_images_per_entity,
            text_asset_chunk_chars=args.text_asset_chunk_chars,
            min_text_asset_chunk_chars=args.min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity=args.max_text_asset_chunks_per_entity,
            workers=args.web_workers,
            max_in_flight=args.web_max_in_flight,
            fairness_lookahead=args.web_fairness_lookahead,
            flush_every_records=flush_every,
            failure_recorder=web_failures,
            media_failure_recorder=media_failures,
        )

    table_asset_links_writer = ShardedJsonlWriter(
        output_dir / "table_asset_links", records_per_shard
    )
    with table_asset_links_writer as link_handle:
        table_asset_link_count = write_table_asset_links_from_jsonl(
            source_writer.paths(),
            [],
            link_handle,
            wiki_to_entity_id,
            entity_to_assets,
            flush_every,
        )

    splits = join_builder.source_splits(source_split_records, args)
    source_to_split = join_builder.split_map(splits)
    assets = join_builder.load_assets(bridge_assets_writer.paths())
    model_cache_path = cache_dir / "model_attribute_extractions.jsonl"
    cache = join_builder.ExtractionCache(
        model_cache_path,
        reuse=not args.no_reuse_model_cache,
    )
    query_auto_check_cache_path = cache_dir / "query_recovery_auto_checks.jsonl"

    query_auto_check_cache = join_builder.ExtractionCache(
        query_auto_check_cache_path,
        reuse=not args.no_reuse_model_cache,
        record_key_alias=join_builder.query_recovery_auto_check_record_key,
    )
    concurrency_state = join_builder.ModelConcurrencyState.from_args(args)
    progress: Any | None = None
    if getattr(args, "model_progress", True):
        planned_keys = join_builder.estimate_model_analysis_keys(
            source_paths=source_writer.paths(),
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
        )
        cached_keys = {
            key
            for key in planned_keys
            if key in cache.items
            and join_builder.cached_extraction_is_reusable(cache.items[key], args)
        }
        progress = join_builder.ModelAnalysisProgress(
            total=len(planned_keys), cached_keys=cached_keys, enabled=True
        )

    precomputed_text_task_count = 0
    precomputed_image_task_count = 0
    extractor: Any | None = None
    if args.precompute_model_cache or args.precompute_text_model_cache:
        text_tasks = join_builder.collect_extraction_tasks_from_tables(
            source_paths=source_writer.paths(),
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
            asset_types={"text"},
        )
        image_tasks = (
            join_builder.collect_extraction_tasks_from_tables(
                source_paths=source_writer.paths(),
                assets=assets,
                entity_to_assets=entity_to_assets,
                wiki_to_entity_id=wiki_to_entity_id,
                args=args,
                asset_types={"image"},
            )
            if args.precompute_model_cache
            else []
        )
        pending_text_tasks = join_builder.tasks_requiring_model_analysis(
            text_tasks, cache, args
        )
        pending_image_tasks = join_builder.tasks_requiring_model_analysis(
            image_tasks, cache, args
        )
        precomputed_text_task_count = len(pending_text_tasks)
        precomputed_image_task_count = len(pending_image_tasks)
        task_groups = {
            "text": pending_text_tasks,
            "image": pending_image_tasks,
        }
        marker_context = join_builder.build_model_marker_context(
            args=args,
            tasks_by_kind=task_groups,
            upstream_identities=join_builder.source_shard_identities(
                source_writer.paths()
            ),
        )
        join_builder.write_model_start_marker(
            clean_text(args.model_start_marker),
            context=marker_context,
        )
        if pending_text_tasks or pending_image_tasks:
            join_builder.wait_for_model_ready_marker(
                clean_text(args.model_ready_marker),
                context=marker_context,
                timeout_seconds=args.model_ready_timeout_seconds,
            )
            extractor = _new_extractor(args, extractor_factory)
            join_builder.precompute_extraction_task_groups(
                extractor=extractor,
                cache=cache,
                tasks_by_kind=task_groups,
                args=args,
                state=concurrency_state,
                progress=progress,
                marker_context=marker_context,
            )
        else:
            for model_kind in ("text", "image"):
                join_builder.write_model_done_marker(
                    join_builder.model_done_marker_for_kind(args, model_kind),
                    model_kind=model_kind,
                    task_count=0,
                    context=marker_context,
                )
        if not args.precompute_model_cache and image_asset_count:
            extractor = extractor or _new_extractor(args, extractor_factory)
    else:
        marker_context = join_builder.build_model_marker_context(
            args=args,
            tasks_by_kind={"text": [], "image": []},
            upstream_identities=join_builder.source_shard_identities(
                source_writer.paths()
            ),
        )
        join_builder.write_model_start_marker(
            clean_text(args.model_start_marker),
            context=marker_context,
        )
        extractor = _new_extractor(args, extractor_factory)

    if join_builder.auto_check_required(extractor):
        final_query_auto_check_plans: list[
            join_builder.QueryRecoveryAutoCheckPlan
        ] = []
        for source_table in iter_jsonl_records(source_writer.paths()):
            source_table_id = str(source_table["source_table_id"])
            join_builder.build_table_join_records(
                source_table=source_table,
                split=source_to_split.get(source_table_id, "test"),
                assets=assets,
                entity_to_assets=entity_to_assets,
                wiki_to_entity_id=wiki_to_entity_id,
                extractor=extractor,
                cache=cache,
                progress=None,
                concurrency_state=concurrency_state,
                extraction_writer=join_builder.ListRecordWriter(),
                recovery_writer=join_builder.ListRecordWriter(),
                args=args,
                query_auto_check_cache=query_auto_check_cache,
                apply_query_auto_check=False,
                query_recovery_plans_out=final_query_auto_check_plans,
            )
        join_builder.finalize_query_recovery_auto_checks(
            plans=final_query_auto_check_plans,
            extractor=extractor,
            cache=query_auto_check_cache,
            args=args,
            concurrency_state=concurrency_state,
        )

    query_writer = ShardedJsonlWriter(output_dir / "query_tables", records_per_shard)
    data_lake_writer = ShardedJsonlWriter(
        output_dir / "data_lake_tables", records_per_shard
    )
    extraction_writer = ShardedJsonlWriter(
        output_dir / "attribute_extractions", records_per_shard
    )
    recovery_writer = ShardedJsonlWriter(
        output_dir / "evidence_recoveries", records_per_shard
    )
    qrels: list[dict[str, Any]] = []
    table_decisions: list[dict[str, Any]] = []
    query_table_count = 0
    data_lake_table_count = 0
    queryable_source_tables = 0
    multimodal_queryable_source_tables = 0
    explicit_join_source_tables = 0
    rejected_source_tables = 0
    implicit_query_table_count = 0
    explicit_join_query_table_count = 0
    implicit_query_counts_by_split = {
        "train": 0,
        "dev": 0,
        "test": 0,
    }
    explicit_candidate_splits: dict[str, str] = {}
    explicit_candidate_source_ids: dict[str, str] = {}
    explicit_candidate_decision_indices: dict[str, int] = {}
    explicit_candidate_counts = {"train": 0, "dev": 0, "test": 0}
    explicit_candidate_source_counts = {"train": 0, "dev": 0, "test": 0}
    try:
        with (
            query_writer as query_handle,
            data_lake_writer as data_lake_handle,
            extraction_writer as extraction_handle,
            recovery_writer as recovery_handle,
        ):
            for source_table in iter_jsonl_records(source_writer.paths()):
                source_table_id = str(source_table["source_table_id"])
                split = source_to_split.get(source_table_id, "test")
                query_tables, data_lake_tables, table_qrels, decision = (
                    join_builder.build_table_join_records(
                        source_table=source_table,
                        split=split,
                        assets=assets,
                        entity_to_assets=entity_to_assets,
                        wiki_to_entity_id=wiki_to_entity_id,
                        extractor=extractor,
                        cache=cache,
                        progress=progress,
                        concurrency_state=concurrency_state,
                        extraction_writer=extraction_handle,
                        recovery_writer=recovery_handle,
                        args=args,
                        query_auto_check_cache=query_auto_check_cache,
                        finalize_query_recoveries=True,
                    )
                )
                if query_tables:
                    queryable_source_tables += 1
                    if decision.get("reason") == "explicit_join_fallback":
                        explicit_join_source_tables += 1
                        explicit_join_query_table_count += len(query_tables)
                    else:
                        multimodal_queryable_source_tables += 1
                        implicit_query_table_count += len(query_tables)
                        implicit_query_counts_by_split[split] += len(
                            query_tables
                        )
                else:
                    rejected_source_tables += 1
                candidates = decision.get("explicit_join_candidates")
                if not isinstance(candidates, list):
                    candidate = decision.get("explicit_join_candidate")
                    candidates = [candidate] if isinstance(candidate, dict) else []
                deferred_candidate = (
                    args.explicit_join_fallback_mode == "match_implicit"
                    and bool(candidates)
                )
                decision["source_table_id"] = source_table_id
                decision["split"] = split
                if deferred_candidate:
                    for candidate in candidates:
                        candidate_id = join_builder.clean_text(
                            candidate.get("candidate_id")
                        )
                        if not candidate_id:
                            raise ValueError(
                                "explicit join candidate is missing candidate_id: "
                                f"{source_table_id}"
                            )
                        explicit_candidate_splits[candidate_id] = split
                        explicit_candidate_source_ids[candidate_id] = source_table_id
                        explicit_candidate_decision_indices[candidate_id] = len(
                            table_decisions
                        )
                table_decisions.append(decision)
                for record in query_tables:
                    write_jsonl_record(query_handle, record)
                    query_table_count += 1
                    splits[split]["query_table_ids"].append(record["table_id"])
                for record in ([] if deferred_candidate else data_lake_tables):
                    write_jsonl_record(data_lake_handle, record)
                    data_lake_table_count += 1
                    splits[split]["data_lake_table_ids"].append(record["table_id"])
                qrels.extend(table_qrels)
                if (query_table_count + data_lake_table_count) % flush_every == 0:
                    query_handle.flush()
                    data_lake_handle.flush()
                    extraction_handle.flush()
                    recovery_handle.flush()
                    evidence_label_handle.flush()

            if args.explicit_join_fallback_mode == "match_implicit":
                selected_explicit, explicit_candidate_counts = (
                    join_builder.select_balanced_explicit_join_candidates(
                        candidate_splits=explicit_candidate_splits,
                        implicit_query_counts=implicit_query_counts_by_split,
                        args=args,
                    )
                )
                source_candidates_by_split: dict[str, set[str]] = {
                    "train": set(),
                    "dev": set(),
                    "test": set(),
                }
                for candidate_id, candidate_split in explicit_candidate_splits.items():
                    source_candidates_by_split[candidate_split].add(
                        explicit_candidate_source_ids[candidate_id]
                    )
                explicit_candidate_source_counts = {
                    split_name: len(source_ids)
                    for split_name, source_ids in source_candidates_by_split.items()
                }
                seen_candidates: set[str] = set()
                selected_by_source: dict[str, list[str]] = defaultdict(list)
                for candidate_id in selected_explicit:
                    selected_by_source[
                        explicit_candidate_source_ids[candidate_id]
                    ].append(candidate_id)
                candidate_source_ids = sorted(
                    set(explicit_candidate_source_ids.values())
                )
                for source_table in iter_jsonl_records(source_writer.paths()):
                    source_table_id = str(source_table["source_table_id"])
                    if source_table_id not in set(candidate_source_ids):
                        continue
                    split = source_to_split.get(source_table_id, "test")
                    selected_candidate_ids = sorted(
                        selected_by_source.get(source_table_id, [])
                    )
                    if not selected_candidate_ids:
                        record = join_builder.raw_data_lake_record(
                            source_table, split
                        )
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
                        splits[split]["data_lake_table_ids"].append(
                            record["table_id"]
                        )
                        continue
                    seen_candidates.update(
                        explicit_candidate_source_ids[candidate_id]
                        for candidate_id in selected_candidate_ids
                    )
                    decision_index = explicit_candidate_decision_indices[
                        selected_candidate_ids[0]
                    ]
                    original_decision = table_decisions[decision_index]
                    explicit_queries: list[dict[str, Any]] = []
                    explicit_targets: list[dict[str, Any]] = []
                    explicit_qrels: list[dict[str, Any]] = []
                    explicit_decisions: list[dict[str, Any]] = []
                    selected_candidate_decisions = [
                        next(
                            item
                            for item in original_decision[
                                "explicit_join_candidates"
                            ]
                            if item.get("candidate_id") == candidate_id
                        )
                        for candidate_id in selected_candidate_ids
                    ]
                    selected_candidates = (
                        join_builder.rebuild_selected_explicit_join_candidates(
                            source_table=source_table,
                            split=split,
                            candidate_decisions=selected_candidate_decisions,
                            args=args,
                        )
                    )
                    for candidate in selected_candidates:
                        (
                            candidate_queries,
                            candidate_targets,
                            candidate_qrels,
                            candidate_result_decision,
                        ) = join_builder.materialize_balanced_explicit_join_candidate(
                            source_table=source_table,
                            split=split,
                            candidate_decision=candidate,
                            args=args,
                        )
                        explicit_queries.extend(candidate_queries)
                        explicit_targets.extend(candidate_targets)
                        explicit_qrels.extend(candidate_qrels)
                        explicit_decisions.append(candidate_result_decision)
                    explicit_decision = {
                        **original_decision,
                        **explicit_decisions[0],
                        "source_table_id": source_table_id,
                        "split": split,
                        "qualified_columns": [
                            qualified
                            for item in explicit_decisions
                            for qualified in item.get("qualified_columns", [])
                        ],
                        "explicit_join_candidates": selected_candidates,
                        "explicit_join_candidate": selected_candidates[0],
                        "explicit_join_query_count": len(explicit_queries),
                    }
                    table_decisions[decision_index] = explicit_decision
                    rejected_source_tables -= 1
                    queryable_source_tables += 1
                    explicit_join_source_tables += 1
                    for record in explicit_queries:
                        write_jsonl_record(query_handle, record)
                        query_table_count += 1
                        explicit_join_query_table_count += 1
                        splits[split]["query_table_ids"].append(
                            record["table_id"]
                        )
                    for record in explicit_targets:
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
                        splits[split]["data_lake_table_ids"].append(
                            record["table_id"]
                        )
                    qrels.extend(explicit_qrels)
                if seen_candidates != set(candidate_source_ids):
                    raise ValueError(
                        "explicit join candidate source replay is incomplete"
                    )
                if explicit_join_query_table_count != implicit_query_table_count:
                    raise ValueError(
                        "explicit and implicit query counts are not balanced: "
                        f"explicit={explicit_join_query_table_count}, "
                        f"implicit={implicit_query_table_count}"
                    )
    finally:
        if progress is not None:
            progress.close()

    for split in ("train", "dev", "test"):
        splits[split]["query_table_ids"] = sorted(splits[split]["query_table_ids"])
        splits[split]["data_lake_table_ids"] = sorted(
            splits[split]["data_lake_table_ids"]
        )
    join_builder.validate_implicit_query_uniqueness(
        qrels,
        expected_query_count=implicit_query_table_count,
    )
    qrels_count = write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "table_queryability_decisions.jsonl", table_decisions)
    write_json(output_dir / "splits.json", splits)

    stats = {
        "processed_tables": processed_tables,
        "scanned_files": scanned_files,
        "scanned_file_cap_reached": scanned_file_cap_reached,
        "source_table_cap_reached": source_table_cap_reached,
        "skipped_tables": skipped_tables,
        "source_tables": source_table_count,
        "malformed_rows": malformed_rows,
        "row_capped_source_tables": row_capped_source_tables,
        "source_entities": entity_count,
        "queryable_source_tables": queryable_source_tables,
        "multimodal_queryable_source_tables": multimodal_queryable_source_tables,
        "explicit_join_source_tables": explicit_join_source_tables,
        "implicit_join_query_tables": implicit_query_table_count,
        "explicit_join_query_tables": explicit_join_query_table_count,
        "explicit_join_candidate_tables": sum(
            explicit_candidate_source_counts.values()
        ),
        "explicit_join_candidate_tables_by_split": explicit_candidate_source_counts,
        "explicit_join_candidate_queries": sum(explicit_candidate_counts.values()),
        "explicit_join_candidate_queries_by_split": explicit_candidate_counts,
        "implicit_join_query_tables_by_split": implicit_query_counts_by_split,
        "rejected_source_tables": rejected_source_tables,
        "query_tables": query_table_count,
        "data_lake_tables": data_lake_table_count,
        "qrels": qrels_count,
        "text_assets": text_asset_count,
        "image_assets": image_asset_count,
        "table_asset_links": table_asset_link_count,
        "web_fetch_failures": web_failures.count,
        "media_download_failures": media_failures.count,
        "attribute_extractions": extraction_writer.total_records,
        "evidence_recoveries": recovery_writer.total_records,
        "model_inference": join_builder.model_call_stats_summary(extractor),
        "model_auto_check": {
            **join_builder.summarize_model_auto_check_records(
                extraction_writer.paths()
            ),
            "current_process": join_builder.model_auto_check_summary(extractor),
        },
        "model_concurrency": concurrency_state.summary(),
        "precomputed_text_model_cache_tasks": precomputed_text_task_count,
        "precomputed_image_model_cache_tasks": precomputed_image_task_count,
        "web_workers": max(1, int(args.web_workers)),
        "web_max_in_flight": max(
            max(1, int(args.web_workers)), int(args.web_max_in_flight or args.web_workers * 2)
        ),
        "max_rows_per_source_table": args.max_rows_per_source_table,
        "max_source_tables": args.max_source_tables,
        "allow_unbounded": args.allow_unbounded,
        "min_free_disk_bytes": args.min_free_disk_bytes,
        "max_total_image_bytes": args.web_max_total_image_bytes,
        "max_total_cache_bytes": args.web_max_total_cache_bytes,
        "max_response_seconds": args.web_max_response_seconds,
        "web_fairness_lookahead": args.web_fairness_lookahead,
        "min_recovered_value_ratio": args.min_recovered_value_ratio,
        "min_recovery_denominator": args.min_recovery_denominator,
        "query_rows_per_table": args.query_rows_per_table,
        "max_train_query_row_views_per_join": (
            args.max_train_query_row_views_per_join
        ),
        "explicit_join_fallback_mode": args.explicit_join_fallback_mode,
        "explicit_join_fallback_ratio": args.explicit_join_fallback_ratio,
        "skipped_reasons": dict(skip_reasons),
        "safety_config": {
            "max_scanned_files": args.max_scanned_files,
            "max_source_tables": args.max_source_tables,
            "max_rows_per_source_table": args.max_rows_per_source_table,
            "max_entities": args.max_entities,
            "max_images_per_entity": args.max_images_per_entity,
            "max_total_cache_bytes": args.web_max_total_cache_bytes,
            "min_free_disk_bytes": args.min_free_disk_bytes,
            "allow_unbounded": args.allow_unbounded,
        },
        "notes": [
            "WDC page_url is fetched for every selected entity even when direct images succeed",
            "direct image-column URLs and webpage images share one per-entity quota",
            "the source image attribute is excluded from every emitted table",
            "tables without enough linked entity rows for one query are filtered before network and model work",
            "train join chains emit deterministic disjoint row views while dev/test retain one canonical view",
            "query row views balance recoverable evidence while projected targets retain every source row",
            "wide source tables may emit one query variant per qualifying bridge attribute",
            "qualified attributes with the same exact visible query row view are "
            "merged into one query with multiple positive targets",
            "match_implicit deterministically selects one viable explicit join per implicit query within each split",
            "query/target/qrel/evidence construction is delegated to build_mm_joinability_dataset.py",
            "every local-positive evidence candidate for a final accepted query receives an exhaustive auto-check before evidence_recoveries are materialized",
            "evidence_recoveries contain supported paths only; omitted evidence is an implicit negative",
        ],
    }
    write_json(output_dir / "stats.json", stats)

    manifest = {
        "format": "sharded_jsonl",
        "source_corpus": "WDC Schema.org Table Corpus 2023",
        "records_per_shard": records_per_shard,
        "artifact_references": {
            "data_lake_tables": {
                "field": "source_table_ref",
                "target_artifact": "source_tables",
                "resolution": "stream_by_source_table_id",
            }
        },
        "artifacts": {
            "source_tables": source_writer.manifest(output_dir),
            "query_tables": query_writer.manifest(output_dir),
            "data_lake_tables": data_lake_writer.manifest(output_dir),
            "entities": entities_writer.manifest(output_dir),
            "bridge_assets": bridge_assets_writer.manifest(output_dir),
            "table_asset_links": table_asset_links_writer.manifest(output_dir),
            "attribute_extractions": extraction_writer.manifest(output_dir),
            "evidence_recoveries": recovery_writer.manifest(output_dir),
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "stats": "stats.json",
            "table_queryability_decisions": "table_queryability_decisions.jsonl",
            "web_fetch_failures": "web_fetch_failures.jsonl",
            "media_download_failures": "media_download_failures.jsonl",
            "model_attribute_errors": str(model_error_file.relative_to(output_dir))
            if model_error_file.is_relative_to(output_dir)
            else str(model_error_file),
        },
        "query_construction": {
            "query_rows_per_table": args.query_rows_per_table,
            "query_row_selection": "recovery_balanced_disjoint_train_views",
            "max_train_query_row_views_per_join": (
                args.max_train_query_row_views_per_join
            ),
            "evaluation_query_row_views_per_join": 1,
            "explicit_join_fallback_mode": args.explicit_join_fallback_mode,
            "explicit_join_fallback_ratio": args.explicit_join_fallback_ratio,
            "explicit_join_column_policy": "seeded_random_non_entity_visible_column",
            "target_row_scope": "all_source_rows",
            "min_rows_per_output_table": args.min_rows_per_output_table,
            "min_recovered_value_ratio": args.min_recovered_value_ratio,
            "min_recovery_denominator": args.min_recovery_denominator,
            "max_query_tables_per_source_table": args.max_query_tables_per_source_table,
            "max_query_context_attrs": args.max_query_context_attrs,
            "max_target_context_attrs": args.max_target_context_attrs,
            "context_attr_limit_policy": "compatibility_flags_ignored",
            "context_partition_policy": (
                "source_level_seeded_gaussian_target_ratio_mean_0.5_"
                "std_0.1_clipped_0.3_0.7"
            ),
            "explicit_context_partition_scope": (
                "post_balance_selected_join_columns_only"
            ),
            "qualified_attribute_policy": "all_safe_variants",
            "sibling_source_column_policy": (
                "qualified_bridge_columns_excluded_from_shared_context_pools"
            ),
            "identical_visible_query_policy": (
                "merge_exact_row_view_with_all_distinct_positive_targets"
            ),
            "target_column_order_policy": (
                "independently_seeded_shuffle_per_join_column"
            ),
        },
        "model_endpoints": {
            "model_endpoint_config": args.model_endpoint_config,
            "text_model_base_url": args.text_model_base_url,
            "text_model_base_urls": args.text_model_base_urls,
            "text_model_base_urls_file": args.text_model_base_urls_file,
            "remote_text_model_base_url": args.remote_text_model_base_url,
            "remote_text_model_base_urls": args.remote_text_model_base_urls,
            "remote_text_model_base_urls_file": (
                args.remote_text_model_base_urls_file
            ),
            "text_model_name": args.text_model_name,
            "image_model_base_url": args.image_model_base_url,
            "image_model_base_urls": args.image_model_base_urls,
            "image_model_base_urls_file": args.image_model_base_urls_file,
            "remote_image_model_base_url": args.remote_image_model_base_url,
            "remote_image_model_base_urls": args.remote_image_model_base_urls,
            "remote_image_model_base_urls_file": (
                args.remote_image_model_base_urls_file
            ),
            "image_model_name": args.image_model_name,
            "prompt_version": join_builder.PROMPT_VERSION,
            "precompute_model_cache": args.precompute_model_cache,
            "precompute_text_model_cache": args.precompute_text_model_cache,
            "model_start_marker": args.model_start_marker,
            "model_ready_marker": args.model_ready_marker,
            "model_text_done_marker": args.model_text_done_marker,
            "model_image_done_marker": args.model_image_done_marker,
            "configured_text_model_workers": args.text_model_workers,
            "configured_image_model_workers": args.image_model_workers,
            "configured_remote_text_model_workers": (
                args.remote_text_model_workers
            ),
            "configured_remote_image_model_workers": (
                args.remote_image_model_workers
            ),
            "final_model_concurrency": stats["model_concurrency"],
            "inference_stats": stats["model_inference"],
            "auto_check": stats["model_auto_check"],
        },
        "web_cache": {
            "cache_dir": str(cache_dir),
            "database": str(cache_dir / "wdc_web.sqlite3"),
            "image_dir": str(cache_dir / "wdc_images"),
            "workers": stats["web_workers"],
            "max_in_flight": stats["web_max_in_flight"],
            "fairness_lookahead": stats["web_fairness_lookahead"],
            "always_fetch_page_url": True,
            "direct_images_have_priority": True,
            "shared_max_images_per_entity": args.max_images_per_entity,
            "max_image_pixels": args.web_max_image_pixels,
            "max_total_image_bytes": args.web_max_total_image_bytes,
            "max_total_cache_bytes": args.web_max_total_cache_bytes,
            "max_response_seconds": args.web_max_response_seconds,
            "min_free_disk_bytes": args.min_free_disk_bytes,
        },
        "cache": {
            "root_dir": str(cache_dir),
            "model_attribute_extractions": str(model_cache_path),
            "query_recovery_auto_checks": str(query_auto_check_cache_path),
        },
        "note": "Read only shards listed here; a reused output directory may contain stale unlisted files.",
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multimodal joinability dataset from WDC Schema.org 2023 "
            "gzip host tables and generic webpage assets."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--max_source_tables", type=int, default=100)
    parser.add_argument(
        "--max_tables",
        type=int,
        dest="max_source_tables",
        default=argparse.SUPPRESS,
        help="Deprecated alias for --max_source_tables.",
    )
    parser.add_argument(
        "--max_rows_per_source_table",
        type=int,
        default=100,
        help="Maximum retained valid rows per WDC host table; 0 requires --allow_unbounded.",
    )
    parser.add_argument(
        "--max_scanned_files",
        type=int,
        default=1000,
        help="Maximum gzip files attempted, including malformed or rejected files.",
    )
    parser.add_argument(
        "--allow_unbounded",
        action="store_true",
        help=(
            "Explicitly allow non-positive scan/source-table caps or a zero "
            "per-table row cap."
        ),
    )
    parser.add_argument(
        "--min_free_disk_bytes",
        type=int,
        default=1_000_000_000,
        help="Abort before building when output/cache storage has less free space.",
    )
    parser.add_argument("--max_entities", type=int, default=None)
    parser.add_argument("--min_rows", type=int, default=2)
    parser.add_argument("--min_cols", type=int, default=2)
    parser.add_argument("--records_per_shard", type=int, default=50000)
    parser.add_argument("--flush_every_records", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--max_images_per_entity", type=int, default=2)
    parser.add_argument("--text_asset_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_asset_chunk_chars", type=int, default=120)
    parser.add_argument("--max_text_asset_chunks_per_entity", type=int, default=3)
    parser.add_argument("--web_workers", type=int, default=4)
    parser.add_argument(
        "--web_max_in_flight",
        type=int,
        default=0,
        help="Maximum submitted web futures per bounded batch; 0 uses 2*web_workers.",
    )
    parser.add_argument(
        "--web_fairness_lookahead",
        type=int,
        default=256,
        help="Bounded entity lookahead used to interleave different page hosts.",
    )
    parser.add_argument(
        "--web_user_agent",
        default="MMDD-WDC-DatasetBuilder/0.1 (research dataset construction)",
    )
    parser.add_argument("--web_connect_timeout", type=float, default=10.0)
    parser.add_argument("--web_read_timeout", type=float, default=30.0)
    parser.add_argument("--web_max_retries", type=int, default=2)
    parser.add_argument("--web_retry_base_seconds", type=float, default=0.25)
    parser.add_argument("--web_max_page_bytes", type=int, default=2_000_000)
    parser.add_argument("--web_max_image_bytes", type=int, default=10_000_000)
    parser.add_argument("--web_max_image_pixels", type=int, default=25_000_000)
    parser.add_argument(
        "--web_max_total_image_bytes", type=int, default=100_000_000_000
    )
    parser.add_argument(
        "--web_max_total_cache_bytes", type=int, default=120_000_000_000
    )
    parser.add_argument("--web_max_response_seconds", type=float, default=120.0)
    parser.add_argument("--web_max_redirects", type=int, default=3)
    parser.add_argument("--web_min_image_side", type=int, default=32)
    parser.add_argument("--web_max_image_aspect_ratio", type=float, default=20.0)
    parser.add_argument("--web_host_delay", type=float, default=0.5)

    parser.add_argument(
        "--split_by", choices=["source_table_id", "page_title"], default="page_title"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_column_non_empty_ratio", type=float, default=0.5)
    parser.add_argument("--min_recovered_value_ratio", type=float, default=0.6)
    parser.add_argument("--min_recovery_denominator", type=int, default=2)
    parser.add_argument("--min_rows_per_output_table", type=int, default=2)
    parser.add_argument("--query_rows_per_table", type=int, default=5)
    parser.add_argument(
        "--max_train_query_row_views_per_join",
        type=int,
        default=5,
        help=(
            "Maximum deterministic disjoint query row views per train join chain; "
            "0 means use every feasible view. Dev/test always use one."
        ),
    )
    parser.add_argument("--max_query_tables_per_source_table", type=int, default=0)
    parser.add_argument(
        "--max_query_context_attrs",
        type=int,
        default=1,
        help="Deprecated compatibility option; all query-pool columns are emitted.",
    )
    parser.add_argument(
        "--max_target_context_attrs",
        type=int,
        default=2,
        help="Deprecated compatibility option; all target-pool columns are emitted.",
    )
    parser.add_argument(
        "--explicit_join_fallback_mode",
        choices=join_builder.EXPLICIT_JOIN_FALLBACK_MODES,
        default=join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE,
        help=(
            "match_implicit selects exactly one visible-join query per "
            "implicit query within each split; ratio retains legacy sampling."
        ),
    )
    parser.add_argument(
        "--explicit_join_fallback_ratio",
        type=float,
        default=join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO,
        help=(
            "Seeded fraction of tables rejected by multimodal recovery to turn "
            "into ordinary joins with a visible non-entity join column in both "
            "query and target; 0 disables the fallback."
        ),
    )

    parser.add_argument(
        "--model_endpoint_config",
        default=None,
        help=(
            "Optional mmdd-model-endpoints-v1 JSON config shared with the "
            "EntiTables builder."
        ),
    )
    parser.add_argument("--text_model_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--text_model_base_urls", nargs="*", default=None)
    parser.add_argument("--text_model_base_urls_file", default=None)
    parser.add_argument("--remote_text_model_base_url", default=None)
    parser.add_argument("--remote_text_model_base_urls", nargs="*", default=None)
    parser.add_argument("--remote_text_model_base_urls_file", default=None)
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_model_api_key", default=None)
    parser.add_argument("--remote_text_model_api_key", default=None)
    parser.add_argument("--image_model_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--image_model_base_urls", nargs="*", default=None)
    parser.add_argument("--image_model_base_urls_file", default=None)
    parser.add_argument("--remote_image_model_base_url", default=None)
    parser.add_argument("--remote_image_model_base_urls", nargs="*", default=None)
    parser.add_argument("--remote_image_model_base_urls_file", default=None)
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--image_model_api_key", default=None)
    parser.add_argument("--remote_image_model_api_key", default=None)
    parser.add_argument("--model_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--model_temperature", type=float, default=0.0)
    parser.add_argument("--model_max_tokens", type=int, default=1024)
    parser.add_argument(
        "--image_model_max_tokens",
        type=int,
        default=join_builder.DEFAULT_IMAGE_MODEL_MAX_TOKENS,
    )
    parser.add_argument(
        "--image_request_max_pixels",
        type=int,
        default=join_builder.DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
    )
    parser.add_argument("--text_model_workers", type=int, default=1)
    parser.add_argument("--image_model_workers", type=int, default=1)
    parser.add_argument("--remote_text_model_workers", type=int, default=0)
    parser.add_argument("--remote_image_model_workers", type=int, default=0)
    parser.add_argument(
        "--enable_thinking",
        dest="disable_thinking",
        action="store_false",
        help=(
            "Deprecated and rejected: all text and image requests disable "
            "thinking."
        ),
    )
    parser.add_argument(
        "--no_reparse_cached_model_outputs",
        dest="reparse_cached_model_outputs",
        action="store_false",
    )
    parser.add_argument("--refresh_invalid_model_cache", action="store_true")
    parser.add_argument("--model_max_retries", type=int, default=2)
    parser.add_argument("--model_retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--no_reuse_model_cache", action="store_true")
    parser.add_argument("--cache_failed_model_outputs", action="store_true")
    join_builder.add_model_auto_check_arguments(parser)
    parser.add_argument("--model_attribute_errors_path", default="")
    parser.add_argument(
        "--context_retry_image_max_pixels",
        type=int,
        default=join_builder.DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS,
    )
    parser.add_argument(
        "--no_model_progress", dest="model_progress", action="store_false"
    )
    parser.add_argument("--precompute_model_cache", action="store_true")
    parser.add_argument("--precompute_text_model_cache", action="store_true")
    parser.add_argument("--model_start_marker", default=None)
    parser.add_argument("--model_ready_marker", default=None)
    parser.add_argument("--model_ready_timeout_seconds", type=float, default=None)
    parser.add_argument("--model_text_done_marker", default=None)
    parser.add_argument("--model_image_done_marker", default=None)
    parser.add_argument("--model_round_control_dir", default=None)
    parser.add_argument("--model_round_run_id", default=None)
    parser.add_argument("--run_fingerprint", default="")
    parser.set_defaults(
        disable_thinking=True,
        reparse_cached_model_outputs=True,
        model_progress=True,
    )
    args = parser.parse_args(argv)
    if not args.disable_thinking:
        parser.error(
            "thinking mode cannot be enabled for dataset construction"
        )
    if min(
        args.text_model_workers,
        args.image_model_workers,
        args.remote_text_model_workers,
        args.remote_image_model_workers,
    ) < 0:
        parser.error("model worker counts must be non-negative")
    if args.max_rows_per_source_table < 0:
        parser.error("--max_rows_per_source_table must be >= 0")
    if not args.allow_unbounded and (
        args.max_source_tables <= 0
        or args.max_scanned_files <= 0
        or args.max_rows_per_source_table == 0
    ):
        parser.error(
            "non-positive --max_source_tables/--max_scanned_files or zero "
            "--max_rows_per_source_table requires --allow_unbounded"
        )
    return args


def main() -> None:
    setup_logging()
    stats = build_dataset(parse_args())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

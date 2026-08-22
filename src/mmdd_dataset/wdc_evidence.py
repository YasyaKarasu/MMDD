from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import socket
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from .utils import clean_text, stable_hash
from .wdc_runtime import bounded_map, stable_digest


class _PageParser(HTMLParser):
    TEXT_TAGS = {"title", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    SKIP_TAGS = {"script", "style", "svg", "template", "noscript"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.text_parts: list[str] = []
        self.image_urls: list[str] = []
        self._text_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {name.casefold(): value or "" for name, value in attrs}
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.TEXT_TAGS:
            self._text_depth += 1
        if tag == "img":
            source = clean_text(
                attributes.get("src")
                or attributes.get("data-src")
                or attributes.get("data-original")
            )
            if source:
                self.image_urls.append(urljoin(self.base_url, source))
        if tag == "meta":
            name = clean_text(
                attributes.get("property") or attributes.get("name")
            ).casefold()
            content = clean_text(attributes.get("content"))
            if content and name in {
                "description",
                "og:description",
                "twitter:description",
            }:
                self.text_parts.append(content)
            if content and name in {"og:image", "twitter:image"}:
                self.image_urls.append(urljoin(self.base_url, content))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self.SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
        elif not self._skip_depth and tag in self.TEXT_TAGS and self._text_depth:
            self._text_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._text_depth:
            value = clean_text(data)
            if value:
                self.text_parts.append(value)


def normalize_public_url(
    value: Any,
    *,
    resolve: bool = False,
    resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str | None:
    """Return a canonical public HTTP(S) URL, rejecting local destinations."""
    text = clean_text(value)
    if not text or any(character.isspace() for character in text):
        return None
    try:
        parsed = urlsplit(text)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        hostname = (
            parsed.hostname.encode("idna").decode("ascii").casefold().rstrip(".")
        )
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
        return None

    addresses: list[str] = []
    try:
        addresses.append(str(ipaddress.ip_address(hostname)))
    except ValueError:
        if resolve:
            try:
                addresses.extend(
                    str(item[4][0])
                    for item in resolver(
                        hostname,
                        port or (80 if parsed.scheme.casefold() == "http" else 443),
                        type=socket.SOCK_STREAM,
                    )
                )
            except (OSError, TypeError):
                return None
    if addresses:
        try:
            if any(
                not ipaddress.ip_address(address).is_global for address in addresses
            ):
                return None
        except ValueError:
            return None

    displayed_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = 80 if parsed.scheme.casefold() == "http" else 443
    netloc = displayed_host if port in {None, default_port} else f"{displayed_host}:{port}"
    return urlunsplit(
        (parsed.scheme.casefold(), netloc, parsed.path or "/", parsed.query, "")
    )


class ResultCache:
    """Small durable cache shared by evidence and model task runners."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS results (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (namespace, cache_key)
                ) WITHOUT ROWID
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def get(self, namespace: str, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM results WHERE namespace = ? AND cache_key = ?",
                (namespace, cache_key),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None

    def put(self, namespace: str, cache_key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO results(namespace, cache_key, payload) VALUES (?, ?, ?)",
                (namespace, cache_key, payload),
            )

    def count(self, namespace: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM results WHERE namespace = ?", (namespace,)
                ).fetchone()[0]
            )


class EvidenceClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        user_agent: str,
        timeout: float,
        max_page_bytes: int,
        max_image_bytes: int,
        max_redirects: int = 3,
    ) -> None:
        self.cache_dir = cache_dir
        self.image_dir = cache_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.timeout = timeout
        self.max_page_bytes = max_page_bytes
        self.max_image_bytes = max_image_bytes
        self.max_redirects = max_redirects
        self.policy = stable_digest(
            "wdc_public_http_v1", max_page_bytes, max_image_bytes, max_redirects
        )

    def _get(self, original_url: str, *, stream: bool) -> requests.Response:
        current = original_url
        for _ in range(self.max_redirects + 1):
            safe_url = normalize_public_url(current, resolve=True)
            if safe_url is None:
                raise ValueError("unsafe_url")
            response = self.session.get(
                safe_url,
                timeout=self.timeout,
                allow_redirects=False,
                stream=stream,
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("redirect_without_location")
                current = urljoin(safe_url, location)
                continue
            response.raise_for_status()
            return response
        raise ValueError("too_many_redirects")

    def fetch_page(self, url: str) -> dict[str, Any]:
        response = self._get(url, stream=True)
        try:
            content = bytearray()
            for chunk in response.iter_content(64 * 1024):
                content.extend(chunk)
                if len(content) > self.max_page_bytes:
                    raise ValueError("page_too_large")
            encoding = response.encoding or "utf-8"
            html = bytes(content).decode(encoding, errors="replace")
            parser = _PageParser(response.url)
            parser.feed(html)
            image_urls = []
            for candidate in parser.image_urls:
                normalized = normalize_public_url(candidate)
                if normalized and normalized not in image_urls:
                    image_urls.append(normalized)
            return {
                "status": "success",
                "url": normalize_public_url(response.url) or response.url,
                "text": clean_text(" ".join(parser.text_parts)),
                "image_urls": image_urls,
                "bytes": len(content),
            }
        finally:
            response.close()

    def fetch_image(self, url: str) -> dict[str, Any]:
        response = self._get(url, stream=True)
        try:
            content_type = (
                response.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .casefold()
            )
            if not content_type.startswith("image/"):
                raise ValueError("not_an_image")
            suffix = mimetypes.guess_extension(content_type) or ".img"
            asset_hash = stable_hash("image", response.url, length=40)
            path = self.image_dir / f"{asset_hash}{suffix[:8]}"
            temporary = path.with_name(path.name + ".tmp")
            total = 0
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(64 * 1024):
                    total += len(chunk)
                    if total > self.max_image_bytes:
                        raise ValueError("image_too_large")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return {
                "status": "success",
                "url": normalize_public_url(response.url) or response.url,
                "cache_path": str(path),
                "content_type": content_type,
                "bytes": total,
            }
        finally:
            response.close()


def execute_tasks(
    tasks: Iterable[dict[str, Any]],
    *,
    kind: str,
    cache: ResultCache,
    policy: str,
    workers: int,
    fetch: Callable[[str], dict[str, Any]],
    reuse: Callable[[str, str], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Execute one bounded shard, fetching each URL at most once per policy."""
    task_records = list(tasks)
    unique_urls = list(dict.fromkeys(str(task["url"]) for task in task_records))
    namespace = f"{kind}:{policy}"

    def get_or_fetch(url: str) -> tuple[str, dict[str, Any]]:
        key = stable_hash(url, length=40)
        cached = cache.get(namespace, key)
        if cached is not None:
            return url, cached
        reused = reuse(kind, url) if reuse is not None else None
        if reused is not None:
            cache.put(namespace, key, reused)
            return url, reused
        try:
            result = fetch(url)
        except Exception as error:
            result = {
                "status": "terminal",
                "url": url,
                "error": f"{type(error).__name__}: {clean_text(error)}",
            }
        cache.put(namespace, key, result)
        return url, result

    outcomes = dict(
        bounded_map(get_or_fetch, unique_urls, workers=workers, max_pending=workers * 2)
    )
    return [
        {
            **task,
            **outcomes[str(task["url"])],
            "task_id": task["task_id"],
            "entity_id": task["entity_id"],
        }
        for task in task_records
    ]

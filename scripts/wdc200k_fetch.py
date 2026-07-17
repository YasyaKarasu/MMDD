"""Bounded, resumable fetching of globally unique WDC page URLs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

import fcntl

try:
    from stage1_io import stable_hash
    from wdc200k_io import (
        CompletedShard,
        Job,
        SqliteJobStore,
        validate_completed_shard,
    )
except ModuleNotFoundError as error:
    if error.name not in {"stage1_io", "wdc200k_io"}:
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        from stage1_io import stable_hash
        from wdc200k_io import (
            CompletedShard,
            Job,
            SqliteJobStore,
            validate_completed_shard,
        )
    finally:
        sys.path.remove(scripts_directory)


FETCH_SCHEMA_VERSION = "wdc200k-page-fetch-v1"


@dataclass(frozen=True)
class FetchPolicy:
    """Network and scheduler policy defining one reusable outcome namespace."""

    retries: int = 0
    deadline_seconds: float = 8.0
    global_concurrency: int = 128
    per_host_concurrency: int = 2
    policy_version: str = FETCH_SCHEMA_VERSION
    network_policy_fingerprint: str = "wdc-web-v1"

    def __post_init__(self) -> None:
        if self.retries != 0:
            raise ValueError("WDC page fetching permits exactly zero retries")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        if self.global_concurrency <= 0:
            raise ValueError("global_concurrency must be positive")
        if self.per_host_concurrency <= 0:
            raise ValueError("per_host_concurrency must be positive")
        if not self.policy_version:
            raise ValueError("policy_version must not be empty")
        if not self.network_policy_fingerprint:
            raise ValueError("network_policy_fingerprint must not be empty")

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            self.policy_version,
            self.network_policy_fingerprint,
            self.retries,
            self.deadline_seconds,
            self.global_concurrency,
            self.per_host_concurrency,
            length=40,
        )


@dataclass(frozen=True)
class FetchResult:
    unique: int
    success: int
    terminal: int
    inflight: int
    leased: int
    remaining: int
    complete: bool
    maximum_inflight: int
    maximum_claimed: int
    maximum_host_limiters: int
    outcomes_path: Path
    failure_path: Path
    progress_path: Path
    policy_fingerprint: str


class PageOutcomeStore:
    """Disk-backed outcome and reference mappings keyed by policy and URL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
            connection.execute("BEGIN IMMEDIATE")
            counts_existed = (
                connection.execute(
                    """
                    SELECT 1 FROM sqlite_master
                    WHERE type = 'table' AND name = 'policy_outcome_counts'
                    """
                ).fetchone()
                is not None
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS page_outcomes (
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
                CREATE TABLE IF NOT EXISTS page_references (
                    policy_fingerprint TEXT NOT NULL,
                    url_key TEXT NOT NULL,
                    reference_key TEXT NOT NULL,
                    entity_id TEXT NOT NULL DEFAULT '',
                    source_table_id TEXT NOT NULL DEFAULT '',
                    row_id_json TEXT NOT NULL DEFAULT 'null',
                    PRIMARY KEY (
                        policy_fingerprint, url_key, reference_key
                    )
                )
                """
            )
            reference_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(page_references)"
                )
            }
            for column, declaration in (
                ("entity_id", "TEXT NOT NULL DEFAULT ''"),
                ("source_table_id", "TEXT NOT NULL DEFAULT ''"),
                ("row_id_json", "TEXT NOT NULL DEFAULT 'null'"),
            ):
                if column not in reference_columns:
                    connection.execute(
                        f"ALTER TABLE page_references "
                        f"ADD COLUMN {column} {declaration}"
                    )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS page_host_sequences (
                    policy_fingerprint TEXT NOT NULL,
                    host TEXT NOT NULL,
                    next_ordinal INTEGER NOT NULL,
                    PRIMARY KEY (policy_fingerprint, host)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_outcome_counts (
                    policy_fingerprint TEXT PRIMARY KEY,
                    success INTEGER NOT NULL,
                    terminal INTEGER NOT NULL,
                    total INTEGER NOT NULL
                )
                """
            )
            if not counts_existed:
                connection.execute(
                    """
                    INSERT INTO policy_outcome_counts (
                        policy_fingerprint, success, terminal, total
                    )
                    SELECT
                        policy_fingerprint,
                        SUM(
                            CASE WHEN status = 'success' THEN 1 ELSE 0 END
                        ),
                        SUM(
                            CASE WHEN status = 'terminal' THEN 1 ELSE 0 END
                        ),
                        COUNT(*)
                    FROM page_outcomes
                    GROUP BY policy_fingerprint
                    """
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def get(
        self,
        policy_fingerprint: str,
        url_key: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            return self._get_with_connection(
                connection,
                policy_fingerprint,
                url_key,
            )

    @classmethod
    def _get_with_connection(
        cls,
        connection: sqlite3.Connection,
        policy_fingerprint: str,
        url_key: str,
    ) -> dict[str, Any] | None:
        row = connection.execute(
                """
                SELECT *
                FROM page_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
        return None if row is None else cls._decode(row)

    def put(
        self,
        policy_fingerprint: str,
        url_key: str,
        page_url: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        status = str(outcome["status"])
        if status not in {"success", "terminal"}:
            raise ValueError(f"non-terminal page outcome: {status}")
        canonical = {
            "policy_fingerprint": policy_fingerprint,
            "url_key": url_key,
            "page_url": page_url,
            "status": status,
            "final_url": (
                str(outcome.get("final_url") or page_url)
                if status == "success"
                else None
            ),
            "text": str(outcome.get("text") or "") if status == "success" else None,
            "image_urls": (
                [
                    str(value)
                    for value in outcome.get("image_urls", [])
                    if isinstance(value, str)
                ]
                if status == "success"
                else []
            ),
            "error_class": (
                _safe_error_class(
                    outcome.get("error_class") or "fetch_failed"
                )
                if status == "terminal"
                else None
            ),
            "http_status": (
                _optional_int(outcome.get("http_status"))
                if status == "terminal"
                else None
            ),
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_sha256 = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT payload_sha256
                FROM page_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if existing is not None:
                if str(existing["payload_sha256"]) != payload_sha256:
                    raise ValueError(
                        "conflicting immutable page outcome"
                    )
                persisted = self._get_with_connection(
                    connection,
                    policy_fingerprint,
                    url_key,
                )
                if persisted is None:
                    raise RuntimeError("page outcome disappeared")
                return persisted
            connection.execute(
                """
                INSERT INTO page_outcomes (
                    policy_fingerprint, url_key, page_url, status,
                    final_url, text, image_urls_json, error_class,
                    http_status, payload_sha256, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy_fingerprint,
                    url_key,
                    page_url,
                    status,
                    canonical["final_url"],
                    canonical["text"],
                    json.dumps(canonical["image_urls"], ensure_ascii=False),
                    canonical["error_class"],
                    canonical["http_status"],
                    payload_sha256,
                    time.time(),
                ),
            )
            success_increment = 1 if status == "success" else 0
            terminal_increment = 1 if status == "terminal" else 0
            connection.execute(
                """
                INSERT INTO policy_outcome_counts (
                    policy_fingerprint, success, terminal, total
                ) VALUES (?, ?, ?, 1)
                ON CONFLICT(policy_fingerprint) DO UPDATE SET
                    success = success + excluded.success,
                    terminal = terminal + excluded.terminal,
                    total = total + 1
                """,
                (
                    policy_fingerprint,
                    success_increment,
                    terminal_increment,
                ),
            )
        persisted = self.get(policy_fingerprint, url_key)
        if persisted is None:
            raise RuntimeError("page outcome disappeared after durable write")
        return persisted

    def iter(
        self,
        policy_fingerprint: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        connection = self._connect()
        try:
            if policy_fingerprint is None:
                cursor = connection.execute(
                    """
                    SELECT outcomes.*, (
                        SELECT COUNT(*)
                        FROM page_references AS refs
                        WHERE refs.policy_fingerprint =
                              outcomes.policy_fingerprint
                          AND refs.url_key = outcomes.url_key
                    ) AS reference_count
                    FROM page_outcomes AS outcomes
                    ORDER BY policy_fingerprint, url_key
                    """
                )
            else:
                cursor = connection.execute(
                    """
                    SELECT outcomes.*, (
                        SELECT COUNT(*)
                        FROM page_references AS refs
                        WHERE refs.policy_fingerprint =
                              outcomes.policy_fingerprint
                          AND refs.url_key = outcomes.url_key
                    ) AS reference_count
                    FROM page_outcomes AS outcomes
                    WHERE outcomes.policy_fingerprint = ?
                    ORDER BY url_key
                    """,
                    (policy_fingerprint,),
                )
            for row in cursor:
                yield self._decode(row)
        finally:
            connection.close()

    def counts(self, policy_fingerprint: str) -> tuple[int, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT success, terminal
                FROM policy_outcome_counts
                WHERE policy_fingerprint = ?
                """,
                (policy_fingerprint,),
            ).fetchone()
        if row is None:
            return 0, 0
        return int(row["success"]), int(row["terminal"])

    def rebuild_counts(self) -> None:
        """Explicitly rebuild counters for migration or integrity recovery."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM policy_outcome_counts")
            connection.execute(
                """
                INSERT INTO policy_outcome_counts (
                    policy_fingerprint, success, terminal, total
                )
                SELECT
                    policy_fingerprint,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'terminal' THEN 1 ELSE 0 END),
                    COUNT(*)
                FROM page_outcomes
                GROUP BY policy_fingerprint
                """
            )

    def reference_count(
        self,
        policy_fingerprint: str,
        url_key: str,
    ) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM page_references
                    WHERE policy_fingerprint = ? AND url_key = ?
                    """,
                    (policy_fingerprint, url_key),
                ).fetchone()[0]
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        image_urls = json.loads(row["image_urls_json"] or "[]")
        return {
            "policy_fingerprint": str(row["policy_fingerprint"]),
            "url_key": str(row["url_key"]),
            "page_url": str(row["page_url"]),
            "status": str(row["status"]),
            "final_url": (
                None if row["final_url"] is None else str(row["final_url"])
            ),
            "text": None if row["text"] is None else str(row["text"]),
            "image_urls": image_urls if isinstance(image_urls, list) else [],
            "error_class": (
                None
                if row["error_class"] is None
                else str(row["error_class"])
            ),
            "http_status": _optional_int(row["http_status"]),
            "payload_sha256": str(row["payload_sha256"]),
            "affected_reference_count": (
                int(row["reference_count"])
                if "reference_count" in row.keys()
                else 0
            ),
        }


@dataclass
class _HostState:
    semaphore: threading.BoundedSemaphore
    jobs: deque[Job]
    active: int = 0


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _host_for_url(url: str) -> str:
    try:
        host = urlsplit(url).hostname
    except ValueError as error:
        raise ValueError("invalid page URL") from error
    if not host:
        raise ValueError("page URL has no hostname")
    try:
        return host.encode("idna").decode("ascii").casefold().rstrip(".")
    except UnicodeError as error:
        raise ValueError("invalid page hostname") from error


def _validated_ref(record: dict[str, Any]) -> tuple[str, str, str]:
    page_url = str(record.get("page_url") or "")
    parsed = urlsplit(page_url)
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("page reference is not a safe HTTP(S) URL")
    host = _host_for_url(page_url)
    url_key = str(record.get("url_key") or "")
    expected_key = hashlib.sha256(page_url.encode("utf-8")).hexdigest()
    if url_key != expected_key:
        raise ValueError("page reference url_key does not match page_url")
    return url_key, page_url, host


def _job_kind(policy_fingerprint: str) -> str:
    return f"wdc200k-page:{policy_fingerprint}"


def _job_count(store: SqliteJobStore, kind: str) -> int:
    with store._connect() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = ?",
                (kind,),
            ).fetchone()[0]
        )


def _leased_count(
    store: SqliteJobStore,
    kind: str,
) -> int:
    with store._connect() as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM jobs
                WHERE kind = ? AND status = 'leased'
                """,
                (kind,),
            ).fetchone()[0]
        )


def _safe_error_class(value: Any) -> str:
    candidate = str(value or "")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,99}", candidate):
        return candidate
    return "fetch_failed"


def _transport_cached_outcome(
    transport: Any,
    page_url: str,
    policy: FetchPolicy,
) -> dict[str, Any] | None:
    getter = getattr(transport, "cached_page_outcome", None)
    if not callable(getter):
        return None
    cached = getter(page_url)
    if not isinstance(cached, dict):
        return None
    cached_policy = cached.get("policy_fingerprint")
    if cached_policy != policy.network_policy_fingerprint:
        return None
    status = str(cached.get("status") or "")
    if status == "success":
        return {
            "status": "success",
            "final_url": cached.get("final_url") or page_url,
            "text": cached.get("text") or "",
            "image_urls": cached.get("image_urls") or [],
        }
    if status in {"terminal", "retryable"}:
        return {
            "status": "terminal",
            "error_class": _safe_error_class(
                cached.get("error_class") or "cached_failure"
            ),
            "http_status": cached.get("http_status"),
        }
    return None


def _fetch_one(
    job: Job,
    *,
    transport: Any,
    policy: FetchPolicy,
) -> dict[str, Any]:
    page_url = str(job.payload["page_url"])
    cached = _transport_cached_outcome(transport, page_url, policy)
    if cached is not None:
        return cached
    try:
        payload = transport.fetch_page(
            page_url,
            deadline_seconds=policy.deadline_seconds,
            max_retries=policy.retries,
        )
    except Exception as error:
        return {
            "status": "terminal",
            "error_class": type(error).__name__,
            "http_status": _optional_int(getattr(error, "status_code", None)),
        }
    if isinstance(payload, dict):
        return {
            "status": "success",
            "final_url": payload.get("final_url") or page_url,
            "text": payload.get("text") or "",
            "image_urls": payload.get("image_urls") or [],
        }
    cached = getattr(transport, "cached_page_outcome", lambda _url: None)(
        page_url
    )
    return {
        "status": "terminal",
        "error_class": _safe_error_class(
            (
                cached.get("error_class")
                if isinstance(cached, dict)
                else None
            )
            or "fetch_failed"
        ),
        "http_status": (
            cached.get("http_status") if isinstance(cached, dict) else None
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def _snapshot_guard(progress_path: Path) -> Iterator[None]:
    lock_path = progress_path.with_name(
        f".{progress_path.name}.snapshot.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _atomic_failure_snapshot(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sanitized_failure(
    outcome: dict[str, Any],
    reference_count: int,
) -> dict[str, Any]:
    parsed = urlsplit(str(outcome["page_url"]))
    sanitized_url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, "", "")
    )
    return {
        "failure_type": "web_fetch_failure",
        "stage": "page_fetch",
        "status": "terminal",
        "page_url": sanitized_url,
        "url_key": outcome["url_key"],
        "error_class": outcome["error_class"],
        "http_status": outcome["http_status"],
        "affected_reference_count": reference_count,
        "policy_fingerprint": outcome["policy_fingerprint"],
    }


def _publish_snapshots(
    outcome_store: PageOutcomeStore,
    *,
    store: SqliteJobStore,
    kind: str,
    policy_fingerprint: str,
    unique: int,
    failure_path: Path,
    progress_path: Path,
    inflight: int,
) -> tuple[int, int]:
    with _snapshot_guard(progress_path):
        _atomic_failure_snapshot(
            failure_path,
            (
                _sanitized_failure(
                    outcome,
                    int(outcome["affected_reference_count"]),
                )
                for outcome in outcome_store.iter(policy_fingerprint)
                if outcome["status"] == "terminal"
            ),
        )
        return _publish_progress_unlocked(
            outcome_store,
            store=store,
            kind=kind,
            policy_fingerprint=policy_fingerprint,
            unique=unique,
            progress_path=progress_path,
            inflight=inflight,
        )


def _publish_progress(
    outcome_store: PageOutcomeStore,
    *,
    store: SqliteJobStore,
    kind: str,
    policy_fingerprint: str,
    unique: int,
    progress_path: Path,
    inflight: int,
) -> tuple[int, int]:
    with _snapshot_guard(progress_path):
        return _publish_progress_unlocked(
            outcome_store,
            store=store,
            kind=kind,
            policy_fingerprint=policy_fingerprint,
            unique=unique,
            progress_path=progress_path,
            inflight=inflight,
        )


def _publish_progress_unlocked(
    outcome_store: PageOutcomeStore,
    *,
    store: SqliteJobStore,
    kind: str,
    policy_fingerprint: str,
    unique: int,
    progress_path: Path,
    inflight: int,
) -> tuple[int, int]:
    success, terminal = outcome_store.counts(policy_fingerprint)
    leased = _leased_count(store, kind)
    remaining = max(0, unique - success - terminal)
    observed_inflight = max(inflight, leased)
    _atomic_json(
        progress_path,
        {
            "policy_fingerprint": policy_fingerprint,
            "unique": unique,
            "success": success,
            "terminal": terminal,
            "pending": max(0, remaining - leased),
            "leased": leased,
            "remaining": remaining,
            "complete": success + terminal == unique,
            "inflight": observed_inflight,
            "updated_at": time.time(),
        },
    )
    return success, terminal


def _enqueue_page_refs(
    page_refs: Iterable[dict[str, Any]],
    *,
    store: SqliteJobStore,
    outcome_store: PageOutcomeStore,
    policy_fingerprint: str,
    kind: str,
    commit_every: int = 10_000,
) -> None:
    """Stream refs into two idempotent SQLite indexes in bounded transactions."""
    job_connection = store._connect()
    reference_connection = outcome_store._connect()
    pending = 0
    try:
        for record in page_refs:
            url_key, page_url, host = _validated_ref(record)
            reference_key = stable_hash(
                record.get("entity_id", ""),
                record.get("source_table_id", ""),
                record.get("row_id", ""),
                length=40,
            )
            reference_connection.execute(
                """
                INSERT OR IGNORE INTO page_references (
                    policy_fingerprint, url_key, reference_key,
                    entity_id, source_table_id, row_id_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(
                    policy_fingerprint, url_key, reference_key
                ) DO UPDATE SET
                    entity_id = excluded.entity_id,
                    source_table_id = excluded.source_table_id,
                    row_id_json = excluded.row_id_json
                """,
                (
                    policy_fingerprint,
                    url_key,
                    reference_key,
                    str(record.get("entity_id") or ""),
                    str(record.get("source_table_id") or ""),
                    json.dumps(
                        record.get("row_id"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            host_ordinal = int(
                reference_connection.execute(
                    """
                    INSERT INTO page_host_sequences (
                        policy_fingerprint, host, next_ordinal
                    ) VALUES (?, ?, 1)
                    ON CONFLICT(policy_fingerprint, host) DO UPDATE SET
                        next_ordinal = next_ordinal + 1
                    RETURNING next_ordinal - 1
                    """,
                    (policy_fingerprint, host),
                ).fetchone()[0]
            )
            job_connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, kind, payload_json, status, updated_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (
                    f"{policy_fingerprint}:{url_key}",
                    kind,
                    json.dumps(
                        {
                            "url_key": url_key,
                            "page_url": page_url,
                            "host": host,
                            "policy_fingerprint": policy_fingerprint,
                        },
                        ensure_ascii=False,
                    ),
                    float(host_ordinal),
                ),
            )
            pending += 1
            if pending >= commit_every:
                reference_connection.commit()
                job_connection.commit()
                pending = 0
        reference_connection.commit()
        job_connection.commit()
    except BaseException:
        reference_connection.rollback()
        job_connection.rollback()
        raise
    finally:
        reference_connection.close()
        job_connection.close()


def fetch_unique_pages(
    page_refs: Iterable[dict[str, Any]],
    store: SqliteJobStore,
    transport: Any,
    policy: FetchPolicy = FetchPolicy(),
    *,
    outcomes_path: Path | None = None,
    failure_path: Path | None = None,
    progress_path: Path | None = None,
    claim_buffer: int | None = None,
    lease_seconds: float | None = None,
    progress_every: int = 1_000,
    max_wait_seconds: float = 0.0,
    poll_interval_seconds: float = 0.05,
    after_cache_write: Callable[[dict[str, Any]], None] | None = None,
) -> FetchResult:
    """Fetch each Task-3 URL key once for this exact policy.

    Production callers must obtain ``page_refs`` from
    :func:`iter_finalized_page_refs`, which enforces Task 3's global final
    barrier. Direct iterables are retained as the low-level/test interface.
    """
    fingerprint = policy.fingerprint
    kind = _job_kind(fingerprint)
    transport_policy = getattr(
        transport,
        "network_policy_fingerprint",
        None,
    )
    if transport_policy != policy.network_policy_fingerprint:
        raise ValueError(
            "transport network policy fingerprint does not match FetchPolicy"
        )
    outcomes_path = Path(
        outcomes_path
        or store.path.with_name(f"{store.path.stem}-page-outcomes.sqlite3")
    )
    failure_path = Path(
        failure_path
        or store.path.with_name(f"{store.path.stem}-page-failures.jsonl")
    )
    progress_path = Path(
        progress_path
        or store.path.with_name(f"{store.path.stem}-page-progress.json")
    )
    outcome_store = PageOutcomeStore(outcomes_path)
    _enqueue_page_refs(
        page_refs,
        store=store,
        outcome_store=outcome_store,
        policy_fingerprint=fingerprint,
        kind=kind,
    )

    unique = _job_count(store, kind)
    owner = f"fetch-{os.getpid()}-{uuid.uuid4().hex}"
    buffer_limit = (
        max(policy.global_concurrency, policy.global_concurrency * 4)
        if claim_buffer is None
        else int(claim_buffer)
    )
    if buffer_limit < policy.global_concurrency:
        raise ValueError("claim_buffer must be at least global_concurrency")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if max_wait_seconds < 0:
        raise ValueError("max_wait_seconds must be non-negative")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be positive")
    effective_lease_seconds = (
        float(lease_seconds)
        if lease_seconds is not None
        else (
            policy.deadline_seconds
            * (
                (buffer_limit + policy.per_host_concurrency - 1)
                // policy.per_host_concurrency
                + 2
            )
            + 30.0
        )
    )
    host_states: dict[str, _HostState] = {}
    ready_hosts: deque[str] = deque()
    ready_set: set[str] = set()
    futures: dict[Future[dict[str, Any]], tuple[Job, str]] = {}
    claimed_count = 0
    maximum_claimed = 0
    maximum_inflight = 0
    maximum_host_limiters = 0
    completions_since_progress = 0
    wait_started = time.monotonic()
    _publish_progress(
        outcome_store,
        store=store,
        kind=kind,
        policy_fingerprint=fingerprint,
        unique=unique,
        progress_path=progress_path,
        inflight=0,
    )

    def add_ready(host: str) -> None:
        state = host_states[host]
        if state.jobs and host not in ready_set:
            ready_hosts.append(host)
            ready_set.add(host)

    def complete_from_cache(job: Job, outcome: dict[str, Any]) -> None:
        store.finish(
            job.job_id,
            status=str(outcome["status"]),
            result={
                "url_key": job.payload["url_key"],
                "policy_fingerprint": fingerprint,
            },
            owner=owner,
            lease_id=job.lease_id,
        )

    def claim_more() -> int:
        nonlocal claimed_count, maximum_claimed, maximum_host_limiters
        nonlocal completions_since_progress
        need = buffer_limit - claimed_count - len(futures)
        if need <= 0:
            return 0
        claimed = store.claim(
            kind,
            limit=need,
            owner=owner,
            lease_seconds=effective_lease_seconds,
        )
        for job in claimed:
            cached = outcome_store.get(
                fingerprint,
                str(job.payload["url_key"]),
            )
            if cached is not None:
                complete_from_cache(job, cached)
                completions_since_progress += 1
                if completions_since_progress >= progress_every:
                    _publish_progress(
                        outcome_store,
                        store=store,
                        kind=kind,
                        policy_fingerprint=fingerprint,
                        unique=unique,
                        progress_path=progress_path,
                        inflight=len(futures),
                    )
                    completions_since_progress = 0
                continue
            host = str(job.payload["host"])
            state = host_states.get(host)
            if state is None:
                state = _HostState(
                    semaphore=threading.BoundedSemaphore(
                        policy.per_host_concurrency
                    ),
                    jobs=deque(),
                )
                host_states[host] = state
            state.jobs.append(job)
            claimed_count += 1
            add_ready(host)
        maximum_claimed = max(
            maximum_claimed,
            claimed_count + len(futures),
        )
        maximum_host_limiters = max(
            maximum_host_limiters,
            len(host_states),
        )
        return len(claimed)

    def submit_ready(pool: ThreadPoolExecutor) -> None:
        nonlocal claimed_count, maximum_inflight
        stalled = 0
        while (
            ready_hosts
            and len(futures) < policy.global_concurrency
            and stalled <= len(ready_hosts)
        ):
            host = ready_hosts.popleft()
            ready_set.discard(host)
            state = host_states[host]
            if not state.jobs:
                if state.active == 0:
                    del host_states[host]
                continue
            if not state.semaphore.acquire(blocking=False):
                add_ready(host)
                stalled += 1
                continue
            stalled = 0
            job = state.jobs.popleft()
            state.active += 1
            claimed_count -= 1
            future = pool.submit(
                _fetch_one,
                job,
                transport=transport,
                policy=policy,
            )
            futures[future] = (job, host)
            add_ready(host)
            maximum_inflight = max(maximum_inflight, len(futures))

    try:
        with ThreadPoolExecutor(
            max_workers=policy.global_concurrency,
            thread_name_prefix="wdc-page",
        ) as pool:
            while True:
                claimed_now = claim_more()
                submit_ready(pool)
                if not futures:
                    if claimed_count:
                        raise RuntimeError("page scheduler made no progress")
                    if claimed_now == 0:
                        success_now, terminal_now = outcome_store.counts(
                            fingerprint
                        )
                        unresolved = unique - success_now - terminal_now
                        active_leases = _leased_count(store, kind)
                        elapsed = time.monotonic() - wait_started
                        if (
                            active_leases
                            and (
                                unresolved > 0
                                or elapsed < max_wait_seconds
                            )
                            and elapsed < max_wait_seconds
                        ):
                            _publish_progress(
                                outcome_store,
                                store=store,
                                kind=kind,
                                policy_fingerprint=fingerprint,
                                unique=unique,
                                progress_path=progress_path,
                                inflight=0,
                            )
                            time.sleep(
                                min(
                                    poll_interval_seconds,
                                    max_wait_seconds - elapsed,
                                )
                            )
                            continue
                        break
                    continue
                completed, _pending = wait(
                    tuple(futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    completions_since_progress += 1
                    job, host = futures.pop(future)
                    state = host_states[host]
                    state.active -= 1
                    state.semaphore.release()
                    if state.jobs:
                        add_ready(host)
                    elif state.active == 0:
                        host_states.pop(host, None)
                        ready_set.discard(host)
                    outcome = future.result()
                    persisted = outcome_store.put(
                        fingerprint,
                        str(job.payload["url_key"]),
                        str(job.payload["page_url"]),
                        outcome,
                    )
                    if after_cache_write is not None:
                        after_cache_write(persisted)
                    store.finish(
                        job.job_id,
                        status=str(persisted["status"]),
                        result={
                            "url_key": job.payload["url_key"],
                            "policy_fingerprint": fingerprint,
                        },
                        owner=owner,
                        lease_id=job.lease_id,
                    )
                    if completions_since_progress >= progress_every:
                        _publish_progress(
                            outcome_store,
                            store=store,
                            kind=kind,
                            policy_fingerprint=fingerprint,
                            unique=unique,
                            progress_path=progress_path,
                            inflight=len(futures),
                        )
                        completions_since_progress = 0
                submit_ready(pool)
    except BaseException:
        _publish_snapshots(
            outcome_store,
            store=store,
            kind=kind,
            policy_fingerprint=fingerprint,
            unique=unique,
            failure_path=failure_path,
            progress_path=progress_path,
            inflight=len(futures),
        )
        raise

    success, terminal = _publish_snapshots(
        outcome_store,
        store=store,
        kind=kind,
        policy_fingerprint=fingerprint,
        unique=unique,
        failure_path=failure_path,
        progress_path=progress_path,
        inflight=0,
    )
    leased = _leased_count(store, kind)
    remaining = max(0, unique - success - terminal)
    return FetchResult(
        unique=unique,
        success=success,
        terminal=terminal,
        inflight=leased,
        leased=leased,
        remaining=remaining,
        complete=success + terminal == unique,
        maximum_inflight=maximum_inflight,
        maximum_claimed=maximum_claimed,
        maximum_host_limiters=maximum_host_limiters,
        outcomes_path=outcomes_path,
        failure_path=failure_path,
        progress_path=progress_path,
        policy_fingerprint=fingerprint,
    )


def iter_page_outcomes(
    outcomes_path: Path,
    policy_fingerprint: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream durable page outcomes without loading them into memory."""
    yield from PageOutcomeStore(Path(outcomes_path)).iter(policy_fingerprint)


def iter_page_fanout(
    outcomes_path: Path,
    policy_fingerprint: str,
) -> Iterator[dict[str, Any]]:
    """Stream the disk-backed entity-ref to unique-outcome join for Task 5."""
    outcome_store = PageOutcomeStore(Path(outcomes_path))
    connection = outcome_store._connect()
    try:
        rows = connection.execute(
            """
            SELECT
                refs.entity_id,
                refs.source_table_id,
                refs.row_id_json,
                outcomes.url_key,
                outcomes.page_url,
                outcomes.status,
                outcomes.final_url,
                outcomes.text,
                outcomes.image_urls_json,
                outcomes.error_class,
                outcomes.http_status,
                outcomes.payload_sha256
            FROM page_references AS refs
            JOIN page_outcomes AS outcomes
              ON outcomes.policy_fingerprint = refs.policy_fingerprint
             AND outcomes.url_key = refs.url_key
            WHERE refs.policy_fingerprint = ?
            ORDER BY refs.url_key, refs.reference_key
            """,
            (policy_fingerprint,),
        )
        for row in rows:
            try:
                row_id = json.loads(str(row["row_id_json"]))
                image_urls = json.loads(row["image_urls_json"] or "[]")
            except json.JSONDecodeError as error:
                raise ValueError("corrupt page fanout record") from error
            yield {
                "entity_id": str(row["entity_id"]),
                "source_table_id": str(row["source_table_id"]),
                "row_id": row_id,
                "url_key": str(row["url_key"]),
                "page_url": str(row["page_url"]),
                "status": str(row["status"]),
                "final_url": (
                    None
                    if row["final_url"] is None
                    else str(row["final_url"])
                ),
                "text": None if row["text"] is None else str(row["text"]),
                "image_urls": (
                    image_urls if isinstance(image_urls, list) else []
                ),
                "error_class": (
                    None
                    if row["error_class"] is None
                    else str(row["error_class"])
                ),
                "http_status": _optional_int(row["http_status"]),
                "payload_sha256": str(row["payload_sha256"]),
                "policy_fingerprint": policy_fingerprint,
            }
    finally:
        connection.close()


def iter_finalized_page_refs(
    output_root: Path,
    finalized_selection_manifest: Path,
    structural_manifest_paths: Iterable[Path] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield only page refs covered by Task 3's completed global barrier."""
    output_root = Path(output_root).resolve()
    manifest_path = Path(finalized_selection_manifest).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        barrier = json.load(handle)
    if (
        barrier.get("stage") != "wdc200k_validated_selection"
        or barrier.get("complete") is not True
    ):
        raise ValueError("Task 3 global validated-selection barrier is incomplete")
    barrier_shards = barrier.get("completed_shards") or []
    if len(barrier_shards) != 1:
        raise ValueError("Task 3 global barrier has invalid artifacts")
    completed = CompletedShard(
        path=str(barrier_shards[0]["path"]),
        records=int(barrier_shards[0]["records"]),
        bytes=int(barrier_shards[0]["bytes"]),
        sha256=str(barrier_shards[0]["sha256"]),
    )
    if not validate_completed_shard(completed, output_root):
        raise ValueError("Task 3 global barrier checksum validation failed")

    structural_manifests = sorted(
        (
            Path(path).resolve()
            for path in structural_manifest_paths
        )
        if structural_manifest_paths is not None
        else (output_root / "stage_manifests").glob("structural-*.json")
    )
    if not structural_manifests:
        raise ValueError("Task 3 structural manifests are missing")
    manifest_hashes: list[tuple[str, str]] = []
    page_shards: list[CompletedShard] = []
    structural_table_count = 0
    for structural_manifest in structural_manifests:
        digest = hashlib.sha256(structural_manifest.read_bytes()).hexdigest()
        manifest_hashes.append((structural_manifest.as_posix(), digest))
        with structural_manifest.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            payload.get("stage") != "wdc200k_structural"
            or payload.get("complete") is not True
        ):
            raise ValueError("Task 3 structural manifest is incomplete")
        current_page_shards = [
            item
            for item in payload.get("completed_shards", [])
            if str(item.get("path", "")).startswith("page_refs/")
        ]
        validated_shards = [
            item
            for item in payload.get("completed_shards", [])
            if str(item.get("path", "")).startswith("selection/validated-")
        ]
        if len(current_page_shards) != 1 or len(validated_shards) != 1:
            raise ValueError("Task 3 structural page refs are incomplete")
        page_shard = CompletedShard(
            path=str(current_page_shards[0]["path"]),
            records=int(current_page_shards[0]["records"]),
            bytes=int(current_page_shards[0]["bytes"]),
            sha256=str(current_page_shards[0]["sha256"]),
        )
        if not validate_completed_shard(page_shard, output_root):
            raise ValueError("Task 3 page refs checksum validation failed")
        page_shards.append(page_shard)
        structural_table_count += int(validated_shards[0]["records"])

    expected_input_fingerprint = stable_hash(
        "wdc200k-structural-v2",
        *(
            f"{path}:{digest}"
            for path, digest in manifest_hashes
        ),
        length=40,
    )
    if barrier.get("input_fingerprint") != expected_input_fingerprint:
        raise ValueError(
            "Task 3 structural manifests do not match the global barrier"
        )
    if structural_table_count != completed.records:
        raise ValueError(
            "Task 3 structural table count does not match the global barrier"
        )
    if len({shard.path for shard in page_shards}) != len(page_shards):
        raise ValueError("Task 3 page ref shard is duplicated")

    for page_shard in page_shards:
        path = output_root / page_shard.path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Task 3 page reference is not an object")
                    yield record

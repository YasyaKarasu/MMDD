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
        GuardedTextWriter,
        GuardedWriteTracker,
        Job,
        PreWriteGuard,
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
            GuardedTextWriter,
            GuardedWriteTracker,
            Job,
            PreWriteGuard,
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
    job_store_path: Path
    job_kind: str


class PageOutcomeStore:
    """Disk-backed outcome and reference mappings keyed by policy and URL."""

    def __init__(
        self,
        path: Path,
        *,
        write_tracker: GuardedWriteTracker | None = None,
    ) -> None:
        self.path = Path(path)
        self._write_tracker = write_tracker
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            journal_deadline = time.monotonic() + 30.0
            while True:
                try:
                    connection.execute("PRAGMA journal_mode=WAL").fetchone()
                    break
                except sqlite3.OperationalError as error:
                    if (
                        "locked" not in str(error).casefold()
                        or time.monotonic() >= journal_deadline
                    ):
                        raise
                    time.sleep(0.01)
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
            if self._write_tracker is not None:
                self._write_tracker.before_commit(0)
            connection.commit()
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
    pre_write_guard: PreWriteGuard | None = None,
    target_path: Path | None = None,
) -> dict[str, Any]:
    page_url = str(job.payload["page_url"])
    cached = _transport_cached_outcome(transport, page_url, policy)
    if cached is not None:
        return cached
    if pre_write_guard is not None:
        pre_write_guard(
            target_path or Path("."),
            int(getattr(transport, "max_page_bytes", 0) or 0),
        )
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


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as raw_handle:
            handle = GuardedTextWriter(raw_handle, tracker)
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        tracker.before_commit(0)
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
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as raw_handle:
            handle = GuardedTextWriter(raw_handle, tracker)
            for record in records:
                json.dump(
                    record,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        tracker.before_commit(0)
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
    pre_write_guard: PreWriteGuard | None = None,
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
            pre_write_guard,
        )
        return _publish_progress_unlocked(
            outcome_store,
            store=store,
            kind=kind,
            policy_fingerprint=policy_fingerprint,
            unique=unique,
            progress_path=progress_path,
            inflight=inflight,
            pre_write_guard=pre_write_guard,
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
    pre_write_guard: PreWriteGuard | None = None,
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
            pre_write_guard=pre_write_guard,
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
    pre_write_guard: PreWriteGuard | None = None,
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
        pre_write_guard,
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
    outcome_write_tracker: GuardedWriteTracker | None = None,
) -> None:
    """Stream refs into two idempotent SQLite indexes in bounded transactions."""
    job_connection = store._connect()
    reference_connection = outcome_store._connect()
    pending = 0
    try:
        for record in page_refs:
            encoded_record = json.dumps(record, ensure_ascii=False)
            estimated = 8192 + 2 * len(encoded_record.encode("utf-8"))
            store.reserve_write(estimated)
            if outcome_write_tracker is not None:
                outcome_write_tracker.before_write(estimated)
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
                if outcome_write_tracker is not None:
                    outcome_write_tracker.before_commit(0)
                store.guard_commit()
                reference_connection.commit()
                job_connection.commit()
                pending = 0
        if outcome_write_tracker is not None:
            outcome_write_tracker.before_commit(0)
        store.guard_commit()
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
    pre_write_guard: PreWriteGuard | None = None,
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
    if pre_write_guard is not None:
        for target in (store.path, outcomes_path, failure_path, progress_path):
            pre_write_guard(target, 0)
    outcome_write_tracker = GuardedWriteTracker(
        outcomes_path,
        pre_write_guard,
    )
    outcome_write_tracker.before_write(64 * 1024)
    outcome_store = PageOutcomeStore(
        outcomes_path,
        write_tracker=outcome_write_tracker,
    )
    _enqueue_page_refs(
        page_refs,
        store=store,
        outcome_store=outcome_store,
        policy_fingerprint=fingerprint,
        kind=kind,
        outcome_write_tracker=outcome_write_tracker,
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
        pre_write_guard=pre_write_guard,
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
                pre_write_guard=pre_write_guard,
                target_path=outcomes_path,
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
                                pre_write_guard=pre_write_guard,
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
                    outcome_write_tracker.before_write(
                        4096
                        + 2 * len(
                            json.dumps(
                                outcome,
                                ensure_ascii=False,
                            ).encode("utf-8")
                        )
                    )
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
                            pre_write_guard=pre_write_guard,
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
            pre_write_guard=pre_write_guard,
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
        pre_write_guard=pre_write_guard,
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
        job_store_path=store.path,
        job_kind=kind,
    )


def validate_complete_page_fetch(
    result: FetchResult,
    page_refs: Iterable[dict[str, Any]],
    *,
    validation_database: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> dict[str, Any]:
    """Re-derive a complete Task-4 identity from durable producer state."""
    if (
        not result.complete
        or result.remaining != 0
        or result.leased != 0
        or result.inflight != 0
    ):
        raise ValueError("page fetch result is incomplete")
    expected_kind = _job_kind(result.policy_fingerprint)
    if result.job_kind != expected_kind:
        raise ValueError("page fetch job store kind mismatch")
    paths = (
        Path(result.outcomes_path),
        Path(result.failure_path),
        Path(result.progress_path),
        Path(result.job_store_path),
    )
    if not all(path.is_file() for path in paths):
        raise ValueError("page fetch job store or snapshot is missing")

    validation_database = Path(validation_database)
    write_tracker = GuardedWriteTracker(
        validation_database,
        pre_write_guard,
    )
    write_tracker.before_write(64 * 1024)
    validation_database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(validation_database) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS expected_page_refs (
                url_key TEXT NOT NULL,
                reference_key TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                row_id_json TEXT NOT NULL,
                PRIMARY KEY (url_key, reference_key)
            );
            CREATE TABLE IF NOT EXISTS expected_page_urls (
                url_key TEXT PRIMARY KEY,
                page_url TEXT NOT NULL
            );
            DELETE FROM expected_page_refs;
            DELETE FROM expected_page_urls;
            """
        )
        for record in page_refs:
            encoded_record = json.dumps(record, ensure_ascii=False)
            write_tracker.before_write(
                8192 + 2 * len(encoded_record.encode("utf-8"))
            )
            url_key, page_url, _host = _validated_ref(record)
            reference_key = stable_hash(
                record.get("entity_id", ""),
                record.get("source_table_id", ""),
                record.get("row_id", ""),
                length=40,
            )
            reference_values = (
                url_key,
                reference_key,
                str(record.get("entity_id") or ""),
                str(record.get("source_table_id") or ""),
                json.dumps(
                    record.get("row_id"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            existing = connection.execute(
                """
                SELECT url_key, reference_key, entity_id,
                       source_table_id, row_id_json
                FROM expected_page_refs
                WHERE url_key = ? AND reference_key = ?
                """,
                (url_key, reference_key),
            ).fetchone()
            if existing is not None and tuple(existing) != reference_values:
                raise ValueError("conflicting expected page reference")
            connection.execute(
                """
                INSERT OR IGNORE INTO expected_page_refs
                VALUES (?, ?, ?, ?, ?)
                """,
                reference_values,
            )
            url_row = connection.execute(
                """
                SELECT page_url FROM expected_page_urls
                WHERE url_key = ?
                """,
                (url_key,),
            ).fetchone()
            if url_row is not None and str(url_row[0]) != page_url:
                raise ValueError("conflicting expected page URL")
            connection.execute(
                """
                INSERT OR IGNORE INTO expected_page_urls
                VALUES (?, ?)
                """,
                (url_key, page_url),
            )
        write_tracker.before_commit(0)
        connection.commit()
        connection.execute(
            "ATTACH DATABASE ? AS outcomes_db",
            (str(Path(result.outcomes_path).resolve()),),
        )
        connection.execute(
            "ATTACH DATABASE ? AS jobs_db",
            (str(Path(result.job_store_path).resolve()),),
        )
        policy = result.policy_fingerprint
        expected_ref_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM expected_page_refs"
            ).fetchone()[0]
        )
        actual_ref_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM outcomes_db.page_references
                WHERE policy_fingerprint = ?
                """,
                (policy,),
            ).fetchone()[0]
        )
        missing_refs = connection.execute(
            """
            SELECT COUNT(*)
            FROM expected_page_refs AS expected
            LEFT JOIN outcomes_db.page_references AS actual
              ON actual.policy_fingerprint = ?
             AND actual.url_key = expected.url_key
             AND actual.reference_key = expected.reference_key
             AND actual.entity_id = expected.entity_id
             AND actual.source_table_id = expected.source_table_id
             AND actual.row_id_json = expected.row_id_json
            WHERE actual.url_key IS NULL
            """,
            (policy,),
        ).fetchone()[0]
        if int(missing_refs) != 0 or actual_ref_count != expected_ref_count:
            raise ValueError("page fetch reference set mismatch")
        expected_url_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM expected_page_urls"
            ).fetchone()[0]
        )
        actual_url_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM outcomes_db.page_outcomes
                WHERE policy_fingerprint = ?
                """,
                (policy,),
            ).fetchone()[0]
        )
        missing_urls = connection.execute(
            """
            SELECT COUNT(*)
            FROM expected_page_urls AS expected
            LEFT JOIN outcomes_db.page_outcomes AS actual
              ON actual.policy_fingerprint = ?
             AND actual.url_key = expected.url_key
             AND actual.page_url = expected.page_url
            WHERE actual.url_key IS NULL
            """,
            (policy,),
        ).fetchone()[0]
        if int(missing_urls) != 0 or actual_url_count != expected_url_count:
            raise ValueError("page fetch outcome URL set mismatch")
        job_rows = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS success,
                SUM(CASE WHEN status = 'terminal' THEN 1 ELSE 0 END)
                    AS terminal,
                SUM(CASE WHEN status NOT IN ('success', 'terminal')
                         THEN 1 ELSE 0 END) AS incomplete
            FROM jobs_db.jobs WHERE kind = ?
            """,
            (expected_kind,),
        ).fetchone()
        expected_unique = int(
            connection.execute(
                "SELECT COUNT(*) FROM expected_page_urls"
            ).fetchone()[0]
        )
        outcome_rows = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS success,
                SUM(CASE WHEN status = 'terminal' THEN 1 ELSE 0 END)
                    AS terminal
            FROM outcomes_db.page_outcomes
            WHERE policy_fingerprint = ?
            """,
            (policy,),
        ).fetchone()
        counts = {
            "unique": expected_unique,
            "success": int(outcome_rows["success"] or 0),
            "terminal": int(outcome_rows["terminal"] or 0),
        }
        if (
            int(outcome_rows["total"] or 0) != expected_unique
            or int(job_rows["total"] or 0) != expected_unique
            or int(job_rows["success"] or 0) != counts["success"]
            or int(job_rows["terminal"] or 0) != counts["terminal"]
            or int(job_rows["incomplete"] or 0) != 0
            or result.unique != counts["unique"]
            or result.success != counts["success"]
            or result.terminal != counts["terminal"]
        ):
            raise ValueError("page fetch job store counts mismatch")
        for row in connection.execute(
            """
            SELECT expected.url_key, expected.page_url,
                   jobs.job_id, jobs.payload_json, jobs.status,
                   jobs.result_json, jobs.owner, jobs.lease_expires,
                   jobs.lease_id, outcomes.status AS outcome_status
            FROM expected_page_urls AS expected
            LEFT JOIN jobs_db.jobs AS jobs
              ON jobs.job_id = ? || ':' || expected.url_key
             AND jobs.kind = ?
            LEFT JOIN outcomes_db.page_outcomes AS outcomes
              ON outcomes.policy_fingerprint = ?
             AND outcomes.url_key = expected.url_key
            ORDER BY expected.url_key
            """,
            (policy, expected_kind, policy),
        ):
            expected_payload = {
                "url_key": str(row["url_key"]),
                "page_url": str(row["page_url"]),
                "host": _host_for_url(str(row["page_url"])),
                "policy_fingerprint": policy,
            }
            expected_result = {
                "url_key": str(row["url_key"]),
                "policy_fingerprint": policy,
            }
            try:
                payload = json.loads(str(row["payload_json"]))
                result_payload = json.loads(str(row["result_json"]))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    "page fetch job store membership mismatch"
                ) from error
            if (
                str(row["job_id"]) != f"{policy}:{row['url_key']}"
                or payload != expected_payload
                or result_payload != expected_result
                or str(row["status"]) != str(row["outcome_status"])
                or row["owner"] is not None
                or row["lease_expires"] is not None
                or row["lease_id"] is not None
            ):
                raise ValueError(
                    "page fetch job store membership mismatch"
                )

        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "schema_version": FETCH_SCHEMA_VERSION,
                    "policy_fingerprint": policy,
                    **counts,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for row in connection.execute(
            """
            SELECT url_key, page_url, status, payload_sha256
            FROM outcomes_db.page_outcomes
            WHERE policy_fingerprint = ?
            ORDER BY url_key
            """,
            (policy,),
        ):
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        for row in connection.execute(
            """
            SELECT url_key, reference_key, entity_id,
                   source_table_id, row_id_json
            FROM expected_page_refs
            ORDER BY url_key, reference_key
            """
        ):
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )

    for outcome in PageOutcomeStore(result.outcomes_path).iter(
        result.policy_fingerprint
    ):
        canonical = {
            "policy_fingerprint": outcome["policy_fingerprint"],
            "url_key": outcome["url_key"],
            "page_url": outcome["page_url"],
            "status": outcome["status"],
            "final_url": outcome["final_url"],
            "text": outcome["text"],
            "image_urls": outcome["image_urls"],
            "error_class": outcome["error_class"],
            "http_status": outcome["http_status"],
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != outcome[
            "payload_sha256"
        ]:
            raise ValueError("page fetch outcome checksum mismatch")

    progress = json.loads(
        Path(result.progress_path).read_text(encoding="utf-8")
    )
    expected_progress = {
        "policy_fingerprint": result.policy_fingerprint,
        "unique": counts["unique"],
        "success": counts["success"],
        "terminal": counts["terminal"],
        "pending": 0,
        "leased": 0,
        "remaining": 0,
        "complete": True,
        "inflight": 0,
    }
    if any(progress.get(key) != value for key, value in expected_progress.items()):
        raise ValueError("page fetch progress snapshot mismatch")
    expected_failures = [
        _sanitized_failure(
            outcome,
            int(outcome["affected_reference_count"]),
        )
        for outcome in PageOutcomeStore(result.outcomes_path).iter(
            result.policy_fingerprint
        )
        if outcome["status"] == "terminal"
    ]
    actual_failures = []
    with Path(result.failure_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                actual_failures.append(json.loads(line))
    if actual_failures != expected_failures:
        raise ValueError("page fetch failure snapshot mismatch")
    return {
        **counts,
        "identity": digest.hexdigest(),
        "failure_records": len(expected_failures),
    }


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

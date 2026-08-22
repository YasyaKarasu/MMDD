"""Durable, bounded-memory I/O primitives for the WDC 200K pipeline."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO

try:
    from build_mm_table_dataset import (
        iter_jsonl_records,
        stable_hash,
        write_jsonl_record,
    )
except ModuleNotFoundError as error:
    if error.name != "build_mm_table_dataset":
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        from build_mm_table_dataset import (
            iter_jsonl_records,
            stable_hash,
            write_jsonl_record,
        )
    finally:
        sys.path.remove(scripts_directory)


@dataclass(frozen=True)
class CompletedShard:
    path: str
    records: int
    bytes: int
    sha256: str


PreWriteGuard = Callable[[Path, int], None]


def _guard_write(
    guard: PreWriteGuard | None,
    path: Path,
    estimated_bytes: int = 0,
) -> None:
    if guard is not None:
        guard(Path(path), max(0, int(estimated_bytes)))


class GuardedWriteTracker:
    """Amortize filesystem checks while reserving bounded write windows."""

    DEFAULT_INTERVAL_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        path: Path,
        guard: PreWriteGuard | None,
        *,
        interval_bytes: int | None = None,
    ) -> None:
        interval = (
            self.DEFAULT_INTERVAL_BYTES
            if interval_bytes is None
            else int(interval_bytes)
        )
        if interval <= 0:
            raise ValueError("guard interval must be positive")
        self.path = Path(path)
        self.guard = guard
        self.interval_bytes = interval
        self._remaining = 0
        self._lock = threading.Lock()
        _guard_write(self.guard, self.path, 0)

    def before_write(self, estimated_bytes: int) -> None:
        estimated = max(0, int(estimated_bytes))
        if self.guard is None or estimated == 0:
            return
        with self._lock:
            if estimated > self._remaining:
                reservation = max(self.interval_bytes, estimated)
                _guard_write(self.guard, self.path, reservation)
                self._remaining = reservation
            self._remaining -= estimated

    def before_commit(
        self,
        replacement_delta_bytes: int = 0,
        *,
        force: bool = True,
    ) -> None:
        replacement_delta = max(0, int(replacement_delta_bytes))
        if force:
            _guard_write(self.guard, self.path, replacement_delta)
            return
        if self.guard is None:
            return
        with self._lock:
            if replacement_delta > self._remaining:
                reservation = max(self.interval_bytes, replacement_delta)
                _guard_write(self.guard, self.path, reservation)
                self._remaining = reservation
            self._remaining -= replacement_delta
            if self._remaining == 0:
                _guard_write(self.guard, self.path, 0)


class GuardedTextWriter:
    """Text writer that accounts UTF-8 bytes before touching the file."""

    def __init__(
        self,
        handle: TextIO,
        tracker: GuardedWriteTracker,
    ) -> None:
        self.handle = handle
        self.tracker = tracker

    def write(self, value: str) -> int:
        self.tracker.before_write(len(value.encode("utf-8")))
        return self.handle.write(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AtomicJsonlShard:
    """Write a JSONL shard that becomes visible only after a durable commit."""

    DEFAULT_GUARD_INTERVAL_BYTES = GuardedWriteTracker.DEFAULT_INTERVAL_BYTES

    def __init__(
        self,
        path: Path,
        *,
        pre_write_guard: PreWriteGuard | None = None,
        guard_interval_bytes: int = DEFAULT_GUARD_INTERVAL_BYTES,
    ) -> None:
        if guard_interval_bytes <= 0:
            raise ValueError("guard_interval_bytes must be positive")
        self.path = path
        self.temporary_path = path.with_suffix(path.suffix + ".tmp")
        self.pre_write_guard = pre_write_guard
        self.guard_interval_bytes = int(guard_interval_bytes)
        self._tracker = GuardedWriteTracker(
            self.path,
            pre_write_guard,
            interval_bytes=self.guard_interval_bytes,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary_path.open("w", encoding="utf-8")
        self._records = 0
        self._committed = False

    def write(self, record: dict[str, Any]) -> None:
        if self._handle.closed:
            raise RuntimeError("cannot write to a closed shard")
        encoded = json.dumps(record, ensure_ascii=False) + "\n"
        self.write_text(encoded)
        self._records += 1

    def write_text(self, value: str) -> None:
        self._tracker.before_write(len(value.encode("utf-8")))
        self._handle.write(value)

    def commit(self) -> CompletedShard:
        if self._handle.closed:
            raise RuntimeError("cannot commit a closed shard")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        size = self.temporary_path.stat().st_size
        self._tracker.before_commit(0)
        self._handle.close()
        digest = _sha256_path(self.temporary_path)
        self.temporary_path.replace(self.path)
        _fsync_directory(self.path.parent)
        self._committed = True
        return CompletedShard(
            path=self.path.name,
            records=self._records,
            bytes=size,
            sha256=digest,
        )

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        if not self._committed:
            self.temporary_path.unlink(missing_ok=True)


def validate_completed_shard(record: CompletedShard, root: Path) -> bool:
    """Return whether a completed-shard record still matches its file."""
    root = root.resolve()
    path = (root / record.path).resolve()
    if not path.is_relative_to(root):
        return False
    if not path.is_file() or path.stat().st_size != record.bytes:
        return False
    if _sha256_path(path) != record.sha256:
        return False
    with path.open("rb") as handle:
        return sum(1 for _ in handle) == record.records


@dataclass(frozen=True)
class StageFingerprint:
    stage: str
    input_fingerprint: str
    parameter_fingerprint: str
    schema_version: str = ""

    @property
    def digest(self) -> str:
        values = (
            self.stage,
            self.input_fingerprint,
            self.parameter_fingerprint,
        )
        if self.schema_version:
            values = (*values, self.schema_version)
        return stable_hash(*values, length=40)


class StageManifest:
    """Atomically persisted stage progress that can be safely resumed."""

    def __init__(
        self,
        path: Path,
        fingerprint: StageFingerprint,
        *,
        pre_write_guard: PreWriteGuard | None = None,
    ) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.pre_write_guard = pre_write_guard
        self.completed_shards: list[CompletedShard] = []
        self.complete = False
        if self.path.exists():
            self._load()
        else:
            self._save()

    @property
    def total_records(self) -> int:
        return sum(shard.records for shard in self.completed_shards)

    @property
    def total_bytes(self) -> int:
        return sum(shard.bytes for shard in self.completed_shards)

    def record_shard(self, shard: CompletedShard) -> None:
        if self.complete:
            raise RuntimeError("cannot add a shard to a completed manifest")
        existing = {
            completed.path: completed for completed in self.completed_shards
        }.get(shard.path)
        if existing is not None:
            if existing != shard:
                raise ValueError(f"conflicting completed shard: {shard.path}")
            return
        self.completed_shards.append(shard)
        self._save()

    def mark_complete(self) -> None:
        self.complete = True
        self._save()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        stored_fingerprint = StageFingerprint(
            stage=str(payload["stage"]),
            input_fingerprint=str(payload["input_fingerprint"]),
            parameter_fingerprint=str(payload["parameter_fingerprint"]),
            schema_version=str(payload.get("schema_version", "")),
        )
        if stored_fingerprint != self.fingerprint:
            raise ValueError(
                "stage manifest fingerprint does not match requested stage"
            )
        self.completed_shards = [
            CompletedShard(
                path=str(shard["path"]),
                records=int(shard["records"]),
                bytes=int(shard["bytes"]),
                sha256=str(shard["sha256"]),
            )
            for shard in payload.get("completed_shards", [])
        ]
        self.complete = bool(payload.get("complete", False))

    def _save(self) -> None:
        tracker = GuardedWriteTracker(self.path, self.pre_write_guard)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "stage": self.fingerprint.stage,
            "input_fingerprint": self.fingerprint.input_fingerprint,
            "parameter_fingerprint": self.fingerprint.parameter_fingerprint,
            "completed_shards": [
                {
                    "path": shard.path,
                    "records": shard.records,
                    "bytes": shard.bytes,
                    "sha256": shard.sha256,
                }
                for shard in self.completed_shards
            ],
            "totals": {
                "shards": len(self.completed_shards),
                "records": self.total_records,
                "bytes": self.total_bytes,
            },
            "complete": self.complete,
        }
        if self.fingerprint.schema_version:
            payload["schema_version"] = self.fingerprint.schema_version
        try:
            with temporary_path.open("w", encoding="utf-8") as raw_handle:
                handle = GuardedTextWriter(raw_handle, tracker)
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                raw_handle.flush()
                os.fsync(raw_handle.fileno())
            tracker.before_commit(0)
            temporary_path.replace(self.path)
            _fsync_directory(self.path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


@dataclass(frozen=True)
class Job:
    job_id: str
    kind: str
    payload: dict[str, Any]
    status: str
    result: dict[str, Any] | None
    owner: str | None
    lease_expires: float | None
    lease_id: str | None


class SqliteJobStore:
    """Persistent SQLite-backed leases and terminal outcomes."""

    def __init__(
        self,
        path: Path,
        *,
        pre_write_guard: PreWriteGuard | None = None,
        guard_interval_bytes: int | None = None,
    ) -> None:
        self.path = path
        self._write_tracker = GuardedWriteTracker(
            path,
            pre_write_guard,
            interval_bytes=guard_interval_bytes,
        )
        self._write_tracker.before_write(64 * 1024)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    owner TEXT,
                    lease_expires REAL,
                    lease_id TEXT,
                    updated_at REAL NOT NULL
                )
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)")
            }
            if "lease_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN lease_id TEXT")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS jobs_kind_status
                ON jobs(kind, status, updated_at)
                """
            )
            self._write_tracker.before_commit(0)
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

    def reserve_write(self, estimated_bytes: int) -> None:
        """Reserve bounded capacity before a caller-managed transaction."""
        self._write_tracker.before_write(estimated_bytes)

    def guard_commit(self) -> None:
        """Recheck the reserve before committing already allocated pages."""
        self._write_tracker.before_commit(0)

    def enqueue(self, kind: str, job_id: str, payload: dict[str, Any]) -> None:
        now = time.time()
        encoded_payload = json.dumps(payload, ensure_ascii=False)
        self._write_tracker.before_write(
            4096 + 2 * len(encoded_payload.encode("utf-8"))
        )
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, kind, payload_json, status, updated_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (job_id, kind, encoded_payload, now),
            )
            self._write_tracker.before_commit(0)
            connection.commit()
        finally:
            connection.close()

    def claim(
        self,
        kind: str,
        limit: int,
        owner: str,
        lease_seconds: float = 300.0,
        *,
        pending_statuses: tuple[str, ...] = ("pending", "retryable"),
        lease_status: str = "leased",
    ) -> list[Job]:
        if not pending_statuses or any(not status for status in pending_statuses):
            raise ValueError("pending statuses must not be empty")
        if not lease_status:
            raise ValueError("lease status must not be empty")
        now = time.time()
        lease_expires = now + lease_seconds
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            status_placeholders = ",".join("?" for _ in pending_statuses)
            rows = connection.execute(
                f"""
                SELECT job_id
                FROM jobs
                WHERE kind = ?
                  AND (
                    status IN ({status_placeholders})
                    OR (status = ? AND lease_expires <= ?)
                  )
                ORDER BY updated_at, job_id
                LIMIT ?
                """,
                (
                    kind,
                    *pending_statuses,
                    lease_status,
                    now,
                    max(0, limit),
                ),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            if job_ids:
                self._write_tracker.before_write(
                    4096 + len(job_ids) * 512
                )
                placeholders = ",".join("?" for _ in job_ids)
                for job_id in job_ids:
                    connection.execute(
                        """
                    UPDATE jobs
                    SET status = ?, owner = ?, lease_expires = ?,
                        lease_id = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                        (
                            lease_status,
                            owner,
                            lease_expires,
                            uuid.uuid4().hex,
                            now,
                            job_id,
                        ),
                    )
                claimed = connection.execute(
                    f"""
                    SELECT *
                    FROM jobs
                    WHERE job_id IN ({placeholders})
                    ORDER BY updated_at, job_id
                    """,
                    job_ids,
                ).fetchall()
            else:
                claimed = []
            if job_ids:
                self._write_tracker.before_commit(0)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return [self._job_from_row(row) for row in claimed]

    def release_owner_leases(self, kind: str, *, owner: str) -> int:
        """Make unfinished leases for one kind and execution claimable."""
        if not kind:
            raise ValueError("job kind must not be empty")
        if not owner:
            raise ValueError("job lease owner must not be empty")
        now = time.time()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM jobs
                    WHERE kind = ? AND status = 'leased' AND owner = ?
                    """,
                    (kind, owner),
                ).fetchone()[0]
            )
            if count:
                self._write_tracker.before_write(4096 + count * 512)
                cursor = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'retryable', owner = NULL,
                        lease_expires = NULL, lease_id = NULL, updated_at = ?
                    WHERE kind = ? AND status = 'leased' AND owner = ?
                    """,
                    (now, kind, owner),
                )
                if cursor.rowcount != count:
                    raise RuntimeError("job lease release count changed")
                self._write_tracker.before_commit(0)
            connection.commit()
            return count
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finish(
        self,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        owner: str | None = None,
        lease_id: str | None = None,
    ) -> None:
        if status not in {"success", "terminal", "retryable"}:
            raise ValueError(f"invalid finish status: {status}")
        if not owner:
            raise ValueError("owner is required to finish a leased job")
        if not lease_id:
            raise ValueError("lease_id is required to finish a leased job")
        now = time.time()
        encoded_result = (
            None
            if result is None
            else json.dumps(result, ensure_ascii=False)
        )
        self._write_tracker.before_write(
            4096
            + (
                0
                if encoded_result is None
                else 2 * len(encoded_result.encode("utf-8"))
            )
        )
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, owner = NULL,
                    lease_expires = NULL, lease_id = NULL, updated_at = ?
                WHERE job_id = ?
                  AND status = 'leased'
                  AND owner = ?
                  AND lease_id = ?
                  AND lease_expires > ?
                """,
                (
                    status,
                    encoded_result,
                    now,
                    job_id,
                    owner,
                    lease_id,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"job {job_id!r} has no active lease owned by {owner!r}"
                )
            self._write_tracker.before_commit(0)
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> Job:
        return Job(
            job_id=str(row["job_id"]),
            kind=str(row["kind"]),
            payload=json.loads(row["payload_json"]),
            status=str(row["status"]),
            result=(
                None
                if row["result_json"] is None
                else json.loads(row["result_json"])
            ),
            owner=None if row["owner"] is None else str(row["owner"]),
            lease_expires=(
                None
                if row["lease_expires"] is None
                else float(row["lease_expires"])
            ),
            lease_id=(
                None if row["lease_id"] is None else str(row["lease_id"])
            ),
        )


def _external_key(key: Any) -> str:
    try:
        return json.dumps(
            key,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("external uniqueness keys must be JSON serializable") from error


def _write_sorted_run(
    records: list[tuple[str, int, dict[str, Any]]],
    path: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    records.sort(key=lambda item: (item[0], item[1]))
    with path.open("w", encoding="utf-8") as raw_handle:
        handle = GuardedTextWriter(raw_handle, tracker)
        for key, ordinal, record in records:
            write_jsonl_record(handle, [key, ordinal, record])


def _iter_sorted_run(handle: TextIO) -> Iterator[tuple[str, int, dict[str, Any]]]:
    for line in handle:
        key, ordinal, record = json.loads(line)
        yield str(key), int(ordinal), record


def _iter_merged_runs(
    run_paths: list[Path],
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    with ExitStack() as stack:
        handles = [
            stack.enter_context(run_path.open("r", encoding="utf-8"))
            for run_path in run_paths
        ]
        yield from heapq.merge(
            *(_iter_sorted_run(handle) for handle in handles),
            key=lambda item: (item[0], item[1]),
        )


def _merge_run_group(
    run_paths: list[Path],
    output_path: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(output_path, pre_write_guard)
    with output_path.open("w", encoding="utf-8") as raw_handle:
        handle = GuardedTextWriter(raw_handle, tracker)
        for key, ordinal, record in _iter_merged_runs(run_paths):
            write_jsonl_record(handle, [key, ordinal, record])


class _RunAccumulator:
    """Incrementally compact sorted runs with bounded per-level metadata."""

    def __init__(
        self,
        temporary_dir: Path,
        merge_fan_in: int,
        pre_write_guard: PreWriteGuard | None = None,
    ) -> None:
        self.temporary_dir = temporary_dir
        self.merge_fan_in = merge_fan_in
        self._levels: list[list[Path]] = []
        self._merge_index = 0
        self.pre_write_guard = pre_write_guard

    @property
    def pending_path_count(self) -> int:
        return sum(len(level) for level in self._levels)

    def add(self, run_path: Path) -> None:
        self._add_at_level(run_path, level_index=0)

    def pending_paths(self) -> list[Path]:
        return [
            run_path
            for level in self._levels
            for run_path in level
        ]

    def _add_at_level(self, run_path: Path, level_index: int) -> None:
        while len(self._levels) <= level_index:
            self._levels.append([])
        level = self._levels[level_index]
        level.append(run_path)
        if len(level) < self.merge_fan_in:
            return

        group = list(level)
        level.clear()
        merged_path = (
            self.temporary_dir
            / f"merge-online-{self._merge_index:08d}.jsonl"
        )
        self._merge_index += 1
        _merge_run_group(group, merged_path, self.pre_write_guard)
        for grouped_path in group:
            grouped_path.unlink()
        self._add_at_level(merged_path, level_index + 1)


def _reduce_sorted_runs(
    run_paths: list[Path],
    temporary_dir: Path,
    merge_fan_in: int,
    pre_write_guard: PreWriteGuard | None = None,
) -> list[Path]:
    merge_pass = 0
    while len(run_paths) > merge_fan_in:
        reduced_paths: list[Path] = []
        for group_index, start in enumerate(
            range(0, len(run_paths), merge_fan_in)
        ):
            group = run_paths[start : start + merge_fan_in]
            if len(group) == 1:
                reduced_paths.append(group[0])
                continue
            merged_path = (
                temporary_dir
                / f"merge-{merge_pass:04d}-{group_index:08d}.jsonl"
            )
            _merge_run_group(group, merged_path, pre_write_guard)
            for run_path in group:
                run_path.unlink()
            reduced_paths.append(merged_path)
        run_paths = reduced_paths
        merge_pass += 1
    return run_paths


def external_unique_jsonl(
    input_paths: Iterable[Path],
    output_path: Path,
    key_fn: Callable[[dict[str, Any]], Any],
    chunk_records: int,
    merge_fan_in: int = 64,
    pre_write_guard: PreWriteGuard | None = None,
    progress_callback: Callable[[dict[str, int | str]], None] | None = None,
    progress_every: int = 10_000,
    total_records: int | None = None,
) -> CompletedShard:
    """Externally sort JSONL records and keep the first record for each key."""
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    if merge_fan_in < 2:
        raise ValueError("merge_fan_in must be at least 2")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if total_records is not None and total_records < 0:
        raise ValueError("total_records must be non-negative")

    def report(phase: str, completed: int, total: int) -> None:
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": phase,
                    "completed": completed,
                    "total": total,
                }
            )

    _guard_write(pre_write_guard, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".wdc200k-unique-",
        dir=output_path.parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        run_accumulator = _RunAccumulator(
            temporary_dir,
            merge_fan_in,
            pre_write_guard,
        )
        run_index = 0
        chunk: list[tuple[str, int, dict[str, Any]]] = []
        ordinal = 0
        expected_total = 0 if total_records is None else total_records
        report("read_records", 0, expected_total)
        for record in iter_jsonl_records(input_paths):
            chunk.append((_external_key(key_fn(record)), ordinal, record))
            ordinal += 1
            if ordinal % progress_every == 0:
                report(
                    "read_records",
                    ordinal,
                    max(expected_total, ordinal),
                )
            if len(chunk) >= chunk_records:
                run_path = temporary_dir / f"run-{run_index:08d}.jsonl"
                run_index += 1
                _write_sorted_run(chunk, run_path, pre_write_guard)
                run_accumulator.add(run_path)
                chunk = []
        report("read_records", ordinal, max(expected_total, ordinal))
        if chunk:
            run_path = temporary_dir / f"run-{run_index:08d}.jsonl"
            _write_sorted_run(chunk, run_path, pre_write_guard)
            run_accumulator.add(run_path)
        run_paths = _reduce_sorted_runs(
            run_accumulator.pending_paths(),
            temporary_dir,
            merge_fan_in,
            pre_write_guard,
        )

        output = AtomicJsonlShard(
            output_path,
            pre_write_guard=pre_write_guard,
        )
        try:
            previous_key: str | None = None
            has_previous_key = False
            merged_records = 0
            report("write_records", 0, ordinal)
            for key, _ordinal, record in _iter_merged_runs(run_paths):
                merged_records += 1
                if merged_records % progress_every == 0:
                    report("write_records", merged_records, ordinal)
                if has_previous_key and key == previous_key:
                    continue
                output.write(record)
                previous_key = key
                has_previous_key = True
            report("write_records", merged_records, ordinal)
            return output.commit()
        except BaseException:
            output.abort()
            raise

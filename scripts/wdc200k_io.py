"""Durable, bounded-memory I/O primitives for the WDC 200K pipeline."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import sys
import tempfile
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

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary_path = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temporary_path.open("w", encoding="utf-8")
        self._records = 0
        self._committed = False

    def write(self, record: dict[str, Any]) -> None:
        if self._handle.closed:
            raise RuntimeError("cannot write to a closed shard")
        write_jsonl_record(self._handle, record)
        self._records += 1

    def commit(self) -> CompletedShard:
        if self._handle.closed:
            raise RuntimeError("cannot commit a closed shard")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        digest = _sha256_path(self.temporary_path)
        size = self.temporary_path.stat().st_size
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

    @property
    def digest(self) -> str:
        return stable_hash(
            self.stage,
            self.input_fingerprint,
            self.parameter_fingerprint,
            length=40,
        )


class StageManifest:
    """Atomically persisted stage progress that can be safely resumed."""

    def __init__(self, path: Path, fingerprint: StageFingerprint) -> None:
        self.path = path
        self.fingerprint = fingerprint
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
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
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

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
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
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def enqueue(self, kind: str, job_id: str, payload: dict[str, Any]) -> None:
        now = time.time()
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, kind, payload_json, status, updated_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (job_id, kind, json.dumps(payload, ensure_ascii=False), now),
            )
            connection.commit()
        finally:
            connection.close()

    def claim(
        self,
        kind: str,
        limit: int,
        owner: str,
        lease_seconds: float = 300.0,
    ) -> list[Job]:
        now = time.time()
        lease_expires = now + lease_seconds
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT job_id
                FROM jobs
                WHERE kind = ?
                  AND (
                    status IN ('pending', 'retryable')
                    OR (status = 'leased' AND lease_expires <= ?)
                  )
                ORDER BY updated_at, job_id
                LIMIT ?
                """,
                (kind, now, max(0, limit)),
            ).fetchall()
            job_ids = [str(row["job_id"]) for row in rows]
            if job_ids:
                placeholders = ",".join("?" for _ in job_ids)
                for job_id in job_ids:
                    connection.execute(
                        """
                    UPDATE jobs
                    SET status = 'leased', owner = ?, lease_expires = ?,
                        lease_id = ?, updated_at = ?
                    WHERE job_id = ?
                    """,
                        (owner, lease_expires, uuid.uuid4().hex, now, job_id),
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
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return [self._job_from_row(row) for row in claimed]

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
                    None if result is None else json.dumps(result, ensure_ascii=False),
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
) -> None:
    records.sort(key=lambda item: (item[0], item[1]))
    with path.open("w", encoding="utf-8") as handle:
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


def _merge_run_group(run_paths: list[Path], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for key, ordinal, record in _iter_merged_runs(run_paths):
            write_jsonl_record(handle, [key, ordinal, record])


class _RunAccumulator:
    """Incrementally compact sorted runs with bounded per-level metadata."""

    def __init__(self, temporary_dir: Path, merge_fan_in: int) -> None:
        self.temporary_dir = temporary_dir
        self.merge_fan_in = merge_fan_in
        self._levels: list[list[Path]] = []
        self._merge_index = 0

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
        _merge_run_group(group, merged_path)
        for grouped_path in group:
            grouped_path.unlink()
        self._add_at_level(merged_path, level_index + 1)


def _reduce_sorted_runs(
    run_paths: list[Path],
    temporary_dir: Path,
    merge_fan_in: int,
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
            _merge_run_group(group, merged_path)
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
) -> CompletedShard:
    """Externally sort JSONL records and keep the first record for each key."""
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    if merge_fan_in < 2:
        raise ValueError("merge_fan_in must be at least 2")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wdc200k-unique-") as temporary:
        temporary_dir = Path(temporary)
        run_accumulator = _RunAccumulator(temporary_dir, merge_fan_in)
        run_index = 0
        chunk: list[tuple[str, int, dict[str, Any]]] = []
        ordinal = 0
        for record in iter_jsonl_records(input_paths):
            chunk.append((_external_key(key_fn(record)), ordinal, record))
            ordinal += 1
            if len(chunk) >= chunk_records:
                run_path = temporary_dir / f"run-{run_index:08d}.jsonl"
                run_index += 1
                _write_sorted_run(chunk, run_path)
                run_accumulator.add(run_path)
                chunk = []
        if chunk:
            run_path = temporary_dir / f"run-{run_index:08d}.jsonl"
            _write_sorted_run(chunk, run_path)
            run_accumulator.add(run_path)
        run_paths = _reduce_sorted_runs(
            run_accumulator.pending_paths(),
            temporary_dir,
            merge_fan_in,
        )

        output = AtomicJsonlShard(output_path)
        try:
            previous_key: str | None = None
            has_previous_key = False
            for key, _ordinal, record in _iter_merged_runs(run_paths):
                if has_previous_key and key == previous_key:
                    continue
                output.write(record)
                previous_key = key
                has_previous_key = True
            return output.commit()
        except BaseException:
            output.abort()
            raise

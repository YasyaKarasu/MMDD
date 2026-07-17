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
                    updated_at REAL NOT NULL
                )
                """
            )
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
                connection.execute(
                    f"""
                    UPDATE jobs
                    SET status = 'leased', owner = ?, lease_expires = ?,
                        updated_at = ?
                    WHERE job_id IN ({placeholders})
                    """,
                    (owner, lease_expires, now, *job_ids),
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
    ) -> None:
        if status not in {"success", "terminal", "retryable"}:
            raise ValueError(f"invalid finish status: {status}")
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, owner = NULL,
                    lease_expires = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    None if result is None else json.dumps(result, ensure_ascii=False),
                    time.time(),
                    job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(job_id)
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


def external_unique_jsonl(
    input_paths: Iterable[Path],
    output_path: Path,
    key_fn: Callable[[dict[str, Any]], Any],
    chunk_records: int,
) -> CompletedShard:
    """Externally sort JSONL records and keep the first record for each key."""
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wdc200k-unique-") as temporary:
        temporary_dir = Path(temporary)
        run_paths: list[Path] = []
        chunk: list[tuple[str, int, dict[str, Any]]] = []
        ordinal = 0
        for record in iter_jsonl_records(input_paths):
            chunk.append((_external_key(key_fn(record)), ordinal, record))
            ordinal += 1
            if len(chunk) >= chunk_records:
                run_path = temporary_dir / f"run-{len(run_paths):08d}.jsonl"
                _write_sorted_run(chunk, run_path)
                run_paths.append(run_path)
                chunk = []
        if chunk:
            run_path = temporary_dir / f"run-{len(run_paths):08d}.jsonl"
            _write_sorted_run(chunk, run_path)
            run_paths.append(run_path)

        output = AtomicJsonlShard(output_path)
        handles: list[TextIO] = []
        try:
            handles = [
                run_path.open("r", encoding="utf-8")
                for run_path in run_paths
            ]
            merged = heapq.merge(
                *(_iter_sorted_run(handle) for handle in handles),
                key=lambda item: (item[0], item[1]),
            )
            previous_key: str | None = None
            has_previous_key = False
            for key, _ordinal, record in merged:
                if has_previous_key and key == previous_key:
                    continue
                output.write(record)
                previous_key = key
                has_previous_key = True
            return output.commit()
        except BaseException:
            output.abort()
            raise
        finally:
            for handle in handles:
                handle.close()

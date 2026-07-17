"""Persistent, resumable model queues for the WDC 200K pipeline.

The module deliberately keeps model payloads and outcomes in SQLite until a
job set is complete.  This gives each model call an individually durable
commit while avoiding an in-memory asset/task index.  Complete outcomes are
then published as checksummed JSONL shards for the materializer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    from build_mm_joinability_dataset import (
        PROMPT_VERSION,
        ExtractionTask,
        extraction_cache_key,
        run_extraction_task_group,
    )
    from stage1_io import clean_text, stable_hash
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        SqliteJobStore,
        validate_completed_shard,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "build_mm_joinability_dataset",
        "stage1_io",
        "wdc200k_io",
    }:
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        from build_mm_joinability_dataset import (
            PROMPT_VERSION,
            ExtractionTask,
            extraction_cache_key,
            run_extraction_task_group,
        )
        from stage1_io import clean_text, stable_hash
        from wdc200k_io import (
            AtomicJsonlShard,
            CompletedShard,
            SqliteJobStore,
            validate_completed_shard,
        )
    finally:
        sys.path.remove(scripts_directory)


MODEL_QUEUE_SCHEMA_VERSION = "wdc200k-model-queues-v1"
MODEL_OUTPUT_SCHEMA_VERSION = "wdc200k-model-outputs-v1"
MODEL_POLICY_VERSION = "existing-extraction-semantics-v1"
_PREVIEW_LIMIT = 16
_ENQUEUE_BATCH_SIZE = 1_000


@dataclass(frozen=True)
class ModelJobInfo:
    job_id: str
    cache_key: str
    modality: str
    asset_fingerprint: str


@dataclass(frozen=True)
class ModelJobSet:
    database_path: Path
    input_fingerprint: str
    prompt_version: str
    text_fingerprint: str
    image_fingerprint: str
    text_kind: str
    image_kind: str
    text_tasks: int
    image_tasks: int
    jobs: tuple[ModelJobInfo, ...] = ()

    @property
    def total_tasks(self) -> int:
        return self.text_tasks + self.image_tasks

    def kind_for(self, modality: str) -> str:
        if modality == "text":
            return self.text_kind
        if modality == "image":
            return self.image_kind
        raise ValueError(f"unsupported model modality: {modality}")

    def fingerprint_for(self, modality: str) -> str:
        if modality == "text":
            return self.text_fingerprint
        if modality == "image":
            return self.image_fingerprint
        raise ValueError(f"unsupported model modality: {modality}")


@dataclass(frozen=True)
class ModelStageResult:
    output_root: Path
    manifest_path: Path
    extraction_paths: tuple[Path, ...]
    error_paths: tuple[Path, ...]
    text_total: int
    image_total: int
    success: int
    terminal: int
    pending: int
    leased: int
    complete: bool
    jobset: ModelJobSet


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return connection


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
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
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _initialize_tables(path: Path) -> None:
    with _connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_jobsets (
                fingerprint TEXT PRIMARY KEY,
                modality TEXT NOT NULL,
                job_kind TEXT NOT NULL UNIQUE,
                input_fingerprint TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_identity TEXT NOT NULL,
                policy_fingerprint TEXT NOT NULL,
                task_count INTEGER NOT NULL DEFAULT 0,
                enqueue_complete INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_job_members (
                jobset_fingerprint TEXT NOT NULL,
                job_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                asset_fingerprint TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY (jobset_fingerprint, job_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_jobset_pairs (
                identity TEXT PRIMARY KEY,
                input_fingerprint TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                text_fingerprint TEXT NOT NULL,
                image_fingerprint TEXT NOT NULL,
                text_kind TEXT NOT NULL,
                image_kind TEXT NOT NULL,
                text_tasks INTEGER NOT NULL,
                image_tasks INTEGER NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS model_members_job
            ON model_job_members(job_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_results (
                job_id TEXT PRIMARY KEY,
                jobset_fingerprint TEXT NOT NULL,
                modality TEXT NOT NULL,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS model_results_jobset
            ON model_results(jobset_fingerprint, status, job_id)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_cache (
                cache_key TEXT PRIMARY KEY,
                record_json TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )


def _model_identity(args: argparse.Namespace, modality: str) -> str:
    attribute = (
        "image_model_name" if modality == "image" else "text_model_name"
    )
    value = clean_text(getattr(args, attribute, ""))
    if not value:
        raise ValueError(f"{attribute} must not be empty")
    return value


def _modality_fingerprint(
    *,
    modality: str,
    input_fingerprint: str,
    prompt_version: str,
    model_identity: str,
    policy_fingerprint: str,
) -> str:
    return stable_hash(
        MODEL_QUEUE_SCHEMA_VERSION,
        modality,
        input_fingerprint,
        prompt_version,
        model_identity,
        policy_fingerprint,
        length=40,
    )


def _task_payload(
    record: dict[str, Any],
    *,
    order: int,
    args: argparse.Namespace,
    prompt_version: str,
    model_identity: str,
    policy_fingerprint: str,
    jobset_fingerprint: str,
) -> tuple[str, str, dict[str, Any]]:
    nested = record.get("extraction_task")
    task_record = nested if isinstance(nested, dict) else record
    asset_value = task_record.get("asset")
    asset = (
        dict(asset_value)
        if isinstance(asset_value, dict)
        else {
            key: value
            for key, value in record.items()
            if key
            not in {
                "candidate_attribute_names",
                "entity",
                "extraction_task",
                "source_table_id",
                "source_row_id",
                "entity_column_index",
                "entity_column_name",
            }
        }
    )
    asset_id = clean_text(asset.get("asset_id"))
    modality = clean_text(asset.get("asset_type"))
    if not asset_id:
        raise ValueError("model asset is missing asset_id")
    if modality not in {"text", "image"}:
        raise ValueError(f"unsupported model asset type: {modality!r}")
    entity_value = task_record.get("entity")
    entity = (
        dict(entity_value)
        if isinstance(entity_value, dict)
        else {
            "entity_id": clean_text(record.get("entity_id")),
            "wiki_title": (
                clean_text(record.get("entity_wiki_title"))
                or clean_text(record.get("wiki_title"))
            ),
            "cell_text": (
                clean_text(record.get("entity_text"))
                or clean_text(record.get("cell_text"))
                or clean_text(record.get("entity_wiki_title"))
            ),
            "entity_column_index": int(
                record.get("entity_column_index") or 0
            ),
            "entity_column_name": clean_text(
                record.get("entity_column_name")
            ),
        }
    )
    entity_id = clean_text(entity.get("entity_id"))
    if not entity_id:
        raise ValueError(f"model asset {asset_id!r} is missing entity_id")
    entity.setdefault("wiki_title", "")
    entity.setdefault("cell_text", "")
    candidates_value = task_record.get("candidate_attribute_names", [])
    if not isinstance(candidates_value, list):
        raise ValueError("candidate_attribute_names must be a list")
    candidates = [clean_text(value) for value in candidates_value]
    candidates = [value for value in candidates if value]
    legacy_cache_key = extraction_cache_key(
        asset_id=asset_id,
        entity_id=entity_id,
        candidate_attribute_names=candidates,
        asset_type=modality,
        args=args,
    )
    asset_fingerprint = _digest_json(asset)
    payload = {
        "order": int(task_record.get("order", order)),
        "cache_key": legacy_cache_key,
        "source_table_id": clean_text(
            task_record.get("source_table_id")
        ),
        "source_row_id": int(task_record.get("source_row_id") or 0),
        "entity_column_index": int(
            task_record.get("entity_column_index")
            or entity.get("entity_column_index")
            or 0
        ),
        "entity_column_name": (
            clean_text(task_record.get("entity_column_name"))
            or clean_text(entity.get("entity_column_name"))
        ),
        "entity": entity,
        "asset": asset,
        "candidate_attribute_names": candidates,
        "asset_fingerprint": asset_fingerprint,
        "prompt_version": prompt_version,
        "model_identity": model_identity,
        "modality": modality,
        "policy_fingerprint": policy_fingerprint,
        "jobset_fingerprint": jobset_fingerprint,
    }
    job_id = stable_hash(
        MODEL_QUEUE_SCHEMA_VERSION,
        legacy_cache_key,
        asset_fingerprint,
        prompt_version,
        model_identity,
        modality,
        policy_fingerprint,
        length=40,
    )
    return job_id, asset_fingerprint, payload


def enqueue_model_tasks(
    assets: Iterable[dict[str, Any]],
    store: SqliteJobStore,
    *,
    args: argparse.Namespace | None = None,
    input_fingerprint: str = "adhoc-model-input-v1",
    text_input_fingerprint: str | None = None,
    image_input_fingerprint: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    policy_fingerprint: str = MODEL_POLICY_VERSION,
) -> ModelJobSet:
    """Stream complete extraction payloads into isolated modality job sets."""
    if not input_fingerprint:
        raise ValueError("input_fingerprint must not be empty")
    args = args or argparse.Namespace(
        text_model_name="text",
        image_model_name="image",
    )
    _initialize_tables(store.path)
    input_by_kind = {
        "text": text_input_fingerprint or input_fingerprint,
        "image": image_input_fingerprint or input_fingerprint,
    }
    identities = {
        modality: _model_identity(args, modality)
        for modality in ("text", "image")
    }
    fingerprints = {
        modality: _modality_fingerprint(
            modality=modality,
            input_fingerprint=input_by_kind[modality],
            prompt_version=prompt_version,
            model_identity=identities[modality],
            policy_fingerprint=policy_fingerprint,
        )
        for modality in ("text", "image")
    }
    kinds = {
        modality: f"model-{modality}-{fingerprints[modality]}"
        for modality in ("text", "image")
    }
    now = time.time()
    with _connect(store.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for modality in ("text", "image"):
            connection.execute(
                """
                INSERT OR IGNORE INTO model_jobsets (
                    fingerprint, modality, job_kind, input_fingerprint,
                    prompt_version, model_identity, policy_fingerprint,
                    task_count, enqueue_complete, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
                """,
                (
                    fingerprints[modality],
                    modality,
                    kinds[modality],
                    input_by_kind[modality],
                    prompt_version,
                    identities[modality],
                    policy_fingerprint,
                    now,
                ),
            )
    previews: list[ModelJobInfo] = []
    buffered: list[
        tuple[str, str, dict[str, Any], str, str, str]
    ] = []

    def flush() -> None:
        if not buffered:
            return
        committed_at = time.time()
        with _connect(store.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for (
                modality,
                job_id,
                payload,
                asset_fingerprint,
                payload_digest,
                encoded,
            ) in buffered:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        job_id, kind, payload_json, status, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?)
                    """,
                    (
                        job_id,
                        kinds[modality],
                        encoded,
                        committed_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_job_members (
                        jobset_fingerprint, job_id, cache_key,
                        asset_fingerprint, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        fingerprints[modality],
                        job_id,
                        payload["cache_key"],
                        asset_fingerprint,
                        payload_digest,
                    ),
                )
        buffered.clear()

    for order, record in enumerate(assets):
        if not isinstance(record, dict):
            raise ValueError("model task input must be an object")
        status = clean_text(record.get("status"))
        if status and status != "success":
            continue
        nested = record.get("extraction_task")
        nested_asset = (
            nested.get("asset")
            if isinstance(nested, dict)
            and isinstance(nested.get("asset"), dict)
            else None
        )
        modality = clean_text(
            (nested_asset or record).get("asset_type")
        )
        if modality not in {"text", "image"}:
            continue
        job_id, asset_fingerprint, payload = _task_payload(
            record,
            order=order,
            args=args,
            prompt_version=prompt_version,
            model_identity=identities[modality],
            policy_fingerprint=policy_fingerprint,
            jobset_fingerprint=fingerprints[modality],
        )
        encoded = _canonical_json(payload)
        payload_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        buffered.append(
            (
                modality,
                job_id,
                payload,
                asset_fingerprint,
                payload_digest,
                encoded,
            )
        )
        if len(buffered) >= _ENQUEUE_BATCH_SIZE:
            flush()
        if len(previews) < _PREVIEW_LIMIT:
            previews.append(
                ModelJobInfo(
                    job_id=job_id,
                    cache_key=str(payload["cache_key"]),
                    modality=modality,
                    asset_fingerprint=asset_fingerprint,
                )
            )
    flush()
    counts: dict[str, int] = {}
    with _connect(store.path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for modality in ("text", "image"):
            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM model_job_members
                    WHERE jobset_fingerprint = ?
                    """,
                    (fingerprints[modality],),
                ).fetchone()[0]
            )
            counts[modality] = count
            connection.execute(
                """
                UPDATE model_jobsets
                SET task_count = ?, enqueue_complete = 1, updated_at = ?
                WHERE fingerprint = ?
                """,
                (count, time.time(), fingerprints[modality]),
            )
    result = ModelJobSet(
        database_path=store.path,
        input_fingerprint=input_fingerprint,
        prompt_version=prompt_version,
        text_fingerprint=fingerprints["text"],
        image_fingerprint=fingerprints["image"],
        text_kind=kinds["text"],
        image_kind=kinds["image"],
        text_tasks=counts["text"],
        image_tasks=counts["image"],
        jobs=tuple(previews),
    )
    with _connect(store.path) as connection:
        connection.execute(
            """
            INSERT INTO model_jobset_pairs (
                identity, input_fingerprint, prompt_version,
                text_fingerprint, image_fingerprint,
                text_kind, image_kind, text_tasks, image_tasks, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(identity) DO UPDATE SET
                text_tasks = excluded.text_tasks,
                image_tasks = excluded.image_tasks,
                updated_at = excluded.updated_at
            """,
            (
                _manifest_identity(result),
                result.input_fingerprint,
                result.prompt_version,
                result.text_fingerprint,
                result.image_fingerprint,
                result.text_kind,
                result.image_kind,
                result.text_tasks,
                result.image_tasks,
                time.time(),
            ),
        )
    return result


def _payload_to_task(payload: dict[str, Any]) -> ExtractionTask:
    return ExtractionTask(
        order=int(payload["order"]),
        cache_key=str(payload["cache_key"]),
        source_table_id=str(payload["source_table_id"]),
        source_row_id=int(payload["source_row_id"]),
        entity_column_index=int(payload["entity_column_index"]),
        entity_column_name=str(payload["entity_column_name"]),
        entity=dict(payload["entity"]),
        asset=dict(payload["asset"]),
        candidate_attribute_names=list(
            payload["candidate_attribute_names"]
        ),
    )


def _decode_checked(encoded: str, expected_digest: str) -> dict[str, Any]:
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if digest != expected_digest:
        raise ValueError("stored model record checksum mismatch")
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("stored model record is not an object")
    return payload


class _PersistentCache:
    def __init__(self, database_path: Path, delegate: Any = None) -> None:
        self.database_path = database_path
        self.delegate = delegate
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        if self.delegate is not None:
            record = self.delegate.get(cache_key)
            if record is not None:
                return record
        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT record_json, record_sha256
                FROM model_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return _decode_checked(
            str(row["record_json"]),
            str(row["record_sha256"]),
        )

    def put(self, cache_key: str, record: dict[str, Any]) -> None:
        encoded = _canonical_json(record)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with self._lock:
            if self.delegate is not None:
                self.delegate.put(cache_key, record)
            with _connect(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO model_cache (
                        cache_key, record_json, record_sha256, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        record_json = excluded.record_json,
                        record_sha256 = excluded.record_sha256,
                        updated_at = excluded.updated_at
                    """,
                    (cache_key, encoded, digest, time.time()),
                )


def _put_result(
    database_path: Path,
    *,
    job_id: str,
    jobset_fingerprint: str,
    modality: str,
    status: str,
    record: dict[str, Any],
) -> None:
    encoded = _canonical_json(record)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT record_sha256, status
            FROM model_results
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if existing is not None and (
            str(existing["record_sha256"]) != digest
            or str(existing["status"]) != status
        ):
            raise ValueError(f"conflicting durable model result: {job_id}")
        connection.execute(
            """
            INSERT OR IGNORE INTO model_results (
                job_id, jobset_fingerprint, modality, status,
                record_json, record_sha256, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                jobset_fingerprint,
                modality,
                status,
                encoded,
                digest,
                time.time(),
            ),
        )


def _repair_durable_results(
    database_path: Path,
    jobset: ModelJobSet,
) -> int:
    repaired = 0
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for modality in ("text", "image"):
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = (
                        SELECT model_results.status
                        FROM model_results
                        WHERE model_results.job_id = jobs.job_id
                    ),
                    result_json = (
                        SELECT model_results.record_json
                        FROM model_results
                        WHERE model_results.job_id = jobs.job_id
                    ),
                    owner = NULL, lease_expires = NULL, lease_id = NULL,
                    updated_at = ?
                WHERE kind = ?
                  AND status NOT IN ('success', 'terminal')
                  AND EXISTS (
                        SELECT 1 FROM model_results
                        WHERE model_results.job_id = jobs.job_id
                    )
                """,
                (time.time(), jobset.kind_for(modality)),
            )
            repaired += max(0, int(cursor.rowcount))
    return repaired


def _repair_cached_leases(
    database_path: Path,
    jobset: ModelJobSet,
    cache: _PersistentCache,
) -> int:
    repaired = 0
    with _connect(database_path) as connection:
        for modality in ("text", "image"):
            cursor = connection.execute(
                """
                SELECT job_id, payload_json
                FROM jobs
                WHERE kind = ? AND status = 'leased'
                ORDER BY job_id
                """,
                (jobset.kind_for(modality),),
            )
            for row in cursor:
                payload = json.loads(str(row["payload_json"]))
                cached = cache.get(str(payload["cache_key"]))
                if cached is None:
                    continue
                _put_result(
                    database_path,
                    job_id=str(row["job_id"]),
                    jobset_fingerprint=str(
                        payload["jobset_fingerprint"]
                    ),
                    modality=modality,
                    status="success",
                    record=cached,
                )
                repaired += 1
    if repaired:
        _repair_durable_results(database_path, jobset)
    return repaired


def _extend_leases(
    database_path: Path,
    jobs: list[Any],
    *,
    owner: str,
    lease_seconds: float,
) -> int:
    renewed = 0
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        for job in jobs:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET lease_expires = ?, updated_at = ?
                WHERE job_id = ? AND status = 'leased'
                  AND owner = ? AND lease_id = ?
                  AND lease_expires > ?
                """,
                (
                    now + lease_seconds,
                    now,
                    job.job_id,
                    owner,
                    job.lease_id,
                    now,
                ),
            )
            renewed += int(cursor.rowcount)
    return renewed


class _LeaseHeartbeat:
    def __init__(
        self,
        database_path: Path,
        jobs: list[Any],
        *,
        owner: str,
        lease_seconds: float,
        interval: float,
    ) -> None:
        self.database_path = database_path
        self.jobs = jobs
        self.owner = owner
        self.lease_seconds = lease_seconds
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def __enter__(self) -> "_LeaseHeartbeat":
        def heartbeat() -> None:
            while not self._stop.wait(self.interval):
                try:
                    _extend_leases(
                        self.database_path,
                        self.jobs,
                        owner=self.owner,
                        lease_seconds=self.lease_seconds,
                    )
                except BaseException as error:
                    self.error = error
                    self._stop.set()

        self._thread = threading.Thread(
            target=heartbeat,
            name=f"model-lease-heartbeat-{self.owner}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval * 2))
        if self.error is not None and _exc[0] is None:
            raise RuntimeError("model lease heartbeat failed") from self.error


def _finish_safely(
    store: SqliteJobStore,
    job: Any,
    *,
    status: str,
    record: dict[str, Any],
) -> None:
    try:
        store.finish(
            job.job_id,
            status=status,
            result=record,
            owner=job.owner,
            lease_id=job.lease_id,
        )
    except RuntimeError:
        with _connect(store.path) as connection:
            row = connection.execute(
                "SELECT status FROM jobs WHERE job_id = ?",
                (job.job_id,),
            ).fetchone()
        if row is None or str(row["status"]) != status:
            raise


def _process_claimed_group(
    store: SqliteJobStore,
    extractor: Any,
    claimed: list[Any],
    *,
    modality: str,
    cache: _PersistentCache,
    workers: int,
    owner: str,
    lease_seconds: float,
    heartbeat_seconds: float,
    after_result_write: (
        Callable[[str, dict[str, Any]], None] | None
    ),
    after_cache_write: (
        Callable[[str, dict[str, Any]], None] | None
    ),
) -> int:
    model_jobs: list[Any] = []
    task_by_key: dict[str, Any] = {}
    job_by_key: dict[str, Any] = {}
    handled = 0
    for job in claimed:
        payload = job.payload
        cached = cache.get(str(payload["cache_key"]))
        if cached is None:
            model_jobs.append(job)
            task = _payload_to_task(payload)
            task_by_key[task.cache_key] = task
            job_by_key[task.cache_key] = job
            continue
        _put_result(
            store.path,
            job_id=job.job_id,
            jobset_fingerprint=str(payload["jobset_fingerprint"]),
            modality=modality,
            status="success",
            record=cached,
        )
        if after_result_write is not None:
            after_result_write(job.job_id, cached)
        _finish_safely(store, job, status="success", record=cached)
        handled += 1
    if not model_jobs:
        return handled
    if extractor is None:
        for job in model_jobs:
            store.finish(
                job.job_id,
                status="retryable",
                result={"reason": "model extractor is unavailable"},
                owner=job.owner,
                lease_id=job.lease_id,
            )
        raise RuntimeError(
            "model analysis is required but no extractor was provided"
        )

    def commit_record(cache_key: str, record: dict[str, Any]) -> None:
        nonlocal handled
        job = job_by_key[cache_key]
        payload = job.payload
        error = clean_text(record.get("error"))
        status = "terminal" if error else "success"
        if status == "success":
            cache.put(cache_key, record)
            if after_cache_write is not None:
                after_cache_write(job.job_id, record)
        _put_result(
            store.path,
            job_id=job.job_id,
            jobset_fingerprint=str(payload["jobset_fingerprint"]),
            modality=modality,
            status=status,
            record=record,
        )
        if after_result_write is not None:
            after_result_write(job.job_id, record)
        _finish_safely(store, job, status=status, record=record)
        handled += 1

    with _LeaseHeartbeat(
        store.path,
        model_jobs,
        owner=owner,
        lease_seconds=lease_seconds,
        interval=heartbeat_seconds,
    ):
        run_extraction_task_group(
            extractor=extractor,
            tasks=[task_by_key[job.payload["cache_key"]] for job in model_jobs],
            workers=workers,
            on_record=commit_record,
        )
    return handled


def _job_snapshot(
    database_path: Path,
    jobset: ModelJobSet,
) -> dict[str, int]:
    snapshot = {
        "total": 0,
        "success": 0,
        "terminal": 0,
        "pending": 0,
        "retryable": 0,
        "leased": 0,
    }
    with _connect(database_path) as connection:
        for modality in ("text", "image"):
            rows = connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM jobs
                WHERE kind = ?
                GROUP BY status
                """,
                (jobset.kind_for(modality),),
            ).fetchall()
            for row in rows:
                status = str(row["status"])
                count = int(row["count"])
                snapshot["total"] += count
                snapshot[status] = snapshot.get(status, 0) + count
    return snapshot


def _relative_shard(
    shard: CompletedShard,
    path: Path,
    root: Path,
) -> CompletedShard:
    return CompletedShard(
        path=path.relative_to(root).as_posix(),
        records=shard.records,
        bytes=shard.bytes,
        sha256=shard.sha256,
    )


def _shard_payload(shard: CompletedShard) -> dict[str, Any]:
    return {
        "path": shard.path,
        "records": shard.records,
        "bytes": shard.bytes,
        "sha256": shard.sha256,
    }


def _completed_from_payload(payload: dict[str, Any]) -> CompletedShard:
    return CompletedShard(
        path=str(payload["path"]),
        records=int(payload["records"]),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
    )


def _manifest_identity(jobset: ModelJobSet) -> str:
    return stable_hash(
        MODEL_OUTPUT_SCHEMA_VERSION,
        jobset.text_fingerprint,
        jobset.image_fingerprint,
        length=40,
    )


def _latest_jobset(database_path: Path) -> ModelJobSet:
    _initialize_tables(database_path)
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM model_jobset_pairs
            ORDER BY updated_at DESC, identity DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        raise ValueError("no completed model job set has been enqueued")
    return ModelJobSet(
        database_path=database_path,
        input_fingerprint=str(row["input_fingerprint"]),
        prompt_version=str(row["prompt_version"]),
        text_fingerprint=str(row["text_fingerprint"]),
        image_fingerprint=str(row["image_fingerprint"]),
        text_kind=str(row["text_kind"]),
        image_kind=str(row["image_kind"]),
        text_tasks=int(row["text_tasks"]),
        image_tasks=int(row["image_tasks"]),
    )


def _load_valid_manifest(
    path: Path,
    *,
    output_root: Path,
    jobset: ModelJobSet,
) -> ModelStageResult | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != _manifest_identity(jobset):
        raise ValueError("model output manifest identity mismatch")
    if payload.get("complete") is not True:
        return None
    extraction_shards = [
        _completed_from_payload(item)
        for item in payload.get("extraction_shards", [])
    ]
    error_shards = [
        _completed_from_payload(item)
        for item in payload.get("error_shards", [])
    ]
    if not all(
        validate_completed_shard(shard, output_root)
        for shard in (*extraction_shards, *error_shards)
    ):
        raise ValueError("model output shard checksum validation failed")
    counts = payload["counts"]
    return ModelStageResult(
        output_root=output_root,
        manifest_path=path,
        extraction_paths=tuple(
            output_root / shard.path for shard in extraction_shards
        ),
        error_paths=tuple(
            output_root / shard.path for shard in error_shards
        ),
        text_total=int(counts["text_total"]),
        image_total=int(counts["image_total"]),
        success=int(counts["success"]),
        terminal=int(counts["terminal"]),
        pending=0,
        leased=0,
        complete=True,
        jobset=jobset,
    )


def _publish_outputs(
    database_path: Path,
    *,
    output_root: Path,
    jobset: ModelJobSet,
    records_per_shard: int,
) -> ModelStageResult:
    identity = _manifest_identity(jobset)
    stage_root = output_root / identity
    manifest_path = stage_root / "model-stage-manifest.json"
    resumed = _load_valid_manifest(
        manifest_path,
        output_root=stage_root,
        jobset=jobset,
    )
    if resumed is not None:
        return resumed
    stage_root.mkdir(parents=True, exist_ok=True)
    lock_path = stage_root / ".publish.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        resumed = _load_valid_manifest(
            manifest_path,
            output_root=stage_root,
            jobset=jobset,
        )
        if resumed is not None:
            return resumed

        extraction_shards: list[CompletedShard] = []
        error_shards: list[CompletedShard] = []
        writers: dict[str, AtomicJsonlShard | None] = {
            "success": None,
            "terminal": None,
        }
        counts = {"success": 0, "terminal": 0}

        def commit_writer(status: str) -> None:
            writer = writers[status]
            if writer is None:
                return
            completed = writer.commit()
            root_name = (
                "attribute_extractions"
                if status == "success"
                else "model_attribute_errors"
            )
            relative = _relative_shard(
                completed,
                stage_root / root_name / completed.path,
                stage_root,
            )
            (
                extraction_shards
                if status == "success"
                else error_shards
            ).append(relative)
            writers[status] = None

        try:
            with _connect(database_path) as connection:
                for modality in ("text", "image"):
                    cursor = connection.execute(
                        """
                        SELECT status, record_json, record_sha256
                        FROM model_results
                        WHERE jobset_fingerprint = ?
                        ORDER BY job_id
                        """,
                        (jobset.fingerprint_for(modality),),
                    )
                    for row in cursor:
                        status = str(row["status"])
                        record = _decode_checked(
                            str(row["record_json"]),
                            str(row["record_sha256"]),
                        )
                        if writers[status] is None:
                            root_name = (
                                "attribute_extractions"
                                if status == "success"
                                else "model_attribute_errors"
                            )
                            index = (
                                len(extraction_shards)
                                if status == "success"
                                else len(error_shards)
                            )
                            writers[status] = AtomicJsonlShard(
                                stage_root
                                / root_name
                                / f"part-{index:05d}.jsonl"
                            )
                        writers[status].write(record)
                        counts[status] += 1
                        if counts[status] % records_per_shard == 0:
                            commit_writer(status)
            commit_writer("success")
            commit_writer("terminal")
        except BaseException:
            for writer in writers.values():
                if writer is not None:
                    writer.abort()
            raise

        payload = {
            "stage": "wdc200k_model_outputs",
            "schema_version": MODEL_OUTPUT_SCHEMA_VERSION,
            "identity": identity,
            "jobsets": {
                "text": jobset.text_fingerprint,
                "image": jobset.image_fingerprint,
            },
            "extraction_shards": [
                _shard_payload(shard) for shard in extraction_shards
            ],
            "error_shards": [
                _shard_payload(shard) for shard in error_shards
            ],
            "counts": {
                "text_total": jobset.text_tasks,
                "image_total": jobset.image_tasks,
                **counts,
            },
            "complete": True,
        }
        _atomic_json(manifest_path, payload)
    published = _load_valid_manifest(
        manifest_path,
        output_root=stage_root,
        jobset=jobset,
    )
    if published is None:
        raise RuntimeError("published model output manifest is incomplete")
    return published


def run_model_stage(
    store: SqliteJobStore,
    extractor: Any,
    *,
    jobset: ModelJobSet | None = None,
    stop_after: int | None = None,
    group_size: int = 32,
    workers: int = 1,
    workers_by_kind: dict[str, int] | None = None,
    owner: str | None = None,
    lease_seconds: float = 3600.0,
    heartbeat_seconds: float | None = None,
    cache: Any = None,
    output_root: Path | None = None,
    records_per_shard: int = 10_000,
    after_result_write: (
        Callable[[str, dict[str, Any]], None] | None
    ) = None,
    after_cache_write: (
        Callable[[str, dict[str, Any]], None] | None
    ) = None,
    start_marker: Path | None = None,
    ready_marker: Path | None = None,
    network_manifests: Iterable[Path] = (),
    assets_manifest: Path | None = None,
    ready_timeout_seconds: float | None = None,
    text_done_marker: Path | None = None,
    image_done_marker: Path | None = None,
    run_fingerprint: str = "",
) -> ModelStageResult:
    """Run bounded claims, durably committing each result before job finish."""
    jobset = jobset or _latest_jobset(store.path)
    if jobset.database_path.resolve() != store.path.resolve():
        raise ValueError("model job set belongs to a different job store")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be positive")
    if stop_after is not None and stop_after < 0:
        raise ValueError("stop_after must be non-negative")
    heartbeat_seconds = (
        min(lease_seconds / 3.0, 30.0)
        if heartbeat_seconds is None
        else heartbeat_seconds
    )
    if heartbeat_seconds <= 0 or heartbeat_seconds >= lease_seconds:
        raise ValueError(
            "heartbeat_seconds must be positive and less than lease_seconds"
        )
    output_root = Path(
        output_root or store.path.parent / "model_outputs"
    )
    owner = owner or f"model-worker-{os.getpid()}-{uuid.uuid4().hex}"
    _initialize_tables(store.path)
    if start_marker is not None:
        if assets_manifest is None:
            raise ValueError(
                "assets_manifest is required with start_marker"
            )
        write_model_start_marker(
            start_marker,
            jobset,
            network_manifests=network_manifests,
            assets_manifest=assets_manifest,
            run_fingerprint=run_fingerprint,
        )
        if jobset.total_tasks and ready_marker is not None:
            wait_for_model_ready_marker(
                ready_marker,
                run_fingerprint=run_fingerprint,
                text_jobset_fingerprint=jobset.text_fingerprint,
                image_jobset_fingerprint=jobset.image_fingerprint,
                timeout_seconds=ready_timeout_seconds,
            )
    persistent_cache = _PersistentCache(store.path, delegate=cache)
    _repair_durable_results(store.path, jobset)
    _repair_cached_leases(store.path, jobset, persistent_cache)
    processed = 0
    for modality in ("text", "image"):
        while stop_after is None or processed < stop_after:
            allowance = (
                group_size
                if stop_after is None
                else min(group_size, stop_after - processed)
            )
            if allowance <= 0:
                break
            claimed = store.claim(
                jobset.kind_for(modality),
                limit=allowance,
                owner=owner,
                lease_seconds=lease_seconds,
            )
            if not claimed:
                break
            processed += _process_claimed_group(
                store,
                extractor,
                claimed,
                modality=modality,
                cache=persistent_cache,
                workers=max(
                    1,
                    int(
                        (workers_by_kind or {}).get(modality, workers)
                    ),
                ),
                owner=owner,
                lease_seconds=lease_seconds,
                heartbeat_seconds=heartbeat_seconds,
                after_result_write=after_result_write,
                after_cache_write=after_cache_write,
            )
        snapshot = _job_snapshot(store.path, jobset)
        modality_complete = _kind_is_complete(
            store.path,
            jobset.kind_for(modality),
        )
        marker = (
            text_done_marker if modality == "text" else image_done_marker
        )
        if modality_complete and marker is not None:
            write_model_done_marker(
                marker,
                model_kind=modality,
                task_count=(
                    jobset.text_tasks
                    if modality == "text"
                    else jobset.image_tasks
                ),
                jobset_fingerprint=jobset.fingerprint_for(modality),
                run_fingerprint=run_fingerprint,
            )
    snapshot = _job_snapshot(store.path, jobset)
    complete = (
        snapshot["total"] == jobset.total_tasks
        and snapshot["success"] + snapshot["terminal"]
        == jobset.total_tasks
    )
    if complete:
        return _publish_outputs(
            store.path,
            output_root=output_root,
            jobset=jobset,
            records_per_shard=records_per_shard,
        )
    return ModelStageResult(
        output_root=output_root,
        manifest_path=(
            output_root
            / _manifest_identity(jobset)
            / "model-stage-manifest.json"
        ),
        extraction_paths=(),
        error_paths=(),
        text_total=jobset.text_tasks,
        image_total=jobset.image_tasks,
        success=snapshot["success"],
        terminal=snapshot["terminal"],
        pending=snapshot["pending"] + snapshot["retryable"],
        leased=snapshot["leased"],
        complete=False,
        jobset=jobset,
    )


def _kind_is_complete(database_path: Path, kind: str) -> bool:
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN status IN ('success', 'terminal')
                            THEN 1 ELSE 0 END) AS completed
            FROM jobs WHERE kind = ?
            """,
            (kind,),
        ).fetchone()
    return int(row["total"] or 0) == int(row["completed"] or 0)


def validate_model_stage(result: ModelStageResult) -> bool:
    if not result.complete:
        return False
    try:
        loaded = _load_valid_manifest(
            result.manifest_path,
            output_root=result.manifest_path.parent,
            jobset=result.jobset,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return loaded is not None


def _validated_complete_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"upstream manifest is unreadable: {path}") from error
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"upstream manifest is not complete: {path}")
    return payload


def _validate_declared_shards(
    path: Path,
    payload: dict[str, Any],
    *,
    field: str,
) -> None:
    declared = payload.get(field)
    if declared is None:
        return
    if not isinstance(declared, list):
        raise ValueError(f"upstream manifest has invalid {field}: {path}")
    try:
        shards = [_completed_from_payload(item) for item in declared]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"upstream manifest has invalid {field}: {path}"
        ) from error
    if not all(validate_completed_shard(shard, path.parent) for shard in shards):
        raise ValueError(f"upstream manifest shard validation failed: {path}")


def write_model_start_marker(
    path: Path,
    jobset: ModelJobSet,
    *,
    network_manifests: Iterable[Path],
    assets_manifest: Path,
    run_fingerprint: str,
) -> None:
    """Publish exact staged counts only after upstream completion is proven."""
    if not run_fingerprint:
        raise ValueError("run_fingerprint must not be empty")
    network_paths = [Path(value) for value in network_manifests]
    upstream = []
    for manifest_path in network_paths:
        payload = _validated_complete_manifest(manifest_path)
        _validate_declared_shards(
            manifest_path,
            payload,
            field="completed_shards",
        )
        upstream.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256_path(manifest_path),
            }
        )
    assets_path = Path(assets_manifest)
    assets_payload = _validated_complete_manifest(assets_path)
    if assets_payload.get("stage") == "wdc200k_asset_materialization":
        _validate_declared_shards(
            assets_path,
            assets_payload,
            field="bridge_asset_shards",
        )
        _validate_declared_shards(
            assets_path,
            assets_payload,
            field="table_asset_link_shards",
        )
    upstream.append(
        {
            "path": str(assets_path),
            "sha256": _sha256_path(assets_path),
        }
    )
    _atomic_json(
        Path(path),
        {
            "status": "model_cache_ready_to_start",
            "run_fingerprint": run_fingerprint,
            "text_jobset_fingerprint": jobset.text_fingerprint,
            "image_jobset_fingerprint": jobset.image_fingerprint,
            "text_task_count": jobset.text_tasks,
            "image_task_count": jobset.image_tasks,
            "upstream_manifests": upstream,
            "timestamp": time.time(),
        },
    )


def write_model_done_marker(
    path: Path,
    *,
    model_kind: str,
    task_count: int,
    jobset_fingerprint: str,
    run_fingerprint: str,
) -> None:
    if model_kind not in {"text", "image"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    _atomic_json(
        Path(path),
        {
            "status": f"{model_kind}_model_cache_precomputed",
            "model_kind": model_kind,
            "task_count": task_count,
            f"{model_kind}_task_count": task_count,
            "jobset_fingerprint": jobset_fingerprint,
            "run_fingerprint": run_fingerprint,
            "timestamp": time.time(),
        },
    )


def marker_matches(
    path: Path,
    *,
    run_fingerprint: str,
    jobset_fingerprint: str | None = None,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("run_fingerprint") != run_fingerprint:
        return False
    if (
        jobset_fingerprint is not None
        and payload.get("jobset_fingerprint") != jobset_fingerprint
    ):
        return False
    if (
        text_jobset_fingerprint is not None
        and payload.get("text_jobset_fingerprint")
        != text_jobset_fingerprint
    ):
        return False
    if (
        image_jobset_fingerprint is not None
        and payload.get("image_jobset_fingerprint")
        != image_jobset_fingerprint
    ):
        return False
    return True


def wait_for_model_ready_marker(
    path: Path,
    *,
    run_fingerprint: str,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> None:
    started = time.monotonic()
    while not marker_matches(
        Path(path),
        run_fingerprint=run_fingerprint,
        text_jobset_fingerprint=text_jobset_fingerprint,
        image_jobset_fingerprint=image_jobset_fingerprint,
    ):
        if (
            timeout_seconds is not None
            and time.monotonic() - started > timeout_seconds
        ):
            raise RuntimeError(f"timed out waiting for ready marker: {path}")
        time.sleep(poll_seconds)


def iter_assets_from_materialization_manifest(
    manifest_path: Path,
) -> Iterator[dict[str, Any]]:
    """Validate Task-5 output and stream only canonical successful assets."""
    manifest_path = Path(manifest_path)
    payload = _validated_complete_manifest(manifest_path)
    if payload.get("stage") != "wdc200k_asset_materialization":
        raise ValueError("not a WDC asset materialization manifest")
    root = manifest_path.parent
    shards = [
        _completed_from_payload(item)
        for item in payload.get("bridge_asset_shards", [])
    ]
    if not all(validate_completed_shard(shard, root) for shard in shards):
        raise ValueError("asset materialization shard validation failed")
    for shard in shards:
        path = root / shard.path
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("bridge asset record is not an object")
                status = clean_text(record.get("status"))
                if status and status != "success":
                    continue
                if record.get("asset_type") in {"text", "image"}:
                    yield record


def enqueue_model_tasks_from_manifest(
    manifest_path: Path,
    store: SqliteJobStore,
    *,
    args: argparse.Namespace | None = None,
    input_fingerprint: str | None = None,
    task_factory: (
        Callable[
            [dict[str, Any]],
            dict[str, Any] | Iterable[dict[str, Any]],
        ]
        | None
    ) = None,
    text_input_fingerprint: str | None = None,
    image_input_fingerprint: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    policy_fingerprint: str = MODEL_POLICY_VERSION,
) -> ModelJobSet:
    """Validate Task 5 and enqueue its assets without loading all of them."""
    manifest_path = Path(manifest_path)

    def extraction_inputs() -> Iterator[dict[str, Any]]:
        for current_asset in iter_assets_from_materialization_manifest(
            manifest_path
        ):
            if task_factory is None:
                yield current_asset
                continue
            produced = task_factory(current_asset)
            if isinstance(produced, dict):
                yield produced
            else:
                yield from produced

    return enqueue_model_tasks(
        extraction_inputs(),
        store,
        args=args,
        input_fingerprint=(
            input_fingerprint or _sha256_path(manifest_path)
        ),
        text_input_fingerprint=text_input_fingerprint,
        image_input_fingerprint=image_input_fingerprint,
        prompt_version=prompt_version,
        policy_fingerprint=policy_fingerprint,
    )

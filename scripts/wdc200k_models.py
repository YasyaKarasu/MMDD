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
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    from build_mm_joinability_dataset import (
        PROMPT_VERSION,
        ExtractionTask,
        collect_table_extraction_tasks,
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
            collect_table_extraction_tasks,
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
MODEL_PARSER_SCHEMA_VERSION = "connection-evidence-parser-v1"
_PREVIEW_LIMIT = 16
_ENQUEUE_BATCH_SIZE = 1_000


@dataclass(frozen=True)
class ModelJobInfo:
    job_id: str
    cache_key: str
    model_call_key: str
    modality: str
    asset_fingerprint: str
    entity_prompt_fingerprint: str


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


@dataclass(frozen=True)
class AdaptedModelTasks:
    output_root: Path
    task_paths: tuple[Path, ...]
    error_paths: tuple[Path, ...]
    manifest_path: Path
    input_fingerprint: str
    tasks: int
    errors: int


@dataclass(frozen=True)
class AssetStageBarrier:
    """Authoritative Task-5 identity supplied by the stage orchestrator."""

    fingerprint: dict[str, Any]
    bridge_assets: int
    table_asset_links: int

    def __post_init__(self) -> None:
        if not isinstance(self.fingerprint, dict):
            raise TypeError("Task-5 barrier fingerprint must be an object")
        object.__setattr__(
            self,
            "fingerprint",
            json.loads(_canonical_json(self.fingerprint)),
        )
        if self.bridge_assets < 0 or self.table_asset_links < 0:
            raise ValueError("Task-5 barrier counts must be non-negative")


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
                membership_digest TEXT NOT NULL DEFAULT '',
                enqueue_complete INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL
            )
            """
        )
        jobset_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(model_jobsets)"
            )
        }
        if "membership_digest" not in jobset_columns:
            connection.execute(
                """
                ALTER TABLE model_jobsets
                ADD COLUMN membership_digest TEXT NOT NULL DEFAULT ''
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
                commit_owner TEXT NOT NULL,
                commit_lease_id TEXT NOT NULL,
                commit_lease_expires REAL NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0,
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
        result_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(model_results)"
            )
        }
        for name, definition in (
            ("commit_owner", "TEXT NOT NULL DEFAULT ''"),
            ("commit_lease_id", "TEXT NOT NULL DEFAULT ''"),
            ("commit_lease_expires", "REAL NOT NULL DEFAULT 0"),
            ("committed", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in result_columns:
                connection.execute(
                    f"ALTER TABLE model_results ADD COLUMN {name} {definition}"
                )
        required_job_columns = {
            "job_id",
            "kind",
            "payload_json",
            "status",
            "result_json",
            "owner",
            "lease_expires",
            "lease_id",
            "updated_at",
        }
        actual_job_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(jobs)")
        }
        if not required_job_columns <= actual_job_columns:
            raise ValueError("Task-1 job schema is incompatible")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS model_call_cache (
                model_call_key TEXT PRIMARY KEY,
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
        MODEL_PARSER_SCHEMA_VERSION,
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
    entity_prompt_fingerprint = _digest_json(
        {
            "entity": entity,
            "candidate_attribute_names": candidates,
        }
    )
    model_call_key = stable_hash(
        MODEL_QUEUE_SCHEMA_VERSION,
        legacy_cache_key,
        asset_fingerprint,
        prompt_version,
        model_identity,
        modality,
        policy_fingerprint,
        MODEL_PARSER_SCHEMA_VERSION,
        entity_prompt_fingerprint,
        _canonical_json(candidates),
        length=64,
    )
    payload = {
        "order": 0,
        "cache_key": legacy_cache_key,
        "model_call_key": model_call_key,
        "source_table_id": "",
        "source_row_id": 0,
        "entity_column_index": int(entity.get("entity_column_index") or 0),
        "entity_column_name": clean_text(
            entity.get("entity_column_name")
        ),
        "entity": entity,
        "asset": asset,
        "candidate_attribute_names": candidates,
        "asset_fingerprint": asset_fingerprint,
        "entity_prompt_fingerprint": entity_prompt_fingerprint,
        "prompt_version": prompt_version,
        "model_identity": model_identity,
        "modality": modality,
        "policy_fingerprint": policy_fingerprint,
        "parser_schema_version": MODEL_PARSER_SCHEMA_VERSION,
        "jobset_fingerprint": jobset_fingerprint,
    }
    job_id = f"{jobset_fingerprint}:{model_call_key}"
    payload["job_id"] = job_id
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
    staging = sqlite3.connect("")
    staging.row_factory = sqlite3.Row
    staging.execute(
        """
        CREATE TABLE incoming (
            job_id TEXT PRIMARY KEY,
            modality TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            model_call_key TEXT NOT NULL,
            asset_fingerprint TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    try:
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
            payload_digest = hashlib.sha256(
                encoded.encode("utf-8")
            ).hexdigest()
            existing = staging.execute(
                "SELECT * FROM incoming WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            values = (
                job_id,
                modality,
                str(payload["cache_key"]),
                str(payload["model_call_key"]),
                asset_fingerprint,
                payload_digest,
                encoded,
            )
            if existing is not None:
                if tuple(existing) != values:
                    raise ValueError(
                        f"conflicting incoming model member: {job_id}"
                    )
                continue
            staging.execute(
                """
                INSERT INTO incoming (
                    job_id, modality, cache_key, model_call_key,
                    asset_fingerprint, payload_sha256, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            if len(previews) < _PREVIEW_LIMIT:
                previews.append(
                    ModelJobInfo(
                        job_id=job_id,
                        cache_key=str(payload["cache_key"]),
                        model_call_key=str(payload["model_call_key"]),
                        modality=modality,
                        asset_fingerprint=asset_fingerprint,
                        entity_prompt_fingerprint=str(
                            payload["entity_prompt_fingerprint"]
                        ),
                    )
                )
        staging.commit()

        counts: dict[str, int] = {}
        membership_digests: dict[str, str] = {}
        for modality in ("text", "image"):
            digest = hashlib.sha256()
            count = 0
            for row in staging.execute(
                """
                SELECT job_id, cache_key, asset_fingerprint,
                       payload_sha256
                FROM incoming
                WHERE modality = ?
                ORDER BY job_id
                """,
                (modality,),
            ):
                digest.update(_canonical_json(tuple(row)).encode("utf-8"))
                digest.update(b"\n")
                count += 1
            counts[modality] = count
            membership_digests[modality] = digest.hexdigest()

        with _connect(store.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            frozen: dict[str, bool] = {}
            for modality in ("text", "image"):
                jobset_row = connection.execute(
                    """
                    SELECT task_count, membership_digest,
                           enqueue_complete
                    FROM model_jobsets
                    WHERE fingerprint = ?
                    """,
                    (fingerprints[modality],),
                ).fetchone()
                if jobset_row is None:
                    raise ValueError("model jobset metadata disappeared")
                frozen[modality] = (
                    int(jobset_row["enqueue_complete"]) == 1
                )
                if not frozen[modality]:
                    for durable in connection.execute(
                        """
                        SELECT members.job_id, members.cache_key,
                               members.asset_fingerprint,
                               members.payload_sha256, jobs.kind,
                               jobs.payload_json
                        FROM model_job_members AS members
                        JOIN jobs USING (job_id)
                        WHERE members.jobset_fingerprint = ?
                        ORDER BY members.job_id
                        """,
                        (fingerprints[modality],),
                    ):
                        incoming = staging.execute(
                            """
                            SELECT cache_key, asset_fingerprint,
                                   payload_sha256, payload_json
                            FROM incoming WHERE job_id = ?
                            """,
                            (str(durable["job_id"]),),
                        ).fetchone()
                        if incoming is None or tuple(durable)[1:] != (
                            str(incoming["cache_key"]),
                            str(incoming["asset_fingerprint"]),
                            str(incoming["payload_sha256"]),
                            kinds[modality],
                            str(incoming["payload_json"]),
                        ):
                            raise ValueError(
                                f"incomplete {modality} jobset "
                                "membership conflict"
                            )
                    continue
                existing_digest = hashlib.sha256()
                existing_count = 0
                for member in connection.execute(
                    """
                    SELECT job_id, cache_key, asset_fingerprint,
                           payload_sha256
                    FROM model_job_members
                    WHERE jobset_fingerprint = ?
                    ORDER BY job_id
                    """,
                    (fingerprints[modality],),
                ):
                    existing_digest.update(
                        _canonical_json(tuple(member)).encode("utf-8")
                    )
                    existing_digest.update(b"\n")
                    existing_count += 1
                actual_digest = existing_digest.hexdigest()
                if (
                    existing_count != counts[modality]
                    or int(jobset_row["task_count"]) != existing_count
                    or actual_digest != membership_digests[modality]
                    or (
                        clean_text(jobset_row["membership_digest"])
                        and str(jobset_row["membership_digest"])
                        != actual_digest
                    )
                ):
                    raise ValueError(
                        f"completed {modality} jobset membership conflict"
                    )
                for incoming in staging.execute(
                    """
                    SELECT job_id, cache_key, asset_fingerprint,
                           payload_sha256, payload_json
                    FROM incoming
                    WHERE modality = ?
                    ORDER BY job_id
                    """,
                    (modality,),
                ):
                    durable = connection.execute(
                        """
                        SELECT members.cache_key,
                               members.asset_fingerprint,
                               members.payload_sha256,
                               jobs.kind, jobs.payload_json
                        FROM model_job_members AS members
                        JOIN jobs USING (job_id)
                        WHERE members.jobset_fingerprint = ?
                          AND members.job_id = ?
                        """,
                        (
                            fingerprints[modality],
                            str(incoming["job_id"]),
                        ),
                    ).fetchone()
                    if durable is None or tuple(durable) != (
                        str(incoming["cache_key"]),
                        str(incoming["asset_fingerprint"]),
                        str(incoming["payload_sha256"]),
                        kinds[modality],
                        str(incoming["payload_json"]),
                    ):
                        raise ValueError(
                            f"completed {modality} jobset "
                            "membership conflict"
                        )

            committed_at = time.time()
            for modality in ("text", "image"):
                if frozen[modality]:
                    continue
                for incoming in staging.execute(
                    "SELECT * FROM incoming WHERE modality = ?",
                    (modality,),
                ):
                    existing_job = connection.execute(
                        """
                        SELECT kind, payload_json
                        FROM jobs WHERE job_id = ?
                        """,
                        (str(incoming["job_id"]),),
                    ).fetchone()
                    if existing_job is not None and tuple(existing_job) != (
                        kinds[modality],
                        str(incoming["payload_json"]),
                    ):
                        raise ValueError(
                            "conflicting durable model job: "
                            f"{incoming['job_id']}"
                        )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO jobs (
                            job_id, kind, payload_json, status, updated_at
                        ) VALUES (?, ?, ?, 'pending', ?)
                        """,
                        (
                            str(incoming["job_id"]),
                            kinds[modality],
                            str(incoming["payload_json"]),
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
                            str(incoming["job_id"]),
                            str(incoming["cache_key"]),
                            str(incoming["asset_fingerprint"]),
                            str(incoming["payload_sha256"]),
                        ),
                    )
                connection.execute(
                    """
                    UPDATE model_jobsets
                    SET task_count = ?, membership_digest = ?,
                        enqueue_complete = 1, updated_at = ?
                    WHERE fingerprint = ?
                    """,
                    (
                        counts[modality],
                        membership_digests[modality],
                        committed_at,
                        fingerprints[modality],
                    ),
                )
            for modality in ("text", "image"):
                if frozen[modality]:
                    connection.execute(
                        """
                        UPDATE model_jobsets
                        SET membership_digest = ?
                        WHERE fingerprint = ?
                          AND membership_digest = ''
                        """,
                        (
                            membership_digests[modality],
                            fingerprints[modality],
                        ),
                    )
    finally:
        staging.close()
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
        cache_key=str(payload["model_call_key"]),
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

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        model_call_key = str(payload["model_call_key"])
        with _connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT record_json, record_sha256
                FROM model_call_cache
                WHERE model_call_key = ?
                """,
                (model_call_key,),
            ).fetchone()
        if row is not None:
            record = _decode_checked(
                str(row["record_json"]),
                str(row["record_sha256"]),
            )
            if not _record_matches_payload(record, payload):
                raise ValueError(
                    "model-call cache provenance does not match its key"
                )
            return record
        if self.delegate is None:
            return None
        legacy = self.delegate.get(str(payload["cache_key"]))
        if legacy is None or not _record_matches_payload(legacy, payload):
            return None
        return dict(legacy)

    def put(
        self,
        payload: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        encoded = _canonical_json(record)
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        with _connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO model_call_cache (
                    model_call_key, record_json, record_sha256, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(model_call_key) DO UPDATE SET
                    record_json = excluded.record_json,
                    record_sha256 = excluded.record_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    str(payload["model_call_key"]),
                    encoded,
                    digest,
                    time.time(),
                ),
            )


def _record_matches_payload(
    record: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    expected = {
        "cache_key": str(payload["cache_key"]),
        "model_call_key": str(payload["model_call_key"]),
        "prompt_version": str(payload["prompt_version"]),
        "model_identity": str(payload["model_identity"]),
        "asset_fingerprint": str(payload["asset_fingerprint"]),
        "entity_prompt_fingerprint": str(
            payload["entity_prompt_fingerprint"]
        ),
        "asset_type": str(payload["modality"]),
        "modality": str(payload["modality"]),
        "policy_fingerprint": str(payload["policy_fingerprint"]),
        "parser_schema_version": str(payload["parser_schema_version"]),
        "candidate_attribute_names": list(
            payload["candidate_attribute_names"]
        ),
    }
    return all(record.get(key) == value for key, value in expected.items())


def _canonical_extraction_record(
    payload: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    canonical = {
        **record,
        "cache_key": str(payload["cache_key"]),
        "model_call_key": str(payload["model_call_key"]),
        "jobset_fingerprint": str(payload["jobset_fingerprint"]),
        "prompt_version": str(payload["prompt_version"]),
        "model_identity": str(payload["model_identity"]),
        "asset_fingerprint": str(payload["asset_fingerprint"]),
        "entity_prompt_fingerprint": str(
            payload["entity_prompt_fingerprint"]
        ),
        "job_id": str(payload["job_id"]),
        "asset_type": str(payload["modality"]),
        "modality": str(payload["modality"]),
        "policy_fingerprint": str(payload["policy_fingerprint"]),
        "parser_schema_version": str(payload["parser_schema_version"]),
        "candidate_attribute_names": list(
            payload["candidate_attribute_names"]
        ),
    }
    if not _record_matches_payload(canonical, payload):
        raise ValueError("canonical extraction provenance mismatch")
    return canonical


def _repair_durable_results(
    database_path: Path,
    jobset: ModelJobSet,
) -> int:
    repaired = 0
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for modality in ("text", "image"):
            rows = connection.execute(
                """
                SELECT jobs.job_id, jobs.owner, jobs.lease_id,
                       model_results.status,
                       model_results.record_json,
                       model_results.record_sha256,
                       jobs.payload_json
                FROM jobs
                JOIN model_results
                  ON model_results.job_id = jobs.job_id
                WHERE jobs.kind = ?
                  AND jobs.status = 'leased'
                  AND jobs.lease_expires <= ?
                  AND model_results.committed = 0
                  AND model_results.commit_owner = jobs.owner
                  AND model_results.commit_lease_id = jobs.lease_id
                """,
                (jobset.kind_for(modality), time.time()),
            ).fetchall()
            for row in rows:
                now = time.time()
                cursor = connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, result_json = ?, owner = NULL,
                        lease_expires = NULL, lease_id = NULL,
                        updated_at = ?
                    WHERE job_id = ? AND status = 'leased'
                      AND owner = ? AND lease_id = ?
                      AND lease_expires <= ?
                    """,
                    (
                        str(row["status"]),
                        str(row["record_json"]),
                        now,
                        str(row["job_id"]),
                        str(row["owner"]),
                        str(row["lease_id"]),
                        now,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                if str(row["status"]) == "success":
                    payload = json.loads(str(row["payload_json"]))
                    connection.execute(
                        """
                        INSERT INTO model_call_cache (
                            model_call_key, record_json,
                            record_sha256, updated_at
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(model_call_key) DO UPDATE SET
                            record_json = excluded.record_json,
                            record_sha256 = excluded.record_sha256,
                            updated_at = excluded.updated_at
                        """,
                        (
                            str(payload["model_call_key"]),
                            str(row["record_json"]),
                            str(row["record_sha256"]),
                            now,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE model_results
                    SET committed = 1, updated_at = ?
                    WHERE job_id = ? AND committed = 0
                      AND commit_owner = ? AND commit_lease_id = ?
                    """,
                    (
                        now,
                        str(row["job_id"]),
                        str(row["owner"]),
                        str(row["lease_id"]),
                    ),
                )
                repaired += 1
    return repaired


def _extend_leases(
    database_path: Path,
    jobs: list[Any],
    *,
    owner: str,
    lease_seconds: float,
) -> set[str]:
    renewed: set[str] = set()
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
            if cursor.rowcount == 1:
                renewed.add(str(job.job_id))
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
        self._lost: set[str] = set()
        self._lock = threading.Lock()

    def is_lost(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._lost

    def __enter__(self) -> "_LeaseHeartbeat":
        def heartbeat() -> None:
            while not self._stop.wait(self.interval):
                try:
                    renewed = _extend_leases(
                        self.database_path,
                        self.jobs,
                        owner=self.owner,
                        lease_seconds=self.lease_seconds,
                    )
                    expected = {str(job.job_id) for job in self.jobs}
                    if isinstance(renewed, int):
                        renewed_ids = expected if renewed == len(expected) else set()
                    else:
                        renewed_ids = set(renewed)
                    missing = expected - renewed_ids
                    if missing:
                        with self._lock:
                            self._lost.update(missing)
                except BaseException as error:
                    self.error = error
                    with self._lock:
                        self._lost.update(
                            str(job.job_id) for job in self.jobs
                        )
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


def _fenced_commit_model_record(
    database_path: Path,
    *,
    job: Any,
    expected_kind: str,
    payload: dict[str, Any],
    record: dict[str, Any],
    status: str,
    heartbeat: _LeaseHeartbeat,
    after_cache_write: (
        Callable[[str, dict[str, Any]], None] | None
    ),
    after_result_write: (
        Callable[[str, dict[str, Any]], None] | None
    ),
) -> bool:
    if heartbeat.is_lost(str(job.job_id)):
        return False
    canonical = _canonical_extraction_record(payload, record)
    encoded = _canonical_json(canonical)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    now = time.time()
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT kind, payload_json, status, owner, lease_id,
                   lease_expires
            FROM jobs
            WHERE job_id = ?
            """,
            (job.job_id,),
        ).fetchone()
        if (
            row is None
            or heartbeat.is_lost(str(job.job_id))
            or str(row["kind"]) != expected_kind
            or _canonical_json(json.loads(str(row["payload_json"])))
            != _canonical_json(payload)
            or str(row["status"]) != "leased"
            or str(row["owner"]) != str(job.owner)
            or str(row["lease_id"]) != str(job.lease_id)
            or float(row["lease_expires"] or 0.0) <= now
        ):
            connection.rollback()
            return False
        connection.execute(
            """
            INSERT INTO model_results (
                job_id, jobset_fingerprint, modality, status,
                record_json, record_sha256, commit_owner,
                commit_lease_id, commit_lease_expires,
                committed, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                jobset_fingerprint = excluded.jobset_fingerprint,
                modality = excluded.modality,
                status = excluded.status,
                record_json = excluded.record_json,
                record_sha256 = excluded.record_sha256,
                commit_owner = excluded.commit_owner,
                commit_lease_id = excluded.commit_lease_id,
                commit_lease_expires = excluded.commit_lease_expires,
                committed = 0,
                updated_at = excluded.updated_at
            """,
            (
                job.job_id,
                str(payload["jobset_fingerprint"]),
                str(payload["modality"]),
                status,
                encoded,
                digest,
                str(job.owner),
                str(job.lease_id),
                float(row["lease_expires"]),
                now,
            ),
        )
        connection.commit()
        if status == "success" and after_cache_write is not None:
            after_cache_write(str(job.job_id), canonical)
        if after_result_write is not None:
            after_result_write(str(job.job_id), canonical)
    if heartbeat.is_lost(str(job.job_id)):
        return False
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        finish_time = time.time()
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = ?, result_json = ?, owner = NULL,
                lease_expires = NULL, lease_id = NULL, updated_at = ?
            WHERE job_id = ? AND kind = ? AND status = 'leased'
              AND owner = ? AND lease_id = ? AND lease_expires > ?
              AND EXISTS (
                    SELECT 1
                    FROM model_results
                    WHERE model_results.job_id = jobs.job_id
                      AND model_results.commit_owner = ?
                      AND model_results.commit_lease_id = ?
                      AND model_results.record_sha256 = ?
                      AND model_results.committed = 0
              )
            """,
            (
                status,
                encoded,
                finish_time,
                job.job_id,
                expected_kind,
                job.owner,
                job.lease_id,
                finish_time,
                job.owner,
                job.lease_id,
                digest,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        if status == "success":
            connection.execute(
                """
                INSERT INTO model_call_cache (
                    model_call_key, record_json,
                    record_sha256, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(model_call_key) DO UPDATE SET
                    record_json = excluded.record_json,
                    record_sha256 = excluded.record_sha256,
                    updated_at = excluded.updated_at
                """,
                (
                    str(payload["model_call_key"]),
                    encoded,
                    digest,
                    finish_time,
                ),
            )
        connection.execute(
            """
            UPDATE model_results
            SET committed = 1, updated_at = ?
            WHERE job_id = ? AND commit_owner = ?
              AND commit_lease_id = ? AND record_sha256 = ?
            """,
            (
                finish_time,
                job.job_id,
                job.owner,
                job.lease_id,
                digest,
            ),
        )
        connection.commit()
    return True


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
    with _LeaseHeartbeat(
        store.path,
        claimed,
        owner=owner,
        lease_seconds=lease_seconds,
        interval=heartbeat_seconds,
    ) as heartbeat:
        for job in claimed:
            payload = job.payload
            cached = cache.get(payload)
            if cached is None:
                model_jobs.append(job)
                task = _payload_to_task(payload)
                task_by_key[task.cache_key] = task
                job_by_key[task.cache_key] = job
                continue
            if _fenced_commit_model_record(
                store.path,
                job=job,
                expected_kind=job.kind,
                payload=payload,
                record=cached,
                status=(
                    "terminal"
                    if clean_text(cached.get("error"))
                    else "success"
                ),
                heartbeat=heartbeat,
                after_cache_write=after_cache_write,
                after_result_write=after_result_write,
            ):
                handled += 1
        if not model_jobs:
            return handled
        if extractor is None:
            for job in model_jobs:
                if heartbeat.is_lost(str(job.job_id)):
                    continue
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

        def commit_record(
            model_call_key: str,
            record: dict[str, Any],
        ) -> None:
            nonlocal handled
            job = job_by_key[model_call_key]
            payload = job.payload
            status = (
                "terminal"
                if clean_text(record.get("error"))
                else "success"
            )
            if _fenced_commit_model_record(
                store.path,
                job=job,
                expected_kind=job.kind,
                payload=payload,
                record=record,
                status=status,
                heartbeat=heartbeat,
                after_cache_write=after_cache_write,
                after_result_write=after_result_write,
            ):
                handled += 1

        run_extraction_task_group(
            extractor=extractor,
            tasks=[
                task_by_key[job.payload["model_call_key"]]
                for job in model_jobs
            ],
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


def _jobset_provenance(jobset: ModelJobSet) -> dict[str, Any]:
    rows: dict[str, sqlite3.Row] = {}
    with _connect(jobset.database_path) as connection:
        for modality in ("text", "image"):
            row = connection.execute(
                """
                SELECT *
                FROM model_jobsets
                WHERE fingerprint = ?
                """,
                (jobset.fingerprint_for(modality),),
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"missing durable {modality} model jobset metadata"
                )
            rows[modality] = row
    for modality, row in rows.items():
        if (
            str(row["modality"]) != modality
            or str(row["job_kind"]) != jobset.kind_for(modality)
            or int(row["task_count"]) != (
                jobset.text_tasks
                if modality == "text"
                else jobset.image_tasks
            )
            or int(row["enqueue_complete"]) != 1
            or str(row["prompt_version"]) != jobset.prompt_version
        ):
            raise ValueError(
                f"durable {modality} model jobset metadata mismatch"
            )
    return {
        "queue_schema_version": MODEL_QUEUE_SCHEMA_VERSION,
        "parser_schema_version": MODEL_PARSER_SCHEMA_VERSION,
        "prompt_version": jobset.prompt_version,
        "input_fingerprints": {
            modality: str(rows[modality]["input_fingerprint"])
            for modality in ("text", "image")
        },
        "model_identities": {
            modality: str(rows[modality]["model_identity"])
            for modality in ("text", "image")
        },
        "policy_fingerprints": {
            modality: str(rows[modality]["policy_fingerprint"])
            for modality in ("text", "image")
        },
    }


def _validate_output_record_provenance(
    record: dict[str, Any],
    *,
    status: str,
    jobset: ModelJobSet,
    expected: dict[str, Any],
) -> None:
    modality = clean_text(record.get("modality"))
    if modality not in {"text", "image"}:
        raise ValueError("model output record modality is invalid")
    expected_values = {
        "jobset_fingerprint": jobset.fingerprint_for(modality),
        "prompt_version": expected["prompt_version"],
        "model_identity": expected["model_identities"][modality],
        "policy_fingerprint": expected["policy_fingerprints"][modality],
        "parser_schema_version": expected["parser_schema_version"],
        "asset_type": modality,
    }
    if any(
        record.get(field) != value
        for field, value in expected_values.items()
    ):
        raise ValueError(
            f"model output {status} record provenance mismatch"
        )
    if not all(
        clean_text(record.get(field))
        for field in (
            "job_id",
            "cache_key",
            "model_call_key",
            "asset_fingerprint",
            "entity_prompt_fingerprint",
        )
    ):
        raise ValueError(
            f"model output {status} record provenance is incomplete"
        )
    candidates = record.get("candidate_attribute_names")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not all(clean_text(value) for value in candidates)
    ):
        raise ValueError(
            f"model output {status} record candidates are invalid"
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
    if (
        payload.get("stage") != "wdc200k_model_outputs"
        or payload.get("schema_version") != MODEL_OUTPUT_SCHEMA_VERSION
    ):
        raise ValueError("model output manifest stage/schema mismatch")
    if payload.get("identity") != _manifest_identity(jobset):
        raise ValueError("model output manifest identity mismatch")
    if payload.get("complete") is not True:
        return None
    expected_jobsets = {
        "text": jobset.text_fingerprint,
        "image": jobset.image_fingerprint,
    }
    if payload.get("jobsets") != expected_jobsets:
        raise ValueError("model output manifest jobset mismatch")
    expected_provenance = _jobset_provenance(jobset)
    if payload.get("provenance") != expected_provenance:
        raise ValueError("model output manifest provenance mismatch")
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
    counts_value = payload.get("counts")
    if not isinstance(counts_value, dict):
        raise ValueError("model output manifest count metadata is missing")
    try:
        counts = {
            key: int(counts_value[key])
            for key in (
                "text_total",
                "image_total",
                "success",
                "terminal",
            )
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "model output manifest count metadata is invalid"
        ) from error
    if any(value < 0 for value in counts.values()):
        raise ValueError("model output manifest count is negative")
    if (
        counts["text_total"] != jobset.text_tasks
        or counts["image_total"] != jobset.image_tasks
        or counts["success"] + counts["terminal"] != jobset.total_tasks
        or sum(shard.records for shard in extraction_shards)
        != counts["success"]
        or sum(shard.records for shard in error_shards)
        != counts["terminal"]
    ):
        raise ValueError("model output manifest count mismatch")
    snapshot = _job_snapshot(jobset.database_path, jobset)
    if (
        snapshot["total"] != jobset.total_tasks
        or snapshot["success"] != counts["success"]
        or snapshot["terminal"] != counts["terminal"]
        or snapshot["pending"] + snapshot["retryable"] + snapshot["leased"]
        != 0
    ):
        raise ValueError("model output manifest/store count mismatch")
    with _connect(jobset.database_path) as connection:
        connection.execute(
            "CREATE TEMP TABLE observed_outputs (job_id TEXT PRIMARY KEY)"
        )
        for status, shards in (
            ("success", extraction_shards),
            ("terminal", error_shards),
        ):
            observed = 0
            for shard in shards:
                with (output_root / shard.path).open(
                    "r", encoding="utf-8"
                ) as handle:
                    for line in handle:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(
                                "model output shard record is not an object"
                            )
                        _validate_output_record_provenance(
                            record,
                            status=status,
                            jobset=jobset,
                            expected=expected_provenance,
                        )
                        error_text = clean_text(record.get("error"))
                        if (
                            (status == "success" and error_text)
                            or (status == "terminal" and not error_text)
                        ):
                            raise ValueError(
                                f"model output {status} error semantics "
                                "are invalid"
                            )
                        job_id = str(record["job_id"])
                        try:
                            connection.execute(
                                """
                                INSERT INTO observed_outputs (job_id)
                                VALUES (?)
                                """,
                                (job_id,),
                            )
                        except sqlite3.IntegrityError as error:
                            raise ValueError(
                                "duplicate model output job_id"
                            ) from error
                        durable = connection.execute(
                            """
                            SELECT results.status,
                                   results.record_sha256,
                                   results.jobset_fingerprint,
                                   results.modality,
                                   results.committed,
                                   jobs.kind,
                                   members.payload_sha256
                            FROM model_results AS results
                            JOIN jobs USING (job_id)
                            JOIN model_job_members AS members
                              ON members.job_id = results.job_id
                             AND members.jobset_fingerprint =
                                 results.jobset_fingerprint
                            WHERE results.job_id = ?
                            """,
                            (job_id,),
                        ).fetchone()
                        record_digest = hashlib.sha256(
                            _canonical_json(record).encode("utf-8")
                        ).hexdigest()
                        modality = str(record["modality"])
                        if (
                            durable is None
                            or str(durable["status"]) != status
                            or str(durable["record_sha256"])
                            != record_digest
                            or str(durable["jobset_fingerprint"])
                            != jobset.fingerprint_for(modality)
                            or str(durable["modality"]) != modality
                            or int(durable["committed"]) != 1
                            or str(durable["kind"])
                            != jobset.kind_for(modality)
                        ):
                            raise ValueError(
                                "model output does not match its durable "
                                "member/result"
                            )
                        observed += 1
            if observed != counts[status]:
                raise ValueError(
                    "model output shard record count mismatch"
                )
        missing = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM model_results AS results
                JOIN model_job_members AS members
                  ON members.job_id = results.job_id
                 AND members.jobset_fingerprint =
                     results.jobset_fingerprint
                WHERE results.committed = 1
                  AND results.jobset_fingerprint IN (?, ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM observed_outputs
                      WHERE observed_outputs.job_id = results.job_id
                  )
                """,
                (
                    jobset.text_fingerprint,
                    jobset.image_fingerprint,
                ),
            ).fetchone()[0]
        )
        if missing:
            raise ValueError(
                "model output manifest is missing durable results"
            )
    return ModelStageResult(
        output_root=output_root,
        manifest_path=path,
        extraction_paths=tuple(
            output_root / shard.path for shard in extraction_shards
        ),
        error_paths=tuple(
            output_root / shard.path for shard in error_shards
        ),
        text_total=counts["text_total"],
        image_total=counts["image_total"],
        success=counts["success"],
        terminal=counts["terminal"],
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
                          AND committed = 1
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
            "provenance": _jobset_provenance(jobset),
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
    assets_barrier: AssetStageBarrier | None = None,
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
        if assets_barrier is None:
            raise ValueError("assets_barrier is required with start_marker")
        write_model_start_marker(
            start_marker,
            jobset,
            network_manifests=network_manifests,
            assets_manifest=assets_manifest,
            assets_barrier=assets_barrier,
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


def _validate_network_manifest(
    path: Path,
    payload: dict[str, Any],
) -> None:
    if (
        payload.get("stage") != "wdc200k_network_fetch"
        or payload.get("schema_version")
        != "wdc200k-network-fetch-v1"
        or not clean_text(payload.get("policy_fingerprint"))
    ):
        raise ValueError(f"invalid WDC network manifest: {path}")
    declared = payload.get("completed_shards")
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            f"WDC network manifest is missing required shards: {path}"
        )
    _validate_declared_shards(path, payload, field="completed_shards")
    counts_value = payload.get("counts")
    if not isinstance(counts_value, dict):
        raise ValueError(f"WDC network manifest counts are missing: {path}")
    try:
        counts = {
            key: int(counts_value[key])
            for key in (
                "unique",
                "success",
                "terminal",
                "pending",
                "leased",
            )
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"WDC network manifest counts are invalid: {path}"
        ) from error
    if (
        any(value < 0 for value in counts.values())
        or counts["success"] + counts["terminal"] != counts["unique"]
        or counts["pending"] != 0
        or counts["leased"] != 0
        or sum(
            int(item["records"])
            for item in declared
            if isinstance(item, dict)
        )
        != counts["unique"]
    ):
        raise ValueError(
            f"WDC network manifest counts are incomplete: {path}"
        )


def write_model_start_marker(
    path: Path,
    jobset: ModelJobSet,
    *,
    network_manifests: Iterable[Path],
    assets_manifest: Path,
    assets_barrier: AssetStageBarrier,
    run_fingerprint: str,
) -> None:
    """Publish exact staged counts only after upstream completion is proven."""
    if not run_fingerprint:
        raise ValueError("run_fingerprint must not be empty")
    network_paths = [Path(value) for value in network_manifests]
    if not network_paths:
        raise ValueError("at least one WDC network manifest is required")
    upstream = []
    for manifest_path in network_paths:
        payload = _validated_complete_manifest(manifest_path)
        _validate_network_manifest(manifest_path, payload)
        upstream.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256_path(manifest_path),
            }
        )
    assets_path = Path(assets_manifest)
    _strict_asset_manifest(assets_path, barrier=assets_barrier)
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
    expected_status: str | None = None,
    model_kind: str | None = None,
    task_count: int | None = None,
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
    if (
        expected_status is not None
        and payload.get("status") != expected_status
    ):
        return False
    if model_kind is not None and payload.get("model_kind") != model_kind:
        return False
    if task_count is not None and payload.get("task_count") != task_count:
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
        expected_status="vllm_servers_ready",
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
    *,
    assets_barrier: AssetStageBarrier,
) -> Iterator[dict[str, Any]]:
    """Validate Task-5 output and stream only canonical successful assets."""
    manifest_path = Path(manifest_path)
    _payload, asset_paths, _link_paths = _strict_asset_manifest(
        manifest_path,
        barrier=assets_barrier,
    )
    for path in asset_paths:
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


class _AdapterShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        records_per_shard: int,
    ) -> None:
        self.root = root
        self.records_per_shard = records_per_shard
        self.completed: list[CompletedShard] = []
        self.writer: AtomicJsonlShard | None = None
        self.current_records = 0

    def write(self, record: dict[str, Any]) -> None:
        if self.writer is None:
            self.writer = AtomicJsonlShard(
                self.root / f"part-{len(self.completed):05d}.jsonl"
            )
            self.current_records = 0
        self.writer.write(record)
        self.current_records += 1
        if self.current_records >= self.records_per_shard:
            self._commit()

    def _commit(self) -> None:
        if self.writer is None:
            return
        completed = self.writer.commit()
        self.completed.append(
            CompletedShard(
                path=(self.root / completed.path).as_posix(),
                records=completed.records,
                bytes=completed.bytes,
                sha256=completed.sha256,
            )
        )
        self.writer = None

    def close(self) -> list[CompletedShard]:
        self._commit()
        return self.completed

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.abort()


def _iter_jsonl_paths(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"non-object JSONL record: {path}")
                yield record


def _strict_asset_manifest(
    manifest_path: Path,
    *,
    barrier: AssetStageBarrier,
) -> tuple[dict[str, Any], list[Path], list[Path]]:
    payload = _validated_complete_manifest(manifest_path)
    if (
        payload.get("stage") != "wdc200k_asset_materialization"
        or not isinstance(payload.get("fingerprint"), dict)
        or payload["fingerprint"].get("schema_version")
        != "wdc200k-asset-materialization-v1"
    ):
        raise ValueError("invalid Task-5 asset materialization manifest")
    fingerprint = payload["fingerprint"]
    if fingerprint != barrier.fingerprint:
        raise ValueError("Task-5 materialization fingerprint mismatch")
    required_fields = {
        "input_fingerprint",
        "schema_version",
        "planning_manifest_sha256",
        "unique_job_manifest_sha256",
        "unique_job_sha256",
        "image_fetch_manifest_sha256",
        "image_policy_fingerprint",
        "image_outcome_digest",
        "image_outcome_count",
        "image_outcome_url_key_digest",
        "attempts_per_entity",
        "retained_per_entity",
        "text_asset_chunk_chars",
        "min_text_asset_chunk_chars",
        "max_text_asset_chunks_per_entity",
        "records_per_shard",
    }
    if set(fingerprint) != required_fields:
        raise ValueError(
            "Task-5 materialization fingerprint fields are incomplete"
        )
    digest_fields = {
        "planning_manifest_sha256",
        "unique_job_manifest_sha256",
        "unique_job_sha256",
        "image_fetch_manifest_sha256",
        "image_outcome_digest",
        "image_outcome_url_key_digest",
    }
    if any(
        len(str(fingerprint[field])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(fingerprint[field]).lower()
        )
        for field in digest_fields
    ):
        raise ValueError(
            "Task-5 materialization fingerprint digest is invalid"
        )
    if (
        not clean_text(fingerprint["input_fingerprint"])
        or not clean_text(fingerprint["image_policy_fingerprint"])
    ):
        raise ValueError(
            "Task-5 materialization fingerprint identity is empty"
        )
    try:
        integer_values = {
            field: int(fingerprint[field])
            for field in (
                "image_outcome_count",
                "attempts_per_entity",
                "retained_per_entity",
                "text_asset_chunk_chars",
                "min_text_asset_chunk_chars",
                "max_text_asset_chunks_per_entity",
                "records_per_shard",
            )
        }
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Task-5 materialization fingerprint count is invalid"
        ) from error
    if (
        integer_values["image_outcome_count"] < 0
        or integer_values["attempts_per_entity"] < 0
        or integer_values["retained_per_entity"] < 0
        or integer_values["text_asset_chunk_chars"] <= 0
        or integer_values["min_text_asset_chunk_chars"] <= 0
        or integer_values["min_text_asset_chunk_chars"]
        > integer_values["text_asset_chunk_chars"]
        or integer_values["max_text_asset_chunks_per_entity"] < 0
        or integer_values["records_per_shard"] <= 0
    ):
        raise ValueError(
            "Task-5 materialization fingerprint count is inconsistent"
        )
    root = manifest_path.parent
    asset_shards = [
        _completed_from_payload(item)
        for item in payload.get("bridge_asset_shards", [])
    ]
    link_shards = [
        _completed_from_payload(item)
        for item in payload.get("table_asset_link_shards", [])
    ]
    if not asset_shards or not link_shards:
        raise ValueError("Task-5 manifest is missing required shards")
    if not all(
        validate_completed_shard(shard, root)
        for shard in (*asset_shards, *link_shards)
    ):
        raise ValueError("Task-5 manifest shard validation failed")
    asset_count = sum(shard.records for shard in asset_shards)
    link_count = sum(shard.records for shard in link_shards)
    if (
        asset_count != barrier.bridge_assets
        or link_count != barrier.table_asset_links
    ):
        raise ValueError("Task-5 materialization count mismatch")
    return (
        payload,
        [root / shard.path for shard in asset_shards],
        [root / shard.path for shard in link_shards],
    )


def adapt_model_tasks_from_manifests(
    *,
    structural_output_root: Path,
    structural_manifests: Iterable[Path],
    finalized_selection_manifest: Path,
    assets_manifest: Path,
    assets_barrier: AssetStageBarrier,
    output_root: Path,
    args: argparse.Namespace,
    records_per_shard: int = 10_000,
) -> AdaptedModelTasks:
    """Disk-index Task-3/Task-5 artifacts into authoritative model tasks."""
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    import wdc200k_structural as structural

    structural_output_root = Path(structural_output_root)
    structural_paths = sorted(Path(path) for path in structural_manifests)
    if not structural_paths:
        raise ValueError("structural manifests are required")
    source_paths: list[Path] = []
    entity_paths: list[Path] = []
    validated_selection_paths: list[Path] = []
    table_count = 0
    manifest_hashes: list[tuple[Path, str]] = []
    for manifest_path in structural_paths:
        validated, records, manifest_hash = (
            structural._validated_shard_from_manifest(
                manifest_path,
                output_root=structural_output_root,
            )
        )
        validated_selection_paths.append(validated)
        table_count += records
        manifest_hashes.append(
            (manifest_path.resolve(), manifest_hash)
        )
        payload = _validated_complete_manifest(manifest_path)
        completed = [
            _completed_from_payload(item)
            for item in payload["completed_shards"]
        ]
        source_paths.extend(
            structural_output_root / shard.path
            for shard in completed
            if shard.path.startswith("source_tables/")
        )
        entity_paths.extend(
            structural_output_root / shard.path
            for shard in completed
            if shard.path.startswith("entities/")
        )

    final_path = Path(finalized_selection_manifest)
    final_payload = _validated_complete_manifest(final_path)
    if (
        final_payload.get("stage") != "wdc200k_validated_selection"
        or not clean_text(final_payload.get("input_fingerprint"))
        or not clean_text(final_payload.get("parameter_fingerprint"))
        or len(final_payload.get("completed_shards") or []) != 1
    ):
        raise ValueError("invalid Task-3 finalized-selection barrier")
    final_shard = _completed_from_payload(
        final_payload["completed_shards"][0]
    )
    expected_final_input = stable_hash(
        structural.STRUCTURAL_SCHEMA_VERSION,
        *(
            f"{path.as_posix()}:{digest}"
            for path, digest in manifest_hashes
        ),
        length=40,
    )
    expected_final_parameters = stable_hash(
        "validated-selection-global-v1",
        table_count,
        length=40,
    )
    if (
        final_payload.get("input_fingerprint") != expected_final_input
        or final_payload.get("parameter_fingerprint")
        != expected_final_parameters
        or final_shard.records != table_count
        or not validate_completed_shard(
            final_shard,
            structural_output_root,
        )
    ):
        raise ValueError(
            "Task-3 finalized-selection fingerprint/validation failed"
        )
    sentinel = object()
    expected_records = _iter_jsonl_paths(validated_selection_paths)
    actual_records = _iter_jsonl_paths(
        [structural_output_root / final_shard.path]
    )
    for expected_record, actual_record in zip_longest(
        expected_records,
        actual_records,
        fillvalue=sentinel,
    ):
        if (
            expected_record is sentinel
            or actual_record is sentinel
            or _canonical_json(expected_record)
            != _canonical_json(actual_record)
        ):
            raise ValueError(
                "Task-3 finalized-selection content binding failed"
            )
    _, asset_paths, link_paths = _strict_asset_manifest(
        Path(assets_manifest),
        barrier=assets_barrier,
    )
    input_fingerprint = stable_hash(
        MODEL_QUEUE_SCHEMA_VERSION,
        *(digest for _path, digest in manifest_hashes),
        _sha256_path(final_path),
        _sha256_path(Path(assets_manifest)),
        length=40,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "model-task-adapter.sqlite3"
    index_path.unlink(missing_ok=True)
    with sqlite3.connect(index_path) as connection:
        connection.execute(
            "CREATE TABLE assets (asset_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE entities (entity_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        connection.execute(
            """
            CREATE TABLE links (
                source_table_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                PRIMARY KEY (source_table_id, entity_id, asset_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX links_source ON links(source_table_id)"
        )
        for record in _iter_jsonl_paths(asset_paths):
            connection.execute(
                "INSERT OR REPLACE INTO assets VALUES (?, ?)",
                (str(record["asset_id"]), _canonical_json(record)),
            )
        for record in _iter_jsonl_paths(entity_paths):
            connection.execute(
                "INSERT OR REPLACE INTO entities VALUES (?, ?)",
                (str(record["entity_id"]), _canonical_json(record)),
            )
        for record in _iter_jsonl_paths(link_paths):
            for asset_id in record.get("asset_ids") or []:
                connection.execute(
                    "INSERT OR IGNORE INTO links VALUES (?, ?, ?)",
                    (
                        str(record["source_table_id"]),
                        str(record["entity_id"]),
                        str(asset_id),
                    ),
                )
        connection.commit()

    task_writer = _AdapterShardWriter(
        output_root / "tasks",
        records_per_shard=records_per_shard,
    )
    error_writer = _AdapterShardWriter(
        output_root / "planning_errors",
        records_per_shard=records_per_shard,
    )
    task_count = 0
    error_count = 0
    adapter_args = argparse.Namespace(**vars(args))
    if not hasattr(adapter_args, "min_column_non_empty_ratio"):
        adapter_args.min_column_non_empty_ratio = 0.5
    try:
        with sqlite3.connect(index_path) as connection:
            connection.row_factory = sqlite3.Row
            for source_table in _iter_jsonl_paths(source_paths):
                source_table_id = str(source_table["source_table_id"])
                link_rows = connection.execute(
                    """
                    SELECT entity_id, asset_id
                    FROM links
                    WHERE source_table_id = ?
                    ORDER BY entity_id, asset_id
                    """,
                    (source_table_id,),
                ).fetchall()
                entity_to_assets: dict[str, list[str]] = {}
                assets: dict[str, dict[str, Any]] = {}
                wiki_to_entity_id: dict[str, str] = {}
                for link in link_rows:
                    entity_id = str(link["entity_id"])
                    asset_id = str(link["asset_id"])
                    entity_to_assets.setdefault(entity_id, []).append(
                        asset_id
                    )
                    asset_row = connection.execute(
                        "SELECT payload FROM assets WHERE asset_id = ?",
                        (asset_id,),
                    ).fetchone()
                    entity_row = connection.execute(
                        "SELECT payload FROM entities WHERE entity_id = ?",
                        (entity_id,),
                    ).fetchone()
                    if asset_row is None or entity_row is None:
                        continue
                    assets[asset_id] = json.loads(str(asset_row["payload"]))
                    entity = json.loads(str(entity_row["payload"]))
                    wiki_to_entity_id[str(entity["wiki_title"])] = entity_id
                tasks = collect_table_extraction_tasks(
                    source_table=source_table,
                    assets=assets,
                    entity_to_assets=entity_to_assets,
                    wiki_to_entity_id=wiki_to_entity_id,
                    args=adapter_args,
                )
                if link_rows and not tasks:
                    error_writer.write(
                        {
                            "status": "terminal",
                            "error_class": "no_candidate_attributes",
                            "source_table_id": source_table_id,
                        }
                    )
                    error_count += 1
                for task in tasks:
                    if not task.candidate_attribute_names:
                        raise ValueError(
                            "adapter produced an empty candidate task"
                        )
                    task_writer.write(
                        {
                            "extraction_task": {
                                "order": task.order,
                                "cache_key": task.cache_key,
                                "source_table_id": task.source_table_id,
                                "source_row_id": task.source_row_id,
                                "entity_column_index": (
                                    task.entity_column_index
                                ),
                                "entity_column_name": (
                                    task.entity_column_name
                                ),
                                "entity": task.entity,
                                "asset": task.asset,
                                "candidate_attribute_names": (
                                    task.candidate_attribute_names
                                ),
                            }
                        }
                    )
                    task_count += 1
        task_shards = task_writer.close()
        error_shards = error_writer.close()
    except BaseException:
        task_writer.abort()
        error_writer.abort()
        raise
    task_shards = [
        CompletedShard(
            path=Path(shard.path).relative_to(output_root).as_posix(),
            records=shard.records,
            bytes=shard.bytes,
            sha256=shard.sha256,
        )
        for shard in task_shards
    ]
    error_shards = [
        CompletedShard(
            path=Path(shard.path).relative_to(output_root).as_posix(),
            records=shard.records,
            bytes=shard.bytes,
            sha256=shard.sha256,
        )
        for shard in error_shards
    ]
    manifest_path = output_root / "model-task-adapter-manifest.json"
    _atomic_json(
        manifest_path,
        {
            "stage": "wdc200k_model_task_adapter",
            "schema_version": MODEL_QUEUE_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "task_shards": [_shard_payload(item) for item in task_shards],
            "error_shards": [_shard_payload(item) for item in error_shards],
            "counts": {"tasks": task_count, "errors": error_count},
            "complete": True,
        },
    )
    return AdaptedModelTasks(
        output_root=output_root,
        task_paths=tuple(output_root / item.path for item in task_shards),
        error_paths=tuple(output_root / item.path for item in error_shards),
        manifest_path=manifest_path,
        input_fingerprint=input_fingerprint,
        tasks=task_count,
        errors=error_count,
    )


def enqueue_model_tasks_from_manifest(
    manifest_path: Path,
    store: SqliteJobStore,
    *,
    args: argparse.Namespace | None = None,
    assets_barrier: AssetStageBarrier | None = None,
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
    if task_factory is None:
        raise ValueError(
            "task_factory is required; raw Task-5 bridge assets do not "
            "contain candidate attributes"
        )
    if assets_barrier is None:
        raise ValueError("assets_barrier is required for Task-5 input")
    _manifest, asset_paths, _link_paths = _strict_asset_manifest(
        manifest_path,
        barrier=assets_barrier,
    )

    def extraction_inputs() -> Iterator[dict[str, Any]]:
        for current_asset in _iter_jsonl_paths(asset_paths):
            status = clean_text(current_asset.get("status"))
            if status and status != "success":
                continue
            if current_asset.get("asset_type") not in {"text", "image"}:
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

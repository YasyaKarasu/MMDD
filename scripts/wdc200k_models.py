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
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import model_marker_protocol as model_markers

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
        GuardedTextWriter,
        GuardedWriteTracker,
        PreWriteGuard,
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
            GuardedTextWriter,
            GuardedWriteTracker,
            PreWriteGuard,
            SqliteJobStore,
            validate_completed_shard,
        )
    finally:
        sys.path.remove(scripts_directory)


MODEL_QUEUE_SCHEMA_VERSION = "wdc200k-model-queues-v1"
MODEL_OUTPUT_SCHEMA_VERSION = "wdc200k-model-outputs-v1"
MODEL_POLICY_VERSION = "existing-extraction-semantics-v1"
MODEL_PARSER_SCHEMA_VERSION = "connection-evidence-parser-v1"
MODEL_MARKER_SCHEMA_VERSION = model_markers.MODEL_MARKER_SCHEMA_VERSION
MODEL_START_STAGE = model_markers.MODEL_START_STAGE
MODEL_READY_STAGE = model_markers.MODEL_READY_STAGE
MODEL_DONE_STAGE = model_markers.MODEL_DONE_STAGE
STRUCTURAL_STAGE_SCHEMA_VERSION = "wdc200k-structural-v2"
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
class ModelProgressSnapshot:
    """In-memory model queue progress; not part of persisted stage state."""

    modality: str
    total: int
    success: int
    terminal: int
    leased: int
    pending: int

    @property
    def completed(self) -> int:
        return self.success + self.terminal


@dataclass(frozen=True)
class ModelStageAuthority:
    """Expected Task-6 configuration supplied by the stage orchestrator."""

    text_model_identity: str
    image_model_identity: str
    prompt_version: str = PROMPT_VERSION
    policy_fingerprint: str = MODEL_POLICY_VERSION
    parser_schema_version: str = MODEL_PARSER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = (
            self.text_model_identity,
            self.image_model_identity,
            self.prompt_version,
            self.policy_fingerprint,
            self.parser_schema_version,
        )
        if any(not clean_text(value) for value in values):
            raise ValueError("model stage authority values must not be empty")
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError("model stage authority prompt version mismatch")
        if self.parser_schema_version != MODEL_PARSER_SCHEMA_VERSION:
            raise ValueError("model stage authority parser schema mismatch")

    @classmethod
    def current(
        cls,
        args: argparse.Namespace,
    ) -> ModelStageAuthority:
        return cls(
            text_model_identity=_model_identity(args, "text"),
            image_model_identity=_model_identity(args, "image"),
            prompt_version=PROMPT_VERSION,
            policy_fingerprint=MODEL_POLICY_VERSION,
            parser_schema_version=MODEL_PARSER_SCHEMA_VERSION,
        )

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
    ) -> ModelStageAuthority:
        """Compatibility alias; production orchestration should use current."""
        return cls.current(args)


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


@dataclass(frozen=True)
class StructuralStageBarrier:
    """Exact Task-3 manifest set and finalized-selection identity."""

    schema_version: str
    manifest_count: int
    manifest_sha256: dict[str, str]
    input_fingerprints: dict[str, str]
    parameter_fingerprints: dict[str, str]
    final_manifest_sha256: str
    final_selection: dict[str, Any]

    def __post_init__(self) -> None:
        mappings = (
            self.manifest_sha256,
            self.input_fingerprints,
            self.parameter_fingerprints,
        )
        if not all(isinstance(value, dict) for value in mappings):
            raise TypeError(
                "structural barrier manifest identities must be objects"
            )
        normalized = tuple(
            {
                str(key): str(value)
                for key, value in sorted(mapping.items())
            }
            for mapping in mappings
        )
        key_sets = tuple(set(mapping) for mapping in normalized)
        if (
            self.manifest_count <= 0
            or len(normalized[0]) != self.manifest_count
            or key_sets[0] != key_sets[1]
            or key_sets[0] != key_sets[2]
        ):
            raise ValueError("structural barrier manifest set/count mismatch")
        if any(
            not value
            for mapping in normalized[1:]
            for value in mapping.values()
        ):
            raise ValueError("structural barrier fingerprint is empty")
        if any(
            not _is_sha256(value) for value in normalized[0].values()
        ):
            raise ValueError("structural barrier manifest checksum is invalid")
        if not _is_sha256(self.final_manifest_sha256):
            raise ValueError("structural barrier final checksum is invalid")
        if not isinstance(self.final_selection, dict):
            raise TypeError(
                "structural barrier final selection must be an object"
            )
        final_selection = json.loads(_canonical_json(self.final_selection))
        if (
            set(final_selection) != {"path", "records", "bytes", "sha256"}
            or not clean_text(final_selection.get("path"))
            or int(final_selection.get("records", -1)) < 0
            or int(final_selection.get("bytes", -1)) < 0
            or not _is_sha256(str(final_selection.get("sha256", "")))
        ):
            raise ValueError("structural barrier final selection is invalid")
        object.__setattr__(self, "manifest_sha256", normalized[0])
        object.__setattr__(self, "input_fingerprints", normalized[1])
        object.__setattr__(self, "parameter_fingerprints", normalized[2])
        object.__setattr__(self, "final_selection", final_selection)


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _verify_task5_asset_bytes(record: dict[str, Any]) -> dict[str, Any]:
    verified = dict(record)
    if clean_text(verified.get("asset_type")) != "image":
        return verified
    local_path = Path(clean_text(verified.get("local_path")))
    declared = clean_text(verified.get("sha256")).lower()
    if not local_path.is_file():
        raise ValueError("Task-5 image local_path is missing")
    if not _is_sha256(declared):
        raise ValueError("Task-5 image declared hash is invalid")
    if (
        local_path.name != clean_text(verified.get("file_name"))
        or not local_path.name.startswith(f"image_{declared}.")
    ):
        raise ValueError("Task-5 image path is not content-addressed")
    actual = _sha256_path(local_path)
    if actual != declared:
        raise ValueError("Task-5 image content hash mismatch")
    verified["verified_content_sha256"] = actual
    return verified


def _image_evidence_error(payload: dict[str, Any]) -> str:
    asset = payload.get("asset")
    if not isinstance(asset, dict) or asset.get("asset_type") != "image":
        return ""
    verified = clean_text(asset.get("verified_content_sha256")).lower()
    if not verified:
        return ""
    local_path = Path(clean_text(asset.get("local_path")))
    if not local_path.is_file():
        return "image content hash changed since enqueue: file is missing"
    if _sha256_path(local_path) != verified:
        return "image content hash changed since enqueue"
    return ""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as raw_handle:
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
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _initialize_tables(
    path: Path,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    tracker.before_write(64 * 1024)
    connection = _connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN IMMEDIATE")
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
        tracker.before_commit(0)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


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
    staging_dir: Path | None = None,
    pre_write_guard: PreWriteGuard | None = None,
) -> ModelJobSet:
    """Stream complete extraction payloads into isolated modality job sets."""
    if not input_fingerprint:
        raise ValueError("input_fingerprint must not be empty")
    args = args or argparse.Namespace(
        text_model_name="text",
        image_model_name="image",
    )
    store.reserve_write(64 * 1024)
    effective_guard = (
        pre_write_guard
        if pre_write_guard is not None
        else getattr(store._write_tracker, "guard", None)
    )
    _initialize_tables(store.path, pre_write_guard=effective_guard)
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
        store.guard_commit()
    previews: list[ModelJobInfo] = []
    staging_dir = Path(
        staging_dir or store.path.parent / ".model-enqueue-staging"
    )
    if pre_write_guard is not None:
        pre_write_guard(staging_dir, 0)
    staging_dir.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix="incoming-",
        suffix=".sqlite3",
        dir=staging_dir,
    )
    os.close(descriptor)
    staging_path = Path(staging_name)
    staging: sqlite3.Connection | None = None
    try:
        staging_tracker = GuardedWriteTracker(
            staging_path,
            pre_write_guard,
        )
        staging_tracker.before_write(64 * 1024)
        staging = sqlite3.connect(staging_path)
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
            staging_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
            store.reserve_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
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
        staging_tracker.before_commit(0)
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
            store.guard_commit()
    finally:
        if staging is not None:
            staging.close()
        for suffix in ("", "-wal", "-shm"):
            Path(f"{staging_path}{suffix}").unlink(missing_ok=True)
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
        store.reserve_write(4096)
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
        store.guard_commit()
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
    *,
    write_tracker: GuardedWriteTracker | None = None,
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
                if write_tracker is not None:
                    write_tracker.before_write(
                        8192
                        + 4
                        * len(str(row["record_json"]).encode("utf-8"))
                    )
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
        if repaired and write_tracker is not None:
            write_tracker.before_commit(0)
        connection.commit()
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
    write_tracker: GuardedWriteTracker | None = None,
) -> bool:
    if heartbeat.is_lost(str(job.job_id)):
        return False
    canonical = _canonical_extraction_record(payload, record)
    encoded = _canonical_json(canonical)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    now = time.time()
    if write_tracker is not None:
        write_tracker.before_write(
            8192 + 4 * len(encoded.encode("utf-8"))
        )
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
        if write_tracker is not None:
            write_tracker.before_commit(0)
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
        if write_tracker is not None:
            write_tracker.before_commit(0)
        connection.commit()
    return True


def _fenced_retry_model_job(
    database_path: Path,
    *,
    job: Any,
    expected_kind: str,
    record: dict[str, Any],
    write_tracker: GuardedWriteTracker | None = None,
) -> bool:
    encoded = _canonical_json(record)
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        now = time.time()
        row = connection.execute(
            """
            SELECT kind, status, owner, lease_id, lease_expires
            FROM jobs WHERE job_id = ?
            """,
            (job.job_id,),
        ).fetchone()
        if (
            row is None
            or str(row["kind"]) != expected_kind
            or str(row["status"]) != "leased"
            or str(row["owner"]) != str(job.owner)
            or str(row["lease_id"]) != str(job.lease_id)
            or float(row["lease_expires"] or 0.0) <= now
        ):
            connection.rollback()
            return False
        if write_tracker is not None:
            write_tracker.before_write(
                8192 + 2 * len(encoded.encode("utf-8"))
            )
        connection.execute(
            "DELETE FROM model_results WHERE job_id = ? AND committed = 0",
            (job.job_id,),
        )
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = 'retryable', result_json = ?, owner = NULL,
                lease_expires = NULL, lease_id = NULL, updated_at = ?
            WHERE job_id = ? AND kind = ? AND status = 'leased'
              AND owner = ? AND lease_id = ? AND lease_expires > ?
            """,
            (
                encoded,
                now,
                job.job_id,
                expected_kind,
                job.owner,
                job.lease_id,
                now,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        if write_tracker is not None:
            write_tracker.before_commit(0)
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
    progress_tracker: _ModelProgressTracker | None = None,
    write_tracker: GuardedWriteTracker | None = None,
) -> int:
    model_jobs: list[Any] = []
    task_by_key: dict[str, Any] = {}
    job_by_key: dict[str, Any] = {}
    handled = 0
    state_lock = threading.Lock()
    transient_errors: list[str] = []
    with _LeaseHeartbeat(
        store.path,
        claimed,
        owner=owner,
        lease_seconds=lease_seconds,
        interval=heartbeat_seconds,
    ) as heartbeat:
        for job in claimed:
            payload = job.payload
            evidence_error = _image_evidence_error(payload)
            if evidence_error:
                if _fenced_commit_model_record(
                    store.path,
                    job=job,
                    expected_kind=job.kind,
                    payload=payload,
                    record={
                        "attributes": [],
                        "raw_response": "",
                        "error": evidence_error,
                        "error_class": "image_evidence_changed",
                    },
                    status="terminal",
                    heartbeat=heartbeat,
                    after_cache_write=after_cache_write,
                    after_result_write=after_result_write,
                    write_tracker=write_tracker,
                ):
                    handled += 1
                    if progress_tracker is not None:
                        progress_tracker.finished(modality, "terminal")
                continue
            cached = cache.get(payload)
            if cached is None:
                model_jobs.append(job)
                task = _payload_to_task(payload)
                task_by_key[task.cache_key] = task
                job_by_key[task.cache_key] = job
                continue
            cached_status = (
                "terminal"
                if clean_text(cached.get("error"))
                else "success"
            )
            if _fenced_commit_model_record(
                store.path,
                job=job,
                expected_kind=job.kind,
                payload=payload,
                record=cached,
                status=cached_status,
                heartbeat=heartbeat,
                after_cache_write=after_cache_write,
                after_result_write=after_result_write,
                write_tracker=write_tracker,
            ):
                handled += 1
                if progress_tracker is not None:
                    progress_tracker.finished(modality, cached_status)
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
                if progress_tracker is not None:
                    progress_tracker.finished(modality, "retryable")
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
            if record.get("error_class") == "model_endpoint_transient":
                with state_lock:
                    transient_errors.append(clean_text(record.get("error")))
                retried = _fenced_retry_model_job(
                    store.path,
                    job=job,
                    expected_kind=job.kind,
                    record=record,
                    write_tracker=write_tracker,
                )
                if retried:
                    with state_lock:
                        handled += 1
                    if progress_tracker is not None:
                        progress_tracker.finished(modality, "retryable")
                return
            status = (
                "terminal"
                if clean_text(record.get("error"))
                else "success"
            )
            committed = _fenced_commit_model_record(
                store.path,
                job=job,
                expected_kind=job.kind,
                payload=payload,
                record=record,
                status=status,
                heartbeat=heartbeat,
                after_cache_write=after_cache_write,
                after_result_write=after_result_write,
                write_tracker=write_tracker,
            )
            if committed:
                with state_lock:
                    handled += 1
                if progress_tracker is not None:
                    progress_tracker.finished(modality, status)

        run_extraction_task_group(
            extractor=extractor,
            tasks=[
                task_by_key[job.payload["model_call_key"]]
                for job in model_jobs
            ],
            workers=workers,
            on_record=commit_record,
        )
        if transient_errors:
            errors = sorted(set(error for error in transient_errors if error))
            detail = f": {'; '.join(errors)}" if errors else ""
            raise RuntimeError(
                "transient model endpoint failure encountered; owned jobs "
                "were left retryable"
                f"{detail}"
            )
    return handled


def _claimable_modalities(
    database_path: Path,
    jobset: ModelJobSet,
) -> set[str]:
    now = time.time()
    claimable: set[str] = set()
    with _connect(database_path) as connection:
        for modality in ("text", "image"):
            row = connection.execute(
                """
                SELECT 1
                FROM jobs
                WHERE kind = ?
                  AND (
                    status IN ('pending', 'retryable')
                    OR (status = 'leased' AND lease_expires <= ?)
                  )
                LIMIT 1
                """,
                (jobset.kind_for(modality), now),
            ).fetchone()
            if row is not None:
                claimable.add(modality)
    return claimable


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


def _initial_model_progress_counts(
    database_path: Path,
    jobset: ModelJobSet,
) -> dict[str, dict[str, int]]:
    """Read one indexed queue snapshot and treat expired leases as pending."""
    counts = {
        modality: {
            "total": 0,
            "success": 0,
            "terminal": 0,
            "leased": 0,
            "pending": 0,
        }
        for modality in ("text", "image")
    }
    modality_by_kind = {
        jobset.text_kind: "text",
        jobset.image_kind: "image",
    }
    now = time.time()
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT kind, status, COUNT(*) AS count,
                   SUM(
                       CASE WHEN status = 'leased'
                                  AND COALESCE(lease_expires, 0) <= ?
                            THEN 1 ELSE 0 END
                   ) AS expired
            FROM jobs
            WHERE kind IN (?, ?)
            GROUP BY kind, status
            """,
            (now, jobset.text_kind, jobset.image_kind),
        ).fetchall()
    for row in rows:
        current = counts[modality_by_kind[str(row["kind"])]]
        status = str(row["status"])
        count = int(row["count"])
        expired = int(row["expired"] or 0)
        current["total"] += count
        if status in {"success", "terminal"}:
            current[status] += count
        elif status in {"pending", "retryable"}:
            current["pending"] += count
        elif status == "leased":
            current["pending"] += expired
            current["leased"] += count - expired
        else:
            raise ValueError(f"unsupported model job status: {status}")
    expected = {"text": jobset.text_tasks, "image": jobset.image_tasks}
    if any(counts[key]["total"] != expected[key] for key in expected):
        raise ValueError("model progress totals do not match job set")
    return counts


class _ModelProgressTracker:
    """Maintain exact local queue counts after one indexed initial snapshot."""

    def __init__(
        self,
        counts: dict[str, dict[str, int]],
        callback: Callable[[ModelProgressSnapshot], None],
    ) -> None:
        self._counts = counts
        self._callback = callback
        self._lock = threading.Lock()

    def _snapshot_locked(self, modality: str) -> ModelProgressSnapshot:
        if modality not in self._counts:
            raise ValueError(f"unsupported model progress modality: {modality}")
        counts = self._counts[modality]
        return ModelProgressSnapshot(
            modality=modality,
            total=counts["total"],
            success=counts["success"],
            terminal=counts["terminal"],
            leased=counts["leased"],
            pending=counts["pending"],
        )

    def _emit(self, snapshot: ModelProgressSnapshot) -> None:
        try:
            self._callback(snapshot)
        except Exception:
            pass

    def set_modality(self, modality: str) -> None:
        with self._lock:
            snapshot = self._snapshot_locked(modality)
        self._emit(snapshot)

    def claimed(self, modality: str, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            counts = self._counts[modality]
            from_pending = min(count, counts["pending"])
            counts["pending"] -= from_pending
            counts["leased"] += from_pending
            snapshot = self._snapshot_locked(modality)
        self._emit(snapshot)

    def finished(self, modality: str, status: str) -> None:
        with self._lock:
            counts = self._counts[modality]
            if counts["leased"] <= 0:
                raise ValueError("model progress finished without a lease")
            counts["leased"] -= 1
            if status in {"success", "terminal"}:
                counts[status] += 1
            elif status in {"pending", "retryable"}:
                counts["pending"] += 1
            else:
                raise ValueError(f"unsupported model progress status: {status}")
            snapshot = self._snapshot_locked(modality)
        self._emit(snapshot)


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


def _latest_jobset(
    database_path: Path,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> ModelJobSet:
    _initialize_tables(
        database_path,
        pre_write_guard=pre_write_guard,
    )
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
    pre_write_guard: PreWriteGuard | None = None,
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
    validation_dir = (
        jobset.database_path.parent / ".model-validation-staging"
    )
    if pre_write_guard is not None:
        pre_write_guard(validation_dir, 0)
    validation_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        temporary_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="observed-",
                    dir=validation_dir,
                )
            )
        )
        membership_path = temporary_root / "membership.sqlite3"
        validation_tracker = GuardedWriteTracker(
            membership_path,
            pre_write_guard,
        )
        validation_tracker.before_write(64 * 1024)
        connection = stack.enter_context(_connect(jobset.database_path))
        connection.execute(
            "ATTACH DATABASE ? AS validation_db",
            (str(membership_path),),
        )
        connection.execute(
            """
            CREATE TABLE validation_db.observed_outputs (
                job_id TEXT PRIMARY KEY
            )
            """
        )
        try:
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
                            validation_tracker.before_write(
                                4096 + 2 * len(line.encode("utf-8"))
                            )
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
                                    INSERT INTO
                                        validation_db.observed_outputs (
                                            job_id
                                        )
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
                                       results.record_json,
                                       results.record_sha256,
                                       results.jobset_fingerprint,
                                       results.modality,
                                       results.committed,
                                       jobs.kind,
                                       jobs.payload_json,
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
                            modality = str(record["modality"])
                            if durable is None:
                                raise ValueError(
                                    "model output does not match its durable "
                                    "member/result"
                                )
                            payload_encoded = str(durable["payload_json"])
                            payload_digest = hashlib.sha256(
                                payload_encoded.encode("utf-8")
                            ).hexdigest()
                            try:
                                task_payload = json.loads(payload_encoded)
                                durable_record = _decode_checked(
                                    str(durable["record_json"]),
                                    str(durable["record_sha256"]),
                                )
                            except (
                                TypeError,
                                ValueError,
                                json.JSONDecodeError,
                            ) as error:
                                raise ValueError(
                                    "durable model payload/result is invalid"
                                ) from error
                            if not isinstance(task_payload, dict):
                                raise ValueError(
                                    "durable model payload is not an object"
                                )
                            expected_kind = jobset.kind_for(modality)
                            expected_jobset = jobset.fingerprint_for(modality)
                            expected_record = {
                                **durable_record,
                                "job_kind": expected_kind,
                                "payload_sha256": payload_digest,
                            }
                            if (
                                str(durable["status"]) != status
                                or str(durable["jobset_fingerprint"])
                                != expected_jobset
                                or str(durable["modality"]) != modality
                                or int(durable["committed"]) != 1
                                or str(durable["kind"]) != expected_kind
                                or str(durable["payload_sha256"])
                                != payload_digest
                                or task_payload.get("job_id") != job_id
                                or task_payload.get("jobset_fingerprint")
                                != expected_jobset
                                or task_payload.get("modality") != modality
                                or record.get("job_kind") != expected_kind
                                or record.get("payload_sha256")
                                != payload_digest
                                or not _record_matches_payload(
                                    record, task_payload
                                )
                                or _canonical_json(record)
                                != _canonical_json(expected_record)
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
                      SELECT 1
                      FROM validation_db.observed_outputs
                           AS observed_outputs
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
        finally:
            validation_tracker.before_commit(0)
            connection.commit()
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
    pre_write_guard: PreWriteGuard | None = None,
) -> ModelStageResult:
    identity = _manifest_identity(jobset)
    stage_root = output_root / identity
    manifest_path = stage_root / "model-stage-manifest.json"
    resumed = _load_valid_manifest(
        manifest_path,
        output_root=stage_root,
        jobset=jobset,
        pre_write_guard=pre_write_guard,
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
            pre_write_guard=pre_write_guard,
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
                        SELECT results.status, results.record_json,
                               results.record_sha256, jobs.kind,
                               members.payload_sha256
                        FROM model_results AS results
                        JOIN jobs USING (job_id)
                        JOIN model_job_members AS members
                          ON members.job_id = results.job_id
                         AND members.jobset_fingerprint =
                             results.jobset_fingerprint
                        WHERE results.jobset_fingerprint = ?
                          AND results.committed = 1
                        ORDER BY results.job_id
                        """,
                        (jobset.fingerprint_for(modality),),
                    )
                    for row in cursor:
                        status = str(row["status"])
                        record = _decode_checked(
                            str(row["record_json"]),
                            str(row["record_sha256"]),
                        )
                        record = {
                            **record,
                            "job_kind": str(row["kind"]),
                            "payload_sha256": str(
                                row["payload_sha256"]
                            ),
                        }
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
                                / f"part-{index:05d}.jsonl",
                                pre_write_guard=pre_write_guard,
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
        if pre_write_guard is not None:
            pre_write_guard(manifest_path, 0)
        _atomic_json(manifest_path, payload, pre_write_guard)
    published = _load_valid_manifest(
        manifest_path,
        output_root=stage_root,
        jobset=jobset,
        pre_write_guard=pre_write_guard,
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
    group_size: int = 256,
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
    model_progress_callback: (
        Callable[[ModelProgressSnapshot], None] | None
    ) = None,
    start_marker: Path | None = None,
    ready_marker: Path | None = None,
    network_manifests: Iterable[Path] = (),
    assets_manifest: Path | None = None,
    assets_barrier: AssetStageBarrier | None = None,
    ready_timeout_seconds: float | None = None,
    endpoint_ready_timeout_seconds: float = 0.0,
    text_done_marker: Path | None = None,
    image_done_marker: Path | None = None,
    run_fingerprint: str = "",
    pre_write_guard: PreWriteGuard | None = None,
) -> ModelStageResult:
    """Run bounded claims, durably committing each result before job finish."""
    jobset = jobset or _latest_jobset(
        store.path,
        pre_write_guard=pre_write_guard,
    )
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
    if pre_write_guard is not None:
        pre_write_guard(store.path, 0)
        pre_write_guard(output_root, 0)
    owner = owner or f"model-worker-{os.getpid()}-{uuid.uuid4().hex}"
    _initialize_tables(
        store.path,
        pre_write_guard=pre_write_guard,
    )
    if (
        ready_marker is not None
        or text_done_marker is not None
        or image_done_marker is not None
    ) and start_marker is None:
        raise ValueError(
            "ready/done markers require an authoritative start marker"
        )
    start_fingerprint = ""
    if start_marker is not None:
        if assets_manifest is None:
            raise ValueError(
                "assets_manifest is required with start_marker"
            )
        if assets_barrier is None:
            raise ValueError("assets_barrier is required with start_marker")
        start_fingerprint = write_model_start_marker(
            start_marker,
            jobset,
            network_manifests=network_manifests,
            assets_manifest=assets_manifest,
            assets_barrier=assets_barrier,
            run_fingerprint=run_fingerprint,
            pre_write_guard=pre_write_guard,
        )
        if jobset.total_tasks and ready_marker is not None:
            wait_for_model_ready_marker(
                ready_marker,
                run_fingerprint=run_fingerprint,
                text_jobset_fingerprint=jobset.text_fingerprint,
                image_jobset_fingerprint=jobset.image_fingerprint,
                text_task_count=jobset.text_tasks,
                image_task_count=jobset.image_tasks,
                start_fingerprint=start_fingerprint,
                timeout_seconds=ready_timeout_seconds,
            )
    persistent_cache = _PersistentCache(store.path, delegate=cache)
    result_write_tracker = GuardedWriteTracker(
        store.path,
        pre_write_guard,
    )
    _repair_durable_results(
        store.path,
        jobset,
        write_tracker=result_write_tracker,
    )
    progress_tracker = (
        _ModelProgressTracker(
            _initial_model_progress_counts(store.path, jobset),
            model_progress_callback,
        )
        if model_progress_callback is not None
        else None
    )
    ensure_endpoints_ready = getattr(
        extractor,
        "ensure_endpoints_ready",
        None,
    )
    preflighted_modalities: set[str] = set()
    if callable(ensure_endpoints_ready):
        modalities = _claimable_modalities(store.path, jobset)
        if modalities:
            ensure_endpoints_ready(
                modalities=modalities,
                timeout_seconds=endpoint_ready_timeout_seconds,
            )
            preflighted_modalities.update(modalities)
    stop_modalities = threading.Event()
    finished_modalities = {
        modality: threading.Event() for modality in ("text", "image")
    }
    idle_modalities: set[str] = set()
    idle_lock = threading.Lock()

    def process_modality(
        modality: str,
        *,
        limit: int | None = None,
        wait_for_peer: bool = False,
    ) -> int:
        if progress_tracker is not None:
            progress_tracker.set_modality(modality)
        processed = 0
        modality_owner = f"{owner}:{modality}"
        while (
            not stop_modalities.is_set()
            and (limit is None or processed < limit)
        ):
            allowance = (
                group_size
                if limit is None
                else min(group_size, limit - processed)
            )
            if allowance <= 0:
                break
            if (
                callable(ensure_endpoints_ready)
                and modality not in preflighted_modalities
            ):
                modality_is_claimable = modality in _claimable_modalities(
                    store.path,
                    jobset,
                )
                if modality_is_claimable:
                    ensure_endpoints_ready(
                        modalities={modality},
                        timeout_seconds=endpoint_ready_timeout_seconds,
                    )
                    preflighted_modalities.add(modality)
            claimed = store.claim(
                jobset.kind_for(modality),
                limit=allowance,
                owner=modality_owner,
                lease_seconds=lease_seconds,
            )
            if not claimed:
                peer = "image" if modality == "text" else "text"
                with idle_lock:
                    idle_modalities.add(modality)
                    peer_is_idle = peer in idle_modalities
                if (
                    wait_for_peer
                    and not peer_is_idle
                    and not finished_modalities[peer].is_set()
                ):
                    finished_modalities[peer].wait(timeout=0.05)
                    continue
                if (
                    wait_for_peer
                    and modality
                    in _claimable_modalities(store.path, jobset)
                ):
                    # The peer can make a previously leased job claimable in
                    # its final result callback immediately before signalling
                    # completion. Recheck once through the normal readiness
                    # and claim path instead of racing that transition.
                    continue
                break
            if (
                callable(ensure_endpoints_ready)
                and modality not in preflighted_modalities
            ):
                # Claimability can change after the preflight check and
                # before claim() acquires its transaction.
                try:
                    ensure_endpoints_ready(
                        modalities={modality},
                        timeout_seconds=endpoint_ready_timeout_seconds,
                    )
                except BaseException:
                    store.release_owner_leases(
                        jobset.kind_for(modality),
                        owner=modality_owner,
                    )
                    raise
                preflighted_modalities.add(modality)
            with idle_lock:
                idle_modalities.discard(modality)
            if progress_tracker is not None:
                progress_tracker.claimed(modality, len(claimed))
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
                owner=modality_owner,
                lease_seconds=lease_seconds,
                heartbeat_seconds=heartbeat_seconds,
                after_result_write=after_result_write,
                after_cache_write=after_cache_write,
                progress_tracker=progress_tracker,
                write_tracker=result_write_tracker,
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
                text_jobset_fingerprint=jobset.text_fingerprint,
                image_jobset_fingerprint=jobset.image_fingerprint,
                text_task_count=jobset.text_tasks,
                image_task_count=jobset.image_tasks,
                start_fingerprint=start_fingerprint,
                pre_write_guard=pre_write_guard,
            )
        return processed

    def process_modality_and_signal(modality: str) -> int:
        try:
            return process_modality(modality, wait_for_peer=True)
        finally:
            finished_modalities[modality].set()

    processed = 0
    if stop_after is None:
        first_error: BaseException | None = None
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="wdc-model-modality",
        ) as executor:
            futures = {
                executor.submit(
                    process_modality_and_signal,
                    modality,
                ): modality
                for modality in ("text", "image")
            }
            for future in as_completed(futures):
                try:
                    processed += future.result()
                except BaseException as error:
                    stop_modalities.set()
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error
    else:
        for modality in ("text", "image"):
            if processed >= stop_after:
                break
            processed += process_modality(
                modality,
                limit=stop_after - processed,
            )
    snapshot = _job_snapshot(store.path, jobset)
    complete = (
        snapshot["total"] == jobset.total_tasks
        and snapshot["success"] + snapshot["terminal"]
        == jobset.total_tasks
    )
    if complete:
        result_write_tracker.before_commit(0)
        return _publish_outputs(
            store.path,
            output_root=output_root,
            jobset=jobset,
            records_per_shard=records_per_shard,
            pre_write_guard=pre_write_guard,
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
    pre_write_guard: PreWriteGuard | None = None,
) -> str:
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
    context = model_markers.build_marker_context(
        run_fingerprint=run_fingerprint,
        text_jobset_fingerprint=jobset.text_fingerprint,
        image_jobset_fingerprint=jobset.image_fingerprint,
        text_task_count=jobset.text_tasks,
        image_task_count=jobset.image_tasks,
        upstream_identities=upstream,
    )
    model_markers.atomic_write_json(
        Path(path),
        model_markers.start_marker_payload(
            context,
            timestamp=time.time(),
        ),
        pre_write_guard=pre_write_guard,
    )
    return context.start_fingerprint


def write_model_done_marker(
    path: Path,
    *,
    model_kind: str,
    task_count: int,
    jobset_fingerprint: str,
    run_fingerprint: str,
    text_jobset_fingerprint: str,
    image_jobset_fingerprint: str,
    text_task_count: int,
    image_task_count: int,
    start_fingerprint: str,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    if model_kind not in {"text", "image"}:
        raise ValueError(f"unsupported model kind: {model_kind}")
    context = model_markers.ModelMarkerContext(
        run_fingerprint=run_fingerprint,
        text_jobset_fingerprint=text_jobset_fingerprint,
        image_jobset_fingerprint=image_jobset_fingerprint,
        text_task_count=text_task_count,
        image_task_count=image_task_count,
        upstream_identities=(),
        start_fingerprint=start_fingerprint,
    )
    if jobset_fingerprint != context.fingerprint_for(model_kind):
        raise ValueError(
            f"{model_kind} done jobset fingerprint is inconsistent"
        )
    model_markers.atomic_write_json(
        Path(path),
        model_markers.done_marker_payload(
            context,
            model_kind=model_kind,
            task_count=task_count,
            timestamp=time.time(),
        ),
        pre_write_guard=pre_write_guard,
    )


def marker_matches(
    path: Path,
    *,
    expected_stage: str | None = None,
    expected_status: str | None = None,
    model_kind: str | None = None,
    task_count: int | None = None,
    run_fingerprint: str,
    jobset_fingerprint: str | None = None,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
    text_task_count: int | None = None,
    image_task_count: int | None = None,
    start_fingerprint: str | None = None,
) -> bool:
    inferred_stage = expected_stage
    if inferred_stage is None:
        inferred_stage = {
            "model_cache_ready_to_start": MODEL_START_STAGE,
            "vllm_servers_ready": MODEL_READY_STAGE,
            "text_model_cache_precomputed": MODEL_DONE_STAGE,
            "image_model_cache_precomputed": MODEL_DONE_STAGE,
        }.get(expected_status or "")
    if inferred_stage is None or expected_status is None:
        return False
    return model_markers.marker_matches(
        path,
        expected_stage=inferred_stage,
        expected_status=expected_status,
        model_kind=model_kind,
        task_count=task_count,
        run_fingerprint=run_fingerprint,
        jobset_fingerprint=jobset_fingerprint,
        text_jobset_fingerprint=text_jobset_fingerprint,
        image_jobset_fingerprint=image_jobset_fingerprint,
        text_task_count=text_task_count,
        image_task_count=image_task_count,
        start_fingerprint=start_fingerprint,
    )


def wait_for_model_ready_marker(
    path: Path,
    *,
    run_fingerprint: str,
    text_jobset_fingerprint: str | None = None,
    image_jobset_fingerprint: str | None = None,
    text_task_count: int,
    image_task_count: int,
    start_fingerprint: str,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> None:
    started = time.monotonic()
    while not marker_matches(
        Path(path),
        expected_stage=MODEL_READY_STAGE,
        expected_status="vllm_servers_ready",
        run_fingerprint=run_fingerprint,
        text_jobset_fingerprint=text_jobset_fingerprint,
        image_jobset_fingerprint=image_jobset_fingerprint,
        text_task_count=text_task_count,
        image_task_count=image_task_count,
        start_fingerprint=start_fingerprint,
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
                    yield _verify_task5_asset_bytes(record)


class _AdapterShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        records_per_shard: int,
        pre_write_guard: PreWriteGuard | None = None,
    ) -> None:
        self.root = root
        self.records_per_shard = records_per_shard
        self.completed: list[CompletedShard] = []
        self.writer: AtomicJsonlShard | None = None
        self.current_records = 0
        self.pre_write_guard = pre_write_guard

    def write(self, record: dict[str, Any]) -> None:
        if self.writer is None:
            self.writer = AtomicJsonlShard(
                self.root / f"part-{len(self.completed):05d}.jsonl",
                pre_write_guard=self.pre_write_guard,
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
    validate_shards: bool = True,
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
    if not link_shards and barrier.table_asset_links != 0:
        raise ValueError("Task-5 manifest is missing required shards")
    if not asset_shards and barrier.bridge_assets != 0:
        raise ValueError("Task-5 manifest is missing required shards")
    if validate_shards and not all(
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


def _sampling_manifest_entity_paths(manifest_path: Path) -> tuple[Path, ...]:
    from wdc200k_sampling import SAMPLING_SCHEMA_VERSION

    payload = _validated_complete_manifest(manifest_path)
    if (
        payload.get("stage") != "wdc200k_entity_sampling"
        or payload.get("schema_version") != SAMPLING_SCHEMA_VERSION
    ):
        raise ValueError("sampling artifact manifest identity mismatch")
    root = manifest_path.parent
    paths = tuple(
        sorted(
            root / str(item["path"])
            for item in payload.get("completed_shards", [])
            if str(item.get("path", "")).startswith("sampled_entities/")
        )
    )
    if not paths:
        raise ValueError("sampling manifest has no sampled entity paths")
    return paths


def _model_adapter_parameter_fingerprint(
    args: argparse.Namespace,
    *,
    sampled_entity_paths: tuple[Path, ...] | None,
    sampling_manifest: Path | None,
) -> tuple[str, bool]:
    ratio = float(getattr(args, "min_column_non_empty_ratio", 0.5))
    text_model_name = str(getattr(args, "text_model_name", ""))
    image_model_name = str(getattr(args, "image_model_name", ""))
    if sampling_manifest is not None:
        if sampled_entity_paths is None:
            raise ValueError(
                "sampled entity paths are required with sampling manifest"
            )
        declared_paths = _sampling_manifest_entity_paths(sampling_manifest)
        if tuple(
            path.resolve() for path in sampled_entity_paths
        ) != tuple(path.resolve() for path in declared_paths):
            raise ValueError("sampled entity path identity mismatch")
        sampled_identity: Any = {
            "authority": "sampling_manifest",
            "paths": [path.resolve().as_posix() for path in declared_paths],
        }
    elif sampled_entity_paths is None:
        sampled_identity = {"authority": "structural_entities"}
    else:
        sampled_identity = {
            "authority": "explicit_paths",
            "paths": [
                {
                    "path": path.resolve().as_posix(),
                    "sha256": _sha256_path(path),
                }
                for path in sampled_entity_paths
            ],
        }
    payload = {
        "min_column_non_empty_ratio": ratio,
        "text_model_name": text_model_name,
        "image_model_name": image_model_name,
        "sampled_entities": sampled_identity,
    }
    fingerprint = stable_hash(
        "wdc200k-model-task-adapter-parameters-v1",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        length=40,
    )
    legacy_safe = (
        sampling_manifest is not None
        and ratio == 0.5
        and text_model_name == "Qwen3.5-9B"
        and image_model_name == "Qwen3-VL-8B-Thinking"
    )
    return fingerprint, legacy_safe


def adapt_model_tasks_from_manifests(
    *,
    structural_output_root: Path,
    structural_manifests: Iterable[Path],
    finalized_selection_manifest: Path,
    structural_barrier: StructuralStageBarrier,
    assets_manifest: Path,
    assets_barrier: AssetStageBarrier,
    output_root: Path,
    args: argparse.Namespace,
    records_per_shard: int = 10_000,
    pre_write_guard: PreWriteGuard | None = None,
    sampled_entity_paths: Iterable[Path] | None = None,
    sampling_manifest: Path | None = None,
) -> AdaptedModelTasks:
    """Disk-index Task-3/Task-5 artifacts into authoritative model tasks."""
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    import wdc200k_structural as structural
    from wdc200k_sampling import (
        validate_sampling_artifacts,
        validate_sampling_source_authority,
    )

    structural_output_root = Path(structural_output_root)
    structural_paths = sorted(Path(path) for path in structural_manifests)
    sampled_paths = (
        None
        if sampled_entity_paths is None
        else tuple(
            sorted(Path(path).resolve() for path in sampled_entity_paths)
        )
    )
    if not structural_paths:
        raise ValueError("structural manifests are required")
    if (
        structural_barrier.schema_version
        != structural.STRUCTURAL_SCHEMA_VERSION
        or structural_barrier.schema_version
        != STRUCTURAL_STAGE_SCHEMA_VERSION
    ):
        raise ValueError("structural barrier schema mismatch")
    structural_keys = {
        path.resolve().as_posix() for path in structural_paths
    }
    if (
        len(structural_paths) != structural_barrier.manifest_count
        or len(structural_keys) != len(structural_paths)
        or structural_keys != set(structural_barrier.manifest_sha256)
    ):
        raise ValueError("structural barrier manifest set/count mismatch")
    manifest_hashes: list[tuple[Path, str]] = []
    structural_payloads: dict[Path, dict[str, Any]] = {}
    for manifest_path in structural_paths:
        resolved = manifest_path.resolve()
        manifest_key = resolved.as_posix()
        actual_manifest_sha256 = _sha256_path(manifest_path)
        if (
            actual_manifest_sha256
            != structural_barrier.manifest_sha256[manifest_key]
        ):
            raise ValueError("structural barrier manifest checksum mismatch")
        manifest_payload = _validated_complete_manifest(manifest_path)
        if (
            manifest_payload.get("stage") != "wdc200k_structural"
            or manifest_payload.get("schema_version")
            != structural_barrier.schema_version
            or manifest_payload.get("input_fingerprint")
            != structural_barrier.input_fingerprints[manifest_key]
            or manifest_payload.get("parameter_fingerprint")
            != structural_barrier.parameter_fingerprints[manifest_key]
        ):
            raise ValueError("structural barrier fingerprint mismatch")
        manifest_hashes.append((resolved, actual_manifest_sha256))
        structural_payloads[resolved] = manifest_payload

    final_path = Path(finalized_selection_manifest)
    final_payload = _validated_complete_manifest(final_path)
    if (
        _sha256_path(final_path)
        != structural_barrier.final_manifest_sha256
    ):
        raise ValueError("structural final manifest checksum mismatch")
    if (
        final_payload.get("stage") != "wdc200k_validated_selection"
        or final_payload.get("schema_version")
        != structural_barrier.schema_version
        or not clean_text(final_payload.get("input_fingerprint"))
        or not clean_text(final_payload.get("parameter_fingerprint"))
        or len(final_payload.get("completed_shards") or []) != 1
    ):
        raise ValueError("invalid Task-3 finalized-selection barrier")
    final_shard = _completed_from_payload(
        final_payload["completed_shards"][0]
    )
    if _shard_payload(final_shard) != structural_barrier.final_selection:
        raise ValueError("structural final artifact identity mismatch")
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
        final_shard.records,
        length=40,
    )
    if (
        final_payload.get("input_fingerprint") != expected_final_input
        or final_payload.get("parameter_fingerprint")
        != expected_final_parameters
    ):
        raise ValueError(
            "Task-3 finalized-selection fingerprint/validation failed"
        )
    _strict_asset_manifest(
        Path(assets_manifest),
        barrier=assets_barrier,
        validate_shards=False,
    )
    input_fingerprint = model_adapter_input_fingerprint(
        (digest for _path, digest in manifest_hashes),
        finalized_selection_manifest=final_path,
        assets_manifest=Path(assets_manifest),
    )
    if sampling_manifest is not None:
        input_fingerprint = stable_hash(
            input_fingerprint,
            _sha256_path(Path(sampling_manifest)),
            length=40,
        )
    parameter_fingerprint, legacy_parameter_safe = (
        _model_adapter_parameter_fingerprint(
            args,
            sampled_entity_paths=sampled_paths,
            sampling_manifest=(
                Path(sampling_manifest)
                if sampling_manifest is not None
                else None
            ),
        )
    )
    output_root = Path(output_root)
    adapter_manifest_path = (
        output_root / "model-task-adapter-manifest.json"
    )
    resumed = _load_completed_adapted_model_tasks(
        output_root=output_root,
        manifest_path=adapter_manifest_path,
        expected_input_fingerprint=input_fingerprint,
        expected_parameter_fingerprint=parameter_fingerprint,
        allow_legacy_parameter=legacy_parameter_safe,
    )
    if resumed is not None:
        return resumed

    source_paths: list[Path] = []
    entity_paths: list[Path] = []
    validated_selection_paths: list[Path] = []
    table_count = 0
    compact_authority = (
        validate_sampling_source_authority(
            Path(sampling_manifest),
            structural_output_root=structural_output_root,
        )
        if sampling_manifest is not None
        else None
    )
    sampled_authority = (
        validate_sampling_artifacts(Path(sampling_manifest))
        if sampling_manifest is not None
        else None
    )
    for manifest_path in structural_paths:
        actual_manifest_sha256 = structural_barrier.manifest_sha256[
            manifest_path.resolve().as_posix()
        ]
        manifest_payload = structural_payloads[manifest_path.resolve()]
        if compact_authority is None:
            validated, records, _manifest_hash = (
                structural._validated_shard_from_manifest(
                    manifest_path,
                    output_root=structural_output_root,
                )
            )
            validated_selection_paths.append(validated)
            table_count += records
        else:
            manifest_hash = actual_manifest_sha256
            source_shards = [
                item
                for item in manifest_payload["completed_shards"]
                if str(item["path"]).startswith("source_tables/")
            ]
            if len(source_shards) != 1:
                raise ValueError("compact structural source set mismatch")
            table_count += int(source_shards[0]["records"])
        payload = manifest_payload
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

    if sampled_paths is not None:
        entity_paths = list(sampled_paths)
        if not entity_paths or any(not path.is_file() for path in entity_paths):
            raise ValueError("sampled entity paths are missing")
        if sampled_authority is not None and tuple(
            path.resolve() for path in entity_paths
        ) != tuple(
            path.resolve()
            for path in sampled_authority.artifact_paths["sampled_entities"]
        ):
            raise ValueError("sampled entity paths do not match manifest authority")
    if compact_authority is not None:
        source_paths = list(compact_authority.source_tables)

    if (
        final_shard.records != table_count
        or not validate_completed_shard(
            final_shard,
            structural_output_root,
        )
    ):
        raise ValueError(
            "Task-3 finalized-selection fingerprint/validation failed"
        )
    if compact_authority is None:
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
    if pre_write_guard is not None:
        pre_write_guard(output_root, 0)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "model-task-adapter.sqlite3"
    index_path.unlink(missing_ok=True)
    index_tracker = GuardedWriteTracker(index_path, pre_write_guard)
    index_tracker.before_write(64 * 1024)
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
            record = _verify_task5_asset_bytes(record)
            encoded = _canonical_json(record)
            index_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
            connection.execute(
                "INSERT OR REPLACE INTO assets VALUES (?, ?)",
                (str(record["asset_id"]), encoded),
            )
        for record in _iter_jsonl_paths(entity_paths):
            encoded = _canonical_json(record)
            index_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
            connection.execute(
                "INSERT OR REPLACE INTO entities VALUES (?, ?)",
                (str(record["entity_id"]), encoded),
            )
        for record in _iter_jsonl_paths(link_paths):
            for asset_id in record.get("asset_ids") or []:
                index_tracker.before_write(4096)
                connection.execute(
                    "INSERT OR IGNORE INTO links VALUES (?, ?, ?)",
                    (
                        str(record["source_table_id"]),
                        str(record["entity_id"]),
                        str(asset_id),
                    ),
                )
        index_tracker.before_commit(0)
        connection.commit()

    task_writer = _AdapterShardWriter(
        output_root / "tasks",
        records_per_shard=records_per_shard,
        pre_write_guard=pre_write_guard,
    )
    error_writer = _AdapterShardWriter(
        output_root / "planning_errors",
        records_per_shard=records_per_shard,
        pre_write_guard=pre_write_guard,
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
    if pre_write_guard is not None:
        pre_write_guard(adapter_manifest_path, 0)
    _atomic_json(
        adapter_manifest_path,
        {
            "stage": "wdc200k_model_task_adapter",
            "schema_version": MODEL_QUEUE_SCHEMA_VERSION,
            "input_fingerprint": input_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
            "task_shards": [_shard_payload(item) for item in task_shards],
            "error_shards": [_shard_payload(item) for item in error_shards],
            "counts": {"tasks": task_count, "errors": error_count},
            "complete": True,
        },
        pre_write_guard,
    )
    return AdaptedModelTasks(
        output_root=output_root,
        task_paths=tuple(output_root / item.path for item in task_shards),
        error_paths=tuple(output_root / item.path for item in error_shards),
        manifest_path=adapter_manifest_path,
        input_fingerprint=input_fingerprint,
        tasks=task_count,
        errors=error_count,
    )


def model_adapter_input_fingerprint(
    structural_manifest_sha256: Iterable[str],
    *,
    finalized_selection_manifest: Path,
    assets_manifest: Path,
) -> str:
    """Return the canonical Task-6 adapter identity formula."""
    digests = tuple(str(value) for value in structural_manifest_sha256)
    if not digests or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests
    ):
        raise ValueError("structural manifest digest set is invalid")
    return stable_hash(
        MODEL_QUEUE_SCHEMA_VERSION,
        *digests,
        _sha256_path(Path(finalized_selection_manifest)),
        _sha256_path(Path(assets_manifest)),
        length=40,
    )


def validate_adapted_model_tasks(
    adapted: AdaptedModelTasks,
    *,
    expected_input_fingerprint: str,
) -> AdaptedModelTasks:
    """Validate the adapter manifest and every declared task/error shard."""
    reconstructed = _validated_adapted_model_tasks_from_manifest(
        output_root=Path(adapted.output_root),
        manifest_path=Path(adapted.manifest_path),
        expected_input_fingerprint=expected_input_fingerprint,
    )
    if reconstructed != adapted:
        raise ValueError("model adapter result does not match manifest")
    return reconstructed


def _validated_adapted_model_tasks_from_manifest(
    *,
    output_root: Path,
    manifest_path: Path,
    expected_input_fingerprint: str,
    expected_parameter_fingerprint: str | None = None,
    allow_legacy_parameter: bool = False,
) -> AdaptedModelTasks:
    """Return the exact adapter result declared by one complete manifest."""
    payload = _validated_complete_manifest(manifest_path)
    _validate_adapter_manifest_identity(
        payload,
        expected_input_fingerprint=expected_input_fingerprint,
    )
    declared_parameters = payload.get("parameter_fingerprint")
    if declared_parameters is not None and (
        not isinstance(declared_parameters, str)
        or len(declared_parameters) != 40
        or any(
            character not in "0123456789abcdef"
            for character in declared_parameters
        )
    ):
        raise ValueError("model adapter parameter identity is invalid")
    if expected_parameter_fingerprint is not None and (
        (
            declared_parameters is None
            and not allow_legacy_parameter
        )
        or (
            declared_parameters is not None
            and declared_parameters != expected_parameter_fingerprint
        )
    ):
        raise ValueError("model adapter parameter identity mismatch")
    _validate_adapter_shard_paths(payload)
    try:
        task_shards = _parse_adapter_shards(
            payload,
            field="task_shards",
        )
        error_shards = _parse_adapter_shards(
            payload,
            field="error_shards",
        )
        counts = payload["counts"]
        if not isinstance(counts, dict):
            raise ValueError("counts must be an object")
        expected_tasks = _strict_adapter_nonnegative_int(
            counts["tasks"],
            field="counts.tasks",
        )
        expected_errors = _strict_adapter_nonnegative_int(
            counts["errors"],
            field="counts.errors",
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model adapter metadata is invalid") from error
    root = Path(output_root)
    if (
        expected_tasks < 0
        or expected_errors < 0
        or sum(item.records for item in task_shards) != expected_tasks
        or sum(item.records for item in error_shards) != expected_errors
        or not all(
            validate_completed_shard(item, root)
            for item in (*task_shards, *error_shards)
        )
    ):
        raise ValueError("model adapter shard validation failed")
    reconstructed = AdaptedModelTasks(
        output_root=root,
        task_paths=tuple(root / item.path for item in task_shards),
        error_paths=tuple(root / item.path for item in error_shards),
        manifest_path=Path(manifest_path),
        input_fingerprint=expected_input_fingerprint,
        tasks=expected_tasks,
        errors=expected_errors,
    )
    return reconstructed


def _validate_adapter_manifest_identity(
    payload: dict[str, Any],
    *,
    expected_input_fingerprint: str,
) -> None:
    if (
        payload.get("stage") != "wdc200k_model_task_adapter"
        or payload.get("schema_version") != MODEL_QUEUE_SCHEMA_VERSION
        or payload.get("input_fingerprint")
        != expected_input_fingerprint
    ):
        raise ValueError("model adapter input identity mismatch")


def _validate_adapter_shard_paths(payload: dict[str, Any]) -> None:
    all_paths: list[str] = []
    for field, directory in (
        ("task_shards", "tasks"),
        ("error_shards", "planning_errors"),
    ):
        declared = payload.get(field)
        if not isinstance(declared, list):
            raise ValueError("model adapter shard path declaration is invalid")
        for item in declared:
            if not isinstance(item, dict) or not isinstance(
                item.get("path"), str
            ):
                raise ValueError("model adapter shard path declaration is invalid")
            value = item["path"]
            normalized = Path(value)
            if (
                not value
                or "\\" in value
                or normalized.is_absolute()
                or normalized.as_posix() != value
                or len(normalized.parts) < 2
                or normalized.parts[0] != directory
                or any(part in {"", ".", ".."} for part in normalized.parts)
            ):
                raise ValueError(
                    "model adapter shard path declaration is invalid"
                )
            all_paths.append(value)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("model adapter shard paths must be unique")


def _strict_adapter_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative JSON integer")
    return value


def _parse_adapter_shards(
    payload: dict[str, Any],
    *,
    field: str,
) -> tuple[CompletedShard, ...]:
    declared = payload.get(field)
    if not isinstance(declared, list):
        raise ValueError(f"{field} must be a list")
    parsed: list[CompletedShard] = []
    for item in declared:
        if not isinstance(item, dict):
            raise ValueError(f"{field} item must be an object")
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str):
            raise ValueError(f"{field}.path must be a string")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{field}.sha256 must be lowercase hexadecimal")
        parsed.append(
            CompletedShard(
                path=path,
                records=_strict_adapter_nonnegative_int(
                    item.get("records"),
                    field=f"{field}.records",
                ),
                bytes=_strict_adapter_nonnegative_int(
                    item.get("bytes"),
                    field=f"{field}.bytes",
                ),
                sha256=digest,
            )
        )
    return tuple(parsed)


def _load_completed_adapted_model_tasks(
    *,
    output_root: Path,
    manifest_path: Path,
    expected_input_fingerprint: str,
    expected_parameter_fingerprint: str,
    allow_legacy_parameter: bool,
) -> AdaptedModelTasks | None:
    """Reuse a valid complete adapter; incomplete manifests remain rebuildable."""
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("model adapter manifest is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("model adapter manifest is invalid")
    _validate_adapter_manifest_identity(
        payload,
        expected_input_fingerprint=expected_input_fingerprint,
    )
    if not isinstance(payload.get("complete"), bool):
        raise ValueError("model adapter completion flag is invalid")
    if payload["complete"] is False:
        return None
    return _validated_adapted_model_tasks_from_manifest(
        output_root=output_root,
        manifest_path=manifest_path,
        expected_input_fingerprint=expected_input_fingerprint,
        expected_parameter_fingerprint=expected_parameter_fingerprint,
        allow_legacy_parameter=allow_legacy_parameter,
    )


def validate_model_stage_for_adapter(
    result: ModelStageResult,
    adapted: AdaptedModelTasks,
    *,
    args: argparse.Namespace,
    authority: ModelStageAuthority,
    validation_store_path: Path,
) -> bool:
    """Rebuild adapter membership and compare it to the durable model run."""
    adapted = validate_adapted_model_tasks(
        adapted,
        expected_input_fingerprint=adapted.input_fingerprint,
    )
    if authority.parser_schema_version != MODEL_PARSER_SCHEMA_VERSION:
        raise ValueError("model stage authority parser schema mismatch")
    if authority.policy_fingerprint != MODEL_POLICY_VERSION:
        raise ValueError("model stage authority policy mismatch")
    authority_identities = {
        "text": authority.text_model_identity,
        "image": authority.image_model_identity,
    }
    if any(
        _model_identity(args, modality)
        != authority_identities[modality]
        for modality in ("text", "image")
    ):
        raise ValueError("model stage authority CLI identity mismatch")
    if (
        result.jobset.input_fingerprint != adapted.input_fingerprint
        or result.jobset.prompt_version != authority.prompt_version
    ):
        raise ValueError("model stage authority jobset mismatch")
    for modality in ("text", "image"):
        expected_fingerprint = _modality_fingerprint(
            modality=modality,
            input_fingerprint=adapted.input_fingerprint,
            prompt_version=authority.prompt_version,
            model_identity=authority_identities[modality],
            policy_fingerprint=authority.policy_fingerprint,
        )
        if (
            result.jobset.fingerprint_for(modality)
            != expected_fingerprint
            or result.jobset.kind_for(modality)
            != f"model-{modality}-{expected_fingerprint}"
        ):
            raise ValueError("model stage authority fingerprint mismatch")
    if not validate_model_stage(result):
        raise ValueError("model stage result validation failed")
    expected = enqueue_model_tasks(
        _iter_jsonl_paths(adapted.task_paths),
        SqliteJobStore(Path(validation_store_path)),
        args=args,
        input_fingerprint=adapted.input_fingerprint,
        text_input_fingerprint=adapted.input_fingerprint,
        image_input_fingerprint=adapted.input_fingerprint,
        prompt_version=authority.prompt_version,
        policy_fingerprint=authority.policy_fingerprint,
    )
    comparable_fields = (
        "input_fingerprint",
        "prompt_version",
        "text_fingerprint",
        "image_fingerprint",
        "text_kind",
        "image_kind",
        "text_tasks",
        "image_tasks",
    )
    if any(
        getattr(expected, field) != getattr(result.jobset, field)
        for field in comparable_fields
    ):
        raise ValueError("model stage does not belong to adapter task set")
    for modality in ("text", "image"):
        expected_fingerprint = expected.fingerprint_for(modality)
        actual_fingerprint = result.jobset.fingerprint_for(modality)
        with _connect(expected.database_path) as connection:
            expected_row = connection.execute(
                """
                SELECT input_fingerprint, prompt_version, model_identity,
                       policy_fingerprint, task_count, membership_digest,
                       enqueue_complete
                FROM model_jobsets WHERE fingerprint = ?
                """,
                (expected_fingerprint,),
            ).fetchone()
        with _connect(result.jobset.database_path) as connection:
            actual_row = connection.execute(
                """
                SELECT input_fingerprint, prompt_version, model_identity,
                       policy_fingerprint, task_count, membership_digest,
                       enqueue_complete
                FROM model_jobsets WHERE fingerprint = ?
                """,
                (actual_fingerprint,),
            ).fetchone()
        if (
            expected_row is None
            or actual_row is None
            or str(expected_row["input_fingerprint"])
            != adapted.input_fingerprint
            or str(actual_row["input_fingerprint"])
            != adapted.input_fingerprint
            or tuple(expected_row) != tuple(actual_row)
            or int(actual_row["enqueue_complete"]) != 1
        ):
            raise ValueError("model adapter membership digest mismatch")
    return True


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
    staging_dir: Path | None = None,
    pre_write_guard: PreWriteGuard | None = None,
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
            current_asset = _verify_task5_asset_bytes(current_asset)
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
        staging_dir=staging_dir,
        pre_write_guard=pre_write_guard,
    )

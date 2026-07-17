"""Bounded planning, fetching, and materialization of WDC bridge assets."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import urlsplit

try:
    from build_mm_table_dataset import (
        normalize_title,
        select_relevant_text_chunks,
        split_text_asset_content,
    )
    from stage1_io import clean_text, stable_hash
    from wdc200k_fetch import FetchPolicy
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        StageFingerprint,
        StageManifest,
        SqliteJobStore,
        external_unique_jsonl,
        validate_completed_shard,
    )
    from wdc200k_structural import _normalize_http_url
except ModuleNotFoundError as error:
    if error.name not in {
        "build_mm_table_dataset",
        "stage1_io",
        "wdc200k_fetch",
        "wdc200k_io",
        "wdc200k_structural",
    }:
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        build_helpers = importlib.import_module("build_mm_table_dataset")
        normalize_title = build_helpers.normalize_title
        select_relevant_text_chunks = (
            build_helpers.select_relevant_text_chunks
        )
        split_text_asset_content = build_helpers.split_text_asset_content
        stage1_io = importlib.import_module("stage1_io")
        clean_text = stage1_io.clean_text
        stable_hash = stage1_io.stable_hash
        FetchPolicy = importlib.import_module("wdc200k_fetch").FetchPolicy
        io_helpers = importlib.import_module("wdc200k_io")
        AtomicJsonlShard = io_helpers.AtomicJsonlShard
        CompletedShard = io_helpers.CompletedShard
        StageFingerprint = io_helpers.StageFingerprint
        StageManifest = io_helpers.StageManifest
        SqliteJobStore = io_helpers.SqliteJobStore
        external_unique_jsonl = io_helpers.external_unique_jsonl
        validate_completed_shard = io_helpers.validate_completed_shard
        _normalize_http_url = importlib.import_module(
            "wdc200k_structural"
        )._normalize_http_url
    finally:
        sys.path.remove(scripts_directory)


ASSET_PLANNING_SCHEMA_VERSION = "wdc200k-asset-planning-v1"
UNIQUE_IMAGE_JOB_SCHEMA_VERSION = "wdc200k-unique-image-jobs-v1"
ASSET_MATERIALIZATION_SCHEMA_VERSION = "wdc200k-asset-materialization-v1"


def structural_asset_input_identity(
    structural_manifest_sha256: Iterable[str],
    finalized_selection_manifest_sha256: str,
) -> str:
    """Collapse exact validated Task-3 artifacts into a Task-5 input ID."""
    digests = tuple(str(value) for value in structural_manifest_sha256)
    values = (*digests, str(finalized_selection_manifest_sha256))
    if not digests or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in values
    ):
        raise ValueError("structural asset input digests are invalid")
    return stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        *digests,
        finalized_selection_manifest_sha256,
        length=64,
    )


def asset_planning_input_fingerprint(
    structural_identity: str,
    page_fetch_identity: str,
) -> str:
    """Bind asset planning to exact validated Task-3 and Task-4 inputs."""
    if not structural_identity or not page_fetch_identity:
        raise ValueError("asset planning identities must not be empty")
    return stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        structural_identity,
        page_fetch_identity,
        length=40,
    )


def asset_materialization_input_fingerprint(
    planning_manifest_path: Path,
    image_fetch_manifest_path: Path,
) -> str:
    """Bind final Task-5 shards to exact planning and image-fetch manifests."""
    return stable_hash(
        ASSET_MATERIALIZATION_SCHEMA_VERSION,
        _sha256_file(Path(planning_manifest_path)),
        _sha256_file(Path(image_fetch_manifest_path)),
        length=40,
    )


@dataclass(frozen=True)
class ImageBudget:
    """Independent attempted-candidate and retained-success limits."""

    attempts_per_entity: int = 3
    retained_per_entity: int = 3

    def __post_init__(self) -> None:
        if self.attempts_per_entity < 0:
            raise ValueError("attempts_per_entity must be non-negative")
        if self.retained_per_entity < 0:
            raise ValueError("retained_per_entity must be non-negative")


@dataclass(frozen=True)
class ImageReference:
    entity_id: str
    source_table_id: str
    row_id: Any
    image_url: str
    url_key: str
    ordinal: int
    source: str
    page_url: str

    def as_record(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "source_table_id": self.source_table_id,
            "row_id": self.row_id,
            "image_url": self.image_url,
            "url_key": self.url_key,
            "ordinal": self.ordinal,
            "source": self.source,
            "page_url": self.page_url,
        }


@dataclass(frozen=True)
class EntityAssetPlan:
    entity_id: str
    image_refs: tuple[ImageReference, ...]
    page_was_required: bool = True


@dataclass(frozen=True)
class MaterializedEntityAssets:
    entity_id: str
    bridge_assets: list[dict[str, Any]]
    table_asset_links: list[dict[str, Any]]


@dataclass(frozen=True)
class AssetPlanShards:
    output_root: Path
    entity_plan_paths: tuple[Path, ...]
    image_mapping_paths: tuple[Path, ...]
    manifest_path: Path
    entities: int
    image_mappings: int


@dataclass(frozen=True)
class UniqueImageJobs:
    output_path: Path
    manifest_path: Path
    completed_shard: CompletedShard
    input_fingerprint: str
    parameter_fingerprint: str
    complete: bool = True

    @property
    def records(self) -> int:
        return self.completed_shard.records


@dataclass(frozen=True)
class ImageFetchResult:
    unique: int
    success: int
    terminal: int
    complete: bool
    outcomes_path: Path
    policy_fingerprint: str
    maximum_inflight: int
    maximum_claimed: int
    maximum_host_states: int
    unique_jobs: UniqueImageJobs
    job_store_path: Path
    job_kind: str
    fetch_manifest_path: Path
    fetch_manifest_sha256: str
    outcome_digest: str
    outcomes_count: int
    outcome_url_key_digest: str
    leased: int
    remaining: int


@dataclass(frozen=True)
class MaterializedAssetShards:
    output_root: Path
    bridge_asset_paths: tuple[Path, ...]
    table_asset_link_paths: tuple[Path, ...]
    manifest_path: Path
    bridge_assets: int
    table_asset_links: int


@dataclass(frozen=True)
class ImageUrlLease:
    owner: str
    lease_id: str
    lease_until: float


@dataclass(frozen=True)
class ImageUrlClaimDecision:
    outcome: dict[str, Any] | None = None
    lease: ImageUrlLease | None = None
    retry_at: float | None = None


@dataclass(frozen=True)
class ImageJobExecution:
    outcome: dict[str, Any]
    lease: ImageUrlLease | None


class ImageOutcomeStore:
    """Durable policy-scoped terminal outcomes for unique image URLs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            journal_deadline = time.monotonic() + 30.0
            while True:
                try:
                    connection.execute(
                        "PRAGMA journal_mode=WAL"
                    ).fetchone()
                    break
                except sqlite3.OperationalError as error:
                    if (
                        "locked" not in str(error).casefold()
                        or time.monotonic() >= journal_deadline
                    ):
                        raise
                    time.sleep(0.01)
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_outcomes (
                    policy_fingerprint TEXT NOT NULL,
                    url_key TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (policy_fingerprint, url_key)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_url_claims (
                    policy_fingerprint TEXT NOT NULL,
                    url_key TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    lease_until REAL NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (policy_fingerprint, url_key)
                )
                """
            )
            claim_columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(image_url_claims)"
                )
            }
            required_claim_columns = {
                "policy_fingerprint",
                "url_key",
                "owner",
                "lease_id",
                "lease_until",
                "status",
                "updated_at",
            }
            if not required_claim_columns <= claim_columns:
                raise ValueError("image URL claim schema is incompatible")
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
            row = connection.execute(
                """
                SELECT outcome_json, payload_sha256
                FROM image_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
        if row is None:
            return None
        return self._decode(row)

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        encoded = str(row["outcome_json"])
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if digest != str(row["payload_sha256"]):
            raise ValueError("image outcome payload checksum mismatch")
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise ValueError("image outcome payload is not an object")
        return payload

    @staticmethod
    def _encode_outcome(
        policy_fingerprint: str,
        image_url: str,
        outcome: dict[str, Any],
    ) -> tuple[dict[str, Any], str, str]:
        status = str(outcome.get("status") or "")
        if status not in {"success", "terminal"}:
            raise ValueError(f"non-terminal image outcome: {status}")
        canonical = {
            **outcome,
            "status": status,
            "image_url": image_url,
            "original_url": (
                clean_text(outcome.get("original_url")) or image_url
            ),
            "policy_fingerprint": policy_fingerprint,
        }
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return canonical, encoded, digest

    def put(
        self,
        policy_fingerprint: str,
        url_key: str,
        image_url: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        canonical, encoded, digest = self._encode_outcome(
            policy_fingerprint,
            image_url,
            outcome,
        )
        status = str(canonical["status"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT outcome_json
                FROM image_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO image_outcomes (
                        policy_fingerprint, url_key, image_url, status,
                        outcome_json, payload_sha256, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        policy_fingerprint,
                        url_key,
                        image_url,
                        status,
                        encoded,
                        digest,
                        time.time(),
                    ),
                )
                return canonical
            persisted = json.loads(str(existing["outcome_json"]))
            if persisted != canonical:
                raise ValueError(
                    "conflicting terminal image outcome for URL"
                )
            return persisted

    def claim_url(
        self,
        policy_fingerprint: str,
        url_key: str,
        *,
        owner: str,
        lease_seconds: float,
        now: float | None = None,
    ) -> ImageUrlClaimDecision:
        """Atomically return an outcome, acquire an expired URL, or wait."""
        if not owner:
            raise ValueError("URL claim owner must not be empty")
        if lease_seconds <= 0:
            raise ValueError("URL claim lease_seconds must be positive")
        current_time = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outcome_row = connection.execute(
                """
                SELECT outcome_json, payload_sha256
                FROM image_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if outcome_row is not None:
                outcome = self._decode(outcome_row)
                connection.execute(
                    """
                    DELETE FROM image_url_claims
                    WHERE policy_fingerprint = ? AND url_key = ?
                    """,
                    (policy_fingerprint, url_key),
                )
                connection.commit()
                return ImageUrlClaimDecision(outcome=outcome)

            current = connection.execute(
                """
                SELECT owner, lease_id, lease_until, status
                FROM image_url_claims
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if (
                current is not None
                and str(current["status"]) == "leased"
                and float(current["lease_until"]) > current_time
            ):
                retry_at = float(current["lease_until"])
                connection.commit()
                return ImageUrlClaimDecision(retry_at=retry_at)

            lease = ImageUrlLease(
                owner=owner,
                lease_id=uuid.uuid4().hex,
                lease_until=current_time + lease_seconds,
            )
            connection.execute(
                """
                INSERT INTO image_url_claims (
                    policy_fingerprint, url_key, owner, lease_id,
                    lease_until, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'leased', ?)
                ON CONFLICT(policy_fingerprint, url_key) DO UPDATE SET
                    owner = excluded.owner,
                    lease_id = excluded.lease_id,
                    lease_until = excluded.lease_until,
                    status = 'leased',
                    updated_at = excluded.updated_at
                """,
                (
                    policy_fingerprint,
                    url_key,
                    lease.owner,
                    lease.lease_id,
                    lease.lease_until,
                    current_time,
                ),
            )
            connection.commit()
            return ImageUrlClaimDecision(lease=lease)
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def put_claimed(
        self,
        policy_fingerprint: str,
        url_key: str,
        image_url: str,
        outcome: dict[str, Any],
        *,
        lease: ImageUrlLease,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Persist an outcome only while the exact fenced URL lease is live."""
        current_time = time.time() if now is None else float(now)
        canonical, encoded, digest = self._encode_outcome(
            policy_fingerprint,
            image_url,
            outcome,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            claim = connection.execute(
                """
                SELECT owner, lease_id, lease_until, status
                FROM image_url_claims
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if (
                claim is None
                or str(claim["owner"]) != lease.owner
                or str(claim["lease_id"]) != lease.lease_id
                or str(claim["status"]) != "leased"
                or float(claim["lease_until"]) <= current_time
            ):
                raise ValueError("stale or expired image URL claim")
            existing = connection.execute(
                """
                SELECT outcome_json, payload_sha256
                FROM image_outcomes
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO image_outcomes (
                        policy_fingerprint, url_key, image_url, status,
                        outcome_json, payload_sha256, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        policy_fingerprint,
                        url_key,
                        image_url,
                        str(canonical["status"]),
                        encoded,
                        digest,
                        current_time,
                    ),
                )
                persisted = canonical
            else:
                persisted = self._decode(existing)
                if persisted != canonical:
                    raise ValueError(
                        "conflicting terminal image outcome for URL"
                    )
            connection.commit()
            return persisted
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def finish_claim(
        self,
        policy_fingerprint: str,
        url_key: str,
        *,
        lease: ImageUrlLease,
        now: float | None = None,
    ) -> bool:
        """Release only the exact URL lease; stale owners are fenced out."""
        current_time = time.time() if now is None else float(now)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                DELETE FROM image_url_claims
                WHERE policy_fingerprint = ? AND url_key = ?
                  AND owner = ? AND lease_id = ? AND status = 'leased'
                  AND lease_until > ?
                """,
                (
                    policy_fingerprint,
                    url_key,
                    lease.owner,
                    lease.lease_id,
                    current_time,
                ),
            )
            connection.commit()
            return cursor.rowcount == 1
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def clear_claim_if_outcome(
        self,
        policy_fingerprint: str,
        url_key: str,
    ) -> bool:
        """Fence any leftover claimant after a durable outcome is visible."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outcome_exists = (
                connection.execute(
                    """
                    SELECT 1
                    FROM image_outcomes
                    WHERE policy_fingerprint = ? AND url_key = ?
                    """,
                    (policy_fingerprint, url_key),
                ).fetchone()
                is not None
            )
            if not outcome_exists:
                connection.commit()
                return False
            cursor = connection.execute(
                """
                DELETE FROM image_url_claims
                WHERE policy_fingerprint = ? AND url_key = ?
                """,
                (policy_fingerprint, url_key),
            )
            connection.commit()
            return cursor.rowcount == 1
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def iter(
        self,
        policy_fingerprint: str,
    ) -> Iterator[dict[str, Any]]:
        connection = self._connect()
        try:
            for row in connection.execute(
                """
                SELECT outcome_json, payload_sha256
                FROM image_outcomes
                WHERE policy_fingerprint = ?
                ORDER BY url_key
                """,
                (policy_fingerprint,),
            ):
                yield self._decode(row)
        finally:
            connection.close()

    def counts(self, policy_fingerprint: str) -> tuple[int, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status = 'terminal' THEN 1 ELSE 0 END)
                FROM image_outcomes
                WHERE policy_fingerprint = ?
                """,
                (policy_fingerprint,),
            ).fetchone()
        return int(row[0] or 0), int(row[1] or 0)

    def snapshot(self, policy_fingerprint: str) -> dict[str, Any]:
        digest = hashlib.sha256()
        key_digest = hashlib.sha256()
        success = 0
        terminal = 0
        count = 0
        connection = self._connect()
        try:
            for row in connection.execute(
                """
                SELECT url_key, status, outcome_json, payload_sha256
                FROM image_outcomes
                WHERE policy_fingerprint = ?
                ORDER BY url_key
                """,
                (policy_fingerprint,),
            ):
                self._decode(row)
                url_key = str(row["url_key"])
                payload_sha256 = str(row["payload_sha256"])
                digest.update(url_key.encode("utf-8"))
                digest.update(b"\0")
                digest.update(payload_sha256.encode("ascii"))
                digest.update(b"\n")
                key_digest.update(url_key.encode("utf-8"))
                key_digest.update(b"\n")
                status = str(row["status"])
                success += int(status == "success")
                terminal += int(status == "terminal")
                count += 1
        finally:
            connection.close()
        return {
            "count": count,
            "success": success,
            "terminal": terminal,
            "digest": digest.hexdigest(),
            "url_key_digest": key_digest.hexdigest(),
        }

    def snapshot_for_jobs(
        self,
        policy_fingerprint: str,
        jobs: UniqueImageJobs,
        *,
        commit_every: int = 10_000,
    ) -> dict[str, Any]:
        """Digest only the current job-set outcomes using a disk-backed join."""
        if commit_every <= 0:
            raise ValueError("commit_every must be positive")
        digest = hashlib.sha256()
        key_digest = hashlib.sha256()
        success = 0
        terminal = 0
        count = 0
        missing = 0
        inserted = 0
        previous: str | None = None
        connection = self._connect()
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                """
                CREATE TEMP TABLE current_image_job_keys (
                    url_key TEXT PRIMARY KEY
                ) WITHOUT ROWID
                """
            )
            for record in _iter_jsonl([jobs.output_path]):
                url_key = str(record.get("url_key") or "")
                if (
                    not url_key
                    or (previous is not None and url_key <= previous)
                ):
                    raise ValueError(
                        "unique image jobs are not strictly ordered"
                    )
                connection.execute(
                    """
                    INSERT INTO current_image_job_keys (url_key)
                    VALUES (?)
                    """,
                    (url_key,),
                )
                inserted += 1
                previous = url_key
                if inserted % commit_every == 0:
                    connection.commit()
            connection.commit()
            if inserted != jobs.records:
                raise ValueError("unique image job record count mismatch")

            rows = connection.execute(
                """
                SELECT
                    keys.url_key,
                    outcomes.status,
                    outcomes.outcome_json,
                    outcomes.payload_sha256
                FROM current_image_job_keys AS keys
                LEFT JOIN image_outcomes AS outcomes
                  ON outcomes.policy_fingerprint = ?
                 AND outcomes.url_key = keys.url_key
                ORDER BY keys.url_key
                """,
                (policy_fingerprint,),
            )
            for row in rows:
                if row["outcome_json"] is None:
                    missing += 1
                    continue
                self._decode(row)
                url_key = str(row["url_key"])
                payload_sha256 = str(row["payload_sha256"])
                digest.update(url_key.encode("utf-8"))
                digest.update(b"\0")
                digest.update(payload_sha256.encode("ascii"))
                digest.update(b"\n")
                key_digest.update(url_key.encode("utf-8"))
                key_digest.update(b"\n")
                status = str(row["status"])
                success += int(status == "success")
                terminal += int(status == "terminal")
                count += 1
        finally:
            connection.close()
        return {
            "count": count,
            "success": success,
            "terminal": terminal,
            "missing": missing,
            "digest": digest.hexdigest(),
            "url_key_digest": key_digest.hexdigest(),
        }


class _BoundedShardWriter:
    def __init__(
        self,
        directory: Path,
        records_per_shard: int,
    ) -> None:
        self.directory = directory
        self.records_per_shard = records_per_shard
        self.current: AtomicJsonlShard | None = None
        self.current_records = 0
        self.completed: list[CompletedShard] = []

    def write(self, record: dict[str, Any]) -> None:
        if (
            self.current is None
            or self.current_records >= self.records_per_shard
        ):
            self._commit_current()
            path = (
                self.directory
                / f"part-{len(self.completed):05d}.jsonl"
            )
            self.current = AtomicJsonlShard(path)
            self.current_records = 0
        self.current.write(record)
        self.current_records += 1

    def close(self) -> list[CompletedShard]:
        self._commit_current()
        return self.completed

    def abort(self) -> None:
        if self.current is not None:
            self.current.abort()
            self.current = None

    def _commit_current(self) -> None:
        if self.current is None:
            return
        self.completed.append(self.current.commit())
        self.current = None
        self.current_records = 0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def _atomic_json_if_changed(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            if json.loads(path.read_text(encoding="utf-8")) == payload:
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    _atomic_json(path, payload)


def _relative_completed(
    completed: CompletedShard,
    path: Path,
    root: Path,
) -> CompletedShard:
    return CompletedShard(
        path=path.relative_to(root).as_posix(),
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )


def _completed_from_payload(payload: dict[str, Any]) -> CompletedShard:
    return CompletedShard(
        path=str(payload["path"]),
        records=int(payload["records"]),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
    )


def _shard_payload(shard: CompletedShard) -> dict[str, Any]:
    return {
        "path": shard.path,
        "records": shard.records,
        "bytes": shard.bytes,
        "sha256": shard.sha256,
    }


def _load_completed_plan(
    *,
    output_root: Path,
    manifest_path: Path,
    input_fingerprint: str,
    budget: ImageBudget,
    records_per_shard: int,
) -> AssetPlanShards | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_parameters = {
        "attempts_per_entity": budget.attempts_per_entity,
        "retained_per_entity": budget.retained_per_entity,
        "records_per_shard": records_per_shard,
    }
    if (
        payload.get("stage") != "wdc200k_asset_planning"
        or payload.get("input_fingerprint") != input_fingerprint
        or payload.get("parameters") != expected_parameters
    ):
        raise ValueError("asset planning fingerprint mismatch")
    if payload.get("complete") is not True:
        return None
    entity_shards = [
        _completed_from_payload(item)
        for item in payload.get("entity_plan_shards", [])
    ]
    mapping_shards = [
        _completed_from_payload(item)
        for item in payload.get("image_mapping_shards", [])
    ]
    if not all(
        validate_completed_shard(shard, output_root)
        for shard in (*entity_shards, *mapping_shards)
    ):
        raise ValueError("asset planning shard checksum validation failed")
    return AssetPlanShards(
        output_root=output_root,
        entity_plan_paths=tuple(
            output_root / shard.path for shard in entity_shards
        ),
        image_mapping_paths=tuple(
            output_root / shard.path for shard in mapping_shards
        ),
        manifest_path=manifest_path,
        entities=sum(shard.records for shard in entity_shards),
        image_mappings=sum(shard.records for shard in mapping_shards),
    )


def _successful_page(page: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(page, dict):
        return None
    status = page.get("status")
    if status not in {None, "success"}:
        return None
    return page


def iter_entity_page_join(
    entity_paths: Iterable[Path],
    page_fanout: Iterable[dict[str, Any]],
    *,
    join_path: Path,
    commit_every: int = 10_000,
) -> Iterator[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Stream Task 3 entities joined to Task 4 fanout via bounded disk state."""
    if commit_every <= 0:
        raise ValueError("commit_every must be positive")
    join_path = Path(join_path)
    join_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(join_path, timeout=30.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL").fetchone()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TABLE IF EXISTS entity_pages")
        connection.execute(
            """
            CREATE TABLE entity_pages (
                entity_id TEXT PRIMARY KEY,
                page_json TEXT NOT NULL
            )
            """
        )
        connection.commit()

        pending = 0
        for page in page_fanout:
            if not isinstance(page, dict):
                raise ValueError("page fanout record must be an object")
            entity_id = clean_text(page.get("entity_id"))
            if not entity_id:
                raise ValueError("page fanout record has no entity_id")
            encoded = json.dumps(
                page,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            existing = connection.execute(
                """
                SELECT page_json
                FROM entity_pages
                WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
            if existing is not None:
                if str(existing[0]) != encoded:
                    raise ValueError(
                        "conflicting page outcomes for one entity"
                    )
                continue
            connection.execute(
                """
                INSERT INTO entity_pages (entity_id, page_json)
                VALUES (?, ?)
                """,
                (entity_id, encoded),
            )
            pending += 1
            if pending >= commit_every:
                connection.commit()
                pending = 0
        connection.commit()

        for current_entity in _iter_jsonl(entity_paths):
            if not isinstance(current_entity, dict):
                raise ValueError("entity shard record must be an object")
            entity_id = clean_text(current_entity.get("entity_id"))
            if not entity_id:
                raise ValueError("entity shard record has no entity_id")
            row = connection.execute(
                """
                SELECT page_json
                FROM entity_pages
                WHERE entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
            page = None if row is None else json.loads(str(row[0]))
            if page is not None and not isinstance(page, dict):
                raise ValueError("corrupt entity page join record")
            yield current_entity, page
    finally:
        connection.close()


def plan_entity_assets(
    entity: dict[str, Any],
    page: dict[str, Any] | None,
    budget: ImageBudget = ImageBudget(),
) -> EntityAssetPlan:
    """Plan direct-first distinct image candidates for one retained entity."""
    if budget.attempts_per_entity == 0:
        return EntityAssetPlan(
            entity_id=str(entity["entity_id"]),
            image_refs=(),
        )
    entity_id = str(entity["entity_id"])
    appearances = entity.get("appears_in") or [{}]
    appearance = (
        appearances[0] if isinstance(appearances[0], dict) else {}
    )
    source_table_id = str(appearance.get("source_table_id") or "")
    row_id = appearance.get("row_id")
    page_url = clean_text(entity.get("page_url"))
    successful_page = _successful_page(page)
    candidate_groups = (
        (entity.get("image_urls") or [], "wdc_image_column"),
        (
            (successful_page or {}).get("image_urls") or [],
            "wdc_page_image",
        ),
    )
    seen: set[str] = set()
    references: list[ImageReference] = []
    for candidates, source in candidate_groups:
        for candidate in candidates:
            normalized = _normalize_http_url(candidate)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            references.append(
                ImageReference(
                    entity_id=entity_id,
                    source_table_id=source_table_id,
                    row_id=row_id,
                    image_url=normalized,
                    url_key=hashlib.sha256(
                        normalized.encode("utf-8")
                    ).hexdigest(),
                    ordinal=len(references),
                    source=source,
                    page_url=page_url,
                )
            )
            if len(references) >= budget.attempts_per_entity:
                return EntityAssetPlan(
                    entity_id=entity_id,
                    image_refs=tuple(references),
                )
    return EntityAssetPlan(
        entity_id=entity_id,
        image_refs=tuple(references),
    )


def persist_entity_asset_plans(
    entity_pages: Iterable[
        tuple[dict[str, Any], dict[str, Any] | None]
    ],
    *,
    output_root: Path,
    input_fingerprint: str,
    budget: ImageBudget = ImageBudget(),
    records_per_shard: int = 10_000,
) -> AssetPlanShards:
    """Persist entity/page inputs and all selected URL mappings in shards."""
    if not input_fingerprint:
        raise ValueError("input_fingerprint must not be empty")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    output_root = Path(output_root)
    manifest_path = output_root / "asset-planning-manifest.json"
    resumed = _load_completed_plan(
        output_root=output_root,
        manifest_path=manifest_path,
        input_fingerprint=input_fingerprint,
        budget=budget,
        records_per_shard=records_per_shard,
    )
    if resumed is not None:
        return resumed

    entity_writer = _BoundedShardWriter(
        output_root / "entity_plans",
        records_per_shard,
    )
    mapping_writer = _BoundedShardWriter(
        output_root / "image_mappings",
        records_per_shard,
    )
    try:
        for entity, page_outcome in entity_pages:
            plan = plan_entity_assets(entity, page_outcome, budget)
            entity_writer.write(
                {
                    "entity": entity,
                    "page": page_outcome,
                    "page_was_required": plan.page_was_required,
                }
            )
            for reference in plan.image_refs:
                mapping_writer.write(reference.as_record())
        entity_completed = entity_writer.close()
        mapping_completed = mapping_writer.close()
    except BaseException:
        entity_writer.abort()
        mapping_writer.abort()
        raise

    entity_shards = [
        _relative_completed(
            completed,
            output_root / "entity_plans" / completed.path,
            output_root,
        )
        for completed in entity_completed
    ]
    mapping_shards = [
        _relative_completed(
            completed,
            output_root / "image_mappings" / completed.path,
            output_root,
        )
        for completed in mapping_completed
    ]
    _atomic_json(
        manifest_path,
        {
            "stage": "wdc200k_asset_planning",
            "input_fingerprint": input_fingerprint,
            "parameters": {
                "attempts_per_entity": budget.attempts_per_entity,
                "retained_per_entity": budget.retained_per_entity,
                "records_per_shard": records_per_shard,
            },
            "entity_plan_shards": [
                _shard_payload(shard) for shard in entity_shards
            ],
            "image_mapping_shards": [
                _shard_payload(shard) for shard in mapping_shards
            ],
            "complete": True,
        },
    )
    return AssetPlanShards(
        output_root=output_root,
        entity_plan_paths=tuple(
            output_root / shard.path for shard in entity_shards
        ),
        image_mapping_paths=tuple(
            output_root / shard.path for shard in mapping_shards
        ),
        manifest_path=manifest_path,
        entities=sum(shard.records for shard in entity_shards),
        image_mappings=sum(shard.records for shard in mapping_shards),
    )


def build_unique_image_jobs(
    planned: AssetPlanShards,
    output_path: Path,
    *,
    chunk_records: int = 100_000,
    merge_fan_in: int = 64,
) -> UniqueImageJobs:
    """Publish one image job per globally unique normalized URL."""
    resumed = _validated_planning_result(planned)
    output_path = Path(output_path)
    planning_manifest_sha256 = _sha256_file(resumed.manifest_path)
    input_fingerprint = stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        planning_manifest_sha256,
        length=40,
    )
    parameter_fingerprint = stable_hash(
        UNIQUE_IMAGE_JOB_SCHEMA_VERSION,
        "normalize-http-url-v1",
        "url-key-sha256-v1",
        "first-record-per-url-key",
        chunk_records,
        merge_fan_in,
        length=40,
    )
    manifest_path = output_path.with_suffix(
        output_path.suffix + ".manifest.json"
    )
    manifest = StageManifest(
        manifest_path,
        StageFingerprint(
            stage=UNIQUE_IMAGE_JOB_SCHEMA_VERSION,
            input_fingerprint=input_fingerprint,
            parameter_fingerprint=parameter_fingerprint,
        ),
    )
    if manifest.complete:
        if len(manifest.completed_shards) != 1:
            raise ValueError("unique image job manifest has invalid shards")
        completed = manifest.completed_shards[0]
        if not validate_completed_shard(completed, output_path.parent):
            raise ValueError("unique image job shard checksum validation failed")
    else:
        completed = external_unique_jsonl(
            resumed.image_mapping_paths,
            output_path,
            key_fn=lambda record: record["url_key"],
            chunk_records=chunk_records,
            merge_fan_in=merge_fan_in,
        )
        manifest.record_shard(completed)
        manifest.mark_complete()
    return UniqueImageJobs(
        output_path=output_path,
        manifest_path=manifest_path,
        completed_shard=completed,
        input_fingerprint=input_fingerprint,
        parameter_fingerprint=parameter_fingerprint,
    )


def _validated_unique_image_jobs(
    jobs: UniqueImageJobs,
) -> UniqueImageJobs:
    manifest = StageManifest(
        jobs.manifest_path,
        StageFingerprint(
            stage=UNIQUE_IMAGE_JOB_SCHEMA_VERSION,
            input_fingerprint=jobs.input_fingerprint,
            parameter_fingerprint=jobs.parameter_fingerprint,
        ),
    )
    if not manifest.complete or len(manifest.completed_shards) != 1:
        raise ValueError("unique image job stage is incomplete")
    completed = manifest.completed_shards[0]
    if completed != jobs.completed_shard:
        raise ValueError("unique image job result does not match manifest")
    if not validate_completed_shard(completed, jobs.output_path.parent):
        raise ValueError("unique image job shard checksum validation failed")
    return jobs


def _validated_planning_result(
    planned: AssetPlanShards,
) -> AssetPlanShards:
    """Reload a complete plan and validate every consumed shard checksum."""
    payload = json.loads(
        planned.manifest_path.read_text(encoding="utf-8")
    )
    parameters = payload.get("parameters") or {}
    resumed = _load_completed_plan(
        output_root=planned.output_root,
        manifest_path=planned.manifest_path,
        input_fingerprint=str(payload.get("input_fingerprint") or ""),
        budget=ImageBudget(
            attempts_per_entity=int(
                parameters["attempts_per_entity"]
            ),
            retained_per_entity=int(
                parameters["retained_per_entity"]
            ),
        ),
        records_per_shard=int(parameters["records_per_shard"]),
    )
    if resumed is None:
        raise ValueError("asset planning is not complete")
    return resumed


def validate_asset_plan_shards(
    planned: AssetPlanShards,
    *,
    expected_input_fingerprint: str,
) -> AssetPlanShards:
    """Validate planning shards against an independently derived input."""
    resumed = _validated_planning_result(planned)
    payload = json.loads(
        Path(planned.manifest_path).read_text(encoding="utf-8")
    )
    if payload.get("input_fingerprint") != expected_input_fingerprint:
        raise ValueError("asset planning input fingerprint mismatch")
    if resumed != planned:
        raise ValueError("asset planning result does not match manifest")
    return resumed


def validate_unique_image_jobs(
    jobs: UniqueImageJobs,
    *,
    planned: AssetPlanShards,
) -> UniqueImageJobs:
    """Validate unique image jobs against the exact planning manifest."""
    planned = _validated_planning_result(planned)
    expected_input = stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        _sha256_file(planned.manifest_path),
        length=40,
    )
    if jobs.input_fingerprint != expected_input:
        raise ValueError("unique image jobs planning fingerprint mismatch")
    return _validated_unique_image_jobs(jobs)


def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _image_policy_fingerprint(policy: FetchPolicy) -> str:
    return stable_hash(
        "wdc200k-image-fetch-v1",
        policy.fingerprint,
        length=40,
    )


def _image_kind(
    policy_fingerprint: str,
    unique_jobs: UniqueImageJobs,
) -> str:
    return (
        f"wdc200k-image:{policy_fingerprint}:"
        f"{_image_job_set_fingerprint(unique_jobs)}"
    )


def _image_job_set_fingerprint(
    unique_jobs: UniqueImageJobs,
) -> str:
    return stable_hash(
        unique_jobs.input_fingerprint,
        unique_jobs.parameter_fingerprint,
        unique_jobs.completed_shard.sha256,
        length=40,
    )


def _image_fetch_manifest_path(
    outcomes_path: Path,
    kind: str,
) -> Path:
    manifest_root = outcomes_path.with_suffix(
        outcomes_path.suffix + ".fetch-manifests"
    )
    return manifest_root / f"{stable_hash(kind, length=40)}.json"


def _enqueue_unique_images(
    paths: Iterable[Path],
    *,
    store: SqliteJobStore,
    kind: str,
    policy_fingerprint: str,
    commit_every: int = 10_000,
) -> None:
    connection = store._connect()
    pending = 0
    try:
        for record in _iter_jsonl(paths):
            image_url = _normalize_http_url(record.get("image_url"))
            if image_url is None:
                raise ValueError("image job URL is not valid HTTP(S)")
            url_key = hashlib.sha256(
                image_url.encode("utf-8")
            ).hexdigest()
            if record.get("url_key") != url_key:
                raise ValueError("image job url_key does not match image_url")
            payload = {
                "url_key": url_key,
                "image_url": image_url,
                "host": str(urlsplit(image_url).hostname or "").casefold(),
                "entity_id": str(record.get("entity_id") or ""),
                "source": str(record.get("source") or "wdc_page_image"),
                "page_url": str(record.get("page_url") or ""),
                "policy_fingerprint": policy_fingerprint,
            }
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs (
                    job_id, kind, payload_json, status, updated_at
                ) VALUES (?, ?, ?, 'pending', ?)
                """,
                (
                    f"{kind}:{url_key}",
                    kind,
                    json.dumps(payload, ensure_ascii=False),
                    time.time(),
                ),
            )
            pending += 1
            if pending >= commit_every:
                connection.commit()
                pending = 0
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _job_count(store: SqliteJobStore, kind: str) -> int:
    with store._connect() as connection:
        return int(
            connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = ?",
                (kind,),
            ).fetchone()[0]
        )


def _job_snapshot(store: SqliteJobStore, kind: str) -> dict[str, int]:
    with store._connect() as connection:
        rows = {
            str(row["status"]): int(row["count"])
            for row in connection.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM jobs
                WHERE kind = ?
                GROUP BY status
                """,
                (kind,),
            )
        }
    return {
        "total": sum(rows.values()),
        "success": rows.get("success", 0),
        "terminal": rows.get("terminal", 0),
        "pending": rows.get("pending", 0),
        "retryable": rows.get("retryable", 0),
        "leased": rows.get("leased", 0),
    }


def _unique_job_url_key_digest(jobs: UniqueImageJobs) -> str:
    digest = hashlib.sha256()
    count = 0
    previous: str | None = None
    for record in _iter_jsonl([jobs.output_path]):
        url_key = str(record.get("url_key") or "")
        if not url_key or (previous is not None and url_key <= previous):
            raise ValueError("unique image jobs are not strictly ordered")
        digest.update(url_key.encode("utf-8"))
        digest.update(b"\n")
        previous = url_key
        count += 1
    if count != jobs.records:
        raise ValueError("unique image job record count mismatch")
    return digest.hexdigest()


_CONTENT_LOCKS = tuple(threading.Lock() for _ in range(128))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_extension(outcome: dict[str, Any]) -> str:
    extensions = {
        "image/bmp": ".bmp",
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }
    extension = extensions.get(
        clean_text(outcome.get("mime_type")).casefold()
    )
    if extension:
        return extension
    suffix = Path(clean_text(outcome.get("file_name"))).suffix.casefold()
    return suffix if suffix in set(extensions.values()) else ".img"


def _content_address_outcome(
    outcome: dict[str, Any],
    image_dir: Path,
) -> dict[str, Any]:
    image_dir.mkdir(parents=True, exist_ok=True)
    content_hash = clean_text(outcome.get("sha256"))
    source = Path(clean_text(outcome.get("local_path")))
    if (
        len(content_hash) != 64
        or any(character not in "0123456789abcdef" for character in content_hash)
        or not source.is_file()
        or _sha256_file(source) != content_hash
    ):
        raise ValueError("downloaded image content hash is invalid")
    target = (
        image_dir
        / f"image_{content_hash}{_content_extension(outcome)}"
    )
    lock = _CONTENT_LOCKS[
        int(content_hash[:8], 16) % len(_CONTENT_LOCKS)
    ]
    with lock:
        if source.resolve() != target.resolve():
            try:
                os.link(source, target)
            except FileExistsError:
                if _sha256_file(target) != content_hash:
                    raise ValueError(
                        "content-addressed image hash collision"
                    )
            source.unlink(missing_ok=True)
        elif _sha256_file(target) != content_hash:
            raise ValueError("content-addressed image hash mismatch")
    return {
        **outcome,
        "file_name": target.name,
        "local_path": str(target),
        "relative_path": f"{image_dir.name}/{target.name}",
        "sha256": content_hash,
    }


def _transport_image_outcome(
    transport: Any,
    image_url: str,
) -> dict[str, Any] | None:
    getter = getattr(transport, "cached_image_outcome", None)
    if not callable(getter):
        return None
    cached = getter(image_url)
    if not isinstance(cached, dict):
        return None
    if cached.get("status") == "success":
        cached = dict(cached)
        file_name = clean_text(cached.get("file_name"))
        image_root = getattr(transport, "image_dir", None)
        cache_root = getattr(transport, "cache_dir", None)
        if (
            not clean_text(cached.get("local_path"))
            and image_root is not None
            and file_name
        ):
            local_path = Path(image_root) / file_name
            cached["local_path"] = str(local_path)
            if cache_root is not None:
                try:
                    cached["relative_path"] = (
                        local_path.relative_to(Path(cache_root)).as_posix()
                    )
                except ValueError:
                    cached["relative_path"] = file_name
            else:
                cached["relative_path"] = file_name
            cached["downloaded"] = False
        return {
            **cached,
            "status": "success",
            "original_url": (
                cached.get("original_url") or image_url
            ),
        }
    if cached.get("status") in {"terminal", "retryable"}:
        return {
            "status": "terminal",
            "image_url": image_url,
            "original_url": image_url,
            "error_class": (
                clean_text(cached.get("error_class"))
                or "cached_failure"
            ),
            **(
                {}
                if cached.get("http_status") is None
                else {"http_status": int(cached["http_status"])}
            ),
        }
    return None


def _fetch_image_job(
    payload: dict[str, Any],
    *,
    transport: Any,
    image_dir: Path,
) -> dict[str, Any]:
    image_url = str(payload["image_url"])
    cached = _transport_image_outcome(transport, image_url)
    if cached is not None:
        if cached["status"] == "success":
            return _content_address_outcome(cached, image_dir)
        return cached
    try:
        downloaded = transport.download_image(
            image_url,
            page_url=str(payload["page_url"]),
            source=str(payload["source"]),
            entity_id=str(payload["entity_id"]),
        )
    except Exception as error:
        cached = _transport_image_outcome(transport, image_url)
        if cached is not None:
            return cached
        return {
            "status": "terminal",
            "image_url": image_url,
            "original_url": image_url,
            "error_class": type(error).__name__,
        }
    if isinstance(downloaded, dict):
        return _content_address_outcome(
            {
                **downloaded,
                "status": "success",
                "original_url": (
                    downloaded.get("original_url") or image_url
                ),
            },
            image_dir,
        )
    cached = _transport_image_outcome(transport, image_url)
    if cached is not None:
        return cached
    return {
        "status": "terminal",
        "image_url": image_url,
        "original_url": image_url,
        "error_class": "download_or_validation_failed",
    }


def _execute_image_job(
    payload: dict[str, Any],
    *,
    transport: Any,
    image_dir: Path,
    outcome_store: ImageOutcomeStore,
    policy_fingerprint: str,
    claim_owner: str,
    claim_lease_seconds: float,
    claim_poll_seconds: float,
    after_url_claim: Callable[[ImageUrlLease], None] | None,
) -> ImageJobExecution:
    """Wait for or acquire the shared policy+URL claim before networking."""
    url_key = str(payload["url_key"])
    while True:
        decision = outcome_store.claim_url(
            policy_fingerprint,
            url_key,
            owner=claim_owner,
            lease_seconds=claim_lease_seconds,
        )
        if decision.outcome is not None:
            return ImageJobExecution(
                outcome=decision.outcome,
                lease=None,
            )
        if decision.lease is not None:
            if after_url_claim is not None:
                after_url_claim(decision.lease)
            return ImageJobExecution(
                outcome=_fetch_image_job(
                    payload,
                    transport=transport,
                    image_dir=image_dir,
                ),
                lease=decision.lease,
            )
        if decision.retry_at is None:
            raise RuntimeError("image URL claim made no decision")
        wait_seconds = min(
            claim_poll_seconds,
            max(0.001, decision.retry_at - time.time()),
        )
        time.sleep(wait_seconds)


def fetch_unique_images(
    unique_jobs: UniqueImageJobs,
    store: SqliteJobStore,
    transport: Any,
    policy: FetchPolicy = FetchPolicy(
        policy_version="wdc200k-image-fetch-v1",
    ),
    *,
    outcomes_path: Path | None = None,
    image_dir: Path,
    claim_buffer: int | None = None,
    lease_seconds: float | None = None,
    url_claim_lease_seconds: float | None = None,
    url_claim_poll_seconds: float = 0.05,
    after_url_claim: Callable[[ImageUrlLease], None] | None = None,
    after_cache_write: Callable[[dict[str, Any]], None] | None = None,
) -> ImageFetchResult:
    """Fetch globally unique image URLs with bounded fair durable jobs."""
    unique_jobs = _validated_unique_image_jobs(unique_jobs)
    if policy.retries != 0:
        raise ValueError("image fetching permits exactly zero retries")
    transport_policy = getattr(
        transport,
        "network_policy_fingerprint",
        None,
    )
    if transport_policy != policy.network_policy_fingerprint:
        raise ValueError(
            "transport network policy fingerprint does not match FetchPolicy"
        )
    if int(getattr(transport, "max_retries", 0)) != 0:
        raise ValueError("image transport must use zero retries")
    if (
        float(
            getattr(
                transport,
                "max_response_seconds",
                policy.deadline_seconds,
            )
        )
        != policy.deadline_seconds
    ):
        raise ValueError(
            "image transport deadline does not match FetchPolicy"
        )
    fingerprint = _image_policy_fingerprint(policy)
    kind = _image_kind(fingerprint, unique_jobs)
    outcomes_path = Path(
        outcomes_path
        or store.path.with_name(
            f"{store.path.stem}-image-outcomes.sqlite3"
        )
    )
    outcome_store = ImageOutcomeStore(outcomes_path)
    _enqueue_unique_images(
        [unique_jobs.output_path],
        store=store,
        kind=kind,
        policy_fingerprint=fingerprint,
    )
    unique = unique_jobs.records
    enqueued = _job_count(store, kind)
    if enqueued != unique:
        raise ValueError("image job set was not enqueued completely")
    owner = f"image-{os.getpid()}-{uuid.uuid4().hex}"
    buffer_limit = (
        policy.global_concurrency * 4
        if claim_buffer is None
        else int(claim_buffer)
    )
    if buffer_limit < policy.global_concurrency:
        raise ValueError("claim_buffer must be at least global_concurrency")
    if url_claim_poll_seconds <= 0:
        raise ValueError("url_claim_poll_seconds must be positive")
    effective_url_claim_lease = (
        policy.deadline_seconds + 30.0
        if url_claim_lease_seconds is None
        else float(url_claim_lease_seconds)
    )
    if effective_url_claim_lease <= 0:
        raise ValueError("url_claim_lease_seconds must be positive")
    effective_lease = (
        float(lease_seconds)
        if lease_seconds is not None
        else policy.deadline_seconds * (buffer_limit + 2) + 30.0
    )
    host_queues: dict[str, deque[Any]] = {}
    active_by_host: dict[str, int] = {}
    ready_hosts: deque[str] = deque()
    ready_set: set[str] = set()
    futures: dict[Future[ImageJobExecution], tuple[Any, str]] = {}
    claimed_buffered = 0
    maximum_claimed = 0
    maximum_inflight = 0
    maximum_host_states = 0

    def add_ready(host: str) -> None:
        if host_queues.get(host) and host not in ready_set:
            ready_set.add(host)
            ready_hosts.append(host)

    def finish_cached(job: Any, outcome: dict[str, Any]) -> None:
        outcome_store.clear_claim_if_outcome(
            fingerprint,
            str(job.payload["url_key"]),
        )
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
        nonlocal claimed_buffered, maximum_claimed, maximum_host_states
        need = buffer_limit - claimed_buffered - len(futures)
        if need <= 0:
            return 0
        claimed = store.claim(
            kind,
            limit=need,
            owner=owner,
            lease_seconds=effective_lease,
        )
        for job in claimed:
            cached = outcome_store.get(
                fingerprint,
                str(job.payload["url_key"]),
            )
            if cached is not None:
                finish_cached(job, cached)
                continue
            host = str(job.payload["host"])
            host_queues.setdefault(host, deque()).append(job)
            claimed_buffered += 1
            add_ready(host)
        maximum_host_states = max(
            maximum_host_states,
            len(host_queues),
        )
        maximum_claimed = max(
            maximum_claimed,
            claimed_buffered + len(futures),
        )
        return len(claimed)

    def submit_ready(pool: ThreadPoolExecutor) -> None:
        nonlocal claimed_buffered, maximum_inflight
        rotations = len(ready_hosts)
        while (
            ready_hosts
            and len(futures) < policy.global_concurrency
            and rotations > 0
        ):
            host = ready_hosts.popleft()
            ready_set.discard(host)
            queue = host_queues[host]
            if not queue:
                if active_by_host.get(host, 0) == 0:
                    host_queues.pop(host, None)
                rotations -= 1
                continue
            if (
                active_by_host.get(host, 0)
                >= policy.per_host_concurrency
            ):
                add_ready(host)
                rotations -= 1
                continue
            job = queue.popleft()
            claimed_buffered -= 1
            active_by_host[host] = active_by_host.get(host, 0) + 1
            future = pool.submit(
                _execute_image_job,
                job.payload,
                transport=transport,
                image_dir=Path(image_dir),
                outcome_store=outcome_store,
                policy_fingerprint=fingerprint,
                claim_owner=(
                    f"{owner}:{job.job_id}:{job.lease_id}"
                ),
                claim_lease_seconds=effective_url_claim_lease,
                claim_poll_seconds=float(url_claim_poll_seconds),
                after_url_claim=after_url_claim,
            )
            futures[future] = (job, host)
            add_ready(host)
            maximum_inflight = max(maximum_inflight, len(futures))
            rotations = len(ready_hosts) or 1

    with ThreadPoolExecutor(
        max_workers=policy.global_concurrency,
        thread_name_prefix="wdc-image",
    ) as pool:
        while True:
            claimed_now = claim_more()
            submit_ready(pool)
            if not futures:
                if claimed_buffered:
                    raise RuntimeError("image scheduler made no progress")
                if claimed_now == 0:
                    break
                continue
            completed, _pending = wait(
                tuple(futures),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                job, host = futures.pop(future)
                active_by_host[host] -= 1
                if active_by_host[host] == 0:
                    del active_by_host[host]
                if host_queues.get(host):
                    add_ready(host)
                else:
                    host_queues.pop(host, None)
                    ready_set.discard(host)
                execution = future.result()
                if execution.lease is None:
                    persisted = execution.outcome
                    outcome_store.clear_claim_if_outcome(
                        fingerprint,
                        str(job.payload["url_key"]),
                    )
                else:
                    persisted = outcome_store.put_claimed(
                        fingerprint,
                        str(job.payload["url_key"]),
                        str(job.payload["image_url"]),
                        execution.outcome,
                        lease=execution.lease,
                    )
                    if after_cache_write is not None:
                        after_cache_write(persisted)
                    released = outcome_store.finish_claim(
                        fingerprint,
                        str(job.payload["url_key"]),
                        lease=execution.lease,
                    )
                    if not released:
                        outcome_store.clear_claim_if_outcome(
                            fingerprint,
                            str(job.payload["url_key"]),
                        )
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
            submit_ready(pool)

    outcome_snapshot = outcome_store.snapshot_for_jobs(
        fingerprint,
        unique_jobs,
    )
    success = int(outcome_snapshot["success"])
    terminal = int(outcome_snapshot["terminal"])
    job_snapshot = _job_snapshot(store, kind)
    unique_key_digest = _unique_job_url_key_digest(unique_jobs)
    complete = (
        success + terminal == unique
        and int(outcome_snapshot["count"]) == unique
        and job_snapshot["success"] == success
        and job_snapshot["terminal"] == terminal
        and job_snapshot["total"] == unique
        and job_snapshot["pending"] == 0
        and job_snapshot["retryable"] == 0
        and job_snapshot["leased"] == 0
        and int(outcome_snapshot["missing"]) == 0
        and outcome_snapshot["url_key_digest"] == unique_key_digest
    )
    fetch_manifest_path = _image_fetch_manifest_path(
        outcomes_path,
        kind,
    )
    fetch_manifest = {
        "stage": "wdc200k-image-fetch-v1",
        "complete": complete,
        "policy_fingerprint": fingerprint,
        "unique_jobs": {
            "manifest_sha256": _sha256_file(unique_jobs.manifest_path),
            "input_fingerprint": unique_jobs.input_fingerprint,
            "parameter_fingerprint": unique_jobs.parameter_fingerprint,
            "path": unique_jobs.completed_shard.path,
            "records": unique_jobs.records,
            "bytes": unique_jobs.completed_shard.bytes,
            "sha256": unique_jobs.completed_shard.sha256,
            "url_key_digest": unique_key_digest,
        },
        "job_store": {
            "kind": kind,
            **job_snapshot,
        },
        "outcomes": {
            "path": str(outcomes_path),
            **outcome_snapshot,
        },
    }
    _atomic_json_if_changed(fetch_manifest_path, fetch_manifest)
    fetch_manifest_sha256 = _sha256_file(fetch_manifest_path)
    return ImageFetchResult(
        unique=unique,
        success=success,
        terminal=terminal,
        complete=complete,
        outcomes_path=outcomes_path,
        policy_fingerprint=fingerprint,
        maximum_inflight=maximum_inflight,
        maximum_claimed=maximum_claimed,
        maximum_host_states=maximum_host_states,
        unique_jobs=unique_jobs,
        job_store_path=store.path,
        job_kind=kind,
        fetch_manifest_path=fetch_manifest_path,
        fetch_manifest_sha256=fetch_manifest_sha256,
        outcome_digest=str(outcome_snapshot["digest"]),
        outcomes_count=int(outcome_snapshot["count"]),
        outcome_url_key_digest=str(
            outcome_snapshot["url_key_digest"]
        ),
        leased=job_snapshot["leased"],
        remaining=max(
            (
                job_snapshot["pending"]
                + job_snapshot["retryable"]
                + job_snapshot["leased"]
            ),
            unique - success - terminal,
        ),
    )


def iter_image_outcomes(
    outcomes_path: Path,
    policy_fingerprint: str,
) -> Iterator[dict[str, Any]]:
    """Stream durable unique image outcomes without all-result loading."""
    yield from ImageOutcomeStore(Path(outcomes_path)).iter(
        policy_fingerprint
    )


def iter_image_failures(
    result: ImageFetchResult,
    *,
    planned: AssetPlanShards,
    aggregation_database: Path,
) -> Iterator[dict[str, Any]]:
    """Aggregate current-jobset failure fanout without loading mappings."""
    planned = _validated_planning_result(planned)
    validate_unique_image_jobs(result.unique_jobs, planned=planned)
    validate_complete_image_fetch(
        result,
        unique_jobs=result.unique_jobs,
    )
    outcome_store = ImageOutcomeStore(result.outcomes_path)
    aggregation_database = Path(aggregation_database)
    aggregation_database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(aggregation_database) as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS current_image_failure_jobs (
                url_key TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                page_url TEXT NOT NULL,
                source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS current_image_failure_refs (
                url_key TEXT NOT NULL,
                reference_key TEXT NOT NULL,
                PRIMARY KEY (url_key, reference_key)
            );
            DELETE FROM current_image_failure_jobs;
            DELETE FROM current_image_failure_refs;
            """
        )
        for job in _iter_jsonl((result.unique_jobs.output_path,)):
            url_key = clean_text(job.get("url_key"))
            if not url_key:
                raise ValueError("unique image failure job has no url_key")
            connection.execute(
                """
                INSERT INTO current_image_failure_jobs (
                    url_key, entity_id, page_url, source
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    url_key,
                    clean_text(job.get("entity_id")),
                    clean_text(job.get("page_url")),
                    clean_text(job.get("source")),
                ),
            )
        for mapping in _iter_jsonl(planned.image_mapping_paths):
            url_key = clean_text(mapping.get("url_key"))
            current = connection.execute(
                """
                SELECT 1 FROM current_image_failure_jobs
                WHERE url_key = ?
                """,
                (url_key,),
            ).fetchone()
            if current is None:
                continue
            reference_key = stable_hash(
                ASSET_PLANNING_SCHEMA_VERSION,
                json.dumps(
                    mapping,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                length=64,
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO current_image_failure_refs (
                    url_key, reference_key
                ) VALUES (?, ?)
                """,
                (url_key, reference_key),
            )
        connection.commit()
        for job in connection.execute(
            """
            SELECT jobs.url_key, jobs.entity_id, jobs.page_url,
                   jobs.source, COUNT(refs.reference_key) AS ref_count
            FROM current_image_failure_jobs AS jobs
            LEFT JOIN current_image_failure_refs AS refs
              ON refs.url_key = jobs.url_key
            GROUP BY jobs.url_key
            ORDER BY jobs.url_key
            """
        ):
            outcome = outcome_store.get(
                result.policy_fingerprint,
                str(job["url_key"]),
            )
            if outcome is None or outcome.get("status") != "terminal":
                continue
            affected_reference_count = int(job["ref_count"])
            if affected_reference_count <= 0:
                raise ValueError(
                    "image failure job has no planning references"
                )
            yield {
                "failure_type": "media_download_failure",
                "stage": "image_fetch",
                "status": "terminal",
                "entity_id": str(job["entity_id"]),
                "page_url": str(job["page_url"]),
                "source": str(job["source"]),
                "image_url": clean_text(outcome.get("image_url")),
                "url_key": str(job["url_key"]),
                "error_class": clean_text(
                    outcome.get("error_class")
                ),
                "http_status": outcome.get("http_status"),
                "affected_reference_count": (
                    affected_reference_count
                ),
                "policy_fingerprint": result.policy_fingerprint,
            }


def _table_asset_links(
    entity: dict[str, Any],
    asset_ids: list[str],
) -> list[dict[str, Any]]:
    entity_id = str(entity["entity_id"])
    normalized_title = normalize_title(str(entity.get("wiki_title") or ""))
    links: list[dict[str, Any]] = []
    for appearance in entity.get("appears_in") or []:
        if not isinstance(appearance, dict):
            continue
        source_table_id = str(appearance.get("source_table_id") or "")
        query_view_id = appearance.get("query_view_id")
        row_id = appearance.get("row_id")
        column_index = appearance.get("column_index")
        links.append(
            {
                "link_id": (
                    "link_"
                    + stable_hash(
                        source_table_id,
                        query_view_id,
                        row_id,
                        column_index,
                        entity_id,
                    )
                ),
                "source_table_id": source_table_id,
                "query_view_id": query_view_id,
                "row_id": row_id,
                "column_index": column_index,
                "column_name": appearance.get("column_name"),
                "cell_text": (
                    (entity.get("display_texts") or [""])[0]
                    if entity.get("display_texts")
                    else ""
                ),
                "entity_id": entity_id,
                "entity_wiki_title": normalized_title,
                "asset_ids": list(asset_ids),
            }
        )
    return links


def materialize_entity_assets(
    entity: dict[str, Any],
    page: dict[str, Any] | None,
    image_outcomes: dict[str, dict[str, Any]],
    budget: ImageBudget = ImageBudget(),
    *,
    text_asset_chunk_chars: int = 800,
    min_text_asset_chunk_chars: int = 120,
    max_text_asset_chunks_per_entity: int = 3,
    image_dir: Path | None = None,
) -> MaterializedEntityAssets:
    """Materialize current-reader-compatible records for one entity."""
    entity_id = str(entity["entity_id"])
    records: list[dict[str, Any]] = []
    successful_page = _successful_page(page)
    page_url = clean_text(entity.get("page_url"))
    if successful_page is not None:
        text_chunks = split_text_asset_content(
            successful_page.get("text"),
            max_chars=text_asset_chunk_chars,
            min_chars=min_text_asset_chunk_chars,
            max_chunks=0,
        )
        selected_chunks = select_relevant_text_chunks(
            text_chunks,
            entity,
            max_text_asset_chunks_per_entity,
        )
        source_asset_id = (
            "asset_text_"
            + stable_hash(entity_id, "wdc_page_text")
        )
        final_page_url = (
            clean_text(successful_page.get("final_url")) or page_url
        )
        for chunk_index, chunk, score in selected_chunks:
            records.append(
                {
                    "asset_id": (
                        f"{source_asset_id}_{chunk_index:03d}"
                    ),
                    "source_asset_id": source_asset_id,
                    "entity_id": entity_id,
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "text",
                    "content": chunk,
                    "text_chunk_index": chunk_index,
                    "text_chunk_count": len(text_chunks),
                    "selected_text_chunk_count": len(selected_chunks),
                    "text_chunk_relevance_score": round(score, 6),
                    "source": "wdc_page_text_chunk",
                    "url": final_page_url,
                    "page_url": page_url,
                    "final_url": final_page_url,
                }
            )

    plan = plan_entity_assets(entity, successful_page, budget)
    seen_content: set[str] = set()
    retained = 0
    if budget.retained_per_entity == 0:
        asset_ids = [record["asset_id"] for record in records]
        return MaterializedEntityAssets(
            entity_id=entity_id,
            bridge_assets=records,
            table_asset_links=_table_asset_links(entity, asset_ids),
        )
    for reference in plan.image_refs:
        outcome = image_outcomes.get(reference.image_url)
        if outcome is None:
            outcome = image_outcomes.get(reference.url_key)
        if (
            not isinstance(outcome, dict)
            or outcome.get("status") != "success"
        ):
            continue
        content_hash = clean_text(outcome.get("sha256"))
        if content_hash and content_hash in seen_content:
            continue
        if content_hash:
            seen_content.add(content_hash)
        file_name = clean_text(outcome.get("file_name"))
        local_path = clean_text(outcome.get("local_path"))
        relative_path = clean_text(outcome.get("relative_path"))
        if image_dir is not None and file_name:
            canonical_path = Path(image_dir) / file_name
            local_path = str(canonical_path)
            relative_path = (
                canonical_path.relative_to(Path(image_dir).parent).as_posix()
            )
        original_url = (
            clean_text(outcome.get("original_url"))
            or reference.image_url
        )
        records.append(
            {
                "asset_id": (
                    "asset_img_"
                    + stable_hash(
                        entity_id,
                        reference.source,
                        original_url,
                        length=20,
                    )
                ),
                "entity_id": entity_id,
                "asset_type": "image",
                "source": reference.source,
                "image_url": original_url,
                "original_url": original_url,
                "final_url": (
                    clean_text(outcome.get("final_url"))
                    or original_url
                ),
                "page_url": page_url,
                "local_path": local_path,
                "relative_path": relative_path,
                "file_name": file_name,
                "bytes": int(outcome.get("bytes") or 0),
                "sha256": content_hash,
                "width": int(outcome.get("width") or 0),
                "height": int(outcome.get("height") or 0),
                "mime_type": clean_text(outcome.get("mime_type")),
                "downloaded": bool(outcome.get("downloaded", False)),
            }
        )
        retained += 1
        if retained >= budget.retained_per_entity:
            break
    asset_ids = [record["asset_id"] for record in records]
    return MaterializedEntityAssets(
        entity_id=entity_id,
        bridge_assets=records,
        table_asset_links=_table_asset_links(entity, asset_ids),
    )


def _load_materialized_assets(
    *,
    output_root: Path,
    manifest_path: Path,
    expected: dict[str, Any],
) -> MaterializedAssetShards | None:
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("stage") != "wdc200k_asset_materialization"
        or payload.get("fingerprint") != expected
    ):
        raise ValueError("asset materialization fingerprint mismatch")
    if payload.get("complete") is not True:
        return None
    asset_shards = [
        _completed_from_payload(item)
        for item in payload.get("bridge_asset_shards", [])
    ]
    link_shards = [
        _completed_from_payload(item)
        for item in payload.get("table_asset_link_shards", [])
    ]
    if not all(
        validate_completed_shard(shard, output_root)
        for shard in (*asset_shards, *link_shards)
    ):
        raise ValueError(
            "asset materialization shard checksum validation failed"
        )
    return MaterializedAssetShards(
        output_root=output_root,
        bridge_asset_paths=tuple(
            output_root / shard.path for shard in asset_shards
        ),
        table_asset_link_paths=tuple(
            output_root / shard.path for shard in link_shards
        ),
        manifest_path=manifest_path,
        bridge_assets=sum(shard.records for shard in asset_shards),
        table_asset_links=sum(shard.records for shard in link_shards),
    )


def _validate_complete_image_fetch(
    result: ImageFetchResult,
) -> dict[str, Any]:
    if not result.complete:
        raise ValueError("image fetch result is not complete")
    unique_jobs = _validated_unique_image_jobs(result.unique_jobs)
    expected_kind = _image_kind(
        result.policy_fingerprint,
        unique_jobs,
    )
    expected_manifest_path = _image_fetch_manifest_path(
        result.outcomes_path,
        expected_kind,
    )
    if (
        result.job_kind != expected_kind
        or result.fetch_manifest_path.resolve()
        != expected_manifest_path.resolve()
    ):
        raise ValueError("image fetch job-set identity mismatch")
    if not result.fetch_manifest_path.is_file():
        raise ValueError("image fetch manifest is missing")
    actual_manifest_sha256 = _sha256_file(result.fetch_manifest_path)
    if actual_manifest_sha256 != result.fetch_manifest_sha256:
        raise ValueError("image fetch manifest checksum mismatch")
    manifest = json.loads(
        result.fetch_manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.get("stage") != "wdc200k-image-fetch-v1"
        or manifest.get("complete") is not True
        or manifest.get("policy_fingerprint")
        != result.policy_fingerprint
    ):
        raise ValueError("image fetch manifest is not complete")

    unique_manifest = manifest.get("unique_jobs") or {}
    expected_unique = {
        "manifest_sha256": _sha256_file(unique_jobs.manifest_path),
        "input_fingerprint": unique_jobs.input_fingerprint,
        "parameter_fingerprint": unique_jobs.parameter_fingerprint,
        "path": unique_jobs.completed_shard.path,
        "records": unique_jobs.records,
        "bytes": unique_jobs.completed_shard.bytes,
        "sha256": unique_jobs.completed_shard.sha256,
        "url_key_digest": _unique_job_url_key_digest(unique_jobs),
    }
    if unique_manifest != expected_unique:
        raise ValueError("image fetch unique-job fingerprint mismatch")

    expected_outcomes = manifest.get("outcomes") or {}
    manifest_outcomes_path = Path(
        str(expected_outcomes.get("path") or "")
    )
    if manifest_outcomes_path.resolve() != result.outcomes_path.resolve():
        raise ValueError("image fetch outcome path mismatch")
    if not result.outcomes_path.is_file():
        raise ValueError("image fetch outcome store is missing")
    outcome_store = ImageOutcomeStore(result.outcomes_path)
    outcome_snapshot = outcome_store.snapshot_for_jobs(
        result.policy_fingerprint,
        unique_jobs,
    )
    if (
        int(outcome_snapshot["count"]) != result.outcomes_count
        or str(outcome_snapshot["digest"]) != result.outcome_digest
        or str(outcome_snapshot["url_key_digest"])
        != result.outcome_url_key_digest
        or int(outcome_snapshot["success"]) != result.success
        or int(outcome_snapshot["terminal"]) != result.terminal
        or {
            key: expected_outcomes.get(key)
            for key in (
                "count",
                "success",
                "terminal",
                "missing",
                "digest",
                "url_key_digest",
            )
        }
        != outcome_snapshot
    ):
        raise ValueError("image fetch outcome digest/count mismatch")
    if (
        outcome_snapshot["url_key_digest"]
        != expected_unique["url_key_digest"]
        or int(outcome_snapshot["missing"]) != 0
    ):
        raise ValueError("image fetch outcome job set mismatch")

    if not result.job_store_path.is_file():
        raise ValueError("image fetch durable job store is missing")
    job_snapshot = _job_snapshot(
        SqliteJobStore(result.job_store_path),
        result.job_kind,
    )
    expected_jobs = manifest.get("job_store") or {}
    if (
        result.job_kind != expected_jobs.get("kind")
        or {
            key: expected_jobs.get(key)
            for key in job_snapshot
        }
        != job_snapshot
        or job_snapshot["total"] != unique_jobs.records
        or job_snapshot["success"] != result.success
        or job_snapshot["terminal"] != result.terminal
        or any(
            job_snapshot[key]
            for key in ("pending", "retryable", "leased")
        )
    ):
        raise ValueError("image fetch durable job set is incomplete")
    return {
        "manifest_sha256": actual_manifest_sha256,
        "unique_jobs": expected_unique,
        "outcomes": outcome_snapshot,
        "job_store": job_snapshot,
    }


def validate_complete_image_fetch(
    result: ImageFetchResult,
    *,
    unique_jobs: UniqueImageJobs,
) -> dict[str, Any]:
    """Validate image fetch state against independently supplied jobs."""
    validated_jobs = _validated_unique_image_jobs(unique_jobs)
    if result.unique_jobs != validated_jobs:
        raise ValueError("image fetch unique-job result mismatch")
    return _validate_complete_image_fetch(result)


def validate_materialized_asset_shards(
    materialized: MaterializedAssetShards,
    *,
    planned: AssetPlanShards,
    image_fetch_result: ImageFetchResult,
    expected_input_fingerprint: str,
) -> tuple[MaterializedAssetShards, Any]:
    """Reconstruct Task-5 fingerprint, shards, counts, and strict barrier."""
    planned = _validated_planning_result(planned)
    fetch_snapshot = validate_complete_image_fetch(
        image_fetch_result,
        unique_jobs=image_fetch_result.unique_jobs,
    )
    expected_unique_input = stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        _sha256_file(planned.manifest_path),
        length=40,
    )
    if image_fetch_result.unique_jobs.input_fingerprint != expected_unique_input:
        raise ValueError("image fetch does not belong to asset planning")
    expected_stage_input = asset_materialization_input_fingerprint(
        planned.manifest_path,
        image_fetch_result.fetch_manifest_path,
    )
    if expected_input_fingerprint != expected_stage_input:
        raise ValueError("asset materialization input fingerprint mismatch")
    try:
        payload = json.loads(
            Path(materialized.manifest_path).read_text(encoding="utf-8")
        )
        fingerprint = payload["fingerprint"]
        budget = ImageBudget(
            attempts_per_entity=int(
                fingerprint["attempts_per_entity"]
            ),
            retained_per_entity=int(
                fingerprint["retained_per_entity"]
            ),
        )
        expected = {
            "input_fingerprint": expected_input_fingerprint,
            "schema_version": ASSET_MATERIALIZATION_SCHEMA_VERSION,
            "planning_manifest_sha256": _sha256_file(
                planned.manifest_path
            ),
            "unique_job_manifest_sha256": (
                fetch_snapshot["unique_jobs"]["manifest_sha256"]
            ),
            "unique_job_sha256": (
                fetch_snapshot["unique_jobs"]["sha256"]
            ),
            "image_fetch_manifest_sha256": (
                fetch_snapshot["manifest_sha256"]
            ),
            "image_policy_fingerprint": (
                image_fetch_result.policy_fingerprint
            ),
            "image_outcome_digest": (
                fetch_snapshot["outcomes"]["digest"]
            ),
            "image_outcome_count": (
                fetch_snapshot["outcomes"]["count"]
            ),
            "image_outcome_url_key_digest": (
                fetch_snapshot["outcomes"]["url_key_digest"]
            ),
            "attempts_per_entity": budget.attempts_per_entity,
            "retained_per_entity": budget.retained_per_entity,
            "text_asset_chunk_chars": int(
                fingerprint["text_asset_chunk_chars"]
            ),
            "min_text_asset_chunk_chars": int(
                fingerprint["min_text_asset_chunk_chars"]
            ),
            "max_text_asset_chunks_per_entity": int(
                fingerprint["max_text_asset_chunks_per_entity"]
            ),
            "records_per_shard": int(
                fingerprint["records_per_shard"]
            ),
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            "asset materialization fingerprint is invalid"
        ) from error
    resumed = _load_materialized_assets(
        output_root=materialized.output_root,
        manifest_path=materialized.manifest_path,
        expected=expected,
    )
    if resumed is None:
        raise ValueError("asset materialization is incomplete")
    if resumed != materialized:
        raise ValueError(
            "asset materialization result does not match manifest"
        )
    from wdc200k_models import AssetStageBarrier

    barrier = AssetStageBarrier(
        fingerprint=expected,
        bridge_assets=resumed.bridge_assets,
        table_asset_links=resumed.table_asset_links,
    )
    return resumed, barrier


def materialize_asset_shards(
    planned: AssetPlanShards,
    *,
    fetch_result: ImageFetchResult,
    output_root: Path,
    input_fingerprint: str,
    budget: ImageBudget = ImageBudget(),
    text_asset_chunk_chars: int = 800,
    min_text_asset_chunk_chars: int = 120,
    max_text_asset_chunks_per_entity: int = 3,
    records_per_shard: int = 10_000,
) -> MaterializedAssetShards:
    """Stream canonical bridge assets and source-table links into shards."""
    if not input_fingerprint:
        raise ValueError("materialization fingerprints must not be empty")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    planned = _validated_planning_result(planned)
    planning_manifest_sha256 = _sha256_file(planned.manifest_path)
    expected_unique_input = stable_hash(
        ASSET_PLANNING_SCHEMA_VERSION,
        planning_manifest_sha256,
        length=40,
    )
    if fetch_result.unique_jobs.input_fingerprint != expected_unique_input:
        raise ValueError(
            "image fetch jobs do not belong to this asset planning manifest"
        )
    fetch_snapshot = _validate_complete_image_fetch(fetch_result)
    output_root = Path(output_root)
    manifest_path = output_root / "asset-materialization-manifest.json"
    expected = {
        "input_fingerprint": input_fingerprint,
        "schema_version": ASSET_MATERIALIZATION_SCHEMA_VERSION,
        "planning_manifest_sha256": planning_manifest_sha256,
        "unique_job_manifest_sha256": (
            fetch_snapshot["unique_jobs"]["manifest_sha256"]
        ),
        "unique_job_sha256": (
            fetch_snapshot["unique_jobs"]["sha256"]
        ),
        "image_fetch_manifest_sha256": (
            fetch_snapshot["manifest_sha256"]
        ),
        "image_policy_fingerprint": fetch_result.policy_fingerprint,
        "image_outcome_digest": (
            fetch_snapshot["outcomes"]["digest"]
        ),
        "image_outcome_count": (
            fetch_snapshot["outcomes"]["count"]
        ),
        "image_outcome_url_key_digest": (
            fetch_snapshot["outcomes"]["url_key_digest"]
        ),
        "attempts_per_entity": budget.attempts_per_entity,
        "retained_per_entity": budget.retained_per_entity,
        "text_asset_chunk_chars": int(text_asset_chunk_chars),
        "min_text_asset_chunk_chars": int(
            min_text_asset_chunk_chars
        ),
        "max_text_asset_chunks_per_entity": int(
            max_text_asset_chunks_per_entity
        ),
        "records_per_shard": records_per_shard,
    }
    resumed = _load_materialized_assets(
        output_root=output_root,
        manifest_path=manifest_path,
        expected=expected,
    )
    if resumed is not None:
        return resumed

    outcome_store = ImageOutcomeStore(fetch_result.outcomes_path)
    asset_writer = _BoundedShardWriter(
        output_root / "bridge_assets",
        records_per_shard,
    )
    link_writer = _BoundedShardWriter(
        output_root / "table_asset_links",
        records_per_shard,
    )
    try:
        for planned_record in _iter_jsonl(planned.entity_plan_paths):
            entity = planned_record["entity"]
            page_outcome = planned_record.get("page")
            entity_plan = plan_entity_assets(
                entity,
                page_outcome,
                budget,
            )
            image_outcomes = {
                reference.url_key: outcome
                for reference in entity_plan.image_refs
                if (
                    outcome := outcome_store.get(
                        fetch_result.policy_fingerprint,
                        reference.url_key,
                    )
                )
                is not None
            }
            materialized = materialize_entity_assets(
                entity,
                page_outcome,
                image_outcomes,
                budget,
                text_asset_chunk_chars=text_asset_chunk_chars,
                min_text_asset_chunk_chars=(
                    min_text_asset_chunk_chars
                ),
                max_text_asset_chunks_per_entity=(
                    max_text_asset_chunks_per_entity
                ),
            )
            for asset in materialized.bridge_assets:
                asset_writer.write(asset)
            for link in materialized.table_asset_links:
                link_writer.write(link)
        completed_assets = asset_writer.close()
        completed_links = link_writer.close()
    except BaseException:
        asset_writer.abort()
        link_writer.abort()
        raise

    asset_shards = [
        _relative_completed(
            completed,
            output_root / "bridge_assets" / completed.path,
            output_root,
        )
        for completed in completed_assets
    ]
    link_shards = [
        _relative_completed(
            completed,
            output_root / "table_asset_links" / completed.path,
            output_root,
        )
        for completed in completed_links
    ]
    _atomic_json(
        manifest_path,
        {
            "stage": "wdc200k_asset_materialization",
            "fingerprint": expected,
            "bridge_asset_shards": [
                _shard_payload(shard) for shard in asset_shards
            ],
            "table_asset_link_shards": [
                _shard_payload(shard) for shard in link_shards
            ],
            "complete": True,
        },
    )
    return MaterializedAssetShards(
        output_root=output_root,
        bridge_asset_paths=tuple(
            output_root / shard.path for shard in asset_shards
        ),
        table_asset_link_paths=tuple(
            output_root / shard.path for shard in link_shards
        ),
        manifest_path=manifest_path,
        bridge_assets=sum(shard.records for shard in asset_shards),
        table_asset_links=sum(shard.records for shard in link_shards),
    )

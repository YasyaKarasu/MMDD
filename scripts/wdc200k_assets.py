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
        SqliteJobStore = io_helpers.SqliteJobStore
        external_unique_jsonl = io_helpers.external_unique_jsonl
        validate_completed_shard = io_helpers.validate_completed_shard
        _normalize_http_url = importlib.import_module(
            "wdc200k_structural"
        )._normalize_http_url
    finally:
        sys.path.remove(scripts_directory)


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
class ImageFetchResult:
    unique: int
    success: int
    terminal: int
    complete: bool
    outcomes_path: Path
    policy_fingerprint: str
    maximum_inflight: int
    maximum_claimed: int


@dataclass(frozen=True)
class MaterializedAssetShards:
    output_root: Path
    bridge_asset_paths: tuple[Path, ...]
    table_asset_link_paths: tuple[Path, ...]
    manifest_path: Path
    bridge_assets: int
    table_asset_links: int


class ImageOutcomeStore:
    """Durable policy-scoped terminal outcomes for unique image URLs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL").fetchone()
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

    def put(
        self,
        policy_fingerprint: str,
        url_key: str,
        image_url: str,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
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
) -> CompletedShard:
    """Publish one image job per globally unique normalized URL."""
    resumed = _validated_planning_result(planned)
    return external_unique_jsonl(
        resumed.image_mapping_paths,
        Path(output_path),
        key_fn=lambda record: record["url_key"],
        chunk_records=chunk_records,
        merge_fan_in=merge_fan_in,
    )


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


def _image_kind(policy_fingerprint: str) -> str:
    return f"wdc200k-image:{policy_fingerprint}"


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
                    f"{policy_fingerprint}:{url_key}",
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


def fetch_unique_images(
    image_job_paths: Iterable[Path],
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
    after_cache_write: Callable[[dict[str, Any]], None] | None = None,
) -> ImageFetchResult:
    """Fetch globally unique image URLs with bounded fair durable jobs."""
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
    kind = _image_kind(fingerprint)
    outcomes_path = Path(
        outcomes_path
        or store.path.with_name(
            f"{store.path.stem}-image-outcomes.sqlite3"
        )
    )
    outcome_store = ImageOutcomeStore(outcomes_path)
    _enqueue_unique_images(
        image_job_paths,
        store=store,
        kind=kind,
        policy_fingerprint=fingerprint,
    )
    unique = _job_count(store, kind)
    owner = f"image-{os.getpid()}-{uuid.uuid4().hex}"
    buffer_limit = (
        policy.global_concurrency * 4
        if claim_buffer is None
        else int(claim_buffer)
    )
    if buffer_limit < policy.global_concurrency:
        raise ValueError("claim_buffer must be at least global_concurrency")
    effective_lease = (
        float(lease_seconds)
        if lease_seconds is not None
        else policy.deadline_seconds * (buffer_limit + 2) + 30.0
    )
    host_queues: dict[str, deque[Any]] = {}
    active_by_host: dict[str, int] = {}
    ready_hosts: deque[str] = deque()
    ready_set: set[str] = set()
    futures: dict[Future[dict[str, Any]], tuple[Any, str]] = {}
    claimed_buffered = 0
    maximum_claimed = 0
    maximum_inflight = 0

    def add_ready(host: str) -> None:
        if host_queues.get(host) and host not in ready_set:
            ready_set.add(host)
            ready_hosts.append(host)

    def finish_cached(job: Any, outcome: dict[str, Any]) -> None:
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
        nonlocal claimed_buffered, maximum_claimed
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
                _fetch_image_job,
                job.payload,
                transport=transport,
                image_dir=Path(image_dir),
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
                add_ready(host)
                outcome = future.result()
                persisted = outcome_store.put(
                    fingerprint,
                    str(job.payload["url_key"]),
                    str(job.payload["image_url"]),
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
            submit_ready(pool)

    success, terminal = outcome_store.counts(fingerprint)
    return ImageFetchResult(
        unique=unique,
        success=success,
        terminal=terminal,
        complete=success + terminal == unique,
        outcomes_path=outcomes_path,
        policy_fingerprint=fingerprint,
        maximum_inflight=maximum_inflight,
        maximum_claimed=maximum_claimed,
    )


def iter_image_outcomes(
    outcomes_path: Path,
    policy_fingerprint: str,
) -> Iterator[dict[str, Any]]:
    """Stream durable unique image outcomes without all-result loading."""
    yield from ImageOutcomeStore(Path(outcomes_path)).iter(
        policy_fingerprint
    )


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


def materialize_asset_shards(
    planned: AssetPlanShards,
    *,
    outcomes_path: Path,
    policy_fingerprint: str,
    output_root: Path,
    input_fingerprint: str,
    budget: ImageBudget = ImageBudget(),
    text_asset_chunk_chars: int = 800,
    min_text_asset_chunk_chars: int = 120,
    max_text_asset_chunks_per_entity: int = 3,
    records_per_shard: int = 10_000,
) -> MaterializedAssetShards:
    """Stream canonical bridge assets and source-table links into shards."""
    if not input_fingerprint or not policy_fingerprint:
        raise ValueError("materialization fingerprints must not be empty")
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    planned = _validated_planning_result(planned)
    output_root = Path(output_root)
    manifest_path = output_root / "asset-materialization-manifest.json"
    expected = {
        "input_fingerprint": input_fingerprint,
        "image_policy_fingerprint": policy_fingerprint,
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

    outcome_store = ImageOutcomeStore(Path(outcomes_path))
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
                        policy_fingerprint,
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

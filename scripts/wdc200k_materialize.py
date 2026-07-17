"""Streaming canonical materialization for the WDC 200K pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import asdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import build_mm_joinability_dataset as join_builder
    import wdc200k_structural as structural
    from build_mm_table_dataset import normalize_title
    from stage1_io import clean_text, stable_hash
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        validate_completed_shard,
    )
    from wdc200k_models import (
        AssetStageBarrier,
        ModelStageResult,
        StructuralStageBarrier,
        _strict_asset_manifest,
        validate_model_stage,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "build_mm_joinability_dataset",
        "build_mm_table_dataset",
        "stage1_io",
        "wdc200k_io",
        "wdc200k_models",
        "wdc200k_structural",
    }:
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        import build_mm_joinability_dataset as join_builder
        import wdc200k_structural as structural
        from build_mm_table_dataset import normalize_title
        from stage1_io import clean_text, stable_hash
        from wdc200k_io import (
            AtomicJsonlShard,
            CompletedShard,
            validate_completed_shard,
        )
        from wdc200k_models import (
            AssetStageBarrier,
            ModelStageResult,
            StructuralStageBarrier,
            _strict_asset_manifest,
            validate_model_stage,
        )
    finally:
        sys.path.remove(scripts_directory)


MATERIALIZATION_SCHEMA_VERSION = "wdc200k-materialization-v1"
_CORE_ARTIFACTS = (
    "source_tables",
    "query_tables",
    "data_lake_tables",
    "entities",
    "bridge_assets",
    "table_asset_links",
    "attribute_extractions",
    "evidence_recoveries",
)
_DIAGNOSTIC_FILES = (
    "media_download_failures.jsonl",
    "model_attribute_errors.jsonl",
    "web_fetch_failures.jsonl",
)


@dataclass(frozen=True)
class MaterializationShardInputs:
    """Already-validated upstream paths needed for one-table materialization."""

    source_table: dict[str, Any]
    entity_paths: tuple[Path, ...]
    asset_paths: tuple[Path, ...]
    link_paths: tuple[Path, ...]
    extraction_paths: tuple[Path, ...]
    error_paths: tuple[Path, ...]
    lookup_database: Path


@dataclass(frozen=True)
class MaterializationInputs:
    structural_output_root: Path
    structural_manifests: tuple[Path, ...]
    finalized_selection_manifest: Path
    structural_barrier: StructuralStageBarrier
    assets_manifest: Path
    assets_barrier: AssetStageBarrier
    model_result: ModelStageResult
    work_root: Path


@dataclass(frozen=True)
class MaterializedTable:
    source_table: dict[str, Any]
    entities: list[dict[str, Any]]
    bridge_assets: list[dict[str, Any]]
    table_asset_links: list[dict[str, Any]]
    query_tables: list[dict[str, Any]]
    data_lake_tables: list[dict[str, Any]]
    qrels: list[dict[str, Any]]
    decision: dict[str, Any]
    attribute_extractions: list[dict[str, Any]]
    evidence_recoveries: list[dict[str, Any]]


@dataclass(frozen=True)
class MaterializationResult:
    output_root: Path
    manifest_path: Path
    stats: dict[str, Any]
    input_identity: str
    parameter_fingerprint: str
    complete: bool = True


@dataclass(frozen=True)
class _ValidatedUpstream:
    source_paths: tuple[Path, ...]
    entity_paths: tuple[Path, ...]
    structural_failure_paths: tuple[Path, ...]
    asset_paths: tuple[Path, ...]
    link_paths: tuple[Path, ...]
    extraction_paths: tuple[Path, ...]
    model_error_paths: tuple[Path, ...]
    expected_tables: int
    expected_entities: int
    expected_assets: int
    expected_links: int
    expected_extractions: int
    identity: str
    provenance: dict[str, Any]


class _RecordSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class _TableExtractionCache:
    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        self.items = {
            str(record["cache_key"]): dict(record)
            for record in records
            if clean_text(record.get("cache_key"))
        }

    def get(self, key: str) -> dict[str, Any] | None:
        record = self.items.get(key)
        return dict(record) if record is not None else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.items[key] = dict(record)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _completed_from_payload(payload: dict[str, Any]) -> CompletedShard:
    return CompletedShard(
        path=str(payload["path"]),
        records=int(payload["records"]),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
    )


def _manifest_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(f"upstream manifest is unreadable: {path}") from error
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"upstream manifest is not complete: {path}")
    return payload


def _structural_inputs(
    inputs: MaterializationInputs,
) -> tuple[
    list[Path],
    list[Path],
    list[Path],
    int,
    int,
    list[tuple[Path, str]],
]:
    root = Path(inputs.structural_output_root)
    manifests = sorted(Path(path) for path in inputs.structural_manifests)
    barrier = inputs.structural_barrier
    if (
        not manifests
        or barrier.schema_version != structural.STRUCTURAL_SCHEMA_VERSION
    ):
        raise ValueError("Task-3 structural barrier schema/set mismatch")
    keys = {path.resolve().as_posix() for path in manifests}
    if (
        len(manifests) != barrier.manifest_count
        or len(keys) != len(manifests)
        or keys != set(barrier.manifest_sha256)
    ):
        raise ValueError("Task-3 structural barrier set/count mismatch")

    source_paths: list[Path] = []
    entity_paths: list[Path] = []
    failure_paths: list[Path] = []
    validated_paths: list[Path] = []
    manifest_hashes: list[tuple[Path, str]] = []
    tables = 0
    entities = 0
    for manifest_path in manifests:
        key = manifest_path.resolve().as_posix()
        digest = _sha256_path(manifest_path)
        if digest != barrier.manifest_sha256[key]:
            raise ValueError("Task-3 structural manifest checksum mismatch")
        payload = _manifest_payload(manifest_path)
        if (
            payload.get("stage") != "wdc200k_structural"
            or payload.get("schema_version") != barrier.schema_version
            or payload.get("input_fingerprint")
            != barrier.input_fingerprints[key]
            or payload.get("parameter_fingerprint")
            != barrier.parameter_fingerprints[key]
        ):
            raise ValueError("Task-3 structural fingerprint mismatch")
        validated_path, records, validated_digest = (
            structural._validated_shard_from_manifest(
                manifest_path,
                output_root=root,
            )
        )
        validated_paths.append(validated_path)
        tables += records
        manifest_hashes.append((manifest_path.resolve(), validated_digest))
        completed = [
            _completed_from_payload(item)
            for item in payload.get("completed_shards", [])
        ]
        for shard in completed:
            path = root / shard.path
            if shard.path.startswith("source_tables/"):
                source_paths.append(path)
            elif shard.path.startswith("entities/"):
                entity_paths.append(path)
                entities += shard.records
            elif shard.path.startswith("structural_failures/"):
                failure_paths.append(path)

    final_path = Path(inputs.finalized_selection_manifest)
    final_payload = _manifest_payload(final_path)
    if _sha256_path(final_path) != barrier.final_manifest_sha256:
        raise ValueError("Task-3 final manifest checksum mismatch")
    if (
        final_payload.get("stage") != "wdc200k_validated_selection"
        or final_payload.get("schema_version") != barrier.schema_version
        or len(final_payload.get("completed_shards") or []) != 1
    ):
        raise ValueError("Task-3 global barrier stage/schema mismatch")
    final_shard = _completed_from_payload(
        final_payload["completed_shards"][0]
    )
    if asdict(final_shard) != barrier.final_selection:
        raise ValueError("Task-3 global artifact identity mismatch")
    expected_input = stable_hash(
        structural.STRUCTURAL_SCHEMA_VERSION,
        *(
            f"{path.as_posix()}:{digest}"
            for path, digest in manifest_hashes
        ),
        length=40,
    )
    expected_parameters = stable_hash(
        "validated-selection-global-v1",
        tables,
        length=40,
    )
    if (
        final_payload.get("input_fingerprint") != expected_input
        or final_payload.get("parameter_fingerprint")
        != expected_parameters
        or final_shard.records != tables
        or not validate_completed_shard(final_shard, root)
    ):
        raise ValueError(
            "Task-3 global barrier fingerprint/validation failed"
        )
    sentinel = object()
    expected_records = _iter_jsonl(validated_paths)
    actual_records = _iter_jsonl([root / final_shard.path])
    for expected, actual in zip_longest(
        expected_records,
        actual_records,
        fillvalue=sentinel,
    ):
        if (
            expected is sentinel
            or actual is sentinel
            or _canonical_json(expected) != _canonical_json(actual)
        ):
            raise ValueError(
                "Task-3 global selection content binding failed"
            )
    return (
        source_paths,
        entity_paths,
        failure_paths,
        tables,
        entities,
        manifest_hashes,
    )


def _validate_upstream(
    inputs: MaterializationInputs,
) -> _ValidatedUpstream:
    (
        source_paths,
        entity_paths,
        failure_paths,
        tables,
        entities,
        structural_hashes,
    ) = _structural_inputs(inputs)
    _assets_payload, asset_paths, link_paths = _strict_asset_manifest(
        Path(inputs.assets_manifest),
        barrier=inputs.assets_barrier,
    )
    retained_link_count = sum(
        1
        for link in _iter_jsonl(link_paths)
        if any(
            clean_text(asset_id)
            for asset_id in (link.get("asset_ids") or [])
        )
    )
    if not validate_model_stage(inputs.model_result):
        raise ValueError("Task-6 model result manifest validation failed")
    model_payload = _manifest_payload(inputs.model_result.manifest_path)
    model_root = inputs.model_result.manifest_path.parent
    model_extraction_shards = [
        _completed_from_payload(item)
        for item in model_payload.get("extraction_shards", [])
    ]
    model_error_shards = [
        _completed_from_payload(item)
        for item in model_payload.get("error_shards", [])
    ]
    model_counts = model_payload.get("counts")
    if not isinstance(model_counts, dict):
        raise ValueError("Task-6 model result counts are missing")
    model_success = int(model_counts.get("success", -1))
    model_terminal = int(model_counts.get("terminal", -1))
    if (
        model_success < 0
        or model_terminal < 0
        or sum(shard.records for shard in model_extraction_shards)
        != model_success
        or sum(shard.records for shard in model_error_shards)
        != model_terminal
    ):
        raise ValueError("Task-6 model result counts are invalid")
    model_manifest_sha256 = _sha256_path(
        inputs.model_result.manifest_path
    )
    asset_manifest_sha256 = _sha256_path(inputs.assets_manifest)
    provenance = {
        "structural_schema_version": (
            inputs.structural_barrier.schema_version
        ),
        "structural_manifests": [
            {
                "path": path.as_posix(),
                "sha256": digest,
            }
            for path, digest in structural_hashes
        ],
        "finalized_selection_manifest_sha256": (
            inputs.structural_barrier.final_manifest_sha256
        ),
        "asset_manifest_sha256": asset_manifest_sha256,
        "asset_fingerprint": inputs.assets_barrier.fingerprint,
        "model_manifest_sha256": model_manifest_sha256,
        "model_jobsets": {
            "text": inputs.model_result.jobset.text_fingerprint,
            "image": inputs.model_result.jobset.image_fingerprint,
        },
        "model_input_fingerprint": (
            inputs.model_result.jobset.input_fingerprint
        ),
    }
    identity = stable_hash(
        MATERIALIZATION_SCHEMA_VERSION,
        _canonical_json(provenance),
        length=40,
    )
    return _ValidatedUpstream(
        source_paths=tuple(source_paths),
        entity_paths=tuple(entity_paths),
        structural_failure_paths=tuple(failure_paths),
        asset_paths=tuple(asset_paths),
        link_paths=tuple(link_paths),
        extraction_paths=tuple(
            model_root / shard.path for shard in model_extraction_shards
        ),
        model_error_paths=tuple(
            model_root / shard.path for shard in model_error_shards
        ),
        expected_tables=tables,
        expected_entities=entities,
        expected_assets=inputs.assets_barrier.bridge_assets,
        expected_links=retained_link_count,
        expected_extractions=model_success + model_terminal,
        identity=identity,
        provenance=provenance,
    )


def _iter_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL record at {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"non-object JSONL record at {path}:{line_number}"
                    )
                yield record


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL,
                source_row_id INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS entities_source
                ON entities(source_table_id, source_row_id, entity_id);

            CREATE TABLE IF NOT EXISTS entity_aliases (
                alias TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS assets_entity
                ON assets(entity_id, asset_id);

            CREATE TABLE IF NOT EXISTS links (
                link_id TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL,
                source_row_id INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS links_source
                ON links(source_table_id, source_row_id, link_id);

            CREATE TABLE IF NOT EXISTS extractions (
                cache_key TEXT PRIMARY KEY,
                model_call_key TEXT NOT NULL,
                job_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS extractions_entity_asset
                ON extractions(entity_id, asset_id, cache_key);
            CREATE INDEX IF NOT EXISTS extractions_source
                ON extractions(source_table_id, cache_key);
            CREATE UNIQUE INDEX IF NOT EXISTS extractions_full_call
                ON extractions(model_call_key)
                WHERE model_call_key <> '';

            CREATE TABLE IF NOT EXISTS evidence (
                recovery_id TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS errors (
                error_key TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL,
                record_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS source_catalog (
                source_table_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                page_title TEXT NOT NULL,
                split_group TEXT NOT NULL,
                split TEXT,
                record_sha256 TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS source_catalog_group
                ON source_catalog(split_group, source_table_id);

            CREATE TABLE IF NOT EXISTS split_groups (
                split_group TEXT PRIMARY KEY,
                rank_hash TEXT NOT NULL,
                first_ordinal INTEGER NOT NULL,
                split TEXT
            );

            CREATE TABLE IF NOT EXISTS source_units (
                source_table_id TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL,
                split TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                complete INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS materialized_records (
                artifact TEXT NOT NULL,
                record_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                source_ordinal INTEGER NOT NULL,
                record_ordinal INTEGER NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (artifact, record_id)
            );
            CREATE INDEX IF NOT EXISTS materialized_order
                ON materialized_records(
                    artifact, source_ordinal, record_ordinal, record_id
                );

            CREATE TABLE IF NOT EXISTS table_ids (
                table_id TEXT PRIMARY KEY,
                artifact TEXT NOT NULL,
                source_table_id TEXT NOT NULL
            );
            """
        )


def _insert_identity_record(
    connection: sqlite3.Connection,
    *,
    table: str,
    key_column: str,
    key: str,
    columns: dict[str, Any],
    kind: str,
) -> None:
    encoded = _canonical_json(columns.pop("record"))
    values = {**columns, "record_json": encoded}
    existing = connection.execute(
        f"SELECT record_json FROM {table} WHERE {key_column} = ?",
        (key,),
    ).fetchone()
    if existing is not None:
        if str(existing["record_json"]) != encoded:
            raise ValueError(f"conflicting {kind} identity: {key}")
        return
    column_names = [key_column, *values]
    placeholders = ", ".join("?" for _ in column_names)
    connection.execute(
        f"INSERT INTO {table} ({', '.join(column_names)}) "
        f"VALUES ({placeholders})",
        [key, *(values[name] for name in values)],
    )


def _entity_appearance(record: dict[str, Any]) -> tuple[str, int]:
    appearances = record.get("appears_in")
    if not isinstance(appearances, list) or not appearances:
        raise ValueError("entity is missing source-table appearance")
    appearance = appearances[0]
    if not isinstance(appearance, dict):
        raise ValueError("entity appearance is not an object")
    source_table_id = clean_text(appearance.get("source_table_id"))
    if not source_table_id:
        raise ValueError("entity appearance is missing source_table_id")
    return source_table_id, int(appearance.get("row_id", 0))


def _insert_alias(
    connection: sqlite3.Connection,
    alias: str,
    entity_id: str,
) -> None:
    alias = clean_text(alias)
    if not alias:
        return
    existing = connection.execute(
        "SELECT entity_id FROM entity_aliases WHERE alias = ?",
        (alias,),
    ).fetchone()
    if existing is not None:
        if str(existing["entity_id"]) != entity_id:
            raise ValueError(f"conflicting entity alias: {alias}")
        return
    connection.execute(
        "INSERT INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
        (alias, entity_id),
    )


def _index_entities(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    commit_every: int = 0,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        entity_id = clean_text(record.get("entity_id"))
        if not entity_id:
            raise ValueError("entity is missing entity_id")
        source_table_id, source_row_id = _entity_appearance(record)
        _insert_identity_record(
            connection,
            table="entities",
            key_column="entity_id",
            key=entity_id,
            columns={
                "source_table_id": source_table_id,
                "source_row_id": source_row_id,
                "record": record,
            },
            kind="entity",
        )
        wiki_title = clean_text(record.get("wiki_title"))
        aliases = {
            wiki_title,
            normalize_title(wiki_title),
            wiki_title.casefold(),
            normalize_title(wiki_title).casefold(),
        }
        for alias in aliases:
            _insert_alias(connection, alias, entity_id)
        _commit_index_batch(connection, count, commit_every)


def _index_assets(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    commit_every: int = 0,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        asset_id = clean_text(record.get("asset_id"))
        entity_id = clean_text(record.get("entity_id"))
        if not asset_id or not entity_id:
            raise ValueError("asset is missing asset_id/entity_id")
        _insert_identity_record(
            connection,
            table="assets",
            key_column="asset_id",
            key=asset_id,
            columns={"entity_id": entity_id, "record": record},
            kind="asset",
        )
        _commit_index_batch(connection, count, commit_every)


def _index_links(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    commit_every: int = 0,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        source_table_id = clean_text(record.get("source_table_id"))
        entity_id = clean_text(record.get("entity_id"))
        link_id = clean_text(record.get("link_id"))
        if not link_id:
            link_id = _canonical_json(
                {
                    "source_table_id": source_table_id,
                    "row_id": record.get("row_id"),
                    "entity_id": entity_id,
                }
            )
        if not source_table_id or not entity_id:
            raise ValueError("asset link is missing source/entity identity")
        _insert_identity_record(
            connection,
            table="links",
            key_column="link_id",
            key=link_id,
            columns={
                "source_table_id": source_table_id,
                "source_row_id": int(record.get("row_id", 0)),
                "entity_id": entity_id,
                "record": record,
            },
            kind="asset link",
        )
        for alias in (
            clean_text(record.get("entity_wiki_title")),
            normalize_title(clean_text(record.get("entity_wiki_title"))),
        ):
            _insert_alias(connection, alias, entity_id)
            _insert_alias(connection, alias.casefold(), entity_id)
        _commit_index_batch(connection, count, commit_every)


def _extraction_identity(record: dict[str, Any]) -> tuple[str, str, str]:
    cache_key = clean_text(record.get("cache_key"))
    entity_id = clean_text(record.get("entity_id"))
    asset_id = clean_text(record.get("asset_id"))
    if not cache_key or not entity_id or not asset_id:
        raise ValueError(
            "model extraction is missing cache/entity/asset identity"
        )
    return cache_key, entity_id, asset_id


def _index_extractions(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    status: str,
    commit_every: int = 0,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        cache_key, entity_id, asset_id = _extraction_identity(record)
        entity_row = connection.execute(
            """
            SELECT source_table_id FROM entities WHERE entity_id = ?
            """,
            (entity_id,),
        ).fetchone()
        source_table_id = (
            str(entity_row["source_table_id"])
            if entity_row is not None
            else clean_text(record.get("source_table_id"))
        )
        if not source_table_id:
            raise ValueError(
                "model extraction entity has no source-table binding"
            )
        _insert_identity_record(
            connection,
            table="extractions",
            key_column="cache_key",
            key=cache_key,
            columns={
                "model_call_key": clean_text(
                    record.get("model_call_key")
                ),
                "job_id": clean_text(record.get("job_id")),
                "entity_id": entity_id,
                "asset_id": asset_id,
                "source_table_id": source_table_id,
                "status": status,
                "record": record,
            },
            kind="model extraction",
        )
        if status == "terminal":
            error_key = (
                clean_text(record.get("job_id"))
                or clean_text(record.get("model_call_key"))
                or cache_key
            )
            _insert_identity_record(
                connection,
                table="errors",
                key_column="error_key",
                key=error_key,
                columns={
                    "source_table_id": source_table_id,
                    "record": record,
                },
                kind="model error",
            )
        _commit_index_batch(connection, count, commit_every)


def _commit_index_batch(
    connection: sqlite3.Connection,
    count: int,
    commit_every: int,
) -> None:
    if commit_every > 0 and count % commit_every == 0:
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")


def _build_index(inputs: MaterializationShardInputs) -> None:
    _initialize_index(inputs.lookup_database)
    with _connect(inputs.lookup_database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _index_entities(connection, inputs.entity_paths)
        _index_assets(connection, inputs.asset_paths)
        _index_links(connection, inputs.link_paths)
        _index_extractions(
            connection,
            inputs.extraction_paths,
            status="success",
        )
        _index_extractions(
            connection,
            inputs.error_paths,
            status="terminal",
        )
        connection.commit()


def _prepare_authoritative_index(
    database_path: Path,
    upstream: _ValidatedUpstream,
) -> None:
    _initialize_index(database_path)
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        stored = connection.execute(
            "SELECT value FROM metadata WHERE key = 'upstream_identity'"
        ).fetchone()
        if stored is not None and str(stored["value"]) != upstream.identity:
            raise ValueError(
                "materialization index upstream identity mismatch"
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO metadata (key, value)
            VALUES ('upstream_identity', ?)
            """,
            (upstream.identity,),
        )
        _index_entities(
            connection, upstream.entity_paths, commit_every=1_000
        )
        _index_assets(
            connection, upstream.asset_paths, commit_every=1_000
        )
        _index_links(
            connection, upstream.link_paths, commit_every=1_000
        )
        _index_extractions(
            connection,
            upstream.extraction_paths,
            status="success",
            commit_every=1_000,
        )
        _index_extractions(
            connection,
            upstream.model_error_paths,
            status="terminal",
            commit_every=1_000,
        )
        connection.commit()


def _parameter_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "split_by": str(args.split_by),
        "train_ratio": float(args.train_ratio),
        "dev_ratio": float(args.dev_ratio),
        "test_ratio": float(args.test_ratio),
        "query_rows_per_table": int(
            join_builder.configured_query_rows_per_table(args)
        ),
        "min_rows_per_output_table": int(
            args.min_rows_per_output_table
        ),
        "min_column_non_empty_ratio": float(
            args.min_column_non_empty_ratio
        ),
        "min_recovered_value_ratio": float(
            args.min_recovered_value_ratio
        ),
        "min_recovery_denominator": int(
            args.min_recovery_denominator
        ),
        "max_query_tables_per_source_table": int(
            args.max_query_tables_per_source_table
        ),
        "max_query_context_attrs": int(args.max_query_context_attrs),
        "max_target_context_attrs": int(args.max_target_context_attrs),
        "reparse_cached_model_outputs": bool(
            args.reparse_cached_model_outputs
        ),
        "refresh_invalid_model_cache": bool(
            args.refresh_invalid_model_cache
        ),
    }


def _parameter_fingerprint(args: argparse.Namespace) -> str:
    return stable_hash(
        MATERIALIZATION_SCHEMA_VERSION,
        _canonical_json(_parameter_payload(args)),
        length=40,
    )


def _split_group(
    source_table: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    source_table_id = str(source_table["source_table_id"])
    if args.split_by == "page_title":
        return clean_text(source_table.get("page_title")) or source_table_id
    return source_table_id


def _catalog_sources(
    database_path: Path,
    source_paths: Iterable[Path],
    *,
    args: argparse.Namespace,
    expected_tables: int,
) -> None:
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        observed_stream = 0
        for ordinal, source_table in enumerate(_iter_jsonl(source_paths)):
            observed_stream += 1
            source_table_id = clean_text(
                source_table.get("source_table_id")
            )
            if not source_table_id:
                raise ValueError("source table is missing source_table_id")
            encoded = _canonical_json(source_table)
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            existing = connection.execute(
                """
                SELECT ordinal, page_title, split_group,
                       record_sha256, record_json
                FROM source_catalog WHERE source_table_id = ?
                """,
                (source_table_id,),
            ).fetchone()
            expected_values = (
                ordinal,
                clean_text(source_table.get("page_title")),
                _split_group(source_table, args),
                digest,
                encoded,
            )
            if existing is not None:
                actual_values = (
                    int(existing["ordinal"]),
                    str(existing["page_title"]),
                    str(existing["split_group"]),
                    str(existing["record_sha256"]),
                    str(existing["record_json"]),
                )
                if actual_values != expected_values:
                    raise ValueError(
                        f"conflicting source table ID: {source_table_id}"
                    )
            else:
                try:
                    connection.execute(
                        """
                        INSERT INTO source_catalog (
                            source_table_id, ordinal, page_title,
                            split_group, record_sha256, record_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (source_table_id, *expected_values),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"duplicate source table ID: {source_table_id}"
                    ) from error
            if (ordinal + 1) % 10 == 0:
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
        observed = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_catalog"
            ).fetchone()[0]
        )
        if (
            observed_stream != expected_tables
            or observed != expected_tables
        ):
            raise ValueError(
                f"source table count {observed_stream}/{observed} does "
                f"not match expected {expected_tables}"
            )
        connection.commit()


def _assign_splits(
    database_path: Path,
    args: argparse.Namespace,
) -> None:
    ratios = [
        float(args.train_ratio),
        float(args.dev_ratio),
        float(args.test_ratio),
    ]
    total = sum(ratios)
    if total <= 0:
        raise ValueError(
            "train/dev/test ratios must sum to a positive value"
        )
    ratios = [value / total for value in ratios]
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        assigned = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_catalog WHERE split IS NOT NULL"
            ).fetchone()[0]
        )
        source_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_catalog"
            ).fetchone()[0]
        )
        if assigned:
            if assigned != source_count:
                raise ValueError("partial split assignment in work index")
            connection.commit()
            return
        for row in connection.execute(
            """
            SELECT split_group, MIN(ordinal) AS first_ordinal
            FROM source_catalog
            GROUP BY split_group
            """
        ):
            group = str(row["split_group"])
            connection.execute(
                """
                INSERT INTO split_groups (
                    split_group, rank_hash, first_ordinal
                ) VALUES (?, ?, ?)
                """,
                (
                    group,
                    stable_hash("split", args.seed, group),
                    int(row["first_ordinal"]),
                ),
            )
        group_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM split_groups"
            ).fetchone()[0]
        )
        train_n = int(group_count * ratios[0])
        dev_n = int(group_count * ratios[1])
        if group_count >= 3:
            train_n = max(1, train_n) if ratios[0] > 0 else 0
            dev_n = max(1, dev_n) if ratios[1] > 0 else 0
            if train_n + dev_n >= group_count:
                dev_n = max(0, group_count - train_n - 1)
        for rank, row in enumerate(
            connection.execute(
                """
                SELECT split_group
                FROM split_groups
                ORDER BY rank_hash, first_ordinal
                """
            )
        ):
            split = (
                "train"
                if rank < train_n
                else "dev"
                if rank < train_n + dev_n
                else "test"
            )
            group = str(row["split_group"])
            connection.execute(
                "UPDATE split_groups SET split = ? WHERE split_group = ?",
                (split, group),
            )
            connection.execute(
                "UPDATE source_catalog SET split = ? WHERE split_group = ?",
                (split, group),
            )
        connection.commit()


def _decode_records(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...],
) -> list[dict[str, Any]]:
    return [
        json.loads(str(row["record_json"]))
        for row in connection.execute(query, parameters)
    ]


def _source_wiki_titles(source_table: dict[str, Any]) -> list[str]:
    entity_column = join_builder.choose_entity_column(source_table)
    if entity_column is None:
        return []
    titles = []
    for row in source_table.get("rows", []):
        title = clean_text(
            join_builder.get_cell(row, entity_column).get("wiki_title")
        )
        if title:
            titles.append(title)
    return titles


def _table_inputs(
    database_path: Path,
    source_table: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    source_table_id = clean_text(source_table.get("source_table_id"))
    with _connect(database_path) as connection:
        entities = _decode_records(
            connection,
            """
            SELECT record_json FROM entities
            WHERE source_table_id = ?
            ORDER BY source_row_id, entity_id
            """,
            (source_table_id,),
        )
        links = _decode_records(
            connection,
            """
            SELECT record_json FROM links
            WHERE source_table_id = ?
            ORDER BY source_row_id, link_id
            """,
            (source_table_id,),
        )
        asset_ids = [
            clean_text(asset_id)
            for link in links
            for asset_id in (link.get("asset_ids") or [])
            if clean_text(asset_id)
        ]
        assets = []
        extractions = []
        seen_assets: set[str] = set()
        seen_extractions: set[str] = set()
        for asset_id in asset_ids:
            if asset_id in seen_assets:
                continue
            seen_assets.add(asset_id)
            asset_row = connection.execute(
                "SELECT record_json FROM assets WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            if asset_row is None:
                continue
            assets.append(json.loads(str(asset_row["record_json"])))
            for extraction_row in connection.execute(
                """
                SELECT cache_key, record_json
                FROM extractions
                WHERE asset_id = ?
                ORDER BY cache_key
                """,
                (asset_id,),
            ):
                cache_key = str(extraction_row["cache_key"])
                if cache_key in seen_extractions:
                    continue
                seen_extractions.add(cache_key)
                extractions.append(
                    json.loads(str(extraction_row["record_json"]))
                )

        wiki_to_entity_id = {
            clean_text(entity.get("wiki_title")): str(entity["entity_id"])
            for entity in entities
            if clean_text(entity.get("wiki_title"))
        }
        for title in _source_wiki_titles(source_table):
            for alias in (
                title,
                normalize_title(title),
                title.casefold(),
                normalize_title(title).casefold(),
            ):
                row = connection.execute(
                    "SELECT entity_id FROM entity_aliases WHERE alias = ?",
                    (alias,),
                ).fetchone()
                if row is not None:
                    wiki_to_entity_id[title] = str(row["entity_id"])
                    break
    return entities, assets, links, extractions, wiki_to_entity_id


def _materialize_from_index(
    source_table: dict[str, Any],
    database_path: Path,
    *,
    args: argparse.Namespace,
    split: str,
) -> MaterializedTable:
    (
        entities,
        assets,
        links,
        extractions,
        wiki_to_entity_id,
    ) = _table_inputs(database_path, source_table)
    asset_by_id = {
        str(asset["asset_id"]): asset for asset in assets
    }
    entity_to_assets: dict[str, list[str]] = {}
    for link in links:
        entity_id = clean_text(link.get("entity_id"))
        if not entity_id:
            continue
        current = entity_to_assets.setdefault(entity_id, [])
        for asset_id in link.get("asset_ids") or []:
            asset_id = clean_text(asset_id)
            if asset_id and asset_id not in current:
                current.append(asset_id)

    extraction_sink = _RecordSink()
    recovery_sink = _RecordSink()
    materialize_args = copy.copy(args)
    materialize_args.cache_failed_model_outputs = True
    materialize_args.model_attribute_errors_path = ""
    (
        query_tables,
        data_lake_tables,
        qrels,
        decision,
    ) = join_builder.build_table_join_records(
        source_table=source_table,
        split=split,
        assets=asset_by_id,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=None,
        cache=_TableExtractionCache(extractions),
        progress=None,
        concurrency_state=join_builder.ModelConcurrencyState.from_args(
            materialize_args
        ),
        extraction_writer=extraction_sink,
        recovery_writer=recovery_sink,
        args=materialize_args,
    )
    return MaterializedTable(
        source_table=source_table,
        entities=entities,
        bridge_assets=assets,
        table_asset_links=links,
        query_tables=query_tables,
        data_lake_tables=data_lake_tables,
        qrels=qrels,
        decision=decision,
        attribute_extractions=extraction_sink.records,
        evidence_recoveries=recovery_sink.records,
    )


def materialize_dataset_shard(
    inputs: MaterializationShardInputs,
    *,
    args: argparse.Namespace,
    split: str,
) -> MaterializedTable:
    """Materialize one source table using only its SQLite-selected records."""
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid split: {split}")
    _build_index(inputs)
    return _materialize_from_index(
        inputs.source_table,
        inputs.lookup_database,
        args=args,
        split=split,
    )


def _record_id(
    artifact: str,
    record: dict[str, Any],
    source_table_id: str,
    ordinal: int,
) -> str:
    fields = {
        "source_tables": "source_table_id",
        "query_tables": "table_id",
        "data_lake_tables": "table_id",
        "entities": "entity_id",
        "bridge_assets": "asset_id",
        "table_asset_links": "link_id",
        "evidence_recoveries": "recovery_id",
    }
    field = fields.get(artifact)
    if field and clean_text(record.get(field)):
        return clean_text(record[field])
    if artifact == "attribute_extractions":
        return stable_hash(
            source_table_id,
            record.get("source_row_id"),
            record.get("cache_key"),
            record.get("asset_id"),
            length=40,
        )
    if artifact == "qrels":
        return stable_hash(
            record.get("query_table_id"),
            record.get("target_table_id"),
            length=40,
        )
    if artifact == "table_queryability_decisions":
        return source_table_id
    return stable_hash(
        artifact,
        source_table_id,
        ordinal,
        _canonical_json(record),
        length=40,
    )


def _insert_materialized_records(
    connection: sqlite3.Connection,
    *,
    artifact: str,
    records: Iterable[dict[str, Any]],
    source_table_id: str,
    source_ordinal: int,
) -> int:
    count = 0
    for ordinal, record in enumerate(records):
        record_id = _record_id(
            artifact,
            record,
            source_table_id,
            ordinal,
        )
        try:
            connection.execute(
                """
                INSERT INTO materialized_records (
                    artifact, record_id, source_table_id,
                    source_ordinal, record_ordinal, record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact,
                    record_id,
                    source_table_id,
                    source_ordinal,
                    ordinal,
                    _canonical_json(record),
                ),
            )
            if artifact in {
                "source_tables",
                "query_tables",
                "data_lake_tables",
            }:
                connection.execute(
                    """
                    INSERT INTO table_ids (
                        table_id, artifact, source_table_id
                    ) VALUES (?, ?, ?)
                    """,
                    (record_id, artifact, source_table_id),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"duplicate {artifact} ID: {record_id}"
            ) from error
        count += 1
    return count


def _store_table_unit(
    database_path: Path,
    materialized: MaterializedTable,
    *,
    source_ordinal: int,
    source_sha256: str,
    split: str,
) -> bool:
    source_table_id = str(
        materialized.source_table["source_table_id"]
    )
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            """
            SELECT source_sha256, split, complete
            FROM source_units WHERE source_table_id = ?
            """,
            (source_table_id,),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["source_sha256"]) != source_sha256
                or str(existing["split"]) != split
                or int(existing["complete"]) != 1
            ):
                raise ValueError(
                    f"source unit identity mismatch: {source_table_id}"
                )
            connection.commit()
            return False
        decision = {
            **materialized.decision,
            "source_table_id": source_table_id,
            "split": split,
        }
        artifact_records: dict[str, Iterable[dict[str, Any]]] = {
            "source_tables": [materialized.source_table],
            "query_tables": materialized.query_tables,
            "data_lake_tables": materialized.data_lake_tables,
            "entities": materialized.entities,
            "bridge_assets": materialized.bridge_assets,
            "table_asset_links": materialized.table_asset_links,
            "attribute_extractions": (
                materialized.attribute_extractions
            ),
            "evidence_recoveries": materialized.evidence_recoveries,
            "qrels": materialized.qrels,
            "table_queryability_decisions": [decision],
        }
        artifact_records["table_asset_links"] = (
            link
            for link in materialized.table_asset_links
            if any(
                clean_text(asset_id)
                for asset_id in (link.get("asset_ids") or [])
            )
        )
        counts = {
            artifact: _insert_materialized_records(
                connection,
                artifact=artifact,
                records=records,
                source_table_id=source_table_id,
                source_ordinal=source_ordinal,
            )
            for artifact, records in artifact_records.items()
        }
        for recovery in materialized.evidence_recoveries:
            recovery_id = clean_text(recovery.get("recovery_id"))
            if not recovery_id:
                raise ValueError("evidence recovery is missing recovery_id")
            _insert_identity_record(
                connection,
                table="evidence",
                key_column="recovery_id",
                key=recovery_id,
                columns={
                    "source_table_id": source_table_id,
                    "record": recovery,
                },
                kind="evidence recovery",
            )
        connection.execute(
            """
            INSERT INTO source_units (
                source_table_id, source_sha256, split,
                counts_json, complete
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (
                source_table_id,
                source_sha256,
                split,
                _canonical_json(counts),
            ),
        )
        connection.commit()
    return True


def _materialize_all_tables(
    database_path: Path,
    *,
    args: argparse.Namespace,
    expected_tables: int,
    after_table_commit: Callable[[str], None] | None = None,
) -> None:
    with _connect(database_path) as connection:
        cursor = connection.execute(
            """
            SELECT source_table_id, ordinal, split,
                   record_sha256, record_json
            FROM source_catalog
            ORDER BY ordinal
            """
        )
        observed = 0
        for row in cursor:
            observed += 1
            source_table_id = str(row["source_table_id"])
            split = clean_text(row["split"])
            if split not in {"train", "dev", "test"}:
                raise ValueError(
                    f"source table has invalid split: {source_table_id}"
                )
            completed = connection.execute(
                """
                SELECT source_sha256, split, complete
                FROM source_units WHERE source_table_id = ?
                """,
                (source_table_id,),
            ).fetchone()
            if completed is not None:
                if (
                    str(completed["source_sha256"])
                    != str(row["record_sha256"])
                    or str(completed["split"]) != split
                    or int(completed["complete"]) != 1
                ):
                    raise ValueError(
                        f"source unit resume mismatch: {source_table_id}"
                    )
                continue
            source_table = json.loads(str(row["record_json"]))
            if any(
                clean_text(column.get("column_name")).casefold()
                == "image"
                for column in source_table.get("columns", [])
            ):
                raise ValueError(
                    f"source table retains forbidden image column: "
                    f"{source_table_id}"
                )
            materialized = _materialize_from_index(
                source_table,
                database_path,
                args=args,
                split=split,
            )
            _store_table_unit(
                database_path,
                materialized,
                source_ordinal=int(row["ordinal"]),
                source_sha256=str(row["record_sha256"]),
                split=split,
            )
            if after_table_commit is not None:
                after_table_commit(source_table_id)
    if observed != expected_tables:
        raise ValueError("materialization source iteration count mismatch")
    with _connect(database_path) as connection:
        completed_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_units WHERE complete = 1
                """
            ).fetchone()[0]
        )
    if completed_count != expected_tables:
        raise ValueError("materialization table barrier is incomplete")


class _AtomicArtifactWriter:
    def __init__(
        self,
        output_root: Path,
        artifact: str,
        records_per_shard: int,
    ) -> None:
        self.output_root = output_root
        self.artifact = artifact
        self.records_per_shard = records_per_shard
        self.directory = output_root / artifact
        self.directory.mkdir(parents=True, exist_ok=True)
        self.completed: list[CompletedShard] = []
        self._writer: AtomicJsonlShard | None = None
        self._current_count = 0

    def _open(self) -> None:
        path = (
            self.directory
            / f"part-{len(self.completed):05d}.jsonl"
        )
        self._writer = AtomicJsonlShard(path)
        self._current_count = 0

    def _commit(self) -> None:
        if self._writer is None:
            return
        completed = self._writer.commit()
        path = self.directory / completed.path
        self.completed.append(
            CompletedShard(
                path=path.relative_to(self.output_root).as_posix(),
                records=completed.records,
                bytes=completed.bytes,
                sha256=completed.sha256,
            )
        )
        self._writer = None
        self._current_count = 0

    def write(self, record: dict[str, Any]) -> None:
        if self._writer is None:
            self._open()
        if self._current_count >= self.records_per_shard:
            self._commit()
            self._open()
        assert self._writer is not None
        self._writer.write(record)
        self._current_count += 1

    def close(self) -> tuple[CompletedShard, ...]:
        self._commit()
        return tuple(self.completed)

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.abort()
            self._writer = None


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> CompletedShard:
    writer = AtomicJsonlShard(path)
    try:
        writer.write(payload)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _atomic_jsonl_from_records(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> CompletedShard:
    writer = AtomicJsonlShard(path)
    try:
        for record in records:
            writer.write(record)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _iter_materialized(
    database_path: Path,
    artifact: str,
) -> Iterator[dict[str, Any]]:
    with _connect(database_path) as connection:
        for row in connection.execute(
            """
            SELECT record_json
            FROM materialized_records
            WHERE artifact = ?
            ORDER BY source_ordinal, record_ordinal, record_id
            """,
            (artifact,),
        ):
            yield json.loads(str(row["record_json"]))


def _artifact_count(database_path: Path, artifact: str) -> int:
    with _connect(database_path) as connection:
        return int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = ?
                """,
                (artifact,),
            ).fetchone()[0]
        )


def _validate_global_counts(
    database_path: Path,
    upstream: _ValidatedUpstream,
) -> dict[str, int]:
    counts = {
        artifact: _artifact_count(database_path, artifact)
        for artifact in (
            *_CORE_ARTIFACTS,
            "qrels",
            "table_queryability_decisions",
        )
    }
    expected = {
        "source_tables": upstream.expected_tables,
        "entities": upstream.expected_entities,
        "bridge_assets": upstream.expected_assets,
        "table_asset_links": upstream.expected_links,
        "attribute_extractions": upstream.expected_extractions,
        "table_queryability_decisions": upstream.expected_tables,
    }
    for artifact, expected_count in expected.items():
        if counts[artifact] != expected_count:
            raise ValueError(
                f"global {artifact} count {counts[artifact]} does not "
                f"match expected {expected_count}"
            )
    if counts["data_lake_tables"] < upstream.expected_tables:
        raise ValueError("global data-lake table coverage is incomplete")
    if counts["query_tables"] != counts["qrels"]:
        raise ValueError("global query/qrel count mismatch")
    with _connect(database_path) as connection:
        duplicate_qrels = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT json_extract(record_json, '$.query_table_id')
                           AS query_id, COUNT(*) AS count
                    FROM materialized_records
                    WHERE artifact = 'qrels'
                    GROUP BY query_id HAVING count > 1
                )
                """
            ).fetchone()[0]
        )
    if duplicate_qrels:
        raise ValueError("duplicate qrel/query ID")
    return counts


def _write_splits(
    database_path: Path,
    path: Path,
    args: argparse.Namespace,
) -> CompletedShard:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("{")
            first_split = True
            with _connect(database_path) as connection:
                for split in ("train", "dev", "test"):
                    if not first_split:
                        handle.write(",")
                    first_split = False
                    handle.write(json.dumps(split))
                    handle.write(":{")
                    for index, (name, query, parameters) in enumerate(
                        (
                            (
                                "source_table_ids",
                                """
                                SELECT source_table_id AS value
                                FROM source_catalog WHERE split = ?
                                ORDER BY source_table_id
                                """,
                                (split,),
                            ),
                            (
                                "query_table_ids",
                                """
                                SELECT record_id AS value
                                FROM materialized_records
                                WHERE artifact = 'query_tables'
                                  AND json_extract(
                                      record_json, '$.split'
                                  ) = ?
                                ORDER BY record_id
                                """,
                                (split,),
                            ),
                            (
                                "data_lake_table_ids",
                                """
                                SELECT record_id AS value
                                FROM materialized_records
                                WHERE artifact = 'data_lake_tables'
                                  AND json_extract(
                                      record_json, '$.split'
                                  ) = ?
                                ORDER BY record_id
                                """,
                                (split,),
                            ),
                        )
                    ):
                        if index:
                            handle.write(",")
                        handle.write(json.dumps(name))
                        handle.write(":[")
                        first = True
                        for row in connection.execute(query, parameters):
                            if not first:
                                handle.write(",")
                            first = False
                            handle.write(
                                json.dumps(str(row["value"]))
                            )
                        handle.write("]")
                    handle.write("}")
            handle.write(',"split_key":')
            handle.write(
                json.dumps(
                    "page_title_or_source_table_id"
                    if args.split_by == "page_title"
                    else "source_table_id"
                )
            )
            handle.write(',"note":')
            handle.write(
                json.dumps(
                    "source-level split; data_lake contains generated "
                    "targets for queryable tables and raw tables for "
                    "rejected tables"
                )
            )
            handle.write("}\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256_path(temporary)
        size = temporary.stat().st_size
        temporary.replace(path)
        _fsync_directory(path.parent)
        return CompletedShard(
            path=path.name,
            records=1,
            bytes=size,
            sha256=digest,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failure_records(
    paths: Iterable[Path],
) -> Iterator[dict[str, Any]]:
    yield from _iter_jsonl(paths)


def _stats_payload(
    database_path: Path,
    counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    with _connect(database_path) as connection:
        queryable = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'table_queryability_decisions'
                  AND json_extract(record_json, '$.reason') = 'queryable'
                """
            ).fetchone()[0]
        )
        text_assets = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'bridge_assets'
                  AND json_extract(record_json, '$.asset_type') = 'text'
                """
            ).fetchone()[0]
        )
        image_assets = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'bridge_assets'
                  AND json_extract(record_json, '$.asset_type') = 'image'
                """
            ).fetchone()[0]
        )
    source_tables = counts["source_tables"]
    return {
        "processed_tables": source_tables,
        "skipped_tables": 0,
        "source_tables": source_tables,
        "queryable_source_tables": queryable,
        "rejected_source_tables": source_tables - queryable,
        "query_tables": counts["query_tables"],
        "data_lake_tables": counts["data_lake_tables"],
        "qrels": counts["qrels"],
        "unique_wiki_entities": counts["entities"],
        "text_assets": text_assets,
        "image_assets": image_assets,
        "table_asset_links": counts["table_asset_links"],
        "attribute_extractions": counts["attribute_extractions"],
        "evidence_recoveries": counts["evidence_recoveries"],
        "min_recovered_value_ratio": args.min_recovered_value_ratio,
        "min_recovery_denominator": args.min_recovery_denominator,
        "query_rows_per_table": (
            join_builder.configured_query_rows_per_table(args)
        ),
        "notes": [
            "source_tables are the fixed data-lake base pool",
            "source-table rows are never capped",
            "query construction is delegated to "
            "build_mm_joinability_dataset.py",
        ],
    }


def _completed_payload(shard: CompletedShard) -> dict[str, Any]:
    return asdict(shard)


def _artifact_manifest(
    artifact: str,
    shards: Iterable[CompletedShard],
    records_per_shard: int,
    output_root: Path,
) -> dict[str, Any]:
    shard_list = list(shards)
    return {
        "directory": artifact,
        "total_records": sum(shard.records for shard in shard_list),
        "max_records_per_shard": records_per_shard,
        "shards": [
            {
                **_completed_payload(shard),
                "mtime_ns": (
                    output_root / shard.path
                ).stat().st_mtime_ns,
            }
            for shard in shard_list
        ],
    }


def _load_published_result(
    output_root: Path,
    *,
    upstream: _ValidatedUpstream,
    parameter_fingerprint: str,
    records_per_shard: int,
) -> MaterializationResult | None:
    manifest_path = output_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "published dataset manifest validation failed"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("stage") != "wdc200k_materialization"
        or payload.get("schema_version")
        != MATERIALIZATION_SCHEMA_VERSION
        or payload.get("complete") is not True
        or payload.get("input_identity") != upstream.identity
        or payload.get("parameter_fingerprint")
        != parameter_fingerprint
        or int(payload.get("records_per_shard", 0))
        != records_per_shard
    ):
        raise ValueError(
            "published dataset manifest identity validation failed"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        _CORE_ARTIFACTS
    ):
        raise ValueError(
            "published dataset artifact manifest validation failed"
        )
    try:
        for artifact in _CORE_ARTIFACTS:
            current = artifacts[artifact]
            declared_shards = current.get("shards", [])
            shards = [
                _completed_from_payload(item) for item in declared_shards
            ]
            if (
                int(current.get("total_records", -1))
                != sum(shard.records for shard in shards)
                or not all(
                    validate_completed_shard(shard, output_root)
                    for shard in shards
                )
                or any(
                    int(item.get("mtime_ns", -1))
                    != (output_root / shard.path).stat().st_mtime_ns
                    for item, shard in zip(declared_shards, shards)
                )
            ):
                raise ValueError
        singles = payload["published_single_files"]
        required_singles = {
            "qrels.jsonl",
            "splits.json",
            "stats.json",
            "table_queryability_decisions.jsonl",
            *_DIAGNOSTIC_FILES,
        }
        if not isinstance(singles, dict) or set(singles) != required_singles:
            raise ValueError
        for item in singles.values():
            shard = _completed_from_payload(item)
            if (
                not validate_completed_shard(shard, output_root)
                or int(item.get("mtime_ns", -1))
                != (output_root / shard.path).stat().st_mtime_ns
            ):
                raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("published shard validation failed") from None
    stats = json.loads(
        (output_root / "stats.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(stats, dict)
        or int(stats.get("source_tables", -1))
        != upstream.expected_tables
    ):
        raise ValueError("published stats validation failed")
    return MaterializationResult(
        output_root=output_root,
        manifest_path=manifest_path,
        stats=stats,
        input_identity=upstream.identity,
        parameter_fingerprint=parameter_fingerprint,
    )


def _finalize_dataset(
    database_path: Path,
    *,
    output_root: Path,
    upstream: _ValidatedUpstream,
    args: argparse.Namespace,
    parameter_fingerprint: str,
    records_per_shard: int,
) -> MaterializationResult:
    counts = _validate_global_counts(database_path, upstream)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_shards: dict[str, tuple[CompletedShard, ...]] = {}
    for artifact in _CORE_ARTIFACTS:
        writer = _AtomicArtifactWriter(
            output_root,
            artifact,
            records_per_shard,
        )
        try:
            for record in _iter_materialized(database_path, artifact):
                writer.write(record)
            artifact_shards[artifact] = writer.close()
        except BaseException:
            writer.abort()
            raise

    qrels = _atomic_jsonl_from_records(
        output_root / "qrels.jsonl",
        _iter_materialized(database_path, "qrels"),
    )
    decisions = _atomic_jsonl_from_records(
        output_root / "table_queryability_decisions.jsonl",
        _iter_materialized(
            database_path, "table_queryability_decisions"
        ),
    )
    splits = _write_splits(
        database_path,
        output_root / "splits.json",
        args,
    )
    stats_payload = _stats_payload(database_path, counts, args)
    stats = _atomic_json(output_root / "stats.json", stats_payload)
    diagnostics = {
        "web_fetch_failures.jsonl": _atomic_jsonl_from_records(
            output_root / "web_fetch_failures.jsonl",
            _failure_records(upstream.structural_failure_paths),
        ),
        "media_download_failures.jsonl": _atomic_jsonl_from_records(
            output_root / "media_download_failures.jsonl",
            (),
        ),
        "model_attribute_errors.jsonl": _atomic_jsonl_from_records(
            output_root / "model_attribute_errors.jsonl",
            _failure_records(upstream.model_error_paths),
        ),
    }
    single_shards = {
        "qrels.jsonl": CompletedShard(
            path="qrels.jsonl",
            records=qrels.records,
            bytes=qrels.bytes,
            sha256=qrels.sha256,
        ),
        "splits.json": CompletedShard(
            path="splits.json",
            records=splits.records,
            bytes=splits.bytes,
            sha256=splits.sha256,
        ),
        "stats.json": CompletedShard(
            path="stats.json",
            records=stats.records,
            bytes=stats.bytes,
            sha256=stats.sha256,
        ),
        "table_queryability_decisions.jsonl": CompletedShard(
            path="table_queryability_decisions.jsonl",
            records=decisions.records,
            bytes=decisions.bytes,
            sha256=decisions.sha256,
        ),
        **{
            name: CompletedShard(
                path=name,
                records=shard.records,
                bytes=shard.bytes,
                sha256=shard.sha256,
            )
            for name, shard in diagnostics.items()
        },
    }
    manifest = {
        "stage": "wdc200k_materialization",
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "format": "sharded_jsonl",
        "records_per_shard": records_per_shard,
        "input_identity": upstream.identity,
        "parameter_fingerprint": parameter_fingerprint,
        "upstream": upstream.provenance,
        "artifacts": {
            artifact: _artifact_manifest(
                artifact,
                artifact_shards[artifact],
                records_per_shard,
                output_root,
            )
            for artifact in _CORE_ARTIFACTS
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "stats": "stats.json",
            "table_queryability_decisions": (
                "table_queryability_decisions.jsonl"
            ),
        },
        "published_single_files": {
            name: {
                **_completed_payload(shard),
                "mtime_ns": (
                    output_root / shard.path
                ).stat().st_mtime_ns,
            }
            for name, shard in single_shards.items()
        },
        "query_construction": _parameter_payload(args),
        "wdc_sampling": {
            "validated_source_tables": upstream.expected_tables,
            "source_rows_capped": False,
        },
        "source_provenance": {
            "builder": "build_wdc_mm_joinability_dataset.py",
            "corpus": "WDC Schema.org Table Corpus 2023",
        },
        "complete": True,
        "note": (
            "Read only shards listed in this manifest; pipeline work and "
            "cache state are stored outside this output directory."
        ),
    }
    manifest_path = output_root / "dataset_manifest.json"
    _atomic_json(manifest_path, manifest)
    return MaterializationResult(
        output_root=output_root,
        manifest_path=manifest_path,
        stats=stats_payload,
        input_identity=upstream.identity,
        parameter_fingerprint=parameter_fingerprint,
    )


def materialize_dataset(
    inputs: MaterializationInputs,
    *,
    output_root: Path,
    args: argparse.Namespace,
    records_per_shard: int = 50_000,
    after_table_commit: Callable[[str], None] | None = None,
) -> MaterializationResult:
    """Validate upstream barriers and stream the canonical final dataset."""
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    output_root = Path(output_root).resolve()
    work_root = Path(inputs.work_root).resolve()
    if (
        output_root == work_root
        or output_root.is_relative_to(work_root)
        or work_root.is_relative_to(output_root)
    ):
        raise ValueError("work_root and output_root must be separate")
    upstream = _validate_upstream(inputs)
    parameter_fingerprint = _parameter_fingerprint(args)
    resumed = _load_published_result(
        output_root,
        upstream=upstream,
        parameter_fingerprint=parameter_fingerprint,
        records_per_shard=records_per_shard,
    )
    if resumed is not None:
        return resumed

    work_root.mkdir(parents=True, exist_ok=True)
    database_path = (
        work_root
        / "materialization"
        / (
            f"index-{upstream.identity}-"
            f"{parameter_fingerprint}.sqlite3"
        )
    )
    _prepare_authoritative_index(database_path, upstream)
    _catalog_sources(
        database_path,
        upstream.source_paths,
        args=args,
        expected_tables=upstream.expected_tables,
    )
    _assign_splits(database_path, args)
    _materialize_all_tables(
        database_path,
        args=args,
        expected_tables=upstream.expected_tables,
        after_table_commit=after_table_commit,
    )
    return _finalize_dataset(
        database_path,
        output_root=output_root,
        upstream=upstream,
        args=args,
        parameter_fingerprint=parameter_fingerprint,
        records_per_shard=records_per_shard,
    )

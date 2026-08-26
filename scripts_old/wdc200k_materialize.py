"""Streaming canonical materialization for the WDC 200K pipeline."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from itertools import zip_longest
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import build_mm_joinability_dataset as join_builder
    import wdc200k_structural as structural
    from build_mm_table_dataset import normalize_title
    from stage1_io import clean_text, stable_hash
    from wdc200k_assets import (
        AssetPlanShards,
        ImageFetchResult,
        MaterializedAssetShards,
        UniqueImageJobs,
        asset_materialization_input_fingerprint,
        asset_planning_input_fingerprint,
        iter_image_failures,
        structural_asset_input_identity,
        validate_asset_plan_shards,
        validate_complete_image_fetch,
        validate_materialized_asset_shards,
        validate_unique_image_jobs,
    )
    from wdc200k_fetch import (
        FetchResult,
        validate_complete_page_fetch,
    )
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        GuardedTextWriter,
        GuardedWriteTracker,
        PreWriteGuard,
        validate_completed_shard,
    )
    from wdc200k_models import (
        AdaptedModelTasks,
        ModelJobInfo,
        ModelJobSet,
        ModelStageAuthority,
        ModelStageResult,
        StructuralStageBarrier,
        model_adapter_input_fingerprint,
        validate_adapted_model_tasks,
        validate_model_stage_for_adapter,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "build_mm_joinability_dataset",
        "build_mm_table_dataset",
        "stage1_io",
        "wdc200k_assets",
        "wdc200k_fetch",
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
        from wdc200k_assets import (
            AssetPlanShards,
            ImageFetchResult,
            MaterializedAssetShards,
            UniqueImageJobs,
            asset_materialization_input_fingerprint,
            asset_planning_input_fingerprint,
            iter_image_failures,
            structural_asset_input_identity,
            validate_asset_plan_shards,
            validate_complete_image_fetch,
            validate_materialized_asset_shards,
            validate_unique_image_jobs,
        )
        from wdc200k_fetch import (
            FetchResult,
            validate_complete_page_fetch,
        )
        from wdc200k_io import (
            AtomicJsonlShard,
            CompletedShard,
            GuardedTextWriter,
            GuardedWriteTracker,
            PreWriteGuard,
            validate_completed_shard,
        )
        from wdc200k_models import (
            AdaptedModelTasks,
            ModelJobInfo,
            ModelJobSet,
            ModelStageAuthority,
            ModelStageResult,
            StructuralStageBarrier,
            model_adapter_input_fingerprint,
            validate_adapted_model_tasks,
            validate_model_stage_for_adapter,
        )
    finally:
        sys.path.remove(scripts_directory)


MATERIALIZATION_SCHEMA_VERSION = "wdc200k-materialization-v7"
UPSTREAM_CERTIFICATE_SCHEMA_VERSION = (
    "wdc200k-upstream-certificate-v1"
)
MAX_MATERIALIZATION_VALIDATION_WORKERS = 4
DATASET_REFERENCE_FORMAT = "source-table-reference-v1"
_MATERIALIZED_REFERENCE_STORAGE_VERSION = (
    "materialized-source-references-v1"
)
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
    page_fetch_result: FetchResult
    asset_plan_result: AssetPlanShards
    unique_image_jobs: UniqueImageJobs
    image_fetch_result: ImageFetchResult
    materialized_assets: MaterializedAssetShards
    adapted_model_tasks: AdaptedModelTasks
    model_result: ModelStageResult
    model_authority: ModelStageAuthority
    work_root: Path
    sampling_manifest: Path | None = None
    sampled_entity_paths: tuple[Path, ...] = ()
    sampled_page_ref_paths: tuple[Path, ...] = ()
    upstream_stage_registry: Path | None = None


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
class _MaterializationWorkItem:
    source_table_id: str
    source_ordinal: int
    source_sha256: str
    split: str


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
    page_ref_paths: tuple[Path, ...]
    structural_failure_paths: tuple[Path, ...]
    page_failure_path: Path
    asset_paths: tuple[Path, ...]
    link_paths: tuple[Path, ...]
    extraction_paths: tuple[Path, ...]
    model_error_paths: tuple[Path, ...]
    adapter_error_paths: tuple[Path, ...]
    asset_plan_result: AssetPlanShards
    image_fetch_result: ImageFetchResult
    image_failure_aggregation_database: Path
    expected_tables: int
    expected_entities: int
    expected_assets: int
    expected_links: int
    expected_extractions: int
    identity: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _FastResumeState:
    upstream: _ValidatedUpstream
    database_path: Path
    resumed_source_units: int


class _CertificateMismatch(ValueError):
    """A fast-resume certificate cannot authorize the current inputs."""


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
        self.transient_items: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        record = self.items.get(key)
        return dict(record) if record is not None else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.items[key] = dict(record)

    def get_transient(self, key: str) -> dict[str, Any] | None:
        record = self.transient_items.get(key)
        return dict(record) if record is not None else None

    def put_transient(self, key: str, record: dict[str, Any]) -> None:
        self.transient_items[key] = dict(record)


class _CachedQueryAutoCheckExtractor:
    """Require cached query checks without carrying a live client to workers."""

    auto_check_enabled = True

    def __init__(self, review_policy: str) -> None:
        self.auto_check_luna_reviewer = (
            object()
            if review_policy
            == join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE
            else None
        )

    def extract_auto_check_value(self, **_kwargs: Any) -> str:
        raise RuntimeError(
            "query auto-check cache is incomplete during final materialization"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_INLINE_JSON_MAX_UTF8_BYTES = 4 * 1024 * 1024
_JSON_TEXT_CHUNK_CHARACTERS = 1024 * 1024
_MATERIALIZATION_READ_BATCH_RECORDS = 32
_SQLITE_IN_BATCH_RECORDS = 900


def _json_text_identity(value: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    for offset in range(0, len(value), _JSON_TEXT_CHUNK_CHARACTERS):
        block = value[
            offset : offset + _JSON_TEXT_CHUNK_CHARACTERS
        ].encode("utf-8")
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def _external_json_relative_path(
    namespace: str,
    identity: str,
    digest: str,
) -> Path:
    safe_namespace = Path(*Path(namespace).parts)
    if (
        safe_namespace.is_absolute()
        or ".." in safe_namespace.parts
        or not safe_namespace.parts
    ):
        raise ValueError(f"invalid external JSON namespace: {namespace}")
    if not identity:
        raise ValueError("external JSON identity must not be empty")
    return (
        Path("large-json")
        / "sha256"
        / digest[:2]
        / f"{digest}.json.gz"
    )


def _external_json_stub(
    record: dict[str, Any],
    *,
    digest: str,
    size: int,
) -> str:
    external = {
        "sha256": digest,
        "utf8_bytes": size,
    }
    rows = record.get("rows")
    if isinstance(rows, list):
        external["row_count"] = len(rows)
    stub: dict[str, Any] = {"_external_record": external}
    for key in (
        "split",
        "source_table_id",
        "table_id",
        "query_table_id",
        "role",
        "asset_type",
        "reason",
    ):
        if key in record:
            stub[key] = record[key]
    return _canonical_json(stub)


def _write_external_json(
    database_path: Path,
    relative_path: Path,
    encoded: str,
) -> None:
    root = database_path.parent.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"external JSON path escapes index root: {path}")
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                for offset in range(
                    0,
                    len(encoded),
                    _JSON_TEXT_CHUNK_CHARACTERS,
                ):
                    compressed.write(
                        encoded[
                            offset : offset
                            + _JSON_TEXT_CHUNK_CHARACTERS
                        ].encode("utf-8")
                    )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _stored_json_values(
    database_path: Path,
    *,
    namespace: str,
    identity: str,
    record: dict[str, Any],
    encoded: str,
    encoded_size: int,
    digest: str,
) -> tuple[str, str]:
    if encoded_size <= _INLINE_JSON_MAX_UTF8_BYTES:
        return encoded, ""
    relative_path = _external_json_relative_path(
        namespace,
        identity,
        digest,
    )
    _write_external_json(database_path, relative_path, encoded)
    return (
        _external_json_stub(
            record,
            digest=digest,
            size=encoded_size,
        ),
        relative_path.as_posix(),
    )


def _load_stored_json(
    database_path: Path,
    record_json: Any,
    record_path: Any,
) -> dict[str, Any]:
    relative = clean_text(record_path)
    if not relative:
        record = json.loads(str(record_json))
    else:
        root = database_path.parent.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"external JSON record is missing or unsafe: {relative}"
            )
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            record = json.load(handle)
    if not isinstance(record, dict):
        raise ValueError("stored JSON record is not an object")
    return record


def _migrate_legacy_external_json_paths(
    database_path: Path,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> int:
    """Atomically relink referenced legacy blobs into the shared CAS."""

    root = database_path.parent.resolve()
    write_tracker = GuardedWriteTracker(
        database_path,
        pre_write_guard,
    )
    with _connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT record_path, MIN(record_json) AS record_json
            FROM (
                SELECT record_path, record_json
                FROM source_catalog
                WHERE record_path <> ''
                  AND record_path NOT LIKE 'large-json/sha256/%'
                UNION ALL
                SELECT record_path, record_json
                FROM materialized_records
                WHERE record_path <> ''
                  AND record_path NOT LIKE 'large-json/sha256/%'
            )
            GROUP BY record_path
            ORDER BY record_path
            """
        ).fetchall()
    migrated = 0
    for row in rows:
        legacy_relative = Path(str(row["record_path"]))
        legacy_path = (root / legacy_relative).resolve()
        if (
            not legacy_path.is_relative_to(root)
            or not legacy_path.is_file()
        ):
            raise ValueError(
                "legacy external JSON record is missing or unsafe: "
                f"{legacy_relative.as_posix()}"
            )
        stub = json.loads(str(row["record_json"]))
        external = stub.get("_external_record")
        if not isinstance(external, dict):
            raise ValueError(
                "legacy external JSON record has no identity: "
                f"{legacy_relative.as_posix()}"
            )
        digest = clean_text(external.get("sha256"))
        if len(digest) != 64:
            raise ValueError(
                "legacy external JSON digest is invalid: "
                f"{legacy_relative.as_posix()}"
            )
        canonical_relative = _external_json_relative_path(
            "legacy-migration",
            legacy_relative.as_posix(),
            digest,
        )
        canonical_path = (root / canonical_relative).resolve()
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        if not canonical_path.exists():
            os.link(legacy_path, canonical_path)
            _fsync_directory(canonical_path.parent)

        write_tracker.before_write(16 * 1024)
        with _connect(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE source_catalog SET record_path = ?
                WHERE record_path = ?
                """,
                (
                    canonical_relative.as_posix(),
                    legacy_relative.as_posix(),
                ),
            )
            connection.execute(
                """
                UPDATE materialized_records SET record_path = ?
                WHERE record_path = ?
                """,
                (
                    canonical_relative.as_posix(),
                    legacy_relative.as_posix(),
                ),
            )
            write_tracker.before_commit(0)
            connection.commit()
        legacy_path.unlink()
        migrated += 1

    legacy_root = root / "large-json"
    for directory in sorted(
        {
            (root / Path(str(row["record_path"]))).resolve().parent
            for row in rows
        },
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        while (
            directory != legacy_root
            and directory.is_relative_to(legacy_root)
        ):
            try:
                directory.rmdir()
            except OSError:
                break
            directory = directory.parent
    if migrated:
        _checkpoint_wal(database_path)
    return migrated


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
    list[Path],
    int,
    int,
    list[tuple[Path, str]],
]:
    root = Path(inputs.structural_output_root)
    from wdc200k_sampling import validate_sampling_source_authority

    compact_authority = (
        validate_sampling_source_authority(
            Path(inputs.sampling_manifest),
            structural_output_root=root,
        )
        if inputs.sampling_manifest is not None
        else None
    )
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
    page_ref_paths: list[Path] = []
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
        if compact_authority is None:
            validated_path, records, validated_digest = (
                structural._validated_shard_from_manifest(
                    manifest_path,
                    output_root=root,
                )
            )
            validated_paths.append(validated_path)
            tables += records
        else:
            validated_digest = digest
            source_items = [
                item
                for item in payload.get("completed_shards", [])
                if str(item["path"]).startswith("source_tables/")
            ]
            if len(source_items) != 1:
                raise ValueError("compact structural source set mismatch")
            tables += int(source_items[0]["records"])
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
            elif shard.path.startswith("page_refs/"):
                page_ref_paths.append(path)

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
    if compact_authority is None:
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
    else:
        source_paths = list(compact_authority.source_tables)
        if not inputs.sampled_entity_paths:
            raise ValueError("sampled entity authority is missing")
        entity_paths = [Path(path) for path in inputs.sampled_entity_paths]
        entities = sum(1 for _record in _iter_jsonl(entity_paths))
    return (
        source_paths,
        entity_paths,
        page_ref_paths,
        failure_paths,
        tables,
        entities,
        manifest_hashes,
    )


def _materialization_validation_workers(args: argparse.Namespace) -> int:
    workers = int(
        getattr(args, "materialization_validation_workers", 1)
    )
    if not 1 <= workers <= MAX_MATERIALIZATION_VALIDATION_WORKERS:
        raise ValueError(
            "materialization validation worker count must be between "
            f"1 and {MAX_MATERIALIZATION_VALIDATION_WORKERS}"
        )
    return workers


def _report_validation_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    mode: str,
    completed: int,
    total: int,
    **details: Any,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": "upstream_validation",
                "mode": mode,
                "completed": completed,
                "total": total,
                **details,
            }
        )


def _report_work_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    completed: int,
    total: int,
    **details: Any,
) -> None:
    if callback is not None:
        callback(
            {
                "phase": phase,
                "completed": completed,
                "total": total,
                **details,
            }
        )


def _truncate_validation_wals(paths: Iterable[Path]) -> None:
    for path in paths:
        database_path = Path(path)
        if not database_path.is_file():
            continue
        try:
            with sqlite3.connect(database_path, timeout=30.0) as connection:
                connection.execute("PRAGMA busy_timeout=30000")
                connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as error:
            logging.warning(
                "could not truncate validation WAL for %s: %s",
                database_path,
                error,
            )


def _validate_upstream(
    inputs: MaterializationInputs,
    *,
    args: argparse.Namespace,
    pre_write_guard: PreWriteGuard | None = None,
    validation_progress_callback: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> _ValidatedUpstream:
    (
        source_paths,
        entity_paths,
        page_ref_paths,
        failure_paths,
        tables,
        entities,
        structural_hashes,
    ) = _structural_inputs(inputs)
    from wdc200k_sampling import validate_sampling_artifacts

    effective_page_ref_paths = (
        list(inputs.sampled_page_ref_paths)
        if inputs.sampled_page_ref_paths
        else page_ref_paths
    )
    validation_root = Path(inputs.work_root) / "upstream-validation"
    structural_identity = (
        _sha256_path(Path(inputs.sampling_manifest))
        if inputs.sampling_manifest is not None
        else structural_asset_input_identity(
            (digest for _path, digest in structural_hashes),
            inputs.structural_barrier.final_manifest_sha256,
        )
    )
    expected_adapter_input = model_adapter_input_fingerprint(
        (digest for _path, digest in structural_hashes),
        finalized_selection_manifest=(
            inputs.finalized_selection_manifest
        ),
        assets_manifest=inputs.materialized_assets.manifest_path,
    )
    if inputs.sampling_manifest is not None:
        expected_adapter_input = stable_hash(
            expected_adapter_input,
            _sha256_path(Path(inputs.sampling_manifest)),
            length=40,
        )
    total_validation_tasks = 6
    completed_validation_tasks = 0
    workers = _materialization_validation_workers(args)
    _report_validation_progress(
        validation_progress_callback,
        mode="strict",
        completed=0,
        total=total_validation_tasks,
        workers=workers,
    )
    logging.info(
        "materialization upstream validation: strict path with %d worker(s)",
        workers,
    )

    def validate_sampling() -> Any:
        return (
            validate_sampling_artifacts(Path(inputs.sampling_manifest))
            if inputs.sampling_manifest is not None
            else None
        )

    def validate_page_and_plan() -> tuple[dict[str, Any], AssetPlanShards]:
        snapshot = validate_complete_page_fetch(
            inputs.page_fetch_result,
            _iter_jsonl(effective_page_ref_paths),
            validation_database=validation_root / "page-fetch.sqlite3",
            pre_write_guard=pre_write_guard,
        )
        expected_planning_input = asset_planning_input_fingerprint(
            structural_identity,
            str(snapshot["identity"]),
        )
        return (
            snapshot,
            validate_asset_plan_shards(
                inputs.asset_plan_result,
                expected_input_fingerprint=expected_planning_input,
            ),
        )

    def validate_unique() -> UniqueImageJobs:
        return validate_unique_image_jobs(
            inputs.unique_image_jobs,
            planned=inputs.asset_plan_result,
            validation_database=(
                validation_root / "unique-image-membership.sqlite3"
            ),
            pre_write_guard=pre_write_guard,
        )

    def validate_image() -> dict[str, Any]:
        if (
            inputs.image_fetch_result.unique_jobs
            != inputs.unique_image_jobs
        ):
            raise ValueError("Task-5 image fetch unique-job substitution")
        return validate_complete_image_fetch(
            inputs.image_fetch_result,
            unique_jobs=inputs.unique_image_jobs,
            pre_write_guard=pre_write_guard,
        )

    def validate_models() -> AdaptedModelTasks:
        current = validate_adapted_model_tasks(
            inputs.adapted_model_tasks,
            expected_input_fingerprint=expected_adapter_input,
        )
        validate_model_stage_for_adapter(
            inputs.model_result,
            current,
            args=args,
            authority=inputs.model_authority,
            validation_store_path=(
                validation_root / "model-membership.sqlite3"
            ),
        )
        return current

    def task_finished(name: str) -> None:
        nonlocal completed_validation_tasks
        completed_validation_tasks += 1
        logging.info(
            "materialization upstream validation completed %s (%d/%d)",
            name,
            completed_validation_tasks,
            total_validation_tasks,
        )
        _report_validation_progress(
            validation_progress_callback,
            mode="strict",
            completed=completed_validation_tasks,
            total=total_validation_tasks,
            workers=workers,
            task=name,
        )

    def check_sampled_authority(sampled_authority: Any) -> None:
        if sampled_authority is None:
            return
        expected_entities = sampled_authority.artifact_paths[
            "sampled_entities"
        ]
        expected_pages = sampled_authority.artifact_paths[
            "sampled_page_refs"
        ]
        if tuple(
            path.resolve() for path in inputs.sampled_entity_paths
        ) != tuple(path.resolve() for path in expected_entities) or tuple(
            path.resolve() for path in effective_page_ref_paths
        ) != tuple(path.resolve() for path in expected_pages):
            raise ValueError(
                "sampled materialization paths do not match manifest authority"
            )

    if workers == 1:
        sampled_authority = validate_sampling()
        task_finished("sampling")
        check_sampled_authority(sampled_authority)
        page_snapshot, planned = validate_page_and_plan()
        task_finished("page_and_planning")
        unique_jobs = validate_unique()
        task_finished("unique_image_membership")
        validate_image()
        task_finished("image_membership")
        adapted = None
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="wdc-upstream-validation",
        ) as executor:
            sampling_future = executor.submit(validate_sampling)
            page_future = executor.submit(validate_page_and_plan)
            model_future = executor.submit(validate_models)
            unique_future = executor.submit(validate_unique)
            image_future = executor.submit(validate_image)

            sampled_authority = sampling_future.result()
            task_finished("sampling")
            check_sampled_authority(sampled_authority)
            page_snapshot, planned = page_future.result()
            task_finished("page_and_planning")
            unique_jobs = unique_future.result()
            task_finished("unique_image_membership")
            image_future.result()
            task_finished("image_membership")
            adapted = model_future

    if inputs.image_fetch_result.unique_jobs != unique_jobs:
        raise ValueError("Task-5 image fetch unique-job substitution")
    expected_asset_input = asset_materialization_input_fingerprint(
        planned.manifest_path,
        inputs.image_fetch_result.fetch_manifest_path,
    )
    materialized_assets, assets_barrier = validate_materialized_asset_shards(
        inputs.materialized_assets,
        planned=planned,
        image_fetch_result=inputs.image_fetch_result,
        expected_input_fingerprint=expected_asset_input,
        pre_write_guard=pre_write_guard,
    )
    task_finished("asset_closure")
    if workers == 1:
        adapted = validate_models()
    else:
        adapted = adapted.result()
    task_finished("model_membership")
    _truncate_validation_wals(
        (
            validation_root / "page-fetch.sqlite3",
            validation_root / "unique-image-membership.sqlite3",
            validation_root / "model-membership.sqlite3",
        )
    )
    asset_paths = materialized_assets.bridge_asset_paths
    link_paths = materialized_assets.table_asset_link_paths
    retained_link_count = sum(
        1
        for link in _iter_jsonl(link_paths)
        if any(
            clean_text(asset_id)
            for asset_id in (link.get("asset_ids") or [])
        )
    )
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
    asset_manifest_sha256 = _sha256_path(
        materialized_assets.manifest_path
    )
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
        "page_fetch_identity": page_snapshot["identity"],
        "sampling_manifest_sha256": (
            _sha256_path(Path(inputs.sampling_manifest))
            if inputs.sampling_manifest is not None
            else None
        ),
        "asset_planning_manifest_sha256": _sha256_path(
            planned.manifest_path
        ),
        "unique_image_job_manifest_sha256": _sha256_path(
            unique_jobs.manifest_path
        ),
        "image_fetch_manifest_sha256": _sha256_path(
            inputs.image_fetch_result.fetch_manifest_path
        ),
        "asset_manifest_sha256": asset_manifest_sha256,
        "asset_fingerprint": assets_barrier.fingerprint,
        "model_adapter_manifest_sha256": _sha256_path(
            adapted.manifest_path
        ),
        "model_adapter_input_fingerprint": adapted.input_fingerprint,
        "model_authority": asdict(inputs.model_authority),
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
        page_ref_paths=tuple(effective_page_ref_paths),
        structural_failure_paths=tuple(failure_paths),
        page_failure_path=Path(inputs.page_fetch_result.failure_path),
        asset_paths=tuple(asset_paths),
        link_paths=tuple(link_paths),
        extraction_paths=tuple(
            model_root / shard.path for shard in model_extraction_shards
        ),
        model_error_paths=tuple(
            model_root / shard.path for shard in model_error_shards
        ),
        adapter_error_paths=adapted.error_paths,
        asset_plan_result=planned,
        image_fetch_result=inputs.image_fetch_result,
        image_failure_aggregation_database=(
            validation_root / "image-failure-fanout.sqlite3"
        ),
        expected_tables=tables,
        expected_entities=entities,
        expected_assets=assets_barrier.bridge_assets,
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
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA wal_autocheckpoint=1000")
    return connection


def _checkpoint_wal(path: Path) -> None:
    with _connect(path) as connection:
        result = connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
    if result is not None and int(result[0]) != 0:
        raise RuntimeError(
            f"materialization WAL checkpoint remained busy: {path}"
        )


def _initialize_index(
    path: Path,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    tracker.before_write(64 * 1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL,
                source_row_id INTEGER NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS entities_source
                ON entities(source_table_id, source_row_id, entity_id);

            CREATE TABLE IF NOT EXISTS entity_sources (
                entity_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                source_row_id INTEGER NOT NULL,
                PRIMARY KEY (entity_id, source_table_id, source_row_id)
            );
            CREATE INDEX IF NOT EXISTS entity_sources_source
                ON entity_sources(source_table_id, source_row_id, entity_id);

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

            CREATE TABLE IF NOT EXISTS link_assets (
                link_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                PRIMARY KEY (link_id, asset_id)
            );
            CREATE INDEX IF NOT EXISTS link_assets_asset
                ON link_assets(asset_id, link_id);

            CREATE TABLE IF NOT EXISTS extractions (
                cache_key TEXT PRIMARY KEY,
                model_call_key TEXT NOT NULL,
                job_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                source_row_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS extractions_entity_asset
                ON extractions(entity_id, asset_id, cache_key);
            CREATE INDEX IF NOT EXISTS extractions_asset
                ON extractions(asset_id, cache_key);
            CREATE INDEX IF NOT EXISTS extractions_source
                ON extractions(source_table_id, cache_key);
            CREATE UNIQUE INDEX IF NOT EXISTS extractions_full_call
                ON extractions(model_call_key)
                WHERE model_call_key <> '';
            CREATE UNIQUE INDEX IF NOT EXISTS extractions_job
                ON extractions(job_id)
                WHERE job_id <> '';

            CREATE TABLE IF NOT EXISTS query_auto_checks (
                cache_key TEXT PRIMARY KEY,
                extraction_cache_key TEXT NOT NULL,
                record_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS query_auto_checks_extraction
                ON query_auto_checks(extraction_cache_key, cache_key);

            CREATE TABLE IF NOT EXISTS query_auto_check_units (
                source_table_id TEXT PRIMARY KEY,
                source_sha256 TEXT NOT NULL,
                plan_count INTEGER NOT NULL,
                cached_check_count INTEGER NOT NULL,
                complete INTEGER NOT NULL
            );

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
                record_json TEXT NOT NULL,
                record_path TEXT NOT NULL DEFAULT ''
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
                record_path TEXT NOT NULL DEFAULT '',
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
        extraction_columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(extractions)"
            )
        }
        if "source_row_id" not in extraction_columns:
            connection.execute(
                """
                ALTER TABLE extractions
                ADD COLUMN source_row_id INTEGER NOT NULL DEFAULT 0
                """
            )
        for table in ("source_catalog", "materialized_records"):
            columns = {
                str(row["name"])
                for row in connection.execute(
                    f"PRAGMA table_info({table})"
                )
            }
            if "record_path" not in columns:
                connection.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD COLUMN record_path TEXT NOT NULL DEFAULT ''
                    """
                )
        tracker.before_commit(0)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


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
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        if write_tracker is not None:
            encoded = _canonical_json(record)
            write_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
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
        appearances = record.get("appears_in")
        if not isinstance(appearances, list) or not appearances:
            raise ValueError("entity is missing source-table appearance")
        for appearance in appearances:
            if not isinstance(appearance, dict):
                raise ValueError("entity appearance is not an object")
            appearance_source = clean_text(
                appearance.get("source_table_id")
            )
            if not appearance_source:
                raise ValueError(
                    "entity appearance is missing source_table_id"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO entity_sources (
                    entity_id, source_table_id, source_row_id
                ) VALUES (?, ?, ?)
                """,
                (
                    entity_id,
                    appearance_source,
                    int(appearance.get("row_id", 0)),
                ),
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
        _commit_index_batch(
            connection,
            count,
            commit_every,
            write_tracker=write_tracker,
        )


def _index_assets(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    commit_every: int = 0,
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        if write_tracker is not None:
            encoded = _canonical_json(record)
            write_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
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
        _commit_index_batch(
            connection,
            count,
            commit_every,
            write_tracker=write_tracker,
        )


def _index_links(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    commit_every: int = 0,
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        if write_tracker is not None:
            encoded = _canonical_json(record)
            write_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
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
        asset_ids = record.get("asset_ids") or []
        if not isinstance(asset_ids, list):
            raise ValueError("asset link asset_ids is not a list")
        for asset_id_value in asset_ids:
            asset_id = clean_text(asset_id_value)
            if not asset_id:
                raise ValueError("asset link contains an empty asset ID")
            connection.execute(
                """
                INSERT OR IGNORE INTO link_assets (link_id, asset_id)
                VALUES (?, ?)
                """,
                (link_id, asset_id),
            )
        for alias in (
            clean_text(record.get("entity_wiki_title")),
            normalize_title(clean_text(record.get("entity_wiki_title"))),
        ):
            _insert_alias(connection, alias, entity_id)
            _insert_alias(connection, alias.casefold(), entity_id)
        _commit_index_batch(
            connection,
            count,
            commit_every,
            write_tracker=write_tracker,
        )


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
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    for count, record in enumerate(_iter_jsonl(paths), start=1):
        if write_tracker is not None:
            encoded = _canonical_json(record)
            write_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
        cache_key, entity_id, asset_id = _extraction_identity(record)
        entity_row = connection.execute(
            """
            SELECT source_table_id, source_row_id
            FROM entities WHERE entity_id = ?
            """,
            (entity_id,),
        ).fetchone()
        declared_source_table_id = clean_text(
            record.get("source_table_id")
        )
        source_table_id = declared_source_table_id or (
            str(entity_row["source_table_id"])
            if entity_row is not None
            else ""
        )
        declared_source_row_id = record.get("source_row_id")
        source_row_id = (
            int(declared_source_row_id)
            if declared_source_row_id is not None
            else (
                int(entity_row["source_row_id"])
                if entity_row is not None
                else 0
            )
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
                "source_row_id": source_row_id,
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
        _commit_index_batch(
            connection,
            count,
            commit_every,
            write_tracker=write_tracker,
        )


def _validate_relation_closure(
    connection: sqlite3.Connection,
) -> None:
    missing_asset_entity = connection.execute(
        """
        SELECT assets.asset_id
        FROM assets
        LEFT JOIN entities USING (entity_id)
        WHERE entities.entity_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if missing_asset_entity is not None:
        raise ValueError(
            "asset entity relation is missing: "
            f"{missing_asset_entity['asset_id']}"
        )
    missing_link_entity = connection.execute(
        """
        SELECT links.link_id
        FROM links
        LEFT JOIN entities
          ON entities.entity_id = links.entity_id
        LEFT JOIN entity_sources
          ON entity_sources.entity_id = links.entity_id
         AND entity_sources.source_table_id = links.source_table_id
         AND entity_sources.source_row_id = links.source_row_id
        WHERE entities.entity_id IS NULL
           OR entity_sources.entity_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if missing_link_entity is not None:
        raise ValueError(
            "link entity/source relation is missing: "
            f"{missing_link_entity['link_id']}"
        )
    invalid_link_asset = connection.execute(
        """
        SELECT link_assets.link_id, link_assets.asset_id,
               links.entity_id AS link_entity_id,
               assets.entity_id AS asset_entity_id
        FROM link_assets
        JOIN links USING (link_id)
        LEFT JOIN assets USING (asset_id)
        WHERE assets.asset_id IS NULL
           OR assets.entity_id <> links.entity_id
        LIMIT 1
        """
    ).fetchone()
    if invalid_link_asset is not None:
        if invalid_link_asset["asset_entity_id"] is None:
            raise ValueError(
                "link asset relation is missing: "
                f"{invalid_link_asset['asset_id']}"
            )
        raise ValueError(
            "link asset entity mismatch: "
            f"{invalid_link_asset['link_id']}"
        )
    invalid_extraction = connection.execute(
        """
        SELECT extractions.cache_key,
               extractions.entity_id,
               extractions.asset_id,
               extractions.source_table_id,
               entities.entity_id AS found_entity_id,
               assets.asset_id AS found_asset_id,
               assets.entity_id AS asset_entity_id,
               entity_sources.entity_id AS found_source_entity_id
        FROM extractions
        LEFT JOIN entities
          ON entities.entity_id = extractions.entity_id
        LEFT JOIN assets
          ON assets.asset_id = extractions.asset_id
        LEFT JOIN entity_sources
          ON entity_sources.entity_id = extractions.entity_id
         AND entity_sources.source_table_id =
             extractions.source_table_id
         AND entity_sources.source_row_id =
             extractions.source_row_id
        WHERE entities.entity_id IS NULL
           OR assets.asset_id IS NULL
           OR assets.entity_id <> extractions.entity_id
           OR entity_sources.entity_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    if invalid_extraction is not None:
        if invalid_extraction["found_entity_id"] is None:
            reason = "entity"
        elif invalid_extraction["found_asset_id"] is None:
            reason = "asset"
        elif (
            invalid_extraction["asset_entity_id"]
            != invalid_extraction["entity_id"]
        ):
            reason = "entity"
        else:
            reason = "source"
        raise ValueError(
            f"extraction {reason} relation mismatch: "
            f"{invalid_extraction['cache_key']}"
        )


def _validate_source_catalog_closure(
    connection: sqlite3.Connection,
    *,
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    database_rows = connection.execute("PRAGMA database_list").fetchall()
    database_file = next(
        (
            str(row[2])
            for row in database_rows
            if str(row[1]) == "main" and str(row[2])
        ),
        "",
    )
    if not database_file:
        raise ValueError(
            "source catalog closure requires a file-backed database"
        )
    database_path = Path(database_file)
    validation_root = (
        database_path.parent / ".source-closure-validation"
    )
    guard = None if write_tracker is None else write_tracker.guard
    if guard is not None:
        guard(validation_root, 0)
    validation_root.mkdir(parents=True, exist_ok=True)
    source_rows = int(
        connection.execute(
            """
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN record_path = ''
                        THEN json_array_length(record_json, '$.rows')
                        ELSE json_extract(
                            record_json,
                            '$._external_record.row_count'
                        )
                    END
                ),
                0
            )
            FROM source_catalog
            """
        ).fetchone()[0]
    )
    with ExitStack() as stack:
        temporary_root = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(
                    prefix="closure-",
                    dir=validation_root,
                )
            )
        )
        validation_path = temporary_root / "source-rows.sqlite3"
        validation_tracker = GuardedWriteTracker(
            validation_path,
            guard,
        )
        validation_tracker.before_write(
            64 * 1024 + source_rows * 256
        )
        attached = False
        try:
            connection.execute(
                "ATTACH DATABASE ? AS validation_db",
                (str(validation_path),),
            )
            attached = True
            connection.execute(
                """
                CREATE TABLE validation_db.source_row_validation (
                    source_table_id TEXT NOT NULL,
                    source_row_id INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO validation_db.source_row_validation (
                    source_table_id,
                    source_row_id
                )
                SELECT source_catalog.source_table_id,
                       CAST(
                           json_extract(
                               source_row.value,
                               '$.row_id'
                           ) AS INTEGER
                       )
                FROM source_catalog
                JOIN json_each(
                    source_catalog.record_json,
                    '$.rows'
                ) AS source_row
                WHERE json_type(
                    source_row.value,
                    '$.row_id'
                ) IN ('integer', 'text')
                  AND source_catalog.record_path = ''
                """
            )
            external_sources = connection.execute(
                """
                SELECT source_table_id, record_json, record_path
                FROM source_catalog
                WHERE record_path <> ''
                ORDER BY ordinal
                """
            ).fetchall()
            for catalog_row in external_sources:
                source_table_id = str(
                    catalog_row["source_table_id"]
                )
                source_table = _load_stored_json(
                    database_path,
                    catalog_row["record_json"],
                    catalog_row["record_path"],
                )

                def external_rows() -> Iterator[tuple[str, Any]]:
                    for source_row in source_table.get("rows", []):
                        if not isinstance(source_row, dict):
                            continue
                        row_id = source_row.get("row_id")
                        if (
                            isinstance(row_id, bool)
                            or not isinstance(row_id, (int, str))
                        ):
                            continue
                        yield source_table_id, row_id

                connection.executemany(
                    """
                    INSERT INTO
                        validation_db.source_row_validation (
                            source_table_id,
                            source_row_id
                        )
                    VALUES (?, CAST(? AS INTEGER))
                    """,
                    external_rows(),
                )
            connection.execute(
                """
                CREATE INDEX
                    validation_db.source_row_validation_identity
                ON source_row_validation (
                    source_table_id,
                    source_row_id
                )
                """
            )
            relations = (
                (
                    "entity",
                    "entity_sources",
                    "entity_id",
                ),
                (
                    "link",
                    "links",
                    "link_id",
                ),
                (
                    "extraction",
                    "extractions",
                    "cache_key",
                ),
            )
            for kind, table, identity_column in relations:
                missing = connection.execute(
                    f"""
                    SELECT relation.{identity_column} AS identity,
                           relation.source_table_id,
                           relation.source_row_id
                    FROM {table} AS relation
                    LEFT JOIN source_catalog
                      ON source_catalog.source_table_id =
                         relation.source_table_id
                    LEFT JOIN
                        validation_db.source_row_validation
                        AS valid_source_rows
                      ON valid_source_rows.source_table_id =
                         relation.source_table_id
                     AND valid_source_rows.source_row_id =
                         relation.source_row_id
                    WHERE source_catalog.source_table_id IS NULL
                       OR valid_source_rows.source_table_id IS NULL
                    LIMIT 1
                    """
                ).fetchone()
                if missing is not None:
                    raise ValueError(
                        f"{kind} source catalog relation is missing: "
                        f"{missing['identity']} "
                        f"({missing['source_table_id']}, "
                        f"{missing['source_row_id']})"
                    )
            validation_tracker.before_commit(0)
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()
            if attached:
                connection.execute("DETACH DATABASE validation_db")


def _commit_index_batch(
    connection: sqlite3.Connection,
    count: int,
    commit_every: int,
    *,
    write_tracker: GuardedWriteTracker | None = None,
) -> None:
    if commit_every > 0 and count % commit_every == 0:
        if write_tracker is not None:
            write_tracker.before_commit(0)
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")


def _build_index(
    inputs: MaterializationShardInputs,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    write_tracker = GuardedWriteTracker(
        inputs.lookup_database,
        pre_write_guard,
    )
    write_tracker.before_write(64 * 1024)
    _initialize_index(
        inputs.lookup_database,
        pre_write_guard=pre_write_guard,
    )
    with _connect(inputs.lookup_database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _index_entities(
            connection,
            inputs.entity_paths,
            write_tracker=write_tracker,
        )
        _index_assets(
            connection,
            inputs.asset_paths,
            write_tracker=write_tracker,
        )
        _index_links(
            connection,
            inputs.link_paths,
            write_tracker=write_tracker,
        )
        _index_extractions(
            connection,
            inputs.extraction_paths,
            status="success",
            write_tracker=write_tracker,
        )
        _index_extractions(
            connection,
            inputs.error_paths,
            status="terminal",
            write_tracker=write_tracker,
        )
        _validate_relation_closure(connection)
        write_tracker.before_commit(0)
        connection.commit()


def _prepare_authoritative_index(
    database_path: Path,
    upstream: _ValidatedUpstream,
    *,
    pre_write_guard: PreWriteGuard | None = None,
    validate_relation_closure: bool = True,
) -> None:
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    write_tracker.before_write(64 * 1024)
    _initialize_index(
        database_path,
        pre_write_guard=pre_write_guard,
    )
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
            connection,
            upstream.entity_paths,
            commit_every=1_000,
            write_tracker=write_tracker,
        )
        _index_assets(
            connection,
            upstream.asset_paths,
            commit_every=1_000,
            write_tracker=write_tracker,
        )
        _index_links(
            connection,
            upstream.link_paths,
            commit_every=1_000,
            write_tracker=write_tracker,
        )
        _index_extractions(
            connection,
            upstream.extraction_paths,
            status="success",
            commit_every=1_000,
            write_tracker=write_tracker,
        )
        _index_extractions(
            connection,
            upstream.model_error_paths,
            status="terminal",
            commit_every=1_000,
            write_tracker=write_tracker,
        )
        if validate_relation_closure:
            _validate_relation_closure(connection)
        write_tracker.before_commit(0)
        connection.commit()


def _validate_materialization_index_closures(
    database_path: Path,
    *,
    validation_workers: int,
    pre_write_guard: PreWriteGuard | None = None,
    validation_progress_callback: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> None:
    workers = min(2, validation_workers)

    def validate_relations() -> None:
        with _connect(database_path) as connection:
            _validate_relation_closure(connection)

    def validate_sources() -> None:
        tracker = GuardedWriteTracker(
            database_path,
            pre_write_guard,
        )
        with _connect(database_path) as connection:
            _validate_source_catalog_closure(
                connection,
                write_tracker=tracker,
            )
            tracker.before_commit(0)

    _report_validation_progress(
        validation_progress_callback,
        mode="strict",
        completed=0,
        total=2,
        validation_scope="materialization_index_closure",
        workers=workers,
    )
    if workers == 1:
        validate_relations()
        _report_validation_progress(
            validation_progress_callback,
            mode="strict",
            completed=1,
            total=2,
            validation_scope="materialization_index_closure",
            workers=workers,
            task="relation_closure",
        )
        validate_sources()
    else:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="wdc-index-validation",
        ) as executor:
            relation_future = executor.submit(validate_relations)
            source_future = executor.submit(validate_sources)
            relation_future.result()
            _report_validation_progress(
                validation_progress_callback,
                mode="strict",
                completed=1,
                total=2,
                validation_scope="materialization_index_closure",
                workers=workers,
                task="relation_closure",
            )
            source_future.result()
    _report_validation_progress(
        validation_progress_callback,
        mode="strict",
        completed=2,
        total=2,
        validation_scope="materialization_index_closure",
        workers=workers,
        task="source_catalog_closure",
    )


def _materialization_review_policy(extractor: Any | None) -> str:
    if not join_builder.auto_check_required(extractor):
        return "disabled"
    return join_builder.model_auto_check_review_policy(extractor)


def _parameter_payload(
    args: argparse.Namespace,
    *,
    review_policy: str,
) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "split_by": str(args.split_by),
        "train_ratio": float(args.train_ratio),
        "dev_ratio": float(args.dev_ratio),
        "test_ratio": float(args.test_ratio),
        "query_rows_per_table": int(
            join_builder.configured_query_rows_per_table(args)
        ),
        "max_train_query_row_views_per_join": int(
            join_builder.configured_max_train_query_row_views_per_join(args)
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
        "explicit_join_fallback_mode": (
            join_builder.configured_explicit_join_fallback_mode(args)
        ),
        "explicit_join_fallback_ratio": float(
            join_builder.configured_explicit_join_fallback_ratio(args)
        ),
        "explicit_join_column_policy": (
            "seeded_random_non_entity_visible_column"
        ),
        "query_row_selection": "recovery_balanced_disjoint_train_views",
        "evaluation_query_row_views_per_join": 1,
        "target_row_scope": "all_source_rows",
        "qualified_attribute_policy": "all_safe_variants",
        "sibling_source_column_policy": (
            "globally_disjoint_query_and_target_sides"
        ),
        "identical_visible_query_policy": (
            "keep_best_recovery_single_target"
        ),
        "auto_check_schema_version": (
            join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION
        ),
        "auto_check_review_policy": review_policy,
        "reparse_cached_model_outputs": bool(
            args.reparse_cached_model_outputs
        ),
        "refresh_invalid_model_cache": bool(
            args.refresh_invalid_model_cache
        ),
        "unrecoverable_replacement_rounds": int(
            getattr(args, "unrecoverable_replacement_rounds", 0)
        ),
        "unrecoverable_drop_probability": float(
            getattr(args, "unrecoverable_drop_probability", 0.5)
        ),
        "recovery_replacement_round_index": int(
            getattr(args, "recovery_replacement_round_index", 0)
        ),
    }


def _parameter_fingerprint(
    args: argparse.Namespace,
    *,
    review_policy: str,
) -> str:
    return stable_hash(
        MATERIALIZATION_SCHEMA_VERSION,
        _canonical_json(
            _parameter_payload(args, review_policy=review_policy)
        ),
        length=40,
    )


def _certificate_config_fingerprint(
    args: argparse.Namespace,
    records_per_shard: int,
    *,
    review_policy: str,
) -> str:
    return stable_hash(
        UPSTREAM_CERTIFICATE_SCHEMA_VERSION,
        _parameter_fingerprint(args, review_policy=review_policy),
        int(records_per_shard),
        length=40,
    )


def _certificate_path(
    work_root: Path,
    config_fingerprint: str,
) -> Path:
    return (
        Path(work_root)
        / "materialization"
        / f"upstream-certificate-{config_fingerprint}.json"
    )


def _materialization_database_path(
    work_root: Path,
    upstream_identity: str,
    parameter_fingerprint: str,
) -> Path:
    return (
        Path(work_root)
        / "materialization"
        / (
            f"index-{upstream_identity}-"
            f"{parameter_fingerprint}.sqlite3"
        )
    )


def _certificate_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, dict):
        return {
            str(key): _certificate_jsonable(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_certificate_jsonable(item) for item in value]
    return value


def _materialization_input_descriptor(
    inputs: MaterializationInputs,
) -> str:
    return stable_hash(
        UPSTREAM_CERTIFICATE_SCHEMA_VERSION,
        _canonical_json(_certificate_jsonable(asdict(inputs))),
        length=40,
    )


def _certificate_manifest_paths(
    inputs: MaterializationInputs,
) -> tuple[Path, ...]:
    paths = {
        *(Path(path).resolve() for path in inputs.structural_manifests),
        Path(inputs.finalized_selection_manifest).resolve(),
        Path(inputs.asset_plan_result.manifest_path).resolve(),
        Path(inputs.unique_image_jobs.manifest_path).resolve(),
        Path(inputs.image_fetch_result.fetch_manifest_path).resolve(),
        Path(inputs.materialized_assets.manifest_path).resolve(),
        Path(inputs.adapted_model_tasks.manifest_path).resolve(),
        Path(inputs.model_result.manifest_path).resolve(),
    }
    if inputs.sampling_manifest is not None:
        paths.add(Path(inputs.sampling_manifest).resolve())
    if inputs.upstream_stage_registry is not None:
        paths.add(Path(inputs.upstream_stage_registry).resolve())
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _certificate_declared_shard_paths(
    inputs: MaterializationInputs,
) -> tuple[Path, ...]:
    declared: set[Path] = set()
    manifests_and_roots: list[tuple[Path, Path, bool]] = [
        *(
            (
                Path(path),
                Path(inputs.structural_output_root).resolve(),
                inputs.sampling_manifest is not None,
            )
            for path in inputs.structural_manifests
        ),
        (
            Path(inputs.finalized_selection_manifest),
            Path(inputs.structural_output_root).resolve(),
            False,
        ),
    ]
    if inputs.sampling_manifest is not None:
        sampling_manifest = Path(inputs.sampling_manifest)
        manifests_and_roots.append(
            (
                sampling_manifest,
                sampling_manifest.parent.resolve(),
                False,
            )
        )
    for manifest_path, root, compact_structural in manifests_and_roots:
        payload = _manifest_payload(manifest_path)
        for item in payload.get("completed_shards") or []:
            try:
                relative = Path(str(item["path"]))
            except (KeyError, TypeError, ValueError) as error:
                raise _CertificateMismatch(
                    "certificate manifest shard declaration is invalid"
                ) from error
            if compact_structural and relative.parts[0] in {
                "entities",
                "page_refs",
                "direct_image_refs",
                "selection",
            }:
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise _CertificateMismatch(
                    "certificate manifest shard escapes its root"
                )
            declared.add(path)
    return tuple(sorted(declared, key=lambda path: path.as_posix()))


def _certificate_input_paths(
    inputs: MaterializationInputs,
    upstream: _ValidatedUpstream | None = None,
) -> tuple[Path, ...]:
    page_result = inputs.page_fetch_result
    image_result = inputs.image_fetch_result
    paths = {
        *_certificate_manifest_paths(inputs),
        *(Path(path).resolve() for path in inputs.sampled_entity_paths),
        *(Path(path).resolve() for path in inputs.sampled_page_ref_paths),
        *(
            Path(path).resolve()
            for path in inputs.asset_plan_result.entity_plan_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.asset_plan_result.image_mapping_paths
        ),
        Path(inputs.unique_image_jobs.output_path).resolve(),
        Path(page_result.outcomes_path).resolve(),
        Path(page_result.failure_path).resolve(),
        Path(page_result.progress_path).resolve(),
        Path(page_result.job_store_path).resolve(),
        Path(image_result.outcomes_path).resolve(),
        Path(image_result.job_store_path).resolve(),
        *(
            Path(path).resolve()
            for path in inputs.materialized_assets.bridge_asset_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.materialized_assets.table_asset_link_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.adapted_model_tasks.task_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.adapted_model_tasks.error_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.model_result.extraction_paths
        ),
        *(
            Path(path).resolve()
            for path in inputs.model_result.error_paths
        ),
        Path(inputs.model_result.jobset.database_path).resolve(),
    }
    if upstream is not None:
        paths.update(_certificate_declared_shard_paths(inputs))
        paths.update(
            Path(path).resolve()
            for path in (
                *upstream.source_paths,
                *upstream.entity_paths,
                *upstream.page_ref_paths,
                *upstream.structural_failure_paths,
                upstream.page_failure_path,
                *upstream.asset_paths,
                *upstream.link_paths,
                *upstream.extraction_paths,
                *upstream.model_error_paths,
                *upstream.adapter_error_paths,
            )
        )
    for candidate in tuple(paths):
        if candidate.suffix not in {".db", ".sqlite", ".sqlite3"}:
            continue
        # A non-empty WAL can contain durable, uncheckpointed records and is
        # therefore part of the authoritative input.  SQLite readers may
        # create an empty WAL together with transient SHM coordination state;
        # neither sidecar represents a logical input change in that case.
        wal_path = Path(f"{candidate}-wal")
        if wal_path.is_file() and wal_path.stat().st_size > 0:
            paths.add(wal_path.resolve())
    return tuple(sorted(paths, key=lambda path: path.as_posix()))


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise _CertificateMismatch(
            f"certificate input is not a file: {resolved}"
        )
    stat = resolved.stat()
    return {
        "path": resolved.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def _file_identities(paths: Iterable[Path]) -> list[dict[str, Any]]:
    try:
        return [_file_identity(path) for path in paths]
    except OSError as error:
        raise _CertificateMismatch(
            f"certificate input identity is unavailable: {error}"
        ) from error


def _manifest_hashes(paths: Iterable[Path]) -> list[dict[str, str]]:
    try:
        return [
            {
                "path": Path(path).resolve(strict=True).as_posix(),
                "sha256": _sha256_path(Path(path).resolve(strict=True)),
            }
            for path in paths
        ]
    except OSError as error:
        raise _CertificateMismatch(
            f"certificate manifest is unavailable: {error}"
        ) from error


def _serialized_upstream(
    upstream: _ValidatedUpstream,
) -> dict[str, Any]:
    return {
        "source_paths": [
            path.resolve().as_posix() for path in upstream.source_paths
        ],
        "entity_paths": [
            path.resolve().as_posix() for path in upstream.entity_paths
        ],
        "page_ref_paths": [
            path.resolve().as_posix() for path in upstream.page_ref_paths
        ],
        "structural_failure_paths": [
            path.resolve().as_posix()
            for path in upstream.structural_failure_paths
        ],
        "page_failure_path": upstream.page_failure_path.resolve().as_posix(),
        "asset_paths": [
            path.resolve().as_posix() for path in upstream.asset_paths
        ],
        "link_paths": [
            path.resolve().as_posix() for path in upstream.link_paths
        ],
        "extraction_paths": [
            path.resolve().as_posix() for path in upstream.extraction_paths
        ],
        "model_error_paths": [
            path.resolve().as_posix() for path in upstream.model_error_paths
        ],
        "adapter_error_paths": [
            path.resolve().as_posix()
            for path in upstream.adapter_error_paths
        ],
        "image_failure_aggregation_database": (
            upstream.image_failure_aggregation_database.resolve().as_posix()
        ),
        "expected_tables": upstream.expected_tables,
        "expected_entities": upstream.expected_entities,
        "expected_assets": upstream.expected_assets,
        "expected_links": upstream.expected_links,
        "expected_extractions": upstream.expected_extractions,
        "identity": upstream.identity,
        "provenance": upstream.provenance,
    }


def _upstream_from_certificate(
    payload: dict[str, Any],
    inputs: MaterializationInputs,
) -> _ValidatedUpstream:
    try:
        upstream = payload["upstream"]
        if not isinstance(upstream, dict):
            raise TypeError("upstream payload is not an object")

        def paths(name: str) -> tuple[Path, ...]:
            values = upstream[name]
            if not isinstance(values, list):
                raise TypeError(f"{name} is not a list")
            return tuple(Path(str(value)) for value in values)

        provenance = upstream["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("provenance is not an object")
        return _ValidatedUpstream(
            source_paths=paths("source_paths"),
            entity_paths=paths("entity_paths"),
            page_ref_paths=paths("page_ref_paths"),
            structural_failure_paths=paths(
                "structural_failure_paths"
            ),
            page_failure_path=Path(
                str(upstream["page_failure_path"])
            ),
            asset_paths=paths("asset_paths"),
            link_paths=paths("link_paths"),
            extraction_paths=paths("extraction_paths"),
            model_error_paths=paths("model_error_paths"),
            adapter_error_paths=paths("adapter_error_paths"),
            asset_plan_result=inputs.asset_plan_result,
            image_fetch_result=inputs.image_fetch_result,
            image_failure_aggregation_database=Path(
                str(upstream["image_failure_aggregation_database"])
            ),
            expected_tables=int(upstream["expected_tables"]),
            expected_entities=int(upstream["expected_entities"]),
            expected_assets=int(upstream["expected_assets"]),
            expected_links=int(upstream["expected_links"]),
            expected_extractions=int(upstream["expected_extractions"]),
            identity=str(upstream["identity"]),
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _CertificateMismatch(
            "upstream certificate payload is invalid"
        ) from error


def _certificate_digest(payload: dict[str, Any]) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "certificate_sha256"
    }
    return hashlib.sha256(
        _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()


def _validate_resume_index_units(
    connection: sqlite3.Connection,
    *,
    expected_tables: int,
) -> int:
    catalog = connection.execute(
        """
        SELECT COUNT(*) AS total,
               COUNT(DISTINCT ordinal) AS ordinals,
               SUM(CASE WHEN split IN ('train', 'dev', 'test')
                        THEN 0 ELSE 1 END) AS invalid_splits
        FROM source_catalog
        """
    ).fetchone()
    if (
        int(catalog["total"]) != expected_tables
        or int(catalog["ordinals"]) != expected_tables
        or int(catalog["invalid_splits"] or 0) != 0
    ):
        raise _CertificateMismatch(
            "materialization source catalog resume mismatch"
        )
    invalid_units = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM source_units AS units
            LEFT JOIN source_catalog AS catalog
              ON catalog.source_table_id = units.source_table_id
            WHERE units.complete != 1
               OR catalog.source_table_id IS NULL
               OR units.source_sha256 != catalog.record_sha256
               OR units.split != catalog.split
            """
        ).fetchone()[0]
    )
    completed = int(
        connection.execute(
            "SELECT COUNT(*) FROM source_units"
        ).fetchone()[0]
    )
    invalid_count_json = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM source_units
            WHERE CASE
                WHEN json_valid(counts_json)
                THEN json_type(counts_json) != 'object'
                ELSE 1
            END
            """
        ).fetchone()[0]
    )
    if (
        invalid_units
        or invalid_count_json
        or completed > expected_tables
    ):
        raise _CertificateMismatch(
            "materialization source unit resume mismatch"
        )
    return completed


def _resume_index_source_units(
    database_path: Path,
    *,
    upstream: _ValidatedUpstream,
    certificate_sha256: str,
    config_fingerprint: str,
) -> int:
    if not database_path.is_file():
        raise _CertificateMismatch(
            "materialization resume index is missing"
        )
    try:
        with _connect(database_path) as connection:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    """
                    SELECT key, value FROM metadata
                    WHERE key IN (
                        'upstream_identity',
                        'upstream_certificate_sha256',
                        'upstream_certificate_config',
                        'upstream_certificate_ready'
                    )
                    """
                )
            }
            if metadata != {
                "upstream_identity": upstream.identity,
                "upstream_certificate_sha256": certificate_sha256,
                "upstream_certificate_config": config_fingerprint,
                "upstream_certificate_ready": "1",
            }:
                raise _CertificateMismatch(
                    "materialization resume index certificate mismatch"
                )
            completed = _validate_resume_index_units(
                connection,
                expected_tables=upstream.expected_tables,
            )
    except sqlite3.Error as error:
        raise _CertificateMismatch(
            f"materialization resume index is unreadable: {error}"
        ) from error
    return completed


def _load_fast_resume_state(
    inputs: MaterializationInputs,
    *,
    args: argparse.Namespace,
    records_per_shard: int,
    review_policy: str,
) -> _FastResumeState:
    parameter_fingerprint = _parameter_fingerprint(
        args,
        review_policy=review_policy,
    )
    config_fingerprint = _certificate_config_fingerprint(
        args,
        records_per_shard,
        review_policy=review_policy,
    )
    path = _certificate_path(inputs.work_root, config_fingerprint)
    if not path.is_file():
        raise _CertificateMismatch("upstream certificate is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _CertificateMismatch(
            "upstream certificate is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise _CertificateMismatch(
            "upstream certificate is not an object"
        )
    actual_digest = _certificate_digest(payload)
    if (
        payload.get("schema_version")
        != UPSTREAM_CERTIFICATE_SCHEMA_VERSION
        or payload.get("complete") is not True
        or payload.get("config_fingerprint") != config_fingerprint
        or payload.get("input_descriptor_fingerprint")
        != _materialization_input_descriptor(inputs)
        or payload.get("certificate_sha256") != actual_digest
    ):
        raise _CertificateMismatch(
            "upstream certificate identity mismatch"
        )
    current_manifests = _manifest_hashes(
        _certificate_manifest_paths(inputs)
    )
    if payload.get("manifest_hashes") != current_manifests:
        raise _CertificateMismatch(
            "upstream certificate manifest hash mismatch"
        )
    upstream = _upstream_from_certificate(payload, inputs)
    current_files = _file_identities(
        _certificate_input_paths(inputs, upstream)
    )
    if payload.get("input_files") != current_files:
        raise _CertificateMismatch(
            "upstream certificate input file identity mismatch"
        )
    database_path = _materialization_database_path(
        inputs.work_root,
        upstream.identity,
        parameter_fingerprint,
    )
    completed = _resume_index_source_units(
        database_path,
        upstream=upstream,
        certificate_sha256=actual_digest,
        config_fingerprint=config_fingerprint,
    )
    return _FastResumeState(
        upstream=upstream,
        database_path=database_path,
        resumed_source_units=completed,
    )


def _persist_upstream_certificate(
    inputs: MaterializationInputs,
    upstream: _ValidatedUpstream,
    *,
    args: argparse.Namespace,
    records_per_shard: int,
    review_policy: str,
    database_path: Path,
    manifest_hashes: list[dict[str, str]],
    input_files: list[dict[str, Any]],
    pre_write_guard: PreWriteGuard | None = None,
) -> Path:
    config_fingerprint = _certificate_config_fingerprint(
        args,
        records_per_shard,
        review_policy=review_policy,
    )
    payload = {
        "schema_version": UPSTREAM_CERTIFICATE_SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint,
        "input_descriptor_fingerprint": (
            _materialization_input_descriptor(inputs)
        ),
        "inputs": _certificate_jsonable(asdict(inputs)),
        "manifest_hashes": manifest_hashes,
        "input_files": input_files,
        "upstream": _serialized_upstream(upstream),
        "complete": True,
    }
    payload["certificate_sha256"] = _certificate_digest(payload)
    path = _certificate_path(inputs.work_root, config_fingerprint)
    _atomic_json(
        path,
        payload,
        pre_write_guard=pre_write_guard,
    )
    tracker = GuardedWriteTracker(database_path, pre_write_guard)
    tracker.before_write(16 * 1024)
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            """
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
            """,
            (
                (
                    "upstream_certificate_sha256",
                    payload["certificate_sha256"],
                ),
                ("upstream_certificate_config", config_fingerprint),
                ("upstream_certificate_ready", "1"),
            ),
        )
        tracker.before_commit(0)
        connection.commit()
    return path


def _inputs_from_certificate_payload(
    payload: dict[str, Any],
) -> MaterializationInputs:
    try:
        raw = payload["inputs"]
        if not isinstance(raw, dict):
            raise TypeError("inputs snapshot is not an object")

        def path(value: Any) -> Path:
            return Path(str(value))

        def path_tuple(values: Any) -> tuple[Path, ...]:
            if not isinstance(values, list):
                raise TypeError("path collection is not a list")
            return tuple(path(value) for value in values)

        barrier_raw = raw["structural_barrier"]
        barrier = StructuralStageBarrier(
            schema_version=str(barrier_raw["schema_version"]),
            manifest_count=int(barrier_raw["manifest_count"]),
            manifest_sha256=dict(barrier_raw["manifest_sha256"]),
            input_fingerprints=dict(
                barrier_raw["input_fingerprints"]
            ),
            parameter_fingerprints=dict(
                barrier_raw["parameter_fingerprints"]
            ),
            final_manifest_sha256=str(
                barrier_raw["final_manifest_sha256"]
            ),
            final_selection=dict(barrier_raw["final_selection"]),
        )
        page_raw = raw["page_fetch_result"]
        page_result = FetchResult(
            unique=int(page_raw["unique"]),
            success=int(page_raw["success"]),
            terminal=int(page_raw["terminal"]),
            inflight=int(page_raw["inflight"]),
            leased=int(page_raw["leased"]),
            remaining=int(page_raw["remaining"]),
            complete=page_raw["complete"] is True,
            maximum_inflight=int(page_raw["maximum_inflight"]),
            maximum_claimed=int(page_raw["maximum_claimed"]),
            maximum_host_limiters=int(
                page_raw["maximum_host_limiters"]
            ),
            outcomes_path=path(page_raw["outcomes_path"]),
            failure_path=path(page_raw["failure_path"]),
            progress_path=path(page_raw["progress_path"]),
            policy_fingerprint=str(page_raw["policy_fingerprint"]),
            job_store_path=path(page_raw["job_store_path"]),
            job_kind=str(page_raw["job_kind"]),
            transport_attempt_summary=dict(
                page_raw["transport_attempt_summary"]
            ),
        )
        plan_raw = raw["asset_plan_result"]
        planned = AssetPlanShards(
            output_root=path(plan_raw["output_root"]),
            entity_plan_paths=path_tuple(
                plan_raw["entity_plan_paths"]
            ),
            image_mapping_paths=path_tuple(
                plan_raw["image_mapping_paths"]
            ),
            manifest_path=path(plan_raw["manifest_path"]),
            entities=int(plan_raw["entities"]),
            image_mappings=int(plan_raw["image_mappings"]),
        )
        jobs_raw = raw["unique_image_jobs"]
        shard_raw = jobs_raw["completed_shard"]
        unique_jobs = UniqueImageJobs(
            output_path=path(jobs_raw["output_path"]),
            manifest_path=path(jobs_raw["manifest_path"]),
            completed_shard=CompletedShard(
                path=str(shard_raw["path"]),
                records=int(shard_raw["records"]),
                bytes=int(shard_raw["bytes"]),
                sha256=str(shard_raw["sha256"]),
            ),
            input_fingerprint=str(jobs_raw["input_fingerprint"]),
            parameter_fingerprint=str(
                jobs_raw["parameter_fingerprint"]
            ),
            complete=jobs_raw["complete"] is True,
        )
        image_raw = raw["image_fetch_result"]
        if image_raw["unique_jobs"] != _certificate_jsonable(
            asdict(unique_jobs)
        ):
            raise ValueError("image unique jobs snapshot mismatch")
        image_result = ImageFetchResult(
            unique=int(image_raw["unique"]),
            success=int(image_raw["success"]),
            terminal=int(image_raw["terminal"]),
            complete=image_raw["complete"] is True,
            outcomes_path=path(image_raw["outcomes_path"]),
            policy_fingerprint=str(image_raw["policy_fingerprint"]),
            maximum_inflight=int(image_raw["maximum_inflight"]),
            maximum_claimed=int(image_raw["maximum_claimed"]),
            maximum_host_states=int(image_raw["maximum_host_states"]),
            unique_jobs=unique_jobs,
            job_store_path=path(image_raw["job_store_path"]),
            job_kind=str(image_raw["job_kind"]),
            fetch_manifest_path=path(
                image_raw["fetch_manifest_path"]
            ),
            fetch_manifest_sha256=str(
                image_raw["fetch_manifest_sha256"]
            ),
            outcome_digest=str(image_raw["outcome_digest"]),
            outcomes_count=int(image_raw["outcomes_count"]),
            outcome_url_key_digest=str(
                image_raw["outcome_url_key_digest"]
            ),
            leased=int(image_raw["leased"]),
            remaining=int(image_raw["remaining"]),
            transport_attempt_summary=dict(
                image_raw["transport_attempt_summary"]
            ),
        )
        assets_raw = raw["materialized_assets"]
        materialized_assets = MaterializedAssetShards(
            output_root=path(assets_raw["output_root"]),
            bridge_asset_paths=path_tuple(
                assets_raw["bridge_asset_paths"]
            ),
            table_asset_link_paths=path_tuple(
                assets_raw["table_asset_link_paths"]
            ),
            manifest_path=path(assets_raw["manifest_path"]),
            bridge_assets=int(assets_raw["bridge_assets"]),
            table_asset_links=int(
                assets_raw["table_asset_links"]
            ),
        )
        adapted_raw = raw["adapted_model_tasks"]
        adapted = AdaptedModelTasks(
            output_root=path(adapted_raw["output_root"]),
            task_paths=path_tuple(adapted_raw["task_paths"]),
            error_paths=path_tuple(adapted_raw["error_paths"]),
            manifest_path=path(adapted_raw["manifest_path"]),
            input_fingerprint=str(
                adapted_raw["input_fingerprint"]
            ),
            tasks=int(adapted_raw["tasks"]),
            errors=int(adapted_raw["errors"]),
        )
        model_raw = raw["model_result"]
        jobset_raw = model_raw["jobset"]
        jobs = tuple(
            ModelJobInfo(
                job_id=str(item["job_id"]),
                cache_key=str(item["cache_key"]),
                model_call_key=str(item["model_call_key"]),
                modality=str(item["modality"]),
                asset_fingerprint=str(item["asset_fingerprint"]),
                entity_prompt_fingerprint=str(
                    item["entity_prompt_fingerprint"]
                ),
            )
            for item in jobset_raw.get("jobs") or []
        )
        jobset = ModelJobSet(
            database_path=path(jobset_raw["database_path"]),
            input_fingerprint=str(jobset_raw["input_fingerprint"]),
            prompt_version=str(jobset_raw["prompt_version"]),
            text_fingerprint=str(jobset_raw["text_fingerprint"]),
            image_fingerprint=str(jobset_raw["image_fingerprint"]),
            text_kind=str(jobset_raw["text_kind"]),
            image_kind=str(jobset_raw["image_kind"]),
            text_tasks=int(jobset_raw["text_tasks"]),
            image_tasks=int(jobset_raw["image_tasks"]),
            jobs=jobs,
        )
        model_result = ModelStageResult(
            output_root=path(model_raw["output_root"]),
            manifest_path=path(model_raw["manifest_path"]),
            extraction_paths=path_tuple(
                model_raw["extraction_paths"]
            ),
            error_paths=path_tuple(model_raw["error_paths"]),
            text_total=int(model_raw["text_total"]),
            image_total=int(model_raw["image_total"]),
            success=int(model_raw["success"]),
            terminal=int(model_raw["terminal"]),
            pending=int(model_raw["pending"]),
            leased=int(model_raw["leased"]),
            complete=model_raw["complete"] is True,
            jobset=jobset,
        )
        authority_raw = raw["model_authority"]
        authority = ModelStageAuthority(
            text_model_identity=str(
                authority_raw["text_model_identity"]
            ),
            image_model_identity=str(
                authority_raw["image_model_identity"]
            ),
            prompt_version=str(authority_raw["prompt_version"]),
            policy_fingerprint=str(
                authority_raw["policy_fingerprint"]
            ),
            parser_schema_version=str(
                authority_raw["parser_schema_version"]
            ),
        )
        return MaterializationInputs(
            structural_output_root=path(
                raw["structural_output_root"]
            ),
            structural_manifests=path_tuple(
                raw["structural_manifests"]
            ),
            finalized_selection_manifest=path(
                raw["finalized_selection_manifest"]
            ),
            structural_barrier=barrier,
            page_fetch_result=page_result,
            asset_plan_result=planned,
            unique_image_jobs=unique_jobs,
            image_fetch_result=image_result,
            materialized_assets=materialized_assets,
            adapted_model_tasks=adapted,
            model_result=model_result,
            model_authority=authority,
            work_root=path(raw["work_root"]),
            sampling_manifest=(
                path(raw["sampling_manifest"])
                if raw.get("sampling_manifest") is not None
                else None
            ),
            sampled_entity_paths=path_tuple(
                raw.get("sampled_entity_paths") or []
            ),
            sampled_page_ref_paths=path_tuple(
                raw.get("sampled_page_ref_paths") or []
            ),
            upstream_stage_registry=(
                path(raw["upstream_stage_registry"])
                if raw.get("upstream_stage_registry") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise _CertificateMismatch(
            "upstream certificate input snapshot is invalid"
        ) from error


def load_certified_materialization_inputs(
    work_root: Path,
    *,
    args: argparse.Namespace,
    records_per_shard: int,
    extractor: Any | None = None,
) -> MaterializationInputs:
    """Load inputs only after the certificate and resume index verify."""
    review_policy = _materialization_review_policy(extractor)
    config_fingerprint = _certificate_config_fingerprint(
        args,
        records_per_shard,
        review_policy=review_policy,
    )
    certificate = _certificate_path(work_root, config_fingerprint)
    try:
        payload = json.loads(certificate.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise _CertificateMismatch(
            "upstream certificate is unavailable"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("certificate_sha256")
        != _certificate_digest(payload)
        or payload.get("config_fingerprint") != config_fingerprint
    ):
        raise _CertificateMismatch(
            "upstream certificate identity mismatch"
        )
    inputs = _inputs_from_certificate_payload(payload)
    if Path(inputs.work_root).resolve() != Path(work_root).resolve():
        raise _CertificateMismatch(
            "upstream certificate work root mismatch"
        )
    if inputs.model_authority != ModelStageAuthority.current(args):
        raise _CertificateMismatch(
            "upstream certificate model authority mismatch"
        )
    _load_fast_resume_state(
        inputs,
        args=args,
        records_per_shard=records_per_shard,
        review_policy=review_policy,
    )
    return inputs


def _split_group(
    source_table: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    source_table_id = str(source_table["source_table_id"])
    if args.split_by == "page_title":
        return clean_text(source_table.get("page_title")) or source_table_id
    return source_table_id


_SOURCE_CATALOG_INSERT_SQL = """
    INSERT INTO source_catalog (
        source_table_id, ordinal, page_title,
        split_group, record_sha256, record_json, record_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def _sqlite_parameter_summary(values: tuple[Any, ...]) -> str:
    parts = []
    for index, value in enumerate(values):
        value_type = f"{type(value).__module__}.{type(value).__name__}"
        try:
            size = len(value)
        except TypeError:
            parts.append(f"{index}:{value_type}")
        else:
            parts.append(f"{index}:{value_type}[{size}]")
    return ",".join(parts)


def _insert_source_catalog_record(
    connection: sqlite3.Connection,
    values: tuple[Any, ...],
    *,
    ordinal: int,
    source_table_id: str,
) -> None:
    try:
        connection.execute(_SOURCE_CATALOG_INSERT_SQL, values)
        return
    except sqlite3.InterfaceError as error:
        logging.warning(
            "Transient source catalog SQLite binding failure; retrying once: "
            "ordinal=%d source_table_id=%s parameters=%s error=%s",
            ordinal,
            source_table_id,
            _sqlite_parameter_summary(values),
            error,
        )
    try:
        connection.execute(_SOURCE_CATALOG_INSERT_SQL, values)
    except sqlite3.InterfaceError as error:
        error_code = getattr(error, "sqlite_errorcode", None)
        error_name = getattr(error, "sqlite_errorname", None)
        raise sqlite3.InterfaceError(
            "source catalog SQLite binding failed after one retry: "
            f"ordinal={ordinal} source_table_id={source_table_id} "
            f"parameters={_sqlite_parameter_summary(values)} "
            f"sqlite_errorcode={error_code!r} "
            f"sqlite_errorname={error_name!r}"
        ) from error


def _catalog_source_records(
    database_path: Path,
    source_tables: Iterable[dict[str, Any]],
    *,
    args: argparse.Namespace,
    expected_tables: int,
    write_tracker: GuardedWriteTracker | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    _report_work_progress(
        progress_callback,
        phase="catalog_sources",
        completed=0,
        total=expected_tables,
    )
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        observed_stream = 0
        for ordinal, source_table in enumerate(source_tables):
            observed_stream += 1
            source_table_id = clean_text(
                source_table.get("source_table_id")
            )
            if not source_table_id:
                raise ValueError("source table is missing source_table_id")
            encoded = _canonical_json(source_table)
            encoded_size, digest = _json_text_identity(encoded)
            if write_tracker is not None:
                write_tracker.before_write(
                    4096 + 2 * encoded_size
                )
            stored_json, record_path = _stored_json_values(
                database_path,
                namespace="source-catalog",
                identity=source_table_id,
                record=source_table,
                encoded=encoded,
                encoded_size=encoded_size,
                digest=digest,
            )
            existing = connection.execute(
                """
                SELECT ordinal, page_title, split_group,
                       record_sha256, record_json, record_path
                FROM source_catalog WHERE source_table_id = ?
                """,
                (source_table_id,),
            ).fetchone()
            expected_values = (
                ordinal,
                clean_text(source_table.get("page_title")),
                _split_group(source_table, args),
                digest,
                stored_json,
                record_path,
            )
            if existing is not None:
                actual_identity = (
                    int(existing["ordinal"]),
                    str(existing["page_title"]),
                    str(existing["split_group"]),
                    str(existing["record_sha256"]),
                )
                if actual_identity != expected_values[:4]:
                    raise ValueError(
                        f"conflicting source table ID: {source_table_id}"
                    )
                existing_json = str(existing["record_json"])
                existing_path = str(existing["record_path"])
                if not existing_path and record_path:
                    if existing_json != encoded:
                        raise ValueError(
                            "conflicting source table ID: "
                            f"{source_table_id}"
                        )
                    connection.execute(
                        """
                        UPDATE source_catalog
                        SET record_json = ?, record_path = ?
                        WHERE source_table_id = ?
                        """,
                        (
                            stored_json,
                            record_path,
                            source_table_id,
                        ),
                    )
                elif existing_path and record_path:
                    if (
                        existing_json != stored_json
                        or existing_path != record_path
                    ):
                        connection.execute(
                            """
                            UPDATE source_catalog
                            SET record_json = ?, record_path = ?
                            WHERE source_table_id = ?
                            """,
                            (
                                stored_json,
                                record_path,
                                source_table_id,
                            ),
                        )
                elif existing_path:
                    # Keep a valid external representation when a later
                    # storage policy would otherwise inline the same digest.
                    pass
                elif (
                    existing_json != stored_json
                    or existing_path != record_path
                ):
                    raise ValueError(
                        f"conflicting source table ID: {source_table_id}"
                    )
            else:
                try:
                    _insert_source_catalog_record(
                        connection,
                        (source_table_id, *expected_values),
                        ordinal=ordinal,
                        source_table_id=source_table_id,
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        f"duplicate source table ID: {source_table_id}"
                    ) from error
            if (ordinal + 1) % 10 == 0:
                if write_tracker is not None:
                    write_tracker.before_commit(0)
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
            if (ordinal + 1) % 100 == 0:
                _report_work_progress(
                    progress_callback,
                    phase="catalog_sources",
                    completed=ordinal + 1,
                    total=expected_tables,
                )
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
        if write_tracker is not None:
            write_tracker.before_commit(0)
        connection.commit()
    _report_work_progress(
        progress_callback,
        phase="catalog_sources",
        completed=expected_tables,
        total=expected_tables,
    )


def _catalog_sources(
    database_path: Path,
    source_paths: Iterable[Path],
    *,
    args: argparse.Namespace,
    expected_tables: int,
    pre_write_guard: PreWriteGuard | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    _catalog_source_records(
        database_path,
        _iter_jsonl(source_paths),
        args=args,
        expected_tables=expected_tables,
        write_tracker=write_tracker,
        progress_callback=progress_callback,
    )


def _assign_splits(
    database_path: Path,
    args: argparse.Namespace,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
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
            write_tracker.before_write(
                4096 + 2 * len(group.encode("utf-8"))
            )
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
            write_tracker.before_write(8192)
            connection.execute(
                "UPDATE split_groups SET split = ? WHERE split_group = ?",
                (split, group),
            )
            connection.execute(
                "UPDATE source_catalog SET split = ? WHERE split_group = ?",
                (split, group),
            )
        write_tracker.before_commit(0)
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
        ordered_asset_ids = list(dict.fromkeys(asset_ids))
        assets_by_id: dict[str, dict[str, Any]] = {}
        extractions_by_asset: dict[
            str, list[tuple[str, dict[str, Any]]]
        ] = {}
        for offset in range(
            0,
            len(ordered_asset_ids),
            _SQLITE_IN_BATCH_RECORDS,
        ):
            asset_batch = ordered_asset_ids[
                offset : offset + _SQLITE_IN_BATCH_RECORDS
            ]
            placeholders = ",".join("?" for _ in asset_batch)
            for asset_row in connection.execute(
                f"""
                SELECT asset_id, record_json
                FROM assets
                WHERE asset_id IN ({placeholders})
                """,
                tuple(asset_batch),
            ):
                assets_by_id[str(asset_row["asset_id"])] = json.loads(
                    str(asset_row["record_json"])
                )
            for extraction_row in connection.execute(
                f"""
                SELECT asset_id, cache_key, record_json
                FROM extractions
                WHERE asset_id IN ({placeholders})
                ORDER BY asset_id, cache_key
                """,
                tuple(asset_batch),
            ):
                asset_id = str(extraction_row["asset_id"])
                extractions_by_asset.setdefault(asset_id, []).append(
                    (
                        str(extraction_row["cache_key"]),
                        json.loads(str(extraction_row["record_json"])),
                    )
                )

        assets = [
            assets_by_id[asset_id]
            for asset_id in ordered_asset_ids
            if asset_id in assets_by_id
        ]
        extractions = []
        seen_extractions: set[str] = set()
        for asset_id in ordered_asset_ids:
            if asset_id not in assets_by_id:
                continue
            for cache_key, extraction in extractions_by_asset.get(
                asset_id, []
            ):
                if cache_key in seen_extractions:
                    continue
                seen_extractions.add(cache_key)
                extractions.append(extraction)

        wiki_to_entity_id = {
            clean_text(entity.get("wiki_title")): str(entity["entity_id"])
            for entity in entities
            if clean_text(entity.get("wiki_title"))
        }
        source_titles = _source_wiki_titles(source_table)
        aliases_by_title = {
            title: (
                title,
                normalize_title(title),
                title.casefold(),
                normalize_title(title).casefold(),
            )
            for title in source_titles
        }
        ordered_aliases = list(
            dict.fromkeys(
                alias
                for aliases in aliases_by_title.values()
                for alias in aliases
            )
        )
        entity_id_by_alias: dict[str, str] = {}
        for offset in range(
            0,
            len(ordered_aliases),
            _SQLITE_IN_BATCH_RECORDS,
        ):
            alias_batch = ordered_aliases[
                offset : offset + _SQLITE_IN_BATCH_RECORDS
            ]
            placeholders = ",".join("?" for _ in alias_batch)
            entity_id_by_alias.update(
                {
                    str(row["alias"]): str(row["entity_id"])
                    for row in connection.execute(
                        f"""
                        SELECT alias, entity_id
                        FROM entity_aliases
                        WHERE alias IN ({placeholders})
                        """,
                        tuple(alias_batch),
                    )
                }
            )
        for title in source_titles:
            for alias in aliases_by_title[title]:
                entity_id = entity_id_by_alias.get(alias)
                if entity_id is not None:
                    wiki_to_entity_id[title] = entity_id
                    break
            else:
                wiki_to_entity_id[title] = (
                    "ent_" + stable_hash(title, length=16)
                )
    return entities, assets, links, extractions, wiki_to_entity_id


def _query_auto_check_records_for_extractions(
    database_path: Path,
    extraction_cache_keys: Iterable[str],
) -> list[dict[str, Any]]:
    keys = list(
        dict.fromkeys(
            clean_text(key) for key in extraction_cache_keys if clean_text(key)
        )
    )
    records: list[dict[str, Any]] = []
    with _connect(database_path) as connection:
        for offset in range(0, len(keys), _SQLITE_IN_BATCH_RECORDS):
            batch = keys[offset : offset + _SQLITE_IN_BATCH_RECORDS]
            placeholders = ",".join("?" for _ in batch)
            for row in connection.execute(
                f"""
                SELECT cache_key, record_json
                FROM query_auto_checks
                WHERE extraction_cache_key IN ({placeholders})
                ORDER BY cache_key
                """,
                tuple(batch),
            ):
                record = json.loads(str(row["record_json"]))
                record["cache_key"] = str(row["cache_key"])
                records.append(record)
    return records


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
    query_auto_check_required = bool(
        getattr(materialize_args, "_query_auto_check_required", False)
    )
    query_auto_check_records = (
        _query_auto_check_records_for_extractions(
            database_path,
            (
                clean_text(record.get("cache_key"))
                for record in extractions
            ),
        )
        if query_auto_check_required
        else []
    )
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
        extractor=(
            _CachedQueryAutoCheckExtractor(
                str(materialize_args._query_auto_check_review_policy)
            )
            if query_auto_check_required
            else None
        ),
        cache=_TableExtractionCache(extractions),
        progress=None,
        concurrency_state=join_builder.ModelConcurrencyState.from_args(
            materialize_args
        ),
        extraction_writer=extraction_sink,
        recovery_writer=recovery_sink,
        args=materialize_args,
        query_auto_check_cache=(
            _TableExtractionCache(query_auto_check_records)
            if query_auto_check_required
            else None
        ),
        finalize_query_recoveries=True,
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


def _load_materialization_source(
    database_path: Path,
    item: _MaterializationWorkItem,
) -> dict[str, Any]:
    with _connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT record_sha256, record_json, record_path
            FROM source_catalog
            WHERE source_table_id = ?
            """,
            (item.source_table_id,),
        ).fetchone()
    if row is None:
        raise ValueError(
            f"materialization source is missing: {item.source_table_id}"
        )
    if str(row["record_sha256"]) != item.source_sha256:
        raise ValueError(
            f"materialization source identity mismatch: "
            f"{item.source_table_id}"
        )
    source_table = _load_stored_json(
        database_path,
        row["record_json"],
        row["record_path"],
    )
    if clean_text(source_table.get("source_table_id")) != item.source_table_id:
        raise ValueError(
            f"materialization source ID mismatch: {item.source_table_id}"
        )
    if any(
        clean_text(column.get("column_name")).casefold() == "image"
        for column in source_table.get("columns", [])
    ):
        raise ValueError(
            f"source table retains forbidden image column: "
            f"{item.source_table_id}"
        )
    return source_table


def _materialize_work_item(
    database_path: Path,
    args: argparse.Namespace,
    item: _MaterializationWorkItem,
) -> MaterializedTable:
    source_table = _load_materialization_source(database_path, item)
    materialized = _materialize_from_index(
        source_table,
        database_path,
        args=args,
        split=item.split,
    )
    # The writer only needs the stable ID. Avoid sending a potentially very
    # large source table back through the process-pool result pipe.
    return replace(
        materialized,
        source_table={"source_table_id": item.source_table_id},
    )


_WORKER_DATABASE_PATH: Path | None = None
_WORKER_ARGS: argparse.Namespace | None = None


def _initialize_materialization_worker(
    database_path: Path,
    args: argparse.Namespace,
) -> None:
    global _WORKER_DATABASE_PATH, _WORKER_ARGS
    _WORKER_DATABASE_PATH = database_path
    _WORKER_ARGS = args


def _run_materialization_worker(
    item: _MaterializationWorkItem,
) -> MaterializedTable:
    if _WORKER_DATABASE_PATH is None or _WORKER_ARGS is None:
        raise RuntimeError("materialization worker is not initialized")
    return _materialize_work_item(
        _WORKER_DATABASE_PATH,
        _WORKER_ARGS,
        item,
    )


def materialize_dataset_shard(
    inputs: MaterializationShardInputs,
    *,
    args: argparse.Namespace,
    split: str,
    pre_write_guard: PreWriteGuard | None = None,
) -> MaterializedTable:
    """Materialize one source table using only its SQLite-selected records."""
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"invalid split: {split}")
    _build_index(inputs, pre_write_guard=pre_write_guard)
    catalog_tracker = GuardedWriteTracker(
        inputs.lookup_database,
        pre_write_guard,
    )
    _catalog_source_records(
        inputs.lookup_database,
        [inputs.source_table],
        args=args,
        expected_tables=1,
        write_tracker=catalog_tracker,
    )
    with _connect(inputs.lookup_database) as connection:
        _validate_source_catalog_closure(
            connection,
            write_tracker=catalog_tracker,
        )
        catalog_tracker.before_commit(0)
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


def _source_catalog_reference(source_table_id: str) -> dict[str, Any]:
    return {
        "source_table_id": source_table_id,
        "_materialized_record_ref": {
            "table": "source_catalog",
            "source_table_id": source_table_id,
        },
    }


def _raw_data_lake_reference(
    source_table_id: str,
    split: str,
) -> dict[str, Any]:
    table_id = f"dl_raw_{source_table_id}"
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": "raw_data_lake_table",
        "split": split,
        "source_table_id": source_table_id,
        "source_table_ref": {
            "artifact": "source_tables",
            "source_table_id": source_table_id,
        },
        "queryable": False,
        "reason": "no_column_met_recovered_value_ratio",
    }


def _compact_materialized_table_copies(
    database_path: Path,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    """Replace resumable full-table copies with deterministic references."""

    write_tracker = GuardedWriteTracker(
        database_path,
        pre_write_guard,
    )
    with _connect(database_path) as connection:
        stored = connection.execute(
            """
            SELECT value FROM metadata
            WHERE key = 'materialized_reference_storage'
            """
        ).fetchone()
    if (
        stored is not None
        and str(stored["value"])
        == _MATERIALIZED_REFERENCE_STORAGE_VERSION
    ):
        return

    for artifact in ("source_tables", "data_lake_tables"):
        last_ordinal = -1
        while True:
            condition = (
                ""
                if artifact == "source_tables"
                else (
                    "AND materialized.record_id = "
                    "'dl_raw_' || materialized.source_table_id"
                )
            )
            with _connect(database_path) as connection:
                rows = connection.execute(
                    f"""
                    SELECT materialized.record_id,
                           materialized.source_table_id,
                           materialized.source_ordinal,
                           LENGTH(materialized.record_json)
                               AS inline_bytes,
                           source_catalog.split
                    FROM materialized_records AS materialized
                    JOIN source_catalog
                      ON source_catalog.source_table_id =
                         materialized.source_table_id
                    WHERE materialized.artifact = ?
                      AND materialized.source_ordinal > ?
                      {condition}
                    ORDER BY materialized.source_ordinal
                    LIMIT ?
                    """,
                    (
                        artifact,
                        last_ordinal,
                        _MATERIALIZATION_READ_BATCH_RECORDS,
                    ),
                ).fetchall()
            if not rows:
                break
            write_tracker.before_write(
                4096
                + 2
                * sum(int(row["inline_bytes"] or 0) for row in rows)
            )
            with _connect(database_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                for row in rows:
                    source_table_id = str(row["source_table_id"])
                    record = (
                        _source_catalog_reference(source_table_id)
                        if artifact == "source_tables"
                        else _raw_data_lake_reference(
                            source_table_id,
                            clean_text(row["split"]),
                        )
                    )
                    connection.execute(
                        """
                        UPDATE materialized_records
                        SET record_json = ?, record_path = ''
                        WHERE artifact = ? AND record_id = ?
                        """,
                        (
                            _canonical_json(record),
                            artifact,
                            str(row["record_id"]),
                        ),
                    )
                write_tracker.before_commit(0)
                connection.commit()
            last_ordinal = int(rows[-1]["source_ordinal"])
            _checkpoint_wal(database_path)

    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('materialized_reference_storage', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_MATERIALIZED_REFERENCE_STORAGE_VERSION,),
        )
        write_tracker.before_commit(0)
        connection.commit()
    _checkpoint_wal(database_path)


def _insert_materialized_records(
    connection: sqlite3.Connection,
    *,
    database_path: Path,
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
        encoded = _canonical_json(record)
        encoded_size, digest = _json_text_identity(encoded)
        stored_json, record_path = _stored_json_values(
            database_path,
            namespace=f"materialized-records/{artifact}",
            identity=record_id,
            record=record,
            encoded=encoded,
            encoded_size=encoded_size,
            digest=digest,
        )
        try:
            connection.execute(
                """
                INSERT INTO materialized_records (
                    artifact, record_id, source_table_id,
                    source_ordinal, record_ordinal,
                    record_json, record_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact,
                    record_id,
                    source_table_id,
                    source_ordinal,
                    ordinal,
                    stored_json,
                    record_path,
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
    write_tracker: GuardedWriteTracker | None = None,
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
            "source_tables": [
                _source_catalog_reference(source_table_id)
            ],
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
                database_path=database_path,
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
        if write_tracker is not None:
            write_tracker.before_commit(0)
        connection.commit()
    return True


def _query_auto_check_plans_for_table(
    database_path: Path,
    source_table: dict[str, Any],
    *,
    split: str,
    args: argparse.Namespace,
) -> tuple[
    list[Any],
    dict[str, dict[str, Any]],
]:
    (
        _entities,
        assets,
        links,
        extractions,
        wiki_to_entity_id,
    ) = _table_inputs(database_path, source_table)
    assets_by_id = {
        str(asset["asset_id"]): asset for asset in assets
    }
    entity_to_assets: dict[str, list[str]] = {}
    for link in links:
        entity_id = clean_text(link.get("entity_id"))
        if not entity_id:
            continue
        linked = entity_to_assets.setdefault(entity_id, [])
        for asset_id in link.get("asset_ids") or []:
            asset_id = clean_text(asset_id)
            if asset_id and asset_id not in linked:
                linked.append(asset_id)

    plans: list[Any] = []
    prepass_args = copy.copy(args)
    prepass_args.cache_failed_model_outputs = True
    prepass_args.model_attribute_errors_path = ""
    join_builder.build_table_join_records(
        source_table=source_table,
        split=split,
        assets=assets_by_id,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=None,
        cache=_TableExtractionCache(extractions),
        progress=None,
        concurrency_state=join_builder.ModelConcurrencyState.from_args(
            prepass_args
        ),
        extraction_writer=_RecordSink(),
        recovery_writer=_RecordSink(),
        args=prepass_args,
        apply_query_auto_check=False,
        query_recovery_plans_out=plans,
    )
    return plans, {
        clean_text(record.get("cache_key")): record
        for record in extractions
        if clean_text(record.get("cache_key"))
    }


def _legacy_query_auto_check_record(
    candidate: Any,
    extraction: dict[str, Any] | None,
    *,
    extractor: Any,
) -> dict[str, Any] | None:
    """Split one completed eager Task-6 review into a query-level record."""
    if not isinstance(extraction, dict):
        return None
    auto_check = extraction.get("auto_check")
    if (
        not isinstance(auto_check, dict)
        or clean_text(auto_check.get("schema_version"))
        != join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION
    ):
        return None
    recorded_policy = clean_text(auto_check.get("review_policy"))
    recovered = candidate.recovery["recovered_attribute"]
    target_name = join_builder.normalize(recovered.get("column_name"))
    target_value = clean_text(recovered.get("value"))
    for review_value in auto_check.get("reviews") or []:
        if not isinstance(review_value, dict):
            continue
        review = dict(review_value)
        if (
            join_builder.normalize(review.get("attribute_name"))
            != target_name
            or not join_builder.values_match(
                review.get("claimed_value"),
                target_value,
                attribute_name=recovered.get("column_name"),
                entity_column_name=candidate.task.entity_column_name,
            )
            or not bool(review.get("review_complete"))
            or clean_text(review.get("error_code"))
        ):
            continue
        if not recorded_policy:
            if join_builder.query_recovery_remote_review_is_complete(
                {
                    "auto_check": {
                        "schema_version": (
                            join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION
                        ),
                        "reviewed_attributes": 1,
                        "reviews": [review],
                    }
                }
            ):
                recorded_policy = join_builder.AUTO_CHECK_REVIEW_POLICY_LEGACY
            else:
                recorded_policy = join_builder.AUTO_CHECK_REVIEW_POLICY_LOCAL
        supported = clean_text(review.get("verdict")) == "supported"
        key = join_builder.query_recovery_auto_check_key(candidate, extractor)
        return {
            "cache_key": key,
            "extraction_cache_key": candidate.task.cache_key,
            "review_policy": recorded_policy,
            "query_row_attributes": (
                join_builder.canonical_extraction_row_attributes(
                    candidate.task.entity.get("row_attributes")
                )
            ),
            "attribute_name": recovered.get("column_name"),
            "claimed_value": target_value,
            "evidence_identity": (
                join_builder.query_recovery_remote_evidence_identity(candidate)
            ),
            "schema_version": join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "supported": supported,
            "auto_check": {
                "schema_version": join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION,
                "policy": auto_check.get("policy")
                or "keep_source_canonical_supported_only_fail_closed",
                "review_policy": recorded_policy,
                "reviewed_attributes": 1,
                "supported_attributes": int(supported),
                "filtered_attributes": int(not supported),
                "reviews": [review],
            },
        }
    return None


def _migrate_legacy_query_auto_checks(
    plans: Iterable[Any],
    extraction_records: dict[str, dict[str, Any]],
    cache: Any,
    *,
    extractor: Any,
) -> int:
    migrated = 0
    for plan in plans:
        for candidate in plan.candidates:
            key = join_builder.query_recovery_auto_check_key(
                candidate,
                extractor,
            )
            if (
                join_builder.query_recovery_cached_check(
                    key,
                    cache,
                    candidate,
                    extractor=extractor,
                )
                is not None
            ):
                continue
            record = _legacy_query_auto_check_record(
                candidate,
                extraction_records.get(candidate.task.cache_key),
                extractor=extractor,
            )
            if record is None:
                continue
            cache.put(key, record)
            migrated += 1
    return migrated


def _persist_query_auto_check_batch(
    database_path: Path,
    units: list[tuple[_MaterializationWorkItem, list[Any]]],
    *,
    cache: Any,
    extractor: Any,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[int, int]:
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    persisted_checks = 0
    persisted_units = 0
    with _connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for item, plans in units:
            checks: dict[str, dict[str, Any]] = {}
            for plan in plans:
                for candidate in plan.candidates:
                    key = join_builder.query_recovery_auto_check_key(
                        candidate,
                        extractor,
                    )
                    record = join_builder.query_recovery_cached_check(
                        key,
                        cache,
                        candidate,
                        extractor=extractor,
                    )
                    if record is None:
                        continue
                    stored = {
                        **record,
                        "cache_key": key,
                        "extraction_cache_key": candidate.task.cache_key,
                    }
                    checks[key] = stored
            for key, record in checks.items():
                encoded = _canonical_json(record)
                write_tracker.before_write(
                    4096 + 2 * len(encoded.encode("utf-8"))
                )
                connection.execute(
                    """
                    INSERT INTO query_auto_checks (
                        cache_key, extraction_cache_key, record_json
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        extraction_cache_key = excluded.extraction_cache_key,
                        record_json = excluded.record_json
                    """,
                    (
                        key,
                        clean_text(record.get("extraction_cache_key")),
                        encoded,
                    ),
                )
            connection.execute(
                """
                INSERT INTO query_auto_check_units (
                    source_table_id, source_sha256, plan_count,
                    cached_check_count, complete
                ) VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(source_table_id) DO UPDATE SET
                    source_sha256 = excluded.source_sha256,
                    plan_count = excluded.plan_count,
                    cached_check_count = excluded.cached_check_count,
                    complete = 1
                """,
                (
                    item.source_table_id,
                    item.source_sha256,
                    len(plans),
                    len(checks),
                ),
            )
            persisted_checks += len(checks)
            persisted_units += 1
        write_tracker.before_commit(0)
        connection.commit()
    return persisted_units, persisted_checks


def _run_query_auto_check_prepass(
    database_path: Path,
    *,
    extractor: Any,
    cache: Any,
    args: argparse.Namespace,
    expected_tables: int,
    pre_write_guard: PreWriteGuard | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    with _connect(database_path) as connection:
        completed = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM query_auto_check_units
                WHERE complete = 1
                """
            ).fetchone()[0]
        )
    _report_work_progress(
        progress_callback,
        phase="query_auto_check",
        completed=completed,
        total=expected_tables,
    )
    last_ordinal = -1
    observed = 0
    migrated_total = 0
    cached_total = 0
    while True:
        with _connect(database_path) as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT source_catalog.source_table_id,
                           source_catalog.ordinal,
                           source_catalog.split,
                           source_catalog.record_sha256,
                           query_auto_check_units.source_sha256
                               AS checked_sha256,
                           query_auto_check_units.complete AS checked
                    FROM source_catalog
                    LEFT JOIN query_auto_check_units
                      ON query_auto_check_units.source_table_id =
                         source_catalog.source_table_id
                    WHERE source_catalog.ordinal > ?
                    ORDER BY source_catalog.ordinal
                    LIMIT ?
                    """,
                    (last_ordinal, _MATERIALIZATION_READ_BATCH_RECORDS),
                )
            ]
        if not rows:
            break
        observed += len(rows)
        pending: list[_MaterializationWorkItem] = []
        for row in rows:
            if row["checked"] is not None:
                if (
                    int(row["checked"]) != 1
                    or str(row["checked_sha256"])
                    != str(row["record_sha256"])
                ):
                    raise ValueError(
                        "query auto-check source unit resume mismatch: "
                        f"{row['source_table_id']}"
                    )
                continue
            pending.append(
                _MaterializationWorkItem(
                    source_table_id=str(row["source_table_id"]),
                    source_ordinal=int(row["ordinal"]),
                    source_sha256=str(row["record_sha256"]),
                    split=clean_text(row["split"]),
                )
            )

        units: list[tuple[_MaterializationWorkItem, list[Any]]] = []
        plans: list[Any] = []
        for item in pending:
            source_table = _load_materialization_source(database_path, item)
            table_plans, extraction_records = (
                _query_auto_check_plans_for_table(
                    database_path,
                    source_table,
                    split=item.split,
                    args=args,
                )
            )
            migrated_total += _migrate_legacy_query_auto_checks(
                table_plans,
                extraction_records,
                cache,
                extractor=extractor,
            )
            units.append((item, table_plans))
            plans.extend(table_plans)
        join_builder.finalize_query_recovery_auto_checks(
            plans=plans,
            extractor=extractor,
            cache=cache,
            args=args,
            concurrency_state=join_builder.ModelConcurrencyState.from_args(args),
        )
        persisted_units, persisted_checks = _persist_query_auto_check_batch(
            database_path,
            units,
            cache=cache,
            extractor=extractor,
            pre_write_guard=pre_write_guard,
        )
        completed += persisted_units
        cached_total += persisted_checks
        _report_work_progress(
            progress_callback,
            phase="query_auto_check",
            completed=completed,
            total=expected_tables,
            plans=len(plans),
            cached_checks=cached_total,
            migrated_legacy_checks=migrated_total,
        )
        last_ordinal = int(rows[-1]["ordinal"])
        _checkpoint_wal(database_path)
    if observed != expected_tables:
        raise ValueError("query auto-check source iteration count mismatch")
    with _connect(database_path) as connection:
        durable_completed = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM query_auto_check_units
                WHERE complete = 1
                """
            ).fetchone()[0]
        )
    if durable_completed != expected_tables:
        raise ValueError("query auto-check table barrier is incomplete")
    _report_work_progress(
        progress_callback,
        phase="query_auto_check",
        completed=durable_completed,
        total=expected_tables,
        cached_checks=cached_total,
        migrated_legacy_checks=migrated_total,
    )


def _materialize_all_tables(
    database_path: Path,
    *,
    args: argparse.Namespace,
    expected_tables: int,
    after_table_commit: Callable[[str], None] | None = None,
    pre_write_guard: PreWriteGuard | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    worker_count = int(getattr(args, "materialization_workers", 1))
    if worker_count <= 0:
        raise ValueError("materialization worker count must be positive")
    observed = 0
    last_ordinal = -1
    with _connect(database_path) as connection:
        completed_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM source_units WHERE complete = 1"
            ).fetchone()[0]
        )
    _report_work_progress(
        progress_callback,
        phase="materialize_tables",
        completed=completed_count,
        total=expected_tables,
    )
    executor: ProcessPoolExecutor | None = None
    with ExitStack() as stack:
        while True:
            with _connect(database_path) as connection:
                batch = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT source_catalog.source_table_id,
                               source_catalog.ordinal,
                               source_catalog.split,
                               source_catalog.record_sha256,
                               source_units.source_sha256
                                   AS completed_sha256,
                               source_units.split AS completed_split,
                               source_units.complete AS completed
                        FROM source_catalog
                        LEFT JOIN source_units
                          ON source_units.source_table_id =
                             source_catalog.source_table_id
                        WHERE source_catalog.ordinal > ?
                        ORDER BY source_catalog.ordinal
                        LIMIT ?
                        """,
                        (
                            last_ordinal,
                            _MATERIALIZATION_READ_BATCH_RECORDS,
                        ),
                    )
                ]
            if not batch:
                break
            observed += len(batch)
            work_items: list[_MaterializationWorkItem] = []
            for row in batch:
                source_table_id = str(row["source_table_id"])
                split = clean_text(row["split"])
                if split not in {"train", "dev", "test"}:
                    raise ValueError(
                        f"source table has invalid split: {source_table_id}"
                    )
                if row["completed"] is not None:
                    if (
                        str(row["completed_sha256"])
                        != str(row["record_sha256"])
                        or str(row["completed_split"]) != split
                        or int(row["completed"]) != 1
                    ):
                        raise ValueError(
                            f"source unit resume mismatch: {source_table_id}"
                        )
                    continue
                work_items.append(
                    _MaterializationWorkItem(
                        source_table_id=source_table_id,
                        source_ordinal=int(row["ordinal"]),
                        source_sha256=str(row["record_sha256"]),
                        split=split,
                    )
                )
            if executor is None and worker_count > 1 and work_items:
                executor = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=get_context("spawn"),
                        initializer=_initialize_materialization_worker,
                        initargs=(database_path, args),
                    )
                )
            materialized_tables: Iterable[MaterializedTable]
            if executor is None:
                materialized_tables = (
                    _materialize_work_item(database_path, args, item)
                    for item in work_items
                )
            else:
                materialized_tables = executor.map(
                    _run_materialization_worker,
                    work_items,
                    chunksize=1,
                )
            for item, materialized in zip(
                work_items,
                materialized_tables,
                strict=True,
            ):
                estimated_bytes = 4096
                for record in (
                    _source_catalog_reference(item.source_table_id),
                    materialized.decision,
                    *materialized.entities,
                    *materialized.bridge_assets,
                    *materialized.table_asset_links,
                    *materialized.query_tables,
                    *materialized.data_lake_tables,
                    *materialized.qrels,
                    *materialized.attribute_extractions,
                    *materialized.evidence_recoveries,
                ):
                    estimated_bytes += 2 * len(
                        _canonical_json(record).encode("utf-8")
                    )
                write_tracker.before_write(estimated_bytes)
                _store_table_unit(
                    database_path,
                    materialized,
                    source_ordinal=item.source_ordinal,
                    source_sha256=item.source_sha256,
                    split=item.split,
                    write_tracker=write_tracker,
                )
                completed_count += 1
                _report_work_progress(
                    progress_callback,
                    phase="materialize_tables",
                    completed=completed_count,
                    total=expected_tables,
                    source_table_id=item.source_table_id,
                )
                if after_table_commit is not None:
                    after_table_commit(item.source_table_id)
            last_ordinal = int(batch[-1]["ordinal"])
            _checkpoint_wal(database_path)
    if observed != expected_tables:
        raise ValueError("materialization source iteration count mismatch")
    with _connect(database_path) as connection:
        durable_completed_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM source_units WHERE complete = 1
                """
            ).fetchone()[0]
        )
    if durable_completed_count != expected_tables:
        raise ValueError("materialization table barrier is incomplete")
    _report_work_progress(
        progress_callback,
        phase="materialize_tables",
        completed=durable_completed_count,
        total=expected_tables,
    )


def _balance_explicit_join_records(
    database_path: Path,
    *,
    args: argparse.Namespace,
    pre_write_guard: PreWriteGuard | None = None,
) -> dict[str, Any] | None:
    """Promote deterministic explicit candidates to match implicit queries."""
    if (
        join_builder.configured_explicit_join_fallback_mode(args)
        != "match_implicit"
    ):
        return None

    decisions: dict[str, tuple[dict[str, Any], str]] = {}
    query_counts: dict[str, int] = {}
    splits: dict[str, str] = {}
    with _connect(database_path) as connection:
        stored_balance = connection.execute(
            "SELECT value FROM metadata WHERE key = 'explicit_join_balance_v1'"
        ).fetchone()
        for row in connection.execute(
            """
            SELECT decisions.source_table_id, decisions.record_id,
                   decisions.record_json, decisions.record_path,
                   catalog.split
            FROM materialized_records AS decisions
            JOIN source_catalog AS catalog
              ON catalog.source_table_id = decisions.source_table_id
            WHERE decisions.artifact = 'table_queryability_decisions'
            """
        ):
            source_table_id = str(row["source_table_id"])
            decisions[source_table_id] = (
                _load_stored_json(
                    database_path,
                    row["record_json"],
                    row["record_path"],
                ),
                str(row["record_id"]),
            )
            splits[source_table_id] = str(row["split"])
        query_counts = {
            str(row["source_table_id"]): int(row["records"])
            for row in connection.execute(
                """
                SELECT source_table_id, COUNT(*) AS records
                FROM materialized_records
                WHERE artifact = 'query_tables'
                GROUP BY source_table_id
                """
            )
        }

    implicit_by_split = {"train": 0, "dev": 0, "test": 0}
    explicit_by_split = {"train": 0, "dev": 0, "test": 0}
    candidate_splits: dict[str, str] = {}
    candidate_decisions: dict[str, dict[str, Any]] = {}
    candidate_sources: dict[str, str] = {}
    already_explicit: set[str] = set()
    for source_table_id, (decision, _record_id_value) in decisions.items():
        split = splits[source_table_id]
        reason = str(decision.get("reason") or "")
        queries = query_counts.get(source_table_id, 0)
        if reason == "queryable":
            implicit_by_split[split] += queries
            continue
        if reason == "explicit_join_fallback":
            already_explicit.add(source_table_id)
            explicit_by_split[split] += queries
        candidates = decision.get("explicit_join_candidates")
        if not isinstance(candidates, list):
            candidate = decision.get("explicit_join_candidate")
            candidates = [candidate] if isinstance(candidate, dict) else []
        if reason == "explicit_join_fallback" and not candidates:
            candidates = [decision]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = clean_text(candidate.get("candidate_id"))
            if not candidate_id:
                candidate_id = join_builder._explicit_join_candidate_id(
                    source_table_id,
                    int(candidate["entity_column_index"]),
                    int(candidate["join_column_index"]),
                )
                candidate = {**candidate, "candidate_id": candidate_id}
            candidate_splits[candidate_id] = split
            candidate_decisions[candidate_id] = candidate
            candidate_sources[candidate_id] = source_table_id

    selected, candidate_counts = (
        join_builder.select_balanced_explicit_join_candidates(
            candidate_splits=candidate_splits,
            implicit_query_counts=implicit_by_split,
            args=args,
        )
    )
    candidate_sources_by_split: dict[str, set[str]] = {
        "train": set(),
        "dev": set(),
        "test": set(),
    }
    for candidate_id, split in candidate_splits.items():
        candidate_sources_by_split[split].add(candidate_sources[candidate_id])
    candidate_table_counts = {
        split: len(source_ids)
        for split, source_ids in candidate_sources_by_split.items()
    }
    selected_sources = {
        candidate_sources[candidate_id] for candidate_id in selected
    }
    if not already_explicit.issubset(selected_sources):
        raise ValueError("materialized explicit joins violate balance selection")

    if stored_balance is not None:
        payload = json.loads(str(stored_balance["value"]))
        if (
            not isinstance(payload, dict)
            or payload.get("implicit_query_tables_by_split")
            != implicit_by_split
            or payload.get("candidate_tables_by_split")
            != candidate_table_counts
            or explicit_by_split != implicit_by_split
        ):
            raise ValueError("explicit join balance resume mismatch")
        return payload

    selected_by_source: dict[str, list[str]] = {}
    for candidate_id in selected:
        selected_by_source.setdefault(candidate_sources[candidate_id], []).append(
            candidate_id
        )

    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    for source_table_id in sorted(selected_by_source):
        if source_table_id in already_explicit:
            continue
        selected_candidate_ids = sorted(selected_by_source[source_table_id])
        with _connect(database_path) as connection:
            source_row = connection.execute(
                """
                SELECT record_json, record_path
                FROM source_catalog WHERE source_table_id = ?
                """,
                (source_table_id,),
            ).fetchone()
        if source_row is None:
            raise ValueError(
                f"balanced explicit source is missing: {source_table_id}"
            )
        source_table = _load_stored_json(
            database_path,
            source_row["record_json"],
            source_row["record_path"],
        )
        split = splits[source_table_id]
        explicit_queries: list[dict[str, Any]] = []
        explicit_targets: list[dict[str, Any]] = []
        explicit_qrels: list[dict[str, Any]] = []
        explicit_decisions: list[dict[str, Any]] = []
        selected_candidates: list[dict[str, Any]] = []
        for candidate_id in selected_candidate_ids:
            (
                candidate_queries,
                candidate_targets,
                candidate_qrels,
                candidate_result_decision,
            ) = join_builder.materialize_balanced_explicit_join_candidate(
                source_table=source_table,
                split=split,
                candidate_decision=candidate_decisions[candidate_id],
                args=args,
            )
            explicit_queries.extend(candidate_queries)
            explicit_targets.extend(candidate_targets)
            explicit_qrels.extend(candidate_qrels)
            explicit_decisions.append(candidate_result_decision)
            selected_candidates.append(candidate_decisions[candidate_id])
        explicit_decision = {
            **explicit_decisions[0],
            "source_table_id": source_table_id,
            "split": split,
            "qualified_columns": [
                qualified
                for item in explicit_decisions
                for qualified in item.get("qualified_columns", [])
            ],
            "explicit_join_candidates": selected_candidates,
            "explicit_join_candidate": selected_candidates[0],
            "explicit_join_query_count": len(explicit_queries),
        }
        estimated_bytes = 4096 + 2 * sum(
            len(_canonical_json(record).encode("utf-8"))
            for record in (
                *explicit_queries,
                *explicit_targets,
                *explicit_qrels,
                explicit_decision,
            )
        )
        write_tracker.before_write(estimated_bytes)
        with _connect(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                DELETE FROM table_ids
                WHERE source_table_id = ? AND artifact = 'data_lake_tables'
                """,
                (source_table_id,),
            )
            deleted = connection.execute(
                """
                DELETE FROM materialized_records
                WHERE source_table_id = ? AND artifact = 'data_lake_tables'
                """,
                (source_table_id,),
            ).rowcount
            if deleted != 1:
                raise ValueError(
                    "balanced explicit source does not have one raw table: "
                    f"{source_table_id}"
                )
            inserted_counts = {
                artifact: _insert_materialized_records(
                    connection,
                    database_path=database_path,
                    artifact=artifact,
                    records=records,
                    source_table_id=source_table_id,
                    source_ordinal=int(
                        connection.execute(
                            """
                            SELECT ordinal FROM source_catalog
                            WHERE source_table_id = ?
                            """,
                            (source_table_id,),
                        ).fetchone()[0]
                    ),
                )
                for artifact, records in (
                    ("query_tables", explicit_queries),
                    ("data_lake_tables", explicit_targets),
                    ("qrels", explicit_qrels),
                )
            }
            decision_json = _canonical_json(explicit_decision)
            decision_id = decisions[source_table_id][1]
            stored_json, record_path = _stored_json_values(
                database_path,
                namespace="materialized-records/table_queryability_decisions",
                identity=decision_id,
                record=explicit_decision,
                encoded=decision_json,
                encoded_size=len(decision_json.encode("utf-8")),
                digest=hashlib.sha256(
                    decision_json.encode("utf-8")
                ).hexdigest(),
            )
            updated = connection.execute(
                """
                UPDATE materialized_records
                SET record_json = ?, record_path = ''
                WHERE artifact = 'table_queryability_decisions'
                  AND source_table_id = ?
                """,
                (stored_json, source_table_id),
            ).rowcount
            if updated != 1 or record_path:
                raise ValueError("balanced explicit decision update failed")
            counts_row = connection.execute(
                "SELECT counts_json FROM source_units WHERE source_table_id = ?",
                (source_table_id,),
            ).fetchone()
            if counts_row is None:
                raise ValueError("balanced explicit source unit is missing")
            counts = json.loads(str(counts_row["counts_json"]))
            counts.update(inserted_counts)
            connection.execute(
                "UPDATE source_units SET counts_json = ? WHERE source_table_id = ?",
                (_canonical_json(counts), source_table_id),
            )
            write_tracker.before_commit(0)
            connection.commit()
        _checkpoint_wal(database_path)

    explicit_by_split = {"train": 0, "dev": 0, "test": 0}
    with _connect(database_path) as connection:
        for row in connection.execute(
            """
            SELECT catalog.split, COUNT(*) AS records
            FROM materialized_records AS queries
            JOIN source_catalog AS catalog
              ON catalog.source_table_id = queries.source_table_id
            JOIN materialized_records AS decisions
              ON decisions.source_table_id = queries.source_table_id
             AND decisions.artifact = 'table_queryability_decisions'
            WHERE queries.artifact = 'query_tables'
              AND json_extract(decisions.record_json, '$.reason') =
                  'explicit_join_fallback'
            GROUP BY catalog.split
            """
        ):
            explicit_by_split[str(row["split"])] = int(row["records"])
        if explicit_by_split != implicit_by_split:
            raise ValueError(
                "explicit and implicit query counts are not balanced: "
                f"explicit={explicit_by_split}, implicit={implicit_by_split}"
            )
        payload = {
            "mode": "match_implicit",
            "implicit_query_tables_by_split": implicit_by_split,
            "explicit_query_tables_by_split": explicit_by_split,
            "candidate_tables_by_split": candidate_table_counts,
        }
        connection.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('explicit_join_balance_v1', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (_canonical_json(payload),),
        )
        write_tracker.before_commit(0)
        connection.commit()
    _checkpoint_wal(database_path)
    return payload


class _AtomicArtifactWriter:
    def __init__(
        self,
        output_root: Path,
        artifact: str,
        records_per_shard: int,
        pre_write_guard: PreWriteGuard | None = None,
    ) -> None:
        self.output_root = output_root
        self.artifact = artifact
        self.records_per_shard = records_per_shard
        self.pre_write_guard = pre_write_guard
        self.directory = output_root / artifact
        if pre_write_guard is not None:
            pre_write_guard(self.directory, 0)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.completed: list[CompletedShard] = []
        self._writer: AtomicJsonlShard | None = None
        self._current_count = 0

    def _open(self) -> None:
        path = (
            self.directory
            / f"part-{len(self.completed):05d}.jsonl"
        )
        self._writer = AtomicJsonlShard(
            path,
            pre_write_guard=self.pre_write_guard,
        )
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
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> CompletedShard:
    writer = AtomicJsonlShard(path, pre_write_guard=pre_write_guard)
    try:
        writer.write(payload)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _atomic_jsonl_from_records(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> CompletedShard:
    writer = AtomicJsonlShard(path, pre_write_guard=pre_write_guard)
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
            SELECT record_json, record_path
            FROM materialized_records
            WHERE artifact = ?
            ORDER BY source_ordinal, record_ordinal, record_id
            """,
            (artifact,),
        ):
            record = _load_stored_json(
                database_path,
                row["record_json"],
                row["record_path"],
            )
            reference = record.get("_materialized_record_ref")
            if artifact == "source_tables" and isinstance(
                reference, dict
            ):
                source_table_id = clean_text(
                    reference.get("source_table_id")
                )
                source_row = connection.execute(
                    """
                    SELECT record_json, record_path
                    FROM source_catalog
                    WHERE source_table_id = ?
                    """,
                    (source_table_id,),
                ).fetchone()
                if source_row is None:
                    raise ValueError(
                        "materialized source-table reference is missing: "
                        f"{source_table_id}"
                    )
                record = _load_stored_json(
                    database_path,
                    source_row["record_json"],
                    source_row["record_path"],
                )
            yield record


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
    with _connect(database_path) as connection:
        queries_without_qrels = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM materialized_records AS queries
                WHERE queries.artifact = 'query_tables'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM materialized_records AS qrels
                      WHERE qrels.artifact = 'qrels'
                        AND json_extract(
                            qrels.record_json, '$.query_table_id'
                        ) = queries.record_id
                  )
                """
            ).fetchone()[0]
        )
        invalid_qrel_references = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM materialized_records AS qrels
                LEFT JOIN materialized_records AS queries
                  ON queries.artifact = 'query_tables'
                 AND queries.record_id = json_extract(
                     qrels.record_json, '$.query_table_id'
                 )
                LEFT JOIN materialized_records AS targets
                  ON targets.artifact = 'data_lake_tables'
                 AND targets.record_id = json_extract(
                     qrels.record_json, '$.target_table_id'
                 )
                WHERE qrels.artifact = 'qrels'
                  AND (
                      queries.record_id IS NULL
                      OR targets.record_id IS NULL
                      OR queries.source_table_id != qrels.source_table_id
                      OR targets.source_table_id != qrels.source_table_id
                  )
                """
            ).fetchone()[0]
        )
        ambiguous_implicit_queries = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT json_extract(
                        record_json, '$.query_table_id'
                    ) AS query_table_id
                    FROM materialized_records
                    WHERE artifact = 'qrels'
                      AND json_extract(
                          record_json, '$.reason'
                      ) = 'model_recoverable_join_column'
                    GROUP BY query_table_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )
    if queries_without_qrels:
        raise ValueError("global query table without qrel")
    if invalid_qrel_references:
        raise ValueError("global qrel reference closure is invalid")
    if ambiguous_implicit_queries:
        raise ValueError("global implicit query has multiple qrels")
    return counts


def _write_splits(
    database_path: Path,
    path: Path,
    args: argparse.Namespace,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> CompletedShard:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as raw_handle:
            handle = GuardedTextWriter(raw_handle, tracker)
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
                    "targets for queryable tables and source-table "
                    "references for rejected tables"
                )
            )
            handle.write("}\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        digest = _sha256_path(temporary)
        size = temporary.stat().st_size
        tracker.before_commit(0)
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


def _deduplicated_failures(
    database_path: Path,
    *,
    kind: str,
    records: Iterable[dict[str, Any]],
    pre_write_guard: PreWriteGuard | None = None,
) -> Iterator[dict[str, Any]]:
    write_tracker = GuardedWriteTracker(
        database_path,
        pre_write_guard,
    )
    write_tracker.before_write(64 * 1024)
    with _connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnostic_seen (
                kind TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                PRIMARY KEY (kind, record_sha256)
            )
            """
        )
        connection.execute(
            "DELETE FROM diagnostic_seen WHERE kind = ?",
            (kind,),
        )
        for record in records:
            encoded = _canonical_json(record)
            write_tracker.before_write(
                4096 + 2 * len(encoded.encode("utf-8"))
            )
            digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO diagnostic_seen (
                    kind, record_sha256
                ) VALUES (?, ?)
                """,
                (kind, digest),
            )
            if cursor.rowcount == 1:
                yield record
        write_tracker.before_commit(0)
        connection.commit()


def _combined_records(
    *record_sets: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    for records in record_sets:
        yield from records


def _stats_payload(
    database_path: Path,
    counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    with _connect(database_path) as connection:
        multimodal_queryable = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'table_queryability_decisions'
                  AND json_extract(record_json, '$.reason') = 'queryable'
                """
            ).fetchone()[0]
        )
        explicit_join = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'table_queryability_decisions'
                  AND json_extract(record_json, '$.reason') =
                      'explicit_join_fallback'
                """
            ).fetchone()[0]
        )
        query_counts_by_kind = {
            "implicit": {"train": 0, "dev": 0, "test": 0},
            "explicit": {"train": 0, "dev": 0, "test": 0},
        }
        for row in connection.execute(
            """
            SELECT catalog.split,
                   json_extract(decisions.record_json, '$.reason') AS reason,
                   COUNT(*) AS records
            FROM materialized_records AS queries
            JOIN source_catalog AS catalog
              ON catalog.source_table_id = queries.source_table_id
            JOIN materialized_records AS decisions
              ON decisions.source_table_id = queries.source_table_id
             AND decisions.artifact = 'table_queryability_decisions'
            WHERE queries.artifact = 'query_tables'
            GROUP BY catalog.split, reason
            """
        ):
            kind = (
                "explicit"
                if str(row["reason"]) == "explicit_join_fallback"
                else "implicit"
            )
            query_counts_by_kind[kind][str(row["split"])] += int(
                row["records"]
            )
        explicit_candidates = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM materialized_records
                WHERE artifact = 'table_queryability_decisions'
                  AND (
                    json_extract(record_json, '$.reason') =
                        'explicit_join_fallback'
                    OR json_type(
                        record_json, '$.explicit_join_candidate'
                    ) = 'object'
                  )
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
    queryable = multimodal_queryable + explicit_join
    return {
        "processed_tables": source_tables,
        "skipped_tables": 0,
        "source_tables": source_tables,
        "queryable_source_tables": queryable,
        "multimodal_queryable_source_tables": multimodal_queryable,
        "explicit_join_source_tables": explicit_join,
        "implicit_join_query_tables": sum(
            query_counts_by_kind["implicit"].values()
        ),
        "explicit_join_query_tables": sum(
            query_counts_by_kind["explicit"].values()
        ),
        "implicit_join_query_tables_by_split": query_counts_by_kind[
            "implicit"
        ],
        "explicit_join_query_tables_by_split": query_counts_by_kind[
            "explicit"
        ],
        "explicit_join_candidate_tables": explicit_candidates,
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
        "max_train_query_row_views_per_join": (
            join_builder.configured_max_train_query_row_views_per_join(args)
        ),
        "explicit_join_fallback_mode": (
            join_builder.configured_explicit_join_fallback_mode(args)
        ),
        "explicit_join_fallback_ratio": (
            join_builder.configured_explicit_join_fallback_ratio(args)
        ),
        "unrecoverable_replacement_rounds": int(
            getattr(args, "unrecoverable_replacement_rounds", 0)
        ),
        "unrecoverable_drop_probability": float(
            getattr(args, "unrecoverable_drop_probability", 0.5)
        ),
        "recovery_replacement_round_index": int(
            getattr(args, "recovery_replacement_round_index", 0)
        ),
        "notes": [
            "source_tables are the fixed data-lake base pool",
            "source-table rows are never capped",
            "sampling rejects source tables that cannot fill one query row view",
            "train join chains emit deterministic disjoint row views while dev/test retain one canonical view",
            "wide source tables may emit multiple query variants, one per "
            "qualifying bridge attribute",
            "when qualified attributes produce identical visible queries, only the highest-recovery deterministic attribute/target is retained so every implicit query has exactly one qrel",
            "match_implicit deterministically selects one viable explicit join per implicit query within each split",
            "query construction is delegated to "
            "build_mm_joinability_dataset.py",
            "every local-positive evidence candidate for a final accepted query receives an exhaustive auto-check before evidence_recoveries are materialized",
            "evidence_recoveries contain supported paths only; omitted evidence is an implicit negative",
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
        isinstance(payload, dict)
        and payload.get("stage") == "wdc200k_materialization"
            and payload.get("schema_version")
            in {
                "wdc200k-materialization-v1",
                "wdc200k-materialization-v2",
                "wdc200k-materialization-v3",
                "wdc200k-materialization-v4",
                "wdc200k-materialization-v5",
                "wdc200k-materialization-v6",
            }
    ):
        return None
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
    if payload.get("reference_format") != DATASET_REFERENCE_FORMAT:
        return None
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
    review_policy: str,
    parameter_fingerprint: str,
    records_per_shard: int,
    after_finalize_commit: Callable[[str], None] | None = None,
    pre_write_guard: PreWriteGuard | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> MaterializationResult:
    finalize_total = len(_CORE_ARTIFACTS) + 7
    finalized = 0

    def report_finalize(item: str) -> None:
        nonlocal finalized
        finalized += 1
        _report_work_progress(
            progress_callback,
            phase="publish_artifacts",
            completed=finalized,
            total=finalize_total,
            item=item,
        )

    _report_work_progress(
        progress_callback,
        phase="publish_artifacts",
        completed=0,
        total=finalize_total,
    )
    counts = _validate_global_counts(database_path, upstream)
    if pre_write_guard is not None:
        pre_write_guard(output_root, 0)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_shards: dict[str, tuple[CompletedShard, ...]] = {}
    for artifact in _CORE_ARTIFACTS:
        writer = _AtomicArtifactWriter(
            output_root,
            artifact,
            records_per_shard,
            pre_write_guard=pre_write_guard,
        )
        try:
            for record in _iter_materialized(database_path, artifact):
                writer.write(record)
            artifact_shards[artifact] = writer.close()
            report_finalize(f"artifact:{artifact}")
            if after_finalize_commit is not None:
                after_finalize_commit(f"artifact:{artifact}")
        except BaseException:
            writer.abort()
            raise

    qrels = _atomic_jsonl_from_records(
        output_root / "qrels.jsonl",
        _iter_materialized(database_path, "qrels"),
        pre_write_guard=pre_write_guard,
    )
    report_finalize("single:qrels.jsonl")
    if after_finalize_commit is not None:
        after_finalize_commit("single:qrels.jsonl")
    decisions = _atomic_jsonl_from_records(
        output_root / "table_queryability_decisions.jsonl",
        _iter_materialized(
            database_path, "table_queryability_decisions"
        ),
        pre_write_guard=pre_write_guard,
    )
    report_finalize("single:table_queryability_decisions.jsonl")
    if after_finalize_commit is not None:
        after_finalize_commit(
            "single:table_queryability_decisions.jsonl"
        )
    splits = _write_splits(
        database_path,
        output_root / "splits.json",
        args,
        pre_write_guard=pre_write_guard,
    )
    report_finalize("single:splits.json")
    if after_finalize_commit is not None:
        after_finalize_commit("single:splits.json")
    stats_payload = _stats_payload(database_path, counts, args)
    stats = _atomic_json(
        output_root / "stats.json",
        stats_payload,
        pre_write_guard=pre_write_guard,
    )
    report_finalize("single:stats.json")
    if after_finalize_commit is not None:
        after_finalize_commit("single:stats.json")
    diagnostics: dict[str, CompletedShard] = {}
    web_failure_name = "web_fetch_failures.jsonl"
    diagnostics[web_failure_name] = _atomic_jsonl_from_records(
        output_root / web_failure_name,
        _deduplicated_failures(
            database_path,
            kind="web",
            records=_combined_records(
                _failure_records(upstream.structural_failure_paths),
                _failure_records((upstream.page_failure_path,)),
            ),
            pre_write_guard=pre_write_guard,
        ),
        pre_write_guard=pre_write_guard,
    )
    report_finalize(f"single:{web_failure_name}")
    if after_finalize_commit is not None:
        after_finalize_commit(f"single:{web_failure_name}")
    media_failure_name = "media_download_failures.jsonl"
    diagnostics[media_failure_name] = _atomic_jsonl_from_records(
        output_root / media_failure_name,
        _deduplicated_failures(
            database_path,
            kind="media",
            records=iter_image_failures(
                upstream.image_fetch_result,
                planned=upstream.asset_plan_result,
                aggregation_database=(
                    upstream.image_failure_aggregation_database
                ),
                pre_write_guard=pre_write_guard,
            ),
            pre_write_guard=pre_write_guard,
        ),
        pre_write_guard=pre_write_guard,
    )
    report_finalize(f"single:{media_failure_name}")
    if after_finalize_commit is not None:
        after_finalize_commit(f"single:{media_failure_name}")
    model_error_name = "model_attribute_errors.jsonl"
    diagnostics[model_error_name] = _atomic_jsonl_from_records(
        output_root / model_error_name,
        _deduplicated_failures(
            database_path,
            kind="model",
            records=_combined_records(
                _failure_records(upstream.adapter_error_paths),
                _failure_records(upstream.model_error_paths),
            ),
            pre_write_guard=pre_write_guard,
        ),
        pre_write_guard=pre_write_guard,
    )
    report_finalize(f"single:{model_error_name}")
    if after_finalize_commit is not None:
        after_finalize_commit(f"single:{model_error_name}")
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
        "reference_format": DATASET_REFERENCE_FORMAT,
        "format": "sharded_jsonl",
        "records_per_shard": records_per_shard,
        "input_identity": upstream.identity,
        "parameter_fingerprint": parameter_fingerprint,
        "upstream": upstream.provenance,
        "artifact_references": {
            "data_lake_tables": {
                "field": "source_table_ref",
                "target_artifact": "source_tables",
                "resolution": "stream_by_source_table_id",
            }
        },
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
            "web_fetch_failures": "web_fetch_failures.jsonl",
            "media_download_failures": (
                "media_download_failures.jsonl"
            ),
            "model_attribute_errors": (
                "model_attribute_errors.jsonl"
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
        "query_construction": _parameter_payload(
            args,
            review_policy=review_policy,
        ),
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
    _atomic_json(
        manifest_path,
        manifest,
        pre_write_guard=pre_write_guard,
    )
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
    extractor: Any | None = None,
    records_per_shard: int = 50_000,
    after_table_commit: Callable[[str], None] | None = None,
    after_finalize_commit: Callable[[str], None] | None = None,
    pre_write_guard: PreWriteGuard | None = None,
    validation_progress_callback: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
    if pre_write_guard is not None:
        pre_write_guard(work_root / "upstream-validation", 0)
    query_auto_check_required = join_builder.auto_check_required(extractor)
    review_policy = _materialization_review_policy(extractor)
    parameter_fingerprint = _parameter_fingerprint(
        args,
        review_policy=review_policy,
    )
    validation_workers = _materialization_validation_workers(args)
    fast_resume: _FastResumeState | None = None
    try:
        fast_resume = _load_fast_resume_state(
            inputs,
            args=args,
            records_per_shard=records_per_shard,
            review_policy=review_policy,
        )
    except _CertificateMismatch as error:
        logging.info(
            "materialization upstream validation: fast resume unavailable "
            "(%s); falling back to strict validation",
            error,
        )
        _report_validation_progress(
            validation_progress_callback,
            mode="fallback",
            completed=0,
            total=1,
            reason=str(error),
        )

    strict_manifest_hashes: list[dict[str, str]] | None = None
    strict_input_files: list[dict[str, Any]] | None = None
    if fast_resume is None:
        manifest_paths = _certificate_manifest_paths(inputs)
        try:
            initial_manifest_hashes = _manifest_hashes(manifest_paths)
            initial_input_files = _file_identities(
                _certificate_input_paths(inputs)
            )
        except _CertificateMismatch:
            # Preserve the strict validators as the authority for corrupt
            # or missing input errors and their established messages.
            initial_manifest_hashes = None
            initial_input_files = None
        upstream = _validate_upstream(
            inputs,
            args=args,
            pre_write_guard=pre_write_guard,
            validation_progress_callback=(
                validation_progress_callback
            ),
        )
        current_manifest_hashes = _manifest_hashes(manifest_paths)
        current_input_files = _file_identities(
            _certificate_input_paths(inputs)
        )
        if (
            initial_manifest_hashes is not None
            and initial_manifest_hashes != current_manifest_hashes
        ):
            raise ValueError(
                "upstream manifests changed during strict validation"
            )
        if (
            initial_input_files is not None
            and initial_input_files != current_input_files
        ):
            raise ValueError(
                "upstream input identity changed during strict validation"
            )
        strict_manifest_hashes = current_manifest_hashes
        strict_input_files = _file_identities(
            _certificate_input_paths(inputs, upstream)
        )
        database_path = _materialization_database_path(
            work_root,
            upstream.identity,
            parameter_fingerprint,
        )
    else:
        upstream = fast_resume.upstream
        database_path = fast_resume.database_path
        logging.info(
            "materialization upstream validation: fast resume certificate "
            "accepted; continuing after %d/%d source units",
            fast_resume.resumed_source_units,
            upstream.expected_tables,
        )
        _report_validation_progress(
            validation_progress_callback,
            mode="fast_resume",
            completed=1,
            total=1,
            resumed_source_units=fast_resume.resumed_source_units,
            expected_source_units=upstream.expected_tables,
        )

    resumed = _load_published_result(
        output_root,
        upstream=upstream,
        parameter_fingerprint=parameter_fingerprint,
        records_per_shard=records_per_shard,
    )
    if resumed is not None:
        return resumed

    if pre_write_guard is not None:
        pre_write_guard(work_root, 0)
    work_root.mkdir(parents=True, exist_ok=True)
    if fast_resume is None:
        if pre_write_guard is not None:
            pre_write_guard(database_path, 0)
        _prepare_authoritative_index(
            database_path,
            upstream,
            pre_write_guard=pre_write_guard,
        )
        _migrate_legacy_external_json_paths(
            database_path,
            pre_write_guard=pre_write_guard,
        )
        if pre_write_guard is not None:
            pre_write_guard(database_path, 0)
        _catalog_sources(
            database_path,
            upstream.source_paths,
            args=args,
            expected_tables=upstream.expected_tables,
            pre_write_guard=pre_write_guard,
            progress_callback=progress_callback,
        )
        _validate_materialization_index_closures(
            database_path,
            validation_workers=validation_workers,
            pre_write_guard=pre_write_guard,
            validation_progress_callback=(
                validation_progress_callback
            ),
        )
        if pre_write_guard is not None:
            pre_write_guard(database_path, 0)
        _assign_splits(
            database_path,
            args,
            pre_write_guard=pre_write_guard,
        )
        _compact_materialized_table_copies(
            database_path,
            pre_write_guard=pre_write_guard,
        )
        with _connect(database_path) as connection:
            _validate_resume_index_units(
                connection,
                expected_tables=upstream.expected_tables,
            )
        if (
            strict_manifest_hashes
            != _manifest_hashes(_certificate_manifest_paths(inputs))
            or strict_input_files
            != _file_identities(
                _certificate_input_paths(inputs, upstream)
            )
        ):
            raise ValueError(
                "upstream identity changed before certificate persistence"
            )
        _persist_upstream_certificate(
            inputs,
            upstream,
            args=args,
            records_per_shard=records_per_shard,
            review_policy=review_policy,
            database_path=database_path,
            manifest_hashes=strict_manifest_hashes,
            input_files=strict_input_files,
            pre_write_guard=pre_write_guard,
        )
    materialize_args = copy.copy(args)
    materialize_args._query_auto_check_required = query_auto_check_required
    materialize_args._query_auto_check_review_policy = review_policy
    if query_auto_check_required:
        query_auto_check_cache = join_builder.ExtractionCache(
            Path(args.cache_dir).expanduser().resolve()
            / "query_recovery_auto_checks.jsonl",
            reuse=not bool(getattr(args, "no_reuse_model_cache", False)),
            record_key_alias=(
                lambda record: join_builder.query_recovery_auto_check_record_key(
                    record
                )
            ),
        )
        _run_query_auto_check_prepass(
            database_path,
            extractor=extractor,
            cache=query_auto_check_cache,
            args=materialize_args,
            expected_tables=upstream.expected_tables,
            pre_write_guard=pre_write_guard,
            progress_callback=progress_callback,
        )
    _materialize_all_tables(
        database_path,
        args=materialize_args,
        expected_tables=upstream.expected_tables,
        after_table_commit=after_table_commit,
        pre_write_guard=pre_write_guard,
        progress_callback=progress_callback,
    )
    _report_work_progress(
        progress_callback,
        phase="balance_explicit_joins",
        completed=0,
        total=1,
    )
    _balance_explicit_join_records(
        database_path,
        args=materialize_args,
        pre_write_guard=pre_write_guard,
    )
    _report_work_progress(
        progress_callback,
        phase="balance_explicit_joins",
        completed=1,
        total=1,
    )
    return _finalize_dataset(
        database_path,
        output_root=output_root,
        upstream=upstream,
        args=materialize_args,
        review_policy=review_policy,
        parameter_fingerprint=parameter_fingerprint,
        records_per_shard=records_per_shard,
        after_finalize_commit=after_finalize_commit,
        pre_write_guard=pre_write_guard,
        progress_callback=progress_callback,
    )

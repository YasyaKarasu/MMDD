"""Durable full-row structural expansion for selected WDC host tables."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    import build_wdc_mm_joinability_dataset as wdc_adapter
    from stage1_io import clean_text, stable_hash
    from wdc200k_io import (
        AtomicJsonlShard,
        CompletedShard,
        GuardedTextWriter,
        GuardedWriteTracker,
        PreWriteGuard,
        StageFingerprint,
        StageManifest,
        validate_completed_shard,
    )
    from wdc200k_selection import (
        ReplacementClaim,
        ReserveManager,
        TableCandidate,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "build_wdc_mm_joinability_dataset",
        "stage1_io",
        "wdc200k_io",
        "wdc200k_selection",
    }:
        raise
    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        wdc_adapter = importlib.import_module(
            "build_wdc_mm_joinability_dataset"
        )
        from stage1_io import clean_text, stable_hash
        from wdc200k_io import (
            AtomicJsonlShard,
            CompletedShard,
            GuardedTextWriter,
            GuardedWriteTracker,
            PreWriteGuard,
            StageFingerprint,
            StageManifest,
            validate_completed_shard,
        )
        from wdc200k_selection import (
            ReplacementClaim,
            ReserveManager,
            TableCandidate,
        )
    finally:
        sys.path.remove(scripts_directory)


CONTENT_HASH_SEMANTICS = "sha256-uncompressed-jsonl-bytes"
STRUCTURAL_SCHEMA_VERSION = "wdc200k-structural-v2"


class StructuralExpansionError(RuntimeError):
    """Raised when a selected table cannot produce a structural record."""


@dataclass(frozen=True)
class StructuralExpansionResult:
    source_tables: Path
    entities: Path
    page_refs: Path
    direct_image_refs: Path
    structural_failures: Path
    validated_selection: Path
    manifest: Path
    tables: int
    entities_count: int
    page_references: int
    direct_image_references: int


@dataclass(frozen=True)
class FinalizedSelectionResult:
    """The only validated-selection artifact downstream stages may consume."""

    validated_selection: Path
    manifest: Path
    tables: int


@dataclass(frozen=True)
class _ExpandedTable:
    source_table: dict[str, Any]
    raw_rows_path: Path
    content_hash: str


class _AtomicSourceTableShard(AtomicJsonlShard):
    """Task-1-compatible atomic shard with streamed source-table rows."""

    _encoder = json.JSONEncoder(ensure_ascii=False)

    def _write_value(self, value: Any) -> None:
        for chunk in self._encoder.iterencode(value):
            self.write_text(chunk)

    def write_source_table(
        self,
        source_table: dict[str, Any],
        rows: Iterable[dict[str, Any]],
    ) -> None:
        if self._handle.closed:
            raise RuntimeError("cannot write to a closed shard")
        prefix_fields = (
            "source_table_id",
            "source_file",
            "page_title",
            "caption",
            "section_title",
            "num_rows",
            "num_cols",
            "columns",
        )
        suffix_fields = ("provenance_builder", "metadata")
        self.write_text("{")
        first_field = True
        for field in prefix_fields:
            if not first_field:
                self.write_text(", ")
            self._write_value(field)
            self.write_text(": ")
            self._write_value(source_table[field])
            first_field = False
        self.write_text(", ")
        self._write_value("rows")
        self.write_text(": [")
        first_row = True
        row_count = 0
        for row in rows:
            if not first_row:
                self.write_text(", ")
            self._write_value(row)
            first_row = False
            row_count += 1
        if row_count != int(source_table["num_rows"]):
            raise StructuralExpansionError(
                f"streamed {row_count} rows but source table expected "
                f"{source_table['num_rows']}"
            )
        self.write_text("]")
        for field in suffix_fields:
            self.write_text(", ")
            self._write_value(field)
            self.write_text(": ")
            self._write_value(source_table[field])
        self.write_text("}\n")
        self._records += 1


def _candidate_from_record(
    record: TableCandidate | dict[str, Any],
) -> TableCandidate:
    if isinstance(record, TableCandidate):
        return record
    return TableCandidate(
        schema_class=str(record["schema_class"]),
        subset=str(record["subset"]),
        host=str(record["host"]),
        relative_path=str(record["relative_path"]),
        rows=int(record["rows"]),
        columns=int(record["columns"]),
    )


def _record_for_fingerprint(record: TableCandidate | dict[str, Any]) -> str:
    return json.dumps(
        _canonical_selection_record(record),
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_selection_record(
    record: TableCandidate | dict[str, Any],
) -> dict[str, Any]:
    candidate = _candidate_from_record(record)
    selection_seed = (
        int(record.get("selection_seed", 13))
        if isinstance(record, dict)
        else 13
    )
    return {
        "schema_class": candidate.schema_class,
        "subset": candidate.subset,
        "host": candidate.host,
        "relative_path": candidate.relative_path,
        "rows": candidate.rows,
        "columns": candidate.columns,
        "selection_seed": selection_seed,
    }


def _selection_fingerprint(
    records: Iterable[TableCandidate | dict[str, Any]],
    *,
    shard_id: str,
    explicit: str | None,
) -> str:
    if isinstance(records, Sequence):
        records_fingerprint = stable_hash(
            STRUCTURAL_SCHEMA_VERSION,
            shard_id,
            *(_record_for_fingerprint(record) for record in records),
            length=40,
        )
        if explicit:
            return stable_hash(
                explicit,
                records_fingerprint,
                length=40,
            )
        return records_fingerprint
    if explicit:
        return explicit
    else:
        raise ValueError(
            "input_fingerprint is required when selection_records is "
            "a streamed iterable"
        )


def _iter_selection_spool(
    path: Path,
) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prepare_selection_records(
    records: Iterable[TableCandidate | dict[str, Any]],
    *,
    shard_id: str,
    explicit_fingerprint: str | None,
    spool_root: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[
    Iterable[TableCandidate | dict[str, Any]],
    str,
    Path | None,
]:
    if isinstance(records, Sequence):
        return (
            records,
            _selection_fingerprint(
                records,
                shard_id=shard_id,
                explicit=explicit_fingerprint,
            ),
            None,
        )
    if not explicit_fingerprint:
        raise ValueError(
            "input_fingerprint is required when selection_records is "
            "a streamed iterable"
        )

    spool_path = spool_root / "selection-records.jsonl"
    tracker = GuardedWriteTracker(spool_path, pre_write_guard)
    digest = hashlib.sha256()
    try:
        with spool_path.open("w", encoding="utf-8") as raw_handle:
            handle = GuardedTextWriter(raw_handle, tracker)
            for record in records:
                canonical = _canonical_selection_record(record)
                encoded = json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest.update(encoded.encode("utf-8"))
                digest.update(b"\n")
                handle.write(encoded)
                handle.write("\n")
    except BaseException:
        spool_path.unlink(missing_ok=True)
        raise
    combined_fingerprint = stable_hash(
        STRUCTURAL_SCHEMA_VERSION,
        shard_id,
        explicit_fingerprint,
        digest.hexdigest(),
        length=40,
    )
    return (
        _iter_selection_spool(spool_path),
        combined_fingerprint,
        spool_path,
    )


def _paths(output_root: Path, shard_id: str) -> StructuralExpansionResult:
    filename = f"part-{shard_id}.jsonl"
    return StructuralExpansionResult(
        source_tables=output_root / "source_tables" / filename,
        entities=output_root / "entities" / filename,
        page_refs=output_root / "page_refs" / filename,
        direct_image_refs=output_root / "direct_image_refs" / filename,
        structural_failures=output_root / "structural_failures" / filename,
        validated_selection=(
            output_root / "selection" / f"validated-{shard_id}.jsonl"
        ),
        manifest=(
            output_root
            / "stage_manifests"
            / f"structural-{shard_id}.json"
        ),
        tables=0,
        entities_count=0,
        page_references=0,
        direct_image_references=0,
    )


def _relative_completed(
    completed: CompletedShard,
    path: Path,
    output_root: Path,
) -> CompletedShard:
    return replace(
        completed,
        path=path.relative_to(output_root).as_posix(),
    )


def _normalize_http_url(value: Any) -> str | None:
    text = clean_text(value)
    if not text or any(character.isspace() for character in text):
        return None
    try:
        parsed = urlsplit(text)
        scheme = parsed.scheme.lower()
        if (
            scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = 80 if scheme == "http" else 443
    netloc = hostname if port in {None, default_port} else f"{hostname}:{port}"
    return urlunsplit(
        (scheme, netloc, parsed.path or "", parsed.query, "")
    )


def _url_key(normalized_url: str) -> str:
    return hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()


def _read_table_once(
    path: Path,
    *,
    input_root: Path,
    min_rows: int,
    min_cols: int,
    spool_root: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> _ExpandedTable:
    digest = hashlib.sha256()
    spool_root.mkdir(parents=True, exist_ok=True)
    raw_descriptor, raw_name = tempfile.mkstemp(
        prefix="raw-table-",
        suffix=".jsonl",
        dir=spool_root,
    )
    os.close(raw_descriptor)
    profile_descriptor, profile_name = tempfile.mkstemp(
        prefix="profiles-",
        suffix=".sqlite3",
        dir=spool_root,
    )
    os.close(profile_descriptor)
    raw_rows_path = Path(raw_name)
    profile_path = Path(profile_name)
    raw_tracker = GuardedWriteTracker(raw_rows_path, pre_write_guard)
    profile_tracker = GuardedWriteTracker(profile_path, pre_write_guard)
    profile_tracker.before_write(64 * 1024)
    connection = sqlite3.connect(profile_path)
    column_names: list[str] = []
    seen_columns: set[str] = set()
    non_empty_counts: dict[str, int] = {}
    numeric_counts: dict[str, int] = {}
    examples: dict[str, list[str]] = {}
    row_count = 0
    try:
        connection.execute(
            """
            CREATE TABLE distinct_values (
                column_name TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (column_name, value)
            ) WITHOUT ROWID
            """
        )
        with raw_rows_path.open("w", encoding="utf-8") as raw_handle:
            raw_spool = GuardedTextWriter(raw_handle, raw_tracker)
            with gzip.open(path, "rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    text = raw_line.decode("utf-8")
                    payload = json.loads(text)
                    if not isinstance(payload, dict):
                        raise ValueError(
                            f"non-object JSON row at line {line_number}"
                        )
                    row_count += 1
                    profile_tracker.before_write(
                        4096 + 2 * len(text.encode("utf-8"))
                    )
                    for column_name, raw_value in payload.items():
                        if column_name in wdc_adapter.EXCLUDED_COLUMNS:
                            continue
                        if column_name not in seen_columns:
                            seen_columns.add(column_name)
                            column_names.append(column_name)
                            non_empty_counts[column_name] = 0
                            numeric_counts[column_name] = 0
                            examples[column_name] = []
                        value = clean_text(
                            wdc_adapter._cell_text(raw_value)
                        )
                        if not value:
                            continue
                        non_empty_counts[column_name] += 1
                        if wdc_adapter.is_numeric_text(value):
                            numeric_counts[column_name] += 1
                        inserted = connection.execute(
                            """
                            INSERT OR IGNORE INTO distinct_values (
                                column_name, value
                            ) VALUES (?, ?)
                            """,
                            (column_name, value),
                        ).rowcount
                        if inserted and len(examples[column_name]) < 8:
                            examples[column_name].append(value)
                    json.dump(payload, raw_spool, ensure_ascii=False)
                    raw_spool.write("\n")
        profile_tracker.before_commit(0)
        if row_count < min_rows:
            raise ValueError("too_few_rows")
        if len(column_names) < min_cols:
            raise ValueError("too_few_columns")
        entity_column = wdc_adapter._entity_column(column_names)
        if entity_column is None:
            raise ValueError("missing_entity_column")

        distinct_counts = {
            str(column_name): int(count)
            for column_name, count in connection.execute(
                """
                SELECT column_name, COUNT(*)
                FROM distinct_values
                GROUP BY column_name
                """
            )
        }
        relative_source = wdc_adapter._relative_source(path, input_root)
        schema_class = wdc_adapter._schema_class(relative_source)
        source_table_id = (
            "st_wdc_"
            + stable_hash(schema_class, relative_source, length=16)
        )
        columns: list[dict[str, Any]] = []
        profiles: list[dict[str, Any]] = []
        for column_index, column_name in enumerate(column_names):
            non_empty = non_empty_counts[column_name]
            distinct = distinct_counts.get(column_name, 0)
            profile = {
                "non_empty_ratio": non_empty / max(1, row_count),
                "unique_ratio": distinct / max(1, non_empty),
                "numeric_ratio": (
                    numeric_counts[column_name] / max(1, non_empty)
                ),
                "distinct_count": distinct,
                "examples": examples[column_name],
            }
            columns.append(
                {
                    "column_index": column_index,
                    "column_name": column_name,
                    "is_numeric_column": (
                        float(profile["numeric_ratio"]) >= 0.8
                    ),
                }
            )
            profiles.append(
                {
                    "column_index": column_index,
                    "column_name": column_name,
                    **profile,
                }
            )
        source_table: dict[str, Any] = {
            "source_table_id": source_table_id,
            "source_file": relative_source,
            "page_title": schema_class,
            "caption": "",
            "section_title": "",
            "num_rows": row_count,
            "num_cols": len(columns),
            "columns": columns,
            "provenance_builder": "build_wdc_mm_joinability_dataset.py",
            "metadata": {
                "candidate_entity_columns": [entity_column],
                "column_profiles": profiles,
            },
        }
        connection.commit()
        return _ExpandedTable(
            source_table=source_table,
            raw_rows_path=raw_rows_path,
            content_hash=digest.hexdigest(),
        )
    except BaseException:
        raw_rows_path.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        profile_path.unlink(missing_ok=True)


def _iter_canonical_rows(
    expanded: _ExpandedTable,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    source = expanded.source_table
    columns = source["columns"]
    column_names = [str(column["column_name"]) for column in columns]
    entity_column = int(
        source["metadata"]["candidate_entity_columns"][0]
    )
    schema_class = str(source["page_title"])
    relative_source = str(source["source_file"])
    source_table_id = str(source["source_table_id"])
    with expanded.raw_rows_path.open("r", encoding="utf-8") as handle:
        for fallback, line in enumerate(handle):
            raw_row = json.loads(line)
            source_row_id = wdc_adapter._source_row_id(
                raw_row,
                fallback,
            )
            display_text = wdc_adapter._cell_text(
                raw_row.get(column_names[entity_column])
            )
            page_url = clean_text(raw_row.get("page_url"))
            entity_key = (
                "wdc_"
                + stable_hash(
                    schema_class,
                    relative_source,
                    source_row_id,
                    page_url,
                    display_text,
                    length=20,
                )
            )
            entity_id = f"ent_{stable_hash(entity_key, length=16)}"
            image_urls = wdc_adapter.extract_image_urls(
                raw_row.get("image"),
                page_url,
            )
            cells: list[dict[str, Any]] = []
            for column_index, column_name in enumerate(column_names):
                raw_value = raw_row.get(column_name)
                is_entity = column_index == entity_column
                cells.append(
                    {
                        "column_index": column_index,
                        "column_name": column_name,
                        "raw": raw_value,
                        "text": wdc_adapter._cell_text(raw_value),
                        "wiki_title": entity_key if is_entity else None,
                        "has_wiki_link": is_entity,
                    }
                )
            appears_in = {
                "source_table_id": source_table_id,
                "query_view_id": None,
                "row_id": source_row_id,
                "column_index": entity_column,
                "column_name": column_names[entity_column],
            }
            entity = {
                "entity_id": entity_id,
                "wiki_title": entity_key,
                "display_texts": [display_text] if display_text else [],
                "context_terms": wdc_adapter._context_terms(
                    cells,
                    entity_column,
                ),
                "appears_in": [appears_in],
                "page_url": page_url,
                "image_urls": image_urls,
            }
            yield (
                {"row_id": source_row_id, "cells": cells},
                entity,
            )


def _failure_reason(error: Exception) -> str:
    return f"{type(error).__name__}: {clean_text(error)}"


def _operation_key(shard_id: str, invalid: TableCandidate) -> str:
    return stable_hash(
        "wdc200k-structural-replacement",
        shard_id,
        invalid.relative_path,
        length=40,
    )


def _claim_chain(
    reserve_manager: ReserveManager,
    claim: ReplacementClaim,
) -> list[ReplacementClaim]:
    seen: set[str] = set()
    current = claim
    chain = [current]
    while current.status == "superseded":
        if (
            current.operation_key in seen
            or not current.successor_operation_key
        ):
            raise StructuralExpansionError(
                "replacement journal contains a broken successor chain"
            )
        seen.add(current.operation_key)
        successor = reserve_manager.get_claim_by_operation(
            current.successor_operation_key
        )
        if successor is None:
            raise StructuralExpansionError(
                "replacement journal contains a broken successor chain"
            )
        current = successor
        chain.append(current)
    return chain


def _validated_record(
    candidate: TableCandidate,
    expanded: _ExpandedTable,
    *,
    selection_seed: int,
    lineage: list[str],
    reasons: list[str],
    operation_key: str | None,
) -> dict[str, Any]:
    source = expanded.source_table
    return {
        "schema_class": candidate.schema_class,
        "subset": candidate.subset,
        "host": candidate.host,
        "relative_path": candidate.relative_path,
        "rank": stable_hash(selection_seed, candidate.relative_path),
        "selection_seed": selection_seed,
        "rows": int(source["num_rows"]),
        "columns": int(source["num_cols"]),
        "content_hash": expanded.content_hash,
        "content_hash_semantics": CONTENT_HASH_SEMANTICS,
        "source_table_id": str(source["source_table_id"]),
        "replaces_path": lineage[0] if lineage else None,
        "replacement_reason": reasons[0] if reasons else None,
        "replacement_chain": lineage,
        "replacement_reasons": reasons,
        "replacement_operation_key": operation_key,
    }


def _completed_shard_from_payload(payload: dict[str, Any]) -> CompletedShard:
    return CompletedShard(
        path=str(payload["path"]),
        records=int(payload["records"]),
        bytes=int(payload["bytes"]),
        sha256=str(payload["sha256"]),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_shard_from_manifest(
    manifest_path: Path,
    *,
    output_root: Path,
) -> tuple[Path, int, str]:
    if not manifest_path.is_file():
        raise StructuralExpansionError(
            f"missing structural manifest: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = [
            _completed_shard_from_payload(record)
            for record in payload["completed_shards"]
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StructuralExpansionError(
            f"invalid structural manifest: {manifest_path}"
        ) from error
    if (
        payload.get("stage") != "wdc200k_structural"
        or payload.get("schema_version") != STRUCTURAL_SCHEMA_VERSION
        or payload.get("complete") is not True
    ):
        raise StructuralExpansionError(
            f"incomplete structural manifest: {manifest_path}"
        )
    if not completed:
        raise StructuralExpansionError(
            f"structural manifest has no shards: {manifest_path}"
        )
    manifest_prefix = "structural-"
    if (
        not manifest_path.stem.startswith(manifest_prefix)
        or len(manifest_path.stem) == len(manifest_prefix)
    ):
        raise StructuralExpansionError(
            f"invalid structural manifest name: {manifest_path}"
        )
    shard_id = manifest_path.stem.removeprefix(manifest_prefix)
    expected_paths = {
        "source_tables": f"source_tables/part-{shard_id}.jsonl",
        "entities": f"entities/part-{shard_id}.jsonl",
        "page_refs": f"page_refs/part-{shard_id}.jsonl",
        "direct_image_refs": (
            f"direct_image_refs/part-{shard_id}.jsonl"
        ),
        "structural_failures": (
            f"structural_failures/part-{shard_id}.jsonl"
        ),
        "validated_selection": (
            f"selection/validated-{shard_id}.jsonl"
        ),
    }
    completed_paths = [shard.path for shard in completed]
    if (
        len(completed_paths) != len(expected_paths)
        or set(completed_paths) != set(expected_paths.values())
    ):
        raise StructuralExpansionError(
            "structural manifest artifact set is incomplete, "
            "duplicated, or unknown"
        )
    artifacts = {
        artifact_type: next(
            shard
            for shard in completed
            if shard.path == expected_path
        )
        for artifact_type, expected_path in expected_paths.items()
    }
    for shard in completed:
        if not validate_completed_shard(shard, output_root):
            raise StructuralExpansionError(
                "structural shard checksum validation failed: "
                f"{shard.path}"
            )
    validated = artifacts["validated_selection"]
    source = artifacts["source_tables"]
    entities = artifacts["entities"]
    page_refs = artifacts["page_refs"]
    structural_failures = artifacts["structural_failures"]
    if validated.records != source.records:
        raise StructuralExpansionError(
            "structural source and validated-selection counts differ"
        )
    validated_path = output_root / validated.path
    validated_entity_rows = 0
    try:
        with validated_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                rows = int(record["rows"])
                if rows < 0:
                    raise ValueError("negative rows")
                validated_entity_rows += rows
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StructuralExpansionError(
            "validated selection rows are invalid"
        ) from error
    if entities.records != validated_entity_rows:
        raise StructuralExpansionError(
            f"structural entity count {entities.records} does not match "
            f"validated rows {validated_entity_rows}"
        )

    terminal_page_failures = 0
    failures_path = output_root / structural_failures.path
    try:
        with failures_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                failure = json.loads(line)
                if (
                    failure.get("failure_type")
                    == "structural_page_url_failure"
                    and failure.get("status") == "terminal"
                ):
                    terminal_page_failures += 1
    except (AttributeError, json.JSONDecodeError) as error:
        raise StructuralExpansionError(
            "structural failure record is invalid"
        ) from error
    page_coverage = page_refs.records + terminal_page_failures
    if page_coverage != entities.records:
        raise StructuralExpansionError(
            f"structural page coverage {page_coverage} does not match "
            f"entity count {entities.records}"
        )
    return (
        validated_path,
        validated.records,
        _sha256_path(manifest_path),
    )


def finalize_validated_selection(
    structural_manifest_paths: Iterable[Path],
    *,
    output_root: Path,
    target_tables: int,
    pre_write_guard: PreWriteGuard | None = None,
) -> FinalizedSelectionResult:
    """Atomically publish the global selection after every structural shard.

    Tasks 4 and later must consume only ``validated_selection`` returned by
    this function, never provisional per-shard selection records.
    """
    if target_tables <= 0:
        raise ValueError("target_tables must be positive")
    output_root = Path(output_root)
    manifests = [Path(path) for path in structural_manifest_paths]
    if not manifests:
        raise StructuralExpansionError("missing structural manifests")
    canonical_manifests = [path.resolve() for path in manifests]
    if len(canonical_manifests) != len(set(canonical_manifests)):
        raise StructuralExpansionError("duplicate structural manifest")

    validated_shards: list[tuple[Path, int]] = []
    manifest_hashes: list[tuple[str, str]] = []
    manifest_record_total = 0
    for manifest_path in sorted(canonical_manifests):
        validated_path, records, manifest_hash = (
            _validated_shard_from_manifest(
                manifest_path,
                output_root=output_root,
            )
        )
        validated_shards.append((validated_path, records))
        manifest_record_total += records
        manifest_hashes.append((manifest_path.as_posix(), manifest_hash))
    if manifest_record_total != target_tables:
        raise StructuralExpansionError(
            f"structural manifest target is {manifest_record_total}, "
            f"expected target {target_tables}"
        )

    final_path = (
        output_root / "selection" / "validated-selected-tables.jsonl"
    )
    manifest_path = (
        output_root
        / "stage_manifests"
        / "validated-selection-global.json"
    )
    fingerprint = StageFingerprint(
        stage="wdc200k_validated_selection",
        input_fingerprint=stable_hash(
            STRUCTURAL_SCHEMA_VERSION,
            *(
                f"{path}:{digest}"
                for path, digest in manifest_hashes
            ),
            length=40,
        ),
        parameter_fingerprint=stable_hash(
            "validated-selection-global-v1",
            target_tables,
            length=40,
        ),
        schema_version=STRUCTURAL_SCHEMA_VERSION,
    )
    final_manifest: StageManifest | None = None
    if manifest_path.exists():
        final_manifest = StageManifest(
            manifest_path,
            fingerprint,
            pre_write_guard=pre_write_guard,
        )
        if final_manifest.complete:
            expected_final_path = final_path.relative_to(
                output_root
            ).as_posix()
            if (
                len(final_manifest.completed_shards) != 1
                or final_manifest.completed_shards[0].path
                != expected_final_path
                or final_manifest.completed_shards[0].records
                != target_tables
                or not validate_completed_shard(
                    final_manifest.completed_shards[0],
                    output_root,
                )
            ):
                raise StructuralExpansionError(
                    "completed global validated selection failed validation"
                )
            return FinalizedSelectionResult(
                validated_selection=final_path,
                manifest=manifest_path,
                tables=target_tables,
            )

    spool_root = output_root / ".structural_spool" / "global-finalize"
    spool_root.mkdir(parents=True, exist_ok=True)
    for stale_spool_file in spool_root.iterdir():
        if stale_spool_file.is_file():
            stale_spool_file.unlink()
    descriptor, database_name = tempfile.mkstemp(
        prefix="validated-selection-",
        suffix=".sqlite3",
        dir=spool_root,
    )
    os.close(descriptor)
    database_path = Path(database_name)
    database_tracker = GuardedWriteTracker(
        database_path,
        pre_write_guard,
    )
    database_tracker.before_write(64 * 1024)
    writer = AtomicJsonlShard(
        final_path,
        pre_write_guard=pre_write_guard,
    )
    connection = sqlite3.connect(database_path)
    record_count = 0
    required_fields = {
        "source_table_id",
        "relative_path",
        "rows",
        "columns",
        "content_hash",
        "replacement_chain",
        "replacement_reasons",
        "replaces_path",
        "replacement_reason",
    }
    try:
        connection.execute(
            """
            CREATE TABLE seen (
                relative_path TEXT PRIMARY KEY,
                source_table_id TEXT NOT NULL UNIQUE
            )
            """
        )
        for validated_path, expected_records in validated_shards:
            shard_records = 0
            with validated_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    database_tracker.before_write(
                        4096 + 2 * len(line.encode("utf-8"))
                    )
                    if (
                        not isinstance(record, dict)
                        or not required_fields.issubset(record)
                    ):
                        raise StructuralExpansionError(
                            "validated selection record is incomplete"
                        )
                    try:
                        connection.execute(
                            """
                            INSERT INTO seen (
                                relative_path, source_table_id
                            ) VALUES (?, ?)
                            """,
                            (
                                str(record["relative_path"]),
                                str(record["source_table_id"]),
                            ),
                        )
                    except sqlite3.IntegrityError as error:
                        raise StructuralExpansionError(
                            "duplicate relative path or source table ID "
                            "in validated selection"
                        ) from error
                    writer.write(record)
                    shard_records += 1
                    record_count += 1
            if shard_records != expected_records:
                raise StructuralExpansionError(
                    "validated selection shard record count changed"
                )
        if record_count != target_tables:
            raise StructuralExpansionError(
                f"validated selection target is {record_count}, "
                f"expected target {target_tables}"
            )
        database_tracker.before_commit(0)
        connection.commit()
        completed = writer.commit()
        completed = _relative_completed(
            completed,
            final_path,
            output_root,
        )
        if final_manifest is None:
            final_manifest = StageManifest(
                manifest_path,
                fingerprint,
                pre_write_guard=pre_write_guard,
            )
        final_manifest.record_shard(completed)
        final_manifest.mark_complete()
    except BaseException:
        writer.abort()
        raise
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)

    return FinalizedSelectionResult(
        validated_selection=final_path,
        manifest=manifest_path,
        tables=record_count,
    )


def _acknowledge_validated_replacements(
    validated_selection: Path,
    reserve_manager: ReserveManager | None,
) -> None:
    if reserve_manager is None:
        return
    with validated_selection.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            operation_key = clean_text(
                record.get("replacement_operation_key")
            )
            if operation_key:
                reserve_manager.acknowledge(
                    operation_key=operation_key,
                    replacement_path=str(record["relative_path"]),
                )


def _completed_result(
    paths: StructuralExpansionResult,
) -> StructuralExpansionResult:
    tables = 0
    entities_count = 0
    page_references = 0
    direct_image_references = 0
    for path, counter_name in (
        (paths.source_tables, "tables"),
        (paths.entities, "entities"),
        (paths.page_refs, "pages"),
        (paths.direct_image_refs, "images"),
    ):
        with path.open("rb") as handle:
            count = sum(1 for line in handle if line.strip())
        if counter_name == "tables":
            tables = count
        elif counter_name == "entities":
            entities_count = count
        elif counter_name == "pages":
            page_references = count
        else:
            direct_image_references = count
    return replace(
        paths,
        tables=tables,
        entities_count=entities_count,
        page_references=page_references,
        direct_image_references=direct_image_references,
    )


def expand_selected_shard(
    selection_records: Iterable[TableCandidate | dict[str, Any]],
    *,
    output_root: Path,
    input_root: Path,
    reserve_manager: ReserveManager | None = None,
    shard_id: str = "00000",
    input_fingerprint: str | None = None,
    min_rows: int = 1,
    min_cols: int = 1,
    pre_write_guard: PreWriteGuard | None = None,
) -> StructuralExpansionResult:
    """Expand one provisional-selection shard with durable replacement recovery."""
    if not shard_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in shard_id
    ):
        raise ValueError("shard_id must contain only letters, digits, '-' or '_'")
    if min_rows <= 0 or min_cols <= 0:
        raise ValueError("min_rows and min_cols must be positive")
    output_root = Path(output_root)
    input_root = Path(input_root)
    paths = _paths(output_root, shard_id)
    spool_root = output_root / ".structural_spool" / shard_id
    if pre_write_guard is not None:
        pre_write_guard(spool_root, 0)
    spool_root.mkdir(parents=True, exist_ok=True)
    for stale_spool_file in spool_root.iterdir():
        if stale_spool_file.is_file():
            stale_spool_file.unlink()
    (
        selection_records,
        prepared_selection_fingerprint,
        selection_spool_path,
    ) = _prepare_selection_records(
        selection_records,
        shard_id=shard_id,
        explicit_fingerprint=input_fingerprint,
        spool_root=spool_root,
        pre_write_guard=pre_write_guard,
    )
    fingerprint = StageFingerprint(
        stage="wdc200k_structural",
        input_fingerprint=prepared_selection_fingerprint,
        parameter_fingerprint=stable_hash(
            STRUCTURAL_SCHEMA_VERSION,
            min_rows,
            min_cols,
            CONTENT_HASH_SEMANTICS,
            length=40,
        ),
        schema_version=STRUCTURAL_SCHEMA_VERSION,
    )
    try:
        manifest = StageManifest(
            paths.manifest,
            fingerprint,
            pre_write_guard=pre_write_guard,
        )
    except BaseException:
        if selection_spool_path is not None:
            selection_spool_path.unlink(missing_ok=True)
        raise
    if manifest.complete:
        try:
            if not manifest.completed_shards or not all(
                validate_completed_shard(shard, output_root)
                for shard in manifest.completed_shards
            ):
                raise StructuralExpansionError(
                    "completed structural output failed checksum validation"
                )
            _acknowledge_validated_replacements(
                paths.validated_selection,
                reserve_manager,
            )
            return _completed_result(paths)
        finally:
            if selection_spool_path is not None:
                selection_spool_path.unlink(missing_ok=True)

    try:
        writers = {
            "source_tables": _AtomicSourceTableShard(
                paths.source_tables,
                pre_write_guard=pre_write_guard,
            ),
            "entities": AtomicJsonlShard(
                paths.entities,
                pre_write_guard=pre_write_guard,
            ),
            "page_refs": AtomicJsonlShard(
                paths.page_refs,
                pre_write_guard=pre_write_guard,
            ),
            "direct_image_refs": AtomicJsonlShard(
                paths.direct_image_refs,
                pre_write_guard=pre_write_guard,
            ),
            "structural_failures": AtomicJsonlShard(
                paths.structural_failures,
                pre_write_guard=pre_write_guard,
            ),
            "validated_selection": AtomicJsonlShard(
                paths.validated_selection,
                pre_write_guard=pre_write_guard,
            ),
        }
    except BaseException:
        if selection_spool_path is not None:
            selection_spool_path.unlink(missing_ok=True)
        raise
    selected_count = 0
    successful_tables = 0
    entities_count = 0
    page_references = 0
    direct_image_references = 0
    try:
        for selected_record in selection_records:
            selected_count += 1
            current = _candidate_from_record(selected_record)
            selection_seed = (
                int(selected_record.get("selection_seed", 13))
                if isinstance(selected_record, dict)
                else 13
            )
            lineage: list[str] = []
            reasons: list[str] = []
            final_operation_key: str | None = None
            while True:
                path = input_root / current.relative_path
                try:
                    expanded = _read_table_once(
                        path,
                        input_root=input_root,
                        min_rows=min_rows,
                        min_cols=min_cols,
                        spool_root=spool_root,
                        pre_write_guard=pre_write_guard,
                    )
                    break
                except Exception as error:
                    reason = _failure_reason(error)
                    lineage.append(current.relative_path)
                    reasons.append(reason)
                    if reserve_manager is None:
                        raise StructuralExpansionError(reason) from error
                    claim = reserve_manager.claim_replacement(
                        operation_key=_operation_key(shard_id, current),
                        invalid_candidate=current,
                        reason=reason,
                    )
                    claim_chain = _claim_chain(reserve_manager, claim)
                    for journal_claim in claim_chain:
                        if (
                            not lineage
                            or lineage[-1] != journal_claim.invalid_path
                        ):
                            lineage.append(journal_claim.invalid_path)
                            reasons.append(journal_claim.reason)
                    claim = claim_chain[-1]
                    if claim.status != "pending" or claim.replacement is None:
                        raise StructuralExpansionError(
                            f"replacement operation {claim.operation_key} "
                            f"is {claim.status}"
                        )
                    current = claim.replacement
                    final_operation_key = claim.operation_key

            source = expanded.source_table
            try:
                def emitted_rows() -> Iterable[dict[str, Any]]:
                    nonlocal entities_count
                    nonlocal page_references
                    nonlocal direct_image_references
                    for row, entity in _iter_canonical_rows(expanded):
                        writers["entities"].write(entity)
                        entities_count += 1
                        appearance = entity["appears_in"][0]
                        normalized_page = _normalize_http_url(
                            entity.get("page_url")
                        )
                        if normalized_page is None:
                            raw_page = clean_text(entity.get("page_url"))
                            writers["structural_failures"].write(
                                {
                                    "failure_type": (
                                        "structural_page_url_failure"
                                    ),
                                    "stage": "structural",
                                    "status": "terminal",
                                    "error_class": (
                                        "missing_page_url"
                                        if not raw_page
                                        else "invalid_page_url"
                                    ),
                                    "entity_id": entity["entity_id"],
                                    "source_table_id": (
                                        source["source_table_id"]
                                    ),
                                    "row_id": appearance["row_id"],
                                    "page_url": raw_page,
                                }
                            )
                        else:
                            writers["page_refs"].write(
                                {
                                    "url_key": _url_key(normalized_page),
                                    "page_url": normalized_page,
                                    "entity_id": entity["entity_id"],
                                    "source_table_id": (
                                        source["source_table_id"]
                                    ),
                                    "row_id": appearance["row_id"],
                                }
                            )
                            page_references += 1
                        for ordinal, image_url in enumerate(
                            entity.get("image_urls") or []
                        ):
                            normalized_image = _normalize_http_url(
                                image_url
                            )
                            if normalized_image is None:
                                continue
                            writers["direct_image_refs"].write(
                                {
                                    "url_key": _url_key(normalized_image),
                                    "image_url": image_url,
                                    "entity_id": entity["entity_id"],
                                    "source_table_id": (
                                        source["source_table_id"]
                                    ),
                                    "row_id": appearance["row_id"],
                                    "ordinal": ordinal,
                                }
                            )
                            direct_image_references += 1
                        yield row

                writers["source_tables"].write_source_table(
                    source,
                    emitted_rows(),
                )
                successful_tables += 1
                writers["validated_selection"].write(
                    _validated_record(
                        current,
                        expanded,
                        selection_seed=selection_seed,
                        lineage=lineage,
                        reasons=reasons,
                        operation_key=final_operation_key,
                    )
                )
            finally:
                expanded.raw_rows_path.unlink(missing_ok=True)

        expected_tables = selected_count
        if successful_tables != expected_tables:
            raise StructuralExpansionError(
                f"expanded {successful_tables} tables but target is "
                f"{expected_tables}"
            )

        completed: list[tuple[CompletedShard, Path]] = []
        for name in (
            "source_tables",
            "entities",
            "page_refs",
            "direct_image_refs",
            "structural_failures",
            "validated_selection",
        ):
            writer = writers[name]
            completed.append((writer.commit(), writer.path))
        for shard, path in completed:
            manifest.record_shard(
                _relative_completed(shard, path, output_root)
            )
        manifest.mark_complete()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        if selection_spool_path is not None:
            selection_spool_path.unlink(missing_ok=True)
        raise

    if selection_spool_path is not None:
        selection_spool_path.unlink(missing_ok=True)
    _acknowledge_validated_replacements(
        paths.validated_selection,
        reserve_manager,
    )
    return replace(
        paths,
        tables=successful_tables,
        entities_count=entities_count,
        page_references=page_references,
        direct_image_references=direct_image_references,
    )

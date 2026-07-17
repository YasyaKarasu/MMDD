"""Durable full-row structural expansion for selected WDC host tables."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import sys
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
STRUCTURAL_SCHEMA_VERSION = "wdc200k-structural-v1"


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
class _ExpandedTable:
    source_table: dict[str, Any]
    entities: list[dict[str, Any]]
    content_hash: str


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
    candidate = _candidate_from_record(record)
    return json.dumps(
        {
            "schema_class": candidate.schema_class,
            "subset": candidate.subset,
            "host": candidate.host,
            "relative_path": candidate.relative_path,
            "rows": candidate.rows,
            "columns": candidate.columns,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _selection_fingerprint(
    records: Iterable[TableCandidate | dict[str, Any]],
    *,
    shard_id: str,
    explicit: str | None,
) -> str:
    if explicit:
        return explicit
    if not isinstance(records, Sequence):
        raise ValueError(
            "input_fingerprint is required when selection_records is "
            "a streamed iterable"
        )
    return stable_hash(
        STRUCTURAL_SCHEMA_VERSION,
        shard_id,
        *(_record_for_fingerprint(record) for record in records),
        length=40,
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
) -> _ExpandedTable:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
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
            rows.append(payload)
    adapted = wdc_adapter._adapt_wdc_rows(
        rows,
        path,
        input_root,
        min_rows,
        min_cols,
    )
    if adapted.source_table is None:
        raise ValueError(adapted.skip_reason or "invalid_wdc_table")
    return _ExpandedTable(
        source_table=adapted.source_table,
        entities=adapted.entities,
        content_hash=digest.hexdigest(),
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


def _claim_index(
    reserve_manager: ReserveManager,
) -> dict[str, ReplacementClaim]:
    return {
        claim.operation_key: claim
        for claim in [
            *reserve_manager.pending_claims(),
            *reserve_manager.terminal_claims(),
        ]
    }


def _claim_chain(
    reserve_manager: ReserveManager,
    claim: ReplacementClaim,
) -> list[ReplacementClaim]:
    claims = _claim_index(reserve_manager)
    seen: set[str] = set()
    current = claim
    chain = [current]
    while current.status == "superseded":
        if (
            current.operation_key in seen
            or not current.successor_operation_key
            or current.successor_operation_key not in claims
        ):
            raise StructuralExpansionError(
                "replacement journal contains a broken successor chain"
            )
        seen.add(current.operation_key)
        current = claims[current.successor_operation_key]
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
        "replaces_path": lineage[0] if lineage else None,
        "replacement_reason": reasons[0] if reasons else None,
        "replacement_chain": lineage,
        "replacement_reasons": reasons,
        "replacement_operation_key": operation_key,
    }


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
    fingerprint = StageFingerprint(
        stage="wdc200k_structural",
        input_fingerprint=_selection_fingerprint(
            selection_records,
            shard_id=shard_id,
            explicit=input_fingerprint,
        ),
        parameter_fingerprint=stable_hash(
            STRUCTURAL_SCHEMA_VERSION,
            min_rows,
            min_cols,
            CONTENT_HASH_SEMANTICS,
            length=40,
        ),
    )
    manifest = StageManifest(paths.manifest, fingerprint)
    if manifest.complete:
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

    writers = {
        "source_tables": AtomicJsonlShard(paths.source_tables),
        "entities": AtomicJsonlShard(paths.entities),
        "page_refs": AtomicJsonlShard(paths.page_refs),
        "direct_image_refs": AtomicJsonlShard(paths.direct_image_refs),
        "structural_failures": AtomicJsonlShard(
            paths.structural_failures
        ),
        "validated_selection": AtomicJsonlShard(
            paths.validated_selection
        ),
    }
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
            writers["source_tables"].write(source)
            successful_tables += 1
            for entity in expanded.entities:
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
                            "failure_type": "structural_page_url_failure",
                            "stage": "structural",
                            "status": "terminal",
                            "error_class": (
                                "missing_page_url"
                                if not raw_page
                                else "invalid_page_url"
                            ),
                            "entity_id": entity["entity_id"],
                            "source_table_id": source["source_table_id"],
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
                            "source_table_id": source["source_table_id"],
                            "row_id": appearance["row_id"],
                        }
                    )
                    page_references += 1
                for ordinal, image_url in enumerate(
                    entity.get("image_urls") or []
                ):
                    normalized_image = _normalize_http_url(image_url)
                    if normalized_image is None:
                        continue
                    writers["direct_image_refs"].write(
                        {
                            "url_key": _url_key(normalized_image),
                            "image_url": image_url,
                            "entity_id": entity["entity_id"],
                            "source_table_id": source["source_table_id"],
                            "row_id": appearance["row_id"],
                            "ordinal": ordinal,
                        }
                    )
                    direct_image_references += 1
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
        raise

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

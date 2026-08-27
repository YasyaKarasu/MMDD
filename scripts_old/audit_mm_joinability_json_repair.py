#!/usr/bin/env python
"""Audit malformed historical model extraction JSON and its rebuild impact."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from build_mm_joinability_dataset import (
    PROMPT_VERSION,
    candidate_attribute_columns,
    canonical_extraction_row_attributes,
    choose_entity_column,
    clean_text,
    extraction_row_attributes,
    get_cell,
    get_column_name,
    iter_jsonl_records,
    model_auto_check_is_complete,
    normalize,
    normalize_extracted_attributes,
    parse_json_object,
    row_id,
    values_match,
)


def _normalized_stored_attributes(record: dict[str, Any]) -> list[dict[str, Any]]:
    target_field = (
        "model_attributes"
        if isinstance(record.get("auto_check"), dict)
        else "attributes"
    )
    return normalize_extracted_attributes(
        {"attributes": record.get(target_field)},
        record.get("candidate_attribute_names") or [],
    )


def audit_extraction_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_response = clean_text(record.get("raw_response"))
    candidate_names = [
        clean_text(name)
        for name in record.get("candidate_attribute_names") or []
        if clean_text(name)
    ]
    parsed = parse_json_object(raw_response)
    repaired_attributes = normalize_extracted_attributes(
        parsed.payload,
        candidate_names,
    )
    legacy_attributes: list[dict[str, Any]] = []
    if parsed.method == "json_repair":
        legacy_attributes = normalize_extracted_attributes(
            parse_json_object(raw_response, allow_repair=False).payload,
            candidate_names,
        )
    repairable = bool(
        parsed.method == "json_repair"
        and repaired_attributes
        and not legacy_attributes
    )
    return {
        "parse_method": parsed.method,
        "repairable": repairable,
        "requires_cache_update": (
            repairable
            and repaired_attributes != _normalized_stored_attributes(record)
        ),
        "repaired_attributes": repaired_attributes,
        "legal_empty": (
            parsed.method == "json"
            and parsed.payload.get("attributes") == []
        ),
    }


def _row_identity(
    *,
    wiki_title: Any,
    row_attributes: Any,
    candidate_attribute_names: Any,
) -> str:
    return json.dumps(
        {
            "wiki_title": clean_text(wiki_title),
            "row_attributes": canonical_extraction_row_attributes(
                row_attributes
            ),
            "candidate_attribute_names": [
                clean_text(name)
                for name in candidate_attribute_names or []
                if clean_text(name)
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _scan_extraction_cache_range(
    path_value: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    path = Path(path_value)
    counts: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    repairable_by_prompt: Counter[str] = Counter()
    repaired_attribute_names: Counter[str] = Counter()
    repairable_keys: set[str] = set()

    with path.open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        while handle.tell() < end:
            line = handle.readline()
            if not line:
                break
            counts["historical_records"] += 1
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["invalid_cache_jsonl_records"] += 1
                continue
            cache_key = clean_text(record.get("cache_key"))
            if not cache_key:
                counts["records_without_cache_key"] += 1
                continue
            result = audit_extraction_record(record)
            method = str(result["parse_method"])
            methods[method] += 1
            if result["legal_empty"]:
                counts["legal_empty_attribute_responses"] += 1
            if method not in {"json", "json_non_object", "empty"}:
                counts["non_native_json_responses"] += 1
            if method == "json_repair":
                counts["responses_parsed_by_json_repair"] += 1
                if not result["repaired_attributes"]:
                    counts["repair_parser_results_without_valid_attributes"] += 1
            if result["repairable"]:
                counts["historical_repairable_records"] += 1
                repaired = list(result["repaired_attributes"])
                counts["historical_repaired_attributes"] += len(repaired)
                prompt_version = clean_text(record.get("prompt_version"))
                repairable_by_prompt[prompt_version or "<missing>"] += 1
                repaired_attribute_names.update(
                    clean_text(item.get("name")) for item in repaired
                )
                repairable_keys.add(cache_key)

    return {
        "counts": dict(counts),
        "methods": dict(methods),
        "repairable_by_prompt": dict(repairable_by_prompt),
        "repaired_attribute_names": dict(repaired_attribute_names),
        "repairable_keys": sorted(repairable_keys),
    }


def _effective_repairable_records(
    path: Path,
    repairable_keys: set[str],
) -> dict[str, dict[str, Any]]:
    final_records: dict[str, tuple[int, dict[str, Any]]] = {}
    with path.open("rb") as handle:
        while True:
            byte_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            cache_key = clean_text(record.get("cache_key"))
            if cache_key in repairable_keys:
                final_records[cache_key] = (byte_offset, record)

    effective: dict[str, dict[str, Any]] = {}
    for cache_key, (byte_offset, record) in final_records.items():
        result = audit_extraction_record(record)
        if not result["repairable"]:
            continue
        effective[cache_key] = {
            "cache_key": cache_key,
            "byte_offset": byte_offset,
            "prompt_version": clean_text(record.get("prompt_version")),
            "entity_id": clean_text(record.get("entity_id")),
            "entity_wiki_title": clean_text(
                record.get("entity_wiki_title")
            ),
            "row_attributes": canonical_extraction_row_attributes(
                record.get("row_attributes")
            ),
            "asset_id": clean_text(record.get("asset_id")),
            "asset_type": clean_text(record.get("asset_type")),
            "candidate_attribute_names": list(
                record.get("candidate_attribute_names") or []
            ),
            "repaired_attributes": list(result["repaired_attributes"]),
            "requires_cache_update": bool(result["requires_cache_update"]),
        }
    return effective


def scan_extraction_cache(
    path: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    size = path.stat().st_size
    workers = max(1, min(workers, size or 1))
    ranges = [
        (index * size // workers, (index + 1) * size // workers)
        for index in range(workers)
    ]
    counts: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    repairable_by_prompt: Counter[str] = Counter()
    repaired_attribute_names: Counter[str] = Counter()
    repairable_keys: set[str] = set()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _scan_extraction_cache_range,
                str(path),
                start,
                end,
            )
            for start, end in ranges
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            partial = future.result()
            counts.update(partial["counts"])
            methods.update(partial["methods"])
            repairable_by_prompt.update(partial["repairable_by_prompt"])
            repaired_attribute_names.update(
                partial["repaired_attribute_names"]
            )
            repairable_keys.update(partial["repairable_keys"])
            print(
                f"Extraction cache repair scan: {completed}/{workers} chunks",
                flush=True,
            )

    print(
        "Resolving final append-only versions for "
        f"{len(repairable_keys)} repairable keys",
        flush=True,
    )
    active_repairable = _effective_repairable_records(
        path,
        repairable_keys,
    )

    active_current = {
        key: record
        for key, record in active_repairable.items()
        if record["prompt_version"] == PROMPT_VERSION
        and record["requires_cache_update"]
    }
    summary = {
        **counts,
        "parse_methods": dict(sorted(methods.items())),
        "repairable_records_by_prompt_version": dict(
            sorted(repairable_by_prompt.items())
        ),
        "historical_repaired_attribute_names": dict(
            repaired_attribute_names.most_common()
        ),
        "effective_repairable_cache_keys": len(active_repairable),
        "effective_current_prompt_cache_keys_requiring_update": len(
            active_current
        ),
        "effective_current_prompt_repaired_attributes": sum(
            len(item["repaired_attributes"])
            for item in active_current.values()
        ),
    }
    return summary, active_current


def _checkpoint_chunks(checkpoint_dir: Path) -> list[Path]:
    paths = sorted((checkpoint_dir / "chunks").glob("*/*.jsonl"))
    if not paths:
        raise FileNotFoundError(
            f"no materialized source sample chunks under {checkpoint_dir}"
        )
    return paths


def map_repairable_records_to_source_rows(
    records: dict[str, dict[str, Any]],
    source_paths: Iterable[Path],
    *,
    query_rows_per_table: int,
    min_column_non_empty_ratio: float,
) -> tuple[dict[str, Any], set[tuple[str, str, str]]]:
    keys_by_identity: dict[str, list[str]] = defaultdict(list)
    for key, record in records.items():
        if not record["row_attributes"]:
            continue
        keys_by_identity[
            _row_identity(
                wiki_title=record["entity_wiki_title"],
                row_attributes=record["row_attributes"],
                candidate_attribute_names=record["candidate_attribute_names"],
            )
        ].append(key)

    matched_keys: set[str] = set()
    table_rows: dict[str, set[int]] = defaultdict(set)
    table_attributes: dict[str, set[str]] = defaultdict(set)
    potential_checks: set[tuple[str, str, str]] = set()
    table_details: dict[str, dict[str, Any]] = {}

    for checkpoint_record in iter_jsonl_records(source_paths):
        source_table = checkpoint_record.get("source_table")
        if not isinstance(source_table, dict):
            continue
        entity_col = choose_entity_column(
            source_table,
            min_linked_rows=query_rows_per_table,
        )
        if entity_col is None:
            continue
        attribute_cols = candidate_attribute_columns(
            source_table,
            entity_col,
            min_column_non_empty_ratio,
        )
        candidate_names = [
            get_column_name(source_table, col) for col in attribute_cols
        ]
        if not candidate_names:
            continue
        source_table_id = clean_text(source_table.get("source_table_id"))
        for fallback, source_row in enumerate(source_table.get("rows") or []):
            entity_cell = get_cell(source_row, entity_col)
            identity = _row_identity(
                wiki_title=entity_cell.get("wiki_title"),
                row_attributes=extraction_row_attributes(
                    source_table,
                    source_row,
                    entity_col,
                ),
                candidate_attribute_names=candidate_names,
            )
            matched = keys_by_identity.get(identity)
            if not matched:
                continue
            source_row_id = row_id(source_row, fallback)
            source_values = {
                normalize(item.get("name")): (
                    clean_text(item.get("name")),
                    clean_text(item.get("value")),
                )
                for item in extraction_row_attributes(
                    source_table,
                    source_row,
                    entity_col,
                )
            }
            table_rows[source_table_id].add(source_row_id)
            for key in matched:
                matched_keys.add(key)
                for repaired in records[key]["repaired_attributes"]:
                    source_value = source_values.get(
                        normalize(repaired.get("name"))
                    )
                    if source_value is None:
                        continue
                    attribute_name, claimed_value = source_value
                    if not values_match(
                        repaired.get("value"),
                        claimed_value,
                        attribute_name=attribute_name,
                        entity_column_name=get_column_name(
                            source_table, entity_col
                        ),
                    ):
                        continue
                    table_attributes[source_table_id].add(attribute_name)
                    potential_checks.add(
                        (key, normalize(attribute_name), claimed_value)
                    )
            table_details[source_table_id] = {
                "source_table_id": source_table_id,
                "page_title": clean_text(source_table.get("page_title")),
            }

    details = []
    for table_id in sorted(table_rows):
        details.append(
            {
                **table_details[table_id],
                "source_row_ids": sorted(table_rows[table_id]),
                "repaired_attribute_names": sorted(
                    table_attributes[table_id]
                ),
            }
        )
    summary = {
        "mapped_cache_keys": len(matched_keys),
        "unmapped_cache_keys": len(records) - len(matched_keys),
        "affected_source_tables": len(table_rows),
        "affected_source_rows": sum(len(rows) for rows in table_rows.values()),
        "affected_table_attributes": sum(
            len(names) for names in table_attributes.values()
        ),
        "potential_new_recovery_candidates": len(potential_checks),
        "tables": details,
    }
    return summary, potential_checks


def estimate_incremental_auto_checks(
    path: Path,
    potential_checks: set[tuple[str, str, str]],
) -> dict[str, int]:
    completed_candidates: set[tuple[str, str, str]] = set()
    historical_records = 0
    for record in iter_jsonl_records([path]):
        historical_records += 1
        if not model_auto_check_is_complete(record, required=True):
            continue
        candidate = (
            clean_text(record.get("extraction_cache_key")),
            normalize(record.get("attribute_name")),
            clean_text(record.get("claimed_value")),
        )
        if candidate in potential_checks:
            completed_candidates.add(candidate)
    return {
        "historical_auto_check_records": historical_records,
        "potential_new_recovery_candidates": len(potential_checks),
        "matching_completed_candidate_checks": len(completed_candidates),
        "estimated_additional_auto_checks": (
            len(potential_checks - completed_candidates)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction_cache", type=Path, required=True)
    parser.add_argument("--auto_check_cache", type=Path, required=True)
    parser.add_argument("--source_sample_checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query_rows_per_table", type=int, default=5)
    parser.add_argument("--min_column_non_empty_ratio", type=float, default=0.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_summary, active_records = scan_extraction_cache(
        args.extraction_cache,
        workers=args.workers,
    )
    print("Mapping repaired keys to source sample rows", flush=True)
    source_summary, potential_checks = map_repairable_records_to_source_rows(
        active_records,
        _checkpoint_chunks(args.source_sample_checkpoint_dir),
        query_rows_per_table=args.query_rows_per_table,
        min_column_non_empty_ratio=args.min_column_non_empty_ratio,
    )
    auto_check_summary = estimate_incremental_auto_checks(
        args.auto_check_cache,
        potential_checks,
    )
    print("Writing audit report", flush=True)
    report = {
        "schema_version": "mm-joinability-json-repair-audit-v1",
        "parser": "json-repair",
        "current_prompt_version": PROMPT_VERSION,
        "inputs": {
            "extraction_cache": str(args.extraction_cache.resolve()),
            "auto_check_cache": str(args.auto_check_cache.resolve()),
            "source_sample_checkpoint_dir": str(
                args.source_sample_checkpoint_dir.resolve()
            ),
        },
        "cache_audit": cache_summary,
        "source_impact": source_summary,
        "incremental_auto_check_estimate": auto_check_summary,
        "estimate_note": (
            "Candidate-level estimate for repaired current-prompt records whose "
            "value matches the source row. The rebuild still applies the unchanged "
            "query-level eligibility and exhaustive accepted-evidence policy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

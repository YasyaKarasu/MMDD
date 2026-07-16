#!/usr/bin/env python
"""Adapt WDC Schema.org gzip JSONL host tables to the internal table schema."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

from stage1_io import clean_text, column_profiles, is_numeric_text, stable_hash


ENTITY_COLUMN_PRIORITY = ("name", "headline", "title", "identifier", "page_url")
EXCLUDED_COLUMNS = {"row_id", "image"}


@dataclass
class WdcTableResult:
    source_table: dict[str, Any] | None
    entities: list[dict[str, Any]]
    image_urls_by_entity: dict[str, list[str]]
    skip_reason: str | None
    malformed_rows: int


def _looks_like_relative_url(value: str) -> bool:
    return (
        value.startswith(("/", "./", "../"))
        or "/" in value
        or "." in value
    )


def extract_image_urls(value: Any, base_url: str = "") -> list[str]:
    """Return first-seen HTTP(S) URLs found recursively in an image value."""
    urls: list[str] = []
    seen: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
            return
        if not isinstance(item, str):
            return

        candidate = clean_text(item)
        if not candidate or any(char.isspace() for char in candidate):
            return
        try:
            parsed_candidate = urlsplit(candidate)
        except ValueError:
            return
        if not parsed_candidate.scheme and not _looks_like_relative_url(candidate):
            return
        try:
            resolved = urljoin(base_url, candidate)
            parsed = urlsplit(resolved)
        except ValueError:
            return
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            return
        if resolved not in seen:
            seen.add(resolved)
            urls.append(resolved)

    visit(value)
    return urls


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return clean_text(value)


def _relative_source(path: Path, input_root: Path) -> str:
    try:
        relative = path.relative_to(input_root)
    except ValueError:
        relative = Path(path.name)
    return relative.as_posix()


def _schema_class(relative_source: str) -> str:
    parts = Path(relative_source).parts
    if len(parts) > 1:
        return parts[0]
    filename = parts[0] if parts else relative_source
    return filename.split("_", 1)[0].removesuffix(".json.gz")


def _read_rows(path: Path, max_rows: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_rows = 0
    with gzip.open(path, "rb") as handle:
        for raw_line in handle:
            if max_rows > 0 and len(rows) >= max_rows:
                break
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed_rows += 1
                continue
            if not isinstance(row, dict):
                malformed_rows += 1
                continue
            rows.append(row)
    return rows, malformed_rows


def _first_seen_columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name in EXCLUDED_COLUMNS or name in seen:
                continue
            seen.add(name)
            columns.append(name)
    return columns


def _entity_column(columns: list[str]) -> int | None:
    for name in ENTITY_COLUMN_PRIORITY:
        if name in columns:
            return columns.index(name)
    return None


def _source_row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def _context_terms(cells: list[dict[str, Any]], entity_column: int) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for cell in cells:
        if cell["column_index"] == entity_column:
            continue
        text = clean_text(cell.get("text"))
        name = clean_text(cell.get("column_name"))
        for term in (text if text and not is_numeric_text(text) else "", name):
            if term and term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def _empty_result(reason: str, malformed_rows: int) -> WdcTableResult:
    return WdcTableResult(
        source_table=None,
        entities=[],
        image_urls_by_entity={},
        skip_reason=reason,
        malformed_rows=malformed_rows,
    )


def read_wdc_table(
    path: Path,
    input_root: Path,
    min_rows: int,
    min_cols: int,
    max_rows: int = 0,
) -> WdcTableResult:
    """Read one WDC gzip host table and adapt it to the internal table contract."""
    raw_rows, malformed_rows = _read_rows(path, max_rows)
    if len(raw_rows) < min_rows:
        return _empty_result("too_few_rows", malformed_rows)

    column_names = _first_seen_columns(raw_rows)
    if len(column_names) < min_cols:
        return _empty_result("too_few_columns", malformed_rows)

    entity_column = _entity_column(column_names)
    if entity_column is None:
        return _empty_result("missing_entity_column", malformed_rows)

    relative_source = _relative_source(path, input_root)
    schema_class = _schema_class(relative_source)
    source_table_id = f"st_wdc_{stable_hash(schema_class, relative_source, length=16)}"
    columns = [
        {"column_index": index, "column_name": name, "is_numeric_column": False}
        for index, name in enumerate(column_names)
    ]
    rows: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    image_urls_by_entity: dict[str, list[str]] = {}

    for fallback, raw_row in enumerate(raw_rows):
        source_row_id = _source_row_id(raw_row, fallback)
        display_text = _cell_text(raw_row.get(column_names[entity_column]))
        page_url = clean_text(raw_row.get("page_url"))
        entity_key = f"wdc_{stable_hash(schema_class, relative_source, source_row_id, page_url, display_text, length=20)}"
        entity_id = f"ent_{stable_hash(entity_key, length=16)}"
        image_urls = extract_image_urls(raw_row.get("image"), page_url)

        cells: list[dict[str, Any]] = []
        for column_index, column_name in enumerate(column_names):
            raw_value = raw_row.get(column_name)
            is_entity = column_index == entity_column
            cells.append(
                {
                    "column_index": column_index,
                    "column_name": column_name,
                    "raw": raw_value,
                    "text": _cell_text(raw_value),
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
        entities.append(
            {
                "entity_id": entity_id,
                "wiki_title": entity_key,
                "display_texts": [display_text] if display_text else [],
                "context_terms": _context_terms(cells, entity_column),
                "appears_in": [appears_in],
                "page_url": page_url,
                "image_urls": image_urls,
            }
        )
        image_urls_by_entity[entity_id] = image_urls
        rows.append({"row_id": source_row_id, "cells": cells})

    source_table: dict[str, Any] = {
        "source_table_id": source_table_id,
        "source_file": relative_source,
        "page_title": schema_class,
        "caption": "",
        "section_title": "",
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": columns,
        "rows": rows,
        "metadata": {"candidate_entity_columns": [entity_column], "column_profiles": []},
    }
    profiles = column_profiles(source_table)
    source_table["metadata"]["column_profiles"] = [
        {
            "column_index": column["column_index"],
            "column_name": column["column_name"],
            **profiles[column["column_index"]],
        }
        for column in columns
    ]
    for column in columns:
        profile = profiles[column["column_index"]]
        column["is_numeric_column"] = float(profile.get("numeric_ratio", 0.0)) >= 0.8

    return WdcTableResult(
        source_table=source_table,
        entities=entities,
        image_urls_by_entity=image_urls_by_entity,
        skip_reason=None,
        malformed_rows=malformed_rows,
    )

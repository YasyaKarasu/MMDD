from __future__ import annotations

import csv
import gzip
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlsplit

from .tables import column_profiles
from .utils import clean_text, stable_hash


ENTITY_COLUMN_PRIORITY = ("name", "headline", "title", "identifier", "page_url")
EXCLUDED_COLUMNS = {"row_id", "image"}


@dataclass(frozen=True)
class WdcCandidate:
    relative_path: str
    schema_class: str
    subset: str
    host: str
    rows: int | None
    columns: int | None


@dataclass
class AdaptedWdcTable:
    table: dict[str, Any]
    entities: list[dict[str, Any]]


def _statistics_archives(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*_statistics.zip"))


def _candidate_from_statistics(
    archive: Path, input_dir: Path
) -> Iterator[WdcCandidate]:
    schema_class = archive.stem.removesuffix("_statistics")
    with zipfile.ZipFile(archive) as zipped:
        members = set(zipped.namelist())
        for subset in ("top100", "minimum3", "rest"):
            member = (
                f"table_statistics/{schema_class}_October2023"
                f"_statistics_{subset}.csv"
            )
            if member not in members:
                continue
            with zipped.open(member) as raw:
                rows = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8", newline="")
                )
                for row in rows:
                    host = clean_text(row.get("host"))
                    if not host:
                        continue
                    filename = f"{schema_class}_{host}_October2023.json.gz"
                    archive_class_dir = archive.parent.relative_to(input_dir)
                    if archive_class_dir.name == schema_class:
                        relative = archive_class_dir / filename
                    else:
                        relative = Path(schema_class) / filename
                    yield WdcCandidate(
                        relative_path=relative.as_posix(),
                        schema_class=schema_class,
                        subset=subset,
                        host=host,
                        rows=int(row["number_of_rows"]),
                        columns=int(row["column_count"]),
                    )


def iter_candidates(input_dir: Path) -> Iterator[WdcCandidate]:
    """Stream the catalog without enumerating every gzip when statistics exist."""
    archives = _statistics_archives(input_dir)
    if archives:
        for archive in archives:
            yield from _candidate_from_statistics(archive, input_dir)
        return
    for path in iter_gzip_paths(input_dir):
        relative = path.relative_to(input_dir)
        schema_class = (
            relative.parts[0]
            if len(relative.parts) > 1
            else path.name.split("_", 1)[0]
        )
        host = path.name.removeprefix(f"{schema_class}_").removesuffix(
            "_October2023.json.gz"
        )
        yield WdcCandidate(
            relative_path=relative.as_posix(),
            schema_class=schema_class,
            subset="unstratified",
            host=host,
            rows=None,
            columns=None,
        )


def iter_gzip_paths(input_dir: Path) -> Iterator[Path]:
    """Walk lazily and deterministically without building a corpus-wide path list."""
    stack = [input_dir]
    while stack:
        directory = stack.pop()
        entries: list[os.DirEntry[str]] = []
        with os.scandir(directory) as iterator:
            entries.extend(iterator)
        for entry in sorted(entries, key=lambda item: item.name, reverse=True):
            if entry.is_dir(follow_symlinks=False):
                stack.append(Path(entry.path))
        for entry in sorted(entries, key=lambda item: item.name):
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json.gz"):
                yield Path(entry.path)


def iter_rows(path: Path, max_rows: int | None = None) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        emitted = 0
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                continue
            yield value
            emitted += 1
            if max_rows is not None and emitted >= max_rows:
                return


def cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return clean_text(value)


def extract_image_urls(value: Any, base_url: str = "") -> list[str]:
    urls: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            candidate = clean_text(item)
            if not candidate or any(character.isspace() for character in candidate):
                return
            try:
                resolved = urljoin(base_url, candidate)
                parsed = urlsplit(resolved)
            except ValueError:
                return
            if parsed.scheme.lower() in {"http", "https"} and parsed.hostname:
                if resolved not in urls:
                    urls.append(resolved)

    visit(value)
    return urls


def _source_row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def _entity_column(names: list[str]) -> int | None:
    for name in ENTITY_COLUMN_PRIORITY:
        if name in names:
            return names.index(name)
    return None


def _entity_record(
    raw_row: dict[str, Any],
    *,
    fallback: int,
    schema_class: str,
    relative_source: str,
    source_table_id: str,
    entity_column: int,
    column_names: list[str],
) -> dict[str, Any]:
    row_id = _source_row_id(raw_row, fallback)
    display_text = cell_text(raw_row.get(column_names[entity_column]))
    page_url = clean_text(raw_row.get("page_url"))
    entity_key = "wdc_" + stable_hash(
        schema_class, relative_source, row_id, page_url, display_text, length=20
    )
    entity_id = "ent_" + stable_hash(entity_key, length=16)
    return {
        "entity_id": entity_id,
        "wiki_title": entity_key,
        "display_texts": [display_text] if display_text else [],
        "appears_in": [
            {
                "source_table_id": source_table_id,
                "query_view_id": None,
                "row_id": row_id,
                "column_index": entity_column,
                "column_name": column_names[entity_column],
            }
        ],
        "source": "wdc",
        "source_table_id": source_table_id,
        "source_row_id": row_id,
        "page_url": page_url,
        "image_urls": extract_image_urls(raw_row.get("image"), page_url),
    }


def adapt_table(
    path: Path,
    input_dir: Path,
    *,
    min_rows: int,
    min_cols: int,
    max_rows: int | None,
) -> AdaptedWdcTable:
    raw_rows = list(iter_rows(path, max_rows))
    if len(raw_rows) < min_rows:
        raise ValueError("too_few_rows")
    column_names: list[str] = []
    seen: set[str] = set()
    for row in raw_rows:
        for name in row:
            if name not in EXCLUDED_COLUMNS and name not in seen:
                seen.add(name)
                column_names.append(name)
    if len(column_names) < min_cols:
        raise ValueError("too_few_columns")
    entity_column = _entity_column(column_names)
    if entity_column is None:
        raise ValueError("missing_entity_column")

    relative_source = path.relative_to(input_dir).as_posix()
    schema_class = Path(relative_source).parts[0]
    source_table_id = "st_wdc_" + stable_hash(
        schema_class, relative_source, length=16
    )
    columns = [
        {"column_index": index, "column_name": name, "is_numeric_column": False}
        for index, name in enumerate(column_names)
    ]
    rows: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    for fallback, raw_row in enumerate(raw_rows):
        row_id = _source_row_id(raw_row, fallback)
        cells = [
            {
                "column_index": index,
                "column_name": name,
                "raw": raw_row.get(name),
                "text": cell_text(raw_row.get(name)),
                "wiki_title": None,
                "has_wiki_link": index == entity_column,
            }
            for index, name in enumerate(column_names)
        ]
        entity = _entity_record(
            raw_row,
            fallback=fallback,
            schema_class=schema_class,
            relative_source=relative_source,
            source_table_id=source_table_id,
            entity_column=entity_column,
            column_names=column_names,
        )
        cells[entity_column]["wiki_title"] = entity["wiki_title"]
        rows.append({"row_id": row_id, "cells": cells})
        entities.append(entity)

    profiles, _ = column_profiles(rows, columns, 1.0)
    for column, profile in zip(columns, profiles):
        column["is_numeric_column"] = profile["numeric_ratio"] >= 0.8
    table = {
        "source_table_id": source_table_id,
        "source_file": relative_source,
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": columns,
        "rows": rows,
        "provenance_builder": "mmdd_dataset.wdc_adapter",
        "metadata": {
            "candidate_entity_columns": [entity_column],
            "column_profiles": profiles,
        },
    }
    return AdaptedWdcTable(table=table, entities=entities)


def sample_entities(
    path: Path,
    input_dir: Path,
    *,
    count: int,
    seed: int,
    min_rows: int,
    min_cols: int,
    max_rows: int | None,
) -> tuple[list[dict[str, Any]], int, int]:
    column_names: list[str] = []
    seen: set[str] = set()
    row_count = 0
    for raw_row in iter_rows(path, max_rows):
        row_count += 1
        for name in raw_row:
            if name not in EXCLUDED_COLUMNS and name not in seen:
                seen.add(name)
                column_names.append(name)
    if row_count < min_rows:
        raise ValueError("too_few_rows")
    if len(column_names) < min_cols:
        raise ValueError("too_few_columns")
    entity_column = _entity_column(column_names)
    if entity_column is None:
        raise ValueError("missing_entity_column")

    relative_source = path.relative_to(input_dir).as_posix()
    schema_class = Path(relative_source).parts[0]
    source_table_id = "st_wdc_" + stable_hash(
        schema_class, relative_source, length=16
    )
    selected: list[tuple[str, str, dict[str, Any]]] = []
    for fallback, raw_row in enumerate(iter_rows(path, max_rows)):
        entity = _entity_record(
            raw_row,
            fallback=fallback,
            schema_class=schema_class,
            relative_source=relative_source,
            source_table_id=source_table_id,
            entity_column=entity_column,
            column_names=column_names,
        )
        selected.append(
            (
                stable_hash(seed, entity["entity_id"], length=40),
                entity["entity_id"],
                entity,
            )
        )
        if len(selected) > count:
            selected.pop(
                max(range(len(selected)), key=lambda index: selected[index][:2])
            )
    selected.sort(key=lambda item: item[:2])
    return [item[2] for item in selected], row_count, len(column_names)

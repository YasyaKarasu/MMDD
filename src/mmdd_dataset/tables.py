from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mmdd_progress import progress

from .utils import clean_text, stable_hash


WIKI_CELL = re.compile(r"^\[([^\]|]+)(?:\|([^\]]+))?\]$")
NON_ENTITY_NAMESPACES = {
    "category",
    "file",
    "help",
    "image",
    "portal",
    "special",
    "talk",
    "template",
    "user",
    "wikipedia",
}


@dataclass
class PreparedData:
    source_tables: list[dict[str, Any]]
    entities: list[dict[str, Any]]
    skipped: dict[str, int]


def parse_wiki_cell(value: Any) -> dict[str, Any]:
    raw = clean_text(value)
    match = WIKI_CELL.fullmatch(raw)
    if not match:
        return {"raw": value, "text": raw, "wiki_title": None, "has_wiki_link": False}

    title = clean_text(match.group(1)).replace("_", " ")
    text = clean_text(match.group(2) or title).replace("_", " ")
    namespace = title.partition(":")[0].casefold() if ":" in title else ""
    return {
        "raw": value,
        "text": text,
        "wiki_title": None if namespace in NON_ENTITY_NAMESPACES else title,
        "has_wiki_link": True,
    }


def _column_names(raw_names: Any, num_cols: int) -> list[str]:
    names = list(raw_names) if isinstance(raw_names, list) else []
    used: dict[str, int] = {}
    result: list[str] = []
    for index in range(num_cols):
        base = clean_text(names[index]) if index < len(names) else ""
        base = base or f"col_{index}"
        duplicate_index = used.get(base, 0)
        used[base] = duplicate_index + 1
        result.append(base if duplicate_index == 0 else f"{base}_{duplicate_index}")
    return result


def _is_numeric(text: str) -> bool:
    try:
        float(text.replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


def column_profiles(
    rows: list[dict[str, Any]], columns: list[dict[str, Any]], wiki_threshold: float
) -> tuple[list[dict[str, Any]], list[int]]:
    profiles: list[dict[str, Any]] = []
    entity_columns: list[int] = []
    for column in columns:
        index = column["column_index"]
        cells = [row["cells"][index] for row in rows]
        texts = [clean_text(cell.get("text")) for cell in cells]
        non_empty = [text for text in texts if text]
        wiki_count = sum(bool(cell.get("wiki_title")) for cell in cells)
        denominator = max(1, len(rows))
        non_empty_ratio = len(non_empty) / denominator
        wiki_ratio = wiki_count / denominator
        unique_ratio = len(set(non_empty)) / max(1, len(non_empty))
        numeric_ratio = sum(_is_numeric(text) for text in non_empty) / max(1, len(non_empty))
        is_entity = (
            wiki_ratio >= wiki_threshold
            and non_empty_ratio >= 0.5
            and unique_ratio >= 0.2
            and numeric_ratio < 0.5
        )
        if is_entity:
            entity_columns.append(index)
        profiles.append(
            {
                "column_index": index,
                "non_empty_ratio": round(non_empty_ratio, 6),
                "wiki_link_ratio": round(wiki_ratio, 6),
                "unique_ratio": round(unique_ratio, 6),
                "numeric_ratio": round(numeric_ratio, 6),
            }
        )
    return profiles, entity_columns


def _entitable(
    table_id: str,
    raw_table: Any,
    source_file: Path,
    input_dir: Path,
    min_rows: int,
    min_cols: int,
    wiki_threshold: float,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw_table, dict) or not isinstance(raw_table.get("data"), list):
        return None, "missing_data"
    raw_rows = [row for row in raw_table["data"] if isinstance(row, list)]
    if len(raw_rows) < min_rows:
        return None, "too_few_rows"

    raw_titles = raw_table.get("title", [])
    declared_cols = int(raw_table.get("numCols") or 0)
    observed_cols = max((len(row) for row in raw_rows), default=0)
    title_cols = len(raw_titles) if isinstance(raw_titles, list) else 0
    num_cols = max(declared_cols, observed_cols, title_cols)
    if num_cols < min_cols:
        return None, "too_few_columns"

    names = _column_names(raw_titles, num_cols)
    columns = [
        {
            "column_index": index,
            "column_name": name,
        }
        for index, name in enumerate(names)
    ]
    rows: list[dict[str, Any]] = []
    for row_index, raw_row in enumerate(raw_rows):
        cells = []
        for column_index in range(num_cols):
            parsed = parse_wiki_cell(raw_row[column_index] if column_index < len(raw_row) else "")
            cells.append(
                {
                    "column_index": column_index,
                    "column_name": names[column_index],
                    **parsed,
                }
            )
        rows.append({"row_id": row_index, "cells": cells})

    profiles, entity_columns = column_profiles(rows, columns, wiki_threshold)
    relative_source = source_file.relative_to(input_dir).as_posix()
    source_id = f"st_{stable_hash(relative_source, table_id, length=20)}"
    return {
        "source_table_id": source_id,
        "source_file": relative_source,
        "num_rows": len(rows),
        "num_cols": num_cols,
        "columns": columns,
        "rows": rows,
        "metadata": {
            "column_profiles": profiles,
            "candidate_entity_columns": entity_columns,
        },
    }, None


def _entity_occurrences(table: dict[str, Any]) -> Iterable[tuple[str, str, dict[str, Any]]]:
    entity_columns = set(table["metadata"]["candidate_entity_columns"])
    for row in table["rows"]:
        for cell in row["cells"]:
            title = clean_text(cell.get("wiki_title"))
            if title and cell["column_index"] in entity_columns:
                occurrence = {
                    "source_table_id": table["source_table_id"],
                    "row_id": row["row_id"],
                    "column_index": cell["column_index"],
                    "column_name": cell["column_name"],
                }
                yield title, clean_text(cell.get("text")), occurrence


def _entitables_entities(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for table in tables:
        for title, display_text, occurrence in _entity_occurrences(table):
            entity = entities.setdefault(
                title,
                {
                    "entity_id": f"ent_{stable_hash(title)}",
                    "wiki_title": title,
                    "display_texts": [],
                    "appears_in": [],
                    "source": "entitables",
                },
            )
            if display_text and display_text not in entity["display_texts"]:
                entity["display_texts"].append(display_text)
            entity["appears_in"].append(occurrence)
    return list(entities.values())


def prepare_entitables(
    input_dir: Path,
    *,
    min_rows: int = 5,
    min_cols: int = 2,
    wiki_threshold: float = 0.3,
    max_tables: int | None = None,
) -> PreparedData:
    tables: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    paths = sorted(input_dir.rglob("*.json"))
    for path in progress(paths, desc="Load EntiTables", unit="file"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for table_id, raw_table in payload.items():
            table, reason = _entitable(
                str(table_id), raw_table, path, input_dir, min_rows, min_cols, wiki_threshold
            )
            if table is None:
                skipped[reason or "invalid_table"] = skipped.get(reason or "invalid_table", 0) + 1
                continue
            tables.append(table)
            if max_tables and len(tables) >= max_tables:
                return PreparedData(tables, _entitables_entities(tables), skipped)
    return PreparedData(tables, _entitables_entities(tables), skipped)


def prepare_wdc(
    input_dir: Path,
    *,
    min_rows: int = 5,
    min_cols: int = 2,
    max_tables: int | None = None,
    max_rows: int | None = None,
) -> PreparedData:
    """Compatibility wrapper over the standalone WDC adapter."""
    from .wdc_adapter import adapt_table, iter_gzip_paths

    tables: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for path in progress(
        iter_gzip_paths(input_dir), desc="Load WDC tables", unit="table"
    ):
        try:
            adapted = adapt_table(
                path,
                input_dir,
                min_rows=min_rows,
                min_cols=min_cols,
                max_rows=max_rows,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            reason = clean_text(error) or "invalid_table"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        tables.append(adapted.table)
        entities.extend(adapted.entities)
        if max_tables and len(tables) >= max_tables:
            break
    return PreparedData(tables, entities, skipped)

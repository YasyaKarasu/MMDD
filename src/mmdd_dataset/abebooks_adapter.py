"""AbeBooks lake -> the shared builder's table / entity / asset shapes.

This is the third provider, after EntiTables and WDC, and it is shaped like
``wdc_adapter`` on purpose: a corpus with no Wikipedia behind it still has to
fill the shared builder's ``wiki_title`` slot, because that field is the
row -> entity pointer that the entity-column gate (``choose_entity_column``) and
the asset linkage (``entity_to_assets``) read.  WDC puts an opaque
``wdc_<hash>`` there; this puts ``abe_<hash>``.  Nothing contacts Wikipedia, and
the value is only ever compared for equality.

Two properties of the lake make this adapter thinner than WDC's:

* the tables are already in the shared shape -- ``columns``/``rows``/``cells``
  carry the same keys ``_entitable`` produces -- so only the entity column's
  cells and the column profiles need rewriting;
* the bridge assets already exist (text excerpts and cover photographs), so
  nothing is fetched.  They are keyed by ``row_id`` in the lake and have to be
  given an ``entity_id`` here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .tables import PreparedData, column_profiles
from .utils import clean_text, stable_hash

#: How the shared builder knows this corpus mints its entity keys.
ABEBOOKS_BUILDER = "abebooks_mm_joinability_dataset"

#: Row ids in the lake are ``bk_0001`` / ``sl_0001``.  The shared builder keys
#: extraction tasks and join records by an *integer* row id, so the numeric
#: suffix is what travels; the prefix stays in ``key_map.jsonl`` for audit.
ROW_ID_PREFIX = {"bk": "book", "sl": "seller"}


def row_id_to_int(row_id: str) -> int:
    """``bk_0001`` -> ``1``.  Book and seller ids never share a table."""
    _, _, suffix = clean_text(row_id).partition("_")
    if not suffix.isdigit():
        raise ValueError(f"row id {row_id!r} does not end in digits")
    return int(suffix)


def entity_key(source_table_id: str, row_id: str, text: str) -> str:
    """The opaque ``wiki_title`` for one row's entity cell."""
    return "abe_" + stable_hash(source_table_id, row_id, text, length=20)


def entity_id_for(key: str) -> str:
    return "ent_" + stable_hash(key, length=16)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _adapted_table(
    table: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One lake table, with the entity column linked and profiles recomputed."""
    entity_column = int(table["metadata"]["candidate_entity_columns"][0])
    source_table_id = clean_text(table["source_table_id"])
    columns = [
        {"column_index": int(column["column_index"]), "column_name": clean_text(column["column_name"])}
        for column in table["columns"]
    ]
    rows: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    for row in table["rows"]:
        row_id = clean_text(row["row_id"])
        cells = []
        for cell in row["cells"]:
            index = int(cell["column_index"])
            cells.append({
                "column_index": index,
                "column_name": clean_text(cell["column_name"]),
                "raw": cell.get("raw"),
                "text": clean_text(cell.get("text")),
                "wiki_title": None,
                "has_wiki_link": index == entity_column,
            })
        entity_text = cells[entity_column]["text"]
        key = entity_key(source_table_id, row_id, entity_text)
        entity = {
            "entity_id": entity_id_for(key),
            "wiki_title": key,
            "display_texts": [entity_text] if entity_text else [],
            "appears_in": [
                {
                    "source_table_id": source_table_id,
                    "row_id": row_id_to_int(row_id),
                    "column_index": entity_column,
                    "column_name": columns[entity_column]["column_name"],
                }
            ],
            "source": "abebooks",
            "source_table_id": source_table_id,
            "source_row_id": row_id_to_int(row_id),
            "lake_row_id": row_id,
        }
        # The gate reads this field, so it has to be on the cell itself.
        cells[entity_column]["wiki_title"] = key
        rows.append({"row_id": row_id_to_int(row_id), "cells": cells})
        entities.append(entity)
    profiles, _ = column_profiles(rows, columns, 1.0)
    adapted = {
        "source_table_id": source_table_id,
        # Read by the shared builder to decide whether a table may carry the
        # synthetic ``entity_url`` column.  This corpus has no page to point at
        # -- the wiki_title slot holds an opaque key -- so it must not.
        "provenance_builder": ABEBOOKS_BUILDER,
        # ``_entitable`` names this ``source_file``; the lake calls it
        # ``source_name``.
        "source_file": clean_text(table.get("source_file") or table.get("source_name")),
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": columns,
        "rows": rows,
        "metadata": {
            "candidate_entity_columns": [entity_column],
            "column_profiles": profiles,
        },
    }
    return adapted, entities


def prepare_abebooks(
    lake_dir: Path,
    *,
    min_rows: int = 5,
    min_cols: int = 2,
    max_tables: int | None = None,
) -> PreparedData:
    """Read a lake directory into the shared builder's ``PreparedData``."""
    lake_dir = Path(lake_dir)
    if not (lake_dir / "source_tables.jsonl").exists():
        raise FileNotFoundError(
            f"{lake_dir} is not a lake: no source_tables.jsonl in it")
    source_tables: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for table in _read_jsonl(lake_dir / "source_tables.jsonl"):
        if max_tables is not None and len(source_tables) >= max_tables:
            break
        if int(table.get("num_rows") or 0) < min_rows:
            skipped["too_few_rows"] = skipped.get("too_few_rows", 0) + 1
            continue
        if int(table.get("num_cols") or 0) < min_cols:
            skipped["too_few_columns"] = skipped.get("too_few_columns", 0) + 1
            continue
        if not (table.get("metadata", {}).get("candidate_entity_columns") or []):
            skipped["no_candidate_entity_column"] = (
                skipped.get("no_candidate_entity_column", 0) + 1)
            continue
        adapted, table_entities = _adapted_table(table)
        source_tables.append(adapted)
        entities.extend(table_entities)
    return PreparedData(source_tables=source_tables, entities=entities, skipped=skipped)


def adapt_assets(lake_dir: Path, *, root: Path | None = None) -> list[dict[str, Any]]:
    """The lake's bridge assets, linked to the entities their rows produced.

    The shared builder reaches a row's evidence through
    ``entity_to_assets[entity_id]``; the lake keys assets by ``row_id``, so the
    link is rebuilt here.  Image paths in the lake are relative to whatever
    ``--image-root`` the lake was built with (the repository root, in practice)
    and are absolutised against ``root`` because the extraction client opens
    them -- a relative path only resolves from one working directory.
    """
    lake_dir = Path(lake_dir)
    image_root = Path(root) if root is not None else Path.cwd()
    row_to_entity: dict[str, str] = {}
    row_to_table: dict[str, str] = {}
    for table in _read_jsonl(lake_dir / "source_tables.jsonl"):
        source_table_id = clean_text(table["source_table_id"])
        entity_column = int(table["metadata"]["candidate_entity_columns"][0])
        for row in table["rows"]:
            row_id = clean_text(row["row_id"])
            text = clean_text(row["cells"][entity_column].get("text"))
            row_to_entity[row_id] = entity_id_for(
                entity_key(source_table_id, row_id, text))
            row_to_table[row_id] = source_table_id

    assets: list[dict[str, Any]] = []
    for asset in _read_jsonl(lake_dir / "bridge_assets.jsonl"):
        row_id = clean_text(asset.get("row_id"))
        entity_id = row_to_entity.get(row_id)
        if not entity_id:
            continue
        record = dict(asset)
        record["entity_id"] = entity_id
        record["source_table_id"] = row_to_table[row_id]
        record["source_row_id"] = row_id_to_int(row_id)
        local_path = clean_text(record.get("local_path"))
        if local_path:
            path = Path(local_path)
            record["local_path"] = str(path if path.is_absolute() else (image_root / path).resolve())
        assets.append(record)
    return assets


def entities_by_row(entities: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``lake_row_id`` -> entity, for callers holding lake-shaped rows."""
    return {
        clean_text(entity.get("lake_row_id")): entity
        for entity in entities
        if clean_text(entity.get("lake_row_id"))
    }

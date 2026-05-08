#!/usr/bin/env python
"""Leakage-safe object serialization for Qwen3-VL embeddings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from stage1_io import clean_text, compute_profile_from_values, get_cell_text

FORBIDDEN_SERIALIZATION_KEYS = {
    "role",
    "fragment_role",
    "chain_id",
    "source_table_id",
    "hidden_bridge_col",
    "hidden_bridge_col_name",
    "hidden_bridge_values",
    "label",
    "qrel",
    "positive_pair",
    "negative_pair",
    "source_column_indices",
    "source_row_indices",
    "provenance",
    "statement",
    "visible_bridge",
}


def _column_name(column: dict[str, Any], fallback: int) -> str:
    return clean_text(column.get("column_name")) or clean_text(column.get("name")) or f"col_{fallback}"


def _visible_columns(table_like_object: dict[str, Any]) -> list[dict[str, Any]]:
    columns = table_like_object.get("columns")
    if isinstance(columns, list) and columns:
        return [col for col in columns if isinstance(col, dict)]
    rows = table_like_object.get("rows") or []
    width = 0
    if rows and isinstance(rows[0], dict):
        width = len(rows[0].get("cells", []) or [])
    return [{"column_index": i, "column_name": f"col_{i}"} for i in range(width)]


def _row_values(row: dict[str, Any], width: int) -> list[str]:
    values = []
    cells = row.get("cells", [])
    for idx in range(width):
        value = ""
        if isinstance(cells, list):
            if idx < len(cells) and isinstance(cells[idx], dict):
                value = clean_text(cells[idx].get("text"))
            if not value:
                value = clean_text(get_cell_text(row, idx))
        values.append(value)
    return values


def _profile_phrase(name: str, values: list[str], max_values_per_col: int) -> str:
    profile = compute_profile_from_values(values)
    bits: list[str] = []
    if profile["non_empty_ratio"] >= 0.9:
        bits.append("mostly filled")
    elif profile["non_empty_ratio"] >= 0.5:
        bits.append("partly filled")
    else:
        bits.append("sparse")
    if profile["numeric_ratio"] >= 0.8:
        bits.append("numeric values")
    elif profile["unique_ratio"] >= 0.8:
        bits.append("mostly unique text values")
    else:
        bits.append("repeated categorical text values")
    examples = ", ".join(profile["examples"][:max_values_per_col])
    suffix = f"; examples: {examples}" if examples else ""
    return f"- {name}: {', '.join(bits)}{suffix}"


def infer_candidate_entity_columns(table_like_object: dict[str, Any], max_columns: int = 2) -> list[str]:
    columns = _visible_columns(table_like_object)
    rows = table_like_object.get("rows") or []
    candidates: list[tuple[float, str]] = []
    for idx, column in enumerate(columns):
        values = [_row_values(row, len(columns))[idx] for row in rows if isinstance(row, dict)]
        profile = compute_profile_from_values(values)
        score = profile["non_empty_ratio"] + profile["unique_ratio"] - profile["numeric_ratio"]
        name = _column_name(column, idx)
        if profile["non_empty_ratio"] >= 0.5 and profile["numeric_ratio"] <= 0.4:
            candidates.append((score, name))
    candidates.sort(reverse=True)
    return [name for _, name in candidates[:max_columns]]


def serialize_table_for_embedding(
    table_like_object: dict[str, Any],
    max_rows: int = 5,
    max_values_per_col: int = 8,
) -> str:
    """Serialize only online-visible table content.

    The function deliberately ignores provenance, role, qrel, chain, hidden
    bridge, and training-label fields. It is shared by training fragments and
    online query tables to avoid train-test mismatch.
    """

    columns = _visible_columns(table_like_object)
    col_names = [_column_name(col, idx) for idx, col in enumerate(columns)]
    rows = [row for row in table_like_object.get("rows", []) if isinstance(row, dict)]
    row_values = [_row_values(row, len(columns)) for row in rows]

    lines: list[str] = ["[Table Context]"]
    title = clean_text(table_like_object.get("title") or table_like_object.get("page_title"))
    caption = clean_text(table_like_object.get("caption"))
    section = clean_text(table_like_object.get("section_title"))
    if title:
        lines.append(f"Title: {title}")
    if caption:
        lines.append(f"Caption: {caption}")
    if section:
        lines.append(f"Section: {section}")

    lines.append("")
    lines.append("[Columns]")
    for idx, name in enumerate(col_names, 1):
        lines.append(f"{idx}. {name}")

    lines.append("")
    lines.append("[Column Profiles]")
    for idx, name in enumerate(col_names):
        values = [values[idx] for values in row_values if idx < len(values)]
        lines.append(_profile_phrase(name, values, max_values_per_col))

    candidates = infer_candidate_entity_columns(table_like_object)
    if candidates:
        lines.append("")
        lines.append("[Candidate Entity Columns]")
        lines.append(", ".join(candidates))

    lines.append("")
    lines.append("[Example Rows]")
    for row_idx, values in enumerate(row_values[:max_rows], 1):
        assignments = [
            f"{name} = {values[col_idx]}"
            for col_idx, name in enumerate(col_names)
            if col_idx < len(values) and values[col_idx]
        ]
        if assignments:
            lines.append(f"{row_idx}. " + " ; ".join(assignments))

    lines.append("")
    lines.append("[Table Summary]")
    summary_cols = ", ".join(col_names)
    lines.append(f"This is a table with columns {summary_cols} and sample rows shown above.")
    return "\n".join(lines).strip()


def serialize_text_asset_for_embedding(asset: dict[str, Any], max_text_chars: int = 2048) -> str:
    title = clean_text(asset.get("entity_wiki_title"))
    content = clean_text(asset.get("content"))[:max_text_chars]
    parts = ["Entity evidence text."]
    if title:
        parts.append(f"Entity: {title}")
    source = clean_text(asset.get("source"))
    if source:
        parts.append(f"Source: {source}")
    if content:
        parts.append(f"Text: {content}")
    return "\n".join(parts)


def serialize_image_asset_prompt(asset: dict[str, Any]) -> str:
    title = clean_text(asset.get("entity_wiki_title"))
    metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    ext = metadata.get("extmetadata") if isinstance(metadata.get("extmetadata"), dict) else {}
    snippets: list[str] = []
    for key in ("ObjectName", "ImageDescription", "Categories", "Credit"):
        value = ext.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        value = re.sub(r"<[^>]+>", " ", clean_text(value))
        if value:
            snippets.append(f"{key}: {value}")
    file_name = clean_text(asset.get("file_name"))
    lines = [
        "Represent this image as evidence for multimodal table discovery. Focus on what factual attributes about the entity can be inferred from the image."
    ]
    if title:
        lines.append(f"Entity: {title}")
    if file_name:
        lines.append(f"File: {file_name}")
    lines.extend(snippets[:4])
    return "\n".join(lines)


def image_local_path(input_dir: Path, asset: dict[str, Any]) -> Path | None:
    local = clean_text(asset.get("local_path"))
    if local:
        return Path(local)
    relative = clean_text(asset.get("relative_path"))
    if relative:
        return input_dir / relative
    return None

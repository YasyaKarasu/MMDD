#!/usr/bin/env python
"""Build a multimodal table dataset and query workload from EntiTables JSON.

This script constructs dataset artifacts only. It does not create positive or
negative examples, joinability labels, augmentation targets, or query-target
pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import mimetypes
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlparse

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None  # type: ignore[assignment]


MEDIAWIKI_API_URL = "https://en.wikipedia.org/w/api.php"
NON_ENTITY_NAMESPACES = {
    "category",
    "file",
    "image",
    "template",
    "help",
    "portal",
    "wikipedia",
    "special",
    "talk",
    "user",
}
USELESS_IMAGE_PATTERNS = re.compile(
    r"(?:"
    r"commons-logo|wikimedia|wikipedia-logo|edit-icon|OOjs|"
    r"ambox|question_book|symbol_|disambig|stub|"
    r"flag_of|coat_of_arms|icon|logo|placeholder|blank|"
    r"transparent|no_image|map_marker"
    r")",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?%?$")
YEAR_RE = re.compile(r"^(?:1[5-9]\d{2}|20\d{2}|21\d{2})(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?$")
ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)?$", re.IGNORECASE)
SCORE_RE = re.compile(r"^\(?\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*\)?$")
WHITESPACE_RE = re.compile(r"\s+")
PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def normalize_title(title: str) -> str:
    """Normalize a Wikipedia title into a stable display/title form."""
    try:
        value = html.unescape(unquote(str(title or ""))).strip()
        value = value.replace("_", " ")
        value = value.split("#", 1)[0].strip()
        value = WHITESPACE_RE.sub(" ", value)
        if not value:
            return ""
        return value[:1].upper() + value[1:]
    except Exception:
        return str(title or "").strip()


def clean_text(value: Any) -> str:
    try:
        if value is None:
            return ""
        text = html.unescape(str(value)).replace("\xa0", " ").strip()
        return WHITESPACE_RE.sub(" ", text)
    except Exception:
        return ""


def _split_long_text_unit(text: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    rest = clean_text(text)
    while len(rest) > max_chars:
        split_at = rest.rfind(" ", 0, max_chars + 1)
        if split_at < max(1, max_chars // 2):
            split_at = max_chars
        piece = clean_text(rest[:split_at])
        if piece:
            pieces.append(piece)
        rest = clean_text(rest[split_at:])
    if rest:
        pieces.append(rest)
    return pieces


def split_text_asset_content(
    content: Any,
    max_chars: int = 800,
    min_chars: int = 120,
    max_chunks: int = 0,
) -> list[str]:
    """Split a Wikipedia extract into stable text-asset sized fragments."""
    raw = "" if content is None else html.unescape(str(content)).replace("\xa0", " ").strip()
    if not raw:
        return []

    max_chars = max(1, int(max_chars))
    min_chars = max(1, min(int(min_chars), max_chars))
    paragraphs = [clean_text(part) for part in PARAGRAPH_BREAK_RE.split(raw)]
    paragraphs = [part for part in paragraphs if part]
    if not paragraphs:
        paragraphs = [clean_text(raw)]

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in SENTENCE_BREAK_RE.split(paragraph):
            sentence = clean_text(sentence)
            if not sentence:
                continue
            units.extend(_split_long_text_unit(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = unit
    if current:
        chunks.append(current)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chars and len(merged[-1]) + 1 + len(chunk) <= max_chars:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    if max_chunks and max_chunks > 0:
        return merged[:max_chunks]
    return merged


def add_relevance_term(terms: dict[str, float], value: Any, weight: float) -> None:
    term = clean_text(value)
    if len(term) < 2:
        return
    terms[term.casefold()] = max(terms.get(term.casefold(), 0.0), weight)


def entity_text_relevance_terms(entity: dict[str, Any]) -> dict[str, float]:
    terms: dict[str, float] = {}
    add_relevance_term(terms, entity.get("wiki_title"), 5.0)
    for text in entity.get("display_texts") or []:
        add_relevance_term(terms, text, 4.0)
    for term in entity.get("context_terms") or []:
        add_relevance_term(terms, term, 2.0)
    for appearance in entity.get("appears_in") or []:
        if isinstance(appearance, dict):
            add_relevance_term(terms, appearance.get("column_name"), 1.0)
    return terms


def score_text_chunk(chunk: str, terms: dict[str, float]) -> float:
    content = clean_text(chunk).casefold()
    score = 0.0
    for term, weight in terms.items():
        if term and term in content:
            score += weight * content.count(term)
    return score


def select_relevant_text_chunks(
    chunks: list[str],
    entity: dict[str, Any],
    max_chunks: int,
) -> list[tuple[int, str, float]]:
    scored = [
        (idx, chunk, score_text_chunk(chunk, entity_text_relevance_terms(entity)))
        for idx, chunk in enumerate(chunks)
    ]
    if max_chunks and max_chunks > 0:
        scored = sorted(scored, key=lambda item: (-item[2], item[0]))[:max_chunks]
    return sorted(scored, key=lambda item: item[0])


def parse_wiki_cell(cell: str) -> dict[str, Any]:
    """Parse a cell that may contain EntiTables-style Wikipedia markup.

    Supported formats:
    - [Page_Title|Display Text]
    - [Page_Title]
    - plain text

    Links in non-entity namespaces such as Category:, File:, and Image: are
    detected as links but are not treated as entity links by default.
    """
    raw = "" if cell is None else str(cell)
    try:
        raw_clean = clean_text(raw)
        match = re.fullmatch(r"\[([^\[\]\|]+)(?:\|([^\[\]]*))?\]", raw_clean)
        if not match:
            return {
                "raw": raw,
                "text": clean_text(raw_clean),
                "wiki_title": None,
                "has_wiki_link": False,
            }

        page = normalize_title(match.group(1))
        label = clean_text(match.group(2) if match.group(2) is not None else page)
        namespace = page.split(":", 1)[0].lower() if ":" in page else ""
        wiki_title = None if namespace in NON_ENTITY_NAMESPACES else page
        return {
            "raw": raw,
            "text": label,
            "wiki_title": wiki_title or None,
            "has_wiki_link": True,
        }
    except Exception:
        return {
            "raw": raw,
            "text": clean_text(raw),
            "wiki_title": None,
            "has_wiki_link": False,
        }


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def slug(value: str, max_len: int = 48) -> str:
    text = normalize_title(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len] or "item"


def iter_with_progress(items: list[Any], desc: str) -> Iterable[Any]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    if hasattr(handle, "write_record"):
        handle.write_record(record)
        return
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ShardedJsonlWriter:
    """Write JSONL records into bounded-size part files."""

    def __init__(self, output_dir: Path, max_records_per_shard: int, prefix: str = "part") -> None:
        self.output_dir = output_dir
        self.max_records_per_shard = max(1, max_records_per_shard)
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handle: Any | None = None
        self._current_count = 0
        self._shard_index = -1
        self.total_records = 0
        self.shards: list[dict[str, Any]] = []

    def __enter__(self) -> "ShardedJsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _open_next_shard(self) -> None:
        self.close()
        self._shard_index += 1
        self._current_count = 0
        shard_path = self.output_dir / f"{self.prefix}-{self._shard_index:05d}.jsonl"
        self._handle = shard_path.open("w", encoding="utf-8")
        self.shards.append({"path": shard_path, "records": 0})

    def write_record(self, record: dict[str, Any]) -> None:
        if self._handle is None or self._current_count >= self.max_records_per_shard:
            self._open_next_shard()
        assert self._handle is not None
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._current_count += 1
        self.total_records += 1
        self.shards[-1]["records"] = self._current_count

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def paths(self) -> list[Path]:
        return [shard["path"] for shard in self.shards if shard["records"] > 0]

    def manifest(self, root: Path) -> dict[str, Any]:
        return {
            "directory": str(self.output_dir.relative_to(root)).replace("\\", "/"),
            "total_records": self.total_records,
            "max_records_per_shard": self.max_records_per_shard,
            "shards": [
                {
                    "path": str(shard["path"].relative_to(root)).replace("\\", "/"),
                    "records": shard["records"],
                }
                for shard in self.shards
                if shard["records"] > 0
            ],
        }


def write_sharded_jsonl(
    output_dir: Path,
    records: Iterable[dict[str, Any]],
    max_records_per_shard: int,
) -> ShardedJsonlWriter:
    writer = ShardedJsonlWriter(output_dir, max_records_per_shard)
    with writer:
        for record in records:
            writer.write_record(record)
    return writer


def iter_jsonl_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_jsonl_cache(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = record.get(key_field)
                if key:
                    cache[str(key)] = record
            except json.JSONDecodeError:
                continue
    return cache


def is_numeric_text(text: str) -> bool:
    value = clean_text(text)
    if not value:
        return False
    if NUMERIC_RE.fullmatch(value):
        return True
    try:
        float(value.replace(",", "").rstrip("%"))
        return True
    except ValueError:
        return False


def looks_like_non_entity_column(column_name: str, values: list[str]) -> bool:
    non_empty = [clean_text(v) for v in values if clean_text(v)]
    if not non_empty:
        return True

    lowered_name = column_name.lower()
    if any(
        token in lowered_name
        for token in (
            "year",
            "date",
            "rank",
            "no.",
            "number",
            "score",
            "rating",
            "time",
            "age",
            "price",
            "population",
            "height",
            "weight",
            "length",
        )
    ):
        pattern_hits = sum(
            1
            for value in non_empty
            if YEAR_RE.fullmatch(value)
            or ORDINAL_RE.fullmatch(value)
            or SCORE_RE.fullmatch(value)
            or is_numeric_text(value)
        )
        if pattern_hits / max(1, len(non_empty)) >= 0.65:
            return True

    year_hits = sum(1 for value in non_empty if YEAR_RE.fullmatch(value))
    score_hits = sum(1 for value in non_empty if SCORE_RE.fullmatch(value))
    ordinal_hits = sum(1 for value in non_empty if ORDINAL_RE.fullmatch(value))
    return (
        year_hits / len(non_empty) >= 0.8
        or score_hits / len(non_empty) >= 0.8
        or ordinal_hits / len(non_empty) >= 0.8
    )


def normalize_column_names(raw_titles: list[Any], num_cols: int) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    seen: Counter[str] = Counter()
    for idx in range(num_cols):
        raw_name = clean_text(raw_titles[idx]) if idx < len(raw_titles) else ""
        parsed_header = parse_wiki_cell(raw_name)
        base = clean_text(parsed_header["text"])
        if not base:
            base = f"col_{idx}"
        if len(base) > 80:
            base = base[:77].rstrip() + "..."

        current_count = seen[base]
        seen[base] += 1
        column_name = base if current_count == 0 else f"{base}_{current_count}"
        columns.append(
            {
                "column_index": idx,
                "column_name": column_name,
                "raw_column_name": raw_name,
                "is_numeric_column": False,
            }
        )
    return columns


def normalize_numeric_columns(raw_numeric_columns: Any, columns: list[dict[str, Any]]) -> set[int]:
    numeric_indices: set[int] = set()
    if not isinstance(raw_numeric_columns, list):
        return numeric_indices

    names = {col["raw_column_name"]: col["column_index"] for col in columns}
    names.update({col["column_name"]: col["column_index"] for col in columns})
    for item in raw_numeric_columns:
        if isinstance(item, int) and 0 <= item < len(columns):
            numeric_indices.add(item)
        elif isinstance(item, str):
            if item.isdigit() and 0 <= int(item) < len(columns):
                numeric_indices.add(int(item))
            elif item in names:
                numeric_indices.add(names[item])
    return numeric_indices


def normalize_rows(data: Any, num_cols: int) -> tuple[list[list[Any]], float]:
    if not isinstance(data, list):
        return [], 1.0

    normalized: list[list[Any]] = []
    inconsistent = 0
    for row in data:
        if not isinstance(row, list):
            inconsistent += 1
            continue
        if len(row) != num_cols:
            inconsistent += 1
        if len(row) < num_cols:
            row = row + [""] * (num_cols - len(row))
        elif len(row) > num_cols:
            row = row[:num_cols]
        normalized.append(row)

    bad_ratio = inconsistent / max(1, len(data))
    return normalized, bad_ratio


def compute_column_profiles(
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    wiki_link_threshold: float,
) -> tuple[list[dict[str, Any]], list[int]]:
    profiles: list[dict[str, Any]] = []
    candidate_entity_columns: list[int] = []
    row_count = len(rows)

    for column in columns:
        idx = column["column_index"]
        cells = [row["cells"][idx] for row in rows if idx < len(row["cells"])]
        texts = [clean_text(cell.get("text", "")) for cell in cells]
        non_empty_texts = [text for text in texts if text]
        non_empty_ratio = len(non_empty_texts) / max(1, row_count)
        wiki_link_ratio = sum(1 for cell in cells if cell.get("wiki_title")) / max(1, row_count)
        unique_ratio = len(set(non_empty_texts)) / max(1, len(non_empty_texts))
        numeric_ratio = sum(1 for text in non_empty_texts if is_numeric_text(text)) / max(
            1, len(non_empty_texts)
        )
        avg_text_length = sum(len(text) for text in non_empty_texts) / max(1, len(non_empty_texts))
        is_candidate = (
            wiki_link_ratio >= wiki_link_threshold
            and non_empty_ratio >= 0.5
            and numeric_ratio <= 0.4
            and unique_ratio >= 0.1
            and not looks_like_non_entity_column(column["column_name"], texts)
        )
        profile = {
            "column_index": idx,
            "column_name": column["column_name"],
            "wiki_link_ratio": round(wiki_link_ratio, 6),
            "non_empty_ratio": round(non_empty_ratio, 6),
            "unique_ratio": round(unique_ratio, 6),
            "numeric_ratio": round(numeric_ratio, 6),
            "avg_text_length": round(avg_text_length, 6),
            "is_candidate_entity_column": bool(is_candidate),
        }
        profiles.append(profile)
        if is_candidate:
            candidate_entity_columns.append(idx)
    return profiles, candidate_entity_columns


@dataclass
class ParsedTableResult:
    source_table: dict[str, Any] | None
    skip_reason: str | None = None


def parse_source_table(
    table_id: str,
    table_obj: Any,
    source_file: Path,
    input_dir: Path,
    min_rows: int,
    min_cols: int,
    wiki_link_threshold: float,
) -> ParsedTableResult:
    if not isinstance(table_obj, dict):
        return ParsedTableResult(None, "table_object_not_dict")

    data = table_obj.get("data")
    if not data:
        return ParsedTableResult(None, "empty_data")

    try:
        declared_cols = int(table_obj.get("numCols") or 0)
    except (TypeError, ValueError):
        declared_cols = 0
    title = table_obj.get("title") if isinstance(table_obj.get("title"), list) else []
    inferred_cols = max([len(row) for row in data if isinstance(row, list)] + [len(title), declared_cols])
    num_cols = declared_cols or inferred_cols
    if num_cols < min_cols:
        return ParsedTableResult(None, "too_few_columns")

    try:
        num_data_rows = int(table_obj.get("numDataRows") or 0)
    except (TypeError, ValueError):
        num_data_rows = 0
    if num_data_rows == 0:
        return ParsedTableResult(None, "numDataRows_zero")

    normalized_data, bad_row_ratio = normalize_rows(data, num_cols)
    if bad_row_ratio > 0.3:
        return ParsedTableResult(None, "row_column_count_too_inconsistent")
    if len(normalized_data) < min_rows:
        return ParsedTableResult(None, "too_few_rows")

    total_cells = len(normalized_data) * num_cols
    non_empty_cells = sum(1 for row in normalized_data for cell in row if clean_text(cell))
    if total_cells == 0:
        return ParsedTableResult(None, "empty_table")
    if non_empty_cells / total_cells < 0.2:
        return ParsedTableResult(None, "low_non_empty_cell_ratio")

    columns = normalize_column_names(title, num_cols)
    numeric_indices = normalize_numeric_columns(table_obj.get("numericColumns"), columns)
    for column in columns:
        column["is_numeric_column"] = column["column_index"] in numeric_indices

    rows: list[dict[str, Any]] = []
    for row_idx, raw_row in enumerate(normalized_data):
        cells = []
        for col_idx, raw_cell in enumerate(raw_row):
            parsed = parse_wiki_cell(raw_cell)
            cells.append(
                {
                    "column_index": col_idx,
                    "column_name": columns[col_idx]["column_name"],
                    "raw": parsed["raw"],
                    "text": parsed["text"],
                    "wiki_title": parsed["wiki_title"],
                    "has_wiki_link": parsed["has_wiki_link"],
                }
            )
        rows.append({"row_id": row_idx, "cells": cells})

    column_profiles, candidate_entity_columns = compute_column_profiles(
        rows, columns, wiki_link_threshold
    )
    relative_source = str(source_file.relative_to(input_dir)) if source_file.is_relative_to(input_dir) else source_file.name
    source_table_id = f"st_{slug(table_id, 32)}_{stable_hash(relative_source, table_id, length=10)}"
    source_table = {
        "source_table_id": source_table_id,
        "source_file": relative_source.replace("\\", "/"),
        "page_title": clean_text(table_obj.get("pgTitle")),
        "caption": clean_text(table_obj.get("caption")),
        "section_title": clean_text(table_obj.get("secondTitle")),
        "num_rows": len(rows),
        "num_cols": num_cols,
        "columns": columns,
        "rows": rows,
        "metadata": {
            "column_profiles": column_profiles,
            "candidate_entity_columns": candidate_entity_columns,
        },
    }
    return ParsedTableResult(source_table)


def profile_by_index(source_table: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(profile["column_index"]): profile
        for profile in source_table["metadata"].get("column_profiles", [])
    }


def build_query_view(
    source_table: dict[str, Any],
    selected_indices: list[int],
    derivation_type: str,
    ordinal: int,
) -> dict[str, Any]:
    selected_set = set(selected_indices)
    selected_columns = [source_table["columns"][idx] for idx in selected_indices]
    hidden_columns = [
        column for column in source_table["columns"] if column["column_index"] not in selected_set
    ]
    rows: list[dict[str, Any]] = []
    for row in source_table["rows"]:
        rows.append(
            {
                "row_id": row["row_id"],
                "cells": [row["cells"][idx] for idx in selected_indices if idx < len(row["cells"])],
            }
        )

    source_candidates = source_table["metadata"].get("candidate_entity_columns", [])
    candidates_in_view = [idx for idx in selected_indices if idx in source_candidates]
    query_view_id = (
        f"qv_{source_table['source_table_id']}_{ordinal:03d}_"
        f"{slug(derivation_type, 32)}_{stable_hash(selected_indices, length=8)}"
    )
    return {
        "query_view_id": query_view_id,
        "source_table_id": source_table["source_table_id"],
        "source_file": source_table["source_file"],
        "page_title": source_table["page_title"],
        "caption": source_table["caption"],
        "derivation_type": derivation_type,
        "selected_column_indices": selected_indices,
        "selected_column_names": [column["column_name"] for column in selected_columns],
        "hidden_column_indices": [column["column_index"] for column in hidden_columns],
        "hidden_column_names": [column["column_name"] for column in hidden_columns],
        "rows": rows,
        "metadata": {
            "candidate_entity_columns_in_view": candidates_in_view,
            "source_candidate_entity_columns": source_candidates,
            "note": "hidden_columns are provenance only, not labels",
        },
    }


def generate_query_views(
    source_table: dict[str, Any],
    max_query_views_per_source_table: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_query_views_per_source_table <= 0:
        return []

    num_cols = source_table["num_cols"]
    if num_cols == 0:
        return []

    profiles = profile_by_index(source_table)
    candidate_cols = list(source_table["metadata"].get("candidate_entity_columns", []))
    if not candidate_cols:
        return []

    rng = random.Random(f"{seed}:{source_table['source_table_id']}")
    attr_cols = [
        idx
        for idx in range(num_cols)
        if idx not in candidate_cols
        and profiles.get(idx, {}).get("non_empty_ratio", 0.0) >= 0.5
        and profiles.get(idx, {}).get("numeric_ratio", 0.0) <= 0.95
    ]
    rng.shuffle(candidate_cols)
    rng.shuffle(attr_cols)

    plans: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def add_plan(kind: str, indices: list[int]) -> None:
        key = tuple(indices)
        if key not in seen and 0 < len(indices) <= min(4, num_cols):
            seen.add(key)
            plans.append((kind, indices))

    for entity_col in candidate_cols:
        for attr_col in attr_cols:
            add_plan("entity_plus_one_attr", [entity_col, attr_col])

    for entity_col in candidate_cols:
        for left_pos in range(len(attr_cols)):
            for right_pos in range(left_pos + 1, len(attr_cols)):
                add_plan("entity_plus_two_attrs", [entity_col, attr_cols[left_pos], attr_cols[right_pos]])

    for entity_col in candidate_cols:
        add_plan("entity_only", [entity_col])

    random_attempts = 0
    all_indices = list(range(num_cols))
    while (
        len(plans) < max_query_views_per_source_table * 3
        and random_attempts < max(20, max_query_views_per_source_table * 10)
        and num_cols >= 2
    ):
        random_attempts += 1
        width = rng.randint(2, min(4, num_cols))
        selected = sorted(rng.sample(all_indices, width))
        if not any(idx in candidate_cols for idx in selected):
            selected[rng.randrange(width)] = rng.choice(candidate_cols)
            selected = sorted(set(selected))
        if len(selected) >= 2:
            add_plan("random_projection", selected)

    query_views: list[dict[str, Any]] = []
    for ordinal, (kind, indices) in enumerate(plans[:max_query_views_per_source_table], start=1):
        query_views.append(build_query_view(source_table, indices, kind, ordinal))
    return query_views


def collect_entities(
    source_tables: list[dict[str, Any]],
    query_views: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entity_records: dict[str, dict[str, Any]] = {}
    wiki_to_entity_id: dict[str, str] = {}

    def add_cell(
        source_table_id: str,
        query_view_id: str | None,
        row_id: int,
        cell: dict[str, Any],
    ) -> None:
        wiki_title = cell.get("wiki_title")
        if not wiki_title:
            return
        normalized = normalize_title(wiki_title)
        entity_id = wiki_to_entity_id.setdefault(
            normalized, f"ent_{stable_hash(normalized, length=16)}"
        )
        if entity_id not in entity_records:
            entity_records[entity_id] = {
                "entity_id": entity_id,
                "wiki_title": normalized,
                "display_texts": set(),
                "appears_in": [],
            }
        record = entity_records[entity_id]
        if clean_text(cell.get("text")):
            record["display_texts"].add(clean_text(cell.get("text")))
        record["appears_in"].append(
            {
                "source_table_id": source_table_id,
                "query_view_id": query_view_id,
                "row_id": row_id,
                "column_index": cell.get("column_index"),
                "column_name": cell.get("column_name"),
            }
        )

    for table in source_tables:
        for row in table["rows"]:
            for cell in row["cells"]:
                add_cell(table["source_table_id"], None, row["row_id"], cell)

    for view in query_views:
        for row in view["rows"]:
            for cell in row["cells"]:
                add_cell(view["source_table_id"], view["query_view_id"], row["row_id"], cell)

    entities: list[dict[str, Any]] = []
    for record in entity_records.values():
        entities.append(
            {
                "entity_id": record["entity_id"],
                "wiki_title": record["wiki_title"],
                "display_texts": sorted(record["display_texts"]),
                "appears_in": record["appears_in"],
            }
        )
    entities.sort(key=lambda item: item["wiki_title"])
    return entities, wiki_to_entity_id


def add_entity_cell(
    entity_records: dict[str, dict[str, Any]],
    wiki_to_entity_id: dict[str, str],
    source_table_id: str,
    query_view_id: str | None,
    row_id: int,
    cell: dict[str, Any],
    row: dict[str, Any] | None = None,
) -> None:
    wiki_title = cell.get("wiki_title")
    if not wiki_title:
        return
    normalized = normalize_title(wiki_title)
    entity_id = wiki_to_entity_id.setdefault(normalized, f"ent_{stable_hash(normalized, length=16)}")
    if entity_id not in entity_records:
        entity_records[entity_id] = {
            "entity_id": entity_id,
            "wiki_title": normalized,
            "display_texts": set(),
            "context_terms": Counter(),
            "appears_in": [],
        }
    record = entity_records[entity_id]
    if clean_text(cell.get("text")):
        record["display_texts"].add(clean_text(cell.get("text")))
    record["appears_in"].append(
        {
            "source_table_id": source_table_id,
            "query_view_id": query_view_id,
            "row_id": row_id,
            "column_index": cell.get("column_index"),
            "column_name": cell.get("column_name"),
        }
    )
    if row:
        for context_cell in row.get("cells", []):
            if context_cell is cell:
                continue
            context_text = clean_text(context_cell.get("text"))
            context_name = clean_text(context_cell.get("column_name"))
            if context_text and not is_numeric_text(context_text):
                record["context_terms"][context_text] += 1
            if context_name:
                record["context_terms"][context_name] += 1


def update_entities_from_table(
    entity_records: dict[str, dict[str, Any]],
    wiki_to_entity_id: dict[str, str],
    source_table: dict[str, Any],
) -> None:
    for row in source_table["rows"]:
        for cell in row["cells"]:
            add_entity_cell(
                entity_records,
                wiki_to_entity_id,
                source_table["source_table_id"],
                None,
                row["row_id"],
                cell,
                row,
            )


def update_entities_from_query_view(
    entity_records: dict[str, dict[str, Any]],
    wiki_to_entity_id: dict[str, str],
    query_view: dict[str, Any],
) -> None:
    for row in query_view["rows"]:
        for cell in row["cells"]:
            add_entity_cell(
                entity_records,
                wiki_to_entity_id,
                query_view["source_table_id"],
                query_view["query_view_id"],
                row["row_id"],
                cell,
                row,
            )


def finalize_entities(entity_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for record in entity_records.values():
        entities.append(
            {
                "entity_id": record["entity_id"],
                "wiki_title": record["wiki_title"],
                "display_texts": sorted(record["display_texts"]),
                "context_terms": [
                    term for term, _count in record.get("context_terms", Counter()).most_common(80)
                ],
                "appears_in": record["appears_in"],
            }
        )
    entities.sort(key=lambda item: item["wiki_title"])
    return entities


class WikipediaClient:
    """Small MediaWiki Action API client with JSONL cache."""

    def __init__(
        self,
        cache_dir: Path,
        image_output_dir: Path,
        output_dir: Path,
        sleep: float,
        user_agent: str,
    ) -> None:
        if requests is None:
            raise RuntimeError("requests is required unless --no_wikipedia is used")
        self.cache_dir = cache_dir
        self.image_output_dir = image_output_dir
        self.output_dir = output_dir
        self.sleep = max(0.0, sleep)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.page_cache_path = cache_dir / "wiki_pages.jsonl"
        self.image_cache_path = cache_dir / "wiki_images.jsonl"
        self.page_cache = load_jsonl_cache(self.page_cache_path, "wiki_title")
        self.image_cache = load_jsonl_cache(self.image_cache_path, "file_title")
        self.last_request_time = 0.0
        self.api_failures = 0

    def _wait(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.sleep:
            time.sleep(self.sleep - elapsed)

    def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        self._wait()
        try:
            response = self.session.get(MEDIAWIKI_API_URL, params=params, timeout=30)
            self.last_request_time = time.time()
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            self.api_failures += 1
            logging.warning("MediaWiki API request failed: %s", exc)
            return None

    def get_page(self, wiki_title: str) -> dict[str, Any] | None:
        normalized = normalize_title(wiki_title)
        if not normalized:
            return None
        if normalized in self.page_cache:
            return self.page_cache[normalized]

        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "titles": normalized,
            "prop": "extracts|pageimages|images|info",
            "explaintext": 1,
            "piprop": "thumbnail|original|name",
            "pithumbsize": 600,
            "imlimit": 50,
            "inprop": "url",
        }
        payload = self._get(params)
        if not payload:
            return None
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        record = {
            "wiki_title": normalized,
            "pageid": page.get("pageid"),
            "title": page.get("title", normalized),
            "extract": page.get("extract", ""),
            "canonicalurl": page.get("canonicalurl")
            or f"https://en.wikipedia.org/wiki/{quote(normalized.replace(' ', '_'))}",
            "pageimage": page.get("pageimage"),
            "thumbnail": page.get("thumbnail"),
            "original": page.get("original"),
            "images": page.get("images", []),
            "missing": bool(page.get("missing")),
        }
        self.page_cache[normalized] = record
        append_jsonl(self.page_cache_path, record)
        return record

    def get_imageinfo(self, file_title: str) -> dict[str, Any] | None:
        normalized = normalize_title(file_title)
        if not normalized:
            return None
        if not normalized.lower().startswith("file:"):
            normalized = f"File:{normalized}"
        if normalized in self.image_cache:
            return self.image_cache[normalized]

        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
            "titles": normalized,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|mediatype|extmetadata",
        }
        payload = self._get(params)
        if not payload:
            return None
        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        infos = page.get("imageinfo") or []
        info = infos[0] if infos else {}
        record = {
            "file_title": normalized,
            "pageid": page.get("pageid"),
            "url": info.get("url"),
            "descriptionurl": info.get("descriptionurl"),
            "mime": info.get("mime"),
            "mediatype": info.get("mediatype"),
            "width": info.get("width"),
            "height": info.get("height"),
            "size": info.get("size"),
            "extmetadata": info.get("extmetadata") or {},
            "missing": bool(page.get("missing")),
        }
        self.image_cache[normalized] = record
        append_jsonl(self.image_cache_path, record)
        return record

    def download_image(self, imageinfo: dict[str, Any], asset_id: str) -> dict[str, Any] | None:
        url = imageinfo.get("url")
        if not url:
            return None

        extension = infer_image_extension(imageinfo)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.image_output_dir / f"{asset_id}{extension}"
        if image_path.exists():
            return file_download_record(image_path, self.output_dir, downloaded=False)

        tmp_path = image_path.with_suffix(image_path.suffix + ".tmp")
        digest = hashlib.sha256()
        total_bytes = 0
        self._wait()
        try:
            with self.session.get(url, stream=True, timeout=60) as response:
                self.last_request_time = time.time()
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if content_type and not content_type.lower().startswith("image/"):
                    logging.warning("Skipping non-image response for %s: %s", url, content_type)
                    return None
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 128):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        total_bytes += len(chunk)
            tmp_path.replace(image_path)
            record = file_download_record(image_path, self.output_dir, downloaded=True)
            record["sha256"] = digest.hexdigest()
            record["bytes"] = total_bytes
            return record
        except Exception as exc:
            self.api_failures += 1
            logging.warning("Image download failed for %s: %s", url, exc)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            return None


def infer_image_extension(imageinfo: dict[str, Any]) -> str:
    mime = str(imageinfo.get("mime") or "").lower()
    extension = mimetypes.guess_extension(mime) if mime else None
    if extension == ".jpe":
        extension = ".jpg"
    if not extension:
        parsed_suffix = Path(urlparse(str(imageinfo.get("url") or "")).path).suffix.lower()
        extension = parsed_suffix if re.fullmatch(r"\.[a-z0-9]{2,5}", parsed_suffix) else ".img"
    return extension


def file_download_record(path: Path, output_dir: Path, downloaded: bool) -> dict[str, Any]:
    stat = path.stat()
    try:
        relative_path = path.relative_to(output_dir)
    except ValueError:
        relative_path = path
    return {
        "local_path": str(path),
        "relative_path": str(relative_path).replace("\\", "/"),
        "file_name": path.name,
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
        "downloaded": downloaded,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_useful_image(file_title: str, imageinfo: dict[str, Any] | None = None) -> bool:
    if not file_title:
        return False
    name = normalize_title(file_title)
    if USELESS_IMAGE_PATTERNS.search(name):
        return False
    if name.lower().endswith(".svg") and re.search(r"(icon|logo|flag|symbol)", name, re.I):
        return False
    if imageinfo:
        mediatype = str(imageinfo.get("mediatype") or "").upper()
        mime = str(imageinfo.get("mime") or "").lower()
        width = int(imageinfo.get("width") or 0)
        height = int(imageinfo.get("height") or 0)
        extmetadata = imageinfo.get("extmetadata") or {}
        metadata_text = " ".join(
            clean_text(value.get("value"))
            for value in extmetadata.values()
            if isinstance(value, dict)
        )
        if mediatype and mediatype not in {"BITMAP", "DRAWING"}:
            return False
        if 0 < width <= 96 and 0 < height <= 96:
            return False
        if mime == "image/svg+xml" and re.search(r"(icon|logo|flag|symbol)", name, re.I):
            return False
        if USELESS_IMAGE_PATTERNS.search(metadata_text):
            return False
    return True


def build_bridge_assets(
    entities: list[dict[str, Any]],
    max_entities: int | None,
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    wikipedia_client: WikipediaClient | None,
    asset_writer: ShardedJsonlWriter,
    flush_every_records: int,
) -> tuple[dict[str, list[str]], int, int, int]:
    if wikipedia_client is None:
        return defaultdict(list), 0, 0, 0

    entity_to_assets: dict[str, list[str]] = defaultdict(list)
    selected_entities = entities[:max_entities] if max_entities else entities
    text_asset_count = 0
    image_asset_count = 0
    written_assets = 0

    for entity in iter_with_progress(selected_entities, "Fetching Wikipedia assets"):
        page = wikipedia_client.get_page(entity["wiki_title"])
        if not page or page.get("missing"):
            continue

        text_chunks = split_text_asset_content(
            page.get("extract"),
            max_chars=text_asset_chunk_chars,
            min_chars=min_text_asset_chunk_chars,
            max_chunks=0,
        )
        selected_text_chunks = select_relevant_text_chunks(
            text_chunks,
            entity,
            max_text_asset_chunks_per_entity,
        )
        source_asset_id = f"asset_text_{stable_hash(entity['entity_id'], 'extract')}"
        for chunk_index, chunk, chunk_score in selected_text_chunks:
            asset_id = f"{source_asset_id}_{chunk_index:03d}"
            write_jsonl_record(
                asset_writer,
                {
                    "asset_id": asset_id,
                    "source_asset_id": source_asset_id,
                    "entity_id": entity["entity_id"],
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "text",
                    "content": chunk,
                    "text_chunk_index": chunk_index,
                    "text_chunk_count": len(text_chunks),
                    "selected_text_chunk_count": len(selected_text_chunks),
                    "text_chunk_relevance_score": round(chunk_score, 6),
                    "source": "wikipedia_extract_chunk",
                    "url": page.get("canonicalurl")
                    or f"https://en.wikipedia.org/wiki/{quote(entity['wiki_title'].replace(' ', '_'))}",
                },
            )
            entity_to_assets[entity["entity_id"]].append(asset_id)
            text_asset_count += 1
            written_assets += 1
            if flush_every_records > 0 and written_assets % flush_every_records == 0:
                asset_writer.flush()

        image_titles: list[str] = []
        if page.get("pageimage"):
            image_titles.append(f"File:{page['pageimage']}")
        for image in page.get("images") or []:
            title = image.get("title") if isinstance(image, dict) else None
            if title:
                image_titles.append(title)

        seen_images: set[str] = set()
        kept = 0
        for image_title in image_titles:
            normalized_title = normalize_title(image_title)
            if normalized_title in seen_images or not is_useful_image(normalized_title):
                continue
            seen_images.add(normalized_title)
            imageinfo = wikipedia_client.get_imageinfo(normalized_title)
            if not imageinfo or not imageinfo.get("url"):
                continue
            if not is_useful_image(normalized_title, imageinfo):
                continue
            asset_id = f"asset_img_{stable_hash(entity['entity_id'], normalized_title)}"
            downloaded_image = wikipedia_client.download_image(imageinfo, asset_id)
            if downloaded_image is None:
                continue
            write_jsonl_record(
                asset_writer,
                {
                    "asset_id": asset_id,
                    "entity_id": entity["entity_id"],
                    "entity_wiki_title": entity["wiki_title"],
                    "asset_type": "image",
                    "image_url": imageinfo.get("url"),
                    "description_url": imageinfo.get("descriptionurl"),
                    "local_path": downloaded_image["local_path"],
                    "relative_path": downloaded_image["relative_path"],
                    "file_name": downloaded_image["file_name"],
                    "bytes": downloaded_image["bytes"],
                    "sha256": downloaded_image["sha256"],
                    "metadata": {
                        "file_title": imageinfo.get("file_title"),
                        "mime": imageinfo.get("mime"),
                        "mediatype": imageinfo.get("mediatype"),
                        "width": imageinfo.get("width"),
                        "height": imageinfo.get("height"),
                        "size": imageinfo.get("size"),
                        "extmetadata": imageinfo.get("extmetadata") or {},
                        "downloaded": downloaded_image["downloaded"],
                    },
                    "source": "wikipedia_image_download",
                },
            )
            entity_to_assets[entity["entity_id"]].append(asset_id)
            image_asset_count += 1
            written_assets += 1
            kept += 1
            if flush_every_records > 0 and written_assets % flush_every_records == 0:
                asset_writer.flush()
            if kept >= max_images_per_entity:
                break
    asset_writer.flush()

    return entity_to_assets, wikipedia_client.api_failures, text_asset_count, image_asset_count


def build_table_asset_links(
    source_tables: list[dict[str, Any]],
    query_views: list[dict[str, Any]],
    wiki_to_entity_id: dict[str, str],
    entity_to_assets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []

    def add_link(
        source_table_id: str,
        query_view_id: str | None,
        row_id: int,
        cell: dict[str, Any],
    ) -> None:
        wiki_title = cell.get("wiki_title")
        if not wiki_title:
            return
        normalized = normalize_title(wiki_title)
        entity_id = wiki_to_entity_id.get(normalized)
        if not entity_id:
            return
        link_id = f"link_{stable_hash(source_table_id, query_view_id, row_id, cell.get('column_index'), entity_id)}"
        links.append(
            {
                "link_id": link_id,
                "source_table_id": source_table_id,
                "query_view_id": query_view_id,
                "row_id": row_id,
                "column_index": cell.get("column_index"),
                "column_name": cell.get("column_name"),
                "cell_text": cell.get("text"),
                "entity_id": entity_id,
                "entity_wiki_title": normalized,
                "asset_ids": list(entity_to_assets.get(entity_id, [])),
            }
        )

    for table in source_tables:
        for row in table["rows"]:
            for cell in row["cells"]:
                add_link(table["source_table_id"], None, row["row_id"], cell)

    for view in query_views:
        for row in view["rows"]:
            for cell in row["cells"]:
                add_link(view["source_table_id"], view["query_view_id"], row["row_id"], cell)
    return links


def write_table_asset_links_from_jsonl(
    source_table_paths: list[Path],
    query_view_paths: list[Path],
    link_writer: ShardedJsonlWriter,
    wiki_to_entity_id: dict[str, str],
    entity_to_assets: dict[str, list[str]],
    flush_every_records: int,
) -> int:
    link_count = 0

    def emit_link(
        source_table_id: str,
        query_view_id: str | None,
        row_id: int,
        cell: dict[str, Any],
    ) -> None:
        nonlocal link_count
        wiki_title = cell.get("wiki_title")
        if not wiki_title:
            return
        normalized = normalize_title(wiki_title)
        entity_id = wiki_to_entity_id.get(normalized)
        if not entity_id:
            return
        link_id = f"link_{stable_hash(source_table_id, query_view_id, row_id, cell.get('column_index'), entity_id)}"
        write_jsonl_record(
            link_writer,
            {
                "link_id": link_id,
                "source_table_id": source_table_id,
                "query_view_id": query_view_id,
                "row_id": row_id,
                "column_index": cell.get("column_index"),
                "column_name": cell.get("column_name"),
                "cell_text": cell.get("text"),
                "entity_id": entity_id,
                "entity_wiki_title": normalized,
                "asset_ids": list(entity_to_assets.get(entity_id, [])),
            },
        )
        link_count += 1
        if flush_every_records > 0 and link_count % flush_every_records == 0:
            link_writer.flush()

    for record in iter_jsonl_records(source_table_paths):
        source_table_id = record["source_table_id"]
        for row in record["rows"]:
            for cell in row["cells"]:
                emit_link(source_table_id, None, row["row_id"], cell)

    for record in iter_jsonl_records(query_view_paths):
        source_table_id = record["source_table_id"]
        query_view_id = record["query_view_id"]
        for row in record["rows"]:
            for cell in row["cells"]:
                emit_link(source_table_id, query_view_id, row["row_id"], cell)

    link_writer.flush()
    return link_count


def make_splits(
    source_tables: list[dict[str, Any]],
    query_views: list[dict[str, Any]],
    split_by: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    total_ratio = train_ratio + dev_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train/dev/test ratios must sum to a positive value")
    train_ratio, dev_ratio, test_ratio = (
        train_ratio / total_ratio,
        dev_ratio / total_ratio,
        test_ratio / total_ratio,
    )

    groups: dict[str, list[str]] = defaultdict(list)
    for table in source_tables:
        if split_by == "page_title":
            key = clean_text(table.get("page_title")) or table["source_table_id"]
        else:
            key = table["source_table_id"]
        groups[key].append(table["source_table_id"])

    group_keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    train_cut = math.floor(n_groups * train_ratio)
    dev_cut = train_cut + math.floor(n_groups * dev_ratio)
    if n_groups >= 3:
        train_cut = max(1, train_cut)
        dev_cut = max(train_cut + 1, dev_cut)
        dev_cut = min(dev_cut, n_groups - 1)

    split_group_keys = {
        "train": group_keys[:train_cut],
        "dev": group_keys[train_cut:dev_cut],
        "test": group_keys[dev_cut:],
    }

    views_by_source: dict[str, list[str]] = defaultdict(list)
    for view in query_views:
        views_by_source[view["source_table_id"]].append(view["query_view_id"])

    result: dict[str, Any] = {}
    for split_name, keys in split_group_keys.items():
        source_ids = sorted({source_id for key in keys for source_id in groups[key]})
        query_ids = sorted(
            query_view_id for source_id in source_ids for query_view_id in views_by_source[source_id]
        )
        result[split_name] = {
            "source_table_ids": source_ids,
            "query_view_ids": query_ids,
        }
    result["split_key"] = "page_title_or_source_table_id" if split_by == "page_title" else "source_table_id"
    result["note"] = "splits are source-level to avoid leakage across query views"
    return result


def make_splits_from_index(
    source_split_records: list[dict[str, str]],
    views_by_source: dict[str, list[str]],
    split_by: str,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    total_ratio = train_ratio + dev_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train/dev/test ratios must sum to a positive value")
    train_ratio, dev_ratio, test_ratio = (
        train_ratio / total_ratio,
        dev_ratio / total_ratio,
        test_ratio / total_ratio,
    )

    groups: dict[str, list[str]] = defaultdict(list)
    for record in source_split_records:
        if split_by == "page_title":
            key = clean_text(record.get("page_title")) or record["source_table_id"]
        else:
            key = record["source_table_id"]
        groups[key].append(record["source_table_id"])

    group_keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    train_cut = math.floor(n_groups * train_ratio)
    dev_cut = train_cut + math.floor(n_groups * dev_ratio)
    if n_groups >= 3:
        train_cut = max(1, train_cut)
        dev_cut = max(train_cut + 1, dev_cut)
        dev_cut = min(dev_cut, n_groups - 1)

    split_group_keys = {
        "train": group_keys[:train_cut],
        "dev": group_keys[train_cut:dev_cut],
        "test": group_keys[dev_cut:],
    }

    result: dict[str, Any] = {}
    for split_name, keys in split_group_keys.items():
        source_ids = sorted({source_id for key in keys for source_id in groups[key]})
        query_ids = sorted(
            query_view_id for source_id in source_ids for query_view_id in views_by_source[source_id]
        )
        result[split_name] = {
            "source_table_ids": source_ids,
            "query_view_ids": query_ids,
        }
    result["split_key"] = "page_title_or_source_table_id" if split_by == "page_title" else "source_table_id"
    result["note"] = "splits are source-level to avoid leakage across query views"
    return result


def read_entitables_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            logging.warning("Skipping %s: top-level JSON is not a dict", path)
            return None
        return payload
    except Exception as exc:
        logging.warning("Skipping malformed JSON file %s: %s", path, exc)
        return None


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_tables_dir = output_dir / "source_tables"
    query_views_dir = output_dir / "query_views"
    entities_dir = output_dir / "entities"
    bridge_assets_dir = output_dir / "bridge_assets"
    table_asset_links_dir = output_dir / "table_asset_links"

    entity_records: dict[str, dict[str, Any]] = {}
    wiki_to_entity_id: dict[str, str] = {}
    source_split_records: list[dict[str, str]] = []
    views_by_source: dict[str, list[str]] = defaultdict(list)
    skip_reasons: Counter[str] = Counter()
    processed_tables = 0
    skipped_tables = 0
    source_table_count = 0
    query_view_count = 0
    query_view_rows_sum = 0
    query_view_cols_sum = 0

    json_files = sorted(input_dir.rglob("*.json"))
    logging.info("Found %d JSON files under %s", len(json_files), input_dir)

    stop = False
    flush_every = max(1, args.flush_every_records)
    records_per_shard = max(1, args.records_per_shard)
    source_writer = ShardedJsonlWriter(source_tables_dir, records_per_shard)
    query_writer = ShardedJsonlWriter(query_views_dir, records_per_shard)
    with source_writer as source_handle, query_writer as query_handle:
        for json_file in iter_with_progress(json_files, "Reading EntiTables JSON"):
            if stop:
                break
            payload = read_entitables_json(json_file)
            if payload is None:
                skipped_tables += 1
                skip_reasons["malformed_json_file"] += 1
                continue
            for table_id, table_obj in payload.items():
                if args.max_tables is not None and processed_tables >= args.max_tables:
                    stop = True
                    break
                processed_tables += 1
                result = parse_source_table(
                    str(table_id),
                    table_obj,
                    json_file,
                    input_dir,
                    args.min_rows,
                    args.min_cols,
                    args.wiki_link_threshold,
                )
                if result.source_table is None:
                    skipped_tables += 1
                    reason = result.skip_reason or "unknown"
                    skip_reasons[reason] += 1
                    logging.info("Skipped table %s in %s: %s", table_id, json_file, reason)
                    continue

                source_table = result.source_table
                write_jsonl_record(source_handle, source_table)
                source_table_count += 1
                source_split_records.append(
                    {
                        "source_table_id": source_table["source_table_id"],
                        "page_title": source_table.get("page_title") or "",
                    }
                )
                update_entities_from_table(entity_records, wiki_to_entity_id, source_table)

                generated_views = generate_query_views(
                    source_table,
                    args.max_query_views_per_source_table,
                    args.seed,
                )
                for query_view in generated_views:
                    write_jsonl_record(query_handle, query_view)
                    query_view_count += 1
                    query_view_rows_sum += len(query_view["rows"])
                    query_view_cols_sum += len(query_view["selected_column_indices"])
                    views_by_source[query_view["source_table_id"]].append(query_view["query_view_id"])
                    update_entities_from_query_view(entity_records, wiki_to_entity_id, query_view)

                if source_table_count % flush_every == 0:
                    source_handle.flush()
                    query_handle.flush()
                    logging.info(
                        "Flushed %d source tables and %d query views",
                        source_table_count,
                        query_view_count,
                    )
        source_handle.flush()
        query_handle.flush()

    entities = finalize_entities(entity_records)

    wikipedia_client: WikipediaClient | None = None
    if not args.no_wikipedia:
        wikipedia_client = WikipediaClient(
            cache_dir=output_dir / "cache",
            image_output_dir=output_dir / "images",
            output_dir=output_dir,
            sleep=args.sleep,
            user_agent=(
                "MMTableDatasetBuilder/0.1 "
                "(https://example.invalid; research dataset construction)"
            ),
        )
    bridge_assets_writer = ShardedJsonlWriter(bridge_assets_dir, records_per_shard)
    with bridge_assets_writer:
        entity_to_assets, api_failures, text_asset_count, image_asset_count = build_bridge_assets(
            entities,
            args.max_entities,
            args.max_images_per_entity,
            args.text_asset_chunk_chars,
            args.min_text_asset_chunk_chars,
            args.max_text_asset_chunks_per_entity,
            wikipedia_client,
            bridge_assets_writer,
            flush_every,
        )

    table_asset_links_writer = ShardedJsonlWriter(table_asset_links_dir, records_per_shard)
    with table_asset_links_writer:
        table_asset_link_count = write_table_asset_links_from_jsonl(
            source_writer.paths(),
            query_writer.paths(),
            table_asset_links_writer,
            wiki_to_entity_id,
            entity_to_assets,
            flush_every,
        )
    splits = make_splits_from_index(
        source_split_records,
        views_by_source,
        args.split_by,
        args.train_ratio,
        args.dev_ratio,
        args.test_ratio,
        args.seed,
    )

    entities_writer = write_sharded_jsonl(entities_dir, entities, records_per_shard)
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, ensure_ascii=False, indent=2)

    stats = {
        "processed_tables": processed_tables,
        "skipped_tables": skipped_tables,
        "source_tables": source_table_count,
        "query_views": query_view_count,
        "unique_wiki_entities": len(entities),
        "text_assets": text_asset_count,
        "image_assets": image_asset_count,
        "table_asset_links": table_asset_link_count,
        "api_failures": api_failures,
        "avg_query_views_per_source_table": round(query_view_count / max(1, source_table_count), 6),
        "avg_rows_per_query_view": round(query_view_rows_sum / max(1, query_view_count), 6),
        "avg_columns_per_query_view": round(query_view_cols_sum / max(1, query_view_count), 6),
        "flush_every_records": flush_every,
        "records_per_shard": records_per_shard,
        "text_asset_chunk_chars": args.text_asset_chunk_chars,
        "min_text_asset_chunk_chars": args.min_text_asset_chunk_chars,
        "max_text_asset_chunks_per_entity": args.max_text_asset_chunks_per_entity,
        "skipped_reasons": dict(skip_reasons),
        "notes": [
            "query_views are query workload tables, not labels",
            "hidden_columns are provenance only, not augmentation targets",
            "no positive/negative pairs or joinability labels are generated",
            "large JSONL artifacts are written incrementally into sharded part files",
        ],
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    manifest = {
        "format": "sharded_jsonl",
        "records_per_shard": records_per_shard,
        "artifacts": {
            "source_tables": source_writer.manifest(output_dir),
            "query_views": query_writer.manifest(output_dir),
            "entities": entities_writer.manifest(output_dir),
            "bridge_assets": bridge_assets_writer.manifest(output_dir),
            "table_asset_links": table_asset_links_writer.manifest(output_dir),
        },
        "single_files": {
            "splits": "splits.json",
            "stats": "stats.json",
        },
        "note": "Read shards listed in this manifest; stale files from older runs may exist if an output directory is reused.",
    }
    with (output_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multimodal table dataset and query workload from EntiTables JSON. "
            "This script does not build joinability benchmark labels."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing EntiTables .json files.")
    parser.add_argument("--output_dir", required=True, help="Directory where dataset artifacts are written.")
    parser.add_argument("--max_tables", type=int, default=None, help="Maximum number of raw tables to process.")
    parser.add_argument(
        "--max_query_views_per_source_table",
        type=int,
        default=5,
        help="Maximum query workload views generated from each source table.",
    )
    parser.add_argument(
        "--max_entities",
        type=int,
        default=None,
        help="Maximum number of entities for Wikipedia asset fetching. Entity extraction is not limited.",
    )
    parser.add_argument("--max_images_per_entity", type=int, default=3)
    parser.add_argument(
        "--text_asset_chunk_chars",
        type=int,
        default=800,
        help="Maximum characters per Wikipedia extract text asset chunk.",
    )
    parser.add_argument(
        "--min_text_asset_chunk_chars",
        type=int,
        default=120,
        help="Prefer merging trailing Wikipedia text chunks shorter than this when possible.",
    )
    parser.add_argument(
        "--max_text_asset_chunks_per_entity",
        type=int,
        default=3,
        help="Maximum relevant Wikipedia text chunks per entity. Use 0 for no limit.",
    )
    parser.add_argument("--wiki_link_threshold", type=float, default=0.3)
    parser.add_argument("--min_rows", type=int, default=2)
    parser.add_argument("--min_cols", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between API requests.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--flush_every_records",
        type=int,
        default=500,
        help="Flush streaming JSONL writers after this many source/link/asset records.",
    )
    parser.add_argument(
        "--records_per_shard",
        type=int,
        default=50000,
        help="Maximum records per JSONL shard file before starting the next part file.",
    )
    parser.add_argument(
        "--no_wikipedia",
        action="store_true",
        help="Skip MediaWiki API calls. entities.jsonl and table_asset_links.jsonl are still produced.",
    )
    parser.add_argument(
        "--split_by",
        choices=["source_table_id", "page_title"],
        default="page_title",
        help="Source-level grouping key for train/dev/test split.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        stats = build_dataset(args)
    except Exception as exc:
        logging.exception("Dataset build failed: %s", exc)
        return 1

    logging.info(
        "Done: %d source tables, %d query views, %d entities, %d asset links",
        stats["source_tables"],
        stats["query_views"],
        stats["unique_wiki_entities"],
        stats["table_asset_links"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

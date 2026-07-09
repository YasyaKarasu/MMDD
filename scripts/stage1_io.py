#!/usr/bin/env python
"""Shared utilities for stage-1 logic connectivity data construction."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

LOG = logging.getLogger("stage1")
WHITESPACE_RE = re.compile(r"\s+")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?%?$")
URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
USELESS_COLUMN_NAMES = {
    "no",
    "no.",
    "#",
    "rank",
    "index",
    "ref",
    "reference",
    "note",
    "notes",
    "remarks",
    "source",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value)).replace("\xa0", " ").strip()
    return WHITESPACE_RE.sub(" ", text)


def sanitize_cell_text_for_model(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if not URL_RE.search(text):
        return text
    without_urls = URL_RE.sub(" ", text)
    without_urls = WHITESPACE_RE.sub(" ", without_urls).strip(" ,;:-|()[]{}")
    return without_urls or "[url]"


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def is_numeric_text(value: Any) -> bool:
    text = clean_text(value).replace(",", "")
    if not text:
        return False
    if NUMERIC_RE.match(text):
        return True
    try:
        float(text.rstrip("%"))
        return True
    except ValueError:
        return False


def normalize_key(value: Any) -> str:
    return clean_text(value).casefold()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def iter_jsonl_paths(paths: Iterable[Path], log_every: int = 50000) -> Iterator[dict[str, Any]]:
    count = 0
    for path in paths:
        LOG.info("Reading %s", path)
        for record in iter_jsonl(path):
            count += 1
            if log_every > 0 and count % log_every == 0:
                LOG.info("Read %s records", f"{count:,}")
            yield record


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def manifest_artifact_paths(input_dir: Path, artifact: str) -> list[Path]:
    manifest_path = input_dir / "dataset_manifest.json"
    manifest = load_json(manifest_path)
    item = manifest.get("artifacts", {}).get(artifact)
    if not item:
        raise KeyError(f"Artifact {artifact!r} not found in {manifest_path}")
    paths = [input_dir / shard["path"] for shard in item.get("shards", [])]
    if not paths:
        raise ValueError(f"Artifact {artifact!r} has no shards in {manifest_path}")
    return paths


def iter_manifest_records(input_dir: Path, artifact: str, log_every: int = 50000) -> Iterator[dict[str, Any]]:
    return iter_jsonl_paths(manifest_artifact_paths(input_dir, artifact), log_every=log_every)


def load_split_map(input_dir: Path) -> dict[str, str]:
    path = input_dir / "splits.json"
    data = load_json(path)
    split_map: dict[str, str] = {}
    for split, payload in data.items():
        if not isinstance(payload, dict):
            continue
        for source_id in payload.get("source_table_ids", []):
            split_map[str(source_id)] = split
    return split_map


def update_stage1_manifest(stage1_dir: Path, section: str, payload: dict[str, Any]) -> None:
    path = stage1_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        manifest = load_json(path)
    manifest.setdefault("created_or_updated_unix", int(time.time()))
    manifest["last_updated_unix"] = int(time.time())
    manifest[section] = payload
    write_json(path, manifest)


def get_column_name(table: dict[str, Any], idx: int) -> str:
    for column in table.get("columns", []):
        try:
            if int(column.get("column_index", -1)) == int(idx):
                return clean_text(column.get("column_name")) or f"col_{idx}"
        except (TypeError, ValueError):
            continue
    return f"col_{idx}"


def get_cell(row: dict[str, Any], col_idx: int) -> dict[str, Any]:
    cells = row.get("cells", [])
    if isinstance(cells, list):
        for cell in cells:
            try:
                if int(cell.get("column_index", -1)) == int(col_idx):
                    return cell
            except (TypeError, ValueError, AttributeError):
                continue
        if 0 <= col_idx < len(cells) and isinstance(cells[col_idx], dict):
            return cells[col_idx]
    return {"column_index": col_idx, "column_name": f"col_{col_idx}", "text": ""}


def get_cell_text(row: dict[str, Any], col_idx: int) -> str:
    return clean_text(get_cell(row, col_idx).get("text"))


def row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def column_values(table: dict[str, Any], col_idx: int) -> list[str]:
    return [get_cell_text(row, col_idx) for row in table.get("rows", [])]


def compute_profile_from_values(values: list[str]) -> dict[str, Any]:
    total = len(values)
    non_empty = [clean_text(v) for v in values if clean_text(v)]
    distinct = set(non_empty)
    return {
        "non_empty_ratio": len(non_empty) / max(1, total),
        "unique_ratio": len(distinct) / max(1, len(non_empty)),
        "numeric_ratio": sum(1 for value in non_empty if is_numeric_text(value)) / max(1, len(non_empty)),
        "distinct_count": len(distinct),
        "examples": list(dict.fromkeys(non_empty))[:8],
    }


def column_profiles(table: dict[str, Any]) -> dict[int, dict[str, Any]]:
    profiles: dict[int, dict[str, Any]] = {}
    for profile in table.get("metadata", {}).get("column_profiles", []) or []:
        try:
            profiles[int(profile["column_index"])] = dict(profile)
        except (KeyError, TypeError, ValueError):
            continue
    for column in table.get("columns", []):
        try:
            idx = int(column.get("column_index"))
        except (TypeError, ValueError):
            continue
        if idx not in profiles:
            profiles[idx] = compute_profile_from_values(column_values(table, idx))
        else:
            computed = compute_profile_from_values(column_values(table, idx))
            profiles[idx].setdefault("non_empty_ratio", computed["non_empty_ratio"])
            profiles[idx].setdefault("unique_ratio", computed["unique_ratio"])
            profiles[idx].setdefault("numeric_ratio", computed["numeric_ratio"])
            profiles[idx].setdefault("distinct_count", computed["distinct_count"])
            profiles[idx].setdefault("examples", computed["examples"])
    return profiles


def is_useless_column_name(name: str) -> bool:
    key = clean_text(name).lower().strip(" .:")
    return key in USELESS_COLUMN_NAMES


def fd_purity(rows: list[dict[str, Any]], x_col: int, y_col: int) -> tuple[float, int]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    support = 0
    for row in rows:
        x_val = normalize_key(get_cell_text(row, x_col))
        y_val = normalize_key(get_cell_text(row, y_col))
        if not x_val or not y_val:
            continue
        buckets[x_val][y_val] += 1
        support += 1
    if support == 0:
        return 0.0, 0
    winners = sum(counter.most_common(1)[0][1] for counter in buckets.values())
    return winners / support, support


def dominant_by_key(rows: list[dict[str, Any]], key_col: int, value_col: int) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = clean_text(get_cell_text(row, key_col))
        value = clean_text(get_cell_text(row, value_col))
        if key and value:
            grouped[key][value] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in grouped.items() if counter}


def project_rows(
    source_table: dict[str, Any],
    column_indices: list[int],
    dedupe_col: int | None = None,
    min_required_cols: int | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    output_rows: list[dict[str, Any]] = []
    source_row_indices: list[int] = []
    seen: set[str] = set()
    required = set(column_indices if min_required_cols is None else column_indices[:min_required_cols])
    for fallback, row in enumerate(source_table.get("rows", [])):
        values = {idx: get_cell_text(row, idx) for idx in column_indices}
        if any(not values.get(idx) for idx in required):
            continue
        if dedupe_col is not None:
            key = normalize_key(values.get(dedupe_col))
            if not key or key in seen:
                continue
            seen.add(key)
        cells = []
        for out_idx, source_idx in enumerate(column_indices):
            original = dict(get_cell(row, source_idx))
            original["column_index"] = out_idx
            original["source_column_index"] = source_idx
            original["column_name"] = get_column_name(source_table, source_idx)
            original["text"] = sanitize_cell_text_for_model(values.get(source_idx, ""))
            cells.append(original)
        output_rows.append({"row_id": len(output_rows), "source_row_id": row_id(row, fallback), "cells": cells})
        source_row_indices.append(row_id(row, fallback))
    return output_rows, source_row_indices


def make_columns(source_table: dict[str, Any], column_indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "column_index": out_idx,
            "source_column_index": source_idx,
            "column_name": get_column_name(source_table, source_idx),
        }
        for out_idx, source_idx in enumerate(column_indices)
    ]


def l2_normalize_array(array: Any, eps: float = 1e-12) -> Any:
    import numpy as np

    arr = np.asarray(array, dtype="float32")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return arr / norm


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

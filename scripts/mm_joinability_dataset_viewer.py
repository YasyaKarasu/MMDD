#!/usr/bin/env python
"""Flask GUI for browsing build_mm_joinability_dataset outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

import ijson
from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for
from ijson.common import ObjectBuilder

from stage1_gui import format_gui_urls, resolve_gui_host
from stage1_io import (
    clean_text,
    iter_jsonl,
    load_json,
)


LOG = logging.getLogger(__name__)
VIEWER_INDEX_SCHEMA_VERSION = "mm-joinability-viewer-index-v2"
DEFAULT_INDEX_FILENAME = ".mm_joinability_viewer.sqlite3"
IMPLICIT_JOIN_REASON = "model_recoverable_join_column"
JSONL_SCAN_CHUNK_BYTES = 1024 * 1024
JSONL_ID_PREFIX_BYTES = 64 * 1024

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MM Joinability Dataset Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d6dce5;
      --accent: #1f6feb;
      --soft: #eef2f7;
      --green: #0b7a46;
      --amber: #8a5a00;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 14px 22px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    h2 { margin: 0 0 10px; font-size: 16px; }
    h3 { margin: 0 0 4px; font-size: 14px; }
    main { width: min(1560px, 100%); margin: 0 auto; padding: 18px 22px 36px; }
    form, .pager, .filters, .summary, .row, .path-head { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    input, select, button, a.button {
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 10px;
      font: inherit;
      text-decoration: none;
    }
    button, a.button { cursor: pointer; display: inline-flex; align-items: center; }
    button.primary, a.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .muted { color: var(--muted); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .summary { justify-content: space-between; align-items: flex-start; }
    .kv { display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; width: 100%; }
    .label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }
    .value { overflow-wrap: anywhere; font-weight: 650; }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--soft);
      color: #344054;
      font-size: 12px;
      font-weight: 650;
    }
    .badge.path { background: #e7f7ef; color: var(--green); }
    .badge.join { background: #fff4d6; color: var(--amber); }
    .flow {
      display: grid;
      grid-template-columns: minmax(260px, 0.95fr) 80px minmax(260px, 1fr) 80px minmax(260px, 0.95fr);
      gap: 10px;
      align-items: stretch;
    }
    .node {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 10px;
      min-width: 0;
    }
    .arrow { align-self: center; justify-self: center; color: var(--muted); font-size: 22px; }
    .tables {
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 14px;
      align-items: start;
    }
    .table-box { min-width: 0; overflow: auto; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
    th, td { border: 1px solid var(--line); padding: 5px 6px; vertical-align: top; overflow-wrap: anywhere; }
    th { background: #eef2f7; text-align: left; }
    tr.highlight td { background: #fff7e0; }
    .paths { display: grid; gap: 10px; }
    .path {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 10px;
    }
    .path-head { justify-content: space-between; align-items: flex-start; }
    .path-grid {
      display: grid;
      grid-template-columns: minmax(240px, 0.8fr) minmax(320px, 1.2fr);
      gap: 12px;
      margin-top: 8px;
    }
    .asset-box {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      min-width: 0;
    }
    .asset-text {
      max-height: 260px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #344054;
      margin-top: 6px;
    }
    .asset-image {
      display: block;
      max-width: 100%;
      max-height: 320px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      margin-top: 8px;
    }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .pager { justify-content: center; margin: 16px 0 4px; }
    .pager input { width: 88px; }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .kv, .tables, .path-grid, .flow { grid-template-columns: 1fr; }
      .arrow { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MM Joinability Dataset Viewer</h1>
      <div class="muted">{{ total }} query-target pairs{% if query %} matching "{{ query }}"{% endif %}</div>
    </div>
    <form method="get" action="{{ url_for('index') }}">
      <input name="q" value="{{ query }}" placeholder="query, target, attribute, asset">
      <select name="split">
        <option value="">all splits</option>
        {% for s in splits %}
        <option value="{{ s }}" {{ "selected" if split == s else "" }}>{{ s }}</option>
        {% endfor %}
      </select>
      <select name="row_view">
        <option value="all" {{ "selected" if row_view == "all" else "" }}>all row views</option>
        <option value="canonical" {{ "selected" if row_view == "canonical" else "" }}>canonical views</option>
        <option value="augmented" {{ "selected" if row_view == "augmented" else "" }}>augmented views</option>
      </select>
      <select name="asset_type">
        <option value="">all assets</option>
        {% for t in asset_types %}
        <option value="{{ t }}" {{ "selected" if asset_type == t else "" }}>{{ t }}</option>
        {% endfor %}
      </select>
      <button class="primary" type="submit">Filter</button>
      <a class="button" href="{{ url_for('reload') }}">Reload</a>
    </form>
  </header>

  <main>
    {% if error %}
    <section class="panel"><strong>Cannot load dataset.</strong><div class="muted">{{ error }}</div></section>
    {% elif pair %}
    <section class="panel summary">
      <div class="kv">
        <div><span class="label">Query Table</span><span class="value">{{ pair.query_table_id }}</span></div>
        <div><span class="label">Target Table</span><span class="value">{{ pair.target_table_id }}</span></div>
        <div><span class="label">Source Table</span><span class="value">{{ pair.source_table_id }}</span></div>
        <div><span class="label">Split</span><span class="value">{{ pair.split }}</span></div>
        <div><span class="label">Chain</span><span class="value">{{ pair.chain_id }}</span></div>
        <div><span class="label">Row View</span><span class="value">{{ pair.row_view_kind }} · {{ pair.row_view_number }} of {{ pair.row_view_count }}</span></div>
        <div><span class="label">Relevance</span><span class="value">{{ pair.rel }}</span></div>
        <div><span class="label">Join Attribute</span><span class="value">{{ pair.join_attribute.column_name or pair.join_col_name }}</span></div>
        <div><span class="label">Evidence Paths</span><span class="value">{{ pair.path_count }}</span></div>
      </div>
    </section>

    <section class="panel">
      <div class="flow">
        <div class="node">
          <span class="badge">query</span>
          <h3>{{ pair.query_table.page_title or pair.query_table_id }}</h3>
          <div class="muted">{{ pair.query_table.caption }}</div>
          <div>{{ pair.query_table.columns|join(", ") }}</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <span class="badge path">material path</span>
          <h3>{{ pair.path_count }} recoveries</h3>
          <div class="muted">{{ pair.asset_type_summary }}</div>
          <div>{{ pair.reason }}</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <span class="badge">target</span>
          <h3>{{ pair.target_table.page_title or pair.target_table_id }}</h3>
          <div class="muted">{{ pair.target_table.caption }}</div>
          <div>{{ pair.target_table.columns|join(", ") }}</div>
        </div>
      </div>
    </section>

    <section class="tables">
      <article class="panel table-box">
        <h2>Query Table</h2>
        <div class="muted">{{ pair.query_table.role }} · {{ pair.query_table.rows|length }} displayed rows{% if pair.query_table.truncated %} · truncated{% endif %}</div>
        {{ render_table(pair.query_table, pair.query_highlight_rows) | safe }}
      </article>
      <article class="panel table-box">
        <h2>Target Table</h2>
        <div class="muted">{{ pair.target_table.role }} · {{ pair.target_table.rows|length }} displayed rows{% if pair.target_table.truncated %} · truncated{% endif %}</div>
        {{ render_table(pair.target_table, pair.target_highlight_rows) | safe }}
      </article>
    </section>

    <section class="panel">
      <h2>Path Materials</h2>
      <div class="paths">
        {% for path in pair.paths %}
        <div class="path">
          <div class="path-head">
            <div>
              <strong>{{ path.recovered_attribute.column_name }} = {{ path.recovered_attribute.value }}</strong>
              <div class="muted">{{ path.path_id }}</div>
            </div>
            <div class="row">
              <span class="badge">{{ path.asset_type }}</span>
              <span class="badge join">query row {{ path.query_row_id }} → target rows {{ path.target_row_ids|join(", ") }}</span>
            </div>
          </div>
          <div class="path-grid">
            <div class="asset-box">
              <span class="label">Entity</span>
              <div><strong>{{ path.query_entity.cell_text or path.query_entity.wiki_title }}</strong></div>
              <div class="muted">{{ path.query_entity.wiki_title }}</div>
              <hr>
              <div><span class="label">Model Value</span>{{ path.recovered_attribute.model_value }}</div>
              <div><span class="label">Expected Value</span>{{ path.recovered_attribute.value }}</div>
              <div><span class="label">Model Evidence</span>{{ path.model_evidence }}</div>
            </div>
            <div class="asset-box">
              <div class="path-head">
                <div>
                  <strong>{{ path.asset_title or path.asset_id }}</strong>
                  <div class="muted">{{ path.asset_id }}</div>
                </div>
                <span class="badge">{{ path.asset_source or "asset" }}{% if path.asset_chunk_label %} · {{ path.asset_chunk_label }}{% endif %}</span>
              </div>
              {% if path.asset_url %}
              <div><a href="{{ path.asset_url }}" target="_blank" rel="noreferrer">{{ path.asset_url }}</a></div>
              {% endif %}
              {% if path.asset_type == "text" %}
              <div class="asset-text">{{ path.asset_content or "No text content available." }}</div>
              {% elif path.asset_type == "image" %}
              <div class="muted">{{ path.asset_file_name or path.asset_local_path or "No image filename available." }}</div>
              {% if path.asset_image_available %}
              <img class="asset-image" src="{{ url_for('asset_image', asset_id=path.asset_id) }}" alt="{{ path.asset_title or path.asset_id }}">
              {% else %}
              <div class="muted">Image file is not available on this machine.</div>
              {% endif %}
              {% else %}
              <div class="muted">No asset preview available.</div>
              {% endif %}
            </div>
          </div>
        </div>
        {% else %}
        <div class="muted">No evidence recoveries for this pair.</div>
        {% endfor %}
      </div>
    </section>
    {% else %}
    <section class="panel">
      <strong>No query-target pairs found.</strong>
      <div class="muted">The current qrels file is empty or no records match the active filters.</div>
    </section>
    {% endif %}

    {% if viewer_stats %}
    <section class="panel">
      <h2>Viewer Stats</h2>
      <div class="stats">
        {% for item in viewer_stats %}
        <div><span class="label">{{ item.label }}</span><span class="value">{{ item.value }}</span></div>
        {% endfor %}
      </div>
    </section>
    {% endif %}

    {% if stats %}
    <section class="panel">
      <h2>Dataset Stats</h2>
      <div class="stats">
        {% for key, value in stats.items() %}
        {% if value is not mapping and value is not sequence or value is string %}
        <div><span class="label">{{ key }}</span><span class="value">{{ value }}</span></div>
        {% endif %}
        {% endfor %}
      </div>
    </section>
    {% endif %}

    <nav class="pager">
      <a class="button" href="{{ page_url(1) }}">First</a>
      <a class="button" href="{{ page_url(page - 1) }}">Prev</a>
      <form method="get" action="{{ url_for('index') }}">
        <input type="hidden" name="q" value="{{ query }}">
        <input type="hidden" name="split" value="{{ split }}">
        <input type="hidden" name="row_view" value="{{ row_view }}">
        <input type="hidden" name="asset_type" value="{{ asset_type }}">
        <input type="number" name="page" min="1" max="{{ pages }}" value="{{ page }}">
        <button type="submit">Go</button>
      </form>
      <span class="muted">Page {{ page }} / {{ pages }}</span>
      <a class="button" href="{{ page_url(page + 1) }}">Next</a>
      <a class="button" href="{{ page_url(pages) }}">Last</a>
    </nav>
  </main>
</body>
</html>
"""


def escape_html(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "dataset_manifest.json"


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = manifest_path(output_dir)
    return load_json(path) if path.exists() else {}


def artifact_paths(output_dir: Path, artifact: str, manifest: dict[str, Any] | None = None) -> list[Path]:
    manifest = manifest if manifest is not None else load_manifest(output_dir)
    if manifest:
        item = manifest.get("artifacts", {}).get(artifact) or {}
        return [output_dir / shard["path"] for shard in item.get("shards", []) if shard.get("path")]
    artifact_dir = output_dir / artifact
    if artifact_dir.exists():
        return sorted(artifact_dir.glob("*.jsonl"))
    flat = output_dir / f"{artifact}.jsonl"
    return [flat] if flat.exists() else []


def single_file_path(output_dir: Path, key: str, fallback: str, manifest: dict[str, Any] | None = None) -> Path:
    manifest = manifest if manifest is not None else load_manifest(output_dir)
    rel = manifest.get("single_files", {}).get(key, fallback) if manifest else fallback
    return output_dir / rel


def load_json_if_exists(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def table_id(record: dict[str, Any]) -> str:
    return clean_text(record.get("table_id") or record.get("object_id"))


def load_qrels(output_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    path = single_file_path(output_dir, "qrels", "qrels.jsonl", manifest)
    return list(iter_jsonl(path)) if path.exists() else []


def text_chunk_label(asset: dict[str, Any]) -> str:
    if clean_text(asset.get("asset_type")) != "text" or asset.get("text_chunk_index") is None:
        return ""
    try:
        chunk_index = int(asset.get("text_chunk_index")) + 1
    except (TypeError, ValueError):
        return ""
    try:
        chunk_count = int(asset.get("text_chunk_count") or 0)
    except (TypeError, ValueError):
        chunk_count = 0
    return f"chunk {chunk_index}/{chunk_count}" if chunk_count > 0 else f"chunk {chunk_index}"


def resolve_asset_file(output_dir: Path, asset: dict[str, Any]) -> Path | None:
    candidates = []
    local = clean_text(asset.get("local_path"))
    if local:
        candidates.append(Path(local))
    relative = clean_text(asset.get("relative_path"))
    if relative:
        candidates.append(output_dir / relative)
    file_name = clean_text(asset.get("file_name"))
    if file_name:
        candidates.append(output_dir / "images" / file_name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def column_names(record: dict[str, Any]) -> list[str]:
    names = []
    for idx, column in enumerate(record.get("columns", []) or []):
        name = clean_text(column.get("column_name")) if isinstance(column, dict) else clean_text(column)
        names.append(name or f"col_{idx}")
    return names


def cell_texts(row: dict[str, Any], width: int) -> list[str]:
    cells = row.get("cells", [])
    out = [""] * width
    if not isinstance(cells, list):
        return out
    for fallback, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        try:
            idx = int(cell.get("column_index", fallback))
        except (TypeError, ValueError):
            idx = fallback
        if 0 <= idx < width:
            out[idx] = clean_text(cell.get("text"))
    return out


def coerce_int_set(values: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for value in values:
        try:
            out.add(int(value))
        except (TypeError, ValueError):
            continue
    return out


def preview_table(record: dict[str, Any], max_rows: int, include_rows: set[int] | None = None) -> dict[str, Any]:
    include_rows = include_rows or set()
    names = column_names(record)
    rows = []
    raw_rows = record.get("rows", []) or []
    for ordinal, row in enumerate(raw_rows):
        try:
            rid = int(row.get("row_id", ordinal))
        except (TypeError, ValueError):
            rid = ordinal
        if len(rows) >= max_rows and rid not in include_rows:
            continue
        rows.append(
            {
                "row_id": rid,
                "source_row_id": clean_text(row.get("source_row_id")),
                "cells": cell_texts(row, len(names)),
            }
        )
    return {
        "table_id": table_id(record),
        "role": clean_text(record.get("role")),
        "split": clean_text(record.get("split")),
        "source_table_id": clean_text(record.get("source_table_id")),
        "page_title": clean_text(record.get("page_title")),
        "caption": clean_text(record.get("caption")),
        "section_title": clean_text(record.get("section_title")),
        "columns": names,
        "rows": rows,
        "truncated": bool(record.get("_viewer_truncated"))
        or len(raw_rows) > len(rows),
    }


def render_table(table: dict[str, Any], highlight_rows: Iterable[int] | None = None) -> str:
    highlights = set(highlight_rows or [])
    columns = table.get("columns", [])
    rows = table.get("rows", [])
    head = "<th>row</th><th>source row</th>" + "".join(f"<th>{escape_html(col)}</th>" for col in columns)
    body = []
    for row in rows:
        try:
            row_id = int(row.get("row_id"))
        except (TypeError, ValueError):
            row_id = -1
        klass = ' class="highlight"' if row_id in highlights else ""
        cells = [
            f"<td>{escape_html(row.get('row_id'))}</td>",
            f"<td>{escape_html(row.get('source_row_id'))}</td>",
        ]
        cells.extend(f"<td>{escape_html(value)}</td>" for value in row.get("cells", []))
        body.append(f"<tr{klass}>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def strip_html(value: Any) -> str:
    return re.sub(r"<[^>]+>", " ", clean_text(value))


def asset_url(asset: dict[str, Any]) -> str:
    return clean_text(asset.get("url") or asset.get("description_url") or asset.get("image_url"))


def merge_asset_preview(asset: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    merged = dict(asset)
    for key, value in evidence.items():
        if clean_text(value) or key not in merged:
            merged[key] = value
    return merged


def recovery_preview(
    record: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    output_dir: Path,
    max_asset_chars: int,
) -> dict[str, Any]:
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    recovered = record.get("recovered_attribute") if isinstance(record.get("recovered_attribute"), dict) else {}
    entity = record.get("query_entity") if isinstance(record.get("query_entity"), dict) else {}
    asset_id = clean_text(evidence.get("asset_id"))
    asset = merge_asset_preview(assets.get(asset_id, {}), evidence)
    if asset_id:
        asset.setdefault("asset_id", asset_id)
        assets[asset_id] = merge_asset_preview(assets.get(asset_id, {}), asset)
    asset_type = clean_text(asset.get("asset_type"))
    content = clean_text(asset.get("content")) or clean_text(asset.get("content_snippet"))
    local_file = resolve_asset_file(output_dir, asset)
    return {
        "recovery_id": clean_text(record.get("recovery_id")),
        "path_id": clean_text(record.get("path_id")),
        "query_row_id": clean_text(record.get("query_row_id")),
        "target_row_ids": [clean_text(item) for item in record.get("target_row_ids", [])],
        "source_row_id": clean_text(record.get("source_row_id")),
        "query_entity": {
            "entity_id": clean_text(entity.get("entity_id")),
            "wiki_title": clean_text(entity.get("wiki_title")),
            "cell_text": clean_text(entity.get("cell_text")),
            "entity_column_name": clean_text(entity.get("entity_column_name")),
            "row_attributes": [
                {
                    "name": clean_text(item.get("name")),
                    "value": clean_text(item.get("value")),
                    "is_entity": bool(item.get("is_entity")),
                }
                for item in entity.get("row_attributes") or []
                if isinstance(item, dict)
                and clean_text(item.get("name"))
                and clean_text(item.get("value"))
            ],
        },
        "recovered_attribute": {
            "column_name": clean_text(recovered.get("column_name")),
            "value": clean_text(recovered.get("value")),
            "model_value": clean_text(recovered.get("model_value")),
        },
        "asset_id": asset_id,
        "asset_type": asset_type,
        "asset_title": clean_text(asset.get("title") or asset.get("entity_wiki_title")),
        "asset_source": clean_text(asset.get("source")),
        "asset_url": asset_url(asset),
        "asset_file_name": clean_text(asset.get("file_name")),
        "asset_local_path": str(local_file) if local_file else clean_text(asset.get("local_path") or asset.get("relative_path")),
        "asset_image_available": bool(local_file and asset_type == "image"),
        "asset_chunk_label": text_chunk_label(asset),
        "asset_content": content[:max_asset_chars],
        "model_evidence": strip_html(evidence.get("model_evidence")),
    }


def target_id_from_qrel(qrel: dict[str, Any]) -> str:
    return clean_text(qrel.get("target_table_id") or qrel.get("data_lake_table_id") or qrel.get("target_id"))


def query_id_from_qrel(qrel: dict[str, Any]) -> str:
    return clean_text(qrel.get("query_table_id") or qrel.get("query_id"))


def qrel_key(qrel: dict[str, Any]) -> tuple[str, str]:
    return query_id_from_qrel(qrel), target_id_from_qrel(qrel)


def recovery_key(record: dict[str, Any]) -> tuple[str, str]:
    return clean_text(record.get("query_table_id")), clean_text(record.get("target_table_id") or record.get("data_lake_table_id"))


def asset_type_summary(paths: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for path in paths:
        key = clean_text(path.get("asset_type")) or "asset"
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts)) if counts else "no path materials"


def pair_search_text(pair: dict[str, Any]) -> str:
    parts = [
        pair.get("query_table_id", ""),
        pair.get("target_table_id", ""),
        pair.get("source_table_id", ""),
        pair.get("split", ""),
        pair.get("chain_id", ""),
        pair.get("row_view_kind", ""),
        pair.get("row_view_index", ""),
        pair.get("reason", ""),
    ]
    join_attribute = pair.get("join_attribute") if isinstance(pair.get("join_attribute"), dict) else {}
    parts.extend(join_attribute.values())
    for table_key in ("query_table", "target_table"):
        table = pair.get(table_key) or {}
        parts.extend(
            [
                table.get("page_title", ""),
                table.get("caption", ""),
                table.get("section_title", ""),
                " ".join(table.get("columns", [])),
            ]
        )
        for row in table.get("rows", []):
            parts.extend(row.get("cells", []))
    for path in pair.get("paths", []):
        parts.extend(
            [
                path.get("path_id", ""),
                path.get("asset_id", ""),
                path.get("asset_title", ""),
                path.get("asset_content", ""),
                path.get("model_evidence", ""),
                path.get("query_entity", {}).get("cell_text", ""),
                path.get("query_entity", {}).get("wiki_title", ""),
                path.get("recovered_attribute", {}).get("column_name", ""),
                path.get("recovered_attribute", {}).get("value", ""),
                path.get("recovered_attribute", {}).get("model_value", ""),
            ]
        )
    return " ".join(clean_text(part).casefold() for part in parts)


def coerce_row_view_index(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def build_pair_preview(
    *,
    qrel: dict[str, Any],
    query_record: dict[str, Any],
    target_record: dict[str, Any],
    raw_paths: list[dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    output_dir: Path,
    max_rows: int,
    max_paths: int,
    max_asset_chars: int,
    row_view_count: int = 1,
) -> dict[str, Any]:
    query_id = query_id_from_qrel(qrel)
    target_id = target_id_from_qrel(qrel)
    query_highlights = coerce_int_set(
        path.get("query_row_id") for path in raw_paths
    )
    target_highlights = coerce_int_set(
        row
        for path in raw_paths
        for row in (path.get("target_row_ids") or [])
    )
    shown_raw_paths = raw_paths[:max_paths]
    paths = [
        recovery_preview(path, assets, output_dir, max_asset_chars)
        for path in shown_raw_paths
    ]
    query_table = preview_table(query_record, max_rows, query_highlights)
    target_table = preview_table(target_record, max_rows, target_highlights)
    join_attribute = (
        qrel.get("join_attribute")
        if isinstance(qrel.get("join_attribute"), dict)
        else {}
    )
    row_view_index = coerce_row_view_index(qrel.get("row_view_index"))
    pair = {
        "query_table_id": query_id,
        "target_table_id": target_id,
        "data_lake_table_id": clean_text(qrel.get("data_lake_table_id")),
        "source_table_id": clean_text(
            qrel.get("source_table_id") or query_table.get("source_table_id")
        ),
        "split": clean_text(qrel.get("split") or query_table.get("split")),
        "chain_id": clean_text(qrel.get("chain_id")),
        "row_view_index": row_view_index,
        "row_view_number": row_view_index + 1,
        "row_view_count": max(1, int(row_view_count)),
        "row_view_kind": "canonical" if row_view_index == 0 else "augmented",
        "rel": clean_text(qrel.get("rel")),
        "reason": clean_text(qrel.get("reason")),
        "join_attribute": {
            "column_name": clean_text(join_attribute.get("column_name")),
            "recovered_rows": clean_text(join_attribute.get("recovered_rows")),
            "eligible_rows": clean_text(join_attribute.get("eligible_rows")),
            "recovered_value_ratio": clean_text(
                join_attribute.get("recovered_value_ratio")
            ),
        },
        "join_col_name": clean_text(qrel.get("join_col_name")),
        "query_table": query_table,
        "target_table": target_table,
        "query_highlight_rows": sorted(query_highlights),
        "target_highlight_rows": sorted(target_highlights),
        "paths": paths,
        "path_count": len(raw_paths),
        "path_count_displayed": len(paths),
        "asset_type_summary": asset_type_summary(paths),
        "asset_types": sorted(
            {
                path.get("asset_type")
                for path in paths
                if path.get("asset_type")
            }
        ),
    }
    pair["_search"] = pair_search_text(pair)
    return pair


def viewer_pair_key(query_id: str, target_id: str) -> str:
    return json.dumps([query_id, target_id], ensure_ascii=False, separators=(",", ":"))


def iter_artifact_record_offsets(
    output_dir: Path,
    artifact: str,
    manifest: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], Path, int, int]]:
    """Read raw JSONL records without expanding data-lake source references."""
    for path in artifact_paths(output_dir, artifact, manifest):
        if not path.exists():
            continue
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                yield json.loads(line), path.resolve(), offset, len(line)


def iter_jsonl_byte_ranges(
    path: Path,
) -> Iterator[tuple[bytes, int, int]]:
    """Yield a bounded prefix and byte range for each JSONL record."""
    with path.open("rb") as handle:
        record_start = 0
        record_length = 0
        prefix = bytearray()
        absolute_offset = 0
        while True:
            chunk = handle.read(JSONL_SCAN_CHUNK_BYTES)
            if not chunk:
                break
            cursor = 0
            while cursor < len(chunk):
                newline = chunk.find(b"\n", cursor)
                stop = len(chunk) if newline < 0 else newline + 1
                segment = chunk[cursor:stop]
                if len(prefix) < JSONL_ID_PREFIX_BYTES:
                    remaining = JSONL_ID_PREFIX_BYTES - len(prefix)
                    prefix.extend(segment[:remaining])
                segment_length = len(segment)
                record_length += segment_length
                absolute_offset += segment_length
                cursor = stop
                if newline >= 0:
                    if prefix.strip():
                        yield bytes(prefix), record_start, record_length
                    record_start = absolute_offset
                    record_length = 0
                    prefix.clear()
        if record_length and prefix.strip():
            yield bytes(prefix), record_start, record_length


def json_string_field_from_prefix(prefix: bytes, field: str) -> str:
    field_bytes = re.escape(field.encode("ascii"))
    match = re.search(
        rb'"' + field_bytes + rb'"\s*:\s*"((?:\\.|[^"\\])*)"',
        prefix,
    )
    if match is None:
        return ""
    try:
        return clean_text(json.loads(b'"' + match.group(1) + b'"'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def iter_artifact_id_offsets(
    output_dir: Path,
    artifact: str,
    manifest: dict[str, Any],
    id_field: str,
) -> Iterator[tuple[str, Path, int, int]]:
    for path in artifact_paths(output_dir, artifact, manifest):
        if not path.exists():
            continue
        resolved = path.resolve()
        for prefix, offset, length in iter_jsonl_byte_ranges(path):
            record_id = json_string_field_from_prefix(prefix, id_field)
            if not record_id and id_field == "table_id":
                record_id = json_string_field_from_prefix(prefix, "object_id")
            if record_id:
                yield record_id, resolved, offset, length


def stream_table_preview_record(
    handle: BinaryIO,
    *,
    max_rows: int,
    include_rows: set[int],
) -> dict[str, Any]:
    """Materialize table metadata, columns, and only rows needed by the UI."""
    scalar_fields = {
        "table_id",
        "object_id",
        "role",
        "split",
        "source_table_id",
        "page_title",
        "caption",
        "section_title",
    }
    record: dict[str, Any] = {"columns": [], "rows": []}
    pending_rows = set(include_rows)
    row_limit = max(0, int(max_rows))
    rows_seen = 0
    truncated = False
    builder: ObjectBuilder | None = None
    builder_prefix = ""
    builder_kind = ""

    for prefix, event, value in ijson.parse(handle):
        if builder is not None:
            builder.event(event, value)
            if prefix == builder_prefix and event == "end_map":
                item = builder.value
                builder = None
                if builder_kind == "column":
                    record["columns"].append(item)
                else:
                    try:
                        row_id = int(item.get("row_id", rows_seen))
                    except (AttributeError, TypeError, ValueError):
                        row_id = rows_seen
                    rows_seen += 1
                    if rows_seen <= row_limit or row_id in include_rows:
                        record["rows"].append(item)
                    pending_rows.discard(row_id)
                    if rows_seen > row_limit and not pending_rows:
                        truncated = True
                        break
                builder_prefix = ""
                builder_kind = ""
            continue

        if prefix in scalar_fields and event in {
            "string",
            "number",
            "boolean",
            "null",
        }:
            record[prefix] = value
        elif prefix == "columns.item" and event == "start_map":
            builder = ObjectBuilder()
            builder.event(event, value)
            builder_prefix = prefix
            builder_kind = "column"
        elif prefix == "columns.item" and event in {
            "string",
            "number",
            "boolean",
            "null",
        }:
            record["columns"].append(value)
        elif prefix == "rows.item" and event == "start_map":
            builder = ObjectBuilder()
            builder.event(event, value)
            builder_prefix = prefix
            builder_kind = "row"
        elif prefix == "rows" and event == "end_array":
            break

    record["_viewer_truncated"] = truncated
    return record


class IndexedRecordReader:
    def __init__(self) -> None:
        self._handles: dict[str, BinaryIO] = {}

    def read(self, path: str, offset: int, length: int) -> dict[str, Any]:
        handle = self._handles.get(path)
        if handle is None:
            handle = Path(path).open("rb")
            self._handles[path] = handle
        handle.seek(offset)
        return json.loads(handle.read(length))

    def read_table_preview(
        self,
        path: str,
        offset: int,
        *,
        max_rows: int,
        include_rows: set[int],
    ) -> dict[str, Any]:
        handle = self._handles.get(path)
        if handle is None:
            handle = Path(path).open("rb")
            self._handles[path] = handle
        handle.seek(offset)
        return stream_table_preview_record(
            handle,
            max_rows=max_rows,
            include_rows=include_rows,
        )

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "IndexedRecordReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def viewer_index_signature(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    max_rows: int,
    max_paths: int,
    max_asset_chars: int,
) -> str:
    paths = [manifest_path(output_dir)]
    paths.append(single_file_path(output_dir, "qrels", "qrels.jsonl", manifest))
    for artifact in (
        "query_tables",
        "data_lake_tables",
        "bridge_assets",
        "evidence_recoveries",
    ):
        paths.extend(artifact_paths(output_dir, artifact, manifest))
    file_state = []
    for path in paths:
        resolved = path.resolve()
        if resolved.exists():
            stat = resolved.stat()
            file_state.append(
                {
                    "path": str(resolved),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            file_state.append({"path": str(resolved), "missing": True})
    payload = {
        "schema_version": VIEWER_INDEX_SCHEMA_VERSION,
        "files": file_state,
        "preview": {
            "max_rows": max_rows,
            "max_paths": max_paths,
            "max_asset_chars": max_asset_chars,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class ViewerDataset:
    """Bounded-memory access to a joinability dataset through a local offset index."""

    def __init__(
        self,
        output_dir: Path,
        max_rows: int,
        max_paths: int,
        max_asset_chars: int,
        index_path: Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.max_rows = max_rows
        self.max_paths = max_paths
        self.max_asset_chars = max_asset_chars
        self.manifest = load_manifest(self.output_dir)
        self.index_path = (
            Path(index_path).resolve()
            if index_path is not None
            else self.output_dir / DEFAULT_INDEX_FILENAME
        )
        self.signature = viewer_index_signature(
            self.output_dir,
            self.manifest,
            max_rows=max_rows,
            max_paths=max_paths,
            max_asset_chars=max_asset_chars,
        )
        self._ensure_index()

    def _connect(self, path: Path | None = None) -> sqlite3.Connection:
        connection = sqlite3.connect(path or self.index_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _index_is_current(self) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with self._connect() as connection:
                metadata = dict(connection.execute("SELECT key, value FROM meta"))
            return (
                metadata.get("schema_version") == VIEWER_INDEX_SCHEMA_VERSION
                and metadata.get("signature") == self.signature
                and metadata.get("complete") == "1"
            )
        except (OSError, sqlite3.DatabaseError):
            return False

    def _ensure_index(self) -> None:
        if self._index_is_current():
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_name(
            f".{self.index_path.name}.{uuid.uuid4().hex}.tmp"
        )
        LOG.info("Building viewer index at %s", self.index_path)
        try:
            self._build_index(temporary_path)
            os.replace(temporary_path, self.index_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE pairs (
                pair_key TEXT PRIMARY KEY,
                query_table_id TEXT NOT NULL,
                target_table_id TEXT NOT NULL,
                data_lake_table_id TEXT NOT NULL,
                source_table_id TEXT NOT NULL,
                split TEXT NOT NULL,
                chain_id TEXT NOT NULL,
                row_view_index INTEGER NOT NULL,
                rel TEXT NOT NULL,
                reason TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                search_text TEXT NOT NULL DEFAULT '',
                path_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE records (
                artifact TEXT NOT NULL,
                record_id TEXT NOT NULL,
                path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                PRIMARY KEY (artifact, record_id)
            );
            CREATE TABLE recoveries (
                ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_key TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                path TEXT NOT NULL,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL
            );
            CREATE TABLE pair_asset_types (
                pair_key TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                PRIMARY KEY (pair_key, asset_type)
            );
            CREATE INDEX pairs_order_idx ON pairs (
                split, source_table_id, chain_id, row_view_index,
                query_table_id, target_table_id
            );
            CREATE INDEX recoveries_pair_idx ON recoveries (pair_key, ordinal);
            CREATE INDEX recoveries_asset_idx ON recoveries (asset_id);
            CREATE INDEX pair_asset_types_type_idx ON pair_asset_types (
                asset_type, pair_key
            );
            """
        )

    @staticmethod
    def _insert_pair(
        connection: sqlite3.Connection,
        qrel: dict[str, Any],
    ) -> str:
        query_id = query_id_from_qrel(qrel)
        target_id = target_id_from_qrel(qrel)
        key = viewer_pair_key(query_id, target_id)
        connection.execute(
            """
            INSERT OR REPLACE INTO pairs (
                pair_key, query_table_id, target_table_id, data_lake_table_id,
                source_table_id, split, chain_id, row_view_index, rel, reason,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                query_id,
                target_id,
                clean_text(qrel.get("data_lake_table_id")),
                clean_text(qrel.get("source_table_id")),
                clean_text(qrel.get("split")),
                clean_text(qrel.get("chain_id")),
                coerce_row_view_index(qrel.get("row_view_index")),
                clean_text(qrel.get("rel")),
                clean_text(qrel.get("reason")),
                json.dumps(qrel, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        return key

    def _build_index(self, temporary_path: Path) -> None:
        connection = self._connect(temporary_path)
        try:
            self._create_schema(connection)
            pair_keys: set[str] = set()
            query_ids: set[str] = set()
            target_ids: set[str] = set()
            for qrel in load_qrels(self.output_dir, self.manifest):
                query_id, target_id = qrel_key(qrel)
                if not query_id or not target_id:
                    continue
                key = self._insert_pair(connection, qrel)
                pair_keys.add(key)
                query_ids.add(query_id)
                target_ids.add(target_id)

            asset_ids: set[str] = set()
            recovery_count = 0
            for record, path, offset, length in iter_artifact_record_offsets(
                self.output_dir, "evidence_recoveries", self.manifest
            ):
                query_id, target_id = recovery_key(record)
                if not query_id or not target_id:
                    continue
                key = viewer_pair_key(query_id, target_id)
                if key not in pair_keys:
                    synthetic_qrel = {
                        "query_table_id": query_id,
                        "target_table_id": target_id,
                        "data_lake_table_id": target_id,
                        "split": record.get("split"),
                        "source_table_id": record.get("source_table_id"),
                        "rel": "",
                        "reason": "evidence_recovery_only",
                    }
                    self._insert_pair(connection, synthetic_qrel)
                    pair_keys.add(key)
                    query_ids.add(query_id)
                    target_ids.add(target_id)
                evidence = (
                    record.get("evidence")
                    if isinstance(record.get("evidence"), dict)
                    else {}
                )
                asset_id = clean_text(evidence.get("asset_id"))
                asset_type = clean_text(evidence.get("asset_type"))
                connection.execute(
                    """
                    INSERT INTO recoveries (
                        pair_key, asset_id, asset_type, path,
                        byte_offset, byte_length
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, asset_id, asset_type, str(path), offset, length),
                )
                if asset_type:
                    connection.execute(
                        "INSERT OR IGNORE INTO pair_asset_types VALUES (?, ?)",
                        (key, asset_type),
                    )
                if asset_id:
                    asset_ids.add(asset_id)
                recovery_count += 1
                if recovery_count % 10000 == 0:
                    connection.commit()
                    LOG.info("Indexed %s evidence recoveries", f"{recovery_count:,}")

            self._index_artifact_records(
                connection,
                artifact="query_tables",
                wanted_ids=query_ids,
                id_field="table_id",
            )
            self._index_artifact_records(
                connection,
                artifact="data_lake_tables",
                wanted_ids=target_ids,
                id_field="table_id",
            )
            self._index_artifact_records(
                connection,
                artifact="bridge_assets",
                wanted_ids=asset_ids,
                id_field="asset_id",
            )

            pair_rows = connection.execute(
                "SELECT pair_key FROM pairs ORDER BY pair_key"
            ).fetchall()
            with IndexedRecordReader() as reader:
                for index, row in enumerate(pair_rows, start=1):
                    pair = self._hydrate_pair_from_connection(
                        connection,
                        row["pair_key"],
                        reader,
                        include_view_count=False,
                    )
                    connection.execute(
                        "UPDATE pairs SET search_text = ?, path_count = ? WHERE pair_key = ?",
                        (pair["_search"], pair["path_count"], row["pair_key"]),
                    )
                    if index % 1000 == 0:
                        connection.commit()
                        LOG.info("Prepared search text for %s pairs", f"{index:,}")

            connection.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", VIEWER_INDEX_SCHEMA_VERSION),
                    ("signature", self.signature),
                    ("complete", "1"),
                ],
            )
            connection.commit()
        finally:
            connection.close()

    def _index_artifact_records(
        self,
        connection: sqlite3.Connection,
        *,
        artifact: str,
        wanted_ids: set[str],
        id_field: str,
    ) -> None:
        found = 0
        with IndexedRecordReader() as reader:
            offsets = iter_artifact_id_offsets(
                self.output_dir,
                artifact,
                self.manifest,
                id_field,
            )
            for record_id, path, offset, length in offsets:
                if record_id not in wanted_ids:
                    continue
                connection.execute(
                    """
                    INSERT OR REPLACE INTO records (
                        artifact, record_id, path, byte_offset, byte_length
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (artifact, record_id, str(path), offset, length),
                )
                record = (
                    {}
                    if artifact == "data_lake_tables"
                    else reader.read(str(path), offset, length)
                )
                if artifact == "query_tables":
                    connection.execute(
                        """
                        UPDATE pairs
                        SET source_table_id = CASE
                                WHEN source_table_id = '' THEN ? ELSE source_table_id END,
                            split = CASE WHEN split = '' THEN ? ELSE split END
                        WHERE query_table_id = ?
                        """,
                        (
                            clean_text(record.get("source_table_id")),
                            clean_text(record.get("split")),
                            record_id,
                        ),
                    )
                if artifact == "bridge_assets":
                    asset_type = clean_text(record.get("asset_type"))
                    if asset_type:
                        connection.execute(
                            """
                            UPDATE recoveries SET asset_type = ?
                            WHERE asset_id = ? AND asset_type = ''
                            """,
                            (asset_type, record_id),
                        )
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO pair_asset_types (pair_key, asset_type)
                            SELECT pair_key, ? FROM recoveries WHERE asset_id = ?
                            """,
                            (asset_type, record_id),
                        )
                found += 1
        connection.commit()
        LOG.info(
            "Indexed %s/%s referenced %s records",
            f"{found:,}",
            f"{len(wanted_ids):,}",
            artifact,
        )

    @staticmethod
    def _read_record(
        connection: sqlite3.Connection,
        reader: IndexedRecordReader,
        artifact: str,
        record_id: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT path, byte_offset, byte_length FROM records
            WHERE artifact = ? AND record_id = ?
            """,
            (artifact, record_id),
        ).fetchone()
        if row is None:
            return {}
        return reader.read(row["path"], row["byte_offset"], row["byte_length"])

    def _read_table_preview_record(
        self,
        connection: sqlite3.Connection,
        reader: IndexedRecordReader,
        artifact: str,
        record_id: str,
        include_rows: set[int],
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT path, byte_offset FROM records
            WHERE artifact = ? AND record_id = ?
            """,
            (artifact, record_id),
        ).fetchone()
        if row is None:
            return {}
        return reader.read_table_preview(
            row["path"],
            row["byte_offset"],
            max_rows=self.max_rows,
            include_rows=include_rows,
        )

    @staticmethod
    def _row_view_count(
        connection: sqlite3.Connection,
        pair_row: sqlite3.Row,
    ) -> int:
        if not pair_row["chain_id"]:
            return 1
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT row_view_index) AS count
            FROM pairs
            WHERE split = ? AND source_table_id = ? AND chain_id = ?
            """,
            (
                pair_row["split"],
                pair_row["source_table_id"],
                pair_row["chain_id"],
            ),
        ).fetchone()
        return max(1, int(row["count"] or 0))

    def _hydrate_pair_from_connection(
        self,
        connection: sqlite3.Connection,
        pair_key: str,
        reader: IndexedRecordReader,
        *,
        include_view_count: bool,
    ) -> dict[str, Any]:
        pair_row = connection.execute(
            "SELECT * FROM pairs WHERE pair_key = ?", (pair_key,)
        ).fetchone()
        if pair_row is None:
            raise KeyError(f"Unknown pair: {pair_key}")
        qrel = json.loads(pair_row["raw_json"])
        recovery_rows = connection.execute(
            """
            SELECT path, byte_offset, byte_length FROM recoveries
            WHERE pair_key = ? ORDER BY ordinal
            """,
            (pair_key,),
        ).fetchall()
        raw_paths = [
            reader.read(row["path"], row["byte_offset"], row["byte_length"])
            for row in recovery_rows
        ]
        query_highlights = coerce_int_set(
            path.get("query_row_id") for path in raw_paths
        )
        target_highlights = coerce_int_set(
            row
            for path in raw_paths
            for row in (path.get("target_row_ids") or [])
        )
        query_record = self._read_table_preview_record(
            connection,
            reader,
            "query_tables",
            pair_row["query_table_id"],
            query_highlights,
        )
        target_record = self._read_table_preview_record(
            connection,
            reader,
            "data_lake_tables",
            pair_row["target_table_id"],
            target_highlights,
        )
        assets: dict[str, dict[str, Any]] = {}
        for record in raw_paths[: self.max_paths]:
            evidence = (
                record.get("evidence")
                if isinstance(record.get("evidence"), dict)
                else {}
            )
            asset_id = clean_text(evidence.get("asset_id"))
            if asset_id and asset_id not in assets:
                assets[asset_id] = self._read_record(
                    connection, reader, "bridge_assets", asset_id
                )
        pair = build_pair_preview(
            qrel=qrel,
            query_record=query_record,
            target_record=target_record,
            raw_paths=raw_paths,
            assets=assets,
            output_dir=self.output_dir,
            max_rows=self.max_rows,
            max_paths=self.max_paths,
            max_asset_chars=self.max_asset_chars,
            row_view_count=(
                self._row_view_count(connection, pair_row)
                if include_view_count
                else 1
            ),
        )
        pair["asset_types"] = [
            row["asset_type"]
            for row in connection.execute(
                """
                SELECT asset_type FROM pair_asset_types
                WHERE pair_key = ? ORDER BY asset_type
                """,
                (pair_key,),
            )
        ]
        return pair

    def hydrate_pair(self, pair_key: str) -> dict[str, Any]:
        with self._connect() as connection, IndexedRecordReader() as reader:
            return self._hydrate_pair_from_connection(
                connection,
                pair_key,
                reader,
                include_view_count=True,
            )

    def hydrate_full_pair_tables(
        self,
        pair_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load complete normalized query and target tables for one pair."""
        with self._connect() as connection, IndexedRecordReader() as reader:
            pair_row = connection.execute(
                "SELECT query_table_id,target_table_id FROM pairs WHERE pair_key = ?",
                (pair_key,),
            ).fetchone()
            if pair_row is None:
                raise KeyError(f"Unknown pair: {pair_key}")
            query_record = self._read_record(
                connection,
                reader,
                "query_tables",
                pair_row["query_table_id"],
            )
            target_record = self._read_record(
                connection,
                reader,
                "data_lake_tables",
                pair_row["target_table_id"],
            )

        def full_preview(record: dict[str, Any]) -> dict[str, Any]:
            rows = record.get("rows") if isinstance(record.get("rows"), list) else []
            return preview_table(record, max(1, len(rows)))

        return full_preview(query_record), full_preview(target_record)

    @staticmethod
    def _filter_sql(
        *,
        query: str,
        split: str,
        asset_type: str,
        row_view: str,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if split:
            clauses.append("pairs.split = ?")
            parameters.append(split)
        if asset_type:
            clauses.append(
                """
                EXISTS (
                    SELECT 1 FROM pair_asset_types
                    WHERE pair_asset_types.pair_key = pairs.pair_key
                      AND pair_asset_types.asset_type = ?
                )
                """
            )
            parameters.append(asset_type)
        if row_view == "canonical":
            clauses.append("pairs.row_view_index = 0")
        elif row_view == "augmented":
            clauses.append("pairs.row_view_index > 0")
        if query:
            clauses.append("instr(pairs.search_text, ?) > 0")
            parameters.append(query.casefold())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, parameters

    def filtered_pair(
        self,
        *,
        query: str,
        split: str,
        asset_type: str,
        row_view: str,
        page: int,
    ) -> tuple[dict[str, Any] | None, int]:
        where, parameters = self._filter_sql(
            query=query,
            split=split,
            asset_type=asset_type,
            row_view=row_view,
        )
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM pairs{where}", parameters
                ).fetchone()[0]
            )
            if total == 0:
                return None, 0
            page = min(max(1, page), total)
            row = connection.execute(
                f"""
                SELECT pair_key FROM pairs{where}
                ORDER BY split, source_table_id, chain_id, row_view_index,
                         query_table_id, target_table_id
                LIMIT 1 OFFSET ?
                """,
                [*parameters, max(0, page - 1)],
            ).fetchone()
        return self.hydrate_pair(row["pair_key"]), total

    def filter_options(self) -> tuple[list[str], list[str]]:
        with self._connect() as connection:
            splits = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT split FROM pairs WHERE split != '' ORDER BY split"
                )
            ]
            asset_types = [
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT asset_type FROM pair_asset_types
                    WHERE asset_type != '' ORDER BY asset_type
                    """
                )
            ]
        return splits, asset_types

    def viewer_stats(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT pair_key, query_table_id, split, source_table_id,
                       chain_id, row_view_index
                FROM pairs
                """
            ).fetchall()
        chains: dict[tuple[str, ...], set[int]] = {}
        query_ids: set[str] = set()
        canonical = 0
        augmented = 0
        for row in rows:
            query_ids.add(row["query_table_id"])
            row_view_index = int(row["row_view_index"])
            if row_view_index == 0:
                canonical += 1
            else:
                augmented += 1
            chain_key = (
                (row["split"], row["source_table_id"], row["chain_id"])
                if row["chain_id"]
                else ("pair", row["pair_key"])
            )
            chains.setdefault(chain_key, set()).add(row_view_index)
        return [
            {"label": "Query-target pairs", "value": len(rows)},
            {"label": "Unique query tables", "value": len(query_ids)},
            {"label": "Unique join chains", "value": len(chains)},
            {
                "label": "Multi-view chains",
                "value": sum(1 for views in chains.values() if len(views) > 1),
            },
            {"label": "Canonical pairs", "value": canonical},
            {"label": "Augmented pairs", "value": augmented},
        ]

    def all_pair_keys(self) -> list[str]:
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    """
                    SELECT pair_key FROM pairs
                    ORDER BY split, source_table_id, chain_id, row_view_index,
                             query_table_id, target_table_id
                    """
                )
            ]

    def implicit_query_ids(self) -> list[str]:
        """Return the unique implicit-query IDs represented by the dataset."""
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    """
                    SELECT DISTINCT query_table_id FROM pairs
                    WHERE reason = ?
                    ORDER BY query_table_id
                    """,
                    (IMPLICIT_JOIN_REASON,),
                )
            ]

    def validate_implicit_query_uniqueness(self) -> int:
        """Require exactly one target qrel for every implicit query."""
        with self._connect() as connection:
            invalid = connection.execute(
                """
                SELECT query_table_id, COUNT(*) AS qrel_count
                FROM pairs
                WHERE reason = ?
                GROUP BY query_table_id
                HAVING COUNT(*) != 1
                ORDER BY query_table_id
                LIMIT 1
                """,
                (IMPLICIT_JOIN_REASON,),
            ).fetchone()
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM pairs WHERE reason = ?",
                    (IMPLICIT_JOIN_REASON,),
                ).fetchone()[0]
            )
        if invalid is not None:
            raise ValueError(
                "checker requires one attribute/target per implicit query: "
                f"query_table_id={invalid['query_table_id']!r}, "
                f"qrels={invalid['qrel_count']}"
            )
        return total

    def pair_keys_for_query(
        self,
        query_table_id: str,
        *,
        implicit_only: bool = False,
    ) -> list[str]:
        """Return all query-target pair keys for one query in stable order."""
        clauses = ["query_table_id = ?"]
        parameters: list[Any] = [clean_text(query_table_id)]
        if implicit_only:
            clauses.append("reason = ?")
            parameters.append(IMPLICIT_JOIN_REASON)
        with self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    f"""
                    SELECT pair_key FROM pairs
                    WHERE {' AND '.join(clauses)}
                    ORDER BY target_table_id, pair_key
                    """,
                    parameters,
                )
            ]

    def referenced_assets(self) -> dict[str, dict[str, Any]]:
        assets: dict[str, dict[str, Any]] = {}
        with self._connect() as connection, IndexedRecordReader() as reader:
            rows = connection.execute(
                """
                SELECT record_id, path, byte_offset, byte_length FROM records
                WHERE artifact = 'bridge_assets'
                """
            )
            for row in rows:
                assets[row["record_id"]] = reader.read(
                    row["path"], row["byte_offset"], row["byte_length"]
                )
        return assets

    def asset(self, asset_id: str) -> dict[str, Any]:
        with self._connect() as connection, IndexedRecordReader() as reader:
            return self._read_record(
                connection, reader, "bridge_assets", clean_text(asset_id)
            )

    def stats(self) -> dict[str, Any]:
        return load_json_if_exists(
            single_file_path(self.output_dir, "stats", "stats.json", self.manifest)
        )


def load_pairs(
    output_dir: Path,
    max_rows: int,
    max_paths: int,
    max_asset_chars: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    dataset = ViewerDataset(output_dir, max_rows, max_paths, max_asset_chars)
    pairs = [dataset.hydrate_pair(key) for key in dataset.all_pair_keys()]
    return pairs, dataset.stats(), dataset.referenced_assets()


def create_app(
    output_dir: Path,
    max_rows: int,
    max_paths: int,
    max_asset_chars: int,
    index_path: Path | None = None,
) -> Flask:
    app = Flask(__name__)
    output_dir = Path(output_dir)

    @lru_cache(maxsize=1)
    def cached_dataset() -> ViewerDataset:
        return ViewerDataset(
            output_dir,
            max_rows,
            max_paths,
            max_asset_chars,
            index_path=index_path,
        )

    @app.get("/")
    def index() -> str:
        error = ""
        try:
            dataset = cached_dataset()
            stats = dataset.stats()
            viewer_stats = dataset.viewer_stats()
            splits, asset_types = dataset.filter_options()
        except Exception as exc:  # pragma: no cover - visible in browser for local debugging.
            dataset = None
            stats = {}
            viewer_stats = []
            splits = []
            asset_types = []
            error = str(exc)
        query = clean_text(request.args.get("q", ""))
        split = clean_text(request.args.get("split", ""))
        asset_type = clean_text(request.args.get("asset_type", ""))
        row_view = clean_text(request.args.get("row_view", "all"))
        if row_view not in {"all", "canonical", "augmented"}:
            row_view = "all"
        try:
            page = max(1, int(request.args.get("page", "1") or "1"))
        except ValueError:
            page = 1
        if dataset is None:
            pair = None
            total = 0
        else:
            pair, total = dataset.filtered_pair(
                query=query,
                split=split,
                asset_type=asset_type,
                row_view=row_view,
                page=page,
            )
        pages = max(1, math.ceil(total))
        page = min(page, pages)

        def page_url(target: int) -> str:
            target = min(max(1, target), pages)
            return url_for(
                "index",
                page=target,
                q=query,
                split=split,
                row_view=row_view,
                asset_type=asset_type,
            )

        return render_template_string(
            PAGE_TEMPLATE,
            pair=pair,
            total=total,
            page=page,
            pages=pages,
            query=query,
            split=split,
            row_view=row_view,
            asset_type=asset_type,
            splits=splits,
            asset_types=asset_types,
            stats=stats,
            viewer_stats=viewer_stats,
            error=error,
            page_url=page_url,
            render_table=render_table,
        )

    @app.get("/reload")
    def reload() -> Any:
        cached_dataset.cache_clear()
        return redirect(url_for("index"))

    @app.get("/asset/<asset_id>")
    def asset_image(asset_id: str) -> Any:
        asset = cached_dataset().asset(asset_id)
        if not asset:
            abort(404, f"Unknown asset_id: {asset_id}")
        image_path = resolve_asset_file(output_dir, asset)
        if image_path is None:
            abort(404, f"No local image file for asset_id: {asset_id}")
        return send_file(image_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output_mm_joinability")
    parser.add_argument("--host", default=None)
    parser.add_argument("--lan", action="store_true", help="Expose the viewer GUI on the LAN by binding to 0.0.0.0.")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--max_rows", type=int, default=12)
    parser.add_argument("--max_paths", type=int, default=50)
    parser.add_argument("--max_asset_chars", type=int, default=2400)
    parser.add_argument(
        "--index_path",
        default=None,
        help=(
            "Persistent SQLite viewer index path. Defaults to "
            f"<output_dir>/{DEFAULT_INDEX_FILENAME}."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.host = resolve_gui_host(args.host, args.lan)
    app = create_app(
        Path(args.output_dir),
        args.max_rows,
        args.max_paths,
        args.max_asset_chars,
        index_path=Path(args.index_path) if args.index_path else None,
    )
    print(format_gui_urls("MM joinability viewer", args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Flask GUI for browsing build_mm_joinability_dataset outputs."""

from __future__ import annotations

import argparse
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

from stage1_gui import format_gui_urls, resolve_gui_host
from stage1_io import (
    clean_text,
    iter_jsonl,
    iter_manifest_records,
    load_json,
)

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


def iter_artifact_records(output_dir: Path, artifact: str, manifest: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    if (output_dir / "dataset_manifest.json").exists():
        yield from iter_manifest_records(
            output_dir,
            artifact,
            log_every=0,
        )
        return
    for path in artifact_paths(output_dir, artifact, manifest):
        if path.exists():
            yield from iter_jsonl(path)


def single_file_path(output_dir: Path, key: str, fallback: str, manifest: dict[str, Any] | None = None) -> Path:
    manifest = manifest if manifest is not None else load_manifest(output_dir)
    rel = manifest.get("single_files", {}).get(key, fallback) if manifest else fallback
    return output_dir / rel


def load_json_if_exists(path: Path) -> dict[str, Any]:
    return load_json(path) if path.exists() else {}


def table_id(record: dict[str, Any]) -> str:
    return clean_text(record.get("table_id") or record.get("object_id"))


def load_tables(output_dir: Path, artifact: str, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for record in iter_artifact_records(output_dir, artifact, manifest):
        ident = table_id(record)
        if ident:
            tables[ident] = record
    return tables


def load_assets(output_dir: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for record in iter_artifact_records(output_dir, "bridge_assets", manifest):
        asset_id = clean_text(record.get("asset_id"))
        if asset_id:
            assets[asset_id] = record
    return assets


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
    for ordinal, row in enumerate(record.get("rows", []) or []):
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
        "truncated": len(record.get("rows", []) or []) > len(rows),
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


def load_pairs(output_dir: Path, max_rows: int, max_paths: int, max_asset_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_manifest(output_dir)
    query_tables = load_tables(output_dir, "query_tables", manifest)
    target_tables = load_tables(output_dir, "data_lake_tables", manifest)
    assets = load_assets(output_dir, manifest)
    qrels = load_qrels(output_dir, manifest)
    recoveries_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in iter_artifact_records(output_dir, "evidence_recoveries", manifest):
        key = recovery_key(record)
        if all(key):
            recoveries_by_pair.setdefault(key, []).append(record)

    qrels_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for qrel in qrels:
        key = qrel_key(qrel)
        if all(key):
            qrels_by_key[key] = qrel
    for key, records in recoveries_by_pair.items():
        if key not in qrels_by_key and records:
            qrels_by_key[key] = {
                "query_table_id": key[0],
                "target_table_id": key[1],
                "data_lake_table_id": key[1],
                "split": records[0].get("split"),
                "source_table_id": records[0].get("source_table_id"),
                "rel": "",
                "reason": "evidence_recovery_only",
            }

    pairs = []
    for key, qrel in qrels_by_key.items():
        query_id, target_id = key
        raw_paths = recoveries_by_pair.get(key, [])
        query_highlights = coerce_int_set(path.get("query_row_id") for path in raw_paths)
        target_highlights = coerce_int_set(row for path in raw_paths for row in (path.get("target_row_ids") or []))
        shown_raw_paths = raw_paths[:max_paths]
        paths = [recovery_preview(path, assets, output_dir, max_asset_chars) for path in shown_raw_paths]
        query_table = preview_table(query_tables.get(query_id, {}), max_rows, query_highlights)
        target_table = preview_table(target_tables.get(target_id, {}), max_rows, target_highlights)
        join_attribute = qrel.get("join_attribute") if isinstance(qrel.get("join_attribute"), dict) else {}
        pair = {
            "query_table_id": query_id,
            "target_table_id": target_id,
            "data_lake_table_id": clean_text(qrel.get("data_lake_table_id")),
            "source_table_id": clean_text(qrel.get("source_table_id") or query_table.get("source_table_id")),
            "split": clean_text(qrel.get("split") or query_table.get("split")),
            "chain_id": clean_text(qrel.get("chain_id")),
            "rel": clean_text(qrel.get("rel")),
            "reason": clean_text(qrel.get("reason")),
            "join_attribute": {
                "column_name": clean_text(join_attribute.get("column_name")),
                "recovered_rows": clean_text(join_attribute.get("recovered_rows")),
                "eligible_rows": clean_text(join_attribute.get("eligible_rows")),
                "recovered_value_ratio": clean_text(join_attribute.get("recovered_value_ratio")),
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
            "asset_types": sorted({path.get("asset_type") for path in paths if path.get("asset_type")}),
        }
        pair["_search"] = pair_search_text(pair)
        pairs.append(pair)
    pairs.sort(key=lambda item: (item.get("split", ""), item.get("source_table_id", ""), item.get("query_table_id", ""), item.get("target_table_id", "")))
    stats = load_json_if_exists(single_file_path(output_dir, "stats", "stats.json", manifest))
    return pairs, stats, assets


def create_app(output_dir: Path, max_rows: int, max_paths: int, max_asset_chars: int) -> Flask:
    app = Flask(__name__)
    output_dir = Path(output_dir)

    @lru_cache(maxsize=1)
    def cached_context() -> dict[str, Any]:
        pairs, stats, assets = load_pairs(output_dir, max_rows, max_paths, max_asset_chars)
        return {"pairs": pairs, "stats": stats, "assets": assets}

    @app.get("/")
    def index() -> str:
        error = ""
        try:
            context = cached_context()
            pairs = list(context["pairs"])
            stats = context["stats"]
        except Exception as exc:  # pragma: no cover - visible in browser for local debugging.
            pairs = []
            stats = {}
            error = str(exc)
        query = clean_text(request.args.get("q", ""))
        split = clean_text(request.args.get("split", ""))
        asset_type = clean_text(request.args.get("asset_type", ""))
        try:
            page = max(1, int(request.args.get("page", "1") or "1"))
        except ValueError:
            page = 1
        all_pairs = pairs
        splits = sorted({pair["split"] for pair in all_pairs if pair.get("split")})
        asset_types = sorted({atype for pair in all_pairs for atype in pair.get("asset_types", []) if atype})
        if split:
            pairs = [pair for pair in pairs if pair.get("split") == split]
        if asset_type:
            pairs = [pair for pair in pairs if asset_type in pair.get("asset_types", [])]
        if query:
            needle = query.casefold()
            pairs = [pair for pair in pairs if needle in pair["_search"]]
        total = len(pairs)
        pages = max(1, math.ceil(total))
        page = min(page, pages)
        pair = pairs[page - 1] if pairs else None

        def page_url(target: int) -> str:
            target = min(max(1, target), pages)
            return url_for("index", page=target, q=query, split=split, asset_type=asset_type)

        return render_template_string(
            PAGE_TEMPLATE,
            pair=pair,
            total=total,
            page=page,
            pages=pages,
            query=query,
            split=split,
            asset_type=asset_type,
            splits=splits,
            asset_types=asset_types,
            stats=stats,
            error=error,
            page_url=page_url,
            render_table=render_table,
        )

    @app.get("/reload")
    def reload() -> Any:
        cached_context.cache_clear()
        return redirect(url_for("index"))

    @app.get("/asset/<asset_id>")
    def asset_image(asset_id: str) -> Any:
        context = cached_context()
        asset = context["assets"].get(asset_id)
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
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.host = resolve_gui_host(args.host, args.lan)
    app = create_app(Path(args.output_dir), args.max_rows, args.max_paths, args.max_asset_chars)
    print(format_gui_urls("MM joinability viewer", args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Quality checker for a deterministic random sample of implicit MM queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, redirect, render_template_string, request, send_file, url_for

from mm_joinability_dataset_viewer import (
    ViewerDataset,
    clean_text,
    render_table,
    resolve_asset_file,
)
from stage1_gui import format_gui_urls, resolve_gui_host


CHECKER_SCHEMA_VERSION = "mm-joinability-quality-checker-v1"
VALID_RATINGS = {"qualified", "unqualified"}

PAGE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ dataset_name }} 数据质量 Checker</title>
  <style>
    :root { --bg:#f5f7fa; --panel:#fff; --ink:#17202a; --muted:#667085; --line:#d6dce5;
      --blue:#1769e0; --green:#087443; --red:#b42318; --amber:#9a6700; --soft:#eef2f7; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    header { position:sticky; top:0; z-index:5; padding:13px 20px; background:rgba(255,255,255,.97);
      border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:16px; }
    h1 { margin:0; font-size:19px; } h2 { margin:0 0 10px; font-size:17px; } h3 { margin:0 0 6px; font-size:14px; }
    main { width:min(1580px,100%); margin:auto; padding:18px 20px 42px; }
    .muted { color:var(--muted); } .panel { background:var(--panel); border:1px solid var(--line); border-radius:9px;
      padding:14px; margin-bottom:14px; } .stats { display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); gap:10px; }
    .stat { padding:10px; border-radius:7px; background:#f8fafc; } .label { display:block; color:var(--muted); font-size:12px; }
    .value { font-size:18px; font-weight:700; overflow-wrap:anywhere; } .result { border:2px solid var(--green); background:#ecfdf3; }
    .result.pending { border-color:var(--amber); background:#fffaeb; }
    .review { position:sticky; top:70px; z-index:4; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    input,textarea,button,a.button { border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--ink);
      padding:8px 11px; font:inherit; text-decoration:none; } textarea { min-width:300px; min-height:40px; flex:1; resize:vertical; }
    button,a.button { cursor:pointer; display:inline-flex; align-items:center; justify-content:center; min-height:38px; }
    .good { background:var(--green); border-color:var(--green); color:white; } .bad { background:var(--red); border-color:var(--red); color:white; }
    .primary { background:var(--blue)!important; border-color:var(--blue)!important; color:white!important; }
    .badge { display:inline-flex; padding:2px 8px; border-radius:999px; background:var(--soft); font-size:12px; font-weight:650; }
    .badge.good-text { background:#dcfae6; color:var(--green); } .badge.bad-text { background:#fee4e2; color:var(--red); }
    .query-head,.pair-head,.pager,.row { display:flex; gap:10px; align-items:center; justify-content:space-between; flex-wrap:wrap; }
    .recovery-list { display:grid; gap:10px; }
    .recovery-head,.recovery-row { display:grid; grid-template-columns:minmax(220px,.85fr) minmax(340px,1.35fr) minmax(210px,.72fr) minmax(250px,1fr); gap:10px; }
    .recovery-head { padding:0 10px; color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
    .recovery-row { padding:10px; border:1px solid var(--line); border-left:4px solid #93c5fd; border-radius:8px; background:#fbfcff; }
    .recovery-cell { min-width:0; padding:10px; border:1px solid var(--line); border-radius:7px; background:white; }
    .cell-list { display:grid; gap:4px; margin-top:8px; }
    .cell-item { display:grid; grid-template-columns:minmax(70px,.45fr) minmax(90px,1fr); gap:7px; padding-top:4px; border-top:1px solid #eef2f7; }
    .attribute-value { margin:8px 0; padding:9px; border-radius:6px; background:#ecfdf3; color:var(--green); font-size:16px; font-weight:700; overflow-wrap:anywhere; }
    .missing { padding:10px; border-radius:6px; background:#fff1f0; color:var(--red); }
    .full-tables { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
    .full-table { min-width:0; margin:0; }
    .full-table .table-box { max-height:520px; overflow:auto; }
    .table-box { overflow:auto; } table { width:100%; border-collapse:collapse; table-layout:fixed; font-size:12px; }
    th,td { padding:5px 6px; border:1px solid var(--line); vertical-align:top; overflow-wrap:anywhere; }
    th { background:#eef2f7; text-align:left; } tr.highlight td { background:#fff7e0; }
    .asset-text { margin-top:6px; max-height:230px; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; }
    .asset-image { display:block; max-width:100%; max-height:300px; object-fit:contain; margin-top:7px; border:1px solid var(--line); border-radius:6px; }
    .pair { border-top:4px solid #dbeafe; } .pager { justify-content:center; margin-top:18px; }
    @media(max-width:1150px) { .stats { grid-template-columns:repeat(3,1fr); } .full-tables { grid-template-columns:1fr; } .recovery-head { display:none; } .recovery-row { grid-template-columns:1fr 1fr; } }
    @media(max-width:760px) { .recovery-row { grid-template-columns:1fr; } }
    @media(max-width:650px) { main { padding:12px; } header { position:static; } .review { position:static; } .stats { grid-template-columns:repeat(2,1fr); } textarea { min-width:100%; } }
  </style>
</head>
<body>
<header>
  <div><h1>{{ dataset_name }} 数据质量 Checker</h1><div class="muted">固定随机种子 {{ seed }} · 抽取 implicit query 的 {{ sample_percent }}</div></div>
  <div class="row"><a class="button" href="{{ url_for('next_unreviewed') }}">下一个未评</a><a class="button" href="{{ url_for('export_reviews') }}">导出审核结果</a></div>
</header>
<main>
  {% if error %}<section class="panel"><strong>无法加载 Checker</strong><div class="muted">{{ error }}</div></section>
  {% else %}
  <section class="panel stats">
    <div class="stat"><span class="label">Implicit query 总数</span><span class="value">{{ summary.population }}</span></div>
    <div class="stat"><span class="label">{{ sample_percent }} 抽检数</span><span class="value">{{ summary.sampled }}</span></div>
    <div class="stat"><span class="label">审核进度</span><span class="value">{{ summary.reviewed }}/{{ summary.sampled }}</span></div>
    <div class="stat"><span class="label">合格</span><span class="value">{{ summary.qualified }}</span></div>
    <div class="stat"><span class="label">不合格</span><span class="value">{{ summary.unqualified }}</span></div>
    <div class="stat"><span class="label">当前合格率</span><span class="value">{{ summary.current_rate }}</span></div>
  </section>
  <section class="panel result {{ '' if summary.complete else 'pending' }}">
    {% if summary.complete %}<strong>抽检完成：这 {{ sample_percent }} 数据中有 {{ summary.final_rate }} 是合格数据（{{ summary.qualified }}/{{ summary.sampled }}）。</strong>
    {% else %}<strong>最终统计将在全部审核完成后给出。</strong><span class="muted"> 还剩 {{ summary.pending }} 个 query 未评。</span>{% endif %}
  </section>

  {% if group %}
  <form class="panel review" method="post" action="{{ url_for('review') }}">
    <input type="hidden" name="query_id" value="{{ group.query_id }}"><input type="hidden" name="page" value="{{ page }}">
    <strong>质量判断</strong>
    <span class="badge {{ 'good-text' if group.rating == 'qualified' else 'bad-text' if group.rating == 'unqualified' else '' }}">{{ rating_label }}</span>
    <textarea name="note" placeholder="可选：记录不合格原因或备注">{{ group.note }}</textarea>
    <button class="good" name="rating" value="qualified" type="submit">✓ 合格并继续</button>
    <button class="bad" name="rating" value="unqualified" type="submit">✕ 不合格并继续</button>
    {% if group.rating %}<button name="rating" value="clear" type="submit">清除判断</button>{% endif %}
  </form>

  {% set pair = group.pair %}
  <section class="full-tables">
    <article class="panel full-table">
      <div class="query-head"><div><span class="badge">QUERY TABLE</span><h2>完整 Query 表</h2></div><div class="muted">{{ pair.query_table_id }}</div></div>
      <h3>{{ group.full_query_table.page_title or pair.query_table_id }}</h3>
      <div class="muted">{{ group.full_query_table.caption }} {{ group.full_query_table.section_title }} · {{ group.full_query_table.rows|length }} 行</div>
      <div class="table-box">{{ render_table(group.full_query_table, pair.query_highlight_rows)|safe }}</div>
    </article>
    <article class="panel full-table">
      <div class="query-head"><div><span class="badge">TARGET TABLE</span><h2>完整 Target 表</h2></div><div class="muted">{{ pair.target_table_id }}</div></div>
      <h3>{{ group.full_target_table.page_title or pair.target_table_id }}</h3>
      <div class="muted">{{ group.full_target_table.caption }} {{ group.full_target_table.section_title }} · {{ group.full_target_table.rows|length }} 行</div>
      <div class="table-box">{{ render_table(group.full_target_table, pair.target_highlight_rows)|safe }}</div>
    </article>
  </section>

  <section class="panel pair">
    <div class="pair-head">
      <div><h2>Query Row → Evidence → Attribute → Target Row</h2><div class="muted">{{ pair.query_table.page_title or pair.query_table_id }} · {{ pair.path_count }} 条 recovery path</div></div>
      <div class="muted">{{ pair.query_table_id }} → {{ pair.target_table_id }} · 唯一 attribute/target</div>
    </div>
    <div class="recovery-list">
      <div class="recovery-head"><div>Query row</div><div>Multimodal evidence</div><div>Recovered attribute</div><div>Target row</div></div>
      {% for item in group.review_rows %}
      <article class="recovery-row" data-query-row-id="{{ item.query_row_id }}">
        <div class="recovery-cell">
          <div class="pair-head"><span class="badge">QUERY ROW {{ item.query_row_id }}</span>{% if item.path %}<span class="muted">{{ item.path.query_entity.entity_column_name }}</span>{% endif %}</div>
          {% if item.path %}<h3>{{ item.path.query_entity.cell_text or item.path.query_entity.wiki_title or '未命名 entity' }}</h3>{% endif %}
          <div class="cell-list">{% for cell in item.query_cells %}<div class="cell-item"><span class="label">{{ cell.column }}</span><span>{{ cell.value or '—' }}</span></div>{% endfor %}</div>
        </div>
        <div class="recovery-cell">
          {% if item.path %}
          <div class="pair-head"><strong>{{ item.path.asset_title or item.path.asset_id }}</strong><span class="badge">{{ item.path.asset_type or 'asset' }}</span></div>
          <div class="muted">{{ item.path.asset_id }}{% if item.path.asset_chunk_label %} · {{ item.path.asset_chunk_label }}{% endif %}</div>
          {% if item.path.asset_url %}<a href="{{ item.path.asset_url }}" target="_blank" rel="noreferrer">source</a>{% endif %}
          {% if item.path.asset_type == 'text' %}<div class="asset-text">{{ item.path.asset_content or item.path.model_evidence or '无文本内容' }}</div>
          {% elif item.path.asset_type == 'image' and item.path.asset_image_available %}<img class="asset-image" src="{{ url_for('asset_image', asset_id=item.path.asset_id) }}" alt="evidence">
          {% else %}<div class="muted">本机无可用预览</div>{% endif %}
          {% else %}<div class="missing">这一行没有 evidence recovery</div>{% endif %}
        </div>
        <div class="recovery-cell">
          {% if item.path %}
          <span class="badge">ATTRIBUTE</span><h3>{{ item.path.recovered_attribute.column_name or pair.join_attribute.column_name or pair.join_col_name }}</h3>
          <div class="attribute-value">{{ item.path.recovered_attribute.value or '—' }}</div>
          {% if item.path.recovered_attribute.model_value %}<div><span class="label">Model value</span>{{ item.path.recovered_attribute.model_value }}</div>{% endif %}
          {% if item.path.model_evidence %}<div><span class="label">Model evidence</span>{{ item.path.model_evidence }}</div>{% endif %}
          {% else %}<div class="missing">无可对应 attribute</div>{% endif %}
        </div>
        <div class="recovery-cell">
          <div class="pair-head"><span class="badge">TARGET</span><span class="muted">rows {{ item.target_row_ids|join(', ') or '—' }}</span></div>
          <h3>{{ pair.target_table.page_title or pair.target_table_id }}</h3>
          {% for target_row in item.target_rows %}<div class="cell-list"><div class="muted">target row {{ target_row.row_id }}</div>{% for cell in target_row.cells %}<div class="cell-item"><span class="label">{{ cell.column }}</span><span>{{ cell.value or '—' }}</span></div>{% endfor %}</div>
          {% else %}<div class="muted">没有可显示的 target row。</div>{% endfor %}
        </div>
      </article>
      {% else %}<div class="missing">当前 query 没有可显示的 row。</div>{% endfor %}
    </div>
  </section>

  <nav class="pager"><a class="button" href="{{ page_url(1) }}">首页</a><a class="button" href="{{ page_url(page-1) }}">上一页</a>
    <span>第 {{ page }} / {{ pages }} 页</span><a class="button primary" href="{{ page_url(page+1) }}">下一页</a><a class="button" href="{{ page_url(pages) }}">末页</a></nav>
  {% else %}<section class="panel">数据集中没有可抽检的 implicit query。</section>{% endif %}
  {% endif %}
</main></body></html>
"""


def sampled_query_ids(query_ids: list[str], sample_rate: float, seed: int) -> list[str]:
    """Select an exact, reproducible pseudo-random fraction of unique query IDs."""
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must be within (0, 1]")
    unique_ids = sorted({clean_text(value) for value in query_ids if clean_text(value)})
    if not unique_ids:
        return []
    sample_size = min(len(unique_ids), max(1, math.ceil(len(unique_ids) * sample_rate)))
    return sorted(
        unique_ids,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest(),
    )[:sample_size]


def query_population_signature(query_ids: list[str]) -> str:
    payload = "\n".join(sorted(query_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class QualityReviewStore:
    """Persistent sample membership and one quality judgment per query."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(
        self,
        *,
        sample_ids: list[str],
        population: int,
        population_signature: str,
        sample_rate: float,
        seed: int,
        dataset_name: str,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples (
                    ordinal INTEGER PRIMARY KEY, query_table_id TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    query_table_id TEXT PRIMARY KEY,
                    rating TEXT NOT NULL CHECK (rating IN ('qualified','unqualified')),
                    note TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL,
                    FOREIGN KEY (query_table_id) REFERENCES samples(query_table_id)
                );
                """
            )
            existing = dict(connection.execute("SELECT key, value FROM meta"))
            expected = {
                "schema_version": CHECKER_SCHEMA_VERSION,
                "population_signature": population_signature,
                "population": str(population),
                "sample_rate": repr(sample_rate),
                "seed": str(seed),
                "dataset_name": dataset_name,
            }
            if existing:
                mismatches = [key for key, value in expected.items() if existing.get(key) != value]
                if mismatches:
                    raise ValueError(
                        "审核数据库与当前数据集/抽样配置不匹配（"
                        + ", ".join(mismatches)
                        + "）；请指定新的 --review_db，以免覆盖已有审核结果。"
                    )
                return
            connection.executemany("INSERT INTO meta(key,value) VALUES (?,?)", expected.items())
            connection.executemany(
                "INSERT INTO samples(ordinal,query_table_id) VALUES (?,?)",
                enumerate(sample_ids, start=1),
            )

    def query_id_at(self, page: int) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT query_table_id FROM samples WHERE ordinal = ?", (page,)
            ).fetchone()
        return str(row[0]) if row else None

    def review_for(self, query_id: str) -> dict[str, str]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT rating,note,updated_at FROM reviews WHERE query_table_id = ?",
                (query_id,),
            ).fetchone()
        return dict(row) if row else {"rating": "", "note": "", "updated_at": ""}

    def save(self, query_id: str, rating: str, note: str) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM samples WHERE query_table_id = ?", (query_id,)
            ).fetchone()
            if not exists:
                raise KeyError(query_id)
            if rating == "clear":
                connection.execute("DELETE FROM reviews WHERE query_table_id = ?", (query_id,))
                return
            if rating not in VALID_RATINGS:
                raise ValueError(f"invalid rating: {rating}")
            connection.execute(
                """
                INSERT INTO reviews(query_table_id,rating,note,updated_at) VALUES (?,?,?,?)
                ON CONFLICT(query_table_id) DO UPDATE SET
                    rating=excluded.rating,note=excluded.note,updated_at=excluded.updated_at
                """,
                (query_id, rating, clean_text(note), datetime.now(timezone.utc).isoformat()),
            )

    def summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            meta = dict(connection.execute("SELECT key,value FROM meta"))
            sampled = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
            counts = dict(
                connection.execute("SELECT rating,COUNT(*) FROM reviews GROUP BY rating")
            )
        qualified = int(counts.get("qualified", 0))
        unqualified = int(counts.get("unqualified", 0))
        reviewed = qualified + unqualified
        complete = sampled > 0 and reviewed == sampled
        return {
            "population": int(meta.get("population", 0)), "sampled": sampled,
            "reviewed": reviewed, "qualified": qualified, "unqualified": unqualified,
            "pending": sampled - reviewed, "complete": complete,
            "current_rate": f"{qualified / reviewed:.2%}" if reviewed else "—",
            "final_rate": f"{qualified / sampled:.2%}" if complete else "—",
        }

    def next_unreviewed_page(self, after_page: int = 0) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.ordinal FROM samples s LEFT JOIN reviews r USING(query_table_id)
                WHERE r.query_table_id IS NULL AND s.ordinal > ? ORDER BY s.ordinal LIMIT 1
                """,
                (after_page,),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT s.ordinal FROM samples s LEFT JOIN reviews r USING(query_table_id)
                    WHERE r.query_table_id IS NULL ORDER BY s.ordinal LIMIT 1
                    """
                ).fetchone()
        return int(row[0]) if row else None

    def export_rows(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.ordinal,s.query_table_id,COALESCE(r.rating,'') rating,
                       COALESCE(r.note,'') note,COALESCE(r.updated_at,'') updated_at
                FROM samples s LEFT JOIN reviews r USING(query_table_id) ORDER BY s.ordinal
                """
            ).fetchall()
        return [dict(row) for row in rows]


def _labeled_row_cells(
    row: dict[str, Any] | None,
    columns: list[str],
) -> list[dict[str, str]]:
    if not row:
        return []
    values = row.get("cells") if isinstance(row.get("cells"), list) else []
    return [
        {
            "column": clean_text(column) or f"col_{index}",
            "value": clean_text(values[index]) if index < len(values) else "",
        }
        for index, column in enumerate(columns)
    ]


def build_review_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    """Align each displayed query row with its evidence, attribute, and target rows."""
    query_table = pair.get("query_table") or {}
    target_table = pair.get("target_table") or {}
    query_columns = list(query_table.get("columns") or [])
    target_columns = list(target_table.get("columns") or [])
    query_rows = list(query_table.get("rows") or [])
    target_rows = {
        clean_text(row.get("row_id")): row
        for row in target_table.get("rows") or []
        if isinstance(row, dict)
    }
    paths_by_query_row: dict[str, list[dict[str, Any]]] = {}
    for path in pair.get("paths") or []:
        row_id = clean_text(path.get("query_row_id"))
        paths_by_query_row.setdefault(row_id, []).append(path)

    review_rows: list[dict[str, Any]] = []

    def append_row(query_row: dict[str, Any] | None, path: dict[str, Any] | None) -> None:
        query_row_id = clean_text(
            query_row.get("row_id") if query_row is not None else path.get("query_row_id")
        )
        target_row_ids = list(path.get("target_row_ids") or []) if path else []
        review_rows.append(
            {
                "query_row_id": query_row_id or "?",
                "query_cells": _labeled_row_cells(query_row, query_columns),
                "path": path,
                "target_row_ids": [clean_text(row_id) for row_id in target_row_ids],
                "target_rows": [
                    {
                        "row_id": clean_text(target_rows[row_id].get("row_id")),
                        "cells": _labeled_row_cells(target_rows[row_id], target_columns),
                    }
                    for row_id in (clean_text(value) for value in target_row_ids)
                    if row_id in target_rows
                ],
            }
        )

    displayed_query_ids: set[str] = set()
    for query_row in query_rows:
        if not isinstance(query_row, dict):
            continue
        row_id = clean_text(query_row.get("row_id"))
        displayed_query_ids.add(row_id)
        matching_paths = paths_by_query_row.get(row_id) or [None]
        for path in matching_paths:
            append_row(query_row, path)

    for row_id, paths in paths_by_query_row.items():
        if row_id in displayed_query_ids:
            continue
        for path in paths:
            append_row(None, path)
    return review_rows


def create_checker_app(
    output_dir: Path,
    *,
    dataset_name: str,
    sample_rate: float = 0.01,
    seed: int = 13,
    review_db: Path | None = None,
    max_rows: int = 12,
    max_paths: int = 50,
    max_asset_chars: int = 2400,
    index_path: Path | None = None,
) -> Flask:
    output_dir = Path(output_dir).resolve()
    dataset = ViewerDataset(output_dir, max_rows, max_paths, max_asset_chars, index_path=index_path)
    dataset.validate_implicit_query_uniqueness()
    implicit_ids = dataset.implicit_query_ids()
    selected_ids = sampled_query_ids(implicit_ids, sample_rate, seed)
    db_path = review_db or output_dir / f".{dataset_name.casefold()}_quality_checker.sqlite3"
    store = QualityReviewStore(db_path)
    store.initialize(
        sample_ids=selected_ids, population=len(implicit_ids),
        population_signature=query_population_signature(implicit_ids),
        sample_rate=sample_rate, seed=seed, dataset_name=dataset_name,
    )
    app = Flask(__name__)
    app.config.update(QUALITY_CHECKER_STORE=store, QUALITY_CHECKER_DATASET=dataset)

    def hydrate_group(query_id: str) -> dict[str, Any]:
        pair_keys = dataset.pair_keys_for_query(query_id, implicit_only=True)
        if len(pair_keys) != 1:
            raise ValueError(
                "checker requires exactly one implicit qrel for "
                f"{query_id!r}; found {len(pair_keys)}"
            )
        pair = dataset.hydrate_pair(pair_keys[0])
        full_query_table, full_target_table = dataset.hydrate_full_pair_tables(
            pair_keys[0]
        )
        review_data = store.review_for(query_id)
        return {
            "query_id": query_id,
            "query_table": pair["query_table"],
            "query_highlight_rows": pair["query_highlight_rows"],
            "pair": pair,
            "full_query_table": full_query_table,
            "full_target_table": full_target_table,
            "review_rows": build_review_rows(pair),
            **review_data,
        }

    @app.get("/")
    def index() -> str:
        summary = store.summary()
        pages = max(1, summary["sampled"])
        try:
            page = min(pages, max(1, int(request.args.get("page", "1"))))
        except ValueError:
            page = 1
        query_id = store.query_id_at(page)
        group = hydrate_group(query_id) if query_id else None
        rating_label = {"qualified": "已评：合格", "unqualified": "已评：不合格"}.get(
            group["rating"] if group else "", "未评"
        )

        def page_url(target: int) -> str:
            return url_for("index", page=min(pages, max(1, target)))

        return render_template_string(
            PAGE_TEMPLATE, dataset_name=dataset_name, seed=seed,
            sample_percent=f"{sample_rate:.2%}", summary=summary, group=group,
            page=page, pages=pages, rating_label=rating_label,
            page_url=page_url, render_table=render_table, error="",
        )

    @app.post("/review")
    def review() -> Any:
        query_id = clean_text(request.form.get("query_id"))
        rating = clean_text(request.form.get("rating"))
        note = clean_text(request.form.get("note"))
        try:
            page = max(1, int(request.form.get("page", "1")))
            store.save(query_id, rating, note)
        except (KeyError, ValueError) as exc:
            abort(400, str(exc))
        next_page = store.next_unreviewed_page(after_page=page)
        return redirect(url_for("index", page=next_page or page))

    @app.get("/next-unreviewed")
    def next_unreviewed() -> Any:
        try:
            current = max(0, int(request.args.get("after", "0")))
        except ValueError:
            current = 0
        return redirect(url_for("index", page=store.next_unreviewed_page(current) or 1))

    @app.get("/export")
    def export_reviews() -> Response:
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in store.export_rows())
        return Response(
            content, mimetype="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{dataset_name.casefold()}_quality_reviews.jsonl"'},
        )

    @app.get("/asset/<asset_id>")
    def asset_image(asset_id: str) -> Any:
        asset = dataset.asset(asset_id)
        if not asset:
            abort(404, f"Unknown asset_id: {asset_id}")
        image_path = resolve_asset_file(output_dir, asset)
        if image_path is None:
            abort(404, f"No local image file for asset_id: {asset_id}")
        return send_file(image_path)

    return app


def checker_parser(default_output_dir: str, default_port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--sample_rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--review_db", default=None)
    parser.add_argument("--index_path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--max_rows", type=int, default=12)
    parser.add_argument("--max_paths", type=int, default=50)
    parser.add_argument("--max_asset_chars", type=int, default=2400)
    parser.add_argument("--debug", action="store_true")
    return parser


def run_checker(dataset_name: str, default_output_dir: str, default_port: int) -> None:
    args = checker_parser(default_output_dir, default_port).parse_args()
    host = resolve_gui_host(args.host, args.lan)
    app = create_checker_app(
        Path(args.output_dir), dataset_name=dataset_name, sample_rate=args.sample_rate,
        seed=args.seed, review_db=Path(args.review_db) if args.review_db else None,
        max_rows=args.max_rows, max_paths=args.max_paths,
        max_asset_chars=args.max_asset_chars,
        index_path=Path(args.index_path) if args.index_path else None,
    )
    print(format_gui_urls(f"{dataset_name} quality checker", host, args.port))
    app.run(host=host, port=args.port, debug=args.debug)


def main() -> None:
    run_checker("MM-Joinability", "output_mm_joinability_v15", 7864)


if __name__ == "__main__":
    main()

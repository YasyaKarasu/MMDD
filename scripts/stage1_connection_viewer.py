#!/usr/bin/env python
"""Flask GUI for browsing constructed Stage-1 connection groups."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

from stage1_io import clean_text, iter_jsonl, iter_manifest_records, load_json

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage-1 Connection Viewer</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d7dee9;
      --accent: #1f6feb;
      --green: #0b7a46;
      --red: #b42318;
      --amber: #966700;
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
      gap: 18px;
      padding: 14px 22px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main { width: min(1500px, 100%); margin: 0 auto; padding: 18px 22px 36px; }
    form, .pager, .summary, .flow, .grid { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
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
    .kv { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 10px; width: 100%; }
    .kv div { min-width: 0; }
    .label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }
    .value { overflow-wrap: anywhere; font-weight: 600; }
    .flow { align-items: stretch; }
    .node {
      flex: 1 1 220px;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcff;
    }
    .node h3 { margin: 0 0 4px; font-size: 14px; }
    .arrow { align-self: center; color: var(--muted); font-size: 22px; padding: 0 2px; }
    .grid { align-items: stretch; }
    .table-card { flex: 1 1 420px; min-width: 320px; }
    .table-title { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
    th, td { border: 1px solid var(--line); padding: 5px 6px; vertical-align: top; overflow-wrap: anywhere; }
    th { background: #eef2f7; text-align: left; }
    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef2f7;
      color: #344054;
      font-size: 12px;
      font-weight: 600;
    }
    .badge.train { background: #e7f7ef; color: var(--green); }
    .badge.dev { background: #fff5d9; color: var(--amber); }
    .badge.test { background: #fde7e7; color: var(--red); }
    .list { display: grid; gap: 8px; }
    .item { border: 1px solid var(--line); border-radius: 6px; padding: 9px; background: #fbfcff; }
    .item-head { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 4px; }
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
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #344054;
    }
    .asset-image {
      display: block;
      max-width: 100%;
      max-height: 280px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      margin-top: 8px;
    }
    .pager { justify-content: center; margin: 16px 0 4px; }
    .pager input { width: 88px; }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .kv { grid-template-columns: 1fr; }
      .arrow { display: none; }
      .path-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Stage-1 Connection Viewer</h1>
      <div class="muted">{{ total }} groups{% if query %} matching "{{ query }}"{% endif %}</div>
    </div>
    <form method="get" action="{{ url_for('index') }}">
      <input name="q" value="{{ query }}" placeholder="chain, table, column, text">
      <select name="split">
        <option value="">all splits</option>
        {% for s in splits %}
        <option value="{{ s }}" {{ "selected" if split == s else "" }}>{{ s }}</option>
        {% endfor %}
      </select>
      <button class="primary" type="submit">Filter</button>
    </form>
  </header>

  <main>
    {% if group %}
    <section class="panel summary">
      <div class="kv">
        <div><span class="label">Group</span><span class="value">{{ group.chain_id }}</span></div>
        <div><span class="label">Source Table</span><span class="value">{{ group.source_table_id }}</span></div>
        <div><span class="label">Page</span><span class="value">{{ group.page_title }}</span></div>
        <div><span class="label">Split</span><span class="badge {{ group.split }}">{{ group.split }}</span></div>
      </div>
    </section>

    <section class="panel">
      <div class="flow">
        <div class="node">
          <h3>Visible Query</h3>
          <div class="muted">{{ group.visible.statement or "missing" }}</div>
          <div>{{ group.visible.fragment_id or "" }}</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <h3>Hidden Query</h3>
          <div class="muted">{{ group.hidden.statement or "missing" }}</div>
          <div>{{ group.hidden.fragment_id or "" }}</div>
        </div>
        <div class="arrow">→</div>
        <div class="node">
          <h3>Target Table</h3>
          <div class="muted">{{ group.target.statement or "missing" }}</div>
          <div>{{ group.target.fragment_id or "" }}</div>
        </div>
      </div>
    </section>

    <section class="grid">
      {% for frag in [group.visible, group.hidden, group.target] %}
      {% if frag.fragment_id %}
      <article class="panel table-card">
        <div class="table-title">
          <strong>{{ frag.role }}</strong>
          <span class="muted">{{ frag.rows|length }} rows</span>
        </div>
        {{ render_table(frag) | safe }}
      </article>
      {% endif %}
      {% endfor %}
    </section>

    <section class="panel">
      <h2>Logic Pairs</h2>
      <div class="list">
        {% for pair in group.pairs %}
        <div class="item">
          <div class="item-head"><strong>{{ pair.pair_id }}</strong><span class="badge">label={{ pair.label }} weight={{ pair.weight }}</span></div>
          <div>{{ pair.reason }}</div>
        </div>
        {% else %}
        <div class="muted">No logic pairs for this group.</div>
        {% endfor %}
      </div>
    </section>

    <section class="panel">
      <h2>Qrels</h2>
      <div class="list">
        {% for qrel in group.qrels %}
        <div class="item">
          <div class="item-head">
            <strong>{{ qrel.query_role }} → {{ qrel.target_role }}</strong>
            <span class="badge">rel={{ qrel.rel }}</span>
          </div>
          <div class="muted">{{ qrel.query_id }} → {{ qrel.target_id }}</div>
        </div>
        {% else %}
        <div class="muted">No qrels for this group.</div>
        {% endfor %}
      </div>
    </section>

    <section class="panel">
      <h2>Evidence Paths</h2>
      <div class="list">
        {% for path in group.evidence_paths %}
        <div class="item">
          <div class="item-head">
            <strong>{{ path.claim_text or path.path_id }}</strong>
            <span class="badge">{{ path.asset_type }}{% if path.weak_label %} · {{ path.weak_label }}{% endif %}{% if path.human_label is defined and path.human_label is not none %} · human={{ path.human_label }}{% endif %}</span>
          </div>
          <div class="path-grid">
            <div class="asset-box">
              <span class="label">Path</span>
              <div><strong>{{ path.query_fragment_id }}</strong></div>
              <div class="muted">→ {{ path.asset_id }}</div>
              <div><strong>→ {{ path.target_fragment_id }}</strong></div>
              <hr>
              <div><span class="label">Entity</span>{{ path.entity_text or path.asset_title }}</div>
              <div><span class="label">Bridge</span>{{ path.bridge_col_name }} = {{ path.bridge_value }}</div>
              <div><span class="label">Target Bridge</span>{{ path.target_bridge_col_name }}</div>
              <div><span class="label">Path ID</span><span class="muted">{{ path.path_id }}</span></div>
            </div>
            <div class="asset-box">
              <div class="item-head">
                <strong>{{ path.asset_title or path.asset_id }}</strong>
                <span class="badge">{{ path.asset_source or "asset" }}</span>
              </div>
              {% if path.asset_url %}
              <div><a href="{{ path.asset_url }}" target="_blank" rel="noreferrer">{{ path.asset_url }}</a></div>
              {% endif %}
              {% if path.asset_type == "text" %}
              <div class="asset-text">{{ path.asset_content_snippet or "No text content available." }}</div>
              {% elif path.asset_type == "image" %}
              <div class="muted">{{ path.asset_file_name or path.asset_local_path or "No image filename available." }}</div>
              {% if path.asset_metadata_snippet %}
              <div class="asset-text">{{ path.asset_metadata_snippet }}</div>
              {% endif %}
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
        <div class="muted">No evidence paths for this group.</div>
        {% endfor %}
      </div>
    </section>
    {% else %}
    <section class="panel"><strong>No groups found.</strong></section>
    {% endif %}

    <nav class="pager">
      <a class="button" href="{{ page_url(1) }}">First</a>
      <a class="button" href="{{ page_url(page - 1) }}">Prev</a>
      <form method="get" action="{{ url_for('index') }}">
        <input type="hidden" name="q" value="{{ query }}">
        <input type="hidden" name="split" value="{{ split }}">
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


def table_preview(fragment: dict[str, Any], max_rows: int) -> dict[str, Any]:
    if not fragment:
        return {}
    columns = [clean_text(col.get("column_name")) for col in fragment.get("columns", [])]
    rows = []
    for row in fragment.get("rows", [])[:max_rows]:
        cells = row.get("cells", [])
        rows.append({columns[i]: clean_text(cell.get("text")) for i, cell in enumerate(cells) if i < len(columns)})
    return {
        "fragment_id": fragment.get("fragment_id", ""),
        "role": fragment.get("role", ""),
        "statement": fragment.get("statement", ""),
        "columns": columns,
        "rows": rows,
    }


def render_table(fragment: dict[str, Any]) -> str:
    columns = fragment.get("columns", [])
    rows = fragment.get("rows", [])
    head = "".join(f"<th>{escape_html(col)}</th>" for col in columns)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape_html(row.get(col, ''))}</td>" for col in columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def escape_html(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def group_search_text(group: dict[str, Any]) -> str:
    parts = [
        group.get("chain_id", ""),
        group.get("source_table_id", ""),
        group.get("page_title", ""),
        group.get("split", ""),
    ]
    for key in ("visible", "hidden", "target"):
        frag = group.get(key) or {}
        parts.append(frag.get("statement", ""))
        parts.extend(frag.get("columns", []))
    for path in group.get("evidence_paths", []):
        parts.extend(
            [
                path.get("claim_text", ""),
                path.get("entity_text", ""),
                path.get("bridge_value", ""),
                path.get("asset_title", ""),
                path.get("asset_content_snippet", ""),
                path.get("asset_metadata_snippet", ""),
            ]
        )
    return " ".join(clean_text(part).casefold() for part in parts)


def stage1_input_dir(stage1_dir: Path) -> Path:
    manifest_path = stage1_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        input_dir = (
            manifest.get("evidence_paths", {}).get("input_dir")
            or manifest.get("logic_connectivity", {}).get("input_dir")
            or "output_medium"
        )
        return Path(input_dir)
    return Path("output_medium")


def load_assets(stage1_dir: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    input_dir = stage1_input_dir(stage1_dir)
    assets: dict[str, dict[str, Any]] = {}
    if not (input_dir / "dataset_manifest.json").exists():
        return assets, input_dir
    for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000):
        if asset.get("asset_id"):
            assets[asset["asset_id"]] = asset
    return assets, input_dir


def resolve_asset_file(input_dir: Path, asset: dict[str, Any]) -> Path | None:
    candidates = []
    local = clean_text(asset.get("local_path"))
    if local:
        candidates.append(Path(local))
    relative = clean_text(asset.get("relative_path"))
    if relative:
        candidates.append(input_dir / relative)
    file_name = clean_text(asset.get("file_name"))
    if file_name:
        candidates.append(input_dir / "images" / file_name)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def image_metadata_snippet(asset: dict[str, Any], max_chars: int = 900) -> str:
    metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    ext = metadata.get("extmetadata") if isinstance(metadata.get("extmetadata"), dict) else {}
    parts = []
    for key in ("ObjectName", "ImageDescription", "Categories", "Credit", "Artist"):
        value = ext.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        value = re.sub(r"<[^>]+>", " ", clean_text(value))
        if value:
            parts.append(f"{key}: {value}")
    return "\n".join(parts)[:max_chars]


def enrich_evidence_path(path: dict[str, Any], asset: dict[str, Any] | None, input_dir: Path, max_text_chars: int = 1400) -> dict[str, Any]:
    out = dict(path)
    asset = asset or {}
    out["asset_title"] = clean_text(asset.get("entity_wiki_title")) or clean_text(path.get("entity_text"))
    out["asset_source"] = clean_text(asset.get("source"))
    out["asset_url"] = clean_text(asset.get("url") or asset.get("description_url") or asset.get("image_url"))
    out["asset_file_name"] = clean_text(asset.get("file_name"))
    out["asset_content_snippet"] = clean_text(asset.get("content"))[:max_text_chars]
    local_file = resolve_asset_file(input_dir, asset)
    out["asset_local_path"] = str(local_file) if local_file else clean_text(asset.get("local_path") or asset.get("relative_path"))
    out["asset_image_available"] = bool(local_file and out.get("asset_type") == "image")
    out["asset_metadata_snippet"] = image_metadata_snippet(asset)
    return out


def load_groups(
    stage1_dir: Path,
    max_rows: int,
    max_evidence_paths: int,
    assets: dict[str, dict[str, Any]] | None = None,
    input_dir: Path | None = None,
) -> list[dict[str, Any]]:
    assets = assets or {}
    input_dir = input_dir or stage1_input_dir(stage1_dir)
    fragments_by_chain: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl"):
        role = rec.get("role", "unknown")
        fragments_by_chain[rec["chain_id"]][role] = rec

    pairs_by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pairs_path = stage1_dir / "logic_pairs.jsonl"
    if pairs_path.exists():
        for rec in iter_jsonl(pairs_path):
            pairs_by_chain[rec["chain_id"]].append(rec)

    qrels_by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qrels_path = stage1_dir / "qrels.jsonl"
    if qrels_path.exists():
        for rec in iter_jsonl(qrels_path):
            qrels_by_chain[rec["chain_id"]].append(rec)

    evidence_by_chain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_path = stage1_dir / "hitl_pool.jsonl"
    if not evidence_path.exists():
        evidence_path = stage1_dir / "evidence_paths.jsonl"
    if evidence_path.exists():
        for rec in iter_jsonl(evidence_path):
            bucket = evidence_by_chain[rec["chain_id"]]
            if len(bucket) < max_evidence_paths:
                bucket.append(enrich_evidence_path(rec, assets.get(rec.get("asset_id")), input_dir))

    groups = []
    for chain_id, fragments in fragments_by_chain.items():
        visible = fragments.get("left_visible", {})
        hidden = fragments.get("left_hidden", {})
        target = fragments.get("right_target", {})
        meta = visible or hidden or target
        group = {
            "chain_id": chain_id,
            "source_table_id": meta.get("source_table_id", ""),
            "page_title": meta.get("page_title", ""),
            "split": meta.get("split", ""),
            "visible": table_preview(visible, max_rows),
            "hidden": table_preview(hidden, max_rows),
            "target": table_preview(target, max_rows),
            "pairs": pairs_by_chain.get(chain_id, []),
            "qrels": qrels_by_chain.get(chain_id, []),
            "evidence_paths": evidence_by_chain.get(chain_id, []),
        }
        group["_search"] = group_search_text(group)
        groups.append(group)
    groups.sort(key=lambda item: (item.get("split", ""), item.get("page_title", ""), item["chain_id"]))
    return groups


def create_app(stage1_dir: Path, max_rows: int, max_evidence_paths: int) -> Flask:
    app = Flask(__name__)
    stage1_dir = Path(stage1_dir)

    @lru_cache(maxsize=1)
    def cached_assets() -> tuple[dict[str, dict[str, Any]], Path]:
        return load_assets(stage1_dir)

    @lru_cache(maxsize=1)
    def cached_groups() -> tuple[dict[str, Any], ...]:
        assets, input_dir = cached_assets()
        return tuple(load_groups(stage1_dir, max_rows, max_evidence_paths, assets, input_dir))

    @app.get("/")
    def index() -> str:
        if not (stage1_dir / "logic_fragments.jsonl").exists():
            abort(404, f"Missing {stage1_dir / 'logic_fragments.jsonl'}")
        query = clean_text(request.args.get("q", ""))
        split = clean_text(request.args.get("split", ""))
        try:
            page = max(1, int(request.args.get("page", "1") or "1"))
        except ValueError:
            page = 1
        groups = list(cached_groups())
        splits = sorted({group["split"] for group in groups if group.get("split")})
        if split:
            groups = [group for group in groups if group.get("split") == split]
        if query:
            needle = query.casefold()
            groups = [group for group in groups if needle in group["_search"]]
        total = len(groups)
        pages = max(1, math.ceil(total))
        page = min(page, pages)
        group = groups[page - 1] if groups else None

        def page_url(target: int) -> str:
            target = min(max(1, target), pages)
            return url_for("index", page=target, q=query, split=split)

        return render_template_string(
            PAGE_TEMPLATE,
            group=group,
            total=total,
            page=page,
            pages=pages,
            query=query,
            split=split,
            splits=splits,
            page_url=page_url,
            render_table=render_table,
        )

    @app.get("/reload")
    def reload() -> Any:
        cached_assets.cache_clear()
        cached_groups.cache_clear()
        return redirect(url_for("index"))

    @app.get("/asset/<asset_id>")
    def asset_image(asset_id: str) -> Any:
        assets, input_dir = cached_assets()
        asset = assets.get(asset_id)
        if not asset:
            abort(404, f"Unknown asset_id: {asset_id}")
        image_path = resolve_asset_file(input_dir, asset)
        if image_path is None:
            abort(404, f"No local image file for asset_id: {asset_id}")
        return send_file(image_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--max_rows", type=int, default=8)
    parser.add_argument("--max_evidence_paths", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.stage1_dir), args.max_rows, args.max_evidence_paths)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Flask GUI for inspecting Stage-1 recalled targets per query."""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

from eval_stage1_recall import beam_search_tables, load_hnsw, load_projected, load_student, relation_query_from_projected
from stage1_connection_viewer import load_assets, resolve_asset_file, table_preview, render_table
from stage1_gui import format_gui_urls, resolve_gui_host
from stage1_io import clean_text, iter_jsonl
from train_student import TYPES, load_embeddings

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage-1 Recall Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d6dce5;
      --accent: #2563eb;
      --hit: #0b7a46;
      --miss: #9b1c1c;
      --soft: #eef2f7;
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
    main { width: min(1500px, 100%); margin: 0 auto; padding: 18px 22px 36px; }
    form, .pager, .summary, .row, .filters { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
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
    .label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }
    .value { overflow-wrap: anywhere; font-weight: 600; }
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
    .badge.hit { background: #e7f7ef; color: var(--hit); }
    .badge.miss { background: #fde7e7; color: var(--miss); }
    .badge.bridge { background: #e8eefc; color: #1d4ed8; }
    .tables { display: grid; grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr); gap: 14px; align-items: start; }
    .table-box { min-width: 0; overflow: auto; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }
    th, td { border: 1px solid var(--line); padding: 5px 6px; vertical-align: top; overflow-wrap: anywhere; }
    th { background: #eef2f7; text-align: left; }
    .target-list { display: grid; gap: 10px; }
    .target {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
      padding: 10px;
    }
    .target-head { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; align-items: flex-start; }
    .target-title { min-width: 0; overflow-wrap: anywhere; }
    .target-table { margin-top: 8px; overflow: auto; }
    .path {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 8px;
      margin-top: 8px;
    }
    .node {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 8px;
      min-width: 0;
    }
    .node-title { font-weight: 650; overflow-wrap: anywhere; }
    .asset-text {
      max-height: 120px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin-top: 6px;
      color: #344054;
    }
    .asset-image {
      display: block;
      max-width: 100%;
      max-height: 180px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      margin-top: 8px;
    }
    .pager { justify-content: center; margin: 16px 0 4px; }
    .pager input { width: 88px; }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .kv, .tables { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Stage-1 Recall Viewer</h1>
      <div class="muted">{{ total }} queries{% if query %} matching "{{ query }}"{% endif %}</div>
    </div>
    <form method="get" action="{{ url_for('index') }}">
      <input name="q" value="{{ query }}" placeholder="query, target, bridge, path">
      <select name="split">
        <option value="">all splits</option>
        {% for s in splits %}
        <option value="{{ s }}" {{ "selected" if split == s else "" }}>{{ s }}</option>
        {% endfor %}
      </select>
      <select name="role">
        <option value="">all roles</option>
        {% for r in roles %}
        <option value="{{ r }}" {{ "selected" if role == r else "" }}>{{ r }}</option>
        {% endfor %}
      </select>
      <button class="primary" type="submit">Filter</button>
      <a class="button" href="{{ url_for('reload') }}">Reload</a>
    </form>
  </header>

    <main>
    {% if error %}
    <section class="panel"><strong>Cannot load recall data.</strong><div class="muted">{{ error }}</div></section>
    {% elif missing_records %}
    <section class="panel">
      <strong>No evaluation recall records found.</strong>
      <div class="muted">{{ missing_records }}</div>
    </section>
    {% elif card %}
    <section class="panel summary">
      <div class="kv">
        <div><span class="label">Query</span><span class="value">{{ card.query_id }}</span></div>
        <div><span class="label">Role</span><span class="value">{{ card.query_role }}</span></div>
        <div><span class="label">Split</span><span class="value">{{ card.split }}</span></div>
        <div><span class="label">Relevant Targets</span><span class="value">{{ card.relevant_targets|length }}</span></div>
      </div>
    </section>

    <section class="tables">
      <article class="panel table-box">
        <h2>Query Fragment</h2>
        <div class="muted">{{ card.query_fragment.statement }}</div>
        {{ render_table(card.query_fragment) | safe }}
      </article>
      <article class="panel table-box">
        <h2>Gold Targets</h2>
        <div class="target-list">
          {% for target in card.relevant_targets %}
          <div class="target">
            <div class="target-head">
              <div class="target-title"><strong>{{ target.title }}</strong><div class="muted">{{ target.target_id }}</div></div>
              <span class="badge hit">rel={{ target.rel }}</span>
            </div>
            <div class="muted">{{ target.page_title }}</div>
            {% if target.target_fragment.columns %}
            <div class="target-table">{{ render_table(target.target_fragment) | safe }}</div>
            {% endif %}
          </div>
          {% else %}
          <div class="muted">No qrels for this query.</div>
          {% endfor %}
        </div>
      </article>
    </section>

    <section class="panel">
      <h2>Direct Table Recall</h2>
      {{ render_targets(card.direct_targets, "direct") | safe }}
    </section>

    <section class="panel">
      <h2>Bridge-aware Recall</h2>
      {{ render_targets(card.path_targets, "path") | safe }}
    </section>
    {% else %}
    <section class="panel"><strong>No queries found.</strong></section>
    {% endif %}

    <nav class="pager">
      <a class="button" href="{{ page_url(1) }}">First</a>
      <a class="button" href="{{ page_url(page - 1) }}">Prev</a>
      <form method="get" action="{{ url_for('index') }}">
        <input type="hidden" name="q" value="{{ query }}">
        <input type="hidden" name="split" value="{{ split }}">
        <input type="hidden" name="role" value="{{ role }}">
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


def fragment_label(fragment: dict[str, Any] | None, fallback: str) -> str:
    fragment = fragment or {}
    return clean_text(fragment.get("statement")) or clean_text(fragment.get("page_title")) or fallback


def asset_label(asset: dict[str, Any] | None, fallback: str) -> str:
    asset = asset or {}
    return clean_text(asset.get("entity_wiki_title")) or clean_text(asset.get("title")) or clean_text(asset.get("entity_text")) or fallback


def object_type_label(object_type: str) -> str:
    return {"table_fragment": "table", "text_asset": "text bridge", "image_asset": "image bridge"}.get(object_type, object_type)


def load_fragments(stage1_dir: Path) -> dict[str, dict[str, Any]]:
    return {rec["fragment_id"]: rec for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl") if rec.get("fragment_id")}


def load_qrels(path: Path) -> list[dict[str, Any]]:
    return [rec for rec in iter_jsonl(path)]


def evidence_path_file(stage1_dir: Path) -> Path | None:
    for name in ("hitl_pool.jsonl", "evidence_paths.jsonl"):
        candidate = stage1_dir / name
        if candidate.exists():
            return candidate
    return None


def load_path_records(stage1_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    path_file = evidence_path_file(stage1_dir)
    if path_file is None:
        return {}
    records = {}
    for rec in iter_jsonl(path_file):
        key = (clean_text(rec.get("query_fragment_id")), clean_text(rec.get("asset_id")), clean_text(rec.get("target_fragment_id")))
        if all(key):
            records[key] = rec
    return records


def make_path_nodes(
    path: list[tuple[str, str]],
    fragments: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    path_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    nodes = []
    for node_id, node_type in path:
        if node_type == "table_fragment":
            fragment = fragments.get(node_id, {})
            nodes.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "type_label": object_type_label(node_type),
                    "title": fragment_label(fragment, node_id),
                    "subtitle": clean_text(fragment.get("role")),
                    "content": clean_text(fragment.get("page_title")),
                    "asset_id": "",
                }
            )
        else:
            asset = assets.get(node_id, {})
            preview = {}
            if path_record and clean_text(path_record.get("asset_id")) == node_id:
                preview = path_record.get("asset_preview") if isinstance(path_record.get("asset_preview"), dict) else {}
            merged_asset = {**asset, **preview}
            content = clean_text(merged_asset.get("content_snippet") or merged_asset.get("content"))
            title = (
                clean_text(merged_asset.get("title"))
                or clean_text(merged_asset.get("entity_wiki_title"))
                or clean_text(path_record.get("entity_text") if path_record else "")
                or node_id
            )
            nodes.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "type_label": object_type_label(node_type),
                    "title": title,
                    "subtitle": clean_text(merged_asset.get("source")),
                    "content": content[:700],
                    "asset_id": node_id if node_type == "image_asset" else "",
                    "file_name": clean_text(merged_asset.get("file_name") or merged_asset.get("local_path") or merged_asset.get("relative_path")),
                    "url": clean_text(merged_asset.get("url")),
                }
            )
    if path_record:
        for node in nodes:
            if node["type"] in {"text_asset", "image_asset"}:
                bridge = " = ".join(
                    part for part in [clean_text(path_record.get("bridge_col_name")), clean_text(path_record.get("bridge_value"))] if part
                )
                if bridge:
                    node["bridge"] = bridge
                node["path_id"] = clean_text(path_record.get("path_id"))
                if path_record.get("claim_text"):
                    node["claim_text"] = clean_text(path_record.get("claim_text"))
    return nodes


def find_path_record(
    query_id: str,
    target_id: str,
    path: list[tuple[str, str]],
    path_records: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    for node_id, node_type in path:
        if node_type in {"text_asset", "image_asset"}:
            record = path_records.get((query_id, node_id, target_id))
            if record:
                return record
    return None


def build_target_item(
    rank: int,
    target_id: str,
    score: float | None,
    path: list[tuple[str, str]],
    relevant: dict[str, int],
    fragments: dict[str, dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    path_records: dict[tuple[str, str, str], dict[str, Any]],
    query_id: str,
    max_rows: int,
    path_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fragment = fragments.get(target_id, {})
    path_record = path_metadata or find_path_record(query_id, target_id, path, path_records)
    path_nodes = make_path_nodes(path, fragments, assets, path_record) if path else []
    return {
        "rank": rank,
        "target_id": target_id,
        "score": score,
        "title": fragment_label(fragment, target_id),
        "page_title": clean_text(fragment.get("page_title")),
        "statement": clean_text(fragment.get("statement")),
        "target_fragment": table_preview(fragment, max_rows),
        "is_relevant": target_id in relevant,
        "rel": relevant.get(target_id),
        "has_bridge": any(node_type in {"text_asset", "image_asset"} for _, node_type in path),
        "path_nodes": path_nodes,
        "path_record": path_record or {},
    }


def assemble_query_cards(
    qrels: list[dict[str, Any]],
    fragments: dict[str, dict[str, Any]],
    direct_by_query: dict[str, list[dict[str, Any]]],
    path_by_query: dict[str, list[dict[str, Any]]],
    path_records: dict[tuple[str, str, str], dict[str, Any]],
    assets: dict[str, dict[str, Any]],
    max_rows: int,
) -> list[dict[str, Any]]:
    qrels_by_query: dict[str, list[dict[str, Any]]] = {}
    for qrel in qrels:
        qrels_by_query.setdefault(qrel["query_id"], []).append(qrel)
    cards = []
    for query_id, query_qrels in qrels_by_query.items():
        first = query_qrels[0]
        query_fragment = fragments.get(query_id, {})
        relevant = {qrel["target_id"]: int(qrel.get("rel", 1)) for qrel in query_qrels}
        relevant_targets = []
        for target_id, rel in sorted(relevant.items()):
            fragment = fragments.get(target_id, {})
            relevant_targets.append(
                {
                    "target_id": target_id,
                    "rel": rel,
                    "title": fragment_label(fragment, target_id),
                    "page_title": clean_text(fragment.get("page_title")),
                    "target_fragment": table_preview(fragment, max_rows),
                }
            )
        direct_targets = [
            build_target_item(
                idx,
                item["target_id"],
                item.get("score"),
                item.get("path", []),
                relevant,
                fragments,
                assets,
                path_records,
                query_id,
                max_rows,
                item.get("path_metadata"),
            )
            for idx, item in enumerate(direct_by_query.get(query_id, []), 1)
        ]
        path_targets = [
            build_target_item(
                idx,
                item["target_id"],
                item.get("score"),
                item.get("path", []),
                relevant,
                fragments,
                assets,
                path_records,
                query_id,
                max_rows,
                item.get("path_metadata"),
            )
            for idx, item in enumerate(path_by_query.get(query_id, []), 1)
        ]
        card = {
            "query_id": query_id,
            "query_role": clean_text(first.get("query_role")),
            "split": clean_text(first.get("split")),
            "chain_id": clean_text(first.get("chain_id")),
            "query_fragment": table_preview(query_fragment, max_rows),
            "relevant_targets": relevant_targets,
            "direct_targets": direct_targets,
            "path_targets": path_targets,
        }
        card["_search"] = query_search_text(card)
        cards.append(card)
    cards.sort(key=lambda item: (item.get("split", ""), item.get("query_role", ""), item.get("query_id", "")))
    return cards


def query_search_text(card: dict[str, Any]) -> str:
    parts = [card.get("query_id", ""), card.get("query_role", ""), card.get("split", ""), card.get("chain_id", "")]
    parts.extend(card.get("query_fragment", {}).get("columns", []))
    parts.append(card.get("query_fragment", {}).get("statement", ""))
    for collection in ("relevant_targets", "direct_targets", "path_targets"):
        for item in card.get(collection, []):
            parts.extend([item.get("target_id", ""), item.get("title", ""), item.get("statement", ""), item.get("page_title", "")])
            for node in item.get("path_nodes", []):
                parts.extend([node.get("id", ""), node.get("title", ""), node.get("bridge", ""), node.get("content", "")])
    return " ".join(clean_text(part).casefold() for part in parts)


def direct_hnsw_rankings(
    model: Any,
    projected: dict[str, tuple[str, np.ndarray]],
    hnsw_dir: Path,
    query_ids: list[str],
    topn: int,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    if not projected:
        return {}
    dim = len(next(iter(projected.values()))[1])
    index, table_ids = load_hnsw(hnsw_dir, "table_fragment", dim)
    if index is None or not table_ids:
        return {}
    rankings = {}
    k = min(max(1, topn), len(table_ids))
    for query_id in query_ids:
        if query_id not in projected:
            rankings[query_id] = []
            continue
        source_type, projected_vec = projected[query_id]
        query = relation_query_from_projected(model, projected_vec, source_type, "table_fragment", device)
        labels, distances = index.knn_query(query, k=k)
        items = []
        for label, distance in zip(labels[0], distances[0]):
            target_id = table_ids[int(label)]
            items.append(
                {
                    "target_id": target_id,
                    "score": float(1.0 - distance),
                    "path": [(query_id, source_type), (target_id, "table_fragment")],
                }
            )
        rankings[query_id] = items
    return rankings


def path_hnsw_rankings(
    args: argparse.Namespace,
    model: Any,
    vectors: dict[str, np.ndarray],
    projected: dict[str, tuple[str, np.ndarray]],
    query_ids: list[str],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    if not projected:
        return {}
    dim = len(next(iter(projected.values()))[1])
    indexes: dict[str, Any] = {}
    ids_by_type: dict[str, list[str]] = {}
    for object_type in TYPES:
        index, ids = load_hnsw(Path(args.hnsw_dir), object_type, dim)
        if index is not None and ids:
            indexes[object_type] = index
            ids_by_type[object_type] = ids
    if "table_fragment" not in indexes:
        return {}
    rankings = {}
    for query_id in query_ids:
        ranked, best_paths = beam_search_tables(args, model, vectors, projected, indexes, ids_by_type, query_id, device)
        items = []
        for target_id in ranked[: args.topn]:
            payload = best_paths.get(target_id, {})
            items.append(
                {
                    "target_id": target_id,
                    "score": float(payload.get("score", 0.0)),
                    "path": payload.get("path", []),
                }
            )
        rankings[query_id] = items
    return rankings


def default_recall_records_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "recall_records", "") or Path(args.stage1_dir) / "recall_rankings.jsonl")


def tuple_path(path: list[Any]) -> list[tuple[str, str]]:
    out = []
    for node in path:
        if isinstance(node, dict):
            node_id = clean_text(node.get("node_id"))
            node_type = clean_text(node.get("node_type"))
        else:
            try:
                node_id, node_type = node
            except (TypeError, ValueError):
                continue
        if node_id and node_type:
            out.append((node_id, node_type))
    return out


def load_recorded_recall_cards(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    records_path = default_recall_records_path(args)
    if not records_path.exists():
        return None
    stage1_dir = Path(args.stage1_dir)
    fragments = load_fragments(stage1_dir)
    assets: dict[str, dict[str, Any]] = {}
    path_records = load_path_records(stage1_dir)
    qrels_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    direct_by_query: dict[str, list[dict[str, Any]]] = {}
    path_by_query: dict[str, list[dict[str, Any]]] = {}
    for record in iter_jsonl(records_path):
        if record.get("sample_kind") != "recall_ranking":
            continue
        query_id = clean_text(record.get("query_id"))
        if not query_id:
            continue
        for relevant in record.get("relevant_targets", []):
            target_id = clean_text(relevant.get("target_id"))
            if target_id:
                qrels_by_key[(query_id, target_id)] = {
                    "query_id": query_id,
                    "target_id": target_id,
                    "rel": int(relevant.get("rel", 1)),
                    "split": record.get("split"),
                    "query_role": record.get("query_role"),
                    "chain_id": relevant.get("chain_id") or record.get("chain_id"),
                    "target_role": relevant.get("target_role"),
                }
        bucket = path_by_query if record.get("retrieval_mode") == "bridge_aware" else direct_by_query
        items = []
        for target in record.get("targets", []):
            target_id = clean_text(target.get("target_id"))
            if not target_id:
                continue
            path = tuple_path(target.get("path", []))
            if not path and record.get("retrieval_mode") == "direct_table":
                path = [(query_id, "table_fragment"), (target_id, "table_fragment")]
            items.append(
                {
                    "target_id": target_id,
                    "score": target.get("score"),
                    "path": path,
                    "path_metadata": target.get("path_metadata") or {},
                }
            )
        bucket[query_id] = items
    qrels = list(qrels_by_key.values())
    return assemble_query_cards(qrels, fragments, direct_by_query, path_by_query, path_records, assets, args.max_rows)


def load_runtime_context(args: argparse.Namespace) -> dict[str, Any]:
    recorded_cards = load_recorded_recall_cards(args)
    if recorded_cards is not None:
        return {
            "recorded": True,
            "cards_by_query": {card["query_id"]: card for card in recorded_cards},
            "query_ids": [card["query_id"] for card in recorded_cards],
            "summaries": {
                card["query_id"]: {
                    "query_id": card["query_id"],
                    "query_role": card.get("query_role", ""),
                    "split": card.get("split", ""),
                    "_search": card.get("_search", ""),
                }
                for card in recorded_cards
            },
        }
    if not getattr(args, "dynamic", False):
        records_path = default_recall_records_path(args)
        return {
            "recorded": False,
            "missing_records": f"Run evaluation first to create {records_path}, or restart this viewer with --dynamic to compute recall on demand.",
            "query_ids": [],
            "summaries": {},
        }
    stage1_dir = Path(args.stage1_dir)
    qrels = load_qrels(Path(args.qrels))
    fragments = load_fragments(stage1_dir)
    if args.query_limit and args.query_limit > 0:
        keep = set(sorted({qrel["query_id"] for qrel in qrels})[: args.query_limit])
        qrels = [qrel for qrel in qrels if qrel["query_id"] in keep]
    qrels_by_query: dict[str, list[dict[str, Any]]] = {}
    for qrel in qrels:
        qrels_by_query.setdefault(qrel["query_id"], []).append(qrel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors, _, _, _ = load_embeddings(Path(args.embedding_dir))
    model = load_student(Path(args.student_dir), device)
    projected = load_projected(Path(args.student_dir))
    dim = len(next(iter(projected.values()))[1]) if projected else 0
    indexes: dict[str, Any] = {}
    ids_by_type: dict[str, list[str]] = {}
    if dim:
        object_types = ("table_fragment",) if args.skip_path else TYPES
        for object_type in object_types:
            index, ids = load_hnsw(Path(args.hnsw_dir), object_type, dim)
            if index is not None and ids:
                indexes[object_type] = index
                ids_by_type[object_type] = ids
    assets, _ = load_assets(stage1_dir)
    path_records = load_path_records(stage1_dir)
    summaries = {
        query_id: query_summary(query_id, query_qrels, fragments, path_records)
        for query_id, query_qrels in qrels_by_query.items()
    }
    query_ids = sorted(qrels_by_query, key=lambda query_id: (summaries[query_id]["split"], summaries[query_id]["query_role"], query_id))
    return {
        "qrels": qrels,
        "qrels_by_query": qrels_by_query,
        "fragments": fragments,
        "query_ids": query_ids,
        "summaries": summaries,
        "vectors": vectors,
        "model": model,
        "projected": projected,
        "indexes": indexes,
        "ids_by_type": ids_by_type,
        "assets": assets,
        "path_records": path_records,
        "device": device,
    }


def query_summary(
    query_id: str,
    qrels: list[dict[str, Any]],
    fragments: dict[str, dict[str, Any]],
    path_records: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, str]:
    first = qrels[0]
    query_fragment = fragments.get(query_id, {})
    parts = [
        query_id,
        clean_text(first.get("query_role")),
        clean_text(first.get("split")),
        clean_text(first.get("chain_id")),
        clean_text(query_fragment.get("statement")),
        clean_text(query_fragment.get("page_title")),
    ]
    for target_id in {qrel["target_id"] for qrel in qrels}:
        target = fragments.get(target_id, {})
        parts.extend([target_id, clean_text(target.get("statement")), clean_text(target.get("page_title"))])
    for (qid, asset_id, target_id), path in path_records.items():
        if qid == query_id:
            parts.extend(
                [
                    asset_id,
                    target_id,
                    clean_text(path.get("path_id")),
                    clean_text(path.get("claim_text")),
                    clean_text(path.get("entity_text")),
                    clean_text(path.get("bridge_col_name")),
                    clean_text(path.get("bridge_value")),
                ]
            )
    return {
        "query_id": query_id,
        "query_role": clean_text(first.get("query_role")),
        "split": clean_text(first.get("split")),
        "_search": " ".join(clean_text(part).casefold() for part in parts),
    }


def direct_query_ranking(context: dict[str, Any], query_id: str, topn: int) -> list[dict[str, Any]]:
    projected = context["projected"]
    indexes = context["indexes"]
    ids_by_type = context["ids_by_type"]
    if query_id not in projected or "table_fragment" not in indexes:
        return []
    table_ids = ids_by_type.get("table_fragment", [])
    if not table_ids:
        return []
    source_type, projected_vec = projected[query_id]
    query = relation_query_from_projected(context["model"], projected_vec, source_type, "table_fragment", context["device"])
    k = min(max(1, topn), len(table_ids))
    labels, distances = indexes["table_fragment"].knn_query(query, k=k)
    return [
        {
            "target_id": table_ids[int(label)],
            "score": float(1.0 - distance),
            "path": [(query_id, source_type), (table_ids[int(label)], "table_fragment")],
        }
        for label, distance in zip(labels[0], distances[0])
    ]


def path_query_ranking(args: argparse.Namespace, context: dict[str, Any], query_id: str) -> list[dict[str, Any]]:
    if args.skip_path or "table_fragment" not in context["indexes"]:
        return []
    ranked, best_paths = beam_search_tables(
        args,
        context["model"],
        context["vectors"],
        context["projected"],
        context["indexes"],
        context["ids_by_type"],
        query_id,
        context["device"],
    )
    items = []
    for target_id in ranked[: args.topn]:
        payload = best_paths.get(target_id, {})
        items.append({"target_id": target_id, "score": float(payload.get("score", 0.0)), "path": payload.get("path", [])})
    return items


def build_query_card(args: argparse.Namespace, context: dict[str, Any], query_id: str) -> dict[str, Any]:
    if context.get("recorded"):
        return context["cards_by_query"].get(query_id, {})
    qrels = context["qrels_by_query"].get(query_id, [])
    direct_by_query = {query_id: direct_query_ranking(context, query_id, args.topn)}
    path_by_query = {query_id: path_query_ranking(args, context, query_id)}
    cards = assemble_query_cards(
        qrels,
        context["fragments"],
        direct_by_query,
        path_by_query,
        context["path_records"],
        context["assets"],
        args.max_rows,
    )
    return cards[0] if cards else {}


def load_recall_cards(args: argparse.Namespace) -> list[dict[str, Any]]:
    stage1_dir = Path(args.stage1_dir)
    qrels = load_qrels(Path(args.qrels))
    fragments = load_fragments(stage1_dir)
    query_ids = sorted({qrel["query_id"] for qrel in qrels})
    if args.query_limit and args.query_limit > 0:
        query_ids = query_ids[: args.query_limit]
        qrels = [qrel for qrel in qrels if qrel["query_id"] in set(query_ids)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors, _, _, _ = load_embeddings(Path(args.embedding_dir))
    model = load_student(Path(args.student_dir), device)
    projected = load_projected(Path(args.student_dir))
    direct_by_query = direct_hnsw_rankings(model, projected, Path(args.hnsw_dir), query_ids, args.topn, device)
    path_by_query = {} if args.skip_path else path_hnsw_rankings(args, model, vectors, projected, query_ids, device)
    assets, _ = load_assets(stage1_dir)
    path_records = load_path_records(stage1_dir)
    return assemble_query_cards(qrels, fragments, direct_by_query, path_by_query, path_records, assets, args.max_rows)


def format_score(score: Any) -> str:
    if score is None:
        return ""
    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return ""


def render_targets(targets: list[dict[str, Any]], mode: str) -> str:
    if not targets:
        return '<div class="muted">No recalled targets available.</div>'
    chunks = ['<div class="target-list">']
    for item in targets:
        hit_class = "hit" if item.get("is_relevant") else "miss"
        hit_text = f"hit rel={item.get('rel')}" if item.get("is_relevant") else "not in qrels"
        score = format_score(item.get("score"))
        chunks.append('<div class="target">')
        chunks.append('<div class="target-head">')
        chunks.append(
            '<div class="target-title">'
            f"<strong>#{item.get('rank')} {escape_html(item.get('title'))}</strong>"
            f"<div class=\"muted\">{escape_html(item.get('target_id'))}</div>"
            f"<div>{escape_html(item.get('statement'))}</div>"
            "</div>"
        )
        chunks.append('<div class="row">')
        if score:
            chunks.append(f'<span class="badge">score={escape_html(score)}</span>')
        chunks.append(f'<span class="badge {hit_class}">{escape_html(hit_text)}</span>')
        if item.get("has_bridge"):
            chunks.append('<span class="badge bridge">bridge path</span>')
        chunks.append("</div></div>")
        target_fragment = item.get("target_fragment") or {}
        if target_fragment.get("columns"):
            chunks.append(f'<div class="target-table">{render_table(target_fragment)}</div>')
        if mode == "path" and item.get("path_nodes"):
            chunks.append('<div class="path">')
            for node in item["path_nodes"]:
                chunks.append('<div class="node">')
                chunks.append(f'<span class="badge">{escape_html(node.get("type_label"))}</span>')
                chunks.append(f'<div class="node-title">{escape_html(node.get("title"))}</div>')
                chunks.append(f'<div class="muted">{escape_html(node.get("id"))}</div>')
                if node.get("bridge"):
                    chunks.append(f'<div><span class="label">Bridge</span>{escape_html(node.get("bridge"))}</div>')
                if node.get("path_id"):
                    chunks.append(f'<div><span class="label">Path ID</span>{escape_html(node.get("path_id"))}</div>')
                if node.get("claim_text"):
                    chunks.append(f'<div><span class="label">Claim</span>{escape_html(node.get("claim_text"))}</div>')
                if node.get("content"):
                    chunks.append('<span class="label">Text snippet</span>')
                    chunks.append(f'<div class="asset-text">{escape_html(node.get("content"))}</div>')
                if node.get("file_name") and not node.get("content"):
                    chunks.append(f'<div><span class="label">Image/file</span>{escape_html(node.get("file_name"))}</div>')
                if node.get("url"):
                    chunks.append(f'<div><a href="{escape_html(node.get("url"))}" target="_blank" rel="noreferrer">{escape_html(node.get("url"))}</a></div>')
                if node.get("asset_id"):
                    chunks.append(
                        f'<img class="asset-image" src="/asset/{escape_html(node.get("asset_id"))}" '
                        f'alt="{escape_html(node.get("title"))}">'
                    )
                chunks.append("</div>")
            chunks.append("</div>")
        chunks.append("</div>")
    chunks.append("</div>")
    return "".join(chunks)


def create_app(args: argparse.Namespace) -> Flask:
    app = Flask(__name__)
    stage1_dir = Path(args.stage1_dir)

    @lru_cache(maxsize=1)
    def cached_context() -> dict[str, Any]:
        return load_runtime_context(args)

    @lru_cache(maxsize=512)
    def cached_card(query_id: str) -> dict[str, Any]:
        return build_query_card(args, cached_context(), query_id)

    @lru_cache(maxsize=1)
    def cached_assets() -> tuple[dict[str, dict[str, Any]], Path]:
        return load_assets(stage1_dir)

    @app.get("/")
    def index() -> str:
        error = ""
        try:
            context = cached_context()
            query_ids = list(context["query_ids"])
            summaries = context["summaries"]
        except Exception as exc:  # pragma: no cover - visible in browser for local debugging.
            context = {}
            query_ids = []
            summaries = {}
            error = str(exc)
        query = clean_text(request.args.get("q", ""))
        split = clean_text(request.args.get("split", ""))
        role = clean_text(request.args.get("role", ""))
        try:
            page = max(1, int(request.args.get("page", "1") or "1"))
        except ValueError:
            page = 1
        splits = sorted({summary["split"] for summary in summaries.values() if summary.get("split")})
        roles = sorted({summary["query_role"] for summary in summaries.values() if summary.get("query_role")})
        if split:
            query_ids = [query_id for query_id in query_ids if summaries[query_id].get("split") == split]
        if role:
            query_ids = [query_id for query_id in query_ids if summaries[query_id].get("query_role") == role]
        if query:
            needle = query.casefold()
            query_ids = [query_id for query_id in query_ids if needle in summaries[query_id]["_search"]]
        total = len(query_ids)
        pages = max(1, math.ceil(total))
        page = min(page, pages)
        card = None
        if query_ids and not error:
            try:
                card = cached_card(query_ids[page - 1])
            except Exception as exc:  # pragma: no cover - visible in browser for local debugging.
                error = str(exc)
        missing_records = clean_text(context.get("missing_records", "")) if context else ""

        def page_url(target: int) -> str:
            target = min(max(1, target), pages)
            return url_for("index", page=target, q=query, split=split, role=role)

        return render_template_string(
            PAGE_TEMPLATE,
            card=card,
            total=total,
            page=page,
            pages=pages,
            query=query,
            split=split,
            role=role,
            splits=splits,
            roles=roles,
            error=error,
            missing_records=missing_records,
            page_url=page_url,
            render_table=render_table,
            render_targets=render_targets,
        )

    @app.get("/reload")
    def reload() -> Any:
        cached_context.cache_clear()
        cached_card.cache_clear()
        cached_assets.cache_clear()
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
    parser.add_argument("--student_dir", default="output_stage1_logic/student")
    parser.add_argument("--embedding_dir", default="output_stage1_logic/embeddings")
    parser.add_argument("--hnsw_dir", default="output_stage1_logic/hnsw_indices")
    parser.add_argument("--qrels", default="output_stage1_logic/qrels.jsonl")
    parser.add_argument("--recall_records", default=None, help="Evaluation recall_rankings.jsonl to display; defaults to stage1_dir/recall_rankings.jsonl.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--lan", action="store_true", help="Expose the viewer GUI on the LAN by binding to 0.0.0.0.")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--max_rows", type=int, default=8)
    parser.add_argument("--max_hops", type=int, default=3)
    parser.add_argument("--beam_width", type=int, default=64)
    parser.add_argument("--beam_neighbors", type=int, default=50)
    parser.add_argument("--path_composition", choices=["min", "product"], default="min")
    parser.add_argument("--query_limit", type=int, default=0, help="Debug limit for the number of queries loaded; 0 means all.")
    parser.add_argument("--dynamic", action="store_true", help="Compute recall on demand when recall_rankings.jsonl is missing.")
    parser.add_argument("--skip_path", action="store_true", help="With --dynamic, only show direct table recall; skip bridge-aware beam search.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.host = resolve_gui_host(args.host, args.lan)
    app = create_app(args)
    print(format_gui_urls("Recall viewer", args.host, args.port))
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Flask annotation UI for human-in-the-loop evidence path labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_file

from merge_human_labels import run as merge_human_labels
from stage1_io import iter_jsonl, update_stage1_manifest, write_json, write_jsonl

ALLOWED_LABELS = {
    "2": "Direct Bridge",
    "1": "Indirect Bridge",
    "0": "Related Only",
    "-1": "Wrong/Irrelevant",
}

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HITL Annotation Round {{ round_id }}</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #1b6ef3;
      --ok: #087443;
      --warn: #a15c00;
      --bad: #aa1e1e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 24px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main { width: min(1440px, 100%); margin: 0 auto; padding: 18px 24px 40px; }
    button, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .progress {
      min-width: 180px;
      color: var(--muted);
      text-align: right;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
    }
    .status.ok { color: var(--ok); }
    .status.bad { color: var(--bad); }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
    }
    article {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    article.done {
      border-color: #9fd5b7;
    }
    .meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }
    .claim {
      font-size: 16px;
      font-weight: 650;
      margin: 0 0 10px;
    }
    .evidence {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      min-height: 74px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    img.evidence-image {
      display: block;
      max-width: 100%;
      max-height: 260px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      margin-top: 8px;
    }
    .tables {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin: 12px 0;
    }
    .table-box {
      min-width: 0;
    }
    .table-title {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 5px 6px;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { background: #eef2f7; text-align: left; }
    .controls {
      display: grid;
      grid-template-columns: 210px 1fr auto;
      gap: 10px;
      align-items: start;
      margin-top: 12px;
    }
    select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
    }
    select { padding: 8px; }
    textarea {
      min-height: 38px;
      resize: vertical;
      padding: 8px;
    }
    @media (max-width: 780px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      .tables { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
      .progress { text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>HITL Annotation Round {{ round_id }}</h1>
      <div class="status" id="status">{{ progress.done }} / {{ progress.total }} labeled</div>
    </div>
    <div class="toolbar">
      <div class="progress" id="progress">{{ progress.done }} / {{ progress.total }}</div>
      <button type="button" id="merge" class="primary">Merge Round</button>
    </div>
  </header>
  <main>
    <section class="grid">
      {% for item in items %}
      <article data-path-id="{{ item.path_id }}" class="{{ 'done' if item.label else '' }}">
        <div class="meta">
          <span>{{ item.path_id }}</span>
          <span>{{ item.asset_type }} | {{ item.bridge_col_name }} = {{ item.bridge_value }}</span>
        </div>
        <p class="claim">{{ item.claim_text or '' }}</p>
        <div class="evidence">{{ item.evidence_text_snippet or item.image_local_path or '' }}</div>
        {% if item.image_exists %}
        <img class="evidence-image" src="{{ url_for('asset', path_id=item.path_id) }}" alt="Evidence image">
        {% endif %}
        <div class="tables">
          {{ preview_table("Query", item.query_fragment_preview) | safe }}
          {{ preview_table("Target", item.target_fragment_preview) | safe }}
        </div>
        <div class="controls">
          <select name="label">
            <option value="">Unlabeled</option>
            {% for value, text in allowed_labels.items() %}
            <option value="{{ value }}" {{ 'selected' if item.label|string == value else '' }}>{{ value }} - {{ text }}</option>
            {% endfor %}
          </select>
          <textarea name="annotator_notes" placeholder="Notes">{{ item.annotator_notes or '' }}</textarea>
          <button type="button" class="save">Save</button>
        </div>
      </article>
      {% endfor %}
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const progressEl = document.getElementById("progress");
    const mergeBtn = document.getElementById("merge");

    function setStatus(text, cls) {
      statusEl.textContent = text;
      statusEl.className = "status" + (cls ? " " + cls : "");
    }

    function updateProgress(progress) {
      progressEl.textContent = `${progress.done} / ${progress.total}`;
      setStatus(`${progress.done} / ${progress.total} labeled`, progress.done === progress.total ? "ok" : "");
    }

    async function saveCard(card) {
      const payload = {
        path_id: card.dataset.pathId,
        label: card.querySelector("select").value,
        annotator_notes: card.querySelector("textarea").value
      };
      const resp = await fetch("/api/label", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        setStatus(data.error || "Save failed", "bad");
        return;
      }
      card.classList.toggle("done", Boolean(payload.label));
      updateProgress(data.progress);
    }

    document.querySelectorAll("article").forEach((card) => {
      card.querySelector(".save").addEventListener("click", () => saveCard(card));
      card.querySelector("select").addEventListener("change", () => saveCard(card));
    });

    mergeBtn.addEventListener("click", async () => {
      mergeBtn.disabled = true;
      const resp = await fetch("/api/merge", {method: "POST"});
      const data = await resp.json();
      if (!resp.ok) {
        setStatus(data.error || "Merge failed", "bad");
        mergeBtn.disabled = false;
        return;
      }
      updateProgress(data.progress);
      setStatus("Merged into human_labeled_paths.jsonl", "ok");
    });
  </script>
</body>
</html>
"""


def round_template_path(stage1_dir: Path, round_id: int) -> Path:
    return stage1_dir / f"human_labels_template_round_{round_id}.jsonl"


def round_filled_path(stage1_dir: Path, round_id: int) -> Path:
    return stage1_dir / f"human_labels_filled_round_{round_id}.jsonl"


def round_status_path(stage1_dir: Path, round_id: int) -> Path:
    return stage1_dir / f"hitl_round_{round_id}_annotation_status.json"


def load_round_items(stage1_dir: Path, round_id: int) -> list[dict[str, Any]]:
    template_path = round_template_path(stage1_dir, round_id)
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template file: {template_path}")
    items = [dict(rec) for rec in iter_jsonl(template_path)]
    filled_path = round_filled_path(stage1_dir, round_id)
    filled = {rec["path_id"]: rec for rec in iter_jsonl(filled_path)} if filled_path.exists() else {}
    for item in items:
        saved = filled.get(item["path_id"])
        if saved:
            item["label"] = saved.get("label", item.get("label", ""))
            item["annotator_notes"] = saved.get("annotator_notes", item.get("annotator_notes", ""))
        image_path = Path(str(item.get("image_local_path") or ""))
        item["image_exists"] = bool(item.get("image_local_path")) and image_path.exists() and image_path.is_file()
    return items


def save_round_items(stage1_dir: Path, round_id: int, items: list[dict[str, Any]]) -> int:
    return write_jsonl(round_filled_path(stage1_dir, round_id), items)


def progress(items: list[dict[str, Any]]) -> dict[str, int]:
    return {"done": sum(1 for item in items if str(item.get("label", "")) in ALLOWED_LABELS), "total": len(items)}


def preview_table(title: str, preview: dict[str, Any] | None) -> str:
    preview = preview or {}
    cols = preview.get("columns") or []
    rows = preview.get("rows") or []
    head = "".join(f"<th>{escape_html(col)}</th>" for col in cols)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape_html(row.get(col, ''))}</td>" for col in cols)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-box">'
        f'<div class="table-title">{escape_html(title)}: {escape_html(preview.get("fragment_id", ""))}</div>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
        "</div>"
    )


def escape_html(value: Any) -> str:
    return (
        str(value if value is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def create_app(stage1_dir: Path, round_id: int) -> Flask:
    app = Flask(__name__)
    stage1_dir = Path(stage1_dir)

    @app.get("/")
    def index() -> str:
        items = load_round_items(stage1_dir, round_id)
        return render_template_string(
            PAGE_TEMPLATE,
            round_id=round_id,
            items=items,
            progress=progress(items),
            allowed_labels=ALLOWED_LABELS,
            preview_table=preview_table,
        )

    @app.get("/api/status")
    def api_status() -> Any:
        items = load_round_items(stage1_dir, round_id)
        status = {"round_id": round_id, "progress": progress(items), "merged": False}
        status_path = round_status_path(stage1_dir, round_id)
        if status_path.exists():
            status.update(json.loads(status_path.read_text(encoding="utf-8")))
        return jsonify(status)

    @app.post("/api/label")
    def api_label() -> Any:
        payload = request.get_json(force=True, silent=True) or {}
        path_id = str(payload.get("path_id") or "")
        label = str(payload.get("label") or "")
        if label and label not in ALLOWED_LABELS:
            return jsonify({"error": f"Invalid label {label!r}"}), 400
        items = load_round_items(stage1_dir, round_id)
        for item in items:
            if item["path_id"] == path_id:
                item["label"] = label
                item["annotator_notes"] = str(payload.get("annotator_notes") or "")
                save_round_items(stage1_dir, round_id, items)
                return jsonify({"ok": True, "progress": progress(items)})
        return jsonify({"error": f"Unknown path_id {path_id!r}"}), 404

    @app.post("/api/merge")
    def api_merge() -> Any:
        items = load_round_items(stage1_dir, round_id)
        missing = [item["path_id"] for item in items if str(item.get("label", "")) not in ALLOWED_LABELS]
        if missing:
            return jsonify({"error": f"{len(missing)} items are still unlabeled", "missing": missing[:10]}), 400
        filled_path = round_filled_path(stage1_dir, round_id)
        save_round_items(stage1_dir, round_id, items)
        merge_human_labels(argparse.Namespace(stage1_dir=str(stage1_dir), human_labels=str(filled_path)))
        status = {"round_id": round_id, "merged": True, "filled_path": str(filled_path), "progress": progress(items)}
        write_json(round_status_path(stage1_dir, round_id), status)
        update_stage1_manifest(stage1_dir, f"hitl_round_{round_id}_annotations", status)
        return jsonify({"ok": True, **status})

    @app.get("/asset/<path_id>")
    def asset(path_id: str) -> Any:
        items = load_round_items(stage1_dir, round_id)
        item = next((candidate for candidate in items if candidate["path_id"] == path_id), None)
        if not item:
            return jsonify({"error": "Unknown path_id"}), 404
        image_path = Path(str(item.get("image_local_path") or ""))
        if not image_path.exists() or not image_path.is_file():
            return jsonify({"error": "Image file is not available on this machine"}), 404
        return send_file(image_path)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--round_id", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.stage1_dir), args.round_id)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

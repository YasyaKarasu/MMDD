#!/usr/bin/env python
"""Select an active-learning batch for human labels."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from stage1_io import clean_text, iter_jsonl, setup_logging, update_stage1_manifest, write_jsonl


ROUND_RE = re.compile(r"hitl_selected_round_(\d+)\.jsonl$")


def load_teacher_scores(path: Path | None) -> dict[str, float]:
    if not path or not path.exists():
        return {}
    scores = {}
    for rec in iter_jsonl(path):
        if rec.get("path_id") and rec.get("path_score") is not None:
            scores[rec["path_id"]] = float(rec["path_score"])
    return scores


def weak_default_score(path: dict[str, Any]) -> float:
    if path.get("weak_score") is not None:
        return float(path["weak_score"])
    if path.get("asset_type") == "text":
        return 0.45
    return 0.35


def row_match_score(row: dict[str, Any], values: list[str]) -> int:
    candidates = [clean_text(value).casefold() for value in values if clean_text(value)]
    if not candidates:
        return 0
    score = 0
    for cell in row.get("cells", []):
        text = clean_text(cell.get("text")).casefold()
        if not text:
            continue
        for value in candidates:
            if text == value:
                score += 2
            elif len(value) >= 3 and value in text:
                score += 1
    return score


def row_id_matches(row: dict[str, Any], focus_row_id: int | None) -> bool:
    if focus_row_id is None:
        return False
    for key in ("source_row_id", "row_id"):
        try:
            if int(row.get(key)) == int(focus_row_id):
                return True
        except (TypeError, ValueError):
            continue
    return False


def choose_focus_row(rows: list[dict[str, Any]], values: list[str], focus_row_id: int | None = None) -> dict[str, Any] | None:
    for row in rows:
        if row_id_matches(row, focus_row_id):
            return row
    scored = [(row_match_score(row, values), -idx, row) for idx, row in enumerate(rows)]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return scored[0][2]


def preview_table_like(
    table: dict[str, Any],
    focus_values: list[str] | None = None,
    max_rows: int | None = 5,
    focus_row_id: int | None = None,
) -> dict[str, Any]:
    cols = [col.get("column_name") for col in table.get("columns", [])]
    table_rows = list(table.get("rows", []))
    focus_row = choose_focus_row(table_rows, focus_values or [], focus_row_id)
    if max_rows is None:
        selected_rows = table_rows
    else:
        selected_rows = list(table_rows[:max_rows])
        if focus_row is not None and focus_row not in selected_rows:
            if len(selected_rows) >= max_rows:
                selected_rows = selected_rows[: max(0, max_rows - 1)]
            selected_rows.append(focus_row)
    rows = []
    focus_id = id(focus_row) if focus_row is not None else None
    for row in selected_rows:
        preview_row = {}
        for i, cell in enumerate(row.get("cells", [])):
            if i >= len(cols):
                continue
            preview_row[cols[i]] = cell.get("text")
        if id(row) == focus_id:
            preview_row["_focus"] = True
        rows.append(preview_row)
    return {
        "columns": cols,
        "rows": rows,
    }


def preview_fragment(
    fragment: dict[str, Any],
    focus_values: list[str] | None = None,
    max_rows: int = 5,
    focus_row_id: int | None = None,
) -> dict[str, Any]:
    return preview_table_like(fragment, focus_values, max_rows, focus_row_id)


def load_excluded_path_ids(stage1_dir: Path, round_id: int, allow_reselect_previous: bool, allow_reselect_labeled: bool) -> set[str]:
    excluded: set[str] = set()
    if not allow_reselect_labeled:
        human_path = stage1_dir / "human_labeled_paths.jsonl"
        if human_path.exists():
            excluded.update(str(rec["path_id"]) for rec in iter_jsonl(human_path) if rec.get("path_id"))
    if not allow_reselect_previous:
        for selected_path in stage1_dir.glob("hitl_selected_round_*.jsonl"):
            match = ROUND_RE.search(selected_path.name)
            if not match or int(match.group(1)) == round_id:
                continue
            excluded.update(str(rec["path_id"]) for rec in iter_jsonl(selected_path) if rec.get("path_id"))
    return excluded


def load_assets_for_selected(stage1_dir: Path, selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    input_dir = Path(manifest.get("evidence_paths", {}).get("input_dir", "output_medium"))
    from stage1_io import iter_manifest_records

    needed = {item["asset_id"] for item in selected}
    assets = {}
    for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000):
        if asset.get("asset_id") in needed:
            assets[asset["asset_id"]] = asset
            if len(assets) == len(needed):
                break
    return assets


def load_source_tables_for_selected(stage1_dir: Path, selected: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    input_dir = Path(manifest.get("evidence_paths", {}).get("input_dir", "output_medium"))
    from stage1_io import iter_manifest_records

    needed = {item.get("source_table_id") for item in selected if item.get("source_table_id")}
    tables = {}
    if not needed:
        return tables
    try:
        source_iter = iter_manifest_records(input_dir, "source_tables", log_every=1000)
        for table in source_iter:
            if table.get("source_table_id") in needed:
                tables[table["source_table_id"]] = table
                if len(tables) == len(needed):
                    break
    except (KeyError, FileNotFoundError):
        return tables
    return tables


def run(args: argparse.Namespace) -> None:
    setup_logging()
    rng = random.Random(args.seed)
    stage1_dir = Path(args.stage1_dir)
    pool_path = stage1_dir / "hitl_pool.jsonl"
    excluded = load_excluded_path_ids(
        stage1_dir,
        args.round_id,
        bool(getattr(args, "allow_reselect_previous", False)),
        bool(getattr(args, "allow_reselect_labeled", False)),
    )
    pool = [p for p in iter_jsonl(pool_path) if p.get("human_label") is None and str(p.get("path_id")) not in excluded]
    teacher_scores = load_teacher_scores(Path(args.teacher_scores) if args.teacher_scores else None)
    for path in pool:
        score = teacher_scores.get(path["path_id"], weak_default_score(path))
        path["_path_score"] = score
        path["_uncertainty"] = 1.0 - abs(score - 0.5) * 2.0
    pool.sort(key=lambda item: (item["_uncertainty"], item.get("weak_label") is None), reverse=True)
    candidates = pool[: args.candidate_top_n]

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    fragments = {rec["fragment_id"]: rec for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")}
    for item in candidates:
        key = (
            clean_text(item.get("bridge_col_name")),
            clean_text(item.get("asset_type")),
            clean_text(item.get("split")),
        )
        strata[key].append(item)
    selected: list[dict[str, Any]] = []
    keys = list(strata)
    rng.shuffle(keys)
    while keys and len(selected) < args.batch_size:
        next_keys = []
        for key in keys:
            bucket = strata[key]
            if bucket and len(selected) < args.batch_size:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys

    assets = load_assets_for_selected(stage1_dir, selected)
    source_tables = load_source_tables_for_selected(stage1_dir, selected)
    templates = []
    for item in selected:
        asset = assets.get(item["asset_id"], {})
        query = fragments.get(item["query_fragment_id"], {})
        target = fragments.get(item["target_fragment_id"], {})
        source_table = source_tables.get(item.get("source_table_id"), {})
        focus_values = [item.get("entity_text", ""), item.get("bridge_value", "")]
        snippet = ""
        image_path = ""
        if item.get("asset_type") == "text":
            snippet = clean_text(asset.get("content"))[:1200]
        else:
            image_path = clean_text(asset.get("local_path") or asset.get("relative_path"))
        templates.append(
            {
                "path_id": item["path_id"],
                "query_fragment_preview": preview_fragment(query, focus_values, focus_row_id=item.get("source_row_id")),
                "target_fragment_preview": preview_fragment(target, [item.get("bridge_value", ""), item.get("entity_text", "")]),
                "source_table_preview": preview_table_like(source_table, focus_values, max_rows=None, focus_row_id=item.get("source_row_id")) if source_table else None,
                "claim_text": item.get("claim_text"),
                "asset_type": item.get("asset_type"),
                "evidence_text_snippet": snippet,
                "image_local_path": image_path,
                "bridge_col_name": item.get("bridge_col_name"),
                "bridge_value": item.get("bridge_value"),
                "label": "",
                "allowed_labels": {
                    "2": "Direct Bridge",
                    "1": "Indirect Bridge",
                    "0": "Related Only",
                    "-1": "Wrong/Irrelevant",
                },
                "annotator_notes": "",
            }
        )
    selected_path = stage1_dir / f"hitl_selected_round_{args.round_id}.jsonl"
    template_path = stage1_dir / f"human_labels_template_round_{args.round_id}.jsonl"
    counts = {"selected": write_jsonl(selected_path, selected), "template": write_jsonl(template_path, templates)}
    update_stage1_manifest(
        stage1_dir,
        f"hitl_round_{args.round_id}",
        {"counts": counts, "excluded": len(excluded), "args": vars(args)},
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--teacher_scores", default=None)
    parser.add_argument("--round_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--candidate_top_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow_reselect_previous", action="store_true")
    parser.add_argument("--allow_reselect_labeled", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

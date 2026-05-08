#!/usr/bin/env python
"""Generate weak labels for candidate evidence paths."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from stage1_io import clean_text, iter_jsonl, iter_manifest_records, setup_logging, stable_hash, update_stage1_manifest, write_jsonl


def text_contains(haystack: str, needle: str) -> bool:
    hay = clean_text(haystack).casefold()
    nee = clean_text(needle).casefold()
    return bool(nee) and nee in hay


def image_metadata_text(asset: dict[str, Any]) -> str:
    metadata = asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {}
    ext = metadata.get("extmetadata") if isinstance(metadata.get("extmetadata"), dict) else {}
    parts = [asset.get("file_name"), asset.get("entity_wiki_title")]
    for value in ext.values():
        if isinstance(value, dict):
            parts.append(value.get("value"))
        else:
            parts.append(value)
    text = " ".join(clean_text(part) for part in parts if part)
    return re.sub(r"<[^>]+>", " ", text)


def load_assets_from_manifest(stage1_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads((stage1_dir / "manifest.json").read_text(encoding="utf-8"))
    input_dir = Path(manifest.get("evidence_paths", {}).get("input_dir", "output_medium"))
    return {asset["asset_id"]: asset for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000)}


def label_path(path: dict[str, Any], asset: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(path)
    if not asset:
        return out
    bridge_value = clean_text(path.get("bridge_value"))
    entity_text = clean_text(path.get("entity_text"))
    bridge_col = clean_text(path.get("bridge_col_name"))
    if asset.get("asset_type") == "text":
        content = clean_text(asset.get("content"))
        has_entity = text_contains(content, entity_text) or text_contains(content, asset.get("entity_wiki_title"))
        has_bridge = text_contains(content, bridge_value)
        has_col = text_contains(content, bridge_col)
        if has_bridge and has_entity and has_col:
            out.update({"weak_label": "weak_direct", "weak_score": 0.9})
        elif has_bridge and has_entity:
            out.update({"weak_label": "weak_direct", "weak_score": 0.8})
        elif has_bridge:
            out.update({"weak_label": "weak_indirect", "weak_score": 0.5})
    elif asset.get("asset_type") == "image":
        meta_text = image_metadata_text(asset)
        if text_contains(meta_text, bridge_value):
            out.update({"weak_label": "weak_indirect", "weak_score": 0.5})
    return out


def make_corrupted_negatives(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if path.get("weak_label") == "weak_direct":
            by_entity[str(path.get("entity_id"))].append(path)
    negatives: list[dict[str, Any]] = []
    for entity_id, items in by_entity.items():
        bridge_values = sorted({clean_text(item.get("bridge_value")) for item in items if item.get("bridge_value")})
        if len(bridge_values) < 2:
            continue
        for item in items[:2]:
            wrong = next((value for value in bridge_values if value != item.get("bridge_value")), "")
            if not wrong:
                continue
            neg = dict(item)
            neg["path_id"] = f"path_neg_{stable_hash(item['path_id'], wrong)}"
            neg["bridge_value"] = wrong
            neg["claim_text"] = f"{item.get('entity_text')} -- {item.get('bridge_col_name')} --> {wrong}"
            neg["weak_label"] = "weak_negative"
            neg["weak_score"] = 0.0
            neg["reason"] = "weak_negative:corrupted_bridge_value"
            neg["original_path_id"] = item["path_id"]
            negatives.append(neg)
    return negatives


def run(args: argparse.Namespace) -> None:
    setup_logging()
    stage1_dir = Path(args.stage1_dir)
    assets = load_assets_from_manifest(stage1_dir)
    labeled = [label_path(path, assets.get(path.get("asset_id"))) for path in iter_jsonl(stage1_dir / "evidence_paths.jsonl")]
    labeled.extend(make_corrupted_negatives(labeled))
    counts = {
        "weak_labeled_paths": write_jsonl(stage1_dir / "weak_labeled_paths.jsonl", labeled),
        "hitl_pool": write_jsonl(stage1_dir / "hitl_pool.jsonl", labeled),
    }
    stats = defaultdict(int)
    for item in labeled:
        stats[str(item.get("weak_label") or "unlabeled")] += 1
    update_stage1_manifest(stage1_dir, "weak_labels", {"counts": counts, "label_stats": dict(stats), "args": vars(args)})
    print(json.dumps({"counts": counts, "label_stats": dict(stats)}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

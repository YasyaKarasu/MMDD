#!/usr/bin/env python
"""Build Q_hidden -> asset -> target paths from intrinsic-content recoveries."""

from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path
from typing import Any

from stage1_io import (
    clean_text,
    get_cell_text,
    get_column_name,
    iter_jsonl,
    iter_manifest_records,
    setup_logging,
    stable_hash,
    update_stage1_manifest,
    write_jsonl,
)


def load_fragments(stage1_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    fragments = {rec["fragment_id"]: rec for rec in iter_jsonl(stage1_dir / "logic_fragments.jsonl")}
    target_by_chain = {rec["chain_id"]: rec for rec in fragments.values() if rec.get("role") == "right_target"}
    return fragments, target_by_chain


def load_source_tables(input_dir: Path, needed: set[str]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for table in iter_manifest_records(input_dir, "source_tables", log_every=1000):
        if table.get("source_table_id") in needed:
            tables[table["source_table_id"]] = table
            if len(tables) >= len(needed):
                break
    return tables


def load_assets(input_dir: Path) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for asset in iter_manifest_records(input_dir, "bridge_assets", log_every=50000):
        assets[asset["asset_id"]] = asset
    return assets


def target_bridge_values(target: dict[str, Any]) -> set[str]:
    values = set()
    for row in target.get("rows", []):
        value = clean_text(get_cell_text(row, 0))
        if value:
            values.add(value)
    return values


def run(args: argparse.Namespace) -> None:
    setup_logging()
    input_dir = Path(args.input_dir)
    stage1_dir = Path(args.stage1_dir)
    fragments, target_by_chain = load_fragments(stage1_dir)
    hidden = [rec for rec in fragments.values() if rec.get("role") == "left_hidden"]
    needed_tables = {rec["source_table_id"] for rec in hidden}
    source_tables = load_source_tables(input_dir, needed_tables)
    assets = load_assets(input_dir)

    hidden_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for frag in hidden:
        table = source_tables.get(frag["source_table_id"])
        target = target_by_chain.get(frag["chain_id"])
        if not table or not target:
            continue
        b_col = int(frag["hidden_bridge_col"])
        valid_b = target_bridge_values(target)
        for row_idx in frag.get("source_row_indices", []):
            try:
                source_row = table["rows"][int(row_idx)]
            except (IndexError, TypeError, ValueError):
                continue
            bridge_value = clean_text(get_cell_text(source_row, b_col))
            if not bridge_value or bridge_value not in valid_b:
                continue
            key = (frag["source_table_id"], int(row_idx), b_col)
            hidden_by_key.setdefault(key, []).append(frag)

    paths: list[dict[str, Any]] = []
    seen_path_ids: set[str] = set()
    try:
        recovery_records = iter_manifest_records(
            input_dir,
            "evidence_recoveries",
            log_every=50000,
        )
        recoveries = iter(recovery_records)
        first_recovery = next(recoveries, None)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise ValueError(
            "Stage-1 multimodal paths require evidence_recoveries produced by "
            "intrinsic-content extraction; table_asset_links are intentionally not used"
        ) from error
    recovery_stream = (
        [] if first_recovery is None else [first_recovery]
    )
    for recovery in chain(recovery_stream, recoveries):
        recovered_attribute = recovery.get("recovered_attribute")
        evidence = recovery.get("evidence")
        if not isinstance(recovered_attribute, dict) or not isinstance(evidence, dict):
            continue
        try:
            key = (
                recovery["source_table_id"],
                int(recovery["source_row_id"]),
                int(recovered_attribute["column_index"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if key not in hidden_by_key:
            continue
        asset_id = clean_text(evidence.get("asset_id"))
        asset = assets.get(asset_id)
        if not asset:
            continue
        for frag in hidden_by_key.get(key, []):
            table = source_tables[frag["source_table_id"]]
            target = target_by_chain[frag["chain_id"]]
            a_col = int(frag["source_column_indices"][0])
            b_col = int(frag["hidden_bridge_col"])
            source_row = table["rows"][key[1]]
            entity_text = clean_text(get_cell_text(source_row, a_col))
            bridge_value = clean_text(get_cell_text(source_row, b_col))
            asset_type = asset.get("asset_type")
            if asset_type not in {"text", "image"}:
                continue
            path_id = f"path_{stable_hash(frag['fragment_id'], target['fragment_id'], asset_id, key[1])}"
            if path_id in seen_path_ids:
                continue
            seen_path_ids.add(path_id)
            paths.append(
                {
                    "path_id": path_id,
                    "split": frag.get("split"),
                    "chain_id": frag["chain_id"],
                    "source_table_id": frag["source_table_id"],
                    "source_row_id": key[1],
                    "query_fragment_id": frag["fragment_id"],
                    "target_fragment_id": target["fragment_id"],
                    "asset_id": asset_id,
                    "asset_type": asset_type,
                    "entity_text": entity_text,
                    "bridge_col_index": b_col,
                    "bridge_col_name": get_column_name(table, b_col),
                    "bridge_value": bridge_value,
                    "target_bridge_col_name": target.get("target_bridge_col_name") or get_column_name(table, b_col),
                    "claim_text": f"{entity_text} -- {get_column_name(table, b_col)} --> {bridge_value}",
                    "weak_label": None,
                    "weak_score": None,
                    "human_label": None,
                    "teacher_score": None,
                    "reason": "candidate_path:intrinsic_recovery:Q_hidden->asset->T",
                }
            )

    counts = {
        "evidence_paths": write_jsonl(stage1_dir / "evidence_paths.jsonl", paths),
        "hitl_pool": write_jsonl(stage1_dir / "hitl_pool.jsonl", paths),
    }
    update_stage1_manifest(
        stage1_dir,
        "evidence_paths",
        {
            "input_dir": args.input_dir,
            "path_source": "evidence_recoveries",
            "counts": counts,
            "args": vars(args),
        },
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output_medium")
    parser.add_argument("--stage1_dir", default="output_stage1_logic")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

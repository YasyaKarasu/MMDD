#!/usr/bin/env python
"""Build A->B->C logic fragments and qrels for stage-1 coarse recall."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from stage1_io import (
    clean_text,
    column_profiles,
    dominant_by_key,
    fd_purity,
    get_cell_text,
    get_column_name,
    is_useless_column_name,
    iter_manifest_records,
    load_split_map,
    make_columns,
    project_rows,
    setup_logging,
    stable_hash,
    update_stage1_manifest,
    write_jsonl,
)


def valid_bridge_profile(profile: dict[str, Any], max_bridge_unique_ratio: float) -> bool:
    return (
        float(profile.get("non_empty_ratio", 0.0)) >= 0.6
        and float(profile.get("numeric_ratio", 0.0)) <= 0.8
        and float(profile.get("unique_ratio", 1.0)) <= max_bridge_unique_ratio
    )


def valid_target_profile(table: dict[str, Any], idx: int, profile: dict[str, Any]) -> bool:
    distinct = len({clean_text(get_cell_text(row, idx)) for row in table.get("rows", []) if get_cell_text(row, idx)})
    return float(profile.get("non_empty_ratio", 0.0)) >= 0.6 and distinct >= 2


def choose_query_context_cols(table: dict[str, Any], exclude: set[int], max_cols: int) -> list[int]:
    if max_cols <= 0:
        return []
    profiles = column_profiles(table)
    candidates = []
    for idx, profile in profiles.items():
        if idx in exclude or is_useless_column_name(get_column_name(table, idx)):
            continue
        non_empty_ratio = float(profile.get("non_empty_ratio", 0.0))
        if non_empty_ratio >= 0.6:
            numeric_ratio = float(profile.get("numeric_ratio", 0.0))
            unique_ratio = float(profile.get("unique_ratio", 1.0))
            candidates.append((numeric_ratio <= 0.5, non_empty_ratio, -unique_ratio, -idx, idx))
    candidates.sort(reverse=True)
    return [candidate[-1] for candidate in candidates[:max_cols]]


def build_target_rows(source_table: dict[str, Any], b_col: int, c_cols: list[int]) -> tuple[list[dict[str, Any]], list[int]]:
    b_to_rows: dict[str, list[dict[str, Any]]] = {}
    b_to_source_rows: dict[str, list[int]] = {}
    for fallback, row in enumerate(source_table.get("rows", [])):
        b_val = clean_text(get_cell_text(row, b_col))
        if not b_val:
            continue
        if any(not clean_text(get_cell_text(row, c_col)) for c_col in c_cols):
            continue
        b_to_rows.setdefault(b_val, []).append(row)
        b_to_source_rows.setdefault(b_val, []).append(int(row.get("row_id", fallback)))
    dominant = {c_col: dominant_by_key(source_table.get("rows", []), b_col, c_col) for c_col in c_cols}
    rows = []
    source_row_indices: list[int] = []
    for b_val in sorted(b_to_rows):
        cells = [
            {
                "column_index": 0,
                "source_column_index": b_col,
                "column_name": get_column_name(source_table, b_col),
                "text": b_val,
            }
        ]
        complete = True
        for out_idx, c_col in enumerate(c_cols, 1):
            value = dominant[c_col].get(b_val, "")
            if not value:
                complete = False
                break
            cells.append(
                {
                    "column_index": out_idx,
                    "source_column_index": c_col,
                    "column_name": get_column_name(source_table, c_col),
                    "text": value,
                }
            )
        if complete:
            rows.append({"row_id": len(rows), "source_row_id": b_to_source_rows[b_val][0], "cells": cells})
            source_row_indices.extend(b_to_source_rows[b_val])
    return rows, sorted(set(source_row_indices))


def make_fragment(
    source_table: dict[str, Any],
    split: str,
    chain_id: str,
    role: str,
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_indices: list[int],
    statement: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    fragment_id = f"frag_{stable_hash(chain_id, role)}"
    return {
        "fragment_id": fragment_id,
        "object_id": fragment_id,
        "object_type": "table_fragment",
        "role": role,
        "split": split,
        "chain_id": chain_id,
        "source_table_id": source_table["source_table_id"],
        "page_title": clean_text(source_table.get("page_title")),
        "caption": clean_text(source_table.get("caption")),
        "section_title": clean_text(source_table.get("section_title")),
        "columns": make_columns(source_table, column_indices),
        "rows": rows,
        "source_column_indices": column_indices,
        "source_row_indices": source_row_indices,
        "statement": statement,
        "provenance": {
            "builder": "build_stage1_logic_connectivity.py",
            "source_file": source_table.get("source_file"),
        },
        **extra,
    }


def query_bridge_columns(fragment: dict[str, Any]) -> set[int]:
    role = fragment.get("role")
    if role == "left_hidden" and fragment.get("hidden_bridge_col") is not None:
        try:
            return {int(fragment["hidden_bridge_col"])}
        except (TypeError, ValueError):
            return set()
    if role == "left_visible":
        bridge_col = fragment.get("visible_bridge_col")
        if bridge_col is not None:
            try:
                return {int(bridge_col)}
            except (TypeError, ValueError):
                return set()
        source_cols = fragment.get("source_column_indices") or []
        if len(source_cols) >= 2 and fragment.get("visible_bridge"):
            try:
                return {int(source_cols[1])}
            except (TypeError, ValueError):
                return set()
    return set()


def source_column_set(fragment: dict[str, Any]) -> set[int]:
    cols: set[int] = set()
    for col in fragment.get("source_column_indices", []) or []:
        try:
            cols.add(int(col))
        except (TypeError, ValueError):
            continue
    return cols


def add_same_source_bridge_positive_pairs(
    fragments: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
) -> int:
    existing_pairs = {
        (pair.get("query_fragment_id"), pair.get("target_fragment_id"))
        for pair in pairs
    }
    existing_qrels = {
        (qrel.get("query_id"), qrel.get("target_id"))
        for qrel in qrels
    }
    targets_by_group: dict[tuple[Any, Any], list[tuple[dict[str, Any], set[int]]]] = {}
    for frag in fragments:
        if frag.get("role") == "right_target":
            key = (frag.get("source_table_id"), frag.get("split"))
            targets_by_group.setdefault(key, []).append((frag, source_column_set(frag)))
    queries = [frag for frag in fragments if frag.get("role") in {"left_visible", "left_hidden"}]
    added = 0

    for query in queries:
        bridge_cols = query_bridge_columns(query)
        if not bridge_cols:
            continue
        query_id = query["fragment_id"]
        source_table_id = query.get("source_table_id")
        split = query.get("split")
        for target, target_cols in targets_by_group.get((source_table_id, split), []):
            matched_cols = sorted(bridge_cols & target_cols)
            if not matched_cols:
                continue
            target_id = target["fragment_id"]
            if (query_id, target_id) in existing_pairs:
                continue

            bridge_col = matched_cols[0]
            bridge_name = (
                query.get("hidden_bridge_col_name")
                or query.get("visible_bridge_col_name")
                or get_column_name(query, bridge_col)
            )
            is_hidden = query.get("role") == "left_hidden"
            pairs.append(
                {
                    "pair_id": f"pair_{stable_hash(query_id, target_id, 'same_source_bridge')}",
                    "source_table_id": source_table_id,
                    "split": split,
                    "chain_id": query.get("chain_id"),
                    "target_chain_id": target.get("chain_id"),
                    "query_fragment_id": query_id,
                    "target_fragment_id": target_id,
                    "label": 1,
                    "weight": 0.4 if is_hidden else 1.0,
                    "reason": f"same_source_bridge_column:{bridge_name}",
                }
            )
            existing_pairs.add((query_id, target_id))
            if (query_id, target_id) not in existing_qrels:
                qrels.append(
                    {
                        "query_id": query_id,
                        "target_id": target_id,
                        "rel": 2 if is_hidden else 3,
                        "split": split,
                        "chain_id": query.get("chain_id"),
                        "target_chain_id": target.get("chain_id"),
                        "query_role": query.get("role"),
                        "target_role": "right_target",
                        "reason": f"same_source_bridge_column:{bridge_name}",
                    }
                )
                existing_qrels.add((query_id, target_id))
            added += 1
    return added


def build_readme(stage1_dir: Path) -> None:
    text = """# Stage-1 Logic Connectivity Outputs

This directory contains the stage-1 coarse recall benchmark and training artifacts built from existing `output_medium` only.

Key files:

- `logic_fragments.jsonl`: table fragments with stable `fragment_id`, `object_type=table_fragment`, `role`, `split`, `chain_id`, source columns/rows, and provenance.
- `logic_pairs.jsonl`: self-supervised and weak table-table connectivity pairs.
- `evidence_paths.jsonl`: candidate `Q_hidden -> asset -> T` paths. These are candidates, not automatic positives.
- `hitl_pool.jsonl`: weak-labeled and unlabeled paths for human-in-the-loop selection.
- `train_pairs.jsonl`: pair and path samples used by the pairwise teacher.
- `embeddings/`: frozen Qwen3-VL-Embedding object vectors.
- `teacher/` and `teacher_scores.jsonl`: pairwise MLP teacher checkpoint and scores.
- `student/`: distilled type-projection and relation-matrix student.
- `hnsw_indices/`: one HNSW index per object type.
- `qrels.jsonl`: table-to-table relevance judgements for evaluation.
- `eval_results.json`: stage-1 recall metrics.

All example commands use the existing MMDD conda environment, for example:

```bash
conda run -n MMDD python scripts/build_stage1_logic_connectivity.py --input_dir output_medium --output_dir output_stage1_logic --seed 13
```

Embedding serialization is intentionally leakage-safe. `role`, `chain_id`, hidden bridge columns/values, source ids, qrels, labels, and pair/path annotations are provenance or supervision fields only and are not serialized into encoder inputs.
"""
    (stage1_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    setup_logging()
    rng = random.Random(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    table_only = bool(getattr(args, "table_only", False))
    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = load_split_map(input_dir)

    fragments: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    debug_examples: list[dict[str, Any]] = []
    chain_count = 0

    for table_idx, table in enumerate(iter_manifest_records(input_dir, "source_tables", log_every=1000), 1):
        source_table_id = table.get("source_table_id")
        split = split_map.get(source_table_id, "unknown")
        profiles = column_profiles(table)
        anchors = list(table.get("metadata", {}).get("candidate_entity_columns", []) or [])
        if not anchors:
            continue
        table_chains = 0
        all_cols = [int(col.get("column_index")) for col in table.get("columns", [])]
        for a_col in anchors:
            if table_chains >= args.max_chains_per_table:
                break
            try:
                a_col = int(a_col)
            except (TypeError, ValueError):
                continue
            bridges_for_anchor = 0
            for b_col in all_cols:
                if table_chains >= args.max_chains_per_table:
                    break
                if bridges_for_anchor >= args.max_bridges_per_anchor:
                    break
                if b_col == a_col or is_useless_column_name(get_column_name(table, b_col)):
                    continue
                if not valid_bridge_profile(profiles.get(b_col, {}), args.max_bridge_unique_ratio):
                    continue
                ab_purity, ab_support = fd_purity(table.get("rows", []), a_col, b_col)
                if ab_purity < args.min_ab_purity or ab_support < args.min_support:
                    continue

                valid_cs: list[tuple[float, int]] = []
                for c_col in all_cols:
                    if c_col in {a_col, b_col} or is_useless_column_name(get_column_name(table, c_col)):
                        continue
                    if not valid_target_profile(table, c_col, profiles.get(c_col, {})):
                        continue
                    bc_purity, bc_support = fd_purity(table.get("rows", []), b_col, c_col)
                    if bc_purity >= args.min_bc_purity and bc_support >= args.min_support:
                        valid_cs.append((bc_purity, c_col))
                if not valid_cs:
                    continue
                valid_cs.sort(reverse=True)
                c_cols = [c for _, c in valid_cs[: args.max_target_attrs]]

                query_context_cols = choose_query_context_cols(
                    table,
                    {a_col, b_col, *c_cols},
                    args.max_query_context_attrs,
                )
                qv_cols = [a_col, b_col] + query_context_cols
                qv_rows, qv_source_rows = project_rows(table, qv_cols, dedupe_col=a_col, min_required_cols=2)
                t_cols = [b_col] + c_cols
                t_rows, t_source_rows = build_target_rows(table, b_col, c_cols)
                if table_only:
                    if min(len(qv_rows), len(t_rows)) < args.min_rows_per_fragment:
                        continue
                    qh_rows: list[dict[str, Any]] = []
                    qh_source_rows: list[int] = []
                    qh_cols: list[int] = []
                else:
                    qh_cols = [a_col] + query_context_cols
                    qh_rows, qh_source_rows = project_rows(table, qh_cols, dedupe_col=a_col, min_required_cols=1)
                    if min(len(qv_rows), len(qh_rows), len(t_rows)) < args.min_rows_per_fragment:
                        continue
                if min(len(qv_rows), len(t_rows)) < args.min_rows_per_fragment:
                    continue

                chain_id = f"chain_{stable_hash(source_table_id, a_col, b_col, ','.join(map(str, c_cols)))}"
                a_name = get_column_name(table, a_col)
                b_name = get_column_name(table, b_col)
                c_names = [get_column_name(table, c_col) for c_col in c_cols]
                q_visible = make_fragment(
                    table,
                    split,
                    chain_id,
                    "left_visible",
                    qv_cols,
                    qv_rows,
                    qv_source_rows,
                    f"{a_name} -> {b_name}",
                    {
                        "visible_bridge": True,
                        "visible_bridge_col": b_col,
                        "visible_bridge_col_name": b_name,
                        "query_context_col_names": [get_column_name(table, col) for col in query_context_cols],
                    },
                )
                q_hidden = None
                if not table_only:
                    q_hidden = make_fragment(
                        table,
                        split,
                        chain_id,
                        "left_hidden",
                        qh_cols,
                        qh_rows,
                        qh_source_rows,
                        f"{a_name} -> hidden({b_name})",
                        {
                            "visible_bridge": False,
                            "hidden_bridge_col": b_col,
                            "hidden_bridge_col_name": b_name,
                            "query_context_col_names": [get_column_name(table, col) for col in query_context_cols],
                        },
                    )
                target = make_fragment(
                    table,
                    split,
                    chain_id,
                    "right_target",
                    t_cols,
                    t_rows,
                    t_source_rows,
                    f"{b_name} -> {', '.join(c_names)}",
                    {"target_bridge_col_name": b_name},
                )
                fragments.extend([q_visible, target] if table_only else [q_visible, q_hidden, target])
                pairs.append(
                    {
                        "pair_id": f"pair_{stable_hash(q_visible['fragment_id'], target['fragment_id'])}",
                        "source_table_id": source_table_id,
                        "split": split,
                        "chain_id": chain_id,
                        "query_fragment_id": q_visible["fragment_id"],
                        "target_fragment_id": target["fragment_id"],
                        "label": 1,
                        "weight": 1.0,
                        "reason": f"visible_chain:{a_name}->{b_name}+{b_name}->{','.join(c_names)}",
                    }
                )
                if q_hidden is not None:
                    pairs.append(
                        {
                            "pair_id": f"pair_{stable_hash(q_hidden['fragment_id'], target['fragment_id'])}",
                            "source_table_id": source_table_id,
                            "split": split,
                            "chain_id": chain_id,
                            "query_fragment_id": q_hidden["fragment_id"],
                            "target_fragment_id": target["fragment_id"],
                            "label": 1,
                            "weight": 0.4,
                            "reason": f"latent_chain:{a_name}->hidden({b_name})+{b_name}->{','.join(c_names)}",
                        }
                    )
                qrels.append(
                    {
                        "query_id": q_visible["fragment_id"],
                        "target_id": target["fragment_id"],
                        "rel": 3,
                        "split": split,
                        "chain_id": chain_id,
                        "query_role": "left_visible",
                        "target_role": "right_target",
                    }
                )
                if q_hidden is not None:
                    qrels.append(
                        {
                            "query_id": q_hidden["fragment_id"],
                            "target_id": target["fragment_id"],
                            "rel": 2,
                            "split": split,
                            "chain_id": chain_id,
                            "query_role": "left_hidden",
                            "target_role": "right_target",
                        }
                    )
                if len(debug_examples) < 25:
                    debug_examples.append(
                        {
                            "chain_id": chain_id,
                            "source_table_id": source_table_id,
                            "split": split,
                            "columns": {"A": a_name, "B": b_name, "C": c_names},
                            "rows": q_visible["rows"][:3],
                        }
                    )
                table_chains += 1
                chain_count += 1
                bridges_for_anchor += 1

        if table_idx % 500 == 0:
            print(f"processed_tables={table_idx} chains={chain_count}")

    bridge_positive_pairs = add_same_source_bridge_positive_pairs(fragments, pairs, qrels)
    rng.shuffle(debug_examples)
    counts = {
        "logic_fragments": write_jsonl(output_dir / "logic_fragments.jsonl", fragments),
        "logic_pairs": write_jsonl(output_dir / "logic_pairs.jsonl", pairs),
        "qrels": write_jsonl(output_dir / "qrels.jsonl", qrels),
        "debug_examples": write_jsonl(output_dir / "debug_chain_examples.jsonl", debug_examples),
        "chains": chain_count,
        "same_source_bridge_positive_pairs": bridge_positive_pairs,
    }
    build_readme(output_dir)
    update_stage1_manifest(output_dir, "logic_connectivity", {"input_dir": args.input_dir, "counts": counts, "args": vars(args)})
    print(json.dumps(counts, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="output_medium")
    parser.add_argument("--output_dir", default="output_stage1_logic")
    parser.add_argument("--min_rows_per_fragment", type=int, default=5)
    parser.add_argument("--max_chains_per_table", type=int, default=10)
    parser.add_argument("--max_bridges_per_anchor", type=int, default=5)
    parser.add_argument("--max_target_attrs", type=int, default=2)
    parser.add_argument("--max_query_context_attrs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min_ab_purity", type=float, default=0.95)
    parser.add_argument("--min_bc_purity", type=float, default=0.85)
    parser.add_argument("--min_support", type=int, default=6)
    parser.add_argument("--max_bridge_unique_ratio", type=float, default=0.85)
    parser.add_argument("--table_only", action="store_true", help="Build only visible table-table connectivity; do not create hidden bridge queries.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

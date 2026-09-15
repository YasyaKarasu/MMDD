#!/usr/bin/env python3
"""Build frozen Qwen column representations for the R25 F2 diagnostic."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cache_stage1_features import _load_embedder_class, encode_inputs
from mmdd_stage2.data import column_values, load_stage2_index


def _ranking_ids(paths: list[Path]) -> tuple[set[str], set[str]]:
    query_ids: set[str] = set()
    target_ids: set[str] = set()
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = row.get("query_id")
                if query_id is not None:
                    query_ids.add(str(query_id))
                if "scorer_ids" in row:
                    scorers = row["scorer_ids"]
                    for key in ("DIRECT_ANN", "QT_OVER_U"):
                        for value in scorers.get(key, {}).get("candidate_ids", []):
                            target_ids.add(str(value))
                else:
                    for key in ("direct_ann", "evidence_ann", "U"):
                        target_ids.update(str(value) for value in row.get(key, []))
                for result in row.get("results", []):
                    target_ids.add(str(result["target_id"]))
    return query_ids, target_ids


def _column_text(table: dict[str, Any], column: dict[str, Any], max_values: int) -> str:
    index = int(column["column_index"])
    values = column_values(table, index, include_empty=True)[:max_values]
    payload = {"column_name": str(column.get("column_name", "")), "visible_cells": values}
    return json.dumps(payload, ensure_ascii=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    ranking_paths = [Path(path).resolve() for path in args.ranking]
    query_ids, target_ids = _ranking_ids(ranking_paths)
    if args.train_retrieval:
        train_path = Path(args.train_retrieval).resolve()
        with train_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    query_ids.add(str(row["query_id"]))
                    target_ids.update(str(item["target_id"]) for item in row.get("results", []))
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    selected_targets = sorted(target_ids)
    selected_targets = [value for index, value in enumerate(selected_targets) if index % args.shard_count == args.shard_index]
    selected_queries = sorted(query_ids) if args.include_queries else []
    wanted = set(selected_targets)
    loaded = load_stage2_index(
        dataset_root,
        query_ids=set(selected_queries),
        target_ids=wanted,
        evidence_ids=set(),
    )
    targets = loaded.targets
    queries = loaded.queries
    missing = wanted - targets.keys()
    if missing:
        raise KeyError(f"missing target tables: {sorted(missing)[:3]}")
    if selected_queries:
        missing_queries = set(selected_queries) - queries.keys()
        if missing_queries:
            raise KeyError(f"missing query tables: {sorted(missing_queries)[:3]}")

    embedder_class = _load_embedder_class(Path(args.model_dir).resolve())
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    embedder = embedder_class(model_name_or_path=str(Path(args.model_dir).resolve()), torch_dtype=dtype)
    device = torch.device(args.device)
    embedder.model.to(device)
    embedder.model.eval()
    items: list[tuple[str, str, str]] = []
    instruction = "Represent this visible table column for joinability comparison. Preserve the column name and visible cell values; do not infer hidden attributes."
    for table_id, table in list(targets.items()) + list(queries.items()):
        role = "target" if table_id in targets else "query"
        for column in table["columns"]:
            items.append((role, table_id, str(column["column_index"])))
    vectors: dict[str, torch.Tensor] = {}
    partial = Path(args.output).resolve().with_suffix(Path(args.output).suffix + ".partial")
    if partial.is_file():
        payload = torch.load(partial, map_location="cpu", weights_only=False)
        vectors.update(payload.get("vectors", {}))
    pending_items = [item for item in items if f"{item[0]}:{item[1]}:{item[2]}" not in vectors]
    for start in range(0, len(pending_items), args.batch_size):
        batch_items = pending_items[start : start + args.batch_size]
        model_items = []
        for role, table_id, column_index in batch_items:
            table = targets.get(table_id) or queries[table_id]
            column = next(column for column in table["columns"] if str(column["column_index"]) == column_index)
            model_items.append({"text": _column_text(table, column, args.max_values), "instruction": instruction})
        encoded = encode_inputs(embedder, model_items, include_hidden=False)
        for key, (embedding, _hidden, _ids) in zip(batch_items, encoded, strict=True):
            role, table_id, column_index = key
            vectors[f"{role}:{table_id}:{column_index}"] = F.normalize(embedding.float(), dim=0).half().cpu()
        torch.cuda.empty_cache()
        if start // args.batch_size % 20 == 0:
            print(json.dumps({"columns": min(start + args.batch_size, len(pending_items)), "total": len(pending_items), "cached": len(vectors)}), flush=True)
        if start // args.batch_size % 500 == 0:
            torch.save({"format_version": 1, "model_dir": str(Path(args.model_dir).resolve()), "vectors": vectors}, partial)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_dir": str(Path(args.model_dir).resolve()), "instruction": instruction, "max_values": args.max_values, "vectors": vectors}, output)
    if partial.exists():
        partial.unlink()
    result = {"status": "complete", "target_tables": len(targets), "query_tables": len(queries), "column_vectors": len(vectors), "output": str(output)}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--ranking", nargs="+", required=True)
    parser.add_argument("--train-retrieval")
    parser.add_argument("--model-dir", default="hf_models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-values", type=int, default=5)
    parser.add_argument("--include-queries", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

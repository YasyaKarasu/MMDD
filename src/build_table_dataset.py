#!/usr/bin/env python
"""Build a multimodal table lake and projected query workload."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mmdd_dataset.assets import fetch_assets
from mmdd_dataset.joinability import table_asset_links
from mmdd_dataset.tables import prepare_entitables, prepare_wdc
from mmdd_dataset.utils import read_jsonl, source_splits, write_json, write_jsonl
from mmdd_dataset.workload import generate_query_views, query_view_asset_links


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "entitables":
        prepared = prepare_entitables(
            input_dir,
            min_rows=args.min_rows,
            min_cols=args.min_cols,
            wiki_threshold=args.wiki_link_threshold,
            max_tables=args.max_tables,
        )
    else:
        prepared = prepare_wdc(
            input_dir,
            min_rows=args.min_rows,
            min_cols=args.min_cols,
            max_tables=args.max_tables,
            max_rows=args.max_rows_per_table,
        )

    views = [
        view
        for table in prepared.source_tables
        for view in generate_query_views(table, args.max_query_views_per_table, args.seed)
    ]
    splits, split_of = source_splits(
        prepared.source_tables,
        split_by=args.split_by,
        ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
        seed=args.seed,
    )
    for split in ("train", "dev", "test"):
        splits[split]["query_view_ids"] = [
            view["query_view_id"]
            for view in views
            if split_of[view["source_table_id"]] == split
        ]
        splits[split].pop("query_table_ids")
        splits[split].pop("data_lake_table_ids")

    failures: list[dict[str, str]] = []
    assets: list[dict[str, Any]] = []
    if args.assets_jsonl:
        assets = list(read_jsonl(Path(args.assets_jsonl)))
    elif args.fetch_assets:
        assets, failures = fetch_assets(
            prepared.entities,
            output_dir,
            max_entities=args.max_entities,
            max_images_per_entity=args.max_images_per_entity,
            user_agent=args.user_agent,
        )
    links = table_asset_links(prepared.source_tables, prepared.entities, assets)
    links.extend(query_view_asset_links(views, prepared.entities, assets))

    artifacts = {
        "source_tables": prepared.source_tables,
        "query_views": views,
        "entities": prepared.entities,
        "bridge_assets": assets,
        "table_asset_links": links,
    }
    counts = {
        name: write_jsonl(output_dir / f"{name}.jsonl", records)
        for name, records in artifacts.items()
    }
    stats = {
        **counts,
        "skipped_tables": sum(prepared.skipped.values()),
        "skipped_reasons": prepared.skipped,
        "asset_failures": len(failures),
    }
    write_json(output_dir / "splits.json", splits)
    write_json(output_dir / "stats.json", stats)
    write_jsonl(output_dir / "asset_failures.jsonl", failures)
    write_json(
        output_dir / "dataset_manifest.json",
        {
            "format": "mmdd_table_workload_research_v1",
            "artifacts": {
                name: {"path": f"{name}.jsonl", "records": count}
                for name, count in counts.items()
            },
            "single_files": {"splits": "splits.json", "stats": "stats.json"},
        },
    )
    return stats


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source", choices=("entitables", "wdc"), default="entitables")
    result.add_argument("--input-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--min-rows", type=int, default=5)
    result.add_argument("--min-cols", type=int, default=2)
    result.add_argument("--max-tables", type=int)
    result.add_argument("--max-rows-per-table", type=int)
    result.add_argument("--wiki-link-threshold", type=float, default=0.3)
    result.add_argument("--max-query-views-per-table", type=int, default=5)
    result.add_argument("--seed", type=int, default=13)
    result.add_argument("--split-by", choices=("page_title", "source_table_id"), default="page_title")
    result.add_argument("--train-ratio", type=float, default=0.8)
    result.add_argument("--dev-ratio", type=float, default=0.1)
    result.add_argument("--test-ratio", type=float, default=0.1)
    result.add_argument("--fetch-assets", action="store_true")
    result.add_argument("--assets-jsonl")
    result.add_argument("--max-entities", type=int)
    result.add_argument("--max-images-per-entity", type=int, default=1)
    result.add_argument("--user-agent", default="MMDD research dataset builder")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    stats = build(args)
    print(f"built {stats['source_tables']} source tables and {stats['query_views']} query views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

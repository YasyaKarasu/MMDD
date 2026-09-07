from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .assets import fetch_assets
from .extraction import (
    OpenAICompatibleExtractor,
    auto_check_recoveries,
    build_extractions,
)
from .joinability import (
    JOINABILITY_POLICY_VERSION,
    MIN_IMPLICIT_CONTEXT_COLUMNS,
    BuildConfig,
    build_joinability_dataset,
    table_asset_links,
)
from .tables import prepare_entitables, prepare_wdc
from .utils import read_jsonl, source_splits, write_json, write_jsonl

ARTIFACTS = (
    "source_tables",
    "entities",
    "bridge_assets",
    "table_asset_links",
    "attribute_extractions",
    "query_tables",
    "data_lake_tables",
    "qrels",
    "evidence_recoveries",
    "table_queryability_decisions",
)


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

    splits, split_of = source_splits(
        prepared.source_tables,
        ratios=(args.train_ratio, args.dev_ratio, args.test_ratio),
        seed=args.seed,
    )

    failures: list[dict[str, str]] = []
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
    else:
        assets = []

    text_extractor = None
    image_extractor = None
    luna_extractor = None
    terra_extractor = None
    if args.extractions_jsonl:
        extractions = list(read_jsonl(Path(args.extractions_jsonl)))
    elif args.text_model_base_url or args.image_model_base_url:
        text_extractor = (
            OpenAICompatibleExtractor(
                args.text_model_base_url,
                args.text_model_name,
                api_key_env=args.text_model_api_key_env,
                timeout=args.model_timeout,
            )
            if args.text_model_base_url
            else None
        )
        image_extractor = (
            OpenAICompatibleExtractor(
                args.image_model_base_url,
                args.image_model_name,
                api_key_env=args.image_model_api_key_env,
                timeout=args.model_timeout,
            )
            if args.image_model_base_url
            else None
        )
        if args.auto_check_mode == "cascade":
            luna_extractor = OpenAICompatibleExtractor(
                args.auto_check_luna_base_url,
                args.auto_check_luna_model,
                api_key_env=args.auto_check_api_key_env,
                timeout=args.model_timeout,
            )
            terra_extractor = OpenAICompatibleExtractor(
                args.auto_check_terra_base_url,
                args.auto_check_terra_model,
                api_key_env=args.auto_check_api_key_env,
                timeout=args.model_timeout,
            )
        extractions = build_extractions(
            prepared.source_tables,
            prepared.entities,
            assets,
            text_extractor,
            image_extractor=image_extractor,
            min_column_non_empty_ratio=args.min_column_non_empty_ratio,
        )
    else:
        extractions = []

    config = BuildConfig(
        seed=args.seed,
        query_rows=args.query_rows,
        min_target_rows=args.min_target_rows,
        min_recovered_ratio=args.min_recovered_ratio,
        min_recovered_rows=args.min_recovered_rows,
        min_column_non_empty_ratio=args.min_column_non_empty_ratio,
        max_query_additional_columns=args.max_query_additional_columns,
        max_target_additional_columns=args.max_target_additional_columns,
    )
    artifacts = build_joinability_dataset(
        prepared.source_tables, assets, extractions, split_of, config
    )
    if text_extractor is not None or image_extractor is not None:
        artifacts = auto_check_recoveries(
            artifacts,
            assets,
            text_extractor,
            image_extractor=image_extractor,
            review_mode=args.auto_check_mode,
            luna_extractor=luna_extractor,
            terra_extractor=terra_extractor,
        )
    artifacts.update(
        {
            "source_tables": prepared.source_tables,
            "entities": prepared.entities,
            "bridge_assets": assets,
            "attribute_extractions": extractions,
            "table_asset_links": table_asset_links(
                prepared.source_tables, prepared.entities, assets
            ),
        }
    )

    for query in artifacts["query_tables"]:
        splits[query["split"]]["query_table_ids"].append(query["table_id"])
    splits["data_lake_table_ids"] = sorted(
        target["table_id"] for target in artifacts["data_lake_tables"]
    )

    counts = {
        artifact: write_jsonl(output_dir / f"{artifact}.jsonl", artifacts[artifact])
        for artifact in ARTIFACTS
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
            "format": "mmdd_joinability_research_v2",
            "query_construction": {
                "policy_version": JOINABILITY_POLICY_VERSION,
                "context_attr_limit_policy": "compatibility_flags_ignored",
                "identical_visible_query_policy": "multiple_positive_targets",
                "target_column_order_policy": "seeded_shuffle_per_join_column",
                "min_implicit_context_columns": MIN_IMPLICIT_CONTEXT_COLUMNS,
                "sibling_source_column_policy": (
                    "exact_redundancy_groups_one_query_bridge_with_physical_target_fanout"
                ),
                "cell_text_policy": "clean_and_truncate_1024",
            },
            "artifacts": {
                artifact: {"path": f"{artifact}.jsonl", "records": counts[artifact]}
                for artifact in ARTIFACTS
            },
            "single_files": {"splits": "splits.json", "stats": "stats.json"},
        },
    )
    return stats


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the MMDD multimodal joinability research dataset."
    )
    result.add_argument("--source", choices=("entitables", "wdc"), required=True)
    result.add_argument("--input-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--min-rows", type=int, default=5)
    result.add_argument("--min-cols", type=int, default=2)
    result.add_argument("--max-tables", type=int)
    result.add_argument("--max-rows-per-table", type=int)
    result.add_argument("--wiki-link-threshold", type=float, default=0.3)
    result.add_argument("--seed", type=int, default=13)
    result.add_argument("--train-ratio", type=float, default=0.8)
    result.add_argument("--dev-ratio", type=float, default=0.1)
    result.add_argument("--test-ratio", type=float, default=0.1)

    evidence = result.add_argument_group("evidence")
    evidence.add_argument("--assets-jsonl")
    evidence.add_argument("--fetch-assets", action="store_true")
    evidence.add_argument("--max-entities", type=int)
    evidence.add_argument("--max-images-per-entity", type=int, default=1)
    evidence.add_argument("--user-agent", default="MMDD research dataset builder")

    model = result.add_argument_group("attribute extraction")
    model.add_argument("--extractions-jsonl")
    model.add_argument("--text-model-base-url")
    model.add_argument("--text-model-name", default="Qwen3.5-9B")
    model.add_argument("--text-model-api-key-env", default="VLLM_API_KEY")
    model.add_argument("--image-model-base-url")
    model.add_argument("--image-model-name", default="Qwen3-VL-8B-Instruct")
    model.add_argument("--image-model-api-key-env", default="VLLM_API_KEY")
    model.add_argument("--model-timeout", type=float, default=120)
    model.add_argument(
        "--auto-check-mode",
        choices=("cascade", "local"),
        default="cascade",
        help=(
            "Use local+Luna+Terra consensus checking, or local-only checking "
            "for builds that must not call a remote API."
        ),
    )
    model.add_argument(
        "--auto-check-luna-base-url",
        default="https://api.openai.com/v1",
    )
    model.add_argument("--auto-check-luna-model", default="gpt-5.6-luna")
    model.add_argument(
        "--auto-check-terra-base-url",
        default="https://api.openai.com/v1",
    )
    model.add_argument("--auto-check-terra-model", default="gpt-5.6-terra")
    model.add_argument("--auto-check-api-key-env", default="OPENAI_API_KEY")

    algorithm = result.add_argument_group("joinability algorithm")
    algorithm.add_argument("--query-rows", type=int, default=5)
    algorithm.add_argument("--min-target-rows", type=int, default=5)
    algorithm.add_argument("--min-recovered-ratio", type=float, default=0.6)
    algorithm.add_argument("--min-recovered-rows", type=int, default=3)
    algorithm.add_argument("--min-column-non-empty-ratio", type=float, default=0.5)
    algorithm.add_argument(
        "--max-query-additional-columns",
        type=int,
        default=1,
        help="Deprecated compatibility option; all query-pool columns are emitted.",
    )
    algorithm.add_argument(
        "--max-target-additional-columns",
        type=int,
        default=2,
        help="Deprecated compatibility option; all target-pool columns are emitted.",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    stats = build(args)
    print(
        f"built {stats['source_tables']} source tables, "
        f"{stats['query_tables']} queries, and {stats['data_lake_tables']} targets"
    )
    return 0

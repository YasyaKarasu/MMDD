#!/usr/bin/env python
"""Build the AbeBooks joinability dataset with the EntiTables pipeline.

This is the AbeBooks sibling of ``build_wdc_mm_joinability_dataset.py``: it owns
the input adaptation and the command line, and **reuses the EntiTables builder's
construction core** rather than reimplementing it --

* ``collect_extraction_tasks_from_tables`` builds one task per (row, asset)
  asking for every candidate attribute at once;
* ``build_table_join_records`` runs extraction, the blind attribute check and the
  join construction, including the two-layout protocol (a preliminary layout is
  what the check reviews, and its verdicts are frozen before the final layout);
* ``finalize_query_recovery_auto_checks`` drives the reviewer cascade, which
  routes across every provider the auto-check API config declares;
* ``source_splits``, ``ExtractionCache``, ``ShardedJsonlWriter`` and the manifest
  shape come from the same place, so the artifacts are the EntiTables format.

What is AbeBooks-specific lives in two places: ``mmdd_dataset.abebooks_adapter``
(the lake -> tables/entities/assets) and this file's argument surface.  The lake
is expected to have been built by ``build_abebooks_lake.py`` -- cells with the
shared shape, bridge assets keyed by ``row_id``, and the copy-channel columns
already excluded, so no text asset is a verbatim copy of a column it could be
asked to "recover".

Usage::

    python scripts_old/build_abebooks_mm_joinability_dataset.py \\
        --lake_dir output/abebooks_lake_no_copy \\
        --output_dir output/abebooks_joinability_entitables_style \\
        --text_model_base_url http://127.0.0.1:18000/v1 --text_model_name Qwen3.5-9B \\
        --image_model_base_url http://127.0.0.1:18000/v1 --image_model_name Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import json
import signal
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_mm_joinability_dataset as join_builder  # noqa: E402
from build_mm_table_dataset import ShardedJsonlWriter, write_jsonl  # noqa: E402
from mmdd_dataset.abebooks_adapter import adapt_assets, prepare_abebooks  # noqa: E402
from mmdd_dataset.joinability import JOINABILITY_POLICY_VERSION  # noqa: E402
from mmdd_dataset.utils import clean_text, write_json  # noqa: E402

DEFAULT_CACHE_DIR = Path("cache/abebooks_mm_joinability")


def add_lake_arguments(parser: argparse.ArgumentParser) -> None:
    """The AbeBooks-side options; everything else comes from the parent parser."""
    parser.add_argument(
        "--lake_dir", required=True,
        help="a directory built by build_abebooks_lake.py")
    parser.add_argument("--output_dir", required=True)
    # Named explicitly so a run can never land in the EntiTables cache: the
    # parent's default (``cache/mm_joinability``) is a 7.9 GB corpus from another
    # dataset, and appending to it is both rude and confusing.
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument(
        "--allow_unbounded", action="store_true",
        help="read every table in the lake instead of --max_source_tables")
    parser.add_argument(
        "--image_root", default=".",
        help="what the lake's relative image paths resolve against, as at build time")
    parser.add_argument(
        "--split_by", default="source_table_id", choices=("source_table_id", "page_title"),
        help="the lake has no page titles, so grouping by source table is the only "
             "split that spreads tables across train/dev/test")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    """``(parent_namespace, abebooks_namespace)``.

    The parent's parser is the source of every model, cache, ratio and
    auto-check option -- declaring them twice is how a sibling drifts from the
    builder it is supposed to be reusing.  Only the lake options are ours; the
    parent's required ``--input_dir`` is satisfied from ``--lake_dir``.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    own = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False,
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_lake_arguments(own)
    if "--help" in raw or "-h" in raw:
        # Two parsers, so the help has to be two texts: ours first, then the
        # parent's -- which prints and exits.
        own.print_help()
        print("\nEverything else is the EntiTables builder's own surface:\n")
        join_builder.parse_args(["--input_dir", ".", "--help"])
        raise SystemExit(0)
    lake_args, passthrough = own.parse_known_args(raw)
    args = join_builder.parse_args([
        "--input_dir", str(lake_args.lake_dir),
        "--output_dir", str(lake_args.output_dir),
        "--cache_dir", str(lake_args.cache_dir),
        *passthrough,
    ])
    # The parent defaults to splitting by page title, which no AbeBooks table has.
    if "--split_by" not in (argv or sys.argv[1:]):
        args.split_by = lake_args.split_by
    args.lake_dir = lake_args.lake_dir
    args.image_root = lake_args.image_root
    args.allow_unbounded = lake_args.allow_unbounded
    return args, lake_args


def _write_shard(writer: ShardedJsonlWriter, records: Iterable[dict[str, Any]]) -> int:
    written = 0
    for record in records:
        writer.write_record(record)
        written += 1
    return written


def _table_asset_links(
    source_tables: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """One link per (row, entity) with every asset that row can be evidenced by.

    The shared builder reads assets through ``entity_to_assets`` directly, so
    this artifact is for consumers that need the link spelled out; it is written
    in the same shape the WDC sibling writes.
    """
    entity_by_row = {
        (entity["source_table_id"], int(entity["source_row_id"])): entity
        for entity in entities
    }
    links: list[dict[str, Any]] = []
    for table in source_tables:
        entity_column = int(table["metadata"]["candidate_entity_columns"][0])
        for row in table["rows"]:
            entity = entity_by_row.get((table["source_table_id"], int(row["row_id"])))
            if entity is None:
                continue
            asset_ids = entity_to_assets.get(entity["entity_id"], [])
            if not asset_ids:
                continue
            links.append({
                "source_table_id": table["source_table_id"],
                "row_id": row["row_id"],
                "column_index": entity_column,
                "entity_id": entity["entity_id"],
                "asset_ids": list(asset_ids),
            })
    return links


def install_stack_dump_handler() -> None:
    """Dump every thread's Python stack on SIGUSR1.

    The reviewer pool can stall with the main thread blocked acquiring a lock --
    measured on a real run: 134 threads all in ``futex_wait``, 0% CPU, and
    SIGINT unable to interrupt, because the interpreter never gets to run a
    Python-level handler.  ``py-spy`` needs ptrace permission the deployment may
    not grant; ``faulthandler`` writes from the signal handler itself, so it
    still produces a stack for a run that cannot otherwise be inspected.  A
    missing or unsupported signal must never cost the run.
    """
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, OSError, ValueError):
        pass


def build_dataset(
    args: argparse.Namespace,
    lake_args: argparse.Namespace,
    *,
    extractor: Any | None = None,
) -> dict[str, Any]:
    lake_dir = Path(lake_args.lake_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records_per_shard = int(getattr(args, "records_per_shard", 50000))

    prepared = prepare_abebooks(
        lake_dir,
        min_rows=int(getattr(args, "min_rows", 5)),
        min_cols=int(getattr(args, "min_cols", 2)),
        max_tables=(
            None if getattr(args, "allow_unbounded", False)
            else int(getattr(args, "max_source_tables", 100))
        ),
    )
    if not prepared.source_tables:
        raise SystemExit(f"{lake_dir} produced no usable tables: {prepared.skipped}")
    assets = adapt_assets(lake_dir, root=Path(lake_args.image_root).resolve())
    entity_to_assets: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        entity_to_assets[clean_text(asset["entity_id"])].append(asset["asset_id"])
    wiki_to_entity_id = {
        clean_text(entity["wiki_title"]): clean_text(entity["entity_id"])
        for entity in prepared.entities
    }

    source_writer = ShardedJsonlWriter(output_dir / "source_tables", records_per_shard)
    entity_writer = ShardedJsonlWriter(output_dir / "entities", records_per_shard)
    bridge_writer = ShardedJsonlWriter(output_dir / "bridge_assets", records_per_shard)
    link_writer = ShardedJsonlWriter(output_dir / "table_asset_links", records_per_shard)
    with source_writer as handle:
        source_table_count = _write_shard(handle, prepared.source_tables)
    with entity_writer as handle:
        entity_count = _write_shard(handle, prepared.entities)
    with bridge_writer as handle:
        asset_count = _write_shard(handle, assets)
    with link_writer as handle:
        link_count = _write_shard(
            handle,
            _table_asset_links(prepared.source_tables, prepared.entities, entity_to_assets),
        )

    split_records = [
        {"source_table_id": table["source_table_id"]}
        for table in prepared.source_tables
    ]
    splits, source_to_split = join_builder.source_splits(split_records, args)

    loaded_assets = join_builder.load_assets(bridge_writer.paths())
    cache = join_builder.ExtractionCache(
        cache_dir / "model_attribute_extractions.jsonl",
        reuse=not getattr(args, "no_reuse_model_cache", False),
    )
    query_auto_check_cache = join_builder.ExtractionCache(
        cache_dir / "query_recovery_auto_checks.jsonl",
        reuse=not getattr(args, "no_reuse_model_cache", False),
        record_key_alias=join_builder.query_recovery_auto_check_record_key,
    )
    concurrency_state = join_builder.ModelConcurrencyState.from_args(args)
    if extractor is None:
        extractor = join_builder.LocalAttributeExtractor(args)

    # The check's verdicts have to be frozen before the final layout is built,
    # so the plans are collected over every table first and only then resolved.
    if join_builder.auto_check_required(extractor):
        final_plans: list[Any] = []
        for table in prepared.source_tables:
            join_builder.build_table_join_records(
                source_table=table,
                split=source_to_split.get(table["source_table_id"], "test"),
                assets=loaded_assets,
                entity_to_assets=entity_to_assets,
                wiki_to_entity_id=wiki_to_entity_id,
                extractor=extractor,
                cache=cache,
                progress=None,
                concurrency_state=concurrency_state,
                extraction_writer=join_builder.ListRecordWriter(),
                recovery_writer=join_builder.ListRecordWriter(),
                args=args,
                query_auto_check_cache=query_auto_check_cache,
                apply_query_auto_check=False,
                query_recovery_plans_out=final_plans,
            )
        join_builder.finalize_query_recovery_auto_checks(
            plans=final_plans,
            extractor=extractor,
            cache=query_auto_check_cache,
            args=args,
            concurrency_state=concurrency_state,
        )

    query_writer = ShardedJsonlWriter(output_dir / "query_tables", records_per_shard)
    data_lake_writer = ShardedJsonlWriter(output_dir / "data_lake_tables", records_per_shard)
    extraction_writer = ShardedJsonlWriter(output_dir / "attribute_extractions", records_per_shard)
    recovery_writer = ShardedJsonlWriter(output_dir / "evidence_recoveries", records_per_shard)
    qrels: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counts = defaultdict(int)
    implicit_query_counts_by_split = {"train": 0, "dev": 0, "test": 0}
    # A table can hold a multimodal query *and* be an explicit-join candidate; the
    # candidate is only realised if the balance pass picks it, so its data-lake
    # tables are written after the loop rather than here.
    explicit_candidate_splits: dict[str, str] = {}
    explicit_candidate_source_ids: dict[str, str] = {}
    explicit_candidate_decision_indices: dict[str, int] = {}
    deferred_tables: set[str] = set()
    with (
        query_writer as query_handle,
        data_lake_writer as data_lake_handle,
        extraction_writer as extraction_handle,
        recovery_writer as recovery_handle,
    ):
        for table in prepared.source_tables:
            source_table_id = table["source_table_id"]
            split = source_to_split.get(source_table_id, "test")
            query_tables, data_lake_tables, table_qrels, decision = (
                join_builder.build_table_join_records(
                    source_table=table,
                    split=split,
                    assets=loaded_assets,
                    entity_to_assets=entity_to_assets,
                    wiki_to_entity_id=wiki_to_entity_id,
                    extractor=extractor,
                    cache=cache,
                    progress=None,
                    concurrency_state=concurrency_state,
                    extraction_writer=extraction_handle,
                    recovery_writer=recovery_handle,
                    args=args,
                    query_auto_check_cache=query_auto_check_cache,
                    finalize_query_recoveries=True,
                )
            )
            if query_tables:
                counts["queryable_tables"] += 1
                if decision.get("reason") == "explicit_join_fallback":
                    counts["explicit_join_tables"] += 1
                else:
                    counts["multimodal_queryable_tables"] += 1
                    implicit_query_counts_by_split[split] += len(query_tables)
            else:
                counts["rejected_tables"] += 1
            candidates = decision.get("explicit_join_candidates")
            if not isinstance(candidates, list):
                candidate = decision.get("explicit_join_candidate")
                candidates = [candidate] if isinstance(candidate, dict) else []
            deferred_candidate = (
                args.explicit_join_fallback_mode == "match_implicit"
                and bool(candidates)
            )
            decision["source_table_id"] = source_table_id
            if args.explicit_join_fallback_mode == "match_implicit" and not candidates:
                # Any table can host an explicit join -- it only needs a visible
                # non-entity column with enough rows -- so the candidate pool is
                # every table, not just the ones whose multimodal recovery failed.
                # The parent draws candidates for those failures only, and at a
                # low recovery threshold most tables succeed: too few candidates
                # survive to give one explicit join per multimodal query, which is
                # what ``match_implicit`` has to balance.
                entity_col = (
                    table["metadata"].get("candidate_entity_columns") or [None]
                )[0]
                candidates = join_builder.build_explicit_join_fallback_candidates(
                    source_table=table,
                    split=split,
                    entity_col=entity_col,
                    rejected_multimodal_reason=clean_text(decision.get("reason")),
                    args=args,
                    force=True,
                )
                if candidates:
                    decision["explicit_join_candidates"] = candidates
                    decision["explicit_join_candidate"] = candidates[0]
            if args.explicit_join_fallback_mode == "match_implicit" and candidates:
                for candidate in candidates:
                    candidate_id = clean_text(candidate.get("candidate_id"))
                    if not candidate_id:
                        raise ValueError(
                            f"explicit join candidate for {source_table_id} has no "
                            "candidate_id, so the balance pass cannot select it")
                    explicit_candidate_splits[candidate_id] = split
                    explicit_candidate_source_ids[candidate_id] = source_table_id
                    explicit_candidate_decision_indices[candidate_id] = len(decisions)
            if deferred_candidate:
                deferred_tables.add(source_table_id)
            decisions.append(decision)
            for record in query_tables:
                query_handle.write_record(record)
                counts["query_tables"] += 1
            for record in ([] if deferred_candidate else data_lake_tables):
                data_lake_handle.write_record(record)
                counts["data_lake_tables"] += 1
            qrels.extend(table_qrels)
            counts["qrels"] += len(table_qrels)

        if args.explicit_join_fallback_mode == "match_implicit":
            # One explicit join per multimodal query per split, so the two kinds
            # come out balanced and the split holds for both.  This is the mode
            # the EntiTables dataset was built with; ``ratio`` (the default) just
            # lets whichever tables failed multimodal recovery fall back.
            selected_explicit, counts["explicit_candidates_by_split"] = (
                join_builder.select_balanced_explicit_join_candidates(
                    candidate_splits=explicit_candidate_splits,
                    implicit_query_counts=implicit_query_counts_by_split,
                    args=args,
                )
            )
            counts["explicit_selected_ids"] = len(selected_explicit)
            counts["explicit_needed"] = sum(implicit_query_counts_by_split.values())
            selected_by_source: dict[str, list[str]] = defaultdict(list)
            for candidate_id in selected_explicit:
                selected_by_source[explicit_candidate_source_ids[candidate_id]].append(
                    candidate_id)
            for table in prepared.source_tables:
                source_table_id = table["source_table_id"]
                if source_table_id not in set(explicit_candidate_source_ids.values()):
                    continue
                split = source_to_split.get(source_table_id, "test")
                selected_ids = sorted(selected_by_source.get(source_table_id, []))
                if not selected_ids:
                    if source_table_id in deferred_tables:
                        # Its data-lake record was held back in the loop; a
                        # candidate nobody picked is still a data-lake table.
                        data_lake_handle.write_record(
                            join_builder.raw_data_lake_record(table))
                        counts["data_lake_tables"] += 1
                    continue
                decision_index = explicit_candidate_decision_indices[selected_ids[0]]
                original = decisions[decision_index]
                materialized: list[dict[str, Any]] = []
                explicit_queries: list[dict[str, Any]] = []
                explicit_targets: list[dict[str, Any]] = []
                explicit_qrels: list[dict[str, Any]] = []
                for candidate_id in selected_ids:
                    candidate_decision = next(
                        item for item in original["explicit_join_candidates"]
                        if clean_text(item.get("candidate_id")) == candidate_id)
                    for candidate in join_builder.rebuild_selected_explicit_join_candidates(
                            source_table=table, split=split,
                            candidate_decisions=[candidate_decision], args=args):
                        queries, targets, candidate_qrels, result = (
                            join_builder.materialize_balanced_explicit_join_candidate(
                                source_table=table, split=split,
                                candidate_decision=candidate, args=args))
                        materialized.append(candidate)
                        explicit_queries.extend(queries)
                        explicit_targets.extend(targets)
                        explicit_qrels.extend(candidate_qrels)
                decisions[decision_index] = {
                    **original,
                    "source_table_id": source_table_id,
                    "explicit_join_candidates": materialized,
                    "explicit_join_query_count": len(explicit_queries),
                }
                counts["explicit_join_tables"] += 1
                counts["explicit_selected"] += len(selected_ids)
                counts["explicit_materialized"] += len(explicit_queries)
                for record in explicit_queries:
                    query_handle.write_record(record)
                    counts["query_tables"] += 1
                for record in explicit_targets:
                    data_lake_handle.write_record(record)
                    counts["data_lake_tables"] += 1
                qrels.extend(explicit_qrels)
                counts["qrels"] += len(explicit_qrels)

    write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "table_queryability_decisions.jsonl", decisions)
    write_json(output_dir / "splits.json", splits)
    stats = {
        "provider": "abebooks",
        "lake_dir": str(lake_dir),
        "source_tables": source_table_count,
        "entities": entity_count,
        "bridge_assets": asset_count,
        "table_asset_links": link_count,
        "query_tables": counts["query_tables"],
        "data_lake_tables": counts["data_lake_tables"],
        "qrels": counts["qrels"],
        "decisions": dict(sorted(
            Counter(clean_text(d.get("reason")) or "unknown" for d in decisions).items())),
        "tables": {key: counts[key] for key in sorted(counts) if key not in
                   {"query_tables", "data_lake_tables", "qrels"}},
        "skipped_source_tables": prepared.skipped,
    }
    write_json(output_dir / "stats.json", stats)
    write_json(output_dir / "dataset_manifest.json", {
        "format": "sharded_jsonl",
        "records_per_shard": records_per_shard,
        "artifact_references": {
            "data_lake_tables": {
                "field": "source_table_ref",
                "target_artifact": "source_tables",
                "resolution": "stream_by_source_table_id",
            }
        },
        "artifacts": {
            "source_tables": source_writer.manifest(output_dir),
            "entities": entity_writer.manifest(output_dir),
            "query_tables": query_writer.manifest(output_dir),
            "data_lake_tables": data_lake_writer.manifest(output_dir),
            "bridge_assets": bridge_writer.manifest(output_dir),
            "table_asset_links": link_writer.manifest(output_dir),
            "attribute_extractions": extraction_writer.manifest(output_dir),
            "evidence_recoveries": recovery_writer.manifest(output_dir),
        },
        "single_files": {
            "qrels": "qrels.jsonl",
            "splits": "splits.json",
            "stats": "stats.json",
            "table_queryability_decisions": "table_queryability_decisions.jsonl",
        },
        "query_construction": {
            "provider": "abebooks",
            "entity_layer": "opaque_abe_key_in_the_wiki_title_slot",
            "policy_version": JOINABILITY_POLICY_VERSION,
        },
    })
    return stats


def main(argv: list[str] | None = None) -> int:
    install_stack_dump_handler()
    args, lake_args = parse_args(argv)
    stats = build_dataset(args, lake_args)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mm_joinability_dataset as join_builder
import wdc200k_materialize as materializer
from wdc200k_materialize import (
    MaterializationInputs,
    MaterializationShardInputs,
    materialize_dataset,
    materialize_dataset_shard,
)
from wdc200k_io import (
    AtomicJsonlShard,
    CompletedShard,
    SqliteJobStore,
    StageFingerprint,
    StageManifest,
)
from wdc200k_models import (
    AssetStageBarrier,
    StructuralStageBarrier,
    enqueue_model_tasks,
    run_model_stage,
)
from wdc200k_structural import (
    STRUCTURAL_SCHEMA_VERSION,
    finalize_validated_selection,
)
from stage1_io import iter_manifest_records, load_split_map


class _MemoryCache:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.items = {
            str(record["cache_key"]): dict(record) for record in records
        }

    def get(self, key: str) -> dict[str, Any] | None:
        record = self.items.get(key)
        return dict(record) if record is not None else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.items[key] = dict(record)


class _RecordSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def _args(tmp_path: Path) -> argparse.Namespace:
    args = join_builder.parse_args(
        [
            "--input_dir",
            str(tmp_path),
            "--output_dir",
            str(tmp_path / "unused-output"),
            "--cache_dir",
            str(tmp_path / "cache"),
            "--query_rows_per_table",
            "2",
            "--min_rows_per_output_table",
            "2",
            "--min_recovery_denominator",
            "1",
            "--min_recovered_value_ratio",
            "0.5",
            "--no_reparse_cached_model_outputs",
            "--no_model_progress",
        ]
    )
    args.cache_failed_model_outputs = True
    return args


def _source_table(
    source_table_id: str = "source-1",
) -> dict[str, Any]:
    columns = [
        {
            "column_index": 0,
            "column_name": "Name",
            "is_numeric_column": False,
        },
        {
            "column_index": 1,
            "column_name": "State",
            "is_numeric_column": False,
        },
        {
            "column_index": 2,
            "column_name": "Category",
            "is_numeric_column": False,
        },
    ]
    rows = []
    for row_id, (name, state) in enumerate(
        (("Alpha", "Texas"), ("Beta", "Ohio"))
    ):
        rows.append(
            {
                "row_id": row_id,
                "cells": [
                    {
                        "column_index": 0,
                        "column_name": "Name",
                        "raw": name,
                        "text": name,
                        "wiki_title": f"wdc_{name.casefold()}",
                        "has_wiki_link": True,
                    },
                    {
                        "column_index": 1,
                        "column_name": "State",
                        "raw": state,
                        "text": state,
                        "wiki_title": None,
                        "has_wiki_link": False,
                    },
                    {
                        "column_index": 2,
                        "column_name": "Category",
                        "raw": "Place",
                        "text": "Place",
                        "wiki_title": None,
                        "has_wiki_link": False,
                    },
                ],
            }
        )
    return {
        "source_table_id": source_table_id,
        "source_file": f"Thing/{source_table_id}.json.gz",
        "page_title": "Thing",
        "caption": "",
        "section_title": "",
        "num_rows": len(rows),
        "num_cols": len(columns),
        "columns": columns,
        "rows": rows,
        "provenance_builder": "build_wdc_mm_joinability_dataset.py",
        "metadata": {
            "candidate_entity_columns": [0],
            "column_profiles": [
                {
                    "column_index": index,
                    "column_name": column["column_name"],
                    "non_empty_ratio": 1.0,
                    "unique_ratio": 1.0,
                    "numeric_ratio": 0.0,
                }
                for index, column in enumerate(columns)
            ],
        },
    }


def _entities() -> list[dict[str, Any]]:
    return [
        {
            "entity_id": f"entity-{name.casefold()}",
            "wiki_title": f"wdc_{name.casefold()}",
            "display_texts": [name],
            "context_terms": [],
            "appears_in": [
                {
                    "source_table_id": "source-1",
                    "query_view_id": None,
                    "row_id": row_id,
                    "column_index": 0,
                    "column_name": "Name",
                }
            ],
            "page_url": f"https://example.test/{name.casefold()}",
            "image_urls": [],
        }
        for row_id, name in enumerate(("Alpha", "Beta"))
    ]


def _assets() -> list[dict[str, Any]]:
    return [
        {
            "asset_id": f"asset-{name.casefold()}-{ordinal}",
            "source_asset_id": f"page-{name.casefold()}",
            "entity_id": f"entity-{name.casefold()}",
            "entity_wiki_title": f"wdc_{name.casefold()}",
            "asset_type": "text",
            "content": f"{name} is located in {state}.",
            "text_chunk_index": ordinal,
            "text_chunk_count": 2,
            "selected_text_chunk_count": 2,
            "text_chunk_relevance_score": 1.0,
            "source": "wdc_page_text_chunk",
            "url": f"https://example.test/{name.casefold()}",
            "page_url": f"https://example.test/{name.casefold()}",
            "final_url": f"https://example.test/{name.casefold()}",
        }
        for name, state in (("Alpha", "Texas"), ("Beta", "Ohio"))
        for ordinal in range(2)
    ]


def _links() -> list[dict[str, Any]]:
    return [
        {
            "link_id": f"link-{name.casefold()}",
            "source_table_id": "source-1",
            "query_view_id": None,
            "row_id": row_id,
            "column_index": 0,
            "column_name": "Name",
            "cell_text": name,
            "entity_id": f"entity-{name.casefold()}",
            "entity_wiki_title": f"WDC_{name.casefold()}",
            "asset_ids": [
                f"asset-{name.casefold()}-0",
                f"asset-{name.casefold()}-1",
            ],
        }
        for row_id, name in enumerate(("Alpha", "Beta"))
    ]


def _extractions(
    args: argparse.Namespace,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_by_entity = {
        "entity-alpha": "Texas",
        "entity-beta": "Ohio",
    }
    records = []
    for asset in assets:
        entity_id = str(asset["entity_id"])
        cache_key = join_builder.extraction_cache_key(
            asset_id=str(asset["asset_id"]),
            entity_id=entity_id,
            candidate_attribute_names=["State", "Category"],
            asset_type="text",
            args=args,
        )
        records.append(
            {
                "cache_key": cache_key,
                "entity_id": entity_id,
                "entity_text": entity_id.removeprefix("entity-").title(),
                "entity_wiki_title": str(asset["entity_wiki_title"]),
                "asset_id": asset["asset_id"],
                "asset_type": "text",
                "candidate_attribute_names": ["State", "Category"],
                "attributes": [
                    {
                        "name": "State",
                        "value": state_by_entity[entity_id],
                        "evidence": "page statement",
                        "connection_evidence": "same named entity",
                    }
                ],
                "raw_response": "",
                "error": "",
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    return path


def _relative_completed(
    completed: CompletedShard,
    path: Path,
    root: Path,
) -> CompletedShard:
    return replace(
        completed,
        path=path.relative_to(root).as_posix(),
    )


def _atomic_shard(
    root: Path,
    relative_path: str,
    records: list[dict[str, Any]],
) -> CompletedShard:
    path = root / relative_path
    writer = AtomicJsonlShard(path)
    for record in records:
        writer.write(record)
    return _relative_completed(writer.commit(), path, root)


def _structural_upstream(
    tmp_path: Path,
) -> tuple[
    Path,
    tuple[Path, ...],
    Path,
    StructuralStageBarrier,
]:
    root = tmp_path / "structural"
    source = _source_table()
    entities = _entities()
    shards = [
        _atomic_shard(
            root, "source_tables/part-00000.jsonl", [source]
        ),
        _atomic_shard(
            root, "entities/part-00000.jsonl", entities
        ),
        _atomic_shard(
            root,
            "page_refs/part-00000.jsonl",
            [
                {
                    "url_key": hashlib.sha256(
                        str(entity["page_url"]).encode()
                    ).hexdigest(),
                    "page_url": entity["page_url"],
                    "entity_id": entity["entity_id"],
                    "source_table_id": "source-1",
                    "row_id": row_id,
                }
                for row_id, entity in enumerate(entities)
            ],
        ),
        _atomic_shard(
            root, "direct_image_refs/part-00000.jsonl", []
        ),
        _atomic_shard(
            root, "structural_failures/part-00000.jsonl", []
        ),
        _atomic_shard(
            root,
            "selection/validated-00000.jsonl",
            [
                {
                    "source_table_id": "source-1",
                    "schema_class": "Thing",
                    "subset": "minimum3",
                    "host": "example.test",
                    "relative_path": "Thing/source-1.json.gz",
                    "rows": 2,
                    "columns": 3,
                    "selection_seed": 13,
                    "content_hash": "a" * 64,
                    "replacement_chain": [],
                    "replacement_reasons": [],
                    "replaces_path": None,
                    "replacement_reason": None,
                }
            ],
        ),
    ]
    manifest_path = root / "stage_manifests/structural-00000.json"
    fingerprint = StageFingerprint(
        stage="wdc200k_structural",
        input_fingerprint="selection-v1",
        parameter_fingerprint="structural-policy-v1",
        schema_version=STRUCTURAL_SCHEMA_VERSION,
    )
    manifest = StageManifest(manifest_path, fingerprint)
    for shard in shards:
        manifest.record_shard(shard)
    manifest.mark_complete()
    finalized = finalize_validated_selection(
        [manifest_path],
        output_root=root,
        target_tables=1,
    )
    manifest_payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    final_payload = json.loads(
        finalized.manifest.read_text(encoding="utf-8")
    )
    manifest_key = manifest_path.resolve().as_posix()
    final_shard = final_payload["completed_shards"][0]
    barrier = StructuralStageBarrier(
        schema_version=STRUCTURAL_SCHEMA_VERSION,
        manifest_count=1,
        manifest_sha256={
            manifest_key: hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        },
        input_fingerprints={
            manifest_key: manifest_payload["input_fingerprint"]
        },
        parameter_fingerprints={
            manifest_key: manifest_payload["parameter_fingerprint"]
        },
        final_manifest_sha256=hashlib.sha256(
            finalized.manifest.read_bytes()
        ).hexdigest(),
        final_selection=dict(final_shard),
    )
    return root, (manifest_path,), finalized.manifest, barrier


def _task5_fingerprint() -> dict[str, Any]:
    return {
        "input_fingerprint": "structural-page-image-v1",
        "schema_version": "wdc200k-asset-materialization-v1",
        "planning_manifest_sha256": "1" * 64,
        "unique_job_manifest_sha256": "2" * 64,
        "unique_job_sha256": "3" * 64,
        "image_fetch_manifest_sha256": "4" * 64,
        "image_policy_fingerprint": "image-policy-v1",
        "image_outcome_digest": "5" * 64,
        "image_outcome_count": 0,
        "image_outcome_url_key_digest": "6" * 64,
        "attempts_per_entity": 3,
        "retained_per_entity": 3,
        "text_asset_chunk_chars": 800,
        "min_text_asset_chunk_chars": 120,
        "max_text_asset_chunks_per_entity": 3,
        "records_per_shard": 100,
    }


def _asset_upstream(
    tmp_path: Path,
    *,
    assets: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
) -> tuple[Path, AssetStageBarrier]:
    root = tmp_path / "assets"
    assets = assets if assets is not None else []
    links = links if links is not None else [
        {**link, "asset_ids": []} for link in _links()
    ]
    asset_shard = _atomic_shard(
        root, "bridge_assets/part-00000.jsonl", assets
    )
    link_shard = _atomic_shard(
        root, "table_asset_links/part-00000.jsonl", links
    )
    fingerprint = _task5_fingerprint()
    manifest_path = root / "asset-materialization-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": "wdc200k_asset_materialization",
                "fingerprint": fingerprint,
                "bridge_asset_shards": [asdict(asset_shard)],
                "table_asset_link_shards": [asdict(link_shard)],
                "complete": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return (
        manifest_path,
        AssetStageBarrier(
            fingerprint=fingerprint,
            bridge_assets=len(assets),
            table_asset_links=len(links),
        ),
    )


def _model_upstream(
    tmp_path: Path,
    args: argparse.Namespace,
    tasks: list[dict[str, Any]] | None = None,
    extractor: Any = None,
) -> Any:
    store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        tasks or [],
        store,
        args=args,
        input_fingerprint="assets-v1",
    )
    return run_model_stage(
        store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "model-outputs",
    )


def _authoritative_inputs(
    tmp_path: Path,
) -> tuple[MaterializationInputs, argparse.Namespace]:
    args = _args(tmp_path)
    (
        structural_root,
        structural_manifests,
        final_manifest,
        structural_barrier,
    ) = _structural_upstream(tmp_path)
    assets_manifest, assets_barrier = _asset_upstream(tmp_path)
    model_result = _model_upstream(tmp_path, args)
    return (
        MaterializationInputs(
            structural_output_root=structural_root,
            structural_manifests=structural_manifests,
            finalized_selection_manifest=final_manifest,
            structural_barrier=structural_barrier,
            assets_manifest=assets_manifest,
            assets_barrier=assets_barrier,
            model_result=model_result,
            work_root=tmp_path / "work",
        ),
        args,
    )


class _StateExtractor:
    def extract(
        self,
        _asset: dict[str, Any],
        entity: dict[str, Any],
        _candidates: list[str],
    ) -> dict[str, Any]:
        value = {
            "entity-alpha": "Texas",
            "entity-beta": "Ohio",
        }[str(entity["entity_id"])]
        return {
            "attributes": [
                {
                    "name": "State",
                    "value": value,
                    "evidence": f"The page states {value}.",
                    "connection_evidence": "The entity name is visible.",
                }
            ],
            "raw_response": '{"attributes":[]}',
            "error": "",
        }


def _model_tasks(
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "extraction_task": {
                "source_table_id": "source-1",
                "source_row_id": (
                    0 if asset["entity_id"] == "entity-alpha" else 1
                ),
                "entity_column_index": 0,
                "entity_column_name": "Name",
                "entity": {
                    "entity_id": asset["entity_id"],
                    "wiki_title": asset["entity_wiki_title"],
                    "cell_text": (
                        "Alpha"
                        if asset["entity_id"] == "entity-alpha"
                        else "Beta"
                    ),
                    "entity_column_index": 0,
                    "entity_column_name": "Name",
                },
                "asset": asset,
                "candidate_attribute_names": ["State", "Category"],
            }
        }
        for asset in assets
    ]


def _shard_inputs(
    tmp_path: Path,
    *,
    entities: list[dict[str, Any]] | None = None,
    assets: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    extractions: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> MaterializationShardInputs:
    return MaterializationShardInputs(
        source_table=_source_table(),
        entity_paths=(
            _write_jsonl(
                tmp_path / "inputs" / "entities.jsonl",
                entities if entities is not None else _entities(),
            ),
        ),
        asset_paths=(
            _write_jsonl(
                tmp_path / "inputs" / "assets.jsonl",
                assets if assets is not None else [],
            ),
        ),
        link_paths=(
            _write_jsonl(
                tmp_path / "inputs" / "links.jsonl",
                links if links is not None else [],
            ),
        ),
        extraction_paths=(
            _write_jsonl(
                tmp_path / "inputs" / "extractions.jsonl",
                extractions if extractions is not None else [],
            ),
        ),
        error_paths=(
            _write_jsonl(
                tmp_path / "inputs" / "errors.jsonl",
                errors if errors is not None else [],
            ),
        ),
        lookup_database=tmp_path / "lookup.sqlite3",
    )


def test_materializer_matches_existing_query_builder_field_for_field(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    source_table = _source_table()
    entities = _entities()
    assets = _assets()
    links = _links()
    extractions = _extractions(args, assets)
    entity_to_assets = {
        str(link["entity_id"]): list(link["asset_ids"]) for link in links
    }
    wiki_to_entity_id = {
        str(entity["wiki_title"]): str(entity["entity_id"])
        for entity in entities
    }
    extraction_sink = _RecordSink()
    recovery_sink = _RecordSink()
    expected = join_builder.build_table_join_records(
        source_table=source_table,
        split="train",
        assets={str(asset["asset_id"]): asset for asset in assets},
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=None,
        cache=_MemoryCache(extractions),
        progress=None,
        concurrency_state=join_builder.ModelConcurrencyState.from_args(args),
        extraction_writer=extraction_sink,
        recovery_writer=recovery_sink,
        args=args,
    )

    actual = materialize_dataset_shard(
        _shard_inputs(
            tmp_path,
            entities=entities,
            assets=assets,
            links=links,
            extractions=extractions,
        ),
        args=args,
        split="train",
    )

    assert actual.query_tables == expected[0]
    assert actual.data_lake_tables == expected[1]
    assert actual.qrels == expected[2]
    assert actual.decision == expected[3]
    assert actual.attribute_extractions == extraction_sink.records
    assert actual.evidence_recoveries == recovery_sink.records
    assert len(actual.query_tables) == 1
    assert len(actual.evidence_recoveries) == 4


def test_empty_assets_keep_full_source_and_raw_data_lake_table(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    actual = materialize_dataset_shard(
        _shard_inputs(tmp_path),
        args=args,
        split="test",
    )

    assert actual.query_tables == []
    assert actual.qrels == []
    assert actual.attribute_extractions == []
    assert actual.evidence_recoveries == []
    assert actual.data_lake_tables[0]["role"] == "raw_data_lake_table"
    assert actual.data_lake_tables[0]["source_row_indices"] == [0, 1]
    assert len(actual.data_lake_tables[0]["rows"]) == 2
    assert actual.decision["reason"] == "no_column_met_recovered_value_ratio"


def test_terminal_model_error_is_consumed_without_a_model_retry(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    asset = _assets()[0]
    entity = _entities()[0]
    link = _links()[0]
    link["asset_ids"] = [asset["asset_id"]]
    error = _extractions(args, [asset])[0]
    error["attributes"] = []
    error["error"] = "terminal model failure"

    actual = materialize_dataset_shard(
        _shard_inputs(
            tmp_path,
            entities=[entity],
            assets=[asset],
            links=[link],
            errors=[error],
        ),
        args=args,
        split="dev",
    )

    assert actual.query_tables == []
    assert len(actual.attribute_extractions) == 1
    assert {
        key: actual.attribute_extractions[0][key] for key in error
    } == error
    assert actual.attribute_extractions[0]["source_table_id"] == "source-1"
    assert actual.attribute_extractions[0]["source_row_id"] == 0
    assert actual.data_lake_tables[0]["role"] == "raw_data_lake_table"


def test_duplicate_asset_id_with_different_payload_is_rejected(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    assets = _assets()[:2]
    conflicting = {**assets[0], "content": "different"}
    second_path = _write_jsonl(
        tmp_path / "inputs" / "assets-conflict.jsonl",
        [conflicting],
    )
    inputs = _shard_inputs(
        tmp_path,
        assets=assets,
    )
    inputs = MaterializationShardInputs(
        source_table=inputs.source_table,
        entity_paths=inputs.entity_paths,
        asset_paths=(*inputs.asset_paths, second_path),
        link_paths=inputs.link_paths,
        extraction_paths=inputs.extraction_paths,
        error_paths=inputs.error_paths,
        lookup_database=inputs.lookup_database,
    )

    with pytest.raises(ValueError, match="conflicting.*asset"):
        materialize_dataset_shard(inputs, args=args, split="train")


def test_irrelevant_records_are_disk_indexed_without_global_python_maps(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    relevant_assets = _assets()
    irrelevant_assets = [
        {
            **relevant_assets[0],
            "asset_id": f"irrelevant-{index:05d}",
            "entity_id": f"other-{index:05d}",
            "content": f"unrelated {index}",
        }
        for index in range(10_000)
    ]
    tracemalloc.start()
    try:
        actual = materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                assets=[*irrelevant_assets, *relevant_assets],
                links=_links(),
                extractions=_extractions(args, relevant_assets),
            ),
            args=args,
            split="train",
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(actual.bridge_assets) == len(relevant_assets)
    assert len(actual.query_tables) == 1
    assert peak < 24 * 1024 * 1024


def test_full_materialization_writes_current_canonical_layout_and_resumes(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"

    result = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=1,
    )

    expected_directories = {
        "source_tables",
        "query_tables",
        "data_lake_tables",
        "entities",
        "bridge_assets",
        "table_asset_links",
        "attribute_extractions",
        "evidence_recoveries",
    }
    expected_files = {
        "qrels.jsonl",
        "splits.json",
        "stats.json",
        "table_queryability_decisions.jsonl",
        "dataset_manifest.json",
        "media_download_failures.jsonl",
        "model_attribute_errors.jsonl",
        "web_fetch_failures.jsonl",
    }
    assert expected_directories <= {
        path.name for path in output_root.iterdir() if path.is_dir()
    }
    assert expected_files <= {
        path.name for path in output_root.iterdir() if path.is_file()
    }
    manifest = json.loads(
        (output_root / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["format"] == "sharded_jsonl"
    assert manifest["complete"] is True
    assert set(manifest["artifacts"]) == expected_directories
    assert all(
        "mtime_ns" in shard
        for artifact in manifest["artifacts"].values()
        for shard in artifact["shards"]
    )
    assert all(
        "mtime_ns" in item
        for item in manifest["published_single_files"].values()
    )
    assert result.stats["source_tables"] == 1
    assert result.stats["data_lake_tables"] == 1
    assert result.stats["query_tables"] == 0
    assert result.stats["table_asset_links"] == 0
    assert (
        manifest["artifacts"]["table_asset_links"]["total_records"]
        == 0
    )
    source_records = [
        json.loads(line)
        for path in (output_root / "source_tables").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    data_lake_records = [
        json.loads(line)
        for path in (output_root / "data_lake_tables").glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert source_records == [_source_table()]
    assert list(
        iter_manifest_records(output_root, "source_tables", log_every=0)
    ) == [_source_table()]
    assert load_split_map(output_root) == {"source-1": "test"}
    assert len(data_lake_records[0]["rows"]) == 2
    assert data_lake_records[0]["role"] == "raw_data_lake_table"

    published = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
    ]
    mtimes = {path: path.stat().st_mtime_ns for path in published}
    resumed = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=1,
    )

    assert resumed == result
    assert {path: path.stat().st_mtime_ns for path in published} == mtimes


def test_resume_revalidates_upstream_checksums_before_skipping(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"
    materialize_dataset(inputs, output_root=output_root, args=args)
    asset_payload = json.loads(
        inputs.assets_manifest.read_text(encoding="utf-8")
    )
    asset_path = (
        inputs.assets_manifest.parent
        / asset_payload["bridge_asset_shards"][0]["path"]
    )
    asset_path.write_text('{"forged":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Task-5.*validation"):
        materialize_dataset(inputs, output_root=output_root, args=args)


def test_forged_barriers_and_missing_model_manifest_are_rejected(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    forged_fingerprint = dict(inputs.assets_barrier.fingerprint)
    forged_fingerprint["planning_manifest_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="fingerprint"):
        materialize_dataset(
            replace(
                inputs,
                assets_barrier=AssetStageBarrier(
                    fingerprint=forged_fingerprint,
                    bridge_assets=inputs.assets_barrier.bridge_assets,
                    table_asset_links=(
                        inputs.assets_barrier.table_asset_links
                    ),
                ),
            ),
            output_root=tmp_path / "forged-output",
            args=args,
        )

    forged_error = _write_jsonl(
        tmp_path / "forged-model-errors.jsonl",
        [{"error": "caller-controlled payload"}],
    )
    derived_output = tmp_path / "derived-model-paths"
    materialize_dataset(
        replace(
            inputs,
            model_result=replace(
                inputs.model_result,
                extraction_paths=(forged_error,),
                error_paths=(forged_error,),
                success=999,
                terminal=999,
            ),
        ),
        output_root=derived_output,
        args=args,
    )
    assert (
        derived_output / "model_attribute_errors.jsonl"
    ).read_text(encoding="utf-8") == ""

    inputs.model_result.manifest_path.rename(
        inputs.model_result.manifest_path.with_suffix(".missing")
    )
    with pytest.raises(ValueError, match="Task-6"):
        materialize_dataset(
            inputs,
            output_root=tmp_path / "missing-model-output",
            args=args,
        )


def test_output_manifest_corruption_is_not_silently_rewritten(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"
    materialize_dataset(inputs, output_root=output_root, args=args)
    source_path = next((output_root / "source_tables").glob("*.jsonl"))
    source_path.write_text('{"corrupt":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="published.*validation"):
        materialize_dataset(inputs, output_root=output_root, args=args)


def test_interruption_after_table_commit_resumes_without_reprocessing(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    committed: list[str] = []

    def interrupt(source_table_id: str) -> None:
        committed.append(source_table_id)
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_dataset(
            inputs,
            output_root=tmp_path / "output",
            args=args,
            after_table_commit=interrupt,
        )
    assert committed == ["source-1"]
    assert not (tmp_path / "output" / "dataset_manifest.json").exists()

    resumed = materialize_dataset(
        inputs,
        output_root=tmp_path / "output",
        args=args,
        after_table_commit=interrupt,
    )

    assert resumed.complete is True
    assert committed == ["source-1"]


def test_nonempty_task6_outputs_materialize_query_qrel_and_evidence(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    (
        structural_root,
        structural_manifests,
        final_manifest,
        structural_barrier,
    ) = _structural_upstream(tmp_path)
    assets = _assets()
    assets_manifest, assets_barrier = _asset_upstream(
        tmp_path,
        assets=assets,
        links=_links(),
    )
    model_result = _model_upstream(
        tmp_path,
        args,
        tasks=_model_tasks(assets),
        extractor=_StateExtractor(),
    )
    inputs = MaterializationInputs(
        structural_output_root=structural_root,
        structural_manifests=structural_manifests,
        finalized_selection_manifest=final_manifest,
        structural_barrier=structural_barrier,
        assets_manifest=assets_manifest,
        assets_barrier=assets_barrier,
        model_result=model_result,
        work_root=tmp_path / "work",
    )
    output_root = tmp_path / "output"

    result = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=2,
    )

    assert result.stats["queryable_source_tables"] == 1
    assert result.stats["query_tables"] == 1
    assert result.stats["qrels"] == 1
    assert result.stats["attribute_extractions"] == 4
    assert result.stats["evidence_recoveries"] == 4
    query = list(
        iter_manifest_records(output_root, "query_tables", log_every=0)
    )
    qrels = [
        json.loads(line)
        for line in (output_root / "qrels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    recoveries = list(
        iter_manifest_records(
            output_root, "evidence_recoveries", log_every=0
        )
    )
    assert len(query) == len(qrels) == 1
    assert qrels[0]["query_table_id"] == query[0]["table_id"]
    assert all(
        recovery["source_table_id"] == "source-1"
        and recovery["evidence"]["asset_type"] == "text"
        for recovery in recoveries
    )


def test_partial_source_catalog_import_resumes_after_committed_batch(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "catalog.sqlite3"
    source_path = tmp_path / "sources.jsonl"
    args = _args(tmp_path)
    records = [
        _source_table(f"source-{index:02d}") for index in range(11)
    ]
    source_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in records[:10]
        )
        + "{broken-json\n",
        encoding="utf-8",
    )
    materializer._initialize_index(database_path)

    with pytest.raises(ValueError, match="invalid JSONL"):
        materializer._catalog_sources(
            database_path,
            [source_path],
            args=args,
            expected_tables=11,
        )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_catalog"
        ).fetchone() == (10,)

    _write_jsonl(source_path, records)
    materializer._catalog_sources(
        database_path,
        [source_path],
        args=args,
        expected_tables=11,
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_catalog"
        ).fetchone() == (11,)

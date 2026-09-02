from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import threading
import tracemalloc
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts_old"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mm_joinability_dataset as join_builder
import wdc200k_materialize as materializer
from wdc200k_assets import (
    ImageBudget,
    asset_materialization_input_fingerprint,
    asset_planning_input_fingerprint,
    build_unique_image_jobs,
    fetch_unique_images,
    iter_entity_page_join,
    materialize_asset_shards,
    persist_entity_asset_plans,
    structural_asset_input_identity,
    validate_materialized_asset_shards,
)
from wdc200k_fetch import (
    FetchPolicy,
    fetch_unique_pages,
    iter_page_fanout,
    validate_complete_page_fetch,
)
from wdc200k_materialize import (
    MaterializationInputs,
    MaterializationShardInputs,
    load_certified_materialization_inputs,
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
    ModelStageAuthority,
    StructuralStageBarrier,
    adapt_model_tasks_from_manifests,
    enqueue_model_tasks,
    run_model_stage,
)
from wdc200k_structural import (
    STRUCTURAL_SCHEMA_VERSION,
    expand_selected_shard,
    finalize_validated_selection,
)
from stage1_io import (
    iter_manifest_records,
    load_split_map,
    resolve_source_table_reference,
)
from stage1_io import stable_hash


class _MemoryCache:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.items = {
            str(record["cache_key"]): dict(record) for record in records
        }
        self.transient_items: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        record = self.items.get(key)
        return dict(record) if record is not None else None

    def put(self, key: str, record: dict[str, Any]) -> None:
        self.items[key] = dict(record)

    def get_transient(self, key: str) -> dict[str, Any] | None:
        record = self.transient_items.get(key)
        return dict(record) if record is not None else None

    def put_transient(self, key: str, record: dict[str, Any]) -> None:
        self.transient_items[key] = dict(record)


def test_materialize_schema_initialization_commit_uses_live_guard(
    tmp_path: Path,
) -> None:
    path = tmp_path / "materialize.sqlite3"
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("materialize schema reserve exhausted")

    with pytest.raises(OSError, match="materialize schema reserve"):
        materializer._initialize_index(
            path,
            pre_write_guard=reject_commit,
        )

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name = 'entities'"
        ).fetchone() == (0,)


class _RecordSink:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def test_unsampled_source_rows_receive_stable_derived_entity_identity(
    tmp_path: Path,
) -> None:
    source = _source_table()
    sampled = _entities()[:1]
    entity_path = tmp_path / "sampled_entities.jsonl"
    writer = AtomicJsonlShard(entity_path)
    writer.write(sampled[0])
    writer.commit()
    database = tmp_path / "index.sqlite3"
    materializer._initialize_index(database)
    with materializer._connect(database) as connection:
        materializer._index_entities(connection, [entity_path])
        connection.commit()

    entities, _assets, _links, _extractions, wiki_to_entity = (
        materializer._table_inputs(database, source)
    )

    unsampled_title = source["rows"][1]["cells"][0]["wiki_title"]
    assert len(entities) == 1
    assert wiki_to_entity[unsampled_title] == (
        "ent_" + stable_hash(unsampled_title, length=16)
    )


def test_batched_table_inputs_match_single_table_reads(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    args = _args(tmp_path)
    sources = []
    for source_index in range(2):
        source = _source_table(f"source-{source_index}")
        for row in source["rows"]:
            cell = row["cells"][0]
            cell["wiki_title"] = (
                f"{cell['wiki_title']}_{source_index}"
            )
        sources.append(source)

    materializer._initialize_index(database)
    materializer._catalog_source_records(
        database,
        sources,
        args=args,
        expected_tables=len(sources),
    )
    with materializer._connect(database) as connection:
        for source_index, source in enumerate(sources):
            source_id = str(source["source_table_id"])
            for row in source["rows"]:
                row_id = int(row["row_id"])
                wiki_title = str(row["cells"][0]["wiki_title"])
                entity_id = f"entity-{source_index}-{row_id}"
                asset_id = f"asset-{source_index}-{row_id}"
                link_id = f"link-{source_index}-{row_id}"
                cache_key = f"cache-{source_index}-{row_id}"
                entity = {
                    "entity_id": entity_id,
                    "wiki_title": wiki_title,
                }
                asset = {
                    "asset_id": asset_id,
                    "entity_id": entity_id,
                    "asset_type": "text",
                    "content": f"content-{source_index}-{row_id}",
                }
                link = {
                    "link_id": link_id,
                    "source_table_id": source_id,
                    "row_id": row_id,
                    "entity_id": entity_id,
                    "asset_ids": [asset_id],
                }
                extraction = {
                    "cache_key": cache_key,
                    "entity_id": entity_id,
                    "asset_id": asset_id,
                    "attributes": [],
                }
                connection.execute(
                    """
                    INSERT INTO entities (
                        entity_id, source_table_id, source_row_id,
                        record_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        source_id,
                        row_id,
                        materializer._canonical_json(entity),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO assets (asset_id, entity_id, record_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        asset_id,
                        entity_id,
                        materializer._canonical_json(asset),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO links (
                        link_id, source_table_id, source_row_id,
                        entity_id, record_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        link_id,
                        source_id,
                        row_id,
                        entity_id,
                        materializer._canonical_json(link),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO extractions (
                        cache_key, model_call_key, job_id,
                        entity_id, asset_id, source_table_id,
                        source_row_id, status, record_json
                    ) VALUES (?, '', '', ?, ?, ?, ?, 'success', ?)
                    """,
                    (
                        cache_key,
                        entity_id,
                        asset_id,
                        source_id,
                        row_id,
                        materializer._canonical_json(extraction),
                    ),
                )
                for alias in {
                    wiki_title,
                    materializer.normalize_title(wiki_title),
                    wiki_title.casefold(),
                    materializer.normalize_title(wiki_title).casefold(),
                }:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO entity_aliases (
                            alias, entity_id
                        ) VALUES (?, ?)
                        """,
                        (alias, entity_id),
                    )
        rows = {
            str(row["source_table_id"]): row
            for row in connection.execute(
                """
                SELECT source_table_id, ordinal, record_sha256
                FROM source_catalog
                """
            )
        }
        connection.commit()

    items = [
        materializer._MaterializationWorkItem(
            source_table_id=source_id,
            source_ordinal=int(rows[source_id]["ordinal"]),
            source_sha256=str(rows[source_id]["record_sha256"]),
            split="train",
        )
        for source_id in ("source-1", "source-0")
    ]
    loaded = materializer._load_materialization_sources(database, items)

    assert [source["source_table_id"] for source in loaded] == [
        "source-1",
        "source-0",
    ]
    assert materializer._table_inputs_batch(database, loaded) == [
        materializer._table_inputs(database, source) for source in loaded
    ]


def test_extraction_lookup_has_asset_leading_index(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    materializer._initialize_index(database)

    with materializer._connect(database) as connection:
        plan = [
            str(row["detail"])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT cache_key, record_json
                FROM extractions
                WHERE asset_id = ?
                ORDER BY cache_key
                """,
                ("asset-1",),
            )
        ]

    assert any("extractions_asset" in detail for detail in plan)


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
        {
            "column_index": 3,
            "column_name": "Type",
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
                    {
                        "column_index": 3,
                        "column_name": "Type",
                        "raw": "Location",
                        "text": "Location",
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


def _multi_attribute_source_table(
    context_column_names: list[str],
) -> dict[str, Any]:
    column_names = [
        "Name",
        "Bridge B",
        "Bridge C",
        *context_column_names,
    ]
    columns = [
        {
            "column_index": index,
            "column_name": column_name,
            "is_numeric_column": False,
        }
        for index, column_name in enumerate(column_names)
    ]
    rows = []
    for row_id, name in enumerate(("Alpha", "Beta")):
        rows.append(
            {
                "row_id": row_id,
                "cells": [
                    {
                        "column_index": index,
                        "column_name": column_name,
                        "raw": (
                            name
                            if index == 0
                            else f"{column_name} value {name}"
                        ),
                        "text": (
                            name
                            if index == 0
                            else f"{column_name} value {name}"
                        ),
                        "wiki_title": (
                            f"wdc_{name.casefold()}"
                            if index == 0
                            else None
                        ),
                        "has_wiki_link": index == 0,
                    }
                    for index, column_name in enumerate(column_names)
                ],
            }
        )
    return {
        "source_table_id": "source-1",
        "source_file": "Thing/source-1.json.gz",
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


def _row_attributes_for_asset(
    source_table: dict[str, Any],
    asset: dict[str, Any],
) -> list[dict[str, Any]]:
    entity_col = join_builder.choose_entity_column(
        source_table,
        min_linked_rows=1,
    )
    assert entity_col is not None
    wiki_title = str(asset["entity_wiki_title"])
    source_row = next(
        row
        for row in source_table["rows"]
        if str(join_builder.get_cell(row, entity_col).get("wiki_title"))
        == wiki_title
    )
    return join_builder.extraction_row_attributes(
        source_table,
        source_row,
        entity_col,
    )


def _extractions(
    args: argparse.Namespace,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_table = _source_table()
    state_by_entity = {
        "entity-alpha": "Texas",
        "entity-beta": "Ohio",
    }
    records = []
    for asset in assets:
        entity_id = str(asset["entity_id"])
        row_attributes = _row_attributes_for_asset(source_table, asset)
        cache_key = join_builder.extraction_cache_key(
            asset_id=str(asset["asset_id"]),
            entity_id=entity_id,
            candidate_attribute_names=["State", "Category", "Type"],
            asset_type="text",
            args=args,
            row_attributes=row_attributes,
        )
        records.append(
            {
                "cache_key": cache_key,
                "entity_id": entity_id,
                "entity_text": entity_id.removeprefix("entity-").title(),
                "entity_wiki_title": str(asset["entity_wiki_title"]),
                "asset_id": asset["asset_id"],
                "asset_type": "text",
                "candidate_attribute_names": ["State", "Category", "Type"],
                "row_attributes": row_attributes,
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


def _multi_attribute_extractions(
    args: argparse.Namespace,
    assets: list[dict[str, Any]],
    context_column_names: list[str],
) -> list[dict[str, Any]]:
    source_table = _multi_attribute_source_table(context_column_names)
    candidate_attribute_names = [
        "Bridge B",
        "Bridge C",
        *context_column_names,
    ]
    records = []
    for asset in assets:
        entity_id = str(asset["entity_id"])
        entity_name = entity_id.removeprefix("entity-").title()
        row_attributes = _row_attributes_for_asset(source_table, asset)
        cache_key = join_builder.extraction_cache_key(
            asset_id=str(asset["asset_id"]),
            entity_id=entity_id,
            candidate_attribute_names=candidate_attribute_names,
            asset_type="text",
            args=args,
            row_attributes=row_attributes,
        )
        records.append(
            {
                "cache_key": cache_key,
                "entity_id": entity_id,
                "entity_text": entity_name,
                "entity_wiki_title": str(asset["entity_wiki_title"]),
                "asset_id": asset["asset_id"],
                "asset_type": "text",
                "candidate_attribute_names": candidate_attribute_names,
                "row_attributes": row_attributes,
                "attributes": [
                    {
                        "name": column_name,
                        "value": f"{column_name} value {entity_name}",
                        "evidence": "page statement",
                        "connection_evidence": "same named entity",
                    }
                    for column_name in ("Bridge B", "Bridge C")
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


def _real_structural_upstream(
    tmp_path: Path,
) -> tuple[
    Path,
    tuple[Path, ...],
    Path,
    StructuralStageBarrier,
]:
    input_root = tmp_path / "raw"
    table_path = (
        input_root
        / "Thing"
        / "Thing_example.test_October2023.json.gz"
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": name,
            "State": state,
            "Category": "Place",
            "Type": "Location",
            "page_url": f"https://example.test/{name.casefold()}",
            "image": "",
        }
        for name, state in (("Alpha", "Texas"), ("Beta", "Ohio"))
    ]
    with gzip.open(table_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    selection = {
        "schema_class": "Thing",
        "subset": "minimum3",
        "host": "example.test",
        "relative_path": table_path.relative_to(input_root).as_posix(),
        "rows": 2,
        "columns": 6,
        "rank": "rank",
        "selection_seed": 13,
    }
    root = tmp_path / "structural"
    expanded = expand_selected_shard(
        [selection],
        output_root=root,
        input_root=input_root,
        min_rows=2,
        min_cols=3,
    )
    manifest_path = expanded.manifest
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


class _PageTransport:
    network_policy_fingerprint = "test-page-network-v1"

    def __init__(
        self,
        outcomes: dict[str, dict[str, Any] | BaseException],
    ) -> None:
        self.outcomes = outcomes

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict[str, Any]:
        assert deadline_seconds > 0
        assert max_retries == 0
        outcome = self.outcomes[url]
        if isinstance(outcome, BaseException):
            raise outcome
        return {
            "page_url": url,
            "final_url": url,
            "text": str(outcome.get("text") or ""),
            "image_urls": list(outcome.get("image_urls") or []),
        }


class _ImageTransport:
    network_policy_fingerprint = "test-image-network-v1"
    max_retries = 0
    max_response_seconds = 30.0

    def download_image(self, *_args: Any, **_kwargs: Any) -> None:
        raise TimeoutError("image download timed out")


def _authoritative_inputs(
    tmp_path: Path,
    *,
    page_success: bool = False,
    extractor: Any = None,
    page_image_url: str | None = None,
    real_structural: bool = False,
) -> tuple[MaterializationInputs, argparse.Namespace]:
    args = _args(tmp_path)
    (
        structural_root,
        structural_manifests,
        final_manifest,
        structural_barrier,
    ) = (
        _real_structural_upstream(tmp_path)
        if real_structural
        else _structural_upstream(tmp_path)
    )
    structural_payload = json.loads(
        structural_manifests[0].read_text(encoding="utf-8")
    )
    completed = [
        _completed_from_payload
        for _completed_from_payload in structural_payload[
            "completed_shards"
        ]
    ]
    entity_paths = tuple(
        structural_root / item["path"]
        for item in completed
        if item["path"].startswith("entities/")
    )
    page_ref_paths = tuple(
        structural_root / item["path"]
        for item in completed
        if item["path"].startswith("page_refs/")
    )
    page_refs = [
        json.loads(line)
        for path in page_ref_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    page_outcomes: dict[str, dict[str, Any] | BaseException] = {}
    for reference in page_refs:
        page_url = str(reference["page_url"])
        if page_success:
            entity_name = page_url.rsplit("/", 1)[-1].title()
            page_outcomes[page_url] = {
                "text": (
                    f"{entity_name} has a State value recorded on this "
                    "page. "
                )
                * 20,
                "image_urls": (
                    [page_image_url] if page_image_url else []
                ),
            }
        else:
            page_outcomes[page_url] = TimeoutError("page timed out")
    page_result = fetch_unique_pages(
        page_refs,
        SqliteJobStore(tmp_path / "page-jobs.sqlite3"),
        _PageTransport(page_outcomes),
        FetchPolicy(
            retries=0,
            network_policy_fingerprint="test-page-network-v1",
        ),
    )
    page_snapshot = validate_complete_page_fetch(
        page_result,
        page_refs,
        validation_database=tmp_path / "page-validation.sqlite3",
    )
    structural_identity = structural_asset_input_identity(
        (
            structural_barrier.manifest_sha256[
                path.resolve().as_posix()
            ]
            for path in sorted(structural_manifests)
        ),
        structural_barrier.final_manifest_sha256,
    )
    planning_input = asset_planning_input_fingerprint(
        structural_identity,
        str(page_snapshot["identity"]),
    )
    planned = persist_entity_asset_plans(
        iter_entity_page_join(
            entity_paths,
            iter_page_fanout(
                page_result.outcomes_path,
                page_result.policy_fingerprint,
            ),
            join_path=tmp_path / "entity-page-join.sqlite3",
        ),
        output_root=tmp_path / "asset-plans",
        input_fingerprint=planning_input,
        budget=ImageBudget(3, 3),
    )
    unique_jobs = build_unique_image_jobs(
        planned,
        tmp_path / "unique-images.jsonl",
    )
    image_result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(tmp_path / "image-jobs.sqlite3"),
        _ImageTransport(),
        FetchPolicy(
            retries=0,
            deadline_seconds=30.0,
            network_policy_fingerprint="test-image-network-v1",
            policy_version="wdc200k-image-fetch-v1",
        ),
        outcomes_path=tmp_path / "image-outcomes.sqlite3",
        image_dir=tmp_path / "image-content",
    )
    asset_input = asset_materialization_input_fingerprint(
        planned.manifest_path,
        image_result.fetch_manifest_path,
    )
    materialized_assets = materialize_asset_shards(
        planned,
        fetch_result=image_result,
        output_root=tmp_path / "assets",
        input_fingerprint=asset_input,
    )
    _validated_assets, assets_barrier = (
        validate_materialized_asset_shards(
            materialized_assets,
            planned=planned,
            image_fetch_result=image_result,
            expected_input_fingerprint=asset_input,
        )
    )
    adapted = adapt_model_tasks_from_manifests(
        structural_output_root=structural_root,
        structural_manifests=structural_manifests,
        finalized_selection_manifest=final_manifest,
        structural_barrier=structural_barrier,
        assets_manifest=materialized_assets.manifest_path,
        assets_barrier=assets_barrier,
        output_root=tmp_path / "adapted-model-tasks",
        args=args,
    )
    task_records = (
        json.loads(line)
        for path in adapted.task_paths
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    model_store = SqliteJobStore(tmp_path / "models.sqlite3")
    jobset = enqueue_model_tasks(
        task_records,
        model_store,
        args=args,
        input_fingerprint=adapted.input_fingerprint,
        text_input_fingerprint=adapted.input_fingerprint,
        image_input_fingerprint=adapted.input_fingerprint,
    )
    model_result = run_model_stage(
        model_store,
        extractor,
        jobset=jobset,
        output_root=tmp_path / "model-outputs",
    )
    return (
        MaterializationInputs(
            structural_output_root=structural_root,
            structural_manifests=structural_manifests,
            finalized_selection_manifest=final_manifest,
            structural_barrier=structural_barrier,
            page_fetch_result=page_result,
            asset_plan_result=planned,
            unique_image_jobs=unique_jobs,
            image_fetch_result=image_result,
            materialized_assets=materialized_assets,
            adapted_model_tasks=adapted,
            model_result=model_result,
            model_authority=ModelStageAuthority.current(args),
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


class _FailingExtractor:
    def extract(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("terminal model failure")


class _GenericExtractor:
    def extract(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "attributes": [
                {
                    "name": "State",
                    "value": "Texas",
                    "evidence": "The page states Texas.",
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
                "candidate_attribute_names": ["State", "Category", "Type"],
            }
        }
        for asset in assets
    ]


def _shard_inputs(
    tmp_path: Path,
    *,
    source_table: dict[str, Any] | None = None,
    entities: list[dict[str, Any]] | None = None,
    assets: list[dict[str, Any]] | None = None,
    links: list[dict[str, Any]] | None = None,
    extractions: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> MaterializationShardInputs:
    return MaterializationShardInputs(
        source_table=(
            source_table
            if source_table is not None
            else _source_table()
        ),
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


def test_wdc_materializer_merges_query_with_multiple_positive_targets(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    context_columns = ["Query X", "Query Y", "Target Z"]
    source_table = _multi_attribute_source_table(context_columns)
    assets = _assets()

    actual = materialize_dataset_shard(
        _shard_inputs(
            tmp_path,
            source_table=source_table,
            entities=_entities(),
            assets=assets,
            links=_links(),
            extractions=_multi_attribute_extractions(
                args,
                assets,
                context_columns,
            ),
        ),
        args=args,
        split="train",
    )

    assert actual.decision["reason"] == "queryable"
    assert len(actual.query_tables) == 1
    assert len(actual.data_lake_tables) == 2
    assert len(actual.qrels) == 2
    assert {
        qrel["query_table_id"] for qrel in actual.qrels
    } == {
        actual.query_tables[0]["table_id"]
    }
    assert set(actual.query_tables[0]["target_table_ids"]) == {
        target["table_id"] for target in actual.data_lake_tables
    }
    assert {
        target["join_col_name"] for target in actual.data_lake_tables
    } == {"Bridge B", "Bridge C"}
    assert all(
        not ({1, 2} & set(query["source_column_indices"]))
        for query in actual.query_tables
    )
    assert all(
        not (
            set(query["source_column_indices"])
            & set(target["source_column_indices"])
        )
        for query in actual.query_tables
        for target in actual.data_lake_tables
    )


def test_wdc_global_validation_keeps_distinct_qrels_for_merged_query(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    context_columns = ["Query X", "Target Z"]
    source_table = _multi_attribute_source_table(context_columns)
    assets = _assets()
    inputs = _shard_inputs(
        tmp_path,
        source_table=source_table,
        entities=_entities(),
        assets=assets,
        links=_links(),
        extractions=_multi_attribute_extractions(
            args,
            assets,
            context_columns,
        ),
    )
    actual = materialize_dataset_shard(
        inputs,
        args=args,
        split="train",
    )

    assert len(actual.query_tables) == 1
    assert len(actual.data_lake_tables) == 2
    assert len(actual.qrels) == 2
    assert {qrel["query_table_id"] for qrel in actual.qrels} == {
        actual.query_tables[0]["table_id"]
    }
    assert materializer._store_table_unit(
        inputs.lookup_database,
        actual,
        source_ordinal=0,
        source_sha256="source-sha",
        split="train",
    )
    counts = materializer._validate_global_counts(
        inputs.lookup_database,
        argparse.Namespace(
            expected_tables=1,
            expected_entities=len(actual.entities),
            expected_assets=len(actual.bridge_assets),
            expected_links=len(actual.table_asset_links),
            expected_extractions=len(actual.attribute_extractions),
        ),
    )

    assert counts["query_tables"] == 1
    assert counts["qrels"] == 2


def test_public_shard_materializer_guards_actual_index_and_cleans_validation(
    tmp_path: Path,
) -> None:
    inputs = _shard_inputs(tmp_path)
    calls: list[tuple[Path, int]] = []

    materialize_dataset_shard(
        inputs,
        args=_args(tmp_path),
        split="train",
        pre_write_guard=lambda path, size=0: calls.append(
            (Path(path), size)
        ),
    )

    assert calls
    validation_root = (
        inputs.lookup_database.parent
        / ".source-closure-validation"
    )
    assert all(
        path == inputs.lookup_database
        or path == validation_root
        or validation_root in path.parents
        for path, _size in calls
    )
    assert any(
        path == inputs.lookup_database for path, _size in calls
    )
    assert any(
        path == validation_root or validation_root in path.parents
        for path, _size in calls
    )
    assert any(size >= 64 * 1024 * 1024 for _path, size in calls)
    with sqlite3.connect(inputs.lookup_database) as connection:
        validation_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'source_row_validation'
            """
        ).fetchone()
    assert validation_table is None
    assert list(validation_root.iterdir()) == []


def test_source_catalog_closure_failure_does_not_pollute_durable_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "materialization.sqlite3"
    materializer._initialize_index(database_path)
    source_record = {
        "source_table_id": "source-1",
        "rows": [{"row_id": 0}],
    }
    with materializer._connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO source_catalog (
                source_table_id, ordinal, page_title, split_group,
                split, record_sha256, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "source-1",
                0,
                "Source",
                "source-1",
                "train",
                "source-sha",
                json.dumps(source_record),
            ),
        )
        connection.execute(
            """
            INSERT INTO entity_sources (
                entity_id, source_table_id, source_row_id
            ) VALUES (?, ?, ?)
            """,
            ("entity-missing-row", "source-1", 99),
        )
    calls: list[tuple[Path, int]] = []
    tracker = materializer.GuardedWriteTracker(
        database_path,
        lambda path, size=0: calls.append((Path(path), size)),
    )
    calls.clear()

    with materializer._connect(database_path) as connection:
        with pytest.raises(ValueError, match="source catalog relation"):
            materializer._validate_source_catalog_closure(
                connection,
                write_tracker=tracker,
            )

    validation_root = (
        database_path.parent / ".source-closure-validation"
    )
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'source_row_validation'
            """
        ).fetchone() == (0,)
    assert calls
    assert all(
        path == validation_root or validation_root in path.parents
        for path, _size in calls
    )
    assert list(validation_root.iterdir()) == []


def test_table_unit_commit_guard_failure_rolls_back_and_resumes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "materialization.sqlite3"
    materializer._initialize_index(database_path)
    table = materializer.MaterializedTable(
        source_table={"source_table_id": "source-1"},
        entities=[],
        bridge_assets=[],
        table_asset_links=[],
        query_tables=[],
        data_lake_tables=[],
        qrels=[],
        decision={},
        attribute_extractions=[],
        evidence_recoveries=[],
    )
    zero_checks = 0

    def reject_commit(_path: Path, estimated_bytes: int = 0) -> None:
        nonlocal zero_checks
        if estimated_bytes == 0:
            zero_checks += 1
            if zero_checks == 2:
                raise OSError("table unit commit reserve exhausted")

    tracker = materializer.GuardedWriteTracker(
        database_path,
        reject_commit,
        interval_bytes=1,
    )
    tracker.before_write(4096)
    with pytest.raises(OSError, match="table unit commit reserve"):
        materializer._store_table_unit(
            database_path,
            table,
            source_ordinal=0,
            source_sha256="source-sha",
            split="train",
            write_tracker=tracker,
        )

    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_units"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM materialized_records"
        ).fetchone() == (0,)
    assert materializer._store_table_unit(
        database_path,
        table,
        source_ordinal=0,
        source_sha256="source-sha",
        split="train",
    )


def test_failure_deduplication_recovers_after_mid_batch_guard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "materialization.sqlite3"
    records = [
        {"entity_id": f"entity-{index}", "error": "failed"}
        for index in range(24)
    ]
    positive_calls = 0
    monkeypatch.setattr(
        materializer.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    def interrupt(path: Path, estimated_bytes: int = 0) -> None:
        nonlocal positive_calls
        assert Path(path) == database_path
        if estimated_bytes > 0:
            positive_calls += 1
            if positive_calls == 2:
                raise RuntimeError("synthetic diagnostic reserve exhausted")

    with pytest.raises(RuntimeError, match="diagnostic reserve"):
        list(
            materializer._deduplicated_failures(
                database_path,
                kind="page",
                records=records,
                pre_write_guard=interrupt,
            )
        )

    resumed = list(
        materializer._deduplicated_failures(
            database_path,
            kind="page",
            records=records,
        )
    )
    assert positive_calls == 2
    assert resumed == records


def test_empty_assets_keep_source_reference_in_raw_data_lake_table(
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
    assert actual.data_lake_tables[0]["source_table_ref"] == {
        "artifact": "source_tables",
        "source_table_id": "source-1",
    }
    assert "rows" not in actual.data_lake_tables[0]
    assert "columns" not in actual.data_lake_tables[0]
    columns = [
        int(column["column_index"])
        for column in actual.source_table["columns"]
    ]
    legacy_rows, legacy_source_rows = (
        join_builder.project_selected_rows(
            actual.source_table,
            columns,
            {0, 1},
            min_required_cols=0,
        )
    )
    legacy_record = join_builder.table_record(
        table_id="dl_raw_source-1",
        role="raw_data_lake_table",
        split="test",
        source_table=actual.source_table,
        column_indices=columns,
        rows=legacy_rows,
        source_row_indices=legacy_source_rows,
        extra={
            "queryable": False,
            "reason": "no_column_met_recovered_value_ratio",
        },
    )
    assert resolve_source_table_reference(
        actual.data_lake_tables[0],
        actual.source_table,
    ) == legacy_record
    assert actual.decision["reason"] == "no_column_met_recovered_value_ratio"


def test_empty_assets_can_materialize_visible_join_fallback(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.explicit_join_fallback_ratio = 1.0

    actual = materialize_dataset_shard(
        _shard_inputs(tmp_path),
        args=args,
        split="test",
    )

    assert actual.decision["reason"] == "explicit_join_fallback"
    assert len(actual.query_tables) == len(actual.data_lake_tables) == 1
    assert len(actual.qrels) == 1
    query = actual.query_tables[0]
    target = actual.data_lake_tables[0]
    join_col = actual.decision["join_column_index"]
    assert join_col != query["query_entity_col"]
    assert join_col in query["source_column_indices"]
    assert join_col in target["source_column_indices"]
    assert query["hidden_attributes"] == []
    assert target["role"] == "target_data_lake_table"
    assert actual.evidence_recoveries == []


def test_materialization_index_balances_explicit_queries_and_resumes(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.explicit_join_fallback_mode = "match_implicit"
    implicit_source = _source_table("source-implicit")
    candidate_source = _source_table("source-candidate")
    candidate = materialize_dataset_shard(
        _shard_inputs(
            tmp_path,
            source_table=candidate_source,
            entities=[],
            assets=[],
            links=[],
            extractions=[],
        ),
        args=args,
        split="test",
    )
    assert candidate.decision.get("explicit_join_candidate")

    implicit = materializer.MaterializedTable(
        source_table=implicit_source,
        entities=[],
        bridge_assets=[],
        table_asset_links=[],
        query_tables=[
            {
                "table_id": "query-implicit",
                "source_table_id": "source-implicit",
                "split": "test",
            },
            {
                "table_id": "query-implicit-2",
                "source_table_id": "source-implicit",
                "split": "test",
            },
        ],
        data_lake_tables=[
            {
                "table_id": "target-implicit",
                "source_table_id": "source-implicit",
                "split": "test",
            },
            {
                "table_id": "target-implicit-2",
                "source_table_id": "source-implicit",
                "split": "test",
            },
        ],
        qrels=[
            {
                "query_table_id": "query-implicit",
                "target_table_id": "target-implicit",
                "source_table_id": "source-implicit",
                "split": "test",
            },
            {
                "query_table_id": "query-implicit-2",
                "target_table_id": "target-implicit-2",
                "source_table_id": "source-implicit",
                "split": "test",
            },
        ],
        decision={"reason": "queryable"},
        attribute_extractions=[],
        evidence_recoveries=[],
    )
    database_path = tmp_path / "balance.sqlite3"
    materializer._initialize_index(database_path)
    materializer._catalog_source_records(
        database_path,
        [implicit_source, candidate_source],
        args=args,
        expected_tables=2,
    )
    with materializer._connect(database_path) as connection:
        connection.execute("UPDATE source_catalog SET split = 'test'")
        catalog = {
            str(row["source_table_id"]): (
                int(row["ordinal"]),
                str(row["record_sha256"]),
            )
            for row in connection.execute(
                """
                SELECT source_table_id, ordinal, record_sha256
                FROM source_catalog
                """
            )
        }
        connection.commit()
    for materialized in (implicit, candidate):
        source_table_id = str(
            materialized.source_table["source_table_id"]
        )
        ordinal, source_sha256 = catalog[source_table_id]
        assert materializer._store_table_unit(
            database_path,
            materialized,
            source_ordinal=ordinal,
            source_sha256=source_sha256,
            split="test",
        )

    events: list[dict[str, object]] = []
    payload = materializer._balance_explicit_join_records(
        database_path,
        args=args,
        progress_callback=events.append,
    )

    assert payload == {
        "mode": "match_implicit",
        "implicit_query_tables_by_split": {
            "train": 0,
            "dev": 0,
            "test": 2,
        },
        "explicit_query_tables_by_split": {
            "train": 0,
            "dev": 0,
            "test": 2,
        },
        "candidate_tables_by_split": {
            "train": 0,
            "dev": 0,
            "test": 1,
        },
    }
    assert len(
        list(materializer._iter_materialized(database_path, "query_tables"))
    ) == 4
    assert len(
        list(
            materializer._iter_materialized(
                database_path,
                "data_lake_tables",
            )
        )
    ) == 4
    decisions = list(
        materializer._iter_materialized(
            database_path,
            "table_queryability_decisions",
        )
    )
    assert {decision["reason"] for decision in decisions} == {
        "queryable",
        "explicit_join_fallback",
    }
    assert materializer._balance_explicit_join_records(
        database_path,
        args=args,
    ) == payload
    assert {
        str(event["subphase"])
        for event in events
        if event.get("phase") == "balance_explicit_joins"
    } >= {"scan_decisions", "select_candidates", "materialize_sources", "verify"}
    with materializer._connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM explicit_join_balance_units WHERE complete = 1"
        ).fetchone()[0] == 1


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


def test_link_to_missing_asset_is_rejected(
    tmp_path: Path,
) -> None:
    links = _links()
    links[0] = {**links[0], "asset_ids": ["missing-asset"]}

    with pytest.raises(ValueError, match="link.*asset"):
        materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                assets=_assets(),
                links=links,
            ),
            args=_args(tmp_path),
            split="train",
        )


def test_link_to_asset_owned_by_another_entity_is_rejected(
    tmp_path: Path,
) -> None:
    assets = _assets()
    assets[0] = {**assets[0], "entity_id": "entity-beta"}

    with pytest.raises(ValueError, match="link.*entity"):
        materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                assets=assets,
                links=_links(),
            ),
            args=_args(tmp_path),
            split="train",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"asset_id": "missing-asset"}, "extraction.*asset"),
        ({"entity_id": "entity-beta"}, "extraction.*entity"),
        ({"source_table_id": "foreign-source"}, "extraction.*source"),
        ({"source_row_id": 999}, "extraction.*source"),
    ],
)
def test_extraction_relation_mismatches_are_rejected(
    tmp_path: Path,
    mutation: dict[str, Any],
    message: str,
) -> None:
    args = _args(tmp_path)
    assets = _assets()
    extraction = {
        **_extractions(args, assets)[0],
        **mutation,
    }

    with pytest.raises(ValueError, match=message):
        materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                assets=assets,
                links=_links(),
                extractions=[extraction],
            ),
            args=args,
            split="train",
        )


@pytest.mark.parametrize("mutation", ["missing-source", "missing-row"])
def test_relations_must_close_over_the_source_catalog(
    tmp_path: Path,
    mutation: str,
) -> None:
    entities = _entities()
    links = _links()
    if mutation == "missing-source":
        for entity in entities:
            entity["appears_in"][0]["source_table_id"] = "foreign-source"
        links = [
            {**link, "source_table_id": "foreign-source"}
            for link in links
        ]
    else:
        for entity in entities:
            entity["appears_in"][0]["row_id"] = 999
        links = [{**link, "row_id": 999} for link in links]

    with pytest.raises(ValueError, match="source catalog"):
        materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                entities=entities,
                assets=_assets(),
                links=links,
            ),
            args=_args(tmp_path),
            split="train",
        )


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
    irrelevant_entities = [
        {
            **_entities()[0],
            "entity_id": f"other-{index:05d}",
            "wiki_title": f"wdc_other_{index:05d}",
            "display_texts": [f"Other {index}"],
        }
        for index in range(10_000)
    ]
    tracemalloc.start()
    try:
        actual = materialize_dataset_shard(
            _shard_inputs(
                tmp_path,
                entities=[*_entities(), *irrelevant_entities],
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
    assert (
        manifest["schema_version"]
        == materializer.MATERIALIZATION_SCHEMA_VERSION
    )
    assert manifest["complete"] is True
    assert (
        manifest["query_construction"]["qualified_attribute_policy"]
        == "recovery_qualified_variants_after_context_floor"
    )
    assert manifest["query_construction"]["min_implicit_context_columns"] == 2
    assert (
        manifest["query_construction"]["identical_visible_query_policy"]
        == "merge_exact_row_view_with_all_distinct_positive_targets"
    )
    assert (
        manifest["query_construction"]["query_row_selection"]
        == "recovery_balanced_disjoint_train_views"
    )
    assert manifest["query_construction"]["max_train_query_row_views_per_join"] == 5
    assert manifest["query_construction"]["evaluation_query_row_views_per_join"] == 1
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
    assert "rows" not in data_lake_records[0]
    assert data_lake_records[0]["source_table_ref"] == {
        "artifact": "source_tables",
        "source_table_id": "source-1",
    }
    assert data_lake_records[0]["role"] == "raw_data_lake_table"
    resolved_data_lake = list(
        iter_manifest_records(
            output_root,
            "data_lake_tables",
            log_every=0,
        )
    )
    assert len(resolved_data_lake[0]["rows"]) == 2
    assert resolved_data_lake[0]["source_row_indices"] == [0, 1]
    assert "source_table_ref" not in resolved_data_lake[0]

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


def test_full_materialization_counts_visible_join_fallback_as_queryable(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    args.explicit_join_fallback_ratio = 1.0

    result = materialize_dataset(
        inputs,
        output_root=tmp_path / "output-explicit",
        args=args,
        records_per_shard=1,
    )

    assert result.stats["queryable_source_tables"] == 1
    assert result.stats["multimodal_queryable_source_tables"] == 0
    assert result.stats["explicit_join_source_tables"] == 1
    assert result.stats["rejected_source_tables"] == 0
    assert result.stats["query_tables"] == 1
    assert result.stats["qrels"] == 1


@pytest.mark.parametrize(
    "schema_version",
    [
        "wdc200k-materialization-v1",
        "wdc200k-materialization-v2",
        "wdc200k-materialization-v3",
        "wdc200k-materialization-v4",
        "wdc200k-materialization-v5",
        "wdc200k-materialization-v6",
    ],
)
def test_legacy_published_materialization_schema_is_rebuilt(
    tmp_path: Path,
    schema_version: str,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "stage": "wdc200k_materialization",
                "schema_version": schema_version,
                "complete": True,
            }
        ),
        encoding="utf-8",
    )

    assert materializer._load_published_result(
        output_root,
        upstream=argparse.Namespace(identity="upstream"),
        parameter_fingerprint="parameters",
        records_per_shard=1,
    ) is None


def test_real_fetch_and_model_failures_reach_canonical_diagnostics(
    tmp_path: Path,
) -> None:
    web_inputs, web_args = _authoritative_inputs(tmp_path / "web")
    web_output = tmp_path / "web-output"
    materialize_dataset(
        web_inputs,
        output_root=web_output,
        args=web_args,
    )
    web_failures = [
        json.loads(line)
        for line in (web_output / "web_fetch_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(web_failures) == 2
    assert {record["stage"] for record in web_failures} == {
        "page_fetch"
    }

    failed_inputs, failed_args = _authoritative_inputs(
        tmp_path / "media-model",
        page_success=True,
        page_image_url="https://images.test/failure.jpg",
        extractor=_FailingExtractor(),
    )
    failed_output = tmp_path / "media-model-output"
    materialize_dataset(
        failed_inputs,
        output_root=failed_output,
        args=failed_args,
    )
    media_failures = [
        json.loads(line)
        for line in (failed_output / "media_download_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    model_failures = [
        json.loads(line)
        for line in (failed_output / "model_attribute_errors.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = json.loads(
        (failed_output / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(media_failures) == 1
    assert media_failures[0]["failure_type"] == (
        "media_download_failure"
    )
    assert media_failures[0]["error_class"] == "TimeoutError"
    assert media_failures[0]["affected_reference_count"] == 2
    assert model_failures
    assert {
        "web_fetch_failures",
        "media_download_failures",
        "model_attribute_errors",
    } <= set(manifest["single_files"])


def test_real_task3_expand_through_task7_is_readable(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        extractor=_GenericExtractor(),
        real_structural=True,
    )
    output_root = tmp_path / "output"

    result = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=1,
    )

    source_records = list(
        iter_manifest_records(output_root, "source_tables", log_every=0)
    )
    assert result.complete is True
    assert len(source_records) == 1
    assert len(source_records[0]["rows"]) == 2
    assert "image" not in {
        column["column_name"]
        for column in source_records[0]["columns"]
    }
    assert load_split_map(output_root)


def test_cross_run_task5_and_task6_substitution_is_rejected(
    tmp_path: Path,
) -> None:
    first, args = _authoritative_inputs(tmp_path / "first")
    second, _second_args = _authoritative_inputs(
        tmp_path / "second",
        page_success=True,
        extractor=_StateExtractor(),
    )

    with pytest.raises(ValueError, match="materialization"):
        materialize_dataset(
            replace(
                first,
                materialized_assets=second.materialized_assets,
            ),
            output_root=tmp_path / "foreign-task5-output",
            args=args,
        )
    with pytest.raises(ValueError, match="adapter"):
        materialize_dataset(
            replace(
                first,
                adapted_model_tasks=second.adapted_model_tasks,
                model_result=second.model_result,
            ),
            output_root=tmp_path / "foreign-task6-output",
            args=args,
        )


@pytest.mark.parametrize("interrupt_after", [1, 5, 10, 13, 15])
def test_global_finalize_interruptions_resume_without_publishing_partial_manifest(
    tmp_path: Path,
    interrupt_after: int,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"
    commits: list[str] = []

    def interrupt(name: str) -> None:
        commits.append(name)
        if len(commits) == interrupt_after:
            raise RuntimeError("global finalize interrupted")

    with pytest.raises(RuntimeError, match="global finalize"):
        materialize_dataset(
            inputs,
            output_root=output_root,
            args=args,
            records_per_shard=1,
            after_finalize_commit=interrupt,
        )
    assert not (output_root / "dataset_manifest.json").exists()
    if interrupt_after == 13:
        assert (output_root / "web_fetch_failures.jsonl").is_file()
        assert not (
            output_root / "media_download_failures.jsonl"
        ).exists()
    stale = output_root / "source_tables" / "part-99999.jsonl"
    stale.write_text('{"stale":true}\n', encoding="utf-8")

    result = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=1,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    declared = {
        item["path"]
        for artifact in manifest["artifacts"].values()
        for item in artifact["shards"]
    }
    assert "source_tables/part-99999.jsonl" not in declared
    assert list(
        iter_manifest_records(output_root, "source_tables", log_every=0)
    )


def test_resume_revalidates_upstream_checksums_before_skipping(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"
    materialize_dataset(inputs, output_root=output_root, args=args)
    asset_payload = json.loads(
        inputs.materialized_assets.manifest_path.read_text(
            encoding="utf-8"
        )
    )
    asset_path = (
        inputs.materialized_assets.manifest_path.parent
        / asset_payload["table_asset_link_shards"][0]["path"]
    )
    asset_path.write_text('{"forged":true}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="asset materialization shard checksum validation",
    ):
        materialize_dataset(inputs, output_root=output_root, args=args)


def test_forged_barriers_and_missing_model_manifest_are_rejected(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)

    with pytest.raises(ValueError, match="model stage authority"):
        materialize_dataset(
            replace(
                inputs,
                model_authority=replace(
                    inputs.model_authority,
                    prompt_version="foreign-prompt-v0",
                ),
            ),
            output_root=tmp_path / "foreign-authority-output",
            args=args,
        )

    with pytest.raises(ValueError, match="does not match manifest"):
        materialize_dataset(
            replace(
                inputs,
                materialized_assets=replace(
                    inputs.materialized_assets,
                    bridge_assets=(
                        inputs.materialized_assets.bridge_assets + 1
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
    with pytest.raises(ValueError, match="model stage result validation"):
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


def test_partial_materialization_resume_uses_verified_upstream_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    args.materialization_validation_workers = 1
    output_root = tmp_path / "output"

    def interrupt(_source_table_id: str) -> None:
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_dataset(
            inputs,
            output_root=output_root,
            args=args,
            after_table_commit=interrupt,
        )

    certificates = list(
        (inputs.work_root / "materialization").glob(
            "upstream-certificate-*.json"
        )
    )
    assert len(certificates) == 1
    certificate = json.loads(certificates[0].read_text(encoding="utf-8"))
    assert certificate["complete"] is True
    assert certificate["certificate_sha256"]
    with sqlite3.connect(
        next(
            (inputs.work_root / "materialization").glob("index-*.sqlite3")
        )
    ) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_units WHERE complete = 1"
        ).fetchone() == (1,)
    assert load_certified_materialization_inputs(
        inputs.work_root,
        args=args,
        records_per_shard=50_000,
    ) == inputs

    monkeypatch.setattr(
        materializer,
        "_validate_upstream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strict upstream validation was repeated")
        ),
    )
    progress: list[dict[str, Any]] = []
    resumed = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        validation_progress_callback=progress.append,
    )

    assert resumed.complete is True
    assert any(
        event.get("mode") == "fast_resume"
        and event.get("resumed_source_units") == 1
        for event in progress
    )


def test_certificate_tracks_nonempty_sqlite_wal_but_ignores_sidecars(
    tmp_path: Path,
) -> None:
    inputs, _args = _authoritative_inputs(tmp_path)
    database_path = Path(inputs.model_result.jobset.database_path).resolve()
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    wal_path.write_bytes(b"")
    shm_path.write_bytes(b"transient SQLite coordination state")

    certificate_paths = set(materializer._certificate_input_paths(inputs))

    assert database_path in certificate_paths
    assert wal_path.resolve() not in certificate_paths
    assert shm_path.resolve() not in certificate_paths

    wal_path.write_bytes(b"durable WAL input")

    certificate_paths = set(materializer._certificate_input_paths(inputs))

    assert wal_path.resolve() in certificate_paths


def test_empty_sqlite_sidecars_before_certificate_do_not_change_upstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    database_path = Path(inputs.model_result.jobset.database_path).resolve()
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    wal_path.unlink(missing_ok=True)
    shm_path.unlink(missing_ok=True)
    compact = materializer._compact_materialized_table_copies

    def compact_with_reader_sidecars(
        *compact_args: Any,
        **compact_kwargs: Any,
    ) -> None:
        compact(*compact_args, **compact_kwargs)
        wal_path.touch()
        shm_path.write_bytes(b"transient SQLite coordination state")

    monkeypatch.setattr(
        materializer,
        "_compact_materialized_table_copies",
        compact_with_reader_sidecars,
    )

    result = materialize_dataset(
        inputs,
        output_root=tmp_path / "output",
        args=args,
    )

    assert result.complete is True


@pytest.mark.parametrize(
    "invalidate",
    [
        "missing",
        "input_identity",
        "certificate_digest",
        "config_fingerprint",
    ],
)
def test_upstream_certificate_miss_falls_back_to_strict_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalidate: str,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    args.materialization_validation_workers = 1
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_dataset(
            inputs,
            output_root=output_root,
            args=args,
            after_table_commit=lambda _source_table_id: (
                (_ for _ in ()).throw(
                    RuntimeError("simulated interruption")
                )
            ),
        )

    certificate = next(
        (inputs.work_root / "materialization").glob(
            "upstream-certificate-*.json"
        )
    )
    if invalidate == "missing":
        certificate.unlink()
    elif invalidate == "certificate_digest":
        payload = json.loads(certificate.read_text(encoding="utf-8"))
        payload["certificate_sha256"] = "0" * 64
        certificate.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    elif invalidate == "input_identity":
        input_path = inputs.materialized_assets.table_asset_link_paths[0]
        stat = input_path.stat()
        os.utime(
            input_path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
        )

    strict_calls = 0
    real_validate = materializer._validate_upstream

    def counting_validate(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal strict_calls
        strict_calls += 1
        return real_validate(*call_args, **call_kwargs)

    monkeypatch.setattr(
        materializer,
        "_validate_upstream",
        counting_validate,
    )
    progress: list[dict[str, Any]] = []
    resumed = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        records_per_shard=(
            1 if invalidate == "config_fingerprint" else 50_000
        ),
        validation_progress_callback=progress.append,
    )

    assert resumed.complete is True
    assert strict_calls == 1
    assert any(event.get("mode") == "fallback" for event in progress)
    assert any(event.get("mode") == "strict" for event in progress)


def test_upstream_certificate_cannot_hide_corrupt_input(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)
    output_root = tmp_path / "output"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_dataset(
            inputs,
            output_root=output_root,
            args=args,
            after_table_commit=lambda _source_table_id: (
                (_ for _ in ()).throw(
                    RuntimeError("simulated interruption")
                )
            ),
        )
    link_path = inputs.materialized_assets.table_asset_link_paths[0]
    link_path.write_text('{"corrupt":true}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="asset materialization shard checksum validation",
    ):
        materialize_dataset(
            inputs,
            output_root=output_root,
            args=args,
        )


def test_parallel_and_serial_upstream_validation_are_equivalent(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        page_image_url="https://images.test/equivalent.jpg",
        extractor=_FailingExtractor(),
    )
    args.materialization_validation_workers = 1
    serial = materializer._validate_upstream(inputs, args=args)

    args.materialization_validation_workers = 4
    parallel = materializer._validate_upstream(inputs, args=args)

    assert parallel == serial


def test_parallel_and_serial_upstream_validation_raise_same_corruption(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        page_image_url="https://images.test/corrupt.jpg",
        extractor=_FailingExtractor(),
    )
    inputs.unique_image_jobs.output_path.write_text(
        '{"corrupt":true}\n',
        encoding="utf-8",
    )
    errors: list[tuple[type[BaseException], str]] = []

    for workers in (1, 4):
        args.materialization_validation_workers = workers
        with pytest.raises(ValueError) as captured:
            materializer._validate_upstream(inputs, args=args)
        errors.append((type(captured.value), str(captured.value)))

    assert errors[1] == errors[0]


def test_relation_and_source_closures_run_on_independent_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, args = _authoritative_inputs(tmp_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        materialize_dataset(
            inputs,
            output_root=tmp_path / "output",
            args=args,
            after_table_commit=lambda _source_table_id: (
                (_ for _ in ()).throw(
                    RuntimeError("simulated interruption")
                )
            ),
        )
    database_path = next(
        (inputs.work_root / "materialization").glob("index-*.sqlite3")
    )
    barrier = threading.Barrier(2)
    real_relations = materializer._validate_relation_closure
    real_sources = materializer._validate_source_catalog_closure

    def synchronized_relations(connection: sqlite3.Connection) -> None:
        barrier.wait(timeout=5)
        real_relations(connection)

    def synchronized_sources(
        connection: sqlite3.Connection,
        *,
        write_tracker: Any = None,
    ) -> None:
        barrier.wait(timeout=5)
        real_sources(connection, write_tracker=write_tracker)

    monkeypatch.setattr(
        materializer,
        "_validate_relation_closure",
        synchronized_relations,
    )
    monkeypatch.setattr(
        materializer,
        "_validate_source_catalog_closure",
        synchronized_sources,
    )

    materializer._validate_materialization_index_closures(
        database_path,
        validation_workers=2,
    )


def test_nonempty_task6_outputs_materialize_query_qrel_and_evidence(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        extractor=_StateExtractor(),
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
    assert result.stats["attribute_extractions"] >= 2
    assert result.stats["evidence_recoveries"] >= 2
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


class _QueryChecker:
    auto_check_enabled = True
    auto_check_parallelism = 1

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.supported_asset_ids: set[str] = set()

    def extract_auto_check_value(
        self,
        *,
        task: Any,
        attribute_name: str,
        claimed_value: str,
        **_kwargs: Any,
    ) -> str:
        asset_id = str(task.asset["asset_id"])
        self.calls.append((task.source_row_id, asset_id))
        if asset_id.endswith("_000"):
            self.supported_asset_ids.add(asset_id)
            return claimed_value
        return ""


@pytest.mark.parametrize("remote_review", [False, True])
def test_query_auto_check_exhausts_final_query_evidence_after_threshold_selection(
    tmp_path: Path,
    remote_review: bool,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        extractor=_StateExtractor(),
    )
    args.auto_check_secondary_openai = remote_review
    checker = _QueryChecker()
    output_root = tmp_path / "output"
    result = materialize_dataset(
        inputs,
        output_root=output_root,
        args=args,
        extractor=checker,
        records_per_shard=2,
    )

    assert len(checker.calls) == 4
    assert [row_id for row_id, _asset_id in checker.calls] == [
        0,
        0,
        1,
        1,
    ]
    assert [
        asset_id.rsplit("_", 1)[-1]
        for _row_id, asset_id in checker.calls
    ] == [
        "000",
        "001",
        "000",
        "001",
    ]
    assert result.stats["query_tables"] == 1
    recoveries = list(
        iter_manifest_records(
            output_root, "evidence_recoveries", log_every=0
        )
    )
    assert len(recoveries) == 2
    assert {
        recovery["evidence"]["asset_id"] for recovery in recoveries
    } == checker.supported_asset_ids
    assert all(
        recovery["auto_check"]["reviews"][0]["verdict"]
        == "supported"
        for recovery in recoveries
    )
    assert (
        Path(args.cache_dir) / "query_recovery_auto_checks.jsonl"
    ).is_file()
    database_path = next(
        (inputs.work_root / "materialization").glob("index-*.sqlite3")
    )
    with materializer._connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM query_auto_check_units WHERE complete = 1"
        ).fetchone()[0] == 1


def test_final_materialization_loads_query_checks_by_exact_cache_key(
    tmp_path: Path,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        extractor=_StateExtractor(),
    )
    args.auto_check_secondary_openai = True
    materialize_dataset(
        inputs,
        output_root=tmp_path / "output",
        args=args,
        extractor=_QueryChecker(),
        records_per_shard=2,
    )
    database_path = next(
        (inputs.work_root / "materialization").glob("index-*.sqlite3")
    )
    with materializer._connect(database_path) as connection:
        connection.execute(
            """
            UPDATE query_auto_checks
            SET extraction_cache_key = 'unrelated-extraction'
            """
        )
        connection.commit()

    materialize_args = argparse.Namespace(**vars(args))
    materialize_args._query_auto_check_required = True
    materialize_args._query_auto_check_review_policy = (
        join_builder.AUTO_CHECK_REVIEW_POLICY_LOCAL
    )
    materialized = materializer._materialize_from_index(
        _source_table(),
        database_path,
        args=materialize_args,
        split="train",
    )

    assert len(materialized.query_tables) == 1
    assert len(materialized.qrels) == 1
    assert len(materialized.evidence_recoveries) == 2


@pytest.mark.parametrize("cache_has_missing_check", [False, True])
def test_global_query_auto_check_repairs_incomplete_completed_unit(
    tmp_path: Path,
    cache_has_missing_check: bool,
) -> None:
    inputs, args = _authoritative_inputs(
        tmp_path,
        page_success=True,
        extractor=_StateExtractor(),
    )
    args.auto_check_secondary_openai = True
    checker = _QueryChecker()
    materialize_dataset(
        inputs,
        output_root=tmp_path / "output",
        args=args,
        extractor=checker,
        records_per_shard=2,
    )
    database_path = next(
        (inputs.work_root / "materialization").glob("index-*.sqlite3")
    )
    with materializer._connect(database_path) as connection:
        rows = connection.execute(
            "SELECT cache_key, record_json FROM query_auto_checks "
            "ORDER BY cache_key"
        ).fetchall()
        assert len(rows) == 4
        connection.execute(
            "DELETE FROM query_auto_checks WHERE cache_key = ?",
            (str(rows[0]["cache_key"]),),
        )
        connection.commit()
    cached_rows = rows if cache_has_missing_check else rows[1:]
    cache = _MemoryCache(
        [json.loads(str(row["record_json"])) for row in cached_rows]
    )
    calls_before_resume = len(checker.calls)

    materializer._run_small_query_auto_check_prepass(
        database_path,
        extractor=checker,
        cache=cache,
        args=args,
        expected_tables=1,
    )

    assert len(checker.calls) == calls_before_resume + int(
        not cache_has_missing_check
    )
    with materializer._connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM query_auto_checks"
        ).fetchone()[0] == 4
        assert connection.execute(
            "SELECT COUNT(*) FROM query_auto_check_units WHERE complete = 1"
        ).fetchone()[0] == 1


def _query_recovery_candidate() -> join_builder.QueryRecoveryCandidate:
    task = join_builder.ExtractionTask(
        order=0,
        cache_key="legacy-extraction",
        source_table_id="source-1",
        source_row_id=0,
        entity_column_index=0,
        entity_column_name="Name",
        entity={
            "entity_id": "entity-alpha",
            "wiki_title": "wdc_alpha",
            "cell_text": "Alpha",
            "row_attributes": [
                {"name": "Name", "value": "Alpha", "is_entity": True},
                {"name": "State", "value": "Texas", "is_entity": False},
            ],
        },
        asset={"asset_id": "asset-alpha", "asset_type": "text"},
        candidate_attribute_names=["State"],
    )
    return join_builder.QueryRecoveryCandidate(
        task=task,
        extraction={"cache_key": task.cache_key},
        recovery={
            "source_row_id": 0,
            "recovered_attribute": {
                "column_name": "State",
                "value": "Texas",
                "model_value": "Texas",
            },
        },
    )


def test_legacy_remote_review_is_migrated_as_reusable_model_stages() -> None:
    candidate = _query_recovery_candidate()
    task = candidate.task
    cascade = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=object(),
    )
    review = {
        "attribute_name": "State",
        "claimed_value": "Texas",
        "verdict": "supported",
        "error_code": "",
        "review_complete": True,
        "decision_source": "terra_adjudication",
        "luna_triggered": True,
        "terra_triggered": True,
    }

    extraction = {
        "cache_key": task.cache_key,
        "auto_check": {
            "schema_version": join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "reviews": [review],
        },
    }

    legacy = materializer._legacy_query_auto_check_record(
        candidate,
        extraction,
        extractor=cascade,
    )

    assert legacy is not None
    assert legacy["review_policy"] == (
        join_builder.AUTO_CHECK_REVIEW_POLICY_LEGACY
    )
    assert join_builder.cached_model_auto_check_review_policy(legacy) == (
        join_builder.AUTO_CHECK_REVIEW_POLICY_LEGACY
    )

    extraction["auto_check"]["review_policy"] = (
        join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE
    )
    migrated = materializer._legacy_query_auto_check_record(
        candidate,
        extraction,
        extractor=cascade,
    )

    assert migrated is not None
    assert migrated["supported"] is True
    assert migrated["review_policy"] == (
        join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE
    )
    assert migrated["auto_check"]["reviews"] == [review]
    assert join_builder.query_recovery_remote_review_is_complete(migrated)


def test_query_auto_check_persistence_uses_cascade_key_over_stale_local(
    tmp_path: Path,
) -> None:
    candidate = _query_recovery_candidate()
    local = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=None,
    )
    cascade = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=object(),
    )
    local_key = join_builder.query_recovery_auto_check_key(candidate, local)
    cascade_key = join_builder.query_recovery_auto_check_key(
        candidate,
        cascade,
    )

    def record(key: str, policy: str, marker: str) -> dict[str, Any]:
        remote = policy == join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE
        return {
            "cache_key": key,
            "extraction_cache_key": candidate.task.cache_key,
            "review_policy": policy,
            "query_row_attributes": candidate.task.entity["row_attributes"],
            "attribute_name": "State",
            "claimed_value": "Texas",
            "evidence_identity": (
                join_builder.query_recovery_remote_evidence_identity(candidate)
            ),
            "schema_version": join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "supported": True,
            "marker": marker,
            "auto_check": {
                "schema_version": join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION,
                "review_policy": policy,
                "reviewed_attributes": 1,
                "reviews": [
                    {
                        "review_complete": True,
                        "error_code": "",
                        "decision_source": (
                            "local_luna_consensus" if remote else "primary_local"
                        ),
                        "luna_triggered": remote,
                    }
                ],
            },
        }

    cache = join_builder.ExtractionCache(
        tmp_path / "query-checks.jsonl",
        reuse=False,
        record_key_alias=join_builder.query_recovery_auto_check_record_key,
    )
    cache.put(
        local_key,
        record(
            local_key,
            join_builder.AUTO_CHECK_REVIEW_POLICY_LOCAL,
            "stale-local",
        ),
    )
    cache.put(
        cascade_key,
        record(
            cascade_key,
            join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE,
            "current-cascade",
        ),
    )
    database = tmp_path / "materialize.sqlite3"
    materializer._initialize_index(database)
    plan = join_builder.QueryRecoveryAutoCheckPlan(
        query_key="query-1",
        required_recovered_rows=1,
        source_row_order=(0,),
        candidates=(candidate,),
    )
    item = materializer._MaterializationWorkItem(
        source_table_id="source-1",
        source_ordinal=0,
        source_sha256="source-sha",
        split="train",
    )

    materializer._persist_query_auto_check_batch(
        database,
        [(item, [plan])],
        cache=cache,
        extractor=cascade,
    )

    with materializer._connect(database) as connection:
        row = connection.execute(
            "SELECT cache_key, record_json FROM query_auto_checks"
        ).fetchone()
    assert row["cache_key"] == cascade_key
    assert json.loads(row["record_json"])["marker"] == "current-cascade"


def test_materialization_identity_tracks_actual_auto_check_policy(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    local = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=None,
    )
    cascade = SimpleNamespace(
        auto_check_enabled=True,
        auto_check_luna_reviewer=object(),
    )
    policies = {
        materializer._materialization_review_policy(extractor)
        for extractor in (None, local, cascade)
    }
    fingerprints = {
        materializer._parameter_fingerprint(
            args,
            review_policy=policy,
        )
        for policy in policies
    }

    assert policies == {
        "disabled",
        join_builder.AUTO_CHECK_REVIEW_POLICY_LOCAL,
        join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE,
    }
    assert len(fingerprints) == len(policies)
    payload = materializer._parameter_payload(
        args,
        review_policy=join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE,
    )
    assert payload["auto_check_schema_version"] == (
        join_builder.MODEL_AUTO_CHECK_SCHEMA_VERSION
    )
    assert payload["auto_check_review_policy"] == (
        join_builder.AUTO_CHECK_REVIEW_POLICY_CASCADE
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


def test_materialization_checkpoints_between_bounded_source_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "materialized.sqlite3"
    args = _args(tmp_path)
    sources = [
        _source_table(f"source-{index:02d}") for index in range(5)
    ]
    materializer._initialize_index(database_path)
    materializer._catalog_source_records(
        database_path,
        sources,
        args=args,
        expected_tables=len(sources),
    )
    materializer._assign_splits(database_path, args)

    checkpoints: list[Path] = []
    checkpoint_wal = materializer._checkpoint_wal

    def recording_checkpoint(path: Path) -> None:
        checkpoints.append(path)
        checkpoint_wal(path)

    monkeypatch.setattr(
        materializer,
        "_MATERIALIZATION_READ_BATCH_RECORDS",
        2,
    )
    monkeypatch.setattr(
        materializer,
        "_checkpoint_wal",
        recording_checkpoint,
    )
    progress: list[dict[str, Any]] = []
    materializer._materialize_all_tables(
        database_path,
        args=args,
        expected_tables=len(sources),
        progress_callback=progress.append,
    )

    assert checkpoints == [database_path] * 3
    assert [item["completed"] for item in progress] == [0, 1, 2, 3, 4, 5, 5]
    assert all(item["total"] == 5 for item in progress)
    with materializer._connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM source_units WHERE complete = 1"
        ).fetchone()[0] == len(sources)


def test_query_auto_check_prepass_selects_execution_path_from_review_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def small(*_args: Any, **_kwargs: Any) -> None:
        calls.append("small")

    def large(*_args: Any, **_kwargs: Any) -> None:
        calls.append("large")

    monkeypatch.setattr(
        materializer,
        "_run_small_query_auto_check_prepass",
        small,
    )
    monkeypatch.setattr(
        materializer,
        "_run_large_query_auto_check_prepass",
        large,
    )
    common = {
        "extractor": object(),
        "cache": object(),
        "expected_tables": 0,
    }

    materializer._run_query_auto_check_prepass(
        tmp_path / "small.sqlite3",
        args=SimpleNamespace(auto_check_secondary_openai=True),
        **common,
    )
    materializer._run_query_auto_check_prepass(
        tmp_path / "large.sqlite3",
        args=SimpleNamespace(auto_check_secondary_openai=False),
        **common,
    )

    assert calls == ["small", "large"]


def test_query_auto_check_batch_tables_scales_with_model_workers() -> None:
    args = SimpleNamespace(
        query_auto_check_batch_tables=0,
        text_model_workers=120,
        image_model_workers=36,
        remote_text_model_workers=0,
        remote_image_model_workers=0,
    )

    assert materializer._query_auto_check_batch_tables(args) == 624
    args.query_auto_check_batch_tables = 256
    assert materializer._query_auto_check_batch_tables(args) == 256
    args.query_auto_check_batch_tables = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        materializer._query_auto_check_batch_tables(args)


def test_query_auto_check_source_batch_skips_only_valid_completed_units(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    sources = [_source_table(f"source-{index}") for index in range(3)]
    materializer._initialize_index(database)
    materializer._catalog_source_records(
        database,
        sources,
        args=_args(tmp_path),
        expected_tables=len(sources),
    )
    with materializer._connect(database) as connection:
        hashes = {
            str(row["source_table_id"]): str(row["record_sha256"])
            for row in connection.execute(
                "SELECT source_table_id, record_sha256 FROM source_catalog"
            )
        }
        connection.executemany(
            """
            INSERT INTO query_auto_check_units (
                source_table_id, source_sha256, plan_count,
                cached_check_count, complete
            ) VALUES (?, ?, 0, 0, 1)
            """,
            [
                ("source-0", hashes["source-0"]),
                ("source-1", "stale-source-hash"),
            ],
        )
        connection.commit()

    rows = materializer._query_auto_check_source_batch(
        database,
        last_ordinal=-1,
        limit=10,
        pending_only=True,
    )

    assert [row["source_table_id"] for row in rows] == [
        "source-1",
        "source-2",
    ]
    with pytest.raises(ValueError, match="resume mismatch"):
        materializer._pending_query_auto_check_items(rows)


def test_large_query_auto_check_prefetches_and_reuses_concurrency_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        query_auto_check_batch_tables=1,
        text_model_workers=2,
        image_model_workers=1,
        remote_text_model_workers=0,
        remote_image_model_workers=0,
    )
    rows = {
        -1: [
            {
                "source_table_id": "source-0",
                "ordinal": 0,
                "split": "train",
                "record_sha256": "sha-0",
                "checked_sha256": None,
                "checked": None,
            }
        ],
        0: [
            {
                "source_table_id": "source-1",
                "ordinal": 1,
                "split": "train",
                "record_sha256": "sha-1",
                "checked_sha256": None,
                "checked": None,
            }
        ],
        1: [],
    }
    second_prepare_started = threading.Event()
    second_prepare_finished = threading.Event()
    release_second_prepare = threading.Event()
    concurrency_states: list[Any] = []
    persisted: list[str] = []
    verified: list[tuple[int, int]] = []

    def source_batch(
        _database_path: Path,
        *,
        last_ordinal: int,
        limit: int,
        pending_only: bool,
    ) -> list[dict[str, Any]]:
        assert limit == 1
        assert pending_only is True
        return rows[last_ordinal]

    def prepare(
        _database_path: Path,
        items: Iterable[Any],
        **_kwargs: Any,
    ) -> tuple[list[tuple[Any, list[str]]], list[str], int]:
        item = next(iter(items))
        if item.source_ordinal == 1:
            second_prepare_started.set()
            assert release_second_prepare.wait(timeout=2)
            second_prepare_finished.set()
        plan = f"plan-{item.source_ordinal}"
        return [(item, [plan])], [plan], 0

    def finalize(*, plans: list[str], concurrency_state: Any, **_kwargs: Any) -> None:
        concurrency_states.append(concurrency_state)
        if plans == ["plan-0"]:
            assert second_prepare_started.wait(timeout=2)
            release_second_prepare.set()

    def persist(
        _database_path: Path,
        units: list[tuple[Any, list[str]]],
        **_kwargs: Any,
    ) -> tuple[int, int]:
        persisted.append(units[0][0].source_table_id)
        return len(units), 0

    monkeypatch.setattr(
        materializer,
        "_query_auto_check_completed_units",
        lambda _path: 0,
    )
    monkeypatch.setattr(
        materializer,
        "_query_auto_check_source_batch",
        source_batch,
    )
    monkeypatch.setattr(
        materializer,
        "_prepare_query_auto_check_units",
        prepare,
    )
    monkeypatch.setattr(
        materializer.join_builder,
        "finalize_query_recovery_auto_checks",
        finalize,
    )
    monkeypatch.setattr(
        materializer,
        "_persist_query_auto_check_batch",
        persist,
    )
    monkeypatch.setattr(
        materializer,
        "_checkpoint_wal",
        lambda _path: (
            None
            if second_prepare_finished.is_set()
            else pytest.fail("WAL checkpoint raced with batch preparation")
        ),
    )
    monkeypatch.setattr(
        materializer,
        "_verify_query_auto_check_prepass",
        lambda _path, *, observed, expected_tables, **_kwargs: (
            verified.append((observed, expected_tables))
        ),
    )

    materializer._run_large_query_auto_check_prepass(
        tmp_path / "materialize.sqlite3",
        extractor=object(),
        cache=object(),
        args=args,
        expected_tables=2,
    )

    assert persisted == ["source-0", "source-1"]
    assert len(concurrency_states) == 2
    assert concurrency_states[0] is concurrency_states[1]
    assert verified == [(2, 2)]


def test_materialization_worker_source_fingerprint_rejects_changed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = materializer._MATERIALIZATION_WORKER_SOURCE_FINGERPRINT
    materializer._require_materialization_worker_source(expected)

    monkeypatch.setattr(
        materializer,
        "_materialization_worker_source_fingerprint",
        lambda: "changed-source",
    )
    with pytest.raises(
        RuntimeError,
        match="source changed after pipeline startup",
    ):
        materializer._require_materialization_worker_source(expected)


def test_parallel_materialization_commits_in_source_order_and_resumes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "materialized.sqlite3"
    args = _args(tmp_path)
    args.materialization_workers = 2
    sources = [
        _source_table(f"source-{index:02d}") for index in range(5)
    ]
    materializer._initialize_index(database_path)
    materializer._catalog_source_records(
        database_path,
        sources,
        args=args,
        expected_tables=len(sources),
    )
    materializer._assign_splits(database_path, args)

    committed: list[str] = []
    materializer._materialize_all_tables(
        database_path,
        args=args,
        expected_tables=len(sources),
        after_table_commit=committed.append,
    )

    expected_ids = [
        str(source["source_table_id"]) for source in sources
    ]
    assert committed == expected_ids
    with materializer._connect(database_path) as connection:
        stored_ids = [
            str(row["source_table_id"])
            for row in connection.execute(
                "SELECT source_table_id FROM source_units ORDER BY rowid"
            )
        ]
    assert stored_ids == expected_ids

    resumed_commits: list[str] = []
    resumed_progress: list[dict[str, Any]] = []
    materializer._materialize_all_tables(
        database_path,
        args=args,
        expected_tables=len(sources),
        after_table_commit=resumed_commits.append,
        progress_callback=resumed_progress.append,
    )
    assert resumed_commits == []
    assert [item["completed"] for item in resumed_progress] == [5, 5]


def test_existing_full_table_copies_migrate_to_references(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "materialized.sqlite3"
    args = _args(tmp_path)
    source_table = _source_table()
    source_table_id = source_table["source_table_id"]
    materializer._initialize_index(database_path)
    materializer._catalog_source_records(
        database_path,
        [source_table],
        args=args,
        expected_tables=1,
    )
    materializer._assign_splits(database_path, args)
    columns = [
        int(column["column_index"])
        for column in source_table["columns"]
    ]
    rows, source_rows = join_builder.project_selected_rows(
        source_table,
        columns,
        {0, 1},
        min_required_cols=0,
    )
    legacy_raw = join_builder.table_record(
        table_id=f"dl_raw_{source_table_id}",
        role="raw_data_lake_table",
        split="test",
        source_table=source_table,
        column_indices=columns,
        rows=rows,
        source_row_indices=source_rows,
        extra={
            "queryable": False,
            "reason": "no_column_met_recovered_value_ratio",
        },
    )
    with materializer._connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        materializer._insert_materialized_records(
            connection,
            database_path=database_path,
            artifact="source_tables",
            records=[source_table],
            source_table_id=source_table_id,
            source_ordinal=0,
        )
        materializer._insert_materialized_records(
            connection,
            database_path=database_path,
            artifact="data_lake_tables",
            records=[legacy_raw],
            source_table_id=source_table_id,
            source_ordinal=0,
        )
        connection.commit()

    materializer._compact_materialized_table_copies(database_path)

    with materializer._connect(database_path) as connection:
        stored_source = json.loads(
            connection.execute(
                """
                SELECT record_json FROM materialized_records
                WHERE artifact = 'source_tables'
                """
            ).fetchone()["record_json"]
        )
        stored_raw = json.loads(
            connection.execute(
                """
                SELECT record_json FROM materialized_records
                WHERE artifact = 'data_lake_tables'
                """
            ).fetchone()["record_json"]
        )
    assert stored_source == materializer._source_catalog_reference(
        source_table_id
    )
    assert stored_raw == materializer._raw_data_lake_reference(
        source_table_id,
        "test",
    )
    assert list(
        materializer._iter_materialized(
            database_path,
            "source_tables",
        )
    ) == [source_table]


def test_source_catalog_insert_retries_one_interface_error() -> None:
    class FlakyConnection:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, sql: str, values: tuple[Any, ...]) -> None:
            assert sql == materializer._SOURCE_CATALOG_INSERT_SQL
            assert values[0] == "source-1"
            self.calls += 1
            if self.calls == 1:
                raise sqlite3.InterfaceError("temporary binding failure")

    connection = FlakyConnection()
    materializer._insert_source_catalog_record(
        connection,
        ("source-1", 7, "Title", "group", "a" * 64, "{}", ""),
        ordinal=7,
        source_table_id="source-1",
    )

    assert connection.calls == 2


def test_source_catalog_insert_reports_safe_binding_diagnostics() -> None:
    class FailingConnection:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, sql: str, values: tuple[Any, ...]) -> None:
            self.calls += 1
            raise sqlite3.InterfaceError("binding failure")

    connection = FailingConnection()
    record_json = '{"secret":"token"}'
    with pytest.raises(sqlite3.InterfaceError) as caught:
        materializer._insert_source_catalog_record(
            connection,
            (
                "source-8",
                71060,
                "CreativeWork",
                "split-group",
                "b" * 64,
                record_json,
                "",
            ),
            ordinal=71060,
            source_table_id="source-8",
        )

    message = str(caught.value)
    assert connection.calls == 2
    assert "ordinal=71060" in message
    assert "source_table_id=source-8" in message
    assert f"5:builtins.str[{len(record_json)}]" in message
    assert record_json not in message


def test_large_source_catalog_record_uses_external_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "catalog.sqlite3"
    source_table = _source_table()
    source_table["rows"][0]["cells"][0]["raw"] = "x" * 2048
    source_table["rows"][0]["cells"][0]["text"] = "x" * 2048
    materializer._initialize_index(database_path)

    monkeypatch.setattr(
        materializer,
        "_INLINE_JSON_MAX_UTF8_BYTES",
        1024 * 1024,
    )
    materializer._catalog_source_records(
        database_path,
        [source_table],
        args=_args(tmp_path),
        expected_tables=1,
    )
    with materializer._connect(database_path) as connection:
        assert connection.execute(
            """
            SELECT record_path FROM source_catalog
            WHERE source_table_id = 'source-1'
            """
        ).fetchone()["record_path"] == ""

    monkeypatch.setattr(
        materializer,
        "_INLINE_JSON_MAX_UTF8_BYTES",
        128,
    )
    materializer._catalog_source_records(
        database_path,
        [source_table],
        args=_args(tmp_path),
        expected_tables=1,
    )
    with materializer._connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT record_json, record_path
            FROM source_catalog
            WHERE source_table_id = 'source-1'
            """
        ).fetchone()
        assert row is not None
        assert row["record_path"]
        assert str(row["record_path"]).startswith(
            "large-json/sha256/"
        )
        assert "x" * 128 not in str(row["record_json"])
        assert materializer._load_stored_json(
            database_path,
            row["record_json"],
            row["record_path"],
        ) == source_table
        materializer._validate_source_catalog_closure(connection)

    canonical_relative = Path(str(row["record_path"]))
    legacy_relative = (
        Path("large-json")
        / "source-catalog"
        / canonical_relative.name
    )
    legacy_path = database_path.parent / legacy_relative
    legacy_path.parent.mkdir(parents=True)
    (database_path.parent / canonical_relative).replace(legacy_path)
    with materializer._connect(database_path) as connection:
        connection.execute(
            """
            UPDATE source_catalog SET record_path = ?
            WHERE source_table_id = 'source-1'
            """,
            (legacy_relative.as_posix(),),
        )
        connection.commit()

    assert materializer._migrate_legacy_external_json_paths(
        database_path
    ) == 1
    assert not legacy_path.exists()
    materializer._catalog_source_records(
        database_path,
        [source_table],
        args=_args(tmp_path),
        expected_tables=1,
    )
    with materializer._connect(database_path) as connection:
        migrated_path = connection.execute(
            """
            SELECT record_path FROM source_catalog
            WHERE source_table_id = 'source-1'
            """
        ).fetchone()["record_path"]
    assert migrated_path == canonical_relative.as_posix()
    assert (database_path.parent / canonical_relative).is_file()
    assert materializer._migrate_legacy_external_json_paths(
        database_path
    ) == 0


def test_external_json_uses_global_content_addressed_path() -> None:
    digest = "a" * 64
    assert materializer._external_json_relative_path(
        "source-catalog",
        "source-1",
        digest,
    ) == materializer._external_json_relative_path(
        "materialized-records/source_tables",
        "different-identity",
        digest,
    )


def test_source_catalog_locator_loads_original_jsonl_record(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "materialization.sqlite3"
    source_path = tmp_path / "source.jsonl"
    source_table = _source_table()
    _write_jsonl(source_path, [source_table])
    materializer._initialize_index(database_path)

    materializer._catalog_sources(
        database_path,
        (source_path,),
        args=_args(tmp_path),
        expected_tables=1,
    )

    with materializer._connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM source_catalog"
        ).fetchone()
        assert row is not None
        assert row["source_path"] == source_path.resolve().as_posix()
        assert row["record_path"] == ""
        assert "rows" not in json.loads(str(row["record_json"]))
        digest = str(row["record_sha256"])
    item = materializer._MaterializationWorkItem(
        source_table_id="source-1",
        source_ordinal=0,
        source_sha256=digest,
        split="train",
    )
    assert materializer._load_materialization_source(
        database_path,
        item,
    ) == source_table


def test_source_table_publication_uses_hard_links(tmp_path: Path) -> None:
    structural_root = tmp_path / "structural"
    source_path = structural_root / "source_tables/part-00000.jsonl"
    writer = AtomicJsonlShard(source_path)
    writer.write(_source_table())
    shard = writer.commit()
    completed = CompletedShard(
        path="source_tables/part-00000.jsonl",
        records=shard.records,
        bytes=shard.bytes,
        sha256=shard.sha256,
    )
    output_root = tmp_path / "output"

    published = materializer._publish_source_table_shards(
        SimpleNamespace(
            source_paths=(source_path,),
            source_shards=(completed,),
        ),
        output_root,
    )

    output_path = output_root / completed.path
    assert published == (completed,)
    assert output_path.stat().st_ino == source_path.stat().st_ino
    assert output_path.stat().st_dev == source_path.stat().st_dev
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "source_table_id"
    ] == "source-1"


def test_certified_cleanup_deletes_only_allowlisted_files(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    structural_root = work_root / "structural"
    candidates = {
        "entities/part-00000.jsonl": b"entity",
        "page_refs/part-00000.jsonl": b"page",
        "selection/part-00000.jsonl": b"selection",
        "selection/protected.jsonl": b"protected",
    }
    completed_shards = []
    for relative, content in candidates.items():
        path = structural_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        completed_shards.append(
            {
                "path": relative,
                "records": 1,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    unrelated = structural_root / "unrelated/keep.jsonl"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep\n", encoding="utf-8")
    manifest_path = structural_root / "stage_manifests/structural.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"complete": True, "completed_shards": completed_shards}),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sampling_manifest = work_root / "sampling/manifest.json"
    sampling_manifest.parent.mkdir(parents=True)
    sampling_manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "compact_source_authority": {
                    "structural_manifests": [
                        {
                            "path": manifest_path.resolve().as_posix(),
                            "sha256": manifest_sha256,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    transient = work_root / "model_outputs/validation.sqlite3"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"temporary")
    Path(f"{transient}-wal").write_bytes(b"wal")
    protected = structural_root / "selection/protected.jsonl"
    certificate = {
        "complete": True,
        "manifest_hashes": [
            {
                "path": manifest_path.resolve().as_posix(),
                "sha256": manifest_sha256,
            }
        ],
        "input_files": [{"path": protected.resolve().as_posix()}],
    }
    certificate["certificate_sha256"] = materializer._certificate_digest(
        certificate
    )
    certificate_path = work_root / "materialization/certificate.json"
    certificate_path.parent.mkdir(parents=True)
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    inputs = SimpleNamespace(
        work_root=work_root,
        structural_output_root=structural_root,
        sampling_manifest=sampling_manifest,
        structural_manifests=(manifest_path,),
    )

    result = materializer._cleanup_certified_intermediates(
        inputs,
        certificate_path,
    )

    assert result["removed_bytes"] > 0
    assert not transient.exists()
    assert not Path(f"{transient}-wal").exists()
    assert not (structural_root / "entities/part-00000.jsonl").exists()
    assert not (structural_root / "page_refs/part-00000.jsonl").exists()
    assert not (structural_root / "selection/part-00000.jsonl").exists()
    assert protected.is_file()
    assert unrelated.is_file()


def test_large_materialized_record_uses_external_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materializer,
        "_INLINE_JSON_MAX_UTF8_BYTES",
        128,
    )
    database_path = tmp_path / "materialized.sqlite3"
    source_table = _source_table()
    source_table["rows"][0]["cells"][0]["raw"] = "y" * 2048
    source_table["rows"][0]["cells"][0]["text"] = "y" * 2048
    materializer._initialize_index(database_path)

    with materializer._connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        assert materializer._insert_materialized_records(
            connection,
            database_path=database_path,
            artifact="source_tables",
            records=[source_table],
            source_table_id="source-1",
            source_ordinal=0,
        ) == 1
        connection.commit()
        row = connection.execute(
            """
            SELECT record_json, record_path
            FROM materialized_records
            WHERE artifact = 'source_tables'
            """
        ).fetchone()
        assert row is not None
        assert row["record_path"]
        assert "y" * 128 not in str(row["record_json"])

    assert list(
        materializer._iter_materialized(
            database_path,
            "source_tables",
        )
    ) == [source_table]

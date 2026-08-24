from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .utils import (
    clean_text,
    get_cell,
    get_column_name,
    normalize,
    stable_hash,
    values_match,
)


@dataclass(frozen=True)
class BuildConfig:
    query_rows: int = 5
    min_target_rows: int = 5
    min_recovered_ratio: float = 0.6
    min_recovered_rows: int = 3
    min_column_non_empty_ratio: float = 0.5
    max_queries_per_source: int = 1
    max_query_additional_columns: int = 1
    max_target_additional_columns: int = 2


def _profiles(table: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        profile["column_index"]: profile
        for profile in table["metadata"]["column_profiles"]
    }


def _entity_column(table: dict[str, Any], query_rows: int) -> int | None:
    profiles = _profiles(table)
    candidates = []
    for column_index in table["metadata"]["candidate_entity_columns"]:
        linked_rows = sum(
            bool(clean_text(get_cell(row, column_index).get("wiki_title")))
            for row in table["rows"]
        )
        if linked_rows >= query_rows:
            candidates.append(column_index)
    if not candidates:
        return None
    return max(candidates, key=lambda index: (profiles[index]["wiki_link_ratio"], -index))


def _project(
    table: dict[str, Any], column_indices: list[int], source_row_ids: set[int]
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    retained_ids: list[int] = []
    for source_row in table["rows"]:
        source_row_id = source_row["row_id"]
        if source_row_id not in source_row_ids:
            continue
        cells = []
        for local_index, source_index in enumerate(column_indices):
            source_cell = get_cell(source_row, source_index)
            cells.append(
                {
                    **source_cell,
                    "column_index": local_index,
                    "source_column_index": source_index,
                    "column_name": get_column_name(table, source_index),
                }
            )
        rows.append({"row_id": len(rows), "source_row_id": source_row_id, "cells": cells})
        retained_ids.append(source_row_id)
    return rows, retained_ids


def _table_record(
    table: dict[str, Any],
    *,
    table_id: str,
    role: str,
    split: str,
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_ids: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": role,
        "split": split,
        "source_table_id": table["source_table_id"],
        "columns": [
            {
                "column_index": local_index,
                "source_column_index": source_index,
                "column_name": get_column_name(table, source_index),
            }
            for local_index, source_index in enumerate(column_indices)
        ],
        "rows": rows,
        "source_column_indices": column_indices,
        "source_row_indices": source_row_ids,
        **extra,
    }


def _rank_additional_columns(
    table: dict[str, Any], excluded: set[int]
) -> list[int]:
    profiles = _profiles(table)
    candidates = [
        column["column_index"]
        for column in table["columns"]
        if column["column_index"] not in excluded
    ]
    return sorted(
        candidates,
        key=lambda index: (
            -profiles[index]["non_empty_ratio"],
            -profiles[index]["unique_ratio"],
            index,
        ),
    )


def _column_layouts(
    table: dict[str, Any],
    entity_col: int,
    qualified: list[dict[str, Any]],
    config: BuildConfig,
) -> list[tuple[dict[str, Any], list[int], list[int]]]:
    selected = qualified[: max(1, config.max_queries_per_source)]
    bridge_columns = {candidate["column_index"] for candidate in qualified}
    ordinary = _rank_additional_columns(table, {entity_col, *bridge_columns})

    query_width = config.max_query_additional_columns
    target_additional = ordinary[: config.max_target_additional_columns]
    query_pool = ordinary[config.max_target_additional_columns :]
    query_additional_sets = list(combinations(query_pool, query_width)) if query_width else [()]
    if len(selected) > 1 and query_additional_sets:
        return [
            (
                candidate,
                list(query_additional_sets[index % len(query_additional_sets)]),
                target_additional,
            )
            for index, candidate in enumerate(selected)
        ]

    best = qualified[0]
    additional = _rank_additional_columns(table, {entity_col, best["column_index"]})
    query_additional = additional[:query_width]
    target_additional = additional[
        query_width : query_width + config.max_target_additional_columns
    ]
    return [(best, query_additional, target_additional)]


def _extraction_index(
    extractions: list[dict[str, Any]],
    asset_ids: set[str],
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    index: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in extractions:
        if record["asset_id"] not in asset_ids:
            continue
        key = (
            record["source_table_id"],
            int(record["source_row_id"]),
            normalize(record["attribute_name"]),
        )
        index.setdefault(key, []).append(record)
    return index


def _qualified_columns(
    table: dict[str, Any],
    entity_col: int,
    extraction_index: dict[tuple[str, int, str], list[dict[str, Any]]],
    config: BuildConfig,
) -> list[dict[str, Any]]:
    profiles = _profiles(table)
    candidates = [
        column["column_index"]
        for column in table["columns"]
        if column["column_index"] != entity_col
        and profiles[column["column_index"]]["non_empty_ratio"]
        >= config.min_column_non_empty_ratio
    ]
    qualified: list[dict[str, Any]] = []
    for attribute_col in candidates:
        attribute_name = get_column_name(table, attribute_col)
        valid_rows = [
            int(row["row_id"])
            for row in table["rows"]
            if clean_text(get_cell(row, entity_col).get("wiki_title"))
        ]
        recoveries: dict[int, list[dict[str, Any]]] = {}
        for row in table["rows"]:
            entity_cell = get_cell(row, entity_col)
            expected = clean_text(get_cell(row, attribute_col).get("text"))
            if not entity_cell.get("wiki_title") or not expected:
                continue
            source_row_id = int(row["row_id"])
            matches = [
                record
                for record in extraction_index.get(
                    (table["source_table_id"], source_row_id, normalize(attribute_name)), []
                )
                if values_match(record.get("value"), expected)
            ]
            if matches:
                recoveries[source_row_id] = matches

        required = max(
            config.min_recovered_rows,
            math.ceil(
                config.min_recovered_ratio
                * min(len(valid_rows), config.query_rows)
            ),
        )
        if len(valid_rows) >= config.query_rows and len(recoveries) >= required:
            qualified.append(
                {
                    "column_index": attribute_col,
                    "column_name": attribute_name,
                    "valid_rows": valid_rows,
                    "recoveries": recoveries,
                    "required_recovered_rows": required,
                    "recovered_rows": len(recoveries),
                    "recovered_value_ratio": len(recoveries) / len(valid_rows),
                }
            )
    return sorted(
        qualified,
        key=lambda item: (-item["recovered_value_ratio"], -item["recovered_rows"], item["column_index"]),
    )


def _select_query_rows(qualified: dict[str, Any], count: int) -> list[int]:
    recovered = sorted(qualified["recoveries"])
    remaining = [row_id for row_id in qualified["valid_rows"] if row_id not in qualified["recoveries"]]
    return (recovered + remaining)[:count]


def _materialize_join(
    table: dict[str, Any],
    split: str,
    entity_col: int,
    qualified: dict[str, Any],
    assets_by_id: dict[str, dict[str, Any]],
    config: BuildConfig,
    query_additional: list[int],
    target_additional: list[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    join_col = qualified["column_index"]
    query_columns = [entity_col, *query_additional]
    target_columns = [join_col, *target_additional]

    query_source_rows = _select_query_rows(qualified, config.query_rows)
    target_source_rows = [
        row["row_id"]
        for row in table["rows"]
        if clean_text(get_cell(row, join_col).get("text"))
    ]
    query_rows, query_source_rows = _project(table, query_columns, set(query_source_rows))
    target_rows, target_source_rows = _project(table, target_columns, set(target_source_rows))

    chain_id = f"chain_{stable_hash(table['source_table_id'], entity_col, join_col)}"
    query_id = f"query_{stable_hash(chain_id, 'query')}"
    target_id = f"target_{stable_hash(chain_id, 'target')}"
    selected_recovered_rows = sum(
        source_row_id in qualified["recoveries"]
        for source_row_id in query_source_rows
    )
    hidden_attribute = {
        "source_column_index": join_col,
        "column_name": qualified["column_name"],
        "role": "model_recoverable_join_column",
        "eligible_rows": len(qualified["valid_rows"]),
        "recovered_rows": selected_recovered_rows,
        "required_recovered_rows": qualified["required_recovered_rows"],
        "recovered_value_ratio": selected_recovered_rows / len(query_source_rows),
    }
    query = _table_record(
        table,
        table_id=query_id,
        role="query",
        split=split,
        column_indices=query_columns,
        rows=query_rows,
        source_row_ids=query_source_rows,
        extra={
            "chain_id": chain_id,
            "query_entity_col": entity_col,
            "query_entity_col_name": get_column_name(table, entity_col),
            "hidden_attributes": [hidden_attribute],
            "target_table_ids": [target_id],
        },
    )
    target = _table_record(
        table,
        table_id=target_id,
        role="target_data_lake_table",
        split=split,
        column_indices=target_columns,
        rows=target_rows,
        source_row_ids=target_source_rows,
        extra={
            "chain_id": chain_id,
            "join_col": join_col,
            "join_col_name": qualified["column_name"],
        },
    )
    qrel = {
        "query_table_id": query_id,
        "target_table_id": target_id,
        "data_lake_table_id": target_id,
        "rel": 3,
        "split": split,
        "chain_id": chain_id,
        "source_table_id": table["source_table_id"],
        "join_attribute": hidden_attribute,
        "reason": "model_recoverable_join_column",
    }

    query_row_by_source = {row["source_row_id"]: row["row_id"] for row in query_rows}
    target_row_by_source = {row["source_row_id"]: row["row_id"] for row in target_rows}
    source_rows = {row["row_id"]: row for row in table["rows"]}
    recoveries: list[dict[str, Any]] = []
    for source_row_id in query_source_rows:
        for extraction in qualified["recoveries"].get(source_row_id, []):
            asset = assets_by_id.get(extraction["asset_id"], {})
            recovery_id = "rec_" + stable_hash(query_id, source_row_id, extraction["asset_id"])
            recoveries.append(
                {
                    "recovery_id": recovery_id,
                    "query_table_id": query_id,
                    "target_table_id": target_id,
                    "source_table_id": table["source_table_id"],
                    "source_row_id": source_row_id,
                    "query_row_id": query_row_by_source[source_row_id],
                    "target_row_ids": [target_row_by_source[source_row_id]],
                    "split": split,
                    "query_entity": {
                        "text": clean_text(get_cell(source_rows[source_row_id], entity_col).get("text")),
                        "wiki_title": clean_text(get_cell(source_rows[source_row_id], entity_col).get("wiki_title")),
                    },
                    "recovered_attribute": {
                        "column_index": join_col,
                        "column_name": qualified["column_name"],
                        "value": clean_text(get_cell(source_rows[source_row_id], join_col).get("text")),
                        "model_value": clean_text(extraction.get("value")),
                        "hidden_in_query": True,
                    },
                    "evidence": {
                        "asset_id": extraction["asset_id"],
                        "asset_type": asset.get("asset_type"),
                        "model_evidence": clean_text(extraction.get("evidence")),
                    },
                }
            )
    return query, target, qrel, recoveries


def _build_joinability_for_table(
    table: dict[str, Any],
    *,
    assets_by_id: dict[str, dict[str, Any]],
    extraction_index: dict[tuple[str, int, str], list[dict[str, Any]]],
    split: str,
    config: BuildConfig,
) -> dict[str, list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    entity_col = _entity_column(table, config.query_rows)
    if entity_col is None:
        decisions.append(
            {"source_table_id": table["source_table_id"], "reason": "no_entity_column"}
        )
        return {
            "query_tables": queries,
            "data_lake_tables": targets,
            "qrels": qrels,
            "evidence_recoveries": recoveries,
            "table_queryability_decisions": decisions,
        }
    qualified = _qualified_columns(table, entity_col, extraction_index, config)
    if not qualified:
        decisions.append(
            {"source_table_id": table["source_table_id"], "reason": "no_recoverable_column"}
        )
        return {
            "query_tables": queries,
            "data_lake_tables": targets,
            "qrels": qrels,
            "evidence_recoveries": recoveries,
            "table_queryability_decisions": decisions,
        }

    emitted = 0
    visible_queries: set[str] = set()
    for candidate, query_additional, target_additional in _column_layouts(
        table, entity_col, qualified, config
    ):
        if sum(
            bool(clean_text(get_cell(row, candidate["column_index"]).get("text")))
            for row in table["rows"]
        ) < config.min_target_rows:
            continue
        query, target, qrel, paths = _materialize_join(
            table,
            split,
            entity_col,
            candidate,
            assets_by_id,
            config,
            query_additional,
            target_additional,
        )
        visible_key = stable_hash(
            query["columns"],
            [[cell["text"] for cell in row["cells"]] for row in query["rows"]],
            length=40,
        )
        if visible_key in visible_queries:
            continue
        visible_queries.add(visible_key)
        queries.append(query)
        targets.append(target)
        qrels.append(qrel)
        recoveries.extend(paths)
        emitted += 1
    decisions.append(
        {
            "source_table_id": table["source_table_id"],
            "reason": "queryable" if emitted else "target_too_small",
            "entity_column_index": entity_col,
            "qualified_columns": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"valid_rows", "recoveries"}
                }
                for candidate in qualified
            ],
        }
    )
    return {
        "query_tables": queries,
        "data_lake_tables": targets,
        "qrels": qrels,
        "evidence_recoveries": recoveries,
        "table_queryability_decisions": decisions,
    }


def build_joinability_for_table(
    table: dict[str, Any],
    assets: list[dict[str, Any]],
    extractions: list[dict[str, Any]],
    split: str,
    config: BuildConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Run the shared joinability algorithm for one normalized source table."""
    assets_by_id = {asset["asset_id"]: asset for asset in assets}
    extraction_index = _extraction_index(extractions, set(assets_by_id))
    return _build_joinability_for_table(
        table,
        assets_by_id=assets_by_id,
        extraction_index=extraction_index,
        split=split,
        config=config,
    )


def build_joinability_dataset(
    tables: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    extractions: list[dict[str, Any]],
    split_of: dict[str, str],
    config: BuildConfig,
) -> dict[str, list[dict[str, Any]]]:
    assets_by_id = {asset["asset_id"]: asset for asset in assets}
    extraction_index = _extraction_index(extractions, set(assets_by_id))
    artifacts = {
        "query_tables": [],
        "data_lake_tables": [],
        "qrels": [],
        "evidence_recoveries": [],
        "table_queryability_decisions": [],
    }
    for table in tables:
        table_artifacts = _build_joinability_for_table(
            table,
            assets_by_id=assets_by_id,
            extraction_index=extraction_index,
            split=split_of[table["source_table_id"]],
            config=config,
        )
        for artifact, records in table_artifacts.items():
            artifacts[artifact].extend(records)
    return artifacts


def table_asset_links(
    tables: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_by_title = {entity["wiki_title"]: entity for entity in entities}
    asset_ids_by_entity: dict[str, list[str]] = {}
    for asset in assets:
        asset_ids_by_entity.setdefault(asset["entity_id"], []).append(asset["asset_id"])
    links: list[dict[str, Any]] = []
    for table in tables:
        for row in table["rows"]:
            for cell in row["cells"]:
                entity = entity_by_title.get(clean_text(cell.get("wiki_title")))
                if entity:
                    links.append(
                        {
                            "source_table_id": table["source_table_id"],
                            "row_id": row["row_id"],
                            "column_index": cell["column_index"],
                            "entity_id": entity["entity_id"],
                            "asset_ids": asset_ids_by_entity.get(entity["entity_id"], []),
                        }
                    )
    return links

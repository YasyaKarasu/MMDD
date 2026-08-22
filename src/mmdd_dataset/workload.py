from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from .utils import clean_text, stable_hash


def _query_view(
    table: dict[str, Any], column_indices: list[int], strategy: str, ordinal: int
) -> dict[str, Any]:
    selected = set(column_indices)
    rows = []
    for source_row in table["rows"]:
        cells = []
        for local_index, source_index in enumerate(column_indices):
            source_cell = source_row["cells"][source_index]
            cells.append(
                {
                    **source_cell,
                    "column_index": local_index,
                    "source_column_index": source_index,
                }
            )
        rows.append(
            {
                "row_id": source_row["row_id"],
                "source_row_id": source_row["row_id"],
                "cells": cells,
            }
        )
    query_view_id = "qv_" + stable_hash(
        table["source_table_id"], strategy, ordinal, column_indices
    )
    return {
        "query_view_id": query_view_id,
        "source_table_id": table["source_table_id"],
        "page_title": table.get("page_title", ""),
        "caption": table.get("caption", ""),
        "derivation_type": strategy,
        "selected_column_indices": column_indices,
        "selected_column_names": [
            table["columns"][index]["column_name"] for index in column_indices
        ],
        "hidden_column_indices": [
            column["column_index"]
            for column in table["columns"]
            if column["column_index"] not in selected
        ],
        "hidden_column_names": [
            column["column_name"]
            for column in table["columns"]
            if column["column_index"] not in selected
        ],
        "rows": rows,
    }


def generate_query_views(
    table: dict[str, Any], max_views: int, seed: int
) -> list[dict[str, Any]]:
    entity_columns = list(table["metadata"]["candidate_entity_columns"])
    if not entity_columns or max_views <= 0:
        return []

    profiles = {
        profile["column_index"]: profile
        for profile in table["metadata"]["column_profiles"]
    }
    attributes = [
        column["column_index"]
        for column in table["columns"]
        if column["column_index"] not in entity_columns
        and profiles[column["column_index"]]["non_empty_ratio"] >= 0.5
        and profiles[column["column_index"]]["numeric_ratio"] <= 0.95
    ]
    rng = random.Random(f"{seed}:{table['source_table_id']}")
    rng.shuffle(entity_columns)
    rng.shuffle(attributes)

    plans: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()

    def add(strategy: str, indices: list[int]) -> None:
        key = tuple(indices)
        if key not in seen:
            seen.add(key)
            plans.append((strategy, indices))

    for entity_col in entity_columns:
        for attribute_col in attributes:
            add("entity_plus_one_attribute", [entity_col, attribute_col])
    for entity_col in entity_columns:
        for left in range(len(attributes)):
            for right in range(left + 1, len(attributes)):
                add(
                    "entity_plus_two_attributes",
                    [entity_col, attributes[left], attributes[right]],
                )
    for entity_col in entity_columns:
        add("entity_only", [entity_col])

    all_columns = range(table["num_cols"])
    random_projections = [
        list(indices)
        for width in range(2, min(4, table["num_cols"]) + 1)
        for indices in combinations(all_columns, width)
        if any(index in entity_columns for index in indices)
    ]
    rng.shuffle(random_projections)
    for indices in random_projections:
        add("random_projection", indices)

    return [
        _query_view(table, indices, strategy, ordinal)
        for ordinal, (strategy, indices) in enumerate(plans[:max_views], 1)
    ]


def query_view_asset_links(
    views: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entity_by_title = {entity["wiki_title"]: entity for entity in entities}
    assets_by_entity: dict[str, list[str]] = {}
    for asset in assets:
        assets_by_entity.setdefault(asset["entity_id"], []).append(asset["asset_id"])

    links: list[dict[str, Any]] = []
    for view in views:
        for row in view["rows"]:
            for cell in row["cells"]:
                entity = entity_by_title.get(clean_text(cell.get("wiki_title")))
                if entity:
                    links.append(
                        {
                            "query_view_id": view["query_view_id"],
                            "source_table_id": view["source_table_id"],
                            "row_id": row["row_id"],
                            "column_index": cell["column_index"],
                            "entity_id": entity["entity_id"],
                            "asset_ids": assets_by_entity.get(entity["entity_id"], []),
                        }
                    )
    return links

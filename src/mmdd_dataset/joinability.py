from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from mmdd_progress import progress

from .utils import (
    clean_text,
    get_cell,
    get_column_name,
    normalize,
    sanitize_cell_text,
    stable_hash,
    values_match,
)


JOINABILITY_POLICY_VERSION = "balanced_context_multi_positive_v2"
MIN_IMPLICIT_CONTEXT_COLUMNS = 2


@dataclass(frozen=True)
class BuildConfig:
    query_rows: int = 5
    min_target_rows: int = 5
    min_recovered_ratio: float = 0.6
    min_recovered_rows: int = 3
    min_column_non_empty_ratio: float = 0.5
    max_query_additional_columns: int = 1
    max_target_additional_columns: int = 2
    seed: int = 13


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
                    "text": sanitize_cell_text(source_cell.get("text")),
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
    split: str | None,
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_ids: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "table_id": table_id,
        "role": role,
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
    if split is not None:
        record["split"] = split
    return record


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
            profiles[index]["unique_ratio"],
            -index,
        ),
    )


def _balanced_context_partition(
    columns: list[int],
    *,
    seed: int,
    source_table_id: str,
) -> tuple[list[int], list[int]]:
    """Split ordinary columns once per source with a near-even random ratio."""
    shuffled = list(columns)
    rng = random.Random(f"context-pool-split:{seed}:{source_table_id}")
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    target_ratio = min(0.7, max(0.3, rng.gauss(0.5, 0.1)))
    target_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * target_ratio)),
    )
    return shuffled[target_count:], shuffled[:target_count]


def _shuffled_target_columns(
    join_col: int,
    target_context: list[int],
    *,
    seed: int,
    source_table_id: str,
) -> list[int]:
    columns = [join_col, *target_context]
    random.Random(
        f"target-column-order:{seed}:{source_table_id}:{join_col}"
    ).shuffle(columns)
    return columns


def _table_column_values(table: dict[str, Any]) -> dict[int, list[str]]:
    return {
        int(column["column_index"]): [
            get_cell(row, int(column["column_index"])).get("text")
            for row in table["rows"]
        ]
        for column in table["columns"]
    }


def _exact_redundancy_groups(
    values_by_column: dict[int, list[str]],
    *,
    value_serializer=None,
) -> list[list[int]]:
    """Return exact, row-aligned duplicate column groups.

    Values are hashed with row count, row index, and a byte-length prefix so
    empty cells and concatenation boundaries stay part of the equivalence
    relation. Detection is table-local and intentionally includes entity
    columns so alias groups can be handled by the planner.
    """
    buckets: dict[str, list[int]] = defaultdict(list)
    for raw_index, values in values_by_column.items():
        try:
            column_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        digest = hashlib.sha256()
        digest.update(len(values).to_bytes(8, "big"))
        for row_index, value in enumerate(values):
            if value_serializer is not None:
                value = value_serializer(value)
            encoded = ("" if value is None else str(value)).encode("utf-8")
            digest.update(row_index.to_bytes(8, "big"))
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        buckets[digest.hexdigest()].append(column_index)
    groups = [sorted(indices) for indices in buckets.values() if len(indices) >= 2]
    groups.sort(key=lambda members: members[0])
    return groups


def _redundancy_group_map(
    values_by_column: dict[int, list[str]],
    groups: list[list[int]],
) -> dict[int, tuple[int, ...]]:
    result: dict[int, tuple[int, ...]] = {}
    for group in groups:
        members = tuple(sorted(group))
        for index in members:
            if index in values_by_column:
                result[index] = members
    return result


def _target_context_for_member(
    target_context: list[int],
    *,
    seed: int,
    source_table_id: str,
    group_key: str,
    member_column_index: int,
    member_ordinal: int,
    group_size: int,
    excluded: set[int],
) -> list[int]:
    """Choose a stable member-specific target context.

    Sibling targets use different context subsets whenever the pool permits,
    while always keeping at least one ordinary context column.
    """
    context = [index for index in target_context if index not in excluded]
    if group_size > 1 and len(context) > 1:
        ordered = sorted(context)
        omit = member_ordinal % len(ordered)
        context = [value for index, value in enumerate(ordered) if index != omit]
    random.Random(
        f"target-context:{seed}:{source_table_id}:{group_key}:{member_column_index}"
    ).shuffle(context)
    return context


def _append_entity_url_column(
    query_rows: list[dict[str, Any]],
    table: dict[str, Any],
    entity_col: int,
) -> None:
    """Append a synthetic ``entity_url`` cell derived from each row's wiki_title.

    The URL is computed at build time from the entity cell's ``wiki_title``
    (no network I/O); rows without a wiki title get an empty string.
    """
    out_index = max((len(row.get("cells") or []) for row in query_rows), default=0)
    for row in query_rows:
        entity_cell = next(
            (
                cell
                for cell in row["cells"]
                if cell.get("source_column_index") == entity_col
            ),
            {},
        )
        wiki_title = clean_text(entity_cell.get("wiki_title"))
        url = (
            f"https://en.wikipedia.org/wiki/{quote(wiki_title.replace(' ', '_'))}"
            if wiki_title
            else ""
        )
        row["cells"].append(
            {
                "column_index": out_index,
                "source_column_index": -1,
                "column_name": "entity_url",
                "text": url,
                "synthetic": True,
            }
        )


def _visible_query_fingerprint(
    table: dict[str, Any],
    column_indices: list[int],
    rows: list[dict[str, Any]],
) -> str:
    return stable_hash(
        "visible-query",
        table["source_table_id"],
        column_indices,
        [row["source_row_id"] for row in rows],
        [
            [clean_text(cell.get("text")) for cell in row.get("cells", [])]
            for row in rows
        ],
    )


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
            expected = sanitize_cell_text(get_cell(row, attribute_col).get("text"))
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
    candidate: dict[str, Any],
    members: tuple[int, ...],
    group_key: str,
    assets_by_id: dict[str, dict[str, Any]],
    config: BuildConfig,
    query_additional: list[int],
    target_additional: list[int],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize one join family: one query plus 1..k physical targets.

    Exact-duplicate columns share the query bridge and fan out to a seeded
    subset of their physical members; each member gets its own chain, target
    table, qrel, and member-specific evidence recoveries.
    """
    source_table_id = table["source_table_id"]
    query_columns = [entity_col, *query_additional]

    fanout_rng = random.Random(
        f"implicit-target-fanout:{config.seed}:{split}:{source_table_id}:{group_key}"
    )
    target_count = fanout_rng.randint(1, len(members))
    shuffled_members = list(members)
    fanout_rng.shuffle(shuffled_members)
    target_members = shuffled_members[:target_count]

    query_source_rows = _select_query_rows(candidate, config.query_rows)
    query_rows, retained_query_rows = _project(table, query_columns, set(query_source_rows))
    _append_entity_url_column(query_rows, table, entity_col)
    query_fingerprint = _visible_query_fingerprint(table, query_columns, query_rows)
    query_id = f"query_{stable_hash(source_table_id, query_fingerprint)}"
    query_row_by_source = {row["source_row_id"]: row["row_id"] for row in query_rows}
    recovered_selected = sum(
        source_row_id in candidate["recoveries"]
        for source_row_id in retained_query_rows
    )
    member_names = {index: get_column_name(table, index) for index in members}
    source_rows_by_id = {row["row_id"]: row for row in table["rows"]}

    targets: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    for member_ordinal, member in enumerate(target_members):
        member_context = _target_context_for_member(
            target_additional,
            seed=config.seed,
            source_table_id=source_table_id,
            group_key=group_key,
            member_column_index=member,
            member_ordinal=member_ordinal,
            group_size=len(target_members),
            excluded={entity_col, *members},
        )
        target_columns = _shuffled_target_columns(
            member,
            member_context,
            seed=config.seed,
            source_table_id=source_table_id,
        )
        target_source_rows = [
            row["row_id"]
            for row in table["rows"]
            if clean_text(get_cell(row, member).get("text"))
        ]
        target_rows, target_source_rows = _project(table, target_columns, set(target_source_rows))
        if len(target_rows) < config.min_target_rows:
            continue
        chain_id = f"chain_{stable_hash(source_table_id, group_key, member)}"
        target_id = f"target_{stable_hash(chain_id, 'target')}"
        hidden_attribute = {
            "source_column_index": member,
            "column_name": member_names[member],
            "role": "model_recoverable_join_column",
            "eligible_rows": len(candidate["valid_rows"]),
            "recovered_rows": recovered_selected,
            "required_recovered_rows": candidate["required_recovered_rows"],
            "recovered_value_ratio": (
                recovered_selected / len(retained_query_rows)
                if retained_query_rows
                else 0.0
            ),
        }
        targets.append(
            _table_record(
                table,
                table_id=target_id,
                role="target_data_lake_table",
                split=None,
                column_indices=target_columns,
                rows=target_rows,
                source_row_ids=target_source_rows,
                extra={
                    "chain_id": chain_id,
                    "join_col": member,
                    "join_col_name": member_names[member],
                },
            )
        )
        qrels.append(
            {
                "query_table_id": query_id,
                "target_table_id": target_id,
                "rel": 3,
                "split": split,
                "chain_id": chain_id,
                "source_table_id": source_table_id,
                "join_attribute": hidden_attribute,
                "reason": "model_recoverable_join_column",
            }
        )
        target_row_by_source = {row["source_row_id"]: row["row_id"] for row in target_rows}
        seen_recovery_ids: set[str] = set()
        for source_row_id in retained_query_rows:
            for extraction in candidate["recoveries"].get(source_row_id, []):
                member_value = sanitize_cell_text(
                    get_cell(source_rows_by_id[source_row_id], member).get("text")
                )
                recovery_id = "rec_" + stable_hash(
                    query_id, target_id, source_row_id, extraction["asset_id"], member_value
                )
                if recovery_id in seen_recovery_ids:
                    continue
                seen_recovery_ids.add(recovery_id)
                asset = assets_by_id.get(extraction["asset_id"], {})
                recoveries.append(
                    {
                        "recovery_id": recovery_id,
                        "query_table_id": query_id,
                        "target_table_id": target_id,
                        "source_table_id": source_table_id,
                        "source_row_id": source_row_id,
                        "query_row_id": query_row_by_source[source_row_id],
                        "target_row_ids": [target_row_by_source[source_row_id]],
                        "split": split,
                        "query_entity": {
                            "text": clean_text(
                                get_cell(source_rows_by_id[source_row_id], entity_col).get("text")
                            ),
                            "wiki_title": clean_text(
                                get_cell(source_rows_by_id[source_row_id], entity_col).get("wiki_title")
                            ),
                        },
                        "recovered_attribute": {
                            "column_index": member,
                            "column_name": member_names[member],
                            "value": member_value,
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

    if not targets:
        return None, [], [], []
    query = _table_record(
        table,
        table_id=query_id,
        role="query",
        split=split,
        column_indices=query_columns,
        rows=query_rows,
        source_row_ids=retained_query_rows,
        extra={
            "chain_id": qrels[0]["chain_id"],
            "chain_ids": [qrel["chain_id"] for qrel in qrels],
            "query_entity_col": entity_col,
            "query_entity_col_name": get_column_name(table, entity_col),
            "hidden_attributes": [qrel["join_attribute"] for qrel in qrels],
            "target_table_ids": [qrel["target_table_id"] for qrel in qrels],
        },
    )
    query["columns"] = [
        *query["columns"],
        {
            "column_index": len(query["columns"]),
            "source_column_index": -1,
            "column_name": "entity_url",
        },
    ]
    return query, targets, qrels, recoveries


def _implicit_context_layout(
    table: dict[str, Any],
    entity_col: int,
    qualified: list[dict[str, Any]],
    config: BuildConfig,
) -> list[tuple[dict[str, Any], tuple[int, ...], str, list[int], list[int]]]:
    """Reserve two context columns and emit one variant per join family.

    Exact-duplicate physical columns share one query bridge; the returned
    variants carry the full member group for target fan-out. Weakest families
    are demoted into the shared context pool until it holds
    MIN_IMPLICIT_CONTEXT_COLUMNS columns; sources that cannot reach the floor
    emit nothing.
    """
    values_by_column = _table_column_values(table)
    groups = _exact_redundancy_groups(
        values_by_column, value_serializer=sanitize_cell_text
    )
    group_by_column = _redundancy_group_map(values_by_column, groups)

    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for candidate in qualified:
        index = int(candidate["column_index"])
        grouped[group_by_column.get(index, (index,))].append(candidate)

    variants: list[tuple[dict[str, Any], tuple[int, ...]]] = []
    for members, member_candidates in grouped.items():
        if entity_col in members:
            # Entity aliases cannot be hidden bridges; their columns stay
            # excluded from the context pools.
            continue
        representative = min(
            member_candidates,
            key=lambda item: (-item["recovered_value_ratio"], item["column_index"]),
        )
        variants.append((representative, members))

    ordered = sorted(
        variants,
        key=lambda item: (-item[0]["recovered_value_ratio"], item[0]["column_index"]),
    )
    excluded = {member for _candidate, members in ordered for member in members}
    excluded |= {
        member
        for members in group_by_column.values()
        if entity_col in members
        for member in members
    }
    ordinary = _rank_additional_columns(table, {entity_col, *excluded})
    emitted = list(ordered)
    for _candidate, members in reversed(ordered):
        if len(ordinary) >= MIN_IMPLICIT_CONTEXT_COLUMNS:
            break
        if len(emitted) == 1:
            continue  # never demote the only variant
        emitted = [item for item in emitted if item[1] != members]
        ordinary.extend(member for member in members if member not in ordinary)
    if len(ordinary) < MIN_IMPLICIT_CONTEXT_COLUMNS:
        return []
    query_additional, target_additional = _balanced_context_partition(
        ordinary,
        seed=config.seed,
        source_table_id=table["source_table_id"],
    )
    return [
        (
            candidate,
            members,
            stable_hash(
                "redundancy-group", table["source_table_id"], *members, length=24
            ),
            list(query_additional),
            list(target_additional),
        )
        for candidate, members in emitted
    ]


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

    layout = _implicit_context_layout(table, entity_col, qualified, config)
    if not layout:
        decisions.append(
            {
                "source_table_id": table["source_table_id"],
                "reason": "context_floor_unreachable",
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

    query_by_id: dict[str, dict[str, Any]] = {}
    qrel_keys: set[tuple[str, str, int]] = set()
    emitted_candidates: list[dict[str, Any]] = []
    for candidate, members, group_key, query_additional, target_additional in layout:
        query, member_targets, member_qrels, paths = _materialize_join(
            table,
            split,
            entity_col,
            candidate,
            members,
            group_key,
            assets_by_id,
            config,
            query_additional,
            target_additional,
        )
        if not member_targets:
            continue
        emitted_candidates.append(candidate)
        existing_query = query_by_id.get(query["table_id"])
        if existing_query is None:
            query_by_id[query["table_id"]] = query
            queries.append(query)
        else:
            existing_query["chain_ids"].extend(query["chain_ids"])
            existing_query["hidden_attributes"].extend(query["hidden_attributes"])
            existing_query["target_table_ids"].extend(query["target_table_ids"])
        for target, qrel in zip(member_targets, member_qrels):
            qrel_key = (
                qrel["query_table_id"],
                qrel["target_table_id"],
                int(qrel["join_attribute"]["source_column_index"]),
            )
            if qrel_key in qrel_keys:
                continue
            qrel_keys.add(qrel_key)
            targets.append(target)
            qrels.append(qrel)
        recoveries.extend(paths)
    decisions.append(
        {
            "source_table_id": table["source_table_id"],
            "reason": "queryable" if queries else "target_too_small",
            "entity_column_index": entity_col,
            "qualified_columns": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"valid_rows", "recoveries"}
                }
                for candidate in emitted_candidates
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
    for table in progress(tables, desc="Build joinability", unit="table"):
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

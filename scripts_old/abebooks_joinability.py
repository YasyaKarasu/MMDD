#!/usr/bin/env python
"""AbeBooks-native multimodal joinability construction.

Why this is not :mod:`mmdd_dataset.joinability`
-----------------------------------------------
The shared builder assumes a Wikipedia-shaped corpus: an *entity column* whose
cells carry a ``wiki_title``, entity ids that group bridge assets, and a
synthetic ``entity_url`` column appended to every query.  An AbeBooks lake has
none of that -- ``wiki_title`` is ``None`` in every cell, and the row-to-asset
link is a plain ``row_id`` that ``bridge_assets.jsonl`` already carries.  Running
the shared builder on a real lake table returns ``no_entity_column`` and zero
queries (measured).

Rather than grow a second provider branch inside the shared module, this file
ports the parts that differ and **imports** the parts that do not (the context
layout, the exact-redundancy fan-out, the record projection helpers), so the
provider-agnostic machinery keeps one definition.  Three deliberate divergences
from the shared materializer, all marked ``ABEBOOKS DIVERGENCE`` below:

1. no synthetic ``entity_url`` column (nothing downstream reads it by name);
2. ``query_entity`` carries ``column_name``/``source_column_index`` instead of
   ``wiki_title``;
3. ``query_columns`` depends on the join shape -- see :data:`JOIN_SHAPES`.

Two correctness guards have no counterpart in the shared module, because the
corpus never needed them:

``copy_channels``
    A text asset is normally a **verbatim copy** of the column it was scraped
    from (``synopsis`` -> ``synopsis_text``, 244/249 rows).  Asking a model to
    "recover" that column from that asset is a lookup, not a recovery, and
    scores a meaningless 100%.  Measured on the real lake, the problem is
    broader than the self-pair: ``synopsis`` content is *contained in*
    ``description`` for 245/249 rows, so the pair
    ``(description asset, synopsis_text column)`` is a near-lookup too -- which
    a ``source_column`` field alone cannot catch.  So the channels are derived
    from the data (equality *or* containment), not from metadata.

``context_leak_guard``
    Even with those channels closed, a hidden column's value can sit in a
    *visible* query column: ``copy_condition_grade`` is derived from the vendor
    description, and ``synopsis_text`` and ``vendor_description`` share text for
    244/249 rows.  Any query context column that contains the hidden column's
    value is moved to the target side.
"""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.joinability import (  # noqa: E402
    BuildConfig,
    MIN_IMPLICIT_CONTEXT_COLUMNS,
    _implicit_context_layout,
    _profiles,
    _project,
    _shuffled_target_columns,
    _table_record,
    _target_context_for_member,
    _visible_query_fingerprint,
)
from mmdd_dataset.utils import (  # noqa: E402
    clean_text,
    get_cell,
    get_column_name,
    normalize,
    sanitize_cell_text,
    stable_hash,
    values_match,
)

#: How a query is made un-joinable.  Both were built so the pilot can decide
#: between them on measured recovery rather than on argument.
#:
#: ``attribute``
#:     The canonical shape, and what the shared module emits: the entity column
#:     (``title``) stays visible in the query, and an ordinary column is hidden
#:     and becomes the join key.  A model can read ``publisher`` off a cover
#:     photograph to rejoin.
#:
#: ``identity``
#:     AbeBooks-native ablation: the *entity column itself* is hidden, so the
#:     query carries no title and a model must read the title off the pixels to
#:     join at all.  This is what a cover photograph is actually good for, and
#:     the shared module structurally cannot express it (it refuses to hide an
#:     entity-alias group).
JOIN_SHAPES = ("attribute", "identity")

#: Share of a column's non-empty rows that one value may occupy before the
#: column is considered too low-information to be worth hiding.  ``currency``,
#: ``language`` and ``shipping_currency`` have exactly one distinct value across
#: all 253 books; ``stock_image_flag`` two.  A model that emits the modal value
#: scores a perfect recovery on those while knowing nothing.
#:
#: Deliberately *not* "at least N distinct values": at the five-row grain that
#: rule also eliminates ``publisher`` in 44/50 tables (30 -> 6), and a cover
#: photograph is exactly what makes ``publisher`` recoverable.
MAX_VALUE_SHARE = 0.5

#: Share of rows on which an asset must reproduce a column's value (by equality
#: or containment) before the pair counts as a lookup rather than a recovery.
COPY_CHANNEL_THRESHOLD = 0.5

#: Shortest string allowed to count as a containment match.  Without a floor,
#: a one-character cell is "contained in" almost any prose and every column
#: would look like a copy channel.
MIN_CONTAINMENT_CHARS = 4

#: What a query-side context column may share with the hidden column before it
#: is treated as leaking the answer.
CONTEXT_LEAK_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# text normalisation for the data-derived guards
# --------------------------------------------------------------------------

def fold(value: str | None) -> str:
    """Casefolded, whitespace-collapsed text -- for comparing, never for output."""
    return " ".join(clean_text(value).casefold().split())


def overlaps(left: str, right: str) -> bool:
    """Whether one folded string contains the other (or they are equal).

    Containment, not equality, because the leaks are containments: the
    ``synopsis`` asset's text is a substring of the ``description`` asset's for
    245/249 rows, and a vendor description states the condition grade it was
    derived from.
    """
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return len(short) >= MIN_CONTAINMENT_CHARS and short in long


def usable_asset(asset: dict[str, Any]) -> bool:
    """Whether an asset can produce an extraction at all.

    An image with no downloaded file and a text asset with no content can only
    ever yield an empty answer, so they are dropped before any model is called.
    """
    if asset.get("asset_type") == "image":
        return bool(asset.get("local_path"))
    if asset.get("asset_type") == "text":
        return bool(clean_text(asset.get("content")))
    return False


def assets_by_row(assets: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Row id -> its usable assets.

    This replaces the shared builder's ``assets_by_entity``: ``bridge_assets.jsonl``
    already carries ``row_id``, so the row-to-asset link needs no entity layer.
    """
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        if usable_asset(asset):
            index[asset["row_id"]].append(asset)
    return dict(index)


def visible_cells(table: dict[str, Any], row: dict[str, Any], attribute_col: int) -> list[dict[str, str]]:
    """The row as a model sees it when asked for ``attribute_col``.

    Leave-one-out, as in the shared extractor: the target column is withheld so
    the model cannot read the answer off the row it is meant to enrich the row
    with.
    """
    return [
        {
            "name": get_column_name(table, column["column_index"]),
            "value": sanitize_cell_text(get_cell(row, column["column_index"]).get("text")),
        }
        for column in table["columns"]
        if column["column_index"] != attribute_col
        and clean_text(get_cell(row, column["column_index"]).get("text"))
    ]


def extraction_tasks(
    table: dict[str, Any],
    by_row: dict[str, list[dict[str, Any]]],
    entity_col: int,
    config: BuildConfig,
    channels: Iterable[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Every ``(row, asset, attribute)`` a model should be asked about.

    Replaces the shared ``build_extractions``, which reaches assets through an
    entity id.  Two rules do the work here:

    **text assets answer only other columns.**  A text asset is a copy of the
    column it was scraped from, so asking it for that column is a lookup, not a
    recovery.  The rule is enforced in both directions through ``channels``,
    which also catches the containment pairs that ``source_column`` misses.

    **images answer anything.**  A cover photograph has no source column: it can
    be read for any attribute of the book it depicts.

    The entity column is extracted under *both* join shapes, so the pilot can
    measure how well a cover yields the title before anyone commits to hiding it.
    A text asset with no ``source_column`` is a hard error rather than a silent
    skip: without it the copy rule cannot be applied, and quietly dropping the
    task would look like a low recovery rate instead of a stale lake.
    """
    profiles = _profiles(table)
    attributes = [
        column for column in table["columns"]
        if profiles[column["column_index"]]["non_empty_ratio"]
        >= config.min_column_non_empty_ratio
    ]
    blocked = set(channels)
    tasks: list[dict[str, Any]] = []
    for row in table["rows"]:
        source_row_id = row["row_id"]
        for asset in by_row.get(source_row_id, []):
            asset_type = asset["asset_type"]
            source_column = asset.get("source_column")
            if asset_type == "text" and not source_column:
                raise ValueError(
                    f"text asset {asset['asset_id']!r} has no source_column, so a "
                    "recovery cannot be told apart from a lookup. Rebuild the lake "
                    "with scripts_old/build_abebooks_lake.py."
                )
            for column in attributes:
                name = column["column_name"]
                if asset_type == "text":
                    if source_column == name:
                        continue                      # the asset *is* this column
                    if (asset["source"], name) in blocked:
                        continue                      # the asset contains this column
                tasks.append({
                    "extraction_id": "ext_" + stable_hash(
                        table["source_table_id"], source_row_id, asset["asset_id"], name
                    ),
                    "source_table_id": table["source_table_id"],
                    "source_row_id": source_row_id,
                    "asset_id": asset["asset_id"],
                    "asset_type": asset_type,
                    "asset_family": asset["source"],
                    "attribute_name": name,
                    "attribute_column_index": column["column_index"],
                    "is_entity_attribute": column["column_index"] == entity_col,
                })
    return tasks


# --------------------------------------------------------------------------
# the data-derived guards
# --------------------------------------------------------------------------

def copy_channels(
    tables: Iterable[dict[str, Any]],
    by_row: dict[str, list[dict[str, Any]]],
    *,
    threshold: float = COPY_CHANNEL_THRESHOLD,
) -> set[tuple[str, str]]:
    """``(asset family, column name)`` pairs where the asset already holds the value.

    Derived from the data rather than from ``source_column``, because the pairs
    that matter are not all self-pairs -- see the module docstring.  Costs one
    pass of string comparison over files already in memory, and is applied both
    when planning tasks (so no model call is wasted) and when indexing
    extractions (so a stale ``--extractions-jsonl`` cannot reopen a channel).
    """
    hits: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for table in tables:
        for column in table["columns"]:
            index = column["column_index"]
            name = column["column_name"]
            for row in table["rows"]:
                contents = {
                    asset["source"]: fold(asset.get("content"))
                    for asset in by_row.get(row["row_id"], [])
                    if asset["asset_type"] == "text"
                }
                value = fold(get_cell(row, index).get("text"))
                if not value or not contents:
                    continue
                for family, text in contents.items():
                    if not text:
                        continue
                    totals[(family, name)] += 1
                    if overlaps(value, text):
                        hits[(family, name)] += 1
    return {
        key for key, count in hits.items()
        if count / totals[key] >= threshold
    }


def value_overlap_share(
    table: dict[str, Any], left: int, right: int
) -> float:
    """Share of rows where two columns' cells contain one another."""
    shared = total = 0
    for row in table["rows"]:
        a = fold(get_cell(row, left).get("text"))
        b = fold(get_cell(row, right).get("text"))
        if not a or not b:
            continue
        total += 1
        if overlaps(a, b):
            shared += 1
    return shared / total if total else 0.0


def drop_uninformative(
    table: dict[str, Any],
    columns: list[int],
    *,
    floor: float,
) -> list[int]:
    """Remove context columns that carry nothing.

    The lake has columns that are empty on every row -- ``edition_marker``,
    ``condition_description`` and friends, named by a scraper spec that never
    filled them.  The shared layout ranks columns by ``non_empty_ratio`` and then
    shuffles the ranking away, so an empty column lands in the query about half
    the time.  On a five-row table that matters more than it sounds: the two-column
    context floor can be met by two empty columns, and the query then shows the
    model its title and nothing else, which is a different -- and much easier --
    task than the one the row is supposed to pose.
    """
    return [
        index for index in columns
        if _non_empty_ratio(table, index) >= floor
    ]


def _non_empty_ratio(table: dict[str, Any], column_index: int) -> float:
    if not table["rows"]:
        return 0.0
    filled = sum(1 for row in table["rows"]
                 if clean_text(get_cell(row, column_index).get("text")))
    return filled / len(table["rows"])


def context_leak_guard(
    table: dict[str, Any],
    hidden_column: int,
    query_additional: list[int],
    target_additional: list[int],
    *,
    threshold: float = CONTEXT_LEAK_THRESHOLD,
) -> tuple[list[int], list[int]]:
    """Move query context columns that contain the hidden value to the target side.

    Hiding ``synopsis_text`` while ``vendor_description`` stays visible in the
    query is not a join -- the answer is on screen.  Same for
    ``copy_condition_grade``, which is derived from the vendor description in the
    first place.
    """
    leaks = [
        index for index in query_additional
        if value_overlap_share(table, index, hidden_column) >= threshold
    ]
    if not leaks:
        return list(query_additional), list(target_additional)
    kept = [index for index in query_additional if index not in leaks]
    return kept, [*target_additional, *leaks]


# --------------------------------------------------------------------------
# screening
# --------------------------------------------------------------------------

def entity_column(table: dict[str, Any], query_rows: int) -> int | None:
    """The column that identifies a row, without a Wikipedia gate.

    The shared version counts cells carrying a ``wiki_title``; here a cell *is*
    the identity when it has text, so the count is of non-empty cells.  Ties go
    to the lowest index, and ``metadata.candidate_entity_columns`` still decides
    which columns may compete (so ``isbn13`` could never win even if present).
    """
    profiles = _profiles(table)
    candidates = [
        index for index in table["metadata"]["candidate_entity_columns"]
        if sum(bool(clean_text(get_cell(row, index).get("text"))) for row in table["rows"])
        >= query_rows
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda index: (profiles[index]["non_empty_ratio"], -index))


def max_value_share(table: dict[str, Any], column_index: int) -> float:
    """Share of the column's non-empty rows taken by its most common value."""
    values = [
        normalize(clean_text(get_cell(row, column_index).get("text")))
        for row in table["rows"]
    ]
    values = [value for value in values if value]
    if not values:
        return 1.0
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts.values()) / len(values)


def extraction_index(
    extractions: Iterable[dict[str, Any]],
    *,
    channels: Iterable[tuple[str, str]] = (),
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """``(source_table_id, source_row_id, normalised attribute)`` -> extractions.

    ``source_row_id`` stays a **string**: lake row ids are ``bk_0001``-style, and
    the shared builder's ``int()`` cast raises on them.
    """
    blocked = set(channels)
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in extractions:
        family = record.get("asset_family")
        if family is not None and (family, record["attribute_name"]) in blocked:
            continue
        index[(
            record["source_table_id"],
            str(record["source_row_id"]),
            normalize(record["attribute_name"]),
        )].append(record)
    return dict(index)


def qualified_columns(
    table: dict[str, Any],
    entity_col: int,
    index: dict[tuple[str, str, str], list[dict[str, Any]]],
    config: BuildConfig,
    *,
    include_entity: bool = False,
    max_value_share_limit: float = MAX_VALUE_SHARE,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Columns a model was actually able to recover, with the reason for the rest.

    Three gates, in order, and the second one is new:

    row gate
        The column needs ``min_target_rows`` non-empty rows, because target rows
        *are* the rows with a non-empty join cell.
    discrimination gate
        ``max_value_share <= limit``.  The shared module has no such guard, and
        this lake needs one: every lake profile sets ``numeric_ratio`` to 0.0, so
        the repo's only low-information heuristic is inert here.
    recovery gate
        The shared predicate, unchanged -- at least
        ``max(min_recovered_rows, ceil(ratio * min(valid_rows, query_rows)))``
        rows must have a matching extraction.
    """
    profiles = _profiles(table)
    candidates = [
        column["column_index"]
        for column in table["columns"]
        if profiles[column["column_index"]]["non_empty_ratio"] >= config.min_column_non_empty_ratio
        and (column["column_index"] != entity_col or include_entity)
    ]
    valid_rows = [
        row["row_id"] for row in table["rows"]
        if clean_text(get_cell(row, entity_col).get("text"))
    ]
    required = max(
        config.min_recovered_rows,
        math.ceil(config.min_recovered_ratio * min(len(valid_rows), config.query_rows)),
    )
    qualified: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for attribute_col in candidates:
        attribute_name = get_column_name(table, attribute_col)
        non_empty = sum(
            1 for row in table["rows"]
            if clean_text(get_cell(row, attribute_col).get("text"))
        )
        if non_empty < config.min_target_rows:
            rejected[attribute_name] = "target_too_small"
            continue
        if max_value_share(table, attribute_col) > max_value_share_limit:
            rejected[attribute_name] = "low_discrimination"
            continue
        recoveries: dict[str, list[dict[str, Any]]] = {}
        for row in table["rows"]:
            source_row_id = row["row_id"]
            expected = sanitize_cell_text(get_cell(row, attribute_col).get("text"))
            if not expected or not clean_text(get_cell(row, entity_col).get("text")):
                continue
            matches = [
                record
                for record in index.get(
                    (table["source_table_id"], source_row_id, normalize(attribute_name)), []
                )
                if values_match(record.get("value"), expected)
            ]
            if matches:
                recoveries[source_row_id] = matches
        if len(valid_rows) >= config.query_rows and len(recoveries) >= required:
            qualified.append({
                "column_index": attribute_col,
                "column_name": attribute_name,
                "valid_rows": valid_rows,
                "recoveries": recoveries,
                "required_recovered_rows": required,
                "recovered_rows": len(recoveries),
                "recovered_value_ratio": len(recoveries) / len(valid_rows),
            })
        else:
            rejected[attribute_name] = "recovery_below_threshold"
    qualified.sort(
        key=lambda item: (-item["recovered_value_ratio"], -item["recovered_rows"],
                          item["column_index"])
    )
    return qualified, rejected


def select_query_rows(
    candidate: dict[str, Any], count: int, join_rows: set[str]
) -> list[str]:
    """Query rows, restricted to rows the target table also carries.

    The shared version is ``(recovered + remaining)[:count]`` and ``remaining``
    may include rows whose join cell is empty -- which breaks the documented
    invariant ``set(query.source_row_indices) <= set(target.source_row_indices)``,
    because target rows are exactly the rows with a non-empty join cell.
    """
    recovered = [row for row in candidate["valid_rows"]
                 if row in candidate["recoveries"] and row in join_rows]
    remaining = [row for row in candidate["valid_rows"]
                 if row not in candidate["recoveries"] and row in join_rows]
    return (recovered + remaining)[:count]


# --------------------------------------------------------------------------
# materialisation
# --------------------------------------------------------------------------

def materialize_join(
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
    *,
    join_shape: str = "attribute",
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]]]:
    """One query plus 1..k physical targets for a single join family.

    A port of ``mmdd_dataset.joinability._materialize_join`` with three marked
    divergences; ``tests/test_abebooks_joinability.py`` pins it against the
    shared implementation so the two cannot drift apart unnoticed.
    """
    if join_shape not in JOIN_SHAPES:
        raise ValueError(f"unknown join shape: {join_shape!r}")
    source_table_id = table["source_table_id"]

    def query_columns_for(shape: str) -> list[int]:
        # ABEBOOKS DIVERGENCE 3: the identity shape hides the entity column, so
        # the query is built from context alone.
        if shape == "identity":
            return list(query_additional)
        return [entity_col, *query_additional]

    fanout_rng = random.Random(
        f"implicit-target-fanout:{config.seed}:{split}:{source_table_id}:{group_key}"
    )
    shuffled_members = list(members)
    fanout_rng.shuffle(shuffled_members)
    target_members = shuffled_members[:fanout_rng.randint(1, len(members))]

    member_names = {index: get_column_name(table, index) for index in members}
    source_rows_by_id = {row["row_id"]: row for row in table["rows"]}
    query_columns = query_columns_for(join_shape)
    query_source_rows = select_query_rows(
        candidate, config.query_rows, set(candidate["join_rows"])
    )
    query_rows, retained_query_rows = _project(table, query_columns, set(query_source_rows))
    # ABEBOOKS DIVERGENCE 1: no synthetic entity_url column.  Nothing downstream
    # reads it by name, and there is no entity URL to point at.
    query_fingerprint = _visible_query_fingerprint(table, query_columns, query_rows)
    query_id = f"query_{stable_hash(source_table_id, query_fingerprint)}"
    query_row_by_source = {row["source_row_id"]: row["row_id"] for row in query_rows}
    recovered_selected = sum(
        source_row_id in candidate["recoveries"] for source_row_id in retained_query_rows
    )

    targets: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    for ordinal, member in enumerate(target_members):
        member_context = _target_context_for_member(
            target_additional,
            seed=config.seed,
            source_table_id=source_table_id,
            group_key=group_key,
            member_column_index=member,
            member_ordinal=ordinal,
            group_size=len(target_members),
            excluded={entity_col, *members},
        )
        target_columns = _shuffled_target_columns(
            member, member_context, seed=config.seed, source_table_id=source_table_id
        )
        target_source_rows = [
            row["row_id"] for row in table["rows"]
            if clean_text(get_cell(row, member).get("text"))
        ]
        target_rows, target_source_rows = _project(
            table, target_columns, set(target_source_rows)
        )
        if len(target_rows) < config.min_target_rows:
            continue
        chain_id = f"chain_{stable_hash(source_table_id, group_key, member)}"
        target_id = f"target_{stable_hash(chain_id, 'target')}"
        targets.append(_table_record(
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
        ))
        hidden_attribute = {
            "source_column_index": member,
            "column_name": member_names[member],
            "role": "model_recoverable_join_column",
            "eligible_rows": len(candidate["valid_rows"]),
            "recovered_rows": recovered_selected,
            "required_recovered_rows": candidate["required_recovered_rows"],
            "recovered_value_ratio": (
                recovered_selected / len(retained_query_rows) if retained_query_rows else 0.0
            ),
        }
        qrels.append({
            "query_table_id": query_id,
            "target_table_id": target_id,
            "rel": 3,
            "split": split,
            "chain_id": chain_id,
            "source_table_id": source_table_id,
            "join_attribute": hidden_attribute,
            "reason": "model_recoverable_join_column",
        })
        target_row_by_source = {row["source_row_id"]: row["row_id"] for row in target_rows}
        seen: set[str] = set()
        for source_row_id in retained_query_rows:
            for extraction in candidate["recoveries"].get(source_row_id, []):
                member_value = sanitize_cell_text(
                    get_cell(source_rows_by_id[source_row_id], member).get("text")
                )
                recovery_id = "rec_" + stable_hash(
                    query_id, target_id, source_row_id, extraction["asset_id"], member_value
                )
                if recovery_id in seen:
                    continue
                seen.add(recovery_id)
                asset = assets_by_id.get(extraction["asset_id"], {})
                recoveries.append({
                    "recovery_id": recovery_id,
                    "query_table_id": query_id,
                    "target_table_id": target_id,
                    "source_table_id": source_table_id,
                    "source_row_id": source_row_id,
                    "query_row_id": query_row_by_source[source_row_id],
                    "target_row_ids": [target_row_by_source[source_row_id]],
                    "split": split,
                    "query_entity": {
                        # ABEBOOKS DIVERGENCE 2: no wiki_title to report.
                        "text": clean_text(
                            get_cell(source_rows_by_id[source_row_id], entity_col).get("text")
                        ),
                        "column_name": get_column_name(table, entity_col),
                        "source_column_index": entity_col,
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
                        "asset_family": asset.get("source"),
                        "model_evidence": clean_text(extraction.get("evidence")),
                    },
                })

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
    return query, targets, qrels, recoveries


def build_joinability_for_table(
    table: dict[str, Any],
    by_row: dict[str, list[dict[str, Any]]],
    index: dict[tuple[str, str, str], list[dict[str, Any]]],
    assets_by_id: dict[str, dict[str, Any]],
    config: BuildConfig,
    *,
    split: str,
    join_shape: str = "attribute",
    max_value_share_limit: float = MAX_VALUE_SHARE,
    leak_threshold: float = CONTEXT_LEAK_THRESHOLD,
    context_floor: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
           list[dict[str, Any]], list[dict[str, Any]]]:
    """Build every join family one lake table supports.

    Returns ``(decision, queries, targets, qrels, recoveries)``.  The decision
    record names the first gate that stopped the table, so a table that yields
    nothing says why instead of just being absent.
    """
    source_table_id = table["source_table_id"]

    def decision(reason: str, **extra: Any) -> dict[str, Any]:
        return {
            "source_table_id": source_table_id,
            "source_name": table.get("source_name"),
            "num_rows": table["num_rows"],
            "reason": reason,
            **extra,
        }

    entity_col = entity_column(table, config.query_rows)
    if entity_col is None:
        return decision("no_entity_column"), [], [], [], []

    qualified, rejected = qualified_columns(
        table, entity_col, index, config,
        include_entity=join_shape == "identity",
        max_value_share_limit=max_value_share_limit,
    )
    if not qualified:
        return decision("no_recoverable_column", rejected=rejected), [], [], [], []

    # The shared layout refuses to hide an entity-alias group, which is exactly
    # what the identity shape needs -- so it is asked to treat column -1 (which
    # matches nothing) as the entity and is handed only the column to hide.
    layout_entity = -1 if join_shape == "identity" else entity_col
    layout_candidates = (
        [item for item in qualified if item["column_index"] == entity_col]
        if join_shape == "identity" else qualified
    )
    variants = _implicit_context_layout(table, layout_entity, layout_candidates, config)
    if not variants:
        return decision("context_floor_unreachable", rejected=rejected), [], [], [], []

    queries: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    moved: list[str] = []
    dropped: list[str] = []
    emitted: list[dict[str, Any]] = []
    query_by_id: dict[str, dict[str, Any]] = {}
    qrel_keys: set[tuple[str, str, int]] = set()
    for candidate, members, group_key, query_additional, target_additional in variants:
        kept = drop_uninformative(table, query_additional, floor=context_floor)
        dropped.extend(
            get_column_name(table, index_) for index_ in query_additional
            if index_ not in kept
        )
        if len(kept) < MIN_IMPLICIT_CONTEXT_COLUMNS:
            # The floor was met by empty columns; the query would show the model
            # a title and no context at all, which is not the join being measured.
            continue
        guarded_query, guarded_target = context_leak_guard(
            table, members[0], kept, target_additional,
            threshold=leak_threshold,
        )
        moved.extend(
            get_column_name(table, index_)
            for index_ in query_additional if index_ not in guarded_query
        )
        candidate = {
            **candidate,
            # The row gate above already guarantees every valid row has a
            # non-empty cell in the hidden column, but naming it explicitly keeps
            # select_query_rows honest rather than relying on that.
            "join_rows": {row["row_id"] for row in table["rows"]
                          if clean_text(get_cell(row, members[0]).get("text"))},
        }
        query, family_targets, family_qrels, family_recoveries = materialize_join(
            table, split, entity_col, candidate, members, group_key, assets_by_id,
            config, guarded_query, guarded_target, join_shape=join_shape,
        )
        if not family_targets:
            continue
        emitted.append(candidate)
        # Two families can project to the same visible query when the hidden
        # columns are exact duplicates: one query, several chains.  Mirrors the
        # shared builder, including the qrel de-duplication key.
        existing = query_by_id.get(query["table_id"])
        if existing is None:
            query_by_id[query["table_id"]] = query
            queries.append(query)
        else:
            existing["chain_ids"].extend(query["chain_ids"])
            existing["hidden_attributes"].extend(query["hidden_attributes"])
            existing["target_table_ids"].extend(query["target_table_ids"])
        for target, qrel in zip(family_targets, family_qrels):
            key = (
                qrel["query_table_id"],
                qrel["target_table_id"],
                int(qrel["join_attribute"]["source_column_index"]),
            )
            if key in qrel_keys:
                continue
            qrel_keys.add(key)
            targets.append(target)
            qrels.append(qrel)
        recoveries.extend(family_recoveries)

    if not queries:
        return (decision("target_too_small", rejected=rejected), [], [], [], [])
    return (
        decision(
            "queryable",
            hidden_columns=sorted({item["column_name"] for item in emitted}),
            leak_guarded=sorted(set(moved)),
            empty_context_dropped=sorted(set(dropped)),
            rejected=rejected,
        ),
        queries, targets, qrels, recoveries,
    )

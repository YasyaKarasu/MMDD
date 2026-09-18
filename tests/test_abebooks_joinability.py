"""Tests for the AbeBooks-native joinability generator.

Two things make this generator different enough from the shared one to need its
own tests rather than a config flag: the join key here is a lake row id
(``bk_0001``, a string, not an index into an entity table), and the bridge assets
are verbatim copies of the columns they hang off.  The second is the dangerous
one -- an asset that already contains the answer turns "recovered from evidence"
into "read off the evidence", and inflates every number downstream.  Most of what
follows is aimed at that.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))
sys.path.insert(0, str(ROOT / "src"))

import abebooks_joinability as algo  # noqa: E402
import build_abebooks_joinability as builder  # noqa: E402
from mmdd_dataset.joinability import BuildConfig, _materialize_join  # noqa: E402

TABLE_ID = "st_book_001"
NUM_ROWS = 6
COLUMNS = ("title", "authors", "publisher", "currency", "synopsis_text",
           "vendor_description", "edition_marker", "dimensions")
TITLE, AUTHORS, PUBLISHER, CURRENCY, SYNOPSIS, VENDOR, EMPTY, DIMENSIONS = range(8)

#: The description is the synopsis plus a suffix, which is what the real lake
#: looks like: 245 of 249 rows have the synopsis contained in the description.
#: Neither is a copy of the other by ``source_column``, so only a containment
#: check catches the pair.
VENDOR_SUFFIX = " Extra detail from the seller."


def cell(index: int, text: str) -> dict[str, Any]:
    return {"column_index": index, "column_name": COLUMNS[index], "raw": text,
            "text": text, "wiki_title": None, "has_wiki_link": False}


def make_table(num_rows: int = NUM_ROWS) -> dict[str, Any]:
    rows = []
    for n in range(num_rows):
        cells = [
            cell(TITLE, f"Book {n}"),
            cell(AUTHORS, f"Author {n}"),
            cell(PUBLISHER, f"Publisher {n % 3}"),          # 2/6 share: keepable
            cell(CURRENCY, "USD"),                          # 6/6: low discrimination
            cell(SYNOPSIS, f"Synopsis {n}."),
            cell(VENDOR, f"Synopsis {n}.{VENDOR_SUFFIX}"),
            cell(EMPTY, ""),                                # never filled
            cell(DIMENSIONS, f"{n}x{n}" if n < 4 else ""),   # only 4 of 6 rows
        ]
        rows.append({"row_id": f"bk_{n + 1:04d}", "cells": cells})
    return {
        "source_table_id": TABLE_ID,
        "source_name": "book",
        "num_rows": num_rows,
        "num_cols": len(COLUMNS),
        "columns": [{"column_index": i, "column_name": name}
                    for i, name in enumerate(COLUMNS)],
        "rows": rows,
        "metadata": {
            "candidate_entity_columns": [TITLE],
            "column_profiles": [
                {
                    "column_index": i,
                    "non_empty_ratio": (
                        sum(1 for row in rows if row["cells"][i]["text"]) / num_rows
                    ),
                    "wiki_link_ratio": 0.0,
                    "unique_ratio": 1.0,
                    "numeric_ratio": 0.0,
                }
                for i in range(len(COLUMNS))
            ],
        },
    }


def make_assets(table: dict[str, Any], image_dir: Path) -> list[dict[str, Any]]:
    """One cover photograph and two text excerpts per row.

    The text assets are verbatim copies of their source columns -- that is what
    the lake writes, and it is why the copy-channel check has to exist.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    for row in table["rows"]:
        cells = {c["column_name"]: c["text"] for c in row["cells"]}
        picture = image_dir / f"{row['row_id']}.jpg"
        picture.write_bytes(b"\xff\xd8\xff\xd9")
        assets.append({
            "asset_id": f"ev:cover:{row['row_id']}", "asset_type": "image",
            "source": "abebooks_catalogue_cover", "source_column": None,
            "row_id": row["row_id"], "content": None, "url": None,
            "local_path": str(picture), "relative_path": picture.name,
        })
        for family, column in (("abebooks_description", "vendor_description"),
                               ("abebooks_synopsis", "synopsis_text")):
            assets.append({
                "asset_id": f"ev:{family}:{row['row_id']}", "asset_type": "text",
                "source": family, "source_column": column,
                "row_id": row["row_id"], "content": cells[column], "url": None,
                "local_path": None,
            })
    return assets


class Oracle:
    """A perfect extractor: answers every attribute with the row's own value.

    Deliberately stronger than any real model.  What is under test is the gates,
    not the model, so the stub must be able to clear them -- a column that is
    still rejected is rejected by the gate and not by a weak reader.
    """

    def __init__(self, table: dict[str, Any]):
        self.title_column = TITLE
        self.by_row = {
            row["row_id"]: {c["column_name"]: c["text"] for c in row["cells"]}
            for row in table["rows"]
        }
        self.calls = 0

    def extract(self, *, attribute: str, visible_cells: list[dict[str, str]],
                asset: dict[str, Any]) -> dict[str, str]:
        self.calls += 1
        values = self.by_row[asset["row_id"]]
        value = values.get(attribute, "")
        # The title is in the visible row; a real model reading a cover photo
        # could still report it, and the identity shape asks it to.
        if attribute == COLUMNS[self.title_column] and visible_cells:
            value = value or ""
        return {"value": value, "evidence": f"stub:{asset['asset_id']}"}


@pytest.fixture
def lake(tmp_path: Path) -> dict[str, Any]:
    table = make_table()
    assets = make_assets(table, tmp_path / "images")
    lake_dir = tmp_path / "lake"
    lake_dir.mkdir()
    _write_jsonl(lake_dir / "source_tables.jsonl", [table])
    _write_jsonl(lake_dir / "bridge_assets.jsonl", assets)
    return {"dir": lake_dir, "table": table, "assets": assets,
            "by_row": algo.assets_by_row(assets)}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line]


def config(**overrides: Any) -> BuildConfig:
    fields: dict[str, Any] = {"query_rows": 5, "min_target_rows": 5,
                             "min_recovered_ratio": 0.6, "min_recovered_rows": 3,
                             "min_column_non_empty_ratio": 0.5, "seed": 13}
    fields.update(overrides)
    return BuildConfig(**fields)


def run(lake: dict[str, Any], tmp_path: Path, *extra: str,
        extractor: Oracle | None = None) -> tuple[dict[str, Any], Oracle]:
    oracle = extractor or Oracle(lake["table"])
    args = builder.parser().parse_args([
        "--lake-dir", str(lake["dir"]), "--output-dir", str(tmp_path / "out"),
        "--limit-tables", "1", *extra])
    original = builder.build_extractors
    builder.build_extractors = lambda _args: {"text": oracle, "image": oracle}
    try:
        return builder.build(args), oracle
    finally:
        builder.build_extractors = original


# --------------------------------------------------------------------------
# task generation: what a model is never asked
# --------------------------------------------------------------------------

def test_text_asset_never_extracts_its_own_source_column(lake: dict[str, Any]) -> None:
    """A text asset is a verbatim copy of its column, so that pair is a lookup.

    Asking a model to read ``vendor_description`` out of an asset whose content
    *is* ``vendor_description`` scores 100% and measures nothing.
    """
    tasks = algo.extraction_tasks(lake["table"], lake["by_row"], TITLE, config())
    assert tasks, "the fixture produced no extraction tasks at all"
    for task in tasks:
        asset = next(a for a in lake["assets"] if a["asset_id"] == task["asset_id"])
        if asset["asset_type"] == "text":
            assert asset["source_column"] != task["attribute_name"], task
    assert not any(task["asset_family"] == "abebooks_description"
                   and task["attribute_name"] == "vendor_description" for task in tasks)
    assert not any(task["asset_family"] == "abebooks_synopsis"
                   and task["attribute_name"] == "synopsis_text" for task in tasks)


def test_copy_channel_detector_catches_containment_pairs(lake: dict[str, Any]) -> None:
    """The channel ``source_column`` cannot see.

    Every row's description is its synopsis plus a suffix, so reading the
    synopsis out of the description is still a lookup -- but the pair is
    ``(description, synopsis_text)`` and no self-column rule catches it.  On the
    real lake this is 245 of 249 rows.
    """
    channels = algo.copy_channels([lake["table"]], lake["by_row"])
    assert ("abebooks_description", "vendor_description") in channels   # equality
    assert ("abebooks_synopsis", "synopsis_text") in channels           # equality
    assert ("abebooks_description", "synopsis_text") in channels        # containment
    tasks = algo.extraction_tasks(lake["table"], lake["by_row"], TITLE, config(),
                                  channels)
    assert not any(task["asset_family"] == "abebooks_description"
                   and task["attribute_name"] == "synopsis_text" for task in tasks)


def test_image_asset_extracts_every_attribute(lake: dict[str, Any]) -> None:
    """A cover photograph is not a copy of any column, so nothing is blocked."""
    channels = algo.copy_channels([lake["table"]], lake["by_row"])
    tasks = algo.extraction_tasks(lake["table"], lake["by_row"], TITLE, config(), channels)
    images = [task for task in tasks if task["asset_type"] == "image"]
    assert images
    attributes = {task["attribute_name"] for task in images}
    assert attributes == {name for name in COLUMNS if name != "edition_marker"}
    entity = [task for task in images if task["is_entity_attribute"]]
    assert {task["attribute_name"] for task in entity} == {"title"}
    # A text task may still ask for the title: a synopsis or a seller's write-up
    # often names the book, and under the identity shape the title is the column
    # being hidden -- so this is evidence, not a lookup.
    assert any(task["attribute_name"] == "title" for task in tasks
               if task["asset_type"] == "text")


def test_the_same_tasks_are_generated_for_either_join_shape(
        lake: dict[str, Any]) -> None:
    """Shape-independence is what lets the pilot pick a shape off one cache.

    Both shapes are planned from one extraction pass -- the entity column is
    extracted even when the canonical shape cannot hide it.  That costs calls on
    a cold run and buys a cache that a later ``--join-shape`` switch still hits,
    which is the whole point of running the pilot first.
    """
    channels = algo.copy_channels([lake["table"]], lake["by_row"])
    ids = {task["extraction_id"]
           for task in algo.extraction_tasks(lake["table"], lake["by_row"], TITLE,
                                             config(), channels)}
    assert len(ids) == len(algo.extraction_tasks(lake["table"], lake["by_row"], TITLE,
                                                 config(), channels))
    assert any(task["is_entity_attribute"] for task in
               algo.extraction_tasks(lake["table"], lake["by_row"], TITLE, config(), channels))


def test_the_entity_column_is_hidden_only_under_the_identity_shape(
        lake: dict[str, Any]) -> None:
    """The canonical shape cannot hide the entity; the ablation exists to try."""
    index = algo.extraction_index([
        {"source_table_id": TABLE_ID, "source_row_id": row["row_id"],
         "attribute_name": name, "asset_id": "ev:x", "asset_family": "image",
         "value": row["cells"][column]["text"]}
        for row in lake["table"]["rows"]
        for column, name in enumerate(COLUMNS)
    ])
    canonical, _ = algo.qualified_columns(lake["table"], TITLE, index, config())
    assert "title" not in {item["column_name"] for item in canonical}
    identity, _ = algo.qualified_columns(lake["table"], TITLE, index, config(),
                                         include_entity=True)
    assert "title" in {item["column_name"] for item in identity}


def test_a_text_asset_with_no_source_column_is_an_error(lake: dict[str, Any]) -> None:
    """Fail closed on a stale lake rather than silently emitting lookup tasks."""
    stale = [{**asset, "source_column": None} if asset["asset_type"] == "text" else asset
             for asset in lake["assets"]]
    with pytest.raises(ValueError, match="source_column"):
        algo.extraction_tasks(lake["table"], algo.assets_by_row(stale), TITLE, config())


# --------------------------------------------------------------------------
# screening
# --------------------------------------------------------------------------

def test_low_cardinality_column_is_not_qualified(lake: dict[str, Any]) -> None:
    """``currency`` is ``USD`` on every row: the oracle's answer *is* the mode.

    The recovery gate is satisfied and the column is still worthless, which is
    why the discrimination gate is separate from it.  ``distinct >= 5`` would
    also reject it -- and would reject ``publisher``, the column a cover photo
    can actually answer, so the gate is a share and not a count.
    """
    index = algo.extraction_index([
        {"source_table_id": TABLE_ID, "source_row_id": row["row_id"],
         "attribute_name": name, "asset_id": "ev:x", "asset_family": "image",
         "value": row["cells"][column]["text"]}
        for row in lake["table"]["rows"]
        for column, name in enumerate(COLUMNS)
    ])
    qualified, rejected = algo.qualified_columns(lake["table"], TITLE, index, config())
    names = {item["column_name"] for item in qualified}
    assert "currency" not in names
    assert rejected["currency"] == "low_discrimination"
    assert "publisher" in names, "the gate must not be a distinct-value count"
    assert "edition_marker" not in names


def test_target_row_gate_rejects_a_column_filled_on_four_of_six_rows(
        lake: dict[str, Any]) -> None:
    """Target rows are the rows with a non-empty join cell: 4 < 5, so no target."""
    index = algo.extraction_index([
        {"source_table_id": TABLE_ID, "source_row_id": row["row_id"],
         "attribute_name": "dimensions", "asset_id": "ev:x", "asset_family": "image",
         "value": row["cells"][DIMENSIONS]["text"]}
        for row in lake["table"]["rows"] if row["cells"][DIMENSIONS]["text"]
    ])
    _qualified, rejected = algo.qualified_columns(lake["table"], TITLE, index, config())
    assert rejected["dimensions"] == "target_too_small"


def test_required_recovered_rows_matches_the_shared_formula(
        lake: dict[str, Any]) -> None:
    """``max(min_recovered_rows, ceil(ratio * min(valid_rows, query_rows)))``.

    Six valid rows, ``query_rows`` 5, ratio 0.6 -> ``ceil(3.0) == 3``.  Two
    recoverable rows are not enough and three are.
    """
    def qualified_with(recovered: int) -> list[dict[str, Any]]:
        index = algo.extraction_index([
            {"source_table_id": TABLE_ID, "source_row_id": row["row_id"],
             "attribute_name": "publisher", "asset_id": "ev:x",
             "asset_family": "image", "value": row["cells"][PUBLISHER]["text"]}
            for row in lake["table"]["rows"][:recovered]
        ])
        return algo.qualified_columns(lake["table"], TITLE, index, config())[0]

    assert all(item["required_recovered_rows"] == 3 for item in qualified_with(3))
    assert not qualified_with(2)
    assert qualified_with(3)


def test_extraction_index_keys_on_string_row_ids(lake: dict[str, Any]) -> None:
    """The shared index casts the row id with ``int()``, which a lake id breaks."""
    index = algo.extraction_index([
        {"source_table_id": TABLE_ID, "source_row_id": "bk_0001",
         "attribute_name": "Authors", "asset_id": "ev:x", "asset_family": "image",
         "value": "Author 0"}])
    assert (TABLE_ID, "bk_0001", "authors") in index


def test_context_leak_guard_moves_the_hidden_columns_source_to_the_target(
        lake: dict[str, Any]) -> None:
    """Hiding a column whose evidence sits visible in the query is not a join."""
    table = lake["table"]
    assert algo.value_overlap_share(table, VENDOR, SYNOPSIS) == 1.0
    kept, moved = algo.context_leak_guard(table, SYNOPSIS, [VENDOR, CURRENCY], [])
    assert kept == [CURRENCY]
    assert moved == [VENDOR]
    # An unrelated column is left alone.
    _, untouched = algo.context_leak_guard(table, PUBLISHER, [CURRENCY], [])
    assert untouched == []


def test_empty_context_columns_are_dropped_from_the_query(
        lake: dict[str, Any]) -> None:
    """A column empty on every row is not context; it is noise with a header.

    The lake has four of them, and the shared layout ranks columns by
    ``non_empty_ratio`` and then shuffles the ranking away, so an empty column
    lands in the query about half the time.  Two of them satisfy the two-column
    context floor, which would leave the model a title and nothing else.
    """
    kept = algo.drop_uninformative(lake["table"], [EMPTY, CURRENCY, DIMENSIONS],
                                   floor=0.5)
    assert kept == [CURRENCY, DIMENSIONS]
    assert algo.drop_uninformative(lake["table"], [EMPTY], floor=0.5) == []


# --------------------------------------------------------------------------
# the emitted records
# --------------------------------------------------------------------------

def test_a_query_row_is_always_a_target_row(lake: dict[str, Any], tmp_path: Path) -> None:
    """Documented invariant: target rows *are* the rows with a non-empty join cell."""
    run(lake, tmp_path)
    queries = {q["table_id"]: q for q in _read_jsonl(tmp_path / "out" / "query_tables.jsonl")}
    targets = {t["table_id"]: t for t in
               _read_jsonl(tmp_path / "out" / "data_lake_tables.jsonl")}
    qrels = _read_jsonl(tmp_path / "out" / "qrels.jsonl")
    assert qrels, "the fixture produced no qrels"
    for qrel in qrels:
        query = queries[qrel["query_table_id"]]
        target = targets[qrel["target_table_id"]]
        assert set(query["source_row_indices"]) <= set(target["source_row_indices"])
        assert len(query["rows"]) <= 5
        hidden = qrel["join_attribute"]["column_name"]
        assert hidden not in [c["column_name"] for c in query["columns"]]
        assert hidden in [c["column_name"] for c in target["columns"]]


def test_every_recovery_points_at_an_asset_that_reaches_the_row(
        lake: dict[str, Any], tmp_path: Path) -> None:
    run(lake, tmp_path)
    by_row = lake["by_row"]
    for recovery in _read_jsonl(tmp_path / "out" / "evidence_recoveries.jsonl"):
        asset = next(a for a in lake["assets"]
                     if a["asset_id"] == recovery["evidence"]["asset_id"])
        assert asset["row_id"] == recovery["source_row_id"]
        assert asset in by_row[recovery["source_row_id"]]
        # The value must come from evidence, not from the hidden cell.
        assert recovery["recovered_attribute"]["hidden_in_query"] is True


def test_no_recovery_reads_a_column_the_query_already_shows(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """The pilot's headline defect, as an invariant.

    A text asset is a copy of the column it was scraped from, so an asset can be
    asked about a *different* hidden column and answer it out of the visible row
    (``seller_policy`` -> ``city``, read off the displayed ``terms_of_sale``).
    Measured on the real pilot: 17 of 24 recoveries did this.  Whatever survives
    must come from evidence the query does not display.
    """
    run(lake, tmp_path)
    queries = {q["table_id"]: q
               for q in _read_jsonl(tmp_path / "out" / "query_tables.jsonl")}
    assets = {a["asset_id"]: a for a in lake["assets"]}
    checked = 0
    for recovery in _read_jsonl(tmp_path / "out" / "evidence_recoveries.jsonl"):
        query = queries[recovery["query_table_id"]]
        asset = assets[recovery["evidence"]["asset_id"]]
        if not asset["source_column"]:
            continue  # an image displays nothing; it is always independent
        shown = {c["column_name"] for c in query["columns"]}
        assert asset["source_column"] not in shown, (
            f"{recovery['evidence']['asset_family']} recovered "
            f"{recovery['recovered_attribute']['column_name']} out of "
            f"{asset['source_column']}, which the query displays"
        )
        checked += 1
    assert checked, "no text-derived recovery survived to check"


def test_evidence_that_is_a_visible_copy_is_flagged_rather_than_used(
        lake: dict[str, Any], tmp_path: Path) -> None:
    from abebooks_joinability import evidence_is_already_visible

    table = lake["table"]
    visible = [TITLE, VENDOR]          # the query shows title and vendor_description
    description = {"asset_type": "text", "source_column": "vendor_description"}
    cover = {"asset_type": "image", "source_column": None}
    assert evidence_is_already_visible(table, visible, description) is True
    assert evidence_is_already_visible(table, [TITLE], description) is False
    assert evidence_is_already_visible(table, visible, cover) is False


def test_a_table_dropped_as_a_visible_copy_says_which_column_it_copied(
        lake: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fail-closed drop has to be auditable, not merely counted.

    The reason code says a table was dropped; a reader still has to be able to
    see which hidden column could not be hidden and which displayed column its
    evidence kept being read out of.  ``dropped_as_visible_copy`` carries both --
    it is named for the hidden column, because that is what was lost.
    """
    monkeypatch.setattr(algo, "evidence_is_already_visible", lambda *a, **k: True)
    run(lake, tmp_path)
    decisions = _read_jsonl(tmp_path / "out" / "table_queryability_decisions.jsonl")
    dropped = [record for record in decisions
               if record["reason"] == "evidence_is_a_visible_copy"]
    assert dropped, "forcing the rule on flagged no table at all"
    flagged = dict(dropped[0]["dropped_as_visible_copy"])
    assert flagged, "the flagged table names no dropped column"
    for hidden, sources in flagged.items():
        assert hidden in COLUMNS, hidden
        # Empty is legitimate: a recovery with no source column is an image, and
        # the image is not what the query displays.  What must never happen is a
        # source that is not a column of this table.
        assert set(sources) <= set(COLUMNS), sources
    assert any(sources for _, sources in flagged.items()), (
        "no dropped column recorded where its evidence came from"
    )


def test_extraction_cache_reuses_a_pilot_run(lake: dict[str, Any], tmp_path: Path) -> None:
    """The pilot's cache has to be reusable, or step 3 of the plan pays twice.

    The attribute-extraction *check* is not cached and still calls the model --
    it is a second, independent look at each recovery (query's visible row plus
    raw evidence), so re-running it is the point rather than a waste.  What the
    cache buys is the far larger extraction pass.
    """
    cache = tmp_path / "cache" / "abe.jsonl"
    first_stats, first = run(lake, tmp_path / "one", "--extraction-cache", str(cache))
    assert first_stats["extractor_calls"] > 0 and cache.is_file()
    second_stats, second = run(lake, tmp_path / "two", "--extractions-jsonl", str(cache))
    assert second_stats["extractor_calls"] == 0
    assert second_stats["tasks_cached"] == second_stats["tasks_planned"] > 0
    assert second.calls < first.calls


def test_extractions_survive_a_crash_partway_through_the_pass(
    lake: dict[str, Any], tmp_path: Path
) -> None:
    """The full run is 119,729 calls: the cache cannot wait for the last one.

    A cache written only when the pass finishes throws away every hour that came
    before the crash, and the pass is over five hours long.  What has to hold is
    that the records already produced are on disk *and* are reused on the retry.
    """
    cache = tmp_path / "cache" / "abe.jsonl"

    class Exploding(Oracle):
        def extract(self, **kwargs: Any) -> dict[str, str]:
            if self.calls >= 20:
                raise RuntimeError("endpoint died")
            return super().extract(**kwargs)

    with pytest.raises(RuntimeError, match="endpoint died"):
        run(lake, tmp_path / "one", "--extraction-cache", str(cache),
            extractor=Exploding(lake["table"]))

    written = _read_jsonl(cache)
    assert written, "a crash mid-pass left the cache empty"
    assert all(record["extraction_id"] for record in written)

    retry, _ = run(lake, tmp_path / "two", "--extractions-jsonl", str(cache))
    assert 0 < retry["tasks_cached"] < retry["tasks_planned"], (
        "the retry did not pick the partial cache up, or the crash cached "
        "everything after all"
    )


def test_a_text_only_run_does_not_cache_its_image_tasks(
    lake: dict[str, Any], tmp_path: Path
) -> None:
    """Otherwise the text pilot silently cancels the image pilot.

    With no image extractor configured the image tasks are unasked, not
    unanswered.  Writing them to the shared cache as empty values makes the next
    run -- which passes the image endpoint and reuses the cache -- read them back
    as completed and skip every one of them.
    """
    cache = tmp_path / "cache" / "abe.jsonl"
    oracle = Oracle(lake["table"])
    original = builder.build_extractors
    builder.build_extractors = lambda _args: {"text": oracle, "image": None}
    try:
        args = builder.parser().parse_args([
            "--lake-dir", str(lake["dir"]), "--output-dir", str(tmp_path / "one"),
            "--limit-tables", "1", "--extraction-cache", str(cache)])
        text_stats = builder.build(args)
    finally:
        builder.build_extractors = original

    assert text_stats["tasks_skipped_no_extractor"] > 0
    cached = _read_jsonl(cache)
    assert cached, "the text run extracted nothing"
    assert all(record["asset_type"] == "text" for record in cached)

    image_pass, _ = run(lake, tmp_path / "two", "--extractions-jsonl", str(cache))
    # The text pass is reused; the image tasks are still pending, which is the
    # whole point of sharing the cache between the two pilot runs.
    assert 0 < image_pass["tasks_cached"] < image_pass["tasks_planned"]


def test_extraction_results_come_back_in_a_canonical_order(
    lake: dict[str, Any], tmp_path: Path
) -> None:
    """Two streams finish at different moments; the dataset must not notice.

    The extraction index keeps *every* record for a (row, attribute) key in list
    order, and that order reaches the emitted recoveries.  A merge in completion
    order therefore makes the artifacts depend on thread scheduling.  This was
    not theoretical: the reproducibility test above only failed when another
    test happened to run first and change the timing.
    """
    table = lake["table"]
    table_id = table["source_table_id"]
    tasks = algo.extraction_tasks(table, lake["by_row"], TITLE, config())
    args = builder.parser().parse_args(["--lake-dir", str(lake["dir"])])
    shuffled = list(tasks)
    random.Random(7).shuffle(shuffled)

    records = builder.run_extraction(
        shuffled,
        {table_id: table},
        {(table_id, row["row_id"]): row for row in table["rows"]},
        {asset["asset_id"]: asset for asset in lake["assets"]},
        {"text": Oracle(table), "image": Oracle(table)},
        args,
    )
    assert len(records) == len(tasks)
    order = [(record["source_table_id"], record["source_row_id"],
              record["attribute_name"], record["asset_id"]) for record in records]
    assert order == sorted(order), "extraction order follows completion, not the data"


def test_the_run_is_reproducible(lake: dict[str, Any], tmp_path: Path) -> None:
    run(lake, tmp_path / "a")
    run(lake, tmp_path / "b")
    for name in ("query_tables", "data_lake_tables", "qrels", "evidence_recoveries"):
        assert (_read_jsonl(tmp_path / "a" / "out" / f"{name}.jsonl")
                == _read_jsonl(tmp_path / "b" / "out" / f"{name}.jsonl")), name


# --------------------------------------------------------------------------
# contract with the shared implementation and with Stage 1
# --------------------------------------------------------------------------

def _strip_ids(record: dict[str, Any], ids: dict[str, str]) -> dict[str, Any]:
    return {key: (ids[value] if key in ids and isinstance(value, str) else value)
            for key, value in record.items()}


def test_emitted_records_match_the_shared_materialize_join(lake: dict[str, Any]) -> None:
    """The pin that keeps the port from drifting away from the original.

    ``materialize_join`` here is a port rather than a call because five things
    have to differ, and each is asserted below rather than assumed.  Everything
    else -- the column projections, the row selections, the target fan-out, the
    qrel and recovery contents -- must stay byte-identical, so both
    implementations are run on the same table and compared field by field.

    Ids are compared by position, not by value: the shared fingerprint hashes the
    cell texts of the query it built, which include the synthetic ``entity_url``
    cell it appends, so its ids cannot equal ours.  Mapping one to the other by
    ordinal keeps every other field under the pin.
    """
    table = lake["table"]
    cfg = config()
    channels = algo.copy_channels([table], lake["by_row"])
    tasks = algo.extraction_tasks(table, lake["by_row"], TITLE, cfg, channels)
    index = algo.extraction_index([
        {"source_table_id": TABLE_ID, "source_row_id": task["source_row_id"],
         "attribute_name": task["attribute_name"], "asset_id": task["asset_id"],
         "asset_family": task["asset_family"],
         "value": next(c["text"] for c in
                       next(r for r in table["rows"]
                            if r["row_id"] == task["source_row_id"])["cells"]
                       if c["column_name"] == task["attribute_name"])}
        for task in tasks])
    qualified, _ = algo.qualified_columns(table, TITLE, index, cfg)
    assert qualified, "no qualified column to materialize"
    # ``publisher`` is filled on every row, so the shared selector's ``remaining``
    # and ours pick the same query rows and the comparison is exact.
    candidate = next(item for item in qualified if item["column_name"] == "publisher")
    candidate = {**candidate, "join_rows": set(candidate["valid_rows"])}
    members = (candidate["column_index"],)
    query_additional = [CURRENCY]
    assets_by_id = {asset["asset_id"]: asset for asset in lake["assets"]}

    ours = algo.materialize_join(table, "train", TITLE, candidate, members,
                                 "group", assets_by_id, cfg, query_additional, [])
    theirs = _materialize_join(table, "train", TITLE, candidate, members,
                               "group", assets_by_id, cfg, query_additional, [])
    ours_query, ours_targets, ours_qrels, ours_recoveries = ours
    ref_query, ref_targets, ref_qrels, ref_recoveries = theirs
    assert ref_query is not None
    assert len(ours_targets) == len(ref_targets) > 0
    assert len(ours_recoveries) == len(ref_recoveries) > 0

    rename = {
        ref_query["table_id"]: ours_query["table_id"],
        **{ref["table_id"]: item["table_id"]
           for ref, item in zip(ref_targets, ours_targets)},
        **{ref["chain_id"]: item["chain_id"]
           for ref, item in zip(ref_targets, ours_targets)},
        **{ref["recovery_id"]: item["recovery_id"]
           for ref, item in zip(ref_recoveries, ours_recoveries)},
    }

    def normalize(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: (rename[value] if isinstance(value, str) and value in rename else value)
            for key, value in sorted(record.items())
        }

    # Divergences 1 and 2: the synthetic entity_url column and cell.
    assert ref_query["columns"][-1]["column_name"] == "entity_url"
    assert ours_query["columns"] == ref_query["columns"][:-1]
    ref_query["columns"] = ref_query["columns"][:-1]
    for row in ref_query["rows"]:
        assert row["cells"][-1]["column_name"] == "entity_url"
        row["cells"].pop()
    assert ours_query["rows"] == ref_query["rows"]

    # A target is a data-lake object rather than a split member, so neither side
    # carries a split.  Asserted rather than assumed so the port cannot quietly
    # reintroduce one.
    for ours_item, ref_item in zip(ours_targets, ref_targets):
        assert "split" not in ref_item
        assert "split" not in ours_item

    # Divergences 3 and 4: query_entity names a real column instead of a wiki
    # title, and evidence names the asset family.  Both are additive facts about
    # this data source that the shared record has nowhere to put.
    stripped = []
    for ours_item, ref_item in zip(ours_recoveries, ref_recoveries):
        assert ours_item["query_entity"] == {
            "text": ref_item["query_entity"]["text"],
            "column_name": "title",
            "source_column_index": TITLE,
        }
        assert ref_item["query_entity"]["wiki_title"] == ""
        assert ours_item["evidence"]["asset_family"]

        def without_family(record: dict[str, Any]) -> dict[str, Any]:
            return {**record,
                    "evidence": {k: v for k, v in record["evidence"].items()
                                 if k != "asset_family"}}

        stripped.append((
            without_family({k: v for k, v in ours_item.items() if k != "query_entity"}),
            without_family({k: v for k, v in ref_item.items() if k != "query_entity"}),
        ))
    for ours_item, ref_item in stripped:
        assert normalize(ours_item) == normalize(ref_item)

    # Divergence 5, and the pin: the query, the qrels and the targets are equal
    # field for field once the ids are aligned.
    assert normalize(ours_query) == normalize(ref_query)
    assert [normalize(item) for item in ours_qrels] == [normalize(item) for item in ref_qrels]
    assert [normalize(item) for item in ours_targets] == [normalize(item) for item in ref_targets]
    assert [normalize(item) for item in ours_query["hidden_attributes"]] \
        == [normalize(item) for item in ref_query["hidden_attributes"]]


def test_bridge_assets_are_loadable_by_construction(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """``_asset_object`` needs text or an *absolute* path to a real image.

    The lake writes ``local_path`` relative to the repository root, which
    ``construction.py`` resolves against the dataset root instead -- so the
    relative form misses and it raises.  The generator has to absolutise.
    """
    run(lake, tmp_path)
    from mmdd_stage1.construction import _asset_object
    root = tmp_path / "out"
    records = _read_jsonl(root / "bridge_assets.jsonl")
    assert records
    for record in records:
        obj = _asset_object(record, root)
        if record["asset_type"] == "image":
            assert Path(obj["image"]).is_absolute()
            assert Path(obj["image"]).is_file()
        else:
            assert obj["text"]


def test_the_manifest_lists_every_artifact_construction_reads(
        lake: dict[str, Any], tmp_path: Path) -> None:
    run(lake, tmp_path)
    root = tmp_path / "out"
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "mmdd_joinability_research_v2"
    assert "complete" not in manifest
    for artifact in ("query_tables", "data_lake_tables", "bridge_assets", "qrels",
                     "source_tables", "evidence_recoveries"):
        record = manifest["artifacts"][artifact]
        assert (root / record["path"]).is_file(), artifact
        assert record["records"] == len(_read_jsonl(root / record["path"])), artifact
    assert manifest["query_construction"]["join_shape"] == "attribute"
    assert manifest["query_construction"]["image_local_path_policy"] == "absolute"


def test_the_split_summary_declares_a_query_only_shared_lake(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """Train/dev/test scope the queries; the lake is one shared corpus.

    ``construction.py`` retrieves over every lake table for every query, so a
    builder that partitioned the lake would disagree with Stage 1 while still
    looking correct.
    """
    run(lake, tmp_path)
    root = tmp_path / "out"
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    query_tables = _read_jsonl(root / "query_tables.jsonl")
    lake_tables = _read_jsonl(root / "data_lake_tables.jsonl")
    assert manifest["split_schema_version"] == "query-only-shared-data-lake-v1"
    assert splits["split_policy"] == "query_only"
    assert splits["data_lake_scope"] == "shared"
    assert "data_lake_table_ids" not in splits
    assert splits["data_lake_artifact"] == "data_lake_tables"
    assert splits["data_lake_table_count"] == len(lake_tables)
    assert sum(splits["query_table_counts"].values()) == len(query_tables)
    assert all("split" not in record for record in lake_tables)
    assert all(record["split"] in {"train", "dev", "test"}
               for record in query_tables)


def test_tables_that_yield_no_query_are_still_lake_candidates(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """``construction.py`` needs two candidates per query to build a negative.

    A handful of queries over a sixty-table lake leaves it almost nothing to
    sample, so every source table is also offered as a rows-free reference that
    the consumer expands through ``source_table_ref``.  The second table here
    carries no assets at all, so it can never yield a query and exists only to
    be sampled as a negative.
    """
    barren = make_table(num_rows=5)
    barren = {**barren, "source_table_id": "st_book_002"}
    barren["rows"] = [
        {"row_id": f"bk_{n + 100:04d}", "cells": row["cells"]}
        for n, row in enumerate(barren["rows"])
    ]
    _write_jsonl(lake["dir"] / "source_tables.jsonl", [lake["table"], barren])

    stats = run(lake, tmp_path, "--limit-tables", "2")[0]
    root = tmp_path / "out"
    assert stats["raw_data_lake_refs"] is True
    decisions = {record["source_table_id"]: record["reason"]
                 for record in _read_jsonl(root / "table_queryability_decisions.jsonl")}
    assert decisions["st_book_002"] == "no_recoverable_column"

    refs = [record for record in _read_jsonl(root / "data_lake_tables.jsonl")
            if record.get("source_table_ref")]
    assert [ref["source_table_ref"]["source_table_id"] for ref in refs] == ["st_book_002"]
    for ref in refs:
        assert ref["queryable"] is False
        assert ref["source_table_ref"]["artifact"] == "source_tables"
        assert "rows" not in ref and "columns" not in ref
    source_ids = {record["source_table_id"]
                  for record in _read_jsonl(root / "source_tables.jsonl")}
    assert source_ids == {"st_book_001", "st_book_002"}


def test_a_column_that_yields_no_query_is_not_also_a_raw_candidate(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """The reference is for tables nothing was drawn from, not a duplicate offer."""
    run(lake, tmp_path)
    targets = _read_jsonl(tmp_path / "out" / "data_lake_tables.jsonl")
    assert targets
    assert not any(record.get("source_table_ref") for record in targets)


def test_an_image_with_no_file_is_dropped_rather_than_fatal(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """One failed download must not abort the whole dataset.

    The real lake has an image asset with ``local_path: null``.  Passed through,
    it raises inside ``_asset_object`` and takes every other record with it --
    for a photograph that could not be read in the first place.
    """
    assets = list(lake["assets"])
    covers = [i for i, asset in enumerate(assets) if asset["asset_type"] == "image"]
    broken = assets[covers[0]]["asset_id"]
    assets[covers[0]] = {**assets[covers[0]], "local_path": None}  # failed download
    gone = assets[covers[1]]["asset_id"]
    assets[covers[1]] = {**assets[covers[1]], "local_path": str(tmp_path / "gone.jpg")}
    _write_jsonl(lake["dir"] / "bridge_assets.jsonl", assets)

    stats = run(lake, tmp_path)[0]
    assert stats["images_dropped_missing_file"] == 2
    kept = _read_jsonl(tmp_path / "out" / "bridge_assets.jsonl")
    assert {asset["asset_id"] for asset in kept} == {
        asset["asset_id"] for asset in assets} - {broken, gone}
    for asset in kept:
        if asset["asset_type"] == "image":
            assert Path(asset["local_path"]).is_absolute()
            assert Path(asset["local_path"]).is_file()


def test_construction_consumes_the_generated_dataset(
        lake: dict[str, Any], tmp_path: Path) -> None:
    """The only test that matters if it fails: Stage 1 can actually read this."""
    from mmdd_stage1.construction import build_stage1_training_artifacts
    run(lake, tmp_path)
    artifacts = build_stage1_training_artifacts(tmp_path / "out",
                                                dataset_name="abebooks-test",
                                                max_rows=6)
    assert artifacts["target_lists"]
    object_ids = {record["object_id"] for record in artifacts["stage1_objects"]}
    for edges in artifacts["edge_lists"]:
        for edge in edges.get("edges", []):
            assert edge["source_id"] in object_ids
            assert edge["target_id"] in object_ids

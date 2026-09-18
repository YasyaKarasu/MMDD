#!/usr/bin/env python
"""Build an AbeBooks multimodal-joinability dataset from the two-table lake.

Reads ``source_tables.jsonl`` and ``bridge_assets.jsonl`` from the lake and writes
the artifacts Stage-1 consumes, in the same format as the EntiTables and WDC
datasets (``mmdd_joinability_research_v2``): queries whose join column has been
hidden, the target tables that carry it, the qrels that pair them, and the
evidence recoveries that justify each pair.

The algorithm lives in ``abebooks_joinability``; this module is the entry point --
argument parsing, model wiring, and artifact writing.  ``build(args)`` is a pure
function of its namespace, so the whole pipeline runs under ``tmp_path`` with a
stub extractor in the tests.

Two things this does that the shared pipeline cannot:

* **No entity layer.**  Assets reach rows through the ``row_id`` that
  ``bridge_assets.jsonl`` already carries, so there is no Wikipedia grounding
  step and no entity table.
* **The attribute-extraction check is mandatory by default.**  An extraction
  that is not independently re-confirmed from the query's *visible* row and the
  raw evidence does not count toward a column's recovery rate, and any qrel left
  below its required row count is dropped.  This mirrors ``review_mode="local"``
  in ``scripts_old/build_mm_joinability_dataset.py``
  (``AUTO_CHECK_REVIEW_POLICY_LOCAL``).  ``--no-auto-check`` turns it off and is
  recorded in the manifest when used, because the resulting recovery counts mean
  something different.

Usage::

    # no model calls: which tables could yield a query, and over which evidence
    python scripts_old/build_abebooks_joinability.py --plan-only

    # a five-table pilot
    python scripts_old/build_abebooks_joinability.py --limit-tables 5 \\
        --text-model-base-url http://127.0.0.1:8001/v1 --text-model-name Qwen3.5-9B \\
        --extraction-cache cache/abe.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# Running this file directly puts ``scripts_old`` on ``sys.path``, not ``src``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mmdd_dataset.extraction import OpenAICompatibleExtractor, auto_check_recoveries
from mmdd_dataset.joinability import (
    JOINABILITY_POLICY_VERSION,
    MIN_IMPLICIT_CONTEXT_COLUMNS,
    BuildConfig,
)
from mmdd_dataset.utils import (
    SPLIT_SCHEMA_VERSION,
    clean_text,
    get_cell,
    read_jsonl,
    source_splits,
    stable_hash,
    write_json,
    write_jsonl,
)
from mmdd_dataset.wdc_runtime import bounded_map
from mmdd_progress import progress

from abebooks_joinability import (
    JOIN_SHAPES,
    MAX_VALUE_SHARE,
    assets_by_row,
    build_joinability_for_table,
    copy_channels,
    entity_column,
    extraction_index,
    extraction_tasks,
    max_value_share,
    qualified_columns,
    visible_cells,
)

PROMPT_VERSION = "abebooks_leave_one_attribute_out_v1"

ARTIFACTS = (
    "query_tables",
    "data_lake_tables",
    "bridge_assets",
    "qrels",
    "evidence_recoveries",
    "table_queryability_decisions",
    "source_tables",
)

#: Attributes a lake table declares as possible identities.
ENTITY_COLUMN_OF = {"book": "title", "seller": "seller_name"}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class AbeBooksExtractor(OpenAICompatibleExtractor):
    """The shared client with an AbeBooks prompt.

    The inherited prompt is written for Wikipedia and asks the model to "verify
    that the evidence itself explicitly and unambiguously refers to that
    entity".  That under-elicits here: the row's identity is its title, which is
    already in the visible cells, and the evidence is a photograph of the very
    book being described rather than a document about a named entity.

    What the prompt has to prevent instead is a plausible guess.  An empty answer
    is cheap -- it is checked and discarded.  A guess is expensive: it is
    recorded as a successful recovery and silently inflates the recoverability of
    whatever column was being tested.
    """

    def extract(self, *, attribute: str, visible_cells: list[dict[str, str]],
                asset: dict[str, Any]) -> dict[str, str]:
        prompt = (
            "You are reading evidence about a second-hand book listing. The evidence "
            "is either a photograph of the physical book (its cover, title page, or a "
            "seller's own photo of the copy) or a text excerpt about it (a seller's "
            "description, a synopsis, an author biography, or a shipping or returns "
            "policy).\n"
            f"Report this attribute: {attribute}\n\n"
            "The attribute's own column has been removed from the row below, so it "
            "appears nowhere in the row and can only come from the evidence. Answer "
            "only from what the evidence actually shows or states. If the evidence "
            "does not show or state this attribute -- a photograph of a different "
            "book, an illegible image, or a policy excerpt that says nothing about "
            "the book -- return an empty value. An empty answer is expected and "
            "checked; a plausible guess is recorded as a success and corrupts the "
            "measurement.\n"
            "Return JSON only, as {\"value\": \"...\", \"evidence\": \"...\"}, where "
            "evidence quotes the words, or names the part of the image, that you read "
            "the value from.\n\n"
            f"Row (this attribute is absent from it): "
            f"{json.dumps(visible_cells, ensure_ascii=False)}\n"
        )
        content: str | list[dict[str, Any]] = prompt + "Evidence:\n" + clean_text(asset.get("content"))
        if asset["asset_type"] == "image":
            path = Path(asset["local_path"])
            media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
            ]
        response = self.session.post(
            self.url,
            json=self.request_body(content),
            timeout=self.timeout,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        if isinstance(answer, list):
            answer = "".join(item.get("text", "") for item in answer if isinstance(item, dict))
        start = answer.find("{")
        parsed: dict[str, Any] = {}
        if start >= 0:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(answer[start:])
            except json.JSONDecodeError:
                parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        return {
            "value": clean_text(parsed.get("value")),
            "evidence": clean_text(parsed.get("evidence")),
        }


def build_extractors(args: argparse.Namespace) -> dict[str, Any]:
    def make(base_url: str | None, model: str | None, key_env: str,
             max_tokens: int) -> Any:
        if not base_url or not model:
            return None
        return AbeBooksExtractor(base_url, model, api_key_env=key_env,
                                 timeout=args.model_timeout,
                                 max_tokens=max_tokens)

    return {
        "text": make(args.text_model_base_url, args.text_model_name,
                     args.text_api_key_env, args.text_model_max_tokens),
        "image": make(args.image_model_base_url, args.image_model_name,
                      args.image_api_key_env, args.image_model_max_tokens),
    }


# --------------------------------------------------------------------------
# lake loading
# --------------------------------------------------------------------------

def limit_tables(tables: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    """Take ``count`` whole tables, round-robin across sources.

    Whole tables, never rows: every gate downstream is a property of a table's
    composition (five-row minimum, modal-value share), so a row-level subset
    would measure something else.  Round-robin so a five-table pilot still sees
    both book and seller tables instead of five books.
    """
    if not count or count >= len(tables):
        return list(tables)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for table in tables:
        buckets[table.get("source_name") or "?"].append(table)
    names = sorted(buckets)
    out: list[dict[str, Any]] = []
    while len(out) < count and any(buckets[name] for name in names):
        for name in names:
            if buckets[name] and len(out) < count:
                out.append(buckets[name].pop(0))
    return out


def resolve_images(
    assets: Iterable[dict[str, Any]], image_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Absolutise image paths and drop the ones with no file to read.

    Two separate problems, one pass.  The lake writes ``local_path`` relative to
    the repository root, but ``construction.py`` resolves a relative path against
    the *dataset* root, so both of its fallbacks miss and it raises
    ``FileNotFoundError`` -- the generator has to absolutise.

    And a handful of downloads failed, leaving ``local_path: null`` (or a path
    with nothing behind it).  One such asset aborts ``build_stage1_training_artifacts``
    for the whole dataset, which is a bad trade for a photograph that cannot be
    read anyway, so they are dropped here and counted.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for asset in assets:
        if asset.get("asset_type") != "image":
            kept.append(asset)
            continue
        local_path = asset.get("local_path")
        candidate = Path(local_path) if local_path else None
        if candidate is not None and not candidate.is_absolute():
            candidate = image_root / candidate
        if candidate is not None and candidate.is_file():
            kept.append({**asset, "local_path": str(candidate.resolve())})
        else:
            dropped.append(asset)
    return kept, dropped


def load_lake(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                 list[dict[str, Any]]]:
    lake = Path(args.lake_dir)
    tables = list(read_jsonl(lake / "source_tables.jsonl"))
    assets, dropped = resolve_images(
        read_jsonl(lake / "bridge_assets.jsonl"), Path(args.image_root).resolve())
    return limit_tables(tables, args.limit_tables), assets, dropped


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def plan_tasks(
    tables: list[dict[str, Any]],
    by_row: dict[str, list[dict[str, Any]]],
    config: BuildConfig,
    channels: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tasks for every table that has an entity column, plus its own id.

    A table with no entity column can never yield a query, so no model is called
    for it.
    """
    tasks: list[dict[str, Any]] = []
    entity_of: dict[str, int] = {}
    for table in tables:
        entity_col = entity_column(table, config.query_rows)
        if entity_col is None:
            continue
        entity_of[table["source_table_id"]] = entity_col
        tasks.extend(extraction_tasks(table, by_row, entity_col, config, channels))
    return tasks, entity_of


def run_extraction(
    tasks: list[dict[str, Any]],
    tables_by_id: dict[str, dict[str, Any]],
    row_index: dict[tuple[str, str], dict[str, Any]],
    asset_by_id: dict[str, dict[str, Any]],
    extractors: dict[str, Any],
    args: argparse.Namespace,
    *,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every pending task, handing each finished record to ``on_record``.

    ``on_record`` exists for the full run: 119,729 calls is over five hours of
    wall clock, and a cache that is only written when the pass *ends* throws all
    of it away if anything dies at hour five.  ``bounded_map`` already yields a
    batch at a time, so the callback is called as the work completes and a
    restart resumes from the cache instead of from zero.

    Text and image run as two streams with separate worker budgets rather than
    one flat pool.  The endpoint's budget is per modality -- 104 text, 32 image,
    128 total -- and a flat pool sized for the total sends every task it holds at
    the image budget's expense the moment the tail of the pass is image-only.
    """
    def one(task: dict[str, Any]) -> dict[str, Any]:
        record = {
            "extraction_id": task["extraction_id"],
            "source_table_id": task["source_table_id"],
            "source_row_id": task["source_row_id"],
            "asset_id": task["asset_id"],
            "asset_type": task["asset_type"],
            "asset_family": task["asset_family"],
            "attribute_name": task["attribute_name"],
            "value": "",
            "evidence": "",
            "prompt_version": PROMPT_VERSION,
        }
        extractor = extractors.get(task["asset_type"])
        if extractor is None:
            # Run the other modality first and this task is not merely unsolved,
            # it is *unasked*.  Returning the empty record anyway would append it
            # to --extraction-cache, where the next run reads it back as a
            # completed extraction and skips the task forever -- so the image
            # pilot would silently do nothing after the text pilot.  Report it as
            # attempted (None) and let the caller drop it.
            return None
        table = tables_by_id[task["source_table_id"]]
        row = row_index[(task["source_table_id"], task["source_row_id"])]
        result = extractor.extract(
            attribute=task["attribute_name"],
            visible_cells=visible_cells(table, row, task["attribute_column_index"]),
            asset=asset_by_id[task["asset_id"]],
        )
        return {**record, **result}

    produced: list[dict[str, Any]] = []
    lock = threading.Lock()

    def drain(asset_type: str, workers: int) -> list[dict[str, Any]]:
        batch = [task for task in tasks if task["asset_type"] == asset_type]
        if not batch:
            return []
        collected: list[dict[str, Any]] = []
        for record in progress(
            bounded_map(one, batch, workers=workers),
            total=len(batch), desc=f"Extract {asset_type}", unit="call",
        ):
            if record is None:
                continue
            collected.append(record)
            # Written on completion, not at the end: the cache has to survive a
            # crash five hours in.  Locked because both streams share the handle
            # and a line has to land whole.
            with lock:
                if on_record is not None:
                    on_record(record)
        return collected

    budgets = [("text", args.text_model_workers), ("image", args.image_model_workers)]
    active = [stream for stream in budgets if any(t["asset_type"] == stream[0]
                                                 for t in tasks)]
    if not active:
        return produced
    if len(active) == 1:
        produced = drain(*active[0])
    else:
        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            futures = [pool.submit(drain, *stream) for stream in active]
            for future in futures:
                produced.extend(future.result())
    # Canonical order, not completion order.  The index keeps *every* extraction
    # for a (row, attribute) key in list order, and that order reaches the
    # emitted recoveries, so two streams finishing at different moments would
    # otherwise hand back a different dataset per run.
    produced.sort(key=lambda record: (
        record["source_table_id"], record["source_row_id"],
        record["attribute_name"], record["asset_id"],
    ))
    return produced


def load_cached(paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    for name in paths:
        candidate = Path(name)
        if candidate.exists():
            cached.update({
                record["extraction_id"]: record for record in read_jsonl(candidate)
            })
    return cached


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def plan_report(
    tables: list[dict[str, Any]],
    by_row: dict[str, list[dict[str, Any]]],
    config: BuildConfig,
    channels: set[tuple[str, str]],
    *,
    max_value_share_limit: float,
) -> dict[str, Any]:
    """What the run would do, without calling a model.

    The point of this is to answer "is there anything here to measure?" before
    paying for several thousand model calls.  A column can only be hidden and
    recovered if evidence *other than a copy of itself* actually reaches the rows
    it lives on, so for every column that clears the structural gates this
    reports which families are open channels to it and which are copies.  A
    column whose every channel is a copy, on a table with no images, cannot be
    measured no matter what the model does.
    """
    per_table: list[dict[str, Any]] = []
    for table in tables:
        entity_col = entity_column(table, config.query_rows)
        if entity_col is None:
            per_table.append({"source_table_id": table["source_table_id"],
                              "source_name": table.get("source_name"),
                              "reason": "no_entity_column"})
            continue
        present: set[str] = set()
        for row in table["rows"]:
            present.update(asset["source"] for asset in by_row.get(row["row_id"], ()))
        rows: list[dict[str, Any]] = []
        for column in table["columns"]:
            index = column["column_index"]
            if index == entity_col or not clears_structural(
                    table, index, config, max_value_share_limit):
                continue
            name = column["column_name"]
            rows.append({
                "column": name,
                "families": sorted(present),
                "copy_families": sorted(f for f in present if (f, name) in channels),
                "open_families": sorted(f for f in present if (f, name) not in channels),
                "max_value_share": round(max_value_share(table, index), 2),
                "target_rows": non_empty_rows(table, index),
            })
        per_table.append({
            "source_table_id": table["source_table_id"],
            "source_name": table.get("source_name"),
            "num_rows": table["num_rows"],
            "entity_column": table["columns"][entity_col]["column_name"],
            "columns": rows,
            "measurable": [row["column"] for row in rows if row["open_families"]],
            "unmeasurable": [row["column"] for row in rows if not row["open_families"]],
        })
    return {"tables": per_table, "copy_channels": sorted(f"{f}->{c}" for f, c in channels)}


def non_empty_rows(table: dict[str, Any], column_index: int) -> int:
    return sum(1 for row in table["rows"]
               if clean_text(get_cell(row, column_index).get("text")))


def clears_structural(table: dict[str, Any], column_index: int, config: BuildConfig,
                      max_value_share_limit: float) -> bool:
    """The two gates that need no model: enough rows, and not one modal value.

    Neither is a proxy for the other.  The row gate is what a five-row table
    makes expensive -- the join column has to be non-empty on essentially all of
    them.  The discrimination gate kills a column like ``currency``, where every
    row shares a value and any model that emits that value scores 100%.
    """
    return (non_empty_rows(table, column_index) >= config.min_target_rows
            and max_value_share(table, column_index) <= max_value_share_limit)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(args: argparse.Namespace) -> dict[str, Any]:
    tables, assets, dropped_images = load_lake(args)
    by_row = assets_by_row(assets)
    asset_by_id = {asset["asset_id"]: asset for asset in assets}
    channels = copy_channels(tables, by_row)

    config = BuildConfig(
        query_rows=args.query_rows,
        min_target_rows=args.min_target_rows,
        min_recovered_ratio=args.min_recovered_ratio,
        min_recovered_rows=args.min_recovered_rows,
        min_column_non_empty_ratio=args.min_column_non_empty_ratio,
        seed=args.seed,
    )

    if args.plan_only:
        report = plan_report(tables, by_row, config, channels,
                             max_value_share_limit=args.max_value_share)
        report["images_dropped_missing_file"] = len(dropped_images)
        report["extraction_tasks"] = len(
            plan_tasks(tables, by_row, config, channels)[0])
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return {"plan_only": True, "tables": len(tables)}

    tasks, entity_of = plan_tasks(tables, by_row, config, channels)
    tables_by_id = {table["source_table_id"]: table for table in tables}
    row_index = {
        (table["source_table_id"], row["row_id"]): row
        for table in tables for row in table["rows"]
    }
    cached = load_cached([*args.extractions_jsonl, *([args.extraction_cache]
                                                      if args.extraction_cache else [])])
    pending = [task for task in tasks if task["extraction_id"] not in cached]

    extractors = build_extractors(args)
    if pending and not any(extractors.values()):
        raise SystemExit(
            f"{len(pending)} extraction tasks pending but no model endpoint is "
            "configured; pass --text-model-base-url/--text-model-name (and the "
            "image pair), or --plan-only to just report")
    # Append as the work completes rather than at the end of the pass: only the
    # records that were actually produced are written (a task whose extractor is
    # missing returns None and is never cached), and a crash an hour in keeps the
    # hour.
    cache_handle = None
    if args.extraction_cache:
        cache_path = Path(args.extraction_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = cache_path.open("a", encoding="utf-8")

    def append_to_cache(record: dict[str, Any]) -> None:
        assert cache_handle is not None
        cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        cache_handle.flush()

    try:
        produced = run_extraction(pending, tables_by_id, row_index, asset_by_id,
                                  extractors, args,
                                  on_record=append_to_cache if cache_handle else None)
    finally:
        if cache_handle is not None:
            cache_handle.close()
    skipped_no_extractor = len(pending) - len(produced)

    extractions = [*cached.values(), *produced]
    index = extraction_index(extractions, channels=channels)

    splits, split_of = source_splits(tables, ratios=tuple(args.split_ratio), seed=args.seed)
    artifacts: dict[str, list[dict[str, Any]]] = {
        "query_tables": [], "data_lake_tables": [], "qrels": [],
        "evidence_recoveries": [], "table_queryability_decisions": [],
    }
    reasons: dict[str, int] = defaultdict(int)
    for table in progress(tables, desc="Build joins", unit="table"):
        decision, queries, targets, qrels, recoveries = build_joinability_for_table(
            table, by_row, index, asset_by_id, config,
            split=split_of[table["source_table_id"]],
            join_shape=args.join_shape,
            max_value_share_limit=args.max_value_share,
            leak_threshold=args.max_context_leak_ratio,
            context_floor=args.min_context_non_empty_ratio,
        )
        reasons[decision["reason"]] += 1
        artifacts["table_queryability_decisions"].append(decision)
        artifacts["query_tables"].extend(queries)
        artifacts["data_lake_tables"].extend(targets)
        artifacts["qrels"].extend(qrels)
        artifacts["evidence_recoveries"].extend(recoveries)

    if not args.no_auto_check:
        if extractors["text"] is None:
            raise SystemExit(
                "the attribute-extraction check needs the text model; pass "
                "--text-model-base-url/--text-model-name, or --no-auto-check to "
                "record that the recovery counts were not checked")
        # Mandatory by default: an extraction is not evidence of recoverability
        # until it is re-confirmed from the query's visible row and the raw
        # evidence.  Fail-closed -- qrels below their required row count drop.
        artifacts = auto_check_recoveries(
            artifacts, assets, extractors["text"],
            image_extractor=extractors["image"], review_mode="local",
        )

    artifacts["bridge_assets"] = assets
    artifacts["source_tables"] = _source_tables(tables, assets)
    if args.raw_data_lake_refs:
        artifacts["data_lake_tables"] = _with_raw_refs(
            tables, artifacts["data_lake_tables"]
        )

    query_table_counts = {"train": 0, "dev": 0, "test": 0}
    for query in artifacts["query_tables"]:
        query_table_counts[str(query["split"])] += 1
    splits["query_table_counts"] = query_table_counts
    splits["data_lake_table_count"] = len(artifacts["data_lake_tables"])
    splits["data_lake_artifact"] = "data_lake_tables"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        artifact: write_jsonl(out_dir / f"{artifact}.jsonl", artifacts[artifact])
        for artifact in ARTIFACTS
    }
    hidden: dict[str, int] = defaultdict(int)
    for qrel in artifacts["qrels"]:
        hidden[qrel["join_attribute"]["column_name"]] += 1
    stats = {
        **counts,
        "join_shape": args.join_shape,
        "tables_by_source": dict(_count_by(tables, "source_name")),
        "decisions": dict(sorted(reasons.items())),
        "hidden_columns": dict(sorted(hidden.items())),
        "tasks_planned": len(tasks),
        "tasks_cached": len(cached),
        "tasks_skipped_no_extractor": skipped_no_extractor,
        "extractions_used": len(extractions),
        "extractions_empty": sum(1 for record in extractions if not record.get("value")),
        "extractor_calls": len(pending) - skipped_no_extractor,
        "images_dropped_missing_file": len(dropped_images),
        "copy_channels": sorted(f"{family}->{name}" for family, name in channels),
        "auto_check": "skipped" if args.no_auto_check else "local",
        "raw_data_lake_refs": bool(args.raw_data_lake_refs),
        "queries_by_split": dict(query_table_counts),
    }
    write_json(out_dir / "splits.json", splits)
    write_json(out_dir / "stats.json", stats)
    write_json(out_dir / "dataset_manifest.json", {
        "format": "mmdd_joinability_research_v2",
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_lake": str(args.lake_dir),
        "query_construction": {
            "policy_version": JOINABILITY_POLICY_VERSION,
            "provider": "abebooks",
            "join_shape": args.join_shape,
            "entity_layer": "none_row_id_keyed_assets",
            "copy_channel_policy": "derived_from_data_equality_or_containment",
            "discrimination_gate": f"max_value_share<={args.max_value_share}",
            "context_leak_guard": f"containment_share<{args.max_context_leak_ratio}",
            "visible_context_floor": f"non_empty_ratio>={args.min_context_non_empty_ratio}",
            "attribute_extraction_check": (
                "skipped" if args.no_auto_check else "local_model_fail_closed"
            ),
            "raw_data_lake_refs": bool(args.raw_data_lake_refs),
            "row_id_scheme": "string_positional",
            "image_local_path_policy": "absolute",
            "identical_visible_query_policy": "multiple_positive_targets",
            "min_implicit_context_columns": MIN_IMPLICIT_CONTEXT_COLUMNS,
            "cell_text_policy": "clean_and_truncate_1024",
        },
        # The resolved model config lives here on purpose: probing a pipeline
        # weeks later, the run's own command line is usually gone (scrollback
        # buffer, shell history), and the gates cannot be reinterpreted without
        # knowing which endpoint and which decoding settings produced the
        # extractions.
        "model": {
            "text": _model_record(args, args.text_model_base_url, args.text_model_name,
                                  args.text_model_max_tokens),
            "image": _model_record(args, args.image_model_base_url, args.image_model_name,
                                   args.image_model_max_tokens),
            "prompt_version": PROMPT_VERSION,
            "thinking_policy": "enable_thinking=false_on_every_call",
            "workers": {"text": args.text_model_workers,
                        "image": args.image_model_workers},
            "timeout_seconds": args.model_timeout,
        },
        "artifact_references": (
            {"data_lake_tables": {"field": "source_table_ref",
                                  "target_artifact": "source_tables",
                                  "resolution": "stream_by_source_table_id"}}
            if args.raw_data_lake_refs else {}
        ),
        "artifacts": {
            artifact: {"path": f"{artifact}.jsonl", "records": counts[artifact]}
            for artifact in ARTIFACTS
        },
        "single_files": {"splits": "splits.json", "stats": "stats.json"},
        "note": (
            "Read the shards this manifest lists; stale files from older runs may "
            "exist in the same directory."
        ),
    })
    return stats


def _model_record(args: argparse.Namespace, base_url: str | None, model: str | None,
                  max_tokens: int) -> dict[str, Any]:
    if not base_url or not model:
        return {"enabled": False}
    return {
        "enabled": True,
        "base_url": base_url,
        "model": model,
        "max_tokens": max_tokens,
    }


def _count_by(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.get(field) or "?"] += 1
    return dict(counts)


def _source_tables(tables: list[dict[str, Any]], assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The lake tables as an artifact, for expansion of the raw data-lake refs."""
    del assets
    return [
        {
            "source_table_id": table["source_table_id"],
            "source_name": table.get("source_name"),
            "columns": table["columns"],
            "rows": table["rows"],
            "num_rows": table["num_rows"],
            "num_cols": table["num_cols"],
        }
        for table in tables
    ]


def _with_raw_refs(
    tables: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add every lake table as a lightweight data-lake candidate.

    ``construction.py`` needs at least two candidates per query to build a
    negative, and a handful of queries over sixty objects leaves it almost
    nothing to sample.  A reference carries no rows -- the consumer expands it
    through ``source_table_ref`` -- so this costs a line per table.
    """
    present = {target["source_table_id"] for target in targets}
    extra = [
        {
            "table_id": "dl_raw_" + stable_hash("raw", table["source_table_id"]),
            "object_id": "dl_raw_" + stable_hash("raw", table["source_table_id"]),
            "object_type": "table",
            "role": "raw_data_lake_table",
            "source_table_id": table["source_table_id"],
            "source_table_ref": {"artifact": "source_tables",
                                 "source_table_id": table["source_table_id"]},
            "queryable": False,
            "reason": "no_query_emitted_for_source_table",
        }
        for table in tables if table["source_table_id"] not in present
    ]
    return [*targets, *extra]


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--lake-dir", default="output/abebooks_lake")
    result.add_argument("--output-dir", default="output/abebooks_joinability")
    result.add_argument("--image-root", default=".",
                        help="root the lake's relative image paths resolve against")
    result.add_argument("--join-shape", choices=JOIN_SHAPES, default="attribute",
                        help="'attribute' hides an ordinary column and keeps the "
                             "entity visible; 'identity' hides the entity column "
                             "itself, so the title must be read off the pixels")
    result.add_argument("--limit-tables", type=int,
                        help="build only this many whole tables (round-robin by source)")
    result.add_argument("--plan-only", action="store_true",
                        help="report the gates and the copy channels without calling a model")

    result.add_argument("--query-rows", type=int, default=5)
    result.add_argument("--min-target-rows", type=int, default=5)
    result.add_argument("--min-recovered-ratio", type=float, default=0.6)
    result.add_argument("--min-recovered-rows", type=int, default=3)
    result.add_argument("--min-column-non-empty-ratio", type=float, default=0.5)
    result.add_argument("--max-value-share", type=float, default=MAX_VALUE_SHARE,
                        help="reject a column whose modal value fills more of it than "
                             "this; low-cardinality columns are 'recovered' by any "
                             "model that emits the mode")
    result.add_argument("--max-context-leak-ratio", type=float, default=0.5,
                        help="move a query context column to the target side when it "
                             "shares this much of the hidden column's value")
    result.add_argument("--min-context-non-empty-ratio", type=float, default=0.5,
                        help="drop a query context column filled on less than this "
                             "share of rows; the lake has columns that are empty on "
                             "every row, and two of them can satisfy the two-column "
                             "context floor while showing the model nothing")
    result.add_argument("--seed", type=int, default=13)
    result.add_argument("--split-ratio", type=float, nargs=3,
                        default=(0.8, 0.1, 0.1), metavar=("TRAIN", "DEV", "TEST"))

    result.add_argument("--extractions-jsonl", action="append", default=[],
                        help="reuse these extraction records (repeatable)")
    result.add_argument("--extraction-cache", help="append new extractions here")
    result.add_argument("--text-model-base-url")
    result.add_argument("--text-model-name")
    result.add_argument("--text-api-key-env", default="VLLM_API_KEY")
    result.add_argument("--image-model-base-url")
    result.add_argument("--image-model-name")
    result.add_argument("--image-api-key-env", default="VLLM_API_KEY")
    result.add_argument(
        "--text-model-workers", type=int,
        default=104,
        help="in-flight text requests; default is the budget the endpoint "
             "declares in configs/model_endpoints.qwen35.wdc.remote_gpu0_only.json")
    result.add_argument(
        "--image-model-workers", type=int,
        default=32,
        help="in-flight image requests; text and image are budgeted separately "
             "because the vision encoder is the scarcer of the two")
    result.add_argument("--model-timeout", type=float, default=120.0)
    # Output is one small JSON object; the caps only matter as a guard against a
    # runaway generation, and they mirror the EntiTables builder's (1024/384).
    result.add_argument("--text-model-max-tokens", type=int, default=1024)
    result.add_argument("--image-model-max-tokens", type=int, default=384)
    result.add_argument("--no-auto-check", action="store_true",
                        help="skip the local attribute-extraction check; recovery "
                             "counts are then unverified and the manifest says so")
    result.add_argument("--raw-data-lake-refs", dest="raw_data_lake_refs",
                        action="store_true", default=True)
    result.add_argument("--no-raw-data-lake-refs", dest="raw_data_lake_refs",
                        action="store_false")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    stats = build(args)
    if stats.get("plan_only"):
        return 0
    print(f"{stats['query_tables']} queries, {stats['data_lake_tables']} targets, "
          f"{stats['qrels']} qrels, {stats['evidence_recoveries']} recoveries "
          f"({stats['join_shape']}, auto-check {stats['auto_check']})")
    print(f"  decisions: {stats['decisions']}")
    print(f"  hidden columns: {stats['hidden_columns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

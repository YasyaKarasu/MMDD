#!/usr/bin/env python
"""Build a multimodal joinability discovery dataset from EntiTables JSON.

The builder first fixes a source-table pool that acts as the data-lake base.
It then asks local text/image models which row-level entity attributes can be
extracted from each entity's Wikipedia assets. A source column becomes a query
join column only when enough row values can be recovered from multimodal
evidence.
"""

from __future__ import annotations

import argparse
import base64
import heapq
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import queue
import random
import re
import threading
import time
import unicodedata
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field as dataclass_field
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote

try:
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    requests = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    tqdm = None  # type: ignore[assignment]

from build_mm_table_dataset import (
    ShardedJsonlWriter,
    WikipediaClient,
    build_bridge_assets,
    default_wikipedia_user_agent,
    finalize_entities,
    iter_jsonl_records,
    iter_with_progress,
    is_useful_image,
    normalize_title,
    parse_source_table,
    read_entitables_json,
    select_relevant_text_chunks,
    split_text_asset_content,
    update_entities_from_table,
    write_jsonl_record,
    write_sharded_jsonl,
    write_table_asset_links_from_jsonl,
)
from image_preprocessing import target_size
import gpu_priority_protocol as gpu_priority
import model_marker_protocol as model_markers
from remote_vllm_layout import (
    ControllerConfig as RemoteLayoutControllerConfig,
    EndpointConfig as RemoteLayoutEndpointConfig,
    ExclusiveControllerLock,
    LayoutProtocolError,
    RemoteLayoutController,
    RoutingScheduler,
    RoutingUnavailableError,
    WorkloadSnapshot,
)
from stage1_io import (
    clean_text,
    column_profiles,
    get_cell,
    get_cell_text,
    get_column_name,
    make_columns,
    sanitize_cell_text_for_model,
    setup_logging,
    stable_hash,
    write_json,
    write_jsonl,
)
from wikimedia_media import MediaFailureRecorder, MediaPolicyConfig


PROMPT_VERSION = "entity_attribute_extraction_v5_batched_leave_one_out"
DEFAULT_SHARED_CACHE_DIR = Path("cache") / "mm_joinability"
DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS = 262_144
DEFAULT_IMAGE_REQUEST_MAX_PIXELS = 512_000
DEFAULT_IMAGE_MODEL_MAX_TOKENS = 384
DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO = 0.2
DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE = "ratio"
EXPLICIT_JOIN_FALLBACK_MODES = (
    "disabled",
    "ratio",
    "match_implicit",
)
_MODEL_ERROR_LOG_LOCK = threading.Lock()


@dataclass(frozen=True)
class ReplacementPolicy:
    rounds: int
    drop_probability: float


@dataclass(frozen=True)
class CandidateEvaluation:
    source_table: dict[str, Any]
    queryable: bool
    decision: dict[str, Any]


class ListRecordWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self) -> None:
        pass


@dataclass(frozen=True)
class ReplacementRoundStats:
    round_index: int
    evaluated: int
    unrecoverable: int
    discarded: int
    retained_failed: int
    replacements: int


@dataclass(frozen=True)
class ReplacementSelection:
    final_evaluations: list[CandidateEvaluation]
    rounds: list[ReplacementRoundStats]
    candidates_consumed: int
    candidate_exhausted: bool
    unfilled_slots: int


@dataclass
class SourceCandidateCounters:
    processed_tables: int = 0
    skipped_tables: int = 0
    skip_reasons: Counter[str] = dataclass_field(default_factory=Counter)


@dataclass(frozen=True, order=True)
class SelectedSourceTableRef:
    priority: int
    relative_path: str
    table_id: str


@dataclass(frozen=True)
class _DescendingSelectedSourceTableRef:
    ref: SelectedSourceTableRef

    def __lt__(self, other: _DescendingSelectedSourceTableRef) -> bool:
        return self.ref > other.ref


@dataclass(frozen=True)
class CandidateDependencies:
    entities: frozenset[str] = dataclass_field(default_factory=frozenset)
    assets: frozenset[str] = dataclass_field(default_factory=frozenset)
    paths: frozenset[Path] = dataclass_field(default_factory=frozenset)
    urls: frozenset[str] = dataclass_field(default_factory=frozenset)
    page_keys: frozenset[str] = dataclass_field(default_factory=frozenset)
    imageinfo_keys: frozenset[str] = dataclass_field(default_factory=frozenset)
    model_keys: frozenset[str] = dataclass_field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for field_name in (
            "entities",
            "assets",
            "paths",
            "urls",
            "page_keys",
            "imageinfo_keys",
            "model_keys",
        ):
            object.__setattr__(self, field_name, frozenset(getattr(self, field_name)))


@dataclass
class CandidateEvaluationContext:
    entity_records: dict[str, dict[str, Any]]
    wiki_to_entity_id: dict[str, str]
    assets: dict[str, dict[str, Any]]
    entity_to_assets: dict[str, list[str]]
    wikipedia_client: WikipediaClient | None
    extractor: LocalAttributeExtractor | None
    cache: ExtractionCache
    progress: ModelAnalysisProgress | None
    concurrency_state: ModelConcurrencyState
    registry: CandidateMaterialRegistry
    max_entities: int | None = None
    eligible_entity_ids: set[str] = dataclass_field(default_factory=set)
    entity_imageinfo_keys: dict[str, set[str]] = dataclass_field(default_factory=dict)
    text_task_count: int = 0
    image_task_count: int = 0


@dataclass
class CacheCleanupStats:
    entities_removed: int = 0
    assets_removed: int = 0
    page_records_removed: int = 0
    imageinfo_records_removed: int = 0
    model_records_removed: int = 0
    image_files_removed: int = 0
    image_bytes_removed: int = 0
    shared_dependencies_protected: int = 0
    errors: int = 0

    def add(self, other: CacheCleanupStats) -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


def compact_keyed_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CandidateMaterialRegistry:
    def __init__(
        self,
        *,
        assets: dict[str, dict[str, Any]],
        entity_to_assets: dict[str, list[str]],
        wikipedia_client: Any,
        extraction_cache: ExtractionCache,
    ) -> None:
        self.assets = assets
        self.entity_to_assets = entity_to_assets
        self.wikipedia_client = wikipedia_client
        self.extraction_cache = extraction_cache
        self.dependencies: dict[str, CandidateDependencies] = {}

    def register(self, table_id: str, dependencies: CandidateDependencies) -> None:
        self.dependencies[table_id] = dependencies

    @staticmethod
    def _union_dependencies(
        dependencies: Iterable[CandidateDependencies],
    ) -> CandidateDependencies:
        unions: dict[str, set[Any]] = {
            "entities": set(),
            "assets": set(),
            "paths": set(),
            "urls": set(),
            "page_keys": set(),
            "imageinfo_keys": set(),
            "model_keys": set(),
        }
        for candidate_dependencies in dependencies:
            for field_name, values in unions.items():
                values.update(getattr(candidate_dependencies, field_name))
        return CandidateDependencies(**unions)

    def _retained_dependencies(self) -> CandidateDependencies:
        return self._union_dependencies(self.dependencies.values())

    def discard_many(self, table_ids: Iterable[str]) -> CacheCleanupStats:
        """Remove discarded candidates from active output material only.

        Downloaded images and Wikipedia/model cache entries are deliberately
        retained so a later run can reuse work performed for a candidate that
        was replaced during this run.
        """
        discarded_dependencies = [
            dependencies
            for table_id in dict.fromkeys(table_ids)
            if (dependencies := self.dependencies.pop(table_id, None)) is not None
        ]
        stats = CacheCleanupStats()
        if not discarded_dependencies:
            return stats
        discarded = self._union_dependencies(discarded_dependencies)
        retained = self._retained_dependencies()
        stats.shared_dependencies_protected = sum(
            len(getattr(discarded, field_name) & getattr(retained, field_name))
            for field_name in (
                "entities",
                "assets",
                "paths",
                "urls",
                "page_keys",
                "imageinfo_keys",
                "model_keys",
            )
        )

        for entity_id in discarded.entities - retained.entities:
            if self.entity_to_assets.pop(entity_id, None) is not None:
                stats.entities_removed += 1

        exclusive_asset_ids = discarded.assets - retained.assets
        for asset_id in exclusive_asset_ids:
            asset = self.assets.pop(asset_id, None)
            if asset is not None:
                stats.assets_removed += 1
        for entity_id, asset_ids in self.entity_to_assets.items():
            self.entity_to_assets[entity_id] = [
                asset_id for asset_id in asset_ids if asset_id not in exclusive_asset_ids
            ]
        return stats

    def discard(self, table_id: str) -> CacheCleanupStats:
        return self.discard_many([table_id])

    def sweep(self, final_table_ids: Iterable[str]) -> CacheCleanupStats:
        retained_ids = set(final_table_ids)
        return self.discard_many(
            table_id
            for table_id in list(self.dependencies)
            if table_id not in retained_ids
        )


def _candidate_entity_ids(
    source_table: dict[str, Any], wiki_to_entity_id: dict[str, str]
) -> list[str]:
    entity_ids: list[str] = []
    seen: set[str] = set()
    for row in source_table.get("rows", []):
        for cell in row.get("cells", []):
            wiki_title = clean_text(cell.get("wiki_title"))
            if not wiki_title:
                continue
            entity_id = wiki_to_entity_id.get(normalize_title(wiki_title))
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                entity_ids.append(entity_id)
    return entity_ids


def _eligible_candidate_entity_ids(
    candidate_entity_ids: Iterable[str], context: CandidateEvaluationContext
) -> set[str]:
    candidate_ids = list(candidate_entity_ids)
    for entity_id in candidate_ids:
        if entity_id in context.eligible_entity_ids:
            continue
        if context.max_entities and len(context.eligible_entity_ids) >= context.max_entities:
            continue
        context.eligible_entity_ids.add(entity_id)
    return set(candidate_ids) & context.eligible_entity_ids


def _attempted_imageinfo_keys(page: dict[str, Any] | None) -> set[str]:
    if not page or page.get("missing"):
        return set()
    image_titles: list[str] = []
    if page.get("pageimage"):
        image_titles.append(f"File:{page['pageimage']}")
    for image in page.get("images") or []:
        title = image.get("title") if isinstance(image, dict) else None
        if title:
            image_titles.append(title)
    return {
        normalized_title
        for image_title in image_titles
        if is_useful_image(normalized_title := normalize_title(image_title))
    }


def prepare_candidate_batch(
    source_tables: list[dict[str, Any]],
    context: CandidateEvaluationContext,
    args: argparse.Namespace,
) -> None:
    ordered_entity_ids: list[str] = []
    seen_entity_ids: set[str] = set()
    source_table_iterator: Iterable[dict[str, Any]] = source_tables
    if tqdm is not None:
        source_table_iterator = tqdm(
            source_table_iterator,
            total=len(source_tables),
            desc="Preparing candidate batch materials",
            unit="table",
            dynamic_ncols=True,
            disable=not args.model_progress,
        )
    for source_table in source_table_iterator:
        update_entities_from_table(
            context.entity_records, context.wiki_to_entity_id, source_table
        )
        entity_ids = _eligible_candidate_entity_ids(
            _candidate_entity_ids(source_table, context.wiki_to_entity_id), context
        )
        for entity_id in _candidate_entity_ids(
            source_table, context.wiki_to_entity_id
        ):
            if (
                entity_id in entity_ids
                and entity_id not in context.entity_to_assets
                and entity_id not in seen_entity_ids
            ):
                ordered_entity_ids.append(entity_id)
                seen_entity_ids.add(entity_id)
        if tqdm is not None:
            source_table_iterator.set_postfix(  # type: ignore[attr-defined]
                eligible_entities=len(context.eligible_entity_ids)
            )

    if not ordered_entity_ids:
        return

    entities_by_id = {
        entity["entity_id"]: entity
        for entity in finalize_entities(context.entity_records)
        if entity["entity_id"] in seen_entity_ids
    }
    entities = [entities_by_id[entity_id] for entity_id in ordered_entity_ids]
    writer = ListRecordWriter()
    batch_entity_to_assets: dict[str, list[str]] = {}
    if context.wikipedia_client is not None:
        batch_entity_to_assets, _api_failures, _text_count, _image_count = (
            build_bridge_assets(
                entities=entities,
                max_entities=None,
                max_images_per_entity=args.max_images_per_entity,
                text_asset_chunk_chars=args.text_asset_chunk_chars,
                min_text_asset_chunk_chars=args.min_text_asset_chunk_chars,
                max_text_asset_chunks_per_entity=args.max_text_asset_chunks_per_entity,
                wikipedia_client=context.wikipedia_client,
                asset_writer=writer,
                flush_every_records=args.flush_every_records,
                show_progress=args.model_progress,
            )
        )

    for asset in writer.records:
        context.assets[str(asset["asset_id"])] = asset
    for entity_id in ordered_entity_ids:
        context.entity_to_assets[entity_id] = list(
            batch_entity_to_assets.get(entity_id, [])
        )

    if context.wikipedia_client is None or args.max_images_per_entity <= 0:
        return
    page_cache = getattr(context.wikipedia_client, "page_cache", {})
    for entity in entities:
        entity_id = str(entity["entity_id"])
        page = page_cache.get(normalize_title(str(entity["wiki_title"])))
        context.entity_imageinfo_keys.setdefault(entity_id, set()).update(
            _attempted_imageinfo_keys(page)
        )


def _candidate_dependencies(
    *,
    entity_ids: set[str],
    context: CandidateEvaluationContext,
    extraction_records: Iterable[dict[str, Any]],
    recovery_records: Iterable[dict[str, Any]],
    imageinfo_keys_accessed: Iterable[str] = (),
) -> CandidateDependencies:
    asset_ids = {
        asset_id
        for entity_id in entity_ids
        for asset_id in context.entity_to_assets.get(entity_id, [])
        if asset_id in context.assets
    }
    referenced_assets = [context.assets[asset_id] for asset_id in asset_ids]
    model_keys = {
        clean_text(record.get("cache_key"))
        for record in extraction_records
        if clean_text(record.get("cache_key"))
    }
    model_keys.update(
        clean_text(record.get("evidence", {}).get("extraction_cache_key"))
        for record in recovery_records
        if clean_text(record.get("evidence", {}).get("extraction_cache_key"))
    )
    return CandidateDependencies(
        entities=entity_ids,
        assets=asset_ids,
        paths={
            Path(local_path)
            for asset in referenced_assets
            if (local_path := clean_text(asset.get("local_path")))
        },
        urls={
            image_url
            for asset in referenced_assets
            if (image_url := clean_text(asset.get("image_url")))
        },
        page_keys={
            normalize_title(str(context.entity_records[entity_id]["wiki_title"]))
            for entity_id in entity_ids
        },
        imageinfo_keys={normalize_title(key) for key in imageinfo_keys_accessed}
        | {
            normalize_title(file_title)
            for asset in referenced_assets
            if (file_title := clean_text(asset.get("metadata", {}).get("file_title")))
        },
        model_keys=model_keys,
    )


def _precompute_candidate_batch_tasks(
    source_tables: list[dict[str, Any]],
    context: CandidateEvaluationContext,
    args: argparse.Namespace,
) -> dict[str, int]:
    full_precompute = getattr(args, "precompute_model_cache", False)
    text_precompute = getattr(args, "precompute_text_model_cache", False)
    precompute_enabled = full_precompute or text_precompute
    visible_progress = context.progress is not None and context.progress.enabled
    if not precompute_enabled and not visible_progress:
        return {}

    batch_tasks_by_key: dict[str, ExtractionTask] = {}
    asset_types = None if visible_progress or full_precompute else {"text"}
    for source_table in source_tables:
        for task in collect_table_extraction_tasks(
            source_table=source_table,
            assets=context.assets,
            entity_to_assets=context.entity_to_assets,
            wiki_to_entity_id=context.wiki_to_entity_id,
            args=args,
            asset_types=asset_types,
        ):
            batch_tasks_by_key.setdefault(task.cache_key, task)

    if visible_progress:
        context.progress.register(batch_tasks_by_key)

    if not precompute_enabled:
        tasks_requiring_model_analysis(
            list(batch_tasks_by_key.values()),
            context.cache,
            args,
            progress=context.progress,
        )
        return {}

    precompute_tasks = list(batch_tasks_by_key.values())
    if not full_precompute:
        precompute_tasks = [
            task
            for task in precompute_tasks
            if task.asset.get("asset_type") == "text"
        ]
    pending_tasks = tasks_requiring_model_analysis(
        precompute_tasks,
        context.cache,
        args,
        progress=context.progress,
    )
    tasks_by_kind = {
        "text": [
            task for task in pending_tasks if task.asset.get("asset_type") == "text"
        ]
    }
    if full_precompute:
        tasks_by_kind["image"] = [
            task
            for task in pending_tasks
            if task.asset.get("asset_type") == "image"
        ]
    return precompute_extraction_task_groups(
        extractor=context.extractor,
        cache=context.cache,
        tasks_by_kind=tasks_by_kind,
        args=args,
        state=context.concurrency_state,
        progress=context.progress,
        write_done_markers=False,
    )


def evaluate_candidate_batch(
    source_tables: list[dict[str, Any]],
    context: CandidateEvaluationContext,
    args: argparse.Namespace,
) -> list[CandidateEvaluation]:
    for source_table in source_tables:
        update_entities_from_table(
            context.entity_records, context.wiki_to_entity_id, source_table
        )

    counts = _precompute_candidate_batch_tasks(source_tables, context, args)
    context.text_task_count += counts.get("text", 0)
    context.image_task_count += counts.get("image", 0)

    evaluations: list[CandidateEvaluation] = []
    for source_table in source_tables:
        entity_ids = _eligible_candidate_entity_ids(
            _candidate_entity_ids(source_table, context.wiki_to_entity_id), context
        )
        imageinfo_keys_accessed = {
            imageinfo_key
            for entity_id in entity_ids
            for imageinfo_key in context.entity_imageinfo_keys.get(entity_id, set())
        }
        extraction_writer = ListRecordWriter()
        recovery_writer = ListRecordWriter()
        query_tables, _data_lake_tables, _qrels, decision = build_table_join_records(
            source_table=source_table,
            split="candidate",
            assets=context.assets,
            entity_to_assets=context.entity_to_assets,
            wiki_to_entity_id=context.wiki_to_entity_id,
            extractor=context.extractor,
            cache=context.cache,
            progress=context.progress,
            concurrency_state=context.concurrency_state,
            extraction_writer=extraction_writer,
            recovery_writer=recovery_writer,
            args=args,
        )
        table_id = str(source_table["source_table_id"])
        context.registry.register(
            table_id,
            _candidate_dependencies(
                entity_ids=entity_ids,
                context=context,
                extraction_records=extraction_writer.records,
                recovery_records=recovery_writer.records,
                imageinfo_keys_accessed=imageinfo_keys_accessed,
            ),
        )
        evaluations.append(
            CandidateEvaluation(
                source_table=source_table,
                queryable=bool(query_tables),
                decision=dict(decision),
            )
        )
    return evaluations


def replacement_policy_from_args(args: argparse.Namespace) -> ReplacementPolicy:
    rounds = int(args.unrecoverable_replacement_rounds)
    probability = float(args.unrecoverable_drop_probability)
    if rounds < 0:
        raise ValueError("unrecoverable replacement rounds must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("unrecoverable drop probability must be within [0, 1]")
    return ReplacementPolicy(rounds, probability)


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def iter_random_source_tables(
    input_dir: Path,
    args: argparse.Namespace,
    counters: SourceCandidateCounters,
) -> Iterator[dict[str, Any]]:
    query_rows_per_table = configured_query_rows_per_table(args)
    json_files = sorted(
        input_dir.rglob("*.json"),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    json_file_iterator: Iterable[Path] = json_files
    if tqdm is not None:
        json_file_iterator = tqdm(
            json_file_iterator,
            total=len(json_files),
            desc="Scanning EntiTables for global sample",
            unit="file",
            dynamic_ncols=True,
            disable=not args.model_progress,
        )
    capacity = (
        None
        if args.max_source_tables is None
        else args.max_source_tables * (args.unrecoverable_replacement_rounds + 1)
    )
    selected_heap: list[_DescendingSelectedSourceTableRef] = []
    selected_refs: list[SelectedSourceTableRef] = []
    for json_file in json_file_iterator:
        payload = read_entitables_json(json_file)
        if payload is None:
            counters.skipped_tables += 1
            counters.skip_reasons["malformed_json_file"] += 1
            continue
        relative_path = json_file.relative_to(input_dir).as_posix()
        for table_id, table_obj in payload.items():
            counters.processed_tables += 1
            result = parse_source_table(
                str(table_id),
                table_obj,
                json_file,
                input_dir,
                args.min_rows,
                args.min_cols,
                args.wiki_link_threshold,
            )
            if result.source_table is None:
                counters.skipped_tables += 1
                counters.skip_reasons[result.skip_reason or "unknown"] += 1
                continue
            if not result.source_table.get("metadata", {}).get(
                "candidate_entity_columns"
            ):
                counters.skipped_tables += 1
                counters.skip_reasons["no_candidate_entity_column"] += 1
                continue
            entity_col = choose_entity_column(
                result.source_table,
                min_linked_rows=query_rows_per_table,
            )
            if entity_col is None:
                counters.skipped_tables += 1
                counters.skip_reasons["too_few_candidate_entity_rows"] += 1
                continue
            ref = SelectedSourceTableRef(
                priority=int(
                    stable_hash(
                        "global-source-table",
                        args.seed,
                        relative_path,
                        table_id,
                        length=40,
                    ),
                    16,
                ),
                relative_path=relative_path,
                table_id=str(table_id),
            )
            if capacity is None:
                selected_refs.append(ref)
            elif capacity > 0:
                entry = _DescendingSelectedSourceTableRef(ref)
                if len(selected_heap) < capacity:
                    heapq.heappush(selected_heap, entry)
                elif ref < selected_heap[0].ref:
                    heapq.heapreplace(selected_heap, entry)

    if capacity is not None:
        selected_refs = [entry.ref for entry in selected_heap]
    selected_refs.sort()
    if not selected_refs:
        return

    chunk_size = args.max_source_tables or len(selected_refs)
    materialization_progress = None
    if tqdm is not None:
        materialization_progress = tqdm(
            total=(len(selected_refs) + chunk_size - 1) // chunk_size,
            desc="Materializing global EntiTables sample",
            unit="chunk",
            dynamic_ncols=True,
            disable=not args.model_progress,
        )
    try:
        for chunk_start in range(0, len(selected_refs), chunk_size):
            chunk = selected_refs[chunk_start : chunk_start + chunk_size]
            refs_by_path: dict[str, list[SelectedSourceTableRef]] = defaultdict(list)
            for ref in chunk:
                refs_by_path[ref.relative_path].append(ref)
            materialized: dict[tuple[str, str], dict[str, Any]] = {}
            for relative_path, file_refs in refs_by_path.items():
                json_file = input_dir / relative_path
                payload = read_entitables_json(json_file)
                if payload is None:
                    raise RuntimeError(
                        "Failed to rematerialize selected source table "
                        f"{relative_path}#{file_refs[0].table_id}: "
                        "source file could not be read"
                    )
                for ref in file_refs:
                    table_obj = payload.get(ref.table_id)
                    if table_obj is None:
                        raise RuntimeError(
                            "Failed to rematerialize selected source table "
                            f"{relative_path}#{ref.table_id}: table is missing"
                        )
                    result = parse_source_table(
                        ref.table_id,
                        table_obj,
                        json_file,
                        input_dir,
                        args.min_rows,
                        args.min_cols,
                        args.wiki_link_threshold,
                    )
                    if result.source_table is None:
                        raise RuntimeError(
                            "Failed to rematerialize selected source table "
                            f"{relative_path}#{ref.table_id}: "
                            f"table is no longer valid ({result.skip_reason or 'unknown'})"
                        )
                    materialized[(relative_path, ref.table_id)] = result.source_table
            if materialization_progress is not None:
                materialization_progress.update(1)
            for ref in chunk:
                yield materialized[(ref.relative_path, ref.table_id)]
    finally:
        if materialization_progress is not None:
            materialization_progress.close()


def run_replacement_rounds(
    *,
    candidate_tables: Iterator[dict[str, Any]],
    target_count: int,
    policy: ReplacementPolicy,
    rng: Any,
    prepare_batch: Callable[[list[dict[str, Any]]], None] | None = None,
    evaluate_batch: Callable[[list[dict[str, Any]]], list[CandidateEvaluation]],
    discard_tables: Callable[[list[str]], None],
    on_initial_batch: Callable[[list[dict[str, Any]]], None] | None = None,
    on_initial_batch_prepared: Callable[[list[dict[str, Any]]], None] | None = None,
) -> ReplacementSelection:
    """Evaluate candidates and resample every current failure on each pass.

    Only newly assigned replacements are evaluated again, while the replacement
    draw pool includes failures deferred by every earlier pass.
    """
    if target_count < 0:
        raise ValueError("target count must be non-negative")

    slot_tables: list[dict[str, Any]] = []
    candidate_exhausted = False
    while len(slot_tables) < target_count:
        try:
            slot_tables.append(next(candidate_tables))
        except StopIteration:
            candidate_exhausted = True
            break

    if prepare_batch is not None:
        prepare_batch(slot_tables)
    initial_batch_callback = on_initial_batch_prepared or on_initial_batch
    if initial_batch_callback is not None:
        initial_batch_callback(slot_tables)

    candidates_consumed = len(slot_tables)
    current_evaluations: list[CandidateEvaluation | None] = [None] * len(slot_tables)
    pending_slots = list(range(len(slot_tables)))
    pending_discards: list[str] = []
    round_stats: list[ReplacementRoundStats] = []

    if pending_slots:
        for round_index in range(policy.rounds + 1):
            evaluations: list[CandidateEvaluation] = []
            if pending_slots:
                batch = [slot_tables[slot_index] for slot_index in pending_slots]
                if round_index > 0 and prepare_batch is not None:
                    prepare_batch(batch)
                evaluations = evaluate_batch(batch)
                if len(evaluations) != len(batch):
                    raise ValueError(
                        "candidate evaluation count does not match the requested batch"
                    )
                for slot_index, source_table, evaluation in zip(
                    pending_slots, batch, evaluations
                ):
                    expected_id = source_table.get("source_table_id")
                    actual_id = evaluation.source_table.get("source_table_id")
                    if actual_id != expected_id:
                        raise ValueError(
                            "candidate evaluation source ID does not match the "
                            "requested batch"
                        )
                    current_evaluations[slot_index] = evaluation

            if pending_discards:
                discard_tables(pending_discards)
                pending_discards = []

            failed_slots = [
                slot_index
                for slot_index, evaluation in enumerate(current_evaluations)
                if evaluation is not None and not evaluation.queryable
            ]
            discarded = 0
            replacements = 0
            next_pending_slots: list[int] = []
            retained_failed = len(failed_slots)

            if round_index < policy.rounds:
                for slot_index in failed_slots:
                    if rng.random() >= policy.drop_probability:
                        continue
                    if candidate_exhausted:
                        continue
                    try:
                        replacement_table = next(candidate_tables)
                    except StopIteration:
                        candidate_exhausted = True
                        continue

                    evaluation = current_evaluations[slot_index]
                    if evaluation is None:
                        raise RuntimeError(
                            "failed replacement slot is missing its evaluation"
                        )
                    source_table_id = str(
                        evaluation.source_table["source_table_id"]
                    )
                    pending_discards.append(source_table_id)
                    slot_tables[slot_index] = replacement_table
                    current_evaluations[slot_index] = None
                    candidates_consumed += 1
                    discarded += 1
                    replacements += 1
                    retained_failed -= 1
                    next_pending_slots.append(slot_index)

            round_stats.append(
                ReplacementRoundStats(
                    round_index=round_index,
                    evaluated=len(evaluations),
                    unrecoverable=len(failed_slots),
                    discarded=discarded,
                    retained_failed=retained_failed,
                    replacements=replacements,
                )
            )
            if (
                not failed_slots
                or round_index >= policy.rounds
                or (candidate_exhausted and not replacements)
            ):
                break
            pending_slots = next_pending_slots

    return ReplacementSelection(
        final_evaluations=[
            evaluation
            for evaluation in current_evaluations
            if evaluation is not None
        ],
        rounds=round_stats,
        candidates_consumed=candidates_consumed,
        candidate_exhausted=candidate_exhausted,
        unfilled_slots=target_count - len(slot_tables),
    )


def resolve_shared_cache_paths(args: argparse.Namespace) -> dict[str, Path]:
    root_dir = Path(args.cache_dir).expanduser().resolve()
    wikipedia_cache_dir = (
        Path(args.wikipedia_cache_dir).expanduser().resolve()
        if args.wikipedia_cache_dir
        else root_dir / "wikipedia"
    )
    wikipedia_image_dir = (
        Path(args.wikipedia_image_dir).expanduser().resolve()
        if args.wikipedia_image_dir
        else root_dir / "images"
    )
    return {
        "root_dir": root_dir,
        "wikipedia_cache_dir": wikipedia_cache_dir,
        "wikipedia_image_dir": wikipedia_image_dir,
        "model_attribute_extractions": root_dir / "model_attribute_extractions.jsonl",
    }


def media_policy_config_from_args(args: argparse.Namespace) -> MediaPolicyConfig:
    return MediaPolicyConfig(
        workers=args.media_download_workers,
        max_mbps=args.media_max_mbps,
        max_retries=args.media_max_retries,
        retry_base_seconds=args.media_retry_base_seconds,
        retry_max_seconds=args.media_retry_max_seconds,
        chunk_bytes=args.media_chunk_bytes,
    ).validate()


def normalize(value: Any) -> str:
    return clean_text(value).casefold()


_NUMERIC_VALUE_RE = re.compile(
    r"^([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*([%a-zA-Z°²³/_-]*)$"
)
_TEMPORAL_ATTRIBUTE_RE = re.compile(
    r"(?:^|\W)(?:year|date|born|birth|death|died|season|term|opened|founded|released)(?:$|\W)",
    re.IGNORECASE,
)


def _match_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).split())


def _numeric_value(value: Any) -> tuple[Decimal, str] | None:
    text = unicodedata.normalize("NFKC", clean_text(value)).strip()
    match = _NUMERIC_VALUE_RE.fullmatch(text)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return number, match.group(2).casefold()


def _contains_whole_phrase(container: str, phrase: str) -> bool:
    return f" {phrase} " in f" {container} "


def values_match(
    predicted: Any,
    expected: Any,
    *,
    attribute_name: str = "",
) -> bool:
    pred = _match_text(predicted)
    exp = _match_text(expected)
    if not pred or not exp:
        return False
    if pred == exp:
        return True

    pred_number = _numeric_value(predicted)
    exp_number = _numeric_value(expected)
    if pred_number is not None and exp_number is not None:
        return pred_number == exp_number
    if pred_number is not None or exp_number is not None:
        if not _TEMPORAL_ATTRIBUTE_RE.search(clean_text(attribute_name)):
            return False
        number_text = pred if pred_number is not None else exp
        phrase_text = exp if pred_number is not None else pred
        return (
            len(number_text) == 4
            and number_text.isdigit()
            and _contains_whole_phrase(phrase_text, number_text)
        )

    shorter, longer = sorted((pred, exp), key=len)
    compact_shorter = shorter.replace(" ", "")
    if len(compact_shorter) < 4:
        return False
    return _contains_whole_phrase(longer, shorter)


def iter_json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    start = 0
    while True:
        start = text.find("{", start)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start += 1
            continue
        if isinstance(payload, dict):
            yield payload
        start += max(1, end)


def safe_json_object(text: str) -> dict[str, Any]:
    text = clean_text(text)
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    objects = list(iter_json_objects(text))
    for payload in reversed(objects):
        if isinstance(payload.get("attributes"), list):
            return payload
    return objects[-1] if objects else {}


def is_placeholder_text(value: str) -> bool:
    value = clean_text(value)
    return len(value) >= 2 and value.startswith("<") and value.endswith(">")


def normalize_extracted_attributes(
    payload: dict[str, Any],
    candidate_attributes: list[str] | None = None,
) -> list[dict[str, Any]]:
    items = payload.get("attributes")
    if not isinstance(items, list):
        return []
    candidate_names = {normalize(name) for name in candidate_attributes or [] if clean_text(name)}
    attrs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name") or item.get("attribute") or item.get("column_name"))
        value = clean_text(item.get("value"))
        if not name or not value:
            continue
        if is_placeholder_text(name) or is_placeholder_text(value):
            continue
        if candidate_names and normalize(name) not in candidate_names:
            continue
        attrs.append({"name": name, "value": value})
    return attrs


def canonical_extraction_row_attributes(
    row_attributes: Any,
) -> list[dict[str, Any]]:
    if not isinstance(row_attributes, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in row_attributes:
        if not isinstance(item, dict):
            continue
        name = sanitize_cell_text_for_model(item.get("name"))
        value = sanitize_cell_text_for_model(item.get("value"))
        if not name or not value:
            continue
        normalized.append(
            {
                "name": name,
                "value": value,
                "is_entity": bool(item.get("is_entity")),
            }
        )
    return normalized


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def image_mode_has_alpha(mode: str, info: dict[str, Any]) -> bool:
    return mode in {"RGBA", "LA"} or "transparency" in info


def resized_image_data_url(path: Path, max_pixels: int) -> str:
    if max_pixels <= 0:
        return image_data_url(path)
    try:
        from PIL import Image
        from PIL import ImageFile
    except ImportError as exc:
        raise RuntimeError("Pillow is required to resize images after VL context-length errors") from exc

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as image:
                width, height = image.size
                new_width, new_height = target_size(width, height, max_pixels)
                has_alpha = image_mode_has_alpha(image.mode, image.info)
                image.draft("RGB", (new_width, new_height))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((new_width, new_height), resampling)
                if image.width * image.height > max_pixels:
                    image = image.resize(target_size(image.width, image.height, max_pixels), resampling)

                buffer = io.BytesIO()
                if has_alpha:
                    if image.mode != "RGBA":
                        image = image.convert("RGBA")
                    image.save(buffer, format="PNG", optimize=True)
                    mime = "image/png"
                else:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(buffer, format="JPEG", quality=85, optimize=True)
                    mime = "image/jpeg"
                data = base64.b64encode(buffer.getvalue()).decode("ascii")
                return f"data:{mime};base64,{data}"
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def is_context_length_error(message: Any) -> bool:
    text = clean_text(message).casefold()
    return "input length" in text and "maximum context length" in text


def usage_token_count(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def model_call_stats_summary(extractor: Any) -> dict[str, dict[str, int | float]]:
    stats = getattr(extractor, "model_call_stats", None)
    summary = getattr(stats, "summary", None)
    if callable(summary):
        return summary()
    return ModelCallStats().summary()


def normalize_model_base_urls(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = values.replace(",", "\n").splitlines()
    else:
        raw_values = []
        for value in values:
            raw_values.extend(clean_text(value).replace(",", "\n").splitlines())
    urls: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        url = clean_text(value).rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


class TransientModelEndpointError(RuntimeError):
    """A model endpoint failure that may succeed when retried later."""


def model_api_key(explicit_value: Any, modality_environment_variable: str) -> str | None:
    for value in (
        explicit_value,
        os.environ.get(modality_environment_variable),
        os.environ.get("VLLM_API_KEY"),
    ):
        if value is None:
            continue
        key = str(value).strip()
        if key:
            return key
    return None


def is_transient_request_exception(exc: Exception) -> bool:
    if requests is None:
        return False
    exceptions = getattr(requests, "exceptions", None)
    transient_types = tuple(
        exception_type
        for exception_type in (
            getattr(exceptions, "ConnectionError", None),
            getattr(exceptions, "Timeout", None),
        )
        if isinstance(exception_type, type)
    )
    return bool(transient_types) and isinstance(exc, transient_types)


class ModelCallStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stats: dict[str, dict[str, int | float]] = {
            "text": self._empty_bucket(),
            "image": self._empty_bucket(),
        }

    @staticmethod
    def _empty_bucket() -> dict[str, int | float]:
        return {
            "requests": 0,
            "failed_requests": 0,
            "elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "responses_with_usage": 0,
        }

    def record(
        self,
        model_kind: str,
        *,
        elapsed_seconds: float,
        usage: dict[str, Any] | None = None,
        failed: bool = False,
    ) -> None:
        with self._lock:
            bucket = self._stats.setdefault(model_kind, self._empty_bucket())
            bucket["requests"] = int(bucket["requests"]) + 1
            bucket["elapsed_seconds"] = float(bucket["elapsed_seconds"]) + max(0.0, elapsed_seconds)
            if failed:
                bucket["failed_requests"] = int(bucket["failed_requests"]) + 1
            if isinstance(usage, dict):
                prompt_tokens = usage_token_count(usage, "prompt_tokens", "input_tokens")
                completion_tokens = usage_token_count(usage, "completion_tokens", "output_tokens")
                total_tokens = usage_token_count(usage, "total_tokens")
                if total_tokens <= 0:
                    total_tokens = prompt_tokens + completion_tokens
                bucket["prompt_tokens"] = int(bucket["prompt_tokens"]) + prompt_tokens
                bucket["completion_tokens"] = int(bucket["completion_tokens"]) + completion_tokens
                bucket["total_tokens"] = int(bucket["total_tokens"]) + total_tokens
                bucket["responses_with_usage"] = int(bucket["responses_with_usage"]) + 1

    def summary(self) -> dict[str, dict[str, int | float]]:
        summary: dict[str, dict[str, int | float]] = {}
        with self._lock:
            items = [(model_kind, dict(bucket)) for model_kind, bucket in self._stats.items()]
        for model_kind, bucket in sorted(items):
            summary[model_kind] = {
                "requests": int(bucket["requests"]),
                "failed_requests": int(bucket["failed_requests"]),
                "elapsed_seconds": round(float(bucket["elapsed_seconds"]), 6),
                "prompt_tokens": int(bucket["prompt_tokens"]),
                "completion_tokens": int(bucket["completion_tokens"]),
                "total_tokens": int(bucket["total_tokens"]),
                "responses_with_usage": int(bucket["responses_with_usage"]),
            }
        return summary


class LocalAttributeExtractor:
    """OpenAI-compatible client for local text and image extraction models."""

    def __init__(self, args: argparse.Namespace) -> None:
        if requests is None:
            raise RuntimeError("requests is required for local model calls")
        routing_manifest = clean_text(
            getattr(args, "model_routing_manifest", "")
        )
        self.routing_scheduler = (
            RoutingScheduler(Path(routing_manifest).resolve())
            if routing_manifest
            else None
        )
        remote_routing_manifest = clean_text(
            getattr(args, "remote_model_routing_manifest", "")
        )
        self.remote_routing_scheduler = (
            RoutingScheduler(
                Path(remote_routing_manifest).resolve(),
                load_existing=False,
            )
            if remote_routing_manifest
            else None
        )
        configured_text_urls = normalize_model_base_urls(getattr(args, "text_model_base_urls", None))
        fallback_text_url = clean_text(getattr(args, "text_model_base_url", "")).rstrip("/")
        if fallback_text_url:
            configured_text_urls = normalize_model_base_urls([fallback_text_url, *configured_text_urls])
        if not configured_text_urls and self.routing_scheduler is None:
            raise ValueError("at least one text model base URL is required")
        self.text_model_base_urls = configured_text_urls
        self.text_model_base_urls_file = clean_text(getattr(args, "text_model_base_urls_file", ""))
        self._text_endpoint_lock = threading.Lock()
        self._text_endpoint_index = 0
        self._text_endpoint_inflight: dict[str, int] = {}
        self.text_model_name = args.text_model_name
        self.text_model_api_key = model_api_key(
            getattr(args, "text_model_api_key", None),
            "MMDD_TEXT_MODEL_API_KEY",
        )
        configured_remote_text_urls = normalize_model_base_urls(
            getattr(args, "remote_text_model_base_urls", None)
        )
        remote_text_url = clean_text(
            getattr(args, "remote_text_model_base_url", "")
        ).rstrip("/")
        if remote_text_url:
            configured_remote_text_urls = normalize_model_base_urls(
                [remote_text_url, *configured_remote_text_urls]
            )
        self.remote_text_model_base_urls = configured_remote_text_urls
        self.remote_text_model_base_urls_file = clean_text(
            getattr(args, "remote_text_model_base_urls_file", "")
        )
        self._remote_text_endpoint_lock = threading.Lock()
        self._remote_text_endpoint_index = 0
        self._remote_text_endpoint_inflight: dict[str, int] = {}
        self.remote_text_model_api_key = model_api_key(
            getattr(args, "remote_text_model_api_key", None),
            "MMDD_REMOTE_TEXT_MODEL_API_KEY",
        ) or self.text_model_api_key
        self.remote_text_model_workers = max(
            0,
            int(getattr(args, "remote_text_model_workers", 0) or 0),
        )
        if (
            self.remote_text_model_workers
            and not self.remote_text_model_base_urls
            and not self.remote_text_model_base_urls_file
            and self.remote_routing_scheduler is None
        ):
            raise ValueError(
                "remote text workers require a remote text model endpoint"
            )
        configured_image_urls = normalize_model_base_urls(getattr(args, "image_model_base_urls", None))
        fallback_image_url = clean_text(getattr(args, "image_model_base_url", "")).rstrip("/")
        if fallback_image_url:
            configured_image_urls = normalize_model_base_urls([fallback_image_url, *configured_image_urls])
        if not configured_image_urls and self.routing_scheduler is None:
            raise ValueError("at least one image model base URL is required")
        self.image_model_base_urls = configured_image_urls
        self.image_model_base_urls_file = clean_text(getattr(args, "image_model_base_urls_file", ""))
        self._image_endpoint_lock = threading.Lock()
        self._image_endpoint_index = 0
        self._image_endpoint_inflight: dict[str, int] = {}
        self.image_model_name = args.image_model_name
        self.image_model_api_key = model_api_key(
            getattr(args, "image_model_api_key", None),
            "MMDD_IMAGE_MODEL_API_KEY",
        )
        configured_remote_image_urls = normalize_model_base_urls(
            getattr(args, "remote_image_model_base_urls", None)
        )
        remote_image_url = clean_text(
            getattr(args, "remote_image_model_base_url", "")
        ).rstrip("/")
        if remote_image_url:
            configured_remote_image_urls = normalize_model_base_urls(
                [remote_image_url, *configured_remote_image_urls]
            )
        self.remote_image_model_base_urls = configured_remote_image_urls
        self.remote_image_model_base_urls_file = clean_text(
            getattr(args, "remote_image_model_base_urls_file", "")
        )
        self._remote_image_endpoint_lock = threading.Lock()
        self._remote_image_endpoint_index = 0
        self._remote_image_endpoint_inflight: dict[str, int] = {}
        self.remote_image_model_api_key = model_api_key(
            getattr(args, "remote_image_model_api_key", None),
            "MMDD_REMOTE_IMAGE_MODEL_API_KEY",
        ) or self.image_model_api_key
        self.remote_image_model_workers = max(
            0,
            int(getattr(args, "remote_image_model_workers", 0) or 0),
        )
        if (
            self.remote_image_model_workers
            and not self.remote_image_model_base_urls
            and not self.remote_image_model_base_urls_file
            and self.remote_routing_scheduler is None
        ):
            raise ValueError(
                "remote image workers require a remote image model endpoint"
            )
        self.timeout = args.model_timeout_seconds
        self.temperature = args.model_temperature
        self.max_tokens = args.model_max_tokens
        self.image_max_tokens = max(
            1,
            int(getattr(args, "image_model_max_tokens", DEFAULT_IMAGE_MODEL_MAX_TOKENS) or DEFAULT_IMAGE_MODEL_MAX_TOKENS),
        )
        self.disable_thinking = args.disable_thinking
        self.max_retries = args.model_max_retries
        self.retry_sleep = args.model_retry_sleep_seconds
        self.image_request_max_pixels = max(
            0,
            int(getattr(args, "image_request_max_pixels", DEFAULT_IMAGE_REQUEST_MAX_PIXELS) or 0),
        )
        self.context_retry_image_max_pixels = max(
            1,
            int(getattr(args, "context_retry_image_max_pixels", DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS) or 1),
        )
        self.model_call_stats = ModelCallStats()

    def current_text_model_base_urls(self) -> list[str]:
        if self.routing_scheduler is not None:
            return [
                endpoint.base_url
                for endpoint in self.routing_scheduler.endpoints("text")
            ]
        if self.text_model_base_urls_file:
            path = Path(self.text_model_base_urls_file)
            if path.exists():
                try:
                    return normalize_model_base_urls(
                        path.read_text(encoding="utf-8")
                    )
                except OSError as exc:
                    logging.warning("Failed to read text endpoint file %s: %s", path, exc)
                    return []
        return list(self.text_model_base_urls)

    def next_text_model_base_url(self) -> str:
        with self._text_endpoint_lock:
            urls = self.current_text_model_base_urls()
            if not urls:
                raise RuntimeError("no text model endpoints are configured")
            index = self._text_endpoint_index % len(urls)
            self._text_endpoint_index += 1
            return urls[index]

    def current_remote_text_model_base_urls(self) -> list[str]:
        if self.remote_routing_scheduler is not None:
            return [
                endpoint.base_url
                for endpoint in self.remote_routing_scheduler.endpoints("text")
            ]
        if self.remote_text_model_base_urls_file:
            path = Path(self.remote_text_model_base_urls_file)
            if path.exists():
                try:
                    return normalize_model_base_urls(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    logging.warning("Failed to read remote text endpoint file %s: %s", path, exc)
                    return []
        return list(self.remote_text_model_base_urls)

    def next_remote_text_model_base_url(self) -> str:
        with self._remote_text_endpoint_lock:
            urls = self.current_remote_text_model_base_urls()
            if not urls:
                raise RuntimeError("no remote text model endpoints are configured")
            index = self._remote_text_endpoint_index % len(urls)
            self._remote_text_endpoint_index += 1
            return urls[index]

    def current_image_model_base_urls(self) -> list[str]:
        if self.routing_scheduler is not None:
            return [
                endpoint.base_url
                for endpoint in self.routing_scheduler.endpoints("image")
            ]
        if self.image_model_base_urls_file:
            path = Path(self.image_model_base_urls_file)
            if path.exists():
                try:
                    return normalize_model_base_urls(
                        path.read_text(encoding="utf-8")
                    )
                except OSError as exc:
                    logging.warning("Failed to read image endpoint file %s: %s", path, exc)
                    return []
        return list(self.image_model_base_urls)

    def next_image_model_base_url(self) -> str:
        with self._image_endpoint_lock:
            urls = self.current_image_model_base_urls()
            if not urls:
                raise RuntimeError("no image model endpoints are configured")
            index = self._image_endpoint_index % len(urls)
            self._image_endpoint_index += 1
            return urls[index]

    def current_remote_image_model_base_urls(self) -> list[str]:
        if self.remote_routing_scheduler is not None:
            return [
                endpoint.base_url
                for endpoint in self.remote_routing_scheduler.endpoints("image")
            ]
        if self.remote_image_model_base_urls_file:
            path = Path(self.remote_image_model_base_urls_file)
            if path.exists():
                try:
                    return normalize_model_base_urls(path.read_text(encoding="utf-8"))
                except OSError as exc:
                    logging.warning("Failed to read remote image endpoint file %s: %s", path, exc)
                    return []
        return list(self.remote_image_model_base_urls)

    def next_remote_image_model_base_url(self) -> str:
        with self._remote_image_endpoint_lock:
            urls = self.current_remote_image_model_base_urls()
            if not urls:
                raise RuntimeError("no remote image model endpoints are configured")
            index = self._remote_image_endpoint_index % len(urls)
            self._remote_image_endpoint_index += 1
            return urls[index]

    def _endpoint_pool_state(
        self,
        model_kind: str,
        endpoint_pool: str,
    ) -> tuple[
        threading.Lock,
        Callable[[], list[str]],
        dict[str, int],
        str,
    ]:
        if endpoint_pool not in {"local", "remote"}:
            raise ValueError(f"unsupported endpoint pool: {endpoint_pool}")
        prefix = "_remote" if endpoint_pool == "remote" else ""
        if model_kind == "text":
            urls_getter = (
                self.current_remote_text_model_base_urls
                if endpoint_pool == "remote"
                else self.current_text_model_base_urls
            )
        elif model_kind == "image":
            urls_getter = (
                self.current_remote_image_model_base_urls
                if endpoint_pool == "remote"
                else self.current_image_model_base_urls
            )
        else:
            raise ValueError(f"unsupported model kind: {model_kind}")
        return (
            getattr(self, f"{prefix}_{model_kind}_endpoint_lock"),
            urls_getter,
            getattr(self, f"{prefix}_{model_kind}_endpoint_inflight"),
            f"{prefix}_{model_kind}_endpoint_index",
        )

    def _acquire_model_base_url(
        self,
        model_kind: str,
        endpoint_pool: str = "local",
    ) -> str:
        lock, urls_getter, inflight, index_name = self._endpoint_pool_state(
            model_kind,
            endpoint_pool,
        )

        with lock:
            urls = urls_getter()
            if not urls:
                raise RuntimeError(
                    f"no {endpoint_pool} {model_kind} model endpoints are configured"
                )
            minimum_inflight = min(inflight.get(url, 0) for url in urls)
            candidates = [
                url
                for url in urls
                if inflight.get(url, 0) == minimum_inflight
            ]
            index = getattr(self, index_name)
            base_url = candidates[index % len(candidates)]
            setattr(self, index_name, index + 1)
            inflight[base_url] = inflight.get(base_url, 0) + 1
            return base_url

    def _release_model_base_url(
        self,
        model_kind: str,
        base_url: str,
        endpoint_pool: str = "local",
    ) -> None:
        lock, _urls_getter, inflight, _index_name = self._endpoint_pool_state(
            model_kind,
            endpoint_pool,
        )

        with lock:
            count = inflight.get(base_url, 0)
            if count <= 1:
                inflight.pop(base_url, None)
            else:
                inflight[base_url] = count - 1

    @contextmanager
    def lease_model_base_url(
        self,
        model_kind: str,
        endpoint_pool: str = "local",
    ) -> Iterator[str]:
        """Lease the least-loaded current endpoint for one complete model call."""

        scheduler = (
            self.routing_scheduler
            if endpoint_pool == "local"
            else self.remote_routing_scheduler
        )
        if scheduler is not None:
            try:
                with scheduler.lease(model_kind) as base_url:
                    yield base_url
            except RoutingUnavailableError as error:
                raise TransientModelEndpointError(str(error)) from None
            return

        base_url = self._acquire_model_base_url(model_kind, endpoint_pool)
        try:
            yield base_url
        finally:
            self._release_model_base_url(model_kind, base_url, endpoint_pool)

    def _endpoint_is_current(
        self,
        model_kind: str,
        base_url: str,
        endpoint_pool: str = "local",
    ) -> bool:
        scheduler = (
            self.routing_scheduler
            if endpoint_pool == "local"
            else self.remote_routing_scheduler
        )
        if scheduler is not None:
            return scheduler.is_current(model_kind, base_url)
        _lock, urls_getter, _inflight, _index_name = self._endpoint_pool_state(
            model_kind,
            endpoint_pool,
        )
        urls = urls_getter()
        return base_url in urls

    def routing_capacity(self, model_kind: str) -> int | None:
        """Return authoritative dynamic capacity, or ``None`` in static mode."""
        if self.routing_scheduler is None:
            return None
        return self.routing_scheduler.capacity(model_kind)

    def wait_for_endpoint_pool(
        self,
        model_kind: str,
        endpoint_pool: str,
        timeout_seconds: float,
    ) -> bool:
        scheduler = (
            self.routing_scheduler
            if endpoint_pool == "local"
            else self.remote_routing_scheduler
        )
        if scheduler is not None:
            return scheduler.wait_for_capacity(model_kind, timeout_seconds)
        urls = (
            self.current_remote_text_model_base_urls()
            if endpoint_pool == "remote" and model_kind == "text"
            else self.current_remote_image_model_base_urls()
            if endpoint_pool == "remote"
            else self.current_text_model_base_urls()
            if model_kind == "text"
            else self.current_image_model_base_urls()
        )
        if urls:
            return True
        if timeout_seconds > 0:
            time.sleep(timeout_seconds)
        return False

    def _probe_endpoint(
        self,
        model_kind: str,
        base_url: str,
        model: str,
        api_key: str | None,
        request_timeout: float,
    ) -> None:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        models_url = f"{base_url.rstrip('/')}/models"
        transport_error: RuntimeError | None = None
        try:
            response = requests.get(models_url, headers=headers, timeout=request_timeout)
        except Exception as exc:
            message = (
                f"{model_kind} model endpoint {base_url} readiness check failed "
                f"({type(exc).__name__})"
            )
            if is_transient_request_exception(exc):
                transport_error = TransientModelEndpointError(message)
            else:
                transport_error = RuntimeError(message)
        if transport_error is not None:
            raise transport_error from None

        status_code = int(getattr(response, "status_code", 200) or 200)
        if status_code == 429 or status_code >= 500:
            raise TransientModelEndpointError(
                f"{model_kind} model endpoint {base_url} readiness check returned HTTP {status_code}"
            )
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(
                f"{model_kind} model endpoint {base_url} readiness check returned HTTP {status_code}"
            )
        try:
            payload = response.json()
            data = payload["data"]
            served_models = {
                clean_text(item.get("id"))
                for item in data
                if isinstance(item, dict) and clean_text(item.get("id"))
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{model_kind} model endpoint {base_url} returned invalid models JSON"
            ) from exc
        if model not in served_models:
            raise RuntimeError(
                f"{model_kind} model endpoint {base_url} does not serve configured model {model!r}"
            )

    def ensure_endpoints_ready(
        self,
        modalities: set[str],
        timeout_seconds: float,
        poll_seconds: float = 2.0,
    ) -> None:
        """Poll every selected endpoint within one shared readiness deadline.

        A zero timeout performs one probe per endpoint, with each HTTP request capped
        at one second. Positive timeouts cap every request by the deadline remaining
        immediately before that request.
        """
        timeout_seconds = float(timeout_seconds)
        poll_seconds = float(poll_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
            raise ValueError("timeout_seconds must be a finite non-negative number")
        if not math.isfinite(poll_seconds) or poll_seconds < 0:
            raise ValueError("poll_seconds must be a finite non-negative number")

        endpoint_configs: list[tuple[str, str, str, str, str | None]] = []
        for model_kind in ("text", "image"):
            if model_kind not in modalities:
                continue
            if model_kind == "text":
                model = self.text_model_name
                pools = [
                    (
                        "local",
                        self.current_text_model_base_urls(),
                        self.text_model_api_key,
                    )
                ]
                if self.remote_text_model_workers:
                    pools.append(
                        (
                            "remote",
                            self.current_remote_text_model_base_urls(),
                            self.remote_text_model_api_key,
                        )
                    )
            else:
                model = self.image_model_name
                pools = [
                    (
                        "local",
                        self.current_image_model_base_urls(),
                        self.image_model_api_key,
                    )
                ]
                if self.remote_image_model_workers:
                    pools.append(
                        (
                            "remote",
                            self.current_remote_image_model_base_urls(),
                            self.remote_image_model_api_key,
                        )
                    )
            for endpoint_pool, urls, api_key in pools:
                if not urls:
                    raise TransientModelEndpointError(
                        f"no {endpoint_pool} {model_kind} model endpoints are currently configured"
                    )
                endpoint_configs.extend(
                    (endpoint_pool, model_kind, url, model, api_key)
                    for url in urls
                )

        configured_request_timeout = float(self.timeout)
        if not math.isfinite(configured_request_timeout) or configured_request_timeout <= 0:
            configured_request_timeout = 1.0
        single_probe = timeout_seconds == 0
        deadline = time.monotonic() + timeout_seconds
        while True:
            retry_error: TransientModelEndpointError | None = None
            for endpoint_pool, model_kind, url, model, api_key in endpoint_configs:
                if not self._endpoint_is_current(model_kind, url, endpoint_pool):
                    continue
                if single_probe:
                    request_timeout = min(configured_request_timeout, 1.0)
                else:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TransientModelEndpointError(
                            f"{model_kind} model endpoint {url} readiness deadline expired"
                        )
                    request_timeout = min(configured_request_timeout, remaining_seconds)
                try:
                    self._probe_endpoint(model_kind, url, model, api_key, request_timeout)
                    if not single_probe and time.monotonic() >= deadline:
                        raise TransientModelEndpointError(
                            f"{model_kind} model endpoint {url} readiness deadline expired"
                        )
                except TransientModelEndpointError as error:
                    if not self._endpoint_is_current(
                        model_kind,
                        url,
                        endpoint_pool,
                    ):
                        logging.info(
                            "Ignoring readiness failure for withdrawn %s endpoint %s",
                            model_kind,
                            url,
                        )
                        continue
                    retry_error = error
                    break
                except RuntimeError:
                    if not self._endpoint_is_current(
                        model_kind,
                        url,
                        endpoint_pool,
                    ):
                        logging.info(
                            "Ignoring readiness failure for withdrawn %s endpoint %s",
                            model_kind,
                            url,
                        )
                        continue
                    raise
            if retry_error is None:
                return
            if single_probe or time.monotonic() >= deadline:
                raise retry_error
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    def chat(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        messages: list[dict[str, Any]],
        model_kind: str = "text",
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.image_max_tokens if model_kind == "image" else self.max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                status_code = int(getattr(response, "status_code", 200) or 200)
                if status_code >= 400:
                    body = clean_text(getattr(response, "text", ""))[:500]
                    error_message = f"HTTP {status_code}: {body}"
                    if status_code == 429 or status_code >= 500:
                        raise TransientModelEndpointError(error_message)
                    raise RuntimeError(error_message)
                response.raise_for_status()
                data = response.json()
                content = clean_text(data["choices"][0]["message"]["content"])
            except Exception as exc:  # pragma: no cover - integration only.
                self.model_call_stats.record(
                    model_kind,
                    elapsed_seconds=time.perf_counter() - started,
                    failed=True,
                )
                if is_transient_request_exception(exc):
                    last_error = TransientModelEndpointError(
                        f"{model_kind} model endpoint {base_url} request failed "
                        f"({type(exc).__name__})"
                    )
                else:
                    last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep)
            else:
                usage = data.get("usage") if isinstance(data, dict) else None
                self.model_call_stats.record(
                    model_kind,
                    elapsed_seconds=time.perf_counter() - started,
                    usage=usage if isinstance(usage, dict) else None,
                )
                return content
        message = f"Local {model_kind} model call to {base_url} failed: {last_error}"
        if isinstance(last_error, TransientModelEndpointError):
            raise TransientModelEndpointError(message) from last_error
        raise RuntimeError(message) from last_error

    def extraction_prompt(
        self,
        *,
        entity_text: str,
        row_attributes: list[dict[str, Any]],
        candidate_attributes: list[str],
    ) -> str:
        attributes = canonical_extraction_row_attributes(row_attributes)
        if not attributes and clean_text(entity_text):
            attributes = [
                {
                    "name": "Entity",
                    "value": sanitize_cell_text_for_model(entity_text),
                    "is_entity": True,
                }
            ]
        row_lines = []
        for attribute in attributes:
            marker = " [ENTITY; NEVER MASK]" if attribute["is_entity"] else ""
            row_lines.append(
                f'- {attribute["name"]}{marker}: {attribute["value"]}'
            )
        return (
            "You evaluate one table row and one independent material item using only their intrinsic content. "
            "Their presence in the same request does not imply that they are related.\n"
            "Table row attributes:\n"
            + ("\n".join(row_lines) if row_lines else "- (no non-empty attributes)")
            + "\nCandidate attributes to recover:\n"
            + "\n".join(f"- {name}" for name in candidate_attributes)
            + "\nPerform a separate leave-one-attribute-out test for every candidate, while handling all candidates in this single request. "
            "During each test, mentally mask and completely ignore only that candidate's displayed table value. "
            "The entity attribute marked ENTITY is always visible and must never be masked. "
            "All other row attributes remain available when they are not the candidate under test. "
            "Use those remaining attributes and the material's intrinsic content to decide whether the material itself can be connected to the row, "
            "then recover the masked value from the material. "
            "Do not copy, verify, compare against, or otherwise use the candidate's displayed table value during its own test. "
            "Omit the candidate unless the connection and recovered value are independently supported without that value.\n"
            "Return strict JSON only in this shape:\n"
            '{"attributes":[{"name":"<one candidate attribute name>","value":"<extracted value>"}]}\n'
            "Output only extractable candidate attribute names and their extracted values. "
            "Do not output evidence, explanations, rationale, confidence, or any other fields or text. "
            'If no candidate attribute can be recovered, return exactly {"attributes":[]}. '
            "Do not guess. Do not include attributes outside the candidate list."
        )

    @contextmanager
    def _lease_endpoint_pool(
        self,
        model_kind: str,
        endpoint_pool: str,
    ) -> Iterator[str]:
        if endpoint_pool == "local":
            # Preserve subclasses whose lease method predates endpoint pools.
            with self.lease_model_base_url(model_kind) as base_url:
                yield base_url
            return
        with self.lease_model_base_url(
            model_kind,
            endpoint_pool=endpoint_pool,
        ) as base_url:
            yield base_url

    def extract(
        self,
        asset: dict[str, Any],
        entity: dict[str, Any],
        candidate_attributes: list[str],
    ) -> dict[str, Any]:
        return self.extract_from_pool(
            asset,
            entity,
            candidate_attributes,
            endpoint_pool="local",
        )

    def extract_from_pool(
        self,
        asset: dict[str, Any],
        entity: dict[str, Any],
        candidate_attributes: list[str],
        *,
        endpoint_pool: str,
    ) -> dict[str, Any]:
        if endpoint_pool not in {"local", "remote"}:
            raise ValueError(f"unsupported endpoint pool: {endpoint_pool}")
        prompt = self.extraction_prompt(
            entity_text=clean_text(entity.get("cell_text")),
            row_attributes=canonical_extraction_row_attributes(
                entity.get("row_attributes")
            ),
            candidate_attributes=candidate_attributes,
        )
        if asset.get("asset_type") == "text":
            content = clean_text(asset.get("content"))[:6000]
            messages = [
                {"role": "system", "content": "You are a precise information extraction engine."},
                {"role": "user", "content": f"{prompt}\n\nIndependent text material:\n{content}"},
            ]
            api_key = (
                self.remote_text_model_api_key
                if endpoint_pool == "remote"
                else self.text_model_api_key
            )
            with self._lease_endpoint_pool("text", endpoint_pool) as base_url:
                raw = self.chat(
                    base_url=base_url,
                    model=self.text_model_name,
                    api_key=api_key,
                    messages=messages,
                    model_kind="text",
                )
        elif asset.get("asset_type") == "image":
            image_url = clean_text(asset.get("image_url"))
            local_path = clean_text(asset.get("local_path"))
            local_image_path = Path(local_path) if local_path and Path(local_path).exists() else None
            if local_image_path is not None:
                image_url = resized_image_data_url(local_image_path, self.image_request_max_pixels)
            if not image_url.startswith("data:"):
                raise ValueError(
                    f"Image asset {asset.get('asset_id')} has no usable local image or data URL"
                )
            messages = [
                {"role": "system", "content": "You are a precise visual information extraction engine."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
            try:
                api_key = (
                    self.remote_image_model_api_key
                    if endpoint_pool == "remote"
                    else self.image_model_api_key
                )
                with self._lease_endpoint_pool("image", endpoint_pool) as base_url:
                    raw = self.chat(
                        base_url=base_url,
                        model=self.image_model_name,
                        api_key=api_key,
                        messages=messages,
                        model_kind="image",
                    )
            except Exception as exc:
                if local_image_path is None or not is_context_length_error(exc):
                    raise
                retry_messages = [
                    dict(message)
                    for message in messages
                ]
                retry_content = [
                    dict(item)
                    for item in retry_messages[1]["content"]
                ]
                retry_content[1] = {
                    "type": "image_url",
                    "image_url": {
                        "url": resized_image_data_url(local_image_path, self.context_retry_image_max_pixels),
                    },
                }
                retry_messages[1]["content"] = retry_content
                with self._lease_endpoint_pool("image", endpoint_pool) as base_url:
                    raw = self.chat(
                        base_url=base_url,
                        model=self.image_model_name,
                        api_key=api_key,
                        messages=retry_messages,
                        model_kind="image",
                    )
        else:
            return {"attributes": [], "raw_response": "", "error": f"unsupported_asset_type:{asset.get('asset_type')}"}
        payload = safe_json_object(raw)
        return {
            "attributes": normalize_extracted_attributes(
                payload,
                candidate_attributes,
            ),
            "raw_response": raw,
            "error": "",
        }


class ExtractionCache:
    def __init__(self, path: Path, reuse: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict[str, Any]] = {}
        self.transient_items: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if reuse and path.exists():
            for record in iter_jsonl_records([path]):
                key = clean_text(record.get("cache_key"))
                if key:
                    self.items[key] = record

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.items.get(key)

    def get_transient(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.transient_items.get(key)

    def put(self, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            self.items[key] = record
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def put_transient(self, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            self.transient_items[key] = record


def cache_get_transient(cache: Any, key: str) -> dict[str, Any] | None:
    getter = getattr(cache, "get_transient", None)
    return getter(key) if callable(getter) else None


def cache_put_transient(
    cache: Any,
    key: str,
    record: dict[str, Any],
) -> None:
    putter = getattr(cache, "put_transient", None)
    if callable(putter):
        putter(key, record)
    else:
        cache.put(key, record)


class ModelAnalysisProgress:
    def __init__(self, *, total: int, cached_keys: set[str], enabled: bool) -> None:
        self.enabled = enabled and tqdm is not None
        self.total = max(total, len(cached_keys))
        self.planned_keys: set[str] = set()
        self._unassigned_total = self.total - len(cached_keys)
        self.completed_keys: set[str] = set(cached_keys)
        self.cached = len(cached_keys)
        self.model = 0
        self.errors = 0
        self.bar = None
        self._closed = False
        self._lock = threading.Lock()
        if self.enabled:
            self.bar = tqdm(
                total=self.total,
                initial=len(cached_keys),
                desc="Local model analysis",
                unit="asset",
                dynamic_ncols=True,
            )
            self._postfix()

    def register(self, cache_keys: Iterable[str]) -> int:
        keys = {key for key in cache_keys if key}
        with self._lock:
            new_keys = keys - self.planned_keys - self.completed_keys
            if not new_keys:
                return 0
            self.planned_keys.update(new_keys)
            growth = max(0, len(new_keys) - self._unassigned_total)
            self._unassigned_total = max(0, self._unassigned_total - len(new_keys))
            self.total += growth
            if self.bar is not None:
                self.bar.total = self.total
                self._postfix(refresh=False)
                self.bar.refresh()
            return len(new_keys)

    def _postfix(self, *, refresh: bool = True) -> None:
        if self.bar is not None:
            self.bar.set_postfix(
                cached=self.cached,
                model=self.model,
                errors=self.errors,
                refresh=refresh,
            )

    def mark(self, cache_key: str, status: str) -> None:
        with self._lock:
            if cache_key in self.completed_keys:
                return
            self.planned_keys.discard(cache_key)
            self.completed_keys.add(cache_key)
            if status == "cached":
                self.cached += 1
            elif status == "error":
                self.model += 1
                self.errors += 1
            else:
                self.model += 1
            if self.bar is not None:
                self.bar.update(1)
                self._postfix()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.bar is not None:
                self.bar.close()


@dataclass(frozen=True)
class ExtractionTask:
    order: int
    cache_key: str
    source_table_id: str
    source_row_id: int
    entity_column_index: int
    entity_column_name: str
    entity: dict[str, Any]
    asset: dict[str, Any]
    candidate_attribute_names: list[str]


@dataclass
class ModelConcurrencyState:
    text_workers: int
    image_workers: int
    remote_text_workers: int = 0
    remote_image_workers: int = 0
    text_oom_downgrades: int = 0
    image_oom_downgrades: int = 0
    remote_text_oom_downgrades: int = 0
    remote_image_oom_downgrades: int = 0
    target_text_workers: int | None = None
    target_image_workers: int | None = None
    target_remote_text_workers: int | None = None
    target_remote_image_workers: int | None = None

    def __post_init__(self) -> None:
        self.text_workers = max(1, int(self.text_workers or 1))
        self.image_workers = max(1, int(self.image_workers or 1))
        self.remote_text_workers = max(0, int(self.remote_text_workers or 0))
        self.remote_image_workers = max(0, int(self.remote_image_workers or 0))
        if self.target_text_workers is None:
            self.target_text_workers = self.text_workers
        else:
            self.target_text_workers = max(1, int(self.target_text_workers or 1))
        if self.target_image_workers is None:
            self.target_image_workers = self.image_workers
        else:
            self.target_image_workers = max(1, int(self.target_image_workers or 1))
        self.target_text_workers = max(self.target_text_workers, self.text_workers)
        self.target_image_workers = max(self.target_image_workers, self.image_workers)
        if self.target_remote_text_workers is None:
            self.target_remote_text_workers = self.remote_text_workers
        else:
            self.target_remote_text_workers = max(
                0,
                int(self.target_remote_text_workers or 0),
            )
        if self.target_remote_image_workers is None:
            self.target_remote_image_workers = self.remote_image_workers
        else:
            self.target_remote_image_workers = max(
                0,
                int(self.target_remote_image_workers or 0),
            )
        self.target_remote_text_workers = max(
            self.target_remote_text_workers,
            self.remote_text_workers,
        )
        self.target_remote_image_workers = max(
            self.target_remote_image_workers,
            self.remote_image_workers,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelConcurrencyState":
        text_workers = max(1, int(getattr(args, "text_model_workers", 1) or 1))
        image_workers = max(1, int(getattr(args, "image_model_workers", 1) or 1))
        remote_text_workers = max(
            0,
            int(getattr(args, "remote_text_model_workers", 0) or 0),
        )
        remote_image_workers = max(
            0,
            int(getattr(args, "remote_image_model_workers", 0) or 0),
        )
        return cls(
            text_workers=text_workers,
            image_workers=image_workers,
            remote_text_workers=remote_text_workers,
            remote_image_workers=remote_image_workers,
            target_text_workers=text_workers,
            target_image_workers=image_workers,
            target_remote_text_workers=remote_text_workers,
            target_remote_image_workers=remote_image_workers,
        )

    def workers_for(self, model_kind: str, endpoint_pool: str = "local") -> int:
        prefix = "remote_" if endpoint_pool == "remote" else ""
        return int(getattr(self, f"{prefix}{model_kind}_workers"))

    def downgrade_after_oom(
        self,
        model_kind: str,
        attempted_workers: int | None = None,
        endpoint_pool: str = "local",
    ) -> int:
        worker_name = (
            f"remote_{model_kind}_workers"
            if endpoint_pool == "remote"
            else f"{model_kind}_workers"
        )
        downgrade_name = (
            f"remote_{model_kind}_oom_downgrades"
            if endpoint_pool == "remote"
            else f"{model_kind}_oom_downgrades"
        )
        current_workers = self.workers_for(model_kind, endpoint_pool)
        attempted_workers = max(1, int(attempted_workers or current_workers))
        next_workers = max(1, (attempted_workers + 1) // 2)
        if current_workers > 1:
            setattr(self, downgrade_name, getattr(self, downgrade_name) + 1)
        setattr(self, worker_name, next_workers)
        return next_workers

    def recover_after_non_oom_window(
        self,
        model_kind: str,
        endpoint_pool: str = "local",
    ) -> int:
        prefix = "remote_" if endpoint_pool == "remote" else ""
        worker_name = f"{prefix}{model_kind}_workers"
        target_name = f"target_{prefix}{model_kind}_workers"
        current_workers = int(getattr(self, worker_name))
        target_workers = int(getattr(self, target_name) or current_workers)
        if current_workers <= 0:
            return current_workers
        next_workers = min(target_workers, current_workers + 1)
        setattr(self, worker_name, next_workers)
        return next_workers

    def summary(self) -> dict[str, int]:
        return {
            "text_workers": self.text_workers,
            "image_workers": self.image_workers,
            "remote_text_workers": self.remote_text_workers,
            "remote_image_workers": self.remote_image_workers,
            "text_oom_downgrades": self.text_oom_downgrades,
            "image_oom_downgrades": self.image_oom_downgrades,
            "remote_text_oom_downgrades": self.remote_text_oom_downgrades,
            "remote_image_oom_downgrades": self.remote_image_oom_downgrades,
        }


def model_kind_for_asset(asset: dict[str, Any]) -> str:
    return "image" if asset.get("asset_type") == "image" else "text"


def is_oom_error(message: Any) -> bool:
    text = clean_text(message).casefold()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "out of memory",
            "cuda oom",
            "cuda error: out of memory",
            "torch.cuda.outofmemoryerror",
            "cudnn_status_alloc_failed",
            "memory allocation",
            "kv cache",
        )
    )


def extraction_record_from_result(task: ExtractionTask, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "cache_key": task.cache_key,
        "prompt_version": PROMPT_VERSION,
        "entity_id": task.entity["entity_id"],
        "entity_text": task.entity["cell_text"],
        "entity_wiki_title": task.entity["wiki_title"],
        "row_attributes": canonical_extraction_row_attributes(
            task.entity.get("row_attributes")
        ),
        "asset_id": task.asset["asset_id"],
        "asset_type": task.asset.get("asset_type"),
        "candidate_attribute_names": task.candidate_attribute_names,
        "attributes": result.get("attributes", []),
        "raw_response": result.get("raw_response", ""),
        "error": clean_text(result.get("error")),
    }
    if "error_class" in result:
        record["error_class"] = clean_text(result.get("error_class"))
    return record


def run_extraction_task(
    extractor: LocalAttributeExtractor,
    task: ExtractionTask,
    endpoint_pool: str = "local",
) -> dict[str, Any]:
    try:
        if endpoint_pool == "local":
            result = extractor.extract(
                task.asset,
                task.entity,
                task.candidate_attribute_names,
            )
        else:
            extract_from_pool = getattr(extractor, "extract_from_pool", None)
            if not callable(extract_from_pool):
                raise RuntimeError(
                    "extractor does not support remote endpoint pools"
                )
            result = extract_from_pool(
                task.asset,
                task.entity,
                task.candidate_attribute_names,
                endpoint_pool=endpoint_pool,
            )
    except TransientModelEndpointError:
        result = {
            "attributes": [],
            "raw_response": "",
            "error": "model endpoint temporarily unavailable",
            "error_class": "model_endpoint_transient",
        }
    except Exception as exc:
        result = {"attributes": [], "raw_response": "", "error": str(exc)}
    return extraction_record_from_result(task, result)


def run_extraction_task_group(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    workers: int,
    endpoint_pool: str = "local",
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        records = {}
        for task in tasks:
            record = run_extraction_task(extractor, task, endpoint_pool)
            records[task.cache_key] = record
            if on_record is not None:
                on_record(task.cache_key, record)
        return records

    records: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_task = {
            pool.submit(
                run_extraction_task,
                extractor,
                task,
                endpoint_pool,
            ): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                records[task.cache_key] = future.result()
            except Exception as exc:  # pragma: no cover - run_extraction_task catches normal failures.
                records[task.cache_key] = extraction_record_from_result(
                    task,
                    {"attributes": [], "raw_response": "", "error": str(exc)},
                )
            if on_record is not None:
                on_record(task.cache_key, records[task.cache_key])
    return records


def run_extraction_kind_adaptive(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    model_kind: str,
    state: ModelConcurrencyState,
    endpoint_pool: str = "local",
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    configured_workers = state.workers_for(model_kind, endpoint_pool)
    if configured_workers <= 0:
        return {}
    workers = max(1, min(configured_workers, len(tasks)))
    delayed_oom_records: dict[str, dict[str, Any]] = {}

    def handle_initial_record(cache_key: str, record: dict[str, Any]) -> None:
        if workers > 1 and is_oom_error(record.get("error")):
            delayed_oom_records[cache_key] = record
            return
        if on_record is not None:
            on_record(cache_key, record)

    records = run_extraction_task_group(
        extractor=extractor,
        tasks=tasks,
        workers=workers,
        endpoint_pool=endpoint_pool,
        on_record=handle_initial_record,
    )
    oom_tasks = [
        task
        for task in tasks
        if task.cache_key in delayed_oom_records
    ]
    if oom_tasks and workers > 1:
        next_workers = state.downgrade_after_oom(
            model_kind,
            attempted_workers=workers,
            endpoint_pool=endpoint_pool,
        )
        logging.warning(
            "%s %s model pool hit OOM-like errors with %d workers; retrying %d failed tasks serially and downgrading future workers to %d",
            endpoint_pool,
            model_kind,
            workers,
            len(oom_tasks),
            next_workers,
        )
        records.update(
            run_extraction_task_group(
                extractor=extractor,
                tasks=oom_tasks,
                workers=1,
                endpoint_pool=endpoint_pool,
                on_record=on_record,
            )
        )
    elif tasks and workers == configured_workers:
        state.recover_after_non_oom_window(model_kind, endpoint_pool)
    return records


def run_extraction_kind_distributed(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    model_kind: str,
    state: ModelConcurrencyState,
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Let independent local and remote worker groups consume one task queue."""

    configured_workers = {
        endpoint_pool: state.workers_for(model_kind, endpoint_pool)
        for endpoint_pool in ("local", "remote")
    }
    configured_workers = {
        endpoint_pool: workers
        for endpoint_pool, workers in configured_workers.items()
        if workers > 0
    }
    if len(configured_workers) == 1:
        endpoint_pool = next(iter(configured_workers))
        return run_extraction_kind_adaptive(
            extractor=extractor,
            tasks=tasks,
            model_kind=model_kind,
            state=state,
            endpoint_pool=endpoint_pool,
            on_record=on_record,
        )
    if not tasks:
        return {}

    task_queue: queue.Queue[ExtractionTask] = queue.Queue()
    result_queue: queue.Queue[
        tuple[str, ExtractionTask, dict[str, Any]]
    ] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    worker_specs: list[str] = []
    worker_limit = min(len(tasks), sum(configured_workers.values()))
    while len(worker_specs) < worker_limit:
        made_progress = False
        for endpoint_pool in ("local", "remote"):
            if worker_specs.count(endpoint_pool) >= configured_workers.get(
                endpoint_pool,
                0,
            ):
                continue
            worker_specs.append(endpoint_pool)
            made_progress = True
            if len(worker_specs) >= worker_limit:
                break
        if not made_progress:
            break

    def consume(endpoint_pool: str) -> None:
        while True:
            if endpoint_pool == "remote":
                wait_for_endpoint_pool = getattr(
                    extractor,
                    "wait_for_endpoint_pool",
                    None,
                )
                if callable(wait_for_endpoint_pool):
                    while not task_queue.empty() and not wait_for_endpoint_pool(
                        model_kind,
                        endpoint_pool,
                        0.2,
                    ):
                        pass
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            record = run_extraction_task(extractor, task, endpoint_pool)
            if (
                endpoint_pool == "remote"
                and record.get("error_class") == "model_endpoint_transient"
            ):
                # A priority reclaim can withdraw the route between the
                # availability check and the request lease. Return that task
                # to the shared queue so a local worker can finish it.
                task_queue.put(task)
                time.sleep(0.05)
                continue
            result_queue.put((endpoint_pool, task, record))

    records: dict[str, dict[str, Any]] = {}
    delayed_oom_tasks: dict[str, list[ExtractionTask]] = defaultdict(list)
    processed_by_pool: Counter[str] = Counter()
    with ThreadPoolExecutor(
        max_workers=len(worker_specs),
        thread_name_prefix=f"{model_kind}-model",
    ) as pool:
        futures = [pool.submit(consume, endpoint_pool) for endpoint_pool in worker_specs]
        for _ in tasks:
            endpoint_pool, task, record = result_queue.get()
            processed_by_pool[endpoint_pool] += 1
            records[task.cache_key] = record
            if (
                configured_workers[endpoint_pool] > 1
                and is_oom_error(record.get("error"))
            ):
                delayed_oom_tasks[endpoint_pool].append(task)
            elif on_record is not None:
                on_record(task.cache_key, record)
        for future in futures:
            future.result()

    for endpoint_pool, processed in processed_by_pool.items():
        oom_tasks = delayed_oom_tasks.get(endpoint_pool, [])
        spawned_workers = worker_specs.count(endpoint_pool)
        if oom_tasks:
            next_workers = state.downgrade_after_oom(
                model_kind,
                attempted_workers=spawned_workers,
                endpoint_pool=endpoint_pool,
            )
            logging.warning(
                "%s %s model pool hit OOM-like errors with %d workers; retrying %d failed tasks serially and downgrading future workers to %d",
                endpoint_pool,
                model_kind,
                spawned_workers,
                len(oom_tasks),
                next_workers,
            )
            records.update(
                run_extraction_task_group(
                    extractor=extractor,
                    tasks=oom_tasks,
                    workers=1,
                    endpoint_pool=endpoint_pool,
                    on_record=on_record,
                )
            )
        elif processed and spawned_workers == configured_workers[endpoint_pool]:
            state.recover_after_non_oom_window(model_kind, endpoint_pool)
    return records


def run_uncached_extraction_tasks(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    state: ModelConcurrencyState,
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[ExtractionTask]] = {"text": [], "image": []}
    for task in tasks:
        groups[model_kind_for_asset(task.asset)].append(task)

    records: dict[str, dict[str, Any]] = {}
    active_groups = [(kind, group_tasks) for kind, group_tasks in groups.items() if group_tasks]
    if len(active_groups) <= 1:
        for kind, group_tasks in active_groups:
            runner = (
                run_extraction_kind_distributed
                if state.workers_for(kind, "remote") > 0
                else run_extraction_kind_adaptive
            )
            records.update(
                runner(
                    extractor=extractor,
                    tasks=group_tasks,
                    model_kind=kind,
                    state=state,
                    on_record=on_record,
                )
            )
        return records

    with ThreadPoolExecutor(max_workers=len(active_groups)) as pool:
        futures = {
            pool.submit(
                (
                    run_extraction_kind_distributed
                    if state.workers_for(kind, "remote") > 0
                    else run_extraction_kind_adaptive
                ),
                extractor=extractor,
                tasks=group_tasks,
                model_kind=kind,
                state=state,
                on_record=on_record,
            ): kind
            for kind, group_tasks in active_groups
        }
        for future in as_completed(futures):
            records.update(future.result())
    return records


def cached_extraction_is_reusable(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if clean_text(record.get("error")) and not getattr(args, "cache_failed_model_outputs", False):
        return False
    if getattr(args, "refresh_invalid_model_cache", False) and should_refresh_cached_extraction(record):
        return False
    return True


def tasks_requiring_model_analysis(
    tasks: list[ExtractionTask],
    cache: ExtractionCache,
    args: argparse.Namespace,
    progress: ModelAnalysisProgress | None = None,
) -> list[ExtractionTask]:
    pending: list[ExtractionTask] = []
    seen: set[str] = set()
    for task in tasks:
        if task.cache_key in seen:
            continue
        seen.add(task.cache_key)
        transient = cache_get_transient(cache, task.cache_key)
        if transient is not None:
            if progress is not None:
                progress.mark(
                    task.cache_key,
                    "error" if clean_text(transient.get("error")) else "model",
                )
            continue
        cached = cache.get(task.cache_key)
        if cached:
            cached_record = cached
            if getattr(args, "reparse_cached_model_outputs", True):
                cached_record, changed = reparse_extraction_record(
                    cached_record,
                    task.candidate_attribute_names,
                )
                if changed:
                    cache.put(task.cache_key, cached_record)
            if cached_extraction_is_reusable(cached_record, args):
                if progress is not None:
                    progress.mark(task.cache_key, "cached")
                continue
        pending.append(task)
    return pending


def append_model_error_record(path_value: str, record: dict[str, Any]) -> None:
    if not clean_text(path_value) or not clean_text(record.get("error")):
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _MODEL_ERROR_LOG_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_extraction_tasks(
    *,
    extractor: LocalAttributeExtractor | None,
    cache: ExtractionCache,
    tasks: list[ExtractionTask],
    args: argparse.Namespace,
    state: ModelConcurrencyState,
    progress: ModelAnalysisProgress | None = None,
) -> list[tuple[ExtractionTask, dict[str, Any]]]:
    resolved_by_key: dict[str, dict[str, Any]] = {}
    uncached_by_key: dict[str, ExtractionTask] = {}
    for task in tasks:
        if task.cache_key in resolved_by_key or task.cache_key in uncached_by_key:
            continue
        transient = cache_get_transient(cache, task.cache_key)
        if transient is not None:
            resolved_by_key[task.cache_key] = transient
            if progress is not None:
                progress.mark(
                    task.cache_key,
                    "error" if clean_text(transient.get("error")) else "model",
                )
            continue
        cached = cache.get(task.cache_key)
        if cached:
            cached_record = cached
            if getattr(args, "reparse_cached_model_outputs", True):
                cached_record, changed = reparse_extraction_record(
                    cached_record,
                    task.candidate_attribute_names,
                )
                if changed:
                    cache.put(task.cache_key, cached_record)
            if cached_extraction_is_reusable(cached_record, args):
                resolved_by_key[task.cache_key] = cached_record
                if progress is not None:
                    progress.mark(task.cache_key, "cached")
                continue
        uncached_by_key[task.cache_key] = task

    def store_model_record(cache_key: str, record: dict[str, Any]) -> None:
        resolved_by_key[cache_key] = record
        has_error = bool(clean_text(record.get("error")))
        if not has_error or getattr(args, "cache_failed_model_outputs", False):
            cache.put(cache_key, record)
        else:
            cache_put_transient(cache, cache_key, record)
        if has_error:
            append_model_error_record(getattr(args, "model_attribute_errors_path", ""), record)
        if progress is not None:
            progress.mark(cache_key, "error" if has_error else "model")

    if uncached_by_key and extractor is None:
        raise RuntimeError("Model analysis is required but no extractor was initialized")
    model_records = (
        run_uncached_extraction_tasks(
            extractor=extractor,
            tasks=list(uncached_by_key.values()),
            state=state,
            on_record=store_model_record,
        )
        if uncached_by_key and extractor is not None
        else {}
    )
    for cache_key, record in model_records.items():
        if cache_key not in resolved_by_key:
            store_model_record(cache_key, record)

    if (
        extractor is not None
        and getattr(extractor, "abort_on_transient_error", False)
        and any(
            record.get("error_class") == "model_endpoint_transient"
            for record in model_records.values()
        )
    ):
        raise TransientModelEndpointError(
            "transient model endpoint failure encountered; successful "
            "extractions were cached and the run can be resumed"
        )

    return [(task, resolved_by_key[task.cache_key]) for task in tasks if task.cache_key in resolved_by_key]


def source_splits(source_records: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    ratios = [args.train_ratio, args.dev_ratio, args.test_ratio]
    total = sum(ratios)
    if total <= 0:
        raise ValueError("train/dev/test ratios must sum to a positive value")
    ratios = [value / total for value in ratios]
    groups: dict[str, list[str]] = defaultdict(list)
    for record in source_records:
        key = clean_text(record.get("page_title")) if args.split_by == "page_title" else ""
        groups[key or record["source_table_id"]].append(record["source_table_id"])
    keys = sorted(groups, key=lambda key: stable_hash("split", args.seed, key))
    train_n = int(len(keys) * ratios[0])
    dev_n = int(len(keys) * ratios[1])
    if len(keys) >= 3:
        train_n = max(1, train_n) if ratios[0] > 0 else 0
        dev_n = max(1, dev_n) if ratios[1] > 0 else 0
        if train_n + dev_n >= len(keys):
            dev_n = max(0, len(keys) - train_n - 1)
    split_keys = {
        "train": keys[:train_n],
        "dev": keys[train_n : train_n + dev_n],
        "test": keys[train_n + dev_n :],
    }
    splits: dict[str, Any] = {}
    for split, selected in split_keys.items():
        splits[split] = {
            "source_table_ids": sorted({source_id for key in selected for source_id in groups[key]}),
            "query_table_ids": [],
            "data_lake_table_ids": [],
        }
    splits["split_key"] = "page_title_or_source_table_id" if args.split_by == "page_title" else "source_table_id"
    splits["note"] = (
        "source-level split; data_lake contains generated targets for "
        "queryable tables and source-table references for rejected tables"
    )
    return splits


def split_map(splits: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split, payload in splits.items():
        if isinstance(payload, dict):
            for source_id in payload.get("source_table_ids", []):
                result[source_id] = split
    return result


def choose_entity_column(
    table: dict[str, Any],
    *,
    min_linked_rows: int = 0,
) -> int | None:
    candidates = [
        int(index)
        for index in (
            table.get("metadata", {}).get("candidate_entity_columns", []) or []
        )
        if linked_entity_row_count(table, int(index)) >= min_linked_rows
    ]
    if not candidates:
        return None
    profiles = column_profiles(table)
    candidates.sort(
        key=lambda idx: (
            -float(profiles.get(idx, {}).get("wiki_link_ratio", 0.0)),
            idx,
        )
    )
    return candidates[0]


def linked_entity_row_count(table: dict[str, Any], entity_col: int) -> int:
    return sum(
        1
        for row in table.get("rows", [])
        if clean_text(get_cell(row, entity_col).get("wiki_title"))
    )


def candidate_attribute_columns(table: dict[str, Any], entity_col: int, min_non_empty_ratio: float) -> list[int]:
    profiles = column_profiles(table)
    cols: list[int] = []
    for column in table.get("columns", []):
        try:
            idx = int(column.get("column_index"))
        except (TypeError, ValueError):
            continue
        if idx == entity_col:
            continue
        if float(profiles.get(idx, {}).get("non_empty_ratio", 0.0)) < min_non_empty_ratio:
            continue
        cols.append(idx)
    return cols


def extraction_row_attributes(
    table: dict[str, Any],
    row: dict[str, Any],
    entity_col: int,
) -> list[dict[str, Any]]:
    attributes: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for fallback, column in enumerate(table.get("columns", [])):
        if not isinstance(column, dict):
            continue
        try:
            column_index = int(column.get("column_index", fallback))
        except (TypeError, ValueError):
            continue
        if column_index in seen_indices:
            continue
        seen_indices.add(column_index)
        name = sanitize_cell_text_for_model(
            clean_text(column.get("column_name"))
            or clean_text(column.get("name"))
            or f"col_{column_index}"
        )
        value = sanitize_cell_text_for_model(get_cell_text(row, column_index))
        if not name or not value:
            continue
        attributes.append(
            {
                "name": name,
                "value": value,
                "is_entity": column_index == entity_col,
            }
        )
    return attributes


def row_id(row: dict[str, Any], fallback: int) -> int:
    try:
        return int(row.get("row_id", fallback))
    except (TypeError, ValueError):
        return fallback


def project_selected_rows(
    table: dict[str, Any],
    column_indices: list[int],
    source_row_ids: set[int],
    *,
    min_required_cols: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    source_rows: list[int] = []
    required = set(column_indices[:min_required_cols])
    for fallback, source_row in enumerate(table.get("rows", [])):
        source_row_id = row_id(source_row, fallback)
        if source_row_id not in source_row_ids:
            continue
        values = {idx: get_cell_text(source_row, idx) for idx in column_indices}
        if any(not values.get(idx) for idx in required):
            continue
        cells = []
        for out_idx, source_idx in enumerate(column_indices):
            cell = dict(get_cell(source_row, source_idx))
            cell["column_index"] = out_idx
            cell["source_column_index"] = source_idx
            cell["column_name"] = get_column_name(table, source_idx)
            cell["text"] = sanitize_cell_text_for_model(values.get(source_idx, ""))
            cells.append(cell)
        rows.append({"row_id": len(rows), "source_row_id": source_row_id, "cells": cells})
        source_rows.append(source_row_id)
    return rows, source_rows


def table_record(
    *,
    table_id: str,
    role: str,
    split: str,
    source_table: dict[str, Any],
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_indices: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": role,
        "split": split,
        "source_table_id": source_table["source_table_id"],
        "page_title": clean_text(source_table.get("page_title")),
        "caption": clean_text(source_table.get("caption")),
        "section_title": clean_text(source_table.get("section_title")),
        "columns": make_columns(source_table, column_indices),
        "rows": rows,
        "source_column_indices": column_indices,
        "source_row_indices": source_row_indices,
        "provenance": {
            "builder": clean_text(source_table.get("provenance_builder")) or "build_mm_joinability_dataset.py",
            "source_file": source_table.get("source_file"),
        },
        **extra,
    }


def raw_data_lake_record(source_table: dict[str, Any], split: str) -> dict[str, Any]:
    source_table_id = clean_text(source_table.get("source_table_id"))
    if not source_table_id:
        raise ValueError("source table is missing source_table_id")
    table_id = f"dl_raw_{source_table_id}"
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": "raw_data_lake_table",
        "split": split,
        "source_table_id": source_table_id,
        "source_table_ref": {
            "artifact": "source_tables",
            "source_table_id": source_table_id,
        },
        "queryable": False,
        "reason": "no_column_met_recovered_value_ratio",
    }


def context_columns(table: dict[str, Any], excluded: set[int], limit: int) -> list[int]:
    profiles = column_profiles(table)
    candidates: list[tuple[float, float, int]] = []
    for column in table.get("columns", []):
        idx = int(column.get("column_index"))
        if idx in excluded:
            continue
        profile = profiles.get(idx, {})
        candidates.append(
            (
                float(profile.get("non_empty_ratio", 0.0)),
                -float(profile.get("unique_ratio", 1.0)),
                idx,
            )
        )
    candidates.sort(reverse=True)
    if limit <= 0:
        return [idx for _non_empty, _unique, idx in candidates]
    return [idx for _non_empty, _unique, idx in candidates[:limit]]


def _explicit_join_fallback_selected(
    source_table: dict[str, Any], args: argparse.Namespace
) -> bool:
    ratio = configured_explicit_join_fallback_ratio(args)
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    draw = int(
        stable_hash(
            "explicit-join-fallback",
            int(getattr(args, "seed", 13)),
            source_table.get("source_table_id"),
            length=40,
        ),
        16,
    )
    return draw / float(1 << 160) < ratio


def _explicit_join_candidate_columns(
    source_table: dict[str, Any],
    entity_col: int,
    args: argparse.Namespace,
) -> list[int]:
    rows = source_table.get("rows", [])
    if not rows:
        return []
    query_rows_per_table = configured_query_rows_per_table(args)
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    min_non_empty_ratio = float(
        getattr(args, "min_column_non_empty_ratio", 0.5)
    )
    candidates: list[int] = []
    for fallback, column in enumerate(source_table.get("columns", [])):
        try:
            column_index = int(column.get("column_index", fallback))
        except (AttributeError, TypeError, ValueError):
            continue
        if column_index == entity_col:
            continue
        non_empty_join_rows = 0
        query_eligible_rows = 0
        for row in rows:
            if not get_cell_text(row, column_index):
                continue
            non_empty_join_rows += 1
            if get_cell_text(row, entity_col):
                query_eligible_rows += 1
        if non_empty_join_rows / len(rows) < min_non_empty_ratio:
            continue
        if non_empty_join_rows < min_target_rows:
            continue
        if query_eligible_rows < query_rows_per_table:
            continue
        candidates.append(column_index)
    return candidates


def _explicit_join_candidate_id(
    source_table_id: str,
    entity_col: int,
    join_col: int,
) -> str:
    return f"explicit_candidate_{stable_hash(source_table_id, entity_col, join_col)}"


def _explicit_join_context_partition(
    *,
    source_table: dict[str, Any],
    entity_col: int,
    join_columns: list[int],
    args: argparse.Namespace,
) -> tuple[list[int], list[int]]:
    """Partition ordinary columns once for a source's explicit variants.

    Every explicit query exposes its own join column.  All other join columns
    are target-only, while the ordinary context columns are split globally
    into query-only and target-only pools.  This keeps a query from one
    variant from sharing a source column with a target from another variant.
    """
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    ordinary = context_columns(
        source_table,
        {entity_col, *join_columns},
        0,
    )
    ordinary = [
        column_index
        for column_index in ordinary
        if sum(
            bool(get_cell_text(source_row, column_index))
            for source_row in source_table.get("rows", [])
        )
        >= min_target_rows
    ]
    max_target_context = int(getattr(args, "max_target_context_attrs", 2))
    if max_target_context <= 0:
        target_context = list(ordinary)
    else:
        target_context = ordinary[:max_target_context]
    query_context_pool = [
        column_index
        for column_index in ordinary
        if column_index not in target_context
    ]
    max_query_context = int(getattr(args, "max_query_context_attrs", 1))
    if max_query_context <= 0:
        query_context = query_context_pool
    else:
        query_context = query_context_pool[:max_query_context]
    return query_context, target_context


def _build_explicit_join_candidate(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int,
    join_col: int,
    query_context: list[int],
    target_context: list[int],
    rejected_multimodal_reason: str,
    rejected_multimodal_decision: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Build one deterministic explicit query/target candidate specification."""
    seed = int(getattr(args, "seed", 13))
    source_table_id = str(source_table["source_table_id"])
    query_cols = [entity_col, join_col, *query_context]
    target_cols = [join_col, *target_context]
    eligible_source_rows = [
        row_id(source_row, fallback)
        for fallback, source_row in enumerate(source_table.get("rows", []))
        if get_cell_text(source_row, entity_col)
        and get_cell_text(source_row, join_col)
    ]
    eligible_source_rows.sort(
        key=lambda source_row_id: (
            stable_hash(
                "explicit-join-query-row",
                seed,
                source_table_id,
                join_col,
                source_row_id,
                length=40,
            ),
            source_row_id,
        )
    )
    query_rows_per_table = configured_query_rows_per_table(args)
    selected_source_rows = eligible_source_rows[:query_rows_per_table]
    query_rows, query_source_rows = project_selected_rows(
        source_table,
        query_cols,
        set(selected_source_rows),
        min_required_cols=2,
    )
    all_source_rows = {
        row_id(source_row, fallback)
        for fallback, source_row in enumerate(source_table.get("rows", []))
    }
    target_rows, target_source_rows = project_selected_rows(
        source_table,
        target_cols,
        all_source_rows,
        min_required_cols=1,
    )
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    if len(query_rows) != query_rows_per_table:
        return None
    if len(target_rows) < min_target_rows:
        return None
    if not set(query_source_rows).issubset(target_source_rows):
        return None

    join_col_name = get_column_name(source_table, join_col)
    chain_id = f"chain_explicit_{stable_hash(source_table_id, entity_col, join_col)}"
    query_table_id = f"query_{stable_hash(chain_id, 'query')}"
    target_table_id = f"target_{stable_hash(chain_id, 'target')}"
    join_attribute = {
        "source_column_index": join_col,
        "column_name": join_col_name,
        "role": "visible_join_column",
        "hidden_in_query": False,
        "selected_rows": len(query_rows),
        "target_rows": len(target_rows),
    }
    return {
        "reason": "explicit_join_fallback",
        "candidate_id": _explicit_join_candidate_id(
            source_table_id, entity_col, join_col
        ),
        "source_table_id": source_table_id,
        "split": split,
        "entity_column_index": entity_col,
        "join_column_index": join_col,
        "join_column_name": join_col_name,
        "query_column_indices": query_cols,
        "target_column_indices": target_cols,
        "query_context_column_indices": query_context,
        "target_context_column_indices": target_context,
        "query_context_col_names": [
            get_column_name(source_table, column_index)
            for column_index in query_context
        ],
        "target_context_col_names": [
            get_column_name(source_table, column_index)
            for column_index in target_context
        ],
        "selected_source_row_ids": query_source_rows,
        "query_table_id": query_table_id,
        "target_table_id": target_table_id,
        "chain_id": chain_id,
        "join_attribute": join_attribute,
        "rejected_multimodal_reason": rejected_multimodal_reason,
        "rejected_multimodal_decision": dict(
            rejected_multimodal_decision
            or {"reason": rejected_multimodal_reason}
        ),
    }


def _materialize_explicit_join_candidate(
    *,
    source_table: dict[str, Any],
    split: str,
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Materialize one previously certified explicit candidate."""
    entity_col = int(candidate["entity_column_index"])
    join_col = int(candidate["join_column_index"])
    query_context = [int(value) for value in candidate.get("query_context_column_indices", [])]
    target_context = [int(value) for value in candidate.get("target_context_column_indices", [])]
    query_cols = [entity_col, join_col, *query_context]
    target_cols = [join_col, *target_context]
    query_rows, query_source_rows = project_selected_rows(
        source_table,
        query_cols,
        {int(value) for value in candidate["selected_source_row_ids"]},
        min_required_cols=2,
    )
    all_source_rows = {
        row_id(source_row, fallback)
        for fallback, source_row in enumerate(source_table.get("rows", []))
    }
    target_rows, target_source_rows = project_selected_rows(
        source_table,
        target_cols,
        all_source_rows,
        min_required_cols=1,
    )
    query_table_id = str(candidate["query_table_id"])
    target_table_id = str(candidate["target_table_id"])
    chain_id = str(candidate["chain_id"])
    join_col_name = get_column_name(source_table, join_col)
    join_attribute = {
        **dict(candidate.get("join_attribute") or {}),
        "source_column_index": join_col,
        "column_name": join_col_name,
        "selected_rows": len(query_rows),
        "target_rows": len(target_rows),
        "hidden_in_query": False,
    }
    query_table = table_record(
        table_id=query_table_id,
        role="query",
        split=split,
        source_table=source_table,
        column_indices=query_cols,
        rows=query_rows,
        source_row_indices=query_source_rows,
        extra={
            "chain_id": chain_id,
            "chain_ids": [chain_id],
            "construction_type": "explicit_visible_join",
            "query_entity_col": entity_col,
            "query_entity_col_name": get_column_name(source_table, entity_col),
            "join_col": join_col,
            "join_col_name": join_col_name,
            "hidden_attributes": [],
            "target_table_ids": [target_table_id],
            "query_context_col_names": [
                get_column_name(source_table, column_index)
                for column_index in query_context
            ],
            "row_view_index": 0,
        },
    )
    target_table = table_record(
        table_id=target_table_id,
        role="target_data_lake_table",
        split=split,
        source_table=source_table,
        column_indices=target_cols,
        rows=target_rows,
        source_row_indices=target_source_rows,
        extra={
            "chain_id": chain_id,
            "construction_type": "explicit_visible_join",
            "queryable_source_table": True,
            "join_col": join_col,
            "join_col_name": join_col_name,
            "target_context_col_names": [
                get_column_name(source_table, column_index)
                for column_index in target_context
            ],
        },
    )
    qrel = {
        "query_table_id": query_table_id,
        "target_table_id": target_table_id,
        "data_lake_table_id": target_table_id,
        "rel": 3,
        "split": split,
        "chain_id": chain_id,
        "row_view_index": 0,
        "source_table_id": str(source_table["source_table_id"]),
        "join_attribute": join_attribute,
        "reason": "explicit_visible_join_column",
    }
    decision = {
        "reason": "explicit_join_fallback",
        "rejected_multimodal_reason": candidate.get(
            "rejected_multimodal_reason", ""
        ),
        "rejected_multimodal_decision": dict(
            candidate.get("rejected_multimodal_decision") or {}
        ),
        "entity_column_index": entity_col,
        "join_column_index": join_col,
        "join_column_name": join_col_name,
        "qualified_columns": [join_attribute],
        "explicit_join_candidate": dict(candidate),
    }
    return [query_table], [target_table], [qrel], decision


def build_explicit_join_fallback_records(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int | None,
    rejected_multimodal_reason: str,
    args: argparse.Namespace,
    rejected_multimodal_decision: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
] | None:
    """Turn a sampled multimodal rejection into one visible-column join pair.

    ``match_implicit`` uses :func:`build_explicit_join_fallback_candidates`
    directly so a source table can contribute more than one query.  Ratio mode
    retains the historical one-candidate behavior through this wrapper.
    """
    if entity_col is None or (
        not force
        and not _explicit_join_fallback_selected(source_table, args)
    ):
        return None

    candidate_columns = _explicit_join_candidate_columns(
        source_table, entity_col, args
    )
    if not candidate_columns:
        return None
    seed = int(getattr(args, "seed", 13))
    source_table_id = str(source_table["source_table_id"])
    join_column = min(
        candidate_columns,
        key=lambda column_index: (
            stable_hash(
                "explicit-join-column",
                seed,
                source_table_id,
                column_index,
                length=40,
            ),
            column_index,
        ),
    )
    candidates = build_explicit_join_fallback_candidates(
        source_table=source_table,
        split=split,
        entity_col=entity_col,
        rejected_multimodal_reason=rejected_multimodal_reason,
        rejected_multimodal_decision=rejected_multimodal_decision,
        args=args,
        join_columns=[join_column],
    )
    if not candidates:
        return None
    return _materialize_explicit_join_candidate(
        source_table=source_table,
        split=split,
        candidate=candidates[0],
        args=args,
    )


def build_explicit_join_fallback_candidates(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int | None,
    rejected_multimodal_reason: str,
    args: argparse.Namespace,
    rejected_multimodal_decision: dict[str, Any] | None = None,
    force: bool = False,
    join_columns: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build all viable explicit query candidates for one source table.

    Candidate join columns are target-only for every sibling variant.  The
    ordinary context columns are partitioned once into a query-only pool and a
    target-only pool, so a query from one variant cannot accidentally match a
    target produced for another variant from the same source table.
    """
    if entity_col is None or (
        not force
        and not _explicit_join_fallback_selected(source_table, args)
    ):
        return []
    candidates = list(join_columns) if join_columns is not None else (
        _explicit_join_candidate_columns(source_table, entity_col, args)
    )
    if not candidates:
        return []
    seed = int(getattr(args, "seed", 13))
    source_table_id = str(source_table["source_table_id"])
    candidates.sort(
        key=lambda column_index: (
            stable_hash(
                "explicit-join-column",
                seed,
                source_table_id,
                column_index,
                length=40,
            ),
            column_index,
        )
    )
    max_variants = int(getattr(args, "max_query_tables_per_source_table", 0))
    if max_variants > 0:
        candidates = candidates[:max_variants]
    query_context, target_context = _explicit_join_context_partition(
        source_table=source_table,
        entity_col=entity_col,
        join_columns=candidates,
        args=args,
    )
    output: list[dict[str, Any]] = []
    for join_col in candidates:
        candidate = _build_explicit_join_candidate(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            join_col=join_col,
            query_context=query_context,
            target_context=target_context,
            rejected_multimodal_reason=rejected_multimodal_reason,
            rejected_multimodal_decision=rejected_multimodal_decision,
            args=args,
        )
        if candidate is not None:
            output.append(candidate)
    return output


def rejected_table_join_records(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int | None,
    decision: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    mode = configured_explicit_join_fallback_mode(args)
    if mode == "match_implicit":
        explicit_candidates = build_explicit_join_fallback_candidates(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            rejected_multimodal_reason=str(decision["reason"]),
            rejected_multimodal_decision=decision,
            args=args,
            force=True,
        )
        if explicit_candidates:
            return (
                [],
                [raw_data_lake_record(source_table, split)],
                [],
                {
                    **decision,
                    "explicit_join_candidates": explicit_candidates,
                    # Keep the singular field for callers written against the
                    # pre-query-level candidate schema.
                    "explicit_join_candidate": explicit_candidates[0],
                },
            )
        return [], [raw_data_lake_record(source_table, split)], [], decision
    if mode == "disabled":
        return [], [raw_data_lake_record(source_table, split)], [], decision
    explicit_records = build_explicit_join_fallback_records(
        source_table=source_table,
        split=split,
        entity_col=entity_col,
        rejected_multimodal_reason=str(decision["reason"]),
        rejected_multimodal_decision=decision,
        args=args,
    )
    if explicit_records is not None:
        return explicit_records
    return [], [raw_data_lake_record(source_table, split)], [], decision


def select_best_qualified_column(qualified_cols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not qualified_cols:
        return []
    return [max(qualified_cols, key=lambda item: float(item["recovered_value_ratio"]))]


def multi_attribute_context_layout(
    *,
    source_table: dict[str, Any],
    entity_col: int,
    qualified_cols: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], list[int], list[int]]]:
    """Assign safe query/target contexts for all qualified bridge columns.

    Every qualified bridge column is target-only. Ordinary context columns are
    partitioned once per source table so no sibling query and target share a
    source column. Narrow tables fall back to the single best bridge column.
    """
    if not qualified_cols:
        return []

    ordered_qualified = sorted(
        qualified_cols,
        key=lambda item: (
            -float(item["recovered_value_ratio"]),
            int(item["column_index"]),
        ),
    )
    all_qualified_indices = {
        int(qualified["column_index"]) for qualified in ordered_qualified
    }
    max_variants = int(getattr(args, "max_query_tables_per_source_table", 0))
    if max_variants > 0:
        ordered_qualified = ordered_qualified[:max_variants]

    ordinary_contexts = context_columns(
        source_table,
        {entity_col, *all_qualified_indices},
        0,
    )
    query_context_width = max(
        1, int(getattr(args, "max_query_context_attrs", 1))
    )
    if (
        len(ordered_qualified) > 1
        and len(ordinary_contexts) >= query_context_width + 1
    ):
        seed = int(getattr(args, "seed", 13))
        source_table_id = str(source_table["source_table_id"])
        ordered_contexts = sorted(
            ordinary_contexts,
            key=lambda column_index: (
                stable_hash(
                    "multi-attribute-context",
                    seed,
                    source_table_id,
                    column_index,
                    length=40,
                ),
                column_index,
            ),
        )
        target_context = [ordered_contexts[0]]
        query_context_pool = ordered_contexts[1:]
        query_contexts = [
            list(context)
            for context in combinations(
                query_context_pool,
                query_context_width,
            )
        ]
        if query_contexts:
            return [
                (
                    qualified,
                    query_contexts[index % len(query_contexts)],
                    target_context,
                )
                for index, qualified in enumerate(ordered_qualified)
            ]

    best = select_best_qualified_column(ordered_qualified)[0]
    join_col = int(best["column_index"])
    other_cols = context_columns(source_table, {entity_col, join_col}, 0)
    if not other_cols:
        return []
    query_context = other_cols[:query_context_width]
    target_context_pool = [
        column_index
        for column_index in other_cols
        if column_index not in query_context
    ]
    target_context = target_context_pool[:1]
    return [(best, query_context, target_context)]


def visible_query_fingerprint(
    *,
    source_table: dict[str, Any],
    query_cols: list[int],
    query_rows: list[dict[str, Any]],
) -> str:
    visible_payload = {
        "column_names": [
            get_column_name(source_table, column_index)
            for column_index in query_cols
        ],
        "rows": [
            [
                clean_text(cell.get("text"))
                for cell in row.get("cells", [])
            ]
            for row in query_rows
        ],
    }
    return stable_hash(
        "visible-query",
        json.dumps(visible_payload, ensure_ascii=False, sort_keys=True),
        length=40,
    )


def required_recovered_row_count(
    valid_entity_rows: int,
    query_rows_per_table: int,
    min_ratio: float,
) -> int:
    denominator = min(valid_entity_rows, query_rows_per_table)
    threshold = Decimal(denominator) * Decimal(str(min_ratio))
    return int(threshold.to_integral_value(rounding=ROUND_CEILING))


def recovery_column_profile(
    *,
    valid_source_rows: set[int],
    recovered_source_rows: set[int],
    query_rows_per_table: int,
    min_recovery_denominator: int,
    min_ratio: float,
) -> dict[str, Any] | None:
    valid_count = len(valid_source_rows)
    recovered_count = len(recovered_source_rows & valid_source_rows)
    required_count = required_recovered_row_count(
        valid_count,
        query_rows_per_table,
        min_ratio,
    )
    if valid_count < min_recovery_denominator or recovered_count < required_count:
        return None
    return {
        "eligible_rows": valid_count,
        "valid_entity_rows": valid_count,
        "recovered_rows": recovered_count,
        "required_recovered_rows": required_count,
        "recovered_value_ratio": round(recovered_count / max(1, valid_count), 6),
    }


def select_query_source_rows(
    *,
    source_row_order: list[int],
    recovered_source_rows: set[int],
    query_rows_per_table: int,
    required_recovered_rows: int,
) -> list[int]:
    recovered = [row for row in source_row_order if row in recovered_source_rows]
    unrecovered = [row for row in source_row_order if row not in recovered_source_rows]
    if len(source_row_order) < query_rows_per_table or len(recovered) < required_recovered_rows:
        return []
    selected = recovered[:query_rows_per_table]
    selected.extend(unrecovered[: query_rows_per_table - len(selected)])
    return selected


def select_query_source_row_views(
    *,
    source_row_order: list[int],
    recovered_source_rows: set[int],
    query_rows_per_table: int,
    required_recovered_rows: int,
    max_views: int,
) -> list[list[int]]:
    """Select deterministic, disjoint query views with a recovery floor per view."""
    if query_rows_per_table <= 0 or required_recovered_rows < 0 or max_views < 0:
        raise ValueError("query row-view limits must be non-negative and row count positive")
    if required_recovered_rows > query_rows_per_table:
        return []

    recovered = [row for row in source_row_order if row in recovered_source_rows]
    total_view_capacity = len(source_row_order) // query_rows_per_table
    recovery_view_capacity = (
        total_view_capacity
        if required_recovered_rows == 0
        else len(recovered) // required_recovered_rows
    )
    view_count = min(total_view_capacity, recovery_view_capacity)
    if max_views > 0:
        view_count = min(view_count, max_views)
    if view_count <= 0:
        return []
    if view_count == 1:
        selected = select_query_source_rows(
            source_row_order=source_row_order,
            recovered_source_rows=recovered_source_rows,
            query_rows_per_table=query_rows_per_table,
            required_recovered_rows=required_recovered_rows,
        )
        return [selected] if selected else []

    views: list[list[int]] = [[] for _ in range(view_count)]
    reserved_rows: set[int] = set()
    for view_index in range(view_count):
        start = view_index * required_recovered_rows
        stop = start + required_recovered_rows
        reserved = recovered[start:stop]
        views[view_index].extend(reserved)
        reserved_rows.update(reserved)

    remaining = [row for row in source_row_order if row not in reserved_rows]
    remaining_index = 0
    row_positions = {row: index for index, row in enumerate(source_row_order)}
    for view in views:
        needed = query_rows_per_table - len(view)
        view.extend(remaining[remaining_index : remaining_index + needed])
        remaining_index += needed
        view.sort(key=row_positions.__getitem__)
    return views


def configured_query_rows_per_table(args: argparse.Namespace) -> int:
    query_rows = int(getattr(args, "query_rows_per_table", 5))
    min_output_rows = int(getattr(args, "min_rows_per_output_table", 2))
    if query_rows <= 0:
        raise ValueError("query_rows_per_table must be positive")
    if min_output_rows > query_rows:
        raise ValueError(
            "min_rows_per_output_table cannot exceed query_rows_per_table"
        )
    return query_rows


def configured_max_train_query_row_views_per_join(
    args: argparse.Namespace,
) -> int:
    max_views = int(getattr(args, "max_train_query_row_views_per_join", 1))
    if max_views < 0:
        raise ValueError("max_train_query_row_views_per_join must be non-negative")
    return max_views


def configured_explicit_join_fallback_mode(args: argparse.Namespace) -> str:
    # Namespaces created by older callers did not carry a mode and retain the
    # historical ratio behavior. CLI parsers also default to ratio for
    # backward-compatible runs; match_implicit is opt-in.
    mode = str(getattr(args, "explicit_join_fallback_mode", "ratio"))
    if mode not in EXPLICIT_JOIN_FALLBACK_MODES:
        raise ValueError(
            "explicit_join_fallback_mode must be one of "
            + ", ".join(EXPLICIT_JOIN_FALLBACK_MODES)
        )
    return mode


def configured_explicit_join_fallback_ratio(args: argparse.Namespace) -> float:
    ratio = float(
        getattr(
            args,
            "explicit_join_fallback_ratio",
            DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO,
        )
    )
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("explicit_join_fallback_ratio must be within [0, 1]")
    return ratio


def select_balanced_explicit_join_candidates(
    *,
    candidate_splits: dict[str, str],
    implicit_query_counts: dict[str, int],
    args: argparse.Namespace,
) -> tuple[set[str], dict[str, int]]:
    """Select one query-level explicit candidate per implicit query per split."""
    candidates_by_split: dict[str, list[str]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for source_table_id, split in candidate_splits.items():
        if split not in candidates_by_split:
            raise ValueError(
                f"explicit join candidate has invalid split: {split}"
            )
        candidates_by_split[split].append(source_table_id)

    seed = int(getattr(args, "seed", 13))
    selected: set[str] = set()
    candidate_counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        needed = int(implicit_query_counts.get(split, 0))
        if needed < 0:
            raise ValueError("implicit query count must be non-negative")
        candidates = candidates_by_split[split]
        candidate_counts[split] = len(candidates)
        if len(candidates) < needed:
            raise ValueError(
                "insufficient explicit join candidates for balanced "
                f"{split} split: required={needed}, available={len(candidates)}"
            )
        candidates.sort(
            key=lambda source_table_id: (
                stable_hash(
                    "explicit-join-balance",
                    seed,
                    split,
                    source_table_id,
                    length=40,
                ),
                source_table_id,
            )
        )
        selected.update(candidates[:needed])
    return selected, candidate_counts


def materialize_balanced_explicit_join_candidate(
    *,
    source_table: dict[str, Any],
    split: str,
    candidate_decision: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Rebuild a previously certified explicit candidate after selection."""
    candidate = dict(candidate_decision)
    if "selected_source_row_ids" not in candidate:
        # Upgrade decisions written by the pre-query-level implementation so
        # staged WDC runs can still be resumed safely.
        generated = build_explicit_join_fallback_candidates(
            source_table=source_table,
            split=split,
            entity_col=int(candidate["entity_column_index"]),
            rejected_multimodal_reason=str(
                candidate.get("rejected_multimodal_reason")
                or candidate.get("reason")
                or "explicit_join_fallback"
            ),
            rejected_multimodal_decision=(
                candidate.get("rejected_multimodal_decision")
                if isinstance(candidate.get("rejected_multimodal_decision"), dict)
                else candidate
            ),
            args=args,
            force=True,
            join_columns=[int(candidate["join_column_index"])],
        )
        if not generated:
            raise ValueError(
                "legacy explicit join candidate is no longer viable: "
                f"{source_table.get('source_table_id')}"
            )
        candidate = {**generated[0], **candidate}
    records = _materialize_explicit_join_candidate(
        source_table=source_table,
        split=split,
        candidate=candidate,
        args=args,
    )
    if records[3].get("join_column_index") != candidate.get(
        "join_column_index"
    ) or records[3].get("explicit_join_candidate", {}).get(
        "candidate_id"
    ) != candidate.get("candidate_id"):
        raise ValueError(
            "explicit join candidate changed during balanced materialization: "
            f"{source_table.get('source_table_id')}"
        )
    return records


def extraction_cache_key(
    *,
    asset_id: str,
    entity_id: str,
    candidate_attribute_names: list[str],
    asset_type: str,
    args: argparse.Namespace,
    row_attributes: list[dict[str, Any]] | None = None,
) -> str:
    model = args.image_model_name if asset_type == "image" else args.text_model_name
    row_context = json.dumps(
        canonical_extraction_row_attributes(row_attributes),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash(
        PROMPT_VERSION,
        model,
        asset_id,
        entity_id,
        "|".join(candidate_attribute_names),
        row_context,
        length=24,
    )


def reparse_extraction_record(
    record: dict[str, Any],
    candidate_attribute_names: list[str],
) -> tuple[dict[str, Any], bool]:
    raw_response = clean_text(record.get("raw_response"))
    if not raw_response:
        return record, False
    attributes = normalize_extracted_attributes(
        safe_json_object(raw_response),
        candidate_attribute_names,
    )
    if attributes == record.get("attributes"):
        return record, False
    updated = dict(record)
    updated["attributes"] = attributes
    updated["reparsed_raw_response"] = True
    return updated, True


def should_refresh_cached_extraction(record: dict[str, Any]) -> bool:
    return not clean_text(record.get("error")) and not record.get("attributes")


def estimate_model_analysis_keys(
    *,
    source_paths: Iterable[Path],
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    args: argparse.Namespace,
) -> set[str]:
    keys: set[str] = set()
    for source_table in iter_jsonl_records(source_paths):
        entity_col = choose_entity_column(
            source_table,
            min_linked_rows=configured_query_rows_per_table(args),
        )
        if entity_col is None:
            continue
        attribute_cols = candidate_attribute_columns(source_table, entity_col, args.min_column_non_empty_ratio)
        if not attribute_cols:
            continue
        candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
        for source_row in source_table.get("rows", []):
            entity_cell = get_cell(source_row, entity_col)
            wiki_title = clean_text(entity_cell.get("wiki_title"))
            entity_id = wiki_to_entity_id.get(wiki_title)
            if not wiki_title or not entity_id:
                continue
            row_attributes = extraction_row_attributes(
                source_table,
                source_row,
                entity_col,
            )
            for asset_id in entity_to_assets.get(entity_id, []):
                asset = assets.get(asset_id)
                if not asset:
                    continue
                keys.add(
                    extraction_cache_key(
                        asset_id=asset_id,
                        entity_id=entity_id,
                        candidate_attribute_names=candidate_attribute_names,
                        asset_type=str(asset.get("asset_type")),
                        args=args,
                        row_attributes=row_attributes,
                    )
                )
    return keys


def collect_table_extraction_tasks(
    *,
    source_table: dict[str, Any],
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    args: argparse.Namespace,
    asset_types: set[str] | None = None,
) -> list[ExtractionTask]:
    entity_col = choose_entity_column(
        source_table,
        min_linked_rows=configured_query_rows_per_table(args),
    )
    if entity_col is None:
        return []
    attribute_cols = candidate_attribute_columns(source_table, entity_col, args.min_column_non_empty_ratio)
    if not attribute_cols:
        return []
    candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
    tasks: list[ExtractionTask] = []
    for fallback, source_row in enumerate(source_table.get("rows", [])):
        source_row_id = row_id(source_row, fallback)
        entity_cell = get_cell(source_row, entity_col)
        wiki_title = clean_text(entity_cell.get("wiki_title"))
        entity_text = clean_text(entity_cell.get("text"))
        entity_id = wiki_to_entity_id.get(wiki_title)
        if not wiki_title or not entity_id:
            continue
        entity = {
            "entity_id": entity_id,
            "wiki_title": wiki_title,
            "cell_text": entity_text,
            "entity_column_index": entity_col,
            "entity_column_name": get_column_name(source_table, entity_col),
            "row_attributes": extraction_row_attributes(
                source_table,
                source_row,
                entity_col,
            ),
        }
        for asset_id in entity_to_assets.get(entity_id, []):
            asset = assets.get(asset_id)
            if not asset:
                continue
            asset_type = clean_text(asset.get("asset_type"))
            if asset_types is not None and asset_type not in asset_types:
                continue
            cache_key = extraction_cache_key(
                asset_id=asset["asset_id"],
                entity_id=entity["entity_id"],
                candidate_attribute_names=candidate_attribute_names,
                asset_type=asset_type,
                args=args,
                row_attributes=entity["row_attributes"],
            )
            tasks.append(
                ExtractionTask(
                    order=len(tasks),
                    cache_key=cache_key,
                    source_table_id=source_table["source_table_id"],
                    source_row_id=source_row_id,
                    entity_column_index=entity_col,
                    entity_column_name=get_column_name(source_table, entity_col),
                    entity=entity,
                    asset=asset,
                    candidate_attribute_names=candidate_attribute_names,
                )
            )
    return tasks


def collect_extraction_tasks_from_tables(
    *,
    source_paths: Iterable[Path],
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    args: argparse.Namespace,
    asset_types: set[str] | None = None,
) -> list[ExtractionTask]:
    tasks: list[ExtractionTask] = []
    seen: set[str] = set()
    for source_table in iter_jsonl_records(source_paths):
        for task in collect_table_extraction_tasks(
            source_table=source_table,
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
            asset_types=asset_types,
        ):
            if task.cache_key in seen:
                continue
            seen.add(task.cache_key)
            tasks.append(
                ExtractionTask(
                    order=len(tasks),
                    cache_key=task.cache_key,
                    source_table_id=task.source_table_id,
                    source_row_id=task.source_row_id,
                    entity_column_index=task.entity_column_index,
                    entity_column_name=task.entity_column_name,
                    entity=task.entity,
                    asset=task.asset,
                    candidate_attribute_names=task.candidate_attribute_names,
                )
            )
    return tasks


def model_jobset_policy_identity(
    args: argparse.Namespace,
    model_kind: str,
) -> dict[str, Any]:
    identity = {
        "disable_thinking": bool(getattr(args, "disable_thinking", True)),
        "model_temperature": float(
            getattr(args, "model_temperature", 0.0)
        ),
        "reparse_cached_model_outputs": bool(
            getattr(args, "reparse_cached_model_outputs", True)
        ),
        "refresh_invalid_model_cache": bool(
            getattr(args, "refresh_invalid_model_cache", False)
        ),
        "cache_failed_model_outputs": bool(
            getattr(args, "cache_failed_model_outputs", False)
        ),
        "no_reuse_model_cache": bool(
            getattr(args, "no_reuse_model_cache", False)
        ),
    }
    if model_kind == "image":
        identity.update(
            {
                "max_tokens": int(
                    getattr(
                        args,
                        "image_model_max_tokens",
                        DEFAULT_IMAGE_MODEL_MAX_TOKENS,
                    )
                ),
                "request_max_pixels": int(
                    getattr(
                        args,
                        "image_request_max_pixels",
                        DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
                    )
                ),
                "context_retry_max_pixels": int(
                    getattr(
                        args,
                        "context_retry_image_max_pixels",
                        DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS,
                    )
                ),
            }
        )
    else:
        identity["max_tokens"] = int(
            getattr(args, "model_max_tokens", 1024)
        )
    return identity


def build_model_marker_context(
    *,
    args: argparse.Namespace,
    tasks_by_kind: dict[str, list[ExtractionTask]],
    upstream_identities: Iterable[dict[str, Any]],
) -> model_markers.ModelMarkerContext:
    tasks = {
        kind: list(tasks_by_kind.get(kind, []))
        for kind in ("text", "image")
    }
    fingerprints = {
        kind: model_markers.task_jobset_fingerprint(
            tasks[kind],
            model_kind=kind,
            model_identity=str(
                getattr(
                    args,
                    f"{kind}_model_name",
                    "Qwen3.5-9B"
                    if kind == "text"
                    else "Qwen3-VL-8B-Thinking",
                )
            ),
            prompt_version=PROMPT_VERSION,
            policy_identity=model_jobset_policy_identity(args, kind),
        )
        for kind in ("text", "image")
    }
    return model_markers.build_marker_context(
        run_fingerprint=str(getattr(args, "run_fingerprint", "")),
        text_jobset_fingerprint=fingerprints["text"],
        image_jobset_fingerprint=fingerprints["image"],
        text_task_count=len(tasks["text"]),
        image_task_count=len(tasks["image"]),
        upstream_identities=upstream_identities,
    )


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _model_round_event_path(
    control_dir: Path, round_id: int, event: str
) -> Path:
    return control_dir / f"round-{round_id:06d}.{event}.json"


def _begin_model_task_round(
    args: argparse.Namespace, counts: dict[str, int]
) -> tuple[Path, int, str] | None:
    path_value = clean_text(getattr(args, "model_round_control_dir", ""))
    if not path_value:
        return None
    control_dir = Path(path_value)
    run_id = clean_text(getattr(args, "model_round_run_id", ""))
    round_id = int(getattr(args, "_model_round_sequence", 0))
    setattr(args, "_model_round_sequence", round_id + 1)
    _write_atomic_json(
        _model_round_event_path(control_dir, round_id, "start"),
        {
            "status": "model_round_start",
            "run_id": run_id,
            "round_id": round_id,
            "text_task_count": counts.get("text", 0),
            "image_task_count": counts.get("image", 0),
            "timestamp": time.time(),
        },
    )
    ready_path = _model_round_event_path(control_dir, round_id, "ready")
    while True:
        try:
            payload = json.loads(ready_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            time.sleep(0.05)
            continue
        if (
            isinstance(payload, dict)
            and payload.get("status") == "model_round_services_ready"
            and payload.get("round_id") == round_id
            and payload.get("run_id", "") == run_id
        ):
            return control_dir, round_id, run_id
        time.sleep(0.05)


def _write_model_round_event(
    model_round: tuple[Path, int, str] | None,
    event: str,
    **payload: Any,
) -> None:
    if model_round is None:
        return
    control_dir, round_id, run_id = model_round
    _write_atomic_json(
        _model_round_event_path(control_dir, round_id, event),
        {
            "round_id": round_id,
            "run_id": run_id,
            "timestamp": time.time(),
            **payload,
        },
    )


def source_shard_identities(
    source_paths: Iterable[Path],
) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for value in source_paths:
        path = Path(value).expanduser().resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        identities.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
            }
        )
    return identities


def write_model_done_marker(
    path_value: str,
    *,
    model_kind: str,
    task_count: int,
    context: model_markers.ModelMarkerContext | None = None,
) -> None:
    if not clean_text(path_value):
        return
    if context is not None:
        payload = model_markers.done_marker_payload(
            context,
            model_kind=model_kind,
            task_count=task_count,
            timestamp=time.time(),
        )
    else:
        payload = {
            "status": f"{model_kind}_model_cache_precomputed",
            "model_kind": model_kind,
            "task_count": task_count,
            f"{model_kind}_task_count": task_count,
            "timestamp": time.time(),
        }
    model_markers.atomic_write_json(
        Path(path_value),
        payload,
    )


def write_model_start_marker(
    path_value: str,
    *,
    context: model_markers.ModelMarkerContext | None = None,
    text_task_count: int = 0,
    image_task_count: int = 0,
    round_mode: bool = False,
    round_mode_requires_services: bool = False,
) -> None:
    if not clean_text(path_value):
        return
    if context is None:
        payload = {
            "status": "model_cache_ready_to_start",
            "text_task_count": text_task_count,
            "image_task_count": image_task_count,
            "round_mode": round_mode,
            "runner_startup_task_count": max(
                text_task_count + image_task_count,
                int(round_mode_requires_services),
            ),
            "timestamp": time.time(),
        }
    else:
        payload = model_markers.start_marker_payload(
            context,
            timestamp=time.time(),
        )
    model_markers.atomic_write_json(
        Path(path_value),
        payload,
    )


def model_ready_marker_matches(
    path_value: str,
    *,
    context: model_markers.ModelMarkerContext,
) -> bool:
    if not clean_text(path_value):
        return False
    return model_markers.marker_matches(
        Path(path_value),
        expected_stage=model_markers.MODEL_READY_STAGE,
        expected_status="vllm_servers_ready",
        context=context,
        model_kind=model_markers.READY_MODEL_KIND,
    )


def wait_for_model_ready_marker(
    path_value: str,
    *,
    context: model_markers.ModelMarkerContext | None = None,
    timeout_seconds: float | None = None,
    poll_seconds: float = 2.0,
) -> None:
    if not clean_text(path_value):
        return
    started = time.monotonic()
    path = Path(path_value)
    while not (
        model_ready_marker_matches(path_value, context=context)
        if context is not None
        else path.exists()
    ):
        if (
            timeout_seconds is not None
            and time.monotonic() - started > timeout_seconds
        ):
            raise RuntimeError(
                f"timed out waiting for ready marker: {path_value}"
            )
        time.sleep(poll_seconds)


def write_text_done_marker(
    path_value: str,
    *,
    task_count: int,
    context: model_markers.ModelMarkerContext | None = None,
) -> None:
    write_model_done_marker(
        path_value,
        model_kind="text",
        task_count=task_count,
        context=context,
    )


def model_done_marker_for_kind(args: argparse.Namespace, model_kind: str) -> str:
    if model_kind == "image":
        return clean_text(getattr(args, "model_image_done_marker", ""))
    return clean_text(getattr(args, "model_text_done_marker", ""))


def write_done_markers_after_selection(
    args: argparse.Namespace,
    text_task_count: int,
    image_task_count: int,
    context: model_markers.ModelMarkerContext | None = None,
) -> None:
    for model_kind, task_count in (
        ("text", text_task_count),
        ("image", image_task_count),
    ):
        marker_path = model_done_marker_for_kind(args, model_kind)
        if context is None:
            write_model_done_marker(
                marker_path,
                model_kind=model_kind,
                task_count=task_count,
            )
        else:
            write_model_done_marker(
                marker_path,
                model_kind=model_kind,
                task_count=task_count,
                context=context,
            )


class _RoundRemoteWorkload:
    """Thread-safe unfinished counts for one EntiTables model round."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._lock = threading.Lock()
        self._identity = uuid.uuid4().hex
        self._counts = {
            "text": max(0, int(counts.get("text", 0))),
            "image": max(0, int(counts.get("image", 0))),
        }

    def complete(self, modality: str) -> None:
        if modality not in self._counts:
            raise ValueError(f"unsupported model modality: {modality}")
        with self._lock:
            self._counts[modality] = 0

    def snapshot(self) -> WorkloadSnapshot:
        with self._lock:
            counts = dict(self._counts)
        return WorkloadSnapshot(
            pair_identity=self._identity,
            text_kind="entitables-text",
            image_kind="entitables-image",
            text_unfinished=counts["text"],
            image_unfinished=counts["image"],
        )


class EntiTablesRemoteGpuBorrower:
    """Borrow the WDC-owned remote layout without delaying local workers."""

    def __init__(
        self,
        *,
        controller_config: RemoteLayoutControllerConfig,
        scheduler: RoutingScheduler,
        workload: _RoundRemoteWorkload,
        coordination_dir: Path | None,
        coordination_poll_seconds: float,
        drain_timeout_seconds: float,
    ) -> None:
        self.controller_config = controller_config
        self.scheduler = scheduler
        self.workload = workload
        self.coordination_paths = (
            gpu_priority.ProtocolPaths(coordination_dir.resolve())
            if coordination_dir is not None
            else None
        )
        self.coordination_poll_seconds = coordination_poll_seconds
        self.drain_timeout_seconds = drain_timeout_seconds
        self.borrower_id = uuid.uuid4().hex
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._controller_stop_lock = threading.Lock()
        self._controller: RemoteLayoutController | None = None
        self._controller_thread: threading.Thread | None = None
        self._controller_started = False
        self._supervisor_thread: threading.Thread | None = None
        self._borrower_lock = (
            ExclusiveControllerLock(self.coordination_paths.borrower_lock)
            if self.coordination_paths is not None
            else None
        )
        self._last_acknowledged: tuple[str, int] | None = None
        self._shutdown_error: RuntimeError | None = None

    def start(self) -> None:
        self.scheduler.clear()
        if self._borrower_lock is not None:
            self._borrower_lock.acquire()
        try:
            self._supervisor_thread = threading.Thread(
                target=self._supervise,
                name="entitables-remote-gpu-borrower",
                daemon=True,
            )
            self._supervisor_thread.start()
        except BaseException:
            if self._borrower_lock is not None:
                self._borrower_lock.release()
            raise

    def complete(self, modality: str) -> None:
        self.workload.complete(modality)

    def close(self) -> None:
        self._stop.set()
        thread = self._supervisor_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(
                timeout=max(
                    1.0,
                    self.drain_timeout_seconds
                    + self.controller_config.request_timeout_seconds * 2,
                )
            )
            if thread.is_alive():
                raise RuntimeError(
                    "timed out stopping the EntiTables remote GPU borrower"
                )
        if self._shutdown_error is not None:
            raise self._shutdown_error

    def _current_request(self) -> gpu_priority.PriorityRequest | None:
        if self.coordination_paths is None:
            return None
        return gpu_priority.read_priority_request(
            self.coordination_paths.request
        )

    def _heartbeat(
        self,
        state: str,
        request: gpu_priority.PriorityRequest | None,
    ) -> None:
        if self.coordination_paths is None:
            return
        gpu_priority.write_borrower_status(
            self.coordination_paths.borrower,
            borrower_id=self.borrower_id,
            state=state,
            request=request,
        )

    def _launch_controller(self) -> None:
        with self._state_lock:
            if self._controller is not None or self._stop.is_set():
                return
            controller = RemoteLayoutController(
                self.controller_config,
                self.scheduler,
            )
            thread = threading.Thread(
                target=self._start_controller,
                args=(controller,),
                name="entitables-remote-layout-controller",
                daemon=True,
            )
            self._controller = controller
            self._controller_thread = thread
            self._controller_started = False
            thread.start()

    def _start_controller(
        self,
        controller: RemoteLayoutController,
    ) -> None:
        try:
            controller.start()
        except LayoutProtocolError as error:
            if not self._stop.is_set():
                logging.info(
                    "Remote GPUs remain with WDC; EntiTables will retry: %s",
                    error,
                )
            self.scheduler.clear()
            with self._state_lock:
                if self._controller is controller:
                    self._controller = None
                    self._controller_started = False
            return
        except BaseException:
            self.scheduler.clear()
            with self._state_lock:
                if self._controller is controller:
                    self._controller = None
                    self._controller_started = False
            if not self._stop.is_set():
                logging.exception(
                    "EntiTables remote GPU borrower failed to start"
                )
            return
        with self._state_lock:
            if self._controller is controller and not self._stop.is_set():
                self._controller_started = True

    def _stop_controller(self) -> bool:
        with self._controller_stop_lock:
            with self._state_lock:
                controller = self._controller
                thread = self._controller_thread
            if controller is not None:
                controller.request_stop()
            if thread is not None and thread is not threading.current_thread():
                deadline = time.monotonic() + max(
                    1.0,
                    self.controller_config.request_timeout_seconds * 2,
                )
                while thread.is_alive():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    thread.join(min(remaining, 0.5))
                    self._heartbeat("draining", self._current_request())
                if thread.is_alive():
                    logging.error(
                        "Remote layout controller did not stop; withholding "
                        "the WDC reclaim acknowledgement"
                    )
                    return False
            if controller is not None:
                try:
                    controller.close(
                        withdraw_routes=True,
                        drain_timeout_seconds=self.drain_timeout_seconds,
                        drain_progress=lambda: self._heartbeat(
                            "draining",
                            self._current_request(),
                        ),
                        require_lease_release=True,
                    )
                except LayoutProtocolError as error:
                    logging.error(
                        "Refusing to acknowledge the WDC remote reclaim: %s",
                        error,
                    )
                    return False
            else:
                self.scheduler.clear()
                if not self.scheduler.wait_all_drained(
                    self.drain_timeout_seconds,
                    on_wait=lambda: self._heartbeat(
                        "draining",
                        self._current_request(),
                    ),
                ):
                    logging.error(
                        "Refusing to release remote routes before all "
                        "EntiTables requests drain"
                    )
                    return False
            with self._state_lock:
                if self._controller is controller:
                    self._controller = None
                    self._controller_thread = None
                    self._controller_started = False
            return True

    def _controller_state(self) -> tuple[bool, bool]:
        with self._state_lock:
            return self._controller is not None, self._controller_started

    def _acknowledge_reclaim(
        self,
        request: gpu_priority.PriorityRequest,
    ) -> None:
        if self.coordination_paths is None:
            return
        if request.token != self._last_acknowledged:
            gpu_priority.write_acknowledgement(
                self.coordination_paths.acknowledgement,
                request,
                borrower_id=self.borrower_id,
                status=gpu_priority.RELEASED_STATUS,
            )
            self._last_acknowledged = request.token

    def _supervise(self) -> None:
        supervisor_error: Exception | None = None
        try:
            while not self._stop.is_set():
                request = self._current_request()
                priority_requested = bool(
                    request is not None
                    and request.state == gpu_priority.PRIORITY_REQUESTED_STATE
                )
                if priority_requested:
                    if self._stop_controller():
                        self._acknowledge_reclaim(request)
                        self._heartbeat(gpu_priority.RELEASED_STATUS, request)
                else:
                    has_controller, started = self._controller_state()
                    self._heartbeat(
                        gpu_priority.SERVING_STATUS
                        if started
                        else "acquiring",
                        request,
                    )
                    if not has_controller:
                        self._launch_controller()
                self._stop.wait(self.coordination_poll_seconds)
        except Exception as error:
            supervisor_error = error
            logging.exception(
                "EntiTables remote GPU borrower supervision failed"
            )
        finally:
            try:
                stopped = self._stop_controller()
            except Exception as error:
                logging.exception(
                    "EntiTables remote GPU borrower cleanup failed"
                )
                stopped = False
                cleanup_detail = f": {error}"
            else:
                cleanup_detail = ""
            if not stopped:
                message = (
                    "EntiTables remote GPU borrower stopped without releasing "
                    f"an undrained controller{cleanup_detail}"
                )
                logging.error(message)
                self._shutdown_error = RuntimeError(message)
            elif supervisor_error is not None:
                self._shutdown_error = RuntimeError(
                    "EntiTables remote GPU borrower supervision failed: "
                    f"{supervisor_error}"
                )
            if stopped and self.coordination_paths is not None:
                gpu_priority.remove_borrower_status(
                    self.coordination_paths.borrower,
                    borrower_id=self.borrower_id,
                )
            if self._borrower_lock is not None:
                self._borrower_lock.release()


def _entitables_remote_gpu_borrower(
    *,
    extractor: LocalAttributeExtractor,
    args: argparse.Namespace,
    counts: dict[str, int],
) -> EntiTablesRemoteGpuBorrower | None:
    control_url = clean_text(
        getattr(args, "remote_layout_control_url", "")
    ).rstrip("/")
    if not control_url:
        return None
    if not any(
        counts.get(modality, 0) > 0
        and getattr(extractor, f"remote_{modality}_model_workers", 0) > 0
        for modality in ("text", "image")
    ):
        return None
    scheduler = extractor.remote_routing_scheduler
    if scheduler is None:
        raise ValueError(
            "remote layout control requires a remote routing manifest"
        )
    required_paths = {
        "token": clean_text(
            getattr(args, "remote_layout_control_token_file", "")
        ),
        "controller_id": clean_text(
            getattr(args, "remote_layout_controller_id_file", "")
        ),
        "lock": clean_text(
            getattr(args, "remote_layout_lock_file", "")
        ),
    }
    if not all(required_paths.values()):
        raise ValueError("remote layout control paths are incomplete")
    primary_url = clean_text(
        getattr(args, "remote_layout_primary_image_url", "")
    ).rstrip("/")
    switchable_url = clean_text(
        getattr(args, "remote_layout_switchable_url", "")
    ).rstrip("/")
    if not primary_url or not switchable_url:
        raise ValueError("remote layout endpoint mappings are incomplete")
    workload = _RoundRemoteWorkload(counts)
    drain_timeout = float(args.remote_layout_drain_timeout_seconds)
    controller_config = RemoteLayoutControllerConfig(
        control_url=control_url,
        token_file=Path(required_paths["token"]).resolve(),
        database_path=None,
        controller_id_file=Path(required_paths["controller_id"]).resolve(),
        lock_file=Path(required_paths["lock"]).resolve(),
        endpoints={
            "primary_image": RemoteLayoutEndpointConfig(
                endpoint_id="primary_image",
                base_url=primary_url,
            ),
            "switchable": RemoteLayoutEndpointConfig(
                endpoint_id="switchable",
                base_url=switchable_url,
            ),
        },
        text_model_id=extractor.text_model_name,
        image_model_id=extractor.image_model_name,
        text_api_key=extractor.remote_text_model_api_key,
        image_api_key=extractor.remote_image_model_api_key,
        lease_ttl_seconds=int(args.remote_layout_lease_ttl_seconds),
        lease_renew_seconds=float(args.remote_layout_lease_renew_seconds),
        request_timeout_seconds=float(
            args.remote_layout_request_timeout_seconds
        ),
        reconnect_timeout_seconds=float(
            args.remote_layout_reconnect_timeout_seconds
        ),
        operation_timeout_seconds=float(
            args.remote_layout_operation_timeout_seconds
        ),
        drain_timeout_seconds=drain_timeout,
        poll_seconds=float(args.remote_layout_poll_seconds),
        workload_poll_seconds=float(
            args.remote_layout_workload_poll_seconds
        ),
        image_burst_stability_seconds=float(
            args.remote_layout_stability_seconds
        ),
        endpoint_health_timeout_seconds=float(
            args.remote_layout_health_timeout_seconds
        ),
        endpoint_health_stable_polls=int(
            args.remote_layout_health_stable_polls
        ),
        workload_reader=workload.snapshot,
    )
    coordination_value = clean_text(
        getattr(args, "remote_layout_coordination_dir", "")
    )
    if not coordination_value:
        raise ValueError(
            "EntiTables remote borrowing requires the shared WDC "
            "coordination directory"
        )
    return EntiTablesRemoteGpuBorrower(
        controller_config=controller_config,
        scheduler=scheduler,
        workload=workload,
        coordination_dir=Path(coordination_value),
        coordination_poll_seconds=float(
            args.remote_layout_coordination_poll_seconds
        ),
        drain_timeout_seconds=drain_timeout,
    )


def precompute_extraction_task_groups(
    *,
    extractor: LocalAttributeExtractor,
    cache: ExtractionCache,
    tasks_by_kind: dict[str, list[ExtractionTask]],
    args: argparse.Namespace,
    state: ModelConcurrencyState,
    progress: ModelAnalysisProgress | None = None,
    write_done_markers: bool = True,
    marker_context: model_markers.ModelMarkerContext | None = None,
) -> dict[str, int]:
    active_groups = {
        kind: tasks
        for kind, tasks in tasks_by_kind.items()
        if kind in {"text", "image"} and tasks
    }
    counts = {kind: len(tasks) for kind, tasks in tasks_by_kind.items() if kind in {"text", "image"}}
    for kind, count in counts.items():
        if write_done_markers and kind not in active_groups:
            write_model_done_marker(
                model_done_marker_for_kind(args, kind),
                model_kind=kind,
                task_count=count,
                context=marker_context,
            )
    if not active_groups:
        return counts

    model_round = _begin_model_task_round(args, counts)
    remote_borrower = _entitables_remote_gpu_borrower(
        extractor=extractor,
        args=args,
        counts=counts,
    )
    round_status = "completed"
    remote_borrower_started = False
    try:
        if remote_borrower is not None:
            remote_borrower.start()
            remote_borrower_started = True
        with ThreadPoolExecutor(max_workers=len(active_groups)) as pool:
            futures = {
                pool.submit(
                    resolve_extraction_tasks,
                    extractor=extractor,
                    cache=cache,
                    tasks=tasks,
                    args=args,
                    state=state,
                    progress=progress,
                ): (kind, len(tasks))
                for kind, tasks in active_groups.items()
            }
            for future in as_completed(futures):
                kind, task_count = futures[future]
                future.result()
                if remote_borrower is not None:
                    remote_borrower.complete(kind)
                _write_model_round_event(
                    model_round,
                    f"{kind}.done",
                    status=f"{kind}_round_tasks_completed",
                    model_kind=kind,
                    task_count=task_count,
                )
                if write_done_markers:
                    write_model_done_marker(
                        model_done_marker_for_kind(args, kind),
                        model_kind=kind,
                        task_count=task_count,
                        context=marker_context,
                    )
    except BaseException:
        round_status = "failed"
        raise
    finally:
        try:
            if remote_borrower is not None and remote_borrower_started:
                remote_borrower.close()
        except BaseException:
            round_status = "failed"
            raise
        finally:
            _write_model_round_event(
                model_round, "done", status=f"model_round_{round_status}"
            )
    return counts


def extract_asset_attributes(
    *,
    extractor: LocalAttributeExtractor,
    cache: ExtractionCache,
    asset: dict[str, Any],
    entity: dict[str, Any],
    candidate_attribute_names: list[str],
    args: argparse.Namespace,
    progress: ModelAnalysisProgress | None = None,
) -> dict[str, Any]:
    cache_key = extraction_cache_key(
        asset_id=asset["asset_id"],
        entity_id=entity["entity_id"],
        candidate_attribute_names=candidate_attribute_names,
        asset_type=str(asset.get("asset_type")),
        args=args,
        row_attributes=entity.get("row_attributes"),
    )
    transient = cache_get_transient(cache, cache_key)
    if transient is not None:
        if progress is not None:
            progress.mark(
                cache_key,
                "error" if clean_text(transient.get("error")) else "model",
            )
        return transient
    cached = cache.get(cache_key)
    if cached:
        cached_record = cached
        if getattr(args, "reparse_cached_model_outputs", True):
            cached_record, changed = reparse_extraction_record(
                cached_record,
                candidate_attribute_names,
            )
            if changed:
                cache.put(cache_key, cached_record)
        if cached_extraction_is_reusable(cached_record, args):
            if progress is not None:
                progress.mark(cache_key, "cached")
            return cached_record
    try:
        result = extractor.extract(asset, entity, candidate_attribute_names)
    except TransientModelEndpointError:
        if getattr(extractor, "abort_on_transient_error", False):
            raise
        result = {
            "attributes": [],
            "raw_response": "",
            "error": "model endpoint temporarily unavailable",
            "error_class": "model_endpoint_transient",
        }
    except Exception as exc:
        result = {"attributes": [], "raw_response": "", "error": str(exc)}
    record = {
        "cache_key": cache_key,
        "prompt_version": PROMPT_VERSION,
        "entity_id": entity["entity_id"],
        "entity_text": entity["cell_text"],
        "entity_wiki_title": entity["wiki_title"],
        "row_attributes": canonical_extraction_row_attributes(
            entity.get("row_attributes")
        ),
        "asset_id": asset["asset_id"],
        "asset_type": asset.get("asset_type"),
        "candidate_attribute_names": candidate_attribute_names,
        "attributes": result.get("attributes", []),
        "raw_response": result.get("raw_response", ""),
        "error": result.get("error", ""),
    }
    if not record["error"] or getattr(args, "cache_failed_model_outputs", False):
        cache.put(cache_key, record)
    else:
        cache_put_transient(cache, cache_key, record)
    if progress is not None:
        progress.mark(cache_key, "error" if record["error"] else "model")
    return record


def asset_preview(asset: dict[str, Any]) -> dict[str, Any]:
    if asset.get("asset_type") == "text":
        return {
            "content_snippet": clean_text(asset.get("content"))[:1600],
        }
    return {}


def build_table_join_records(
    *,
    source_table: dict[str, Any],
    split: str,
    assets: dict[str, dict[str, Any]],
    entity_to_assets: dict[str, list[str]],
    wiki_to_entity_id: dict[str, str],
    extractor: LocalAttributeExtractor | None,
    cache: ExtractionCache,
    progress: ModelAnalysisProgress | None,
    concurrency_state: ModelConcurrencyState,
    extraction_writer: ShardedJsonlWriter,
    recovery_writer: ShardedJsonlWriter,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_rows_per_table = configured_query_rows_per_table(args)
    entity_col = choose_entity_column(
        source_table,
        min_linked_rows=query_rows_per_table,
    )
    if entity_col is None:
        return rejected_table_join_records(
            source_table=source_table,
            split=split,
            entity_col=None,
            decision={"reason": "no_entity_column", "qualified_columns": []},
            args=args,
        )

    attribute_cols = candidate_attribute_columns(source_table, entity_col, args.min_column_non_empty_ratio)
    if not attribute_cols:
        return rejected_table_join_records(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            decision={
                "reason": "no_candidate_attribute_columns",
                "entity_column_index": entity_col,
                "qualified_columns": [],
            },
            args=args,
        )

    candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
    valid_entity_source_rows: set[int] = set()
    valid_entity_source_row_order: list[int] = []
    recovered_rows_by_col: dict[int, set[int]] = defaultdict(set)
    recoveries_by_col: dict[int, list[dict[str, Any]]] = defaultdict(list)
    extraction_count = 0
    extraction_tasks: list[ExtractionTask] = []
    source_rows_by_id = {
        row_id(source_row, fallback): source_row
        for fallback, source_row in enumerate(source_table.get("rows", []))
    }

    for fallback, source_row in enumerate(source_table.get("rows", [])):
        source_row_id = row_id(source_row, fallback)
        entity_cell = get_cell(source_row, entity_col)
        wiki_title = clean_text(entity_cell.get("wiki_title"))
        entity_text = clean_text(entity_cell.get("text"))
        entity_id = wiki_to_entity_id.get(wiki_title)
        if not wiki_title or not entity_id:
            continue
        valid_entity_source_rows.add(source_row_id)
        valid_entity_source_row_order.append(source_row_id)
        asset_ids = entity_to_assets.get(entity_id, [])
        if not asset_ids:
            continue
        entity = {
            "entity_id": entity_id,
            "wiki_title": wiki_title,
            "cell_text": entity_text,
            "entity_column_index": entity_col,
            "entity_column_name": get_column_name(source_table, entity_col),
            "row_attributes": extraction_row_attributes(
                source_table,
                source_row,
                entity_col,
            ),
        }
        for asset_id in asset_ids:
            asset = assets.get(asset_id)
            if not asset:
                continue
            cache_key = extraction_cache_key(
                asset_id=asset["asset_id"],
                entity_id=entity["entity_id"],
                candidate_attribute_names=candidate_attribute_names,
                asset_type=str(asset.get("asset_type")),
                args=args,
                row_attributes=entity["row_attributes"],
            )
            extraction_tasks.append(
                ExtractionTask(
                    order=len(extraction_tasks),
                    cache_key=cache_key,
                    source_table_id=source_table["source_table_id"],
                    source_row_id=source_row_id,
                    entity_column_index=entity_col,
                    entity_column_name=get_column_name(source_table, entity_col),
                    entity=entity,
                    asset=asset,
                    candidate_attribute_names=candidate_attribute_names,
                )
            )

    for task, extraction in resolve_extraction_tasks(
        extractor=extractor,
        cache=cache,
        tasks=extraction_tasks,
        args=args,
        state=concurrency_state,
        progress=progress,
    ):
        source_row_id = task.source_row_id
        entity = task.entity
        asset = task.asset
        write_jsonl_record(
            extraction_writer,
            {
                **extraction,
                "source_table_id": task.source_table_id,
                "source_row_id": source_row_id,
                "entity_column_index": task.entity_column_index,
                "entity_column_name": task.entity_column_name,
            },
        )
        extraction_count += 1
        attr_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in extraction.get("attributes", []):
            attr_by_name[normalize(item.get("name"))].append(item)
        source_row = source_rows_by_id.get(source_row_id)
        if source_row is None:
            continue
        for attr_col in attribute_cols:
            attr_name = get_column_name(source_table, attr_col)
            expected = clean_text(get_cell_text(source_row, attr_col))
            if not expected:
                continue
            for predicted in attr_by_name.get(normalize(attr_name), []):
                if not values_match(
                    predicted.get("value"),
                    expected,
                    attribute_name=attr_name,
                ):
                    continue
                recovered_rows_by_col[attr_col].add(source_row_id)
                recoveries_by_col[attr_col].append(
                    {
                        "source_table_id": source_table["source_table_id"],
                        "source_row_id": source_row_id,
                        "split": split,
                        "query_entity": entity,
                        "recovered_attribute": {
                            "column_index": attr_col,
                            "column_name": attr_name,
                            "value": expected,
                            "model_value": clean_text(predicted.get("value")),
                            "hidden_in_query": True,
                        },
                        "evidence": {
                            "asset_id": asset["asset_id"],
                            "asset_type": asset.get("asset_type"),
                            **asset_preview(asset),
                            "model_evidence": clean_text(predicted.get("evidence")),
                            "model_connection_evidence": clean_text(predicted.get("connection_evidence")),
                            "extraction_cache_key": extraction.get("cache_key"),
                        },
                    }
                )
                break

    qualified_cols: list[dict[str, Any]] = []
    for attr_col in attribute_cols:
        recovered = recovered_rows_by_col.get(attr_col, set())
        profile = recovery_column_profile(
            valid_source_rows=valid_entity_source_rows,
            recovered_source_rows=recovered,
            query_rows_per_table=query_rows_per_table,
            min_recovery_denominator=args.min_recovery_denominator,
            min_ratio=args.min_recovered_value_ratio,
        )
        if profile is not None:
            qualified_cols.append(
                {
                    "column_index": attr_col,
                    "column_name": get_column_name(source_table, attr_col),
                    **profile,
                }
            )

    if not qualified_cols:
        return rejected_table_join_records(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            decision={
                "reason": "no_column_met_recovered_value_ratio",
                "entity_column_index": entity_col,
                "candidate_attribute_columns": candidate_attribute_names,
                "attribute_extractions": extraction_count,
                "qualified_columns": [],
            },
            args=args,
        )

    variant_layouts = multi_attribute_context_layout(
        source_table=source_table,
        entity_col=entity_col,
        qualified_cols=qualified_cols,
        args=args,
    )

    query_tables: list[dict[str, Any]] = []
    query_by_fingerprint: dict[str, dict[str, Any]] = {}
    data_lake_tables: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    emitted_qualified_cols: list[dict[str, Any]] = []
    max_query_row_views = (
        configured_max_train_query_row_views_per_join(args)
        if split == "train"
        else 1
    )
    for qualified, query_context, target_context in variant_layouts:
        join_col = int(qualified["column_index"])
        selected_source_row_views = select_query_source_row_views(
            source_row_order=valid_entity_source_row_order,
            recovered_source_rows=recovered_rows_by_col.get(join_col, set()),
            query_rows_per_table=query_rows_per_table,
            required_recovered_rows=int(qualified["required_recovered_rows"]),
            max_views=max_query_row_views,
        )
        if not selected_source_row_views:
            continue
        query_cols = [entity_col] + query_context
        target_cols = [join_col] + target_context
        if len(query_cols) < 2 or not target_cols:
            continue
        all_source_row_ids = {
            row_id(source_row, fallback)
            for fallback, source_row in enumerate(source_table.get("rows", []))
        }
        target_rows, target_source_rows = project_selected_rows(
            source_table,
            target_cols,
            all_source_row_ids,
            min_required_cols=0,
        )
        if len(target_rows) < args.min_rows_per_output_table:
            continue
        chain_id = f"chain_{stable_hash(source_table['source_table_id'], entity_col, join_col)}"
        target_table_id = f"target_{stable_hash(chain_id, 'target')}"
        emitted_view_count = 0
        emitted_qualified = {
            **qualified,
            "selected_rows": query_rows_per_table,
            "target_rows": len(target_rows),
            "row_views": 0,
        }
        hidden_attribute = {
            "source_column_index": join_col,
            "column_name": qualified["column_name"],
            "role": "model_recoverable_join_column",
            "eligible_rows": qualified["eligible_rows"],
            "valid_entity_rows": qualified["valid_entity_rows"],
            "recovered_rows": qualified["recovered_rows"],
            "required_recovered_rows": qualified["required_recovered_rows"],
            "recovered_value_ratio": qualified["recovered_value_ratio"],
            "selected_rows": query_rows_per_table,
            "target_rows": len(target_rows),
        }
        for row_view_index, selected_source_rows in enumerate(
            selected_source_row_views
        ):
            selected_source_row_set = set(selected_source_rows)
            query_rows, query_source_rows = project_selected_rows(
                source_table,
                query_cols,
                selected_source_row_set,
                min_required_cols=1,
            )
            if not set(query_source_rows).issubset(target_source_rows):
                continue
            if len(query_rows) != query_rows_per_table:
                continue
            query_fingerprint = visible_query_fingerprint(
                source_table=source_table,
                query_cols=query_cols,
                query_rows=query_rows,
            )
            query_table = query_by_fingerprint.get(query_fingerprint)
            if query_table is None:
                query_table_id = (
                    f"query_{stable_hash(source_table['source_table_id'], query_fingerprint)}"
                )
                query_table = table_record(
                    table_id=query_table_id,
                    role="query",
                    split=split,
                    source_table=source_table,
                    column_indices=query_cols,
                    rows=query_rows,
                    source_row_indices=query_source_rows,
                    extra={
                        "chain_id": chain_id,
                        "chain_ids": [chain_id],
                        "query_entity_col": entity_col,
                        "query_entity_col_name": get_column_name(
                            source_table, entity_col
                        ),
                        "hidden_attributes": [hidden_attribute],
                        "target_table_ids": [target_table_id],
                        "query_context_col_names": [
                            get_column_name(source_table, col)
                            for col in query_context
                        ],
                        "row_view_index": row_view_index,
                    },
                )
                query_by_fingerprint[query_fingerprint] = query_table
                query_tables.append(query_table)
            else:
                query_table_id = str(query_table["table_id"])
                if target_table_id in query_table["target_table_ids"]:
                    continue
                query_table["chain_ids"].append(chain_id)
                query_table["hidden_attributes"].append(hidden_attribute)
                query_table["target_table_ids"].append(target_table_id)

            emitted_view_count += 1
            qrels.append(
                {
                    "query_table_id": query_table_id,
                    "target_table_id": target_table_id,
                    "data_lake_table_id": target_table_id,
                    "rel": 3,
                    "split": split,
                    "chain_id": chain_id,
                    "row_view_index": row_view_index,
                    "source_table_id": source_table["source_table_id"],
                    "join_attribute": hidden_attribute,
                    "reason": "model_recoverable_join_column",
                }
            )
            source_to_query_row = {
                row["source_row_id"]: row["row_id"] for row in query_rows
            }
            source_to_target_rows: dict[int, list[int]] = defaultdict(list)
            for row in target_rows:
                source_to_target_rows[int(row["source_row_id"])].append(
                    int(row["row_id"])
                )
            seen_recoveries: set[str] = set()
            for recovery in recoveries_by_col.get(join_col, []):
                source_row_id = int(recovery["source_row_id"])
                if source_row_id not in source_to_query_row:
                    continue
                recovery_id = f"evrec_{stable_hash(query_table_id, target_table_id, source_row_id, recovery['evidence']['asset_id'], recovery['recovered_attribute']['value'])}"
                if recovery_id in seen_recoveries:
                    continue
                seen_recoveries.add(recovery_id)
                path_id = f"path_{stable_hash(query_table_id, recovery['evidence']['asset_id'], target_table_id, source_row_id)}"
                write_jsonl_record(
                    recovery_writer,
                    {
                        "recovery_id": recovery_id,
                        "path_id": path_id,
                        "query_table_id": query_table_id,
                        "target_table_id": target_table_id,
                        "data_lake_table_id": target_table_id,
                        "query_row_id": source_to_query_row[source_row_id],
                        "target_row_ids": source_to_target_rows.get(
                            source_row_id, []
                        ),
                        "path_nodes": [
                            {
                                "node_id": query_table_id,
                                "node_type": "query_table",
                            },
                            {
                                "node_id": recovery["evidence"]["asset_id"],
                                "node_type": f"{recovery['evidence']['asset_type']}_asset",
                            },
                            {
                                "node_id": target_table_id,
                                "node_type": "target_table",
                            },
                        ],
                        **recovery,
                    },
                )

        if emitted_view_count == 0:
            continue
        emitted_qualified["row_views"] = emitted_view_count
        emitted_qualified_cols.append(emitted_qualified)
        data_lake_tables.append(
            table_record(
                table_id=target_table_id,
                role="target_data_lake_table",
                split=split,
                source_table=source_table,
                column_indices=target_cols,
                rows=target_rows,
                source_row_indices=target_source_rows,
                extra={
                    "chain_id": chain_id,
                    "queryable_source_table": True,
                    "join_col": join_col,
                    "join_col_name": qualified["column_name"],
                    "target_context_col_names": [
                        get_column_name(source_table, col)
                        for col in target_context
                    ],
                },
            )
        )

    if not query_tables:
        return rejected_table_join_records(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            decision={
                "reason": "qualified_columns_failed_query_target_split",
                "entity_column_index": entity_col,
                "qualified_columns": [
                    qualified
                    for qualified, _query_context, _target_context in variant_layouts
                ],
            },
            args=args,
        )
    return query_tables, data_lake_tables, qrels, {
        "reason": "queryable",
        "entity_column_index": entity_col,
        "attribute_extractions": extraction_count,
        "qualified_columns": emitted_qualified_cols,
    }


def load_assets(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for asset in iter_jsonl_records(paths):
        if asset.get("asset_id"):
            assets[asset["asset_id"]] = asset
    return assets


def build_bridge_assets_for_entity(
    *,
    entity: dict[str, Any],
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    wikipedia_client: WikipediaClient,
    imageinfo_keys_accessed: set[str] | None = None,
) -> list[dict[str, Any]]:
    page = wikipedia_client.get_page(entity["wiki_title"])
    if not page or page.get("missing"):
        return []

    records: list[dict[str, Any]] = []
    text_chunks = split_text_asset_content(
        page.get("extract"),
        max_chars=text_asset_chunk_chars,
        min_chars=min_text_asset_chunk_chars,
        max_chunks=0,
    )
    selected_text_chunks = select_relevant_text_chunks(
        text_chunks,
        entity,
        max_text_asset_chunks_per_entity,
    )
    source_asset_id = f"asset_text_{stable_hash(entity['entity_id'], 'extract')}"
    for chunk_index, chunk, chunk_score in selected_text_chunks:
        asset_id = f"{source_asset_id}_{chunk_index:03d}"
        records.append(
            {
                "asset_id": asset_id,
                "source_asset_id": source_asset_id,
                "entity_id": entity["entity_id"],
                "entity_wiki_title": entity["wiki_title"],
                "asset_type": "text",
                "content": chunk,
                "text_chunk_index": chunk_index,
                "text_chunk_count": len(text_chunks),
                "selected_text_chunk_count": len(selected_text_chunks),
                "text_chunk_relevance_score": round(chunk_score, 6),
                "source": "wikipedia_extract_chunk",
                "url": page.get("canonicalurl")
                or f"https://en.wikipedia.org/wiki/{quote(entity['wiki_title'].replace(' ', '_'))}",
            }
        )

    image_titles: list[str] = []
    if page.get("pageimage"):
        image_titles.append(f"File:{page['pageimage']}")
    for image in page.get("images") or []:
        title = image.get("title") if isinstance(image, dict) else None
        if title:
            image_titles.append(title)

    seen_images: set[str] = set()
    kept = 0
    for image_title in image_titles:
        normalized_title = normalize_title(image_title)
        if normalized_title in seen_images or not is_useful_image(normalized_title):
            continue
        seen_images.add(normalized_title)
        if imageinfo_keys_accessed is not None:
            imageinfo_keys_accessed.add(normalized_title)
        imageinfo = wikipedia_client.get_imageinfo(normalized_title)
        if not imageinfo or not imageinfo.get("url"):
            continue
        if not is_useful_image(normalized_title, imageinfo):
            continue
        asset_id = f"asset_img_{stable_hash(entity['entity_id'], normalized_title)}"
        downloaded_image = wikipedia_client.download_image(imageinfo, asset_id)
        if downloaded_image is None:
            continue
        records.append(
            {
                "asset_id": asset_id,
                "entity_id": entity["entity_id"],
                "entity_wiki_title": entity["wiki_title"],
                "asset_type": "image",
                "image_url": imageinfo.get("url"),
                "description_url": imageinfo.get("descriptionurl"),
                "local_path": downloaded_image["local_path"],
                "relative_path": downloaded_image["relative_path"],
                "file_name": downloaded_image["file_name"],
                "bytes": downloaded_image["bytes"],
                "sha256": downloaded_image["sha256"],
                "metadata": {
                    "file_title": imageinfo.get("file_title"),
                    "mime": imageinfo.get("mime"),
                    "mediatype": imageinfo.get("mediatype"),
                    "width": imageinfo.get("width"),
                    "height": imageinfo.get("height"),
                    "size": imageinfo.get("size"),
                    "extmetadata": imageinfo.get("extmetadata") or {},
                    "downloaded": downloaded_image["downloaded"],
                },
                "source": "wikipedia_image_download",
            }
        )
        kept += 1
        if kept >= max_images_per_entity:
            break
    return records


def build_bridge_assets_parallel(
    *,
    entities: list[dict[str, Any]],
    max_entities: int | None,
    max_images_per_entity: int,
    text_asset_chunk_chars: int,
    min_text_asset_chunk_chars: int,
    max_text_asset_chunks_per_entity: int,
    wikipedia_client_factory: Callable[[], WikipediaClient] | None,
    asset_writer: ShardedJsonlWriter,
    flush_every_records: int,
    workers: int,
) -> tuple[dict[str, list[str]], int, int, int]:
    if wikipedia_client_factory is None:
        return defaultdict(list), 0, 0, 0
    selected_entities = entities[:max_entities] if max_entities else entities
    workers = max(1, int(workers or 1))
    if workers <= 1:
        return build_bridge_assets(
            entities,
            max_entities,
            max_images_per_entity,
            text_asset_chunk_chars,
            min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity,
            wikipedia_client_factory(),
            asset_writer,
            flush_every_records,
        )

    entity_to_assets: dict[str, list[str]] = defaultdict(list)
    text_asset_count = 0
    image_asset_count = 0
    written_assets = 0
    worker_state = threading.local()
    clients: list[WikipediaClient] = []
    clients_lock = threading.Lock()
    pending_results: dict[int, list[dict[str, Any]]] = {}
    next_write_index = 0

    def worker_client() -> WikipediaClient:
        client = getattr(worker_state, "client", None)
        if client is None:
            client = wikipedia_client_factory()
            worker_state.client = client
            with clients_lock:
                clients.append(client)
        return client

    def fetch_entity(entity: dict[str, Any]) -> list[dict[str, Any]]:
        return build_bridge_assets_for_entity(
            entity=entity,
            max_images_per_entity=max_images_per_entity,
            text_asset_chunk_chars=text_asset_chunk_chars,
            min_text_asset_chunk_chars=min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity=max_text_asset_chunks_per_entity,
            wikipedia_client=worker_client(),
        )

    def write_ready_results() -> None:
        nonlocal next_write_index, written_assets, text_asset_count, image_asset_count
        while next_write_index in pending_results:
            records = pending_results.pop(next_write_index)
            for record in records:
                write_jsonl_record(asset_writer, record)
                entity_to_assets[record["entity_id"]].append(record["asset_id"])
                if record.get("asset_type") == "image":
                    image_asset_count += 1
                else:
                    text_asset_count += 1
                written_assets += 1
                if flush_every_records > 0 and written_assets % flush_every_records == 0:
                    asset_writer.flush()
            next_write_index += 1

    futures = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(selected_entities)))) as pool:
        for idx, entity in enumerate(selected_entities):
            futures[pool.submit(fetch_entity, entity)] = idx
        iterator: Iterable[Any] = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Fetching Wikipedia assets", unit="entity", dynamic_ncols=True)
        for future in iterator:
            idx = futures[future]
            try:
                pending_results[idx] = future.result()
            except Exception as exc:  # pragma: no cover - integration only.
                logging.warning("Wikipedia asset fetch failed for %s: %s", selected_entities[idx].get("wiki_title"), exc)
                pending_results[idx] = []
            write_ready_results()
    write_ready_results()
    asset_writer.flush()

    return entity_to_assets, sum(int(getattr(client, "api_failures", 0)) for client in clients), text_asset_count, image_asset_count


def _build_dataset(
    args: argparse.Namespace,
    owned_progress: list[ModelAnalysisProgress],
    injected_extractor: LocalAttributeExtractor | None = None,
) -> dict[str, Any]:
    args.query_rows_per_table = configured_query_rows_per_table(args)
    args.max_train_query_row_views_per_join = (
        configured_max_train_query_row_views_per_join(args)
    )
    args.explicit_join_fallback_mode = (
        configured_explicit_join_fallback_mode(args)
    )
    args.explicit_join_fallback_ratio = configured_explicit_join_fallback_ratio(
        args
    )
    policy = replacement_policy_from_args(args)
    if args.max_source_tables is not None and args.max_source_tables < 0:
        raise ValueError("max source tables must be non-negative or None")
    media_config = media_policy_config_from_args(args)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    media_failure_recorder = MediaFailureRecorder(
        output_dir / "media_download_failures.jsonl"
    )
    media_failure_recorder.reset()
    cache_paths = resolve_shared_cache_paths(args)
    model_attribute_errors_path = clean_text(getattr(args, "model_attribute_errors_path", ""))
    if not model_attribute_errors_path:
        model_attribute_errors_path = str(output_dir / "model_attribute_errors.jsonl")
        setattr(args, "model_attribute_errors_path", model_attribute_errors_path)
    error_log_path = Path(model_attribute_errors_path)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path.write_text("", encoding="utf-8")

    records_per_shard = max(1, args.records_per_shard)
    flush_every = max(1, args.flush_every_records)
    source_tables_dir = output_dir / "source_tables"
    query_tables_dir = output_dir / "query_tables"
    data_lake_tables_dir = output_dir / "data_lake_tables"
    entities_dir = output_dir / "entities"
    bridge_assets_dir = output_dir / "bridge_assets"
    table_asset_links_dir = output_dir / "table_asset_links"
    extraction_dir = output_dir / "attribute_extractions"
    recovery_dir = output_dir / "evidence_recoveries"

    wikipedia_client: WikipediaClient | None = None
    if not args.no_wikipedia:
        wikipedia_client = WikipediaClient(
            cache_dir=cache_paths["wikipedia_cache_dir"],
            image_output_dir=cache_paths["wikipedia_image_dir"],
            output_dir=output_dir,
            sleep=args.sleep,
            user_agent=args.wikipedia_user_agent,
            media_config=media_config,
            media_failure_recorder=media_failure_recorder,
        )
    cache = ExtractionCache(
        cache_paths["model_attribute_extractions"],
        reuse=not args.no_reuse_model_cache,
    )
    concurrency_state = ModelConcurrencyState.from_args(args)
    progress: ModelAnalysisProgress | None = None
    counters = SourceCandidateCounters()
    candidate_tables: Iterator[dict[str, Any]] = iter_random_source_tables(
        input_dir, args, counters
    )
    if args.max_source_tables is None:
        all_candidates = list(candidate_tables)
        candidate_tables = iter(all_candidates)
        target_count = len(all_candidates)
    else:
        target_count = args.max_source_tables
    candidate_entity_records: dict[str, dict[str, Any]] = {}
    candidate_wiki_to_entity_id: dict[str, str] = {}
    assets: dict[str, dict[str, Any]] = {}
    entity_to_assets: dict[str, list[str]] = {}
    registry = CandidateMaterialRegistry(
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia_client,
        extraction_cache=cache,
    )
    extractor = injected_extractor
    evaluation_context = CandidateEvaluationContext(
        entity_records=candidate_entity_records,
        wiki_to_entity_id=candidate_wiki_to_entity_id,
        assets=assets,
        entity_to_assets=entity_to_assets,
        wikipedia_client=wikipedia_client,
        extractor=extractor,
        cache=cache,
        progress=progress,
        concurrency_state=concurrency_state,
        registry=registry,
        max_entities=args.max_entities,
    )

    def start_models_after_initial_preparation(
        initial_batch: list[dict[str, Any]],
    ) -> None:
        nonlocal extractor, progress
        if getattr(args, "model_progress", True):
            progress = ModelAnalysisProgress(
                total=0,
                cached_keys=set(),
                enabled=True,
            )
            owned_progress.append(progress)
            evaluation_context.progress = progress
        write_model_start_marker(
            clean_text(getattr(args, "model_start_marker", "")),
            text_task_count=0,
            image_task_count=0,
            round_mode=True,
            round_mode_requires_services=bool(initial_batch),
        )
        wait_for_model_ready_marker(
            clean_text(getattr(args, "model_ready_marker", ""))
        )
        if extractor is None:
            extractor = LocalAttributeExtractor(args)
        evaluation_context.extractor = extractor

    cleanup_totals = CacheCleanupStats()
    selection_rng = random.Random(int(stable_hash("replacement", args.seed), 16))
    try:
        try:
            selection = run_replacement_rounds(
                candidate_tables=candidate_tables,
                target_count=target_count,
                policy=policy,
                rng=selection_rng,
                prepare_batch=lambda batch: prepare_candidate_batch(
                    batch, evaluation_context, args
                ),
                evaluate_batch=lambda batch: evaluate_candidate_batch(
                    batch, evaluation_context, args
                ),
                discard_tables=lambda table_ids: cleanup_totals.add(
                    evaluation_context.registry.discard_many(table_ids)
                ),
                on_initial_batch_prepared=start_models_after_initial_preparation,
            )
        finally:
            _close_iterator(candidate_tables)
        if progress is None and getattr(args, "model_progress", True):
            progress = ModelAnalysisProgress(
                total=0,
                cached_keys=set(),
                enabled=True,
            )
            owned_progress.append(progress)
            evaluation_context.progress = progress
        final_source_tables = [
            item.source_table for item in selection.final_evaluations
        ]
        final_table_ids = {
            str(source_table["source_table_id"])
            for source_table in final_source_tables
        }
        cleanup_totals.add(registry.sweep(final_table_ids))
    except BaseException:
        if progress is not None:
            progress.close()
        raise

    entity_records: dict[str, dict[str, Any]] = {}
    wiki_to_entity_id: dict[str, str] = {}
    for source_table in final_source_tables:
        update_entities_from_table(entity_records, wiki_to_entity_id, source_table)
    entities = finalize_entities(entity_records)

    source_split_records = [
        {
            "source_table_id": str(source_table["source_table_id"]),
            "page_title": source_table.get("page_title") or "",
        }
        for source_table in final_source_tables
    ]
    source_table_count = len(final_source_tables)
    source_writer = ShardedJsonlWriter(source_tables_dir, records_per_shard)
    with source_writer as source_handle:
        for source_table in final_source_tables:
            write_jsonl_record(source_handle, source_table)
            if source_writer.total_records % flush_every == 0:
                source_handle.flush()
        source_handle.flush()

    entities_writer = write_sharded_jsonl(entities_dir, entities, records_per_shard)
    bridge_assets_writer = ShardedJsonlWriter(bridge_assets_dir, records_per_shard)
    with bridge_assets_writer as bridge_assets_handle:
        for asset_id in sorted(assets):
            write_jsonl_record(bridge_assets_handle, assets[asset_id])
            if bridge_assets_writer.total_records % flush_every == 0:
                bridge_assets_handle.flush()
    text_asset_count = sum(
        asset.get("asset_type") == "text" for asset in assets.values()
    )
    image_asset_count = sum(
        asset.get("asset_type") == "image" for asset in assets.values()
    )
    api_failures = int(getattr(wikipedia_client, "api_failures", 0))
    wikimedia_media = (
        wikipedia_client.media_summary()
        if wikipedia_client is not None and hasattr(wikipedia_client, "media_summary")
        else {}
    )

    table_asset_links_writer = ShardedJsonlWriter(table_asset_links_dir, records_per_shard)
    with table_asset_links_writer:
        table_asset_link_count = write_table_asset_links_from_jsonl(
            source_writer.paths(),
            [],
            table_asset_links_writer,
            wiki_to_entity_id,
            entity_to_assets,
            flush_every,
        )

    splits = source_splits(source_split_records, args)
    source_to_split = split_map(splits)
    if getattr(args, "model_progress", True):
        planned_keys = estimate_model_analysis_keys(
            source_paths=source_writer.paths(),
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
        )
        cached_keys = {
            key
            for key in planned_keys
            if key in cache.items and cached_extraction_is_reusable(cache.items[key], args)
        }
        progress.register(planned_keys)
        for cache_key in cached_keys:
            progress.mark(cache_key, "cached")

    precomputed_text_task_count = evaluation_context.text_task_count
    precomputed_image_task_count = evaluation_context.image_task_count
    if getattr(args, "precompute_model_cache", False) or getattr(args, "precompute_text_model_cache", False):
        text_tasks = collect_extraction_tasks_from_tables(
            source_paths=source_writer.paths(),
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            args=args,
            asset_types={"text"},
        )
        image_tasks = []
        if getattr(args, "precompute_model_cache", False):
            image_tasks = collect_extraction_tasks_from_tables(
                source_paths=source_writer.paths(),
                assets=assets,
                entity_to_assets=entity_to_assets,
                wiki_to_entity_id=wiki_to_entity_id,
                args=args,
                asset_types={"image"},
            )
        pending_text_tasks = tasks_requiring_model_analysis(
            text_tasks, cache, args, progress=progress
        )
        pending_image_tasks = tasks_requiring_model_analysis(
            image_tasks, cache, args, progress=progress
        )
        precomputed_text_task_count += len(pending_text_tasks)
        precomputed_image_task_count += len(pending_image_tasks)
        logging.info(
            "Pending model extraction tasks before table processing: text=%d image=%d",
            len(pending_text_tasks),
            len(pending_image_tasks),
        )
        if not pending_text_tasks and not pending_image_tasks:
            logging.info("All model extraction tasks are cached; skipping model analysis")
        tasks_by_kind = {"text": pending_text_tasks}
        if getattr(args, "precompute_model_cache", False):
            tasks_by_kind["image"] = pending_image_tasks
        precompute_extraction_task_groups(
            extractor=extractor,
            cache=cache,
            tasks_by_kind=tasks_by_kind,
            args=args,
            state=concurrency_state,
            progress=progress,
            write_done_markers=False,
        )
        write_done_markers_after_selection(
            args,
            text_task_count=precomputed_text_task_count,
            image_task_count=precomputed_image_task_count,
        )
    query_writer = ShardedJsonlWriter(query_tables_dir, records_per_shard)
    data_lake_writer = ShardedJsonlWriter(data_lake_tables_dir, records_per_shard)
    extraction_writer = ShardedJsonlWriter(extraction_dir, records_per_shard)
    recovery_writer = ShardedJsonlWriter(recovery_dir, records_per_shard)
    qrels: list[dict[str, Any]] = []
    table_decisions: list[dict[str, Any]] = []
    query_table_count = 0
    data_lake_table_count = 0
    queryable_source_tables = 0
    multimodal_queryable_source_tables = 0
    explicit_join_source_tables = 0
    rejected_source_tables = 0
    implicit_query_table_count = 0
    explicit_join_query_table_count = 0
    implicit_query_counts_by_split = {
        "train": 0,
        "dev": 0,
        "test": 0,
    }
    # In match_implicit mode candidates are query-level.  A source table may
    # therefore contribute several candidate IDs (one per visible join
    # column), while its queryability decision remains source-level.
    explicit_candidate_splits: dict[str, str] = {}
    explicit_candidate_source_ids: dict[str, str] = {}
    explicit_candidate_decision_indices: dict[str, int] = {}
    explicit_candidate_counts = {"train": 0, "dev": 0, "test": 0}
    explicit_candidate_source_counts = {"train": 0, "dev": 0, "test": 0}
    try:
        with (
            query_writer as query_handle,
            data_lake_writer as data_lake_handle,
            extraction_writer as extraction_handle,
            recovery_writer as recovery_handle,
        ):
            for source_table in iter_jsonl_records(source_writer.paths()):
                source_table_id = str(source_table["source_table_id"])
                split = source_to_split.get(source_table_id, "test")
                query_tables, data_lake_tables, table_qrels, decision = build_table_join_records(
                    source_table=source_table,
                    split=split,
                    assets=assets,
                    entity_to_assets=entity_to_assets,
                    wiki_to_entity_id=wiki_to_entity_id,
                    extractor=extractor,
                    cache=cache,
                    progress=progress,
                    concurrency_state=concurrency_state,
                    extraction_writer=extraction_handle,
                    recovery_writer=recovery_handle,
                    args=args,
                )
                if query_tables:
                    queryable_source_tables += 1
                    if decision.get("reason") == "explicit_join_fallback":
                        explicit_join_source_tables += 1
                        explicit_join_query_table_count += len(query_tables)
                    else:
                        multimodal_queryable_source_tables += 1
                        implicit_query_table_count += len(query_tables)
                        implicit_query_counts_by_split[split] += len(
                            query_tables
                        )
                else:
                    rejected_source_tables += 1
                candidates = decision.get("explicit_join_candidates")
                if not isinstance(candidates, list):
                    candidate = decision.get("explicit_join_candidate")
                    candidates = [candidate] if isinstance(candidate, dict) else []
                deferred_candidate = (
                    args.explicit_join_fallback_mode == "match_implicit"
                    and bool(candidates)
                )
                decision["source_table_id"] = source_table_id
                decision["split"] = split
                if deferred_candidate:
                    for candidate in candidates:
                        candidate_id = clean_text(candidate.get("candidate_id"))
                        if not candidate_id:
                            raise ValueError(
                                "explicit join candidate is missing candidate_id: "
                                f"{source_table_id}"
                            )
                        if candidate_id in explicit_candidate_splits:
                            raise ValueError(
                                "duplicate explicit join candidate: "
                                f"{candidate_id}"
                            )
                        explicit_candidate_splits[candidate_id] = split
                        explicit_candidate_source_ids[candidate_id] = source_table_id
                        explicit_candidate_decision_indices[candidate_id] = len(
                            table_decisions
                        )
                table_decisions.append(decision)
                for record in query_tables:
                    write_jsonl_record(query_handle, record)
                    query_table_count += 1
                    splits[split]["query_table_ids"].append(record["table_id"])
                for record in ([] if deferred_candidate else data_lake_tables):
                    write_jsonl_record(data_lake_handle, record)
                    data_lake_table_count += 1
                    splits[split]["data_lake_table_ids"].append(record["table_id"])
                qrels.extend(table_qrels)
                if (query_table_count + data_lake_table_count) % flush_every == 0:
                    query_handle.flush()
                    data_lake_handle.flush()
                    extraction_handle.flush()
                    recovery_handle.flush()

            if args.explicit_join_fallback_mode == "match_implicit":
                selected_explicit, explicit_candidate_counts = (
                    select_balanced_explicit_join_candidates(
                        candidate_splits=explicit_candidate_splits,
                        implicit_query_counts=implicit_query_counts_by_split,
                        args=args,
                    )
                )
                source_candidates_by_split: dict[str, set[str]] = {
                    "train": set(),
                    "dev": set(),
                    "test": set(),
                }
                for candidate_id, candidate_split in explicit_candidate_splits.items():
                    source_candidates_by_split[candidate_split].add(
                        explicit_candidate_source_ids[candidate_id]
                    )
                explicit_candidate_source_counts = {
                    split_name: len(source_ids)
                    for split_name, source_ids in source_candidates_by_split.items()
                }
                source_by_id = {
                    str(source_table["source_table_id"]): source_table
                    for source_table in final_source_tables
                    if str(source_table["source_table_id"]) in set(
                        explicit_candidate_source_ids.values()
                    )
                }
                selected_by_source: dict[str, list[str]] = defaultdict(list)
                for candidate_id in selected_explicit:
                    selected_by_source[
                        explicit_candidate_source_ids[candidate_id]
                    ].append(candidate_id)
                candidate_source_ids = sorted(
                    set(explicit_candidate_source_ids.values())
                )
                for source_table_id in candidate_source_ids:
                    split = source_to_split.get(source_table_id, "test")
                    source_table = source_by_id[source_table_id]
                    selected_candidate_ids = sorted(
                        selected_by_source.get(source_table_id, [])
                    )
                    if not selected_candidate_ids:
                        record = raw_data_lake_record(source_table, split)
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
                        splits[split]["data_lake_table_ids"].append(
                            record["table_id"]
                        )
                        continue
                    decision_index = explicit_candidate_decision_indices[
                        selected_candidate_ids[0]
                    ]
                    original_decision = table_decisions[decision_index]
                    explicit_queries: list[dict[str, Any]] = []
                    explicit_targets: list[dict[str, Any]] = []
                    explicit_qrels: list[dict[str, Any]] = []
                    explicit_decisions: list[dict[str, Any]] = []
                    selected_candidates: list[dict[str, Any]] = []
                    for candidate_id in selected_candidate_ids:
                        candidate_decision = next(
                            candidate
                            for candidate in original_decision[
                                "explicit_join_candidates"
                            ]
                            if candidate.get("candidate_id") == candidate_id
                        )
                        (
                            candidate_queries,
                            candidate_targets,
                            candidate_qrels,
                            candidate_result_decision,
                        ) = materialize_balanced_explicit_join_candidate(
                            source_table=source_table,
                            split=split,
                            candidate_decision=candidate_decision,
                            args=args,
                        )
                        explicit_queries.extend(candidate_queries)
                        explicit_targets.extend(candidate_targets)
                        explicit_qrels.extend(candidate_qrels)
                        explicit_decisions.append(candidate_result_decision)
                        selected_candidates.append(candidate_decision)
                    explicit_decision = {
                        **original_decision,
                        **explicit_decisions[0],
                        "source_table_id": source_table_id,
                        "split": split,
                        "qualified_columns": [
                            qualified
                            for item in explicit_decisions
                            for qualified in item.get("qualified_columns", [])
                        ],
                        "explicit_join_candidates": selected_candidates,
                        "explicit_join_candidate": selected_candidates[0],
                        "explicit_join_query_count": len(explicit_queries),
                    }
                    table_decisions[decision_index] = explicit_decision
                    rejected_source_tables -= 1
                    queryable_source_tables += 1
                    explicit_join_source_tables += 1
                    for record in explicit_queries:
                        write_jsonl_record(query_handle, record)
                        query_table_count += 1
                        explicit_join_query_table_count += 1
                        splits[split]["query_table_ids"].append(
                            record["table_id"]
                        )
                    for record in explicit_targets:
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
                        splits[split]["data_lake_table_ids"].append(
                            record["table_id"]
                        )
                    qrels.extend(explicit_qrels)
                if explicit_join_query_table_count != implicit_query_table_count:
                    raise ValueError(
                        "explicit and implicit query counts are not balanced: "
                        f"explicit={explicit_join_query_table_count}, "
                        f"implicit={implicit_query_table_count}"
                    )
    finally:
        if progress is not None:
            progress.close()

    for split in ("train", "dev", "test"):
        splits[split]["query_table_ids"] = sorted(splits[split]["query_table_ids"])
        splits[split]["data_lake_table_ids"] = sorted(splits[split]["data_lake_table_ids"])

    qrels_count = write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "table_queryability_decisions.jsonl", table_decisions)
    write_json(output_dir / "splits.json", splits)

    replacement_selection = {
        "rounds": [asdict(round_stats) for round_stats in selection.rounds],
        "candidates_consumed": selection.candidates_consumed,
        "candidate_exhausted": selection.candidate_exhausted,
        "unfilled_slots": selection.unfilled_slots,
    }
    source_sampling = {
        "mode": "seeded_random_file_and_table_order",
        "seed": args.seed,
        "entity_column_policy": "require_query_sized_linked_candidate_before_global_sampling",
        "min_linked_entity_rows": args.query_rows_per_table,
        "unrecoverable_replacement_rounds": policy.rounds,
        "unrecoverable_drop_probability": policy.drop_probability,
        "replacement_scope": "all_current_failed_slots",
        "discarded_candidate_cache_policy": "retain_persistent_cache",
    }
    stats = {
        "processed_tables": counters.processed_tables,
        "skipped_tables": counters.skipped_tables,
        "source_tables": source_table_count,
        "queryable_source_tables": queryable_source_tables,
        "multimodal_queryable_source_tables": multimodal_queryable_source_tables,
        "explicit_join_source_tables": explicit_join_source_tables,
        "implicit_join_query_tables": implicit_query_table_count,
        "explicit_join_query_tables": explicit_join_query_table_count,
        "explicit_join_candidate_tables": sum(
            explicit_candidate_source_counts.values()
        ),
        "explicit_join_candidate_tables_by_split": explicit_candidate_source_counts,
        "explicit_join_candidate_queries": sum(explicit_candidate_counts.values()),
        "explicit_join_candidate_queries_by_split": explicit_candidate_counts,
        "implicit_join_query_tables_by_split": implicit_query_counts_by_split,
        "rejected_source_tables": rejected_source_tables,
        "query_tables": query_table_count,
        "data_lake_tables": data_lake_table_count,
        "qrels": qrels_count,
        "unique_wiki_entities": len(entities),
        "text_assets": text_asset_count,
        "image_assets": image_asset_count,
        "table_asset_links": table_asset_link_count,
        "api_failures": api_failures,
        "attribute_extractions": extraction_writer.total_records,
        "evidence_recoveries": recovery_writer.total_records,
        "model_inference": model_call_stats_summary(extractor),
        "model_concurrency": concurrency_state.summary(),
        "precomputed_text_model_cache_tasks": precomputed_text_task_count,
        "precomputed_image_model_cache_tasks": precomputed_image_task_count,
        "wikipedia_workers": 1,
        "min_recovered_value_ratio": args.min_recovered_value_ratio,
        "min_recovery_denominator": args.min_recovery_denominator,
        "query_rows_per_table": args.query_rows_per_table,
        "max_train_query_row_views_per_join": args.max_train_query_row_views_per_join,
        "explicit_join_fallback_mode": args.explicit_join_fallback_mode,
        "explicit_join_fallback_ratio": args.explicit_join_fallback_ratio,
        "skipped_reasons": dict(counters.skip_reasons),
        "sampling_mode": source_sampling["mode"],
        "sampling_seed": args.seed,
        "entity_column_policy": source_sampling["entity_column_policy"],
        "unrecoverable_replacement_rounds": policy.rounds,
        "unrecoverable_drop_probability": policy.drop_probability,
        "unrecoverable_replacement_scope": source_sampling["replacement_scope"],
        "discarded_candidate_cache_policy": source_sampling[
            "discarded_candidate_cache_policy"
        ],
        "random_candidates_structurally_accepted": selection.candidates_consumed,
        "initial_slots_filled": target_count - selection.unfilled_slots,
        "replacement_selection": replacement_selection,
        "cleanup": asdict(cleanup_totals),
        "notes": [
            "source_tables are the fixed data-lake base pool",
            "tables without a candidate entity column or enough linked entity rows for one query are filtered before the seeded global source-table sample",
            "train join chains emit up to max_train_query_row_views_per_join deterministic disjoint row views; dev/test emit one canonical view",
            "query_tables use a capped recovery threshold over valid entity rows and contain exactly query_rows_per_table sampled rows",
            "wide source tables emit one variant per qualifying bridge attribute; all qualifying bridge columns stay out of every sibling query, and ordinary context columns are partitioned into source-level query-only and target-only sides",
            "identical visible queries from one source table are merged and retain every hidden attribute, target table ID, chain ID, and qrel",
            "generated target data-lake tables retain every source row after column projection; rejected source tables remain raw",
            "match_implicit deterministically selects one viable explicit join per implicit query within each split",
            "evidence_recoveries record query_table -> multimodal evidence -> target_table paths at entity/row/attribute granularity",
            "api_failures is retained for backward compatibility; use manifest.wikimedia_media for media transfer counters",
            "each replacement pass draws from every currently failed slot; retained_failed slots are deferred to the next pass unless the pass is terminal",
            "discarded candidates are pruned from active output material while downloaded images and persistent Wikipedia/model caches are retained for later runs",
        ],
    }
    write_json(output_dir / "stats.json", stats)

    manifest = {
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
            "query_tables": query_writer.manifest(output_dir),
            "data_lake_tables": data_lake_writer.manifest(output_dir),
            "entities": entities_writer.manifest(output_dir),
            "bridge_assets": bridge_assets_writer.manifest(output_dir),
            "table_asset_links": table_asset_links_writer.manifest(output_dir),
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
            "query_rows_per_table": args.query_rows_per_table,
            "query_row_selection": "recovery_balanced_disjoint_train_views",
            "max_train_query_row_views_per_join": args.max_train_query_row_views_per_join,
            "evaluation_query_row_views_per_join": 1,
            "explicit_join_fallback_mode": args.explicit_join_fallback_mode,
            "explicit_join_fallback_ratio": args.explicit_join_fallback_ratio,
            "explicit_join_column_policy": "seeded_random_non_entity_visible_column",
            "target_row_scope": "all_source_rows",
            "min_rows_per_output_table": args.min_rows_per_output_table,
            "min_recovered_value_ratio": args.min_recovered_value_ratio,
            "min_recovery_denominator": args.min_recovery_denominator,
            "max_query_tables_per_source_table": args.max_query_tables_per_source_table,
            "max_query_context_attrs": args.max_query_context_attrs,
            "qualified_attribute_policy": "all_safe_variants",
            "sibling_source_column_policy": "globally_disjoint_query_and_target_sides",
            "identical_visible_query_policy": "merge_with_multiple_qrels",
        },
        "source_sampling": source_sampling,
        "model_endpoints": {
            "text_model_base_url": args.text_model_base_url,
            "text_model_base_urls": getattr(args, "text_model_base_urls", None),
            "text_model_base_urls_file": getattr(args, "text_model_base_urls_file", None),
            "remote_text_model_base_url": getattr(args, "remote_text_model_base_url", None),
            "remote_text_model_base_urls": getattr(args, "remote_text_model_base_urls", None),
            "remote_text_model_base_urls_file": getattr(args, "remote_text_model_base_urls_file", None),
            "text_model_name": args.text_model_name,
            "image_model_base_url": args.image_model_base_url,
            "image_model_base_urls": getattr(args, "image_model_base_urls", None),
            "image_model_base_urls_file": getattr(args, "image_model_base_urls_file", None),
            "remote_image_model_base_url": getattr(args, "remote_image_model_base_url", None),
            "remote_image_model_base_urls": getattr(args, "remote_image_model_base_urls", None),
            "remote_image_model_base_urls_file": getattr(args, "remote_image_model_base_urls_file", None),
            "image_model_name": args.image_model_name,
            "prompt_version": PROMPT_VERSION,
            "image_model_max_tokens": getattr(args, "image_model_max_tokens", DEFAULT_IMAGE_MODEL_MAX_TOKENS),
            "image_request_max_pixels": getattr(args, "image_request_max_pixels", DEFAULT_IMAGE_REQUEST_MAX_PIXELS),
            "precompute_model_cache": getattr(args, "precompute_model_cache", False),
            "precompute_text_model_cache": getattr(args, "precompute_text_model_cache", False),
            "model_start_marker": getattr(args, "model_start_marker", None),
            "model_ready_marker": getattr(args, "model_ready_marker", None),
            "model_text_done_marker": getattr(args, "model_text_done_marker", None),
            "model_image_done_marker": getattr(args, "model_image_done_marker", None),
            "disable_thinking": args.disable_thinking,
            "reparse_cached_model_outputs": args.reparse_cached_model_outputs,
            "refresh_invalid_model_cache": args.refresh_invalid_model_cache,
            "cache_failed_model_outputs": args.cache_failed_model_outputs,
            "model_attribute_errors_path": model_attribute_errors_path,
            "context_retry_image_max_pixels": args.context_retry_image_max_pixels,
            "configured_text_model_workers": args.text_model_workers,
            "configured_image_model_workers": args.image_model_workers,
            "configured_remote_text_model_workers": getattr(args, "remote_text_model_workers", 0),
            "configured_remote_image_model_workers": getattr(args, "remote_image_model_workers", 0),
            "final_model_concurrency": stats["model_concurrency"],
            "inference_stats": stats["model_inference"],
        },
        "cache": {
            "root_dir": str(cache_paths["root_dir"]),
            "wikipedia_cache_dir": str(cache_paths["wikipedia_cache_dir"]),
            "wikipedia_image_dir": str(cache_paths["wikipedia_image_dir"]),
            "model_attribute_extractions": str(cache_paths["model_attribute_extractions"]),
        },
        "wikipedia_cache": {
            "cache_dir": str(cache_paths["wikipedia_cache_dir"]),
            "image_dir": str(cache_paths["wikipedia_image_dir"]),
            "workers": 1,
        },
        "wikimedia_media": wikimedia_media,
        "note": "Read shards listed in this manifest; stale files from older runs may exist if an output directory is reused.",
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    return stats


def build_dataset(
    args: argparse.Namespace,
    *,
    extractor: LocalAttributeExtractor | None = None,
) -> dict[str, Any]:
    owned_progress: list[ModelAnalysisProgress] = []
    try:
        return _build_dataset(
            args,
            owned_progress,
            injected_extractor=extractor,
        )
    finally:
        for progress in owned_progress:
            if not getattr(progress, "_closed", False):
                progress.close()


def parse_args(
    argv: list[str] | None = None,
    *,
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multimodal joinability discovery dataset by asking local "
            "text/image models which entity attributes can be extracted from Wikipedia assets."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing EntiTables .json files.")
    parser.add_argument("--output_dir", required=True, help="Directory where dataset artifacts are written.")
    parser.add_argument("--max_source_tables", type=int, default=20000, help="Number of filtered EntiTables source tables to use as the data-lake base.")
    parser.add_argument("--max_tables", type=int, default=None, dest="max_source_tables", help="(Deprecated) Use --max_source_tables instead.")
    parser.add_argument("--max_entities", type=int, default=None, help="Maximum number of entities for Wikipedia asset fetching.")
    parser.add_argument("--max_images_per_entity", type=int, default=3)
    parser.add_argument("--text_asset_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_asset_chunk_chars", type=int, default=120)
    parser.add_argument("--max_text_asset_chunks_per_entity", type=int, default=3)
    parser.add_argument("--wiki_link_threshold", type=float, default=0.3)
    parser.add_argument("--min_rows", type=int, default=2)
    parser.add_argument("--min_cols", type=int, default=2)
    parser.add_argument("--min_rows_per_output_table", type=int, default=2)
    parser.add_argument(
        "--query_rows_per_table",
        type=int,
        default=5,
        help="Exact number of recoverable-first source rows sampled into each query; generated targets retain all source rows.",
    )
    parser.add_argument(
        "--max_train_query_row_views_per_join",
        type=int,
        default=5,
        help=(
            "Maximum deterministic disjoint query row views per train join chain; "
            "0 means use every feasible view. Dev/test always use one."
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between Action API requests; does not control media downloads.")
    parser.add_argument(
        "--wikipedia_user_agent",
        default=default_wikipedia_user_agent(),
        help=(
            "Descriptive User-Agent for MediaWiki API requests. Include a project name and contact address. "
            "Defaults to $WIKIPEDIA_USER_AGENT when set, otherwise uses the built-in MMDD EntiTables User-Agent."
        ),
    )
    parser.add_argument(
        "--wikipedia_workers",
        type=int,
        default=1,
        help="Deprecated compatibility option. Action API fetching is always serial and this does not control media downloads.",
    )
    parser.add_argument(
        "--media_download_workers",
        type=int,
        default=2,
        help="Concurrent Wikimedia media responses; accepted range 1..2. Independent of Action API request pacing.",
    )
    parser.add_argument(
        "--media_max_mbps",
        type=float,
        default=24.0,
        help="Aggregate Wikimedia media bandwidth in Mbps; accepted range (0, 25].",
    )
    parser.add_argument(
        "--media_max_retries",
        type=int,
        default=5,
        help="Retries per Wikimedia media item after the initial attempt.",
    )
    parser.add_argument(
        "--media_retry_base_seconds",
        type=float,
        default=5.0,
        help="Base delay in seconds for retryable Wikimedia media failures.",
    )
    parser.add_argument(
        "--media_retry_max_seconds",
        type=float,
        default=60.0,
        help="Maximum local backoff in seconds for Wikimedia media retries.",
    )
    parser.add_argument(
        "--media_chunk_bytes",
        type=int,
        default=131072,
        help="Stream chunk size in bytes for Wikimedia media bandwidth accounting.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--unrecoverable_replacement_rounds", type=int, default=2)
    parser.add_argument("--unrecoverable_drop_probability", type=float, default=0.5)
    parser.add_argument("--flush_every_records", type=int, default=500)
    parser.add_argument("--records_per_shard", type=int, default=50000)
    parser.add_argument("--no_wikipedia", action="store_true", help="Skip MediaWiki API calls. Queryable tables will normally be zero.")
    parser.add_argument("--cache_dir", default=str(DEFAULT_SHARED_CACHE_DIR), help="Shared cache root for Wikipedia metadata, downloaded images, and model attribute extraction cache.")
    parser.add_argument("--wikipedia_cache_dir", default=None, help="Override MediaWiki page and imageinfo JSONL cache directory. Defaults to <cache_dir>/wikipedia.")
    parser.add_argument("--wikipedia_image_dir", default=None, help="Override downloaded Wikipedia image directory. Defaults to <cache_dir>/images.")
    parser.add_argument("--split_by", choices=["source_table_id", "page_title"], default="page_title")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_column_non_empty_ratio", type=float, default=0.5)
    parser.add_argument(
        "--min_recovered_value_ratio",
        type=float,
        default=0.6,
        help="Minimum recovered fraction applied to min(valid entity rows, query_rows_per_table).",
    )
    parser.add_argument(
        "--min_recovery_denominator",
        type=int,
        default=2,
        help="Minimum valid entity row count for a candidate join column.",
    )
    parser.add_argument("--max_query_tables_per_source_table", type=int, default=0, help="0 means emit all qualifying join columns.")
    parser.add_argument("--max_query_context_attrs", type=int, default=1)
    parser.add_argument("--max_target_context_attrs", type=int, default=2)
    parser.add_argument(
        "--explicit_join_fallback_mode",
        choices=EXPLICIT_JOIN_FALLBACK_MODES,
        default=DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE,
        help=(
            "match_implicit selects exactly one visible-join query per "
            "implicit query within each split; ratio retains legacy sampling."
        ),
    )
    parser.add_argument(
        "--explicit_join_fallback_ratio",
        type=float,
        default=DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO,
        help=(
            "Seeded fraction of tables rejected by multimodal recovery to turn "
            "into ordinary joins with a visible non-entity join column in both "
            "query and target; 0 disables the fallback."
        ),
    )
    parser.add_argument(
        "--model_routing_manifest",
        default=None,
        help=(
            "Authoritative mmdd-model-routing-v1 manifest. When set, empty "
            "modality routes do not fall back to static model URLs."
        ),
    )
    parser.add_argument(
        "--remote_model_routing_manifest",
        default=None,
        help=(
            "Authoritative routing manifest for an opportunistically borrowed "
            "remote pool. Empty routes never fall back to static remote URLs."
        ),
    )
    parser.add_argument("--text_model_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--text_model_base_urls", nargs="*", default=None, help="Additional text-model OpenAI-compatible base URLs. Values may also be comma-separated.")
    parser.add_argument("--text_model_base_urls_file", default=None, help="Optional newline-separated text-model base URL file re-read before each text request. Dynamic vLLM runners can append endpoints here.")
    parser.add_argument("--remote_text_model_base_url", default=None, help="Primary remote text-model OpenAI-compatible base URL. Kept in a separate concurrency pool from local endpoints.")
    parser.add_argument("--remote_text_model_base_urls", nargs="*", default=None, help="Additional remote text-model base URLs for the remote pool. Values may also be comma-separated.")
    parser.add_argument("--remote_text_model_base_urls_file", default=None, help="Optional newline-separated remote text endpoint file re-read before each remote request.")
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_model_api_key", default=None)
    parser.add_argument("--remote_text_model_api_key", default=None, help="Remote text endpoint API key. Falls back to MMDD_REMOTE_TEXT_MODEL_API_KEY, then the local text key/VLLM_API_KEY.")
    parser.add_argument("--image_model_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--image_model_base_urls", nargs="*", default=None, help="Additional image-model OpenAI-compatible base URLs. Values may also be comma-separated.")
    parser.add_argument("--image_model_base_urls_file", default=None, help="Optional newline-separated image-model base URL file re-read before each image request. Dynamic vLLM runners can append endpoints here.")
    parser.add_argument("--remote_image_model_base_url", default=None, help="Primary remote image-model OpenAI-compatible base URL. Kept in a separate concurrency pool from local endpoints.")
    parser.add_argument("--remote_image_model_base_urls", nargs="*", default=None, help="Additional remote image-model base URLs for the remote pool. Values may also be comma-separated.")
    parser.add_argument("--remote_image_model_base_urls_file", default=None, help="Optional newline-separated remote image endpoint file re-read before each remote request.")
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Thinking")
    parser.add_argument("--image_model_api_key", default=None)
    parser.add_argument("--remote_image_model_api_key", default=None, help="Remote image endpoint API key. Falls back to MMDD_REMOTE_IMAGE_MODEL_API_KEY, then the local image key/VLLM_API_KEY.")
    parser.add_argument(
        "--remote_layout_control_url",
        default=None,
        help="Loopback URL of the SSH-forwarded remote layout control API.",
    )
    parser.add_argument("--remote_layout_control_token_file")
    parser.add_argument("--remote_layout_controller_id_file")
    parser.add_argument("--remote_layout_lock_file")
    parser.add_argument("--remote_layout_primary_image_url")
    parser.add_argument("--remote_layout_switchable_url")
    parser.add_argument("--remote_layout_coordination_dir")
    parser.add_argument(
        "--remote_layout_lease_ttl_seconds", type=int, default=15
    )
    parser.add_argument(
        "--remote_layout_lease_renew_seconds", type=float, default=5.0
    )
    parser.add_argument(
        "--remote_layout_request_timeout_seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--remote_layout_reconnect_timeout_seconds", type=float, default=30.0
    )
    parser.add_argument(
        "--remote_layout_operation_timeout_seconds", type=float, default=1200.0
    )
    parser.add_argument(
        "--remote_layout_drain_timeout_seconds", type=float, default=300.0
    )
    parser.add_argument("--remote_layout_poll_seconds", type=float, default=1.0)
    parser.add_argument(
        "--remote_layout_workload_poll_seconds", type=float, default=2.0
    )
    parser.add_argument(
        "--remote_layout_stability_seconds", type=float, default=0.0
    )
    parser.add_argument(
        "--remote_layout_health_timeout_seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--remote_layout_health_stable_polls", type=int, default=2
    )
    parser.add_argument(
        "--remote_layout_coordination_poll_seconds", type=float, default=0.2
    )
    parser.add_argument("--model_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--model_temperature", type=float, default=0.0)
    parser.add_argument("--model_max_tokens", type=int, default=1024)
    parser.add_argument("--image_model_max_tokens", type=int, default=DEFAULT_IMAGE_MODEL_MAX_TOKENS, help="Maximum completion tokens for image-model extraction calls.")
    parser.add_argument("--image_request_max_pixels", type=int, default=DEFAULT_IMAGE_REQUEST_MAX_PIXELS, help="Resize local images to this pixel budget before image-model requests. Use 0 to send original local images.")
    parser.add_argument("--text_model_workers", type=int, default=1, help="Concurrent text-model requests. Default 1 is conservative for 24GB GPUs.")
    parser.add_argument("--image_model_workers", type=int, default=1, help="Concurrent image-model requests. Default 1 is conservative for 24GB GPUs.")
    parser.add_argument("--remote_text_model_workers", type=int, default=0, help="Concurrent remote text-model requests, independent of --text_model_workers. Zero disables the remote text pool.")
    parser.add_argument("--remote_image_model_workers", type=int, default=0, help="Concurrent remote image-model requests, independent of --image_model_workers. Zero disables the remote image pool.")
    parser.add_argument("--enable_thinking", dest="disable_thinking", action="store_false", help="Allow Qwen thinking mode. By default, chat_template_kwargs disables thinking for extraction calls.")
    parser.add_argument("--no_reparse_cached_model_outputs", dest="reparse_cached_model_outputs", action="store_false", help="Use cached parsed attributes as-is instead of reparsing cached raw_response with the current JSON parser.")
    parser.add_argument("--refresh_invalid_model_cache", action="store_true", help="When a cached raw_response still reparses to no valid candidate attributes, call the model again with the current request settings.")
    parser.add_argument("--model_max_retries", type=int, default=2)
    parser.add_argument("--model_retry_sleep_seconds", type=float, default=2.0)
    parser.add_argument("--no_reuse_model_cache", action="store_true")
    parser.add_argument("--cache_failed_model_outputs", action="store_true", help="Persist failed model calls in the extraction cache. By default failures are written only to this run's output so the next run retries them.")
    parser.add_argument("--model_attribute_errors_path", default="", help="JSONL path for failed model extraction records. Defaults to <output_dir>/model_attribute_errors.jsonl.")
    parser.add_argument("--context_retry_image_max_pixels", type=int, default=DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS, help="Temporary max pixel count for local images retried after VL context-length errors. Original image files are not modified.")
    parser.add_argument("--no_model_progress", dest="model_progress", action="store_false", help="Disable the local model analysis progress bar.")
    parser.add_argument("--precompute_model_cache", action="store_true", help="Run text and image extraction tasks into the shared model cache before table processing, writing per-modality done markers as each modality finishes.")
    parser.add_argument("--precompute_text_model_cache", action="store_true", help="Run all text extraction tasks into the shared model cache before image-heavy table processing.")
    parser.add_argument("--model_start_marker", default=None, help="Write this JSON marker after Wikipedia/material preparation is complete and model requests are about to start.")
    parser.add_argument("--model_ready_marker", default=None, help="Wait for this JSON marker before issuing model requests. Dynamic vLLM runners write it after servers are healthy.")
    parser.add_argument("--model_ready_timeout_seconds", type=float, default=None, help="Optional timeout while waiting for a matching strict model-ready marker.")
    parser.add_argument("--model_text_done_marker", default=None, help="Write this JSON marker after --precompute_text_model_cache completes.")
    parser.add_argument("--model_image_done_marker", default=None, help="Write this JSON marker after image model cache precompute completes.")
    parser.add_argument("--model_round_control_dir", default=None, help="Optional generation-scoped handshake directory used by a dynamic model runner between batched inference rounds.")
    parser.add_argument("--model_round_run_id", default=None, help="Opaque dynamic-run identifier used to reject stale model round markers.")
    parser.add_argument("--run_fingerprint", default="", help="Staged-run identity used to fence stale model markers.")
    if configure_parser is not None:
        configure_parser(parser)
    parser.set_defaults(disable_thinking=True, reparse_cached_model_outputs=True)
    parser.set_defaults(model_progress=True)
    return parser.parse_args(argv)


def main() -> None:
    setup_logging()
    stats = build_dataset(parse_args())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

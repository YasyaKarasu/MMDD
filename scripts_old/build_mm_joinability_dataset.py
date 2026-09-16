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
from dataclasses import asdict, dataclass, field as dataclass_field, replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from urllib.parse import quote

from json_repair import repair_json

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
from model_endpoint_pool import (
    EndpointPoolUnavailableError,
    ModelEndpointScheduler,
    load_model_endpoint_config,
)
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
    table_column_values,
    write_json,
    write_jsonl,
)
from wikimedia_media import MediaFailureRecorder, MediaPolicyConfig


PROMPT_VERSION = "entity_attribute_extraction_v5_batched_leave_one_out"
MODEL_AUTO_CHECK_SCHEMA_VERSION = (
    "model-output-auto-check-v6-redundant-group-mask"
)
AUTO_CHECK_REVIEW_POLICY_LOCAL = "local_only"
AUTO_CHECK_REVIEW_POLICY_LEGACY = (
    "local_then_luna_on_nonmatch_terra_adjudication_v1"
)
AUTO_CHECK_REVIEW_POLICY_CASCADE = (
    "local_luna_consensus_terra_adjudication_v1"
)
QUERY_RECOVERY_REMOTE_EVIDENCE_CACHE_VERSION = (
    "query-recovery-remote-evidence-v2-query-visible"
)
DEFAULT_AUTO_CHECK_LUNA_MODEL = "gpt-5.6-luna"
DEFAULT_AUTO_CHECK_TERRA_MODEL = "gpt-5.6-terra"
DEFAULT_AUTO_CHECK_API_CONFIG_FILE = Path(".auto_check_apis.json")
# Compatibility name: the first OpenAI fallback is now Luna.
DEFAULT_AUTO_CHECK_OPENAI_MODEL = DEFAULT_AUTO_CHECK_LUNA_MODEL
DEFAULT_AUTO_CHECK_OPENAI_BASE_URL = "https://api.openai.com/v1"
MAX_AUTO_CHECK_OPENAI_CONCURRENCY = 5
DEFAULT_AUTO_CHECK_PROFILE_MAX_CONCURRENCY = 20
AUTO_CHECK_PROFILE_INITIAL_CONCURRENCY = 5
AUTO_CHECK_PROFILE_SUCCESSES_PER_INCREASE = 20
AUTO_CHECK_PROFILE_FAILURE_COOLDOWN_SECONDS = 30.0
AUTO_CHECK_PROFILE_MAX_FAILURE_COOLDOWN_SECONDS = 300.0
DEFAULT_SHARED_CACHE_DIR = Path("cache") / "mm_joinability"
DEFAULT_CONTEXT_RETRY_IMAGE_MAX_PIXELS = 262_144
DEFAULT_IMAGE_REQUEST_MAX_PIXELS = 512_000
DEFAULT_IMAGE_MODEL_MAX_TOKENS = 384
DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO = 0.2
DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE = "ratio"
MIN_IMPLICIT_CONTEXT_COLUMNS = 2
EXPLICIT_JOIN_FALLBACK_MODES = (
    "disabled",
    "ratio",
    "match_implicit",
)
SOURCE_SAMPLE_CHECKPOINT_VERSION = "entitables-source-sample-v1"
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
    query_auto_check_cache: ExtractionCache | None = None
    max_entities: int | None = None
    eligible_entity_ids: set[str] = dataclass_field(default_factory=set)
    entity_imageinfo_keys: dict[str, set[str]] = dataclass_field(default_factory=dict)
    text_task_count: int = 0
    image_task_count: int = 0


@dataclass(frozen=True)
class QueryRecoveryCandidate:
    task: ExtractionTask
    extraction: dict[str, Any]
    recovery: dict[str, Any]
    # Internal-only planning metadata.  It is deliberately not copied to
    # canonical recovery records; it is used to ensure that a sibling column
    # from an exact redundancy group never leaks into an auto-check input.
    redundancy_group_attribute_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryRecoveryAutoCheckPlan:
    query_key: str
    required_recovered_rows: int
    source_row_order: tuple[int, ...]
    candidates: tuple[QueryRecoveryCandidate, ...]


class AutoCheckProviderLoadBalancer:
    """Share live provider capacity across initial and final reviewer roles."""

    def __init__(
        self,
        controllers: dict[str, Any],
        *,
        failure_cooldown_seconds: float = (
            AUTO_CHECK_PROFILE_FAILURE_COOLDOWN_SECONDS
        ),
        max_failure_cooldown_seconds: float = (
            AUTO_CHECK_PROFILE_MAX_FAILURE_COOLDOWN_SECONDS
        ),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_cooldown_seconds < 0 or max_failure_cooldown_seconds < 0:
            raise ValueError("auto-check failure cooldowns must be non-negative")
        if max_failure_cooldown_seconds < failure_cooldown_seconds:
            raise ValueError(
                "auto-check max failure cooldown must be at least the base cooldown"
            )
        self.controllers = dict(controllers)
        self.max_parallelism = sum(
            max(1, int(getattr(controller, "max_inflight", 1)))
            for controller in self.controllers.values()
        ) or 1
        self._condition = threading.Condition()
        self._assigned: dict[str, int] = defaultdict(int)
        self._role_cursors: dict[str, int] = defaultdict(int)
        self._failure_streaks: dict[tuple[str, str], int] = defaultdict(int)
        self._cooldown_until: dict[tuple[str, str], float] = defaultdict(float)
        self.failure_cooldown_seconds = float(failure_cooldown_seconds)
        self.max_failure_cooldown_seconds = float(max_failure_cooldown_seconds)
        self._monotonic = monotonic

    def replace_controllers(self, controllers: dict[str, Any]) -> None:
        """Atomically replace provider limits without disturbing active leases."""
        with self._condition:
            self.controllers = dict(controllers)
            self.max_parallelism = sum(
                max(1, int(getattr(controller, "max_inflight", 1)))
                for controller in self.controllers.values()
            ) or 1
            self._condition.notify_all()

    def _current_limit(self, profile_name: str) -> int:
        controller = self.controllers.get(profile_name)
        summary = getattr(controller, "summary", lambda: {})()
        try:
            return max(
                1,
                int(
                    summary.get(
                        "current_inflight_limit",
                        summary.get("max_inflight", 1),
                    )
                ),
            )
        except (AttributeError, TypeError, ValueError):
            return 1

    def _select_profile(
        self,
        reviewers: list[tuple[str, Any]],
        role: str,
        excluded_profiles: set[str],
    ) -> tuple[str, Any] | None:
        available: list[tuple[int, str, Any, int, int]] = []
        now = self._monotonic()
        for index, (profile_name, reviewer) in enumerate(reviewers):
            if profile_name in excluded_profiles:
                continue
            if self._cooldown_until[(profile_name, role)] > now:
                continue
            assigned = self._assigned[profile_name]
            limit = self._current_limit(profile_name)
            if assigned < limit:
                available.append((index, profile_name, reviewer, assigned, limit))
        if not available:
            return None

        # Compare assigned/capacity exactly, then rotate equally loaded endpoints.
        least_loaded = [
            item
            for item in available
            if all(
                item[3] * other[4] <= other[3] * item[4]
                for other in available
            )
        ]
        cursor = self._role_cursors[role] % len(reviewers)
        selected = min(
            least_loaded,
            key=lambda item: (item[0] - cursor) % len(reviewers),
        )
        self._role_cursors[role] = (selected[0] + 1) % len(reviewers)
        return selected[1], selected[2]

    @contextmanager
    def lease(
        self,
        reviewers: list[tuple[str, Any]],
        role: str,
        *,
        excluded_profiles: set[str] | None = None,
    ) -> Iterator[tuple[str, Any]]:
        excluded = set(excluded_profiles or ())
        eligible_profiles = {
            profile_name
            for profile_name, _reviewer in reviewers
            if profile_name not in excluded
        }
        if not eligible_profiles:
            raise RuntimeError(f"no untried auto-check {role} providers remain")
        with self._condition:
            selected = self._select_profile(reviewers, role, excluded)
            while selected is None:
                now = self._monotonic()
                cooldown_waits = [
                    self._cooldown_until[(profile_name, role)] - now
                    for profile_name in eligible_profiles
                    if self._cooldown_until[(profile_name, role)] > now
                ]
                self._condition.wait(
                    timeout=min(cooldown_waits) if cooldown_waits else None
                )
                selected = self._select_profile(reviewers, role, excluded)
            profile_name, reviewer = selected
            self._assigned[profile_name] += 1
        try:
            yield profile_name, reviewer
        finally:
            with self._condition:
                self._assigned[profile_name] -= 1
                self._condition.notify_all()

    def record_success(self, profile_name: str, role: str) -> None:
        with self._condition:
            key = (profile_name, role)
            self._failure_streaks.pop(key, None)
            self._cooldown_until.pop(key, None)
            self._condition.notify_all()

    def record_failure(self, profile_name: str, role: str) -> None:
        with self._condition:
            key = (profile_name, role)
            streak = self._failure_streaks[key] + 1
            self._failure_streaks[key] = streak
            cooldown = min(
                self.max_failure_cooldown_seconds,
                self.failure_cooldown_seconds * (2 ** min(streak - 1, 10)),
            )
            self._cooldown_until[key] = max(
                self._cooldown_until[key],
                self._monotonic() + cooldown,
            )
            self._condition.notify_all()


class BalancedAutoCheckReviewerPool:
    """Dispatch each blind extraction to the least-loaded compatible provider."""

    def __init__(
        self,
        reviewers: list[tuple[str, Any]],
        *,
        role: str,
        load_balancer: AutoCheckProviderLoadBalancer,
        allow_empty: bool = False,
    ) -> None:
        if not reviewers and not allow_empty:
            raise ValueError(f"auto-check {role} reviewer pool must not be empty")
        self.role = role
        self.load_balancer = load_balancer
        self._reviewers_lock = threading.RLock()
        self._reviewers = list(reviewers)
        self._reload_callback: Callable[[], bool] | None = None
        self.companion_final_pool: BalancedAutoCheckReviewerPool | None = None
        self.identity = self._make_identity(self._reviewers)
        self.max_parallelism = self.load_balancer.max_parallelism

    def _make_identity(
        self,
        reviewers: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        return {
            "provider": "balanced_auto_check_reviewer_pool",
            "role": self.role,
            "reviewers": [
                {
                    "api_profile": profile_name,
                    "identity": getattr(reviewer, "identity", None),
                }
                for profile_name, reviewer in reviewers
            ],
        }

    @property
    def reviewers(self) -> list[tuple[str, Any]]:
        """Return the latest reviewer snapshot, reloading JSON if it changed."""
        self.reload_if_changed()
        with self._reviewers_lock:
            return list(self._reviewers)

    def set_reload_callback(self, callback: Callable[[], bool]) -> None:
        self._reload_callback = callback

    def reload_if_changed(self) -> bool:
        callback = self._reload_callback
        return bool(callback()) if callback is not None else False

    def replace_reviewers(self, reviewers: list[tuple[str, Any]]) -> None:
        with self._reviewers_lock:
            self._reviewers = list(reviewers)
            self.identity = self._make_identity(self._reviewers)
            self.max_parallelism = self.load_balancer.max_parallelism

    def has_reviewers(self) -> bool:
        self.reload_if_changed()
        with self._reviewers_lock:
            return bool(self._reviewers)

    def _reviewer_snapshot(self) -> list[tuple[str, Any]]:
        self.reload_if_changed()
        with self._reviewers_lock:
            return list(self._reviewers)

    @staticmethod
    def _selected_identity(profile_name: str, reviewer: Any) -> dict[str, Any]:
        identity = getattr(reviewer, "identity", None)
        return {
            "api_profile": profile_name,
            "model": clean_text(identity.get("model"))
            if isinstance(identity, dict)
            else "",
        }

    def extract_batch(
        self,
        batch: dict[str, Any],
        *,
        on_selected: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, str]] | None:
        reviewers = self._reviewer_snapshot()
        if not reviewers:
            raise RuntimeError(
                f"no auto-check {self.role} providers are configured"
            )
        attempted_profiles: set[str] = set()
        last_error: Exception | None = None
        while len(attempted_profiles) < len(reviewers):
            with self.load_balancer.lease(
                reviewers,
                self.role,
                excluded_profiles=attempted_profiles,
            ) as selected:
                profile_name, reviewer = selected
                attempted_profiles.add(profile_name)
                if on_selected is not None:
                    on_selected(self._selected_identity(profile_name, reviewer))
                try:
                    extracted = reviewer.extract_batches([batch]).get(
                        batch["query_table_id"]
                    )
                except Exception as error:
                    last_error = error
                    self.load_balancer.record_failure(profile_name, self.role)
                    logging.warning(
                        "Auto-check %s provider %s failed (%s); trying another provider",
                        self.role,
                        profile_name,
                        type(error).__name__,
                    )
                    continue
                self.load_balancer.record_success(profile_name, self.role)
                return extracted
        raise TransientModelEndpointError(
            f"all auto-check {self.role} providers failed for one recovery"
        ) from last_error

    def extract_batches(
        self,
        batches: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, str]]]:
        extracted: dict[str, list[dict[str, str]]] = {}
        for batch in batches:
            values = self.extract_batch(batch)
            if values is not None:
                extracted[batch["query_table_id"]] = values
        return extracted


# Compatibility alias for callers that imported the earlier pool name.
StableAutoCheckReviewerPool = BalancedAutoCheckReviewerPool


class AutoCheckAPIConfigReloader:
    """Reload a protected provider file after atomic, validated changes."""

    def __init__(
        self,
        path: Path,
        *,
        reload_config: Callable[[], tuple[int, int, int]],
        active_profile_count: Callable[[], int],
        initial_signature: tuple[Any, ...] | None = None,
    ) -> None:
        self.path = path
        self._reload_config = reload_config
        self._active_profile_count = active_profile_count
        self._lock = threading.Lock()
        self._last_seen_signature = (
            self.file_signature(path)
            if initial_signature is None
            else initial_signature
        )

    @staticmethod
    def file_signature(path: Path) -> tuple[Any, ...]:
        try:
            stat = path.stat()
        except OSError as error:
            return ("unavailable", error.errno)
        return (
            "file",
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_mode & 0o777,
        )

    def reload_if_changed(self) -> bool:
        signature = self.file_signature(self.path)
        with self._lock:
            if signature == self._last_seen_signature:
                return False
            self._last_seen_signature = signature
            try:
                initial_count, final_count, profile_count = self._reload_config()
            except Exception as error:
                logging.warning(
                    "Ignoring updated auto-check API config %s (%s: %s); "
                    "continuing with the last valid configuration (%d profiles)",
                    self.path,
                    type(error).__name__,
                    error,
                    self._active_profile_count(),
                )
                return False
            logging.info(
                "Reloaded auto-check API config %s: %d profiles "
                "(%d initial, %d final)",
                self.path,
                profile_count,
                initial_count,
                final_count,
            )
            return True


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

    auto_check_plans: list[QueryRecoveryAutoCheckPlan] = []
    if auto_check_required(context.extractor):
        if context.query_auto_check_cache is None:
            raise RuntimeError(
                "auto-check-enabled candidate evaluation requires a query cache"
            )
        for source_table in source_tables:
            build_table_join_records(
                source_table=source_table,
                split="candidate",
                assets=context.assets,
                entity_to_assets=context.entity_to_assets,
                wiki_to_entity_id=context.wiki_to_entity_id,
                extractor=context.extractor,
                cache=context.cache,
                progress=None,
                concurrency_state=context.concurrency_state,
                extraction_writer=ListRecordWriter(),
                recovery_writer=ListRecordWriter(),
                args=args,
                query_auto_check_cache=context.query_auto_check_cache,
                apply_query_auto_check=False,
                query_recovery_plans_out=auto_check_plans,
            )
        run_query_recovery_auto_check_round(
            plans=auto_check_plans,
            extractor=context.extractor,
            cache=context.query_auto_check_cache,
            args=args,
            concurrency_state=context.concurrency_state,
        )

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
            query_auto_check_cache=context.query_auto_check_cache,
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


def _source_sample_checkpoint_dir(args: argparse.Namespace) -> Path | None:
    if bool(getattr(args, "no_source_sample_checkpoint", False)):
        return None
    configured = clean_text(getattr(args, "source_sample_checkpoint_dir", ""))
    if configured:
        return Path(configured).resolve()
    output_dir = clean_text(getattr(args, "output_dir", ""))
    if not output_dir:
        return None
    return Path(output_dir).resolve() / "_source_sample_checkpoint"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _source_sample_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": int(args.seed),
        "min_rows": int(args.min_rows),
        "min_cols": int(args.min_cols),
        "wiki_link_threshold": float(args.wiki_link_threshold),
        "query_rows_per_table": configured_query_rows_per_table(args),
        "max_source_tables": (
            None if args.max_source_tables is None else int(args.max_source_tables)
        ),
        "unrecoverable_replacement_rounds": int(
            args.unrecoverable_replacement_rounds
        ),
        "max_scanned_files": (
            None
            if getattr(args, "max_scanned_files", None) is None
            else int(args.max_scanned_files)
        ),
    }


def _source_inventory(
    input_dir: Path,
    json_files: list[Path],
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in json_files:
        stat = path.stat()
        inventory.append(
            {
                "path": path.relative_to(input_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return inventory


def _source_sample_fingerprint(
    inventory: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:
    payload = {
        "schema_version": SOURCE_SAMPLE_CHECKPOINT_VERSION,
        "inventory": inventory,
        "config": config,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _restore_source_sample_refs(
    manifest_path: Path,
    fingerprint: str,
    counters: SourceCandidateCounters,
) -> list[SelectedSourceTableRef] | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SOURCE_SAMPLE_CHECKPOINT_VERSION:
            logging.info("Ignoring source sample checkpoint with an old schema")
            return None
        if manifest.get("fingerprint") != fingerprint:
            logging.info("Source sample checkpoint inputs or parameters changed")
            return None
        raw_counters = manifest["counters"]
        raw_refs = manifest["selected_refs"]
        if not isinstance(raw_counters, dict) or not isinstance(raw_refs, list):
            raise ValueError("checkpoint counters or selected_refs has the wrong type")
        refs = [
            SelectedSourceTableRef(
                priority=int(item["priority"]),
                relative_path=str(item["relative_path"]),
                table_id=str(item["table_id"]),
            )
            for item in raw_refs
        ]
        if refs != sorted(refs):
            raise ValueError("checkpoint selected_refs are not sorted")
        processed_tables = int(raw_counters["processed_tables"])
        skipped_tables = int(raw_counters["skipped_tables"])
        skip_reasons = Counter(
            {
                str(reason): int(count)
                for reason, count in dict(raw_counters["skip_reasons"]).items()
            }
        )
        counters.processed_tables = processed_tables
        counters.skipped_tables = skipped_tables
        counters.skip_reasons = skip_reasons
        logging.info(
            "Reusing source sample checkpoint with %d selected tables: %s",
            len(refs),
            manifest_path,
        )
        return refs
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning(
            "Ignoring invalid source sample checkpoint %s: %s",
            manifest_path,
            exc,
        )
        return None


def _write_source_sample_manifest(
    manifest_path: Path,
    *,
    fingerprint: str,
    inventory_count: int,
    config: dict[str, Any],
    counters: SourceCandidateCounters,
    selected_refs: list[SelectedSourceTableRef],
) -> None:
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": SOURCE_SAMPLE_CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "inventory_file_count": inventory_count,
            "config": config,
            "counters": {
                "processed_tables": counters.processed_tables,
                "skipped_tables": counters.skipped_tables,
                "skip_reasons": dict(counters.skip_reasons),
            },
            "selected_refs": [asdict(ref) for ref in selected_refs],
        },
    )


def _read_source_sample_chunk(
    path: Path,
    refs: list[SelectedSourceTableRef],
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    tables: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line_number > len(refs):
                    raise ValueError("chunk contains extra records")
                ref = refs[line_number - 1]
                record = json.loads(line)
                if (
                    record.get("relative_path") != ref.relative_path
                    or record.get("table_id") != ref.table_id
                    or not isinstance(record.get("source_table"), dict)
                ):
                    raise ValueError(f"reference mismatch on line {line_number}")
                tables.append(record["source_table"])
        if len(tables) != len(refs):
            raise ValueError(
                f"chunk contains {len(tables)} records; expected {len(refs)}"
            )
        return tables
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("Ignoring invalid source sample chunk %s: %s", path, exc)
        return None


def _write_source_sample_chunk(
    path: Path,
    refs: list[SelectedSourceTableRef],
    tables: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for ref, source_table in zip(refs, tables, strict=True):
                handle.write(
                    json.dumps(
                        {
                            "relative_path": ref.relative_path,
                            "table_id": ref.table_id,
                            "source_table": source_table,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def iter_random_source_tables(
    input_dir: Path,
    args: argparse.Namespace,
    counters: SourceCandidateCounters,
) -> Iterator[dict[str, Any]]:
    query_rows_per_table = configured_query_rows_per_table(args)
    checkpoint_dir = _source_sample_checkpoint_dir(args)
    json_files = sorted(
        (
            path
            for path in input_dir.rglob("*.json")
            if checkpoint_dir is None or not _is_within(path, checkpoint_dir)
        ),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    max_scanned_files = getattr(args, "max_scanned_files", None)
    if max_scanned_files is not None:
        max_scanned_files = int(max_scanned_files)
        if max_scanned_files <= 0:
            raise ValueError("max scanned files must be positive or None")
        json_files = json_files[:max_scanned_files]
    sample_config = _source_sample_config(args)
    fingerprint = ""
    manifest_path: Path | None = None
    if checkpoint_dir is not None:
        inventory = _source_inventory(input_dir, json_files)
        fingerprint = _source_sample_fingerprint(inventory, sample_config)
        manifest_path = checkpoint_dir / "manifest.json"
    else:
        inventory = []
    capacity = (
        None
        if args.max_source_tables is None
        else args.max_source_tables * (args.unrecoverable_replacement_rounds + 1)
    )
    selected_heap: list[_DescendingSelectedSourceTableRef] = []
    selected_refs: list[SelectedSourceTableRef] | None = None
    if (
        manifest_path is not None
        and not bool(getattr(args, "refresh_source_sample_checkpoint", False))
    ):
        selected_refs = _restore_source_sample_refs(
            manifest_path,
            fingerprint,
            counters,
        )
    if selected_refs is None:
        counters.processed_tables = 0
        counters.skipped_tables = 0
        counters.skip_reasons = Counter()
        selected_refs = []
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
        if manifest_path is not None:
            _write_source_sample_manifest(
                manifest_path,
                fingerprint=fingerprint,
                inventory_count=len(inventory),
                config=sample_config,
                counters=counters,
                selected_refs=selected_refs,
            )
            logging.info(
                "Saved source sample checkpoint with %d selected tables: %s",
                len(selected_refs),
                manifest_path,
            )
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
            chunk_path = None
            materialized_tables = None
            if checkpoint_dir is not None:
                chunk_path = (
                    checkpoint_dir
                    / "chunks"
                    / fingerprint
                    / f"candidates_{chunk_start:08d}_{chunk_start + len(chunk):08d}.jsonl"
                )
                if not bool(
                    getattr(args, "refresh_source_sample_checkpoint", False)
                ):
                    materialized_tables = _read_source_sample_chunk(
                        chunk_path,
                        chunk,
                    )
            if materialized_tables is not None:
                logging.info(
                    "Reusing %d materialized source candidates from %s",
                    len(materialized_tables),
                    chunk_path,
                )
                if materialization_progress is not None:
                    materialization_progress.update(1)
                yield from materialized_tables
                continue
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
            materialized_tables = [
                materialized[(ref.relative_path, ref.table_id)] for ref in chunk
            ]
            if chunk_path is not None:
                _write_source_sample_chunk(chunk_path, chunk, materialized_tables)
            if materialization_progress is not None:
                materialization_progress.update(1)
            yield from materialized_tables
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
        "query_recovery_auto_checks": root_dir / "query_recovery_auto_checks.jsonl",
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
_STATION_ENTITY_COLUMN_RE = re.compile(r"(?:station|駅|站)", re.IGNORECASE)
_STATION_NAME_ATTRIBUTE_RE = re.compile(
    r"(?:name|japanese|english|kanji|kana|native|romanized|romaji|日本語|和名)",
    re.IGNORECASE,
)
_STATION_NAME_SUFFIXES = (" railway station", " station", "駅", "站")


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


def _station_name_without_type_suffix(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().strip()
    for suffix in _STATION_NAME_SUFFIXES:
        if text.endswith(suffix):
            stem = text[: -len(suffix)].strip()
            if len(stem.replace(" ", "")) >= 2:
                return stem
    return text


def _station_name_type_suffix_match(
    predicted: Any,
    expected: Any,
    *,
    attribute_name: str,
    entity_column_name: str,
) -> bool:
    """Allow only a station-type suffix difference in station-name fields."""
    if not _STATION_ENTITY_COLUMN_RE.search(clean_text(entity_column_name)):
        return False
    if not _STATION_NAME_ATTRIBUTE_RE.search(clean_text(attribute_name)):
        return False
    pred = unicodedata.normalize("NFKC", clean_text(predicted)).casefold().strip()
    exp = unicodedata.normalize("NFKC", clean_text(expected)).casefold().strip()
    if not pred or not exp or pred == exp:
        return False
    return (
        _station_name_without_type_suffix(pred)
        == _station_name_without_type_suffix(exp)
    )


def values_match(
    predicted: Any,
    expected: Any,
    *,
    attribute_name: str = "",
    entity_column_name: str = "",
) -> bool:
    pred = _match_text(predicted)
    exp = _match_text(expected)
    if not pred or not exp:
        return False
    if pred == exp:
        return True
    if _station_name_type_suffix_match(
        predicted,
        expected,
        attribute_name=attribute_name,
        entity_column_name=entity_column_name,
    ):
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


_QUOTED_ATTRIBUTES_KEY_RE = re.compile(r'''["']attributes["']\s*:''', re.I)
_UNQUOTED_ATTRIBUTES_KEY_RE = re.compile(r"\battributes\s*:", re.I)


@dataclass(frozen=True)
class JsonObjectParseResult:
    payload: dict[str, Any]
    method: str


def parse_json_object(
    text: str,
    *,
    allow_repair: bool = True,
) -> JsonObjectParseResult:
    text = clean_text(text)
    if not text:
        return JsonObjectParseResult({}, "empty")
    try:
        payload = json.loads(text)
        return JsonObjectParseResult(
            payload if isinstance(payload, dict) else {},
            "json" if isinstance(payload, dict) else "json_non_object",
        )
    except json.JSONDecodeError:
        pass
    objects = list(iter_json_objects(text))
    for payload in reversed(objects):
        if isinstance(payload.get("attributes"), list):
            return JsonObjectParseResult(payload, "embedded_json")
    repair_can_produce_attributes = bool(
        _QUOTED_ATTRIBUTES_KEY_RE.search(text)
        or ("{" in text and _UNQUOTED_ATTRIBUTES_KEY_RE.search(text))
    )
    if allow_repair and repair_can_produce_attributes:
        try:
            repaired = repair_json(
                text,
                return_objects=True,
                skip_json_loads=True,
            )
        except Exception:
            repaired = None
        if isinstance(repaired, dict):
            return JsonObjectParseResult(repaired, "json_repair")
    if objects:
        return JsonObjectParseResult(objects[-1], "embedded_json_fallback")
    return JsonObjectParseResult({}, "invalid")


def safe_json_object(text: str) -> dict[str, Any]:
    return parse_json_object(text).payload


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

    def __init__(
        self,
        message: str,
        *,
        model_endpoint: str = "",
        model_kind: str = "",
        failure_type: str = "",
    ) -> None:
        super().__init__(message)
        self.model_endpoint = clean_text(model_endpoint)
        self.model_kind = clean_text(model_kind)
        self.failure_type = clean_text(failure_type)


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


class ModelAutoCheckStats:
    """Thread-safe counters for the post-analysis extraction gate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()

    def record(
        self,
        verdict: str,
        *,
        error: bool = False,
        luna_triggered: bool = False,
        luna_verdict: str | None = None,
        terra_triggered: bool = False,
        terra_verdict: str | None = None,
        decision_source: str = "",
    ) -> None:
        with self._lock:
            self._counts["reviewed"] += 1
            self._counts[verdict] += 1
            if verdict != "supported":
                self._counts["filtered"] += 1
            if error:
                self._counts["errors"] += 1
            if luna_triggered:
                self._counts["luna_triggered"] += 1
            if luna_verdict in {"supported", "contradicted", "insufficient"}:
                self._counts["luna_completed"] += 1
                self._counts[f"luna_{luna_verdict}"] += 1
            if terra_triggered:
                self._counts["terra_triggered"] += 1
                self._counts["final_judge_triggered"] += 1
            if terra_verdict in {"supported", "contradicted", "insufficient"}:
                self._counts["terra_completed"] += 1
                self._counts[f"terra_{terra_verdict}"] += 1
                self._counts["final_judge_completed"] += 1
                self._counts[f"final_judge_{terra_verdict}"] += 1

    def summary(self) -> dict[str, int | str | bool]:
        with self._lock:
            counts = dict(self._counts)
        return {
            "enabled": True,
            "schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "reviewed": int(counts.get("reviewed", 0)),
            "supported": int(counts.get("supported", 0)),
            "contradicted": int(counts.get("contradicted", 0)),
            "insufficient": int(counts.get("insufficient", 0)),
            "filtered": int(counts.get("filtered", 0)),
            "errors": int(counts.get("errors", 0)),
            "luna_triggered": int(counts.get("luna_triggered", 0)),
            "luna_completed": int(counts.get("luna_completed", 0)),
            "luna_supported": int(counts.get("luna_supported", 0)),
            "terra_triggered": int(counts.get("terra_triggered", 0)),
            "terra_completed": int(counts.get("terra_completed", 0)),
            "terra_supported": int(counts.get("terra_supported", 0)),
            "final_judge_triggered": int(
                counts.get("final_judge_triggered", 0)
            ),
            "final_judge_completed": int(
                counts.get("final_judge_completed", 0)
            ),
            "final_judge_supported": int(
                counts.get("final_judge_supported", 0)
            ),
            # Compatibility counters: the secondary stage is Luna in v2.
            "secondary_triggered": int(counts.get("luna_triggered", 0)),
            "secondary_completed": int(counts.get("luna_completed", 0)),
            "secondary_supported": int(counts.get("luna_supported", 0)),
        }


def model_auto_check_summary(extractor: Any | None) -> dict[str, Any]:
    stats = getattr(extractor, "model_auto_check_stats", None)
    if stats is None or not hasattr(stats, "summary"):
        return {
            "enabled": bool(getattr(extractor, "auto_check_enabled", False)),
            "schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
            "reviewed": 0,
            "supported": 0,
            "contradicted": 0,
            "insufficient": 0,
            "filtered": 0,
            "errors": 0,
            "luna_triggered": 0,
            "luna_completed": 0,
            "luna_supported": 0,
            "terra_triggered": 0,
            "terra_completed": 0,
            "terra_supported": 0,
            "final_judge_triggered": 0,
            "final_judge_completed": 0,
            "final_judge_supported": 0,
            "secondary_triggered": 0,
            "secondary_completed": 0,
            "secondary_supported": 0,
        }
    return dict(stats.summary())


def summarize_model_auto_check_records(
    paths: Iterable[Path],
) -> dict[str, Any]:
    """Summarize durable checker decisions, including cache-reused records."""
    counts: Counter[str] = Counter()
    for record in iter_jsonl_records(paths):
        if clean_text(record.get("error")):
            counts["model_error_records"] += 1
            continue
        auto_check = record.get("auto_check")
        if not isinstance(auto_check, dict):
            counts["unchecked_records"] += 1
            continue
        counts["checked_records"] += 1
        reviews = auto_check.get("reviews")
        if not isinstance(reviews, list):
            counts["invalid_check_records"] += 1
            continue
        for review in reviews:
            if not isinstance(review, dict):
                counts["invalid_check_reviews"] += 1
                continue
            verdict = clean_text(review.get("verdict"))
            if verdict not in {"supported", "contradicted", "insufficient"}:
                counts["invalid_check_reviews"] += 1
                continue
            counts["reviewed"] += 1
            counts[verdict] += 1
            if verdict != "supported":
                counts["filtered"] += 1
            if clean_text(review.get("error_code")):
                counts["errors"] += 1
            if bool(review.get("luna_triggered")):
                counts["luna_triggered"] += 1
            luna_verdict = clean_text(review.get("luna_verdict"))
            if luna_verdict in {"supported", "contradicted", "insufficient"}:
                counts["luna_completed"] += 1
                if luna_verdict == "supported":
                    counts["luna_supported"] += 1
            if bool(review.get("terra_triggered")):
                counts["terra_triggered"] += 1
            if bool(
                review.get("final_judge_triggered")
                or review.get("terra_triggered")
            ):
                counts["final_judge_triggered"] += 1
            terra_verdict = clean_text(review.get("terra_verdict"))
            if terra_verdict in {"supported", "contradicted", "insufficient"}:
                counts["terra_completed"] += 1
                if terra_verdict == "supported":
                    counts["terra_supported"] += 1
                counts["final_judge_completed"] += 1
                if terra_verdict == "supported":
                    counts["final_judge_supported"] += 1
    return {
        "enabled": True,
        "schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
        "checked_records": int(counts.get("checked_records", 0)),
        "model_error_records": int(counts.get("model_error_records", 0)),
        "unchecked_records": int(counts.get("unchecked_records", 0)),
        "invalid_check_records": int(counts.get("invalid_check_records", 0)),
        "invalid_check_reviews": int(counts.get("invalid_check_reviews", 0)),
        "reviewed": int(counts.get("reviewed", 0)),
        "supported": int(counts.get("supported", 0)),
        "contradicted": int(counts.get("contradicted", 0)),
        "insufficient": int(counts.get("insufficient", 0)),
        "filtered": int(counts.get("filtered", 0)),
        "errors": int(counts.get("errors", 0)),
        "luna_triggered": int(counts.get("luna_triggered", 0)),
        "luna_completed": int(counts.get("luna_completed", 0)),
        "luna_supported": int(counts.get("luna_supported", 0)),
        "terra_triggered": int(counts.get("terra_triggered", 0)),
        "terra_completed": int(counts.get("terra_completed", 0)),
        "terra_supported": int(counts.get("terra_supported", 0)),
        "final_judge_triggered": int(
            counts.get("final_judge_triggered", 0)
        ),
        "final_judge_completed": int(
            counts.get("final_judge_completed", 0)
        ),
        "final_judge_supported": int(
            counts.get("final_judge_supported", 0)
        ),
        "secondary_triggered": int(counts.get("luna_triggered", 0)),
        "secondary_completed": int(counts.get("luna_completed", 0)),
        "secondary_supported": int(counts.get("luna_supported", 0)),
    }


def add_model_auto_check_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared local -> Luna -> Terra checker settings."""
    group = parser.add_argument_group("post-analysis auto checker")
    group.add_argument(
        "--no_auto_check_secondary_openai",
        dest="auto_check_secondary_openai",
        action="store_false",
        help=(
            "Use the local auto-check model only. By default Luna reviews every "
            "local extraction and Terra adjudicates local/Luna disagreements."
        ),
    )
    group.add_argument(
        "--auto_check_api_config_file",
        default="",
        help=(
            "Chmod-600 JSON file containing named initial/final auto-check "
            "API profiles. If omitted, ./.auto_check_apis.json is loaded "
            "when present. Valid changes are reloaded while the builder runs; "
            "invalid updates warn and leave the last valid profiles active."
        ),
    )
    group.add_argument(
        "--auto_check_openai_env_file",
        default="",
        help=(
            "Legacy chmod-600 dotenv path for the auto checker. If neither "
            "JSON nor dotenv is selected, ./.env.openai is loaded when "
            "present."
        ),
    )
    group.add_argument(
        "--auto_check_luna_model",
        "--auto_check_openai_model",
        dest="auto_check_openai_model",
        default=DEFAULT_AUTO_CHECK_LUNA_MODEL,
    )
    group.add_argument("--auto_check_openai_base_url", default="")
    group.add_argument(
        "--auto_check_openai_api_key_env",
        default="OPENAI_API_KEY",
    )
    group.add_argument(
        "--auto_check_luna_reasoning_effort",
        "--auto_check_openai_reasoning_effort",
        dest="auto_check_openai_reasoning_effort",
        choices=("omit", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    group.add_argument(
        "--auto_check_openai_verbosity",
        choices=("low", "medium", "high"),
        default="low",
    )
    group.add_argument(
        "--auto_check_openai_max_output_tokens",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--auto_check_terra_model",
        default=DEFAULT_AUTO_CHECK_TERRA_MODEL,
    )
    group.add_argument(
        "--auto_check_terra_reasoning_effort",
        choices=("omit", "none", "minimal", "low", "medium", "high", "xhigh"),
        default="none",
    )
    group.add_argument(
        "--auto_check_terra_max_output_tokens",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--auto_check_openai_max_inflight",
        type=int,
        default=MAX_AUTO_CHECK_OPENAI_CONCURRENCY,
    )
    group.add_argument(
        "--auto_check_openai_image_detail",
        choices=("auto", "low", "high"),
        default="auto",
    )
    group.add_argument(
        "--auto_check_openai_image_max_pixels",
        type=int,
        default=DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
    )
    group.add_argument(
        "--auto_check_openai_timeout_seconds",
        type=float,
        default=180.0,
    )
    group.add_argument(
        "--auto_check_openai_requests_per_minute",
        type=int,
        default=0,
    )
    group.add_argument(
        "--auto_check_openai_tokens_per_minute",
        type=int,
        default=0,
    )
    parser.set_defaults(auto_check_secondary_openai=True)


def prepare_model_auto_check_reviewers(
    args: argparse.Namespace,
) -> tuple[Any | None, Any | None]:
    """Create Luna consensus-review and final-judge reviewer pools."""
    if not bool(getattr(args, "auto_check_secondary_openai", False)):
        return None, None

    from mm_joinability_dataset_auto_checker import (
        DEFAULT_OPENAI_ENV_FILE,
        prepare_reviewer,
    )
    from openai_attribute_extractor import (
        OpenAIRequestController,
        load_openai_auto_check_api_config,
        load_openai_compatible_api_profiles,
        load_openai_environment_file,
    )

    configured_api_config_path = clean_text(
        getattr(args, "auto_check_api_config_file", "")
    )
    configured_env_path = clean_text(
        getattr(args, "auto_check_openai_env_file", "")
    )
    if configured_api_config_path and configured_env_path:
        raise ValueError(
            "--auto_check_api_config_file and --auto_check_openai_env_file "
            "are mutually exclusive"
        )
    api_config_path = (
        Path(configured_api_config_path)
        if configured_api_config_path
        else DEFAULT_AUTO_CHECK_API_CONFIG_FILE
    )
    use_api_config = bool(
        configured_api_config_path
        or not configured_env_path and api_config_path.exists()
    )
    if use_api_config:
        api_config_path = api_config_path.resolve()
        initial_api_config_signature = AutoCheckAPIConfigReloader.file_signature(
            api_config_path
        )
        profiles = load_openai_auto_check_api_config(
            api_config_path,
            default_max_concurrency=(
                DEFAULT_AUTO_CHECK_PROFILE_MAX_CONCURRENCY
            ),
        )
    else:
        environment_path = (
            Path(configured_env_path)
            if configured_env_path
            else DEFAULT_OPENAI_ENV_FILE
        )
        if configured_env_path or environment_path.exists():
            load_openai_environment_file(environment_path)
        profiles = load_openai_compatible_api_profiles(
            default_model=clean_text(
                getattr(
                    args,
                    "auto_check_openai_model",
                    DEFAULT_AUTO_CHECK_LUNA_MODEL,
                )
            ),
            default_max_concurrency=DEFAULT_AUTO_CHECK_PROFILE_MAX_CONCURRENCY,
        )

    max_inflight = int(
        getattr(
            args,
            "auto_check_openai_max_inflight",
            MAX_AUTO_CHECK_OPENAI_CONCURRENCY,
        )
    )
    if not 1 <= max_inflight <= MAX_AUTO_CHECK_OPENAI_CONCURRENCY:
        raise ValueError(
            "--auto_check_openai_max_inflight must be between 1 and "
            f"{MAX_AUTO_CHECK_OPENAI_CONCURRENCY}"
        )
    output_dir = Path(getattr(args, "output_dir", ".")).resolve()
    api_base_url = clean_text(
        getattr(args, "auto_check_openai_base_url", "")
    ) or os.environ.get("OPENAI_BASE_URL", DEFAULT_AUTO_CHECK_OPENAI_BASE_URL)

    def make_reviewer(
        *,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        role: str,
        profile_name: str = "legacy",
        api_key: str = "",
        api_base_url_override: str = "",
        profile_max_concurrency: int | None = None,
        request_controller: Any | None = None,
        use_responses: bool = False,
    ) -> Any:
        model = clean_text(model)
        if not model:
            raise ValueError(f"--auto_check_{role}_model must not be empty")
        if max_output_tokens <= 0:
            raise ValueError(
                f"--auto_check_{role}_max_output_tokens must be positive"
            )
        model_component = re.sub(
            r"[^A-Za-z0-9._-]+",
            "-",
            model,
        ).strip("-._") or role
        usage_path = output_dir / "auto_checker_usage" / (
            f"{profile_name}-{role}-{model_component[:64]}.jsonl"
        )
        reviewer_max_inflight = (
            max_inflight
            if profile_max_concurrency is None
            else int(profile_max_concurrency)
        )
        reviewer_args = argparse.Namespace(
            provider="openai",
            openai_profile_name=profile_name,
            openai_model=model,
            openai_base_url=api_base_url_override or api_base_url,
            openai_api_key=api_key,
            openai_api_key_env=clean_text(
                getattr(args, "auto_check_openai_api_key_env", "OPENAI_API_KEY")
            ),
            openai_reasoning_effort=reasoning_effort,
            openai_verbosity=getattr(
                args,
                "auto_check_openai_verbosity",
                "low",
            ),
            openai_max_output_tokens=max_output_tokens,
            openai_image_detail=getattr(
                args,
                "auto_check_openai_image_detail",
                "auto",
            ),
            openai_image_max_pixels=int(
                getattr(
                    args,
                    "auto_check_openai_image_max_pixels",
                    DEFAULT_IMAGE_REQUEST_MAX_PIXELS,
                )
            ),
            openai_max_inflight=reviewer_max_inflight,
            openai_adaptive_concurrency=request_controller is not None,
            openai_initial_inflight=(
                AUTO_CHECK_PROFILE_INITIAL_CONCURRENCY
                if request_controller is not None
                else None
            ),
            openai_successes_per_increase=(
                AUTO_CHECK_PROFILE_SUCCESSES_PER_INCREASE
            ),
            openai_request_controller=request_controller,
            openai_use_responses=use_responses,
            openai_requests_per_minute=int(
                getattr(args, "auto_check_openai_requests_per_minute", 0)
            ),
            openai_tokens_per_minute=int(
                getattr(args, "auto_check_openai_tokens_per_minute", 0)
            ),
            model_timeout_seconds=float(
                getattr(args, "auto_check_openai_timeout_seconds", 180.0)
            ),
            # Profile pools retry across providers; retrying the same failing
            # endpoint here would hold a workflow worker for several timeouts.
            model_max_retries=(
                0
                if request_controller is not None
                else int(getattr(args, "model_max_retries", 2))
            ),
            model_retry_sleep_seconds=float(
                getattr(args, "model_retry_sleep_seconds", 2.0)
            ),
            openai_retry_max_seconds=60.0,
        )
        return prepare_reviewer(
            reviewer_args,
            usage_journal_path=usage_path,
            ensure_ready=False,
        )

    if profiles:
        active_profiles: dict[str, Any] = {}
        active_controllers: dict[str, Any] = {}
        active_initial_reviewers: dict[str, Any] = {}
        active_final_reviewers: dict[str, Any] = {}

        def build_profile_snapshot(
            configured_profiles: list[Any],
        ) -> tuple[
            dict[str, Any],
            dict[str, Any],
            list[tuple[str, Any]],
            list[tuple[str, Any]],
        ]:
            profiles_by_name: dict[str, Any] = {}
            profile_controllers: dict[str, Any] = {}
            initial_reviewers: list[tuple[str, Any]] = []
            final_reviewers: list[tuple[str, Any]] = []
            for profile in configured_profiles:
                profiles_by_name[profile.name] = profile
                unchanged = active_profiles.get(profile.name) == profile
                controller = (
                    active_controllers[profile.name]
                    if unchanged
                    else OpenAIRequestController(
                        max_inflight=profile.max_concurrency,
                        adaptive=True,
                        initial_inflight=AUTO_CHECK_PROFILE_INITIAL_CONCURRENCY,
                        successes_per_increase=(
                            AUTO_CHECK_PROFILE_SUCCESSES_PER_INCREASE
                        ),
                        requests_per_minute=int(
                            getattr(
                                args,
                                "auto_check_openai_requests_per_minute",
                                0,
                            )
                        ),
                        tokens_per_minute=int(
                            getattr(
                                args,
                                "auto_check_openai_tokens_per_minute",
                                0,
                            )
                        ),
                    )
                )
                profile_controllers[profile.name] = controller
                if profile.model is not None:
                    reviewer = (
                        active_initial_reviewers[profile.name]
                        if unchanged
                        else make_reviewer(
                            model=profile.model,
                            reasoning_effort=getattr(
                                args,
                                "auto_check_openai_reasoning_effort",
                                "medium",
                            ),
                            max_output_tokens=int(
                                getattr(
                                    args,
                                    "auto_check_openai_max_output_tokens",
                                    2048,
                                )
                            ),
                            role="initial",
                            profile_name=profile.name,
                            api_key=profile.initial_api_key or "",
                            api_base_url_override=(
                                profile.initial_api_base_url or ""
                            ),
                            profile_max_concurrency=profile.max_concurrency,
                            request_controller=controller,
                            use_responses=profile.initial_use_responses,
                        )
                    )
                    initial_reviewers.append((profile.name, reviewer))
                if profile.final_judge_model is not None:
                    reviewer = (
                        active_final_reviewers[profile.name]
                        if unchanged
                        else make_reviewer(
                            model=profile.final_judge_model,
                            reasoning_effort=getattr(
                                args,
                                "auto_check_terra_reasoning_effort",
                                "medium",
                            ),
                            max_output_tokens=int(
                                getattr(
                                    args,
                                    "auto_check_terra_max_output_tokens",
                                    2048,
                                )
                            ),
                            role="final",
                            profile_name=profile.name,
                            api_key=profile.final_judge_api_key or "",
                            api_base_url_override=(
                                profile.final_judge_api_base_url or ""
                            ),
                            profile_max_concurrency=profile.max_concurrency,
                            request_controller=controller,
                            use_responses=profile.final_judge_use_responses,
                        )
                    )
                    final_reviewers.append((profile.name, reviewer))
            if not initial_reviewers:
                raise ValueError(
                    "auto-check API configuration must include at least one "
                    "initial reviewer"
                )
            return (
                profiles_by_name,
                profile_controllers,
                initial_reviewers,
                final_reviewers,
            )

        (
            active_profiles,
            active_controllers,
            initial_reviewers,
            final_reviewers,
        ) = build_profile_snapshot(profiles)
        active_initial_reviewers = dict(initial_reviewers)
        active_final_reviewers = dict(final_reviewers)
        load_balancer = AutoCheckProviderLoadBalancer(active_controllers)
        initial_pool = BalancedAutoCheckReviewerPool(
            initial_reviewers,
            role="initial",
            load_balancer=load_balancer,
        )
        dynamic_final_pool = BalancedAutoCheckReviewerPool(
            final_reviewers,
            role="final",
            load_balancer=load_balancer,
            allow_empty=True,
        )
        # The builder can discover a final-only provider that is added after a
        # run started even when the initial configuration had no final stage.
        initial_pool.companion_final_pool = dynamic_final_pool

        if use_api_config:
            def reload_profile_config() -> tuple[int, int, int]:
                nonlocal active_profiles
                nonlocal active_controllers
                nonlocal active_initial_reviewers
                nonlocal active_final_reviewers

                updated_profiles = load_openai_auto_check_api_config(
                    api_config_path,
                    default_max_concurrency=(
                        DEFAULT_AUTO_CHECK_PROFILE_MAX_CONCURRENCY
                    ),
                )
                (
                    profiles_by_name,
                    controllers,
                    updated_initial_reviewers,
                    updated_final_reviewers,
                ) = build_profile_snapshot(updated_profiles)
                load_balancer.replace_controllers(controllers)
                initial_pool.replace_reviewers(updated_initial_reviewers)
                dynamic_final_pool.replace_reviewers(updated_final_reviewers)
                active_profiles = profiles_by_name
                active_controllers = controllers
                active_initial_reviewers = dict(updated_initial_reviewers)
                active_final_reviewers = dict(updated_final_reviewers)
                return (
                    len(updated_initial_reviewers),
                    len(updated_final_reviewers),
                    len(profiles_by_name),
                )

            reloader = AutoCheckAPIConfigReloader(
                api_config_path,
                reload_config=reload_profile_config,
                active_profile_count=lambda: len(active_profiles),
                initial_signature=initial_api_config_signature,
            )
            initial_pool.set_reload_callback(reloader.reload_if_changed)
            dynamic_final_pool.set_reload_callback(reloader.reload_if_changed)

        return (
            initial_pool,
            dynamic_final_pool if final_reviewers else None,
        )

    luna = make_reviewer(
        model=getattr(
            args,
            "auto_check_openai_model",
            DEFAULT_AUTO_CHECK_LUNA_MODEL,
        ),
        reasoning_effort=getattr(
            args,
            "auto_check_openai_reasoning_effort",
            "low",
        ),
        max_output_tokens=int(
            getattr(args, "auto_check_openai_max_output_tokens", 2048)
        ),
        role="luna",
    )
    terra = make_reviewer(
        model=getattr(
            args,
            "auto_check_terra_model",
            DEFAULT_AUTO_CHECK_TERRA_MODEL,
        ),
        reasoning_effort=getattr(
            args,
            "auto_check_terra_reasoning_effort",
            "medium",
        ),
        max_output_tokens=int(
            getattr(args, "auto_check_terra_max_output_tokens", 2048)
        ),
        role="terra",
    )
    return luna, terra


def prepare_model_auto_check_secondary(args: argparse.Namespace) -> Any | None:
    """Backward-compatible helper returning the Luna recovery reviewer."""
    luna, _terra = prepare_model_auto_check_reviewers(args)
    return luna


class LocalAttributeExtractor:
    """OpenAI-compatible client for local text and image extraction models."""

    def __init__(self, args: argparse.Namespace) -> None:
        if requests is None:
            raise RuntimeError("requests is required for local model calls")
        endpoint_config_path = clean_text(
            getattr(args, "model_endpoint_config", "")
        )
        self.model_endpoint_config_path = endpoint_config_path
        self.model_endpoint_scheduler = None
        if endpoint_config_path:
            endpoint_config = load_model_endpoint_config(
                Path(endpoint_config_path).resolve()
            )
            self.model_endpoint_scheduler = ModelEndpointScheduler(
                endpoint_config
            )
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
        if self.model_endpoint_scheduler is not None and (
            self.routing_scheduler is not None
            or self.remote_routing_scheduler is not None
        ):
            raise ValueError(
                "--model_endpoint_config cannot be combined with dynamic "
                "routing manifests"
            )
        if self.model_endpoint_scheduler is not None:
            configured_text_urls = self.model_endpoint_scheduler.urls(
                "local", "text"
            )
        else:
            configured_text_urls = normalize_model_base_urls(
                getattr(args, "text_model_base_urls", None)
            )
            fallback_text_url = clean_text(
                getattr(args, "text_model_base_url", "")
            ).rstrip("/")
            if fallback_text_url:
                configured_text_urls = normalize_model_base_urls(
                    [fallback_text_url, *configured_text_urls]
                )
        if not configured_text_urls and self.routing_scheduler is None:
            raise ValueError("at least one text model base URL is required")
        self.text_model_base_urls = configured_text_urls
        self.text_model_base_urls_file = clean_text(getattr(args, "text_model_base_urls_file", ""))
        self._text_endpoint_lock = threading.Lock()
        self._text_endpoint_index = 0
        self._text_endpoint_inflight: dict[str, int] = {}
        self.text_model_name = (
            self.model_endpoint_scheduler.config.served_model_name
            if self.model_endpoint_scheduler is not None
            else args.text_model_name
        )
        self.text_model_api_key = model_api_key(
            getattr(args, "text_model_api_key", None),
            "MMDD_TEXT_MODEL_API_KEY",
        )
        if self.model_endpoint_scheduler is not None:
            configured_remote_text_urls = self.model_endpoint_scheduler.urls(
                "remote", "text"
            )
        else:
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
        if self.model_endpoint_scheduler is not None:
            configured_image_urls = self.model_endpoint_scheduler.urls(
                "local", "image"
            )
        else:
            configured_image_urls = normalize_model_base_urls(
                getattr(args, "image_model_base_urls", None)
            )
            fallback_image_url = clean_text(
                getattr(args, "image_model_base_url", "")
            ).rstrip("/")
            if fallback_image_url:
                configured_image_urls = normalize_model_base_urls(
                    [fallback_image_url, *configured_image_urls]
                )
        if not configured_image_urls and self.routing_scheduler is None:
            raise ValueError("at least one image model base URL is required")
        self.image_model_base_urls = configured_image_urls
        self.image_model_base_urls_file = clean_text(getattr(args, "image_model_base_urls_file", ""))
        self._image_endpoint_lock = threading.Lock()
        self._image_endpoint_index = 0
        self._image_endpoint_inflight: dict[str, int] = {}
        self.image_model_name = (
            self.model_endpoint_scheduler.config.served_model_name
            if self.model_endpoint_scheduler is not None
            else args.image_model_name
        )
        self.image_model_api_key = model_api_key(
            getattr(args, "image_model_api_key", None),
            "MMDD_IMAGE_MODEL_API_KEY",
        )
        if self.model_endpoint_scheduler is not None:
            configured_remote_image_urls = self.model_endpoint_scheduler.urls(
                "remote", "image"
            )
        else:
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
        if not bool(getattr(args, "disable_thinking", True)):
            raise ValueError(
                "thinking mode is disabled for every text and image model request"
            )
        self.disable_thinking = True
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
        # Production builders always apply the independent, physically masked
        # checker after the batched analysis call. Lightweight injected test
        # extractors opt in explicitly by exposing the same attribute/method.
        self.auto_check_enabled = True
        self.model_auto_check_stats = ModelAutoCheckStats()
        (
            self.auto_check_luna_reviewer,
            self.auto_check_terra_reviewer,
        ) = prepare_model_auto_check_reviewers(args)
        self.auto_check_parallelism = max(
            int(
                getattr(
                    self.auto_check_luna_reviewer,
                    "max_parallelism",
                    getattr(
                        args,
                        "auto_check_openai_max_inflight",
                        MAX_AUTO_CHECK_OPENAI_CONCURRENCY,
                    ),
                )
            ),
            int(
                getattr(
                    self.auto_check_terra_reviewer,
                    "max_parallelism",
                    1,
                )
            ),
        )
        # Compatibility alias: the v2 secondary stage is Luna.
        self.auto_check_secondary_reviewer = self.auto_check_luna_reviewer

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

        if self.model_endpoint_scheduler is not None:
            _lock, urls_getter, _inflight, _index_name = (
                self._endpoint_pool_state(model_kind, endpoint_pool)
            )
            try:
                with self.model_endpoint_scheduler.lease(
                    endpoint_pool,
                    model_kind,
                    available_urls_getter=urls_getter,
                ) as endpoint:
                    yield endpoint.base_url
            except EndpointPoolUnavailableError as error:
                raise TransientModelEndpointError(str(error)) from None
            return

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
        if self.model_endpoint_scheduler is not None:
            _lock, urls_getter, _inflight, _index_name = (
                self._endpoint_pool_state(model_kind, endpoint_pool)
            )
            configured = {
                endpoint.base_url
                for endpoint in self.model_endpoint_scheduler.endpoints(
                    endpoint_pool, model_kind
                )
            }
            return base_url in configured and base_url in urls_getter()
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
        if self.model_endpoint_scheduler is not None:
            return self.model_endpoint_scheduler.capacity("local", model_kind)
        if self.routing_scheduler is None:
            return None
        return self.routing_scheduler.capacity(model_kind)

    def wait_for_endpoint_pool(
        self,
        model_kind: str,
        endpoint_pool: str,
        timeout_seconds: float,
    ) -> bool:
        if self.model_endpoint_scheduler is not None:
            _lock, urls_getter, _inflight, _index_name = (
                self._endpoint_pool_state(model_kind, endpoint_pool)
            )
            return self.model_endpoint_scheduler.wait_for_capacity(
                endpoint_pool,
                model_kind,
                timeout_seconds,
                available_urls_getter=urls_getter,
            )
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
        response: requests.Response | None = None
        try:
            try:
                response = requests.get(
                    models_url,
                    headers=headers,
                    timeout=request_timeout,
                )
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
        finally:
            close_response = getattr(response, "close", None)
            if callable(close_response):
                close_response()

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
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
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
        # This is deliberately unconditional: both text and visual Qwen calls
        # must use their direct-answer path on every endpoint.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        response_schema_name
                        or "mm_joinability_structured_extraction"
                    ),
                    "strict": True,
                    "schema": response_schema,
                },
            }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            response: requests.Response | None = None
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
                        raise TransientModelEndpointError(
                            error_message,
                            model_endpoint=base_url,
                            model_kind=model_kind,
                            failure_type=f"HTTP{status_code}",
                        )
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
                        f"({type(exc).__name__})",
                        model_endpoint=base_url,
                        model_kind=model_kind,
                        failure_type=type(exc).__name__,
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
            finally:
                close_response = getattr(response, "close", None)
                if callable(close_response):
                    close_response()
        message = f"Local {model_kind} model call to {base_url} failed: {last_error}"
        if isinstance(last_error, TransientModelEndpointError):
            raise TransientModelEndpointError(
                message,
                model_endpoint=last_error.model_endpoint or base_url,
                model_kind=last_error.model_kind or model_kind,
                failure_type=(
                    last_error.failure_type
                    or type(last_error).__name__
                ),
            ) from last_error
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

    def _auto_check_review_batch(
        self,
        *,
        task: ExtractionTask,
        attribute_name: str,
        claimed_value: str,
    ) -> dict[str, Any]:
        target_name = normalize(attribute_name)
        query_row_attributes = canonical_extraction_row_attributes(
            task.entity.get("row_attributes")
        )
        attribute_was_present = any(
            normalize(item.get("name")) == target_name
            for item in query_row_attributes
        )
        masked_row = [
            dict(item)
            for item in query_row_attributes
            if normalize(item.get("name")) != target_name
        ]
        review_id = "model_check_" + stable_hash(
            task.cache_key,
            attribute_name,
            claimed_value,
            length=20,
        )
        return {
            "query_table_id": (
                f"model_check:{task.source_table_id}:{task.source_row_id}"
            ),
            "target_table_id": "",
            "source_table_id": task.source_table_id,
            "split": "model_auto_check",
            "query_row_ids": [str(task.source_row_id)],
            "items": [
                {
                    "review_id": review_id,
                    "recovery_id": "",
                    "path_id": "",
                    "query_row_id": str(task.source_row_id),
                    "masked_row": masked_row,
                    "masked_attribute_was_present": attribute_was_present,
                    "attribute": {
                        "name": attribute_name,
                        # review_messages intentionally omits this value.
                        "value": claimed_value,
                    },
                    "evidence": {
                        "asset_id": clean_text(task.asset.get("asset_id")),
                        "asset_type": clean_text(task.asset.get("asset_type")),
                        "content": clean_text(task.asset.get("content"))[:6000],
                        "image_sha256": clean_text(task.asset.get("sha256")),
                    },
                    "image_path": clean_text(task.asset.get("local_path")),
                }
            ],
        }

    def extract_auto_check_value(
        self,
        *,
        task: ExtractionTask,
        attribute_name: str,
        claimed_value: str,
        endpoint_pool: str = "local",
    ) -> str:
        """Blindly re-extract one physically masked attribute from one asset."""
        # Import lazily because the standalone checker imports this builder for
        # normalization and image helpers.
        from mm_joinability_dataset_auto_checker import (
            AUTO_CHECK_EXTRACTION_SCHEMA,
            parse_model_extractions,
            review_messages,
        )

        batch = self._auto_check_review_batch(
            task=task,
            attribute_name=attribute_name,
            claimed_value=claimed_value,
        )
        model_kind = model_kind_for_asset(task.asset)
        api_key = (
            self.remote_image_model_api_key
            if endpoint_pool == "remote" and model_kind == "image"
            else self.remote_text_model_api_key
            if endpoint_pool == "remote"
            else self.image_model_api_key
            if model_kind == "image"
            else self.text_model_api_key
        )
        model_name = (
            self.image_model_name if model_kind == "image" else self.text_model_name
        )
        messages = review_messages(
            [batch],
            image_max_pixels=self.image_request_max_pixels,
        )
        with self._lease_endpoint_pool(model_kind, endpoint_pool) as base_url:
            raw = self.chat(
                base_url=base_url,
                model=model_name,
                api_key=api_key,
                messages=messages,
                model_kind=model_kind,
                response_schema=AUTO_CHECK_EXTRACTION_SCHEMA,
                response_schema_name=(
                    "mm_joinability_single_attribute_extraction"
                ),
            )
        extraction = parse_model_extractions(raw, [batch])[batch["query_table_id"]]
        if len(extraction) != 1:
            raise ValueError("auto checker returned an invalid extraction count")
        return clean_text(extraction[0].get("extracted_value"))

    def review_auto_check_attribute(
        self,
        *,
        task: ExtractionTask,
        attribute_name: str,
        claimed_value: str,
        endpoint_pool: str = "local",
        defer_remote: bool = False,
        local_review: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run local -> Luna consensus -> Terra adjudication, fail closed."""
        from mm_joinability_dataset_auto_checker import _safe_error_code

        def classify(value: str) -> tuple[str, str]:
            if not value:
                return "insufficient", "empty_extraction"
            if values_match(
                value,
                claimed_value,
                attribute_name=attribute_name,
                entity_column_name=task.entity_column_name,
            ):
                return "supported", "normalized_values_match"
            return "contradicted", "extracted_value_mismatch"

        def results_agree(left_value: str, right_value: str) -> bool:
            left_value = clean_text(left_value)
            right_value = clean_text(right_value)
            if not left_value or not right_value:
                return not left_value and not right_value
            return values_match(
                left_value,
                right_value,
                attribute_name=attribute_name,
                entity_column_name=task.entity_column_name,
            )

        def extract_with(
            reviewer: Any,
            batch: dict[str, Any],
            *,
            on_selected: Callable[[dict[str, Any]], None] | None = None,
        ) -> str:
            extract_batch = getattr(reviewer, "extract_batch", None)
            if callable(extract_batch):
                extracted = extract_batch(batch, on_selected=on_selected)
            else:
                extracted = reviewer.extract_batches([batch]).get(
                    batch["query_table_id"]
                )
            if not isinstance(extracted, list) or len(extracted) != 1:
                raise ValueError("auto checker returned an invalid extraction count")
            return clean_text(extracted[0].get("extracted_value"))

        state: dict[str, Any] = {
            "primary_extracted_value": "",
            "primary_verdict": None,
            "primary_comparison": "auto_check_failed",
            "primary_error_code": "",
            "luna_triggered": False,
            "luna_extracted_value": None,
            "luna_verdict": None,
            "luna_comparison": None,
            "luna_agrees_with_local": None,
            "luna_error_code": "",
            "terra_triggered": False,
            "terra_extracted_value": None,
            "terra_verdict": None,
            "terra_comparison": None,
            "terra_error_code": "",
            "initial_reviewer_profile": None,
            "initial_reviewer_model": None,
            "final_judge_profile": None,
            "final_judge_model": None,
            "final_judge_triggered": False,
        }

        def finish(
            *,
            value: str,
            verdict: str,
            comparison: str,
            source: str,
            complete: bool,
            error_code: str = "",
        ) -> dict[str, Any]:
            return {
                "extracted_value": value,
                "verdict": verdict,
                "comparison": comparison,
                "decision_source": source,
                "review_complete": complete,
                "error_code": error_code,
                **state,
                # Backward-compatible aliases; secondary means Luna in v2.
                "secondary_triggered": state["luna_triggered"],
                "secondary_extracted_value": state["luna_extracted_value"],
                "secondary_verdict": state["luna_verdict"],
                "secondary_comparison": state["luna_comparison"],
            }

        if local_review is None:
            try:
                primary_value = self.extract_auto_check_value(
                    task=task,
                    attribute_name=attribute_name,
                    claimed_value=claimed_value,
                    endpoint_pool=endpoint_pool,
                )
            except Exception as error:
                error_code = _safe_error_code(error)
                primary_value = ""
                primary_verdict = "insufficient"
                primary_comparison = "auto_check_failed"
                state.update(
                    primary_extracted_value=primary_value,
                    primary_verdict=primary_verdict,
                    primary_comparison=primary_comparison,
                    primary_error_code=error_code,
                )
            else:
                primary_verdict, primary_comparison = classify(primary_value)
                state.update(
                    primary_extracted_value=primary_value,
                    primary_verdict=primary_verdict,
                    primary_comparison=primary_comparison,
                )
        else:
            raw_primary_value = (
                local_review.get("primary_extracted_value")
                if "primary_extracted_value" in local_review
                else local_review.get("extracted_value")
            )
            primary_value = clean_text(raw_primary_value)
            primary_verdict = clean_text(local_review.get("primary_verdict"))
            primary_comparison = clean_text(
                local_review.get("primary_comparison")
            )
            if primary_verdict not in {
                "supported",
                "contradicted",
                "insufficient",
            }:
                primary_verdict, primary_comparison = classify(primary_value)
            state.update(
                primary_extracted_value=primary_value,
                primary_verdict=primary_verdict,
                primary_comparison=primary_comparison,
                primary_error_code=clean_text(
                    local_review.get("primary_error_code")
                ),
                initial_reviewer_profile=(
                    clean_text(local_review.get("initial_reviewer_profile"))
                    or None
                ),
                initial_reviewer_model=(
                    clean_text(local_review.get("initial_reviewer_model"))
                    or None
                ),
                final_judge_profile=(
                    clean_text(local_review.get("final_judge_profile"))
                    or None
                ),
                final_judge_model=(
                    clean_text(local_review.get("final_judge_model"))
                    or None
                ),
            )
        luna = getattr(self, "auto_check_luna_reviewer", None)
        if luna is None:
            if state["primary_error_code"]:
                # Without a secondary reviewer no retry can change the
                # outcome, so fail closed as a terminal decision. The
                # empty top-level error_code keeps the record reusable
                # by model_auto_check_is_complete; the cause stays in
                # primary_error_code for diagnostics.
                return finish(
                    value=primary_value,
                    verdict="insufficient",
                    comparison="auto_check_failed",
                    source="primary_local_failed",
                    complete=True,
                )
            return finish(
                value=primary_value,
                verdict=primary_verdict,
                comparison=primary_comparison,
                source="primary_local",
                complete=True,
            )

        if defer_remote:
            return finish(
                value=primary_value,
                verdict=primary_verdict,
                comparison=primary_comparison,
                source="remote_review_pending",
                complete=False,
            )

        batch = self._auto_check_review_batch(
            task=task,
            attribute_name=attribute_name,
            claimed_value=claimed_value,
        )
        def record_initial_identity(initial_identity: dict[str, Any]) -> None:
            state["initial_reviewer_profile"] = clean_text(
                initial_identity.get("api_profile")
            ) or None
            state["initial_reviewer_model"] = clean_text(
                initial_identity.get("model")
            ) or None
        cached_luna_value: Any = None
        cached_luna_available = False
        if local_review is not None:
            if local_review.get("luna_extracted_value") is not None:
                cached_luna_value = local_review.get("luna_extracted_value")
            elif local_review.get("secondary_extracted_value") is not None:
                cached_luna_value = local_review.get("secondary_extracted_value")
            cached_luna_verdict = clean_text(
                local_review.get("luna_verdict")
                or local_review.get("secondary_verdict")
            )
            cached_luna_available = bool(
                (
                    local_review.get("luna_triggered")
                    or local_review.get("secondary_triggered")
                )
                and cached_luna_value is not None
                and not clean_text(local_review.get("luna_error_code"))
                and cached_luna_verdict
                in {"supported", "contradicted", "insufficient"}
            )

        state["luna_triggered"] = True
        if cached_luna_available:
            luna_value = clean_text(cached_luna_value)
        else:
            try:
                luna_value = extract_with(
                    luna,
                    batch,
                    on_selected=record_initial_identity,
                )
            except Exception as error:
                error_code = _safe_error_code(error)
                state["luna_error_code"] = error_code
                return finish(
                    value="",
                    verdict="insufficient",
                    comparison="luna_recovery_failed",
                    source="luna_recovery_incomplete",
                    complete=False,
                    error_code=error_code,
                )

        luna_verdict, luna_comparison = classify(luna_value)
        state.update(
            luna_extracted_value=luna_value,
            luna_verdict=luna_verdict,
            luna_comparison=luna_comparison,
            luna_agrees_with_local=(
                not state["primary_error_code"]
                and results_agree(primary_value, luna_value)
            ),
        )
        if state["luna_agrees_with_local"]:
            return finish(
                value=luna_value,
                verdict=luna_verdict,
                comparison=luna_comparison,
                source="local_luna_consensus",
                complete=True,
            )

        terra = getattr(self, "auto_check_terra_reviewer", None)
        if terra is None:
            dynamic_final_pool = getattr(luna, "companion_final_pool", None)
            if dynamic_final_pool is not None and dynamic_final_pool.has_reviewers():
                terra = dynamic_final_pool
        if terra is None:
            return finish(
                value="",
                verdict="insufficient",
                comparison="final_judge_failed",
                source="final_judge_incomplete",
                complete=False,
                error_code="model_review_failed:final_judge_not_configured",
            )
        state["terra_triggered"] = True
        state["final_judge_triggered"] = True
        def record_final_identity(final_identity: dict[str, Any]) -> None:
            state["final_judge_profile"] = clean_text(
                final_identity.get("api_profile")
            ) or None
            state["final_judge_model"] = clean_text(
                final_identity.get("model")
            ) or None
        cached_terra_value: Any = None
        cached_terra_available = False
        if local_review is not None:
            cached_terra_value = local_review.get("terra_extracted_value")
            cached_terra_verdict = clean_text(local_review.get("terra_verdict"))
            cached_terra_available = bool(
                (
                    local_review.get("terra_triggered")
                    or local_review.get("final_judge_triggered")
                )
                and cached_terra_value is not None
                and not clean_text(local_review.get("terra_error_code"))
                and cached_terra_verdict
                in {"supported", "contradicted", "insufficient"}
            )
        if cached_terra_available:
            terra_value = clean_text(cached_terra_value)
        else:
            try:
                terra_value = extract_with(
                    terra,
                    batch,
                    on_selected=record_final_identity,
                )
            except Exception as error:
                error_code = _safe_error_code(error)
                state["terra_error_code"] = error_code
                return finish(
                    value="",
                    verdict="insufficient",
                    comparison="final_judge_failed",
                    source="final_judge_incomplete",
                    complete=False,
                    error_code=error_code,
                )

        terra_verdict, terra_comparison = classify(terra_value)
        state.update(
            terra_extracted_value=terra_value,
            terra_verdict=terra_verdict,
            terra_comparison=terra_comparison,
        )
        return finish(
            value=terra_value,
            verdict=terra_verdict,
            comparison=terra_comparison,
            source=(
                "final_judge"
                if state["final_judge_model"]
                else "terra_adjudication"
            ),
            complete=True,
        )

    def complete_auto_check_attribute_review(
        self,
        *,
        task: ExtractionTask,
        attribute_name: str,
        claimed_value: str,
        local_review: dict[str, Any],
    ) -> dict[str, Any]:
        """Finish a deferred Luna/Terra review without repeating local inference."""
        return self.review_auto_check_attribute(
            task=task,
            attribute_name=attribute_name,
            claimed_value=claimed_value,
            local_review=local_review,
        )


class ExtractionCacheSnapshot:
    def __init__(
        self,
        items: dict[str, dict[str, Any]],
        transient_items: dict[str, dict[str, Any]],
    ) -> None:
        self.items = items
        self.transient_items = transient_items

    def get(self, key: str) -> dict[str, Any] | None:
        return self.items.get(key)

    def get_transient(self, key: str) -> dict[str, Any] | None:
        return self.transient_items.get(key)


class ExtractionCache:
    def __init__(
        self,
        path: Path,
        reuse: bool = True,
        record_key_alias: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict[str, Any]] = {}
        self.transient_items: dict[str, dict[str, Any]] = {}
        self.record_key_alias = record_key_alias
        self._lock = threading.Lock()
        if reuse and path.exists():
            for record in iter_jsonl_records([path]):
                key = clean_text(record.get("cache_key"))
                if key:
                    self.items[key] = record
                alias = record_key_alias(record) if record_key_alias else None
                if alias:
                    self.items[alias] = record

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.items.get(key)

    def get_transient(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self.transient_items.get(key)

    def snapshot(self, keys: Iterable[str]) -> ExtractionCacheSnapshot:
        unique_keys = tuple(dict.fromkeys(keys))
        with self._lock:
            return ExtractionCacheSnapshot(
                {
                    key: self.items[key]
                    for key in unique_keys
                    if key in self.items
                },
                {
                    key: self.transient_items[key]
                    for key in unique_keys
                    if key in self.transient_items
                },
            )

    def put(self, key: str, record: dict[str, Any]) -> None:
        with self._lock:
            self.items[key] = record
            alias = self.record_key_alias(record) if self.record_key_alias else None
            if alias:
                self.items[alias] = record
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


def apply_model_auto_check(
    *,
    extractor: Any,
    task: ExtractionTask,
    record: dict[str, Any],
    endpoint_pool: str = "local",
    defer_remote: bool = False,
    existing_reviews: dict[str, dict[str, Any]] | None = None,
    source_row_attributes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Keep only model attributes confirmed by an independent blind check.

    The first model response is retained in ``model_attributes`` for auditability.
    Downstream construction consumes only ``attributes``, which is fail-closed:
    mismatches, empty extractions, and checker errors are all removed.
    """
    if not getattr(extractor, "auto_check_enabled", False):
        return record
    reviewer = getattr(extractor, "review_auto_check_attribute", None)
    checker = getattr(extractor, "extract_auto_check_value", None)
    deferred_reviewer = getattr(
        extractor,
        "complete_auto_check_attribute_review",
        None,
    )
    if not callable(reviewer) and not callable(checker):
        raise RuntimeError(
            "auto-check is enabled but the extractor has no "
            "review_auto_check_attribute or extract_auto_check_value method"
        )

    model_attributes = normalize_extracted_attributes(
        {"attributes": record.get("attributes")},
        task.candidate_attribute_names,
    )
    source_values: dict[str, tuple[str, str]] = {}
    comparison_attributes = (
        source_row_attributes
        if source_row_attributes is not None
        else task.entity.get("row_attributes")
    )
    for item in canonical_extraction_row_attributes(comparison_attributes):
        normalized_name = normalize(item.get("name"))
        if normalized_name and normalized_name not in source_values:
            source_values[normalized_name] = (
                clean_text(item.get("name")),
                clean_text(item.get("value")),
            )

    kept: list[dict[str, str]] = []
    reviews: list[dict[str, Any]] = []
    stats = getattr(extractor, "model_auto_check_stats", None)
    predictions_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for predicted in model_attributes:
        normalized_name = normalize(predicted.get("name"))
        if normalized_name:
            predictions_by_name[normalized_name].append(predicted)
    for normalized_name, predictions in predictions_by_name.items():
        existing_review = (existing_reviews or {}).get(normalized_name)
        source_attribute = source_values.get(normalized_name)
        model_values = [clean_text(item.get("value")) for item in predictions]
        model_value = model_values[0] if model_values else ""
        extracted_value = ""
        error_code = ""
        decision_source = "model_analysis"
        review_complete = True
        primary_extracted_value: str | None = None
        primary_verdict: str | None = None
        primary_comparison: str | None = None
        primary_error_code = ""
        luna_triggered = False
        luna_extracted_value: str | None = None
        luna_verdict: str | None = None
        luna_comparison: str | None = None
        luna_agrees_with_local: bool | None = None
        luna_error_code = ""
        terra_triggered = False
        terra_extracted_value: str | None = None
        terra_verdict: str | None = None
        terra_comparison: str | None = None
        terra_error_code = ""
        initial_reviewer_profile: str | None = None
        initial_reviewer_model: str | None = None
        final_judge_profile: str | None = None
        final_judge_model: str | None = None
        final_judge_triggered = False
        secondary_triggered = False
        secondary_extracted_value: str | None = None
        secondary_verdict: str | None = None
        secondary_comparison: str | None = None
        if source_attribute is None:
            attribute_name = clean_text(predictions[0].get("name"))
            claimed_value = ""
            verdict = "insufficient"
            comparison = "source_attribute_missing"
        else:
            attribute_name, claimed_value = source_attribute
            matching_model_values = [
                value
                for value in model_values
                if values_match(
                    value,
                    claimed_value,
                    attribute_name=attribute_name,
                    entity_column_name=task.entity_column_name,
                )
            ]
            if not matching_model_values:
                verdict = "contradicted"
                comparison = "model_value_mismatch"
            else:
                model_value = matching_model_values[0]
                if callable(reviewer):
                    try:
                        if existing_review is not None and bool(
                            existing_review.get("review_complete")
                        ):
                            review = dict(existing_review)
                        elif existing_review is not None and callable(
                            deferred_reviewer
                        ):
                            review = deferred_reviewer(
                                task=task,
                                attribute_name=attribute_name,
                                claimed_value=claimed_value,
                                local_review=existing_review,
                            )
                        else:
                            review_kwargs = {
                                "task": task,
                                "attribute_name": attribute_name,
                                "claimed_value": claimed_value,
                                "endpoint_pool": endpoint_pool,
                            }
                            if defer_remote and callable(deferred_reviewer):
                                review_kwargs["defer_remote"] = True
                            review = reviewer(**review_kwargs)
                        if not isinstance(review, dict):
                            raise TypeError(
                                "auto checker review must be an object"
                            )
                    except Exception as error:
                        verdict = "insufficient"
                        comparison = "auto_check_failed"
                        error_code = type(error).__name__
                        review_complete = False
                        decision_source = "checker_failed"
                    else:
                        extracted_value = clean_text(
                            review.get("extracted_value")
                        )
                        verdict = clean_text(review.get("verdict"))
                        if verdict not in {
                            "supported",
                            "contradicted",
                            "insufficient",
                        }:
                            verdict = "insufficient"
                        comparison = clean_text(review.get("comparison")) or (
                            "auto_check_failed"
                        )
                        decision_source = clean_text(
                            review.get("decision_source")
                        ) or "primary_local"
                        review_complete = bool(review.get("review_complete"))
                        error_code = clean_text(review.get("error_code"))
                        primary_extracted_value = clean_text(
                            review.get("primary_extracted_value")
                        )
                        primary_verdict = clean_text(
                            review.get("primary_verdict")
                        ) or None
                        primary_comparison = clean_text(
                            review.get("primary_comparison")
                        ) or None
                        primary_error_code = clean_text(
                            review.get("primary_error_code")
                        )
                        luna_triggered = bool(review.get("luna_triggered"))
                        raw_luna_value = review.get("luna_extracted_value")
                        luna_extracted_value = (
                            clean_text(raw_luna_value)
                            if raw_luna_value is not None
                            else None
                        )
                        luna_verdict = clean_text(review.get("luna_verdict")) or None
                        luna_comparison = (
                            clean_text(review.get("luna_comparison")) or None
                        )
                        raw_luna_agreement = review.get("luna_agrees_with_local")
                        luna_agrees_with_local = (
                            bool(raw_luna_agreement)
                            if raw_luna_agreement is not None
                            else None
                        )
                        luna_error_code = clean_text(
                            review.get("luna_error_code")
                        )
                        terra_triggered = bool(review.get("terra_triggered"))
                        raw_terra_value = review.get("terra_extracted_value")
                        terra_extracted_value = (
                            clean_text(raw_terra_value)
                            if raw_terra_value is not None
                            else None
                        )
                        terra_verdict = (
                            clean_text(review.get("terra_verdict")) or None
                        )
                        terra_comparison = (
                            clean_text(review.get("terra_comparison")) or None
                        )
                        terra_error_code = clean_text(
                            review.get("terra_error_code")
                        )
                        initial_reviewer_profile = (
                            clean_text(review.get("initial_reviewer_profile"))
                            or None
                        )
                        initial_reviewer_model = (
                            clean_text(review.get("initial_reviewer_model"))
                            or None
                        )
                        final_judge_profile = (
                            clean_text(review.get("final_judge_profile"))
                            or None
                        )
                        final_judge_model = (
                            clean_text(review.get("final_judge_model"))
                            or None
                        )
                        final_judge_triggered = bool(
                            review.get("final_judge_triggered")
                        )
                        secondary_triggered = bool(
                            review.get("secondary_triggered")
                        )
                        raw_secondary_value = review.get(
                            "secondary_extracted_value"
                        )
                        secondary_extracted_value = (
                            clean_text(raw_secondary_value)
                            if raw_secondary_value is not None
                            else None
                        )
                        secondary_verdict = clean_text(
                            review.get("secondary_verdict")
                        ) or None
                        secondary_comparison = clean_text(
                            review.get("secondary_comparison")
                        ) or None
                        if verdict == "supported" and review_complete:
                            kept.append(
                                {
                                    "name": attribute_name,
                                    "value": claimed_value,
                                }
                            )
                else:
                    try:
                        extracted_value = clean_text(
                            checker(
                                task=task,
                                attribute_name=attribute_name,
                                claimed_value=claimed_value,
                                endpoint_pool=endpoint_pool,
                            )
                        )
                    except Exception as error:
                        verdict = "insufficient"
                        comparison = "auto_check_failed"
                        error_code = type(error).__name__
                        review_complete = False
                        decision_source = "checker_failed"
                    else:
                        if not extracted_value:
                            verdict = "insufficient"
                            comparison = "empty_extraction"
                        elif values_match(
                            extracted_value,
                            claimed_value,
                            attribute_name=attribute_name,
                            entity_column_name=task.entity_column_name,
                        ):
                            verdict = "supported"
                            comparison = "normalized_values_match"
                            decision_source = "primary_local"
                            kept.append(
                                {
                                    "name": attribute_name,
                                    "value": claimed_value,
                                }
                            )
                        else:
                            verdict = "contradicted"
                            comparison = "extracted_value_mismatch"
                            decision_source = "primary_local"
        if (
            decision_source != "remote_review_pending"
            and not (
                existing_review is not None
                and bool(existing_review.get("review_complete"))
            )
            and stats is not None
            and hasattr(stats, "record")
        ):
            stats.record(
                verdict,
                error=bool(error_code),
                luna_triggered=luna_triggered,
                luna_verdict=luna_verdict,
                terra_triggered=terra_triggered,
                terra_verdict=terra_verdict,
                decision_source=decision_source,
            )
        reviews.append(
            {
                "attribute_name": attribute_name,
                "claimed_value": claimed_value,
                "model_value": model_value,
                "extracted_value": extracted_value,
                "verdict": verdict,
                "comparison": comparison,
                "error_code": error_code,
                "review_complete": review_complete,
                "decision_source": decision_source,
                "primary_extracted_value": primary_extracted_value,
                "primary_verdict": primary_verdict,
                "primary_comparison": primary_comparison,
                "primary_error_code": primary_error_code,
                "luna_triggered": luna_triggered,
                "luna_extracted_value": luna_extracted_value,
                "luna_verdict": luna_verdict,
                "luna_comparison": luna_comparison,
                "luna_agrees_with_local": luna_agrees_with_local,
                "luna_error_code": luna_error_code,
                "terra_triggered": terra_triggered,
                "terra_extracted_value": terra_extracted_value,
                "terra_verdict": terra_verdict,
                "terra_comparison": terra_comparison,
                "terra_error_code": terra_error_code,
                "initial_reviewer_profile": initial_reviewer_profile,
                "initial_reviewer_model": initial_reviewer_model,
                "final_judge_profile": final_judge_profile,
                "final_judge_model": final_judge_model,
                "final_judge_triggered": final_judge_triggered,
                "secondary_triggered": secondary_triggered,
                "secondary_extracted_value": secondary_extracted_value,
                "secondary_verdict": secondary_verdict,
                "secondary_comparison": secondary_comparison,
            }
        )

    updated = dict(record)
    updated["model_attributes"] = model_attributes
    updated["attributes"] = kept
    updated["auto_check"] = {
        "schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
        "policy": "keep_source_canonical_supported_only_fail_closed",
        "review_policy": model_auto_check_review_policy(extractor),
        "reviewed_attributes": len(reviews),
        "supported_attributes": len(kept),
        "filtered_attributes": len(reviews) - len(kept),
        "reviews": reviews,
    }
    return updated


def model_auto_check_has_remote_pending(record: dict[str, Any]) -> bool:
    auto_check = record.get("auto_check")
    reviews = auto_check.get("reviews") if isinstance(auto_check, dict) else None
    return bool(
        isinstance(reviews, list)
        and any(
            isinstance(review, dict)
            and not bool(review.get("review_complete"))
            and clean_text(review.get("decision_source"))
            == "remote_review_pending"
            for review in reviews
        )
    )


def complete_deferred_model_auto_check(
    *,
    extractor: Any,
    task: ExtractionTask,
    record: dict[str, Any],
    source_row_attributes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Complete pending OpenAI reviews without repeating model analysis/checks."""
    auto_check = record.get("auto_check")
    reviews = auto_check.get("reviews") if isinstance(auto_check, dict) else []
    existing_reviews = {
        normalize(review.get("attribute_name")): {
            **review,
            "review_complete": False,
        }
        for review in reviews
        if isinstance(review, dict) and normalize(review.get("attribute_name"))
    }
    analysis_record = dict(record)
    analysis_record["attributes"] = list(record.get("model_attributes") or [])
    analysis_record.pop("auto_check", None)
    return apply_model_auto_check(
        extractor=extractor,
        task=task,
        record=analysis_record,
        existing_reviews=existing_reviews,
        source_row_attributes=source_row_attributes,
    )


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
    for field_name in (
        "model_endpoint",
        "model_kind",
        "model_failure_type",
    ):
        if clean_text(result.get(field_name)):
            record[field_name] = clean_text(result[field_name])
    return record


def run_extraction_task(
    extractor: LocalAttributeExtractor,
    task: ExtractionTask,
    endpoint_pool: str = "local",
    apply_auto_check_gate: bool = True,
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
    except TransientModelEndpointError as error:
        result = {
            "attributes": [],
            "raw_response": "",
            "error": "model endpoint temporarily unavailable",
            "error_class": "model_endpoint_transient",
        }
        diagnostics = {
            "model_endpoint": error.model_endpoint,
            "model_kind": error.model_kind,
            "model_failure_type": error.failure_type,
        }
        result.update(
            {
                field_name: value
                for field_name, value in diagnostics.items()
                if value
            }
        )
    except Exception as exc:
        result = {"attributes": [], "raw_response": "", "error": str(exc)}
    record = extraction_record_from_result(task, result)
    if apply_auto_check_gate and not clean_text(record.get("error")):
        record = apply_model_auto_check(
            extractor=extractor,
            task=task,
            record=record,
            endpoint_pool=endpoint_pool,
        )
    return record


def run_extraction_task_group(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    workers: int,
    endpoint_pool: str = "local",
    apply_auto_check_gate: bool = True,
    on_record: Callable[[str, dict[str, Any]], None] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> dict[str, dict[str, Any]]:
    if not tasks:
        return {}
    workers = max(1, min(workers, len(tasks)))
    if workers == 1 and executor is None:
        records = {}
        for task in tasks:
            record = run_extraction_task(
                extractor,
                task,
                endpoint_pool,
                apply_auto_check_gate,
            )
            records[task.cache_key] = record
            if on_record is not None:
                on_record(task.cache_key, record)
        return records

    def collect(pool: ThreadPoolExecutor) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        future_to_task = {
            pool.submit(
                run_extraction_task,
                extractor,
                task,
                endpoint_pool,
                apply_auto_check_gate,
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

    if executor is not None:
        return collect(executor)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return collect(pool)


def run_extraction_kind_adaptive(
    *,
    extractor: LocalAttributeExtractor,
    tasks: list[ExtractionTask],
    model_kind: str,
    state: ModelConcurrencyState,
    endpoint_pool: str = "local",
    apply_auto_check_gate: bool = True,
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
        apply_auto_check_gate=apply_auto_check_gate,
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
                apply_auto_check_gate=apply_auto_check_gate,
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
    apply_auto_check_gate: bool = True,
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
            apply_auto_check_gate=apply_auto_check_gate,
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
            record = run_extraction_task(
                extractor,
                task,
                endpoint_pool,
                apply_auto_check_gate,
            )
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
                    apply_auto_check_gate=apply_auto_check_gate,
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
    apply_auto_check_gate: bool = True,
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
                    apply_auto_check_gate=apply_auto_check_gate,
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
                apply_auto_check_gate=apply_auto_check_gate,
                on_record=on_record,
            ): kind
            for kind, group_tasks in active_groups
        }
        for future in as_completed(futures):
            records.update(future.result())
    return records


def model_auto_check_is_complete(
    record: dict[str, Any],
    *,
    required: bool = False,
) -> bool:
    """Return whether an attached post-analysis check reached a final decision.

    Records without ``auto_check`` remain valid for callers that explicitly use
    an extractor without the gate. When ``required`` is true, a legacy analysis
    record without the current gate is incomplete and must be upgraded in place.
    Malformed reviews, checker errors, and interrupted reviews must be retried.
    A primary-local error is still terminal when the OpenAI secondary completed;
    only the final ``error_code`` and ``review_complete`` fields govern reuse.
    Without a configured secondary, a primary-local failure is recorded as a
    terminal fail-closed decision (``review_complete`` set, empty top-level
    ``error_code``) so it is cached persistently instead of retried forever.
    """
    if "auto_check" not in record:
        return not required
    auto_check = record.get("auto_check")
    if not isinstance(auto_check, dict):
        return False
    if clean_text(auto_check.get("schema_version")) != MODEL_AUTO_CHECK_SCHEMA_VERSION:
        return False
    reviews = auto_check.get("reviews")
    if not isinstance(reviews, list):
        return False
    try:
        reviewed_attributes = int(auto_check.get("reviewed_attributes", -1))
    except (TypeError, ValueError):
        return False
    if reviewed_attributes != len(reviews):
        return False
    return all(
        isinstance(review, dict)
        and bool(review.get("review_complete"))
        and not clean_text(review.get("error_code"))
        for review in reviews
    )


def auto_check_required(extractor: Any | None) -> bool:
    return bool(extractor is not None and getattr(extractor, "auto_check_enabled", False))


def model_auto_check_review_policy(extractor: Any | None) -> str:
    """Return the cache-visible policy used for final auto-check decisions."""
    if extractor is not None and getattr(
        extractor,
        "auto_check_luna_reviewer",
        None,
    ) is not None:
        return AUTO_CHECK_REVIEW_POLICY_CASCADE
    return AUTO_CHECK_REVIEW_POLICY_LOCAL


def cached_model_auto_check_review_policy(record: dict[str, Any]) -> str:
    """Return the policy that produced a cached final decision."""
    auto_check = record.get("auto_check")
    recorded = clean_text(record.get("review_policy"))
    if not recorded and isinstance(auto_check, dict):
        recorded = clean_text(auto_check.get("review_policy"))
    if recorded:
        return recorded
    if query_recovery_remote_review_is_complete(record):
        return AUTO_CHECK_REVIEW_POLICY_LEGACY
    return AUTO_CHECK_REVIEW_POLICY_LOCAL


def model_auto_check_cache_schema_version(extractor: Any | None) -> str:
    """Keep legacy local-only keys while isolating the new cascade policy."""
    policy = model_auto_check_review_policy(extractor)
    if policy == AUTO_CHECK_REVIEW_POLICY_LOCAL:
        return MODEL_AUTO_CHECK_SCHEMA_VERSION
    return f"{MODEL_AUTO_CHECK_SCHEMA_VERSION}:{policy}"


def cached_extraction_is_reusable(
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    require_auto_check: bool = False,
) -> bool:
    # Model extraction is a high-recall discovery cache.  Query-level recovery
    # checks are intentionally not part of its reuse contract.
    if clean_text(record.get("error")) and not getattr(args, "cache_failed_model_outputs", False):
        return False
    if getattr(args, "refresh_invalid_model_cache", False) and should_refresh_cached_extraction(record):
        return False
    return True


def candidate_extraction_record(record: dict[str, Any]) -> dict[str, Any]:
    """Expose the original model candidates, including legacy gated caches."""
    updated = dict(record)
    model_attributes = record.get("model_attributes")
    if isinstance(model_attributes, list):
        updated["attributes"] = model_attributes
    # The old field represented a global pre-query gate.  Keep any completed
    # reviews available for migration, but do not expose its filtered values.
    updated.pop("auto_check", None)
    return updated


def _query_recovery_auto_check_key_fields(
    *,
    schema_version: str,
    extraction_cache_key: Any,
    query_row_attributes: Any,
    attribute_name: Any,
    claimed_value: Any,
    masked_attribute_names: Iterable[Any] = (),
) -> str:
    masked_names = {
        normalize(name)
        for name in (*tuple(masked_attribute_names), attribute_name)
        if normalize(name)
    }
    query_row = json.dumps(
        [
            item
            for item in canonical_extraction_row_attributes(query_row_attributes)
            if normalize(item.get("name")) not in masked_names
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return stable_hash(
        schema_version,
        extraction_cache_key,
        query_row,
        attribute_name,
        claimed_value,
        length=32,
    )


def _query_recovery_evidence_identity_fields(
    *,
    asset_id: Any,
    asset_type: Any,
    row_attributes: Any,
    attribute_name: Any,
    claimed_value: Any,
    masked_attribute_names: Iterable[Any] = (),
) -> dict[str, Any]:
    """Return the model-independent semantic identity of one evidence check."""
    target_name = normalize(attribute_name)
    masked_names = {
        normalize(name)
        for name in (*tuple(masked_attribute_names), target_name)
        if normalize(name)
    }
    masked_row = [
        item
        for item in canonical_extraction_row_attributes(row_attributes)
        if normalize(item.get("name")) not in masked_names
    ]
    return {
        "cache_version": QUERY_RECOVERY_REMOTE_EVIDENCE_CACHE_VERSION,
        "auto_check_schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
        "review_policy": AUTO_CHECK_REVIEW_POLICY_CASCADE,
        "asset_id": clean_text(asset_id),
        "asset_type": clean_text(asset_type),
        "masked_row": masked_row,
        "attribute_name": target_name,
        # The remote model never sees the claimed value, but the cached final
        # verdict is obtained by comparing its blind extraction with this value.
        "claimed_value": clean_text(claimed_value),
    }


def _query_recovery_evidence_identity_key(
    identity: dict[str, Any],
) -> str:
    return stable_hash(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        length=32,
    )


def query_recovery_remote_evidence_identity(
    candidate: QueryRecoveryCandidate,
) -> dict[str, Any]:
    recovered = candidate.recovery["recovered_attribute"]
    task = candidate.task
    return _query_recovery_evidence_identity_fields(
        asset_id=task.asset.get("asset_id"),
        asset_type=task.asset.get("asset_type"),
        row_attributes=task.entity.get("row_attributes"),
        attribute_name=recovered.get("column_name"),
        claimed_value=recovered.get("value"),
        masked_attribute_names=candidate.redundancy_group_attribute_names,
    )


def query_recovery_remote_evidence_key(
    candidate: QueryRecoveryCandidate,
) -> str:
    return _query_recovery_evidence_identity_key(
        query_recovery_remote_evidence_identity(candidate)
    )


def query_recovery_remote_review_is_complete(
    record: dict[str, Any],
) -> bool:
    """Whether a completed result contains a successful external API review."""
    auto_check = record.get("auto_check")
    if not model_auto_check_is_complete(
        {"auto_check": auto_check}, required=True
    ):
        return False
    reviews = auto_check.get("reviews") if isinstance(auto_check, dict) else []
    remote_sources = {
        "luna_recovery",
        "local_luna_consensus",
        "terra_adjudication",
        "final_judge",
    }
    return any(
        isinstance(review, dict)
        and (
            bool(review.get("luna_triggered"))
            or bool(review.get("secondary_triggered"))
            or bool(review.get("terra_triggered"))
            or bool(review.get("final_judge_triggered"))
            or bool(clean_text(review.get("initial_reviewer_model")))
            or bool(clean_text(review.get("final_judge_model")))
            or clean_text(review.get("decision_source")) in remote_sources
        )
        for review in reviews or []
    )


def query_recovery_auto_check_key(
    candidate: QueryRecoveryCandidate,
    extractor: Any,
) -> str:
    recovered = candidate.recovery["recovered_attribute"]
    return _query_recovery_auto_check_key_fields(
        schema_version=model_auto_check_cache_schema_version(extractor),
        extraction_cache_key=candidate.task.cache_key,
        query_row_attributes=candidate.task.entity.get("row_attributes"),
        attribute_name=recovered.get("column_name"),
        claimed_value=recovered.get("value"),
        masked_attribute_names=candidate.redundancy_group_attribute_names,
    )


def query_recovery_candidate_identity(
    candidate: QueryRecoveryCandidate,
) -> str:
    """Identify one discovered recovery independently of context layout."""
    recovered = candidate.recovery["recovered_attribute"]
    return stable_hash(
        "query-recovery-candidate-v1",
        candidate.task.cache_key,
        candidate.recovery.get("source_table_id"),
        candidate.recovery.get("source_row_id"),
        normalize(recovered.get("column_name")),
        clean_text(recovered.get("value")),
        *(
            normalize(name)
            for name in candidate.redundancy_group_attribute_names
        ),
        length=32,
    )


def query_recovery_auto_check_record_key(
    record: dict[str, Any],
) -> str | None:
    """Derive a reusable alias for a completed recovery review.

    External API decisions use a model-independent evidence alias. Purely local
    decisions retain the extraction-model-specific alias.
    """
    extraction_cache_key = clean_text(record.get("extraction_cache_key"))
    if not extraction_cache_key:
        return None
    if "attribute_name" not in record or "claimed_value" not in record:
        return None
    auto_check = record.get("auto_check")
    schema_version = clean_text(record.get("schema_version"))
    if not schema_version and isinstance(auto_check, dict):
        schema_version = clean_text(auto_check.get("schema_version"))
    if schema_version != MODEL_AUTO_CHECK_SCHEMA_VERSION:
        return None
    if not model_auto_check_is_complete(
        {"auto_check": auto_check}, required=True
    ):
        return None
    if query_recovery_remote_review_is_complete(record):
        identity = record.get("evidence_identity")
        if isinstance(identity, dict):
            return _query_recovery_evidence_identity_key(identity)
        return None
    query_row_attributes = record.get("query_row_attributes")
    if not isinstance(query_row_attributes, list):
        return None
    masked_attribute_names = record.get("masked_attribute_names")
    if not isinstance(masked_attribute_names, (list, tuple)):
        # A v6 local record without its full group mask cannot safely be
        # aliased: its stored visible row may have exposed a sibling column.
        return None
    review_policy = cached_model_auto_check_review_policy(record)
    cache_schema_version = schema_version
    if review_policy != AUTO_CHECK_REVIEW_POLICY_LOCAL:
        cache_schema_version = f"{schema_version}:{review_policy}"
    return _query_recovery_auto_check_key_fields(
        schema_version=cache_schema_version,
        extraction_cache_key=extraction_cache_key,
        query_row_attributes=query_row_attributes,
        attribute_name=record.get("attribute_name"),
        claimed_value=record.get("claimed_value"),
        masked_attribute_names=masked_attribute_names,
    )


def query_recovery_plan_row_groups(
    plan: QueryRecoveryAutoCheckPlan,
    extractor: Any,
) -> list[tuple[int, list[QueryRecoveryCandidate]]]:
    candidates_by_row: dict[int, list[QueryRecoveryCandidate]] = defaultdict(list)
    seen_keys_by_row: dict[int, set[str]] = defaultdict(set)
    for candidate in plan.candidates:
        row_id = int(candidate.recovery["source_row_id"])
        key = query_recovery_auto_check_key(candidate, extractor)
        if key in seen_keys_by_row[row_id]:
            continue
        seen_keys_by_row[row_id].add(key)
        candidates_by_row[row_id].append(candidate)
    ordered_rows = list(dict.fromkeys(plan.source_row_order))
    ordered_rows.extend(
        row_id for row_id in candidates_by_row if row_id not in ordered_rows
    )
    return [
        (row_id, candidates_by_row[row_id])
        for row_id in ordered_rows
        if candidates_by_row.get(row_id)
    ]


def query_recovery_cached_check(
    key: str,
    cache: ExtractionCache,
    candidate: QueryRecoveryCandidate | None = None,
    *,
    extractor: Any | None = None,
) -> dict[str, Any] | None:
    requested_policy = model_auto_check_review_policy(extractor)
    transient = cache.get_transient(key)
    if (
        transient is not None
        and cached_model_auto_check_review_policy(transient) == requested_policy
        and model_auto_check_is_complete(
            {"auto_check": transient.get("auto_check")}, required=True
        )
    ):
        return transient
    cached = cache.get(key)
    if (
        cached
        and cached_model_auto_check_review_policy(cached) == requested_policy
        and model_auto_check_is_complete(
            {"auto_check": cached.get("auto_check")}, required=True
        )
    ):
        return cached
    if (
        candidate is not None
        and requested_policy == AUTO_CHECK_REVIEW_POLICY_CASCADE
    ):
        remote_key = query_recovery_remote_evidence_key(candidate)
        remote_cached = cache.get(remote_key)
        if (
            remote_cached
            and cached_model_auto_check_review_policy(remote_cached)
            == requested_policy
            and query_recovery_remote_review_is_complete(remote_cached)
        ):
            return remote_cached
    return None


def query_recovery_prior_auto_check(
    candidate: QueryRecoveryCandidate,
    extractor: Any,
    cache: ExtractionCache,
) -> dict[str, Any] | None:
    """Return reusable model-stage outputs from an earlier review policy."""
    if model_auto_check_review_policy(extractor) != AUTO_CHECK_REVIEW_POLICY_CASCADE:
        return None
    recovered = candidate.recovery["recovered_attribute"]
    identity = query_recovery_remote_evidence_identity(candidate)
    identity.pop("review_policy", None)
    prior_keys = [
        query_recovery_auto_check_key(candidate, extractor),
        _query_recovery_evidence_identity_key(identity),
        _query_recovery_auto_check_key_fields(
            schema_version=MODEL_AUTO_CHECK_SCHEMA_VERSION,
            extraction_cache_key=candidate.task.cache_key,
            query_row_attributes=candidate.task.entity.get("row_attributes"),
            attribute_name=recovered.get("column_name"),
            claimed_value=recovered.get("value"),
            masked_attribute_names=candidate.redundancy_group_attribute_names,
        ),
    ]
    seen: set[str] = set()
    for prior_key in prior_keys:
        if prior_key in seen:
            continue
        seen.add(prior_key)
        record = cache.get(prior_key)
        if not isinstance(record, dict):
            continue
        auto_check = record.get("auto_check")
        reviews = auto_check.get("reviews") if isinstance(auto_check, dict) else []
        for review in reviews or []:
            if (
                isinstance(review, dict)
                and normalize(review.get("attribute_name"))
                == normalize(recovered.get("column_name"))
                and "primary_extracted_value" in review
                and not clean_text(review.get("primary_error_code"))
            ):
                return record
    return None


def query_recovery_plan_needs_model_check(
    plan: QueryRecoveryAutoCheckPlan,
    extractor: Any,
    cache: ExtractionCache,
    *,
    exhaustive: bool = False,
) -> bool:
    if exhaustive:
        return any(
            query_recovery_cached_check(
                query_recovery_auto_check_key(candidate, extractor),
                cache,
                candidate,
                extractor=extractor,
            )
            is None
            for _row_id, row_candidates in query_recovery_plan_row_groups(
                plan, extractor
            )
            for candidate in row_candidates
        )
    required = max(0, int(plan.required_recovered_rows))
    if required == 0:
        return False
    supported_rows = 0
    groups = query_recovery_plan_row_groups(plan, extractor)
    for group_index, (_row_id, row_candidates) in enumerate(groups):
        row_supported = False
        has_pending = False
        for candidate in row_candidates:
            key = query_recovery_auto_check_key(candidate, extractor)
            record = query_recovery_cached_check(
                key, cache, candidate, extractor=extractor
            )
            if record is None:
                has_pending = True
                continue
            if record.get("supported"):
                row_supported = True
                break
        if row_supported:
            supported_rows += 1
            if supported_rows >= required:
                return False
        elif has_pending:
            return True
        remaining_rows = len(groups) - group_index - 1
        if supported_rows + remaining_rows < required:
            return False
    return False


def query_recovery_plan_is_supported(
    plan: QueryRecoveryAutoCheckPlan,
    extractor: Any,
    cache: ExtractionCache,
) -> bool:
    """Return whether cached checks satisfy one query view's table floor."""
    required = max(0, int(plan.required_recovered_rows))
    if required == 0:
        return True
    supported_rows = 0
    for _row_id, row_candidates in query_recovery_plan_row_groups(
        plan, extractor
    ):
        if any(
            bool(
                (
                    query_recovery_cached_check(
                        query_recovery_auto_check_key(candidate, extractor),
                        cache,
                        candidate,
                        extractor=extractor,
                    )
                    or {}
                ).get("supported")
            )
            for candidate in row_candidates
        ):
            supported_rows += 1
            if supported_rows >= required:
                return True
    return False


def check_query_recovery_candidate(
    *,
    candidate: QueryRecoveryCandidate,
    extractor: Any,
    endpoint_pool: str = "local",
    defer_remote: bool = False,
    local_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovered = candidate.recovery["recovered_attribute"]
    predicted = {
        "name": recovered["column_name"],
        "value": recovered["model_value"],
    }
    analysis_record = {
        **candidate.extraction,
        "attributes": [predicted],
        "error": "",
    }
    source_row_attributes = [
        {
            "name": recovered["column_name"],
            "value": recovered["value"],
            "is_entity": False,
        }
    ]
    if local_result is None:
        checked = apply_model_auto_check(
            extractor=extractor,
            task=candidate.task,
            record=analysis_record,
            endpoint_pool=endpoint_pool,
            defer_remote=defer_remote,
            source_row_attributes=source_row_attributes,
        )
    else:
        pending_record = {
            **analysis_record,
            "model_attributes": [predicted],
            "auto_check": local_result.get("auto_check"),
        }
        checked = complete_deferred_model_auto_check(
            extractor=extractor,
            task=candidate.task,
            record=pending_record,
            source_row_attributes=source_row_attributes,
        )
    return {
        "schema_version": MODEL_AUTO_CHECK_SCHEMA_VERSION,
        "review_policy": model_auto_check_review_policy(extractor),
        "supported": bool(checked.get("attributes")),
        "auto_check": checked.get("auto_check"),
    }


class QueryRecoveryLocalCheckScheduler:
    """Run blind local checks independently of external-review workers."""

    def __init__(
        self,
        *,
        extractor: Any,
        state: ModelConcurrencyState,
    ) -> None:
        self.extractor = extractor
        self._lock = threading.Lock()
        self._assigned: dict[tuple[str, str], int] = defaultdict(int)
        self._cursors: dict[str, int] = defaultdict(int)
        self._capacities: dict[tuple[str, str], int] = {}
        self._executors: dict[tuple[str, str], ThreadPoolExecutor] = {}
        for model_kind in ("text", "image"):
            for endpoint_pool in ("local", "remote"):
                workers = state.workers_for(model_kind, endpoint_pool)
                if workers <= 0:
                    continue
                key = (model_kind, endpoint_pool)
                self._capacities[key] = workers
                self._executors[key] = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=(
                        f"query-recovery-{endpoint_pool}-{model_kind}"
                    ),
                )
        self.max_parallelism = sum(self._capacities.values()) or 1

    def _select_pool(self, model_kind: str) -> tuple[str, str]:
        candidates = [
            key for key in self._capacities if key[0] == model_kind
        ]
        if not candidates:
            raise RuntimeError(
                f"no query-recovery {model_kind} model workers are configured"
            )
        least_loaded = [
            key
            for key in candidates
            if all(
                self._assigned[key] * self._capacities[other]
                <= self._assigned[other] * self._capacities[key]
                for other in candidates
            )
        ]
        cursor = self._cursors[model_kind] % len(candidates)
        selected = min(
            least_loaded,
            key=lambda key: (candidates.index(key) - cursor) % len(candidates),
        )
        self._cursors[model_kind] = (
            candidates.index(selected) + 1
        ) % len(candidates)
        return selected

    def submit(self, candidate: QueryRecoveryCandidate) -> Any:
        model_kind = model_kind_for_asset(candidate.task.asset)
        with self._lock:
            selected = self._select_pool(model_kind)
            self._assigned[selected] += 1
        endpoint_pool = selected[1]
        future = self._executors[selected].submit(
            check_query_recovery_candidate,
            candidate=candidate,
            extractor=self.extractor,
            endpoint_pool=endpoint_pool,
            defer_remote=True,
        )

        def release(_future: Any) -> None:
            with self._lock:
                self._assigned[selected] -= 1

        future.add_done_callback(release)
        return future

    def close(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=True)


def resolve_query_recovery_auto_checks(
    *,
    candidates: list[QueryRecoveryCandidate],
    extractor: Any | None,
    cache: ExtractionCache,
    args: argparse.Namespace,
    required_recovered_rows: int | None = None,
    source_row_order: Iterable[int] | None = None,
    exhaustive: bool = False,
) -> dict[str, dict[str, Any]]:
    """Check one query view, optionally exhausting every evidence candidate."""
    if not candidates:
        return {}
    ordered_rows = tuple(
        dict.fromkeys(
            int(row_id)
            for row_id in (
                source_row_order
                if source_row_order is not None
                else (
                    candidate.recovery["source_row_id"]
                    for candidate in candidates
                )
            )
        )
    )
    candidate_rows = {
        int(candidate.recovery["source_row_id"]) for candidate in candidates
    }
    default_required = len(candidate_rows)
    required = (
        default_required
        if required_recovered_rows is None
        else int(required_recovered_rows)
    )
    plan = QueryRecoveryAutoCheckPlan(
        query_key="query_view_"
        + stable_hash(
            *(candidate.task.source_table_id for candidate in candidates),
            *(str(row_id) for row_id in ordered_rows),
            length=24,
        ),
        required_recovered_rows=required,
        source_row_order=ordered_rows,
        candidates=tuple(candidates),
    )
    return resolve_query_recovery_auto_check_plans(
        plans=[plan],
        extractor=extractor,
        cache=cache,
        args=args,
        exhaustive=exhaustive,
    )


def resolve_query_recovery_auto_check_plans(
    *,
    plans: list[QueryRecoveryAutoCheckPlan],
    extractor: Any | None,
    cache: ExtractionCache,
    args: argparse.Namespace,
    concurrency_state: ModelConcurrencyState | None = None,
    exhaustive: bool = False,
) -> dict[str, dict[str, Any]]:
    """Resolve query plans with independent local and external-review stages."""
    if not plans:
        return {}
    if not auto_check_required(extractor):
        return {
            query_recovery_auto_check_key(candidate, extractor): {
                "supported": True,
                "auto_check": None,
            }
            for plan in plans
            for candidate in plan.candidates
        }

    progress_bar = None
    if tqdm is not None and any(
        query_recovery_plan_needs_model_check(
            plan, extractor, cache, exhaustive=exhaustive
        )
        for plan in plans
    ):
        progress_bar = tqdm(
            total=0,
            desc=(
                "Query recovery accepted-evidence check"
                if exhaustive
                else "Query recovery eligibility check"
            ),
            unit="recovery",
            dynamic_ncols=True,
            disable=not bool(getattr(args, "model_progress", True)),
        )
    progress_lock = threading.Lock()

    def begin_check() -> None:
        if progress_bar is None:
            return
        with progress_lock:
            progress_bar.total += 1
            refresh = getattr(progress_bar, "refresh", None)
            if callable(refresh):
                refresh()

    def finish_check() -> None:
        if progress_bar is None:
            return
        with progress_lock:
            progress_bar.update(1)

    @dataclass
    class PlanState:
        groups: list[tuple[int, list[QueryRecoveryCandidate]]]
        required: int
        row_index: int = 0
        supported_rows: int = 0
        pending_candidates: list[QueryRecoveryCandidate] = dataclass_field(
            default_factory=list
        )
        candidate_index: int = 0
        row_initialized: bool = False
        awaiting_key: str | None = None
        done: bool = False

    states = [
        PlanState(
            groups=query_recovery_plan_row_groups(plan, extractor),
            required=max(0, int(plan.required_recovered_rows)),
        )
        for plan in plans
    ]
    resolved: dict[str, dict[str, Any]] = {}
    in_flight: dict[str, QueryRecoveryCandidate] = {}
    waiters: dict[str, list[PlanState]] = defaultdict(list)
    completions: queue.Queue[
        tuple[
            str,
            str,
            QueryRecoveryCandidate,
            dict[str, Any] | None,
            BaseException | None,
        ]
    ] = queue.Queue()
    local_scheduler = QueryRecoveryLocalCheckScheduler(
        extractor=extractor,
        state=concurrency_state or ModelConcurrencyState.from_args(args),
    )
    external_workers = max(
        1,
        int(
            getattr(
                extractor,
                "auto_check_parallelism",
                getattr(
                    args,
                    "auto_check_openai_max_inflight",
                    MAX_AUTO_CHECK_OPENAI_CONCURRENCY,
                ),
            )
        ),
    )
    external_executor = ThreadPoolExecutor(
        max_workers=external_workers,
        thread_name_prefix="query-recovery-external-review",
    )

    def queue_completion(
        future: Any,
        *,
        phase: str,
        key: str,
        candidate: QueryRecoveryCandidate,
    ) -> None:
        try:
            result = future.result()
        except BaseException as error:
            completions.put((phase, key, candidate, None, error))
        else:
            completions.put((phase, key, candidate, result, None))

    def submit_local(key: str, candidate: QueryRecoveryCandidate) -> None:
        begin_check()
        in_flight[key] = candidate
        future = local_scheduler.submit(candidate)
        future.add_done_callback(
            lambda completed, key=key, candidate=candidate: queue_completion(
                completed,
                phase="local",
                key=key,
                candidate=candidate,
            )
        )

    def submit_external(
        key: str,
        candidate: QueryRecoveryCandidate,
        local_result: dict[str, Any],
    ) -> None:
        future = external_executor.submit(
            check_query_recovery_candidate,
            candidate=candidate,
            extractor=extractor,
            local_result=local_result,
        )
        future.add_done_callback(
            lambda completed, key=key, candidate=candidate: queue_completion(
                completed,
                phase="external",
                key=key,
                candidate=candidate,
            )
        )

    def submit_prior(
        key: str,
        candidate: QueryRecoveryCandidate,
        prior_result: dict[str, Any],
    ) -> None:
        begin_check()
        in_flight[key] = candidate
        submit_external(key, candidate, prior_result)

    def finish_row(state: PlanState, *, supported: bool) -> None:
        if supported:
            state.supported_rows += 1
        state.row_index += 1
        state.pending_candidates = []
        state.candidate_index = 0
        state.row_initialized = False
        if state.row_index >= len(state.groups) or (
            not exhaustive
            and (
                state.supported_rows >= state.required
                or state.supported_rows
                + len(state.groups)
                - state.row_index
                < state.required
            )
        ):
            state.done = True

    def advance(state: PlanState) -> None:
        while not state.done and state.awaiting_key is None:
            if (
                (state.required == 0 and not exhaustive)
                or state.row_index >= len(state.groups)
            ):
                state.done = True
                return
            if not state.row_initialized:
                state.pending_candidates = []
                state.candidate_index = 0
                state.row_initialized = True
                _row_id, row_candidates = state.groups[state.row_index]
                cached_support = False
                for candidate in row_candidates:
                    key = query_recovery_auto_check_key(candidate, extractor)
                    record = query_recovery_cached_check(
                        key, cache, candidate, extractor=extractor
                    )
                    if record is None:
                        state.pending_candidates.append(candidate)
                        continue
                    resolved[key] = record
                    if record.get("supported"):
                        cached_support = True
                        if not exhaustive:
                            break
                if cached_support and not exhaustive:
                    finish_row(state, supported=True)
                    continue

            if state.candidate_index >= len(state.pending_candidates):
                finish_row(state, supported=False)
                continue

            candidate = state.pending_candidates[state.candidate_index]
            state.candidate_index += 1
            key = query_recovery_auto_check_key(candidate, extractor)
            cached = query_recovery_cached_check(
                key, cache, candidate, extractor=extractor
            )
            if cached is not None:
                resolved[key] = cached
                if cached.get("supported") and not exhaustive:
                    finish_row(state, supported=True)
                continue
            prior_result = query_recovery_prior_auto_check(
                candidate,
                extractor,
                cache,
            )
            state.awaiting_key = key
            waiters[key].append(state)
            if key not in in_flight:
                if prior_result is None:
                    submit_local(key, candidate)
                else:
                    submit_prior(key, candidate, prior_result)

    def store_final_result(
        key: str,
        candidate: QueryRecoveryCandidate,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "cache_key": key,
            "extraction_cache_key": candidate.task.cache_key,
            "review_policy": model_auto_check_review_policy(extractor),
            "query_row_attributes": canonical_extraction_row_attributes(
                candidate.task.entity.get("row_attributes")
            ),
            "masked_attribute_names": list(
                candidate.redundancy_group_attribute_names
            ),
            "attribute_name": candidate.recovery["recovered_attribute"][
                "column_name"
            ],
            "claimed_value": candidate.recovery["recovered_attribute"]["value"],
            "evidence_identity": query_recovery_remote_evidence_identity(
                candidate
            ),
            **result,
        }
        if model_auto_check_is_complete(
            {"auto_check": record.get("auto_check")}, required=True
        ):
            cache.put(key, record)
        else:
            cache.put_transient(key, record)
        resolved[key] = record
        return record

    for state in states:
        advance(state)

    try:
        while any(not state.done for state in states):
            phase, key, candidate, result, error = completions.get()
            if error is not None:
                raise error
            if result is None:
                raise RuntimeError("query recovery check returned no result")
            if phase == "local" and model_auto_check_has_remote_pending(
                {"auto_check": result.get("auto_check")}
            ):
                submit_external(key, candidate, result)
                continue

            record = store_final_result(key, candidate, result)
            in_flight.pop(key, None)
            candidate_waiters = waiters.pop(key, [])
            finish_check()
            for state in candidate_waiters:
                if state.awaiting_key != key:
                    continue
                state.awaiting_key = None
                if record.get("supported") and not exhaustive:
                    finish_row(state, supported=True)
                advance(state)
    finally:
        local_scheduler.close()
        external_executor.shutdown(wait=True)
        if progress_bar is not None:
            progress_bar.close()
    return resolved


def tasks_requiring_model_analysis(
    tasks: list[ExtractionTask],
    cache: ExtractionCache,
    args: argparse.Namespace,
    progress: ModelAnalysisProgress | None = None,
    extractor: Any | None = None,
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
            if cached_extraction_is_reusable(
                cached_record,
                args,
            ):
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
    on_local_phase_done: Callable[[], None] | None = None,
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
            if cached_extraction_is_reusable(
                cached_record,
                args,
            ):
                resolved_by_key[task.cache_key] = candidate_extraction_record(
                    cached_record
                )
                if progress is not None:
                    progress.mark(task.cache_key, "cached")
                continue
        uncached_by_key[task.cache_key] = task

    def store_model_record(
        cache_key: str,
        record: dict[str, Any],
        *,
        success_status: str = "model",
    ) -> None:
        record = candidate_extraction_record(record)
        resolved_by_key[cache_key] = record
        has_error = bool(clean_text(record.get("error")))
        if not has_error or getattr(args, "cache_failed_model_outputs", False):
            cache.put(cache_key, record)
        else:
            cache_put_transient(cache, cache_key, record)
        if has_error:
            append_model_error_record(getattr(args, "model_attribute_errors_path", ""), record)
        if progress is not None:
            progress.mark(
                cache_key,
                "error" if has_error else success_status,
            )

    if uncached_by_key and extractor is None:
        raise RuntimeError("Model analysis is required but no extractor was initialized")

    model_records = (
        run_uncached_extraction_tasks(
            extractor=extractor,
            tasks=list(uncached_by_key.values()),
            state=state,
            apply_auto_check_gate=False,
            on_record=store_model_record,
        )
        if uncached_by_key and extractor is not None
        else {}
    )
    for cache_key, record in model_records.items():
        if cache_key not in resolved_by_key:
            store_model_record(cache_key, record)
    if on_local_phase_done is not None:
        on_local_phase_done()

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
    source_to_split: dict[str, str] = {}
    for split, selected in split_keys.items():
        for key in selected:
            for source_id in groups[key]:
                source_to_split[source_id] = split
    splits: dict[str, Any] = {
        "split_key": (
            "page_title_or_source_table_id"
            if args.split_by == "page_title"
            else "source_table_id"
        ),
        "split_policy": "query_only",
        "data_lake_scope": "shared",
    }
    return splits, source_to_split


def choose_entity_column(
    table: dict[str, Any],
    *,
    min_linked_rows: int = 0,
    profiles: dict[int, dict[str, Any]] | None = None,
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
    if profiles is None:
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


def candidate_attribute_columns(
    table: dict[str, Any],
    entity_col: int,
    min_non_empty_ratio: float,
    *,
    profiles: dict[int, dict[str, Any]] | None = None,
) -> list[int]:
    if profiles is None:
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


def append_entity_url_column(
    query_rows: list[dict[str, Any]],
    source_table: dict[str, Any],
    entity_col: int,
) -> list[dict[str, Any]]:
    """Append a synthetic ``entity_url`` cell (derived from ``wiki_title``) to each query row.

    The URL is computed at build time from the entity cell's ``wiki_title`` field
    (never via network I/O): ``canonicalurl`` is not available at this stage, so we
    reuse the same fallback formula used for bridge asset URLs. Rows whose entity
    cell has no ``wiki_title`` (non-entity values) get an empty string, not an error.
    The new column is appended as an extra column positioned after the last
    projected column, with its own explicit ``column_name`` so downstream code
    never needs to reverse-lookup ``source_table["columns"]`` for it.
    """
    new_out_idx = 0
    for row in query_rows:
        cells = row.get("cells") or []
        new_out_idx = max(new_out_idx, len(cells))
    for row in query_rows:
        cells = row.get("cells") or []
        entity_cell = next(
            (cell for cell in cells if cell.get("source_column_index") == entity_col),
            None,
        )
        wiki_title = clean_text((entity_cell or {}).get("wiki_title"))
        if wiki_title:
            url = f"https://en.wikipedia.org/wiki/{quote(wiki_title.replace(' ', '_'))}"
        else:
            url = ""
        cells.append(
            {
                "column_index": new_out_idx,
                "source_column_index": -1,
                "column_name": "entity_url",
                "text": url,
                "synthetic": True,
            }
        )
        row["cells"] = cells
    return query_rows


def query_visible_row_attributes(
    query_row: dict[str, Any],
    *,
    entity_col: int,
) -> list[dict[str, Any]]:
    """Return only cells physically present in one materialized query row."""
    attributes: list[dict[str, Any]] = []
    for cell in query_row.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        name = sanitize_cell_text_for_model(cell.get("column_name"))
        value = sanitize_cell_text_for_model(cell.get("text"))
        if not name or not value:
            continue
        try:
            source_column_index = int(cell.get("source_column_index"))
        except (TypeError, ValueError):
            source_column_index = -1
        attributes.append(
            {
                "name": name,
                "value": value,
                "is_entity": source_column_index == entity_col,
            }
        )
    return attributes


def query_visible_recovery_candidates(
    candidates: Iterable[QueryRecoveryCandidate],
    *,
    query_rows: Iterable[dict[str, Any]],
    entity_col: int,
) -> list[QueryRecoveryCandidate]:
    """Bind recovery checks to their final query-row projection.

    Discovery tasks retain the complete source row for the high-recall first
    pass.  The independent auto-check must instead receive exactly the columns
    visible in the materialized query and no target-only source columns.
    """
    attributes_by_source_row = {
        int(row["source_row_id"]): query_visible_row_attributes(
            row,
            entity_col=entity_col,
        )
        for row in query_rows
    }
    visible_candidates: list[QueryRecoveryCandidate] = []
    for candidate in candidates:
        source_row_id = int(candidate.recovery["source_row_id"])
        if source_row_id not in attributes_by_source_row:
            continue
        blocked_names = {
            normalize(name)
            for name in candidate.redundancy_group_attribute_names
        }
        entity = {
            **candidate.task.entity,
            "row_attributes": [
                item
                for item in attributes_by_source_row[source_row_id]
                if normalize(item.get("name"))
                not in blocked_names
            ],
        }
        visible_candidates.append(
            QueryRecoveryCandidate(
                task=replace(candidate.task, entity=entity),
                extraction=candidate.extraction,
                recovery=candidate.recovery,
                redundancy_group_attribute_names=(
                    candidate.redundancy_group_attribute_names
                ),
            )
        )
    return visible_candidates


def table_record(
    *,
    table_id: str,
    role: str,
    split: str | None,
    source_table: dict[str, Any],
    column_indices: list[int],
    rows: list[dict[str, Any]],
    source_row_indices: list[int],
    extra: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": role,
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
    if split is not None:
        record["split"] = split
    return record


def raw_data_lake_record(source_table: dict[str, Any]) -> dict[str, Any]:
    source_table_id = clean_text(source_table.get("source_table_id"))
    if not source_table_id:
        raise ValueError("source table is missing source_table_id")
    table_id = f"dl_raw_{source_table_id}"
    return {
        "table_id": table_id,
        "object_id": table_id,
        "object_type": "table",
        "role": "raw_data_lake_table",
        "source_table_id": source_table_id,
        "source_table_ref": {
            "artifact": "source_tables",
            "source_table_id": source_table_id,
        },
        "queryable": False,
        "reason": "no_column_met_recovered_value_ratio",
    }


def exact_redundancy_groups(
    values_by_column: dict[int, Iterable[Any]],
    entity_col: int | None = None,
    *,
    value_serializer: Callable[[Any], Any] | None = None,
) -> list[list[int]]:
    """Return exact, row-aligned duplicate column groups.

    ``values_by_column`` should contain the values after the same serialization
    step used by the dataset projection; callers may instead provide that
    step as ``value_serializer`` (the WDC builder uses
    ``sanitize_cell_text_for_model``).  Values are hashed with row count,
        row index, and a length prefix so empty cells and concatenation
        boundaries remain part of the equivalence relation.  The optional
        ``entity_col`` is accepted for callers that want to document the
        selected entity column; detection itself intentionally includes entity
        columns so alias groups can be handled by the planner.
    """
    del entity_col  # Detection is table-local and includes entity aliases.
    buckets: dict[str, list[int]] = defaultdict(list)
    for raw_index, raw_values in values_by_column.items():
        try:
            column_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        values = (
            raw_values if hasattr(raw_values, "__len__") else list(raw_values)
        )
        digest = hashlib.sha256()
        digest.update(len(values).to_bytes(8, "big", signed=False))
        for row_index, value in enumerate(values):
            if value_serializer is not None:
                value = value_serializer(value)
            if value is None:
                text = ""
            elif isinstance(value, str):
                text = value
            else:
                text = str(value)
            encoded = text.encode("utf-8")
            digest.update(row_index.to_bytes(8, "big", signed=False))
            digest.update(len(encoded).to_bytes(8, "big", signed=False))
            digest.update(encoded)
        buckets[digest.hexdigest()].append(column_index)

    # Hash collisions are not expected for SHA-256, but keep deterministic
    # ordering and avoid exposing singleton buckets to the planner.
    groups = [
        sorted(indices) for indices in buckets.values() if len(indices) >= 2
    ]
    groups.sort(key=lambda members: members[0])
    return groups


def redundancy_group_map(
    values_by_column: dict[int, Iterable[Any]],
    groups: Iterable[Iterable[int]],
) -> dict[int, tuple[int, ...]]:
    """Map every column in a detected group to its sorted physical members."""
    result: dict[int, tuple[int, ...]] = {}
    for group in groups:
        members = tuple(sorted({int(index) for index in group}))
        if len(members) < 2:
            continue
        for index in members:
            if index in values_by_column:
                result[index] = members
    return result


def context_columns(
    table: dict[str, Any],
    excluded: set[int],
    limit: int,
    *,
    profiles: dict[int, dict[str, Any]] | None = None,
) -> list[int]:
    if profiles is None:
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


def balanced_context_partition(
    columns: list[int],
    *,
    seed: int,
    source_table_id: str,
    target_only_columns: Iterable[int] = (),
) -> tuple[list[int], list[int]]:
    """Split ordinary columns with the legacy seeded Gaussian ratio.

    ``target_only_columns`` are retained in the table pair but can never be
    exposed to the query.  This is used for name-equivalent answer aliases;
    exact value aliases are excluded earlier as members of the hidden join
    family itself.
    """
    shuffled = list(columns)
    rng = random.Random(f"context-pool-split:{seed}:{source_table_id}")
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    target_only = {int(index) for index in target_only_columns}
    forced_target = [index for index in shuffled if index in target_only]
    safe = [index for index in shuffled if index not in target_only]
    target_ratio = min(0.7, max(0.3, rng.gauss(0.5, 0.1)))
    target_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * target_ratio)),
    )
    target_safe_count = max(0, target_count - len(forced_target))
    target_context = [*forced_target, *safe[:target_safe_count]]
    query_context = safe[target_safe_count:]
    return query_context, target_context


def preliminary_context_partition(
    columns: list[int],
    *,
    seed: int,
    source_table_id: str,
) -> tuple[list[int], list[int]]:
    """Reproduce the legacy layout used by query-level auto-check keys.

    Candidate qualification historically used this seeded Gaussian split.
    Keep it stable so a presentation-only final re-layout does not invalidate
    completed auto-checks.  Changing this function requires an intentional
    auto-check cache migration.
    """
    return balanced_context_partition(
        columns,
        seed=seed,
        source_table_id=source_table_id,
    )


def _column_name_semantic_key(value: Any) -> str:
    """Canonicalize obvious header aliases for conservative leakage blocking."""
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    modifiers = {
        "a",
        "alias",
        "an",
        "description",
        "label",
        "name",
        "of",
        "the",
        "text",
        "value",
    }
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    canonical: list[str] = []
    for token in tokens:
        if token in modifiers:
            continue
        if token == "ranking":
            token = "rank"
        elif len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        canonical.append(token)
    if not canonical:
        canonical = tokens
    return " ".join(sorted(canonical))


def shuffled_target_columns(
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


def target_context_for_member(
    target_context: Iterable[int],
    *,
    seed: int,
    source_table_id: str,
    group_key: str,
    member_column_index: int,
    member_ordinal: int = 0,
    group_size: int = 1,
    excluded: Iterable[int] = (),
) -> list[int]:
    """Choose a stable member-specific target context.

    The candidate pool is already disjoint from the query context.  A stable
    per-member subset/shuffle avoids sharing one mutable list and yields
    different sibling contexts whenever the pool has at least two columns,
    without introducing any extra source scan or persistent state.
    """
    blocked = {int(index) for index in excluded}
    context = [int(index) for index in target_context if int(index) not in blocked]
    if group_size > 1 and len(context) > 1:
        # Keep at least one ordinary context column, while making sibling
        # targets use different subsets whenever the pool permits it.
        ordered = sorted(context)
        omit = int(member_ordinal) % len(ordered)
        context = [value for index, value in enumerate(ordered) if index != omit]
    random.Random(
        f"target-context:{seed}:{source_table_id}:{group_key}:{member_column_index}"
    ).shuffle(context)
    return context


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
    values_by_column: dict[int, list[str]] | None = None,
) -> list[int]:
    rows = source_table.get("rows", [])
    if not rows:
        return []
    query_rows_per_table = configured_query_rows_per_table(args)
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    min_non_empty_ratio = float(
        getattr(args, "min_column_non_empty_ratio", 0.5)
    )
    entity_values = (
        values_by_column.get(entity_col)
        if values_by_column is not None
        else None
    )
    candidates: list[int] = []
    for fallback, column in enumerate(source_table.get("columns", [])):
        try:
            column_index = int(column.get("column_index", fallback))
        except (AttributeError, TypeError, ValueError):
            continue
        if column_index == entity_col:
            continue
        join_values = (
            values_by_column.get(column_index)
            if values_by_column is not None
            else None
        )
        non_empty_join_rows = 0
        query_eligible_rows = 0
        for row_index, row in enumerate(rows):
            join_value = (
                join_values[row_index]
                if join_values is not None
                else get_cell_text(row, column_index)
            )
            if not join_value:
                continue
            non_empty_join_rows += 1
            entity_value = (
                entity_values[row_index]
                if entity_values is not None
                else get_cell_text(row, entity_col)
            )
            if entity_value:
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
    join_group_members: Iterable[int] | None = None,
) -> str:
    members = tuple(sorted({int(join_col), *(join_group_members or ())}))
    return f"explicit_candidate_{stable_hash(source_table_id, entity_col, *members)}"


def _redundancy_groups_for_table(
    source_table: dict[str, Any],
    values_by_column: dict[int, list[str]],
) -> tuple[list[list[int]], dict[int, tuple[int, ...]]]:
    groups = exact_redundancy_groups(
        values_by_column,
        value_serializer=sanitize_cell_text_for_model,
    )
    return groups, redundancy_group_map(values_by_column, groups)


def _explicit_join_context_partition(
    *,
    source_table: dict[str, Any],
    entity_col: int,
    join_columns: list[int],
    args: argparse.Namespace,
    profiles: dict[int, dict[str, Any]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
) -> tuple[list[int], list[int]]:
    """Partition ordinary columns once for a source's explicit variants."""
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    ordinary = context_columns(
        source_table,
        {entity_col, *join_columns},
        0,
        profiles=profiles,
    )
    ordinary = [
        column_index
        for column_index in ordinary
        if sum(
            bool(value)
            for value in (
                values_by_column.get(column_index, [])
                if values_by_column is not None
                else (
                    get_cell_text(source_row, column_index)
                    for source_row in source_table.get("rows", [])
                )
            )
        )
        >= min_target_rows
    ]
    if len(ordinary) == 1:
        return [], ordinary
    return balanced_context_partition(
        ordinary,
        seed=int(getattr(args, "seed", 13)),
        source_table_id=str(source_table["source_table_id"]),
    )


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
    values_by_column: dict[int, list[str]] | None = None,
    join_group_members: Iterable[int] | None = None,
    target_member_indices: Iterable[int] | None = None,
) -> dict[str, Any] | None:
    """Build one deterministic explicit query/target candidate specification."""
    seed = int(getattr(args, "seed", 13))
    source_table_id = str(source_table["source_table_id"])
    group_members = tuple(
        sorted({int(join_col), *(int(index) for index in (join_group_members or ()))})
    )
    selected_set = {
        int(index) for index in (target_member_indices or (join_col,))
    }
    selected_set.intersection_update(group_members)
    selected_set.add(join_col)
    selected_target_members = (
        join_col,
        *sorted(index for index in selected_set if index != join_col),
    )
    forbidden_group_columns = {entity_col, *group_members}
    query_context = [
        column_index
        for column_index in query_context
        if column_index not in forbidden_group_columns
    ]
    target_context = [
        column_index
        for column_index in target_context
        if column_index not in forbidden_group_columns
    ]
    query_cols = [entity_col, join_col, *query_context]
    target_cols = shuffled_target_columns(
        join_col,
        target_context,
        seed=seed,
        source_table_id=source_table_id,
    )
    source_rows = source_table.get("rows", [])
    indexed_source_rows = [
        (fallback, row_id(source_row, fallback), source_row)
        for fallback, source_row in enumerate(source_rows)
    ]
    entity_values = (
        values_by_column.get(entity_col)
        if values_by_column is not None
        else None
    )
    join_values = (
        values_by_column.get(join_col)
        if values_by_column is not None
        else None
    )
    eligible_source_rows = [
        source_row_id
        for row_index, source_row_id, source_row in indexed_source_rows
        if (
            entity_values[row_index]
            if entity_values is not None
            else get_cell_text(source_row, entity_col)
        )
        and (
            join_values[row_index]
            if join_values is not None
            else get_cell_text(source_row, join_col)
        )
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
    selected_source_row_set = set(selected_source_rows)
    query_source_rows = [
        source_row_id
        for row_index, source_row_id, source_row in indexed_source_rows
        if source_row_id in selected_source_row_set
        and (
            entity_values[row_index]
            if entity_values is not None
            else get_cell_text(source_row, entity_col)
        )
        and (
            join_values[row_index]
            if join_values is not None
            else get_cell_text(source_row, join_col)
        )
    ]
    target_source_rows = [
        source_row_id
        for row_index, source_row_id, source_row in indexed_source_rows
        if (
            join_values[row_index]
            if join_values is not None
            else get_cell_text(source_row, join_col)
        )
    ]
    min_target_rows = int(getattr(args, "min_rows_per_output_table", 2))
    if len(query_source_rows) != query_rows_per_table:
        return None
    if len(target_source_rows) < min_target_rows:
        return None
    if not set(query_source_rows).issubset(target_source_rows):
        return None

    join_col_name = get_column_name(source_table, join_col)
    group_key = stable_hash(
        "redundancy-group", source_table_id, *group_members, length=24
    )
    # Keep the candidate/query chain aligned with the mandatory visible target;
    # sibling targets receive their own physical-member chain at materialize.
    chain_id = f"chain_explicit_{stable_hash(source_table_id, group_key, join_col)}"
    query_table_id = f"query_{stable_hash(chain_id, 'query')}"
    target_table_id = f"target_{stable_hash(chain_id, selected_target_members[0], 'target')}"
    join_attribute = {
        "source_column_index": join_col,
        "column_name": join_col_name,
        "role": "visible_join_column",
        "hidden_in_query": False,
        "selected_rows": len(query_source_rows),
        "target_rows": len(target_source_rows),
    }
    return {
        "reason": "explicit_join_fallback",
        "candidate_id": _explicit_join_candidate_id(
            source_table_id, entity_col, join_col, group_members
        ),
        "source_table_id": source_table_id,
        "split": split,
        "entity_column_index": entity_col,
        "join_column_index": join_col,
        "join_column_name": join_col_name,
        "join_group_column_indices": list(group_members),
        "join_group_column_names": [
            get_column_name(source_table, index) for index in group_members
        ],
        "redundancy_group_key": group_key,
        "target_member_indices": list(selected_target_members),
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
    """Materialize one explicit group into one query and 1..k targets."""
    entity_col = int(candidate["entity_column_index"])
    join_col = int(candidate["join_column_index"])
    group_members = tuple(
        sorted(
            {
                int(value)
                for value in candidate.get(
                    "join_group_column_indices", [join_col]
                )
            }
        )
    )
    target_members = tuple(
        dict.fromkeys(
            int(value)
            for value in candidate.get("target_member_indices", [join_col])
            if int(value) in group_members
        )
    )
    if join_col not in target_members:
        target_members = (join_col, *target_members)
    query_context = [
        int(value) for value in candidate.get("query_context_column_indices", [])
    ]
    target_context = [
        int(value) for value in candidate.get("target_context_column_indices", [])
    ]
    query_cols = [entity_col, join_col, *query_context]
    query_rows, query_source_rows = project_selected_rows(
        source_table,
        query_cols,
        {int(value) for value in candidate["selected_source_row_ids"]},
        min_required_cols=2,
    )
    query_rows = append_entity_url_column(query_rows, source_table, entity_col)
    all_source_rows = {
        row_id(source_row, fallback)
        for fallback, source_row in enumerate(source_table.get("rows", []))
    }
    query_table_id = str(candidate["query_table_id"])
    source_table_id = str(source_table["source_table_id"])
    group_key = str(
        candidate.get("redundancy_group_key")
        or stable_hash("redundancy-group", source_table_id, *group_members, length=24)
    )
    query_chain_id = str(candidate["chain_id"])
    join_col_name = get_column_name(source_table, join_col)
    query_table = table_record(
        table_id=query_table_id,
        role="query",
        split=split,
        source_table=source_table,
        column_indices=query_cols,
        rows=query_rows,
        source_row_indices=query_source_rows,
        extra={
            "chain_id": query_chain_id,
            "chain_ids": [],
            "construction_type": "explicit_visible_join",
            "query_entity_col": entity_col,
            "query_entity_col_name": get_column_name(source_table, entity_col),
            "join_col": join_col,
            "join_col_name": join_col_name,
            "hidden_attributes": [],
            "target_table_ids": [],
            "query_context_col_names": [
                get_column_name(source_table, column_index)
                for column_index in query_context
            ],
            "row_view_index": 0,
        },
    )
    query_table["columns"] = [
        *query_table["columns"],
        {
            "column_index": len(query_table["columns"]),
            "source_column_index": -1,
            "column_name": "entity_url",
        },
    ]
    targets: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    for member_ordinal, member_index in enumerate(target_members):
        member_context = target_context_for_member(
            target_context,
            seed=int(getattr(args, "seed", 13)),
            source_table_id=source_table_id,
            group_key=group_key,
            member_column_index=member_index,
            member_ordinal=member_ordinal,
            group_size=len(target_members),
            excluded={entity_col, *group_members},
        )
        target_cols = shuffled_target_columns(
            member_index,
            member_context,
            seed=int(getattr(args, "seed", 13)),
            source_table_id=source_table_id,
        )
        target_rows, target_source_rows = project_selected_rows(
            source_table,
            target_cols,
            all_source_rows,
            min_required_cols=1,
        )
        if len(target_rows) < int(getattr(args, "min_rows_per_output_table", 2)):
            continue
        member_name = get_column_name(source_table, member_index)
        chain_id = (
            f"chain_explicit_{stable_hash(source_table_id, group_key, member_index)}"
        )
        target_table_id = (
            str(candidate["target_table_id"])
            if member_index == join_col
            else f"target_{stable_hash(chain_id, 'target')}"
        )
        target_table = table_record(
            table_id=target_table_id,
            role="target_data_lake_table",
            split=None,
            source_table=source_table,
            column_indices=target_cols,
            rows=target_rows,
            source_row_indices=target_source_rows,
            extra={
                "chain_id": chain_id,
                "construction_type": "explicit_visible_join",
                "queryable_source_table": True,
                "join_col": member_index,
                "join_col_name": member_name,
                "target_context_col_names": [
                    get_column_name(source_table, column_index)
                    for column_index in member_context
                ],
            },
        )
        target_attribute = {
            **dict(candidate.get("join_attribute") or {}),
            "source_column_index": member_index,
            "column_name": member_name,
            "selected_rows": len(query_rows),
            "target_rows": len(target_rows),
            "hidden_in_query": False,
        }
        targets.append(target_table)
        query_table["chain_ids"].append(chain_id)
        query_table["target_table_ids"].append(target_table_id)
        qrels.append(
            {
                "query_table_id": query_table_id,
                "target_table_id": target_table_id,
                "data_lake_table_id": target_table_id,
                "rel": 3,
                "split": split,
                "chain_id": chain_id,
                "row_view_index": 0,
                "source_table_id": source_table_id,
                "join_attribute": target_attribute,
                "reason": "explicit_visible_join_column",
            }
        )
    if not targets:
        return [], [], [], {
            "reason": "explicit_join_fallback",
            "entity_column_index": entity_col,
            "join_column_index": join_col,
            "join_column_name": join_col_name,
            "qualified_columns": [],
            "explicit_join_candidate": dict(candidate),
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
        "qualified_columns": [
            qrel["join_attribute"] for qrel in qrels
        ],
        "explicit_join_candidate": dict(candidate),
    }
    return [query_table], targets, qrels, decision


def build_explicit_join_fallback_records(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int | None,
    rejected_multimodal_reason: str,
    args: argparse.Namespace,
    rejected_multimodal_decision: dict[str, Any] | None = None,
    force: bool = False,
    profiles: dict[int, dict[str, Any]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
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

    if values_by_column is None:
        values_by_column = table_column_values(source_table)
    candidate_columns = _explicit_join_candidate_columns(
        source_table,
        entity_col,
        args,
        values_by_column,
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
        profiles=profiles,
        values_by_column=values_by_column,
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
    profiles: dict[int, dict[str, Any]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Build all viable explicit query candidates for one source table.

    Candidate join columns are initially excluded from the shared context
    partition.  After balancing, the selected subset is rebuilt together so
    unselected join candidates return to the ordinary context pool while the
    materialized sibling joins remain disjoint.
    """
    if entity_col is None or (
        not force
        and not _explicit_join_fallback_selected(source_table, args)
    ):
        return []
    if values_by_column is None:
        values_by_column = table_column_values(source_table)
    candidates = list(join_columns) if join_columns is not None else (
        _explicit_join_candidate_columns(
            source_table,
            entity_col,
            args,
            values_by_column,
        )
    )
    if not candidates:
        return []
    seed = int(getattr(args, "seed", 13))
    source_table_id = str(source_table["source_table_id"])
    _groups, group_by_column = _redundancy_groups_for_table(
        source_table, values_by_column
    )
    grouped_candidates: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for column_index in candidates:
        members = group_by_column.get(int(column_index), (int(column_index),))
        # Entity-containing groups may use a non-entity member for explicit
        # fallback, but the entity itself is never projected as a target.
        members = tuple(member for member in members if member != entity_col)
        if not members:
            continue
        grouped_candidates[members].append(int(column_index))
    candidates = []
    for members, physical_candidates in grouped_candidates.items():
        visible = min(
            physical_candidates,
            key=lambda column_index: (
                stable_hash(
                    "explicit-join-column",
                    seed,
                    source_table_id,
                    *members,
                    column_index,
                    length=40,
                ),
                column_index,
            ),
        )
        candidates.append(visible)
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
    all_join_members = {
        member
        for column_index in candidates
        for member in group_by_column.get(column_index, (column_index,))
        if member != entity_col
    }
    query_context, target_context = _explicit_join_context_partition(
        source_table=source_table,
        entity_col=entity_col,
        join_columns=sorted(all_join_members),
        args=args,
        profiles=profiles,
        values_by_column=values_by_column,
    )
    output: list[dict[str, Any]] = []
    for join_col in candidates:
        group_members = tuple(
            member
            for member in group_by_column.get(join_col, (join_col,))
            if member != entity_col
        )
        # Group fanout is selected once at candidate construction time.  The
        # visible member is mandatory; siblings are sampled without
        # replacement with a stable, completion-order-independent seed.
        fanout_rng = random.Random(
            f"explicit-target-fanout:{seed}:{split}:{source_table_id}:"
            f"{stable_hash('redundancy-group', source_table_id, *group_members, length=24)}"
        )
        fanout = fanout_rng.randint(1, len(group_members))
        sibling_order = list(group_members)
        sibling_order.remove(join_col)
        fanout_rng.shuffle(sibling_order)
        target_members = [join_col, *sibling_order[: max(0, fanout - 1)]]
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
            values_by_column=values_by_column,
            join_group_members=group_members,
            target_member_indices=target_members,
        )
        if candidate is not None:
            output.append(candidate)
    return output


def rebuild_selected_explicit_join_candidates(
    *,
    source_table: dict[str, Any],
    split: str,
    candidate_decisions: list[dict[str, Any]],
    args: argparse.Namespace,
    profiles: dict[int, dict[str, Any]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Repartition context around only the explicit joins selected to emit."""
    if not candidate_decisions:
        return []
    source_table_id = str(source_table["source_table_id"])
    entity_cols = {
        int(candidate["entity_column_index"])
        for candidate in candidate_decisions
    }
    if len(entity_cols) != 1:
        raise ValueError(
            "selected explicit candidates disagree on entity column: "
            f"{source_table_id}"
        )
    join_columns = [
        int(candidate["join_column_index"])
        for candidate in candidate_decisions
    ]
    if len(set(join_columns)) != len(join_columns):
        raise ValueError(
            f"selected explicit candidates repeat a join column: {source_table_id}"
        )
    if any(
        str(candidate.get("source_table_id")) != source_table_id
        for candidate in candidate_decisions
    ):
        raise ValueError(
            f"selected explicit candidate belongs to another source: {source_table_id}"
        )

    if values_by_column is None:
        values_by_column = table_column_values(source_table)
    entity_col = next(iter(entity_cols))
    all_join_columns = sorted(
        {
            member
            for candidate in candidate_decisions
            for member in candidate.get(
                "join_group_column_indices",
                [int(candidate["join_column_index"])],
            )
            if int(member) != entity_col
        }
    )
    query_context, target_context = _explicit_join_context_partition(
        source_table=source_table,
        entity_col=entity_col,
        join_columns=all_join_columns,
        args=args,
        profiles=profiles,
        values_by_column=values_by_column,
    )
    rebuilt: list[dict[str, Any]] = []
    for candidate in candidate_decisions:
        refreshed = _build_explicit_join_candidate(
            source_table=source_table,
            split=split,
            entity_col=entity_col,
            join_col=int(candidate["join_column_index"]),
            query_context=list(query_context),
            target_context=list(target_context),
            rejected_multimodal_reason=str(
                candidate.get("rejected_multimodal_reason") or ""
            ),
            rejected_multimodal_decision=(
                candidate.get("rejected_multimodal_decision")
                if isinstance(candidate.get("rejected_multimodal_decision"), dict)
                else None
            ),
            args=args,
            values_by_column=values_by_column,
            join_group_members=tuple(
                int(value)
                for value in candidate.get(
                    "join_group_column_indices", [int(candidate["join_column_index"])]
                )
                if int(value) != entity_col
            ),
            target_member_indices=candidate.get(
                "target_member_indices", [int(candidate["join_column_index"])]
            ),
        )
        if refreshed is None:
            raise ValueError(
                f"selected explicit candidate is no longer viable: {source_table_id}"
            )
        if refreshed["candidate_id"] != candidate.get("candidate_id"):
            raise ValueError(
                f"selected explicit candidate identity changed: {source_table_id}"
            )
        rebuilt.append({**candidate, **refreshed})
    return rebuilt


def rejected_table_join_records(
    *,
    source_table: dict[str, Any],
    split: str,
    entity_col: int | None,
    decision: dict[str, Any],
    args: argparse.Namespace,
    profiles: dict[int, dict[str, Any]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
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
            profiles=profiles,
            values_by_column=values_by_column,
        )
        if explicit_candidates:
            return (
                [],
                [raw_data_lake_record(source_table)],
                [],
                {
                    **decision,
                    "explicit_join_candidates": explicit_candidates,
                    # Keep the singular field for callers written against the
                    # pre-query-level candidate schema.
                    "explicit_join_candidate": explicit_candidates[0],
                },
            )
        return [], [raw_data_lake_record(source_table)], [], decision
    if mode == "disabled":
        return [], [raw_data_lake_record(source_table)], [], decision
    explicit_records = build_explicit_join_fallback_records(
        source_table=source_table,
        split=split,
        entity_col=entity_col,
        rejected_multimodal_reason=str(decision["reason"]),
        rejected_multimodal_decision=decision,
        args=args,
        profiles=profiles,
        values_by_column=values_by_column,
    )
    if explicit_records is not None:
        return explicit_records
    return [], [raw_data_lake_record(source_table)], [], decision


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
    profiles: dict[int, dict[str, Any]] | None = None,
    redundancy_groups: list[list[int]] | None = None,
    values_by_column: dict[int, list[str]] | None = None,
    final_survivor_layout: bool = True,
) -> list[tuple[dict[str, Any], list[int], list[int]]]:
    """Reserve two context columns and emit one variant per join family.

    Physical columns remain in ``member_column_indices`` for target fanout,
    while the returned list contains only one query-level variant per exact
    redundancy group.  Calls from older code/tests may omit ``redundancy_groups``;
    in that case the groups are derived from the source table values.
    """
    if not qualified_cols:
        return []

    if values_by_column is None:
        values_by_column = table_column_values(source_table)
    if redundancy_groups is None:
        # A few legacy callers pass a schema-only table to exercise context
        # layout.  There are no row values to compare in that case, so retain
        # the historical physical-column behavior.
        if source_table.get("rows"):
            redundancy_groups = exact_redundancy_groups(
                values_by_column,
                value_serializer=sanitize_cell_text_for_model,
            )
        else:
            redundancy_groups = []
    group_by_column = redundancy_group_map(values_by_column, redundancy_groups)

    # Build group-level representatives.  Singleton physical columns are
    # represented by a one-member family, so existing behavior is preserved.
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for qualified in qualified_cols:
        index = int(qualified["column_index"])
        members = group_by_column.get(index, (index,))
        grouped[members].append(qualified)

    group_variants: list[dict[str, Any]] = []
    for members, member_qualified in grouped.items():
        if entity_col in members:
            # Entity aliases cannot be hidden bridges.  They remain physical
            # source columns and can still be used by explicit fallback.
            continue
        representative = min(
            member_qualified,
            key=lambda item: (
                -float(item.get("recovered_value_ratio", 0.0)),
                int(item["column_index"]),
            ),
        )
        variant = dict(representative)
        variant["_redundancy_group_members"] = tuple(members)
        variant["_redundancy_group_key"] = stable_hash(
            "redundancy-group", source_table["source_table_id"], *members, length=24
        )
        variant["_redundancy_group_names"] = tuple(
            get_column_name(source_table, index) for index in members
        )
        group_variants.append(variant)

    ordered_qualified = sorted(
        group_variants,
        key=lambda item: (
            -float(item["recovered_value_ratio"]),
            int(item["column_index"]),
        ),
    )
    all_qualified_indices = {
        member
        for qualified in ordered_qualified
        for member in qualified["_redundancy_group_members"]
    }
    all_qualified_indices.update(
        member
        for members in group_by_column.values()
        if entity_col in members
        for member in members
    )
    max_variants = int(getattr(args, "max_query_tables_per_source_table", 0))
    emitted_qualified = list(ordered_qualified)
    if max_variants > 0:
        emitted_qualified = emitted_qualified[:max_variants]

    emitted_groups = {
        tuple(qualified["_redundancy_group_members"])
        for qualified in emitted_qualified
    }
    ordinary_contexts = context_columns(
        source_table,
        {entity_col, *all_qualified_indices},
        0,
        profiles=profiles,
    )
    # Prefer unselected and weakly recovered families as context while keeping
    # every member of an emitted family hidden from the query.
    for qualified in reversed(ordered_qualified):
        if len(ordinary_contexts) >= MIN_IMPLICIT_CONTEXT_COLUMNS:
            break
        members = tuple(qualified["_redundancy_group_members"])
        if members in emitted_groups:
            if len(emitted_groups) == 1:
                continue
            emitted_groups.remove(members)
            emitted_qualified = [
                item
                for item in emitted_qualified
                if tuple(item["_redundancy_group_members"]) != members
            ]
            ordinary_contexts.extend(
                member for member in members if member not in ordinary_contexts
            )
            continue
        ordinary_contexts.extend(
            member for member in members if member not in ordinary_contexts
        )
    if len(ordinary_contexts) < MIN_IMPLICIT_CONTEXT_COLUMNS:
        return []

    partition_kwargs = {
        "seed": int(getattr(args, "seed", 13)),
        "source_table_id": str(source_table["source_table_id"]),
    }
    if final_survivor_layout:
        hidden_name_keys = {
            _column_name_semantic_key(get_column_name(source_table, member))
            for qualified in emitted_qualified
            for member in qualified["_redundancy_group_members"]
        }
        target_only_contexts = [
            column_index
            for column_index in ordinary_contexts
            if _column_name_semantic_key(
                get_column_name(source_table, column_index)
            )
            in hidden_name_keys
        ]
        query_context, target_context = balanced_context_partition(
            ordinary_contexts,
            target_only_columns=target_only_contexts,
            **partition_kwargs,
        )
        if not query_context or not target_context:
            return []
    else:
        query_context, target_context = preliminary_context_partition(
            ordinary_contexts,
            **partition_kwargs,
        )
    return [
        (qualified, list(query_context), list(target_context))
        for qualified in emitted_qualified
    ]


def visible_query_fingerprint(
    *,
    source_table: dict[str, Any],
    query_cols: list[int],
    query_rows: list[dict[str, Any]],
) -> str:
    visible_payload = {
        "source_column_indices": query_cols,
        "column_names": [
            get_column_name(source_table, column_index)
            for column_index in query_cols
        ],
        "source_row_ids": [
            int(row["source_row_id"])
            for row in query_rows
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


def validate_implicit_query_uniqueness(
    qrels: Iterable[dict[str, Any]],
    *,
    expected_query_count: int | None = None,
) -> int:
    """Allow multiple positives per query while rejecting duplicate labels."""
    seen_queries: set[str] = set()
    seen_qrels: set[tuple[str, str, str]] = set()
    for qrel in qrels:
        if clean_text(qrel.get("reason")) != "model_recoverable_join_column":
            continue
        query_id = clean_text(qrel.get("query_table_id"))
        target_id = clean_text(
            qrel.get("target_table_id") or qrel.get("data_lake_table_id")
        )
        join_attribute = (
            qrel.get("join_attribute")
            if isinstance(qrel.get("join_attribute"), dict)
            else {}
        )
        attribute = clean_text(
            join_attribute.get("source_column_index")
            if join_attribute.get("source_column_index") is not None
            else join_attribute.get("column_name")
        )
        current = (query_id, target_id, attribute)
        if current in seen_qrels:
            raise ValueError(
                "duplicate implicit query qrel: "
                f"query_table_id={query_id!r}, target_table_id={target_id!r}, "
                f"attribute={attribute!r}"
            )
        seen_qrels.add(current)
        seen_queries.add(query_id)
    if (
        expected_query_count is not None
        and len(seen_queries) != expected_query_count
    ):
        raise ValueError(
            "implicit query/qrel count mismatch: "
            f"queries={expected_query_count}, qrel_queries={len(seen_queries)}"
        )
    return len(seen_queries)


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
    parsed = parse_json_object(raw_response)
    attributes = normalize_extracted_attributes(
        parsed.payload,
        candidate_attribute_names,
    )
    target_field = (
        "model_attributes"
        if isinstance(record.get("auto_check"), dict)
        else "attributes"
    )
    if attributes == record.get(target_field):
        return record, False
    updated = dict(record)
    updated[target_field] = attributes
    updated["reparsed_raw_response"] = True
    updated["raw_response_parse_method"] = parsed.method
    return updated, True


def should_refresh_cached_extraction(record: dict[str, Any]) -> bool:
    candidate_record = candidate_extraction_record(record)
    return (
        not clean_text(record.get("error"))
        and not candidate_record.get("attributes")
    )


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
        "auto_check_secondary_openai": bool(
            getattr(args, "auto_check_secondary_openai", False)
        ),
        "auto_check_luna_model": clean_text(
            getattr(args, "auto_check_openai_model", "")
        ),
        "auto_check_luna_reasoning_effort": clean_text(
            getattr(args, "auto_check_openai_reasoning_effort", "")
        ),
        "auto_check_openai_max_output_tokens": int(
            getattr(args, "auto_check_openai_max_output_tokens", 0) or 0
        ),
        "auto_check_openai_image_detail": clean_text(
            getattr(args, "auto_check_openai_image_detail", "")
        ),
        "auto_check_openai_image_max_pixels": int(
            getattr(args, "auto_check_openai_image_max_pixels", 0) or 0
        ),
        "auto_check_terra_model": clean_text(
            getattr(args, "auto_check_terra_model", "")
        ),
        "auto_check_terra_reasoning_effort": clean_text(
            getattr(args, "auto_check_terra_reasoning_effort", "")
        ),
        "auto_check_terra_max_output_tokens": int(
            getattr(args, "auto_check_terra_max_output_tokens", 0) or 0
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
                    else "Qwen3-VL-8B-Instruct",
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


def run_query_recovery_auto_check_round(
    *,
    plans: list[QueryRecoveryAutoCheckPlan],
    extractor: Any | None,
    cache: ExtractionCache,
    args: argparse.Namespace,
    concurrency_state: ModelConcurrencyState | None = None,
    exhaustive: bool = False,
) -> dict[str, dict[str, Any]]:
    """Keep local model services alive for one query-recovery check round."""
    if not auto_check_required(extractor) or not plans:
        return {}
    active_plans = [
        plan
        for plan in plans
        if query_recovery_plan_needs_model_check(
            plan, extractor, cache, exhaustive=exhaustive
        )
    ]
    if not active_plans:
        return {}

    pending_candidates: dict[str, QueryRecoveryCandidate] = {}
    for plan in active_plans:
        for candidate in plan.candidates:
            key = query_recovery_auto_check_key(candidate, extractor)
            if query_recovery_cached_check(
                key, cache, candidate, extractor=extractor
            ) is None:
                pending_candidates.setdefault(key, candidate)
    counts = Counter(
        model_kind_for_asset(candidate.task.asset)
        for candidate in pending_candidates.values()
    )
    round_counts = {
        "text": counts.get("text", 0),
        "image": counts.get("image", 0),
    }
    model_round = _begin_model_task_round(args, round_counts)
    round_status = "completed"
    try:
        return resolve_query_recovery_auto_check_plans(
            plans=active_plans,
            extractor=extractor,
            cache=cache,
            args=args,
            concurrency_state=concurrency_state,
            exhaustive=exhaustive,
        )
    except BaseException:
        round_status = "failed"
        raise
    finally:
        for kind in ("text", "image"):
            _write_model_round_event(
                model_round,
                f"{kind}.done",
                status=f"{kind}_round_tasks_completed",
                model_kind=kind,
                task_count=round_counts[kind],
            )
        _write_model_round_event(
            model_round,
            "done",
            status=f"model_round_{round_status}",
        )


def finalize_query_recovery_auto_checks(
    *,
    plans: list[QueryRecoveryAutoCheckPlan],
    extractor: Any | None,
    cache: ExtractionCache,
    args: argparse.Namespace,
    concurrency_state: ModelConcurrencyState | None = None,
) -> list[QueryRecoveryAutoCheckPlan]:
    """Resolve table eligibility, then exhaust evidence for accepted plans."""
    if not auto_check_required(extractor) or not plans:
        return list(plans)
    run_query_recovery_auto_check_round(
        plans=plans,
        extractor=extractor,
        cache=cache,
        args=args,
        concurrency_state=concurrency_state,
    )
    acceptance_cache = cache
    snapshot = getattr(cache, "snapshot", None)
    if callable(snapshot):
        snapshot_keys: list[str] = []
        include_remote_keys = (
            model_auto_check_review_policy(extractor)
            == AUTO_CHECK_REVIEW_POLICY_CASCADE
        )
        for plan in plans:
            for candidate in plan.candidates:
                snapshot_keys.append(
                    query_recovery_auto_check_key(candidate, extractor)
                )
                if include_remote_keys:
                    snapshot_keys.append(
                        query_recovery_remote_evidence_key(candidate)
                    )
        acceptance_cache = snapshot(snapshot_keys)
    accepted = [
        plan
        for plan in plans
        if query_recovery_plan_is_supported(
            plan,
            extractor,
            acceptance_cache,
        )
    ]
    incomplete = accepted
    for _attempt in range(
        max(1, int(getattr(args, "model_max_retries", 2)) + 1)
    ):
        run_query_recovery_auto_check_round(
            plans=incomplete,
            extractor=extractor,
            cache=cache,
            args=args,
            concurrency_state=concurrency_state,
            exhaustive=True,
        )
        incomplete = [
            plan
            for plan in incomplete
            if query_recovery_plan_needs_model_check(
                plan,
                extractor,
                cache,
                exhaustive=True,
            )
        ]
        if not incomplete:
            break
    if incomplete:
        raise TransientModelEndpointError(
            "accepted query evidence auto-check remained incomplete: "
            f"{incomplete[0].query_key}"
        )
    return accepted


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
            local_phase_events = {
                kind: threading.Event()
                for kind in active_groups
            }

            def mark_local_phase_done(kind: str, task_count: int) -> None:
                event = local_phase_events[kind]
                if event.is_set():
                    return
                event.set()
                if remote_borrower is not None:
                    remote_borrower.complete(kind)
                _write_model_round_event(
                    model_round,
                    f"{kind}.done",
                    status=f"{kind}_round_tasks_completed",
                    model_kind=kind,
                    task_count=task_count,
                )

            futures = {
                pool.submit(
                    resolve_extraction_tasks,
                    extractor=extractor,
                    cache=cache,
                    tasks=tasks,
                    args=args,
                    state=state,
                    progress=progress,
                    on_local_phase_done=(
                        lambda kind=kind, task_count=len(tasks):
                        mark_local_phase_done(kind, task_count)
                    ),
                ): (kind, len(tasks))
                for kind, tasks in active_groups.items()
            }
            for future in as_completed(futures):
                kind, task_count = futures[future]
                future.result()
                mark_local_phase_done(kind, task_count)
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
        if cached_extraction_is_reusable(
            cached_record,
            args,
        ):
            if progress is not None:
                progress.mark(cache_key, "cached")
            return candidate_extraction_record(cached_record)
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


def _build_table_join_records_once(
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
    query_auto_check_cache: ExtractionCache | None = None,
    apply_query_auto_check: bool = True,
    finalize_query_recoveries: bool = False,
    query_recovery_candidates_out: list[QueryRecoveryCandidate] | None = None,
    query_recovery_plans_out: list[QueryRecoveryAutoCheckPlan] | None = None,
    qualified_join_column_indices: set[int] | None = None,
    frozen_query_auto_checks: dict[str, dict[str, Any]] | None = None,
    final_survivor_layout: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_rows_per_table = configured_query_rows_per_table(args)
    values_by_column = table_column_values(source_table)
    redundancy_groups, group_by_column = _redundancy_groups_for_table(
        source_table, values_by_column
    )
    profiles = column_profiles(
        source_table,
        values_by_column=values_by_column,
    )
    entity_col = choose_entity_column(
        source_table,
        min_linked_rows=query_rows_per_table,
        profiles=profiles,
    )
    if entity_col is None:
        return rejected_table_join_records(
            source_table=source_table,
            split=split,
            entity_col=None,
            decision={"reason": "no_entity_column", "qualified_columns": []},
            args=args,
            profiles=profiles,
            values_by_column=values_by_column,
        )

    attribute_cols = candidate_attribute_columns(
        source_table,
        entity_col,
        args.min_column_non_empty_ratio,
        profiles=profiles,
    )
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
            profiles=profiles,
            values_by_column=values_by_column,
        )

    candidate_attribute_names = [get_column_name(source_table, col) for col in attribute_cols]
    valid_entity_source_rows: set[int] = set()
    valid_entity_source_row_order: list[int] = []
    recovered_rows_by_col: dict[int, set[int]] = defaultdict(set)
    recoveries_by_col: dict[int, list[QueryRecoveryCandidate]] = defaultdict(list)
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
            # Recovery and redundancy semantics are defined over the value
            # actually written into projected dataset cells.  Keep this in
            # lockstep with ``project_selected_rows`` so URL stripping and
            # other model-safe serialization cannot create a false mismatch.
            expected = sanitize_cell_text_for_model(
                get_cell_text(source_row, attr_col)
            )
            if not expected:
                continue
            predictions = attr_by_name.get(normalize(attr_name), [])
            matched_prediction = next(
                (
                    predicted
                    for predicted in predictions
                    if values_match(
                        predicted.get("value"),
                        expected,
                        attribute_name=attr_name,
                        entity_column_name=task.entity_column_name,
                    )
                ),
                None,
            )
            if matched_prediction is None:
                continue
            recovered_rows_by_col[attr_col].add(source_row_id)
            recovery = {
                "source_table_id": source_table["source_table_id"],
                "source_row_id": source_row_id,
                "split": split,
                "query_entity": entity,
                "recovered_attribute": {
                    "column_index": attr_col,
                    "column_name": attr_name,
                    "value": expected,
                    "model_value": clean_text(
                        matched_prediction.get("value")
                    ),
                    "hidden_in_query": True,
                },
                "evidence": {
                    "asset_id": asset["asset_id"],
                    "asset_type": asset.get("asset_type"),
                    **asset_preview(asset),
                    "model_evidence": clean_text(
                        matched_prediction.get("evidence")
                    ),
                    "model_connection_evidence": clean_text(
                        matched_prediction.get("connection_evidence")
                    ),
                    "extraction_cache_key": extraction.get("cache_key"),
                },
            }
            recoveries_by_col[attr_col].append(
                QueryRecoveryCandidate(
                    task=task,
                    extraction=extraction,
                    recovery=recovery,
                    redundancy_group_attribute_names=tuple(
                        get_column_name(source_table, member)
                        for member in group_by_column.get(attr_col, (attr_col,))
                    ),
                )
            )

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
            profiles=profiles,
            values_by_column=values_by_column,
        )

    preliminary_qualified_cols = qualified_cols
    if qualified_join_column_indices is not None:
        qualified_cols = [
            qualified
            for qualified in qualified_cols
            if int(qualified["column_index"])
            in qualified_join_column_indices
        ]
        if not qualified_cols:
            return rejected_table_join_records(
                source_table=source_table,
                split=split,
                entity_col=entity_col,
                decision={
                    "reason": "qualified_columns_failed_query_target_split",
                    "entity_column_index": entity_col,
                    "candidate_attribute_columns": candidate_attribute_names,
                    "attribute_extractions": extraction_count,
                    "qualified_columns": [],
                    "preliminary_qualified_columns": [
                        {
                            key: value
                            for key, value in qualified.items()
                            if not key.startswith("_")
                        }
                        for qualified in preliminary_qualified_cols
                    ],
                },
                args=args,
                profiles=profiles,
                values_by_column=values_by_column,
            )

    variant_layouts = multi_attribute_context_layout(
        source_table=source_table,
        entity_col=entity_col,
        qualified_cols=qualified_cols,
        args=args,
        profiles=profiles,
        redundancy_groups=redundancy_groups,
        values_by_column=values_by_column,
        final_survivor_layout=final_survivor_layout,
    )

    query_tables: list[dict[str, Any]] = []
    query_by_fingerprint: dict[str, dict[str, Any]] = {}
    data_lake_tables: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    qrel_keys: set[tuple[str, str, int]] = set()
    emitted_qualified_cols: list[dict[str, Any]] = []
    max_query_row_views = (
        configured_max_train_query_row_views_per_join(args)
        if split == "train"
        else 1
    )
    for qualified, query_context, target_context in variant_layouts:
        join_col = int(qualified["column_index"])
        group_members = tuple(
            int(value)
            for value in qualified.get(
                "_redundancy_group_members", (join_col,)
            )
        )
        group_key = str(
            qualified.get("_redundancy_group_key")
            or stable_hash(
                "redundancy-group",
                source_table["source_table_id"],
                *group_members,
                length=24,
            )
        )
        target_rng = random.Random(
            f"implicit-target-fanout:{getattr(args, 'seed', 13)}:{split}:"
            f"{source_table['source_table_id']}:{group_key}"
        )
        target_count = target_rng.randint(1, len(group_members))
        shuffled_members = list(group_members)
        target_rng.shuffle(shuffled_members)
        target_members = shuffled_members[:target_count]
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
        all_source_row_ids = {
            row_id(source_row, fallback)
            for fallback, source_row in enumerate(source_table.get("rows", []))
        }
        target_materializations: list[dict[str, Any]] = []
        for member_ordinal, member_index in enumerate(target_members):
            member_context = target_context_for_member(
                target_context,
                seed=int(getattr(args, "seed", 13)),
                source_table_id=str(source_table["source_table_id"]),
                group_key=group_key,
                member_column_index=member_index,
                member_ordinal=member_ordinal,
                group_size=len(target_members),
                excluded={entity_col, *group_members},
            )
            target_cols = shuffled_target_columns(
                member_index,
                member_context,
                seed=int(getattr(args, "seed", 13)),
                source_table_id=str(source_table["source_table_id"]),
            )
            target_rows, target_source_rows = project_selected_rows(
                source_table,
                target_cols,
                all_source_row_ids,
                min_required_cols=0,
            )
            if len(target_rows) < args.min_rows_per_output_table:
                continue
            chain_id = f"chain_{stable_hash(source_table['source_table_id'], group_key, member_index)}"
            target_table_id = f"target_{stable_hash(chain_id, 'target')}"
            target_materializations.append(
                {
                    "member_index": member_index,
                    "member_context": member_context,
                    "target_cols": target_cols,
                    "target_rows": target_rows,
                    "target_source_rows": target_source_rows,
                    "chain_id": chain_id,
                    "target_table_id": target_table_id,
                }
            )
        if not target_materializations:
            continue
        emitted_view_count = 0
        emitted_qualified = {
            **qualified,
            "selected_rows": query_rows_per_table,
            "target_rows": len(target_materializations[0]["target_rows"]),
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
            "target_rows": len(target_materializations[0]["target_rows"]),
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
            query_rows = append_entity_url_column(query_rows, source_table, entity_col)
            if not set(query_source_rows).issubset(target_source_rows):
                continue
            if len(query_rows) != query_rows_per_table:
                continue
            query_fingerprint = visible_query_fingerprint(
                source_table=source_table,
                query_cols=query_cols,
                query_rows=query_rows,
            )
            view_candidates = query_visible_recovery_candidates(
                (
                    candidate
                    for candidate in recoveries_by_col.get(join_col, [])
                    if int(candidate.recovery["source_row_id"])
                    in selected_source_row_set
                ),
                query_rows=query_rows,
                entity_col=entity_col,
            )
            if query_recovery_candidates_out is not None:
                query_recovery_candidates_out.extend(view_candidates)
            auto_check_plan = QueryRecoveryAutoCheckPlan(
                query_key="query_view_"
                + stable_hash(
                    source_table["source_table_id"],
                    query_fingerprint,
                    join_col,
                    length=24,
                ),
                required_recovered_rows=int(
                    qualified["required_recovered_rows"]
                ),
                source_row_order=tuple(selected_source_rows),
                candidates=tuple(view_candidates),
            )
            if query_recovery_plans_out is not None:
                query_recovery_plans_out.append(auto_check_plan)
            if frozen_query_auto_checks is not None:
                check_results = {}
                for candidate in view_candidates:
                    identity = query_recovery_candidate_identity(candidate)
                    record = frozen_query_auto_checks.get(identity)
                    if record is not None:
                        check_results[identity] = record
                approved_candidates = [
                    candidate
                    for candidate in view_candidates
                    if check_results.get(
                        query_recovery_candidate_identity(candidate), {}
                    ).get("supported")
                ]
            elif apply_query_auto_check and auto_check_required(extractor):
                if query_auto_check_cache is None:
                    raise RuntimeError(
                        "query recovery auto-check requires its dedicated cache"
                    )
                check_results = resolve_query_recovery_auto_checks(
                    candidates=view_candidates,
                    extractor=extractor,
                    cache=query_auto_check_cache,
                    args=args,
                    required_recovered_rows=(
                        auto_check_plan.required_recovered_rows
                    ),
                    source_row_order=auto_check_plan.source_row_order,
                )
                approved_candidates = [
                    candidate
                    for candidate in view_candidates
                    if check_results.get(
                        query_recovery_auto_check_key(candidate, extractor), {}
                    ).get("supported")
                ]
            else:
                check_results = {}
                approved_candidates = view_candidates
            approved_source_rows = {
                int(candidate.recovery["source_row_id"])
                for candidate in approved_candidates
            }
            if len(approved_source_rows) < int(
                qualified["required_recovered_rows"]
            ):
                continue
            final_check_results = check_results
            if finalize_query_recoveries:
                if frozen_query_auto_checks is not None:
                    final_check_results = check_results
                elif auto_check_required(extractor):
                    if query_auto_check_cache is None:
                        raise RuntimeError(
                            "final query recovery materialization requires the "
                            "query recovery cache"
                        )
                    final_check_results = {}
                    missing_check_keys: list[str] = []
                    for candidate in view_candidates:
                        key = query_recovery_auto_check_key(
                            candidate, extractor
                        )
                        record = query_recovery_cached_check(
                            key,
                            query_auto_check_cache,
                            candidate,
                            extractor=extractor,
                        )
                        if record is None:
                            missing_check_keys.append(key)
                            continue
                        final_check_results[key] = record
                    approved_candidates = [
                        candidate
                        for candidate in view_candidates
                        if final_check_results.get(
                            query_recovery_auto_check_key(
                                candidate, extractor
                            ),
                            {},
                        ).get("supported")
                    ]
                    approved_source_rows = {
                        int(candidate.recovery["source_row_id"])
                        for candidate in approved_candidates
                    }
                    if len(approved_source_rows) < int(
                        qualified["required_recovered_rows"]
                    ):
                        if missing_check_keys:
                            raise RuntimeError(
                                "final query evidence auto-check is incomplete: "
                                f"{auto_check_plan.query_key} "
                                f"{missing_check_keys[0]}"
                            )
                        continue
            query_table_id = (
                f"query_{stable_hash(source_table['source_table_id'], query_fingerprint)}"
            )
            query_table = query_by_fingerprint.get(query_fingerprint)
            if query_table is None:
                query_table = table_record(
                    table_id=query_table_id,
                    role="query",
                    split=split,
                    source_table=source_table,
                    column_indices=query_cols,
                    rows=query_rows,
                    source_row_indices=query_source_rows,
                    extra={
                        "chain_id": target_materializations[0]["chain_id"],
                        "chain_ids": [],
                        "query_entity_col": entity_col,
                        "query_entity_col_name": get_column_name(
                            source_table, entity_col
                        ),
                        "hidden_attributes": [],
                        "target_table_ids": [],
                        "query_context_col_names": [
                            get_column_name(source_table, col)
                            for col in query_context
                        ],
                        "row_view_index": row_view_index,
                    },
                )
                query_table["columns"] = [
                    *query_table["columns"],
                    {
                        "column_index": len(query_table["columns"]),
                        "source_column_index": -1,
                        "column_name": "entity_url",
                    },
                ]
                query_by_fingerprint[query_fingerprint] = query_table
                query_tables.append(query_table)

            source_to_query_row = {
                row["source_row_id"]: row["row_id"] for row in query_rows
            }
            view_emitted = False
            for target_materialization in target_materializations:
                member_index = int(target_materialization["member_index"])
                target_table_id = str(target_materialization["target_table_id"])
                target_rows = target_materialization["target_rows"]
                target_source_rows = target_materialization["target_source_rows"]
                member_name = get_column_name(source_table, member_index)
                view_hidden_attribute = {
                    **hidden_attribute,
                    "source_column_index": member_index,
                    "column_name": member_name,
                    "recovered_rows": len(approved_source_rows),
                    "recovered_value_ratio": len(approved_source_rows)
                    / query_rows_per_table,
                    "target_rows": len(target_rows),
                }
                qrel_key = (query_table_id, target_table_id, member_index)
                if qrel_key in qrel_keys:
                    continue
                qrel_keys.add(qrel_key)
                query_table["chain_ids"].append(
                    str(target_materialization["chain_id"])
                )
                query_table["hidden_attributes"].append(view_hidden_attribute)
                query_table["target_table_ids"].append(target_table_id)
                view_emitted = True
                qrels.append(
                    {
                        "query_table_id": query_table_id,
                        "target_table_id": target_table_id,
                        "data_lake_table_id": target_table_id,
                        "rel": 3,
                        "split": split,
                        "chain_id": str(target_materialization["chain_id"]),
                        "row_view_index": row_view_index,
                        "source_table_id": source_table["source_table_id"],
                        "join_attribute": view_hidden_attribute,
                        "reason": "model_recoverable_join_column",
                    }
                )
                source_to_target_rows: dict[int, list[int]] = defaultdict(list)
                for row in target_rows:
                    source_to_target_rows[int(row["source_row_id"])].append(
                        int(row["row_id"])
                    )
                seen_recoveries: set[str] = set()
                for candidate in approved_candidates:
                    recovery = candidate.recovery
                    source_row_id = int(recovery["source_row_id"])
                    if source_row_id not in source_to_query_row:
                        continue
                    source_row = source_rows_by_id.get(source_row_id)
                    member_value = (
                        sanitize_cell_text_for_model(
                            get_cell_text(source_row, member_index)
                        )
                        if source_row is not None
                        else sanitize_cell_text_for_model(
                            recovery["recovered_attribute"].get("value")
                        )
                    )
                    recovery_id = f"evrec_{stable_hash(query_table_id, target_table_id, source_row_id, recovery['evidence']['asset_id'], member_value)}"
                    if recovery_id in seen_recoveries:
                        continue
                    seen_recoveries.add(recovery_id)
                    path_id = f"path_{stable_hash(query_table_id, recovery['evidence']['asset_id'], target_table_id, source_row_id)}"
                    recovery_record = {
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
                            {"node_id": query_table_id, "node_type": "query_table"},
                            {
                                "node_id": recovery["evidence"]["asset_id"],
                                "node_type": f"{recovery['evidence']['asset_type']}_asset",
                            },
                            {"node_id": target_table_id, "node_type": "target_table"},
                        ],
                        **recovery,
                    }
                    recovery_record["recovered_attribute"] = {
                        **recovery["recovered_attribute"],
                        "column_index": member_index,
                        "column_name": member_name,
                        "value": member_value,
                    }
                    check_result_key = (
                        query_recovery_candidate_identity(candidate)
                        if frozen_query_auto_checks is not None
                        else query_recovery_auto_check_key(candidate, extractor)
                    )
                    check = final_check_results.get(
                        check_result_key, {}
                    ).get("auto_check")
                    if isinstance(check, dict):
                        recovery_record["auto_check"] = check
                    write_jsonl_record(recovery_writer, recovery_record)

            if not view_emitted:
                continue
            # ``row_views`` is a query-view count, not a qrel/target count.
            # A redundant group may fan out to several physical targets while
            # still contributing exactly one emitted view for this query.
            emitted_view_count += 1

        if emitted_view_count == 0:
            continue
        emitted_qualified["row_views"] = emitted_view_count
        emitted_qualified_cols.append(
            {
                key: value
                for key, value in emitted_qualified.items()
                if not key.startswith("_")
            }
        )
        for target_materialization in target_materializations:
            member_index = int(target_materialization["member_index"])
            data_lake_tables.append(
                table_record(
                    table_id=str(target_materialization["target_table_id"]),
                    role="target_data_lake_table",
                    split=None,
                    source_table=source_table,
                    column_indices=list(target_materialization["target_cols"]),
                    rows=target_materialization["target_rows"],
                    source_row_indices=target_materialization["target_source_rows"],
                    extra={
                        "chain_id": str(target_materialization["chain_id"]),
                        "queryable_source_table": True,
                        "join_col": member_index,
                        "join_col_name": get_column_name(source_table, member_index),
                        "target_context_col_names": [
                            get_column_name(source_table, col)
                            for col in target_materialization["member_context"]
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
                    {
                        key: value
                        for key, value in qualified.items()
                        if not key.startswith("_")
                    }
                    for qualified, _query_context, _target_context in variant_layouts
                ],
            },
            args=args,
            profiles=profiles,
            values_by_column=values_by_column,
        )
    validate_implicit_query_uniqueness(
        qrels,
        expected_query_count=len(query_tables),
    )
    return query_tables, data_lake_tables, qrels, {
        "reason": "queryable",
        "entity_column_index": entity_col,
        "attribute_extractions": extraction_count,
        "qualified_columns": emitted_qualified_cols,
    }


def _accepted_implicit_join_columns(decision: dict[str, Any]) -> set[int]:
    if clean_text(decision.get("reason")) != "queryable":
        return set()
    return {
        int(qualified["column_index"])
        for qualified in decision.get("qualified_columns", [])
        if isinstance(qualified, dict) and "column_index" in qualified
    }


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
    query_auto_check_cache: ExtractionCache | None = None,
    apply_query_auto_check: bool = True,
    finalize_query_recoveries: bool = False,
    query_recovery_candidates_out: list[QueryRecoveryCandidate] | None = None,
    query_recovery_plans_out: list[QueryRecoveryAutoCheckPlan] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Qualify implicit candidates, freeze verdicts, then lay out records.

    The preliminary layout defines the exact query-visible input reviewed by
    auto-check.  Once those verdicts determine the surviving join columns,
    their candidate identities and review records are frozen.  A final layout
    then returns rejected columns to the safe context pool without feeding that
    presentation change back into candidate qualification or model calls.
    """
    if not apply_query_auto_check or not auto_check_required(extractor):
        collecting_preliminary_plans = (
            not apply_query_auto_check
            and query_recovery_plans_out is not None
        )
        return _build_table_join_records_once(
            source_table=source_table,
            split=split,
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            extractor=extractor,
            cache=cache,
            progress=progress,
            concurrency_state=concurrency_state,
            extraction_writer=extraction_writer,
            recovery_writer=recovery_writer,
            args=args,
            query_auto_check_cache=query_auto_check_cache,
            apply_query_auto_check=apply_query_auto_check,
            finalize_query_recoveries=finalize_query_recoveries,
            query_recovery_candidates_out=query_recovery_candidates_out,
            query_recovery_plans_out=query_recovery_plans_out,
            final_survivor_layout=not collecting_preliminary_plans,
        )

    if query_auto_check_cache is None:
        raise RuntimeError(
            "query recovery auto-check requires its dedicated cache"
        )

    preliminary_candidates: list[QueryRecoveryCandidate] = []
    preliminary_plans: list[QueryRecoveryAutoCheckPlan] = []
    _queries, _targets, _qrels, preliminary_decision = (
        _build_table_join_records_once(
            source_table=source_table,
            split=split,
            assets=assets,
            entity_to_assets=entity_to_assets,
            wiki_to_entity_id=wiki_to_entity_id,
            extractor=extractor,
            cache=cache,
            progress=None,
            concurrency_state=concurrency_state,
            extraction_writer=ListRecordWriter(),
            recovery_writer=ListRecordWriter(),
            args=args,
            query_auto_check_cache=query_auto_check_cache,
            apply_query_auto_check=True,
            finalize_query_recoveries=False,
            query_recovery_candidates_out=preliminary_candidates,
            query_recovery_plans_out=preliminary_plans,
            final_survivor_layout=False,
        )
    )
    if query_recovery_plans_out is not None:
        query_recovery_plans_out.extend(preliminary_plans)
    if query_recovery_candidates_out is not None:
        query_recovery_candidates_out.extend(preliminary_candidates)

    accepted_columns = _accepted_implicit_join_columns(preliminary_decision)
    if finalize_query_recoveries and preliminary_plans:
        finalize_query_recovery_auto_checks(
            plans=preliminary_plans,
            extractor=extractor,
            cache=query_auto_check_cache,
            args=args,
            concurrency_state=concurrency_state,
        )

    frozen_checks: dict[str, dict[str, Any]] = {}
    for plan in preliminary_plans:
        for candidate in plan.candidates:
            record = query_recovery_cached_check(
                query_recovery_auto_check_key(candidate, extractor),
                query_auto_check_cache,
                candidate,
                extractor=extractor,
            )
            if record is not None:
                frozen_checks[
                    query_recovery_candidate_identity(candidate)
                ] = record

    return _build_table_join_records_once(
        source_table=source_table,
        split=split,
        assets=assets,
        entity_to_assets=entity_to_assets,
        wiki_to_entity_id=wiki_to_entity_id,
        extractor=extractor,
        cache=cache,
        progress=progress,
        concurrency_state=concurrency_state,
        extraction_writer=extraction_writer,
        recovery_writer=recovery_writer,
        args=args,
        query_auto_check_cache=query_auto_check_cache,
        apply_query_auto_check=True,
        finalize_query_recoveries=finalize_query_recoveries,
        qualified_join_column_indices=accepted_columns,
        frozen_query_auto_checks=frozen_checks,
        final_survivor_layout=True,
    )


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
    query_recovery_alias_stats: Counter[str] = Counter()

    def query_recovery_record_alias(record: dict[str, Any]) -> str | None:
        alias = query_recovery_auto_check_record_key(record)
        if query_recovery_remote_review_is_complete(record):
            if isinstance(record.get("evidence_identity"), dict):
                query_recovery_alias_stats["remote_native"] += 1
            else:
                query_recovery_alias_stats["remote_unmapped"] += 1
        else:
            query_recovery_alias_stats["local_model_specific"] += 1
        return alias

    query_auto_check_cache = ExtractionCache(
        cache_paths["query_recovery_auto_checks"],
        reuse=not args.no_reuse_model_cache,
        record_key_alias=query_recovery_record_alias,
    )
    if not args.no_reuse_model_cache:
        logging.info(
            "Query recovery cache aliases: remote_native=%d, "
            "remote_unmapped=%d, local_model_specific=%d",
            query_recovery_alias_stats["remote_native"],
            query_recovery_alias_stats["remote_unmapped"],
            query_recovery_alias_stats["local_model_specific"],
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
        query_auto_check_cache=query_auto_check_cache,
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

    splits, source_to_split = source_splits(source_split_records, args)
    query_table_counts = {"train": 0, "dev": 0, "test": 0}
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
            if key in cache.items
            and cached_extraction_is_reusable(
                cache.items[key],
                args,
            )
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
            text_tasks,
            cache,
            args,
            progress=progress,
        )
        pending_image_tasks = tasks_requiring_model_analysis(
            image_tasks,
            cache,
            args,
            progress=progress,
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
    if auto_check_required(extractor):
        final_query_auto_check_plans: list[QueryRecoveryAutoCheckPlan] = []
        for source_table in iter_jsonl_records(source_writer.paths()):
            source_table_id = str(source_table["source_table_id"])
            build_table_join_records(
                source_table=source_table,
                split=source_to_split.get(source_table_id, "test"),
                assets=assets,
                entity_to_assets=entity_to_assets,
                wiki_to_entity_id=wiki_to_entity_id,
                extractor=extractor,
                cache=cache,
                progress=None,
                concurrency_state=concurrency_state,
                extraction_writer=ListRecordWriter(),
                recovery_writer=ListRecordWriter(),
                args=args,
                query_auto_check_cache=query_auto_check_cache,
                apply_query_auto_check=False,
                query_recovery_plans_out=final_query_auto_check_plans,
            )
        finalize_query_recovery_auto_checks(
            plans=final_query_auto_check_plans,
            extractor=extractor,
            cache=query_auto_check_cache,
            args=args,
            concurrency_state=concurrency_state,
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
                    query_auto_check_cache=query_auto_check_cache,
                    finalize_query_recoveries=True,
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
                    query_table_counts[split] += 1
                for record in ([] if deferred_candidate else data_lake_tables):
                    write_jsonl_record(data_lake_handle, record)
                    data_lake_table_count += 1
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
                        record = raw_data_lake_record(source_table)
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
                        continue
                    decision_index = explicit_candidate_decision_indices[
                        selected_candidate_ids[0]
                    ]
                    original_decision = table_decisions[decision_index]
                    explicit_queries: list[dict[str, Any]] = []
                    explicit_targets: list[dict[str, Any]] = []
                    explicit_qrels: list[dict[str, Any]] = []
                    explicit_decisions: list[dict[str, Any]] = []
                    selected_candidate_decisions = [
                        next(
                            candidate
                            for candidate in original_decision[
                                "explicit_join_candidates"
                            ]
                            if candidate.get("candidate_id") == candidate_id
                        )
                        for candidate_id in selected_candidate_ids
                    ]
                    selected_candidates = rebuild_selected_explicit_join_candidates(
                        source_table=source_table,
                        split=split,
                        candidate_decisions=selected_candidate_decisions,
                        args=args,
                    )
                    for candidate_decision in selected_candidates:
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
                    explicit_decision = {
                        **original_decision,
                        **explicit_decisions[0],
                        "source_table_id": source_table_id,
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
                        query_table_counts[split] += 1
                    for record in explicit_targets:
                        write_jsonl_record(data_lake_handle, record)
                        data_lake_table_count += 1
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

    validate_implicit_query_uniqueness(
        qrels,
        expected_query_count=implicit_query_table_count,
    )
    qrels_count = write_jsonl(output_dir / "qrels.jsonl", qrels)
    write_jsonl(output_dir / "table_queryability_decisions.jsonl", table_decisions)
    splits["query_table_counts"] = query_table_counts
    splits["data_lake_table_count"] = data_lake_table_count
    splits["data_lake_artifact"] = "data_lake_tables"
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
        "model_auto_check": {
            **summarize_model_auto_check_records(recovery_writer.paths()),
            "current_process": model_auto_check_summary(extractor),
        },
        "model_concurrency": concurrency_state.summary(),
        "precomputed_text_model_cache_tasks": precomputed_text_task_count,
        "precomputed_image_model_cache_tasks": precomputed_image_task_count,
        "wikipedia_workers": 1,
        "min_recovered_value_ratio": args.min_recovered_value_ratio,
        "min_recovery_denominator": args.min_recovery_denominator,
        "min_implicit_context_columns": MIN_IMPLICIT_CONTEXT_COLUMNS,
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
            "wide source tables emit variants from the qualifying bridge attributes "
            "remaining after context allocation; "
            "ordinary columns form a shared source-level context pool, and the "
            "weakest qualifying bridge columns are demoted to context when needed "
            "to give the query and target at least one disjoint context column each",
            "qualified attributes with the same exact visible query row view are "
            "merged into one query with multiple positive targets",
            "generated target data-lake tables retain every source row after column projection; rejected source tables remain raw",
            "match_implicit deterministically selects one viable explicit join per implicit query within each split",
            "evidence_recoveries record query_table -> multimodal evidence -> target_table paths at entity/row/attribute granularity",
            "every local-positive evidence candidate for a final accepted query receives exhaustive local/Luna consensus review, with final-judge adjudication on disagreement, before evidence_recoveries are materialized",
            "evidence_recoveries contain supported paths only; omitted evidence is an implicit negative",
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
        "split_schema_version": "query-only-shared-data-lake-v1",
        "query_construction": {
            "split_policy": "query_only",
            "data_lake_scope": "shared",
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
            "min_implicit_context_columns": MIN_IMPLICIT_CONTEXT_COLUMNS,
            "max_query_tables_per_source_table": args.max_query_tables_per_source_table,
            "max_query_context_attrs": args.max_query_context_attrs,
            "max_target_context_attrs": args.max_target_context_attrs,
            "context_attr_limit_policy": "compatibility_flags_ignored",
            "context_partition_policy": (
                "source_level_seeded_gaussian_target_ratio_mean_0.5_"
                "std_0.1_clipped_0.3_0.7"
            ),
            "explicit_context_partition_scope": (
                "post_balance_selected_join_columns_only"
            ),
            "qualified_attribute_policy": (
                "recovery_qualified_variants_after_context_floor"
            ),
            "sibling_source_column_policy": (
                "exact_redundancy_groups_one_query_bridge_with_physical_target_fanout"
            ),
            "identical_visible_query_policy": (
                "merge_exact_row_view_with_all_distinct_positive_targets"
            ),
            "target_column_order_policy": (
                "independently_seeded_shuffle_per_join_column"
            ),
        },
        "source_sampling": source_sampling,
        "model_endpoints": {
            "model_endpoint_config": getattr(
                args, "model_endpoint_config", None
            ),
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
            "disable_thinking": True,
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
            "auto_check": stats["model_auto_check"],
        },
        "cache": {
            "root_dir": str(cache_paths["root_dir"]),
            "wikipedia_cache_dir": str(cache_paths["wikipedia_cache_dir"]),
            "wikipedia_image_dir": str(cache_paths["wikipedia_image_dir"]),
            "model_attribute_extractions": str(cache_paths["model_attribute_extractions"]),
            "query_recovery_auto_checks": str(
                cache_paths["query_recovery_auto_checks"]
            ),
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
    parser.add_argument(
        "--max_scanned_files",
        type=int,
        default=None,
        help=(
            "Optional deterministic prefix limit on EntiTables JSON files "
            "scanned before stable-hash sampling. Intended for smoke runs; "
            "the default scans the full input directory."
        ),
    )
    parser.add_argument(
        "--source_sample_checkpoint_dir",
        default=None,
        help=(
            "Directory for the validated global-sample reference and "
            "materialization checkpoint. Defaults to "
            "<output_dir>/_source_sample_checkpoint."
        ),
    )
    parser.add_argument(
        "--refresh_source_sample_checkpoint",
        action="store_true",
        help=(
            "Ignore reusable source-sample checkpoints and rebuild them from "
            "the current input files."
        ),
    )
    parser.add_argument(
        "--no_source_sample_checkpoint",
        action="store_true",
        help="Disable source-sample checkpoint reads and writes.",
    )
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
    parser.add_argument(
        "--max_query_context_attrs",
        type=int,
        default=1,
        help="Deprecated compatibility option; all query-pool columns are emitted.",
    )
    parser.add_argument(
        "--max_target_context_attrs",
        type=int,
        default=2,
        help="Deprecated compatibility option; all target-pool columns are emitted.",
    )
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
    parser.add_argument(
        "--model_endpoint_config",
        default=None,
        help=(
            "Optional mmdd-model-endpoints-v1 JSON config. It makes one "
            "served model authoritative across local/remote endpoints and "
            "enforces per-text, per-image, and shared total limits per URL."
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
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Instruct")
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
    parser.add_argument(
        "--enable_thinking",
        dest="disable_thinking",
        action="store_false",
        help=(
            "Deprecated and rejected: dataset construction always disables "
            "thinking for both text and image requests."
        ),
    )
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
    add_model_auto_check_arguments(parser)
    if configure_parser is not None:
        configure_parser(parser)
    parser.set_defaults(disable_thinking=True, reparse_cached_model_outputs=True)
    parser.set_defaults(model_progress=True)
    args = parser.parse_args(argv)
    if not args.disable_thinking:
        parser.error(
            "thinking mode cannot be enabled for dataset construction"
        )
    if min(
        args.text_model_workers,
        args.image_model_workers,
        args.remote_text_model_workers,
        args.remote_image_model_workers,
    ) < 0:
        parser.error("model worker counts must be non-negative")
    return args


def main() -> None:
    setup_logging()
    stats = build_dataset(parse_args())
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

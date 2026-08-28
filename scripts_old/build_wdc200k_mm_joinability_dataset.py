#!/usr/bin/env python
"""Run the staged WDC Schema.org 2023 200K dataset pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import build_mm_joinability_dataset as join_builder
import build_wdc_mm_joinability_dataset as legacy_wdc_builder
from gpu_priority_protocol import PriorityGpuOwner
from remote_vllm_layout import (
    ControllerConfig as RemoteLayoutControllerConfig,
    EndpointConfig as RemoteLayoutEndpointConfig,
    RemoteLayoutController,
)
from stage1_io import stable_hash
from wdc200k_assets import (
    AssetPlanShards,
    ImageBudget,
    ImageFetchResult,
    MaterializedAssetShards,
    UniqueImageJobs,
    asset_materialization_input_fingerprint,
    asset_planning_input_fingerprint,
    build_unique_image_jobs,
    fetch_unique_images,
    iter_entity_page_join,
    iter_image_outcomes,
    materialize_asset_shards,
    persist_entity_asset_plans,
    validate_asset_plan_shards,
    validate_complete_image_fetch,
    validate_materialized_asset_shards,
    validate_unique_image_jobs,
)
from wdc200k_archive import ArchiveResult, archive_pipeline_state
from wdc200k_fetch import (
    FetchPolicy,
    FetchResult,
    fetch_unique_pages,
    iter_page_fanout,
    iter_page_outcomes,
    reconcile_page_jobs_from_outcomes,
    validate_complete_page_fetch,
)
from wdc200k_io import (
    AtomicJsonlShard,
    CompletedShard,
    GuardedTextWriter,
    GuardedWriteTracker,
    PreWriteGuard,
    SqliteJobStore,
    validate_completed_shard,
)
from wdc200k_eta import (
    MAX_URL_COMPLETION_PUBLICATIONS,
    MAX_URL_EXECUTION_EPOCHS,
    MAX_URL_STAGE_SAMPLES,
    URL_TELEMETRY_SCHEMA_VERSION,
    UrlEtaEstimate,
    UrlProgressSnapshot,
    decode_histogram_blob,
    encode_histogram_blob,
    estimate_url_eta,
    url_estimator_metadata,
)
from wdc200k_materialize import (
    MaterializationInputs,
    MaterializationResult,
    load_certified_materialization_inputs,
    materialize_dataset,
)
from wdc200k_models import (
    AdaptedModelTasks,
    AssetStageBarrier,
    ModelProgressSnapshot,
    ModelStageAuthority,
    ModelStageResult,
    ModelTaskAdapterProgress,
    StructuralStageBarrier,
    adapt_model_tasks_from_manifests,
    enqueue_model_tasks,
    run_model_stage,
    validate_model_stage_for_adapter,
)
from wdc200k_runtime import resolve_work_dir
from wdc200k_sampling import (
    SamplingPolicy,
    SamplingResult,
    sample_structural_artifacts,
    validate_sampling_artifacts,
    validate_sampling_consumed_paths,
    validate_sampling_source_authority,
)
from wdc200k_selection import (
    ReserveExhaustedError,
    ReserveManager,
    SelectionPolicy,
    TableCandidate,
    run_selection,
)
from wdc200k_structural import (
    FinalizedSelectionResult,
    StructuralExpansionResult,
    expand_selected_shard,
    finalize_validated_selection,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional console enhancement.
    tqdm = None  # type: ignore[assignment]


STAGES = (
    "selection",
    "structural",
    "sampling",
    "pages",
    "asset_planning",
    "images",
    "models",
    "materialize",
)


class DiskSpaceInsufficientError(RuntimeError):
    """Raised before a stage could violate the configured disk reserve."""


class DiskGuard:
    """Check the filesystem containing an actual write target."""

    def __init__(
        self,
        reserve_bytes: int,
        *,
        usage_fn: Callable[[Path], Any] | None = None,
    ) -> None:
        if reserve_bytes < 0:
            raise ValueError("disk reserve must be non-negative")
        self.reserve_bytes = int(reserve_bytes)
        self.usage_fn = usage_fn or shutil.disk_usage

    @staticmethod
    def _existing_ancestor(path: Path) -> Path:
        probe = Path(path).resolve()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return probe

    def __call__(self, path: Path, estimated_bytes: int = 0) -> None:
        estimated = max(0, int(estimated_bytes))
        target = Path(path).resolve()
        probe = self._existing_ancestor(target)
        free = int(self.usage_fn(probe).free)
        required = self.reserve_bytes + estimated
        if free < required:
            raise DiskSpaceInsufficientError(
                "insufficient disk for target "
                f"{target}: free={free}, reserve={self.reserve_bytes}, "
                f"estimated={estimated}, required={required}"
            )


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    work_dir: Path
    cache_dir: Path
    max_source_tables: int = 200_000
    max_rows_per_source_table: None = None
    selection_seed: int = 13
    top100_policy: str = "bounded"
    top100_per_class: int = 10
    include_rest: bool = False
    min_candidate_rows: int = 5
    min_candidate_columns: int = 3
    minimum3_fraction: float = 1.0
    class_max_tables: int = 40_000
    web_max_retries: int = 0
    web_max_response_seconds: float = 8.0
    web_global_concurrency: int = 128
    web_per_host_concurrency: int = 2
    max_image_attempts_per_entity: int = 3
    max_images_per_entity: int = 3
    sampled_entities_per_table: int = 8
    sampled_entity_expansion_schedule: tuple[int, ...] = (12, 20, 25)
    sampling_expansion_round_index: int = 0
    entity_sampling_seed: int = 20260720
    global_entity_budget: int | None = None
    min_free_disk_bytes: int = 1_000_000_000
    selection_shard_tables: int = 100
    records_per_shard: int = 10_000
    web_max_page_bytes: int = 2_000_000
    web_max_image_bytes: int = 10_000_000
    estimated_page_result_bytes: int = 32_768
    estimated_image_result_bytes: int = 500_000
    progress_interval_seconds: float = 5.0
    text_asset_chunk_chars: int = 800
    min_text_asset_chunk_chars: int = 120
    max_text_asset_chunks_per_entity: int = 3
    split_by: str = "page_title"
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    min_column_non_empty_ratio: float = 0.5
    min_recovered_value_ratio: float = 0.6
    min_recovery_denominator: int = 2
    min_rows_per_output_table: int = 2
    query_rows_per_table: int = 5
    max_train_query_row_views_per_join: int = 5
    max_query_tables_per_source_table: int = 0
    max_query_context_attrs: int = 1
    max_target_context_attrs: int = 2
    explicit_join_fallback_mode: str = (
        join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE
    )
    explicit_join_fallback_ratio: float = (
        join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO
    )
    model_endpoint_config: str | None = None
    text_model_name: str = "Qwen3.5-9B"
    image_model_name: str = "Qwen3-VL-8B-Instruct"
    text_model_base_url: str = "http://localhost:8001/v1"
    text_model_base_urls: tuple[str, ...] = ()
    text_model_base_urls_file: str | None = None
    text_model_api_key: str | None = None
    remote_text_model_base_url: str | None = None
    remote_text_model_base_urls: tuple[str, ...] = ()
    remote_text_model_base_urls_file: str | None = None
    remote_text_model_api_key: str | None = None
    image_model_base_url: str = "http://localhost:8000/v1"
    image_model_base_urls: tuple[str, ...] = ()
    image_model_base_urls_file: str | None = None
    image_model_api_key: str | None = None
    remote_image_model_base_url: str | None = None
    remote_image_model_base_urls: tuple[str, ...] = ()
    remote_image_model_base_urls_file: str | None = None
    remote_image_model_api_key: str | None = None
    text_model_workers: int = 1
    image_model_workers: int = 1
    remote_text_model_workers: int = 0
    remote_image_model_workers: int = 0
    remote_layout_control_url: str | None = None
    remote_layout_control_token_file: Path | None = None
    remote_layout_routing_manifest: Path | None = None
    remote_layout_controller_id_file: Path | None = None
    remote_layout_lock_file: Path | None = None
    remote_layout_primary_image_url: str | None = None
    remote_layout_switchable_url: str | None = None
    remote_layout_lease_ttl_seconds: int = 15
    remote_layout_lease_renew_seconds: float = 5.0
    remote_layout_request_timeout_seconds: float = 3.0
    remote_layout_reconnect_timeout_seconds: float = 120.0
    remote_layout_operation_timeout_seconds: float = 1200.0
    remote_layout_drain_timeout_seconds: float = 300.0
    remote_layout_poll_seconds: float = 1.0
    remote_layout_workload_poll_seconds: float = 2.0
    remote_layout_stability_seconds: float = 10.0
    remote_layout_health_timeout_seconds: float = 3.0
    remote_layout_health_stable_polls: int = 2
    remote_layout_coordination_dir: Path | None = None
    remote_layout_reclaim_timeout_seconds: float = 90.0
    remote_layout_borrower_stale_seconds: float = 10.0
    remote_layout_unregistered_grace_seconds: float = 1.0
    remote_layout_coordination_poll_seconds: float = 0.2
    materialization_workers: int = 1
    materialization_validation_workers: int = 3
    run_fingerprint: str = ""
    runtime_dir: Path | None = None
    model_start_marker: Path | None = None
    model_ready_marker: Path | None = None
    model_ready_timeout_seconds: float | None = None
    model_endpoint_ready_timeout_seconds: float = 30.0
    model_timeout_seconds: float = 120.0
    model_max_retries: int = 2
    model_retry_sleep_seconds: float = 2.0
    model_cache_database_path: Path | None = None
    unrecoverable_replacement_rounds: int = 0
    unrecoverable_drop_probability: float = 0.5
    recovery_replacement_round_index: int = 0
    auto_check_secondary_openai: bool = True
    auto_check_api_config_file: str = ""
    auto_check_openai_env_file: str = ""
    auto_check_openai_model: str = join_builder.DEFAULT_AUTO_CHECK_OPENAI_MODEL
    auto_check_openai_base_url: str = ""
    auto_check_openai_api_key_env: str = "OPENAI_API_KEY"
    auto_check_openai_reasoning_effort: str = "none"
    auto_check_openai_verbosity: str = "low"
    auto_check_openai_max_output_tokens: int = 2048
    auto_check_terra_model: str = join_builder.DEFAULT_AUTO_CHECK_TERRA_MODEL
    auto_check_terra_reasoning_effort: str = "none"
    auto_check_terra_max_output_tokens: int = 2048
    auto_check_openai_max_inflight: int = (
        join_builder.MAX_AUTO_CHECK_OPENAI_CONCURRENCY
    )
    auto_check_openai_image_detail: str = "auto"
    auto_check_openai_image_max_pixels: int = (
        join_builder.DEFAULT_IMAGE_REQUEST_MAX_PIXELS
    )
    auto_check_openai_timeout_seconds: float = 180.0
    auto_check_openai_requests_per_minute: int = 0
    auto_check_openai_tokens_per_minute: int = 0
    model_text_done_marker: Path | None = None
    model_image_done_marker: Path | None = None
    refresh_page_cache: bool = False
    refresh_image_cache: bool = False
    dry_run: bool = False
    resume: bool = True
    from_stage: str | None = None
    stop_after: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        input_dir = Path(args.input_dir).resolve()
        output_dir = Path(args.output_dir).resolve()
        work_dir = resolve_work_dir(output_dir, args.work_dir)
        cache_dir = (
            Path(args.cache_dir).resolve()
            if args.cache_dir
            else output_dir.parent / "cache" / "wdc_webtable"
        )
        roots = (input_dir, output_dir, work_dir, cache_dir)
        if len(set(roots)) != len(roots):
            raise ValueError("input, work, cache, and output roots must be separate")
        for index, first in enumerate(roots):
            for second in roots[index + 1 :]:
                if first.is_relative_to(second) or second.is_relative_to(first):
                    raise ValueError(
                        "input, work, cache, and output roots must be separate"
                    )
        return cls(
            input_dir=input_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            cache_dir=cache_dir,
            max_source_tables=args.max_source_tables,
            selection_seed=args.selection_seed,
            top100_policy=args.top100_policy,
            top100_per_class=args.top100_per_class,
            include_rest=args.include_rest,
            min_candidate_rows=args.min_candidate_rows,
            min_candidate_columns=args.min_candidate_columns,
            minimum3_fraction=args.minimum3_fraction,
            class_max_tables=args.class_max_tables,
            web_max_retries=args.web_max_retries,
            web_max_response_seconds=args.web_max_response_seconds,
            web_global_concurrency=args.web_global_concurrency,
            web_per_host_concurrency=args.web_per_host_concurrency,
            max_image_attempts_per_entity=(
                args.max_image_attempts_per_entity
            ),
            max_images_per_entity=args.max_images_per_entity,
            sampled_entities_per_table=args.sampled_entities_per_table,
            sampled_entity_expansion_schedule=tuple(
                args.sampled_entity_expansion_schedule
            ),
            entity_sampling_seed=args.entity_sampling_seed,
            global_entity_budget=args.global_entity_budget,
            min_free_disk_bytes=args.min_free_disk_bytes,
            selection_shard_tables=args.selection_shard_tables,
            records_per_shard=args.records_per_shard,
            web_max_page_bytes=args.web_max_page_bytes,
            web_max_image_bytes=args.web_max_image_bytes,
            estimated_page_result_bytes=args.estimated_page_result_bytes,
            estimated_image_result_bytes=args.estimated_image_result_bytes,
            progress_interval_seconds=args.progress_interval_seconds,
            text_asset_chunk_chars=args.text_asset_chunk_chars,
            min_text_asset_chunk_chars=args.min_text_asset_chunk_chars,
            max_text_asset_chunks_per_entity=(
                args.max_text_asset_chunks_per_entity
            ),
            split_by=args.split_by,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            min_column_non_empty_ratio=args.min_column_non_empty_ratio,
            min_recovered_value_ratio=args.min_recovered_value_ratio,
            min_recovery_denominator=args.min_recovery_denominator,
            min_rows_per_output_table=args.min_rows_per_output_table,
            query_rows_per_table=args.query_rows_per_table,
            max_train_query_row_views_per_join=(
                args.max_train_query_row_views_per_join
            ),
            max_query_tables_per_source_table=(
                args.max_query_tables_per_source_table
            ),
            max_query_context_attrs=args.max_query_context_attrs,
            max_target_context_attrs=args.max_target_context_attrs,
            explicit_join_fallback_mode=(
                args.explicit_join_fallback_mode
            ),
            explicit_join_fallback_ratio=(
                args.explicit_join_fallback_ratio
            ),
            model_endpoint_config=(
                str(Path(args.model_endpoint_config).resolve())
                if args.model_endpoint_config
                else None
            ),
            text_model_name=args.text_model_name,
            image_model_name=args.image_model_name,
            text_model_base_url=args.text_model_base_url,
            text_model_base_urls=tuple(args.text_model_base_urls or ()),
            text_model_base_urls_file=(
                str(Path(args.text_model_base_urls_file).resolve())
                if args.text_model_base_urls_file
                else None
            ),
            text_model_api_key=args.text_model_api_key,
            remote_text_model_base_url=args.remote_text_model_base_url,
            remote_text_model_base_urls=tuple(
                args.remote_text_model_base_urls or ()
            ),
            remote_text_model_base_urls_file=(
                str(Path(args.remote_text_model_base_urls_file).resolve())
                if args.remote_text_model_base_urls_file
                else None
            ),
            remote_text_model_api_key=args.remote_text_model_api_key,
            image_model_base_url=args.image_model_base_url,
            image_model_base_urls=tuple(args.image_model_base_urls or ()),
            image_model_base_urls_file=(
                str(Path(args.image_model_base_urls_file).resolve())
                if args.image_model_base_urls_file
                else None
            ),
            image_model_api_key=args.image_model_api_key,
            remote_image_model_base_url=args.remote_image_model_base_url,
            remote_image_model_base_urls=tuple(
                args.remote_image_model_base_urls or ()
            ),
            remote_image_model_base_urls_file=(
                str(Path(args.remote_image_model_base_urls_file).resolve())
                if args.remote_image_model_base_urls_file
                else None
            ),
            remote_image_model_api_key=args.remote_image_model_api_key,
            text_model_workers=args.text_model_workers,
            image_model_workers=args.image_model_workers,
            remote_text_model_workers=args.remote_text_model_workers,
            remote_image_model_workers=args.remote_image_model_workers,
            remote_layout_control_url=(
                str(args.remote_layout_control_url).rstrip("/")
                if args.remote_layout_control_url
                else None
            ),
            remote_layout_control_token_file=(
                Path(args.remote_layout_control_token_file).resolve()
                if args.remote_layout_control_token_file
                else None
            ),
            remote_layout_routing_manifest=(
                Path(args.remote_layout_routing_manifest).resolve()
                if args.remote_layout_routing_manifest
                else work_dir / "runtime" / "model-routing.json"
            ),
            remote_layout_controller_id_file=(
                Path(args.remote_layout_controller_id_file).resolve()
                if args.remote_layout_controller_id_file
                else work_dir / "runtime" / "remote-layout-controller-id"
            ),
            remote_layout_lock_file=(
                Path(args.remote_layout_lock_file).resolve()
                if args.remote_layout_lock_file
                else work_dir / "runtime" / "remote-layout-controller.lock"
            ),
            remote_layout_primary_image_url=(
                str(args.remote_layout_primary_image_url).rstrip("/")
                if args.remote_layout_primary_image_url
                else None
            ),
            remote_layout_switchable_url=(
                str(args.remote_layout_switchable_url).rstrip("/")
                if args.remote_layout_switchable_url
                else None
            ),
            remote_layout_lease_ttl_seconds=args.remote_layout_lease_ttl_seconds,
            remote_layout_lease_renew_seconds=(
                args.remote_layout_lease_renew_seconds
            ),
            remote_layout_request_timeout_seconds=(
                args.remote_layout_request_timeout_seconds
            ),
            remote_layout_reconnect_timeout_seconds=(
                args.remote_layout_reconnect_timeout_seconds
            ),
            remote_layout_operation_timeout_seconds=(
                args.remote_layout_operation_timeout_seconds
            ),
            remote_layout_drain_timeout_seconds=(
                args.remote_layout_drain_timeout_seconds
            ),
            remote_layout_poll_seconds=args.remote_layout_poll_seconds,
            remote_layout_workload_poll_seconds=(
                args.remote_layout_workload_poll_seconds
            ),
            remote_layout_stability_seconds=(
                args.remote_layout_stability_seconds
            ),
            remote_layout_health_timeout_seconds=(
                args.remote_layout_health_timeout_seconds
            ),
            remote_layout_health_stable_polls=(
                args.remote_layout_health_stable_polls
            ),
            remote_layout_coordination_dir=(
                Path(args.remote_layout_coordination_dir).resolve()
                if args.remote_layout_coordination_dir
                else None
            ),
            remote_layout_reclaim_timeout_seconds=(
                args.remote_layout_reclaim_timeout_seconds
            ),
            remote_layout_borrower_stale_seconds=(
                args.remote_layout_borrower_stale_seconds
            ),
            remote_layout_unregistered_grace_seconds=(
                args.remote_layout_unregistered_grace_seconds
            ),
            remote_layout_coordination_poll_seconds=(
                args.remote_layout_coordination_poll_seconds
            ),
            materialization_workers=args.materialization_workers,
            materialization_validation_workers=(
                args.materialization_validation_workers
            ),
            run_fingerprint=args.run_fingerprint,
            runtime_dir=(
                Path(args.runtime_dir).resolve()
                if args.runtime_dir
                else work_dir / "runtime"
            ),
            model_start_marker=(
                Path(args.model_start_marker).resolve()
                if args.model_start_marker
                else None
            ),
            model_ready_marker=(
                Path(args.model_ready_marker).resolve()
                if args.model_ready_marker
                else None
            ),
            model_ready_timeout_seconds=args.model_ready_timeout_seconds,
            model_endpoint_ready_timeout_seconds=(
                args.model_endpoint_ready_timeout_seconds
            ),
            model_timeout_seconds=args.model_timeout_seconds,
            model_max_retries=args.model_max_retries,
            model_retry_sleep_seconds=args.model_retry_sleep_seconds,
            model_cache_database_path=(
                Path(args.model_cache_database_path).resolve()
                if args.model_cache_database_path
                else None
            ),
            unrecoverable_replacement_rounds=(
                args.unrecoverable_replacement_rounds
            ),
            unrecoverable_drop_probability=(
                args.unrecoverable_drop_probability
            ),
            auto_check_secondary_openai=args.auto_check_secondary_openai,
            auto_check_api_config_file=args.auto_check_api_config_file,
            auto_check_openai_env_file=args.auto_check_openai_env_file,
            auto_check_openai_model=args.auto_check_openai_model,
            auto_check_openai_base_url=args.auto_check_openai_base_url,
            auto_check_openai_api_key_env=args.auto_check_openai_api_key_env,
            auto_check_openai_reasoning_effort=(
                args.auto_check_openai_reasoning_effort
            ),
            auto_check_openai_verbosity=args.auto_check_openai_verbosity,
            auto_check_openai_max_output_tokens=(
                args.auto_check_openai_max_output_tokens
            ),
            auto_check_terra_model=args.auto_check_terra_model,
            auto_check_terra_reasoning_effort=(
                args.auto_check_terra_reasoning_effort
            ),
            auto_check_terra_max_output_tokens=(
                args.auto_check_terra_max_output_tokens
            ),
            auto_check_openai_max_inflight=(
                args.auto_check_openai_max_inflight
            ),
            auto_check_openai_image_detail=(
                args.auto_check_openai_image_detail
            ),
            auto_check_openai_image_max_pixels=(
                args.auto_check_openai_image_max_pixels
            ),
            auto_check_openai_timeout_seconds=(
                args.auto_check_openai_timeout_seconds
            ),
            auto_check_openai_requests_per_minute=(
                args.auto_check_openai_requests_per_minute
            ),
            auto_check_openai_tokens_per_minute=(
                args.auto_check_openai_tokens_per_minute
            ),
            model_text_done_marker=(
                Path(args.model_text_done_marker).resolve()
                if args.model_text_done_marker
                else None
            ),
            model_image_done_marker=(
                Path(args.model_image_done_marker).resolve()
                if args.model_image_done_marker
                else None
            ),
            refresh_page_cache=args.refresh_page_cache,
            refresh_image_cache=args.refresh_image_cache,
            dry_run=args.dry_run,
            resume=args.resume,
            from_stage=args.from_stage,
            stop_after=args.stop_after,
        )


@dataclass(frozen=True)
class PipelineResult:
    status: str
    stage: str | None
    statistics_archives: int
    counters: dict[str, int]
    output_manifest: Path | None = None


@dataclass(frozen=True)
class ActiveRecoverySelection:
    """Durable active WDC slot assignment for one recovery round."""

    round_index: int
    path: Path
    manifest_path: Path
    records: int


@dataclass(frozen=True)
class SamplingExpansionState:
    """Durable per-table evidence caps for progressive sampling."""

    round_index: int
    path: Path | None
    limits: dict[str, int]


def _model_jobs_path(config: PipelineConfig) -> Path:
    configured = config.model_cache_database_path
    if configured is not None:
        return Path(configured).resolve()
    return config.cache_dir / "model_cache" / "jobs.sqlite3"


def _promote_legacy_model_cache(
    config: PipelineConfig,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> Path:
    """Copy the old work-local SQLite cache before staged archival can move it."""
    target = _model_jobs_path(config)
    if config.model_cache_database_path is not None or target.is_file():
        return target
    legacy = config.work_dir / "model_outputs" / "jobs.sqlite3"
    if not legacy.is_file():
        return target
    if pre_write_guard is not None:
        pre_write_guard(target, legacy.stat().st_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / (
        f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    source = sqlite3.connect(
        legacy.resolve().as_uri() + "?mode=ro",
        uri=True,
    )
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination)
        destination.commit()
    except BaseException:
        destination.close()
        source.close()
        temporary.unlink(missing_ok=True)
        raise
    destination.close()
    source.close()
    if target.exists():
        temporary.unlink(missing_ok=True)
    else:
        temporary.replace(target)
    return target


def _recovery_root(config: PipelineConfig) -> Path:
    return config.work_dir / "recovery_replacement"


def _sampling_expansion_root(config: PipelineConfig) -> Path:
    return config.work_dir / "sampling_expansion"


def _sampling_expansion_policy_identity(config: PipelineConfig) -> str:
    # The schedule is deliberately excluded so a later invocation can append
    # a larger cap without invalidating already completed expansion rounds.
    return stable_hash(
        "wdc200k-progressive-sampling-policy-v1",
        config.sampled_entities_per_table,
        config.entity_sampling_seed,
        length=40,
    )


def _load_sampling_expansion_state(
    config: PipelineConfig,
) -> SamplingExpansionState:
    pointer_path = _sampling_expansion_root(config) / "current.json"
    if not pointer_path.is_file():
        return SamplingExpansionState(0, None, {})
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_version")
        != "wdc200k-sampling-expansion-pointer-v1"
        or pointer.get("policy_identity")
        != _sampling_expansion_policy_identity(config)
    ):
        raise ValueError("sampling expansion pointer identity mismatch")
    round_index = int(pointer.get("round_index", -1))
    if round_index <= 0:
        raise ValueError("sampling expansion round must be positive")
    state_path = Path(str(pointer.get("state_path") or "")).resolve()
    root = _sampling_expansion_root(config).resolve()
    if not state_path.is_relative_to(root) or not state_path.is_file():
        raise ValueError("sampling expansion state is outside the work root")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "wdc200k-sampling-expansion-state-v1"
        or payload.get("complete") is not True
        or payload.get("policy_identity")
        != _sampling_expansion_policy_identity(config)
        or int(payload.get("round_index", -1)) != round_index
        or not isinstance(payload.get("limits"), dict)
    ):
        raise ValueError("sampling expansion state is invalid")
    limits = {
        str(table_id): int(limit)
        for table_id, limit in payload["limits"].items()
    }
    if any(
        limit < config.sampled_entities_per_table
        for limit in limits.values()
    ):
        raise ValueError("sampling expansion limit is below initial cap")
    return SamplingExpansionState(round_index, state_path, limits)


def _write_sampling_expansion_state(
    config: PipelineConfig,
    *,
    previous: SamplingExpansionState,
    limits: Mapping[str, int],
    expanded_table_ids: Sequence[str],
    pre_write_guard: PreWriteGuard | None = None,
) -> SamplingExpansionState:
    round_index = previous.round_index + 1
    root = _sampling_expansion_root(config)
    state_path = root / f"round-{round_index:05d}.json"
    normalized = {
        str(table_id): int(limit)
        for table_id, limit in sorted(limits.items())
    }
    payload = {
        "schema_version": "wdc200k-sampling-expansion-state-v1",
        "policy_identity": _sampling_expansion_policy_identity(config),
        "round_index": round_index,
        "previous_state_path": (
            None if previous.path is None else str(previous.path.resolve())
        ),
        "expanded_source_table_ids": sorted(
            str(table_id) for table_id in expanded_table_ids
        ),
        "limits": normalized,
        "complete": True,
    }
    if state_path.is_file():
        if json.loads(state_path.read_text(encoding="utf-8")) != payload:
            raise ValueError("conflicting persisted sampling expansion round")
    else:
        _atomic_json(
            state_path,
            payload,
            pre_write_guard=pre_write_guard,
        )
    _atomic_json(
        root / "current.json",
        {
            "schema_version": "wdc200k-sampling-expansion-pointer-v1",
            "policy_identity": _sampling_expansion_policy_identity(config),
            "round_index": round_index,
            "state_path": str(state_path.resolve()),
        },
        pre_write_guard=pre_write_guard,
    )
    return SamplingExpansionState(round_index, state_path, normalized)


def _recovery_policy_identity(config: PipelineConfig) -> str:
    # The maximum number of rounds is deliberately excluded so a later run
    # can request more passes without invalidating already evaluated rounds.
    return stable_hash(
        "wdc200k-recovery-replacement-policy-v1",
        config.max_source_tables,
        config.selection_seed,
        config.unrecoverable_drop_probability,
        length=40,
    )


def _completed_shard_from_json(value: dict[str, Any]) -> CompletedShard:
    return CompletedShard(
        path=str(value["path"]),
        records=int(value["records"]),
        bytes=int(value["bytes"]),
        sha256=str(value["sha256"]),
    )


def _load_active_recovery_selection(
    config: PipelineConfig,
) -> ActiveRecoverySelection | None:
    pointer_path = _recovery_root(config) / "current.json"
    if not pointer_path.is_file():
        return None
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if (
        not isinstance(pointer, dict)
        or pointer.get("schema_version")
        != "wdc200k-recovery-selection-pointer-v1"
        or pointer.get("policy_identity")
        != _recovery_policy_identity(config)
    ):
        raise ValueError("recovery replacement pointer identity mismatch")
    round_index = int(pointer["round_index"])
    if round_index <= 0:
        raise ValueError("active recovery round must be positive")
    if round_index > config.unrecoverable_replacement_rounds:
        raise ValueError(
            "persisted recovery round exceeds configured replacement rounds"
        )
    manifest_path = Path(str(pointer["manifest_path"])).resolve()
    expected_root = _recovery_root(config).resolve()
    if not manifest_path.is_relative_to(expected_root):
        raise ValueError("active recovery manifest is outside the work root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = manifest.get("completed_shards") or []
    if (
        manifest.get("stage") != "wdc200k_selection"
        or manifest.get("schema_version")
        != "wdc200k-recovery-selection-v1"
        or manifest.get("complete") is not True
        or manifest.get("policy_identity")
        != _recovery_policy_identity(config)
        or int(manifest.get("round_index", -1)) != round_index
        or not isinstance(completed, list)
        or len(completed) != 1
    ):
        raise ValueError("active recovery selection manifest is invalid")
    shard = _completed_shard_from_json(completed[0])
    if (
        shard.records != config.max_source_tables
        or not validate_completed_shard(shard, manifest_path.parent)
    ):
        raise ValueError("active recovery selection shard is invalid")
    return ActiveRecoverySelection(
        round_index=round_index,
        path=manifest_path.parent / shard.path,
        manifest_path=manifest_path,
        records=shard.records,
    )


@dataclass
class _ProgressState:
    stage: str = "preflight"
    detail: str = "initializing"
    completed_shards: int = 0
    total_shards: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    known_work_bytes: int = 0
    known_cache_bytes: int = 0
    known_output_bytes: int = 0
    completed_units: int = 0
    total_units: int = 0
    rate_basis: str | None = None
    unit_baseline_completed: int = 0
    unit_baseline_at: float | None = None


class ProgressReporter:
    """Publish detailed JSON telemetry and readable console progress."""

    _ROLLING_WINDOW_SECONDS = 60.0
    _NON_TTY_CONSOLE_INTERVAL_SECONDS = 60.0
    _MAX_STAGE_SAMPLES = MAX_URL_STAGE_SAMPLES
    _MAX_LEGACY_STAGE_SAMPLES = 256
    _MAX_MIXED_STAGE_SAMPLES = _MAX_LEGACY_STAGE_SAMPLES + _MAX_STAGE_SAMPLES
    _MAX_COMPLETION_PUBLICATIONS = MAX_URL_COMPLETION_PUBLICATIONS
    _MAX_EXECUTION_EPOCHS = MAX_URL_EXECUTION_EPOCHS

    def __init__(
        self,
        config: PipelineConfig,
        *,
        pre_write_guard: PreWriteGuard | None = None,
    ) -> None:
        self.config = config
        self.path = config.work_dir / "progress.json"
        self._state = _ProgressState()
        self._rolling_samples: deque[tuple[float, int]] = deque()
        self._model_progress: dict[str, ModelProgressSnapshot] = {}
        self._model_samples: dict[
            str, deque[tuple[float, int]]
        ] = {}
        self._stage_bar: Any | None = None
        self._stage_bar_key: tuple[str, str] | None = None
        self._model_bars: dict[str, Any] = {}
        self._tty_line_length = 0
        self._last_console_at: float | None = None
        self._force_console = True
        self._stage_telemetry: dict[str, dict[str, Any]] = {}
        self._disk_roots: dict[str, dict[str, int]] = {}
        self._active_epochs: dict[str, tuple[str, float]] = {}
        self._logical_clock_offset = 0.0
        self._logical_time_floor = float("-inf")
        self._restore_stage_telemetry()
        self._pre_write_guard = pre_write_guard
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _eta_completion_summary(
        samples: Sequence[dict[str, Any]],
        *,
        total: int,
        completed_at: float,
    ) -> dict[str, Any]:
        eligible = 0
        excluded = 0
        factors: list[float] = []
        for sample in samples:
            completed = int(sample["completed_units"])
            if total > 0 and completed * 2 < total:
                continue
            predicted = sample["predicted_remaining_seconds"]
            actual = completed_at - float(sample["timestamp"])
            if predicted is None or float(predicted) <= 0 or actual <= 0:
                excluded += 1
                continue
            eligible += 1
            predicted_value = float(predicted)
            factors.append(
                max(predicted_value / actual, actual / predicted_value)
            )
        return {
            "eligible_final_half_samples": eligible,
            "excluded_final_half_samples": excluded,
            "max_symmetric_eta_factor": max(factors) if factors else None,
        }

    @staticmethod
    def _uint(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"progress {name} is not a non-negative integer")
        return value

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"progress {name} is not finite")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"progress {name} is not finite")
        return result

    @staticmethod
    def _estimate_fields(estimate: UrlEtaEstimate) -> dict[str, Any]:
        return {
            "durable_rate": estimate.durable_rate,
            "rate_eta": estimate.rate_eta,
            "queue_eta": estimate.queue_eta,
            "inflight_eta": estimate.inflight_eta,
            "commit_eta": estimate.commit_eta,
            "overflow_eta": estimate.overflow_eta,
            "predicted_remaining_seconds": (
                estimate.predicted_remaining_seconds
            ),
            "fallback": estimate.fallback,
        }

    @classmethod
    def _restore_v1_sample(cls, raw: dict[str, Any]) -> dict[str, Any]:
        try:
            cls._finite(raw["timestamp"], "v1 timestamp")
            completed = cls._uint(raw["completed_units"], "v1 completed")
            total = cls._uint(raw["total_units"], "v1 total")
            rate = cls._finite(raw["rate"], "v1 rate")
            rolling = cls._finite(raw["rolling_rate"], "v1 rolling rate")
            predicted_raw = raw["predicted_remaining_seconds"]
            predicted = (
                None
                if predicted_raw is None
                else cls._finite(predicted_raw, "v1 ETA")
            )
        except KeyError as error:
            raise ValueError("progress v1 sample is incomplete") from error
        if (
            total < completed
            or rate < 0.0
            or rolling < 0.0
            or (predicted is not None and predicted < 0.0)
        ):
            raise ValueError("progress v1 sample is invalid")
        return dict(raw)

    @classmethod
    def _restore_v2_sample(cls, raw: dict[str, Any]) -> tuple[
        dict[str, Any], UrlProgressSnapshot
    ]:
        if raw.get("telemetry_schema_version") != URL_TELEMETRY_SCHEMA_VERSION:
            raise ValueError("progress URL telemetry schema is invalid")
        try:
            transport, active, commit = decode_histogram_blob(
                raw["histogram_blob"]
            )
            snapshot = UrlProgressSnapshot(
                execution_epoch=raw["execution_epoch"],
                baseline_completed=raw["baseline_completed"],
                completed_durable=raw["completed_units"],
                total=raw["total_units"],
                local_buffered_not_started=raw[
                    "local_buffered_not_started"
                ],
                in_flight_jobs=raw["in_flight_jobs"],
                physical_in_flight=raw["physical_in_flight"],
                finished_not_durable=raw["finished_not_durable"],
                unobserved_nonlocal=raw["unobserved_nonlocal"],
                deadline_seconds=raw["deadline_seconds"],
                effective_concurrency=raw["effective_concurrency"],
                epoch_elapsed_seconds=raw["epoch_elapsed_seconds"],
                transport_event_histogram=transport,
                active_censor_histogram=active,
                commit_event_histogram=commit,
                transport_overflow_events=raw["transport_overflow_events"],
                active_overflow_censors=raw["active_overflow_censors"],
                commit_overflow_events=raw["commit_overflow_events"],
            )
            baseline_timestamp = cls._finite(
                raw["baseline_timestamp"], "baseline timestamp"
            )
            timestamp = cls._finite(raw["timestamp"], "sample timestamp")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("progress v2 sample is invalid") from error
        if timestamp != baseline_timestamp + snapshot.epoch_elapsed_seconds:
            raise ValueError("progress v2 logical timestamp is inconsistent")
        estimate = estimate_url_eta(snapshot)
        expected_fields = cls._estimate_fields(estimate)
        for name, expected in expected_fields.items():
            if name not in raw or raw[name] != expected:
                raise ValueError(f"progress v2 {name} is inconsistent")
        restored = dict(raw)
        restored.update(
            {
                "timestamp": timestamp,
                "baseline_timestamp": baseline_timestamp,
                **expected_fields,
            }
        )
        return restored, snapshot

    def _restore_disk(self, payload: dict[str, Any]) -> None:
        disk = payload.get("disk")
        if disk is None:
            return
        if not isinstance(disk, dict):
            raise ValueError("progress disk state is invalid")
        roots = disk.get("roots")
        if roots is None:
            return
        if not isinstance(roots, dict):
            raise ValueError("progress disk roots are invalid")
        if set(roots) != {"work", "cache", "output"}:
            raise ValueError("progress disk roots are incomplete")
        for name, raw in roots.items():
            if not isinstance(raw, dict):
                raise ValueError("progress disk root state is invalid")
            restored = {
                field: self._uint(raw.get(field), f"disk {name} {field}")
                for field in (
                    "start_bytes",
                    "peak_bytes",
                    "current_bytes",
                    "start_free_bytes",
                    "min_free_bytes",
                )
            }
            if (
                restored["peak_bytes"] < restored["start_bytes"]
                or restored["peak_bytes"] < restored["current_bytes"]
                or restored["min_free_bytes"] > restored["start_free_bytes"]
            ):
                raise ValueError("progress disk extrema are inconsistent")
            self._disk_roots[name] = restored
            setattr(
                self._state,
                f"known_{name}_bytes",
                restored["current_bytes"],
            )

    def _restore_stage_telemetry(self) -> None:
        """Restore bounded URL telemetry and recompute every v2 estimate."""
        if not self.config.resume or not self.path.is_file():
            return
        if self.path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("progress telemetry exceeds bounded state size")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("progress snapshot must be an object")
        self._restore_disk(payload)
        raw_stages = payload.get("stage_telemetry", {})
        if not isinstance(raw_stages, dict):
            raise ValueError("progress stage_telemetry must be an object")
        has_v2_telemetry = False
        marker_pairs: list[tuple[bool, bool]] = []
        for raw in raw_stages.values():
            if not isinstance(raw, dict):
                continue
            stage_has_v2 = "telemetry_schema_version" in raw
            if stage_has_v2:
                if raw["telemetry_schema_version"] != (
                    URL_TELEMETRY_SCHEMA_VERSION
                ):
                    raise ValueError("progress stage schema is invalid")
                has_v2_telemetry = True
            sample_has_v2 = False
            samples = raw.get("samples")
            if isinstance(samples, list):
                for sample in samples:
                    if not isinstance(sample, dict) or (
                        "telemetry_schema_version" not in sample
                    ):
                        continue
                    if sample["telemetry_schema_version"] != (
                        URL_TELEMETRY_SCHEMA_VERSION
                    ):
                        raise ValueError("progress sample schema is invalid")
                    sample_has_v2 = True
                    has_v2_telemetry = True
            marker_pairs.append((stage_has_v2, sample_has_v2))
        if has_v2_telemetry and set(self._disk_roots) != {
            "work",
            "cache",
            "output",
        }:
            raise ValueError("progress v2 disk roots are incomplete")
        if any(stage != sample for stage, sample in marker_pairs):
            raise ValueError("progress stage and sample schemas are inconsistent")
        invalidated: set[str] = set()
        if self.config.from_stage is not None:
            invalidated = set(STAGES[STAGES.index(self.config.from_stage) :])
        for stage, raw in raw_stages.items():
            if stage not in STAGES or stage in invalidated:
                continue
            if not isinstance(raw, dict):
                raise ValueError("progress stage telemetry must be an object")
            expected_basis = {"pages": "page_urls", "images": "image_urls"}.get(
                stage
            )
            rate_basis = raw.get("rate_basis")
            samples = raw.get("samples")
            if rate_basis != expected_basis:
                raise ValueError("progress rate_basis does not match stage")
            if (
                not isinstance(samples, list)
                or not samples
                or len(samples) > self._MAX_MIXED_STAGE_SAMPLES
            ):
                raise ValueError("progress stage samples are not bounded")

            restored_samples: list[dict[str, Any]] = []
            previous_completed = -1
            previous_timestamp = float("-inf")
            stage_total: int | None = None
            stage_deadline: float | None = None
            current_epoch: str | None = None
            seen_epochs: set[str] = set()
            epoch_baseline = -1
            epoch_baseline_timestamp = float("-inf")
            epoch_elapsed = -1.0
            epoch_concurrency = -1
            prior_transport: tuple[int, ...] | None = None
            prior_commit: tuple[int, ...] | None = None
            prior_transport_overflow: int | None = None
            prior_commit_overflow: int | None = None
            v2_started = False
            legacy_sample_count = 0
            v2_sample_count = 0
            completion_publication_count = 0

            for sample_raw in samples:
                if not isinstance(sample_raw, dict):
                    raise ValueError("progress stage sample is invalid")
                is_v2 = "telemetry_schema_version" in sample_raw
                if not is_v2:
                    if v2_started:
                        raise ValueError("progress v1 sample follows v2 telemetry")
                    legacy_sample_count += 1
                    if legacy_sample_count > self._MAX_LEGACY_STAGE_SAMPLES:
                        raise ValueError("progress legacy samples exceed 256")
                    sample = self._restore_v1_sample(sample_raw)
                    completed = int(sample["completed_units"])
                    total = int(sample["total_units"])
                    timestamp = float(sample["timestamp"])
                else:
                    v2_started = True
                    v2_sample_count += 1
                    if v2_sample_count > self._MAX_STAGE_SAMPLES:
                        raise ValueError("progress v2 samples exceed 256")
                    sample, snapshot = self._restore_v2_sample(sample_raw)
                    completed = snapshot.completed_durable
                    total = snapshot.total
                    timestamp = float(sample["timestamp"])
                    if stage_deadline is None:
                        stage_deadline = snapshot.deadline_seconds
                    elif snapshot.deadline_seconds != stage_deadline:
                        raise ValueError("progress v2 deadline changed")
                    epoch = snapshot.execution_epoch
                    if epoch != current_epoch:
                        if epoch in seen_epochs:
                            raise ValueError("progress v2 epoch is noncontiguous")
                        if len(seen_epochs) >= self._MAX_EXECUTION_EPOCHS:
                            raise ValueError(
                                "progress URL execution epochs exceed 32"
                            )
                        seen_epochs.add(epoch)
                        current_epoch = epoch
                        epoch_baseline = snapshot.baseline_completed
                        epoch_baseline_timestamp = float(
                            sample["baseline_timestamp"]
                        )
                        epoch_elapsed = snapshot.epoch_elapsed_seconds
                        epoch_concurrency = snapshot.effective_concurrency
                        prior_transport = snapshot.transport_event_histogram
                        prior_commit = snapshot.commit_event_histogram
                        prior_transport_overflow = (
                            snapshot.transport_overflow_events
                        )
                        prior_commit_overflow = snapshot.commit_overflow_events
                        if (
                            snapshot.epoch_elapsed_seconds != 0.0
                            or snapshot.completed_durable
                            != snapshot.baseline_completed
                            or any(snapshot.transport_event_histogram)
                            or any(snapshot.active_censor_histogram)
                            or any(snapshot.commit_event_histogram)
                            or snapshot.baseline_completed < previous_completed
                            or epoch_baseline_timestamp < previous_timestamp
                        ):
                            raise ValueError("progress v2 epoch baseline is invalid")
                    else:
                        completion_publication_count += 1
                        if (
                            completion_publication_count
                            > self._MAX_COMPLETION_PUBLICATIONS
                        ):
                            raise ValueError(
                                "progress completion publications exceed 224"
                            )
                        if (
                            snapshot.transport_overflow_events
                            < (prior_transport_overflow or 0)
                            or snapshot.commit_overflow_events
                            < (prior_commit_overflow or 0)
                        ):
                            raise ValueError(
                                "progress v2 cumulative overflow decreased"
                            )
                        if (
                            snapshot.baseline_completed != epoch_baseline
                            or float(sample["baseline_timestamp"])
                            != epoch_baseline_timestamp
                            or snapshot.epoch_elapsed_seconds < epoch_elapsed
                            or snapshot.effective_concurrency != epoch_concurrency
                            or any(
                                after < before
                                for before, after in zip(
                                    prior_transport or (),
                                    snapshot.transport_event_histogram,
                                )
                            )
                            or any(
                                after < before
                                for before, after in zip(
                                    prior_commit or (),
                                    snapshot.commit_event_histogram,
                                )
                            )
                        ):
                            raise ValueError("progress v2 epoch history is invalid")
                        epoch_elapsed = snapshot.epoch_elapsed_seconds
                        prior_transport = snapshot.transport_event_histogram
                        prior_commit = snapshot.commit_event_histogram
                        prior_transport_overflow = (
                            snapshot.transport_overflow_events
                        )
                        prior_commit_overflow = snapshot.commit_overflow_events

                if stage_total is None:
                    stage_total = total
                if (
                    total != stage_total
                    or completed < previous_completed
                    or timestamp < previous_timestamp
                ):
                    raise ValueError("progress stage samples are not monotonic")
                restored_samples.append(sample)
                previous_completed = completed
                previous_timestamp = timestamp
                self._logical_time_floor = max(
                    self._logical_time_floor, timestamp
                )

            last_sample = restored_samples[-1]
            restored_completed = self._uint(
                raw.get("completed_units"), "stage completed"
            )
            restored_total = self._uint(raw.get("total_units"), "stage total")
            if (
                restored_completed != int(last_sample["completed_units"])
                or restored_total != int(last_sample["total_units"])
            ):
                raise ValueError("progress stage summary is inconsistent")
            completed_at_raw = raw.get("completed_at")
            completed_at = (
                None
                if completed_at_raw is None
                else self._finite(completed_at_raw, "completed_at")
            )
            completed_at_authority: Any = completed_at
            if completed_at is None:
                if restored_completed == restored_total:
                    raise ValueError("progress completed stage lacks completed_at")
                expected_summary = {
                    "eligible_final_half_samples": 0,
                    "excluded_final_half_samples": 0,
                    "max_symmetric_eta_factor": None,
                }
                for key, expected in expected_summary.items():
                    if raw.get(key) != expected:
                        raise ValueError("progress ETA summary is inconsistent")
            elif v2_started:
                if (
                    restored_completed != restored_total
                    or completed_at < float(last_sample["timestamp"])
                    or completed_at != float(last_sample["timestamp"])
                ):
                    raise ValueError("progress completed_at is inconsistent")
                expected_summary = self._eta_completion_summary(
                    restored_samples,
                    total=restored_total,
                    completed_at=completed_at,
                )
                for key, expected in expected_summary.items():
                    if raw.get(key) != expected:
                        raise ValueError("progress ETA summary is inconsistent")
            else:
                if (
                    restored_completed != restored_total
                    or completed_at < float(last_sample["timestamp"])
                ):
                    raise ValueError("progress completed_at is inconsistent")
                eligible = self._uint(
                    raw.get("eligible_final_half_samples"),
                    "legacy eligible sample count",
                )
                excluded = self._uint(
                    raw.get("excluded_final_half_samples"),
                    "legacy excluded sample count",
                )
                factor_raw = raw.get("max_symmetric_eta_factor")
                factor = (
                    None
                    if factor_raw is None
                    else self._finite(factor_raw, "legacy ETA factor")
                )
                if (
                    (factor is not None and factor < 1.0)
                    or (eligible == 0) != (factor is None)
                ):
                    raise ValueError("progress legacy ETA summary is invalid")
                completed_at_authority = completed_at_raw
                expected_summary = {
                    "eligible_final_half_samples": eligible,
                    "excluded_final_half_samples": excluded,
                    "max_symmetric_eta_factor": factor_raw,
                }

            telemetry: dict[str, Any] = {
                "rate_basis": rate_basis,
                "samples": restored_samples,
                "completed_units": restored_completed,
                "total_units": restored_total,
                "completed_at": completed_at_authority,
                **expected_summary,
            }
            if v2_started:
                if raw.get("telemetry_schema_version") != (
                    URL_TELEMETRY_SCHEMA_VERSION
                ):
                    raise ValueError("progress stage schema is inconsistent")
                expected_estimator = url_estimator_metadata(stage_deadline)
                if raw.get("estimator") != expected_estimator:
                    raise ValueError("progress estimator metadata is inconsistent")
                telemetry.update(
                    telemetry_schema_version=URL_TELEMETRY_SCHEMA_VERSION,
                    estimator=expected_estimator,
                )
            self._stage_telemetry[stage] = telemetry
            if completed_at is not None:
                self._logical_time_floor = max(
                    self._logical_time_floor, completed_at
                )
        if math.isfinite(self._logical_time_floor):
            wall_now = time.time()
            if not math.isfinite(wall_now):
                raise ValueError("progress wall clock is invalid")
            self._logical_clock_offset = max(
                0.0, self._logical_time_floor - wall_now
            )

    def _progress_now_locked(self) -> float:
        wall_now = time.time()
        if not math.isfinite(wall_now):
            raise ValueError("progress wall clock is invalid")
        candidate = wall_now + self._logical_clock_offset
        if candidate < self._logical_time_floor:
            self._logical_clock_offset += self._logical_time_floor - candidate
            candidate = self._logical_time_floor
        self._logical_time_floor = candidate
        return candidate

    def start(self) -> None:
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.publish()
        self._thread = threading.Thread(
            target=self._run,
            name="wdc200k-progress",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.config.progress_interval_seconds):
            self.publish()

    def update(
        self,
        *,
        stage: str | None = None,
        detail: str | None = None,
        completed_shards: int | None = None,
        total_shards: int | None = None,
        counters: dict[str, int] | None = None,
        known_work_bytes: int | None = None,
        known_cache_bytes: int | None = None,
        known_output_bytes: int | None = None,
        url_snapshot: UrlProgressSnapshot | None = None,
    ) -> None:
        with self._lock:
            if stage is not None and stage != self._state.stage:
                if self._state.stage == "models":
                    self._close_model_bars()
                self._complete_unit_stage_locked()
                self._state.stage = stage
                self._state.detail = ""
                self._state.stage_started_at = time.time()
                self._rolling_samples.clear()
                self._state.completed_units = 0
                self._state.total_units = 0
                self._state.rate_basis = None
                if stage == "models":
                    self._model_progress.clear()
                    self._model_samples.clear()
                self._force_console = True
            if detail is not None:
                self._state.detail = str(detail)
            if completed_shards is not None:
                self._state.completed_shards = completed_shards
            if total_shards is not None:
                self._state.total_shards = total_shards
            if counters:
                self._state.counters.update(
                    {key: int(value) for key, value in counters.items()}
                )
            if known_work_bytes is not None:
                self._state.known_work_bytes = int(known_work_bytes)
            if known_cache_bytes is not None:
                self._state.known_cache_bytes = int(known_cache_bytes)
            if known_output_bytes is not None:
                self._state.known_output_bytes = int(known_output_bytes)
            if url_snapshot is not None:
                self._update_url_locked(url_snapshot)

    def update_model_progress(self, snapshot: ModelProgressSnapshot) -> None:
        """Record transient console-only model progress from durable events."""
        if not isinstance(snapshot, ModelProgressSnapshot):
            raise ValueError("model progress snapshot has an invalid type")
        if snapshot.modality not in {"text", "image"}:
            raise ValueError("model progress modality is invalid")
        values = (
            snapshot.total,
            snapshot.success,
            snapshot.terminal,
            snapshot.leased,
            snapshot.pending,
        )
        if any(isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("model progress counts must be non-negative")
        if snapshot.total != sum(values[1:]):
            raise ValueError("model progress counts do not sum to total")
        with self._lock:
            previous = self._model_progress.get(snapshot.modality)
            if previous is not None and (
                snapshot.total != previous.total
                or snapshot.completed < previous.completed
            ):
                raise ValueError("model progress is not monotonic")
            if previous is None:
                self._model_samples[snapshot.modality] = deque()
                self._force_console = True
            self._model_progress[snapshot.modality] = snapshot
            now = time.monotonic()
            samples = self._model_samples[snapshot.modality]
            samples.append((now, snapshot.completed))
            if snapshot.completed == snapshot.total:
                self._force_console = True
            cutoff = now - self._ROLLING_WINDOW_SECONDS
            while len(samples) > 1 and samples[1][0] <= cutoff:
                samples.popleft()

    def _current_stage_samples_locked(self) -> list[dict[str, Any]]:
        telemetry = self._stage_telemetry.get(self._state.stage)
        if telemetry is None:
            return []
        return telemetry["samples"]

    def _unit_rates_locked(self, now: float) -> tuple[float, float]:
        del now
        samples = self._current_stage_samples_locked()
        if not samples:
            return 0.0, 0.0
        rate = samples[-1].get("durable_rate")
        if rate is None:
            rate = samples[-1].get("rate", 0.0)
        return float(rate or 0.0), float(rate or 0.0)

    @staticmethod
    def _v2_sample(
        snapshot: UrlProgressSnapshot,
        *,
        baseline_timestamp: float,
    ) -> dict[str, Any]:
        estimate = estimate_url_eta(snapshot)
        return {
            "telemetry_schema_version": URL_TELEMETRY_SCHEMA_VERSION,
            "timestamp": baseline_timestamp + snapshot.epoch_elapsed_seconds,
            "execution_epoch": snapshot.execution_epoch,
            "baseline_completed": snapshot.baseline_completed,
            "baseline_timestamp": baseline_timestamp,
            "epoch_elapsed_seconds": snapshot.epoch_elapsed_seconds,
            "completed_units": snapshot.completed_durable,
            "total_units": snapshot.total,
            "local_buffered_not_started": snapshot.local_buffered_not_started,
            "in_flight_jobs": snapshot.in_flight_jobs,
            "physical_in_flight": snapshot.physical_in_flight,
            "finished_not_durable": snapshot.finished_not_durable,
            "unobserved_nonlocal": snapshot.unobserved_nonlocal,
            "deadline_seconds": snapshot.deadline_seconds,
            "effective_concurrency": snapshot.effective_concurrency,
            "transport_overflow_events": snapshot.transport_overflow_events,
            "active_overflow_censors": snapshot.active_overflow_censors,
            "commit_overflow_events": snapshot.commit_overflow_events,
            "histogram_blob": encode_histogram_blob(snapshot),
            **ProgressReporter._estimate_fields(estimate),
        }

    def _update_url_locked(self, snapshot: UrlProgressSnapshot) -> None:
        if not isinstance(snapshot, UrlProgressSnapshot):
            raise ValueError("url_snapshot must be UrlProgressSnapshot")
        stage = self._state.stage
        rate_basis = {"pages": "page_urls", "images": "image_urls"}.get(stage)
        if rate_basis is None:
            raise ValueError("URL progress belongs to pages or images")
        telemetry = self._stage_telemetry.get(stage)
        if telemetry is not None and telemetry.get("completed_at") is not None:
            if not (
                snapshot.epoch_elapsed_seconds == 0.0
                and snapshot.baseline_completed
                == snapshot.completed_durable
                == snapshot.total
                == int(telemetry["total_units"])
                and not any(snapshot.transport_event_histogram)
                and not any(snapshot.active_censor_histogram)
                and not any(snapshot.commit_event_histogram)
            ):
                raise ValueError("progress URL stage is already complete")
            self._state.completed_units = snapshot.completed_durable
            self._state.total_units = snapshot.total
            self._state.rate_basis = rate_basis
            return
        if (
            telemetry is not None
            and stage not in self._active_epochs
            and telemetry.get("completed_at") is None
            and snapshot.total != int(telemetry["total_units"])
            and int(telemetry["completed_units"]) == 0
            and all(
                int(sample["completed_units"]) == 0
                for sample in telemetry["samples"]
            )
            and snapshot.epoch_elapsed_seconds == 0.0
            and snapshot.baseline_completed
            == snapshot.completed_durable
            == 0
            and snapshot.local_buffered_not_started == 0
            and snapshot.in_flight_jobs == 0
            and snapshot.physical_in_flight == 0
            and snapshot.finished_not_durable == 0
            and snapshot.unobserved_nonlocal == snapshot.total
            and not any(snapshot.transport_event_histogram)
            and not any(snapshot.active_censor_histogram)
            and not any(snapshot.commit_event_histogram)
        ):
            # A previous run may have persisted the initial URL telemetry
            # before its authoritative job scope was replaced (for example,
            # full-corpus jobs superseded by sampled jobs). With no durable
            # work in either epoch, retaining that obsolete scope is unsafe
            # and provides no progress history worth preserving.
            self._stage_telemetry.pop(stage)
            telemetry = None
        samples = [] if telemetry is None else telemetry["samples"]
        v2_samples = [
            sample
            for sample in samples
            if "telemetry_schema_version" in sample
        ]
        if len(v2_samples) >= self._MAX_STAGE_SAMPLES:
            raise ValueError("progress URL v2 suffix exceeds 256 samples")
        existing_epochs = {
            sample["execution_epoch"] for sample in v2_samples
        }
        completion_publications = len(v2_samples) - len(existing_epochs)
        active = self._active_epochs.get(stage)
        if (
            active is not None
            and completion_publications >= self._MAX_COMPLETION_PUBLICATIONS
        ):
            raise ValueError("progress completion publications exceed 224")
        if active is None:
            prior_epochs = existing_epochs
            if snapshot.execution_epoch in prior_epochs:
                raise ValueError("progress execution epoch must be unique")
            if len(prior_epochs) >= self._MAX_EXECUTION_EPOCHS:
                raise ValueError("progress URL execution epochs exceed 32")
            if (
                snapshot.epoch_elapsed_seconds != 0.0
                or snapshot.completed_durable != snapshot.baseline_completed
                or any(snapshot.transport_event_histogram)
                or any(snapshot.active_censor_histogram)
                or any(snapshot.commit_event_histogram)
            ):
                raise ValueError("progress first epoch snapshot is invalid")
            if samples:
                prior = samples[-1]
                if (
                    snapshot.baseline_completed < int(prior["completed_units"])
                    or snapshot.total != int(prior["total_units"])
                ):
                    raise ValueError("progress epoch baseline is not durable")
            baseline_timestamp = self._progress_now_locked()
            self._active_epochs[stage] = (
                snapshot.execution_epoch,
                baseline_timestamp,
            )
        else:
            active_epoch, baseline_timestamp = active
            if snapshot.execution_epoch != active_epoch:
                raise ValueError("progress execution epoch changed in process")
        sample = self._v2_sample(
            snapshot, baseline_timestamp=baseline_timestamp
        )
        trial = [*samples, sample]
        trial_raw = {
            "rate_basis": rate_basis,
            "samples": trial,
            "completed_units": snapshot.completed_durable,
            "total_units": snapshot.total,
            "completed_at": None,
            "eligible_final_half_samples": 0,
            "excluded_final_half_samples": 0,
            "max_symmetric_eta_factor": None,
            "telemetry_schema_version": URL_TELEMETRY_SCHEMA_VERSION,
            "estimator": url_estimator_metadata(snapshot.deadline_seconds),
        }
        # Reuse the strict decoder for append validation without trusting the
        # freshly serialized scalar components.
        previous_completed = -1
        previous_timestamp = float("-inf")
        previous_deadline: float | None = None
        previous_total: int | None = None
        current_epoch: str | None = None
        seen_epochs: set[str] = set()
        epoch_baseline = -1
        epoch_baseline_timestamp = float("-inf")
        epoch_concurrency = -1
        epoch_elapsed = -1.0
        prior_transport: tuple[int, ...] | None = None
        prior_commit: tuple[int, ...] | None = None
        prior_transport_overflow: int | None = None
        prior_commit_overflow: int | None = None
        for candidate in trial:
            if "telemetry_schema_version" not in candidate:
                previous_completed = int(candidate["completed_units"])
                previous_timestamp = float(candidate["timestamp"])
                previous_total = int(candidate["total_units"])
                continue
            restored, decoded = self._restore_v2_sample(candidate)
            if previous_total is not None and decoded.total != previous_total:
                raise ValueError("progress total changed within stage")
            if previous_deadline is None:
                previous_deadline = decoded.deadline_seconds
            elif decoded.deadline_seconds != previous_deadline:
                raise ValueError("progress deadline changed within stage")
            if decoded.execution_epoch != current_epoch:
                if decoded.execution_epoch in seen_epochs:
                    raise ValueError("progress epoch is noncontiguous")
                seen_epochs.add(decoded.execution_epoch)
                current_epoch = decoded.execution_epoch
                epoch_baseline = decoded.baseline_completed
                epoch_baseline_timestamp = float(
                    restored["baseline_timestamp"]
                )
                epoch_concurrency = decoded.effective_concurrency
                if (
                    decoded.epoch_elapsed_seconds != 0.0
                    or decoded.completed_durable != decoded.baseline_completed
                    or decoded.baseline_completed < previous_completed
                    or float(restored["baseline_timestamp"])
                    < previous_timestamp
                    or any(decoded.transport_event_histogram)
                    or any(decoded.active_censor_histogram)
                    or any(decoded.commit_event_histogram)
                ):
                    raise ValueError("progress epoch baseline is invalid")
            else:
                if (
                    decoded.transport_overflow_events
                    < (prior_transport_overflow or 0)
                    or decoded.commit_overflow_events
                    < (prior_commit_overflow or 0)
                ):
                    raise ValueError("progress cumulative overflow decreased")
                if (
                    decoded.baseline_completed != epoch_baseline
                    or float(restored["baseline_timestamp"])
                    != epoch_baseline_timestamp
                    or decoded.effective_concurrency != epoch_concurrency
                    or decoded.epoch_elapsed_seconds < epoch_elapsed
                    or any(
                        after < before
                        for before, after in zip(
                            prior_transport or (),
                            decoded.transport_event_histogram,
                        )
                    )
                    or any(
                        after < before
                        for before, after in zip(
                            prior_commit or (), decoded.commit_event_histogram
                        )
                    )
                ):
                    raise ValueError("progress epoch samples are not monotonic")
            if (
                decoded.completed_durable < previous_completed
                or float(restored["timestamp"]) < previous_timestamp
            ):
                raise ValueError("progress samples are not monotonic")
            previous_completed = decoded.completed_durable
            previous_timestamp = float(restored["timestamp"])
            previous_total = decoded.total
            epoch_elapsed = decoded.epoch_elapsed_seconds
            prior_transport = decoded.transport_event_histogram
            prior_commit = decoded.commit_event_histogram
            prior_transport_overflow = decoded.transport_overflow_events
            prior_commit_overflow = decoded.commit_overflow_events

        telemetry = trial_raw
        telemetry["samples"] = trial
        self._stage_telemetry[stage] = telemetry
        self._state.completed_units = snapshot.completed_durable
        self._state.total_units = snapshot.total
        self._state.rate_basis = rate_basis
        self._logical_time_floor = max(
            self._logical_time_floor, float(sample["timestamp"])
        )
        self._complete_unit_stage_locked()

    def _complete_unit_stage_locked(self) -> None:
        if self._state.rate_basis is None:
            return
        if self._state.completed_units != self._state.total_units:
            return
        telemetry = self._stage_telemetry.get(self._state.stage)
        if telemetry is None or telemetry["completed_at"] is not None:
            return
        completed_at = float(telemetry["samples"][-1]["timestamp"])
        total = self._state.total_units
        telemetry.update(
            {
                "completed_at": completed_at,
                **self._eta_completion_summary(
                    telemetry["samples"],
                    total=total,
                    completed_at=completed_at,
                ),
            }
        )

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            unit_now = self._progress_now_locked()
            elapsed = max(0.0, now - self._state.stage_started_at)
            complete = self._state.completed_shards
            total = self._state.total_shards
            rate = complete / elapsed if elapsed > 0 else 0.0
            self._rolling_samples.append((now, complete))
            cutoff = now - self._ROLLING_WINDOW_SECONDS
            while (
                len(self._rolling_samples) > 1
                and self._rolling_samples[1][0] <= cutoff
            ):
                self._rolling_samples.popleft()
            rolling_rate = 0.0
            if len(self._rolling_samples) > 1:
                first_time, first_complete = self._rolling_samples[0]
                rolling_elapsed = now - first_time
                if rolling_elapsed > 0:
                    rolling_rate = max(
                        0.0,
                        (complete - first_complete) / rolling_elapsed,
                    )
            remaining = max(0, total - complete)
            shard_eta = remaining / rate if rate > 0 else None
            unit_rate, unit_rolling_rate = self._unit_rates_locked(unit_now)
            unit_samples = self._current_stage_samples_locked()
            unit_eta = (
                unit_samples[-1].get("predicted_remaining_seconds")
                if unit_samples
                else None
            )
            eta = unit_eta if self._state.rate_basis is not None else shard_eta
            free_by_root = {
                name: int(
                    shutil.disk_usage(DiskGuard._existing_ancestor(path)).free
                )
                for name, path in (
                    ("work", self.config.work_dir),
                    ("cache", self.config.cache_dir),
                    ("output", self.config.output_dir),
                )
            }
            current_by_root = {
                "work": self._uint(
                    self._state.known_work_bytes, "work current bytes"
                ),
                "cache": self._uint(
                    self._state.known_cache_bytes, "cache current bytes"
                ),
                "output": self._uint(
                    self._state.known_output_bytes, "output current bytes"
                ),
            }
            for name in ("work", "cache", "output"):
                current_bytes = current_by_root[name]
                free_bytes = self._uint(
                    free_by_root[name], f"{name} current free bytes"
                )
                prior = self._disk_roots.get(name)
                if prior is None:
                    self._disk_roots[name] = {
                        "start_bytes": current_bytes,
                        "peak_bytes": current_bytes,
                        "current_bytes": current_bytes,
                        "start_free_bytes": free_bytes,
                        "min_free_bytes": free_bytes,
                    }
                else:
                    prior["peak_bytes"] = max(
                        prior["peak_bytes"], current_bytes
                    )
                    prior["current_bytes"] = current_bytes
                    prior["min_free_bytes"] = min(
                        prior["min_free_bytes"], free_bytes
                    )
            return {
                "stage": self._state.stage,
                "detail": self._state.detail,
                "completed_shards": complete,
                "total_shards": total,
                "completed_units": self._state.completed_units,
                "total_units": self._state.total_units,
                "rate_basis": self._state.rate_basis,
                "counters": dict(sorted(self._state.counters.items())),
                "rates": {
                    "shards_per_second": rate,
                    "rolling_shards_per_second": rolling_rate,
                    "units_per_second": unit_rate,
                    "rolling_units_per_second": unit_rolling_rate,
                },
                "eta_seconds": eta,
                "stage_telemetry": {
                    stage: {
                        **telemetry,
                        "samples": [dict(sample) for sample in telemetry["samples"]],
                    }
                    for stage, telemetry in sorted(
                        self._stage_telemetry.items()
                    )
                },
                "elapsed_seconds": max(0.0, now - self._state.started_at),
                "updated_at": now,
                "disk": {
                    "work_bytes": self._state.known_work_bytes,
                    "cache_bytes": self._state.known_cache_bytes,
                    "output_bytes": self._state.known_output_bytes,
                    "free_bytes": free_by_root["work"],
                    "free_bytes_by_root": free_by_root,
                    "roots": {
                        name: dict(values)
                        for name, values in sorted(self._disk_roots.items())
                    },
                    "reserve_bytes": self.config.min_free_disk_bytes,
                },
            }

    def stage_completion_summary(self, stage: str) -> dict[str, Any]:
        """Return the bounded, completed URL-unit authority for one stage."""
        with self._lock:
            telemetry = self._stage_telemetry.get(stage)
            if telemetry is None or telemetry.get("completed_at") is None:
                raise ValueError(
                    f"URL telemetry is incomplete for stage: {stage}"
                )
            summary = {
                key: telemetry[key]
                for key in (
                    "rate_basis",
                    "completed_units",
                    "total_units",
                    "completed_at",
                    "eligible_final_half_samples",
                    "excluded_final_half_samples",
                    "max_symmetric_eta_factor",
                )
            }
            if telemetry.get("telemetry_schema_version") is not None:
                summary.update(
                    telemetry_schema_version=telemetry[
                        "telemetry_schema_version"
                    ],
                    estimator=dict(telemetry["estimator"]),
                )
        expected_basis = {"pages": "page_urls", "images": "image_urls"}.get(
            stage
        )
        if (
            summary["rate_basis"] != expected_basis
            or int(summary["completed_units"]) != int(summary["total_units"])
        ):
            raise ValueError(
                f"URL telemetry identity mismatch for stage: {stage}"
            )
        return summary

    @staticmethod
    def _format_console_eta(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds) or seconds < 0:
            return "--:--"
        rounded = int(seconds + 0.5)
        hours, remainder = divmod(rounded, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _model_console_metrics(
        self,
        now: float,
    ) -> list[tuple[ModelProgressSnapshot, float, float | None]]:
        with self._lock:
            metrics = []
            for modality in ("text", "image"):
                progress = self._model_progress.get(modality)
                if progress is None:
                    continue
                samples = self._model_samples.setdefault(modality, deque())
                if not samples or samples[-1] != (now, progress.completed):
                    samples.append((now, progress.completed))
                cutoff = now - self._ROLLING_WINDOW_SECONDS
                while len(samples) > 1 and samples[1][0] <= cutoff:
                    samples.popleft()
                first_at, first_completed = samples[0]
                elapsed = now - first_at
                rate = (
                    max(
                        0.0,
                        (progress.completed - first_completed) / elapsed,
                    )
                    if elapsed > 0
                    else 0.0
                )
                remaining = max(0, progress.total - progress.completed)
                eta = remaining / rate if rate > 0 else None
                metrics.append((progress, rate, eta))
            return metrics

    def _console_line(self, snapshot: dict[str, Any], *, is_tty: bool) -> str:
        if snapshot["stage"] == "models":
            model_metrics = self._model_console_metrics(time.monotonic())
            if model_metrics:
                rendered = []
                for progress, rate, eta in model_metrics:
                    bar = ""
                    if is_tty:
                        width = 10
                        filled = (
                            width * progress.completed // progress.total
                            if progress.total
                            else width
                        )
                        bar = (
                            f" |{'#' * filled}"
                            f"{'-' * (width - filled)}|"
                        )
                    rendered.append(
                        f"models:{progress.modality}{bar} "
                        f"{progress.completed}/{progress.total} "
                        f"[{rate:.2f} job/s, ETA "
                        f"{self._format_console_eta(eta)}] "
                        f"success={progress.success} "
                        f"terminal={progress.terminal} "
                        f"leased={progress.leased} "
                        f"pending={progress.pending}"
                    )
                return f"[wdc200k] {' || '.join(rendered)}"
        eta = self._format_console_eta(snapshot["eta_seconds"])
        return (
            f"[wdc200k] stage={snapshot['stage']} "
            f"progress={snapshot['completed_shards']}/"
            f"{snapshot['total_shards']} "
            f"detail={snapshot['detail'] or '-'} "
            f"units={snapshot['completed_units']}/"
            f"{snapshot['total_units']} ETA={eta} "
            f"work={snapshot['disk']['work_bytes']} "
            f"cache={snapshot['disk']['cache_bytes']} "
            f"output={snapshot['disk']['output_bytes']} "
            f"free={snapshot['disk']['free_bytes']} "
            "free_work="
            f"{snapshot['disk']['free_bytes_by_root']['work']} "
            "free_cache="
            f"{snapshot['disk']['free_bytes_by_root']['cache']} "
            "free_output="
            f"{snapshot['disk']['free_bytes_by_root']['output']} "
            f"reserve={snapshot['disk']['reserve_bytes']}"
        )

    def _refresh_model_bars(
        self,
        metrics: Sequence[
            tuple[ModelProgressSnapshot, float, float | None]
        ],
    ) -> None:
        if tqdm is None:
            return
        if self._tty_line_length:
            print(flush=True)
            self._tty_line_length = 0
        positions = {"text": 0, "image": 1}
        for progress, _rate, _eta in metrics:
            bar = self._model_bars.get(progress.modality)
            if bar is None:
                bar = tqdm(
                    total=progress.total,
                    initial=progress.completed,
                    desc=f"WDC {progress.modality}",
                    unit="job",
                    position=positions[progress.modality],
                    dynamic_ncols=True,
                    leave=True,
                    file=sys.stdout,
                    mininterval=max(
                        0.1,
                        self.config.progress_interval_seconds,
                    ),
                )
                self._model_bars[progress.modality] = bar
            bar.total = progress.total
            if progress.completed > bar.n:
                bar.update(progress.completed - bar.n)
            bar.set_postfix(
                success=progress.success,
                terminal=progress.terminal,
                leased=progress.leased,
                pending=progress.pending,
                refresh=False,
            )
            bar.refresh()

    @staticmethod
    def _stage_bar_values(
        snapshot: dict[str, Any],
    ) -> tuple[tuple[str, str], int, int, str]:
        rate_basis = snapshot.get("rate_basis")
        if rate_basis is not None:
            unit = {
                "page_urls": "url",
                "image_urls": "url",
            }.get(str(rate_basis), str(rate_basis).removesuffix("s"))
            return (
                (str(snapshot["stage"]), str(rate_basis)),
                int(snapshot["completed_units"]),
                int(snapshot["total_units"]),
                unit,
            )
        stage = str(snapshot["stage"])
        detail = str(snapshot.get("detail") or "")
        unit = {
            "structural": "table",
            "sampling": "shard",
            "asset_planning": "record",
            "selection": "step",
            "materialize": "step",
        }.get(stage, "step")
        progress_key = unit
        if stage == "selection" and detail:
            progress_key = detail
            unit = (
                "archive"
                if detail == "scan statistics archives"
                else "record"
            )
        return (
            (stage, progress_key),
            int(snapshot["completed_shards"]),
            int(snapshot["total_shards"]),
            unit,
        )

    def _refresh_stage_bar(self, snapshot: dict[str, Any]) -> None:
        if tqdm is None:
            return
        key, completed, total, unit = self._stage_bar_values(snapshot)
        if self._stage_bar is not None and self._stage_bar_key != key:
            self._close_stage_bar()
        if self._stage_bar is None:
            self._stage_bar = tqdm(
                total=total,
                initial=completed,
                desc=f"WDC {key[0].replace('_', ' ')}",
                unit=unit,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                mininterval=max(
                    0.1,
                    self.config.progress_interval_seconds,
                ),
            )
            self._stage_bar_key = key
        self._stage_bar.total = total
        if completed > self._stage_bar.n:
            self._stage_bar.update(completed - self._stage_bar.n)
        detail = str(snapshot.get("detail") or "")
        if detail:
            self._stage_bar.set_postfix_str(detail, refresh=False)
        self._stage_bar.refresh()

    def _close_stage_bar(self) -> None:
        if self._stage_bar is not None:
            self._stage_bar.close()
            self._stage_bar = None
        self._stage_bar_key = None

    def _close_model_bars(self) -> None:
        for modality in ("text", "image"):
            bar = self._model_bars.pop(modality, None)
            if bar is not None:
                bar.close()

    def publish(self) -> None:
        with self._publish_lock:
            snapshot = self._snapshot()
            _atomic_json(
                self.path,
                snapshot,
                pre_write_guard=self._pre_write_guard,
            )
            is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
            console_now = time.monotonic()
            with self._lock:
                should_print = (
                    is_tty
                    or self._force_console
                    or self._last_console_at is None
                    or console_now - self._last_console_at
                    >= self._NON_TTY_CONSOLE_INTERVAL_SECONDS
                )
                if should_print:
                    self._force_console = False
                    self._last_console_at = console_now
            if not should_print:
                return
            if (
                is_tty
                and snapshot["stage"] == "models"
                and tqdm is not None
            ):
                self._close_stage_bar()
                self._refresh_model_bars(
                    self._model_console_metrics(console_now)
                )
                return
            if is_tty and tqdm is not None:
                self._close_model_bars()
                self._refresh_stage_bar(snapshot)
                return
            line = self._console_line(snapshot, is_tty=is_tty)
            if is_tty:
                padding = " " * max(0, self._tty_line_length - len(line))
                print(f"\r{line}{padding}", end="", flush=True)
                self._tty_line_length = len(line)
            else:
                print(line, flush=True)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.config.progress_interval_seconds))
        with self._lock:
            self._force_console = True
        self.publish()
        self._close_stage_bar()
        self._close_model_bars()
        if self._tty_line_length:
            print(flush=True)
            self._tty_line_length = 0


def _atomic_json(
    path: Path,
    payload: dict[str, Any],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    tracker = GuardedWriteTracker(path, pre_write_guard)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as raw_handle:
            handle = GuardedTextWriter(raw_handle, tracker)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        tracker.before_commit(0)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _producer_registry_path(config: PipelineConfig, stage: str) -> Path:
    return config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"


def _has_complete_sampling_authority(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("complete") is True
        and isinstance(payload.get("compact_source_authority"), dict)
    )


@dataclass(frozen=True)
class ProducerManifestRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class StageRegistry:
    stage: str
    producer_type: str
    producer_manifests: tuple[ProducerManifestRef, ...]
    upstream_identity: str
    config_fingerprint: str
    counters: dict[str, int]
    complete: bool


_PRODUCER_TYPES = {
    "selection": "wdc200k-selection",
    "structural": "wdc200k-structural-barrier",
    "sampling": "wdc200k-entity-sampling",
    "pages": "wdc200k-page-network",
    "asset_planning": "wdc200k-asset-planning",
    "images": "wdc200k-image-and-assets",
    "models": "wdc200k-model-stage",
    "materialize": "wdc200k-dataset",
}


_STAGE_CONFIG_FIELDS: dict[str, tuple[str, ...]] = {
    "selection": (
        "max_source_tables",
        "selection_seed",
        "top100_policy",
        "top100_per_class",
        "include_rest",
        "min_candidate_rows",
        "min_candidate_columns",
        "minimum3_fraction",
        "class_max_tables",
    ),
    "structural": ("selection_shard_tables",),
    "sampling": (
        "sampled_entities_per_table",
        "sampled_entity_expansion_schedule",
        "sampling_expansion_round_index",
        "entity_sampling_seed",
        "global_entity_budget",
        "query_rows_per_table",
        "min_column_non_empty_ratio",
        "min_recovered_value_ratio",
        "min_recovery_denominator",
        "min_rows_per_output_table",
    ),
    "pages": (
        "web_max_retries",
        "web_max_response_seconds",
        "web_global_concurrency",
        "web_per_host_concurrency",
        "web_max_page_bytes",
    ),
    "asset_planning": (
        "max_image_attempts_per_entity",
        "max_images_per_entity",
        "records_per_shard",
    ),
    "images": (
        "web_max_response_seconds",
        "web_global_concurrency",
        "web_per_host_concurrency",
        "web_max_image_bytes",
        "max_image_attempts_per_entity",
        "max_images_per_entity",
        "text_asset_chunk_chars",
        "min_text_asset_chunk_chars",
        "max_text_asset_chunks_per_entity",
        "records_per_shard",
    ),
    "models": (
        "text_model_name",
        "image_model_name",
        "text_model_base_url",
        "text_model_base_urls",
        "image_model_base_url",
        "image_model_base_urls",
    ),
    "materialize": (
        "split_by",
        "train_ratio",
        "dev_ratio",
        "test_ratio",
        "min_column_non_empty_ratio",
        "min_recovered_value_ratio",
        "min_recovery_denominator",
        "min_rows_per_output_table",
        "query_rows_per_table",
        "max_train_query_row_views_per_join",
        "max_query_tables_per_source_table",
        "max_query_context_attrs",
        "max_target_context_attrs",
        "explicit_join_fallback_mode",
        "explicit_join_fallback_ratio",
        "records_per_shard",
        "unrecoverable_replacement_rounds",
        "unrecoverable_drop_probability",
        "recovery_replacement_round_index",
    ),
}


def _stage_config_fingerprint(config: PipelineConfig, stage: str) -> str:
    values = {
        field_name: getattr(config, field_name)
        for field_name in _STAGE_CONFIG_FIELDS[stage]
    }
    return stable_hash(
        "wdc200k-pipeline-config-v1",
        stage,
        json.dumps(values, sort_keys=True, default=str),
        length=40,
    )


def _load_stage_registry(path: Path) -> StageRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline registry is not an object: {path}")
    references = tuple(
        ProducerManifestRef(
            path=Path(str(item["path"])),
            sha256=str(item["sha256"]),
        )
        for item in payload.get("producer_manifests") or []
    )
    return StageRegistry(
        stage=str(payload.get("stage") or ""),
        producer_type=str(payload.get("producer_type") or ""),
        producer_manifests=references,
        upstream_identity=str(payload.get("upstream_identity") or ""),
        config_fingerprint=str(payload.get("config_fingerprint") or ""),
        counters={
            str(key): int(value)
            for key, value in (payload.get("counters") or {}).items()
        },
        complete=payload.get("complete") is True,
    )


_ATTEMPT_COUNTER_FIELDS = (
    "transport_attempts",
    "duplicate_physical_requests",
    "terminal_replays",
    "unfinished_transport_attempts",
    "blocked_durable_replays",
)


def _read_only_sqlite(path: Path) -> sqlite3.Connection:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"telemetry SQLite database is missing: {path}")
    wal_path = Path(f"{path}-wal")
    if wal_path.is_file() and wal_path.stat().st_size > 0:
        raise ValueError(
            f"telemetry SQLite database has an uncheckpointed WAL: {path}"
        )
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro&immutable=1",
        uri=True,
        timeout=30.0,
    )
    connection.row_factory = sqlite3.Row
    return connection


def _read_only_job_scope(path: Path, kind: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"schema_version": "wdc200k-job-scope-v1", "kind": kind},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    records = 0
    with _read_only_sqlite(path) as connection:
        for row in connection.execute(
            """
            SELECT job_id, kind, payload_json, status, result_json,
                   owner, lease_expires, lease_id, updated_at
            FROM jobs WHERE kind = ? ORDER BY job_id
            """,
            (kind,),
        ):
            records += 1
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return {"records": records, "digest": digest.hexdigest()}


def _read_only_transport_attempt_summary(
    database_path: Path,
    *,
    stage: str,
    policy_fingerprint: str,
    job_store_path: Path,
    job_kind: str,
    job_id_prefix: str,
) -> dict[str, Any]:
    outcome_table = {
        "pages": "page_outcomes",
        "images": "image_outcomes",
    }.get(stage)
    if outcome_table is None:
        raise ValueError(f"unsupported transport stage: {stage}")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": "wdc200k-transport-attempt-v1",
                "stage": stage,
                "policy_fingerprint": policy_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    records = 0
    physical = 0
    terminal_replays = 0
    unfinished = 0
    blocked = 0
    duplicates = 0
    previous_physical_url: str | None = None
    previous_physical_count = 0
    job_uri = (
        f"{Path(job_store_path).resolve().as_uri()}?mode=ro&immutable=1"
    )
    with _read_only_sqlite(database_path) as connection:
        if not Path(job_store_path).resolve().is_file():
            raise ValueError(
                f"telemetry SQLite database is missing: {job_store_path}"
            )
        connection.execute(
            "ATTACH DATABASE ? AS transport_attempt_jobs",
            (job_uri,),
        )
        rows = connection.execute(
            f"""
            SELECT attempt.attempt_id, attempt.execution_id,
                   attempt.stage, attempt.policy_fingerprint,
                   attempt.url_key, attempt.url, attempt.started_at,
                   attempt.finished_at, attempt.final_status,
                   attempt.baseline_outcome_status, attempt.suppressed,
                   baseline.status AS durable_baseline_outcome_status
            FROM transport_attempts AS attempt
            LEFT JOIN {outcome_table} AS baseline
              ON baseline.policy_fingerprint = attempt.policy_fingerprint
             AND baseline.url_key = attempt.url_key
             AND baseline.updated_at <= attempt.started_at
            WHERE attempt.stage = ?
              AND attempt.policy_fingerprint = ?
              AND EXISTS (
                  SELECT 1 FROM transport_attempt_jobs.jobs AS job
                  WHERE job.kind = ?
                    AND job.job_id = ? || ':' || attempt.url_key
              )
            ORDER BY attempt.url_key, attempt.started_at,
                     attempt.attempt_id
            """,
            (stage, policy_fingerprint, job_kind, job_id_prefix),
        )
        for row in rows:
            values = list(row)
            recorded_baseline = row["baseline_outcome_status"]
            durable_baseline = row["durable_baseline_outcome_status"]
            baseline = recorded_baseline or durable_baseline
            suppressed = bool(row["suppressed"])
            finished = row["finished_at"] is not None
            final_status = row["final_status"]
            if (
                str(row["stage"]) != stage
                or str(row["policy_fingerprint"]) != policy_fingerprint
                or recorded_baseline not in {None, "success", "terminal"}
                or durable_baseline not in {None, "success", "terminal"}
                or (
                    recorded_baseline is not None
                    and durable_baseline is not None
                    and recorded_baseline != durable_baseline
                )
                or (finished != (final_status is not None))
                or (suppressed and final_status != "suppressed")
                or (
                    not suppressed
                    and final_status
                    not in {None, "success", "terminal", "exception"}
                )
            ):
                raise ValueError("invalid transport attempt ledger row")
            records += 1
            if suppressed:
                blocked += 1
            else:
                physical += 1
                url_key = str(row["url_key"])
                if url_key != previous_physical_url:
                    duplicates += max(0, previous_physical_count - 1)
                    previous_physical_url = url_key
                    previous_physical_count = 1
                else:
                    previous_physical_count += 1
                terminal_replays += int(baseline == "terminal")
                unfinished += int(not finished)
            digest.update(
                json.dumps(
                    values,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    duplicates += max(0, previous_physical_count - 1)
    return {
        "stage": stage,
        "policy_fingerprint": policy_fingerprint,
        "database_path": str(Path(database_path).resolve()),
        "records": records,
        "digest": digest.hexdigest(),
        "transport_attempts": physical,
        "duplicate_physical_requests": duplicates,
        "terminal_replays": terminal_replays,
        "unfinished_transport_attempts": unfinished,
        "blocked_durable_replays": blocked,
    }


def _expected_network_scope(
    config: PipelineConfig,
    stage: str,
) -> tuple[str, str, str]:
    policy = FetchPolicy(
        retries=0,
        deadline_seconds=config.web_max_response_seconds,
        global_concurrency=config.web_global_concurrency,
        per_host_concurrency=config.web_per_host_concurrency,
        **(
            {"policy_version": "wdc200k-image-fetch-v1"}
            if stage == "images"
            else {}
        ),
    )
    if stage == "pages":
        fingerprint = policy.fingerprint
        return fingerprint, f"wdc200k-page:{fingerprint}", fingerprint
    if stage != "images":
        raise ValueError(f"unsupported network telemetry stage: {stage}")
    fingerprint = stable_hash(
        "wdc200k-image-fetch-v1",
        policy.fingerprint,
        length=40,
    )
    manifest_path = (
        config.work_dir / "image_jobs" / "unique-images.jsonl.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = manifest.get("completed_shards") or []
    if (
        manifest.get("stage") != "wdc200k-unique-image-jobs-v1"
        or manifest.get("complete") is not True
        or len(completed) != 1
    ):
        raise ValueError("unique image job authority is invalid")
    job_set = stable_hash(
        str(manifest.get("input_fingerprint") or ""),
        str(manifest.get("parameter_fingerprint") or ""),
        str(completed[0].get("sha256") or ""),
        length=40,
    )
    kind = f"wdc200k-image:{fingerprint}:{job_set}"
    return fingerprint, kind, kind


def _network_telemetry_counters(
    stage: str,
    attempts: dict[str, Any],
    completion: dict[str, Any],
) -> dict[str, int]:
    prefix = {"pages": "page", "images": "image"}.get(stage)
    if prefix is None:
        raise ValueError(f"unsupported network telemetry stage: {stage}")
    counters = {
        f"{prefix}_{field}": int(attempts[field])
        for field in _ATTEMPT_COUNTER_FIELDS
    }
    counters.update(
        {
            f"{prefix}_url_completed": int(completion["completed_units"]),
            f"{prefix}_url_total": int(completion["total_units"]),
            f"{prefix}_eta_eligible_final_half_samples": int(
                completion["eligible_final_half_samples"]
            ),
            f"{prefix}_eta_excluded_final_half_samples": int(
                completion["excluded_final_half_samples"]
            ),
        }
    )
    if any(value < 0 for value in counters.values()):
        raise ValueError("network telemetry counters must be non-negative")
    return counters


def _validate_network_telemetry(
    config: PipelineConfig,
    registry_stage: str,
    payload: dict[str, Any],
    path: Path,
) -> None:
    attempts = payload.get("transport_attempts")
    completion = payload.get("url_completion")
    authority = payload.get("transport_attempt_authority")
    if not all(
        isinstance(value, dict)
        for value in (attempts, completion, authority)
    ):
        raise ValueError(f"network telemetry is missing: {path}")
    policy = str(payload.get("policy_fingerprint") or "")
    expected_policy, expected_kind, expected_prefix = _expected_network_scope(
        config,
        registry_stage,
    )
    if policy != expected_policy:
        raise ValueError("transport attempt policy mismatch")
    if attempts.get("policy_fingerprint") != policy:
        raise ValueError("transport attempt policy mismatch")
    if attempts.get("stage") != registry_stage:
        raise ValueError("transport attempt stage mismatch")
    expected_paths = {
        "pages": (
            config.cache_dir / "page_cache" / "outcomes.sqlite3",
            config.work_dir / "page_jobs" / "jobs.sqlite3",
        ),
        "images": (
            config.cache_dir / "image_cache" / "outcomes.sqlite3",
            config.work_dir / "image_jobs" / "jobs.sqlite3",
        ),
    }
    try:
        expected_database, expected_jobs = expected_paths[registry_stage]
    except KeyError as error:
        raise ValueError(
            "network manifest belongs to a non-network stage"
        ) from error
    database_path = Path(str(authority.get("database_path") or "")).resolve()
    job_store_path = Path(str(authority.get("job_store_path") or "")).resolve()
    job_kind = str(authority.get("job_kind") or "")
    job_id_prefix = str(authority.get("job_id_prefix") or "")
    declared_scope = authority.get("job_scope")
    if (
        database_path != expected_database.resolve()
        or job_store_path != expected_jobs.resolve()
        or str(attempts.get("database_path") or "") != str(database_path)
        or not database_path.is_file()
        or not job_store_path.is_file()
    ):
        raise ValueError("transport attempt authority identity mismatch")
    if job_kind != expected_kind or job_id_prefix != expected_prefix:
        raise ValueError("transport attempt scope mismatch")
    try:
        actual_scope = _read_only_job_scope(job_store_path, job_kind)
    except ValueError as error:
        raise ValueError("transport attempt authority mismatch") from error
    if (
        not isinstance(declared_scope, dict)
        or declared_scope != actual_scope
        or int(actual_scope["records"])
        != int((payload.get("counts") or {}).get("unique", -1))
    ):
        raise ValueError("transport attempt scope mismatch")
    try:
        actual = _read_only_transport_attempt_summary(
            database_path,
            stage=registry_stage,
            policy_fingerprint=policy,
            job_store_path=job_store_path,
            job_kind=job_kind,
            job_id_prefix=job_id_prefix,
        )
    except ValueError as error:
        raise ValueError("transport attempt authority mismatch") from error
    if actual != attempts:
        raise ValueError("transport attempt authority mismatch")
    counts = payload.get("counts") or {}
    expected_basis = {"pages": "page_urls", "images": "image_urls"}[
        registry_stage
    ]
    try:
        completed = int(completion["completed_units"])
        total = int(completion["total_units"])
        completed_at = float(completion["completed_at"])
        eligible = int(completion["eligible_final_half_samples"])
        excluded = int(completion["excluded_final_half_samples"])
        factor_raw = completion["max_symmetric_eta_factor"]
        factor = None if factor_raw is None else float(factor_raw)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("URL completion summary is invalid") from error
    schema = completion.get("telemetry_schema_version")
    estimator = completion.get("estimator")
    if schema is None:
        estimator_is_valid = "estimator" not in completion
    else:
        try:
            deadline = float(estimator["deadline_seconds"])
            estimator_is_valid = (
                schema == URL_TELEMETRY_SCHEMA_VERSION
                and isinstance(estimator, dict)
                and estimator == url_estimator_metadata(deadline)
            )
        except (KeyError, TypeError, ValueError):
            estimator_is_valid = False
    if (
        not estimator_is_valid
        or completion.get("rate_basis") != expected_basis
        or completed != total
        or total != int(counts.get("unique", -1))
        or not math.isfinite(completed_at)
        or completed_at < 0
        or eligible < 0
        or excluded < 0
        or (factor is not None and (not math.isfinite(factor) or factor < 1.0))
    ):
        raise ValueError("URL completion summary identity mismatch")
    try:
        progress_reporter = ProgressReporter(
            replace(config, resume=True, from_stage=None)
        )
        progress_completion = progress_reporter.stage_completion_summary(
            registry_stage
        )
    except ValueError as error:
        if "completed_at" in str(error):
            raise ValueError("progress URL completion mismatch") from error
        raise ValueError("progress ETA summary mismatch") from error
    if progress_completion != completion:
        raise ValueError("progress URL completion mismatch")
    if eligible > 0 and factor is None:
        raise ValueError("progress ETA summary mismatch")
    _network_telemetry_counters(registry_stage, attempts, completion)


def _validate_producer_manifest(
    config: PipelineConfig,
    stage: str,
    path: Path,
    *,
    allow_network_shard_repair: bool = False,
    compact_replacements: bool = False,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"{stage} producer manifest is incomplete: {path}")
    producer_stage = str(payload.get("stage") or "")
    expected = {
        "selection": {"wdc200k_selection"},
        "structural": {"wdc200k_structural", "wdc200k_validated_selection"},
        "sampling": {"wdc200k_entity_sampling"},
        "pages": {"wdc200k_network_fetch"},
        "asset_planning": {"wdc200k_asset_planning"},
        "images": {
            "wdc200k-unique-image-jobs-v1",
            "wdc200k-image-fetch-v1",
            "wdc200k_asset_materialization",
            "wdc200k_network_fetch",
        },
        "models": {"wdc200k_model_task_adapter", "wdc200k_model_outputs"},
        "materialize": {"wdc200k_materialization"},
    }[stage]
    if producer_stage not in expected:
        raise ValueError(
            f"unexpected {stage} producer type {producer_stage!r}: {path}"
        )
    if producer_stage == "wdc200k_network_fetch":
        _validate_network_telemetry(config, stage, payload, path)
    root = (
        path.parent.parent
        if producer_stage in {"wdc200k_structural", "wdc200k_validated_selection"}
        else path.parent
    )
    shard_fields = (
        "completed_shards",
        "entity_plan_shards",
        "image_mapping_shards",
        "bridge_asset_shards",
        "table_asset_link_shards",
        "task_shards",
        "error_shards",
        "extraction_shards",
    )
    for field_name in shard_fields:
        declared = payload.get(field_name)
        if declared is None:
            continue
        if not isinstance(declared, list):
            raise ValueError(
                f"{stage} producer manifest has invalid {field_name}: {path}"
            )
        for item in declared:
            try:
                completed = CompletedShard(
                    path=str(item["path"]),
                    records=int(item["records"]),
                    bytes=int(item["bytes"]),
                    sha256=str(item["sha256"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{stage} producer manifest has invalid shard: {path}"
                ) from error
            if compact_replacements and completed.path.startswith(
                ("entities/", "page_refs/", "direct_image_refs/")
            ):
                continue
            if not validate_completed_shard(completed, root):
                if (
                    allow_network_shard_repair
                    and producer_stage == "wdc200k_network_fetch"
                ):
                    continue
                raise ValueError(
                    f"{stage} producer shard checksum mismatch: "
                    f"{root / completed.path}"
                )


def _validate_stage_registry(
    config: PipelineConfig,
    stage: str,
    *,
    expected_upstream_identity: str,
    allow_network_shard_repair: bool = False,
) -> StageRegistry:
    path = _producer_registry_path(config, stage)
    registry = _load_stage_registry(path)
    if (
        not registry.complete
        or registry.stage != stage
        or registry.producer_type != _PRODUCER_TYPES[stage]
        or registry.upstream_identity != expected_upstream_identity
        or registry.config_fingerprint != _stage_config_fingerprint(config, stage)
        or not registry.producer_manifests
    ):
        raise ValueError(f"pipeline registry identity mismatch: {path}")
    expected_telemetry_counters: dict[str, int] = {}
    compact_replacements = False
    if stage == "structural":
        sampling_manifest = config.work_dir / "sampling" / "manifest.json"
        if _has_complete_sampling_authority(sampling_manifest):
            try:
                validate_sampling_source_authority(
                    sampling_manifest,
                    structural_output_root=config.work_dir / "structural",
                )
            except ValueError as error:
                raise ValueError(
                    "structural producer shard checksum mismatch: "
                    "compact source authority"
                ) from error
            compact_replacements = True
    for reference in registry.producer_manifests:
        if not reference.path.is_file():
            raise ValueError(f"producer manifest is missing: {reference.path}")
        if _sha256_path(reference.path) != reference.sha256:
            raise ValueError(
                f"producer manifest checksum mismatch: {reference.path}"
            )
        _validate_producer_manifest(
            config,
            stage,
            reference.path,
            allow_network_shard_repair=allow_network_shard_repair,
            compact_replacements=compact_replacements,
        )
        producer_payload = json.loads(
            reference.path.read_text(encoding="utf-8")
        )
        if producer_payload.get("stage") == "wdc200k_network_fetch":
            expected_telemetry_counters.update(
                _network_telemetry_counters(
                    stage,
                    producer_payload["transport_attempts"],
                    producer_payload["url_completion"],
                )
            )
    if any(
        registry.counters.get(key) != value
        for key, value in expected_telemetry_counters.items()
    ):
        raise ValueError("pipeline registry telemetry counters mismatch")
    return registry


def _write_stage_registry(
    config: PipelineConfig,
    stage: str,
    *,
    producer_manifests: Iterable[Path],
    counters: dict[str, int],
    upstream_identity: str,
    pre_write_guard: PreWriteGuard | None = None,
) -> StageRegistry:
    manifests = [
        {
            "path": str(Path(path).resolve()),
            "sha256": _sha256_path(Path(path)),
        }
        for path in producer_manifests
    ]
    _atomic_json(
        _producer_registry_path(config, stage),
        {
            "stage": stage,
            "schema_version": "wdc200k-pipeline-registry-v1",
            "producer_type": _PRODUCER_TYPES[stage],
            "producer_manifests": manifests,
            "upstream_identity": upstream_identity,
            "config_fingerprint": _stage_config_fingerprint(config, stage),
            "counters": dict(sorted(counters.items())),
            "complete": True,
        },
        pre_write_guard=pre_write_guard,
    )
    return _validate_stage_registry(
        config,
        stage,
        expected_upstream_identity=upstream_identity,
    )


_STAGE_WORK_PATHS: dict[str, tuple[str, ...]] = {
    "selection": ("selection",),
    "structural": ("structural",),
    "sampling": ("sampling",),
    "pages": ("page_jobs",),
    "asset_planning": ("asset_planning",),
    "images": ("image_jobs", "materialized_assets"),
    "models": ("adapted_model_tasks", "model_outputs"),
    "materialize": ("materialization",),
}


def invalidate_from_stage(
    config: PipelineConfig,
    stage: str,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> ArchiveResult:
    """Recoverably archive the named and downstream state without deletion."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    return archive_pipeline_state(
        work_dir=config.work_dir,
        output_dir=config.output_dir,
        cache_dir=config.cache_dir,
        stages=STAGES,
        from_stage=stage,
        stage_work_paths=_STAGE_WORK_PATHS,
        stage_registry_paths={
            current: _producer_registry_path(config, current)
            for current in STAGES
        },
        runtime_dir=config.runtime_dir,
        refresh_page_cache=config.refresh_page_cache,
        refresh_image_cache=config.refresh_image_cache,
        pre_write_guard=pre_write_guard,
    )


def _statistics_archives(input_dir: Path) -> tuple[Path, ...]:
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    archives = []
    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        archive = child / f"{child.name}_statistics.zip"
        if archive.is_file():
            archives.append(archive)
    if not archives:
        raise ValueError(f"no statistics archives found under {input_dir}")
    return tuple(sorted(archives))


def _validate_runtime_paths(config: PipelineConfig) -> None:
    if config.runtime_dir is None:
        raise ValueError("runtime_dir must be configured")
    runtime_dir = config.runtime_dir.resolve()
    required_runtime = (config.work_dir.resolve() / "runtime").resolve()
    if runtime_dir != required_runtime:
        raise ValueError(
            "runtime_dir must equal work_dir/runtime: "
            f"{runtime_dir} != {required_runtime}"
        )
    candidates: list[tuple[str, Path]] = [("runtime_dir", runtime_dir)]
    for name, value in (
        ("text_model_base_urls_file", config.text_model_base_urls_file),
        ("image_model_base_urls_file", config.image_model_base_urls_file),
        ("model_start_marker", config.model_start_marker),
        ("model_ready_marker", config.model_ready_marker),
        ("model_text_done_marker", config.model_text_done_marker),
        ("model_image_done_marker", config.model_image_done_marker),
    ):
        if value is not None:
            candidates.append((name, Path(value).resolve()))
    protected = (
        ("input", config.input_dir),
        ("cache", config.cache_dir),
        ("output", config.output_dir),
    )
    for name, path in candidates:
        for root_name, root in protected:
            if path == root or path.is_relative_to(root):
                raise ValueError(
                    f"{name} must be outside {root_name} root: {path}"
                )
    seen: dict[Path, str] = {}
    for name, path in candidates:
        previous = seen.get(path)
        if previous is not None:
            raise ValueError(
                f"runtime paths conflict: {previous} and {name}: {path}"
            )
        seen[path] = name
        if name != "runtime_dir" and not path.is_relative_to(
            runtime_dir
        ):
            raise ValueError(
                f"{name} must be inside runtime_dir: {path}"
            )


def _preflight(
    config: PipelineConfig,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, ...]:
    archives = _statistics_archives(config.input_dir)
    if config.max_source_tables <= 0:
        raise ValueError("max_source_tables must be positive")
    SelectionPolicy(
        target_tables=config.max_source_tables,
        seed=config.selection_seed,
        top100_policy=config.top100_policy,
        top100_per_class=config.top100_per_class,
        include_rest=config.include_rest,
        min_candidate_rows=config.min_candidate_rows,
        min_candidate_columns=config.min_candidate_columns,
        minimum3_fraction=config.minimum3_fraction,
        rest_base_per_class=0,
        class_cap=config.class_max_tables,
    )
    if config.web_max_retries != 0:
        raise ValueError("web_max_retries must be zero")
    if config.web_max_response_seconds <= 0:
        raise ValueError("web_max_response_seconds must be positive")
    if config.max_image_attempts_per_entity < 0:
        raise ValueError("max_image_attempts_per_entity must be non-negative")
    if config.max_images_per_entity < 0:
        raise ValueError("max_images_per_entity must be non-negative")
    if config.max_train_query_row_views_per_join < 0:
        raise ValueError(
            "max_train_query_row_views_per_join must be non-negative"
        )
    if not 0.0 <= config.explicit_join_fallback_ratio <= 1.0:
        raise ValueError(
            "explicit_join_fallback_ratio must be within [0, 1]"
        )
    if config.explicit_join_fallback_mode not in (
        join_builder.EXPLICIT_JOIN_FALLBACK_MODES
    ):
        raise ValueError("explicit_join_fallback_mode is invalid")
    SamplingPolicy(
        sampled_entities_per_table=config.sampled_entities_per_table,
        entity_sampling_seed=config.entity_sampling_seed,
        query_rows_per_table=config.query_rows_per_table,
        min_column_non_empty_ratio=config.min_column_non_empty_ratio,
        min_recovered_value_ratio=config.min_recovered_value_ratio,
        min_recovery_denominator=config.min_recovery_denominator,
        min_rows_per_output_table=config.min_rows_per_output_table,
        global_entity_budget=config.global_entity_budget,
    )
    schedule = config.sampled_entity_expansion_schedule
    if tuple(sorted(set(schedule))) != schedule:
        raise ValueError(
            "sampled entity expansion schedule must be strictly increasing"
        )
    if any(limit <= config.sampled_entities_per_table for limit in schedule):
        raise ValueError(
            "sampled entity expansion caps must exceed the initial cap"
        )
    if config.sampling_expansion_round_index < 0:
        raise ValueError("sampling expansion round index must be non-negative")
    marker_values = (
        config.model_start_marker,
        config.model_ready_marker,
        config.model_text_done_marker,
        config.model_image_done_marker,
    )
    if any(marker_values) and not all(marker_values):
        raise ValueError("all four staged model markers must be provided together")
    _validate_runtime_paths(config)
    if config.refresh_page_cache and (
        config.from_stage is None
        or STAGES.index(config.from_stage) > STAGES.index("pages")
    ):
        raise ValueError(
            "--refresh_page_cache requires --from_stage pages or earlier"
        )
    if config.refresh_image_cache and (
        config.from_stage is None
        or STAGES.index(config.from_stage) > STAGES.index("images")
    ):
        raise ValueError(
            "--refresh_image_cache requires --from_stage images or earlier"
        )
    guard = disk_guard or DiskGuard(config.min_free_disk_bytes)
    for root in (config.work_dir, config.cache_dir, config.output_dir):
        guard(root, 0)
    guard(config.runtime_dir, 0)
    return archives


def _check_disk_reserve(
    config: PipelineConfig,
    stage: str,
    target: Path | None = None,
    estimated_bytes: int = 0,
) -> None:
    try:
        DiskGuard(config.min_free_disk_bytes)(
            target or config.work_dir,
            estimated_bytes,
        )
    except DiskSpaceInsufficientError as error:
        raise DiskSpaceInsufficientError(
            f"insufficient disk before {stage}: {error}"
        ) from None


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(
                                follow_symlinks=False
                            ).st_size
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            continue
    return total


def _refresh_known_disk(
    reporter: ProgressReporter,
    config: PipelineConfig,
    *,
    cache: bool = False,
    output: bool = False,
) -> None:
    reporter.update(
        known_work_bytes=_tree_bytes(config.work_dir),
        known_cache_bytes=(
            _tree_bytes(config.cache_dir) if cache else None
        ),
        known_output_bytes=(
            _tree_bytes(config.output_dir) if output else None
        ),
    )


def _input_identity(archives: Iterable[Path]) -> str:
    return stable_hash(
        "wdc200k-statistics-input-v1",
        *(
            f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            for path in archives
        ),
        length=40,
    )


def _registry_identity(config: PipelineConfig, stage: str) -> str:
    return _sha256_path(_producer_registry_path(config, stage))


def _validate_upstream_for_refresh(
    config: PipelineConfig,
    stage: str,
    archives: Iterable[Path],
) -> None:
    upstream = _input_identity(archives)
    for current in STAGES[: STAGES.index(stage)]:
        _validate_stage_registry(
            config,
            current,
            expected_upstream_identity=upstream,
        )
        upstream = _registry_identity(config, current)


def _validate_existing_registry_chain(
    config: PipelineConfig,
    archives: Iterable[Path],
) -> None:
    upstream = _input_identity(archives)
    found_gap = False
    for stage in STAGES:
        path = _producer_registry_path(config, stage)
        if not path.exists():
            found_gap = True
            continue
        if found_gap:
            raise ValueError(
                f"pipeline registry chain has downstream state without {stage} upstream"
            )
        _validate_stage_registry(
            config,
            stage,
            expected_upstream_identity=upstream,
            allow_network_shard_repair=True,
        )
        upstream = _registry_identity(config, stage)


def _fast_model_resume_candidate(config: PipelineConfig) -> bool:
    """Return whether an interrupted model run can skip upstream replay."""
    if (
        not config.resume
        or config.from_stage is not None
        or config.stop_after not in {None, "models"}
        or any(
            (
                config.model_start_marker,
                config.model_ready_marker,
                config.model_text_done_marker,
                config.model_image_done_marker,
            )
        )
    ):
        return False
    return (
        _producer_registry_path(config, "images").is_file()
        and not _producer_registry_path(config, "models").exists()
        and (
            config.work_dir
            / "adapted_model_tasks"
            / "model-task-adapter-manifest.json"
        ).is_file()
        and _model_jobs_path(config).is_file()
    )


def _fast_materialization_resume_candidate(
    config: PipelineConfig,
) -> bool:
    """Return whether a partial materialization may use its certificate."""
    return (
        config.resume
        and config.from_stage is None
        and config.stop_after in {None, "materialize"}
        and _producer_registry_path(config, "models").is_file()
        and not _producer_registry_path(config, "materialize").exists()
        and (config.work_dir / "materialization").is_dir()
    )


def _validate_fast_producer_manifest(
    stage: str,
    path: Path,
    *,
    compact_replacements: bool,
) -> None:
    """Validate completed upstream artifacts using metadata and file sizes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"{stage} producer manifest is incomplete: {path}")
    producer_stage = str(payload.get("stage") or "")
    expected = {
        "selection": {"wdc200k_selection"},
        "structural": {"wdc200k_structural", "wdc200k_validated_selection"},
        "sampling": {"wdc200k_entity_sampling"},
        "pages": {"wdc200k_network_fetch"},
        "asset_planning": {"wdc200k_asset_planning"},
        "images": {
            "wdc200k-unique-image-jobs-v1",
            "wdc200k-image-fetch-v1",
            "wdc200k_asset_materialization",
            "wdc200k_network_fetch",
        },
        "models": {
            "wdc200k_model_task_adapter",
            "wdc200k_model_outputs",
        },
    }[stage]
    if producer_stage not in expected:
        raise ValueError(
            f"unexpected {stage} producer type {producer_stage!r}: {path}"
        )
    root = (
        path.parent.parent
        if producer_stage in {"wdc200k_structural", "wdc200k_validated_selection"}
        else path.parent
    ).resolve()
    for field_name in (
        "completed_shards",
        "entity_plan_shards",
        "image_mapping_shards",
        "bridge_asset_shards",
        "table_asset_link_shards",
        "task_shards",
        "error_shards",
        "extraction_shards",
    ):
        declared = payload.get(field_name)
        if declared is None:
            continue
        if not isinstance(declared, list):
            raise ValueError(
                f"{stage} producer manifest has invalid {field_name}: {path}"
            )
        for item in declared:
            try:
                relative = str(item["path"])
                expected_bytes = int(item["bytes"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{stage} producer manifest has invalid shard: {path}"
                ) from error
            if compact_replacements and relative.startswith(
                ("entities/", "page_refs/", "direct_image_refs/")
            ):
                continue
            artifact = (root / relative).resolve()
            if (
                not artifact.is_relative_to(root)
                or not artifact.is_file()
                or artifact.stat().st_size != expected_bytes
            ):
                raise ValueError(
                    f"{stage} producer shard metadata mismatch: {artifact}"
                )


def _validate_fast_model_resume_chain(
    config: PipelineConfig,
    archives: Iterable[Path],
) -> None:
    """Check immutable upstream identities without rescanning their contents."""
    upstream = _input_identity(archives)
    compact_replacements = _has_complete_sampling_authority(
        config.work_dir / "sampling" / "manifest.json"
    )
    for stage in STAGES[: STAGES.index("models")]:
        path = _producer_registry_path(config, stage)
        registry = _load_stage_registry(path)
        if (
            not registry.complete
            or registry.stage != stage
            or registry.producer_type != _PRODUCER_TYPES[stage]
            or registry.upstream_identity != upstream
            or registry.config_fingerprint
            != _stage_config_fingerprint(config, stage)
            or not registry.producer_manifests
        ):
            raise ValueError(f"pipeline registry identity mismatch: {path}")
        for reference in registry.producer_manifests:
            if (
                not reference.path.is_file()
                or _sha256_path(reference.path) != reference.sha256
            ):
                raise ValueError(
                    f"producer manifest checksum mismatch: {reference.path}"
                )
            _validate_fast_producer_manifest(
                stage,
                reference.path,
                compact_replacements=(
                    stage == "structural" and compact_replacements
                ),
            )
        upstream = _registry_identity(config, stage)
    adapter_manifest = (
        config.work_dir
        / "adapted_model_tasks"
        / "model-task-adapter-manifest.json"
    )
    adapter_payload = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(adapter_payload, dict)
        or adapter_payload.get("complete") is not True
        or adapter_payload.get("stage") != "wdc200k_model_task_adapter"
    ):
        raise ValueError("model task adapter manifest is incomplete")


def _validate_fast_materialization_resume_chain(
    config: PipelineConfig,
    archives: Iterable[Path],
) -> None:
    """Validate the registry envelope without replaying shard contents."""
    _validate_fast_model_resume_chain(config, archives)
    stage = "models"
    path = _producer_registry_path(config, stage)
    registry = _load_stage_registry(path)
    if (
        not registry.complete
        or registry.stage != stage
        or registry.producer_type != _PRODUCER_TYPES[stage]
        or registry.upstream_identity
        != _registry_identity(config, "images")
        or registry.config_fingerprint
        != _stage_config_fingerprint(config, stage)
        or not registry.producer_manifests
    ):
        raise ValueError(f"pipeline registry identity mismatch: {path}")
    for reference in registry.producer_manifests:
        if (
            not reference.path.is_file()
            or _sha256_path(reference.path) != reference.sha256
        ):
            raise ValueError(
                f"producer manifest checksum mismatch: {reference.path}"
            )
        _validate_fast_producer_manifest(
            stage,
            reference.path,
            compact_replacements=False,
        )


def _run_fast_model_resume(
    config: PipelineConfig,
    *,
    archives: tuple[Path, ...],
    page_transport: Any | None,
    image_transport: Any | None,
    extractor: Any | None,
) -> PipelineResult:
    """Resume durable model queues directly, then do one strict final pass."""
    disk_guard = DiskGuard(config.min_free_disk_bytes)
    reporter = ProgressReporter(config, pre_write_guard=disk_guard)
    reporter.start()
    try:
        reporter.update(stage="models", completed_shards=1, total_shards=2)
        args = _runtime_args(config)
        if extractor is None:
            extractor = join_builder.LocalAttributeExtractor(args)
        store = SqliteJobStore(
            _model_jobs_path(config),
            pre_write_guard=disk_guard,
        )
        with _remote_layout_control(
            config,
            extractor=extractor,
            database_path=store.path,
        ):
            result = run_model_stage(
                store,
                extractor,
                workers_by_kind={
                    "text": config.text_model_workers,
                    "image": config.image_model_workers,
                },
                output_root=config.work_dir / "model_outputs",
                records_per_shard=config.records_per_shard,
                model_progress_callback=reporter.update_model_progress,
                endpoint_ready_timeout_seconds=(
                    config.model_endpoint_ready_timeout_seconds
                ),
                pre_write_guard=disk_guard,
            )
        counters = {
            "model_text_tasks": result.text_total,
            "model_image_tasks": result.image_total,
            "model_success": result.success,
            "model_terminal": result.terminal,
        }
        if not result.complete or config.stop_after == "models":
            return PipelineResult(
                status="stopped",
                stage="models",
                statistics_archives=len(archives),
                counters=counters,
            )
    finally:
        reporter.close()
    return _run_pipeline_once(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=extractor,
        _allow_fast_model_resume=False,
    )


def _run_fast_materialization_resume(
    config: PipelineConfig,
    *,
    archives: tuple[Path, ...],
    inputs: MaterializationInputs,
    args: argparse.Namespace,
    extractor: Any | None = None,
) -> PipelineResult:
    """Resume Task 7 directly from a verified materialization certificate."""
    disk_guard = DiskGuard(config.min_free_disk_bytes)
    reporter = ProgressReporter(config, pre_write_guard=disk_guard)
    reporter.start()
    try:
        table_total = config.max_source_tables
        finalize_steps = 15
        total_work = table_total * 2 + finalize_steps
        reporter.update(
            stage="materialize",
            detail="resume materialization",
            completed_shards=0,
            total_shards=total_work,
        )

        def report_validation(event: dict[str, Any]) -> None:
            counters = {
                "materialization_fast_resume": int(
                    event.get("mode") == "fast_resume"
                ),
                "materialization_upstream_validation_completed": int(
                    event.get("completed", 0)
                ),
                "materialization_upstream_validation_total": int(
                    event.get("total", 0)
                ),
            }
            if event.get("resumed_source_units") is not None:
                counters["materialization_resumed_source_units"] = int(
                    event["resumed_source_units"]
                )
            reporter.update(counters=counters)

        def report_materialization(event: dict[str, Any]) -> None:
            phase = str(event.get("phase") or "materialize")
            completed = int(event.get("completed", 0))
            if phase == "query_auto_check_prepare":
                overall = completed
                detail = "prepare global auto-check plans"
            elif phase == "query_auto_check":
                overall = completed
                detail = "auto-check selected query evidence"
            elif phase == "materialize_tables":
                overall = table_total + completed
                detail = "materialize source tables"
            elif phase == "balance_explicit_joins":
                overall = table_total * 2
                detail = "balance explicit joins"
            elif phase == "publish_artifacts":
                overall = table_total * 2 + completed
                detail = "publish dataset artifacts"
            else:
                overall = 0
                detail = phase.replace("_", " ")
            reporter.update(
                detail=detail,
                completed_shards=overall,
                total_shards=total_work,
                counters={
                    "materialization_phase_completed": completed,
                    "materialization_phase_total": int(
                        event.get("total", 0)
                    ),
                    **{
                        f"materialization_{key}": int(event[key])
                        for key in (
                            "plans",
                            "cached_checks",
                            "migrated_legacy_checks",
                        )
                        if event.get(key) is not None
                    },
                },
            )

        if extractor is None:
            extractor = join_builder.LocalAttributeExtractor(args)
        result = materialize_dataset(
            inputs,
            output_root=config.output_dir,
            args=args,
            extractor=extractor,
            records_per_shard=config.records_per_shard,
            pre_write_guard=disk_guard,
            validation_progress_callback=report_validation,
            progress_callback=report_materialization,
        )
        counters = {
            str(key): int(value)
            for key, value in result.stats.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        _write_stage_registry(
            config,
            "materialize",
            producer_manifests=(result.manifest_path,),
            counters=counters,
            upstream_identity=_registry_identity(config, "models"),
            pre_write_guard=disk_guard,
        )
        reporter.update(
            completed_shards=total_work,
            total_shards=total_work,
            counters=counters,
        )
        _refresh_known_disk(reporter, config, output=True)
        return PipelineResult(
            status="complete",
            stage="materialize",
            statistics_archives=len(archives),
            counters=counters,
            output_manifest=result.manifest_path,
        )
    finally:
        reporter.close()


def _runtime_args(config: PipelineConfig) -> argparse.Namespace:
    argv = [
        "--input_dir",
        str(config.input_dir),
        "--output_dir",
        str(config.output_dir),
        "--cache_dir",
        str(config.cache_dir),
        "--max_source_tables",
        str(config.max_source_tables),
        "--max_scanned_files",
        str(max(1, config.max_source_tables)),
        "--max_rows_per_source_table",
        "0",
        "--allow_unbounded",
        "--records_per_shard",
        str(config.records_per_shard),
        "--seed",
        str(config.selection_seed),
        "--max_images_per_entity",
        str(config.max_images_per_entity),
        "--text_asset_chunk_chars",
        str(config.text_asset_chunk_chars),
        "--min_text_asset_chunk_chars",
        str(config.min_text_asset_chunk_chars),
        "--max_text_asset_chunks_per_entity",
        str(config.max_text_asset_chunks_per_entity),
        "--split_by",
        config.split_by,
        "--train_ratio",
        str(config.train_ratio),
        "--dev_ratio",
        str(config.dev_ratio),
        "--test_ratio",
        str(config.test_ratio),
        "--min_column_non_empty_ratio",
        str(config.min_column_non_empty_ratio),
        "--min_recovered_value_ratio",
        str(config.min_recovered_value_ratio),
        "--min_recovery_denominator",
        str(config.min_recovery_denominator),
        "--min_rows_per_output_table",
        str(config.min_rows_per_output_table),
        "--query_rows_per_table",
        str(config.query_rows_per_table),
        "--max_train_query_row_views_per_join",
        str(config.max_train_query_row_views_per_join),
        "--max_query_tables_per_source_table",
        str(config.max_query_tables_per_source_table),
        "--max_query_context_attrs",
        str(config.max_query_context_attrs),
        "--max_target_context_attrs",
        str(config.max_target_context_attrs),
        "--explicit_join_fallback_mode",
        config.explicit_join_fallback_mode,
        "--explicit_join_fallback_ratio",
        str(config.explicit_join_fallback_ratio),
        "--text_model_base_url",
        config.text_model_base_url,
        "--text_model_name",
        config.text_model_name,
        "--image_model_base_url",
        config.image_model_base_url,
        "--image_model_name",
        config.image_model_name,
        "--text_model_workers",
        str(config.text_model_workers),
        "--image_model_workers",
        str(config.image_model_workers),
        "--remote_text_model_workers",
        str(config.remote_text_model_workers),
        "--remote_image_model_workers",
        str(config.remote_image_model_workers),
        "--model_timeout_seconds",
        str(config.model_timeout_seconds),
        "--model_max_retries",
        str(config.model_max_retries),
        "--model_retry_sleep_seconds",
        str(config.model_retry_sleep_seconds),
        "--web_max_retries",
        "0",
        "--web_max_response_seconds",
        str(config.web_max_response_seconds),
        "--min_free_disk_bytes",
        str(config.min_free_disk_bytes),
    ]
    if config.model_endpoint_config:
        argv.extend(
            ["--model_endpoint_config", config.model_endpoint_config]
        )
    if config.text_model_base_urls:
        argv.extend(["--text_model_base_urls", *config.text_model_base_urls])
    if config.text_model_base_urls_file:
        argv.extend(
            ["--text_model_base_urls_file", config.text_model_base_urls_file]
        )
    if config.text_model_api_key:
        argv.extend(["--text_model_api_key", config.text_model_api_key])
    if config.remote_text_model_base_url:
        argv.extend(
            ["--remote_text_model_base_url", config.remote_text_model_base_url]
        )
    if config.remote_text_model_base_urls:
        argv.extend(
            [
                "--remote_text_model_base_urls",
                *config.remote_text_model_base_urls,
            ]
        )
    if config.remote_text_model_base_urls_file:
        argv.extend(
            [
                "--remote_text_model_base_urls_file",
                config.remote_text_model_base_urls_file,
            ]
        )
    if config.remote_text_model_api_key:
        argv.extend(
            ["--remote_text_model_api_key", config.remote_text_model_api_key]
        )
    if config.image_model_base_urls:
        argv.extend(["--image_model_base_urls", *config.image_model_base_urls])
    if config.image_model_base_urls_file:
        argv.extend(
            ["--image_model_base_urls_file", config.image_model_base_urls_file]
        )
    if config.image_model_api_key:
        argv.extend(["--image_model_api_key", config.image_model_api_key])
    if config.remote_image_model_base_url:
        argv.extend(
            [
                "--remote_image_model_base_url",
                config.remote_image_model_base_url,
            ]
        )
    if config.remote_image_model_base_urls:
        argv.extend(
            [
                "--remote_image_model_base_urls",
                *config.remote_image_model_base_urls,
            ]
        )
    if config.remote_image_model_base_urls_file:
        argv.extend(
            [
                "--remote_image_model_base_urls_file",
                config.remote_image_model_base_urls_file,
            ]
        )
    if config.remote_image_model_api_key:
        argv.extend(
            [
                "--remote_image_model_api_key",
                config.remote_image_model_api_key,
            ]
        )
    args = legacy_wdc_builder.parse_args(argv)
    args.max_rows_per_source_table = None
    args.model_endpoint_ready_timeout_seconds = (
        config.model_endpoint_ready_timeout_seconds
    )
    args.materialization_workers = config.materialization_workers
    args.materialization_validation_workers = (
        config.materialization_validation_workers
    )
    args.auto_check_secondary_openai = config.auto_check_secondary_openai
    args.auto_check_api_config_file = config.auto_check_api_config_file
    args.auto_check_openai_env_file = config.auto_check_openai_env_file
    args.auto_check_openai_model = config.auto_check_openai_model
    args.auto_check_openai_base_url = config.auto_check_openai_base_url
    args.auto_check_openai_api_key_env = config.auto_check_openai_api_key_env
    args.auto_check_openai_reasoning_effort = (
        config.auto_check_openai_reasoning_effort
    )
    args.auto_check_openai_verbosity = config.auto_check_openai_verbosity
    args.auto_check_openai_max_output_tokens = (
        config.auto_check_openai_max_output_tokens
    )
    args.auto_check_terra_model = config.auto_check_terra_model
    args.auto_check_terra_reasoning_effort = (
        config.auto_check_terra_reasoning_effort
    )
    args.auto_check_terra_max_output_tokens = (
        config.auto_check_terra_max_output_tokens
    )
    args.auto_check_openai_max_inflight = (
        config.auto_check_openai_max_inflight
    )
    args.auto_check_openai_image_detail = (
        config.auto_check_openai_image_detail
    )
    args.auto_check_openai_image_max_pixels = (
        config.auto_check_openai_image_max_pixels
    )
    args.auto_check_openai_timeout_seconds = (
        config.auto_check_openai_timeout_seconds
    )
    args.auto_check_openai_requests_per_minute = (
        config.auto_check_openai_requests_per_minute
    )
    args.auto_check_openai_tokens_per_minute = (
        config.auto_check_openai_tokens_per_minute
    )
    args.unrecoverable_replacement_rounds = (
        config.unrecoverable_replacement_rounds
    )
    args.unrecoverable_drop_probability = (
        config.unrecoverable_drop_probability
    )
    args.recovery_replacement_round_index = (
        config.recovery_replacement_round_index
    )
    args.model_routing_manifest = (
        str(config.remote_layout_routing_manifest)
        if config.remote_layout_control_url
        and config.remote_layout_routing_manifest is not None
        else None
    )
    return args


@contextmanager
def _remote_layout_control(
    config: PipelineConfig,
    *,
    extractor: Any,
    database_path: Path,
) -> Iterator[RemoteLayoutController | None]:
    """Run the local controller only around durable model execution."""
    if not config.remote_layout_control_url:
        yield None
        return
    scheduler = getattr(extractor, "routing_scheduler", None)
    if scheduler is None:
        raise ValueError(
            "remote layout control requires an authoritative routing scheduler"
        )
    required_paths = (
        config.remote_layout_control_token_file,
        config.remote_layout_controller_id_file,
        config.remote_layout_lock_file,
    )
    if any(path is None for path in required_paths):
        raise ValueError("remote layout control paths are incomplete")
    if (
        not config.remote_layout_primary_image_url
        or not config.remote_layout_switchable_url
    ):
        raise ValueError("remote layout endpoint tunnel mappings are incomplete")
    priority_owner = (
        PriorityGpuOwner(
            config.remote_layout_coordination_dir,
            gpu_ids=("remote:primary_image", "remote:switchable"),
            owner="wdc-remote",
            reclaim_timeout_seconds=(
                config.remote_layout_reclaim_timeout_seconds
            ),
            borrower_stale_seconds=(
                config.remote_layout_borrower_stale_seconds
            ),
            unregistered_grace_seconds=(
                config.remote_layout_unregistered_grace_seconds
            ),
            poll_seconds=config.remote_layout_coordination_poll_seconds,
            borrower_label="EntiTables remote GPU borrower",
            owner_action_label="WDC remote inference",
        )
        if config.remote_layout_coordination_dir is not None
        else None
    )
    controller = RemoteLayoutController(
        RemoteLayoutControllerConfig(
            control_url=config.remote_layout_control_url,
            token_file=config.remote_layout_control_token_file,
            database_path=database_path,
            controller_id_file=config.remote_layout_controller_id_file,
            lock_file=config.remote_layout_lock_file,
            endpoints={
                "primary_image": RemoteLayoutEndpointConfig(
                    endpoint_id="primary_image",
                    base_url=config.remote_layout_primary_image_url,
                ),
                "switchable": RemoteLayoutEndpointConfig(
                    endpoint_id="switchable",
                    base_url=config.remote_layout_switchable_url,
                ),
            },
            text_model_id=config.text_model_name,
            image_model_id=config.image_model_name,
            text_api_key=(
                getattr(extractor, "text_model_api_key", None)
                or config.text_model_api_key
            ),
            image_api_key=(
                getattr(extractor, "image_model_api_key", None)
                or config.image_model_api_key
            ),
            lease_ttl_seconds=config.remote_layout_lease_ttl_seconds,
            lease_renew_seconds=config.remote_layout_lease_renew_seconds,
            request_timeout_seconds=config.remote_layout_request_timeout_seconds,
            reconnect_timeout_seconds=(
                config.remote_layout_reconnect_timeout_seconds
            ),
            operation_timeout_seconds=(
                config.remote_layout_operation_timeout_seconds
            ),
            drain_timeout_seconds=config.remote_layout_drain_timeout_seconds,
            poll_seconds=config.remote_layout_poll_seconds,
            workload_poll_seconds=(
                config.remote_layout_workload_poll_seconds
            ),
            image_burst_stability_seconds=(
                config.remote_layout_stability_seconds
            ),
            endpoint_health_timeout_seconds=(
                config.remote_layout_health_timeout_seconds
            ),
            endpoint_health_stable_polls=(
                config.remote_layout_health_stable_polls
            ),
        ),
        scheduler,
    )
    if priority_owner is not None:
        priority_owner.request_gpus(reason="wdc_model_stage")
    try:
        with controller:
            yield controller
    finally:
        if priority_owner is not None:
            priority_owner.release_gpus(reason="wdc_model_stage_complete")


def _structural_barrier(
    results: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
) -> StructuralStageBarrier:
    manifest_sha256: dict[str, str] = {}
    input_fingerprints: dict[str, str] = {}
    parameter_fingerprints: dict[str, str] = {}
    schema_version = ""
    for result in results:
        path = result.manifest.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = path.as_posix()
        manifest_sha256[key] = _sha256_path(path)
        input_fingerprints[key] = str(payload["input_fingerprint"])
        parameter_fingerprints[key] = str(payload["parameter_fingerprint"])
        current_schema = str(payload["schema_version"])
        if schema_version and schema_version != current_schema:
            raise ValueError("structural manifests use different schemas")
        schema_version = current_schema
    final_payload = json.loads(finalized.manifest.read_text(encoding="utf-8"))
    completed = final_payload.get("completed_shards") or []
    if len(completed) != 1:
        raise ValueError("finalized selection has invalid artifact count")
    return StructuralStageBarrier(
        schema_version=schema_version,
        manifest_count=len(results),
        manifest_sha256=manifest_sha256,
        input_fingerprints=input_fingerprints,
        parameter_fingerprints=parameter_fingerprints,
        final_manifest_sha256=_sha256_path(finalized.manifest),
        final_selection=dict(completed[0]),
    )


def _publish_network_manifest(
    root: Path,
    records: Iterable[dict[str, Any]],
    *,
    policy_fingerprint: str,
    unique: int,
    success: int,
    terminal: int,
    pending: int,
    leased: int,
    records_per_shard: int,
    transport_attempts: dict[str, Any] | None = None,
    transport_attempt_authority: dict[str, Any] | None = None,
    url_completion: dict[str, Any] | None = None,
    pre_write_guard: PreWriteGuard | None = None,
) -> Path:
    if records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")
    manifest_path = root / "network-manifest.json"
    counts = {
        "unique": unique,
        "success": success,
        "terminal": terminal,
        "pending": pending,
        "leased": leased,
    }
    previous: dict[str, CompletedShard] = {}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = payload.get("counts") or {}
        if not (
            payload.get("stage") == "wdc200k_network_fetch"
            and payload.get("complete") is True
            and payload.get("policy_fingerprint") == policy_fingerprint
            and int(payload.get("records_per_shard", -1))
            == records_per_shard
            and declared == counts
            and payload.get("transport_attempts") == transport_attempts
            and payload.get("transport_attempt_authority")
            == transport_attempt_authority
            and payload.get("url_completion") == url_completion
        ):
            raise ValueError(
                f"network manifest conflicts with durable state: {manifest_path}"
            )
        for item in payload.get("completed_shards") or []:
            completed = CompletedShard(
                path=str(item["path"]),
                records=int(item["records"]),
                bytes=int(item["bytes"]),
                sha256=str(item["sha256"]),
            )
            previous[completed.path] = completed

    completed_shards: list[CompletedShard] = []
    chunk: list[dict[str, Any]] = []

    def publish_chunk(index: int, batch: list[dict[str, Any]]) -> None:
        relative_path = f"outcomes/part-{index:05d}.jsonl"
        prior = previous.get(relative_path)
        if (
            prior is not None
            and prior.records == len(batch)
            and validate_completed_shard(prior, root)
        ):
            completed_shards.append(prior)
            return
        path = root / relative_path
        writer = AtomicJsonlShard(
            path,
            pre_write_guard=pre_write_guard,
        )
        try:
            for record in batch:
                writer.write(record)
            committed = writer.commit()
        except BaseException:
            writer.abort()
            raise
        completed_shards.append(
            CompletedShard(
                path=relative_path,
                records=committed.records,
                bytes=committed.bytes,
                sha256=committed.sha256,
            )
        )

    for record in records:
        chunk.append(record)
        if len(chunk) >= records_per_shard:
            publish_chunk(len(completed_shards), chunk)
            chunk = []
    if chunk or not completed_shards:
        publish_chunk(len(completed_shards), chunk)
    if sum(shard.records for shard in completed_shards) != unique:
        raise ValueError("network outcome count does not match durable jobs")
    manifest_payload = {
        "stage": "wdc200k_network_fetch",
        "schema_version": "wdc200k-network-fetch-v1",
        "policy_fingerprint": policy_fingerprint,
        "records_per_shard": records_per_shard,
        "counts": counts,
        "completed_shards": [asdict(shard) for shard in completed_shards],
        "complete": True,
    }
    if transport_attempts is not None:
        manifest_payload.update(
            {
                "transport_attempts": transport_attempts,
                "transport_attempt_authority": transport_attempt_authority,
                "url_completion": url_completion,
            }
        )
    _atomic_json(
        manifest_path,
        manifest_payload,
        pre_write_guard=pre_write_guard,
    )
    return manifest_path


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"JSONL record is not an object: {path}")
                yield record


def _selection_chunks(
    path: Path,
    shard_tables: int,
) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for record in _iter_jsonl(path):
        chunk.append(record)
        if len(chunk) >= shard_tables:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _structural_exact_counts(
    root: Path,
    results: Sequence[StructuralExpansionResult],
    config: PipelineConfig,
    pre_write_guard: PreWriteGuard | None = None,
) -> dict[str, int]:
    database_path = root / "structural-counts.sqlite3"
    write_tracker = GuardedWriteTracker(database_path, pre_write_guard)
    write_tracker.before_write(64 * 1024)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS page_urls "
            "(url_key TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM page_urls")
        for result in results:
            for record in _iter_jsonl(result.page_refs):
                encoded = json.dumps(record, ensure_ascii=False)
                write_tracker.before_write(
                    4096 + 2 * len(encoded.encode("utf-8"))
                )
                connection.execute(
                    "INSERT OR IGNORE INTO page_urls (url_key) VALUES (?)",
                    (str(record["url_key"]),),
                )
        unique_pages = int(
            connection.execute("SELECT COUNT(*) FROM page_urls").fetchone()[0]
        )
        write_tracker.before_commit(0)
    entities = sum(result.entities_count for result in results)
    direct_images = sum(
        result.direct_image_references for result in results
    )
    page_references = sum(result.page_references for result in results)
    validated_tables = sum(result.tables for result in results)
    image_requests = entities * config.max_image_attempts_per_entity * 2
    page_upper_bytes = unique_pages * config.web_max_page_bytes
    image_upper_bytes = image_requests * config.web_max_image_bytes
    estimated_next_stage_bytes = (
        unique_pages * config.estimated_page_result_bytes
        + image_requests * config.estimated_image_result_bytes
    )
    structural_bytes = sum(
        path.stat().st_size
        for result in results
        for path in (
            result.source_tables,
            result.entities,
            result.page_refs,
            result.direct_image_refs,
            result.structural_failures,
            result.validated_selection,
            result.manifest,
        )
        if path.is_file()
    )
    return {
        "validated_tables": validated_tables,
        "entities": entities,
        "page_references": page_references,
        "unique_page_urls": unique_pages,
        "direct_image_references": direct_images,
        "page_request_upper_bound": unique_pages,
        "image_request_upper_bound": image_requests,
        "page_disk_upper_bound_bytes": page_upper_bytes,
        "image_disk_upper_bound_bytes": image_upper_bytes,
        "network_disk_upper_bound_bytes": page_upper_bytes
        + image_upper_bytes,
        "estimated_next_stage_bytes": estimated_next_stage_bytes,
        "structural_output_bytes": structural_bytes,
    }


def _run_selection_and_structural(
    config: PipelineConfig,
    reporter: ProgressReporter,
    *,
    input_identity: str,
    selection_only: bool = False,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[
    tuple[StructuralExpansionResult, ...],
    FinalizedSelectionResult | None,
    dict[str, int],
]:
    policy = SelectionPolicy(
        target_tables=config.max_source_tables,
        seed=config.selection_seed,
        top100_policy=config.top100_policy,
        top100_per_class=config.top100_per_class,
        include_rest=config.include_rest,
        min_candidate_rows=config.min_candidate_rows,
        min_candidate_columns=config.min_candidate_columns,
        minimum3_fraction=config.minimum3_fraction,
        rest_base_per_class=0,
        class_cap=config.class_max_tables,
    )
    reporter.update(
        stage="selection",
        detail="scan statistics archives",
        completed_shards=0,
        total_shards=0,
    )

    def report_selection(event: dict[str, object]) -> None:
        phase = str(event.get("phase") or "selection")
        counters: dict[str, int] = {}
        if event.get("candidates") is not None:
            counters["selection_candidates_scanned"] = int(
                event["candidates"]
            )
        reporter.update(
            detail=phase.replace("_", " "),
            completed_shards=int(event["completed"]),
            total_shards=int(event["total"]),
            counters=counters,
        )

    selected_count, reserve_count = run_selection(
        config.input_dir,
        config.work_dir,
        policy,
        pre_write_guard=pre_write_guard,
        progress_callback=report_selection,
    )
    selection_dir = config.work_dir / "selection"
    selection_manifest = selection_dir / "manifest.json"
    active_selection = _load_active_recovery_selection(config)
    selection_manifests = [selection_manifest]
    selected_path = selection_dir / "selected_tables.jsonl"
    if active_selection is not None:
        selected_path = active_selection.path
        selected_count = active_selection.records
        selection_manifests.append(active_selection.manifest_path)
    selection_counters = {
        "selected_tables": selected_count,
        "reserve_tables": reserve_count,
    }
    _write_stage_registry(
        config,
        "selection",
        producer_manifests=selection_manifests,
        counters=selection_counters,
        upstream_identity=input_identity,
        pre_write_guard=pre_write_guard,
    )
    reporter.update(counters=selection_counters)
    if selection_only:
        return (), None, selection_counters

    sampling_manifest = config.work_dir / "sampling" / "manifest.json"
    structural_registry = _producer_registry_path(config, "structural")
    if (
        _has_complete_sampling_authority(sampling_manifest)
        and structural_registry.is_file()
    ):
        authority = validate_sampling_source_authority(
            sampling_manifest,
            structural_output_root=config.work_dir / "structural",
        )
        structural_root = config.work_dir / "structural"
        reconstructed: list[StructuralExpansionResult] = []
        for manifest_path, source_path in zip(
            authority.structural_manifests,
            authority.source_tables,
        ):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed = payload.get("completed_shards") or []

            def artifact_record(prefix: str) -> dict[str, Any]:
                return next(
                    raw
                    for raw in completed
                    if str(raw["path"]).startswith(prefix)
                )

            def artifact(prefix: str) -> Path:
                return structural_root / str(artifact_record(prefix)["path"])

            source_record = artifact_record("source_tables/")
            if (structural_root / str(source_record["path"])).resolve() != (
                source_path.resolve()
            ):
                raise ValueError("compact source shard path mismatch")

            reconstructed.append(
                StructuralExpansionResult(
                    source_tables=source_path,
                    entities=artifact("entities/"),
                    page_refs=artifact("page_refs/"),
                    direct_image_refs=artifact("direct_image_refs/"),
                    structural_failures=artifact("structural_failures/"),
                    validated_selection=artifact("selection/validated-"),
                    manifest=manifest_path,
                    tables=int(source_record["records"]),
                    entities_count=0,
                    page_references=0,
                    direct_image_references=0,
                )
            )
        final_manifest = (
            structural_root
            / "stage_manifests"
            / "validated-selection-global.json"
        )
        final_payload = json.loads(final_manifest.read_text(encoding="utf-8"))
        final_completed = final_payload.get("completed_shards") or []
        if len(final_completed) != 1:
            raise ValueError("compact final selection authority is invalid")
        finalized = FinalizedSelectionResult(
            validated_selection=structural_root / str(final_completed[0]["path"]),
            manifest=final_manifest,
            tables=int(final_completed[0]["records"]),
        )
        counters = dict(_load_stage_registry(structural_registry).counters)
        return tuple(reconstructed), finalized, counters

    reserve_path = selection_dir / "reserve_tables.jsonl"
    reserve_database = selection_dir / "reserve.sqlite3"
    if pre_write_guard is not None:
        pre_write_guard(reserve_database, 0)
    reserve_manager = (
        ReserveManager.open(
            reserve_database,
            policy,
            pre_write_guard=pre_write_guard,
        )
        if reserve_database.exists()
        else ReserveManager.create_from_jsonl(
            reserve_database,
            reserve_path=reserve_path,
            selected_path=selected_path,
            policy=policy,
            pre_write_guard=pre_write_guard,
        )
    )
    reporter.update(
        stage="structural",
        detail="expand source tables",
        completed_shards=0,
        total_shards=selected_count,
    )
    structural_root = config.work_dir / "structural"
    results = []
    validated_tables = 0
    emitted_entities = 0
    for index, records in enumerate(
        _selection_chunks(selected_path, config.selection_shard_tables)
    ):
        shard_baseline_tables = validated_tables

        def report_structural(event: dict[str, Any]) -> None:
            phase = str(event.get("phase") or "expand_table")
            relative_path = str(
                event.get("replacement_path")
                or event.get("relative_path")
                or ""
            )
            detail = phase.replace("_", " ")
            if relative_path:
                detail = f"{detail}: {Path(relative_path).name}"
            local_tables = int(event.get("completed", 0))
            reporter.update(
                detail=detail,
                completed_shards=shard_baseline_tables + local_tables,
                total_shards=selected_count,
                counters={
                    "structural_tables_completed_live": (
                        shard_baseline_tables + local_tables
                    ),
                    "entities_live": (
                        emitted_entities + int(event.get("entities", 0))
                    ),
                },
            )

        result = expand_selected_shard(
            records,
            output_root=structural_root,
            input_root=config.input_dir,
            reserve_manager=reserve_manager,
            shard_id=f"{index:05d}",
            min_rows=1,
            min_cols=1,
            pre_write_guard=pre_write_guard,
            progress_callback=report_structural,
        )
        results.append(result)
        validated_tables += result.tables
        emitted_entities += result.entities_count
        reporter.update(
            detail="structural shard complete",
            completed_shards=validated_tables,
            total_shards=selected_count,
            counters={
                "validated_tables": validated_tables,
                "entities": emitted_entities,
            },
        )
    finalized = finalize_validated_selection(
        (result.manifest for result in results),
        output_root=structural_root,
        target_tables=config.max_source_tables,
        pre_write_guard=pre_write_guard,
    )
    counters = {
        **selection_counters,
        **_structural_exact_counts(
            structural_root,
            results,
            config,
            pre_write_guard=pre_write_guard,
        ),
    }
    _write_stage_registry(
        config,
        "structural",
        producer_manifests=(
            *(result.manifest for result in results),
            finalized.manifest,
        ),
        counters=counters,
        upstream_identity=_registry_identity(config, "selection"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        detail="structural expansion complete",
        completed_shards=selected_count,
        total_shards=selected_count,
        counters=counters,
        known_work_bytes=counters["structural_output_bytes"],
    )
    return tuple(results), finalized, counters


def _new_web_transport(
    config: PipelineConfig,
    namespace: str,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> Any:
    return legacy_wdc_builder.WdcWebClient(
        config.cache_dir / f"{namespace}_transport",
        max_retries=0,
        max_page_bytes=config.web_max_page_bytes,
        max_image_bytes=config.web_max_image_bytes,
        min_free_disk_bytes=config.min_free_disk_bytes,
        max_response_seconds=config.web_max_response_seconds,
        host_delay=0.0,
        pre_write_guard=pre_write_guard,
    )


class _EpochElapsedNormalizer:
    """Map raw tracker elapsed values to each epoch's first callback."""

    def __init__(self) -> None:
        self._first_elapsed_by_epoch: dict[str, float] = {}

    def __call__(self, snapshot: UrlProgressSnapshot) -> UrlProgressSnapshot:
        if not isinstance(snapshot, UrlProgressSnapshot):
            raise ValueError("URL progress snapshot is invalid")
        first_elapsed = self._first_elapsed_by_epoch.setdefault(
            snapshot.execution_epoch, snapshot.epoch_elapsed_seconds
        )
        return replace(
            snapshot,
            epoch_elapsed_seconds=(
                snapshot.epoch_elapsed_seconds - first_elapsed
            ),
        )


def _page_refs(
    sampling: SamplingResult,
) -> Iterator[dict[str, Any]]:
    authority = validate_sampling_consumed_paths(
        sampling.manifest_path,
        sampling.artifact_paths,
    )
    for path in authority.artifact_paths["sampled_page_refs"]:
        yield from _iter_jsonl(path)


def _run_sampling(
    config: PipelineConfig,
    reporter: ProgressReporter,
    structural: Sequence[StructuralExpansionResult],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> SamplingResult:
    total_shards = len(structural)
    reporter.update(
        stage="sampling",
        detail="sample structural shards",
        completed_shards=0,
        total_shards=total_shards,
    )
    expansion = _load_sampling_expansion_state(config)
    if expansion.round_index != config.sampling_expansion_round_index:
        raise ValueError("sampling expansion/config round mismatch")

    def report_sampling(event: dict[str, Any]) -> None:
        reporter.update(
            detail="sample structural shards",
            completed_shards=int(event["completed"]),
            total_shards=int(event["total"]),
            counters={
                "sampling_shards_completed": int(event["completed"]),
                "sampling_shards_total": int(event["total"]),
            },
        )

    result = sample_structural_artifacts(
        structural_output_root=config.work_dir / "structural",
        structural_manifests=tuple(item.manifest for item in structural),
        output_root=config.work_dir / "sampling",
        policy=SamplingPolicy(
            sampled_entities_per_table=config.sampled_entities_per_table,
            entity_sampling_seed=config.entity_sampling_seed,
            query_rows_per_table=config.query_rows_per_table,
            min_column_non_empty_ratio=config.min_column_non_empty_ratio,
            min_recovered_value_ratio=config.min_recovered_value_ratio,
            min_recovery_denominator=config.min_recovery_denominator,
            min_rows_per_output_table=config.min_rows_per_output_table,
            global_entity_budget=config.global_entity_budget,
        ),
        per_table_entity_limits=expansion.limits,
        progress_callback=report_sampling,
        pre_write_guard=pre_write_guard,
    )
    counters = {
        "eligible_tables": result.eligible_tables,
        "prefilter_rejected_tables": result.rejected_tables,
        "sampled_entities": result.sampled_entities,
        "sampled_page_references": result.sampled_page_refs,
        "sampled_direct_image_references": result.sampled_direct_image_refs,
        "progressively_expanded_tables": len(expansion.limits),
        "sampling_expansion_round": expansion.round_index,
        **{
            f"sampled_{stratum}_entities": count
            for stratum, count in result.strata.items()
        },
    }
    _write_stage_registry(
        config,
        "sampling",
        producer_manifests=(result.manifest_path,),
        counters=counters,
        upstream_identity=_registry_identity(config, "structural"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        detail="sampling complete",
        completed_shards=total_shards,
        total_shards=total_shards,
        counters=counters,
    )
    return result


def _run_pages(
    config: PipelineConfig,
    reporter: ProgressReporter,
    sampling: SamplingResult,
    transport: Any,
    *,
    after_cache_write: Any | None = None,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[FetchResult, dict[str, Any], Path]:
    reporter.update(stage="pages", completed_shards=0, total_shards=1)
    root = config.work_dir / "page_jobs"
    policy = FetchPolicy(
        retries=0,
        deadline_seconds=config.web_max_response_seconds,
        global_concurrency=config.web_global_concurrency,
        per_host_concurrency=config.web_per_host_concurrency,
    )
    jobs_path = root / "jobs.sqlite3"
    outcomes_path = config.cache_dir / "page_cache" / "outcomes.sqlite3"
    if pre_write_guard is not None:
        pre_write_guard(jobs_path, 0)
        pre_write_guard(outcomes_path, 0)
    SqliteJobStore(jobs_path, pre_write_guard=pre_write_guard)
    reconcile_page_jobs_from_outcomes(
        jobs_path,
        outcomes_path,
        policy.fingerprint,
        pre_write_guard=pre_write_guard,
    )
    page_completed = 0
    page_status_counts: dict[str, int] = {}

    def after_page_outcome(record: dict[str, Any]) -> None:
        nonlocal page_completed
        status = str(record.get("status") or "terminal")
        page_completed += 1
        page_status_counts[status] = page_status_counts.get(status, 0) + 1
        reporter.update(
            counters={
                "page_completed_live": page_completed,
                f"page_{status}_live": page_status_counts[status],
            }
        )
        if after_cache_write is not None:
            after_cache_write(record)

    normalize_page_elapsed = _EpochElapsedNormalizer()

    def page_url_progress(snapshot: UrlProgressSnapshot) -> None:
        reporter.update(url_snapshot=normalize_page_elapsed(snapshot))

    result = fetch_unique_pages(
        _page_refs(sampling),
        SqliteJobStore(jobs_path, pre_write_guard=pre_write_guard),
        transport,
        policy,
        outcomes_path=outcomes_path,
        failure_path=root / "page-failures.jsonl",
        progress_path=root / "page-progress.json",
        after_cache_write=after_page_outcome,
        progress_callback=page_url_progress,
        pre_write_guard=pre_write_guard,
    )
    page_validation_database = root / "validation.sqlite3"
    if pre_write_guard is not None:
        pre_write_guard(page_validation_database, 0)
    snapshot = validate_complete_page_fetch(
        result,
        _page_refs(sampling),
        validation_database=page_validation_database,
        pre_write_guard=pre_write_guard,
    )
    reporter.publish()
    url_completion = reporter.stage_completion_summary("pages")
    transport_authority = {
        "database_path": str(result.outcomes_path.resolve()),
        "job_store_path": str(result.job_store_path.resolve()),
        "job_kind": result.job_kind,
        "job_id_prefix": result.policy_fingerprint,
        "job_scope": _read_only_job_scope(
            result.job_store_path,
            result.job_kind,
        ),
    }
    network_manifest = _publish_network_manifest(
        root / "network",
        iter_page_outcomes(
            result.outcomes_path,
            result.policy_fingerprint,
            job_store_path=result.job_store_path,
            job_kind=result.job_kind,
        ),
        policy_fingerprint=result.policy_fingerprint,
        unique=result.unique,
        success=result.success,
        terminal=result.terminal,
        pending=result.remaining,
        leased=result.leased,
        records_per_shard=config.records_per_shard,
        transport_attempts=result.transport_attempt_summary,
        transport_attempt_authority=transport_authority,
        url_completion=url_completion,
        pre_write_guard=pre_write_guard,
    )
    counters = {
        "unique_page_jobs": result.unique,
        "page_success": result.success,
        "page_terminal": result.terminal,
        "page_remaining": result.remaining,
        **_network_telemetry_counters(
            "pages",
            result.transport_attempt_summary,
            url_completion,
        ),
    }
    _write_stage_registry(
        config,
        "pages",
        producer_manifests=(network_manifest,),
        counters=counters,
        upstream_identity=_registry_identity(config, "sampling"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        completed_shards=1,
        total_shards=1,
        counters=counters,
    )
    _refresh_known_disk(reporter, config, cache=True)
    return result, snapshot, network_manifest


def _run_asset_planning(
    config: PipelineConfig,
    reporter: ProgressReporter,
    sampling: SamplingResult,
    page_result: FetchResult,
    page_snapshot: dict[str, Any],
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[AssetPlanShards, str]:
    page_work = sampling.sampled_page_refs
    entity_work = sampling.sampled_entities
    total_work = page_work + entity_work
    reporter.update(
        stage="asset_planning",
        detail="index page fanout",
        completed_shards=0,
        total_shards=total_work,
    )
    structural_identity = _sha256_path(sampling.manifest_path)
    planning_input = asset_planning_input_fingerprint(
        structural_identity,
        str(page_snapshot["identity"]),
    )
    entity_page_join = (
        config.work_dir / "asset_planning" / "entity-pages.sqlite3"
    )
    if pre_write_guard is not None:
        pre_write_guard(entity_page_join, 0)
    page_fanout = iter_page_fanout(
        page_result.outcomes_path,
        page_result.policy_fingerprint,
        job_store_path=page_result.job_store_path,
    )

    def tracked_page_fanout() -> Iterator[dict[str, Any]]:
        completed = 0
        for record in page_fanout:
            yield record
            completed += 1
            if completed % 1_000 == 0:
                reporter.update(
                    detail="index page fanout",
                    completed_shards=completed,
                    counters={"asset_page_refs_indexed": completed},
                )
        reporter.update(
            detail="index page fanout",
            completed_shards=completed,
            counters={"asset_page_refs_indexed": completed},
        )

    entity_pages = iter_entity_page_join(
        sampling.artifact_paths["sampled_entities"],
        tracked_page_fanout(),
        join_path=entity_page_join,
        pre_write_guard=pre_write_guard,
    )
    budget = ImageBudget(
        attempts_per_entity=config.max_image_attempts_per_entity,
        retained_per_entity=config.max_images_per_entity,
    )

    def report_asset_planning(event: dict[str, Any]) -> None:
        completed_entities = int(event["completed"])
        reporter.update(
            detail="plan entity assets",
            completed_shards=page_work + completed_entities,
            total_shards=total_work,
            counters={
                "asset_entities_planned_live": completed_entities,
                "asset_entities_total": entity_work,
            },
        )

    planned = persist_entity_asset_plans(
        entity_pages,
        output_root=config.work_dir / "asset_planning",
        input_fingerprint=planning_input,
        budget=budget,
        records_per_shard=config.records_per_shard,
        total_entities=entity_work,
        progress_callback=report_asset_planning,
        pre_write_guard=pre_write_guard,
    )
    validate_asset_plan_shards(
        planned,
        expected_input_fingerprint=planning_input,
    )
    counters = {
        "planned_entities": planned.entities,
        "planned_image_references": planned.image_mappings,
    }
    _write_stage_registry(
        config,
        "asset_planning",
        producer_manifests=(planned.manifest_path,),
        counters=counters,
        upstream_identity=_registry_identity(config, "pages"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        detail="asset planning complete",
        completed_shards=total_work,
        total_shards=total_work,
        counters=counters,
    )
    return planned, planning_input


def _run_images(
    config: PipelineConfig,
    reporter: ProgressReporter,
    planned: AssetPlanShards,
    transport: Any,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[
    UniqueImageJobs,
    ImageFetchResult,
    MaterializedAssetShards,
    AssetStageBarrier,
    Path,
]:
    reporter.update(stage="images", completed_shards=0, total_shards=3)
    root = config.work_dir / "image_jobs"
    unique_jobs = build_unique_image_jobs(
        planned,
        root / "unique-images.jsonl",
        pre_write_guard=pre_write_guard,
    )
    unique_validation_database = root / "unique-validation.sqlite3"
    if pre_write_guard is not None:
        pre_write_guard(unique_validation_database, 0)
    validate_unique_image_jobs(
        unique_jobs,
        planned=planned,
        validation_database=unique_validation_database,
        pre_write_guard=pre_write_guard,
    )
    reporter.update(completed_shards=1, total_shards=3)
    policy = FetchPolicy(
        retries=0,
        deadline_seconds=config.web_max_response_seconds,
        global_concurrency=config.web_global_concurrency,
        per_host_concurrency=config.web_per_host_concurrency,
        policy_version="wdc200k-image-fetch-v1",
    )
    image_completed = 0

    def after_image_outcome(_record: dict[str, Any]) -> None:
        nonlocal image_completed
        image_completed += 1
        reporter.update(counters={"image_completed_live": image_completed})

    normalize_image_elapsed = _EpochElapsedNormalizer()

    def image_url_progress(snapshot: UrlProgressSnapshot) -> None:
        reporter.update(url_snapshot=normalize_image_elapsed(snapshot))

    if pre_write_guard is not None:
        pre_write_guard(root / "jobs.sqlite3", 0)
    image_result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(
            root / "jobs.sqlite3",
            pre_write_guard=pre_write_guard,
        ),
        transport,
        policy,
        outcomes_path=config.cache_dir / "image_cache" / "outcomes.sqlite3",
        image_dir=config.cache_dir / "images",
        after_cache_write=after_image_outcome,
        progress_callback=image_url_progress,
        pre_write_guard=pre_write_guard,
    )
    validate_complete_image_fetch(
        image_result,
        unique_jobs=unique_jobs,
        pre_write_guard=pre_write_guard,
    )
    reporter.update(completed_shards=2, total_shards=3)
    asset_input = asset_materialization_input_fingerprint(
        planned.manifest_path,
        image_result.fetch_manifest_path,
    )
    budget = ImageBudget(
        attempts_per_entity=config.max_image_attempts_per_entity,
        retained_per_entity=config.max_images_per_entity,
    )
    materialized = materialize_asset_shards(
        planned,
        fetch_result=image_result,
        output_root=config.work_dir / "materialized_assets",
        input_fingerprint=asset_input,
        budget=budget,
        text_asset_chunk_chars=config.text_asset_chunk_chars,
        min_text_asset_chunk_chars=config.min_text_asset_chunk_chars,
        max_text_asset_chunks_per_entity=(
            config.max_text_asset_chunks_per_entity
        ),
        records_per_shard=config.records_per_shard,
        pre_write_guard=pre_write_guard,
    )
    materialized, barrier = validate_materialized_asset_shards(
        materialized,
        planned=planned,
        image_fetch_result=image_result,
        expected_input_fingerprint=asset_input,
        pre_write_guard=pre_write_guard,
    )
    reporter.publish()
    url_completion = reporter.stage_completion_summary("images")
    transport_authority = {
        "database_path": str(image_result.outcomes_path.resolve()),
        "job_store_path": str(image_result.job_store_path.resolve()),
        "job_kind": image_result.job_kind,
        "job_id_prefix": image_result.job_kind,
        "job_scope": _read_only_job_scope(
            image_result.job_store_path,
            image_result.job_kind,
        ),
    }
    network_manifest = _publish_network_manifest(
        root / "network",
        iter_image_outcomes(
            image_result.outcomes_path,
            image_result.policy_fingerprint,
            job_store_path=image_result.job_store_path,
            job_kind=image_result.job_kind,
        ),
        policy_fingerprint=image_result.policy_fingerprint,
        unique=image_result.unique,
        success=image_result.success,
        terminal=image_result.terminal,
        pending=image_result.remaining,
        leased=image_result.leased,
        records_per_shard=config.records_per_shard,
        transport_attempts=image_result.transport_attempt_summary,
        transport_attempt_authority=transport_authority,
        url_completion=url_completion,
        pre_write_guard=pre_write_guard,
    )
    counters = {
        "unique_image_jobs": unique_jobs.records,
        "image_success": image_result.success,
        "image_terminal": image_result.terminal,
        "bridge_assets": materialized.bridge_assets,
        "table_asset_links": materialized.table_asset_links,
        "image_outcomes": image_result.outcomes_count,
        **_network_telemetry_counters(
            "images",
            image_result.transport_attempt_summary,
            url_completion,
        ),
    }
    _write_stage_registry(
        config,
        "images",
        producer_manifests=(
            unique_jobs.manifest_path,
            image_result.fetch_manifest_path,
            materialized.manifest_path,
            network_manifest,
        ),
        counters=counters,
        upstream_identity=_registry_identity(config, "asset_planning"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        completed_shards=3,
        total_shards=3,
        counters=counters,
    )
    _refresh_known_disk(reporter, config, cache=True)
    return unique_jobs, image_result, materialized, barrier, network_manifest


def _run_models(
    config: PipelineConfig,
    reporter: ProgressReporter,
    structural: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
    structural_barrier: StructuralStageBarrier,
    sampling: SamplingResult,
    materialized_assets: MaterializedAssetShards,
    assets_barrier: AssetStageBarrier,
    network_manifests: Sequence[Path],
    extractor: Any,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[AdaptedModelTasks, ModelStageResult, ModelStageAuthority, argparse.Namespace]:
    reporter.update(stage="models", completed_shards=0, total_shards=2)
    args = _runtime_args(config)

    def adapter_progress(snapshot: ModelTaskAdapterProgress) -> None:
        reporter.update(
            detail=(
                f"adapter {snapshot.phase}: "
                f"{snapshot.completed_shards}/{snapshot.total_shards}"
            ),
            counters={
                "model_adapter_shards_completed": snapshot.completed_shards,
                "model_adapter_shards_total": snapshot.total_shards,
                "model_adapter_tasks_live": snapshot.tasks,
                "model_adapter_errors_live": snapshot.errors,
            },
        )

    adapted = adapt_model_tasks_from_manifests(
        structural_output_root=config.work_dir / "structural",
        structural_manifests=(item.manifest for item in structural),
        finalized_selection_manifest=finalized.manifest,
        structural_barrier=structural_barrier,
        assets_manifest=materialized_assets.manifest_path,
        assets_barrier=assets_barrier,
        output_root=config.work_dir / "adapted_model_tasks",
        args=args,
        records_per_shard=config.records_per_shard,
        pre_write_guard=pre_write_guard,
        sampled_entity_paths=sampling.artifact_paths["sampled_entities"],
        sampling_manifest=sampling.manifest_path,
        progress_callback=adapter_progress,
    )
    reporter.update(completed_shards=1, total_shards=2)
    model_jobs_path = _model_jobs_path(config)
    if pre_write_guard is not None:
        pre_write_guard(model_jobs_path, 0)
    store = SqliteJobStore(
        model_jobs_path,
        pre_write_guard=pre_write_guard,
    )
    jobset = enqueue_model_tasks(
        (
            record
            for path in adapted.task_paths
            for record in _iter_jsonl(path)
        ),
        store,
        args=args,
        input_fingerprint=adapted.input_fingerprint,
        text_input_fingerprint=adapted.input_fingerprint,
        image_input_fingerprint=adapted.input_fingerprint,
        staging_dir=config.work_dir / "model_outputs" / "enqueue-staging",
        pre_write_guard=pre_write_guard,
    )
    run_fingerprint = config.run_fingerprint or stable_hash(
        "wdc200k-model-run-v1",
        adapted.input_fingerprint,
        jobset.text_fingerprint,
        jobset.image_fingerprint,
        length=40,
    )
    with _remote_layout_control(
        config,
        extractor=extractor,
        database_path=store.path,
    ):
        result = run_model_stage(
            store,
            extractor,
            jobset=jobset,
            workers_by_kind={
                "text": config.text_model_workers,
                "image": config.image_model_workers,
            },
            output_root=config.work_dir / "model_outputs",
            records_per_shard=config.records_per_shard,
            model_progress_callback=reporter.update_model_progress,
            start_marker=config.model_start_marker,
            ready_marker=config.model_ready_marker,
            network_manifests=network_manifests,
            assets_manifest=materialized_assets.manifest_path,
            assets_barrier=assets_barrier,
            ready_timeout_seconds=config.model_ready_timeout_seconds,
            endpoint_ready_timeout_seconds=(
                config.model_endpoint_ready_timeout_seconds
            ),
            text_done_marker=config.model_text_done_marker,
            image_done_marker=config.model_image_done_marker,
            run_fingerprint=run_fingerprint,
            pre_write_guard=pre_write_guard,
        )
    authority = ModelStageAuthority.current(args)
    model_validation_store = (
        config.work_dir / "model_outputs" / "validation.sqlite3"
    )
    if pre_write_guard is not None:
        pre_write_guard(model_validation_store, 0)
    validate_model_stage_for_adapter(
        result,
        adapted,
        args=args,
        authority=authority,
        validation_store_path=model_validation_store,
    )
    counters = {
        "model_text_tasks": result.text_total,
        "model_image_tasks": result.image_total,
        "model_success": result.success,
        "model_terminal": result.terminal,
    }
    auto_check = join_builder.summarize_model_auto_check_records(
        result.extraction_paths
    )
    counters.update(
        {
            "model_auto_check_reviewed": int(auto_check["reviewed"]),
            "model_auto_check_supported": int(auto_check["supported"]),
            "model_auto_check_filtered": int(auto_check["filtered"]),
            "model_auto_check_errors": int(auto_check["errors"]),
        }
    )
    _write_stage_registry(
        config,
        "models",
        producer_manifests=(adapted.manifest_path, result.manifest_path),
        counters=counters,
        upstream_identity=_registry_identity(config, "images"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        completed_shards=2,
        total_shards=2,
        counters=counters,
    )
    _refresh_known_disk(reporter, config)
    return adapted, result, authority, args


def _run_materialize(
    config: PipelineConfig,
    reporter: ProgressReporter,
    structural: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
    structural_barrier: StructuralStageBarrier,
    sampling: SamplingResult,
    page_result: FetchResult,
    planned: AssetPlanShards,
    unique_jobs: UniqueImageJobs,
    image_result: ImageFetchResult,
    materialized_assets: MaterializedAssetShards,
    adapted: AdaptedModelTasks,
    model_result: ModelStageResult,
    authority: ModelStageAuthority,
    args: argparse.Namespace,
    extractor: Any,
    *,
    pre_write_guard: PreWriteGuard | None = None,
) -> MaterializationResult:
    table_total = config.max_source_tables
    finalize_steps = 15
    total_work = table_total * 3 + finalize_steps
    reporter.update(
        stage="materialize",
        detail="validate upstream",
        completed_shards=0,
        total_shards=total_work,
    )

    def report_validation(event: dict[str, Any]) -> None:
        scope = str(
            event.get("validation_scope") or "upstream_validation"
        )
        mode = str(event.get("mode") or "strict")
        counters = {
            f"materialization_{scope}_completed": int(
                event.get("completed", 0)
            ),
            f"materialization_{scope}_total": int(
                event.get("total", 0)
            ),
            "materialization_fast_resume": int(
                mode == "fast_resume"
            ),
        }
        resumed_units = event.get("resumed_source_units")
        if resumed_units is not None:
            counters["materialization_resumed_source_units"] = int(
                resumed_units
            )
        reporter.update(
            detail=f"validate upstream: {scope}",
            completed_shards=(
                table_total + int(resumed_units)
                if resumed_units is not None
                else None
            ),
            counters=counters,
        )

    def report_materialization(event: dict[str, Any]) -> None:
        phase = str(event.get("phase") or "materialize")
        completed = int(event.get("completed", 0))
        total = int(event.get("total", 0))
        if phase == "catalog_sources":
            overall = completed
            detail = "catalog source tables"
        elif phase == "query_auto_check_prepare":
            overall = table_total + completed
            detail = "prepare global auto-check plans"
        elif phase == "query_auto_check":
            overall = table_total + completed
            detail = "auto-check selected query evidence"
        elif phase == "materialize_tables":
            overall = table_total * 2 + completed
            detail = "materialize source tables"
        elif phase == "balance_explicit_joins":
            overall = table_total * 3
            detail = "balance explicit joins"
        elif phase == "publish_artifacts":
            overall = table_total * 3 + completed
            detail = "publish dataset artifacts"
        else:
            overall = 0
            detail = phase.replace("_", " ")
        counters = {
            "materialization_phase_completed": completed,
            "materialization_phase_total": total,
        }
        counters.update(
            {
                f"materialization_{key}": int(event[key])
                for key in (
                    "plans",
                    "cached_checks",
                    "migrated_legacy_checks",
                )
                if event.get(key) is not None
            }
        )
        item = event.get("item")
        if item:
            detail = f"{detail}: {item}"
        reporter.update(
            detail=detail,
            completed_shards=overall,
            total_shards=total_work,
            counters=counters,
        )

    result = materialize_dataset(
        MaterializationInputs(
            structural_output_root=config.work_dir / "structural",
            structural_manifests=tuple(item.manifest for item in structural),
            finalized_selection_manifest=finalized.manifest,
            structural_barrier=structural_barrier,
            page_fetch_result=page_result,
            asset_plan_result=planned,
            unique_image_jobs=unique_jobs,
            image_fetch_result=image_result,
            materialized_assets=materialized_assets,
            adapted_model_tasks=adapted,
            model_result=model_result,
            model_authority=authority,
            work_root=config.work_dir,
            sampling_manifest=sampling.manifest_path,
            sampled_entity_paths=sampling.artifact_paths["sampled_entities"],
            sampled_page_ref_paths=sampling.artifact_paths["sampled_page_refs"],
            upstream_stage_registry=_producer_registry_path(
                config,
                "models",
            ),
        ),
        output_root=config.output_dir,
        args=args,
        extractor=extractor,
        records_per_shard=config.records_per_shard,
        pre_write_guard=pre_write_guard,
        validation_progress_callback=report_validation,
        progress_callback=report_materialization,
    )
    counters = {
        str(key): int(value)
        for key, value in result.stats.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }
    _write_stage_registry(
        config,
        "materialize",
        producer_manifests=(result.manifest_path,),
        counters=counters,
        upstream_identity=_registry_identity(config, "models"),
        pre_write_guard=pre_write_guard,
    )
    reporter.update(
        detail="materialization complete",
        completed_shards=total_work,
        total_shards=total_work,
        counters=counters,
    )
    _refresh_known_disk(reporter, config, output=True)
    return result


def _run_pipeline_once(
    config: PipelineConfig,
    *,
    page_transport: Any | None = None,
    image_transport: Any | None = None,
    extractor: Any | None = None,
    after_page_cache_write: Any | None = None,
    _allow_fast_model_resume: bool = True,
) -> PipelineResult:
    """Run the validated staged pipeline without replaying durable outcomes."""
    disk_guard = DiskGuard(config.min_free_disk_bytes)
    archives = _preflight(config, disk_guard)
    source_identity = _input_identity(archives)
    if config.dry_run:
        return PipelineResult(
            status="dry_run",
            stage=None,
            statistics_archives=len(archives),
            counters={},
        )
    if config.from_stage:
        _validate_upstream_for_refresh(config, config.from_stage, archives)
        invalidate_from_stage(
            config,
            config.from_stage,
            pre_write_guard=disk_guard,
        )
    elif config.resume:
        if _fast_materialization_resume_candidate(config):
            try:
                _validate_fast_materialization_resume_chain(
                    config,
                    archives,
                )
                fast_args = _runtime_args(config)
                fast_extractor = extractor
                if fast_extractor is None:
                    fast_extractor = join_builder.LocalAttributeExtractor(
                        fast_args
                    )
                fast_inputs = load_certified_materialization_inputs(
                    config.work_dir,
                    args=fast_args,
                    records_per_shard=config.records_per_shard,
                    extractor=fast_extractor,
                )
            except (OSError, ValueError) as error:
                logging.info(
                    "pipeline materialization fast resume unavailable "
                    "(%s); falling back to strict registry validation",
                    error,
                )
            else:
                return _run_fast_materialization_resume(
                    config,
                    archives=archives,
                    inputs=fast_inputs,
                    args=fast_args,
                    extractor=fast_extractor,
                )
        if (
            _allow_fast_model_resume
            and _fast_model_resume_candidate(config)
        ):
            _validate_fast_model_resume_chain(config, archives)
            return _run_fast_model_resume(
                config,
                archives=archives,
                page_transport=page_transport,
                image_transport=image_transport,
                extractor=extractor,
            )
        _validate_existing_registry_chain(config, archives)
    elif any(_producer_registry_path(config, stage).exists() for stage in STAGES):
        raise ValueError(
            "pipeline state exists; use --resume or --from_stage selection"
        )
    reporter = ProgressReporter(config, pre_write_guard=disk_guard)
    reporter.start()
    try:
        structural, finalized, counters = (
            _run_selection_and_structural(
                config,
                reporter,
                input_identity=source_identity,
                selection_only=config.stop_after == "selection",
                pre_write_guard=disk_guard,
            )
        )
        if config.stop_after == "selection":
            return PipelineResult(
                status="stopped",
                stage="selection",
                statistics_archives=len(archives),
                counters={
                    key: counters[key]
                    for key in ("selected_tables", "reserve_tables")
                },
            )
        if finalized is None:
            raise RuntimeError("structural finalization did not complete")
        if config.stop_after == "structural":
            return PipelineResult(
                status="stopped",
                stage="structural",
                statistics_archives=len(archives),
                counters=counters,
            )
        sampling = _run_sampling(
            config,
            reporter,
            structural,
            pre_write_guard=disk_guard,
        )
        counters.update(
            {
                "eligible_tables": sampling.eligible_tables,
                "prefilter_rejected_tables": sampling.rejected_tables,
                "sampled_entities": sampling.sampled_entities,
                "sampled_page_references": sampling.sampled_page_refs,
                "sampled_direct_image_references": (
                    sampling.sampled_direct_image_refs
                ),
                "page_references": sampling.sampled_page_refs,
                "page_request_upper_bound": sampling.sampled_page_refs,
                "direct_image_references": (
                    sampling.sampled_direct_image_refs
                ),
                "image_request_upper_bound": (
                    sampling.sampled_entities
                    * config.max_image_attempts_per_entity
                    * 2
                ),
                "page_disk_upper_bound_bytes": (
                    sampling.sampled_page_refs * config.web_max_page_bytes
                ),
                "image_disk_upper_bound_bytes": (
                    sampling.sampled_entities
                    * config.max_image_attempts_per_entity
                    * 2
                    * config.web_max_image_bytes
                ),
                **{
                    f"sampled_{stratum}_entities": count
                    for stratum, count in sampling.strata.items()
                },
            }
        )
        counters["network_disk_upper_bound_bytes"] = (
            counters["page_disk_upper_bound_bytes"]
            + counters["image_disk_upper_bound_bytes"]
        )
        counters["estimated_next_stage_bytes"] = (
            sampling.sampled_page_refs * config.estimated_page_result_bytes
            + sampling.sampled_entities
            * config.max_image_attempts_per_entity
            * 2
            * config.estimated_image_result_bytes
        )
        if config.stop_after == "sampling":
            return PipelineResult(
                status="stopped",
                stage="sampling",
                statistics_archives=len(archives),
                counters=counters,
            )
        structural_barrier = _structural_barrier(structural, finalized)
        if page_transport is None:
            page_transport = _new_web_transport(
                config,
                "page",
                pre_write_guard=disk_guard,
            )
        page_result, page_snapshot, page_network_manifest = _run_pages(
            config,
            reporter,
            sampling,
            page_transport,
            after_cache_write=after_page_cache_write,
            pre_write_guard=disk_guard,
        )
        counters.update(
            {
                "unique_page_jobs": page_result.unique,
                "page_success": page_result.success,
                "page_terminal": page_result.terminal,
                **_network_telemetry_counters(
                    "pages",
                    page_result.transport_attempt_summary,
                    reporter.stage_completion_summary("pages"),
                ),
            }
        )
        if config.stop_after == "pages":
            return PipelineResult(
                status="stopped",
                stage="pages",
                statistics_archives=len(archives),
                counters=counters,
            )
        planned, _planning_input = _run_asset_planning(
            config,
            reporter,
            sampling,
            page_result,
            page_snapshot,
            pre_write_guard=disk_guard,
        )
        counters.update(
            {
                "planned_entities": planned.entities,
                "planned_image_references": planned.image_mappings,
            }
        )
        if config.stop_after == "asset_planning":
            return PipelineResult(
                status="stopped",
                stage="asset_planning",
                statistics_archives=len(archives),
                counters=counters,
            )
        if image_transport is None:
            image_transport = _new_web_transport(
                config,
                "image",
                pre_write_guard=disk_guard,
            )
        (
            unique_jobs,
            image_result,
            materialized_assets,
            assets_barrier,
            image_network_manifest,
        ) = _run_images(
            config,
            reporter,
            planned,
            image_transport,
            pre_write_guard=disk_guard,
        )
        counters.update(
            {
                "unique_image_jobs": unique_jobs.records,
                "image_success": image_result.success,
                "image_terminal": image_result.terminal,
                "bridge_assets": materialized_assets.bridge_assets,
                "table_asset_links": materialized_assets.table_asset_links,
                **_network_telemetry_counters(
                    "images",
                    image_result.transport_attempt_summary,
                    reporter.stage_completion_summary("images"),
                ),
            }
        )
        if config.stop_after == "images":
            return PipelineResult(
                status="stopped",
                stage="images",
                statistics_archives=len(archives),
                counters=counters,
            )
        args = _runtime_args(config)
        if extractor is None:
            extractor = join_builder.LocalAttributeExtractor(args)
        adapted, model_result, authority, args = _run_models(
            config,
            reporter,
            structural,
            finalized,
            structural_barrier,
            sampling,
            materialized_assets,
            assets_barrier,
            (page_network_manifest, image_network_manifest),
            extractor,
            pre_write_guard=disk_guard,
        )
        counters.update(
            {
                "model_text_tasks": model_result.text_total,
                "model_image_tasks": model_result.image_total,
                "model_success": model_result.success,
                "model_terminal": model_result.terminal,
            }
        )
        if config.stop_after == "models":
            return PipelineResult(
                status="stopped",
                stage="models",
                statistics_archives=len(archives),
                counters=counters,
            )
        materialized = _run_materialize(
            config,
            reporter,
            structural,
            finalized,
            structural_barrier,
            sampling,
            page_result,
            planned,
            unique_jobs,
            image_result,
            materialized_assets,
            adapted,
            model_result,
            authority,
            args,
            extractor,
            pre_write_guard=disk_guard,
        )
        return PipelineResult(
            status="complete",
            stage="materialize",
            statistics_archives=len(archives),
            counters={
                **counters,
                **{
                    str(key): int(value)
                    for key, value in materialized.stats.items()
                    if isinstance(value, int) and not isinstance(value, bool)
                },
            },
            output_manifest=materialized.manifest_path,
        )
    finally:
        reporter.close()


def _validated_active_selection_path(config: PipelineConfig) -> Path:
    manifest_path = (
        config.work_dir
        / "structural"
        / "stage_manifests"
        / "validated-selection-global.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = payload.get("completed_shards") or []
    if (
        payload.get("stage") != "wdc200k_validated_selection"
        or payload.get("complete") is not True
        or not isinstance(completed, list)
        or len(completed) != 1
    ):
        raise ValueError("validated recovery selection authority is invalid")
    shard = _completed_shard_from_json(completed[0])
    structural_root = config.work_dir / "structural"
    if (
        shard.records != config.max_source_tables
        or not validate_completed_shard(shard, structural_root)
    ):
        raise ValueError("validated recovery selection shard is invalid")
    return structural_root / shard.path


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record is not an object: {path}")
            records.append(value)
    return records


def _recovery_evaluation(
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    active_records = _read_jsonl_records(
        _validated_active_selection_path(config)
    )
    if len(active_records) != config.max_source_tables:
        raise ValueError("recovery evaluation active-table count mismatch")
    decisions_path = config.output_dir / "table_queryability_decisions.jsonl"
    decisions = {
        str(record["source_table_id"]): record
        for record in _read_jsonl_records(decisions_path)
    }
    active_ids = {
        str(record["source_table_id"]) for record in active_records
    }
    if set(decisions) != active_ids:
        raise ValueError("recovery decisions do not match active source tables")
    # Match EntiTables replacement semantics: only multimodal recovery keeps a
    # slot during replacement passes.  Explicit joins are balanced from the
    # terminal round's remaining recovery failures, not used to shield an
    # otherwise unrecoverable candidate from replacement.
    queryable_reasons = {"queryable"}
    failures = [
        record
        for record in active_records
        if str(decisions[str(record["source_table_id"])].get("reason") or "")
        not in queryable_reasons
    ]
    reasons: dict[str, int] = {}
    for record in failures:
        reason = str(
            decisions[str(record["source_table_id"])].get("reason")
            or "unknown"
        )
        reasons[reason] = reasons.get(reason, 0) + 1
    return active_records, failures, reasons


def _sampling_prefilter_decisions(
    config: PipelineConfig,
) -> dict[str, dict[str, Any]]:
    authority = validate_sampling_artifacts(
        config.work_dir / "sampling" / "manifest.json"
    )
    decisions: dict[str, dict[str, Any]] = {}
    for path in authority.artifact_paths["prefilter_tables"]:
        for record in _read_jsonl_records(path):
            table_id = str(record["source_table_id"])
            if table_id in decisions:
                raise ValueError("duplicate sampling prefilter decision")
            decisions[table_id] = record
    return decisions


def _plan_sampling_expansion(
    config: PipelineConfig,
    *,
    state: SamplingExpansionState,
    failures: Sequence[dict[str, Any]],
) -> tuple[dict[str, int], list[str], dict[str, int]]:
    """Increase only rejected tables that have untried evidence rows."""
    limits = dict(state.limits)
    failure_ids = {
        str(record["source_table_id"]) for record in failures
    }
    decisions = _sampling_prefilter_decisions(config)
    expanded: list[str] = []
    eligible_failures = 0
    exhausted_failures = 0
    for table_id in sorted(failure_ids):
        decision = decisions.get(table_id)
        if not decision or decision.get("eligible") is not True:
            exhausted_failures += 1
            continue
        eligible_failures += 1
        sampled = int(decision.get("sampled_entities", 0))
        attemptable = int(decision.get("attemptable_entities", sampled))
        current_cap = int(
            decision.get(
                "sample_limit",
                limits.get(table_id, config.sampled_entities_per_table),
            )
        )
        if attemptable <= sampled:
            exhausted_failures += 1
            continue
        next_limit: int | None = None
        for scheduled_limit in config.sampled_entity_expansion_schedule:
            if scheduled_limit <= current_cap:
                continue
            candidate_limit = min(scheduled_limit, attemptable)
            if candidate_limit > sampled:
                next_limit = candidate_limit
                break
        if next_limit is None:
            exhausted_failures += 1
            continue
        limits[table_id] = next_limit
        expanded.append(table_id)
    return limits, expanded, {
        "failed_tables": len(failure_ids),
        "eligible_failed_tables": eligible_failures,
        "expanded_tables": len(expanded),
        "sampling_exhausted_failed_tables": exhausted_failures,
    }


def _sampling_registry_matches_expansion(
    config: PipelineConfig,
) -> bool:
    path = _producer_registry_path(config, "sampling")
    if not path.is_file():
        return False
    try:
        registry = _load_stage_registry(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return registry.config_fingerprint == _stage_config_fingerprint(
        config, "sampling"
    )


def _drop_recovery_failure(
    config: PipelineConfig,
    *,
    round_index: int,
    relative_path: str,
) -> bool:
    probability = config.unrecoverable_drop_probability
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    rank = int(
        stable_hash(
            "wdc200k-recovery-drop-v1",
            config.selection_seed,
            round_index,
            relative_path,
            length=16,
        ),
        16,
    )
    return rank / float(1 << 64) < probability


def _replacement_selection_record(
    candidate: TableCandidate,
    *,
    config: PipelineConfig,
    invalid: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    return {
        "schema_class": candidate.schema_class,
        "subset": candidate.subset,
        "host": candidate.host,
        "relative_path": candidate.relative_path,
        "rows": candidate.rows,
        "columns": candidate.columns,
        "rank": stable_hash(config.selection_seed, candidate.relative_path),
        "selection_seed": config.selection_seed,
        "recovery_replaces_path": str(invalid["relative_path"]),
        "recovery_replaces_source_table_id": str(
            invalid["source_table_id"]
        ),
        "recovery_replacement_round": round_index,
    }


def _write_active_recovery_selection(
    config: PipelineConfig,
    *,
    round_index: int,
    records: Sequence[dict[str, Any]],
    previous_selection_path: Path,
    pre_write_guard: PreWriteGuard | None = None,
) -> ActiveRecoverySelection:
    if len(records) != config.max_source_tables:
        raise ValueError("active recovery selection must fill every slot")
    round_root = _recovery_root(config) / f"round-{round_index:05d}"
    shard_path = round_root / "active_selection.jsonl"
    manifest_path = round_root / "manifest.json"
    expected_records_fingerprint = stable_hash(
        "wdc200k-recovery-active-records-v1",
        *(json.dumps(record, sort_keys=True) for record in records),
        length=40,
    )
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = manifest.get("completed_shards") or []
        if (
            manifest.get("records_fingerprint")
            != expected_records_fingerprint
            or len(completed) != 1
        ):
            raise ValueError("conflicting persisted recovery selection round")
        shard = _completed_shard_from_json(completed[0])
        if not validate_completed_shard(shard, round_root):
            raise ValueError("persisted recovery selection shard is invalid")
    else:
        writer = AtomicJsonlShard(
            shard_path,
            pre_write_guard=pre_write_guard,
        )
        try:
            for record in records:
                writer.write(record)
            shard = writer.commit()
        except BaseException:
            writer.abort()
            raise
        _atomic_json(
            manifest_path,
            {
                "stage": "wdc200k_selection",
                "schema_version": "wdc200k-recovery-selection-v1",
                "input_fingerprint": _sha256_path(previous_selection_path),
                "parameter_fingerprint": stable_hash(
                    _recovery_policy_identity(config),
                    round_index,
                    length=40,
                ),
                "policy_identity": _recovery_policy_identity(config),
                "round_index": round_index,
                "records_fingerprint": expected_records_fingerprint,
                "completed_shards": [asdict(shard)],
                "complete": True,
            },
            pre_write_guard=pre_write_guard,
        )
    _atomic_json(
        _recovery_root(config) / "current.json",
        {
            "schema_version": "wdc200k-recovery-selection-pointer-v1",
            "policy_identity": _recovery_policy_identity(config),
            "round_index": round_index,
            "manifest_path": str(manifest_path.resolve()),
        },
        pre_write_guard=pre_write_guard,
    )
    return ActiveRecoverySelection(
        round_index=round_index,
        path=shard_path,
        manifest_path=manifest_path,
        records=len(records),
    )


def _claim_recovery_replacements(
    config: PipelineConfig,
    *,
    round_index: int,
    active_records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    pre_write_guard: PreWriteGuard | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    policy = SelectionPolicy(
        target_tables=config.max_source_tables,
        seed=config.selection_seed,
        top100_policy=config.top100_policy,
        top100_per_class=config.top100_per_class,
        include_rest=config.include_rest,
        min_candidate_rows=config.min_candidate_rows,
        min_candidate_columns=config.min_candidate_columns,
        minimum3_fraction=config.minimum3_fraction,
        rest_base_per_class=0,
        class_cap=config.class_max_tables,
    )
    manager = ReserveManager.open(
        config.work_dir / "selection" / "reserve.sqlite3",
        policy,
        pre_write_guard=pre_write_guard,
    )
    failures_by_path = {
        str(record["relative_path"]): record for record in failures
    }
    output: list[dict[str, Any]] = []
    replaced = 0
    retained_by_probability = 0
    reserve_exhausted = 0
    for record in active_records:
        path = str(record["relative_path"])
        invalid = failures_by_path.get(path)
        if invalid is None:
            output.append(dict(record))
            continue
        if not _drop_recovery_failure(
            config,
            round_index=round_index,
            relative_path=path,
        ):
            retained_by_probability += 1
            output.append(dict(record))
            continue
        operation_key = stable_hash(
            "wdc200k-recovery-replacement-v1",
            round_index,
            path,
            length=40,
        )
        persisted_candidate = manager.active_candidate(path)
        if persisted_candidate is None:
            raise ValueError(
                "recovery failure is not an active selection: " + path
            )
        try:
            claim = manager.claim_replacement(
                operation_key=operation_key,
                invalid_candidate=persisted_candidate,
                reason=f"unrecoverable_after_auto_check_round_{round_index - 1}",
                retain_on_exhaustion=True,
            )
        except ReserveExhaustedError:
            reserve_exhausted += 1
            output.append(dict(record))
            continue
        if claim.replacement is None:
            raise RuntimeError("recovery replacement claim has no candidate")
        output.append(
            {
                **_replacement_selection_record(
                claim.replacement,
                config=config,
                invalid=invalid,
                round_index=round_index,
                ),
                # Structural publication is the durable acknowledgment point
                # for both structural and post-recovery reserve claims.
                "replacement_operation_key": claim.operation_key,
            }
        )
        replaced += 1
    return output, {
        "failed": len(failures),
        "replaced": replaced,
        "retained_by_probability": retained_by_probability,
        "reserve_exhausted": reserve_exhausted,
    }


def _selection_registry_references_active(
    config: PipelineConfig,
    active: ActiveRecoverySelection,
) -> bool:
    path = _producer_registry_path(config, "selection")
    if not path.is_file():
        return False
    try:
        registry = _load_stage_registry(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    expected = active.manifest_path.resolve()
    return any(reference.path.resolve() == expected for reference in registry.producer_manifests)


def _write_recovery_round_evaluation(
    config: PipelineConfig,
    *,
    round_index: int,
    failures: Sequence[dict[str, Any]],
    reasons: dict[str, int],
    replacement: dict[str, int] | None,
    evaluated: int | None = None,
    pre_write_guard: PreWriteGuard | None = None,
) -> None:
    _atomic_json(
        _recovery_root(config) / f"evaluation-{round_index:05d}.json",
        {
            "schema_version": "wdc200k-recovery-evaluation-v1",
            "policy_identity": _recovery_policy_identity(config),
            "round_index": round_index,
            "evaluated": (
                config.max_source_tables if evaluated is None else evaluated
            ),
            "unrecoverable": len(failures),
            "failure_reasons": dict(sorted(reasons.items())),
            "replacement": replacement,
            "failed_source_table_ids": sorted(
                str(record["source_table_id"]) for record in failures
            ),
            "complete": True,
        },
        pre_write_guard=pre_write_guard,
    )


def run_pipeline(
    config: PipelineConfig,
    *,
    page_transport: Any | None = None,
    image_transport: Any | None = None,
    extractor: Any | None = None,
    after_page_cache_write: Any | None = None,
    _allow_fast_model_resume: bool = True,
) -> PipelineResult:
    """Expand rejected evidence samples, then replace exhausted failures."""
    if (
        not config.dry_run
        and config.stop_after in {None, "models", "materialize"}
    ):
        _promote_legacy_model_cache(
            config,
            pre_write_guard=DiskGuard(config.min_free_disk_bytes),
        )
    if (
        config.dry_run
        or config.stop_after is not None
        or (
            not config.sampled_entity_expansion_schedule
            and config.unrecoverable_replacement_rounds == 0
        )
    ):
        return _run_pipeline_once(
            config,
            page_transport=page_transport,
            image_transport=image_transport,
            extractor=extractor,
            after_page_cache_write=after_page_cache_write,
            _allow_fast_model_resume=_allow_fast_model_resume,
        )

    active = _load_active_recovery_selection(config)
    expansion = _load_sampling_expansion_state(config)
    round_index = 0 if active is None else active.round_index
    effective = replace(
        config,
        recovery_replacement_round_index=round_index,
        sampling_expansion_round_index=expansion.round_index,
    )
    if (
        active is not None
        and effective.from_stage is None
        and not _selection_registry_references_active(effective, active)
    ):
        # The next active selection was committed after the preceding dataset.
        # Archive that preceding structural/output generation before resuming.
        effective = replace(effective, from_stage="structural")
    elif (
        expansion.round_index > 0
        and effective.from_stage is None
        and not _sampling_registry_matches_expansion(effective)
    ):
        # The next per-table cap state was committed after the preceding
        # dataset. URL and model caches remain outside the archived stages.
        effective = replace(effective, from_stage="sampling")

    disk_guard = DiskGuard(config.min_free_disk_bytes)
    while True:
        result = _run_pipeline_once(
            effective,
            page_transport=page_transport,
            image_transport=image_transport,
            extractor=extractor,
            after_page_cache_write=after_page_cache_write,
            _allow_fast_model_resume=_allow_fast_model_resume,
        )
        if result.status != "complete" or result.stage != "materialize":
            return result

        active_records, failures, reasons = _recovery_evaluation(effective)
        if failures and config.sampled_entity_expansion_schedule:
            next_limits, expanded_ids, _expansion_stats = (
                _plan_sampling_expansion(
                    effective,
                    state=expansion,
                    failures=failures,
                )
            )
            if expanded_ids:
                expansion = _write_sampling_expansion_state(
                    effective,
                    previous=expansion,
                    limits=next_limits,
                    expanded_table_ids=expanded_ids,
                    pre_write_guard=disk_guard,
                )
                effective = replace(
                    config,
                    recovery_replacement_round_index=round_index,
                    sampling_expansion_round_index=expansion.round_index,
                    resume=True,
                    from_stage="sampling",
                )
                continue
        if round_index >= config.unrecoverable_replacement_rounds or not failures:
            _write_recovery_round_evaluation(
                effective,
                round_index=round_index,
                failures=failures,
                reasons=reasons,
                replacement=None,
                pre_write_guard=disk_guard,
            )
            return replace(
                result,
                counters={
                    **result.counters,
                    "recovery_replacement_round": round_index,
                    "recovery_unrecoverable_tables": len(failures),
                    "sampling_expansion_round": expansion.round_index,
                    "progressively_expanded_tables": len(expansion.limits),
                },
            )

        evaluated = config.max_source_tables
        while True:
            next_round = round_index + 1
            next_records, replacement_stats = _claim_recovery_replacements(
                effective,
                round_index=next_round,
                active_records=active_records,
                failures=failures,
                pre_write_guard=disk_guard,
            )
            _write_recovery_round_evaluation(
                effective,
                round_index=round_index,
                failures=failures,
                reasons=reasons,
                replacement=replacement_stats,
                evaluated=evaluated,
                pre_write_guard=disk_guard,
            )
            if replacement_stats["replaced"]:
                break
            # Match EntiTables: a pass where the probability draw retains all
            # failures still consumes one round, but does not reevaluate the
            # unchanged slots before the next seeded draw.
            round_index = next_round
            evaluated = 0
            if (
                round_index >= config.unrecoverable_replacement_rounds
                or replacement_stats["reserve_exhausted"] == len(failures)
                or config.unrecoverable_drop_probability == 0.0
            ):
                return replace(
                    result,
                    counters={
                        **result.counters,
                        "recovery_replacement_round": round_index,
                        "recovery_unrecoverable_tables": len(failures),
                        "sampling_expansion_round": expansion.round_index,
                        "progressively_expanded_tables": len(
                            expansion.limits
                        ),
                    },
                )

        active = _write_active_recovery_selection(
            effective,
            round_index=next_round,
            records=next_records,
            previous_selection_path=_validated_active_selection_path(effective),
            pre_write_guard=disk_guard,
        )
        round_index = next_round
        effective = replace(
            config,
            recovery_replacement_round_index=round_index,
            sampling_expansion_round_index=expansion.round_index,
            resume=True,
            from_stage="structural",
        )


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    configure_parser: (
        Callable[[argparse.ArgumentParser], None] | None
    ) = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the staged 200K-table multimodal joinability dataset "
            "from WDC Schema.org 2023."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--work_dir", default="")
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--max_source_tables", type=int, default=200_000)
    parser.add_argument(
        "--max_rows_per_source_table",
        type=int,
        default=None,
        help="Compatibility option; only an omitted value is accepted.",
    )
    parser.add_argument("--selection_seed", type=int, default=13)
    parser.add_argument(
        "--top100_policy",
        choices=("all", "bounded", "exclude"),
        default="bounded",
    )
    parser.add_argument("--top100_per_class", type=int, default=10)
    parser.add_argument(
        "--include_rest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--min_candidate_rows", type=int, default=5)
    parser.add_argument("--min_candidate_columns", type=int, default=3)
    parser.add_argument("--minimum3_fraction", type=float, default=1.0)
    parser.add_argument("--class_max_tables", type=int, default=40_000)
    parser.add_argument("--page_attempts", type=int, default=1)
    parser.add_argument("--image_attempts", type=int, default=1)
    parser.add_argument("--web_max_retries", type=int, default=0)
    parser.add_argument("--web_max_response_seconds", type=float, default=8.0)
    parser.add_argument("--web_global_concurrency", type=int, default=128)
    parser.add_argument("--web_per_host_concurrency", type=int, default=2)
    parser.add_argument(
        "--max_image_attempts_per_entity", type=int, default=3
    )
    parser.add_argument("--max_images_per_entity", type=int, default=3)
    parser.add_argument("--sampled_entities_per_table", type=int, default=8)
    parser.add_argument(
        "--sampled_entity_expansion_schedule",
        type=int,
        nargs="*",
        default=(12, 20, 25),
        metavar="N",
        help=(
            "Per-table sample caps tried after a model/auto-check rejection; "
            "an empty list disables progressive expansion."
        ),
    )
    parser.add_argument("--entity_sampling_seed", type=int, default=20260720)
    parser.add_argument("--global_entity_budget", type=int)
    parser.add_argument("--min_free_disk_bytes", type=int, default=1_000_000_000)
    parser.add_argument("--selection_shard_tables", type=int, default=100)
    parser.add_argument("--records_per_shard", type=int, default=10_000)
    parser.add_argument("--web_max_page_bytes", type=int, default=2_000_000)
    parser.add_argument("--web_max_image_bytes", type=int, default=10_000_000)
    parser.add_argument("--estimated_page_result_bytes", type=int, default=32_768)
    parser.add_argument("--estimated_image_result_bytes", type=int, default=500_000)
    parser.add_argument("--progress_interval_seconds", type=float, default=5.0)
    parser.add_argument("--text_asset_chunk_chars", type=int, default=800)
    parser.add_argument("--min_text_asset_chunk_chars", type=int, default=120)
    parser.add_argument("--max_text_asset_chunks_per_entity", type=int, default=3)
    parser.add_argument(
        "--split_by", choices=("source_table_id", "page_title"), default="page_title"
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--min_column_non_empty_ratio", type=float, default=0.5)
    parser.add_argument("--min_recovered_value_ratio", type=float, default=0.6)
    parser.add_argument("--min_recovery_denominator", type=int, default=2)
    parser.add_argument("--min_rows_per_output_table", type=int, default=2)
    parser.add_argument("--query_rows_per_table", type=int, default=5)
    parser.add_argument(
        "--max_train_query_row_views_per_join",
        type=int,
        default=5,
        help=(
            "Maximum deterministic disjoint query row views per train join chain; "
            "0 means use every feasible view. Dev/test always use one."
        ),
    )
    parser.add_argument("--max_query_tables_per_source_table", type=int, default=0)
    parser.add_argument("--max_query_context_attrs", type=int, default=1)
    parser.add_argument("--max_target_context_attrs", type=int, default=2)
    parser.add_argument(
        "--explicit_join_fallback_mode",
        choices=join_builder.EXPLICIT_JOIN_FALLBACK_MODES,
        default=join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_MODE,
        help=(
            "match_implicit selects exactly one visible-join query per "
            "implicit query within each split; ratio retains legacy sampling."
        ),
    )
    parser.add_argument(
        "--explicit_join_fallback_ratio",
        type=float,
        default=join_builder.DEFAULT_EXPLICIT_JOIN_FALLBACK_RATIO,
        help=(
            "Seeded fraction of multimodal rejections materialized as ordinary "
            "joins with a visible non-entity join column in query and target."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--from_stage", choices=STAGES)
    parser.add_argument("--stop_after", choices=STAGES)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--refresh_page_cache", action="store_true")
    parser.add_argument("--refresh_image_cache", action="store_true")

    # Dynamic-vLLM runner compatibility.
    parser.add_argument("--run_fingerprint", default="")
    parser.add_argument("--runtime_dir", default="")
    parser.add_argument("--model_start_marker")
    parser.add_argument("--model_ready_marker")
    parser.add_argument("--model_ready_timeout_seconds", type=float)
    parser.add_argument(
        "--model_endpoint_ready_timeout_seconds", type=float, default=30.0
    )
    parser.add_argument("--model_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--model_max_retries", type=int, default=2)
    parser.add_argument(
        "--model_retry_sleep_seconds", type=float, default=2.0
    )
    parser.add_argument(
        "--model_cache_database_path",
        help=(
            "Persistent model/auto-check SQLite cache. Keeping this outside "
            "the work directory lets recovery-replacement rounds reuse all "
            "previous model judgments."
        ),
    )
    parser.add_argument(
        "--unrecoverable_replacement_rounds",
        type=int,
        default=0,
        help=(
            "Post-recovery replacement passes; zero preserves the legacy "
            "single-pass WDC behavior."
        ),
    )
    parser.add_argument(
        "--unrecoverable_drop_probability",
        type=float,
        default=0.5,
        help="Seeded fraction of current recovery failures replaced per pass.",
    )
    parser.add_argument("--model_text_done_marker")
    parser.add_argument("--model_image_done_marker")
    parser.add_argument("--model_endpoint_config")
    parser.add_argument("--text_model_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--text_model_base_urls", nargs="*", default=None)
    parser.add_argument("--text_model_base_urls_file")
    parser.add_argument("--remote_text_model_base_url")
    parser.add_argument("--remote_text_model_base_urls", nargs="*", default=None)
    parser.add_argument("--remote_text_model_base_urls_file")
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_model_api_key")
    parser.add_argument("--remote_text_model_api_key")
    parser.add_argument("--image_model_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--image_model_base_urls", nargs="*", default=None)
    parser.add_argument("--image_model_base_urls_file")
    parser.add_argument("--remote_image_model_base_url")
    parser.add_argument("--remote_image_model_base_urls", nargs="*", default=None)
    parser.add_argument("--remote_image_model_base_urls_file")
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--image_model_api_key")
    parser.add_argument("--remote_image_model_api_key")
    parser.add_argument("--remote_text_model_workers", type=int, default=0)
    parser.add_argument("--remote_image_model_workers", type=int, default=0)
    parser.add_argument(
        "--remote_layout_control_url",
        default=None,
        help="Loopback URL of the existing SSH-forwarded layout control API.",
    )
    parser.add_argument("--remote_layout_control_token_file")
    parser.add_argument("--remote_layout_routing_manifest")
    parser.add_argument("--remote_layout_controller_id_file")
    parser.add_argument("--remote_layout_lock_file")
    parser.add_argument(
        "--remote_layout_primary_image_url",
        help="Local inference tunnel URL mapped from endpoint ID primary_image.",
    )
    parser.add_argument(
        "--remote_layout_switchable_url",
        help="Local inference tunnel URL mapped from endpoint ID switchable.",
    )
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
        "--remote_layout_reconnect_timeout_seconds", type=float, default=120.0
    )
    parser.add_argument(
        "--remote_layout_operation_timeout_seconds", type=float, default=1200.0
    )
    parser.add_argument(
        "--remote_layout_drain_timeout_seconds", type=float, default=300.0
    )
    parser.add_argument(
        "--remote_layout_poll_seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--remote_layout_workload_poll_seconds", type=float, default=2.0
    )
    parser.add_argument(
        "--remote_layout_stability_seconds", type=float, default=10.0
    )
    parser.add_argument(
        "--remote_layout_health_timeout_seconds", type=float, default=3.0
    )
    parser.add_argument(
        "--remote_layout_health_stable_polls", type=int, default=2
    )
    parser.add_argument(
        "--remote_layout_coordination_dir",
        help=(
            "Shared priority directory used to reclaim the remote GPUs from "
            "an opportunistic EntiTables borrower."
        ),
    )
    parser.add_argument(
        "--remote_layout_reclaim_timeout_seconds", type=float, default=90.0
    )
    parser.add_argument(
        "--remote_layout_borrower_stale_seconds", type=float, default=10.0
    )
    parser.add_argument(
        "--remote_layout_unregistered_grace_seconds", type=float, default=1.0
    )
    parser.add_argument(
        "--remote_layout_coordination_poll_seconds", type=float, default=0.2
    )
    parser.add_argument("--precompute_model_cache", action="store_true")
    parser.add_argument("--precompute_text_model_cache", action="store_true")
    parser.add_argument("--text_model_workers", type=int, default=1)
    parser.add_argument("--image_model_workers", type=int, default=1)
    parser.add_argument("--materialization_workers", type=int, default=1)
    parser.add_argument(
        "--materialization_validation_workers",
        type=int,
        default=3,
        help=(
            "Concurrent strict materialization validators (1-4); "
            "each SQLite writer uses an independent validation database."
        ),
    )
    join_builder.add_model_auto_check_arguments(parser)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args(argv)
    if args.max_rows_per_source_table is not None:
        parser.error("source-table rows are unbounded; omit --max_rows_per_source_table")
    if args.page_attempts != 1 or args.image_attempts != 1:
        parser.error("page_attempts and image_attempts must both be exactly 1")
    if args.selection_shard_tables <= 0 or args.records_per_shard <= 0:
        parser.error("shard sizes must be positive")
    if args.progress_interval_seconds <= 0:
        parser.error("--progress_interval_seconds must be positive")
    if min(
        args.text_asset_chunk_chars,
        args.min_text_asset_chunk_chars,
        args.max_text_asset_chunks_per_entity,
    ) <= 0:
        parser.error("text asset chunk limits must be positive")
    if min(args.text_model_workers, args.image_model_workers) <= 0:
        parser.error("model worker counts must be positive")
    if min(
        args.remote_text_model_workers,
        args.remote_image_model_workers,
    ) < 0:
        parser.error("remote model worker counts must be non-negative")
    remote_layout_required = (
        args.remote_layout_control_token_file,
        args.remote_layout_primary_image_url,
        args.remote_layout_switchable_url,
    )
    if args.remote_layout_control_url and not all(remote_layout_required):
        parser.error(
            "--remote_layout_control_url requires token file and both endpoint tunnel URLs"
        )
    if not args.remote_layout_control_url and any(remote_layout_required):
        parser.error(
            "remote layout token and endpoint mappings require --remote_layout_control_url"
        )
    if args.remote_layout_coordination_dir and not args.remote_layout_control_url:
        parser.error(
            "--remote_layout_coordination_dir requires remote layout control"
        )
    if args.remote_layout_health_stable_polls <= 0:
        parser.error("remote layout stable polls must be positive")
    if not 5 <= args.remote_layout_lease_ttl_seconds <= 60:
        parser.error("--remote_layout_lease_ttl_seconds must be between 5 and 60")
    if not (
        0
        < args.remote_layout_lease_renew_seconds
        < args.remote_layout_lease_ttl_seconds
    ):
        parser.error("remote layout lease renewal must be positive and below TTL")
    remote_positive_timeouts = (
        args.remote_layout_request_timeout_seconds,
        args.remote_layout_reconnect_timeout_seconds,
        args.remote_layout_operation_timeout_seconds,
        args.remote_layout_drain_timeout_seconds,
        args.remote_layout_poll_seconds,
        args.remote_layout_workload_poll_seconds,
        args.remote_layout_health_timeout_seconds,
        args.remote_layout_reclaim_timeout_seconds,
        args.remote_layout_borrower_stale_seconds,
        args.remote_layout_coordination_poll_seconds,
    )
    if any(
        not math.isfinite(value) or value <= 0
        for value in remote_positive_timeouts
    ):
        parser.error("remote layout timeouts must be finite and positive")
    if (
        not math.isfinite(args.remote_layout_unregistered_grace_seconds)
        or args.remote_layout_unregistered_grace_seconds < 0
    ):
        parser.error("remote layout unregistered grace must be non-negative")
    if (
        not math.isfinite(args.remote_layout_stability_seconds)
        or args.remote_layout_stability_seconds < 0
    ):
        parser.error("--remote_layout_stability_seconds must be non-negative")
    if args.materialization_workers <= 0:
        parser.error("--materialization_workers must be positive")
    if not 1 <= args.materialization_validation_workers <= 4:
        parser.error(
            "--materialization_validation_workers must be between 1 and 4"
        )
    if (
        not math.isfinite(args.model_endpoint_ready_timeout_seconds)
        or args.model_endpoint_ready_timeout_seconds < 0
    ):
        parser.error("--model_endpoint_ready_timeout_seconds must be non-negative")
    if (
        not math.isfinite(args.model_timeout_seconds)
        or args.model_timeout_seconds <= 0
    ):
        parser.error("--model_timeout_seconds must be positive")
    if args.model_max_retries < 0:
        parser.error("--model_max_retries must be non-negative")
    if args.unrecoverable_replacement_rounds < 0:
        parser.error("--unrecoverable_replacement_rounds must be non-negative")
    if not (
        math.isfinite(args.unrecoverable_drop_probability)
        and 0.0 <= args.unrecoverable_drop_probability <= 1.0
    ):
        parser.error(
            "--unrecoverable_drop_probability must be between zero and one"
        )
    if (
        not math.isfinite(args.model_retry_sleep_seconds)
        or args.model_retry_sleep_seconds < 0
    ):
        parser.error("--model_retry_sleep_seconds must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run_pipeline(PipelineConfig.from_args(parse_args(argv)))
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "stage": result.stage,
                "statistics_archives": result.statistics_archives,
                "counters": result.counters,
                "output_manifest": (
                    None
                    if result.output_manifest is None
                    else str(result.output_manifest)
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run the staged WDC Schema.org 2023 200K dataset pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import build_mm_joinability_dataset as join_builder
import build_wdc_mm_joinability_dataset as legacy_wdc_builder
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
    structural_asset_input_identity,
    validate_asset_plan_shards,
    validate_complete_image_fetch,
    validate_materialized_asset_shards,
    validate_unique_image_jobs,
)
from wdc200k_fetch import (
    FetchPolicy,
    FetchResult,
    fetch_unique_pages,
    iter_finalized_page_refs,
    iter_page_fanout,
    iter_page_outcomes,
    validate_complete_page_fetch,
)
from wdc200k_io import (
    AtomicJsonlShard,
    CompletedShard,
    SqliteJobStore,
    validate_completed_shard,
)
from wdc200k_materialize import (
    MaterializationInputs,
    MaterializationResult,
    materialize_dataset,
)
from wdc200k_models import (
    AdaptedModelTasks,
    AssetStageBarrier,
    ModelStageAuthority,
    ModelStageResult,
    StructuralStageBarrier,
    adapt_model_tasks_from_manifests,
    enqueue_model_tasks,
    run_model_stage,
    validate_model_stage_for_adapter,
)
from wdc200k_selection import (
    ReserveManager,
    SelectionPolicy,
    run_selection,
)
from wdc200k_structural import (
    FinalizedSelectionResult,
    StructuralExpansionResult,
    expand_selected_shard,
    finalize_validated_selection,
)


STAGES = (
    "selection",
    "structural",
    "pages",
    "asset_planning",
    "images",
    "models",
    "materialize",
)


class DiskSpaceInsufficientError(RuntimeError):
    """Raised before a stage could violate the configured disk reserve."""


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    work_dir: Path
    cache_dir: Path
    max_source_tables: int = 200_000
    max_rows_per_source_table: None = None
    selection_seed: int = 13
    minimum3_fraction: float = 0.90
    class_max_tables: int = 40_000
    web_max_retries: int = 0
    web_max_response_seconds: float = 8.0
    web_global_concurrency: int = 128
    web_per_host_concurrency: int = 2
    max_image_attempts_per_entity: int = 3
    max_images_per_entity: int = 3
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
    max_query_tables_per_source_table: int = 0
    max_query_context_attrs: int = 1
    max_target_context_attrs: int = 2
    text_model_name: str = "Qwen3.5-9B"
    image_model_name: str = "Qwen3-VL-8B-Thinking"
    text_model_base_url: str = "http://localhost:8001/v1"
    text_model_base_urls: tuple[str, ...] = ()
    text_model_base_urls_file: str | None = None
    text_model_api_key: str | None = None
    image_model_base_url: str = "http://localhost:8000/v1"
    image_model_base_urls: tuple[str, ...] = ()
    image_model_base_urls_file: str | None = None
    image_model_api_key: str | None = None
    text_model_workers: int = 1
    image_model_workers: int = 1
    run_fingerprint: str = ""
    runtime_dir: Path | None = None
    model_start_marker: Path | None = None
    model_ready_marker: Path | None = None
    model_ready_timeout_seconds: float | None = None
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
        work_dir = (
            Path(args.work_dir).resolve()
            if args.work_dir
            else output_dir.parent / "work_wdc_200k"
        )
        cache_dir = (
            Path(args.cache_dir).resolve()
            if args.cache_dir
            else output_dir.parent / "cache" / "wdc_200k"
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
            max_query_tables_per_source_table=(
                args.max_query_tables_per_source_table
            ),
            max_query_context_attrs=args.max_query_context_attrs,
            max_target_context_attrs=args.max_target_context_attrs,
            text_model_name=args.text_model_name,
            image_model_name=args.image_model_name,
            text_model_base_url=args.text_model_base_url,
            text_model_base_urls=tuple(args.text_model_base_urls or ()),
            text_model_base_urls_file=args.text_model_base_urls_file,
            text_model_api_key=args.text_model_api_key,
            image_model_base_url=args.image_model_base_url,
            image_model_base_urls=tuple(args.image_model_base_urls or ()),
            image_model_base_urls_file=args.image_model_base_urls_file,
            image_model_api_key=args.image_model_api_key,
            text_model_workers=args.text_model_workers,
            image_model_workers=args.image_model_workers,
            run_fingerprint=args.run_fingerprint,
            runtime_dir=(Path(args.runtime_dir).resolve() if args.runtime_dir else None),
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


@dataclass
class _ProgressState:
    stage: str = "preflight"
    completed_shards: int = 0
    total_shards: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    stage_started_at: float = field(default_factory=time.time)
    known_work_bytes: int = 0
    known_cache_bytes: int = 0
    known_output_bytes: int = 0


class ProgressReporter:
    """Publish bounded-cost progress to JSON and direct stdout."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.path = config.work_dir / "progress.json"
        self._state = _ProgressState()
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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
        completed_shards: int | None = None,
        total_shards: int | None = None,
        counters: dict[str, int] | None = None,
        known_work_bytes: int | None = None,
        known_cache_bytes: int | None = None,
        known_output_bytes: int | None = None,
    ) -> None:
        with self._lock:
            if stage is not None and stage != self._state.stage:
                self._state.stage = stage
                self._state.stage_started_at = time.time()
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

    def _snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed = max(0.0, now - self._state.stage_started_at)
            complete = self._state.completed_shards
            total = self._state.total_shards
            rate = complete / elapsed if elapsed > 0 else 0.0
            remaining = max(0, total - complete)
            eta = remaining / rate if rate > 0 else None
            disk_probe = self.config.work_dir
            free = shutil.disk_usage(disk_probe).free
            return {
                "stage": self._state.stage,
                "completed_shards": complete,
                "total_shards": total,
                "counters": dict(sorted(self._state.counters.items())),
                "rates": {
                    "shards_per_second": rate,
                    "rolling_shards_per_second": rate,
                },
                "eta_seconds": eta,
                "elapsed_seconds": max(0.0, now - self._state.started_at),
                "updated_at": now,
                "disk": {
                    "work_bytes": self._state.known_work_bytes,
                    "cache_bytes": self._state.known_cache_bytes,
                    "output_bytes": self._state.known_output_bytes,
                    "free_bytes": free,
                    "reserve_bytes": self.config.min_free_disk_bytes,
                },
            }

    def publish(self) -> None:
        with self._publish_lock:
            snapshot = self._snapshot()
            _atomic_json(self.path, snapshot)
            print(
                "[wdc200k] "
                f"stage={snapshot['stage']} "
                f"shards={snapshot['completed_shards']}/"
                f"{snapshot['total_shards']} "
                f"rate={snapshot['rates']['shards_per_second']:.3f}/s "
                f"eta={snapshot['eta_seconds']} "
                f"free={snapshot['disk']['free_bytes']} "
                f"counters={json.dumps(snapshot['counters'], sort_keys=True)}",
                flush=True,
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.config.progress_interval_seconds))
        self.publish()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
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
        "minimum3_fraction",
        "class_max_tables",
    ),
    "structural": ("selection_shard_tables",),
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
        "max_query_tables_per_source_table",
        "max_query_context_attrs",
        "max_target_context_attrs",
        "records_per_shard",
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


def _validate_producer_manifest(stage: str, path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"{stage} producer manifest is incomplete: {path}")
    producer_stage = str(payload.get("stage") or "")
    expected = {
        "selection": {"wdc200k_selection"},
        "structural": {"wdc200k_structural", "wdc200k_validated_selection"},
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
            if not validate_completed_shard(completed, root):
                raise ValueError(
                    f"{stage} producer shard checksum mismatch: "
                    f"{root / completed.path}"
                )


def _validate_stage_registry(
    config: PipelineConfig,
    stage: str,
    *,
    expected_upstream_identity: str,
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
    for reference in registry.producer_manifests:
        if not reference.path.is_file():
            raise ValueError(f"producer manifest is missing: {reference.path}")
        if _sha256_path(reference.path) != reference.sha256:
            raise ValueError(
                f"producer manifest checksum mismatch: {reference.path}"
            )
        _validate_producer_manifest(stage, reference.path)
    return registry


def _write_stage_registry(
    config: PipelineConfig,
    stage: str,
    *,
    producer_manifests: Iterable[Path],
    counters: dict[str, int],
    upstream_identity: str,
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
    )
    return _validate_stage_registry(
        config,
        stage,
        expected_upstream_identity=upstream_identity,
    )


_STAGE_WORK_PATHS: dict[str, tuple[str, ...]] = {
    "selection": ("selection",),
    "structural": ("structural",),
    "pages": ("page_jobs",),
    "asset_planning": ("asset_planning",),
    "images": ("image_jobs", "materialized_assets"),
    "models": ("adapted_model_tasks", "model_outputs"),
    "materialize": ("materialization",),
}


def invalidate_from_stage(config: PipelineConfig, stage: str) -> Path:
    """Atomically archive the named and downstream state without deleting it."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    timestamp = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{time.time_ns()}"
    stale = config.work_dir / "stale" / timestamp
    start = STAGES.index(stage)
    for current in STAGES[start:]:
        registry = _producer_registry_path(config, current)
        if registry.exists():
            destination = stale / registry.relative_to(config.work_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            registry.replace(destination)
        for relative in _STAGE_WORK_PATHS[current]:
            source = config.work_dir / relative
            if not source.exists():
                continue
            destination = stale / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
    if start <= STAGES.index("materialize") and config.output_dir.exists():
        destination = stale / "final_output"
        destination.parent.mkdir(parents=True, exist_ok=True)
        config.output_dir.replace(destination)
    return stale


def _archive_requested_caches(config: PipelineConfig, stale: Path) -> None:
    relative_paths: list[Path] = []
    if config.refresh_page_cache:
        relative_paths.extend((Path("page_cache"), Path("page_transport")))
    if config.refresh_image_cache:
        relative_paths.extend(
            (Path("image_cache"), Path("images"), Path("image_transport"))
        )
    for relative in relative_paths:
        source = config.cache_dir / relative
        if not source.exists():
            continue
        destination = stale / "cache" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)


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


def _preflight(config: PipelineConfig) -> tuple[Path, ...]:
    archives = _statistics_archives(config.input_dir)
    if config.max_source_tables <= 0:
        raise ValueError("max_source_tables must be positive")
    if config.web_max_retries != 0:
        raise ValueError("web_max_retries must be zero")
    if config.web_max_response_seconds <= 0:
        raise ValueError("web_max_response_seconds must be positive")
    if config.max_image_attempts_per_entity < 0:
        raise ValueError("max_image_attempts_per_entity must be non-negative")
    if config.max_images_per_entity < 0:
        raise ValueError("max_images_per_entity must be non-negative")
    marker_values = (
        config.model_start_marker,
        config.model_ready_marker,
        config.model_text_done_marker,
        config.model_image_done_marker,
    )
    if any(marker_values) and not all(marker_values):
        raise ValueError("all four staged model markers must be provided together")
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
    probe = config.work_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < config.min_free_disk_bytes:
        raise DiskSpaceInsufficientError(
            f"insufficient disk: free={free}, reserve={config.min_free_disk_bytes}"
        )
    return archives


def _check_disk_reserve(config: PipelineConfig, stage: str) -> None:
    probe = config.work_dir if config.work_dir.exists() else config.work_dir.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < config.min_free_disk_bytes:
        raise DiskSpaceInsufficientError(
            f"insufficient disk before {stage}: free={free}, "
            f"reserve={config.min_free_disk_bytes}"
        )


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
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
        )
        upstream = _registry_identity(config, stage)


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
        "--max_query_tables_per_source_table",
        str(config.max_query_tables_per_source_table),
        "--max_query_context_attrs",
        str(config.max_query_context_attrs),
        "--max_target_context_attrs",
        str(config.max_target_context_attrs),
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
        "--web_max_retries",
        "0",
        "--web_max_response_seconds",
        str(config.web_max_response_seconds),
        "--min_free_disk_bytes",
        str(config.min_free_disk_bytes),
    ]
    if config.text_model_base_urls:
        argv.extend(["--text_model_base_urls", *config.text_model_base_urls])
    if config.text_model_base_urls_file:
        argv.extend(
            ["--text_model_base_urls_file", config.text_model_base_urls_file]
        )
    if config.text_model_api_key:
        argv.extend(["--text_model_api_key", config.text_model_api_key])
    if config.image_model_base_urls:
        argv.extend(["--image_model_base_urls", *config.image_model_base_urls])
    if config.image_model_base_urls_file:
        argv.extend(
            ["--image_model_base_urls_file", config.image_model_base_urls_file]
        )
    if config.image_model_api_key:
        argv.extend(["--image_model_api_key", config.image_model_api_key])
    args = legacy_wdc_builder.parse_args(argv)
    args.max_rows_per_source_table = None
    return args


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
) -> Path:
    manifest_path = root / "network-manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = payload.get("counts") or {}
        if (
            payload.get("stage") == "wdc200k_network_fetch"
            and payload.get("complete") is True
            and payload.get("policy_fingerprint") == policy_fingerprint
            and declared
            == {
                "unique": unique,
                "success": success,
                "terminal": terminal,
                "pending": pending,
                "leased": leased,
            }
        ):
            _validate_producer_manifest("pages", manifest_path)
            return manifest_path
        raise ValueError(f"network manifest conflicts with durable state: {manifest_path}")
    writer = AtomicJsonlShard(root / "outcomes" / "part-00000.jsonl")
    try:
        for record in records:
            writer.write(record)
        completed = writer.commit()
    except BaseException:
        writer.abort()
        raise
    relative = CompletedShard(
        path=(root / "outcomes" / "part-00000.jsonl").relative_to(root).as_posix(),
        records=completed.records,
        bytes=completed.bytes,
        sha256=completed.sha256,
    )
    if relative.records != unique:
        raise ValueError("network outcome count does not match durable jobs")
    _atomic_json(
        manifest_path,
        {
            "stage": "wdc200k_network_fetch",
            "schema_version": "wdc200k-network-fetch-v1",
            "policy_fingerprint": policy_fingerprint,
            "counts": {
                "unique": unique,
                "success": success,
                "terminal": terminal,
                "pending": pending,
                "leased": leased,
            },
            "completed_shards": [asdict(relative)],
            "complete": True,
        },
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
) -> dict[str, int]:
    database_path = root / "structural-counts.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS page_urls "
            "(url_key TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM page_urls")
        for result in results:
            for record in _iter_jsonl(result.page_refs):
                connection.execute(
                    "INSERT OR IGNORE INTO page_urls (url_key) VALUES (?)",
                    (str(record["url_key"]),),
                )
        unique_pages = int(
            connection.execute("SELECT COUNT(*) FROM page_urls").fetchone()[0]
        )
    entities = sum(result.entities_count for result in results)
    direct_images = sum(
        result.direct_image_references for result in results
    )
    page_references = sum(result.page_references for result in results)
    validated_tables = sum(result.tables for result in results)
    image_requests = entities * config.max_image_attempts_per_entity
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
) -> tuple[
    tuple[StructuralExpansionResult, ...],
    FinalizedSelectionResult | None,
    dict[str, int],
]:
    policy = SelectionPolicy(
        target_tables=config.max_source_tables,
        seed=config.selection_seed,
        minimum3_fraction=config.minimum3_fraction,
        class_cap=config.class_max_tables,
    )
    reporter.update(stage="selection")
    selected_count, reserve_count = run_selection(
        config.input_dir,
        config.work_dir,
        policy,
    )
    selection_dir = config.work_dir / "selection"
    selection_manifest = selection_dir / "manifest.json"
    selection_counters = {
        "selected_tables": selected_count,
        "reserve_tables": reserve_count,
    }
    _write_stage_registry(
        config,
        "selection",
        producer_manifests=(selection_manifest,),
        counters=selection_counters,
        upstream_identity=input_identity,
    )
    reporter.update(counters=selection_counters)
    if selection_only:
        return (), None, selection_counters

    selected_path = selection_dir / "selected_tables.jsonl"
    reserve_path = selection_dir / "reserve_tables.jsonl"
    reserve_database = selection_dir / "reserve.sqlite3"
    reserve_manager = (
        ReserveManager.open(reserve_database, policy)
        if reserve_database.exists()
        else ReserveManager.create_from_jsonl(
            reserve_database,
            reserve_path=reserve_path,
            selected_path=selected_path,
            policy=policy,
        )
    )
    total_shards = (
        selected_count + config.selection_shard_tables - 1
    ) // config.selection_shard_tables
    reporter.update(
        stage="structural",
        completed_shards=0,
        total_shards=total_shards,
    )
    structural_root = config.work_dir / "structural"
    results = []
    validated_tables = 0
    emitted_entities = 0
    for index, records in enumerate(
        _selection_chunks(selected_path, config.selection_shard_tables)
    ):
        result = expand_selected_shard(
            records,
            output_root=structural_root,
            input_root=config.input_dir,
            reserve_manager=reserve_manager,
            shard_id=f"{index:05d}",
            min_rows=1,
            min_cols=1,
        )
        _check_disk_reserve(config, "structural")
        results.append(result)
        validated_tables += result.tables
        emitted_entities += result.entities_count
        reporter.update(
            completed_shards=index + 1,
            counters={
                "validated_tables": validated_tables,
                "entities": emitted_entities,
            },
        )
    finalized = finalize_validated_selection(
        (result.manifest for result in results),
        output_root=structural_root,
        target_tables=config.max_source_tables,
    )
    counters = {
        **selection_counters,
        **_structural_exact_counts(structural_root, results, config),
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
    )
    reporter.update(
        completed_shards=total_shards,
        counters=counters,
        known_work_bytes=counters["structural_output_bytes"],
    )
    return tuple(results), finalized, counters


def _reconcile_page_jobs_from_outcomes(
    jobs_path: Path,
    outcomes_path: Path,
    policy_fingerprint: str,
) -> None:
    """Finish crash-window jobs from already durable terminal outcomes."""
    if not jobs_path.is_file() or not outcomes_path.is_file():
        return
    with sqlite3.connect(jobs_path) as connection:
        connection.execute("ATTACH DATABASE ? AS page_cache", (str(outcomes_path),))
        kind = f"wdc200k-page:{policy_fingerprint}"
        connection.execute(
            """
            UPDATE jobs
            SET status = (
                    SELECT outcomes.status
                    FROM page_cache.page_outcomes AS outcomes
                    WHERE outcomes.policy_fingerprint = ?
                      AND outcomes.url_key = json_extract(
                          jobs.payload_json, '$.url_key'
                      )
                ),
                result_json = json_object(
                    'url_key', json_extract(payload_json, '$.url_key'),
                    'policy_fingerprint', ?
                ),
                owner = NULL,
                lease_expires = NULL,
                lease_id = NULL,
                updated_at = ?
            WHERE kind = ?
              AND status NOT IN ('success', 'terminal')
              AND EXISTS (
                    SELECT 1
                    FROM page_cache.page_outcomes AS outcomes
                    WHERE outcomes.policy_fingerprint = ?
                      AND outcomes.url_key = json_extract(
                          jobs.payload_json, '$.url_key'
                      )
                      AND outcomes.status IN ('success', 'terminal')
                )
            """,
            (
                policy_fingerprint,
                policy_fingerprint,
                time.time(),
                kind,
                policy_fingerprint,
            ),
        )
        connection.commit()


def _new_web_transport(config: PipelineConfig, namespace: str) -> Any:
    return legacy_wdc_builder.WdcWebClient(
        config.cache_dir / f"{namespace}_transport",
        max_retries=0,
        max_page_bytes=config.web_max_page_bytes,
        max_image_bytes=config.web_max_image_bytes,
        min_free_disk_bytes=config.min_free_disk_bytes,
        max_response_seconds=config.web_max_response_seconds,
        host_delay=0.0,
    )


def _page_refs(
    structural_root: Path,
    structural: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
) -> Iterator[dict[str, Any]]:
    return iter_finalized_page_refs(
        structural_root,
        finalized.manifest,
        (item.manifest for item in structural),
    )


def _run_pages(
    config: PipelineConfig,
    reporter: ProgressReporter,
    structural: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
    transport: Any,
    *,
    after_cache_write: Any | None = None,
) -> tuple[FetchResult, dict[str, Any], Path]:
    _check_disk_reserve(config, "pages")
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
    SqliteJobStore(jobs_path)
    _reconcile_page_jobs_from_outcomes(
        jobs_path,
        outcomes_path,
        policy.fingerprint,
    )
    page_completed = 0
    page_status_counts: dict[str, int] = {}

    def after_page_outcome(record: dict[str, Any]) -> None:
        nonlocal page_completed
        _check_disk_reserve(config, "pages")
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

    result = fetch_unique_pages(
        _page_refs(config.work_dir / "structural", structural, finalized),
        SqliteJobStore(jobs_path),
        transport,
        policy,
        outcomes_path=outcomes_path,
        failure_path=root / "page-failures.jsonl",
        progress_path=root / "page-progress.json",
        after_cache_write=after_page_outcome,
    )
    snapshot = validate_complete_page_fetch(
        result,
        _page_refs(config.work_dir / "structural", structural, finalized),
        validation_database=root / "validation.sqlite3",
    )
    network_manifest = _publish_network_manifest(
        root / "network",
        iter_page_outcomes(result.outcomes_path, result.policy_fingerprint),
        policy_fingerprint=result.policy_fingerprint,
        unique=result.unique,
        success=result.success,
        terminal=result.terminal,
        pending=result.remaining,
        leased=result.leased,
    )
    counters = {
        "unique_page_jobs": result.unique,
        "page_success": result.success,
        "page_terminal": result.terminal,
        "page_remaining": result.remaining,
    }
    _write_stage_registry(
        config,
        "pages",
        producer_manifests=(network_manifest,),
        counters=counters,
        upstream_identity=_registry_identity(config, "structural"),
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
    structural: Sequence[StructuralExpansionResult],
    finalized: FinalizedSelectionResult,
    page_result: FetchResult,
    page_snapshot: dict[str, Any],
) -> tuple[AssetPlanShards, str]:
    _check_disk_reserve(config, "asset_planning")
    reporter.update(
        stage="asset_planning", completed_shards=0, total_shards=1
    )
    structural_identity = structural_asset_input_identity(
        (_sha256_path(item.manifest) for item in structural),
        _sha256_path(finalized.manifest),
    )
    planning_input = asset_planning_input_fingerprint(
        structural_identity,
        str(page_snapshot["identity"]),
    )
    entity_pages = iter_entity_page_join(
        (item.entities for item in structural),
        iter_page_fanout(
            page_result.outcomes_path,
            page_result.policy_fingerprint,
        ),
        join_path=config.work_dir / "asset_planning" / "entity-pages.sqlite3",
    )
    budget = ImageBudget(
        attempts_per_entity=config.max_image_attempts_per_entity,
        retained_per_entity=config.max_images_per_entity,
    )
    planned = persist_entity_asset_plans(
        entity_pages,
        output_root=config.work_dir / "asset_planning",
        input_fingerprint=planning_input,
        budget=budget,
        records_per_shard=config.records_per_shard,
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
    )
    reporter.update(
        completed_shards=1,
        total_shards=1,
        counters=counters,
    )
    return planned, planning_input


def _run_images(
    config: PipelineConfig,
    reporter: ProgressReporter,
    planned: AssetPlanShards,
    transport: Any,
) -> tuple[
    UniqueImageJobs,
    ImageFetchResult,
    MaterializedAssetShards,
    AssetStageBarrier,
    Path,
]:
    _check_disk_reserve(config, "images")
    reporter.update(stage="images", completed_shards=0, total_shards=3)
    root = config.work_dir / "image_jobs"
    unique_jobs = build_unique_image_jobs(
        planned,
        root / "unique-images.jsonl",
    )
    validate_unique_image_jobs(
        unique_jobs,
        planned=planned,
        validation_database=root / "unique-validation.sqlite3",
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
        _check_disk_reserve(config, "images")
        reporter.update(counters={"image_completed_live": image_completed})

    image_result = fetch_unique_images(
        unique_jobs,
        SqliteJobStore(root / "jobs.sqlite3"),
        transport,
        policy,
        outcomes_path=config.cache_dir / "image_cache" / "outcomes.sqlite3",
        image_dir=config.cache_dir / "images",
        after_cache_write=after_image_outcome,
    )
    validate_complete_image_fetch(
        image_result,
        unique_jobs=unique_jobs,
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
    )
    materialized, barrier = validate_materialized_asset_shards(
        materialized,
        planned=planned,
        image_fetch_result=image_result,
        expected_input_fingerprint=asset_input,
    )
    network_manifest = _publish_network_manifest(
        root / "network",
        iter_image_outcomes(
            image_result.outcomes_path,
            image_result.policy_fingerprint,
        ),
        policy_fingerprint=image_result.policy_fingerprint,
        unique=image_result.unique,
        success=image_result.success,
        terminal=image_result.terminal,
        pending=image_result.remaining,
        leased=image_result.leased,
    )
    counters = {
        "unique_image_jobs": unique_jobs.records,
        "image_success": image_result.success,
        "image_terminal": image_result.terminal,
        "bridge_assets": materialized.bridge_assets,
        "table_asset_links": materialized.table_asset_links,
        "image_outcomes": image_result.outcomes_count,
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
    materialized_assets: MaterializedAssetShards,
    assets_barrier: AssetStageBarrier,
    network_manifests: Sequence[Path],
    extractor: Any,
) -> tuple[AdaptedModelTasks, ModelStageResult, ModelStageAuthority, argparse.Namespace]:
    _check_disk_reserve(config, "models")
    reporter.update(stage="models", completed_shards=0, total_shards=2)
    args = _runtime_args(config)
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
    )
    reporter.update(completed_shards=1, total_shards=2)
    store = SqliteJobStore(config.work_dir / "model_outputs" / "jobs.sqlite3")
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
    )
    run_fingerprint = config.run_fingerprint or stable_hash(
        "wdc200k-model-run-v1",
        adapted.input_fingerprint,
        jobset.text_fingerprint,
        jobset.image_fingerprint,
        length=40,
    )
    model_completed = 0

    def after_model_result(_job_id: str, _record: dict[str, Any]) -> None:
        nonlocal model_completed
        model_completed += 1
        _check_disk_reserve(config, "models")
        reporter.update(counters={"model_completed_live": model_completed})

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
        after_result_write=after_model_result,
        start_marker=config.model_start_marker,
        ready_marker=config.model_ready_marker,
        network_manifests=network_manifests,
        assets_manifest=materialized_assets.manifest_path,
        assets_barrier=assets_barrier,
        ready_timeout_seconds=config.model_ready_timeout_seconds,
        text_done_marker=config.model_text_done_marker,
        image_done_marker=config.model_image_done_marker,
        run_fingerprint=run_fingerprint,
    )
    authority = ModelStageAuthority.current(args)
    validate_model_stage_for_adapter(
        result,
        adapted,
        args=args,
        authority=authority,
        validation_store_path=(
            config.work_dir / "model_outputs" / "validation.sqlite3"
        ),
    )
    counters = {
        "model_text_tasks": result.text_total,
        "model_image_tasks": result.image_total,
        "model_success": result.success,
        "model_terminal": result.terminal,
    }
    _write_stage_registry(
        config,
        "models",
        producer_manifests=(adapted.manifest_path, result.manifest_path),
        counters=counters,
        upstream_identity=_registry_identity(config, "images"),
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
    page_result: FetchResult,
    planned: AssetPlanShards,
    unique_jobs: UniqueImageJobs,
    image_result: ImageFetchResult,
    materialized_assets: MaterializedAssetShards,
    adapted: AdaptedModelTasks,
    model_result: ModelStageResult,
    authority: ModelStageAuthority,
    args: argparse.Namespace,
) -> MaterializationResult:
    _check_disk_reserve(config, "materialize")
    reporter.update(stage="materialize", completed_shards=0, total_shards=1)
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
        ),
        output_root=config.output_dir,
        args=args,
        records_per_shard=config.records_per_shard,
        after_table_commit=(
            lambda _source_table_id: _check_disk_reserve(
                config, "materialize"
            )
        ),
        after_finalize_commit=(
            lambda _artifact: _check_disk_reserve(config, "materialize")
        ),
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
    )
    reporter.update(
        completed_shards=1,
        total_shards=1,
        counters=counters,
    )
    _refresh_known_disk(reporter, config, output=True)
    return result


def run_pipeline(
    config: PipelineConfig,
    *,
    page_transport: Any | None = None,
    image_transport: Any | None = None,
    extractor: Any | None = None,
    after_page_cache_write: Any | None = None,
) -> PipelineResult:
    """Run the validated staged pipeline without replaying durable outcomes."""
    archives = _preflight(config)
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
        stale = invalidate_from_stage(config, config.from_stage)
        _archive_requested_caches(config, stale)
    elif config.resume:
        _validate_existing_registry_chain(config, archives)
    elif any(_producer_registry_path(config, stage).exists() for stage in STAGES):
        raise ValueError(
            "pipeline state exists; use --resume or --from_stage selection"
        )
    reporter = ProgressReporter(config)
    reporter.start()
    try:
        structural, finalized, counters = (
            _run_selection_and_structural(
                config,
                reporter,
                input_identity=source_identity,
                selection_only=config.stop_after == "selection",
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
        structural_barrier = _structural_barrier(structural, finalized)
        if page_transport is None:
            page_transport = _new_web_transport(config, "page")
        page_result, page_snapshot, page_network_manifest = _run_pages(
            config,
            reporter,
            structural,
            finalized,
            page_transport,
            after_cache_write=after_page_cache_write,
        )
        counters.update(
            {
                "unique_page_jobs": page_result.unique,
                "page_success": page_result.success,
                "page_terminal": page_result.terminal,
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
            structural,
            finalized,
            page_result,
            page_snapshot,
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
            image_transport = _new_web_transport(config, "image")
        (
            unique_jobs,
            image_result,
            materialized_assets,
            assets_barrier,
            image_network_manifest,
        ) = _run_images(config, reporter, planned, image_transport)
        counters.update(
            {
                "unique_image_jobs": unique_jobs.records,
                "image_success": image_result.success,
                "image_terminal": image_result.terminal,
                "bridge_assets": materialized_assets.bridge_assets,
                "table_asset_links": materialized_assets.table_asset_links,
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
            materialized_assets,
            assets_barrier,
            (page_network_manifest, image_network_manifest),
            extractor,
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
            page_result,
            planned,
            unique_jobs,
            image_result,
            materialized_assets,
            adapted,
            model_result,
            authority,
            args,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--top100_policy", choices=("all",), default="all")
    parser.add_argument("--minimum3_fraction", type=float, default=0.90)
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
    parser.add_argument("--max_query_tables_per_source_table", type=int, default=0)
    parser.add_argument("--max_query_context_attrs", type=int, default=1)
    parser.add_argument("--max_target_context_attrs", type=int, default=2)
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
    parser.add_argument("--model_text_done_marker")
    parser.add_argument("--model_image_done_marker")
    parser.add_argument("--text_model_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--text_model_base_urls", nargs="*", default=None)
    parser.add_argument("--text_model_base_urls_file")
    parser.add_argument("--text_model_name", default="Qwen3.5-9B")
    parser.add_argument("--text_model_api_key")
    parser.add_argument("--image_model_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--image_model_base_urls", nargs="*", default=None)
    parser.add_argument("--image_model_base_urls_file")
    parser.add_argument("--image_model_name", default="Qwen3-VL-8B-Thinking")
    parser.add_argument("--image_model_api_key")
    parser.add_argument("--precompute_model_cache", action="store_true")
    parser.add_argument("--precompute_text_model_cache", action="store_true")
    parser.add_argument("--text_model_workers", type=int, default=1)
    parser.add_argument("--image_model_workers", type=int, default=1)
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

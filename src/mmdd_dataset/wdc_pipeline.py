from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .extraction import OpenAICompatibleExtractor, PROMPT_VERSION
from .joinability import BuildConfig, build_joinability_for_table, table_asset_links
from .utils import clean_text, get_cell, get_column_name, stable_hash
from .wdc_adapter import adapt_table, iter_candidates, iter_gzip_paths, sample_entities
from .wdc_evidence import (
    EvidenceClient,
    ResultCache,
    execute_tasks,
    normalize_public_url,
)
from .wdc_evidence_cache import EvidenceReuseCache, evidence_cache_identity
from .wdc_runtime import (
    AtomicJsonlShard,
    ShardInfo,
    add_stage_shard,
    atomic_write_json,
    bounded_map,
    check_disk_space,
    input_tree_fingerprint,
    iter_jsonl,
    load_stage_manifest,
    manifest_shards,
    publish_stage_manifest,
    sha256_path,
    stable_digest,
)


STAGES = ("select_sample", "normalize", "fetch_evidence", "extract", "materialize")
FINAL_ARTIFACTS = (
    "source_tables",
    "entities",
    "bridge_assets",
    "table_asset_links",
    "attribute_extractions",
    "query_tables",
    "data_lake_tables",
    "qrels",
    "evidence_recoveries",
    "table_queryability_decisions",
    "split_assignments",
)


@dataclass(frozen=True)
class WdcPipelineConfig:
    input_dir: Path
    work_dir: Path
    output_dir: Path
    target_tables: int = 200_000
    seed: int = 13
    shard_size: int = 100
    concurrency: int = 16
    entities_per_table: int = 8
    min_rows: int = 5
    min_cols: int = 2
    max_rows_per_table: int | None = None
    max_images_per_entity: int = 1
    max_attributes_per_table: int = 2
    min_free_bytes: int = 1_000_000_000
    user_agent: str = "MMDD WDC research dataset builder"
    request_timeout: float = 30.0
    max_page_bytes: int = 2_000_000
    max_image_bytes: int = 10_000_000
    reuse_evidence_cache: Path | None = None
    text_model_base_url: str | None = None
    text_model_name: str = "Qwen3.5-9B"
    text_model_api_key_env: str = "VLLM_API_KEY"
    image_model_base_url: str | None = None
    image_model_name: str = "Qwen3-VL-8B-Instruct"
    image_model_api_key_env: str = "VLLM_API_KEY"
    model_timeout: float = 120.0
    import_extractions: Path | None = None
    query_rows: int = 5
    min_target_rows: int = 5
    min_recovered_ratio: float = 0.6
    min_recovered_rows: int = 3
    min_column_non_empty_ratio: float = 0.5
    max_queries_per_source: int = 1
    max_query_additional_columns: int = 1
    max_target_additional_columns: int = 2
    split_by: str = "source_table_id"
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.target_tables <= 0:
            raise ValueError("target_tables must be positive")
        if self.shard_size <= 0 or self.concurrency <= 0:
            raise ValueError("shard_size and concurrency must be positive")
        if self.entities_per_table <= 0 or self.max_attributes_per_table <= 0:
            raise ValueError("entity and attribute limits must be positive")
        roots = {
            self.input_dir.resolve(),
            self.work_dir.resolve(),
            self.output_dir.resolve(),
        }
        if len(roots) != 3:
            raise ValueError("input, work, and output directories must be distinct")


def _stage_root(config: WdcPipelineConfig, stage: str) -> Path:
    return config.work_dir / stage


def _manifest_path(config: WdcPipelineConfig, stage: str) -> Path:
    return _stage_root(config, stage) / "manifest.json"


def _plain_parameters(config: WdcPipelineConfig, names: Iterable[str]) -> dict[str, Any]:
    values = asdict(config)
    return {
        name: str(values[name]) if isinstance(values[name], Path) else values[name]
        for name in names
    }


def _input_paths(config: WdcPipelineConfig) -> Iterator[Path]:
    archives = sorted(config.input_dir.rglob("*_statistics.zip"))
    if archives:
        yield from archives
    else:
        yield from iter_gzip_paths(config.input_dir)


def _input_fingerprint(config: WdcPipelineConfig) -> str:
    return input_tree_fingerprint(_input_paths(config), config.input_dir)


def _stage_input_fingerprint(config: WdcPipelineConfig, *stages: str) -> str:
    identities = []
    for stage in stages:
        path = _manifest_path(config, stage)
        if not path.is_file():
            raise ValueError(f"required stage has not run: {stage}")
        identities.append((stage, sha256_path(path)))
    return stable_digest(identities)


def _stage_is_complete(config: WdcPipelineConfig, stage: str) -> bool:
    path = _manifest_path(config, stage)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("complete") is True
    except (OSError, json.JSONDecodeError):
        return False


def _part_index(path: str) -> int:
    return int(Path(path).stem.removeprefix("part-"))


def _completed_indices(manifest: dict[str, Any], *artifacts: str) -> set[int]:
    groups = []
    for artifact in artifacts:
        groups.append(
            {
                _part_index(record["path"])
                for record in manifest.get("outputs", {}).get(artifact, [])
            }
        )
    return set.intersection(*groups) if groups else set()


def _artifact_records(manifest: dict[str, Any], artifact: str) -> list[dict[str, Any]]:
    return sorted(
        manifest.get("outputs", {}).get(artifact, []), key=lambda item: item["path"]
    )


def _artifact_path(root: Path, manifest: dict[str, Any], artifact: str, index: int) -> Path:
    for record in _artifact_records(manifest, artifact):
        if _part_index(record["path"]) == index:
            return root / record["path"]
    raise KeyError(f"missing {artifact} shard {index}")


def _new_shard(root: Path, artifact: str, index: int) -> AtomicJsonlShard:
    return AtomicJsonlShard(
        root / artifact / f"part-{index:05d}.jsonl",
        artifact=artifact,
        root=root,
    )


def _catalog_database(root: Path) -> sqlite3.Connection:
    path = root / "catalog.sqlite3"
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            relative_path TEXT PRIMARY KEY,
            schema_class TEXT NOT NULL,
            subset_name TEXT NOT NULL,
            host TEXT NOT NULL,
            rows_count INTEGER,
            columns_count INTEGER,
            rank TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS candidate_rank ON candidates(rank, relative_path)"
    )
    return connection


def _populate_catalog(
    config: WdcPipelineConfig,
    connection: sqlite3.Connection,
) -> int:
    pending = []
    for candidate in iter_candidates(config.input_dir):
        pending.append(
            (
                candidate.relative_path,
                candidate.schema_class,
                candidate.subset,
                candidate.host,
                candidate.rows,
                candidate.columns,
                stable_hash(config.seed, candidate.relative_path, length=40),
            )
        )
        if len(pending) >= 2_000:
            connection.executemany(
                "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", pending
            )
            connection.commit()
            pending.clear()
    if pending:
        connection.executemany(
            "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", pending
        )
        connection.commit()
    return int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])


def run_select_sample(
    config: WdcPipelineConfig,
    *,
    after_shard: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    stage = "select_sample"
    root = _stage_root(config, stage)
    parameters = _plain_parameters(
        config,
        (
            "target_tables",
            "shard_size",
            "entities_per_table",
            "min_rows",
            "min_cols",
            "max_rows_per_table",
        ),
    )
    fingerprint = _input_fingerprint(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root,
        stage,
        parameters=parameters,
        random_seed=config.seed,
        input_fingerprint=fingerprint,
    )
    if manifest.get("complete"):
        return manifest
    publish_stage_manifest(root, manifest)
    check_disk_space(root, config.target_tables * 4_096, config.min_free_bytes)

    with _catalog_database(root) as connection:
        if not manifest.get("state", {}).get("catalog_complete"):
            candidate_count = _populate_catalog(config, connection)
            manifest.setdefault("state", {})["catalog_complete"] = True
            manifest["counts"]["candidate_tables"] = candidate_count
            publish_stage_manifest(root, manifest)

        selected_count = int(manifest.get("counts", {}).get("selected_tables", 0))
        shard_index = len(_artifact_records(manifest, "selected_tables"))
        state = manifest.setdefault("state", {})
        last_rank = str(state.get("last_rank", ""))
        last_path = str(state.get("last_path", ""))
        skipped = int(manifest.get("counts", {}).get("skipped_tables", 0))
        cursor = connection.execute(
            """
            SELECT * FROM candidates
            WHERE rank > ? OR (rank = ? AND relative_path > ?)
            ORDER BY rank, relative_path
            """,
            (last_rank, last_rank, last_path),
        )

        selection_writer = _new_shard(root, "selected_tables", shard_index)
        entity_writer = _new_shard(root, "sampled_entities", shard_index)
        in_shard = 0
        try:
            for candidate in cursor:
                if selected_count >= config.target_tables:
                    break
                relative_path = str(candidate["relative_path"])
                path = config.input_dir / relative_path
                try:
                    sampled, rows, columns = sample_entities(
                        path,
                        config.input_dir,
                        count=config.entities_per_table,
                        seed=config.seed,
                        min_rows=config.min_rows,
                        min_cols=config.min_cols,
                        max_rows=config.max_rows_per_table,
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    skipped += 1
                    last_rank = str(candidate["rank"])
                    last_path = relative_path
                    continue
                source_table_id = sampled[0]["source_table_id"] if sampled else ""
                selection_writer.write(
                    {
                        "relative_path": relative_path,
                        "schema_class": candidate["schema_class"],
                        "subset": candidate["subset_name"],
                        "host": candidate["host"],
                        "rows": rows,
                        "columns": columns,
                        "source_table_id": source_table_id,
                        "rank": candidate["rank"],
                    }
                )
                for entity in sampled:
                    entity_writer.write(entity)
                selected_count += 1
                in_shard += 1
                last_rank = str(candidate["rank"])
                last_path = relative_path
                if in_shard == config.shard_size:
                    selection_info = selection_writer.commit()
                    entity_info = entity_writer.commit()
                    add_stage_shard(root, manifest, selection_info)
                    add_stage_shard(root, manifest, entity_info)
                    manifest["counts"]["selected_tables"] = selected_count
                    manifest["counts"]["skipped_tables"] = skipped
                    state.update({"last_rank": last_rank, "last_path": last_path})
                    publish_stage_manifest(root, manifest)
                    if after_shard:
                        after_shard(shard_index)
                    shard_index += 1
                    in_shard = 0
                    selection_writer = _new_shard(root, "selected_tables", shard_index)
                    entity_writer = _new_shard(root, "sampled_entities", shard_index)
            if in_shard:
                add_stage_shard(root, manifest, selection_writer.commit())
                add_stage_shard(root, manifest, entity_writer.commit())
                if after_shard:
                    after_shard(shard_index)
            else:
                selection_writer.abort()
                entity_writer.abort()
        except BaseException:
            selection_writer.abort()
            entity_writer.abort()
            raise

    manifest["counts"]["selected_tables"] = selected_count
    manifest["counts"]["skipped_tables"] = skipped
    state.update({"last_rank": last_rank, "last_path": last_path})
    if selected_count != config.target_tables:
        publish_stage_manifest(root, manifest)
        raise ValueError(
            f"requested {config.target_tables} valid tables, found {selected_count}"
        )
    manifest["complete"] = True
    publish_stage_manifest(root, manifest)
    print(
        f"select_sample: selected={selected_count} sampled_entities="
        f"{manifest['counts'].get('sampled_entities', 0)} skipped={skipped}"
    )
    return manifest


def run_normalize(
    config: WdcPipelineConfig,
    *,
    after_shard: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if not _stage_is_complete(config, "select_sample"):
        raise ValueError("select_sample must complete before normalize")
    stage = "normalize"
    root = _stage_root(config, stage)
    parameters = _plain_parameters(
        config, ("shard_size", "min_rows", "min_cols", "max_rows_per_table")
    )
    input_fingerprint = _stage_input_fingerprint(config, "select_sample")
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root,
        stage,
        parameters=parameters,
        random_seed=config.seed,
        input_fingerprint=input_fingerprint,
    )
    if manifest.get("complete"):
        return manifest
    publish_stage_manifest(root, manifest)
    check_disk_space(root, config.target_tables * 512_000, config.min_free_bytes)

    selection_root = _stage_root(config, "select_sample")
    selection_manifest = json.loads(
        _manifest_path(config, "select_sample").read_text(encoding="utf-8")
    )
    complete = _completed_indices(manifest, "source_tables", "entities")
    selection_shards = _artifact_records(selection_manifest, "selected_tables")
    for index, shard_record in enumerate(selection_shards):
        if index in complete:
            continue
        table_writer = _new_shard(root, "source_tables", index)
        entity_writer = _new_shard(root, "entities", index)
        try:
            for selected in iter_jsonl(selection_root / shard_record["path"]):
                adapted = adapt_table(
                    config.input_dir / selected["relative_path"],
                    config.input_dir,
                    min_rows=config.min_rows,
                    min_cols=config.min_cols,
                    max_rows=config.max_rows_per_table,
                )
                table_writer.write(adapted.table)
                for entity in adapted.entities:
                    entity_writer.write(entity)
            add_stage_shard(root, manifest, table_writer.commit())
            add_stage_shard(root, manifest, entity_writer.commit())
        except BaseException:
            table_writer.abort()
            entity_writer.abort()
            raise
        print(
            f"normalize: shard={index + 1}/{len(selection_shards)} "
            f"tables={manifest['counts'].get('source_tables', 0)}"
        )
        if after_shard:
            after_shard(index)
    manifest["complete"] = True
    publish_stage_manifest(root, manifest)
    return manifest


def _sampled_entity_shards(config: WdcPipelineConfig) -> tuple[Path, ...]:
    return manifest_shards(
        _manifest_path(config, "select_sample"),
        "sampled_entities",
    )


def _evidence_parameters(config: WdcPipelineConfig) -> dict[str, Any]:
    parameters = _plain_parameters(
        config,
        (
            "shard_size",
            "concurrency",
            "max_images_per_entity",
            "user_agent",
            "request_timeout",
            "max_page_bytes",
            "max_image_bytes",
            "reuse_evidence_cache",
        ),
    )
    parameters["reuse_evidence_cache_identity"] = (
        evidence_cache_identity(config.reuse_evidence_cache)
        if config.reuse_evidence_cache
        else None
    )
    return parameters


def _page_tasks(entities: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    tasks = []
    unsafe = 0
    for entity in entities:
        url = normalize_public_url(entity.get("page_url"))
        if url is None:
            if clean_text(entity.get("page_url")):
                unsafe += 1
            continue
        tasks.append(
            {
                "task_id": "page_" + stable_hash(entity["entity_id"], url, length=32),
                "entity_id": entity["entity_id"],
                "source_table_id": entity["source_table_id"],
                "source_row_id": entity["source_row_id"],
                "url": url,
            }
        )
    return tasks, unsafe


def _image_tasks(
    entities: Iterable[dict[str, Any]],
    page_results: Iterable[dict[str, Any]],
    *,
    max_images: int,
) -> tuple[list[dict[str, Any]], int]:
    discovered: dict[str, list[str]] = defaultdict(list)
    for result in page_results:
        if result.get("status") == "success":
            discovered[str(result["entity_id"])].extend(result.get("image_urls") or [])
    tasks = []
    unsafe = 0
    for entity in entities:
        urls = [*(entity.get("image_urls") or []), *discovered.get(entity["entity_id"], [])]
        selected: list[str] = []
        for raw_url in urls:
            url = normalize_public_url(raw_url)
            if url is None:
                if clean_text(raw_url):
                    unsafe += 1
                continue
            if url not in selected:
                selected.append(url)
            if len(selected) >= max_images:
                break
        for url in selected:
            tasks.append(
                {
                    "task_id": "image_" + stable_hash(entity["entity_id"], url, length=32),
                    "entity_id": entity["entity_id"],
                    "source_table_id": entity["source_table_id"],
                    "source_row_id": entity["source_row_id"],
                    "url": url,
                }
            )
    return tasks, unsafe


def _write_task_outcomes(
    root: Path,
    manifest: dict[str, Any],
    *,
    index: int,
    task_artifact: str,
    result_artifact: str,
    tasks: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> None:
    task_writer = _new_shard(root, task_artifact, index)
    result_writer = _new_shard(root, result_artifact, index)
    try:
        for task in tasks:
            task_writer.write(task)
        for outcome in outcomes:
            result_writer.write(outcome)
        add_stage_shard(root, manifest, task_writer.commit())
        add_stage_shard(root, manifest, result_writer.commit())
    except BaseException:
        task_writer.abort()
        result_writer.abort()
        raise


def run_fetch_evidence(
    config: WdcPipelineConfig,
    *,
    evidence_kind: str = "all",
    page_fetcher: Callable[[str], dict[str, Any]] | None = None,
    image_fetcher: Callable[[str], dict[str, Any]] | None = None,
    after_shard: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    if evidence_kind not in {"pages", "images", "all"}:
        raise ValueError("evidence_kind must be pages, images, or all")
    if not _stage_is_complete(config, "select_sample"):
        raise ValueError("select_sample must complete before fetch_evidence")
    stage = "fetch_evidence"
    root = _stage_root(config, stage)
    input_fingerprint = _stage_input_fingerprint(config, "select_sample")
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root,
        stage,
        parameters=_evidence_parameters(config),
        random_seed=config.seed,
        input_fingerprint=input_fingerprint,
    )
    if manifest.get("complete"):
        return manifest
    publish_stage_manifest(root, manifest)
    check_disk_space(
        root,
        config.target_tables * config.entities_per_table * 4_096,
        config.min_free_bytes,
    )
    cache = ResultCache(root / "evidence_cache.sqlite3")
    client = None
    if page_fetcher is None or image_fetcher is None:
        client = EvidenceClient(
            root / "cache",
            user_agent=config.user_agent,
            timeout=config.request_timeout,
            max_page_bytes=config.max_page_bytes,
            max_image_bytes=config.max_image_bytes,
        )
    page_fetch = page_fetcher or client.fetch_page  # type: ignore[union-attr]
    image_fetch = image_fetcher or client.fetch_image  # type: ignore[union-attr]
    policy = client.policy if client is not None else stable_digest("injected-evidence-v1")
    reuse_cache = (
        EvidenceReuseCache(config.reuse_evidence_cache)
        if config.reuse_evidence_cache
        else None
    )
    reuse = reuse_cache.get if reuse_cache is not None else None
    sampled_shards = _sampled_entity_shards(config)

    if evidence_kind in {"pages", "all"}:
        completed = _completed_indices(manifest, "page_tasks", "page_results")
        for index, entity_path in enumerate(sampled_shards):
            if index in completed:
                continue
            tasks, unsafe = _page_tasks(iter_jsonl(entity_path))
            outcomes = execute_tasks(
                tasks,
                kind="page",
                cache=cache,
                policy=policy,
                workers=config.concurrency,
                fetch=page_fetch,
                reuse=reuse,
            )
            _write_task_outcomes(
                root,
                manifest,
                index=index,
                task_artifact="page_tasks",
                result_artifact="page_results",
                tasks=tasks,
                outcomes=outcomes,
            )
            manifest["counts"]["unsafe_page_urls"] = (
                int(manifest["counts"].get("unsafe_page_urls", 0)) + unsafe
            )
            manifest["counts"]["reused_page_results"] = (
                int(manifest["counts"].get("reused_page_results", 0))
                + sum(bool(item.get("cache_source")) for item in outcomes)
            )
            publish_stage_manifest(root, manifest)
            print(
                f"fetch_evidence/pages: shard={index + 1}/{len(sampled_shards)} "
                f"tasks={len(tasks)}"
            )
            if after_shard:
                after_shard("pages", index)
        manifest.setdefault("substeps", {})["pages_complete"] = True
        publish_stage_manifest(root, manifest)

    if evidence_kind in {"images", "all"}:
        completed = _completed_indices(manifest, "image_tasks", "image_results")
        for index, entity_path in enumerate(sampled_shards):
            if index in completed:
                continue
            page_records: Iterable[dict[str, Any]] = ()
            try:
                page_path = _artifact_path(root, manifest, "page_results", index)
            except KeyError:
                pass
            else:
                page_records = iter_jsonl(page_path)
            tasks, unsafe = _image_tasks(
                iter_jsonl(entity_path),
                page_records,
                max_images=config.max_images_per_entity,
            )
            outcomes = execute_tasks(
                tasks,
                kind="image",
                cache=cache,
                policy=policy,
                workers=config.concurrency,
                fetch=image_fetch,
                reuse=reuse,
            )
            _write_task_outcomes(
                root,
                manifest,
                index=index,
                task_artifact="image_tasks",
                result_artifact="image_results",
                tasks=tasks,
                outcomes=outcomes,
            )
            manifest["counts"]["unsafe_image_urls"] = (
                int(manifest["counts"].get("unsafe_image_urls", 0)) + unsafe
            )
            manifest["counts"]["reused_image_results"] = (
                int(manifest["counts"].get("reused_image_results", 0))
                + sum(bool(item.get("cache_source")) for item in outcomes)
            )
            publish_stage_manifest(root, manifest)
            print(
                f"fetch_evidence/images: shard={index + 1}/{len(sampled_shards)} "
                f"tasks={len(tasks)}"
            )
            if after_shard:
                after_shard("images", index)
        manifest.setdefault("substeps", {})["images_complete"] = True
        publish_stage_manifest(root, manifest)

    substeps = manifest.setdefault("substeps", {})
    manifest["complete"] = bool(
        substeps.get("pages_complete") and substeps.get("images_complete")
    )
    publish_stage_manifest(root, manifest)
    return manifest


def _successful_assets_for_shard(
    config: WdcPipelineConfig,
    evidence_manifest: dict[str, Any],
    index: int,
) -> list[dict[str, Any]]:
    root = _stage_root(config, "fetch_evidence")
    assets: list[dict[str, Any]] = []
    for result in iter_jsonl(_artifact_path(root, evidence_manifest, "page_results", index)):
        if result.get("status") != "success" or not clean_text(result.get("text")):
            continue
        asset_id = "asset_txt_" + stable_hash(
            result["entity_id"], result.get("url"), result.get("text"), length=24
        )
        assets.append(
            {
                "asset_id": asset_id,
                "entity_id": result["entity_id"],
                "source_table_id": result["source_table_id"],
                "source_row_id": result["source_row_id"],
                "asset_type": "text",
                "source": "wdc_page",
                "page_url": result.get("url"),
                "content": clean_text(result.get("text")),
            }
        )
    for result in iter_jsonl(_artifact_path(root, evidence_manifest, "image_results", index)):
        if result.get("status") != "success" or not result.get("cache_path"):
            continue
        asset_id = "asset_img_" + stable_hash(
            result["entity_id"], result.get("url"), length=24
        )
        assets.append(
            {
                "asset_id": asset_id,
                "entity_id": result["entity_id"],
                "source_table_id": result["source_table_id"],
                "source_row_id": result["source_row_id"],
                "asset_type": "image",
                "source": "wdc_page",
                "image_url": result.get("url"),
                "local_path": result["cache_path"],
                "content_type": result.get("content_type"),
                "bytes": result.get("bytes", 0),
            }
        )
    return assets


def _candidate_attributes(
    table: dict[str, Any],
    *,
    maximum: int,
    min_non_empty_ratio: float,
) -> list[int]:
    entity_columns = set(table["metadata"]["candidate_entity_columns"])
    profiles = {
        int(profile["column_index"]): profile
        for profile in table["metadata"]["column_profiles"]
    }
    candidates = [
        int(column["column_index"])
        for column in table["columns"]
        if int(column["column_index"]) not in entity_columns
        and float(profiles[int(column["column_index"])]["non_empty_ratio"])
        >= min_non_empty_ratio
    ]
    return sorted(
        candidates,
        key=lambda index: (
            -float(profiles[index]["non_empty_ratio"]),
            -float(profiles[index]["unique_ratio"]),
            index,
        ),
    )[:maximum]


def _model_tasks_for_shard(
    config: WdcPipelineConfig,
    table_path: Path,
    sampled_path: Path,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sampled_by_table: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for entity in iter_jsonl(sampled_path):
        sampled_by_table[str(entity["source_table_id"])][int(entity["source_row_id"])] = entity
    assets_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for asset in assets:
        assets_by_entity[str(asset["entity_id"])].append(asset)
    allow_text = bool(config.text_model_base_url or config.import_extractions)
    allow_image = bool(config.image_model_base_url or config.import_extractions)
    tasks: list[dict[str, Any]] = []
    for table in iter_jsonl(table_path):
        source_table_id = str(table["source_table_id"])
        sampled = sampled_by_table.get(source_table_id, {})
        if not sampled:
            continue
        entity_col = int(table["metadata"]["candidate_entity_columns"][0])
        attributes = _candidate_attributes(
            table,
            maximum=config.max_attributes_per_table,
            min_non_empty_ratio=config.min_column_non_empty_ratio,
        )
        for row in table["rows"]:
            source_row_id = int(row["row_id"])
            entity = sampled.get(source_row_id)
            if entity is None:
                continue
            entity_cell = get_cell(row, entity_col)
            for attribute_col in attributes:
                if not clean_text(get_cell(row, attribute_col).get("text")):
                    continue
                attribute_name = get_column_name(table, attribute_col)
                visible_cells = [
                    {
                        "name": get_column_name(table, int(column["column_index"])),
                        "value": clean_text(
                            get_cell(row, int(column["column_index"])).get("text")
                        ),
                    }
                    for column in table["columns"]
                    if int(column["column_index"]) != attribute_col
                    and clean_text(get_cell(row, int(column["column_index"])).get("text"))
                ]
                for asset in assets_by_entity.get(str(entity["entity_id"]), []):
                    if asset["asset_type"] == "text" and not allow_text:
                        continue
                    if asset["asset_type"] == "image" and not allow_image:
                        continue
                    task_id = "ext_" + stable_hash(
                        source_table_id,
                        source_row_id,
                        asset["asset_id"],
                        attribute_name,
                        length=32,
                    )
                    task_asset = dict(asset)
                    if task_asset["asset_type"] == "text":
                        task_asset["content"] = clean_text(task_asset.get("content"))[:8_000]
                    tasks.append(
                        {
                            "task_id": task_id,
                            "source_table_id": source_table_id,
                            "source_row_id": source_row_id,
                            "entity_id": entity["entity_id"],
                            "entity": clean_text(entity_cell.get("text")),
                            "attribute_name": attribute_name,
                            "asset_id": asset["asset_id"],
                            "asset_type": asset["asset_type"],
                            "visible_cells": visible_cells,
                            "asset": task_asset,
                            "prompt_version": PROMPT_VERSION,
                        }
                    )
    return tasks


def _import_paths(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.rglob("*.jsonl"))
    else:
        raise FileNotFoundError(path)


def _result_task_id(record: dict[str, Any]) -> str:
    if record.get("task_id"):
        return str(record["task_id"])
    return "ext_" + stable_hash(
        record["source_table_id"],
        int(record["source_row_id"]),
        record["asset_id"],
        record["attribute_name"],
        length=32,
    )


def _load_imported_results(
    cache: ResultCache,
    namespace: str,
    path: Path,
) -> int:
    count = 0
    for input_path in _import_paths(path):
        for record in iter_jsonl(input_path):
            task_id = _result_task_id(record)
            cache.put(namespace, task_id, record)
            count += 1
    return count


def _canonical_extraction(
    task: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "extraction_id": task["task_id"],
        "source_table_id": task["source_table_id"],
        "source_row_id": task["source_row_id"],
        "entity_id": task["entity_id"],
        "asset_id": task["asset_id"],
        "asset_type": task["asset_type"],
        "attribute_name": task["attribute_name"],
        "value": clean_text(result.get("value")),
        "evidence": clean_text(result.get("evidence")),
        "error": clean_text(result.get("error")) or None,
        "prompt_version": task["prompt_version"],
    }


def _run_model_tasks(
    tasks: list[dict[str, Any]],
    *,
    cache: ResultCache,
    namespace: str,
    workers: int,
    text_extractor: OpenAICompatibleExtractor | None,
    image_extractor: OpenAICompatibleExtractor | None,
) -> tuple[list[dict[str, Any]], int]:
    def execute(task: dict[str, Any]) -> dict[str, Any] | None:
        cached = cache.get(namespace, str(task["task_id"]))
        if cached is not None:
            return _canonical_extraction(task, cached)
        extractor = image_extractor if task["asset_type"] == "image" else text_extractor
        if extractor is None:
            return None
        try:
            result = extractor.extract(
                entity=task["entity"],
                attribute=task["attribute_name"],
                visible_cells=task["visible_cells"],
                asset=task["asset"],
            )
        except Exception as error:
            result = {
                "value": "",
                "evidence": "",
                "error": f"{type(error).__name__}: {clean_text(error)}",
            }
        cache.put(namespace, str(task["task_id"]), result)
        return _canonical_extraction(task, result)

    raw_results = list(
        bounded_map(execute, tasks, workers=workers, max_pending=workers * 2)
    )
    return [result for result in raw_results if result is not None], sum(
        result is None for result in raw_results
    )


def run_extract(
    config: WdcPipelineConfig,
    *,
    after_shard: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if not _stage_is_complete(config, "normalize") or not _stage_is_complete(
        config, "fetch_evidence"
    ):
        raise ValueError("normalize and fetch_evidence must complete before extract")
    stage = "extract"
    root = _stage_root(config, stage)
    parameters = _plain_parameters(
        config,
        (
            "shard_size",
            "concurrency",
            "max_attributes_per_table",
            "min_column_non_empty_ratio",
            "text_model_base_url",
            "text_model_name",
            "image_model_base_url",
            "image_model_name",
            "model_timeout",
            "import_extractions",
        ),
    )
    input_fingerprint = _stage_input_fingerprint(config, "normalize", "fetch_evidence")
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root,
        stage,
        parameters=parameters,
        random_seed=config.seed,
        input_fingerprint=input_fingerprint,
    )
    if manifest.get("complete"):
        return manifest
    publish_stage_manifest(root, manifest)
    check_disk_space(
        root,
        config.target_tables * config.entities_per_table * 8_192,
        config.min_free_bytes,
    )

    model_identity = stable_digest(
        PROMPT_VERSION,
        config.text_model_base_url,
        config.text_model_name,
        config.image_model_base_url,
        config.image_model_name,
    )
    namespace = f"model:{model_identity}"
    cache = ResultCache(root / "model_cache.sqlite3")
    if config.import_extractions:
        imported = _load_imported_results(cache, namespace, config.import_extractions)
        manifest["counts"]["imported_extractions"] = imported
        publish_stage_manifest(root, manifest)

    text_extractor = (
        OpenAICompatibleExtractor(
            config.text_model_base_url,
            config.text_model_name,
            api_key_env=config.text_model_api_key_env,
            timeout=config.model_timeout,
        )
        if config.text_model_base_url
        else None
    )
    image_extractor = (
        OpenAICompatibleExtractor(
            config.image_model_base_url,
            config.image_model_name,
            api_key_env=config.image_model_api_key_env,
            timeout=config.model_timeout,
        )
        if config.image_model_base_url
        else None
    )
    normalize_manifest = json.loads(
        _manifest_path(config, "normalize").read_text(encoding="utf-8")
    )
    evidence_manifest = json.loads(
        _manifest_path(config, "fetch_evidence").read_text(encoding="utf-8")
    )
    normalize_root = _stage_root(config, "normalize")
    sampled_shards = _sampled_entity_shards(config)
    table_records = _artifact_records(normalize_manifest, "source_tables")

    completed_tasks = {
        _part_index(record["path"])
        for record in _artifact_records(manifest, "model_tasks")
    }
    for index, table_record in enumerate(table_records):
        if index in completed_tasks:
            continue
        assets = _successful_assets_for_shard(config, evidence_manifest, index)
        tasks = _model_tasks_for_shard(
            config,
            normalize_root / table_record["path"],
            sampled_shards[index],
            assets,
        )
        writer = _new_shard(root, "model_tasks", index)
        try:
            for task in tasks:
                writer.write(task)
            add_stage_shard(root, manifest, writer.commit())
        except BaseException:
            writer.abort()
            raise

    completed_results = {
        _part_index(record["path"])
        for record in _artifact_records(manifest, "model_results")
    }
    total_pending = 0
    for record in _artifact_records(manifest, "model_tasks"):
        index = _part_index(record["path"])
        if index in completed_results:
            continue
        tasks = list(iter_jsonl(root / record["path"]))
        results, pending = _run_model_tasks(
            tasks,
            cache=cache,
            namespace=namespace,
            workers=config.concurrency,
            text_extractor=text_extractor,
            image_extractor=image_extractor,
        )
        if pending:
            total_pending += pending
            continue
        writer = _new_shard(root, "model_results", index)
        try:
            for result in results:
                writer.write(result)
            add_stage_shard(root, manifest, writer.commit())
        except BaseException:
            writer.abort()
            raise
        print(f"extract: shard={index + 1}/{len(table_records)} tasks={len(tasks)}")
        if after_shard:
            after_shard(index)

    manifest["counts"]["pending_model_tasks"] = total_pending
    manifest["model_identity"] = {
        "prompt_version": PROMPT_VERSION,
        "text_model": config.text_model_name if config.text_model_base_url else None,
        "image_model": config.image_model_name if config.image_model_base_url else None,
    }
    manifest["complete"] = (
        len(_artifact_records(manifest, "model_results")) == len(table_records)
    )
    publish_stage_manifest(root, manifest)
    return manifest


def _split_for(table: dict[str, Any], config: WdcPipelineConfig) -> str:
    key = clean_text(table.get(config.split_by)) or str(table["source_table_id"])
    value = int(stable_hash(config.seed, key, length=15), 16) / float(16**15)
    ratio_sum = config.train_ratio + config.dev_ratio + config.test_ratio
    train_end = config.train_ratio / ratio_sum
    dev_end = train_end + config.dev_ratio / ratio_sum
    if value < train_end:
        return "train"
    if value < dev_end:
        return "dev"
    return "test"


def _copy_final_image(config: WdcPipelineConfig, asset: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(asset["local_path"]))
    suffix = source.suffix[:8] or ".img"
    relative = Path("images") / f"{asset['asset_id']}{suffix}"
    target = config.output_dir / relative
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
    result = dict(asset)
    result["relative_path"] = relative.as_posix()
    result["local_path"] = relative.as_posix()
    return result


def _finalize_assets(
    config: WdcPipelineConfig, assets: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for asset in assets:
        if asset["asset_type"] == "image":
            output.append(_copy_final_image(config, asset))
        else:
            output.append(dict(asset))
    return output


def _group_by_table(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_table_id"])].append(record)
    return grouped


def _build_config(config: WdcPipelineConfig) -> BuildConfig:
    return BuildConfig(
        query_rows=config.query_rows,
        min_target_rows=config.min_target_rows,
        min_recovered_ratio=config.min_recovered_ratio,
        min_recovered_rows=config.min_recovered_rows,
        min_column_non_empty_ratio=config.min_column_non_empty_ratio,
        max_queries_per_source=config.max_queries_per_source,
        max_query_additional_columns=config.max_query_additional_columns,
        max_target_additional_columns=config.max_target_additional_columns,
    )


def _materialize_one_shard(
    config: WdcPipelineConfig,
    *,
    index: int,
    normalize_manifest: dict[str, Any],
    evidence_manifest: dict[str, Any],
    extract_manifest: dict[str, Any],
) -> list[ShardInfo]:
    normalize_root = _stage_root(config, "normalize")
    extract_root = _stage_root(config, "extract")
    table_path = _artifact_path(normalize_root, normalize_manifest, "source_tables", index)
    entity_path = _artifact_path(normalize_root, normalize_manifest, "entities", index)
    result_path = _artifact_path(extract_root, extract_manifest, "model_results", index)
    entities_by_table = _group_by_table(iter_jsonl(entity_path))
    assets = _finalize_assets(
        config, _successful_assets_for_shard(config, evidence_manifest, index)
    )
    assets_by_table = _group_by_table(assets)
    extractions = list(iter_jsonl(result_path))
    extractions_by_table = _group_by_table(extractions)
    writers = {
        artifact: AtomicJsonlShard(
            config.output_dir / artifact / f"part-{index:05d}.jsonl",
            artifact=artifact,
            root=config.output_dir,
        )
        for artifact in FINAL_ARTIFACTS
    }
    algorithm_config = _build_config(config)
    try:
        for asset in assets:
            writers["bridge_assets"].write(asset)
        for extraction in extractions:
            writers["attribute_extractions"].write(extraction)
        for table in iter_jsonl(table_path):
            source_table_id = str(table["source_table_id"])
            table_entities = entities_by_table.get(source_table_id, [])
            table_assets = assets_by_table.get(source_table_id, [])
            table_extractions = extractions_by_table.get(source_table_id, [])
            split = _split_for(table, config)
            writers["source_tables"].write(table)
            for entity in table_entities:
                writers["entities"].write(entity)
            for link in table_asset_links([table], table_entities, table_assets):
                writers["table_asset_links"].write(link)
            built = build_joinability_for_table(
                table,
                table_assets,
                table_extractions,
                split,
                algorithm_config,
            )
            for artifact in (
                "query_tables",
                "data_lake_tables",
                "qrels",
                "evidence_recoveries",
                "table_queryability_decisions",
            ):
                for record in built[artifact]:
                    writers[artifact].write(record)
            writers["split_assignments"].write(
                {
                    "object_id": source_table_id,
                    "object_type": "source_table",
                    "split": split,
                }
            )
            for query in built["query_tables"]:
                writers["split_assignments"].write(
                    {
                        "object_id": query["table_id"],
                        "object_type": "query_table",
                        "source_table_id": source_table_id,
                        "split": split,
                    }
                )
            for target in built["data_lake_tables"]:
                writers["split_assignments"].write(
                    {
                        "object_id": target["table_id"],
                        "object_type": "data_lake_table",
                        "source_table_id": source_table_id,
                        "split": split,
                    }
                )
        return [writers[artifact].commit() for artifact in FINAL_ARTIFACTS]
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise


def _final_manifest(
    config: WdcPipelineConfig,
    materialize_manifest: dict[str, Any],
    input_fingerprint: str,
) -> dict[str, Any]:
    extraction_identity = json.loads(
        _manifest_path(config, "extract").read_text(encoding="utf-8")
    ).get("model_identity", {})
    artifacts = {}
    for artifact in FINAL_ARTIFACTS:
        shards = []
        for record in _artifact_records(materialize_manifest, artifact):
            shards.append(
                {
                    key: value
                    for key, value in record.items()
                    if key in {"path", "records", "bytes", "sha256"}
                }
            )
        artifacts[artifact] = {
            "directory": artifact,
            "total_records": sum(int(record["records"]) for record in shards),
            "shards": shards,
        }
    return {
        "format": "mmdd_joinability_sharded_v2",
        "source": "wdc_schemaorg_tables",
        "input_fingerprint": input_fingerprint,
        "build": {
            "random_seed": config.seed,
            "target_tables": config.target_tables,
            "records_per_input_shard": config.shard_size,
            "joinability": asdict(_build_config(config)),
            "extraction": extraction_identity,
            "split": {
                "split_by": config.split_by,
                "train_ratio": config.train_ratio,
                "dev_ratio": config.dev_ratio,
                "test_ratio": config.test_ratio,
            },
        },
        "artifacts": artifacts,
        "single_files": {
            "stats": {
                "path": "stats.json",
                "bytes": (config.output_dir / "stats.json").stat().st_size,
                "sha256": sha256_path(config.output_dir / "stats.json"),
            },
            "splits": {
                "path": "splits.json",
                "bytes": (config.output_dir / "splits.json").stat().st_size,
                "sha256": sha256_path(config.output_dir / "splits.json"),
            },
        },
        "complete": True,
        "note": "All paths are relative to this dataset directory.",
    }


def run_materialize(
    config: WdcPipelineConfig,
    *,
    after_shard: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    required = ("normalize", "fetch_evidence", "extract")
    if not all(_stage_is_complete(config, stage) for stage in required):
        raise ValueError("normalize, fetch_evidence, and extract must complete first")
    stage = "materialize"
    root = _stage_root(config, stage)
    parameters = _plain_parameters(
        config,
        (
            "target_tables",
            "shard_size",
            "query_rows",
            "min_target_rows",
            "min_recovered_ratio",
            "min_recovered_rows",
            "min_column_non_empty_ratio",
            "max_queries_per_source",
            "max_query_additional_columns",
            "max_target_additional_columns",
            "split_by",
            "train_ratio",
            "dev_ratio",
            "test_ratio",
        ),
    )
    input_fingerprint = _stage_input_fingerprint(config, *required)
    root.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_stage_manifest(
        root,
        stage,
        parameters=parameters,
        random_seed=config.seed,
        input_fingerprint=input_fingerprint,
        shard_root=config.output_dir,
    )
    if manifest.get("complete"):
        return manifest
    publish_stage_manifest(root, manifest)
    check_disk_space(
        config.output_dir,
        config.target_tables * 800_000,
        config.min_free_bytes,
    )
    normalize_manifest = json.loads(
        _manifest_path(config, "normalize").read_text(encoding="utf-8")
    )
    evidence_manifest = json.loads(
        _manifest_path(config, "fetch_evidence").read_text(encoding="utf-8")
    )
    extract_manifest = json.loads(
        _manifest_path(config, "extract").read_text(encoding="utf-8")
    )
    table_shards = _artifact_records(normalize_manifest, "source_tables")
    completed = _completed_indices(manifest, *FINAL_ARTIFACTS)
    for index in range(len(table_shards)):
        if index in completed:
            continue
        infos = _materialize_one_shard(
            config,
            index=index,
            normalize_manifest=normalize_manifest,
            evidence_manifest=evidence_manifest,
            extract_manifest=extract_manifest,
        )
        for info in infos:
            add_stage_shard(root, manifest, info)
        print(f"materialize: shard={index + 1}/{len(table_shards)}")
        if after_shard:
            after_shard(index)

    stats = {
        artifact: int(manifest["counts"].get(artifact, 0))
        for artifact in FINAL_ARTIFACTS
    }
    split_counts = {"train": 0, "dev": 0, "test": 0}
    for record in _artifact_records(manifest, "split_assignments"):
        for assignment in iter_jsonl(config.output_dir / record["path"]):
            if assignment["object_type"] == "source_table":
                split_counts[str(assignment["split"])] += 1
    atomic_write_json(config.output_dir / "stats.json", stats)
    atomic_write_json(
        config.output_dir / "splits.json",
        {
            "split_key": config.split_by,
            "source_table_counts": split_counts,
            "assignments_artifact": "split_assignments",
        },
    )
    dataset_manifest = _final_manifest(config, manifest, input_fingerprint)
    atomic_write_json(config.output_dir / "dataset_manifest.json", dataset_manifest)
    manifest["complete"] = True
    manifest["dataset_manifest_sha256"] = sha256_path(
        config.output_dir / "dataset_manifest.json"
    )
    publish_stage_manifest(root, manifest)
    return manifest


def dry_run(config: WdcPipelineConfig) -> dict[str, Any]:
    candidates = 0
    declared_rows = 0
    declared_row_tables = 0
    for candidate in iter_candidates(config.input_dir):
        candidates += 1
        if candidate.rows is not None:
            declared_rows += candidate.rows
            declared_row_tables += 1
    tables = min(config.target_tables, candidates)
    entities = tables * config.entities_per_table
    page_tasks = entities
    image_tasks = entities * config.max_images_per_entity
    model_tasks = (
        entities
        * config.max_attributes_per_table
        * (1 + config.max_images_per_entity)
    )
    average_rows = declared_rows / declared_row_tables if declared_row_tables else 50.0
    estimated_bytes = int(
        tables * max(64_000, average_rows * 1_600) * 2
        + (page_tasks + image_tasks) * 4_096
        + model_tasks * 3_072
    )
    result = {
        "dry_run": True,
        "candidate_tables": candidates,
        "selected_tables": tables,
        "sampled_entities": entities,
        "estimated_page_tasks": page_tasks,
        "estimated_image_tasks": image_tasks,
        "estimated_model_tasks_upper_bound": model_tasks,
        "estimated_disk_bytes": estimated_bytes,
        "network_or_model_calls": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def run_pipeline(config: WdcPipelineConfig, *, start_at_first_incomplete: bool) -> int:
    started = not start_at_first_incomplete
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "select_sample": lambda: run_select_sample(config),
        "normalize": lambda: run_normalize(config),
        "fetch_evidence": lambda: run_fetch_evidence(config, evidence_kind="all"),
        "extract": lambda: run_extract(config),
        "materialize": lambda: run_materialize(config),
    }
    for stage in STAGES:
        if not started:
            started = not _stage_is_complete(config, stage)
        if not started:
            continue
        result = runners[stage]()
        if result.get("complete") is not True:
            print(f"{stage}: incomplete; provide the required endpoint or imported results")
            return 2
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Build the WDC joinability dataset with resumable JSONL stages."
    )
    result.add_argument(
        "command", choices=(*STAGES, "run", "resume"), help="stage or orchestration command"
    )
    result.add_argument("--input-dir", required=True)
    result.add_argument("--work-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--target-tables", type=int, default=200_000)
    result.add_argument("--seed", type=int, default=13)
    result.add_argument("--shard-size", type=int, default=100)
    result.add_argument("--concurrency", type=int, default=16)
    result.add_argument("--entities-per-table", type=int, default=8)
    result.add_argument("--min-rows", type=int, default=5)
    result.add_argument("--min-cols", type=int, default=2)
    result.add_argument("--max-rows-per-table", type=int)
    result.add_argument("--max-images-per-entity", type=int, default=1)
    result.add_argument("--max-attributes-per-table", type=int, default=2)
    result.add_argument("--min-free-gb", type=float, default=1.0)
    result.add_argument("--user-agent", default="MMDD WDC research dataset builder")
    result.add_argument("--request-timeout", type=float, default=30.0)
    result.add_argument("--max-page-bytes", type=int, default=2_000_000)
    result.add_argument("--max-image-bytes", type=int, default=10_000_000)
    result.add_argument("--reuse-evidence-cache")
    result.add_argument("--evidence-kind", choices=("pages", "images", "all"), default="all")
    result.add_argument("--text-model-base-url")
    result.add_argument("--text-model-name", default="Qwen3.5-9B")
    result.add_argument("--text-model-api-key-env", default="VLLM_API_KEY")
    result.add_argument("--image-model-base-url")
    result.add_argument("--image-model-name", default="Qwen3-VL-8B-Instruct")
    result.add_argument("--image-model-api-key-env", default="VLLM_API_KEY")
    result.add_argument("--model-timeout", type=float, default=120.0)
    result.add_argument("--import-extractions")
    result.add_argument("--query-rows", type=int, default=5)
    result.add_argument("--min-target-rows", type=int, default=5)
    result.add_argument("--min-recovered-ratio", type=float, default=0.6)
    result.add_argument("--min-recovered-rows", type=int, default=3)
    result.add_argument("--min-column-non-empty-ratio", type=float, default=0.5)
    result.add_argument("--max-queries-per-source", type=int, default=1)
    result.add_argument("--max-query-additional-columns", type=int, default=1)
    result.add_argument("--max-target-additional-columns", type=int, default=2)
    result.add_argument("--split-by", choices=("source_table_id",), default="source_table_id")
    result.add_argument("--train-ratio", type=float, default=0.8)
    result.add_argument("--dev-ratio", type=float, default=0.1)
    result.add_argument("--test-ratio", type=float, default=0.1)
    result.add_argument("--dry-run", action="store_true")
    return result


def _config_from_args(args: argparse.Namespace) -> WdcPipelineConfig:
    return WdcPipelineConfig(
        input_dir=Path(args.input_dir),
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        target_tables=args.target_tables,
        seed=args.seed,
        shard_size=args.shard_size,
        concurrency=args.concurrency,
        entities_per_table=args.entities_per_table,
        min_rows=args.min_rows,
        min_cols=args.min_cols,
        max_rows_per_table=args.max_rows_per_table,
        max_images_per_entity=args.max_images_per_entity,
        max_attributes_per_table=args.max_attributes_per_table,
        min_free_bytes=int(args.min_free_gb * 1024**3),
        user_agent=args.user_agent,
        request_timeout=args.request_timeout,
        max_page_bytes=args.max_page_bytes,
        max_image_bytes=args.max_image_bytes,
        reuse_evidence_cache=(
            Path(args.reuse_evidence_cache) if args.reuse_evidence_cache else None
        ),
        text_model_base_url=args.text_model_base_url,
        text_model_name=args.text_model_name,
        text_model_api_key_env=args.text_model_api_key_env,
        image_model_base_url=args.image_model_base_url,
        image_model_name=args.image_model_name,
        image_model_api_key_env=args.image_model_api_key_env,
        model_timeout=args.model_timeout,
        import_extractions=(Path(args.import_extractions) if args.import_extractions else None),
        query_rows=args.query_rows,
        min_target_rows=args.min_target_rows,
        min_recovered_ratio=args.min_recovered_ratio,
        min_recovered_rows=args.min_recovered_rows,
        min_column_non_empty_ratio=args.min_column_non_empty_ratio,
        max_queries_per_source=args.max_queries_per_source,
        max_query_additional_columns=args.max_query_additional_columns,
        max_target_additional_columns=args.max_target_additional_columns,
        split_by=args.split_by,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = _config_from_args(args)
    if args.dry_run:
        dry_run(config)
        return 0
    if args.command == "run":
        return run_pipeline(config, start_at_first_incomplete=False)
    if args.command == "resume":
        return run_pipeline(config, start_at_first_incomplete=True)
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "select_sample": lambda: run_select_sample(config),
        "normalize": lambda: run_normalize(config),
        "fetch_evidence": lambda: run_fetch_evidence(
            config, evidence_kind=args.evidence_kind
        ),
        "extract": lambda: run_extract(config),
        "materialize": lambda: run_materialize(config),
    }
    stage_result = runners[args.command]()
    if args.command == "extract" and stage_result.get("complete") is not True:
        return 2
    return 0

from __future__ import annotations

import json
import gzip
import hashlib
import io
import os
import base64
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts_old"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_wdc200k_mm_joinability_dataset import (  # noqa: E402
    STAGES,
    DiskGuard,
    DiskSpaceInsufficientError,
    PipelineConfig,
    ProgressReporter,
    _publish_network_manifest,
    invalidate_from_stage,
    parse_args,
    run_pipeline,
)
import build_wdc200k_mm_joinability_dataset as pipeline_module  # noqa: E402
import wdc200k_balance_progress as balance_progress  # noqa: E402
from wdc200k_assets import ImageOutcomeStore  # noqa: E402
from wdc200k_fetch import (  # noqa: E402
    FetchPolicy,
    PageOutcomeStore,
    fetch_unique_pages,
)
from wdc200k_io import SqliteJobStore  # noqa: E402
from wdc200k_materialize import MaterializationInputs  # noqa: E402
from wdc200k_eta import (  # noqa: E402
    DurableUrlCounts,
    UrlProgressSnapshot,
    UrlProgressTracker,
    decode_histogram_blob,
    encode_histogram_blob,
)


URL_TELEMETRY_V2 = "wdc200k-url-telemetry-v2"
UINT64_MAX = 2**64 - 1


def _statistics_archive(input_dir: Path, *, rows: int = 2) -> Path:
    class_dir = input_dir / "Thing"
    class_dir.mkdir(parents=True)
    archive = class_dir / "Thing_statistics.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for subset in ("top100", "minimum3", "rest"):
            zipped.writestr(
                "table_statistics/"
                f"Thing_October2023_statistics_{subset}.csv",
                "host,number_of_rows,column_count,column_name_and_density\n"
                f"{subset}.example,{rows},5,\n",
            )
    return archive


def _write_selected_table(
    input_dir: Path,
    *,
    rows_count: int = 2,
    material_rows: int | None = None,
) -> None:
    path = (
        input_dir
        / "Thing"
        / "Thing_top100.example_October2023.json.gz"
    )
    rows = [
        {
            "name": "Alpha",
            "State": "Texas",
            "Category": "Place",
            "page_url": "https://pages.example/shared",
            "image": "https://images.example/alpha.jpg",
        },
        {
            "name": "Beta",
            "State": "Ohio",
            "Category": "Place",
            "page_url": "https://pages.example/shared",
            "image": [
                "https://images.example/beta-1.jpg",
                "https://images.example/beta-2.jpg",
            ],
        },
    ]
    rows.extend(
        {
            "name": name,
            "State": state,
            "Category": "Place",
            "page_url": "https://pages.example/shared",
            "image": "",
        }
        for name, state in (
            ("Gamma", "Utah"),
            ("Delta", "Maine"),
            ("Epsilon", "Iowa"),
        )[: max(0, rows_count - 2)]
    )
    if material_rows is not None:
        for index, row in enumerate(rows):
            if index >= material_rows:
                row["page_url"] = ""
                row["image"] = ""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_cli_defaults_match_approved_policy(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    config = PipelineConfig.from_args(args)
    assert config.cache_dir == (tmp_path / "cache" / "wdc_webtable").resolve()
    assert args.max_source_tables == 200_000
    assert args.max_rows_per_source_table is None
    assert args.selection_seed == 13
    assert args.top100_policy == "bounded"
    assert args.top100_per_class == 10
    assert args.include_rest is False
    assert args.min_candidate_rows == 5
    assert args.min_candidate_columns == 3
    assert args.minimum3_fraction == 1.0
    assert args.web_max_retries == 0
    assert args.web_max_response_seconds == 8
    assert args.web_global_concurrency == 128
    assert args.web_per_host_concurrency == 2
    assert args.web_proxy_url is None
    assert args.max_image_attempts_per_entity == 3
    assert args.max_images_per_entity == 3
    assert tuple(args.sampled_entity_expansion_schedule) == (12, 20, 25)
    assert args.resume is True
    assert args.refresh_page_cache is False
    assert args.refresh_image_cache is False
    assert args.model_endpoint_ready_timeout_seconds == 30.0
    assert args.model_timeout_seconds == 120.0
    assert args.model_max_retries == 2
    assert args.model_retry_sleep_seconds == 2.0
    assert args.model_cache_database_path is None
    assert args.unrecoverable_replacement_rounds == 0
    assert args.unrecoverable_drop_probability == 0.5
    assert args.model_adapter_workers == 1
    assert args.materialization_workers == 1
    assert args.materialization_validation_workers == 3


def test_cli_configures_explicit_web_proxy(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "out"),
            "--web_proxy_url",
            "http://127.0.0.1:7890",
        ]
    )

    assert PipelineConfig.from_args(args).web_proxy_url == (
        "http://127.0.0.1:7890"
    )


def test_progressive_sampling_expands_failed_tables_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
            ]
        )
    )
    decisions = {
        "expand": {
            "source_table_id": "expand",
            "eligible": True,
            "sampled_entities": 8,
            "attemptable_entities": 25,
            "sample_limit": 8,
        },
        "exhausted": {
            "source_table_id": "exhausted",
            "eligible": True,
            "sampled_entities": 8,
            "attemptable_entities": 8,
            "sample_limit": 8,
        },
        "prefiltered": {
            "source_table_id": "prefiltered",
            "eligible": False,
            "sampled_entities": 0,
        },
    }
    monkeypatch.setattr(
        pipeline_module,
        "_sampling_prefilter_decisions",
        lambda _config: decisions,
    )
    failures = [
        {"source_table_id": table_id}
        for table_id in ("expand", "exhausted", "prefiltered")
    ]
    state = pipeline_module.SamplingExpansionState(0, None, {})

    limits, expanded, stats = pipeline_module._plan_sampling_expansion(
        config,
        state=state,
        failures=failures,
    )

    assert limits == {"expand": 12}
    assert expanded == ["expand"]
    assert stats == {
        "failed_tables": 3,
        "eligible_failed_tables": 2,
        "expanded_tables": 1,
        "sampling_exhausted_failed_tables": 2,
    }
    persisted = pipeline_module._write_sampling_expansion_state(
        config,
        previous=state,
        limits=limits,
        expanded_table_ids=expanded,
    )
    assert pipeline_module._load_sampling_expansion_state(config) == persisted

    decisions["expand"].update(
        sampled_entities=12,
        sample_limit=12,
    )
    next_limits, next_expanded, _ = (
        pipeline_module._plan_sampling_expansion(
            config,
            state=persisted,
            failures=failures,
        )
    )
    assert next_limits == {"expand": 20}
    assert next_expanded == ["expand"]


def test_legacy_work_model_cache_is_copied_to_shared_cache(
    tmp_path: Path,
) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
            ]
        )
    )
    legacy = config.work_dir / "model_outputs" / "jobs.sqlite3"
    legacy.parent.mkdir(parents=True)
    with sqlite3.connect(legacy) as connection:
        connection.execute("CREATE TABLE cached (value TEXT NOT NULL)")
        connection.execute("INSERT INTO cached VALUES ('reused')")

    target = pipeline_module._promote_legacy_model_cache(config)

    assert target == config.cache_dir / "model_cache" / "jobs.sqlite3"
    assert legacy.is_file()
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT value FROM cached").fetchone()[0] == (
            "reused"
        )


def test_recovery_replacement_claims_every_failed_slot_and_persists_active_round(
    tmp_path: Path,
) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--max_source_tables",
                "2",
                "--class_max_tables",
                "2",
                "--unrecoverable_replacement_rounds",
                "5",
                "--unrecoverable_drop_probability",
                "1",
            ]
        )
    )
    from wdc200k_selection import ReserveManager, SelectionPolicy, TableCandidate

    selected = [
        TableCandidate(
            "Thing",
            "minimum3",
            f"selected-{index}.test",
            f"Thing/Thing_selected-{index}.test_October2023.json.gz",
            8,
            4,
        )
        for index in range(2)
    ]
    reserve = [
        TableCandidate(
            "Thing",
            "minimum3",
            f"reserve-{index}.test",
            f"Thing/Thing_reserve-{index}.test_October2023.json.gz",
            9,
            5,
        )
        for index in range(2)
    ]
    selection_root = config.work_dir / "selection"
    selection_root.mkdir(parents=True)
    ReserveManager.create(
        selection_root / "reserve.sqlite3",
        reserve=reserve,
        selected=selected,
        policy=SelectionPolicy(
            target_tables=2,
            seed=13,
            top100_policy=config.top100_policy,
            top100_per_class=config.top100_per_class,
            include_rest=config.include_rest,
            min_candidate_rows=config.min_candidate_rows,
            min_candidate_columns=config.min_candidate_columns,
            minimum3_fraction=config.minimum3_fraction,
            rest_base_per_class=0,
            class_cap=2,
        ),
    )
    active = [
        {
            **candidate.__dict__,
            "source_table_id": f"source-{index}",
            "selection_seed": 13,
        }
        for index, candidate in enumerate(selected)
    ]
    # Structural validation records the materialized table shape, which can
    # differ from the statistics metadata that identifies reserve candidates.
    active[0]["columns"] += 1
    active[1]["rows"] += 2

    replaced, stats = pipeline_module._claim_recovery_replacements(
        config,
        round_index=1,
        active_records=active,
        failures=active,
    )

    assert stats == {
        "failed": 2,
        "replaced": 2,
        "retained_by_probability": 0,
        "reserve_exhausted": 0,
    }
    assert {record["relative_path"] for record in replaced} == {
        candidate.relative_path for candidate in reserve
    }
    previous = tmp_path / "previous.jsonl"
    previous.write_text("{}\n", encoding="utf-8")
    state = pipeline_module._write_active_recovery_selection(
        config,
        round_index=1,
        records=replaced,
        previous_selection_path=previous,
    )
    assert state.records == 2
    assert pipeline_module._load_active_recovery_selection(config) == state


def test_wdc200k_cli_forwards_shared_endpoint_pool_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "model-endpoints.json"
    args = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "out"),
            "--model_endpoint_config",
            str(config_path),
            "--remote_text_model_workers",
            "64",
            "--remote_image_model_workers",
            "16",
        ]
    )

    assert args.model_endpoint_config == str(config_path)
    assert args.remote_text_model_workers == 64
    assert args.remote_image_model_workers == 16
    assert args.max_train_query_row_views_per_join == 5
    assert args.explicit_join_fallback_mode == "ratio"
    assert args.explicit_join_fallback_ratio == 0.2
    config = PipelineConfig.from_args(args)
    assert config.materialization_validation_workers == 3
    assert config.max_train_query_row_views_per_join == 5
    assert config.explicit_join_fallback_mode == "ratio"
    assert config.explicit_join_fallback_ratio == 0.2
    assert config.runtime_dir == config.work_dir / "runtime"
    assert STAGES == (
        "selection",
        "structural",
        "sampling",
        "pages",
        "asset_planning",
        "images",
        "models",
        "materialize",
    )


def test_model_adapter_workers_round_trip_to_runtime_args(tmp_path: Path) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--model_adapter_workers",
                "4",
            ]
        )
    )

    runtime_args = pipeline_module._runtime_args(config)

    assert config.model_adapter_workers == 4
    assert runtime_args.model_adapter_workers == 4


def test_balance_view_renders_model_adapter_progress() -> None:
    rendered = balance_progress.render(
        {
            "stage": "models",
            "detail": "adapter tasks",
            "completed_shards": 0,
            "total_shards": 2,
            "counters": {
                "model_adapter_shards_completed": 7,
                "model_adapter_shards_total": 19,
                "model_adapter_tasks_live": 123,
                "model_adapter_errors_live": 2,
            },
            "elapsed_seconds": 1.0,
            "disk": {},
        }
    )

    assert "adapter " in rendered
    assert "7/19" in rendered
    assert "model adapter tasks=123" in rendered


def test_remote_model_cli_options_reach_legacy_runtime_args(tmp_path: Path) -> None:
    secret = "explicit-secret-must-not-be-persisted"
    auto_check_api_config = tmp_path / "auto-check.json"
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                "--text_model_base_url",
                "http://127.0.0.1:18001/v1",
                "--text_model_name",
                "remote-text",
                "--text_model_api_key",
                secret,
                "--image_model_base_url",
                "http://127.0.0.1:18000/v1",
                "--image_model_name",
                "remote-image",
                "--image_model_api_key",
                secret,
                "--model_endpoint_ready_timeout_seconds",
                "45.5",
                "--model_timeout_seconds",
                "300",
                "--model_max_retries",
                "4",
                "--model_retry_sleep_seconds",
                "0.25",
                "--materialization_workers",
                "8",
                "--materialization_validation_workers",
                "4",
                "--auto_check_api_config_file",
                str(auto_check_api_config),
                "--max_train_query_row_views_per_join",
                "7",
            ]
        )
    )

    runtime_args = pipeline_module._runtime_args(config)

    assert runtime_args.text_model_base_url == "http://127.0.0.1:18001/v1"
    assert runtime_args.text_model_name == "remote-text"
    assert runtime_args.text_model_api_key == secret
    assert runtime_args.image_model_base_url == "http://127.0.0.1:18000/v1"
    assert runtime_args.image_model_name == "remote-image"
    assert runtime_args.image_model_api_key == secret
    assert runtime_args.model_endpoint_ready_timeout_seconds == 45.5
    assert runtime_args.model_timeout_seconds == 300.0
    assert runtime_args.model_max_retries == 4
    assert runtime_args.model_retry_sleep_seconds == 0.25
    assert runtime_args.materialization_workers == 8
    assert runtime_args.materialization_validation_workers == 4
    assert runtime_args.auto_check_api_config_file == str(auto_check_api_config)
    assert runtime_args.max_train_query_row_views_per_join == 7
    assert config.model_endpoint_ready_timeout_seconds == 45.5
    assert config.materialization_workers == 8
    assert config.materialization_validation_workers == 4
    assert config.auto_check_api_config_file == str(auto_check_api_config)
    assert config.max_train_query_row_views_per_join == 7


def test_train_row_view_cap_changes_only_materialization_fingerprint(
    tmp_path: Path,
) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
            ]
        )
    )
    changed = replace(config, max_train_query_row_views_per_join=2)

    for stage in STAGES:
        fingerprints_match = pipeline_module._stage_config_fingerprint(
            config, stage
        ) == pipeline_module._stage_config_fingerprint(changed, stage)
        assert fingerprints_match is (stage != "materialize")


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--model_endpoint_ready_timeout_seconds", "-0.1"),
        ("--model_endpoint_ready_timeout_seconds", "nan"),
        ("--model_endpoint_ready_timeout_seconds", "inf"),
        ("--model_endpoint_ready_timeout_seconds", "-inf"),
        ("--model_timeout_seconds", "0"),
        ("--model_timeout_seconds", "nan"),
        ("--model_timeout_seconds", "inf"),
        ("--model_timeout_seconds", "-inf"),
        ("--model_max_retries", "-1"),
        ("--model_retry_sleep_seconds", "-0.1"),
        ("--model_retry_sleep_seconds", "nan"),
        ("--model_retry_sleep_seconds", "inf"),
        ("--model_retry_sleep_seconds", "-inf"),
        ("--materialization_validation_workers", "0"),
        ("--materialization_validation_workers", "5"),
    ),
)
def test_remote_model_cli_options_reject_invalid_values(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
                option,
                value,
            ]
        )


def test_model_api_keys_do_not_change_stage_config_fingerprint(
    tmp_path: Path,
) -> None:
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(tmp_path / "input"),
                "--output_dir",
                str(tmp_path / "output"),
            ]
        )
    )
    with_secrets = replace(
        config,
        text_model_api_key="explicit-text-secret",
        image_model_api_key="explicit-image-secret",
    )

    assert pipeline_module._stage_config_fingerprint(
        config, "models"
    ) == pipeline_module._stage_config_fingerprint(with_secrets, "models")


def test_runtime_args_leave_api_key_environment_precedence_to_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "common-secret")
    monkeypatch.setenv("MMDD_TEXT_MODEL_API_KEY", "text-secret")
    monkeypatch.setenv("MMDD_IMAGE_MODEL_API_KEY", "image-secret")

    parsed = parse_args(
        [
            "--input_dir",
            str(tmp_path / "input"),
            "--output_dir",
            str(tmp_path / "output"),
            "--no_auto_check_secondary_openai",
        ]
    )
    runtime_args = pipeline_module._runtime_args(
        PipelineConfig.from_args(parsed)
    )
    extractor = pipeline_module.join_builder.LocalAttributeExtractor(runtime_args)

    assert parsed.text_model_api_key is None
    assert parsed.image_model_api_key is None
    assert runtime_args.text_model_api_key is None
    assert runtime_args.image_model_api_key is None
    assert extractor.text_model_api_key == "text-secret"
    assert extractor.image_model_api_key == "image-secret"


def test_pipeline_config_requires_separate_roots(tmp_path: Path) -> None:
    common = tmp_path / "same"
    with pytest.raises(ValueError, match="separate"):
        PipelineConfig.from_args(
            parse_args(
                [
                    "--input_dir",
                    str(tmp_path),
                    "--output_dir",
                    str(common),
                    "--work_dir",
                    str(common),
                ]
            )
        )


def test_runtime_state_must_stay_outside_input_cache_and_output(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--runtime_dir",
                str(tmp_path / "output" / "runtime"),
                "--dry_run",
            ]
        )
    )

    with pytest.raises(ValueError, match="equal work_dir/runtime"):
        run_pipeline(config)


def test_runtime_root_must_equal_work_runtime(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--runtime_dir",
                str(tmp_path / "other-runtime"),
                "--dry_run",
            ]
        )
    )

    with pytest.raises(ValueError, match="equal work_dir/runtime"):
        run_pipeline(config)


def test_runtime_files_must_be_children_of_the_runtime_root(
    tmp_path: Path,
) -> None:
    base = _full_pipeline_config(tmp_path)
    outside = base.work_dir / "page_jobs" / "start.json"
    config = replace(
        base,
        model_start_marker=outside,
        model_ready_marker=base.runtime_dir / "ready.json",
        model_text_done_marker=base.runtime_dir / "text-done.json",
        model_image_done_marker=base.runtime_dir / "image-done.json",
        dry_run=True,
    )

    with pytest.raises(ValueError, match="inside runtime_dir"):
        run_pipeline(config)


def test_runtime_files_inside_runtime_root_are_accepted(
    tmp_path: Path,
) -> None:
    base = _full_pipeline_config(tmp_path)
    runtime = base.runtime_dir
    config = replace(
        base,
        text_model_base_urls_file=str(runtime / "text-endpoints.json"),
        image_model_base_urls_file=str(runtime / "image-endpoints.json"),
        model_start_marker=runtime / "start.json",
        model_ready_marker=runtime / "ready.json",
        model_text_done_marker=runtime / "text-done.json",
        model_image_done_marker=runtime / "image-done.json",
        dry_run=True,
    )

    assert run_pipeline(config).status == "dry_run"


def test_disk_guard_checks_the_actual_target_filesystem(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    for root in (work, cache, output):
        root.mkdir()
    free_by_root = {work: 1_000, cache: 150, output: 500}

    def usage(path: Path) -> Any:
        return shutil._ntuple_diskusage(2_000, 2_000 - free_by_root[path], free_by_root[path])

    guard = DiskGuard(100, usage_fn=usage)
    guard(work / "nested" / "part.jsonl", 800)
    guard(output / "dataset.json", 400)
    with pytest.raises(DiskSpaceInsufficientError, match="estimated=60"):
        guard(cache / "images" / "asset.jpg", 60)


def test_structural_exact_count_database_recovers_after_guard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "structural"
    root.mkdir()
    page_refs = root / "page-refs.jsonl"
    records = [
        {"url_key": f"url-{index}", "page_url": f"https://e.test/{index}"}
        for index in range(24)
    ]
    page_refs.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        source_tables=root / "source-tables.jsonl",
        entities=root / "entities.jsonl",
        page_refs=page_refs,
        direct_image_refs=root / "direct-images.jsonl",
        structural_failures=root / "failures.jsonl",
        validated_selection=root / "validated-selection.jsonl",
        manifest=root / "manifest.json",
        entities_count=24,
        direct_image_references=0,
        page_references=24,
        tables=1,
        output_bytes=page_refs.stat().st_size,
    )
    database_path = root / "structural-counts.sqlite3"
    positive_calls = 0
    monkeypatch.setattr(
        pipeline_module.GuardedWriteTracker,
        "DEFAULT_INTERVAL_BYTES",
        1,
    )

    def interrupt(path: Path, estimated_bytes: int = 0) -> None:
        nonlocal positive_calls
        assert Path(path) == database_path
        if estimated_bytes > 0:
            positive_calls += 1
            if positive_calls == 2:
                raise DiskSpaceInsufficientError(
                    "synthetic structural reserve exhausted"
                )

    config = _full_pipeline_config(tmp_path / "config")
    with pytest.raises(DiskSpaceInsufficientError, match="structural reserve"):
        pipeline_module._structural_exact_counts(
            root,
            (result,),
            config,
            pre_write_guard=interrupt,
        )

    counts = pipeline_module._structural_exact_counts(
        root,
        (result,),
        config,
    )
    assert positive_calls == 2
    assert counts["unique_page_urls"] == len(records)


def test_dry_run_validates_without_writes_or_network(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    cache_dir = tmp_path / "cache"

    class NoNetwork:
        network_policy_fingerprint = "wdc-web-v1"

        def fetch_page(self, *_args: object, **_kwargs: object) -> dict:
            raise AssertionError("dry-run attempted network")

        def fetch_image(self, *_args: object, **_kwargs: object) -> dict:
            raise AssertionError("dry-run attempted network")

    result = run_pipeline(
        PipelineConfig.from_args(
            parse_args(
                [
                    "--input_dir",
                    str(input_dir),
                    "--output_dir",
                    str(output_dir),
                    "--work_dir",
                    str(work_dir),
                    "--cache_dir",
                    str(cache_dir),
                    "--dry_run",
                ]
            )
        ),
        page_transport=NoNetwork(),
        image_transport=NoNetwork(),
    )

    assert result.status == "dry_run"
    assert result.statistics_archives == 1
    assert not output_dir.exists()
    assert not work_dir.exists()
    assert not cache_dir.exists()
    assert not (work_dir / "progress.json").exists()
    assert not (work_dir / "page_jobs/network/network-manifest.json").exists()
    assert not (work_dir / "image_jobs/network/network-manifest.json").exists()


def test_cli_help_and_absolute_script_smoke() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_wdc200k_mm_joinability_dataset.py"),
            "--help",
        ],
        cwd=Path("/tmp"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--stop_after" in completed.stdout
    assert "--model_start_marker" in completed.stdout


def test_stop_after_structural_emits_exact_counts_without_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    _write_selected_table(input_dir)

    class NoNetwork:
        network_policy_fingerprint = "wdc-web-v1"

        def fetch_page(self, *_args: object, **_kwargs: object) -> dict:
            raise AssertionError("structural stop attempted network")

        def fetch_image(self, *_args: object, **_kwargs: object) -> dict:
            raise AssertionError("structural stop attempted network")

    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--max_source_tables",
                "1",
                "--class_max_tables",
                "1",
                "--min_free_disk_bytes",
                "0",
                "--selection_shard_tables",
                "1",
                "--stop_after",
                "structural",
                "--top100_policy",
                "all",
                "--include_rest",
                "--min_candidate_rows",
                "0",
                "--min_candidate_columns",
                "0",
            ]
        )
    )
    result = run_pipeline(
        config,
        page_transport=NoNetwork(),
        image_transport=NoNetwork(),
    )

    assert result.status == "stopped"
    assert result.stage == "structural"
    assert result.counters["selected_tables"] == 1
    assert result.counters["validated_tables"] == 1
    assert result.counters["entities"] == 2
    assert result.counters["page_references"] == 2
    assert result.counters["unique_page_urls"] == 1
    assert result.counters["direct_image_references"] == 3
    assert result.counters["page_request_upper_bound"] == 1
    assert result.counters["image_request_upper_bound"] == 12
    progress = json.loads(
        (config.work_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["stage"] == "structural"
    assert progress["counters"]["entities"] == 2
    assert progress["stage_telemetry"] == {}
    assert progress["disk"]["free_bytes"] >= 0
    stdout = capsys.readouterr().out
    assert "[wdc200k]" in stdout
    assert "work=" in stdout and "cache=" in stdout and "output=" in stdout
    assert "free_work=" in stdout and "free_cache=" in stdout
    assert "free_output=" in stdout and "reserve=" in stdout
    assert not config.cache_dir.exists()
    assert not config.output_dir.exists()
    assert not (
        config.work_dir / "page_jobs/network/network-manifest.json"
    ).exists()
    assert not (
        config.work_dir / "image_jobs/network/network-manifest.json"
    ).exists()


def test_from_stage_moves_named_and_downstream_without_deleting(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--from_stage",
                "pages",
                "--min_free_disk_bytes",
                "0",
            ]
        )
    )
    for stage in STAGES:
        marker = config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(stage, encoding="utf-8")
    (config.work_dir / "page_jobs").mkdir()
    (config.work_dir / "page_jobs" / "state").write_text(
        "page", encoding="utf-8"
    )
    (config.cache_dir / "page_cache").mkdir(parents=True)
    (config.cache_dir / "page_cache" / "keep").write_text(
        "cache", encoding="utf-8"
    )

    archived = invalidate_from_stage(config, "pages")

    assert (
        config.work_dir / "stage_manifests/pipeline-structural.json"
    ).is_file()
    assert not (
        config.work_dir / "stage_manifests/pipeline-pages.json"
    ).exists()
    destinations = {move.source: move.destination for move in archived.moves}
    assert destinations[
        config.work_dir / "stage_manifests/pipeline-pages.json"
    ].read_text() == "pages"
    assert destinations[config.work_dir / "page_jobs"].joinpath("state").read_text() == "page"
    assert (config.cache_dir / "page_cache/keep").read_text() == "cache"


def test_preflight_stops_before_disk_reserve_is_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    config = PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--min_free_disk_bytes",
                "100",
                "--dry_run",
            ]
        )
    )
    monkeypatch.setattr(
        pipeline_module.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(1_000, 950, 50),
    )

    with pytest.raises(DiskSpaceInsufficientError, match="free=50"):
        run_pipeline(config)

    assert not config.work_dir.exists()


class _PipelinePageTransport:
    network_policy_fingerprint = "wdc-web-v1"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def fetch_page(
        self,
        url: str,
        *,
        deadline_seconds: float,
        max_retries: int,
    ) -> dict[str, Any]:
        assert deadline_seconds == 8
        assert max_retries == 0
        self.calls += 1
        if self.fail:
            raise TimeoutError("synthetic page failure")
        return {
            "page_url": url,
            "final_url": url,
            "text": (
                "Alpha is in Texas and Beta is in Ohio. "
                "This page contains entity facts for the table. "
            )
            * 8,
            "image_urls": [],
        }


class _PipelineImageTransport:
    network_policy_fingerprint = "wdc-web-v1"
    max_retries = 0
    max_response_seconds = 8.0

    def __init__(self) -> None:
        self.calls = 0

    def download_image(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls += 1
        raise TimeoutError("synthetic image failure")


class _PipelineExtractor:
    def extract(
        self,
        _asset: dict[str, Any],
        entity: dict[str, Any],
        candidates: list[str],
    ) -> dict[str, Any]:
        assert "State" in candidates
        entity_name = str(entity["display_texts"][0])
        value = {
            "Alpha": "Texas",
            "Beta": "Ohio",
            "Gamma": "Utah",
            "Delta": "Maine",
            "Epsilon": "Iowa",
        }[entity_name]
        return {
            "attributes": [
                {
                    "name": "State",
                    "value": value,
                    "evidence": f"{entity_name} is in {value}.",
                    "connection_evidence": f"The text names {entity_name}.",
                }
            ]
        }


def _full_pipeline_config(
    tmp_path: Path,
    *,
    material_rows: int = 5,
) -> PipelineConfig:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir, rows=5)
    _write_selected_table(
        input_dir,
        rows_count=5,
        material_rows=material_rows,
    )
    return PipelineConfig.from_args(
        parse_args(
            [
                "--input_dir",
                str(input_dir),
                "--output_dir",
                str(tmp_path / "output"),
                "--work_dir",
                str(tmp_path / "work"),
                "--cache_dir",
                str(tmp_path / "cache"),
                "--max_source_tables",
                "1",
                "--class_max_tables",
                "1",
                "--selection_shard_tables",
                "1",
                "--records_per_shard",
                "2",
                "--min_free_disk_bytes",
                "0",
                "--progress_interval_seconds",
                "0.01",
            ]
        )
    )


def _url_snapshot(
    *,
    epoch: str,
    baseline: int,
    completed: int,
    total: int,
    elapsed: float,
    deadline: float = 8.0,
    effective_concurrency: int = 128,
    transport_overflow: int = 0,
    active_overflow: int = 0,
    commit_overflow: int = 0,
) -> UrlProgressSnapshot:
    completed_in_epoch = completed - baseline
    transport = [0] * 64
    commit = [0] * 32
    if completed_in_epoch:
        transport[0] = completed_in_epoch
        commit[0] = completed_in_epoch
    transport[-1] = max(transport[-1], transport_overflow)
    commit[-1] = max(commit[-1], commit_overflow)
    return UrlProgressSnapshot(
        execution_epoch=epoch,
        baseline_completed=baseline,
        completed_durable=completed,
        total=total,
        local_buffered_not_started=0,
        in_flight_jobs=0,
        physical_in_flight=0,
        finished_not_durable=0,
        unobserved_nonlocal=total - completed,
        deadline_seconds=deadline,
        effective_concurrency=effective_concurrency,
        epoch_elapsed_seconds=elapsed,
        transport_event_histogram=tuple(transport),
        active_censor_histogram=(0,) * 64,
        commit_event_histogram=tuple(commit),
        transport_overflow_events=transport_overflow,
        active_overflow_censors=active_overflow,
        commit_overflow_events=commit_overflow,
    )


def test_full_pipeline_uses_real_stage_contracts_and_resume_does_not_refetch(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    page_transport = _PipelinePageTransport()
    image_transport = _PipelineImageTransport()

    first = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert first.status == "complete"
    assert first.stage == "materialize"
    assert first.output_manifest == config.output_dir / "dataset_manifest.json"
    assert first.output_manifest.is_file()
    assert page_transport.calls == 1
    assert image_transport.calls == 3
    first_calls = (page_transport.calls, image_transport.calls)
    first_progress = json.loads(
        (config.work_dir / "progress.json").read_text(encoding="utf-8")
    )
    for stage, basis, total in (
        ("pages", "page_urls", 1),
        ("images", "image_urls", 3),
    ):
        telemetry = first_progress["stage_telemetry"][stage]
        assert telemetry["rate_basis"] == basis
        assert telemetry["samples"][-1]["completed_units"] == total
        assert telemetry["telemetry_schema_version"] == URL_TELEMETRY_V2
        assert all(
            sample["telemetry_schema_version"] == URL_TELEMETRY_V2
            for sample in telemetry["samples"]
        )
        assert telemetry["completed_at"] is not None

    first_network_telemetry = {}
    for stage, expected_attempts, basis in (
        ("pages", 1, "page_urls"),
        ("images", 3, "image_urls"),
    ):
        registry_path = (
            config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"
        )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        network_ref = next(
            item
            for item in registry["producer_manifests"]
            if json.loads(Path(item["path"]).read_text(encoding="utf-8"))[
                "stage"
            ]
            == "wdc200k_network_fetch"
        )
        network = json.loads(
            Path(network_ref["path"]).read_text(encoding="utf-8")
        )
        attempts = network["transport_attempts"]
        completion = network["url_completion"]
        assert attempts["stage"] == stage
        assert attempts["policy_fingerprint"] == network["policy_fingerprint"]
        assert attempts["transport_attempts"] == expected_attempts
        assert attempts["duplicate_physical_requests"] == 0
        assert attempts["terminal_replays"] == 0
        assert attempts["unfinished_transport_attempts"] == 0
        assert completion["rate_basis"] == basis
        assert completion["completed_units"] == network["counts"]["unique"]
        assert completion["total_units"] == network["counts"]["unique"]
        assert completion["completed_at"] is not None
        assert completion["telemetry_schema_version"] == URL_TELEMETRY_V2
        assert completion["estimator"] == {
            "transport_bins": 64,
            "active_censor_bins": 64,
            "commit_bins": 32,
            "maturity_numerator": 1,
            "maturity_denominator": 3,
            "deadline_seconds": 8.0,
        }
        assert registry["counters"][f"{stage[:-1]}_transport_attempts"] == (
            expected_attempts
        )
        assert registry["counters"][
            f"{stage[:-1]}_duplicate_physical_requests"
        ] == 0
        assert registry["counters"][
            f"{stage[:-1]}_eta_eligible_final_half_samples"
        ] == completion["eligible_final_half_samples"]
        first_network_telemetry[stage] = {
            "transport_attempts": attempts,
            "url_completion": completion,
            "registry_counters": registry["counters"],
            "network_bytes": Path(network_ref["path"]).read_bytes(),
            "registry_bytes": registry_path.read_bytes(),
        }

    adapter_manifest_path = (
        config.work_dir
        / "adapted_model_tasks"
        / "model-task-adapter-manifest.json"
    )
    original_adapter_manifest = adapter_manifest_path.read_bytes()
    legacy_adapter_manifest = json.loads(
        original_adapter_manifest
    )
    assert isinstance(
        legacy_adapter_manifest.pop("parameter_fingerprint"),
        str,
    )
    adapter_manifest_path.write_text(
        json.dumps(legacy_adapter_manifest),
        encoding="utf-8",
    )
    import wdc200k_models as model_module

    sampling_manifest = config.work_dir / "sampling" / "manifest.json"
    sampled_paths = model_module._sampling_manifest_entity_paths(
        sampling_manifest
    )
    expected_parameters, legacy_safe = (
        model_module._model_adapter_parameter_fingerprint(
            pipeline_module._runtime_args(config),
            sampled_entity_paths=sampled_paths,
            sampling_manifest=sampling_manifest,
        )
    )
    assert legacy_safe is True
    assert model_module._load_completed_adapted_model_tasks(
        output_root=adapter_manifest_path.parent,
        manifest_path=adapter_manifest_path,
        expected_input_fingerprint=legacy_adapter_manifest[
            "input_fingerprint"
        ],
        expected_parameter_fingerprint=expected_parameters,
        allow_legacy_parameter=legacy_safe,
    ).manifest_path == adapter_manifest_path
    with pytest.raises(ValueError, match="sampled entity path identity"):
        model_module._model_adapter_parameter_fingerprint(
            pipeline_module._runtime_args(config),
            sampled_entity_paths=(tmp_path / "foreign-sampled.jsonl",),
            sampling_manifest=sampling_manifest,
        )
    with pytest.raises(ValueError, match="sampled entity paths are required"):
        model_module._model_adapter_parameter_fingerprint(
            pipeline_module._runtime_args(config),
            sampled_entity_paths=None,
            sampling_manifest=sampling_manifest,
        )
    whitespace_args = pipeline_module._runtime_args(config)
    whitespace_args.text_model_name = f" {whitespace_args.text_model_name} "
    _whitespace_fingerprint, whitespace_legacy_safe = (
        model_module._model_adapter_parameter_fingerprint(
            whitespace_args,
            sampled_entity_paths=sampled_paths,
            sampling_manifest=sampling_manifest,
        )
    )
    assert whitespace_legacy_safe is False
    adapter_manifest_path.write_bytes(original_adapter_manifest)

    resumed = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert resumed.status == "complete"
    assert (page_transport.calls, image_transport.calls) == first_calls
    resumed_progress = json.loads(
        (config.work_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert resumed_progress["stage_telemetry"] == first_progress[
        "stage_telemetry"
    ]
    for stage in STAGES:
        registry_path = (
            config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"
        )
        registry = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
        assert registry["stage"] == stage
        assert registry["complete"] is True
        assert registry["producer_type"]
        if stage in first_network_telemetry:
            network_ref = next(
                item
                for item in registry["producer_manifests"]
                if json.loads(Path(item["path"]).read_text(encoding="utf-8"))[
                    "stage"
                ]
                == "wdc200k_network_fetch"
            )
            network = json.loads(
                Path(network_ref["path"]).read_text(encoding="utf-8")
            )
            assert Path(network_ref["path"]).read_bytes() == (
                first_network_telemetry[stage]["network_bytes"]
            )
            assert registry_path.read_bytes() == first_network_telemetry[stage][
                "registry_bytes"
            ]
            assert {
                "transport_attempts": network["transport_attempts"],
                "url_completion": network["url_completion"],
                "registry_counters": registry["counters"],
            } == {
                key: first_network_telemetry[stage][key]
                for key in (
                    "transport_attempts",
                    "url_completion",
                    "registry_counters",
                )
            }


def test_incomplete_models_fast_resume_skips_completed_upstream_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="images")
    page_transport = _PipelinePageTransport()
    image_transport = _PipelineImageTransport()
    first = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
    )
    assert first.stage == "images"
    first_calls = (page_transport.calls, image_transport.calls)

    adapter_manifest = (
        config.work_dir
        / "adapted_model_tasks"
        / "model-task-adapter-manifest.json"
    )
    adapter_manifest.parent.mkdir(parents=True)
    adapter_manifest.write_text(
        json.dumps(
            {
                "stage": "wdc200k_model_task_adapter",
                "complete": True,
            }
        ),
        encoding="utf-8",
    )
    model_database = config.cache_dir / "model_cache" / "jobs.sqlite3"
    model_database.parent.mkdir(parents=True)
    sqlite3.connect(model_database).close()

    fast_calls: list[PipelineConfig] = []

    def fake_fast_resume(
        resumed_config: PipelineConfig,
        **_kwargs: Any,
    ) -> pipeline_module.PipelineResult:
        fast_calls.append(resumed_config)
        return pipeline_module.PipelineResult(
            status="stopped",
            stage="models",
            statistics_archives=1,
            counters={},
        )

    monkeypatch.setattr(
        pipeline_module,
        "_run_fast_model_resume",
        fake_fast_resume,
    )
    resumed = run_pipeline(
        replace(config, stop_after=None),
        page_transport=page_transport,
        image_transport=image_transport,
    )

    assert resumed.stage == "models"
    assert len(fast_calls) == 1
    assert (page_transport.calls, image_transport.calls) == first_calls


def test_materialization_certificate_bypasses_strict_stage_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )
    assert first.status == "complete"
    (
        config.work_dir
        / "stage_manifests"
        / "pipeline-materialize.json"
    ).unlink()
    fast_calls: list[MaterializationInputs] = []

    def reject_strict_replay(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("strict registry replay was repeated")

    def fake_fast_resume(
        _config: PipelineConfig,
        *,
        inputs: MaterializationInputs,
        **_kwargs: Any,
    ) -> pipeline_module.PipelineResult:
        fast_calls.append(inputs)
        return pipeline_module.PipelineResult(
            status="complete",
            stage="materialize",
            statistics_archives=1,
            counters={},
            output_manifest=config.output_dir / "dataset_manifest.json",
        )

    monkeypatch.setattr(
        pipeline_module,
        "_validate_existing_registry_chain",
        reject_strict_replay,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_fast_materialization_resume",
        fake_fast_resume,
    )

    resumed = run_pipeline(config, extractor=_PipelineExtractor())

    assert resumed.status == "complete"
    assert len(fast_calls) == 1
    assert fast_calls[0].upstream_stage_registry == (
        config.work_dir / "stage_manifests" / "pipeline-models.json"
    )


def test_pipeline_resume_after_sampling_does_not_require_replaced_full_shards(
    tmp_path: Path,
) -> None:
    initial = replace(_full_pipeline_config(tmp_path), stop_after="sampling")
    assert run_pipeline(initial).stage == "sampling"
    for directory in ("entities", "page_refs", "direct_image_refs"):
        for path in (initial.work_dir / "structural" / directory).glob("*.jsonl"):
            path.unlink()

    transport = _PipelinePageTransport()
    resumed = run_pipeline(
        replace(initial, stop_after="pages", from_stage="pages"),
        page_transport=transport,
    )

    assert resumed.stage == "pages"
    assert transport.calls == 1


def test_completed_sampling_resume_does_not_iterate_source_tables_to_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="sampling")
    assert run_pipeline(config).stage == "sampling"
    original_iter = pipeline_module._iter_jsonl

    def reject_source_table_scan(path: Path):
        if Path(path).parent.name == "source_tables":
            raise AssertionError("resume scanned a complete source-table shard")
        yield from original_iter(path)

    monkeypatch.setattr(
        pipeline_module,
        "_iter_jsonl",
        reject_source_table_scan,
    )

    resumed = run_pipeline(config)

    assert resumed.stage == "sampling"


def test_fast_structural_resume_accepts_compacted_derivatives(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="structural")
    assert run_pipeline(config).stage == "structural"
    for directory in (
        "entities",
        "page_refs",
        "direct_image_refs",
    ):
        for path in (config.work_dir / "structural" / directory).glob(
            "*.jsonl"
        ):
            path.unlink()
    for path in (config.work_dir / "structural/selection").glob(
        "validated-[0-9]*.jsonl"
    ):
        path.unlink()

    pipeline_module._validate_fast_structural_resume_chain(
        config,
        pipeline_module._statistics_archives(config.input_dir),
    )


def test_pipeline_resumes_incomplete_sampling_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="sampling")
    original_commit = pipeline_module.AtomicJsonlShard.commit
    interrupted = False

    def interrupt(self):
        nonlocal interrupted
        if not interrupted and self.path.parent.name == "sampled_page_refs":
            interrupted = True
            raise RuntimeError("sampling commit interrupted")
        return original_commit(self)

    monkeypatch.setattr(pipeline_module.AtomicJsonlShard, "commit", interrupt)
    with pytest.raises(RuntimeError, match="sampling commit"):
        run_pipeline(config)
    monkeypatch.setattr(pipeline_module.AtomicJsonlShard, "commit", original_commit)

    resumed = run_pipeline(config)

    assert resumed.stage == "sampling"


def test_partial_sampling_manifest_matches_current_expansion(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    manifest_root = config.work_dir / "structural" / "stage_manifests"
    manifest_root.mkdir(parents=True)
    structural_manifest = manifest_root / "structural-00000.json"
    structural_manifest.write_text(
        json.dumps({"stage": "wdc200k_structural"}) + "\n",
        encoding="utf-8",
    )
    registry_path = config.work_dir / "stage_manifests/pipeline-structural.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "stage": "structural",
                "producer_type": "wdc200k-structural-barrier",
                "producer_manifests": [
                    {"path": str(structural_manifest), "sha256": "unused"}
                ],
                "upstream_identity": "selection",
                "config_fingerprint": "structural",
                "counters": {},
                "complete": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expansion = pipeline_module.SamplingExpansionState(
        round_index=1,
        path=config.work_dir / "sampling_expansion/round-00001.json",
        limits={"table-1": 12},
    )
    fingerprint = pipeline_module.sampling_stage_fingerprint(
        (structural_manifest,),
        pipeline_module._sampling_policy(config),
        per_table_entity_limits=expansion.limits,
    )
    sampling_manifest = config.work_dir / "sampling/manifest.json"
    sampling_manifest.parent.mkdir(parents=True)
    sampling_manifest.write_text(
        json.dumps(
            {
                "stage": fingerprint.stage,
                "input_fingerprint": fingerprint.input_fingerprint,
                "parameter_fingerprint": fingerprint.parameter_fingerprint,
                "schema_version": fingerprint.schema_version,
                "completed_shards": [],
                "complete": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert pipeline_module._partial_sampling_matches_expansion(
        config, expansion
    )
    assert not pipeline_module._partial_sampling_matches_expansion(
        config,
        replace(expansion, limits={"table-1": 20}),
    )


def test_run_pipeline_preserves_matching_partial_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    expansion = pipeline_module.SamplingExpansionState(
        round_index=1,
        path=config.work_dir / "sampling_expansion/round-00001.json",
        limits={"table-1": 12},
    )
    effective_configs: list[PipelineConfig] = []

    monkeypatch.setattr(
        pipeline_module, "_promote_legacy_model_cache", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline_module, "_load_active_recovery_selection", lambda _config: None
    )
    monkeypatch.setattr(
        pipeline_module, "_load_sampling_expansion_state", lambda _config: expansion
    )
    monkeypatch.setattr(
        pipeline_module, "_sampling_registry_matches_expansion", lambda _config: False
    )
    monkeypatch.setattr(
        pipeline_module,
        "_partial_sampling_matches_expansion",
        lambda _config, _expansion: True,
    )

    def stop_after_capture(effective: PipelineConfig, **_kwargs):
        effective_configs.append(effective)
        return pipeline_module.PipelineResult(
            status="stopped",
            stage="sampling",
            statistics_archives=1,
            counters={},
        )

    monkeypatch.setattr(pipeline_module, "_run_pipeline_once", stop_after_capture)

    result = run_pipeline(config)

    assert result.stage == "sampling"
    assert len(effective_configs) == 1
    assert effective_configs[0].sampling_expansion_round_index == 1
    assert effective_configs[0].from_stage is None


def test_structural_registry_validates_compact_authority_once_for_many_refs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="sampling")
    run_pipeline(config)
    registry_path = config.work_dir / "stage_manifests/pipeline-structural.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["producer_manifests"].append(payload["producer_manifests"][0])
    registry_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls = 0
    original = pipeline_module.validate_sampling_source_authority

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_module,
        "validate_sampling_source_authority",
        counted,
    )
    pipeline_module._validate_stage_registry(
        config,
        "structural",
        expected_upstream_identity=payload["upstream_identity"],
    )
    assert calls == 1


def test_models_and_materialize_resume_after_full_structural_shards_removed(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )
    for directory in ("entities", "page_refs", "direct_image_refs"):
        for path in (config.work_dir / "structural" / directory).glob("*.jsonl"):
            path.unlink()

    resumed = run_pipeline(
        replace(config, from_stage="models"),
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )

    assert resumed.status == "complete"
    assert resumed.counters["unique_wiki_entities"] == 5


def test_prefilter_rejected_table_still_materializes_as_complete_raw_table(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path, material_rows=2)

    result = run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )

    assert result.status == "complete"
    assert result.counters["prefilter_rejected_tables"] == 1
    assert result.counters["source_tables"] == 1
    assert result.counters["data_lake_tables"] == 1
    source_path = next((config.output_dir / "source_tables").glob("*.jsonl"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert len(source["rows"]) == 5


def test_resume_rejects_tampered_page_attempt_authority(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    outcomes = config.cache_dir / "page_cache/outcomes.sqlite3"
    with sqlite3.connect(outcomes) as connection:
        connection.execute(
            "UPDATE transport_attempts SET final_status = 'exception'"
        )
        connection.commit()

    with pytest.raises(
        ValueError,
        match="transport attempt authority mismatch",
    ):
        run_pipeline(config, page_transport=_PipelinePageTransport())


def test_resume_rejects_tampered_attempt_registry_counter(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    registry_path = config.work_dir / "stage_manifests/pipeline-pages.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["counters"]["page_transport_attempts"] += 1
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="telemetry counters mismatch"):
        run_pipeline(config, page_transport=_PipelinePageTransport())


def test_resume_rejects_stale_image_attempt_policy_even_with_updated_checksum(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="images")
    run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
    )
    registry_path = config.work_dir / "stage_manifests/pipeline-images.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    network_ref = next(
        item
        for item in registry["producer_manifests"]
        if json.loads(Path(item["path"]).read_text(encoding="utf-8"))[
            "stage"
        ]
        == "wdc200k_network_fetch"
    )
    network_path = Path(network_ref["path"])
    network = json.loads(network_path.read_text(encoding="utf-8"))
    network["transport_attempts"]["policy_fingerprint"] = "stale-policy"
    network_path.write_text(
        json.dumps(network, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    network_ref["sha256"] = pipeline_module._sha256_path(network_path)
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="transport attempt policy mismatch"):
        run_pipeline(
            config,
            page_transport=_PipelinePageTransport(),
            image_transport=_PipelineImageTransport(),
        )


def _rewrite_network_registry(
    config: PipelineConfig,
    stage: str,
    mutate: Any,
) -> None:
    registry_path = (
        config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    network_ref = next(
        item
        for item in registry["producer_manifests"]
        if json.loads(Path(item["path"]).read_text(encoding="utf-8"))[
            "stage"
        ]
        == "wdc200k_network_fetch"
    )
    network_path = Path(network_ref["path"])
    network = json.loads(network_path.read_text(encoding="utf-8"))
    mutate(network, registry)
    network_path.write_text(
        json.dumps(network, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    network_ref["sha256"] = pipeline_module._sha256_path(network_path)
    registry_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_existing_network_registry(
    config: PipelineConfig,
    stage: str,
) -> None:
    registry_path = (
        config.work_dir / "stage_manifests" / f"pipeline-{stage}.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    pipeline_module._validate_stage_registry(
        config,
        stage,
        expected_upstream_identity=registry["upstream_identity"],
    )


def test_registry_rejects_empty_forged_attempt_scope(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    outcomes = config.cache_dir / "page_cache/outcomes.sqlite3"
    jobs = config.work_dir / "page_jobs/jobs.sqlite3"
    manifest_path = config.work_dir / "page_jobs/network/network-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hidden_summary = PageOutcomeStore(outcomes).transport_attempt_summary(
        manifest["policy_fingerprint"],
        job_store_path=jobs,
        job_kind="hidden-kind",
        job_id_prefix="hidden-prefix",
    )

    def forge(network: dict[str, Any], registry: dict[str, Any]) -> None:
        network["transport_attempts"] = hidden_summary
        network["transport_attempt_authority"]["job_kind"] = "hidden-kind"
        network["transport_attempt_authority"][
            "job_id_prefix"
        ] = "hidden-prefix"
        registry["counters"].update(
            pipeline_module._network_telemetry_counters(
                "pages",
                hidden_summary,
                network["url_completion"],
            )
        )

    _rewrite_network_registry(config, "pages", forge)

    with pytest.raises(ValueError, match="transport attempt scope mismatch"):
        _validate_existing_network_registry(config, "pages")


def test_registry_binds_url_completion_to_progress_state(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    progress_path = config.work_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["stage_telemetry"]["pages"]["completed_at"] += 100.0
    progress_path.write_text(
        json.dumps(progress, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="progress URL completion mismatch"):
        _validate_existing_network_registry(config, "pages")


def test_registry_accepts_completed_v1_progress_and_manifest_authority(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    progress_path = config.work_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    legacy = {
        "rate_basis": "page_urls",
        "completed_units": 1,
        "total_units": 1,
        "completed_at": 11.0,
        "eligible_final_half_samples": 0,
        "excluded_final_half_samples": 1,
        "max_symmetric_eta_factor": None,
        "samples": [
            {
                "timestamp": 10.0,
                "completed_units": 0,
                "total_units": 1,
                "rate": 0.0,
                "rolling_rate": 0.0,
                "predicted_remaining_seconds": None,
            },
            {
                "timestamp": 11.0,
                "completed_units": 1,
                "total_units": 1,
                "rate": 1.0,
                "rolling_rate": 1.0,
                "predicted_remaining_seconds": 0.0,
            },
        ],
    }
    progress["stage_telemetry"]["pages"] = legacy
    progress_path.write_text(
        json.dumps(progress, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def downgrade(network: dict[str, Any], registry: dict[str, Any]) -> None:
        network["url_completion"] = {
            key: legacy[key]
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
        registry["counters"]["page_eta_eligible_final_half_samples"] = 0
        registry["counters"]["page_eta_excluded_final_half_samples"] = 1

    _rewrite_network_registry(config, "pages", downgrade)
    _validate_existing_network_registry(config, "pages")

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["stage_telemetry"]["pages"][
        "excluded_final_half_samples"
    ] = 0
    progress_path.write_text(
        json.dumps(progress, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="progress URL completion mismatch"):
        _validate_existing_network_registry(config, "pages")


def test_registry_recomputes_eta_summary_before_accepting_factor(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    progress_path = config.work_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_telemetry = progress["stage_telemetry"]["pages"]
    progress_telemetry["eligible_final_half_samples"] = 1
    progress_telemetry["excluded_final_half_samples"] = 0
    progress_telemetry["max_symmetric_eta_factor"] = None
    progress_path.write_text(
        json.dumps(progress, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    def forge(network: dict[str, Any], registry: dict[str, Any]) -> None:
        completion = network["url_completion"]
        completion["eligible_final_half_samples"] = 1
        completion["excluded_final_half_samples"] = 0
        completion["max_symmetric_eta_factor"] = None
        registry["counters"][
            "page_eta_eligible_final_half_samples"
        ] = 1
        registry["counters"][
            "page_eta_excluded_final_half_samples"
        ] = 0

    _rewrite_network_registry(config, "pages", forge)

    with pytest.raises(ValueError, match="progress ETA summary mismatch"):
        _validate_existing_network_registry(config, "pages")


def test_registry_validation_is_read_only_for_transport_databases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )
    tracked_roots = (
        config.cache_dir / "page_cache",
        config.cache_dir / "image_cache",
        config.work_dir / "page_jobs",
        config.work_dir / "image_jobs",
    )

    def snapshot() -> dict[str, tuple[int, int, str]]:
        return {
            str(path): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                pipeline_module._sha256_path(path),
            )
            for root in tracked_roots
            for path in root.rglob("*")
            if path.is_file()
        }

    before = snapshot()
    constructed: list[str] = []
    page_init = PageOutcomeStore.__init__
    image_init = ImageOutcomeStore.__init__

    def track_page(self: Any, *args: Any, **kwargs: Any) -> None:
        constructed.append("pages")
        page_init(self, *args, **kwargs)

    def track_image(self: Any, *args: Any, **kwargs: Any) -> None:
        constructed.append("images")
        image_init(self, *args, **kwargs)

    monkeypatch.setattr(PageOutcomeStore, "__init__", track_page)
    monkeypatch.setattr(
        ImageOutcomeStore,
        "__init__",
        track_image,
    )
    _validate_existing_network_registry(config, "pages")
    _validate_existing_network_registry(config, "images")

    assert constructed == []
    assert snapshot() == before


def test_resume_repairs_corrupt_network_snapshot_from_authoritative_store(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    page_transport = _PipelinePageTransport()
    image_transport = _PipelineImageTransport()
    run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )
    manifest_path = config.work_dir / "page_jobs/network/network-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outcome_path = manifest_path.parent / manifest["completed_shards"][0]["path"]
    expected = outcome_path.read_bytes()
    outcome_path.write_text("corrupt\n", encoding="utf-8")
    calls = (page_transport.calls, image_transport.calls)

    resumed = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert resumed.status == "complete"
    assert outcome_path.read_bytes() == expected
    assert (page_transport.calls, image_transport.calls) == calls


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_page_outcome_survives_interruption_before_job_commit(
    tmp_path: Path,
    terminal_failure: bool,
) -> None:
    config = _full_pipeline_config(tmp_path)
    page_transport = _PipelinePageTransport(fail=terminal_failure)
    image_transport = _PipelineImageTransport()

    class InjectedInterrupt(BaseException):
        pass

    def interrupt_after_outcome(_record: dict[str, Any]) -> None:
        raise InjectedInterrupt()

    with pytest.raises(InjectedInterrupt):
        run_pipeline(
            config,
            page_transport=page_transport,
            image_transport=image_transport,
            extractor=_PipelineExtractor(),
            after_page_cache_write=interrupt_after_outcome,
        )
    assert page_transport.calls == 1

    resumed = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert resumed.status == "complete"
    assert page_transport.calls == 1


def test_from_stage_rejects_tampered_upstream_before_moving_state(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )
    structural_registry = json.loads(
        (
            config.work_dir / "stage_manifests/pipeline-structural.json"
        ).read_text(encoding="utf-8")
    )
    producer = Path(structural_registry["producer_manifests"][0]["path"])
    producer.write_text(
        producer.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    refresh = replace(config, from_stage="pages", stop_after="pages")

    with pytest.raises(ValueError, match="checksum mismatch"):
        run_pipeline(
            refresh,
            page_transport=_PipelinePageTransport(),
            image_transport=_PipelineImageTransport(),
            extractor=_PipelineExtractor(),
        )

    assert (config.work_dir / "page_jobs").is_dir()
    assert not (config.work_dir / ".archive-transactions").exists()


def test_from_stage_rejects_corrupt_upstream_shard_before_moving_state(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    structural_registry = json.loads(
        (
            config.work_dir / "stage_manifests/pipeline-structural.json"
        ).read_text(encoding="utf-8")
    )
    producer = Path(structural_registry["producer_manifests"][0]["path"])
    manifest = json.loads(producer.read_text(encoding="utf-8"))
    source_shard = next(
        item
        for item in manifest["completed_shards"]
        if item["path"].startswith("source_tables/")
    )
    source_path = producer.parent.parent / source_shard["path"]
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="producer shard checksum"):
        run_pipeline(
            replace(config, from_stage="pages"),
            page_transport=_PipelinePageTransport(),
        )

    assert (config.work_dir / "page_jobs").is_dir()
    assert not (config.work_dir / ".archive-transactions").exists()


def test_stop_after_selection_does_not_expand_structural_tables(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="selection")

    result = run_pipeline(config)

    assert result.status == "stopped"
    assert result.stage == "selection"
    assert not (config.work_dir / "structural").exists()


def test_fresh_no_resume_page_telemetry_validates_from_progress_state(
    tmp_path: Path,
) -> None:
    config = replace(
        _full_pipeline_config(tmp_path),
        resume=False,
        stop_after="pages",
    )

    result = run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
    )

    assert result.status == "stopped"
    assert result.stage == "pages"
    assert result.counters["page_url_completed"] == (
        result.counters["page_url_total"]
    )


def test_progress_stdout_is_bounded_between_periodic_snapshots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(
        _full_pipeline_config(tmp_path),
        stop_after="structural",
        progress_interval_seconds=60.0,
    )

    run_pipeline(config)

    progress_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[wdc200k]")
    ]
    assert len(progress_lines) <= 3


def test_model_progress_non_tty_is_compact_and_does_not_change_json_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock[0])
    reporter.update(
        stage="models",
        counters={"enormous_unrelated_counter_name": 123456789},
    )
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot(
            modality="text",
            total=100,
            success=10,
            terminal=5,
            leased=2,
            pending=83,
        )
    )
    clock[0] = 5.0
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot(
            modality="text",
            total=100,
            success=18,
            terminal=7,
            leased=2,
            pending=73,
        )
    )

    reporter.publish()

    output = capsys.readouterr().out
    assert "models:text" in output
    assert "25/100" in output
    assert "success=18" in output
    assert "terminal=7" in output
    assert "leased=2" in output
    assert "pending=73" in output
    assert "2.00 job/s" in output
    assert "ETA 00:38" in output
    assert "counters=" not in output
    assert "enormous_unrelated_counter_name" not in output
    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert "model_progress" not in payload
    assert payload["counters"]["enormous_unrelated_counter_name"] == 123456789


def test_model_progress_tty_uses_two_native_tqdm_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    bars: list[Any] = []

    class FakeTqdm:
        def __init__(self, **kwargs: Any) -> None:
            self.total = kwargs["total"]
            self.n = kwargs["initial"]
            self.desc = kwargs["desc"]
            self.position = kwargs["position"]
            self.postfix: dict[str, int] = {}
            self.refreshes = 0
            self.closed = False
            bars.append(self)

        def set_postfix(
            self,
            *,
            refresh: bool,
            **values: int,
        ) -> None:
            assert refresh is False
            self.postfix = values

        def update(self, amount: int) -> None:
            self.n += amount

        def refresh(self) -> None:
            self.refreshes += 1

        def close(self) -> None:
            self.closed = True

    stream = TtyBuffer()
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    monkeypatch.setattr(pipeline_module, "tqdm", FakeTqdm)
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(stage="models")
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot(
            modality="text",
            total=100,
            success=20,
            terminal=1,
            leased=3,
            pending=76,
        )
    )
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot(
            modality="image",
            total=10,
            success=3,
            terminal=1,
            leased=2,
            pending=4,
        )
    )

    reporter.publish()
    reporter.publish()

    assert len(bars) == 2
    assert [(bar.desc, bar.position) for bar in bars] == [
        ("WDC text", 0),
        ("WDC image", 1),
    ]
    assert [(bar.n, bar.total) for bar in bars] == [(21, 100), (4, 10)]
    assert bars[0].postfix == {
        "success": 20,
        "terminal": 1,
        "leased": 3,
        "pending": 76,
    }
    assert bars[1].postfix == {
        "success": 3,
        "terminal": 1,
        "leased": 2,
        "pending": 4,
    }
    assert all(bar.refreshes == 2 for bar in bars)


def test_non_model_progress_tty_uses_native_tqdm_bar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    bars: list[Any] = []

    class FakeTqdm:
        def __init__(self, **kwargs: Any) -> None:
            self.total = kwargs["total"]
            self.n = kwargs["initial"]
            self.desc = kwargs["desc"]
            self.unit = kwargs["unit"]
            self.refreshes = 0
            self.closed = False
            bars.append(self)

        def refresh(self) -> None:
            self.refreshes += 1

        def update(self, amount: int) -> None:
            self.n += amount

        def close(self) -> None:
            self.closed = True

    stream = TtyBuffer()
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    monkeypatch.setattr(pipeline_module, "tqdm", FakeTqdm)
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(
        stage="structural",
        completed_shards=2,
        total_shards=10,
    )

    reporter.publish()
    reporter.update(completed_shards=3)
    reporter.publish()

    assert len(bars) == 1
    assert bars[0].desc == "WDC structural"
    assert bars[0].unit == "table"
    assert (bars[0].n, bars[0].total) == (3, 10)
    assert bars[0].refreshes == 2
    assert "[wdc200k]" not in stream.getvalue()


def test_non_model_progress_publishes_detail_to_json_and_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(
        stage="selection",
        detail="rank candidates",
        completed_shards=4,
        total_shards=7,
    )

    reporter.publish()

    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert payload["detail"] == "rank candidates"
    assert payload["completed_shards"] == 4
    assert payload["total_shards"] == 7
    assert "progress=4/7" in stream.getvalue()
    assert "detail=rank candidates" in stream.getvalue()


def test_non_model_progress_tty_displays_detail_as_postfix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    bars: list[Any] = []

    class FakeTqdm:
        def __init__(self, **kwargs: Any) -> None:
            self.total = kwargs["total"]
            self.n = kwargs["initial"]
            self.unit = kwargs["unit"]
            self.postfix = ""
            bars.append(self)

        def update(self, amount: int) -> None:
            self.n += amount

        def set_postfix_str(self, value: str, *, refresh: bool) -> None:
            assert refresh is False
            self.postfix = value

        def refresh(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline_module.sys, "stdout", TtyBuffer())
    monkeypatch.setattr(pipeline_module, "tqdm", FakeTqdm)
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(
        stage="asset_planning",
        detail="plan entity assets",
        completed_shards=1_000,
        total_shards=8_000,
    )

    reporter.publish()

    assert len(bars) == 1
    assert bars[0].unit == "record"
    assert (bars[0].n, bars[0].total) == (1_000, 8_000)
    assert bars[0].postfix == "plan entity assets"


def test_url_progress_tty_replaces_shard_bar_with_url_bar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TtyBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    bars: list[Any] = []

    class FakeTqdm:
        def __init__(self, **kwargs: Any) -> None:
            self.total = kwargs["total"]
            self.n = kwargs["initial"]
            self.desc = kwargs["desc"]
            self.unit = kwargs["unit"]
            self.closed = False
            bars.append(self)

        def refresh(self) -> None:
            pass

        def update(self, amount: int) -> None:
            self.n += amount

        def close(self) -> None:
            self.closed = True

    stream = TtyBuffer()
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    monkeypatch.setattr(pipeline_module, "tqdm", FakeTqdm)
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(stage="pages", completed_shards=0, total_shards=1)
    reporter.publish()
    reporter.update(
        url_snapshot=_url_snapshot(
            epoch="test",
            baseline=0,
            completed=0,
            total=20,
            elapsed=0.0,
        )
    )
    reporter.publish()
    reporter.update(
        url_snapshot=_url_snapshot(
            epoch="test",
            baseline=0,
            completed=6,
            total=20,
            elapsed=1.0,
        )
    )
    reporter.publish()

    assert len(bars) == 2
    assert bars[0].closed is True
    assert bars[1].desc == "WDC pages"
    assert bars[1].unit == "url"
    assert (bars[1].n, bars[1].total) == (6, 20)


def test_model_progress_non_tty_throttles_console_but_not_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock[0])
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(stage="models")
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot("text", 10, 0, 0, 0, 10)
    )
    reporter.publish()
    first_mtime = reporter.path.stat().st_mtime_ns
    clock[0] = 5.0
    reporter.publish()
    second_mtime = reporter.path.stat().st_mtime_ns
    clock[0] = 10.0
    reporter.publish()

    assert len(stream.getvalue().splitlines()) == 1
    assert second_mtime >= first_mtime
    assert "\r" not in stream.getvalue()
    assert "\x1b" not in stream.getvalue()


def test_model_progress_tracks_modalities_independently_and_stall_hides_eta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.sys, "stdout", stream)
    monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock[0])
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(stage="models")
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot("text", 100, 10, 0, 0, 90)
    )
    clock[0] = 5.0
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot("text", 100, 20, 0, 0, 80)
    )
    reporter.publish()
    assert "2.00 job/s" in stream.getvalue()

    clock[0] = 6.0
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot("image", 20, 0, 0, 0, 20)
    )
    reporter.publish()
    line = stream.getvalue().splitlines()[-1]
    assert "models:text" in line
    assert "models:image" in line
    assert "models:image 0/20 [0.00 job/s, ETA --:--]" in line

    clock[0] = 11.0
    reporter.update_model_progress(
        pipeline_module.ModelProgressSnapshot("image", 20, 5, 0, 0, 15)
    )
    reporter.publish()
    clock[0] = 72.0
    reporter.publish()
    line = stream.getvalue().splitlines()[-1]
    assert "models:image 5/20 [0.00 job/s, ETA --:--]" in line


def test_progress_rolling_rate_uses_a_fixed_time_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    reporter = ProgressReporter(config)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    reporter._state.stage = "test"
    reporter._state.started_at = 0.0
    reporter._state.stage_started_at = 0.0
    reporter._rolling_samples.clear()

    clock[0] = 10.0
    reporter.update(completed_shards=90, total_shards=200)
    reporter._snapshot()
    clock[0] = 70.0
    reporter.update(completed_shards=100)
    snapshot = reporter._snapshot()

    assert snapshot["rates"]["shards_per_second"] == pytest.approx(100 / 70)
    assert snapshot["rates"]["rolling_shards_per_second"] == pytest.approx(10 / 60)
    assert snapshot["rates"]["rolling_shards_per_second"] != snapshot["rates"]["shards_per_second"]


def test_progress_v2_maps_two_epochs_to_rollback_safe_logical_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="epoch-1", baseline=5, completed=5, total=10, elapsed=0.0
        ),
    )
    first.update(
        url_snapshot=_url_snapshot(
            epoch="epoch-1", baseline=5, completed=7, total=10, elapsed=2.0
        )
    )
    first.publish()
    original_samples = json.loads(first.path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ]["pages"]["samples"]

    clock[0] = 50.0
    resumed = ProgressReporter(config)
    resumed.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="epoch-2", baseline=7, completed=7, total=10, elapsed=0.0
        ),
    )
    clock[0] = 10.0
    resumed.update(
        url_snapshot=_url_snapshot(
            epoch="epoch-2", baseline=7, completed=10, total=10, elapsed=3.0
        )
    )
    resumed.publish()
    payload = json.loads(resumed.path.read_text(encoding="utf-8"))
    samples = payload["stage_telemetry"]["pages"]["samples"]

    assert samples[:2] == original_samples
    assert [sample["execution_epoch"] for sample in samples] == [
        "epoch-1",
        "epoch-1",
        "epoch-2",
        "epoch-2",
    ]
    assert samples[0]["epoch_elapsed_seconds"] == 0.0
    assert samples[2]["epoch_elapsed_seconds"] == 0.0
    assert samples[0]["timestamp"] == samples[0]["baseline_timestamp"]
    assert samples[1]["timestamp"] == samples[0]["baseline_timestamp"] + 2.0
    assert samples[2]["baseline_timestamp"] >= samples[1]["timestamp"]
    assert samples[3]["timestamp"] == samples[2]["baseline_timestamp"] + 3.0
    assert payload["stage_telemetry"]["pages"]["completed_at"] == samples[-1][
        "timestamp"
    ]
    ProgressReporter(config)


@pytest.mark.parametrize("stage", ["pages", "images"])
def test_progress_v2_replaces_zero_progress_telemetry_when_scope_changes(
    tmp_path: Path,
    stage: str,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = ProgressReporter(config)
    first.update(
        stage=stage,
        url_snapshot=_url_snapshot(
            epoch="old-full-scope",
            baseline=0,
            completed=0,
            total=27_795_316,
            elapsed=0.0,
        ),
    )
    first.publish()

    resumed = ProgressReporter(config)
    new_snapshot = _url_snapshot(
        epoch="new-sampled-scope",
        baseline=0,
        completed=0,
        total=1_106_793,
        elapsed=0.0,
    )
    assert new_snapshot.unobserved_nonlocal == new_snapshot.total
    resumed.update(
        stage=stage,
        url_snapshot=new_snapshot,
    )
    resumed.publish()

    telemetry = json.loads(resumed.path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ][stage]
    assert telemetry["completed_units"] == 0
    assert telemetry["total_units"] == 1_106_793
    assert len(telemetry["samples"]) == 1
    assert telemetry["samples"][0]["execution_epoch"] == (
        "new-sampled-scope"
    )
    ProgressReporter(config)


@pytest.mark.parametrize(
    "nonzero_field",
    [
        "local_buffered_not_started",
        "in_flight_jobs",
        "physical_in_flight",
        "finished_not_durable",
    ],
)
def test_progress_v2_scope_replacement_requires_empty_new_topology(
    tmp_path: Path,
    nonzero_field: str,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="old-empty-scope",
            baseline=0,
            completed=0,
            total=10,
            elapsed=0.0,
        ),
    )
    first.publish()

    snapshot = _url_snapshot(
        epoch="new-nonempty-scope",
        baseline=0,
        completed=0,
        total=5,
        elapsed=0.0,
    )
    changes: dict[str, Any] = {
        nonzero_field: 1,
        "unobserved_nonlocal": snapshot.total - 1,
    }
    if nonzero_field == "physical_in_flight":
        active = [0] * 64
        active[0] = 1
        changes.update(
            in_flight_jobs=1,
            active_censor_histogram=tuple(active),
        )
    nonempty = replace(snapshot, **changes)
    assert nonempty.unobserved_nonlocal == nonempty.total - 1

    resumed = ProgressReporter(config)
    with pytest.raises(ValueError, match="baseline|progress|total|scope"):
        resumed.update(stage="pages", url_snapshot=nonempty)


@pytest.mark.parametrize("stage", ["pages", "images"])
def test_progress_v2_rejects_scope_change_after_durable_completion(
    tmp_path: Path,
    stage: str,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = ProgressReporter(config)
    first.update(
        stage=stage,
        url_snapshot=_url_snapshot(
            epoch="old-active-scope",
            baseline=0,
            completed=0,
            total=10,
            elapsed=0.0,
        ),
    )
    first.update(
        url_snapshot=_url_snapshot(
            epoch="old-active-scope",
            baseline=0,
            completed=1,
            total=10,
            elapsed=1.0,
        )
    )
    first.publish()

    resumed = ProgressReporter(config)
    with pytest.raises(ValueError, match="baseline|total|scope"):
        resumed.update(
            stage=stage,
            url_snapshot=_url_snapshot(
                epoch="new-scope",
                baseline=0,
                completed=0,
                total=5,
                elapsed=0.0,
            ),
        )


def test_progress_v2_completed_stage_rejects_scope_change(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="completed-scope",
            baseline=0,
            completed=0,
            total=1,
            elapsed=0.0,
        ),
    )
    first.update(
        url_snapshot=_url_snapshot(
            epoch="completed-scope",
            baseline=0,
            completed=1,
            total=1,
            elapsed=1.0,
        )
    )
    first.publish()

    resumed = ProgressReporter(config)
    with pytest.raises(ValueError, match="already complete"):
        resumed.update(
            stage="pages",
            url_snapshot=_url_snapshot(
                epoch="changed-after-completion",
                baseline=0,
                completed=0,
                total=2,
                elapsed=0.0,
            ),
        )


def test_progress_v2_active_epoch_rejects_scope_change(
    tmp_path: Path,
) -> None:
    reporter = ProgressReporter(_full_pipeline_config(tmp_path))
    reporter.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="active-scope",
            baseline=0,
            completed=0,
            total=10,
            elapsed=0.0,
        ),
    )

    with pytest.raises(ValueError, match="total"):
        reporter.update(
            url_snapshot=_url_snapshot(
                epoch="active-scope",
                baseline=0,
                completed=0,
                total=5,
                elapsed=0.0,
            )
        )


def test_progress_v2_same_scope_zero_resume_preserves_old_telemetry(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="same-scope-before-resume",
            baseline=0,
            completed=0,
            total=10,
            elapsed=0.0,
        ),
    )
    first.publish()

    resumed = ProgressReporter(config)
    resumed.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="same-scope-after-resume",
            baseline=0,
            completed=0,
            total=10,
            elapsed=0.0,
        ),
    )
    telemetry = resumed._stage_telemetry["pages"]

    assert [sample["execution_epoch"] for sample in telemetry["samples"]] == [
        "same-scope-before-resume",
        "same-scope-after-resume",
    ]


@pytest.mark.parametrize("stage", ["pages", "images"])
def test_runner_epoch_elapsed_uses_first_callback_as_rate_and_time_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    config = _full_pipeline_config(tmp_path)
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 100.0)
    reporter = ProgressReporter(config)
    normalize = pipeline_module._EpochElapsedNormalizer()
    monotonic = [10.0]
    tracker = UrlProgressTracker(
        total=7,
        deadline_seconds=8.0,
        effective_concurrency=128,
        execution_epoch=f"{stage}-epoch-1",
        baseline_completed=2,
        monotonic=lambda: monotonic[0],
    )

    for completed, now in ((2, 10.25), (3, 11.25), (4, 12.75)):
        monotonic[0] = now
        reporter.update(
            stage=stage if completed == 2 else None,
            url_snapshot=normalize(
                tracker.snapshot(
                    DurableUrlCounts(
                        completed=completed,
                        pending=7 - completed,
                        leased=0,
                        total=7,
                    )
                )
            ),
        )
    reporter.publish()

    resumed = ProgressReporter(config)
    normalize_resumed = pipeline_module._EpochElapsedNormalizer()
    monotonic[0] = 20.0
    resumed_tracker = UrlProgressTracker(
        total=7,
        deadline_seconds=8.0,
        effective_concurrency=128,
        execution_epoch=f"{stage}-epoch-2",
        baseline_completed=4,
        monotonic=lambda: monotonic[0],
    )
    for completed, now in ((4, 24.5), (6, 26.5)):
        monotonic[0] = now
        resumed.update(
            stage=stage if completed == 4 else None,
            url_snapshot=normalize_resumed(
                resumed_tracker.snapshot(
                    DurableUrlCounts(
                        completed=completed,
                        pending=7 - completed,
                        leased=0,
                        total=7,
                    )
                )
            ),
        )
    resumed.publish()

    samples = json.loads(resumed.path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ][stage]["samples"]
    assert [sample["epoch_elapsed_seconds"] for sample in samples] == [
        0.0,
        1.0,
        2.5,
        0.0,
        2.0,
    ]
    first_baseline = samples[0]["baseline_timestamp"]
    second_baseline = samples[3]["baseline_timestamp"]
    assert [sample["timestamp"] for sample in samples] == [
        first_baseline,
        first_baseline + 1.0,
        first_baseline + 2.5,
        second_baseline,
        second_baseline + 2.0,
    ]
    assert samples[1]["durable_rate"] == 1.0
    assert samples[2]["durable_rate"] == 0.8
    assert samples[4]["durable_rate"] == 1.0


def _stub_network_runner_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    network_manifest = tmp_path / "network-manifest.json"
    monkeypatch.setattr(
        pipeline_module,
        "_publish_network_manifest",
        lambda *_args, **_kwargs: network_manifest,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_network_telemetry_counters",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        pipeline_module, "_read_only_job_scope", lambda *_args: {}
    )
    monkeypatch.setattr(
        pipeline_module, "_write_stage_registry", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        pipeline_module, "_registry_identity", lambda *_args: "upstream"
    )
    monkeypatch.setattr(
        pipeline_module, "_refresh_known_disk", lambda *_args, **_kwargs: None
    )
    return network_manifest


def test_run_pages_wires_nonzero_tracker_elapsed_through_reporter_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 100.0)
    _stub_network_runner_tail(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline_module, "_page_refs", lambda *_args: iter(()))
    monkeypatch.setattr(
        pipeline_module,
        "reconcile_page_jobs_from_outcomes",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_complete_page_fetch",
        lambda *_args, **_kwargs: {"identity": "pages"},
    )
    monkeypatch.setattr(
        pipeline_module,
        "iter_page_outcomes",
        lambda *_args, **_kwargs: iter(()),
    )

    result = SimpleNamespace(
        job_store_path=config.work_dir / "page_jobs" / "jobs.sqlite3",
        job_kind="page-kind",
        outcomes_path=config.cache_dir / "page_cache" / "outcomes.sqlite3",
        policy_fingerprint="page-policy",
        unique=2,
        success=2,
        terminal=0,
        remaining=0,
        leased=0,
        transport_attempt_summary={},
    )

    def fake_fetch(*_args: Any, progress_callback, **_kwargs: Any):
        for completed, elapsed in ((0, 5.0), (1, 6.5), (2, 7.0)):
            progress_callback(
                _url_snapshot(
                    epoch="wired-pages",
                    baseline=0,
                    completed=completed,
                    total=2,
                    elapsed=elapsed,
                )
            )
        return result

    monkeypatch.setattr(pipeline_module, "fetch_unique_pages", fake_fetch)
    reporter = ProgressReporter(config)
    pipeline_module._run_pages(
        config,
        reporter,
        SimpleNamespace(artifact_paths={"sampled_page_refs": ()}),
        object(),
    )

    restored = ProgressReporter(replace(config, resume=True))
    samples = restored._stage_telemetry["pages"]["samples"]
    assert [sample["epoch_elapsed_seconds"] for sample in samples] == [
        0.0,
        1.5,
        2.0,
    ]
    assert [sample["timestamp"] for sample in samples] == [100.0, 101.5, 102.0]


def test_run_images_wires_nonzero_tracker_elapsed_through_reporter_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 200.0)
    network_manifest = _stub_network_runner_tail(monkeypatch, tmp_path)
    unique_jobs = SimpleNamespace(
        records=2,
        manifest_path=tmp_path / "unique-manifest.json",
    )
    monkeypatch.setattr(
        pipeline_module, "build_unique_image_jobs", lambda *_args, **_kwargs: unique_jobs
    )
    monkeypatch.setattr(
        pipeline_module, "validate_unique_image_jobs", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        pipeline_module, "validate_complete_image_fetch", lambda *_args, **_kwargs: None
    )
    image_outcome_scope: dict[str, Any] = {}

    def fake_iter_image_outcomes(*_args: Any, **kwargs: Any):
        image_outcome_scope.update(kwargs)
        return iter(())

    monkeypatch.setattr(
        pipeline_module,
        "iter_image_outcomes",
        fake_iter_image_outcomes,
    )
    monkeypatch.setattr(
        pipeline_module,
        "asset_materialization_input_fingerprint",
        lambda *_args: "asset-input",
    )
    materialized = SimpleNamespace(
        bridge_assets=0,
        table_asset_links=0,
        manifest_path=tmp_path / "materialized-manifest.json",
    )
    monkeypatch.setattr(
        pipeline_module,
        "materialize_asset_shards",
        lambda *_args, **_kwargs: materialized,
    )
    barrier = SimpleNamespace()
    monkeypatch.setattr(
        pipeline_module,
        "validate_materialized_asset_shards",
        lambda *_args, **_kwargs: (materialized, barrier),
    )
    image_result = SimpleNamespace(
        job_store_path=config.work_dir / "image_jobs" / "jobs.sqlite3",
        job_kind="image-kind",
        outcomes_path=config.cache_dir / "image_cache" / "outcomes.sqlite3",
        policy_fingerprint="image-policy",
        fetch_manifest_path=tmp_path / "fetch-manifest.json",
        unique=2,
        success=2,
        terminal=0,
        remaining=0,
        leased=0,
        outcomes_count=2,
        transport_attempt_summary={},
    )

    def fake_download(*_args: Any, progress_callback, **_kwargs: Any):
        for completed, elapsed in ((0, 4.25), (1, 5.75), (2, 6.25)):
            progress_callback(
                _url_snapshot(
                    epoch="wired-images",
                    baseline=0,
                    completed=completed,
                    total=2,
                    elapsed=elapsed,
                )
            )
        return image_result

    monkeypatch.setattr(pipeline_module, "fetch_unique_images", fake_download)
    reporter = ProgressReporter(config)
    returned = pipeline_module._run_images(
        config,
        reporter,
        SimpleNamespace(manifest_path=tmp_path / "plan-manifest.json"),
        object(),
    )
    assert returned[-1] == network_manifest
    assert image_outcome_scope == {
        "job_store_path": image_result.job_store_path,
        "job_kind": image_result.job_kind,
    }

    restored = ProgressReporter(replace(config, resume=True))
    samples = restored._stage_telemetry["images"]["samples"]
    assert [sample["epoch_elapsed_seconds"] for sample in samples] == [
        0.0,
        1.5,
        2.0,
    ]
    assert [sample["timestamp"] for sample in samples] == [200.0, 201.5, 202.0]


def test_progress_v2_tracks_all_disk_root_extrema_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    for root in (config.work_dir, config.cache_dir, config.output_dir):
        root.mkdir(parents=True, exist_ok=True)
    free = {
        config.work_dir.resolve(): iter((900, 700, 800)),
        config.cache_dir.resolve(): iter((800, 600, 750)),
        config.output_dir.resolve(): iter((700, 500, 650)),
    }
    monkeypatch.setattr(
        pipeline_module.shutil,
        "disk_usage",
        lambda path: shutil._ntuple_diskusage(
            1_000, 1_000 - (value := next(free[Path(path).resolve()])), value
        ),
    )
    reporter = ProgressReporter(config)
    reporter.update(
        known_work_bytes=10, known_cache_bytes=20, known_output_bytes=30
    )
    reporter.publish()
    reporter.update(
        known_work_bytes=40, known_cache_bytes=15, known_output_bytes=35
    )
    reporter.publish()

    resumed = ProgressReporter(config)
    resumed.update(
        known_work_bytes=5, known_cache_bytes=25, known_output_bytes=32
    )
    resumed.publish()
    roots = json.loads(resumed.path.read_text(encoding="utf-8"))["disk"][
        "roots"
    ]

    assert roots == {
        "work": {
            "start_bytes": 10,
            "peak_bytes": 40,
            "current_bytes": 5,
            "start_free_bytes": 900,
            "min_free_bytes": 700,
        },
        "cache": {
            "start_bytes": 20,
            "peak_bytes": 25,
            "current_bytes": 25,
            "start_free_bytes": 800,
            "min_free_bytes": 600,
        },
        "output": {
            "start_bytes": 30,
            "peak_bytes": 35,
            "current_bytes": 32,
            "start_free_bytes": 700,
            "min_free_bytes": 500,
        },
    }


def test_progress_resume_start_preserves_persisted_disk_current_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _full_pipeline_config(tmp_path), progress_interval_seconds=60.0
    )
    monkeypatch.setattr(
        pipeline_module.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(2_000, 1_000, 1_000),
    )
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        known_work_bytes=11,
        known_cache_bytes=22,
        known_output_bytes=33,
        url_snapshot=_url_snapshot(
            epoch="disk-current-1",
            baseline=0,
            completed=0,
            total=1,
            elapsed=0.0,
        ),
    )
    first.publish()

    resumed = ProgressReporter(config)
    resumed.start()
    resumed.close()
    payload = json.loads(resumed.path.read_text(encoding="utf-8"))

    assert payload["disk"]["work_bytes"] == 11
    assert payload["disk"]["cache_bytes"] == 22
    assert payload["disk"]["output_bytes"] == 33
    assert {
        name: (
            root["start_bytes"],
            root["peak_bytes"],
            root["current_bytes"],
            root["start_free_bytes"],
            root["min_free_bytes"],
        )
        for name, root in payload["disk"]["roots"].items()
    } == {
        "work": (11, 11, 11, 1_000, 1_000),
        "cache": (22, 22, 22, 1_000, 1_000),
        "output": (33, 33, 33, 1_000, 1_000),
    }


def test_progress_v2_rejects_225th_completion_without_changing_published_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 1e300)
    reporter = ProgressReporter(config)
    baseline = UINT64_MAX - 256
    for stage, epoch in (("pages", "wide-pages"), ("images", "wide-images")):
        for offset in range(225):
            reporter.update(
                stage=stage if offset == 0 else None,
                url_snapshot=_url_snapshot(
                    epoch=epoch,
                    baseline=baseline,
                    completed=baseline + offset,
                    total=UINT64_MAX,
                    elapsed=float(offset),
                    deadline=1e300,
                    effective_concurrency=UINT64_MAX,
                ),
            )
        reporter.publish()
        published = reporter.path.read_bytes()
        with pytest.raises(ValueError, match="completion|224|bound"):
            reporter.update(
                url_snapshot=_url_snapshot(
                    epoch=epoch,
                    baseline=baseline,
                    completed=baseline + 225,
                    total=UINT64_MAX,
                    elapsed=225.0,
                    deadline=1e300,
                    effective_concurrency=UINT64_MAX,
                )
            )
        assert reporter.path.read_bytes() == published

    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert reporter.path.stat().st_size < 3 * 1024 * 1024
    for stage in ("pages", "images"):
        samples = payload["stage_telemetry"][stage]["samples"]
        assert len(samples) == 225
        for sample in samples:
            decoded = base64.b64decode(sample["histogram_blob"])
            assert len(decoded) == 1_280
            transport, active, commit = decode_histogram_blob(
                sample["histogram_blob"]
            )
            rebuilt = _url_snapshot(
                epoch=sample["execution_epoch"],
                baseline=sample["baseline_completed"],
                completed=sample["completed_units"],
                total=sample["total_units"],
                elapsed=sample["epoch_elapsed_seconds"],
                deadline=sample["deadline_seconds"],
                effective_concurrency=sample["effective_concurrency"],
            )
            assert transport == rebuilt.transport_event_histogram
            assert active == rebuilt.active_censor_histogram
            assert commit == rebuilt.commit_event_histogram
            assert encode_histogram_blob(rebuilt) == sample["histogram_blob"]
    ProgressReporter(config)


def test_progress_v2_rejects_33rd_epoch_without_changing_published_bytes(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), resume=True)
    for ordinal in range(32):
        reporter = ProgressReporter(config)
        reporter.update(
            stage="pages",
            url_snapshot=_url_snapshot(
                epoch=f"epoch-{ordinal}",
                baseline=0,
                completed=0,
                total=1,
                elapsed=0.0,
            ),
        )
        reporter.publish()

    published = reporter.path.read_bytes()
    overflow = ProgressReporter(config)
    with pytest.raises(ValueError, match="epoch|32"):
        overflow.update(
            stage="pages",
            url_snapshot=_url_snapshot(
                epoch="epoch-32",
                baseline=0,
                completed=0,
                total=1,
                elapsed=0.0,
            ),
        )
    assert overflow.path.read_bytes() == published


def test_page_progress_absolute_schedule_preserves_interrupted_epoch_authority(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    urls = [f"https://progress.example/{ordinal}" for ordinal in range(256)]
    refs = [
        {
            "entity_id": str(ordinal),
            "page_url": url,
            "url_key": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        }
        for ordinal, url in enumerate(urls)
    ]

    class FastTransport:
        network_policy_fingerprint = "wdc-web-v1"

        def fetch_page(self, url: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "page_url": url,
                "final_url": url,
                "text": url,
                "image_urls": [],
            }

    policy = FetchPolicy(
        global_concurrency=16,
        per_host_concurrency=16,
    )
    jobs = SqliteJobStore(tmp_path / "progress-jobs.sqlite3")
    outcomes = tmp_path / "progress-outcomes.sqlite3"
    first = ProgressReporter(config)
    first_normalizer = pipeline_module._EpochElapsedNormalizer()

    def interrupting_callback(snapshot: UrlProgressSnapshot) -> None:
        first.update(
            stage="pages" if first._state.stage != "pages" else None,
            url_snapshot=first_normalizer(snapshot),
        )
        first.publish()
        if 200 <= snapshot.completed_durable < snapshot.total:
            raise KeyboardInterrupt("synthetic durable cutoff")

    with pytest.raises(KeyboardInterrupt, match="durable cutoff"):
        fetch_unique_pages(
            refs,
            jobs,
            FastTransport(),
            policy,
            outcomes_path=outcomes,
            claim_buffer=16,
            lease_seconds=60.0,
            progress_callback=interrupting_callback,
            progress_callback_every=1,
        )

    before_resume = json.loads(first.path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ]["pages"]["samples"]
    cutoff = before_resume[-1]["completed_units"]
    assert 200 <= cutoff < 256
    eligible_before_resume = [
        sample
        for sample in before_resume
        if sample["completed_units"] * 2 >= sample["total_units"]
    ]
    assert eligible_before_resume

    with sqlite3.connect(jobs.path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires = 0 WHERE status = 'leased'"
        )

    resumed = ProgressReporter(replace(config, resume=True))
    resumed_normalizer = pipeline_module._EpochElapsedNormalizer()
    resumed_generated: list[dict[str, Any]] = []

    def resumed_callback(snapshot: UrlProgressSnapshot) -> None:
        resumed.update(
            stage="pages" if resumed._state.stage != "pages" else None,
            url_snapshot=resumed_normalizer(snapshot),
        )
        resumed_generated.append(
            json.loads(
                json.dumps(
                    resumed._stage_telemetry["pages"]["samples"][-1]
                )
            )
        )
        resumed.publish()

    result = fetch_unique_pages(
        refs,
        jobs,
        FastTransport(),
        policy,
        outcomes_path=outcomes,
        claim_buffer=16,
        lease_seconds=60.0,
        progress_callback=resumed_callback,
        progress_callback_every=1,
    )
    assert result.complete

    telemetry = json.loads(resumed.path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ]["pages"]
    samples = telemetry["samples"]
    authoritative_samples = [*before_resume, *resumed_generated]
    assert samples == authoritative_samples
    assert len(samples) <= 256
    assert len({sample["execution_epoch"] for sample in samples}) == 2
    assert [sample["completed_units"] for sample in samples].count(cutoff) == 2
    expected_authority = ProgressReporter._eta_completion_summary(
        authoritative_samples,
        total=256,
        completed_at=telemetry["completed_at"],
    )
    assert {
        key: telemetry[key] for key in expected_authority
    } == expected_authority


def test_progress_restore_rejects_files_above_four_mibibytes(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    (config.work_dir / "progress.json").write_bytes(
        b" " * (4 * 1024 * 1024 + 1)
    )

    with pytest.raises(ValueError, match="bounded state size"):
        ProgressReporter(config)


@pytest.mark.parametrize("from_stage", [None, "pages"])
@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda payload: payload.pop("disk"),
            id="missing-disk",
        ),
        pytest.param(
            lambda payload: payload["disk"].pop("roots"),
            id="missing-roots",
        ),
        pytest.param(
            lambda payload: payload["disk"]["roots"].pop("cache"),
            id="missing-one-root",
        ),
    ],
)
def test_progress_v2_restore_requires_complete_disk_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
    from_stage: str | None,
) -> None:
    config, path = _published_v2_progress(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tamper(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="disk roots"):
        ProgressReporter(replace(config, from_stage=from_stage))


def _drop_sample_schema_markers(payload: dict[str, Any]) -> None:
    for sample in payload["stage_telemetry"]["pages"]["samples"]:
        sample.pop("telemetry_schema_version")


def _drop_sample_schema_markers_and_disk(payload: dict[str, Any]) -> None:
    _drop_sample_schema_markers(payload)
    payload.pop("disk")


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            _drop_sample_schema_markers_and_disk,
            id="stage-v2-samples-unmarked-disk-missing",
        ),
        pytest.param(
            _drop_sample_schema_markers,
            id="stage-v2-samples-unmarked",
        ),
        pytest.param(
            lambda payload: payload["stage_telemetry"]["pages"].pop(
                "telemetry_schema_version"
            ),
            id="sample-v2-stage-schema-missing",
        ),
        pytest.param(
            lambda payload: payload["stage_telemetry"]["pages"].__setitem__(
                "telemetry_schema_version", "wrong-schema"
            ),
            id="sample-v2-stage-schema-wrong",
        ),
        pytest.param(
            lambda payload: payload["stage_telemetry"]["pages"]["samples"][
                -1
            ].__setitem__("telemetry_schema_version", "wrong-schema"),
            id="stage-v2-sample-schema-wrong",
        ),
    ],
)
def test_progress_v2_schema_markers_fail_closed_before_from_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
) -> None:
    config, path = _published_v2_progress(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tamper(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="progress"):
        ProgressReporter(replace(config, from_stage="pages"))


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda root: root.pop("peak_bytes"),
            id="missing-extremum",
        ),
        pytest.param(
            lambda root: root.__setitem__("min_free_bytes", -1),
            id="negative-extremum",
        ),
        pytest.param(
            lambda root: root.__setitem__(
                "current_bytes", root["peak_bytes"] + 1
            ),
            id="inconsistent-extrema",
        ),
    ],
)
def test_progress_v2_restore_requires_valid_disk_extrema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
) -> None:
    config, path = _published_v2_progress(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tamper(payload["disk"]["roots"]["work"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="progress disk"):
        ProgressReporter(config)


def _published_v2_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PipelineConfig, Path]:
    monkeypatch.setattr(
        pipeline_module.time, "time", lambda: 1_700_000_000.0
    )
    config = _full_pipeline_config(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="tamper-1", baseline=0, completed=0, total=3, elapsed=0.0
        ),
    )
    reporter.update(
        url_snapshot=_url_snapshot(
            epoch="tamper-1", baseline=0, completed=1, total=3, elapsed=1.0
        )
    )
    reporter.update(
        url_snapshot=_url_snapshot(
            epoch="tamper-1", baseline=0, completed=2, total=3, elapsed=2.0
        )
    )
    reporter.publish()
    return config, reporter.path


@pytest.mark.parametrize(
    "tamper",
    [
        lambda samples: samples[-1].__setitem__(
            "histogram_blob", samples[-1]["histogram_blob"][:-4]
        ),
        lambda samples: samples[-1].__setitem__(
            "histogram_blob",
            base64.b64encode(base64.b64decode(samples[-1]["histogram_blob"]) + b"x").decode("ascii"),
        ),
        lambda samples: samples[-1].__setitem__(
            "histogram_blob", samples[-1]["histogram_blob"] + "\n"
        ),
        lambda samples: samples[-1].__setitem__(
            "baseline_completed", 2**64
        ),
        lambda samples: samples[-1].__setitem__(
            "unobserved_nonlocal", samples[-1]["unobserved_nonlocal"] + 1
        ),
        lambda samples: samples[-1].__setitem__("deadline_seconds", 9.0),
        lambda samples: samples[-1].__setitem__("total_units", 4),
        lambda samples: samples[-1].__setitem__(
            "predicted_remaining_seconds",
            samples[-1]["predicted_remaining_seconds"] + 1.0,
        ),
        lambda samples: samples[-1].pop("overflow_eta"),
    ],
)
def test_progress_v2_restore_rejects_blob_counter_topology_and_eta_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
) -> None:
    config, path = _published_v2_progress(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tamper(payload["stage_telemetry"]["pages"]["samples"])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="progress"):
        ProgressReporter(config)


def test_progress_v2_restore_rejects_decreasing_events_and_reused_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, path = _published_v2_progress(tmp_path, monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["stage_telemetry"]["pages"]["samples"]
    transport, active, commit = decode_histogram_blob(samples[-1]["histogram_blob"])
    forged = _url_snapshot(
        epoch="tamper-1",
        baseline=0,
        completed=2,
        total=3,
        elapsed=2.0,
    )
    forged = replace(
        forged,
        transport_event_histogram=(0,) * 64,
        commit_event_histogram=commit,
    )
    samples[-1]["histogram_blob"] = encode_histogram_blob(forged)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="progress"):
        ProgressReporter(config)

    config, path = _published_v2_progress(tmp_path / "epochs", monkeypatch)
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["stage_telemetry"]["pages"]["samples"]
    samples[1]["execution_epoch"] = "epoch-2"
    samples[2]["execution_epoch"] = "tamper-1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="progress"):
        ProgressReporter(config)


@pytest.mark.parametrize(
    ("argument_name", "counter_name"),
    [
        ("transport_overflow", "transport_overflow_events"),
        ("commit_overflow", "commit_overflow_events"),
    ],
)
def test_progress_v2_restore_rejects_decreasing_cumulative_overflow(
    tmp_path: Path,
    argument_name: str,
    counter_name: str,
) -> None:
    config = _full_pipeline_config(tmp_path)
    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="overflow-1",
            baseline=0,
            completed=0,
            total=3,
            elapsed=0.0,
        ),
    )
    second = _url_snapshot(
        epoch="overflow-1",
        baseline=0,
        completed=1,
        total=3,
        elapsed=1.0,
        **{argument_name: 1},
    )
    reporter.update(url_snapshot=second)
    third = _url_snapshot(
        epoch="overflow-1",
        baseline=0,
        completed=2,
        total=3,
        elapsed=2.0,
        **{argument_name: 2},
    )
    reporter.update(url_snapshot=third)
    reporter.publish()
    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    sample = payload["stage_telemetry"]["pages"]["samples"][-1]
    forged = replace(third, **{counter_name: 0})
    payload["stage_telemetry"]["pages"]["samples"][-1] = (
        ProgressReporter._v2_sample(
            forged,
            baseline_timestamp=sample["baseline_timestamp"],
        )
    )
    reporter.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="overflow"):
        ProgressReporter(config)


@pytest.mark.parametrize(
    "changed",
    [
        _url_snapshot(
            epoch="same-epoch",
            baseline=1,
            completed=1,
            total=3,
            elapsed=1.0,
        ),
        _url_snapshot(
            epoch="same-epoch",
            baseline=0,
            completed=1,
            total=3,
            elapsed=1.0,
            effective_concurrency=64,
        ),
    ],
)
def test_progress_v2_rejects_changed_epoch_baseline_or_concurrency_before_write(
    tmp_path: Path,
    changed: UrlProgressSnapshot,
) -> None:
    config = _full_pipeline_config(tmp_path)
    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="same-epoch",
            baseline=0,
            completed=0,
            total=3,
            elapsed=0.0,
        ),
    )

    with pytest.raises(ValueError, match="progress"):
        reporter.update(url_snapshot=changed)


def test_progress_completed_v1_preserves_recorded_eta_and_summary_authority(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    recorded = {
        "rate_basis": "page_urls",
        "completed_units": 2,
        "total_units": 2,
        "completed_at": 12,
        "eligible_final_half_samples": 2,
        "excluded_final_half_samples": 0,
        "max_symmetric_eta_factor": 9,
        "samples": [
            {
                "timestamp": 10.0,
                "completed_units": 1,
                "total_units": 2,
                "rate": 1.0,
                "rolling_rate": 1.0,
                "predicted_remaining_seconds": 9.0,
            },
            {
                "timestamp": 12.0,
                "completed_units": 2,
                "total_units": 2,
                "rate": 1.0,
                "rolling_rate": 1.0,
                "predicted_remaining_seconds": 4.0,
            },
        ],
    }
    path = config.work_dir / "progress.json"
    path.write_text(
        json.dumps({"stage_telemetry": {"pages": recorded}}),
        encoding="utf-8",
    )

    restored = ProgressReporter(config)

    authority = restored.stage_completion_summary("pages")
    assert authority == {
        key: recorded[key]
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
    assert type(authority["completed_at"]) is int
    assert type(authority["max_symmetric_eta_factor"]) is int
    assert restored._stage_telemetry["pages"]["samples"] == recorded["samples"]


def _legacy_incomplete_telemetry(sample_count: int, *, total: int) -> dict[str, Any]:
    samples = [
        {
            "timestamp": float(index + 1),
            "completed_units": index + 1,
            "total_units": total,
            "rate": 1.0,
            "rolling_rate": 1.0,
            "predicted_remaining_seconds": float(total - index - 1),
        }
        for index in range(sample_count)
    ]
    return {
        "rate_basis": "page_urls",
        "completed_units": sample_count,
        "total_units": total,
        "completed_at": None,
        "eligible_final_half_samples": 0,
        "excluded_final_half_samples": 0,
        "max_symmetric_eta_factor": None,
        "samples": samples,
    }


@pytest.mark.parametrize("legacy_count", [255, 256])
def test_incomplete_v1_full_prefix_upgrades_and_keeps_completion_authority(
    tmp_path: Path,
    legacy_count: int,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    total = legacy_count + 1
    legacy = _legacy_incomplete_telemetry(legacy_count, total=total)
    path = config.work_dir / "progress.json"
    path.write_text(
        json.dumps({"stage_telemetry": {"pages": legacy}}),
        encoding="utf-8",
    )

    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch=f"upgrade-{legacy_count}",
            baseline=legacy_count,
            completed=legacy_count,
            total=total,
            elapsed=0.0,
        ),
    )
    reporter.update(
        url_snapshot=_url_snapshot(
            epoch=f"upgrade-{legacy_count}",
            baseline=legacy_count,
            completed=total,
            total=total,
            elapsed=1.0,
        )
    )
    reporter.publish()

    restored = ProgressReporter(config)
    telemetry = restored._stage_telemetry["pages"]
    assert telemetry["samples"][:legacy_count] == legacy["samples"]
    assert len(telemetry["samples"]) == legacy_count + 2
    expected = ProgressReporter._eta_completion_summary(
        telemetry["samples"],
        total=total,
        completed_at=telemetry["completed_at"],
    )
    assert {key: telemetry[key] for key in expected} == expected


def test_progress_restore_rejects_257_legacy_samples(tmp_path: Path) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    legacy = _legacy_incomplete_telemetry(257, total=300)
    (config.work_dir / "progress.json").write_text(
        json.dumps({"stage_telemetry": {"pages": legacy}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bound|256|sample"):
        ProgressReporter(config)


def test_progress_restore_rejects_more_than_224_v2_completion_publications(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ProgressReporter._v2_sample(
            _url_snapshot(
                epoch="forged-completions",
                baseline=0,
                completed=completed,
                total=300,
                elapsed=float(completed),
            ),
            baseline_timestamp=100.0,
        )
        for completed in range(226)
    ]
    stage = {
        "rate_basis": "page_urls",
        "samples": samples,
        "completed_units": 225,
        "total_units": 300,
        "completed_at": None,
        "eligible_final_half_samples": 0,
        "excluded_final_half_samples": 0,
        "max_symmetric_eta_factor": None,
        "telemetry_schema_version": URL_TELEMETRY_V2,
        "estimator": pipeline_module.url_estimator_metadata(8.0),
    }
    disk_root = {
        "start_bytes": 0,
        "peak_bytes": 0,
        "current_bytes": 0,
        "start_free_bytes": 1,
        "min_free_bytes": 1,
    }
    (config.work_dir / "progress.json").write_text(
        json.dumps(
            {
                "stage_telemetry": {"pages": stage},
                "disk": {
                    "roots": {
                        name: dict(disk_root)
                        for name in ("work", "cache", "output")
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="224|completion|bound"):
        ProgressReporter(config)


def _maximum_mixed_stage(rate_basis: str) -> dict[str, Any]:
    total = 1_000
    legacy = _legacy_incomplete_telemetry(256, total=total)
    v2_samples: list[dict[str, Any]] = []
    completed = 256
    for epoch_index in range(32):
        baseline = completed
        baseline_timestamp = 1_000.0 + epoch_index * 10.0
        for offset in range(8):
            current = baseline + offset
            v2_samples.append(
                ProgressReporter._v2_sample(
                    _url_snapshot(
                        epoch=f"mixed-{rate_basis}-{epoch_index}",
                        baseline=baseline,
                        completed=current,
                        total=total,
                        elapsed=float(offset),
                    ),
                    baseline_timestamp=baseline_timestamp,
                )
            )
        completed += 7
    return {
        **legacy,
        "rate_basis": rate_basis,
        "samples": [*legacy["samples"], *v2_samples],
        "completed_units": completed,
        "telemetry_schema_version": URL_TELEMETRY_V2,
        "estimator": pipeline_module.url_estimator_metadata(8.0),
    }


def test_maximum_mixed_v1_v2_suffix_is_bounded_and_257th_v2_fails_closed(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    disk_root = {
        "start_bytes": 0,
        "peak_bytes": 0,
        "current_bytes": 0,
        "start_free_bytes": 1,
        "min_free_bytes": 1,
    }
    payload = {
        "stage_telemetry": {
            "pages": _maximum_mixed_stage("page_urls"),
            "images": _maximum_mixed_stage("image_urls"),
        },
        "disk": {
            "roots": {
                name: dict(disk_root) for name in ("work", "cache", "output")
            }
        },
    }
    path = config.work_dir / "progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    reporter = ProgressReporter(config)
    reporter.publish()
    assert path.stat().st_size < 3 * 1024 * 1024
    for stage in ("pages", "images"):
        samples = reporter._stage_telemetry[stage]["samples"]
        assert sum("telemetry_schema_version" not in item for item in samples) == 256
        assert sum("telemetry_schema_version" in item for item in samples) == 256

    published = path.read_bytes()
    with pytest.raises(ValueError, match="256|32|sample|epoch"):
        reporter.update(
            stage="pages",
            url_snapshot=_url_snapshot(
                epoch="mixed-overflow",
                baseline=480,
                completed=480,
                total=1_000,
                elapsed=0.0,
            ),
        )
    assert path.read_bytes() == published


def test_progress_incomplete_v1_recorded_eta_cannot_change_v2_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    legacy_sample = {
        "timestamp": 10.0,
        "completed_units": 1,
        "total_units": 3,
        "rate": 1.0,
        "rolling_rate": 1.0,
        "predicted_remaining_seconds": 99.0,
    }
    incomplete = {
        "rate_basis": "page_urls",
        "completed_units": 1,
        "total_units": 3,
        "completed_at": None,
        "eligible_final_half_samples": 0,
        "excluded_final_half_samples": 0,
        "max_symmetric_eta_factor": None,
        "samples": [legacy_sample],
    }
    path = config.work_dir / "progress.json"
    path.write_text(
        json.dumps({"stage_telemetry": {"pages": incomplete}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 5.0)

    upgraded = ProgressReporter(config)
    upgraded.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="v2-after-recorded-v1",
            baseline=1,
            completed=1,
            total=3,
            elapsed=0.0,
        ),
    )
    upgraded.publish()
    samples = json.loads(path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ]["pages"]["samples"]

    assert samples[0] == legacy_sample
    assert samples[1]["baseline_completed"] == 1
    assert samples[1]["completed_units"] == 1
    assert samples[1]["epoch_elapsed_seconds"] == 0.0
    assert samples[1]["timestamp"] >= legacy_sample["timestamp"]


def test_progress_completed_v1_compatibility_and_incomplete_v1_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    completed_summary = {
        "rate_basis": "page_urls",
        "completed_units": 2,
        "total_units": 2,
        "completed_at": 12.0,
        "eligible_final_half_samples": 1,
        "excluded_final_half_samples": 1,
        "max_symmetric_eta_factor": 2.0,
        "samples": [
            {
                "timestamp": 10.0,
                "completed_units": 1,
                "total_units": 2,
                "rate": 1.0,
                "rolling_rate": 1.0,
                "predicted_remaining_seconds": 1.0,
            },
            {
                "timestamp": 12.0,
                "completed_units": 2,
                "total_units": 2,
                "rate": 1.0,
                "rolling_rate": 1.0,
                "predicted_remaining_seconds": 0.0,
            },
        ],
    }
    progress_path = config.work_dir / "progress.json"
    progress_path.write_text(
        json.dumps({"stage_telemetry": {"pages": completed_summary}}),
        encoding="utf-8",
    )
    restored = ProgressReporter(config)
    assert restored.stage_completion_summary("pages") == {
        key: completed_summary[key]
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

    incomplete_sample = dict(completed_summary["samples"][0])
    incomplete_sample.update(
        total_units=3,
        predicted_remaining_seconds=2.0,
    )
    incomplete = dict(completed_summary)
    incomplete.update(
        completed_units=1,
        total_units=3,
        completed_at=None,
        eligible_final_half_samples=0,
        excluded_final_half_samples=0,
        max_symmetric_eta_factor=None,
        samples=[incomplete_sample],
    )
    progress_path.write_text(
        json.dumps({"stage_telemetry": {"pages": incomplete}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_module.time, "time", lambda: 5.0)
    upgraded = ProgressReporter(config)
    upgraded.update(
        stage="pages",
        url_snapshot=_url_snapshot(
            epoch="v2-upgrade", baseline=1, completed=1, total=3, elapsed=0.0
        ),
    )
    upgraded.publish()
    samples = json.loads(progress_path.read_text(encoding="utf-8"))[
        "stage_telemetry"
    ]["pages"]["samples"]
    assert samples[0] == incomplete["samples"][0]
    assert samples[1]["telemetry_schema_version"] == URL_TELEMETRY_V2
    assert samples[1]["baseline_completed"] == 1
    assert samples[1]["epoch_elapsed_seconds"] == 0.0


def test_progress_guard_failure_preserves_previous_bytes(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    reporter = ProgressReporter(config)
    reporter.publish()
    previous = reporter.path.read_bytes()

    def fail_guard(_path: Path, _estimated_bytes: int = 0) -> None:
        raise DiskSpaceInsufficientError("synthetic progress reserve exhausted")

    reporter._pre_write_guard = fail_guard
    reporter.update(counters={"after_failure": 1})
    with pytest.raises(DiskSpaceInsufficientError, match="progress reserve"):
        reporter.publish()
    assert reporter.path.read_bytes() == previous
    assert not list(reporter.path.parent.glob(f".{reporter.path.name}.*.tmp"))


def test_tree_bytes_tolerates_files_removed_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VanishedEntry:
        path = str(tmp_path / ".progress.tmp")

        def is_dir(self, *, follow_symlinks: bool) -> bool:
            return False

        def is_file(self, *, follow_symlinks: bool) -> bool:
            return True

        def stat(self, *, follow_symlinks: bool) -> Any:
            raise FileNotFoundError(self.path)

    class Entries:
        def __enter__(self) -> list[VanishedEntry]:
            return [VanishedEntry()]

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(pipeline_module.os, "scandir", lambda _path: Entries())

    assert pipeline_module._tree_bytes(tmp_path) == 0


def test_network_outcome_manifest_repairs_only_the_corrupt_shard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "network"
    records = [{"job_id": index} for index in range(5)]
    arguments = {
        "policy_fingerprint": "policy-v1",
        "unique": 5,
        "success": 5,
        "terminal": 0,
        "pending": 0,
        "leased": 0,
        "records_per_shard": 2,
    }
    manifest_path = _publish_network_manifest(root, iter(records), **arguments)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [root / item["path"] for item in manifest["completed_shards"]]
    assert [item["records"] for item in manifest["completed_shards"]] == [2, 2, 1]
    mtimes = [path.stat().st_mtime_ns for path in shards]

    shards[1].write_text("corrupt\n", encoding="utf-8")
    os.utime(shards[1], ns=(1, 1))
    corrupt_mtime = shards[1].stat().st_mtime_ns
    repaired_path = _publish_network_manifest(root, iter(records), **arguments)
    repaired = json.loads(repaired_path.read_text(encoding="utf-8"))

    assert len(repaired["completed_shards"]) == 3
    assert shards[0].stat().st_mtime_ns == mtimes[0]
    assert shards[2].stat().st_mtime_ns == mtimes[2]
    assert shards[1].stat().st_mtime_ns != corrupt_mtime


def test_scoped_image_cache_publishes_only_current_job_set(
    tmp_path: Path,
) -> None:
    policy_fingerprint = "image-policy-v1"
    current_kind = "current-image-jobs"
    outcomes_path = tmp_path / "outcomes.sqlite3"
    jobs_path = tmp_path / "jobs.sqlite3"
    outcome_store = ImageOutcomeStore(outcomes_path)
    job_store = SqliteJobStore(jobs_path)
    current_urls = [
        "https://i.test/current-a.jpg",
        "https://i.test/current-b.jpg",
    ]
    cached_urls = [*current_urls, "https://i.test/other-dataset.jpg"]

    for image_url in cached_urls:
        url_key = hashlib.sha256(image_url.encode("utf-8")).hexdigest()
        outcome_store.put(
            policy_fingerprint,
            url_key,
            image_url,
            {"status": "success"},
        )
        if image_url in current_urls:
            job_store.enqueue(
                current_kind,
                f"{current_kind}:{url_key}",
                {"url_key": url_key, "image_url": image_url},
            )

    manifest_path = _publish_network_manifest(
        tmp_path / "network",
        pipeline_module.iter_image_outcomes(
            outcomes_path,
            policy_fingerprint,
            job_store_path=jobs_path,
            job_kind=current_kind,
        ),
        policy_fingerprint=policy_fingerprint,
        unique=2,
        success=2,
        terminal=0,
        pending=0,
        leased=0,
        records_per_shard=10,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert sum(
        int(shard["records"])
        for shard in manifest["completed_shards"]
    ) == 2


def test_resume_rejects_tampered_existing_registry_chain(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    run_pipeline(config, page_transport=_PipelinePageTransport())
    selection_registry = json.loads(
        (
            config.work_dir / "stage_manifests/pipeline-selection.json"
        ).read_text(encoding="utf-8")
    )
    producer = Path(selection_registry["producer_manifests"][0]["path"])
    producer.write_text(
        producer.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        run_pipeline(config, page_transport=_PipelinePageTransport())


def test_page_cache_guard_stops_before_outcome_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    original = pipeline_module.DiskGuard.__call__
    guarded_targets: list[Path] = []

    def fail_before_cache_write(
        guard: DiskGuard,
        target: Path,
        estimated_bytes: int = 0,
    ) -> None:
        target = Path(target).resolve()
        guarded_targets.append(target)
        if target == config.cache_dir / "page_cache" / "outcomes.sqlite3":
            raise DiskSpaceInsufficientError("synthetic cache reserve exhausted")
        original(guard, target, estimated_bytes)

    monkeypatch.setattr(
        pipeline_module.DiskGuard,
        "__call__",
        fail_before_cache_write,
    )

    with pytest.raises(DiskSpaceInsufficientError, match="cache reserve"):
        run_pipeline(config, page_transport=_PipelinePageTransport())

    outcomes = config.cache_dir / "page_cache" / "outcomes.sqlite3"
    assert outcomes in guarded_targets
    assert not outcomes.exists()
    assert (config.work_dir / "stage_manifests/pipeline-structural.json").is_file()


def test_output_guard_stops_before_first_materialized_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    original = pipeline_module.DiskGuard.__call__

    def fail_on_output_descendant(
        guard: DiskGuard,
        target: Path,
        estimated_bytes: int = 0,
    ) -> None:
        target = Path(target).resolve()
        if target != config.output_dir and target.is_relative_to(config.output_dir):
            raise DiskSpaceInsufficientError("synthetic output reserve exhausted")
        original(guard, target, estimated_bytes)

    monkeypatch.setattr(
        pipeline_module.DiskGuard,
        "__call__",
        fail_on_output_descendant,
    )

    with pytest.raises(DiskSpaceInsufficientError, match="output reserve"):
        run_pipeline(
            config,
            page_transport=_PipelinePageTransport(),
            image_transport=_PipelineImageTransport(),
            extractor=_PipelineExtractor(),
        )

    assert not (config.output_dir / "source_tables").exists()
    assert (config.work_dir / "stage_manifests/pipeline-models.json").is_file()


def test_dynamic_model_markers_are_forwarded_to_authoritative_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _full_pipeline_config(tmp_path)
    runtime = base.work_dir / "runtime"
    config = replace(
        base,
        runtime_dir=runtime,
        run_fingerprint="dynamic-run-v1",
        model_start_marker=runtime / "start.json",
        model_ready_marker=runtime / "ready.json",
        model_ready_timeout_seconds=7.0,
        model_text_done_marker=runtime / "text-done.json",
        model_image_done_marker=runtime / "image-done.json",
        model_endpoint_ready_timeout_seconds=45.5,
    )
    captured: dict[str, Any] = {}
    real_runner = pipeline_module.run_model_stage

    def capture_runner(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        kwargs.update(
            start_marker=None,
            ready_marker=None,
            text_done_marker=None,
            image_done_marker=None,
        )
        return real_runner(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "run_model_stage", capture_runner)

    result = run_pipeline(
        config,
        page_transport=_PipelinePageTransport(),
        image_transport=_PipelineImageTransport(),
        extractor=_PipelineExtractor(),
    )

    assert result.status == "complete"
    assert captured["start_marker"] == config.model_start_marker
    assert captured["ready_marker"] == config.model_ready_marker
    assert captured["text_done_marker"] == config.model_text_done_marker
    assert captured["image_done_marker"] == config.model_image_done_marker
    assert captured["run_fingerprint"] == "dynamic-run-v1"
    assert captured["ready_timeout_seconds"] == 7.0
    assert captured["endpoint_ready_timeout_seconds"] == 45.5
    assert callable(captured["model_progress_callback"])
    assert "after_result_write" not in captured
    assert callable(captured["pre_write_guard"])
    assert len(tuple(captured["network_manifests"])) == 2
    assert captured["assets_manifest"].is_file()


def test_from_stage_rebuilds_downstream_and_reuses_durable_url_cache(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    page_transport = _PipelinePageTransport()
    image_transport = _PipelineImageTransport()
    run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )
    physical_calls = (page_transport.calls, image_transport.calls)

    refreshed = run_pipeline(
        replace(config, from_stage="pages"),
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert refreshed.status == "complete"
    assert (page_transport.calls, image_transport.calls) == physical_calls
    stale_outputs = tuple(
        (
            config.output_dir.parent
            / f".{config.output_dir.name}.wdc200k-stale"
        ).glob("*/root")
    )
    assert len(stale_outputs) == 1
    assert (stale_outputs[0] / "dataset_manifest.json").is_file()


def test_explicit_page_cache_refresh_archives_cache_without_deleting(
    tmp_path: Path,
) -> None:
    config = _full_pipeline_config(tmp_path)
    page_transport = _PipelinePageTransport()
    image_transport = _PipelineImageTransport()
    run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )
    image_calls = image_transport.calls

    run_pipeline(
        replace(
            config,
            from_stage="pages",
            refresh_page_cache=True,
        ),
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert page_transport.calls == 2
    assert image_transport.calls == image_calls
    archived = tuple(
        (
            config.cache_dir.parent
            / f".{config.cache_dir.name}.wdc200k-stale"
        ).glob("*/page_cache/outcomes.sqlite3")
    )
    assert len(archived) == 1
    assert (config.cache_dir / "page_cache/outcomes.sqlite3").is_file()


def test_remote_layout_control_reuses_extractor_api_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _full_pipeline_config(tmp_path)
    runtime = base.work_dir / "runtime"
    config = replace(
        base,
        remote_layout_control_url="http://127.0.0.1:18999",
        remote_layout_control_token_file=tmp_path / "layout-token",
        remote_layout_routing_manifest=runtime / "routing.json",
        remote_layout_controller_id_file=runtime / "controller-id",
        remote_layout_lock_file=runtime / "controller.lock",
        remote_layout_primary_image_url="http://127.0.0.1:18000/v1",
        remote_layout_switchable_url="http://127.0.0.1:18001/v1",
    )
    captured: dict[str, Any] = {}

    class FakeController:
        def __init__(self, controller_config: Any, scheduler: Any) -> None:
            captured["config"] = controller_config
            captured["scheduler"] = scheduler

        def __enter__(self) -> "FakeController":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    scheduler = object()
    extractor = SimpleNamespace(
        routing_scheduler=scheduler,
        text_model_api_key="resolved-text-key",
        image_model_api_key="resolved-image-key",
    )
    monkeypatch.setattr(
        pipeline_module,
        "RemoteLayoutController",
        FakeController,
    )

    with pipeline_module._remote_layout_control(
        config,
        extractor=extractor,
        database_path=tmp_path / "jobs.sqlite3",
    ):
        pass

    controller_config = captured["config"]
    assert captured["scheduler"] is scheduler
    assert controller_config.text_api_key == "resolved-text-key"
    assert controller_config.image_api_key == "resolved-image-key"


def test_remote_layout_control_reclaims_before_start_and_releases_afterward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _full_pipeline_config(tmp_path)
    runtime = base.work_dir / "runtime"
    config = replace(
        base,
        remote_layout_control_url="http://127.0.0.1:18999",
        remote_layout_control_token_file=tmp_path / "layout-token",
        remote_layout_routing_manifest=runtime / "routing.json",
        remote_layout_controller_id_file=runtime / "controller-id",
        remote_layout_lock_file=runtime / "controller.lock",
        remote_layout_primary_image_url="http://127.0.0.1:18000/v1",
        remote_layout_switchable_url="http://127.0.0.1:18001/v1",
        remote_layout_coordination_dir=tmp_path / "remote-priority",
    )
    events: list[str] = []

    class FakePriorityOwner:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def request_gpus(self, *, reason: str) -> None:
            events.append(f"request:{reason}")

        def release_gpus(self, *, reason: str) -> None:
            events.append(f"release:{reason}")

    class FakeController:
        def __init__(self, _config: Any, _scheduler: Any) -> None:
            pass

        def __enter__(self) -> "FakeController":
            events.append("controller:enter")
            return self

        def __exit__(self, *_args: Any) -> None:
            events.append("controller:exit")

    monkeypatch.setattr(
        pipeline_module,
        "PriorityGpuOwner",
        FakePriorityOwner,
    )
    monkeypatch.setattr(
        pipeline_module,
        "RemoteLayoutController",
        FakeController,
    )
    extractor = SimpleNamespace(
        routing_scheduler=object(),
        text_model_api_key=None,
        image_model_api_key=None,
    )

    with pipeline_module._remote_layout_control(
        config,
        extractor=extractor,
        database_path=tmp_path / "jobs.sqlite3",
    ):
        events.append("model:run")

    assert events == [
        "request:wdc_model_stage",
        "controller:enter",
        "model:run",
        "controller:exit",
        "release:wdc_model_stage_complete",
    ]

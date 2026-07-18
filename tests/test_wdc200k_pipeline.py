from __future__ import annotations

import json
import gzip
import os
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
SCRIPTS = ROOT / "scripts"
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
from wdc200k_assets import ImageOutcomeStore  # noqa: E402
from wdc200k_fetch import PageOutcomeStore  # noqa: E402


def _statistics_archive(input_dir: Path) -> Path:
    class_dir = input_dir / "Thing"
    class_dir.mkdir(parents=True)
    archive = class_dir / "Thing_statistics.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for subset in ("top100", "minimum3", "rest"):
            zipped.writestr(
                "table_statistics/"
                f"Thing_October2023_statistics_{subset}.csv",
                "host,number_of_rows,column_count,column_name_and_density\n"
                f"{subset}.example,2,4,\n",
            )
    return archive


def _write_selected_table(input_dir: Path) -> None:
    path = (
        input_dir
        / "Thing"
        / "Thing_top100.example_October2023.json.gz"
    )
    rows = [
        {
            "name": "Alpha",
            "State": "Texas",
            "page_url": "https://pages.example/shared",
            "image": "https://images.example/alpha.jpg",
        },
        {
            "name": "Beta",
            "State": "Ohio",
            "page_url": "https://pages.example/shared",
            "image": [
                "https://images.example/beta-1.jpg",
                "https://images.example/beta-2.jpg",
            ],
        },
    ]
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
    assert args.max_source_tables == 200_000
    assert args.max_rows_per_source_table is None
    assert args.selection_seed == 13
    assert args.web_max_retries == 0
    assert args.web_max_response_seconds == 8
    assert args.web_global_concurrency == 128
    assert args.web_per_host_concurrency == 2
    assert args.max_image_attempts_per_entity == 3
    assert args.max_images_per_entity == 3
    assert args.resume is True
    assert args.refresh_page_cache is False
    assert args.refresh_image_cache is False
    config = PipelineConfig.from_args(args)
    assert config.runtime_dir == config.work_dir / "runtime"
    assert STAGES == (
        "selection",
        "structural",
        "pages",
        "asset_planning",
        "images",
        "models",
        "materialize",
    )


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
    assert result.counters["image_request_upper_bound"] == 6
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
        value = {"Alpha": "Texas", "Beta": "Ohio"}[entity_name]
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


def _full_pipeline_config(tmp_path: Path) -> PipelineConfig:
    input_dir = tmp_path / "input"
    _statistics_archive(input_dir)
    _write_selected_table(input_dir)
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
        }

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
        registry = json.loads(
            (
                config.work_dir
                / "stage_manifests"
                / f"pipeline-{stage}.json"
            ).read_text(encoding="utf-8")
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
            assert {
                "transport_attempts": network["transport_attempts"],
                "url_completion": network["url_completion"],
                "registry_counters": registry["counters"],
            } == first_network_telemetry[stage]


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


def test_progress_url_units_use_resume_baseline_and_remain_monotonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    reporter = ProgressReporter(config)

    reporter.update(
        stage="pages",
        completed_units=5,
        total_units=10,
        rate_basis="page_urls",
    )
    clock[0] = 2.0
    reporter.update(completed_units=7)
    snapshot = reporter._snapshot()
    reporter.update(completed_units=6)
    monotonic = reporter._snapshot()

    assert snapshot["completed_units"] == 7
    assert snapshot["total_units"] == 10
    assert snapshot["rate_basis"] == "page_urls"
    assert snapshot["rates"]["units_per_second"] == pytest.approx(1.0)
    assert snapshot["eta_seconds"] == pytest.approx(3.0)
    assert monotonic["completed_units"] == 7


def test_progress_keeps_bounded_stage_samples_and_final_half_eta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        completed_units=0,
        total_units=300,
        rate_basis="page_urls",
    )
    for completed in range(1, 301):
        clock[0] = float(completed)
        reporter.update(completed_units=completed)
    reporter.update(stage="images")
    snapshot = reporter._snapshot()
    pages = snapshot["stage_telemetry"]["pages"]

    assert len(pages["samples"]) == 256
    assert pages["completed_units"] == 300
    assert pages["total_units"] == 300
    assert pages["completed_at"] == 300.0
    assert pages["eligible_final_half_samples"] > 0
    assert pages["excluded_final_half_samples"] == 1
    assert pages["max_symmetric_eta_factor"] == pytest.approx(1.0)
    assert snapshot["stage"] == "images"
    assert snapshot["completed_units"] == 0
    assert snapshot["total_units"] == 0
    assert snapshot["rate_basis"] is None


def test_progress_atomically_publishes_guarded_url_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    guarded: list[tuple[Path, int]] = []

    def guard(path: Path, estimated_bytes: int = 0) -> None:
        guarded.append((Path(path), estimated_bytes))

    reporter = ProgressReporter(
        config,
        pre_write_guard=guard,
    )
    reporter.update(
        stage="pages",
        completed_units=0,
        total_units=2,
        rate_basis="page_urls",
    )
    clock[0] = 1.0
    reporter.update(completed_units=1)
    reporter.publish()

    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert payload["stage_telemetry"]["pages"]["samples"][-1][
        "completed_units"
    ] == 1
    assert guarded
    assert not list(reporter.path.parent.glob(f".{reporter.path.name}.*.tmp"))


def test_progress_restores_bounded_samples_across_process_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        completed_units=0,
        total_units=4,
        rate_basis="page_urls",
    )
    clock[0] = 2.0
    first.update(completed_units=2)
    first.publish()

    resumed = ProgressReporter(config)
    resumed.update(
        stage="pages",
        completed_units=1,
        total_units=4,
        rate_basis="page_urls",
    )
    snapshot = resumed._snapshot()

    assert snapshot["completed_units"] == 2
    assert [
        sample["completed_units"]
        for sample in snapshot["stage_telemetry"]["pages"]["samples"]
    ] == [0, 2]


def test_progress_resume_survives_wall_clock_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [100.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    first = ProgressReporter(config)
    first.update(
        stage="pages",
        completed_units=0,
        total_units=4,
        rate_basis="page_urls",
    )
    clock[0] = 110.0
    first.update(completed_units=2)
    first.publish()

    clock[0] = 50.0
    resumed = ProgressReporter(config)
    resumed.update(
        stage="pages",
        completed_units=3,
        total_units=4,
        rate_basis="page_urls",
    )
    clock[0] = 51.0
    resumed.update(completed_units=4)
    resumed.publish()
    payload = json.loads(resumed.path.read_text(encoding="utf-8"))
    samples = payload["stage_telemetry"]["pages"]["samples"]

    assert [sample["timestamp"] for sample in samples] == sorted(
        sample["timestamp"] for sample in samples
    )
    assert payload["stage_telemetry"]["pages"]["completed_at"] >= (
        samples[-1]["timestamp"]
    )
    ProgressReporter(config)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda telemetry: telemetry["samples"][0].__setitem__(
            "timestamp", float("nan")
        ),
        lambda telemetry: telemetry.__setitem__("completed_units", 999),
        lambda telemetry: telemetry.__setitem__(
            "eligible_final_half_samples", -1
        ),
    ],
)
def test_progress_restore_rejects_tampered_unit_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: Any,
) -> None:
    config = _full_pipeline_config(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(pipeline_module.time, "time", lambda: clock[0])
    reporter = ProgressReporter(config)
    reporter.update(
        stage="pages",
        completed_units=0,
        total_units=2,
        rate_basis="page_urls",
    )
    clock[0] = 1.0
    reporter.update(completed_units=1)
    reporter.publish()
    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    tamper(payload["stage_telemetry"]["pages"])
    reporter.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="progress"):
        ProgressReporter(config)


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
        model_text_done_marker=runtime / "text-done.json",
        model_image_done_marker=runtime / "image-done.json",
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

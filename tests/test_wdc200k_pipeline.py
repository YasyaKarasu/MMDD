from __future__ import annotations

import json
import gzip
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_wdc200k_mm_joinability_dataset import (  # noqa: E402
    STAGES,
    DiskSpaceInsufficientError,
    PipelineConfig,
    invalidate_from_stage,
    parse_args,
    run_pipeline,
)
import build_wdc200k_mm_joinability_dataset as pipeline_module  # noqa: E402


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
            str(tmp_path),
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
    assert progress["disk"]["free_bytes"] >= 0
    assert "[wdc200k]" in capsys.readouterr().out
    assert not config.cache_dir.exists()
    assert not config.output_dir.exists()


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

    stale = invalidate_from_stage(config, "pages")

    assert (
        config.work_dir / "stage_manifests/pipeline-structural.json"
    ).is_file()
    assert not (
        config.work_dir / "stage_manifests/pipeline-pages.json"
    ).exists()
    assert (stale / "stage_manifests/pipeline-pages.json").read_text() == "pages"
    assert (stale / "page_jobs/state").read_text() == "page"
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

    resumed = run_pipeline(
        config,
        page_transport=page_transport,
        image_transport=image_transport,
        extractor=_PipelineExtractor(),
    )

    assert resumed.status == "complete"
    assert (page_transport.calls, image_transport.calls) == first_calls
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
    assert not (config.work_dir / "stale").exists()


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
    assert not (config.work_dir / "stale").exists()


def test_stop_after_selection_does_not_expand_structural_tables(
    tmp_path: Path,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="selection")

    result = run_pipeline(config)

    assert result.status == "stopped"
    assert result.stage == "selection"
    assert not (config.work_dir / "structural").exists()


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


def test_page_commit_rechecks_disk_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_full_pipeline_config(tmp_path), stop_after="pages")
    page_checks = 0
    original = pipeline_module._check_disk_reserve

    def fail_after_page_commit(current: PipelineConfig, stage: str) -> None:
        nonlocal page_checks
        if stage == "pages":
            page_checks += 1
            if page_checks == 2:
                raise DiskSpaceInsufficientError("synthetic reserve exhausted")
        original(current, stage)

    monkeypatch.setattr(
        pipeline_module,
        "_check_disk_reserve",
        fail_after_page_commit,
    )

    with pytest.raises(DiskSpaceInsufficientError, match="reserve exhausted"):
        run_pipeline(config, page_transport=_PipelinePageTransport())

    assert page_checks == 2


def test_dynamic_model_markers_are_forwarded_to_authoritative_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _full_pipeline_config(tmp_path)
    runtime = tmp_path / "runtime"
    config = replace(
        base,
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
    stale_outputs = tuple((config.work_dir / "stale").glob("*/final_output"))
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
        (config.work_dir / "stale").glob("*/cache/page_cache/outcomes.sqlite3")
    )
    assert len(archived) == 1
    assert (config.cache_dir / "page_cache/outcomes.sqlite3").is_file()

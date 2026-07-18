from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.wdc200k_archive as archive_module
from scripts.wdc200k_archive import (
    ArchiveConflictError,
    ArchiveCrossDeviceError,
    archive_pipeline_state,
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _archive_arguments(tmp_path: Path) -> dict[str, object]:
    work_dir = tmp_path / "work"
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    runtime_dir = work_dir / "runtime"
    return {
        "work_dir": work_dir,
        "output_dir": output_dir,
        "cache_dir": cache_dir,
        "runtime_dir": runtime_dir,
        "stages": ("selection", "pages", "images", "materialize"),
        "from_stage": "pages",
        "stage_work_paths": {
            "selection": ("selection",),
            "pages": ("page_jobs", "runtime"),
            "images": ("image_jobs",),
            "materialize": ("materialization",),
        },
        "stage_registry_paths": {
            stage: work_dir / "stage_manifests" / f"pipeline-{stage}.json"
            for stage in ("selection", "pages", "images", "materialize")
        },
        "refresh_page_cache": True,
        "refresh_image_cache": False,
    }


def test_archive_uses_root_local_stale_trees_and_keeps_runtime(
    tmp_path: Path,
) -> None:
    arguments = _archive_arguments(tmp_path)
    work_dir = arguments["work_dir"]
    output_dir = arguments["output_dir"]
    cache_dir = arguments["cache_dir"]
    runtime_dir = arguments["runtime_dir"]
    assert isinstance(work_dir, Path)
    assert isinstance(output_dir, Path)
    assert isinstance(cache_dir, Path)
    assert isinstance(runtime_dir, Path)

    _write(work_dir / "selection/keep", "upstream")
    _write(work_dir / "page_jobs/page", "page")
    _write(work_dir / "image_jobs/image", "image")
    _write(runtime_dir / "heartbeat", "live")
    _write(work_dir / "stage_manifests/pipeline-selection.json", "selection")
    _write(work_dir / "stage_manifests/pipeline-pages.json", "pages")
    _write(output_dir / "dataset_manifest.json", "output")
    _write(cache_dir / "page_cache/outcomes.sqlite3", "cache")
    _write(cache_dir / "images/keep", "image-cache")

    result = archive_pipeline_state(**arguments)

    work_stale = work_dir.parent / f".{work_dir.name}.wdc200k-stale"
    output_stale = output_dir.parent / f".{output_dir.name}.wdc200k-stale"
    cache_stale = cache_dir.parent / f".{cache_dir.name}.wdc200k-stale"
    assert (work_dir / "selection/keep").read_text() == "upstream"
    assert (runtime_dir / "heartbeat").read_text() == "live"
    assert (
        work_stale / result.transaction_id / "page_jobs/page"
    ).read_text() == "page"
    assert (
        work_stale / result.transaction_id / "image_jobs/image"
    ).read_text() == "image"
    assert (
        work_stale
        / result.transaction_id
        / "stage_manifests/pipeline-pages.json"
    ).read_text() == "pages"
    assert (
        output_stale / result.transaction_id / "root/dataset_manifest.json"
    ).read_text() == "output"
    assert (
        cache_stale
        / result.transaction_id
        / "page_cache/outcomes.sqlite3"
    ).read_text() == "cache"
    assert (cache_dir / "images/keep").read_text() == "image-cache"
    assert result.complete is True
    assert result.recovered is False
    assert result.journal_path.parent == work_dir / ".archive-transactions"
    journal = json.loads(result.journal_path.read_text(encoding="utf-8"))
    assert journal["complete"] is True
    assert all(
        item["status"] in {"complete", "absent"} for item in journal["moves"]
    )


def test_archive_recovers_after_move_failure_without_repeating_completed_moves(
    tmp_path: Path,
) -> None:
    arguments = _archive_arguments(tmp_path)
    work_dir = arguments["work_dir"]
    output_dir = arguments["output_dir"]
    assert isinstance(work_dir, Path)
    assert isinstance(output_dir, Path)
    _write(work_dir / "page_jobs/one", "one")
    _write(work_dir / "image_jobs/two", "two")
    _write(output_dir / "three", "three")

    first_calls: list[Path] = []

    def fail_second(source: Path, destination: Path) -> None:
        first_calls.append(source)
        if len(first_calls) == 2:
            raise OSError("injected move failure")
        source.replace(destination)

    with pytest.raises(OSError, match="injected move failure"):
        archive_pipeline_state(**arguments, move_path=fail_second)

    journal_paths = tuple(
        (work_dir / ".archive-transactions").glob("*.json")
    )
    assert len(journal_paths) == 1
    interrupted = json.loads(journal_paths[0].read_text(encoding="utf-8"))
    assert interrupted["complete"] is False
    completed_source = next(
        Path(item["source"])
        for item in interrupted["moves"]
        if item["status"] == "complete"
    )

    resumed_calls: list[Path] = []

    def record_move(source: Path, destination: Path) -> None:
        resumed_calls.append(source)
        source.replace(destination)

    result = archive_pipeline_state(**arguments, move_path=record_move)

    assert result.recovered is True
    assert result.journal_path == journal_paths[0]
    assert completed_source not in resumed_calls
    assert result.complete is True
    archived_files = [
        path
        for move in result.moves
        if move.status == "complete"
        for path in (
            (move.destination,)
            if move.destination.is_file()
            else tuple(move.destination.rglob("*"))
        )
        if path.is_file()
    ]
    contents = sorted(path.read_text(encoding="utf-8") for path in archived_files)
    assert contents == ["one", "three", "two"]


def test_archive_rejects_cross_device_plan_before_any_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _archive_arguments(tmp_path)
    work_dir = arguments["work_dir"]
    assert isinstance(work_dir, Path)
    source = work_dir / "page_jobs/state"
    _write(source, "page")

    real_device_id = archive_module._device_id

    def split_device(path: Path) -> int:
        if ".work.wdc200k-stale" in str(path):
            return 999_999
        return real_device_id(path)

    monkeypatch.setattr(archive_module, "_device_id", split_device)

    with pytest.raises(ArchiveCrossDeviceError, match="cross filesystems"):
        archive_pipeline_state(**arguments)

    assert source.read_text(encoding="utf-8") == "page"
    journal = next((work_dir / ".archive-transactions").glob("*.json"))
    payload = json.loads(journal.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert not any(item["status"] == "complete" for item in payload["moves"])


def test_archive_recovers_when_rename_succeeded_before_journal_update(
    tmp_path: Path,
) -> None:
    arguments = _archive_arguments(tmp_path)
    work_dir = arguments["work_dir"]
    assert isinstance(work_dir, Path)
    source = work_dir / "page_jobs/state"
    _write(source, "page")

    def move_then_fail(source_path: Path, destination: Path) -> None:
        source_path.replace(destination)
        raise OSError("crash after rename")

    with pytest.raises(OSError, match="crash after rename"):
        archive_pipeline_state(**arguments, move_path=move_then_fail)

    resumed_moves: list[Path] = []

    def record_move(source_path: Path, destination: Path) -> None:
        resumed_moves.append(source_path)
        source_path.replace(destination)

    result = archive_pipeline_state(**arguments, move_path=record_move)

    archived = next(
        move.destination
        for move in result.moves
        if move.source == source.parent
    )
    assert result.recovered is True
    assert source.parent not in resumed_moves
    assert (archived / "state").read_text(encoding="utf-8") == "page"


def test_archive_stops_on_source_destination_conflict(
    tmp_path: Path,
) -> None:
    arguments = _archive_arguments(tmp_path)
    work_dir = arguments["work_dir"]
    assert isinstance(work_dir, Path)
    source = work_dir / "page_jobs/state"
    _write(source, "source")

    calls = 0

    def fail_before_move(source_path: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        _write(destination / "state", "destination")
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        archive_pipeline_state(**arguments, move_path=fail_before_move)

    with pytest.raises(ArchiveConflictError, match="both exist"):
        archive_pipeline_state(**arguments)

    assert calls == 1
    assert source.read_text(encoding="utf-8") == "source"

#!/usr/bin/env python3
"""Recoverable, same-filesystem archival for WDC 200K pipeline state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

try:
    from wdc200k_io import GuardedWriteTracker, PreWriteGuard
except ModuleNotFoundError as error:
    if error.name != "wdc200k_io":
        raise
    import sys

    scripts_directory = str(Path(__file__).resolve().parent)
    sys.path.insert(0, scripts_directory)
    try:
        from wdc200k_io import GuardedWriteTracker, PreWriteGuard
    finally:
        sys.path.remove(scripts_directory)


JOURNAL_SCHEMA_VERSION = "wdc200k-archive-transaction-v1"
DEFAULT_PAGE_CACHE_PATHS = ("page_cache", "page_transport")
DEFAULT_IMAGE_CACHE_PATHS = ("image_cache", "images", "image_transport")
DIRECTORY_ENTRY_RESERVE_BYTES = 4096


class ArchiveError(RuntimeError):
    """Base exception for a state archival failure."""


class ArchivePlanError(ArchiveError):
    """Raised when an archival request cannot produce a safe plan."""


class ArchiveConflictError(ArchiveError):
    """Raised when source and destination both exist."""


class ArchiveCrossDeviceError(ArchiveError):
    """Raised before a move whose destination is on another filesystem."""


class ArchiveJournalError(ArchiveError):
    """Raised when durable transaction state is invalid or ambiguous."""


@dataclass(frozen=True)
class ArchiveMove:
    source: Path
    destination: Path
    root_kind: str
    status: str


@dataclass(frozen=True)
class ArchiveResult:
    transaction_id: str
    journal_path: Path
    moves: tuple[ArchiveMove, ...]
    complete: bool
    recovered: bool


MovePath = Callable[[Path, Path], None]


def _replace_path(source: Path, destination: Path) -> None:
    source.replace(destination)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    write_tracker: GuardedWriteTracker,
) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    write_tracker.before_write(len(encoded) + DIRECTORY_ENTRY_RESERVE_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(path.parent.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    committed = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        write_tracker.before_commit(0)
        os.replace(temporary, path)
        committed = True
        _fsync_directory(path.parent)
    finally:
        if not committed:
            temporary.unlink(missing_ok=True)


def _device_id(path: Path) -> int:
    candidate = Path(path)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ArchivePlanError(
                f"cannot find existing ancestor for archive path: {path}"
            )
        candidate = parent
    return candidate.stat().st_dev


def _normal_relative(value: str | Path, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ArchivePlanError(f"{label} must be a non-empty relative path: {value}")
    return relative


def _path_under(root: Path, value: str | Path, *, label: str) -> tuple[Path, Path]:
    candidate = Path(value)
    source = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as error:
        raise ArchivePlanError(f"{label} is outside {root}: {value}") from error
    if relative == Path("."):
        raise ArchivePlanError(f"{label} cannot be the archive root itself: {value}")
    return source, relative


def _stale_root(root: Path) -> Path:
    return root.parent / f".{root.name}.wdc200k-stale"


def _request_payload(
    *,
    work_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    runtime_dir: Path | None,
    stages: Sequence[str],
    from_stage: str,
    stage_work_paths: Mapping[str, Sequence[str | Path]],
    stage_registry_paths: Mapping[str, str | Path],
    refresh_page_cache: bool,
    refresh_image_cache: bool,
    page_cache_paths: Sequence[str | Path],
    image_cache_paths: Sequence[str | Path],
) -> dict[str, object]:
    return {
        "work_dir": str(work_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "runtime_dir": str(runtime_dir) if runtime_dir is not None else None,
        "stages": list(stages),
        "from_stage": from_stage,
        "stage_work_paths": {
            stage: [str(path) for path in stage_work_paths.get(stage, ())]
            for stage in stages
        },
        "stage_registry_paths": {
            stage: str(stage_registry_paths[stage])
            for stage in stages
            if stage in stage_registry_paths
        },
        "refresh_page_cache": refresh_page_cache,
        "refresh_image_cache": refresh_image_cache,
        "page_cache_paths": [str(path) for path in page_cache_paths],
        "image_cache_paths": [str(path) for path in image_cache_paths],
    }


def _request_key(request: Mapping[str, object]) -> str:
    encoded = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_transaction_id() -> str:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{timestamp}-{time.time_ns()}-{uuid.uuid4().hex[:12]}"


def _move_payload(
    *,
    source: Path,
    destination: Path,
    root_kind: str,
) -> dict[str, str]:
    return {
        "source": str(source),
        "destination": str(destination),
        "root_kind": root_kind,
        "status": "pending" if source.exists() else "absent",
    }


def _append_move(
    moves: list[dict[str, str]],
    seen_sources: set[Path],
    *,
    source: Path,
    destination: Path,
    root_kind: str,
    runtime_dir: Path | None,
    journal_dir: Path,
) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if runtime_dir is not None:
        if source == runtime_dir or source.is_relative_to(runtime_dir):
            return
        if runtime_dir.is_relative_to(source):
            raise ArchivePlanError(
                f"archive source would contain active runtime: {source}"
            )
    if source == journal_dir or journal_dir.is_relative_to(source):
        raise ArchivePlanError(
            f"archive source would contain transaction journals: {source}"
        )
    if source in seen_sources:
        return
    seen_sources.add(source)
    moves.append(
        _move_payload(
            source=source,
            destination=destination,
            root_kind=root_kind,
        )
    )


def _build_moves(
    *,
    transaction_id: str,
    work_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    runtime_dir: Path | None,
    stages: Sequence[str],
    from_stage: str,
    stage_work_paths: Mapping[str, Sequence[str | Path]],
    stage_registry_paths: Mapping[str, str | Path],
    refresh_page_cache: bool,
    refresh_image_cache: bool,
    page_cache_paths: Sequence[str | Path],
    image_cache_paths: Sequence[str | Path],
) -> list[dict[str, str]]:
    start = stages.index(from_stage)
    downstream = stages[start:]
    work_stale = _stale_root(work_dir) / transaction_id
    cache_stale = _stale_root(cache_dir) / transaction_id
    journal_dir = work_dir / ".archive-transactions"
    moves: list[dict[str, str]] = []
    seen_sources: set[Path] = set()

    for stage in downstream:
        registry_value = stage_registry_paths.get(stage)
        if registry_value is not None:
            source, relative = _path_under(
                work_dir,
                registry_value,
                label=f"{stage} registry",
            )
            _append_move(
                moves,
                seen_sources,
                source=source,
                destination=work_stale / relative,
                root_kind="work",
                runtime_dir=runtime_dir,
                journal_dir=journal_dir,
            )
        for value in stage_work_paths.get(stage, ()):
            source, relative = _path_under(
                work_dir,
                value,
                label=f"{stage} work path",
            )
            _append_move(
                moves,
                seen_sources,
                source=source,
                destination=work_stale / relative,
                root_kind="work",
                runtime_dir=runtime_dir,
                journal_dir=journal_dir,
            )

    if "materialize" in downstream:
        _append_move(
            moves,
            seen_sources,
            source=output_dir,
            destination=_stale_root(output_dir) / transaction_id / "root",
            root_kind="output",
            runtime_dir=runtime_dir,
            journal_dir=journal_dir,
        )

    cache_values: list[str | Path] = []
    if refresh_page_cache:
        cache_values.extend(page_cache_paths)
    if refresh_image_cache:
        cache_values.extend(image_cache_paths)
    for value in cache_values:
        relative = _normal_relative(value, label="cache path")
        source, relative = _path_under(
            cache_dir,
            relative,
            label="cache path",
        )
        _append_move(
            moves,
            seen_sources,
            source=source,
            destination=cache_stale / relative,
            root_kind="cache",
            runtime_dir=runtime_dir,
            journal_dir=journal_dir,
        )
    return moves


def _load_journal(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchiveJournalError(f"cannot read archive journal: {path}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != JOURNAL_SCHEMA_VERSION
        or not isinstance(payload.get("moves"), list)
    ):
        raise ArchiveJournalError(f"invalid archive journal: {path}")
    return payload


def _find_recoverable_journal(
    journal_dir: Path,
    *,
    request_key: str,
) -> tuple[Path, dict[str, object]] | None:
    if not journal_dir.is_dir():
        return None
    matching: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(journal_dir.glob("*.json")):
        payload = _load_journal(path)
        if (
            payload.get("request_key") == request_key
            and payload.get("complete") is not True
        ):
            matching.append((path, payload))
    if len(matching) > 1:
        raise ArchiveJournalError(
            f"multiple incomplete archive transactions for request: {request_key}"
        )
    return matching[0] if matching else None


def _completed_archive_journals(
    journal_dir: Path,
) -> list[tuple[Path, dict[str, object]]]:
    if not journal_dir.is_dir():
        return []
    completed: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(journal_dir.glob("*.json")):
        payload = _load_journal(path)
        if payload.get("complete") is not True:
            continue
        transaction_id = payload.get("transaction_id")
        if (
            not isinstance(transaction_id, str)
            or not transaction_id
            or Path(transaction_id).name != transaction_id
            or path.stem != transaction_id
        ):
            raise ArchiveJournalError(
                f"invalid completed archive transaction id: {path}"
            )
        completed.append((path, payload))
    return completed


def _prune_completed_archives(
    *,
    work_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> tuple[str, ...]:
    """Remove superseded rollback generations, never incomplete journals."""
    journal_dir = work_dir / ".archive-transactions"
    removed: list[str] = []
    for journal_path, payload in _completed_archive_journals(journal_dir):
        transaction_id = str(payload["transaction_id"])
        transaction_roots = (
            _stale_root(work_dir) / transaction_id,
            _stale_root(output_dir) / transaction_id,
            _stale_root(cache_dir) / transaction_id,
        )
        for root in transaction_roots:
            if root.is_symlink():
                raise ArchiveJournalError(
                    f"refusing to prune symlinked archive root: {root}"
                )
            if not root.exists():
                continue
            if not root.is_dir():
                raise ArchiveJournalError(
                    f"archive transaction root is not a directory: {root}"
                )
            shutil.rmtree(root)
            _fsync_directory(root.parent)
        journal_path.unlink()
        _fsync_directory(journal_dir)
        removed.append(transaction_id)
    return tuple(removed)


def _validate_move_payload(item: object, *, journal_path: Path) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ArchiveJournalError(f"invalid move in archive journal: {journal_path}")
    required = ("source", "destination", "root_kind", "status")
    if any(not isinstance(item.get(field), str) for field in required):
        raise ArchiveJournalError(f"invalid move in archive journal: {journal_path}")
    if item["status"] not in {"pending", "complete", "absent"}:
        raise ArchiveJournalError(f"invalid move status in archive journal: {journal_path}")
    return item


def _preflight_moves(
    payload: dict[str, object],
    *,
    journal_path: Path,
) -> bool:
    changed = False
    raw_moves = payload["moves"]
    assert isinstance(raw_moves, list)
    for raw_item in raw_moves:
        item = _validate_move_payload(raw_item, journal_path=journal_path)
        source = Path(item["source"])
        destination = Path(item["destination"])
        source_exists = source.exists()
        destination_exists = destination.exists()
        status = item["status"]
        if status == "absent":
            if source_exists or destination_exists:
                raise ArchiveConflictError(
                    f"archive path appeared after plan creation: {source}"
                )
            continue
        if status == "complete":
            if source_exists or not destination_exists:
                raise ArchiveJournalError(
                    f"completed archive move has inconsistent state: {source}"
                )
            continue
        if source_exists and destination_exists:
            raise ArchiveConflictError(
                f"archive source and destination both exist: {source} -> {destination}"
            )
        if not source_exists and destination_exists:
            item["status"] = "complete"
            changed = True
            continue
        if not source_exists:
            raise ArchiveJournalError(
                f"pending archive source and destination are both missing: {source}"
            )
        source_device = _device_id(source)
        destination_device = _device_id(destination)
        if source_device != destination_device:
            raise ArchiveCrossDeviceError(
                "archive move would cross filesystems: "
                f"{source} ({source_device}) -> {destination} ({destination_device})"
            )
    return changed


def _result(
    payload: Mapping[str, object],
    *,
    journal_path: Path,
    recovered: bool,
) -> ArchiveResult:
    moves = tuple(
        ArchiveMove(
            source=Path(item["source"]),
            destination=Path(item["destination"]),
            root_kind=str(item["root_kind"]),
            status=str(item["status"]),
        )
        for item in payload["moves"]  # type: ignore[index]
    )
    return ArchiveResult(
        transaction_id=str(payload["transaction_id"]),
        journal_path=journal_path,
        moves=moves,
        complete=payload.get("complete") is True,
        recovered=recovered,
    )


def archive_pipeline_state(
    *,
    work_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    stages: Sequence[str],
    from_stage: str,
    stage_work_paths: Mapping[str, Sequence[str | Path]],
    stage_registry_paths: Mapping[str, str | Path],
    runtime_dir: Path | None = None,
    refresh_page_cache: bool = False,
    refresh_image_cache: bool = False,
    page_cache_paths: Sequence[str | Path] = DEFAULT_PAGE_CACHE_PATHS,
    image_cache_paths: Sequence[str | Path] = DEFAULT_IMAGE_CACHE_PATHS,
    move_path: MovePath = _replace_path,
    pre_write_guard: PreWriteGuard | None = None,
) -> ArchiveResult:
    """Archive named and downstream pipeline state with durable recovery.

    Every move stays within the source root's filesystem sibling stale tree.
    Repeating an interrupted request resumes its journal before making a new
    transaction.
    """
    work_dir = Path(work_dir).resolve()
    output_dir = Path(output_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    runtime_dir = Path(runtime_dir).resolve() if runtime_dir is not None else None
    stages = tuple(stages)
    if not stages or len(set(stages)) != len(stages):
        raise ArchivePlanError("stages must be a non-empty unique sequence")
    if from_stage not in stages:
        raise ArchivePlanError(f"unknown archive stage: {from_stage}")

    journal_dir = work_dir / ".archive-transactions"
    request = _request_payload(
        work_dir=work_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        runtime_dir=runtime_dir,
        stages=stages,
        from_stage=from_stage,
        stage_work_paths=stage_work_paths,
        stage_registry_paths=stage_registry_paths,
        refresh_page_cache=refresh_page_cache,
        refresh_image_cache=refresh_image_cache,
        page_cache_paths=page_cache_paths,
        image_cache_paths=image_cache_paths,
    )
    request_key = _request_key(request)
    recoverable = _find_recoverable_journal(
        journal_dir,
        request_key=request_key,
    )
    recovered = recoverable is not None
    if recoverable is not None:
        journal_path, payload = recoverable
    else:
        transaction_id = _new_transaction_id()
        journal_path = journal_dir / f"{transaction_id}.json"
        moves = _build_moves(
            transaction_id=transaction_id,
            work_dir=work_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            runtime_dir=runtime_dir,
            stages=stages,
            from_stage=from_stage,
            stage_work_paths=stage_work_paths,
            stage_registry_paths=stage_registry_paths,
            refresh_page_cache=refresh_page_cache,
            refresh_image_cache=refresh_image_cache,
            page_cache_paths=page_cache_paths,
            image_cache_paths=image_cache_paths,
        )
        # A newly archived active generation supersedes all completed rollback
        # generations. Prune before moving it, while every active source still
        # exists; incomplete transactions remain available for recovery.
        if any(item["status"] == "pending" for item in moves):
            _prune_completed_archives(
                work_dir=work_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )
        payload = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "transaction_id": transaction_id,
            "request_key": request_key,
            "request": request,
            "moves": moves,
            "complete": False,
        }
    journal_tracker = GuardedWriteTracker(journal_path, pre_write_guard)
    if recoverable is None:
        _atomic_json(
            journal_path,
            payload,
            write_tracker=journal_tracker,
        )

    if _preflight_moves(payload, journal_path=journal_path):
        _atomic_json(
            journal_path,
            payload,
            write_tracker=journal_tracker,
        )

    raw_moves = payload["moves"]
    assert isinstance(raw_moves, list)
    move_trackers: dict[str, GuardedWriteTracker] = {}
    for raw_item in raw_moves:
        item = _validate_move_payload(raw_item, journal_path=journal_path)
        if item["status"] != "pending":
            continue
        source = Path(item["source"])
        destination = Path(item["destination"])
        root_kind = item["root_kind"]
        tracker = move_trackers.get(root_kind)
        if tracker is None:
            tracker = GuardedWriteTracker(destination, pre_write_guard)
            move_trackers[root_kind] = tracker
        tracker.before_write(DIRECTORY_ENTRY_RESERVE_BYTES)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(destination.parent.parent)
        tracker.before_commit(0)
        move_path(source, destination)
        _fsync_directory(source.parent)
        _fsync_directory(destination.parent)
        item["status"] = "complete"
        _atomic_json(
            journal_path,
            payload,
            write_tracker=journal_tracker,
        )

    payload["complete"] = True
    _atomic_json(
        journal_path,
        payload,
        write_tracker=journal_tracker,
    )
    return _result(payload, journal_path=journal_path, recovered=recovered)


__all__ = [
    "ArchiveConflictError",
    "ArchiveCrossDeviceError",
    "ArchiveError",
    "ArchiveJournalError",
    "ArchiveMove",
    "ArchivePlanError",
    "ArchiveResult",
    "archive_pipeline_state",
]

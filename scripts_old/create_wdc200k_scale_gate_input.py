#!/usr/bin/env python3
"""Create a deterministic symlink-only WDC scale-gate subcorpus."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import heapq
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    from wdc200k_selection import (
        SUBSETS,
        TableCandidate,
        read_statistics_catalog,
        stable_hash,
    )
except ModuleNotFoundError as error:
    if error.name != "wdc200k_selection":
        raise
    from scripts_old.wdc200k_selection import (
        SUBSETS,
        TableCandidate,
        read_statistics_catalog,
        stable_hash,
    )


SELECTION_MODES = ("round_robin", "global_lowest")


@dataclass(frozen=True)
class _RankedCandidate:
    key: tuple[int, str, str]
    candidate: TableCandidate
    source_path: Path

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _RankedCandidate):
            return NotImplemented
        return self.key > other.key


class _GlobalLowestCandidatePool:
    """Keep a bounded heap containing the globally lowest-ranked candidates."""

    def __init__(self, table_count: int) -> None:
        if table_count <= 0:
            raise ValueError("table_count must be positive")
        self.table_count = table_count
        self._heap: list[_RankedCandidate] = []

    @property
    def retained_count(self) -> int:
        return len(self._heap)

    def add(self, ranked: _RankedCandidate) -> None:
        if len(self._heap) < self.table_count:
            heapq.heappush(self._heap, ranked)
        elif ranked.key < self._heap[0].key:
            heapq.heapreplace(self._heap, ranked)

    def selected(self) -> list[_RankedCandidate]:
        return sorted(self._heap, key=lambda ranked: ranked.key)


def _ordered_bucket_keys(
    buckets: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    classes = sorted(
        {
            schema_class
            for schema_class, _subset in buckets
        }
    )
    return [
        (schema_class, subset)
        for subset in SUBSETS
        for schema_class in classes
        if (schema_class, subset) in buckets
    ]


def _allocate_round_robin_quotas(
    capacities: dict[tuple[str, str], int],
    *,
    table_count: int,
) -> dict[tuple[str, str], int]:
    if sum(capacities.values()) < table_count:
        raise ValueError(
            f"requested {table_count} existing tables but found "
            f"only {sum(capacities.values())}"
        )
    bucket_order = _ordered_bucket_keys(capacities)
    quotas = {key: 0 for key in bucket_order}
    remaining = table_count
    while remaining:
        added = False
        for key in bucket_order:
            if quotas[key] >= capacities[key]:
                continue
            quotas[key] += 1
            remaining -= 1
            added = True
            if not remaining:
                break
        if not added:
            raise ValueError("round-robin quota allocation exhausted capacity")
    return quotas


class _RoundRobinCandidatePool:
    """Keep a bounded low-row heap per bucket for round-robin output."""

    def __init__(self, quotas: dict[tuple[str, str], int]) -> None:
        if not quotas:
            raise ValueError("quotas must not be empty")
        self.quotas = dict(quotas)
        self.table_count = sum(quotas.values())
        self._buckets: dict[
            tuple[str, str], list[_RankedCandidate]
        ] = {key: [] for key in quotas}

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def retained_count(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def add(self, ranked: _RankedCandidate) -> None:
        candidate = ranked.candidate
        bucket_key = (candidate.schema_class, candidate.subset)
        if bucket_key not in self._buckets:
            raise ValueError("candidate introduced an unexpected bucket")
        cap = self.quotas[bucket_key]
        if cap == 0:
            return
        bucket = self._buckets[bucket_key]
        if len(bucket) < cap:
            heapq.heappush(bucket, ranked)
        elif ranked.key < bucket[0].key:
            heapq.heapreplace(bucket, ranked)

    def selected(self) -> list[_RankedCandidate]:
        bucket_order = _ordered_bucket_keys(self.quotas)
        ordered_buckets = {
            key: sorted(self._buckets[key], key=lambda ranked: ranked.key)
            for key in bucket_order
        }
        offsets = {key: 0 for key in bucket_order}
        selected: list[_RankedCandidate] = []
        while len(selected) < self.table_count:
            added = False
            for key in bucket_order:
                offset = offsets[key]
                bucket = ordered_buckets[key]
                if offset >= len(bucket):
                    continue
                selected.append(bucket[offset])
                offsets[key] = offset + 1
                added = True
                if len(selected) == self.table_count:
                    break
            if not added:
                break
        return selected


@dataclass(frozen=True)
class ScaleGateInput:
    target_dir: Path
    manifest_path: Path
    checksums_path: Path
    table_count: int
    selection_mode: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics_archives(source_dir: Path) -> tuple[Path, ...]:
    archives = []
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
        archive = class_dir / f"{class_dir.name}_statistics.zip"
        if archive.is_file():
            archives.append(archive)
    if not archives:
        raise ValueError(f"no statistics archives found under {source_dir}")
    return tuple(sorted(archives))


def _validated_source_path(
    source_dir: Path,
    candidate: TableCandidate,
) -> Path:
    relative_text = candidate.relative_path
    relative = PurePosixPath(relative_text)
    expected_name = (
        f"{candidate.schema_class}_{candidate.host}"
        "_October2023.json.gz"
    )
    expected = f"{candidate.schema_class}/{expected_name}"
    if (
        not relative_text
        or candidate.host in {"", ".", ".."}
        or "/" in candidate.host
        or "\\" in candidate.host
        or "\\" in relative_text
        or relative.is_absolute()
        or len(relative.parts) != 2
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != relative_text
        or relative_text != expected
    ):
        raise ValueError(
            f"invalid candidate relative_path: {relative_text!r}"
        )
    source_path = (
        source_dir.joinpath(*relative.parts).resolve(strict=False)
    )
    if not source_path.is_relative_to(source_dir):
        raise ValueError(
            f"candidate relative_path escapes source_dir: {relative_text!r}"
        )
    if not source_path.exists():
        return source_path
    source_path = source_path.resolve(strict=True)
    if not source_path.is_relative_to(source_dir):
        raise ValueError(
            f"candidate source resolves outside source_dir: {relative_text!r}"
        )
    if not source_path.is_file():
        raise ValueError(
            f"candidate source is not a file: {relative_text!r}"
        )
    return source_path


def _validated_destination(
    staging_dir: Path,
    relative_path: str,
) -> Path:
    relative = PurePosixPath(relative_path)
    destination = staging_dir.joinpath(*relative.parts)
    resolved = destination.parent.resolve(strict=False) / destination.name
    if not resolved.is_relative_to(staging_dir):
        raise ValueError(
            f"candidate destination escapes staging: {relative_path!r}"
        )
    return destination


def _select_candidates(
    source_dir: Path,
    staging_dir: Path,
    *,
    table_count: int,
    seed: int,
    selection_mode: str,
) -> list[_RankedCandidate]:
    index_path = staging_dir / ".candidate-index.sqlite3"
    connection = sqlite3.connect(index_path)
    capacities: dict[tuple[str, str], int] = defaultdict(int)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE candidate_paths (relative_path TEXT PRIMARY KEY)"
        )
        for archive in _statistics_archives(source_dir):
            for candidate in read_statistics_catalog(archive):
                source_path = _validated_source_path(
                    source_dir,
                    candidate,
                )
                try:
                    connection.execute(
                        "INSERT INTO candidate_paths(relative_path) VALUES (?)",
                        (candidate.relative_path,),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(
                        "duplicate candidate relative_path: "
                        f"{candidate.relative_path!r}"
                    ) from error
                # Valid statistics rows whose source gzip is absent remain
                # provisional and are skipped; malformed rows fail above.
                if not source_path.is_file():
                    continue
                capacities[
                    (candidate.schema_class, candidate.subset)
                ] += 1
        connection.commit()
        if not capacities:
            raise ValueError("no existing candidate tables were found")
        if sum(capacities.values()) < table_count:
            raise ValueError(
                f"requested {table_count} existing tables but found "
                f"only {sum(capacities.values())}"
            )
        pool: _RoundRobinCandidatePool | _GlobalLowestCandidatePool
        if selection_mode == "round_robin":
            quotas = _allocate_round_robin_quotas(
                dict(capacities),
                table_count=table_count,
            )
            pool = _RoundRobinCandidatePool(quotas)
        else:
            pool = _GlobalLowestCandidatePool(table_count)
        for archive in _statistics_archives(source_dir):
            for candidate in read_statistics_catalog(archive):
                source_path = _validated_source_path(
                    source_dir,
                    candidate,
                )
                if not source_path.is_file():
                    continue
                pool.add(
                    _RankedCandidate(
                        key=(
                            candidate.rows,
                            stable_hash(seed, candidate.relative_path),
                            candidate.relative_path,
                        ),
                        candidate=candidate,
                        source_path=source_path,
                    )
                )
    finally:
        connection.close()
    index_path.unlink()
    journal_path = index_path.with_name(index_path.name + "-journal")
    journal_path.unlink(missing_ok=True)
    selected = pool.selected()
    if len(selected) != table_count:
        raise ValueError(
            f"requested {table_count} existing tables but found "
            f"only {len(selected)}"
        )
    return selected


def _csv_bytes(records: list[TableCandidate]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=("host", "number_of_rows", "column_count"),
        lineterminator="\n",
    )
    writer.writeheader()
    for candidate in records:
        writer.writerow(
            {
                "host": candidate.host,
                "number_of_rows": candidate.rows,
                "column_count": candidate.columns,
            }
        )
    return text.getvalue().encode("utf-8")


def _write_statistics_archive(
    path: Path,
    *,
    schema_class: str,
    records: list[TableCandidate],
) -> None:
    by_subset: dict[str, list[TableCandidate]] = defaultdict(list)
    for candidate in records:
        by_subset[candidate.subset].append(candidate)
    with zipfile.ZipFile(path, "w") as zipped:
        for subset in SUBSETS:
            member = (
                f"table_statistics/{schema_class}_October2023"
                f"_statistics_{subset}.csv"
            )
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zipped.writestr(
                info,
                _csv_bytes(
                    sorted(
                        by_subset[subset],
                        key=lambda candidate: candidate.relative_path,
                    )
                ),
            )
    _fsync_file(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            _fsync_file(path)
    for directory in sorted(
        directories,
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        _fsync_directory(directory)


def _write_manifest(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _write_checksums(
    path: Path,
    payload: dict[str, object],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_staging(
    staging_dir: Path,
    *,
    selected: list[_RankedCandidate],
    manifest_path: Path,
    checksums_path: Path,
    archive_paths: list[Path],
) -> None:
    if (staging_dir / ".candidate-index.sqlite3").exists():
        raise ValueError("candidate index must not be published")
    manifest_records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(manifest_records) != len(selected):
        raise ValueError("manifest record count does not match selection")
    selected_by_path = {
        ranked.candidate.relative_path: ranked
        for ranked in selected
    }
    manifest_targets = [str(record["target"]) for record in manifest_records]
    if (
        len(set(manifest_targets)) != len(manifest_targets)
        or set(manifest_targets) != set(selected_by_path)
    ):
        raise ValueError("manifest targets do not match unique selection")
    manifest_by_target = {
        str(record["target"]): record
        for record in manifest_records
    }
    for relative_path, ranked in selected_by_path.items():
        manifest_record = manifest_by_target[relative_path]
        if (
            manifest_record.get("source") != str(ranked.source_path)
            or manifest_record.get("source_sha256")
            != _sha256_path(ranked.source_path)
        ):
            raise ValueError(
                f"manifest source validation failed: {relative_path}"
            )
        destination = _validated_destination(
            staging_dir,
            relative_path,
        )
        if not destination.is_symlink():
            raise ValueError(f"missing staged symlink: {relative_path}")
        if destination.resolve(strict=True) != ranked.source_path:
            raise ValueError(f"staged symlink target mismatch: {relative_path}")

    archived_paths = []
    for archive_path in archive_paths:
        archived_paths.extend(
            candidate.relative_path
            for candidate in read_statistics_catalog(archive_path)
        )
    if (
        len(archived_paths) != len(selected)
        or set(archived_paths) != set(selected_by_path)
    ):
        raise ValueError("filtered statistics archives do not match selection")

    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    expected_files = {
        str(path.relative_to(staging_dir)): _sha256_path(path)
        for path in [manifest_path, *archive_paths]
    }
    if checksums.get("files") != expected_files:
        raise ValueError("generated file checksums do not validate")


def _target_identity(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without replacing any concurrently created path."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise RuntimeError(
            "atomic no-clobber publish is unavailable: libc renameat2 missing"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ValueError(
            f"target appeared during atomic publish: {destination}"
        )
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise RuntimeError(
            "atomic no-clobber publish is unavailable: "
            f"renameat2 failed with errno {error_number}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _publish_staging(
    staging_dir: Path,
    target_dir: Path,
    *,
    initial_target_identity: tuple[int, int] | None,
) -> None:
    if target_dir.is_symlink():
        raise ValueError(f"target became a symlink: {target_dir}")
    current_identity = _target_identity(target_dir)
    if initial_target_identity is None:
        if current_identity is not None or os.path.lexists(target_dir):
            raise ValueError(f"target appeared during build: {target_dir}")
    else:
        if current_identity != initial_target_identity:
            raise ValueError(f"target changed during build: {target_dir}")
        if not target_dir.is_dir() or any(target_dir.iterdir()):
            raise ValueError(
                f"target became non-empty during build: {target_dir}"
            )
        target_dir.rmdir()
    if os.path.lexists(target_dir):
        raise ValueError(f"target appeared during publish: {target_dir}")
    _rename_noreplace(staging_dir, target_dir)
    _fsync_directory(target_dir.parent)


def create_scale_gate_input(
    *,
    source_dir: Path,
    target_dir: Path,
    table_count: int,
    seed: int = 13,
    selection_mode: str = "round_robin",
) -> ScaleGateInput:
    """Build an exact-size real-table subcorpus without copying source data."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(
            f"selection_mode must be one of {SELECTION_MODES}: "
            f"{selection_mode!r}"
        )
    source_dir = source_dir.absolute().resolve()
    target_input = target_dir.absolute()
    if target_input.is_symlink():
        raise ValueError(f"target must not be a symlink: {target_input}")
    target_dir = target_input.resolve(strict=False)
    if table_count <= 0:
        raise ValueError("table_count must be positive")
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    if (
        target_dir == source_dir
        or target_dir.is_relative_to(source_dir)
        or source_dir.is_relative_to(target_dir)
    ):
        raise ValueError(
            "source and target directories must not overlap: "
            f"{source_dir} and {target_dir}"
        )
    if target_dir.exists():
        if not target_dir.is_dir():
            raise ValueError(f"target must be a directory: {target_dir}")
        if any(target_dir.iterdir()):
            raise ValueError(f"target directory must be empty: {target_dir}")
    initial_target_identity = _target_identity(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    resolved_after_parent_create = target_input.resolve(strict=False)
    if resolved_after_parent_create != target_dir:
        raise ValueError("target resolution changed while preparing parent")
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{target_dir.name}.",
            suffix=".tmp",
            dir=target_dir.parent,
        )
    ).resolve()
    try:
        selected = _select_candidates(
            source_dir,
            staging_dir,
            table_count=table_count,
            seed=seed,
            selection_mode=selection_mode,
        )
        manifest_records = []
        for ranked in selected:
            candidate = ranked.candidate
            manifest_records.append(
                {
                    "schema_class": candidate.schema_class,
                    "subset": candidate.subset,
                    "host": candidate.host,
                    "rows": candidate.rows,
                    "columns": candidate.columns,
                    "source": str(ranked.source_path),
                    "target": candidate.relative_path,
                    "source_sha256": _sha256_path(ranked.source_path),
                }
            )

        selected_by_class: dict[str, list[TableCandidate]] = defaultdict(list)
        for ranked in selected:
            candidate = ranked.candidate
            selected_by_class[candidate.schema_class].append(candidate)
            target_path = _validated_destination(
                staging_dir,
                candidate.relative_path,
            )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(ranked.source_path, target_path)

        archive_paths = []
        for schema_class, records in sorted(selected_by_class.items()):
            archive_path = (
                staging_dir
                / schema_class
                / f"{schema_class}_statistics.zip"
            )
            _write_statistics_archive(
                archive_path,
                schema_class=schema_class,
                records=records,
            )
            archive_paths.append(archive_path)

        manifest_path = staging_dir / "scale_gate_manifest.jsonl"
        _write_manifest(manifest_path, manifest_records)
        checksums_path = staging_dir / "scale_gate_checksums.json"
        checksums = {
            "schema_version": "wdc200k-scale-gate-input-v1",
            "source_dir": str(source_dir),
            "table_count": table_count,
            "seed": seed,
            "selection_mode": selection_mode,
            "files": {
                str(path.relative_to(staging_dir)): _sha256_path(path)
                for path in [manifest_path, *archive_paths]
            },
        }
        _write_checksums(checksums_path, checksums)
        _fsync_tree(staging_dir)
        _validate_staging(
            staging_dir,
            selected=selected,
            manifest_path=manifest_path,
            checksums_path=checksums_path,
            archive_paths=archive_paths,
        )
        _publish_staging(
            staging_dir,
            target_dir,
            initial_target_identity=initial_target_identity,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return ScaleGateInput(
        target_dir=target_dir,
        manifest_path=target_dir / "scale_gate_manifest.jsonl",
        checksums_path=target_dir / "scale_gate_checksums.json",
        table_count=table_count,
        selection_mode=selection_mode,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an exact-size, symlink-only real WDC subcorpus for "
            "100/1000-table scale gates."
        )
    )
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--target_dir", type=Path, required=True)
    parser.add_argument("--table_count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--selection_mode",
        choices=SELECTION_MODES,
        default="round_robin",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = create_scale_gate_input(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        table_count=args.table_count,
        seed=args.seed,
        selection_mode=args.selection_mode,
    )
    print(
        json.dumps(
            {
                "target_dir": str(result.target_dir),
                "manifest": str(result.manifest_path),
                "checksums": str(result.checksums_path),
                "table_count": result.table_count,
                "selection_mode": result.selection_mode,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a deterministic symlink-only WDC scale-gate subcorpus."""

from __future__ import annotations

import argparse
import csv
import heapq
import hashlib
import io
import json
import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

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
    from scripts.wdc200k_selection import (
        SUBSETS,
        TableCandidate,
        read_statistics_catalog,
        stable_hash,
    )


@dataclass(frozen=True)
class _RankedCandidate:
    key: tuple[int, str, str]
    candidate: TableCandidate
    source_path: Path

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _RankedCandidate):
            return NotImplemented
        return self.key > other.key


@dataclass(frozen=True)
class ScaleGateInput:
    target_dir: Path
    manifest_path: Path
    checksums_path: Path
    table_count: int


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


def _select_candidates(
    source_dir: Path,
    *,
    table_count: int,
    seed: int,
) -> list[_RankedCandidate]:
    buckets: dict[tuple[str, str], list[_RankedCandidate]] = defaultdict(list)
    for archive in _statistics_archives(source_dir):
        for candidate in read_statistics_catalog(archive):
            source_path = source_dir / candidate.relative_path
            if not source_path.is_file():
                continue
            bucket = buckets[(candidate.schema_class, candidate.subset)]
            ranked = _RankedCandidate(
                key=(
                    candidate.rows,
                    stable_hash(seed, candidate.relative_path),
                    candidate.relative_path,
                ),
                candidate=candidate,
                source_path=source_path.resolve(),
            )
            if len(bucket) < table_count:
                heapq.heappush(bucket, ranked)
            elif ranked.key < bucket[0].key:
                heapq.heapreplace(bucket, ranked)

    classes = sorted({schema_class for schema_class, _subset in buckets})
    bucket_order = [
        (schema_class, subset)
        for subset in SUBSETS
        for schema_class in classes
        if buckets.get((schema_class, subset))
    ]
    ordered_buckets = {
        key: sorted(buckets[key], key=lambda ranked: ranked.key)
        for key in bucket_order
    }
    offsets = {key: 0 for key in bucket_order}
    selected: list[_RankedCandidate] = []
    while len(selected) < table_count:
        added = False
        for key in bucket_order:
            offset = offsets[key]
            bucket = ordered_buckets[key]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[key] = offset + 1
            added = True
            if len(selected) == table_count:
                break
        if not added:
            break
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


def create_scale_gate_input(
    *,
    source_dir: Path,
    target_dir: Path,
    table_count: int,
    seed: int = 13,
) -> ScaleGateInput:
    """Build an exact-size real-table subcorpus without copying source data."""
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

    selected = _select_candidates(
        source_dir,
        table_count=table_count,
        seed=seed,
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

    target_dir.mkdir(parents=True, exist_ok=True)
    selected_by_class: dict[str, list[TableCandidate]] = defaultdict(list)
    for ranked in selected:
        candidate = ranked.candidate
        selected_by_class[candidate.schema_class].append(candidate)
        target_path = target_dir / candidate.relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(ranked.source_path, target_path)

    archive_paths = []
    for schema_class, records in sorted(selected_by_class.items()):
        archive_path = (
            target_dir / schema_class / f"{schema_class}_statistics.zip"
        )
        _write_statistics_archive(
            archive_path,
            schema_class=schema_class,
            records=records,
        )
        archive_paths.append(archive_path)

    manifest_path = target_dir / "scale_gate_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    checksums_path = target_dir / "scale_gate_checksums.json"
    checksums = {
        "schema_version": "wdc200k-scale-gate-input-v1",
        "source_dir": str(source_dir),
        "table_count": table_count,
        "seed": seed,
        "files": {
            str(path.relative_to(target_dir)): _sha256_path(path)
            for path in [manifest_path, *archive_paths]
        },
    }
    checksums_path.write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ScaleGateInput(
        target_dir=target_dir,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        table_count=table_count,
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = create_scale_gate_input(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        table_count=args.table_count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "target_dir": str(result.target_dir),
                "manifest": str(result.manifest_path),
                "checksums": str(result.checksums_path),
                "table_count": result.table_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

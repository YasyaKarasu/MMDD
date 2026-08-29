#!/usr/bin/env python
"""Atomically merge Teacher-feature staging directories into one cache."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from cache_stage1_features import TEACHER_MANIFEST, _completed_records


def _copy_feature(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _install_feature(source: Path, destination: Path, *, move: bool) -> None:
    if not move:
        _copy_feature(source, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = Path(args.cache_dir)
    manifest_path = cache_dir / TEACHER_MANIFEST
    merged = _completed_records(manifest_path)
    added = 0
    skipped = 0
    excluded = set()
    for value in getattr(args, "exclude_object_ids", []):
        with Path(value).open(encoding="utf-8") as handle:
            excluded.update(
                str(json.loads(line)["object_id"])
                for line in handle
                if line.strip()
            )
    for value in args.staging_dirs:
        staging_dir = Path(value)
        for object_id, record in _completed_records(
            staging_dir / TEACHER_MANIFEST
        ).items():
            if object_id in excluded:
                skipped += 1
                continue
            if object_id in merged:
                if merged[object_id] != record:
                    raise ValueError(
                        f"{object_id}: staged Teacher record conflicts with the main cache"
                    )
                skipped += 1
                continue
            relative_path = Path(str(record["teacher_feature_path"]))
            source = staging_dir / relative_path
            if not source.is_file():
                raise FileNotFoundError(f"Staging manifest references missing {source}")
            _install_feature(
                source,
                cache_dir / relative_path,
                move=bool(getattr(args, "move", False)),
            )
            merged[object_id] = record
            added += 1

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in merged.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    summary = {
        "cache_dir": str(cache_dir),
        "staging_dirs": args.staging_dirs,
        "teacher_objects_added": added,
        "teacher_objects_skipped": skipped,
        "teacher_objects_total": len(merged),
        "excluded_objects": len(excluded),
        "moved": bool(getattr(args, "move", False)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--staging-dirs", nargs="+", required=True)
    parser.add_argument("--exclude-object-ids", nargs="*", default=[])
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move staged feature files into a cache on the same filesystem.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

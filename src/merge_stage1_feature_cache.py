#!/usr/bin/env python
"""Hard-link staged base and Teacher features into one cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cache_stage1_features import TEACHER_MANIFEST, _completed_records

TIERS = (
    ("manifest.jsonl", "feature_path"),
    (TEACHER_MANIFEST, "teacher_feature_path"),
)


def _link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if not destination.samefile(source):
            raise ValueError(f"Existing feature conflicts with staged file: {destination}")


def merge(cache_dir: Path, staging_dirs: list[Path]) -> dict[str, Any]:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"format_version": 1, "cache_dir": str(cache_dir.resolve())}
    for staging_dir in staging_dirs:
        staged_metadata = json.loads(
            (staging_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if staged_metadata != metadata:
            raise ValueError(f"{staging_dir}: cache metadata differs")

    for manifest_name, path_field in TIERS:
        destination_manifest = cache_dir / manifest_name
        merged = _completed_records(destination_manifest)
        added = 0
        skipped = 0
        for staging_dir in staging_dirs:
            for object_id, record in _completed_records(staging_dir / manifest_name).items():
                if object_id in merged:
                    if merged[object_id] != record:
                        raise ValueError(f"{object_id}: staged record conflicts with cache")
                    skipped += 1
                    continue
                relative_path = Path(str(record[path_field]))
                _link(staging_dir / relative_path, cache_dir / relative_path)
                merged[object_id] = record
                added += 1
        temporary = destination_manifest.with_suffix(destination_manifest.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in merged.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(destination_manifest)
        summary[manifest_name] = {
            "added": added,
            "skipped": skipped,
            "total": len(merged),
        }
    summary["staging_dirs"] = [str(path.resolve()) for path in staging_dirs]
    (cache_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--staging-dirs", type=Path, nargs="+", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge(args.cache_dir.resolve(), [path.resolve() for path in args.staging_dirs])

#!/usr/bin/env python
"""Split one mixed Stage-1 corpus using authoritative per-lake object IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence


LAKE_NAME = re.compile(r"[a-z0-9_]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_ids(path: Path) -> set[str]:
    object_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in object_ids:
                raise ValueError(f"{path}:{line_number}: duplicate object_id {object_id!r}")
            object_ids.add(object_id)
    if not object_ids:
        raise ValueError(f"{path}: corpus is empty")
    return object_ids


def parse_lake(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if separator != "=" or not LAKE_NAME.fullmatch(name) or not raw_path:
        raise argparse.ArgumentTypeError("lakes must use NAME=SOURCE_CORPUS")
    return name, Path(raw_path)


def split_corpus(
    mixed_corpus: Path,
    lakes: Sequence[tuple[str, Path]],
    output_dir: Path,
) -> dict[str, object]:
    """Write one mixed-corpus subset per lake after proving an exact partition."""

    if len(lakes) < 2:
        raise ValueError("at least two lakes are required")
    names = [name for name, _path in lakes]
    if len(set(names)) != len(names):
        raise ValueError("lake names must be unique")

    lake_ids: dict[str, set[str]] = {}
    owner: dict[str, str] = {}
    for name, source in lakes:
        ids = _object_ids(source)
        overlap = ids.intersection(owner)
        if overlap:
            example = min(overlap)
            raise ValueError(
                f"lake corpora overlap at {example!r}: {owner[example]!r} and {name!r}"
            )
        lake_ids[name] = ids
        owner.update((object_id, name) for object_id in ids)

    records: dict[str, list[str]] = {name: [] for name in names}
    mixed_ids: set[str] = set()
    with mixed_corpus.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id in mixed_ids:
                raise ValueError(
                    f"{mixed_corpus}:{line_number}: duplicate object_id {object_id!r}"
                )
            mixed_ids.add(object_id)
            lake = owner.get(object_id)
            if lake is None:
                raise ValueError(
                    f"{mixed_corpus}:{line_number}: object {object_id!r} has no lake owner"
                )
            records[lake].append(line if line.endswith("\n") else line + "\n")

    missing = set(owner).difference(mixed_ids)
    if missing:
        raise ValueError(
            f"mixed corpus is missing {len(missing)} source objects; example={min(missing)!r}"
        )
    if len(mixed_ids) != sum(len(ids) for ids in lake_ids.values()):
        raise AssertionError("lake partition count does not match the mixed corpus")

    output_dir.mkdir(parents=True, exist_ok=True)
    lake_summary: dict[str, dict[str, object]] = {}
    for name, source in lakes:
        output = output_dir / f"{name}_corpus.jsonl"
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.writelines(records[name])
        temporary.replace(output)
        lake_summary[name] = {
            "source_corpus": str(source.resolve()),
            "source_sha256": _sha256(source),
            "output_corpus": str(output.resolve()),
            "output_sha256": _sha256(output),
            "objects": len(records[name]),
        }

    summary: dict[str, object] = {
        "format_version": 1,
        "mixed_corpus": str(mixed_corpus.resolve()),
        "mixed_corpus_sha256": _sha256(mixed_corpus),
        "mixed_objects": len(mixed_ids),
        "partition_exact": True,
        "pairwise_disjoint": True,
        "lakes": lake_summary,
    }
    summary_path = output_dir / "split_summary.json"
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-corpus", type=Path, required=True)
    parser.add_argument("--lake", type=parse_lake, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = split_corpus(arguments.mixed_corpus, arguments.lake, arguments.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))

#!/usr/bin/env python
"""Materialize exact text/image content keys for G5 evidence deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from mmdd_progress import progress
from mmdd_stage1.artifacts import checkpoint_fingerprint
from mmdd_stage1.features import normalize_object_type


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_key(record: dict[str, Any], input_root: Path) -> str | None:
    """Return an exact-content key for an evidence object."""

    object_type = normalize_object_type(str(record["object_type"]))
    if object_type == "text":
        digest = hashlib.sha256(str(record.get("text", "")).encode("utf-8"))
    elif object_type == "image":
        image_path = Path(str(record["image"]))
        if not image_path.is_absolute():
            image_path = input_root / image_path
        digest_value = _file_sha256(image_path.resolve())
        return f"image:{digest_value}"
    else:
        return None
    return f"text:{digest.hexdigest()}"


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    with (
        input_path.open(encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(
            progress(source, desc="Hash evidence content", unit="object"), 1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number}: expected a JSON object")
            key = content_key(record, input_path.parent)
            if key is None:
                continue
            object_id = str(record["object_id"])
            object_type = normalize_object_type(str(record["object_type"]))
            destination.write(
                json.dumps(
                    {
                        "object_id": object_id,
                        "object_type": object_type,
                        "content_key": key,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts[object_type] += 1
            content_counts[key] += 1
    temporary.replace(output_path)
    duplicate_groups = [count for count in content_counts.values() if count > 1]
    metadata = {
        "format_version": 1,
        "definition": "SHA-256 of exact UTF-8 text or exact image file bytes",
        "input": str(input_path),
        "input_sha256": checkpoint_fingerprint(input_path),
        "output": str(output_path),
        "output_sha256": checkpoint_fingerprint(output_path),
        "objects": sum(counts.values()),
        "objects_by_type": dict(sorted(counts.items())),
        "duplicate_content_groups": len(duplicate_groups),
        "objects_in_duplicate_groups": sum(duplicate_groups),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(Path(args.objects), Path(args.output))

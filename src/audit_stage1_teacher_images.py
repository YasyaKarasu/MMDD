#!/usr/bin/env python
"""Find unreadable or unsafe images before caching Teacher hidden states."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from PIL import Image

from cache_stage1_features import _resolve_image


def _selected_ids(paths: list[str]) -> set[str]:
    selected = set()
    for value in paths:
        with Path(value).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    selected.add(str(json.loads(line)["object_id"]))
    return selected


def _existing_invalid(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["object_id"])] = record
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_jsonl).resolve()
    selected = _selected_ids(args.selected_ids)
    output = Path(args.output)
    invalid = _existing_invalid(output)
    newly_invalid = 0
    selected_images = 0
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            object_id = str(record["object_id"])
            if object_id not in selected or record.get("object_type") != "image":
                continue
            selected_images += 1
            try:
                image_path = _resolve_image(record, input_path.parent)
                assert image_path is not None
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(image_path) as image:
                        image.load()
            except Exception as error:
                newly_invalid += object_id not in invalid
                invalid[object_id] = {
                    "object_id": object_id,
                    "input_line": line_number,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for object_id in sorted(invalid):
            record = invalid[object_id]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    summary = {
        "selected_objects": len(selected),
        "selected_images": selected_images,
        "new_invalid_images": newly_invalid,
        "invalid_images": len(invalid),
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--selected-ids", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return " ".join(html.unescape(str(value)).replace("\xa0", " ").split())


def normalize(value: Any) -> str:
    text = clean_text(value).casefold().replace("_", " ")
    return " ".join(re.sub(r"[^\w.%+-]+", " ", text).split())


def values_match(predicted: Any, expected: Any) -> bool:
    left, right = normalize(predicted), normalize(expected)
    if not left or not right:
        return False
    if left == right:
        return True
    try:
        return math.isclose(float(left.replace(",", "")), float(right.replace(",", "")))
    except ValueError:
        return min(len(left), len(right)) >= 4 and (left in right or right in left)


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def get_cell(row: dict[str, Any], column_index: int) -> dict[str, Any]:
    for cell in row["cells"]:
        if cell["column_index"] == column_index:
            return cell
    return {}


def get_column_name(table: dict[str, Any], column_index: int) -> str:
    return clean_text(table["columns"][column_index]["column_name"])


def source_splits(
    tables: list[dict[str, Any]],
    *,
    split_by: str,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    if split_by != "source_table_id":
        raise ValueError("split_by must be 'source_table_id'")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must have a positive sum")
    train_ratio, dev_ratio, _ = (ratio / ratio_sum for ratio in ratios)

    groups: dict[str, list[str]] = {}
    for table in tables:
        source_id = table["source_table_id"]
        groups.setdefault(source_id, []).append(source_id)

    keys = list(groups)
    random.Random(seed).shuffle(keys)
    train_end = int(len(keys) * train_ratio)
    dev_end = train_end + int(len(keys) * dev_ratio)
    split_keys = {
        "train": keys[:train_end],
        "dev": keys[train_end:dev_end],
        "test": keys[dev_end:],
    }

    split_of: dict[str, str] = {}
    splits: dict[str, Any] = {}
    for split, selected_keys in split_keys.items():
        source_ids = sorted(
            source_id for key in selected_keys for source_id in groups[key]
        )
        splits[split] = {
            "source_table_ids": source_ids,
            "query_table_ids": [],
            "data_lake_table_ids": [],
        }
        split_of.update({source_id: split for source_id in source_ids})
    splits["split_key"] = split_by
    return splits, split_of

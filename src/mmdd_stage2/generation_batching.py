"""R5b modality buckets with an explicit positive batch budget."""
from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any
import hashlib


def task_sort(task: dict[str, Any]) -> tuple:
    # Same key/arithmetic as the inherited tasks.task_sort implementation.
    value = json.dumps([task["query_rows"], task["column_name"], task["evidence_ids"]],
                       sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    fallback = hashlib.sha256(value.encode()).hexdigest()
    return (task["modality"], task["image_count"],
            len(json.dumps(task["query_rows"], ensure_ascii=False)) // 256,
            task.get("semantic_input_key", fallback))


def microbatches(tasks: Sequence[dict[str, Any]], batch: int) -> Iterator[list[dict[str, Any]]]:
    """Keep every request intact and preserve the original modality buckets."""
    if batch <= 0:
        raise ValueError("Batch size must be positive")
    bucket: list[dict[str, Any]] = []
    key = None
    for task in sorted(tasks, key=task_sort):
        next_key = (task["modality"], task["image_count"])
        if bucket and (next_key != key or len(bucket) == batch):
            yield bucket
            bucket = []
        key = next_key
        bucket.append(task)
    if bucket:
        yield bucket

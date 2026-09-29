from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.generation_batching import microbatches


def original_key(t):
    value = [t["query_rows"], t["column_name"], t["evidence_ids"]]
    fallback = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return (t["modality"], t["image_count"], len(json.dumps(t["query_rows"], ensure_ascii=False)) // 256,
            t.get("semantic_input_key", fallback))


def test_batch4_matches_inherited_order_and_larger_batches_preserve_inputs():
    tasks = []
    for i in range(103):
        images = i % 3
        tasks.append({"task_id": str(i), "modality": "text" if not images else "mixed", "image_count": images,
                      "query_rows": [{"query_row_id": i % 5, "cells": [{"text": "实体" * (1 + i)}]}],
                      "column_name": "年份", "evidence_ids": ["e" + str(i)]})
    tasks[20]["semantic_input_key"] = "known-key"
    random.Random(13).shuffle(tasks)
    expected = sorted(tasks, key=original_key)
    for batch in (1, 4, 8, 16, 32):
        buckets = list(microbatches(tasks, batch))
        assert [t for bucket in buckets for t in bucket] == expected
        assert all(0 < len(bucket) <= batch for bucket in buckets)
        assert all(len({(t["modality"], t["image_count"]) for t in bucket}) == 1 for bucket in buckets)
    assert len(list(microbatches(tasks, 16))) < len(list(microbatches(tasks, 4)))
    assert list(microbatches([], 4)) == []


def test_nonpositive_generation_batch_is_rejected():
    with pytest.raises(ValueError):
        list(microbatches([], 0))

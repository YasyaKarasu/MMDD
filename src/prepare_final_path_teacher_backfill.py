#!/usr/bin/env python3
"""Prepare and verify exact frozen-Teacher hidden-state backfill shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image

from cache_stage1_features import _source_fingerprint
from evaluate_final_path_rerank import (
    FEATURES,
    ROOT,
    endpoints,
    file_record,
    teacher_feature_paths,
    validate_and_collect_pairs,
)
from mmdd_stage1.features import FeatureStore


OUT = ROOT / "work/final_rerank_20260916/teacher_hidden_backfill"
SOURCE = ROOT / "work/stage1_optimization_r10_20260907/stage1_data/stage1_objects.jsonl"
MODEL_CONFIG = ROOT / "hf_models/Qwen3-VL-Embedding-8B/config.json"


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cost(row: dict[str, Any]) -> int:
    if row["object_type"] == "text":
        return max(len(str(row.get("text") or "")), 1)
    image_path = Path(str(row["image"]))
    if not image_path.is_absolute():
        image_path = (SOURCE.parent / image_path).resolve()
    try:
        with Image.open(image_path) as image:
            return max(min(image.width * image.height, 4_000_000), 1)
    except (OSError, Image.DecompressionBombError):
        return 1


def prepare() -> dict[str, Any]:
    required_pairs: set[tuple[str, str]] = set()
    endpoint_counts = {}
    for endpoint in endpoints("all"):
        pairs, counts = validate_and_collect_pairs(endpoint)
        required_pairs.update(pairs)
        endpoint_counts[endpoint.endpoint] = counts
    required_objects = {object_id for pair in required_pairs for object_id in pair}

    existing_paths = [
        path
        for path in teacher_feature_paths()
        if OUT not in path.resolve().parents and path.resolve() != OUT
    ]
    store = FeatureStore.from_path(FEATURES, teacher_paths=existing_paths)
    missing = {
        object_id
        for object_id in required_objects
        if not store.has_teacher_features(object_id)
    }

    base = {
        str(row["object_id"]): row
        for row in read_rows(FEATURES / "manifest.jsonl")
        if str(row["object_id"]) in missing
    }
    source = {
        str(row["object_id"]): row
        for row in read_rows(SOURCE)
        if str(row["object_id"]) in missing
    }
    if set(base) != missing or set(source) != missing:
        raise ValueError("Frozen base manifest/source does not cover every missing object")
    for object_id in sorted(missing):
        row = source[object_id]
        if str(row["object_type"]) not in {"text", "image"}:
            raise ValueError(f"Unsupported backfill object type: {object_id}")
        if str(base[object_id]["object_type"]) != str(row["object_type"]):
            raise ValueError(f"Base/source object type mismatch: {object_id}")
        if str(base[object_id]["source_fingerprint"]) != _source_fingerprint(row):
            raise ValueError(f"Base/source fingerprint mismatch: {object_id}")

    shards: list[list[dict[str, Any]]] = [[], []]
    shard_costs = [0, 0]
    for row in sorted(source.values(), key=lambda value: (-_cost(value), value["object_id"])):
        shard = min(range(2), key=lambda index: (shard_costs[index], index))
        shards[shard].append(row)
        shard_costs[shard] += _cost(row)

    shard_records = []
    for shard, values in enumerate(shards):
        values.sort(key=lambda row: str(row["object_id"]))
        input_path = OUT / f"gpu{shard}/inputs.jsonl"
        write_rows(input_path, values)
        shard_records.append(
            {
                "shard": shard,
                "objects": len(values),
                "estimated_cost": shard_costs[shard],
                "by_type": dict(Counter(str(row["object_type"]) for row in values)),
                "inputs": file_record(input_path),
                "output_dir": str((OUT / f"gpu{shard}").resolve()),
            }
        )

    result = {
        "status": "prepared",
        "policy": (
            "Frozen-Qwen hidden-state supplement only; canonical embeddings, fixed C100, "
            "and retained path multisets remain unchanged"
        ),
        "required_pairs": len(required_pairs),
        "required_objects": len(required_objects),
        "missing_objects": len(missing),
        "missing_by_type": dict(Counter(str(row["object_type"]) for row in source.values())),
        "missing_object_ids": sorted(missing),
        "source_fingerprints": {
            object_id: str(base[object_id]["source_fingerprint"])
            for object_id in sorted(missing)
        },
        "source_fingerprints_match": True,
        "endpoint_counts": endpoint_counts,
        "source": file_record(SOURCE),
        "base_manifest": file_record(FEATURES / "manifest.jsonl"),
        "existing_teacher_manifests": [
            file_record(path / "teacher_manifest.jsonl") for path in existing_paths
        ],
        "model_config": file_record(MODEL_CONFIG),
        "backfill_runner": file_record(ROOT / "src/backfill_stage1_teacher.py"),
        "shards": shard_records,
    }
    write_json(OUT / "PREFLIGHT.json", result)
    return result


def verify() -> dict[str, Any]:
    torch.set_num_threads(2)
    preflight = json.loads((OUT / "PREFLIGHT.json").read_text(encoding="utf-8"))
    expected = set(preflight["missing_object_ids"])
    records: dict[str, tuple[dict[str, Any], Path]] = {}
    manifests = []
    for shard in range(2):
        root = OUT / f"gpu{shard}"
        manifest = root / "teacher_manifest.jsonl"
        manifests.append(file_record(manifest))
        for row in read_rows(manifest):
            object_id = str(row["object_id"])
            if object_id in records:
                raise ValueError(f"Duplicate backfilled object: {object_id}")
            if row["source_fingerprint"] != preflight["source_fingerprints"][object_id]:
                raise ValueError(f"Backfill source fingerprint changed: {object_id}")
            records[object_id] = (row, root)
    if set(records) != expected:
        raise ValueError("Backfill manifests do not exactly cover the preflight missing set")

    checks = []
    for index, object_id in enumerate(sorted(expected), 1):
        row, root = records[object_id]
        path = root / row["teacher_feature_path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        hidden = payload.get("hidden_states")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 2
            or hidden.shape[1] != 4096
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError(f"Invalid Teacher hidden states: {object_id}")
        checks.append(
            {
                "object_id": object_id,
                "object_type": row["object_type"],
                "shape": list(hidden.shape),
                "feature": file_record(path),
            }
        )
        if index % 250 == 0:
            print(json.dumps({"verified": index, "total": len(expected)}), flush=True)

    result = {
        "status": "completed",
        "objects": len(checks),
        "by_type": dict(Counter(row["object_type"] for row, _root in records.values())),
        "manifests": manifests,
        "preflight": file_record(OUT / "PREFLIGHT.json"),
        "model_config": file_record(MODEL_CONFIG),
        "backfill_runner": file_record(ROOT / "src/backfill_stage1_teacher.py"),
        "checks": checks,
        "base_embedding_manifest_unchanged": True,
        "source_fingerprints_match": True,
    }
    write_json(OUT / "BACKFILL_RECEIPT.json", result)
    return {key: value for key, value in result.items() if key != "checks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify() if args.verify else prepare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

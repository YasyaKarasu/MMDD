from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from fresh_path.features import ContentStore, compress_bins
from mmdd_stage1.construction import serialize_table_parts

from .config import Paths
from .io import iter_jsonl, sha256_file, sha256_json, write_json

TARGET_MAX_ROWS = 20
MAX_CELL_CHARS = 1024
CONTENT_BINS = 64
EMBED_DIM = 4096


@dataclass(frozen=True)
class TableInput:
    object_id: str
    role: str
    parts: tuple[str, ...]
    original_rows: int
    encoded_rows: int
    source_table_id: str | None


def _source_id(reference: object) -> str | None:
    if reference is None:
        return None
    if isinstance(reference, dict):
        return str(reference["source_table_id"])
    return str(reference)


def _load_sources(paths: Paths, needed: set[str]) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    for part in sorted((paths.dataset_root / "source_tables").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            source_id = str(row["source_table_id"])
            if source_id in needed:
                sources[source_id] = row
    missing = needed - sources.keys()
    if missing:
        raise KeyError(f"unresolved source_table_ref values: {sorted(missing)[:10]}")
    return sources


def table_inputs(paths: Paths) -> dict[str, TableInput]:
    """Rebuild every table input from the original dataset, including target refs."""

    targets = list(iter_jsonl(paths.dataset_root / "data_lake_tables" / "part-00000.jsonl"))
    needed = {
        source_id
        for row in targets
        if (source_id := _source_id(row.get("source_table_ref"))) is not None
    }
    sources = _load_sources(paths, needed)
    result: dict[str, TableInput] = {}

    for row in iter_jsonl(paths.dataset_root / "query_tables" / "part-00000.jsonl"):
        object_id = str(row["table_id"])
        parts = serialize_table_parts(
            row,
            max_rows=len(row.get("rows") or []),
            max_cell_chars=MAX_CELL_CHARS,
            row_format="values",
        )
        result[object_id] = TableInput(
            object_id=object_id,
            role="query",
            parts=tuple(parts),
            original_rows=len(row.get("rows") or []),
            encoded_rows=len(parts) - 1,
            source_table_id=str(row.get("source_table_id")) if row.get("source_table_id") else None,
        )

    for row in targets:
        object_id = str(row["table_id"])
        source_id = _source_id(row.get("source_table_ref"))
        visible = sources[source_id] if source_id is not None else row
        table = {"columns": visible["columns"], "rows": visible["rows"]}
        parts = serialize_table_parts(
            table,
            max_rows=TARGET_MAX_ROWS,
            max_cell_chars=MAX_CELL_CHARS,
            row_format="values",
        )
        if object_id in result:
            raise ValueError(f"duplicate table object ID: {object_id}")
        result[object_id] = TableInput(
            object_id=object_id,
            role="target",
            parts=tuple(parts),
            original_rows=len(visible["rows"]),
            encoded_rows=len(parts) - 1,
            source_table_id=source_id,
        )
    return result


def expected_object_types(paths: Paths, tables: dict[str, TableInput]) -> dict[str, str]:
    expected = {object_id: "table" for object_id in tables}
    for part in sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            object_id = str(row["asset_id"])
            object_type = str(row["asset_type"])
            if object_type not in {"text", "image"}:
                raise ValueError(f"{object_id}: unsupported asset type {object_type!r}")
            if object_id in expected:
                raise ValueError(f"duplicate dataset object ID: {object_id}")
            expected[object_id] = object_type
    return expected


def _index(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing index fields {missing}")
    if len(payload["ids"]) != len(set(payload["ids"])):
        raise ValueError(f"{path}: duplicate object IDs")
    return payload


def _content_lengths(content_root: Path, index: dict[str, Any]) -> np.ndarray:
    result = np.empty(len(index["ids"]), dtype=np.int64)
    chunk_cache: dict[int, np.ndarray] = {}
    for position, (chunk, row) in enumerate(zip(index["chunks"], index["rows"])):
        chunk = int(chunk)
        if chunk not in chunk_cache:
            path = content_root / "chunks" / f"chunk_{chunk:06d}.lens.npy"
            chunk_cache[chunk] = np.load(path, allow_pickle=False)
        lens = chunk_cache[chunk]
        row = int(row)
        if row < 0 or row >= len(lens):
            raise ValueError(f"content index row {row} is outside chunk {chunk}")
        result[position] = int(lens[row])
    return result


def _z_statistics(z_path: Path, *, chunk_rows: int = 4096) -> dict[str, float | int]:
    z = np.load(z_path, mmap_mode="r", allow_pickle=False)
    if z.ndim != 2 or z.shape[1] != EMBED_DIM or z.dtype != np.float32:
        raise ValueError(f"invalid z array: shape={z.shape}, dtype={z.dtype}")
    nonfinite = 0
    min_norm = float("inf")
    max_norm = 0.0
    max_norm_error = 0.0
    for start in range(0, len(z), chunk_rows):
        block = np.asarray(z[start : start + chunk_rows])
        nonfinite += int((~np.isfinite(block)).sum())
        norms = np.linalg.norm(block, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        max_norm = max(max_norm, float(norms.max()))
        max_norm_error = max(max_norm_error, float(np.abs(norms - 1.0).max()))
    return {
        "objects": int(len(z)),
        "nonfinite_values": nonfinite,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "max_abs_norm_error": max_norm_error,
    }


def audit_pure_cache(paths: Paths, pure_cache_dir: Path) -> dict[str, Any]:
    """Perform the full CPU-side population audit before any cache is reused."""

    pure_cache_dir = Path(pure_cache_dir).resolve()
    z_root = pure_cache_dir / "z"
    content_root = pure_cache_dir / "content"
    z_index_path = z_root / "z_index.json"
    content_index_path = content_root / "index.json"
    z_index = _index(z_index_path, ("ids", "types", "shape", "dtype"))
    content_index = _index(content_index_path, ("ids", "types", "chunks", "rows"))

    tables = table_inputs(paths)
    expected = expected_object_types(paths, tables)
    z_types = dict(zip(map(str, z_index["ids"]), map(str, z_index["types"])))
    content_types = dict(zip(map(str, content_index["ids"]), map(str, content_index["types"])))
    expected_ids = set(expected)
    z_ids = set(z_types)
    content_ids = set(content_types)
    type_mismatches = sorted(
        object_id
        for object_id, object_type in expected.items()
        if z_types.get(object_id) != object_type or content_types.get(object_id) != object_type
    )

    lengths = _content_lengths(content_root, content_index)
    length_by_id = dict(zip(map(str, content_index["ids"]), map(int, lengths)))
    table_length_mismatches = []
    query_rows_preserved = 0
    targets_over_12 = 0
    targets_over_12_with_more_than_13_groups = 0
    for object_id, table in tables.items():
        actual = length_by_id.get(object_id)
        expected_length = 1 + table.encoded_rows
        if actual != expected_length:
            table_length_mismatches.append(
                {"object_id": object_id, "role": table.role, "expected": expected_length, "actual": actual}
            )
        if table.role == "query" and table.original_rows == table.encoded_rows == (actual or -1) - 1:
            query_rows_preserved += 1
        if table.role == "target" and table.original_rows > 12:
            targets_over_12 += 1
            if actual is not None and actual > 13:
                targets_over_12_with_more_than_13_groups += 1

    evidence_length_errors = sum(
        1
        for object_id, object_type in expected.items()
        if object_type != "table" and not (1 <= length_by_id.get(object_id, 0) <= CONTENT_BINS)
    )
    z_stats = _z_statistics(z_root / "z.f32.npy")
    errors = {
        "missing_z": len(expected_ids - z_ids),
        "extra_z": len(z_ids - expected_ids),
        "missing_content": len(expected_ids - content_ids),
        "extra_content": len(content_ids - expected_ids),
        "type_mismatches": len(type_mismatches),
        "table_length_mismatches": len(table_length_mismatches),
        "evidence_length_errors": evidence_length_errors,
        "z_nonfinite_values": z_stats["nonfinite_values"],
        "z_shape_mismatch": int(list(z_index["shape"]) != [len(expected), EMBED_DIM]),
        "z_dtype_mismatch": int(z_index["dtype"] != "float32"),
        "z_norm_error": int(float(z_stats["max_abs_norm_error"]) > 5e-5),
    }
    status = "PASS" if not any(errors.values()) else "FAIL"
    report = {
        "status": status,
        "scope": "pure_frozen_backbone_cache_only",
        "cache_dir": str(pure_cache_dir),
        "read_only_reuse": True,
        "expected_objects": len(expected),
        "expected_types": {
            object_type: sum(value == object_type for value in expected.values())
            for object_type in ("table", "text", "image")
        },
        "z": {
            **z_stats,
            "index_sha256": sha256_file(z_index_path),
            "array_bytes": (z_root / "z.f32.npy").stat().st_size,
        },
        "content": {
            "objects": len(content_index["ids"]),
            "index_sha256": sha256_file(content_index_path),
            "chunks": len(set(map(int, content_index["chunks"]))),
            "evidence_bins_max": CONTENT_BINS,
        },
        "tables": {
            "objects": len(tables),
            "query_objects": sum(table.role == "query" for table in tables.values()),
            "target_objects": sum(table.role == "target" for table in tables.values()),
            "query_rows_preserved": query_rows_preserved,
            "targets_with_source_table_ref": sum(
                table.role == "target" and table.source_table_id is not None
                for table in tables.values()
            ),
            "targets_over_12_rows": targets_over_12,
            "targets_over_12_with_cached_groups_proving_not_12_cap": targets_over_12_with_more_than_13_groups,
            "serialization_fingerprint": sha256_json(
                {
                    object_id: {"role": table.role, "parts": table.parts}
                    for object_id, table in sorted(tables.items())
                }
            ),
        },
        "legacy_metadata_disposition": {
            "reported_target_max_rows_12": "STALE_METADATA_NOT_USED_AS_CONTRACT",
            "actual_decision_basis": "all table group counts plus current-backbone sample recomputation",
        },
        "errors": errors,
        "examples": {
            "type_mismatches": type_mismatches[:10],
            "table_length_mismatches": table_length_mismatches[:10],
        },
        "recompute_required_before_reuse": True,
    }
    write_json(paths.work_dir / "PURE_CACHE_STRUCTURAL_AUDIT.json", report)
    if status != "PASS":
        raise RuntimeError(f"pure cache structural audit failed: {errors}")
    return report


def _priority(kind: str, object_id: str) -> bytes:
    return hashlib.sha256(f"MMDD-FRESH-RECOVERY-v3.1|cache-probe|{kind}|{object_id}".encode()).digest()


def _asset_samples(paths: Paths, per_kind: int) -> dict[str, list[dict]]:
    samples: dict[str, list[tuple[bytes, dict]]] = {"text": [], "image": []}
    for part in sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            kind = str(row["asset_type"])
            if kind not in samples:
                continue
            object_id = str(row["asset_id"])
            samples[kind].append((_priority(kind, object_id), row))
            samples[kind].sort(key=lambda pair: pair[0])
            del samples[kind][per_kind:]
    return {kind: [row for _, row in rows] for kind, rows in samples.items()}


def _table_samples(tables: dict[str, TableInput], per_kind: int) -> list[TableInput]:
    queries = sorted(
        (table for table in tables.values() if table.role == "query"),
        key=lambda table: _priority("query", table.object_id),
    )[:per_kind]
    long_targets = sorted(
        (
            table
            for table in tables.values()
            if table.role == "target" and table.original_rows > 12
        ),
        key=lambda table: _priority("target_gt12", table.object_id),
    )[:per_kind]
    referenced = sorted(
        (
            table
            for table in tables.values()
            if table.role == "target" and table.source_table_id is not None
        ),
        key=lambda table: _priority("target_ref", table.object_id),
    )[:per_kind]
    unique = {table.object_id: table for table in [*queries, *long_targets, *referenced]}
    return list(unique.values())


def _image_path(paths: Paths, row: dict) -> Path:
    local = Path(str(row.get("local_path") or ""))
    if local.is_file():
        return local.resolve()
    if not local.is_absolute() and (paths.dataset_root / local).is_file():
        return (paths.dataset_root / local).resolve()
    relative = paths.dataset_root / str(row.get("relative_path") or "")
    if relative.is_file():
        return relative.resolve()
    raise FileNotFoundError(f"{row['asset_id']}: image file is unavailable")


def _asset_content_batches(
    paths: Paths, selected_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Rebuild the historical pure-content batch containing each selected asset.

    The cache being audited encoded the complete asset population in original
    file order, split by an object-ID hash over two shards.  Text batches held
    eight objects and image batches held four.  Rebuilding those groups from
    the dataset root avoids treating padding-dependent bfloat16 drift as an
    input-contract mismatch.
    """

    batch_sizes = {"text": 8, "image": 4}
    pending: dict[tuple[str, int], list[dict[str, Any]]] = {
        (kind, shard): [] for kind in batch_sizes for shard in range(2)
    }
    result: dict[str, list[dict[str, Any]]] = {}

    def finish(batch: list[dict[str, Any]]) -> None:
        if any(str(row["asset_id"]) in selected_ids for row in batch):
            for row in batch:
                object_id = str(row["asset_id"])
                if object_id in selected_ids:
                    result[object_id] = list(batch)

    for part in sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            kind = str(row["asset_type"])
            if kind not in batch_sizes:
                continue
            object_id = str(row["asset_id"])
            shard = int.from_bytes(
                hashlib.sha256(object_id.encode("utf-8")).digest()[:8], "big"
            ) % 2
            key = (kind, shard)
            pending[key].append(row)
            if len(pending[key]) == batch_sizes[kind]:
                finish(pending[key])
                pending[key] = []
    for batch in pending.values():
        if batch:
            finish(batch)
    missing = selected_ids - result.keys()
    if missing:
        raise KeyError(f"could not rebuild content batches for: {sorted(missing)}")
    return result


def _cosine_summary(fresh: torch.Tensor, cached: torch.Tensor) -> dict[str, float] | None:
    if fresh.shape != cached.shape:
        return None
    values = F.cosine_similarity(fresh.float(), cached.float(), dim=-1)
    return {"min": float(values.min()), "mean": float(values.mean())}


def recompute_cache_samples(
    paths: Paths,
    pure_cache_dir: Path,
    *,
    device: str = "cuda:0",
    samples_per_kind: int = 1,
) -> dict[str, Any]:
    """Re-encode deterministic objects with the public backbone and compare values."""

    if samples_per_kind <= 0:
        raise ValueError("samples_per_kind must be positive")
    structural_path = paths.work_dir / "PURE_CACHE_STRUCTURAL_AUDIT.json"
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    if structural.get("status") != "PASS":
        raise RuntimeError("structural pure-cache audit must pass before GPU recomputation")

    pure_cache_dir = Path(pure_cache_dir).resolve()
    tables = table_inputs(paths)
    prompts = json.loads((paths.package_dir / "ENCODER_PROMPTS.json").read_text(encoding="utf-8"))[
        "instructions"
    ]
    z_index = _index(pure_cache_dir / "z" / "z_index.json", ("ids", "types", "shape", "dtype"))
    z_rows = {str(object_id): row for row, object_id in enumerate(z_index["ids"])}
    z = np.load(pure_cache_dir / "z" / "z.f32.npy", mmap_mode="r", allow_pickle=False)
    content = ContentStore(pure_cache_dir / "content")

    from cache_stage1_features import (
        _load_embedder_class,
        build_object_features,
        encode_inputs,
    )

    torch.cuda.set_device(torch.device(device))
    embedder = _load_embedder_class(paths.backbone_dir)(
        str(paths.backbone_dir), max_length=8192, min_pixels=4096, max_pixels=1843200
    )
    records: list[tuple[dict[str, Any], str, str]] = []
    for table in _table_samples(tables, samples_per_kind):
        role_key = f"{table.role}/table"
        record = {
            "object_id": table.object_id,
            "object_type": "table",
            "embedding_role": table.role,
            "table_parts": list(table.parts),
            "instruction": prompts[role_key],
        }
        records.append((record, role_key, sha256_json(list(table.parts))))
    for kind, rows in _asset_samples(paths, samples_per_kind).items():
        for row in rows:
            object_id = str(row["asset_id"])
            record: dict[str, Any] = {
                "object_id": object_id,
                "object_type": kind,
                "embedding_role": "evidence",
                "instruction": prompts[f"evidence/{kind}"],
            }
            if kind == "text":
                record["text"] = str(row["content"])
                input_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            else:
                image_path = _image_path(paths, row)
                record["image"] = str(image_path)
                input_hash = sha256_file(image_path)
            records.append((record, f"evidence/{kind}", input_hash))

    asset_records = {
        str(record["object_id"]): record
        for record, _role_key, _input_hash in records
        if record["object_type"] in {"text", "image"}
    }
    content_batches = _asset_content_batches(paths, set(asset_records))
    batch_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    prompts_by_kind = {
        kind: prompts[f"evidence/{kind}"] for kind in ("text", "image")
    }
    for object_id, batch in content_batches.items():
        if object_id in batch_outputs:
            continue
        kind = str(asset_records[object_id]["object_type"])
        items = []
        for row in batch:
            row_kind = str(row["asset_type"])
            if row_kind != kind:
                raise ValueError("content batch mixed evidence modalities")
            items.append(
                {
                    "text": str(row["content"]) if kind == "text" else None,
                    "image": str(_image_path(paths, row)) if kind == "image" else None,
                    "instruction": prompts_by_kind[kind],
                }
            )
        outputs = encode_inputs(embedder, items, include_hidden=True)
        for row, (embedding, hidden, _input_ids) in zip(batch, outputs, strict=True):
            row_id = str(row["asset_id"])
            if row_id in asset_records:
                assert hidden is not None
                batch_outputs[row_id] = (embedding.float(), compress_bins(hidden.float(), CONTENT_BINS))

    comparisons = []
    for record, role_key, input_hash in records:
        object_id = str(record["object_id"])
        payload = build_object_features(
            embedder,
            record,
            input_dir=paths.dataset_root,
            instruction=None,
            storage_dtype=torch.bfloat16,
            include_hidden=True,
            include_row_embeddings=False,
            table_row_batch_size=8,
            table_tokens_per_group=1,
        )
        fresh_z = payload["embedding"].float()
        cached_z = torch.from_numpy(np.array(z[z_rows[object_id]], copy=True)).float()
        if record["object_type"] == "table":
            fresh_content = payload["hidden_states"].float()
        else:
            fresh_content = compress_bins(payload["hidden_states"].float(), CONTENT_BINS)
        fresh_disk = fresh_content.to(torch.float16)
        cached_content = content.get(object_id)
        z_max_abs = float((fresh_z - cached_z).abs().max())
        z_cosine = float(F.cosine_similarity(fresh_z, cached_z, dim=0))
        content_shape_equal = list(fresh_disk.shape) == list(cached_content.shape)
        content_max_abs = (
            float((fresh_disk - cached_content).abs().max()) if content_shape_equal else None
        )
        content_cosine = _cosine_summary(fresh_disk, cached_content)
        batch_z_max_abs = None
        batch_z_cosine = None
        batch_content_max_abs = None
        batch_content_cosine = None
        if object_id in batch_outputs:
            batch_z, batch_content = batch_outputs[object_id]
            batch_disk = batch_content.to(torch.float16)
            batch_z_max_abs = float((batch_z - cached_z).abs().max())
            batch_z_cosine = float(F.cosine_similarity(batch_z, cached_z, dim=0))
            if list(batch_disk.shape) == list(cached_content.shape):
                batch_content_max_abs = float((batch_disk - cached_content).abs().max())
                batch_content_cosine = _cosine_summary(batch_disk, cached_content)
        z_exact = z_max_abs <= 5e-5 or (
            batch_z_max_abs is not None and batch_z_max_abs <= 5e-5
        )
        z_semantic = max(
            z_cosine,
            batch_z_cosine if batch_z_cosine is not None else -1.0,
        ) >= 0.999
        content_exact = content_shape_equal and (
            (content_max_abs is not None and content_max_abs <= 1e-3)
            or (batch_content_max_abs is not None and batch_content_max_abs <= 1e-3)
        )
        passed = z_semantic and content_exact
        comparisons.append(
            {
                "object_id": object_id,
                "object_type": record["object_type"],
                "role": role_key,
                "input_sha256": input_hash,
                "prompt_sha256": hashlib.sha256(prompts[role_key].encode("utf-8")).hexdigest(),
                "fresh_z_norm": float(torch.linalg.vector_norm(fresh_z)),
                "single_z_max_abs": z_max_abs,
                "single_z_cosine": z_cosine,
                "content_batch_z_max_abs": batch_z_max_abs,
                "content_batch_z_cosine": batch_z_cosine,
                "z_exact": z_exact,
                "z_acceptance": "exact" if z_exact else "cosine>=0.999_batch_drift",
                "fresh_content_shape": list(fresh_disk.shape),
                "cached_content_shape": list(cached_content.shape),
                "single_content_max_abs_after_float16": content_max_abs,
                "single_content_cosine": content_cosine,
                "batch_content_max_abs_after_float16": batch_content_max_abs,
                "batch_content_cosine": batch_content_cosine,
                "content_match": (
                    "single_exact"
                    if content_max_abs is not None and content_max_abs <= 1e-3
                    else "generation_batch_exact"
                    if batch_content_max_abs is not None and batch_content_max_abs <= 1e-3
                    else "none"
                ),
                "status": "PASS" if passed else "FAIL",
            }
        )

    status = "PASS" if comparisons and all(row["status"] == "PASS" for row in comparisons) else "FAIL"
    report = {
        "status": status,
        "backbone": str(paths.backbone_dir),
        "device": device,
        "compute_dtype": "bfloat16",
        "sample_policy": "smallest SHA256(protocol|kind|object_id), including target>12 and source_table_ref",
        "numeric_contract": {
            "content": "float16 exact (max_abs<=1e-3) under single or reconstructed generation batch",
            "z": "exact max_abs<=5e-5 when available; otherwise cosine>=0.999 for padding-dependent batch drift",
            "evidence_content_batches": {"shards": 2, "text": 8, "image": 4},
        },
        "samples_per_kind": samples_per_kind,
        "comparisons": comparisons,
    }
    write_json(paths.work_dir / "PURE_CACHE_RECOMPUTE_AUDIT.json", report)
    if status != "PASS":
        raise RuntimeError("frozen-backbone cache sample recomputation failed")

    structural["recompute_required_before_reuse"] = False
    structural["recompute_audit"] = "PURE_CACHE_RECOMPUTE_AUDIT.json"
    structural["reuse_decision"] = "PASS_READ_ONLY_REUSE"
    write_json(structural_path, structural)
    write_json(paths.work_dir / "PURE_FEATURE_SOURCE.json", {
        "status": "PASS",
        "mode": "read_only_compatible_pure_backbone_cache",
        "cache_dir": str(pure_cache_dir),
        "z": str((pure_cache_dir / "z" / "z.f32.npy").resolve()),
        "z_index": str((pure_cache_dir / "z" / "z_index.json").resolve()),
        "content": str((pure_cache_dir / "content").resolve()),
        "structural_audit": "PURE_CACHE_STRUCTURAL_AUDIT.json",
        "recompute_audit": "PURE_CACHE_RECOMPUTE_AUDIT.json",
        "forbidden_descendants": ["historical_PCA", "historical_lists", "historical_logits", "historical_task_weights"],
    })
    return report


def _query_feature_record(table: TableInput) -> dict[str, Any]:
    if table.role != "query":
        raise ValueError(f"{table.object_id}: expected a query table")
    return {
        "object_id": table.object_id,
        "object_type": "table",
        "embedding_role": "query",
        "table_parts": list(table.parts),
    }


def _approved_pure_z(paths: Paths) -> tuple[Path, dict[str, Any], np.ndarray]:
    source_path = paths.work_dir / "PURE_FEATURE_SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise RuntimeError("pure frozen feature source is not approved")
    z_path = Path(str(source["z"])).resolve()
    index_path = Path(str(source["z_index"])).resolve()
    index = _index(index_path, ("ids", "types", "shape", "dtype"))
    z = np.load(z_path, mmap_mode="r", allow_pickle=False)
    return source_path, index, z


def audit_row_cache(paths: Paths, row_cache_dir: Path) -> dict[str, Any]:
    """Verify that a historical cache is a compatible pure query-row cache."""

    row_cache_dir = Path(row_cache_dir).resolve()
    metadata_path = row_cache_dir / "metadata.json"
    manifest_path = row_cache_dir / "manifest.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompts = json.loads(
        (paths.package_dir / "ENCODER_PROMPTS.json").read_text(encoding="utf-8")
    )["instructions"]
    expected_metadata = {
        "model": Path(str(metadata.get("model_dir", ""))).resolve() == paths.backbone_dir,
        "dtype": metadata.get("dtype") == "bf16",
        "prompt_version": metadata.get("prompt_version") == "role_modality_v2_object_only",
        "query_prompt": metadata.get("embedding_instructions", {}).get("query_table")
        == prompts["query/table"],
        "row_prompt": metadata.get("embedding_instructions", {}).get("query_row_table")
        == prompts["query_row/table"],
        "instruction_override": metadata.get("instruction_override") is None,
    }

    tables = {
        object_id: table
        for object_id, table in table_inputs(paths).items()
        if table.role == "query"
    }
    manifest: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in iter_jsonl(manifest_path):
        object_id = str(row["object_id"])
        if object_id in manifest:
            duplicate_ids.append(object_id)
        manifest[object_id] = row

    pure_source_path, z_index, z = _approved_pure_z(paths)
    z_rows = {str(object_id): index for index, object_id in enumerate(z_index["ids"])}
    cache_root = row_cache_dir.resolve()
    errors = {
        "metadata_mismatches": sum(not value for value in expected_metadata.values()),
        "duplicate_manifest_ids": len(duplicate_ids),
        "missing_queries": 0,
        "source_fingerprint_mismatches": 0,
        "object_type_mismatches": 0,
        "missing_feature_files": 0,
        "invalid_feature_paths": 0,
        "missing_row_embeddings": 0,
        "row_shape_mismatches": 0,
        "row_dtype_mismatches": 0,
        "row_nonfinite_values": 0,
        "row_norm_errors": 0,
        "missing_same_object_z": 0,
        "same_object_z_mismatches": 0,
    }
    examples: dict[str, list[Any]] = {key: [] for key in errors}
    row_index: list[dict[str, Any]] = []
    min_row_norm = float("inf")
    max_row_norm = 0.0
    max_row_norm_error = 0.0
    max_same_object_z_abs = 0.0

    def note(key: str, value: Any) -> None:
        errors[key] += 1
        if len(examples[key]) < 10:
            examples[key].append(value)

    for object_id, table in sorted(tables.items()):
        entry = manifest.get(object_id)
        if entry is None:
            note("missing_queries", object_id)
            continue
        record = _query_feature_record(table)
        if entry.get("source_fingerprint") != sha256_json(record):
            note("source_fingerprint_mismatches", object_id)
        if entry.get("object_type") != "table":
            note("object_type_mismatches", object_id)
        feature_path = (row_cache_dir / str(entry.get("feature_path", ""))).resolve()
        if cache_root not in feature_path.parents:
            note("invalid_feature_paths", {"object_id": object_id, "path": str(feature_path)})
            continue
        if not feature_path.is_file():
            note("missing_feature_files", {"object_id": object_id, "path": str(feature_path)})
            continue
        payload = torch.load(feature_path, map_location="cpu", weights_only=True)
        rows = payload.get("row_embeddings")
        if not isinstance(rows, torch.Tensor):
            note("missing_row_embeddings", object_id)
            continue
        if list(rows.shape) != [table.encoded_rows, EMBED_DIM]:
            note(
                "row_shape_mismatches",
                {"object_id": object_id, "expected": [table.encoded_rows, EMBED_DIM], "actual": list(rows.shape)},
            )
        if rows.dtype != torch.float32:
            note("row_dtype_mismatches", {"object_id": object_id, "dtype": str(rows.dtype)})
        nonfinite = int((~torch.isfinite(rows)).sum())
        if nonfinite:
            errors["row_nonfinite_values"] += nonfinite
            if len(examples["row_nonfinite_values"]) < 10:
                examples["row_nonfinite_values"].append({"object_id": object_id, "values": nonfinite})
        norms = torch.linalg.vector_norm(rows.float(), dim=1)
        min_row_norm = min(min_row_norm, float(norms.min()))
        max_row_norm = max(max_row_norm, float(norms.max()))
        norm_error = float((norms - 1.0).abs().max())
        max_row_norm_error = max(max_row_norm_error, norm_error)
        if norm_error > 5e-5:
            note("row_norm_errors", {"object_id": object_id, "max_abs": norm_error})

        cached_object_z = payload.get("embedding")
        pure_position = z_rows.get(object_id)
        if pure_position is None or not isinstance(cached_object_z, torch.Tensor):
            note("missing_same_object_z", object_id)
        elif list(cached_object_z.shape) != [EMBED_DIM]:
            note("same_object_z_mismatches", {"object_id": object_id, "shape": list(cached_object_z.shape)})
        else:
            pure_z = torch.from_numpy(np.array(z[pure_position], copy=True))
            z_abs = float((cached_object_z.float() - pure_z.float()).abs().max())
            max_same_object_z_abs = max(max_same_object_z_abs, z_abs)
            if z_abs > 5e-5:
                note("same_object_z_mismatches", {"object_id": object_id, "max_abs": z_abs})

        row_index.append(
            {
                "query_id": object_id,
                "feature_path": str(feature_path),
                "original_row_ids": list(range(table.original_rows)),
                "encoded_row_ids": list(range(table.encoded_rows)),
            }
        )

    extra_query_ids = sorted(
        object_id
        for object_id, entry in manifest.items()
        if object_id.startswith("query_") and object_id not in tables
    )
    errors["extra_queries"] = len(extra_query_ids)
    examples["extra_queries"] = extra_query_ids[:10]
    status = "PASS" if all(expected_metadata.values()) and not any(errors.values()) else "FAIL"
    contract = {
        "status": "STRUCTURAL_PASS_RECOMPUTE_REQUIRED" if status == "PASS" else "FAIL",
        "backbone": str(paths.backbone_dir),
        "compute_dtype": "bfloat16",
        "embedding_dim": EMBED_DIM,
        "serialization": "schema_part + '\\n' + single_row_part",
        "schema_part": "Columns: original column names joined by ' | '",
        "row_part": "Row: original-order cell values joined by ' | '",
        "query_rows": "all original rows in original order",
        "original_row_mapping": "zero-based row position; one output for each original row",
        "instruction": prompts["query_row/table"],
        "pooling": "last valid token followed by float32 L2 normalization",
        "row_batch_size_max": 8,
        "storage_dtype": "float32",
        "numeric_tolerance": {"max_abs": 5e-5, "norm_max_abs_error": 5e-5},
    }
    report = {
        "status": status,
        "scope": "query_row_embeddings_only",
        "cache_dir": str(row_cache_dir),
        "read_only_reuse": True,
        "queries": len(tables),
        "rows": sum(table.encoded_rows for table in tables.values()),
        "metadata_checks": expected_metadata,
        "row_statistics": {
            "min_norm": min_row_norm,
            "max_norm": max_row_norm,
            "max_abs_norm_error": max_row_norm_error,
        },
        "same_object_pure_z_check": {
            "source": str(pure_source_path),
            "queries_compared": len(row_index) - errors["missing_same_object_z"],
            "max_abs": max_same_object_z_abs,
            "tolerance": 5e-5,
        },
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "errors": errors,
        "examples": examples,
        "recompute_required_before_reuse": True,
    }
    write_json(paths.work_dir / "ROW_ENCODER_CONTRACT.json", contract)
    write_json(paths.work_dir / "ROW_CACHE_STRUCTURAL_AUDIT.json", report)
    if status != "PASS":
        raise RuntimeError(f"row cache structural audit failed: {errors}")
    write_json(paths.work_dir / "ROW_INDEX.json", {"queries": row_index})
    return report


def recompute_row_samples(
    paths: Paths,
    row_cache_dir: Path,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Recompute one deterministic query per split using the original row batch."""

    structural_path = paths.work_dir / "ROW_CACHE_STRUCTURAL_AUDIT.json"
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    row_cache_dir = Path(row_cache_dir).resolve()
    if structural.get("status") != "PASS" or Path(structural["cache_dir"]).resolve() != row_cache_dir:
        raise RuntimeError("matching row-cache structural audit must pass first")

    tables = table_inputs(paths)
    selected: dict[str, tuple[bytes, TableInput]] = {}
    query_path = paths.dataset_root / "query_tables" / "part-00000.jsonl"
    for row in iter_jsonl(query_path):
        split = str(row["split"])
        object_id = str(row["table_id"])
        priority = _priority(f"row_{split}", object_id)
        if split not in selected or priority < selected[split][0]:
            selected[split] = (priority, tables[object_id])
    if set(selected) != {"train", "dev", "test"}:
        raise ValueError(f"unexpected query splits: {sorted(selected)}")

    row_index = json.loads((paths.work_dir / "ROW_INDEX.json").read_text(encoding="utf-8"))
    feature_paths = {
        str(row["query_id"]): Path(str(row["feature_path"]))
        for row in row_index["queries"]
    }
    prompts = json.loads(
        (paths.package_dir / "ENCODER_PROMPTS.json").read_text(encoding="utf-8")
    )["instructions"]
    from cache_stage1_features import _load_embedder_class, encode_inputs

    torch.cuda.set_device(torch.device(device))
    embedder = _load_embedder_class(paths.backbone_dir)(
        str(paths.backbone_dir), max_length=8192, min_pixels=4096, max_pixels=1843200
    )
    comparisons = []
    for split, (_priority_value, table) in sorted(selected.items()):
        routing_items = [
            {
                "text": f"{table.parts[0]}\n{row_part}",
                "instruction": prompts["query_row/table"],
            }
            for row_part in table.parts[1:]
        ]
        outputs = encode_inputs(embedder, routing_items, include_hidden=False)
        fresh = torch.stack([embedding.float() for embedding, _, _ in outputs])
        payload = torch.load(feature_paths[table.object_id], map_location="cpu", weights_only=True)
        cached = payload["row_embeddings"].float()
        shape_equal = list(fresh.shape) == list(cached.shape)
        max_abs = float((fresh - cached).abs().max()) if shape_equal else None
        cosine = _cosine_summary(fresh, cached) if shape_equal else None
        passed = shape_equal and max_abs is not None and max_abs <= 5e-5
        comparisons.append(
            {
                "split": split,
                "query_id": table.object_id,
                "row_ids": list(range(table.encoded_rows)),
                "batch_rows": len(routing_items),
                "input_sha256": sha256_json([item["text"] for item in routing_items]),
                "prompt_sha256": hashlib.sha256(
                    prompts["query_row/table"].encode("utf-8")
                ).hexdigest(),
                "fresh_shape": list(fresh.shape),
                "cached_shape": list(cached.shape),
                "max_abs": max_abs,
                "cosine": cosine,
                "status": "PASS" if passed else "FAIL",
            }
        )

    status = "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL"
    report = {
        "status": status,
        "scope": "one deterministic query per original split",
        "cache_dir": str(row_cache_dir),
        "device": device,
        "compute_dtype": "bfloat16",
        "row_batch_size_max": 8,
        "numeric_contract": "float32 exact within max_abs<=5e-5 under the original per-query batch",
        "comparisons": comparisons,
    }
    write_json(paths.work_dir / "ROW_CACHE_RECOMPUTE_AUDIT.json", report)
    if status != "PASS":
        raise RuntimeError("query-row cache sample recomputation failed")

    structural["recompute_required_before_reuse"] = False
    structural["recompute_audit"] = "ROW_CACHE_RECOMPUTE_AUDIT.json"
    structural["reuse_decision"] = "PASS_READ_ONLY_REUSE"
    write_json(structural_path, structural)
    contract_path = paths.work_dir / "ROW_ENCODER_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "PASS"
    contract["recompute_audit"] = "ROW_CACHE_RECOMPUTE_AUDIT.json"
    write_json(contract_path, contract)
    write_json(
        paths.work_dir / "ROW_FEATURE_SOURCE.json",
        {
            "status": "PASS",
            "mode": "read_only_compatible_pure_query_row_cache",
            "cache_dir": str(row_cache_dir),
            "manifest": str((row_cache_dir / "manifest.jsonl").resolve()),
            "index": "ROW_INDEX.json",
            "permitted_payload": "row_embeddings_only",
            "ignored_payloads": ["embedding", "teacher_hidden_states"],
            "structural_audit": "ROW_CACHE_STRUCTURAL_AUDIT.json",
            "recompute_audit": "ROW_CACHE_RECOMPUTE_AUDIT.json",
            "forbidden_descendants": [
                "historical_PCA",
                "historical_lists",
                "historical_logits",
                "historical_task_weights",
                "historical_row_support_calibration",
            ],
        },
    )
    return report

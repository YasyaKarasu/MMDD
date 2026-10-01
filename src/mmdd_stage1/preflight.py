"""Fail-closed input, feature-recipe, cache, and alias preflight for Stage-1 CQET."""
from __future__ import annotations

import gzip
import hashlib
import json
import platform
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from torch.nn import functional as F

from . import EXPERIMENT_ID, SCHEMA_VERSION
from .config import Paths


REPO_ROOT = Path(__file__).resolve().parents[2]
# Bound by configure() from the protocol's ``paths`` / ``hardware`` blocks; never hardcoded.
DATASET_ROOT: Path = None
BACKBONE_DIR: Path = None
PURE_CACHE_DIR: Path = None
UPSTREAM_CACHE_DIR: Path = None
UPSTREAM_DATA_DIR: Path = None
PACKAGE_DIR: Path = None
RUN_ROOT: Path = None
GPU_UUID: str = None


def configure(paths: Paths) -> None:
    """Bind the module to one run. ``package_dir`` holds ``next_round/protocol.json``."""
    global DATASET_ROOT, BACKBONE_DIR, PURE_CACHE_DIR, UPSTREAM_CACHE_DIR, UPSTREAM_DATA_DIR
    global PACKAGE_DIR, RUN_ROOT, GPU_UUID
    DATASET_ROOT = paths.dataset_root
    BACKBONE_DIR = paths.backbone_dir
    PURE_CACHE_DIR = paths.pure_cache_dir
    UPSTREAM_CACHE_DIR = paths.upstream_cache_dir or paths.row_cache_manifest.parent
    UPSTREAM_DATA_DIR = paths.upstream_data_dir
    PACKAGE_DIR = paths.package_dir or paths.run_root / "protocol_package"
    RUN_ROOT = paths.run_root
    GPU_UUID = paths.gpu_uuid


def _require_configured() -> None:
    if RUN_ROOT is None or DATASET_ROOT is None:
        raise RuntimeError("preflight is not bound to a run; call preflight.configure(paths) first")

# Match the real frozen extractor's image boundary. Dataset images are trusted
# local inputs and some legitimately exceed Pillow's heuristic pixel ceiling.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_identity(path: Path, *, root: Path | None = None, role: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    relative = root is not None and path.is_relative_to(root.resolve())
    record: dict[str, Any] = {
        "path": str(path.relative_to(root.resolve())) if relative else str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        record["role"] = role
    return record


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected object")
            yield value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _required_dataset_files() -> list[Path]:
    paths = [
        DATASET_ROOT / "dataset_manifest.json",
        DATASET_ROOT / "splits.json",
        DATASET_ROOT / "qrels.jsonl",
        DATASET_ROOT / "query_tables" / "part-00000.jsonl",
        DATASET_ROOT / "data_lake_tables" / "part-00000.jsonl",
        DATASET_ROOT / "source_tables" / "part-00000.jsonl",
        DATASET_ROOT / "evidence_recoveries" / "part-00000.jsonl",
    ]
    paths.extend(sorted((DATASET_ROOT / "bridge_assets").glob("part-*.jsonl")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing original inputs: {missing}")
    return paths


def _hash_files(paths: list[Path], *, root: Path, role: str) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda path: file_identity(path, root=root, role=role), paths))
    return sorted(records, key=lambda row: str(row["path"]).encode("utf-8"))


def validate_dataset() -> dict[str, Any]:
    splits = json.loads((DATASET_ROOT / "splits.json").read_text(encoding="utf-8"))
    queries: dict[str, dict[str, str]] = {}
    split_counts: Counter[str] = Counter()
    source_groups: dict[str, str] = {}
    for row in iter_jsonl(DATASET_ROOT / "query_tables" / "part-00000.jsonl"):
        query_id = str(row["table_id"])
        if query_id in queries:
            raise ValueError(f"duplicate query: {query_id}")
        split = str(row["split"])
        source_group = str(row.get("source_table_id") or "")
        if not source_group:
            raise ValueError(f"query lacks source_table_id: {query_id}")
        queries[query_id] = {"split": split, "source_group": source_group}
        source_groups[query_id] = source_group
        split_counts[split] += 1

    targets: set[str] = set()
    for row in iter_jsonl(DATASET_ROOT / "data_lake_tables" / "part-00000.jsonl"):
        target_id = str(row["table_id"])
        if target_id in targets:
            raise ValueError(f"duplicate target: {target_id}")
        targets.add(target_id)

    assets: set[str] = set()
    modality_counts: Counter[str] = Counter()
    for part in sorted((DATASET_ROOT / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            asset_id = str(row["asset_id"])
            if asset_id in assets:
                raise ValueError(f"duplicate evidence: {asset_id}")
            assets.add(asset_id)
            modality_counts[str(row["asset_type"])] += 1

    qrel_pairs: set[tuple[str, str]] = set()
    gold_counts: Counter[str] = Counter()
    query_kinds: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(DATASET_ROOT / "qrels.jsonl"):
        query_id = str(row["query_table_id"])
        target_id = str(row["target_table_id"])
        pair = (query_id, target_id)
        if pair in qrel_pairs:
            raise ValueError(f"duplicate qrel: {pair}")
        if query_id not in queries or target_id not in targets:
            raise ValueError(f"qrel foreign-key error: {pair}")
        if str(row["split"]) != queries[query_id]["split"]:
            raise ValueError(f"qrel split mismatch: {pair}")
        qrel_pairs.add(pair)
        if float(row.get("rel", 0)) > 0:
            gold_counts[queries[query_id]["split"]] += 1
            reason = str(row.get("reason") or "unknown")
            query_kinds[query_id].add(reason)

    recovery_count = 0
    for row in iter_jsonl(DATASET_ROOT / "evidence_recoveries" / "part-00000.jsonl"):
        query_id = str(row["query_table_id"])
        target_id = str(row["target_table_id"])
        evidence = row.get("evidence") or {}
        asset_id = str(evidence.get("asset_id"))
        if (query_id, target_id) not in qrel_pairs or asset_id not in assets:
            raise ValueError(f"recovery foreign-key error: {(query_id, target_id, asset_id)}")
        if str(row["split"]) != queries[query_id]["split"]:
            raise ValueError(f"recovery split mismatch: {(query_id, target_id, asset_id)}")
        recovery_count += 1

    declared = splits.get("query_table_counts") or {}
    if declared and any(int(declared.get(key, -1)) != value for key, value in split_counts.items()):
        raise ValueError(f"splits.json count mismatch: declared={declared}, actual={dict(split_counts)}")

    return {
        "queries": len(queries),
        "query_split_counts": dict(split_counts),
        "source_groups": len(set(source_groups.values())),
        "targets": len(targets),
        "evidence": len(assets),
        "evidence_modality_counts": dict(modality_counts),
        "qrel_pairs": len(qrel_pairs),
        "positive_qrel_counts": dict(gold_counts),
        "recoveries": recovery_count,
        "query_kind_reason_sets": dict(Counter("+".join(sorted(v)) for v in query_kinds.values())),
        "status": "PASS",
    }


def _backbone_files() -> list[Path]:
    names = (
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "scripts/qwen3_vl_embedding.py",
    )
    files = [BACKBONE_DIR / name for name in names]
    files.extend(sorted(BACKBONE_DIR.glob("model-*.safetensors")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen-backbone files: {missing}")
    return files


def build_recipe_lock() -> dict[str, Any]:
    source_paths = [
        REPO_ROOT / "src" / "cache_stage1_features.py",
        REPO_ROOT / "src" / "run_stage1.py",
        REPO_ROOT / "src" / "mmdd_stage1" / "content.py",
        REPO_ROOT / "src" / "mmdd_stage1" / "content_encoder.py",
        REPO_ROOT / "src" / "mmdd_stage1" / "construction.py",
        RUN_ROOT / "protocol.json",
    ]
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "status": "SOURCE_RECOVERED_PENDING_NUMERIC_PROBE",
        "backbone": "Qwen3-VL-Embedding-8B",
        "backbone_dir": str(BACKBONE_DIR),
        "frozen": True,
        "compute_dtype": "bfloat16",
        "task_input_dtype": "float32",
        "max_length": 8192,
        "image": {
            "min_pixels": 4096,
            "max_pixels": 1843200,
            "processor": "official Qwen3VLEmbedder -> qwen_vl_utils.process_vision_info",
            "do_resize_in_processor": False,
        },
        "serialization": {
            "prompts": "role_modality_v2_object_only cache_stage1_features.EMBEDDING_INSTRUCTIONS",
            "query_table": "Columns part plus every original Row part",
            "target_table": "Columns part plus first 20 rows; max cell chars 1024",
            "query_row": "schema part + LF + one row part",
            "text": "raw content field",
            "image": "raw local_path decoded by official vision utility",
        },
        "retrieval_z": {
            "pooling": "last attention-mask-valid token",
            "normalization": "torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)",
            "storage_dtype": "float32",
            "embedding_dim": 4096,
            "original_batching": {
                "shards": 2,
                "assignment": "zero-based stage1_objects record position modulo 2",
                "non_table_buffer": 32,
                "sort": "text character count or bounded decoded image pixel count",
                "batch_size": 4,
                "teacher_selected_and_tables": "single object",
            },
        },
        "content_tokens": {
            "source": "last_hidden_state at valid attention-mask positions",
            "table": "one float32 contiguous mean per schema/row group, then float16 content store",
            "text_image": "at most 64 deterministic consecutive-bin means, then float16",
            "generation_batching": {"text": 8, "image": 4, "shards": 2},
        },
        "row_z": {
            "serialization": "schema part + LF + one row part",
            "batch_size": 8,
            "pooling": "same last-valid FP32-L2 retrieval pooling",
            "storage_dtype": "float32",
        },
        "numeric_probe": {"samples_per_modality": 16, "train_queries": 16, "rtol": 1e-4, "atol": 1e-5},
        "source_files": _hash_files(source_paths, root=REPO_ROOT, role="frozen_recipe_source"),
        "backbone_files": _hash_files(_backbone_files(), root=BACKBONE_DIR, role="frozen_backbone"),
    }
    recipe["recipe_sha256"] = sha256_json(recipe)
    write_json(RUN_ROOT / "FROZEN_RECIPE_LOCK.json", recipe)
    return recipe


def build_content_aliases() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    image_file_cache: dict[str, tuple[str, str, int, int]] = {}
    for part in sorted((DATASET_ROOT / "bridge_assets").glob("part-*.jsonl")):
        for raw in iter_jsonl(part):
            asset_id = str(raw["asset_id"])
            modality = str(raw["asset_type"])
            if modality == "text":
                visible = unicodedata.normalize("NFC", str(raw["content"]).replace("\r\n", "\n").replace("\r", "\n"))
                content_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "asset_id": asset_id,
                    "modality": modality,
                    "visible_content_sha256": content_hash,
                    "raw_file_sha256": None,
                    "pixel_sha256": None,
                    "width": None,
                    "height": None,
                }
            elif modality == "image":
                image_path = str(Path(str(raw.get("local_path") or raw.get("relative_path"))).resolve())
                identity = image_file_cache.get(image_path)
                if identity is None:
                    path = Path(image_path)
                    file_hash = sha256_file(path)
                    advertised = raw.get("sha256")
                    if advertised and str(advertised) != file_hash:
                        raise ValueError(f"image hash mismatch for {asset_id}")
                    with Image.open(path) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        width, height = image.size
                        pixel_digest = hashlib.sha256()
                        pixel_digest.update(width.to_bytes(8, "big"))
                        pixel_digest.update(height.to_bytes(8, "big"))
                        pixel_digest.update(image.tobytes())
                        pixel_hash = pixel_digest.hexdigest()
                    identity = (file_hash, pixel_hash, width, height)
                    image_file_cache[image_path] = identity
                file_hash, content_hash, width, height = identity
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "asset_id": asset_id,
                    "modality": modality,
                    "visible_content_sha256": content_hash,
                    "raw_file_sha256": file_hash,
                    "pixel_sha256": content_hash,
                    "width": width,
                    "height": height,
                }
            else:
                raise ValueError(f"unsupported evidence modality: {modality}")
            if content_hash == asset_id:
                raise ValueError(f"fake content hash for {asset_id}")
            groups[(modality, content_hash)].append(asset_id)
            rows.append(record)

    canonical = {
        key: min(ids, key=lambda value: value.encode("utf-8"))
        for key, ids in groups.items()
    }
    for row in rows:
        key = (str(row["modality"]), str(row["visible_content_sha256"]))
        row["canonical_evidence_id"] = canonical[key]
        row["alias_count"] = len(groups[key])
    rows.sort(key=lambda row: str(row["asset_id"]).encode("utf-8"))

    output = RUN_ROOT / "CONTENT_ALIASES.jsonl.gz"
    temporary = output.with_suffix(".jsonl.gz.tmp")
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", fileobj=raw_handle, mode="wb", mtime=0) as gz_handle:
            for row in rows:
                gz_handle.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    temporary.replace(output)
    report = {
        "status": "PASS",
        "objects": len(rows),
        "canonical_objects": len(groups),
        "alias_objects": sum(len(ids) - 1 for ids in groups.values()),
        "alias_groups": sum(len(ids) > 1 for ids in groups.values()),
        "modalities": dict(Counter(str(row["modality"]) for row in rows)),
        "file": file_identity(output, root=RUN_ROOT, role="content_aliases"),
    }
    write_json(RUN_ROOT / "CONTENT_ALIAS_REPORT.json", report)
    return report


def build_identity() -> dict[str, Any]:
    protocol = RUN_ROOT / "protocol.json"
    expected_protocol = PACKAGE_DIR / "next_round" / "protocol.json"
    if expected_protocol.exists() and sha256_file(protocol) != sha256_file(expected_protocol):
        raise ValueError("run protocol differs from delivered protocol")
    dataset_files = _required_dataset_files()
    dataset_records = _hash_files(dataset_files, root=DATASET_ROOT, role="original_dataset")
    dataset_report = validate_dataset()
    dataset_identity = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "root": str(DATASET_ROOT),
        "files": dataset_records,
        "validation": dataset_report,
    }
    dataset_identity["identity_sha256"] = sha256_json(dataset_identity)
    write_json(RUN_ROOT / "DATASET_IDENTITY.json", dataset_identity)

    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol": file_identity(protocol, root=RUN_ROOT, role="protocol"),
        "dataset_identity_sha256": dataset_identity["identity_sha256"],
        "git_head": git_head,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "historical_training_dependencies": [],
        "declared_inputs": [str(DATASET_ROOT), str(BACKBONE_DIR), str(PURE_CACHE_DIR), str(UPSTREAM_CACHE_DIR)],
        "status": "PREFLIGHT_IN_PROGRESS",
    }
    run_identity["identity_sha256"] = sha256_json(run_identity)
    write_json(RUN_ROOT / "RUN_IDENTITY.json", run_identity)
    return run_identity


def build_cache_manifest() -> dict[str, Any]:
    paths = [PURE_CACHE_DIR / "z" / "z_index.json", PURE_CACHE_DIR / "z" / "z.f32.npy"]
    paths.extend(sorted((PURE_CACHE_DIR / "content").glob("*.json")))
    paths.extend(sorted((PURE_CACHE_DIR / "content" / "chunks").glob("chunk_*.npy")))
    records = _hash_files(paths, root=PURE_CACHE_DIR, role="pure_frozen_cache")
    write_jsonl(RUN_ROOT / "CACHE_MANIFEST.jsonl", records)
    report = {
        "status": "PASS",
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "manifest": file_identity(RUN_ROOT / "CACHE_MANIFEST.jsonl", root=RUN_ROOT, role="cache_manifest"),
    }
    report["identity_sha256"] = sha256_json(report)
    write_json(RUN_ROOT / "CACHE_IDENTITY.json", report)
    return report


def _hash_sample(rows: Iterable[dict[str, Any]], count: int, namespace: str) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{namespace}|{row['object_id']}".encode("utf-8")).digest(),
            str(row["object_id"]).encode("utf-8"),
        ),
    )
    if len(ranked) < count:
        raise ValueError(f"{namespace}: only {len(ranked)} objects available; need {count}")
    return ranked[:count]


def _feature_samples() -> dict[str, list[dict[str, Any]]]:
    query_split = {
        str(row["table_id"]): str(row["split"])
        for row in iter_jsonl(DATASET_ROOT / "query_tables" / "part-00000.jsonl")
    }
    categories: dict[str, list[dict[str, Any]]] = {
        "target_table": [],
        "text": [],
        "image": [],
        "train_query": [],
    }
    for row in iter_jsonl(UPSTREAM_DATA_DIR / "stage1_objects.jsonl"):
        object_id = str(row["object_id"])
        kind = str(row["object_type"])
        role = str(row.get("embedding_role") or "")
        if kind == "table" and role == "target":
            categories["target_table"].append(row)
        elif kind in ("text", "image"):
            categories[kind].append(row)
        elif kind == "table" and role == "query" and query_split.get(object_id) == "train":
            categories["train_query"].append(row)
    return {
        category: _hash_sample(rows, 16, f"V4.1_FEATURE_PROBE|{category}")
        for category, rows in categories.items()
    }


def _original_retrieval_batches(
    selected_ids: set[str], teacher_ids: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Recover exact original cache batches from the deterministic sharding script."""
    from cache_stage1_features import _base_object_batch_cost

    selected_batches: dict[str, list[dict[str, Any]]] = {}
    selected_records: dict[str, dict[str, Any]] = {}
    buffers: list[list[dict[str, Any]]] = [[], []]

    def flush(shard: int) -> None:
        batch = buffers[shard]
        if not batch:
            return
        if any(str(row["object_id"]) in selected_ids for row in batch):
            ordered = sorted(
                batch,
                key=lambda row: _base_object_batch_cost(
                    row,
                    input_dir=UPSTREAM_DATA_DIR,
                    max_image_pixels=1843200,
                ),
            )
            for start in range(0, len(ordered), 4):
                chunk = ordered[start : start + 4]
                for row in chunk:
                    object_id = str(row["object_id"])
                    if object_id in selected_ids:
                        selected_batches[object_id] = chunk
        batch.clear()

    for position, row in enumerate(iter_jsonl(UPSTREAM_DATA_DIR / "stage1_objects.jsonl")):
        object_id = str(row["object_id"])
        shard = position % 2
        kind = str(row["object_type"])
        batchable = kind != "table" and object_id not in teacher_ids
        if object_id in selected_ids:
            selected_records[object_id] = row
        if not batchable:
            if object_id in selected_ids:
                selected_batches[object_id] = [row]
            continue
        if buffers[shard] and str(buffers[shard][-1]["object_type"]) != kind:
            flush(shard)
        buffers[shard].append(row)
        if len(buffers[shard]) == 32:
            flush(shard)
    flush(0)
    flush(1)
    missing = selected_ids - selected_batches.keys()
    if missing:
        raise KeyError(f"could not recover original retrieval batches: {sorted(missing)}")
    return selected_batches, selected_records


def _content_generation_batches(selected_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    batch_sizes = {"text": 8, "image": 4}
    pending: dict[tuple[str, int], list[dict[str, Any]]] = {
        (kind, shard): [] for kind in batch_sizes for shard in range(2)
    }
    result: dict[str, list[dict[str, Any]]] = {}

    def finish(batch: list[dict[str, Any]]) -> None:
        selected = selected_ids & {str(row["asset_id"]) for row in batch}
        for object_id in selected:
            result[object_id] = list(batch)

    for part in sorted((DATASET_ROOT / "bridge_assets").glob("part-*.jsonl")):
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
        finish(batch)
    missing = selected_ids - result.keys()
    if missing:
        raise KeyError(f"could not recover content batches: {sorted(missing)}")
    return result


def _manifest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        object_id = str(row["object_id"])
        if object_id in output:
            raise ValueError(f"duplicate manifest object: {object_id}")
        output[object_id] = row
    return output


def _allclose_report(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    shape_equal = tuple(actual.shape) == tuple(expected.shape)
    if not shape_equal:
        return {
            "shape_equal": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "allclose": False,
            "max_abs": None,
            "max_rel": None,
        }
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-30)
    return {
        "shape_equal": True,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "allclose": bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": float((difference / denominator).max()) if difference.numel() else 0.0,
    }


def _raw_asset_item(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    kind = str(row["asset_type"])
    return {
        "text": str(row["content"]) if kind == "text" else None,
        "image": str(Path(str(row.get("local_path") or row.get("relative_path"))).resolve()) if kind == "image" else None,
        "instruction": prompt,
    }


def verify_feature_provenance(device: str = "cuda:0") -> dict[str, Any]:
    """Recompute the locked sample using the exact original batching recipes."""
    from cache_stage1_features import (
        EMBEDDING_INSTRUCTIONS,
        _load_embedder_class,
        _source_fingerprint,
        build_base_object_features_batch,
        build_object_features,
        encode_inputs,
        teacher_object_ids,
    )
    from .content import ContentStore, compress_bins

    _require_configured()
    if UPSTREAM_DATA_DIR is None:
        raise RuntimeError("protocol paths.upstream_data_dir is required to replay the feature recipe")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("BLOCKED_GPU_IDENTITY: expected one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = str(getattr(properties, "uuid", ""))
    expected_uuid = (GPU_UUID or "").removeprefix("GPU-")
    if not expected_uuid or runtime_uuid.lower() != expected_uuid.lower():
        raise RuntimeError(
            f"BLOCKED_GPU_IDENTITY: runtime cuda:0 is GPU-{runtime_uuid}, protocol pins {GPU_UUID}"
        )
    torch.cuda.set_device(torch.device(device))

    samples = _feature_samples()
    sampled_records = [row for rows in samples.values() for row in rows]
    selected_ids = {str(row["object_id"]) for row in sampled_records}
    evidence_ids = {
        str(row["object_id"])
        for row in sampled_records
        if str(row["object_type"]) in ("text", "image")
    }
    teacher_ids = teacher_object_ids(
        [UPSTREAM_DATA_DIR / "edge_lists.jsonl", UPSTREAM_DATA_DIR / "target_lists.jsonl"],
        split=None,
    )
    retrieval_batches, source_records = _original_retrieval_batches(selected_ids, teacher_ids)
    content_batches = _content_generation_batches(evidence_ids)
    base_manifest = _manifest_by_id(UPSTREAM_CACHE_DIR / "manifest.jsonl")

    z_index = json.loads((PURE_CACHE_DIR / "z" / "z_index.json").read_text(encoding="utf-8"))
    z_positions = {str(object_id): index for index, object_id in enumerate(z_index["ids"])}
    z = np.load(PURE_CACHE_DIR / "z" / "z.f32.npy", mmap_mode="r", allow_pickle=False)
    content = ContentStore(PURE_CACHE_DIR / "content")

    embedder = _load_embedder_class(BACKBONE_DIR)(
        str(BACKBONE_DIR),
        torch_dtype=torch.bfloat16,
        max_length=8192,
        min_pixels=4096,
        max_pixels=1843200,
    )
    embedder.model.to(torch.device(device))
    embedder.model.eval()

    recomputed_z: dict[str, torch.Tensor] = {}
    completed_batches: set[tuple[str, ...]] = set()
    for object_id in sorted(selected_ids, key=lambda value: value.encode("utf-8")):
        batch = retrieval_batches[object_id]
        batch_key = tuple(str(row["object_id"]) for row in batch)
        if batch_key in completed_batches:
            continue
        completed_batches.add(batch_key)
        if len(batch) == 1:
            record = batch[0]
            result = build_object_features(
                embedder,
                record,
                input_dir=UPSTREAM_DATA_DIR,
                instruction=None,
                storage_dtype=torch.bfloat16,
                include_hidden=str(record["object_id"]) in teacher_ids,
                include_row_embeddings=str(record["object_type"]) == "table",
                table_row_batch_size=8,
                table_tokens_per_group=1,
            )
            recomputed_z[str(record["object_id"])] = result["embedding"].cpu()
        else:
            outputs = build_base_object_features_batch(
                embedder,
                batch,
                input_dir=UPSTREAM_DATA_DIR,
                instruction=None,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(f"original batch replay collapsed: {batch_key}")
            for record, output in zip(batch, outputs, strict=True):
                recomputed_z[str(record["object_id"])] = output["embedding"].cpu()

    content_recomputed: dict[str, torch.Tensor] = {}
    query_rows_recomputed: dict[str, torch.Tensor] = {}
    table_ids = {
        str(row["object_id"])
        for row in sampled_records
        if str(row["object_type"]) == "table"
    }
    for object_id in sorted(table_ids, key=lambda value: value.encode("utf-8")):
        record = source_records[object_id]
        output = build_object_features(
            embedder,
            record,
            input_dir=UPSTREAM_DATA_DIR,
            instruction=None,
            storage_dtype=torch.bfloat16,
            include_hidden=True,
            include_row_embeddings=str(record.get("embedding_role")) == "query",
            table_row_batch_size=8,
            table_tokens_per_group=1,
        )
        content_recomputed[object_id] = output["hidden_states"].to(torch.float16).cpu()
        if "row_embeddings" in output:
            query_rows_recomputed[object_id] = output["row_embeddings"].float().cpu()

    content_candidates: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for object_id in sorted(evidence_ids, key=lambda value: value.encode("utf-8")):
        record = source_records[object_id]
        output = build_object_features(
            embedder,
            record,
            input_dir=UPSTREAM_DATA_DIR,
            instruction=None,
            storage_dtype=torch.bfloat16,
            include_hidden=True,
            include_row_embeddings=False,
        )
        content_candidates[object_id]["upstream_single"] = compress_bins(
            output["hidden_states"].float()
        ).to(torch.float16).cpu()

    completed_content_batches: set[tuple[str, ...]] = set()
    for object_id in sorted(evidence_ids, key=lambda value: value.encode("utf-8")):
        batch = content_batches[object_id]
        batch_key = tuple(str(row["asset_id"]) for row in batch)
        if batch_key in completed_content_batches:
            continue
        completed_content_batches.add(batch_key)
        kind = str(batch[0]["asset_type"])
        outputs = encode_inputs(
            embedder,
            [_raw_asset_item(row, EMBEDDING_INSTRUCTIONS[("evidence", kind)]) for row in batch],
            include_hidden=True,
        )
        if len(outputs) != len(batch):
            raise RuntimeError(f"content generation batch replay collapsed: {batch_key}")
        for row, (_embedding, hidden, _input_ids) in zip(batch, outputs, strict=True):
            assert hidden is not None
            row_id = str(row["asset_id"])
            if row_id in evidence_ids:
                content_candidates[row_id]["content_encoder_batch"] = compress_bins(
                    hidden.float()
                ).to(torch.float16).cpu()

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    categories_by_id = {
        str(row["object_id"]): category
        for category, rows in samples.items()
        for row in rows
    }
    for object_id in sorted(selected_ids, key=lambda value: value.encode("utf-8")):
        manifest_record = base_manifest.get(object_id)
        source_record = source_records[object_id]
        source_fingerprint_ok = bool(
            manifest_record
            and manifest_record.get("source_fingerprint") == _source_fingerprint(source_record)
        )
        cached_z = torch.from_numpy(np.array(z[z_positions[object_id]], copy=True))
        z_result = _allclose_report(recomputed_z[object_id], cached_z)
        cached_content = content.get(object_id)
        if object_id in content_recomputed:
            candidates = {"locked_table_recipe": content_recomputed[object_id]}
        else:
            candidates = content_candidates[object_id]
        content_results = {
            name: _allclose_report(candidate, cached_content)
            for name, candidate in candidates.items()
        }
        matching_content_recipes = [
            name for name, result in content_results.items() if result["allclose"]
        ]
        row_result = None
        if object_id in query_rows_recomputed:
            feature_path = UPSTREAM_CACHE_DIR / str(manifest_record["feature_path"])
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            row_result = _allclose_report(query_rows_recomputed[object_id], payload["row_embeddings"])
        passed = bool(
            source_fingerprint_ok
            and z_result["allclose"]
            and matching_content_recipes
            and (row_result is None or row_result["allclose"])
        )
        record = {
            "object_id": object_id,
            "category": categories_by_id[object_id],
            "source_fingerprint_ok": source_fingerprint_ok,
            "retrieval_batch_ids": [str(row["object_id"]) for row in retrieval_batches[object_id]],
            "z": z_result,
            "content": content_results,
            "matching_content_recipes": matching_content_recipes,
            "row_embeddings": row_result,
            "status": "PASS" if passed else "FAIL",
        }
        comparisons.append(record)
        if not passed:
            failures.append(record)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not failures and len(comparisons) == 64 else "FAIL",
        "device": "cuda:0",
        "gpu_uuid": f"GPU-{runtime_uuid}",
        "gpu_model": properties.name,
        "selection": {category: [str(row["object_id"]) for row in rows] for category, rows in samples.items()},
        "rtol": 1e-4,
        "atol": 1e-5,
        "comparisons": comparisons,
        "failure_count": len(failures),
    }
    output = RUN_ROOT / "tests" / "real_tensor_probes" / "feature_provenance.json"
    write_json(output, report)

    recipe_path = RUN_ROOT / "FROZEN_RECIPE_LOCK.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe["status"] = "PASS" if report["status"] == "PASS" else "BLOCKED_FEATURE_PROVENANCE"
    recipe["numeric_probe_result"] = file_identity(output, root=RUN_ROOT, role="feature_provenance_probe")
    recipe.pop("recipe_sha256", None)
    recipe["recipe_sha256"] = sha256_json(recipe)
    write_json(recipe_path, recipe)

    phase = {
        "schema_version": SCHEMA_VERSION,
        "phase": "PREFLIGHT",
        "status": "PASS_READY_FOR_REFERENCE_TESTS" if report["status"] == "PASS" else "BLOCKED_FEATURE_PROVENANCE",
        "formal_training_started": False,
    }
    write_json(RUN_ROOT / "PHASE_STATUS.json", phase)
    if report["status"] != "PASS":
        raise RuntimeError("BLOCKED_FEATURE_PROVENANCE")
    return report


def write_access_ledger() -> None:
    records = [
        {
            "schema_version": SCHEMA_VERSION,
            "resolved_realpath": str(path.resolve()),
            "kind": kind,
            "source": source,
            "stage": "PREFLIGHT",
            "mode": "read",
        }
        for path, kind, source in (
            (DATASET_ROOT, "original_dataset", "fresh_allowed"),
            (BACKBONE_DIR, "public_frozen_backbone", "fresh_allowed"),
            (PURE_CACHE_DIR, "candidate_pure_cache", "reuse_only_after_numeric_probe"),
            (UPSTREAM_CACHE_DIR, "candidate_row_cache", "reuse_only_after_numeric_probe"),
            (UPSTREAM_DATA_DIR, "pure_extractor_input_reconstruction", "provenance_only_not_training_material"),
            (PACKAGE_DIR, "locked_protocol", "fresh_allowed"),
        )
        if path is not None
    ]
    write_jsonl(RUN_ROOT / "INPUT_ACCESS_LEDGER.jsonl", records)


def run_lock() -> dict[str, Any]:
    _require_configured()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    result = {
        "run_identity": build_identity(),
        "recipe": build_recipe_lock(),
        "aliases": build_content_aliases(),
        "cache": build_cache_manifest(),
    }
    write_access_ledger()
    write_json(RUN_ROOT / "PHASE_STATUS.json", {
        "schema_version": SCHEMA_VERSION,
        "phase": "PREFLIGHT",
        "status": "SOURCE_AND_CACHE_LOCKED_PENDING_NUMERIC_PROBE",
        "formal_training_started": False,
    })
    return result

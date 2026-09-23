from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import torch

from . import PROTOCOL_ID, PROTOCOL_VERSION
from .config import Paths
from .io import file_identity, iter_jsonl, sha256_file, write_json, write_jsonl

# v3.1 is self-contained.  Do not pull audit evidence or source maps from the
# older rebuild package; the delivered decision/control/acceptance files are
# part of this run's immutable protocol lock.
AUDIT_FILES = (
    "README.md",
    "CODEX_PROMPT.md",
    "EXPERIMENT_SPEC.zh-CN.md",
    "protocol.json",
    "EXECUTION_DAG.json",
    "DECISION_RECORD.md",
    "CONTROL_MATRIX.md",
    "ACCEPTANCE.md",
    "ENCODER_PROMPTS.json",
)


def _dataset_files(paths: Paths) -> list[Path]:
    root = paths.dataset_root
    files = [root / name for name in ("dataset_manifest.json", "splits.json", "qrels.jsonl")]
    for directory in ("query_tables", "data_lake_tables", "source_tables", "bridge_assets", "evidence_recoveries"):
        files.extend(sorted((root / directory).glob("part-*.jsonl")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required original input shards: {missing}")
    return files


def resolve_inputs(paths: Paths) -> dict:
    dataset_files = _dataset_files(paths)
    package_files = [paths.package_dir / name for name in AUDIT_FILES]
    backbone_files = [
        paths.backbone_dir / "config.json",
        paths.backbone_dir / "preprocessor_config.json",
        paths.backbone_dir / "tokenizer_config.json",
        paths.backbone_dir / "scripts" / "qwen3_vl_embedding.py",
    ]
    missing = [str(path) for path in package_files + backbone_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing protocol/backbone contract files: {missing}")

    manifest = json.loads((paths.dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = json.loads((paths.dataset_root / "splits.json").read_text(encoding="utf-8"))
    root_inputs = {
        "protocol_id": PROTOCOL_ID,
        "version": PROTOCOL_VERSION,
        "dataset_root": str(paths.dataset_root),
        "public_backbone": str(paths.backbone_dir),
        "audit_package": str(paths.package_dir),
        "allowed_root_roles": ["original_dataset", "public_frozen_backbone", "current_v3_code"],
        "forbidden_training_inputs": [
            "historical_task_weights", "historical_optimizer", "historical_PCA",
            "historical_training_lists", "historical_hard_negatives", "historical_graphs",
            "historical_teacher_logits", "learned_compression_cache",
        ],
        "dataset_files": [file_identity(path, root=paths.dataset_root) for path in dataset_files],
        "package_files": [file_identity(path, root=paths.package_dir) for path in package_files],
        "backbone_contract_files": [file_identity(path, root=paths.backbone_dir) for path in backbone_files],
    }
    write_json(paths.work_dir / "ROOT_INPUTS.json", root_inputs)
    write_json(paths.work_dir / "DATA_SPLIT_REPORT.json", {
        "split_policy": manifest.get("split_schema_version"),
        "data_lake_scope": splits.get("data_lake_scope"),
        "query_table_counts": splits.get("query_table_counts"),
        "data_lake_table_count": splits.get("data_lake_table_count"),
        "train_scope": "all_original_train",
        "legacy_train_fit": False,
        "dev_test_labels_visible_to_training": False,
        "transductive_content_scope": "shared_unlabelled_lake_targets_and_evidence",
        "test_status": "historically_exposed_regression",
    })
    write_json(paths.work_dir / "SCHEMA_MAP.json", {
        "query": {"artifact": "query_tables", "id": "table_id", "split": "split", "rows": "rows"},
        "target": {
            "artifact": "data_lake_tables", "id": "table_id", "source_reference": "source_table_ref",
            "resolved_artifact": "source_tables", "resolved_id": "source_table_id",
        },
        "qrel": {
            "artifact": "qrels.jsonl", "query": "query_table_id", "target": "target_table_id",
            "positive": "rel > 0", "direct_reason": "explicit_visible_join_column", "split": "split",
        },
        "witness": {
            "artifact": "evidence_recoveries", "query": "query_table_id", "target": "target_table_id",
            "asset": "evidence.asset_id", "modality": "evidence.asset_type", "split": "split",
        },
    })
    prompts = json.loads((paths.package_dir / "ENCODER_PROMPTS.json").read_text(encoding="utf-8"))
    write_json(paths.work_dir / "ENCODER_CONTRACT.json", {
        "backbone": "Qwen3-VL-Embedding-8B",
        "backbone_dir": str(paths.backbone_dir),
        "frozen": True,
        "dimension": 4096,
        "compute_dtype": "bfloat16",
        "max_tokens": 8192,
        "z": {"dtype": "float32", "pool": "last_valid_token", "normalization": "l2"},
        "content": {"disk_dtype": "float16", "bins": 64, "continuous_bins": True},
        "query": {"max_rows": None, "row_summaries": "one_independent_per_original_row"},
        "target": {"max_rows": 20, "max_cell_chars": 1024, "row_format": "values"},
        "prompts_sha256": sha256_file(paths.package_dir / "ENCODER_PROMPTS.json"),
        "instructions": prompts["instructions"],
        "old_cache_compatibility_rule": (
            "actual per-object contract equality is required; conflicting legacy metadata is quarantined "
            "until table group counts and frozen-backbone recomputation prove the stored objects use 20 rows"
        ),
    })
    code_files = sorted((Path(__file__).resolve().parent).glob("*.py"))
    write_json(paths.work_dir / "SOURCE_LOCK.json", {
        "protocol_id": PROTOCOL_ID,
        "version": PROTOCOL_VERSION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "code": [file_identity(path) for path in code_files],
        "protocol": [file_identity(path) for path in package_files],
    })
    return root_inputs


def build_content_aliases(paths: Paths) -> dict[str, str]:
    records: list[dict] = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    errors: list[dict] = []
    modality_counts: Counter[str] = Counter()
    image_digests: dict[Path, str] = {}
    for part in sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            asset_id = str(row["asset_id"])
            modality = str(row["asset_type"])
            modality_counts[modality] += 1
            if modality == "text":
                if "content" not in row or not isinstance(row["content"], str):
                    errors.append({"asset_id": asset_id, "kind": "missing_text_content", "locator": row["_locator"]})
                    continue
                raw = row["content"].encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                source = row["_locator"]
            elif modality == "image":
                image_path = Path(str(row.get("local_path") or row.get("relative_path") or ""))
                if not image_path.is_file():
                    errors.append({"asset_id": asset_id, "kind": "missing_image_file", "path": str(image_path),
                                   "locator": row["_locator"]})
                    continue
                resolved_image = image_path.resolve()
                digest = image_digests.get(resolved_image)
                if digest is None:
                    digest = sha256_file(resolved_image)
                    image_digests[resolved_image] = digest
                advertised = row.get("sha256")
                if advertised and str(advertised) != digest:
                    errors.append({"asset_id": asset_id, "kind": "image_sha256_mismatch", "path": str(image_path),
                                   "advertised": str(advertised), "actual": digest, "locator": row["_locator"]})
                    continue
                source = str(resolved_image)
            else:
                errors.append({"asset_id": asset_id, "kind": "unknown_modality", "modality": modality,
                               "locator": row["_locator"]})
                continue
            content_key = f"{modality}:sha256:{digest}"
            groups[(modality, digest)].append(asset_id)
            records.append({
                "asset_id": asset_id,
                "modality": modality,
                "content_sha256": digest,
                "content_key": content_key,
                "source": source,
                "source_record": row["_locator"],
            })
    if errors:
        write_jsonl(paths.work_dir / "CONTENT_ALIAS_ERRORS.jsonl", errors)
        raise RuntimeError(f"content alias construction failed for {len(errors)} assets")

    canonical: dict[str, str] = {}
    group_size: dict[str, int] = {}
    for members in groups.values():
        chosen = min(members, key=lambda value: value.encode("utf-8"))
        for asset_id in members:
            canonical[asset_id] = chosen
            group_size[asset_id] = len(members)
    for record in records:
        record["canonical_id"] = canonical[record["asset_id"]]
        record["alias_group_size"] = group_size[record["asset_id"]]
    records.sort(key=lambda row: row["asset_id"].encode("utf-8"))
    write_jsonl(paths.work_dir / "CONTENT_ALIASES.jsonl.gz", records, gzip_output=True)
    summary = {
        "objects": len(records),
        "canonical_objects": len(groups),
        "alias_objects": len(records) - len(groups),
        "modality_objects": dict(sorted(modality_counts.items())),
        "unique_image_files_hashed": len(image_digests),
        "canonical_rule": "UTF-8 minimum asset_id within identical modality and raw-content SHA256",
        "text_bytes": "decoded JSON content encoded exactly as UTF-8; no cleaning",
        "image_bytes": "SHA256 recomputed from local image file and checked against dataset record",
        "records": "CONTENT_ALIASES.jsonl.gz",
    }
    write_json(paths.work_dir / "CONTENT_ALIASES.json", summary)
    write_json(paths.labels_dir / "content_canonical.json", canonical)
    return canonical

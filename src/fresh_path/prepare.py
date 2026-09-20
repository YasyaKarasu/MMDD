"""P0 entry points: root inputs, split report, labels and raw candidates."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from . import config, inputs
from .inputs import sha256_file

CONTENT_KEY_SOURCE = Path("work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl")


def _role(path: Path, kind: str, purpose: str, *, training: bool) -> dict:
    path = Path(path)
    payload = {
        "path": str(path.resolve()),
        "source_kind": kind,
        "purpose": purpose,
        "training_readable": training,
    }
    if path.is_file():
        payload["bytes"] = path.stat().st_size
        payload["fingerprint"] = sha256_file(path)
    elif path.is_dir():
        entries = sorted(p.name for p in path.iterdir())
        import hashlib

        payload["fingerprint"] = hashlib.sha256("\n".join(entries).encode()).hexdigest()
        payload["entries"] = len(entries)
    else:
        raise FileNotFoundError(f"missing root input {path}")
    return payload


def write_root_inputs(paths: config.Paths, protocol: dict, out: Path) -> dict:
    ds = paths.dataset_root
    backbone = paths.backbone_dir
    payload = {
        "run_id": "mmdd_stage1_fresh_path_v2_1_20260920",
        "protocol_version": protocol["version"],
        "dataset": _role(ds, "raw_dataset", "original 20K snapshot root", training=True),
        "raw_splits": _role(ds / "splits.json", "raw_dataset", "official query-only split", training=True),
        "raw_train_GT": _role(ds / "qrels.jsonl", "raw_dataset", "train positives + witness records", training=True),
        "raw_dev_GT": _role(ds / "qrels.jsonl", "raw_dataset", "dev positives (evaluation only)", training=False),
        "raw_test_GT_location": _role(ds / "qrels.jsonl", "raw_dataset", "test positives (opened once at the end)", training=False),
        "public_backbone": _role(backbone, "public_backbone", "frozen Qwen3-VL-Embedding-8B", training=False),
        "encoder_contract": _role(ds / "dataset_manifest.json", "raw_dataset", "encoder serialization contract", training=False),
        "code": _role(Path(__file__).resolve().parents[1], "source_code", "this run's pipeline code", training=False),
        "protocol": _role(paths.protocol, "protocol", "v2.1 protocol json", training=False),
        "pure_caches": [],
    }
    forbidden = [r for r in config.FORBIDDEN_ROOT_ROLES if r in payload]
    if forbidden:
        raise ValueError(f"forbidden roles: {forbidden}")
    (out / "ROOT_INPUTS.json").write_text(json.dumps(payload, indent=1))
    return payload


def write_schema_map(view: inputs.DatasetView, out: Path) -> dict:
    mapping = {
        "format": "sharded_jsonl",
        "split_source": {"artifact": "query_tables", "field": "split", "policy": view.splits["split_policy"]},
        "gold": {"artifact": "qrels", "fields": {"query": "query_table_id", "target": "target_table_id",
                                                 "relevance": "rel", "reason": "reason", "split": "split"},
                 "rule": "rel>0 is positive; reason explicit_visible_join_column is direct, model_recoverable_join_column is implicit"},
        "witness": {"artifact": "evidence_recoveries", "fields": {"query": "query_table_id", "target": "target_table_id",
                                                                  "asset": "evidence.asset_id", "modality": "evidence.asset_type",
                                                                  "split": "split"},
                    "support_rule": "every recovery record in this snapshot is auto-check supported "
                                    "(dataset_manifest.model_endpoints.auto_check: checked=reviewed=supported=23592, "
                                    "contradicted=0, insufficient=0); no extra verdict filter is applied"},
        "target_universe": {"artifact": "data_lake_tables", "field": "table_id",
                            "expansion": "source_table_ref -> source_tables by source_table_id"},
        "target_serialization": {"artifact": "data_lake_tables", "fields": {"columns": "columns", "rows": "rows"},
                                 "rule": "Columns: ... then Row: ... ; first 20 rows (user override U02); cells <=1024 chars"},
        "query_serialization": {"artifact": "query_tables", "fields": {"columns": "columns", "rows": "rows"},
                                "rule": "all rows, all visible columns, no cell truncation"},
        "evidence_content": {"artifact": "bridge_assets", "fields": {"text": "content", "image": "local_path"}},
        "content_alias": {"artifact": "evidence_content_keys.jsonl", "field": "content_key",
                          "format": "<image|text>:<sha256>"},
        "source_group": {"artifact": "query_tables", "field": "source_table_id"},
        "forbidden_as_input": ["qrels", "recovery target ids", "target_table_ids", "hidden_attributes",
                               "recovered_attribute", "candidate ranks/sources", "object ids as tokens"],
    }
    (out / "SCHEMA_MAP.json").write_text(json.dumps({"map": mapping, "todos": []}, indent=1, ensure_ascii=False))
    return mapping


def write_split_report(view: inputs.DatasetView, out: Path) -> dict:
    per_split: dict[str, dict] = {}
    for split in ("train", "dev", "test"):
        qs = [q for q, s in view.query_split.items() if s == split]
        per_split[split] = {"queries": len(qs)}
    group_by_split: dict[str, set] = {"train": set(), "dev": set(), "test": set()}
    for q, split in view.query_split.items():
        group_by_split[split].add(view.query_source_group.get(q))
    overlaps = {
        "train_dev": len(group_by_split["train"] & group_by_split["dev"]),
        "train_test": len(group_by_split["train"] & group_by_split["test"]),
        "dev_test": len(group_by_split["dev"] & group_by_split["test"]),
    }
    report = {
        "split_source": "original dataset query_tables.split",
        "split_policy": view.splits["split_policy"],
        "data_lake_scope": view.splits["data_lake_scope"],
        "counts": per_split,
        "source_groups_per_split": {k: len(v) for k, v in group_by_split.items()},
        "source_group_cross_split_overlap": overlaps,
        "calibration_bucket": None,
        "note": "The shared data lake is retrieved by every split; shared lake objects are not query-label leakage.",
    }
    (out / "DATA_SPLIT_REPORT.json").write_text(json.dumps(report, indent=1))
    return report


def write_labels(labels: inputs.TrainLabels, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    n = inputs.write_jsonl_gz(
        out / "train_queries.jsonl.gz",
        (labels.queries[q] for q in sorted(labels.queries, key=lambda x: x.encode("utf-8"))),
    )
    inputs.write_jsonl_gz(
        out / "asset_epos.jsonl.gz",
        ({"asset_id": a, "epos": t} for a, t in sorted(labels.epos.items(), key=lambda kv: kv[0].encode("utf-8"))),
    )
    inputs.write_jsonl_gz(out / "label_conflicts.jsonl", labels.conflicts)
    (out / "label_stats.json").write_text(json.dumps(labels.stats, indent=1))
    (out / "legal_targets.json").write_text(json.dumps(labels.legal))
    return {"queries_written": n, "stats": labels.stats}


def load_labels(labels_dir: Path) -> inputs.TrainLabels:
    labels = inputs.TrainLabels()
    labels.queries = {r["query_id"]: r for r in inputs.read_jsonl_gz(Path(labels_dir) / "train_queries.jsonl.gz")}
    labels.epos = {r["asset_id"]: r["epos"] for r in inputs.read_jsonl_gz(Path(labels_dir) / "asset_epos.jsonl.gz")}
    labels.legal = json.loads((Path(labels_dir) / "legal_targets.json").read_text())
    labels.stats = json.loads((Path(labels_dir) / "label_stats.json").read_text())
    conflicts = Path(labels_dir) / "label_conflicts.jsonl"
    labels.conflicts = list(inputs.read_jsonl_gz(conflicts)) if conflicts.exists() else []
    return labels

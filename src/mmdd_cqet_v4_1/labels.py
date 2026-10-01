"""Labels, ground truth, and protection sets for CLEAN-QET v4.0."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import SCHEMA_VERSION
from .config import Paths
from .data import (
    iter_jsonl,
    load_split_gt,
    read_json,
    sha256_file,
    utf8_sorted,
    write_json,
    write_jsonl,
    write_jsonl_gz,
)

RELATIONS = ("QT", "Q_text", "Q_image", "text_T", "image_T")


@dataclass
class Labels:
    queries: dict[str, dict]
    epos: dict[str, list[str]]
    legal_targets: list[str]
    edge_anchors: list[dict]
    canonical_map: dict[str, str]
    modality: dict[str, str]
    content_hash: dict[str, str]
    canonical_text: list[str]
    canonical_image: list[str]
    stats: dict = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return utf8_sorted(self.queries)

    def library(self, relation: str) -> list[str]:
        if relation == "QT":
            return self.legal_targets
        if relation == "Q_text":
            return self.canonical_text
        if relation == "Q_image":
            return self.canonical_image
        if relation in ("text_T", "image_T"):
            return self.legal_targets
        raise ValueError(f"unknown relation: {relation}")

    def protect_set(self, query_id: str, target_id: str) -> set[str]:
        """SPEC 7.2: Protect(q, t) = W(q, t) union (union_{t' in G_q} W(q, t')) union E^+(t)."""
        entry = self.queries.get(query_id)
        if not entry:
            return set()
        w_qt = set(entry["W"].get(target_id, ()))
        w_all_g = {asset for assets in entry["W"].values() for asset in assets}
        # E^+(t)
        e_plus_t = {asset for asset, targets in self.epos.items() if target_id in targets}
        return w_qt | w_all_g | e_plus_t


def build_labels(paths: Paths, canonical_map: dict[str, str]) -> dict:
    root = paths.dataset_root
    labels_dir = paths.labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read query tables
    query_rows = list(iter_jsonl(root / "query_tables" / "part-00000.jsonl"))
    query_by_id = {str(row["table_id"]): row for row in query_rows}

    # 2. Read lake tables
    target_ids = {
        str(row["table_id"])
        for row in iter_jsonl(root / "data_lake_tables" / "part-00000.jsonl")
    }

    # 3. Read asset modalities
    asset_modality: dict[str, str] = {}
    for part in sorted((root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            asset_modality[str(row["asset_id"])] = str(row["asset_type"])

    # 4. Read qrels
    gold: dict[str, set[str]] = defaultdict(set)
    direct: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(root / "qrels.jsonl"):
        if str(row.get("split")) != "train" or int(row.get("rel", 0)) <= 0:
            continue
        qid = str(row["query_table_id"])
        tid = str(row["target_table_id"])
        gold[qid].add(tid)
        reason = str(row.get("reason"))
        reasons[qid].add(reason)
        if reason == "explicit_visible_join_column":
            direct[qid].add(tid)

    # 5. Read evidence recoveries
    witness: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in iter_jsonl(root / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != "train":
            continue
        qid = str(row["query_table_id"])
        tid = str(row["target_table_id"])
        evidence = row.get("evidence") or {}
        raw_asset = str(evidence.get("asset_id"))
        if raw_asset not in canonical_map:
            continue
        asset = canonical_map[raw_asset]
        witness[(qid, tid)].add(asset)

    # 6. Assemble train queries
    all_train_queries = utf8_sorted(
        qid for qid, row in query_by_id.items()
        if str(row.get("split")) == "train"
    )
    train_queries = [qid for qid in all_train_queries if qid in gold]

    epos: dict[str, set[str]] = defaultdict(set)
    query_records: list[dict] = []
    for qid in train_queries:
        g_set = gold[qid]
        d_set = direct[qid]
        w_map: dict[str, list[str]] = {}
        for (w_qid, tid), assets in witness.items():
            if w_qid != qid or tid not in g_set:
                continue
            w_map[tid] = utf8_sorted(assets)
            for asset in w_map[tid]:
                epos[asset].add(tid)

        qpos = {
            m: utf8_sorted(
                asset for assets in w_map.values() for asset in assets
                if asset_modality.get(asset) == m
            )
            for m in ("text", "image")
        }
        reason_set = reasons[qid]
        if reason_set == {"model_recoverable_join_column"}:
            qkind = "implicit"
        elif reason_set == {"explicit_visible_join_column"}:
            qkind = "explicit"
        else:
            qkind = "mixed"

        query_records.append(
            {
                "query_id": qid,
                "split": "train",
                "source_group": str(query_by_id[qid]["source_table_id"]),
                "query_kind": qkind,
                "G": utf8_sorted(g_set),
                "D": utf8_sorted(d_set),
                "W": {tid: w_map[tid] for tid in utf8_sorted(w_map)},
                "Qpos": qpos,
            }
        )

    query_records.sort(key=lambda r: r["query_id"].encode("utf-8"))
    epos_records = [
        {"asset_id": asset, "modality": asset_modality[asset], "positive_ids": utf8_sorted(targets)}
        for asset, targets in sorted(epos.items(), key=lambda p: p[0].encode("utf-8"))
    ]

    edge_anchors: list[dict] = []
    for record in query_records:
        qid = record["query_id"]
        edge_anchors.append(
            {
                "item_id": f"QT:{qid}",
                "relation": "QT",
                "anchor_id": qid,
                "positive_ids": record["G"],
            }
        )
        for m in ("text", "image"):
            positives = record["Qpos"][m]
            if positives:
                edge_anchors.append(
                    {
                        "item_id": f"Q_{m}:{qid}",
                        "relation": f"Q_{m}",
                        "anchor_id": qid,
                        "positive_ids": positives,
                    }
                )

    for row in epos_records:
        asset = row["asset_id"]
        rel = f"{row['modality']}_T"
        edge_anchors.append(
            {
                "item_id": f"{rel}:{asset}",
                "relation": rel,
                "anchor_id": asset,
                "positive_ids": row["positive_ids"],
            }
        )

    edge_anchors.sort(key=lambda r: (RELATIONS.index(r["relation"]), r["anchor_id"].encode("utf-8")))

    write_jsonl_gz(labels_dir / "train_queries.jsonl.gz", query_records)
    write_jsonl_gz(labels_dir / "asset_epos.jsonl.gz", epos_records)
    write_jsonl_gz(labels_dir / "edge_anchors.jsonl.gz", edge_anchors)
    write_json(labels_dir / "legal_targets.json", utf8_sorted(target_ids))
    write_jsonl_gz(
        labels_dir / "training_exposure.jsonl.gz",
        (
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": qid,
                "split": "train",
                "eligible": qid in gold,
                "gold_count": len(gold.get(qid, ())),
                "exclusion_reason": None if qid in gold else "no_positive_qrel",
            }
            for qid in all_train_queries
        ),
    )

    contract_dir = labels_dir / "train"
    query_contract = [
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": row["query_id"],
            "split": "train",
            "source_group": row["source_group"],
            "query_kind": row["query_kind"],
        }
        for row in query_records
    ]
    qrel_contract = [
        {"schema_version": SCHEMA_VERSION, "query_id": qid, "target_id": target, "rel": 1}
        for qid in utf8_sorted(gold)
        for target in utf8_sorted(gold[qid])
    ]
    witness_contract = [
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": row["query_id"],
            "target_id": target,
            "evidence_id": evidence,
            "canonical_evidence_id": evidence,
            "modality": asset_modality[evidence],
        }
        for row in query_records
        for target in utf8_sorted(row["W"])
        for evidence in row["W"][target]
    ]
    closure_contract = [
        {
            "schema_version": SCHEMA_VERSION,
            "query_id": row["query_id"],
            "G": row["G"],
            "W": row["W"],
            "Qpos": row["Qpos"],
            "protect": {
                target: utf8_sorted(
                    set(row["W"].get(target, ()))
                    | {e for values in row["W"].values() for e in values}
                    | {e for e, targets_ in epos.items() if target in targets_}
                )
                for target in row["G"]
            },
        }
        for row in query_records
    ]
    edge_contract = [{"schema_version": SCHEMA_VERSION, **row} for row in edge_anchors]
    write_jsonl_gz(contract_dir / "queries.jsonl.gz", query_contract)
    write_jsonl_gz(contract_dir / "qrels.jsonl.gz", qrel_contract)
    write_jsonl_gz(contract_dir / "witness.jsonl.gz", witness_contract)
    write_jsonl_gz(contract_dir / "closure.jsonl.gz", closure_contract)
    write_jsonl_gz(contract_dir / "edge_anchors.jsonl.gz", edge_contract)

    stats = {
        "all_original_train_queries": len(all_train_queries),
        "train_queries": len(query_records),
        "excluded_train_queries": len(all_train_queries) - len(query_records),
        "legal_targets": len(target_ids),
        "gold_pairs": sum(len(r["G"]) for r in query_records),
        "direct_pairs": sum(len(r["D"]) for r in query_records),
        "witness_target_pairs": sum(len(r["W"]) for r in query_records),
        "canonical_epos_assets": len(epos_records),
        "edge_anchors_count": len(edge_anchors),
    }
    write_json(labels_dir / "label_stats.json", stats)
    return stats


def export_eval_labels(paths: Paths, canonical_map: dict[str, str], split: str) -> dict[str, dict]:
    """Materialize one evaluation split only when its firewall permits reading it."""
    if split not in ("dev", "test"):
        raise ValueError(f"invalid evaluation split: {split}")
    gt = load_split_gt(paths, split, canonical_map)
    asset_modality: dict[str, str] = {}
    for part in sorted((paths.dataset_root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            raw_id = str(row["asset_id"])
            if raw_id in canonical_map:
                asset_modality[canonical_map[raw_id]] = str(row["asset_type"])

    rows_by_name = {
        "queries.jsonl": [
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": qid,
                "split": split,
                "source_group": gt[qid]["source_group"],
                "query_kind": gt[qid]["kind"],
            }
            for qid in utf8_sorted(gt)
        ],
        "qrels.jsonl": [
            {"schema_version": SCHEMA_VERSION, "query_id": qid, "target_id": target, "rel": 1}
            for qid in utf8_sorted(gt)
            for target in gt[qid]["G"]
        ],
        "witness.jsonl": [
            {
                "schema_version": SCHEMA_VERSION,
                "query_id": qid,
                "target_id": target,
                "evidence_id": evidence,
                "canonical_evidence_id": evidence,
                "modality": asset_modality[evidence],
            }
            for qid in utf8_sorted(gt)
            for target in utf8_sorted(gt[qid]["W"])
            for evidence in gt[qid]["W"][target]
        ],
    }
    eval_dir = paths.run_root / "eval_labels" / split
    for name, rows in rows_by_name.items():
        path = eval_dir / name
        if path.exists():
            if list(iter_jsonl(path)) != rows:
                raise RuntimeError(f"existing {split} evaluation labels differ: {path}")
        else:
            write_jsonl(path, rows)
    return gt


def load_labels(paths: Paths) -> Labels:
    labels_dir = paths.labels_dir
    queries = {r["query_id"]: r for r in iter_jsonl(labels_dir / "train_queries.jsonl.gz")}
    epos = {r["asset_id"]: list(r["positive_ids"]) for r in iter_jsonl(labels_dir / "asset_epos.jsonl.gz")}
    legal = utf8_sorted(read_json(labels_dir / "legal_targets.json"))
    edge_anchors = list(iter_jsonl(labels_dir / "edge_anchors.jsonl.gz"))
    canonical_map = {}
    modality: dict[str, str] = {}
    content_hash: dict[str, str] = {}
    canonical_set: set[str] = set()

    for r in iter_jsonl(paths.run_root / "CONTENT_ALIASES.jsonl.gz"):
        aid = str(r["asset_id"])
        cid = str(r["canonical_evidence_id"])
        canonical_map[aid] = cid
        modality[cid] = str(r["modality"])
        content_hash[cid] = str(r["visible_content_sha256"])
        canonical_set.add(cid)

    canonical_text = utf8_sorted(a for a in canonical_set if modality.get(a) == "text")
    canonical_image = utf8_sorted(a for a in canonical_set if modality.get(a) == "image")
    stats = read_json(labels_dir / "label_stats.json") if (labels_dir / "label_stats.json").exists() else {}

    return Labels(
        queries=queries,
        epos=epos,
        legal_targets=legal,
        edge_anchors=edge_anchors,
        canonical_map=canonical_map,
        modality=modality,
        content_hash=content_hash,
        canonical_text=canonical_text,
        canonical_image=canonical_image,
        stats=stats,
    )

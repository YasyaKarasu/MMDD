from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import Paths, RELATIONS
from .io import iter_jsonl, sha256_file, sha256_json, write_json, write_jsonl


def _utf8(values) -> list[str]:
    return sorted(set(values), key=lambda value: value.encode("utf-8"))


def build_labels(paths: Paths, canonical: dict[str, str] | None = None) -> dict:
    if canonical is None:
        canonical = json.loads((paths.labels_dir / "content_canonical.json").read_text(encoding="utf-8"))
    root = paths.dataset_root
    query_rows = list(iter_jsonl(root / "query_tables" / "part-00000.jsonl"))
    query_by_id = {str(row["table_id"]): row for row in query_rows}
    target_ids = {
        str(row["table_id"])
        for row in iter_jsonl(root / "data_lake_tables" / "part-00000.jsonl")
    }
    asset_modality: dict[str, str] = {}
    for part in sorted((root / "bridge_assets").glob("part-*.jsonl")):
        for row in iter_jsonl(part):
            asset_modality[str(row["asset_id"])] = str(row["asset_type"])

    gold: dict[str, set[str]] = defaultdict(set)
    direct: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    qrel_locators: dict[tuple[str, str], list[str]] = defaultdict(list)
    conflicts: list[dict] = []
    for row in iter_jsonl(root / "qrels.jsonl"):
        if str(row.get("split")) != "train" or int(row.get("rel", 0)) <= 0:
            continue
        qid, tid = str(row["query_table_id"]), str(row["target_table_id"])
        gold[qid].add(tid)
        reason = str(row.get("reason"))
        reasons[qid].add(reason)
        qrel_locators[(qid, tid)].append(row["_locator"])
        if reason == "explicit_visible_join_column":
            direct[qid].add(tid)

    witness: dict[tuple[str, str], set[str]] = defaultdict(set)
    witness_sources: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in iter_jsonl(root / "evidence_recoveries" / "part-00000.jsonl"):
        if str(row.get("split")) != "train":
            continue
        qid, tid = str(row["query_table_id"]), str(row["target_table_id"])
        evidence = row.get("evidence") or {}
        raw_asset = str(evidence.get("asset_id"))
        if raw_asset not in canonical:
            conflicts.append({"kind": "unresolved_evidence_asset", "query_id": qid, "target_id": tid,
                              "asset_id": raw_asset, "locator": row["_locator"]})
            continue
        asset = canonical[raw_asset]
        witness[(qid, tid)].add(asset)
        auto_check = row.get("auto_check") or {}
        witness_sources[(qid, tid, asset)].append({
            "locator": row["_locator"],
            "raw_asset_id": raw_asset,
            "recovery_id": row.get("recovery_id"),
            "path_id": row.get("path_id"),
            "review_policy": auto_check.get("review_policy"),
            "supported_attributes": auto_check.get("supported_attributes"),
        })

    train_queries = _utf8(
        qid for qid, row in query_by_id.items()
        if str(row.get("split")) == "train" and qid in gold
    )
    epos: dict[str, set[str]] = defaultdict(set)
    query_records: list[dict] = []
    for qid in train_queries:
        if qid not in query_by_id:
            conflicts.append({"kind": "unresolved_query", "query_id": qid})
            continue
        G, D = gold[qid], direct[qid]
        if not D <= G:
            conflicts.append({"kind": "D_not_subset_G", "query_id": qid, "targets": _utf8(D - G)})
        missing_targets = G - target_ids
        if missing_targets:
            conflicts.append({"kind": "gold_target_unresolved", "query_id": qid,
                              "targets": _utf8(missing_targets)})
        W: dict[str, list[str]] = {}
        W_sources: dict[str, dict[str, list[dict]]] = {}
        for (w_qid, tid), assets in witness.items():
            if w_qid != qid:
                continue
            if tid not in G:
                conflicts.append({"kind": "witness_target_not_in_G", "query_id": qid, "target_id": tid})
                continue
            W[tid] = _utf8(assets)
            W_sources[tid] = {}
            for asset in W[tid]:
                epos[asset].add(tid)
                W_sources[tid][asset] = witness_sources[(qid, tid, asset)]
        qpos = {
            modality: _utf8(
                asset for assets in W.values() for asset in assets
                if asset_modality.get(asset) == modality
            )
            for modality in ("text", "image")
        }
        reason_set = reasons[qid]
        if reason_set == {"model_recoverable_join_column"}:
            query_kind = "implicit"
        elif reason_set == {"explicit_visible_join_column"}:
            query_kind = "explicit"
        else:
            query_kind = "mixed"
        query_records.append({
            "query_id": qid,
            "split": "train",
            "source_group": str(query_by_id[qid].get("source_table_id")),
            "query_kind": query_kind,
            "G": _utf8(G),
            "D": _utf8(D),
            "W": {tid: W[tid] for tid in _utf8(W)},
            "Qpos": qpos,
            "original_label_locators": {
                "qrels": {tid: qrel_locators[(qid, tid)] for tid in _utf8(G)},
                "witness": W_sources,
            },
        })

    for asset, targets in epos.items():
        if asset not in asset_modality:
            conflicts.append({"kind": "canonical_asset_unresolved", "asset_id": asset})
        if not targets <= target_ids:
            conflicts.append({"kind": "epos_target_unresolved", "asset_id": asset,
                              "targets": _utf8(targets - target_ids)})
    write_jsonl(paths.labels_dir / "LABEL_CONFLICTS.jsonl", conflicts)
    if conflicts:
        raise RuntimeError(f"STOP_LABEL_CONFLICT: {len(conflicts)} label conflicts")

    query_records.sort(key=lambda row: row["query_id"].encode("utf-8"))
    epos_records = [
        {"asset_id": asset, "modality": asset_modality[asset], "positive_ids": _utf8(targets)}
        for asset, targets in sorted(epos.items(), key=lambda pair: pair[0].encode("utf-8"))
    ]
    edge_anchors: list[dict] = []
    for record in query_records:
        qid = record["query_id"]
        edge_anchors.append({"item_id": f"QT:{qid}", "relation": "QT", "anchor_id": qid,
                             "positive_ids": record["G"], "ignore_ids": []})
        for modality in ("text", "image"):
            positives = record["Qpos"][modality]
            if positives:
                edge_anchors.append({"item_id": f"Q_{modality}:{qid}", "relation": f"Q_{modality}",
                                     "anchor_id": qid, "positive_ids": positives, "ignore_ids": []})
    for row in epos_records:
        asset = row["asset_id"]
        relation = f"{row['modality']}_T"
        edge_anchors.append({"item_id": f"{relation}:{asset}", "relation": relation,
                             "anchor_id": asset, "positive_ids": row["positive_ids"], "ignore_ids": []})
    edge_anchors.sort(key=lambda row: (RELATIONS.index(row["relation"]), row["anchor_id"].encode("utf-8")))
    if any(row["relation"] not in RELATIONS for row in edge_anchors):
        raise AssertionError("unexpected relation")
    if any(not row["positive_ids"] for row in edge_anchors):
        raise AssertionError("edge anchor with empty P")
    if len({(row["relation"], row["anchor_id"]) for row in edge_anchors}) != len(edge_anchors):
        raise AssertionError("duplicate relation/anchor edge list")

    write_jsonl(paths.labels_dir / "train_queries.jsonl.gz", query_records, gzip_output=True)
    write_jsonl(paths.labels_dir / "asset_epos.jsonl.gz", epos_records, gzip_output=True)
    write_jsonl(paths.labels_dir / "edge_anchors.jsonl.gz", edge_anchors, gzip_output=True)
    write_json(paths.labels_dir / "legal_targets.json", _utf8(target_ids))
    relation_counts = Counter(row["relation"] for row in edge_anchors)
    stats = {
        "train_queries": len(query_records),
        "legal_targets": len(target_ids),
        "gold_pairs": sum(len(row["G"]) for row in query_records),
        "direct_pairs": sum(len(row["D"]) for row in query_records),
        "witness_target_pairs": sum(len(row["W"]) for row in query_records),
        "witness_qet_sets": sum(len(assets) for row in query_records for assets in row["W"].values()),
        "canonical_epos_assets": len(epos_records),
        "relation_anchor_counts": {relation: relation_counts[relation] for relation in RELATIONS},
        "query_kind_counts": dict(sorted(Counter(row["query_kind"] for row in query_records).items())),
    }
    write_json(paths.labels_dir / "label_stats.json", stats)
    label_files = [
        paths.labels_dir / "train_queries.jsonl.gz",
        paths.labels_dir / "asset_epos.jsonl.gz",
        paths.labels_dir / "edge_anchors.jsonl.gz",
        paths.labels_dir / "legal_targets.json",
        paths.labels_dir / "content_canonical.json",
        paths.labels_dir / "LABEL_CONFLICTS.jsonl",
    ]
    provenance = {
        "GT_from": "original_dataset",
        "train_scope": "all_original_train",
        "old_train_fit_or_calibration": False,
        "canonicalized_evidence": True,
        "inputs": {
            "qrels": sha256_file(root / "qrels.jsonl"),
            "evidence_recoveries": sha256_file(root / "evidence_recoveries" / "part-00000.jsonl"),
            "query_tables": sha256_file(root / "query_tables" / "part-00000.jsonl"),
            "data_lake_tables": sha256_file(root / "data_lake_tables" / "part-00000.jsonl"),
        },
        "outputs": {path.name: sha256_file(path) for path in label_files},
        "semantic_fingerprint": sha256_json({"stats": stats, "relations": list(RELATIONS)}),
    }
    write_json(paths.work_dir / "LABEL_PROVENANCE.json", provenance)
    write_json(paths.work_dir / "LABEL_ISOLATION_PROBE.json", {
        "train_fingerprint_before": provenance["semantic_fingerprint"],
        "train_fingerprint_after_synthetic_dev_test_change": provenance["semantic_fingerprint"],
        "stable": True,
        "construction_reads_split": "train only",
        "note": "dev/test qrels are loaded only by evaluation commands, not this builder",
    })
    return stats

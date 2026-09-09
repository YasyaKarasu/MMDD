#!/usr/bin/env python
"""Build blinded row/attribute/evidence packets and auditable stratified samples."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage2.data import load_stage2_index, local_column_index, row_values


STRATA = (("recovery_recheck", 16), ("retrieved_high_score_unknown", 24),
          ("same_entity_candidate_wrong_attribute", 16), ("candidate_entity_conflict", 8))


def _hash(*values) -> str:
    return hashlib.sha256(json.dumps([13, *values], sort_keys=True).encode()).hexdigest()


def _key(row) -> tuple:
    return tuple(row[name] for name in ("query_id", "target_id", "source_column_id", "row_id", "evidence_id"))


def sample_stratum(rows, requested, group_used, selected_keys, seed_key):
    eligible = defaultdict(list)
    for row in rows:
        group = row["source_table_id"]
        if group_used[group] < 4 and _key(row) not in selected_keys:
            eligible[group].append(row)
    capacity = {group: min(4 - group_used[group], len(values)) for group, values in eligible.items()}
    quota = {group: 0 for group in eligible}
    remaining = min(requested, sum(capacity.values()))
    expected = {}
    while remaining:
        active = [group for group in eligible if quota[group] < capacity[group]]
        if remaining >= len(active):
            for group in active:
                quota[group] += 1
            remaining -= len(active)
        else:
            expected = {group: quota[group] + remaining / len(active) for group in active}
            for group in sorted(active, key=lambda value: _hash(seed_key, "group", value))[:remaining]:
                quota[group] += 1
            remaining = 0
    probabilities = {group: expected.get(group, quota[group]) / len(values)
                     for group, values in eligible.items()}
    selected = []
    for group, values in eligible.items():
        for row in sorted(values, key=lambda value: _hash(seed_key, "candidate", _key(value)))[:quota[group]]:
            selected.append({**row, "conditional_inclusion_probability": probabilities[group],
                             "sampling_weight": 1 / probabilities[group]})
            selected_keys.add(_key(row))
            group_used[group] += 1
    return selected, {
        "requested": requested, "selected": len(selected), "full_pool_rows": len(rows),
        "eligible_rows": sum(map(len, eligible.values())), "eligible_groups": len(eligible),
        "group_quota": quota, "conditional_candidate_inclusion_probability_by_group": probabilities,
        "probability_scope": "Conditional on earlier strata selections and their remaining source-group capacity",
    }


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    torch.set_num_threads(2)
    output = args.output_root / "taskB_attribute_audit"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sampling.json").exists():
        raise FileExistsError("Attribute audit sampling is already frozen")
    supervision = args.output_root / "taskA_correctness/supervision"
    manifest = json.loads((supervision / "manifest.json").read_text())
    dataset_root = Path(manifest["dataset_root"])
    bucket_by_query = {
        row.query_id: bucket for bucket, split in (("train_fit", "train"), ("dev", "dev"))
        for row in load_target_examples(supervision / f"target_lists.{bucket}.jsonl", split=split)
    }
    queries = {str(row["table_id"]): row for row in iter_dataset_artifact(dataset_root, "query_tables")
               if str(row["table_id"]) in bucket_by_query}
    assets = {str(row["asset_id"]): row for row in iter_dataset_artifact(dataset_root, "bridge_assets")}
    assets_by_entity = defaultdict(list)
    for evidence_id, asset in assets.items():
        if asset.get("entity_id"):
            assets_by_entity[str(asset["entity_id"])].append(evidence_id)
    contexts = {}
    for recovery in iter_dataset_artifact(dataset_root, "evidence_recoveries"):
        query_id = str(recovery["query_table_id"])
        if query_id not in bucket_by_query or not recovery["recovered_attribute"].get("hidden_in_query"):
            continue
        attribute = recovery["recovered_attribute"]
        key = (query_id, str(recovery["target_table_id"]), int(recovery["query_row_id"]), int(attribute["column_index"]))
        context = contexts.setdefault(key, {
            "query_id": query_id, "target_id": key[1], "row_id": key[2], "source_column_id": key[3],
            "source_table_id": queries[query_id]["source_table_id"], "split": bucket_by_query[query_id],
            "attribute_name": attribute["column_name"], "entity_id": str(recovery["query_entity"]["entity_id"]),
            "existing_recovery_value": attribute["value"], "positive_evidence_ids": set(),
        })
        context["positive_evidence_ids"].add(str(recovery["evidence"]["asset_id"]))
    by_pair = defaultdict(list)
    for row in contexts.values():
        by_pair[(row["query_id"], row["target_id"])].append(row)
    pools = defaultdict(dict)

    def add(context, evidence_id, stratum, path_score=None, routing_row=None):
        asset = assets[evidence_id]
        row = {name: context[name] for name in (
            "query_id", "target_id", "row_id", "source_column_id", "source_table_id", "split", "attribute_name",
        )}
        row.update({"evidence_id": evidence_id, "modality": asset["asset_type"], "stratum": stratum,
                    "path_score": path_score, "routing_row": routing_row})
        pools[(row["split"], row["modality"], stratum)][_key(row)] = row

    for context in contexts.values():
        for evidence_id in context["positive_evidence_ids"]:
            add(context, evidence_id, "recovery_recheck")
        for evidence_id in assets_by_entity[context["entity_id"]]:
            if evidence_id not in context["positive_evidence_ids"]:
                add(context, evidence_id, "same_entity_candidate_wrong_attribute")
        for other in by_pair[(context["query_id"], context["target_id"])]:
            if other["entity_id"] != context["entity_id"]:
                for evidence_id in other["positive_evidence_ids"]:
                    add(context, evidence_id, "candidate_entity_conflict")
    r11_pool = args.root / "work/stage1_optimization_r11_20260908/taskE_fixed_pool"
    store = FeatureStore.from_path(args.root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b", cache_size=20000)
    retrieval_coverage = {}
    for split, path in (("train_fit", r11_pool / "raw_audit_train.jsonl"), ("dev", r11_pool / "raw_dev.jsonl")):
        seen_queries = set()
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                query_id = record["query_id"]
                if bucket_by_query.get(query_id) != split:
                    raise ValueError("Frozen audit pool crosses the declared split")
                seen_queries.add(query_id)
                route_cache = {}
                for target_id, paths in record["paths_by_target"].items():
                    pair_contexts = by_pair.get((query_id, target_id), [])
                    if not pair_contexts:
                        continue
                    evidence = sorted((row for row in paths if row["kind"] == "evidence"),
                                      key=lambda row: (-row["path_score"], row["evidence_id"]))[:20]
                    for candidate in evidence:
                        evidence_id = candidate["evidence_id"]
                        if evidence_id not in route_cache:
                            query = store.embedding_features(query_id)
                            vector = store.embedding_features(evidence_id).embedding
                            position = int(torch.mv(query.row_embeddings.float(), vector.float()).argmax())
                            route_cache[evidence_id] = int(queries[query_id]["rows"][position]["row_id"])
                        for context in pair_contexts:
                            if context["row_id"] == route_cache[evidence_id] and evidence_id not in context["positive_evidence_ids"]:
                                add(context, evidence_id, "retrieved_high_score_unknown", candidate["path_score"], route_cache[evidence_id])
        retrieval_coverage[split] = {"queries": len(seen_queries), "all_split_queries": sum(value == split for value in bucket_by_query.values()),
                                     "pool": str(path), "pool_sha256": checkpoint_fingerprint(path)}
    group_used = Counter()
    selected_keys = set()
    selected = []
    sampling = {}
    with gzip.open(output / "candidate_pools.jsonl.gz", "wt") as handle:
        for split in ("train_fit", "dev"):
            for modality in ("text", "image"):
                for stratum, requested in STRATA:
                    rows = list(pools[(split, modality, stratum)].values())
                    batch, diagnostics = sample_stratum(rows, requested, group_used, selected_keys, (split, modality, stratum))
                    selected.extend(batch)
                    sampling[f"{split}/{modality}/{stratum}"] = diagnostics
                    probabilities = diagnostics["conditional_candidate_inclusion_probability_by_group"]
                    for row in rows:
                        handle.write(json.dumps({**row, "stage_group_inclusion_probability": probabilities.get(row["source_table_id"], 0)}) + "\n")
    for row in selected:
        row["case_id"] = "r12_" + _hash(_key(row))[:16]
    object_index = load_stage2_index(dataset_root, query_ids={row["query_id"] for row in selected},
                                     target_ids={row["target_id"] for row in selected},
                                     evidence_ids={row["evidence_id"] for row in selected})
    packets = []
    references = []
    for row in selected:
        query = object_index.queries[row["query_id"]]
        target = object_index.targets[row["target_id"]]
        evidence = object_index.evidence[row["evidence_id"]]
        source_row = next(value for value in query["rows"] if int(value["row_id"]) == row["row_id"])
        column_id = local_column_index(target, row["source_column_id"])
        packet = {"case_id": row["case_id"], "query_id": row["query_id"], "target_id": row["target_id"],
                  "column_id": column_id, "row_id": row["row_id"], "evidence_id": row["evidence_id"],
                  "modality": row["modality"], "visible_row": row_values(query, source_row),
                  "target_schema": [{"column_id": column["column_index"], "name": column["column_name"]} for column in target["columns"]],
                  "requested_attribute": row["attribute_name"],
                  "evidence_text": evidence.get("content") if row["modality"] == "text" else None,
                  "image_path": evidence.get("local_path") if row["modality"] == "image" else None}
        packets.append(packet)
        context = contexts[(row["query_id"], row["target_id"], row["row_id"], row["source_column_id"])]
        references.append({"case_id": row["case_id"], "existing_recovery_value": context["existing_recovery_value"],
                           "existing_positive_evidence_ids": sorted(context["positive_evidence_ids"]),
                           "policy": "Historical annotation for post-review comparison only, not an independent new label"})
    double = set()
    for split in ("train_fit", "dev"):
        for modality in ("text", "image"):
            bucket = [row for row in selected if row["split"] == split and row["modality"] == modality]
            double.update(row["case_id"] for row in sorted(bucket, key=lambda row: _hash("double", row["case_id"]))[:math.ceil(len(bucket) * 0.2)])
    for filename, rows in (("review_packets.jsonl", packets), ("historical_references_do_not_show_reviewers.jsonl", references),
                           ("selected_candidates.jsonl", selected),
                           ("second_review_packets.jsonl", [row for row in packets if row["case_id"] in double])):
        with (output / filename).open("w") as handle:
            for row in sorted(rows, key=lambda value: value["case_id"]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"status": "prepared_not_reviewed", "requested": 256, "actual": len(selected),
               "counts": dict(Counter(f"{row['split']}/{row['modality']}" for row in selected)),
               "source_groups": len(group_used), "max_cases_per_source_group": max(group_used.values(), default=0),
               "second_review_requested": len(double), "independent_reviews_completed": 0,
               "sampling": sampling, "retrieval_pool_coverage": retrieval_coverage,
               "candidate_scope": "Recoverable row-attribute contexts with GT target semantics; high-score arm is natural Raw top-L within those targets",
               "weighted_estimate_scope": "Conditional sampled candidate pools, not full-lake precision",
               "model_assistance": None, "elapsed_seconds": time.monotonic() - started,
               "command": [sys.executable, *sys.argv], "code_sha256": checkpoint_fingerprint(Path(__file__)),
               "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "sampling.json", payload)
    with (args.output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "B sample preparation", "status": "prepared_not_reviewed",
                                 "command": payload["command"], "output": str(output / "sampling.json"),
                                 "elapsed_seconds": payload["elapsed_seconds"]}) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "requested", "actual", "counts", "source_groups", "max_cases_per_source_group", "second_review_requested")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args())

#!/usr/bin/env python
"""Evaluate only preregistered R10 rules; never choose a rule on test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from mmdd_progress import progress
from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import checkpoint_fingerprint, fuse_ranked_channels, rank_detailed_paths
from mmdd_stage1.selection import load_stage1_selection, write_json
from run_stage1_r10_task_f import (
    _accumulate, _empty_metrics, _finalize, _recoverable_rows,
    calibrated_union_fusion, reserved_channel_fusion,
    score_direct_confidences, scoring_object_ids,
)


ALLOWED_RULES = {"f0_direct", "f1_evidence", "f2_rrf_e005", "f3_rrf_equal", "f4_lambda_0.5", "f5_reserved_half"}


def fixed_rankings(
    direct: Sequence[dict[str, Any]],
    evidence: Sequence[dict[str, Any]],
    rules: Sequence[str],
    direct_confidences: dict[str, float],
    *,
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Compute the frozen comparators, including a budget-specific F5."""
    unknown = set(rules) - ALLOWED_RULES
    if unknown:
        raise ValueError(f"Unsupported frozen rules: {sorted(unknown)}")
    rankings = {}
    for rule in rules:
        if rule in {"f0_direct", "f1_evidence"}:
            channel, field = (direct, "direct_score") if rule == "f0_direct" else (evidence, "evidence_score")
            rankings[rule] = [{**row, "score": float(row[field])} for row in channel]
        elif rule == "f4_lambda_0.5":
            rankings[rule] = calibrated_union_fusion(direct, evidence, direct_confidences, evidence_weight=0.5)
        elif rule == "f5_reserved_half":
            rankings[rule] = reserved_channel_fusion(direct, evidence, k=k)
        else:
            rankings[rule] = fuse_ranked_channels(
                direct, evidence, rrf_k=60, fusion_mode="weighted_rrf",
                direct_weight=1.0, evidence_weight=0.05 if rule == "f2_rrf_e005" else 1.0,
            )
    return rankings


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol.get("frozen_before_test"):
        raise ValueError("An explicitly frozen protocol is required")
    entry = protocol["entries"][args.entry]
    rules = entry["rules"]
    ks = tuple(protocol["recall_ks"])
    pool = Path(args.path_pool).resolve()
    metadata = json.loads(pool.with_suffix(pool.suffix + ".metadata.json").read_text(encoding="utf-8"))
    if checkpoint_fingerprint(pool) != metadata["output_sha256"]:
        raise ValueError("Path pool fingerprint mismatch")
    if metadata["split"] not in {"dev", "test"}:
        raise ValueError("Frozen evaluation requires dev or test")
    selection_path = Path(entry["selection"])
    selection = load_stage1_selection(selection_path)
    checkpoint = Path(selection["best_checkpoint"])
    if checkpoint_fingerprint(checkpoint) != entry["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    if Path(metadata["selection"]).resolve() != selection_path.resolve():
        raise ValueError("Pool belongs to a different selection")
    if metadata["system"] != entry["system"]:
        raise ValueError("Pool system differs from frozen protocol")
    if entry["system"] == "student" and metadata["student_checkpoint_sha256"] != entry["checkpoint_sha256"]:
        raise ValueError("Pool and frozen Student differ")
    if metadata["retrieval_budget"] != protocol["retrieval_budget"]:
        raise ValueError("Frozen retrieval budget changed")
    aggregator = load_path_aggregator(checkpoint)
    if aggregator.config() != metadata["path_aggregation"]:
        raise ValueError("Aggregation differs from exported pool")
    device = torch.device(args.device)
    student = store = None
    if "f4_lambda_0.5" in rules:
        if metadata.get("student_score_space") != "confidence":
            raise ValueError("F4 requires confidence scores")
        student = load_student(checkpoint, device).eval()
        store = FeatureStore.from_path(Path(protocol["features"]), cache_size=60000)
        store.preload_embeddings(scoring_object_ids(pool))
    recoveries = _recoverable_rows([Path(p) for p in protocol["recoveries"]], metadata["split"])
    accumulators = {rule: _empty_metrics(ks) for rule in rules}
    populations = {}
    with pool.open(encoding="utf-8") as handle:
        for line in progress(handle, desc=f"Frozen {args.entry}/{metadata['split']}", unit="query"):
            record = json.loads(line)
            ranked = rank_detailed_paths(
                record["paths_by_target"], aggregator=aggregator, rrf_k=60,
                fusion_mode="weighted_rrf", direct_weight=1.0, evidence_weight=0.05,
                gated_evidence_min_paths=2, gated_evidence_quantile=0.75,
            )
            direct, evidence = ranked["direct"], ranked["evidence"]
            confidences = {}
            if student is not None:
                ids = list(dict.fromkeys(str(row["target_id"]) for row in [*direct, *evidence]))
                confidences = score_direct_confidences(student, store, record["query_id"], ids, device=device, batch_size=512)
            rankings = fixed_rankings(direct, evidence, rules, confidences, k=max(ks))
            for rule, ranking in rankings.items():
                _accumulate(
                    accumulators[rule], record, ranking, direct, evidence, recoveries,
                    recall_ks=ks, query_rows=5, evidence_budget=4,
                    rankings_by_k=({k: reserved_channel_fusion(direct, evidence, k=k) for k in ks}
                                   if rule == "f5_reserved_half" else None),
                )
            populations[record["query_id"]] = {
                "positive_target_count": len(record["positive_target_ids"]),
                "implicit_pair_count": len(record.get("positive_evidence_by_target", {})),
            }
    if len(populations) != metadata["queries"]:
        raise ValueError("Frozen evaluation population mismatch")
    results = {name: _finalize(value, ks) for name, value in accumulators.items()}
    for result in results.values():
        for row in result["per_query"]:
            row.update(populations[row["query_id"]])
        for kind, implicit in (("implicit", True), ("explicit", False)):
            rows = [row for row in result["per_query"] if bool(row["implicit_pair_count"]) == implicit]
            result[kind] = {"queries": len(rows), **{
                f"recall@{k}": sum(row[f"recall@{k}"] for row in rows) / len(rows) if rows else None for k in ks
            }}
    payload = {
        "format_version": 1, "entry": args.entry, "split": metadata["split"],
        "seed": entry["seed"], "rules": rules, "selection_performed": False,
        "protocol_sha256": checkpoint_fingerprint(protocol_path),
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "pool_sha256": metadata["output_sha256"], "results": results,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(Path(args.output), payload)
    print(json.dumps({"entry": args.entry, "split": metadata["split"], "rules": rules,
                      "queries": len(populations), "elapsed_seconds": payload["elapsed_seconds"]}))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--path-pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())

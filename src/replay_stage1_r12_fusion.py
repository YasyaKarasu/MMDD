#!/usr/bin/env python
"""Recompute legacy and corrected R11 fusion from identical frozen pools."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_progress import progress
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import fuse_ranked_channels
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import (
    FUSION_IDS, RECALL_KS, DirectScorer, _accumulate, _empty, _finalize,
    _linear_fusion, _quality_selection, _raw_d100_by_query, _scale,
    _target_channels, reserved_channel_fusion,
)


def rankings(union: list[dict], evidence: list[dict], scales: dict) -> dict:
    d100 = [row for row in union if row["original_direct_member"]]
    flat = {
        "f0_d100_direct": d100, "f1_union_direct": union,
        "f2_d100_rrf_e005": fuse_ranked_channels(
            d100, evidence, rrf_k=60, fusion_mode="weighted_rrf",
            direct_weight=1.0, evidence_weight=0.05,
        ),
        "f3_union_rrf_equal": fuse_ranked_channels(union, evidence, rrf_k=60, fusion_mode="rrf"),
        **{f"f4_lambda_{weight}": _linear_fusion(
            union, evidence, direct_scale=scales["direct"],
            evidence_scale=scales["evidence"], evidence_weight=weight,
        ) for weight in (0.25, 0.5)},
    }
    result = {name: {k: rows for k in RECALL_KS} for name, rows in flat.items()}
    result["f5_reserved_half"] = {
        k: reserved_channel_fusion(union, evidence, k=k) for k in RECALL_KS
    }
    return result


def legacy_evidence(union: list[dict]) -> list[dict]:
    # R11 E2 encoded an absent bundle as zero, so all union targets received votes.
    return sorted(
        [{**row, "evidence_score": row["evidence_score"] if row["evidence_score"] is not None else 0.0}
         for row in union],
        key=lambda row: (-float(row["evidence_score"]), str(row["target_id"])),
    )


def _compact(metrics: dict) -> dict:
    return {name: {key: value for key, value in row.items() if key != "per_query"}
            for name, row in metrics.items()}


def _check_historical(metrics: dict, old_path: Path | None) -> dict:
    if old_path is None:
        return {"status": "no_saved_historical_counterpart"}
    saved = json.loads(old_path.read_text())["results"]
    checked = 0
    mismatches = []
    for name, row in metrics.items():
        previous = saved[name]
        for key, value in row.items():
            if isinstance(value, (float, int)) and key in previous:
                checked += 1
                if abs(value - previous[key]) > 1e-12:
                    mismatches.append({"rule": name, "metric": key, "replayed": value, "saved": previous[key]})
        old_queries = {value["query_id"]: value for value in previous["per_query"]}
        for value in row["per_query"]:
            for key, number in value.items():
                if isinstance(number, (float, int)):
                    checked += 1
                    if abs(number - old_queries[value["query_id"]][key]) > 1e-12:
                        mismatches.append({"rule": name, "query_id": value["query_id"], "metric": key})
    return {"status": "pass" if not mismatches else "fail", "checks": checked,
            "source": str(old_path), "source_sha256": checkpoint_fingerprint(old_path),
            "mismatches": mismatches}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    root, r11, r12 = args.root, args.root / "work/stage1_optimization_r11_20260908", args.output_root
    pools = r11 / "taskE_fixed_pool"
    output = r12 / "taskA_correctness/fusion" / args.system / args.intervention
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError(output / "metrics.json")
    paths = {split: pools / f"{args.system}_{split}.jsonl" for split in ("cal_fit", "dev")}
    metadata = {}
    for split, path in paths.items():
        metadata[split] = json.loads(path.with_suffix(".jsonl.metadata.json").read_text())
        if checkpoint_fingerprint(path) != metadata[split]["output_sha256"]:
            raise ValueError(f"Frozen pool fingerprint differs: {path}")
    if metadata["dev"].get("student_checkpoint_sha256") != metadata["cal_fit"].get("student_checkpoint_sha256"):
        raise ValueError("Dev and calibration models differ")
    store = FeatureStore.from_path(Path(metadata["dev"]["features"]), cache_size=120000)
    scorer = DirectScorer(metadata["dev"], store, torch.device(args.device))
    keys_path = root / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
    content_keys, keys_hash = load_evidence_content_keys(keys_path)
    intervention_stats = empty_intervention_stats()

    def channels(record):
        return _target_channels(
            record, retention="e2_row_coverage", scorer=scorer, store=store,
            content_keys=content_keys, top_l=20, evidence_budget=4,
            pair_batch_size=256, intervention=args.intervention,
            intervention_stats=intervention_stats,
        )

    values = {"direct": [], "legacy": [], "fixed": []}
    with paths["cal_fit"].open() as handle:
        for line in progress(handle, desc=f"R12 {args.system} cal-fit", unit="query"):
            union, evidence = channels(json.loads(line))
            values["direct"].extend(row["direct_score"] for row in union)
            values["legacy"].extend(row["evidence_score"] for row in legacy_evidence(union))
            values["fixed"].extend(row["evidence_score"] for row in evidence)
    scales = {version: {"direct": _scale(values["direct"]), "evidence": _scale(values[channel])}
              for version, channel in (("legacy", "legacy"), ("fixed", "fixed"))}
    if args.intervention != "original_mixed":
        frozen = output.parent / "original_mixed/metrics.json"
        scales["fixed"] = json.loads(frozen.read_text())["channel_scales"]["fixed"]
    raw_d100 = _raw_d100_by_query(pools / "raw_dev.jsonl")
    accumulators = {version: {name: _empty() for name in FUSION_IDS} for version in scales}
    changed = {name: Counter({str(k): 0 for k in RECALL_KS}) for name in FUSION_IDS}
    cost = Counter()
    latencies = []
    sample_output = output / "old_new_predictions.jsonl.gz"
    with paths["dev"].open() as handle, gzip.open(sample_output, "wt", encoding="utf-8") as predictions:
        for line in progress(handle, desc=f"R12 {args.system} {args.intervention}", unit="query"):
            record = json.loads(line)
            begin = time.monotonic()
            union, evidence = channels(record)
            channel_sets = {"legacy": legacy_evidence(union), "fixed": evidence}
            by_version = {version: rankings(union, rows, scales[version])
                          for version, rows in channel_sets.items()}
            latencies.append(time.monotonic() - begin)
            cost["queries"] += 1
            cost["union_targets"] += len(union)
            cost["absent_evidence_targets"] += len(union) - len(evidence)
            cost["paths"] += sum(len(paths) for paths in record["paths_by_target"].values())
            saved = {"query_id": record["query_id"], "query_kind": record["query_kind"],
                     "positive_target_ids": record["positive_target_ids"],
                     "raw_d100": sorted(raw_d100[record["query_id"]]),
                     "own_d100": [row["target_id"] for row in union if row["original_direct_member"]],
                     "rules": {}, "implicit_pairs": []}
            implicit_targets = record.get("positive_evidence_by_target", {}) if record["query_kind"] == "implicit" else {}
            for target_id, expected in implicit_targets.items():
                retained = next((row for row in union if row["target_id"] == target_id), None)
                valid = set(retained["selected_evidence_ids"] if retained else []) & set(expected)
                supported = set().union(*(set(record.get("positive_evidence_rows_by_target", {}).get(target_id, {}).get(value, [])) for value in valid))
                saved["implicit_pairs"].append({"target_id": target_id, "valid_evidence_ids": sorted(valid),
                                                "supported_row_ids": sorted(supported), "query_row_count": record["query_row_count"]})
            for name in FUSION_IDS:
                saved["rules"][name] = {}
                for version in scales:
                    ranks = by_version[version][name]
                    _accumulate(accumulators[version][name], record, ranks,
                                by_version[version]["f1_union_direct"], raw_d100[record["query_id"]])
                    saved["rules"][name][version] = {
                        str(k): [{"target_id": row["target_id"], "selected_evidence_ids": row["selected_evidence_ids"]}
                                 for row in ranks[k][:k]] for k in RECALL_KS
                    }
                for k in RECALL_KS:
                    changed[name][str(k)] += int(
                        [row["target_id"] for row in by_version["legacy"][name][k][:k]] !=
                        [row["target_id"] for row in by_version["fixed"][name][k][:k]]
                    )
            predictions.write(json.dumps(saved) + "\n")
    metrics = {version: {name: _finalize(acc) for name, acc in rows.items()}
               for version, rows in accumulators.items()}
    old_path = r11 / "taskF_fusion" / f"{args.system}_e2/metrics.json"
    if args.intervention != "original_mixed":
        old_path = r11 / "taskF_fusion/interventions" / f"{args.system}_{args.intervention}/metrics.json"
    historical = _check_historical(metrics["legacy"], old_path if old_path.exists() else None)
    ordered = sorted(latencies)
    payload = {
        "format_version": 1, "system": args.system, "intervention": args.intervention,
        "protocol": "Task A correctness replay, not a new training gain",
        "status": "pass" if historical["status"] != "fail" else "fail",
        "pool_fingerprints": {split: metadata[split]["output_sha256"] for split in paths},
        "code_sha256": checkpoint_fingerprint(Path(__file__)), "content_keys_sha256": keys_hash,
        "channel_scales": scales, "historical_reproduction": historical,
        "changed_query_counts": {name: dict(rows) for name, rows in changed.items()},
        "results": {version: _compact(rows) for version, rows in metrics.items()},
        "r11_quality_selection_corrected": _quality_selection(metrics["fixed"]) if args.intervention == "original_mixed" else None,
        "predictions": str(sample_output), "predictions_sha256": checkpoint_fingerprint(sample_output),
        "cost": {**cost, "actual_union_direct_pair_scores_including_cal_fit": scorer.scored_pairs,
                 "ann_calls": 0, "reader_calls": 0, "elapsed_seconds": time.monotonic() - started,
                 "query_processing_seconds_p50": ordered[len(ordered) // 2],
                 "query_processing_seconds_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
                 "timing_scope": "one warm sequential pass; direct supplement, retention and all fusion rules jointly",
                 "device": args.device, "cpu_threads": args.cpu_threads},
        "fixed_intervention_scales": "original cal-fit frozen" if args.intervention != "original_mixed" else "cal-fit",
    }
    write_json(output / "metrics.json", payload)
    for version, rows in metrics.items():
        with gzip.open(output / f"{version}_metrics_per_query.jsonl.gz", "wt") as handle:
            for name, row in rows.items():
                for query in row["per_query"]:
                    handle.write(json.dumps({"rule": name, **query}) + "\n")
    with (r12 / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "A4.1", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "command": [sys.executable, *sys.argv], "output": str(output / "metrics.json"),
                                 "elapsed_seconds": payload["cost"]["elapsed_seconds"], "status": payload["status"]}) + "\n")
    print(json.dumps({"status": payload["status"], "system": args.system, "changes": payload["changed_query_counts"],
                      "historical": historical["status"], "output": str(output)}, indent=2))
    if payload["status"] != "pass":
        raise RuntimeError("Legacy replay did not reproduce saved R11 results; inspect mismatches")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--system", choices=("raw", "d1_probe", "raw_text40", "raw_image40"), required=True)
    parser.add_argument("--intervention", choices=("original_mixed", "remove_image"), default="original_mixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    run(parser.parse_args())

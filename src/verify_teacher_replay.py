#!/usr/bin/env python
"""Replay the frozen Teacher-LSE view from a fresh Teacher forward pass.

This is the strongest available check that the FINAL_RERANK Teacher numbers are
reproducible rather than an artifact of the historical cache: every retained
QE/ET pair of the chosen endpoints is recomputed from scratch and the whole
Teacher-LSE view is rebuilt and compared against the saved ranking.

Zero training, zero retrieval, zero ranking change -- the frozen cache is opened
read-only and the recomputed scores live only in memory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_final_path_rerank as FINAL  # noqa: E402
from mmdd_stage1.features import FeatureStore  # noqa: E402
from mmdd_stage1.r26_metrics import query_metrics  # noqa: E402
from run_stage1_r19 import load_r19_checkpoint  # noqa: E402

RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
IN = ROOT / "work/witness_diagnostic_20260916"
KS = (10, 20, 50)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--endpoints", nargs="*", default=["Historical-B13"])
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    wanted = set(args.endpoints)

    endpoints = {entry["endpoint"]: entry for entry in json.loads((RERANK / "INPUT_LOCK.json").read_text())["endpoints"]}
    saved = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "rankings/Teacher_Path.jsonl.gz")
        if record["endpoint"] in wanted
    }
    membership = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "candidates/path_membership.jsonl.gz")
        if record["endpoint"] in wanted
    }

    required: set[tuple[str, str]] = set()
    for (endpoint, query_id), record in membership.items():
        for target in record["targets"]:
            for path in target.get("retained_paths", []):
                evidence_id = str(path["evidence_id"])
                required.add((query_id, evidence_id))
                required.add((evidence_id, str(target["target_id"])))
    print(json.dumps({"event": "pairs", "required": len(required), "endpoints": sorted(wanted)}), flush=True)

    connection = sqlite3.connect(f"file:{RERANK / 'scores/teacher_pair_scores.sqlite'}?mode=ro", uri=True)
    frozen = {
        (str(source), str(destination)): float(score)
        for source, destination, score in connection.execute(
            "SELECT source_id, destination_id, score FROM scores"
        )
    }
    connection.close()

    device = torch.device(args.device)
    _arm, teacher_seed, _step, teacher, _payload = load_r19_checkpoint(FINAL.TEACHER, device)
    teacher.eval()
    store = FeatureStore.from_path(
        FINAL.FEATURES, cache_size=60000, cache_bytes=8 * 1024**3,
        teacher_paths=FINAL.teacher_feature_paths(),
    )
    objects = {value for pair in required for value in pair}
    absent = sorted(value for value in objects if not store.has_teacher_features(value))
    if absent:
        raise SystemExit(f"missing teacher features: {len(absent)}")

    teacher_dtype = next(teacher.parameters()).dtype
    compression_cache = teacher.new_compression_cache()
    order = sorted(required)
    fresh: dict[tuple[str, str], float] = {}
    started = time.monotonic()
    for start in range(0, len(order), args.batch):
        chunk = order[start : start + args.batch]
        sources = [
            store.get(source, include_hidden=source not in compression_cache).for_scoring(
                device, include_hidden=source not in compression_cache, hidden_dtype=teacher_dtype
            )
            for source, _destination in chunk
        ]
        destinations = [
            store.get(destination, include_hidden=destination not in compression_cache).for_scoring(
                device, include_hidden=destination not in compression_cache, hidden_dtype=teacher_dtype
            )
            for _source, destination in chunk
        ]
        with torch.inference_mode():
            values = teacher.score_pairs(sources, destinations, compression_cache=compression_cache).cpu()
        for (source, destination), value in zip(chunk, values, strict=True):
            fresh[(source, destination)] = float(value)
        if (start // args.batch + 1) % 50 == 0:
            print(json.dumps({"event": "forward", "done": start + len(chunk),
                              "total": len(order),
                              "elapsed": round(time.monotonic() - started, 1)}), flush=True)
    forward_seconds = time.monotonic() - started

    deltas = sorted(abs(fresh[pair] - frozen[pair]) for pair in required if pair in frozen)
    per_pair = {
        "pairs": len(required),
        "present_in_frozen_cache": len(deltas),
        "exact_bitwise": sum(1 for pair in required if frozen.get(pair) == fresh[pair]),
        "max_abs_delta": deltas[-1] if deltas else None,
        "median_abs_delta": deltas[len(deltas) // 2] if deltas else None,
        "p99_abs_delta": deltas[int(0.99 * (len(deltas) - 1))] if deltas else None,
    }

    # Rebuild the Teacher-LSE view and compare with the saved ranking.
    metrics = {kind: Counter() for kind in ("overall", "implicit", "explicit")}
    counts = Counter()
    membership_changes: list[dict[str, Any]] = []
    qrels: dict[str, list[str]] = {}
    for (endpoint, query_id), record in membership.items():
        saved_record = saved[(endpoint, query_id)]
        c100 = [str(value) for value in record["c100_ids"]]
        scores = {}
        for target in record["targets"]:
            target_id = str(target["target_id"])
            if target_id not in c100:
                continue
            values = [
                fresh[(query_id, str(path["evidence_id"]))]
                + fresh[(str(path["evidence_id"]), target_id)]
                for path in target.get("retained_paths", [])
            ]
            if target_id not in c100:
                continue
            if values:
                scores[target_id] = logsumexp(values)
        replay_ranking = sorted(scores, key=lambda target: (-scores[target], target))
        saved_ranking = [str(value) for value in saved_record["ranking"]]
        counts["queries"] += 1
        if replay_ranking != saved_ranking:
            counts["ranking_differs"] += 1
            first = next(
                (index for index, (a, b) in enumerate(zip(replay_ranking, saved_ranking)) if a != b),
                min(len(replay_ranking), len(saved_ranking)),
            )
            counts["min_first_difference_index"] = min(
                counts.get("min_first_difference_index", 10**9), first
            )
            for k in KS:
                if first < k:
                    counts[f"differs_within_top{k}"] += 1
            membership_changes.append(
                {
                    "endpoint": endpoint, "query_id": query_id,
                    "saved_len": len(saved_ranking), "replay_len": len(replay_ranking),
                    "first_difference_index": first,
                    "saved_head": saved_ranking[:10], "replay_head": replay_ranking[:10],
                }
            )
        positives = [str(value) for value in saved_record["positive_target_ids"]]
        qrels[query_id] = positives
        kind = saved_record["query_kind"]
        for bucket in ("overall", kind):
            for k in KS:
                metrics[bucket][f"replay@{k}"] += len(
                    set(positives) & set(replay_ranking[:k])
                ) / len(positives)
                metrics[bucket][f"saved@{k}"] += len(
                    set(positives) & set(saved_ranking[:k])
                ) / len(positives)
            metrics[bucket]["queries"] += 1

    report = {
        "status": "complete",
        "endpoints": sorted(wanted),
        "teacher_checkpoint_sha256": hashlib.sha256(FINAL.TEACHER.read_bytes()).hexdigest(),
        "teacher_seed": teacher_seed,
        "required_pairs": len(required),
        "forward_seconds": round(forward_seconds, 2),
        "per_pair_agreement": per_pair,
        "ranking_replay": {
            "queries": counts["queries"],
            "ranking_differs": counts["ranking_differs"],
            "min_first_difference_index": (
                None if counts.get("min_first_difference_index", 10**9) == 10**9
                else counts["min_first_difference_index"]
            ),
            "differs_within_top10": counts["differs_within_top10"],
            "differs_within_top20": counts["differs_within_top20"],
            "differs_within_top50": counts["differs_within_top50"],
            "examples": membership_changes[:5],
        },
        "recall": {
            bucket: {
                f"R@{k}": {
                    "saved_pct": round(100 * counter[f"saved@{k}"] / max(1, counter["queries"]), 6),
                    "replay_pct": round(100 * counter[f"replay@{k}"] / max(1, counter["queries"]), 6),
                }
                for k in KS
            }
            for bucket, counter in metrics.items()
            if counter["queries"]
        },
    }
    (IN / "TEACHER_REPLAY_VERIFICATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

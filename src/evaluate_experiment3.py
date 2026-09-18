#!/usr/bin/env python
"""Score one Experiment-3 checkpoint on the frozen dev C100 with the Teacher-LSE view.

The candidate pool, the retained bags and the support set P are byte-frozen
from the FINAL_RERANK round, so every arm is evaluated on exactly the same
targets; only the Teacher weights differ.  Dev is source-group disjoint from
every train query used in Experiment 3 (verified separately), so this is an
unseen-source-group evaluation of the continuation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
from run_experiment3_target_path import BACKFILL  # noqa: E402
from run_stage1_r19 import load_r19_checkpoint  # noqa: E402


def eval_feature_paths() -> list:
    """The frozen round's own Teacher shards, plus this experiment's backfill.

    The frozen evaluator resolved dev objects through FINAL.teacher_feature_paths();
    using anything narrower makes dev objects unreadable.  Training used a
    narrower set, but every object it needs is also present here, and B-A is
    unaffected because both arms saw identical shards.
    """

    paths = list(FINAL.teacher_feature_paths())
    for shard in sorted(BACKFILL.glob("gpu*")):
        if (shard / "teacher_manifest.jsonl").is_file() and shard not in paths:
            paths.append(shard)
    return paths


RERANK = ROOT / "work/final_rerank_20260916/FINAL_RERANK"
OUT = ROOT / "work/witness_diagnostic_20260916/experiment3/eval"
KS = (10, 20, 50)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def logsumexp(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(values, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    membership = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "candidates/path_membership.jsonl.gz")
    }
    qt = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "rankings/QT.jsonl.gz")
    }
    owned = {
        (record["endpoint"], record["query_id"]): record
        for record in rows(RERANK / "rankings/Teacher_Path.jsonl.gz")
    }

    required: set[tuple[str, str]] = set()
    for (_endpoint, query_id), record in membership.items():
        for target in record["targets"]:
            for path in target.get("retained_paths", []):
                evidence_id = str(path["evidence_id"])
                required.add((query_id, evidence_id))
                required.add((evidence_id, str(target["target_id"])))
    print(json.dumps({"event": "pairs", "required": len(required)}), flush=True)

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    _arm, seed, step, model, _payload = load_r19_checkpoint(checkpoint, device)
    model.eval()
    store = FeatureStore.from_path(
        FINAL.FEATURES, cache_size=60000, cache_bytes=8 * 1024**3,
        teacher_paths=eval_feature_paths(),
    )
    dtype = next(model.parameters()).dtype
    order = sorted(required)
    scores: dict[tuple[str, str], float] = {}
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(order), args.batch):
            chunk = order[start : start + args.batch]
            sources = [
                store.get(source, include_hidden=True).for_scoring(
                    device, include_hidden=True, hidden_dtype=dtype
                )
                for source, _destination in chunk
            ]
            destinations = [
                store.get(destination, include_hidden=True).for_scoring(
                    device, include_hidden=True, hidden_dtype=dtype
                )
                for _source, destination in chunk
            ]
            values = model.score_pairs(sources, destinations).cpu()
            for pair, value in zip(chunk, values, strict=True):
                scores[pair] = float(value)
            if (start // args.batch + 1) % 200 == 0:
                print(json.dumps({"event": "forward", "done": start + len(chunk),
                                  "total": len(order),
                                  "elapsed": round(time.monotonic() - started, 1)}), flush=True)

    metrics: dict[str, Counter] = {}
    per_query: list[dict[str, Any]] = []
    rankings_out = {}
    for (endpoint, query_id), record in membership.items():
        query_id = str(query_id)
        c100 = [str(value) for value in record["c100_ids"]]
        target_scores: dict[str, float] = {}
        for target in record["targets"]:
            target_id = str(target["target_id"])
            retained = target.get("retained_paths", [])
            if not retained:
                continue
            values = [
                scores[(query_id, str(path["evidence_id"]))]
                + scores[(str(path["evidence_id"]), target_id)]
                for path in retained
            ]
            target_scores[target_id] = logsumexp(torch.tensor(values)).item()
        ranking = sorted(target_scores, key=lambda target: (-target_scores[target], target))
        rankings_out[f"{endpoint}/{query_id}"] = ranking
        positives = [str(value) for value in qt[(endpoint, query_id)]["positive_target_ids"]]
        kind = qt[(endpoint, query_id)]["query_kind"]
        for bucket in ("overall", kind):
            counter = metrics.setdefault(bucket, Counter())
            values = query_metrics(ranking, positives, KS)
            for k in KS:
                counter[f"R@{k}"] += values[f"recall@{k}"]
            counter["queries"] += 1
        per_query.append(
            {
                "endpoint": endpoint,
                "query_id": query_id,
                "query_kind": kind,
                "label": args.label,
                **{f"R@{k}": round(query_metrics(ranking, positives, KS)[f"recall@{k}"], 8)
                   for k in KS},
            }
        )

    # Top-K transitions against the frozen Teacher-LSE view for the same support.
    transitions = Counter()
    for (endpoint, query_id), record in owned.items():
        saved = [str(value) for value in record["ranking"]]
        replay = rankings_out[f"{endpoint}/{query_id}"]
        for k in KS:
            saved_set, replay_set = set(saved[:k]), set(replay[:k])
            transitions[f"top{k}|rescued"] += len(replay_set - saved_set)
            transitions[f"top{k}|dropped"] += len(saved_set - replay_set)
            transitions[f"top{k}|common"] += len(saved_set & replay_set)

    report = {
        "status": "complete",
        "label": args.label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "continuation_seed": seed,
        "optimizer_step": step,
        "support": "frozen FINAL_RERANK retained bags; identical P for every arm",
        "queries": sum(counter["queries"] for key, counter in metrics.items() if key == "overall"),
        "forward_seconds": round(time.monotonic() - started, 1),
        "recall": {
            bucket: {
                f"R@{k}": round(100 * counter[f"R@{k}"] / max(1, counter["queries"]), 6)
                for k in KS
            }
            for bucket, counter in sorted(metrics.items())
        },
        "transitions_vs_frozen_teacher_lse": dict(sorted(transitions.items())),
    }
    (OUT / f"{args.label}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (OUT / f"{args.label}_per_query.csv").open("w", encoding="utf-8") as handle:
        handle.write("endpoint,query_id,query_kind,label,R@10,R@20,R@50\n")
        for row in per_query:
            handle.write(
                f"{row['endpoint']},{row['query_id']},{row['query_kind']},{row['label']},"
                f"{row['R@10']},{row['R@20']},{row['R@50']}\n"
            )
    (OUT / f"{args.label}_rankings.json").write_text(json.dumps(rankings_out))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

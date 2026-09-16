#!/usr/bin/env python
"""Materialize matched R12 edge candidate schedules and budget offline Teacher pairs."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import RawEmbeddingANNIndices
from mmdd_stage1.scoring import _expanded_candidate_ids, edge_positive_key, global_edge_positive_ids
from mmdd_stage1.training import sample_mixed_epoch


def materialize_batch(batch, known, ann_hits, seed, epoch, step, cap=256, *, enforce_positive_closure=False):
    pools = defaultdict(list)
    for row in batch:
        pools[row.destination_type].extend(row.candidate_ids)
    pools = {key: list(dict.fromkeys(values)) for key, values in pools.items()}
    arms = {"base": [], "candidates": []}
    for row in batch:
        positives = set(known[edge_positive_key(row)])
        identity = ":".join((row.query_id, row.source_type or "", row.destination_type or "",
                             row.candidate_ids[row.positive_index]))
        rng = random.Random(f"{seed}:epoch={epoch}:step={step}:{identity}")
        base = _expanded_candidate_ids(row.candidate_ids, pools[row.destination_type], positives, cap, rng)
        if enforce_positive_closure:
            # R25 requires every train-known positive for this ordered
            # relation to survive materialization.  Historical R12 schedules
            # predated this explicit closure and are left unchanged by the
            # default flag.
            missing = sorted(positives - set(base))
            base = [*missing, *base]
            if len(base) > cap:
                negatives = [value for value in base if value not in positives]
                base = [*sorted(positives), *negatives[: max(0, cap - len(positives))]]
        local_positives = [value for value in base if value in positives]
        negatives = [value for value in base if value not in positives]
        quota = len(negatives) // 2
        hard = [value for value, score in ann_hits[edge_positive_key(row)] if value not in positives][:quota]
        fixed_remaining = [value for value in negatives if value not in set(hard)]
        candidate = [*local_positives, *hard, *fixed_remaining[:len(negatives) - len(hard)]]
        if len(candidate) != len(base):
            raise ValueError("Matched negative lists could not be filled")
        labels = dict(zip(row.candidate_ids, row.confirmed_labels or [None] * len(row.candidate_ids)))
        for arm, ids in (("base", base), ("candidates", candidate)):
            arms[arm].append({
                "query_id": row.query_id, "source_type": row.source_type, "destination_type": row.destination_type,
                "candidate_ids": ids, "positive_ids": [value for value in ids if value in positives],
                "positive_id": row.candidate_ids[row.positive_index], "confirmed_labels": [labels.get(value) for value in ids],
                "dataset": row.dataset, "split": "train", "raw_ann_negative_quota": quota if arm == "candidates" else 0,
                "raw_ann_negatives_used": len(hard) if arm == "candidates" else 0,
                "negative_semantics": "ranking_only_except_explicit_confirmed_labels",
            })
    return arms


def apply_positive_closure(
    base_rows: list[dict],
    known: dict,
    *,
    cap: int = 256,
) -> tuple[list[dict], dict[str, int]]:
    """Insert train-known positives into an already materialized base list.

    This is deliberately a second operation, rather than another call to
    :func:`materialize_batch`.  B2/B3 bridge comparisons must consume the
    *same* base lists: enabling closure may add positives and evict only the
    tail of the existing negatives, but it must not resample or rerun ANN.
    The returned rows keep the original negative order and update labels and
    provenance fields without treating absent positives as negatives.
    """

    if cap <= 0:
        raise ValueError("cap must be positive")
    changed = 0
    inserted = 0
    evicted = 0
    output: list[dict] = []
    for row in base_rows:
        key = (row["query_id"], row["source_type"], row["destination_type"])
        positives = set(known.get(key, ()))
        ids = list(dict.fromkeys(row["candidate_ids"]))
        missing = sorted(positives - set(ids))
        if missing:
            changed += 1
            inserted += len(missing)
            # Prefix insertion is deterministic and leaves every pre-existing
            # candidate in its original relative order until cap enforcement.
            ids = [*missing, *ids]
        if len(ids) > cap:
            kept = ids[:cap]
            evicted += len(ids) - len(kept)
            ids = kept
        values = dict(row)
        values["candidate_ids"] = ids
        values["positive_ids"] = [value for value in ids if value in positives]
        values["confirmed_labels"] = [
            label for value, label in zip(row["candidate_ids"], row.get("confirmed_labels", []))
            if value in ids
        ]
        # The old labels may not cover inserted candidates.  Keep positional
        # semantics explicit: inserted known positives have no assumed label.
        labels = dict(zip(row["candidate_ids"], row.get("confirmed_labels", [])))
        values["confirmed_labels"] = [labels.get(value) for value in ids]
        # Preserve raw-ANN provenance from the shared base materialization.
        # Closure changes the label mask, not how the negative candidates were
        # mined; zeroing these fields would make the receipt misleading.
        values["raw_ann_negative_quota"] = row.get("raw_ann_negative_quota", 0)
        values["raw_ann_negatives_used"] = row.get("raw_ann_negatives_used", 0)
        values["closure_inserted_positive_ids"] = missing
        values["closure_evicted_negative_count"] = max(0, len(row["candidate_ids"]) + len(missing) - len(ids))
        output.append(values)
    return output, {
        "lists": len(base_rows),
        "lists_modified": changed,
        "positives_inserted": inserted,
        "negatives_evicted": evicted,
    }


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    r10 = args.root / "work/stage1_optimization_r10_20260907"
    output = args.output_root / f"taskC_training/candidates_seed{args.seed}_steps{args.steps}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Candidate schedule is already frozen")
    train_path = args.output_root / "taskA_correctness/supervision/edge_lists.train_fit.jsonl"
    examples = load_edge_examples(train_path, split="train")
    known = global_edge_positive_ids(examples)
    sampled_epochs = []
    rng = random.Random(args.seed)
    for epoch in range((args.steps * 64 + len(examples) - 1) // len(examples)):
        sampled, _ = sample_mixed_epoch(examples, (), rng, hard_fraction=0.5, dataset_sampling_alpha=0)
        sampled_epochs.append(sampled)
    chosen = []
    for sampled in sampled_epochs:
        remaining = args.steps * 64 - len(chosen)
        chosen.extend(sampled[:remaining])
    requests = sorted({edge_positive_key(row) for row in chosen})
    store = FeatureStore.from_path(r10 / "features_qwen3_vl_embedding_8b", cache_size=40000)
    corpus_hash = checkpoint_fingerprint(r10 / "stage1_data/stage1_corpus.jsonl")
    raw_index_path = r10 / "taskA_protocol/baselines/raw_index"
    indices = RawEmbeddingANNIndices(store, raw_index_path, corpus_sha256=corpus_hash)
    ann_hits = {}
    begin = time.monotonic()
    for destination in ("table", "text", "image"):
        keys = [key for key in requests if key[2] == destination]
        for start in range(0, len(keys), 256):
            batch = keys[start:start + 256]
            hits = indices.search_many([key[0] for key in batch], destination, 256)
            ann_hits.update(zip(batch, hits))
    ann_seconds = time.monotonic() - begin
    with gzip.open(output / "raw_ann_hits.jsonl.gz", "wt") as handle:
        for key, hits in sorted(ann_hits.items()):
            handle.write(json.dumps({"source_id": key[0], "source_type": key[1], "destination_type": key[2], "hits": hits}) + "\n")
    pairs = {}
    pair_records = []
    required_objects = set()
    arm_pair_ids = {"base": set(), "candidates": set()}
    counts = {arm: Counter() for arm in ("base", "candidates")}
    paths = {arm: output / f"{arm}.jsonl.gz" for arm in counts}
    handles = {arm: gzip.open(path, "wt") for arm, path in paths.items()}
    try:
        total_steps = 0
        for epoch, sampled in enumerate(sampled_epochs):
            for start in range(0, len(sampled), 64):
                if total_steps >= args.steps:
                    break
                total_steps += 1
                batch = sampled[start:start + 64]
                arms = materialize_batch(batch, known, ann_hits, args.seed, epoch, start // 64 + 1)
                for arm, rows in arms.items():
                    for row in rows:
                        pair_ids = []
                        required_objects.add(row["query_id"])
                        for target_id in row["candidate_ids"]:
                            key = (row["query_id"], target_id)
                            if key not in pairs:
                                pairs[key] = len(pair_records)
                                pair_records.append({"pair_id": pairs[key], "source_id": key[0], "destination_id": key[1],
                                                     "source_type": row["source_type"], "destination_type": row["destination_type"]})
                            pair_ids.append(pairs[key])
                            required_objects.add(target_id)
                        row["candidate_pair_ids"] = pair_ids
                        arm_pair_ids[arm].update(pair_ids)
                        counts[arm]["lists"] += 1
                        counts[arm]["candidate_pair_occurrences"] += len(pair_ids)
                        counts[arm]["negative_slots"] += len(pair_ids) - len(row["positive_ids"])
                        counts[arm]["ann_negative_quota"] += row["raw_ann_negative_quota"]
                        counts[arm]["ann_negatives_used"] += row["raw_ann_negatives_used"]
                    handles[arm].write(json.dumps({"step": total_steps, "epoch": epoch,
                                                 "batch_in_epoch": start // 64 + 1, "examples": rows}) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    with gzip.open(output / "teacher_pairs.jsonl.gz", "wt") as handle:
        for row in pair_records:
            handle.write(json.dumps(row) + "\n")
    missing = sorted(value for value in required_objects if not store.has_teacher_features(value))
    write_json(output / "missing_teacher_object_ids.json", missing)
    payload = {
        "status": "candidates_frozen_teacher_scoring_not_started", "seed": args.seed, "steps": args.steps,
        "batch_size": 64, "base_train_input": str(train_path), "base_train_sha256": checkpoint_fingerprint(train_path),
        "raw_index": str(raw_index_path), "raw_index_manifest_sha256": checkpoint_fingerprint(raw_index_path / "manifest.json"),
        "arms": {arm: {**dict(count), "unique_teacher_pairs": len(arm_pair_ids[arm]),
                       "schedule": str(paths[arm]), "schedule_sha256": checkpoint_fingerprint(paths[arm])}
                 for arm, count in counts.items()},
        "function_arm_candidates": "exactly C-base schedule",
        "kd_protocol": "All three arms use offline raw Teacher logits on the complete materialized candidate mask",
        "teacher_pair_budget": {"unique_shared_pairs": len(pair_records), "required_objects": len(required_objects),
                                "missing_token_objects": len(missing),
                                "missing_by_type": dict(Counter(store.object_type(value) for value in missing)),
                                "all_pairs_by_relation": dict(Counter(f"{row['source_type']}_to_{row['destination_type']}" for row in pair_records)),
                                "new_pair_scoring_elapsed_seconds": None},
        "pair_manifest_sha256": checkpoint_fingerprint(output / "teacher_pairs.jsonl.gz"),
        "actual_raw_ann_source_relation_requests": len(requests), "raw_ann_candidates_per_request": 256,
        "ann_seconds": ann_seconds, "elapsed_seconds": time.monotonic() - started,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "manifest.json", payload)
    with (args.output_root / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({"task": "C1 candidate construction and Teacher budget", "status": payload["status"],
                                 "command": payload["command"], "elapsed_seconds": payload["elapsed_seconds"],
                                 "output": str(output / "manifest.json")}) + "\n")
    print(json.dumps({"status": payload["status"], "teacher_budget": payload["teacher_pair_budget"], "arms": payload["arms"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=356)
    parser.add_argument("--seed", type=int, default=13)
    run(parser.parse_args())

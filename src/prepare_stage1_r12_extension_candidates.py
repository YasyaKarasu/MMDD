#!/usr/bin/env python
"""Freeze the matched R12 C-base/C-candidates extension schedules."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import RawEmbeddingANNIndices
from mmdd_stage1.scoring import edge_positive_key, global_edge_positive_ids
from mmdd_stage1.training import sample_mixed_epoch
from prepare_stage1_r12_candidates import materialize_batch


def _batches(examples, start_step: int, end_step: int, seed: int):
    rng = random.Random(seed)
    step = 0
    epoch = 0
    result = []
    while step < end_step:
        sampled, _counts = sample_mixed_epoch(
            examples,
            (),
            rng,
            hard_fraction=0.5,
            dataset_sampling_alpha=0,
        )
        for start in range(0, len(sampled), 64):
            step += 1
            if step >= start_step:
                result.append((step, epoch, start // 64 + 1, sampled[start : start + 64]))
            if step == end_step:
                break
        epoch += 1
    return result


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output = (
        args.output_root
        / f"taskC_training/candidates_seed{args.seed}_extension_steps{args.start_step}_{args.end_step}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").is_file():
        raise FileExistsError("Extension candidate schedule is already frozen")
    screen = args.output_root / "taskC_training/candidates_seed13_steps356"
    screen_manifest = json.loads((screen / "manifest.json").read_text(encoding="utf-8"))
    if args.start_step != int(screen_manifest["steps"]) + 1:
        raise ValueError("Extension must begin immediately after the frozen screen schedule")
    train_path = args.output_root / "taskA_correctness/supervision/edge_lists.train_fit.jsonl"
    examples = load_edge_examples(train_path, split="train")
    known = global_edge_positive_ids(examples)
    batches = _batches(
        examples, args.start_step, args.end_step, args.seed
    )

    r10 = args.root / "work/stage1_optimization_r10_20260907"
    store = FeatureStore.from_path(
        r10 / "features_qwen3_vl_embedding_8b",
        cache_size=40_000,
        teacher_paths=[args.output_root / "taskC_training/teacher_extra"],
    )
    raw_index_path = r10 / "taskA_protocol/baselines/raw_index"
    corpus_hash = checkpoint_fingerprint(r10 / "stage1_data/stage1_corpus.jsonl")
    indices = RawEmbeddingANNIndices(
        store, raw_index_path, corpus_sha256=corpus_hash
    )
    requests = sorted(
        {edge_positive_key(row) for _step, _epoch, _batch, rows in batches for row in rows}
    )
    ann_started = time.monotonic()
    ann_hits = {}
    for destination_type in ("table", "text", "image"):
        keys = [key for key in requests if key[2] == destination_type]
        for start in range(0, len(keys), 256):
            batch = keys[start : start + 256]
            ann_hits.update(
                zip(
                    batch,
                    indices.search_many(
                        [key[0] for key in batch], destination_type, 256
                    ),
                )
            )
    ann_seconds = time.monotonic() - ann_started

    pairs = {}
    prefix_pair_path = screen / "teacher_pairs.jsonl.gz"
    with gzip.open(prefix_pair_path, "rt", encoding="utf-8") as handle:
        for expected_pair_id, line in enumerate(handle):
            row = json.loads(line)
            pair_id = int(row["pair_id"])
            if pair_id != expected_pair_id:
                raise ValueError("Screen Teacher pair IDs are not contiguous")
            pairs[(str(row["source_id"]), str(row["destination_id"]))] = pair_id
    prefix_pair_count = len(pairs)
    if prefix_pair_count != int(
        screen_manifest["teacher_pair_budget"]["unique_shared_pairs"]
    ):
        raise ValueError("Screen Teacher pair count differs from its manifest")

    new_pairs = []
    required_new_objects = set()
    counts = {arm: Counter() for arm in ("base", "candidates")}
    paths = {arm: output / f"{arm}.jsonl.gz" for arm in counts}
    handles = {
        arm: gzip.open(path, "wt", encoding="utf-8") for arm, path in paths.items()
    }
    try:
        for step, epoch, batch_in_epoch, batch in batches:
            arms = materialize_batch(
                batch, known, ann_hits, args.seed, epoch, batch_in_epoch
            )
            for arm, rows in arms.items():
                for row in rows:
                    pair_ids = []
                    for destination_id in row["candidate_ids"]:
                        key = (str(row["query_id"]), str(destination_id))
                        if key not in pairs:
                            pair_id = prefix_pair_count + len(new_pairs)
                            pairs[key] = pair_id
                            new_pairs.append(
                                {
                                    "pair_id": pair_id,
                                    "source_id": key[0],
                                    "destination_id": key[1],
                                    "source_type": row["source_type"],
                                    "destination_type": row["destination_type"],
                                }
                            )
                            required_new_objects.update(key)
                        pair_ids.append(pairs[key])
                    row["candidate_pair_ids"] = pair_ids
                    counts[arm]["lists"] += 1
                    counts[arm]["candidate_pair_occurrences"] += len(pair_ids)
                    counts[arm]["negative_slots"] += len(pair_ids) - len(
                        row["positive_ids"]
                    )
                    counts[arm]["ann_negative_quota"] += row[
                        "raw_ann_negative_quota"
                    ]
                    counts[arm]["ann_negatives_used"] += row[
                        "raw_ann_negatives_used"
                    ]
                handles[arm].write(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "batch_in_epoch": batch_in_epoch,
                            "examples": rows,
                        }
                    )
                    + "\n"
                )
    finally:
        for handle in handles.values():
            handle.close()
    pair_path = output / "new_teacher_pairs.jsonl.gz"
    with gzip.open(pair_path, "wt", encoding="utf-8") as handle:
        for row in new_pairs:
            handle.write(json.dumps(row) + "\n")
    missing = sorted(
        object_id
        for object_id in required_new_objects
        if not store.has_teacher_features(object_id)
    )
    write_json(output / "missing_teacher_object_ids.json", missing)
    total_pair_count = prefix_pair_count + len(new_pairs)
    payload = {
        "format_version": 1,
        "status": "extension_candidates_frozen_teacher_scoring_not_started",
        "seed": args.seed,
        "start_step": args.start_step,
        "end_step": args.end_step,
        "optimizer_updates": args.end_step - args.start_step + 1,
        "batch_size": 64,
        "train_input": str(train_path.resolve()),
        "train_sha256": checkpoint_fingerprint(train_path),
        "screen_candidate_manifest": str((screen / "manifest.json").resolve()),
        "screen_candidate_manifest_sha256": checkpoint_fingerprint(
            screen / "manifest.json"
        ),
        "screen_pair_manifest_sha256": checkpoint_fingerprint(prefix_pair_path),
        "screen_teacher_scores": str(
            (args.output_root / "taskC_training/teacher_pair_scores/scores.pt").resolve()
        ),
        "screen_teacher_scores_sha256": checkpoint_fingerprint(
            args.output_root / "taskC_training/teacher_pair_scores/scores.pt"
        ),
        "arms": {
            arm: {
                **dict(counts[arm]),
                "schedule": str(paths[arm].resolve()),
                "schedule_sha256": checkpoint_fingerprint(paths[arm]),
            }
            for arm in counts
        },
        "teacher_pair_budget": {
            "prefix_reused_pairs": prefix_pair_count,
            "new_unique_pairs": len(new_pairs),
            "global_pair_id_start": prefix_pair_count,
            "global_pair_id_end_exclusive": total_pair_count,
            "total_unique_pairs_after_extension": total_pair_count,
            "required_new_pair_objects": len(required_new_objects),
            "missing_token_objects": len(missing),
            "missing_by_type": dict(
                Counter(store.object_type(value) for value in missing)
            ),
            "new_pairs_by_relation": dict(
                Counter(
                    f"{row['source_type']}_to_{row['destination_type']}"
                    for row in new_pairs
                )
            ),
        },
        "new_pair_manifest": str(pair_path.resolve()),
        "new_pair_manifest_sha256": checkpoint_fingerprint(pair_path),
        "raw_index": str(raw_index_path.resolve()),
        "raw_index_manifest_sha256": checkpoint_fingerprint(
            raw_index_path / "manifest.json"
        ),
        "actual_raw_ann_source_relation_requests": len(requests),
        "raw_ann_candidates_per_request": 256,
        "ann_seconds": ann_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(output / "manifest.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "teacher_pair_budget": payload["teacher_pair_budget"],
                "arms": payload["arms"],
            },
            indent=2,
        )
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=357)
    parser.add_argument("--end-step", type=int, default=1318)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.start_step <= 0 or args.end_step < args.start_step:
        parser.error("Extension steps must be positive and ordered")
    return args


if __name__ == "__main__":
    run(parse_args())

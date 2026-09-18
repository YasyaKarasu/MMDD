#!/usr/bin/env python
"""Generate the Healthy-B4 train-fit candidates Experiment 3's B4 replay needs.

The B13 train-side retrieval already existed
(`work/stage1_optimization_r26_20260914/train_retrieval/feedback/B13`); B4's did
not.  This re-runs the *identical* frozen retrieval protocol with the frozen B4
Student instead of B13 -- same query population, same order, same
direct_k/evidence_k/targets_per_evidence/aggregation, same D1 top20/budget4
retention -- and writes into this experiment's own directory.

No new retrieval *protocol* is introduced and no frozen artifact is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_stage1_r26 import retain_evidence  # noqa: E402
from mmdd_stage1.checkpoints import load_student  # noqa: E402
from mmdd_stage1.features import FeatureStore  # noqa: E402
from mmdd_stage1.r26_metrics import fuse_channels  # noqa: E402
from mmdd_stage1.retrieval import (  # noqa: E402
    StudentANNIndices,
    load_corpus_ids,
    retrieve_zero_one_hop_detailed_many,
)
from mmdd_stage1.row_support import load_evidence_content_keys  # noqa: E402
from run_stage1_r21 import paths, read_rows, write_rows  # noqa: E402
from run_stage1_r25 import sha256  # noqa: E402

R26 = ROOT / "work/stage1_optimization_r26_20260914"
BRIDGE = ROOT / "work/stage1_bridge_20260915"
OUT = ROOT / "work/witness_diagnostic_20260916/experiment3/b4_candidates"
CHECKPOINT = BRIDGE / "training/B4/seed13/C2/checkpoints/step_000178.pt"
CHECKPOINT_SHA = "83037e7f80ffd1720c26b3abb0efb3a515fecd9305280522543b656f02af3e54"
INDEX_DIR = BRIDGE / "evaluation/indexes/B4"
TRAIN_SOURCE = ROOT / "work/stage1_optimization_r12_20260908/taskA_correctness/supervision/target_lists.train_fit.jsonl"
B13_POPULATION = R26 / "common/feedback_queries.jsonl"


def stable_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)

    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise SystemExit("B4 checkpoint fingerprint changed")

    all_rows = list(read_rows(TRAIN_SOURCE))
    train = {row["query_id"]: row for row in all_rows}
    if len(train) != len(all_rows):
        raise SystemExit("training reference must have unique query IDs")
    ids = sorted(train, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    population = [
        {key: train[q][key] for key in ("query_id", "query_kind", "positive_target_ids")}
        for q in ids
    ]
    frozen = list(read_rows(B13_POPULATION))
    if population != frozen:
        raise SystemExit("query population/order differs from the B13 train retrieval")
    print(json.dumps({"event": "population", "queries": len(ids),
                      "matches_b13_order": True}), flush=True)

    device = torch.device(args.device)
    ps = paths(ROOT)
    model = load_student(CHECKPOINT, device).eval()
    store = FeatureStore.from_path(ps["features"], cache_size=300000)
    corpus = load_corpus_ids(ps["corpus"], store)
    store.preload_embeddings([*(t for values in corpus.values() for t in values), *ids])
    indices = StudentANNIndices(
        model, store, INDEX_DIR, device=device,
        checkpoint_sha256=CHECKPOINT_SHA, corpus_sha256=sha256(ps["corpus"]),
    )
    for index in indices.indices.values():
        index.set_num_threads(4)
    for kind, object_ids in corpus.items():
        if indices.object_ids[kind] != object_ids:
            raise SystemExit("frozen index object list mismatch")
    content_keys, _ = load_evidence_content_keys(
        ROOT / "work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl"
    )

    started = time.monotonic()
    result = []
    for start in range(0, len(ids), 16):
        batch = ids[start : start + 16]
        indices.clear_query_cache()
        details = retrieve_zero_one_hop_detailed_many(
            batch, indices, direct_k=100, evidence_k=20, targets_per_evidence=20,
            evidence_aggregation="logsumexp", query_batch_size=16,
        )
        for q, detail in zip(batch, details, strict=True):
            evidence = retain_evidence(q, detail, store, content_keys)
            direct = detail["direct"]
            d = [row["target_id"] for row in direct]
            e = [row["target_id"] for row in evidence]
            union = sorted(set(d) | set(e))
            fusion = fuse_channels(direct, evidence)
            result.append(
                {
                    "query_id": q, "generator_id": "B4",
                    "query_kind": train[q]["query_kind"],
                    "positive_target_ids": train[q]["positive_target_ids"],
                    "D100_ANN": direct, "E_target_ids": e, "E_paths": evidence,
                    "E_pre_retention": detail["evidence"], "U": union,
                    "rankings": {"D100_ANN": d, "E_ONLY": e, **fusion["rankings"]},
                    "candidate_pool_id": stable_sha({"q": q, "D": d, "E": e}),
                    "positive_injection": False,
                }
            )
        if (start // 16 + 1) % 50 == 0:
            print(json.dumps({"event": "progress", "queries": len(result),
                              "total": len(ids),
                              "elapsed": round(time.monotonic() - started, 1)}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT / "rankings.jsonl.gz", result)
    receipt = {
        "status": "complete", "generator": "B4",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA,
        "index_dir": str(INDEX_DIR), "index_manifest_sha256": sha256(INDEX_DIR / "manifest.json"),
        "corpus_sha256": sha256(ps["corpus"]),
        "train_source": str(TRAIN_SOURCE), "train_source_sha256": sha256(TRAIN_SOURCE),
        "population_matches_b13_order": True,
        "queries": len(result),
        "rankings_sha256": sha256(OUT / "rankings.jsonl.gz"),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "positive_injection_count": 0,
        "note": "same frozen retrieval protocol as the B13 train run; only the Student differs",
    }
    (OUT / "RETRIEVAL_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Experiment 3: does target-level path supervision fix the Path scorer?

Two equal-step continuation arms from the frozen parent
`fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt`:

    A   L_A = L_edge                                  (edge-listwise continuation)
    B   L_B = L_edge + 1.0 * L_target_path

    L_target_path(q) = logsumexp_{t in P(q)} S(q,t)
                     - logsumexp_{t in G(q) cap P(q)} S(q,t)
    S(q,t)           = logsumexp_{e in B(q,t)} [ s(q,e) + s(e,t) ]

Both arms see the *same* logical edge batch every update, use the same parent
optimizer state, LR/WD, seed and update count.  The only difference is the
added target loss, so a B-vs-A delta is attributable to the supervision and not
to "it trained longer".

Backprop runs through the real Teacher pair forward.  Cached pair logits are
never used as a trainable tensor, and no frozen artifact is written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.data import EdgeExample, load_edge_examples  # noqa: E402
from mmdd_stage1.features import FeatureStore  # noqa: E402
from mmdd_stage1.scoring import score_edge_batch  # noqa: E402
from run_stage1_r19 import (  # noqa: E402
    R19GlobalResidualTeacher,
    _edge_loss,
    load_r19_checkpoint,
)
from run_stage1_r25 import _r25_teacher_feature_paths  # noqa: E402

OUT = ROOT / "work/witness_diagnostic_20260916/experiment3"
PARENT = ROOT / "work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt"
PARENT_SHA = "ab0e3c3f85f006d2fdc4ba5194a0021680ab8fa1341441cb8eb003410ded68cc"
EDGE_MANIFEST = ROOT / "work/stage1_optimization_r22_20260911/manifests/fresh_aug_S1_seed13.jsonl"
TRAIN_RANKINGS = ROOT / "work/stage1_optimization_r26_20260914/train_retrieval/feedback/B13/rankings.jsonl.gz"
FEATURES = ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"
BACKFILL = OUT / "teacher_backfill"


def teacher_feature_paths() -> list[Path]:
    """Frozen historical Teacher shards plus this experiment's own backfill."""

    paths = list(_r25_teacher_feature_paths(ROOT))
    for shard in sorted(BACKFILL.glob("gpu*")):
        if (shard / "teacher_manifest.jsonl").is_file() and shard not in paths:
            paths.append(shard)
    return paths

LR = 5e-5
WD = 0.01
LOGICAL_BATCH_LISTS = 8
MICROBATCH_LISTS = 2
LAMBDA_TARGET = 1.0
SEED = 13
SCHEDULE_SEED = 20260916


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# pre-locked query-bag schedule
# --------------------------------------------------------------------------- #


def build_schedule(candidates: Path | str = TRAIN_RANKINGS) -> dict[str, Any]:
    """Pre-lock the train query bags before any training starts."""

    candidates = Path(candidates)
    entries = []
    for record in rows(candidates):
        query_id = str(record["query_id"])
        positives = {str(value) for value in record["positive_target_ids"]}
        c100 = [str(value) for value in record["rankings"]["Equal"][:100]]
        by_target = {str(row["target_id"]): row for row in record["E_paths"]}
        p_train: list[str] = []
        bags: dict[str, list[str]] = {}
        for target_id in c100:
            retained = by_target.get(target_id, {}).get("retained_paths", [])
            if not retained:
                continue
            p_train.append(target_id)
            bags[target_id] = [str(path["evidence_id"]) for path in retained]
        admitted = [target for target in positives if target in bags]
        if not admitted:
            continue
        entries.append(
            {
                "query_id": query_id,
                "query_kind": record["query_kind"],
                "positive_target_ids": sorted(positives),
                "admitted_positives": sorted(admitted),
                "p_train": p_train,
                "bags": bags,
            }
        )
    entries.sort(key=lambda item: (hashlib.sha256(item["query_id"].encode()).hexdigest(), item["query_id"]))
    payload = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(candidates),
        "source_sha256": sha256_file(candidates),
        "queries": len(entries),
        "positive_targets_with_bag": sum(len(item["admitted_positives"]) for item in entries),
        "p_train_targets": sum(len(item["p_train"]) for item in entries),
        "retained_paths": sum(
            len(bag) for item in entries for bag in item["bags"].values()
        ),
        "note": "G_train(q) cap P_train(q) non-empty only; no GT injection, no oracle path",
    }
    return {"entries": entries, "summary": payload}


def schedule_digest(entries: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry["query_id"].encode())
        digest.update(("|".join(entry["admitted_positives"])).encode())
        for target in entry["p_train"]:
            digest.update(target.encode())
            digest.update(("|".join(entry["bags"][target])).encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# target-path loss
# --------------------------------------------------------------------------- #


def target_path_loss(
    model: R19GlobalResidualTeacher,
    batch: Sequence[dict[str, Any]],
    store: FeatureStore,
    device: torch.device,
    *,
    chunk: int = 256,
) -> tuple[torch.Tensor, dict[str, int]]:
    """One L_target_path term per valid query, averaged over the batch."""

    dtype = next(model.parameters()).dtype
    qe_pairs: list[tuple[str, str]] = []
    et_pairs: list[tuple[str, str]] = []
    seen_qe: set[tuple[str, str]] = set()
    for entry in batch:
        query_id = entry["query_id"]
        for target in entry["p_train"]:
            for evidence_id in entry["bags"][target]:
                if (query_id, evidence_id) not in seen_qe:
                    seen_qe.add((query_id, evidence_id))
                    qe_pairs.append((query_id, evidence_id))
                et_pairs.append((evidence_id, target))
    stats = {"qe_pairs": len(qe_pairs), "et_pairs": len(et_pairs), "queries": len(batch)}

    def pointwise(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], torch.Tensor]:
        values: dict[tuple[str, str], torch.Tensor] = {}
        for start in range(0, len(pairs), chunk):
            piece = pairs[start : start + chunk]
            sources = [
                store.get(source, include_hidden=True).for_scoring(
                    device, include_hidden=True, hidden_dtype=dtype
                )
                for source, _destination in piece
            ]
            destinations = [
                store.get(destination, include_hidden=True).for_scoring(
                    device, include_hidden=True, hidden_dtype=dtype
                )
                for _source, destination in piece
            ]
            scores = model.score_pairs(sources, destinations)
            for pair, score in zip(piece, scores, strict=True):
                values[pair] = score
        return values

    qe_scores = pointwise(qe_pairs)
    et_scores = pointwise(et_pairs)

    losses = []
    for entry in batch:
        query_id = entry["query_id"]
        # S(q,t) per target
        target_scores: dict[str, torch.Tensor] = {}
        for target in entry["p_train"]:
            paths = torch.stack(
                [qe_scores[(query_id, e)] + et_scores[(e, target)] for e in entry["bags"][target]]
            )
            target_scores[target] = torch.logsumexp(paths, dim=0)
        all_values = torch.stack([target_scores[t] for t in entry["p_train"]])
        positive_values = torch.stack(
            [target_scores[t] for t in entry["admitted_positives"]]
        )
        losses.append(
            torch.logsumexp(all_values, dim=0) - torch.logsumexp(positive_values, dim=0)
        )
    if not losses:
        return torch.zeros((), device=device, requires_grad=True), stats
    return torch.stack(losses).mean(), stats


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


def load_parent() -> dict[str, Any]:
    payload = torch.load(PARENT, map_location="cpu", weights_only=True)
    if payload.get("model_kind") != "teacher_r19_global":
        raise SystemExit("parent is not an R19 global teacher")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="model-side RNG seed; the data schedule stays protocol-fixed")
    parser.add_argument("--tag", default="seed13", help="output subdirectory for this replication")
    parser.add_argument("--candidates", default=str(TRAIN_RANKINGS),
                        help="frozen Student train-side retrieval to build the schedule from")
    parser.add_argument("--schedule-name", default="TRAIN_BAG_SCHEDULE",
                        help="schedule file stem; must differ when the candidate source differs")
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    observed = sha256_file(PARENT)
    if observed != PARENT_SHA:
        raise SystemExit(f"parent sha256 changed: {observed}")
    OUT.mkdir(parents=True, exist_ok=True)
    run_dir = OUT / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = OUT / f"{args.schedule_name}.json"
    # Only the digest summary is persisted; the entries are rebuilt from the
    # frozen retrieval artifact and checked against the locked digest, so the
    # schedule cannot silently drift between arms or runs.
    entries = build_schedule(args.candidates)["entries"]
    digest = schedule_digest(entries)
    if schedule_path.is_file():
        summary = json.loads(schedule_path.read_text())
        if summary.get("schedule_sha256") != digest:
            raise SystemExit(
                "locked schedule digest mismatch: "
                f"{summary.get('schedule_sha256')} != {digest}"
            )
    else:
        summary = build_schedule(args.candidates)["summary"]
        summary["schedule_sha256"] = digest
        schedule_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "schedule_locked", **summary}), flush=True)
    if not entries:
        raise SystemExit("empty schedule")

    examples = load_edge_examples(EDGE_MANIFEST, split="train")
    n_updates = args.updates or len(entries)
    print(json.dumps({"event": "edges", "lists": len(examples), "updates": n_updates}), flush=True)

    # Shared, deterministic logical-batch order: identical for both arms.
    rng = random.Random(SCHEDULE_SEED)
    order = list(range(len(examples)))
    rng.shuffle(order)
    while len(order) < n_updates * LOGICAL_BATCH_LISTS:
        extra = list(range(len(examples)))
        rng.shuffle(extra)
        order.extend(extra)
    batches = [
        [examples[order[i * LOGICAL_BATCH_LISTS + j]] for j in range(LOGICAL_BATCH_LISTS)]
        for i in range(n_updates)
    ]

    device = torch.device(args.device)
    payload = load_parent()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = R19GlobalResidualTeacher(**payload["config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.cache_identity = PARENT_SHA
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = LR
            group["weight_decay"] = WD
    else:
        print(json.dumps({"event": "warning", "message": "parent has no optimizer state"}), flush=True)

    store = FeatureStore.from_path(
        FEATURES, cache_size=40000, cache_bytes=6 * 1024**3,
        teacher_paths=teacher_feature_paths(),
    )

    history = []
    started = time.monotonic()
    stopped_early = False
    for update in range(n_updates):
        batch = batches[update]
        bag = [entries[update % len(entries)]]
        optimizer.zero_grad(set_to_none=True)

        edge_total = torch.zeros((), device=device)
        for start in range(0, len(batch), MICROBATCH_LISTS):
            piece = batch[start : start + MICROBATCH_LISTS]
            scores = score_edge_batch(model, piece, store, device)
            edge_total = edge_total + _edge_loss(scores) * (len(piece) / len(batch))

        if args.arm == "B":
            target_total, stats = target_path_loss(model, bag, store, device)
            loss = edge_total + LAMBDA_TARGET * target_total
        else:
            target_total, stats = torch.zeros((), device=device), {}
            loss = edge_total

        if not torch.isfinite(loss):
            raise SystemExit(f"arm {args.arm}: non-finite loss at update {update}")
        loss.backward()
        optimizer.step()

        if update % 25 == 0 or update == n_updates - 1:
            history.append(
                {
                    "update": update,
                    "edge_loss": round(float(edge_total.detach()), 6),
                    "target_loss": round(float(target_total.detach()), 6),
                    "elapsed": round(time.monotonic() - started, 1),
                    **stats,
                }
            )
            print(json.dumps(history[-1]), flush=True)

        if args.checkpoint_every and (update + 1) % args.checkpoint_every == 0:
            save_checkpoint(run_dir, model, optimizer, args.arm, update + 1)
        if args.max_seconds and time.monotonic() - started > args.max_seconds:
            stopped_early = True
            print(json.dumps({"event": "time_limit", "update": update + 1}), flush=True)
            break

    final_update = history[-1]["update"] + 1 if history else 0
    path = save_checkpoint(run_dir, model, optimizer, args.arm,
                           n_updates if not stopped_early else final_update)
    receipt = {
        "status": "complete",
        "arm": args.arm,
        "lambda_target": LAMBDA_TARGET if args.arm == "B" else 0.0,
        "parent": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "seed": args.seed,
        "schedule_seed": SCHEDULE_SEED,
        "lr": LR, "weight_decay": WD,
        "logical_batch_lists": LOGICAL_BATCH_LISTS,
        "microbatch_lists": MICROBATCH_LISTS,
        "updates_requested": n_updates,
        "updates_completed": final_update,
        "stopped_early": stopped_early,
        "edge_lists": len(examples),
        "edge_list_exposure": final_update * LOGICAL_BATCH_LISTS,
        "edge_epochs": round(final_update * LOGICAL_BATCH_LISTS / len(examples), 4),
        "bag_queries_available": len(entries),
        "bag_queries_seen": min(final_update, len(entries)),
        "schedule_sha256": digest,
        "candidate_source": args.candidates,
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "history_tail": history[-6:],
        "checkpoint": path,
    }
    (run_dir / f"TRAIN_RECEIPT_{args.arm}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


def save_checkpoint(root: Path, model, optimizer, arm: str, step: int) -> str:
    directory = root / f"arm{arm}/checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step_{step:06d}.pt"
    torch.save(
        {
            "format_version": 1,
            "model_kind": "teacher_r19_global",
            "completed_stage": f"exp3-arm{arm}",
            "r19_arm": "continuation",
            "continuation_seed": SEED,
            "optimizer_step": step,
            "config": model.config(),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    return str(path)


if __name__ == "__main__":
    raise SystemExit(main())

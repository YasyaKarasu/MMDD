"""Conditional R23 ET repair (H0 old-list vs H1 natural-hard).

The core R23 endpoint is used as a common starting point.  All projections,
table-query, table-evidence, and evidence-query relations are frozen; only
text->table and image->table relations are updated.  H1 uses fixed top-32
full-lake candidates mined from train evidence with the common endpoint's
Student ANN index, while H0 keeps the original five-way ET lists.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, build_indices, load_corpus_ids, retrieve_zero_one_hop_detailed_many
from mmdd_stage1.scoring import score_edge_batch
from mmdd_stage1.training import _student_edge_losses, student_gradient_norms
from run_stage1_r21 import _load_teacher, _score_id_pairs, out as r21_out, paths as r21_paths
from run_stage1_r23 import FINAL_STEP, _score_target_ids, out as r23_out, read_rows, r23_paths, teacher_feature_paths, write_rows

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("H0-old-list", "H1-natural-hard")
SEEDS = (13, 29)
BATCH = 64
TRAIN_LISTS = 42143
UPDATES_PER_EPOCH = math.ceil(TRAIN_LISTS / BATCH)
FINAL_STEP = 2 * UPDATES_PER_EPOCH
SEED_HASH = 230913
HARD_BUDGET = 32


def root_out(root: Path) -> Path:
    return r23_out(root) / "conditional_et"


def _et_rows(root: Path) -> list[dict[str, Any]]:
    rows = [dict(row) for row in read_rows(r21_paths(root)["train_manifest"])
            if row.get("source_type") in {"text", "image"} and row.get("destination_type") == "table"]
    if not rows:
        raise RuntimeError("No train ET rows found")
    return rows


def build_manifests(root: Path, seed: int, device_name: str) -> dict[str, Any]:
    destination = root_out(root) / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    old = _et_rows(root)
    write_rows(destination / "H0-old-list.jsonl", old)
    h1_path = destination / f"H1-natural-hard_seed{seed}.jsonl"
    if not h1_path.exists():
        device = torch.device(device_name)
        common_ck = r23_out(root) / "G2-QTKD-U" / f"seed{seed}" / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
        store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=100000)
        corpus_ids = load_corpus_ids(r21_paths(root)["corpus"], store)
        index_dir = r23_out(root) / "indexes" / "G2-QTKD-U" / f"seed{seed}" / f"step_{FINAL_STEP:06d}"
        common = load_student(common_ck, device).eval()
        indices = StudentANNIndices(common, store, index_dir, device=device,
                                    checkpoint_sha256=checkpoint_fingerprint(common_ck),
                                    corpus_sha256=checkpoint_fingerprint(r21_paths(root)["corpus"]))
        source_ids = list(dict.fromkeys(str(row["query_id"]) for row in old))
        mined: dict[str, list[str]] = {}
        for start in range(0, len(source_ids), 256):
            for source_id, hits in zip(source_ids[start:start + 256], indices.search_many(source_ids[start:start + 256], "table", HARD_BUDGET)):
                mined[source_id] = [str(target) for target, _score in hits]
        rows = []
        for row in old:
            positive = [str(x) for x in row.get("positive_ids", [])] or [str(row["positive_id"])]
            ids = list(dict.fromkeys([*mined[str(row["query_id"])], *positive]))
            item = dict(row)
            item["candidate_ids"] = ids
            item["positive_ids"] = positive
            item["positive_id"] = positive[0]
            # Unknown natural candidates are intentionally not relabelled as
            # confirmed negatives; listwise training uses the known positive.
            item["confirmed_labels"] = [1 if str(value) in set(positive) else None for value in ids]
            item["candidate_source"] = "G2-final-Student-ANN-on-train-evidence"
            item["hard_budget"] = HARD_BUDGET
            rows.append(item)
        write_rows(h1_path, rows)
    result = {
        "format_version": 1, "status": "complete", "seed": seed,
        "old_rows": len(old), "h1_rows": sum(1 for _ in read_rows(h1_path)),
        "h0_sha256": checkpoint_fingerprint(destination / "H0-old-list.jsonl"),
        "h1_sha256": checkpoint_fingerprint(h1_path), "hard_budget": HARD_BUDGET,
        "candidate_generation": "train evidence only; G2 final Student ANN; no dev/test qrels",
    }
    write_json(root_out(root) / f"MANIFEST_SUMMARY_seed{seed}.json", result)
    return result


def _freeze_except_et(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "set_projection_frozen"):
        model.set_projection_frozen(True)
    trainable = []
    for relation in ("text_to_table", "image_to_table"):
        parameter = model.relations[relation]
        parameter.requires_grad_(True)
        trainable.append(parameter)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected = ["relations.image_to_table", "relations.text_to_table"]
    if sorted(names) != expected:
        raise RuntimeError(f"Unexpected ET trainable set: {names}")
    return trainable


def _save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, job: Path, arm: str, seed: int, step: int, stage: str) -> str:
    path = job / "checkpoints" / f"step_{step:06d}.pt"
    payload = {
        "format_version": 1, "model_kind": "student", "completed_stage": stage,
        "arm": arm, "seed": seed, "step": step, "config": model.config(),
        "trainable_parameters": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)
    return checkpoint_fingerprint(path)


def train(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError(f"arm must be one of {ARMS}, seed one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    build_manifests(root, seed, device_name)
    manifest = root_out(root) / "manifests" / ("H0-old-list.jsonl" if arm == "H0-old-list" else f"H1-natural-hard_seed{seed}.jsonl")
    job = root_out(root) / arm / f"seed{seed}"
    final = job / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if final.exists() and (job / "config.json").exists():
        return json.loads((job / "config.json").read_text())
    device = torch.device(device_name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    common_ck = r23_out(root) / "G2-QTKD-U" / f"seed{seed}" / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    model = load_student(common_ck, device).train()
    _freeze_except_et(model)
    optimizer = torch.optim.AdamW([{"params": model.relations["text_to_table"], "lr": 1e-5}, {"params": model.relations["image_to_table"], "lr": 1e-5}], weight_decay=0.01)
    examples = load_edge_examples(manifest, split="train")
    if not examples:
        raise RuntimeError("Empty ET manifest")
    expanded = (examples * math.ceil(TRAIN_LISTS / len(examples)))[:TRAIN_LISTS]
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=100000)
    job.mkdir(parents=True, exist_ok=True); (job / "checkpoints").mkdir(exist_ok=True)
    init_hash = _save(model, optimizer, job, arm, seed, 0, "r23-et-initial-common-G2")
    rng = random.Random(SEED_HASH + seed)
    order = list(range(len(expanded)))
    history, diagnostics = [], []
    step = 0; started = time.monotonic()
    for epoch in range(2):
        rng.shuffle(order); epoch_losses = []
        for start in range(0, len(order), BATCH):
            batch = [expanded[index] for index in order[start:start + BATCH]]
            optimizer.zero_grad(set_to_none=True)
            scores = score_edge_batch(model, batch, store, device)
            terms = _student_edge_losses(model, batch, scores, None, scores, None,
                                         ranking_weight=1.0, temperature=1.0,
                                         distillation_weight=0.0, edge_bce_weight=0.0,
                                         anchor_weight=0.1, anchor_weight_evidence=0.1)
            terms["loss"].backward()
            grad = student_gradient_norms(model)
            optimizer.step(); step += 1
            value = float(terms["loss"].detach().cpu()); epoch_losses.append(value)
            if step <= 2 or step % 100 == 0:
                diagnostics.append({"step": step, "epoch": epoch + 1, "loss": value,
                                    "supervised_loss": float(terms["supervised_loss"].detach().cpu()),
                                    "gradient": grad})
        ck_hash = _save(model, optimizer, job, arm, seed, step, "r23-et")
        history.append({"epoch": epoch + 1, "step": step, "loss": statistics.fmean(epoch_losses), "checkpoint_sha256": ck_hash})
        write_rows(job / "train_history.jsonl", history)
        write_rows(job / "step_diagnostics.jsonl", diagnostics)
    cfg = {
        "format_version": 1, "status": "pass", "arm": arm, "seed": seed,
        "stage": "conditional_ET", "common_initialization": "R23-G2-final",
        "common_checkpoint_sha256": checkpoint_fingerprint(common_ck),
        "manifest_sha256": checkpoint_fingerprint(manifest), "hard_budget": HARD_BUDGET if arm == "H1-natural-hard" else None,
        "frozen_components": ["all projections", "relations.table_to_table", "relations.table_to_text", "relations.table_to_image"],
        "trainable_parameters": ["relations.text_to_table", "relations.image_to_table"],
        "optimizer": {"relation_lr": 1e-5, "weight_decay": 0.01},
        "anchor": {"weight": 0.1, "evidence": 0.1}, "batch_size": BATCH,
        "epochs": 2, "updates": step, "initial_checkpoint_sha256": init_hash,
        "history": history, "diagnostics": str((job / "step_diagnostics.jsonl").resolve()),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Correct the human-readable frozen list after keeping the exact parameter
    # names above as the source of truth.
    cfg["frozen_components"] = ["all projections", "relations.table_to_table", "relations.table_to_text", "relations.table_to_image"]
    write_json(job / "config.json", cfg)
    return cfg


@torch.inference_mode()
def evaluate(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    """Re-run full-lake fusion with ET-repaired checkpoints."""
    device = torch.device(device_name)
    ck = root_out(root) / arm / f"seed{seed}" / "checkpoints" / f"step_{FINAL_STEP:06d}.pt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=120000)
    model = load_student(ck, device).eval()
    ids_by_type = load_corpus_ids(r21_paths(root)["corpus"], store)
    index_dir = root_out(root) / "indexes" / arm / f"seed{seed}" / f"step_{FINAL_STEP:06d}"
    if not (index_dir / "manifest.json").exists():
        build_indices(model, store, ids_by_type, index_dir, device=device,
                      checkpoint_sha256=checkpoint_fingerprint(ck), corpus_sha256=checkpoint_fingerprint(r21_paths(root)["corpus"]), batch_size=4096)
    indices = StudentANNIndices(model, store, index_dir, device=device,
                                checkpoint_sha256=checkpoint_fingerprint(ck), corpus_sha256=checkpoint_fingerprint(r21_paths(root)["corpus"] ))
    pools = list(read_rows(r23_paths(root)["candidate_pools"]))
    detailed = retrieve_zero_one_hop_detailed_many([str(row["query_id"]) for row in pools], indices, k=100, direct_k=100, evidence_k=20, targets_per_evidence=20, query_batch_size=16)
    rows = []
    for pool, ret in zip(pools, detailed):
        positives = set(map(str, pool["positive_target_ids"]))
        direct = [str(item["target_id"]) for item in ret["direct"]]
        evidence = [str(item["target_id"]) for item in ret["evidence"]]
        union = list(dict.fromkeys([*direct, *evidence]))
        scores = _score_target_ids(model, store, str(pool["query_id"]), union, device)
        ranked = sorted(union, key=lambda value: (-scores[value], value))
        rows.append({"query_id": str(pool["query_id"]), "query_kind": pool["query_kind"], "positive_target_ids": sorted(positives),
                     "direct_ann": direct, "evidence_ann": evidence, "U": union, "evidence_paths": ret.get("evidence", []),
                     "direct_raw_recall@100": len(positives & set(direct)) / len(positives) if positives else 0.0,
                     "u_raw_recall": len(positives & set(union)) / len(positives) if positives else 0.0,
                     "u_exact_ranking": ranked, "u_scores": [scores[value] for value in ranked],
                     **{f"u_recall@{k}": len(positives & set(ranked[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}})
    destination = root_out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "rankings.jsonl.gz", rows)
    result = {"format_version": 1, "status": "complete", "arm": arm, "seed": seed, "step": FINAL_STEP,
              "queries": len(rows), "direct_raw@100": statistics.fmean(row["direct_raw_recall@100"] for row in rows),
              "u_raw_recall": statistics.fmean(row["u_raw_recall"] for row in rows), "u_r10": statistics.fmean(row["u_recall@10"] for row in rows),
              "u_r20": statistics.fmean(row["u_recall@20"] for row in rows), "u_r50": statistics.fmean(row["u_recall@50"] for row in rows),
              "rankings": str((destination / "rankings.jsonl.gz").resolve()), "checkpoint_sha256": checkpoint_fingerprint(ck)}
    write_json(destination / "metrics.json", result)
    return result


@torch.inference_mode()
def evaluate_fixed_teacher(root: Path, arm: str, seed: int, device_name: str) -> dict[str, Any]:
    """Score each H U candidate set with the frozen R21 D2 Teacher."""
    device = torch.device(device_name)
    source = root_out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "rankings.jsonl.gz"
    if not source.exists():
        raise FileNotFoundError(source)
    destination = source.parent / "fixed_teacher"
    metrics_path = destination / "metrics.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text())
        if previous.get("status") == "complete" and previous.get("teacher_candidate_coverage", 0.0) >= 1.0:
            return previous
    teacher, _teacher_payload = _load_teacher(root, seed, device)
    teacher_path = r21_paths(root)[f"teacher{seed}"]
    teacher.eval()
    store = FeatureStore.from_path(r23_paths(root)["features"], cache_size=30000, teacher_paths=teacher_feature_paths(root))
    cache = teacher.new_compression_cache()
    rows = []
    missing_candidates = 0
    total_candidates = 0
    for row in read_rows(source):
        candidate_ids = list(dict.fromkeys(map(str, row["U"])))
        available, unavailable = [], []
        for candidate in candidate_ids:
            total_candidates += 1
            try:
                if store.get(candidate, include_hidden=True).hidden_states is None:
                    raise ValueError("missing Teacher hidden states")
            except ValueError:
                unavailable.append(candidate)
            else:
                available.append(candidate)
        missing_candidates += len(unavailable)
        values = _score_id_pairs(teacher, [(str(row["query_id"]), candidate) for candidate in available], store, device, batch_size=256, cache=cache) if available else []
        ranking = [candidate for candidate, _value in sorted(zip(available, values), key=lambda pair: (-pair[1], pair[0]))] + sorted(unavailable)
        positives = set(map(str, row["positive_target_ids"]))
        rows.append({"query_id": str(row["query_id"]), "query_kind": row["query_kind"], "positive_target_ids": sorted(positives), "candidate_ids": candidate_ids, "ranking": ranking,
                     "raw_recall": len(positives & set(candidate_ids)) / len(positives) if positives else 0.0,
                     **{f"recall@{k}": len(positives & set(ranking[:k])) / len(positives) if positives else 0.0 for k in (10, 20, 50)}})
    destination.mkdir(parents=True, exist_ok=True)
    ranking_path = destination / "rankings.jsonl.gz"
    write_rows(ranking_path, rows)
    aggregate = lambda values: {key: statistics.fmean(float(row[key]) for row in values) for key in ("raw_recall", "recall@10", "recall@20", "recall@50")}
    result = {"format_version": 1, "status": "complete" if missing_candidates == 0 else "partial", "arm": arm, "seed": seed, "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_path), "source_rankings_sha256": checkpoint_fingerprint(source), "queries": len(rows), "u": aggregate(rows), "by_query_kind": {kind: aggregate([row for row in rows if row["query_kind"] == kind]) for kind in ("implicit", "explicit")}, "rankings": str(ranking_path.resolve()), "teacher_candidate_coverage": 1.0 - missing_candidates / total_candidates if total_candidates else 0.0, "missing_candidate_count": missing_candidates, "total_candidate_count": total_candidates, "note": "Missing Teacher hidden-state candidates are placed after scored candidates; recall is a lower-bound partial audit." if missing_candidates else None}
    write_json(metrics_path, result)
    return result


def report(root: Path) -> dict[str, Any]:
    jobs = []
    for arm in ARMS:
        for seed in SEEDS:
            base = root_out(root) / arm / f"seed{seed}"
            full = root_out(root) / "full_lake" / arm / f"seed{seed}_step{FINAL_STEP:06d}" / "metrics.json"
            fixed_teacher = full.parent / "fixed_teacher" / "metrics.json"
            teacher_metrics = json.loads(fixed_teacher.read_text()) if fixed_teacher.exists() else None
            jobs.append({"arm": arm, "seed": seed, "train": (base / "config.json").exists(),
                         "full_lake": full.exists(), "fixed_teacher": fixed_teacher.exists(),
                         "fixed_teacher_status": teacher_metrics["status"] if teacher_metrics else "missing"})
    status = "complete" if all(item["train"] and item["full_lake"] and item["fixed_teacher"] for item in jobs) else "partial"
    result = {"format_version": 1, "status": status, "jobs": jobs,
              "note": "Conditional H0/H1 was triggered by two-seed ET exact hub concentration; no T2 was run."}
    write_json(root_out(root) / "CONDITIONAL_ET_AUDIT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-manifests"); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("train"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("evaluate"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0")
    p = sub.add_parser("fixed-teacher"); p.add_argument("--arm", required=True); p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda:0")
    sub.add_parser("report")
    args = parser.parse_args(); root = args.root.resolve()
    if args.cmd == "build-manifests": print(json.dumps(build_manifests(root, args.seed, args.device), indent=2))
    elif args.cmd == "train": print(json.dumps(train(root, args.arm, args.seed, args.device), indent=2))
    elif args.cmd == "evaluate": print(json.dumps(evaluate(root, args.arm, args.seed, args.device), indent=2))
    elif args.cmd == "fixed-teacher": print(json.dumps(evaluate_fixed_teacher(root, args.arm, args.seed, args.device), indent=2))
    else: print(json.dumps(report(root), indent=2))


if __name__ == "__main__":
    main()

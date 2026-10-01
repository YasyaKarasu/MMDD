"""Controlled AbeBooks data ablation using the current CQET training functions.

Five required stages: TA, TB_CQET, Native C1, paired SUP/KD C2. Training
defaults and dev selection match pipeline.py; unrelated QT/LSE controls and
retrospective gradient probes are outside this dataset experiment.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import torch

from run_abebooks_fresh import ROOT
from mmdd_stage1 import pipeline, preflight
from mmdd_stage1.artifacts import save_pool_bundle, save_training_records
from mmdd_stage1.config import resolve_default_paths
from mmdd_stage1.data import (build_content_aliases, iter_jsonl, read_json,
                                sha256_file, write_json, write_jsonl)
from mmdd_stage1.evaluate import evaluate_student_retrieval, evaluate_teacher_matrix
from mmdd_stage1.features import (ContentStore, ObjectBank, build_or_load_row_store,
                                    fit_pca, load_pca, load_z)
from mmdd_stage1.labels import build_labels, export_eval_labels, load_labels
from mmdd_stage1.lists import (build_c1_edge_lists, build_c2_shared_graph,
                                 build_raw_et128_exact, build_raw_pools_split,
                                 build_ta_records, build_tb_records)
from mmdd_stage1.metrics import evaluate_matrix, export_funnels
from mmdd_stage1.models import NativeStudent
from mmdd_stage1.train import (build_teacher_logits_cache, model_state_sha,
                                 train_student_c1, train_student_c2, train_ta, train_tb)


def runtime(run: Path, prepare: bool = False) -> pipeline.Runtime:
    paths = resolve_default_paths(run / "protocol.json", run)
    if prepare:
        preflight.configure(paths)
        preflight.build_content_aliases()
        preflight.build_cache_manifest()
        build_labels(paths, build_content_aliases(paths))
    labels = load_labels(paths)
    z = load_z(paths)
    rows = build_or_load_row_store(paths)
    if prepare:
        fit_pca(paths, z, labels)
    basis, mean = load_pca(paths)
    bank = ObjectBank(z, ContentStore(paths.pure_cache_dir / "content", lru_bytes=8 * 2**30))
    return pipeline.Runtime(paths, read_json(paths.protocol_path), labels, z, rows, bank, basis, mean)


def select(rt: pipeline.Runtime, checkpoints: dict, gt: dict, stage: str) -> dict:
    points = []
    for fraction, checkpoint in sorted(checkpoints.items()):
        model = pipeline._load_native(checkpoint, rt)
        pools = evaluate_student_retrieval(model, rt.z_store, rt.row_store, sorted(gt),
                                          rt.labels, "dev", hnsw_seed=13, generator_id=stage)
        metrics = pipeline._candidate_summary(pools, gt)
        key = [metrics["C150_target_coverage"], metrics["U_target_coverage"],
               metrics["Direct_ANN_R10"], -fraction]
        points.append({"fraction": fraction, "checkpoint": str(checkpoint),
                       "checkpoint_sha256": sha256_file(checkpoint), "metrics": metrics,
                       "state_sha256": model_state_sha(model), "key": key})
        del model, pools
        torch.cuda.empty_cache()
    chosen = max(points, key=lambda row: row["key"])
    result = {"owner": stage, "points": points, "selected": chosen}
    write_json(rt.paths.run_root / f"selection_{stage}.json", result)
    print(f"SELECT {stage} fraction={chosen['fraction']} metrics={chosen['metrics']}", flush=True)
    return result


def train(run: Path, student_epochs: int = 1, student_batch: int = 64,
          student_lr_p: float = 1e-4, student_lr_r: float = 1e-3) -> None:
    if (run / "TRAINING_COMPLETE.json").exists():
        raise FileExistsError("Training already completed; use evaluate")
    started = time.time()
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for directory in
                     (ROOT / "src/mmdd_stage1",)
                     for p in directory.glob("*.py")}
    write_json(run / "EXPERIMENT_PROTOCOL.json", {
        "seed": 13, "old_checkpoints_reused": False,
        "feature_composition": (read_json(run / "FEATURE_COMPOSITION.json")
                                if (run / "FEATURE_COMPOSITION.json").exists() else None),
        "TA_epochs": 2, "TB_epochs": 1, "teacher_batch": 8,
        "C1_passes": student_epochs, "C2_passes": student_epochs, "student_batch": student_batch,
        "student_lr_P": student_lr_p, "student_lr_R": student_lr_r,
        "C2_KD_selection": "same fraction chosen by SUP dev metrics",
        "source_hashes": source_hashes,
        "hub_rule": "baseline Raw top20 per modality, >=10% distinct TRAIN queries, exclude all-split gold content classes",
        "test_labels_usage": "global column curation and gold-evidence protection only; no frequency or hyperparameter tuning",
        "stages": ["TA", "TB_CQET", "NATIVE_C1", "NATIVE_C2_SUP", "NATIVE_C2_KD"],
    })
    print("PREPARE aliases, labels, fresh PCA", flush=True)
    rt = runtime(run, prepare=True)
    ids = rt.labels.query_ids
    gt = export_eval_labels(rt.paths, rt.labels.canonical_map, "dev")
    print("RAW TRAIN POOLS", flush=True)
    raw = build_raw_pools_split(rt.z_store, rt.row_store, ids, rt.labels, "train", hnsw_seed=13)
    counts = Counter()
    for pool in raw.values():
        counts.update({e for values in pool.first_hop.values() for e, _ in values})
    write_jsonl(run / "train_evidence_popularity.jsonl", [
        {"evidence_id": e, "train_queries": n, "fraction": n / len(ids)}
        for e, n in counts.most_common()])
    write_json(run / "TRAIN_POPULARITY_READY.json", {"queries": len(ids), "threshold": math.ceil(0.1 * len(ids))})
    records_dir = run / "training_records"
    save_pool_bundle(records_dir / "raw_train", raw, rt.labels, seed=13, generator="raw")
    for pool in raw.values():
        pool.d1_trace = {}
    raw_et = build_raw_et128_exact(rt.z_store, rt.labels)
    ta = [build_ta_records(q, raw[q], rt.labels, 13, raw_et) for q in ids]
    tb = [build_tb_records(q, raw[q], rt.labels, 13) for q in ids]
    c1 = build_c1_edge_lists(rt.labels, raw, rt.z_store, 13, raw_et)
    for name, records in (("TA", ta), ("TB", tb), ("C1", c1)):
        save_training_records(records_dir / f"{name}.jsonl.gz", records)
    pipeline._set_seed(13, "TA")
    teacher = pipeline._teacher()
    print("TRAIN TA", flush=True)
    ta_ckpt = train_ta(teacher, rt.bank, ta, rt.labels, save_dir=run / "TA", log_path=run / "TA.jsonl")
    del teacher
    pipeline._set_seed(13, "TB_SHARED")
    teacher = pipeline._load_teacher(ta_ckpt)
    print("TRAIN TB_CQET", flush=True)
    tb_ckpt = train_tb(teacher, rt.bank, tb, mode="cqet", save_dir=run / "TB_CQET", log_path=run / "TB_CQET.jsonl")
    del teacher
    student = NativeStudent(rt.pca_basis, rt.pca_mean)
    print("TRAIN C1", flush=True)
    c1_points = train_student_c1(student, c1, None, rt.bank, arm="NATIVE_SUP",
                                 save_dir=run / "C1", log_path=run / "C1.jsonl",
                                 epochs=student_epochs, logical_batch=student_batch,
                                 lr_p=student_lr_p, lr_r=student_lr_r)
    del student
    choice1 = select(rt, c1_points, gt, "C1")
    parent = pipeline._load_native(Path(choice1["selected"]["checkpoint"]), rt)
    parent_hash = model_state_sha(parent)
    c1_pools = evaluate_student_retrieval(parent, rt.z_store, rt.row_store, ids, rt.labels,
                                         "train", hnsw_seed=13, generator_id="selected_Native_C1")
    c2 = build_c2_shared_graph(raw, c1_pools, parent, rt.z_store, rt.row_store,
                              rt.labels, ids, records_dir / "C2_PREPATHS.jsonl.gz")
    del raw, c1_pools, parent
    save_training_records(records_dir / "C2_SHARED.jsonl.gz", c2)
    all_points = {}
    for arm in ("SUP", "KD"):
        student = pipeline._load_native(Path(choice1["selected"]["checkpoint"]), rt)
        logits = None
        if arm == "KD":
            teacher = pipeline._load_teacher(tb_ckpt)
            logits = build_teacher_logits_cache(teacher, rt.bank, c2)
            torch.save(logits, run / "teacher_c2_logits.pt")
            del teacher
        print(f"TRAIN C2 {arm}", flush=True)
        all_points[arm] = train_student_c2(student, c2, None, rt.bank, arm=f"NATIVE_{arm}",
            save_dir=run / f"C2_{arm}", expected_parent_hash=parent_hash, teacher_logits=logits,
            log_path=run / f"C2_{arm}.jsonl", epochs=student_epochs, logical_batch=student_batch,
            lr_p=student_lr_p, lr_r=student_lr_r)
        del student, logits
        torch.cuda.empty_cache()
    choice2 = select(rt, all_points["SUP"], gt, "C2_SUP")
    fraction = choice2["selected"]["fraction"]
    selected = {arm: str(points[fraction]) for arm, points in all_points.items()}
    endpoints = {arm: str(points[1.0]) for arm, points in all_points.items()}
    write_json(run / "SELECTION_FREEZE.json", {"status": "FROZEN_BEFORE_TEST",
        "C1_fraction": choice1["selected"]["fraction"], "C2_fraction": fraction,
        "selected": selected, "endpoints": endpoints, "teacher": str(tb_ckpt),
        "hashes": {p: sha256_file(Path(p)) for p in [*selected.values(), *endpoints.values(), str(tb_ckpt)]}})
    write_json(run / "TRAINING_COMPLETE.json", {"seconds": time.time() - started,
        "counts": {"TA": len(ta), "TB": len(tb), "C1": len(c1), "C2": len(c2)},
        "steps": {"TA": 2 * math.ceil(len(ta) / 8), "TB": math.ceil(len(tb) / 8),
                  "C1": student_epochs * math.ceil(len(c1) / student_batch),
                  "C2_SUP": student_epochs * math.ceil(len(c2) / student_batch),
                  "C2_KD": student_epochs * math.ceil(len(c2) / student_batch)}})
    print("TRAINING COMPLETE", flush=True)


def evaluate(run: Path, splits: tuple[str, ...] = ("dev", "test"), device: str = "cuda:0",
             retrieval_only: bool = False) -> None:
    if not (run.parent / "ALL_SELECTIONS_FROZEN.json").exists():
        raise ValueError("Freeze every registered arm before test evaluation")
    rt = runtime(run)
    freeze = read_json(run / "SELECTION_FREEZE.json")
    for p, digest in freeze["hashes"].items():
        assert sha256_file(Path(p)) == digest
    teacher = None if retrieval_only else pipeline._load_teacher(Path(freeze["teacher"]), device=device)
    checkpoints = {"raw": None, **{f"selected_{k.lower()}": v for k, v in freeze["selected"].items()},
                   **{f"endpoint_{k.lower()}": v for k, v in freeze["endpoints"].items()}}
    summaries, per_query = [], []
    for split in splits:
        gt = export_eval_labels(rt.paths, rt.labels.canonical_map, split)
        for generator, checkpoint in checkpoints.items():
            print(f"EVAL {split} {generator}", flush=True)
            if checkpoint is None:
                pools = build_raw_pools_split(rt.z_store, rt.row_store, sorted(gt), rt.labels, split,
                                             hnsw_seed=13, device=device)
            else:
                model = pipeline._load_native(Path(checkpoint), rt, device=device)
                pools = evaluate_student_retrieval(model, rt.z_store, rt.row_store, sorted(gt), rt.labels,
                                                  split, hnsw_seed=13, generator_id=generator, device=device)
                del model
            out = run / "eval" / split / generator
            save_pool_bundle(out, pools, rt.labels, seed=13, generator=generator)
            orders = {"Direct": {q: [t for t, _ in p.direct] for q, p in pools.items()},
                      "Multimodal_RRF": {q: p.C150 for q, p in pools.items()}}
            if not retrieval_only:
                matrix = evaluate_teacher_matrix({"TB_CQET": teacher}, rt.bank, pools, rt.labels,
                    seed=13, generator=generator, split=split, output_dir=out, split_gt=gt, device=device)
                evaluate_matrix(pools, matrix, gt, out)
                export_funnels(pools, matrix["TB_CQET"]["Real"], gt, rt.labels,
                               out / "funnels", seed=13, generator=generator)
                for view in ("Real", "f0", "Swap"):
                    orders[f"Teacher_{view}"] = {r["query_id"]: r["target_ids"] for r in
                        iter_jsonl(out / f"rankings.TB_CQET.{view}.jsonl.gz")}
                del matrix
            for mode, rankings in orders.items():
                scores = {}
                for q, gold in gt.items():
                    ranked = rankings[q]
                    assert len(ranked) == len(set(ranked))
                    scores[q] = {f"R@{k}": len(set(ranked[:k]) & set(gold["G"])) / len(gold["G"])
                                 for k in (5, 10, 15, 20)}
                    per_query.append({"split": split, "generator": generator, "mode": mode,
                                      "query_id": q, "kind": gold["kind"], **scores[q]})
                for segment in ("overall", "implicit", "explicit"):
                    qids = [q for q in gt if segment == "overall" or gt[q]["kind"] == segment]
                    summaries.append({"split": split, "generator": generator, "mode": mode,
                        "segment": segment, "queries": len(qids), **{f"R@{k}":
                        sum(scores[q][f"R@{k}"] for q in qids) / len(qids) if qids else None
                        for k in (5, 10, 15, 20)}})
            prefix = "retrieval_" if retrieval_only else ""
            write_json(run / f"{prefix}recall_summary.json", summaries)
            write_jsonl(run / f"{prefix}recall_per_query.jsonl", per_query)
            del pools
            torch.cuda.empty_cache()
    completion = "RETRIEVAL_EVALUATION_COMPLETE.json" if retrieval_only else "EVALUATION_COMPLETE.json"
    write_json(run / completion, {"status": "COMPLETE", "splits": list(splits),
                                                "rows": len(summaries), "device": device})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--student-epochs", type=int, default=1)
    parser.add_argument("--student-batch", type=int, default=64)
    parser.add_argument("--student-lr-p", type=float, default=1e-4)
    parser.add_argument("--student-lr-r", type=float, default=1e-3)
    parser.add_argument("--splits", choices=("dev", "test"), nargs="+", default=["dev", "test"])
    parser.add_argument("--device", default="cuda:0", help="Evaluation device; training settings are unchanged")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Evaluate unchanged Direct/RRF retrieval without teacher reranking diagnostics")
    args = parser.parse_args()
    if args.command == "train":
        train(args.run_root.resolve(), args.student_epochs, args.student_batch,
              args.student_lr_p, args.student_lr_r)
    else:
        evaluate(args.run_root.resolve(), tuple(args.splits), device=args.device,
                 retrieval_only=args.retrieval_only)


if __name__ == "__main__":
    main()

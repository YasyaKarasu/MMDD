"""End-to-end smoke test of the CLEAN-R1 pipeline on a tiny query subset.

Runs the real producers (packet building, both training trajectories, the
retrieval engine, the teacher re-ranking) against the real cache, but with a
handful of queries so the whole chain can be exercised in minutes instead of
days.  Nothing here writes to the run's own artifacts: output goes to a separate
`SMOKE` directory, so a smoke failure can never corrupt a real run.

Pass --queries to change the subset size, --device to move it off the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mmdd_stage1_clean import evaluate, models, train
from mmdd_stage1_clean.commands import (
    _load_bank_objects,
    _load_rank_tables,
    _make_builder,
    _model_hash,
    _population_index,
    _save_checkpoint,
    run_split,
    score_selection,
)
from mmdd_stage1_clean.data import build_gt
from mmdd_stage1_clean.models import build_student, build_teacher
from mmdd_stage1_clean.util import read_json, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path,
                        default=REPO / "work/s1_clean_r1_20260917")
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--dev-queries", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = args.run_root / "SMOKE"
    out.mkdir(parents=True, exist_ok=True)
    resolved = read_json(args.run_root / "resolved_config.json")
    report: dict[str, object] = {"stages": {}, "failed_stage": None}

    def stage(name):
        class _S:
            def __enter__(self_):
                self_.t0 = time.time()
                print(f"\n=== {name} ===", flush=True)
                return self_

            def __exit__(self_, exc_type, exc, tb):
                dt = time.time() - self_.t0
                report["stages"][name] = {
                    "seconds": round(dt, 1),
                    "ok": exc_type is None,
                    "error": None if exc_type is None else f"{exc_type.__name__}: {exc}",
                }
                if exc_type is not None:
                    report["failed_stage"] = name
                    traceback.print_exception(exc_type, exc, tb)
                    return True  # swallow so we can report every stage
                return False
        return _S()

    t_all = time.time()
    with stage("load_bank"):
        bank, corpora = _load_bank_objects(args.run_root, resolved)
        print(f"bank={len(bank.object_ids)} objects; "
              f"corpora target={len(corpora['target'])} "
              f"evidence={len(corpora['evidence'])}")

    with stage("build_builder_and_ranks"):
        builder = _make_builder(args.run_root, corpora)
        rank_tables, anchor_rank = _load_rank_tables(
            args.run_root, "train", resolved["retrieval"]
        )
        # A private copy of the GT with a short train population: the builder
        # samples from the full corpus, only the query set shrinks.
        train_rows = builder.population("train")[: args.queries]
        builder.gt["train"] = {"population": train_rows}
        print(f"train queries={len(train_rows)} "
              f"anchor ranks={len(anchor_rank)}")

    with stage("teacher_train"):
        teacher_dir = out / "teacher"
        trainer = train.TeacherTrainer(
            resolved=resolved, bank=bank, builder=builder,
            rank_tables=rank_tables, anchor_rank=anchor_rank,
            output_dir=teacher_dir, device=args.device, max_epochs=args.epochs,
        )
        result = trainer.train()
        _save_checkpoint(teacher_dir / "best.pt",
                         model=trainer.teacher.state_dict(), epoch=trainer.epochs)
        print(f"teacher loss={result['epochs'][-1]['mean_loss']:.4f} "
              f"steps={result['steps']}")
        print("packet terms:", result["epochs"][-1].get("packet_terms"))

    with stage("student_SUP"):
        sup_dir = out / "student_SUP"
        init = build_student(resolved["student"]).state_dict()
        sup = train.StudentTrainer(
            resolved=resolved, bank=bank, builder=builder,
            rank_tables=rank_tables, anchor_rank=anchor_rank,
            output_dir=sup_dir, arm="SUP", init_state=init,
            device=args.device, max_epochs=args.epochs,
        )
        r = sup.train(teacher=None, teacher_batch=None, query_rankings={})
        _save_checkpoint(sup_dir / "best.pt", model=sup.student.state_dict(), arm="SUP")
        print(f"SUP loss={r['epochs'][-1]['mean_loss']:.4f}")

    with stage("student_KD"):
        kd_dir = out / "student_KD"
        teacher, _ = _load_teacher(out / "teacher" / "best.pt", resolved, args.device)
        teacher_hash = _model_hash(teacher)
        batch = train.TeacherBatch(bank, device=args.device)
        kd = train.StudentTrainer(
            resolved=resolved, bank=bank, builder=builder,
            rank_tables=rank_tables, anchor_rank=anchor_rank,
            output_dir=kd_dir, arm="KD", init_state=init,
            device=args.device, max_epochs=args.epochs,
        )
        r = kd.train(teacher=teacher, teacher_batch=batch, query_rankings={})
        _save_checkpoint(kd_dir / "best.pt", model=kd.student.state_dict(), arm="KD")
        print(f"KD loss={r['epochs'][-1]['mean_loss']:.4f}")

    with stage("evaluate_dev"):
        from mmdd_stage1_clean.models import build_student as _bs
        student = _bs(resolved["student"])
        student.load_state_dict(kd.student.state_dict())
        student.to(args.device)
        for p in student.parameters():
            p.requires_grad_(False)
        engine = evaluate.RetrievalEngine(
            student=student, bank=bank, corpora=corpora,
            retrieval=resolved["retrieval"], ann=resolved["ann"],
            seed=int(resolved["seed"]), device=args.device, build_ann=True,
        )
        cache = evaluate.TeacherLogitCache(out / "teacher_logits.sqlite")
        summary = run_split(
            output_root=args.run_root, resolved=resolved, split="dev",
            teacher=teacher, teacher_hash=teacher_hash, teacher_batch=batch,
            engine=engine, method="KD+T", logit_cache=cache,
            max_queries=args.dev_queries, write_artifacts=False, run_exact=True,
        )
        for ranker in ("ann", "exact"):
            view = summary.get("views", {}).get(ranker)
            if view:
                m = view["teacher_reranked"]
                print(f"  {ranker}: R@10={m['overall_R10']:.4f} "
                      f"implicit={m['implicit_R10']} explicit={m['explicit_R10']} "
                      f"C100_recall={view['candidate_recall']['C100']:.4f}")
        print("nn_fidelity:", summary["nn_fidelity"])
        cache.close()

    report["total_seconds"] = round(time.time() - t_all, 1)
    write_json(out / "SMOKE_REPORT.json", report)
    print("\n=== summary ===")
    for name, entry in report["stages"].items():
        flag = "ok" if entry["ok"] else "FAILED"
        print(f"  {name:28s} {entry['seconds']:7.1f}s  {flag}"
              + ("" if entry["ok"] else f"  {entry['error']}"))
    print(f"  total {report['total_seconds']}s")
    return 0 if report["failed_stage"] is None else 1


def _load_teacher(path: Path, resolved: dict, device: str):
    teacher = build_teacher(resolved["teacher"]).to(device)
    import torch

    teacher.load_state_dict(
        torch.load(path, map_location=device, weights_only=False)["model"]
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, None


if __name__ == "__main__":
    raise SystemExit(main())

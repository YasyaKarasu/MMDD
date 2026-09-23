#!/usr/bin/env python
"""MMDD FRESH-RECOVERY v3.1 execution entry point."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fresh_recovery.config import resolve_paths


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="MMDD FRESH-RECOVERY v3.1")
    ap.add_argument("--dataset-root", type=Path, default=root / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9")
    package_dir = root / "MMDD_FRESH_RECOVERY_v3_1_20260923"
    if not package_dir.exists() and (root / "audit" / "MMDD_FRESH_RECOVERY_v3_1_20260923").exists():
        package_dir = root / "audit" / "MMDD_FRESH_RECOVERY_v3_1_20260923"
    ap.add_argument("--package-dir", type=Path, default=package_dir)
    ap.add_argument("--work-dir", type=Path, default=root / "work" / "mmdd_fresh_recovery_v3_1_20260923")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, *, seed=False, device=False, **extra):
        p = sub.add_parser(name)
        if seed:
            p.add_argument("--seed", type=int, required=True)
        if device:
            p.add_argument("--device", default="cuda:0")
        for flag, kwargs in extra.items():
            p.add_argument(f"--{flag.replace('_', '-')}", **kwargs)
        return p

    add("resolve")
    add("build-labels")
    add("prepare-data")
    add("audit-features", pure_cache_dir=dict(type=Path, required=True))
    add("recompute-feature-samples", device=True, pure_cache_dir=dict(type=Path, required=True), samples_per_kind=dict(type=int, default=1))
    add("audit-row-cache", row_cache_dir=dict(type=Path, required=True))
    add("recompute-row-samples", device=True, row_cache_dir=dict(type=Path, required=True))
    add("fit-pca")
    add("build-rows")
    add("build-raw", device=True, split=dict(choices=["train", "dev", "test"], required=True))
    add("build-lists", seed=True)
    add("init-teacher", seed=True)
    add("eval-init", seed=True, device=True)
    add("verify-integration", seed=True, device=True)
    add("train-bootstrap", seed=True, device=True)
    add("refresh-lists", seed=True, device=True)
    add("train-c1", seed=True, device=True, arm=dict(required=True))
    add("select-c1", seed=True, pair=dict(choices=["native", "qt"], required=True))
    add("build-c2-graphs", seed=True, device=True, pair=dict(choices=["native", "qt"], required=True))
    add("train-c2", seed=True, device=True, arm=dict(required=True))
    add("select-c2", seed=True, pair=dict(choices=["native", "qt"], required=True))
    add("mine-qt-hard", seed=True, device=True)
    add("train-qt-teacher", seed=True, device=True)
    add("train-qt-teacher-from-boot", seed=True, device=True)
    add("train-qt-cont", seed=True, device=True)
    add("train-path-teacher", seed=True, device=True)
    add("freeze-models", seed=True)
    add("evaluate-dev", seed=True, device=True)
    add("evaluate-test", seed=True, device=True)
    add("measure-latency", seed=True, device=True)
    add("run", seed=True, gpus=dict(default="0,1"), max_stage=dict(default=None),
        gpu_slots_per_device=dict(type=int, choices=(1, 2), default=1))
    add("package", seed=True)
    add("write-report", seed=True)
    add("compact-eval", seed=True)
    add("recompute-metrics", seed=True, split=dict(choices=["dev", "test"], required=True))
    add("decide", seed=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = resolve_paths(args.dataset_root, args.backbone_dir, args.package_dir, args.work_dir)
    cmd = args.command
    started = time.time()
    if cmd in ("resolve", "build-labels", "prepare-data", "audit-features", "recompute-feature-samples",
               "audit-row-cache", "recompute-row-samples", "fit-pca"):
        from fresh_recovery.features import audit_pure_cache, audit_row_cache, recompute_cache_samples, recompute_row_samples
        from fresh_recovery.labels import build_labels
        from fresh_recovery.pca import fit_pca
        from fresh_recovery.prepare import build_content_aliases, resolve_inputs

        if cmd == "resolve":
            result = resolve_inputs(paths)
        elif cmd == "build-labels":
            result = build_labels(paths, build_content_aliases(paths))
        elif cmd == "prepare-data":
            resolve_inputs(paths)
            result = build_labels(paths, build_content_aliases(paths))
        elif cmd == "audit-features":
            result = audit_pure_cache(paths, args.pure_cache_dir)
        elif cmd == "recompute-feature-samples":
            result = recompute_cache_samples(paths, args.pure_cache_dir, device=args.device, samples_per_kind=args.samples_per_kind)
        elif cmd == "audit-row-cache":
            result = audit_row_cache(paths, args.row_cache_dir)
        elif cmd == "recompute-row-samples":
            result = recompute_row_samples(paths, args.row_cache_dir, device=args.device)
        else:
            result = fit_pca(paths)
    else:
        from fresh_recovery import final_eval, stages

        if cmd == "build-rows":
            result = stages.cmd_build_rows(paths)
        elif cmd == "build-raw":
            result = stages.cmd_build_raw(paths, split=args.split, device=args.device)
        elif cmd == "build-lists":
            result = stages.cmd_build_lists(paths, seed=args.seed)
        elif cmd == "init-teacher":
            result = stages.cmd_init_teacher(paths, seed=args.seed)
        elif cmd == "eval-init":
            result = stages.cmd_eval_init(paths, seed=args.seed, device=args.device)
        elif cmd == "verify-integration":
            from fresh_recovery.integration import cmd_verify_integration

            result = cmd_verify_integration(paths, seed=args.seed, device=args.device)
        elif cmd == "train-bootstrap":
            result = stages.cmd_train_bootstrap(paths, seed=args.seed, device=args.device)
        elif cmd == "refresh-lists":
            result = stages.cmd_refresh_lists(paths, seed=args.seed, device=args.device)
        elif cmd == "train-c1":
            result = stages.cmd_train_c1(paths, seed=args.seed, arm=args.arm, device=args.device)
        elif cmd == "select-c1":
            result = stages.cmd_select(paths, seed=args.seed, pair=args.pair, phase="C1")
        elif cmd == "build-c2-graphs":
            result = stages.cmd_build_c2_graphs(paths, seed=args.seed, pair=args.pair, device=args.device)
        elif cmd == "train-c2":
            result = stages.cmd_train_c2(paths, seed=args.seed, arm=args.arm, device=args.device)
        elif cmd == "select-c2":
            result = stages.cmd_select(paths, seed=args.seed, pair=args.pair, phase="C2")
        elif cmd == "mine-qt-hard":
            result = stages.cmd_mine_hard(paths, seed=args.seed, device=args.device)
        elif cmd == "train-qt-teacher":
            result = stages.cmd_train_qt_teacher(paths, seed=args.seed, device=args.device)
        elif cmd == "train-qt-teacher-from-boot":
            result = stages.cmd_train_qt_teacher_from_boot(paths, seed=args.seed, device=args.device)
        elif cmd == "train-qt-cont":
            result = stages.cmd_train_qt_cont_teacher(paths, seed=args.seed, device=args.device)
        elif cmd == "train-path-teacher":
            result = stages.cmd_train_path_teacher(paths, seed=args.seed, device=args.device)
        elif cmd == "freeze-models":
            result = stages.cmd_freeze_models(paths, seed=args.seed)
        elif cmd == "evaluate-dev":
            result = final_eval.cmd_evaluate(paths, seed=args.seed, split="dev", device=args.device)
        elif cmd == "evaluate-test":
            lock = paths.work_dir / f"seed{args.seed}" / "MODEL_LOCK.json"
            if not lock.exists():
                raise RuntimeError(f"test firewall: seed{args.seed} MODEL_LOCK.json is required")
            result = final_eval.cmd_evaluate(paths, seed=args.seed, split="test", device=args.device)
        elif cmd == "measure-latency":
            result = final_eval.cmd_latency(paths, seed=args.seed, device=args.device)
        elif cmd == "run":
            from fresh_recovery.dag import run_dag

            result = run_dag(
                paths,
                seed=args.seed,
                gpus=[int(x) for x in args.gpus.split(",")],
                max_stage=args.max_stage,
                gpu_slots_per_device=args.gpu_slots_per_device,
            )
        elif cmd == "recompute-metrics":
            from fresh_recovery.recompute import recompute

            result = recompute(paths, seed=args.seed, split=args.split)
        elif cmd == "decide":
            result = final_eval.cmd_decide(paths, seed=args.seed)
        elif cmd == "compact-eval":
            from fresh_recovery.compact import convert_eval_dir

            result = {split: convert_eval_dir(paths.work_dir / f"seed{args.seed}" / f"EVAL_{split}") for split in ("DEV", "TEST")
                      if (paths.work_dir / f"seed{args.seed}" / f"EVAL_{split}").exists()}
        elif cmd == "write-report":
            from fresh_recovery.report import write_report

            result = {"report": str(write_report(paths, args.seed))}
        elif cmd == "package":
            from fresh_recovery.package import cmd_package

            result = cmd_package(paths, seed=args.seed)
        else:
            raise AssertionError(cmd)
    summary = {"command": cmd, "status": "complete", "elapsed_seconds": round(time.time() - started, 1), "result": result}
    text = json.dumps(summary, ensure_ascii=False, default=str)
    print(text if len(text) < 20000 else json.dumps({k: v for k, v in summary.items() if k != "result"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Stage-2 main flow: Stage-1 C30 -> evidence-reading column selector -> 9B value recovery -> B+IDF.

Every command takes ``--run-root``; the run is described by ``<run-root>/config.json``, written
once by ``init`` from ``configs/mmdd_stage2_bidf.json``. Stage 2 reads only the top
``candidate_scope`` (30) Stage-1 targets; Stage-1 ranks 31..50 are appended unchanged.

    init --stage1-handoff H --dataset-root D   bind inputs (H = run_stage1.py export output)
    catalog                 dataset -> catalog.sqlite + population/<split>.json           (CPU)
    jobs                    selector pairs + train column labels (C30 only)              (CPU)
    features --split S      frozen Qwen reader states, train views 0/1, dev/test view 0  (GPU)
    train-head              fresh selector head, fixed epoch count                        (CPU)
    plans                   head logits -> scheduled (target, column) recovery views      (CPU)
    recover                 9B ROW1 recovery with RAEA crops -> bridges                   (GPU)
    score                   D-0.98 bridge scores + C30 visible IDF -> rankings            (CPU)
    evaluate                metrics and source-group bootstrap vs Stage 1                 (CPU)

GPU commands take ``--gpu N`` (physical index) and set CUDA_VISIBLE_DEVICES before torch loads.
Recovery and feature extraction resume from the files they already wrote.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "mmdd_stage2_bidf.json"
GPU_COMMANDS = {"features", "recover"}


def init(run: Path, args: argparse.Namespace) -> None:
    path = run / "config.json"
    if path.exists():
        raise FileExistsError(f"run already initialised: {path}")
    config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    for key in ("dataset_root", "stage1_handoff", "qwen_model", "minilm_model"):
        value = getattr(args, key) or config["paths"][key]
        config["paths"][key] = str((ROOT / value).resolve() if not Path(value).is_absolute() else Path(value))
    config["paths"]["run_root"] = str(run)
    if args.no_selector_evidence:
        config["selector_reads_evidence"] = False  # arm A ablation: the selector reads Q and T only
    for key in ("dataset_root", "stage1_handoff", "qwen_model", "minilm_model"):
        if not Path(config["paths"][key]).exists():
            raise FileNotFoundError(f"{key}: {config['paths'][key]}")
    run.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(path), "paths": config["paths"],
                      "selector_reads_evidence": config["selector_reads_evidence"]}, indent=2), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "catalog", "jobs", "features", "train-head", "plans", "recover", "score", "evaluate"):
        subparsers.add_parser(command).add_argument("--run-root", type=Path, required=True)
    init_parser = subparsers.choices["init"]
    init_parser.add_argument("--stage1-handoff", required=True, help="directory with retrieval.{train,dev,test}.jsonl")
    init_parser.add_argument("--dataset-root")
    init_parser.add_argument("--qwen-model")
    init_parser.add_argument("--minilm-model")
    init_parser.add_argument("--no-selector-evidence", action="store_true",
                             help="ablation: the selector reader does not see the natural evidence")
    subparsers.choices["features"].add_argument("--split", choices=("train", "dev", "test"), required=True)
    for command in GPU_COMMANDS:
        subparsers.choices[command].add_argument("--gpu", type=int, required=True, help="physical GPU index")
    args = parser.parse_args(argv)
    run = args.run_root.resolve()
    if args.command == "init":
        init(run, args)
        return
    if args.command in GPU_COMMANDS:
        # Must happen before the first torch import in this process.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if (run / "EXPERIMENT.json").exists():
        from mmdd_stage2.experiments import verify
        verify(run)

    if args.command == "catalog":
        from mmdd_stage2.catalog import build_catalog
        build_catalog(Path(config["paths"]["dataset_root"]), run)
    elif args.command == "jobs":
        from mmdd_stage2.jobs import build_jobs
        build_jobs(config, run)
    elif args.command == "features":
        from mmdd_stage2.reader import extract_features
        extract_features(config, run, args.split)
    elif args.command == "train-head":
        from mmdd_stage2.selector import train_head
        train_head(config, run)
    elif args.command == "plans":
        from mmdd_stage2.selector import make_plans
        make_plans(config, run)
    elif args.command == "recover":
        from mmdd_stage2.recovery import run_recovery
        run_recovery(config, run)
    elif args.command == "score":
        from mmdd_stage2.rerank import run_scoring
        run_scoring(config, run)
    else:
        from mmdd_stage2.evaluate import evaluate
        evaluate(config, run)


if __name__ == "__main__":
    main()

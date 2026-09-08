#!/usr/bin/env python
"""Freeze completed R10 Students and run the Stage-1-only closing evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from mmdd_stage1.experiment_process import run_logged_subprocess
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.selection import load_stage1_selection, write_json


def freeze(r10: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Do not overwrite the frozen evaluation protocol")
    entries = {}
    base_rules = ["f0_direct", "f1_evidence", "f2_rrf_e005", "f3_rrf_equal", "f5_reserved_half"]
    for seed in (13, 17, 23):
        for arm in ("e01", "p_frozen"):
            if seed == 13:
                parent = r10 / ("taskE_matched/t0/e01" if arm == "e01" else "taskE_student_controls/p_frozen")
            else:
                parent = r10 / "stage1_closeout" / f"seed_{seed}" / arm
            entries[f"{arm}_s{seed}"] = {
                "selection": str(parent / "student_path.pt.selection.json"),
                "seed": seed, "arm": arm, "system": "student",
                "rules": [*base_rules, *(["f4_lambda_0.5"] if arm == "p_frozen" else [])],
                "role": "fixed-Teacher Student seed repeat",
            }
    references = {
        "raw": ("taskA_protocol/baselines/pca_epoch0", "raw"),
        "pca_epoch0": ("taskA_protocol/baselines/pca_epoch0", "student"),
        "r5_repro": ("taskA_protocol/baselines/r5_repro", "student"),
        "c3a": ("taskC_pr_ablation/c3a_p1e-6_r1e-5", "student"),
        "c4": ("taskC_pr_ablation/c4_r_warmup_then_p1e-6_r1e-5/phase2_unfrozen", "student"),
    }
    for name, (parent, system) in references.items():
        entries[name] = {
            "selection": str(r10 / parent / "student_path.pt.selection.json"),
            "seed": 13, "arm": name, "system": system, "rules": base_rules,
            "role": "single-seed reference; no whole-chain variance claim",
        }
    for entry in entries.values():
        selection = load_stage1_selection(Path(entry["selection"]))
        if selection["selection_split"] != "dev":
            raise ValueError("Every closing checkpoint must be dev-selected")
        sha = checkpoint_fingerprint(Path(selection["best_checkpoint"]))
        if sha != selection["best_checkpoint_sha256"]:
            raise ValueError("Selection no longer matches checkpoint")
        entry.update(checkpoint_sha256=sha, best_epoch=selection["best_epoch"])
    write_json(output, {
        "format_version": 1,
        "frozen_before_test": True,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scope": "EntiTables v9 Stage-1 only; Stage-2 stopped by user",
        "selection_rule": "dev implicit ValidPath@10,4 then RowSupport then recall; no test selection",
        "primary_contrasts": [
            "p_frozen F4(0.5) minus p_frozen direct",
            "p_frozen RRF(0.05) minus E01 RRF(0.05)",
            "p_frozen F4(0.5) minus r5-repro RRF(0.05)",
        ],
        "seed_scope": "Student edge/path updates conditional on fixed seed-13 Teacher, C4 initialization and mining pool",
        "features": str(r10 / "features_qwen3_vl_embedding_8b"),
        "corpus": str(r10 / "stage1_data/stage1_corpus.jsonl"),
        "data": {split: str(r10 / f"taskD_edge_labels/lists/target_lists.{split}.jsonl") for split in ("dev", "test")},
        "recoveries": [str(r10.parent.parent / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9/evidence_recoveries/part-00000.jsonl")],
        "recall_ks": [10, 20, 50],
        "retrieval_budget": {"direct_k": 100, "evidence_k_per_modality": 20, "targets_per_evidence": 20, "evidence_types": ["text", "image"]},
        "bootstrap": {"unit": "source_table_id", "replicates": 10000, "seed": 13},
        "entries": entries,
    })
    print(f"Frozen {len(entries)} entries: {output}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = args.protocol.parent
    for name in args.entries:
        entry = protocol["entries"][name]
        for split in ("dev", "test"):
            pool = out / "path_pools" / f"{name}_{split}.jsonl"
            metrics = out / "evaluation" / f"{name}_{split}.json"
            if not pool.with_suffix(".jsonl.metadata.json").is_file():
                run_logged_subprocess([
                    sys.executable, str(root / "src/export_stage1_path_pool.py"),
                    "--selection", entry["selection"], "--system", entry["system"],
                    "--features", protocol["features"], "--corpus", protocol["corpus"],
                    "--data", protocol["data"][split], "--split", split,
                    "--output", str(pool), "--device", "cuda:0",
                ], out / "logs" / f"export_{name}_{split}.log", root=root)
            if metrics.is_file():
                previous = json.loads(metrics.read_text(encoding="utf-8"))
                if previous["protocol_sha256"] != checkpoint_fingerprint(args.protocol):
                    raise ValueError("Frozen protocol changed after evaluation")
                continue
            run_logged_subprocess([
                sys.executable, str(root / "src/evaluate_stage1_r10_frozen.py"),
                "--protocol", str(args.protocol), "--entry", name,
                "--path-pool", str(pool), "--output", str(metrics), "--device", "cuda:0",
            ], out / "logs" / f"evaluate_{name}_{split}.log", root=root)
    print("Requested Stage-1 closing evaluations complete", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--r10-root", type=Path, default=root / "work/stage1_optimization_r10_20260907")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--entries", nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.r10_root.resolve(), args.protocol.resolve())
    else:
        if not args.entries:
            parser.error("--entries is required for evaluate")
        evaluate(args)

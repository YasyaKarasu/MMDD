#!/usr/bin/env python
"""Run the corrected R12 Task F candidate-column prerequisite and full chain."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from mmdd_stage2.checkpoints import (
    load_candidate_scorer,
    load_candidate_scorer_metadata,
)
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.r12_column import (
    build_r12_reader_cache,
    load_r12_column_examples,
    load_r12_reader_cache,
    train_r12_candidate_scorer,
    write_r12_evaluation,
)
from mmdd_stage2.r12_task_f import (
    load_task_f_inputs,
    prepare_task_f_human_audit,
    run_fa_diagnostics,
    run_full_chain,
    summarize_task_f,
)
from mmdd_stage2.routing import SimilarityEvidenceRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9"
)
DEFAULT_R12_ROOT = PROJECT_ROOT / "work/stage1_optimization_r12_20260908"
DEFAULT_MODEL = PROJECT_ROOT / "hf_models/Qwen3.5-9B"
PROTOCOL_SPLITS = ("train_fit", "cal_fit", "cal_check")


def target_list_path(r12_root: Path, protocol_split: str) -> Path:
    return (
        r12_root
        / "taskA_correctness/supervision"
        / f"target_lists.{protocol_split}.jsonl"
    )


def column_root(r12_root: Path) -> Path:
    return r12_root / "taskF_end_to_end/column_scorer"


def run_cache_column(args: argparse.Namespace) -> dict:
    target_lists = {
        split: target_list_path(args.r12_root, split) for split in args.protocol_split
    }
    examples, objects, audit = load_r12_column_examples(
        args.dataset_root,
        target_lists,
        top_k_evidence=args.top_k_evidence,
    )
    backend = QwenStage2Backend(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    manifests = []
    for split in args.protocol_split:
        subset = [item for item in examples if item.protocol_split == split]
        manifests.append(
            build_r12_reader_cache(
                backend,
                subset,
                objects,
                column_root(args.r12_root) / "reader_cache" / split,
                model_path=args.model_dir,
                model_dtype=args.dtype,
                top_k_evidence=args.top_k_evidence,
                column_permutation_seed=args.column_permutation_seed,
                shard_size=args.shard_size,
                reverse_shards=args.reverse_shards,
            )
        )
    audit_path = column_root(args.r12_root) / (
        "data_audit." + ".".join(args.protocol_split) + ".json"
    )
    write_json(audit_path, audit)
    result = {
        "audit": str(audit_path.resolve()),
        "manifests": [
            str(
                (
                    column_root(args.r12_root)
                    / "reader_cache"
                    / split
                    / "manifest.json"
                ).resolve()
            )
            for split in args.protocol_split
        ],
    }
    print(json.dumps(result, indent=2))
    return result


def _cache_dirs(r12_root: Path) -> list[Path]:
    return [column_root(r12_root) / "reader_cache" / split for split in PROTOCOL_SPLITS]


def run_train_column(args: argparse.Namespace) -> dict:
    records, cache_fingerprint = load_r12_reader_cache(_cache_dirs(args.r12_root))
    train_records = [
        record for record in records if record["protocol_split"] == "train_fit"
    ]
    cal_fit_records = [
        record for record in records if record["protocol_split"] == "cal_fit"
    ]
    hidden_dim = int(train_records[0]["open_states"].shape[1])
    scorer_dir = column_root(args.r12_root) / "seed_13"
    scorer, history = train_r12_candidate_scorer(
        train_records,
        cal_fit_records,
        hidden_dim=hidden_dim,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=scorer_dir,
        checkpoint_metadata={
            "model_dir": str(args.model_dir.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "reader_cache_fingerprint": cache_fingerprint,
            "column_permutation_seed": args.column_permutation_seed,
            "training_protocol_split": "train_fit",
            "threshold_protocol_split": "cal_fit",
            "task_f_dev_queries_used": False,
        },
    )
    del scorer
    result = {
        "checkpoint": str((scorer_dir / "candidate.pt").resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(scorer_dir / "candidate.pt"),
        "cache_fingerprint": cache_fingerprint,
        **history,
    }
    print(json.dumps(result, indent=2))
    return result


def run_evaluate_column(args: argparse.Namespace) -> dict:
    records, cache_fingerprint = load_r12_reader_cache(_cache_dirs(args.r12_root))
    cal_check = [
        record for record in records if record["protocol_split"] == "cal_check"
    ]
    checkpoint = column_root(args.r12_root) / "seed_13/candidate.pt"
    metadata = load_candidate_scorer_metadata(checkpoint)
    if metadata.get("reader_cache_fingerprint") != cache_fingerprint:
        raise ValueError("R12 candidate checkpoint and reader cache differ")
    if int(metadata.get("column_permutation_seed", -1)) != args.column_permutation_seed:
        raise ValueError("R12 candidate checkpoint uses another column permutation")
    scorer = load_candidate_scorer(
        checkpoint,
        torch.device("cpu"),
        expected_model_dir=args.model_dir,
    )
    output = column_root(args.r12_root) / "cal_check_metrics.json"
    result = write_r12_evaluation(
        output,
        scorer,
        cal_check,
        threshold=float(metadata["rejection_threshold"]),
    )
    positive_positions = Counter(
        int(record["gold_column_position"])
        for record in cal_check
        if record["example_type"] == "positive"
    )
    majority_position, majority_count = min(
        positive_positions.items(), key=lambda item: (-item[1], item[0])
    )
    result["protocol"] = {
        "training_split": "train_fit",
        "threshold_split": "cal_fit",
        "evaluation_split": "cal_check",
        "task_f_dev_queries_used": False,
        "column_permutation_seed": args.column_permutation_seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "reader_cache_fingerprint": cache_fingerprint,
    }
    result["majority_position_baseline"] = {
        "position": majority_position,
        "correct": majority_count,
        "examples": sum(positive_positions.values()),
        "accuracy": majority_count / sum(positive_positions.values()),
    }
    result["identifiability"] = (
        "position_permuted_with_control_rejection_reported"
        if len(positive_positions) > 1
        else "confounded_by_degenerate_gold_column_position"
    )
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def run_end_to_end(args: argparse.Namespace) -> dict:
    scorer_root = column_root(args.r12_root)
    scorer_checkpoint = scorer_root / "seed_13/candidate.pt"
    scorer_metrics = scorer_root / "cal_check_metrics.json"
    if not scorer_metrics.is_file():
        raise FileNotFoundError("Run evaluate-column before the end-to-end experiment")
    column_evaluation = json.loads(scorer_metrics.read_text(encoding="utf-8"))
    if column_evaluation.get("identifiability") != "position_permuted_with_control_rejection_reported":
        raise ValueError("R12 candidate-column evaluation remains position-confounded")
    metadata = load_candidate_scorer_metadata(scorer_checkpoint)
    task_f_root = args.r12_root / "taskF_end_to_end"
    path_pool = (
        args.r12_root
        / "taskC_training/selected_c2_seed13/path_pools/student_mixed_dev.jsonl"
    )
    admission = (
        args.r12_root
        / "taskE_admission/student_selected_mixed_d1/metrics.json"
    )
    inputs = load_task_f_inputs(
        args.dataset_root,
        task_f_root / "queries_frozen.json",
        args.r12_root / "taskA_correctness/supervision/target_lists.dev.jsonl",
        path_pool,
        admission,
    )
    inputs.manifest["column_scorer"] = {
        "checkpoint": str(scorer_checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(scorer_checkpoint),
        "cal_check_metrics": str(scorer_metrics.resolve()),
        "cal_check_metrics_sha256": checkpoint_fingerprint(scorer_metrics),
        "rejection_threshold": float(metadata["rejection_threshold"]),
        "column_permutation_seed": int(metadata["column_permutation_seed"]),
    }
    inputs.manifest["execution"] = {
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": checkpoint_fingerprint(args.model_dir / "config.json"),
        "dtype": args.dtype,
        "focus_start_layer": args.focus_start_layer,
        "top_k_evidence": args.top_k_evidence,
        "recovery_budget": args.recovery_budget,
        "max_text_evidence_tokens": args.max_text_evidence_tokens,
        "text_overlap_tokens": args.text_overlap_tokens,
        "max_span_tokens": args.max_span_tokens,
        "roi_candidates": args.roi_candidates,
        "embedding_batch_size": args.embedding_batch_size,
        "max_embedding_tokens": args.max_embedding_tokens,
        "similarity_batch_size": args.similarity_batch_size,
        "similarity_threshold": args.similarity_threshold,
        "min_row_coverage": args.min_row_coverage,
        "audit_failure_fraction": args.audit_failure_fraction,
        "runner_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    write_json(task_f_root / "input_manifest.json", inputs.manifest)

    scorer = load_candidate_scorer(
        scorer_checkpoint,
        torch.device("cpu"),
        expected_model_dir=args.model_dir,
    )
    backend = QwenStage2Backend(
        args.model_dir,
        device=args.device,
        dtype=args.dtype,
        focus_start_layer=args.focus_start_layer,
        max_text_evidence_tokens=args.max_text_evidence_tokens,
        text_overlap_tokens=args.text_overlap_tokens,
        max_span_tokens=args.max_span_tokens,
        roi_candidates=args.roi_candidates,
        embedding_batch_size=args.embedding_batch_size,
        max_embedding_tokens=args.max_embedding_tokens,
    )
    scorer.to(backend.device)
    pool_metadata = json.loads(
        path_pool.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    router = SimilarityEvidenceRouter(
        FeatureStore.from_path(Path(pool_metadata["features"]), cache_size=120_000)
    )
    verifier = Stage2Verifier(
        backend,
        scorer,
        evidence_router=router,
        similarity_threshold=args.similarity_threshold,
        min_row_coverage=args.min_row_coverage,
        similarity_batch_size=args.similarity_batch_size,
        column_permutation_seed=int(metadata["column_permutation_seed"]),
        column_rejection_threshold=float(metadata["rejection_threshold"]),
    )
    fa_path = task_f_root / "f_a_predictions.jsonl"
    full_path = task_f_root / "full_chain_predictions.jsonl"
    if args.phase in {"fa", "both"}:
        run_fa_diagnostics(
            verifier,
            inputs,
            fa_path,
            seed=13,
            top_k_evidence=args.top_k_evidence,
        )
    if args.phase in {"full", "both"}:
        run_full_chain(
            verifier,
            inputs,
            full_path,
            recovery_budget=args.recovery_budget,
            top_k_evidence=args.top_k_evidence,
        )
    if fa_path.is_file() and full_path.is_file():
        result = summarize_task_f(
            inputs,
            backend,
            fa_path=fa_path,
            full_chain_path=full_path,
            similarity_threshold=args.similarity_threshold,
        )
        result["prediction_artifacts"] = {
            "f_a": str(fa_path.resolve()),
            "f_a_sha256": checkpoint_fingerprint(fa_path),
            "full_chain": str(full_path.resolve()),
            "full_chain_sha256": checkpoint_fingerprint(full_path),
        }
        result["human_audit"] = prepare_task_f_human_audit(
            inputs,
            backend,
            fa_path=fa_path,
            full_chain_path=full_path,
            output_dir=task_f_root / "human_audit",
            failure_fraction=args.audit_failure_fraction,
            seed=13,
            similarity_threshold=args.similarity_threshold,
        )
        write_json(task_f_root / "metrics.json", result)
        write_json(
            task_f_root / "STATUS.json",
            {
                "format_version": 1,
                "status": result["status"],
                "metrics": str((task_f_root / "metrics.json").resolve()),
                "human_audit_required": True,
                "reason": result["limitations"][0],
            },
        )
    else:
        result = {
            "status": "partial",
            "phase": args.phase,
            "fa_exists": fa_path.is_file(),
            "full_exists": full_path.is_file(),
        }
    print(json.dumps(result, indent=2))
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r12-root", type=Path, default=DEFAULT_R12_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--column-permutation-seed", type=int, default=13)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache-column")
    _common(cache)
    cache.add_argument(
        "--protocol-split",
        nargs="+",
        choices=PROTOCOL_SPLITS,
        required=True,
    )
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    cache.add_argument("--top-k-evidence", type=int, default=4)
    cache.add_argument("--shard-size", type=int, default=32)
    cache.add_argument("--reverse-shards", action="store_true")
    cache.set_defaults(function=run_cache_column)

    train = subparsers.add_parser("train-column")
    _common(train)
    train.add_argument("--seed", type=int, default=13)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.set_defaults(function=run_train_column)

    evaluate = subparsers.add_parser("evaluate-column")
    _common(evaluate)
    evaluate.set_defaults(function=run_evaluate_column)

    end_to_end = subparsers.add_parser("run-end-to-end")
    _common(end_to_end)
    end_to_end.add_argument("--phase", choices=("fa", "full", "both"), default="both")
    end_to_end.add_argument("--device", default="cuda:0")
    end_to_end.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    end_to_end.add_argument("--focus-start-layer", type=int, default=14)
    end_to_end.add_argument("--top-k-evidence", type=int, default=4)
    end_to_end.add_argument("--recovery-budget", type=int, default=10)
    end_to_end.add_argument("--max-text-evidence-tokens", type=int, default=1024)
    end_to_end.add_argument("--text-overlap-tokens", type=int, default=128)
    end_to_end.add_argument("--max-span-tokens", type=int, default=192)
    end_to_end.add_argument("--roi-candidates", type=int, default=4)
    end_to_end.add_argument("--embedding-batch-size", type=int, default=64)
    end_to_end.add_argument("--max-embedding-tokens", type=int, default=128)
    end_to_end.add_argument("--similarity-batch-size", type=int, default=1024)
    end_to_end.add_argument("--similarity-threshold", type=float, default=0.8)
    end_to_end.add_argument("--min-row-coverage", type=float, default=0.6)
    end_to_end.add_argument("--audit-failure-fraction", type=float, default=0.1)
    end_to_end.set_defaults(function=run_end_to_end)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    arguments = parse_args(argv)
    started = time.monotonic()
    result = arguments.function(arguments)
    runs_path = arguments.r12_root / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": f"Task F {arguments.command}",
                    "status": str(result.get("status", "complete")),
                    "command": [sys.executable, *sys.argv],
                    "elapsed_seconds": time.monotonic() - started,
                    "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return result


if __name__ == "__main__":
    main()

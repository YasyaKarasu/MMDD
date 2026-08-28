#!/usr/bin/env python
"""Run resumable multi-round Stage-1 mining, mixed training, and dev selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import cache_stage1_features
import refresh_stage1_hard_negatives
import train_stage1
from cache_stage1_features import teacher_object_ids
from mmdd_stage1.checkpoints import load_path_aggregation, load_student
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    checkpoint_fingerprint,
    load_corpus_ids,
)
from mmdd_stage1.selection import MetricGate, load_stage1_selection, write_json
from mmdd_stage1.workflow import (
    RoundStep,
    validate_round_index,
    workflow_fingerprint,
)


def _selection_checkpoint(selection: dict[str, Any]) -> Path:
    if selection.get("completed_stage") != "student-path":
        raise ValueError("Mining rounds require a dev-gated student-path checkpoint")
    if selection.get("selection_split") != "dev":
        raise ValueError("Mining rounds require a checkpoint selected only on dev")
    path = Path(str(selection["best_checkpoint"]))
    if checkpoint_fingerprint(path) != selection.get("best_checkpoint_sha256"):
        raise ValueError("Selected Student checkpoint fingerprint does not match")
    return path


def _step(
    round_dir: Path,
    name: str,
    config: dict[str, Any],
    inputs: list[Path],
) -> RoundStep:
    return RoundStep(
        round_dir / "steps" / f"{name}.json",
        workflow_fingerprint(config, inputs),
    )


def _teacher_features_ready(features: Path, data_paths: list[Path]) -> bool:
    store = FeatureStore.from_path(features)
    for object_id in teacher_object_ids(data_paths, split="train"):
        try:
            if store.get(object_id, include_hidden=True).hidden_states is None:
                return False
        except (KeyError, ValueError, FileNotFoundError):
            return False
    return True


def _refresh_args(
    args: argparse.Namespace,
    *,
    student_checkpoint: Path,
    index_dir: Path,
    output_targets: Path,
    output_edges: Path,
    mining_round: int,
    mine_only: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        features=args.features,
        teacher_checkpoint=None if mine_only else args.teacher_checkpoint,
        student_checkpoint=str(student_checkpoint),
        index_dir=str(index_dir),
        corpus=args.corpus,
        target_lists=args.base_path_data,
        output_target_lists=str(output_targets),
        output_edge_lists=str(output_edges),
        split="train",
        device=args.device,
        feature_cache_size=args.feature_cache_size,
        teacher_batch_size=args.teacher_batch_size,
        mine_only=mine_only,
        mining_round=mining_round,
        hard_targets_per_query=args.hard_targets_per_query,
        hard_evidence_per_type=args.hard_evidence_per_type,
        hard_paths_per_query=args.hard_paths_per_query,
        direct_k=args.mining_direct_k,
        evidence_k=args.mining_evidence_k,
        targets_per_evidence=args.mining_targets_per_evidence,
        evidence_types=args.evidence_types,
        evidence_aggregation=args.evidence_aggregation,
        evidence_top_k=args.evidence_top_k,
    )


def _train_args(
    args: argparse.Namespace,
    *,
    stage: str,
    base_data: list[str],
    hard_data: Path,
    dev_data: list[str],
    student_checkpoint: Path,
    hard_source_checkpoint: Path,
    output: Path,
    index_root: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        stage=stage,
        features=args.features,
        base_data=base_data,
        hard_data=[str(hard_data)],
        train_data=None,
        dev_data=dev_data,
        output=str(output),
        teacher_checkpoint=args.teacher_checkpoint,
        student_checkpoint=str(student_checkpoint),
        hard_source_checkpoint=str(hard_source_checkpoint),
        corpus=args.corpus if stage == "student-path" else None,
        index_root=str(index_root) if index_root is not None else None,
        split="train",
        dev_split="dev",
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hard_learning_rate=args.hard_learning_rate,
        hard_fraction=args.hard_fraction,
        weight_decay=args.weight_decay,
        seed=args.seed,
        feature_cache_size=args.feature_cache_size,
        dataset_sampling_alpha=args.dataset_sampling_alpha,
        primary_metric=args.primary_metric,
        min_delta=args.min_delta,
        patience=args.patience,
        min_dev_evidence_path_queries=args.min_dev_evidence_path_queries,
        min_dev_evidence_path_coverage=args.min_dev_evidence_path_coverage,
        teacher_dim=512,
        teacher_heads=8,
        teacher_layers=3,
        text_latents=16,
        image_latents=24,
        dropout=0.1,
        student_dim=128,
        temperature=args.temperature,
        distillation_weight=args.distillation_weight,
        evidence_aggregation=args.evidence_aggregation,
        evidence_top_k=args.evidence_top_k,
        direct_k=args.dev_direct_k,
        evidence_k=args.dev_evidence_k,
        targets_per_evidence=args.dev_targets_per_evidence,
        evidence_types=args.evidence_types,
        rrf_k=args.rrf_k,
        index_batch_size=args.index_batch_size,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )


def _training_outputs(checkpoint_path: Path) -> list[Path]:
    paths = train_stage1.CheckpointManager(checkpoint_path).paths
    return [paths["best"], paths["last"], paths["history"], paths["selection"]]


def _evaluate_final_test(
    args: argparse.Namespace,
    selection: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    validate_protocol_split("final_test", "test")
    checkpoint_path = _selection_checkpoint(selection)
    checkpoint_sha256 = checkpoint_fingerprint(checkpoint_path)
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    metrics_path = output_dir / "final_test_metrics.json"
    index_dir = output_dir / "final_test_index"
    step = _step(
        output_dir,
        "final_test",
        {
            "split": "test",
            "checkpoint_sha256": checkpoint_sha256,
            "retrieval": {
                "direct_k": args.dev_direct_k,
                "evidence_k": args.dev_evidence_k,
                "targets_per_evidence": args.dev_targets_per_evidence,
                "rrf_k": args.rrf_k,
            },
        },
        [checkpoint_path, corpus_path, *map(Path, args.test_data)],
    )
    if step.completed():
        validate_round_index(
            index_dir,
            student_checkpoint_sha256=checkpoint_sha256,
            corpus_sha256=corpus_sha256,
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    student = load_student(checkpoint_path, device)
    build_indices(
        student,
        store,
        load_corpus_ids(corpus_path, store),
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    indices = StudentANNIndices(
        student,
        store,
        index_dir,
        device=device,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    test_examples = [
        example
        for path in map(Path, args.test_data)
        for example in load_target_examples(path, split="test", dataset_name=path.stem)
    ]
    aggregation, top_k = load_path_aggregation(checkpoint_path)
    metrics = evaluate_student_retrieval(
        test_examples,
        indices,
        direct_k=args.dev_direct_k,
        evidence_k=args.dev_evidence_k,
        targets_per_evidence=args.dev_targets_per_evidence,
        evidence_types=tuple(args.evidence_types),
        evidence_aggregation=aggregation,
        evidence_top_k=top_k,
        rrf_k=args.rrf_k,
    )
    write_json(metrics_path, metrics)
    step.complete([metrics_path, index_dir / "manifest.json"])
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_mining_rounds <= 0:
        raise ValueError("--max-mining-rounds must be positive")
    if args.round_patience < 0:
        raise ValueError("--round-patience must be non-negative")
    if args.hard_learning_rate <= 0 or args.hard_learning_rate >= args.learning_rate:
        raise ValueError(
            "--hard-learning-rate must be positive and lower than --learning-rate"
        )
    validate_protocol_split("training", "train")
    validate_protocol_split("mining", "train")
    validate_protocol_split("dev_gate", "dev")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    teacher_path = Path(args.teacher_checkpoint)
    initial_selection = load_stage1_selection(Path(args.initial_selection))
    previous_checkpoint = _selection_checkpoint(initial_selection)
    if initial_selection.get("corpus_sha256") != corpus_sha256:
        raise ValueError(
            "Initial Student selection was evaluated against a different corpus"
        )
    global_selection = initial_selection
    selector = MetricGate(
        args.primary_metric,
        min_delta=args.round_min_delta,
        patience=args.round_patience,
    )
    selector.observe(0, initial_selection["best_metrics"])
    round_history = [
        {
            "round": 0,
            "epoch": initial_selection["best_epoch"],
            "checkpoint": str(previous_checkpoint.resolve()),
            "metrics": initial_selection["best_metrics"],
            "improved": True,
        }
    ]
    stop_reason = "max_mining_rounds"

    for mining_round in range(1, args.max_mining_rounds + 1):
        round_dir = output_dir / f"round_{mining_round:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        previous_sha256 = checkpoint_fingerprint(previous_checkpoint)

        index_dir = round_dir / "mining_index"
        index_step = _step(
            round_dir,
            "index",
            {
                "round": mining_round,
                "student_checkpoint_sha256": previous_sha256,
                "corpus_sha256": corpus_sha256,
                "hnsw_m": args.hnsw_m,
                "ef_construction": args.ef_construction,
                "ef_search": args.ef_search,
            },
            [previous_checkpoint, corpus_path],
        )
        if not index_step.completed():
            if (index_dir / "manifest.json").is_file():
                validate_round_index(
                    index_dir,
                    student_checkpoint_sha256=previous_sha256,
                    corpus_sha256=corpus_sha256,
                )
            else:
                device = torch.device(
                    args.device
                    if args.device != "auto"
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                store = FeatureStore.from_path(
                    Path(args.features), cache_size=args.feature_cache_size
                )
                student = load_student(previous_checkpoint, device)
                build_indices(
                    student,
                    store,
                    load_corpus_ids(corpus_path, store),
                    index_dir,
                    device=device,
                    checkpoint_sha256=previous_sha256,
                    corpus_sha256=corpus_sha256,
                    batch_size=args.index_batch_size,
                    m=args.hnsw_m,
                    ef_construction=args.ef_construction,
                    ef_search=args.ef_search,
                )
            index_step.complete([index_dir / "manifest.json"])
        validate_round_index(
            index_dir,
            student_checkpoint_sha256=previous_sha256,
            corpus_sha256=corpus_sha256,
        )

        pending_targets = round_dir / "hard_targets.pending.jsonl"
        pending_edges = round_dir / "hard_edges.pending.jsonl"
        mining_config = {
            "round": mining_round,
            "hard_targets_per_query": args.hard_targets_per_query,
            "hard_evidence_per_type": args.hard_evidence_per_type,
            "hard_paths_per_query": args.hard_paths_per_query,
            "direct_k": args.mining_direct_k,
            "evidence_k": args.mining_evidence_k,
            "targets_per_evidence": args.mining_targets_per_evidence,
        }
        mine_step = _step(
            round_dir,
            "mine",
            mining_config,
            [previous_checkpoint, index_dir / "manifest.json", *map(Path, args.base_path_data)],
        )
        if not mine_step.completed():
            refresh_stage1_hard_negatives.run(
                _refresh_args(
                    args,
                    student_checkpoint=previous_checkpoint,
                    index_dir=index_dir,
                    output_targets=pending_targets,
                    output_edges=pending_edges,
                    mining_round=mining_round,
                    mine_only=True,
                )
            )
            mine_step.complete(
                [
                    pending_targets,
                    pending_edges,
                    pending_targets.with_suffix(".jsonl.metadata.json"),
                    pending_edges.with_suffix(".jsonl.metadata.json"),
                ]
            )

        supplement_step = _step(
            round_dir,
            "teacher_features",
            {"round": mining_round, "feature_cache": str(Path(args.features).resolve())},
            [pending_targets, pending_edges],
        )
        supplement_completed = supplement_step.completed()
        if not _teacher_features_ready(
            Path(args.features), [pending_targets, pending_edges]
        ):
            if args.objects is None or not Path(args.features).is_dir():
                raise ValueError(
                    "Mined objects are missing Teacher features; provide --objects and "
                    "use a two-tier feature directory, then resume"
                )
            cache_stage1_features.run(
                argparse.Namespace(
                    input_jsonl=args.objects,
                    output_dir=args.features,
                    model_dir=args.embedding_model_dir,
                    device=args.device,
                    dtype=args.embedding_dtype,
                    teacher_data=[str(pending_targets), str(pending_edges)],
                    teacher_split="train",
                    instruction=None,
                )
            )
        if not _teacher_features_ready(
            Path(args.features), [pending_targets, pending_edges]
        ):
            raise ValueError("Teacher feature supplementation did not complete")
        if not supplement_completed:
            supplement_step.complete([])

        hard_targets = round_dir / "hard_targets.jsonl"
        hard_edges = round_dir / "hard_edges.jsonl"
        score_step = _step(
            round_dir,
            "teacher_score",
            {
                **mining_config,
                "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_path),
            },
            [
                previous_checkpoint,
                teacher_path,
                index_dir / "manifest.json",
                pending_targets,
                pending_edges,
            ],
        )
        if not score_step.completed():
            refresh_stage1_hard_negatives.run(
                _refresh_args(
                    args,
                    student_checkpoint=previous_checkpoint,
                    index_dir=index_dir,
                    output_targets=hard_targets,
                    output_edges=hard_edges,
                    mining_round=mining_round,
                    mine_only=False,
                )
            )
            score_step.complete(
                [
                    hard_targets,
                    hard_edges,
                    hard_targets.with_suffix(".jsonl.metadata.json"),
                    hard_edges.with_suffix(".jsonl.metadata.json"),
                ]
            )

        edge_checkpoint = round_dir / "student_edge.pt"
        edge_inputs = [
            previous_checkpoint,
            teacher_path,
            hard_edges,
            hard_edges.with_suffix(".jsonl.metadata.json"),
            *map(Path, args.base_edge_data),
            *map(Path, args.dev_edge_data),
        ]
        edge_step = _step(
            round_dir,
            "student_edge",
            {
                "round": mining_round,
                "hard_fraction": args.hard_fraction,
                "learning_rate": args.hard_learning_rate,
                "epochs": args.epochs,
                "patience": args.patience,
            },
            edge_inputs,
        )
        edge_outputs = _training_outputs(edge_checkpoint)
        if not edge_step.completed():
            train_stage1.run(
                _train_args(
                    args,
                    stage="student-edge",
                    base_data=args.base_edge_data,
                    hard_data=hard_edges,
                    dev_data=args.dev_edge_data,
                    student_checkpoint=previous_checkpoint,
                    hard_source_checkpoint=previous_checkpoint,
                    output=edge_checkpoint,
                )
            )
            edge_step.complete(edge_outputs)

        path_checkpoint = round_dir / "student_path.pt"
        path_inputs = [
            edge_checkpoint,
            previous_checkpoint,
            teacher_path,
            hard_targets,
            hard_targets.with_suffix(".jsonl.metadata.json"),
            corpus_path,
            *map(Path, args.base_path_data),
            *map(Path, args.dev_path_data),
        ]
        path_step = _step(
            round_dir,
            "student_path",
            {
                "round": mining_round,
                "hard_fraction": args.hard_fraction,
                "learning_rate": args.hard_learning_rate,
                "epochs": args.epochs,
                "primary_metric": args.primary_metric,
                "min_delta": args.min_delta,
                "patience": args.patience,
            },
            path_inputs,
        )
        path_outputs = _training_outputs(path_checkpoint)
        if not path_step.completed():
            train_stage1.run(
                _train_args(
                    args,
                    stage="student-path",
                    base_data=args.base_path_data,
                    hard_data=hard_targets,
                    dev_data=args.dev_path_data,
                    student_checkpoint=edge_checkpoint,
                    hard_source_checkpoint=previous_checkpoint,
                    output=path_checkpoint,
                    index_root=round_dir / "dev_indices",
                )
            )
            path_step.complete(path_outputs)

        round_selection = load_stage1_selection(path_outputs[-1])
        previous_checkpoint = _selection_checkpoint(round_selection)
        decision = selector.observe(mining_round, round_selection["best_metrics"])
        round_history.append(
            {
                "round": mining_round,
                "epoch": round_selection["best_epoch"],
                "checkpoint": str(previous_checkpoint.resolve()),
                "metrics": round_selection["best_metrics"],
                "improved": decision.improved,
            }
        )
        if decision.improved:
            global_selection = round_selection
        if decision.should_stop:
            stop_reason = f"round_early_stopping_patience_{args.round_patience}"
            break

    selected_round = selector.best_epoch
    test_metrics = _evaluate_final_test(args, global_selection, output_dir)
    final_selection = {
        **global_selection,
        "selected_round": selected_round,
        "round_history": round_history,
        "round_stop_reason": stop_reason,
        "test_metrics": test_metrics,
    }
    final_path = output_dir / "final_selection.json"
    write_json(final_path, final_selection)
    summary = {
        "selected_round": selected_round,
        "selected_epoch": final_selection["best_epoch"],
        "checkpoint": final_selection["best_checkpoint"],
        "dev_metrics": final_selection["best_metrics"],
        "test_metrics": test_metrics,
        "stage2_allowed": final_selection["stage2_allowed"],
        "selection": str(final_path),
        "stop_reason": stop_reason,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--objects", help="Stage-1 objects JSONL, needed only to supplement Teacher features.")
    parser.add_argument("--corpus", required=True, help="One full shared data-lake corpus for dev and test.")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--initial-selection", required=True)
    parser.add_argument("--base-edge-data", nargs="+", required=True)
    parser.add_argument("--base-path-data", nargs="+", required=True)
    parser.add_argument("--dev-edge-data", nargs="+", required=True)
    parser.add_argument("--dev-path-data", nargs="+", required=True)
    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-mining-rounds", type=int, default=3)
    parser.add_argument("--round-patience", type=int, default=2)
    parser.add_argument("--round-min-delta", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hard-learning-rate", type=float, default=2e-5)
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--dataset-sampling-alpha", type=float, default=0.0)
    parser.add_argument("--primary-metric", default="recall@10")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-dev-evidence-path-queries", type=int, default=1)
    parser.add_argument("--min-dev-evidence-path-coverage", type=float, default=0.0)

    parser.add_argument("--hard-targets-per-query", type=int, default=16)
    parser.add_argument("--hard-evidence-per-type", type=int, default=16)
    parser.add_argument("--hard-paths-per-query", type=int, default=16)
    parser.add_argument("--mining-direct-k", type=int, default=200)
    parser.add_argument("--mining-evidence-k", type=int, default=100)
    parser.add_argument("--mining-targets-per-evidence", type=int, default=100)
    parser.add_argument("--dev-direct-k", type=int, default=100)
    parser.add_argument("--dev-evidence-k", type=int, default=50)
    parser.add_argument("--dev-targets-per-evidence", type=int, default=50)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument("--evidence-aggregation", choices=["logsumexp", "topk_mean", "topk_sum"])
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--embedding-model-dir", default="hf_models/Qwen3-VL-Embedding-8B")
    parser.add_argument("--embedding-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

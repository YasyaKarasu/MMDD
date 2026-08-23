#!/usr/bin/env python
"""Refresh Student hard target/evidence/path candidates and Teacher scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from mmdd_stage1.checkpoints import load_student, load_teacher
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.mining import retrieve_hard_candidate_sets, score_hard_candidate_sets
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import StudentANNIndices, checkpoint_fingerprint


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.hard_targets_per_query <= 0 or args.teacher_batch_size <= 0:
        raise ValueError("Hard-target and batch sizes must be positive")
    if min(args.hard_evidence_per_query, args.max_evidence_per_target) < 0:
        raise ValueError("Evidence limits must be non-negative")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split = None if args.split == "all" else args.split
    target_paths = [Path(value) for value in args.target_lists]
    examples = [
        example
        for path in target_paths
        for example in load_target_examples(
            path,
            split=split,
            max_evidence=args.max_evidence_per_target,
            dataset_name=path.stem,
        )
    ]
    store = FeatureStore.from_path(Path(args.features), cache_size=args.feature_cache_size)
    embedding_dim, hidden_dim = store.dimensions()
    if hidden_dim is None:
        raise ValueError("Hard-negative Teacher rescoring requires hidden_states")
    teacher = load_teacher(Path(args.teacher_checkpoint), device)
    teacher_sha256 = checkpoint_fingerprint(Path(args.teacher_checkpoint))
    student_path = Path(args.student_checkpoint)
    student = load_student(student_path, device)
    if teacher.input_dim != hidden_dim or student.input_dim != embedding_dim:
        raise ValueError("Checkpoint input dimensions do not match the feature cache")
    student_sha256 = checkpoint_fingerprint(student_path)
    indices = StudentANNIndices(
        student,
        store,
        Path(args.index_dir),
        device=device,
        checkpoint_sha256=student_sha256,
    )
    aggregator = PathAggregator(args.evidence_aggregation, args.evidence_top_k).to(device)
    candidate_sets = retrieve_hard_candidate_sets(
        examples,
        indices,
        hard_targets_per_query=args.hard_targets_per_query,
        hard_evidence_per_query=args.hard_evidence_per_query,
        max_evidence_per_target=args.max_evidence_per_target,
        retrieval_k=args.retrieval_k,
        direct_k=args.direct_k,
        evidence_k=args.evidence_k,
        targets_per_evidence=args.targets_per_evidence,
        evidence_types=tuple(args.evidence_types),
    )
    target_records, edge_records = score_hard_candidate_sets(
        candidate_sets,
        teacher,
        student,
        store,
        aggregator,
        device=device,
        batch_size=args.teacher_batch_size,
        mining_metadata={
            "mining_round": args.mining_round,
            "student_checkpoint_sha256": student_sha256,
            "teacher_checkpoint_sha256": teacher_sha256,
        },
    )
    _write_jsonl(Path(args.output_target_lists), target_records)
    if args.output_edge_lists:
        _write_jsonl(Path(args.output_edge_lists), edge_records)
    summary = {
        "queries": len(candidate_sets),
        "hard_targets": sum(len(item.hard_target_ids) for item in candidate_sets),
        "hard_evidence": sum(len(item.hard_evidence_ids) for item in candidate_sets),
        "hard_paths": sum(item.hard_path_count for item in candidate_sets),
        "output_target_lists": args.output_target_lists,
        "output_edge_lists": args.output_edge_lists,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--target-lists", required=True, nargs="+")
    parser.add_argument("--output-target-lists", required=True)
    parser.add_argument("--output-edge-lists")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--mining-round", type=int, default=1)
    parser.add_argument("--hard-targets-per-query", type=int, default=16)
    parser.add_argument("--hard-evidence-per-query", type=int, default=32)
    parser.add_argument("--max-evidence-per-target", type=int, default=8)
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--direct-k", type=int, default=200)
    parser.add_argument("--evidence-k", type=int, default=100)
    parser.add_argument("--targets-per-evidence", type=int, default=100)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument(
        "--evidence-aggregation", choices=["logsumexp", "topk_mean", "topk_sum"], default="logsumexp"
    )
    parser.add_argument("--evidence-top-k", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

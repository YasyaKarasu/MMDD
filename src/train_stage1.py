#!/usr/bin/env python
"""Train the four stages of the directed multimodal joinability model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mmdd_stage1.checkpoints import load_path_aggregation, load_student, load_teacher
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import StudentJoinabilityModel, TeacherJoinabilityModel
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import checkpoint_fingerprint
from mmdd_stage1.training import (
    checkpoint,
    train_student_edges,
    train_student_paths,
    train_teacher_edges,
    train_teacher_paths,
)

STAGES = ("teacher-edge", "teacher-path", "student-edge", "student-path")


def _required_path(value: str | None, flag: str, stage: str) -> Path:
    if value is None:
        raise ValueError(f"{flag} is required for stage {stage}")
    return Path(value)


def _load_edge_training_data(paths: list[Path], split: str | None):
    return [
        example
        for path in paths
        for example in load_edge_examples(path, split=split, dataset_name=path.stem)
    ]


def _load_target_training_data(paths: list[Path], split: str | None, max_evidence: int):
    return [
        example
        for path in paths
        for example in load_target_examples(
            path,
            split=split,
            max_evidence=max_evidence,
            dataset_name=path.stem,
        )
    ]


def _validate_cached_teacher(examples, teacher_checkpoint: Path) -> None:
    expected = checkpoint_fingerprint(teacher_checkpoint)
    mismatched = {
        example.teacher_checkpoint_sha256
        for example in examples
        if example.teacher_checkpoint_sha256 is not None and example.teacher_checkpoint_sha256 != expected
    }
    if mismatched:
        raise ValueError("Cached Teacher logits were produced by a different Teacher checkpoint")


def run(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.distillation_weight < 0:
        raise ValueError("--distillation-weight must be non-negative")
    if not 0 <= args.dataset_sampling_alpha <= 1:
        raise ValueError("--dataset-sampling-alpha must be between 0 and 1")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    store = FeatureStore.from_path(Path(args.features), cache_size=args.feature_cache_size)
    embedding_dim, hidden_dim = store.dimensions()
    if hidden_dim is None:
        raise ValueError("All training stages require hidden_states so the frozen Teacher can score candidates")
    split = None if args.split == "all" else args.split
    training_paths = [Path(value) for value in args.train_data]
    aggregator: PathAggregator | None = None
    if args.stage != "teacher-edge":
        aggregation_checkpoint = args.student_checkpoint or args.teacher_checkpoint
        saved_aggregation, saved_top_k = (
            load_path_aggregation(Path(aggregation_checkpoint))
            if aggregation_checkpoint
            else ("logsumexp", 4)
        )
        evidence_aggregation = args.evidence_aggregation or saved_aggregation
        evidence_top_k = (
            args.evidence_top_k
            if args.evidence_top_k is not None
            else saved_top_k
        )
        aggregator = PathAggregator(evidence_aggregation, evidence_top_k)

    teacher: TeacherJoinabilityModel | None = None
    student: StudentJoinabilityModel | None = None
    if args.stage == "teacher-edge":
        teacher = (
            load_teacher(Path(args.teacher_checkpoint), device)
            if args.teacher_checkpoint
            else TeacherJoinabilityModel(
                input_dim=hidden_dim,
                model_dim=args.teacher_dim,
                num_heads=args.teacher_heads,
                num_layers=args.teacher_layers,
                text_latents=args.text_latents,
                image_latents=args.image_latents,
                dropout=args.dropout,
            ).to(device)
        )
        if teacher.input_dim != hidden_dim:
            raise ValueError("Teacher checkpoint input dimension does not match the feature cache")
        examples = _load_edge_training_data(training_paths, split)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history = train_teacher_edges(
            teacher, examples, store, optimizer, device=device, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed,
            dataset_sampling_alpha=args.dataset_sampling_alpha,
        )
        model = teacher
    elif args.stage == "teacher-path":
        teacher_path = _required_path(args.teacher_checkpoint, "--teacher-checkpoint", args.stage)
        teacher = load_teacher(teacher_path, device)
        if teacher.input_dim != hidden_dim:
            raise ValueError("Teacher checkpoint input dimension does not match the feature cache")
        examples = _load_target_training_data(training_paths, split, args.max_evidence_per_target)
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history = train_teacher_paths(
            teacher, examples, store, optimizer, aggregator, device=device, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed,
            dataset_sampling_alpha=args.dataset_sampling_alpha,
        )
        model = teacher
    elif args.stage == "student-edge":
        teacher_path = _required_path(args.teacher_checkpoint, "--teacher-checkpoint", args.stage)
        teacher = load_teacher(teacher_path, device)
        student = (
            load_student(Path(args.student_checkpoint), device)
            if args.student_checkpoint
            else StudentJoinabilityModel(embedding_dim, args.student_dim).to(device)
        )
        if teacher.input_dim != hidden_dim or student.input_dim != embedding_dim:
            raise ValueError("Checkpoint input dimensions do not match the feature cache")
        examples = _load_edge_training_data(training_paths, split)
        _validate_cached_teacher(examples, teacher_path)
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history = train_student_edges(
            student, teacher, examples, store, optimizer, device=device, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, temperature=args.temperature,
            dataset_sampling_alpha=args.dataset_sampling_alpha,
        )
        model = student
    else:
        teacher_path = _required_path(args.teacher_checkpoint, "--teacher-checkpoint", args.stage)
        student_path = _required_path(args.student_checkpoint, "--student-checkpoint", args.stage)
        teacher = load_teacher(teacher_path, device)
        student = load_student(student_path, device)
        if teacher.input_dim != hidden_dim or student.input_dim != embedding_dim:
            raise ValueError("Checkpoint input dimensions do not match the feature cache")
        examples = _load_target_training_data(training_paths, split, args.max_evidence_per_target)
        _validate_cached_teacher(examples, teacher_path)
        optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        history = train_student_paths(
            student, teacher, examples, store, optimizer, aggregator, device=device, epochs=args.epochs,
            batch_size=args.batch_size, seed=args.seed, temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            dataset_sampling_alpha=args.dataset_sampling_alpha,
        )
        model = student

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(model, args.stage, aggregator), output)
    history_path = output.with_suffix(output.suffix + ".history.json")
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "history": str(history_path),
                "examples": len(examples),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--features", required=True, help="Consolidated .pt cache or a directory with manifest.jsonl.")
    parser.add_argument(
        "--train-data",
        required=True,
        nargs="+",
        help="One or more edge-list or target/path-list JSONL files for the selected stage.",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--split", default="train", help="Record split to load; use 'all' to disable filtering.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument(
        "--dataset-sampling-alpha",
        type=float,
        default=0.0,
        help="Dataset sampling mass is n_d ** alpha: 0 balances datasets, 1 keeps natural proportions.",
    )

    parser.add_argument("--teacher-dim", type=int, default=512)
    parser.add_argument("--teacher-heads", type=int, default=8)
    parser.add_argument("--teacher-layers", type=int, default=3)
    parser.add_argument("--text-latents", type=int, default=16)
    parser.add_argument("--image-latents", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--student-dim", type=int, default=128)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--evidence-aggregation", choices=["logsumexp", "topk_mean", "topk_sum"]
    )
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--max-evidence-per-target", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

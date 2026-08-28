#!/usr/bin/env python
"""Train one explicit Stage-1 Teacher/Student stage with per-epoch dev gating."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_path_aggregation, load_student, load_teacher
from mmdd_stage1.data import EdgeExample, TargetExample, load_edge_examples, load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import (
    STUDENT_INITIALIZATIONS,
    StudentJoinabilityModel,
    TeacherJoinabilityModel,
)
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    checkpoint_fingerprint,
    load_or_build_raw_embedding_indices,
    load_corpus_ids,
)
from mmdd_stage1.selection import CheckpointManager, MetricGate, write_json
from mmdd_stage1.teacher_logits import (
    has_teacher_logits,
    load_teacher_logits,
    score_and_cache_teacher_logits,
)
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


def _load_edge_training_data(paths: list[Path], split: str):
    return [
        example
        for path in paths
        for example in load_edge_examples(path, split=split, dataset_name=path.stem)
    ]


def _load_target_training_data(paths: list[Path], split: str):
    return [
        example
        for path in paths
        for example in load_target_examples(
            path,
            split=split,
            dataset_name=path.stem,
        )
    ]


def _referenced_object_ids(examples: list[Any]) -> list[str]:
    object_ids = []
    for example in examples:
        object_ids.append(example.query_id)
        if isinstance(example, EdgeExample):
            object_ids.extend(example.candidate_ids)
        else:
            for candidate in example.candidates:
                object_ids.append(candidate.target_id)
                object_ids.extend(candidate.evidence_ids)
    return list(dict.fromkeys(object_ids))


def _teacher_logit_cache_dir(args: argparse.Namespace) -> Path:
    configured = args.teacher_logit_cache
    if configured:
        return Path(configured)
    features = Path(args.features)
    if features.is_dir():
        return features / "teacher_logits"
    return features.parent / f"{features.name}.teacher_logits"


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"{path}: hard-negative data requires a metadata sidecar")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata_path}: metadata must be a JSON object")
    return payload


def _validate_hard_provenance(
    paths: list[Path],
    *,
    teacher_checkpoint: Path,
    source_student_checkpoint: Path,
    aggregator: PathAggregator | None,
) -> int:
    teacher_sha256 = checkpoint_fingerprint(teacher_checkpoint)
    student_sha256 = checkpoint_fingerprint(source_student_checkpoint)
    rounds = set()
    for path in paths:
        metadata = _metadata(path)
        if metadata.get("teacher_checkpoint_sha256") != teacher_sha256:
            raise ValueError(f"{path}: hard-negative Teacher fingerprint does not match")
        if metadata.get("student_checkpoint_sha256") != student_sha256:
            raise ValueError(f"{path}: hard negatives were mined by a different Student checkpoint")
        if metadata.get("teacher_scoring") == "pending":
            raise ValueError(f"{path}: hard-negative Teacher scoring is still pending")
        if aggregator is not None and (
            metadata.get("evidence_aggregation") != aggregator.evidence_aggregation
            or int(metadata.get("evidence_top_k", 0)) != aggregator.top_k
        ):
            raise ValueError(f"{path}: hard-negative path aggregation does not match")
        rounds.add(int(metadata.get("mining_round", -1)))
    if len(rounds) != 1 or next(iter(rounds)) < 1:
        raise ValueError("Hard-negative inputs must belong to one explicit mining round")
    return next(iter(rounds))


class _EpochController:
    def __init__(
        self,
        *,
        output: Path,
        stage: str,
        aggregator: PathAggregator | None,
        primary_metric: str,
        min_delta: float,
        patience: int,
        store: FeatureStore,
        device: torch.device,
        dev_examples: list[Any],
        corpus_path: Path | None,
        index_root: Path | None,
        raw_index_root: Path | None,
        args: argparse.Namespace,
    ) -> None:
        self.stage = stage
        self.aggregator = aggregator
        self.manager = CheckpointManager(output)
        self.gate = MetricGate(primary_metric, min_delta=min_delta, patience=patience)
        self.store = store
        self.device = device
        self.dev_examples = dev_examples
        self.corpus_path = corpus_path
        self.index_root = index_root
        self.raw_index_root = raw_index_root
        self.args = args
        self.best_metrics: dict[str, Any] | None = None
        self.best_index: Path | None = None
        self.latest_index: Path | None = None
        self.created_indices: list[Path] = []
        self.raw_embedding_metrics: dict[str, Any] | None = None
        self.stop_reason = "max_epochs"
        self.corpus_sha256 = (
            checkpoint_fingerprint(corpus_path) if corpus_path is not None else None
        )
        self.ids_by_type = (
            load_corpus_ids(corpus_path, store) if corpus_path is not None else None
        )

    def __call__(
        self,
        epoch: int,
        model: TeacherJoinabilityModel | StudentJoinabilityModel,
        record: dict[str, Any],
    ) -> bool:
        candidate = self.manager.save_candidate(
            epoch, checkpoint(model, self.stage, self.aggregator)
        )
        candidate_sha256 = checkpoint_fingerprint(candidate)
        record["candidate_checkpoint"] = str(candidate.resolve())
        record["candidate_checkpoint_sha256"] = candidate_sha256

        if self.stage == "student-path":
            assert isinstance(model, StudentJoinabilityModel)
            assert self.index_root is not None
            assert self.raw_index_root is not None
            assert self.ids_by_type is not None
            index_dir = self.index_root / f"epoch_{epoch:03d}"
            self.latest_index = index_dir
            self.created_indices.append(index_dir)
            build_indices(
                model,
                self.store,
                self.ids_by_type,
                index_dir,
                device=self.device,
                checkpoint_sha256=candidate_sha256,
                corpus_sha256=self.corpus_sha256,
                batch_size=self.args.index_batch_size,
                m=self.args.hnsw_m,
                ef_construction=self.args.ef_construction,
                ef_search=self.args.ef_search,
            )
            indices = StudentANNIndices(
                model,
                self.store,
                index_dir,
                device=self.device,
                checkpoint_sha256=candidate_sha256,
                corpus_sha256=self.corpus_sha256,
            )
            retrieval_metrics = evaluate_student_retrieval(
                self.dev_examples,
                indices,
                direct_k=self.args.direct_k,
                evidence_k=self.args.evidence_k,
                targets_per_evidence=self.args.targets_per_evidence,
                evidence_types=tuple(self.args.evidence_types),
                evidence_aggregation=self.aggregator.evidence_aggregation,
                evidence_top_k=self.aggregator.top_k,
                rrf_k=self.args.rrf_k,
            )
            if self.raw_embedding_metrics is None:
                raw_indices = load_or_build_raw_embedding_indices(
                    self.store,
                    self.ids_by_type,
                    self.raw_index_root,
                    corpus_sha256=self.corpus_sha256,
                    batch_size=self.args.index_batch_size,
                    m=self.args.hnsw_m,
                    ef_construction=self.args.ef_construction,
                    ef_search=self.args.ef_search,
                )
                self.raw_embedding_metrics = evaluate_student_retrieval(
                    self.dev_examples,
                    raw_indices,
                    direct_k=self.args.direct_k,
                    evidence_k=self.args.evidence_k,
                    targets_per_evidence=self.args.targets_per_evidence,
                    evidence_types=tuple(self.args.evidence_types),
                    evidence_aggregation=self.aggregator.evidence_aggregation,
                    evidence_top_k=self.aggregator.top_k,
                    rrf_k=self.args.rrf_k,
                )
            retrieval_metrics["raw_embedding"] = self.raw_embedding_metrics
            record["dev_retrieval"] = retrieval_metrics
            gate_metrics = {**retrieval_metrics, "dev_loss": record["dev_loss"]}
        else:
            index_dir = None
            gate_metrics = {"dev_loss": record["dev_loss"]}

        decision = self.gate.observe(epoch, gate_metrics)
        record["gate"] = {
            "primary_metric": self.gate.primary_metric,
            "value": decision.value,
            "improved": decision.improved,
            "bad_epochs": decision.bad_epochs,
        }
        record["best_epoch_so_far"] = decision.best_epoch
        if decision.improved:
            self.manager.update_best(candidate)
            self.best_metrics = gate_metrics
            self.best_index = index_dir
        if decision.should_stop:
            self.stop_reason = f"early_stopping_patience_{self.gate.patience}"
        return decision.should_stop

    def prune_indices(self) -> None:
        retained = {self.best_index, self.latest_index}
        for path in self.created_indices:
            if path not in retained and path.is_dir():
                shutil.rmtree(path)


def _data_paths(args: argparse.Namespace) -> tuple[list[Path], list[Path], list[Path]]:
    legacy = getattr(args, "train_data", None)
    base = getattr(args, "base_data", None)
    if legacy and base:
        raise ValueError("Use --base-data, not both --base-data and legacy --train-data")
    base_values = base or legacy
    if not base_values:
        raise ValueError("--base-data is required")
    dev_values = getattr(args, "dev_data", None)
    if not dev_values:
        raise ValueError("--dev-data is required for per-epoch checkpoint gating")
    return (
        [Path(value) for value in base_values],
        [Path(value) for value in getattr(args, "hard_data", [])],
        [Path(value) for value in dev_values],
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size is None:
        args.batch_size = 64 if args.stage.startswith("student") else 8
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.teacher_logit_batch_size <= 0:
        raise ValueError("--teacher-logit-batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.distillation_weight < 0:
        raise ValueError("--distillation-weight must be non-negative")
    if not 0 <= args.dataset_sampling_alpha <= 1:
        raise ValueError("--dataset-sampling-alpha must be between 0 and 1")
    if not 0 <= args.hard_fraction < 1:
        raise ValueError("--hard-fraction must be in [0, 1)")
    if args.min_delta < 0 or args.patience < 0:
        raise ValueError("--min-delta and --patience must be non-negative")
    if args.learning_rate <= 0 or args.hard_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")
    if args.min_dev_evidence_path_queries < 0:
        raise ValueError("--min-dev-evidence-path-queries must be non-negative")
    if not 0 <= args.min_dev_evidence_path_coverage <= 1:
        raise ValueError("--min-dev-evidence-path-coverage must be in [0, 1]")
    validate_protocol_split("training", args.split)
    validate_protocol_split("dev_gate", args.dev_split)

    base_paths, hard_paths, dev_paths = _data_paths(args)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    embedding_dim = store.embedding_dimension()
    hidden_dim: int | None = None

    is_path = args.stage.endswith("path")
    loader = _load_target_training_data if is_path else _load_edge_training_data
    examples = loader(base_paths, args.split)
    hard_examples = loader(hard_paths, args.split) if hard_paths else []
    dev_examples = loader(dev_paths, args.dev_split)

    aggregator: PathAggregator | None = None
    if args.stage != "teacher-edge":
        aggregation_checkpoint = args.student_checkpoint or args.teacher_checkpoint
        saved_aggregation, saved_top_k = (
            load_path_aggregation(Path(aggregation_checkpoint))
            if aggregation_checkpoint
            else ("logsumexp", 4)
        )
        aggregator = PathAggregator(
            args.evidence_aggregation or saved_aggregation,
            args.evidence_top_k
            if args.evidence_top_k is not None
            else saved_top_k,
        )

    teacher: TeacherJoinabilityModel | None = None
    student: StudentJoinabilityModel | None = None
    mining_round = None
    teacher_cache_paths: list[Path] = []
    teacher_cache_hits = 0
    teacher_cache_generated = False
    preloaded_embeddings = 0
    if args.stage.startswith("student"):
        teacher_path = _required_path(
            args.teacher_checkpoint, "--teacher-checkpoint", args.stage
        )
        teacher_sha256 = checkpoint_fingerprint(teacher_path)
        cache_aggregator = aggregator if is_path else None
        if hard_examples:
            source_checkpoint = _required_path(
                args.hard_source_checkpoint,
                "--hard-source-checkpoint",
                args.stage,
            )
            mining_round = _validate_hard_provenance(
                hard_paths,
                teacher_checkpoint=teacher_path,
                source_student_checkpoint=source_checkpoint,
                aggregator=aggregator,
            )
            if not has_teacher_logits(
                hard_examples, teacher_sha256, cache_aggregator
            ):
                raise ValueError(
                    "Hard-negative data must contain matching cached Teacher logits"
                )

        cache_dir = _teacher_logit_cache_dir(args)
        examples, train_cache_path, train_cache_hit = load_teacher_logits(
            examples, cache_dir, teacher_sha256, cache_aggregator
        )
        dev_examples, dev_cache_path, dev_cache_hit = load_teacher_logits(
            dev_examples, cache_dir, teacher_sha256, cache_aggregator
        )
        teacher_cache_hits = int(train_cache_hit) + int(dev_cache_hit)

        train_logits_ready = has_teacher_logits(
            examples, teacher_sha256, cache_aggregator
        )
        dev_logits_ready = has_teacher_logits(
            dev_examples, teacher_sha256, cache_aggregator
        )
        if not train_logits_ready or not dev_logits_ready:
            frozen_teacher = load_teacher(teacher_path, device)
            teacher_cache_generated = True
            hidden_dim = store.teacher_dimension()
            if hidden_dim is None:
                raise ValueError(
                    "Teacher logit cache is incomplete and the feature cache has no hidden_states"
                )
            if frozen_teacher.input_dim != hidden_dim:
                raise ValueError(
                    "Teacher checkpoint input dimension does not match the feature cache"
                )
            if not train_logits_ready:
                examples, train_cache_path = score_and_cache_teacher_logits(
                    examples,
                    frozen_teacher,
                    store,
                    cache_dir,
                    teacher_sha256,
                    device=device,
                    batch_size=args.teacher_logit_batch_size,
                    aggregator=cache_aggregator,
                )
            if not dev_logits_ready:
                dev_examples, dev_cache_path = score_and_cache_teacher_logits(
                    dev_examples,
                    frozen_teacher,
                    store,
                    cache_dir,
                    teacher_sha256,
                    device=device,
                    batch_size=args.teacher_logit_batch_size,
                    aggregator=cache_aggregator,
                )
            del frozen_teacher
            if device.type == "cuda":
                torch.cuda.empty_cache()
        teacher_cache_paths = [train_cache_path, dev_cache_path]

        if args.preload_embeddings:
            preloaded_embeddings = store.preload_embeddings(
                _referenced_object_ids([*examples, *hard_examples, *dev_examples])
            )
    else:
        hidden_dim = store.teacher_dimension()
        if hidden_dim is None:
            raise ValueError("Teacher training requires cached hidden_states")

    corpus_path = None
    index_root = None
    raw_index_root = None
    primary_metric = "dev_loss"
    if args.stage == "student-path":
        corpus_path = _required_path(args.corpus, "--corpus", args.stage)
        index_root = (
            Path(args.index_root)
            if args.index_root
            else Path(args.output).with_suffix(".dev_indices")
        )
        raw_index_root = (
            Path(args.raw_index_root)
            if args.raw_index_root
            else index_root / "raw_embedding"
        )
        primary_metric = args.primary_metric

    controller = _EpochController(
        output=Path(args.output),
        stage=args.stage,
        aggregator=aggregator,
        primary_metric=primary_metric,
        min_delta=args.min_delta,
        patience=args.patience,
        store=store,
        device=device,
        dev_examples=dev_examples,
        corpus_path=corpus_path,
        index_root=index_root,
        raw_index_root=raw_index_root,
        args=args,
    )
    learning_rate = args.hard_learning_rate if hard_examples else args.learning_rate
    if hard_examples and args.hard_learning_rate >= args.learning_rate:
        raise ValueError(
            "--hard-learning-rate must be lower than --learning-rate"
        )

    common = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset_sampling_alpha": args.dataset_sampling_alpha,
        "hard_examples": hard_examples,
        "hard_fraction": args.hard_fraction,
        "dev_examples": dev_examples,
        "epoch_callback": controller,
    }
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
        optimizer = torch.optim.AdamW(
            teacher.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
        history = train_teacher_edges(teacher, examples, store, optimizer, **common)
    elif args.stage == "teacher-path":
        teacher_path = _required_path(
            args.teacher_checkpoint, "--teacher-checkpoint", args.stage
        )
        teacher = load_teacher(teacher_path, device)
        if teacher.input_dim != hidden_dim:
            raise ValueError("Teacher checkpoint input dimension does not match the feature cache")
        optimizer = torch.optim.AdamW(
            teacher.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
        history = train_teacher_paths(
            teacher, examples, store, optimizer, aggregator, **common
        )
    elif args.stage == "student-edge":
        if hard_examples and not args.student_checkpoint:
            raise ValueError(
                "Hard-negative Student edge training must continue from the previous best Student checkpoint"
            )
        if hard_examples and checkpoint_fingerprint(
            Path(args.student_checkpoint)
        ) != checkpoint_fingerprint(Path(args.hard_source_checkpoint)):
            raise ValueError(
                "Hard-negative Student edge training must start from the Student "
                "checkpoint used for mining"
            )
        if args.student_checkpoint:
            student = load_student(Path(args.student_checkpoint), device)
        else:
            initialization = getattr(args, "student_initialization", "random")
            pca_basis_path = getattr(args, "student_pca_basis", None)
            if initialization == "pca":
                if pca_basis_path is None:
                    raise ValueError("--student-pca-basis is required for PCA initialization")
                initialization_basis = load_pca_projection(
                    Path(pca_basis_path),
                    input_dim=embedding_dim,
                    student_dim=args.student_dim,
                )
            else:
                if pca_basis_path is not None:
                    raise ValueError(
                        "--student-pca-basis requires --student-initialization pca"
                    )
                initialization_basis = None
            student = StudentJoinabilityModel(
                embedding_dim,
                args.student_dim,
                initialization=initialization,
                initialization_noise_std=getattr(
                    args, "student_init_noise_std", 0.01
                ),
                initialization_basis=initialization_basis,
            ).to(device)
        if student.input_dim != embedding_dim:
            raise ValueError("Student checkpoint input dimension does not match the feature cache")
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
        history = train_student_edges(
            student,
            examples,
            store,
            optimizer,
            temperature=args.temperature,
            **common,
        )
    else:
        student_path = _required_path(
            args.student_checkpoint, "--student-checkpoint", args.stage
        )
        student = load_student(student_path, device)
        if student.input_dim != embedding_dim:
            raise ValueError("Student checkpoint input dimension does not match the feature cache")
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
        history = train_student_paths(
            student,
            examples,
            store,
            optimizer,
            aggregator,
            temperature=args.temperature,
            distillation_weight=args.distillation_weight,
            **common,
        )

    controller.prune_indices()
    paths = controller.manager.paths
    assert controller.best_metrics is not None
    best_sha256 = checkpoint_fingerprint(paths["best"])
    evidence_count = int(
        controller.best_metrics.get("positive_evidence_path_queries@10", 0)
    )
    evidence_coverage = float(
        controller.best_metrics.get("positive_evidence_path_coverage@10", 0.0)
    )
    stage2_allowed = (
        args.stage == "student-path"
        and evidence_count >= args.min_dev_evidence_path_queries
        and evidence_coverage >= args.min_dev_evidence_path_coverage
    )
    history_payload = {
        "format_version": 1,
        "completed_stage": args.stage,
        "epochs": history,
        "best_epoch": controller.gate.best_epoch,
        "best_metrics": controller.best_metrics,
        "primary_metric": controller.gate.primary_metric,
        "min_delta": args.min_delta,
        "patience": args.patience,
        "stop_reason": controller.stop_reason,
        "learning_rate": learning_rate,
        "base_examples": len(examples),
        "hard_examples": len(hard_examples),
        "hard_fraction": args.hard_fraction if hard_examples else 0.0,
        "mining_round": mining_round,
        "teacher_cache_generated": teacher_cache_generated,
        "teacher_logit_cache_hits": teacher_cache_hits,
        "teacher_logit_caches": [str(path.resolve()) for path in teacher_cache_paths],
        "preloaded_embeddings": preloaded_embeddings,
    }
    selection = {
        "format_version": 1,
        "completed_stage": args.stage,
        "selection_split": "dev",
        "primary_metric": controller.gate.primary_metric,
        "best_epoch": controller.gate.best_epoch,
        "best_metrics": controller.best_metrics,
        "best_checkpoint": str(paths["best"].resolve()),
        "best_checkpoint_sha256": best_sha256,
        "last_checkpoint": str(paths["last"].resolve()),
        "best_index": (
            str(controller.best_index.resolve())
            if controller.best_index is not None
            else None
        ),
        "latest_index": (
            str(controller.latest_index.resolve())
            if controller.latest_index is not None
            else None
        ),
        "raw_embedding_index": (
            str(controller.raw_index_root.resolve())
            if controller.raw_index_root is not None
            else None
        ),
        "corpus_sha256": controller.corpus_sha256,
        "stage2_allowed": stage2_allowed,
        "stage2_evidence_gate": {
            "minimum_queries": args.min_dev_evidence_path_queries,
            "minimum_coverage": args.min_dev_evidence_path_coverage,
            "observed_queries": evidence_count,
            "observed_coverage": evidence_coverage,
        },
        "stop_reason": controller.stop_reason,
        "mining_round": mining_round,
    }
    if student is not None:
        history_payload["student_config"] = student.config()
    write_json(paths["history"], history_payload)
    write_json(paths["selection"], selection)
    summary = {
        "best_checkpoint": str(paths["best"]),
        "last_checkpoint": str(paths["last"]),
        "history": str(paths["history"]),
        "selection": str(paths["selection"]),
        "best_epoch": controller.gate.best_epoch,
        "stop_reason": controller.stop_reason,
        "base_examples": len(examples),
        "hard_examples": len(hard_examples),
        "stage2_allowed": stage2_allowed,
        "teacher_cache_generated": teacher_cache_generated,
        "teacher_logit_cache_hits": teacher_cache_hits,
        "preloaded_embeddings": preloaded_embeddings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--features", required=True)
    parser.add_argument("--base-data", nargs="+", help="Base random/semantic/structural/corrupted lists.")
    parser.add_argument("--hard-data", nargs="*", default=[], help="ANN-mined hard-negative lists.")
    parser.add_argument("--train-data", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--dev-data", required=True, nargs="+", help="Fixed dev edge or target/path lists.")
    parser.add_argument("--output", required=True, help="Best checkpoint path; last uses a distinct sibling path.")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--hard-source-checkpoint", help="Student checkpoint used to mine --hard-data.")
    parser.add_argument("--corpus", help="Full shared data-lake corpus; required by student-path.")
    parser.add_argument("--index-root", help="Per-epoch dev ANN index root.")
    parser.add_argument(
        "--raw-index-root",
        help="Reusable corpus ANN index built directly from frozen embeddings.",
    )
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--dev-split", default="dev", choices=["dev"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Defaults to 8 for Teacher stages and 64 for vectorized Student stages.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hard-learning-rate", type=float, default=2e-5)
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument(
        "--teacher-logit-cache",
        help="Persistent base/dev Teacher-logit cache; defaults inside the feature cache.",
    )
    parser.add_argument("--teacher-logit-batch-size", type=int, default=8)
    parser.add_argument(
        "--preload-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload all Student-training embeddings into one contiguous CPU tensor.",
    )
    parser.add_argument("--dataset-sampling-alpha", type=float, default=0.0)

    parser.add_argument("--primary-metric", default="recall@10")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-dev-evidence-path-queries", type=int, default=1)
    parser.add_argument("--min-dev-evidence-path-coverage", type=float, default=0.0)

    parser.add_argument("--teacher-dim", type=int, default=512)
    parser.add_argument("--teacher-heads", type=int, default=8)
    parser.add_argument("--teacher-layers", type=int, default=3)
    parser.add_argument("--text-latents", type=int, default=16)
    parser.add_argument("--image-latents", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--student-dim", type=int, default=128)
    parser.add_argument(
        "--student-initialization",
        choices=STUDENT_INITIALIZATIONS,
        default="random",
    )
    parser.add_argument("--student-init-noise-std", type=float, default=0.01)
    parser.add_argument("--student-pca-basis")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--evidence-aggregation", choices=["logsumexp", "topk_mean", "topk_sum"])
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--direct-k", type=int, default=100)
    parser.add_argument("--evidence-k", type=int, default=50)
    parser.add_argument("--targets-per-evidence", type=int, default=50)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

"""Refresh Student hard target/evidence/path candidates and Teacher scores."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from mmdd_stage1.checkpoints import load_path_aggregator, load_student, load_teacher
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.models import STUDENT_SCORE_SPACES
from mmdd_stage1.mining import (
    hard_candidate_records,
    retrieve_hard_candidate_sets,
    score_hard_candidate_sets,
    score_pending_hard_examples,
    summarize_hard_candidate_sets,
)
from mmdd_stage1.objectives import PATH_AGGREGATIONS, PathAggregator
from mmdd_stage1.protocol import validate_protocol_split
from mmdd_stage1.retrieval import StudentANNIndices, checkpoint_fingerprint


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_pending_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"{path}: pending hard negatives require a metadata sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be a JSON object")
    return metadata


def _mining_selection_config(metadata: dict[str, Any]) -> dict[str, Any]:
    configured = metadata.get("mining_selection_config")
    if configured is not None:
        if not isinstance(configured, dict):
            raise ValueError("mining_selection_config must be a JSON object")
        return dict(configured)
    return {
        "student_score_space": metadata.get("student_score_space", "raw_logit"),
        "evidence_aggregation": metadata.get(
            "evidence_aggregation", "logsumexp"
        ),
        "evidence_top_k": int(metadata.get("evidence_top_k", 4)),
        "evidence_temperature": float(
            metadata.get("evidence_temperature", 1.0)
        ),
        "evidence_power": float(metadata.get("evidence_power", 2.0)),
        "path_combination": metadata.get("path_combination", "sum"),
        "evidence_threshold": float(
            metadata.get("evidence_threshold", 0.0)
        ),
        "evidence_target_temperature": float(
            metadata.get("evidence_target_temperature", 1.0)
        ),
    }


def _validate_pending_metadata(
    target_path: Path,
    edge_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    target_metadata = _read_pending_metadata(target_path)
    edge_metadata = _read_pending_metadata(edge_path)
    if target_metadata != edge_metadata:
        raise ValueError("Pending target and edge metadata differ")
    if target_metadata.get("teacher_scoring") != "pending":
        raise ValueError("Pending hard negatives have already been Teacher-scored")
    mining_config = _mining_selection_config(target_metadata)
    for key, value in expected.items():
        default = {
            "evidence_temperature": 1.0,
            "evidence_power": 2.0,
            "path_combination": "sum",
            "evidence_threshold": 0.0,
            "student_score_space": "raw_logit",
            "hard_targets_per_positive_evidence": 0,
        }.get(key)
        actual = (
            mining_config.get(key, default)
            if key == "student_score_space"
            else target_metadata.get(key, default)
        )
        if actual != value:
            raise ValueError(
                f"{target_path}: pending metadata {key!r} does not match this run"
            )
    return target_metadata


def run(args: argparse.Namespace) -> None:
    teacher_ensemble_alpha = getattr(args, "teacher_ensemble_alpha", None)
    teacher_score_space = getattr(args, "teacher_score_space", "raw_logit")
    student_score_space = getattr(args, "student_score_space", "raw_logit")
    training_student_score_space = getattr(
        args, "training_student_score_space", None
    ) or student_score_space
    hard_targets_per_positive_evidence = getattr(
        args, "hard_targets_per_positive_evidence", 0
    )
    pending_target_value = getattr(args, "pending_target_lists", None)
    pending_edge_value = getattr(args, "pending_edge_lists", None)
    if bool(pending_target_value) != bool(pending_edge_value):
        raise ValueError(
            "--pending-target-lists and --pending-edge-lists must be provided together"
        )
    use_pending = bool(pending_target_value)
    if args.mine_only and not args.output_edge_lists:
        raise ValueError("--output-edge-lists is required with --mine-only")
    if args.mine_only and use_pending:
        raise ValueError("Pending hard negatives can only be used for Teacher scoring")
    if use_pending and not args.output_edge_lists:
        raise ValueError("--output-edge-lists is required when scoring pending hard negatives")
    if args.hard_targets_per_query <= 0:
        raise ValueError("Hard-target size must be positive")
    if not args.mine_only and args.teacher_checkpoint is None:
        raise ValueError("--teacher-checkpoint is required unless --mine-only is set")
    if not args.mine_only and args.teacher_batch_size <= 0:
        raise ValueError("Teacher batch size must be positive")
    if teacher_ensemble_alpha is not None and not 0 <= teacher_ensemble_alpha <= 1:
        raise ValueError("--teacher-ensemble-alpha must be in [0, 1]")
    if teacher_ensemble_alpha is not None and teacher_score_space != "raw_logit":
        raise ValueError("Teacher/raw ensembles require --teacher-score-space raw_logit")
    if min(
        args.hard_evidence_per_type,
        hard_targets_per_positive_evidence,
        args.hard_paths_per_query,
    ) < 0:
        raise ValueError("Hard-evidence, E-T, and hard-path sizes must be non-negative")
    if (
        getattr(args, "row_support_top_l", None) is not None
        and args.row_support_top_l <= 0
    ):
        raise ValueError("--row-support-top-l must be positive")
    validate_protocol_split("mining", args.split)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    split = args.split
    target_paths = [Path(value) for value in args.target_lists]
    store = FeatureStore.from_path(
        Path(args.features),
        cache_size=args.feature_cache_size,
        teacher_paths=tuple(
            Path(value) for value in getattr(args, "teacher_features", [])
        ),
    )
    student_path = Path(args.student_checkpoint)
    saved_aggregator = load_path_aggregator(student_path)
    evidence_aggregation = (
        args.evidence_aggregation or saved_aggregator.evidence_aggregation
    )
    evidence_top_k = (
        args.evidence_top_k
        if args.evidence_top_k is not None
        else saved_aggregator.top_k
    )
    evidence_temperature = (
        getattr(args, "evidence_temperature", None)
        if getattr(args, "evidence_temperature", None) is not None
        else saved_aggregator.temperature
    )
    evidence_power = (
        getattr(args, "evidence_power", None)
        if getattr(args, "evidence_power", None) is not None
        else saved_aggregator.power
    )
    path_combination = (
        getattr(args, "path_combination", None)
        if getattr(args, "path_combination", None) is not None
        else saved_aggregator.path_combination
    )
    evidence_threshold = (
        getattr(args, "evidence_threshold", None)
        if getattr(args, "evidence_threshold", None) is not None
        else saved_aggregator.threshold
    )
    evidence_target_temperature = (
        getattr(args, "evidence_target_temperature", None)
        if getattr(args, "evidence_target_temperature", None) is not None
        else saved_aggregator.target_temperature
    )
    row_support_model = (
        getattr(args, "row_support_model", None)
        if getattr(args, "row_support_model", None) is not None
        else saved_aggregator.row_support_model
    )
    row_support_model_sha256 = (
        saved_aggregator.row_support_model_sha256
        if row_support_model == saved_aggregator.row_support_model
        else None
    )
    row_support_top_l = (
        getattr(args, "row_support_top_l", None)
        if getattr(args, "row_support_top_l", None) is not None
        else saved_aggregator.row_support_top_l
    )
    evidence_content_keys = (
        getattr(args, "evidence_content_keys", None)
        if getattr(args, "evidence_content_keys", None) is not None
        else saved_aggregator.evidence_content_keys
    )
    evidence_content_keys_sha256 = (
        saved_aggregator.evidence_content_keys_sha256
        if evidence_content_keys == saved_aggregator.evidence_content_keys
        else None
    )
    student_sha256 = checkpoint_fingerprint(student_path)
    corpus_sha256 = checkpoint_fingerprint(Path(args.corpus))
    evidence_types = tuple(dict.fromkeys(args.evidence_types))
    index_manifest_sha256 = checkpoint_fingerprint(
        Path(args.index_dir) / "manifest.json"
    )
    pending_metadata = None
    pending_targets = []
    pending_edges = []
    candidate_sets = []
    if use_pending:
        pending_target_path = Path(str(pending_target_value))
        pending_edge_path = Path(str(pending_edge_value))
        pending_metadata = _validate_pending_metadata(
            pending_target_path,
            pending_edge_path,
            {
                "mining_round": args.mining_round,
                "student_checkpoint_sha256": student_sha256,
                "corpus_sha256": corpus_sha256,
                "index_manifest_sha256": index_manifest_sha256,
                "student_score_space": student_score_space,
                "hard_targets_per_query": args.hard_targets_per_query,
                "hard_evidence_per_type": args.hard_evidence_per_type,
                "hard_targets_per_positive_evidence": (
                    hard_targets_per_positive_evidence
                ),
                "hard_paths_per_query": args.hard_paths_per_query,
                "direct_k": args.direct_k,
                "evidence_k": args.evidence_k,
                "targets_per_evidence": args.targets_per_evidence,
                "evidence_types": list(evidence_types),
            },
        )
        pending_targets = load_target_examples(pending_target_path, split=split)
        pending_edges = load_edge_examples(pending_edge_path, split=split)
    else:
        examples = [
            example
            for path in target_paths
            for example in load_target_examples(
                path,
                split=split,
                dataset_name=path.stem,
            )
        ]
        embedding_dim = store.embedding_dimension()
        student = load_student(student_path, device)
        if student.input_dim != embedding_dim:
            raise ValueError("Student input dimension does not match the feature cache")
        indices = StudentANNIndices(
            student,
            store,
            Path(args.index_dir),
            device=device,
            checkpoint_sha256=student_sha256,
            corpus_sha256=corpus_sha256,
            score_space=student_score_space,
        )
        candidate_sets = retrieve_hard_candidate_sets(
            examples,
            indices,
            hard_targets_per_query=args.hard_targets_per_query,
            hard_evidence_per_type=args.hard_evidence_per_type,
            hard_targets_per_positive_evidence=(
                hard_targets_per_positive_evidence
            ),
            hard_paths_per_query=args.hard_paths_per_query,
            direct_k=args.direct_k,
            evidence_k=args.evidence_k,
            targets_per_evidence=args.targets_per_evidence,
            evidence_types=evidence_types,
        )
    teacher_sha256 = None
    if args.mine_only:
        target_records, edge_records = hard_candidate_records(candidate_sets, store)
    else:
        hidden_dim = store.teacher_dimension()
        if hidden_dim is None:
            raise ValueError(
                "Hard-negative Teacher rescoring requires a populated Teacher feature tier"
            )
        teacher_path = Path(args.teacher_checkpoint)
        teacher = load_teacher(teacher_path, device)
        teacher_sha256 = checkpoint_fingerprint(teacher_path)
        if teacher.input_dim != hidden_dim:
            raise ValueError("Teacher input dimension does not match the feature cache")
        aggregator = PathAggregator(
            evidence_aggregation,
            evidence_top_k,
            temperature=evidence_temperature,
            power=evidence_power,
            path_combination=path_combination,
            threshold=evidence_threshold,
            target_temperature=evidence_target_temperature,
            row_support_model=row_support_model,
            row_support_model_sha256=row_support_model_sha256,
            row_support_top_l=row_support_top_l,
            evidence_content_keys=evidence_content_keys,
            evidence_content_keys_sha256=evidence_content_keys_sha256,
        )
        if (
            aggregator.evidence_aggregation == "greedy_row_support"
            and (
                not aggregator.row_support_models
                or not aggregator.evidence_content_key_by_id
            )
        ):
            raise ValueError(
                "greedy_row_support requires --row-support-model and "
                "--evidence-content-keys"
            )
        if use_pending:
            target_records, edge_records = score_pending_hard_examples(
                pending_targets,
                pending_edges,
                teacher,
                store,
                aggregator,
                device=device,
                batch_size=args.teacher_batch_size,
                ensemble_alpha=teacher_ensemble_alpha,
                teacher_score_space=teacher_score_space,
            )
        else:
            target_records, edge_records = score_hard_candidate_sets(
                candidate_sets,
                teacher,
                store,
                aggregator,
                device=device,
                batch_size=args.teacher_batch_size,
                ensemble_alpha=teacher_ensemble_alpha,
                teacher_score_space=teacher_score_space,
            )
    _write_jsonl(Path(args.output_target_lists), target_records)
    if args.output_edge_lists:
        _write_jsonl(Path(args.output_edge_lists), edge_records)
    metadata = dict(pending_metadata) if pending_metadata is not None else {
        "mining_round": args.mining_round,
        "student_checkpoint_sha256": student_sha256,
        "corpus_sha256": corpus_sha256,
        "index_manifest_sha256": index_manifest_sha256,
        "evidence_aggregation": evidence_aggregation,
        "evidence_top_k": evidence_top_k,
        "evidence_temperature": evidence_temperature,
        "evidence_power": evidence_power,
        "path_combination": path_combination,
        "evidence_threshold": evidence_threshold,
        "student_score_space": student_score_space,
        "teacher_target_channels": ["direct", "evidence"],
        "hard_negative_mining": {
            "hard_evidence": "query_to_evidence_ann",
            "hard_evidence_target": "positive_evidence_to_target_ann",
            "hard_target": "query_to_target_ann",
            "path_hard": "raw_query_evidence_target_path_score",
        },
        "hard_targets_per_query": args.hard_targets_per_query,
        "hard_evidence_per_type": args.hard_evidence_per_type,
        "hard_targets_per_positive_evidence": (
            hard_targets_per_positive_evidence
        ),
        "hard_paths_per_query": args.hard_paths_per_query,
        "direct_k": args.direct_k,
        "evidence_k": args.evidence_k,
        "targets_per_evidence": args.targets_per_evidence,
        "evidence_types": list(evidence_types),
        "teacher_ensemble_alpha": teacher_ensemble_alpha,
    }
    metadata.setdefault("mining_selection_config", _mining_selection_config(metadata))
    if teacher_sha256 is not None:
        teacher_logit_mode = (
            "ensemble"
            if teacher_ensemble_alpha is not None
            else "teacher"
            if teacher_score_space == "raw_logit"
            else f"teacher_{teacher_score_space}"
        )
        metadata["teacher_checkpoint_sha256"] = teacher_sha256
        metadata["teacher_scoring"] = "complete"
        metadata["teacher_score_space"] = teacher_score_space
        metadata["training_score_config"] = {
            **aggregator.config(),
            "teacher_score_space": teacher_score_space,
            "student_score_space": training_student_score_space,
            "teacher_ensemble_alpha": teacher_ensemble_alpha,
        }
        metadata["teacher_target_logit_mode"] = teacher_logit_mode
        metadata["teacher_target_ensemble_alpha"] = teacher_ensemble_alpha
        metadata["teacher_edge_logit_mode"] = teacher_logit_mode
        metadata["teacher_edge_ensemble_alpha"] = teacher_ensemble_alpha
    else:
        metadata["teacher_scoring"] = "pending"
    if candidate_sets:
        metadata["mining_pool_statistics"] = summarize_hard_candidate_sets(
            candidate_sets
        )
    output_paths = [Path(args.output_target_lists)]
    if args.output_edge_lists:
        output_paths.append(Path(args.output_edge_lists))
    metadata_paths = [path.with_suffix(path.suffix + ".metadata.json") for path in output_paths]
    for metadata_path in metadata_paths:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "queries": len(pending_targets) if use_pending else len(candidate_sets),
        "target_lists": args.output_target_lists,
        "edge_lists": args.output_edge_lists,
        "metadata": str(metadata_paths[0]),
        "teacher_scoring": "pending" if args.mine_only else "complete",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument(
        "--teacher-features",
        nargs="*",
        default=[],
        help="Optional Teacher-only cache directories or manifest paths.",
    )
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument(
        "--student-score-space",
        choices=STUDENT_SCORE_SPACES,
        default="raw_logit",
    )
    parser.add_argument(
        "--training-student-score-space",
        choices=STUDENT_SCORE_SPACES,
        help=(
            "Student score space that will consume the rescored candidate pool; "
            "defaults to the mining Student score space."
        ),
    )
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--corpus", required=True, help="Full shared corpus used for the ANN index.")
    parser.add_argument("--target-lists", required=True, nargs="+")
    parser.add_argument("--output-target-lists", required=True)
    parser.add_argument("--output-edge-lists")
    parser.add_argument(
        "--pending-target-lists",
        help="Persisted mine-only target candidates to score without repeating ANN retrieval.",
    )
    parser.add_argument(
        "--pending-edge-lists",
        help="Persisted mine-only edge candidates to score without repeating ANN retrieval.",
    )
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-cache-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--teacher-ensemble-alpha", type=float)
    parser.add_argument(
        "--teacher-score-space",
        choices=STUDENT_SCORE_SPACES,
        default="raw_logit",
    )
    parser.add_argument(
        "--mine-only",
        action="store_true",
        help=(
            "Write mined candidates without Teacher logits. Use them to supplement the "
            "Teacher feature tier, then rerun without this flag."
        ),
    )
    parser.add_argument("--mining-round", type=int, default=1)
    parser.add_argument("--hard-targets-per-query", type=int, default=16)
    parser.add_argument("--hard-evidence-per-type", type=int, default=16)
    parser.add_argument(
        "--hard-targets-per-positive-evidence",
        type=int,
        default=0,
        help="Independently mine this many E-to-target negatives per positive evidence.",
    )
    parser.add_argument("--hard-paths-per-query", type=int, default=16)
    parser.add_argument("--direct-k", type=int, default=200)
    parser.add_argument("--evidence-k", type=int, default=100)
    parser.add_argument("--targets-per-evidence", type=int, default=100)
    parser.add_argument("--evidence-types", nargs="+", choices=["text", "image"], default=["text", "image"])
    parser.add_argument(
        "--evidence-aggregation", choices=sorted(PATH_AGGREGATIONS)
    )
    parser.add_argument("--evidence-top-k", type=int)
    parser.add_argument("--evidence-temperature", type=float)
    parser.add_argument("--evidence-power", type=float)
    parser.add_argument("--path-combination", choices=["sum", "min", "product"])
    parser.add_argument("--evidence-threshold", type=float)
    parser.add_argument("--evidence-target-temperature", type=float)
    parser.add_argument("--row-support-model")
    parser.add_argument("--row-support-top-l", type=int)
    parser.add_argument("--evidence-content-keys")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

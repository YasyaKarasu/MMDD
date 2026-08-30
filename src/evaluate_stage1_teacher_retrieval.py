#!/usr/bin/env python
"""Evaluate a Teacher on fixed raw ANN pools with full zero/one-hop retrieval."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.checkpoints import load_path_aggregation, load_teacher
from mmdd_stage1.data import load_target_examples
from mmdd_stage1.evaluation import evaluate_student_retrieval
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.retrieval import (
    checkpoint_fingerprint,
    load_corpus_ids,
    load_or_build_raw_embedding_indices,
)
from mmdd_stage1.selection import write_json
from mmdd_stage1.significance import paired_bootstrap_delta
from mmdd_stage1.teacher_rerank import TeacherRerankedANNIndices


def _modality_weight(value: str) -> tuple[str, float]:
    modality, separator, raw_weight = value.partition("=")
    if separator != "=" or modality not in {"text", "image"}:
        raise argparse.ArgumentTypeError("modality weights must use text=VALUE or image=VALUE")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("modality weight must be numeric") from exc
    if weight < 0:
        raise argparse.ArgumentTypeError("modality weight must be non-negative")
    return modality, weight


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


class _TeacherFeatureView:
    """Expose an explicit pooled-embedding fallback for retrieval-only scoring."""

    def __init__(
        self,
        store: FeatureStore,
        *,
        missing_policy: str,
        allowed_fallback_ids: set[str] | None = None,
    ) -> None:
        self.store = store
        self.missing_policy = missing_policy
        self.allowed_fallback_ids = allowed_fallback_ids or set()
        self.requested_ids: set[str] = set()
        self.allowed_fallback_ids_used: set[str] = set()
        self.unexpected_fallback_ids: set[str] = set()

    def get(self, object_id: str, *, include_hidden: bool = True) -> ObjectFeatures:
        features = self.store.get(object_id, include_hidden=include_hidden)
        if not include_hidden:
            return features
        self.requested_ids.add(object_id)
        if features.hidden_states is not None:
            return features
        if object_id in self.allowed_fallback_ids:
            self.allowed_fallback_ids_used.add(object_id)
        elif self.missing_policy == "error":
            return features
        else:
            self.unexpected_fallback_ids.add(object_id)
        return ObjectFeatures(
            object_id=features.object_id,
            object_type=features.object_type,
            embedding=features.embedding,
            hidden_states=features.embedding.unsqueeze(0),
            row_embeddings=features.row_embeddings,
        )

    def coverage(self) -> dict[str, Any]:
        requested = len(self.requested_ids)
        allowed_fallback = len(self.allowed_fallback_ids_used)
        unexpected_fallback = len(self.unexpected_fallback_ids)
        fallback = allowed_fallback + unexpected_fallback
        cached = requested - fallback
        return {
            "policy": self.missing_policy,
            "unique_objects_scored": requested,
            "cached_hidden_objects": cached,
            "pooled_embedding_fallback_objects": fallback,
            "allowed_pooled_embedding_fallback_objects": allowed_fallback,
            "unexpected_pooled_embedding_fallback_objects": unexpected_fallback,
            "cached_hidden_fraction": cached / requested if requested else 1.0,
        }

    def missing_ids(self) -> list[str]:
        return sorted(self.unexpected_fallback_ids)

    def allowed_ids_used(self) -> list[str]:
        return sorted(self.allowed_fallback_ids_used)


def _load_object_ids(paths: list[str]) -> set[str]:
    object_ids: set[str] = set()
    for value in paths:
        path = Path(value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                object_id = record.get("object_id") if isinstance(record, dict) else record
                if not isinstance(object_id, str) or not object_id:
                    raise ValueError(f"{path}:{line_number}: expected an object_id")
                object_ids.add(object_id)
    return object_ids


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.gamma <= 0 or args.gamma_evidence <= 0:
        raise ValueError("gamma and gamma-evidence must be positive")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    store = FeatureStore.from_path(
        Path(args.features), cache_size=args.feature_cache_size
    )
    teacher_path = Path(args.teacher_checkpoint)
    teacher = load_teacher(teacher_path, device)
    if store.teacher_dimension() != teacher.input_dim:
        raise ValueError("Teacher checkpoint does not match cached hidden states")
    corpus_path = Path(args.corpus)
    corpus_sha256 = checkpoint_fingerprint(corpus_path)
    raw_indices = load_or_build_raw_embedding_indices(
        store,
        load_corpus_ids(corpus_path, store),
        Path(args.raw_index_root),
        corpus_sha256=corpus_sha256,
        batch_size=args.index_batch_size,
        m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
    )
    allowed_fallback_paths = [Path(value) for value in args.allowed_pooled_fallback_ids]
    teacher_store = _TeacherFeatureView(
        store,
        missing_policy=args.missing_teacher_feature_policy,
        allowed_fallback_ids=_load_object_ids(args.allowed_pooled_fallback_ids),
    )
    teacher_indices = TeacherRerankedANNIndices(
        raw_indices,
        teacher,
        teacher_store,
        device=device,
        batch_size=args.teacher_batch_size,
    )
    examples = [
        example
        for value in args.dev_data
        for example in load_target_examples(
            Path(value), split="dev", dataset_name=Path(value).stem
        )
    ]
    aggregation, top_k = load_path_aggregation(teacher_path)
    evidence_modality_weights = dict(args.evidence_modality_weights)
    common = {
        "recall_ks": args.recall_ks,
        "gamma": args.gamma,
        "gamma_evidence": args.gamma_evidence,
        "evidence_types": args.evidence_types,
        "evidence_aggregation": aggregation,
        "evidence_top_k": top_k,
        "evidence_modality_weights": evidence_modality_weights,
        "fusion_mode": "weighted_rrf",
        "evidence_weight": 0.05,
        "return_per_query": True,
    }
    raw_start = time.perf_counter()
    raw_metrics = evaluate_student_retrieval(examples, raw_indices, **common)
    raw_seconds = time.perf_counter() - raw_start
    teacher_start = time.perf_counter()
    teacher_metrics = evaluate_student_retrieval(examples, teacher_indices, **common)
    teacher_seconds = time.perf_counter() - teacher_start
    paired = paired_bootstrap_delta(
        teacher_metrics["per_query"]["fused"]["recall@10"],
        raw_metrics["per_query"]["fused"]["recall@10"],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing_path = output_dir / "missing_teacher_ids.jsonl"
    missing_path.write_text(
        "".join(
            json.dumps({"object_id": object_id}, ensure_ascii=False) + "\n"
            for object_id in teacher_store.missing_ids()
        ),
        encoding="utf-8",
    )
    allowed_path = output_dir / "allowed_pooled_fallback_ids.jsonl"
    allowed_path.write_text(
        "".join(
            json.dumps({"object_id": object_id}, ensure_ascii=False) + "\n"
            for object_id in teacher_store.allowed_ids_used()
        ),
        encoding="utf-8",
    )
    feature_coverage = teacher_store.coverage()
    feature_coverage["missing_teacher_ids"] = str(missing_path.resolve())
    feature_coverage["allowed_pooled_fallback_ids"] = str(allowed_path.resolve())
    payload = {
        "format_version": 1,
        "teacher_checkpoint": str(teacher_path.resolve()),
        "teacher_checkpoint_sha256": checkpoint_fingerprint(teacher_path),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": corpus_sha256,
        "parameters": {
            "recall_ks": list(args.recall_ks),
            "gamma": args.gamma,
            "gamma_evidence": args.gamma_evidence,
            "fusion": "weighted_rrf",
            "evidence_weight": 0.05,
            "evidence_types": list(args.evidence_types),
            "evidence_modality_weights": evidence_modality_weights,
            "evidence_aggregation": aggregation,
            "evidence_top_k": top_k,
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "missing_teacher_feature_policy": args.missing_teacher_feature_policy,
            "allowed_pooled_fallback_id_files": [
                str(path.resolve()) for path in allowed_fallback_paths
            ],
        },
        "raw": {"metrics": raw_metrics, "total_seconds": raw_seconds},
        "teacher": {
            "metrics": teacher_metrics,
            "total_seconds": teacher_seconds,
            "vs_raw_fused_recall@10": paired,
            "feature_coverage": feature_coverage,
        },
    }
    write_json(output_dir / "metrics.json", payload)
    lines = [
        "# Full Teacher zero/one-hop retrieval",
        "",
        "Each Teacher edge reranks the same fixed raw ANN edge pool.",
        "",
        "| System | Fused R@10 | Direct R@10 | Evidence R@10 | Coverage@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, metrics in (("Raw", raw_metrics), ("Teacher", teacher_metrics)):
        lines.append(
            f"| {label} | {_percent(metrics['recall@10'])} | "
            f"{_percent(metrics['direct']['recall@10'])} | "
            f"{_percent(metrics['evidence']['recall@10'])} | "
            f"{_percent(metrics['positive_evidence_path_coverage@10'])} |"
        )
    lines.extend(
        [
            "",
            "Teacher fused R@10 delta vs raw / 95% CI: "
            f"{_percent(paired['mean'])} "
            f"[{_percent(paired['ci_low'])}, {_percent(paired['ci_high'])}].",
            "",
            "Teacher feature coverage: "
            f"{_percent(payload['teacher']['feature_coverage']['cached_hidden_fraction'])} "
            "cached hidden states; "
            f"{payload['teacher']['feature_coverage']['allowed_pooled_embedding_fallback_objects']} "
            "audited objects used the allowed pooled-embedding fallback; "
            f"{payload['teacher']['feature_coverage']['unexpected_pooled_embedding_fallback_objects']} "
            "unexpected objects used fallback.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--dev-data", nargs="+", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--raw-index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-batch-size", type=int, default=16)
    parser.add_argument("--feature-cache-size", type=int, default=60_000)
    parser.add_argument("--index-batch-size", type=int, default=1024)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--recall-ks", type=_positive_ints, default=(10, 20, 30, 40, 50))
    parser.add_argument("--gamma", type=int, default=10)
    parser.add_argument("--gamma-evidence", type=int, default=2)
    parser.add_argument(
        "--evidence-types",
        nargs="+",
        choices=["text", "image"],
        default=["text", "image"],
    )
    parser.add_argument(
        "--evidence-modality-weights",
        nargs="*",
        type=_modality_weight,
        default=[],
        metavar="MODALITY=WEIGHT",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument(
        "--missing-teacher-feature-policy",
        choices=["error", "pooled_embedding"],
        default="error",
        help=(
            "How retrieval-only Teacher scoring handles raw-pool objects without "
            "cached hidden states. Training remains strict."
        ),
    )
    parser.add_argument(
        "--allowed-pooled-fallback-ids",
        nargs="*",
        default=[],
        metavar="JSONL",
        help=(
            "JSONL object-id files allowed to use pooled embeddings even when "
            "the general missing-feature policy is error."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

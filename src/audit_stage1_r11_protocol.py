#!/usr/bin/env python
"""Audit R11 supervision provenance and in-batch positive handling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.data import EdgeExample, load_edge_examples
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import (
    edge_positive_key, global_edge_positive_ids, score_edge_batch_in_batch,
)
from mmdd_stage1.training import sample_mixed_epoch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _expansion_epoch(
    examples: list[EdgeExample],
    *,
    seed: int,
    epoch: int,
    batch_size: int,
    cap: int,
) -> dict[str, int]:
    rng = random.Random(seed)
    sampled: list[EdgeExample] = []
    for _current_epoch in range(epoch + 1):
        sampled, _sources = sample_mixed_epoch(
            examples,
            (),
            rng,
            hard_fraction=0.5,
            dataset_sampling_alpha=0.0,
        )
    known = global_edge_positive_ids(examples)
    audit = Counter({
        "legacy_known_positive_as_negative": 0,
        "fixed_known_positive_as_negative": 0,
    })
    model = StudentJoinabilityModel(1, 1, initialization="identity")
    for start in range(0, len(sampled), batch_size):
        batch = sampled[start : start + batch_size]
        step = start // batch_size + 1
        types: dict[str, str] = {}
        for example in batch:
            types[example.query_id] = str(example.source_type)
            types.update((value, str(example.destination_type))
                         for value in example.candidate_ids)
        # Values do not affect expansion/masks; use real IDs with scalar features.
        store = FeatureStore({
            value: ObjectFeatures(value, kind, torch.zeros(1))
            for value, kind in types.items()
        })
        audit["lists"] += len(batch)
        for label, global_mask in (("legacy", False), ("fixed", True)):
            with torch.inference_mode():
                scores = score_edge_batch_in_batch(
                    model, batch, store, torch.device("cpu"),
                    max_negatives=cap, known_positive_ids=known,
                    use_global_positive_mask=global_mask,
                    sampling_seed=seed,
                    sampling_context=f"epoch={epoch}:step={step}",
                )
            for index, (example, candidates) in enumerate(
                zip(batch, scores.candidate_ids)
            ):
                errors = sum(
                    candidate in known[edge_positive_key(example)]
                    and bool(scores.candidate_mask[index, column])
                    and not bool(scores.positive_mask[index, column])
                    for column, candidate in enumerate(candidates)
                )
                audit[f"{label}_known_positive_as_negative"] += errors
                audit[f"{label}_affected_lists"] += int(bool(errors))
    return dict(audit)


def _label_summary(examples: list[EdgeExample]) -> dict[str, Any]:
    relations: dict[str, Counter[str]] = defaultdict(Counter)
    conflicts = 0
    omitted_original_positives = 0
    for example in examples:
        relation = f"{example.source_type}_to_{example.destination_type}"
        counts = relations[relation]
        labels = example.confirmed_labels or (None,) * len(example.candidate_ids)
        positives = set(example.positive_ids)
        omitted_original_positives += int(
            example.candidate_ids[example.positive_index] not in positives
        )
        for candidate_id, label in zip(example.candidate_ids, labels):
            counts[
                "confirmed_positive"
                if label == 1
                else "confirmed_negative"
                if label == 0
                else "unknown"
            ] += 1
            conflicts += int(candidate_id in positives and label == 0)
        counts["lists"] += 1
        counts["ranking_positives"] += len(positives)
    return {
        "by_relation": {
            key: dict(sorted(values.items()))
            for key, values in sorted(relations.items())
        },
        "ranking_positive_confirmed_negative_conflicts": conflicts,
        "designated_positives_omitted_from_positive_ids": (
            omitted_original_positives
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    supervision_dir = Path(args.supervision_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_manifest = json.loads(
        (supervision_dir / "manifest.json").read_text(encoding="utf-8")
    )
    buckets = tuple(source_manifest["outputs"])
    files = {
        bucket: {
            "edge": supervision_dir / f"edge_lists.{bucket}.jsonl",
            "target": supervision_dir / f"target_lists.{bucket}.jsonl",
        }
        for bucket in buckets
    }
    train_edges = load_edge_examples(files["train_fit"]["edge"], split="train")
    expansion = [
        _expansion_epoch(
            train_edges,
            seed=args.seed,
            epoch=epoch,
            batch_size=args.batch_size,
            cap=args.cap,
        )
        for epoch in range(args.epochs)
    ]
    label_summary = _label_summary(train_edges)
    status = "pass" if (
        all(row["fixed_known_positive_as_negative"] == 0 for row in expansion)
        and label_summary["ranking_positive_confirmed_negative_conflicts"] == 0
        and label_summary["designated_positives_omitted_from_positive_ids"] == 0
    ) else "fail"

    inputs = {
        "format_version": 1,
        "dataset": {
            "root": source_manifest["dataset_root"],
            "manifest_sha256": source_manifest["dataset_manifest_sha256"],
        },
        "features": _fingerprint(Path(args.features) / "manifest.jsonl"),
        "feature_metadata": _fingerprint(Path(args.features) / "metadata.json"),
        "pca": _fingerprint(Path(args.pca)),
        "corpus": _fingerprint(Path(args.corpus)),
        "supervision_manifest": _fingerprint(
            supervision_dir / "manifest.json"
        ),
        "transductive_retrieval": (
            "The shared unlabeled EntiTables-v9 lake is indexed; only "
            "train-fit qrels/recoveries provide P/R or Teacher supervision."
        ),
        "independent_confirmation": "not_available",
    }
    splits = {
        "format_version": 1,
        "seed": args.seed,
        "source_group_key": "source_table_id",
        "counts": {
            bucket: source_manifest["outputs"][bucket]
            for bucket in buckets
        },
        "calibration": source_manifest["calibration"],
        "uses": {
            "train_fit": "P/R and fresh Teacher supervision",
            "cal_fit": "score mappings only",
            "cal_check": "calibration evaluation only",
            "dev": "checkpoint and configuration selection",
            "r10_test_regression": "adaptive historical regression only",
        },
    }
    provenance = {
        "format_version": 1,
        "status": status,
        "policy": source_manifest["supervision_policy"],
        "train_fit": label_summary,
        "in_batch_replay": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "cap": args.cap,
            "epochs": expansion,
            "fixed_total_known_positive_as_negative": sum(
                row["fixed_known_positive_as_negative"] for row in expansion
            ),
            "legacy_total_known_positive_as_negative": sum(
                row["legacy_known_positive_as_negative"] for row in expansion
            ),
        },
        "bce_policy": (
            "Only confirmed_labels 0/1 enter BCE; null candidates remain unknown."
        ),
    }
    selection = {
        "format_version": 1,
        "numeric_tolerance": 1e-12,
        "training_checkpoint_order": [
            {"metric": "evidence_funnel.valid_pool_count", "maximize": True},
            {"metric": "evidence_funnel.row_b", "maximize": True},
            {"metric": "evidence_funnel.valid_b_count", "maximize": True},
            {"metric": "direct.recall@10", "maximize": True},
            {"metric": "step", "maximize": False},
        ],
        "final_quality_constraint": {
            "overall_recall_at_10_max_drop": 0.02,
            "implicit_recall_at_10_max_drop": 0.02,
            "reference": "same-model union-direct-only",
        },
        "final_order": [
            "row_support_count",
            "multi_row_2_count",
            "valid_path_count",
            "recall@10",
            "measured_cost",
            "config_id",
        ],
        "historical_r10_replay": "single primary metric; ties retain earliest",
    }
    _write_json(output_dir / "inputs.json", inputs)
    _write_json(output_dir / "splits.json", splits)
    _write_json(output_dir / "label_provenance.json", provenance)
    _write_json(output_dir / "selection_spec.json", selection)
    summary = {
        "status": status,
        "fixed_known_positive_as_negative": provenance["in_batch_replay"][
            "fixed_total_known_positive_as_negative"
        ],
        "legacy_known_positive_as_negative": provenance["in_batch_replay"][
            "legacy_total_known_positive_as_negative"
        ],
        "train_edge_lists": len(train_edges),
    }
    print(json.dumps(summary, indent=2))
    if status != "pass":
        raise ValueError("R11 Task-A protocol audit failed")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervision-dir", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cap", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

#!/usr/bin/env python
"""Run and evaluate Stage-2 for every query in one gated retrieval split."""

from __future__ import annotations

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import torch

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.feature_cache import FeatureStore
from mmdd_stage1.export import validate_stage2_gate
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import (
    direct_target_ids,
    iter_retrieval_results,
    load_stage2_objects,
    validate_retrieval_path_budget,
)
from mmdd_stage2.pipeline import Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import build_evidence_bundles


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _local_column(target: dict[str, Any], source_column: int) -> int | None:
    for column in target["columns"]:
        if int(column.get("source_column_index", column["column_index"])) == source_column:
            return int(column["column_index"])
    return None


def _recall(order: list[str], relevant: set[str], k: int) -> float:
    return len(set(order[:k]) & relevant) / len(relevant) if relevant else 0.0


def _metrics(
    outputs: list[dict[str, Any]],
    retrieval_by_query: dict[str, dict[str, Any]],
    qrels_by_query: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    stage1_recall: dict[int, list[float]] = defaultdict(list)
    final_recall: dict[int, list[float]] = defaultdict(list)
    recoverable_pairs = 0
    retrieved_recoverable = 0
    column_predictions = 0
    correct_columns = 0
    selected_for_recovery = 0
    evidence_verified = 0
    evidence_joinable = 0
    final_joinable = 0
    evidence_coverages: list[float] = []
    recovered_nonempty_rows = 0
    recovered_rows = 0

    for output in outputs:
        query_id = str(output["query_id"])
        relevant = {str(row["target_table_id"]) for row in qrels_by_query.get(query_id, [])}
        stage1_order = [
            str(row["target_id"])
            for row in retrieval_by_query[query_id]["results"]
        ]
        final_candidates = output["reranked_candidates"] + output["unattempted_candidates"]
        final_order = [str(row["target_id"]) for row in final_candidates]
        if relevant:
            for k in (1, 10, 20, 50):
                stage1_recall[k].append(_recall(stage1_order, relevant, k))
                final_recall[k].append(_recall(final_order, relevant, k))

        candidate_by_target = {str(row["target_id"]): row for row in final_candidates}
        gold_by_target = {
            str(row["target_id"]): row for row in output.get("gold_targets", [])
        }
        for qrel in qrels_by_query.get(query_id, []):
            if qrel.get("reason") != "model_recoverable_join_column":
                continue
            recoverable_pairs += 1
            target_id = str(qrel["target_table_id"])
            candidate = candidate_by_target.get(target_id)
            if candidate is None:
                continue
            retrieved_recoverable += 1
            selection = candidate.get("selection")
            expected = gold_by_target.get(target_id, {}).get("expected_local_column")
            if selection is not None and expected is not None:
                column_predictions += 1
                correct_columns += int(int(selection["column_index"]) == int(expected))
            selected_for_recovery += int(bool(candidate.get("selected_for_recovery")))
            evidence = candidate.get("branches", {}).get("evidence")
            if evidence and evidence.get("status") == "verified":
                evidence_verified += 1
                check = evidence["verification"]
                evidence_coverages.append(float(check["coverage"]))
                evidence_joinable += int(bool(check["joinable"]))
                rows = evidence.get("rows", [])
                recovered_rows += len(rows)
                recovered_nonempty_rows += sum(bool(str(row.get("value", "")).strip()) for row in rows)
            check = candidate.get("verification")
            final_joinable += int(bool(check and check.get("joinable")))

    return {
        "status": "COMPLETE" if len(outputs) == len(retrieval_by_query) else "PARTIAL",
        "queries_completed": len(outputs),
        "queries_expected": len(retrieval_by_query),
        "retrieval": {
            f"stage1_recall@{k}": fmean(values) if values else None
            for k, values in sorted(stage1_recall.items())
        },
        "final_join_verification": {
            f"stage2_recall@{k}": fmean(values) if values else None
            for k, values in sorted(final_recall.items())
        },
        "evidence_mechanism": {
            "recoverable_gold_pairs": recoverable_pairs,
            "retrieved_recoverable_pairs": retrieved_recoverable,
            "column_predictions": column_predictions,
            "correct_columns": correct_columns,
            "column_accuracy": correct_columns / column_predictions if column_predictions else None,
            "selected_for_recovery": selected_for_recovery,
            "evidence_verified": evidence_verified,
            "evidence_joinable": evidence_joinable,
            "mean_value_recovery_coverage": fmean(evidence_coverages) if evidence_coverages else None,
            "nonempty_generated_row_rate": (
                recovered_nonempty_rows / recovered_rows if recovered_rows else None
            ),
            "final_joinable_gold_pairs": final_joinable,
        },
    }


def run(args: argparse.Namespace) -> None:
    validate_stage2_gate(args.stage1_gate, [args.retrieval_results])
    records = list(iter_retrieval_results(args.retrieval_results))
    retrieval_by_query = {str(row["query_id"]): row for row in records}
    if len(retrieval_by_query) != len(records):
        raise ValueError("Retrieval file contains duplicate query IDs")
    splits = {str(row.get("split")) for row in records}
    if len(splits) != 1:
        raise ValueError("Batch retrieval must contain exactly one split")
    split = next(iter(splits))

    qrels_by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qrel in iter_dataset_artifact(args.dataset_root, "qrels"):
        if qrel.get("split", "train") == split:
            qrels_by_query[str(qrel["query_table_id"])].append(qrel)

    outputs = []
    if args.output.is_file():
        outputs = list(iter_retrieval_results(args.output))
    completed = {str(row["query_id"]) for row in outputs}
    if not completed <= retrieval_by_query.keys():
        raise ValueError("Resume output contains queries outside the retrieval split")

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
    backend.reader_batch_size = args.reader_batch_size
    scorer = load_candidate_scorer(
        args.scorer_checkpoint,
        torch.device("cpu"),
        expected_model_dir=args.model_dir,
    ).to(backend.device)
    router = SimilarityEvidenceRouter(FeatureStore.from_path(args.stage1_features))
    verifier = Stage2Verifier(
        backend,
        scorer,
        evidence_router=router,
        similarity_threshold=args.similarity_threshold,
        min_row_coverage=args.min_row_coverage,
        similarity_batch_size=args.similarity_batch_size,
    )

    ordered = sorted(records, key=lambda row: str(row["query_id"]).encode("utf-8"))
    for index, record in enumerate(ordered, 1):
        query_id = str(record["query_id"])
        if query_id in completed:
            continue
        validate_retrieval_path_budget(
            record,
            max_targets=args.input_candidate_budget,
            top_k_evidence=args.top_k_evidence,
        )
        results = record["results"][: args.input_candidate_budget]
        bundles = build_evidence_bundles(results, top_k_evidence=args.top_k_evidence)
        objects = load_stage2_objects(
            args.dataset_root,
            query_id,
            bundles,
            extra_target_ids=direct_target_ids(results),
        )
        payload = verifier.verify(
            objects.query,
            results,
            objects.targets,
            objects.evidence,
            recovery_budget=args.recovery_budget,
            top_k_evidence=args.top_k_evidence,
        ).to_dict()
        payload["split"] = split
        payload["input_candidate_budget"] = args.input_candidate_budget
        payload["gold_targets"] = []
        for qrel in qrels_by_query.get(query_id, []):
            target_id = str(qrel["target_table_id"])
            source_column = int(qrel["join_attribute"]["source_column_index"])
            payload["gold_targets"].append(
                {
                    "target_id": target_id,
                    "reason": qrel.get("reason"),
                    "expected_source_column": source_column,
                    "expected_local_column": (
                        _local_column(objects.targets[target_id], source_column)
                        if target_id in objects.targets else None
                    ),
                }
            )
        outputs.append(payload)
        outputs.sort(key=lambda row: str(row["query_id"]).encode("utf-8"))
        _write_jsonl(args.output, outputs)
        metrics = _metrics(outputs, retrieval_by_query, qrels_by_query)
        _write_json(args.metrics_output, metrics)
        print(
            json.dumps(
                {"query": query_id, "completed": len(outputs), "total": len(records)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = _metrics(outputs, retrieval_by_query, qrels_by_query)
    _write_json(args.metrics_output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--stage1-gate", type=Path, required=True)
    parser.add_argument("--scorer-checkpoint", type=Path, required=True)
    parser.add_argument("--stage1-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("hf_models/Qwen3.5-9B"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--focus-start-layer", type=int, default=14)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--input-candidate-budget", type=int, default=50)
    parser.add_argument("--recovery-budget", type=int, default=20)
    parser.add_argument("--max-text-evidence-tokens", type=int, default=1024)
    parser.add_argument("--text-overlap-tokens", type=int, default=128)
    parser.add_argument("--max-span-tokens", type=int, default=192)
    parser.add_argument("--roi-candidates", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-embedding-tokens", type=int, default=128)
    parser.add_argument("--reader-batch-size", type=int, default=2)
    parser.add_argument("--similarity-batch-size", type=int, default=1024)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--min-row-coverage", type=float, default=0.6)
    args = parser.parse_args()
    if min(args.input_candidate_budget, args.top_k_evidence, args.reader_batch_size) <= 0:
        parser.error("candidate and evidence budgets must be positive")
    if args.recovery_budget < 0:
        parser.error("--recovery-budget must be non-negative")
    return args


if __name__ == "__main__":
    run(parse_args())

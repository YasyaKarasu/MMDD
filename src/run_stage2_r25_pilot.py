#!/usr/bin/env python3
"""Run the fixed R25 Stage-2 Real/NoE-fill pilot with one loaded Qwen backend."""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path

import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import (
    column_values,
    direct_target_ids,
    load_stage2_objects,
    row_values,
    validate_retrieval_path_budget,
)
from mmdd_stage2.pipeline import (
    CandidateResult,
    EvidenceVerification,
    LocalizedEvidence,
    RowPrediction,
    Stage2Result,
    Stage2Verifier,
    joinability_sort_key,
)
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import build_evidence_bundles


def _write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _no_evidence_recovery(
    verifier: Stage2Verifier,
    query: dict,
    target: dict,
    selection,
    row_mask: set[int],
) -> EvidenceVerification:
    """Run the same generator only on Real's frozen row opportunities."""

    empty = LocalizedEvidence(
        evidence_id="__empty_evidence__",
        evidence_type="text",
        text="",
        text_span_relevance=0.0,
    )
    predictions = []
    for row in query["rows"]:
        row_id = int(row["row_id"])
        value = ""
        if row_id in row_mask:
            value = verifier.backend.generate_value(
                row_values(query, row),
                attribute_name=selection.column_name,
                evidence=empty,
            )
        predictions.append(RowPrediction(row_id=row_id, value=value, evidence=None))
    generated = [row.value for row in predictions]
    target_values = column_values(target, selection.column_index)
    return EvidenceVerification(
        selection,
        tuple(predictions),
        verifier._semantic_check(generated, target_values),
    )


def _assemble_result(
    query_id: str,
    retrieval_results: list[dict],
    bundles,
    scores,
    direct,
    recovered,
    recovery_budget: int,
) -> Stage2Result:
    bundle_by_id = {bundle.target_id: bundle for bundle in bundles}
    score_by_id = {item.selection.target_id: item for item in scores}
    candidates = []
    for stage1_rank, result in enumerate(retrieval_results, 1):
        target_id = str(result["target_id"])
        bundle = bundle_by_id.get(target_id)
        table_prior = bundle.retrieval_score if bundle else result.get("stage2_table_score")
        candidates.append(
            CandidateResult(
                target_id=target_id,
                stage1_rank=stage1_rank,
                stage1_score=float(result["score"]) if result.get("score") is not None else None,
                table_prior=float(table_prior) if table_prior is not None else None,
                bundle=bundle,
                scores=score_by_id.get(target_id),
                selected_for_recovery=target_id in recovered,
                direct=direct.get(target_id),
                evidence=recovered.get(target_id),
            )
        )
    verified = sorted(
        (candidate for candidate in candidates if candidate.semantic_joinability is not None),
        key=lambda candidate: joinability_sort_key(candidate.semantic_joinability, candidate.stage1_rank),
    )
    return Stage2Result(
        query_id=query_id,
        recovery_budget=recovery_budget,
        reranked_candidates=tuple(replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(verified, 1)),
        unattempted_candidates=tuple(candidate for candidate in candidates if candidate.semantic_joinability is None),
    )


def _verify_paired(
    verifier: Stage2Verifier,
    query: dict,
    results: list[dict],
    targets: dict,
    evidence: dict,
    *,
    recovery_budget: int,
    top_k_evidence: int,
) -> tuple[Stage2Result, Stage2Result, dict]:
    bundles = build_evidence_bundles(results, top_k_evidence=top_k_evidence)
    scores = verifier.score_candidates(query, bundles, targets, evidence)
    selected = sorted(
        (item for item in scores if item.accepted),
        key=lambda item: -item.recovery_priority,
    )[:recovery_budget]
    selected_ids = {item.selection.target_id for item in selected}
    direct = {
        item.target_id: item
        for item in verifier.verify_direct(query, targets, direct_target_ids(results))
    }
    real = {}
    no_evidence = {}
    row_masks = {}
    for bundle, item in zip(bundles, scores, strict=True):
        target_id = bundle.target_id
        if target_id not in selected_ids:
            continue
        real[target_id] = verifier.recover_candidate(
            query, bundle, item.selection, targets, evidence
        )
        row_mask = {
            prediction.row_id
            for prediction in real[target_id].rows
            if prediction.evidence is not None
        }
        row_masks[target_id] = sorted(row_mask)
        no_evidence[target_id] = _no_evidence_recovery(
            verifier,
            query,
            targets[target_id],
            item.selection,
            row_mask,
        )
    signature = {
        "candidate_ids": [str(item["target_id"]) for item in results],
        "selected_target_ids": sorted(selected_ids),
        "predicted_columns": {
            item.selection.target_id: {
                "column_index": item.selection.column_index,
                "column_name": item.selection.column_name,
            }
            for item in selected
        },
        "row_masks": row_masks,
    }
    query_id = str(query["table_id"])
    return (
        _assemble_result(query_id, results, bundles, scores, direct, real, recovery_budget),
        _assemble_result(query_id, results, bundles, scores, direct, no_evidence, recovery_budget),
        signature,
    )


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    stage2 = root / "work/stage1_optimization_r25_final_20260914/stage2"
    pilot_manifest = stage2 / "pilot_queries_64.jsonl"
    retrieval_path = stage2 / "r25_b13_pilot_retrieval.jsonl"
    scorer_path = stage2 / "r25_b13_column_scorer.pt"
    dataset_root = Path(args.dataset_root).resolve()
    if not all(path.is_file() for path in (pilot_manifest, retrieval_path, scorer_path)):
        raise FileNotFoundError("R25 Stage2 pilot inputs or trained scorer are missing")
    pilot_ids = list(dict.fromkeys(json.loads(line)["model_input"]["query_id"] for line in pilot_manifest.open(encoding="utf-8") if line.strip()))
    retrieval = {str(row["query_id"]): row for row in (json.loads(line) for line in retrieval_path.open(encoding="utf-8") if line.strip())}
    missing = set(pilot_ids) - retrieval.keys()
    if missing:
        raise KeyError(f"Pilot retrieval is missing query IDs: {sorted(missing)[:3]}")
    device = torch.device(args.device)
    scorer = load_candidate_scorer(scorer_path, torch.device("cpu"), expected_model_dir=Path(args.model_dir))
    backend = QwenStage2Backend(Path(args.model_dir), device=args.device, dtype=args.dtype, max_new_tokens=args.max_new_tokens)
    scorer.to(backend.device)
    router = SimilarityEvidenceRouter(FeatureStore.from_path(root / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"))
    verifier = Stage2Verifier(backend, scorer, evidence_router=router, similarity_threshold=args.similarity_threshold, min_row_coverage=args.min_row_coverage)
    generators = (args.generator,) if args.generator != "all" else ("B13", "raw-Qwen")
    output = Path(args.output).resolve() if args.output else stage2 / "S2_PILOT_RESULTS.jsonl.gz"
    rows = []
    if output.is_file():
        with gzip.open(output, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if any(row.get("generator_id") not in generators for row in rows):
            raise ValueError(f"{output}: resume rows belong to another generator")
    completed_queries = {
        query_id
        for query_id in pilot_ids
        if {(row["condition"]) for row in rows if row.get("query_id") == query_id} == {"Real", "NoE-fill"}
    }
    for generator in generators:
        for index, query_id in enumerate(pilot_ids, 1):
            if query_id in completed_queries:
                continue
            source = retrieval[query_id]
            results = [dict(item) for item in source["results"][: args.candidate_budget]]
            validate_retrieval_path_budget({"results": results}, max_targets=len(results), top_k_evidence=args.top_k_evidence)
            bundles = build_evidence_bundles(results, top_k_evidence=args.top_k_evidence)
            try:
                objects = load_stage2_objects(dataset_root, query_id, bundles, extra_target_ids=direct_target_ids(results))
                real, no_evidence, opportunity = _verify_paired(
                    verifier,
                    objects.query,
                    results,
                    objects.targets,
                    objects.evidence,
                    recovery_budget=args.recovery_budget,
                    top_k_evidence=args.top_k_evidence,
                )
                outcomes = (("Real", real.to_dict(), None), ("NoE-fill", no_evidence.to_dict(), None))
            except Exception as exc:  # retain both pre-registered rows on a query-level fault
                failure = {"query_id": query_id, "input_candidate_count": len(results), "recovery_budget": args.recovery_budget, "reranked_candidates": [], "unattempted_candidates": []}
                opportunity = {"candidate_ids": [str(item["target_id"]) for item in results], "selected_target_ids": [], "predicted_columns": {}, "row_masks": {}}
                error = f"{type(exc).__name__}: {exc}"
                outcomes = (("Real", failure, error), ("NoE-fill", failure, error))
            for condition, payload, error in outcomes:
                rows.append({"generator_id": generator, "condition": condition, "query_id": query_id, "candidate_budget": args.candidate_budget, "effective_candidate_count": len(results), "recovery_budget": args.recovery_budget, "status": "failed" if error else "complete", "error": error, "opportunity_signature": opportunity, "result": payload})
            _write_rows(output, rows)
            torch.cuda.empty_cache()
            print(json.dumps({"stage": "S2", "generator": generator, "query": index, "total": len(pilot_ids), "statuses": {condition: "failed" if error else "complete" for condition, _payload, error in outcomes}}), flush=True)
    complete = sum(row["status"] == "complete" for row in rows)
    receipt = {"format_version": 1, "module": "S2", "status": "complete" if complete == len(rows) else "partial", "generators": list(generators), "conditions": ["Real", "NoE-fill"], "queries": len(pilot_ids), "rows": len(rows), "complete_rows": complete, "failed_rows": len(rows) - complete, "candidate_budget": args.candidate_budget, "recovery_budget": args.recovery_budget, "output": str(output.resolve())}
    receipt_name = "S2_COMPLETION_RECEIPT.json" if args.generator == "all" else f"S2_{args.generator.replace('-', '_')}_RECEIPT.json"
    (stage2 / receipt_name).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-dir", default="hf_models/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--generator", choices=["all", "B13", "raw-Qwen"], default="all")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--candidate-budget", type=int, default=50)
    parser.add_argument("--recovery-budget", type=int, default=10)
    parser.add_argument("--top-k-evidence", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--min-row-coverage", type=float, default=0.6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Locked C18 pilot with shared localization and crop/crop+original/NoE comparisons."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import torch

from mmdd_stage1.features import FeatureStore
from mmdd_stage1.r26_metrics import population_metrics
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.data import column_values, direct_target_ids, load_stage2_index, row_values
from mmdd_stage2.pipeline import CandidateResult, EvidenceVerification, RowPrediction, Stage2Verifier, joinability_sort_key
from mmdd_stage2.r26_generation import R26QwenBackend, VALUE_PROMPT
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.verifier import build_evidence_bundles
from audit_stage2_r26_engineering import DATASET
from prepare_stage1_r26 import ROOT, OUT, file_record, stable_sha
from run_stage1_r21 import read_rows, write_rows
from run_stage1_r25 import _json, out as r25_out, sha256

CONDITIONS = ("Real-crop", "Real-crop+original", "NoE-fill")
KS = (1, 3, 5, 7, 9)


def final_candidates(results, bundles, scores, direct, recovered) -> list[dict]:
    bundle_map = {b.target_id: b for b in bundles}
    score_map = {s.selection.target_id: s for s in scores}
    candidates = [CandidateResult(r["target_id"], i, r["score"], r["stage2_table_score"], bundle_map.get(r["target_id"]),
                                  score_map.get(r["target_id"]), r["target_id"] in recovered,
                                  direct.get(r["target_id"]), recovered.get(r["target_id"])) for i, r in enumerate(results, 1)]
    # Retain all C18 candidates. No-recovery targets keep their fixed direct
    # verification if present, otherwise zero coverage/similarity + Stage1 rank.
    candidates.sort(key=lambda c: joinability_sort_key(c.semantic_joinability, c.stage1_rank)
                    if c.semantic_joinability is not None else (0., 0., c.stage1_rank))
    return [c.to_dict() for c in candidates]


@torch.inference_mode()
def run(generator: str) -> dict:
    torch.set_num_threads(2)
    engineering = json.loads((OUT / "stage2/engineering/SUMMARY.json").read_text())
    if not engineering["generation_interface_valid"]:
        raise ValueError("Stage2 generation correctness gate failed")
    directory = OUT / "stage2/pilot" / generator
    directory.mkdir(parents=True, exist_ok=True)
    source = OUT / "stage2/inputs" / generator / "retrieval.jsonl"
    inputs = list(read_rows(source))
    if len(inputs) != 64 or any(len(r["results"]) > 18 for r in inputs):
        raise ValueError("Stage2 input population/budget mismatch")
    scorer_path = r25_out(ROOT) / "stage2/r25_b13_column_scorer.pt"
    model_dir = ROOT / "hf_models/Qwen3.5-9B"
    signature = {"input": file_record(source), "scorer": file_record(scorer_path), "model_config": file_record(model_dir / "config.json"),
                 "template": file_record(model_dir / "chat_template.jinja"), "prompt_sha": stable_sha(VALUE_PROMPT),
                 "code": {name: sha256(ROOT / "src" / name) for name in ("run_stage2_r26.py", "mmdd_stage2/r26_generation.py", "mmdd_stage1/r26_metrics.py")},
                 "conditions": CONDITIONS, "candidate_budget": 18, "recovery_budget": 10, "recall_ks": KS}
    signature_hash = stable_sha(signature)
    outputs_path = directory / "results.jsonl"
    outputs = list(read_rows(outputs_path)) if outputs_path.exists() else []
    if any(r["run_signature"] != signature_hash for r in outputs):
        raise ValueError("Stage2 resume identity changed")
    _json(directory / "RUN_IDENTITY.json", signature)
    completed = {(r["generator_id"], r["query_id"], r["condition"], r["input_pool_sha"], r["run_signature"]) for r in outputs}
    bundles_by_q = {r["query_id"]: build_evidence_bundles(r["results"], top_k_evidence=4) for r in inputs}
    objects = load_stage2_index(DATASET, query_ids={r["query_id"] for r in inputs},
        target_ids={t["target_id"] for r in inputs for t in r["results"]},
        evidence_ids={e for bundles in bundles_by_q.values() for b in bundles for e in b.evidence_ids})
    scorer = load_candidate_scorer(scorer_path, torch.device("cpu"), expected_model_dir=model_dir)
    backend = R26QwenBackend(model_dir, device="cuda:1", dtype="bf16")
    scorer.to(backend.device)
    raw_path = directory / "raw_completions.jsonl"
    if raw_path.exists():
        backend.generation_records = list(read_rows(raw_path))
    router = SimilarityEvidenceRouter(FeatureStore.from_path(ROOT / "work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b"))
    verifier = Stage2Verifier(backend, scorer, evidence_router=router)
    for index, source_row in enumerate(inputs, 1):
        qid, pool_sha = source_row["query_id"], source_row["input_pool_sha"]
        if all((generator, qid, c, pool_sha, signature_hash) in completed for c in CONDITIONS):
            continue
        results, query = source_row["results"], objects.queries[qid]
        bundles = bundles_by_q[qid]
        shared_error = None
        try:
            scores = verifier.score_candidates(query, bundles, objects.targets, objects.evidence)
            selected = sorted((s for s in scores if s.accepted), key=lambda s: -s.recovery_priority)[:10]
            direct = {d.target_id: d for d in verifier.verify_direct(query, objects.targets, direct_target_ids(results))}
            bundle_map = {b.target_id: b for b in bundles}
            opportunities = {}
            for candidate in selected:
                tid = candidate.selection.target_id
                bundle = bundle_map[tid]
                assignments = router.assign(qid, bundle.evidence_ids, row_count=len(query["rows"]))
                localized_rows = {}
                for position in sorted(set(assignments.values()))[:4]:
                    row = query["rows"][position]
                    visible = row_values(query, row)
                    local = [backend.localize_evidence(visible, attribute_name=candidate.selection.column_name, evidence=objects.evidence[e])
                             for e in bundle.evidence_ids if assignments[e] == position]
                    best = local[0] if len(local) == 1 else local[int(backend.evidence_logits(visible, attribute_name=candidate.selection.column_name, candidates=local).argmax())]
                    localized_rows[position] = best
                opportunities[tid] = (candidate.selection, localized_rows)
            opportunity_record = {tid: {"selection": asdict(selection), "rows": {str(p): best.record() for p, best in local.items()}}
                                  for tid, (selection, local) in opportunities.items()}
        except (RuntimeError, ValueError, OSError) as exc:
            shared_error = type(exc).__name__
            opportunity_record = {}
        for condition in CONDITIONS:
            key = (generator, qid, condition, pool_sha, signature_hash)
            if key in completed:
                continue
            errors = []
            candidates = []
            if not shared_error:
                recovered = {}
                backend.condition = condition
                for tid, (selection, local) in opportunities.items():
                    predictions = []
                    for position, row in enumerate(query["rows"]):
                        best = local.get(position)
                        value = ""
                        if best is not None:
                            backend.generation_context = {"generator_id": generator, "query_id": qid, "target_id": tid, "row_id": row["row_id"], "input_pool_sha": pool_sha, "run_signature": signature_hash}
                            try:
                                value = backend.generate_value(row_values(query, row), attribute_name=selection.column_name, evidence=best,
                                                               original_evidence=objects.evidence[best.evidence_id])
                            except (RuntimeError, ValueError, OSError) as exc:
                                errors.append({"target_id": tid, "row_id": row["row_id"], "error_type": type(exc).__name__})
                        predictions.append(RowPrediction(int(row["row_id"]), value, best.record() if best else None))
                    check = verifier._semantic_check([r.value for r in predictions], column_values(objects.targets[tid], selection.column_index))
                    recovered[tid] = EvidenceVerification(selection, tuple(predictions), check)
                candidates = final_candidates(results, bundles, scores, direct, recovered)
            failed = bool(shared_error or errors)
            outputs.append({"generator_id": generator, "query_id": qid, "condition": condition, "input_pool_sha": pool_sha,
                "run_signature": signature_hash, "candidate_budget": 18, "effective_candidate_count": len(results),
                "status": "failed" if failed else "ran", "shared_error": shared_error, "generation_errors": errors,
                "opportunities": opportunity_record, "opportunity_sha": stable_sha(opportunity_record),
                "ranking": [] if failed else [c["target_id"] for c in candidates], "diagnostic_candidates": candidates})
            completed.add(key)
            write_rows(outputs_path, outputs)
            write_rows(raw_path, backend.generation_records)
        torch.cuda.empty_cache()
        print(json.dumps({"generator": generator, "query": index, "total": len(inputs), "completed_rows": len(outputs)}), flush=True)
    population = {r["query_id"]: r for r in read_rows(OUT / "common/dev_queries.jsonl")}
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = {}
        for kind in ("overall", "implicit", "explicit"):
            qrels = {r["query_id"]: population[r["query_id"]]["positive_target_ids"] for r in inputs if kind == "overall" or population[r["query_id"]]["query_kind"] == kind}
            ranks = {r["query_id"]: r["ranking"] for r in outputs if r["condition"] == condition}
            metrics[condition][kind] = population_metrics(ranks, qrels, KS)
    _json(directory / "METRICS.json", metrics)
    receipt = {"generator": generator, "rows": len(outputs), "expected_rows": 192, "failed_rows": sum(r["status"] == "failed" for r in outputs),
               "metrics": metrics, "results": file_record(outputs_path), "raw_generation": file_record(raw_path), "signature": signature}
    _json(directory / "PILOT_RECEIPT.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("Qwen-Raw", "B13"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.generator)))

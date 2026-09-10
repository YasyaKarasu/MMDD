#!/usr/bin/env python
"""Export R15 fixed train-fit E-channel formulas and their numerical audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_path_aggregator, load_student
from mmdd_stage1.data import TargetExample
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import _aggregate_path_channels
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.training import _student_path_losses, _target_teacher_scores
from run_stage1_r13 import _paths as r13_paths
from run_stage1_r14 import _schedule


FP32_ATOL = 1e-4
FP32_RTOL = 1e-5


def lse_responsibilities(values: list[float], temperature: float) -> list[float]:
    """Return each retained path's derivative of temperature-scaled logsumexp."""

    if not values:
        return []
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    denominator = sum(weights)
    return [value / denominator for value in weights]


@torch.inference_mode()
def audit_batch(
    model: Any,
    batch: list[TargetExample],
    store: FeatureStore,
    device: torch.device,
    aggregator: PathAggregator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decompose the actual scoring entrypoint on one unmodified training batch."""

    if aggregator.evidence_aggregation != "logsumexp" or aggregator.path_combination != "sum":
        raise ValueError("R15 training audit requires the frozen all-path LSE/sum recipe")
    actual = score_target_batch(model, batch, store, device, aggregator)
    sources, destinations, pair_keys = [], [], []
    for example in batch:
        for candidate in example.candidates:
            for evidence_id in sorted(candidate.evidence_ids):
                for source_id, destination_id in (
                    (example.query_id, evidence_id),
                    (evidence_id, candidate.target_id),
                ):
                    sources.append(store.embedding_features(source_id))
                    destinations.append(store.embedding_features(destination_id))
                    pair_keys.append((source_id, destination_id))
    raw = model.score_pairs_in_space(sources, destinations, "raw_logit").cpu().tolist()
    edges = dict(zip(pair_keys, raw))
    records = []
    path_errors, aggregation_errors, inference_errors = [], [], []
    for query_index, example in enumerate(batch):
        for target_index, candidate in enumerate(example.candidates):
            evidence_ids = sorted(candidate.evidence_ids)
            known = set((example.positive_evidence_by_target or {}).get(candidate.target_id, ()))
            path_values = [
                edges[(example.query_id, evidence_id)] + edges[(evidence_id, candidate.target_id)]
                for evidence_id in evidence_ids
            ]
            actual_paths = actual.path_logits[query_index][target_index].cpu().tolist()
            path_errors.extend(abs(left - right) for left, right in zip(actual_paths, path_values))
            responsibility = lse_responsibilities(path_values, aggregator.temperature)
            actual_evidence = float(actual.evidence.logits[query_index, target_index])
            reconstructed = (
                (max(path_values) + aggregator.temperature * math.log(sum(
                    math.exp((value - max(path_values)) / aggregator.temperature)
                    for value in path_values
                ))) / aggregator.target_temperature
                if path_values else 0.0
            )
            aggregation_errors.append(abs(actual_evidence - reconstructed))
            inference_paths = [
                {"kind": "evidence", "evidence_id": evidence_id, "path_score": value}
                for evidence_id, value in zip(evidence_ids, path_values)
            ]
            _direct, inference_score, _selected = _aggregate_path_channels(inference_paths, aggregator)
            if inference_score is not None:
                inference_errors.append(abs(actual_evidence - inference_score))
            evidence_positive = bool(actual.evidence.positive_mask[query_index, target_index])
            records.append({
                "query_id": example.query_id,
                "query_index_in_fixed_batch": query_index,
                "target_id": candidate.target_id,
                "target_index_in_training_list": target_index,
                "positive_target_ids": list(example.positive_target_ids),
                "direct_candidate_mask": bool(actual.direct.candidate_mask[query_index, target_index]),
                "direct_positive_mask": bool(actual.direct.positive_mask[query_index, target_index]),
                "evidence_candidate_mask": bool(actual.evidence.candidate_mask[query_index, target_index]),
                "evidence_positive_mask": evidence_positive,
                "direct_raw_score": float(actual.direct.logits[query_index, target_index]),
                "evidence_actual_training_score": actual_evidence,
                "evidence_reconstructed_score": reconstructed,
                "natural_lse_same_bag_score": inference_score,
                "training_path_count": len(evidence_ids),
                "training_paths_used_by_lse": len(evidence_ids),
                "known_witness_ids": sorted(known),
                "known_witness_responsibility_mass": sum(
                    mass for evidence_id, mass in zip(evidence_ids, responsibility) if evidence_id in known
                ) if evidence_ids else None,
                "known_witness_is_top_path": (
                    min(zip(evidence_ids, path_values), key=lambda row: (-row[1], row[0]))[0] in known
                    if evidence_ids else None
                ),
                "paths": [{
                    "evidence_id": evidence_id,
                    "modality": store.embedding_features(evidence_id).object_type,
                    "known_witness": evidence_id in known,
                    "query_evidence_raw_score": edges[(example.query_id, evidence_id)],
                    "evidence_target_raw_score": edges[(evidence_id, candidate.target_id)],
                    "path_score_sum": value,
                    "path_score_actual_training_entrypoint": actual_value,
                    "lse_responsibility": mass,
                    "known_support_rows": list((example.positive_evidence_rows_by_target or {}).get(
                        candidate.target_id, {}
                    ).get(evidence_id, ())),
                } for evidence_id, value, actual_value, mass in zip(
                    evidence_ids, path_values, actual_paths, responsibility
                )],
            })
    score_scale = max((abs(row["evidence_actual_training_score"]) for row in records), default=1.0)
    max_errors = {
        "path_sum_vs_actual_path_logits": max(path_errors, default=0.0),
        "manual_lse_vs_actual_training_score": max(aggregation_errors, default=0.0),
        "same_bag_inference_lse_vs_actual_training_score": max(inference_errors, default=0.0),
    }
    known_positive_records = [row for row in records if row["evidence_positive_mask"] and row["known_witness_ids"]]
    return records, {
        "queries": len(batch),
        "candidate_pairs": len(records),
        "path_occurrences": sum(row["training_path_count"] for row in records),
        "positive_pairs_with_known_witness_labels": len(known_positive_records),
        "mean_known_witness_responsibility_on_labeled_positive_pairs": (
            sum(float(row["known_witness_responsibility_mass"] or 0.0) for row in known_positive_records)
            / len(known_positive_records) if known_positive_records else None
        ),
        "path_count_distribution": {str(count): sum(row["training_path_count"] == count for row in records)
                                    for count in sorted({row["training_path_count"] for row in records})},
        "max_absolute_errors": max_errors,
        "fp32_atol": FP32_ATOL,
        "fp32_rtol": FP32_RTOL,
        "score_scale": score_scale,
        "passed": all(error <= FP32_ATOL + FP32_RTOL * max(1.0, score_scale) for error in max_errors.values()),
    }


def endpoints(root: Path) -> dict[str, Path]:
    r14 = root / "work/stage1_optimization_r14_20260909"
    r15 = root / "work/stage1_optimization_r15_20260909/stageI_interaction"
    return {
        "s_full": root / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
        "s_eoff": r14 / "stage1_B_branch_ablation/b_d_e_loss_off_seed13/checkpoints/step_000178.pt",
        "l_full": r14 / "stage1_M_projection_capacity/m_l_linear_residual_seed13/checkpoints/step_000178.pt",
        "n_full": r14 / "stage1_M_projection_capacity/m_n_gelu_residual_seed13/checkpoints/step_000178.pt",
        "l_eoff": r15 / "l_eoff_seed13/checkpoints/step_000178.pt",
        "n_eoff": r15 / "n_eoff_seed13/checkpoints/step_000178.pt",
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    paths = r13_paths(args.root)
    batches, order = _schedule(args.root, 13)
    batch = batches[0]
    output = args.root / "work/stage1_optimization_r15_20260909/stageG_correctness/training_scores"
    output.mkdir(parents=True, exist_ok=True)
    panel_json = json.dumps([asdict(example) for example in batch], ensure_ascii=False, sort_keys=True)
    panel_hash = hashlib.sha256(panel_json.encode()).hexdigest()
    store = FeatureStore.from_path(paths["features"], cache_size=60000)
    results = {}
    for arm, checkpoint_path in endpoints(args.root).items():
        started = time.monotonic()
        model = load_student(checkpoint_path, device).eval()
        aggregator = load_path_aggregator(checkpoint_path)
        records, summary = audit_batch(model, batch, store, device, aggregator)
        scores = score_target_batch(model, batch, store, device, aggregator)
        loss_weight = 0.0 if arm.endswith("eoff") else 1.0
        objective = _student_path_losses(
            model, scores, _target_teacher_scores(batch, device), None,
            temperature=1.0, distillation_weight=0.3, anchor_weight=0.1,
            anchor_weight_evidence=0.1, distillation_rows=None,
            positive_loss_mode="sum_probability", evidence_loss_weight=loss_weight,
        )
        path = output / f"{arm}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"arm": arm, **record}, ensure_ascii=False) + "\n")
        summary.update({
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_fingerprint(checkpoint_path),
            "batch_sha256": panel_hash,
            "evidence_loss_weight": loss_weight,
            "actual_fixed_batch_objective": {name: float(value.detach()) for name, value in objective.items()},
            "checkpoint_aggregation": {
                "kind": aggregator.evidence_aggregation, "top_k_field": aggregator.top_k,
                "temperature": aggregator.temperature, "target_temperature": aggregator.target_temperature,
                "path_combination": aggregator.path_combination,
                "all_paths_used": True, "top_k_field_active_for_this_aggregation": False,
            },
            "records": str(path.resolve()), "records_sha256": checkpoint_fingerprint(path),
            "elapsed_seconds": time.monotonic() - started,
        })
        results[arm] = summary
        print(json.dumps({"arm": arm, "passed": summary["passed"], "max_errors": summary["max_absolute_errors"]}), flush=True)
        del model, scores, objective
    payload = {
        "format_version": 1,
        "status": "complete",
        "passed": all(row["passed"] for row in results.values()),
        "panel": {"selection": "unchanged first seed13 train-fit batch", "query_ids": [row.query_id for row in batch],
                  "batch_sha256": panel_hash, "schedule_order_indices": order[:len(batch)],
                  "training_path_hard_source": str(paths["path_hard"]),
                  "training_path_hard_source_sha256": checkpoint_fingerprint(paths["path_hard"]),
                  "witness_metadata_source": str(paths["train_targets"]),
                  "witness_metadata_source_sha256": checkpoint_fingerprint(paths["train_targets"]),
                  "order_source": str(args.root / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/schedule_order.json"),
                  "order_source_sha256": checkpoint_fingerprint(args.root / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/schedule_order.json")},
        "formula": {
            "edge_score": "s(a,b) = F(a)^T R_{a,b} F(b)",
            "training_edge_transform": "identity (raw_logit); neither sigmoid nor 10*sigmoid is applied",
            "path_score": "p(q,e,t) = s(q,e) + s(e,t)",
            "training_evidence_score": "tau * log(sum_{all candidate evidence IDs} exp(p/tau)) / target_temperature; empty bag -> 0 with candidate mask false",
            "responsibility": "softmax(p / tau); derivative before target_temperature division; all candidate paths, no top4 truncation",
            "positive_mask": "candidate target is in all known positive_target_ids (legacy positive index fallback), evidence mask additionally requires a nonempty evidence bag",
            "supervised_loss": "mean(logsumexp(all eligible candidate scores)-logsumexp(all positive scores)); E only rows with positive and nonpositive candidate",
            "distillation_loss": "T^2 * mean KL(softmax(Teacher/T) || softmax(Student/T)), T=1, cached Teacher, E only usable rows",
            "objective": "D_CE + 0.3 D_KD + evidence_loss_weight*(E_CE+0.3 E_KD) + 0.1 base P/R anchor",
            "base_anchor": "sum_active_relations ||R-I||_F^2/1024^2 + sum_types ||P-P_initial_full_chain||_F^2/(4096*1024); A/B and full F are not in this anchor",
            "gradient_position": "loss.backward then student_gradient_norms then optimizer.step; no gradient clipping",
        },
        "inference_alignment": {
            "natural_path_edges": "raw_logit with same QE+ET sum",
            "same_bag_natural_lse": "numerically compared to actual training score using checkpoint temperature",
            "deployed_evidence_score": "e2_row_coverage: exact-content dedup then path-score Top20 then greedy max row coverage with budget4 and sigmoid(path_score) quality",
            "deployed_row_support": "clip((frozen row embedding dot frozen evidence embedding + 1)/2, 0, 1)",
            "deployed_temperature": None,
            "f1": "rank raw candidate union by direct score; retained evidence does not change F1 target score",
            "equal_rrf": "fixed equal reciprocal ranks of union-direct and deployed evidence channel, rrf_k=60",
            "causal_boundary": "Same raw path formula, different training-bag candidate support and deployed aggregation. Training LSE responsibility is not deployed greedy responsibility.",
        },
        "results": results,
        "optimizer_updates": 0,
        "new_teacher_inference": 0,
        "responsibility_scope": "The fixed first training batch has labeled positive bags whose available paths are all known witnesses. Unit known-witness mass here is a candidate-bag property, not evidence of natural retrieval quality.",
        "device": args.device,
        "source_hashes": {str(path.relative_to(args.root)): checkpoint_fingerprint(path) for path in (
            Path(__file__).resolve(), args.root / "src/mmdd_stage1/models.py",
            args.root / "src/mmdd_stage1/scoring.py", args.root / "src/mmdd_stage1/objectives.py",
            args.root / "src/mmdd_stage1/training.py", args.root / "src/mmdd_stage1/retrieval.py",
            args.root / "src/run_stage1_r11_task_e.py", args.root / "src/run_stage1_r11_task_f.py",
        )},
    }
    write_json(output / "summary.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    run(parser.parse_args())

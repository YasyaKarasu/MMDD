#!/usr/bin/env python
"""Complete the frozen R13 Stage-1 mechanism and reproducibility audit."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.data import load_edge_examples, load_target_examples
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.pca import load_pca_projection
from mmdd_stage1.retrieval import StudentANNIndices, load_corpus_ids
from diagnose_stage1_r11_exact_search import _source_groups
from reevaluate_stage1_r12_checkpoints import _requests, exact_relation
from run_stage1_r13 import _output_root, _paths, freeze_plan


ARMS = {
    "s0": (
        "taskA_stage1_protocol/s0/evaluation_step0",
        None,
    ),
    "c_s_shared": (
        "taskC_role_projection/c_s_shared/evaluation_step178",
        "taskC_role_projection/c_s_shared/checkpoints/step_000178.pt",
    ),
    "c_r_split": (
        "taskC_role_projection/c_r_split/evaluation_step178",
        "taskC_role_projection/c_r_split/checkpoints/step_000178.pt",
    ),
    "p_s_target_only": (
        "taskD_witness_supervision/p_s_target_only/evaluation_step178",
        "taskD_witness_supervision/p_s_target_only/checkpoints/step_000178.pt",
    ),
    "p_w_witness": (
        "taskD_witness_supervision/p_w_witness/evaluation_step178",
        "taskD_witness_supervision/p_w_witness/checkpoints/step_000178.pt",
    ),
    "b1_kd_on_hard356": (
        "taskB_diagnostics_and_kd/b1_kd_on_hard356/evaluation_step356",
        "taskB_diagnostics_and_kd/b1_kd_on_hard356/checkpoints/step_000356.pt",
    ),
    "b1_kd_off_hard356": (
        "taskB_diagnostics_and_kd/b1_kd_off_hard356/evaluation_step356",
        "taskB_diagnostics_and_kd/b1_kd_off_hard356/checkpoints/step_000356.pt",
    ),
}

USED_RELATIONS = {
    "table_to_table",
    "table_to_text",
    "table_to_image",
    "text_to_table",
    "image_to_table",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint(output: Path, plan: dict[str, Any], relative: str | None) -> Path:
    return Path(plan["s0"]["path"]) if relative is None else output / relative


def _parameter_audit(
    model: Any,
    s0: Any,
    table_row_basis: torch.Tensor,
) -> dict[str, Any]:
    stored = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    gradient_bearing = 0
    relation_parameters = 0
    used_relation_parameters = 0
    for name, parameter in model.named_parameters():
        if name.startswith(("relations.", "relation_as.", "relation_bs.")):
            relation_parameters += parameter.numel()
            relation = name.split(".", 2)[1]
            if relation in USED_RELATIONS:
                used_relation_parameters += parameter.numel()
        else:
            gradient_bearing += parameter.numel()
    gradient_bearing += used_relation_parameters
    original = s0.projections["table"].weight.detach()
    role_drift = {}
    for role in ("query", "target"):
        key = "table" if model.projection_mode == "shared" else f"table_{role}"
        delta = model.projections[key].weight.detach() - original
        outside = delta - (delta @ table_row_basis) @ table_row_basis.T
        norm = float(delta.double().norm())
        outside_norm = float(outside.double().norm())
        role_drift[role] = {
            "delta_frobenius": norm,
            "outside_s0_rowspace_frobenius": outside_norm,
            "outside_fraction": outside_norm / norm if norm else 0.0,
        }
    return {
        "stored_parameters": stored,
        "trainable_parameters": trainable,
        "stored_relation_parameters": relation_parameters,
        "relations_receiving_gradients": sorted(USED_RELATIONS),
        "parameters_receiving_gradients_in_frozen_schedule": gradient_bearing,
        "float32_weight_bytes": stored * 4,
        "estimated_train_weight_gradient_two_adam_moments_bytes": trainable * 16,
        "table_role_drift_from_s0": role_drift,
    }


def _path_mechanism(path: Path) -> dict[str, Any]:
    pair_count = 0
    known_pair_count = 0
    known_present_count = 0
    known_top_path_count = 0
    support_masses = []
    maximum_known_responsibilities = []
    row_distribution = Counter()
    evidence_path_counts = []
    unique_union_targets = []
    actual_search_vectors = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            paths_by_target = record["paths_by_target"]
            unique_union_targets.append(len(paths_by_target))
            evidence_ids = {
                str(item["evidence_id"])
                for paths in paths_by_target.values()
                for item in paths
                if item.get("kind") == "evidence"
            }
            actual_search_vectors.append(3 + len(evidence_ids))
            known_by_target = record.get("positive_evidence_by_target", {})
            rows_by_target = record.get("positive_evidence_rows_by_target", {})
            for target_id in record["positive_target_ids"]:
                pair_count += 1
                paths = [
                    item
                    for item in paths_by_target.get(str(target_id), [])
                    if item.get("kind") == "evidence"
                ]
                evidence_path_counts.append(len(paths))
                known = {str(value) for value in known_by_target.get(str(target_id), [])}
                if not known:
                    row_distribution[0] += 1
                    continue
                known_pair_count += 1
                present = [item for item in paths if str(item["evidence_id"]) in known]
                known_present_count += int(bool(present))
                if paths:
                    maximum = max(float(item["path_score"]) for item in paths)
                    weights = [math.exp(float(item["path_score"]) - maximum) for item in paths]
                    denominator = sum(weights)
                    known_weights = [
                        weight
                        for item, weight in zip(paths, weights)
                        if str(item["evidence_id"]) in known
                    ]
                    support_masses.append(sum(known_weights) / denominator)
                    maximum_known_responsibilities.append(
                        max(known_weights, default=0.0) / denominator
                    )
                    known_top_path_count += int(
                        str(paths[max(range(len(paths)), key=lambda index: weights[index])]["evidence_id"])
                        in known
                    )
                else:
                    support_masses.append(0.0)
                    maximum_known_responsibilities.append(0.0)
                row_ids = set()
                rows = rows_by_target.get(str(target_id), {})
                for item in present:
                    row_ids.update(int(value) for value in rows.get(str(item["evidence_id"]), []))
                row_distribution[min(len(row_ids), 5)] += 1
    return {
        "positive_target_pairs": pair_count,
        "pairs_with_known_witness_metadata": known_pair_count,
        "pairs_with_known_witness_in_natural_pool": known_present_count,
        "known_witness_pool_coverage": (
            known_present_count / known_pair_count if known_pair_count else None
        ),
        "mean_known_witness_lse_responsibility": (
            sum(support_masses) / len(support_masses) if support_masses else None
        ),
        "mean_maximum_single_known_witness_responsibility": (
            sum(maximum_known_responsibilities) / len(maximum_known_responsibilities)
            if maximum_known_responsibilities
            else None
        ),
        "known_witness_is_top_path_rate": (
            known_top_path_count / known_pair_count if known_pair_count else None
        ),
        "known_supported_row_union_distribution_0_to_5_plus": {
            str(value): row_distribution[value] for value in range(6)
        },
        "mean_evidence_paths_per_positive_target": (
            sum(evidence_path_counts) / len(evidence_path_counts)
            if evidence_path_counts
            else 0.0
        ),
        "mean_unique_union_targets": sum(unique_union_targets) / len(unique_union_targets),
        "actual_search_vectors_per_query": {
            "min": min(actual_search_vectors),
            "max": max(actual_search_vectors),
            "mean": sum(actual_search_vectors) / len(actual_search_vectors),
        },
        "wrong_attribute_path_rate": None,
        "wrong_entity_path_rate": None,
        "independent_negative_label_coverage": 0.0,
    }


def _ranking_maps(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {row["query_id"]: row for row in map(json.loads, handle)}


def _direct_members(path: Path) -> dict[str, set[str]]:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[row["query_id"]] = {
                str(target_id)
                for target_id, paths in row["paths_by_target"].items()
                if any(item.get("kind") == "direct" for item in paths)
            }
    return result


def _attribution(
    rankings: dict[str, dict[str, Any]],
    own_direct: dict[str, set[str]],
    s0_rankings: dict[str, dict[str, Any]],
    s0_direct: dict[str, set[str]],
) -> dict[str, Any]:
    result = {}
    for k in (10, 20, 50):
        delta_pure = []
        delta_s0 = []
        rescued_pure = displaced_pure = rescued_s0 = displaced_s0 = 0
        final_hits_outside_own_d100 = final_hits_outside_s0_d100 = 0
        positive_pairs = own_d100_hits = s0_d100_hits = 0
        for query_id, row in rankings.items():
            positives = set(row["positive_target_ids"])
            denominator = row["positive_denominator"]
            final = set(row["rankings"]["f1_union_direct"][str(k)]["target_ids"])
            pure = set(row["rankings"]["pure_direct100"][str(k)]["target_ids"])
            s0 = set(
                s0_rankings[query_id]["rankings"]["f1_union_direct"][str(k)]["target_ids"]
            )
            delta_pure.append((len(final & positives) - len(pure & positives)) / denominator)
            delta_s0.append((len(final & positives) - len(s0 & positives)) / denominator)
            rescued_pure += len((final - pure) & positives)
            displaced_pure += len((pure - final) & positives)
            rescued_s0 += len((final - s0) & positives)
            displaced_s0 += len((s0 - final) & positives)
            positive_pairs += len(positives)
            own_d100_hits += len(positives & own_direct[query_id])
            s0_d100_hits += len(positives & s0_direct[query_id])
            final_hits_outside_own_d100 += len(
                positives & final - own_direct[query_id]
            )
            final_hits_outside_s0_d100 += len(
                positives & final - s0_direct[query_id]
            )
        result[str(k)] = {
            "macro_recall_delta_vs_pure_direct100": sum(delta_pure) / len(delta_pure),
            "macro_recall_delta_vs_s0_f1": sum(delta_s0) / len(delta_s0),
            "positive_rescued_vs_pure": rescued_pure,
            "positive_displaced_vs_pure": displaced_pure,
            "positive_rescued_vs_s0": rescued_s0,
            "positive_displaced_vs_s0": displaced_s0,
            "known_positive_pairs": positive_pairs,
            "known_positive_pairs_in_model_own_d100": own_d100_hits,
            "known_positive_pairs_in_fixed_s0_d100": s0_d100_hits,
            "final_hits_outside_model_own_d100": final_hits_outside_own_d100,
            "final_hits_outside_fixed_s0_d100": final_hits_outside_s0_d100,
        }
    return result


def _input_amendment(root: Path, output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    paths = _paths(root)
    dependencies = {}
    for name, path in paths.items():
        if name in {"r12", "features"}:
            continue
        dependencies[name] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_fingerprint(path),
        }
    for path in (
        root / "src/mmdd_stage1/models.py",
        root / "src/mmdd_stage1/scoring.py",
        root / "src/mmdd_stage1/training.py",
        root / "src/mmdd_stage1/retrieval.py",
        root / "src/mmdd_stage1/witness_supervision.py",
        root / "src/run_stage1_r13.py",
        root / "src/diagnose_stage1_r13.py",
        root / "src/diagnose_stage1_r13_witness_gradient.py",
        root / "src/audit_stage1_r13_candidates.py",
        root / "src/validate_stage1_r13_artifacts.py",
        root / "src/finalize_stage1_r13.py",
        root / "src/profile_stage1_r13.py",
        Path(__file__),
    ):
        dependencies[f"code:{path.name}"] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_fingerprint(path),
        }
    payload = {
        "format_version": 1,
        "status": "post_execution_dependency_amendment",
        "original_plan_frozen_sha256": checkpoint_fingerprint(output / "PLAN_FROZEN.json"),
        "plan_sha256": plan["plan_sha256"],
        "changes_selection_or_protocol": False,
        "reason": (
            "The immutable pre-run freeze omitted path-hard, evidence-content-key, "
            "historical-test and implementation fingerprints. This amendment records "
            "them without pretending they were pre-registered or changing selection."
        ),
        "dependencies": dependencies,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "PLAN_FROZEN_AMENDMENT.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    output = _output_root(args.root)
    plan = freeze_plan(args.root)
    amendment = _input_amendment(args.root, output, plan)
    if args.dependencies_only:
        target = output / "statistics/mechanism_and_reproducibility_audit.json"
        if target.is_file():
            payload = _read_json(target)
            payload["dependency_amendment_sha256"] = checkpoint_fingerprint(
                output / "PLAN_FROZEN_AMENDMENT.json"
            )
            write_json(target, payload)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str((output / "PLAN_FROZEN_AMENDMENT.json").resolve()),
                },
                indent=2,
            )
        )
        return amendment
    paths = _paths(args.root)
    targets = load_target_examples(paths["dev_targets"], split="dev")
    edges = load_edge_examples(paths["dev_edges"], split="dev")
    supervision_manifest = _read_json(
        paths["r12"] / "taskA_correctness/supervision/manifest.json"
    )
    requests = _requests(
        targets, edges, Path(supervision_manifest["dataset_root"])
    )
    sample_ids = {name: ids for name, ids, *_rest in requests}
    sample_path = output / "statistics/mechanism_panel_samples.json"
    write_json(sample_path, sample_ids)
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    store.preload_embeddings(
        [
            *(value for ids in ids_by_type.values() for value in ids),
            *(row.query_id for row in targets),
        ]
    )
    basis = load_pca_projection(
        args.root
        / "work/stage1_optimization_r10_20260907/baselines/pca_entitables_v9_1024.pt",
        input_dim=store.embedding_dimension(),
        student_dim=1024,
    ).to(device)
    s0 = load_student(Path(plan["s0"]["path"]), device).eval()
    table_row_basis, _ = torch.linalg.qr(
        s0.projections["table"].weight.detach().T, mode="reduced"
    )
    s0_eval_dir = output / ARMS["s0"][0]
    s0_rankings = _ranking_maps(s0_eval_dir / "rankings.jsonl.gz")
    s0_direct = _direct_members(s0_eval_dir / "path_pool.jsonl.gz")
    exact = {}
    mechanisms = {}
    parameters = {}
    attribution = {}
    costs = {}
    for arm, (evaluation_relative, checkpoint_relative) in ARMS.items():
        evaluation = output / evaluation_relative
        checkpoint_path = _checkpoint(output, plan, checkpoint_relative)
        model = load_student(checkpoint_path, device).eval()
        indices = StudentANNIndices(
            model,
            store,
            evaluation / "index",
            device=device,
            checkpoint_sha256=checkpoint_fingerprint(checkpoint_path),
            corpus_sha256=plan["inputs"]["corpus"]["sha256"],
            score_space="raw_logit",
        )
        exact[arm] = {
            request[0]: exact_relation(
                model, indices, store, ids_by_type, request, basis, device
            )
            for request in requests
        }
        parameters[arm] = _parameter_audit(model, s0, table_row_basis)
        pool_path = evaluation / "path_pool.jsonl.gz"
        mechanisms[arm] = _path_mechanism(pool_path)
        arm_rankings = _ranking_maps(evaluation / "rankings.jsonl.gz")
        arm_direct = _direct_members(pool_path)
        attribution[arm] = _attribution(
            arm_rankings, arm_direct, s0_rankings, s0_direct
        )
        metric = _read_json(evaluation / "metrics.json")
        index_manifest = _read_json(evaluation / "index/manifest.json")
        costs[arm] = {
            **metric["cost"],
            "index_entries": {
                key: int(value["objects"])
                for key, value in index_manifest["types"].items()
            },
            "index_dimension": int(index_manifest["student_dim"]),
            "index_resident_memory_bytes": None,
            "phase_latency": {
                "QT": None,
                "QE": None,
                "ET": None,
                "role_or_H_vector_construction": None,
                "union_direct_supplement": None,
                "retention": None,
                "fusion": None,
            },
            "missing_cost_reason": (
                "The completed natural run timed retrieval+ranking jointly. A separate "
                "same-hardware selected-vs-S0 phase profiler records these fields."
            ),
        }
        del indices, model
        torch.cuda.empty_cache()
    payload = {
        "format_version": 1,
        "status": "complete",
        "plan_sha256": plan["plan_sha256"],
        "dependency_amendment_sha256": checkpoint_fingerprint(
            output / "PLAN_FROZEN_AMENDMENT.json"
        ),
        "mechanism_panel": {
            "sampling": (
                "seed13, source-group fixed; 64 implicit + 64 explicit queries and "
                "up to 64 evidence sources per modality"
            ),
            "sample_path": str(sample_path.resolve()),
            "sample_sha256": checkpoint_fingerprint(sample_path),
            "exact_and_ann_by_arm": exact,
        },
        "natural_path_mechanism_by_arm": mechanisms,
        "parameter_and_role_update_by_arm": parameters,
        "rescue_displace_and_d100_by_arm": attribution,
        "cost_by_arm": costs,
        "conditional_components_not_implemented": {
            "Residual": "Task E trigger was not met",
            "H": "Task E trigger was not met",
            "Stage2": "explicitly deferred by the R13 plan",
        },
        "wrong_path_label_policy": (
            "wrong-attribute and wrong-entity rates are null with zero independent "
            "negative-label coverage; unknown paths are not treated as errors"
        ),
        "cost": {
            "elapsed_seconds": time.monotonic() - started,
            "device": args.device,
            "cpu_threads": args.cpu_threads,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "code_sha256": checkpoint_fingerprint(Path(__file__)),
    }
    target = output / "statistics/mechanism_and_reproducibility_audit.json"
    write_json(target, payload)
    with (output / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": "R13 mechanism and reproducibility audit",
                    "status": "complete",
                    "output": str(target.resolve()),
                    "command": payload["command"],
                    "cost": payload["cost"],
                }
            )
            + "\n"
        )
    print(json.dumps({"status": "complete", "output": str(target)}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--dependencies-only", action="store_true")
    run(parser.parse_args())

#!/usr/bin/env python
"""Audit frozen B13 candidate generation, admission, and retained witnesses for R15."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from analyze_stage1_r14 import _target_vectors
from finalize_stage1_r13 import _bootstrap, _source_map
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import fuse_ranked_channels, load_corpus_ids
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import empty_intervention_stats
from run_stage1_r11_task_f import _target_channels
from run_stage1_r13 import _DirectScorer, _paths as r13_paths


RULES = ("pure_direct100", "f1_union_direct", "union_rrf_equal")
KS = (10, 20, 50)


def output_directory(root: Path) -> Path:
    return root / "work/stage1_optimization_r15_20260909/stageC_candidate_delivery"


def read_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def freeze(root: Path) -> dict[str, Any]:
    output = output_directory(root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "C_CONFIG_FROZEN.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    arm = root / "work/stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only"
    paths = r13_paths(root)
    inputs = {
        "checkpoint": arm / "checkpoints/step_000178.pt",
        "pool": arm / "evaluation_step178/path_pool.jsonl.gz",
        "rankings": arm / "evaluation_step178/rankings.jsonl.gz",
        "metrics": arm / "evaluation_step178/metrics.json",
        "corpus": paths["corpus"],
        "dev_targets": paths["dev_targets"],
        "content_keys": paths["evidence_content_keys"],
        "plan": root / "stage1_optimization_r15_plan_20260909.md",
    }
    payload = {
        "format_version": 1,
        "arm": "B13_S_full_seed13",
        "queries": 1198,
        "rules": list(RULES),
        "ks": list(KS),
        "candidate_budget": 50,
        "retention": {"strategy": "e2_row_coverage", "top_l": 20, "budget": 4},
        "rrf_k": 60,
        "random_controls": {
            "seed": 15,
            "repeats": 100,
            "generator": "numpy.default_rng(SeedSequence([15, sorted_query_index, control_index]))",
            "control_index": {"pool_random": 0, "lake_random": 1},
            "fixed_members": "RRF C50 intersect own ANN D100",
            "external_slots": "count(RRF C50 minus own ANN D100), computed without labels",
            "pool_population": "E minus own ANN D100",
            "lake_population": "legal T minus own ANN D100",
            "sampling": "uniform without replacement independently per repeat",
            "top10": None,
            "top10_reason": "Admission-only controls; no Stage2 ranking has been run.",
        },
        "exact": {"dtype": "float32", "tie_break": "score descending, target ID ascending"},
        "bootstrap": {"unit": "source_table_id", "iterations": 10000, "seed": 15},
        "queue_definition": "implicit positive pairs outside own ANN D100 with known witness QET in saved raw pool",
        "queue_reconstruction_reason": "The review's named CSV is absent locally; reconstruct its prespecified definition and verify all reported counts.",
        "expected_ann_evidence_only_pairs": 213,
        "expected_known_witness_queue_pairs": 82,
        "inputs": {
            name: {"path": str(value.resolve()), "sha256": checkpoint_fingerprint(value)}
            for name, value in inputs.items()
        },
    }
    write_json(path, payload)
    return payload


def candidate_sets(record: dict[str, Any]) -> tuple[list[str], set[str], set[str]]:
    """Recover natural ANN direct order and E/U from untruncated saved paths."""
    direct = []
    evidence = set()
    for target_id, paths in record["paths_by_target"].items():
        for path in paths:
            if path["kind"] == "direct":
                direct.append((str(target_id), float(path["path_score"])))
            elif path["kind"] == "evidence":
                evidence.add(str(target_id))
    direct.sort(key=lambda row: (-row[1], row[0]))
    ids = [target_id for target_id, _ in direct]
    return ids, evidence, set(ids) | evidence


def exact_order_and_ranks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Input positions follow ascending target ID, so stable sort fixes ties."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, ranks


def random_admission(
    rrf50: list[str],
    direct100: set[str],
    population: list[str],
    rng: np.random.Generator,
    repeats: int,
) -> tuple[list[str], list[list[str]]]:
    """Freeze RRF's direct members/slot count; selection never receives qrels."""
    fixed = [target_id for target_id in rrf50 if target_id in direct100]
    slots = len(rrf50) - len(fixed)
    if set(population) & direct100:
        raise ValueError("Random external population overlaps ANN D100")
    draws = [
        [population[int(position)] for position in rng.choice(len(population), slots, replace=False)]
        for _ in range(repeats)
    ]
    return fixed, draws


def witness_details(
    record: dict[str, Any], target_id: str, selected_ids: list[str]
) -> dict[str, Any]:
    """Separate known raw paths, their row support, and actually retained evidence."""
    known = set(record.get("positive_evidence_by_target", {}).get(target_id, []))
    paths = [
        dict(path)
        for path in record["paths_by_target"].get(target_id, [])
        if path.get("kind") == "evidence"
    ]
    known_paths = [path for path in paths if path["evidence_id"] in known]
    rows_by_evidence = record.get("positive_evidence_rows_by_target", {}).get(target_id, {})
    raw_ids = {str(path["evidence_id"]) for path in known_paths}
    retained = sorted(raw_ids & set(selected_ids))
    raw_rows = sorted({int(row) for value in raw_ids for row in rows_by_evidence.get(value, [])})
    retained_rows = sorted({int(row) for value in retained for row in rows_by_evidence.get(value, [])})
    return {
        "known_witness_ids": sorted(known),
        "known_qet_witness_ids": sorted(raw_ids),
        "known_qet_paths": known_paths,
        "known_qet_modalities": sorted({str(path["evidence_type"]) for path in known_paths}),
        "known_rows_by_evidence": {value: list(rows_by_evidence.get(value, [])) for value in sorted(known)},
        "raw_supported_rows": raw_rows,
        "raw_supported_row_count": len(raw_rows),
        "selected_evidence_ids": list(selected_ids),
        "retained_known_witness_ids": retained,
        "retained_supported_rows": retained_rows,
        "retained_supported_row_count": len(retained_rows),
        "retention_keeps_known_witness": bool(retained),
        "known_qet_path_occurrences": len(known_paths),
        "all_qet_path_occurrences": len(paths),
        "needs_independent_content_review": True,
        "value_recovery_verified": None,
        "correct_join_verified": None,
    }


def recall(ids: Iterable[str], positives: set[str]) -> float:
    return len(set(ids) & positives) / len(positives)


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for kind in ("all", "implicit", "explicit"):
        selected = [row for row in rows if kind == "all" or row["query_kind"] == kind]
        metrics = {
            "queries": len(selected),
            "source_groups": len({row["source_table_id"] for row in selected}),
            "RawUnionRecall": float(np.mean([row["metrics"]["RawUnionRecall"] for row in selected])),
            "ANN_D100_Recall": float(np.mean([row["metrics"]["ANN_D100_Recall"] for row in selected])),
            "exact_D100_Recall": float(np.mean([row["metrics"]["exact_D100_Recall"] for row in selected])),
            "mean_raw_union_size": float(np.mean([len(row["U"]) for row in selected])),
            "rules": {},
        }
        for rule in RULES:
            metrics["rules"][rule] = {
                f"recall@{k}": float(np.mean([row["metrics"][rule][str(k)] for row in selected]))
                for k in KS
            }
            metrics["rules"][rule]["CandidateRecall@50"] = metrics["rules"][rule]["recall@50"]
        result[kind] = metrics
    return result


def export_verification(root: Path) -> None:
    """Export the Stage1 facts without substituting labels for content validation."""
    output = output_directory(root)
    rows = [
        row for row in read_gzip(output / "all_positive_pairs.jsonl.gz")
        if row["ann_evidence_only"] or row["exact_evidence_only"]
    ]
    queue = {
        (row["query_id"], row["target_id"])
        for row in read_gzip(output / "B13_evidence_only_known_witness_cases.jsonl.gz")
    }
    write_gzip(output / "evidence_only_verification.jsonl.gz", ({
        "arm": row["arm"], "query_id": row["query_id"],
        "source_table_id": row["source_table_id"], "target_id": row["target_id"],
        "query_kind": row["query_kind"],
        "priority_82_queue_member": (row["query_id"], row["target_id"]) in queue,
        "exact_direct_rank": row["exact_direct_rank"],
        "ann_evidence_only": row["ann_evidence_only"],
        "chain_step1_outside_same_model_exact_direct100": row["exact_evidence_only"],
        "chain_step2_natural_qet_enters_raw_pool": row["in_E"],
        "chain_step3_delivered_C50_by_rule": row["in_C50"],
        "delivered_evidence_ids_by_rule": row["actual_evidence_delivered_by_rule"],
        "known_witness_annotation_in_raw_pool": bool(row["known_qet_witness_ids"]),
        "retention_keeps_known_witness_annotation": row["retention_keeps_known_witness"],
        "known_supported_row_count_raw": row["raw_supported_row_count"],
        "known_supported_row_count_retained": row["retained_supported_row_count"],
        "chain_step4_independent_evidence_attribute_support": None,
        "chain_step5_recovered_value_correct": None,
        "chain_step5_target_entity_and_attribute_correct": None,
        "chain_step5_correct_join": None,
        "chain_step6_final_stage2_top10_by_rule": {rule: None for rule in RULES},
        "no_evidence_counterfactual": None,
        "independently_reviewed_wrong_evidence_counterfactual": None,
        "image_information_missing_from_visible_text": None,
        "stage2_execution_status": "not_executed_no_separate_verification_budget",
        "independent_review_status": "not_reviewed",
        "reason_for_null": "Stage2 and independent value/entity/attribute review were not performed; known labels only establish candidate/witness availability.",
    } for row in rows))


@torch.inference_mode()
def analyze(args: argparse.Namespace) -> None:
    root = args.root
    config = freeze(root)
    output = output_directory(root)
    if (output / "SUMMARY.json").exists():
        raise FileExistsError("Inspect existing completed C audit before rerunning")
    for value in config["inputs"].values():
        if checkpoint_fingerprint(Path(value["path"])) != value["sha256"]:
            raise ValueError(f"Frozen C input changed: {value['path']}")
    inputs = {name: Path(value["path"]) for name, value in config["inputs"].items()}
    pools = {row["query_id"]: row for row in read_gzip(inputs["pool"])}
    saved_rankings = {row["query_id"]: row for row in read_gzip(inputs["rankings"])}
    if len(pools) != config["queries"] or pools.keys() != saved_rankings.keys():
        raise ValueError("Frozen B13 query identities differ")
    query_ids = sorted(pools)
    paths = r13_paths(root)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    model = load_student(inputs["checkpoint"], device).eval()
    store = FeatureStore.from_path(paths["features"], cache_size=260_000)
    ids_by_type = load_corpus_ids(paths["corpus"], store)
    target_ids = sorted(ids_by_type["table"])
    target_position = {value: index for index, value in enumerate(target_ids)}
    source_by_query, dataset_manifest = _source_map(root)
    used_evidence_ids = {
        str(path["evidence_id"])
        for record in pools.values()
        for paths_for_target in record["paths_by_target"].values()
        for path in paths_for_target
        if path["kind"] == "evidence"
    }
    started = time.monotonic()
    store.preload_embeddings([*target_ids, *query_ids, *sorted(used_evidence_ids)])
    target_vectors = _target_vectors(model, store, target_ids, device, 4096)
    scorer = _DirectScorer(model, store, device)
    content_keys, _ = load_evidence_content_keys(paths["evidence_content_keys"])
    intervention_stats = empty_intervention_stats()
    provenance = []
    all_positive_pairs = []
    ann_new_pairs = []
    queue = []
    delivery_rows = []
    reproduction_errors = []
    exact_invariant_failures = []
    random_draw_seconds = 0.0
    random_metrics = {name: [] for name in ("pool_random", "lake_random")}
    write_gzip(output / "legal_target_ids.jsonl.gz", [{"target_ids": target_ids}])
    with gzip.open(output / "random_control_draws.jsonl.gz", "wt", encoding="utf-8") as draw_handle:
        for query_index, query_id in enumerate(query_ids):
            record = pools[query_id]
            positives = set(record["positive_target_ids"])
            direct_ann, evidence_set, union = candidate_sets(record)
            direct_set = set(direct_ann)
            if len(direct_ann) != 100 or not union <= target_position.keys():
                raise ValueError(f"Invalid natural candidate pool for {query_id}")
            query_vector = model.relation_query(
                store.embedding_features(query_id).embedding.to(device=device, dtype=torch.float32),
                "table", "table", source_role="query",
            )
            scores = (query_vector @ target_vectors.T).float().cpu().numpy()
            exact_order, exact_ranks = exact_order_and_ranks(scores)
            exact100 = [target_ids[int(position)] for position in exact_order[:100]]
            exact_set = set(exact100)
            direct, evidence = _target_channels(
                record, retention="e2_row_coverage", scorer=scorer, store=store,
                content_keys=content_keys, top_l=20, evidence_budget=4,
                pair_batch_size=256, intervention="original_mixed",
                intervention_stats=intervention_stats,
            )
            ranked = {
                "pure_direct100": [row for row in direct if row["original_direct_member"]],
                "f1_union_direct": direct,
                "union_rrf_equal": fuse_ranked_channels(direct, evidence, rrf_k=60, fusion_mode="rrf"),
            }
            rank_by_rule = {
                rule: {row["target_id"]: offset for offset, row in enumerate(values, 1)}
                for rule, values in ranked.items()
            }
            row_by_target = {row["target_id"]: row for row in direct}
            c50 = {rule: [row["target_id"] for row in values[:50]] for rule, values in ranked.items()}
            top10 = {rule: values[:10] for rule, values in c50.items()}
            for rule in RULES:
                for k in KS:
                    if c50[rule][:k] != saved_rankings[query_id]["rankings"][rule][str(k)]["target_ids"]:
                        reproduction_errors.append({"query_id": query_id, "rule": rule, "k": k})
                delivery_rows.append({
                    "arm": config["arm"], "query_id": query_id, "source_table_id": source_by_query[query_id],
                    "rule": rule, "candidate_budget": 50, "final_top10_stage2": None,
                    "queue": [{
                        "target_id": row["target_id"], "final_rank": offset,
                        "selected_evidence_ids": list(row["selected_evidence_ids"]),
                        "direct_score": row["direct_score"], "evidence_score": row["evidence_score"],
                        "paths": [path for path in record["paths_by_target"][row["target_id"]]
                                  if path["kind"] == "direct" or path["evidence_id"] in row["selected_evidence_ids"]],
                    } for offset, row in enumerate(ranked[rule][:50], 1)],
                })
            exact_union = sorted(exact_set | evidence_set, key=lambda value: (-float(scores[target_position[value]]), value))
            for k in KS:
                if exact_union[:k] != exact100[:k]:
                    exact_invariant_failures.append({"query_id": query_id, "k": k})
            positive_rows = []
            for target_id in sorted(positives):
                selection = row_by_target.get(target_id, {})
                selected_ids = list(selection.get("selected_evidence_ids", []))
                position = target_position[target_id]
                target_score = scores[position]
                detail = {
                    "arm": config["arm"], "query_id": query_id, "source_table_id": source_by_query[query_id],
                    "query_kind": record["query_kind"], "target_id": target_id,
                    "natural_direct_rank": direct_ann.index(target_id) + 1 if target_id in direct_set else None,
                    "natural_direct_rank_censored_at": 100,
                    "exact_direct_rank": int(exact_ranks[position]),
                    "exact_direct_rank_min": int((scores > target_score).sum()) + 1,
                    "exact_direct_rank_max": int((scores >= target_score).sum()),
                    "exact_direct_score": float(target_score),
                    "in_ann_D100": target_id in direct_set, "in_exact_D100": target_id in exact_set,
                    "in_E": target_id in evidence_set, "in_U": target_id in union,
                    "ann_evidence_only": target_id in evidence_set - direct_set,
                    "exact_evidence_only": target_id in evidence_set - exact_set,
                    "relative_to_frozen_B13_exact_direct_new": target_id in evidence_set - exact_set,
                    "evidence_aggregate_rank": selection.get("evidence_rank"),
                    "final_rank_by_rule": {rule: ranks.get(target_id) for rule, ranks in rank_by_rule.items()},
                    "in_C50": {rule: target_id in values for rule, values in c50.items()},
                    "in_Top10_stage1": {rule: target_id in values for rule, values in top10.items()},
                    "actual_evidence_delivered_by_rule": {
                        rule: selected_ids if target_id in values else [] for rule, values in c50.items()
                    },
                    **witness_details(record, target_id, selected_ids),
                }
                positive_rows.append(detail)
                all_positive_pairs.append(detail)
                if detail["ann_evidence_only"]:
                    ann_new_pairs.append(detail)
                    if record["query_kind"] == "implicit" and detail["known_qet_witness_ids"]:
                        queue.append(detail)
            metrics = {
                "RawUnionRecall": recall(union, positives),
                "ANN_D100_Recall": recall(direct_set, positives),
                "exact_D100_Recall": recall(exact_set, positives),
                **{rule: {str(k): recall(values[:k], positives) for k in KS} for rule, values in c50.items()},
            }
            random_record = {"query_id": query_id, "external_target_indices": {}}
            random_per_query = {}
            draw_started = time.monotonic()
            for control_index, control in enumerate(random_metrics):
                population = sorted(evidence_set - direct_set) if control == "pool_random" else [value for value in target_ids if value not in direct_set]
                rng = np.random.default_rng(np.random.SeedSequence([15, query_index, control_index]))
                fixed, draws = random_admission(c50["union_rrf_equal"], direct_set, population, rng, 100)
                values = [recall([*fixed, *draw], positives) for draw in draws]
                slots = 50 - len(fixed)
                analytic = (len(positives & set(fixed)) + slots * len(positives & set(population)) / len(population)) / len(positives)
                random_metrics[control].append(values)
                random_record["fixed_direct_target_indices"] = [target_position[value] for value in fixed]
                random_record["external_target_indices"][control] = [[target_position[value] for value in draw] for draw in draws]
                random_per_query[control] = {
                    "external_slots": slots, "population_size": len(population),
                    "CandidateRecall@50_by_repeat": values,
                    "CandidateRecall@50_analytic_expectation": analytic,
                    "CandidateRecall@50_mean": float(np.mean(values)),
                }
            random_draw_seconds += time.monotonic() - draw_started
            draw_handle.write(json.dumps(random_record) + "\n")
            provenance.append({
                "arm": config["arm"], "query_id": query_id, "source_table_id": source_by_query[query_id],
                "query_kind": record["query_kind"], "positive_target_ids": sorted(positives),
                "positive_denominator": len(positives), "D100_ANN": direct_ann, "D100_exact": exact100,
                "E": sorted(evidence_set), "U": sorted(union), "C50": c50, "Top10_stage1": top10,
                "Top10_stage2": None, "positive_targets": positive_rows, "metrics": metrics,
                "random_controls": random_per_query,
                "rrf_C50_entered_vs_f1": sorted(set(c50["union_rrf_equal"]) - set(c50["f1_union_direct"])),
                "rrf_C50_exited_vs_f1": sorted(set(c50["f1_union_direct"]) - set(c50["union_rrf_equal"])),
                "search_vectors": saved_rankings[query_id]["search_vectors"],
                "returned_objects_before_dedup": 100 + 40 + sum(1 for values in record["paths_by_target"].values() for path in values if path["kind"] == "evidence"),
            })
            if (query_index + 1) % 100 == 0:
                print(json.dumps({"stage": "C", "queries": query_index + 1, "total": len(query_ids)}), flush=True)
    if len(ann_new_pairs) != 213 or len(queue) != 82:
        raise ValueError(f"Frozen queue reconstruction differs: {len(ann_new_pairs)} / {len(queue)}")
    write_gzip(output / "candidate_provenance.jsonl.gz", provenance)
    write_gzip(output / "all_positive_pairs.jsonl.gz", all_positive_pairs)
    write_gzip(output / "B13_ann_evidence_only_positive_pairs.jsonl.gz", ann_new_pairs)
    write_gzip(output / "B13_evidence_only_known_witness_cases.jsonl.gz", queue)
    write_gzip(output / "delivery_queues.jsonl.gz", delivery_rows)
    with (output / "B13_evidence_only_known_witness_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "source_table_id", "target_id", "natural_direct_rank", "exact_direct_rank", "exact_evidence_only", "raw_supported_row_count", "retained_supported_row_count", "retention_keeps_known_witness", "evidence_aggregate_rank", "known_qet_modalities", "known_qet_witness_ids", "selected_evidence_ids", "final_rank_by_rule", "in_C50", "needs_independent_content_review"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (list, dict)) else row[key] for key in fields} for row in queue)
    random_summary = {}
    for control, values in random_metrics.items():
        matrix = np.asarray(values)
        random_summary[control] = {}
        for kind in ("all", "implicit", "explicit"):
            mask = np.array([kind == "all" or row["query_kind"] == kind for row in provenance])
            means = matrix[mask].mean(axis=0)
            random_summary[control][kind] = {
                "CandidateRecall@50_mean": float(means.mean()), "monte_carlo_sd": float(means.std(ddof=1)),
                "monte_carlo_95pct_range": np.quantile(means, [0.025, 0.975]).tolist(),
                "monte_carlo_min_max": [float(means.min()), float(means.max())],
                "by_repeat": means.tolist(), "Student_repeats": 0, "random_repeats": 100,
                "analytic_expectation": float(np.mean([row["random_controls"][control]["CandidateRecall@50_analytic_expectation"] for row, keep in zip(provenance, mask) if keep])),
            }
    stats = {}
    for left, right in (("union_rrf_equal", "f1_union_direct"), ("f1_union_direct", "pure_direct100"), ("union_rrf_equal", "pure_direct100")):
        stats[f"{left}_minus_{right}"] = {}
        for k in KS:
            deltas = np.array([row["metrics"][left][str(k)] - row["metrics"][right][str(k)] for row in provenance])
            stats[f"{left}_minus_{right}"][str(k)] = {
                **_bootstrap(deltas, [row["source_table_id"] for row in provenance], seed=15),
                "win_loss_tie_queries": [int((deltas > 0).sum()), int((deltas < 0).sum()), int((deltas == 0).sum())],
            }
    summary = {
        "format_version": 1, "status": "complete", "arm": config["arm"],
        "metrics": summarize_metrics(provenance), "legal_targets": len(target_ids),
        "ann_evidence_only_positive_pairs": len(ann_new_pairs),
        "ann_evidence_only_by_kind": dict(Counter(row["query_kind"] for row in ann_new_pairs)),
        "exact_evidence_only_positive_pairs": sum(row["exact_evidence_only"] for row in all_positive_pairs),
        "ann_evidence_only_pairs_outside_exact100": sum(row["exact_evidence_only"] for row in ann_new_pairs),
        "ann_evidence_only_pairs_inside_exact100": sum(row["in_exact_D100"] for row in ann_new_pairs),
        "priority_queue": {
            "pairs": len(queue), "queries": len({row["query_id"] for row in queue}),
            "source_groups": len({row["source_table_id"] for row in queue}),
            "outside_exact100": sum(row["exact_evidence_only"] for row in queue),
            "text_pairs": sum("text" in row["known_qet_modalities"] for row in queue),
            "image_pairs": sum("image" in row["known_qet_modalities"] for row in queue),
            "both_modalities": sum(len(row["known_qet_modalities"]) == 2 for row in queue),
            "raw_supported_row_histogram": dict(Counter(row["raw_supported_row_count"] for row in queue)),
            "retained_supported_row_histogram": dict(Counter(row["retained_supported_row_count"] for row in queue)),
            "retention_keeps_known_witness_pairs": sum(row["retention_keeps_known_witness"] for row in queue),
            "retention_loses_all_known_witness_pairs": sum(not row["retention_keeps_known_witness"] for row in queue),
            "delivered_pairs_by_rule": {rule: sum(row["in_C50"][rule] for row in queue) for rule in RULES},
            "top10_pairs_by_rule": {rule: sum(row["in_Top10_stage1"][rule] for row in queue) for rule in RULES},
            "delivered_with_known_witness_by_rule": {rule: sum(row["in_C50"][rule] and row["retention_keeps_known_witness"] for row in queue) for rule in RULES},
            "exact_new_delivered_with_known_witness_by_rule": {rule: sum(row["exact_evidence_only"] and row["in_C50"][rule] and row["retention_keeps_known_witness"] for row in queue) for rule in RULES},
        },
        "random_controls": random_summary, "paired_source_group_statistics": stats,
        "validation": {
            "saved_rankings_reproduced": not reproduction_errors,
            "ranking_reproduction_errors": reproduction_errors,
            "C4_exact_direct_invariant_holds": not exact_invariant_failures,
            "C4_failures": exact_invariant_failures,
            "all_213_exact_ranks_exported": all(row["exact_direct_rank"] is not None for row in ann_new_pairs),
            "all_query_C50_sizes_equal_50": all(len(ids) == 50 for row in provenance for ids in row["C50"].values()),
            "frozen_82_reconstruction_count_matches": True,
            "independent_value_or_join_validation_completed": False,
        },
        "cost": {
            "wall_seconds": time.monotonic() - started, "random_sampling_and_scoring_cpu_wall_seconds": random_draw_seconds,
            "new_Student_updates": 0, "new_Teacher_inferences": 0,
            "search_vectors_per_query": sorted({row["search_vectors"] for row in provenance}),
            "mean_returned_objects_before_dedup": float(np.mean([row["returned_objects_before_dedup"] for row in provenance])),
            "retrieval_reused": True, "isolated_retrieval_latency_comparison": None,
        },
        "source_dataset_manifest": {"path": str(dataset_manifest), "sha256": checkpoint_fingerprint(dataset_manifest)},
        "interpretation_boundary": "Known qrels/witnesses are incomplete. These are Stage1 path/admission facts, not verified value recovery or correct joins; random intervals are Monte Carlo variation, not Student seeds.",
    }
    write_json(output / "SUMMARY.json", summary)
    export_verification(root)
    print(json.dumps({"status": "complete", "output": str(output), "priority_queue": summary["priority_queue"], "validation": summary["validation"]}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task", choices=("freeze", "analyze", "verification"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.task == "freeze":
        print(json.dumps(freeze(arguments.root), indent=2))
    elif arguments.task == "verification":
        export_verification(arguments.root)
    else:
        analyze(arguments)

#!/usr/bin/env python
"""Recompute the six-cell R15 interaction and endpoint witness diagnostics."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from finalize_stage1_r13 import _source_map
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.row_support import load_evidence_content_keys
from run_stage1_r11_task_e import _evidence_paths, _sigmoid, select_evidence
from run_stage1_r13 import _paths


KS = (10, 20, 50)
RULES = ("f1_union_direct", "union_rrf_equal", "pure_direct100")
KINDS = ("all", "implicit", "explicit")
ARMS = {
    "s_full": "stage1_optimization_r13_20260909/taskD_witness_supervision/p_s_target_only",
    "s_eoff": "stage1_optimization_r14_20260909/stage1_B_branch_ablation/b_d_e_loss_off_seed13",
    "l_full": "stage1_optimization_r14_20260909/stage1_M_projection_capacity/m_l_linear_residual_seed13",
    "n_full": "stage1_optimization_r14_20260909/stage1_M_projection_capacity/m_n_gelu_residual_seed13",
    "l_eoff": "stage1_optimization_r15_20260909/stageI_interaction/l_eoff_seed13",
    "n_eoff": "stage1_optimization_r15_20260909/stageI_interaction/n_eoff_seed13",
}
CONTRASTS = {
    "l_eoff_minus_s_eoff": {"l_eoff": 1, "s_eoff": -1},
    "n_eoff_minus_l_eoff": {"n_eoff": 1, "l_eoff": -1},
    "n_eoff_minus_s_eoff": {"n_eoff": 1, "s_eoff": -1},
    "s_full_minus_s_eoff": {"s_full": 1, "s_eoff": -1},
    "l_full_minus_l_eoff": {"l_full": 1, "l_eoff": -1},
    "n_full_minus_n_eoff": {"n_full": 1, "n_eoff": -1},
    "n_full_minus_l_full": {"n_full": 1, "l_full": -1},
    "I_L": {"l_full": 1, "l_eoff": -1, "s_full": -1, "s_eoff": 1},
    "I_N": {"n_full": 1, "n_eoff": -1, "s_full": -1, "s_eoff": 1},
    "I_N_minus_L": {"n_full": 1, "n_eoff": -1, "l_full": -1, "l_eoff": 1},
}


def read_rows(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(row["query_id"]): row for row in map(json.loads, handle)}


def dependency(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path)}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_columns(
    deltas: np.ndarray, sources: list[str], *, iterations: int, seed: int
) -> list[dict[str, Any]]:
    """Resample whole source groups while preserving the query-macro estimand."""

    groups = sorted(set(sources))
    group_index = {value: index for index, value in enumerate(groups)}
    sums = np.zeros((len(groups), deltas.shape[1]), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.int64)
    for row, source in zip(deltas, sources):
        sums[group_index[source]] += row
        counts[group_index[source]] += 1
    sampled = np.empty((iterations, deltas.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, 100):
        stop = min(iterations, start + 100)
        draw = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        weights = np.asarray(
            [np.bincount(row, minlength=len(groups)) for row in draw],
            dtype=np.float64,
        )
        sampled[start:stop] = (weights @ sums) / (weights @ counts)[:, None]
    intervals = np.quantile(sampled, [0.025, 0.975], axis=0)
    return [
        {
            "point_delta": float(deltas[:, column].mean()),
            "ci95_percentile": intervals[:, column].tolist(),
            "win_queries": int((deltas[:, column] > 1e-12).sum()),
            "loss_queries": int((deltas[:, column] < -1e-12).sum()),
            "tie_queries": int((np.abs(deltas[:, column]) <= 1e-12).sum()),
            "bootstrap_probability_delta_gt_0": float((sampled[:, column] > 0).mean()),
            "queries": len(sources),
            "source_groups": len(groups),
        }
        for column in range(deltas.shape[1])
    ]


def candidate_source(paths: list[dict[str, Any]]) -> dict[str, Any]:
    direct = any(path.get("kind") == "direct" for path in paths)
    modalities = sorted(
        {str(path["evidence_type"]) for path in paths if path.get("kind") == "evidence"}
    )
    return {
        "ann_direct100": direct,
        "evidence_target_member": bool(modalities),
        "raw_union_member": bool(paths),
        "evidence_modalities": modalities,
        "source": "both" if direct and modalities else "direct" if direct else "evidence" if modalities else "absent",
    }


def raw_lse_responsibility(
    paths: list[dict[str, Any]], known_ids: set[str], *, temperature: float
) -> float | None:
    """Responsibility for the real untruncated LSE rule, on this raw pool."""

    evidence = _evidence_paths(paths)
    if not evidence:
        return None
    maximum = max(float(path["path_score"]) for path in evidence)
    weights = [math.exp((float(path["path_score"]) - maximum) / temperature) for path in evidence]
    return sum(
        weight for weight, path in zip(weights, evidence)
        if str(path["evidence_id"]) in known_ids
    ) / sum(weights)


def greedy_responsibility(
    paths: list[dict[str, Any]], selected: list[str], support: dict[str, list[float]],
    known_ids: set[str],
) -> dict[str, Any]:
    """Attribute the deployed greedy score by its actual sequential row gains."""

    by_id = {str(path["evidence_id"]): path for path in _evidence_paths(paths)}
    if not selected:
        return {"score": None, "known_mass": None, "path_marginal_gains": []}
    current = [0.0] * len(support[selected[0]])
    gains = []
    for evidence_id in selected:
        quality = _sigmoid(float(by_id[evidence_id]["path_score"]))
        updated = [max(old, quality * value) for old, value in zip(current, support[evidence_id])]
        gain = (sum(updated) - sum(current)) / len(current)
        gains.append({"evidence_id": evidence_id, "gain": gain, "known": evidence_id in known_ids})
        current = updated
    score = sum(current) / len(current)
    return {
        "score": score,
        "known_mass": sum(row["gain"] for row in gains if row["known"]) / score if score else None,
        "path_marginal_gains": gains,
    }


def supported_rows(ids: set[str], rows_by_id: dict[str, list[int]]) -> list[int]:
    return sorted({int(row) for evidence_id in ids for row in rows_by_id.get(evidence_id, [])})


def pair_witness(
    record: dict[str, Any], target_id: str, ranking: dict[str, Any],
    qe_by_modality: dict[str, set[str]], store: FeatureStore, content_keys: dict[str, str],
    support_cache: dict[str, list[float]],
) -> dict[str, Any]:
    paths = record["paths_by_target"].get(target_id, [])
    evidence = _evidence_paths(paths)
    known = set(record.get("positive_evidence_by_target", {}).get(target_id, []))
    rows_by_id = record.get("positive_evidence_rows_by_target", {}).get(target_id, {})
    selected, score = select_evidence(
        "e2_row_coverage", paths, query_id=record["query_id"], store=store,
        content_keys=content_keys, top_l=20, budget=4, support_cache=support_cache,
    )
    responsibility = greedy_responsibility(paths, selected, support_cache, known)
    if score is not None and not math.isclose(score, responsibility["score"], abs_tol=1e-12):
        raise ValueError("Replayed deployed greedy score does not equal marginal-gain sum")
    qet_ids = {str(path["evidence_id"]) for path in evidence}
    retained_known = set(selected) & known
    modalities = {}
    for modality in ("text", "image"):
        modality_paths = [path for path in evidence if path["evidence_type"] == modality]
        known_paths = [path for path in modality_paths if str(path["evidence_id"]) in known]
        modalities[modality] = {
            "known_QE_occurrences": len(known & qe_by_modality[modality]),
            "known_QE_pair": bool(known & qe_by_modality[modality]),
            "raw_QET_occurrences": len(modality_paths),
            "known_QET_occurrences": len(known_paths),
            "known_QET_pair": bool(known_paths),
            "retained_known_occurrences": sum(str(path["evidence_id"]) in retained_known for path in modality_paths),
            "retained_known_pair": any(str(path["evidence_id"]) in retained_known for path in modality_paths),
        }
    delivery = {}
    for rule in RULES:
        delivery[rule] = {}
        for k in KS:
            ids = ranking["rankings"][rule][str(k)]["target_ids"]
            in_list = target_id in ids
            delivery[rule][str(k)] = {
                "delivered": in_list,
                "final_rank": ids.index(target_id) + 1 if in_list else None,
                "delivered_evidence_ids": selected if in_list else [],
                "known_witness_delivered": in_list and bool(retained_known),
                "known_supported_row_count": len(supported_rows(retained_known, rows_by_id)) if in_list else 0,
            }
    return {
        "target_id": target_id, "known_witness_annotated": bool(known),
        "known_evidence_ids": sorted(known), "candidate_source": candidate_source(paths),
        "raw_QET_occurrences": len(evidence),
        "known_QE_occurrences": len(known & set.union(*qe_by_modality.values())),
        "known_QET_occurrences": sum(str(path["evidence_id"]) in known for path in evidence),
        "known_QE_pair": bool(known & set.union(*qe_by_modality.values())),
        "known_QET_pair": bool(known & qet_ids),
        "modalities": modalities,
        "raw_top_path_id": str(evidence[0]["evidence_id"]) if evidence else None,
        "raw_top_path_known": str(evidence[0]["evidence_id"]) in known if evidence else None,
        "raw_pool_lse_responsibility_known": raw_lse_responsibility(paths, known, temperature=1.0),
        "raw_known_supported_rows": supported_rows(qet_ids & known, rows_by_id),
        "deployed_selected_evidence_ids": selected,
        "deployed_known_evidence_ids": sorted(retained_known),
        "deployed_known_supported_rows": supported_rows(retained_known, rows_by_id),
        "deployed_greedy_marginal_responsibility_known": responsibility,
        "delivery": delivery,
        "independent_value_verification": None,
        "correct_join_verified": None,
        "unknown_witness_is_not_negative": True,
    }


def hub_summary(counter: Counter[str]) -> dict[str, Any]:
    return {
        "distinct_objects": len(counter), "query_object_occurrences": sum(counter.values()),
        "highest_frequency_query_count": max(counter.values(), default=0),
        "top20": [{"object_id": key, "query_count": value} for key, value in counter.most_common(20)],
    }


def summarize_witness(rows: list[dict[str, Any]], hubs: dict[str, Counter[str]]) -> dict[str, Any]:
    strata = {}
    for kind in KINDS:
        selected = [row for row in rows if kind == "all" or row["query_kind"] == kind]
        strata[kind] = {
            "positive_pairs": len(selected),
            "known_witness_annotated_pairs": sum(row["known_witness_annotated"] for row in selected),
            "raw_QET_occurrences_on_positive_targets": sum(row["raw_QET_occurrences"] for row in selected),
            "known_QE_occurrences": sum(row["known_QE_occurrences"] for row in selected),
            "known_QET_occurrences": sum(row["known_QET_occurrences"] for row in selected),
            "known_QE_unique_pairs": sum(row["known_QE_pair"] for row in selected),
            "known_QET_unique_pairs": sum(row["known_QET_pair"] for row in selected),
            "retained_known_unique_pairs": sum(bool(row["deployed_known_evidence_ids"]) for row in selected),
            "known_raw_top_path_pairs": sum(row["raw_top_path_known"] is True for row in selected),
            "raw_supported_rows_0_to_5": {str(n): sum(len(row["raw_known_supported_rows"]) == n for row in selected) for n in range(6)},
            "retained_supported_rows_0_to_5": {str(n): sum(len(row["deployed_known_supported_rows"]) == n for row in selected) for n in range(6)},
            "modality": {
                modality: {key: sum(row["modalities"][modality][key] for row in selected) for key in selected[0]["modalities"][modality]}
                for modality in ("text", "image")
            },
            "delivered_known_pairs": {
                rule: {str(k): sum(row["delivery"][rule][str(k)]["known_witness_delivered"] for row in selected) for k in KS}
                for rule in RULES
            },
        }
        for label, values in (
            ("raw_pool_lse_responsibility", [row["raw_pool_lse_responsibility_known"] for row in selected]),
            ("deployed_greedy_known_responsibility", [row["deployed_greedy_marginal_responsibility_known"]["known_mass"] for row in selected]),
        ):
            present = [value for value in values if value is not None]
            strata[kind][label] = {
                "eligible_pairs_with_evidence": len(present),
                "conditional_mean": float(np.mean(present)) if present else None,
                "all_positive_pair_mean_missing_zero": sum(present) / len(selected),
            }
    return {"strata": strata, "hubs": {key: hub_summary(value) for key, value in hubs.items()}}


def run(root: Path, *, iterations: int, seed: int, cpu_threads: int) -> dict[str, Any]:
    started = time.monotonic()
    torch.set_num_threads(cpu_threads)
    output = root / "work/stage1_optimization_r15_20260909"
    statistics_dir = output / "stageI_interaction/statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    source_by_query, source_manifest = _source_map(root)
    paths = _paths(root)
    store = FeatureStore.from_path(paths["features"], cache_size=60_000)
    content_keys, content_hash = load_evidence_content_keys(paths["evidence_content_keys"])
    rankings = {}
    metrics = {}
    provenance = {}
    witness_summaries = {}
    dependencies = {"source_manifest": dependency(source_manifest), "content_keys_sha256": content_hash}
    b13_direct_by_query = {}
    witness_path = output / "witness_funnel.jsonl.gz"
    with gzip.open(witness_path, "wt", encoding="utf-8") as witness_handle:
        for arm, relative in ARMS.items():
            directory = root / "work" / relative / "evaluation_step178"
            metrics[arm] = json.loads((directory / "metrics.json").read_text())
            rankings[arm] = read_rows(directory / "rankings.jsonl.gz")
            dependencies[arm] = {name: dependency(directory / name) for name in ("metrics.json", "rankings.jsonl.gz", "path_pool.jsonl.gz")}
            provenance[arm] = {}
            hubs = {key: Counter() for key in ("direct100", "QE_text", "QE_image", "ET_text", "ET_image", *[f"delivery50_{rule}" for rule in RULES])}
            raw_path_counts: Counter[str] = Counter()
            retention_checks = 0
            existing_top10 = {
                row["query_id"]: {item["target_id"]: item["selected_evidence_ids"] for item in row["final_top10"]}
                for row in metrics[arm]["primary"]["per_query"]
            }
            arm_rows = []
            pool_macro = {kind: {key: [] for key in ("direct100", "raw_union", "evidence", "evidence_outside_own_direct100", "evidence_outside_B13_direct100")} for kind in KINDS}
            with gzip.open(directory / "path_pool.jsonl.gz", "rt", encoding="utf-8") as handle:
                for record in map(json.loads, handle):
                    query_id = record["query_id"]
                    ranking = rankings[arm][query_id]
                    positives = set(record["positive_target_ids"])
                    if positives != set(ranking["positive_target_ids"]):
                        raise ValueError("Path-pool and ranking qrels differ")
                    direct_ids = set()
                    evidence_targets = set()
                    et_targets = {modality: set() for modality in ("text", "image")}
                    qe_ids = {modality: set() for modality in ("text", "image")}
                    for target_id, target_paths in record["paths_by_target"].items():
                        for path in target_paths:
                            if path["kind"] == "direct":
                                direct_ids.add(target_id)
                                raw_path_counts["direct_occurrences"] += 1
                            else:
                                modality = path["evidence_type"]
                                raw_path_counts[f"QET_{modality}_occurrences"] += 1
                                qe_ids[modality].add(str(path["evidence_id"]))
                                evidence_targets.add(target_id)
                                et_targets[modality].add(target_id)
                    if arm == "s_full":
                        b13_direct_by_query[query_id] = direct_ids
                    sets = {"direct100": direct_ids, "evidence": evidence_targets, "raw_union": direct_ids | evidence_targets,
                            "evidence_outside_own_direct100": evidence_targets - direct_ids,
                            "evidence_outside_B13_direct100": evidence_targets - b13_direct_by_query[query_id]}
                    for kind in ("all", record["query_kind"]):
                        for label, values in sets.items():
                            pool_macro[kind][label].append(len(positives & values) / len(positives))
                    hubs["direct100"].update(direct_ids)
                    for modality in qe_ids:
                        hubs[f"QE_{modality}"].update(qe_ids[modality])
                        hubs[f"ET_{modality}"].update(et_targets[modality])
                    for rule in RULES:
                        hubs[f"delivery50_{rule}"].update(ranking["rankings"][rule]["50"]["target_ids"])
                    provenance[arm][query_id] = {target: candidate_source(record["paths_by_target"].get(target, [])) for target in positives}
                    support_cache = {}
                    for target_id in sorted(positives):
                        row = {"arm": arm, "query_id": query_id, "source_table_id": source_by_query[query_id],
                               "query_kind": record["query_kind"], "positive_denominator": len(positives),
                               **pair_witness(record, target_id, ranking, qe_ids, store, content_keys, support_cache)}
                        arm_rows.append(row)
                        if target_id in existing_top10[query_id]:
                            if row["deployed_selected_evidence_ids"] != existing_top10[query_id][target_id]:
                                raise ValueError(f"Retention replay disagrees with saved F1 output: {arm}/{query_id}/{target_id}")
                            retention_checks += 1
                        witness_handle.write(json.dumps(row) + "\n")
            witness_summaries[arm] = summarize_witness(arm_rows, hubs)
            witness_summaries[arm]["all_raw_pool_path_occurrences"] = dict(raw_path_counts)
            witness_summaries[arm]["deployed_retention_replay_validation"] = {
                "saved_F1_top10_positive_pairs_checked": retention_checks,
                "selected_evidence_order_exact_match": True,
                "scope": "all saved F1 top10 positive pairs; every positive target additionally checks greedy gain sum against actual select_evidence score",
            }
            witness_summaries[arm]["raw_pool_query_macro_recall"] = {
                kind: {key: float(np.mean(values)) for key, values in value.items()} for kind, value in pool_macro.items()
            }
            print(json.dumps({"arm": arm, "positive_pairs": len(arm_rows), "primary_R10": metrics[arm]["primary"]["recall@10"]}), flush=True)

    query_ids = sorted(rankings["s_full"])
    for arm, rows in rankings.items():
        if set(rows) != set(query_ids):
            raise ValueError(f"Query population differs: {arm}")
        for query_id in query_ids:
            reference = rankings["s_full"][query_id]
            for key in ("positive_target_ids", "positive_denominator", "query_kind"):
                if rows[query_id][key] != reference[key]:
                    raise ValueError(f"Frozen query denominator differs: {arm}/{query_id}/{key}")
    per_query = []
    delta_rows = {contrast: [] for contrast in CONTRASTS}
    columns = [(contrast, rule, k) for contrast in CONTRASTS for rule in RULES for k in KS]
    matrix = np.empty((len(query_ids), len(columns)), dtype=np.float64)
    for query_index, query_id in enumerate(query_ids):
        reference = rankings["s_full"][query_id]
        shared = {"query_id": query_id, "source_table_id": source_by_query[query_id],
                  "query_kind": reference["query_kind"], "positive_target_ids": reference["positive_target_ids"],
                  "positive_denominator": reference["positive_denominator"]}
        per_query.append({**shared, "arms": {
            arm: {"rankings": rankings[arm][query_id]["rankings"], "positive_candidate_sources": provenance[arm][query_id],
                  "union_unique_targets": rankings[arm][query_id]["union_unique_targets"], "search_vectors": rankings[arm][query_id]["search_vectors"]}
            for arm in ARMS
        }})
        for contrast, coefficients in CONTRASTS.items():
            row = {**shared, "contrast": contrast, "arm_coefficients": coefficients, "metrics": {}}
            for rule in RULES:
                row["metrics"][rule] = {}
                for k in KS:
                    delta = sum(weight * rankings[arm][query_id]["rankings"][rule][str(k)]["recall"] for arm, weight in coefficients.items())
                    item = {"delta": delta, "outcome": "win" if delta > 1e-12 else "loss" if delta < -1e-12 else "tie"}
                    if len(coefficients) == 2:
                        left = next(arm for arm, weight in coefficients.items() if weight == 1)
                        right = next(arm for arm, weight in coefficients.items() if weight == -1)
                        left_hits = set(rankings[left][query_id]["rankings"][rule][str(k)]["hit_ids"])
                        right_hits = set(rankings[right][query_id]["rankings"][rule][str(k)]["hit_ids"])
                        for name, ids in (("positive_entered", left_hits - right_hits), ("positive_exited", right_hits - left_hits)):
                            item[name] = [{"target_id": target_id, "experimental_source": provenance[left][query_id][target_id], "control_source": provenance[right][query_id][target_id]} for target_id in sorted(ids)]
                    else:
                        item["arm_hit_ids"] = {arm: rankings[arm][query_id]["rankings"][rule][str(k)]["hit_ids"] for arm in coefficients}
                    row["metrics"][rule][str(k)] = item
            delta_rows[contrast].append(row)
        for column_index, (contrast, rule, k) in enumerate(columns):
            matrix[query_index, column_index] = delta_rows[contrast][-1]["metrics"][rule][str(k)]["delta"]

    summaries = {contrast: {kind: {rule: {} for rule in RULES} for kind in KINDS} for contrast in CONTRASTS}
    for kind in KINDS:
        indices = [index for index, row in enumerate(per_query) if kind == "all" or row["query_kind"] == kind]
        bootstraps = bootstrap_columns(matrix[indices], [per_query[index]["source_table_id"] for index in indices], iterations=iterations, seed=seed)
        for (contrast, rule, k), bootstrap in zip(columns, bootstraps):
            summaries[contrast][kind][rule][str(k)] = bootstrap
    contrast_paths = {}
    for contrast, rows in delta_rows.items():
        path = statistics_dir / f"{contrast}.jsonl.gz"
        write_rows(path, rows)
        contrast_paths[contrast] = dependency(path)
    per_query_path = output / "per_query_metrics.jsonl.gz"
    write_rows(per_query_path, per_query)
    score_rows = {}
    validation = {}
    for arm in ARMS:
        score_rows[arm] = {}
        for kind in KINDS:
            selected = [row for row in rankings[arm].values() if kind == "all" or row["query_kind"] == kind]
            score_rows[arm][kind] = {rule: {str(k): float(np.mean([row["rankings"][rule][str(k)]["recall"] for row in selected])) for k in KS} for rule in RULES}
        validation[arm] = {
            rule: {str(k): abs(score_rows[arm]["all"][rule][str(k)] - metrics[arm][key][f"recall@{k}"]) < 1e-12 for k in KS}
            for rule, key in (("f1_union_direct", "primary"), ("union_rrf_equal", "sensitivity_equal_union_rrf"), ("pure_direct100", "pure_direct100"))
        }
    summary = {
        "status": "complete", "seed": 13, "queries": len(query_ids),
        "source_groups": len({source_by_query[query_id] for query_id in query_ids}),
        "primary": "all-query macro F1 target recall@10 with fixed G_q",
        "recall_unit": "fraction; multiply by 100 for percent and delta percentage points",
        "arm_metrics": score_rows, "contrasts": summaries, "contrast_files": contrast_paths,
        "bootstrap": {"unit": "source_table_id", "iterations": iterations, "random_seed": seed,
                      "paired_joint_resampling": True, "exploratory_dev_intervals": True,
                      "scope": "conditional on frozen S0, teacher, lake, candidates and one Student seed; not equivalence or full-pipeline variance"},
        "metric_recomputation_pass": validation,
        "inputs": dependencies, "per_query_metrics": dependency(per_query_path), "witness_funnel": dependency(witness_path),
        "cost": {arm: {**metrics[arm]["cost"], "isolated_latency": False} for arm in ARMS},
        "analysis_seconds": time.monotonic() - started,
        "code": dependency(Path(__file__)),
    }
    witness_payload = {
        "status": "complete", "arms": witness_summaries,
        "aggregation": {
            "training_rule": "untruncated logsumexp(QE_raw_logit + ET_raw_logit), temperature=1; top_k=4 inactive for plain logsumexp",
            "raw_pool_responsibility": "softmax at actual training temperature=1 over all natural evidence paths; descriptive replay, not the fixed training bag",
            "deployed_rule": "e2_row_coverage; exact-content dedup; best path per content; top20; greedy positive marginal gain; budget4",
            "deployed_quality": "sigmoid(QE_raw_logit + ET_raw_logit)",
            "deployed_row_support": "clip((raw frozen row/evidence dot product + 1)/2, 0, 1)",
            "deployed_score": "mean over rows of max selected evidence quality * row_support",
            "deployed_lse_temperature": None,
            "deployed_responsibility": "fraction of sequential greedy marginal gains attributable to known selected witness; actual order/retention, not LSE softmax",
            "selection_uses_known_labels": False,
        },
        "counting": "known QE counts distinct evidence once per positive pair; QET occurrences count paths; unique positive pairs separate; modalities may overlap",
        "support_interpretation": "known-label row support, not independently verified attribute-value recovery or correct joins",
        "raw_pool_scope": "full lake natural retrieval, D100 plus 20 text and 20 image each ET20, 43 search vectors per query",
        "independent_verification": None,
    }
    write_json(statistics_dir / "interaction_summary.json", summary)
    write_json(statistics_dir / "witness_summary.json", witness_payload)
    print(json.dumps({"status": "complete", "summary": str(statistics_dir / "interaction_summary.json"), "seconds": summary["analysis_seconds"]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=13)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    run(args.root.resolve(), iterations=args.bootstrap_iterations, seed=args.bootstrap_seed, cpu_threads=args.cpu_threads)


if __name__ == "__main__":
    main()

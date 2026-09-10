#!/usr/bin/env python
"""Validate final R15 evidence, refreshing only explicitly allowed source snapshots.

This checks saved results without training, model scoring, GPU use, or remote calls.
It never upgrades unavailable historical evidence or structural HTML QA to full proof.
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_stage1_r15_interaction import ARMS, CONTRASTS, KS, RULES
from finalize_stage1_r13 import _source_map


ARM_LABELS = {"s_full": "S-full / B13", "s_eoff": "S-Eoff", "l_full": "L-full",
              "l_eoff": "L-Eoff", "n_full": "N-full", "n_eoff": "N-Eoff"}
RELATIONS = {"table_to_table", "table_to_text", "table_to_image", "text_to_table", "image_to_table"}
RECEIPT = "FINAL_DELIVERY_VALIDATION.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    require(path.name != ".env.openai", "Protected environment files are outside this audit")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def unique_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple, dict]:
    result = {tuple(row[key] for key in keys): row for row in rows}
    require(len(result) == len(rows), f"Duplicate records for key {keys}")
    return result


def check_saved_flags(output: Path) -> dict[str, Any]:
    deployment = read_json(output / "stageG_correctness/deployment_supplement/SUMMARY.json")
    scores = read_json(output / "stageG_correctness/training_scores/summary.json")
    training = read_json(output / "stageI_interaction/statistics/training_invariants.json")
    candidates = read_json(output / "stageC_candidate_delivery/SUMMARY.json")
    interaction = read_json(output / "stageI_interaction/statistics/interaction_summary.json")
    tests = read_json(output / "tests/verification.json")
    require(deployment["passed"] is True, "Deployment supplement failed")
    require(scores["passed"] is True, "Training score formula audit failed")
    require(training["training_invariants_status"] == "passed", "Training invariants failed")
    require(all(all(arm["checks"].values()) for arm in training["arms"].values()), "A saved training invariant failed")
    require(training["total_new_student_updates"] == 356 and training["new_teacher_inference"] == 0, "Training budget differs")
    require(all(training["arms"][arm]["step0_files_byte_identical"] for arm in ("l_eoff", "n_eoff")), "Initialization bytes differ")
    require(all(v for arm in interaction["metric_recomputation_pass"].values() for rule in arm.values() for v in rule.values()), "Saved metric recomputation failed")
    for key in ("saved_rankings_reproduced", "C4_exact_direct_invariant_holds", "all_213_exact_ranks_exported", "all_query_C50_sizes_equal_50", "frozen_82_reconstruction_count_matches"):
        require(candidates["validation"][key] is True, f"Candidate audit failed: {key}")
    require(tests["status"] == "passed" and tests["failed"] == 0 and tests["passed"] == 35,
            "Expected 35 passing recorded unit checks")
    require(tests["working_directory"] == "/tmp" and tests["environment"] == "MMDD", "Unit test isolation/environment differs")
    require(interaction["bootstrap"]["iterations"] == 10_000 and interaction["bootstrap"]["paired_joint_resampling"], "Bootstrap protocol differs")
    require(candidates["validation"]["independent_value_or_join_validation_completed"] is False, "Unplanned independent validation claim")
    return {"deployment_numerical_pass": True, "training_formula_pass": True,
            "training_invariants_pass": True, "candidate_recomputation_pass": True,
            "recorded_unit_tests_passed": 35, "bootstrap_iterations": 10_000,
            "new_student_updates": 356, "new_teacher_inferences": 0}


def check_entrypoint_record(root: Path, record: dict[str, Any], names: set[str],
                           manifest_key: str = "training_manifest_dependencies") -> dict[str, Any]:
    """Prove recorded bytes and claimed function identity without importing old code."""
    fingerprints = {}
    functions = {}
    for key in ("current_source", "recovered_source"):
        expected = record[key]
        path = Path(expected["path"]).resolve()
        require(path.is_relative_to(root), "Recovered/current entrypoint is outside the repository")
        actual = dependency(path)
        require(actual == expected, f"Entrypoint dependency differs: {key}")
        fingerprints[key] = actual
        functions[key] = {node.name: ast.dump(node, include_attributes=False)
                          for node in ast.parse(path.read_text(encoding="utf-8")).body
                          if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    claimed = record["current_training_and_shared_function_AST_unchanged"]
    require(set(claimed) == names and all(value is True for value in claimed.values()), "Entrypoint AST claim population differs")
    require(all(functions["current_source"].get(name) == functions["recovered_source"].get(name)
                and name in functions["current_source"] for name in names), "Recovered training/shared function AST differs")
    manifests = []
    for expected in record[manifest_key]:
        path = Path(expected["path"]).resolve()
        require(path.is_relative_to(root), "Training manifest is outside the repository")
        actual = dependency(path)
        require(actual == expected, "Training manifest dependency changed")
        require(read_json(path)["code_sha256"] == fingerprints["recovered_source"]["sha256"], "Recovered source does not match the recorded training hash")
        manifests.append(actual)
    require(bool(manifests), "No training manifest supports recovered source identity")
    return {**fingerprints, "training_manifests_verified": manifests,
            "function_ASTs_independently_equal": sorted(names),
            "recorded_entrypoint_file_identity_verified": True,
            "historical_code_executed_by_validator": False}


def check_recovered_entrypoints(root: Path, output: Path) -> dict[str, Any]:
    from audit_stage1_r15_runtime_sources import audit as audit_runtime_sources

    recovery = read_json(output / "RECOVERED_TRAINING_SOURCE.json")
    require(recovery["status"] == "hash_matched_training_entrypoint_reconstructed", "R15 entrypoint recovery did not pass")
    require(recovery["both_training_manifest_hashes_match"] is True, "Both R15 manifest matches not recorded")
    require(recovery["historical_code_executed_during_recovery"] is False
            and recovery["optimizer_tensors_recovered"] is False, "Recovery overstates execution or tensor recovery")
    r15_names = {"_dependency", "_r14_arm_dir", "_save_r15_checkpoint", "arm_directory",
                 "freeze_plan", "optimizer_audit", "output_root", "train_arm"}
    r15 = check_entrypoint_record(root, recovery, r15_names)
    require(len(r15["training_manifests_verified"]) == 2, "R15 manifest population differs")
    r14_record = recovery["r14_entrypoint"]
    require(r14_record["status"] == "hash_matched_training_entrypoint_reconstructed"
            and r14_record["all_seven_training_manifest_entrypoints_match"] is True
            and r14_record["optimizer_definition_AST_unchanged"] is True
            and r14_record["optimizer_tensors_recovered"] is False, "R14 entrypoint recovery status/limits differ")
    r14_names = {"_adapter_diagnostics", "_arm_directory", "_arm_id", "_calibrate_projection_scales",
                 "_make_model", "_optimizer", "_output_root", "_pair_counts", "_save_checkpoint", "_schedule",
                 "_sha256_text", "_training_ids_by_type", "freeze_plan", "train_arm"}
    r14 = check_entrypoint_record(root, r14_record, r14_names, "reconstructed_source_training_manifests")
    require(len(r14["training_manifests_verified"]) == 3, "Reconstructed R14 manifest population differs")
    current_manifests = []
    for expected in r14_record["current_source_training_manifests"]:
        path = Path(expected["path"]).resolve()
        require(path.is_relative_to(root), "R14 training manifest is outside the repository")
        actual = dependency(path)
        require(actual == expected and read_json(path)["code_sha256"] == r14["current_source"]["sha256"],
                "Current R14 source does not match a recorded training hash")
        current_manifests.append(actual)
    require(len(current_manifests) == 4, "Current-source R14 manifest population differs")
    r14["current_source_training_manifests_verified"] = current_manifests
    r14["optimizer_definition_source_identity_verified"] = True
    training = read_json(output / "stageI_interaction/statistics/training_invariants.json")
    require(training["R15_entrypoint_recovery"]["recorded_entrypoint_identity_verified"] is True
            and training["R15_entrypoint_recovery"]["imported_module_historical_identity_verified"] is False,
            "Training audit does not distinguish entrypoint identity from imported history")
    require(training["R15_entrypoint_recovery"]["receipt_sha256"] == sha256(output / "RECOVERED_TRAINING_SOURCE.json"), "Training audit recovery receipt hash stale")
    require(training["R14_entrypoint_recovery"]["recorded_entrypoint_identity_verified"] is True
            and training["R14_entrypoint_recovery"]["optimizer_definition_source_identity_verified"] is True
            and training["R14_entrypoint_recovery"]["imported_module_historical_identity_verified"] is False,
            "Training audit R14 recovery evidence differs")
    snapshot = read_json(output / "source_snapshot/MANIFEST.json")
    require(snapshot["recovered_historical_entrypoint"] == r15["recovered_source"], "Snapshot R15 historical source differs")
    require(snapshot["recovered_historical_R14_entrypoint"] == r14["recovered_source"], "Snapshot R14 historical source differs")
    require(snapshot["recovery_receipt"] == dependency(output / "RECOVERED_TRAINING_SOURCE.json"), "Snapshot recovery receipt differs")
    archival = read_json(output / "ARCHIVAL_RECOVERY_AUDIT.json")
    require(archival["original_case_CSV"]["original_individual_ID_comparison_completed"] is False
            and archival["R14_optimizer_state"]["optimizer_or_param_groups_or_Adam_moment_fields_found"] is False,
            "Historical archival audit changed its then-current CSV or Adam state evidence")
    require(archival["R14_entrypoint"]["status"] == "recovered_by_manifest_hash_matched_reconstruction", "Archival audit R14 recovery status stale")
    runtime_path = output / "RUNTIME_SOURCE_CACHE_AUDIT.json"
    runtime = read_json(runtime_path)
    replay = audit_runtime_sources(root)
    require(runtime == replay, "R15 local runtime-source cache audit no longer reproduces")
    require(runtime["status"] == "supporting_cache_evidence_complete_not_process_trace"
            and runtime["passed"] is True and runtime["local_module_count"] == 18
            and runtime["all_timestamp_size_headers_match"] is True
            and runtime["all_caches_precede_training_start"] is True
            and runtime["all_cached_bytecode_matches_current_source"] is True
            and runtime["historical_imported_module_runtime_identity_fully_verified"] is False,
            "R15 local cache evidence is incomplete or overclaims process identity")
    r14_cache = runtime["R14_seed13_dependency_cache_evidence"]
    require(r14_cache["status"] == "partial_pre_run_cache_evidence"
            and r14_cache["local_module_count"] == 17
            and r14_cache["pre_run_cache_supported_count"] == 15
            and r14_cache["post_run_cache_only_count"] == 2
            and r14_cache["post_run_cache_only_modules"]
            == ["mmdd_stage1.models", "mmdd_stage1.training"]
            and r14_cache["all_cached_bytecode_matches_current_source"] is True
            and r14_cache["historical_imported_module_runtime_identity_fully_verified"] is False,
            "R14 seed13 cache evidence is incomplete or overclaims runtime identity")
    return {"R15": r15, "R14": r14, "archival_recovery_audit": dependency(output / "ARCHIVAL_RECOVERY_AUDIT.json"),
            "R15_local_import_cache_evidence": dependency(runtime_path),
            "R15_local_import_cache_modules": runtime["local_module_count"],
            "R15_process_bound_import_trace_verified": False,
            "R14_pre_run_cache_supported_modules": r14_cache["pre_run_cache_supported_count"],
            "R14_post_run_cache_only_modules": r14_cache["post_run_cache_only_modules"],
            "historical_CSV_absence_record": "Preserved as checked on 2026-09-09; current supplied-CSV identity is verified separately",
            "historical_imported_module_runtime_identity_verified": False,
            "historical_Adam_moment_tensors_recovered": False,
            "post_hoc_reconstruction_not_contemporaneous_archive": True,
            "full_historical_execution_verified": False}


def check_original_cases(root: Path, output: Path) -> dict[str, Any]:
    """Replay the bounded original-CSV comparison without changing its receipt."""
    from audit_stage1_r15_original_cases import audit as audit_original_cases

    path = output / "ORIGINAL_CASES_VALIDATION.json"
    saved = read_json(path)
    require(saved["status"] == "passed" and saved["identity_verified"] is True,
            "Supplied original-case identity did not pass")
    for key in ("original", "reconstructed_csv", "reconstructed_queue", "witness_funnel", "raw_B13_pool", "code"):
        expected = saved[key]
        source = Path(expected["path"]).resolve()
        require(source.is_relative_to(root) and source.name != ".env.openai", "Case audit dependency is outside authorized files")
        require(dependency(source) == expected, f"Original-case dependency changed: {key}")
    replay = json.loads(json.dumps(audit_original_cases(root, Path(saved["original"]["path"]))))
    require({key: value for key, value in replay.items() if key != "checked_at_utc"}
            == {key: value for key, value in saved.items() if key != "checked_at_utc"},
            "Read-only original-case semantic comparison differs from its saved receipt")
    require(replay["semantic_cells_checked"] == replay["semantic_cells_matched"] == 2460
            and replay["original_column_count"] == len(replay["fields"]) == 30
            and all(value == {"compared": 82, "mismatches": 0} for value in replay["fields"].values())
            and not replay["mismatches"], "Not all 82-by-30 original semantic cells match")
    require(replay["new_training_or_model_calls"] == 0 and replay["independent_value_or_join_verification"] is None,
            "CSV identity comparison overstates scientific mechanism validation")
    notebook = read_json(output / "original_cases_validation.ipynb")
    execution = notebook["metadata"]["execution"]
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(execution == {"status": "passed", "method": "sequential_plain_python", "code_cells_executed": 4, "kernel_used": False}
            and [cell["execution_count"] for cell in cells] == [1, 2, 3, 4]
            and all(cell["outputs"] and all(item["output_type"] != "error" for item in cell["outputs"]) for cell in cells),
            "Notebook execution record does not support the documented plain-Python execution scope")
    return {"status": "verified_original_semantic_identity", "checked_at_utc": saved["checked_at_utc"],
            "receipt": dependency(path), "original": saved["original"],
            "unique_pairs": replay["profile"]["unique_pairs"], "queries": replay["profile"]["queries"],
            "sources": replay["profile"]["sources"], "original_columns": 30,
            "matching_semantic_cells": 2460, "mismatches": 0,
            "comparison_replayed_read_only": True, "row_order_equal": replay["profile"]["row_order_equal"],
            "CSV_bytes_equal_to_reconstruction": saved["original"]["sha256"] == saved["reconstructed_csv"]["sha256"],
            "notebook_recorded_execution": execution,
            "independent_value_join_verification": None}


def check_populations(root: Path, output: Path) -> dict[str, Any]:
    base_rows = read_rows(output / "per_query_metrics.jsonl.gz")
    base = {key[0]: row for key, row in unique_rows(base_rows, ("query_id",)).items()}
    require(len(base) == 1198, "Primary query population is not 1198")
    sources, _ = _source_map(root)
    require(all(row["source_table_id"] == sources[query_id] for query_id, row in base.items()), "Source groups differ from dataset")
    require(len({row["source_table_id"] for row in base_rows}) == 1000, "Source group count differs")
    require(Counter(row["query_kind"] for row in base_rows) == {"implicit": 599, "explicit": 599}, "Query strata differ")
    interaction = read_json(output / "stageI_interaction/statistics/interaction_summary.json")
    for arm, relative in ARMS.items():
        original = {row["query_id"]: row for row in read_rows(root / "work" / relative / "evaluation_step178/rankings.jsonl.gz")}
        require(set(original) == set(base), f"Original endpoint query IDs differ: {arm}")
        for query_id, row in base.items():
            require(set(row["arms"]) == set(ARMS), "Six-arm endpoint set differs")
            require(set(row["positive_target_ids"]) == set(original[query_id]["positive_target_ids"]), "Fixed G_q differs")
            require(row["positive_denominator"] == len(set(row["positive_target_ids"])), "Positive denominator is not fixed G_q size")
            require(row["arms"][arm]["rankings"] == original[query_id]["rankings"], "Consolidated rankings differ from source")
            require(row["arms"][arm]["search_vectors"] == 43, "Natural retrieval query-vector count differs")
            for rule in RULES:
                for k in KS:
                    value = row["arms"][arm]["rankings"][rule][str(k)]
                    hits = set(value["target_ids"]) & set(row["positive_target_ids"])
                    require(len(set(value["target_ids"])) == k, "Ranking has duplicates or wrong K")
                    require(set(value["hit_ids"]) == hits and close(value["recall"], len(hits) / row["positive_denominator"]), "Recall/hit IDs disagree")
        for kind in ("all", "implicit", "explicit"):
            selected = [row for row in base_rows if kind == "all" or row["query_kind"] == kind]
            for rule in RULES:
                for k in KS:
                    actual = sum(row["arms"][arm]["rankings"][rule][str(k)]["recall"] for row in selected) / len(selected)
                    require(close(actual, interaction["arm_metrics"][arm][kind][rule][str(k)]), "Endpoint macro Recall disagrees")
    for name, coefficients in CONTRASTS.items():
        delta_rows = read_rows(output / "stageI_interaction/statistics" / f"{name}.jsonl.gz")
        require({row["query_id"] for row in delta_rows} == set(base) and len(delta_rows) == 1198, "Contrast query population differs")
        for kind in ("all", "implicit", "explicit"):
            selected = [row for row in delta_rows if kind == "all" or row["query_kind"] == kind]
            for rule in RULES:
                for k in KS:
                    values = []
                    for row in selected:
                        query = base[row["query_id"]]
                        expected = sum(weight * query["arms"][arm]["rankings"][rule][str(k)]["recall"] for arm, weight in coefficients.items())
                        require(close(expected, row["metrics"][rule][str(k)]["delta"]), "Query contrast algebra failed")
                        values.append(expected)
                    summary = interaction["contrasts"][name][kind][rule][str(k)]
                    require(close(sum(values) / len(values), summary["point_delta"]), "Contrast macro delta disagrees")
                    require([sum(v > 1e-12 for v in values), sum(v < -1e-12 for v in values), sum(abs(v) <= 1e-12 for v in values)] == [summary[key] for key in ("win_queries", "loss_queries", "tie_queries")], "Contrast win/loss/tie differs")
    return {"queries": 1198, "arms": 6, "source_groups": 1000, "strata": {"implicit": 599, "explicit": 599},
            "rules": list(RULES), "K": list(KS), "contrasts_algebra_and_point_estimates": len(CONTRASTS),
            "bootstrap_interval_recalculation": "Not rerun here; source-group method tested, saved joint-bootstrap outputs preserved and hashed"}


def check_provenance(output: Path) -> dict[str, Any]:
    candidates = read_rows(output / "candidate_provenance.jsonl.gz")
    candidate_map = unique_rows(candidates, ("arm", "query_id"))
    witnesses = read_rows(output / "witness_funnel.jsonl.gz")
    witness_map = unique_rows(witnesses, ("arm", "query_id", "target_id"))
    verification = read_rows(output / "evidence_only_verification.jsonl.gz")
    verification_map = unique_rows(verification, ("arm", "query_id", "target_id"))
    queue = read_rows(output / "stageC_candidate_delivery/B13_evidence_only_known_witness_cases.jsonl.gz")
    queue_set = set(unique_rows(queue, ("query_id", "target_id")))
    discoveries = read_json(output / "statistics/discovery_summary.json")["arms"]
    require(len(candidate_map) == 6 * 1198 and len(witness_map) == 6 * 1279, "Candidate/witness row count differs")
    require(Counter(row["arm"] for row in candidates) == {arm: 1198 for arm in ARMS}, "Candidate arm population differs")
    expected_pairs, expected_verification, reconstructed_priority = set(), set(), set()
    boundary_ties = 0
    for row in candidates:
        arm, query_id = row["arm"], row["query_id"]
        direct, exact, evidence, union, positives = (set(row[key]) for key in ("D100_ANN", "D100_exact", "E", "U", "positive_target_ids"))
        b13 = candidate_map[("s_full", query_id)]
        require(len(direct) == len(exact) == 100 and union == direct | evidence, "D/exact/E/U set invariant failed")
        require(row["positive_denominator"] == len(positives), "Candidate denominator differs")
        require({value["target_id"] for value in row["positive_targets"]} == positives, "Candidate positive records incomplete")
        for rule in RULES:
            require(len(set(row["C50"][rule])) == 50 and set(row["C50"][rule]) <= union, "C50 budget/ancestry failed")
            require(len(set(row["Top10_stage1"][rule])) == 10 and set(row["Top10_stage1"][rule]) <= set(row["C50"][rule]), "Stage1 top10 is not prefix subset")
        require(row["Top10_stage2"] is None, "Unexecuted Stage2 rank must be null")
        for positive in row["positive_targets"]:
            target = positive["target_id"]
            key = (arm, query_id, target)
            expected_pairs.add(key)
            checks = {"ann_evidence_only": target in evidence - direct,
                      "exact_evidence_only": target in evidence - exact,
                      "relative_B13_ANN_new": target in evidence - set(b13["D100_ANN"]),
                      "relative_B13_exact_new": target in evidence - set(b13["D100_exact"])}
            require(all(positive[name] == value for name, value in checks.items()), "Own/B13 discovery flag differs")
            low, high = positive["exact_direct_rank_min"], positive["exact_direct_rank_max"]
            require(isinstance(low, int) and isinstance(high, int) and 1 <= low <= high, "Positive exact rank missing/invalid")
            boundary_ties += int(low <= 100 < high)
            witness = witness_map[key]
            require(positive["retained_evidence_ids"] == witness["deployed_selected_evidence_ids"] and positive["delivery"] == witness["delivery"], "Provenance/retained witness delivery disagree")
            require(len(witness["deployed_selected_evidence_ids"]) <= 4, "Evidence retention budget exceeded")
            require(set(witness["deployed_known_evidence_ids"]) <= set(witness["deployed_selected_evidence_ids"]) & set(witness["known_evidence_ids"]), "Known retained witness set invalid")
            for name in ("raw_known_supported_rows", "deployed_known_supported_rows"):
                require(set(witness[name]) <= set(range(5)), "Known row support outside 0..4")
            require(witness["correct_join_verified"] is None and witness["independent_value_verification"] is None, "Unknown verification upgraded")
            if any(checks[name] for name in ("ann_evidence_only", "exact_evidence_only", "relative_B13_exact_new")):
                expected_verification.add(key)
                require(verification_map[key]["candidate_provenance"] == positive, "Verification chain provenance differs")
            if arm == "s_full" and checks["ann_evidence_only"] and positive["known_QET"]:
                reconstructed_priority.add((query_id, target))
    require(set(witness_map) == expected_pairs, "Witness positive population incomplete")
    require(set(verification_map) == expected_verification, "New-candidate verification population incomplete")
    flagged = {(row["query_id"], row["target_id"]) for row in verification if row["B13_priority_queue"]}
    require(flagged == queue_set == reconstructed_priority and len(flagged) == 82, "Fixed reconstructed82 priority differs")
    null_fields = ("stage4_independent_attribute_support", "stage5_correct_value_and_join", "stage6_stage2_final_top10", "independent_entity_verification", "independent_image_information", "without_evidence_counterfactual", "reviewed_wrong_evidence_counterfactual")
    require(all(row[key] is None for row in verification for key in null_fields), "Stage2/independent verification fields must stay null")
    for arm in ARMS:
        for kind in ("all", "implicit", "explicit"):
            selected = [row for row in candidates if row["arm"] == arm and (kind == "all" or row["query_kind"] == kind)]
            reference = discoveries[arm][kind]
            for key in ("ann_evidence_only", "exact_evidence_only", "relative_B13_ANN_new", "relative_B13_exact_new"):
                pair_count = sum(positive[key] for row in selected for positive in row["positive_targets"])
                macro = sum(sum(positive[key] for positive in row["positive_targets"]) / row["positive_denominator"] for row in selected) / len(selected)
                require(pair_count == reference[key]["positive_pairs"] and close(macro, reference[key]["query_macro_recall"]), "Discovery summary pair/macro values differ")
    return {"candidate_rows": len(candidates), "witness_rows": len(witnesses),
            "verification_arm_query_target_rows": len(verification),
            "verification_distinct_query_target_pairs": len({(row["query_id"], row["target_id"]) for row in verification}),
            "fixed_reconstructed_priority_pairs": len(flagged), "priority_queries": len({query for query, _ in flagged}),
            "rank_intervals_crossing100": boundary_ties, "known_candidate_chain_population_complete": True,
            "independent_value_join_fields_all_null": True,
            "priority_original_csv_identity": "Verified against the supplied original in the separate original_case_identity check; historical reconstruction is unchanged"}


def check_exact_geometry(output: Path) -> dict[str, Any]:
    exact = read_json(output / "exact_relation_panels.json")
    require(exact["all_panels_same_sources"] is True and set(exact["fixed_source_ids"]) == RELATIONS, "Fixed five-relation panel missing")
    for panel in exact["panels"].values():
        require(set(panel["relations"]) == RELATIONS, "One checkpoint lacks a relation panel")
        for relation, value in panel["relations"].items():
            require([row["source_id"] for row in value["per_source"]] == exact["fixed_source_ids"][relation], "Exact panel source/order differs")
    for arm in ("l_full", "n_full", "l_eoff", "n_eoff"):
        require([row["step"] for row in exact["full_dev_timelines"][arm]] == [0, 45, 89, 178], "Residual timeline incomplete")
    geometry = read_rows(output / "projection_and_optimizer_diagnostics.jsonl.gz")
    require(len(geometry) == 20, "Consolidated geometry row population differs")
    means = read_rows(output / "stageG_correctness/mean_vectors.jsonl.gz")
    mean_manifest = read_json(output / "stageG_correctness/mean_vectors_manifest.json")
    require(mean_manifest["status"] == "complete" and mean_manifest["rows"] == len(means) == 20,
            "G4 mean-vector export is incomplete")
    require(mean_manifest["output_sha256"] == sha256(output / "stageG_correctness/mean_vectors.jsonl.gz"),
            "G4 mean-vector manifest fingerprint differs")
    means_by_key = unique_rows(means, ("arm", "step", "intervention"))
    require(len(means_by_key) == 20, "G4 mean-vector keys are not unique")
    for row in geometry:
        key = (row["arm"], row["step"], row.get("intervention"))
        require(key in means_by_key, f"Missing G4 mean-vector row: {key}")
        saved = means_by_key[key]
        for kind, values in row["geometry"]["by_type"].items():
            original = saved["by_type"][kind]
            require(values["samples"] == original["samples"] == 1024
                    and values["sample_ids_sha256"] == original["sample_ids_sha256"],
                    f"G4 mean-vector panel differs: {key}/{kind}")
            for output_name in ("base_output", "full_output", "s0_output"):
                vector = values[output_name]["mean_vector"]
                require(vector == original[output_name]["mean_vector"] and len(vector) == 1024
                        and all(math.isfinite(value) for value in vector),
                        f"G4 mean vector differs: {key}/{kind}/{output_name}")
                norm = math.sqrt(math.fsum(value * value for value in vector))
                require(math.isclose(norm, values[output_name]["mean_vector_norm"],
                                     rel_tol=0, abs_tol=1e-5),
                        f"G4 mean-vector norm differs: {key}/{kind}/{output_name}")
    random = read_json(output / "stageG_correctness/random_geometry.json")
    require(random["status"] == "complete" and set(random["results"]) == set(ARMS), "Seeded random geometry incomplete")
    for kind, pairing in random["pairing"].items():
        flattened = [item for pair in pairing["paired_object_ids"] for item in pair]
        require(pairing["seed"] == 13 and len(flattened) == len(set(flattened)) == 1024, f"Random pairing invalid: {kind}")
    return {"five_relation_panels": len(exact["panels"]), "residual_timelines": 4,
            "checkpoint_steps": [0, 45, 89, 178], "geometry_rows": 20, "seeded_random_geometry_arms": 6,
            "actual_mean_vector_rows": 20, "mean_vectors_per_row": 9,
            "mean_vector_dimension": 1024,
            "random_scope": "Random pairs within fixed1024-object prefix panels, not a random full-lake sample"}


def check_report(root: Path, output: Path) -> dict[str, Any]:
    artifact = read_json(output / "artifact.json")
    receipt = read_json(output / "report.delivery.json")
    require(receipt["ok"] is True and receipt["stages"]["validation"] == receipt["stages"]["package"] == "passed", "Portable report package failed")
    require(receipt["stages"]["verification"] in {"passed", "structural_only"}, "Portable report verification failed")
    html = (output / "report.html").read_text(encoding="utf-8")
    match = re.search(r'<template id="data-analytics-portable-artifact-payload-source"[^>]*>(.*?)</template>', html, re.S)
    require(match is not None, "Portable HTML lacks embedded data payload")
    embedded = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    require(all(embedded[key] == artifact[key] for key in ("surface", "manifest", "snapshot", "sources")), "HTML embedded report differs from artifact.json")
    source_checks = {}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for relative, expected in value.get("sha256_by_file", {}).items():
                path = (root / relative).resolve()
                require(path.is_relative_to(root) and path.name != ".env.openai", "Report source is outside authorized files")
                actual = source_checks.setdefault(relative, sha256(path))
                require(actual == expected, f"Report source hash stale: {relative}")
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(artifact)
    results = (output / "RESULTS.md").read_text(encoding="utf-8").replace(str(root) + "/", "")
    normalized = re.sub(r"\A# [^\n]+\n", "", results).strip()
    sections = [value.strip() for value in re.split(r"(?m)(?=^## )", normalized) if value.strip()]
    blocks = artifact["manifest"]["blocks"]
    narrative = [block["body"] for block in blocks if block.get("id", "").startswith("narrative_")]
    require(narrative == sections, "Portable report lost or changed reviewed narrative sections")
    notes = read_json(output / "REPORT_SOURCE_NOTES.json")
    require(notes["results_sections_preserved"] == [section.splitlines()[0] for section in sections], "Source notes section map differs")
    manifest, data = artifact["manifest"], artifact["snapshot"]["datasets"]
    require(receipt["counts"]["blocks"] == len(blocks) and receipt["counts"]["charts"] == len(manifest["charts"]) == 2 and receipt["counts"]["tables"] == len(manifest["tables"]) == 2, "Report receipt block/chart/table counts differ")
    interaction = read_json(output / "stageI_interaction/statistics/interaction_summary.json")
    endpoint_rows = {row["condition"]: row for row in data["endpoint_rows"]}
    require(set(endpoint_rows) == set(ARM_LABELS.values()), "Report endpoint chart population differs")
    for arm, label in ARM_LABELS.items():
        value = endpoint_rows[label]
        source = interaction["arm_metrics"][arm]
        for field, kind, k in (("recall", "all", "10"), ("implicit_recall", "implicit", "10"), ("explicit_recall", "explicit", "10"), ("recall20", "all", "20"), ("candidate_recall50", "all", "50")):
            require(close(value[field], source[kind]["f1_union_direct"][k]), "Report endpoint dataset values differ")
    candidates = read_json(output / "stageC_candidate_delivery/SUMMARY.json")
    require(len(data["admission_rows"]) == 5, "Report admission dataset population differs")
    for row in data["admission_rows"]:
        rule = row["rule"]
        for field, kind in (("recall", "all"), ("implicit_recall", "implicit"), ("explicit_recall", "explicit")):
            expected = candidates["metrics"][kind]["rules"][rule]["CandidateRecall@50"] if rule in RULES else candidates["random_controls"][rule][kind]["CandidateRecall@50_mean"]
            require(close(row[field], expected), "Report admission dataset values differ")
    require(len(data["timeline_rows"]) == 16 and len(data["contrast_rows"]) == 6, "Report diagnostic table population differs")
    for row in data["contrast_rows"]:
        source = interaction["contrasts"][row["contrast"]]["all"]["f1_union_direct"]["10"]
        require(abs(float(row["delta_pp"]) - 100 * source["point_delta"]) <= .000051, "Report contrast rounded delta differs")
    if receipt["stages"]["verification"] == "structural_only":
        require(receipt["sourceDialog"] == receipt["sourceInteraction"] == "not_verified" and not receipt["viewports"], "Structural-only receipt overstates browser QA")
    reproduction = (output / "REPRODUCTION.md").read_text(encoding="utf-8")
    require("--task freeze" in reproduction and "--task analyze --device" in reproduction, "Candidate reproduction task arguments missing")
    return {"embedded_HTML_payload_equals_canonical_artifact": True, "source_files_hash_verified": len(source_checks),
            "source_sha256_by_file": source_checks, "reviewed_narrative_sections_preserved": len(sections),
            "native_charts": 2, "native_tables": 2, "blocks": len(blocks),
            "package_validation": "passed", "HTML_verification": receipt["stages"]["verification"],
            "browser_visual_and_interaction_QA_performed": receipt["stages"]["verification"] == "passed",
            "browser_limitation": receipt.get("browserWarning"), "source_interaction": receipt["sourceInteraction"]}


def refresh_snapshots(root: Path, output: Path) -> dict[str, Any]:
    path = output / "source_snapshot/MANIFEST.json"
    manifest = read_json(path)
    existing = {record["source"]: record for record in manifest["files"]}
    allowed = {"src/build_stage1_r15_report.py", "src/validate_stage1_r15_artifacts.py"}
    refreshed = []
    for relative in sorted(set(existing) | allowed):
        source = root / relative
        destination = output / "source_snapshot" / relative
        current = sha256(source)
        previous = existing.get(relative)
        if previous is not None:
            require(sha256(destination) == previous["sha256"], f"Existing snapshot does not match its manifest: {relative}")
            require(current == previous["sha256"] or relative in allowed, f"Unapproved snapshot drift: {relative}")
        if previous is None or previous["sha256"] != current:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            refreshed.append({"source": relative, "old_sha256": previous["sha256"] if previous else None,
                              "new_sha256": current, "reason": "Refresh final reviewed report or validation source" if previous else "Add final artifact validator to executable snapshot"})
            existing[relative] = {"source": relative, "snapshot": str(destination.relative_to(output)), "sha256": current, "bytes": destination.stat().st_size}
        require(sha256(destination) == current, f"Final source snapshot differs: {relative}")
    manifest["files"] = list(existing.values())
    if refreshed:
        manifest.setdefault("refresh_history", []).append({"at_utc": datetime.now(timezone.utc).isoformat(), "entries": refreshed,
                                                           "historical_identity_claim": False})
        write_json(path, manifest)
    return {"files_verified_against_current_sources": len(existing), "refreshed_entries": refreshed,
            "historical_source_identity": "Current-source copies do not establish historical identity; recovered training entrypoints are checked separately against manifest hashes"}


def validate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / "work/stage1_optimization_r15_20260909"
    checks = {"saved_computation_flags": check_saved_flags(output), "fixed_population_metrics": check_populations(root, output),
              "candidate_witness_chain": check_provenance(output), "exact_and_geometry": check_exact_geometry(output),
              "portable_report": check_report(root, output),
              "recovered_training_entrypoints": check_recovered_entrypoints(root, output),
              "original_case_identity": check_original_cases(root, output)}
    completion = read_json(output / "COMPLETION_AUDIT.json")
    require(completion.get("unverified_archival_requirements") and completion.get("irreversible_protocol_deviations"), "Historical and chronology gaps are not explicit")
    require(completion["scientific_mechanism_proven"] is False, "Unverified scientific mechanism upgraded")
    require(completion["original_case_identity"]["status"] == "verified_original_semantic_identity"
            and completion["required_external_inputs"]["original_82_case_CSV"] == checks["original_case_identity"]["original"],
            "Current completion audit original-CSV evidence differs")
    require(not any("CSV" in gap for gap in completion["unverified_archival_requirements"]),
            "Verified original CSV is still listed as a current archival gap")
    checks["current_source_snapshots"] = refresh_snapshots(root, output)
    additional = ("REPRODUCTION.md", "tests/verification.json", "statistics/cost_profiles.json", "artifact.json", "report.html", "report.delivery.json", "REPORT_SOURCE_NOTES.json", "stageG_correctness/random_geometry.json", "stageG_correctness/mean_vectors.jsonl.gz", "stageG_correctness/mean_vectors_manifest.json", "RUNTIME_SOURCE_CACHE_AUDIT.json", "source_snapshot/src/validate_stage1_r15_artifacts.py", "source_snapshot/src/build_stage1_r15_report.py", "source_snapshot/src/recover_stage1_r15_training_source.py", "source_snapshot/src/audit_stage1_r15_mean_vectors.py", "source_snapshot/src/audit_stage1_r15_runtime_sources.py", "source_snapshot/tests/test_stage1_r15_runtime_sources.py", "RECOVERED_TRAINING_SOURCE.json", "ARCHIVAL_RECOVERY_AUDIT.json", "source_snapshot/historical/run_stage1_r15.py", "source_snapshot/historical/run_stage1_r14.py", "ORIGINAL_CASES_VALIDATION.json", "source_snapshot/src/audit_stage1_r15_original_cases.py", "original_cases_validation.ipynb")
    names = sorted((set(completion["required_artifacts"]) | set(additional)) - {RECEIPT})
    artifacts = {name: dependency(output / name) for name in names}
    updated_hashes = [{"artifact": name, "old_sha256": completion["required_artifacts"].get(name, {}).get("sha256"), "new_sha256": record["sha256"]}
                      for name, record in artifacts.items() if completion["required_artifacts"].get(name, {}).get("sha256") != record["sha256"]]
    receipt = {
        "status": "passed_with_documented_archival_gaps", "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sharing_readiness": "share_with_caveats",
        "scope": "Final current-state artifact consistency and report packaging; no training, GPU, model or remote calls",
        "checks": checks, "validated_artifacts": artifacts, "completion_artifact_hash_refreshes": updated_hashes,
        "validated_external_inputs": completion["required_external_inputs"],
        "unverified_archival_requirements": completion["unverified_archival_requirements"],
        "irreversible_protocol_deviations": completion["irreversible_protocol_deviations"],
        "independent_value_join_mechanism_verified": False,
        "conditional_V_executed": False,
        "historical_requirements_waived": False, "original_plan_fully_verified": False,
        "strict_plan_fully_verified": False,
        "goal_completion_claim": False,
        "validator": dependency(Path(__file__)),
    }
    write_json(output / RECEIPT, receipt)
    completion["required_artifacts"] = {**artifacts, RECEIPT: dependency(output / RECEIPT)}
    completion["source_files"] = checks["current_source_snapshots"]["files_verified_against_current_sources"]
    completion["final_delivery_validation"] = {"status": receipt["status"], "receipt": RECEIPT,
                                               "original_plan_fully_verified": False,
                                               "browser_visual_and_interaction_QA_performed": checks["portable_report"]["browser_visual_and_interaction_QA_performed"]}
    write_json(output / "COMPLETION_AUDIT.json", completion)
    for name, record in read_json(output / "COMPLETION_AUDIT.json")["required_artifacts"].items():
        require(sha256(output / name) == record["sha256"] and (output / name).stat().st_size == record["bytes"], f"Final required-artifact fingerprint mismatch: {name}")
    print(json.dumps({"status": receipt["status"], "receipt": str(output / RECEIPT), "source_snapshot_refreshes": checks["current_source_snapshots"]["refreshed_entries"], "required_artifacts": len(completion["required_artifacts"]), "original_plan_fully_verified": False}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    validate(parser.parse_args().root)

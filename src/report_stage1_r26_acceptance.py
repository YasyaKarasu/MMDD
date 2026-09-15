"""Map R26 acceptance requirements to actual production tests and run evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_stage1_r26 import OUT, ROOT, file_record
from run_stage1_r25 import _json


# Prefixes resolve actual parameterized JUnit cases, never inferred test passes.
CHECKS = {
    "M01": ("mmdd_stage1/r26_metrics.py:query_metrics", ["test_recall_differs_from_hit"], [], "Two positives, one hit: Recall=.5 while Hit=1."),
    "M02": ("mmdd_stage1/r26_metrics.py:population_metrics", ["test_recall_differs_from_hit", "test_duplicates_short_lists", "test_stage1_reporting"], ["statistics/stage2/RESULTS.json"], "Failed query stays in frozen denominator; duplicate targets do not add hits."),
    "G01": ("mmdd_stage1/retrieval.py:build_indices", ["test_rebuilt_production_student_index"], [], "Actual index build/query changes expected neighbor after parameter perturbation."),
    "G02": ("evaluate_stage1_r26.py:evaluate", [], ["acceptance/raw_rankings/AUDIT.json", "acceptance/numerical_replay/AUDIT.json", "acceptance/RANK_COVERAGE_RECONCILIATION.json"], "All canonical raw pool/signature invariants; full-lake exact numerical replay on fixed three-query panel per model; later aliases reconciled without relabeling independent observations."),
    "G03": ("mmdd_stage1/r26_metrics.py:fuse_channels", ["test_real_evidence_changes_all_fusions", "test_teacher_then_fusion_changes_e_order"], ["acceptance/raw_rankings/AUDIT.json"], "Actual E changes fusion; Column keeps every union candidate."),
    "G04": ("evaluate_stage1_r26.py:retain_evidence", ["test_real_evidence_changes_all_fusions"], ["acceptance/raw_rankings/AUDIT.json", "acceptance/numerical_replay/AUDIT.json"], "QT/Equal/Conf/natural-LSE bound to saved scores; actual D1 replay matches selected E/coverage on fixed panel; missing E raises."),
    "T01": ("train_stage1_r26.py:train", [], ["acceptance/C1_TENSOR_AUDIT.json", "recovered/B13/RECOVERY_AUDIT.json"], "Actual parent/PCA/identity tensors, anchors, five-relation coverage and checkpoint hashes verified."),
    "T02": ("train_stage1_r26.py:train", [], ["EXECUTION_MATRIX.json"], "Execution audit replays all14 histories:178 updates,11390 actual query IDs in shared frozen order."),
    "T03": ("train_stage1_r26.py:path_terms", ["test_order_only_production_factory_matches_old_loss_grad_and_adamw"], [], "Native and SUP production factories preserve old same-batch loss, every gradient and fresh AdamW update."),
    "T04": ("mmdd_stage1/r26_training.py:edge_query_loss", ["test_query_bound_graph", "test_edge_loss_preserves", "test_path_reassignment"], [], "Common per-query graph supplies QE/ET lists; fixed per-query denominators; positive target does not label every edge."),
    "T05": ("mmdd_stage1/r26_extension_objectives.py:objective", ["test_order_only_production_factory_matches_old_loss_grad_and_adamw[O-EXT"], [], "R26 extension with supplied Teacher/aux inputs but KD/U=0 preserves SUP loss, all gradients and fresh AdamW update."),
    "T06": ("mmdd_stage1/r26_extension_objectives.py:objective", ["test_T07_qt_teacher_requires_same_values", "test_extension_lse_weights_each_query"], ["acceptance/QT_CACHE_AUDIT.json"], "D/E consume same QT Teacher;64 actual cached pairs numerically replayed with tolerance. Native cache remains separate."),
    "T07": ("mmdd_stage1/r26_extension_objectives.py:objective", ["test_extension_lse_weights_each_query", "test_extension_multiple_positives", "test_extension_no_negative"], [], "All-active/E-inactive/mixed: fused SUP and KD weighted per-query before reduction, with matching gradients."),
    "T08": ("mmdd_stage1/r26_training.py:graph_edges", ["test_query_bound_graph", "test_feedback_augmentation"], ["acceptance/C1_TENSOR_AUDIT.json", "EXECUTION_MATRIX.json"], "Full C1 and14 C2 known-positive closure receipts checked; graph fixture rejects target-implies-edge labels; feedback augmentation preserves closure."),
    "T09": ("mmdd_stage1/r26_extension_objectives.py:query_uniform", ["test_extension_uniform", "test_order_only_production_factory_matches_old_loss_grad_and_adamw[O-EXT-SUP-ZERO]"], [], "Non-TT Uniform has nonzero gradient; TT has zero; all fixed aux lists stay in each query denominator; off recovers SUP."),
    "S01": ("mmdd_stage2/r26_generation.py:R26QwenBackend", ["test_actual_r26_generation_retries_once", "test_abstention_is_distinct"], ["stage2/generation_smoke/RESULT.json", "stage2/engineering/SUMMARY.json", "statistics/stage2/RESULTS.json"], "Actual256/512 retry behavior and backend text/image/abstain probes; one failed Raw condition retained."),
    "S02": ("mmdd_stage2/r26_generation.py:parse_value_completion", ["test_final_value_is_selected", "test_generation_rejects_wrong_schema", "test_abstention_is_distinct"], [], "Only valid final value accepted; malformed/ambiguous/truncated objects are not abstentions."),
    "S03": ("prepare_stage2_r26.py:prepare", ["test_stage2_preserves_unattempted_targets"], ["stage2/inputs/Qwen-Raw/INPUT_RECEIPT.json", "stage2/inputs/B13/INPUT_RECEIPT.json", "statistics/stage2/RESULTS.json"], "User override: own EqualC18 including T0 prequeue, all18 retained; final R1/3/5/7/9. Contract C50 superseded."),
    "S04": ("run_stage2_r26.py:run", ["test_actual_generation_inputs", "test_actual_r26_driver_resumes"], ["statistics/stage2/RESULTS.json"], "Same64 queries/candidates/column/row opportunities; crop, crop+original and NoE inputs checked in actual driver."),
    "S05": ("mmdd_stage2/r26_generation.py:R26QwenBackend", ["test_actual_generation_inputs"], [], "Actual inherited reader serialization omits query truth metadata; generation also excludes target values."),
    "S06": ("run_stage2_r26.py:run", ["test_actual_r26_driver_resumes"], [], "B13 complete still executes Raw; later B13 resume skips; changed input hash rejected before skipping."),
    "F01": ("mine_stage1_r26_feedback.py:run; prepare_stage1_r26_refinement.py:prepare", ["test_feedback_augmentation", "test_actual_refinement_preparation_roundtrips"], ["feedback/REFINEMENT_GATE_FROZEN.json", "acceptance/FEEDBACK_AUDIT.json"], "All train-fit old/new natural universes and actual T0 H membership comparisons; production six-job list write/load roundtrip."),
    "F02": ("mmdd_stage1/r26_feedback.py:feedback_gate", ["test_feedback_gate_respects_priority", "test_extension_gate_requires_both"], ["feedback/REFINEMENT_GATE_FROZEN.json", "acceptance/FEEDBACK_AUDIT.json"], "Both-seed health, implicit G intersect(E-D) and actual hard-membership difference; missing higher priority remains unassessable."),
    "A01": ("train_stage1_r26.py:train; train_stage1_r26_extension.py:train", ["test_actual_r26_driver_resumes", "test_teacher_cache_reuses_pairs", "test_actual_student_resume_rejects_changed_identity_inputs"], ["EXECUTION_MATRIX.json"], "Actual core/extension Student completed/incomplete resume rejects changed parent/graph/order/protocol/registry/features/Teacher/code inputs; actual Stage2/cache invalidation also tested."),
    "A02": ("audit_stage1_r26_execution.py:run", [], ["EXECUTION_MATRIX.json"], "Every scientific module has actual valid evidence or an independently reproduced negative budget gate. Archive delivery is verified separately by the package receipt."),
}


def run() -> dict:
    junit_path = OUT / "acceptance/pytest_r26_archive_backup_106.xml"
    cases = list(ET.parse(junit_path).iter("testcase"))
    if any(any(c.tag in ("error","failure","skipped") for c in case) for case in cases):
        raise ValueError("Production acceptance test suite is not all passed")
    tests = [file_record(ROOT / "tests" / name) for name in (
        "test_stage1_r26.py","test_stage2_r26.py","test_stage1_r25.py","test_stage1_retrieval_aligned.py",
        "test_package_stage1_r26.py","test_audit_stage1_r26_refinement.py","test_audit_stage1_r26_refinement_evaluation.py")]
    rows = []
    matrix = json.loads((OUT / "EXECUTION_MATRIX.json").read_text())
    for test_id,(entrypoint,prefixes,paths,assertion) in CHECKS.items():
        matched = [case for case in cases if any(case.attrib["name"].startswith(prefix) for prefix in prefixes)]
        if prefixes and any(not any(case.attrib["name"].startswith(prefix) for case in cases) for prefix in prefixes):
            raise ValueError(f"No actual JUnit case for {test_id}")
        records = [file_record(OUT / path) for path in paths]
        pending = [r["path"] for r in records if not r["exists"]]
        partial = bool(pending) or (test_id == "A02" and not matrix.get("scientific_modules_complete",False))
        row = {"test_id":test_id,"production_entrypoint":entrypoint,
               "inputs_sha":{"production_sources":[file_record(ROOT / "src" / part.strip().split(":")[0]) for part in entrypoint.split(";")],"test_sources":tests},
               "assertion":assertion,"observed_output":{"passed_cases":[case.attrib["name"] for case in matched],
               "actual_artifacts":[r for r in records if r["exists"]],"pending_artifacts":pending},
               "evidence_path":{"junit":file_record(junit_path),"artifacts":records},
               "execution_status":"in_progress" if partial else "ran", "scientific_validity":"partial" if partial else "valid_for_stated_scope"}
        rows.append(row)
    complete = all(r["execution_status"] == "ran" for r in rows)
    report = {"execution_status":"ran" if complete else "in_progress","scientific_validity":"valid_for_stated_scopes" if complete else "partial",
              "generated_at":datetime.now(timezone.utc).isoformat(),"actual_test_cases":len(cases),"requirements":rows,
              "scope":"Production behavior tests plus actual scoped audit artifacts. Existence is provenance, not an independent rerun of an artifact's assertions; inspect its recorded audit. Final scientific completion is not inferred from test passes.",
              "user_overrides":{"Stage1_K":[10,20,30,40,50],"Stage2_K":[1,3,5,7,9],"Stage2_budget":18}}
    _json(OUT / "acceptance/ACCEPTANCE_LEDGER.json",report)
    lines = ["# R26 actual acceptance ledger","",f"Actual JUnit suite: {len(cases)} passed cases. Scientific acceptance: {'complete' if complete else 'in progress'}; archive delivery has its own verification receipt.","",
             "Stage1 R10/20/30/40/50; Stage2 R1/3/5/7/9, C18 including T0. JSON records source hashes, exact tests, artifact hashes and scoped assertions.","",
             "| ID | Execution | Scientific validity | Assertion |","|---|---|---|---|"]
    lines += [f"| {r['test_id']} | {r['execution_status']} | {r['scientific_validity']} | {r['assertion']} |" for r in rows]
    (OUT / "acceptance/ACCEPTANCE_LEDGER.md").write_text("\n".join(lines)+"\n")
    return {"requirements":len(rows),"test_cases":len(cases),"in_progress":[r["test_id"] for r in rows if r["execution_status"] == "in_progress"]}


if __name__ == "__main__":
    print(json.dumps(run()))

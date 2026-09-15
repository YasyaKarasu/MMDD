"""Refresh execution coverage using hashed artifacts and actual training histories."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from prepare_stage1_r26 import OUT, file_record
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json, sha256


def run() -> dict:
    verified = {}

    def check(record: dict) -> None:
        path = Path(record["path"])
        if str(path) not in verified:
            verified[str(path)] = sha256(path)
        if verified[str(path)] != record["sha256"]:
            raise ValueError(f"Artifact identity differs: {path}")

    inventory = json.loads((OUT / "MODEL_INVENTORY.json").read_text())
    coverage = []
    for spec in inventory:
        name = canonical = spec["generator_id"]
        alias = OUT / "rankings" / name / "ALIAS_RECEIPT.json"
        if alias.exists():
            data = json.loads(alias.read_text())
            check(data["checkpoint"])
            check(data["canonical_receipt"])
            canonical = data["canonical_generator"]
        row = {"generator":name,"canonical":canonical,"modules":{}}
        for module,receipt_name,artifact in (("rankings","RETRIEVAL_RECEIPT.json","rankings"),
                ("fusion","COLUMN_RECEIPT.json","rankings"),("teacher","TEACHER_RECEIPT.json","rankings"),
                ("statistics/train_hubs","HUB_RECEIPT.json","neighbors"),
                ("statistics/student_latency/cpu","LATENCY_RECEIPT.json","queries"),
                ("statistics/student_latency/cuda_0","LATENCY_RECEIPT.json","queries"),
                ("statistics/teacher_latency/cuda_1","LATENCY_RECEIPT.json","queries")):
            path = OUT / module / canonical / receipt_name
            if not path.exists():
                row["modules"][module] = {"execution_status":"pending","scientific_validity":"unassessable"}
                continue
            receipt = json.loads(path.read_text())
            check(receipt[artifact])
            if module == "rankings":
                if spec["checkpoint"] and canonical == name:
                    check({"path":spec["checkpoint"],"sha256":receipt["signature"]["checkpoint_sha256"]})
                if receipt["metrics"]["overall"]["queries"] != 1198:
                    raise ValueError("Whole-lake population changed")
            row["modules"][module] = {"execution_status":"verified_parameter_identical_alias" if canonical != name else "ran",
                                      "scientific_validity":"valid","receipt":file_record(path)}
        coverage.append(row)
        print(json.dumps({"execution_audit_generator":name}),flush=True)
    training = []
    common_order = [r["query_id"] for r in read_rows(OUT / "common/c2_order.jsonl")]
    for arm in ("O-NATIVE","O-SUP","E-GRAPH","O-QTKD","O-U","O-UQTKD","O-LSE-QTKD"):
        for seed in (13,29):
            path = OUT / "training" / arm / f"seed{seed}" / "C2_COMPLETION_RECEIPT.json"
            if not path.exists():
                training.append({"arm":arm,"seed":seed,"execution_status":"pending"})
                continue
            data = json.loads(path.read_text())
            for key in ("history","graph_edges"):
                check(data[key])
            for checkpoint in data["checkpoints"]:
                check(checkpoint)
            history = list(read_rows(Path(data["history"]["path"])))
            if len(history)!=178 or [q for r in history for q in r["query_ids"]] != common_order:
                raise ValueError("C2 actual consumed order/budget differs")
            if data["coverage_lists"]!=11390 or data["optimizer_initial_state"] != "fresh":
                raise ValueError("C2 coverage/optimizer differs")
            closure_path = path.parent / "TARGET_CLOSURE.json"
            closure = json.loads(closure_path.read_text())
            if closure["checked_queries"] != 11390 or closure["violations"]:
                raise ValueError("C2 actual full-registry closure audit differs")
            training.append({"arm":arm,"seed":seed,"execution_status":"ran","scientific_validity":"valid",
                             "actual_steps":len(history),"actual_query_lists":len(common_order),
                             "target_closure":file_record(closure_path),"receipt":file_record(path)})
    all_models = lambda module: all(r["modules"][module]["scientific_validity"] == "valid" for r in coverage)
    modules = []

    def append(name: str, complete: bool, evidence: list[str], limitation: str = "") -> None:
        modules.append({"id":name,"execution_status":"ran" if complete else "in_progress",
                        "scientific_validity":"valid" if complete else "partial",
                        "evidence":[file_record(OUT / p) for p in evidence],"limitation":limitation})

    c1 = json.loads((OUT / "acceptance/C1_TENSOR_AUDIT.json").read_text())
    cases = list(ET.parse(OUT / "acceptance/pytest_r26_archive_backup_106.xml").iter("testcase"))
    if len(cases) < 105 or any(any(c.tag in ("failure","error","skipped") for c in case) for case in cases):
        raise ValueError("Final production and evidence-tool suite did not pass completely")
    raw = json.loads((OUT / "acceptance/raw_rankings/AUDIT.json").read_text())
    numerical = json.loads((OUT / "acceptance/numerical_replay/AUDIT.json").read_text())
    a0_valid = (c1["c1_reuse_valid"] and raw["execution_status"] == "ran" and numerical["execution_status"] == "ran"
                and raw["scientific_validity"] == "valid_for_available_canonical_models"
                and numerical["scientific_validity"] == "valid_for_fixed_panel")
    append("A0",a0_valid,["acceptance/C1_TENSOR_AUDIT.json","acceptance/pytest_r26_archive_backup_106.log",
                      "acceptance/pytest_r26_archive_backup_106.xml",
                      "acceptance/raw_rankings/AUDIT.json",
                      "acceptance/numerical_replay/AUDIT.json"],
           f"Actual{len(cases)}-case production/evidence suite, C1 tensors and full canonical raw audit plus fixed3-query numerical panel. Alias coverage reconciled below; no claim of all-query numerical inference replay.")
    initial = [r for r in coverage if not any(r["generator"].startswith("R26-"+a+"/") for a in ("O-QTKD","O-U","O-UQTKD","O-LSE-QTKD"))]
    append("E25",c1["c1_reuse_valid"] and all(r["modules"]["rankings"]["scientific_validity"] == "valid" for r in initial),["EVALUATION_RECONCILIATION.json"])
    append("TRAJ",all_models("rankings") and all_models("statistics/train_hubs"),["statistics/GEOMETRY_AUDIT.json","statistics/TRAIN_HUB_AUDIT.json"],
           "Fixed0/89/178 endpoints; no best checkpoint selection")
    for arm in ("O-NATIVE","O-SUP","E-GRAPH"):
        append(arm,all(r.get("scientific_validity") == "valid" for r in training if r["arm"] == arm),
               [f"training/{arm}/seed{s}/C2_COMPLETION_RECEIPT.json" for s in (13,29)],"Training validated; paired comparisons tracked under STATISTICS")
    append("C2_EXPANSION",all(r.get("scientific_validity") == "valid" for r in training),["acceptance/LOSS_EXTENSION_GATE_FROZEN.json","EXTENSION_TRAINING_RECEIPT.json"])
    append("F",all_models("fusion"),["fusion/columns/COLUMN_RECEIPT.json","statistics/postprocessing/POSTPROCESSING_RECEIPT.json"])
    append("R",all_models("teacher"),["TEACHER_BACKFILL_EVALUATION_RECEIPT.json","EXTENSION_FOLLOWUPS_RECEIPT.json"])
    stage2_path = OUT / "statistics/stage2/RESULTS.json"
    stage2 = json.loads(stage2_path.read_text()) if stage2_path.exists() else None
    append("S2",stage2 is not None and stage2["queries_per_generator"] == 64,["statistics/stage2/RESULTS.json"],
           "Raw Real-crop:1 exhausted retry, empty rank in64 denominator; B13:0 failed. Grounding unknown; deletion gives no Recall gain.")
    gate_path = OUT / "feedback/REFINEMENT_GATE_FROZEN.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"status":"unassessable"}
    feedback_audit_path = OUT / "acceptance/FEEDBACK_AUDIT.json"
    feedback_audit = json.loads(feedback_audit_path.read_text()) if feedback_audit_path.exists() else None
    append("FB-DIAG",gate["status"] in ("triggered","not_triggered") and feedback_audit is not None
           and feedback_audit["scientific_validity"] == "valid",
           ["feedback/REFINEMENT_GATE_CURRENT.json","feedback/REFINEMENT_GATE_FROZEN.json","acceptance/FEEDBACK_AUDIT.json"])
    refinement_evidence = ["feedback/REFINEMENT_EVALUATION.json",
                           "acceptance/REFINEMENT_TRAINING_AUDIT.json",
                           "acceptance/REFINEMENT_EVALUATION_AUDIT.json"]
    refinement_complete = False
    if all((OUT / name).exists() for name in refinement_evidence):
        evaluation, train_audit, eval_audit = [json.loads((OUT / name).read_text()) for name in refinement_evidence]
        for audit in (train_audit,eval_audit):
            for path,sha in audit["verified_artifact_hashes"].items():
                check({"path":path,"sha256":sha})
            check(audit["code"])
        check(eval_audit["evaluation"])
        refinement_complete = (all(r["scientific_validity"] == "valid" for r in (evaluation,train_audit,eval_audit))
                               and len(train_audit["jobs"]) == 6 and not train_audit["pending"]
                               and eval_audit["actual_rows"] == 6*2*1198)
    append("TEACHER_REFINEMENT",refinement_complete,refinement_evidence,
           "Actual full-budget checkpoint/optimizer/history audit plus all-query fixed-pool metric and measured-gate reconstruction required")
    redistill_path = OUT / "feedback/REDISTILL_GATE_FROZEN.json"
    redistill = json.loads(redistill_path.read_text()) if redistill_path.exists() else {"status":"unassessable"}
    skipped_redistill = refinement_complete and redistill["status"] == "not_triggered"
    if refinement_complete and redistill["status"] != eval_audit["redistill_gate"]:
        raise ValueError("Redistill status differs from independently reproduced gate")
    append("REDISTILLATION",skipped_redistill,
           ["feedback/REDISTILL_GATE_FROZEN.json","acceptance/REFINEMENT_EVALUATION_AUDIT.json"] if skipped_redistill
           else ["feedback/REDISTILLATION_COMPLETION_RECEIPT.json"],
           "Measured two-seed gate did not trigger" if skipped_redistill else "A triggered gate requires actual redistillation and its endpoint audit; missing gate is unassessable")
    if skipped_redistill:
        modules[-1]["execution_status"] = "not_triggered"
    findings_path = OUT / "statistics/STAGE1_FINDINGS.json"
    statistics_complete = False
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())
        for record in findings["inputs"].values():
            check(record)
        check(findings["report"])
        check(findings["code"])
        requested = json.loads((OUT / "statistics/stage1_recall/REPORT_RECEIPT.json").read_text())
        post = json.loads((OUT / "statistics/postprocessing/POSTPROCESSING_RECEIPT.json").read_text())
        statistics_complete = not requested["missing"] and requested["source_results"] == 3*len(inventory) and not post["missing"]
    append("STATISTICS",statistics_complete,["statistics/STAGE1_ANALYSIS_RECEIPT.json","statistics/stage1_recall/REPORT_RECEIPT.json","statistics/postprocessing/POSTPROCESSING_RECEIPT.json","statistics/STAGE1_FINDINGS.json"],
           "Complete current-inventory Stage1 and Stage2 statistics; conditional Teacher/redistillation statistics are audited in their separate modules")
    append("COST",all(all_models(m) for m in ("statistics/student_latency/cpu","statistics/student_latency/cuda_0","statistics/teacher_latency/cuda_1")),["statistics/COST_BENCHMARK_RECEIPT.json"],
           "Local-feature Student/T0 and value generation measured separately; backbone extraction excluded")
    result = {"execution_status":"in_progress","scientific_validity":"partial","last_verified_at":datetime.now(timezone.utc).isoformat(),
              "inventory_models":len(inventory),"modules":modules,"model_coverage":coverage,"actual_C2_training":training,
              "hashed_artifacts":len(verified),"completion_policy":"This coverage audit does not finalize the goal. Final scientific narrative/conditional gates/report package require separate review."}
    raw_audit_path = OUT / "acceptance/raw_rankings/AUDIT.json"
    if raw_audit_path.exists():
        raw_audit = json.loads(raw_audit_path.read_text())
        audited = {r["generator"]:r for r in raw_audit["models"]}
        reconciled = []
        for row in coverage:
            if row["canonical"] not in audited:
                raise ValueError("Inventory entry has no canonical full-population rank audit")
            actual = audited[row["canonical"]]
            check(actual["rankings"])
            if actual["observed"]["queries"] != 1198:
                raise ValueError("Canonical raw rank audit did not cover all queries")
            reconciled.append({"generator":row["generator"],"canonical":row["canonical"],
                               "alias":row["generator"] != row["canonical"],
                               "raw_audit":file_record(OUT / "acceptance/raw_rankings" / row["canonical"] / "AUDIT.json")})
        reconciliation = {"execution_status":"ran","scientific_validity":"valid","inventory_entries":len(reconciled),
                          "canonical_models":len(audited),"observations":reconciled,
                          "original_audit":file_record(raw_audit_path),"pending_when_original_audit_scanned":raw_audit["pending"],
                          "currently_pending":[],"scope":"Original56 canonical audits remain immutable. Entries initially pending were subsequently registered as verified parameter-identical aliases; current checkpoint/receipt/rank identities checked by execution audit."}
        _json(OUT / "acceptance/RANK_COVERAGE_RECONCILIATION.json",reconciliation)
        result["raw_rank_coverage"] = file_record(OUT / "acceptance/RANK_COVERAGE_RECONCILIATION.json")
    # The ledger hashes this matrix. Keep its reverse navigation link unhashed
    # so final report provenance has no impossible circular content hash.
    result["acceptance_ledger_path"] = str(OUT / "acceptance/ACCEPTANCE_LEDGER.json")
    all_scientific_modules = all(r["execution_status"] in ("ran","not_triggered") and r["scientific_validity"] == "valid" for r in modules)
    if all_scientific_modules:
        result["execution_status"] = "scientific_work_complete_pending_delivery"
        result["scientific_validity"] = "valid_for_declared_scopes"
    result["scientific_modules_complete"] = all_scientific_modules
    result["actual_test_cases"] = len(cases)
    result["hashed_artifacts"] = len(verified)
    _json(OUT / "EXECUTION_MATRIX.json",result)
    return {"models":len(inventory),"hashed_artifacts":len(verified),"modules":{r["id"]:r["execution_status"] for r in modules}}


if __name__ == "__main__":
    print(json.dumps(run()))

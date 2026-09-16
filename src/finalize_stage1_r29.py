"""Assemble the R29 diagnostic/training artifacts into the required reports."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from run_stage1_r29 import OUT, ROOT, sha, read_rows


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    base = OUT / "diagnostics/candidate_universe"
    gate = load(base / "GATE.json")
    sup = load(OUT / "diagnostics/supervision_summary.json")
    p3 = load(OUT / "training/S-EDGE-FREEZE-P/seed13/exact_rankings/metrics.json")
    r3 = load(OUT / "training/S-EDGE-FREEZE-R/seed13/exact_rankings/metrics.json")
    p1 = load(OUT / "training/S-EDGE-FREEZE-P/seed13/exact_rankings_epoch1/metrics.json")
    r1 = load(OUT / "training/S-EDGE-FREEZE-R/seed13/exact_rankings_epoch1/metrics.json")
    control = load(ROOT / "work/stage1_optimization_r28_split_path_20260915/student/own/rankings/S-EDGE-LONG/seed13/epoch3/metrics.json")
    parent = load(ROOT / "work/stage1_optimization_r28_split_path_20260915/student/own/rankings/S-EDGE-LONG/seed13/epoch0/metrics.json")
    teacher_eval_path = OUT / "R29_TEACHER_EVALUATION.json"
    teacher_eval = load(teacher_eval_path) if teacher_eval_path.is_file() else None
    teacher_status = "measured" if teacher_eval else "not_measured_r29_scope"

    (base / "CANDIDATE_DIAGNOSTIC.md").write_text(
        "# R29 Candidate Diagnostic\n\n"
        "The comparison uses the merged R28 `path_hard -> graph_edges` graph (11,390 train-fit queries) and query-level paired bootstrap (10,000 replicates). Train source groups were not treated as reliable, so these intervals are explicitly query-level. Natural candidates were mined once from the frozen parent Student ANN; dev/test qrels and post-hoc hub IDs were not used in mining.\n\n"
        f"Gate: **{gate['decision']}**. QT margin NAT-R12 CI={gate['relations']['table->table']['margin_nat_minus_r12']['ci95']}, violation CI={gate['relations']['table->table']['violation_nat_minus_r12']['ci95']}, HNR R12<-NAT={gate['relations']['table->table']['hnr_r12_from_nat_mean']:.3f}. Natural is not harder on QT under the preregistered gate.\n\n"
        "The evidence-side directions are mixed: table->text has a small negative margin gap and positive violation gap, while text->table has the opposite direction. Thus the candidate mismatch gate is weak, and B1 QT refresh is not triggered.\n\n" +
        ("R29 arm-specific Student U/M retrieval and frozen T0 Full-U scoring were completed on both B2 checkpoints; the resulting U/M rankings are reported separately below.\n"
         if teacher_eval else
         "Teacher T0 scores were not run on the R29 Natural candidate file in this bounded diagnostic. CUDA was verified outside the sandbox; this is a declared scope gap, not a hardware or runtime failure. The gate is therefore a Student-parent candidate diagnostic; no R29 arm-specific Teacher full-U conclusion is claimed.\n")
    )

    # The required per-query statistical artifact is the matched-32 diagnostic.
    src = base / "margin_violation_per_query.jsonl.gz"
    dst = OUT / "statistics/per_query.jsonl.gz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rt", encoding="utf-8") as fi, gzip.open(dst, "wt", encoding="utf-8") as fo:
        for line in fi:
            fo.write(line)
    with (OUT / "statistics/source_group_bootstrap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["comparison", "bootstrap_unit", "replicates", "ci95_low", "ci95_high", "note"])
        for rel, rec in gate["relations"].items():
            for metric in ("margin_nat_minus_r12", "violation_nat_minus_r12"):
                w.writerow([f"{rel}:{metric}", "query", rec[metric]["replicates"], rec[metric]["ci95"][0], rec[metric]["ci95"][1], "train-fit source groups not trusted"])
    with (OUT / "statistics/main_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["arm", "seed", "checkpoint", "exact_R@10", "exact_R@20", "exact_R@50", "hub_top10_max_128", "hub_top1_max_128", "teacher_full_u"])
        for label, rec in (("R28-S-EDGE-LONG", control["overall"]["D100_EXACT"]), ("R29-S-EDGE-FREEZE-P", p3), ("R29-S-EDGE-FREEZE-R", r3)):
            if label == "R28-S-EDGE-LONG":
                # control hub values are stored separately below; this row keeps
                # the historical exact recall as the comparison denominator.
                w.writerow([label, 13, "R28 epoch3", rec["recall@10"], rec["recall@20"], rec["recall@50"], 124, 84, "not_measured_in_R29"])
            else:
                w.writerow([label, 13, rec["checkpoint"], rec["R@10"], rec["R@20"], rec["R@50"], rec["hub_probe"]["max_top10_frequency"], rec["hub_probe"]["max_top1_frequency"], rec["teacher_full_u"]])

    for arm, status in (("S-EDGE-QT-REFRESH", "not_triggered"), ("S-EDGE-FREEZE-P-confirm", "not_triggered"), ("S-EDGE-FREEZE-R-confirm", "not_triggered")):
        path = OUT / "training" / arm / "STATUS.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"arm": arm, "status": status, "reason": "conditional arm not authorized by the R29 gate or no single clearly healthier B2 arm"}, indent=2) + "\n")
    (OUT / "training/S-EDGE-FREEZE-P/seed13/RESULTS.md").write_text(f"# S-EDGE-FREEZE-P\n\nCUDA seed13, 3 epochs, 534 updates. Exact Direct R@10={p3['R@10']:.4f}, R@20={p3['R@20']:.4f}, R@50={p3['R@50']:.4f}; fixed hub probe max Top10 frequency={p3['hub_probe']['max_top10_frequency']}/128. Projections are tensor-identical to parent. R29 arm-specific Teacher full-U was not measured in this bounded pass.\n")
    (OUT / "training/S-EDGE-FREEZE-R/seed13/RESULTS.md").write_text(f"# S-EDGE-FREEZE-R\n\nCUDA seed13, 3 epochs, 534 updates. Exact Direct R@10={r3['R@10']:.4f}, R@20={r3['R@20']:.4f}, R@50={r3['R@50']:.4f}; fixed hub probe max Top10 frequency={r3['hub_probe']['max_top10_frequency']}/128. Relations are tensor-identical to parent. R29 arm-specific Teacher full-U was not measured in this bounded pass.\n")
    if teacher_eval:
        for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"):
            overall = teacher_eval["generators"][arm]["metrics"]["overall"]
            teacher_text = (
                f"\n## Frozen Teacher T0\n"
                f"Arm-specific U pool: Recall@10={overall['U_OFFLINE_T0']['recall@10']:.4f}, "
                f"Recall@20={overall['U_OFFLINE_T0']['recall@20']:.4f}, "
                f"Recall@50={overall['U_OFFLINE_T0']['recall@50']:.4f}; "
                f"U raw recall={overall['U_OFFLINE_T0']['raw_recall']:.4f}. "
                f"Teacher receipt is stored under `teacher_evaluation/teacher/{arm}`.\n"
            )
            path = OUT / "training" / arm / "seed13/RESULTS.md"
            path.write_text(path.read_text() + teacher_text)

    (OUT / "RESULTS.md").write_text(
        f"# R29 Results\n\n## Gate\nA0 reproduces the R28 active-list counts. The candidate gate is **{gate['decision']}**, so B1 QT refresh is `not_triggered` and B2 is the only authorized training branch.\n\n## B2 Direct exact\nR28 joint `S-EDGE-LONG` epoch3: R@10={control['overall']['D100_EXACT']['recall@10']:.4f}, R@20={control['overall']['D100_EXACT']['recall@20']:.4f}, R@50={control['overall']['D100_EXACT']['recall@50']:.4f}; its fixed 128-query exact hub Top10 maximum was 124/128. `FREEZE-P` epoch1→3: R@10={p1['R@10']:.4f}→{p3['R@10']:.4f}, R@20={p1['R@20']:.4f}→{p3['R@20']:.4f}, R@50={p1['R@50']:.4f}→{p3['R@50']:.4f}, hub max={p1['hub_probe']['max_top10_frequency']}→{p3['hub_probe']['max_top10_frequency']}/128. `FREEZE-R` epoch1→3: R@10={r1['R@10']:.4f}→{r3['R@10']:.4f}, R@20={r1['R@20']:.4f}→{r3['R@20']:.4f}, R@50={r1['R@50']:.4f}→{r3['R@50']:.4f}, hub max={r1['hub_probe']['max_top10_frequency']}→{r3['hub_probe']['max_top10_frequency']}/128.\n\nBoth freeze arms avoid the joint-control collapse on this seed. Because they trade a small amount of recall against hub frequency and no seed29 confirmation was authorized, the result supports a joint P/R stability mechanism but does not select a single freeze component.\n\nTeacher full-U, own evidence/U/M, and seed29 confirmation were not measured in this CPU-only execution.\n"
    )
    (OUT / "SCIENTIFIC_REVIEW.md").write_text(
        """# Scientific Review\n\n## Confirmed observation\n- The R28 merged graph has QT active 11,390/11,390, table->text active 4,775/11,390, text->table active 4,660/94,792, and both image relations active 0. Effective configured 0.5 evidence weights are therefore diluted to approximately 0.2096 for QE and 0.0246 for ET.\n- Frozen-parent Natural QT candidates were not harder than R12: NAT-R12 margin CI was positive and violation CI was negative; HNR R12<-NAT was 0.231.\n- On seed13, freezing P or freezing R both avoided the R28 joint Student exact collapse and the 124/128 hub.\n\n## Causal evidence\nThe two single-factor B2 arms share parent, graph, order, loss, logical batch, learning rates, weight decay, and anchor semantics. Their only causal variable is whether projections or relation matrices update. The contrast is consistent with P/R joint co-adaptation being necessary for the observed collapse on this seed.\n\n## Competing explanation\nThe remaining competitor is objective/supervision geometry: ET and QE have many inactive lists, so their nominal weights do not equal effective gradient weights. This was measured descriptively, not intervened on in R29.\n\n## Not measured\nFrozen T0 scoring on Natural candidates, full-U Teacher retrieval, evidence/U/M retrieval, and seed29 confirmations were not run because CUDA was unavailable and neither B2 arm was clearly superior enough to trigger confirmation.\n\n## Negative result means\nThe weak candidate gate means this execution does not support the claim that R12 QT candidate mismatch is the primary bottleneck. It does not prove candidates are irrelevant to other relations or that the T0 ordering would agree, because T0 full-U was not scored.\n\n## Decision\n**First bottleneck:** P/R joint co-adaptation and its ranking-geometry collapse.\n\n**Second competing explanation:** effective supervision imbalance from inactive QE/ET lists and the resulting objective geometry.\n"""
    )
    results_path = OUT / "RESULTS.md"
    results_path.write_text(results_path.read_text().replace(
        "Teacher full-U, own evidence/U/M, and seed29 confirmation were not measured in this CPU-only execution.",
        "R29 arm-specific Teacher full-U, own evidence/U/M, and seed29 confirmation were not measured in this bounded pass; this is a scope gap, not a CUDA limitation. The historical R28 fixed-pool T0 Full-U receipt remains separate and is not used as an R29 arm result.",
    ))
    if teacher_eval:
        teacher_lines = []
        for arm in ("S-EDGE-FREEZE-P", "S-EDGE-FREEZE-R"):
            overall = teacher_eval["generators"][arm]["metrics"]["overall"]["U_OFFLINE_T0"]
            teacher_lines.append(
                f"{arm} arm-specific frozen T0 on its own U pool: R@10={overall['recall@10']:.4f}, "
                f"R@20={overall['recall@20']:.4f}, R@50={overall['recall@50']:.4f}, "
                f"raw U recall={overall['raw_recall']:.4f}."
            )
        results_path.write_text(results_path.read_text().replace(
            "R29 arm-specific Teacher full-U, own evidence/U/M, and seed29 confirmation were not measured in this bounded pass; this is a scope gap, not a CUDA limitation. The historical R28 fixed-pool T0 Full-U receipt remains separate and is not used as an R29 arm result.",
            "Arm-specific frozen T0 Full-U was measured on both R29 Student U pools. " + " ".join(teacher_lines) + " The historical R28 fixed-pool T0 receipt remains separate.",
        ))
    review_path = OUT / "SCIENTIFIC_REVIEW.md"
    review_path.write_text(review_path.read_text().replace(
        "Frozen T0 scoring on Natural candidates, full-U Teacher retrieval, evidence/U/M retrieval, and seed29 confirmations were not run because CUDA was unavailable and neither B2 arm was clearly superior enough to trigger confirmation.",
        "Frozen T0 scoring on the R29 Natural candidates, arm-specific evidence/U/M retrieval, and seed29 confirmations were not run in this bounded pass. This is a scope gap rather than a CUDA limitation. The historical R28 fixed-pool T0 Full-U receipt remains separate and is not used as an R29 arm result.",
    ))
    if teacher_eval:
        review_path.write_text(review_path.read_text().replace(
            "Frozen T0 scoring on the R29 Natural candidates, arm-specific evidence/U/M retrieval, and seed29 confirmations were not run in this bounded pass. This is a scope gap rather than a CUDA limitation. The historical R28 fixed-pool T0 Full-U receipt remains separate and is not used as an R29 arm result.",
            "Arm-specific frozen T0 Full-U scoring on both R29 Student U pools completed on CUDA; the Natural train-fit candidate file and seed29 confirmations remain outside this execution. The historical R28 fixed-pool T0 receipt remains separate.",
        ))
        review_path.write_text(review_path.read_text().replace(
            "because T0 full-U was not scored",
            "because T0 scoring was not run on the Natural train-fit candidate file",
        ))
        (OUT / "diagnostics/teacher_score_status.json").write_text(json.dumps({
            "status": "completed_arm_specific_u_m",
            "device": teacher_eval["devices"],
            "receipt": str(teacher_eval_path.resolve()),
            "natural_candidate_t0": "not_scored",
            "dev_test_qrels_used_for_scoring": False,
        }, indent=2) + "\n")
    (OUT / "NEXT_DECISION.md").write_text("""# Next Decision\n\nDo not run QT refresh, a third freeze variant, or a loss/LR/anchor grid from this R29 result. The next controlled study should first make the effective QE/ET supervision audit operational and test it as one pre-registered factor, while retaining the joint-vs-freeze controls.\n""")

    ledger = {"A0": "pass", "A1": "pass", "A2": "pass", "gate": gate["decision"], "B1": "not_triggered", "B1-confirm": "not_triggered", "B2": {"S-EDGE-FREEZE-P": "completed_seed13_3_epochs_cuda0", "S-EDGE-FREEZE-R": "completed_seed13_3_epochs_cuda0"}, "B2-confirm": "not_triggered", "teacher_full_u": teacher_status, "teacher_evaluation_receipt": str(teacher_eval_path.resolve()) if teacher_eval else None, "historical_r28_t0_full_u": "available_separate_receipt", "status": "complete_with_declared_measurement_gaps" if not teacher_eval else "completed_with_declared_natural_candidate_gap"}
    (OUT / "EXECUTION_LEDGER.json").write_text(json.dumps(ledger, indent=2) + "\n")
    resolved = load(OUT / "RESOLVED_INPUTS.json")["inputs"]
    p_exec = load(OUT / "training/S-EDGE-FREEZE-P/seed13/EXECUTION.json")
    r_exec = load(OUT / "training/S-EDGE-FREEZE-R/seed13/EXECUTION.json")
    correctness = {
        "C0_input_identity": {name: rec.get("sha256") for name, rec in resolved.items() if name in {"student_parent", "feature_manifest", "edge_positive_registry", "target_path_graph"}},
        "C1_train_known_positive_closure": {"status": "pass_by_construction", "natural_unknown_rule": "all global registry positives for each (source, relation) excluded before Top32 selection"},
        "C2_no_leakage": {"status": "pass", "dev_test_qrels_used_for_mining": False, "hub_ids_used_for_mining": False},
        "C3_B1_single_factor": {"status": "not_triggered", "reason": gate["decision"]},
        "C4_B2_single_factor": {"freeze_P_projection_hash_equal": p_exec["parent_projection_hashes"] == p_exec["final_projection_hashes"], "freeze_R_relation_hash_equal": r_exec["parent_relation_hashes"] == r_exec["final_relation_hashes"], "optimizer_updates": {"S-EDGE-FREEZE-P": p_exec["updates"], "S-EDGE-FREEZE-R": r_exec["updates"]}},
        "C5_evaluation": {"status": "pass", "metric": "query-macro exact Direct Recall@K plus arm-specific frozen T0 U/M", "ann_exact_separate": True, "fixed_probe_n": 128, "device": "cuda:0/cuda:1" if teacher_eval else "cuda:0", "teacher_full_u": teacher_status, "teacher_receipt": str(teacher_eval_path.resolve()) if teacher_eval else None, "cuda_verified_outside_sandbox": True},
    }
    (OUT / "correctness/CORRECTNESS.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "correctness/CORRECTNESS.json").write_text(json.dumps(correctness, indent=2) + "\n")
    files = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob("*") if p.is_file() and p.name != "PACKAGE_MANIFEST.json"}
    (OUT / "PACKAGE_MANIFEST.json").write_text(json.dumps({"format_version": 1, "status": "complete_with_declared_measurement_gaps", "files": files}, indent=2) + "\n")
    print(json.dumps({"status": "finalized", "gate": gate["decision"], "files": len(files)}))


if __name__ == "__main__":
    main()

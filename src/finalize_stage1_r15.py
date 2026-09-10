#!/usr/bin/env python
"""Assemble and independently cross-check the frozen R15 experiment evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_stage1_r15_interaction import ARMS, KS, RULES
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def dependency(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": checkpoint_fingerprint(path),
            "bytes": path.stat().st_size}


def positive_ranks(row: dict[str, Any]) -> dict[str, Any]:
    minimum = row.get("exact_rank_min", row.get("exact_direct_rank_min"))
    maximum = row.get("exact_rank_max", row.get("exact_direct_rank_max"))
    return {
        "exact_direct_rank": row.get("exact_direct_rank", minimum if minimum == maximum else None),
        "exact_direct_rank_min": minimum, "exact_direct_rank_max": maximum,
        "exact_rank_tied": minimum != maximum,
    }


def collect_audits(root: Path, output: Path) -> tuple[dict, list, dict]:
    """Combine complete per-source panels and checkpoint diagnostics, retaining provenance."""

    panels, diagnostic_rows, endpoints, timelines = {}, [], {}, {}
    supplement = output / "stageG_correctness/deployment_supplement"
    for family in ("full", "eoff"):
        for letter in ("l", "n"):
            arm = f"{letter}_{family}"
            timelines[arm] = []
            for step in (0, 45, 89, 178):
                if family == "full" or step == 0:
                    directory = output / "stageG_correctness" / f"{letter}_eoff"
                else:
                    directory = output / "stageI_interaction" / f"{letter}_eoff_seed13/correctness"
                path = directory / f"step_{step:06d}/metrics.json"
                payload = read_json(path)
                if payload["status"] != "complete" or not payload["score_path_consistency"]["passed"]:
                    raise ValueError(f"Incomplete or failed numerical audit: {path}")
                checkpoint_metadata = read_json(Path(payload["checkpoint"]).with_suffix(".json"))
                if family == "eoff":
                    checkpoint_metadata = read_json(
                        output / "stageI_interaction" / f"{letter}_eoff_seed13/checkpoints/step_{step:06d}.json"
                    )
                aggregate = payload["full_dev_direct"]["aggregate"]["all"]
                timelines[arm].append({"step": step, **aggregate, "source": dependency(path)})
                key = f"{arm}_step{step}"
                panels[key] = {"source": dependency(path), "relations": payload["exact_relation_panels"]}
                diagnostic_rows.append({
                    "arm": arm, "step": step, "source": dependency(path),
                    "checkpoint": payload["checkpoint"], "checkpoint_sha256": payload["checkpoint_sha256"],
                    "geometry": payload["projection_and_parameter_diagnostics"],
                    "optimizer": checkpoint_metadata.get("optimizer", payload["optimizer"]),
                    "last_gradient_norms": checkpoint_metadata["last_gradient_norms"],
                    "gradient_measurement": "after loss.backward(), before optimizer.step(); no clipping",
                    "fixed_training_batch": checkpoint_metadata["fixed_training_batch"],
                    "normalization": payload["normalization"],
                    "training_score_decomposition": "stageG_correctness/training_scores/summary.json",
                    "natural_responsibility": "stageI_interaction/statistics/witness_summary.json",
                })
                # Small manifests remain independently readable after temporary index removal.
                manifest_path = path.parent / "index_manifest.json"
                write_json(manifest_path, payload["index"]["manifest"])
                if step == 178:
                    endpoints[arm] = payload["full_dev_direct"]
            if family == "full":
                path = directory / "step_000178_adapter_off/metrics.json"
                payload = read_json(path)
                correction = read_json(supplement / f"{letter}_eoff_full.json")
                panels[f"{arm}_adapter_off"] = {
                    "source": dependency(path), "relations": payload["exact_relation_panels"]
                }
                diagnostic_rows.append({
                    "arm": arm, "step": 178, "intervention": "adapter_off",
                    "geometry": correction["adapter_off_geometry_correction"],
                    "geometry_correction": correction["adapter_off_geometry_correction_reason"],
                    "source": dependency(supplement / f"{letter}_eoff_full.json"),
                    "optimizer": payload["optimizer"],
                    "interpretation": "trained endpoint P/R; mechanism probe, not independently trained S control",
                })
    for arm in ("s_full", "s_eoff"):
        path = output / "stageG_correctness/references" / f"{arm}.json"
        payload = read_json(path)
        panels[f"{arm}_step178"] = {"source": dependency(path), "relations": payload["exact_relation_panels"]}
        if payload.get("full_dev_direct") is not None:
            endpoints[arm] = payload["full_dev_direct"]
        diagnostic_rows.append({
            "arm": arm, "step": 178, "source": dependency(path),
            "geometry": payload["projection_and_parameter_diagnostics"],
            "residual": "not applicable: original linear P",
        })
    fixed_sources = None
    for entry in panels.values():
        source_ids = {}
        for relation, panel in entry["relations"].items():
            source_ids[relation] = [row["source_id"] for row in panel["per_source"]]
            for mode in ("exact", "ann"):
                hubs = Counter(value for row in panel["per_source"] for value in row[f"{mode}_ids"])
                panel[f"{mode}_hub_top20"] = hubs.most_common(20)
                panel[f"{mode}_distinct_objects"] = len(hubs)
                panel[f"{mode}_neighbor_occurrences"] = sum(hubs.values())
            margins = [edge["score"] - row["cutoff_score"]
                       for row in panel["per_source"] for edge in row["gt_edges"]]
            panel["positive_margin_over_cutoff_quantiles"] = (
                dict(zip(("min", "p10", "p50", "p90", "max"),
                         np.quantile(margins, [0, .1, .5, .9, 1]).tolist())) if margins else None
            )
        if fixed_sources is not None and source_ids != fixed_sources:
            raise ValueError("Exact source panel identity/order changed between checkpoints")
        fixed_sources = source_ids
    write_json(output / "exact_relation_panels.json", {
        "status": "complete", "fixed_source_ids": fixed_sources, "all_panels_same_sources": True,
        "panels": panels, "full_dev_timelines": timelines,
        "rank_ties": "Positive rank intervals and torch.topk boundary ties retained in source records.",
    })
    random_geometry = read_json(output / "stageG_correctness/random_geometry.json")
    mean_vector_path = output / "stageG_correctness/mean_vectors.jsonl.gz"
    mean_rows = read_gzip(mean_vector_path)
    mean_by_key = {
        (row["arm"], row["step"], row.get("intervention")): row for row in mean_rows
    }
    if len(mean_rows) != 20 or len(mean_by_key) != 20:
        raise ValueError("G4 mean-vector row population differs")
    recovery = read_json(output / "RECOVERED_TRAINING_SOURCE.json")
    for row in diagnostic_rows:
        arm = row["arm"]
        key = f"step_{row['step']:06d}" + ("_adapter_off" if row.get("intervention") else "")
        random_arm = arm.replace("_eoff", "_full") if row["step"] == 0 else arm
        mean_row = mean_by_key.pop((arm, row["step"], row.get("intervention")))
        for kind, geometry in row["geometry"]["by_type"].items():
            mean_geometry = mean_row["by_type"][kind]
            if (mean_geometry["samples"] != geometry["samples"]
                    or mean_geometry["sample_ids_sha256"] != geometry["sample_ids_sha256"]):
                raise ValueError(f"G4 mean-vector panel differs: {arm}/{row['step']}/{kind}")
            for output_name in ("base_output", "full_output", "s0_output"):
                values = mean_geometry[output_name]
                vector = values["mean_vector"]
                if len(vector) != 1024 or not np.isfinite(vector).all():
                    raise ValueError(
                        f"Invalid G4 mean vector: {arm}/{row['step']}/{kind}/{output_name}"
                    )
                target = geometry.setdefault(output_name, {})
                if "mean_vector_norm" in target and not np.isclose(
                    target["mean_vector_norm"], values["mean_vector_norm"], atol=1e-6, rtol=0
                ):
                    raise ValueError(
                        f"G4 mean-vector norm differs: {arm}/{row['step']}/{kind}/{output_name}"
                    )
                target.update(values)
            geometry["full_output"]["fixed_adjacent_pair_cosine"] = geometry["full_output"].pop("random_pair_cosine")
            geometry["full_output"]["random_pair_cosine"] = random_geometry["results"][random_arm][key]["by_type"][kind]["random_pair_cosine"]
            geometry["random_pairing_source"] = "stageG_correctness/random_geometry.json"
            geometry["mean_vector_source"] = "stageG_correctness/mean_vectors.jsonl.gz"
        optimizer = row.get("optimizer", {})
        if optimizer.get("executed_source_matches_current") is False:
            optimizer["historical_state_limitation"] = (
                "Historical Adam moments not serialized. Groups reconstructed from current source; "
                "the original audit's current-file hash mismatch is preserved. Supplemental post hoc R14 "
                "entrypoint reconstruction exactly matches recorded training hashes and independently "
                "confirms the unchanged optimizer-definition AST, not actual optimizer tensors or full runtime history."
            )
            optimizer["supplemental_entrypoint_source_recovery"] = {
                "receipt": "RECOVERED_TRAINING_SOURCE.json",
                "recovered_source": recovery["r14_entrypoint"]["recovered_source"],
                "optimizer_definition_source_identity_verified": True,
                "historical_Adam_moment_tensors_verified": False,
            }
    if mean_by_key:
        raise ValueError(f"Unused G4 mean-vector rows: {sorted(mean_by_key)}")
    write_gzip(output / "projection_and_optimizer_diagnostics.jsonl.gz", diagnostic_rows)
    return endpoints, diagnostic_rows, timelines


def assemble_provenance(output: Path, endpoints: dict[str, Any], root: Path) -> dict[str, Any]:
    b13 = {row["query_id"]: row for row in read_gzip(output / "stageC_candidate_delivery/candidate_provenance.jsonl.gz")}
    witnesses = {(row["arm"], row["query_id"], row["target_id"]): row
                 for row in read_gzip(output / "witness_funnel.jsonl.gz")}
    per_query = read_gzip(output / "per_query_metrics.jsonl.gz")
    base_queries = {row["query_id"]: row for row in per_query}
    priority = {(row["query_id"], row["target_id"]) for row in read_gzip(
        output / "stageC_candidate_delivery/B13_evidence_only_known_witness_cases.jsonl.gz"
    )}
    output_rows, verification, summary = [], [], {}
    for arm, relative in ARMS.items():
        exact_rows = ({row["query_id"]: row for row in endpoints[arm]["per_query"]}
                      if arm != "s_full" else {})
        rows = read_gzip(root / "work" / relative / "evaluation_step178/path_pool.jsonl.gz")
        arm_rows = []
        for record in rows:
            query_id = record["query_id"]
            base = base_queries[query_id]
            rankings = base["arms"][arm]["rankings"]
            positives = set(record["positive_target_ids"])
            direct_scores, evidence = {}, set()
            for target, paths in record["paths_by_target"].items():
                for path in paths:
                    if path["kind"] == "direct":
                        direct_scores[target] = float(path["path_score"])
                    else:
                        evidence.add(target)
            direct = sorted(direct_scores, key=lambda target: (-direct_scores[target], target))
            exact = b13[query_id]["D100_exact"] if arm == "s_full" else exact_rows[query_id]["exact_ids"]
            if arm != "s_full" and set(direct) != set(exact_rows[query_id]["ann_ids"]):
                raise ValueError(f"Saved natural D100 differs from audited index: {arm}/{query_id}")
            exact_by_target = ({row["target_id"]: row for row in b13[query_id]["positive_targets"]}
                               if arm == "s_full" else
                               {row["target_id"]: row for row in exact_rows[query_id]["positive_targets"]})
            positive_rows = []
            for target in sorted(positives):
                witness = witnesses[(arm, query_id, target)]
                positive = {
                    "target_id": target, "natural_direct_rank": direct.index(target) + 1 if target in direct else None,
                    "natural_direct_rank_censored_at": 100, **positive_ranks(exact_by_target[target]),
                    "in_D100_ANN": target in direct, "in_D100_exact": target in exact,
                    "in_E": target in evidence, "in_U": target in set(direct) | evidence,
                    "ann_evidence_only": target in evidence and target not in direct,
                    "exact_evidence_only": target in evidence and target not in exact,
                    "relative_B13_ANN_new": target in evidence and target not in b13[query_id]["D100_ANN"],
                    "relative_B13_exact_new": target in evidence and target not in b13[query_id]["D100_exact"],
                    "known_QET": witness["known_QET_pair"],
                    "known_witness_ids": witness["known_evidence_ids"],
                    "retained_evidence_ids": witness["deployed_selected_evidence_ids"],
                    "retained_known_ids": witness["deployed_known_evidence_ids"],
                    "raw_known_rows": witness["raw_known_supported_rows"],
                    "retained_known_rows": witness["deployed_known_supported_rows"],
                    "delivery": witness["delivery"],
                    "known_paths_do_not_verify_values": True,
                }
                positive_rows.append(positive)
                if positive["ann_evidence_only"] or positive["exact_evidence_only"] or positive["relative_B13_exact_new"]:
                    verification.append({
                        "arm": arm, "query_id": query_id, "source_table_id": base["source_table_id"],
                        "query_kind": base["query_kind"], "target_id": target,
                        "B13_priority_queue": arm == "s_full" and (query_id, target) in priority,
                        "stage1_outside_own_exact_direct100": target not in exact,
                        "stage2_natural_QET_reached": target in evidence,
                        "stage3_delivered_N50": {rule: witness["delivery"][rule]["50"]["delivered"] for rule in RULES},
                        "known_witness_retained": bool(witness["deployed_known_evidence_ids"]),
                        "actual_delivered_evidence_ids": {rule: witness["delivery"][rule]["50"]["delivered_evidence_ids"] for rule in RULES},
                        "stage4_independent_attribute_support": None,
                        "stage5_correct_value_and_join": None,
                        "stage6_stage2_final_top10": None,
                        "stage1_top10": {rule: witness["delivery"][rule]["10"]["delivered"] for rule in RULES},
                        "independent_entity_verification": None, "independent_image_information": None,
                        "without_evidence_counterfactual": None, "reviewed_wrong_evidence_counterfactual": None,
                        "verification_status": "unknown_not_independently_reviewed",
                        "candidate_provenance": positive,
                    })
            row = {
                "arm": arm, "query_id": query_id, "source_table_id": base["source_table_id"],
                "query_kind": base["query_kind"], "positive_target_ids": sorted(positives),
                "positive_denominator": len(positives), "D100_ANN": direct, "D100_exact": exact,
                "E": sorted(evidence), "U": sorted(set(direct) | evidence),
                "C50": {rule: rankings[rule]["50"]["target_ids"] for rule in RULES},
                "Top10_stage1": {rule: rankings[rule]["10"]["target_ids"] for rule in RULES},
                "Top10_stage2": None, "positive_targets": positive_rows,
                "rrf_C50_entered_vs_f1": sorted(set(rankings["union_rrf_equal"]["50"]["target_ids"]) - set(rankings["f1_union_direct"]["50"]["target_ids"])),
                "rrf_C50_exited_vs_f1": sorted(set(rankings["f1_union_direct"]["50"]["target_ids"]) - set(rankings["union_rrf_equal"]["50"]["target_ids"])),
            }
            if len(direct) != 100 or len(exact) != 100 or any(len(ids) != 50 for ids in row["C50"].values()):
                raise ValueError(f"Fixed candidate budget failed: {arm}/{query_id}")
            for rule in RULES:
                for k in KS:
                    ids = rankings[rule][str(k)]["target_ids"]
                    if not set(ids) <= set(row["U"]):
                        raise ValueError("Delivered target did not originate in natural U")
                    if abs(len(positives & set(ids)) / len(positives) - rankings[rule][str(k)]["recall"]) > 1e-12:
                        raise ValueError("Saved recall does not match target IDs and fixed qrels")
            arm_rows.append(row)
        summary[arm] = {}
        for kind in ("all", "implicit", "explicit"):
            selected = [row for row in arm_rows if kind == "all" or row["query_kind"] == kind]
            pair_rows = [positive for row in selected for positive in row["positive_targets"]]
            summary[arm][kind] = {
                "queries": len(selected), "positive_pairs": len(pair_rows),
                "RawUnionRecall": float(np.mean([len(set(row["positive_target_ids"]) & set(row["U"])) / row["positive_denominator"] for row in selected])),
                "mean_U_size": float(np.mean([len(row["U"]) for row in selected])),
                **{key: {"positive_pairs": sum(row[key] for row in pair_rows),
                          "query_macro_recall": float(np.mean([sum(positive[key] for positive in row["positive_targets"]) / row["positive_denominator"] for row in selected]))}
                   for key in ("ann_evidence_only", "exact_evidence_only", "relative_B13_ANN_new", "relative_B13_exact_new")},
                "exact_new_known_QET_pairs": sum(row["exact_evidence_only"] and row["known_QET"] for row in pair_rows),
                "exact_new_retained_known_C50_pairs": {
                    rule: sum(row["exact_evidence_only"] and row["delivery"][rule]["50"]["known_witness_delivered"] for row in pair_rows)
                    for rule in RULES
                },
            }
        output_rows.extend(arm_rows)
    if len(output_rows) != 6 * 1198 or sum(row["B13_priority_queue"] for row in verification) != 82:
        raise ValueError("Six-arm query population or fixed priority queue is incomplete")
    write_gzip(output / "candidate_provenance.jsonl.gz", output_rows)
    write_gzip(output / "evidence_only_verification.jsonl.gz", verification)
    write_json(output / "statistics/discovery_summary.json", {
        "status": "complete", "arms": summary, "query_rows": len(output_rows),
        "verification_rows": len(verification), "B13_priority_pairs": 82,
        "independent_value_join_verification": None,
    })
    return {"arms": summary, "query_rows": len(output_rows), "verification_rows": len(verification)}


def snapshot_sources(root: Path, output: Path) -> dict[str, Any]:
    previous_manifest = output / "source_snapshot/MANIFEST.json"
    previous = read_json(previous_manifest) if previous_manifest.is_file() else {}
    previous_files = {record["source"]: record for record in previous.get("files", [])}
    paths = [
        "src/mmdd_stage1/models.py", "src/mmdd_stage1/training.py", "src/mmdd_stage1/objectives.py",
        "src/mmdd_stage1/scoring.py", "src/mmdd_stage1/retrieval.py", "src/mmdd_stage1/checkpoints.py",
        "src/mmdd_stage1/features.py", "src/mmdd_stage1/row_support.py", "src/mmdd_stage1/data.py",
        "src/run_stage1_r11_task_e.py", "src/run_stage1_r11_task_f.py", "src/run_stage1_r13.py",
        "src/run_stage1_r14.py", "src/reevaluate_stage1_r12_checkpoints.py", "src/finalize_stage1_r13.py",
        "src/run_stage1_r15.py", "src/audit_stage1_r15_g.py", "src/audit_stage1_r15_deployment.py",
        "src/audit_stage1_r15_training_scores.py", "src/audit_stage1_r15_training_invariants.py",
        "src/audit_stage1_r15_random_geometry.py", "src/audit_stage1_r15_mean_vectors.py",
        "src/audit_stage1_r15_runtime_sources.py",
        "src/audit_stage1_r15_references.py", "src/analyze_stage1_r15_interaction.py",
        "src/analyze_stage1_r15_candidates.py", "src/finalize_stage1_r15.py",
        "src/recover_stage1_r15_training_source.py", "src/validate_stage1_r15_artifacts.py",
        "tests/test_stage1_r15.py", "tests/test_stage1_r15_training_scores.py",
        "tests/test_stage1_r15_interaction.py", "tests/test_stage1_r15_candidates.py",
        "tests/test_stage1_r15_runtime_sources.py",
    ]
    for optional in ("src/build_stage1_r15_report.py", "tests/test_stage1_r15_references.py",
                     "src/audit_stage1_r15_original_cases.py"):
        if (root / optional).is_file():
            paths.append(optional)
    records, refreshed = [], []
    for relative in paths:
        source = root / relative
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({"source": relative, "snapshot": str(destination.relative_to(output)),
                        "sha256": checkpoint_fingerprint(source), "bytes": source.stat().st_size})
        old_sha256 = previous_files.get(relative, {}).get("sha256")
        if old_sha256 != records[-1]["sha256"]:
            refreshed.append({"source": relative, "old_sha256": old_sha256,
                              "new_sha256": records[-1]["sha256"],
                              "reason": "Refresh current executable snapshot after reviewed source-recovery integration"})
    recovery = read_json(output / "RECOVERED_TRAINING_SOURCE.json")
    payload = {"status": "complete", "files": records,
               "scope": "Executable projection, loss, optimizer, index, retention, analysis and tests; no model/data artifacts or secrets.",
               "recovered_historical_entrypoint": recovery["recovered_source"],
               "recovered_historical_R14_entrypoint": recovery["r14_entrypoint"]["recovered_source"],
               "recovery_receipt": dependency(output / "RECOVERED_TRAINING_SOURCE.json"),
               "historical_source_limitation": "R14 and R15 historical entrypoints were reconstructed post hoc with exact SHA256 matches to recorded training manifests; training/shared function ASTs, including the R14 optimizer definition, match current source. Other imported-module runtime identities and Adam moment tensors remain unverified; entrypoint identity and current source snapshots do not establish the full historical execution environment."}
    payload["refresh_history"] = previous.get("refresh_history", [])
    if refreshed:
        payload["refresh_history"].append({"at_utc": datetime.now(timezone.utc).isoformat(),
                                            "entries": refreshed, "historical_identity_claim": False})
    write_json(output / "source_snapshot/MANIFEST.json", payload)
    return payload


def build_report(summary: dict[str, Any]) -> str:
    interaction, candidate, witnesses = summary["interaction"], summary["candidates"], summary["witnesses"]
    original_cases = summary.get("original_cases_validation")
    cases_note = "原报告提到的82-case CSV在初次核验时未提供，因此当时依冻结B13的ANN外且known-witness QET规则重建；没有根据最终成功与否重新挑选。"
    if original_cases is not None:
        cases_note += (f"用户随后提供原CSV，已于{original_cases['checked_at_utc'][:10]}按(query_id,target_id)逐项核验：82个唯一正对与重建队列完全一致，原30列共2460个语义单元全部匹配。原件与扩展重建CSV的行序、列名/新增字段及字节不相同，不将语义一致写成文件字节一致。"
                       f"原件SHA256为{original_cases['original']['sha256']}，证据见ORIGINAL_CASES_VALIDATION.json。旧C配置、C汇总和归档审计中的‘原件缺失’保留为当时状态；该缺项现已补齐，但不构成独立属性/值/实体/连接验证。")
    else:
        cases_note += "数量/模态/行数等已公开特征复现，但尚无法与原件逐ID核验。"
    lines = [
        "# R15：残差失败归因与候选交付审计", "",
        "## 结论与实验状态", "",
        "G/I/C 的计算与结果分析已完成，但历史归档核验仍有缺项，且部分 G 补充检查晚于 I；不能视为计划逐项验证通过。新增训练严格为 L-Eoff 与 N-Eoff 各 178 updates（总 356），没有新 Teacher 推理。V 按计划的独立预算条件未执行，属性值、实体、连接正确性均为 unknown。",
        "两种残差在关闭 E-channel CE/KD 后仍然发生 full-lake exact Q→T 坍塌。E-loss 不是这次直接检索失败的必要条件；但关闭它显著恢复了 known-witness 的证据邻域与路径覆盖。现有证据支持将直接分数退化与证据分支退化分别研究。",
        "冻结 B13 中存在确实位于 exact direct100 之外的已知支持路径；当前 F1 交付规则几乎没有保留这一机会。固定 equal-RRF 在 N=50 保留更多这类路径；总体候选 Recall 的改善区间跨零，Stage1 Top10 更差，尚未证明最终 joinability 收益。保留 B13 为冻结参考，停止本轮残差扩容及额外 seed 网格。", "",
        "## 评测分母与固定干预", "",
        "全部主结果为 1,198 个 dev query（599 implicit、599 explicit，1,000 source groups，1,279 known positive pairs）的 query-macro target Recall；固定每个 G_q，不按 pair micro 或获益子集重加权。以下数值为百分数，差值与区间为百分点。unknown qrels/witness 不作为错误标签。",
        "F1 为已有 union-direct 排序；pure-direct 使用自身 ANN D100；equal-RRF 为同一自然候选池上的固定等权 RRF60。候选交付预算 N=50，Stage1 截断 K=10。完整 U 是未截断并集，RawUnionRecall 不能称为 CandidateRecall@50。",
        "S-full/S-Eoff/L-full/N-full 复用 seed13 的 R13/R14。预定干预仅将 E-channel CE/KD 系数设为0。现有 manifest、checkpoint 与共享 loader 检查支持：新臂复用对应 residual step0 的字节一致 checkpoint、A/B/c_tau、S0 P/R、同一候选/Teacher/顺序，并采用 fresh AdamW、base-only anchor 和完整 evidence forward；P/A/B/R 保持可训练。R14/R15训练入口已按训练manifest的SHA256精确恢复，当前训练及共享函数AST一致；其他导入模块的历史运行版本与完整Adam state仍缺失，因此上述配置和入口身份匹配不等于历史执行过程已被完整独立核验。", "",
        "## 主端点：关闭 E-loss 仍未恢复直接检索", "",
        "| arm | F1 @10 / @20 / @50 | implicit @10 | explicit @10 | pure @10 | RRF @10 / @50 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, metrics in interaction["arm_metrics"].items():
        full, imp, exp = metrics["all"], metrics["implicit"], metrics["explicit"]
        values = " / ".join(f"{100*full['f1_union_direct'][str(k)]:.4f}" for k in KS)
        lines.append(f"| {arm} | {values} | {100*imp['f1_union_direct']['10']:.4f} | {100*exp['f1_union_direct']['10']:.4f} | {100*full['pure_direct100']['10']:.4f} | {100*full['union_rrf_equal']['10']:.4f} / {100*full['union_rrf_equal']['50']:.4f} |")
    lines += ["", "N-Eoff 比 L-Eoff 的小幅提高发生在两者均失败的区域；它远低于 S-Eoff，不构成非线性方法投入依据。没有达到追加 seed17/23 的 +0.5pp 相对完整参考收益及机制/成本门槛。", "",
              "## Exact 时间线与残差干预", "",
              "| arm | step0 exact | step45 exact | step89 exact | step178 exact | step178 ANN |", "|---|---:|---:|---:|---:|---:|"]
    for arm, timeline in summary["timelines"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{100*row['exact_recall@10']:.4f}" for row in timeline) + f" | {100*timeline[-1]['ann_recall@10']:.4f} |")
    lines += ["", "L-full/N-full 的 endpoint exact 与 ANN 均失败；同一已训练 P/R 的 adapter-off exact R@10 分别恢复至 29.2988% / 29.1736%。这说明残差参与的完整打分变化是重要中介，但 P/R 已共同适配，adapter-off 不是独立训练的 S 控制。",
              "L-full 合并 P+BA/c 后五关系 score 最大差不超过 3.58e-7，exact Recall 完全相同，Top100 集合完全相同；10 个 query 的内部顺序存在 FP32 边界差异。合并向量、实际 HNSW 缓存内积、train-fit/dev 五关系和实际 save/load 补充审计均有逐对记录。",
              "线性残差可精确合并，因此它没有扩大该线性函数类；本轮失败应讨论当前优化、尺度与正则配方，不能写作‘容量过大已被证明’。", "",
              "## 配对交互：主端点与证据路径呈现不同变化", "",
              "| contrast | F1 R@10 delta (pp) | source-group 95% CI (pp) | win / loss / tie |", "|---|---:|---:|---:|"]
    for name in ("I_L", "I_N", "I_N_minus_L", "l_eoff_minus_s_eoff", "n_eoff_minus_l_eoff", "n_eoff_minus_s_eoff"):
        row = interaction["contrasts"][name]["all"]["f1_union_direct"]["10"]
        ci = row["ci95_percentile"]
        lines.append(f"| {name} | {100*row['point_delta']:+.4f} | [{100*ci[0]:+.4f}, {100*ci[1]:+.4f}] | {row['win_queries']} / {row['loss_queries']} / {row['tie_queries']} |")
    lines += ["", "交互使用 [full−Eoff] 差中差。10,000 次 paired source-group bootstrap 同步重采样同一 query 的各臂，保留 query-macro 分母。区间是反复使用 dev 的探索性结果，条件于固定湖、S0、Teacher、候选和单一 Student seed，不是等效性证明。全部三种规则、K=10/20/50 和 implicit/explicit 分层以及正例进入/退出见逐 query 统计。", "",
              "| arm | implicit known QE / QET | text / image QET | retained known pairs | distinct image / max hub |", "|---|---:|---:|---:|---:|"]
    for arm, payload in witnesses["arms"].items():
        row = payload["strata"]["implicit"]
        hub = payload["hubs"]["QE_image"]
        lines.append(f"| {arm} | {row['known_QE_unique_pairs']} / {row['known_QET_unique_pairs']} | {row['modality']['text']['known_QET_pair']} / {row['modality']['image']['known_QET_pair']} | {row['retained_known_unique_pairs']} | {hub['distinct_objects']} / {hub['highest_frequency_query_count']} |")
    lines += ["", "上述 QET 为 678 个 implicit 正对的去重计数，text/image 可重叠；路径出现次数、0–5 支持行、raw top path 和实际保留路径责任另有完整输出。L/N full 的证据分支退化比 Eoff 严重，但 Eoff 的 Q→T 依然失败，两个现象不能用一个根因代替。",
              "训练为全路径 raw(QE)+raw(ET) 的 LSE，真实温度τ=1；配置 top_k=4 对 plain logsumexp 不生效，本次固定64-query训练面板的袋最大含10条路径。实际交付 E2 为内容去重→Top20→greedy row coverage（最多4条），quality=sigmoid(QE+ET)，不使用 LSE。固定训练批的已知责任全为1来自这些已标注正袋仅包含 known witness，不能充当自然检索质量。", "",
              "## B13：已有路径在交付环节流失", "",
              f"自然 ANN D100 Recall 为 {100*candidate['metrics']['all']['ANN_D100_Recall']:.4f}%，完整 U Recall 为 {100*candidate['metrics']['all']['RawUnionRecall']:.4f}%，平均候选数 {candidate['metrics']['all']['mean_raw_union_size']:.3f}。213 个 ANN evidence-only 正对中207个位于 exact100 外、6个在内；另外2个 exact100 外正对被 ANN 偶然召回，故完整 exact evidence-only 为209对。",
              "预先定义的82对队列复现为79 query / 75 source groups，75对text、11对image（4对同时）；其中80对在 exact100 外。保留步骤丢掉7对的全部 known witness。RRF@50 交付45对，41对保留known witness，39对同时为exact新发现；F1@50交付2对且均在exact100内。",
              cases_note, "",
              "| admission | CandidateRecall@50 (%) | Stage1 Recall@10 (%) |", "|---|---:|---:|"]
    for rule, row in candidate["metrics"]["all"]["rules"].items():
        lines.append(f"| {rule} | {100*row['CandidateRecall@50']:.4f} | {100*row['recall@10']:.4f} |")
    for rule in ("pool_random", "lake_random"):
        row = candidate["random_controls"][rule]["all"]
        lines.append(f"| {rule}（100次均值） | {100*row['CandidateRecall@50_mean']:.4f} | 不定义随机准入后的排序主端点 |")
    lines += ["", "随机对照固定 RRF 保留的 D100 内成员与外部槽位 m_q，分别在 E\\D100 与合法湖 T\\D100 中均匀无放回抽取；m_q 不读标签，seed15预冻结，报告100次平均与Monte Carlo变化，不能称为Student重复。RRF的N50召回点估计提高1.1547pp，source-group 95%区间[-1.3010,+3.5744]pp跨零；Stage1 Top10下降3.2902pp，区间[-5.6236,-0.9464]pp。implicit/explicit tradeoff明确存在，最终效果须由相同Stage2在全query固定N50/K10测量。",
              "同一 direct score 下，exact D100∪E 的 TopK（K≤100）不能超过湖内 exact TopK；C4在1198 query上按统一tie-break复算通过。因此当前F1结构本身难以让真正的exact evidence-only候选进入前列。", "",
              "## 相对自身与冻结 B13 的发现数", "",
              "| arm | own ANN-new pairs | own exact-new pairs | B13 exact-new pairs | own exact-new known QET |", "|---|---:|---:|---:|---:|"]
    for arm, strata in summary["discovery"]["arms"].items():
        row = strata["all"]
        lines.append(f"| {arm} | {row['ann_evidence_only']['positive_pairs']} | {row['exact_evidence_only']['positive_pairs']} | {row['relative_B13_exact_new']['positive_pairs']} | {row['exact_new_known_QET_pairs']} |")
    lines += ["", "当某臂自身direct已坍塌时，own evidence-only数量可能膨胀；相对冻结B13的参照和原分母宏Recall同时保存。已知路径可达、候选qrel与行支持均不等于已恢复正确属性值。", "",
              "## 几何诊断与竞争解释", "",
              "| arm | type | residual/base RMS | full effective rank | random cosine p50 | F-vs-S0 cosine p50 |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["diagnostic_rows"]:
        if row["step"] != 178 or row.get("intervention") or row["arm"].startswith("s_"):
            continue
        for kind, geometry in row["geometry"]["by_type"].items():
            lines.append(f"| {row['arm']} | {kind} | {geometry['residual_to_base_rms']:.4f} | {geometry['full_output']['effective_rank']:.2f} | {geometry['full_output']['random_pair_cosine']['p50']:.4f} | {geometry['full_vs_s0_direction_cosine']['p50']:.4f} |")
    lines += ["", "完整 F、Pz、残差以及同一固定1024对象面板上的F/P/S0实际1024维均值向量均已记录；A/B更新范数与权重范数分开，Az与Az/c与GELU(Az/c)的RMS/分位数、实际优化器组与梯度位置、五关系margin/hub/谱均可复查。随机余弦使用每类固定前1024对象内seed13产生的512个不重叠随机配对，不是全湖随机抽样；旧日志的相邻配对另行保留。训练袋损失改善而全湖exact变差说明代理与部署检索不一致，尚不能唯一定位过拟合、负例、尺度或base-only anchor哪一项为根因。", "",
              "## 验证层次、成本与执行偏差", "",
              "代码/数值检查、指标重算、exact检索测量、统计支持和完整机制验证分别列于VALIDATION.json。前三类检查通过不代表模型有效；V的独立值/实体/属性/正确join字段为null。旧R12 human audit尚未独立审核，不能替代本轮验证，也不应用joinable=False删除候选或引入3/5行硬门槛。",
              "原GATE在I前通过了已训练forward/IP、exact与ANN以及线性合并检查，但train-fit、实际缓存向量、save→load与合并投影向量在I后补齐；完整G要求未全部先于I，执行顺序偏差保留。补充检查没有发现模型评分错误，因此保留既有356更新结果。原adapter-off几何日志误计禁用残差，已单独更正；原exact/ANN评分不受影响。",
              "历史R14 checkpoint没有保存Adam tensor state，无法恢复其实际moments。新臂保存真实state step/first-second moment摘要，其实际分组与当前共享构造器一致，但moment tensors未序列化。删除训练后新增的evaluate函数、相关imports/CLI参数及分派分支，恢复出15690字节R15训练入口，SHA256为a1a8b4f3fc3796a00d086928b2799644b3160f722b2e695c27a90e6a988811cc，精确匹配两臂训练manifest；8个训练/共享函数AST与当前入口一致。同样恢复24479字节R14入口，SHA256为da774e9fb48c8bdbcb4447f5063722f955e2a67582c2e764879868141f499a67，匹配S-Eoff/L-full/N-full的seed13训练manifest；另4个R14训练manifest匹配当前入口版本。R14的14个训练/共享函数（包括_optimizer）AST均一致，补充确立了历史优化器定义源码身份。两个恢复文件保存在source_snapshot/historical/，恢复过程未执行历史代码；原G记录的当前全文件hash不匹配仍保留，不回写历史日志。R15两臂本地导入闭包的18个CPython缓存均早于推导的训练开始时刻，header的mtime/size与现源码一致，且缓存code object与现源码重新编译结果逐项一致。R14三个seed13臂的17个本地依赖中也有15个满足同一条件；只有核心的mmdd_stage1.models与mmdd_stage1.training现存缓存生成于R14训练之后，故其执行时身份仍未验证。这些是同时代文件系统旁证，不是进程绑定的import trace。此证据确立manifest所记录的入口文件身份并缩小本地模块缺口，但不是当时同步保存的完整归档，也不恢复两个R14核心模块、第三方构建、Adam tensors或完整执行环境。候选与mask哈希仍为事后共享loader复算，顺序与历史R14哈希一致，不伪称执行时已有逐mask归档。",
              "GPU1被外部任务占用，本轮新训练和GPU审计使用GPU0并行。全部自然检索仍为43 query vectors/query，返回对象数、索引字节与构建时间、分项loss及执行成本可复查。并行评估延迟是吞吐条件下的读数，未作隔离延迟测试，不声称相同N50意味着相同计算预算。临时中间索引已删除以控制磁盘，checkpoint、索引manifest和全部测量保留，端点索引保留。", "",
              "## 后续决策与未回答的问题", "",
              "停止本轮投影扩容/激活/LR搜索和新seed追加。若另立后续残差实验，可预注册一个针对完整F的约束与原base-only anchor做单一控制比较；当前结果只提出该假说，没有测试它。候选准入应继续围绕exact新发现、实际证据保留与最终正确连接建立测量链。",
              "若单独启动V，先对固定82对队列做独立内容审核，再在全1198 query上比较相同Stage2、N50/K10以及无evidence/经审核错误evidence反事实。必须允许多种合法连接属性，并独立检查图像信息是否已在可见文本中。V未执行不抹去G/I/C结果，也不能据Stage1结果宣称完整多模态机制已经成立。", "",
              "复现入口与逐项证据见REPRODUCTION.md、COMPLETION_AUDIT.json和source_snapshot/MANIFEST.json。", ""]
    cost_lines = ["## 实测成本与算术边界", "",
                  "| new arm | train seconds | index build seconds | index GiB | eval online p50 / p95 (ms) |",
                  "|---|---:|---:|---:|---:|"]
    for arm, train in summary["costs"]["training"].items():
        evaluation = summary["costs"]["evaluation"][arm]
        cost_lines.append(f"| {arm} | {train['elapsed_seconds']:.3f} | {evaluation['index_build_seconds']:.3f} | {evaluation['index_bytes']/2**30:.3f} | {1000*evaluation['online_seconds_p50']:.3f} / {1000*evaluation['online_seconds_p95']:.3f} |")
    cost_lines += ["", "各臂训练/评估并行，时间不应相加成总墙钟；在线延迟含batch摊销，非隔离性能对比。新增两臂各178更新，0新Teacher推理。",
                   "每对象投影的乘加数：P为4096×1024=4,194,304 MAC，A/B残差增加4096×256+256×1024=1,310,720（31.25%）；这是投影算术，不是总推理延迟预测，且未计GELU。线性L合并后部署投影恢复4,194,304 MAC；N不具备该一般恒等式。索引仍为每对象1024维。", "",
                   "| random control | CR50 mean (%) | MC SD (pp) | MC 95% range (%) |", "|---|---:|---:|---:|"]
    for name in ("pool_random", "lake_random"):
        row = candidate["random_controls"][name]["all"]
        lower, upper = row["monte_carlo_95pct_range"]
        cost_lines.append(f"| {name} | {100*row['CandidateRecall@50_mean']:.4f} | {100*row['monte_carlo_sd']:.4f} | [{100*lower:.4f}, {100*upper:.4f}] |")
    cost_lines += ["", f"100次重复的随机采样与评分累计CPU墙钟为 {candidate['cost']['random_sampling_and_scoring_cpu_wall_seconds']:.3f}s，完整C审计 {candidate['cost']['wall_seconds']:.3f}s。Monte Carlo区间衡量随机准入变化，不能替代source-group置信区间；现成自然检索被复用，未计作新检索。", ""]
    position = lines.index("## 后续决策与未回答的问题")
    lines[position:position] = cost_lines
    return "\n".join(lines)


def run(root: Path, refresh_report_only: bool = False) -> None:
    output = root / "work/stage1_optimization_r15_20260909"
    saved_summary = read_json(output / "statistics/summary.json") if refresh_report_only else None
    if saved_summary is not None:
        diagnostics, timelines, discovery = (saved_summary[key] for key in ("diagnostic_rows", "timelines", "discovery"))
    else:
        endpoints, diagnostics, timelines = collect_audits(root, output)
        discovery = assemble_provenance(output, endpoints, root)
    interaction = read_json(output / "stageI_interaction/statistics/interaction_summary.json")
    candidate = read_json(output / "stageC_candidate_delivery/SUMMARY.json")
    witness = read_json(output / "stageI_interaction/statistics/witness_summary.json")
    training = read_json(output / "stageI_interaction/statistics/training_invariants.json")
    deployment = read_json(output / "stageG_correctness/deployment_supplement/SUMMARY.json")
    score_audit = read_json(output / "stageG_correctness/training_scores/summary.json")
    tests = read_json(output / "tests/verification.json")
    original_cases_path = output / "ORIGINAL_CASES_VALIDATION.json"
    original_cases = read_json(original_cases_path) if original_cases_path.is_file() else None
    runtime_cache_path = output / "RUNTIME_SOURCE_CACHE_AUDIT.json"
    runtime_cache = read_json(runtime_cache_path)
    r14_cache = runtime_cache["R14_seed13_dependency_cache_evidence"]
    if not (runtime_cache["passed"] is True and runtime_cache["local_module_count"] == 18
            and runtime_cache["all_timestamp_size_headers_match"] is True
            and runtime_cache["all_caches_precede_training_start"] is True
            and runtime_cache["all_cached_bytecode_matches_current_source"] is True
            and runtime_cache["historical_imported_module_runtime_identity_fully_verified"] is False
            and r14_cache["local_module_count"] == 17
            and r14_cache["pre_run_cache_supported_count"] == 15
            and r14_cache["post_run_cache_only_modules"]
            == ["mmdd_stage1.models", "mmdd_stage1.training"]
            and r14_cache["historical_imported_module_runtime_identity_fully_verified"] is False):
        raise ValueError("R15 runtime-source cache audit is incomplete or overclaims identity")
    if original_cases is not None:
        if not (original_cases["status"] == "passed" and original_cases["identity_verified"] is True
                and original_cases["semantic_cells_checked"] == original_cases["semantic_cells_matched"] == 2460
                and original_cases["original_column_count"] == len(original_cases["fields"]) == 30
                and not original_cases["mismatches"]
                and all(value == {"compared": 82, "mismatches": 0} for value in original_cases["fields"].values())):
            raise ValueError("Original-case comparison receipt is not fully passing")
        original_path = Path(original_cases["original"]["path"]).resolve()
        if original_path != root / "B13_evidence_only_known_witness_cases.csv" or dependency(original_path) != original_cases["original"]:
            raise ValueError("User-supplied original-case CSV fingerprint differs")
    metric_checks = [value for arm in interaction["metric_recomputation_pass"].values()
                     for rule in arm.values() for value in rule.values()]
    candidate_checks = [candidate["validation"][key] for key in (
        "saved_rankings_reproduced", "C4_exact_direct_invariant_holds", "all_213_exact_ranks_exported",
        "all_query_C50_sizes_equal_50", "frozen_82_reconstruction_count_matches",
    )]
    if not (deployment["passed"] and score_audit["passed"]
            and training["training_invariants_status"] == "passed"
            and tests["status"] == "passed" and tests["failed"] == 0
            and all(metric_checks) and all(candidate_checks)):
        raise ValueError("R15 numerical/training/metric/candidate/test evidence is not passing")
    costs = {"training": {}, "evaluation": interaction["cost"], "candidate_audit": candidate["cost"]}
    for arm in ("l_eoff", "n_eoff"):
        manifest = read_json(output / "stageI_interaction" / f"{arm}_seed13/manifest.json")
        costs["training"][arm] = {**manifest["cost"], "optimizer_updates": manifest["optimizer_updates"]}
    write_json(output / "statistics/cost_profiles.json", costs)
    summary = {"status": "G_I_C_computations_analyzed_archival_verification_incomplete",
               "strict_plan_completion_verified": False, "interaction": interaction,
               "candidates": candidate, "witnesses": witness, "discovery": discovery,
               "timelines": timelines, "diagnostic_rows": diagnostics,
               "training_invariants": training, "deployment_supplement": deployment,
               "training_score_audit": score_audit, "costs": costs, "V": "not_executed_conditional_budget",
               "runtime_source_cache_audit": runtime_cache,
               "selection": "retain frozen B13; no additional training arms or seeds",
               "completed_at_utc": saved_summary["completed_at_utc"] if saved_summary is not None else datetime.now(timezone.utc).isoformat()}
    if refresh_report_only:
        summary["report_refreshed_at_utc"] = datetime.now(timezone.utc).isoformat()
    if original_cases is not None:
        summary["original_cases_validation"] = original_cases
    write_json(output / "statistics/summary.json", summary)
    (output / "RESULTS.md").write_text(build_report(summary), encoding="utf-8")
    snapshot = snapshot_sources(root, output)
    validation = {
        "status": "share_with_caveats_archival_verification_incomplete",
        "strict_plan_completion_verified": False,
        "code_and_numerical_correctness": {
            "deployment_supplement_passed": deployment["passed"],
            "training_formula_passed": score_audit["passed"],
            "unit_tests": dependency(output / "tests/verification.json"),
            "adapter_off_diagnostic_error": "corrected; exact and ANN scores unaffected",
        },
        "metric_recomputation": {"per_query_ranking_recall_recomputed": True,
                                 "queries_per_arm": 1198, "arms": 6, "rules": list(RULES), "ks": list(KS),
                                 "saved_endpoint_metrics": interaction["metric_recomputation_pass"]},
        "exact_measurement": {"residual_checkpoints": [0, 45, 89, 178],
                              "five_relation_source_identity": True, "full_dev_direct": True,
                              "performance_passed": False, "reason": "Both residual Eoff endpoints collapse in exact retrieval."},
        "statistical_support": {"paired_source_group_bootstrap_completed": True,
                                "iterations": 10000, "exploratory_dev": True,
                                "supports_new_superior_full_method": False,
                                "interaction_intervals": {key: interaction["contrasts"][key]["all"]["f1_union_direct"]["10"] for key in ("I_L", "I_N", "I_N_minus_L")}},
        "complete_mechanism_verified": {"status": "unknown", "independent_value_attribute_entity_join": None,
                                        "V_executed": False, "reason": "No separate V budget or completed independent content reviews."},
        "historical_optimizer_tensor_state": {"available": False, "r14_parameter_groups_source_verified": True,
                                               "r14_source_evidence": "Hash-matched recovered R14 entrypoint and unchanged _optimizer AST; not optimizer runtime tensors",
                                               "r15_actual_groups_match_current_builder": True},
        "historical_R15_entrypoint": {"recorded_file_identity_verified": True,
                                       "method": "Post hoc reconstruction with exact SHA256 match to both training manifests",
                                       "evidence": "RECOVERED_TRAINING_SOURCE.json",
                                       "local_import_cache_evidence": dependency(runtime_cache_path),
                                       "local_import_cache_modules_verified": 18,
                                       "process_bound_import_trace_verified": False,
                                       "imported_modules_historical_identity_verified": False,
                                       "full_historical_execution_verified": False},
        "historical_R14_entrypoint": {"recorded_file_identity_verified": True,
                                       "method": "Post hoc SHA256-matched reconstruction for three manifests; four other manifests match current source",
                                       "evidence": "RECOVERED_TRAINING_SOURCE.json",
                                       "optimizer_definition_source_identity_verified": True,
                                       "local_import_cache_evidence": dependency(runtime_cache_path),
                                       "pre_run_cache_supported_modules": 15,
                                       "post_run_cache_only_modules": r14_cache["post_run_cache_only_modules"],
                                       "imported_modules_historical_identity_verified": False,
                                       "full_historical_execution_verified": False},
        "protocol_chronology": {"initial_G_before_I": True, "all_G_requirements_before_I": False,
                                "supplement_completed_after_I": True},
    }
    if original_cases is not None:
        validation["original_82_case_identity"] = {
            "status": "verified_original_semantic_identity", "checked_at_utc": original_cases["checked_at_utc"],
            "receipt": dependency(original_cases_path), "original": original_cases["original"],
            "unique_pairs": 82, "original_columns": 30, "matching_semantic_cells": 2460,
            "CSV_bytes_equal_to_reconstruction": False, "independent_value_join_verification": None,
            "historical_absence_statements_preserved": True,
        }
    write_json(output / "VALIDATION.json", validation)
    write_json(output / "stageG_correctness/FINAL_GATE.json", {
        "status": "passed_with_documented_history_limits", "initial_gate": dependency(output / "stageG_correctness/GATE.json"),
        "supplement": dependency(output / "stageG_correctness/deployment_supplement/SUMMARY.json"),
        "training_scores": dependency(output / "stageG_correctness/training_scores/summary.json"),
        "chronology": validation["protocol_chronology"],
        "optimizer_state": validation["historical_optimizer_tensor_state"],
    })
    required = ["RESULTS.md", "VALIDATION.json", "per_query_metrics.jsonl.gz", "candidate_provenance.jsonl.gz",
                "witness_funnel.jsonl.gz", "exact_relation_panels.json", "projection_and_optimizer_diagnostics.jsonl.gz",
                "source_snapshot/MANIFEST.json", "evidence_only_verification.jsonl.gz",
                "REPRODUCTION.md", "tests/verification.json", "statistics/cost_profiles.json",
                "stageG_correctness/mean_vectors.jsonl.gz", "stageG_correctness/mean_vectors_manifest.json",
                "RUNTIME_SOURCE_CACHE_AUDIT.json",
                "RECOVERED_TRAINING_SOURCE.json", "ARCHIVAL_RECOVERY_AUDIT.json",
                "source_snapshot/historical/run_stage1_r15.py", "source_snapshot/historical/run_stage1_r14.py"]
    if original_cases is not None:
        required.append("ORIGINAL_CASES_VALIDATION.json")
    rows = {name: len(read_gzip(output / name)) for name in required if name.endswith("jsonl.gz")}
    requirements = [
        ("G1 trained train-fit/dev score paths, actual cached vectors, save/load/c_tau", "verified", ["stageG_correctness/deployment_supplement/SUMMARY.json"]),
        ("G1 actual optimizer state and parameter groups", "partially_verified_definition_source_recovered_actual_historical_state_unavailable", ["projection_and_optimizer_diagnostics.jsonl.gz", "VALIDATION.json", "RECOVERED_TRAINING_SOURCE.json"]),
        ("G2 trained linear merge score/vector/exact identity", "verified", ["stageG_correctness/l_eoff/linear_merge_step178.json", "stageG_correctness/deployment_supplement/l_eoff_full.json"]),
        ("G3 exact/ANN full dev timelines and five fixed relation panels", "verified", ["exact_relation_panels.json"]),
        ("G4 F/base/residual geometry, actual F/P/S0 mean vectors, adapter-off, training formulas and responsibility", "verified_with_corrected_diagnostic", ["projection_and_optimizer_diagnostics.jsonl.gz", "stageG_correctness/mean_vectors.jsonl.gz", "stageG_correctness/training_scores/summary.json", "witness_funnel.jsonl.gz"]),
        ("I two exact-matched new Eoff cells, total356 updates, zero Teacher inference", "verified_with_archival_limits", ["stageI_interaction/statistics/training_invariants.json"]),
        ("I main/stratified metrics and required interactions, wins/losses/ties, entered/exited positives", "verified", ["stageI_interaction/statistics/interaction_summary.json", "per_query_metrics.jsonl.gz"]),
        ("I all endpoints natural witness counts, actual retention/rows, modality exact and hubs", "verified", ["witness_funnel.jsonl.gz", "exact_relation_panels.json", "stageI_interaction/statistics/witness_summary.json"]),
        ("C1 own and B13 ANN/exact D100, E/U/C50/Top10 provenance", "verified", ["candidate_provenance.jsonl.gz", "statistics/discovery_summary.json"]),
        ("C2 fixed213/82 queue, exact ranks, actual delivered witness and support",
         "verified_original_82_case_semantic_identity" if original_cases is not None else "verified_reconstructed_missing_csv",
         ["stageC_candidate_delivery/SUMMARY.json", "stageC_candidate_delivery/B13_evidence_only_known_witness_cases.csv"] + (["ORIGINAL_CASES_VALIDATION.json"] if original_cases is not None else [])),
        ("C3 three frozen rules and two fixed-slot random controls, all1198 N50", "verified", ["stageC_candidate_delivery/C_CONFIG_FROZEN.json", "stageC_candidate_delivery/SUMMARY.json", "stageC_candidate_delivery/random_control_draws.jsonl.gz"]),
        ("C4 same-score exact union boundary", "verified", ["stageC_candidate_delivery/SUMMARY.json"]),
        ("V independent attribute/value/entity/join and Stage2 full-query counterfactuals", "conditional_not_executed", ["evidence_only_verification.jsonl.gz", "VALIDATION.json"]),
        ("Frozen1024 single-object indexes,0/1hop,43vectors, no extra training grids", "verified", ["PLAN_FROZEN.json", "stageI_interaction/statistics/training_invariants.json"]),
        ("Actual G-before-I ordering", "documented_deviation", ["VALIDATION.json", "stageG_correctness/GATE.json", "stageG_correctness/FINAL_GATE.json"]),
        ("Key executable source snapshots", "verified_with_entrypoint_history_limit", ["source_snapshot/MANIFEST.json"]),
        ("R15 recorded historical training entrypoint identity", "verified_hash_matched_post_hoc_reconstruction", ["RECOVERED_TRAINING_SOURCE.json", "source_snapshot/historical/run_stage1_r15.py"]),
        ("R15 local imported-source cache evidence", "verified_contemporaneous_cache_not_process_trace", ["RUNTIME_SOURCE_CACHE_AUDIT.json"]),
        ("R14 recorded historical training entrypoint and optimizer-definition identity", "verified_hash_matched_post_hoc_reconstruction", ["RECOVERED_TRAINING_SOURCE.json", "source_snapshot/historical/run_stage1_r14.py", "ARCHIVAL_RECOVERY_AUDIT.json"]),
    ]
    for _, _, evidence in requirements:
        for relative in evidence:
            if not (output / relative).is_file():
                raise FileNotFoundError(output / relative)
    audit = {
        "status": "G_I_C_computations_analyzed_archival_verification_incomplete_V_conditional",
        "strict_plan_completion_verified": False,
        "requirements": [{"requirement": name, "status": status, "evidence": evidence} for name, status, evidence in requirements],
        "required_artifacts": {name: dependency(output / name) for name in required},
        "row_counts": rows, "source_files": len(snapshot["files"]),
        "unverified_archival_requirements": [
            "Historical R14 actual optimizer tensor state; entrypoint and optimizer-definition source identity are recovered",
            "R14 execution-time models.py/training.py, process-bound import traces, third-party builds, and full Adam moment tensors; 15/17 R14 and 18/18 R15 local-module pre-run caches match current source bytecode",
        ] + ([] if original_cases is not None else ["Original supplied82-case CSV individual IDs (deterministic reconstruction provided)"]),
        "irreversible_protocol_deviations": ["Some G supplements were completed after I training."],
        "scientific_mechanism_proven": False,
        "remaining_conditional_work": "Separately scoped/budgeted V with independent content audit and full-query N50/K10 Stage2.",
    }
    if original_cases is not None:
        audit["original_case_identity"] = validation["original_82_case_identity"]
        audit["required_external_inputs"] = {"original_82_case_CSV": original_cases["original"]}
    write_json(output / "COMPLETION_AUDIT.json", audit)
    print(json.dumps({"status": audit["status"], "rows": rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--refresh-report-only", action="store_true", help="Reuse saved scientific summaries without rewriting consolidated G/C artifacts")
    args = parser.parse_args()
    run(args.root.resolve(), args.refresh_report_only)

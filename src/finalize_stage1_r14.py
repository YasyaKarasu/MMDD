#!/usr/bin/env python
"""Finalize R14 Stage-1 attribution, repeats, selection, and report."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from finalize_stage1_r13 import _bootstrap, _source_map
from mmdd_stage1.artifacts import checkpoint_fingerprint, write_json
from run_stage1_r14 import freeze_plan


KS = (10, 20, 50)
RANKINGS = ("f1_union_direct", "union_rrf_equal", "pure_direct100")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_gz(path: Path) -> dict[str, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {str(row["query_id"]): row for row in map(json.loads, handle)}


def _r13_metrics(root: Path) -> dict[str, Path]:
    output = root / "work/stage1_optimization_r13_20260909"
    return {
        "s0": output / "taskA_stage1_protocol/s0/evaluation_step0/metrics.json",
        "c_s_shared": output / "taskC_role_projection/c_s_shared/evaluation_step178/metrics.json",
        "c_r_split": output / "taskC_role_projection/c_r_split/evaluation_step178/metrics.json",
        "p_s_target_only_seed13": output / "taskD_witness_supervision/p_s_target_only/evaluation_step178/metrics.json",
        "p_w_witness": output / "taskD_witness_supervision/p_w_witness/evaluation_step178/metrics.json",
        "b1_kd_on_hard356": output / "taskB_diagnostics_and_kd/b1_kd_on_hard356/evaluation_step356/metrics.json",
        "b1_kd_off_hard356": output / "taskB_diagnostics_and_kd/b1_kd_off_hard356/evaluation_step356/metrics.json",
    }


def _r14_metrics(root: Path) -> dict[str, Path]:
    output = root / "work/stage1_optimization_r14_20260909"
    return {
        "b_d_e_loss_off_seed13": output / "stage1_B_branch_ablation/b_d_e_loss_off_seed13/evaluation_step178/metrics.json",
        "m_l_linear_residual_seed13": output / "stage1_M_projection_capacity/m_l_linear_residual_seed13/evaluation_step178/metrics.json",
        "m_n_gelu_residual_seed13": output / "stage1_M_projection_capacity/m_n_gelu_residual_seed13/evaluation_step178/metrics.json",
        "m0_linear_seed17": output / "stage1_D_student_variance/m0_linear_seed17/evaluation_step178/metrics.json",
        "b_d_e_loss_off_seed17": output / "stage1_B_branch_ablation/b_d_e_loss_off_seed17/evaluation_step178/metrics.json",
        "m0_linear_seed23": output / "stage1_D_student_variance/m0_linear_seed23/evaluation_step178/metrics.json",
        "b_d_e_loss_off_seed23": output / "stage1_B_branch_ablation/b_d_e_loss_off_seed23/evaluation_step178/metrics.json",
    }


def _metric_row(arm: str, payload: dict[str, Any]) -> dict[str, Any]:
    primary = payload["primary"]
    is_r14 = arm.startswith(("b_d_", "m_l_", "m_n_", "m0_"))
    row = {
        "arm": arm,
        "seed": payload.get("seed", 13),
        "step": payload.get("step", 0),
        "known_valid_path_R50": primary["valid_path@50,4"],
        "search_vectors_per_query_max": payload["cost"]["search_vectors_per_query_max"],
        "index_build_seconds": payload["cost"]["index_build_seconds"],
        "online_seconds_p50": payload["cost"]["online_seconds_p50"],
        "online_seconds_p95": payload["cost"]["online_seconds_p95"],
        "evaluation_concurrency_note": (
            "concurrent throughput run; not an isolated latency guard"
            if is_r14
            else None
        ),
    }
    for k in KS:
        row.update(
            {
                f"all_R{k}": primary[f"recall@{k}"],
                f"implicit_R{k}": primary[f"implicit_recall@{k}"],
                f"explicit_R{k}": primary[f"explicit_recall@{k}"],
                f"single_positive_R{k}": primary[f"single_positive_recall@{k}"],
                f"multiple_positive_R{k}": primary[
                    f"multiple_positive_recall@{k}"
                ],
                f"pure_direct_R{k}": payload["pure_direct100"][f"recall@{k}"],
                f"RRF_R{k}": payload["sensitivity_equal_union_rrf"][
                    f"recall@{k}"
                ],
            }
        )
    row["candidate_R50"] = primary["recall@50"]
    return row


def _mechanism(path: Path) -> dict[str, Any]:
    records = _read_jsonl_gz(path)
    qe_pairs = qet_pairs = text_qet_pairs = image_qet_pairs = denominator = 0
    image_queries: Counter[str] = Counter()
    all_images = set()
    for record in records.values():
        images_for_query = set()
        evidence_type = {}
        qe_ids = set()
        for paths in record["paths_by_target"].values():
            for path_row in paths:
                if path_row.get("kind") != "evidence":
                    continue
                evidence_id = str(path_row["evidence_id"])
                qe_ids.add(evidence_id)
                evidence_type[evidence_id] = path_row.get("evidence_type")
                if path_row.get("evidence_type") == "image":
                    images_for_query.add(evidence_id)
        all_images.update(images_for_query)
        image_queries.update(images_for_query)
        if record.get("query_kind") != "implicit":
            continue
        known_by_target = record.get("positive_evidence_by_target", {})
        for target_id in record["positive_target_ids"]:
            known = {str(value) for value in known_by_target.get(target_id, [])}
            denominator += 1
            qe_pairs += int(bool(known & qe_ids))
            target_evidence = {
                str(path_row["evidence_id"])
                for path_row in record["paths_by_target"].get(target_id, [])
                if path_row.get("kind") == "evidence"
            }
            supported = known & target_evidence
            qet_pairs += int(bool(supported))
            text_qet_pairs += int(
                any(evidence_type.get(value) == "text" for value in supported)
            )
            image_qet_pairs += int(
                any(evidence_type.get(value) == "image" for value in supported)
            )
    most_common = image_queries.most_common(1)
    return {
        "implicit_positive_pairs": denominator,
        "known_QE_pairs": qe_pairs,
        "known_QET_pairs": qet_pairs,
        "known_text_QET_pairs": text_qet_pairs,
        "known_image_QET_pairs": image_qet_pairs,
        "known_QET_coverage": qet_pairs / denominator if denominator else None,
        "distinct_retrieved_images": len(all_images),
        "highest_frequency_image": most_common[0][0] if most_common else None,
        "highest_frequency_image_query_count": most_common[0][1] if most_common else 0,
    }


def _oracle_and_contributions(path: Path) -> dict[str, Any]:
    rows = _read_jsonl_gz(path)
    result = {}
    for k in KS:
        recall = []
        oracle_pool = []
        rescued = []
        displaced = []
        for row in rows.values():
            positives = set(row["positive_target_ids"])
            denominator = row["positive_denominator"]
            f1 = row["rankings"]["f1_union_direct"][str(k)]
            pure = row["rankings"]["pure_direct100"][str(k)]
            pool50 = set(row["rankings"]["f1_union_direct"]["50"]["target_ids"])
            recall.append(f1["recall"])
            oracle_pool.append(min(k, len(positives & pool50)) / denominator)
            rescued.append(
                len(positives & (set(f1["target_ids"]) - set(pure["target_ids"])))
                / denominator
            )
            displaced.append(
                len(positives & (set(pure["target_ids"]) - set(f1["target_ids"])))
                / denominator
            )
        actual = statistics.fmean(recall)
        pool = statistics.fmean(oracle_pool)
        result[str(k)] = {
            "actual_recall": actual,
            "known_label_pool_oracle": pool,
            "all_target_capacity_oracle": 1.0,
            "candidate_gap": 1.0 - pool,
            "ranking_gap": pool - actual,
            "weighted_rescued_f1_minus_pure": statistics.fmean(rescued),
            "weighted_displaced_f1_minus_pure": statistics.fmean(displaced),
        }
    return result


def _paired_statistics(
    metric_payloads: dict[str, dict[str, Any]],
    source_by_query: dict[str, str],
    output: Path,
) -> dict[str, Any]:
    comparisons = {
        "d_minus_full_seed13": ("b_d_e_loss_off_seed13", "p_s_target_only_seed13"),
        "d_minus_full_seed17": ("b_d_e_loss_off_seed17", "m0_linear_seed17"),
        "d_minus_full_seed23": ("b_d_e_loss_off_seed23", "m0_linear_seed23"),
    }
    needed = {value for pair in comparisons.values() for value in pair}
    rankings = {
        arm: _read_jsonl_gz(Path(payload["rankings"]["path"]))
        for arm, payload in metric_payloads.items()
        if arm in needed
    }
    per_query_path = output / "statistics/per_query_metrics.jsonl.gz"
    summaries = {}
    all_rows = []
    for comparison, (experimental, control) in comparisons.items():
        left = rankings[experimental]
        right = rankings[control]
        if left.keys() != right.keys():
            raise ValueError(f"Query IDs differ for {comparison}")
        rows = []
        for query_id in sorted(left):
            row = {
                "comparison": comparison,
                "experimental": experimental,
                "control": control,
                "query_id": query_id,
                "source_table_id": source_by_query[query_id],
                "query_kind": left[query_id]["query_kind"],
                "positive_denominator": left[query_id]["positive_denominator"],
                "deltas": {},
            }
            for ranking in RANKINGS:
                row["deltas"][ranking] = {
                    str(k): (
                        left[query_id]["rankings"][ranking][str(k)]["recall"]
                        - right[query_id]["rankings"][ranking][str(k)]["recall"]
                    )
                    for k in KS
                }
            rows.append(row)
        all_rows.extend(rows)
        summaries[comparison] = {}
        for kind in ("all", "implicit", "explicit"):
            selected = [
                row
                for row in rows
                if kind == "all" or row["query_kind"] == kind
            ]
            summaries[comparison][kind] = {}
            for ranking in RANKINGS:
                summaries[comparison][kind][ranking] = {}
                for k in KS:
                    deltas = np.asarray(
                        [row["deltas"][ranking][str(k)] for row in selected],
                        dtype=np.float64,
                    )
                    summaries[comparison][kind][ranking][str(k)] = _bootstrap(
                        deltas,
                        [row["source_table_id"] for row in selected],
                    )
    with gzip.open(per_query_path, "wt", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    return {
        "format_version": 1,
        "status": "complete",
        "unit": "source_table_id",
        "estimand": "query-macro recall delta",
        "iterations": 10_000,
        "seed": 13,
        "comparisons": summaries,
        "per_query": {
            "path": str(per_query_path.resolve()),
            "sha256": checkpoint_fingerprint(per_query_path),
        },
        "exploratory_dev_intervals": True,
    }


def _training_manifest(root: Path, arm: str) -> Path | None:
    output = root / "work/stage1_optimization_r14_20260909"
    matches = list(output.glob(f"stage1_*/*{arm}/manifest.json"))
    return matches[0] if matches else None


def _report(
    rows: list[dict[str, Any]],
    matrix: dict[str, Any],
    bootstrap: dict[str, Any],
    mechanisms: dict[str, Any],
    exact: dict[str, Any],
    oracles: dict[str, Any],
    relation_panel: dict[str, Any],
    historical_test: dict[str, Any],
) -> str:
    by_arm = {row["arm"]: row for row in rows}
    shown = [
        "s0",
        "c_s_shared",
        "c_r_split",
        "p_s_target_only_seed13",
        "p_w_witness",
        "b1_kd_on_hard356",
        "b1_kd_off_hard356",
        "b_d_e_loss_off_seed13",
        "m_l_linear_residual_seed13",
        "m_n_gelu_residual_seed13",
        "m0_linear_seed17",
        "b_d_e_loss_off_seed17",
        "m0_linear_seed23",
        "b_d_e_loss_off_seed23",
    ]
    lines = [
        "# R14 Stage-1 results",
        "",
        "日期：2026-09-09。状态：**完成**。",
        "",
        "## 结论",
        "",
        (
            "R13 的 +1.4399pp 主要来自 direct score 改变：对称四格分解在 "
            f"R@10 给出 score {100*matrix['delta_score']:+.4f}pp、"
            f"pool {100*matrix['delta_pool']:+.4f}pp。"
        ),
        (
            "E-loss-off 在三个 Student seed 都略高于完整 path 控制，但幅度仅为 "
            "+0.0835/+0.2504/+0.1252pp；这是方向一致的小差值，不是等效性证明，"
            "也不证明 S0 的历史训练不需要 evidence。"
        ),
        (
            "两层残差没有改善容量：M-L 的 R@10 为 0，M-N 为 0.1669%。"
            "训练代理改善而全湖检索坍塌，故不做事后调参或多 seed 扩张。"
        ),
        "没有新的完整主方法超过 B13；部署选择保留 p_s_target_only@178。",
        "",
        "## 固定 dev 端点",
        "",
        "| arm | seed | all@10/20/50 | implicit@10/20/50 | explicit@10/20/50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in shown:
        row = by_arm[arm]
        lines.append(
            f"| {arm} | {row['seed']} | "
            f"{100*row['all_R10']:.4f}/{100*row['all_R20']:.4f}/{100*row['all_R50']:.4f} | "
            f"{100*row['implicit_R10']:.4f}/{100*row['implicit_R20']:.4f}/{100*row['implicit_R50']:.4f} | "
            f"{100*row['explicit_R10']:.4f}/{100*row['explicit_R20']:.4f}/{100*row['explicit_R50']:.4f} |"
        )
    lines.extend(
        [
            "",
            "数值均为百分数。固定F1使用D1保留与union-direct排序。",
            "",
            "| arm | pure-direct@10/20/50 | equal-RRF@10/20/50 | valid path@50,4 | vectors/query |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in shown:
        row = by_arm[arm]
        lines.append(
            f"| {arm} | {100*row['pure_direct_R10']:.4f}/{100*row['pure_direct_R20']:.4f}/{100*row['pure_direct_R50']:.4f} | "
            f"{100*row['RRF_R10']:.4f}/{100*row['RRF_R20']:.4f}/{100*row['RRF_R50']:.4f} | "
            f"{100*row['known_valid_path_R50']:.4f} | {row['search_vectors_per_query_max']} |"
        )
    lines.extend(
        [
            "",
            "## F1 与 pure-direct 的已知正例交换",
            "",
            "| arm | k | rescued | displaced | candidate gap | ranking gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in shown:
        for k in KS:
            item = oracles[arm][str(k)]
            lines.append(
                f"| {arm} | {k} | {100*item['weighted_rescued_f1_minus_pure']:.4f} | "
                f"{100*item['weighted_displaced_f1_minus_pure']:.4f} | "
                f"{100*item['candidate_gap']:.4f} | {100*item['ranking_gap']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## E-loss-off 配对结果",
            "",
            "| seed | D−F R@10 | source-cluster 95% CI | implicit差 | explicit差 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in (13, 17, 23):
        comparison = bootstrap["comparisons"][f"d_minus_full_seed{seed}"]
        overall = comparison["all"]["f1_union_direct"]["10"]
        implicit = comparison["implicit"]["f1_union_direct"]["10"]["point_delta"]
        explicit = comparison["explicit"]["f1_union_direct"]["10"]["point_delta"]
        lines.append(
            f"| {seed} | {100*overall['point_delta']:+.4f} | "
            f"[{100*overall['ci95_percentile'][0]:+.4f}, "
            f"{100*overall['ci95_percentile'][1]:+.4f}] | "
            f"{100*implicit:+.4f} | {100*explicit:+.4f} |"
        )
    lines.extend(
        [
            "",
            "三 seed 的 D−F 均为正，但都远小于预声明的 +0.5pp 新配方投入阈值；"
            "D 仍是机制消融，不替换主方法。",
            "M-L/M-N 相对同 seed13 M0(B13) 的 R@10 差分别为 "
            f"{100*(by_arm['m_l_linear_residual_seed13']['all_R10'] - by_arm['p_s_target_only_seed13']['all_R10']):+.4f}pp 与 "
            f"{100*(by_arm['m_n_gelu_residual_seed13']['all_R10'] - by_arm['p_s_target_only_seed13']['all_R10']):+.4f}pp。",
            "",
            "## exact 与 F1 边界",
            "",
            (
                "S0/B13 的 exact-D100∪E 按同一 direct score 排序，在 1,198 个 "
                "query 的 @10/@20/@50 全部与 exact direct 前缀相同。"
            ),
            (
                "ANN 与 exact top10 的平均集合重合率为 "
                f"{100*exact['models']['s0']['checks']['10']['ann_exact_overlap_rate']:.3f}%"
                "（S0）和 "
                f"{100*exact['models']['p_s_target_only']['checks']['10']['ann_exact_overlap_rate']:.3f}%（B13）。"
            ),
            "自然榜单重算在@10完全复现；@20/@50分别有S0 1/8、B13 0/3个query受浮点重算或同分边界影响。",
            "",
            "R13 固定机制面板的五关系 exact/ANN（宏平均正例召回；括号内为top-k集合重合率）：",
            "",
            "| relation | S0 ANN/exact (overlap) | B13 ANN/exact (overlap) |",
            "|---|---:|---:|",
        ]
    )
    for relation in relation_panel["s0"]:
        left = relation_panel["s0"][relation]
        right = relation_panel["p_s_target_only"][relation]
        lines.append(
            f"| {relation} | {100*left['ann_positive_recall_macro']:.3f}/"
            f"{100*left['exact_positive_recall_macro']:.3f} ({100*left['topk_overlap']:.3f}) | "
            f"{100*right['ann_positive_recall_macro']:.3f}/"
            f"{100*right['exact_positive_recall_macro']:.3f} ({100*right['topk_overlap']:.3f}) |"
        )
    lines.extend(
        [
            "",
            "## evidence 机制",
            "",
            "| arm | known QE | known QET | text QET | image QET | distinct images | top image queries |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in shown:
        item = mechanisms[arm]
        lines.append(
            f"| {arm} | {item['known_QE_pairs']} | {item['known_QET_pairs']} | "
            f"{item['known_text_QET_pairs']} | {item['known_image_QET_pairs']} | "
            f"{item['distinct_retrieved_images']} | "
            f"{item['highest_frequency_image_query_count']} |"
        )
    lines.extend(
        [
            "",
            "known QET 的分母为 678 个 implicit 正对；text/image 可重叠。",
            "已知支持只是 qrels 下界，没有独立错误属性标签的字段继续为 null。",
            "训练W资格实际为5,707个source、6,347个正对、14,098个row group；"
            "其作用域是any_known_witness_by_row，不代表存在独立错误属性标签。",
            "",
            "## 选择、成本与限制",
            "",
            "- C-G 未触发：R13 B0 对 path-direct/path-evidence 各只有一个 fresh-AdamW 更新 batch，且没有保存 QE/ET margin 反事实。",
            "- C-RF 未触发：旧记录没有当前 S0 top20 unknown 与旧 hard 列表的预注册重合/排名证据。",
            "- R14 评测为并发吞吐运行，单次 P95 不用于部署成本护栏；新臂均未形成完整候选，故不追加隔离延迟测量。",
            "- R13 C-R 未进 eligible arms，是因为保存 P95 90.45ms 超过 S0 的 1.10 倍护栏；这不是模型内在延迟的普遍结论。",
            "- dev 已被多轮使用，bootstrap 区间是探索性的；三个 seed 只反映固定 S0/Teacher/候选下的 Student 排列方差。",
            "- 历史 test 不参与 R14 选模；最终配方未变，沿用冻结 B13 的既有历史回归读数。",
            f"- 历史test身份：{historical_test['queries']} queries；B13 R@10/20/50="
            f"{100*historical_test['recall@10']:.4f}/{100*historical_test['recall@20']:.4f}/"
            f"{100*historical_test['recall@50']:.4f}，metrics SHA256="
            f"`{historical_test['metrics']['sha256']}`。",
            "",
            "详细可复算数据见 stage1_A_attribution/ 与 statistics/。",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(args: argparse.Namespace) -> None:
    plan = freeze_plan(args.root)
    output = args.root / "work/stage1_optimization_r14_20260909"
    metrics_paths = {**_r13_metrics(args.root), **_r14_metrics(args.root)}
    for path in metrics_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = {arm: _read_json(path) for arm, path in metrics_paths.items()}
    rows = [_metric_row(arm, payload) for arm, payload in metrics.items()]
    mechanisms = {
        arm: _mechanism(Path(payload["path_pool"]["path"]))
        for arm, payload in metrics.items()
    }
    oracles = {
        arm: _oracle_and_contributions(Path(payload["rankings"]["path"]))
        for arm, payload in metrics.items()
    }
    write_json(
        output / "stage1_A_attribution/direct_vs_union_vs_rrf.json",
        {
            "format_version": 1,
            "status": "complete",
            "rows": rows,
            "definitions": {
                "f1": "D1 pool retention followed by union-direct ranking",
                "pure_direct": "ANN D100 reranked by the same raw direct score",
                "rrf": "equal union RRF60 on the same natural pool",
            },
        },
    )
    write_json(
        output / "stage1_A_attribution/known_label_oracle_gaps.json",
        {
            "format_version": 1,
            "status": "complete",
            "arms": oracles,
            "qrels_are_incomplete": True,
        },
    )
    write_json(
        output / "statistics/mechanism_table.json",
        {
            "format_version": 1,
            "status": "complete",
            "arms": mechanisms,
            "wrong_attribute_rate": None,
            "wrong_entity_rate": None,
            "reason_for_null": "No independent scoped negative labels exist.",
        },
    )
    source_by_query, dataset_manifest = _source_map(args.root)
    bootstrap = _paired_statistics(metrics, source_by_query, output)
    write_json(output / "statistics/bootstrap.json", bootstrap)
    seed_deltas = [
        bootstrap["comparisons"][f"d_minus_full_seed{seed}"]["all"][
            "f1_union_direct"
        ]["10"]["point_delta"]
        for seed in (13, 17, 23)
    ]
    exact = _read_json(
        output / "stage1_A_attribution/exact_direct_boundary.json"
    )
    cross_matrix = _read_json(
        output / "stage1_A_attribution/pool_score_cross_matrix.json"
    )
    summary = {
        "format_version": 1,
        "status": "complete",
        "rows": rows,
        "pool_score_cross_matrix": cross_matrix["aggregate"],
        "exact_direct_boundary": exact,
        "d_minus_full_seed_deltas": seed_deltas,
        "d_minus_full_seed_mean": statistics.fmean(seed_deltas),
        "d_minus_full_seed_stdev": statistics.stdev(seed_deltas),
        "selection": "retain R13 p_s_target_only@178",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "statistics/summary.json", summary)
    train_costs = {}
    for arm in _r14_metrics(args.root):
        manifest = _training_manifest(args.root, arm)
        if manifest is not None:
            payload = _read_json(manifest)
            train_costs[arm] = {
                "manifest": str(manifest.resolve()),
                "manifest_sha256": checkpoint_fingerprint(manifest),
                "elapsed_seconds": payload["cost"]["elapsed_seconds"],
                "processed": payload["processed"],
                "new_teacher_inference": 0,
            }
    write_json(
        output / "statistics/cost_profiles.json",
        {
            "format_version": 1,
            "status": "complete",
            "training": train_costs,
            "evaluation": {
                arm: payload["cost"] for arm, payload in metrics.items()
            },
            "r14_evaluations_are_concurrent_throughput_runs": True,
        },
    )
    r13_root = args.root / "work/stage1_optimization_r13_20260909"
    r13_summary = _read_json(r13_root / "statistics/summary.json")
    r13_mechanism = _read_json(
        r13_root / "statistics/mechanism_and_reproducibility_audit.json"
    )
    relation_panel = r13_mechanism["mechanism_panel"]["exact_and_ann_by_arm"]
    c_r = next(row for row in r13_summary["rows"] if row["arm"] == "c_r_split")
    s0 = next(row for row in r13_summary["rows"] if row["arm"] == "s0")
    dependency = {
        "format_version": 1,
        "status": "complete",
        "data": {
            "dev_queries": 1198,
            "implicit_queries": 599,
            "explicit_queries": 599,
            "source_groups": 1000,
            "known_query_target_pairs": 1279,
            "positive_multiplicity": {"1": 1127, "2": 63, "3": 6, "4": 2},
            "dataset_manifest": str(dataset_manifest.resolve()),
            "dataset_manifest_sha256": checkpoint_fingerprint(dataset_manifest),
        },
        "r13_detailed_artifacts": {
            "summary": str((r13_root / "statistics/summary.json").resolve()),
            "bootstrap": str((r13_root / "statistics/bootstrap.json").resolve()),
            "mechanism": str(
                (
                    r13_root
                    / "statistics/mechanism_and_reproducibility_audit.json"
                ).resolve()
            ),
            "b0": str(
                (
                    r13_root
                    / "taskB_diagnostics_and_kd/b0_one_step_and_exact.json"
                ).resolve()
            ),
            "candidate_audit": str(
                (
                    r13_root
                    / "taskB_diagnostics_and_kd/candidate_audit.json"
                ).resolve()
            ),
        },
        "c_r_eligibility": {
            "eligible": False,
            "quality_guard_passed": True,
            "cost_guard_passed": False,
            "saved_p95_seconds": c_r["online_seconds_p95"],
            "s0_p95_seconds": s0["online_seconds_p95"],
            "threshold_seconds": 1.1 * s0["online_seconds_p95"],
            "basis": "R13 finalizer actual eligible-arm predicate",
        },
        "b0_readout": {
            "actual_independent_path_updates_per_branch": 1,
            "path_update_queries": 64,
            "path_holdout_queries": 448,
            "direct_P_table_gradient_norm": 0.132701,
            "evidence_P_table_gradient_norm": 0.258539,
            "qe_et_margin_counterfactual_saved": False,
            "optimizer": "fresh AdamW, not historical optimizer replay",
        },
        "w_readout": {
            "eligible_queries": 5707,
            "eligible_positive_pairs": 6347,
            "eligible_row_groups": 14098,
            "scope": "any_known_witness_by_row",
            "independent_wrong_attribute_labels": 0,
        },
        "conditional_decisions": plan["amendment"],
    }
    write_json(output / "R13_DEPENDENCY_READOUT.json", dependency)
    write_json(
        output / "stage1_C_one_optimization/NOT_TRIGGERED.json",
        {
            "format_version": 1,
            "status": "not_triggered",
            "c_g": plan["amendment"]["conditional_c_g"],
            "c_rf": plan["amendment"]["conditional_c_rf"],
            "replacement_first_batch": plan["amendment"]["first_batch"],
            "basis": "GPT-6 Pro round-2 amendment frozen before R14 training",
        },
    )
    decisions = _read_json(output / "TASK_DECISIONS.json")
    decisions.update(
        {
            "status": "complete",
            "task_d": "completed matched full-vs-E-loss-off repeats at seeds 17/23",
            "task_d_outcome": (
                "D-F was positive for seeds 13/17/23 but below the registered "
                "+0.5pp threshold; retain D as an ablation"
            ),
            "selection": "retain R13 p_s_target_only@178",
        }
    )
    write_json(output / "TASK_DECISIONS.json", decisions)
    selected = plan["r13_dependencies"]["selected_recipe"]
    write_json(
        output / "SELECTED_RECIPE.json",
        {
            "format_version": 1,
            "status": "frozen",
            "selected_arm": "p_s_target_only",
            "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "decision": "retained R13 B13; no R14 full-method candidate exceeded it",
            "d_ablation_not_eligible_for_automatic_deployment": True,
            "m_l_reason": "full-dev retrieval collapse",
            "m_n_reason": "full-dev retrieval collapse",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    deliverables = [
        output / "PLAN_FROZEN.json",
        output / "R13_DEPENDENCY_READOUT.json",
        output / "TASK_DECISIONS.json",
        output / "runs.jsonl",
        output / "stage1_A_attribution/direct_vs_union_vs_rrf.json",
        output / "stage1_A_attribution/exact_direct_boundary.json",
        output / "stage1_A_attribution/pool_score_cross_matrix.json",
        output / "stage1_A_attribution/query_level_contributions.jsonl.gz",
        output / "stage1_A_attribution/known_label_oracle_gaps.json",
        output / "stage1_C_one_optimization/NOT_TRIGGERED.json",
        output / "statistics/summary.json",
        output / "statistics/bootstrap.json",
        output / "statistics/per_query_metrics.jsonl.gz",
        output / "statistics/mechanism_table.json",
        output / "statistics/cost_profiles.json",
        output / "SELECTED_RECIPE.json",
    ]
    report = _report(
        rows,
        cross_matrix["aggregate"]["all"]["10"],
        bootstrap,
        mechanisms,
        exact,
        oracles,
        relation_panel,
        r13_summary["historical_test"],
    )
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    audit = {
        "format_version": 1,
        "status": "complete",
        "task_a": "complete",
        "task_b": "complete",
        "task_c_g": "not_triggered",
        "task_c_rf": "not_triggered",
        "task_m": "complete",
        "task_d": "complete",
        "training_updates_new": 1246,
        "new_teacher_inference": 0,
        "selected_recipe_unchanged": True,
        "required_artifacts": [str(path.resolve()) for path in deliverables],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
    }
    write_json(output / "COMPLETION_AUDIT.json", audit)
    audit_lines = [
        "# R14 completion audit",
        "",
        "- Status: complete",
        "- Task A: complete",
        "- Task B: complete",
        "- Task C-G/C-RF: not triggered under the frozen amendment",
        "- Task M: complete",
        "- Task D: complete",
        "- New Student updates: 1,246",
        "- New Teacher inference: 0",
        "- Selected recipe: unchanged (R13 p_s_target_only@178)",
        "",
        "Every machine-readable deliverable is hash-listed in comparison_manifest.json.",
        "Test validation is recorded separately in VALIDATION.json after the suite runs.",
        "",
    ]
    (output / "COMPLETION_AUDIT.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )
    published = [
        *deliverables,
        output / "RESULTS.md",
        output / "COMPLETION_AUDIT.json",
        output / "COMPLETION_AUDIT.md",
        output / "VALIDATION.json",
    ]
    missing_published = [str(path) for path in published if not path.is_file()]
    if missing_published:
        raise FileNotFoundError(
            "Missing final R14 artifacts: " + ", ".join(missing_published)
        )
    write_json(
        output / "comparison_manifest.json",
        {
            "format_version": 1,
            "status": "complete",
            "artifacts": [
                {
                    "path": str(path.resolve()),
                    "sha256": checkpoint_fingerprint(path),
                    "bytes": path.stat().st_size,
                }
                for path in published
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected": "p_s_target_only",
                "d_minus_full_seed_deltas": seed_deltas,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.root = arguments.root.resolve()
    finalize(arguments)

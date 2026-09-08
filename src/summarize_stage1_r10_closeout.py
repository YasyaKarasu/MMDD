#!/usr/bin/env python
"""Build the final R10 Stage-1 report from frozen evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mmdd_dataset.wdc_runtime import iter_dataset_artifact
from mmdd_stage1.retrieval import checkpoint_fingerprint


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def query_groups(root: Path) -> dict[str, str]:
    return {
        str(record["table_id"]): str(record["source_table_id"])
        for record in iter_dataset_artifact(root, "query_tables")
    }


def result_summary(payload: dict[str, Any], rule: str) -> dict[str, Any]:
    result = payload["results"][rule]
    rows = result["per_query"]
    return {
        "queries": result["queries"],
        "recall_at_10": result["recall@10"],
        "recall_at_20": result["recall@20"],
        "recall_at_50": result["recall@50"],
        "mrr_at_50": result["mrr@50"],
        "valid_path_at_10_4": result["valid_path"]["valid_path_recall@10,4"]["value"],
        "row_support_at_10_4": result["row_support"]["row_support_coverage@10,4"],
        "implicit_queries": sum(bool(row.get("implicit_pair_count")) for row in rows),
        "implicit_recall_at_10": result.get("implicit", {}).get("recall@10"),
        "explicit_recall_at_10": result.get("explicit", {}).get("recall@10"),
        "explicit_queries": result.get("explicit", {}).get("queries"),
        "implicit_pairs": sum(row["implicit_pair_count"] for row in rows),
        "attribution_at_10": {key: value for key, value in result["attribution"].items() if key.endswith("@10")},
        "multi_positive": {
            "queries": sum(row["positive_target_count"] > 1 for row in rows),
            "recall_at_10": statistics.fmean(row["recall@10"] for row in rows if row["positive_target_count"] > 1),
        },
    }


def bootstrap_delta(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    groups: dict[str, str],
    field: str,
    *,
    weight_field: str | None = None,
    samples: int = 10000,
    seed: int = 13,
) -> dict[str, float | int]:
    if not left or left.keys() != right.keys() or set(left) - groups.keys():
        raise ValueError("Paired bootstrap requires the same complete query population")
    keys = sorted(left)
    by_group: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_group[groups[key]].append(key)
    group_names = sorted(by_group)

    # Resample source tables, retaining all queries and their original pair weights.
    numerators = np.zeros(len(group_names), dtype=np.float64)
    denominators = np.zeros(len(group_names), dtype=np.float64)
    for index, group in enumerate(group_names):
        for key in by_group[group]:
            weight = float(left[key][weight_field]) if weight_field else 1.0
            if weight_field and weight != float(right[key][weight_field]):
                raise ValueError("Paired metrics have unequal denominators")
            numerators[index] += (float(left[key][field]) - float(right[key][field])) * weight
            denominators[index] += weight
    if denominators.sum() == 0:
        raise ValueError("Metric has no eligible positive pairs")
    observed = float(numerators.sum() / denominators.sum())
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, samples, 256):
        selected = rng.integers(len(group_names), size=(min(256, samples - start), len(group_names)))
        denominator = denominators[selected].sum(axis=1)
        # A resample with no eligible pairs has an undefined metric, not a zero.
        valid = denominator > 0
        estimates.extend((numerators[selected].sum(axis=1)[valid] / denominator[valid]).tolist())
    estimates.sort()
    return {
        "groups": len(group_names),
        "eligible_groups": int(np.count_nonzero(denominators)),
        "queries": len(keys),
        "samples": samples,
        "seed": seed,
        "valid_replicates": len(estimates),
        "mean_difference": observed,
        "ci95_low": estimates[int(0.025 * (len(estimates) - 1))],
        "ci95_high": estimates[int(0.975 * (len(estimates) - 1))],
    }


def validate_aggregates(payload: dict[str, Any]) -> None:
    for rule, result in payload["results"].items():
        rows = result["per_query"]
        if len(rows) != result["queries"] or len({row["query_id"] for row in rows}) != len(rows):
            raise ValueError(f"{rule}: duplicated or missing query records")
        for k in (10, 20, 50):
            observed = statistics.fmean(row[f"recall@{k}"] for row in rows)
            if abs(observed - result[f"recall@{k}"]) > 1e-10:
                raise ValueError(f"{rule}: inconsistent Recall@{k}")
            denominator = sum(row["implicit_pair_count"] for row in rows)
            for metric, bucket in (("valid_path_recall", "valid_path"), ("row_support_coverage", "row_support")):
                key = f"{metric}@{k},4"
                observed = sum(row[key] * row["implicit_pair_count"] for row in rows) / denominator
                expected = result[bucket][key]
                if isinstance(expected, dict):
                    expected = expected["value"]
                if abs(observed - expected) > 1e-10:
                    raise ValueError(f"{rule}: inconsistent {key}")


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ]


def build(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load(args.protocol)
    groups = query_groups(args.dataset_root)
    payloads = {
        f"{payload['entry']}__{payload['split']}": payload
        for path in sorted(args.evaluation_dir.glob("*.json"))
        for payload in [load(path)]
    }
    required = {
        f"{entry}__{split}"
        for entry in protocol["entries"]
        for split in ("dev", "test")
    }
    missing = sorted(required - payloads.keys())
    if missing:
        raise FileNotFoundError("Missing frozen evaluations: " + ", ".join(missing))
    for payload in payloads.values():
        if payload["protocol_sha256"] != checkpoint_fingerprint(args.protocol) or payload["selection_performed"]:
            raise ValueError("Evaluation does not match the frozen protocol")
        validate_aggregates(payload)

    summary: dict[str, Any] = {
        "format_version": 2,
        "status": "complete_stage1_closeout",
        "protocol_sha256": checkpoint_fingerprint(args.protocol),
        "entries": {
            key: {rule: result_summary(payload, rule) for rule in payload["rules"]}
            for key, payload in payloads.items()
        },
        "bootstrap_test": {},
    }
    summary["three_seed"] = {}
    for arm in ("e01", "p_frozen"):
        for split in ("dev", "test"):
            seed_summaries = [summary["entries"][f"{arm}_s{seed}__{split}"] for seed in (13, 17, 23)]
            summary["three_seed"][f"{arm}__{split}"] = {
                rule: {
                    metric: {
                        "mean": statistics.fmean(item[rule][metric] for item in seed_summaries),
                        "std_population": statistics.pstdev(item[rule][metric] for item in seed_summaries),
                    }
                    for metric in ("recall_at_10", "valid_path_at_10_4", "row_support_at_10_4", "implicit_recall_at_10")
                }
                for rule in seed_summaries[0]
            }
    left = payloads["p_frozen_s13__test"]["results"]["f4_lambda_0.5"]["per_query"]
    right = payloads["p_frozen_s13__test"]["results"]["f0_direct"]["per_query"]
    left_rows = {row["query_id"]: row for row in left}
    right_rows = {row["query_id"]: row for row in right}
    summary["bootstrap_test"]["f4_vs_direct_recall_at_10"] = bootstrap_delta(
        left_rows, right_rows, groups, "recall@10",
        samples=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
    )
    left_rows = {
        key: {
            "metric": row.get("valid_path_recall@10,4", 0.0),
            "implicit_pair_count": row.get("implicit_pair_count", 0),
        }
        for key, row in left_rows.items()
    }
    right_rows = {
        key: {
            "metric": row.get("valid_path_recall@10,4", 0.0),
            "implicit_pair_count": row.get("implicit_pair_count", 0),
        }
        for key, row in right_rows.items()
    }
    summary["bootstrap_test"]["f4_vs_direct_valid_path"] = bootstrap_delta(
        left_rows, right_rows, groups, "metric", weight_field="implicit_pair_count",
        samples=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
    )
    for name, left_entry, left_rule, right_entry, right_rule in (
        ("p_frozen_vs_e01", "p_frozen_s13", "f2_rrf_e005", "e01_s13", "f2_rrf_e005"),
        ("p_frozen_f4_vs_r5", "p_frozen_s13", "f4_lambda_0.5", "r5_repro", "f2_rrf_e005"),
    ):
        lhs = {r["query_id"]: r for r in payloads[f"{left_entry}__test"]["results"][left_rule]["per_query"]}
        rhs = {r["query_id"]: r for r in payloads[f"{right_entry}__test"]["results"][right_rule]["per_query"]}
        for field, weight in (("recall@10", None), ("valid_path_recall@10,4", "implicit_pair_count"), ("row_support_coverage@10,4", "implicit_pair_count")):
            summary["bootstrap_test"][f"{name}__{field}"] = bootstrap_delta(
                lhs, rhs, groups, field, weight_field=weight,
                samples=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
            )
    lhs = {r["query_id"]: r for r in left}
    rhs = {r["query_id"]: r for r in right}
    summary["bootstrap_test"]["f4_vs_direct_row_support"] = bootstrap_delta(
        lhs, rhs, groups, "row_support_coverage@10,4", weight_field="implicit_pair_count",
        samples=protocol["bootstrap"]["replicates"], seed=protocol["bootstrap"]["seed"],
    )
    summary["protocol"] = {
        "recall_ks": protocol["recall_ks"],
        "retrieval_budget": protocol["retrieval_budget"],
        "seed_scope": protocol["seed_scope"],
        "test_selection_performed": False,
    }
    summary["limitations"] = [
        "Stage-2 stopped by user; no completed end-to-end join evaluation.",
        "Student seed repeats conditional on a fixed Teacher, C4 initialization and mining pool.",
        "WDC v9 pending authoritative snapshot; no older WDC substitution.",
        "Pooled-pair Teacher and calibration-only diagnostics were not executed; no claim about token-level Teacher advantage.",
        "Equal-wall-time curves, independent exact-search verification and complete modality/failure-case breakdowns were not completed.",
    ]
    r10 = args.protocol.parent.parent
    histories = {
        cell: load(r10 / "taskE_matched" / ("t0" if cell[1] == "0" else "t1") / cell / "student_path.pt.history.json")
        for cell in ("e00", "e01", "e02", "e10", "e11", "e12", "e13")
    }
    endpoints = {cell: hist["epochs"][-1]["dev_retrieval"] for cell, hist in histories.items()}
    effects = {
        "T1_minus_T0_at_G0": {"e10": 1, "e00": -1},
        "G2b_minus_G0_at_T0": {"e01": 1, "e00": -1},
        "Gstar_minus_G0_at_T0": {"e02": 1, "e00": -1},
        "interaction_G2b": {"e11": 1, "e10": -1, "e01": -1, "e00": 1},
        "interaction_Gstar": {"e12": 1, "e10": -1, "e02": -1, "e00": 1},
    }
    summary["matched_endpoint_effects"] = {
        name: {
            channel: statistics.fmean(
                sum(weight * endpoints[cell]["per_query"][channel]["recall@10"][index] for cell, weight in weights.items())
                for index in range(endpoints["e00"]["queries"])
            ) for channel in ("fused", "direct", "evidence")
        } for name, weights in effects.items()
    }
    summary["matched_endpoint_effects_note"] = "Seed-13 fixed epoch-2 endpoint, shared dev query order; descriptive per-query Recall contrasts, no interaction confidence intervals."
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    test = payloads["p_frozen_s13__test"]
    lines = [
        "# R10 Stage-1 冻结评测附表",
        "",
        "更新时间：2026-09-08（UTC）  ",
        "范围：EntiTables v9；Stage-1 A--F。Stage-2 按用户要求停止。",
        "",
        "## 技术摘要",
        "",
        "本报告收尾 EntiTables v9 的 Stage-1 主实验、两个候选的 Student 三种子复现和冻结 test 评估。数据完整性审计通过，不支持“数据集损坏”的判断。edge-first 的检索退化与 confidence 校准改善同时出现；F4 lambda=0.5 是 dev 选出的机制候选，是否泛化以本报告 test 结果为准。Stage-2 已停止，任何 Stage-1 指标都不代表最终 join 成功。",
        "",
        "## 固定 test 结果",
        "",
        "| 固定规则 | R@10 | ValidPath@10,4 | RowSupport@10,4 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rule in ("f0_direct", "f1_evidence", "f2_rrf_e005", "f3_rrf_equal", "f4_lambda_0.5", "f5_reserved_half"):
        value = result_summary(test, rule)
        lines.append(f"| {rule} | {pct(value['recall_at_10'])} | {pct(value['valid_path_at_10_4'])} | {pct(value['row_support_at_10_4'])} |")
    ci = summary["bootstrap_test"]["f4_vs_direct_recall_at_10"]
    lines.extend([
        "",
        f"上述为 p_frozen seed 13 的 test。F4 lambda=0.5 相对 direct-only 的 R@10 差值为 {ci['mean_difference'] * 100:+.3f} 个百分点，95% source-table bootstrap CI [{ci['ci95_low'] * 100:+.3f}, {ci['ci95_high'] * 100:+.3f}]。这是预先固定配置的检验，不是在 test 上重新选最优点。",
        "",
        "配套 report.html 使用上述冻结数值绘图。总体召回按全部 query 宏平均，有效路径按 implicit 正 (Q,T) 对汇总，分母不同。",
        "",
        "## 数据与定义",
        "",
        "- EntiTables v9：14,994 queries、22,886 lake tables、225,720 bridge assets、15,870 qrels、23,592 recoveries。",
        "- Stage-1 serialization：263,600 objects、57,424 edge lists、248,606-object corpus；retrieval features 263,600，Teacher features 47,261。",
        "- PCA-1024 explained variance ratio 0.9290423，最大 Gram error 为 4.768e-07。",
        "- split：train-fit 11,390、train-calibration 1,240、dev 1,198、test 1,166；source-group 无跨 split 交集。",
        "- R@K：每个 query 检出的正 target 数 / 该 query 全部正 target 数，再对 query 宏平均。dev/test 各包含 implicit 与 explicit query。",
        "- ValidPath@10,4：对每个有 recovery 支持的 implicit 正 (Q,T)，检查 T 是否进入 top-10 且最多 4 条保留 evidence 中至少有一条支持同一实体-属性-值路径；按正 (Q,T) 对汇总，不是按所有 evidence pair 汇总。",
        "- RowSupport@10,4：对每个 implicit 正 (Q,T)，保留 evidence 支持的不同 query 行数 / 5，再对正对宏平均；T 未检出记 0。它是已确认的可用证据行覆盖，不是生成值正确率。",
        "- 固定 ANN 预算为 Q-T=100、Q-text=20、Q-image=20、E-T=20；B=4，K=10/20/50。F5 对每个 K 独立分配名额。本次修正了旧实现从 K=50 结果截断 top-10 的口径，旧 F5 行已被本报告替代。",
        "",
        "## 训练与聚合证据",
        "",
        "### C4 是稳定初始化，edge-first 不是替代方案",
        "",
        "C4 selected epoch 0 的 ValidPath/R@10 为 10.0295%/25.5634%。D1/D2/D3 的 ValidPath 分别为 0.2950%/0.2950%/0%，而 D3 的 edge AUROC 为 0.94079、Brier 0.12584；绝对校准变好并没有恢复路径检索。因此保留 C4，不用 D3 替换。",
        "",
        "### LME、G5 和 matched E 只有局部机制收益",
        "",
        "LME 降低 path-count correlation，但同时削弱 evidence 召回。hard-mining 的 epoch-2 endpoint 相对 control 提高 evidence/ValidPath/RowSupport，却在 checkpoint 选择规则下仍回到 C4。exact-content G5 mined endpoint 为 Fused 26.23%、Evidence 15.99%、ValidPath 9.88%、RowSupport 3.57%，属于机制轨迹证据，不是最终部署结果。",
        "",
        "E01（T0+G2b）是正式 matched cell 中最好的 evidence 点，但 ValidPath 仅 0.2950%；E13（T1+G5）为 0%，未复制 inference-only G5 增益。p_frozen 是 E/F 控制里总体质量最好的 Student control，不是对 C3a/C4/r5-repro 的全面胜出声明。",
        "",
        "## 结论与行动",
        "",
        "1. 数据集完整性审计通过；confirmed/unknown 监督覆盖有限，但不能从未标注边直接推断数据损坏。",
        "2. 在 E/F 两个候选中保留 p_frozen 作为质量控制，同时对照 C3a/C4/r5；F4 lambda=0.5 为 dev 预选机制配置，不能仅凭少量 ValidPath 变化取代质量对照。",
        "3. 不把 edge-first、BCE 或 LME 单独宣称为 join-discovery 成功。",
        "4. Stage-2 reader/FOCUS/row filling/generation 没有完成；本报告不对最终 joinability 做推断。",
        "",
        "## 未完成项与限制",
        "",
        "- WDC v9 缺少权威 root manifest，保持 pending-input，没有使用旧 WDC 替代。",
        "- test 只评估 frozen protocol；没有在 test 选择融合权重、checkpoint 或阈值。多种子方差是在固定 seed-13 Teacher 条件下的 Student 方差，不能冒充完整 Teacher-Student 方差。",
        "- 计划中的 pooled-pair Teacher 和 calibration-only 诊断尚无正式产物，不能宣称 Teacher token 级优势或把 BCE 改善完全归因于排序学习。它们是 Stage-1 未执行诊断，不能归因于停止 Stage-2。",
        "- Stage-2 在 32/16-example reader smoke、线性头 smoke 和 dev retrieval 导出后停止；全量 train reader cache 停在 2,880/7,035。没有完整值恢复、grounding 或最终 join 指标。",
        "",
        "## 可复现证据",
        "",
        "- 冻结协议：stage1_closeout/frozen_protocol.json。",
        "- 机器摘要：final_summary.json。",
        "- 任务报告：各 task*/RESULTS.md。",
        "- 数据/输入审计：taskA_protocol/inputs.json、taskA_protocol/label_audit.json。",
        "- 回归和产物核验详见 VALIDATION.md。",
        "",
    ])
    lines.extend(["## 全部冻结参考模型（seed 13，F2 RRF .05）", ""])
    rows = []
    for name in ("raw", "pca_epoch0", "r5_repro", "c3a", "c4", "e01_s13", "p_frozen_s13"):
        for split in ("dev", "test"):
            v = summary["entries"][f"{name}__{split}"]["f2_rrf_e005"]
            rows.append([name, split, pct(v["recall_at_10"]), pct(v["valid_path_at_10_4"]), pct(v["row_support_at_10_4"])])
    lines.extend(table(["模型", "split", "R@10", "ValidPath", "RowSupport"], rows))
    lines.extend(["", "## Student 三种子均值 ± 总体标准差", "", "seeds=13/17/23，固定同一 Teacher、C4 初始化、mining pool；不是整链方差，也不是置信区间。", ""])
    rows = []
    for name, rules in summary["three_seed"].items():
        for rule in ("f0_direct", "f2_rrf_e005", "f4_lambda_0.5"):
            if rule not in rules:
                continue
            values = [f"{rules[rule][m]['mean'] * 100:.3f} ± {rules[rule][m]['std_population'] * 100:.3f}" for m in ("recall_at_10", "valid_path_at_10_4", "row_support_at_10_4")]
            rows.append([name, rule, *values])
    lines.extend(table(["模型/split", "融合", "R@10 (%)", "ValidPath (%)", "RowSupport (%)"], rows))
    lines.extend(["", "## Test source-group 配对 bootstrap", "", "差值及区间单位均为百分点；10,000 次，seed=13，以 source_table_id 聚类重采样。R@10 保持 query 宏平均，路径/行覆盖保持 implicit 正对加权分母；九个区间未做多重比较校正，不替代训练随机性。", ""])
    rows = [[name, f"{ci['mean_difference'] * 100:+.3f}", f"[{ci['ci95_low'] * 100:+.3f}, {ci['ci95_high'] * 100:+.3f}]", ci["groups"], ci["queries"]] for name, ci in summary["bootstrap_test"].items()]
    lines.extend(table(["差值", "估计 (pp)", "95% CI (pp)", "source groups", "queries"], rows))
    lines.extend(["", "## Test 分层与深度召回（seed 13）", ""])
    rows = []
    for name, rule in (("raw", "f2_rrf_e005"), ("r5_repro", "f2_rrf_e005"), ("e01_s13", "f2_rrf_e005"), ("p_frozen_s13", "f0_direct"), ("p_frozen_s13", "f4_lambda_0.5")):
        v = summary["entries"][f"{name}__test"][rule]
        rows.append([name + "/" + rule, pct(v["implicit_recall_at_10"]), pct(v["explicit_recall_at_10"]), pct(v["recall_at_20"]), pct(v["recall_at_50"]), f"{v['mrr_at_50']:.4f}", v["multi_positive"]["queries"], pct(v["multi_positive"]["recall_at_10"])])
    lines.extend(table(["模型/规则", "implicit R10", "explicit R10", "R20", "R50", "MRR50", "多正例 query 数", "多正例 R10"], rows))
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "entries": len(payloads), "report": str(args.report), "summary": str(args.summary)}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    build(parser.parse_args())

"""Summarize completed ABC replay, including seed-averaged Path/Edge contrasts."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mmdd_stage1.r26_statistics import source_cluster_comparison
from replay_r26_evidence_order_abc import ARMS, read_json, read_rows, write_json


def analyze(output: Path) -> None:
    result = read_json(output / "RESULTS.json")
    assert result["status"] == "completed"
    models = {r["generator"]: r for r in result["models"]}
    contrasts = []
    names = [f"R26-{arm}/seed{seed}/step178" for arm in ("O-SUP", "E-GRAPH") for seed in (13, 29)]
    if all(name in models for name in names):
        rows = {name: {r["query_id"]: r for r in read_rows(output / "models" / name / "per_query.jsonl.gz")}
                for name in names}
        population = rows[names[0]]
        assert all(rr.keys() == population.keys() for rr in rows.values())
        for kind in ("overall", "implicit", "explicit"):
            ids = [q for q, r in population.items() if kind == "overall" or r["query_kind"] == kind]
            for stage, metric in (("E", "recall@10"), ("Equal", "recall@10"), ("C100", "raw_recall"), ("T0", "recall@10")):
                deltas = {}
                for arm in ARMS:
                    seed_deltas = []
                    for seed in (13, 29):
                        path = rows[f"R26-O-SUP/seed{seed}/step178"]
                        edge = rows[f"R26-E-GRAPH/seed{seed}/step178"]
                        seed_deltas.append(np.array([path[q]["metrics"][arm + "/" + stage][metric]
                                                    - edge[q]["metrics"][arm + "/" + stage][metric] for q in ids]))
                    delta = np.mean(seed_deltas, axis=0)
                    deltas[arm] = delta
                    contrasts.append({"kind": kind, "arm": arm, "contrast": "O-SUP_minus_E-GRAPH",
                                      "stage": stage, "metric": metric,
                                      "seed_deltas": {str(s): float(d.mean()) for s, d in zip((13, 29), seed_deltas)},
                                      **source_cluster_comparison(delta, [population[q]["source_table_id"] for q in ids])})
                for arm in ARMS[1:]:
                    contrasts.append({"kind": kind, "arm": arm, "contrast": "Path_minus_Edge_change_vs_A",
                                      "stage": stage, "metric": metric,
                                      **source_cluster_comparison(deltas[arm] - deltas[ARMS[0]], [population[q]["source_table_id"] for q in ids])})
    write_json(output / "PATH_EDGE_CONTRASTS.json", contrasts)
    b13 = models["B13"]
    m = b13["summary"]["overall"]["metrics"]
    h = b13["summary"]["overall"]["strict_hits"]
    lines = ["# R26 Evidence 排序实验结论", "",
             f"已完成 {len(models)} 个模型 × 1198 条冻结 dev queries 的 A/B/C 重放；所有模型均通过 R26 coverage 基线完整排名及历史指标复现。", "",
             "A：D1 coverage；B：retained Path-LSE；C：pre-retention Path-LSE。三臂固定 Direct100、E/U 候选、retained evidence、row routing、Teacher 分数和 C100 预算；只改 E-channel 排序。C 也限于相同 retained targets。", "",
             "## B13 的主要结果", "",
             "| 指标 | A coverage | B retained LSE | C 原始 LSE |",
             "|---|---:|---:|---:|"]
    for label, stage, metric in (("E Recall@10", "E", "recall@10"), ("E Recall@50", "E", "recall@50"),
                                  ("Equal Recall@10", "Equal", "recall@10"), ("C100 Recall", "C100", "raw_recall"),
                                  ("Teacher Recall@10", "T0", "recall@10"), ("Teacher Recall@50", "T0", "recall@50")):
        lines.append(f"| {label} | " + " | ".join(f"{100*m[arm+'/'+stage][metric]:.3f}%" for arm in ARMS) + " |")
    for label, stage, k in (("207 个 strict EO：E Top50", "E", 50), ("207 个 strict EO：C100", "C100", 100),
                             ("207 个 strict EO：Teacher Top10", "T0", 10)):
        lines.append(f"| {label} | " + " | ".join(str(h[arm + '/' + stage][str(k)]) for arm in ARMS) + " |")
    primary = next(c for c in b13["comparisons"] if c["kind"] == "overall" and c["new"] == ARMS[1]
                   and c["old"] == ARMS[0] and c["stage"] == "T0" and c["metric"] == "recall@10")
    ci = primary["bootstrap_95ci"]
    lines += ["", f"B−A 的 Teacher Recall@10 差异为 **{100*primary['mean_delta']:+.3f} pp**，source-cluster bootstrap 95% CI **[{100*ci[0]:+.3f}, {100*ci[1]:+.3f}] pp**。",
              "B13 上，直接换成 retained Path-LSE 没有改善结果；原始 Path-LSE 也未超过 coverage。机制上的 score handoff 确实存在，但本实验不支持“它是效果不佳的主要原因，换回 LSE 即可解决”的解释。", "",
              "207 个 strict evidence-only positives 定义为 G ∩ E − (D_ANN100 ∪ D_EXACT100)。QT_OVER_U Top10/20/50 均为 0；这仍是纯 QT 诊断，不能替代 E/RRF 指标。这里实际 E、C100 和 T0 漏斗同样在换成 LSE 后下降。", "",
              "## 所有模型：Teacher Recall@10", "",
              "| 模型 | A (%) | B (%) | C (%) | B−A (pp) | B−A 95% CI (pp) |",
              "|---|---:|---:|---:|---:|---|"]
    for name, model in models.items():
        mm = model["summary"]["overall"]["metrics"]
        comparison = next(c for c in model["comparisons"] if c["kind"] == "overall" and c["new"] == ARMS[1]
                          and c["old"] == ARMS[0] and c["stage"] == "T0" and c["metric"] == "recall@10")
        lo, hi = comparison["bootstrap_95ci"]
        lines.append(f"| {name} | " + " | ".join(f"{100*mm[a+'/T0']['recall@10']:.3f}" for a in ARMS)
                     + f" | {100*comparison['mean_delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    if contrasts:
        lines += ["", "## Path / Edge 的差异", "",
                  "在同 seed 的 R26-O-SUP 与 R26-E-GRAPH 间作比较，先对两 seed 的逐 query 差异求均值，再按 source_table_id bootstrap。每个模型保留自身候选池；这是评分规则对模型差异的影响分析，不是跨模型固定同一候选池。", "",
                  "| E 排序 | O-SUP−E-GRAPH 的 T0 R10 差异(pp) | 95% CI(pp) |",
                  "|---|---:|---|"]
        for c in contrasts:
            if c["kind"] == "overall" and c["stage"] == "T0" and c["contrast"] == "O-SUP_minus_E-GRAPH":
                lo, hi = c["bootstrap_95ci"]
                lines.append(f"| {c['arm']} | {100*c['mean_delta']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    counts = b13["retention_diagnostics"]
    lines += ["", "## 能支持与不能支持的判断", "",
              f"- B13 D1 retention 删除了 {counts['targets_dropped_by_retention']} 个 target；三臂 U recall 固定为 {100*m[ARMS[0]+'/Equal']['raw_recall']:.3f}%。差异来自排名和 C100 admission。",
              f"- {counts['qrel_positive/best_path_score_lost']}/{counts['qrel_positive/targets']} 个有证据的 qrel-positive query-target 对丢失了最高分路径；这不是语义正确证据被删的标注。原始 LSE 还受路径数量影响，不能仅靠 C−B 区分质量与数量。",
              "- retained evidence 与 row routing 逐 query-target 固定；C100 的证据条数、路由行数、D1 分数总量保存在 per_query 中。没有新的属性正确性标注或属性生成，因此不能宣称正确值恢复、真实属性覆盖或最终 semantic-joinability 有变化。",
              "- Teacher T0 自身是 QT pair scorer，没有读取 paths；本实验观察的 Teacher 变化来自上游 C100 成员变化。",
              "- 不应根据该负结果放弃 evidence 分支。它排除了一个简单修复假设；下一步需要把路径高分的语义质量、路径数量效应及下游真实属性恢复分开验证。",
              "- 结果全部保留，未按 dev 结果筛选模型、改变 RRF 权重或额外训练。主比较为 B13 B−A，其余多模型/分组/Path-Edge 分析属于探索性，未作多重检验校正。", "",
              "完整分组与漏斗见 REPORT.zh-CN.md；逐 query 排名、固定输入哈希和证据记录见 models/；每个模型 RESULTS.json 包含 10,000 次 source-cluster bootstrap。", ""]
    (output / "FINDINGS.zh-CN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output.resolve())

"""Aggregate paired recall and evidence diagnostics for the AbeBooks 2x2 study."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from mmdd_stage1.data import iter_jsonl, read_json, write_json
from mmdd_stage1.evaluate import paired_bootstrap

ARMS = {"baseline": "原始", "hubs": "仅过滤素材", "columns": "仅精简列", "both": "两者同时"}
KS = (5, 10, 15, 20)


def diagnostics(run: Path, generator: str, split: str = "test") -> dict:
    directory = run / "eval" / split / generator
    pools = {r["query_id"]: r for r in iter_jsonl(directory / "pools.jsonl.gz")}
    first = defaultdict(set)
    for r in iter_jsonl(directory / "first_hop.jsonl.gz"):
        first[r["query_id"]].add(r["evidence_id"])
    canonical = {r["asset_id"]: r["canonical_evidence_id"] for r in iter_jsonl(run / "CONTENT_ALIASES.jsonl.gz")}
    gold = defaultdict(set)
    for r in iter_jsonl(run / "dataset_view/qrels.jsonl"):
        if r["split"] == split and r["rel"] > 0:
            gold[r["query_table_id"]].add(r["target_table_id"])
    units = defaultdict(set)
    query_evidence = defaultdict(set)
    for r in iter_jsonl(run / "dataset_view/evidence_recoveries/part-00000.jsonl"):
        if r["split"] == split:
            q, t = r["query_table_id"], r["target_table_id"]
            attr = r["recovered_attribute"]
            key = (q, t, r["query_row_id"], attr["column_name"], str(attr["value"]))
            units[key].add(canonical[r["evidence"]["asset_id"]])
            query_evidence[q].add(canonical[r["evidence"]["asset_id"]])
    pairs = list(iter_jsonl(directory / "funnels/witness_pairs.jsonl.gz"))
    strict = read_json(directory / "funnels/strict_EO_SUMMARY.json")
    retained_units = sum(bool(eids & set(pools[q]["retained_bags"].get(t, []))) for (q, t, *_), eids in units.items())
    first_units = sum(bool(eids & first[q]) for (q, *_), eids in units.items())
    return {"gold_pairs": sum(map(len, gold.values())), "queries": len(gold),
        "C150_macro_coverage": sum(len(set(pools[q]["C150"]) & g) / len(g) for q, g in gold.items()) / len(gold),
        "U_macro_coverage": sum(len(set(pools[q]["U"]) & g) / len(g) for q, g in gold.items()) / len(gold),
        "witness_pairs": len(pairs), **{stage: sum(bool(r[stage]) for r in pairs) for stage in
        ("any_QE20_hit", "any_ET50_hit", "any_D1_retained", "any_target_in_C150", "any_teacher_top10")},
        "annotated_row_attribute_value_units": len(units), "QE20_covered_units": first_units,
        "D1_covered_units": retained_units,
        "QE20_precision_on_annotated_queries": sum(len(first[q] & eids) / len(first[q])
            for q, eids in query_evidence.items()) / len(query_evidence),
        "strict_EO": strict,
        "note": "Coverage of previously annotated recoverable row/attribute/value units; no new attribute extraction or value verification."}


def summarize(root: Path) -> None:
    rows, evidence, selections, training, perquery = [], {}, {}, {}, {}
    for arm in ARMS:
        run = root / arm
        assert read_json(run / "EVALUATION_COMPLETE.json")["status"] == "COMPLETE"
        rows += [{"arm": arm, **r} for r in read_json(run / "recall_summary.json")]
        evidence[arm] = {g: diagnostics(run, g) for g in
                         ("raw", "selected_sup", "selected_kd", "endpoint_sup", "endpoint_kd")}
        selections[arm] = read_json(run / "SELECTION_FREEZE.json")
        training[arm] = read_json(run / "TRAINING_COMPLETE.json")
        perquery[arm] = {(r["generator"], r["mode"], r["query_id"]): r
                        for r in iter_jsonl(run / "recall_per_query.jsonl") if r["split"] == "test"}
    queries = list(iter_jsonl(root / "baseline/dataset_view/query_tables/part-00000.jsonl"))
    groups = {q["table_id"]: q["source_table_id"] for q in queries if q["split"] == "test"}
    contrasts = []
    for arm in ("hubs", "columns", "both"):
        for generator in ("endpoint_sup", "endpoint_kd"):
            for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                for k in KS:
                    deltas = {q: perquery[arm][generator, mode, q][f"R@{k}"] -
                              perquery["baseline"][generator, mode, q][f"R@{k}"] for q in groups}
                    contrasts.append({"arm": arm, "vs": "baseline", "generator": generator,
                                      "mode": mode, "k": k, **paired_bootstrap(deltas, groups)})
    write_json(root / "COMPARISON.json", {"recall": rows, "evidence": evidence,
        "selections": selections, "training": training, "paired_contrasts": contrasts})
    lookup = {(r["arm"], r["split"], r["generator"], r["mode"], r["segment"]): r for r in rows}

    def metric_row(arm: str, generator: str, mode: str, split: str = "test", segment: str = "overall") -> dict:
        return lookup[arm, split, generator, mode, segment]

    def values(row: dict) -> str:
        return " | ".join(f"{100 * row[f'R@{k}']:.2f}%" for k in KS)

    base_e, col_e = evidence["baseline"]["endpoint_sup"], evidence["columns"]["endpoint_sup"]
    student_max = max(metric_row(a, g, m)[f"R@{k}"] for a in ARMS
                      for g in ("endpoint_sup", "endpoint_kd")
                      for m in ("Direct", "Multimodal_RRF") for k in KS)
    findings = [
        "精简列改善了正确证据的召回与保留，但本轮尚未转化为 student 最终排序收益；仅过滤高频无效素材没有改善正确证据覆盖。",
        f"完成训练的 SUP/KD 在四组、Direct/RRF 两种排序及四个 cutoff 中，最高 Recall 为 {100*student_max:.2f}%。",
        f"精简列使正确证据 QE20 命中的正例对从 {base_e['any_QE20_hit']}/{base_e['witness_pairs']} 增至 {col_e['any_QE20_hit']}/{col_e['witness_pairs']}，"
        f"D1 保留从 {base_e['any_D1_retained']}/{base_e['witness_pairs']} 增至 {col_e['any_D1_retained']}/{col_e['witness_pairs']}；"
        f"D1 覆盖的标注行/属性/值单元从 {base_e['D1_covered_units']}/{base_e['annotated_row_attribute_value_units']} 增至 {col_e['D1_covered_units']}/{col_e['annotated_row_attribute_value_units']}。",
        "Teacher Real 的 R@10 在原始、仅过滤、仅精简、组合四组分别为 " +
        "、".join(f"{100*metric_row(a, 'endpoint_sup', 'Teacher_Real')['R@10']:.2f}%" for a in ARMS) +
        "；组合组仅在 R@20 小幅提高，不能称为稳定改善。SUP/KD 完整终点的这些 aggregate 指标相同。",
        "因此，删列方向在证据质量上有正面证据，但当前下游保留、融合评分及有限 student 训练仍需进一步检验；本实验不能断言冗余列是最终 recall 低的唯一原因。",
    ]
    lines = ["# AbeBooks 数据处理 2×2 对照实验", "", *findings, "",
        "四组均从本次新生成的特征开始，调用当前 src 的 TA → TB_CQET → Native C1 → SUP/KD C2。",
        "不复用此前实验的特征、PCA 或 checkpoint；本次实验中不变的素材特征共享，改变的表重新编码，各组 PCA 和模型独立训练。",
        "seed=13，train/dev/test=120/15/15 queries；test 共 16 个 gold target 对。Recall 是逐 query 的 |top-k∩gold|/|gold| 的均值。", "",
        "## 数据处理", "",
        "- 全局列名口径：保留 title 和所有正例 join 曾使用的 16 类列；不向原先没有 title 的投影补入 title。",
        "- 原表、query、target 一致删列，重新编号本地和 source 索引，保持保留单元格的值不变。",
        "- 高频无效素材：基线 Raw 的训练集首跳文本 top20、图片 top20，至少被 12/120 个不同 query 召回，且不属于任何 gold evidence 的内容类别。",
        "- 过滤表由训练集频次一次确定，两个过滤组使用同一清单；所有 split 的 gold evidence 及其精确内容别名都受保护。",
        "- 这是使用标签整理数据集的对照实验：全局 join 列和 gold 素材保护包含测试标签信息，不应当解释为无标签、可直接部署的数据清洗算法。", ""]
    curation = read_json(root / "columns/CURATION.json")
    hubs = read_json(root / "HUB_FILTER.json")
    lines += ["删除的列：`" + "`, `".join(curation["removed_columns"]) + "`。", "",
        f"删除 {hubs['removed_raw']} 个素材：61 个文本、31 张图片；5,441 → 5,349。最高频素材是被 91 个训练 query 召回的 Image Not Available 占位图。",
        "原表平均列数 20.80 → 12.90；query 10.17 → 7.17；target 10.41 → 7.15。154 个正例对、228 条恢复记录保持不变。",
        "原始数据未修改；三个新数据集分别位于 `hubs/dataset_view`、`columns/dataset_view`、`both/dataset_view`。",
        "原始投影中 query 有 146/150 张含 title，但 target 仅 14/167 张含 title；精简上下文不会为其余 target 新增 title，所以不能预期它自动解决 query→target 直接匹配。",
        "基线排除了原始数据中的 2 张缺失图片、7 条空文本；四组共同使用该可编码基线，这 9 项均不是 gold evidence。", "",
        "## Test Recall：完成训练的固定终点", "",
        "下面报告 C2 完整一遍后的终点，保证这些行确实经过参数更新。Direct 为 student 直接检索；RRF 为多模态候选融合排序；Teacher 为相同 student 候选上的 CQET Real 诊断，不计作 student 本身的 recall。", "",
        "| 数据 | Student | 排序 | R@5 | R@10 | R@15 | R@20 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for student in ("sup", "kd"):
            for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                lines.append(f"| {ARMS[arm]} | {student.upper()} | {mode} | {values(metric_row(arm, 'endpoint_' + student, mode))} |")
    lines += ["", "## Dev 选择与 Raw 对照", "",
        "C1 和 C2 均按当前规则最大化 dev C150 覆盖率、U 覆盖率、Direct R@10，再优先较早 checkpoint；C2 只用 SUP 选择，KD 使用相同 fraction。所有组选择固定后才运行本轮 test 排名评测。", "",
        "| 数据 | C1 fraction | C2 fraction | TA/TB/C1/C2(SUP,KD) 更新步数 | 训练流程墙钟秒 |", "|---|---:|---:|---|---:|"]
    for arm in ARMS:
        s, t = selections[arm], training[arm]
        lines.append(f"| {ARMS[arm]} | {s['C1_fraction']} | {s['C2_fraction']} | {list(t['steps'].values())} | {t['seconds']:.1f} |")
    lines += ["", "C2 当前实现固定遍历 120 条 query list 一次，batch=64，所以只有 2 次更新（64+56）；本轮为隔离数据因素没有同时增加训练预算。fraction=0 表示 dev 选择了阶段初始状态，不能据此声称 KD 已优于充分训练的 SUP。", "",
        "| 数据 | Generator | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for generator in ("raw", "selected_sup", "selected_kd"):
            for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                lines.append(f"| {ARMS[arm]} | {generator} | {mode} | {values(metric_row(arm, generator, mode))} |")
    lines += ["", "## 证据链与行/属性覆盖", "",
        "以下使用完成训练的 SUP 终点；KD 与 Raw 的完整记录见 COMPARISON.json。QE20 是正确素材进入首跳，ET50 是该素材连到正确 target，D1 是正确素材最终被保留。pair 统计分母为 7 个带 gold witness 的测试正例对。", "",
        "| 数据 | C150 覆盖率 | QE20 pair | ET50 pair | D1 pair | 标注行/属性/值单元 | 首跳覆盖单元 | D1 覆盖单元 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        e = evidence[arm]["endpoint_sup"]
        lines.append(f"| {ARMS[arm]} | {100*e['C150_macro_coverage']:.2f}% | {e['any_QE20_hit']}/{e['witness_pairs']} | {e['any_ET50_hit']}/{e['witness_pairs']} | {e['any_D1_retained']}/{e['witness_pairs']} | {e['annotated_row_attribute_value_units']} | {e['QE20_covered_units']} | {e['D1_covered_units']} |")
    lines += ["", "这里的行/属性/值覆盖仅判断是否召回了已有标注对应的证据；本轮没有重新调用属性抽取模型或执行二阶段 join 验证，因此没有新的正确值恢复率或最终 joinability 结论。", "",
        "## 证据控制与不确定性", "",
        "相同 SUP 终点候选集上，比较 CQET Real、禁用证据评分的 f0、替换证据的 Swap，检验正确素材是否解释收益。", "",
        "| 数据 | Real R@10 | f0 R@10 | Swap R@10 | implicit Real R@10 | explicit Real R@10 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        vals = [metric_row(arm, "endpoint_sup", mode)["R@10"] for mode in ("Teacher_Real", "Teacher_f0", "Teacher_Swap")]
        vals += [metric_row(arm, "endpoint_sup", "Teacher_Real", segment=seg)["R@10"] for seg in ("implicit", "explicit")]
        lines.append(f"| {ARMS[arm]} | " + " | ".join(f"{100*v:.2f}%" if v is not None else "NA" for v in vals) + " |")
    lines += ["", "相对原始组的 SUP 终点 Teacher Real R@10 差值；95% 区间按 source group 配对 bootstrap 10,000 次。", "",
        "| 数据 | 差值 pp | 95% 区间 pp | query 胜/负/平 |", "|---|---:|---|---|"]
    for c in contrasts:
        if c["generator"] == "endpoint_sup" and c["mode"] == "Teacher_Real" and c["k"] == 10:
            lines.append(f"| {ARMS[c['arm']]} | {c['mean_delta_pp']:+.2f} | [{c['ci_95'][0]:.2f}, {c['ci_95'][1]:.2f}] | {c['wlt']} |")
    lines += ["", "单 seed、15 个测试 query 的结果只适合作为探索性证据；保留全部负结果和权衡。Teacher 各组也重新训练，Teacher 差异反映整条数据处理及训练流程的总效应，不能归因于单个模块。", "",
        "Real 并未稳定超过 f0/Swap，即使正确证据覆盖增加，当前评分也没有稳定体现正确证据带来的排序优势。后续应在独立对照中增加 student 更新量，检查正确证据进入候选后的聚合、融合及梯度贡献；不应仅凭本轮结果就替换为无证据模型。", "",
        "## 复现与核验", "",
        "入口：`src/run_abebooks_fresh.py` 生成新基线；`src/prepare_abebooks_ablation_features.py` 的 curate、encode-tables、compose；`src/run_abebooks_data_ablation.py` 的 train、evaluate；最后运行本汇总脚本。全部在 MMDD 环境及隔离工作目录执行。",
        "训练调用默认超参数：TA 2 epochs、TB_CQET 1 epoch、teacher batch 8；C1/C2 单遍、batch 64、P lr=1e-6、R lr=1e-5。没有新增线上 reranker，也未改变核心训练或检索算法。",
        "必需的五阶段均完成；QT/LSE 等无关方法对照和历史逐 checkpoint 梯度审计不在本轮范围。每组保存了训练 list、梯度/损失日志、checkpoint、选择记录、候选、排名和证据漏斗。",
        "核验：`DATA_INTEGRITY.json` 检查数据语义；`TRAINING_AUDIT.json` 检查两臂相同初值、实际参数变化、相同数据顺序和优化步数；`FEATURE_COMPOSITION.json` 检查表序列化与编码输入一致。",
        "测试：`cd /tmp && conda run --no-capture-output -n MMDD python -m pytest /home/oycy/MMDD/tests/test_src_dataset_builder.py /home/oycy/MMDD/tests/test_abebooks_ablation.py -q`，26 passed。",
        "完整表格及配对统计见 `COMPARISON.json`，逐 query 得分见各组 `recall_per_query.jsonl`，删素材清单见 `HUB_FILTER.json`，删列及映射见各组 `CURATION.json`。", ""]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines))
    print(json.dumps([r for r in rows if r["split"] == "test" and r["segment"] == "overall"
                      and r["generator"] == "endpoint_sup"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    summarize(parser.parse_args().run_root.resolve())

"""Report fresh training on the source-disjoint, column-pruned AbeBooks dataset."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from mmdd_stage1.data import iter_jsonl, read_json, sha256_file, write_json
from mmdd_stage1.evaluate import paired_bootstrap
from run_abebooks_balanced import SCHEDULES
from summarize_abebooks_balanced import learning_diagnostics, training_audit
from summarize_abebooks_data_ablation import diagnostics

KS = (5, 10, 15, 20)
GENERATORS = ("raw", "selected_sup", "selected_kd", "endpoint_sup", "endpoint_kd")


def summarize(root: Path, previous: Path | None = None) -> None:
    assert read_json(root / "SUITE_COMPLETE.json")["status"] == "COMPLETE"
    repo = Path(__file__).resolve().parents[1]
    for relative, digest in read_json(root / "TRAINING_SOURCE_LOCK.json").items():
        assert sha256_file(repo / relative) == digest, relative
    integrity = read_json(root / "DATA_INTEGRITY.json")
    assert integrity["status"] == "PASS"
    curation = read_json(root / "dataset_view/REGENERATION.json")
    training = training_audit(root)
    learning = learning_diagnostics(root)
    recall, evidence, selections, per_query = [], {}, {}, {}
    for schedule in SCHEDULES:
        run = root / schedule
        recall.extend({"schedule": schedule, **r} for r in read_json(run / "recall_summary.json"))
        evidence[schedule] = {g: diagnostics(run, g) for g in GENERATORS}
        selections[schedule] = read_json(run / "SELECTION_FREEZE.json")
        per_query[schedule] = {(r["generator"], r["mode"], r["query_id"]): r for r in
                               iter_jsonl(run / "recall_per_query.jsonl") if r["split"] == "test"}
    lookup = {(r["schedule"], r["generator"], r["mode"], r["segment"]): r
              for r in recall if r["split"] == "test"}
    primary = "ten_epochs_b16"
    queries = list(iter_jsonl(root / "dataset_view/query_tables/part-00000.jsonl"))
    groups = {q["table_id"]: q["source_table_id"] for q in queries if q["split"] == "test"}
    contrasts = []
    for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
        for k in KS:
            delta = {q: per_query[primary]["endpoint_kd", mode, q][f"R@{k}"] -
                        per_query[primary]["endpoint_sup", mode, q][f"R@{k}"] for q in groups}
            contrasts.append({"mode": mode, "k": k, **paired_bootstrap(delta, groups)})
    historical = []
    if previous is not None:
        old_queries = {q["table_id"]: q for q in iter_jsonl(previous / "dataset_view/query_tables/part-00000.jsonl")}
        cohort_gold = []
        for directory in (previous, root):
            gold = defaultdict(set)
            for row in iter_jsonl(directory / "dataset_view/qrels.jsonl"):
                if row["split"] == "test" and row["reason"] == "model_recoverable_join_column" and row["rel"] > 0:
                    gold[row["query_table_id"]].add(row["target_table_id"])
            cohort_gold.append(dict(gold))
        assert cohort_gold[0] == cohort_gold[1], "Historical implicit test queries/gold targets changed"
        for schedule in SCHEDULES:
            old = {(r["generator"], r["mode"], r["query_id"]): r for r in
                   iter_jsonl(previous / schedule / "recall_per_query.jsonl") if r["split"] == "test"}
            for generator in ("raw", "endpoint_sup", "endpoint_kd"):
                for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                    current = {q: r for (g, m, q), r in per_query[schedule].items()
                               if g == generator and m == mode and r["kind"] == "implicit"}
                    assert all(q in old_queries and (generator, mode, q) in old for q in current)
                    historical.append({"schedule": schedule, "generator": generator, "mode": mode,
                        "same_implicit_queries": sorted(current),
                        "previous": {f"R@{k}": sum(old[generator, mode, q][f"R@{k}"] for q in current) / len(current) for k in KS},
                        "current": {f"R@{k}": sum(r[f"R@{k}"] for r in current.values()) / len(current) for k in KS},
                        "scope": "Same implicit query cohort; lake and training data differ; historical dataset had source overlap"})
    write_json(root / "TRAINING_AUDIT.json", training)
    write_json(root / "LEARNING_DIAGNOSTICS.json", learning)
    write_json(root / "COMPARISON.json", {"curation": curation, "integrity": integrity,
        "recall": recall, "evidence": evidence, "selection": selections, "training": training,
        "learning_diagnostics": learning, "SUP_KD_paired_contrasts": contrasts,
        "historical_implicit_cohort": historical})

    def values(row: dict) -> str:
        return " | ".join(f"{row[f'R@{k}'] * 100:.2f}%" for k in KS)

    def metric(schedule: str, generator: str, mode: str, segment: str = "overall") -> dict:
        return lookup[schedule, generator, mode, segment]

    student_max = max(metric(schedule, generator, mode)[f"R@{k}"]
                      for schedule in SCHEDULES for generator in ("endpoint_sup", "endpoint_kd")
                      for mode in ("Direct", "Multimodal_RRF") for k in KS)
    primary_evidence = evidence[primary]["endpoint_sup"]
    primary_learning = learning[primary]["C2_SUP_endpoint"]["train"]
    findings = [
        f"三组完成训练的 SUP/KD，在 Direct/RRF 四个 cutoff 上的最高 Test Recall 为 {100*student_max:.2f}%。",
        f"主设置 SUP/KD 的终点权重 L2 差为 {training['schedules'][primary]['SUP_KD_endpoint_L2']:.6f}；参数发生了不同的更新，是否有检索收益应以结果表为准。",
        f"主设置 SUP 的训练集精确 Direct R@10 为 {100*primary_learning['R@10']:.2f}%，平均最佳 gold 排名为 {primary_learning['best_gold_rank']:.2f}/{curation['targets']}。拟合诊断也显示问题，不能只归结为测试泛化。",
        f"正确证据在首跳命中 {primary_evidence['any_QE20_hit']}/{primary_evidence['witness_pairs']} 个有证据标注的正例对，最终保留 {primary_evidence['any_D1_retained']}/{primary_evidence['witness_pairs']}；完整目标集合 U 的宏观 gold 覆盖率为 {100*primary_evidence['U_macro_coverage']:.2f}%，截断到 C150 后为 {100*primary_evidence['C150_macro_coverage']:.2f}%。需要同时关注候选保留和最终排序。",
    ]
    lines = ["# AbeBooks 来源互斥数据集：完整训练与 SUP/KD 对照", "",
        *findings, "",
        "使用当前 src，从本轮新编码特征开始完成 TA → TB_CQET → C1 → C2 SUP/KD；seed=13。",
        "主设置为 C1/C2 各 10 轮、batch=16；同时预先固定 1 轮/batch64、10 轮/batch64 两个训练步数对照。",
        "所有组的 checkpoint 选择完成并冻结后才评测 test。主设置未根据 test 结果改变。", "",
        "基础 embedding 和细粒度特征均在本轮新编码。已在本轮保存的素材 token 按现有规则压缩；未复用历史实验特征。",
        "GPU 资源争用处理及特征来源见 ENCODING_RECOVERY.json；CPU 尝试未向最终特征集贡献数据。最终训练在 GPU 1 顺序执行。", "",
        "## 数据与删列核验", "",
        f"{curation['implicit_queries']} 个 implicit、{curation['explicit_queries']} 个 explicit，来源原表交集为 0。",
        f"候选湖 {curation['targets']} 张表；{curation['qrels']} 条正例、{curation['evidence_recoveries']} 条恢复证据、{curation['assets']} 个素材。",
        "此前删除的列保持删除，原表、query、target 及实际编码输入 schema 均已检查。没有重新加入列或旧 explicit target。",
        "删除列：" + "、".join(f"`{c}`" for c in integrity["removed_columns_still_absent"]) + "。", "",
        "| Split | implicit | explicit |", "|---|---:|---:|"]
    for split, counts in curation["split_counts"].items():
        lines.append(f"| {split} | {counts['implicit']} | {counts['explicit']} |")
    lines += ["", "## 主设置 Test Recall：完整训练终点", "",
        "Recall 是逐 query 的 |top-k ∩ gold targets| / |gold targets| 的均值。",
        "Direct 是直接检索；Multimodal_RRF 是多模态检索融合；Teacher_Real 是 Teacher 对同一候选集评分后的诊断。", "",
        "| Student | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---:|---:|---:|---:|"]
    for student in ("sup", "kd"):
        for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
            lines.append(f"| {student.upper()} | {mode} | {values(metric(primary, 'endpoint_' + student, mode))} |")
    lines += ["", "## 主设置：Raw 与 dev 选择的 checkpoint", "",
        "| Generator | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---:|---:|---:|---:|"]
    for generator in ("raw", "selected_sup", "selected_kd"):
        for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
            lines.append(f"| {generator} | {mode} | {values(metric(primary, generator, mode))} |")
    lines += ["", "## implicit / explicit 分开统计（主设置终点）", "",
        "| Student | 类型 | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---|---:|---:|---:|---:|"]
    for student in ("sup", "kd"):
        for segment in ("implicit", "explicit"):
            for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                lines.append(f"| {student.upper()} | {segment} | {mode} | {values(metric(primary, 'endpoint_' + student, mode, segment))} |")
    lines += ["", "## 训练更新与选择", "",
        "| 配置 | TA | TB | C1 | C2 SUP | C2 KD | C1 selected fraction | C2 selected fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for schedule in SCHEDULES:
        audit, selection = training["schedules"][schedule], selections[schedule]
        steps = audit["steps"]
        lines.append(f"| {schedule} | " + " | ".join(str(steps[s]) for s in ("TA", "TB_CQET", "C1", "C2_SUP", "C2_KD")) +
                     f" | {selection['C1_fraction']} | {selection['C2_fraction']} |")
    lines += ["", "逐步核验了记录顺序、每轮样本覆盖、有限 loss/梯度、SUP/KD 相同初始化和 batch 顺序、终点参数实际变化。",
        "各组 Teacher 使用相同数据与 seed 从头训练，最终权重一致。重复 epoch 增加更新次数，不增加独立标注数量。",
        "fraction=0 表示该阶段初始化。dev 选择的结果与完成全部更新后的固定终点分别保留；KD 使用 SUP 选定的同一轮次。", "",
        "## 三组完整终点对照", "",
        "| 配置 | Student | 排序 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---|---:|---:|---:|---:|"]
    for schedule in SCHEDULES:
        for student in ("sup", "kd"):
            for mode in ("Direct", "Multimodal_RRF", "Teacher_Real"):
                lines.append(f"| {schedule} | {student.upper()} | {mode} | {values(metric(schedule, 'endpoint_' + student, mode))} |")
    lines += ["", "## 正确证据与行/属性覆盖（主设置）", "",
        "| Generator | QE20 pair | ET50 pair | D1 pair | QE20 单元 | D1 单元 | C150 目标覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for generator in GENERATORS:
        e = evidence[primary][generator]
        lines.append(f"| {generator} | {e['any_QE20_hit']}/{e['witness_pairs']} | {e['any_ET50_hit']}/{e['witness_pairs']} | "
            f"{e['any_D1_retained']}/{e['witness_pairs']} | {e['QE20_covered_units']}/{e['annotated_row_attribute_value_units']} | "
            f"{e['D1_covered_units']}/{e['annotated_row_attribute_value_units']} | {e['C150_macro_coverage']*100:.2f}% |")
    lines += ["", "这里衡量召回已标注的正确证据及其对应行/属性/值单元。没有重新执行二阶段属性抽取或 join 验证，不能据此宣称新值恢复率或最终 joinability 已提高。", "",
        "| Student | 证据诊断 | R@5 | R@10 | R@15 | R@20 |", "|---|---|---:|---:|---:|---:|"]
    for student in ("sup", "kd"):
        for mode in ("Teacher_Real", "Teacher_f0", "Teacher_Swap"):
            lines.append(f"| {student.upper()} | {mode} | {values(metric(primary, 'endpoint_' + student, mode))} |")
    lines += ["", "Real 使用召回证据，f0 禁用证据评分，Swap 替换证据；这些是检验证据作用的诊断。", "",
        "## 训练集拟合（主设置精确 Direct R@10）", "",
        "| 阶段 | train | dev |", "|---|---:|---:|"]
    for stage, scores in learning[primary].items():
        lines.append(f"| {stage} | {scores['train']['R@10']*100:.2f}% | {scores['dev']['R@10']*100:.2f}% |")
    lines += ["", "## SUP/KD 配对差值", "",
        "主设置终点 KD − SUP；95% 区间按 source group 配对 bootstrap 10,000 次。", "",
        "| 排序 | K | 差值 pp | 95% 区间 pp |", "|---|---:|---:|---|"]
    for contrast in contrasts:
        lines.append(f"| {contrast['mode']} | {contrast['k']} | {contrast['mean_delta_pp']:+.2f} | {contrast['ci_95']} |")
    if historical:
        lines += ["", "## 与上一版相同 implicit 测试 query 的描述性对照", "",
            "旧版存在来源重叠，新版湖从 137 变为 174 张表，训练数据也改变；此表不能单独归因于来源互斥修复。新 explicit query 已更换，因此不做逐 query 配对。", "",
            "| Generator | 排序 | 旧 R@10 | 新 R@10 | 旧 R@20 | 新 R@20 |", "|---|---|---:|---:|---:|---:|"]
        for row in historical:
            if row["schedule"] == primary:
                lines.append(f"| {row['generator']} | {row['mode']} | " + " | ".join(
                    f"{row[version][f'R@{k}']*100:.2f}%" for k in (10, 20) for version in ("previous", "current")) + " |")
    lines += ["", "单 seed，test 共 12 个 query（implicit/explicit 各 6），结论受样本量限制。全部训练配置、正负结果和逐 query 排名均保存。", "",
        "## 复现与文件", "",
        "`DATA_INTEGRITY.json`：删列和新数据输入核验；`TRAINING_SOURCE_LOCK.json`：实际训练源代码哈希。",
        "`VALIDATION.json`：训练相关 26 项单元测试通过，以及特征覆盖、有限数值和模型输入 schema 检查。",
        "`COMPARISON.json`：全部 Recall、证据覆盖、配对差值及历史描述性对照。",
        "`TRAINING_AUDIT.json`、`LEARNING_DIAGNOSTICS.json`：更新次数、参数变化、逐轮 loss 和训练拟合。",
        "各配置目录的 `recall_per_query.jsonl`、`eval/` 保存逐 query 得分、召回素材、候选 target、排序和证据链漏斗。",
        "启动入口：`run_abebooks_fresh.py prepare-data/encode` → `run_abebooks_balanced.py run` → `summarize_abebooks_disjoint.py`。",
        "完整命令保存在根目录及各配置目录的 `COMMANDS.jsonl`；所有模型运行都在隔离工作目录执行。", ""]
    (root / "REPORT.zh-CN.md").write_text("\n".join(lines))
    write_json(root / "COMPLETE.json", {"status": "COMPLETE", "primary_schedule": primary,
        "training_audit": "PASS", "column_integrity": "PASS", "source_lock": "PASS"})
    print("\n".join(lines[lines.index("## 主设置 Test Recall：完整训练终点"):lines.index("## 主设置：Raw 与 dev 选择的 checkpoint")]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    summarize(args.run_root.resolve(), args.previous.resolve() if args.previous else None)

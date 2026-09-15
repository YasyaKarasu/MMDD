"""Export static scientific figures and a Chinese R26 diagnostic report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from run_r26_joinability_diagnostic import OUT, ROOT, R26
from mmdd_stage2.join_diagnostic import write_json


def csv_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def render(out: Path, plot_deps: Path | None) -> None:
    if plot_deps:
        sys.path.insert(0,str(plot_deps))
    os.environ.setdefault("MPLCONFIGDIR",str(out / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((out / "SUMMARY.json").read_text())
    protocol = json.loads((out / "PROTOCOL.json").read_text())
    validation = json.loads((out / "VALIDATION.json").read_text())
    widths = csv_rows(out / "table_width.csv")
    samples = csv_rows(out / "width_subsampling.csv")
    distributions = csv_rows(out / "score_distributions.csv")
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    arms = ["A","B","C13","C29"]
    names = {"T0":"T0 input", "A":"A: cell coverage", "B":"B: frozen column", "C13":"C: projection seed 13", "C29":"C: projection seed 29"}
    colors = {"implicit positive":"#be5b26","implicit nonpositive":"#dcaa76","explicit positive":"#2455a4","explicit nonpositive":"#82a9d4"}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.bbox":"tight"})

    def save(fig, name):
        fig.savefig(figures / f"{name}.png",dpi=180)
        fig.savefig(figures / f"{name}.pdf")
        plt.close(fig)

    fig,axes = plt.subplots(2,2,figsize=(13,9),layout="constrained")
    for ax,arm in zip(axes.flat,arms):
        for kind in ("implicit","explicit"):
            for positive in ("True","False"):
                subset = [r for r in widths if r["arm"] == arm and r["kind"] == kind and r["positive"] == positive]
                label = kind + (" positive" if positive == "True" else " nonpositive")
                ax.scatter([int(r["column_pairs"]) for r in subset],[float(r["score"]) for r in subset],
                           s=17 if positive == "True" else 8,alpha=.8 if positive == "True" else .18,c=colors[label],label=label)
        ax.set(xscale="log",xlabel="Visible query columns x target columns",ylabel="Maximum coverage" if arm == "A" else "Maximum cosine",title=names[arm])
        ax.legend(fontsize=8,loc="lower right")
    fig.suptitle("R26 C18 union + missed gold targets: width association (observational)")
    save(fig,"table_width_bias")

    # A common set of pairs with >=16 opportunities avoids changing populations across budgets.
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout="constrained")
    subsample_summary = []
    for ax,arm in zip(axes.flat,arms):
        for kind in ("implicit","explicit"):
            for positive in ("True","False"):
                subset = [r for r in samples if r["arm"] == arm and r["kind"] == kind and r["positive"] == positive and int(r["available_pairs"]) >= 16 and int(r["budget"]) <= 16]
                groups = defaultdict(list)
                for r in subset:
                    groups[int(r["budget"]),r["query_id"]].append(float(r["score"]))
                by_budget = defaultdict(list)
                for (budget,qid), vals in groups.items():
                    by_budget[budget].append(float(np.mean(vals)))
                xs = sorted(by_budget)
                label = kind + (" positive" if positive == "True" else " nonpositive")
                ax.plot(xs,[np.mean(by_budget[x]) for x in xs],"o-",label=label,c=colors[label])
                for x in xs:
                    subsample_summary.append({"arm":arm,"kind":kind,"positive":positive == "True","budget":x,"queries":len(by_budget[x]),"mean":float(np.mean(by_budget[x]))})
        ax.set(xlabel="Random nested column-pair budget",ylabel="Mean maximum coverage" if arm == "A" else "Mean maximum cosine",title=names[arm],xticks=[1,2,4,8,16])
        ax.legend(fontsize=8)
    fig.suptitle("Within-table-pair opportunity control; same >=16-pair cohort; 16 permutations; query balanced")
    save(fig,"controlled_max_bias")
    write_json(out / "controlled_max_bias.json",subsample_summary)

    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,arm in zip(axes.flat,arms):
        for positive,color in (("True","#2455a4"),("False","#be5b26")):
            vals = [float(r["score"]) for r in distributions if r["arm"] == arm and r["split"] == "eval" and r["positive"] == positive]
            ax.hist(vals,bins=20,alpha=.55,density=True,label="Gold column" if positive == "True" else "Other column",color=color)
        ax.set(title=names[arm],xlabel="Coverage + bounded mean tie-break" if arm == "A" else "Cosine",ylabel="Density")
        ax.legend()
    fig.suptitle("Offline known-join-column diagnostic; implicit inputs use oracle source values ONLY here")
    save(fig,"column_score_distributions")

    fig,axes = plt.subplots(1,3,figsize=(15,4.5),layout="constrained",sharey=True)
    for ax,condition in zip(axes,("Real-crop","Real-crop+original","NoE-fill")):
        rs = {r["arm"]:r for r in summary["replay"] if r["generator"] == "B13" and r["condition"] == condition and r["kind"] == "overall"}
        order = ["T0",*arms]
        bars = ax.bar(order,[rs[a]["R9"]*100 for a in order],color=["#666666","#be5b26","#2455a4","#399478","#87b69a"])
        ax.bar_label(bars,fmt="%.2f",fontsize=9)
        ax.set(title=condition,ylabel="Macro Recall@9 (%)",ylim=(0,max(rs[a]["R9"]*100 for a in order)+9))
    fig.suptitle("B13 fixed C18 replay, 64 queries: only verifier function changes")
    save(fig,"b13_replay")

    lengths = csv_rows(out / "target_length_subsampling.csv")
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    length_summary = []
    for ax,metric in zip(axes,("coverage","raw_mean_max_cosine")):
        for kind in ("implicit","explicit"):
            for positive in ("True","False"):
                subset = [r for r in lengths if r["kind"] == kind and r["positive"] == positive and int(r["target_rows"]) >= 20 and int(r["budget"]) <= 20]
                groups = defaultdict(list)
                for r in subset:
                    groups[int(r["budget"]),r["query_id"]].append(float(r[metric]))
                by_budget = defaultdict(list)
                for (budget,qid),vals in groups.items():
                    by_budget[budget].append(float(np.mean(vals)))
                xs = sorted(by_budget)
                label = kind + (" positive" if positive == "True" else " nonpositive")
                ax.plot(xs,[np.mean(by_budget[x]) for x in xs],"o-",label=label,c=colors[label])
                for x in xs:
                    length_summary.append({"kind":kind,"positive":positive == "True","budget":x,"metric":metric,"queries":len(by_budget[x]),"mean":float(np.mean(by_budget[x]))})
        ax.set(xlabel="Nested target-cell budget",ylabel=metric,title="A: same selected column pair; >=20 target rows")
        ax.legend(fontsize=8)
    fig.suptitle("Target-length opportunity control; hash-selected table pairs; 8 permutations; query balanced")
    save(fig,"controlled_target_length")
    write_json(out / "controlled_target_length.json",length_summary)

    main = {r["arm"]:r for r in summary["replay"] if r["generator"] == "B13" and r["condition"] == "Real-crop" and r["kind"] == "overall"}
    lines = ["# R26 joinability function P0 diagnostic", "", "## 核心结果", "",
        "本实验复用 R26 候选、列选择和逐行生成值，没有训练完整 Stage2，没有重新生成 value。所有 arm 固定同一 C18、Stage1/T0 分数、recovery budget 和 branch eligibility。", "",
        "### B13 Real-crop：64 query 固定 replay", "", "| Arm | R@1 | R@5 | R@9 | R@18 | Top9 evidence 数 | ΔR@9 vs Current（95% CI） |", "|---|---:|---:|---:|---:|---:|---|"]
    for arm in ("T0",*arms):
        r = main[arm]
        d = r["delta_vs_A_R9"]
        lines.append(f"| {names[arm]} | {r['R1']*100:.2f}% | {r['R5']*100:.2f}% | {r['R9']*100:.2f}% | {r['R18']*100:.2f}% | {r['evidence_top9']} | {d['delta']*100:+.2f} pp [{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}] |")
    lines += ["", "CI 按 source_table_id 聚类、paired bootstrap 3,000 次；报告两 seed，不从 pilot 挑最佳 seed。", "",
        f"本次主结果：A={main['A']['R9']*100:.2f}%，B={main['B']['R9']*100:.2f}%，训练后 C13={main['C13']['R9']*100:.2f}%，训练后 C29={main['C29']['R9']*100:.2f}%。轻量训练没有超过 frozen B；B 对 A 的提升95% CI跨0，不能作为已确认的总体收益。",
        "B 的 Real-crop 与 NoE-fill R@9 相同，本次未证实更换 scorer 后的检索增益由新增证据正确补值造成。未重跑 dynamic RRF；这里是其列分数代理风险的诊断，不是 dynamic fusion 修复结论。", "",
        "**解释边界：** B 的 R@9 相对 A 有部分恢复，但其 R@1 更低；不要只报告 R@9。已知正确 query join column 的离线任务接近 ceiling，不能代表真实 recovered values 的质量。换 score 也无法召回 C18 之外的目标。", "",
        "**训练 checkpoint 实现纠正：** 初版 selector 把 epoch 0 纳入 C 的候选，而两个随机投影在 calibration MRR 上恰好等于1，导致并列规则选中未训练模型。本报告主 C 只在 epochs 1..30 选择，保留全部同 seed/超参数训练；原 epoch0 结果在 archive_epoch0_selection，随机投影 checkpoint 独立保留。此纠正落实‘trained C’定义，没有按 pilot 选择训练 epoch。", "",
        "随机投影对照（原 epoch0 选择结果）：", "", "| 对照 | B13 Real-crop R@9 |", "|---|---:|"]
    initial_path = out / "archive_epoch0_selection/SUMMARY.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else {"replay":[]}
    for r in initial["replay"]:
        if r["generator"] == "B13" and r["condition"] == "Real-crop" and r["kind"] == "overall" and r["arm"].startswith("C"):
            lines.append(f"| random {r['arm']} | {r['R9']*100:.2f}% |")
    lines += ["",
        "## Scorer 本身：正确列识别", "", "### 给定真实 query join column 的离线诊断", "",
        "implicit 的 join column 从对应 source rows 构造，只用于 train/calibration 和这一离线诊断；该 oracle representation 从未进入 replay。", "",
        "| 子集 | Arm | GT 表对数 | Top1 | MRR | 表内 AUC |", "|---|---|---:|---:|---:|---:|"]
    for kind in ("overall","implicit","explicit"):
        for arm in arms:
            r = summary["column_metrics"]["eval"][kind][arm]
            lines.append(f"| {kind} | {arm} | {r['pairs']} | {r['top1']*100:.2f}% | {r['mrr']:.4f} | {r['auc']:.4f} |")
    lines += ["", "Top1/MRR 对并列分数随机打散取期望，避免 GT 恰好在第一列带来的偏差。AUC 是每个 GT 表内 gold 对其他列的宏平均；不是全库检索 AUC。", "",
        "### 未提供 query join column：对全部 visible query columns 取 max", "", "| 子集 | Arm | GT 表对数 | Target gold-column Top1 | MRR |", "|---|---|---:|---:|---:|"]
    for r in summary["visible_max_column_metrics"]:
        lines.append(f"| {r['kind']} | {r['arm']} | {r['pairs']} | {r['top1']*100:.2f}% | {r['mrr']:.4f} |")
    lines += ["", "implicit 这里的 gold attribute 不可见，因此此项只是检查 incidental visible similarity 是否会选中 gold target column，不能当作恢复能力。", "",
        "### 使用原始实际 recovered values", "", "| 条件（B13） | Arm | GT 表对数 | 列 Top1 | MRR | 原预测列正确率 | 平均非空行 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["recovered_column_metrics"]:
        if r["generator"] == "B13":
            lines.append(f"| {r['condition']} | {r['arm']} | {r['pairs']} | {r['top1']*100:.2f}% | {r['mrr']:.4f} | {r['predicted_column_correct']*100:.2f}% | {r['nonempty_rows']:.2f} |")
    lines += ["", "该诊断比较 recovered column 对 GT target 所有列的评分；replay 仍固定原 predicted column，没有用这项诊断重选列。", "",
        "## False-direct proxy、exact match 与阈值", "", "| 评测范围 | Arm | 表对数 | 高 direct 接受率 |", "|---|---|---:|---:|"]
    for scope, results in summary["false_direct"].items():
        for arm,r in results.items():
            lines.append(f"| {scope} | {arm} | {r['pairs']} | {r['accept_rate']*100:.2f}% |")
    lines += ["", "A 固定 coverage≥0.6 / cosine≥0.8；B/C 的 cosine threshold 只在 explicit calibration true-column positives 上定到至少95%接受率。两类 scorer 没有强行共用0.8。这个表是接受率 proxy，不是人工 adjudication 的 FPR：implicit 构造标签不证明不存在额外真实 direct join，qrels 也不穷尽所有可连接的 context columns。", "",
        "B/C 阈值：`"+json.dumps(summary["thresholds"])+"`。", "",
        "A 在 implicit GT 表对中的高分 winner exact coverage≥0.6 比例："+f"{summary['false_direct']['implicit_positive_all']['A']['exact_ge_06']*100:.2f}%"+"。", "",
        "| A cosine threshold | implicit GT 表对 direct 接受率 |", "|---:|---:|"]
    for r in summary["threshold_sensitivity"]:
        lines.append(f"| {r['cosine_threshold']:.2f} | {r['false_direct_proxy']*100:.2f}% |")
    lines += ["", "各 arm 的 explicit calibration true-column 实际接受率：`"+json.dumps(validation["calibration_explicit_tpr"])+"`。", "",
        "### 严格 Direct100-missed implicit GT（仅离线归因标签）", "", "| Generator | Arm | 表对数 | 假设计算 direct score 时的接受率 |", "|---|---|---:|---:|"]
    for r in validation["strict_evidence_only"]:
        lines.append(f"| {r['generator']} | {r['arm']} | {r['strict_evidence_only_gt_pairs']} | {r['hypothetical_direct_accept_rate']*100:.2f}% |")
    lines += ["", "该定义是 implicit GT 且不在本 generator 原 Direct100 中；没有借此在 replay 开启 direct branch，也不把‘Direct100 没找到’等价成绝无 direct join。", "",
        "| Generator（Real-crop） | Arm | Strict GT 总数 | 其中 C18 可达 | T0 R9 hits | Replay R9 hits |", "|---|---|---:|---:|---:|---:|"]
    for r in validation["strict_replay"]:
        if r["condition"] == "Real-crop":
            lines.append(f"| {r['generator']} | {r['arm']} | {r['strict_pairs']} | {r['strict_in_C18']} | {r['strict_input_R9']} | {r['strict_hits_R9']} |")
    lines += ["", "## 表宽与 max bias", "", "![Table width](figures/table_width_bias.png)", "", "![Controlled maximum](figures/controlled_max_bias.png)", "",
        "第一张是观察相关性；不能仅凭相关性声称因果。第二张对同一 query-target 表对随机抽取 nested 列对集合，在所有预算下固定 ≥16 列对的共同群体；每对16次排列，先 query 内平均再跨 query 平均。max 随机会数增加的抬升是算子性质及其在本数据上的实际幅度；仍不等于单独解释所有 Recall 损失。", "",
        "| Arm | 类型 | Positive | N | Spearman(列对数,max score) |", "|---|---|---|---:|---:|"]
    for r in summary["width_correlations"]:
        rho = "NA (constant)" if r["spearman"] is None else f"{r['spearman']:.4f}"
        lines.append(f"| {r['arm']} | {r['kind']} | {r['positive']} | {r['pairs']} | {rho} |")
    lines += ["", "![Score distributions](figures/column_score_distributions.png)", "", "![B13 replay](figures/b13_replay.png)", "",
        "![Target length](figures/controlled_target_length.png)", "",
        "Target-length 图固定 A 原全列对搜索选中的列对，再随机增加 target cells；每类最多64表对，按固定 hash 取样，8次 nested 排列。绘图各预算固定 target rows≥20 群体。它展示长度机会效应，不把真实 exact matches 的增加都解释为假阳性。", "",
        "## 证据机制：分支选择变化与正确恢复值分开看", "", "| Generator（Real-crop） | Arm | E Top9 | 其中 GT | 其中预测 gold 列正确 | source-correct gold cells | 相比 NoE 新增正确 cells | 有新增正确值的 GT 表 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in validation["evidence_attribution"]:
        if r["condition"] == "Real-crop":
            lines.append(f"| {r['generator']} | {r['arm']} | {r['evidence_top9']} | {r['qrel_positive_E_top9']} | {r['correct_gold_column_E_top9']} | {r['source_correct_gold_cells_E_top9']} | {r['additional_correct_vs_NoE_E_top9']} | {r['qrel_positive_E_with_any_additional_correct_cell']} |")
    lines += ["", "这里只对 GT target 且所选 attribute 等于 gold attribute 时，将原 recovered row 与 source-canonical query-row value 对齐检查。其他属性或非 GT target 不自动算错，也不声称已确认 grounding。新增正确 cells 是相对原 NoE 同行预测的差异；下表的 Real/NoE 最终 Recall 对照才检验检索净效应。更多 evidence branch 被选中本身不是证据机制成功。", "",
        "## 全部固定输入 replay", "", "| Generator | 条件 | 子集 | Arm | N | R@9 | E in Top9 |", "|---|---|---|---|---:|---:|---:|"]
    for r in summary["replay"]:
        lines.append(f"| {r['generator']} | {r['condition']} | {r['kind']} | {r['arm']} | {r['queries']} | {r['R9']*100:.2f}% | {r['evidence_top9']} |")
    lines += ["", "## 复现与边界", "",
        "- 冻结输入：512 train queries、128 train-calibration queries、64 R26 pilot queries；query source groups 无重叠。train/cal 各类别按 source group 去重、固定 hash 选择，标签来自原 qrels。",
        "- C 为共享 4096→256 无 bias 线性投影，1,048,576 参数，两个 seed、30 epoch、AdamW lr=0.001、temperature=0.07、batch=64；只按 source-balanced calibration MRR 在已训练 epochs 1..30 选 checkpoint。GT 显式列或训练侧 source-canonical hidden 列作为训练 anchor；没有解冻 Qwen。",
        "- MNRL 合并已知同 anchor 多正例；同 source 的其他 batch targets 屏蔽；hard wrong columns 最多4个，按 frozen cosine 排，排除 exact coverage≥0.6 的可能真 join。跨 source 负例仍可能存在不完整标注的问题。",
        "- A→B 同时改变 encoder（Qwen3.5→Qwen3-VL）、表示粒度和 target value budget（全列→前5值），因此不是只隔离 cell vs column 的单因素实验；B→C 才只增加 learned projection。",
        "- 实际 recovered column 用原预测列名和原所有 query rows（空值保留）序列化，前5行与已缓存列 recipe 一致；全空列记0，避免只有列名的伪支持；否则 B/C 原始 cosine 不按非空比例额外缩放。",
        "- 原 R26 B13 全部192记录成功；Qwen-Raw 原有1条 Real-crop 失败，各新 arm 同样保留空排名并计入64-query分母，没有重生成修复。",
        "- A replay 使用原始缓存的 ranking / branch score；补算 A 只用于列诊断，避免去重 batching 引起 bf16 tie 漂移污染 baseline。",
        "- 检索实验规模仍是64 query pilot，不支持全量或跨域结论。原始 recovered value 的内容、行覆盖和独立正确值审计均固定；更换打分不会产生新证据或新正确值。",
        "- 数值诊断共1,635个唯一 query-target 表对，包含两 generator 的C18并集及离线补入的missed GT；后者没有加入 replay。目标宽度、类型、正负例分布并非随机分配，观察相关性仅作描述。",
        "- 编码补算在2×RTX4090并行完成：新5,300条去重column text（5,345向量）约82秒，53,508个去重cell约172秒；这些是模型加载后编码计时，不是完整线上延迟。",
        "", "A 数值复算审计：", "", "```json", json.dumps(summary["cell_recompute_parity"],indent=2), "```", "",
        "所选 checkpoint：", "", "```json", json.dumps(summary["checkpoints"],indent=2), "```", "",
        "运行命令（从 /tmp 隔离工作目录，MMDD conda）：", "", "```bash",
        f"conda run -n MMDD python {ROOT}/src/run_r26_joinability_diagnostic.py prepare",
        f"conda run -n MMDD python {ROOT}/src/run_r26_joinability_diagnostic.py columns --device cuda:0 --batch-size 16",
        f"conda run -n MMDD python {ROOT}/src/run_r26_joinability_diagnostic.py cells --device cuda:1 --batch-size 64",
        f"conda run -n MMDD python {ROOT}/src/run_r26_joinability_diagnostic.py fit --device cuda:0 --seed 13",
        f"conda run -n MMDD python {ROOT}/src/run_r26_joinability_diagnostic.py fit --device cuda:1 --seed 29",
        f"conda run -n MMDD python {ROOT}/src/analyze_r26_joinability_diagnostic.py",
        f"conda run -n MMDD python {ROOT}/src/audit_r26_joinability_diagnostic.py",
        f"conda run -n MMDD python {ROOT}/src/report_r26_joinability_diagnostic.py --plot-deps /tmp/r26_diagnostic_plotdeps",
        f"conda run -n MMDD python -m pytest {ROOT}/tests/test_r26_joinability_diagnostic.py {ROOT}/tests/test_stage2_verifier.py -q",
        "```", "", "机器可读结果：SUMMARY.json、replay_per_query.csv、replay_rankings.jsonl、column_metrics.csv、column_pairs.csv、table_width.csv、width_subsampling.csv、cell_parity.csv；figures 内提供 PNG/PDF。"]
    (out / "REPORT.zh-CN.md").write_text("\n".join(lines)+"\n")
    selected = [out / name for name in ("PROTOCOL.json","SUMMARY.json","VALIDATION.json","prepared.json","column_encoding_receipt.json","cell_encoding_receipt.json","fit_seed13.json","fit_seed29.json","projection_seed13.pt","projection_seed29.pt")]
    selected.extend(ROOT / "src" / name for name in ("run_r26_joinability_diagnostic.py","analyze_r26_joinability_diagnostic.py","audit_r26_joinability_diagnostic.py","report_r26_joinability_diagnostic.py","mmdd_stage2/join_diagnostic.py"))
    selected.append(ROOT / "tests/test_r26_joinability_diagnostic.py")
    selected.extend(R26 / f"stage2/{folder}/{g}/{file}" for g in ("B13","Qwen-Raw") for folder,file in (("inputs","retrieval.jsonl"),("pilot","results.jsonl")))
    manifest = [{"path":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected]
    write_json(out / "ARTIFACT_MANIFEST.json",manifest)
    print(str(out / "REPORT.zh-CN.md"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--plot-deps",type=Path)
    args = parser.parse_args()
    render(args.output,args.plot_deps)

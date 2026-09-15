"""Write the completed Stage1 controlled comparisons without closing full R26."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from prepare_stage1_r26 import OUT, file_record
from run_stage1_r21 import read_rows
from run_stage1_r25 import _json, sha256

KS = (10,20,30,40,50)
FAMILIES = ("R25-B13-FULL","R25-SPLIT-SUP","R25-EDGE-CONT","R26-O-NATIVE","R26-O-SUP","R26-E-GRAPH",
            "R25-SPLIT-QTKD","R25-SPLIT-U","R25-SPLIT-UQTKD","R25-LSE-QTKD",
            "R26-O-QTKD","R26-O-U","R26-O-UQTKD","R26-O-LSE-QTKD")
COMPARISONS = ("order_native","order_sup","path_vs_edge","R26_added_QTKD","R26_added_U",
               "R26_U_added_QTKD","R26_QTKD_added_U","R26_LSE_vs_split")


def run() -> dict:
    stats = OUT / "statistics"
    report_path = stats / "stage1_recall/REPORT_RECEIPT.json"
    receipt = json.loads(report_path.read_text())
    if receipt["missing"]:
        raise ValueError("Complete every Stage1 reporting module before writing the full findings")
    source_path = Path(receipt["metrics"]["path"])
    if sha256(source_path) != receipt["metrics"]["sha256"]:
        raise ValueError("Stage1 reporting input changed")
    modules = {(r["generator"],r["module"]):r["metrics"] for r in json.loads(source_path.read_text())}
    paired_path = stats / "paired_source_bootstrap.jsonl"
    paired = list(read_rows(paired_path))
    post_path = stats / "postprocessing/POSTPROCESSING_RECEIPT.json"
    post = json.loads(post_path.read_text())
    if post["missing"]:
        raise ValueError("Complete all fusion/Teacher postprocessing before findings")
    for record in post["files"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Postprocessing source changed")
    factorial = list(read_rows(stats / "postprocessing/teacher_2x2.jsonl"))
    if any(r["status"] != "ran" for r in factorial):
        raise ValueError("Incomplete actual KD by online-T0 factorial")
    groups = {"Qwen-Raw":["Qwen-Raw"],"PCA":["PCA"],"pre-B13-C1":["pre-B13-C1"],"B13":["B13"],
              "N-U":[f"N-U/seed{s}" for s in (13,29)],
              "R25-C1":[f"R25-C1/seed{s}/step659" for s in (13,29)],
              **{family:[f"{family}/seed{s}/step178" for s in (13,29)] for family in FAMILIES}}

    def value(group: str, module: str, kind: str, scorer: str, metric: str) -> float:
        return mean(modules[g,module][kind][scorer][metric] for g in groups[group])

    def pp(number: float) -> str:
        return f"{100*number:.2f}"

    def contrast(row: dict) -> str:
        lower,upper = row["bootstrap_95ci"]
        return f"{100*row['mean_delta']:+.2f} [{100*lower:+.2f}, {100*upper:+.2f}]"

    def selected(name: str,kind: str,metric: str) -> dict:
        return next(r for r in paired if r["comparison"] == name and r["kind"] == kind and r["metric"] == metric)

    endpoint_rows = []
    for group,names in groups.items():
        endpoint_rows.append({"group":group,"generators":names,
            "overall_QT_recall":{str(k):value(group,"rankings","overall","QT_OVER_U",f"recall@{k}") for k in KS},
            "implicit_QT_R10":value(group,"rankings","implicit","QT_OVER_U","recall@10"),
            "U_admission":value(group,"rankings","overall","QT_OVER_U","raw_recall"),
            "scoped_modules":{name:{module:modules[name,module] for module in ("rankings","fusion","teacher")} for name in names}})
    lines = ["# R26 Stage1：完整固定端点与受控比较","",
        "本报告覆盖全部基线与扩展模型的 Stage1 评估；Student→Teacher 反馈续训属于后续条件模块，完整 R26 尚未因此完成。",
        "",
        "Stage1 主口径为 query-macro R@10/20/30/40/50，固定1198个 dev query（599 implicit、599 explicit），全湖22886表。两seed的结果先在同query内平均，再按1000个source cluster做10000次bootstrap，seed260914。表中数值为%，差为百分点；CI未作多重比较校正，全部属于探索性dev分析。历史B13只有一个真实checkpoint，未虚构第二个seed。",
        "",
        "## 1. 自身检索的最终端点","",
        "每个Student使用自己的索引与D/E/U。下表为实际自身U上的QT排序；C2固定第178步，C1固定第659步，N-U沿用既有1318步终点。PCA和pre-B13-C1是初始化/父模型控制；没有按dev选择最佳checkpoint。U admission为候选全集覆盖率，并非TopK排名质量。每seed原值、三桶和全部scorer见配套JSON及stage1_recall/metrics.csv。","",
        "| Generator / two-seed mean | R10 | R20 | R30 | R40 | R50 | Implicit R10 | U admission |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in endpoint_rows:
        values = [row["overall_QT_recall"][str(k)] for k in KS]+[row["implicit_QT_R10"],row["U_admission"]]
        lines.append("| "+row["group"]+" | "+" | ".join(map(pp,values))+" |")
    lines += ["","## 2. 顺序干预、相同图的监督层级与扩展目标","",
        "下列差均为命名比较的左臂减右臂。order_native=混排native−顺序native；order_sup=混排SUP−顺序SUP；path_vs_edge=O-SUP−E-GRAPH。其它为固定同父模型、同图和同混排的目标消融。新增C2 KD与Uniform构成2×2；共同C1已使用Teacher，不能称SUP全流程无Teacher。","",
        "| Contrast | Overall QT R10 Δ [95% CI] | Seed13 / Seed29 Δ | Implicit QT R10 Δ [95% CI] | Implicit EO admission Δ [95% CI] |",
        "|---|---|---|---|---|"]
    for name in COMPARISONS:
        overall = selected(name,"overall","QT_OVER_U/recall@10")
        lines.append(f"| {name} | {contrast(overall)} | {pp(overall['seed_deltas']['13'])} / {pp(overall['seed_deltas']['29'])} | "+
                     contrast(selected(name,"implicit","QT_OVER_U/recall@10"))+" | "+contrast(selected(name,"implicit","EO_ANN"))+" |")
    for name in ("order_native","order_sup"):
        row = selected(name,"overall","QT_OVER_U/recall@10")
        direction = "均下降" if all(v<0 for v in row["seed_deltas"].values()) else "均上升" if all(v>0 for v in row["seed_deltas"].values()) else "方向不一致或有零差"
        lines += ["",f"{name} 两seed的Overall R10{direction}，均值差 {contrast(row)}。此对照只支持实际测到的顺序干预效果，不能凭训练后段没有E正例推断检索机制已改善。"]
    path_edge = selected("path_vs_edge","overall","QT_OVER_U/recall@10")
    qtkd = selected("R26_added_QTKD","overall","QT_OVER_U/recall@10")
    uniform = selected("R26_added_U","overall","QT_OVER_U/recall@10")
    lse = selected("R26_LSE_vs_split","overall","QT_OVER_U/recall@10")
    lines += ["",f"同图监督层级比较 O-SUP−E-GRAPH 为 {contrast(path_edge)}。该对照的置信区间包含零时，不据此宣称path监督优于Edge；仅报告实际方向。",
              "",f"在SUP上新增QT-KD为 {contrast(qtkd)}，新增Uniform为 {contrast(uniform)}；固定QT Teacher下LSE−Split为 {contrast(lse)}。这里测量的是新增C2目标对自身U上QT R10的作用；零差不表示参数或所有排名相同，也不等同于所有机制指标无变化。"]
    lines += ["","## 3. 固定89→178轨迹与证据保留","",
        "历史顺序图末88个batch的E正例active总数为0，混排后为2825；这说明监督暴露不同，不等于后半段没有E候选。下表同时测真实own-E admission和排序变化。E-only/Equal/Conf/Column的E排名沿用历史D1覆盖率策略，选择质量含sigmoid(实际path score)；自然fused-LSE使用保留前全部有效path logits。两种语义不能混称。","",
        "| Fixed trajectory178−89 | Overall QT R10 Δ [95% CI] | Implicit QT R10 Δ [95% CI] | Implicit EO Δ [95% CI] |",
        "|---|---|---|---|"]
    for family in ("R25-B13-FULL","R25-SPLIT-SUP","R26-O-NATIVE","R26-O-SUP","R26-E-GRAPH","R26-O-QTKD","R26-O-U","R26-O-UQTKD","R26-O-LSE-QTKD"):
        name = family+"_trajectory_178_minus_89"
        lines.append("| "+family+" | "+" | ".join(contrast(selected(name,kind,metric)) for kind,metric in (
            ("overall","QT_OVER_U/recall@10"),("implicit","QT_OVER_U/recall@10"),("implicit","EO_ANN")))+" |")
    lines += ["","EO为G∩(E−D100_ANN)的macro admission；另存exact-D100、等规模M及固定raw-exact预算外集合。到达正确target的path不自动成为正确witness；完整EO集合及各scorer Top10/20/30/40/50保留见EO_sets_and_scorer_retention.jsonl.gz。","",
        "## 4. 真实D/E输入的零训练融合","",
        "同一own-U内比较，D-only为Direct100；Column使用真实列向量与固定train256自然Raw-U的query等权CDF。缺列alpha=1，不用dev拟合；列名/类型可靠性不足时取所有可见列的max cosine，generic/ID列可能主导。每对candidate继承query桶标签，并非独立pair-type标注。","",
        "| Generator / mean | Direct R10 | Equal R10 | Conf R10 | Column R10 | Column implicit R10 | Column explicit R10 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [value(group,"fusion","overall",m,"recall@10") for m in ("Direct-only","Equal","Conf","Column")]
        values += [value(group,"fusion",kind,"Column","recall@10") for kind in ("implicit","explicit")]
        lines.append("| "+group+" | "+" | ".join(map(pp,values))+" |")
    lines += ["","Cq/alpha分布、分箱、EO及strict-outside alpha见fusion各receipt和postprocessing/fusion_bins.jsonl。AUC仅描述query桶或qrel-positive对unlabeled candidate，不能当真负例/真假连接分类。融合无法创造U外目标。","",
        "## 5. 固定T0与新增C2 KD的作用","",
        "BT100=各自own-U→真实D/E Equal→取100；同C100的无T0/有T0对照只改变排序。D100→T0使用独立direct候选池；U/M全池T0是离线组成诊断。QT Teacher没有验证path内容。","",
        "| Generator / mean | BT100 no-T0 R10 | BT100 T0 R10 | D100 T0 R10 | U T0 R10 (offline) | M T0 R10 (offline) |",
        "|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        values = [value(group,"teacher","overall",m,"recall@10") for m in ("BT100_NO_T0","BT100_T0","D100_T0","U_OFFLINE_T0","M_OFFLINE_T0")]
        lines.append("| "+group+" | "+" | ".join(map(pp,values))+" |")
    lines += ["","| Added C2 KD contrast | Scorer / interaction | Seed13 / Seed29 R10 Δ | Mean Δ [95% CI] |","|---|---|---|---|"]
    for row in factorial:
        if row["k"] == 10 and row["query_kind"] == "overall":
            lines.append(f"| {row['version']} {row['kd']}−{row['control']} | {row['scorer']} | {pp(row['seed_deltas']['13'])} / {pp(row['seed_deltas']['29'])} | {contrast(row)} |")
    lines += ["","interaction=(KD效应有在线T0)−(KD效应无在线T0)。KD跨臂比较包含各Student自己的候选组成变化；同一臂±在线T0才是严格同池排序比较。其它K、分桶及每个自身池T0收益与U−M的配对CI见postprocessing。B13 Teacher后融合扩展见../teacher_fusion/B13/RESULTS.md。","",
        "## 6. 证据补值机制及剩余范围","",
        "Stage2严格按用户指定C18与最终R1/3/5/7/9。Raw/B13各64query、crop/crop+original/NoE三条件全部保留；Raw crop有1次条件失败且留在分母。三条件Recall相同；实际删除所有独立source-exact新增值（Raw16、B13 13）也不改变Recall。Direct100预算外经E进入C18的正目标Raw0/B13 1，尚无独立确证新正确cell把它连接成完整证据链。当前结果不能证明evidence补属性促成新join；grounding仍unknown。详见stage2/RESULTS.md。","",
        "训练侧反馈、条件Teacher续训及是否再蒸馏按实际门值独立完成；本Stage1报告不缩减这些范围。延迟是本机4090与本地冻结feature条件下实测，分别报告Student检索、T0评分、Stage2生成；不把不同query分布的p50相加称端到端p50。完整成本与最终验收另交付。"]
    destination = stats / "STAGE1_FINDINGS.md"
    destination.write_text("\n".join(lines)+"\n")
    result = {"execution_status":"ran","scientific_validity":"valid_for_completed_Stage1",
              "full_R26_status":"in_progress","endpoint_groups":endpoint_rows,"paired_comparisons":paired,"KD_online_factorial":factorial,
              "inputs":{"requested_K_report":file_record(report_path),"metrics":file_record(source_path),
                        "paired":file_record(paired_path),"postprocessing":file_record(post_path)},
              "report":file_record(destination),"code":file_record(Path(__file__))}
    _json(stats / "STAGE1_FINDINGS.json",result)
    return {"endpoint_groups":len(endpoint_rows),"paired_comparisons":len(paired),"report":str(destination)}


if __name__ == "__main__":
    print(json.dumps(run()))

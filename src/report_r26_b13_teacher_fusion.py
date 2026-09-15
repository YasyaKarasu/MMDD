"""Source-cluster uncertainty and a readable report for the user's B13 extension."""
from __future__ import annotations

import json
import numpy as np

from mmdd_stage1.r26_metrics import query_metrics
from mmdd_stage1.r26_statistics import source_cluster_comparison
from prepare_stage1_r26 import OUT,file_record
from run_stage1_r21 import read_rows,write_rows
from run_stage1_r25 import _json
from report_stage1_r26_recall import KS,summarize


def run() -> dict:
    directory = OUT / "teacher_fusion/B13"
    receipt_path = directory / "TEACHER_FUSION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["column_status"] != "ran":
        raise ValueError("The requested Column comparison is incomplete")
    rows = list(read_rows(directory / "rankings.jsonl.gz"))
    requested_metrics = {}
    for pool in ("Equal_BT100","natural_U_offline"):
        selected = [r for r in rows if r["pool_protocol"] == pool]
        requested_metrics[pool] = summarize(selected,{r["query_id"]:r for r in selected})
    _json(directory / "metrics_R10_20_30_40_50.json",requested_metrics)
    comparisons = []
    methods = ["B13_QT_only","T0_D_T0_E/Equal","T0_D_T0_E/Conf","T0_D_T0_E/Column"]
    for pool in ("Equal_BT100","natural_U_offline"):
        for kind in ("overall","implicit","explicit"):
            subset = [r for r in rows if r["pool_protocol"] == pool and (kind == "overall" or r["query_kind"] == kind)]
            for method in methods:
                for k in KS:
                    delta = np.array([query_metrics(r["rankings"][method],r["positive_target_ids"],(k,))[f"recall@{k}"]-
                                      query_metrics(r["rankings"]["T0_only"],r["positive_target_ids"],(k,))[f"recall@{k}"] for r in subset])
                    comparisons.append({"pool":pool,"kind":kind,"left":method,"right":"T0_only","k":k,
                        **source_cluster_comparison(delta,[r["source_table_id"] for r in subset])})
    write_rows(directory / "paired_source_bootstrap.jsonl",comparisons)
    teacher = json.loads((OUT / "teacher/B13/metrics.json").read_text())
    lines = ["# B13 + Teacher 与 Teacher 后融合实测", "", "新增实验已执行：真实历史 B13 own D/E/U，固定 T0；1198 query（599 implicit / 599 explicit），query-macro Recall，source-cluster bootstrap 10,000 次，seed260914。", "",
             "同一 Equal→C100 候选池中，直接按 T0 分数排序优于 Teacher 后的 Equal/confidence/column。先融合候选再交给 Teacher 有收益；这不意味着 Teacher 排序后再做 rank fusion 有收益。", "",
             "| 固定 C100 方法 | Overall R10 | Implicit R10 | Explicit R10 |", "|---|---:|---:|---:|"]
    labels = {"B13_QT_only":"B13 QT 排序（同 C100）","T0_only":"T0 直接排序","T0_D_T0_E/Equal":"两路 T0 后 Equal","T0_D_T0_E/Conf":"两路 T0 后 confidence","T0_D_T0_E/Column":"两路 T0 后 column"}
    for method,label in labels.items():
        values = [receipt["metrics"]["Equal_BT100"][kind][method]["recall@10"]*100 for kind in ("overall","implicit","explicit")]
        lines.append(f"| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
    lines += ["", "Stage1 同池整体 Recall（K=10/20/30/40/50；小 K 仅用于 Stage2 最终输出）：", "",
              "| 方法 | R10 | R20 | R30 | R40 | R50 |", "|---|---:|---:|---:|---:|---:|"]
    for method,label in labels.items():
        values = [requested_metrics["Equal_BT100"]["overall"][method][f"recall@{k}"]*100 for k in KS]
        lines.append("| "+label+" | "+" | ".join(f"{v:.2f}%" for v in values)+" |")
    lines += ["", "只用 Teacher 重排 D、保留原 E/path 排序的额外对照：", "",
              "| 后融合 | Overall R10 | Implicit R10 | Explicit R10 |", "|---|---:|---:|---:|"]
    for method in ("Equal","Conf","Column"):
        values = [receipt["metrics"]["Equal_BT100"][kind]["T0_D_original_E/"+method]["recall@10"]*100 for kind in ("overall","implicit","explicit")]
        lines.append(f"| {method} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
    lines += ["", "候选通道对照（不同候选池，不能当同池排序比较）：", "", "| B13 路线 | Overall R10 | Implicit R10 | Explicit R10 |", "|---|---:|---:|---:|"]
    for method,label in (("D100_T0","Direct100→T0"),("BT100_T0","Equal→C100→T0"),("U_OFFLINE_T0","完整 U→T0（离线）"),("M_OFFLINE_T0","等大小 exact M→T0（离线）")):
        values = [teacher[kind][method]["recall@10"]*100 for kind in ("overall","implicit","explicit")]
        lines.append(f"| {label} | {values[0]:.2f}% | {values[1]:.2f}% | {values[2]:.2f}% |")
    lines += ["", "后融合相对 T0-only 的 C100 Overall R10 差及配对 source-bootstrap 95% 区间：", ""]
    for result in comparisons:
        if result["pool"] == "Equal_BT100" and result["kind"] == "overall" and result["k"] == 10 and result["left"] != "B13_QT_only":
            lo,hi = result["bootstrap_95ci"]
            lines.append(f"- {labels[result['left']]}：{100*result['mean_delta']:+.2f}pp，[{100*lo:+.2f}, {100*hi:+.2f}]pp；W/L/T={result['wins']}/{result['losses']}/{result['ties']}。")
    lines += ["", "实验语义：C100 在 Teacher 之前固定为真实 B13 Equal 前100；Teacher 最多评分100个不同 Q/T pair，D/E 重排复用这些分数。Teacher 对 E 做的是 QT 重排，并未验证具体 evidence 内容。Stage1 的 Recall@10/20/30/40/50 从完整排名重新计算，保存于 metrics_R10_20_30_40_50.json。完整 U 结果属于离线诊断。", "",
              "Column 使用同一个256条 hash-selected train-fit query 的自然 raw-Qwen U CDF，每query等权；真实列向量、输入文本和复算探针单列保留。当前结果不支持继续增加复杂融合，本实验没有按 dev 重新调权重。", "",
              "以上是重复查看 dev 后的探索性结果，历史 B13 只有一个真实模型，不伪造第二 seed。生成证据质量与正确补值到 join 的机制需独立 Stage2 结果验证。", "",
              "可复算文件：`TEACHER_FUSION_RECEIPT.json`、`rankings.jsonl.gz`、`paired_source_bootstrap.jsonl`；固定 Teacher 全池/Direct/BT100结果在 `../../teacher/B13/`。"]
    (directory / "RESULTS.md").write_text("\n".join(lines)+"\n")
    _json(directory / "REPORT_RECEIPT.json",{"execution_status":"ran","scientific_validity":"valid_exploratory_dev",
        "experiment":file_record(receipt_path),"requested_k_metrics":file_record(directory / "metrics_R10_20_30_40_50.json"),
        "bootstrap":file_record(directory / "paired_source_bootstrap.jsonl"),"report":file_record(directory / "RESULTS.md")})
    extension = OUT / "USER_EXTENSION_B13_TEACHER_FUSION.json"
    request = json.loads(extension.read_text())
    request.update({"status":"ran","scientific_validity":"valid_exploratory_dev","report":file_record(directory / "RESULTS.md")})
    _json(extension,request)
    return {"report":str(directory / "RESULTS.md"),"comparisons":len(comparisons)}


if __name__ == "__main__":
    print(json.dumps(run()))

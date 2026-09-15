"""Publish inspectable numeric tables and static figures for the B13 replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from replay_r26_evidence_order_abc import read_json, write_json, record
from replay_r26_coverage_mechanism import ARMS


def report(output: Path) -> None:
    result = read_json(output / "replay/RESULTS.json")
    latency = read_json(output / "latency/RESULTS.json")
    summary = result["summary"]
    lines = ["# B13 Coverage 机制与候选截断实验", "",
        "1198 条冻结 dev query（implicit/explicit 各599），固定 B13 checkpoint、Direct100 排名、E target 集合、U、Equal RRF(k=60) 和 T0。没有训练，没有 Stage2。",
        "", "2×2 的两个 selector 使用相同的内容去重后 Path Top20 候选，每个 target 最多4条 evidence；greedy 无正增益即停止。greedy_coverage、greedy_lse 是 r26 已有对照的复现。",
        "", "no-path 定义：fixed_greedy_no_weight 固定原 greedy bundle，只去掉 sigmoid 权重；greedy_no_path_same_pool 在原 Top20 池内也去掉选择质量权重；greedy_no_path 进一步用 evidence ID 选内容代表且取消 Path Top20，使冻结 retrieval 之后完全不读 path score。上游候选检索仍来自 B13，不能据此声称不需要 learned retrieval。",
        "", "top_matched_coverage 的 evidence 条数与每个 target 原 greedy 相同，用来控制 greedy 提前停止造成的数量差异。",
        "", "所有 coverage 使用 clip((Qwen row dot evidence + 1)/2,0,1)。连续相似度和路由行数是代理量，不是正确属性覆盖或正确 value 的标注。", ""]
    csv_rows = []
    for kind, group in summary.items():
        lines += [f"## {kind}：C100 机制拆解", "", "百分数为 query-macro recall；固定 Teacher 每 query 100 对。", "",
                  "| arm | E@10 | E@20 | E@50 | C100 | T0@10 | T0@20 | T0@50 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            a = group["arms"][arm]
            m = a["metrics"]
            values = [m["E"][f"recall@{k}"] for k in (10,20,50)] + [m["C"]["raw_recall"]] + [m["T0"][f"recall@{k}"] for k in (10,20,50)]
            lines.append("| " + arm + " | " + " | ".join(f"{100*v:.3f}" for v in values) + " |")
            csv_rows.append({"group":kind,"arm":arm, **dict(zip(("E10","E20","E50","C_recall","T10","T20","T50"),values)),
                             "pairs":a["teacher_pairs"], **a["strict_hits"]})
        lines += ["", f"strict EO 分母为 {group['strict_pairs']} 个固定 query-target positives。", "",
                  "| arm | E Top50 | C100 | T0 Top10 | T0 Top20 | T0 Top50 |", "|---|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            hits = group["arms"][arm]["strict_hits"]
            lines.append("| " + arm + " | " + " | ".join(str(hits[m]) for m in ("E@50","C","T0@10","T0@20","T0@50")) + " |")
        lines += ["", f"## {kind}：候选预算漏斗", "",
                  "| budget | C Recall | T0@10 | T0@20 | T0@50 | EO进入C | EO T10/T20/T50 | Teacher总对数 | 平均对数/query |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for budget in ("C50","C100","C150","C200","Full-U"):
            a = group["arms"][budget]
            m, h = a["metrics"], a["strict_hits"]
            values = [m["C"]["raw_recall"]] + [m["T0"][f"recall@{k}"] for k in (10,20,50)]
            lines.append("| " + budget + " | " + " | ".join(f"{100*v:.3f}" for v in values) +
                         f" | {h['C']} | {h['T0@10']}/{h['T0@20']}/{h['T0@50']} | {a['teacher_pairs']} | {a['mean_pairs']:.2f} |")
            csv_rows.append({"group":kind,"arm":budget, **dict(zip(("C_recall","T10","T20","T50"),values)),
                             "pairs":a["teacher_pairs"], **h})
        lines += [""]
    lines += ["## 配对差异与不确定性", "",
              "按 source_table_id 整簇配对 bootstrap，10,000次，seed260914；下表为 overall T0@10，完整三个分组与各级指标在 replay/RESULTS.json。多重探索对照未校正；CI 跨0不等于统计等效。", "",
              "| 新 − 旧 | Δ pp | 95% CI pp | 胜/负/平 queries |", "|---|---:|---:|---:|"]
    for c in result["comparisons"]:
        if c["kind"] == "overall" and c["stage"] == "T0" and c["metric"] == "recall@10":
            low, high = c["bootstrap_95ci"]
            lines.append(f"| {c['new']} − {c['old']} | {100*c['mean_delta']:+.3f} | [{100*low:+.3f}, {100*high:+.3f}] | {c['wins']}/{c['losses']}/{c['ties']} |")
    lines += ["", "## Selector 的实际行为", "", "| group | target对数 | greedy平均条数 | top4平均条数 | greedy平均路由行数 | top4平均路由行数 | greedy平均增量coverage | top4平均增量coverage | 同bundle比例 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    counts = result["counts"]
    for group in ("all","positive","strict"):
        n = counts[group + "/targets"]
        keys = ("greedy/paths","top4/paths","greedy/routed_rows","top4/routed_rows",
                "greedy/coverage_gain_over_best_single","top4/coverage_gain_over_best_single","greedy_top4_same_set")
        lines.append(f"| {group} | {n} | " + " | ".join(f"{counts[group+'/'+k]/n:.6f}" for k in keys) + " |")
    lines += ["", "增量coverage = bundle coverage − 最佳单条 evidence coverage（在该 bundle 内），与正确语义行覆盖不同。", "",
              "原始路径 sigmoid 的 min/p10/median/p90/max：" + json.dumps(result["diagnostics"]["path_sigmoid_quantiles_0_10_50_90_100"]), "",
              "路径数量与 pre-LSE / coverage 的 Pearson 相关（描述统计，有 query 混杂，非因果）：" +
              f"{result['diagnostics']['pearson_path_count_pre_lse']:.4f} / {result['diagnostics']['pearson_path_count_coverage']:.4f}。", "",
              "## T0 实测延迟", "", "GPU1，32条按 SHA256(query_id) 固定盲选 query，5档预算，每档3次 cold/warm。每 query 先取3次中位数，再计算32条的 p50/p95。无 pair-score cache；每档独立 compression cache；预算次序轮换。", "",
              "预加载冻结本地 features，时间包含 T0 运算、CPU调度和数据传输；不包含 backbone、ANN、coverage、Stage2 和模型初始化。cold 指 compression cache cold，并非操作系统磁盘冷缓存。", "",
              "| budget | cold p50 ms | cold p95 ms | warm p50 ms | warm p95 ms |", "|---|---:|---:|---:|---:|"]
    for budget, modes in latency["latency"].items():
        values = [modes[mode][metric] * 1000 for mode in ("cold_compression","warm_compression") for metric in ("median_seconds","p95_seconds")]
        lines.append("| " + budget + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", f"共 {latency['measurements']} 条计时记录；实测与冻结 T0 最大分数误差 {latency['max_score_error']:.8g}（阈值0.001）；GPU峰值 allocated {latency['peak_allocated_bytes']/1024**2:.1f} MiB。", "",
              "全量准确率来自固定历史 T0 分数的精确 replay，新增 GPU 推理仅为计时和一致性核验。不能用 replay 的零新增推理成本替代线上 Teacher pair 数。", "",
              "## 边界与 gate", "", "保持 coverage 作为本轮基准；只在独立 learned path score 的增益得到机制对照支持且 strict EO 同向时，再考虑 coverage-aligned training。没有进一步训练或挑选额外 seed。", "",
              "本轮只覆盖 B13，不能外推到所有模型；no-path 的上游仍是 B13 retrieval。无属性 truth 标注、value 生成和最终 join 验证，不能声称 evidence 语义正确或缺失属性已恢复。Stage2 按用户要求暂缓。", ""]
    (output / "REPORT.zh-CN.md").write_text("\n".join(lines))
    keys = sorted({k for r in csv_rows for k in r})
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    make_figure(output, summary["overall"], latency)
    write_json(output / "REPORT_ARTIFACTS.json", {p.name: record(p) for p in (
        output / "REPORT.zh-CN.md", output / "metrics.csv", output / "funnel.png", output / "funnel.svg")})


def make_figure(output: Path, group: dict, latency: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    budgets = ("C50","C100","C150","C200","Full-U")
    x = range(len(budgets))
    fig, axes = plt.subplots(1,3,figsize=(15,4.6), layout="constrained")
    for k, color in ((10,"#1664a4"),(20,"#d47d1e"),(50,"#228468")):
        axes[0].plot(x, [100*group["arms"][b]["metrics"]["T0"][f"recall@{k}"] for b in budgets],
                     "o-", color=color, label=f"Teacher R@{k}")
    axes[0].plot(x,[100*group["arms"][b]["metrics"]["C"]["raw_recall"] for b in budgets],"--",color="#687585",label="Candidate recall")
    axes[0].set(ylabel="Query-macro recall (%)",title="B13: candidate budget to Teacher recall")
    for key, name, color in (("C","Admitted", "#687585"),("T0@10","Teacher Top10", "#1664a4"),("T0@50","Teacher Top50", "#228468")):
        values = [group["arms"][b]["strict_hits"][key] for b in budgets]
        axes[1].plot(x, values,"o-",color=color,label=name)
        for i, value in enumerate(values):
            offset = {"C": (0, 12), "T0@10": (0, -15), "T0@50": (10, -5)}[key]
            axes[1].annotate(str(value),(i,value),xytext=offset,textcoords="offset points",
                             ha="center",fontsize=8,color=color)
    axes[1].axhline(207,color="#999999",linestyle=":",linewidth=1)
    axes[1].set(ylabel="Fixed strict EO pairs (of 207)",title="Strict evidence-only funnel",ylim=(0,225))
    for mode,color in (("cold_compression","#1664a4"),("warm_compression","#d47d1e")):
        axes[2].plot(x,[1000*latency["latency"][b][mode]["median_seconds"] for b in budgets],"o-",color=color,label=mode.replace("_"," "))
    axes[2].set(ylabel="T0 p50 latency (ms/query)",title="GPU1: 32 queries, 3 repeats")
    for ax in axes:
        ax.set_xticks(list(x), budgets)
        ax.set_xlabel("Equal-RRF candidate budget")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "funnel.png",dpi=180)
    fig.savefig(output / "funnel.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    report(parser.parse_args().output)

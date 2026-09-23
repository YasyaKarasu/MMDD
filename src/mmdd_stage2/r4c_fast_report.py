"""Chinese Markdown reports for the S2-R4c FAST gate experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def write_pilot_report(
    path: Path,
    *,
    preflight: dict[str, Any],
    lock: dict[str, Any],
    smoke: dict[str, Any] | None,
    summary: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    blocker: str | None = None,
) -> None:
    lines = [
        "# MMDD S2-R4c FAST PILOT 报告",
        "",
        "日期：2026-09-23",
        "",
        "## 执行状态",
        "",
        "| 阶段 | 状态 | 说明 |",
        "|---|---|---|",
        f"| Preflight | {preflight.get('status', 'unknown')} | 固定 dev / R3 / R4 / Qwen3.5-9B 输入 |",
        f"| Label-blind dev lock | {lock.get('status', 'unknown')} | {lock.get('locked_units', 0)} units，{lock.get('source_groups', 0)} source groups；sha256 `{lock.get('lock_sha256', 'unknown')}` |",
        f"| FAST-SMOKE | {(smoke or {}).get('status', 'planned')} | 仅链路检查，不参与选型 |",
        f"| FAST-PILOT | {summary.get('status') if summary else ('blocked' if blocker else 'planned')} | {summary.get('executed_units', 0) if summary else 0}/{lock.get('locked_units', 0)} units 四臂配对；同一批 label-blind lock |",
        f"| FAST-EXPAND | {'planned' if gate and gate.get('continue_expand') else 'skipped_by_gate' if gate else 'planned'} | dev，最多 128 units，V0 vs 唯一 winner；尚未执行 |",
        f"| S0-refill/S1-global integration | {'planned' if gate and gate.get('continue_expand') else 'skipped_by_gate' if gate else 'planned'} | 图像策略通过后，16 source groups；尚未执行 |",
        f"| Test | {'skipped_by_gate' if gate and not gate.get('continue_expand') else 'planned_after_dev_selection'} | 本阶段禁止运行；没有 test 结果 |",
        "",
        "## Stage-1 输入来源",
        "",
        "本轮不是直接截取 Student 的前 50 项。B13 checkpoint 在 dev 上生成自然候选池 U 及其 retained evidence paths；固定 T0 在同一候选池上产生 `U_OFFLINE_T0` 重排，`FROZEN_C50` 截取该重排的前 50 项。S2-R4 从 T0 cache 读取每个候选的原生 Q-T logit，再与固定 Prior 列模型的列分数构成 `J1_ExactProduct_Prior` 排名；`S1_JointGlobal` 据此选择恢复分支。图像小样本只从这些已选分支中按 source group 和图像可用性作 label-blind 抽样，不替换 Stage-1 排名。",
        "",
        "T0 重排表候选，不重新检索或重选 B13 留下的 evidence paths；Prior 列模型属于 Stage-2，不是 Stage-1 teacher。锁定 unit 的 exact ROW_GT 覆盖数不是答对数；覆盖不足时不能判定裁剪方案的准确率收益。",
        "",
    ]
    if blocker:
        lines.extend(["## 阻塞", "", blocker, ""])
    if summary:
        lines.extend([
            "## Label-blind pilot 指标",
            "",
            f"抽样不读取 ROW_GT/witness；锁定后才关联评分。全部 {summary.get('all_units')} locked units；已执行完整四臂 {summary.get('executed_units')} units；全锁中有 exact GT 可评分 {summary.get('evaluable_units')} units，其中已执行 {summary.get('evaluable_executed_units')}。无 GT 只计入覆盖/运行记录，不计入准确率分母。",
            "",
            "主指标为 query-macro row-level StrictValueCorrect。SupportedValueCorrect 要求 strict 正确且引用 ROW_GT witness source；幻觉支持按已引用来源但未形成正确且有 witness 的答案统计，是保守代理量。准确率只对可评分子集汇总，下表成本单独覆盖全部锁定 units。",
            "",
            "| Arm | Units | Strict | Supported | Refusal | Hallucinated support | Parse error | Median latency(s) | P95 latency(s) | Median pixels | Median image tokens | Peak VRAM median (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm, metrics in summary["arms"].items():
            lines.append(
                f"| {arm} | {metrics.get('units', 0)} | {_pct(metrics.get('query_macro_strict_correct'))} | "
                f"{_pct(metrics.get('query_macro_supported_value_correct'))} | {_pct(metrics.get('query_macro_refused'))} | "
                f"{_pct(metrics.get('query_macro_hallucinated_supported_value'))} | {_pct(metrics.get('query_macro_parse_error'))} | "
                f"{_num(metrics.get('median_total_elapsed_seconds'))} | {_num(metrics.get('p95_total_elapsed_seconds'))} | "
                f"{_num(metrics.get('median_input_pixels'), 0)} | {_num(metrics.get('median_image_tokens'), 0)} | "
                f"{_num((metrics.get('median_peak_gpu_memory_bytes') or 0) / 2**30)} |"
            )
        lines.extend(["", "## 配对比较（相对 V0）", ""])
        for arm, comparison in summary["comparisons_vs_v0"].items():
            wlt = comparison["wlt"]
            bootstrap = comparison["bootstrap"]
            lines.append(
                f"- {arm}: W/L/T={wlt['wins']}/{wlt['losses']}/{wlt['ties']}；"
                f"query-macro ΔStrict={_pct(bootstrap.get('point'))}，"
                f"source-group bootstrap 95% CI [{_pct(bootstrap.get('low'))}, {_pct(bootstrap.get('high'))}]。"
            )
        lines.append("")
        lines.extend([
            "全部锁定 units 的运行成本（含无 GT）：", "",
            "| Arm | Units | Median latency(s) | P95 latency(s) | Median pixels | Median image tokens | Median prompt tokens | Peak VRAM median (GiB) | Crop fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm, metrics in summary.get("all_units_runtime", {}).items():
            lines.append(
                f"| {arm} | {metrics.get('units', 0)} | {_num(metrics.get('median_total_elapsed_seconds'))} | "
                f"{_num(metrics.get('p95_total_elapsed_seconds'))} | {_num(metrics.get('median_input_pixels'), 0)} | "
                f"{_num(metrics.get('median_image_tokens'), 0)} | {_num(metrics.get('median_prompt_tokens'), 0)} | "
                f"{_num((metrics.get('median_peak_gpu_memory_bytes') or 0) / 2**30)} | {_pct(metrics.get('crop_fallback'))} |"
            )
        lines.append("")
        lines.extend(["原始运行状态（含无 GT units）：", ""])
        for arm, counts in summary.get("raw_status_counts", {}).items():
            lines.append(f"- {arm}: " + ", ".join(f"{key}={value}" for key, value in counts.items() if value))
        lines.append("")
    if gate:
        lines.extend([
            "## Gate 决策",
            "",
            f"- 状态：`{gate.get('status')}`",
            f"- 锁定方案：`{gate.get('winner')}`",
            f"- 是否进入 FAST-EXPAND：`{str(bool(gate.get('continue_expand'))).lower()}`",
            f"- 原因：{gate.get('reason', '详见门槛检查')}。",
            "",
        ])
        for name, passed in gate.get("checks", {}).items():
            lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
        lines.append("")
    lines.extend([
        "## 边界与限制",
        "",
        "- 本轮先按 source group 和可用图像抽样，不以 gold、witness、历史成败或 crop 质量挑选；仍条件于已有预测分支和自然 retained bundle，不能解释为在线端到端成功率。",
        "- 所有样本均来自 dev；没有运行 full dev 或任何 test 推理。",
        "- 本实验未证明最终 row join 收益；只有视觉策略通过后才允许另做小型 integration pilot。",
        "- PPT 学生实验不是 MMDD 结果；Adapt-v1 的 c/a/p 数值定义与 crop 参数是本轮复现合同。",
        "- 独立包中 witness-conditioned component pilot 与本轮明确的 label-blind 要求冲突；本报告按用户指定的 label-blind FAST-PILOT 执行，不称为组件分层实验。",
        "- 未执行的阶段明确标为 planned / skipped_by_gate，不计入实验结果。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

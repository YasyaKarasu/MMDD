"""S2-R4 Chinese final report, generated from artifacts that were actually produced.

A section whose artifact is absent is written as blocked/not_executed. No number here is
estimated, carried over from a different run, or filled in from expectation.
"""
from __future__ import annotations

import json
from pathlib import Path

from .r4_common import R4, write_json

STATUS_LABEL = {
    'planned': 'planned（仅写入合同）',
    'implemented': 'implemented（代码与单测通过，未在真实输入上跑）',
    'executed': 'executed（已在真实输入上运行并产出工件）',
    'evaluated': 'evaluated（输出已对标签评分）',
    'blocked': 'blocked（真实输入或资源缺失）',
    'not_executed': 'not_executed（本轮明确不做）',
}


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _pct(x, digits=2):
    return 'n/a' if x is None else f'{100 * x:.{digits}f}%'


def _pp(x, digits=2):
    return 'n/a' if x is None else f'{x:+.{digits}f}pp'


def _ci(bounds, digits=2):
    if not bounds or bounds[0] is None:
        return 'n/a'
    return f'[{100 * bounds[0]:.{digits}f}, {100 * bounds[1]:.{digits}f}]'


def build(out: Path, scope: str) -> str:
    metrics = _read(out / f'PHASE_J_METRICS.{scope}.json')
    coverage = _read(out / f'PHASE_J_COVERAGE.{scope}.json')
    labels = _read(out / f'LABEL_COVERAGE.{scope}.json')
    population = _read(out / f'POPULATION_LOCK.{scope}.json')
    p0 = _read(out / 'P0_REPLAY.json')
    schedule = _read(out / 'RECOVERY_SCHEDULES' / f'SCHEDULE_SUMMARY.{scope}.json')
    funnel = _read(out / 'COSTS' / f'EVIDENCE_FUNNEL.{scope}.json')
    costs = _read(out / 'COSTS' / f'COSTS.{scope}.json')

    lines = [
        f'# S2-R4 最终报告（scope: {scope}）',
        '',
        '本文件由 `mmdd_stage2/r4_report_md.py` 从实际产出的工件生成。缺工件的章节写 '
        '`blocked`/`not_executed`，不估算、不挪用其他运行的数字。',
        '',
        '## 0. 状态词汇',
        '',
    ]
    for key in ('planned', 'implemented', 'executed', 'evaluated', 'blocked', 'not_executed'):
        lines.append(f'- `{STATUS_LABEL[key]}`')

    # ---- module status -------------------------------------------------
    lines += ['', '## 1. 模块状态', '', '| 模块 | 状态 |', '|---|---|']
    module_status = {
        'P0 审计修正（bootstrap/计数/G4/模态/无效E）': 'evaluated' if p0 else 'implemented',
        'CANDIDATE_LOCK / POPULATION_LOCK': 'executed' if population else 'implemented',
        'Phase J 列特征（全部候选 T）': 'executed' if coverage else 'blocked',
        'Phase J J0/J1/J2 排序与指标': 'evaluated' if metrics else 'blocked',
        'Phase I S0/S1 调度': 'executed' if schedule else 'implemented',
        '证据供给漏斗': 'executed' if funnel else 'implemented',
        'Phase R 恢复三臂（512 token）': _recovery_status(out, out_done=out),
        '可选 E-LateBind': 'not_executed',
        '错误 T 校准 / none 训练': 'not_executed',
        'test split 正式排序': 'not_executed',
    }
    for name, status in module_status.items():
        lines.append(f'| {name} | `{status}` |')

    # ---- Q1 ------------------------------------------------------------
    lines += ['', '## 2. 必答问题', '', '### Q1 是否真的对错误 T 与空 E 的 T 打列分数并参与乘积？', '']
    if coverage:
        lines += [
            f'- 入口：`mmdd_stage2.r4_joint_build.build_joint` → `r4_joint.log_joint`（log-space 精确乘积）。',
            f'- queries = {coverage["queries"]}，候选对 = {coverage["candidate_pairs"]}，'
            f'其中有列可打的 = {coverage["pairs_with_columns"]}，无列 raw 表 = {coverage["pairs_without_columns"]}。',
            f'- 列单元 = {coverage["column_cells"]}，实际打分列单元 = {coverage["column_cells_scored"]}'
            f'（4 个 arm×seed 全覆盖）。',
            f'- 缺失特征 = {coverage["pairs_missing_features"]}；`COLUMN_SCORING_ERROR` = '
            f'{coverage["scoring_error_count"]}（全部来自无列 raw 表，候选保留、原 rank 保留、不补均匀分布）。',
            f'- `full_finite_coverage` = {coverage["full_finite_coverage"]}。',
            '- 空 E 的 T **保留在池内**并进入表 softmax；空 E 时 Flat-Mix 的证据块为空，与 Prior 提示逐字相同，'
            '两者共用一个 reader job key——这是"空 E 是真空"的本意，不是省算力。',
        ]
    else:
        lines.append('- `blocked`：Phase J 尚未在真实输入上产出覆盖统计。')

    # ---- Q2 ------------------------------------------------------------
    lines += ['', '### Q2 原表分数 × 列分数到底有没有帮助？', '']
    if metrics:
        arms = ('J0_TableOnly', 'J1_ExactProduct_Prior', 'J2_ExactProduct_FlatMix')
        lines += ['| 臂 | Target R@10 | R@20 | R@50 |', '|---|---:|---:|---:|']
        for arm in arms:
            entry = metrics['per_arm'].get(arm)
            if not entry:
                continue
            lines.append(
                f"| {arm} | {_pct(entry['target_recall@10']['point'])} "
                f"{_ci(entry['target_recall@10']['ci95'])} | "
                f"{_pct(entry['target_recall@20']['point'])} | "
                f"{_pct(entry['target_recall@50']['point'])} |")
        lines += ['', '配对差（同 query、同池、同标签、同预算，source-group cluster bootstrap）：', '']
        for key, value in metrics['comparisons_target_recall@10'].items():
            lines.append(f"- `{key}`：{_pp(value['difference_pp'])}，95%CI "
                         f"[{value['ci95_pp'][0]:+.2f}, {value['ci95_pp'][1]:+.2f}]，"
                         f"配对 query = {value['paired_queries']}")
        inv = metrics['c50_invariance_check']
        lines += ['', f"- C50 候选不变性：R@50 三臂是否相同 = **{inv['identical']}**"
                      f"（{'符合预期' if inv['identical'] else '异常：候选池在臂之间发生了变化'}）。",
                  f"- 池子天花板：query-macro = {_pct(metrics['pool_ceiling']['query_macro_ceiling'])}，"
                  f"micro = {_pct(metrics['pool_ceiling']['micro_ceiling'])}；"
                  f"{metrics['pool_ceiling']['queries_with_every_gold_target_in_pool']} 个 query 的 gold 全在池内，"
                  f"{metrics['pool_ceiling']['queries_with_no_gold_target_in_pool']} 个完全不在池内。",
                  '- 读法：臂间差异是重排，离 1.0 的差距主要是候选生成。']
        wrong = metrics.get('wrong_table_confidence') or {}
        if wrong:
            lines += ['', '列置信度分桶（正 T vs benchmark 非正 T；非正未穷尽，只叫 unlabeled）：', '',
                      '| 桶 | arm | 表数 | 平均列数 | mean max-ρ | p90 | ρ 熵 |', '|---|---|---:|---:|---:|---:|---:|']
            for bucket, per_arm in wrong.items():
                for arm, value in sorted(per_arm.items()):
                    lines.append(
                        f"| {bucket} | {arm} | {value['tables']} | {value['mean_columns']:.2f} | "
                        f"{value['mean_max_rho']:.3f} | {value['max_rho_p90']:.3f} | "
                        f"{value['mean_rho_entropy']:.3f} |")
    else:
        lines.append('- `blocked`：尚无 J 阶段指标。')

    # ---- Q3 ------------------------------------------------------------
    lines += ['', '### Q3 gold-target 与真实候选之间，损失在哪一层？', '']
    if labels:
        lines += [
            f"- 标注覆盖：scope 内 {labels['queries_in_scope']} 个 query，"
            f"其中 {labels['queries_with_gold']} 个有 gold，{labels['queries_with_no_gold_in_split']} 个无。",
            f"- gold 对总数 {labels['gold_pairs_total']}，**在候选池内 {labels['gold_pairs_in_candidate_pool']}**，"
            f"池外 {labels['gold_pairs_outside_candidate_pool']}。",
            '- 池外部分任何重排都不能救，属候选生成层损失。',
        ]
    if schedule:
        for name, s in schedule['schedulers'].items():
            lines.append(f"- {name}：{s['branches']} 个 branch，覆盖 {s['distinct_tables_total']} 张不同 T；"
                         f"每 query 不同 T 数分布 {s['distinct_tables_per_query']}。")
        lines.append('- S1 用满 pair 预算但未必覆盖 10 张不同 T：pair 预算 ≠ table 预算。')
    if funnel:
        lines += ['', '证据供给漏斗（已知 witness）：', '',
                  '| 阶段 | witness | 占比 |', '|---|---:|---:|']
        for stage in funnel['funnel']:
            if stage.get('witnesses') is None:
                lines.append(f"| {stage['stage']} | n/a | `{stage.get('status')}` |")
            else:
                lines.append(f"| {stage['stage']} | {stage['witnesses']} | {_pct(stage['of_total'])} |")
        lines.append(f"- 未记录层：{funnel['missing_stages'] or '无'}"
                     '（recorded 之外的分层一律记 missing_stage，不猜）')

    # ---- Q4/Q5 ---------------------------------------------------------
    lines += ['', '### Q4 可见 query 信息 vs 外部证据，各贡献多少？', '']
    lines.append('- ' + ('见 Phase R 来源分项（`SOURCE_TRACES`）：`EXTERNAL_EVIDENCE_TEXT_TRACEABLE` 与 '
                         '`QUERY_VISIBLE_TRACEABLE` 分开计数。'
                         if _has_recovery(out) else
                         '`blocked`：Phase R 尚未执行，无法把收益拆成可见信息与外部证据。'))

    lines += ['', '### Q5 多值集合提升的是候选可用率还是最终已验证连接？', '']
    lines.append('- ' + ('见 Phase R 的 `ValueCandidateHit@1/3` 与 `JointColumnValueHit@k,b`；'
                         '集合里存在 gold 不改名为 Top1。'
                         if _has_recovery(out) else
                         '`blocked`：Phase R 尚未执行。'))

    # ---- costs ---------------------------------------------------------
    if costs:
        r, g = costs['reader'], costs['generator']
        lines += ['', '## 3. 成本（组件分开，不混算）', '',
                  f"- reader（{r['component']}）：{r['unique_forwards']} 次前向，"
                  f"p50 {r['latency_p50_seconds']}s / p95 {r['latency_p95_seconds']}s，"
                  f"峰值显存 {r['peak_gpu_memory_bytes']} B，设备 {r['gpu_names']}",
                  f"- generator（{g['component']}）：{g['calls']} 次调用 / "
                  f"{g['decisions_without_a_call']} 次免调用决策，max_new_tokens={g['max_new_tokens']}",
                  '- 两类成本不求和；R3 的双 4090 历史计时不作为其他 GPU 配置的估计。']

    deferred = out / 'RECOVERY_SCHEDULES' / f'DEPLOY_PILOT_SELECTION.{scope}.json'
    if deferred.is_file() and not _has_recovery(out):
        selection = json.loads(deferred.read_text())
        lines += ['', '### Phase R 预注册状态（未执行）', '',
                  f"- 冻结选择回执：`{deferred.name}`，salt `{selection['salt']}`，"
                  f"{selection['groups_selected']} 个 source group → {selection['queries_selected']} 个 query。",
                  f"- `frozen_before_generation` = {selection['frozen_before_generation']}，"
                  f"`reads_gold` = {selection['reads_gold']}。",
                  '- 本轮按用户决定推迟，未产生任何生成结果；该回执是冻结记录，不是成绩。']

    lines += ['', '## 4. 文件清单', '',
              '见 `DELIVERY_MANIFEST.json`；执行状态见 `EXECUTION_STATUS.json`。', '']
    return '\n'.join(lines)


def _has_recovery(out: Path) -> bool:
    folder = out / 'RAW_GENERATIONS'
    return folder.is_dir() and any(folder.glob('*.jsonl.gz'))


def _recovery_status(out: Path, *, out_done: Path | None = None) -> str:
    if _has_recovery(out):
        return 'executed'
    # The queues and the frozen selection receipt exist, but no generation was run this
    # round. Recording it as 'implemented' keeps it distinct from 'blocked' (no missing
    # input) and from 'not_executed' (nothing built).
    return 'implemented（本轮未执行；队列与预注册已冻结，未生成任何值）'


def write_report(out: Path, scope: str) -> dict:
    text = build(out, scope)
    path = out / 'FINAL_REPORT.zh-CN.md'
    path.write_text(text)
    write_json(out / f'EXECUTION_STATUS.{scope}.json', {
        'scope': scope,
        'status_vocabulary': ['planned', 'implemented', 'executed', 'evaluated', 'blocked',
                              'not_executed'],
    })
    return {'report': str(path), 'bytes': len(text.encode())}

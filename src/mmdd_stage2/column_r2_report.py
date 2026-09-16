"""Generate the Chinese evidence-grounded final report from completed artifacts."""
from __future__ import annotations

from pathlib import Path
from statistics import mean

from .column_data import file_hash, write_json
from .column_r2_audit import read_json


def percent(value: float | None) -> str:
    return 'NA' if value is None else f'{100*value:.3f}%'


def interval(result: dict) -> str:
    if result['difference'] is None:
        return 'NA'
    lo, hi = result['ci95']
    return f'{100*result["difference"]:+.3f} pp [{100*lo:+.3f}, {100*hi:+.3f}]'


def build_report(r1: Path, output: Path) -> None:
    formal = read_json(output/'FORMAL_TEST.json')
    lock = read_json(output/'MODEL_SELECTION_LOCK.json')
    metrics = read_json(output/'ANALYSIS/subgroup_metrics.json')
    boot = read_json(output/'ANALYSIS/paired_bootstrap.json')
    costs = read_json(output/'COST/reader_costs.json')
    support = read_json(output/'SUPPORT_AUDIT/MANIFEST.json')
    phase_a = read_json(output/'PHASE_A/RESULTS.json')
    arms = list(lock['checkpoints'])
    lines = ['# S2-COL-R2 最终报告', '',
        '## 范围与执行状态', '',
        '仅 EntiTables implicit positive-target column localization。WDC 的 canonical artifacts 不可用，未外推到 WDC。',
        '本轮没有 crop、span、属性生成、Stage1 训练或最终 semantic-joinability 评估。高 Hit@3 不代表 Stage2 join 已解决。', '',
        '| 实验 | planned | implemented | actually executed | actually evaluated |',
        '|---|---|---|---|---|',
        '| R1 source/data/cache/checkpoint 审计与 replay | 是 | 是 | 是 | 12 个历史配置精确 replay |',
        '| Phase A concat/single/Mean/LME/LOO/order | 是 | 是 | 是 | dev only |',
        '| oracle-single | 诊断上界 | 是 | 是 | dev only；不可部署 |',
        '| frozen train O-R | 是 | 是 | 是 | provenance / population / retention 审计 |']
    for arm in arms:
        lines.append(f'| {arm} | 是 | 是 | seeds 13, 29；各 20 epochs | dev 选型；locked test |')
    if not support['execute']:
        lines.append('| PVR_SUPPORT | 条件执行 | 是 | 否 | 否：明确 witness 的 train Q-T 少于 500 |')
    lines += ['', '## 可复现性与控制', '',
        '- R1 原始 source、data、reader、feature、checkpoint 哈希经核对；基线 replay 排序一致且最大 logit 差为 0。',
        '- train 7029 Q-T / 6310 queries；dev 675 / 597；test 644 / 581。完整锁定 population 保留 prior miss 与空 evidence。',
        '- train O-R 先独立运行 frozen B13/T0 得到完整自然 U，再查正 T；T 不在 U 时 E=[]，不回填 O-O。',
        '- retention 保持 exact-content dedup -> path-score top20 -> e2_row_coverage -> budget4。',
        '- Flat-Mix 与新增 O-O control 的初始化、seed、每 epoch 单次访问、4400 steps、optimizer、dev 选型规则一致。',
        '- 历史 R1-C2 按 O-O 选型；R2 control/Flat 按 O-R 选型。因此 Flat-Mix - O-O control 才是训练分布的严格归因比较。',
        '- Prior 只读 Q+T，独立训练 C2 MLP。shortlist 只由其 logits 生成并哈希冻结，绝不插入 gold。',
        '- PVR Prior 完全冻结；Bundle/Separate 的 Q-T、evidence IDs、hash schedule 相同；空 E 严格返回 prior。',
        '- shortlist 内使用 g=b+softplus(beta)*delta；shortlist 外按原 prior 顺序保留在其后。保存的外围分数仅是排序 sentinel，不是校准概率。',
        '- 模型只在 dev 选型；正式 test 在 MODEL_SELECTION_LOCK.json 之后执行。历史 R1 test replay 只做一致性验证，不用于 R2 选择。',
        '- 工作区并行修改了两个本轮不调用的数据重建 split 审计函数；其余 AST 与 R1 一致。未撤销用户修改，后续运行固定使用 RUNTIME_SOURCE 中的 R1 原始模块加 R2 实现；细节及哈希见其 MANIFEST.json。',
        '- 所有统计先在 Q 内平均 targets，再同 Q 平均两 seeds，最后按 source group paired cluster bootstrap 10,000 次。',
        '- correction 与 damage 的条件率分母不同，不能只比较两个条件百分比；同时报告 case counts、query-macro net correction 和 EvidenceGain。', '',
        '## Phase A：拼接干扰诊断', '',
        '| seed | concat H1 | Mean H1 | LME H1 | oracle-single H1（仅诊断） |',
        '|---|---|---|---|---|']
    for seed in ('13','29'):
        lines.append('| '+seed+' | '+' | '.join(percent(phase_a[seed][m]['query_macro']['ColHit@1']) for m in ('bundle','Mean','LME','oracle_single_evidence_upper_bound'))+' |')
    for method in ('Mean','LME'):
        lines.append(f'- {method} - concat：H1 {interval(phase_a["paired_bootstrap"][method]["ColHit@1"])}；MRR {interval(phase_a["paired_bootstrap"][method]["MRR"])}。')
    lines += ['','不使用 oracle-single 选 evidence、训练或部署。LOO 的 harmful/helpful 与 order flips 见 PHASE_A 原始记录。', '',
        '## 正式 Test', '', '| 方法 | seed | 子集 | Hit@1 | Hit@2 | Hit@3 | Hit@5 | MRR |', '|---|---|---|---|---|---|---|---|']
    for arm in ['R1_C2']+arms:
        for seed in ('13','29'):
            for sub in ('full','non_empty','M>3'):
                macro = metrics['test'][arm][seed]['subsets'][sub]['query_macro']
                values = [percent(macro[f]) if macro else 'NA' for f in ('ColHit@1','ColHit@2','ColHit@3','ColHit@5','MRR')]
                lines.append(f'| {arm} | {seed} | {sub} | '+' | '.join(values)+' |')
    lines += ['', '完整 pair-micro、M>1/2/3/5、modality、evidence_count、column-count bucket、empty/non-empty 及 dev 指标见 ANALYSIS/subgroup_metrics.json。', '',
        '## Correction 与 Damage', '',
        '| 方法 | seed | 子集 | Admission@3 | Correction rate | Damage rate | corrected/damaged | Net count | Net query-macro | EvidenceGain MRR |',
        '|---|---|---|---|---|---|---|---|---|---|']
    pvr_arms = [a for a in arms if a.startswith('PVR')]
    for arm in pvr_arms:
        for seed in ('13','29'):
            for sub in ('full','non_empty','M>3'):
                m = metrics['test'][arm][seed]['subsets'][sub]
                lines.append(f'| {arm} | {seed} | {sub} | {percent(m.get("PriorAdmissionAt3"))} | {percent(m.get("correction_rate"))} | {percent(m.get("damage_rate"))} | {m.get("corrected_cases")}/{m.get("damaged_cases")} | {m.get("net_correction_count")} | {percent(m.get("net_correction_query_macro"))} | {percent(m.get("EvidenceGain_MRR"))} |')
    lines += ['', '## 配对差值与置信区间', '', '| 比较 | 子集 | Hit@1 差值与 95% CI | MRR 差值与 95% CI |', '|---|---|---|---|']
    for comparison, values in boot['test'].items():
        for sub in ('full','non_empty','M>3'):
            lines.append(f'| {comparison} | {sub} | {interval(values[sub]["ColHit@1"])} | {interval(values[sub]["MRR"])} |')
    lines += ['', '## 五个科学问题', '']
    def result(comparison: str) -> str:
        b = boot['test'][comparison]['non_empty']
        return f'non-empty H1 {interval(b["ColHit@1"])}，MRR {interval(b["MRR"])}'
    def positive(comparison: str, field: str = 'ColHit@1') -> bool:
        return boot['test'][comparison]['non_empty'][field]['ci95'][0] > 0
    lines += ['### 1. 自然 evidence 增益小主要是不是训练分布不匹配？', '',
        result('FLAT_MIX-OO_CONTROL')+'。',
        ('受控比较支持训练分布匹配有正向作用；仅这一对比仍不足以把全部自然 evidence 瓶颈归为分布不匹配。' if positive('FLAT_MIX-OO_CONTROL') else
         '本轮未得到训练分布改变带来稳定正向 H1 收益的证据，不能把原来增益小主要归因于训练分布不匹配。'),
        '历史基线对比：'+result('FLAT_MIX-R1_C2')+'；其中含 checkpoint-selection 差异。', '',
        '### 2. Q+T Prior + Evidence Verify/Rerank 是否优于 flat concat？', '',
        'PVR-Bundle：'+result('PVR_BUNDLE-FLAT_MIX')+'。',
        'PVR-Separate：'+result('PVR_SEPARATE-FLAT_MIX')+'。',
        ('至少一个 PVR 变体相对 Flat-Mix 的 H1 正收益得到 CI 支持；仍需结合 MRR、full 与成本判断。'
         if any(positive(c) for c in ('PVR_BUNDLE-FLAT_MIX','PVR_SEPARATE-FLAT_MIX')) else
         '两个 PVR 变体都未证明在本轮 population 上稳定优于 flat concat；不能仅用单 seed 或点估计宣称成功。'), '',
        '### 3. concat 是否主要瓶颈？', '', result('PVR_SEPARATE-PVR_BUNDLE')+'。']
    phase_positive = any(phase_a['paired_bootstrap'][m]['ColHit@1']['ci95'][0]>0 for m in ('Mean','LME'))
    lines += [('Phase A 与受控 Separate 比较共同支持存在拼接瓶颈。' if phase_positive and positive('PVR_SEPARATE-PVR_BUNDLE') else
        'Phase A 与受控 Separate 比较没有共同提供稳定正向证据，因此不能把 concat 定为明确的主要瓶颈。'), '',
        '### 4. Evidence 的 Correction 是否稳定大于 Damage？', '']
    stable = []
    for arm in pvr_arms:
        case_positive = all(metrics['test'][arm][s]['subsets']['non_empty']['net_correction_count']>0 for s in ('13','29'))
        supported = case_positive and positive(arm+'-PRIOR')
        if supported:
            stable.append(arm)
        lines.append(f'- {arm}：'+('两个 seed 净纠错均为正，且 query-macro H1 增益 CI 下界大于 0。' if supported else
            '未同时满足两个 seed 净纠错为正与 query-macro H1 增益 CI 下界大于 0。')+result(arm+'-PRIOR')+'。')
    lines += ['', '### 5. 是否应保留 Top2/Top3，把 evidence 后移到 row-level 属性恢复？', '']
    admission = mean(metrics['test']['PRIOR'][s]['subsets']['full']['PriorAdmissionAt3'] for s in ('13','29'))
    lines.append(f'PriorAdmission@3（两个 seed query-macro 均值）为 {percent(admission)}。')
    if stable:
        lines.append('本轮有方法显示稳定的净纠错，不能预设 evidence 不适合列选择。应保留已验证的列重排收益，并在下一轮独立检验 row-level 属性恢复；本轮未测量后者。')
    else:
        lines.append('本轮未证实 evidence 主导列选择的稳定净收益。保留 Q+T Top2/Top3 并把主要 evidence 工作后移到 row-level 属性恢复，是与结果相容的下一轮设计，而不是本轮已经验证的恢复能力。应停止用更多 fusion 调参替代对恢复机制的直接检验。')
    lines += ['', '## Support Auxiliary', '',
        f'明确映射到审核 witness 的 train Q-T：{support["pairs_with_explicit_witness"]}；阈值 500；执行：{support["execute"]}。',
        '仅明确审核 witness 为 positive；unknown natural evidence 始终为 unknown；negative 仅现有 Shuffled-E donor synthetic_negative；pairwise ranking lambda=0.2，不扫参。', '',
        '## 计算成本', '',
        '下表为 test 每 Q-T 冷启动串行 reader 成本的汇总；Separate 不属于同成本或效率优化。Prior 可缓存，训练 epoch 不重复运行 reader。', '',
        '| 方法 | forwards 合计 | tokens 合计 | image pixels 合计 | p50 / p95 秒 | peak VRAM GiB | logical cache MiB |',
        '|---|---|---|---|---|---|---|']
    for arm in ('PRIOR','FLAT','PVR_BUNDLE','PVR_SEPARATE','BUNDLE_EVIDENCE_ONLY','SEPARATE_EVIDENCE_ONLY'):
        c = costs['cold_per_qt']['test']['O-R'][arm]['full']
        lines.append(f'| {arm} | {c["reader_forwards"]} | {c["total_tokens"]} | {c["image_pixels"]} | {c["latency_p50_seconds"]:.3f} / {c["latency_p95_seconds"]:.3f} | {c["peak_vram_bytes"]/2**30:.3f} | {c["logical_cache_bytes"]/2**20:.3f} |')
    lines += ['', '成本限制：复用 R1 的 latency 为历史测量，VRAM 为原 cache-worker 上界；R2 为 CUDA 同步逐 pass 测量。R1 pixels 从已保存 resize 尺寸与锁定 processor 重建，R2 直接测 image_grid。非新鲜端到端服务延迟。',
        '每 Q-T 明细、evidence_count、batch size、prior/evidence 分离成本、新计算 forwards 与实际唯一 cache bytes 见 COST/。', '',
        '## Artifacts', '',
        '- SOURCE_AUDIT.json / R1_BASELINE_REPLAY.json：基线一致性。',
        '- NATURAL_TRAIN/MANIFEST.json：冻结自然检索及空 evidence 原因。',
        '- 各方法 checkpoints/*/MANIFEST.json、history.json、FEATURES.json：初始化、20 epochs、selected checkpoint 与 feature 哈希。',
        '- PRIOR/SHORTLIST_MANIFEST.json：固定 label-blind shortlist。',
        '- MODEL_SELECTION_LOCK.json / FORMAL_TEST.json：dev 锁定与正式 test 执行记录。',
        '- ANALYSIS/：全部分项、10,000 次 paired bootstrap、逐例纠错与伤害。',
        '- PHASE_A/：oracle 仅诊断；逐 evidence、LOO、order 原始预测。',
        '- COST/：完整 reader 成本与缓存审计。']
    (output/'FINAL_REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    (output/'README.zh-CN.md').write_text('# S2-COL-R2\n\n入口：src/run_stage2_columns_r2.py。最终结论见 FINAL_REPORT.zh-CN.md。\n\n所有新 reader pass 使用本机两张 RTX 4090；小型 head 使用 CPU，与 R1 recipe 一致。\n\n原始预测与指标均保留，不改变 R1 artifacts，不将 oracle-single 作为方法。\n')
    flips = read_json(output/'ANALYSIS/flip_analysis.json')['cases']
    error_lines = ['# 纠错与伤害案例', '', '按固定字典序取前 20 个 test 案例，不按案例效果挑选；全量见 flip_analysis.json。', '']
    chosen = sorted((r for r in flips if r['split']=='test'),key=lambda r:(r['arm'],r['seed'],r['query_id'],r['target_id']))[:20]
    for r in chosen:
        error_lines.append(f'- {r["arm"]} seed {r["seed"]} Q={r["query_id"]} T={r["target_id"]}：gold={r["gold_column_indices"]}；prior={r["prior_ranking"]}；final={r["final_ranking"]}；E={r["evidence_ids"]}；corrected={r["corrected"]} damaged={r["damaged"]}。')
    (output/'ANALYSIS/error_cases.md').write_text('\n'.join(error_lines)+'\n')
    write_json(output/'EXECUTION_MANIFEST.json', {'scope':'EntiTables implicit positive-target columns only',
        'planned':list(ARMS_FOR_REPORT)+['PVR_SUPPORT conditional'], 'implemented':True,
        'actually_executed':arms, 'actually_evaluated':{'Phase A':'dev only','trained_arms':['dev','test']},
        'support_executed':support['execute'], 'formal_test':formal,
        'files':{str(p.relative_to(output)):file_hash(p) for p in
            [output/'FINAL_REPORT.zh-CN.md',output/'MODEL_SELECTION_LOCK.json',output/'ANALYSIS/paired_bootstrap.json',
             output/'COST/reader_costs.json',output/'COST/cache_costs.json']}})


ARMS_FOR_REPORT = ('OO_CONTROL','FLAT_MIX','PRIOR','PVR_BUNDLE','PVR_SEPARATE')

#!/usr/bin/env python
"""Complete the Stage2 report from verified predictions, traces, and review notes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mmdd_stage2.column_data import file_hash, read_jsonl, write_json
from mmdd_stage2.column_metrics import KS, aggregate, evaluate, pair_key
from mmdd_stage2.column_reporting import report

METRICS = ['ColHit@1', 'ColHit@2', 'ColHit@3', 'ColHit@5', 'MRR']


def paired_summary(high: list[dict], low: list[dict]) -> dict[str, Any]:
    """Query-macro differences with source-cluster resampling, including failures."""
    other = {pair_key(r): r for r in low}
    if set(other) != {pair_key(r) for r in high}:
        raise ValueError('Paired diagnostic populations differ')
    queries = defaultdict(list)
    for r in high:
        key = (r['dataset'], r['query_id'], r['source_table_id'])
        queries[key].append([r[m] - other[pair_key(r)][m] for m in METRICS])
    clusters = defaultdict(list)
    for (lake, _, source), rows in queries.items():
        clusters[(lake, source)].append(np.mean(rows, axis=0))
    totals = np.array([np.sum(v, axis=0) for v in clusters.values()])
    counts = np.array([len(v) for v in clusters.values()])
    draw = np.random.default_rng(13).integers(0, len(clusters), size=(1000, len(clusters)))
    boot = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None]
    delta = totals.sum(axis=0) / counts.sum()
    return {'queries': len(queries), 'source_groups': len(clusters), 'bootstrap_replicates': 1000,
            'delta': dict(zip(METRICS, delta.tolist(), strict=True)),
            'CI95': {m: np.quantile(boot[:, i], [.025, .975]).tolist() for i, m in enumerate(METRICS)}}


def finalize(output: Path) -> None:
    def read(name: str) -> Any:
        return json.loads((output/name).read_text())

    validation = read('EXECUTION_VALIDATION.json')
    if not validation['passed']:
        raise ValueError('Run the independent execution audit first')
    result = report(output)
    if not result['complete']:
        raise ValueError('The formal evaluation matrix is incomplete')
    population = {s: read_jsonl(output/f'COLUMN_POPULATION.{s}.jsonl') for s in ('dev', 'test')}
    diagnostics = read('DIAGNOSTIC_SELECTION.json')
    selected = diagnostics['arm']
    evaluated: dict[tuple, tuple] = {}
    predicted: dict[tuple, dict] = {}

    def scores(arm: str, seed: int, split: str, condition: str, view: int = 0) -> tuple:
        key = (arm, seed, split, condition, view)
        if key not in evaluated:
            path = output/'PREDICTIONS'/arm/str(seed)/f'{split}.{condition}.view{view}.jsonl.gz'
            outputs = read_jsonl(path)
            predicted[key] = {pair_key(r): r for r in outputs}
            evaluated[key] = evaluate(population[split], outputs)
        return evaluated[key]

    input_effects, perturbations, conditional_c50 = [], [], []
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            for split in ('dev', 'test'):
                normal = scores(arm, seed, split, 'O-O')[1]
                natural = scores(arm, seed, split, 'O-R')[1]
                input_effects.append({'arm': arm, 'seed': seed, 'split': split, 'contrast': 'O-R minus O-O',
                                     **paired_summary(natural, normal)})
    for seed in (13, 29):
        normal = scores(selected, seed, 'dev', 'O-O')[1]
        for condition, view in [('No-E', 0), ('Shuffled-E', 0), ('ValueShuffle', 0), ('O-O', 2)]:
            metrics, rows = scores(selected, seed, 'dev', condition, view)
            perturbations.append({'arm': selected, 'seed': seed, 'condition': condition, 'view': view,
                                  'metrics': metrics, 'contrast': 'perturbed minus normal O-O view0',
                                  **paired_summary(rows, normal)})

    c50 = {r['query_id']: {x['target_id'] for x in r['results']} for r in read_jsonl(output/'FROZEN_C50.jsonl.gz')}
    admitted = {pair_key(r) for r in population['dev'] if r['target_id'] in c50.get(r['query_id'], set())}
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            rows = scores(arm, seed, 'dev', 'O-R')[1]
            conditional_c50.append({'arm': arm, 'seed': seed,
                                    'conditional_on_admitted_correct_T': aggregate([r for r in rows if pair_key(r) in admitted])})

    costs = []
    current_input = file_hash(output/'INPUT_MANIFEST.json')
    for path in sorted((output/'CACHE_MANIFESTS').glob('*.json')):
        m = json.loads(path.read_text())
        c = m['contract']
        if c['input_manifest_sha256'] != current_input:
            continue
        costs.append({'manifest': str(path.relative_to(output)), 'sha256': file_hash(path),
                      **{k: c[k] for k in ('reader_layout_version', 'split', 'condition', 'view', 'limit')},
                      **{k: m[k] for k in ('gpu_name', 'peak_gpu_memory_bytes', 'cache_bytes', 'latency_p50', 'latency_p95', 'elapsed_seconds')},
                      'pairs': len(m['records']), 'failures': [r for r in m['records'] if r['status'] != 'ok']})
    supplement = {'evaluated': True, 'diagnostic_recipe': diagnostics, 'input_effects': input_effects,
                  'perturbations': perturbations, 'C50_conditional_columns': conditional_c50, 'cache_costs': costs,
                  'evidence_support': read('EVIDENCE_SUPPORT_AUDIT.json'),
                  'interpretation': 'Input perturbations hold model/Q/T fixed. Unknown donor support is not a verified negative. C50 columns are conditional availability, not joins.'}
    write_json(output/'SUPPLEMENTAL_EVALUATION.json', supplement)

    notes = read('DEV_REVIEW_NOTES.json')
    keys = read('ERROR_REVIEW_DEV_KEYS.json')['keys']
    note_index = {(r['query_id'], r['target_id']): r['note'] for r in notes['notes']}
    if set(note_index) != {tuple(k) for k in keys}:
        raise ValueError('Qualitative review differs from the frozen dev keys')
    objects = read_jsonl(output/'OBJECTS.jsonl.gz')[0]
    labels = {(r['query_id'], r['target_id']): r for r in population['dev']}
    review = ['# 固定 dev 案例审阅', '',
              '24 个 Q–T 按预测前冻结的 hash 顺序选取，包含成功与失败；不挑选 test 错例。',
              'Codex 对正文、表头/行值及六张整图进行了辅助定性审阅，未经外部人工裁决。严格发布标签未修改。',
              '定性审阅检查正文段首、相关事实与可见整图，不构成全部截断正文的独立支持验证；图片见 DEV_REVIEW_IMAGES.jpg。能读出线索不等于所有行均获支持。', '',
              '| query / target | 严格正确列 | C0 O-O rank 13/29 | C1 O-O rank 13/29 | C2 O-O rank 13/29 | 选定模型 O-R rank 13/29 | 选定模型 Top1 O-O；O-R（13/29） | 审阅 |',
              '|---|---|---|---|---|---|---|---|']
    review_rows = []
    for query, target in keys:
        p = labels[(query, target)]
        names = {c['column_index']: c['column_name'] for c in objects['targets'][target]['columns']}
        ranks = {}
        for arm, condition in [('C0','O-O'), ('C1','O-O'), ('C2','O-O'), (selected,'O-R')]:
            ranks[f'{arm}.{condition}'] = [next(r['rank'] for r in scores(arm, seed, 'dev', condition)[1]
                                                    if pair_key(r) == pair_key(p)) for seed in (13,29)]
        rank_text = [' / '.join(str(x) if x is not None else 'FAIL' for x in ranks[f'{a}.{c}'])
                     for a,c in [('C0','O-O'), ('C1','O-O'), ('C2','O-O'), (selected,'O-R')]]
        gold = ', '.join(f'{c}:{names[c]}' for c in p['gold_column_indices'])
        note = note_index[(query,target)]
        top_names = {}
        for condition in ('O-O','O-R'):
            top_names[condition] = []
            for seed in (13,29):
                record = predicted[(selected, seed, 'dev', condition, 0)][pair_key(p)]
                top_names[condition].append(names[record['ranking'][0]] if record['status'] == 'ok' else 'FAIL')
        tops = '; '.join(' / '.join(top_names[c]) for c in ('O-O','O-R'))
        review.append(f'| {query}<br>{target} | {gold} | ' + ' | '.join(rank_text) + f' | {tops} | {note} |')
        review_rows.append({'query_id': query, 'target_id': target, 'gold': gold, 'ranks': ranks,
                            'selected_model_top1': top_names, 'note': note})
    review += ['', '标注歧义仅作为后续独立审计建议；不自动把同名/同值列视为同一属性，也不回改本轮模型选择。',
               '若 rank>3 在这 24 个预先固定样本中未出现，报告未覆盖，不能用事后挑样补成有代表性的错误率。']
    (output/'ERROR_CASES.md').write_text('\n'.join(review)+'\n')
    write_json(output/'ERROR_CASES.json', {'evaluated': True, 'reviewer': notes['reviewer'], 'records': review_rows})

    result['execution_validation'] = validation
    result['supplemental_evaluation'] = 'SUPPLEMENTAL_EVALUATION.json'
    interpretation_path = output/'SCIENTIFIC_INTERPRETATION.md'
    result['scientific_interpretation'] = str(interpretation_path) if interpretation_path.is_file() else None
    for stage in ('NoE_ShuffledE_value_order_diagnostics', 'C50_input_compatibility', 'fixed_dev_case_review',
                  'trajectory_export', 'independent_execution_audit', 'published_support_annotation_audit'):
        result['stage_status'][stage] = dict(planned=True, implemented=True, executed=True, evaluated=True)
    for name in ('TRAJECTORIES.csv', 'TRAJECTORIES.png', 'TRAJECTORIES.pdf'):
        if not (output/name).is_file():
            raise ValueError('Export real trajectories before finalizing')
    result['limitations'] = [
        'Only EntiTables canonical artifacts are available; WDC2K is missing qrels/tables/assets/recoveries.',
        'B13+T0 is a hash-verified historical anchor, not a verified latest strongest pipeline.',
        'Published upstream support annotations are not independent post-truncation validation; primary support remains unknown.',
        'Shared entities and qualitative schema/label ambiguities limit missing-attribute and generalization claims.',
        'C3 and optional raw-Qwen-column are not run; no Stage1 training, value generation, crop, or join verification.',
    ]
    write_json(output/'RESULTS.json', result)

    lines = (output/'RESULTS.zh-CN.md').read_text().splitlines()
    lines += ['', '## 完成核验与训练轨迹', '',
              'EXECUTION_VALIDATION.json 独立重建访问顺序、optimizer steps 和 dev 选择；核验 checkpoint 文件与参数 hash，及 C0/C1 初值一致。',
              '每条正式臂完成 20 × 7029 = 140580 次基础样本 visits；头部训练用 CPU，冻结 9B 特征仅在本机 GPU 1 RTX 4090 上生成。',
              '每 epoch 的 loss/dev MRR/梯度曲线见 TRAJECTORIES.png、PDF 与 CSV。缓存峰值显存、大小、延迟及失败明细见 SUPPLEMENTAL_EVALUATION.json/cache_costs。', '',
              '曲线中的训练 CE 是成功处理样本的未加权 pair 均值；实际反向传播按湖等权、湖内 query 等权加权。分模态沿用冻结人口的正 E 模态，O-R 实际证据可能改变或为空。', '',
              '## 主指标分母、机会基线与列预算', '',
              '| split | queries / pairs | Random@1/2/3/5 | 平均列预算@1/2/3/5 | 候选占比@1/2/3/5 |',
              '|---|---|---|---|---|']
    for split in ('dev','test'):
        m = scores('C0',13,split,'O-O')[0]
        values = [' / '.join(f"{m['query_macro'][f'{prefix}@{k}']:.6f}" for k in KS)
                  for prefix in ('Random','column_budget','candidate_fraction')]
        lines.append(f"| {split} | {m['queries']} / {m['pairs']} | " + ' | '.join(values) + ' |')
    lines += ['', '预算按人口平均 min(k,M)，故障另列。严格 Hit 由同一 canonical-ID tie-break 排序计算。', '',
              '| Arm | seed | split | E | success | failure pairs | M>3 pairs | M>3 Hit3 |',
              '|---|---:|---|---|---:|---:|---:|---:|']
    for r in result['matrix']:
        m = r['metrics']; subset = m['non_saturated']['3']; sm = subset['query_macro']
        hit = f"{sm['ColHit@3']:.6f}" if sm else 'N/A'
        lines.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {r['condition']} | {m['query_macro']['success_rate']:.6f} | {sum(m['failure_reasons'].values())} | {subset['pairs']} | {hit} |")
    lines += ['', '## 独立历史参考与位置对照', '',
              '| Reference | split | E | queries/pairs | Hit1 | Hit2 | Hit3 | Hit5 | MRR |',
              '|---|---|---|---|---:|---:|---:|---:|---:|']
    for split in ('dev','test'):
        for condition in ('O-O','O-R'):
            m = read(f'METRICS/P0-R25/reference/{split}.{condition}.original.json')
            values = ' | '.join(f"{m['query_macro'][k]:.6f}" for k in METRICS)
            lines.append(f"| P0-R25 | {split} | {condition} | {m['queries']}/{m['pairs']} | {values} |")
        m = read(f'BASELINES/{split}.position.json')
        values = ' | '.join(f"{m['query_macro'][k]:.6f}" for k in METRICS)
        lines.append(f"| train-only display position | {split} | 不读E | {m['queries']}/{m['pairs']} | {values} |")
    lines += ['', 'P0-R25 保留历史列序、asset ID 与图像策略，不与 C0 作单变量因果差分。', '',
              '## 受控臂差分', '',
              '| contrast | seed | split | E | ΔHit1 | ΔMRR | ΔMRR source-cluster 95% CI |',
              '|---|---:|---|---|---:|---:|---|']
    paired = read('PAIRED_BOOTSTRAP.json')['comparisons']
    for p in paired:
        high, low = p['contrast'].split('-')
        a = scores(high,p['seed'],p['split'],p['condition'])[0]['query_macro']
        b = scores(low,p['seed'],p['split'],p['condition'])[0]['query_macro']
        ci = p['CI95']['MRR']
        lines.append(f"| {p['contrast']} | {p['seed']} | {p['split']} | {p['condition']} | {a['ColHit@1']-b['ColHit@1']:.6f} | {a['MRR']-b['MRR']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines += ['', '## 列数分桶', '', '| Arm | seed | split | E | M | queries/pairs | Hit1 | Hit2 | Hit3 | Hit5 | MRR |',
              '|---|---:|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['matrix']:
        for bucket in ('1','2','3','4-5','6-10','>10'):
            m = r['metrics']['by_column_count_bucket'].get(bucket)
            text = ' | '.join(f"{m['query_macro'][k]:.6f}" for k in METRICS) if m else 'N/A | N/A | N/A | N/A | N/A'
            lines.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {r['condition']} | {bucket} | {str(m['queries'])+'/'+str(m['pairs']) if m else '0/0'} | {text} |")
    lines += ['', '## 固定模型的输入扰动', '',
              f"诊断 recipe 仅由 dev 选出：{selected}。差分是扰动减正常 O-O；95% CI 按 source cluster bootstrap。", '',
              '| seed | condition / view | Hit1 | MRR | ΔHit1 [95% CI] | ΔMRR [95% CI] |',
              '|---:|---|---:|---:|---|---|']
    for p in perturbations:
        m=p['metrics']['query_macro']
        diffs=['{:.6f} [{:.6f}, {:.6f}]'.format(p['delta'][k],*p['CI95'][k]) for k in ('ColHit@1','MRR')]
        lines.append(f"| {p['seed']} | {p['condition']} / {p['view']} | {m['ColHit@1']:.6f} | {m['MRR']:.6f} | " + ' | '.join(diffs) + ' |')
    lines += ['', 'O-R−O-O 每臂/seed 的成对差分与 CI、自然 E 空/非空分项、C50 中已召回正确 T 的条件列指标均保存在补充 JSON。',
              '## 证据支持与定性限制', '',
              'EVIDENCE_SUPPORT_AUDIT.json 只统计既有审查记录与未截断正文保留率，不生成属性值，不把无标注自然 E 当负证据。',
              '固定 24 个 dev 案例及所有臂 rank 见 ERROR_CASES.md；含跨列标题行污染、时态不同的同类型列，以及 Language/Language_1 疑似语义重复。未经独立裁决，不修改主标签。',
              'explicit 的缺失属性指标为 N/A：它属于可见列直接连接，未与 implicit 缺失属性混算。',
              'C3 未执行：本轮按用户优先范围完成 P0/C0/C1/C2；自然 E train artifact 未建立。raw-Qwen 表级历史引用保留，未重跑他人负责的 baseline。',
              '实际硬件为经用户修正授权的本机 RTX 4090 GPU 1；不是 A100 实验。']
    if interpretation_path.is_file():
        lines += ['', interpretation_path.read_text()]
    (output/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    required = ['INPUT_MANIFEST.json','DATA_AUDIT.json','MODEL_SELECTION.json','UNIT_TEST_RESULTS.json',
                'PER_QUERY_DIFFERENCES.jsonl','ERROR_CASES.md','RESULTS.zh-CN.md','RESULTS.json',
                'EXECUTION_VALIDATION.json','SUPPLEMENTAL_EVALUATION.json','TRAJECTORIES.csv',
                'EXECUTION_COMPLETION.json','PAIRED_BOOTSTRAP.json','EVIDENCE_SUPPORT_AUDIT.json',
                'TRAJECTORIES.png','TRAJECTORIES.pdf',
                *[f'COLUMN_POPULATION.{s}.jsonl' for s in ('train','dev','test')]]
    if interpretation_path.is_file():
        required.append(interpretation_path.name)
    write_json(output/'DELIVERY_MANIFEST.json', {'scope':'P0 + C0/C1/C2 on available EntiTables lake',
        'files': {name: file_hash(output/name) for name in required}, 'states': result['stage_status'],
        'execution_evidence':'EXECUTION_VALIDATION.json + original receipts/visits/checkpoints + recomputed predictions',
        'large_artifacts':'CACHE_MANIFESTS, CHECKPOINTS, PREDICTIONS, METRICS remain local; each formal receipt records its checkpoint/cache hashes'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    finalize(parser.parse_args().output)

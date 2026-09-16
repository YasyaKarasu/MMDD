"""Historical replay and artifact-based status reporting."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any
import torch
from .checkpoints import load_candidate_scorer
from .column_data import file_hash, write_json, write_jsonl
from .column_data import digest
from .column_metrics import evaluate, prediction
from .column_cache import source_fingerprints
from .reader_cache import load_reader_cache


def replay_legacy(output: Path, checkpoint: Path, cache_root: Path) -> None:
    torch.set_num_threads(4)
    scorer = load_candidate_scorer(checkpoint, torch.device('cpu')).eval()
    model_hash = file_hash(checkpoint)
    receipt = {'planned': True, 'implemented': True, 'executed': False, 'evaluated': False,
               'command': sys.argv, 'checkpoint': str(checkpoint), 'checkpoint_sha256': model_hash,
               'scope': 'P0 historical Round1 population; not current v9 or controlled C0 comparison',
               'legacy_cache': True, 'identity_limit': 'Legacy cache has no reader-source/weight-content fingerprint; cannot retroactively verify them',
               'sources': source_fingerprints(), 'splits': {}}
    for split in ('train', 'dev', 'test'):
        dirs = sorted(p.parent for p in cache_root.glob(f'*/{split}/manifest.json'))
        records, fingerprint = load_reader_cache(dirs)
        if len(records) != sum(json.loads((p/'manifest.json').read_text())['sample_count'] for p in dirs):
            raise ValueError('Historical cache sample count is incomplete; cannot reconstruct frozen population')
        if split == 'train':
            meta = torch.load(checkpoint, map_location='cpu', weights_only=True)['metadata']
            # Original runner fingerprints train and dev, in lake/split order.
            # _cache_dirs iterates datasets, then the requested (train, dev) tuple.
            training_manifests = [json.loads((cache_root/lake/s/'manifest.json').read_text())['metadata_fingerprint']
                                  for lake in ('entitables', 'wdc') for s in ('train', 'dev')]
            if meta.get('reader_cache_fingerprint') != digest(training_manifests):
                raise ValueError('Historical checkpoint does not reference these training caches')
        population, outputs, old_mrr, ties = [], [], [], 0
        for r in records:
            p = {k: r[k] for k in ('dataset', 'query_id', 'target_id', 'candidate_column_indices')}
            p.update(split=split, gold_column_indices=[r['gold_column_index']], modality='+'.join(r['evidence_modalities']))
            population.append(p)
            with torch.inference_mode():
                logits = scorer(r['open_states'], r['close_states']).tolist()
            outputs.append(prediction(p, logits, model_hash=model_hash, input_hash=fingerprint))
            gold_logit = logits[r['gold_column_position']]
            old_mrr.append(1 / (1 + sum(x > gold_logit for x in logits)))
            ties += sum(x == gold_logit for x in logits) > 1
        write_jsonl(output / 'P0_HISTORICAL_POPULATION' / f'{split}.jsonl', population)
        metrics, _ = evaluate(population, outputs)
        metrics['old_pair_micro_mrr'] = sum(old_mrr) / len(old_mrr)
        metrics['gold_tied_pairs'] = ties
        metrics['gold_display_positions'] = {str(i): sum(r['candidate_column_indices'].index(r['gold_column_indices'][0]) == i for r in population)
                                            for i in sorted({r['candidate_column_indices'].index(r['gold_column_indices'][0]) for r in population})}
        metrics['position_baseline_note'] = 'Every historical gold is at display position zero; accuracy does not establish attribute understanding'
        write_jsonl(output / 'PREDICTIONS/P0/13' / f'{split}.historical.jsonl.gz', outputs)
        write_json(output / 'METRICS/P0/13' / f'{split}.historical.json', metrics)
        receipt['splits'][split] = {'pairs': len(records), 'cache_fingerprint': fingerprint,
                                    'cache_manifests': [{'path': str(p / 'manifest.json'), 'sha256': file_hash(p / 'manifest.json')} for p in dirs]}
        print(f'P0 {split}: {len(records)} pairs; query-macro {metrics["query_macro"]}', flush=True)
    receipt.update(executed=True, evaluated=True)
    write_json(output / 'RUN_RECEIPTS/P0/13.json', receipt)


def report(output: Path) -> dict[str, Any]:
    """Report only validated receipts and metrics recomputed from saved predictions."""
    from .column_data import read_jsonl
    def optional(name: str) -> Any:
        path = output / name
        return json.loads(path.read_text()) if path.is_file() else None
    audit = optional('DATA_AUDIT.json')
    rows = []
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            receipt_path = output / 'RUN_RECEIPTS' / arm / f'{seed}.json'
            receipt = optional(str(receipt_path.relative_to(output))) or {}
            executed = bool(receipt.get('executed'))
            if executed:
                history = json.loads(Path(receipt['history_path']).read_text())
                count = len(read_jsonl(output/'COLUMN_POPULATION.train.jsonl'))
                if len(history) != 20 or any(r['base_visits'] != count or r['unique_base_visits'] != count for r in history):
                    raise ValueError('Formal receipt lacks 20 full population visits')
                for name, expected in receipt['checkpoint_hashes'].items():
                    if file_hash(output/'CHECKPOINTS'/arm/str(seed)/name) != expected:
                        raise ValueError('Formal checkpoint bytes disagree with receipt')
            for split in ('dev', 'test'):
                population = read_jsonl(output/f'COLUMN_POPULATION.{split}.jsonl')
                for condition in ('O-O', 'O-R'):
                    prediction_path = output/'PREDICTIONS'/arm/str(seed)/f'{split}.{condition}.view0.jsonl.gz'
                    metric_path = output/'METRICS'/arm/str(seed)/f'{split}.{condition}.view0.json'
                    metrics = None
                    if prediction_path.is_file() and metric_path.is_file() and executed:
                        metrics, _ = evaluate(population, read_jsonl(prediction_path))
                        saved = json.loads(metric_path.read_text())
                        if metrics['query_macro'] != saved['query_macro']:
                            raise ValueError('Saved metrics disagree with saved predictions')
                    coverage = sum(r.get('natural_retrieval_available', False) for r in population)
                    rows.append({'arm': arm, 'seed': seed, 'split': split, 'condition': condition,
                                 'planned': True, 'implemented': True, 'executed': executed,
                                 'evaluated': metrics is not None, 'metrics': metrics,
                                 'input_complete': condition == 'O-O' or coverage == len(population),
                                 'receipt': str(receipt_path) if receipt else None})
    p0 = optional('RUN_RECEIPTS/P0/13.json')
    r25 = optional('RUN_RECEIPTS/P0-R25/reference.json')
    visibility = optional('VISIBILITY_PROBE.json')
    tiny = [optional(f'RUN_RECEIPTS/{arm}/tiny_{seed}.json') for arm in ('C0','C1','C2') for seed in (13,29)]
    paired = optional('PAIRED_BOOTSTRAP.json')
    required = ['INPUT_MANIFEST.json', 'DATA_AUDIT.json', 'MODEL_SELECTION.json', 'UNIT_TEST_RESULTS.json',
                'COLUMN_POPULATION.train.jsonl', 'COLUMN_POPULATION.dev.jsonl', 'COLUMN_POPULATION.test.jsonl',
                'PER_QUERY_DIFFERENCES.jsonl', 'ERROR_CASES.md']
    result = {'experiment': 'S2-COL-R1', 'complete': all(r['evaluated'] and r['input_complete'] for r in rows),
              'controlled_training_complete': all(r['executed'] for r in rows), 'matrix': rows,
              'artifact_presence_only_not_execution_proof': {name: (output/name).is_file() for name in required},
              'P0_historical_round1': p0, 'P0_R25_current_population': r25,
              'visibility_probe': visibility, 'tiny_receipts': tiny,
              'stage_status': {
                  'source_audit_and_population': {'planned': True, 'implemented': True, 'executed': bool(audit), 'evaluated': bool(audit and audit['population_locked'])},
                  'P0_historical_round1': {'planned': True, 'implemented': True, 'executed': bool(p0 and p0['executed']), 'evaluated': bool(p0 and p0['evaluated'])},
                  'P0_r25_current_population': {'planned': True, 'implemented': True, 'executed': bool(r25 and r25['executed']), 'evaluated': bool(r25 and r25['evaluated'])},
                  'real_9B_visibility': {'planned': True, 'implemented': True, 'executed': bool(visibility), 'evaluated': bool(visibility and visibility['passed'])},
                  'real_tiny_overfit': {'planned': True, 'implemented': True, 'executed': all(t is not None for t in tiny), 'evaluated': all(t and t['passed'] for t in tiny)},
                  'paired_differences': {'planned': True, 'implemented': True, 'executed': bool(paired and paired['evaluated']), 'evaluated': bool(paired and paired['evaluated'])},
                  'C3': {'planned': 'conditional only', 'implemented': False, 'executed': False, 'evaluated': False},
                  'raw_qwen_column': {'planned': 'optional', 'implemented': False, 'executed': False, 'evaluated': False}}}
    for name in ('DATA_AUDIT', 'UNIT_TEST_RESULTS', 'MODEL_SELECTION', 'C50_INPUT_COMPATIBILITY', 'EXECUTION_BLOCKERS'):
        result[name.lower()] = optional(f'{name}.json')
    write_json(output/'RESULTS.json', result)
    lines = ['# S2-COL-R1 执行报告', '',
             '状态由 checkpoint 哈希、逐 epoch 人口遍历记录和保存预测的重新评测验证。', '',
             '主矩阵完成：' + str(result['complete']) + '；受控训练完成：' + str(result['controlled_training_complete']) + '。', '',
             '| Arm | seed | split | E | executed | evaluated | 输入完整 | Hit1 | Hit2 | Hit3 | Hit5 | MRR |',
             '|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        m = r['metrics']['query_macro'] if r['metrics'] else None
        scores = ' | '.join(f"{m[k]:.6f}" if m else 'N/A' for k in ('ColHit@1','ColHit@2','ColHit@3','ColHit@5','MRR'))
        lines.append(f"| {r['arm']} | {r['seed']} | {r['split']} | {r['condition']} | {r['executed']} | {r['evaluated']} | {r['input_complete']} | {scores} |")
    if audit:
        lines += ['', '人口：' + json.dumps(audit['split_counts'], ensure_ascii=False) + '。',
                  '自然检索覆盖：' + json.dumps(audit.get('natural_retrieval_coverage'), ensure_ascii=False) + '。',
                  '未检索整个 query 与已检索 query 的空 E 严格区分。缺输出留在分母；缺检索产物的轨道不用于推断证据噪声效应。',
                  '正 E 的 upstream recovery 审查不等于截断后的支持审查，暂标 support_unknown。',
                  'WDC 2K 未提供 canonical artifacts；当前科学范围仅为可用的 EntiTables 湖。实体重叠在 DATA_AUDIT 披露，不声称 entity-disjoint 泛化。']
    if p0:
        lines += ['', 'P0 历史 Round1 重放：']
        for split in ('train','dev','test'):
            m = optional(f'METRICS/P0/13/{split}.historical.json')
            if m:
                lines.append(f"- {split}: {m['pairs']} pairs，query-macro Hit1={m['query_macro']['ColHit@1']:.6f}，MRR={m['query_macro']['MRR']:.6f}；gold 显示位置分布={m['gold_display_positions']}。")
        lines += ['', '旧人口的固定 gold 位置使多数位置基线饱和；不能据此证明属性理解。P0 不属于 C0/C1/C2 受控差分。R25 当前人口重放另列 receipt。']
    compatibility = result['c50_input_compatibility']
    if compatibility:
        lines += ['', f"固定 C50 implicit-dev 表 admission Recall={compatibility['R_admit']:.6f}，分母={compatibility['queries']} queries / {compatibility['pairs']} Q–T。列联合可用率见独立 JSON。"]
    lines += ['', '分桶、非饱和子集、随机基线、列预算、候选占比、成功率、失败原因及 pair-micro 详见每条 METRICS。',
              'C1−C0 检验布局；C2−C1 是包含学习率差异的 head recipe 比较。无匹配预测时 PER_QUERY_DIFFERENCES 为空，不能当作零差分。',
              'B13+T0 为经过哈希核验的历史锚点，不声称最新最强。raw-Qwen 仅保留已有表级参考。',
              '列 Hit / MRR 不代表表 Recall、填值准确率或 semantic-joinability。尚未完成的阶段不作因果结论。']
    (output/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n')
    return result

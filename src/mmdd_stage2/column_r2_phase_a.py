"""Dev-only fixed-checkpoint evidence interference diagnostics."""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .checkpoints import load_candidate_scorer
from .column_data import file_hash, read_jsonl, write_json, write_jsonl
from .column_metrics import pair_key, prediction
from .column_r2_cache import FeatureIndex, phase_a_variants
from .column_r2_metrics import report_metrics, paired_bootstrap, query_mean


def normalized_aggregates(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logp = logits.log_softmax(dim=-1)
    return logp.mean(0), torch.logsumexp(logp, 0) - math.log(len(logp))


@torch.inference_mode()
def evaluate_phase_a(r1: Path, output: Path) -> None:
    torch.set_num_threads(4)
    index = FeatureIndex(r1, output)
    objects = read_jsonl(r1 / 'OBJECTS.jsonl.gz')[0]
    all_inputs = read_jsonl(r1 / 'COLUMN_INPUTS.dev.jsonl')
    inputs = [r for r in all_inputs if r['evidence_ids']['O-R']]
    metadata = {pair_key(p): p for p in read_jsonl(r1 / 'COLUMN_POPULATION.dev.jsonl')}
    population = []
    for item in inputs:
        ids = item['evidence_ids']['O-R']
        modality = '+'.join(sorted({objects['evidence'][e]['asset_type'] for e in ids}))
        population.append({**metadata[pair_key(item)], 'modality': modality, 'evidence_count': len(ids)})
    all_predictions, loo, perturbations, results, seed_rows = [], [], [], {}, {}
    full_seed_rows = {}
    for seed in (13, 29):
        cp = r1 / f'CHECKPOINTS/C2/{seed}/selected.pt'
        head = load_candidate_scorer(cp, torch.device('cpu'), expected_reader_layout='tail_candidates_v1').eval()
        methods = {m: [] for m in ('bundle', 'Mean', 'LME', 'oracle_single_evidence_upper_bound')}
        flips = []
        for item, meta in zip(inputs, population, strict=True):
            predictions, logits = {}, {}
            variants = phase_a_variants(item)
            for name, ids in variants.items():
                features = index.get(item, ids)
                scores = head(features['open_states'], features['close_states'])
                logits[name] = scores
                predictions[name] = prediction(features, scores.tolist(), seed=seed, variant=name,
                    evidence_ids=ids, model_hash=file_hash(cp), input_hash=features['input_hash'])
            singles = [logits[f'single_{j}'] for j in range(len(item['evidence_ids']['O-R']))]
            avg, lme = normalized_aggregates(torch.stack(singles))
            methods['bundle'].append(predictions['bundle'])
            for name, scores in [('Mean', avg), ('LME', lme)]:
                methods[name].append(prediction(predictions['bundle'], scores.tolist(), seed=seed, variant=name))
            gold = set(meta['gold_column_indices'])
            single_predictions = [predictions[f'single_{j}'] for j in range(len(singles))]
            # Explicit diagnostic upper bound; never consumed by training or selection.
            oracle = min(single_predictions, key=lambda p: next(i for i, c in enumerate(p['ranking']) if c in gold))
            methods['oracle_single_evidence_upper_bound'].append({**oracle,
                'variant': 'oracle_single_evidence_upper_bound', 'deployable': False})
            all_predictions.extend(single_predictions)
            bundle_correct = predictions['bundle']['ranking'][0] in gold
            harmful = helpful = 0
            for j, eid in enumerate(item['evidence_ids']['O-R']):
                p = predictions[f'loo_{j}']
                correct = p['ranking'][0] in gold
                harmful += int(not bundle_correct and correct)
                helpful += int(bundle_correct and not correct)
                loo.append({**p, 'removed_evidence_id': eid, 'harmful': not bundle_correct and correct,
                            'helpful': bundle_correct and not correct})
            orders = [predictions[n] for n in ('bundle', 'reverse', 'hash_order')]
            perturbations.extend(orders)
            flips.append({**meta, 'order_flip': float(len({p['ranking'][0] for p in orders}) > 1),
                          'harmful_evidence_count': harmful, 'helpful_evidence_count': helpful,
                          'has_harmful_item': float(harmful > 0)})
        results[str(seed)], seed_rows[seed] = {}, {}
        empty_predictions, empty_population = [], []
        for item in all_inputs:
            if item['evidence_ids']['O-R']:
                continue
            features = index.get(item, [])
            scores = head(features['open_states'], features['close_states'])
            empty_predictions.append(prediction(features,scores.tolist(),seed=seed,evidence_ids=[],
                variant='empty_evidence_No-E_fallback',input_hash=features['input_hash']))
            empty_population.append({**metadata[pair_key(item)],'modality':'none','evidence_count':0})
        full_seed_rows[seed] = {}
        for method, preds in methods.items():
            metrics, rows = report_metrics(population, preds)
            full_metrics, full_rows = report_metrics(population+empty_population,preds+empty_predictions)
            metrics['O-R_full'] = full_metrics
            full_seed_rows[seed][method] = full_rows
            results[str(seed)][method] = metrics
            seed_rows[seed][method] = rows
            write_jsonl(output / f'PHASE_A/{seed}.{method}.jsonl.gz', preds)
            write_jsonl(output / f'PHASE_A/{seed}.{method}.full.jsonl.gz',preds+empty_predictions)
        results[str(seed)]['interference'] = {'order_flip_rate_n_ge_2': query_mean([r for r in flips if r['evidence_count'] >= 2], 'order_flip'),
            'harmful_evidence_count': sum(r['harmful_evidence_count'] for r in flips),
            'helpful_evidence_count': sum(r['helpful_evidence_count'] for r in flips),
            'has_harmful_item_query_macro': query_mean(flips, 'has_harmful_item'),
            'by_evidence_count': {str(n): {f: query_mean([r for r in flips if r['evidence_count'] == n], f)
                for f in ('order_flip', 'harmful_evidence_count', 'helpful_evidence_count')}
                for n in (1, 2, 3, 4)},
            'by_modality': {m: {f: query_mean([r for r in flips if r['modality'] == m], f)
                for f in ('order_flip', 'harmful_evidence_count', 'helpful_evidence_count')}
                for m in sorted({r['modality'] for r in flips})}}
    results['paired_bootstrap'] = {method: {metric: paired_bootstrap(
        [seed_rows[s][method] for s in (13, 29)], [seed_rows[s]['bundle'] for s in (13, 29)], metric)
        for metric in ('ColHit@1', 'MRR')} for method in ('Mean', 'LME')}
    results['paired_bootstrap_full'] = {method: {metric: paired_bootstrap(
        [full_seed_rows[s][method] for s in (13,29)], [full_seed_rows[s]['bundle'] for s in (13,29)],metric)
        for metric in ('ColHit@1','MRR')} for method in ('Mean','LME')}
    write_jsonl(output / 'PHASE_A/per_evidence_predictions.jsonl.gz', all_predictions)
    write_jsonl(output / 'PHASE_A/leave_one_out.jsonl.gz', loo)
    write_jsonl(output / 'PHASE_A/order_perturbation.jsonl.gz', perturbations)
    write_json(output / 'PHASE_A/RESULTS.json', results)
    report = ['# Phase A：dev-only evidence fusion 诊断', '',
        '不训练、不选 test。Mean / LME 先对每个 pass 做 log-softmax；oracle-single 仅为事后 gold 诊断上界，不能部署。',
        'O-R full 保留全部 675 Q-T；non-empty 为 418 Q-T。空 evidence 使用原 C2 No-E 输出，不补 O-O。', '',
        '| seed | 方法 | 子集 | Hit@1 | Hit@2 | Hit@3 | Hit@5 | MRR |', '|---|---|---|---|---|---|---|---|']
    for seed in ('13','29'):
        for method in ('bundle','Mean','LME','oracle_single_evidence_upper_bound'):
            for scope in ('full','non_empty'):
                metrics = results[seed][method]['O-R_full'] if scope=='full' else results[seed][method]
                values = [f'{100*metrics["query_macro"][f]:.3f}%' for f in ('ColHit@1','ColHit@2','ColHit@3','ColHit@5','MRR')]
                report.append(f'| {seed} | {method} | {scope} | '+' | '.join(values)+' |')
    report += ['', '两个 seed 先同 query 平均，再按 source group 做 10,000 次 paired bootstrap：', '']
    for scope, name in (('full','paired_bootstrap_full'),('non_empty','paired_bootstrap')):
        for method in ('Mean','LME'):
            for field in ('ColHit@1','MRR'):
                b = results[name][method][field]
                report.append(f'- {scope} {method} - concat {field}: {100*b["difference"]:+.3f} pp；95% CI [{100*b["ci95"][0]:+.3f}, {100*b["ci95"][1]:+.3f}]。')
    report += ['', 'M>k、modality、evidence_count、pair-micro 的完整指标见 RESULTS.json。',
        '逐 evidence 预测、LOO harmful/helpful 与 evidence order perturbation 均保留原始记录；是否为主瓶颈仍需 PVR-Separate 对 PVR-Bundle 的受控比较。']
    (output/'PHASE_A/RESULTS.zh-CN.md').write_text('\n'.join(report)+'\n')
    print({s: {m: results[str(s)][m]['query_macro']['ColHit@1'] for m in ('bundle', 'Mean', 'LME', 'oracle_single_evidence_upper_bound')} for s in (13,29)}, flush=True)

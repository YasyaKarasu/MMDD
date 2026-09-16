"""Lock dev-selected checkpoints, then run the one formal test and paired analysis."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import torch

from .column_data import file_hash, read_jsonl, write_json, write_jsonl
from .column_metrics import pair_key
from .column_r2_audit import read_json
from .column_r2_metrics import paired_bootstrap, report_metrics
from .column_r2_training import TrainingData, inputs_for, load_head, metadata_for, predict_model

ARMS = ('OO_CONTROL', 'FLAT_MIX', 'PRIOR', 'PVR_BUNDLE', 'PVR_SEPARATE')
SEEDS = (13, 29)


def lock_selection(r1: Path, output: Path) -> None:
    destination = output/'MODEL_SELECTION_LOCK.json'
    if destination.exists():
        raise ValueError('Model selection already locked')
    support = read_json(output/'SUPPORT_AUDIT/MANIFEST.json')
    arms = ARMS + (('PVR_SUPPORT',) if support['execute'] else ())
    checkpoints = {}
    for arm in arms:
        checkpoints[arm] = {}
        for seed in SEEDS:
            folder = output/f'{arm}/checkpoints/{seed}'
            receipt = read_json(folder/'MANIFEST.json')
            history = read_json(folder/'history.json')
            selected = folder/'selected.pt'
            if file_hash(selected) != receipt['selected_sha256'] or len(history) != 20:
                raise ValueError('Selected checkpoint or twenty-epoch schedule mismatch')
            if receipt['optimizer_steps'] != 4400 or any(h['base_samples_seen'] != 7029 or h['unique_base_visits'] != 7029 for h in history):
                raise ValueError('Unexpected base population/exposure')
            checkpoints[arm][str(seed)] = {'path': str(selected), 'sha256': file_hash(selected),
                'manifest_sha256': file_hash(folder/'MANIFEST.json'), 'selected_epoch': receipt['selected_epoch']}
    for seed in SEEDS:
        histories = {a: read_json(output/f'{a}/checkpoints/{seed}/history.json') for a in arms}
        historical = read_json(r1/f'CHECKPOINTS/C2/{seed}/history.json')
        if any(a['parameter_sha256'] != b['parameter_sha256'] or a['visits_sha256'] != b['visits_sha256']
               for a,b in zip(historical,histories['OO_CONTROL'],strict=True)):
            raise ValueError('O-O control training trajectory differs from R1 C2')
        for epoch in range(20):
            if len({h[epoch]['visits_sha256'] for h in histories.values()}) != 1:
                raise ValueError('Base visits differ between arms')
            mixed = [a for a in arms if a not in {'OO_CONTROL', 'PRIOR'}]
            if len({histories[a][epoch]['condition_and_evidence_schedule_sha256'] for a in mixed}) != 1:
                raise ValueError('Flat/Bundle/Separate evidence schedules differ')
        control = read_json(output/f'OO_CONTROL/checkpoints/{seed}/MANIFEST.json')
        flat = read_json(output/f'FLAT_MIX/checkpoints/{seed}/MANIFEST.json')
        for key in ('initial_parameter_sha256', 'optimizer_steps', 'lr', 'weight_decay', 'selection'):
            if control[key] != flat[key]:
                raise ValueError(f'Flat-Mix control differs on {key}')
    write_json(destination, {'locked_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoints': checkpoints, 'selection_data': 'dev only', 'test_evaluated_before_lock': False,
        'support_audit_sha256': file_hash(output/'SUPPORT_AUDIT/MANIFEST.json'),
        'shortlist_sha256': file_hash(output/'PRIOR/SHORTLIST_MANIFEST.json'),
        'matched_exposure_and_schedules': True, 'OO_CONTROL_R1_training_trajectory_exact':True})


@torch.inference_mode()
def formal_test(r1: Path, output: Path) -> None:
    if (output/'FORMAL_TEST.json').exists():
        raise ValueError('Formal test already evaluated; use saved predictions for analysis')
    lock = read_json(output/'MODEL_SELECTION_LOCK.json')
    if file_hash(output/'PRIOR/SHORTLIST_MANIFEST.json') != lock['shortlist_sha256']:
        raise ValueError('Shortlist changed after selection lock')
    for arm, checkpoints in lock['checkpoints'].items():
        for entry in checkpoints.values():
            if file_hash(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Model changed after dev selection lock')
    torch.set_num_threads(4)
    data = TrainingData(r1, output)
    items = inputs_for(r1, output, 'test')
    population = metadata_for(r1, items, data.objects, 'test')
    files = {}
    for seed in SEEDS:
        data.freeze_prior_inputs(seed)
        prior = load_head(output/f'PRIOR/checkpoints/{seed}/selected.pt')[0].requires_grad_(False)
        prior_predictions, _ = predict_model(data, prior, items, 'PRIOR')
        for arm in lock['checkpoints']:
            model = load_head(Path(lock['checkpoints'][arm][str(seed)]['path']))[0]
            predictions, _ = predict_model(data, model, items, arm, prior if arm.startswith('PVR') else None)
            metrics, rows = report_metrics(population, predictions, prior_predictions)
            path = output/f'{arm}/predictions/{seed}/test.jsonl.gz'
            write_jsonl(path, predictions)
            write_json(output/f'{arm}/metrics/{seed}/test.json', metrics)
            write_jsonl(output/f'{arm}/metrics/{seed}/test.rows.jsonl.gz', rows)
            files[f'{arm}/{seed}'] = {'predictions_sha256': file_hash(path), 'pairs': len(rows)}
            print(f'Formal test {arm} seed{seed} H1={metrics["query_macro"]["ColHit@1"]:.6f}', flush=True)
    write_json(output/'FORMAL_TEST.json', {'selection_lock_sha256': file_hash(output/'MODEL_SELECTION_LOCK.json'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
        'test_for_selection': False, 'population_pairs': len(population)})


def subset(rows: list[dict], name: str) -> list[dict]:
    if name == 'full':
        return rows
    if name == 'non_empty':
        return [r for r in rows if not r['natural_evidence_empty']]
    if name == 'empty':
        return [r for r in rows if r['natural_evidence_empty']]
    return [r for r in rows if len(r['candidate_column_indices']) > int(name[2:])]


def analyze(r1: Path, output: Path) -> None:
    read_json(output/'FORMAL_TEST.json')
    arms = list(read_json(output/'MODEL_SELECTION_LOCK.json')['checkpoints'])
    objects = read_jsonl(r1/'OBJECTS.jsonl.gz')[0]
    reports, bootstrap, flips = {}, {}, []
    for split in ('dev', 'test'):
        items = inputs_for(r1, output, split)
        population = metadata_for(r1, items, objects, split)
        rows_by_arm = {}
        reports[split] = {}
        for arm in ['R1_C2'] + arms:
            rows_by_arm[arm], reports[split][arm] = [], {}
            for seed in SEEDS:
                path = output/f'BASELINE_REPLAY/{seed}/{split}.O-R.jsonl.gz' if arm == 'R1_C2' else output/f'{arm}/predictions/{seed}/{split}.jsonl.gz'
                preds = read_jsonl(path)
                prior = read_jsonl(output/f'PRIOR/predictions/{seed}/{split}.jsonl.gz')
                report, rows = report_metrics(population, preds, prior)
                reports[split][arm][str(seed)] = report
                rows_by_arm[arm].append(rows)
                write_jsonl(output/f'ANALYSIS/rows/{split}.{arm}.{seed}.jsonl.gz', rows)
                if arm.startswith('PVR'):
                    pred_index, prior_index = {pair_key(p): p for p in preds}, {pair_key(p):p for p in prior}
                    for row in rows:
                        if row['corrected'] or row['damaged']:
                            key = pair_key(row)
                            flips.append({**row, 'split': split, 'arm': arm, 'seed': seed,
                                'prior_ranking': prior_index[key]['ranking'], 'final_ranking': pred_index[key]['ranking'],
                                'evidence_ids': pred_index[key]['evidence_ids']})
        comparisons = [('FLAT_MIX', 'R1_C2'), ('FLAT_MIX', 'OO_CONTROL'),
            ('PVR_BUNDLE', 'FLAT_MIX'), ('PVR_SEPARATE', 'FLAT_MIX'),
            ('PVR_SEPARATE', 'PVR_BUNDLE')]
        comparisons.extend((a, 'PRIOR') for a in arms if a.startswith('PVR'))
        if 'PVR_SUPPORT' in arms:
            comparisons.append(('PVR_SUPPORT', 'PVR_SEPARATE'))
        bootstrap[split] = {}
        for left, right in comparisons:
            result = {}
            for name in ('full', 'non_empty', 'empty', 'M>1', 'M>2', 'M>3', 'M>5'):
                aa = [subset(r, name) for r in rows_by_arm[left]]
                bb = [subset(r, name) for r in rows_by_arm[right]]
                result[name] = {field: paired_bootstrap(aa, bb, field) for field in ('ColHit@1', 'MRR')}
                if left.startswith('PVR') and right == 'PRIOR':
                    result[name]['net_correction'] = paired_bootstrap(aa, bb, 'net_correction')
            bootstrap[split][f'{left}-{right}'] = result
    write_json(output/'ANALYSIS/subgroup_metrics.json', reports)
    write_json(output/'ANALYSIS/paired_bootstrap.json', bootstrap)
    write_json(output/'ANALYSIS/flip_analysis.json', {'cases': flips,
        'note': 'Correction/damage conditional rates have different denominators; compare counts and query-macro net gain too.'})

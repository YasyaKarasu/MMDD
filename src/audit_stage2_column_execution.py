#!/usr/bin/env python
"""Independently verify epoch visits, selection, checkpoint bytes, and prediction isolation."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import torch
from mmdd_stage2.column_data import digest, file_hash, read_jsonl, write_json
from mmdd_stage2.column_metrics import pair_key
from mmdd_stage2.column_training import parameter_hash
from mmdd_stage2.checkpoints import load_candidate_scorer
from mmdd_stage2.column_reporting import report


def audit_execution(output: Path) -> dict:
    population = read_jsonl(output/'COLUMN_POPULATION.train.jsonl')
    manifests = {file_hash(p): p for p in (output/'CACHE_MANIFESTS').glob('*.json')}
    checks, initial = [], {}
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            receipt_path = output/'RUN_RECEIPTS'/arm/f'{seed}.json'
            receipt = json.loads(receipt_path.read_text())
            if not receipt['executed'] or receipt['actual_epochs'] != 20:
                raise ValueError('Missing formal 20-epoch execution')
            views = [json.loads(manifests[h].read_text())['records'] for h in receipt['cache_hashes']]
            history = json.loads(Path(receipt['history_path']).read_text())
            steps = 0
            for epoch, trace in enumerate(history, 1):
                records = views[(epoch - 1) % 2]
                keys = [pair_key(r) for r in records]
                if set(keys) != {pair_key(r) for r in population}:
                    raise ValueError('Epoch population differs from frozen labels')
                order = list(range(len(keys)))
                random.Random(seed * 1000 + epoch).shuffle(order)
                if digest([keys[i] for i in order]) != trace['visits_sha256']:
                    raise ValueError('Cannot reproduce recorded epoch visits')
                if trace['base_visits'] != len(population) or trace['unique_base_visits'] != len(population):
                    raise ValueError('Incomplete base population visit')
                steps += sum(any(records[i]['status'] == 'ok' for i in order[j:j+32]) for j in range(0, len(order), 32))
                if trace['optimizer_steps_total'] != steps or not math.isfinite(trace['grad_norm_mean']):
                    raise ValueError('Optimizer-step/gradient trace mismatch')
            selected = max(history, key=lambda r: (r['dev_query_macro']['MRR'], r['dev_query_macro']['ColHit@3'],
                                                   r['dev_query_macro']['ColHit@1'], -r['epoch']))['epoch']
            if selected != receipt['selected_epoch']:
                raise ValueError('Selection does not follow the preregistered dev criterion')
            folder = output/'CHECKPOINTS'/arm/str(seed)
            for name, expected in receipt['checkpoint_hashes'].items():
                if file_hash(folder/name) != expected:
                    raise ValueError('Checkpoint hash changed')
            for name, expected in [('epoch0.pt', receipt['initial_parameter_hash']), ('end.pt', receipt['end_parameter_hash'])]:
                scorer = load_candidate_scorer(folder/name, torch.device('cpu'), expected_reader_layout=receipt['reader_layout_version'])
                if parameter_hash(scorer) != expected:
                    raise ValueError('Tensor parameter hash changed')
            payload = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
            if payload['metadata']['selected_epoch'] != selected:
                raise ValueError('Selected checkpoint has inconsistent epoch metadata')
            if receipt['initial_parameter_hash'] == receipt['end_parameter_hash']:
                raise ValueError('Head did not update')
            initial[(arm, seed)] = receipt['initial_parameter_hash']
            checks.append({'arm': arm, 'seed': seed, 'epochs': 20, 'visits': 20*len(population),
                           'optimizer_steps': steps, 'selected_epoch': selected, 'receipt_sha256': file_hash(receipt_path)})
    if any(initial[('C0', seed)] != initial[('C1', seed)] for seed in (13,29)):
        raise ValueError('C0/C1 linear head initial weights differ')
    forbidden = {'gold_column_indices','gold_column_position','gold_source_column_index','join_attribute','query_kind','chain_id','reason','source_table_id'}
    prediction_files = sorted((output/'PREDICTIONS').glob('*/*/*.jsonl.gz'))
    count = 0
    for path in prediction_files:
        for row in read_jsonl(path):
            # Failure reasons are operational exceptions, not dataset reason labels.
            illegal = forbidden.intersection(row) - ({'reason'} if row['status'] != 'ok' else set())
            if illegal:
                raise ValueError(f'Prediction embeds supervision: {illegal}')
            count += 1
    result = {'executed': True, 'passed': True, 'formal_runs': checks, 'C0_C1_initial_weights_identical': True,
              'prediction_files_checked': len(prediction_files), 'prediction_rows_checked': count,
              'method': 'reconstruct visits from manifest+seed, verify optimizer steps/selection/parameter and file hashes; inspect prediction schema'}
    write_json(output/'EXECUTION_VALIDATION.json', result)
    report(output)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(audit_execution(parser.parse_args().output)))

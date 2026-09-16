"""Train-only position control, paired differences, and fixed-C50 input audit."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any
from .column_data import read_jsonl, write_json, write_jsonl
from .column_cache import VIEW_SEEDS
from .column_metrics import KS, evaluate, pair_key, prediction
from .data import permute_table_columns


def position_baseline(output: Path) -> None:
    objects = read_jsonl(output/'OBJECTS.jsonl.gz')[0]
    training = read_jsonl(output/'COLUMN_POPULATION.train.jsonl')
    counts = Counter()
    query_targets = Counter((r['dataset'], r['query_id']) for r in training)
    lake_queries = Counter(k[0] for k in query_targets)
    for r in training:
        for view in (0, 1):
            columns = [c['column_index'] for c in permute_table_columns(objects['targets'][r['target_id']], seed=VIEW_SEEDS[view])['columns']]
            weight = 1 / (len(lake_queries) * lake_queries[r['dataset']] * query_targets[(r['dataset'], r['query_id'])] * 2)
            for gold in r['gold_column_indices']:
                counts[columns.index(gold)] += weight / len(r['gold_column_indices'])
    write_json(output/'BASELINES/train_position_prior.json', dict(counts))
    for split in ('train', 'dev', 'test'):
        population = read_jsonl(output/f'COLUMN_POPULATION.{split}.jsonl')
        predictions = []
        for r in population:
            columns = [c['column_index'] for c in permute_table_columns(objects['targets'][r['target_id']], seed=VIEW_SEEDS[0])['columns']]
            predictions.append(prediction({**r, 'candidate_column_indices': columns}, [counts[i] for i in range(len(columns))],
                                          model_hash='train_only_majority_display_position', input_hash='view0'))
        metrics, _ = evaluate(population, predictions)
        write_json(output/'BASELINES'/f'{split}.position.json', metrics)
        write_jsonl(output/'BASELINES'/f'{split}.position.predictions.jsonl.gz', predictions)


def c50_compatibility(output: Path) -> dict[str, Any]:
    path = output/'FROZEN_C50.jsonl.gz'
    if not path.is_file():
        return {'executed': False, 'reason': 'No frozen C50 export'}
    candidates = {r['query_id']: r for r in read_jsonl(path)}
    population = read_jsonl(output/'COLUMN_POPULATION.dev.jsonl')
    by_query = defaultdict(list)
    admitted = set()
    for r in population:
        entry = candidates.get(r['query_id'])
        if entry is None:
            by_query[r['query_id']].append(0.)
            continue
        ids = [x['target_id'] for x in entry['results']]
        if len(ids) != len(set(ids)) or len(ids) > 50:
            raise ValueError('Invalid frozen C50 candidate identities')
        present = r['target_id'] in ids
        by_query[r['query_id']].append(float(present))
        if present:
            admitted.add(pair_key(r))
    report = {'implemented': True, 'executed': True, 'scope': 'canonical-dev implicit input compatibility only',
              'queries': len(by_query), 'pairs': len(population),
              'R_admit': mean(mean(v) for v in by_query.values()),
              'missing_query_retrieval': len(set(by_query) - set(candidates)),
              'admitted_pairs': len(admitted), 'column_joint_availability': {}}
    for arm in ('C0', 'C1', 'C2'):
        for seed in (13, 29):
            p = output/'PREDICTIONS'/arm/str(seed)/'dev.O-R.view0.jsonl.gz'
            if not p.is_file():
                report['column_joint_availability'][f'{arm}/{seed}'] = None
                continue
            _, rows = evaluate(population, read_jsonl(p))
            values = defaultdict(list)
            for r in rows:
                values[r['query_id']].append({str(k): r[f'ColHit@{k}'] if pair_key(r) in admitted else 0. for k in KS})
            report['column_joint_availability'][f'{arm}/{seed}'] = {
                str(k): mean(mean(v[str(k)] for v in vs) for vs in values.values()) for k in KS}
    report['interpretation'] = 'Table admission and correct-table-plus-column availability; neither is value recovery or semantic join success'
    write_json(output/'C50_INPUT_COMPATIBILITY.json', report)
    return report


def paired_differences(output: Path) -> None:
    differences, intervals = [], []
    for split in ('dev', 'test'):
        population = read_jsonl(output/f'COLUMN_POPULATION.{split}.jsonl')
        for condition in ('O-O', 'O-R'):
            for seed in (13, 29):
                for high, low in [('C1', 'C0'), ('C2', 'C1')]:
                    paths = [output/'PREDICTIONS'/a/str(seed)/f'{split}.{condition}.view0.jsonl.gz' for a in (high, low)]
                    if not all(p.is_file() for p in paths):
                        continue
                    rows = [evaluate(population, read_jsonl(p))[1] for p in paths]
                    groups = defaultdict(list)
                    for a, b in zip(*rows, strict=True):
                        if pair_key(a) != pair_key(b):
                            raise ValueError('Paired predictions have mismatched populations')
                        groups[(a['dataset'], a['query_id'], a['source_table_id'])].append({
                            m: a[m] - b[m] for m in ['MRR'] + [f'ColHit@{k}' for k in KS]})
                    local = []
                    for (lake, q, source), records in groups.items():
                        local.append({'contrast': f'{high}-{low}', 'seed': seed, 'split': split, 'condition': condition,
                                      'dataset': lake, 'query_id': q, 'source_table_id': source,
                                      'delta': {m: mean(r[m] for r in records) for m in records[0]}})
                    differences.extend(local)
                    clustered = defaultdict(list)
                    for r in local:
                        clustered[(r['dataset'], r['source_table_id'])].append(r['delta'])
                    group_values = list(clustered.values())
                    generator = random.Random(13)
                    boot = defaultdict(list)
                    for _ in range(1000):
                        sample = [v for _ in group_values for v in generator.choice(group_values)]
                        for metric in sample[0]:
                            boot[metric].append(mean(v[metric] for v in sample))
                    intervals.append({'contrast': f'{high}-{low}', 'seed': seed, 'split': split, 'condition': condition,
                                      'source_groups': len(group_values), 'bootstrap_replicates': 1000,
                                      'CI95': {m: [sorted(v)[25], sorted(v)[974]] for m, v in boot.items()}})
    write_jsonl(output/'PER_QUERY_DIFFERENCES.jsonl', differences)
    write_json(output/'PAIRED_BOOTSTRAP.json', {'evaluated': bool(differences), 'comparisons': intervals})

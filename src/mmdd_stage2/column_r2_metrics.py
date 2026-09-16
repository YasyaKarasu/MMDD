"""R2 query-macro metrics, correction/damage and query-first paired bootstrap."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean

import numpy as np

from .column_metrics import evaluate, pair_key


def query_mean(rows: list[dict], field: str) -> float | None:
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['query_id']].append(row[field])
    return mean(mean(v) for v in groups.values()) if groups else None


def report_metrics(population: list[dict], predictions: list[dict], prior: list[dict] | None = None) -> tuple[dict, list[dict]]:
    report, rows = evaluate(population, predictions)
    if prior is not None:
        indexed = {pair_key(p): p for p in prior}
        for row in rows:
            p = indexed[pair_key(row)]
            gold = set(row['gold_column_indices'])
            admitted = bool(gold.intersection(p['ranking'][:3]))
            correct = p['ranking'][0] in gold
            row.update(prior_admission=float(admitted), prior_correct=float(correct),
                       correction_eligible=float(not correct and admitted),
                       corrected=float(not correct and admitted and row['ColHit@1'] == 1),
                       damaged=float(correct and row['ColHit@1'] == 0))
            prior_rank = next(i for i, c in enumerate(p['ranking'], 1) if c in gold)
            row.update(net_correction=row['corrected'] - row['damaged'],
                       EvidenceGain_H1=row['ColHit@1'] - float(correct),
                       EvidenceGain_MRR=row['MRR'] - 1/prior_rank)

    def summarize(subset: list[dict]) -> dict:
        if not subset:
            return {'pairs': 0, 'query_macro': None}
        result = {'pairs': len(subset), 'queries': len({(r['dataset'], r['query_id']) for r in subset}),
                  'query_macro': {f: query_mean(subset, f) for f in ['MRR'] + [f'ColHit@{k}' for k in (1, 2, 3, 5)]},
                  'pair_micro': {f: mean(r[f] for r in subset) for f in ['MRR'] + [f'ColHit@{k}' for k in (1, 2, 3, 5)]}}
        if prior is not None:
            result.update(PriorAdmissionAt3=query_mean(subset, 'prior_admission'),
                corrected_cases=sum(r['corrected'] for r in subset), damaged_cases=sum(r['damaged'] for r in subset),
                correction_eligible_cases=sum(r['correction_eligible'] for r in subset),
                damage_eligible_cases=sum(r['prior_correct'] for r in subset),
                correction_rate=query_mean([r for r in subset if r['correction_eligible']], 'corrected'),
                damage_rate=query_mean([r for r in subset if r['prior_correct']], 'damaged'),
                net_correction_count=sum(r['net_correction'] for r in subset),
                net_correction_query_macro=query_mean(subset, 'net_correction'),
                EvidenceGain_H1=query_mean(subset, 'EvidenceGain_H1'),
                EvidenceGain_MRR=query_mean(subset, 'EvidenceGain_MRR'))
            eligible = [r for r in subset if r['correction_eligible']]
            correct = [r for r in subset if r['prior_correct']]
            result['pair_micro'].update(
                PriorAdmissionAt3=mean(r['prior_admission'] for r in subset),
                correction_rate=mean(r['corrected'] for r in eligible) if eligible else None,
                damage_rate=mean(r['damaged'] for r in correct) if correct else None,
                net_correction=mean(r['net_correction'] for r in subset))
        return result

    report['subsets'] = {'full': summarize(rows),
        'non_empty': summarize([r for r in rows if not r['natural_evidence_empty']]),
        'empty': summarize([r for r in rows if r['natural_evidence_empty']]),
        **{f'M>{k}': summarize([r for r in rows if len(r['candidate_column_indices']) > k]) for k in (1, 2, 3, 5)}}
    for field in ('modality', 'evidence_count', 'column_count_bucket'):
        report['by_' + field] = {str(value): summarize([r for r in rows if r.get(field) == value])
            for value in sorted({r.get(field, 'unknown') for r in rows}, key=str)}
    return report, rows


def paired_bootstrap(left: list[list[dict]], right: list[list[dict]], field: str,
                     replicates: int = 10000) -> dict:
    if len(left) != 2 or len(right) != 2:
        raise ValueError('Exactly two paired seeds required')
    query_differences = defaultdict(list)
    sources = {}
    for a_rows, b_rows in zip(left, right, strict=True):
        b = {pair_key(r): r for r in b_rows}
        if {pair_key(r) for r in a_rows} != set(b):
            raise ValueError('Bootstrap populations differ')
        per_query = defaultdict(list)
        for a in a_rows:
            q = (a['dataset'], a['query_id'])
            source = (a['dataset'], a['source_table_id'])
            if q in sources and sources[q] != source:
                raise ValueError('Query spans source groups')
            sources[q] = source
            per_query[q].append(a[field] - b[pair_key(a)][field])
        for q, values in per_query.items():
            query_differences[q].append(mean(values))
    if not query_differences:
        return {'queries': 0, 'difference': None, 'ci95': None}
    clusters = defaultdict(list)
    for q, values in query_differences.items():
        if len(values) != 2:
            raise ValueError('Query missing a seed')
        clusters[sources[q]].append(mean(values))
    groups = [clusters[k] for k in sorted(clusters)]
    totals = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(20260916)
    sampled = []
    for _ in range(replicates):
        draw = rng.integers(0, len(groups), size=len(groups))
        sampled.append(float(totals[draw].sum()/counts[draw].sum()))
    return {'difference': float(totals.sum()/counts.sum()), 'ci95': np.quantile(sampled, [.025, .975]).tolist(),
            'replicates': replicates, 'queries': int(counts.sum()), 'source_groups': len(groups),
            'order': 'within-query targets mean, same-query two-seed mean, paired source-cluster bootstrap'}

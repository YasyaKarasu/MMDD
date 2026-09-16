"""Column-only rankings; labels enter only the population evaluator."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Sequence

KS = (1, 2, 3, 5)


def pair_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (record['dataset'], record['query_id'], record['target_id'])


def column_order(logits: Sequence[float], columns: Sequence[int]) -> list[int]:
    """Break ties by canonical ID, independently of label and display order."""
    if len(logits) != len(columns) or len(set(columns)) != len(columns):
        raise ValueError('Logits/column identity mismatch')
    if not all(math.isfinite(x) for x in logits):
        raise ValueError('Non-finite column scores')
    return sorted(range(len(columns)), key=lambda i: (-logits[i], columns[i]))


def prediction(record: dict[str, Any], logits: Sequence[float], **identity: Any) -> dict[str, Any]:
    columns = list(record['candidate_column_indices'])
    order = column_order(logits, columns)
    scores = [math.exp(x - max(logits)) for x in logits]
    probabilities = [x / sum(scores) for x in scores]
    ranked = [columns[i] for i in order]
    return {**{k: record[k] for k in ('dataset', 'query_id', 'target_id')},
            'candidate_column_indices': columns, 'logits': list(logits),
            'ranking': ranked, 'top_k': {str(k): ranked[:k] for k in KS},
            'column_hypotheses': [{'column_index': columns[i], 'column_rank': rank,
                                   'logit': logits[i], 'probability': probabilities[i]}
                                  for rank, i in enumerate(order, 1)],
            'status': 'ok', **identity}


def count_bucket(count: int) -> str:
    return str(count) if count <= 3 else '4-5' if count <= 5 else '6-10' if count <= 10 else '>10'


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {'pairs': 0, 'queries': 0, 'query_macro': None, 'pair_micro': None}
    fields = ['MRR', 'success_rate'] + [f'{prefix}@{k}' for k in KS
              for prefix in ('ColHit', 'Random', 'column_budget', 'candidate_fraction')]
    queries = defaultdict(list)
    for row in rows:
        queries[(row['dataset'], row['query_id'])].append(row)
    return {'pairs': len(rows), 'queries': len(queries),
            'query_macro': {f: mean(mean(r[f] for r in group) for group in queries.values()) for f in fields},
            'pair_micro': {f: mean(r[f] for r in rows) for f in fields}}


def evaluate(population: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Missing or failed outputs remain in the immutable population denominator."""
    indexed = {pair_key(p): p for p in predictions}
    if len(indexed) != len(predictions) or len({pair_key(p) for p in population}) != len(population):
        raise ValueError('Duplicate population or prediction key')
    if set(indexed) - {pair_key(p) for p in population}:
        raise ValueError('Prediction outside frozen population')
    rows = []
    for item in population:
        columns = item['candidate_column_indices']
        gold = set(item['gold_column_indices'])
        if not gold or not gold <= set(columns):
            raise ValueError('Population labels do not map to canonical columns')
        p = indexed.get(pair_key(item), {'status': 'missing_output'})
        success = p['status'] == 'ok'
        ranking = []
        if success:
            if set(p['candidate_column_indices']) != set(columns):
                raise ValueError('Prediction omitted candidates')
            ranking = [p['candidate_column_indices'][i] for i in column_order(p['logits'], p['candidate_column_indices'])]
            if p['ranking'] != ranking:
                raise ValueError('Serialized rank disagrees with scores/tie policy')
        rank = next((i for i, c in enumerate(ranking, 1) if c in gold), None)
        m = len(columns)
        row = {**item, 'status': p['status'], 'rank': rank,
               'MRR': 1 / rank if rank else 0., 'success_rate': float(success),
               'column_count_bucket': count_bucket(m)}
        for k in KS:
            n = min(k, m)
            row[f'ColHit@{k}'] = float(bool(gold.intersection(ranking[:k])))
            row[f'Random@{k}'] = 1 - math.comb(m - len(gold), n) / math.comb(m, n)
            row[f'column_budget@{k}'] = n
            row[f'candidate_fraction@{k}'] = n / m
        rows.append(row)
    metrics = aggregate(rows)
    for field in ('dataset', 'modality', 'column_count_bucket', 'support_status', 'natural_evidence_empty'):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, 'unknown'))].append(row)
        metrics['by_' + field] = {k: aggregate(v) for k, v in sorted(groups.items())}
    metrics['non_saturated'] = {str(k): aggregate([r for r in rows if len(r['candidate_column_indices']) > k]) for k in KS}
    metrics['failure_reasons'] = {s: sum(r['status'] == s for r in rows) for s in sorted({r['status'] for r in rows}) if s != 'ok'}
    metrics['tie_break'] = 'descending logit, ascending canonical column ID'
    return metrics, rows

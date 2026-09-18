"""S2-R4 Phase J metrics: table recall, table-column recall, pair budget, wrong-table confidence.

Primary endpoint is query-macro Target Recall@10 over the single frozen candidate pool,
with R@20 and R@50 alongside. The denominator is every annotated joinable target of the
query, not "hit at least one table". Every difference is a paired estimate over the same
queries, pool, labels and budget, with a source-group cluster bootstrap.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .r4_common import digest, read_jsonl, write_json
from .r4_metrics import cluster_bootstrap, paired_cluster_bootstrap, recall_at_k
from .r4_labels import load_labels

ARM_ORDER = ('J0_TableOnly', 'J1_ExactProduct_Prior', 'J2_ExactProduct_FlatMix')


def _entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _load_rankings(out: Path, scope: str) -> dict:
    tables = defaultdict(dict)
    for row in read_jsonl(out / 'TABLE_RANKINGS' / f'table_rankings.{scope}.jsonl.gz'):
        # TABLE_MARGINAL_CHECK carries no ordering - it is the sum_c J = p_T identity check.
        # Loading it here would overwrite the J0 table-only ranking it is filed under.
        if row['ranking_mode'] not in ('TABLE_RANK_MAX', 'TABLE_RANK_ONLY'):
            continue
        key = row['arm'] if row['seed'] is None else f"{row['arm']}|seed{row['seed']}"
        tables[key][row['query_id']] = row
    pairs = defaultdict(dict)
    pairs_path = out / 'PAIR_RANKINGS' / f'pair_rankings.{scope}.jsonl.gz'
    if pairs_path.is_file():
        for row in read_jsonl(pairs_path):
            pairs[f"{row['arm']}|seed{row['seed']}"][row['query_id']] = row
    return {'tables': tables, 'pairs': pairs}


def _table_order(entry: dict) -> list[str]:
    return [item['target_id'] for item in entry['ranking']]


def _pair_order(entry: dict) -> list[tuple[str, int]]:
    return [(item['target_id'], item['column_id']) for item in entry['ranking']]


def target_recall_rows(rankings: dict, labels: dict[str, set[str]], group_of: dict[str, str],
                       *, metric_prefix: str = 'target') -> list[dict]:
    rows = []
    for key, by_query in rankings.items():
        for query_id, entry in by_query.items():
            gold = labels.get(query_id, set())
            if not gold:
                continue
            order = _table_order(entry)
            row = {'dataset': 'entitables', 'query_id': query_id,
                   'source_group': group_of.get(query_id, query_id), 'arm': key,
                   'gold_targets': len(gold),
                   'pool_size': len(order),
                   'candidate_recall@50': recall_at_k(order, gold, 50)}
            for k in (10, 20, 50):
                row[f'{metric_prefix}_recall@{k}'] = recall_at_k(order, gold, k)
            rows.append(row)
    return rows


def joint_table_column_rows(rankings: dict, pairs: dict, labels_cols: dict, group_of: dict,
                            *, top_tables: int = 10, top_columns: int = 3) -> list[dict]:
    rows = []
    for key, by_query in pairs.items():
        for query_id, entry in by_query.items():
            gold = labels_cols.get(query_id, {})
            if not gold:
                continue
            gold_pairs = {(t, c) for t, cols in gold.items() for c in cols}
            order = _pair_order(entry)
            row = {'dataset': 'entitables', 'query_id': query_id,
                   'source_group': group_of.get(query_id, query_id), 'arm': key,
                   'gold_pairs': len(gold_pairs)}
            for budget in (5, 10, 20, 50):
                row[f'pair_recall@{budget}'] = recall_at_k(
                    [f'{t}|{c}' for t, c in order], {f'{t}|{c}' for t, c in gold_pairs}, budget)
            table_entry = rankings.get(key, {}).get(query_id)
            if table_entry is None:
                rows.append(row)
                continue
            table_order = _table_order(table_entry)
            position = {t: i for i, t in enumerate(table_order)}
            hits = []
            for target, columns in gold.items():
                if target not in position:
                    hits.append(0.0)
                    continue
                if position[target] >= top_tables:
                    hits.append(0.0)
                    continue
                ranked_columns = [c for t, c in order if t == target]
                hits.append(1.0 if set(ranked_columns[:top_columns]) & columns else 0.0)
            row[f'joint_table_column_recall@{top_tables},{top_columns}'] = (
                sum(hits) / len(hits) if hits else None)
            rows.append(row)
    return rows


def wrong_table_confidence(out: Path, scope: str, labels: dict[str, set[str]],
                           group_of: dict) -> dict:
    """max-rho and column count for annotated-positive versus benchmark-nonpositive tables.

    A benchmark table that is not annotated positive is *unlabeled*, not confirmed non-joinable.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(out / 'COLUMN_LOGITS' / f'column_logits.{scope}.jsonl.gz'):
        query_id = row['query_id']
        gold = labels.get(query_id, set())
        if not gold:
            continue
        probabilities = [math.exp(value) for value in row['column_log_probability']]
        key = f"{row['arm']}|seed{row['seed']}"
        buckets['positive' if row['target_id'] in gold else 'benchmark_nonpositive'].append({
            'arm': key, 'query_id': query_id, 'target_id': row['target_id'],
            'source_group': group_of.get(query_id, query_id),
            'max_rho': max(probabilities) if probabilities else None,
            'columns': len(probabilities),
            'rho_entropy': _entropy(probabilities),
        })
    summary = {}
    for bucket, rows in buckets.items():
        by_arm: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_arm[row['arm']].append(row)
        summary[bucket] = {}
        for arm, arm_rows in by_arm.items():
            values = [r['max_rho'] for r in arm_rows if r['max_rho'] is not None]
            summary[bucket][arm] = {
                'tables': len(arm_rows),
                'mean_columns': sum(r['columns'] for r in arm_rows) / len(arm_rows),
                'mean_max_rho': sum(values) / len(values) if values else None,
                'max_rho_p90': sorted(values)[int(0.9 * (len(values) - 1))] if values else None,
                'mean_rho_entropy': (sum(r['rho_entropy'] for r in arm_rows) / len(arm_rows)),
                'label_status': ('annotated_positive' if bucket == 'positive'
                                 else 'unlabeled_benchmark_nonpositive'),
            }
    return summary


def _seed_aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    """Average seed-suffixed arms back onto the base arm name so arms can be paired."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        base = row['arm'].split('|seed')[0]
        buckets[(base,) + tuple(row[k] for k in keys)].append(row)
    merged = []
    for key, group in buckets.items():
        base = dict(group[0])
        base['arm'] = key[0]
        for name, value in group[0].items():
            if name.startswith(('target_recall@', 'pair_recall@', 'joint_table_column_recall@',
                                'candidate_recall@')) and value is not None:
                values = [g[name] for g in group if isinstance(g.get(name), (int, float))]
                base[name] = sum(values) / len(values) if values else None
        merged.append(base)
    return merged


def pool_ceiling(lock: dict, gold_targets: dict, group_of: dict) -> dict:
    """The best any reranker of this pool can reach, before it is asked to have an opinion.

    Every arm reranks one frozen 50-candidate pool, so the achievable query-macro recall is
    fixed by how much of each query's gold set is in that pool. Reporting it prevents a
    reranker from being credited for, or blamed for, the candidate generator's misses.
    """
    achievable = []
    all_in = none_in = 0
    for query_id, gold in gold_targets.items():
        if query_id not in lock:
            continue
        in_pool = len(gold & set(lock[query_id]['candidate_ids']))
        achievable.append(in_pool / len(gold))
        all_in += in_pool == len(gold)
        none_in += in_pool == 0
    in_scope = {q: g for q, g in gold_targets.items() if q in lock}
    micro_total = sum(len(g) for g in in_scope.values())
    micro_in = sum(len(g & set(lock[q]['candidate_ids'])) for q, g in in_scope.items())
    return {
        'queries_scored': len(achievable),
        'query_macro_ceiling': sum(achievable) / len(achievable) if achievable else None,
        'micro_ceiling': micro_in / micro_total if micro_total else None,
        'queries_with_every_gold_target_in_pool': all_in,
        'queries_with_no_gold_target_in_pool': none_in,
        'interpretation': (
            'this is the Recall@|pool| ceiling; every J arm shares the same pool, so a '
            'difference between arms is reranking and a shortfall from 1.0 is candidacy'
        ),
    }


def build_metrics(out: Path, scope: str, *, iterations: int = 10000) -> dict:
    gold_targets, gold_columns = load_labels(out, scope)
    rankings = _load_rankings(out, scope)
    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    group_of = {q: r['source_group'] for q, r in lock.items()}

    # load_labels returns gold columns keyed by (query, target); the joint metric wants them
    # nested per query.
    nested_columns: dict[str, dict[str, set[int]]] = defaultdict(dict)
    for (query_id, target_id), columns in gold_columns.items():
        nested_columns[query_id][target_id] = columns

    recall_rows = target_recall_rows(rankings['tables'], gold_targets, group_of)
    recall_rows = _seed_aggregate(recall_rows, ('query_id',))
    jtc_rows = joint_table_column_rows(rankings['tables'], rankings['pairs'], nested_columns, group_of)
    jtc_rows = _seed_aggregate(jtc_rows, ('query_id',))

    per_arm = {}
    for arm in sorted({r['arm'] for r in recall_rows}):
        subset = [r for r in recall_rows if r['arm'] == arm]
        entry = {
            'queries_scored': len(subset),
            'queries_with_gold': sum(r['gold_targets'] > 0 for r in subset),
            'mean_gold_targets_per_query': (sum(r['gold_targets'] for r in subset) / len(subset)
                                            if subset else None),
        }
        for metric in ('target_recall@10', 'target_recall@20', 'target_recall@50',
                       'candidate_recall@50'):
            ci = cluster_bootstrap(subset, 'source_group', metric, iterations=iterations)
            entry[metric] = {'point': ci['point'], 'ci95': [ci['low'], ci['high']]}
        jtc_subset = [r for r in jtc_rows if r['arm'] == arm]
        for metric in [k for k in (jtc_subset[0] if jtc_subset else {}) if
                       k.startswith(('joint_table_column_recall@', 'pair_recall@'))]:
            ci = cluster_bootstrap(jtc_subset, 'source_group', metric, iterations=iterations)
            entry[metric] = {'point': ci['point'], 'ci95': [ci['low'], ci['high']]}
        per_arm[arm] = entry

    comparisons = {}
    arms_present = sorted(per_arm)
    for left, right in ((a, b) for i, a in enumerate(arms_present) for b in arms_present[i + 1:]):
        joined = []
        left_rows = {r['query_id']: r for r in recall_rows if r['arm'] == left}
        right_rows = {r['query_id']: r for r in recall_rows if r['arm'] == right}
        for query_id in sorted(set(left_rows) & set(right_rows)):
            joined.append({'dataset': 'entitables', 'query_id': query_id,
                           'source_group': left_rows[query_id]['source_group'],
                           'left': left_rows[query_id]['target_recall@10'],
                           'right': right_rows[query_id]['target_recall@10']})
        if not joined:
            continue
        ci = paired_cluster_bootstrap(joined, 'source_group', 'left', 'right', iterations=iterations)
        comparisons[f'{left}__vs__{right}'] = {
            'metric': 'target_recall@10', 'paired_queries': len(joined),
            'difference_pp': None if ci['point'] is None else 100 * ci['point'],
            'ci95_pp': None if ci['low'] is None else [100 * ci['low'], 100 * ci['high']],
        }

    # C50 is frozen, so every arm must reproduce the same R@50 candidate recall.
    r50 = {arm: per_arm[arm]['candidate_recall@50']['point'] for arm in per_arm}
    finite = [v for v in r50.values() if v is not None]
    invariance = {
        'per_arm': r50,
        'identical': bool(finite) and max(finite) - min(finite) < 1e-12,
        'note': 'a differing R@50 means the candidate pool changed between arms, not a reranker effect',
    }

    result = {
        'scope': scope,
        'primary_endpoint': 'query-macro Target Recall@10',
        'denominator': 'every annotated joinable target of the query',
        'splits': {
            'overall': 'identical to implicit for this dev population',
            'implicit': 'reported',
            'explicit': 'N/A: the canonical missing-attribute contract excludes direct '
                        'visible-column qrels from the dev population',
        },
        'per_arm': per_arm,
        'comparisons_target_recall@10': comparisons,
        'c50_invariance_check': invariance,
        'pool_ceiling': pool_ceiling(lock, gold_targets, group_of),
        'wrong_table_confidence': wrong_table_confidence(out, scope, gold_targets, group_of),
        'rows_sha256': digest([{k: v for k, v in r.items()} for r in recall_rows]),
    }
    write_json(out / f'PHASE_J_METRICS.{scope}.json', result)
    return {
        'per_arm_target_recall@10': {a: per_arm[a]['target_recall@10']['point'] for a in per_arm},
        'per_arm_target_recall@20': {a: per_arm[a]['target_recall@20']['point'] for a in per_arm},
        'per_arm_target_recall@50': {a: per_arm[a]['target_recall@50']['point'] for a in per_arm},
        'comparisons': {k: v['difference_pp'] for k, v in comparisons.items()},
        'c50_invariance': invariance['identical'],
        'pool_ceiling': result['pool_ceiling'],
    }

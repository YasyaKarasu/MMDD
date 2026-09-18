"""S2-R4 Phase J driver: score every candidate target's columns and build J0/J1/J2.

Column features are produced for ALL candidate targets of the frozen C50, including
direct-only tables and tables with empty retained evidence. A target whose table has no
columns is kept in the candidate pool and its table score stays in the table softmax;
it is recorded as COLUMN_SCORING_ERROR and contributes no pair, which is reported as
partial coverage rather than silently imputed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .column_r2_cache import job_key
from .column_r2_training import load_head
from .r4_common import R4, read_jsonl, write_json, write_jsonl
from .r4_features import load_feature_matrix
from .r4_joint import log_joint, table_rank_max, table_rank_only, top_k_pairs

R2 = Path('/home/oycy/MMDD/work/S2_COL_R2')
ARMS = {
    'J1_ExactProduct_Prior': ('PRIOR', (13, 29)),
    'J2_ExactProduct_FlatMix': ('FLAT_MIX', (13, 29)),
}
HEAD_PATHS = {
    ('PRIOR', 13): R2 / 'PRIOR/checkpoints/13/selected.pt',
    ('PRIOR', 29): R2 / 'PRIOR/checkpoints/29/selected.pt',
    ('FLAT_MIX', 13): R2 / 'FLAT_MIX/checkpoints/13/selected.pt',
    ('FLAT_MIX', 29): R2 / 'FLAT_MIX/checkpoints/29/selected.pt',
}


def _state(entry: dict[str, Any], cache: dict[str, dict]) -> torch.Tensor:
    path = entry['path']
    if path not in cache:
        cache[path] = torch.load(path, map_location='cpu', weights_only=True)
    record = cache[path]
    if record['input_hash'] != entry['input_hash']:
        raise ValueError(f'feature identity mismatch for {path}')
    return torch.cat([record['open_states'], record['close_states']], dim=-1)


def _job_key(dataset: str, query_id: str, target_id: str, evidence_ids) -> str:
    return job_key({'dataset': dataset, 'query_id': query_id, 'target_id': target_id,
                    'evidence_ids': list(evidence_ids), 'view': 0})


ARM_LABEL = {'PRIOR': 'J1_ExactProduct_Prior', 'FLAT_MIX': 'J2_ExactProduct_FlatMix'}


def arm_name_of(arm: str) -> str:
    return ARM_LABEL[arm]


def _heads() -> dict:
    heads = {}
    for key, path in HEAD_PATHS.items():
        model, meta = load_head(path)
        heads[key] = (model, meta)
    return heads


def build_joint(out: Path, scope: str) -> dict:
    # These are tiny tensor ops on cached reader states; a large thread team costs more
    # than the arithmetic it parallelises.
    torch.set_num_threads(4)
    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    entries = load_feature_matrix(out, scope)
    heads = _heads()
    state_cache: dict[str, dict] = {}

    tables_rows: list[dict] = []
    pairs_rows: list[dict] = []
    column_logit_rows: list[dict] = []
    native_rows: list[dict] = []
    excluded_from_column_softmax: Counter = Counter()
    missing_feature_arms: Counter = Counter()
    coverage = {
        'queries': len(lock),
        'candidate_pairs': 0,
        'pairs_with_columns': 0,
        'pairs_without_columns': 0,
        'column_cells': 0,
        'column_cells_scored': 0,
        'pairs_missing_features': 0,
        'scoring_errors': [],
        'arms': {},
    }

    for query_id in sorted(lock):
        record = lock[query_id]
        targets = record['targets']
        table_logits = []
        for target in targets:
            value = target['native_table_logit']
            if value is None:
                raise ValueError(f'{query_id}/{target["target_id"]}: BLOCKED_MISSING_NATIVE_SCORE')
            table_logits.append(float(value))
        table_tensor = torch.tensor(table_logits, dtype=torch.float64)
        target_ids = [t['target_id'] for t in targets]
        original_ranks = [t['original_rank'] for t in targets]

        native_rows.append({
            'query_id': query_id, 'dataset': record['dataset'], 'source_group': record['source_group'],
            'candidate_pool_id': record['candidate_pool_id'],
            'targets': [
                {'target_id': t['target_id'], 'original_rank': t['original_rank'],
                 'native_table_logit': t['native_table_logit'], 'score_space': t['score_space'],
                 'score_status': t['score_status'],
                 'table_score_source_hash': t['table_score_source_hash'],
                 'retained_evidence_ids': t['retained_evidence_ids'],
                 'evidence_reason': t['evidence_reason'], 'source_label': t['source_label']}
                for t in targets
            ],
        })

        width = max((len(t['column_ids']) for t in targets), default=0)
        column_ids = [t['column_ids'] for t in targets]
        mask = torch.zeros((len(targets), width), dtype=torch.bool)
        for index, ids in enumerate(column_ids):
            mask[index, :len(ids)] = True

        condition_logits: dict[tuple[str, int], torch.Tensor] = {}
        for (arm, seed), (model, meta) in heads.items():
            condition = 'PRIOR' if arm == 'PRIOR' else 'FLAT_MIX'
            # One batched head call per query per arm. Calling the head per target instead
            # lets torch spin up a thread team for a 50x8192 matmul thousands of times, which
            # costs far more than the arithmetic.
            batch, positions = [], []
            failed: dict[int, str] = {}
            for index, target in enumerate(targets):
                ids = target['column_ids']
                if not ids:
                    failed[index] = 'candidate table exposes no columns'
                    continue
                evidence_ids = [] if condition == 'PRIOR' else target['retained_evidence_ids']
                key = _job_key(record['dataset'], query_id, target['target_id'], evidence_ids)
                entry = entries.get(key)
                if entry is None or not entry.get('path'):
                    # Absent, or present-but-failed (an unreadable evidence image). Either way
                    # the pair has no genuine column distribution.
                    failed[index] = ('reader feature missing' if entry is None
                                     else f"reader feature failed: {entry.get('status')}")
                    continue
                batch.append(_state(entry, state_cache))
                positions.append((index, entry['candidate_column_indices'], ids))
            matrix = torch.zeros((len(targets), width), dtype=torch.float64)
            if batch:
                stacked = torch.cat(batch, dim=0)
                half = stacked.shape[-1] // 2
                with torch.no_grad():
                    logits = model(stacked[:, :half], stacked[:, half:])
                offset = 0
                for index, available, ids in positions:
                    count = len(ids)
                    order = [available.index(c) for c in ids]
                    matrix[index, :count] = logits[offset:offset + count].double()[order]
                    offset += count
            for index, detail in sorted(failed.items()):
                target = targets[index]
                coverage['scoring_errors'].append({
                    'query_id': query_id, 'target_id': target['target_id'],
                    'reason': 'COLUMN_SCORING_ERROR', 'detail': detail, 'arm': arm, 'seed': seed,
                    'retained_candidate': True, 'original_rank': target['original_rank'],
                })
                if detail.startswith('reader feature'):
                    coverage['pairs_missing_features'] += 1
            # A target with no genuine column distribution must leave the column softmax
            # entirely. Leaving its zero row in would hand it a uniform rho and multiply the
            # table score by a fabricated column probability - the exact imputation the
            # contract forbids. Its table score is untouched, so the table ranking is intact.
            arm_mask = mask.clone()
            for index in failed:
                arm_mask[index, :] = False
            condition_logits[(arm, seed)] = (matrix, arm_mask)
            for detail in failed.values():
                reason = 'no_columns' if detail.startswith('candidate table') else 'feature_failed'
                excluded_from_column_softmax[f'{arm_name_of(arm)}|seed{seed}|{reason}'] += 1
                if reason == 'feature_failed':
                    missing_feature_arms[f'{arm_name_of(arm)}|seed{seed}'] += 1

        for (arm, seed), (matrix, arm_mask) in sorted(condition_logits.items()):
            joint = log_joint(table_tensor, matrix, arm_mask, target_ids=target_ids,
                              column_ids=column_ids, original_ranks=original_ranks)
            arm_name = 'J1_ExactProduct_Prior' if arm == 'PRIOR' else 'J2_ExactProduct_FlatMix'
            order = joint['pair_order']
            pairs_rows.append({
                'query_id': query_id, 'dataset': record['dataset'], 'source_group': record['source_group'],
                'arm': arm_name, 'seed': seed, 'ranking_mode': 'PAIR_RANK',
                'pair_count': len(order),
                'ranking': [{'target_id': t, 'column_id': c,
                             'log_joint': float(joint['pair_log_probability'][target_ids.index(t), column_ids[target_ids.index(t)].index(c)])}
                            for t, c in order],
            })
            dedup = table_rank_max(joint, target_ids)
            tables_rows.append({
                'query_id': query_id, 'dataset': record['dataset'], 'source_group': record['source_group'],
                'arm': arm_name, 'seed': seed, 'ranking_mode': 'TABLE_RANK_MAX',
                'ranking': [
                    {'target_id': t,
                     'max_pair_log_probability': float(joint['best_pair_log_probability'][target_ids.index(t)]),
                     'table_probability': float(joint['table_probability'][target_ids.index(t)])}
                    for t in dedup
                ],
                'tables_without_columns': joint['tables_without_columns'],
            })
            if arm == 'PRIOR' and seed == 13:
                tables_rows.append({
                    'query_id': query_id, 'dataset': record['dataset'],
                    'source_group': record['source_group'], 'arm': 'J0_TableOnly', 'seed': None,
                    'ranking_mode': 'TABLE_RANK_ONLY',
                    'ranking': [
                        {'target_id': t,
                         'table_probability': float(joint['table_probability'][target_ids.index(t)])}
                        for t in table_rank_only(joint, target_ids, original_ranks)
                    ],
                })
                tables_rows.append({
                    'query_id': query_id, 'dataset': record['dataset'],
                    'source_group': record['source_group'], 'arm': 'J0_TableOnly', 'seed': None,
                    'ranking_mode': 'TABLE_MARGINAL_CHECK',
                    'max_marginal_deviation': float(
                        (joint['marginal'][arm_mask.any(dim=-1)]
                         - joint['table_probability'][arm_mask.any(dim=-1)]).abs().max())
                    if bool(arm_mask.any()) else 0.0,
                    'ranking': [],
                })
            for index, target in enumerate(targets):
                if not arm_mask[index].any():
                    # No genuine column distribution for this arm: emit no row at all rather
                    # than a zero vector that a reader could mistake for uniform logits.
                    continue
                column_logit_rows.append({
                    'query_id': query_id, 'target_id': target['target_id'], 'arm': arm_name,
                    'seed': seed, 'original_rank': target['original_rank'],
                    'evidence_ids': target['retained_evidence_ids'],
                    'column_ids': target['column_ids'],
                    'logits': matrix[index, :len(target['column_ids'])].tolist(),
                    'column_log_probability': joint['column_log_probability'][index, :len(target['column_ids'])].tolist(),
                })

        # Reader states are keyed per (query, target), so nothing is reused across queries -
        # only the second column-model seed re-reads the same file within this loop. Holding
        # them past the query would cost ~11 GB on the full dev pool for no benefit.
        state_cache.clear()

        coverage['candidate_pairs'] += len(targets)
        coverage['pairs_with_columns'] += sum(bool(t['column_ids']) for t in targets)
        coverage['pairs_without_columns'] += sum(not t['column_ids'] for t in targets)
        coverage['column_cells'] += sum(len(t['column_ids']) for t in targets)

    folders = {name: out / name for name in (
        'NATIVE_TABLE_SCORES', 'COLUMN_LOGITS', 'PAIR_RANKINGS', 'TABLE_RANKINGS')}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folders['NATIVE_TABLE_SCORES'] / f'native_table_scores.{scope}.jsonl.gz', native_rows)
    write_jsonl(folders['COLUMN_LOGITS'] / f'column_logits.{scope}.jsonl.gz', column_logit_rows)
    write_jsonl(folders['PAIR_RANKINGS'] / f'pair_rankings.{scope}.jsonl.gz', pairs_rows)
    write_jsonl(folders['TABLE_RANKINGS'] / f'table_rankings.{scope}.jsonl.gz', tables_rows)

    coverage['arms'] = {
        f'{arm}|seed{seed}': sum(1 for r in column_logit_rows
                                 if r['arm'] == ('J1_ExactProduct_Prior' if arm == 'PRIOR' else 'J2_ExactProduct_FlatMix')
                                 and r['seed'] == seed)
        for arm, seed in (('PRIOR', 13), ('PRIOR', 29), ('FLAT_MIX', 13), ('FLAT_MIX', 29))
    }
    coverage['column_cells_scored'] = len(column_logit_rows)
    coverage['scoring_error_count'] = len(coverage['scoring_errors'])
    expected_rows = 4 * coverage['pairs_with_columns']
    coverage['column_cells_expected'] = expected_rows
    coverage['column_cells_coverage_fraction'] = (
        len(column_logit_rows) / expected_rows if expected_rows else 1.0)
    coverage['pairs_excluded_from_column_softmax'] = dict(sorted(
        excluded_from_column_softmax.items()))
    coverage['pairs_missing_feature_by_arm'] = dict(sorted(missing_feature_arms.items()))
    coverage['scoring_errors'] = coverage['scoring_errors'][:50]
    coverage['full_finite_coverage'] = (
        coverage['pairs_missing_features'] == 0
        and coverage['column_cells_scored'] == 4 * coverage['pairs_with_columns']
    )
    write_json(out / f'PHASE_J_COVERAGE.{scope}.json', coverage)
    return {k: v for k, v in coverage.items() if k != 'scoring_errors'} | {
        'head_metadata': {f'{a}|{s}': m for (a, s), (_, m) in heads.items()}}

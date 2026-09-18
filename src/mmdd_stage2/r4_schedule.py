"""S2-R4 Phase I scheduling: which (target, column) branches actually get generated.

The scheduler reads predicted logits only. It never reads known_witness, gold column
admission, ROW_GT or any target row value, so the same schedule is produced with every
GT sidecar deleted.

S0 TableFirst : the top ten distinct tables by the table-only ranking, each contributing
                its Prior Top1 column - at most ten branches, so at most ten distinct tables.
S1 JointGlobal: the top ten pairs of the global J1 pair ranking, capped at three columns
                per table. It may give several branches to one table; how many distinct
                tables it covers is reported rather than assumed.

Both share the same Prior column scores, so the only difference is how the pair budget is
allocated.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .r4_common import digest, read_jsonl, write_json

N_TARGET_CANDIDATES = 50
K_COLUMN_PER_TABLE = 3
B_PAIR_RECOVERY = 10
B_EVIDENCE_READ = 4
MAX_QUERY_ROWS_LEGACY = 5


def visible_query_rows(objects: dict, query_id: str) -> list[dict]:
    """Every visible row of the query table, with column ids. Never filtered by GT."""
    table = objects['queries'][query_id]
    names = {int(c['column_index']): c['column_name'] for c in table['columns']}
    rows = []
    for index, row in enumerate(table['rows']):
        rows.append({
            'query_row_id': index,
            'cells': [{'column_id': int(c['column_index']),
                       'column_name': names[int(c['column_index'])],
                       'text': c.get('text', '')}
                      for c in row['cells']],
        })
    return rows


def _prior_top1(order: list[tuple[str, int, float]], target_id: str) -> int | None:
    """The Prior arm's highest-scoring column for this table, from predicted logits only."""
    best = None
    for candidate_target, column_id, log_joint in order:
        if candidate_target != target_id:
            continue
        if best is None or log_joint > best[1]:
            best = (column_id, log_joint)
    return best[0] if best else None


def build_schedules(out: Path, scope: str) -> dict:
    tables = read_jsonl(out / 'TABLE_RANKINGS' / f'table_rankings.{scope}.jsonl.gz')
    pairs = read_jsonl(out / 'PAIR_RANKINGS' / f'pair_rankings.{scope}.jsonl.gz')

    table_only: dict[str, list[str]] = {}
    pair_rank: dict[str, list[tuple[str, int, float]]] = {}
    source_group: dict[str, str] = {}
    for row in tables:
        if row['ranking_mode'] == 'TABLE_RANK_ONLY':
            table_only[row['query_id']] = [i['target_id'] for i in row['ranking']]
    for row in pairs:
        if row['arm'] == 'J1_ExactProduct_Prior' and row['seed'] == 13:
            pair_rank[row['query_id']] = [
                (i['target_id'], i['column_id'], i['log_joint']) for i in row['ranking']]

    import gzip
    import json

    with gzip.open(out / f'READER_OBJECTS.{scope}.jsonl.gz', 'rt', encoding='utf-8') as handle:
        objects = json.loads(handle.readline())
    for record in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz'):
        source_group[record['query_id']] = record['source_group']

    schedules = {'S0_TableFirst': [], 'S1_JointGlobal': []}
    for query_id in sorted(table_only):
        rows = visible_query_rows(objects, query_id)
        order = pair_rank[query_id]

        chosen_s0 = []
        for target_id in table_only[query_id][:B_PAIR_RECOVERY]:
            column_id = _prior_top1(order, target_id)
            if column_id is not None:
                chosen_s0.append((target_id, column_id))

        chosen_s1 = []
        per_table: Counter = Counter()
        for target_id, column_id, _ in order:
            if per_table[target_id] >= K_COLUMN_PER_TABLE:
                continue
            chosen_s1.append((target_id, column_id))
            per_table[target_id] += 1
            if len(chosen_s1) >= B_PAIR_RECOVERY:
                break

        for name, chosen in (('S0_TableFirst', chosen_s0), ('S1_JointGlobal', chosen_s1)):
            schedules[name].append({
                'query_id': query_id,
                'source_group': source_group.get(query_id),
                'budget': {
                    'n_target_candidates': N_TARGET_CANDIDATES,
                    'k_column_per_table': K_COLUMN_PER_TABLE,
                    'b_pair_recovery': B_PAIR_RECOVERY,
                    'b_evidence_read': B_EVIDENCE_READ,
                    'max_query_rows': None,
                    'max_query_rows_note': 'all visible query rows are scheduled; rows are not '
                                           'trimmed by GT presence, so no cap is applied',
                },
                'branches': [{'target_id': t, 'column_id': c} for t, c in chosen],
                'branch_count': len(chosen),
                'distinct_tables': len({t for t, _ in chosen}),
                'visible_query_rows': len(rows),
                'scheduled_units': len(chosen) * len(rows),
                'schedule_sha256': digest(chosen),
            })

    summaries = {}
    for name, records in schedules.items():
        summaries[name] = {
            'queries': len(records),
            'branches': sum(r['branch_count'] for r in records),
            'distinct_tables_total': sum(r['distinct_tables'] for r in records),
            'scheduled_units': sum(r['scheduled_units'] for r in records),
            'visible_query_rows_total': sum(r['visible_query_rows'] for r in records),
            'branch_count_per_query': dict(Counter(r['branch_count'] for r in records)),
            'distinct_tables_per_query': dict(Counter(r['distinct_tables'] for r in records)),
            'mean_visible_query_rows': (sum(r['visible_query_rows'] for r in records)
                                        / len(records) if records else None),
        }
    folder = out / 'RECOVERY_SCHEDULES'
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in schedules.items():
        write_json(folder / f'{name}.{scope}.json', {'scope': scope, 'scheduler': name,
                                                     'records': records})
    write_json(folder / f'SCHEDULE_SUMMARY.{scope}.json', {
        'scope': scope,
        'schedulers': summaries,
        'invariants': {
            'reads_gold': False,
            'reads_witness_flags': False,
            'reads_target_row_values': False,
            'reads_row_gt': False,
        },
        'note': 'S1 may spend several branches on one table; pair budget is not table budget, '
                'which is why distinct_tables_per_query is reported separately',
    })
    return summaries

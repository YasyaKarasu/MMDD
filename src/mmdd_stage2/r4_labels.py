"""S2-R4 offline evaluation labels.

This module is the only place gold information is read. The runtime path
(candidate lock, reader features, joint ranking, scheduling) never imports it, which is
what the acceptance test 'drop every GT sidecar and randomise gold labels' checks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .r4_common import SPLIT, digest, iterate_artifact, read_jsonl, write_json

R3_ARTIFACTS = Path('/home/oycy/MMDD/work/S2_COL_R3_ROW/ARTIFACTS')

IMPLICIT_REASON = 'model_recoverable_join_column'
EXPLICIT_REASON = 'explicit_visible_join_column'


def local_column_index(table: dict[str, Any], source_column_index: int) -> int:
    """Map a source column position onto the projected target's local column index."""
    for column in table['columns']:
        if int(column.get('source_column_index', column['column_index'])) == int(source_column_index):
            return int(column['column_index'])
    raise KeyError(f'source column {source_column_index} not projected into {table["table_id"]}')


def build_labels(out: Path, scope: str, *, split: str = SPLIT) -> dict:
    lake = {r['table_id']: r for r in iterate_artifact('data_lake_tables')}
    gold_targets: dict[str, set[str]] = defaultdict(set)
    gold_columns: dict[tuple[str, str], set[int]] = defaultdict(set)
    reasons: dict[str, set[str]] = defaultdict(set)
    skipped: list[dict] = []
    for record in iterate_artifact('qrels'):
        if record.get('split') != split:
            continue
        query_id, target_id = record['query_table_id'], record['target_table_id']
        reasons[query_id].add(record['reason'])
        if record['reason'] != IMPLICIT_REASON:
            continue
        table = lake.get(target_id)
        if table is None or not table.get('columns'):
            skipped.append({'query_id': query_id, 'target_id': target_id,
                            'reason': 'gold target is not a queryable projection'})
            continue
        gold_targets[query_id].add(target_id)
        gold_columns[(query_id, target_id)].add(
            local_column_index(table, record['join_attribute']['source_column_index'])
        )

    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    in_pool = 0
    out_of_pool = 0
    for query_id in lock:
        for target_id in gold_targets.get(query_id, ()):
            if target_id in set(lock[query_id]['candidate_ids']):
                in_pool += 1
            else:
                out_of_pool += 1

    label_coverage = {
        'split': split,
        'scope': scope,
        'queries_in_scope': len(lock),
        'queries_with_gold': sum(q in gold_targets for q in lock),
        'gold_pairs_total': sum(len(v) for v in gold_targets.values()),
        'gold_pairs_in_candidate_pool': in_pool,
        'gold_pairs_outside_candidate_pool': out_of_pool,
        'queries_with_no_gold_in_split': sum(q not in gold_targets for q in lock),
        'queries_with_explicit_qrels': sum(EXPLICIT_REASON in v for v in reasons.values()),
        'queries_with_implicit_qrels': sum(IMPLICIT_REASON in v for v in reasons.values()),
        'query_kind_note': (
            'dev population is the implicit missing-attribute task; explicit visible-column '
            'qrels exist in the dataset but are excluded by the canonical contract, so the '
            'overall and implicit splits are identical here and explicit is N/A'
        ),
        'gold_targets_not_queryable': skipped,
        'labels_sha256': digest({'targets': {k: sorted(v) for k, v in gold_targets.items()},
                                 'columns': {f'{k[0]}|{k[1]}': sorted(v) for k, v in gold_columns.items()}}),
    }
    write_json(out / f'LABEL_COVERAGE.{scope}.json', label_coverage)
    with (out / f'EVALUATION_LABELS.{scope}.jsonl.gz').open('wb') as handle:
        import gzip

        with gzip.GzipFile(fileobj=handle, mode='wb') as zipped:
            zipped.write(json.dumps({
                'gold_targets': {k: sorted(v) for k, v in gold_targets.items()},
                'gold_columns': {f'{k[0]}|{k[1]}': sorted(v) for k, v in gold_columns.items()},
            }, ensure_ascii=False, allow_nan=False).encode())
    return label_coverage


def load_labels(out: Path, scope: str) -> tuple[dict[str, set[str]], dict[tuple[str, str], set[int]]]:
    import gzip

    with gzip.open(out / f'EVALUATION_LABELS.{scope}.jsonl.gz', 'rt', encoding='utf-8') as handle:
        payload = json.loads(handle.readline())
    targets = {k: set(v) for k, v in payload['gold_targets'].items()}
    columns = {}
    for key, value in payload['gold_columns'].items():
        query_id, target_id = key.split('|', 1)
        columns[(query_id, target_id)] = set(value)
    return targets, columns


def row_labels() -> list[dict]:
    """R3 row-level GT, used only for the R-component row metrics and never for scheduling."""
    return read_jsonl(R3_ARTIFACTS / 'ROW_GT.dev.jsonl')

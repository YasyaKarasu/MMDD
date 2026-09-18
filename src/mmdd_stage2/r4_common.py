"""S2-R4 shared inputs: query-led candidate locks and full-lake reader objects.

R4 deliberately does not reuse OBJECTS.jsonl.gz as the candidate source. That file
was built from qrels, so it contains only gold targets; a column experiment driven
from it can never score the other 49 candidates. Everything here is derived from the
split manifest plus the frozen candidate export, never from ROW_GT or qrels.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

DATA_ROOT = Path(
    '/home/oycy/MMDD/output_mm_joinability_entitables_20000_retry100_rounds5_qwen35_final_survivor_context_gaussian_v9'
)
R1 = Path('/home/oycy/MMDD/work/S2-COL-R1')
R26 = Path('/home/oycy/MMDD/work/stage1_optimization_r26_20260914')
T0_DEV_CACHE = R26 / 'teacher' / 'T0_pairs.sqlite'
T0_TEST_CACHE = R1 / 'natural_test_T0_pairs.sqlite'
R4 = Path('/home/oycy/MMDD/work/S2_R4')
LAKE_ARTIFACT = 'data_lake_tables'
QUERY_ARTIFACT = 'query_tables'
EVIDENCE_ARTIFACT = 'bridge_assets'

RETRIEVER_ID = 'B13-frozen'
TEACHER_ID = 'T0-frozen'
CANDIDATE_CAP = 50
SPLIT = 'dev'

FROZEN_C50 = {SPLIT: R1 / 'FROZEN_C50.jsonl.gz', 'test': R1 / 'FROZEN_C50_TEST.jsonl.gz'}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    temporary = path.with_name(path.name + '.tmp')
    with opener(temporary, 'wt', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_group(dataset: str, source_table_id: str) -> str:
    return hashlib.sha256(f'{dataset}|{source_table_id}'.encode()).hexdigest()


def visible_table(table: dict[str, Any]) -> dict[str, Any]:
    """Identical allowlist to column_data.visible_table: headers plus first 12 rows."""
    from mmdd_dataset.utils import get_cell

    columns = [
        {'column_index': int(c['column_index']), 'column_name': c.get('column_name', '')}
        for c in table.get('columns') or []
    ]
    return {
        'table_id': table['table_id'],
        'columns': columns,
        'rows': [
            {
                'cells': [
                    {
                        'column_index': c['column_index'],
                        'text': get_cell(r, c['column_index']).get('text', ''),
                    }
                    for c in columns
                ]
            }
            for r in (table.get('rows') or [])[:12]
        ],
    }


def iterate_artifact(artifact: str, root: Path = DATA_ROOT) -> Iterator[dict]:
    import sys

    sys.path.insert(0, '/home/oycy/MMDD/src')
    from mmdd_dataset.wdc_runtime import iter_dataset_artifact

    yield from iter_dataset_artifact(root, artifact)


def load_t0_scores(cache_path: Path) -> dict[str, dict[str, float]]:
    """Native T0 QT scalars keyed by query then target. Never a rank."""
    connection = sqlite3.connect(f'file:{cache_path}?mode=ro', uri=True)
    scores: dict[str, dict[str, float]] = {}
    for q, t, score in connection.execute('select q, t, score from scores'):
        scores.setdefault(q, {})[t] = float(score)
    connection.close()
    return scores


def natural_evidence(result: dict[str, Any]) -> list[str]:
    """Frozen read4 retention: content-dedup order preserved, capped at four."""
    return list(
        dict.fromkeys(
            str(p['evidence_id'])
            for p in result.get('paths', [])
            if p.get('kind') == 'evidence' and p.get('evidence_id')
        )
    )[:4]


def load_c50(split: str = SPLIT) -> dict[str, dict]:
    records = {}
    for record in read_jsonl(FROZEN_C50[split]):
        if record['query_id'] in records:
            raise ValueError(f'duplicate query in frozen C50: {record["query_id"]}')
        records[record['query_id']] = record
    return records


def lake_targets() -> dict[str, dict]:
    """Every queryable target projection in the shared lake, keyed by target_id."""
    return {r['table_id']: r for r in iterate_artifact(LAKE_ARTIFACT)}


def query_tables(wanted: set[str]) -> dict[str, dict]:
    result = {}
    for record in iterate_artifact(QUERY_ARTIFACT):
        if record['table_id'] in wanted:
            result[record['table_id']] = record
    return result


def evidence_assets(wanted: set[str]) -> dict[str, dict]:
    result = {}
    for record in iterate_artifact(EVIDENCE_ARTIFACT):
        if record['asset_id'] in wanted:
            result[record['asset_id']] = {
                k: record[k]
                for k in ('asset_id', 'asset_type', 'local_path')
                if k in record
            }
            result[record['asset_id']].setdefault('content', record.get('content', ''))
    return result


def build_candidate_lock(split: str = SPLIT, *, queries: set[str] | None = None,
                         lake: dict[str, dict] | None = None,
                         query_rows: dict[str, dict] | None = None,
                         evidence: dict[str, dict] | None = None,
                         scores: dict[str, dict[str, float]] | None = None) -> dict[str, dict]:
    """Query-led candidate lock: every frozen candidate, empty evidence included."""
    frozen = load_c50(split)
    if queries is not None:
        frozen = {q: r for q, r in frozen.items() if q in queries}
    if scores is None:
        scores = load_t0_scores(T0_DEV_CACHE if split == SPLIT else T0_TEST_CACHE)
    lake = lake if lake is not None else lake_targets()
    if query_rows is None:
        query_rows = {}
        for record in iterate_artifact(QUERY_ARTIFACT):
            if record['table_id'] in frozen:
                query_rows[record['table_id']] = record
    missing_queries = set(frozen) - set(query_rows)
    if missing_queries:
        raise ValueError(f'{len(missing_queries)} frozen queries absent from the query artifact')
    for query_id, record in query_rows.items():
        if record.get('split') != split:
            raise ValueError(f'{query_id}: query artifact split {record.get("split")!r} != {split!r}')

    if evidence is None:
        evidence_needed: set[str] = set()
        for record in frozen.values():
            for result in record['results']:
                evidence_needed.update(natural_evidence(result))
        evidence = evidence_assets(evidence_needed)
    cache_path = T0_DEV_CACHE if split == SPLIT else T0_TEST_CACHE
    cache_hash = file_hash(cache_path)  # 700 MB: hash once, never inside the target loop

    lock: dict[str, dict] = {}
    for query_id, record in sorted(frozen.items()):
        targets = []
        for result in sorted(record['results'], key=lambda x: x['rank']):
            target_id = result['target_id']
            table = lake.get(target_id)
            retained = natural_evidence(result)
            native = scores.get(query_id, {}).get(target_id)
            if table is None:
                raise ValueError(f'candidate {target_id} absent from the shared lake')
            targets.append(
                {
                    'target_id': target_id,
                    'original_rank': int(result['rank']),
                    'native_table_logit': native,
                    'score_space': 'T0_QT_logit',
                    'table_score_source_hash': cache_hash,
                    'score_status': 'ok' if native is not None else 'BLOCKED_MISSING_NATIVE_SCORE',
                    'source_label': (
                        'NO_RETAINED_PATH' if not retained else
                        ('image' if all(evidence[e]['asset_type'] == 'image' for e in retained) else
                         ('text' if all(evidence[e]['asset_type'] == 'text' for e in retained) else 'mixed'))
                    ),
                    'retained_evidence_ids': retained,
                    'evidence_reason': 'NO_RETAINED_PATH' if not retained else 'FROZEN_READ4',
                    'queryable': bool(table.get('columns')),
                    'column_ids': [int(c['column_index']) for c in table.get('columns', [])],
                    'target_content_hash': digest(visible_table(table)),
                }
            )
        lock[query_id] = {
            'query_id': query_id,
            'split': split,
            'dataset': 'entitables',
            'source_group': source_group('entitables', query_rows[query_id]['source_table_id']),
            'retriever_id': RETRIEVER_ID,
            'teacher_id': TEACHER_ID,
            'candidate_pool_id': record['candidate_pool_id'],
            'candidate_path_hash': record['candidate_path_hash'],
            'candidate_cap': CANDIDATE_CAP,
            'actual_candidate_count': len(targets),
            'candidate_ids': [t['target_id'] for t in targets],
            'targets': targets,
        }
    return lock


def build_reader_objects(split: str = SPLIT, queries: set[str] | None = None) -> dict[str, dict]:
    """Full-lake reader objects: every candidate target, not only gold ones."""
    lake = lake_targets()
    frozen = load_c50(split)
    if queries is not None:
        frozen = {q: r for q, r in frozen.items() if q in queries}
    query_rows = {r['table_id']: r for r in iterate_artifact(QUERY_ARTIFACT) if r['table_id'] in frozen}
    lock = build_candidate_lock(split, queries=queries, lake=lake, query_rows=query_rows)
    targets = {
        t['target_id']
        for r in lock.values()
        for t in r['targets']
        if t['column_ids']
    }
    # Unqueryable raw lake tables carry no columns and no reader prompt; they stay in the
    # candidate lock (and the table softmax) but are recorded as COLUMN_SCORING_ERROR.
    evidence_ids = {e for r in lock.values() for t in r['targets'] for e in t['retained_evidence_ids']}

    evidence = evidence_assets(evidence_ids)
    missing = evidence_ids - set(evidence)
    if missing:
        raise ValueError(f'{len(missing)} candidate evidence assets absent from the dataset')

    return {
        'queries': {q: visible_table(t) for q, t in query_rows.items()},
        'targets': {t: visible_table(lake[t]) for t in targets},
        'evidence': evidence,
    }


def population_summary(lock: dict[str, dict]) -> dict[str, Any]:
    pairs = [t for r in lock.values() for t in r['targets']]
    return {
        'queries': len(lock),
        'candidate_pairs': len(pairs),
        'distinct_targets': len({t['target_id'] for t in pairs}),
        'empty_retained_evidence': sum(not t['retained_evidence_ids'] for t in pairs),
        'queryable_pairs': sum(t['queryable'] for t in pairs),
        'unqueryable_pairs': sum(not t['queryable'] for t in pairs),
        'pairs_missing_native_score': sum(t['native_table_logit'] is None for t in pairs),
        'column_cells': sum(len(t['column_ids']) for t in pairs),
        'source_groups': len({r['source_group'] for r in lock.values()}),
        'candidate_count_distribution': dict(Counter(r['actual_candidate_count'] for r in lock.values())),
    }

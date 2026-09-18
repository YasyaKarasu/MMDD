"""S2-R4 population locks: candidate lock, population lock, source lock."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .r4_common import (
    CANDIDATE_CAP,
    DATA_ROOT,
    FROZEN_C50,
    R1,
    R4,
    RETRIEVER_ID,
    SPLIT,
    T0_DEV_CACHE,
    T0_TEST_CACHE,
    TEACHER_ID,
    build_candidate_lock,
    build_reader_objects,
    digest,
    evidence_assets,
    file_hash,
    iterate_artifact,
    lake_targets,
    population_summary,
    natural_evidence,
    read_jsonl,
    source_group,
    visible_table,
    write_json,
    write_jsonl,
)

PILOT_GROUPS = 100
R3_ARTIFACTS = Path('/home/oycy/MMDD/work/S2_COL_R3_ROW/ARTIFACTS')
B13_PARENT = Path('/home/oycy/MMDD/work/stage1_optimization_r26_20260914/recovered/B13/step_000178.pt')
T0_PARENT = Path(
    '/home/oycy/MMDD/work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt'
)
READER_MODEL = Path('/home/oycy/MMDD/hf_models/Qwen3.5-9B')
EMBEDDING_MODEL = Path('/home/oycy/MMDD/hf_models/Qwen3-VL-Embedding-8B')
GENERATOR_MODEL = Path('/home/oycy/MMDD/hf_models/Qwen3.5-9B')


def pilot_source_groups() -> set[str]:
    """The 100 frozen R3 pilot source groups, read from the frozen pilot inventory."""
    inventory = read_jsonl(R3_ARTIFACTS / 'PILOT_ROW_INVENTORY.dev.jsonl')
    groups = sorted({r['source_group'] for r in inventory})
    if len(groups) != PILOT_GROUPS:
        raise ValueError(f'expected {PILOT_GROUPS} pilot source groups, found {len(groups)}')
    return set(groups)


def query_source_groups() -> dict[str, str]:
    """source_group for every dev query, from the split manifest's query tables."""
    result = {}
    for record in iterate_artifact('query_tables'):
        if record.get('split') == SPLIT:
            result[record['table_id']] = source_group('entitables', record['source_table_id'])
    return result


def source_lock() -> dict:
    import torch

    return {
        'round': 'S2-R4',
        'generated_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'data_root': str(DATA_ROOT),
        'reader_model': str(READER_MODEL),
        'generator_model': str(GENERATOR_MODEL),
        'embedding_model': str(EMBEDDING_MODEL),
        'parents': {
            'B13': {'path': str(B13_PARENT), 'sha256': file_hash(B13_PARENT)},
            'T0': {'path': str(T0_PARENT), 'sha256': file_hash(T0_PARENT)},
        },
        'frozen_exports': {k: {'path': str(v), 'sha256': file_hash(v)} for k, v in FROZEN_C50.items()},
        'native_score_caches': {
            'dev': {'path': str(T0_DEV_CACHE), 'sha256': file_hash(T0_DEV_CACHE)},
            'test': {'path': str(T0_TEST_CACHE), 'sha256': file_hash(T0_TEST_CACHE)},
        },
        'identities': {
            'retriever_id': RETRIEVER_ID,
            'teacher_id': TEACHER_ID,
            'candidate_cap': CANDIDATE_CAP,
        },
        'runtime': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'torch': torch.__version__,
            'transformers': importlib.metadata.version('transformers'),
            'cuda': torch.version.cuda,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        'r3_execution_status_hash_mismatch': {
            'file': 'work/S2_COL_R3_ROW/EXECUTION_STATUS.json',
            'declared_bytes': 3149,
            'actual_bytes': (Path('/home/oycy/MMDD/work/S2_COL_R3_ROW/EXECUTION_STATUS.json')).stat().st_size
            if Path('/home/oycy/MMDD/work/S2_COL_R3_ROW/EXECUTION_STATUS.json').is_file() else None,
            'historical_raw_modified': False,
            'note': 'Recorded, not repaired in place. The historical raw delivery is untouched.',
        },
    }


def build_population(scope: str, out: Path, *, wanted_override: set[str] | None = None) -> dict:
    """scope: 'pilot' (100 R3 source groups) or 'dev' (every dev query).

    Each dataset artifact is read exactly once: iter_dataset_artifact re-verifies every
    shard hash on each call, so repeated iteration over the 22k-table lake is the
    dominant cost here.
    """
    frozen = {r['query_id']: r for r in read_jsonl(FROZEN_C50[SPLIT])}

    dev_groups: dict[str, str] = {}
    all_query_rows: dict[str, dict] = {}
    for record in iterate_artifact('query_tables'):
        if record.get('split') != SPLIT:
            continue
        dev_groups[record['table_id']] = source_group('entitables', record['source_table_id'])
        all_query_rows[record['table_id']] = record

    if wanted_override is not None:
        wanted = set(wanted_override)
    elif scope == 'pilot':
        selected = pilot_source_groups()
        wanted = {q for q, g in dev_groups.items() if g in selected and q in frozen}
        if not wanted:
            raise ValueError('pilot selection is empty')
    elif scope == 'dev':
        wanted = set(frozen)
    else:
        raise ValueError(f'unknown scope {scope!r}')
    missing = wanted - set(all_query_rows)
    if missing:
        raise ValueError(f'{len(missing)} selected queries absent from the query artifact')
    if not wanted <= set(frozen):
        raise ValueError('selected queries are not all in the frozen candidate export')

    lake: dict[str, dict] = {}
    needed_targets = {t['target_id'] for q in wanted for t in frozen[q]['results']}
    for record in iterate_artifact('data_lake_tables'):
        if record['table_id'] in needed_targets:
            lake[record['table_id']] = record
    absent = needed_targets - set(lake)
    if absent:
        raise ValueError(f'{len(absent)} candidate targets absent from the shared lake')

    query_rows = {q: all_query_rows[q] for q in wanted}
    evidence_ids = {
        e
        for q in wanted
        for result in frozen[q]['results']
        for e in natural_evidence(result)
    }
    evidence = evidence_assets(evidence_ids)
    missing_evidence = evidence_ids - set(evidence)
    if missing_evidence:
        raise ValueError(f'{len(missing_evidence)} candidate evidence assets absent from the dataset')
    lock = build_candidate_lock(SPLIT, queries=wanted, lake=lake, query_rows=query_rows,
                                evidence=evidence)

    objects = {
        'queries': {q: visible_table(t) for q, t in query_rows.items()},
        'targets': {
            t['target_id']: visible_table(lake[t['target_id']])
            for r in lock.values() for t in r['targets'] if t['column_ids']
        },
        'evidence': evidence,
    }

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz', [lock[q] for q in sorted(lock)])
    with __import__('gzip').open(out / f'READER_OBJECTS.{scope}.jsonl.gz', 'wt', encoding='utf-8') as handle:
        handle.write(json.dumps(objects, ensure_ascii=False, allow_nan=False) + '\n')
    summary = population_summary(lock)
    write_json(out / f'POPULATION_LOCK.{scope}.json', {
        'scope': scope,
        'split': SPLIT,
        'selection_rule': (
            f'every dev query whose source_group is one of the {PILOT_GROUPS} frozen R3 pilot groups'
            if scope == 'pilot' else 'every query in the frozen dev candidate export'
        ),
        'built_from': 'query split manifest + frozen candidate export + native T0 cache',
        'never_reads': ['ROW_GT', 'qrels', 'gold values', 'witness flags', 'target_row_ids'],
        'summary': summary,
        'candidate_lock_sha256': file_hash(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz'),
        'reader_objects_sha256': file_hash(out / f'READER_OBJECTS.{scope}.jsonl.gz'),
        'query_ids_sha256': digest(sorted(lock)),
        'query_ids': sorted(lock),
    })
    return summary

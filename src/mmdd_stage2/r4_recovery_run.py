"""S2-R4 Phase R runner: expand and execute the recovery queues.

Two queues, reported separately and never mixed:

* ``R-component`` - the frozen R3 pilot's correct targets with the same rows and evidence,
  kept for mechanism analysis against the historical aligned metrics. New inference still
  runs on every visible query row; the rows without recovery GT are simply not scored.
* ``R-deploy``    - the (target, column) branches the Phase I scheduler actually selected,
  run over every visible query row.

The three arms share one semantic-input cache: identical (query row, column, evidence set)
inputs are generated once and read back by the other arms. An arm may never consume a
branch that only another arm's extra budget produced.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .r4_common import digest, file_hash, read_jsonl, write_json, write_jsonl
from .r4_recovery import MAX_NEW_TOKENS, SourceAwareRecoveryBackend, build_prompt
from .r4_schedule import visible_query_rows
from .r4_source_trace import classify_value, summarize

GENERATOR_MODEL = Path('/home/oycy/MMDD/hf_models/Qwen3.5-9B')
ARMS = ('R0', 'R1', 'R2')
R3_ARTIFACTS = Path('/home/oycy/MMDD/work/S2_COL_R3_ROW/ARTIFACTS')


def _objects(out: Path, scope: str) -> dict:
    with gzip.open(out / f'READER_OBJECTS.{scope}.jsonl.gz', 'rt', encoding='utf-8') as handle:
        return json.loads(handle.readline())


def evidence_for(record_objects: dict, target: dict) -> list[dict[str, Any]]:
    result = []
    for asset_id in target['retained_evidence_ids']:
        item = record_objects['evidence'][asset_id]
        result.append({'asset_id': asset_id, 'asset_type': item['asset_type'],
                       'content': item.get('content', ''), 'local_path': item.get('local_path')})
    return result


def component_queue(out: Path, scope: str) -> list[dict]:
    """Correct-target queue from the frozen R3 pilot, expanded over all visible query rows."""
    objects = _objects(out, scope)
    plans = json.loads((R3_ARTIFACTS / 'COLUMN_CANDIDATES.dev.json').read_text())
    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    units = []
    for key, plan in plans.items():
        query_id, target_id = key.split('|', 1)
        if query_id not in lock:
            continue
        target = next((t for t in lock[query_id]['targets'] if t['target_id'] == target_id), None)
        if target is None or not target['column_ids']:
            continue
        rows = visible_query_rows(objects, query_id)
        for column_id in plan['gold_local_column_indices']:
            if column_id not in target['column_ids']:
                continue
            for row in rows:
                units.append({
                    'queue': 'R-component',
                    'dataset': 'entitables', 'query_id': query_id, 'target_id': target_id,
                    'source_group': lock[query_id]['source_group'],
                    'query_row_id': row['query_row_id'],
                    'column_id': column_id,
                    'column_name': plan['column_names'][str(column_id)],
                    'cells': row['cells'],
                    'evidence_ids': target['retained_evidence_ids'],
                    'evidence_source': 'FROZEN_READ4',
                    'is_gold_column': True,
                })
    return units


# Frozen before any generation: which source groups enter the small R-deploy pilot. The
# rule reads only the split manifest's source_group, never GT, column labels or row labels.
DEPLOY_PILOT_SALT = 'R4_DEPLOY_PILOT_V1'


def select_deploy_queries(out: Path, scope: str, groups: int | None) -> list[str]:
    """The first `groups` source groups by SHA256(salt|source_group), then all their queries."""
    lock = read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')
    by_group: dict[str, list[str]] = {}
    for record in lock:
        by_group.setdefault(record['source_group'], []).append(record['query_id'])
    ordered = sorted(by_group, key=lambda g: digest(f'{DEPLOY_PILOT_SALT}|{g}'))
    if groups is None or groups >= len(ordered):
        chosen = ordered
    else:
        chosen = ordered[:groups]
    selected = sorted(q for g in chosen for q in by_group[g])
    receipt = {
        'salt': DEPLOY_PILOT_SALT,
        'rule': f'ascending SHA256(salt|source_group); first {groups} groups, then all their queries',
        'groups_selected': len(chosen),
        'groups_available': len(ordered),
        'queries_selected': len(selected),
        'query_ids': selected,
        'reads_gold': False,
        'frozen_before_generation': True,
    }
    path = out / 'RECOVERY_SCHEDULES' / f'DEPLOY_PILOT_SELECTION.{scope}.json'
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous['query_ids'] != selected:
            raise ValueError('deploy pilot selection changed; it must be frozen before generation')
    else:
        write_json(path, receipt)
    return selected


def deploy_queue(out: Path, scope: str, *, groups: int | None = None) -> list[dict]:
    """Scheduler-selected branches over every visible query row; no GT involvement."""
    objects = _objects(out, scope)
    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    schedule = json.loads((out / 'RECOVERY_SCHEDULES' / f'S0_TableFirst.{scope}.json').read_text())
    allowed = set(select_deploy_queries(out, scope, groups)) if groups is not None else None
    units = []
    for record in schedule['records']:
        query_id = record['query_id']
        if allowed is not None and query_id not in allowed:
            continue
        rows = visible_query_rows(objects, query_id)
        by_id = {t['target_id']: t for t in lock[query_id]['targets']}
        for branch in record['branches']:
            target = by_id[branch['target_id']]
            for row in rows:
                units.append({
                    'queue': 'R-deploy',
                    'dataset': 'entitables', 'query_id': query_id,
                    'target_id': branch['target_id'],
                    'source_group': record['source_group'],
                    'query_row_id': row['query_row_id'],
                    'column_id': branch['column_id'],
                    'column_name': next(c['column_name'] for c in objects['targets'][branch['target_id']]['columns']
                                        if c['column_index'] == branch['column_id']),
                    'cells': row['cells'],
                    'evidence_ids': target['retained_evidence_ids'],
                    'evidence_source': 'FROZEN_READ4',
                    'is_gold_column': None,
                })
    return units


def unit_id(unit: dict, arm: str) -> str:
    return digest({'arm': arm, 'queue': unit['queue'], 'query_id': unit['query_id'],
                   'target_id': unit['target_id'], 'query_row_id': unit['query_row_id'],
                   'column_id': unit['column_id'], 'cells': unit['cells'],
                   'evidence_ids': unit['evidence_ids'],
                   'column_name': unit['column_name']})


def semantic_key(unit: dict) -> str:
    """Shared semantic input: the same content is generated once across arms."""
    return digest({'cells': unit['cells'], 'column_id': unit['column_id'],
                   'column_name': unit['column_name'], 'evidence_ids': unit['evidence_ids']})


def run(out: Path, scope: str, *, queue: str = 'R-deploy', device: str = 'cuda:0',
        shard: int = 0, shards: int = 1, limit: int | None = None,
        groups: int | None = None) -> dict:
    torch.set_num_threads(4)
    units = (component_queue(out, scope) if queue == 'R-component'
             else deploy_queue(out, scope, groups=groups))
    if limit:
        units = units[:limit]
    units = [u for u in units if int(unit_id(u, 'R0'), 16) % shards == shard]

    objects = _objects(out, scope)
    folder = out / 'RAW_GENERATIONS'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / f'{queue}.{scope}.shard{shard}.jsonl.gz'
    done = set()
    if result_path.is_file():
        done = {r['unit_id'] for r in read_jsonl(result_path)}

    backend = None
    written = skipped = 0
    started = time.monotonic()
    handle = gzip.open(result_path, 'at', encoding='utf-8')
    for position, unit in enumerate(units, 1):
        if unit_id(unit, 'R0') in done:
            skipped += 1
            continue
        evidence = evidence_for(objects, unit)
        for arm in ARMS:
            identity = unit_id(unit, arm)
            if identity in done:
                continue
            if arm != 'R0' and not evidence:
                pass  # R1/R2 may still answer QUERY_VISIBLE with no evidence at all
            if arm == 'R0' and not evidence:
                record = {**unit, 'unit_id': identity, 'arm': arm, 'model_called': False,
                          'status': 'INSUFFICIENT_EVIDENCE', 'values': [],
                          'reason': 'empty_evidence_r0_refusal', 'raw_completion': '',
                          'evidence_count': 0, 'finish_reason': 'not_run'}
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
                continue
            if backend is None:
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable: run with authorized GPU access')
                backend = SourceAwareRecoveryBackend(
                    GENERATOR_MODEL, device=device, reader_image_max_pixels=262144,
                    max_new_tokens=MAX_NEW_TOKENS)
            output = backend.recover(arm, unit['cells'], unit['column_name'], evidence)
            traced = [classify_value(v, output['evidence_labels'],
                                     _evidence_text(objects, unit), unit['cells'])
                      for v in output['values']]
            record = {**{k: unit[k] for k in ('queue', 'dataset', 'query_id', 'target_id',
                                              'source_group', 'query_row_id', 'column_id',
                                              'column_name', 'evidence_ids', 'evidence_source',
                                              'is_gold_column')},
                      'unit_id': identity, 'semantic_key': semantic_key(unit),
                      'model_called': True, 'traced_values': traced,
                      'source_summary': summarize(traced), **output}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            written += 1
        if position % 10 == 0:
            rate = (time.monotonic() - started) / position
            print(f'{queue} {device} shard{shard}: {position}/{len(units)} {rate:.2f}s/unit '
                  f'eta {(len(units) - position) * rate / 60:.1f}min', flush=True)
    handle.close()
    return {'queue': queue, 'units': len(units), 'generated': written, 'resumed': skipped,
            'path': str(result_path),
            'elapsed_seconds': time.monotonic() - started}


def _evidence_text(objects: dict, unit: dict) -> dict[str, str]:
    return {asset_id: objects['evidence'][asset_id].get('content', '')
            for asset_id in unit['evidence_ids']
            if objects['evidence'][asset_id]['asset_type'] == 'text'}

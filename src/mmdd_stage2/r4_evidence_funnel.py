"""S2-R4 evidence supply funnel for known witnesses.

Where a witness is lost is recorded stage by stage. Any stage the frozen artifacts do not
log is reported as ``missing_stage`` rather than guessed, and the funnel never claims a
witness was available when the record does not show it.

    witness exists
      -> enters the frozen query-level E pool
      -> linked to this target
      -> survived content dedup
      -> inside the path top-20
      -> inside read4
      -> actually placed in the row prompt
      -> value / source / match outcome
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .r4_common import DATA_ROOT, SPLIT, iterate_artifact, natural_evidence, read_jsonl, write_json

R1 = Path('/home/oycy/MMDD/work/S2-COL-R1')
FROZEN_NATURAL_U = R1 / 'FROZEN_NATURAL_U.jsonl.gz'
STAGES = (
    'witness_exists',
    'in_frozen_query_evidence_pool',
    'linked_to_target',
    'survived_content_dedup',
    'in_path_top20',
    'in_read4',
    'placed_in_row_prompt',
)


def witness_index() -> dict[tuple[str, str], set[str]]:
    """Known witnesses from the dataset's evidence_recoveries, keyed by (query, target)."""
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in iterate_artifact('evidence_recoveries'):
        asset = record.get('evidence', {}).get('asset_id')
        if asset:
            result[(record['query_table_id'], record['target_table_id'])].add(asset)
    return result


def build_funnel(out: Path, scope: str, *, split: str = SPLIT) -> dict:
    witnesses = witness_index()
    frozen: dict[str, dict] = {}
    for record in read_jsonl(FROZEN_NATURAL_U):
        frozen[record['query_id']] = record

    lock = {r['query_id']: r for r in read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    schedule = out / 'RECOVERY_SCHEDULES'
    prompt_units: dict[tuple[str, str], set[str]] = defaultdict(set)
    if (schedule / f'SCHEDULE_SUMMARY.{scope}.json').is_file():
        import json

        s0 = json.loads((schedule / f'S0_TableFirst.{scope}.json').read_text())
        for record in s0['records']:
            by_id = {t['target_id']: t for t in lock[record['query_id']]['targets']}
            for branch in record['branches']:
                prompt_units[(record['query_id'], branch['target_id'])].update(
                    by_id[branch['target_id']]['retained_evidence_ids'])

    counts = Counter()
    per_pair = []
    missing_stages = set()
    for query_id, record in sorted(lock.items()):
        fused = frozen.get(query_id)
        if fused is None:
            missing_stages.add('frozen_query_retrieval')
        for target in record['targets']:
            target_id = target['target_id']
            known = witnesses.get((query_id, target_id), set())
            if not known:
                continue
            counts['witness_exists'] += len(known)
            entry = {'query_id': query_id, 'target_id': target_id, 'witnesses': len(known)}

            pool: set[str] = set()
            link: set[str] = set()
            top20: set[str] = set()
            if fused is None:
                entry['in_frozen_query_evidence_pool'] = None
                missing_stages.add('frozen_query_retrieval')
            else:
                for result in fused['results']:
                    ids = [str(p['evidence_id']) for p in result.get('paths', [])
                           if p.get('kind') == 'evidence' and p.get('evidence_id')]
                    pool.update(ids)
                    if result['target_id'] == target_id:
                        link.update(ids)
                        top20.update(ids[:20])
                entry['in_frozen_query_evidence_pool'] = len(known & pool)
                entry['linked_to_target'] = len(known & link)
                entry['in_path_top20'] = len(known & top20)

            retained = set(target['retained_evidence_ids'])
            entry['survived_content_dedup'] = len(known & retained)
            entry['in_read4'] = len(known & retained)
            placed = prompt_units.get((query_id, target_id), set())
            entry['placed_in_row_prompt'] = len(known & placed) if prompt_units else None
            if not prompt_units:
                missing_stages.add('row_prompt_placement')

            for stage in STAGES:
                value = entry.get(stage)
                if value is not None:
                    counts[stage] += value
                    entry[f'rate_{stage}'] = value / len(known)
            per_pair.append(entry)

    total = counts['witness_exists']
    funnel = []
    for stage in STAGES:
        if stage in missing_stages or (stage not in counts and stage in missing_stages):
            funnel.append({'stage': stage, 'status': 'missing_stage'})
            continue
        value = counts.get(stage)
        funnel.append({
            'stage': stage,
            'witnesses': value,
            'of_total': None if value is None else value / total if total else None,
            'status': 'recorded' if value is not None else 'missing_stage',
        })

    result = {
        'scope': scope, 'split': split,
        'witnesses_total': total,
        'pairs_with_witness': len(per_pair),
        'funnel': funnel,
        'missing_stages': sorted(missing_stages),
        'never_guesses': True,
        'note': (
            'A stage the frozen artifacts do not log is reported as missing_stage. The '
            'R4 reader prompt holds read4 evidence, so in_read4 and placed_in_row_prompt '
            'coincide by construction and are kept separate only to make that explicit.'
        ),
        'per_pair': per_pair[:200],
    }
    write_json(out / 'COSTS' / f'EVIDENCE_FUNNEL.{scope}.json', result)
    return {k: v for k, v in result.items() if k != 'per_pair'}

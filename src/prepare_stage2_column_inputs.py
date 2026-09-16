#!/usr/bin/env python
"""Freeze the verified B13+T0 historical own-U anchor, without new retrieval."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
from mmdd_stage2.column_data import digest, file_hash, read_jsonl, write_json, write_jsonl


def freeze_anchor(root: Path, output: Path) -> None:
    base = root/'work/stage1_optimization_r26_20260914'
    student = base/'recovered/B13/step_000178.pt'
    teacher = root/'work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt'
    retrieval = base/'rankings/B13/rankings.jsonl.gz'
    teacher_ranks = base/'teacher/B13/rankings.jsonl.gz'
    r_receipt = json.loads((base/'rankings/B13/RETRIEVAL_RECEIPT.json').read_text())
    t_receipt = json.loads((base/'teacher/B13/TEACHER_RECEIPT.json').read_text())
    for path, expected in [(student, r_receipt['signature']['checkpoint_sha256']),
                           (retrieval, r_receipt['rankings']['sha256']),
                           (teacher_ranks, t_receipt['rankings']['sha256'])]:
        if file_hash(path) != expected:
            raise ValueError(f'Anchor fingerprint mismatch: {path}')
    teacher_reference = json.loads((base/'stage2/inputs/B13/INPUT_RECEIPT.json').read_text())['teacher']
    if file_hash(teacher) != teacher_reference['sha256']:
        raise ValueError('T0 identity mismatch')
    teachers = {r['query_id']: r for r in read_jsonl(teacher_ranks)}
    exports, c50 = [], []
    with gzip.open(retrieval, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            t = teachers[row['query_id']]
            if row['candidate_pool_id'] != t['candidate_pool_id']:
                raise ValueError('Teacher was evaluated on a different Student pool')
            order = t['rankings']['U_OFFLINE_T0']
            if set(order) != set(row['U']):
                raise ValueError('Teacher U ranking changed candidate membership')
            paths = {r['target_id']: r['retained_paths'] for r in row['E_paths']}
            results = [{'target_id': tid, 'rank': rank, 'paths': paths.get(tid, [])}
                       for rank, tid in enumerate(order, 1)]
            exports.append({'query_id': row['query_id'], 'results': results,
                            'candidate_pool_id': row['candidate_pool_id'], 'candidate_path_hash': digest(results)})
            c50.append({'query_id': row['query_id'], 'results': results[:50],
                        'candidate_pool_id': row['candidate_pool_id'], 'actual_candidates': min(50, len(results)),
                        'candidate_path_hash': digest(results[:50])})
    write_jsonl(output/'FROZEN_NATURAL_U.jsonl.gz', exports)
    write_jsonl(output/'FROZEN_C50.jsonl.gz', c50)
    references = [student, teacher, retrieval, teacher_ranks,
                  base/'rankings/Qwen-Raw/RETRIEVAL_RECEIPT.json', base/'teacher/Qwen-Raw/metrics.json']
    selection = {'selection_status': 'verified_historical_anchor_not_latest_strongest',
                 'reason': 'No verified same-population current best-selection report; B13+T0 fallback is allowed by plan',
                 'pipeline': 'B13 natural U; existing T0 U_OFFLINE_T0 order; freeze first 50',
                 'own_pool': True, 'no_new_retrieval_or_stage1_training': True,
                 'retrieval_parameters': r_receipt['signature']['protocol'],
                 'student_architecture': 'existing B13 checkpoint; parameter identity from original retrieval receipt',
                 'references': [{'path': str(p), 'sha256': file_hash(p) if p.is_file() else None} for p in references],
                 'exports': {p.name: file_hash(p) for p in [output/'FROZEN_NATURAL_U.jsonl.gz', output/'FROZEN_C50.jsonl.gz']},
                 'queries': len(exports), 'budget': {'n_target_candidates': 50, 'k_evidence': 4, 'k_columns': [1,2,3,5]},
                 'raw_qwen': 'Existing table-level references only; no reproduction',
                 'scope': 'Historical canonical-dev queries only; unavailable train/test retrieval is explicitly missing, not empty evidence'}
    write_json(output/'MODEL_SELECTION.json', selection)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    freeze_anchor(args.root,args.output)

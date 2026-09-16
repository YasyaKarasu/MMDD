"""Frozen B13/T0 whole-lake train retrieval, followed by positive-target lookup."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .column_data import digest, file_hash, natural_evidence, read_jsonl, write_json, write_jsonl
from .column_r2_audit import read_json


def export_queries(root: Path, folder: Path, query_ids: list[str]) -> dict:
    from evaluate_stage1_r26 import retain_evidence
    from mmdd_stage1.checkpoints import load_student
    from mmdd_stage1.features import FeatureStore
    from mmdd_stage1.retrieval import StudentANNIndices, retrieve_zero_one_hop_detailed_many
    from mmdd_stage1.row_support import load_evidence_content_keys
    from mmdd_stage1.r26_teacher import TeacherPairCache
    from run_stage1_r19 import load_r19_checkpoint
    from run_stage1_r25 import _r25_teacher_feature_paths

    torch.set_num_threads(4)
    device = torch.device('cpu')
    base = root / 'work/stage1_optimization_r26_20260914'
    student_path = base / 'recovered/B13/step_000178.pt'
    teacher_path = root / 'work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt'
    old = read_json(base / 'rankings/B13/RETRIEVAL_RECEIPT.json')
    identity = read_json(base / 'teacher/CACHE_IDENTITY.json')
    if file_hash(student_path) != old['signature']['checkpoint_sha256']:
        raise ValueError('Frozen B13 changed')
    if file_hash(teacher_path) != identity['teacher']['sha256']:
        raise ValueError('Frozen T0 changed')
    student = load_student(student_path, device).eval().requires_grad_(False)
    _, _, _, teacher, _ = load_r19_checkpoint(teacher_path, device)
    teacher.eval().requires_grad_(False)
    feature_path = root / 'work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b'
    store = FeatureStore.from_path(feature_path, cache_size=10000, cache_bytes=4*1024**3,
                                  teacher_paths=_r25_teacher_feature_paths(root))
    indices = StudentANNIndices(student, store, base / 'indexes/B13', device=device,
        checkpoint_sha256=file_hash(student_path), corpus_sha256=old['signature']['corpus_sha256'])
    for index in indices.indices.values():
        index.set_num_threads(4)
    content_path = root / 'work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl'
    content_keys, content_hash = load_evidence_content_keys(content_path)
    folder.mkdir(parents=True, exist_ok=True)
    pair_cache = TeacherPairCache(folder / 'natural_train_T0_pairs.sqlite', identity['namespace'], teacher, store, device)
    records, costs = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for position, qid in enumerate(query_ids, 1):
            indices.clear_query_cache()
            retrieved = retrieve_zero_one_hop_detailed_many([qid], indices, direct_k=100, evidence_k=20,
                targets_per_evidence=20, evidence_aggregation='logsumexp', query_batch_size=1)[0]
            retained = retain_evidence(qid, retrieved, store, content_keys)
            natural_u = sorted({r['target_id'] for r in retrieved['direct']} | {r['target_id'] for r in retained})
            missing = [t for t in natural_u if not store.has_teacher_features(t)]
            if missing:
                raise ValueError(f'Frozen T0 features missing for {len(missing)} natural targets')
            scores, timing = pair_cache.score(qid, natural_u)
            ordered = sorted(natural_u, key=lambda t: (-scores[t], t))
            paths = {r['target_id']: r['retained_paths'] for r in retained}
            results = [{'target_id': tid, 'rank': rank, 'paths': paths.get(tid, [])}
                       for rank, tid in enumerate(ordered, 1)]
            records.append({'query_id': qid, 'results': results,
                'candidate_pool_id': digest({'query_id': qid, 'U': natural_u}), 'candidate_path_hash': digest(results)})
            costs.append(timing)
            if position % 100 == 0:
                print(f'Frozen train B13/T0: {position}/{len(query_ids)}', flush=True)
    pair_cache.db.close()
    path = folder / 'FROZEN_NATURAL_TRAIN.jsonl.gz'
    write_jsonl(path, records)
    return {'natural_export_sha256': file_hash(path), 'queries': len(query_ids),
        'query_ids_sha256': digest(query_ids), 'student_sha256': file_hash(student_path),
        'teacher_sha256': file_hash(teacher_path), 'teacher_namespace': identity['namespace'],
        'index_manifest_sha256': file_hash(base / 'indexes/B13/manifest.json'),
        'feature_manifest_sha256': file_hash(feature_path / 'manifest.jsonl'),
        'evidence_content_keys_sha256': content_hash, 'protocol': old['signature']['protocol'],
        'elapsed_seconds': time.monotonic()-started, 'teacher_pairs': sum(c['requested_pairs'] for c in costs),
        'no_gold_target_conditioning': True, 'no_stage1_training': True,
        'retention': 'exact-content dedup -> path-score top20 -> e2_row_coverage -> budget4',
        'sources': {str(p.relative_to(root)): file_hash(p) for p in [Path(__file__), root/'src/evaluate_stage1_r26.py',
            root/'src/run_stage1_r11_task_e.py', root/'src/mmdd_stage1/retrieval.py', root/'src/mmdd_stage1/r26_teacher.py']}}


def build_natural_train(root: Path, r1: Path, output: Path) -> None:
    if not (output / 'PHASE_A/RESULTS.json').is_file():
        raise ValueError('Complete Phase A before natural train retrieval')
    folder = output / 'NATURAL_TRAIN'
    inputs = read_jsonl(r1 / 'COLUMN_INPUTS.train.jsonl')
    query_ids = sorted({r['query_id'] for r in inputs})
    receipt = export_queries(root, folder, query_ids)
    natural = {r['query_id']: r for r in read_jsonl(folder / 'FROZEN_NATURAL_TRAIN.jsonl.gz')}
    if set(natural) != set(query_ids):
        raise ValueError('Missing natural query: not equivalent to absent target')
    updated, reasons = [], {'target_absent': 0, 'target_present_no_paths': 0, 'non_empty': 0}
    for item in inputs:
        ids = natural_evidence(natural[item['query_id']], item['target_id'])
        present = item['target_id'] in {r['target_id'] for r in natural[item['query_id']]['results']}
        reason = 'non_empty' if ids else 'target_present_no_paths' if present else 'target_absent'
        reasons[reason] += 1
        updated.append({**item, 'evidence_ids': {**item['evidence_ids'], 'O-R': ids},
                        'natural_retrieval_available': True, 'natural_evidence_reason': reason})
    write_jsonl(folder / 'COLUMN_INPUTS.train.jsonl', updated)
    from .oracle import _selected_artifact_records
    wanted = {e for i in updated for e in i['evidence_ids']['O-R']}
    objects = {}
    for entry in read_json(r1 / 'INPUT_MANIFEST.json')['roots']:
        if entry['status'] != 'audited':
            continue
        dataset_root = Path(entry['root'])
        for eid, e in _selected_artifact_records(dataset_root, 'bridge_assets', wanted).items():
            if e['asset_type'] == 'image':
                local = Path(e.get('local_path', ''))
                e['local_path'] = str(local if local.is_file() else dataset_root / e.get('relative_path', ''))
            objects[eid] = {k: e[k] for k in ('asset_id', 'asset_type', 'content', 'local_path') if k in e}
    if set(objects) != wanted:
        raise ValueError('Natural evidence content missing; cannot fill with O-O')
    write_jsonl(folder / 'EVIDENCE_OBJECTS.jsonl.gz', list(objects.values()))
    write_json(folder / 'MANIFEST.json', {**receipt, 'pair_counts': reasons,
        'inputs_sha256': file_hash(folder / 'COLUMN_INPUTS.train.jsonl'),
        'objects_sha256': file_hash(folder / 'EVIDENCE_OBJECTS.jsonl.gz'),
        'r1_train_population_sha256': file_hash(r1 / 'COLUMN_POPULATION.train.jsonl')})

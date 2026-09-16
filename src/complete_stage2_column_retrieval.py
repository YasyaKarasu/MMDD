#!/usr/bin/env python
"""Export missing test natural U with the frozen B13/T0 anchor, without labels."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import FeatureStore
from mmdd_stage1.retrieval import StudentANNIndices, retrieve_zero_one_hop_detailed_many
from mmdd_stage1.row_support import load_evidence_content_keys
from mmdd_stage1.r26_teacher import TeacherPairCache
from evaluate_stage1_r26 import retain_evidence
from run_stage1_r19 import load_r19_checkpoint
from run_stage1_r25 import _r25_teacher_feature_paths
from mmdd_stage2.column_data import digest, file_hash, read_jsonl, write_json, write_jsonl


def complete_test(root: Path, output: Path) -> None:
    torch.set_num_threads(4)
    device = torch.device('cpu')
    base = root/'work/stage1_optimization_r26_20260914'
    student_path = base/'recovered/B13/step_000178.pt'
    teacher_path = root/'work/stage1_optimization_r22_20260911/fresh_lineage/T1-B/seed13/checkpoints/step_010536.pt'
    old_receipt = json.loads((base/'rankings/B13/RETRIEVAL_RECEIPT.json').read_text())
    old_teacher_identity = json.loads((base/'teacher/CACHE_IDENTITY.json').read_text())
    if file_hash(student_path) != old_receipt['signature']['checkpoint_sha256']:
        raise ValueError('Student differs from the frozen historical anchor')
    if file_hash(teacher_path) != old_teacher_identity['teacher']['sha256']:
        raise ValueError('Teacher differs from the frozen historical anchor')
    source = output/'COLUMN_INPUTS.test.jsonl'
    query_ids = sorted({r['query_id'] for r in read_jsonl(source)})
    model = load_student(student_path, device).eval()
    _, _, _, teacher, _ = load_r19_checkpoint(teacher_path, device)
    teacher.eval()
    model.requires_grad_(False)
    teacher.requires_grad_(False)
    features = root/'work/stage1_optimization_r10_20260907/features_qwen3_vl_embedding_8b'
    store = FeatureStore.from_path(features, cache_size=10000, cache_bytes=4*1024**3,
                                   teacher_paths=_r25_teacher_feature_paths(root))
    index_path = base/'indexes/B13'
    indices = StudentANNIndices(model, store, index_path, device=device,
                                 checkpoint_sha256=file_hash(student_path),
                                 corpus_sha256=old_receipt['signature']['corpus_sha256'])
    for index in indices.indices.values():
        index.set_num_threads(4)
    content_keys, _ = load_evidence_content_keys(root/'work/stage1_optimization_r10_20260907/taskB_g5/evidence_content_keys.jsonl')
    pair_cache = TeacherPairCache(output/'natural_test_T0_pairs.sqlite', old_teacher_identity['namespace'], teacher, store, device)
    records, failures, cost = [], [], []
    start = time.monotonic()
    with torch.inference_mode():
        for position, qid in enumerate(query_ids):
            indices.clear_query_cache()
            retrieved = retrieve_zero_one_hop_detailed_many([qid], indices, direct_k=100, evidence_k=20,
                         targets_per_evidence=20, evidence_aggregation='logsumexp', query_batch_size=1)[0]
            retained = retain_evidence(qid, retrieved, store, content_keys)
            natural_u = sorted({r['target_id'] for r in retrieved['direct']} | {r['target_id'] for r in retained})
            missing = [t for t in natural_u if not store.has_teacher_features(t)]
            if missing:
                failures.append({'query_id': qid, 'reason': 'missing_frozen_T0_target_features', 'target_ids': missing})
                continue
            scores, timing = pair_cache.score(qid, natural_u)
            order = sorted(natural_u, key=lambda t: (-scores[t], t))
            paths = {r['target_id']: r['retained_paths'] for r in retained}
            results = [{'target_id': tid, 'rank': rank, 'paths': paths.get(tid, [])}
                       for rank, tid in enumerate(order, 1)]
            records.append({'query_id': qid, 'results': results, 'candidate_pool_id': digest({'query_id': qid, 'U': natural_u}),
                            'candidate_path_hash': digest(results)})
            cost.append(timing)
            if (position+1) % 25 == 0:
                print(f'frozen B13/T0 test retrieval {position+1}/{len(query_ids)}; failures={len(failures)}', flush=True)
    pair_cache.db.close()
    write_jsonl(output/'FROZEN_NATURAL_TEST.jsonl.gz', records)
    write_jsonl(output/'FROZEN_C50_TEST.jsonl.gz', [{**r, 'results': r['results'][:50]} for r in records])
    receipt = {'planned': True, 'implemented': True, 'executed': True, 'evaluated': False,
               'scope': 'Natural whole-lake test retrieval, no labels and no gold-target-conditioned retrieval',
               'no_training': True, 'source_query_ids_sha256': digest(query_ids), 'queries': len(query_ids),
               'successful_queries': len(records), 'failures': failures, 'elapsed_seconds': time.monotonic()-start,
               'device': 'cpu', 'student_sha256': file_hash(student_path), 'teacher_sha256': file_hash(teacher_path),
               'original_protocol': old_receipt['signature']['protocol'],
               'index_manifest_sha256': file_hash(index_path/'manifest.json'),
               'feature_manifest_sha256': file_hash(features/'manifest.jsonl'),
               'code_sha256': {p.name:file_hash(p) for p in [Path(__file__),root/'src/evaluate_stage1_r26.py',root/'src/mmdd_stage1/retrieval.py',root/'src/mmdd_stage1/r26_teacher.py']},
               'candidate_path_export_sha256': file_hash(output/'FROZEN_NATURAL_TEST.jsonl.gz'),
               'total_teacher_pairs': sum(c['requested_pairs'] for c in cost)}
    write_json(output/'RUN_RECEIPTS/frozen_test_retrieval.json', receipt)
    print(json.dumps({'queries':len(query_ids),'successful':len(records),'failures':len(failures)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    complete_test(args.root,args.output)

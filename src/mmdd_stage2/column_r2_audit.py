"""Immutable R1 provenance checks and checkpoint/feature baseline replay."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .checkpoints import load_candidate_scorer
from .column_cache import condition_input, load_features, reader_identity
from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .data import serialize_table
from .column_metrics import evaluate, pair_key
from .column_training import predict


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def r1_caches(r1: Path) -> dict[tuple[str, str, int], Path]:
    result = {}
    for path in sorted((r1 / 'CACHE_MANIFESTS').glob('*.json')):
        m = read_json(path)
        c = m.get('contract', {})
        if (c.get('reader_layout_version') != 'tail_candidates_v1'
                or c.get('limit') is not None or not m.get('complete')):
            continue
        key = (c['split'], c['condition'], c['view'])
        if key in result:
            raise ValueError(f'Ambiguous R1 cache: {key}')
        result[key] = path
    return result


def audit_reader_inputs(r1: Path, output: Path) -> None:
    objects = read_jsonl(r1/'OBJECTS.jsonl.gz')[0]
    image_hashes, checks = {}, []
    for (split, condition, view), path in r1_caches(r1).items():
        if condition not in {'O-O','O-R','No-E'}:
            continue
        manifest = read_json(path)
        items = {pair_key(i):i for i in read_jsonl(r1/f'COLUMN_INPUTS.{split}.jsonl')}
        for entry in manifest['records']:
            item = items[pair_key(entry)]
            query, target, evidence = condition_input(item, objects, condition=condition, view=view)
            images = {}
            for e in evidence:
                if e['asset_type'] == 'image':
                    if e['local_path'] not in image_hashes:
                        image_hashes[e['local_path']] = file_hash(Path(e['local_path']))
                    images[e['asset_id']] = image_hashes[e['local_path']]
            content = {'query':serialize_table(query),
                'target':serialize_table(target,mark_candidates=True,reader_layout_version='tail_candidates_v1'),
                'columns':[c['column_index'] for c in target['columns']], 'evidence':evidence}
            fingerprint = digest({'contract':manifest['contract'],
                'sample_key':{k:item[k] for k in ('dataset','query_id','target_id')},
                'content':content, 'image_bytes':images})
            if fingerprint != entry['input_hash']:
                raise ValueError('R1 current raw reader content differs from cached input')
        checks.append({'split':split,'condition':condition,'view':view,'matched':len(manifest['records'])})
        print(f'Raw reader input hashes match: {split} {condition} view{view}',flush=True)
    for root in read_json(r1/'INPUT_MANIFEST.json')['roots']:
        if root['status']=='audited' and file_hash(Path(root['root'])/'qrels.jsonl') != root['qrels_sha256']:
            raise ValueError('Original qrels changed')
    write_json(output/'R1_RAW_INPUT_RECHECK.json',{'passed':True,'checks':checks,
        'image_files_sha256':image_hashes,'no_new_reader_forward':True})


def audit_and_replay(root: Path, r1: Path, output: Path, model_dir: Path) -> None:
    torch.set_num_threads(4)
    checks = []

    def check(path: Path, expected: str, kind: str) -> None:
        actual = file_hash(path)
        checks.append({'path': str(path), 'kind': kind, 'expected': expected,
                       'actual': actual, 'match': actual == expected})
        if actual != expected:
            write_json(output / 'SOURCE_AUDIT.json', {'passed': False, 'checks': checks})
            raise ValueError(f'R1 {kind} hash mismatch: {path}')

    for entry in read_json(r1 / 'SOURCE_SNAPSHOT/MANIFEST.json')['files']:
        check(root / entry['path'], entry['sha256'], 'source')
    inputs = read_json(r1 / 'INPUT_MANIFEST.json')
    if not inputs['locked']:
        raise ValueError('R1 population is not locked')
    for name, expected in inputs['files'].items():
        check(r1 / name, expected, 'population_or_input')
    check(r1 / 'OBJECTS.jsonl.gz', inputs['objects_sha256'], 'objects')
    for entry in inputs['retrieval']:
        check(Path(entry['path']), entry['sha256'], 'natural_retrieval')
    identity = reader_identity(model_dir, image_pixels=262144)
    if identity != read_json(r1 / 'READER_IDENTITY.json'):
        write_json(output / 'SOURCE_AUDIT.json', {'passed': False, 'checks': checks,
                                                'reader_identity': identity})
        raise ValueError('Reader model/source/runtime identity differs from R1')
    caches = r1_caches(r1)
    cache_rows = []
    for key, path in caches.items():
        m = read_json(path)
        for entry in m['records']:
            if entry['status'] != 'ok':
                raise ValueError(f'Incomplete R1 cache {path}')
            if file_hash(Path(entry['path'])) != entry['sha256']:
                raise ValueError(f'Changed R1 feature {entry["path"]}')
        cache_rows.append({'key': key, 'path': str(path), 'sha256': file_hash(path),
                           'verified_feature_files': len(m['records'])})
        print(f'R1 verified cache {key}: {len(m["records"])} records', flush=True)
    for seed in (13, 29):
        receipt = read_json(r1 / f'RUN_RECEIPTS/C2/{seed}.json')
        check(r1 / 'INPUT_MANIFEST.json', receipt['input_manifest_hash'], 'training_input_manifest')
        for view, expected in enumerate(receipt['cache_hashes']):
            check(caches['train', 'O-O', view], expected, 'training_cache_manifest')
        for name, expected in receipt['checkpoint_hashes'].items():
            check(r1 / f'CHECKPOINTS/C2/{seed}' / name, expected, 'checkpoint')
    write_json(output / 'SOURCE_AUDIT.json', {
        'passed': True, 'checks': checks, 'caches': cache_rows, 'reader_identity': identity,
        'contract_sha256': file_hash(root / 'S2-COL-R2_EXPERIMENT_PLAN_CLARIFIED.zh-CN.md'),
        'r1_source_snapshot_sha256': file_hash(r1 / 'SOURCE_SNAPSHOT/MANIFEST.json'),
        'scope': 'R1 C2 sources, locked inputs, reader identity, complete tail caches and checkpoints'})
    results = []
    for (split, condition, view), path in caches.items():
        if split not in {'dev', 'test'} or view != 0 or condition not in {'O-O', 'O-R', 'No-E', 'Shuffled-E'}:
            continue
        _, records = load_features(path, expected_layout='tail_candidates_v1')
        population = read_jsonl(r1 / f'COLUMN_POPULATION.{split}.jsonl')
        for seed in (13, 29):
            checkpoint = r1 / f'CHECKPOINTS/C2/{seed}/selected.pt'
            saved_path = r1 / f'PREDICTIONS/C2/{seed}/{split}.{condition}.view0.jsonl.gz'
            if not saved_path.is_file():
                continue
            scorer = load_candidate_scorer(checkpoint, torch.device('cpu'), expected_reader_layout='tail_candidates_v1')
            replay = predict(scorer, records, file_hash(checkpoint))
            saved = {pair_key(p): p for p in read_jsonl(saved_path)}
            if len(saved) != len(replay):
                raise ValueError('R1 prediction population differs')
            rank_equal = all(p['ranking'] == saved[pair_key(p)]['ranking'] for p in replay)
            max_delta = max(abs(a-b) for p in replay for a, b in zip(p['logits'], saved[pair_key(p)]['logits'], strict=True))
            metrics, _ = evaluate(population, replay)
            previous, _ = evaluate(population, list(saved.values()))
            if not rank_equal or max_delta > 1e-6 or metrics != previous:
                raise ValueError('R1 checkpoint/cache replay does not match saved predictions')
            write_jsonl(output / f'BASELINE_REPLAY/{seed}/{split}.{condition}.jsonl.gz', replay)
            results.append({'seed': seed, 'split': split, 'condition': condition, 'ranking_equal': rank_equal,
                            'maximum_logit_difference': max_delta, 'metrics_equal': metrics == previous,
                            'query_macro': metrics['query_macro'], 'checkpoint_sha256': file_hash(checkpoint),
                            'cache_manifest_sha256': file_hash(path), 'saved_prediction_sha256': file_hash(saved_path)})
            print(f'R1 replay {seed} {split} {condition}: exact ranks; max logit delta {max_delta}', flush=True)
    write_json(output / 'R1_BASELINE_REPLAY.json', {'passed': True, 'results': results,
        'scope': 'C2 selected checkpoint over hash-verified frozen reader states; no new reader forward',
        'r2_test_selection': False})

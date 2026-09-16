"""Content-addressed frozen reader features for the column experiment."""
from __future__ import annotations

import copy
import importlib.metadata
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .column_data import digest, file_hash, read_jsonl, write_json
from .data import permute_table_columns, serialize_table
from .data import escape_marker_literals

LAYOUTS = {'C0': 'header_markers_v0', 'C1': 'tail_candidates_v1', 'C2': 'tail_candidates_v1'}
VIEW_SEEDS = (13001, 29001, 47001)


def source_fingerprints() -> dict[str, str]:
    folder = Path(__file__).parent
    return {p.name: file_hash(p) for p in sorted(folder.glob('*.py'))}


def reader_identity(model_dir: Path, *, image_pixels: int, dtype: str = 'bf16') -> dict[str, Any]:
    files = [p for p in sorted(model_dir.iterdir()) if p.is_file() and
             (p.suffix in {'.json', '.jinja', '.safetensors'} or p.name in {'merges.txt', 'vocab.txt'})]
    if not any(p.suffix == '.safetensors' for p in files):
        raise FileNotFoundError('No local reader model weights')
    return {'model_dir': str(model_dir.resolve()), 'model_files': {p.name: file_hash(p) for p in files},
            'sources': {p.name: file_hash(p) for p in [Path(__file__).with_name('qwen.py'), Path(__file__).with_name('data.py'),
                        Path(__file__).with_name('column_data.py'), Path(__file__).parents[1]/'mmdd_dataset/utils.py', Path(__file__)]},
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'Pillow')},
            'dtype': dtype, 'image_policy': {'max_pixels': image_pixels, 'first_frame': 0, 'mode': 'RGB', 'resize': 'LANCZOS'},
            'target_rows': 12, 'text_char_budget': 12000, 'anonymize_evidence': True}


def condition_input(item: dict[str, Any], objects: dict[str, Any], *, condition: str, view: int,
                    donor: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    query = objects['queries'][item['query_id']]
    target = permute_table_columns(objects['targets'][item['target_id']], seed=VIEW_SEEDS[view])
    ids = item['evidence_ids']['O-R' if condition == 'O-R' else 'O-O']
    if condition == 'No-E':
        ids = []
    if condition == 'Shuffled-E':
        if donor is None:
            raise ValueError('No eligible cross-source evidence donor')
        ids = donor['evidence_ids']['O-O']
    evidence = [objects['evidence'][eid] for eid in ids]
    if condition == 'ValueShuffle':
        target = copy.deepcopy(target)
        generator = random.Random(int(digest(item['target_id'])[:12], 16))
        # Shuffle cells among visible columns within each row, preserving access IDs.
        for row in target['rows']:
            texts = [c['text'] for c in row['cells']]
            generator.shuffle(texts)
            for cell, text in zip(row['cells'], texts, strict=True):
                cell['text'] = text
    return query, target, evidence


def build_features(output: Path, model_dir: Path, *, layout: str, split: str, condition: str,
                   view: int, device: str, image_pixels: int = 262144, limit: int | None = None,
                   backend: Any = None, identity: dict[str, Any] | None = None,
                   historical_input: bool = False) -> Path:
    manifest = json.loads((output / 'INPUT_MANIFEST.json').read_text())
    if not manifest['locked']:
        raise ValueError('Data audit has not locked the population')
    for name, expected in manifest['files'].items():
        if file_hash(output / name) != expected:
            raise ValueError('Frozen input/population changed')
    if file_hash(output / 'OBJECTS.jsonl.gz') != manifest['objects_sha256']:
        raise ValueError('Frozen reader objects changed')
    items = read_jsonl(output / f'COLUMN_INPUTS.{split}.jsonl')
    if limit is not None:
        items = items[:limit]
    objects = read_jsonl(output / 'OBJECTS.jsonl.gz')[0]
    identity = identity or reader_identity(model_dir, image_pixels=image_pixels)
    contract = {**identity, 'reader_layout_version': layout, 'condition': condition, 'view': view,
                'input_manifest_sha256': file_hash(output / 'INPUT_MANIFEST.json'),
                'split': split, 'limit': limit}
    if historical_input:
        if layout != 'header_markers_v0' or view != 0:
            raise ValueError('Historical input requires original header layout/order')
        contract.update(anonymize_evidence=False, view='original',
                        image_policy={'mode': 'processor_default_with_original_oom_retry', 'retry_max_pixels': 1048576})
    fingerprint = digest(contract)
    folder = output / 'FEATURES' / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    if backend is None:
        if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
            raise ValueError('Reader requires the explicitly authorized GPU')
        from .qwen import QwenStage2Backend
        backend = QwenStage2Backend(model_dir, device=device, reader_layout_version=layout,
                                    reader_anonymize_evidence=True, reader_image_max_pixels=image_pixels)
    backend.reader_layout_version = layout
    backend.reader_anonymize_evidence = not historical_input
    backend.reader_image_max_pixels = None if historical_input else image_pixels
    if backend.model.training or any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError('Reader must be frozen and in eval mode')
    versions = tuple(p._version for p in backend.model.parameters())
    torch.cuda.reset_peak_memory_stats(backend.device)
    image_hashes, entries, latencies = {}, [], []
    populations = read_jsonl(output / f'COLUMN_POPULATION.{split}.jsonl')
    by_key = {(p['query_id'], p['target_id']): p for p in populations}
    start = time.monotonic()
    for i, item in enumerate(items):
        metadata = {k: item[k] for k in ('dataset', 'query_id', 'target_id')}
        try:
            donor = None
            if condition == 'Shuffled-E':
                original = by_key[(item['query_id'], item['target_id'])]
                known_positive_ids = {eid for record in populations if record['query_id'] == item['query_id']
                                      for eid in record['positive_evidence_ids']}
                eligible = [d for d in items if d['query_id'] != item['query_id']
                            and by_key[(d['query_id'], d['target_id'])]['source_table_id'] != original['source_table_id']
                            and by_key[(d['query_id'], d['target_id'])]['modality'] == original['modality']
                            and d['evidence_ids']['O-O']
                            and not set(d['evidence_ids']['O-O']).intersection(known_positive_ids)]
                # Match serialized text length within modality; deterministic identity tie break.
                def evidence_length(record: dict[str, Any]) -> int:
                    return sum(len(objects['evidence'][eid].get('content', '')) for eid in record['evidence_ids']['O-O'])
                donor = min(eligible, key=lambda d: (abs(evidence_length(d) - evidence_length(item)), digest(d)), default=None)
            query, target, evidence = condition_input(item, objects, condition=condition, view=view, donor=donor)
            if historical_input:
                target = objects['targets'][item['target_id']]
            content = {'query': serialize_table(query), 'target': serialize_table(target, mark_candidates=True, reader_layout_version=layout),
                       'columns': [c['column_index'] for c in target['columns']], 'evidence': evidence}
            images = {}
            image_dimensions = {}
            for e in evidence:
                if e['asset_type'] == 'image':
                    from PIL import Image
                    path = e['local_path']
                    if path not in image_hashes:
                        image_hashes[path] = file_hash(Path(path))
                    images[e['asset_id']] = image_hashes[path]
                    with Image.open(path) as decoded:
                        width, height = decoded.size
                    scale = min(1., (image_pixels / (width * height)) ** .5)
                    image_dimensions[e['asset_id']] = {'original': [width, height],
                        'reader': None if historical_input else [max(1, round(width * scale)), max(1, round(height * scale))]}
            key = digest({'contract': contract, 'sample_key': metadata, 'content': content, 'image_bytes': images})
            path = folder / f'{key}.pt'
            base = {**metadata, 'candidate_column_indices': content['columns'], 'input_hash': key,
                    'evidence_ids': [e['asset_id'] for e in evidence], 'reader_layout_version': layout}
            if condition == 'O-R' and not item['natural_retrieval_available']:
                entries.append({**base, 'status': 'retrieval_query_missing', 'path': None})
                continue
            if not path.is_file():
                before = time.monotonic()
                with torch.inference_mode():
                    opened, closed = backend.reader_states(query, target, evidence)
                latency = time.monotonic() - before
                if opened.shape != closed.shape or opened.shape[0] != len(content['columns']):
                    raise ValueError('Reader state/candidate mismatch')
                payload = {**base, 'status': 'ok', 'open_states': opened, 'close_states': closed,
                           'latency_seconds': latency, 'token_count': backend.last_reader_token_count,
                           'visible_target_rows': len(target['rows']), 'target_serialized_sha256': digest(content['target']),
                           'image_dimensions': image_dimensions,
                           'actual_image_policy': backend.last_reader_image_policy,
                           'text_characters_before_truncation': [len(escape_marker_literals(e.get('content', ''))) for e in evidence if e['asset_type'] == 'text'],
                           'evidence_char_limit': max(1, 12000 // max(1, len(evidence)))}
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
            else:
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if payload['input_hash'] != key:
                    raise ValueError('Cache input identity mismatch')
            latencies.append(payload['latency_seconds'])
            entries.append({**base, 'status': 'ok', 'path': str(path), 'sha256': file_hash(path)})
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
            entries.append({**metadata, 'candidate_column_indices': item['candidate_column_indices'],
                            'status': type(error).__name__, 'reason': str(error)[:300], 'path': None})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if (i + 1) % 32 == 0:
            print(f'cache {layout} {split} {condition} view{view}: {i+1}/{len(items)}', flush=True)
    unchanged = versions == tuple(p._version for p in backend.model.parameters())
    if not unchanged:
        raise RuntimeError('Frozen backbone parameters changed')
    timings = sorted(latencies)
    report = {'contract': contract, 'fingerprint': fingerprint, 'records': entries, 'complete': True,
              'hidden_dim': backend.hidden_dim, 'backbone_frozen': True, 'parameter_versions_unchanged': unchanged,
              'elapsed_seconds': time.monotonic() - start, 'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(backend.device),
              'gpu_name': torch.cuda.get_device_name(backend.device),
              'latency_p50': timings[len(timings)//2] if timings else None,
              'latency_p95': timings[min(len(timings)-1, int(.95*len(timings)))] if timings else None,
              'cache_bytes': sum(Path(e['path']).stat().st_size for e in entries if e['path'])}
    # Runtime timings do not mutate a manifest already referenced by a checkpoint.
    # If a retried failure succeeds, publish a distinct output manifest.
    path = output / 'CACHE_MANIFESTS' / f'{fingerprint}.{digest(entries)[:12]}.json'
    if path.is_file():
        previous = json.loads(path.read_text())
        if previous['contract'] != contract or previous['records'] != entries:
            raise ValueError('Existing content-addressed cache manifest changed')
        print(f'cache manifest reused: {path}', flush=True)
        return path
    write_json(path, report)
    print(f'cache manifest: {path}', flush=True)
    return path


def load_features(path: Path, *, expected_layout: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if 'contract' not in manifest or not manifest.get('complete'):
        raise ValueError('Legacy or incomplete cache is not accepted for C0/C1/C2')
    if expected_layout is not None and manifest['contract']['reader_layout_version'] != expected_layout:
        raise ValueError('Head/cache layout mismatch')
    records = []
    for entry in manifest['records']:
        if entry['status'] == 'ok':
            if file_hash(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Feature file changed')
            record = torch.load(entry['path'], map_location='cpu', weights_only=True)
            if record['input_hash'] != entry['input_hash']:
                raise ValueError('Feature input hash mismatch')
            records.append(record)
        else:
            records.append(entry)
    return manifest, records

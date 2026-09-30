"""Label-blind reader jobs, with reuse of audited R1 features and measured costs."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .column_cache import VIEW_SEEDS, condition_input, validate_view_seeds
from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .column_r2_audit import read_json, r1_caches
from .data import escape_marker_literals


def job(item: dict, evidence_ids: list[str], view: int = 0) -> dict:
    if len(evidence_ids) > 4:
        raise ValueError('R2 read4 evidence budget exceeded')
    return {**{k: item[k] for k in ('dataset', 'query_id', 'target_id')},
            'evidence_ids': list(evidence_ids), 'view': view}


def job_key(record: dict) -> str:
    return digest({k: record[k] for k in ('dataset', 'query_id', 'target_id', 'evidence_ids', 'view')})


def phase_a_variants(item: dict) -> dict[str, list[str]]:
    ids = item['evidence_ids']['O-R']
    shuffled = list(ids)
    random.Random(int(digest([item['query_id'], item['target_id']]), 16)).shuffle(shuffled)
    return {'bundle': ids, 'reverse': list(reversed(ids)), 'hash_order': shuffled,
            **{f'single_{j}': [eid] for j, eid in enumerate(ids)},
            **{f'loo_{j}': ids[:j] + ids[j+1:] for j in range(len(ids))}}


def prepare_phase_a(r1: Path, output: Path) -> None:
    if not read_json(output / 'R1_BASELINE_REPLAY.json')['passed']:
        raise ValueError('Baseline replay must pass before Phase A')
    jobs = {}
    for item in read_jsonl(r1 / 'COLUMN_INPUTS.dev.jsonl'):
        if not item['evidence_ids']['O-R']:
            continue
        for ids in phase_a_variants(item).values():
            row = job(item, ids)
            jobs[job_key(row)] = row
    write_jsonl(output / 'PHASE_A/reader_jobs.jsonl.gz', list(jobs.values()))
    print(f'Phase A: {len(jobs)} distinct reader inputs', flush=True)


class FeatureIndex:
    def __init__(self, r1: Path, output: Path) -> None:
        self.r1, self.output = r1, output
        self.entries = {}
        for (split, condition, view), path in r1_caches(r1).items():
            if condition not in {'O-O', 'O-R', 'No-E'}:
                continue
            manifest = read_json(path)
            for entry in manifest['records']:
                key = job_key({**entry, 'view': view})
                self.entries.setdefault(key, {**entry, 'reused_from': str(path),
                    'peak_gpu_memory_bytes': manifest['peak_gpu_memory_bytes']})
        for path in sorted((output / 'FEATURE_INDEX').glob('*.json')):
            self.entries.update(read_json(path)['entries'])

    def get(self, item: dict, ids: list[str], view: int = 0) -> dict:
        entry = self.entries[job_key(job(item, ids, view))]
        record = torch.load(entry['path'], map_location='cpu', weights_only=True)
        if record['input_hash'] != entry['input_hash']:
            raise ValueError('Feature identity mismatch')
        return record


def reader_worker(r1: Path, output: Path, model_dir: Path, jobs_path: Path,
                  shard: int, shards: int, device: str,
                  view_seeds: tuple[int, ...] | list[int] = VIEW_SEEDS) -> None:
    view_seeds = validate_view_seeds(view_seeds)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run with authorized GPU access')
    torch.cuda.set_device(torch.device(device))
    index = FeatureIndex(r1, output)
    objects = read_jsonl(r1 / 'OBJECTS.jsonl.gz')[0]
    extras = output / 'NATURAL_TRAIN/EVIDENCE_OBJECTS.jsonl.gz'
    if extras.is_file():
        objects['evidence'].update({e['asset_id']: e for e in read_jsonl(extras)})
    jobs = [j for j in read_jsonl(jobs_path) if int(job_key(j), 16) % shards == shard]
    from .qwen import QwenStage2Backend

    class MeasuredReader(QwenStage2Backend):
        def _inputs(self, content: list[dict[str, Any]], *, generation_prompt: bool) -> dict:
            inputs = super()._inputs(content, generation_prompt=generation_prompt)
            grid = inputs.get('image_grid_thw')
            patch = self.processor.image_processor.patch_size
            self.processed_image_pixels = int(grid.prod(dim=1).sum()) * patch**2 if grid is not None else 0
            return inputs

    backend = None
    entries = {}
    identity = read_json(r1 / 'READER_IDENTITY.json')
    image_hashes = {}
    folder = output / 'FEATURES'
    folder.mkdir(parents=True, exist_ok=True)
    reuse_audited = view_seeds == tuple(VIEW_SEEDS)
    seed_tag = digest(list(view_seeds))[:12]
    index_path = output / f'FEATURE_INDEX/{jobs_path.name}.{seed_tag}.{shard}.json'
    for position, row in enumerate(jobs, 1):
        key = job_key(row)
        if reuse_audited and key in index.entries:
            entries[key] = index.entries[key]
            continue
        query, target, evidence = condition_input(
            {**row, 'evidence_ids': {'O-O': row['evidence_ids']}}, objects,
            condition='O-O', view=row['view'], view_seeds=view_seeds)
        images = {}
        for e in evidence:
            if e['asset_type'] == 'image':
                path = e['local_path']
                if path not in image_hashes:
                    image_hashes[path] = file_hash(Path(path))
                images[e['asset_id']] = image_hashes[path]
        fingerprint = digest({'reader_identity': identity, 'layout': 'tail_candidates_v1',
            'view_seeds': list(view_seeds),
            'query': query, 'target': target, 'evidence': evidence, 'image_bytes': images, 'job': row})
        destination = folder / f'{fingerprint}.pt'
        if destination.is_file():
            payload = torch.load(destination, map_location='cpu', weights_only=True)
            if payload['input_hash'] != fingerprint:
                raise ValueError('R2 cache content changed')
        else:
            if backend is None:
                backend = MeasuredReader(model_dir, device=device, reader_layout_version='tail_candidates_v1',
                    reader_anonymize_evidence=True, reader_image_max_pixels=262144)
                versions = tuple(p._version for p in backend.model.parameters())
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            opened, closed = backend.reader_states(query, target, evidence)
            torch.cuda.synchronize(device)
            payload = {**row, 'input_hash': fingerprint, 'status': 'ok',
                'candidate_column_indices': [c['column_index'] for c in target['columns']],
                'reader_layout_version': 'tail_candidates_v1', 'open_states': opened, 'close_states': closed,
                'latency_seconds': time.monotonic() - started, 'token_count': backend.last_reader_token_count,
                'image_pixels': backend.processed_image_pixels, 'reader_forwards': 1, 'batch_size': 1,
                'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device),
                'gpu_name': torch.cuda.get_device_name(device), 'image_bytes': images,
                'text_characters_before_truncation': [len(escape_marker_literals(e.get('content','')))
                    for e in evidence if e['asset_type']=='text'],
                'evidence_char_limit': max(1,12000//max(1,len(evidence)))}
            temporary = destination.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(destination)
        entries[key] = {k: v for k, v in payload.items() if k not in {'open_states', 'close_states'}}
        entries[key].update(path=str(destination), sha256=file_hash(destination), reused_from=None)
        if position % 20 == 0:
            write_json(index_path, {'entries': entries, 'jobs_sha256': file_hash(jobs_path)})
            print(f'{device} {jobs_path.name}: {position}/{len(jobs)}', flush=True)
    if backend is not None and (backend.model.training or any(p.requires_grad for p in backend.model.parameters())
            or versions != tuple(p._version for p in backend.model.parameters())):
        raise ValueError('Reader was not completely frozen')
    write_json(index_path, {'entries': entries, 'jobs_sha256': file_hash(jobs_path), 'complete': True,
        'frozen_reader': True, 'shard': shard, 'shards': shards, 'device': device})
    print(f'{device}: finished {len(jobs)} inputs', flush=True)

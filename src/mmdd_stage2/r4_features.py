"""S2-R4 column features over the real C50 candidate pool.

Every candidate target table gets a reader pass, not only the gold one. Cached R2
features are reused only when the job key (dataset, query, target, evidence ids, view)
matches exactly; everything else is computed here with the frozen reader.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Evidence images come from the frozen local dataset. That dataset was built with
# scripts_old/build_mm_joinability_dataset.py, which sets Image.MAX_IMAGE_PIXELS = None, so
# it legitimately contains images larger than PIL's 178 MP decompression-bomb threshold
# (one dev evidence asset is 273 MP). Memory is still bounded downstream: _image_input
# downscales to reader_image_max_pixels before the processor ever sees the frame.
Image.MAX_IMAGE_PIXELS = None

from .column_r2_cache import FeatureIndex, job, job_key
from .column_cache import reader_identity
from .column_data import digest, file_hash, read_jsonl, write_json, write_jsonl
from .data import permute_table_columns, serialize_table
from .r4_common import R1, R4, SPLIT, read_jsonl as r4_read_jsonl

# The Stage-2 reader/column scorer runs on Qwen3.5-9B; the VL-Embedding-8B is a
# separate frozen component. READER_IDENTITY.json in S2-COL-R1 pins the same config hash.
READER_MODEL = Path('/home/oycy/MMDD/hf_models/Qwen3.5-9B')
R2 = Path('/home/oycy/MMDD/work/S2_COL_R2')
VIEW = 0
LAYOUT = 'tail_candidates_v1'
IMAGE_PIXELS = 262144
READ4 = 4


def _objects(out: Path, scope: str) -> dict:
    import gzip
    import json

    path = out / f'READER_OBJECTS.{scope}.jsonl.gz'
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing; run the population phase first')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.loads(handle.readline())


def candidate_jobs(lock: dict[str, dict]) -> list[dict]:
    """Two reader jobs per candidate pair: Prior (no evidence) and Flat-Mix (read4 O-R).

    When a pair retains no evidence the Flat-Mix prompt's evidence block is empty, which is
    character-for-character the Prior prompt. Those two jobs therefore share one job key and
    one reader pass. That is the intended meaning of "an empty E is genuinely empty", not a
    deduplication shortcut, and the count of shared keys is reported so it stays visible.
    """
    rows = []
    for query_id, record in lock.items():
        for target in record['targets']:
            if not target['column_ids']:
                continue  # unqueryable raw table: COLUMN_SCORING_ERROR, no reader prompt
            for condition, ids in (('PRIOR', []), ('FLAT_MIX', target['retained_evidence_ids'])):
                rows.append({
                    'dataset': record['dataset'],
                    'query_id': query_id,
                    'target_id': target['target_id'],
                    'evidence_ids': list(ids),
                    'view': VIEW,
                    'condition': condition,
                    'source_group': record['source_group'],
                    'original_rank': target['original_rank'],
                })
    return rows


def build_features(out: Path, scope: str, *, device: str, shard: int = 0, shards: int = 1,
                   limit: int | None = None) -> dict:
    torch.set_num_threads(4)
    lock = {r['query_id']: r for r in r4_read_jsonl(out / f'CANDIDATE_LOCK.{scope}.jsonl.gz')}
    objects = _objects(out, scope)
    jobs = candidate_jobs(lock)
    if limit:
        jobs = jobs[:limit]
    jobs = [j for j in jobs if int(job_key(j), 16) % shards == shard]

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run with authorized GPU access')

    index = FeatureIndex(R1, R2)
    # The index is rewritten whole on every flush, so its cadence must scale with the job
    # count or a 100k-job run spends most of its time re-serialising.
    flush_every = max(50, len(jobs) // 100)
    folder = out / 'COLUMN_FEATURES'
    folder.mkdir(parents=True, exist_ok=True)
    index_path = out / f'FEATURE_INDEX.r4.{scope}.{shard}.json'
    entries = {}
    if index_path.is_file():
        entries = json.loads(index_path.read_text())['entries']

    identity = reader_identity(READER_MODEL, image_pixels=IMAGE_PIXELS)
    backend = None
    versions = None
    cached = computed = failures = 0
    started = time.monotonic()
    for position, row in enumerate(jobs, 1):
        key = job_key(row)
        if key in entries:
            continue
        if key in index.entries:
            entries[key] = {**index.entries[key], 'origin': 'reused_r2_cache'}
            cached += 1
            continue
        query = objects['queries'][row['query_id']]
        target = permute_table_columns(objects['targets'][row['target_id']], seed=13001)
        evidence = [objects['evidence'][e] for e in row['evidence_ids']]
        images = {}
        unreadable = []
        for item in evidence:
            if item['asset_type'] == 'image':
                try:
                    images[item['asset_id']] = file_hash(Path(item['local_path']))
                except OSError as error:
                    # A missing or unreadable local image must not end the run; record it and
                    # let the job fail into COLUMN_SCORING_ERROR like any other bad input.
                    unreadable.append({'asset_id': item['asset_id'],
                                       'local_path': item['local_path'], 'error': str(error)[:200]})
                    images[item['asset_id']] = f'unreadable:{error.__class__.__name__}'
        if unreadable:
            failures += 1
            entries[key] = {
                **{k: row[k] for k in ('dataset', 'query_id', 'target_id', 'evidence_ids', 'view')},
                'condition': row['condition'], 'status': 'UnreadableImage',
                'reason': unreadable[:5], 'path': None, 'origin': 'failed_r4',
            }
            continue
        fingerprint = digest({
            'reader_identity': identity, 'layout': LAYOUT, 'query': query, 'target': target,
            'evidence': evidence, 'image_bytes': images, 'job': row,
        })
        destination = folder / f'{fingerprint}.pt'
        try:
            if destination.is_file():
                payload = torch.load(destination, map_location='cpu', weights_only=True)
                if payload['input_hash'] != fingerprint:
                    raise ValueError('R4 feature cache content changed')
            else:
                if backend is None:
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA unavailable: run with authorized GPU access')
                    from .qwen import QwenStage2Backend

                    class MeasuredReader(QwenStage2Backend):
                        """Counts processed image pixels the same way the R2 reader worker does."""

                        def _inputs(self, content, *, generation_prompt: bool) -> dict:
                            inputs = super()._inputs(content, generation_prompt=generation_prompt)
                            grid = inputs.get('image_grid_thw')
                            patch = self.processor.image_processor.patch_size
                            self.processed_image_pixels = (
                                int(grid.prod(dim=1).sum()) * patch ** 2 if grid is not None else 0)
                            return inputs

                    backend = MeasuredReader(
                        READER_MODEL, device=device, reader_layout_version=LAYOUT,
                        reader_anonymize_evidence=True, reader_image_max_pixels=IMAGE_PIXELS,
                    )
                    versions = tuple(p._version for p in backend.model.parameters())
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                began = time.monotonic()
                opened, closed = backend.reader_states(query, target, evidence)
                torch.cuda.synchronize(device)
                payload = {
                    **{k: row[k] for k in ('dataset', 'query_id', 'target_id', 'evidence_ids', 'view')},
                    'condition': row['condition'],
                    'input_hash': fingerprint,
                    'status': 'ok',
                    'candidate_column_indices': [c['column_index'] for c in target['columns']],
                    'reader_layout_version': LAYOUT,
                    'open_states': opened,
                    'close_states': closed,
                    'latency_seconds': time.monotonic() - began,
                    'token_count': backend.last_reader_token_count,
                    'image_pixels': backend.processed_image_pixels,
                    'reader_forwards': 1,
                    'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device),
                    'gpu_name': torch.cuda.get_device_name(device),
                }
                temporary = destination.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(destination)
                computed += 1
                del opened, closed
            entries[key] = {
                **{k: v for k, v in payload.items() if k not in {'open_states', 'close_states'}},
                'path': str(destination), 'sha256': file_hash(destination), 'origin': 'computed_r4',
            }
        except Exception as error:  # noqa: BLE001
            if isinstance(error, (torch.cuda.OutOfMemoryError, torch.cuda.CudaError)) or (
                    isinstance(error, RuntimeError) and 'CUDA' in str(error)):
                # A device-level fault is not a property of this one pair. Recording it per
                # job would turn one lost GPU into tens of thousands of fake scoring errors.
                raise
            # One unreadable asset must not end a 100k-job run. The pair is recorded as a
            # scoring error and stays in the candidate pool; the J arm reports it rather
            # than imputing a column distribution for it.
            failures += 1
            entries[key] = {
                **{k: row[k] for k in ('dataset', 'query_id', 'target_id', 'evidence_ids', 'view')},
                'condition': row['condition'], 'status': type(error).__name__,
                'reason': str(error)[:300], 'path': None, 'origin': 'failed_r4',
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if position % flush_every == 0:
            write_json(index_path, {'entries': entries, 'jobs_sha256': digest(jobs),
                                    'complete': False, 'shard': shard, 'shards': shards})
            rate = (time.monotonic() - started) / position
            print(f'{device} shard{shard}: {position}/{len(jobs)} {rate:.3f}s/job '
                  f'eta {(len(jobs) - position) * rate / 60:.1f}min', flush=True)
        if torch.cuda.is_available() and position % 200 == 0:
            torch.cuda.empty_cache()

    if backend is not None and (backend.model.training
                                or any(p.requires_grad for p in backend.model.parameters())
                                or versions != tuple(p._version for p in backend.model.parameters())):
        raise ValueError('Reader was not completely frozen')

    write_json(index_path, {
        'entries': entries, 'jobs_sha256': digest(jobs), 'complete': True, 'frozen_reader': True,
        'shard': shard, 'shards': shards, 'device': device, 'scope': scope,
        'layout': LAYOUT, 'view': VIEW, 'image_pixels': IMAGE_PIXELS,
        'reused_from_r2_cache': cached, 'computed_r4': computed, 'failed_r4': failures,
        'elapsed_seconds': time.monotonic() - started,
    })
    return {'jobs': len(jobs), 'reused_from_r2_cache': cached, 'computed_r4': computed,
            'failed_r4': failures,
            'elapsed_seconds': time.monotonic() - started, 'index': str(index_path)}


def load_feature_matrix(out: Path, scope: str) -> dict:
    """open/close states per (query, target, condition); missing keys stay explicit."""
    import json

    records = {}
    for path in sorted(out.glob(f'FEATURE_INDEX.r4.{scope}.*.json')):
        payload = json.loads(path.read_text())
        if not payload.get('complete'):
            raise ValueError(f'{path} is incomplete')
        for key, entry in payload['entries'].items():
            records[key] = entry
    return records

"""Reader-only measured costs, separating actual cache creation from cold-QT cost."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .column_data import file_hash, read_jsonl, write_json, write_jsonl
from .column_r2_cache import FeatureIndex, job, job_key
from .column_r2_training import inputs_for
from .data import escape_marker_literals


def processed_pixels(record: dict) -> tuple[int, str]:
    if 'image_pixels' in record:
        return record['image_pixels'], 'measured image_grid_thw * patch_size^2'
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    total = 0
    for dimensions in record.get('image_dimensions', {}).values():
        width, height = dimensions['reader']
        height, width = smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=16777216)
        total += height*width
    return total, 'reconstructed from R1 saved reader dimensions and locked processor smart_resize'


def cost_summary(rows: list[dict]) -> dict:
    if not rows:
        return {'pairs': 0}
    latency = [r['reader_latency_seconds'] for r in rows]
    return {'pairs': len(rows), 'reader_forwards': sum(r['reader_forwards'] for r in rows),
        'total_tokens': sum(r['total_tokens'] for r in rows),
        'image_pixels': sum(r['image_pixels'] for r in rows),
        'latency_p50_seconds': float(np.quantile(latency, .5)),
        'latency_p95_seconds': float(np.quantile(latency, .95)),
        'total_reader_seconds': sum(latency),
        'peak_vram_bytes': max(r['peak_vram_bytes'] for r in rows),
        'logical_cache_bytes': sum(r['cache_bytes'] for r in rows), 'batch_size': 1,
        'latency_scope': 'sum of recorded serial reader passes per Q-T; no head or cache-I/O latency'}


def audit_cost(r1: Path, output: Path) -> None:
    index = FeatureIndex(r1, output)
    objects = read_jsonl(r1/'OBJECTS.jsonl.gz')[0]
    objects['evidence'].update({e['asset_id']:e for e in read_jsonl(output/'NATURAL_TRAIN/EVIDENCE_OBJECTS.jsonl.gz')})
    costs = {}

    def record_cost(item: dict, ids: list[str], view: int) -> dict:
        key = job_key(job(item, ids, view))
        if key not in costs:
            entry = index.entries[key]
            path = Path(entry['path'])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            pixels, origin = processed_pixels(payload)
            limit = max(1,12000//max(1,len(ids)))
            truncated = sum(len(escape_marker_literals(objects['evidence'][e].get('content','')))>limit
                for e in ids if objects['evidence'][e]['asset_type']=='text')
            costs[key] = {'reader_forwards': 1, 'total_tokens': payload['token_count'],
                'truncated_text_items':truncated,
                'image_pixels': pixels, 'pixels_origin': origin,
                'reader_latency_seconds': payload['latency_seconds'],
                'peak_vram_bytes': entry['peak_gpu_memory_bytes'], 'cache_bytes': path.stat().st_size,
                'reused_from': entry.get('reused_from'), 'path': str(path),
                'vram_scope': 'R1 entire cache-worker maximum' if entry.get('reused_from') else 'per-pass maximum',
                'latency_origin': 'historical R1' if entry.get('reused_from') else 'R2 CUDA-synchronized'}
        return costs[key]

    rows = []
    for split in ('train', 'dev', 'test'):
        for item in inputs_for(r1, output, split):
            for view in (0,1) if split == 'train' else (0,):
                prior = record_cost(item, [], view)
                for condition in ('O-O', 'O-R') if split == 'train' else ('O-R',):
                    ids = item['evidence_ids'][condition]
                    if len(ids) > 4:
                        raise ValueError('Read4 budget changed')
                    bundle = [record_cost(item, ids, view)] if ids else []
                    separate = [record_cost(item, [eid], view) for eid in ids]
                    modes = {'PRIOR': [prior], 'FLAT': bundle or [prior],
                        'PVR_BUNDLE': [prior]+bundle, 'PVR_SEPARATE': [prior]+separate,
                        'BUNDLE_EVIDENCE_ONLY': bundle, 'SEPARATE_EVIDENCE_ONLY': separate}
                    for arm, passes in modes.items():
                        latency = [p['reader_latency_seconds'] for p in passes]
                        rows.append({**{k:item[k] for k in ('dataset','query_id','target_id')},
                            'split': split, 'view': view, 'condition': condition, 'arm': arm,
                            'evidence_ids': ids, 'evidence_count': len(ids), 'batch_size': 1,
                            **{k:sum(p[k] for p in passes) for k in
                                ('reader_forwards','total_tokens','image_pixels','reader_latency_seconds','cache_bytes','truncated_text_items')},
                            'peak_vram_bytes': max((p['peak_vram_bytes'] for p in passes), default=0),
                            'reader_pass_latency_p50_seconds': float(np.quantile(latency,.5)) if latency else 0.,
                            'reader_pass_latency_p95_seconds': float(np.quantile(latency,.95)) if latency else 0.,
                            'feature_paths': [p['path'] for p in passes]})
    summary = {}
    for split in ('train','dev','test'):
        summary[split] = {}
        for condition in ('O-O','O-R') if split == 'train' else ('O-R',):
            summary[split][condition] = {}
            for arm in ('PRIOR','FLAT','PVR_BUNDLE','PVR_SEPARATE','BUNDLE_EVIDENCE_ONLY','SEPARATE_EVIDENCE_ONLY'):
                chosen = [r for r in rows if r['split']==split and r['condition']==condition and r['arm']==arm]
                summary[split][condition][arm] = {'full':cost_summary(chosen),
                    'non_empty':cost_summary([r for r in chosen if r['evidence_count']]),
                    'by_evidence_count': {str(n):cost_summary([r for r in chosen if r['evidence_count']==n]) for n in range(5)}}
    write_jsonl(output/'COST/per_qt_reader_costs.jsonl.gz',rows)
    # Include Phase A and support cache creation, not only deployed-method features.
    paths = {entry['path']: entry for entry in index.entries.values() if not entry.get('reused_from')}
    actual = []
    for path, entry in paths.items():
        payload = torch.load(path,map_location='cpu',weights_only=True)
        actual.append({'path':path, 'bytes':Path(path).stat().st_size,
            'reader_forwards':1, 'total_tokens':payload['token_count'], 'image_pixels':payload['image_pixels'],
            'reader_latency_seconds':payload['latency_seconds'], 'peak_vram_bytes':entry['peak_gpu_memory_bytes'],
            'cache_bytes':Path(path).stat().st_size})
    write_json(output/'COST/reader_costs.json', {'cold_per_qt':summary, 'actual_new_R2_cache_creation':cost_summary(actual),
        'text_truncation': {'method_unique_reader_inputs':len(costs),
            'truncated_inputs':sum(c['truncated_text_items']>0 for c in costs.values()),
            'truncated_items':sum(c['truncated_text_items'] for c in costs.values())},
        'notes': ['Not an efficiency optimization or equal-compute comparison.',
            'Each evidence pass outputs all candidate columns, never one pass per column.',
            'Prior is cached; evidence-only costs exclude it explicitly.',
            'Latency combines actual R1 historical and R2 synchronized reader measurements; not a fresh end-to-end benchmark.',
            'R1 peak VRAM is cache-worker-level upper bound, R2 peak is per-pass.',
            'Image pixels count spatial grid pixels, not duplicated temporal padding of still images.',
            'Training head epochs cause zero additional reader forwards.']})
    reused = {r['path']:r for r in costs.values() if r['reused_from']}
    write_json(output/'COST/cache_costs.json', {'new_unique_features':len(paths),
        'new_unique_bytes':sum(p['bytes'] for p in actual),
        'reused_method_unique_features':len(reused), 'reused_method_bytes':sum(r['cache_bytes'] for r in reused.values()),
        'per_qt_costs_sha256':file_hash(output/'COST/per_qt_reader_costs.jsonl.gz'),
        'new_feature_measurements':actual})

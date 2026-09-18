"""S2-R4 delivery: execution status, manifest and cost roll-up.

The status vocabulary is fixed so that no reader has to guess how far a module got:

    planned      - written down in the R4 contract, nothing built
    implemented  - code exists and its unit tests pass
    executed     - it actually ran on the real inputs and produced artifacts
    evaluated    - its outputs were scored against labels
    blocked      - the real input or resource it needs is absent
    not_executed - deliberately out of scope this round
"""
from __future__ import annotations

import json
from pathlib import Path

from .r4_common import R4, digest, file_hash, write_json

STATUSES = ('planned', 'implemented', 'executed', 'evaluated', 'blocked', 'not_executed')


def collect_costs(out: Path, scope: str) -> dict:
    """Reader and generator costs, kept per component; never one blended number."""
    import gzip

    reader = []
    for path in sorted(out.glob(f'FEATURE_INDEX.r4.{scope}.*.json')):
        payload = json.loads(path.read_text())
        for entry in payload['entries'].values():
            if entry.get('origin') == 'computed_r4':
                reader.append(entry)
    latencies = sorted(e['latency_seconds'] for e in reader if 'latency_seconds' in e)
    recovery = []
    folder = out / 'RAW_GENERATIONS'
    if folder.is_dir():
        for path in sorted(folder.glob('*.jsonl.gz')):
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        recovery.append(json.loads(line))
    called = [r for r in recovery if r.get('model_called')]
    generation_tokens = sum(r.get('generated_tokens', 0) for r in called)
    prompt_tokens = sum(r.get('prompt_tokens', 0) for r in called)
    return {
        'components_separate': True,
        'reader': {
            'component': 'Qwen3.5-9B stage-2 reader (column scorer input)',
            'unique_forwards': len(reader),
            'latency_p50_seconds': latencies[len(latencies) // 2] if latencies else None,
            'latency_p95_seconds': latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
            if latencies else None,
            'total_gpu_seconds': sum(latencies),
            'image_pixels_total': sum(e.get('image_pixels', 0) for e in reader),
            'prompt_tokens_total': sum(e.get('token_count', 0) for e in reader),
            'peak_gpu_memory_bytes': max((e.get('peak_gpu_memory_bytes', 0) for e in reader),
                                         default=0),
            'gpu_names': sorted({e.get('gpu_name') for e in reader if e.get('gpu_name')}),
            'cache_hits_reused_from_r2': sum(
                1 for path in out.glob(f'FEATURE_INDEX.r4.{scope}.*.json')
                for e in json.loads(path.read_text())['entries'].values()
                if e.get('origin') == 'reused_r2_cache'),
        },
        'generator': {
            'component': 'Qwen3.5-9B row recovery generator',
            'calls': len(called),
            'decisions_without_a_call': len(recovery) - len(called),
            'max_new_tokens': 512,
            'prompt_tokens_total': prompt_tokens,
            'generated_tokens_total': generation_tokens,
            'latency_p50_seconds': _percentile([r.get('elapsed_seconds', 0) for r in called], 0.5),
            'latency_p95_seconds': _percentile([r.get('elapsed_seconds', 0) for r in called], 0.95),
            'cache_hits': sum(1 for r in recovery if r.get('cache_hit')),
        },
        'not_blended': 'reader and generator costs are never added into a single figure; the R3 '
                       'dual-4090 historical timing is not an estimate for a different GPU',
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


# The column features are content-addressed blobs; the manifest points at their index
# rather than hashing tens of thousands of files that the index already identifies.
FEATURE_DIR = 'COLUMN_FEATURES'


def write_manifest(out: Path, status: dict) -> dict:
    files = {}
    feature_files = 0
    feature_bytes = 0
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path.name == 'DELIVERY_MANIFEST.json':
            continue
        relative = str(path.relative_to(out))
        if relative.startswith(FEATURE_DIR + '/'):
            feature_files += 1
            feature_bytes += path.stat().st_size
            continue
        files[relative] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    for index in sorted(out.glob('FEATURE_INDEX.r4.*.json')):
        files[str(index.relative_to(out))] = {'sha256': file_hash(index),
                                              'bytes': index.stat().st_size}
    manifest = {
        'round': 'S2-R4',
        'file_count': len(files),
        'files': files,
        'manifest_sha256': digest(files),
        'column_features': {
            'directory': FEATURE_DIR,
            'files': feature_files,
            'bytes': feature_bytes,
            'identified_by': 'FEATURE_INDEX.r4.*.json (content-addressed; each entry carries its own sha256)',
        },
        'execution_status': status,
        'note': 'the R3 historical delivery is not modified; this manifest describes the R4 '
                'working tree under work/S2_R4 only',
    }
    write_json(out / 'DELIVERY_MANIFEST.json', manifest)
    return {'file_count': len(files), 'manifest_sha256': manifest['manifest_sha256']}

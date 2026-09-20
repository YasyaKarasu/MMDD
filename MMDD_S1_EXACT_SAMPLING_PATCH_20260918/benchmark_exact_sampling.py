"""Synthetic CPU-only benchmark, not a benchmark of the user's running code."""
import argparse
import json
import platform
import ssl
import statistics
import time
from pathlib import Path
from exact_sampling import Corpus, PrefixTask, build_prefix, resolve_negatives, stable_order_reference


def run(n, repeats):
    ids = [f'evidence_{i:09d}' for i in range(n)]
    t0 = time.perf_counter()
    corpus = Corpus.build(ids)
    build_seconds = time.perf_counter() - t0
    excluded = tuple(ids[::3100])
    namespace = 'clean-r1:13:student:S:packet=E:epoch=1:q=query_synthetic_1234567890:anchor=none:uniform'
    task = PrefixTask(namespace, excluded)
    def reference():
        return tuple(stable_order_reference(corpus.members-set(excluded), namespace)[:31])
    def optimized():
        return build_prefix(corpus, task).ids
    # Warm up each path; compare after every measured call.
    expected = reference()
    assert optimized() == expected
    values = {}
    for name, fn in [('original_full_sort', reference), ('exact_prefix_heap', optimized)]:
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = fn()
            times.append(time.perf_counter()-start)
            assert out == expected
        values[name] = {'seconds': times, 'median_seconds': statistics.median(times)}
    p = build_prefix(corpus, task)
    h = p.ids[:16]
    start = time.perf_counter()
    for _ in range(1000):
        assert len(resolve_negatives(corpus, p, h)) == 31
    values['resolve_cached_prefix_median_not_measured_mean_seconds'] = (time.perf_counter()-start)/1000
    return {
        'kind': 'synthetic_strings_not_server_dataset',
        'python': platform.python_version(), 'openssl': ssl.OPENSSL_VERSION,
        'machine': platform.machine(), 'object_count': n,
        'corpus_build_seconds_one_time': build_seconds,
        'all_sample_IDs_and_order_equal': True, 'timing': values,
        'ratio_full_sort_over_heap': values['original_full_sort']['median_seconds']/values['exact_prefix_heap']['median_seconds'],
        'warning': 'Not compared against DeepSeek current NumPy implementation; not an end-to-end or server speedup claim.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=211349)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('BENCHMARK.json'))
    args = parser.parse_args()
    if args.n < 32 or args.repeats < 1:
        parser.error('n >= 32 and repeats >= 1 required')
    report = run(args.n, args.repeats)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))

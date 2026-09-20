"""Exact SHA256-order sampling helpers for CLEAN-R1 (Python 3.10+).

No torch/numpy, no RNG, no GPU, no approximate/truncated digest comparisons.
This is reference integration code, NOT a patch applied to the user's server.
"""
from __future__ import annotations

import hashlib
import heapq
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Corpus:
    ids: tuple[str, ...]
    encoded: tuple[bytes, ...]
    members: frozenset[str]
    fingerprint: str

    @classmethod
    def build(cls, values: Iterable[str]) -> 'Corpus':
        unique = set(values)
        if any(not isinstance(x, str) for x in unique):
            raise TypeError('Canonical corpus IDs must be strings')
        ids = tuple(sorted(unique, key=lambda x: x.encode('utf-8')))
        encoded = tuple(x.encode('utf-8') for x in ids)
        h = hashlib.sha256()
        for b in encoded:
            # Length framing applies ONLY to this cache fingerprint, not sampling.
            h.update(len(b).to_bytes(8, 'big'))
            h.update(b)
        return cls(ids, encoded, frozenset(ids), h.hexdigest())


def stable_order_reference(values: Iterable[str], namespace: str) -> list[str]:
    """Literal original ordering rule, deliberately not optimized."""
    return sorted(set(values), key=lambda s: (
        hashlib.sha256((namespace + '\0' + s).encode('utf-8')).hexdigest(), s))


def smallest_records(records: Iterable[tuple[bytes, int]], k: int) -> list[tuple[bytes, int]]:
    """Compare the ENTIRE 32-byte digest; canonical ID rank breaks digest ties."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('k must be a nonnegative integer')
    return heapq.nsmallest(k, records)


def exact_topk(corpus: Corpus, namespace: str, k: int,
               excluded: Iterable[str] = ()) -> tuple[str, ...]:
    """Exactly stable_order_reference(corpus minus excluded, namespace)[:k]."""
    if not isinstance(namespace, str):
        raise TypeError('namespace must be a string')
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError('k must be a nonnegative integer')
    if k == 0:
        return ()
    blocked = frozenset(excluded)
    template = hashlib.sha256((namespace + '\0').encode('utf-8'))
    copy_prefix = template.copy

    def records() -> Iterator[tuple[bytes, int]]:
        for i, (object_id, object_bytes) in enumerate(zip(corpus.ids, corpus.encoded)):
            if object_id in blocked:
                continue
            h = copy_prefix()
            h.update(object_bytes)
            yield h.digest(), i

    return tuple(corpus.ids[i] for _, i in smallest_records(records(), k))


@dataclass(frozen=True)
class PrefixTask:
    namespace: str
    # Include P, Excluded, and any other eligibility exclusions independent of H.
    static_excluded: tuple[str, ...]
    negative_budget: int = 31


@dataclass(frozen=True)
class HashPrefix:
    corpus_fingerprint: str
    task: PrefixTask
    ids: tuple[str, ...]


def build_prefix(corpus: Corpus, task: PrefixTask) -> HashPrefix:
    return HashPrefix(corpus.fingerprint, task,
                      exact_topk(corpus, task.namespace, task.negative_budget,
                                 task.static_excluded))


def resolve_negatives(corpus: Corpus, prefix: HashPrefix,
                      hard_rank: Iterable[str], hard_budget: int = 16) -> tuple[str, ...]:
    """Resolve H + first (31-|H|) SHA-ordered remaining IDs, exactly.

    Assumes the original hard-shortfall fill continues the SAME uniform hash
    namespace/order. If production uses another purpose namespace, apply
    exact_topk separately to that call; do NOT silently normalize its behavior.
    """
    if prefix.corpus_fingerprint != corpus.fingerprint:
        raise ValueError('Prefix belongs to a different corpus')
    total = prefix.task.negative_budget
    if not 0 <= hard_budget <= total:
        raise ValueError('Require 0 <= hard_budget <= negative_budget')
    blocked = frozenset(prefix.task.static_excluded)
    hard: list[str] = []
    seen: set[str] = set()
    if hard_budget:
        for x in hard_rank:
            if x not in corpus.members or x in blocked or x in seen:
                continue
            hard.append(x)
            seen.add(x)
            if len(hard) == hard_budget:
                break
    need = total - len(hard)
    uniform = tuple(x for x in prefix.ids if x not in seen)[:need]
    legal_count = len(corpus.members) - len(blocked & corpus.members)
    if len(hard) + len(uniform) != min(total, legal_count):
        raise ValueError('Insufficient prefix: wrong exclusions/budget or corrupt cache')
    return tuple(hard) + uniform


_WORKER_CORPUS: Corpus | None = None


def _init_worker(corpus: Corpus) -> None:
    global _WORKER_CORPUS
    _WORKER_CORPUS = corpus


def _run_task(task: PrefixTask) -> HashPrefix:
    if _WORKER_CORPUS is None:
        raise RuntimeError('Worker not initialized')
    return build_prefix(_WORKER_CORPUS, task)


def ordered_prefixes(corpus: Corpus, tasks: Iterable[PrefixTask], *,
                     workers: int = 4, max_pending: int = 16) -> Iterator[HashPrefix]:
    """Bounded CPU-only prefetch. Yields in INPUT ORDER, not completion order.

    Call from an import-safe script with an `if __name__ == '__main__'` guard.
    Workers receive the static corpus once; tasks send only small descriptors.
    Model, GPU tensors, and model RNG are never passed to workers.
    """
    if workers < 1 or max_pending < 1:
        raise ValueError('workers and max_pending must be positive')
    if workers == 1:
        for task in tasks:
            yield build_prefix(corpus, task)
        return
    iterator = iter(tasks)
    pending = deque()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(corpus,)) as executor:
        for _ in range(max_pending):
            task = next(iterator, None)
            if task is None:
                break
            pending.append(executor.submit(_run_task, task))
        while pending:
            result = pending.popleft().result()
            task = next(iterator, None)
            if task is not None:
                pending.append(executor.submit(_run_task, task))
            yield result

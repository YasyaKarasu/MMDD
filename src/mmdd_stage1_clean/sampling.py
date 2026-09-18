"""Packet construction, stable-hash sampling and negative provenance.

Spec sections 3.1, 7.2 and 7.3.  Every random draw is a deterministic
SHA256-keyed sort over the whole legal remainder, never Python's salted hash.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Sequence

from .util import (
    SamplingCorpus,
    byte_order,
    stable_key,
    stable_order,
    stable_order_topk,
)

HARD_NEGATIVES = 16
UNIFORM_NEGATIVES = 15
HARD_POOL_TOP_N = 128

D_PACKET = "D"
E_PACKET = "E"
C_PACKET = "C"
B_PACKET = "B"

PURPOSE_RANDOM = "uniform"
PURPOSE_LIST_ORDER = "list-order"


def witness_namespace(query_id: str) -> str:
    return f"witness-cycle:13:{query_id}"


def b_view_hash(query_id: str) -> int:
    digest = hashlib.sha256(("B-view:13:" + query_id).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def use_natural_bundle(epoch: int, query_id: str) -> bool:
    """Spec 7.3: even -> natural B, odd -> witness-augmented B."""
    return (epoch + b_view_hash(query_id)) % 2 == 0


def sampling_namespace(
    phase: str,
    sampling_arm: str,
    packet: str,
    epoch: int,
    query_id: str,
    anchor: str | None,
    purpose: str,
) -> str:
    return (
        f"clean-r1:13:{phase}:{sampling_arm}:packet={packet}:epoch={epoch}:"
        f"q={query_id}:anchor={anchor or 'none'}:{purpose}"
    )


def query_order_namespace(phase: str, epoch: int) -> str:
    """``phase`` is T or S; both Student arms share the S order (spec 3.1)."""
    return f"query-order:13:{phase}:epoch={epoch}"


class MakeList:
    """Spec 7.2: all task positives followed by 31 negatives, hash-reordered."""

    def __init__(
        self,
        *,
        phase: str,
        sampling_arm: str,
        corpus_universe: dict[str, Sequence[str]],
        corpus_index: dict[str, SamplingCorpus] | None = None,
    ) -> None:
        self.phase = phase
        self.sampling_arm = sampling_arm
        self.corpus_universe = corpus_universe
        # A caller that builds lists repeatedly (once per packet per query) passes
        # the byte-ordered index in, so it is derived once instead of per call.
        self._corpus: dict[str, SamplingCorpus] = dict(corpus_index or {})

    def corpus(self, destination: str) -> SamplingCorpus:
        cached = self._corpus.get(destination)
        if cached is None:
            cached = SamplingCorpus.build(self.corpus_universe[destination])
            self._corpus[destination] = cached
        return cached

    def hard_pool(
        self,
        hard_rank: Sequence[tuple[str, float]] | None,
        *,
        top_n: int = HARD_POOL_TOP_N,
    ) -> list[str]:
        """HardRank is真实score降序/ID升序; no random perturbation is added."""
        if not hard_rank:
            return []
        ordered = sorted(hard_rank, key=lambda item: (-float(item[1]), item[0]))
        out: list[str] = []
        seen: set[str] = set()
        for value, _ in ordered:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= top_n:
                break
        return out

    def build_legacy(self, **kw):
        """Reconstruction of the pre-patch algorithm, for equivalence testing."""
        return _legacy_build(self, **kw)

    def build(
        self,
        *,
        packet: str,
        epoch: int,
        query_id: str,
        anchor: str | None,
        destination: str,
        positives: Sequence[str],
        excluded: Iterable[str],
        hard_rank: Sequence[tuple[str, float]] | None,
        hard_count: int = HARD_NEGATIVES,
        uniform_count: int = UNIFORM_NEGATIVES,
    ) -> "NegativeList":
        positive_ids = byte_order(positives)
        positive_set = set(positive_ids)
        blocked = frozenset(set(excluded) | positive_set)
        universe_set = self._universe_set(destination)
        namespace = sampling_namespace(
            self.phase, self.sampling_arm, packet, epoch, query_id, anchor, PURPOSE_RANDOM
        )

        # One exact SHA-ordered head of the legal pool serves every hard list this
        # key can produce: hard-fill and the uniform draw use the same namespace, so
        # removing at most `hard_count` ids from that head always leaves the first
        # `budget - |H|` legal ids, which is exactly what the rule asks for.
        corpus = self.corpus(destination)
        legal_count = corpus.legal_count(blocked)
        namespace = sampling_namespace(
            self.phase, self.sampling_arm, packet, epoch, query_id, anchor, PURPOSE_RANDOM
        )

        pool = [
            value
            for value in self.hard_pool(hard_rank)
            if value not in blocked and value in universe_set
        ]
        hard = pool[:hard_count]
        provenance = {value: "hard_negative" for value in hard}

        # Whatever the hard ranking cannot supply is filled from the same
        # SHA-ordered draw that supplies the uniform competitors, so a short hard
        # pool lengthens that draw rather than shortening the list.
        if len(hard) < hard_count:
            fill = stable_order_topk(
                (), namespace, hard_count - len(hard),
                blocked=blocked | frozenset(hard), corpus=corpus,
            )
            hard = hard + fill
            for value in fill:
                provenance[value] = "hard_pool_exhausted_hash_fill"

        uniform = stable_order_topk(
            (), namespace, uniform_count,
            blocked=blocked | frozenset(hard), corpus=corpus,
        )
        if len(hard) + len(uniform) < min(hard_count + uniform_count, legal_count):
            raise HashPoolExhausted(
                f"{packet}/{query_id}: the legal pool holds {legal_count} objects, "
                f"fewer than the {hard_count + uniform_count} required competitors; "
                "the destination corpus cannot supply the pre-registered list size"
            )
        for value in uniform:
            provenance[value] = "uniform_negative"

        negatives = hard + uniform
        # Reorder the whole list so the model can never read a position label.
        whole = stable_order(list(positive_ids) + negatives, namespace)
        return NegativeList(
            positive_ids=positive_ids,
            negative_ids=negatives,
            ordered_ids=whole,
            provenance=provenance,
        )

    def _universe_set(self, destination: str) -> set[str]:
        cached = getattr(self, "_universe_cache", None)
        if cached is None:
            cached = {}
            self._universe_cache = cached
        if destination not in cached:
            cached[destination] = set(self.corpus_universe[destination])
        return cached[destination]


class HashPoolExhausted(RuntimeError):
    """The legal remainder is smaller than the required uniform sample."""


class NegativeList:
    def __init__(
        self,
        *,
        positive_ids: list[str],
        negative_ids: list[str],
        ordered_ids: list[str],
        provenance: dict[str, str],
    ) -> None:
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids
        self.ordered_ids = ordered_ids
        self.provenance = provenance

    def __len__(self) -> int:
        # Spec 7.2: |P| + 31, not a forced 32.
        return len(self.ordered_ids)

    def labels(self) -> dict[str, int]:
        """Label per list entry.  Only reviewed GT becomes a binary 0/1."""
        out: dict[str, int] = {}
        for value in self.positive_ids:
            out[value] = 1
        for value in self.negative_ids:
            out[value] = 0
        return out

    def provenance_counts(self) -> dict[str, int]:
        counts = {"positive": len(self.positive_ids)}
        for value in self.negative_ids:
            counts[self.provenance.get(value, "unknown")] = (
                counts.get(self.provenance.get(value, "unknown"), 0) + 1
            )
        return counts


def select_witness_anchor(witnesses: Sequence[str], query_id: str, epoch: int) -> str | None:
    """Spec 7.3 C packet: position (epoch-1) mod |W_Q| in the stable witness order."""
    if not witnesses:
        return None
    ordered = stable_order(witnesses, witness_namespace(query_id))
    return ordered[(epoch - 1) % len(ordered)]


def augmented_bundle(
    natural: Sequence[str],
    anchor: str | None,
    *,
    modality: dict[str, str],
) -> list[str]:
    """Spec 7.3 B packet: replace the last same-modality entry, else append."""
    bundle = list(natural)
    if anchor is None:
        return bundle[:20]
    if anchor in bundle:
        return bundle[:20]
    anchor_modality = modality.get(anchor)
    same = [i for i, value in enumerate(bundle) if modality.get(value) == anchor_modality]
    if same:
        bundle[same[-1]] = anchor
    else:
        bundle.append(anchor)
    return bundle[:20]


def support_positive_set(
    *,
    direct: set[str],
    implicit: set[str],
    witnesses: dict[str, set[str]],
    context: set[str],
) -> set[str]:
    """Spec Eq. (5)/(6): P_B = D_Q u {T in I_Q : W(Q,T) n B non-empty}."""
    positives = set(direct)
    for target in implicit:
        if witnesses.get(target, set()) & context:
            positives.add(target)
    return positives


def _legacy_build(self, *, packet, epoch, query_id, anchor, destination, positives,
                  excluded, hard_rank, hard_count=HARD_NEGATIVES,
                  uniform_count=UNIFORM_NEGATIVES):
    """The pre-patch list construction: full sorts, no partial selection.

    Kept so a test can assert the optimized path produces identical lists.
    """
    positive_ids = byte_order(positives)
    positive_set = set(positive_ids)
    excluded_set = set(excluded) | positive_set
    universe = self.corpus_universe[destination]
    legal = [value for value in universe if value not in excluded_set]
    pool = [
        value
        for value in self.hard_pool(hard_rank)
        if value not in excluded_set and value in self._universe_set(destination)
    ]
    hard = pool[:hard_count]
    provenance = {value: "hard_negative" for value in hard}
    namespace = sampling_namespace(
        self.phase, self.sampling_arm, packet, epoch, query_id, anchor, PURPOSE_RANDOM
    )
    if len(hard) < hard_count:
        remaining = [value for value in legal if value not in set(hard)]
        top_up = stable_order(remaining, namespace)[: hard_count - len(hard)]
        hard = hard + top_up
        for value in top_up:
            provenance[value] = "hard_pool_exhausted_hash_fill"
    remaining = [value for value in legal if value not in set(hard)]
    if len(remaining) < uniform_count:
        raise HashPoolExhausted(f"{packet}/{query_id}: insufficient legal remainder")
    uniform = stable_order(remaining, namespace)[:uniform_count]
    for value in uniform:
        provenance[value] = "uniform_negative"
    negatives = hard + uniform
    whole = stable_order(list(positive_ids) + negatives, namespace)
    return NegativeList(positive_ids=positive_ids, negative_ids=negatives,
                        ordered_ids=whole, provenance=provenance)

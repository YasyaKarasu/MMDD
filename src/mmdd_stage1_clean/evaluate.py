"""Online retrieval, Teacher re-ranking, metrics and diagnostics.

Spec sections 10 and 12.  The final score is always the shared Teacher's J
value over the full natural bundle; nothing adds, LSEs or RRF-fuses path logits.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from . import reference
from .config import ConfigError
from .models import MODE_J, MODE_P, ObjectBank, TeacherBatch, conditional_keys, project_corpus_keys
from .retrieve import (
    AnnIndex,
    ann_order,
    exact_topk,
    interleave_text_image,
    nn_fidelity,
    retrieve_query,
)
from .timing import Timing
from .util import log_line, stable_digest, write_json, write_jsonl


class TeacherLogitCache:
    """SQLite cache for singleton Teacher scores (spec 9.2).

    Keys include the Teacher identity hash, the mode, the query, the full context
    id list and the candidate id, so a score computed under a different model or
    a different bundle can never be reused (spec 14.14).
    """

    def __init__(self, path: Path, *, cache_name: str | None = None) -> None:
        self.path = Path(path)
        self.cache_name = cache_name
        # A parallel evaluation runs several of these against the same file, so the
        # connection is opened in WAL mode with a busy timeout: writers wait their
        # turn instead of failing with "database is locked".  WAL also lets readers
        # proceed while a writer holds the lock, which is exactly the access pattern
        # when one process scores a query while another records its own.
        self.connection = sqlite3.connect(str(self.path), timeout=120.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=120000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS logits ("
            "k TEXT PRIMARY KEY, teacher_hash TEXT, mode INTEGER, query_id TEXT, "
            "context TEXT, candidate TEXT, value REAL, created_utc TEXT)"
        )
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(teacher_hash: str, mode: int, query_id: str, context: Sequence[str], candidate: str) -> str:
        return stable_digest(teacher_hash, mode, query_id, list(context), candidate)

    def get_many(self, keys: Sequence[str]) -> dict[str, float]:
        if not keys:
            return {}
        out: dict[str, float] = {}
        chunk = 500
        for start in range(0, len(keys), chunk):
            block = keys[start : start + chunk]
            marks = ",".join("?" * len(block))
            for key, value in self.connection.execute(
                f"SELECT k, value FROM logits WHERE k IN ({marks})", block
            ):
                out[key] = float(value)
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, rows: Iterable[tuple[str, Any]]) -> None:
        self.connection.executemany(
            "INSERT OR REPLACE INTO logits (k, teacher_hash, mode, query_id, context, "
            "candidate, value, created_utc) VALUES (?,?,?,?,?,?,?,datetime('now'))",
            rows,
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else None,
            "rows": int(self.connection.execute("SELECT COUNT(*) FROM logits").fetchone()[0]),
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


class RetrievalEngine:
    """Builds the per-path indexes and runs the spec 10.2 pipeline."""

    def __init__(
        self,
        *,
        student: reference.CompactStudent,
        bank: ObjectBank,
        corpora: dict[str, list[str]],
        retrieval: dict[str, Any],
        ann: dict[str, Any],
        seed: int,
        device: str,
        build_ann: bool = True,
        timing: Timing | None = None,
    ) -> None:
        self.student = student
        self.bank = bank
        self.corpora = {k: list(v) for k, v in corpora.items()}
        self.retrieval = retrieval
        self.device = device
        self.timing = timing if timing is not None else Timing()
        self.target_ids = self.corpora["target"]
        self.text_ids = self.corpora["evidence_text"]
        self.image_ids = self.corpora["evidence_image"]
        with self.timing.stage("corpus_keys_project"):
            self.target_keys_C = project_corpus_keys(student, bank, self.target_ids, "C", device)
            self.D_keys = project_corpus_keys(student, bank, self.target_ids, "D", device)
            self.E_text_keys = project_corpus_keys(student, bank, self.text_ids, "E", device)
            self.E_image_keys = project_corpus_keys(student, bank, self.image_ids, "E", device)
        self.indices: dict[str, AnnIndex] = {}
        self.ann_enabled = build_ann
        if build_ann:
            with self.timing.stage("index_build_total"):
                self.indices["target"] = AnnIndex(
                    self.target_keys_C, ann, seed, label="target", timing=self.timing
                )
                if len(self.text_ids):
                    self.indices["text"] = AnnIndex(
                        self.E_text_keys, ann, seed, label="text", timing=self.timing
                    )
                if len(self.image_ids):
                    self.indices["image"] = AnnIndex(
                        self.E_image_keys, ann, seed, label="image", timing=self.timing
                    )

    # -- student-side queries ---------------------------------------------

    def query_vectors(self, query_id: str) -> tuple[np.ndarray, np.ndarray]:
        # The Student path is the part of online latency that is *not* the frozen
        # Qwen encode, which happens once per new query and is reported separately.
        with self.timing.stage("student_query_vectors"):
            base = self._base([query_id])[0]
            direct = _unit(base @ self.student.direct.weight.detach().cpu().numpy().T)
            evidence = _unit(base @ self.student.evidence.weight.detach().cpu().numpy().T)
        return direct, evidence

    def conditional(self, query_id: str, evidence_id: str) -> np.ndarray:
        with self.timing.stage("student_conditional_query"):
            return conditional_keys(
                self.student, self.bank, [query_id], [evidence_id], self.device
            )[0]

    def _base(self, query_ids: Sequence[str]) -> np.ndarray:
        self.student.eval()
        out = np.zeros((len(query_ids), self.student.d), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(query_ids), 512):
                block = list(query_ids[start : start + 512])
                cache, valid, modality, kind = self.bank.keys(block, self.device)
                out[start : start + 512] = (
                    self.student.encode(cache, valid, modality, kind).float().cpu().numpy()
                )
        return out

    def student_target_scores(self, query_id: str, candidates: Sequence[str]) -> np.ndarray:
        direct, _ = self.query_vectors(query_id)
        position = {value: i for i, value in enumerate(self.target_ids)}
        keys = self.target_keys_C[[position[c] for c in candidates]]
        return keys @ direct

    # -- ranking callables -------------------------------------------------

    def exact_rankers(self, query_id: str) -> dict[str, Callable[[np.ndarray, int], list[str]]]:
        def target(vector: np.ndarray, k: int) -> list[str]:
            order, _ = exact_topk(vector[None, :], self.target_keys_C, k,
                                  query_batch=1,
                                  corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
            return [self.target_ids[i] for i in order[0]]

        def text(vector: np.ndarray, k: int) -> list[str]:
            order, _ = exact_topk(vector[None, :], self.E_text_keys, k, query_batch=1,
                                  corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
            return [self.text_ids[i] for i in order[0]]

        def image(vector: np.ndarray, k: int) -> list[str]:
            order, _ = exact_topk(vector[None, :], self.E_image_keys, k, query_batch=1,
                                  corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
            return [self.image_ids[i] for i in order[0]]

        return {"target": target, "text": text, "image": image}

    def ann_rankers(self, query_id: str) -> dict[str, Callable[[np.ndarray, int], list[str]]]:
        if not self.ann_enabled:
            raise ConfigError("ANN ranking requested but no index was built")

        def target(vector: np.ndarray, k: int) -> list[str]:
            return ann_order(self.indices["target"], vector, k, self.target_ids)

        def text(vector: np.ndarray, k: int) -> list[str]:
            return ann_order(self.indices["text"], vector, k, self.text_ids)

        def image(vector: np.ndarray, k: int) -> list[str]:
            return ann_order(self.indices["image"], vector, k, self.image_ids)

        return {"target": target, "text": text, "image": image}

    def pipeline_both(self, query_id: str) -> dict[str, Any]:
        """Steps 2-8 twice for one query: once on ANN, once on the exact index.

        Both runs share one B_Q, because step 4 reads no target and no GT, so the
        bundle is a property of the query only.  The conditional second hop is
        computed once per distinct natural evidence and reused by both rankers.
        """
        with self.timing.stage("pipeline_total"):
            return self._pipeline_both_inner(query_id)

    def pipeline(self, query_id: str, *, use_ann: bool) -> dict[str, Any]:
        """Steps 2-8 once for one query, on a single ranker path.

        ``pipeline_both`` is preferred where both paths are wanted, because it
        computes the conditional second hop once for them.  This entry point exists
        for callers that only need one path -- a per-epoch dev check, for instance,
        scores the Student's own candidates and does not need the exact index.
        """
        with self.timing.stage("pipeline_total"):
            return self._pipeline_both_inner(query_id, paths=(("ann", True),) if use_ann else (("exact", False),))

    def _pipeline_both_inner(
        self, query_id: str, paths: tuple[tuple[str, bool], ...] | None = None
    ) -> dict[str, Any]:
        with self.timing.stage("query_vectors"):
            direct, evidence = self.query_vectors(query_id)
        per = int(self.retrieval["evidence_per_modality"])
        outputs: dict[str, Any] = {}
        bundles: dict[str, list[str]] = {}
        for label, use_ann in (paths if paths is not None else (("ann", True), ("exact", False))):
            if use_ann and not self.ann_enabled:
                continue
            rankers = self.ann_rankers(query_id) if use_ann else self.exact_rankers(query_id)
            with self.timing.stage("evidence_retrieval", path=label):
                e_text = rankers["text"](evidence, per)
                e_image = rankers["image"](evidence, per)
                if use_ann or "ann" not in bundles:
                    b_q = interleave_text_image(e_text, e_image, per, per * 2)
                else:
                    b_q = bundles["ann"]
            bundles[label] = b_q
            with self.timing.stage("conditional_queries", path=label):
                conditions = {value: self.conditional(query_id, value) for value in b_q}
            with self.timing.stage("candidate_admission", path=label):
                outputs[label] = retrieve_query(
                    query_id=query_id,
                    rank_target=rankers["target"],
                    rank_text=rankers["text"],
                    rank_image=rankers["image"],
                    condition=lambda position: conditions[b_q[position]],
                    u_D=direct,
                    u_E=evidence,
                    retrieval=self.retrieval,
                    bundle=b_q,
                )
        return outputs


def _unit(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or (norms < 1e-6).any():
        raise ConfigError("non-finite or degenerate Student vector before unit normalization")
    return values / norms


def recall_at_k(ranked: Sequence[str], positives: Iterable[str], k: int) -> float:
    """Spec Eq. (24).  Targets are de-duplicated, then the top-k is taken."""
    positive = set(positives)
    if not positive:
        return 0.0
    seen: list[str] = []
    marks: set[str] = set()
    for value in ranked:
        if value in marks:
            continue
        marks.add(value)
        seen.append(value)
        if len(seen) >= k:
            break
    return len(positive.intersection(seen)) / len(positive)


def query_metrics(
    ranked: Sequence[str], direct: Iterable[str], implicit: Iterable[str], ks: Sequence[int]
) -> dict[str, float]:
    direct_set = set(direct)
    implicit_set = set(implicit)
    overall = direct_set | implicit_set
    out: dict[str, float] = {}
    for k in ks:
        out[f"overall_R{k}"] = recall_at_k(ranked, overall, k)
        if direct_set:
            out[f"explicit_R{k}"] = recall_at_k(ranked, direct_set, k)
        if implicit_set:
            out[f"implicit_R{k}"] = recall_at_k(ranked, implicit_set, k)
    return out


def macro_metrics(
    per_query: Sequence[dict[str, float]], ks: Sequence[int]
) -> dict[str, Any]:
    """Query macro with separate explicit/implicit denominators (spec 12.1)."""
    out: dict[str, Any] = {"queries": len(per_query)}
    for k in ks:
        overall = [q[f"overall_R{k}"] for q in per_query if f"overall_R{k}" in q]
        out[f"overall_R{k}"] = float(np.mean(overall)) if overall else None
        for view in ("explicit", "implicit"):
            values = [q[f"{view}_R{k}"] for q in per_query if f"{view}_R{k}" in q]
            out[f"{view}_R{k}"] = float(np.mean(values)) if values else None
            out[f"{view}_R{k}_queries"] = len(values)
        if f"overall_R{k}" in out and out[f"overall_R{k}"] is None:
            out[f"overall_R{k}"] = None
    return out


def admissibility(
    ranked: Sequence[str], positives: Iterable[str], ks: Sequence[int]
) -> dict[str, Any]:
    positive = set(positives)
    out: dict[str, Any] = {"positive_targets": len(positive)}
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(ranked, positive, k)
    out["has_any_positive_at_100"] = recall_at_k(ranked, positive, 100) > 0
    return out


class RawEngine:
    """Spec 12.2 RAW-D / RAW-2H / RAW+T: the frozen Qwen z vectors as keys.

    RAW-2H uses z_E directly for the second hop rather than splicing Q and E
    vectors, so it is a frozen-embedding two-hop baseline, not a trained model.
    """

    def __init__(
        self,
        *,
        bank: ObjectBank,
        corpora: dict[str, list[str]],
        retrieval: dict[str, Any],
        ann: dict[str, Any],
        seed: int,
        device: str,
        build_ann: bool = True,
    ) -> None:
        self.corpora = {k: list(v) for k, v in corpora.items()}
        self.target_ids = self.corpora["target"]
        self.text_ids = self.corpora["evidence_text"]
        self.image_ids = self.corpora["evidence_image"]
        self.retrieval = retrieval
        self.device = device
        self.target_z = _unit(bank.z[[bank.position[o] for o in self.target_ids]])
        self.text_z = _unit(bank.z[[bank.position[o] for o in self.text_ids]])
        self.image_z = _unit(bank.z[[bank.position[o] for o in self.image_ids]])
        self.indices: dict[str, AnnIndex] = {}
        self.ann_enabled = build_ann
        if build_ann:
            self.indices["target"] = AnnIndex(self.target_z, ann, seed)
            if len(self.text_ids):
                self.indices["text"] = AnnIndex(self.text_z, ann, seed)
            if len(self.image_ids):
                self.indices["image"] = AnnIndex(self.image_z, ann, seed)

    def _rankers(self, use_ann: bool) -> dict[str, Callable[[np.ndarray, int], list[str]]]:
        if use_ann:
            def target(vector, k):
                return ann_order(self.indices["target"], vector, k, self.target_ids)

            def text(vector, k):
                return ann_order(self.indices["text"], vector, k, self.text_ids)

            def image(vector, k):
                return ann_order(self.indices["image"], vector, k, self.image_ids)
        else:
            def target(vector, k):
                order, _ = exact_topk(vector[None, :], self.target_z, k, query_batch=1,
                                      corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
                return [self.target_ids[i] for i in order[0]]

            def text(vector, k):
                order, _ = exact_topk(vector[None, :], self.text_z, k, query_batch=1,
                                      corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
                return [self.text_ids[i] for i in order[0]]

            def image(vector, k):
                order, _ = exact_topk(vector[None, :], self.image_z, k, query_batch=1,
                                      corpus_chunk=int(self.retrieval["exact_corpus_chunk"]))
                return [self.image_ids[i] for i in order[0]]
        return {"target": target, "text": text, "image": image}

    def _z(self, bank: ObjectBank, query_id: str) -> np.ndarray:
        return _unit(bank.z[bank.position[query_id]][None, :])[0]

    def pipeline_both(
        self, bank: ObjectBank, query_id: str,
        paths: tuple[tuple[str, bool], ...] | None = None,
    ) -> dict[str, Any]:
        from .retrieve import retrieve_query

        per = int(self.retrieval["evidence_per_modality"])
        outputs: dict[str, Any] = {}
        for label, use_ann in (paths if paths is not None else (("ann", True), ("exact", False))):
            if use_ann and not self.ann_enabled:
                continue
            rankers = self._rankers(use_ann)
            query_vector = self._z(bank, query_id)
            # Each retriever path first derives its own natural bundle from the
            # frozen query vector.  The second hop must condition on the z of the
            # bundle entry that is actually at that position -- reusing the query
            # vector here would silently collapse RAW-2H into ``Top20_T(z_Q)``.
            b_q = interleave_text_image(
                rankers["text"](query_vector, per),
                rankers["image"](query_vector, per),
                per,
                per * 2,
            )
            evidence_z = {value: self._z(bank, value) for value in b_q}
            outputs[label] = retrieve_query(
                query_id=query_id,
                rank_target=rankers["target"],
                rank_text=rankers["text"],
                rank_image=rankers["image"],
                condition=lambda position: evidence_z[b_q[position]],
                u_D=query_vector,
                u_E=query_vector,
                retrieval=self.retrieval,
                bundle=b_q,
            )
        return outputs

    def direct_scores(self, bank: ObjectBank, query_id: str, candidates: Sequence[str]) -> dict[str, float]:
        vector = self._z(bank, query_id)
        position = {value: i for i, value in enumerate(self.target_ids)}
        keys = self.target_z[[position[c] for c in candidates]]
        return dict(zip(candidates, (keys @ vector).tolist()))

"""Stage-1 handoff: the frozen ranking and natural evidence bags Stage 2 starts from.

``run_stage1.py export`` writes ``retrieval.<split>.jsonl``: per query the Stage-1 order with its
table scores and, per target, the retained evidence paths (at most four). Stage 2 works on the top
``candidate_scope`` (C30) only; ranks C30+1..``output_depth`` are appended unchanged to every final
ranking so deep cutoffs stay comparable with Stage 1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import iter_jsonl


def load_stage1(handoff: Path, split: str, scope: int, depth: int) -> dict[str, dict[str, Any]]:
    queries = {}
    for record in iter_jsonl(Path(handoff) / f"retrieval.{split}.jsonl"):
        results = record["results"][:depth]
        if len(results) < depth:
            raise ValueError(f"{record['query_id']}: handoff holds {len(results)} < {depth} targets")
        ranking = [str(r["target_id"]) for r in results]
        scores = [float(r["score"]) for r in results]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"{record['query_id']}: Stage-1 ranking is not sorted by score")
        queries[str(record["query_id"])] = {
            "ranking": ranking,  # Stage-1 C<depth> order
            "candidates": ranking[:scope],  # C30: the only targets Stage 2 reads or scores
            "table_logits": {t: s for t, s in zip(ranking[:scope], scores)},
            "evidence": {str(r["target_id"]): [str(p["evidence_id"]) for p in r["paths"] if p["kind"] == "evidence"]
                         for r in results[:scope]},
        }
    return queries

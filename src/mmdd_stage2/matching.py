"""D-0.98 cell matching and the recovered-bridge table score.

A query value matches a target column when its typed key occurs there, or - TEXT only - when a
TEXT value with the same ordered numeric tokens has MiniLM cosine >= 0.98 (frozen
all-MiniLM-L6-v2, masked mean pooling, L2-normalized). NUMBER/DATE/ID/URL/SYMBOL stay exact.

Bridge score of a target = max over (bridge, same-attribute column) of matched recovered rows / 5.
Rows are counted, not distinct values: a value recovered for three rows votes three times, and one
target value may answer several rows. CONFLICT/MISSING rows score zero against the fixed five.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import iter_jsonl, write_jsonl
from .values import ROWS


class TextVectors:
    """MiniLM vectors for normalized cell texts, cached in ``<run>/matching`` and extended on demand."""

    def __init__(self, folder: Path, model_dir: Path, matching: dict[str, Any], cpu_threads: int) -> None:
        self.folder, self.model_dir, self.matching, self.cpu_threads = Path(folder), Path(model_dir), matching, cpu_threads
        self.vectors: dict[str, np.ndarray] = {}
        if (self.folder / "vectors.npy").exists():
            array = np.load(self.folder / "vectors.npy")
            texts = [row["text"] for row in iter_jsonl(self.folder / "texts.jsonl")]
            self.vectors = dict(zip(texts, array))

    def __getitem__(self, text: str) -> np.ndarray:
        return self.vectors[text]

    def ensure(self, texts: Iterable[str]) -> None:
        missing = sorted(set(texts) - set(self.vectors))
        if not missing:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(self.cpu_threads)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(self.model_dir, local_files_only=True, dtype=torch.float32).eval()
        size, chunks = self.matching["batch_size"], []
        with torch.inference_mode():
            for start in range(0, len(missing), size):
                batch = tokenizer(missing[start:start + size], padding=True, truncation=True,
                                  max_length=self.matching["max_length"], return_tensors="pt")
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        self.vectors.update(zip(missing, np.concatenate(chunks).astype(np.float32)))
        texts = list(self.vectors)
        self.folder.mkdir(parents=True, exist_ok=True)
        np.save(self.folder / "vectors.tmp.npy", np.stack([self.vectors[t] for t in texts]))
        write_jsonl(self.folder / "texts.jsonl", ({"text": t} for t in texts))
        (self.folder / "vectors.tmp.npy").replace(self.folder / "vectors.npy")
        print(json.dumps({"minilm_new_texts": len(missing), "minilm_total": len(texts)}), flush=True)


class Matcher:
    """``match(x, column_key, values)``; per-column lookups are built once and reused.

    TEXT values are stacked per numeric-token signature only when a query value with that
    signature asks, so only ``comparable_texts`` ever need a vector.
    """

    def __init__(self, vectors: TextVectors, tau: float) -> None:
        self.vectors, self.tau = vectors, tau
        self.columns: dict[Any, tuple[set[str], dict[tuple[str, ...], list[str]]]] = {}
        self.stacks: dict[tuple[Any, tuple[str, ...]], np.ndarray] = {}

    def match(self, x: dict[str, Any], column_key: Any, values: list[dict[str, Any]]) -> bool:
        if column_key not in self.columns:
            groups: dict[tuple[str, ...], list[str]] = {}
            for value in values:
                if value["kind"] == "TEXT":
                    groups.setdefault(tuple(value["digits"]), []).append(value["text"])
            self.columns[column_key] = ({v["key"] for v in values}, groups)
        keys, groups = self.columns[column_key]
        if x["key"] in keys:
            return True
        signature = tuple(x["digits"])
        if x["kind"] != "TEXT" or signature not in groups:
            return False
        if (column_key, signature) not in self.stacks:
            self.stacks[column_key, signature] = np.stack([self.vectors[t] for t in groups[signature]])
        return float(np.max(self.stacks[column_key, signature] @ self.vectors[x["text"]])) >= self.tau

    def clear(self) -> None:
        self.columns.clear()
        self.stacks.clear()


def comparable_texts(x_values: Iterable[dict[str, Any]], y_values: Iterable[dict[str, Any]]) -> set[str]:
    """TEXT strings the matcher may embed when comparing ``x_values`` against ``y_values``."""
    xs = [x for x in x_values if x["kind"] == "TEXT"]
    signatures = {tuple(x["digits"]) for x in xs}
    if not xs:
        return set()
    return {x["text"] for x in xs} | {y["text"] for y in y_values
                                      if y["kind"] == "TEXT" and tuple(y["digits"]) in signatures}


def bridge_texts(bridges: list[dict[str, Any]], tables: dict[str, dict[str, Any]]) -> set[str]:
    texts: set[str] = set()
    for bridge in bridges:
        for table in tables.values():
            for column in table["columns"]:
                if column["attribute"] == bridge["attribute"]:
                    texts |= comparable_texts(bridge["domain"], column["values"])
    return texts


def bridge_scores(candidates: list[str], bridges: list[dict[str, Any]], tables: dict[str, dict[str, Any]],
                  matcher: Matcher) -> tuple[dict[str, float], dict[str, dict[str, Any] | None]]:
    """Per-target bridge score and its winning (attribute, column)."""
    scores = {t: 0.0 for t in candidates}
    winners: dict[str, dict[str, Any] | None] = {t: None for t in candidates}
    for target in candidates:
        options = []
        for bridge in bridges:
            domain = {v["key"]: v for v in bridge["domain"]}
            rows = [s for s in bridge["slots"] if s["status"] == "VALUE"]
            for column in tables[target]["columns"]:
                if not rows or column["attribute"] != bridge["attribute"] or not column["values"]:
                    continue
                matched = sum(matcher.match(domain[s["value_key"]], (target, column["column_id"]), column["values"])
                              for s in rows)
                options.append((-round(matched / ROWS, 8), bridge["attribute"], column["column_id"], matched))
        if options:
            best = min(options)
            scores[target] = best[3] / ROWS
            winners[target] = {"attribute": best[1], "column_id": best[2], "matched_rows": best[3]}
    return scores, winners


def rrf_fuse(stage1: list[str], reranked: list[str], constant: int) -> list[str]:
    """``1/(k + stage1 rank) + 1/(k + reranked rank)``; ties keep Stage-1 order."""
    first = {t: i + 1 for i, t in enumerate(stage1)}
    second = {t: i + 1 for i, t in enumerate(reranked)}
    fused = {t: 1 / (constant + first[t]) + 1 / (constant + second[t]) for t in stage1}
    return sorted(stage1, key=lambda t: (-round(fused[t], 12), first[t], t))


def bridge_order(candidates: list[str], scores: dict[str, float]) -> list[str]:
    """PURE bridge order: score, then Stage-1 rank."""
    rank = {t: i for i, t in enumerate(candidates)}
    return sorted(candidates, key=lambda t: (-round(scores[t], 8), rank[t], t))

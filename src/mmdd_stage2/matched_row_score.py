"""Cell matching and bridge row scoring for the recovery-scored join pipeline.

This is the merged, renamed form of the R7 fresh-AB package's ``d_matching.py`` (matched
original rows / 5 at a fixed unit-cosine threshold) together with the typed value-key rules it
depended on. The rules are frozen: typed keys compare exactly, and only TEXT-to-TEXT pairs whose
ordered numeric-token signature is identical may fall back to a unit-normalized cosine at
``TAU_COSINE``. Nothing is fuzzy, substring, stemmed or alias-based.

Scoring counts *original query rows*: a value recovered for several rows casts one vote per row, a
repeated target value may answer several rows, and NULL/MISSING/CONFLICT rows contribute zero
while the denominator stays at five. No distinct-value deduplication and no extra gamma factor.
"""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

TAU_COSINE = 0.98
DENOMINATOR_ROWS = 5
MATCH_KIND_EXACT = "TYPED_EXACT"
MATCH_KIND_SEMANTIC = "MINILM_TEXT_DIGITS_GUARDED"
MATCH_KIND_NONE = "NO_COMPATIBLE_VALUE"

NULLS = frozenset(["", "-", "--", "–", "—", "null", "none", "nan", "n/a"])
DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-"})
NUMBER = re.compile(r"^[+-]?(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)(?:\.\d+)?$")
DIGITS = re.compile(r"\d+(?:\.\d+)?")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def text_norm(value: str) -> str:
    """NFKC, dash unification, whitespace collapse and casefold — the shared text normal form."""
    if not isinstance(value, str):
        raise ValueError(f"non-string value: {type(value).__name__}")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).translate(DASHES)).strip().casefold()


def header_key(name: str) -> str:
    """Attribute key for a column header: same normal form, no stemming or alias expansion."""
    return text_norm(name)


def decimal_key(text: str) -> str:
    value = Decimal(text.replace(",", ""))
    return "0" if value == 0 else format(value.normalize(), "f")


def value_info(raw: str) -> dict[str, Any]:
    """Typed description of one cell value: ``key``, display ``text``, ``kind`` and digit list.

    ``key`` is ``None`` exactly for empty/null cells, so a falsy key marks a row with no
    recoverable value rather than a matchable empty string.
    """
    n = text_norm(raw)
    if n in NULLS:
        return {"key": None, "text": n, "kind": "EMPTY", "digits": []}
    if NUMBER.fullmatch(n):
        k = decimal_key(n)
        return {"key": "NUMBER:" + k, "text": k, "kind": "NUMBER", "digits": [k]}
    if n.endswith("%") and NUMBER.fullmatch(n[:-1]):
        k = decimal_key(n[:-1])
        return {"key": "PERCENT:" + k, "text": k + "%", "kind": "PERCENT", "digits": [k]}
    if ISO_DATE.fullmatch(n):
        try:
            valid = dt.date.fromisoformat(n)
        except ValueError:
            pass
        else:
            return {"key": "DATE:" + valid.isoformat(), "text": valid.isoformat(), "kind": "DATE", "digits": []}
    if n.startswith(("https://", "http://")):
        return {"key": "URL:" + n, "text": n, "kind": "URL", "digits": []}
    if re.fullmatch(r"\\[a-z]", n) or re.fullmatch(r"u\+[0-9a-f]{4,6}", n) or not any(c.isalnum() for c in n):
        return {"key": "SYMBOL:" + n, "text": n, "kind": "SYMBOL", "digits": []}
    # Numeric identifiers keep leading zeros and stay opaque exact-only instead of collapsing to 1.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", n):
        return {"key": "ID:" + n, "text": n, "kind": "ID", "digits": []}
    return {"key": "TEXT:" + n, "text": n, "kind": "TEXT", "digits": [decimal_key(x) for x in DIGITS.findall(n)]}


def semantic_allowed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Only TEXT/TEXT pairs sharing the same ordered numeric tokens may be softened."""
    return left["kind"] == "TEXT" and right["kind"] == "TEXT" and left["digits"] == right["digits"]


class TextVectorStore(Protocol):
    """Unit-normalized TEXT embeddings addressed by normalized text."""

    def vector(self, text: str) -> np.ndarray: ...

    def matrix(self, texts: Sequence[str]) -> np.ndarray: ...


class DictVectorStore:
    """In-memory vector store; used by tests and by callers that already hold a cache."""

    def __init__(self, vectors: Mapping[str, np.ndarray]) -> None:
        self.vectors = {text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()}

    def vector(self, text: str) -> np.ndarray:
        return self.vectors[text]

    def matrix(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.vectors[text] for text in texts]) if texts else np.empty((0, 0), dtype=np.float32)


class UnitCosineMatcher:
    """Typed exact match first, then guarded unit-cosine match above ``tau``.

    Exact typed hits never consult embeddings, so non-TEXT and URL/SYMBOL cells stay exact-only.
    """

    def __init__(self, store: TextVectorStore, *, tau: float = TAU_COSINE, chunk: int = 4096) -> None:
        self.store = store
        self.tau = tau
        self.chunk = chunk
        self.cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self.calls = 0

    def best(self, cell: dict[str, Any], values: Sequence[dict[str, Any]]) -> dict[str, Any]:
        by_key = {value["key"]: value for value in values}
        if cell["key"] in by_key:
            hit = by_key[cell["key"]]
            return {"matched": True, "cosine": 1.0, "target_key": hit["key"], "target_text": hit["text"],
                    "match_kind": MATCH_KIND_EXACT}
        allowed = sorted((value for value in values if semantic_allowed(cell, value)), key=lambda value: value["key"])
        if not allowed:
            return {"matched": False, "cosine": None, "target_key": None, "target_text": None,
                    "match_kind": MATCH_KIND_NONE}
        signature = (cell["key"], tuple(value["key"] for value in allowed))
        cached = self.cache.get(signature)
        if cached is not None:
            return cached
        query = self.store.vector(cell["text"])
        best, winner = -np.inf, None
        for start in range(0, len(allowed), self.chunk):
            window = allowed[start:start + self.chunk]
            cosine = self.store.matrix([value["text"] for value in window]) @ query
            self.calls += 1
            index = int(np.argmax(cosine))
            score = float(cosine[index])
            if score > best:
                best, winner = score, window[index]
        if not np.isfinite(best):
            raise ValueError("non-finite cosine")
        result = {"matched": bool(best >= self.tau), "cosine": best, "target_key": winner["key"],
                  "target_text": winner["text"], "match_kind": MATCH_KIND_SEMANTIC}
        self.cache[signature] = result
        return result


def bridge_row_scores(
    base: Sequence[str],
    bridges: Sequence[dict[str, Any]],
    tables: Mapping[str, dict[str, Any]],
    matcher: UnitCosineMatcher,
) -> dict[str, Any]:
    """Row-count score per candidate target: max over (bridge, compatible column) of matched/5.

    ``base`` is the scored candidate scope. Each bridge must carry five unique row slots; only
    VALUE slots vote, and every distinct recovered key is matched once and then charged to each
    of its rows, so a repeated recovered value keeps its row multiplicity.
    """
    scores = {target: 0.0 for target in base}
    winners: dict[str, dict[str, Any] | None] = {target: None for target in base}
    details: list[dict[str, Any]] = []
    for target in base:
        options = []
        for bridge in bridges:
            _validate_bridge(bridge)
            domain = {value["key"]: value for value in bridge["domain"]}
            valid = [slot for slot in bridge["slots"] if slot["status"] == "VALUE"]
            if not valid:
                continue
            for column in tables[target]["columns"]:
                if header_key(column["column_name"]) != bridge["attribute"] or not column["values"]:
                    continue
                hits = {key: matcher.best(domain[key], column["values"]) for key in sorted({s["value_key"] for s in valid})}
                checks = [{"row_id": slot["row_id"], "recovered_key": slot["value_key"], **hits[slot["value_key"]]}
                          for slot in sorted(valid, key=lambda slot: slot["row_id"])]
                matched_rows = sum(check["matched"] for check in checks)
                options.append({
                    "target_id": target,
                    "bridge_attribute": bridge["attribute"],
                    "column_id": column["column_id"],
                    "score": matched_rows / DENOMINATOR_ROWS,
                    "matched_rows": matched_rows,
                    "denominator_rows": DENOMINATOR_ROWS,
                    "recovered_nonconflicting_rows_used": len(valid),
                    "row_checks": checks,
                })
        options.sort(key=lambda option: (-round(option["score"], 8), option["bridge_attribute"],
                                         option["column_id"]))
        details.extend(options)
        if options:
            winner = options[0]
            scores[target] = winner["score"]
            winners[target] = {key: winner[key] for key in ("bridge_attribute", "column_id", "score")}
    return {"table_scores": scores, "winning_columns": winners, "details": details}


def _validate_bridge(bridge: dict[str, Any]) -> None:
    slots = bridge["slots"]
    if len(slots) != DENOMINATOR_ROWS or {slot["row_id"] for slot in slots} != set(range(DENOMINATOR_ROWS)):
        raise ValueError("bridge must hold exactly five unique row slots")


def reference_bridge_row_scores(
    base: Sequence[str],
    bridges: Sequence[dict[str, Any]],
    tables: Mapping[str, dict[str, Any]],
    store: TextVectorStore,
    *,
    tau: float = TAU_COSINE,
    chunk: int = 8192,
) -> dict[str, float]:
    """Independent row-occurrence recomputation used to cross-check :func:`bridge_row_scores`.

    Deliberately shares no helper with the scoring path other than the frozen key rules: it walks
    the five slots one by one, re-derives every candidate window and never groups by value key.
    """
    result = {target: 0.0 for target in base}
    for target in base:
        for bridge in bridges:
            domain = {value["key"]: value for value in bridge["domain"]}
            for column in tables[target]["columns"]:
                if header_key(column["column_name"]) != bridge["attribute"]:
                    continue
                count = 0
                for slot in bridge["slots"]:
                    if slot["status"] != "VALUE":
                        continue
                    cell = domain[slot["value_key"]]
                    if any(value["key"] == cell["key"] for value in column["values"]):
                        count += 1
                        continue
                    if cell["kind"] != "TEXT":
                        continue
                    compatible = [value for value in column["values"]
                                  if value["kind"] == "TEXT" and value["digits"] == cell["digits"]]
                    found = False
                    for start in range(0, len(compatible), chunk):
                        window = compatible[start:start + chunk]
                        if not window:
                            break
                        cosine = store.matrix([value["text"] for value in window]) @ store.vector(cell["text"])
                        if float(np.max(cosine)) >= tau:
                            found = True
                            break
                    count += int(found)
                result[target] = max(result[target], count / DENOMINATOR_ROWS)
    return result

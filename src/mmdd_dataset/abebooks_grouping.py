"""Form equally sized book source tables from bibliographic content, without labels."""
from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)


def _text_features(titles: list[str]) -> np.ndarray:
    documents = []
    for title in titles:
        words = re.findall(r"[a-z]{2,}", title.lower())
        words = [w for w in words if w not in {"a", "an", "the", "of", "to", "for", "and", "in", "with", "on"}]
        documents.append(Counter([*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]))
    frequencies = Counter(term for document in documents for term in document)
    vocabulary = {term: i for i, term in enumerate(sorted(frequencies))}
    features = np.zeros((len(titles), len(vocabulary)), dtype=np.float32)
    for i, document in enumerate(documents):
        for term, count in document.items():
            features[i, vocabulary[term]] = (1 + np.log(count)) * (1 + np.log((1 + len(titles)) / (1 + frequencies[term])))
    return _normalize(features)


def partition_books_by_publisher(tables: list[dict], min_rows: int = 1) -> tuple[list[dict], dict, dict]:
    """Use one source table per exact publisher value, preserving all book rows."""
    books = [t for t in tables if any(c["column_name"] == "title" for c in t["columns"])]
    book_ids = {t["source_table_id"] for t in books}
    groups = defaultdict(list)
    for table in books:
        for row in table["rows"]:
            publisher = next(c["text"] for c in row["cells"] if c["column_name"] == "publisher")
            groups[publisher].append((table["source_table_id"], row))
    output = [copy.deepcopy(t) for t in tables if t["source_table_id"] not in book_ids]
    mapping = {(t["source_table_id"], r["row_id"]): t["source_table_id"]
               for t in output for r in t["rows"]}
    small = sorted(p for p, rows in groups.items() if len(rows) < min_rows)
    small_rows = [entry for p in small for entry in groups.pop(p)]
    group_items = [("st_book_publisher_" + hashlib.sha256(p.encode()).hexdigest()[:12], rows)
                   for p, rows in sorted(groups.items())]
    if small_rows:
        sid = "st_book_publisher_tail_" + hashlib.sha256(repr(small).encode()).hexdigest()[:12]
        group_items.append((sid, small_rows))
    for sid, entries in group_items:
        entries.sort(key=lambda x: (x[0], x[1]["row_id"]))
        table = copy.deepcopy(books[0])
        table.update(source_table_id=sid, rows=[], num_rows=len(entries))
        for old_sid, original in entries:
            row = copy.deepcopy(original)
            row["source_grouping_origin"] = {"source_table_id": old_sid, "row_id": row["row_id"]}
            table["rows"].append(row)
            mapping[old_sid, row["row_id"]] = sid
        output.append(table)
    output.sort(key=lambda t: t["source_table_id"])
    report = {"rule": "One source table per exact original publisher value; no value normalization",
        "book_rows": sum(len(rows) for _, rows in group_items), "book_tables_before": len(books),
        "book_tables_after": len(group_items), "sizes_before": sorted(len(t["rows"]) for t in books),
        "sizes_after": sorted(len(rows) for _, rows in group_items), "labels_used_for_grouping": False,
        "rows_added": 0, "rows_removed": 0}
    if min_rows > 1:
        report.update(rule=report["rule"] + f"; publisher groups with fewer than {min_rows} rows share one tail table",
                      min_publisher_rows=min_rows, coalesced_publishers=len(small), coalesced_rows=len(small_rows))
    return output, mapping, report


def group_book_sources(tables: list[dict], seed: int = 13, *, field: str = "title",
                       title_embeddings: Path | None = None) -> tuple[list[dict], dict, dict]:
    """Balance bibliographic TF-IDF clusters to original sizes; keep every row.

    Five capacity-constrained assignment/centroid updates are fixed in advance.
    The inputs contain source cells only, never annotations or retrieval scores.
    """
    books = [t for t in tables if any(c["column_name"] == "title" for c in t["columns"])]
    entries = [(t["source_table_id"], r) for t in books for r in t["rows"]]
    entries.sort(key=lambda x: (x[0], x[1]["row_id"]))
    texts = [next(c["text"] for c in r["cells"] if c["column_name"] == field) for _, r in entries]
    feature_rule = f"{field}-only word/bigram TF-IDF"
    feature_audit = {}
    if title_embeddings is None:
        features = _text_features(texts)
    else:
        with np.load(title_embeddings, allow_pickle=False) as cached:
            keys = list(zip(cached["source_ids"].tolist(), cached["row_ids"].tolist()))
            positions = {key: i for i, key in enumerate(keys)}
            expected = [(sid, row["row_id"]) for sid, row in entries]
            if len(positions) != len(keys) or set(positions) != set(expected):
                raise ValueError("Title embeddings must cover every book source row exactly once")
            order = [positions[key] for key in expected]
            if cached["titles"][order].tolist() != texts:
                raise ValueError("Title embedding inputs differ from source titles")
            features = cached["vectors"][order]
            if features.ndim != 2 or not np.isfinite(features).all():
                raise ValueError("Invalid title embedding vectors")
        features = _normalize(features)
        feature_rule = "Frozen Qwen normalized-title embeddings"
        feature_audit = {"title_embeddings": str(title_embeddings),
                         "title_embeddings_sha256": hashlib.sha256(title_embeddings.read_bytes()).hexdigest()}
        field = "semantic_title"
    sizes = [len(t["rows"]) for t in books]
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(len(entries)))]
    distances = np.ones(len(entries))
    for _ in range(1, len(books)):
        distances = np.minimum(distances, np.maximum(0, 1 - features @ features[seeds[-1]]))
        distances[seeds] = 0
        if distances.sum() > 0:
            seeds.append(int(rng.choice(len(entries), p=distances / distances.sum())))
        else:
            seeds.append(next(i for i in range(len(entries)) if i not in seeds))
    centers = features[seeds].copy()
    for _ in range(5):
        scores = features @ centers.T
        assignment = np.full(len(entries), -1, dtype=int)
        remaining = np.array(sizes)
        for flat in np.argsort(-scores.ravel(), kind="stable"):
            row, group = divmod(int(flat), len(books))
            if assignment[row] == -1 and remaining[group] > 0:
                assignment[row] = group
                remaining[group] -= 1
        centers = _normalize(np.stack([features[assignment == i].mean(axis=0) for i in range(len(books))]))

    def coherence(groups: list[list[int]]) -> float:
        values = []
        for positions in groups:
            n = len(positions)
            if n > 1:
                block = features[positions]
                values.append(float(((block @ block.T).sum() - (block * block).sum()) / (n * (n - 1))))
        return float(np.mean(values))

    old_groups = defaultdict(list)
    for i, (sid, _) in enumerate(entries):
        old_groups[sid].append(i)
    groups = [np.flatnonzero(assignment == i).tolist() for i in range(len(books))]
    book_ids = {t["source_table_id"] for t in books}
    output = [copy.deepcopy(t) for t in tables if t["source_table_id"] not in book_ids]
    mapping = {(t["source_table_id"], r["row_id"]): t["source_table_id"]
               for t in output for r in t["rows"]}
    for positions in groups:
        keys = [(entries[i][0], entries[i][1]["row_id"]) for i in positions]
        digest = hashlib.sha256(repr(sorted(keys)).encode()).hexdigest()[:12]
        sid = f"st_book_{field}_{digest}"
        table = copy.deepcopy(books[0])
        table.update(source_table_id=sid, rows=[copy.deepcopy(entries[i][1]) for i in positions], num_rows=len(positions))
        for row, key in zip(table["rows"], keys):
            row["source_grouping_origin"] = {"source_table_id": key[0], "row_id": key[1]}
            mapping[key] = sid
        output.append(table)
    output.sort(key=lambda t: t["source_table_id"])
    report = {"rule": f"{feature_rule}; cosine-distance seeding plus five capacity-constrained greedy assignments",
              "seed": seed, "book_rows": len(entries), "book_tables": len(books),
              "sizes_before": sorted(sizes), "sizes_after": sorted(map(len, groups)),
              f"{field}_cosine_before": coherence(list(old_groups.values())),
              f"{field}_cosine_after": coherence(groups), "labels_used_for_grouping": False,
              "rows_added": 0, "rows_removed": 0, **feature_audit}
    return output, mapping, report

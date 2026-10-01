"""Evidence qualification and independent-query construction for AbeBooks."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .abebooks_ablation import read_rows, write_rows
from .abebooks_curation import cell_values, dataset_hashes, file_hash, load_artifacts
from .abebooks_publisher import publisher_key, visible_publisher_hint


POLICY = "abebooks_standalone_equal_rows_v3"


def stable_id(prefix: str, *parts: object) -> str:
    return prefix + "_" + hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:20]


def text_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("’", "'").replace("'", "")
    return " ".join(re.findall(r"[^\W_]+", text))


def author_names(value: str | list[str]) -> tuple[str, ...]:
    """Parse explicit name lists; preserve initials and reject uncertain delimiters."""
    if isinstance(value, list):
        parsed = [author_names(part) for part in value]
        names = [n for part in parsed for n in part]
        return tuple(sorted(names)) if all(parsed) and len(names) == len(set(names)) else ()
    value = unicodedata.normalize("NFKC", str(value)).strip().rstrip(":")
    if re.search(r"\b(editor|edited|eds|professor|et al)\b", value, re.I):
        return ()
    names = []
    for member in re.split(r"\s*;\s*|\s+(?:and|et)\s+|\s*&\s*", value):
        parts = [p.strip() for p in member.split(",")]
        if not all(parts):
            return ()
        if len(parts) > 1:
            if len(parts) % 2 == 0 and all(len(text_key(p).split()) == 1 for p in parts[::2]):
                parts = [f"{parts[i + 1]} {parts[i]}" for i in range(0, len(parts), 2)]
            elif not all(len(text_key(p).split()) >= 2 for p in parts):
                return ()
        for name in parts:
            key = text_key(name)
            if len(key.split()) < 2 or any(c.isdigit() for c in key):
                return ()
            names.append(key)
    if len(names) != len(set(names)):
        return ()
    return tuple(sorted(names))


@lru_cache(maxsize=32768)
def matching_key(value: str) -> str:
    names = author_names(value)
    return " | ".join(names) if names else ""


def join_key(value: str, column: str) -> str:
    return publisher_key(value) if column == "publisher" else matching_key(value)


def visible_join_hint(title: str, value: str, column: str) -> bool:
    return (visible_publisher_hint(title, value) if column == "publisher"
            else visible_author_hint(title, author_names(value)))


def cover_title_matches(observed: str, source: str) -> bool:
    stop = {"the", "a", "an", "of", "for", "and", "to", "with", "in", "on"}
    observed_words = set(text_key(observed).split()) - stop
    source_words = set(text_key(source).split()) - stop
    overlap = observed_words & source_words
    return bool(observed_words) and len(overlap) >= min(2, len(observed_words)) and len(overlap) / len(observed_words) >= 0.65


def title_family_key(title: str) -> str:
    """Keep editions of an otherwise identical title together."""
    title = re.sub(r"\([^)]*(?:edition|\bed\.?\b|paperback|hardcover)[^)]*\)", "", title, flags=re.I)
    title = re.sub(r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+edition\b", "", title, flags=re.I)
    return text_key(title)


def visible_author_hint(title: str, names: tuple[str, ...]) -> bool:
    title = re.sub(r"['’]s\b", "", title, flags=re.I)
    visible = f" {text_key(title)} "
    return any(f" {name.split()[-1]} " in visible for name in names)


def collect_facts(dataset: Path, proposals: Path, historical_reviews: Path) -> tuple[list[dict], list[dict]]:
    """Full-list agreement only. New local output is never called human ground truth."""
    _, data = load_artifacts(dataset)
    sources = {(s["source_table_id"], r["row_id"]): cell_values(r)
               for s in data["source_tables"] for r in s["rows"]}
    assets = {a["asset_id"]: a for a in data["bridge_assets"]}
    history = defaultdict(set)
    for r in read_rows(historical_reviews):
        if r["attribute_name"] != "authors":
            continue
        aid = r["evidence_identity"].get("asset_id")
        for review in r["auto_check"].get("reviews", []):
            names = author_names(review.get("extracted_value") or "")
            if names and review.get("review_complete"):
                history[aid].add(names)
    audit = []
    for proposal in read_rows(proposals):
        loc = (proposal["source_table_id"], proposal["source_row_id"])
        source = sources[loc]
        response = proposal.get("response") or {}
        names = author_names(response.get("authors", []))
        source_names = author_names(source.get("authors", ""))
        reason = "full_author_list_matches_source"
        if not source_names:
            reason = "source_author_list_unresolved"
        elif not names:
            reason = "no_readable_full_author_list"
        elif names != source_names:
            reason = "author_list_differs_from_source"
        elif response.get("editors"):
            reason = "mixed_author_editor_roles_require_review"
        elif proposal["modality"] == "image" and not cover_title_matches(response.get("title", ""), source.get("title", "")):
            reason = "cover_title_not_verified"
        elif proposal["modality"] == "text" and not all(
                f" {name} " in f" {text_key(assets[proposal['asset_id']]['content'])} " for name in names):
            reason = "author_name_not_literal_in_text"
        audit.append({**proposal, "source_author_value": source.get("authors", ""),
                      "source_title": source.get("title", ""), "normalized_authors": list(names),
                      "status": reason, "qualified": reason == "full_author_list_matches_source",
                      "historical_reader_agrees": names in history[proposal["asset_id"]]})
    by_row = defaultdict(list)
    for r in audit:
        if r["qualified"]:
            by_row[r["source_table_id"], r["source_row_id"]].append(r)
    facts = []
    for (sid, rid), records in sorted(by_row.items()):
        # Cover title supports the book-to-name relationship. Biography-only facts
        # remain candidates until an independent entity-link review is available.
        images = [r for r in records if r["modality"] == "image"]
        if not images:
            continue
        modalities = {r["modality"] for r in records}
        strength = "local_cover_with_source_agreement"
        if any(r["historical_reader_agrees"] for r in records):
            strength = "historical_and_local_reader_agreement"
        elif len(modalities) == 2:
            strength = "cover_and_biography_agreement"
        facts.append({"source_table_id": sid, "source_row_id": rid,
                      "original_value": sources[sid, rid]["authors"],
                      "normalized_authors": list(author_names(sources[sid, rid]["authors"])),
                      "evidence_ids": [r["asset_id"] for r in records], "strength": strength,
                      "annotation_status": "model_assisted_full_value_evidence", "human_reviewed": False})
    return facts, audit


def write_fact_audit(dataset: Path, proposals: Path, historical_reviews: Path, output: Path) -> dict:
    facts, audit = collect_facts(dataset, proposals, historical_reviews)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "author_facts.jsonl", facts)
    write_rows(output / "author_evidence_audit.jsonl", audit)
    report = {"facts": len(facts), "source_groups": len({f["source_table_id"] for f in facts}),
              "strengths": dict(Counter(f["strength"] for f in facts)),
              "evidence_statuses": dict(Counter(r["status"] for r in audit)),
              "proposals_read": len(audit), "source_answers_supplied_to_reader": False}
    (output / "FACT_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def project_rows(source: dict, row_ids: list[int], names: list[str]) -> tuple[list[dict], list[dict]]:
    columns = [{"column_index": i, "source_column_index": c["column_index"], "column_name": name}
               for i, name in enumerate(names) for c in source["columns"] if c["column_name"] == name]
    original = {r["row_id"]: r for r in source["rows"]}
    rows = []
    for i, rid in enumerate(row_ids):
        cells = {c["column_name"]: c for c in original[rid]["cells"]}
        rows.append({"row_id": i, "source_row_id": rid,
                     "cells": [{**copy.deepcopy(cells[c["column_name"]]), **c} for c in columns]})
    return columns, rows


def complementary_targets(data: dict, namespace: str) -> tuple[list[dict], list[dict]]:
    """Keep all candidate memberships; expose authors and withhold book titles uniformly."""
    sources = {s["source_table_id"]: s for s in data["source_tables"]}
    targets, changes = [], []
    for original in data["data_lake_tables"]:
        target = copy.deepcopy(original)
        source = sources[target["source_table_id"]]
        before = [c["column_name"] for c in target["columns"]]
        after = before
        if source["source_file"] == "book":
            after = [n for n in before if n != "title"]
            if "authors" not in after:
                after.append("authors")
            target["columns"], target["rows"] = project_rows(
                source, [r["source_row_id"] for r in original["rows"]], after)
            target["source_column_indices"] = [c["source_column_index"] for c in target["columns"]]
            target["join_col"], target["join_col_name"] = 1, "authors"
            target["target_context_col_names"] = [n for n in after if n != "authors"]
        tid = stable_id("target", namespace, original["table_id"])
        target.update(table_id=tid, object_id=tid)
        target.pop("chain_id", None)
        target["provenance"] = {**target.get("provenance", {}), "policy": POLICY,
                                "original_table_id": original["table_id"]}
        changes.append({"original_table_id": original["table_id"], "table_id": tid,
                        "source_table_id": source["source_table_id"],
                        "before_columns": before, "after_columns": after,
                        "rows_preserved": [r["source_row_id"] for r in target["rows"]]
                                          == [r["source_row_id"] for r in original["rows"]]})
        targets.append(target)
    return targets, changes


def judge_join(query: dict, target: dict, source_rows: dict) -> dict:
    """Execute the declared attribute equality, then judge every returned pair."""
    column = query.get("join_column", "authors")
    added = {c["column_name"] for c in target["columns"]} - {c["column_name"] for c in query["columns"]} - {column}
    pairs = []
    for qr in query["rows"]:
        qloc = query["source_table_id"], qr["source_row_id"]
        qs = cell_values(source_rows[qloc])
        key = join_key(qs.get(column, ""), column)
        for tr in target["rows"]:
            values = cell_values(tr)
            if not key or key != join_key(values.get(column, ""), column):
                continue
            tloc = target["source_table_id"], tr["source_row_id"]
            ts = cell_values(source_rows[tloc])
            if qloc == tloc:
                status = "correct_record"
            elif text_key(qs.get("title", "")) == text_key(ts.get("title", "")):
                status = "unjudged_record_identity"
            else:
                status = "different_book"
            pairs.append({"query_row_id": qr["row_id"], "target_row_id": tr["row_id"],
                          "status": status, "added_values": {n: values[n] for n in sorted(added) if values.get(n)}})
    coverage = {p["query_row_id"] for p in pairs if p["added_values"]}
    if any(p["status"] == "different_book" for p in pairs):
        status, reason = "negative", "wrong_book_expansion"
    elif len(coverage) != len(query["rows"]):
        status, reason = "negative", "does_not_complete_every_query_row"
    elif any(p["status"] == "unjudged_record_identity" for p in pairs):
        status, reason = "unjudged", "duplicate_book_identity_requires_review"
    else:
        status, reason = "positive", "all_rows_completed_without_wrong_book_expansion"
    return {"query_table_id": query["table_id"], "target_table_id": target["table_id"],
            "status": status, "reason": reason, "row_pairs": pairs}


def select_disjoint_row_views(row_ids: list[int], supported: set[int], rows_per_query: int,
                              minimum_recovered_rows: int) -> list[list[int]]:
    """Reserve the recovery floor per view, then fill with unused source rows."""
    if rows_per_query < 2 or not 0 <= minimum_recovered_rows <= rows_per_query:
        raise ValueError("Invalid query size or recovery floor")
    recovered = [rid for rid in row_ids if rid in supported]
    count = len(row_ids) // rows_per_query
    if minimum_recovered_rows:
        count = min(count, len(recovered) // minimum_recovered_rows)
    views = [recovered[i * minimum_recovered_rows:(i + 1) * minimum_recovered_rows] for i in range(count)]
    reserved = {rid for view in views for rid in view}
    remaining = [rid for rid in recovered if rid not in reserved] + [rid for rid in row_ids if rid not in supported]
    offset = 0
    positions = {rid: i for i, rid in enumerate(row_ids)}
    for view in views:
        needed = rows_per_query - len(view)
        view.extend(remaining[offset:offset + needed])
        offset += needed
        view.sort(key=positions.get)
    return views


def make_queries(data: dict, targets: list[dict], facts: list[dict], namespace: str, *,
                 implicit_rows: int = 5, explicit_rows: int = 5,
                 minimum_recovered_rows: int = 2, join_column: str = "authors",
                 excluded_rows: set[tuple[str, int]] | None = None) -> tuple[list[dict], list[dict], list[dict]]:
    """Build disjoint queries, keeping row count separate from evidence coverage."""
    source_rows = {(s["source_table_id"], r["row_id"]): r for s in data["source_tables"] for r in s["rows"]}
    fact_map = {(f["source_table_id"], f["source_row_id"]): f for f in facts
                if f.get("column_name", "authors") == join_column}
    queries, judgments, decisions = [], [], []
    targets_by_source = defaultdict(list)
    for target in targets:
        targets_by_source[target["source_table_id"]].append(target)
    for source in data["source_tables"]:
        if source["source_file"] != "book":
            continue
        sid = source["source_table_id"]
        column_index = next(c["column_index"] for c in source["columns"] if c["column_name"] == join_column)
        eligible = []
        for row in source["rows"]:
            loc = sid, row["row_id"]
            if excluded_rows and loc in excluded_rows:
                continue
            v = cell_values(row)
            if not join_key(v.get(join_column, ""), join_column) or not v.get("title"):
                decisions.append({"source_table_id": sid, "row_id": row["row_id"], "reason": "invalid_title_or_join_value"})
                continue
            # Drop rows whose author key cannot identify their own record in any
            # target; do not waste an otherwise valid five-row view on such rows.
            probe = {"table_id": "qualification", "source_table_id": sid,
                     "join_column": join_column,
                     "columns": [{"column_name": "title"}, {"column_name": join_column}],
                     "rows": [{"row_id": 0, "source_row_id": row["row_id"]}]}
            if not any(judge_join(probe, t, source_rows)["status"] == "positive" for t in targets_by_source[sid]):
                decisions.append({"source_table_id": sid, "row_id": row["row_id"], "reason": "no_unambiguous_row_join"})
                continue
            # Check names, including surname hints, across the visible title.
            leak = visible_join_hint(v["title"], v[join_column], join_column)
            kind = "implicit" if not leak else "explicit"
            eligible.append((row["row_id"], kind))
        used = set()
        for kind in ("implicit", "explicit"):
            names = ["title"] if kind == "implicit" else ["title", join_column]
            pool = [rid for rid, k in eligible if rid not in used and (k == kind or kind == "explicit")]
            # One fixed ordering, no combinatorial row resampling.
            pool.sort(key=lambda rid: stable_id("row", 13, sid, rid))
            size = implicit_rows if kind == "implicit" else explicit_rows
            required = minimum_recovered_rows if kind == "implicit" else 0
            supported = {rid for rid in pool if (sid, rid) in fact_map} if kind == "implicit" else set()
            for view_index, selected in enumerate(select_disjoint_row_views(pool, supported, size, required)):
                recovered_rows = sum(rid in supported for rid in selected)
                qid = stable_id("query", namespace, sid, selected, kind, join_column)
                columns, rows = project_rows(source, selected, names)
                q = {"table_id": qid, "object_id": qid, "object_type": "table", "role": "query",
                     "source_table_id": sid, "page_title": "", "caption": "", "section_title": "",
                     "columns": columns, "rows": rows, "query_kind": kind,
                     "join_column": join_column, "join_source_column_index": column_index,
                     "source_column_indices": [c["source_column_index"] for c in columns],
                     "source_row_indices": selected, "query_entity_col": 0, "query_entity_col_name": "title",
                     "query_context_col_names": names[1:], "row_view_index": view_index,
                     "construction": {"rows_per_query": size, "minimum_recovered_rows": required,
                                      "verified_source_row_ids": [rid for rid in selected if rid in supported]},
                     "hidden_attributes": ([{"source_column_index": column_index, "column_name": join_column,
                        "role": "model_recoverable_join_column", "selected_rows": size, "recovered_rows": recovered_rows,
                        "required_recovered_rows": required, "recovered_value_ratio": recovered_rows / size,
                        "unreviewed_rows": size - recovered_rows}] if kind == "implicit" else []),
                     "provenance": {"builder": POLICY, "source_file": "book"}}
                js = [judge_join(q, t, source_rows) for t in targets]
                positives = [j["target_table_id"] for j in js if j["status"] == "positive"]
                reason = "accepted"
                visible = " ".join(cell_values(r)["title"] for r in rows)
                if kind == "implicit" and any(visible_join_hint(visible,
                        cell_values(source_rows[sid, rid])[join_column], join_column) for rid in selected):
                    reason = "join_value_hint_in_visible_title"
                elif len({join_key(cell_values(source_rows[sid, rid])[join_column], join_column) for rid in selected}) < 2:
                    reason = "constant_query_join_key"
                elif any(j["status"] == "unjudged" for j in js):
                    reason = "unjudged_candidate"
                elif not positives:
                    reason = "no_valid_target"
                decisions.append({"query_table_id": qid, "source_table_id": sid,
                                  "source_row_ids": selected, "kind": kind, "join_column": join_column, "reason": reason})
                if reason == "accepted":
                    used.update(selected)
                    q["target_table_ids"] = positives
                    queries.append(q)
                    judgments.extend(js)
    return queries, judgments, decisions


def reveal_authors_for_balance(queries: list[dict], sources: list[dict], namespace: str) -> list[dict]:
    """Assign surplus eligible views to explicit tasks by exposing their authors.

    Each view still occurs once. This is used only when both query kinds have
    the same row count; an odd split leaves one explicit view to downsample.
    """
    source_map = {s["source_table_id"]: s for s in sources}
    changes = []
    for split in ("train", "dev", "test"):
        implicit = [q for q in queries if q["split"] == split and q["query_kind"] == "implicit"]
        explicit_count = sum(q["split"] == split and q["query_kind"] == "explicit" for q in queries)
        count = max(0, (len(implicit) - explicit_count + 1) // 2)
        buckets = defaultdict(list)
        for q in implicit:
            buckets[q["source_table_id"]].append(q)
        groups = sorted(buckets, key=lambda sid: stable_id("reveal_source", 13, split, sid))
        for group in groups:
            buckets[group].sort(key=lambda q: stable_id("reveal_rows", 13, q["source_row_indices"]))
        ordered = [buckets[g][i] for i in range(max(map(len, buckets.values()), default=0))
                   for g in groups if i < len(buckets[g])]
        if len({q.get("join_column", "authors") for q in queries}) > 1:
            # Preserve an implicit example of each attribute within each split.
            # Expose values first for attributes that have an implicit surplus.
            protected = {}
            for q in ordered:
                protected.setdefault(q.get("join_column", "authors"), q["table_id"])
            implicit_counts = Counter(q.get("join_column", "authors") for q in implicit)
            explicit_counts = Counter(q.get("join_column", "authors") for q in queries
                                      if q["split"] == split and q["query_kind"] == "explicit")
            ordered = [q for q in ordered if q["table_id"] not in protected.values()]
            ordered.sort(key=lambda q: explicit_counts[q.get("join_column", "authors")]
                         - implicit_counts[q.get("join_column", "authors")])
        for q in ordered[:count]:
            previous_id = q["table_id"]
            sid, selected = q["source_table_id"], q["source_row_indices"]
            column = q.get("join_column", "authors")
            columns, rows = project_rows(source_map[sid], selected, ["title", column])
            qid = stable_id("query", namespace, sid, selected, "explicit", column)
            q.update(table_id=qid, object_id=qid, query_kind="explicit", columns=columns, rows=rows,
                     source_column_indices=[c["source_column_index"] for c in columns],
                     query_context_col_names=[column], hidden_attributes=[],
                     construction={"rows_per_query": len(rows), "minimum_recovered_rows": 0,
                                   "verified_source_row_ids": []})
            reason = f"{column}_exposed_for_per_split_balance"
            q["provenance"]["kind_assignment"] = reason
            changes.append({"previous_query_id": previous_id, "query_table_id": qid,
                            "source_table_id": sid, "source_row_ids": selected, "split": split,
                            "from_kind": "implicit", "to_kind": "explicit",
                            "reason": reason})
    return changes


def assign_splits(queries: list[dict], source_rows: dict, facts: list[dict], proposals: list[dict],
                  assets: list[dict] | None = None) -> dict:
    """Group source tables and duplicate books/covers; balance counts without model scores."""
    parents = {q["source_table_id"]: q["source_table_id"] for q in queries}
    def root(sid):
        while parents[sid] != sid:
            parents[sid] = parents[parents[sid]]
            sid = parents[sid]
        return sid
    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    keys = defaultdict(set)
    locations = {(q["source_table_id"], r["source_row_id"]) for q in queries for r in q["rows"]}
    for loc in locations:
        v = cell_values(source_rows[loc])
        keys[("title", title_family_key(v["title"]))].add(loc[0])
        for c in source_rows[loc]["cells"]:
            if c.get("wiki_title"):
                keys[("entity", c["wiki_title"])].add(loc[0])
    supported = {aid for f in facts if (f["source_table_id"], f["source_row_id"]) in locations for aid in f["evidence_ids"]}
    for p in proposals:
        loc = p["source_table_id"], p["source_row_id"]
        if loc in locations and (p["modality"] == "image" or p["asset_id"] in supported):
            keys[(p["modality"], p["content_sha256"])].add(loc[0])
    for members in keys.values():
        first = min(members)
        for other in members:
            union(first, other)
    # Different scans/resolutions of a cover must also stay in one split.
    # A title check avoids grouping unrelated covers with similar graphic layouts.
    near_covers = []
    if assets:
        from PIL import Image
        amap = {a["asset_id"]: a for a in assets}
        covers = []
        for p in proposals:
            loc = p["source_table_id"], p["source_row_id"]
            if p["modality"] != "image" or loc not in locations:
                continue
            with Image.open(amap[p["asset_id"]]["local_path"]) as im:
                pixels = list(im.convert("L").resize((9, 8)).getdata())
            bits = sum((pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                       for y in range(8) for x in range(8))
            covers.append((loc[0], p["asset_id"], bits, (p.get("response") or {}).get("title", "")))
        for i, a in enumerate(covers):
            for b in covers[i + 1:]:
                if a[0] != b[0] and (a[2] ^ b[2]).bit_count() <= 2 and (
                        cover_title_matches(a[3], b[3]) or cover_title_matches(b[3], a[3])):
                    union(a[0], b[0])
                    near_covers.append([a[1], b[1]])
    groups = defaultdict(list)
    for q in queries:
        groups[root(q["source_table_id"])].append(q)
    total = Counter((q["query_kind"], q.get("join_column", "authors")) for q in queries)
    fractions = {"train": 0.6, "dev": 0.2, "test": 0.2}
    counts = {s: Counter() for s in fractions}
    assignments = {}
    for gid in sorted(groups, key=lambda g: (-len(groups[g]), stable_id("split", 13, g))):
        n = Counter((q["query_kind"], q.get("join_column", "authors")) for q in groups[gid])
        def cost(split):
            return sum(((counts[s][k] + (n[k] if s == split else 0) - f * total[k]) ** 2)
                       / max(1, f * total[k]) for s, f in fractions.items() for k in total)
        chosen = min(fractions, key=cost)
        counts[chosen].update(n)
        assignments[gid] = chosen
        for q in groups[gid]:
            q["split"] = chosen
            q["split_group"] = gid
    return {"seed": 13, "fractions": fractions,
            "counts": {s: dict(Counter(q["query_kind"]
                                      for q in queries if q["split"] == s)) for s in fractions},
            "independent_components": len(groups), "largest_component_queries": max(map(len, groups.values())),
            "component_splits": assignments, "near_cover_grouping_pairs": near_covers,
            "groups_per_split": dict(Counter(assignments.values())),
            "source_groups_per_split": {s: len({q['source_table_id'] for q in queries if q['split'] == s}) for s in fractions},
            "query_splits": {q["table_id"]: q["split"] for q in queries},
            "split_key": "source_table_and_duplicate_entity_or_qualified_evidence_component",
            "split_policy": "query_only", "data_lake_scope": "shared", "historically_exposed_test": True}


def make_supervision(queries: list[dict], judgments: list[dict], facts: list[dict],
                     assets: list[dict], proposals: list[dict]) -> tuple[list[dict], list[dict]]:
    qmap = {q["table_id"]: q for q in queries}
    fmap = {(f["source_table_id"], f["source_row_id"], f.get("column_name", "authors")): f for f in facts}
    amap = {a["asset_id"]: a for a in assets}
    pmap = {(p["asset_id"], p.get("attribute", "authors")): p for p in proposals}
    qrels, recoveries = [], []
    for judgment in judgments:
        if judgment["status"] != "positive":
            continue
        qid, tid = judgment["query_table_id"], judgment["target_table_id"]
        q = qmap[qid]
        column = q.get("join_column", "authors")
        column_index = q.get("join_source_column_index", 1)
        row_count = len(q["rows"])
        supported_count = sum((q["source_table_id"], r["source_row_id"], column) in fmap for r in q["rows"])
        reason = "model_recoverable_join_column" if q["query_kind"] == "implicit" else "explicit_visible_join_column"
        qrels.append({"query_table_id": qid, "target_table_id": tid, "data_lake_table_id": tid,
                      "rel": 3, "split": q["split"], "source_table_id": q["source_table_id"],
                      "reason": reason, "join_attribute": {"column_name": column, "source_column_index": column_index,
                      "role": reason, "selected_rows": row_count,
                      "recovered_rows": supported_count if q["query_kind"] == "implicit" else 0,
                      "required_recovered_rows": q.get("construction", {}).get("minimum_recovered_rows", 2)
                          if q["query_kind"] == "implicit" else 0},
                      "judgment_policy": "oracle_completes_all_rows_without_wrong_book_expansion"})
        if q["query_kind"] != "implicit":
            continue
        for row in q["rows"]:
            fact = fmap.get((q["source_table_id"], row["source_row_id"], column))
            if fact is None:
                continue
            for aid in fact["evidence_ids"]:
                a, p = amap[aid], pmap[aid, column]
                observed = (fact["observed_values"][aid] if column == "publisher"
                            else "; ".join(p["response"]["authors"]))
                recoveries.append({"recovery_id": stable_id("evrec", qid, tid, row["row_id"], aid),
                    "path_id": stable_id("path", qid, aid, tid),
                    "query_table_id": qid, "target_table_id": tid, "data_lake_table_id": tid,
                    "query_row_id": row["row_id"],
                    "target_row_ids": [v["target_row_id"] for v in judgment["row_pairs"] if v["query_row_id"] == row["row_id"]],
                    "source_table_id": q["source_table_id"], "source_row_id": row["source_row_id"], "split": q["split"],
                    "query_entity": {"cell_text": cell_values(row)["title"], "entity_column_index": 0,
                                     "entity_column_name": "title"},
                    "recovered_attribute": {"column_index": column_index, "column_name": column, "value": fact["original_value"],
                                            "model_value": observed, "hidden_in_query": True},
                    "evidence": {"asset_id": aid, "asset_type": a["asset_type"], "content_sha256": p["content_sha256"]},
                    "path_nodes": [{"node_id": qid, "node_type": "query_table"},
                                   {"node_id": aid, "node_type": a["asset_type"] + "_asset"},
                                   {"node_id": tid, "node_type": "target_table"}],
                    "annotation": {"status": fact["annotation_status"], "human_reviewed": False,
                                   "strength": fact["strength"], "source_answer_provided": False,
                                   "matching_policy": "publisher_brand_aliases_v1_no_parent_imprint_equivalence"
                                       if column == "publisher" else "full_author_list"}})
    return qrels, recoveries


def validate_queries(queries: list[dict], targets: list[dict], judgments: list[dict],
                     recoveries: list[dict], source_rows: dict) -> dict:
    """Independent input, split and executed-row invariants, before writing a copy."""
    by_source, by_row, by_title, by_evidence = defaultdict(set), defaultdict(list), defaultdict(set), defaultdict(set)
    qmap = {q["table_id"]: q for q in queries}
    tmap = {t["table_id"]: t for t in targets}
    covered = defaultdict(set)
    for r in recoveries:
        covered[r["query_table_id"]].add(r["query_row_id"])
        by_evidence[r["evidence"]["content_sha256"]].add(r["split"])
    for q in queries:
        column = q.get("join_column", "authors")
        assert len(q["rows"]) == q.get("construction", {}).get("rows_per_query", len(q["rows"]))
        assert len(q["rows"]) >= 2
        by_source[q["source_table_id"]].add(q["split"])
        for r in q["rows"]:
            loc = q["source_table_id"], r["source_row_id"]
            by_row[loc].append(q["table_id"])
            by_title[title_family_key(cell_values(source_rows[loc])["title"])].add(q["split"])
        if q["query_kind"] == "implicit":
            assert covered[q["table_id"]] <= {r["row_id"] for r in q["rows"]}
            required = q.get("construction", {}).get("minimum_recovered_rows", len(q["rows"]))
            assert len(covered[q["table_id"]]) >= required
            attr = q["hidden_attributes"][0]
            assert attr["selected_rows"] == len(q["rows"])
            assert attr["recovered_rows"] == len(covered[q["table_id"]])
            assert column not in {c["column_name"] for c in q["columns"]}
            visible = " ".join(value for row in q["rows"] for value in cell_values(row).values())
            assert not any(visible_join_hint(visible, cell_values(
                source_rows[q["source_table_id"], row["source_row_id"]])[column], column) for row in q["rows"])
    assert all(len(v) == 1 for v in by_row.values()), "Source row reused across queries"
    assert all(len(v) == 1 for index in (by_source, by_title, by_evidence) for v in index.values()), "Cross-split duplication"
    assert len(judgments) == len(queries) * len(targets)
    assert not any(j["status"] == "unjudged" for j in judgments)
    replay_correct, replay_count, swapped_correct, swapped_count, oracle_count = 0, 0, 0, 0, 0
    for j in judgments:
        if j["status"] != "positive":
            continue
        q, t = qmap[j["query_table_id"]], tmap[j["target_table_id"]]
        column = q.get("join_column", "authors")
        assert all(p["status"] == "correct_record" for p in j["row_pairs"])
        assert {p["query_row_id"] for p in j["row_pairs"] if p["added_values"]} == {r["row_id"] for r in q["rows"]}
        if q["query_kind"] != "implicit":
            continue
        observed = {r["query_row_id"]: r["recovered_attribute"]["model_value"] for r in recoveries
                    if r["query_table_id"] == q["table_id"] and r["target_table_id"] == t["table_id"]}
        gold = {(p["query_row_id"], p["target_row_id"]) for p in j["row_pairs"]}
        oracle_count += len(gold)
        observed_gold = {(rid, tid) for rid, tid in gold if rid in observed}
        row_ids = sorted(observed)
        swapped = {rid: observed[row_ids[(i + 1) % len(row_ids)]] for i, rid in enumerate(row_ids)}
        for is_swap in (False, True):
            supplied = swapped if is_swap else observed
            pairs = {(rid, tr["row_id"]) for rid, value in supplied.items() for tr in t["rows"]
                     if join_key(value, column) and join_key(value, column) == join_key(cell_values(tr).get(column, ""), column)}
            if is_swap:
                swapped_count += len(pairs)
                swapped_correct += len(pairs & gold)
            else:
                assert pairs == observed_gold, "Observed evidence values fail to reproduce judged join on supported rows"
                replay_count += len(pairs)
                replay_correct += len(pairs & gold)
    implicit = [q for q in queries if q["query_kind"] == "implicit"]
    return {"unique_query_rows": len(by_row), "row_reuse": 0, "source_group_split_overlap": 0,
            "title_split_overlap": 0, "qualified_evidence_split_overlap": 0,
            "judged_pairs": len(judgments), "judged_coverage": 1.0,
            "implicit_query_rows": sum(len(q["rows"]) for q in implicit),
            "verified_implicit_rows": sum(len(covered[q["table_id"]]) for q in implicit),
            "unreviewed_implicit_rows": sum(len(q["rows"]) - len(covered[q["table_id"]]) for q in implicit),
            "recovered_rows_per_implicit_query": dict(Counter(len(covered[q["table_id"]]) for q in implicit)),
            "implicit_oracle_pairs": oracle_count,
            "observed_value_join_correct_pairs": replay_correct, "observed_value_join_pairs": replay_count,
            "swapped_value_join_correct_pairs": swapped_correct, "swapped_value_join_pairs": swapped_count,
            "scope": "Dataset qualification and conditional value replay, not an unbiased extraction or retrieval evaluation."}


def build_standalone(source: Path, destination: Path, proposals_path: Path, history_path: Path, *,
                     implicit_rows: int = 5, explicit_rows: int = 5,
                     minimum_recovered_rows: int = 2, balanced: bool = False,
                     publisher_proposals_path: Path | None = None,
                     publisher_reviews_path: Path | None = None) -> dict:
    """Create a versioned training dataset; input records and evidence remain untouched."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    if destination.exists():
        raise FileExistsError(destination)
    before = dataset_hashes(source)
    manifest, data = load_artifacts(source)
    proposals = read_rows(proposals_path)
    facts, evidence_audit = collect_facts(source, proposals_path, history_path)
    publisher_facts, publisher_audit = [], []
    if publisher_proposals_path:
        from .abebooks_publisher import collect_publisher_facts
        if publisher_reviews_path is None:
            raise ValueError("Publisher proposals require content-bound pixel reviews")
        publisher_proposals = read_rows(publisher_proposals_path)
        publisher_facts, publisher_audit = collect_publisher_facts(
            data, publisher_proposals, read_rows(publisher_reviews_path))
        proposals += publisher_proposals
        facts += publisher_facts
    # Bind every annotation to the unchanged evidence bytes before trusting it.
    assets = {a["asset_id"]: a for a in data["bridge_assets"]}
    for p in proposals:
        a = assets[p["asset_id"]]
        actual = file_hash(Path(a["local_path"])) if a["asset_type"] == "image" else hashlib.sha256(a["content"].encode()).hexdigest()
        if actual != p["content_sha256"] or p["source_answer_provided"]:
            raise ValueError(f"Invalid blind evidence annotation: {p['asset_id']}")
    targets, changes = complementary_targets(data, destination.name)
    if publisher_proposals_path:
        for target in targets:
            target["join_columns"] = [c["column_name"] for c in target["columns"]
                                      if c["column_name"] in {"authors", "publisher"}]
    queries, judgments, decisions = [], [], []
    used = set()
    for column in (["publisher", "authors"] if publisher_proposals_path else ["authors"]):
        qs, js, ds = make_queries(data, targets, facts, destination.name,
            implicit_rows=implicit_rows, explicit_rows=explicit_rows, minimum_recovered_rows=minimum_recovered_rows,
            join_column=column, excluded_rows=used)
        queries.extend(qs)
        judgments.extend(js)
        decisions.extend(ds)
        used.update((q["source_table_id"], r["source_row_id"]) for q in qs for r in q["rows"])
    source_rows = {(s["source_table_id"], r["row_id"]): r for s in data["source_tables"] for r in s["rows"]}
    splits = assign_splits(queries, source_rows, facts, proposals, data["bridge_assets"])
    before_balance = {s: dict(c) for s, c in splits["counts"].items()}
    kind_assignments = []
    if balanced:
        from .abebooks_rebalance import select_balanced_standalone
        if implicit_rows == explicit_rows:
            kind_assignments = reveal_authors_for_balance(queries, data["source_tables"], destination.name)
            replacements = {r["previous_query_id"]: r["query_table_id"] for r in kind_assignments}
            changed_ids = set(replacements.values())
            for decision in decisions:
                if decision.get("query_table_id") in replacements:
                    decision.update(previous_query_id=decision["query_table_id"],
                                    query_table_id=replacements[decision["query_table_id"]], kind="explicit",
                                    kind_assignment="authors_exposed_for_per_split_balance")
            # Re-execute labels for the changed visible schema; never just flip a kind flag.
            judgments = [j for j in judgments if j["query_table_id"] not in replacements]
            for q in queries:
                if q["table_id"] in changed_ids:
                    js = [judge_join(q, t, source_rows) for t in targets]
                    assert not any(j["status"] == "unjudged" for j in js)
                    assert [j["target_table_id"] for j in js if j["status"] == "positive"] == q["target_table_ids"]
                    judgments.extend(js)
        queries = select_balanced_standalone(queries)
        kept = {q["table_id"] for q in queries}
        judgments = [j for j in judgments if j["query_table_id"] in kept]
        for decision in decisions:
            if decision.get("reason") == "accepted" and decision["query_table_id"] not in kept:
                decision["reason"] = "explicit_downsampled_for_per_split_balance"
        groups = {q["split_group"]: q["split"] for q in queries}
        splits.update(counts={s: dict(Counter(q["query_kind"] for q in queries if q["split"] == s))
                              for s in ("train", "dev", "test")},
                      query_splits={q["table_id"]: q["split"] for q in queries},
                      independent_components=len(groups), component_splits=groups,
                      groups_per_split=dict(Counter(groups.values())),
                      source_groups_per_split={s: len({q["source_table_id"] for q in queries if q["split"] == s})
                                               for s in ("train", "dev", "test")},
                      largest_component_queries=max(Counter(q["split_group"] for q in queries).values()),
                      balance_policy=("per_split_author_visibility_assignment_then_explicit_downsampling"
                                      if implicit_rows == explicit_rows else
                                      "per_split_explicit_downsampling_round_robin_by_source"), balance_seed=13)
    qrels, recoveries = make_supervision(queries, judgments, facts, data["bridge_assets"], proposals)
    validation = validate_queries(queries, targets, judgments, recoveries, source_rows)
    # Row count and supervision coverage are reported separately. Nonempty
    # splits are a format requirement, not a claim of statistical sufficiency.
    for split in ("train", "dev", "test"):
        counts = splits["counts"][split]
        if not all(counts.get(kind, 0) for kind in ("implicit", "explicit")):
            raise ValueError(f"Missing query type in {split}: {counts}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".env.openai"))
    archive_name = "before_publisher_expansion" if publisher_proposals_path else "before_standalone"
    archive = destination / "provenance" / archive_name
    changed = {"dataset_manifest.json", "qrels.jsonl", "splits.json", "retrieval_catalog.json", "REPORT.md",
               "VALIDATION.json", "REGENERATION.json", "table_queryability_decisions.jsonl", "explicit"}
    if publisher_proposals_path:
        changed.update({"BUILD.json", "TRAINING_PROTOCOL.json", "audit", "splits"})
    for name in ("query_tables", "data_lake_tables", "evidence_recoveries"):
        changed.add(name)
    for name in sorted(changed):
        old = destination / name
        if old.exists():
            new = archive / name
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), new)
    def save_json(name, value):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for name, records in (("query_tables", queries), ("data_lake_tables", targets), ("evidence_recoveries", recoveries)):
        path = f"{name}/part-00000.jsonl"
        write_rows(destination / path, records)
        manifest["artifacts"][name] = {"directory": name, "total_records": len(records), "max_records_per_shard": 50000,
                                      "shards": [{"path": path, "records": len(records)}]}
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "table_queryability_decisions.jsonl", decisions)
    for name, records in (("author_facts", facts), ("evidence_audit", evidence_audit),
                          ("candidate_judgments", judgments), ("target_projection_changes", changes),
                          ("query_kind_assignments", kind_assignments)):
        if name == "author_facts":
            records = [f for f in records if f.get("column_name", "authors") == "authors"]
        write_rows(destination / f"audit/{name}.jsonl", records)
    if publisher_proposals_path:
        write_rows(destination / "audit/publisher_facts.jsonl", publisher_facts)
        write_rows(destination / "audit/publisher_evidence_audit.jsonl", publisher_audit)
    save_json("splits.json", splits)
    save_json("retrieval_catalog.json", {"query_ids": [q["table_id"] for q in queries],
        "target_ids": [t["table_id"] for t in targets], "query_kinds": {q["table_id"]: q["query_kind"] for q in queries},
        "query_splits": splits["query_splits"], "qrels": {q["table_id"]: q["target_table_ids"] for q in queries},
        "numerical_index_status": "requires_fresh_embedding_and_ANN_build", "cache_namespace": destination.name})
    explicit_ids = {q["table_id"] for q in queries if q["query_kind"] == "explicit"}
    write_rows(destination / "explicit/queries.jsonl", [q for q in queries if q["table_id"] in explicit_ids])
    write_rows(destination / "explicit/targets.jsonl", targets)
    write_rows(destination / "explicit/qrels.jsonl", [r for r in qrels if r["query_table_id"] in explicit_ids])
    for split in ("train", "dev", "test"):
        write_rows(destination / f"splits/{split}.queries.jsonl", [q for q in queries if q["split"] == split])
        write_rows(destination / f"splits/{split}.qrels.jsonl", [r for r in qrels if r["split"] == split])
        write_rows(destination / f"splits/{split}.recoveries.jsonl", [r for r in recoveries if r["split"] == split])
    active_facts = {(r["source_table_id"], r["source_row_id"]) for r in recoveries}
    report = {"policy": POLICY, "source": str(source), "destination": str(destination),
        "queries": len(queries), "split_counts": splits["counts"], "candidate_tables": len(targets),
        "query_shape": {"implicit_rows": implicit_rows, "explicit_rows": explicit_rows,
                        "minimum_recovered_rows": minimum_recovered_rows},
        "balanced_each_split": balanced, "counts_before_balance": before_balance,
        "author_visibility_assignments": len(kind_assignments),
        "balance_policy": splits.get("balance_policy"),
        "source_tables": len(data["source_tables"]), "source_rows": len(source_rows), "assets": len(assets),
        "qrels": len(qrels), "recovery_paths": len(recoveries), "distinct_recovery_facts": len(active_facts),
        "qualified_author_rows": sum(f.get("column_name", "authors") == "authors" for f in facts),
        "qualified_publisher_rows": len(publisher_facts),
        "join_column_counts": dict(Counter(q.get("join_column", "authors") for q in queries)),
        "join_column_split_counts": {s: {c: dict(Counter(q["query_kind"] for q in queries
            if q["split"] == s and q.get("join_column", "authors") == c)) for c in ("authors", "publisher")}
            for s in ("train", "dev", "test")},
        "fact_strengths": dict(Counter(f["strength"] for f in facts)),
        "groups_per_split": splits["groups_per_split"], "independent_components": splits["independent_components"],
        "source_groups_per_split": splits["source_groups_per_split"], "largest_component_queries": splits["largest_component_queries"],
        "proposal_count": len(proposals), "annotation_status": "model_assisted_not_human_gold",
        "qualification_counts": dict(Counter(a["status"] for a in evidence_audit)),
        "query_decisions": dict(Counter(a["reason"] for a in decisions)),
        "target_projections_changed": sum(c["before_columns"] != c["after_columns"] for c in changes),
        "all_candidate_row_memberships_preserved": all(c["rows_preserved"] for c in changes),
        "input_hashes": before, "proposals_sha256": file_hash(proposals_path), "history_sha256": file_hash(history_path),
        "original_dataset_unchanged": before == dataset_hashes(source),
        "preserved_artifacts_unchanged": all(file_hash(destination / sh["path"]) == before[sh["path"]]
            for name, spec in manifest["artifacts"].items() if name not in {"query_tables", "data_lake_tables", "evidence_recoveries"}
            for sh in spec["shards"]),
        "validation": validation, "historically_exposed_test": True,
        "intended_use": "standalone_small_data_training_validation_test_with_pretrained_initialization",
        "limitations": ["Model-assisted annotation and qualification select readable evidence; not independent human gold.",
                        "Test queries are held out by source/duplicate groups, but raw records were historically exposed.",
                        "The 174-target corpus and 5441-asset corpus are shared transductively.",
                        "Unqualified assets are not established negative evidence; explicit tasks have no recovery supervision.",
                        "Candidate title removal and source-author projection require new baselines and fresh caches.",
                        "Query size and verified recovery coverage differ; unreviewed rows have source oracle values only.",
                        "Dataset size and format checks do not establish model training gains or natural-lake generalization."]}
    assert report["original_dataset_unchanged"] and report["preserved_artifacts_unchanged"]
    if publisher_proposals_path:
        report.update(publisher_proposals_sha256=file_hash(publisher_proposals_path),
                      publisher_reviews_sha256=file_hash(publisher_reviews_path),
                      publisher_qualification_counts=dict(Counter(r["status"] for r in publisher_audit)),
                      publisher_matching_policy="publisher_brand_aliases_v1_no_parent_imprint_equivalence",
                      mixed_query_allocation="publisher_first_then_authors_on_unused_rows")
        report["limitations"].append("Publisher labels use source-aware Codex pixel review; blind local reader errors are retained in the audit.")
    save_json("BUILD.json", report)
    save_json("VALIDATION.json", validation)
    manifest["single_files"]["stats"] = "BUILD.json"
    manifest["query_construction"] = {"provider": "abebooks", "policy_version": POLICY,
                                      "query_shape": report["query_shape"],
                                      "history": f"provenance/{archive_name}/dataset_manifest.json"}
    manifest.pop("explicit_regeneration", None)
    manifest["curation"] = {"report": "BUILD.json", "cache_namespace": destination.name,
                            "evidence_supervision": "recovery_records_only_no_provenance_fallback",
                            "judged_coverage": 1.0}
    save_json("dataset_manifest.json", manifest)
    return report

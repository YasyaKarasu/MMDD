"""Rebuild AbeBooks from projected source tables and existing evidence labels.

No model client is created. The maintained builder receives a complete replay
cache: approved recovery facts for known row/asset/attributes, empty otherwise.
The original query/target layouts and unreviewed extraction candidates are not
used to manufacture new implicit supervision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .abebooks_ablation import project_table, read_rows, write_rows
from .abebooks_explicit import assert_disjoint_sources, historical_implicit_sources, select_candidates
from .abebooks_rebalance import grouped_split, query_kinds
from .tables import column_profiles


BIBLIOGRAPHIC_COLUMNS = {"title", "authors", "publisher", "publication_year",
                         "region", "seller_description"}
READABLE_HEADERS = {"title": "Book title", "authors": "Book authors", "publisher": "Book publisher",
                    "publication_year": "Publication year", "region": "Seller region",
                    "seller_description": "Seller description"}


def natural_author_names(value: str) -> str:
    """Invert unambiguous catalog names; preserve qualifiers and ambiguous lists."""
    parts, changed = [], False
    for name in value.split(";"):
        pieces = [part.strip() for part in name.split(",")]
        if (len(pieces) == 2 and all(pieces)
                and all(c.isalpha() or c in " .'-’" for piece in pieces for c in piece)
                and not re.search(r"\b(and|ed|eds|editor|jr|sr|et al)\b", name, re.I)):
            parts.append(f"{pieces[1]} {pieces[0]}")
            changed = True
        else:
            parts.append(name.strip())
    return "; ".join(parts) if changed else value


def without_series_notes(title: str) -> str:
    """Remove parenthetical series metadata while retaining the main book title."""
    cleaned = re.sub(r"\s*\([^()]*\bseries\b[^()]*\)", "", title, flags=re.I).strip()
    return cleaned or title


def add_book_title_context(tables: list[dict], assets: list[dict]) -> list[dict]:
    """Restore the source-page book heading to every attached text fragment."""
    titles = {(t["source_table_id"], row["row_id"]): cell["text"]
              for t in tables for row in t["rows"] for cell in row["cells"]
              if cell["column_name"] == "title"}
    changes = []
    for asset in assets:
        title = titles.get((asset.get("source_table_id"), asset.get("source_row_id")))
        if asset["asset_type"] != "text" or title is None:
            continue
        original = asset["content"]
        prefix = f"Book title: {title}\n\n"
        asset["content"] = prefix + original
        changes.append({"asset_id": asset["asset_id"], "source_title": title,
                        "original_content_sha256": hashlib.sha256(original.encode()).hexdigest(),
                        "added_characters": len(prefix), "original_characters": len(original)})
    return changes


def recovery_key(record: dict) -> tuple:
    attribute = record["recovered_attribute"]
    name = record.get("annotation_provenance", {}).get("original_attribute_name", attribute["column_name"])
    return (record.get("annotation_provenance", {}).get("original_source_table_id", record["source_table_id"]), record["source_row_id"],
            record["evidence"]["asset_id"], name,
            record.get("annotation_provenance", {}).get("original_attribute_value", attribute["value"]))


def replay_attributes(task, recoveries: list[dict]) -> list[dict]:
    """Expose every approved fact for this task, never an unreviewed candidate."""
    attributes = {}
    for record in recoveries:
        sid, row, aid = record["source_table_id"], record["source_row_id"], record["evidence"]["asset_id"]
        name = record["recovered_attribute"]["column_name"]
        value = record["recovered_attribute"]["value"]
        if (sid, row, aid) != (task.source_table_id, task.source_row_id, task.asset["asset_id"]):
            continue
        if name in task.candidate_attribute_names:
            attributes[(name, value)] = {"name": name, "value": value,
                "evidence": record["evidence"].get("model_evidence", "")}
    return list(attributes.values())


def rebuild_from_sources(source: Path, destination: Path, *, keep: set[str], seed: int = 13,
                         readable_headers: bool = False, natural_authors: bool = False,
                         source_reference: Path | None = None, group_by_title: bool = False,
                         proportional_splits: bool = False, strip_series_notes: bool = False,
                         group_by_authors: bool = False, title_embeddings: Path | None = None,
                         contextualize_book_text: bool = False, group_by_publisher: bool = False,
                         publisher_min_rows: int = 1) -> dict:
    """Project sources, invoke shared implicit/explicit constructors, then split."""
    if destination.exists():
        raise FileExistsError(destination)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts_old"))
    import build_mm_joinability_dataset as builder
    from clean_abebooks_joinability import modal_share

    manifest = json.loads((source / "dataset_manifest.json").read_text())
    data = {name: [row for shard in spec["shards"] for row in read_rows(source / shard["path"])]
            for name, spec in manifest["artifacts"].items()}
    reference_hashes = {}
    if source_reference is not None:
        reference_manifest = json.loads((source_reference / "dataset_manifest.json").read_text())
        restored = {name: [row for shard in reference_manifest["artifacts"][name]["shards"]
                          for row in read_rows(source_reference / shard["path"])]
                    for name in ("source_tables", "entities", "table_asset_links")}
        # Restore original source fields/indices, never old annotations or tasks.
        original = {t["source_table_id"]: t for t in restored["source_tables"]}
        assert set(original) == {t["source_table_id"] for t in data["source_tables"]}
        for table in data["source_tables"]:
            full = original[table["source_table_id"]]
            assert [r["row_id"] for r in table["rows"]] == [r["row_id"] for r in full["rows"]]
            values = {(r["row_id"], c["column_name"]): c["text"] for r in full["rows"] for c in r["cells"]}
            assert all(values[r["row_id"], c["column_name"]] == c["text"]
                       for r in table["rows"] for c in r["cells"])
        assert {e["entity_id"] for e in data["entities"]} == {e["entity_id"] for e in restored["entities"]}
        available = {a["asset_id"] for a in data["bridge_assets"]}
        for link in restored["table_asset_links"]:
            link["asset_ids"] = [a for a in link["asset_ids"] if a in available]
        data.update(restored)
        reference_files = {"dataset_manifest.json", *(shard["path"] for name in restored
                           for shard in reference_manifest["artifacts"][name]["shards"])}
        reference_hashes = {p: hashlib.sha256((source_reference / p).read_bytes()).hexdigest()
                            for p in sorted(reference_files)}
    old_qrels = read_rows(source / manifest["single_files"]["qrels"])
    history = manifest.get("explicit_regeneration", {}).get(
        "historical_decisions", manifest["single_files"]["table_queryability_decisions"])
    decisions = read_rows(source / history)
    excluded_explicit = historical_implicit_sources(decisions, old_qrels)
    grouping = {}
    if group_by_title or group_by_authors or group_by_publisher or title_embeddings is not None:
        from .abebooks_grouping import group_book_sources, partition_books_by_publisher
        if group_by_publisher:
            data["source_tables"], row_sources, grouping = partition_books_by_publisher(
                data["source_tables"], publisher_min_rows)
        else:
            data["source_tables"], row_sources, grouping = group_book_sources(
                data["source_tables"], seed, field="authors" if group_by_authors else "title",
                title_embeddings=title_embeddings)
        for table in data["source_tables"]:
            table["metadata"]["column_profiles"] = column_profiles(table["rows"], table["columns"], wiki_threshold=0.5)[0]
        for name in ("bridge_assets", "entities", "table_asset_links", "evidence_recoveries"):
            for record in data[name]:
                old_sid = record["source_table_id"]
                row_id = record["row_id"] if name == "table_asset_links" else record["source_row_id"]
                if name == "evidence_recoveries":
                    record.setdefault("annotation_provenance", {}).setdefault("original_source_table_id", old_sid)
                record["source_table_id"] = row_sources[old_sid, row_id]
                for reference in record.get("appears_in", []):
                    reference["source_table_id"] = row_sources[reference["source_table_id"], reference["row_id"]]
    approved = {recovery_key(r): r for r in data["evidence_recoveries"]}
    known_by_task = defaultdict(list)
    replayed_facts = {}
    for key, record in approved.items():
        replay = copy.deepcopy(record)
        if natural_authors and replay["recovered_attribute"]["column_name"] == "authors":
            replay["recovered_attribute"]["value"] = natural_author_names(replay["recovered_attribute"]["value"])
        if readable_headers:
            attr = replay["recovered_attribute"]
            attr["column_name"] = READABLE_HEADERS.get(attr["column_name"], attr["column_name"])
        task_key = (record["source_table_id"], record["source_row_id"], record["evidence"]["asset_id"])
        known_by_task[task_key].append(replay)
        replayed_facts[(*task_key, key[3], replay["recovered_attribute"]["value"])] = record
    original_sources = data["source_tables"]
    maps = {t["source_table_id"]: {c["column_index"]: i for i, c in enumerate(
        c for c in t["columns"] if c["column_name"] in keep)} for t in original_sources}
    sources = [project_table(t, keep, maps[t["source_table_id"]]) for t in original_sources]
    title_changes = []
    if strip_series_notes:
        title_entities = set()
        for table in sources:
            for row in table["rows"]:
                for cell in row["cells"]:
                    if cell["column_name"] != "title":
                        continue
                    title_entities.add(cell.get("wiki_title"))
                    old, new = cell["text"], without_series_notes(cell["text"])
                    if old != new:
                        title_changes.append({"source_table_id": table["source_table_id"],
                            "source_row_id": row["row_id"], "original": old, "normalized": new})
                        cell["text"] = cell["raw"] = new
            table["metadata"]["column_profiles"] = column_profiles(table["rows"], table["columns"], wiki_threshold=0.5)[0]
        for entity in data["entities"]:
            if entity["wiki_title"] in title_entities:
                entity["display_texts"] = [without_series_notes(t) for t in entity.get("display_texts", [])]
    author_changes = []
    if natural_authors:
        for table in sources:
            for row in table["rows"]:
                for cell in row["cells"]:
                    if cell["column_name"] == "authors":
                        old, new = cell["text"], natural_author_names(cell["text"])
                        if old != new:
                            author_changes.append({"source_table_id": table["source_table_id"],
                                "source_row_id": row["row_id"], "original": old, "normalized": new})
                            cell["text"] = cell["raw"] = new
            table["metadata"]["column_profiles"] = column_profiles(
                table["rows"], table["columns"], wiki_threshold=0.5)[0]
    if readable_headers:
        for table in sources:
            named = [*table["columns"], *table.get("metadata", {}).get("column_profiles", []),
                     *(cell for row in table["rows"] for cell in row["cells"])]
            for item in named:
                if "column_name" in item:
                    item["column_name"] = READABLE_HEADERS.get(item["column_name"], item["column_name"])
    source_map = {t["source_table_id"]: t for t in sources}
    text_changes = add_book_title_context(sources, data["bridge_assets"]) if contextualize_book_text else []
    # Hash the exact input artifacts, with no directory-wide walk or secret files.
    input_paths = {"dataset_manifest.json", *manifest["single_files"].values(), history,
                   *(s["path"] for a in manifest["artifacts"].values() for s in a["shards"])}
    before_hashes = {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in sorted(input_paths)}
    destination.mkdir(parents=True)
    write_rows(destination / "source_tables/part-00000.jsonl", sources)
    args = builder.parse_args(["--input_dir", str(destination), "--output_dir", str(destination),
        "--cache_dir", str(destination / "replay_cache"), "--seed", str(seed),
        "--min_recovered_value_ratio", "0.4", "--explicit_join_fallback_mode", "match_implicit"])
    args.reparse_cached_model_outputs = False
    args.refresh_invalid_model_cache = False
    args.text_model_name = args.image_model_name = "existing-approved-label-replay"
    cache = builder.ExtractionCache(destination / "replay_cache/extractions.jsonl", reuse=False)
    assets = {a["asset_id"]: a for a in data["bridge_assets"]}
    entity_assets = defaultdict(list)
    for asset in assets.values():
        entity_assets[asset["entity_id"]].append(asset["asset_id"])
    wiki_entities = {e["wiki_title"]: e["entity_id"] for e in data["entities"]}
    tasks = builder.collect_extraction_tasks_from_tables(
        source_paths=[destination / "source_tables/part-00000.jsonl"], assets=assets,
        entity_to_assets=entity_assets, wiki_to_entity_id=wiki_entities, args=args)
    for task in tasks:
        key = (task.source_table_id, task.source_row_id, task.asset["asset_id"])
        cache.put(task.cache_key, {"cache_key": task.cache_key, "entity_id": task.entity["entity_id"],
            "asset_id": task.asset["asset_id"], "asset_type": task.asset["asset_type"],
            "candidate_attribute_names": task.candidate_attribute_names,
            "attributes": replay_attributes(task, known_by_task[key]), "error": "",
            "annotation_origin": "existing_evidence_recoveries_only"})
    queries, targets, qrels, rebuilt_decisions = [], [], [], []
    extraction_writer, recovery_writer = builder.ListRecordWriter(), builder.ListRecordWriter()
    state = builder.ModelConcurrencyState.from_args(args)
    for table in sources:
        qs, ts, rs, decision = builder.build_table_join_records(
            source_table=table, split="train", assets=assets, entity_to_assets=entity_assets,
            wiki_to_entity_id=wiki_entities, extractor=None, cache=cache, progress=None,
            concurrency_state=state, extraction_writer=extraction_writer,
            recovery_writer=recovery_writer, args=args)
        queries.extend(qs)
        if qs:
            targets.extend(ts)
        qrels.extend(rs)
        rebuilt_decisions.append({"source_table_id": table["source_table_id"], **decision})
    implicit_count = len(queries)
    if not implicit_count:
        raise ValueError("No implicit query survived the source projection")
    excluded_explicit.update(q["source_table_id"] for q in queries)
    candidates = {}
    for table in sources:
        sid = table["source_table_id"]
        if sid in excluded_explicit:
            continue
        entity = (table["metadata"].get("candidate_entity_columns") or [None])[0]
        generated = builder.build_explicit_join_fallback_candidates(
            source_table=table, split="train", entity_col=entity,
            rejected_multimodal_reason="no_current_approved_recovery", args=args, force=True)
        viable = [c for c in generated if modal_share(table, c["join_column_index"]) < 1.0]
        if viable:
            candidates[sid] = viable
    selected, _ = select_candidates(candidates, {"train": implicit_count, "dev": 0, "test": 0}, seed)
    if sum(map(len, selected.values())) != implicit_count:
        raise ValueError(f"Only {sum(map(len, selected.values()))} disjoint explicit candidates for "
                         f"{implicit_count} implicit queries; refusing to discard implicit labels or reuse their sources")
    for sid, group in selected.items():
        rebuilt = builder.rebuild_selected_explicit_join_candidates(
            source_table=source_map[sid], split="train", candidate_decisions=group, args=args)
        for candidate in rebuilt:
            qs, ts, rs, _ = builder.materialize_balanced_explicit_join_candidate(
                source_table=source_map[sid], split="train", candidate_decision=candidate, args=args)
            queries.extend(qs)
            targets.extend(ts)
            qrels.extend(rs)
    represented = {t["source_table_id"] for t in targets}
    targets.extend(builder.raw_data_lake_record(t) for t in sources if t["source_table_id"] not in represented)
    kinds = query_kinds(qrels)
    splits = grouped_split(queries, kinds, seed, proportional=proportional_splits)
    recoveries = recovery_writer.records
    for record in recoveries:
        original_name = record["recovered_attribute"]["column_name"]
        if readable_headers:
            original_name = {v: k for k, v in READABLE_HEADERS.items()}.get(original_name, original_name)
        old = replayed_facts[record["source_table_id"], record["source_row_id"],
                             record["evidence"]["asset_id"], original_name,
                             record["recovered_attribute"]["value"]]
        original_value = recovery_key(old)[4]
        record["annotation_provenance"] = {"original_attribute_name": original_name,
                                          "original_attribute_value": original_value,
                                          "original_source_table_id": recovery_key(old)[0]}
        assert recovery_key(record) in approved  # Every output fact must already be approved.
        record["annotation_provenance"] = {"source_dataset": str(source),
            "original_attribute_name": original_name,
            "original_attribute_value": original_value,
            "original_source_table_id": recovery_key(old)[0],
            "original_recovery_id": old["recovery_id"], "original_auto_check": old.get("auto_check"),
            "note": "Existing evidence/value label; new layout has not been model-reviewed"}
    for record in queries:
        record["split"] = splits[record["table_id"]]
    for record in [*qrels, *recoveries]:
        record["split"] = splits[record["query_table_id"]]
    assert_disjoint_sources(qrels)
    assert Counter(kinds.values()) == {"implicit": implicit_count, "explicit": implicit_count}
    # Rekey generated objects, including every reference, so old feature caches
    # cannot accidentally be used for different source projections.
    namespace = hashlib.sha256(json.dumps({"keep": sorted(keep), "input": before_hashes,
        **({"readable_headers": True} if readable_headers else {}),
        **({"natural_authors": True} if natural_authors else {}),
        **({"source_reference_hashes": reference_hashes} if reference_hashes else {}),
        **({"title_grouping": grouping} if grouping else {}),
        **({"proportional_splits": True} if proportional_splits else {}),
        **({"strip_series_notes": True} if strip_series_notes else {}),
        **({"contextualize_book_text": True} if contextualize_book_text else {}),
        "seed": seed}, sort_keys=True).encode()).hexdigest()[:12]
    id_map = {r["table_id"]: f'{r["table_id"]}_source_{namespace}' for r in [*queries, *targets]}
    def rekey(value):
        if isinstance(value, str):
            return id_map.get(value, value)
        if isinstance(value, list):
            return [rekey(v) for v in value]
        if isinstance(value, dict):
            return {k: rekey(v) for k, v in value.items()}
        return value
    data.update(source_tables=sources, query_tables=queries, data_lake_tables=targets,
                attribute_extractions=extraction_writer.records, evidence_recoveries=recoveries)
    for entity in data["entities"]:
        entity["appears_in"] = [{**r, "column_index": maps[r["source_table_id"]][r["column_index"]]}
            for r in entity["appears_in"] if r["column_index"] in maps[r["source_table_id"]]]
        if readable_headers:
            for reference in entity["appears_in"]:
                if "column_name" in reference:
                    reference["column_name"] = READABLE_HEADERS.get(reference["column_name"], reference["column_name"])
    for link in data["table_asset_links"]:
        link["column_index"] = maps[link["source_table_id"]].get(link["column_index"])
    data, qrels, splits = rekey(data), rekey(qrels), {id_map[q]: s for q, s in splits.items()}
    new_manifest = {"format": "sharded_jsonl", "artifacts": {},
        "artifact_references": {"data_lake_tables": {"field": "source_table_ref",
            "target_artifact": "source_tables", "resolution": "stream_by_source_table_id"}},
        "single_files": {
        "qrels": "qrels.jsonl", "splits": "splits.json",
        "table_queryability_decisions": "table_queryability_decisions.jsonl"},
        "query_construction": {"provider": "abebooks", "method": "maintained_shared_builder",
            "implicit_annotation_origin": "existing_approved_recoveries_only"}}
    for name, records in data.items():
        relative = f"{name}/part-00000.jsonl"
        write_rows(destination / relative, records)
        new_manifest["artifacts"][name] = {"total_records": len(records),
            "shards": [{"path": relative, "records": len(records)}]}
    write_rows(destination / "qrels.jsonl", qrels)
    write_rows(destination / "table_queryability_decisions.jsonl", rebuilt_decisions)
    counts = {s: dict(Counter(kinds[q] for q in kinds if splits[id_map[q]] == s))
              for s in ("train", "dev", "test")}
    retained = {recovery_key(r) for r in recoveries}
    report = {"source": str(source), "seed": seed, "kept_columns": sorted(keep),
        "title_grouping": grouping,
        "proportional_splits": proportional_splits,
        "strip_series_notes": strip_series_notes, "changed_title_cells": len(title_changes),
        "contextualize_book_text": contextualize_book_text, "changed_text_assets": len(text_changes),
        "source_reference": str(source_reference) if source_reference is not None else None,
        "source_reference_hashes": reference_hashes,
        "source_reference_unchanged": all(hashlib.sha256((source_reference / p).read_bytes()).hexdigest() == h
                                          for p, h in reference_hashes.items()),
        "readable_headers": READABLE_HEADERS if readable_headers else {},
        "natural_author_names": natural_authors, "changed_author_cells": len(author_changes),
        "removed_columns": sorted({c["column_name"] for t in original_sources for c in t["columns"]} - keep),
        "source_tables": len(sources), "queries": len(queries), "targets": len(targets),
        "split_counts": counts, "implicit_queries": implicit_count, "explicit_queries": implicit_count,
        "approved_input_facts": len(approved), "retained_approved_facts": len(retained),
        "unretained_approved_facts": [list(k) for k in sorted(set(approved) - retained)],
        "new_implicit_facts": 0, "evidence_records": len(recoveries),
        "explicit_candidate_sources": len(candidates), "input_hashes": before_hashes,
        "source_unchanged": all(hashlib.sha256((source / p).read_bytes()).hexdigest() == h
                                for p, h in before_hashes.items()),
        "source_coverage": len({t["source_table_id"] for t in targets}),
        "test_policy": "Previously exposed data; regression set, not an untouched test set"}
    for filename, obj in [("dataset_manifest.json", new_manifest), ("REBUILD.json", report),
        ("splits.json", {"split_key": "source_table_id", "query_splits": splits, "counts": counts, "seed": seed})]:
        (destination / filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    if natural_authors:
        write_rows(destination / "AUTHOR_NORMALIZATION.jsonl", author_changes)
    if strip_series_notes:
        write_rows(destination / "TITLE_NORMALIZATION.jsonl", title_changes)
    if contextualize_book_text:
        write_rows(destination / "TEXT_CONTEXT.jsonl", text_changes)
    return report

import json
import shutil
from collections import Counter
from types import SimpleNamespace

import pytest

from mmdd_dataset.abebooks_ablation import project_table, read_rows, write_rows
from mmdd_dataset.abebooks_source_rebuild import (natural_author_names, recovery_key,
                                                 rebuild_from_sources, replay_attributes, without_series_notes,
                                                 add_book_title_context)


def test_text_context_uses_attached_source_title_and_preserves_all_original_content():
    tables = [{"source_table_id": "books", "rows": [
        {"row_id": 1, "cells": [{"column_name": "title", "text": "A book"}]},
        {"row_id": 2, "cells": [{"column_name": "title", "text": "Another book"}]}]}]
    assets = [dict(asset_id="biography", asset_type="text", source_table_id="books", source_row_id=1, content="Original author text"),
              dict(asset_id="unlabelled", asset_type="text", source_table_id="books", source_row_id=2, content="Seller description"),
              dict(asset_id="cover", asset_type="image", source_table_id="books", source_row_id=1, content=None),
              dict(asset_id="policy", asset_type="text", source_table_id="sellers", source_row_id=1, content="Policy")]
    changes = add_book_title_context(tables, assets)
    assert assets[0]["content"] == "Book title: A book\n\nOriginal author text"
    assert assets[1]["content"] == "Book title: Another book\n\nSeller description"
    assert assets[2]["content"] is None and assets[3]["content"] == "Policy"
    assert {r["asset_id"] for r in changes} == {"biography", "unlabelled"}


def test_series_metadata_removal_preserves_main_title_and_other_qualifiers():
    assert without_series_notes("Computer Architecture (The Publisher Series in Computing)") == "Computer Architecture"
    assert without_series_notes("Java (2nd edition) (Java Series)") == "Java (2nd edition)"
    for title in ("Time Series Analysis", "Book (2nd edition)", "(A Book Series)"):
        assert without_series_notes(title) == title


def test_title_grouping_keeps_all_source_cells_and_sizes_without_labels():
    from mmdd_dataset.abebooks_grouping import group_book_sources
    tables = []
    for number in range(3):
        tables.append({"source_table_id": f"s{number}", "columns": [{"column_name": "title"}],
            "rows": [{"row_id": number * 3 + i, "cells": [{"column_name": "title", "text": title}]}
                     for i, title in enumerate(("network routing protocols", "database query transactions", "graphics rendering pixels"))]})
    original = json.dumps(tables)
    grouped, mapping, report = group_book_sources(tables)
    assert json.dumps(tables) == original
    assert report["sizes_before"] == report["sizes_after"] == [3, 3, 3]
    assert report["title_cosine_after"] > report["title_cosine_before"]
    assert not report["labels_used_for_grouping"]
    before = {r["row_id"]: r["cells"] for t in tables for r in t["rows"]}
    assert before == {r["row_id"]: r["cells"] for t in grouped for r in t["rows"]}
    assert len(mapping) == 9
    assert group_book_sources(tables)[1] == mapping
    for table in tables:
        table["columns"].append({"column_name": "authors"})
        for i, row in enumerate(table["rows"]):
            row["cells"].append({"column_name": "authors", "text": ["Jane Austen", "Mary Shelley", "Charles Dickens"][i]})
    grouped, _, report = group_book_sources(tables, field="authors")
    assert report["authors_cosine_after"] > report["authors_cosine_before"]
    assert all(len({r["cells"][1]["text"] for r in t["rows"]}) == 1 for t in grouped)


def test_regrouped_fact_replays_current_location_but_preserves_original_identity():
    record = fact("regrouped", 1, "a")
    record["annotation_provenance"] = {"original_source_table_id": "original"}
    assert recovery_key(record)[0] == "original"
    task = SimpleNamespace(source_table_id="regrouped", source_row_id=1, asset={"asset_id": "a"},
                           candidate_attribute_names=["authors"])
    assert replay_attributes(task, [record]) == [{"name": "authors", "value": "Author", "evidence": ""}]


def test_semantic_source_grouping_aligns_vectors_and_rejects_changed_titles(tmp_path):
    import numpy as np
    from mmdd_dataset.abebooks_grouping import group_book_sources

    tables = [{"source_table_id": f"s{i}", "columns": [{"column_name": "title"}],
               "rows": [{"row_id": j, "cells": [{"column_name": "title", "text": f"book{i}{j}"}]}
                        for j in range(2)]} for i in range(2)]
    path = tmp_path / "vectors.npz"
    # Cache order differs from source order; vector identity must follow the row key.
    np.savez(path, source_ids=np.array(["s1", "s0", "s1", "s0"]), row_ids=np.array([1, 0, 0, 1]),
             titles=np.array(["book11", "book00", "book10", "book01"]),
             vectors=np.array([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=np.float32))
    grouped, mapping, report = group_book_sources(tables, title_embeddings=path)
    assert mapping["s0", 0] == mapping["s1", 0]
    assert mapping["s0", 1] == mapping["s1", 1]
    assert mapping["s0", 0] != mapping["s0", 1]
    assert sorted(len(t["rows"]) for t in grouped) == [2, 2]
    assert report["semantic_title_cosine_after"] > report["semantic_title_cosine_before"]
    tables[0]["rows"][0]["cells"][0]["text"] = "changed title"
    with pytest.raises(ValueError, match="inputs differ"):
        group_book_sources(tables, title_embeddings=path)


def test_publisher_partition_keeps_every_row_and_separates_exact_value_domains():
    from mmdd_dataset.abebooks_grouping import partition_books_by_publisher
    tables = [{"source_table_id": f"s{i}", "columns": [{"column_name": "title"}, {"column_name": "publisher"}],
               "rows": [{"row_id": j, "cells": [{"column_name": "title", "text": f"Book{i}{j}"},
                         {"column_name": "publisher", "text": publisher}]} for j, publisher in
                        enumerate(("Publisher A", "Publisher B", "Publisher A"))]} for i in range(2)]
    before = json.dumps(tables)
    grouped, mapping, report = partition_books_by_publisher(tables)
    assert json.dumps(tables) == before
    assert report["sizes_after"] == [2, 4]
    assert len(mapping) == 6
    assert {tuple(sorted({r["cells"][1]["text"] for r in t["rows"]})) for t in grouped} == {
        ("Publisher A",), ("Publisher B",)}
    assert {r["cells"][0]["text"] for t in grouped for r in t["rows"]} == {
        r["cells"][0]["text"] for t in tables for r in t["rows"]}
    coalesced, mapping, report = partition_books_by_publisher(tables, min_rows=5)
    assert len(coalesced) == 1 and len(coalesced[0]["rows"]) == 6
    assert len(mapping) == 6
    assert report["coalesced_publishers"] == 2 and report["coalesced_rows"] == 6


def test_small_dataset_proportional_split_keeps_global_balance_and_all_sources():
    from mmdd_dataset.abebooks_rebalance import grouped_split
    queries = [{"table_id": f"q{i}", "source_table_id": f"s{i}"} for i in range(50)]
    kinds = {q["table_id"]: "implicit" if i < 25 else "explicit" for i, q in enumerate(queries)}
    splits = grouped_split(queries, kinds, 13, proportional=True)
    assert Counter(splits.values()) == {"train": 40, "dev": 5, "test": 5}
    assert Counter(kinds[q] for q in splits) == {"implicit": 25, "explicit": 25}
    assert Counter(kinds[q] for q in splits if splits[q] == "dev") == {"implicit": 2, "explicit": 3}
    assert Counter(kinds[q] for q in splits if splits[q] == "test") == {"implicit": 3, "explicit": 2}


def test_frozen_table_reuse_requires_identical_complete_inputs(tmp_path, monkeypatch):
    import prepare_abebooks_ablation_features as features
    from cache_stage1_features import _source_fingerprint

    old = {"object_id": "q_source_old", "object_type": "table", "embedding_role": "query",
           "table_parts": ["Columns: title", "Row: Book"]}
    new = {**old, "object_id": "q_source_new"}
    reference, run = tmp_path / "reference", tmp_path / "run"
    write_rows(reference / "data/tables.jsonl", [old])
    write_rows(run / "data/stage1_objects.jsonl", [new])
    for name, key in (("manifest.jsonl", "feature_path"), ("teacher_manifest.jsonl", "teacher_feature_path")):
        path = f"{key}/object.pt"
        target = reference / "table_encoder" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"frozen test feature")
        write_rows(reference / "table_encoder" / name, [{"object_id": old["object_id"],
            "object_type": "table", key: path, "source_fingerprint": _source_fingerprint(old)}])

    def compose(destination, source):
        for filename in ("FEATURE_COMPOSITION.json", "FRESH_INPUTS.json"):
            (destination / filename).write_text("{}")
    monkeypatch.setattr(features, "compose_reference_evidence", compose)
    features.reuse_identical_tables(run, reference)
    cached = read_rows(run / "table_encoder/manifest.jsonl")[0]
    assert cached["source_fingerprint"] == _source_fingerprint(new)
    assert (run / "table_encoder" / cached["feature_path"]).read_bytes() == b"frozen test feature"
    changed = tmp_path / "changed"
    write_rows(changed / "data/stage1_objects.jsonl", [{**new, "embedding_role": "target"}])
    with pytest.raises(AssertionError):
        features.reuse_identical_tables(changed, reference)
    assert not (changed / "table_encoder").exists()


def test_changed_text_composition_replaces_both_tiers_and_reuses_only_identical_images(tmp_path):
    import numpy as np
    import torch
    from mmdd_stage1 import content as features
    from prepare_abebooks_ablation_features import compose_reference_evidence

    reference, run = tmp_path / "reference", tmp_path / "run"
    old_text = {"object_id": "text", "object_type": "text", "text": "original"}
    new_text = {**old_text, "text": "Book title: Example\n\noriginal"}
    image = {"object_id": "image", "object_type": "image", "image_path": "unchanged.png"}
    table = {"object_id": "table", "object_type": "table", "table_parts": ["title", "Example"]}
    write_rows(reference / "data/stage1_objects.jsonl", [old_text, image])
    write_rows(run / "data/stage1_objects.jsonl", [new_text, image, table])
    write_rows(run / "data/tables.jsonl", [table])
    write_rows(run / "data/changed_text.jsonl", [new_text])

    def cache(path, records):
        path.mkdir(parents=True)
        base, teacher = [], []
        for record, value in records:
            oid = record["object_id"]
            torch.save({"embedding": torch.full((4096,), value)}, path / f"{oid}.pt")
            torch.save({"hidden_states": torch.full((2, 4096), value)}, path / f"{oid}.teacher.pt")
            base.append({"object_id": oid, "object_type": record["object_type"], "feature_path": f"{oid}.pt"})
            teacher.append({"object_id": oid, "object_type": record["object_type"], "teacher_feature_path": f"{oid}.teacher.pt"})
        write_rows(path / "manifest.jsonl", base)
        write_rows(path / "teacher_manifest.jsonl", teacher)

    cache(reference / "encoder", [(old_text, 1.), (image, 2.)])
    cache(run / "table_encoder", [(table, 3.)])
    cache(run / "text_encoder", [(new_text, 4.)])
    features.write_chunk(reference / "features/content/chunks", 0, [
        ("text", "text", np.ones((2, 4096))), ("image", "image", np.full((2, 4096), 2.))])
    compose_reference_evidence(run, reference)
    index = json.loads((run / "features/z/z_index.json").read_text())
    vectors = np.load(run / "features/z/z.f32.npy")
    store = features.ContentStore(run / "features/content")
    for oid, value in (("text", 4.), ("image", 2.), ("table", 3.)):
        assert np.all(vectors[index["ids"].index(oid)] == value)
        assert torch.all(store.get(oid) == value)
    report = json.loads((run / "FEATURE_COMPOSITION.json").read_text())
    assert report["changed_text_freshly_encoded"] == 1
    assert not report["all_evidence_inputs_equal"]


def test_natural_author_names_preserve_ambiguous_lists_and_qualifiers():
    assert natural_author_names("Johnsonbaugh, Richard ; Schaefer, Marcus") == "Richard Johnsonbaugh; Marcus Schaefer"
    assert natural_author_names("Ben-Ari, M.") == "M. Ben-Ari"
    for value in ("Pohl, Ira, Kelley, Al", "Zelkovitz, Marvin W. Ed.", "Surname, Jr.", "0", "Walter Savitch"):
        assert natural_author_names(value) == value


def fact(sid, row, aid, name="authors", value="Author"):
    return {"recovery_id": f"r-{sid}-{row}", "source_table_id": sid, "source_row_id": row,
            "evidence": {"asset_id": aid}, "recovered_attribute": {"column_name": name, "value": value}}


def test_replay_uses_only_matching_approved_row_asset_and_attribute():
    task = SimpleNamespace(source_table_id="s", source_row_id=1, asset={"asset_id": "a"},
                           candidate_attribute_names=["authors"])
    records = [fact("s", 1, "a"), fact("s", 2, "a"), fact("t", 1, "a"),
               fact("s", 1, "b"), fact("s", 1, "a", "publisher")]
    assert replay_attributes(task, records) == [{"name": "authors", "value": "Author", "evidence": ""}]
    assert replay_attributes(task, []) == []


def test_source_rebuild_preserves_facts_and_uses_disjoint_balanced_sources(tmp_path):
    source = tmp_path / "input"
    names = ["title", "authors", "publisher", "publication_year", "price"]
    tables, entities, assets, recoveries, qrels, decisions = [], [], [], [], [], []
    for number in range(6):
        sid = f"s{number}"
        rows = []
        for row in range(7):
            key, aid = f"entity-{sid}-{row}", f"asset-{sid}-{row}"
            values = [f"Book {number} {row}", f"Family{chr(65 + row)}, Given{chr(65 + number)}", f"Publisher {row}",
                      str(2000 + row), str(row + 1)]
            rows.append({"row_id": row, "cells": [{"column_index": i, "column_name": name,
                "text": values[i], "raw": values[i], "wiki_title": key if i == 0 else None,
                "has_wiki_link": i == 0} for i, name in enumerate(names)]})
            entities.append({"entity_id": key, "wiki_title": key, "appears_in": [
                {"source_table_id": sid, "row_id": row, "column_index": 0}]})
            assets.append({"asset_id": aid, "asset_type": "text", "entity_id": key,
                           "content": f"Book {number} {row} written by {values[1]}"})
            if number < 3 and row < 2:
                recoveries.append(fact(sid, row, aid, value=values[1]))
        tables.append({"source_table_id": sid, "provenance_builder": "abebooks_mm_joinability_dataset",
            "source_file": "synthetic", "num_rows": 7, "num_cols": len(names), "rows": rows,
            "columns": [{"column_index": i, "column_name": n} for i, n in enumerate(names)],
            "metadata": {"candidate_entity_columns": [0]}})
        decisions.append({"source_table_id": sid, "reason": "queryable" if number < 3 else "no_recovery"})
        if number < 3:
            qrels.append({"query_table_id": f"q{number}", "target_table_id": f"t{number}",
                "source_table_id": sid, "rel": 3, "reason": "model_recoverable_join_column"})
    artifacts = {"source_tables": tables, "entities": entities, "bridge_assets": assets,
                 "evidence_recoveries": recoveries, "table_asset_links": [],
                 "query_tables": [], "data_lake_tables": [], "attribute_extractions": []}
    manifest = {"artifacts": {}, "single_files": {"qrels": "qrels.jsonl",
                "table_queryability_decisions": "decisions.jsonl"}}
    for name, records in artifacts.items():
        path = f"{name}/part-00000.jsonl"
        write_rows(source / path, records)
        manifest["artifacts"][name] = {"shards": [{"path": path}]}
    write_rows(source / "qrels.jsonl", qrels)
    write_rows(source / "decisions.jsonl", decisions)
    (source / "dataset_manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "output"
    report = rebuild_from_sources(source, output, keep=set(names) - {"price"})
    assert report["source_unchanged"]
    assert report["implicit_queries"] == report["explicit_queries"] == 3
    assert report["source_coverage"] == 6
    assert all(c == {"implicit": 1, "explicit": 1} for c in report["split_counts"].values())
    assert {recovery_key(r) for r in read_rows(output / "evidence_recoveries/part-00000.jsonl")} <= {
        recovery_key(r) for r in recoveries}
    qs = read_rows(output / "query_tables/part-00000.jsonl")
    assert len({q["table_id"] for q in qs}) == 6
    assert Counter(q["split"] for q in qs) == {"train": 2, "dev": 2, "test": 2}
    for name in ("source_tables", "query_tables", "data_lake_tables"):
        assert all(c["column_name"] != "price" for t in read_rows(output / name / "part-00000.jsonl")
                   for c in t["columns"])
    with pytest.raises(FileExistsError):
        rebuild_from_sources(source, output, keep=set(names))
    readable = tmp_path / "readable"
    renamed = rebuild_from_sources(source, readable, keep=set(names) - {"price"}, readable_headers=True)
    assert renamed["split_counts"] == report["split_counts"]
    assert renamed["retained_approved_facts"] == report["retained_approved_facts"]
    assert {recovery_key(r) for r in read_rows(readable / "evidence_recoveries/part-00000.jsonl")} == {
        recovery_key(r) for r in read_rows(output / "evidence_recoveries/part-00000.jsonl")}
    for old, new in zip(read_rows(output / "source_tables/part-00000.jsonl"),
                        read_rows(readable / "source_tables/part-00000.jsonl")):
        assert new["columns"][0]["column_name"] == "Book title"
        assert [[c["text"] for c in r["cells"]] for r in old["rows"]] == [
            [c["text"] for c in r["cells"]] for r in new["rows"]]
    natural = tmp_path / "natural"
    normalized = rebuild_from_sources(source, natural, keep=set(names) - {"price"}, natural_authors=True)
    assert normalized["changed_author_cells"] == 42
    assert normalized["split_counts"] == report["split_counts"]
    new_facts = read_rows(natural / "evidence_recoveries/part-00000.jsonl")
    assert {recovery_key(r) for r in new_facts} == {
        recovery_key(r) for r in read_rows(output / "evidence_recoveries/part-00000.jsonl")}
    for record in new_facts:
        original = record["annotation_provenance"]["original_attribute_value"]
        assert record["recovered_attribute"]["value"] == natural_author_names(original)
        assert record["recovered_attribute"]["value"] != original
    pruned = tmp_path / "pruned_input"
    shutil.copytree(source, pruned)
    without_anchor = [project_table(t, set(names) - {"title"}, {i: i - 1 for i in range(1, 5)})
                      for t in tables]
    write_rows(pruned / "source_tables/part-00000.jsonl", without_anchor)
    write_rows(pruned / "entities/part-00000.jsonl", [{**e, "appears_in": []} for e in entities])
    restored = tmp_path / "restored"
    restored_report = rebuild_from_sources(pruned, restored, keep=set(names) - {"price"}, source_reference=source)
    assert restored_report["source_reference_unchanged"]
    assert restored_report["implicit_queries"] == restored_report["explicit_queries"] == 3
    assert {recovery_key(r) for r in read_rows(restored / "evidence_recoveries/part-00000.jsonl")} == {
        recovery_key(r) for r in read_rows(output / "evidence_recoveries/part-00000.jsonl")}
    series_source = tmp_path / "series_input"
    shutil.copytree(source, series_source)
    series_tables = read_rows(series_source / "source_tables/part-00000.jsonl")
    for table in series_tables:
        for row in table["rows"]:
            row["cells"][0]["text"] += " (Library Series)"
            row["cells"][0]["raw"] = row["cells"][0]["text"]
    write_rows(series_source / "source_tables/part-00000.jsonl", series_tables)
    no_series = tmp_path / "no_series"
    stripped = rebuild_from_sources(series_source, no_series, keep=set(names) - {"price"}, strip_series_notes=True)
    assert stripped["changed_title_cells"] == 42
    assert stripped["retained_approved_facts"] == report["retained_approved_facts"]
    assert stripped["split_counts"] == report["split_counts"]
    assert {recovery_key(r) for r in read_rows(no_series / "evidence_recoveries/part-00000.jsonl")} == {
        recovery_key(r) for r in recoveries}
    for name in ("source_tables", "query_tables"):
        assert all("Library Series" not in c["text"] for t in read_rows(no_series / name / "part-00000.jsonl")
                   for row in t["rows"] for c in row["cells"])

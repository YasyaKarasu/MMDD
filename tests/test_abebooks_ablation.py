"""Data ablations must preserve joins, column values, and gold content aliases."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_dataset.abebooks_ablation import (commercial_text_assets, duplicate_book_image_assets, global_join_columns, make_view, nongold_hubs, nongold_text_hubs,
                                           nonpositive_source_assets, project_table, read_rows, unanchored_text_assets,
                                           unlabelled_text_assets, unsupported_source_assets, write_rows)


def test_hubs_protect_gold_content_alias_and_keep_infrequent_evidence():
    canonical = {"gold": "alias", "alias": "alias", "hub": "hub", "hub_copy": "hub", "rare": "rare"}
    popularity = [{"evidence_id": "alias", "train_queries": 100},
                  {"evidence_id": "hub", "train_queries": 12},
                  {"evidence_id": "rare", "train_queries": 11}]
    assert nongold_hubs(popularity, canonical, {"gold"}, 12) == {"hub", "hub_copy"}


def test_text_hubs_keep_images_rare_text_and_all_gold_aliases():
    canonical = {"gold": "gold", "alias": "gold", "text": "text", "copy": "text",
                 "image": "image", "rare": "rare"}
    assets = [{"asset_id": aid, "asset_type": "image" if aid == "image" else "text"}
              for aid in canonical]
    popularity = [{"evidence_id": aid, "train_queries": count}
                  for aid, count in [("alias", 30), ("text", 12), ("image", 30), ("rare", 11)]]
    assert nongold_text_hubs(assets, popularity, canonical, {"gold"}, 12) == {"text", "copy"}
    assert unlabelled_text_assets(assets, [{"evidence": {"asset_id": "gold"}}], canonical) == {
        "text", "copy", "rare"}


def test_book_image_copies_preserve_labels_distinct_covers_and_other_editions(tmp_path):
    from PIL import Image

    fields = ["title", "authors", "publisher", "publication_year"]
    sources = [{"source_table_id": "s", "rows": [
        {"row_id": i, "cells": [{"column_name": name, "text": value} for name, value in
                                zip(fields, ["Book", "Author", "Publisher", year])]}
        for i, year in enumerate(["2001", "2002"])]}]
    assets = []
    for aid, color, row in [("gold", 100, 0), ("alias", 100, 0), ("near", 102, 0),
                            ("distinct", 150, 0), ("edition", 102, 1), ("other_gold", 101, 0)]:
        path = tmp_path / f"{aid}.png"
        Image.new("RGB", (40, 60), (color, color, color)).save(path)
        assets.append({"asset_id": aid, "asset_type": "image", "local_path": str(path),
                       "source_table_id": "s", "source_row_id": row})
    canonical = {a["asset_id"]: a["asset_id"] for a in assets}
    canonical["alias"] = "gold"
    recoveries = [{"evidence": {"asset_id": aid}} for aid in ("gold", "other_gold")]
    assert duplicate_book_image_assets(sources, assets, recoveries, canonical) == {"near"}


def test_supported_source_filter_keeps_unlabelled_neighbors_and_gold_aliases():
    assets = [{"asset_id": aid, "source_table_id": sid} for aid, sid in
              [("gold", "supported"), ("neighbor", "supported"),
               ("alias", "other"), ("unlabelled", "other")]]
    recovery = {"source_table_id": "supported", "evidence": {"asset_id": "gold"}}
    canonical = {"gold": "gold", "alias": "gold", "neighbor": "neighbor", "unlabelled": "unlabelled"}
    assert unsupported_source_assets(assets, [recovery], canonical) == {"unlabelled"}


def test_positive_source_filter_keeps_explicit_neighbors_and_protects_all_gold_aliases():
    assets = [{"asset_id": aid, "source_table_id": sid} for aid, sid in
              [("gold", "implicit"), ("explicit_neighbor", "explicit"),
               ("alias", "other"), ("negative", "negative_source")]]
    qrels = [{"source_table_id": sid, "rel": rel} for sid, rel in
             [("implicit", 3), ("explicit", 1), ("negative_source", 0)]]
    canonical = {a["asset_id"]: a["asset_id"] for a in assets}
    canonical["alias"] = "gold"
    assert nonpositive_source_assets(assets, qrels, [{"evidence": {"asset_id": "gold"}}], canonical) == {"negative"}


def test_commercial_filter_preserves_labels_aliases_and_unlabelled_book_evidence():
    sources = {"gold": "abebooks_description", "alias": "abebooks_shipping_policy",
               "sale": "abebooks_seller_policy", "condition": "abebooks_description",
               "synopsis": "abebooks_synopsis", "author": "abebooks_about_author",
               "cover": "abebooks_seller_cover"}
    assets = [{"asset_id": aid, "source": source} for aid, source in sources.items()]
    canonical = {aid: aid for aid in sources}
    canonical["alias"] = "gold"
    assert commercial_text_assets(assets, [{"evidence": {"asset_id": "gold"}}], canonical) == {
        "sale", "condition"}


def test_title_anchors_keep_grounded_text_images_and_labelled_aliases():
    sources = [{"source_table_id": "s", "rows": [
        {"row_id": i, "cells": [{"column_name": "title", "text": title}]}
        for i, title in enumerate(["Common Algebra Geometry", "Common Networks Security"])]}]
    texts = {"grounded": "Algebra and geometry explained", "generic": "Common book for students",
             "partial": "Some algebra", "wrong_book": "Networks security", "gold": "Book in good condition"}
    assets = [{"asset_id": aid, "asset_type": "text", "content": text,
               "source_table_id": "s", "source_row_id": 0} for aid, text in texts.items()]
    assets.extend([{"asset_id": "image", "asset_type": "image"},
                   {"asset_id": "alias", "asset_type": "text"}])
    canonical = {a["asset_id"]: a["asset_id"] for a in assets}
    canonical["alias"] = "gold"
    removed = unanchored_text_assets(sources, assets, [{"evidence": {"asset_id": "gold"}}], canonical)
    assert removed == {"generic", "partial", "wrong_book"}


def test_global_names_keep_column_used_in_another_source():
    qrels = [{"query_table_id": "q", "target_table_id": "t", "rel": 1,
              "source_table_id": "other_source", "join_attribute": {"column_name": "authors"}}]
    keep = global_join_columns(qrels, []) | {"title"}
    table = {"source_table_id": "s", "columns": [
        {"column_index": 0, "column_name": "title"},
        {"column_index": 1, "column_name": "language"},
        {"column_index": 2, "column_name": "authors"}], "rows": [{"row_id": 1, "cells": [
        {"column_index": 0, "column_name": "title", "text": "Book"},
        {"column_index": 1, "column_name": "language", "text": "English"},
        {"column_index": 2, "column_name": "authors", "text": "Writer"}]}]}
    result = project_table(table, keep, {0: 0, 2: 1})
    assert [c["column_name"] for c in result["columns"]] == ["title", "authors"]
    assert result["rows"][0]["cells"][1] == {"column_index": 1, "column_name": "authors", "text": "Writer"}
    assert len(table["columns"]) == 3


def test_view_remaps_source_and_local_indices_and_preserves_join(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "view"
    cols = [{"column_index": i, "column_name": name} for i, name in enumerate(["title", "language", "authors"])]
    cells = [{**c, "text": value} for c, value in zip(cols, ["Book", "English", "Writer"])]
    table = {"source_table_id": "s", "num_cols": 3, "columns": cols, "rows": [{"row_id": 1, "cells": cells}]}
    # Target local order differs from source order; join_col remains SOURCE based.
    target = {"table_id": "t", "source_table_id": "s", "columns": [
        {"column_index": i, "source_column_index": j, "column_name": cols[j]["column_name"]}
        for i, j in enumerate([1, 2])], "rows": [{"row_id": 0, "cells": [
        {**cells[j], "column_index": i, "source_column_index": j} for i, j in enumerate([1, 2])]}],
        "source_column_indices": [1, 2], "join_col": 2, "join_col_name": "authors"}
    query = {"table_id": "q", "source_table_id": "s", "columns": [
        {**cols[0], "source_column_index": 0}], "rows": [{"row_id": 0, "cells": [{**cells[0], "source_column_index": 0}]}],
        "source_column_indices": [0], "query_entity_col": 0, "query_entity_col_name": "title",
        "hidden_attributes": [{"column_name": "authors", "source_column_index": 2}]}
    qrels = [{"query_table_id": "q", "target_table_id": "t", "rel": 3, "split": "test",
        "source_table_id": "s", "join_attribute": {"source_column_index": 2, "column_name": "authors"}}]
    recovery = {"query_table_id": "q", "target_table_id": "t", "source_table_id": "s", "query_row_id": 0,
        "query_entity": {"entity_column_index": 0, "entity_column_name": "title"},
        "recovered_attribute": {"column_index": 2, "column_name": "authors", "value": "Writer"},
        "evidence": {"asset_id": "gold"}}
    raw = {"table_id": "raw", "source_table_id": "s", "source_table_ref": {
        "artifact": "source_tables", "source_table_id": "s"}}
    data = {"source_tables": [table], "query_tables": [query], "data_lake_tables": [target, raw],
        "evidence_recoveries": [recovery], "entities": [], "attribute_extractions": [],
        "table_asset_links": [{"source_table_id": "s", "column_index": 0, "asset_ids": ["gold", "hub"]}],
        "bridge_assets": [{"asset_id": "gold"}, {"asset_id": "hub"}]}
    manifest = {"artifacts": {}, "single_files": {"qrels": "qrels.jsonl", "splits": "splits.json"}}
    for name, rows in data.items():
        relative = f"{name}/part-00000.jsonl"
        write_rows(source / relative, rows)
        manifest["artifacts"][name] = {"shards": [{"path": relative, "records": len(rows)}]}
    write_rows(source / "qrels.jsonl", qrels)
    (source / "splits.json").write_text('{"test":["q"]}')
    (source / "dataset_manifest.json").write_text(json.dumps(manifest))
    report = make_view(source, dest, prune_columns=True, removed_assets={"hub"})
    out_target = read_rows(dest / "data_lake_tables/part-00000.jsonl")[0]
    assert out_target["join_col"] == 1
    assert out_target["columns"] == [{"column_index": 0, "source_column_index": 1, "column_name": "authors"}]
    assert out_target["rows"][0]["cells"][0]["text"] == "Writer"
    assert read_rows(dest / "query_tables/part-00000.jsonl")[0]["hidden_attributes"][0]["source_column_index"] == 1
    assert read_rows(dest / "qrels.jsonl")[0]["join_attribute"]["source_column_index"] == 1
    r = read_rows(dest / "evidence_recoveries/part-00000.jsonl")[0]
    assert r["recovered_attribute"] == {"column_index": 1, "column_name": "authors", "value": "Writer"}
    assert r["evidence"]["asset_id"] == "gold" and report["gold_evidence_removed"] == 0
    assert read_rows(dest / "table_asset_links/part-00000.jsonl")[0]["asset_ids"] == ["gold"]
    assert (source / "splits.json").read_bytes() == (dest / "splits.json").read_bytes()
    assert read_rows(dest / "data_lake_tables/part-00000.jsonl")[1] == raw
    filtered = tmp_path / "only_assets"
    make_view(source, filtered, prune_columns=False, removed_assets={"hub"})
    for name in ("source_tables", "query_tables", "data_lake_tables", "evidence_recoveries"):
        assert read_rows(filtered / name / "part-00000.jsonl") == data[name]
    assert read_rows(filtered / "qrels.jsonl") == qrels
    with pytest.raises(ValueError, match="gold evidence"):
        make_view(source, tmp_path / "bad", prune_columns=True, removed_assets={"gold"})

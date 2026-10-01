import hashlib
import json

import pytest

from mmdd_dataset.abebooks_ablation import read_rows, write_rows
from mmdd_dataset.abebooks_curation import (
    audit_recovery, author_key, curate_dataset, dataset_hashes, execute_author_join, judge_target,
    visible_author_leaks,
)


def row(rid, **values):
    return {"row_id": rid, "source_row_id": rid,
            "cells": [{"column_name": k, "column_index": i, "text": v}
                      for i, (k, v) in enumerate(values.items())]}


def table(tid, rows, source="books"):
    return {"table_id": tid, "object_id": tid, "source_table_id": source,
            "columns": [{"column_name": c["column_name"], "column_index": c["column_index"]}
                        for c in rows[0]["cells"]], "rows": rows, "split": "train"}


def fixture_data():
    source_rows = {
        ("books", 1): row(1, title="Book A", authors="Doe, Jane", publisher="Press A", publication_year="2001"),
        ("books", 2): row(2, title="Book B", authors="John Smith", publisher="Press B", publication_year="2002"),
        ("other", 3): row(3, title="Another book", authors="Jane Doe", publisher="Press C"),
        ("other", 4): row(4, title="Other book", authors="Alex Brown", publisher="Press D"),
    }
    query = table("q", [row(1, title="Book A"), row(2, title="Book B")])
    good = table("t", [row(1, authors="Doe, Jane", publisher="Press A"),
                       row(2, authors="John Smith", publisher="Press B")])
    wrong = table("wrong", [row(3, authors="Jane Doe", publisher="Press C"),
                            row(4, authors="Alex Brown", publisher="Press D")], "other")
    return source_rows, query, good, wrong


def test_author_key_preserves_complete_names_and_ambiguous_lists():
    assert author_key("Doe, Jane") == author_key("Jane Doe")
    assert author_key("Joe Kaplan, Ryan Dunn") == author_key("Joe Kaplan; Ryan Dunn")
    assert author_key("Michael D. Duffy") == author_key("Michael D Duffy")
    assert author_key("William Ford") != author_key("William Ford; William Topp")
    assert author_key("Smith") != author_key("Roderick W. Smith")
    assert not author_key("-") and not author_key("0")


def test_replayed_join_returns_wrong_matches_instead_of_filtering_them_with_gold():
    _, _, good, _ = fixture_data()
    assert execute_author_join({1: "John Smith"}, good) == {(1, 2)}
    assert execute_author_join({1: "no such author", 2: ""}, good) == set()


def test_same_author_different_book_is_not_a_positive_and_unknown_identity_stays_unknown():
    sources, query, good, wrong = fixture_data()
    judged = judge_target(query, good, sources)
    assert judged["status"] == "positive"
    assert judged["useful_query_rows"] == [1, 2]
    assert judge_target(query, wrong, sources)["reason"] == "author_join_expands_to_other_book_records"
    sources["other", 3]["cells"][0]["text"] = "Book A"
    assert judge_target(query, wrong, sources)["status"] == "unjudged"


def test_join_requires_added_values_and_rejects_constant_targets():
    sources, query, good, _ = fixture_data()
    no_added = table("authors_only", [row(1, authors="Doe, Jane"), row(2, authors="John Smith")])
    assert judge_target(query, no_added, sources)["reason"] == "no_nonempty_missing_attribute_added"
    good["rows"] = good["rows"][:1]
    assert judge_target(query, good, sources)["reason"] == "constant_target_author_column"


def test_partial_review_and_visible_name_are_not_treated_as_full_hidden_support():
    record = {"recovery_id": "r", "query_table_id": "q", "source_table_id": "s", "source_row_id": 1,
              "evidence": {"asset_id": "e"}, "recovered_attribute": {"column_name": "authors", "value": "William Ford"}}
    audit = audit_recovery(record, {"status": "full_value_supported", "observed_value": "William Ford; William Topp"})
    assert not audit["usable"]
    assert not audit_recovery(record, None)["usable"]
    sources, query, _, _ = fixture_data()
    query["rows"][0]["cells"][0]["text"] = "Book A by Jane Doe"
    assert visible_author_leaks(query, sources)[0]["matching_key"] == "jane doe"


def make_dataset(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    sources, query, good, wrong = fixture_data()
    alternative = table("alternative", [row(1, authors="Doe, Jane", publication_year="2001"),
                                         row(2, authors="John Smith", publication_year="2002")])
    query.update(target_table_ids=["t"], hidden_attributes=[{"column_name": "authors"}])
    recoveries, reviews, assets = [], [], []
    for rid, value, observed in ((1, "Doe, Jane", "Jane Doe"), (2, "John Smith", "John Smith")):
        assets.append({"asset_id": f"e{rid}", "asset_type": "text", "content": f"Written by {observed}",
                       "source_table_id": "books", "source_row_id": rid})
        recoveries.append({"recovery_id": f"r{rid}", "path_id": f"p{rid}", "query_table_id": "q",
                           "target_table_id": "t", "source_table_id": "books", "source_row_id": rid,
                           "query_row_id": rid, "split": "train", "path_nodes": [{"node_id": "q"}, {"node_id": f"e{rid}"}, {"node_id": "t"}],
                           "evidence": {"asset_id": f"e{rid}"}, "recovered_attribute": {"column_name": "authors", "value": value}})
        reviews.append({"recovery_id": f"r{rid}", "asset_id": f"e{rid}", "status": "full_value_supported",
                        "observed_value": observed, "content_sha256": hashlib.sha256(assets[-1]["content"].encode()).hexdigest()})
    data = {"source_tables": [{"source_table_id": sid, "rows": [r for (s, _), r in sources.items() if s == sid]}
                               for sid in ("books", "other")],
            "query_tables": [query], "data_lake_tables": [good, alternative, wrong],
            "evidence_recoveries": recoveries, "bridge_assets": assets}
    manifest = {"artifacts": {}, "single_files": {"qrels": "qrels.jsonl"}}
    for name, records in data.items():
        path = f"{name}/part-00000.jsonl"
        write_rows(root / path, records)
        manifest["artifacts"][name] = {"shards": [{"path": path, "records": len(records)}]}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest))
    write_rows(root / "qrels.jsonl", [{"query_table_id": "q", "target_table_id": "t", "rel": 3,
                                      "join_attribute": {"column_name": "authors"}, "reason": "model_recoverable_join_column"}])
    (root / "splits.json").write_text(json.dumps({"query_splits": {"q": "train"}}))
    (root / "retrieval_catalog.json").write_text(json.dumps({"query_ids": ["q"], "target_ids": ["t", "alternative", "wrong"],
        "query_kinds": {"q": "implicit", "obsolete": "explicit"},
        "query_splits": {"q": "train", "obsolete": "test"},
        "qrels": {"q": ["t"], "obsolete": ["wrong"]}}))
    reviews_path = tmp_path / "reviews.jsonl"
    write_rows(reviews_path, reviews)
    return root, reviews_path


def test_copy_preserves_inputs_adds_all_positive_views_and_replays_observed_values(tmp_path):
    root, reviews = make_dataset(tmp_path)
    before = dataset_hashes(root)
    output = tmp_path / "curated"
    report = curate_dataset(root, output, reviews)
    assert before == dataset_hashes(root)
    assert report["retained_queries"] == 1 and report["added_positive_pairs"] == 1
    assert report["candidate_hashes_unchanged"] and report["preserved_artifact_hashes_unchanged"]
    assert report["distinct_active_facts"] == 2 and report["active_recovery_paths"] == 4
    assert {r["target_table_id"] for r in read_rows(output / "qrels.jsonl")} == {"t", "alternative"}
    assert read_rows(output / "query_tables/part-00000.jsonl")[0]["target_table_ids"] == ["t", "alternative"]
    catalog = json.loads((output / "retrieval_catalog.json").read_text())
    assert catalog["qrels"] == {"q": ["t", "alternative"]}
    assert catalog["query_kinds"] == {"q": "implicit"}
    assert catalog["query_splits"] == {"q": "train"}
    assert (output / "provenance/pre_author_curation/qrels.jsonl").read_bytes() == (root / "qrels.jsonl").read_bytes()
    assert len({r["recovery_id"] for r in read_rows(output / "evidence_recoveries/part-00000.jsonl")}) == 4
    for diagnostic in read_rows(output / "audit/oracle_and_replay.jsonl"):
        assert diagnostic["approved_value_replay_correct_pairs"] == 2
        assert diagnostic["swapped_evidence_correct_pairs"] == 0
        assert diagnostic["swapped_evidence_pairs"] == 2
    with pytest.raises(FileExistsError):
        curate_dataset(root, output, reviews)
    with pytest.raises(ValueError, match="outside"):
        curate_dataset(root, root, reviews)


def test_stale_content_review_is_rejected_before_copying(tmp_path):
    root, reviews = make_dataset(tmp_path)
    records = read_rows(reviews)
    records[0]["content_sha256"] = "changed"
    write_rows(reviews, records)
    with pytest.raises(ValueError, match="Review content changed"):
        curate_dataset(root, tmp_path / "curated", reviews)
    assert not (tmp_path / "curated").exists()


def test_unjudged_target_excludes_query_without_deleting_candidate_lake(tmp_path):
    root, reviews = make_dataset(tmp_path)
    tables = read_rows(root / "source_tables/part-00000.jsonl")
    tables[1]["rows"][0]["cells"][0]["text"] = "Book A"
    write_rows(root / "source_tables/part-00000.jsonl", tables)
    report = curate_dataset(root, tmp_path / "curated", reviews)
    assert report["retained_queries"] == 0 and report["candidate_tables"] == 3
    assert report["query_exclusion_counts"]["incomplete_candidate_judgments"] == 1
    exclusions = read_rows(tmp_path / "curated/audit/negative_sampling_exclusions.jsonl")
    assert set(exclusions[0]["forbidden_negative_ids"]) == {"t", "alternative", "wrong"}

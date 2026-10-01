import copy
import pytest

from mmdd_dataset.abebooks_publisher import collect_publisher_facts, publisher_key
from mmdd_dataset.abebooks_query_context import add_query_context, direct_context_targets

from mmdd_dataset.abebooks_standalone import (
    assign_splits, author_names, complementary_targets, cover_title_matches,
    judge_join, make_queries, make_supervision, reveal_authors_for_balance, select_disjoint_row_views,
    title_family_key, validate_queries, visible_author_hint,
)


def test_author_list_normalization_recognizes_catalog_order_without_partial_matching():
    assert author_names("Ousterhout, John K.") == author_names(["John K. Ousterhout"])
    assert author_names("Joe Kaplan, Ryan Dunn") == author_names(["Ryan Dunn", "Joe Kaplan"])
    assert author_names("Harold, Ward K., Williamson, Leigh, Kreger, Heather") == author_names(
        ["Ward K. Harold", "Leigh Williamson", "Heather Kreger"])
    assert author_names("Simon, A. R. and Shaffer, S. L.") == author_names(["A. R. Simon", "S. L. Shaffer"])
    assert author_names("William Ford") != author_names(["William Ford", "William Topp"])
    assert author_names("Mark Weiss") != author_names("Mark Allen Weiss")
    assert not author_names(["Joe Kaplan, Ryan Dunn", "Smith"])
    assert not author_names(["Joe Kaplan", "Joe Kaplan"])
    for value in ("-", "0", "Smith", "Loshin, Pete ; Loshin ; Loshin, Peter", "Jones, , James"):
        assert not author_names(value)


def test_title_check_requires_visible_book_title_overlap():
    assert cover_title_matches("NFS Illustrated", "NFS Illustrated (Technical Series)")
    assert not cover_title_matches("TCP/IP Illustrated", "NFS Illustrated")
    assert not cover_title_matches("", "NFS Illustrated")


def test_publisher_aliases_merge_brand_spellings_but_not_parent_imprints():
    assert publisher_key("Morgan Kaufmann Publishers, Inc.") == publisher_key("Morgan Kaufmann (edition 1)")
    assert publisher_key("Pearson Education (US), New Jersey") == publisher_key("Pearson")
    assert publisher_key("Academic Press, San Diego") == publisher_key("Elsevier Academic Press")
    assert publisher_key("Elsevier") != publisher_key("Butterworth-Heinemann")
    assert publisher_key("Pearson") != publisher_key("Prentice Hall")
    assert not publisher_key("-") and not publisher_key("NA")


def test_publisher_five_row_supervision_uses_publisher_values_and_no_author_facts():
    data = book_data()
    targets, _ = complementary_targets(data, "publisher")
    facts = [{"source_table_id": "books", "source_row_id": rid, "column_name": "publisher",
              "original_value": value, "observed_values": {f"p{rid}": value},
              "evidence_ids": [f"p{rid}"], "annotation_status": "model_assisted_publisher_brand_evidence",
              "strength": "codex_pixel_review_after_blind_local_proposal"}
             for rid, value in ((0, "Press A"), (5, "Press F"))]
    # An author fact for another row must not become publisher recovery evidence.
    facts.append({"source_table_id": "books", "source_row_id": 1, "evidence_ids": ["author"]})
    queries, judgments, _ = make_queries(data, targets, facts, "publisher", join_column="publisher")
    q = queries[0]
    q.update(split="train", split_group="books")
    assert q["join_column"] == "publisher" and len(q["rows"]) == 5
    assert q["hidden_attributes"][0]["source_column_index"] == 2
    assert q["hidden_attributes"][0]["recovered_rows"] == 2
    proposals = [{"asset_id": f"p{rid}", "attribute": "publisher", "content_sha256": str(rid)} for rid in (0, 5)]
    assets = [{"asset_id": f"p{rid}", "asset_type": "image"} for rid in (0, 5)]
    qrels, recoveries = make_supervision(queries, judgments, facts, assets, proposals)
    assert qrels[0]["join_attribute"]["column_name"] == "publisher"
    assert {r["recovered_attribute"]["column_index"] for r in recoveries} == {2}
    assert len(recoveries) == 2
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    result = validate_queries(queries, targets, judgments, recoveries, sources)
    assert result["observed_value_join_correct_pairs"] == 2
    assert result["swapped_value_join_correct_pairs"] == 0
    authors, _, _ = make_queries(data, targets, [], "publisher", excluded_rows={
        ("books", r["source_row_id"]) for r in q["rows"]})
    assert not authors


def test_publisher_alias_collision_rejects_wrong_book_expansion():
    data = book_data()
    data["source_tables"][0]["rows"][0]["cells"][2]["text"] = "Morgan Kaufmann"
    data["source_tables"][0]["rows"][1]["cells"][2]["text"] = "Morgan Kaufmann Publishers"
    targets, _ = complementary_targets(data, "collision")
    q = {"table_id": "q", "source_table_id": "books", "join_column": "publisher",
         "columns": [{"column_name": "title"}], "rows": [{"row_id": 0, "source_row_id": 0}]}
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    assert judge_join(q, targets[0], sources)["reason"] == "wrong_book_expansion"


def test_publisher_fact_requires_bound_pixels_and_rejects_parent_only_evidence():
    data = book_data()
    data["source_tables"][0]["rows"][0]["cells"][2]["text"] = "Butterworth-Heinemann"
    proposal = {"asset_id": "e", "source_table_id": "books", "source_row_id": 0,
                "attribute": "publisher", "source_answer_provided": False, "content_sha256": "hash",
                "response": {"title": "Book Alpha", "logo_text": ["Elsevier"]}}
    review = {"asset_id": "e", "content_sha256": "hash", "status": "supported",
              "observed_publisher": "Elsevier"}
    facts, audit = collect_publisher_facts(data, [proposal], [review])
    assert not facts and audit[0]["status"] == "visible_publisher_differs_from_source_brand"
    assert not collect_publisher_facts(data, [proposal], [])[0]
    review["content_sha256"] = "different"
    with pytest.raises(ValueError, match="evidence bytes"):
        collect_publisher_facts(data, [proposal], [review])


def test_kind_balancing_preserves_implicit_examples_for_each_join_attribute():
    queries, sources = [], []
    for i, column in enumerate(["publisher"] + ["authors"] * 6):
        data = book_data()
        sid = f"source{i}"
        data["source_tables"][0]["source_table_id"] = sid
        data["data_lake_tables"][0]["source_table_id"] = sid
        targets, _ = complementary_targets(data, sid)
        facts = [{"source_table_id": sid, "source_row_id": rid, "column_name": column} for rid in (0, 5)]
        qs, _, _ = make_queries(data, targets, facts, sid, join_column=column)
        qs[0].update(split="dev", split_group=sid)
        queries.extend(qs)
        sources.extend(data["source_tables"])
    reveal_authors_for_balance(queries, sources, "mixed")
    assert {q["join_column"] for q in queries if q["query_kind"] == "implicit"} == {"authors", "publisher"}
    assert sum(q["query_kind"] == "explicit" for q in queries) == 4


def test_visible_author_hint_includes_possessive_names():
    names = author_names("Celko, Joe")
    assert visible_author_hint("Joe Celko's SQL Programming Style", names)
    assert visible_author_hint("Joe Celko’s SQL Programming Style", names)
    assert not visible_author_hint("SQL Programming Style", names)


def book_data():
    names = ["title", "authors", "publisher", "publication_year"]
    columns = [{"column_index": i, "column_name": name} for i, name in enumerate(names)]
    rows = [{"row_id": i, "cells": [{**c, "text": str(value)} for c, value in zip(columns, values)]}
            for i, values in enumerate([
                ("Book Alpha", "Jane Doe", "Press A", 2001),
                ("Book Beta", "John Smith", "Press B", 2002),
                ("Book Gamma", "Alice Brown", "Press C", 2003),
                ("Book Delta", "Tom Green", "Press D", 2004),
                ("Book Epsilon", "Mary Black", "Press E", 2005),
                ("Book Zeta", "Peter Gray", "Press F", 2006),
            ])]
    source = {"source_table_id": "books", "source_file": "book", "columns": columns, "rows": rows}
    target = {"table_id": "old_target", "source_table_id": "books", "columns": copy.deepcopy(columns),
              "rows": [{**copy.deepcopy(r), "source_row_id": r["row_id"]} for r in rows]}
    return {"source_tables": [source], "data_lake_tables": [target]}


def test_context_addition_preserves_hidden_authors_and_query_membership():
    data = book_data()
    targets, _ = complementary_targets(data, "context")
    # A real source field absent from the target adds context, not a direct key.
    targets[0]["columns"] = [c for c in targets[0]["columns"] if c["column_name"] != "publication_year"]
    for r in targets[0]["rows"]:
        r["cells"] = [c for c in r["cells"] if c["column_name"] != "publication_year"]
    queries, _, _ = make_queries(data, targets, [{"source_table_id": "books", "source_row_id": i} for i in (0, 5)], "context")
    before = copy.deepcopy(queries[0])
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    q, audit = add_query_context(queries[0], data["source_tables"][0], targets, sources)
    assert queries[0] == before
    assert [c["column_name"] for c in q["columns"]] == ["title", "publication_year"]
    assert q["hidden_attributes"] == before["hidden_attributes"]
    assert q["source_row_indices"] == before["source_row_indices"]
    assert q["query_context_col_names"] == ["publication_year"]
    assert audit[-1]["populated_rows"] == 5
    assert not direct_context_targets(q, "publication_year", targets, sources)


def test_context_rejects_direct_year_key_and_uses_ambiguous_publisher_context():
    data = book_data()
    for row in data["source_tables"][0]["rows"]:
        row["cells"][2]["text"] = "Shared Press"
    targets, _ = complementary_targets(data, "context")
    queries, _, _ = make_queries(data, targets, [{"source_table_id": "books", "source_row_id": i} for i in (0, 5)], "context")
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    q, audit = add_query_context(queries[0], data["source_tables"][0], targets, sources)
    assert audit[0]["reason"] == "visible_context_already_enables_useful_join"
    assert [c["column_name"] for c in q["columns"]] == ["title", "publisher"]
    assert not direct_context_targets(q, "publisher", targets, sources)
    assert judge_join(q, targets[0], sources)["status"] == "positive"


def test_context_cannot_expose_hidden_publisher_or_author_hint():
    data = book_data()
    targets, _ = complementary_targets(data, "context")
    facts = [{"source_table_id": "books", "source_row_id": i, "column_name": "publisher"} for i in (0, 5)]
    queries, _, _ = make_queries(data, targets, facts, "context", join_column="publisher")
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    with pytest.raises(ValueError, match="No safe populated context"):
        add_query_context(queries[0], data["source_tables"][0], targets, sources)
    # Authors remain hidden even if a real publisher string happens to name one.
    for row in data["source_tables"][0]["rows"]:
        row["cells"][2]["text"] = "Jane Doe Press"
    targets, _ = complementary_targets(data, "hint")
    queries, _, _ = make_queries(data, targets, [{"source_table_id": "books", "source_row_id": i} for i in (0, 5)], "hint")
    with pytest.raises(ValueError, match="visible_context_leaks_hidden_join_value"):
        add_query_context(queries[0], data["source_tables"][0], targets, sources)


def test_projection_preserves_source_and_candidates_and_queries_never_reuse_rows():
    data = book_data()
    before = copy.deepcopy(data)
    targets, changes = complementary_targets(data, "new")
    assert data == before
    assert changes[0]["rows_preserved"]
    assert targets[0]["table_id"] != "old_target"
    assert {c["column_name"] for c in targets[0]["columns"]} == {"authors", "publisher", "publication_year"}
    facts = [{"source_table_id": "books", "source_row_id": i} for i in range(3)]
    queries, judgments, _ = make_queries(data, targets, facts, "new", implicit_rows=2, explicit_rows=2)
    assert len(queries) == 3
    assert sum(q["query_kind"] == "implicit" for q in queries) == 1
    selected = [r["source_row_id"] for q in queries for r in q["rows"]]
    assert len(selected) == len(set(selected)) == 6
    assert all(j["status"] == "positive" for j in judgments)


def test_five_row_query_with_two_recoveries_keeps_other_rows_unreviewed():
    data = book_data()
    targets, _ = complementary_targets(data, "five")
    facts = [{"source_table_id": "books", "source_row_id": rid, "original_value": name,
              "evidence_ids": [f"e{rid}"], "annotation_status": "model_assisted_full_value_evidence",
              "strength": "local_cover_with_source_agreement"}
             for rid, name in ((0, "Jane Doe"), (5, "Peter Gray"))]
    queries, judgments, _ = make_queries(data, targets, facts, "five")
    assert len(queries) == 1
    query = queries[0]
    query["split"] = "train"
    assert query["query_kind"] == "implicit" and len(query["rows"]) == 5
    assert len({r["source_row_id"] for r in query["rows"]}) == 5
    attr = query["hidden_attributes"][0]
    assert (attr["selected_rows"], attr["recovered_rows"], attr["required_recovered_rows"]) == (5, 2, 2)
    assert attr["unreviewed_rows"] == 3 and attr["recovered_value_ratio"] == 0.4
    assets = [{"asset_id": f"e{f['source_row_id']}", "asset_type": "image"} for f in facts]
    proposals = [{"asset_id": f"e{f['source_row_id']}", "response": {"authors": [f["original_value"]]},
                  "content_sha256": str(f["source_row_id"])} for f in facts]
    qrels, recoveries = make_supervision(queries, judgments, facts, assets, proposals)
    assert len(recoveries) == 2
    assert qrels[0]["join_attribute"]["selected_rows"] == 5
    assert qrels[0]["join_attribute"]["required_recovered_rows"] == 2
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    result = validate_queries(queries, targets, judgments, recoveries, sources)
    assert result["implicit_oracle_pairs"] == 5
    assert result["observed_value_join_pairs"] == result["observed_value_join_correct_pairs"] == 2
    assert result["unreviewed_implicit_rows"] == 3
    assert result["swapped_value_join_pairs"] == 2 and result["swapped_value_join_correct_pairs"] == 0
    with pytest.raises(AssertionError):
        validate_queries(queries, targets, judgments, recoveries[:1], sources)


def test_disjoint_five_row_views_reserve_two_recoveries_without_reusing_rows():
    supported = {0, 1, 2, 3}
    views = select_disjoint_row_views(list(range(10)), supported, 5, 2)
    assert len(views) == 2
    assert all(len(v) == 5 and len(set(v) & supported) >= 2 for v in views)
    assert sorted(rid for view in views for rid in view) == list(range(10))
    assert not select_disjoint_row_views(list(range(10)), {0}, 5, 2)


def test_explicit_queries_default_to_five_rows_with_visible_authors(tmp_path):
    data = book_data()
    targets, _ = complementary_targets(data, "five")
    queries, judgments, _ = make_queries(data, targets, [], "five")
    assert len(queries) == 1
    query = queries[0]
    assert query["query_kind"] == "explicit" and len(query["rows"]) == 5
    assert [c["column_name"] for c in query["columns"]] == ["title", "authors"]
    assert not query["hidden_attributes"]
    assert query["construction"]["minimum_recovered_rows"] == 0
    assert judgments[0]["status"] == "positive"
    query["split"] = "train"
    qrels, recoveries = make_supervision(queries, judgments, [], [], [])
    assert not recoveries
    from mmdd_dataset.abebooks_ablation import write_rows
    from mmdd_cqet_v4_1.config import Paths
    from mmdd_cqet_v4_1.labels import build_labels
    for name, records in [("query_tables/part-00000.jsonl", queries),
                           ("data_lake_tables/part-00000.jsonl", targets),
                           ("evidence_recoveries/part-00000.jsonl", []), ("qrels.jsonl", qrels)]:
        write_rows(tmp_path / name, records)
    paths = Paths(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "run")
    stats = build_labels(paths, {})
    assert stats["direct_pairs"] == len(qrels)


@pytest.mark.parametrize("view_count", [4, 5])
def test_balancing_exposes_authors_without_reusing_rows_or_leaving_recovery_labels(view_count):
    queries, sources, targets = [], [], []
    for i in range(view_count):
        data = book_data()
        sid = f"books{i}"
        data["source_tables"][0]["source_table_id"] = sid
        data["data_lake_tables"][0]["source_table_id"] = sid
        local_targets, _ = complementary_targets(data, sid)
        facts = [{"source_table_id": sid, "source_row_id": rid} for rid in (0, 5)]
        local_queries, _, _ = make_queries(data, local_targets, facts, sid)
        local_queries[0].update(split="train", split_group=sid)
        queries.extend(local_queries)
        sources.extend(data["source_tables"])
        targets.extend(local_targets)
    before = copy.deepcopy(queries)
    changes = reveal_authors_for_balance(queries, sources, "balanced")
    assert len(changes) == (view_count + 1) // 2
    assert [q["source_row_indices"] for q in queries] == [q["source_row_indices"] for q in before]
    assert len({(q["source_table_id"], r["source_row_id"]) for q in queries for r in q["rows"]}) == 5 * view_count
    source_rows = {(s["source_table_id"], r["row_id"]): r for s in sources for r in s["rows"]}
    target_map = {t["table_id"]: t for t in targets}
    for q in queries:
        assert len(q["rows"]) == 5 and q["split"] == "train"
        if q["query_kind"] == "implicit":
            assert q["hidden_attributes"][0]["recovered_rows"] == 2
            continue
        assert q["table_id"] == q["object_id"]
        assert not q["hidden_attributes"] and not q["construction"]["verified_source_row_ids"]
        assert [c["column_name"] for c in q["columns"]] == ["title", "authors"]
        assert all(len(r["cells"]) == 2 for r in q["rows"])
        js = [judge_join(q, target_map[tid], source_rows) for tid in q["target_table_ids"]]
        assert all(j["status"] == "positive" for j in js)
        _, recoveries = make_supervision([q], js, [], [], [])
        assert not recoveries


def test_full_lake_labels_include_all_views_but_reject_partial_coverage_and_wrong_books():
    data = book_data()
    targets, _ = complementary_targets(data, "new")
    alt = copy.deepcopy(targets[0])
    alt["table_id"] = "alternative"
    queries, judgments, _ = make_queries(data, targets + [alt], [], "new")
    assert all(len(q["target_table_ids"]) == 2 for q in queries)
    q = queries[0]
    sources = {("books", r["row_id"]): r for r in data["source_tables"][0]["rows"]}
    partial = copy.deepcopy(targets[0])
    partial["rows"] = [r for r in partial["rows"] if r["source_row_id"] == q["rows"][0]["source_row_id"]]
    assert judge_join(q, partial, sources)["reason"] == "does_not_complete_every_query_row"
    wrong = copy.deepcopy(targets[0])
    repeated = copy.deepcopy(partial["rows"][0])
    repeated.update(row_id=99, source_row_id=99)
    wrong["rows"].append(repeated)
    sources["books", 99] = {"cells": [{"column_name": "title", "text": "Unrelated Book"}]}
    assert judge_join(q, wrong, sources)["reason"] == "wrong_book_expansion"


def test_splits_group_editions_and_exact_qualified_evidence_without_model_scores():
    queries, sources, proposals = [], {}, []
    for i in range(12):
        sid = f"source{i}"
        queries.append({"table_id": f"q{i}", "source_table_id": sid, "query_kind": "implicit",
                        "rows": [{"source_row_id": 1}]})
        title = f"Book {i}" if i > 1 else ("Data Structures (2nd Edition)" if i else "Data Structures (3rd Edition)")
        sources[sid, 1] = {"cells": [{"column_name": "title", "text": title}]}
        proposals.append({"source_table_id": sid, "source_row_id": 1, "asset_id": f"e{i}",
                          "modality": "image", "content_sha256": "shared" if i in {2, 3} else str(i)})
    report = assign_splits(queries, sources, [], proposals)
    assert queries[0]["split_group"] == queries[1]["split_group"]
    assert queries[2]["split_group"] == queries[3]["split_group"]
    assert report["independent_components"] == 10
    assert all(report["counts"][s]["implicit"] >= 2 for s in ("train", "dev", "test"))
    assert title_family_key("Data Structures (2nd Edition)") == title_family_key("Data Structures, Third Edition")

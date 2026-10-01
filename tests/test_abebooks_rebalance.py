from collections import Counter, defaultdict
import copy

import pytest

from mmdd_dataset.abebooks_rebalance import grouped_split, query_kinds, select_balanced_standalone


def test_standalone_balance_preserves_all_implicit_and_split_boundaries_with_source_coverage():
    queries = []
    for split in ("train", "dev", "test"):
        for i in range(4):
            queries.append({"table_id": f"{split}_i{i}", "query_kind": "implicit", "split": split,
                            "source_table_id": f"{split}_source{i}"})
        for i in range(3):
            for j in range(3):
                queries.append({"table_id": f"{split}_e{i}_{j}", "query_kind": "explicit", "split": split,
                                "source_table_id": f"{split}_source{i}"})
    original = copy.deepcopy(queries)
    selected = select_balanced_standalone(queries)
    assert queries == original
    assert len(selected) == len({q["table_id"] for q in selected}) == 24
    assert {q["table_id"] for q in selected if q["query_kind"] == "implicit"} == {
        q["table_id"] for q in queries if q["query_kind"] == "implicit"}
    for split in ("train", "dev", "test"):
        subset = [q for q in selected if q["split"] == split]
        assert Counter(q["query_kind"] for q in subset) == {"implicit": 4, "explicit": 4}
        assert len({q["source_table_id"] for q in subset if q["query_kind"] == "explicit"}) == 3
    assert {q["table_id"] for q in selected} == {q["table_id"] for q in select_balanced_standalone(list(reversed(queries)))}


def test_standalone_balance_refuses_to_fill_a_shortage_by_repeating_queries():
    queries = [{"table_id": "i", "query_kind": "implicit", "split": "train", "source_table_id": "s"}]
    with pytest.raises(ValueError, match="balance train"):
        select_balanced_standalone(queries)


def test_grouped_split_keeps_sources_together_and_balances_each_split():
    queries = [{"table_id": f"q{i}_{kind}", "source_table_id": f"source{i}"}
               for i in range(20) for kind in ("implicit", "explicit")]
    kinds = {q["table_id"]: q["table_id"].split("_")[1] for q in queries}
    assignments = grouped_split(queries, kinds, 13)
    source_splits = defaultdict(set)
    for query in queries:
        source_splits[query["source_table_id"]].add(assignments[query["table_id"]])
    assert all(len(splits) == 1 for splits in source_splits.values())
    for split, expected in (("train", 16), ("dev", 2), ("test", 2)):
        assert Counter(kinds[q] for q, s in assignments.items() if s == split) == {
            "implicit": expected, "explicit": expected}
    assert assignments == grouped_split(list(reversed(queries)), kinds, 13)


def test_query_balance_counts_queries_instead_of_positive_pairs():
    qrels = [{"query_table_id": "implicit", "rel": 1, "reason": "model_recoverable_join_column"},
             {"query_table_id": "implicit", "rel": 3, "reason": "model_recoverable_join_column"},
             {"query_table_id": "explicit", "rel": 1, "reason": "explicit_visible_join_column"}]
    assert Counter(query_kinds(qrels).values()) == {"implicit": 1, "explicit": 1}

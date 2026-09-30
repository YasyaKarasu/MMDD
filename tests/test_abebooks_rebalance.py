from collections import Counter, defaultdict

from mmdd_dataset.abebooks_rebalance import grouped_split, query_kinds


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

from audit_abebooks_column_noise import distribution, join_impact, retrieval_overlap
from collections import Counter


def test_distribution_excludes_placeholders_from_collision():
    result = distribution(Counter({"good": 3, "fair": 1, "": 4, "-": 2}))
    assert result["missing_rate"] == 0.6
    assert result["collision_probability"] == 0.625
    assert result["top_values"][0]["share_populated"] == 0.75


def test_deleting_one_join_does_not_remove_query_with_another_gold():
    qrels = [{"query_table_id": "q", "target_table_id": t, "rel": 1,
              "join_attribute": {"column_name": c}, "split": "train", "reason": "explicit"}
             for t, c in [("a", "binding"), ("b", "authors")]]
    result = join_impact({"binding"}, qrels, [])
    assert result["positive_pairs"] == 1
    assert result["queries_losing_all_gold"] == 0


def test_overlap_uses_query_specific_gold_and_ignores_empty_values():
    def table(tid, value):
        return {"table_id": tid, "columns": [{"column_name": "condition"}],
                "rows": [{"cells": [{"column_name": "condition", "text": value}]}]}
    queries = [table("q1", "Good"), table("q2", "-")]
    targets = [table("a", "good"), table("b", "-")]
    qrels = [{"query_table_id": "q1", "target_table_id": "b", "rel": 1},
             {"query_table_id": "q2", "target_table_id": "a", "rel": 1}]
    pools = [{"query_id": q, "split": "train", "D100_ANN": ["a", "b"]}
             for q in ("q1", "q2")]
    result = retrieval_overlap(queries, targets, pools, qrels)
    assert result["pair_counts"]["top10_nongold"] == 2
    assert result["columns"]["top10_nongold"]["condition"]["share_any_value"] == 1
    assert result["columns"]["gold"]["condition"]["share_any_value"] == 0

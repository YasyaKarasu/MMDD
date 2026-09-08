import json
import sqlite3

import pytest

from build_stage1_r10_report import CHART_QUERY, markdown_cells
from summarize_stage1_r10_closeout import bootstrap_delta, validate_aggregates


def test_bootstrap_preserves_pair_weighted_estimand_and_source_groups():
    left = {"q1": {"v": 1.0, "n": 1}, "q2": {"v": 0.0, "n": 3}}
    right = {"q1": {"v": 0.0, "n": 1}, "q2": {"v": 0.0, "n": 3}}
    result = bootstrap_delta(left, right, {"q1": "g", "q2": "g"}, "v", weight_field="n", samples=100)
    assert result["groups"] == 1
    assert result["mean_difference"] == 0.25
    assert result["ci95_low"] == result["ci95_high"] == 0.25
    assert result["valid_replicates"] == 100


def test_bootstrap_rejects_unpaired_population_or_weights():
    left = {"q1": {"v": 1.0, "n": 1}}
    with pytest.raises(ValueError, match="complete query population"):
        bootstrap_delta(left, {}, {"q1": "g"}, "v", samples=10)
    with pytest.raises(ValueError, match="complete query population"):
        bootstrap_delta(left, left, {}, "v", samples=10)
    with pytest.raises(ValueError, match="unequal denominators"):
        bootstrap_delta(left, {"q1": {"v": 0.0, "n": 2}}, {"q1": "g"}, "v", weight_field="n", samples=10)


def test_bootstrap_seed_is_repeatable_and_zero_weight_is_excluded():
    left = {"q1": {"v": 1.0, "n": 1}, "q2": {"v": 1.0, "n": 0}}
    right = {"q1": {"v": 0.0, "n": 1}, "q2": {"v": 0.0, "n": 0}}
    groups = {"q1": "g1", "q2": "g2"}
    result = bootstrap_delta(left, right, groups, "v", weight_field="n", samples=100)
    assert result == bootstrap_delta(left, right, groups, "v", weight_field="n", samples=100)
    assert result["mean_difference"] == result["ci95_low"] == result["ci95_high"] == 1.0
    assert result["valid_replicates"] < result["samples"]


def test_aggregate_validation_rejects_duplicate_rows_even_if_unique_count_matches():
    row = {"query_id": "q1"}
    with pytest.raises(ValueError, match="duplicated or missing"):
        validate_aggregates({"results": {"f0": {"queries": 1, "per_query": [row, row]}}})


def test_chart_sql_independently_uses_query_means_and_pair_weighted_path_metrics():
    rows = [
        {"implicit_pair_count": n, "recall@10": recall, "recall@20": recall, "recall@50": recall,
         "mrr@50": recall, "valid_path_recall@10,4": recall, "row_support_coverage@10,4": recall / 5}
        for n, recall in ((1, 1.0), (3, 0.0))
    ]
    with sqlite3.connect(":memory:") as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("CREATE TABLE frozen_evaluations (model TEXT, rule TEXT, label TEXT, rank INTEGER, payload TEXT)")
        connection.execute("INSERT INTO frozen_evaluations VALUES (?, ?, ?, ?, ?)", ("m", "f0", "label", 1, json.dumps({"per_query": rows})))
        result = dict(connection.execute(CHART_QUERY).fetchone())
    assert result["queries"] == 2
    assert result["implicit_pairs"] == 4
    assert result["recall_at_10"] == 0.5
    assert result["valid_path_at_10_4"] == 0.25
    assert result["row_support_at_10_4"] == 0.05


def test_report_cells_preserve_percentages_and_uncertainty_text():
    assert markdown_cells("| `F4` | **7.895%** | 7.946 ± 0.073 |") == ["F4", "7.895%", "7.946 ± 0.073"]

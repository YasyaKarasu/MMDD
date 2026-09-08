from __future__ import annotations

import pytest

from run_stage1_r10_task_b import (
    _rank_pool,
    _selected_evidence,
    aggregation_configs,
)
from run_stage1_r10_task_b_advanced import (
    aggregation_configs as advanced_aggregation_configs,
    confidence_paths,
)
from run_stage1_r10_task_b_g5 import _evidence_candidates


def test_r10_lme_removes_equal_weak_path_count_inversion():
    pool = {
        "strong": [{"kind": "evidence", "path_score": 1.5}],
        "weak": [
            {"kind": "evidence", "evidence_id": f"e-{index}", "path_score": 1.0}
            for index in range(4)
        ],
    }
    configs = {row["name"]: row["aggregator"] for row in aggregation_configs(4)}

    lse = _rank_pool(pool, configs["full_lse_t1"])["evidence"]
    lme = _rank_pool(pool, configs["full_lme_t1"])["evidence"]

    assert [row["target_id"] for row in lse] == ["weak", "strong"]
    assert [row["target_id"] for row in lme] == ["strong", "weak"]


def test_advanced_scan_maps_only_evidence_edges_to_confidence():
    source = {
        "target": [
            {"kind": "direct", "path_score": 0.8},
            {
                "kind": "evidence",
                "evidence_id": "e",
                "query_evidence_score": 0.0,
                "evidence_target_score": 1.0,
                "path_score": 1.0,
            },
        ]
    }

    transformed = confidence_paths(source)

    assert transformed["target"][0]["path_score"] == pytest.approx(0.8)
    assert transformed["target"][1]["query_evidence_score"] == pytest.approx(0.5)
    assert transformed["target"][1]["evidence_target_score"] == pytest.approx(
        0.7310585786
    )
    assert source["target"][1]["query_evidence_score"] == 0.0


def test_advanced_scan_includes_g3_and_g4_grid():
    configs = {row["name"]: row for row in advanced_aggregation_configs(4)}

    assert configs["min_max"]["aggregator"].path_combination == "min"
    assert configs["product_max"]["aggregator"].path_combination == "product"
    assert configs["g3_r4"]["aggregator"].threshold == 0.0
    assert configs["g4_d0.5_r8"]["aggregator"].threshold == 0.5
    assert configs["g4_d0.5_r8"]["aggregator"].power == 8.0


def test_g5_content_deduplication_keeps_stronger_path():
    paths = [
        {
            "kind": "evidence",
            "evidence_id": "duplicate-low",
            "query_evidence_score": 0.0,
            "evidence_target_score": 0.0,
            "path_score": 0.0,
        },
        {
            "kind": "evidence",
            "evidence_id": "duplicate-high",
            "query_evidence_score": 1.0,
            "evidence_target_score": 1.0,
            "path_score": 2.0,
        },
    ]

    selected = _evidence_candidates(
        paths,
        content_keys={
            "duplicate-low": "same-content",
            "duplicate-high": "same-content",
        },
        top_l=20,
    )

    assert [row["evidence_id"] for row in selected] == ["duplicate-high"]


def test_explicit_g5_bundle_controls_valid_path_selection():
    target = {
        "selected_evidence_ids": ["coverage", "quality"],
        "paths": [
            {"kind": "evidence", "evidence_id": "quality", "path_score": 0.9},
            {"kind": "evidence", "evidence_id": "coverage", "path_score": 0.5},
        ],
    }

    assert _selected_evidence(target, 1) == ["coverage"]

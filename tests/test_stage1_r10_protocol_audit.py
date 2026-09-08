from __future__ import annotations

from audit_stage1_r10_protocol import _calibration_split, _support_mode
from materialize_stage1_r10_edge_labels import (
    _corrupted_pairs,
    _support_sets,
    label_edge_records,
)


def test_r10_calibration_split_is_deterministic_and_group_disjoint():
    groups = [f"source-{index}" for index in range(20)]

    fit, calibration = _calibration_split(groups, seed=13)
    repeated_fit, repeated_calibration = _calibration_split(reversed(groups), seed=13)

    assert (fit, calibration) == (repeated_fit, repeated_calibration)
    assert len(calibration) == 2
    assert set(fit).isdisjoint(calibration)
    assert set(fit) | set(calibration) == set(groups)


def test_r10_support_mode_keeps_image_only_distinct():
    assert _support_mode({"text"}) == "text_only"
    assert _support_mode({"image"}) == "image_only"
    assert _support_mode({"text", "image"}) == "text_and_image"
    assert _support_mode(set()) == "none"


def test_r10_edge_labels_keep_unknowns_outside_confirmed_negatives():
    qrels = [
        {"query_table_id": "q", "target_table_id": "t1"},
        {"query_table_id": "q", "target_table_id": "t2"},
    ]
    recoveries = [
        {
            "query_table_id": "q",
            "target_table_id": "t1",
            "evidence": {"asset_id": "e1"},
        }
    ]
    targets_by_query, evidence_by_query, targets_by_evidence = _support_sets(
        qrels, recoveries
    )
    target_records = [
        {
            "query_id": "q",
            "candidates": [
                {"target_id": "t1", "evidence_ids": ["e1"]},
                {"target_id": "bad", "evidence_ids": ["e1"]},
            ],
        }
    ]
    corrupted = _corrupted_pairs(
        target_records, targets_by_query, evidence_by_query
    )
    edges = [
        {
            "query_id": "q",
            "source_type": "table",
            "positive_id": "t1",
            "candidate_ids": ["t1", "t2", "bad"],
            "destination_type": "table",
        },
        {
            "query_id": "q",
            "source_type": "table",
            "positive_id": "e1",
            "candidate_ids": ["e1", "e_unknown"],
            "destination_type": "text",
        },
        {
            "query_id": "e1",
            "source_type": "text",
            "positive_id": "t1",
            "candidate_ids": ["t1", "bad", "unverified"],
            "destination_type": "table",
        },
    ]

    labeled, summary = label_edge_records(
        edges,
        targets_by_query=targets_by_query,
        evidence_by_query=evidence_by_query,
        targets_by_evidence=targets_by_evidence,
        corrupted_pairs=corrupted,
    )

    assert labeled[0]["positive_ids"] == ["t1", "t2"]
    assert labeled[0]["confirmed_labels"] == [None, None, None]
    assert labeled[1]["confirmed_labels"] == [1, None]
    assert labeled[2]["confirmed_labels"] == [1, 0, None]
    assert summary["corrupted_pairs"] == 1


def test_r10_supported_edge_overrides_corruption_from_another_query():
    qrels = [
        {"query_table_id": "q1", "target_table_id": "t1"},
        {"query_table_id": "q2", "target_table_id": "bad"},
    ]
    recoveries = [
        {
            "query_table_id": "q1",
            "target_table_id": "t1",
            "evidence": {"asset_id": "e1"},
        },
        {
            "query_table_id": "q2",
            "target_table_id": "bad",
            "evidence": {"asset_id": "e1"},
        },
    ]
    targets_by_query, evidence_by_query, targets_by_evidence = _support_sets(
        qrels, recoveries
    )
    corrupted = {("e1", "bad")}

    labeled, summary = label_edge_records(
        [
            {
                "query_id": "e1",
                "source_type": "text",
                "positive_id": "t1",
                "candidate_ids": ["t1", "bad"],
                "destination_type": "table",
            }
        ],
        targets_by_query=targets_by_query,
        evidence_by_query=evidence_by_query,
        targets_by_evidence=targets_by_evidence,
        corrupted_pairs=corrupted,
    )

    assert labeled[0]["positive_ids"] == ["t1", "bad"]
    assert labeled[0]["confirmed_labels"] == [1, 1]
    assert summary["supported_corruption_overlaps_excluded"] == 1

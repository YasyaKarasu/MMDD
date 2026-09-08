from __future__ import annotations

import pytest

from materialize_stage1_r10_splits import partition_records


def _target(query_id: str, evidence_id: str, target_id: str):
    return {
        "query_id": query_id,
        "positive_target_ids": [target_id],
        "candidates": [{"target_id": target_id, "evidence_ids": [evidence_id]}],
    }


def test_partition_records_keeps_query_and_evidence_edges_in_one_bucket():
    targets = [_target("q-fit", "e-fit", "t-fit"), _target("q-cal", "e-cal", "t-cal")]
    edges = [
        {
            "query_id": "q-fit",
            "positive_id": "e-fit",
            "source_type": "table",
            "destination_type": "text",
        },
        {
            "query_id": "e-fit",
            "positive_id": "t-fit",
            "source_type": "text",
            "destination_type": "table",
        },
        {
            "query_id": "q-cal",
            "positive_id": "t-cal",
            "source_type": "table",
            "destination_type": "table",
        },
    ]

    target_parts, edge_parts = partition_records(
        targets, edges, {"q-fit": "train_fit", "q-cal": "train_calibration"}
    )

    assert [row["query_id"] for row in target_parts["train_fit"]] == ["q-fit"]
    assert [row["query_id"] for row in edge_parts["train_fit"]] == ["q-fit", "e-fit"]
    assert [row["query_id"] for row in edge_parts["train_calibration"]] == ["q-cal"]


def test_partition_records_rejects_an_unassigned_edge():
    with pytest.raises(ValueError, match="cannot be assigned"):
        partition_records(
            [_target("q", "e", "t")],
            [
                {
                    "query_id": "unknown-e",
                    "positive_id": "t",
                    "source_type": "text",
                    "destination_type": "table",
                }
            ],
            {"q": "train_fit"},
        )

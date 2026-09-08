from __future__ import annotations

import math

import pytest

from mmdd_stage1.data import EdgeExample
from mmdd_stage1.edge_metrics import summarize_edge_quality


def test_edge_quality_reports_multi_positive_recall_and_excludes_unknowns():
    examples = [
        EdgeExample(
            "q1",
            ("p1", "unknown", "p2", "negative"),
            0,
            source_type="table",
            destination_type="text",
            positive_ids=("p1", "p2"),
            confirmed_labels=(1, None, 1, 0),
        ),
        EdgeExample(
            "q2",
            ("negative", "positive"),
            1,
            source_type="image",
            destination_type="table",
            confirmed_labels=(0, 1),
        ),
    ]
    metrics = summarize_edge_quality(
        examples,
        [[0.9, 0.8, 0.7, 0.1], [0.6, 0.5]],
        [[0.9, 0.8, 0.7, 0.1], [0.6, 0.5]],
        recall_ks=(1, 2, 4),
        reliability_bins=2,
    )

    q_to_text = metrics["by_relation"]["table_to_text"]
    assert q_to_text["ranking"] == {
        "recall@1": pytest.approx(0.5),
        "recall@2": pytest.approx(0.5),
        "recall@4": pytest.approx(1.0),
    }
    assert q_to_text["unknown"] == 1
    assert q_to_text["confirmed_quality"]["confirmed"] == 3
    assert q_to_text["confirmed_quality"]["positive"] == 2
    assert q_to_text["confirmed_quality"]["negative"] == 1
    assert metrics["overall"]["confirmed_quality"]["confirmed"] == 5


def test_edge_quality_auroc_and_auprc_are_tie_aware():
    examples = [
        EdgeExample(
            "q",
            ("p1", "n1", "p2", "n2"),
            0,
            source_type="text",
            destination_type="table",
            positive_ids=("p1", "p2"),
            confirmed_labels=(1, 0, 1, 0),
        )
    ]
    metrics = summarize_edge_quality(
        examples,
        [[1.0, 1.0, 0.0, 0.0]],
        [[0.8, 0.8, 0.2, 0.2]],
    )["overall"]["confirmed_quality"]

    assert metrics["auroc"] == pytest.approx(0.5)
    assert metrics["auprc"] == pytest.approx(0.5)


def test_edge_quality_reports_calibration_and_single_class_limitations():
    examples = [
        EdgeExample(
            "q",
            ("p", "n"),
            0,
            source_type="table",
            destination_type="image",
            confirmed_labels=(1, 0),
        ),
        EdgeExample(
            "q2",
            ("p2",),
            0,
            source_type="table",
            destination_type="text",
            confirmed_labels=(1,),
        ),
    ]
    metrics = summarize_edge_quality(
        examples,
        [[2.0, -2.0], [1.0]],
        [[0.75, 0.25], [0.8]],
        reliability_bins=4,
    )

    image = metrics["by_relation"]["table_to_image"]["confirmed_quality"]
    text = metrics["by_relation"]["table_to_text"]["confirmed_quality"]
    assert image["auroc"] == pytest.approx(1.0)
    assert image["auprc"] == pytest.approx(1.0)
    assert image["brier"] == pytest.approx(0.0625)
    assert image["nll"] == pytest.approx(-math.log(0.75))
    assert image["positive_confidence_quantiles"]["p50"] == pytest.approx(0.75)
    assert image["negative_confidence_quantiles"]["p50"] == pytest.approx(0.25)
    assert text["auroc"] is None
    assert text["auprc"] == pytest.approx(1.0)

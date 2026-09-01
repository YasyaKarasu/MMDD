from __future__ import annotations

import torch

from mmdd_stage1.data import TargetCandidate, TargetExample
from run_stage1_r8_task_r1 import exact_positive_ranks, saturation_metrics


def _example(query_id: str, positives: tuple[str, ...]) -> TargetExample:
    candidates = tuple(TargetCandidate(target_id, ()) for target_id in positives)
    return TargetExample(
        query_id,
        candidates,
        direct_positive_index=0,
        evidence_positive_index=0,
        positive_target_ids=positives,
    )


def test_exact_positive_ranks_use_inner_product_and_target_id_ties() -> None:
    rows = exact_positive_ranks(
        ["q"],
        [("b", "missing")],
        ["b", "a", "c"],
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        device=torch.device("cpu"),
    )

    assert rows[0]["rank"] == 2  # a wins the score tie by target ID.
    assert rows[0]["score"] == 1.0
    assert rows[1]["rank"] is None
    assert not rows[1]["in_corpus"]


def test_saturation_metrics_average_recall_per_query() -> None:
    examples = [_example("q1", ("a", "b")), _example("q2", ("c",))]
    rows = [
        {"query_id": "q1", "positive_target_id": "a", "rank": 1},
        {"query_id": "q1", "positive_target_id": "b", "rank": 11},
        {"query_id": "q2", "positive_target_id": "c", "rank": None},
    ]

    metrics = saturation_metrics(
        examples, rows, corpus_size=20, depths=(10, 20, 50, 100, 200)
    )

    curve = {str(row["requested_k"]): row for row in metrics["curve"]}
    assert curve["10"]["recall"] == 0.25
    assert curve["20"]["recall"] == 0.5
    assert curve["50"]["effective_k"] == 20
    assert metrics["rank_quantiles"]["median"] == 6.0
    assert metrics["decision"] == "unsaturated_current_pool"

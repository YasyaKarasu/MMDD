from __future__ import annotations

import pytest

from export_stage1_r10_task_g_retrieval import compact_stage2_ranking
from evaluate_stage1_r10_frozen import fixed_rankings
from run_stage1_r10_task_f import (
    _accumulate,
    _empty_metrics,
    calibrated_union_fusion,
    reserved_channel_fusion,
)


def test_f5_metrics_use_each_ks_own_admission_budget():
    direct = [_row(str(i), direct_score=1.0) for i in range(60)]
    evidence = [_row(str(i), evidence_score=1.0) for i in range(5, 60)]
    rankings = {k: reserved_channel_fusion(direct, evidence, k=k) for k in (10, 50)}
    assert "5" in {row["target_id"] for row in rankings[10]}
    assert "5" not in {row["target_id"] for row in rankings[50][:10]}
    metrics = _empty_metrics((10, 50))
    _accumulate(
        metrics, {"query_id": "q", "positive_target_ids": ["5"]},
        rankings[50], direct, evidence, {}, recall_ks=(10, 50),
        query_rows=5, evidence_budget=4, rankings_by_k=rankings,
    )
    assert metrics["recall"]["recall@10"] == [1.0]


def test_f5_k_one_uses_the_single_direct_reservation():
    result = reserved_channel_fusion(
        [_row("a", direct_score=1.0)], [_row("b", evidence_score=1.0)], k=1,
    )
    assert [row["target_id"] for row in result] == ["a"]


def test_frozen_evaluation_rejects_unregistered_lambda():
    with pytest.raises(ValueError, match="Unsupported frozen"):
        fixed_rankings([], [], ["f4_lambda_0.75"], {}, k=10)


def test_frozen_f4_is_exactly_the_registered_half_weight():
    ranked = fixed_rankings(
        [_row("a", direct_score=0.8)], [_row("b", evidence_score=0.9)],
        ["f0_direct", "f4_lambda_0.5"], {"a": 0.8, "b": 0.7}, k=10,
    )
    assert ranked["f4_lambda_0.5"][0]["target_id"] == "b"
    assert ranked["f4_lambda_0.5"][0]["score"] == pytest.approx(0.8)
    assert ranked["f0_direct"][0]["target_id"] == "a"


def _row(
    target_id: str,
    *,
    direct_score: float | None = None,
    evidence_score: float | None = None,
) -> dict[str, object]:
    paths = []
    if direct_score is not None:
        paths.append({"kind": "direct", "path_score": direct_score})
    if evidence_score is not None:
        paths.append(
            {
                "kind": "evidence",
                "evidence_id": f"e-{target_id}",
                "path_score": evidence_score,
            }
        )
    return {
        "target_id": target_id,
        "direct_score": direct_score,
        "evidence_score": evidence_score,
        "paths": paths,
    }


def test_f4_scores_every_union_target_with_direct_confidence():
    direct = [_row("direct", direct_score=0.9)]
    evidence = [_row("bridge", evidence_score=0.95)]

    fused = calibrated_union_fusion(
        direct,
        evidence,
        {"direct": 0.9, "bridge": 0.8},
        evidence_weight=0.5,
    )

    assert [row["target_id"] for row in fused] == ["bridge", "direct"]
    assert fused[0]["f4_direct_confidence"] == pytest.approx(0.8)
    assert fused[0]["f4_evidence_support"] == pytest.approx(0.95)
    assert fused[0]["score"] == pytest.approx(0.875)


def test_f4_uses_zero_support_for_direct_only_targets():
    direct = [_row("direct", direct_score=0.9)]

    fused = calibrated_union_fusion(
        direct,
        [],
        {"direct": 0.8},
        evidence_weight=0.25,
    )

    assert fused[0]["score"] == pytest.approx(0.6)
    assert fused[0]["f4_evidence_support"] == 0.0


def test_f4_rejects_missing_union_direct_scores():
    with pytest.raises(ValueError, match="Missing direct confidence"):
        calibrated_union_fusion(
            [],
            [_row("bridge", evidence_score=0.9)],
            {},
            evidence_weight=0.5,
        )


def test_f4_tolerates_and_clamps_tiny_score_roundoff():
    fused = calibrated_union_fusion(
        [],
        [_row("bridge", evidence_score=1.0 + 5e-7)],
        {"bridge": -5e-7},
        evidence_weight=0.5,
    )

    assert fused[0]["f4_direct_confidence"] == 0.0
    assert fused[0]["f4_evidence_support"] == 1.0
    assert fused[0]["score"] == pytest.approx(0.5)


def test_f5_reserves_evidence_slots_and_deduplicates_overlap():
    direct = [_row(target_id, direct_score=1.0) for target_id in "abcdef"]
    evidence = [
        _row("a", evidence_score=1.0),
        _row("b", evidence_score=0.9),
        _row("x", evidence_score=0.8),
        _row("y", evidence_score=0.7),
        _row("z", evidence_score=0.6),
    ]

    fused = reserved_channel_fusion(direct, evidence, k=6)

    ids = [str(row["target_id"]) for row in fused]
    assert len(ids) == len(set(ids)) == 6
    assert ids == ["a", "x", "b", "y", "c", "z"]
    assert [row["f5_channel"] for row in fused] == [
        "direct",
        "evidence",
        "direct",
        "evidence",
        "direct",
        "evidence",
    ]


def test_f5_fills_from_remaining_channel_when_one_is_short():
    direct = [_row(target_id, direct_score=1.0) for target_id in "abcdef"]
    evidence = [_row("x", evidence_score=1.0)]

    fused = reserved_channel_fusion(direct, evidence, k=5)

    ids = [str(row["target_id"]) for row in fused]
    assert len(ids) == len(set(ids)) == 5
    assert set(ids) == {"a", "b", "c", "d", "x"}


def test_f5_high_overlap_advances_past_the_reserved_evidence_prefix():
    direct = [_row(target_id, direct_score=1.0) for target_id in "abcd"]
    evidence = [
        _row(target_id, evidence_score=1.0)
        for target_id in ["a", "b", "c", "d", "x", "y", "z"]
    ]

    fused = reserved_channel_fusion(direct, evidence, k=6)

    ids = [str(row["target_id"]) for row in fused]
    assert ids == ["a", "d", "b", "x", "c", "y"]
    assert len(ids) == len(set(ids)) == 6


def test_task_g_compaction_preserves_f4_score_and_stage2_paths():
    ranking = [
        {
            "target_id": "mixed",
            "score": 0.75,
            "direct_score": 0.8,
            "evidence_score": 0.7,
            "paths": [
                {"kind": "direct", "path_score": 0.8},
                {"kind": "evidence", "evidence_id": "e1", "path_score": 0.7},
                {"kind": "evidence", "evidence_id": "e2", "path_score": 0.6},
            ],
        },
        {
            "target_id": "direct-only",
            "score": 0.6,
            "direct_score": 0.8,
            "evidence_score": None,
            "paths": [{"kind": "direct", "path_score": 0.8}],
        },
    ]

    compact = compact_stage2_ranking(
        ranking,
        result_k=2,
        path_result_k=2,
        evidence_path_k=1,
    )

    assert compact[0]["score"] == pytest.approx(0.75)
    assert compact[0]["stage2_table_score"] == pytest.approx(0.75)
    assert compact[0]["evidence_score"] == pytest.approx(0.7)
    assert compact[0]["paths"] == [
        {"kind": "direct"},
        {"kind": "evidence", "evidence_id": "e1", "path_score": 0.7},
    ]
    assert "evidence_score" not in compact[1]
    assert compact[1]["paths"] == [{"kind": "direct"}]


def test_task_g_compaction_honors_explicit_bundle_selection_order():
    compact = compact_stage2_ranking(
        [
            {
                "target_id": "row-support",
                "score": 0.5,
                "evidence_score": 0.4,
                "selected_evidence_ids": ["e2", "e1"],
                "paths": [
                    {"kind": "evidence", "evidence_id": "e1", "path_score": 0.9},
                    {"kind": "evidence", "evidence_id": "e2", "path_score": 0.8},
                ],
            }
        ],
        result_k=1,
        path_result_k=1,
        evidence_path_k=1,
    )

    assert compact[0]["paths"] == [
        {"kind": "evidence", "evidence_id": "e2", "path_score": 0.8}
    ]

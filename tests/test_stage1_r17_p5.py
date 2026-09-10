from __future__ import annotations

from pathlib import Path

from run_stage1_r17_p5 import _candidate_outcomes, choose_wrong_evidence


def test_wrong_evidence_requires_verified_modality_matched_donors(
    tmp_path: Path,
) -> None:
    text_a = {"object_type": "text", "text": "short"}
    text_b = {"object_type": "text", "text": "same size"}
    image_a = {"object_type": "image", "image": str(tmp_path / "a.jpg")}
    image_b = {"object_type": "image", "image": str(tmp_path / "b.jpg")}
    metadata = {"ta": text_a, "tb": text_b, "ia": image_a, "ib": image_b}
    selected, status = choose_wrong_evidence(
        ["ta", "ia"],
        target_id="target",
        query_id="query",
        object_metadata=metadata,
        verified_wrong_by_target={"target": {"text": ["tb"], "image": ["ib"]}},
    )
    assert status == "verified_wrong"
    assert selected == ["tb", "ib"]


def test_wrong_evidence_never_uses_unknown_fallback() -> None:
    selected, status = choose_wrong_evidence(
        ["ta"],
        target_id="target",
        query_id="query",
        object_metadata={"ta": {"object_type": "text", "text": "x"}},
        verified_wrong_by_target={},
    )
    assert selected == []
    assert status == "unavailable_verified_wrong"


def test_candidate_outcomes_requires_recovered_branch_for_correct_join() -> None:
    candidate = {
        "status": "verified",
        "stage1_rank": 20,
        "rerank_rank": 4,
        "verification": {"joinable": True},
        "branches": {
            "direct": {"verification": {"joinable": True}},
            "evidence": {"verification": {"joinable": False}},
        },
        "fixed80_evaluation": {
            "correct_attribute": True,
            "row_metrics": [
                {
                    "truth_available": True,
                    "model_value_correct": True,
                    "correct_value_recovery": True,
                }
            ],
        },
    }
    result = _candidate_outcomes(candidate)
    assert result["correct_attribute"] is True
    assert result["correct_value"] is True
    assert result["correct_join"] is False
    assert result["final_top10"] is True
    assert result["rank_improved"] is True

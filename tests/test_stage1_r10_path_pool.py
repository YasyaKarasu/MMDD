from __future__ import annotations

from export_stage1_path_pool import _paths_by_target, _student_score_space


def test_path_pool_export_keeps_union_without_channel_duplicates():
    direct_path = {"kind": "direct", "path_score": 0.8}
    evidence_path = {
        "kind": "evidence",
        "evidence_id": "e",
        "path_score": 1.2,
    }
    shared = {
        "target_id": "shared",
        "paths": [direct_path, evidence_path],
    }
    detailed = {
        "direct": [shared, {"target_id": "direct", "paths": [direct_path]}],
        "evidence": [shared, {"target_id": "evidence", "paths": [evidence_path]}],
    }

    paths = _paths_by_target(detailed)

    assert set(paths) == {"shared", "direct", "evidence"}
    assert paths["shared"] == [direct_path, evidence_path]


def test_path_pool_export_uses_selection_student_score_space():
    assert _student_score_space({"student_score_space": "confidence"}) == (
        "confidence"
    )
    assert _student_score_space({}) == "raw_logit"

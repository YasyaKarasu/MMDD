from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finalize_stage1_r12 import _percentile, _timing
from finalize_stage1_r12_reviews import (
    _support_model_gate,
    finalize_attribute_reviews,
    finalize_task_f_reviews,
    prepare_templates,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(case_id: str, *, index: int = 0) -> dict:
    return {
        "case_id": case_id,
        "query_id": f"q{index}",
        "target_id": f"t{index}",
        "row_id": index % 5,
        "source_column_id": index % 3,
        "source_table_id": f"group{index % 20}",
        "split": "train_fit",
        "modality": "text",
        "evidence_id": f"e{index}",
        "sampling_weight": 1.0,
        "routing_row": index % 5,
    }


def _review(
    case_id: str,
    reviewer_id: str,
    label: str,
) -> dict:
    return {
        "case_id": case_id,
        "reviewer_id": reviewer_id,
        "reviewer_role": "independent_human",
        "reviewed_at_utc": "2026-09-09T00:00:00+00:00",
        "label": label,
        "supported_value": "value" if label == "confirmed_support" else None,
        "evidence_locator": "paragraph 1" if label == "confirmed_support" else None,
        "notes": None,
    }


def _review_tree(tmp_path: Path) -> Path:
    task_b = tmp_path / "taskB_attribute_audit"
    task_f = tmp_path / "taskF_end_to_end" / "human_audit"
    candidates = [_candidate("a"), _candidate("b", index=1)]
    _write_jsonl(task_b / "review_packets.jsonl", candidates)
    _write_jsonl(task_b / "second_review_packets.jsonl", [candidates[0]])
    _write_jsonl(task_b / "selected_candidates.jsonl", candidates)
    _write_jsonl(task_f / "review_packets.jsonl", [{"case_id": "f"}])
    return tmp_path


def test_prepare_templates_creates_frozen_review_submission_files(tmp_path):
    root = _review_tree(tmp_path)
    workflow = prepare_templates(root)
    assert workflow["attribute_primary_cases"] == 2
    assert workflow["attribute_second_cases"] == 1
    assert workflow["task_f_cases"] == 1
    assert (root / "HUMAN_REVIEW_INSTRUCTIONS.md").is_file()
    primary = json.loads(
        (root / "taskB_attribute_audit/primary_reviews.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert primary["reviewer_role"] == "independent_human"
    assert primary["label"] is None


def test_incomplete_attribute_review_is_rejected(tmp_path):
    root = _review_tree(tmp_path)
    prepare_templates(root)
    with pytest.raises(ValueError, match="reviewer_id is required"):
        finalize_attribute_reviews(root)


def test_double_review_requires_normalized_distinct_reviewer_ids(tmp_path):
    root = _review_tree(tmp_path)
    _write_jsonl(
        root / "taskB_attribute_audit/primary_reviews.jsonl",
        [_review("a", " reviewer ", "confirmed_support"),
         _review("b", "reviewer", "confirmed_wrong_attribute")],
    )
    _write_jsonl(
        root / "taskB_attribute_audit/second_reviews.jsonl",
        [_review("a", "reviewer", "confirmed_support")],
    )
    with pytest.raises(ValueError, match="must use another reviewer"):
        finalize_attribute_reviews(root)


def test_review_timestamp_must_be_utc(tmp_path):
    root = _review_tree(tmp_path)
    primary = [
        _review("a", "one", "confirmed_support"),
        _review("b", "one", "confirmed_wrong_attribute"),
    ]
    primary[0]["reviewed_at_utc"] = "2026-09-09T01:00:00+01:00"
    _write_jsonl(root / "taskB_attribute_audit/primary_reviews.jsonl", primary)
    _write_jsonl(
        root / "taskB_attribute_audit/second_reviews.jsonl",
        [_review("a", "two", "confirmed_support")],
    )
    with pytest.raises(ValueError, match="reviewed_at_utc must be UTC"):
        finalize_attribute_reviews(root)


def test_attribute_disagreement_resolves_to_unknown(tmp_path):
    root = _review_tree(tmp_path)
    _write_jsonl(
        root / "taskB_attribute_audit/primary_reviews.jsonl",
        [_review("a", "one", "confirmed_support"),
         _review("b", "one", "confirmed_wrong_attribute")],
    )
    _write_jsonl(
        root / "taskB_attribute_audit/second_reviews.jsonl",
        [_review("a", "two", "confirmed_wrong_attribute")],
    )
    summary = finalize_attribute_reviews(root)
    resolved = [
        json.loads(line)
        for line in (root / "taskB_attribute_audit/resolved_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert resolved[0]["resolved_label"] == "review_disagreement"
    assert resolved[0]["supported_value"] is None
    assert summary["overall"]["unknown_rate"] == pytest.approx(0.5)


def test_support_model_gate_requires_counts_and_source_group_coverage():
    rows = [
        {
            **_candidate(f"p{index}", index=index),
            "resolved_label": "confirmed_support",
        }
        for index in range(64)
    ] + [
        {
            **_candidate(f"n{index}", index=index),
            "resolved_label": "confirmed_wrong_attribute",
        }
        for index in range(64)
    ]
    assert _support_model_gate(rows)["eligible"]
    for row in rows:
        row["source_table_id"] = "one_group"
    assert not _support_model_gate(rows)["eligible"]


def test_task_f_review_fields_must_be_boolean(tmp_path):
    root = _review_tree(tmp_path)
    invalid = {
        "case_id": "f",
        "reviewer_id": "reviewer",
        "reviewer_role": "independent_human",
        "reviewed_at_utc": "2026-09-09T00:00:00+00:00",
        "evidence_supports_requested_attribute": True,
        "generated_values_supported": None,
        "join_decision_correct": False,
    }
    _write_jsonl(
        root / "taskF_end_to_end/human_audit/human_reviews.jsonl", [invalid]
    )
    with pytest.raises(ValueError, match="generated_values_supported must be boolean"):
        finalize_task_f_reviews(root)


def test_latency_uses_linear_percentiles_and_never_fabricates_empty_cost():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _timing([]) == {
        "observations": 0,
        "seconds_p50": None,
        "seconds_p95": None,
        "seconds_total": None,
    }

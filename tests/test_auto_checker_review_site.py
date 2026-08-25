from __future__ import annotations

import pytest

from scripts_old.build_auto_checker_review_site import (
    reclassify_review_rows,
    render_review_html,
    select_review_rows,
)
from scripts_old.serve_auto_checker_review import validate_review_payload


def _row(key: str, terra: str, luna: str, asset_type: str = "text") -> dict:
    return {
        "cache_key": key,
        "asset_id": f"asset_{key}",
        "asset_type": asset_type,
        "attribute_name": "Year",
        "claimed_value": key,
        "terra_verdict": terra,
        "luna_verdict": luna,
    }


def test_union_supported_selects_agreements_and_disagreements() -> None:
    rows = [
        _row("a", "supported", "supported"),
        _row("b", "supported", "insufficient"),
        _row("c", "contradicted", "supported", "image"),
        _row("d", "insufficient", "contradicted"),
    ]
    selected = select_review_rows(rows, include_mode="union_supported")
    assert {row["cache_key"] for row in selected} == {"a", "b", "c"}


def test_reclassify_review_rows_uses_station_name_context() -> None:
    rows = [
        {
            "cache_key": "station",
            "asset_id": "asset_img_61e0f54f8aea2000",
            "attribute_name": "Japanese",
            "claimed_value": "観音",
            "terra_extracted_value": "観音駅",
            "terra_verdict": "contradicted",
            "luna_extracted_value": "観音",
            "luna_verdict": "supported",
        }
    ]
    records = {
        "station": {
            "row_attributes": [
                {"name": "Station", "value": "Kannon", "is_entity": True},
                {"name": "Japanese", "value": "観音", "is_entity": False},
            ]
        }
    }

    [result] = reclassify_review_rows(rows, records)

    assert result["terra_recorded_verdict"] == "contradicted"
    assert result["terra_verdict"] == "supported"
    assert result["luna_verdict"] == "supported"


def test_render_review_html_escapes_script_terminators() -> None:
    manifest = {"title": "Review <54>", "items": [{"value": "</script>"}]}
    rendered = render_review_html(
        "<title>__REVIEW_TITLE__</title><script>__REVIEW_DATA__</script>",
        manifest,
    )
    assert "Review &lt;54&gt;" in rendered
    assert "</script>" not in rendered.split("<script>", 1)[1].split("</script>", 1)[0]
    assert "\\u003c/script\\u003e" in rendered


def test_validate_review_payload_keeps_only_bounded_fields() -> None:
    payload = {
        "dataset_id": "dataset-1",
        "reviewer": "reviewer",
        "reviews": {
            "review-1": {
                "human_verdict": "supported",
                "human_value": "1999",
                "confidence": "high",
                "issue_type": "normalization",
                "notes": "clear evidence",
                "locked": True,
                "completed": True,
                "updated_at": "2026-08-11T00:00:00Z",
                "unexpected": "discarded",
            }
        },
    }
    result = validate_review_payload(
        payload,
        dataset_id="dataset-1",
        allowed_review_ids={"review-1"},
    )
    assert result["reviews"]["review-1"]["human_verdict"] == "supported"
    assert "unexpected" not in result["reviews"]["review-1"]


def test_validate_review_payload_rejects_unknown_review_id() -> None:
    with pytest.raises(ValueError, match="invalid review item"):
        validate_review_payload(
            {
                "dataset_id": "dataset-1",
                "reviews": {"unknown": {"human_verdict": "supported"}},
            },
            dataset_id="dataset-1",
            allowed_review_ids={"review-1"},
        )

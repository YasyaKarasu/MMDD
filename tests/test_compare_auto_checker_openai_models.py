import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts_old"))

from compare_auto_checker_openai_models import (
    collect_recent_terra_reviews,
    comparison_row,
    iter_jsonl_reverse,
    summarize_comparisons,
)


def _cached_record(cache_key: str, attribute: str, value: str) -> dict:
    return {
        "cache_key": cache_key,
        "auto_check": {
            "reviews": [
                {
                    "attribute_name": attribute,
                    "claimed_value": value,
                    "decision_source": "secondary_openai",
                    "review_complete": True,
                    "error_code": "",
                    "secondary_extracted_value": value,
                    "secondary_verdict": "supported",
                }
            ]
        },
    }


def test_reverse_jsonl_and_recent_review_selection(tmp_path: Path):
    path = tmp_path / "cache.jsonl"
    records = [
        _cached_record("old", "State", "Alabama"),
        _cached_record("new", "State", "Georgia"),
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    assert [row["cache_key"] for row in iter_jsonl_reverse(path)] == ["new", "old"]
    selected = collect_recent_terra_reviews(path, 1)
    assert selected[0]["record"]["cache_key"] == "new"


def test_comparison_summary_tracks_support_disagreements():
    terra_supported = {
        "cache_key": "one",
        "asset_id": "asset-1",
        "asset_type": "text",
        "attribute_name": "State",
        "claimed_value": "Alabama",
        "terra_extracted_value": "Alabama",
        "terra_verdict": "supported",
    }
    agree = comparison_row(terra_supported, "Alabama")
    disagree = comparison_row(
        {**terra_supported, "cache_key": "two", "asset_type": "image"},
        "Georgia",
    )

    summary = summarize_comparisons([agree, disagree], planned=2, errors=0)

    assert summary["complete"] is True
    assert summary["overall"]["verdict_agreement_rate"] == 0.5
    assert summary["overall"]["value_agreement_rate"] == 0.5
    assert summary["overall"]["terra_supported_luna_not"] == 1
    assert summary["by_modality"]["text"]["verdict_agreement_rate"] == 1.0


def test_comparison_reclassifies_station_type_suffix_as_supported():
    row = comparison_row(
        {
            "cache_key": "station",
            "asset_id": "asset_img_61e0f54f8aea2000",
            "asset_type": "image",
            "entity_column_name": "Station",
            "attribute_name": "Japanese",
            "claimed_value": "観音",
            "terra_extracted_value": "観音駅",
            "terra_verdict": "contradicted",
        },
        "観音",
    )

    assert row["terra_recorded_verdict"] == "contradicted"
    assert row["terra_verdict"] == "supported"
    assert row["luna_verdict"] == "supported"
    assert row["value_agreement"] is True
    assert row["verdict_agreement"] is True

from __future__ import annotations

from pathlib import Path

from PIL import Image

from mmdd_stage2.r4c_fast_recovery import build_prompt, parse_completion
from mmdd_stage2.r4c_fast_types import FastUnit, ViewSpec


def _unit() -> FastUnit:
    return FastUnit.create(
        query_id="q",
        target_id="t",
        column_id=1,
        column_name="Artist",
        query_row_id=0,
        source_group="g",
        cells=(("Name", "Example"),),
        evidence_ids=("img",),
        focus_image_id="img",
    )


def test_multiview_uses_one_source_label_and_no_crop_evidence_id(tmp_path: Path) -> None:
    original, crop = tmp_path / "original.png", tmp_path / "crop.png"
    Image.new("RGB", (32, 32), "white").save(original)
    Image.new("RGB", (16, 16), "black").save(crop)
    source = {
        "asset_id": "img",
        "asset_type": "image",
        "views": [
            ViewSpec("img", "ORIGINAL", str(original), None, 262144, "sha-original"),
            ViewSpec("img", "TIGHT_CROP", str(crop), (0, 0, 16, 16), 262144, "sha-crop"),
        ],
    }
    _content, audit, opened = build_prompt(_unit(), [source])
    for image in opened:
        image.close()
    assert audit["evidence_label_map"] == {"E1": "img"}
    assert "E1 (image source; two views of the SAME source)" in audit["raw_prompt"]
    assert "E1_crop" not in audit["raw_prompt"]


def test_parser_rejects_view_or_crop_source_labels() -> None:
    raw = (
        '{"status":"VALUE","value":"X","evidence_ids":["E1_crop"],'
        '"text_support_quotes":[],"image_support_notes":[]}'
    )
    parsed = parse_completion(raw, "stop", {"E1"}, {})
    assert parsed["status"] == "PARSE_ERROR"
    assert parsed["parse_error"] == "PARSE_ERROR_BAD_SOURCE_ID"


def test_parser_validates_text_quote_substrings() -> None:
    raw = (
        '{"status":"VALUE","value":"X","evidence_ids":["E1"],'
        '"text_support_quotes":[{"evidence_id":"E1","quote":"not there"}],'
        '"image_support_notes":[]}'
    )
    parsed = parse_completion(raw, "stop", {"E1"}, {"E1": "source text"})
    assert parsed["status"] == "VALUE"
    assert parsed["invalid_text_quotes"]


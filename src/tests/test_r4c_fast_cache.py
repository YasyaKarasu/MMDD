from __future__ import annotations

from mmdd_stage2.r4c_fast_metrics import decide_gate
from mmdd_stage2.r4c_fast_recovery import generation_input_hash
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


def _sources(crop_sha: str) -> list[dict]:
    return [{
        "asset_id": "img",
        "asset_type": "image",
        "views": [
            ViewSpec("img", "ORIGINAL", "/tmp/original", None, 262144, "original-sha"),
            ViewSpec("img", "TIGHT_CROP", "/tmp/crop", (0, 0, 10, 10), 262144, crop_sha),
        ],
    }]


def test_generation_hash_changes_with_crop_bytes() -> None:
    left = generation_input_hash(_unit(), "V2_RAEA_DUAL", _sources("a"), model_config_hash="m")
    right = generation_input_hash(_unit(), "V2_RAEA_DUAL", _sources("b"), model_config_hash="m")
    assert left != right


def test_generation_hash_changes_with_prompt_arm_row_column_model_and_decode_contract() -> None:
    unit = _unit()
    left = generation_input_hash(unit, "V2_RAEA_DUAL", _sources("a"), model_config_hash="m")
    assert left == generation_input_hash(unit, "V3_CONSENSUS_DUAL", _sources("a"), model_config_hash="m")
    assert left != generation_input_hash(unit, "V2_RAEA_DUAL", _sources("a"), model_config_hash="other")


def test_gate_stops_when_crop_does_not_gain_two_points() -> None:
    def metrics(strict: float, latency: float) -> dict:
        return {
            "units": 48,
            "query_macro_strict_correct": strict,
            "query_macro_known_witness_cited_correct": strict,
            "query_macro_supported_value_correct": strict,
            "query_macro_hallucinated_supported_value": 0.0,
            "median_total_elapsed_seconds": latency,
            "median_elapsed_generation_seconds": latency,
        }

    summary = {
        "arms": {
            "V0_FULL_BASE": metrics(0.50, 1.0),
            "V1_FULL_HIGHRES": metrics(0.50, 1.2),
            "V2_RAEA_DUAL": metrics(0.51, 2.0),
            "V3_CONSENSUS_DUAL": metrics(0.49, 2.1),
        },
        "comparisons_vs_v0": {
            "V1_FULL_HIGHRES": {"wlt": {"wins": 1, "losses": 1, "ties": 46}},
            "V2_RAEA_DUAL": {"wlt": {"wins": 2, "losses": 1, "ties": 45}},
            "V3_CONSENSUS_DUAL": {"wlt": {"wins": 1, "losses": 2, "ties": 45}},
        },
    }
    decision = decide_gate(summary)
    assert decision["status"] == "STOP_USE_V0"
    assert not decision["continue_expand"]


def test_gate_shortcuts_to_faster_highres_within_one_point() -> None:
    def metrics(strict: float, latency: float) -> dict:
        return {
            "units": 48,
            "query_macro_strict_correct": strict,
            "query_macro_known_witness_cited_correct": strict,
            "query_macro_supported_value_correct": strict,
            "query_macro_hallucinated_supported_value": 0.0,
            "median_total_elapsed_seconds": latency,
            "median_elapsed_generation_seconds": latency,
        }

    summary = {
        "arms": {
            "V0_FULL_BASE": metrics(0.50, 1.0),
            "V1_FULL_HIGHRES": metrics(0.52, 1.2),
            "V2_RAEA_DUAL": metrics(0.53, 2.0),
            "V3_CONSENSUS_DUAL": metrics(0.51, 2.1),
        },
        "comparisons_vs_v0": {
            "V1_FULL_HIGHRES": {"wlt": {"wins": 3, "losses": 1, "ties": 44}},
            "V2_RAEA_DUAL": {"wlt": {"wins": 4, "losses": 1, "ties": 43}},
            "V3_CONSENSUS_DUAL": {"wlt": {"wins": 2, "losses": 2, "ties": 44}},
        },
    }
    decision = decide_gate(summary)
    assert decision["status"] == "STOP_COMPLEX_CROP_USE_HIGHRES"
    assert decision["winner"] == "V1_FULL_HIGHRES"


def test_gate_refuses_to_select_from_one_label_blind_gt_unit() -> None:
    decision = decide_gate({"evaluable_units": 1, "arms": {}})
    assert decision["status"] == "STOP_INSUFFICIENT_EVALUABLE_UNITS"
    assert decision["winner"] is None
    assert not decision["continue_expand"]

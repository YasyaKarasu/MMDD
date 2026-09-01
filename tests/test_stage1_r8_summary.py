from __future__ import annotations

from summarize_stage1_r8 import final_choice


def _payload(lake: str, *, point: bool, ci: bool, double_win: bool) -> dict:
    return {
        "lake": lake,
        "selected": "candidate",
        "entitables_double_win": double_win,
        "metrics": {
            "candidate": {
                "r5_point_anchor_satisfied": point,
                "r5_ci_gate_satisfied": ci,
            }
        },
    }


def test_entitables_requires_double_win_to_replace_r5() -> None:
    assert final_choice(
        _payload("entitables", point=True, ci=True, double_win=False)
    ) == "r5"
    assert final_choice(
        _payload("entitables", point=True, ci=True, double_win=True)
    ) == "r8"


def test_wdc_requires_both_r5_gates() -> None:
    assert final_choice(_payload("wdc", point=True, ci=True, double_win=False)) == "r8"
    assert final_choice(_payload("wdc", point=True, ci=False, double_win=False)) == "r5"

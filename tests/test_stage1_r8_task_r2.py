from __future__ import annotations

from run_stage1_r8_task_r2 import select_configuration


def _row(fused: float, evidence: float, coverage: float = 0.0) -> dict:
    return {
        "metrics": {
            "recall@10": fused,
            "evidence": {"recall@10": evidence},
            "positive_evidence_path_coverage@10": coverage,
        }
    }


def test_entitables_selection_prioritizes_double_win_gate() -> None:
    metrics = {
        "high_fused_low_evidence": _row(0.40, 0.09),
        "double_win": _row(0.38, 0.10),
    }

    selected, double_win = select_configuration(metrics, "entitables", 0.3774)

    assert selected == "double_win"
    assert double_win


def test_wdc_selection_uses_point_anchor_then_fused_recall() -> None:
    metrics = {
        "below": _row(0.64, 0.50),
        "anchor": _row(0.6436, 0.20),
        "best": _row(0.65, 0.10),
    }

    selected, double_win = select_configuration(metrics, "wdc", 0.6436)

    assert selected == "best"
    assert not double_win

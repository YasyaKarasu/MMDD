from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_taskk import (
    _decisions,
    _epoch_metrics,
    _run_spec,
    _select_by_lake,
)


def test_taskk_run_spec_parses_training_directory() -> None:
    spec = _run_spec("kd03=KD 0.3,entitables,0.3,/tmp/run")

    assert spec == ("kd03", "KD 0.3", "entitables", 0.3, Path("/tmp/run"))


def test_taskk_epoch_metrics_uses_top_level_fused_channel() -> None:
    record = {
        "epoch": 2,
        "dev_retrieval": {
            "recall@10": 0.4,
            "recall@50": 0.8,
            "direct": {"recall@10": 0.5, "recall@50": 0.9},
            "evidence": {"recall@10": 0.2},
        },
        "relation_drift": {"table_to_table": 1.25},
        "gate": {
            "eligible": True,
            "per_dataset": [{"bootstrap": {"mean": 0.1}}],
        },
    }

    result = _epoch_metrics(record)

    assert result["fused_recall@10"] == 0.4
    assert result["direct_recall@10"] == 0.5
    assert result["direct_recall@10_vs_raw"] == {"mean": 0.1}


def test_taskk_decision_requires_strict_entitables_ablation_chain() -> None:
    def result(lake: str, weight: float, direct: float, raw: float = 0.3) -> dict:
        return {
            "label": f"{lake}-{weight}",
            "lake": lake,
            "kd_weight": weight,
            "selected": {"direct_recall@10": direct},
            "raw": {"direct_recall@10": raw},
            "best_checkpoint": f"/{lake}-{weight}.pt",
            "gate_unsatisfied": False,
        }

    decisions = _decisions(
        {
            "supervised": result("entitables", 0, 0.32),
            "kd03": result("entitables", 0.3, 0.35),
            "kd10": result("entitables", 1.0, 0.34),
            "wdc0": result("wdc", 0, 0.6, raw=0.61),
            "wdc03": result("wdc", 0.3, 0.6, raw=0.61),
        }
    )

    assert decisions["entitables_distillation_chain"] is True
    assert decisions["entitables_best_kd_label"] == "entitables-0.3"
    assert decisions["wdc_all_gates_satisfied"] is True
    assert decisions["task_m_required"] is False


def test_taskk_lake_selection_uses_best_gate_eligible_fused_result() -> None:
    runs = {
        "fallback": {
            "lake": "wdc",
            "label": "fallback",
            "gate_unsatisfied": True,
            "selected": {"fused_recall@10": 0.9},
        },
        "eligible": {
            "lake": "wdc",
            "label": "eligible",
            "gate_unsatisfied": False,
            "selected": {"fused_recall@10": 0.6},
            "best_epoch": 2,
            "best_checkpoint": "/best.pt",
            "best_checkpoint_sha256": "sha",
            "directory": "/run",
        },
    }

    selected = _select_by_lake(runs)

    assert selected["wdc"]["run"] == "eligible"
    assert selected["wdc"]["checkpoint"] == "/best.pt"

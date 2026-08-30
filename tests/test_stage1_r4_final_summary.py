from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_stage1_r4_final import _comparison, _effective_teacher, _run_for


def test_final_summary_selects_strongest_kd_direct_run_per_lake() -> None:
    taskk = {
        "runs": {
            "supervised": {
                "lake": "entitables", "kd_weight": 0,
                "selected": {"direct_recall@10": 0.32},
            },
            "kd03": {
                "lake": "entitables", "kd_weight": 0.3,
                "selected": {"direct_recall@10": 0.36},
            },
            "kd10": {
                "lake": "entitables", "kd_weight": 1.0,
                "selected": {"direct_recall@10": 0.34},
            },
        }
    }

    name, _run = _run_for(taskk, "entitables", kd=True)

    assert name == "kd03"


def test_final_summary_compares_matching_fused_channels() -> None:
    before = {
        "metrics": {"per_query": {"fused": {"recall@10": [0.0, 1.0, 0.0]}}}
    }
    after = {
        "metrics": {"per_query": {"fused": {"recall@10": [1.0, 1.0, 0.0]}}}
    }

    result = _comparison(
        after, before, iterations=500, seed=13,
        candidate_channel="fused", reference_channel="fused",
    )

    assert result["mean"] == 1 / 3


def test_final_summary_records_taskm_teacher_only_for_entitables() -> None:
    taskj_lake = {
        "provenance": {
            "teacher_checkpoint": "/mixed.pt",
            "teacher_checkpoint_sha256": "mixed-sha",
        }
    }
    taskm = {
        "teacher": {
            "gate_pass": True,
            "checkpoint": "/entitables.pt",
            "checkpoint_sha256": "entitables-sha",
        }
    }

    entitables = _effective_teacher("entitables", taskj_lake, taskm)
    wdc = _effective_teacher("wdc", taskj_lake, taskm)

    assert entitables["checkpoint_sha256"] == "entitables-sha"
    assert wdc["checkpoint_sha256"] == "mixed-sha"


def test_final_summary_records_per_lake_taskm_teachers() -> None:
    taskj_lake = {
        "provenance": {
            "teacher_checkpoint": "/mixed.pt",
            "teacher_checkpoint_sha256": "mixed-sha",
        }
    }
    taskm = {
        "lakes": {
            "wdc": {
                "label": "WDC",
                "teacher": {
                    "gate_pass": True,
                    "checkpoint": "/wdc.pt",
                    "checkpoint_sha256": "wdc-sha",
                },
            }
        }
    }

    result = _effective_teacher("wdc", taskj_lake, taskm)

    assert result["checkpoint_sha256"] == "wdc-sha"

from __future__ import annotations

import json
from pathlib import Path

from summarize_stage1_r7_integration import EPOCH0_Q_CONFIG, _epoch0_row


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_epoch0_row_uses_formal_metrics_and_task_q_per_query_ci(
    tmp_path: Path,
) -> None:
    task_root = tmp_path / "r7"
    formal_path = (
        task_root
        / "taskP_lowrank/entitables/k_16/mu_0/"
        "final_evaluation_epoch0_full/metrics.json"
    )
    _write_json(
        formal_path,
        {
            "metrics": {
                "recall@10": 1.0,
                "recall@20": 0.9,
                "recall@30": 0.8,
                "recall@40": 0.7,
                "recall@50": 0.6,
                "mrr@50": 0.5,
                "positive_evidence_path_coverage@10": 0.4,
            }
        },
    )
    q_path = (
        task_root
        / "taskQ_joint_selection/entitables/lowrank_k_16_mu_0/metrics.json"
    )
    _write_json(
        q_path,
        {
            "metrics": {
                EPOCH0_Q_CONFIG: {
                    "metrics": {
                        "recall@10": 1.0,
                        "per_query": {"recall@10": [1.0, 1.0]},
                    }
                }
            }
        },
    )
    r5 = {
        "lakes": {
            "entitables": {
                "rows": [{"per_query_recall@10": [0.0, 0.0]}]
            }
        }
    }

    row = _epoch0_row(task_root, r5, "entitables")

    assert row["recall@20"] == 0.9
    assert row["coverage@10"] == 0.4
    assert row["delta"] == 1.0
    assert row["ci_low"] == 1.0
    assert row["ci_high"] == 1.0
    assert str(formal_path) in row["source"]

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import summarize_stage1_r5_taskx


def _metrics(values: list[float], evidence: float, coverage: float) -> dict:
    recall = sum(values) / len(values)
    return {
        "recall@10": recall,
        "direct": {"recall@10": recall},
        "evidence": {"recall@10": evidence},
        "positive_evidence_path_coverage@10": coverage,
        "per_query": {"fused": {"recall@10": values}},
    }


def _write_variant(
    root: Path,
    values: list[float],
    evidence: float,
    coverage: float,
) -> None:
    (root / "data").mkdir(parents=True)
    (root / "teacher_retrieval").mkdir()
    (root / "student_tau_0.3" / "final_evaluation").mkdir(parents=True)
    (root / "data" / "preflight.json").write_text(
        json.dumps(
            {
                "train_relation_counts_before": {"table_to_image": 10},
                "train_relation_counts_after": {},
                "suppressed_train_relations": ["table_to_image"],
                "evidence_concentration": {"top_1_share": 0.2, "top_10_share": 0.4},
            }
        ),
        encoding="utf-8",
    )
    metrics = _metrics(values, evidence, coverage)
    (root / "teacher_retrieval" / "metrics.json").write_text(
        json.dumps(
            {
                "teacher": {
                    "metrics": metrics,
                    "feature_coverage": {
                        "cached_hidden_fraction": 0.75,
                        "pooled_embedding_fallback_objects": 2,
                        "allowed_pooled_embedding_fallback_objects": 2,
                        "unexpected_pooled_embedding_fallback_objects": 0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "student_tau_0.3" / "final_evaluation" / "metrics.json").write_text(
        json.dumps({"systems": {"student": {"metrics": metrics}}}),
        encoding="utf-8",
    )


def test_taskx_summary_reports_acceptance_and_distribution(tmp_path: Path) -> None:
    task = tmp_path / "task"
    for lake in ("wdc", "entitables"):
        for variant in summarize_stage1_r5_taskx.VARIANTS:
            improved = variant == "text_only" and lake == "wdc"
            _write_variant(
                task / lake / variant,
                [1.0, 0.0, 1.0, 0.0],
                0.15 if improved else 0.10,
                0.20 if improved else 0.15,
            )

    payload = summarize_stage1_r5_taskx.run(
        argparse.Namespace(
            task_root=str(task),
            output_dir=None,
            entitables_tolerance=0.02,
            bootstrap_iterations=100,
            bootstrap_seed=13,
        )
    )

    assert payload["decisions"]["text_only"]["accepted"] is True
    assert payload["decisions"]["image_sparsity_resolved"] is True
    assert "table_to_image" in payload["training_distributions"]["wdc"][
        "current"
    ]["suppressed_train_relations"]
    assert (task / "evidence_modality_ablation.csv").is_file()
    teacher_row = next(
        row
        for row in payload["rows"]
        if row["lake"] == "wdc"
        and row["variant"] == "current"
        and row["model"] == "teacher"
    )
    assert teacher_row["teacher_feature_coverage"]["cached_hidden_fraction"] == 0.75
    assert (
        teacher_row["teacher_feature_coverage"][
            "unexpected_pooled_embedding_fallback_objects"
        ]
        == 0
    )

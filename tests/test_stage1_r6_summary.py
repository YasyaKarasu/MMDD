import json
from pathlib import Path

import summarize_stage1_r6


def test_student_selection_reuses_r5_only_when_weight_and_tau_match(
    monkeypatch,
    tmp_path: Path,
) -> None:
    r5_selection = tmp_path / "r5.selection.json"
    monkeypatch.setattr(
        summarize_stage1_r6,
        "_r5_paths",
        lambda root, lake: (r5_selection, tmp_path / "r5.metrics.json"),
    )

    assert summarize_stage1_r6._student_selection(
        tmp_path, tmp_path / "r6", "entitables", 1.0, 1.0, 1.0
    ) == r5_selection


def test_student_selection_uses_r6_when_adaptive_tau_differs(tmp_path: Path) -> None:
    output_root = tmp_path / "r6"

    assert summarize_stage1_r6._student_selection(
        tmp_path, output_root, "wdc", 1.0, 0.5, 0.7
    ) == (
        output_root
        / "taskE_modality_balance/wdc/relation_weight_1/student_path.pt.selection.json"
    )


def test_student_selection_uses_r6_for_weighted_relation_loss(tmp_path: Path) -> None:
    output_root = tmp_path / "r6"

    assert summarize_stage1_r6._student_selection(
        tmp_path, output_root, "entitables", 4.0, 1.0, 1.0
    ) == (
        output_root
        / "taskE_modality_balance/entitables/relation_weight_4/student_path.pt.selection.json"
    )


def test_final_report_includes_completed_task_f(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "r6"
    for lake in ("entitables", "wdc"):
        task_a = {
            "systems": {
                "student": {
                    "opposite_edge_scale_ratios": {"pair": {"ratio": 1.0}}
                }
            }
        }
        task_c = {
            "selected_tau": 0.7,
            "manual_r5_tau": 0.7,
            "same_candidate_pool": True,
        }
        final = {
            "metrics": {
                "recall@10": 0.5,
                "direct": {"recall@10": 0.5},
                "evidence": {"recall@10": 0.1},
                "positive_evidence_path_coverage@10": 0.1,
                "mrr@50": 0.3,
            },
            "delta_vs_r5": {
                "recall@10": {"mean": 0.0, "ci_low": -0.01, "ci_high": 0.01}
            },
        }
        task_f = {
            "student_evidence_recall@10": {
                "task_f": 0.11,
                "delta": {"mean": 0.01, "ci_low": -0.01, "ci_high": 0.03},
            },
            "teacher_rerank_delta@10": {
                "task_f": 0.04,
                "difference_in_differences": {
                    "mean": 0.02,
                    "ci_low": 0.0,
                    "ci_high": 0.04,
                },
            },
        }
        for relative, payload in (
            (f"taskA_scale_diagnostic/{lake}.json", task_a),
            (f"taskC_adaptive_tau/{lake}.json", task_c),
            (f"task_final/{lake}.json", final),
            (f"taskF_table_representation/{lake}/summary.json", task_f),
            (
                f"taskF_table_representation/tokens_per_group_4/{lake}/summary.json",
                task_f,
            ),
            (
                f"taskF_table_representation/tokens_per_group_4/retrained/{lake}/summary.json",
                task_f,
            ),
        ):
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    r5_metrics = {
        "recall@10": 0.4,
        "evidence": {"recall@10": 0.05},
        "positive_evidence_path_coverage@10": 0.04,
    }
    r5_path = tmp_path / "r5.json"
    r5_path.write_text(
        json.dumps({"systems": {"student": {"metrics": r5_metrics}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        summarize_stage1_r6,
        "_r5_paths",
        lambda root, lake: (tmp_path / f"{lake}.selection.json", r5_path),
    )
    monkeypatch.setattr(
        summarize_stage1_r6,
        "_task_e_metrics",
        lambda output_root, lake, weight: r5_metrics,
    )
    configuration = {
        "lakes": {
            lake: {
                "fusion": {"name": "fusion"},
                "aggregation": {"name": "aggregation"},
                "relation_weight": 2,
                "adaptive_tau": 0.7,
            }
            for lake in ("entitables", "wdc")
        }
    }
    task_e = {
        "selected": {
            "relation_weight": 2,
            "evidence_recall@10": 0.1,
            "coverage@10": 0.1,
        },
        "r5_manual_baseline": {
            "fused_recall@10": 0.4,
            "evidence_recall@10": 0.05,
            "coverage@10": 0.04,
        },
        "candidates": [],
    }

    summarize_stage1_r6._final_report(
        tmp_path, output_root, configuration, task_e
    )

    final_metrics = json.loads((output_root / "final_metrics.json").read_text())
    assert final_metrics["task_f"]["status"] == "complete"
    report = (output_root / "RESULTS.md").read_text()
    assert "Task F table representation" in report
    assert "F1: more rows and named cells" in report
    assert "F2: more tokens per schema/row" in report
    assert "F2 zero-shot diagnostic" in report
    assert "not the formal F2 result" in report
    assert "Task F was skipped" not in report

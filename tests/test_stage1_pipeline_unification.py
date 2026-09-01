from __future__ import annotations

from collections import defaultdict

from run_stage1_pipeline_unification_task_a import (
    aggregation_configs,
    select_parameterizations,
    select_unified_form,
)
from run_stage1_pipeline_unification_tasks_b_to_e import (
    _aggregation_arguments,
    _best_task_a_row,
    _fusion_arguments,
    _select_task_b_aggregation,
    _task_c_decision,
)
from mmdd_stage1.objectives import PATH_AGGREGATIONS


def _records(values: dict[str, tuple[float, float, float]]):
    records = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, (fused, evidence, coverage) in values.items():
        records[name][10]["fused_recall"] = [fused]
        records[name][10]["evidence_recall"] = [evidence]
        records[name][10]["coverage"] = [coverage]
    return records


def test_task_a_covers_all_path_aggregation_forms() -> None:
    assert {config["form"] for config in aggregation_configs()} == PATH_AGGREGATIONS


def test_parameter_selection_is_lake_local_within_one_form() -> None:
    specifications = {
        "logsumexp__weighted_rrf_e0.05": {
            "aggregation": {"form": "logsumexp"},
            "fusion": {"variant": "weighted_rrf"},
        },
        "logsumexp__weighted_rrf_e0.1": {
            "aggregation": {"form": "logsumexp"},
            "fusion": {"variant": "weighted_rrf"},
        },
    }
    records = _records(
        {
            "logsumexp__weighted_rrf_e0.05": (0.4, 0.1, 0.2),
            "logsumexp__weighted_rrf_e0.1": (0.4, 0.2, 0.1),
        }
    )

    selected = select_parameterizations(records, specifications)

    assert selected == {
        "logsumexp__weighted_rrf": "logsumexp__weighted_rrf_e0.1"
    }


def test_unified_selection_prefers_candidates_passing_both_lakes() -> None:
    rows = {
        "high_mean_but_fails": {
            "point_gate": False,
            "ci_gate": True,
            "minimum_delta": -0.001,
            "mean_delta": 0.02,
        },
        "passes": {
            "point_gate": True,
            "ci_gate": True,
            "minimum_delta": 0.0,
            "mean_delta": 0.001,
        },
    }

    selected, eligible = select_unified_form(rows)

    assert selected == "passes"
    assert eligible


def test_unified_selection_returns_maximin_when_training_is_needed() -> None:
    rows = {
        "a": {
            "point_gate": False,
            "ci_gate": True,
            "minimum_delta": -0.03,
            "mean_delta": 0.01,
        },
        "b": {
            "point_gate": False,
            "ci_gate": True,
            "minimum_delta": -0.01,
            "mean_delta": -0.005,
        },
    }

    selected, eligible = select_unified_form(rows)

    assert selected == "b"
    assert not eligible


def test_best_task_a_row_filters_form_and_fusion_then_uses_metrics() -> None:
    summary = {
        "rows": {
            "rrf": {
                "aggregation_form": "comb_mnz",
                "fusion_form": "weighted_rrf",
                "per_lake": {
                    "wdc": {
                        "exact_configuration": "comb_mnz__weighted_rrf_e0.5",
                        "metrics": {
                            "recall@10": 0.7,
                            "evidence": {"recall@10": 0.2},
                        },
                    }
                },
            },
            "normalized": {
                "aggregation_form": "comb_mnz",
                "fusion_form": "normalized_score",
                "per_lake": {
                    "wdc": {
                        "exact_configuration": "comb_mnz__normalized_score_zscore_e0.1",
                        "metrics": {
                            "recall@10": 0.6,
                            "evidence": {"recall@10": 0.3},
                        },
                    }
                },
            },
        }
    }

    assert _best_task_a_row(summary, "wdc", "comb_mnz") == summary["rows"][
        "rrf"
    ]["per_lake"]["wdc"]
    assert _best_task_a_row(
        summary, "wdc", "comb_mnz", "normalized_score"
    ) == summary["rows"]["normalized"]["per_lake"]["wdc"]


def test_unified_training_arguments_include_all_persisted_parameters() -> None:
    configuration = {
        "aggregation": {
            "form": "softmax_weighted_mean",
            "top_k": 3,
            "temperature": 0.3,
            "power": 2.5,
        },
        "fusion": {
            "mode": "normalized_score",
            "normalization": "zscore",
            "evidence_weight": 0.25,
        },
    }

    assert _aggregation_arguments(configuration) == [
        "--evidence-aggregation",
        "softmax_weighted_mean",
        "--evidence-top-k",
        "3",
        "--evidence-temperature",
        "0.3",
        "--evidence-power",
        "2.5",
    ]
    assert _fusion_arguments(configuration) == [
        "--fusion-mode",
        "normalized_score",
        "--direct-weight",
        "1",
        "--evidence-weight",
        "0.25",
        "--fusion-score-normalization",
        "zscore",
        "--fusion-score-temperature",
        "1",
    ]


def test_task_b_adopts_selected_aggregation_only_when_both_lakes_pass() -> None:
    rows = {
        "entitables/comb_mnz": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.01},
        },
        "wdc/comb_mnz": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.02},
        },
    }

    assert _select_task_b_aggregation(rows, "comb_mnz") == ("comb_mnz", True)


def test_task_b_falls_back_to_best_passing_aggregation() -> None:
    rows = {
        "entitables/comb_mnz": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.01},
        },
        "wdc/comb_mnz": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": False,
            "delta_vs_r5_teacher": {"delta_mean": -0.04},
        },
        "entitables/logsumexp": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.02},
        },
        "wdc/logsumexp": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.02},
        },
        "entitables/topk_sum": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.01},
        },
        "wdc/topk_sum": {
            "point_gate_minus_3pt": True,
            "ci_gate_minus_3pt": True,
            "delta_vs_r5_teacher": {"delta_mean": -0.005},
        },
    }

    assert _select_task_b_aggregation(rows, "comb_mnz") == ("topk_sum", False)


def test_task_b_uses_maximin_when_no_aggregation_passes() -> None:
    rows = {
        "entitables/logsumexp": {
            "point_gate_minus_3pt": False,
            "ci_gate_minus_3pt": False,
            "delta_vs_r5_teacher": {"delta_mean": -0.05},
        },
        "wdc/logsumexp": {
            "point_gate_minus_3pt": False,
            "ci_gate_minus_3pt": False,
            "delta_vs_r5_teacher": {"delta_mean": -0.04},
        },
        "entitables/topk_sum": {
            "point_gate_minus_3pt": False,
            "ci_gate_minus_3pt": False,
            "delta_vs_r5_teacher": {"delta_mean": -0.03},
        },
        "wdc/topk_sum": {
            "point_gate_minus_3pt": False,
            "ci_gate_minus_3pt": False,
            "delta_vs_r5_teacher": {"delta_mean": -0.02},
        },
    }

    assert _select_task_b_aggregation(rows, "topk_sum") == ("topk_sum", False)


def test_task_c_retains_r5_when_either_lake_misses_point_gate() -> None:
    decision = _task_c_decision(
        {
            "entitables": {"gate": True},
            "wdc": {"gate": False},
        }
    )

    assert decision == {
        "all_r5_point_gates_passed": False,
        "adopt_unified_as_primary": False,
        "primary_configuration": "r5",
        "unified_role": "training_consistent_upper_bound_analysis",
    }


def test_task_c_adopts_unified_only_when_both_lakes_pass() -> None:
    decision = _task_c_decision(
        {
            "entitables": {"gate": True},
            "wdc": {"gate": True},
        }
    )

    assert decision == {
        "all_r5_point_gates_passed": True,
        "adopt_unified_as_primary": True,
        "primary_configuration": "unified",
        "unified_role": "primary_paper_configuration",
    }

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.construction import build_stage1_training_artifacts
from mmdd_stage1.data import EdgeExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import (
    ListScores,
    global_edge_positive_ids,
    score_edge_batch_in_batch,
)
from mmdd_stage1.selection import MetricCriterion, select_lexicographic
from mmdd_stage1.training import _recall_at_one, train_student_edges
from materialize_stage1_r11_supervision import _filter_teacher_candidates
from fit_stage1_r11_relation_affine import _mapped_score, _rrf_top
from run_stage1_r11_task_e import (
    empty_intervention_stats,
    intervene_paths,
    select_evidence,
)
from run_stage1_r11_task_f import (
    _accumulate,
    _empty,
    _linear_fusion,
    _selection_result,
    reserved_channel_fusion,
)
from summarize_stage1_r11 import grouped_paired_bootstrap
from mmdd_stage1.retrieval import fuse_ranked_channels


def _feature(object_id: str, value: float) -> ObjectFeatures:
    return ObjectFeatures(
        object_id,
        "table",
        torch.tensor([value, value + 0.1, 1.0 - value, -value]),
    )


def _edge_fixture() -> tuple[FeatureStore, list[EdgeExample]]:
    store = FeatureStore(
        {
            object_id: _feature(object_id, value)
            for object_id, value in {
                "q": 0.1,
                "p1": 0.2,
                "p2": 0.3,
                "n1": 0.8,
                "n2": 0.9,
            }.items()
        }
    )
    examples = [
        EdgeExample(
            "q",
            ("p1", "n1"),
            0,
            source_type="table",
            destination_type="table",
        ),
        EdgeExample(
            "q",
            ("p2", "n2"),
            0,
            source_type="table",
            destination_type="table",
        ),
    ]
    return store, examples


def test_global_positive_neighbors_are_excluded_from_in_batch_negatives():
    store, examples = _edge_fixture()
    model = StudentJoinabilityModel(4, 3)
    known = global_edge_positive_ids(examples)
    fixed_audit: dict[str, int] = {}
    legacy_audit: dict[str, int] = {}

    fixed = score_edge_batch_in_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        known_positive_ids=known,
        use_global_positive_mask=True,
        sampling_seed=13,
        sampling_context="epoch=0:step=1",
        expansion_audit=fixed_audit,
    )
    legacy = score_edge_batch_in_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        known_positive_ids=known,
        use_global_positive_mask=False,
        sampling_seed=13,
        sampling_context="epoch=0:step=1",
        expansion_audit=legacy_audit,
    )

    assert fixed.candidate_mask.sum(dim=1).tolist() == [3, 3]
    assert legacy.candidate_mask.sum(dim=1).tolist() == [4, 4]
    assert fixed_audit["known_positive_as_negative"] == 0
    assert legacy_audit["known_positive_as_negative"] == 2


def test_no_in_batch_still_optimizes_supervised_ranking_and_honors_step_cap():
    store, examples = _edge_fixture()
    model = StudentJoinabilityModel(4, 3)
    history = train_student_edges(
        model,
        examples,
        store,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        epochs=2,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.0,
        in_batch_negatives=False,
        max_optimizer_updates=1,
    )

    assert len(history) == 1
    assert history[0]["supervised_loss"] > 0
    assert history[0]["optimizer_updates_total"] == 1
    assert history[0]["optimizer_update_budget_exhausted"]


def test_fixed_pool_dedup_and_row_coverage_retain_complementary_evidence():
    query = ObjectFeatures(
        "q",
        "table",
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        row_embeddings=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
    )
    store = FeatureStore(
        {
            "q": query,
            "e1": ObjectFeatures(
                "e1", "text", torch.tensor([1.0, 0.0, 0.0, 0.0])
            ),
            "e1_duplicate": ObjectFeatures(
                "e1_duplicate", "text", torch.tensor([1.0, 0.0, 0.0, 0.0])
            ),
            "e2": ObjectFeatures(
                "e2", "text", torch.tensor([0.0, 1.0, 0.0, 0.0])
            ),
        }
    )
    paths = [
        {"kind": "evidence", "evidence_id": "e1", "path_score": 2.0},
        {
            "kind": "evidence",
            "evidence_id": "e1_duplicate",
            "path_score": 1.9,
        },
        {"kind": "evidence", "evidence_id": "e2", "path_score": 1.8},
    ]
    content_keys = {"e1": "same", "e1_duplicate": "same", "e2": "other"}

    e0, _ = select_evidence(
        "e0_top_quality",
        paths,
        query_id="q",
        store=store,
        content_keys=content_keys,
        top_l=20,
        budget=2,
        support_cache={},
    )
    e1, _ = select_evidence(
        "e1_content_dedup",
        paths,
        query_id="q",
        store=store,
        content_keys=content_keys,
        top_l=20,
        budget=2,
        support_cache={},
    )
    e2, _ = select_evidence(
        "e2_row_coverage",
        paths,
        query_id="q",
        store=store,
        content_keys=content_keys,
        top_l=20,
        budget=2,
        support_cache={},
    )

    assert e0 == ["e1", "e1_duplicate"]
    assert e1 == ["e1", "e2"]
    assert set(e2) == {"e1", "e2"}


def test_same_row_intervention_is_label_blind_and_preserves_direct_path():
    store = FeatureStore(
        {
            "q": ObjectFeatures(
                "q",
                "table",
                torch.tensor([1.0, 0.0]),
                row_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            "text_row0": ObjectFeatures(
                "text_row0", "text", torch.tensor([1.0, 0.0])
            ),
            "text_row1": ObjectFeatures(
                "text_row1", "text", torch.tensor([0.0, 1.0])
            ),
            "image_row1": ObjectFeatures(
                "image_row1", "image", torch.tensor([0.0, 1.0])
            ),
        }
    )
    paths = [
        {"kind": "direct", "target_id": "t", "path_score": 9.0},
        {
            "kind": "evidence",
            "evidence_id": "text_row0",
            "evidence_type": "text",
            "path_score": 1.2,
        },
        {
            "kind": "evidence",
            "evidence_id": "text_row1",
            "evidence_type": "text",
            "path_score": 1.1,
        },
        {
            "kind": "evidence",
            "evidence_id": "image_row1",
            "evidence_type": "image",
            "path_score": 1.05,
        },
    ]
    stats = empty_intervention_stats()

    intervened = intervene_paths(
        paths,
        intervention="duplicate_same_row",
        query_id="q",
        store=store,
        support_cache={},
        stats=stats,
    )

    assert intervened[0] == paths[0]
    assert [
        path.get("evidence_id") for path in intervened if path.get("kind") == "evidence"
    ] == ["text_row0", "text_row1", "text_row1"]
    assert stats["same_predicted_row_matches"] == 1
    assert stats["fallback_matches"] == 0
    assert stats["image_paths_replaced"] == 1
    assert stats["path_score_gap_sum"] == pytest.approx(0.05)


def test_r10_regression_uses_dev_frozen_rule_without_reselection(tmp_path):
    source = tmp_path / "dev_metrics.json"
    source.write_text(
        json.dumps({"selection": {"selected": "f5_reserved_half"}}),
        encoding="utf-8",
    )

    selection = _selection_result(
        {},
        evaluation_role="r10_test_regression",
        frozen_selection="f5_reserved_half",
        frozen_selection_source=source,
    )

    assert selection["selected"] == "f5_reserved_half"
    assert selection["reselected"] is False
    assert selection["selection_scope"] == "r10_test_regression_historical_only"

    intervention = _selection_result(
        {},
        evaluation_role="dev",
        frozen_selection="f5_reserved_half",
        frozen_selection_source=source,
    )
    assert intervention["selected"] == "f5_reserved_half"
    assert intervention["reselected"] is False
    assert intervention["selection_scope"] == "dev_fixed_intervention"


def test_grouped_bootstrap_preserves_ratio_denominator_and_pairing():
    left = [
        {"query_id": "q1", "value": 1, "n": 2},
        {"query_id": "q2", "value": 2, "n": 3},
    ]
    right = [
        {"query_id": "q1", "value": 0, "n": 2},
        {"query_id": "q2", "value": 1, "n": 3},
    ]

    result = grouped_paired_bootstrap(
        left,
        right,
        {"q1": "source-a", "q2": "source-b"},
        key=lambda row: row["query_id"],
        query_id=lambda row: row["query_id"],
        numerator=lambda row: row["value"],
        denominator=lambda row: row["n"],
        iterations=100,
        seed=13,
    )

    assert result["difference"] == pytest.approx(2 / 5)
    assert result["denominator"] == 5
    assert result["groups"] == 2


def test_union_linear_lambda_zero_preserves_direct_ranking_exactly():
    direct = [
        {"target_id": "a", "direct_score": 3.0},
        {"target_id": "b", "direct_score": 2.0},
        {"target_id": "c", "direct_score": 1.0},
    ]
    evidence = [
        {"target_id": "c", "evidence_score": 99.0},
    ]
    scale = {"q50": 2.0, "denominator": 2.0}

    fused = _linear_fusion(
        direct,
        evidence,
        direct_scale=scale,
        evidence_scale={"q50": 0.0, "denominator": 1.0},
        evidence_weight=0.0,
    )

    assert [row["target_id"] for row in fused] == ["a", "b", "c"]


def test_reserved_fusion_allocates_each_k_independently():
    direct = [
        {"target_id": f"d{index}", "direct_score": 10.0 - index}
        for index in range(8)
    ]
    evidence = [
        {"target_id": f"e{index}", "evidence_score": 10.0 - index}
        for index in range(8)
    ]

    top10 = reserved_channel_fusion(direct, evidence, k=10)
    top4 = reserved_channel_fusion(direct, evidence, k=4)

    assert [row["target_id"] for row in top4] == ["d0", "e0", "d1", "e1"]
    assert [row["target_id"] for row in top10[:4]] == ["d0", "e0", "d1", "e1"]


def test_historical_weighted_rrf_cannot_admit_outside_full_d100_at_k10():
    direct = [
        {
            "target_id": f"d{index:03d}",
            "direct_score": 1.0 - index / 1000.0,
            "paths": [{"kind": "direct"}],
        }
        for index in range(100)
    ]
    evidence = [
        {
            "target_id": "outside",
            "evidence_score": 100.0,
            "paths": [{"kind": "evidence"}],
        }
    ]

    fused = fuse_ranked_channels(
        direct,
        evidence,
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
    )

    assert "outside" not in {row["target_id"] for row in fused[:10]}


def test_fusion_mechanism_denominator_excludes_explicit_queries():
    record = {
        "query_id": "q",
        "query_kind": "explicit",
        "query_row_count": 5,
        "positive_target_ids": ["t"],
        "positive_evidence_by_target": {"t": ["e"]},
        "positive_evidence_rows_by_target": {"t": {"e": [0]}},
    }
    ranking = [
        {"target_id": "t", "selected_evidence_ids": ["e"], "score": 1.0}
    ]
    metrics = _empty()

    _accumulate(
        metrics,
        record,
        {k: ranking for k in (10, 20, 50)},
        {k: ranking for k in (10, 20, 50)},
        {"t"},
    )

    assert metrics["implicit_positive_pairs"] == 0
    assert metrics["valid_path"][10] == 0


def test_list_recall_at_one_accepts_any_positive_and_ignores_padding():
    scores = ListScores(
        logits=torch.tensor([[0.2, 0.8, 99.0], [0.9, 0.1, 99.0]]),
        candidate_mask=torch.tensor([[True, True, False], [True, True, False]]),
        positive_indices=torch.tensor([0, 0]),
        positive_mask=torch.tensor(
            [[True, True, False], [False, True, False]], dtype=torch.bool
        ),
    )

    assert _recall_at_one(scores) == (1, 2)


def test_teacher_candidate_filter_removes_only_optional_objects():
    targets = [
        {
            "query_id": "q",
            "positive_target_ids": ["p"],
            "positive_evidence_by_target": {"p": ["e"]},
            "candidates": [
                {"target_id": "p", "evidence_ids": ["e"]},
                {"target_id": "n", "evidence_ids": ["missing-e"]},
                {"target_id": "missing-t", "evidence_ids": []},
            ],
        }
    ]
    edges = [
        {
            "query_id": "q",
            "positive_id": "p",
            "positive_ids": ["p"],
            "candidate_ids": ["p", "n", "missing-t"],
            "confirmed_labels": [1, None, None],
        }
    ]

    audit = _filter_teacher_candidates(targets, edges, {"q", "p", "e", "n"})

    assert [row["target_id"] for row in targets[0]["candidates"]] == ["p", "n"]
    assert targets[0]["candidates"][1]["evidence_ids"] == []
    assert edges[0]["candidate_ids"] == ["p", "n"]
    assert edges[0]["confirmed_labels"] == [1, None]
    assert audit == {
        "target_candidates_removed": 1,
        "target_evidence_removed": 1,
        "edge_candidates_removed": 1,
    }


def test_r11_training_selection_uses_all_declared_tie_breakers():
    candidates = [
        {
            "id": "late",
            "step": 178,
            "metrics": {
                "valid_pool": 10,
                "row_b": 0.2,
                "valid_b": 8,
                "direct": {"recall@10": 0.3},
            },
        },
        {
            "id": "better-row",
            "step": 178,
            "metrics": {
                "valid_pool": 10,
                "row_b": 0.21,
                "valid_b": 7,
                "direct": {"recall@10": 0.9},
            },
        },
        {
            "id": "early",
            "step": 89,
            "metrics": {
                "valid_pool": 10,
                "row_b": 0.21,
                "valid_b": 7,
                "direct": {"recall@10": 0.9},
            },
        },
    ]

    selected, audit = select_lexicographic(
        candidates,
        [
            MetricCriterion("valid_pool"),
            MetricCriterion("row_b"),
            MetricCriterion("valid_b"),
            MetricCriterion("direct.recall@10"),
        ],
        id_key="id",
    )

    assert selected["id"] == "early"
    assert audit["decisions"][-1]["reason"].startswith("step:")


def _table(table_id: str, split: str) -> dict:
    return {
        "table_id": table_id,
        "split": split,
        "source_table_id": f"source-{table_id}",
        "columns": [{"column_index": 0, "column_name": "entity"}],
        "rows": [
            {
                "row_id": 0,
                "cells": [{"column_index": 0, "text": table_id}],
            }
        ],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_invisible_split_labels_do_not_change_train_fit_materialization(tmp_path):
    queries = [_table("q-fit", "train"), _table("q-hidden", "dev")]
    targets = [_table(value, "") for value in ("t-fit", "t-hidden", "n1", "n2")]
    assets = [
        {
            "asset_id": f"e-{target['table_id']}",
            "asset_type": "text",
            "content": target["table_id"],
            "source_table_id": target["source_table_id"],
        }
        for target in targets
    ]
    qrels = [
        {"query_table_id": "q-fit", "target_table_id": "t-fit"},
        {"query_table_id": "q-hidden", "target_table_id": "t-hidden"},
    ]
    recoveries = [
        {
            "query_table_id": "q-fit",
            "target_table_id": "t-fit",
            "query_row_id": 0,
            "evidence": {"asset_id": "e-t-fit"},
        },
        {
            "query_table_id": "q-hidden",
            "target_table_id": "t-hidden",
            "query_row_id": 0,
            "evidence": {"asset_id": "e-t-hidden"},
        },
    ]
    for name, records in {
        "query_tables": queries,
        "data_lake_tables": targets,
        "bridge_assets": assets,
        "qrels": qrels,
        "evidence_recoveries": recoveries,
    }.items():
        _write_jsonl(tmp_path / f"{name}.jsonl", records)

    before = build_stage1_training_artifacts(
        tmp_path,
        dataset_name="synthetic",
        supervision_query_ids={"q-fit"},
    )
    qrels[1]["target_table_id"] = "n2"
    recoveries[1]["target_table_id"] = "n2"
    _write_jsonl(tmp_path / "qrels.jsonl", qrels)
    _write_jsonl(tmp_path / "evidence_recoveries.jsonl", recoveries)
    after = build_stage1_training_artifacts(
        tmp_path,
        dataset_name="synthetic",
        supervision_query_ids={"q-fit"},
    )

    assert before["edge_lists"] == after["edge_lists"]
    assert before["target_lists"] == after["target_lists"]


def test_metric_tolerance_does_not_override_integer_counts():
    selected, _audit = select_lexicographic(
        [
            {"step": 1, "metrics": {"count": 10}},
            {"step": 2, "metrics": {"count": 11}},
        ],
        [MetricCriterion("count")],
        tolerance=100.0,
    )
    assert selected["metrics"]["count"] == 11


def test_positive_affine_map_preserves_same_relation_order():
    models = {"table_to_text": {"scale": 2.0, "bias": -3.0}}
    raw = [-2.0, 0.5, 4.0]
    mapped = [
        _mapped_score(value, "table", "text", models) for value in raw
    ]
    assert sorted(range(3), key=lambda index: raw[index]) == sorted(
        range(3), key=lambda index: mapped[index]
    )


def test_equal_rrf_allows_evidence_channel_to_change_top_one():
    direct = {"direct": 2.0, "shared": 1.0}
    evidence = {"evidence": 2.0, "shared": 1.0}
    assert _rrf_top(direct, evidence) == "shared"

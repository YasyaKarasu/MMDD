from __future__ import annotations

import json

import pytest
import torch

from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import rank_detailed_paths
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import score_target_batch
from mmdd_stage1.row_support import (
    IsotonicModel,
    fit_isotonic,
    greedy_row_bundle,
    greedy_row_bundle_tensor,
    load_evidence_content_keys,
)
from materialize_stage1_content_keys import run as materialize_content_keys


def test_fit_isotonic_merges_decreasing_adjacent_blocks():
    model = fit_isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])

    assert model.values == pytest.approx((0.0, 0.5, 1.0))
    assert model.predict(-1.0) == pytest.approx(0.0)
    assert model.predict(0.25) == pytest.approx(0.5)
    assert model.predict(1.0) == pytest.approx(1.0)
    assert IsotonicModel.from_json(model.to_json()) == model


def test_greedy_row_bundle_rewards_complementary_rows_and_deduplicates_choice():
    selected, score = greedy_row_bundle(
        [
            {"evidence_id": "same-row", "quality": 0.9},
            {"evidence_id": "new-row", "quality": 0.8},
            {"evidence_id": "duplicate", "quality": 0.85},
        ],
        row_support={
            "same-row": [1.0, 0.0],
            "new-row": [0.0, 1.0],
            "duplicate": [1.0, 0.0],
        },
        budget=2,
        threshold=0.0,
    )

    assert selected == ["same-row", "new-row"]
    assert score == pytest.approx(0.85)


def test_greedy_row_bundle_stops_when_threshold_removes_all_gain():
    selected, score = greedy_row_bundle(
        [{"evidence_id": "weak", "quality": 0.4}],
        row_support={"weak": [1.0]},
        budget=4,
        threshold=0.5,
    )

    assert selected == []
    assert score == 0.0


def test_greedy_row_bundle_tensor_deduplicates_and_keeps_selected_gradients():
    qualities = torch.tensor([0.9, 0.85, 0.8], requires_grad=True)
    support = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    selected, score = greedy_row_bundle_tensor(
        qualities,
        support,
        torch.ones(3, dtype=torch.bool),
        budget=2,
        top_l=3,
        threshold=0.0,
        content_groups=torch.tensor([0, 0, 1]),
    )
    score.backward()

    assert selected == [0, 2]
    assert score.item() == pytest.approx(0.85)
    assert qualities.grad.tolist() == pytest.approx([0.5, 0.0, 0.5])


def test_g5_all_mask_returns_zero_with_finite_zero_gradients():
    query_scores = torch.tensor([[[0.7, 0.6]]], requires_grad=True)
    target_scores = torch.tensor([[[0.8, 0.5]]], requires_grad=True)
    score = PathAggregator(
        "greedy_row_support",
        2,
        path_combination="min",
        threshold=0.5,
    )(
        query_scores,
        target_scores,
        torch.zeros_like(query_scores, dtype=torch.bool),
        row_support=torch.ones(1, 1, 2, 3),
    )
    score.sum().backward()

    assert score.item() == 0.0
    assert query_scores.grad.tolist() == [[[0.0, 0.0]]]
    assert target_scores.grad.tolist() == [[[0.0, 0.0]]]
    assert torch.isfinite(query_scores.grad).all()
    assert torch.isfinite(target_scores.grad).all()


def test_g5_tensor_and_scalar_paths_match_with_threshold_and_content_dedup():
    qualities = torch.tensor([[[0.9, 0.85, 0.8]]], requires_grad=True)
    support = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]])
    aggregator = PathAggregator(
        "greedy_row_support",
        2,
        path_combination="min",
        threshold=0.5,
        row_support_top_l=3,
    )
    tensor_score = aggregator(
        torch.ones_like(qualities),
        qualities,
        torch.ones_like(qualities, dtype=torch.bool),
        row_support=support,
        content_groups=torch.tensor([[[0, 0, 1]]]),
    )
    tensor_score.sum().backward()
    ranked = rank_detailed_paths(
        {
            "target": [
                {
                    "kind": "evidence",
                    "evidence_id": evidence_id,
                    "query_evidence_score": 1.0,
                    "evidence_target_score": float(quality),
                    "path_score": float(quality),
                    "row_support": rows,
                    "evidence_content_key": content,
                }
                for evidence_id, quality, rows, content in (
                    ("e0", 0.9, [1.0, 0.0], "same"),
                    ("e1", 0.85, [1.0, 0.0], "same"),
                    ("e2", 0.8, [0.0, 1.0], "new"),
                )
            ]
        },
        aggregator=aggregator,
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
        gated_evidence_min_paths=2,
        gated_evidence_quantile=0.75,
    )

    assert tensor_score.item() == pytest.approx(0.7)
    assert ranked["evidence"][0]["evidence_score"] == pytest.approx(0.7)
    assert ranked["evidence"][0]["selected_evidence_ids"] == ["e0", "e2"]
    assert torch.isfinite(qualities.grad).all()


def test_path_aggregator_records_frozen_row_support_provenance(tmp_path):
    model_path = tmp_path / "row_support.json"
    model_path.write_text(
        json.dumps(
            {
                "models": {
                    "text": {"upper_bounds": [1.0], "values": [0.5]},
                    "image": {"upper_bounds": [1.0], "values": [0.25]},
                }
            }
        ),
        encoding="utf-8",
    )
    content_path = tmp_path / "content_keys.jsonl"
    content_path.write_text(
        "\n".join(
            json.dumps({"object_id": object_id, "content_key": content_key})
            for object_id, content_key in (
                ("e0", "text:same"),
                ("e1", "text:same"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    aggregator = PathAggregator(
        "greedy_row_support",
        4,
        row_support_model=model_path,
        row_support_top_l=20,
        evidence_content_keys=content_path,
    )

    assert aggregator.row_support_models["text"].predict(0.0) == pytest.approx(0.5)
    assert aggregator.config()["row_support_model"] == str(model_path.resolve())
    assert len(aggregator.config()["row_support_model_sha256"]) == 64
    assert aggregator.config()["evidence_content_keys"] == str(
        content_path.resolve()
    )
    assert len(aggregator.config()["evidence_content_keys_sha256"]) == 64
    assert aggregator.content_key("e0", torch.tensor([1.0])) == aggregator.content_key(
        "e1", torch.tensor([-1.0])
    )


def test_materialized_content_keys_hash_exact_text_and_image_bytes(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"same-image-bytes")
    image_b.write_bytes(b"same-image-bytes")
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"object_id": "t0", "object_type": "text", "text": "same"},
                {"object_id": "t1", "object_type": "text", "text": "same"},
                {"object_id": "i0", "object_type": "image", "image": str(image_a)},
                {"object_id": "i1", "object_type": "image", "image": str(image_b)},
                {"object_id": "q", "object_type": "table"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "content_keys.jsonl"

    metadata = materialize_content_keys(objects, output)
    keys, sha256 = load_evidence_content_keys(output)

    assert keys["t0"] == keys["t1"]
    assert keys["i0"] == keys["i1"]
    assert keys["t0"] != keys["i0"]
    assert metadata["duplicate_content_groups"] == 2
    assert metadata["objects_in_duplicate_groups"] == 4
    assert metadata["output_sha256"] == sha256


def test_score_target_batch_supplies_frozen_row_support_to_g5(tmp_path):
    model_path = tmp_path / "row_support.json"
    model_path.write_text(
        json.dumps(
            {
                "models": {
                    evidence_type: {
                        "upper_bounds": [0.0, 1.0],
                        "values": [0.0, 1.0],
                    }
                    for evidence_type in ("text", "image")
                }
            }
        ),
        encoding="utf-8",
    )
    store = FeatureStore(
        {
            "q": ObjectFeatures(
                "q",
                "table",
                torch.tensor([1.0, 0.0]),
                row_embeddings=torch.eye(2),
            ),
            "t0": ObjectFeatures("t0", "table", torch.tensor([1.0, 0.0])),
            "t1": ObjectFeatures("t1", "table", torch.tensor([0.0, 1.0])),
            "e0": ObjectFeatures("e0", "text", torch.tensor([1.0, 0.0])),
            "e1": ObjectFeatures("e1", "text", torch.tensor([0.0, 1.0])),
        }
    )
    student = StudentJoinabilityModel(
        input_dim=2,
        student_dim=2,
        confidence_transform=True,
    )
    store.preload_embeddings(store.object_ids())
    scores = score_target_batch(
        student,
        [
            TargetExample(
                "q",
                (
                    TargetCandidate("t0", ("e0", "e1")),
                    TargetCandidate("t1", ("e0",)),
                ),
                direct_positive_index=0,
                evidence_positive_index=0,
                positive_target_ids=("t0",),
            )
        ],
        store,
        torch.device("cpu"),
        PathAggregator(
            "greedy_row_support",
            2,
            path_combination="min",
            row_support_model=model_path,
        ),
        student_score_space="confidence",
    )
    scores.evidence.logits.sum().backward()

    assert scores.evidence.logits.shape == (1, 2)
    assert torch.isfinite(scores.evidence.logits).all()
    assert student.relations["table_to_text"].grad is not None
    assert student.relations["text_to_table"].grad is not None

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage2.verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    EvidenceRef,
    best_image_region,
    best_text_span,
    build_evidence_bundles,
    build_row_evidence_query,
    focus_relevance,
    joint_candidate_probabilities,
    joint_relevance,
    propose_image_regions,
    semantic_joinability,
)
from mmdd_stage2.pipeline import LocalizedEvidence, Stage2Verifier
from mmdd_stage2.data import Stage2ObjectIndex
from mmdd_stage2.training import ColumnTrainingExample, train_candidate_scorer
from mmdd_stage2.checkpoints import load_candidate_scorer, save_candidate_scorer


def test_build_evidence_bundles_selects_unique_evidence_by_path_score():
    bundles = build_evidence_bundles(
        [
            {
                "target_id": "t1",
                "score": 4.0,
                "paths": [
                    {"kind": "evidence", "evidence_id": "e2", "evidence_type": "image", "path_score": 1.0},
                    {"kind": "direct", "path_score": 3.0},
                    {"kind": "evidence", "evidence_id": "e1", "evidence_type": "text", "path_score": 2.0},
                    {"kind": "evidence", "evidence_id": "e1", "evidence_type": "text", "path_score": 1.5},
                ],
            }
        ],
        top_k_evidence=2,
    )

    assert bundles[0].target_id == "t1"
    assert bundles[0].evidence_ids == ("e1", "e2")
    assert [path["path_score"] for path in bundles[0].paths] == [2.0, 1.5, 1.0]
    assert sum(item.weight for item in bundles[0].evidence) == pytest.approx(1.0)
    assert bundles[0].evidence[0].path_score > 2.0


def test_candidate_column_probabilities_match_table_times_column_formula():
    retrieval_scores = torch.tensor([[2.0, 1.0]])
    column_logits = torch.tensor([[[0.0, 1.0], [2.0, -5.0]]])
    target_mask = torch.tensor([[True, True]])
    column_mask = torch.tensor([[[True, True], [True, False]]])

    table_probabilities, column_probabilities, joint = joint_candidate_probabilities(
        retrieval_scores, column_logits, target_mask, column_mask
    )

    assert torch.allclose(table_probabilities, torch.softmax(retrieval_scores, dim=-1))
    assert column_probabilities[0, 1].tolist() == [1.0, 0.0]
    assert joint.sum().item() == pytest.approx(1.0)
    assert torch.allclose(joint, table_probabilities.unsqueeze(-1) * column_probabilities)


def test_candidate_column_scorer_uses_both_boundary_states():
    scorer = CandidateColumnScorer(hidden_dim=2)
    open_states = torch.randn(1, 2, 3, 2)
    close_states = torch.randn(1, 2, 3, 2)

    logits = scorer(open_states, close_states)
    logits.sum().backward()

    assert logits.shape == (1, 2, 3)
    assert scorer.weight.weight.grad is not None


def test_row_query_and_joint_relevance_localize_text_and_image():
    query = build_row_evidence_query(
        {"Player": "Messi", "Country": "Argentina"},
        entity_column="Player",
        attribute_name="club",
    )
    evidence = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    entity = torch.tensor([[1.0, 0.0]])
    attribute = torch.tensor([[0.8, 0.2]])
    relevance = joint_relevance(entity, attribute, evidence)
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0], [0.0, 10.0, 10.0, 20.0], [10.0, 10.0, 20.0, 20.0]]
    )

    assert query["entity_anchor"] == "Messi"
    assert query["attribute_name"] == "club"
    assert relevance.sum().item() == pytest.approx(1.0)
    assert best_text_span(relevance, 2) == (0, 2)
    best_patch = boxes[relevance.argmax()]
    assert best_image_region(relevance, boxes, top_fraction=0.25) == tuple(float(value) for value in best_patch)


def test_focus_relevance_and_roi_proposal_use_later_layer_value_features():
    layer = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]]
    )
    relevance = focus_relevance(
        [layer, layer],
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([2, 3, 4, 5]),
    )
    regions = propose_image_regions(
        relevance.reshape(2, 2),
        (200, 100),
        anchors=2,
        min_anchor_distance=1,
        min_side=1,
        max_side=1,
    )

    assert int(relevance.argmax()) == 0
    assert regions
    assert regions[0].box[0] < 100
    assert regions[0].box[1] < 50


def test_semantic_joinability_requires_row_coverage():
    query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    result = semantic_joinability(
        ["NYC", "unsupported"],
        ["New York City", "Boston"],
        query_embeddings=query_embeddings,
        target_embeddings=target_embeddings,
        similarity_threshold=0.8,
        min_coverage=1.0,
    )

    assert result.coverage == pytest.approx(0.5)
    assert not result.joinable


class FakeBackend:
    hidden_dim = 2

    def reader_states(self, query, target, evidence):
        assert sum(weight for _, weight in evidence) == pytest.approx(1.0)
        return torch.tensor([[0.0, 0.0], [4.0, 0.0]]), torch.zeros(2, 2)

    def localize_evidence(self, row, *, entity_column, attribute_name, evidence):
        return LocalizedEvidence(evidence["asset_id"], evidence["asset_type"], 0.9, text=evidence["content"])

    def generate_value(self, row, *, attribute_name, evidence):
        return {"Messi": "Barcelona", "Mbappe": "PSG"}[row["Player"]]

    def embed_texts(self, values):
        vectors = {
            "Barcelona": [1.0, 0.0],
            "PSG": [0.0, 1.0],
            "Argentina": [0.7, 0.7],
            "France": [0.6, 0.8],
        }
        return torch.tensor([vectors.get(value, [0.0, 0.0]) for value in values])


def _table(table_id, columns, rows, **extra):
    return {
        "table_id": table_id,
        "columns": [
            {"column_index": index, "source_column_index": index, "column_name": name}
            for index, name in enumerate(columns)
        ],
        "rows": [
            {
                "row_id": row_index,
                "cells": [
                    {"column_index": column_index, "text": value}
                    for column_index, value in enumerate(values)
                ],
            }
            for row_index, values in enumerate(rows)
        ],
        **extra,
    }


def test_stage2_verifier_runs_column_selection_localization_generation_and_final_check():
    query = _table(
        "q1",
        ["Player", "Country"],
        [["Messi", "Argentina"], ["Mbappe", "France"]],
        query_entity_col=0,
    )
    target = _table(
        "t1",
        ["Country", "Club"],
        [["Argentina", "Barcelona"], ["France", "PSG"]],
    )
    bundle = EvidenceBundle("t1", 2.0, (EvidenceRef("e1", "text", 2.0, 1.0),), ())
    scorer = CandidateColumnScorer(2)
    with torch.no_grad():
        scorer.weight.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        scorer.weight.bias.zero_()

    result = Stage2Verifier(FakeBackend(), scorer, min_row_coverage=1.0).verify(
        query,
        [bundle],
        {"t1": target},
        {"e1": {"asset_id": "e1", "asset_type": "text", "content": "support"}},
        direct_target_ids=["t1"],
    )

    assert result.direct_candidates[0].target_id == "t1"
    assert result.selection.column_name == "Club"
    assert [row.value for row in result.rows] == ["Barcelona", "PSG"]
    assert result.augmented_query["columns"][-1]["column_name"] == "Club"
    assert result.augmented_query["rows"][0]["cells"][-1]["text"] == "Barcelona"
    assert result.semantic_joinability.joinable


def test_candidate_head_training_updates_only_the_small_rata_scorer():
    query = _table("q1", ["Player"], [["Messi"]], query_entity_col=0)
    target = _table("t1", ["Country", "Club"], [["Spain", "Barcelona"]])
    bundle = EvidenceBundle("t1", 2.0, (EvidenceRef("e1", "text", 2.0, 1.0),), ())
    example = ColumnTrainingExample("q1", (bundle,), "t1", 1)
    objects = Stage2ObjectIndex(
        {"q1": query},
        {"t1": target},
        {"e1": {"asset_id": "e1", "asset_type": "text", "content": "support"}},
    )
    scorer = CandidateColumnScorer(2)
    before = scorer.weight.weight.detach().clone()

    history = train_candidate_scorer(
        FakeBackend(),
        scorer,
        [example],
        objects,
        epochs=1,
        learning_rate=0.1,
        weight_decay=0.0,
        seed=1,
    )

    assert history[0]["examples"] == 1
    assert not torch.equal(before, scorer.weight.weight)


def test_candidate_scorer_checkpoint_round_trip(tmp_path):
    scorer = CandidateColumnScorer(2)
    path = tmp_path / "stage2.pt"
    save_candidate_scorer(path, scorer, metadata={"model": "fake"})

    restored = load_candidate_scorer(path, torch.device("cpu"))

    assert torch.equal(restored.weight.weight, scorer.weight.weight)
    assert torch.equal(restored.weight.bias, scorer.weight.bias)

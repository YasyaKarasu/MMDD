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
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import EvidenceRowAssignment, SimilarityEvidenceRouter
from mmdd_stage1.features import FeatureStore, ObjectFeatures
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

    def __init__(self):
        self.embed_batches = []
        self.localization_calls = []
        self.generation_calls = []
        self.reader_evidence_batches = []

    def reader_states(self, query, target, evidence):
        self.reader_evidence_batches.append(tuple(item["asset_id"] for item in evidence))
        return torch.tensor([[0.0, 0.0], [4.0, 0.0]]), torch.zeros(2, 2)

    def localize_evidence(self, row, *, entity_column, attribute_name, evidence):
        self.localization_calls.append((row[entity_column], evidence["asset_id"]))
        return LocalizedEvidence(evidence["asset_id"], evidence["asset_type"], 0.9, text=evidence["content"])

    def generate_value(self, row, *, attribute_name, evidence):
        self.generation_calls.append((row["Player"], evidence.evidence_id))
        return {"Messi": "Barcelona", "Mbappe": "PSG"}[row["Player"]]

    def embed_texts(self, values):
        self.embed_batches.append(tuple(values))
        vectors = {
            "Barcelona": [1.0, 0.0],
            "PSG": [0.0, 1.0],
            "Argentina": [0.7, 0.7],
            "France": [0.6, 0.8],
        }
        return torch.tensor([vectors.get(value, [0.0, 0.0]) for value in values])


class FakeRouter:
    def __init__(self, row_by_evidence):
        self.row_by_evidence = row_by_evidence

    def assign(self, query_id, evidence_ids, *, row_count):
        assert query_id == "q1"
        assert row_count == 2
        return tuple(
            EvidenceRowAssignment(evidence_id, self.row_by_evidence[evidence_id], 0.8)
            for evidence_id in evidence_ids
        )


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
    bundle = EvidenceBundle(
        "t1",
        2.0,
        (
            EvidenceRef("e1", "text", 2.0, 0.5),
            EvidenceRef("e2", "text", 1.5, 0.5),
        ),
        (),
    )
    scorer = CandidateColumnScorer(2)
    with torch.no_grad():
        scorer.weight.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        scorer.weight.bias.zero_()

    backend = FakeBackend()
    result = Stage2Verifier(
        backend,
        scorer,
        evidence_router=FakeRouter({"e1": 0, "e2": 1}),
        min_row_coverage=1.0,
    ).verify(
        query,
        [bundle],
        {"t1": target},
        {
            "e1": {"asset_id": "e1", "asset_type": "text", "content": "Messi support"},
            "e2": {"asset_id": "e2", "asset_type": "text", "content": "Mbappe support"},
        },
        direct_target_ids=["t1"],
    )

    assert result.direct_candidates[0].target_id == "t1"
    assert result.selection.column_name == "Club"
    assert [row.value for row in result.rows] == ["Barcelona", "PSG"]
    assert result.augmented_query["columns"][-1]["column_name"] == "Club"
    assert result.augmented_query["rows"][0]["cells"][-1]["text"] == "Barcelona"
    assert result.semantic_joinability.joinable
    assert backend.localization_calls == [("Messi", "e1"), ("Mbappe", "e2")]
    assert backend.generation_calls == [("Messi", "e1"), ("Mbappe", "e2")]
    assert result.rows[0].evidence["routing_similarity"] == pytest.approx(0.8)
    assert len(backend.embed_batches) == 2
    assert len(backend.embed_batches[0]) == 8


def test_similarity_router_assigns_each_evidence_to_its_nearest_row():
    store = FeatureStore(
        {
            "q1": ObjectFeatures(
                "q1",
                "table",
                torch.tensor([1.0, 1.0]),
                row_embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            "e1": ObjectFeatures("e1", "text", torch.tensor([0.9, 0.1])),
            "e2": ObjectFeatures("e2", "image", torch.tensor([0.1, 0.9])),
        }
    )

    assignments = SimilarityEvidenceRouter(store).assign("q1", ["e1", "e2"], row_count=2)

    assert [(item.evidence_id, item.row_position) for item in assignments] == [("e1", 0), ("e2", 1)]
    assert all(item.similarity > 0.99 for item in assignments)


def test_stage2_skips_rows_without_assigned_evidence():
    query = _table(
        "q1",
        ["Player", "Country"],
        [["Messi", "Argentina"], ["Mbappe", "France"]],
        query_entity_col=0,
    )
    target = _table("t1", ["Country", "Club"], [["Argentina", "Barcelona"], ["France", "PSG"]])
    bundle = EvidenceBundle(
        "t1",
        2.0,
        (EvidenceRef("e1", "text", 2.0, 0.6), EvidenceRef("e2", "text", 1.0, 0.4)),
        (),
    )
    scorer = CandidateColumnScorer(2)
    with torch.no_grad():
        scorer.weight.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        scorer.weight.bias.zero_()
    backend = FakeBackend()

    result = Stage2Verifier(
        backend,
        scorer,
        evidence_router=FakeRouter({"e1": 0, "e2": 0}),
    ).verify(
        query,
        [bundle],
        {"t1": target},
        {
            "e1": {"asset_id": "e1", "asset_type": "text", "content": "first"},
            "e2": {"asset_id": "e2", "asset_type": "text", "content": "second"},
        },
    )

    assert backend.localization_calls == [("Messi", "e1"), ("Messi", "e2")]
    assert backend.generation_calls == [("Messi", "e1")]
    assert result.rows[1].value == ""
    assert result.rows[1].evidence is None


def test_qwen_reader_places_all_evidence_in_one_forward():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    backend.marker_ids = {"<|object_ref_start|>": 10, "<|object_ref_end|>": 11}
    captured_content = []

    def inputs(content, *, generation_prompt):
        assert not generation_prompt
        captured_content.extend(content)
        return {"input_ids": torch.tensor([[10, 1, 11, 10, 2, 11]])}

    class ReaderModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            del kwargs
            self.calls += 1
            return type("Output", (), {"last_hidden_state": torch.arange(12).reshape(1, 6, 2)})()

    reader_model = ReaderModel()
    backend._inputs = inputs
    backend.model = type("Model", (), {"model": reader_model})()
    query = _table("q", ["Player"], [["Messi"]])
    target = _table("t", ["Country", "Club"], [["Argentina", "Barcelona"]])
    evidence = [
        {"asset_id": "e1", "asset_type": "text", "content": "first"},
        {"asset_id": "e2", "asset_type": "text", "content": "second"},
    ]

    open_states, close_states = backend.reader_states(query, target, evidence)

    assert reader_model.calls == 1
    assert open_states.shape == close_states.shape == (2, 2)
    rendered_text = "\n".join(item.get("text", "") for item in captured_content)
    assert "Evidence 1 (e1)" in rendered_text
    assert "Evidence 2 (e2)" in rendered_text


def test_qwen35_focus_hooks_only_full_attention_layers():
    class FullAttentionLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = torch.nn.Module()
            self.self_attn.v_proj = torch.nn.Linear(2, 2, bias=False)

    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    layers = torch.nn.ModuleList(
        [torch.nn.Module(), FullAttentionLayer(), torch.nn.Module(), FullAttentionLayer()]
    )
    language_model = type("LanguageModel", (), {"layers": layers})()
    backend.model = type("Model", (), {"model": type("Base", (), {"language_model": language_model})()})()
    backend.focus_start_layer = 1

    with backend._capture_values() as captured:
        layers[1].self_attn.v_proj(torch.tensor([[[1.0, 2.0]]]))
        layers[3].self_attn.v_proj(torch.tensor([[[3.0, 4.0]]]))

    assert len(captured) == 2
    assert all(value is not None for value in captured)


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
    model_dir = tmp_path / "qwen35"
    save_candidate_scorer(path, scorer, metadata={"model_dir": str(model_dir)})

    restored = load_candidate_scorer(
        path,
        torch.device("cpu"),
        expected_model_dir=model_dir,
    )

    assert torch.equal(restored.weight.weight, scorer.weight.weight)
    assert torch.equal(restored.weight.bias, scorer.weight.bias)

    with pytest.raises(ValueError, match="retrain it for the selected Stage-2 backbone"):
        load_candidate_scorer(
            path,
            torch.device("cpu"),
            expected_model_dir=tmp_path / "old_qwen",
        )

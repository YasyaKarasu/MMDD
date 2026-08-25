from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage2.checkpoints import load_candidate_scorer, save_candidate_scorer
from mmdd_stage2.data import (
    Stage2ObjectIndex,
    direct_target_ids,
    serialize_image_presence_prompt,
    serialize_localization_prompt,
)
from mmdd_stage2.pipeline import LocalizedEvidence, Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.training import ColumnTrainingExample, train_candidate_scorer
from mmdd_stage2.verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    best_text_span,
    build_evidence_bundles,
    focus_relevance,
    joint_candidate_probabilities,
    joint_relevance,
    propose_image_regions,
    semantic_joinability,
)


def test_build_evidence_bundles_selects_unique_evidence_by_path_score():
    bundles = build_evidence_bundles(
        [
            {
                "target_id": "t1",
                "score": 4.0,
                "direct_score": 3.0,
                "evidence_score": 2.25,
                "paths": [
                    {"kind": "evidence", "evidence_id": "e2", "evidence_type": "image", "path_score": 2.0},
                    {"kind": "direct", "path_score": 3.0},
                    {"kind": "evidence", "evidence_id": "e1", "evidence_type": "text", "path_score": 1.5},
                    {"kind": "evidence", "evidence_id": "e1", "evidence_type": "text", "path_score": 1.5},
                ],
            }
        ],
        top_k_evidence=2,
    )

    assert bundles[0].target_id == "t1"
    assert bundles[0].retrieval_score == pytest.approx(2.25)
    assert bundles[0].evidence_ids == ("e1", "e2")


def test_stage2_preserves_global_rrf_order_while_using_route_scores():
    results = [
        {
            "target_id": "mixed",
            "direct_score": 0.1,
            "evidence_score": 0.9,
            "paths": [
                {"kind": "direct", "path_score": 0.1},
                {"kind": "evidence", "evidence_id": "e1", "path_score": 0.9},
            ],
        },
        {
            "target_id": "direct",
            "direct_score": 0.8,
            "evidence_score": None,
            "paths": [{"kind": "direct", "path_score": 0.8}],
        },
        {
            "target_id": "evidence",
            "direct_score": None,
            "evidence_score": 1.0,
            "paths": [{"kind": "evidence", "evidence_id": "e2", "path_score": 1.0}],
        },
    ]

    global_top_k = results[:2]
    bundles = build_evidence_bundles(global_top_k, top_k_evidence=1)

    assert direct_target_ids(global_top_k) == ["mixed", "direct"]
    assert [bundle.target_id for bundle in bundles] == ["mixed"]
    assert bundles[0].retrieval_score == pytest.approx(0.9)


def test_candidate_column_probabilities_match_table_times_column_formula():
    retrieval_scores = torch.tensor([2.0, 1.0])
    column_logits = torch.tensor([[0.0, 1.0], [2.0, -5.0]])
    column_mask = torch.tensor([[True, True], [True, False]])

    joint = joint_candidate_probabilities(
        retrieval_scores, column_logits, column_mask
    )

    table_probabilities = torch.softmax(retrieval_scores, dim=-1)
    column_probabilities = torch.softmax(
        column_logits.masked_fill(~column_mask, -torch.inf), dim=-1
    ).masked_fill(~column_mask, 0.0)
    assert column_probabilities[1].tolist() == [1.0, 0.0]
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


def test_joint_relevance_localizes_text_span():
    evidence = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    row_anchor = torch.tensor([[1.0, 0.0]])
    attribute = torch.tensor([[0.8, 0.2]])
    relevance = joint_relevance(row_anchor, attribute, evidence)

    assert relevance.sum().item() == pytest.approx(1.0)
    assert best_text_span(relevance, 2) == (0, 2)


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


def test_localization_prompt_uses_the_full_row_as_entity_anchor():
    prompt = serialize_localization_prompt(
        {"Player": "Messi", "Country": "Argentina", "Note": ""},
        "Club",
    )

    assert (
        "Query row (entity anchor): <|object_ref_start|>Player=Messi | Country=Argentina | "
        "Note=<|object_ref_end|>"
    ) in prompt
    assert "Requested attribute: <|box_start|>Club<|box_end|>" in prompt
    assert "All query-row attributes jointly identify the entity" in prompt
    assert "An entity mention alone, an unlinked attribute value, or a value for another entity is not valid" in prompt
    assert "do not fill the attribute from the query row or outside knowledge" in prompt


def test_image_presence_prompt_requires_an_extractable_attribute_value():
    prompt = serialize_image_presence_prompt(
        {"Player": "Messi", "Country": "Argentina"},
        "Club",
    )

    assert "Query row (entity identifier only): Player=Messi | Country=Argentina" in prompt
    assert "Requested attribute: Club" in prompt
    assert "links this same entity to an extractable value" in prompt
    assert "attribute keyword or value not linked to the entity" in prompt
    assert "information outside the crop" in prompt
    assert "Answer exactly yes or no" in prompt


class FakeBackend:
    hidden_dim = 2

    def __init__(self):
        self.embed_batches = []
        self.localization_calls = []
        self.generation_calls = []
        self.reader_evidence_batches = []
        self.evidence_logit_calls = []
        self.localization_score_by_id = {}
        self.evidence_logit_by_id = {}

    def reader_states(self, query, target, evidence):
        self.reader_evidence_batches.append(tuple(item["asset_id"] for item in evidence))
        return torch.tensor([[0.0, 0.0], [4.0, 0.0]]), torch.zeros(2, 2)

    def localize_evidence(self, row, *, attribute_name, evidence):
        self.localization_calls.append((dict(row), evidence["asset_id"]))
        score = self.localization_score_by_id.get(evidence["asset_id"], 0.9)
        if evidence["asset_type"] == "image":
            return LocalizedEvidence(
                evidence["asset_id"],
                "image",
                image_presence_probability=score,
            )
        return LocalizedEvidence(
            evidence["asset_id"],
            "text",
            text=evidence["content"],
            text_span_relevance=score,
        )

    def evidence_logits(self, row, *, attribute_name, candidates):
        self.evidence_logit_calls.append(
            (dict(row), attribute_name, tuple(item.evidence_id for item in candidates))
        )
        return torch.tensor([self.evidence_logit_by_id.get(item.evidence_id, 0.0) for item in candidates])

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
        return {
            evidence_id: self.row_by_evidence[evidence_id]
            for evidence_id in evidence_ids
        }


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
    )
    target = _table(
        "t1",
        ["Country", "Club"],
        [["Argentina", "Barcelona"], ["France", "PSG"]],
    )
    bundle = EvidenceBundle(
        "t1",
        2.0,
        ("e1", "e2"),
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
    assert result.semantic_joinability.joinable
    assert backend.localization_calls == [
        ({"Player": "Messi", "Country": "Argentina"}, "e1"),
        ({"Player": "Mbappe", "Country": "France"}, "e2"),
    ]
    assert backend.evidence_logit_calls == []
    assert backend.generation_calls == [("Messi", "e1"), ("Mbappe", "e2")]
    assert result.rows[0].evidence == {
        "evidence_id": "e1",
        "evidence_type": "text",
        "text_span": "Messi support",
        "text_span_relevance": pytest.approx(0.9),
    }
    assert len(backend.embed_batches) == 2
    assert len(backend.embed_batches[0]) == 8

    payload = result.to_dict()
    assert set(payload) == {"query_id", "direct_matches", "selection", "rows", "verification"}
    assert payload["selection"] == {
        "target_id": "t1",
        "column_index": 1,
        "column_name": "Club",
    }
    assert payload["rows"][0] == {
        "row_id": 0,
        "value": "Barcelona",
        "evidence": result.rows[0].evidence,
    }
    assert "augmented_query" not in payload


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

    assert assignments == {"e1": 0, "e2": 1}


def test_stage2_skips_rows_without_assigned_evidence():
    query = _table(
        "q1",
        ["Player", "Country"],
        [["Messi", "Argentina"], ["Mbappe", "France"]],
    )
    target = _table("t1", ["Country", "Club"], [["Argentina", "Barcelona"], ["France", "PSG"]])
    bundle = EvidenceBundle(
        "t1",
        2.0,
        ("e1", "e2"),
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

    assert backend.localization_calls == [
        ({"Player": "Messi", "Country": "Argentina"}, "e1"),
        ({"Player": "Messi", "Country": "Argentina"}, "e2"),
    ]
    assert backend.evidence_logit_calls == [
        ({"Player": "Messi", "Country": "Argentina"}, "Club", ("e1", "e2"))
    ]
    assert backend.generation_calls == [("Messi", "e1")]
    assert result.rows[1].value == ""
    assert result.rows[1].evidence is None


def test_stage2_uses_joint_logits_instead_of_cross_modal_localization_scores():
    query = _table(
        "q1",
        ["Player", "Country"],
        [["Messi", "Argentina"], ["Mbappe", "France"]],
    )
    target = _table("t1", ["Country", "Club"], [["Argentina", "Barcelona"], ["France", "PSG"]])
    bundle = EvidenceBundle(
        "t1",
        2.0,
        ("text", "image"),
    )
    scorer = CandidateColumnScorer(2)
    with torch.no_grad():
        scorer.weight.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        scorer.weight.bias.zero_()
    backend = FakeBackend()
    backend.localization_score_by_id = {"text": 0.99, "image": 0.01}
    backend.evidence_logit_by_id = {"text": -1.0, "image": 2.0}

    result = Stage2Verifier(
        backend,
        scorer,
        evidence_router=FakeRouter({"text": 0, "image": 0}),
    ).verify(
        query,
        [bundle],
        {"t1": target},
        {
            "text": {"asset_id": "text", "asset_type": "text", "content": "text support"},
            "image": {"asset_id": "image", "asset_type": "image", "content": "image support"},
        },
    )

    assert backend.evidence_logit_calls == [
        ({"Player": "Messi", "Country": "Argentina"}, "Club", ("text", "image"))
    ]
    assert backend.generation_calls == [("Messi", "image")]
    assert result.rows[0].evidence["evidence_id"] == "image"
    assert result.rows[0].evidence["image_presence_probability"] == pytest.approx(0.01)
    assert "text_span_relevance" not in result.rows[0].evidence


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
    query.update(page_title="SECRET_QUERY_PAGE", caption="SECRET_QUERY_CAPTION")
    target.update(section_title="SECRET_TARGET_SECTION")
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
    assert "Each complete query row identifies one entity" in rendered_text
    assert "retrieved evidence must explicitly support linking" in rendered_text
    assert "Do not select a column based only on header similarity" in rendered_text
    assert "BEGIN QUERY TABLE" in rendered_text
    assert "END RETRIEVED EVIDENCE" in rendered_text
    assert "BEGIN CANDIDATE TARGET TABLE" in rendered_text
    assert "SECRET_" not in rendered_text


def test_qwen_evidence_logits_compare_text_and_image_in_one_forward():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    captured_content = []

    class Tokenizer:
        @staticmethod
        def encode(value, *, add_special_tokens):
            assert not add_special_tokens
            return {"A": [3], "B": [7]}[value]

    backend.processor = type("Processor", (), {"tokenizer": Tokenizer()})()

    def inputs(content, *, generation_prompt):
        assert generation_prompt
        captured_content.extend(content)
        return {"input_ids": torch.tensor([[1, 2]])}

    class RerankerModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            assert kwargs["logits_to_keep"] == 1
            self.calls += 1
            logits = torch.zeros(1, 1, 8)
            logits[0, 0, 3] = -1.0
            logits[0, 0, 7] = 2.0
            return type("Output", (), {"logits": logits})()

    backend._inputs = inputs
    backend.model = RerankerModel()
    image = object()

    logits = backend.evidence_logits(
        {"Player": "Messi", "Country": "Argentina"},
        attribute_name="Club",
        candidates=[
            LocalizedEvidence(
                "text",
                "text",
                text="Messi played for Barcelona.",
                text_span_relevance=0.99,
            ),
            LocalizedEvidence(
                "image",
                "image",
                image=image,
                image_presence_probability=0.01,
            ),
        ],
    )

    assert backend.model.calls == 1
    assert logits.tolist() == [-1.0, 2.0]
    assert [item["image"] for item in captured_content if item["type"] == "image"] == [image]
    rendered_text = "\n".join(item.get("text", "") for item in captured_content)
    assert "Candidate A (text, text)" in rendered_text
    assert "Candidate B (image, image)" in rendered_text
    assert "Valid labels: A, B" in rendered_text
    assert '"Player": "Messi"' in rendered_text
    assert '"Country": "Argentina"' in rendered_text
    assert "Entity column" not in rendered_text
    assert "A fully valid candidate must link this same entity" in rendered_text
    assert "modality and evidence IDs are labels, not factual support" in rendered_text
    assert "If no candidate is fully valid" in rendered_text


def test_qwen_generation_prompt_uses_evidence_as_the_only_value_source():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    captured_content = []

    def generate(content):
        captured_content.extend(content)
        return '{"value": "Barcelona"}'

    backend._generate = generate
    value = backend.generate_value(
        {"Player": "Messi", "Country": "Argentina"},
        attribute_name="Club",
        evidence=LocalizedEvidence(
            "text",
            "text",
            text="Messi plays for Barcelona.",
            text_span_relevance=0.9,
        ),
    )

    assert value == "Barcelona"
    rendered_text = "\n".join(item.get("text", "") for item in captured_content)
    assert 'Query row (entity identifier only): {"Player": "Messi", "Country": "Argentina"}' in rendered_text
    assert "never copy or derive the output from the query row itself" in rendered_text
    assert "localized evidence supplied with this prompt is the only source" in rendered_text
    assert "belongs to another entity" in rendered_text
    assert "cannot be resolved to one unambiguous cell value" in rendered_text
    assert 'Return exactly one JSON object with no Markdown or explanation: {"value": "..."}' in rendered_text
    assert "Localized evidence: Messi plays for Barcelona." in rendered_text


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
    query = _table("q1", ["Player"], [["Messi"]])
    target = _table("t1", ["Country", "Club"], [["Spain", "Barcelona"]])
    bundle = EvidenceBundle("t1", 2.0, ("e1",))
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

    assert set(history[0]) == {"epoch", "column_loss", "table_loss"}
    assert not torch.equal(before, scorer.weight.weight)


def test_candidate_head_training_accepts_inference_mode_reader_states():
    class InferenceBackend(FakeBackend):
        @torch.inference_mode()
        def reader_states(self, query, target, evidence):
            return super().reader_states(query, target, evidence)

    query = _table("q1", ["Player"], [["Messi"]])
    target = _table("t1", ["Country", "Club"], [["Spain", "Barcelona"]])
    bundle = EvidenceBundle("t1", 2.0, ("e1",))
    example = ColumnTrainingExample("q1", (bundle,), "t1", 1)
    objects = Stage2ObjectIndex(
        {"q1": query},
        {"t1": target},
        {"e1": {"asset_id": "e1", "asset_type": "text", "content": "support"}},
    )
    backend = InferenceBackend()
    open_states, close_states = backend.reader_states(query, target, [objects.evidence["e1"]])
    assert torch.is_inference(open_states)
    assert torch.is_inference(close_states)
    scorer = CandidateColumnScorer(2)
    before = scorer.weight.weight.detach().clone()

    train_candidate_scorer(
        backend,
        scorer,
        [example],
        objects,
        epochs=1,
        learning_rate=0.1,
        weight_decay=0.0,
        seed=1,
    )

    assert not torch.equal(before, scorer.weight.weight)


def test_candidate_head_training_reads_only_the_positive_target():
    query = _table("q1", ["Player"], [["Messi"]])
    negative = _table("t0", ["Country", "City"], [["Spain", "Madrid"]])
    positive = _table("t1", ["Country", "Club"], [["Spain", "Barcelona"]])
    bundles = (
        EvidenceBundle("t0", 2.0, ("negative_evidence",)),
        EvidenceBundle("t1", 1.0, ("positive_evidence",)),
    )
    example = ColumnTrainingExample("q1", bundles, "t1", 1)
    objects = Stage2ObjectIndex(
        {"q1": query},
        {"t0": negative, "t1": positive},
        {
            "negative_evidence": {
                "asset_id": "negative_evidence",
                "asset_type": "text",
                "content": "negative support",
            },
            "positive_evidence": {
                "asset_id": "positive_evidence",
                "asset_type": "text",
                "content": "positive support",
            },
        },
    )
    backend = FakeBackend()

    history = train_candidate_scorer(
        backend,
        CandidateColumnScorer(2),
        [example],
        objects,
        epochs=1,
        learning_rate=0.1,
        weight_decay=0.0,
        seed=1,
    )

    assert backend.reader_evidence_batches == [("positive_evidence",)]
    expected_table_loss = -torch.log_softmax(torch.tensor([2.0, 1.0]), 0)[1].item()
    assert history[0]["table_loss"] == pytest.approx(expected_table_loss)


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

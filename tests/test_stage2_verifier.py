from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mmdd_stage2.qwen as stage2_qwen
import train_stage2 as stage2_train
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage2.checkpoints import load_candidate_scorer, save_candidate_scorer
from mmdd_stage2.data import (
    ATTRIBUTE_CLOSE,
    ATTRIBUTE_OPEN,
    EVIDENCE_CLOSE,
    EVIDENCE_OPEN,
    ROW_ANCHOR_CLOSE,
    ROW_ANCHOR_OPEN,
    Stage2ObjectIndex,
    direct_target_ids,
    serialize_image_presence_prompt,
    serialize_localization_prompt,
    validate_retrieval_path_budget,
)
from mmdd_stage2.pipeline import LocalizedEvidence, Stage2Verifier
from mmdd_stage2.qwen import QwenStage2Backend
from mmdd_stage2.routing import SimilarityEvidenceRouter
from mmdd_stage2.training import (
    ColumnTrainingExample,
    load_column_training_data,
    train_candidate_scorer,
)
from mmdd_stage2.verifier import (
    CandidateColumnScorer,
    EvidenceBundle,
    best_text_span,
    build_evidence_bundles,
    focus_relevance,
    focus_relevance_from_logits,
    joint_candidate_probabilities,
    joint_relevance_logits,
    propose_image_regions,
    semantic_joinability,
)


def test_build_evidence_bundles_preserves_compact_path_order_and_limit():
    bundles = build_evidence_bundles(
        [
            {
                "target_id": "t1",
                "score": 4.0,
                "direct_score": 3.0,
                "evidence_score": 2.25,
                "paths": [
                    {"kind": "direct", "path_score": 3.0},
                    {"kind": "evidence", "evidence_id": "e2"},
                    {"kind": "evidence", "evidence_id": "e1"},
                    {"kind": "evidence", "evidence_id": "e3"},
                ],
            }
        ],
        top_k_evidence=2,
    )

    assert bundles[0].target_id == "t1"
    assert bundles[0].retrieval_score == pytest.approx(2.25)
    assert bundles[0].evidence_ids == ("e2", "e1")


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


def test_stage2_uses_explicit_fused_table_score_when_exported():
    bundles = build_evidence_bundles(
        [
            {
                "target_id": "f4-target",
                "score": 0.7,
                "stage2_table_score": 0.7,
                "evidence_score": 0.2,
                "paths": [
                    {
                        "kind": "evidence",
                        "evidence_id": "e1",
                        "path_score": 0.2,
                    }
                ],
            }
        ],
        top_k_evidence=1,
    )

    assert bundles[0].retrieval_score == pytest.approx(0.7)


def test_stage2_rejects_settings_above_compact_retrieval_path_budget():
    record = {
        "path_aggregation": {"path_result_k": 1, "evidence_path_k": 1},
        "results": [
            {
                "target_id": "evidence",
                "evidence_score": 1.0,
                "paths": [
                    {"kind": "evidence", "evidence_id": "e1", "path_score": 1.0}
                ],
            }
        ],
    }

    validate_retrieval_path_budget(record, max_targets=10, top_k_evidence=1)
    record["results"].append({"target_id": "without_paths"})
    with pytest.raises(ValueError, match="larger --path-result-k"):
        validate_retrieval_path_budget(record, max_targets=2, top_k_evidence=1)
    with pytest.raises(ValueError, match="larger --evidence-path-k"):
        validate_retrieval_path_budget(record, max_targets=1, top_k_evidence=2)

    record["path_aggregation"]["evidence_path_k"] = 0
    record["results"] = [{"target_id": "pruned", "evidence_score": 1.0, "paths": []}]
    with pytest.raises(ValueError, match="larger --evidence-path-k"):
        validate_retrieval_path_budget(record, max_targets=1, top_k_evidence=1)

    record["results"] = [{"target_id": "direct", "paths": [{"kind": "direct"}]}]
    validate_retrieval_path_budget(record, max_targets=1, top_k_evidence=1)


def test_stage2_training_skips_path_budget_validation_without_a_matching_qrel(monkeypatch):
    qrel = {
        "query_table_id": "train",
        "target_table_id": "positive",
        "reason": "model_recoverable_join_column",
        "join_attribute": {"source_column_index": 1},
    }
    records = [
        {
            "query_id": "irrelevant",
            "path_aggregation": {"path_result_k": 0, "evidence_path_k": 0},
            "results": [
                {
                    "target_id": "ignored",
                    "evidence_score": 1.0,
                    "paths": [
                        {"kind": "evidence", "evidence_id": "ignored", "path_score": 1.0}
                    ],
                }
            ],
        },
        {
            "query_id": "train",
            "path_aggregation": {"path_result_k": 2, "evidence_path_k": 1},
            "results": [
                {
                    "target_id": "negative",
                    "evidence_score": 2.0,
                    "paths": [
                        {
                            "kind": "evidence",
                            "evidence_id": "negative_evidence",
                            "path_score": 2.0,
                        }
                    ],
                },
                {
                    "target_id": "positive",
                    "evidence_score": 1.0,
                    "paths": [
                        {"kind": "evidence", "evidence_id": "e1", "path_score": 1.0}
                    ],
                }
            ],
        },
    ]
    loaded_index = Stage2ObjectIndex({}, {}, {})
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_dataset_artifact",
        lambda _output_dir, artifact: iter([qrel]) if artifact == "qrels" else iter(()),
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_retrieval_results", lambda _path: iter(records)
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.load_stage2_index", lambda *_args, **_kwargs: loaded_index
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.load_stage2_evidence", lambda *_args, **_kwargs: {}
    )

    examples, objects = load_column_training_data(
        Path("dataset"),
        [Path("retrieval.jsonl")],
        max_targets=2,
        top_k_evidence=1,
    )

    assert [example.query_id for example in examples] == ["train"]
    assert examples[0].positive_bundle.target_id == "positive"
    assert examples[0].table_loss == pytest.approx(
        -torch.log_softmax(torch.tensor([2.0, 1.0]), 0)[1].item()
    )
    assert objects is loaded_index


def test_stage2_training_keeps_all_qrels_for_a_multi_positive_query(monkeypatch):
    qrels = [
        {
            "query_table_id": "q",
            "target_table_id": target_id,
            "reason": "model_recoverable_join_column",
            "join_attribute": {"source_column_index": source_column},
        }
        for target_id, source_column in (("positive_1", 3), ("positive_2", 5))
    ]
    retrieval = {
        "query_id": "q",
        "results": [
            {
                "target_id": target_id,
                "evidence_score": score,
                "paths": [
                    {
                        "kind": "evidence",
                        "evidence_id": evidence_id,
                        "path_score": score,
                    }
                ],
            }
            for target_id, evidence_id, score in (
                ("positive_1", "e1", 2.0),
                ("positive_2", "e2", 1.0),
                ("negative", "e3", 0.0),
            )
        ],
    }
    loaded_index = Stage2ObjectIndex(
        {"q": {"table_id": "q"}},
        {
            "positive_1": {"table_id": "positive_1"},
            "positive_2": {"table_id": "positive_2"},
        },
        {},
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_dataset_artifact",
        lambda _root, artifact: iter(qrels) if artifact == "qrels" else iter(()),
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_retrieval_results", lambda _path: iter([retrieval])
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.load_stage2_index",
        lambda *_args, **_kwargs: loaded_index,
    )
    monkeypatch.setattr(
        "mmdd_stage2.training.load_stage2_evidence",
        lambda _roots, evidence_ids: {
            evidence_id: {"asset_id": evidence_id} for evidence_id in evidence_ids
        },
    )

    examples, objects = load_column_training_data(
        Path("dataset"),
        [Path("retrieval.jsonl")],
        max_targets=3,
        top_k_evidence=1,
    )

    assert [example.positive_bundle.target_id for example in examples] == [
        "positive_1",
        "positive_2",
    ]
    assert [example.positive_source_column for example in examples] == [3, 5]
    expected = torch.logsumexp(torch.tensor([2.0, 1.0, 0.0]), dim=0) - torch.logsumexp(
        torch.tensor([2.0, 1.0]), dim=0
    )
    assert all(example.table_loss == pytest.approx(expected.item()) for example in examples)
    assert set(objects.evidence) == {"e1", "e2"}


def test_stage2_training_pairs_multiple_dataset_roots_with_retrieval_files(monkeypatch):
    def qrels(root, artifact):
        assert artifact == "qrels"
        suffix = Path(root).name
        return iter(
            [
                {
                    "query_table_id": f"q_{suffix}",
                    "target_table_id": f"t_{suffix}",
                    "reason": "model_recoverable_join_column",
                    "split": "train",
                    "join_attribute": {"source_column_index": 1},
                }
            ]
        )

    def retrieval(path):
        suffix = Path(path).stem
        return iter(
            [
                {
                    "query_id": f"q_{suffix}",
                    "results": [
                        {
                            "target_id": f"t_{suffix}",
                            "evidence_score": 1.0,
                            "paths": [
                                {
                                    "kind": "evidence",
                                    "evidence_id": f"e_{suffix}",
                                    "path_score": 1.0,
                                }
                            ],
                        }
                    ],
                }
            ]
        )

    def stage2_index(root, **_kwargs):
        suffix = Path(root).name
        return Stage2ObjectIndex(
            {f"q_{suffix}": {"table_id": f"q_{suffix}"}},
            {f"t_{suffix}": {"table_id": f"t_{suffix}"}},
            {},
        )

    monkeypatch.setattr("mmdd_stage2.training.iter_dataset_artifact", qrels)
    monkeypatch.setattr("mmdd_stage2.training.iter_retrieval_results", retrieval)
    monkeypatch.setattr("mmdd_stage2.training.load_stage2_index", stage2_index)
    monkeypatch.setattr(
        "mmdd_stage2.training.load_stage2_evidence",
        lambda roots, evidence_ids: {
            evidence_id: {"asset_id": evidence_id, "roots": [Path(root).name for root in roots]}
            for evidence_id in evidence_ids
        },
    )

    examples, objects = load_column_training_data(
        [Path("entitables"), Path("wdc")],
        [Path("entitables.jsonl"), Path("wdc.jsonl")],
        max_targets=1,
        top_k_evidence=1,
    )

    assert [example.query_id for example in examples] == ["q_entitables", "q_wdc"]
    assert set(objects.queries) == {"q_entitables", "q_wdc"}
    assert set(objects.targets) == {"t_entitables", "t_wdc"}
    assert set(objects.evidence) == {"e_entitables", "e_wdc"}
    assert objects.evidence["e_wdc"]["roots"] == ["entitables", "wdc"]


def test_stage2_training_rejects_unpaired_dataset_roots(monkeypatch):
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_dataset_artifact",
        lambda *_args, **_kwargs: pytest.fail("root validation must precede loading"),
    )

    with pytest.raises(ValueError, match="one root per file"):
        load_column_training_data(
            [Path("a"), Path("b")],
            [Path("a.jsonl"), Path("b.jsonl"), Path("c.jsonl")],
            max_targets=1,
            top_k_evidence=1,
        )


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
    relevance = torch.softmax(
        joint_relevance_logits(row_anchor, attribute, evidence), dim=0
    )

    assert relevance.sum().item() == pytest.approx(1.0)
    assert best_text_span(relevance, 2) == (0, 2)


def test_joint_relevance_logits_preserve_the_existing_normalized_map():
    evidence = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    row_anchor = torch.tensor([[1.0, 0.0]])
    attribute = torch.tensor([[0.8, 0.2]])

    logits = joint_relevance_logits(row_anchor, attribute, evidence)
    relevance = torch.softmax(logits, dim=0)
    normalized_evidence = torch.nn.functional.normalize(evidence, dim=-1)
    row_map = torch.softmax(normalized_evidence @ row_anchor[0], dim=0)
    attribute_map = torch.softmax(
        normalized_evidence
        @ torch.nn.functional.normalize(attribute.mean(dim=0), dim=0),
        dim=0,
    )
    previous_relevance = row_map * attribute_map
    previous_relevance /= previous_relevance.sum()

    assert torch.allclose(relevance, previous_relevance)
    assert torch.allclose(
        focus_relevance_from_logits([logits, logits]),
        relevance,
    )


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
        similarity_batch_size=1,
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
    assert backend.embed_batches == [
        (
            "Messi",
            "Mbappe",
            "Argentina",
            "France",
            "Argentina",
            "France",
            "Barcelona",
            "PSG",
        ),
        (
            "Barcelona",
            "PSG",
            "Barcelona",
            "PSG",
        ),
    ]

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

    class TrackingStore:
        def get(self, object_id, *, include_hidden):
            assert include_hidden is False
            return store.get(object_id, include_hidden=include_hidden)

    assignments = SimilarityEvidenceRouter(TrackingStore()).assign(
        "q1", ["e1", "e2"], row_count=2
    )

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


def test_qwen_image_input_uses_first_rgb_frame_for_animated_gif(tmp_path):
    path = tmp_path / "animated.gif"
    first = Image.new("RGB", (4, 3), (255, 0, 0))
    second = Image.new("RGB", (4, 3), (0, 255, 0))
    first.save(path, save_all=True, append_images=[second], duration=10, loop=0)

    image = QwenStage2Backend._image_input(
        {"asset_id": "animated", "asset_type": "image", "local_path": str(path)}
    )

    assert image.mode == "RGB"
    assert image.size == (4, 3)
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_qwen_image_input_can_cap_pixels(tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (200, 100), (1, 2, 3)).save(path)

    image = QwenStage2Backend._image_input(
        {"asset_id": "large", "asset_type": "image", "local_path": str(path)},
        max_pixels=5_000,
    )

    assert image.size == (100, 50)
    assert image.width * image.height <= 5_000


def test_qwen_reader_escapes_marker_literals_in_evidence():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    backend.marker_ids = {"<|object_ref_start|>": 10, "<|object_ref_end|>": 11}
    captured_content = []

    def inputs(content, *, generation_prompt):
        assert not generation_prompt
        captured_content.extend(content)
        return {"input_ids": torch.tensor([[10, 1, 11]])}

    class ReaderModel:
        def __call__(self, **kwargs):
            del kwargs
            return type("Output", (), {"last_hidden_state": torch.zeros(1, 3, 2)})()

    backend._inputs = inputs
    backend.model = type("Model", (), {"model": ReaderModel()})()

    backend.reader_states(
        _table("q", ["Player"], [["Messi"]]),
        _table("t", ["Club"], [["Barcelona"]]),
        [
            {
                "asset_id": "e1",
                "asset_type": "text",
                "content": "literal <|object_ref_start|> marker <|object_ref_end|>",
            }
        ],
    )

    rendered_text = "\n".join(item.get("text", "") for item in captured_content)
    evidence_text = rendered_text.split("BEGIN RETRIEVED EVIDENCE", 1)[1].split(
        "END RETRIEVED EVIDENCE", 1
    )[0]
    assert "<|object_ref_start|>" not in evidence_text
    assert "<|object_ref_end|>" not in evidence_text
    assert "&lt;|object_ref_start|>" in evidence_text
    assert "&lt;|object_ref_end|>" in evidence_text


def test_qwen_marker_range_uses_closing_marker_for_an_empty_field():
    indices = QwenStage2Backend._marker_range(torch.tensor([10, 11]), 10, 11)

    assert torch.equal(indices, torch.tensor([1]))


def test_qwen_text_localization_normalizes_after_merging_windows(monkeypatch):
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    backend.max_text_evidence_tokens = 4
    backend.text_overlap_tokens = 1
    backend.max_span_tokens = 1
    backend.marker_ids = {
        ROW_ANCHOR_OPEN: 100,
        ROW_ANCHOR_CLOSE: 101,
        ATTRIBUTE_OPEN: 102,
        ATTRIBUTE_CLOSE: 103,
        EVIDENCE_OPEN: 104,
        EVIDENCE_CLOSE: 105,
    }

    class Tokenizer:
        @staticmethod
        def encode(value, *, add_special_tokens):
            assert not add_special_tokens
            assert value == "0 1 2 3 4"
            return [10, 11, 12, 13, 14]

        @staticmethod
        def decode(token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return " ".join(str(int(token_id) - 10) for token_id in token_ids)

    backend.processor = type("Processor", (), {"tokenizer": Tokenizer()})()
    windows = iter(
        [
            ([10, 11, 12, 13], [0.5, 0.0, 0.0, 1.0]),
            ([13, 14], [-1.0, 0.0]),
        ]
    )
    seen_windows = []

    def value_forward(_content):
        chunk_ids, logits = next(windows)
        seen_windows.append(chunk_ids)
        input_ids = torch.tensor(
            [100, 20, 101, 102, 21, 103, 104, *chunk_ids, 105]
        )
        layer = torch.zeros((input_ids.numel(), 1))
        layer[7 : 7 + len(chunk_ids), 0] = torch.tensor(logits)
        return [layer], input_ids, {}

    backend._value_forward = value_forward
    monkeypatch.setattr(
        stage2_qwen,
        "joint_relevance_logits",
        lambda _row, _attribute, evidence: evidence[:, 0],
    )

    localized = backend._localize_text(
        {"Player": "Messi"},
        "Club",
        {"asset_id": "e1", "content": "0 1 2 3 4"},
    )

    expected = torch.softmax(torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0]), dim=0)[0]
    assert seen_windows == [[10, 11, 12, 13], [13, 14]]
    assert localized.text == "0"
    assert localized.text_span_relevance == pytest.approx(float(expected))


def test_qwen_embeddings_are_batched_and_truncated():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    backend.device = torch.device("cpu")
    backend.hidden_dim = 2
    backend.embedding_batch_size = 2
    backend.max_embedding_tokens = 7

    class Processor:
        def __init__(self):
            self.calls = []

        def __call__(
            self,
            *,
            text,
            padding,
            truncation,
            max_length,
            return_tensors,
        ):
            self.calls.append(tuple(text))
            assert padding
            assert truncation
            assert max_length == 7
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[int(value)] for value in text]),
                "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
            }

    class EmbeddingModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, input_ids, attention_mask, use_cache, return_dict):
            assert attention_mask.shape == input_ids.shape
            assert not use_cache
            assert return_dict
            self.calls += 1
            hidden = torch.stack(
                (input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32)),
                dim=-1,
            )
            return type("Output", (), {"last_hidden_state": hidden})()

    processor = Processor()
    model = EmbeddingModel()
    backend.processor = processor
    backend.model = type("Model", (), {"model": model})()

    embeddings = backend.embed_texts(["1", "2", "3", "4", "5"])

    assert processor.calls == [("1", "2"), ("3", "4"), ("5",)]
    assert model.calls == 3
    assert embeddings.shape == (5, 2)
    assert torch.allclose(embeddings.norm(dim=-1), torch.ones(5))


def test_qwen_embeddings_leave_blank_values_zero():
    backend = QwenStage2Backend.__new__(QwenStage2Backend)
    backend.device = torch.device("cpu")
    backend.hidden_dim = 2
    backend.embedding_batch_size = 4
    backend.max_embedding_tokens = 7

    class Processor:
        def __call__(
            self,
            *,
            text,
            padding,
            truncation,
            max_length,
            return_tensors,
        ):
            assert text == ["left", "right"]
            assert padding and truncation
            assert max_length == 7
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor([[1, 2, 0], [0, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 0], [0, 1, 1]]),
            }

    class EmbeddingModel:
        def __call__(self, *, input_ids, attention_mask, use_cache, return_dict):
            assert not use_cache and return_dict
            hidden = torch.stack(
                (input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32)),
                dim=-1,
            )
            return type("Output", (), {"last_hidden_state": hidden})()

    backend.processor = Processor()
    backend.model = type("Model", (), {"model": EmbeddingModel()})()

    embeddings = backend.embed_texts(["", "left", "   ", "right"])

    assert torch.equal(embeddings[0], torch.zeros(2))
    assert torch.equal(embeddings[2], torch.zeros(2))
    assert torch.allclose(
        embeddings[1], torch.nn.functional.normalize(torch.tensor([2.0, 1.0]), dim=0)
    )
    assert torch.allclose(
        embeddings[3], torch.nn.functional.normalize(torch.tensor([4.0, 1.0]), dim=0)
    )


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


def _train_stage2_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "dataset_root": "dataset",
        "retrieval_results": ["retrieval.jsonl"],
        "stage1_gate": str(tmp_path / "stage1.selection.json"),
        "output": str(tmp_path / "stage2.pt"),
        "model_dir": "qwen",
        "device": "cpu",
        "dtype": "fp32",
        "top_k_evidence": 1,
        "max_targets": 1,
        "epochs": 1,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "seed": 17,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_train_stage2_rejects_invalid_epochs_before_other_work(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Stage-2 validation must run before seeding or loading data")

    monkeypatch.setattr(stage2_train.torch, "manual_seed", unexpected)
    monkeypatch.setattr(stage2_train, "load_column_training_data", unexpected)
    monkeypatch.setattr(stage2_train, "QwenStage2Backend", unexpected)

    with pytest.raises(ValueError, match="--epochs"):
        stage2_train.run(_train_stage2_args(tmp_path, epochs=0))


def test_train_stage2_requires_stage1_gate_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage2_train,
        "load_column_training_data",
        lambda *_args, **_kwargs: pytest.fail("data loaded before Stage-1 gate"),
    )

    with pytest.raises(ValueError, match="--stage1-gate is required"):
        stage2_train.run(_train_stage2_args(tmp_path, stage1_gate=None))


@pytest.mark.parametrize(
    ("max_targets", "top_k_evidence"),
    [(-1, 1), (0, 1), (1, -1), (1, 0)],
)
def test_stage2_training_loader_rejects_invalid_limits(
    monkeypatch, max_targets, top_k_evidence
):
    monkeypatch.setattr(
        "mmdd_stage2.training.iter_dataset_artifact",
        lambda *_args, **_kwargs: pytest.fail("validation must precede data loading"),
    )

    with pytest.raises(ValueError, match="target and evidence limits"):
        load_column_training_data(
            Path("unused"),
            [],
            max_targets=max_targets,
            top_k_evidence=top_k_evidence,
        )


def test_train_stage2_loads_data_before_seeded_model_initialization(tmp_path, monkeypatch):
    events = []
    scorer_weights = []
    cpu_seeds = []
    cuda_seeds = []

    def manual_seed(seed):
        cpu_seeds.append(seed)
        return torch.random.default_generator.manual_seed(seed)

    def load_data(*_args, **_kwargs):
        events.append("data")
        return [object()], object()

    class Backend:
        hidden_dim = 2
        device = torch.device("cpu")

        def __init__(self, *_args, **_kwargs):
            events.append("backend")

    def build_scorer(hidden_dim):
        events.append("scorer")
        scorer = CandidateColumnScorer(hidden_dim)
        scorer_weights.append(scorer.weight.weight.detach().clone())
        return scorer

    def train(*_args, **_kwargs):
        events.append("train")
        return []

    def save(*_args, **_kwargs):
        events.append("save")

    monkeypatch.setattr(stage2_train.torch, "manual_seed", manual_seed)
    monkeypatch.setattr(stage2_train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stage2_train.torch.cuda, "manual_seed_all", cuda_seeds.append)
    monkeypatch.setattr(stage2_train, "load_column_training_data", load_data)
    monkeypatch.setattr(stage2_train, "validate_stage2_gate", lambda *_args: None)
    monkeypatch.setattr(stage2_train, "QwenStage2Backend", Backend)
    monkeypatch.setattr(stage2_train, "CandidateColumnScorer", build_scorer)
    monkeypatch.setattr(stage2_train, "train_candidate_scorer", train)
    monkeypatch.setattr(stage2_train, "save_candidate_scorer", save)

    args = _train_stage2_args(tmp_path)
    stage2_train.run(args)
    args.output = str(tmp_path / "stage2_second.pt")
    stage2_train.run(args)

    assert events == ["data", "backend", "scorer", "train", "save"] * 2
    assert cpu_seeds == [17, 17]
    assert cuda_seeds == [17, 17]
    assert torch.equal(scorer_weights[0], scorer_weights[1])


def test_candidate_head_training_updates_only_the_small_rata_scorer():
    query = _table("q1", ["Player"], [["Messi"]])
    target = _table("t1", ["Country", "Club"], [["Spain", "Barcelona"]])
    bundle = EvidenceBundle("t1", 2.0, ("e1",))
    example = ColumnTrainingExample("q1", bundle, 1, 0.0)
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
    example = ColumnTrainingExample("q1", bundle, 1, 0.0)
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
    expected_table_loss = -torch.log_softmax(torch.tensor([2.0, 1.0]), 0)[1].item()
    example = ColumnTrainingExample("q1", bundles[1], 1, expected_table_loss)
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

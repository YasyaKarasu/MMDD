from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cache_stage1_features as stage1_cache
import compact_stage1_feature_cache as compact_cache
import mmdd_stage1.features as stage1_features
import mmdd_stage1.training as stage1_training
import refresh_stage1_hard_negatives as hard_negative_refresh
import train_stage1
from cache_stage1_features import (
    EMBEDDING_INSTRUCTIONS,
    build_object_features,
    embedding_instructions,
    teacher_object_ids,
)
from mmdd_stage1.checkpoints import (
    load_path_aggregator,
    load_student,
    load_teacher,
)
from mmdd_stage1.data import (
    EdgeExample,
    TargetCandidate,
    TargetExample,
    load_edge_examples,
    load_target_examples,
)
from mmdd_stage1.features import FeatureStore, ObjectFeatures, normalize_object_type
from mmdd_stage1.mining import (
    HardPath,
    build_hard_candidate_set,
    hard_candidate_records,
    retrieve_hard_candidate_sets,
    score_hard_candidate_sets,
    summarize_hard_candidate_sets,
)
from mmdd_stage1.models import (
    TYPE_TO_ID,
    IdentityStudentJoinabilityModel,
    ProjectedIdentityStudentJoinabilityModel,
    StudentJoinabilityModel,
    TeacherJoinabilityModel,
    structural_table_pool,
)
from mmdd_stage1.objectives import (
    PathAggregator,
    listwise_cross_entropy,
    optional_listwise_cross_entropy,
    positive_indices_to_mask,
    relation_macro_binary_cross_entropy_with_logits,
)
from mmdd_stage1.pca import compute_pca_projection, load_pca_projection
from mmdd_stage1.retrieval import (
    RawEmbeddingANNIndices,
    StudentANNIndices,
    build_indices,
    build_raw_embedding_indices,
    checkpoint_fingerprint,
    retrieve_zero_one_hop,
    retrieve_zero_one_hop_detailed,
)
from mmdd_stage1.scoring import (
    ListScores,
    score_edge_batch,
    score_edge_batch_in_batch,
    score_target_batch,
    score_target_direct_batch_in_batch,
)
from mmdd_stage1.teacher_logits import (
    FROZEN_COSINE_TARGET_SHA256,
    has_teacher_logits,
    load_teacher_logits,
    score_and_cache_cosine_logits,
    score_and_cache_teacher_logits,
)
from mmdd_stage1.teacher_rerank import z_scores
from mmdd_stage1.training import (
    checkpoint,
    sample_balanced_epoch,
    train_student_edges,
    train_student_paths,
    train_teacher_edges,
    train_teacher_paths,
)


def feature(object_id: str, object_type: str, value: float) -> ObjectFeatures:
    embedding = torch.tensor([value, value + 0.2, 1.0 - value, -value])
    hidden_states = torch.stack(
        [embedding, embedding + 0.1, embedding - 0.2, embedding + 0.3]
    )
    groups = torch.tensor([0, 0, 1, 1]) if object_type == "table" else None
    return ObjectFeatures(object_id, object_type, embedding, hidden_states, groups)


def feature_store() -> FeatureStore:
    return FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.25),
            "negative": feature("negative", "table", 0.9),
            "evidence": feature("evidence", "text", 0.3),
        }
    )


class _BatchedSearchMixin:
    def search_many(self, source_ids, destination_type, k):
        return [self.search(source_id, destination_type, k) for source_id in source_ids]


@pytest.mark.parametrize(
    ("legacy_type", "object_type"),
    [
        ("table_fragment", "table"),
        ("text_asset", "text"),
        ("image_asset", "image"),
    ],
)
def test_object_features_normalizes_historical_type_aliases(
    legacy_type, object_type
):
    assert ObjectFeatures("legacy", legacy_type, torch.ones(4)).object_type == object_type


def test_consolidated_store_loads_prepooled_legacy_table_without_groups(tmp_path):
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cache = tmp_path / "features.pt"
    torch.save(
        {
            "objects": {
                "legacy": {
                    "object_type": "table_fragment",
                    "embedding": torch.ones(4),
                    "hidden_states": hidden_states,
                }
            }
        },
        cache,
    )

    features = FeatureStore.from_path(cache).get("legacy")

    assert features.object_type == "table"
    assert torch.equal(features.hidden_states, hidden_states)
    assert features.token_groups is None
    assert teacher().compress(features).shape == (3, 8)


def teacher() -> TeacherJoinabilityModel:
    return TeacherJoinabilityModel(
        input_dim=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
    )


def test_teacher_confidence_transform_checkpoint_round_trip(tmp_path):
    model = TeacherJoinabilityModel(
        input_dim=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
        confidence_transform=True,
    )
    key = model.relation_key("table", "text")
    with torch.no_grad():
        model.confidence_alphas[key].fill_(0.7)
        model.confidence_biases[key].fill_(-0.3)
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint(model, "teacher-edge"), path)

    loaded = load_teacher(path, torch.device("cpu"))

    assert loaded.confidence_transform
    assert loaded.config()["confidence_transform"] is True
    torch.testing.assert_close(
        loaded.transform_edge_scores(
            torch.tensor([-1.0, 2.0]), "table", "text", "confidence_logit"
        ),
        model.transform_edge_scores(
            torch.tensor([-1.0, 2.0]), "table", "text", "confidence_logit"
        ),
    )


def test_legacy_teacher_checkpoint_loads_without_confidence_parameters(tmp_path):
    model = teacher()
    payload = checkpoint(model, "teacher-edge")
    payload["config"].pop("confidence_transform")
    payload["config"].pop("confidence_epsilon")
    path = tmp_path / "legacy_teacher.pt"
    torch.save(payload, path)

    loaded = load_teacher(path, torch.device("cpu"))

    assert not loaded.confidence_transform
    assert loaded.transform_edge_scores(
        torch.tensor([0.0]), "table", "text", "raw_logit"
    ).item() == 0.0


def test_structural_table_pool_returns_schema_and_row_tokens():
    hidden = torch.tensor([[1.0, 1.0], [3.0, 3.0], [6.0, 4.0]])
    groups = torch.tensor([0, 0, 1])

    pooled = structural_table_pool(hidden, groups)

    assert torch.equal(pooled, torch.tensor([[2.0, 2.0], [6.0, 4.0]]))


def test_structural_table_pool_keeps_contiguous_segments_per_group():
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    pooled = structural_table_pool(hidden, groups, tokens_per_group=2)

    expected = torch.stack(
        [hidden[0:2].mean(0), hidden[2:4].mean(0), hidden[4:6].mean(0), hidden[6:8].mean(0)]
    )
    torch.testing.assert_close(pooled, expected)


def test_teacher_retains_multiple_schema_and_row_tokens():
    model = TeacherJoinabilityModel(
        input_dim=2,
        model_dim=4,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        table_tokens_per_group=2,
        dropout=0.0,
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    table = ObjectFeatures(
        "table",
        "table",
        torch.ones(2),
        hidden,
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
    )

    tokens = model.compress(table)

    assert tokens.shape == (4, 4)


def test_teacher_pools_table_groups_before_the_equivalent_adapter_projection():
    model = teacher()
    table = feature("table", "table", 0.2)
    adapter_input_lengths = []
    hook = model.adapters["table"].register_forward_pre_hook(
        lambda _module, inputs: adapter_input_lengths.append(inputs[0].shape[0])
    )

    tokens = model.compress(table)
    hook.remove()

    assert adapter_input_lengths == [2]
    pooled = structural_table_pool(table.hidden_states, table.token_groups)
    token_kinds = torch.tensor([0, 1])
    token_kind_embeddings = model.table_token_embeddings(token_kinds)
    projected_after_pooling = model.adapters["table"](pooled)
    projected_before_pooling = structural_table_pool(
        model.adapters["table"](table.hidden_states), table.token_groups
    )
    torch.testing.assert_close(
        projected_after_pooling,
        projected_before_pooling,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(tokens, projected_after_pooling + token_kind_embeddings)


def test_teacher_uses_ordered_type_pair_and_supports_gradients():
    torch.manual_seed(3)
    model = teacher()
    model.eval()
    table = feature("table", "table", 0.2)
    text = feature("text", "text", 0.4)

    forward_score = model.score_pairs([table], [text])
    reverse_score = model.score_pairs([text], [table])
    loss = forward_score.sum()
    loss.backward()

    assert forward_score.shape == (1,)
    assert not torch.allclose(forward_score, reverse_score)
    assert model.type_pair_embeddings.weight.grad is not None


def test_student_score_is_exact_ann_inner_product():
    torch.manual_seed(5)
    model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    query = feature("q", "table", 0.1)
    target = feature("t", "image", 0.6)

    score = model.score_pairs([query], [target])[0]
    relation_query = model.relation_query(query.embedding, "table", "image")
    index_vector = model.index_vector(target.embedding, "image")

    assert score.item() == pytest.approx(torch.dot(relation_query, index_vector).item())
    assert not torch.allclose(score, model.score_pairs([target], [query])[0])


def test_student_confidence_transform_is_positive_monotonic_and_explicit():
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        confidence_transform=True,
    )
    key = model.relation_key("table", "image")
    raw_scores = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    with torch.no_grad():
        model.confidence_alphas[key].fill_(-1.2)
        model.confidence_biases[key].fill_(0.4)

    logits = model.transform_edge_scores(
        raw_scores, "table", "image", "confidence_logit"
    )
    confidence = model.transform_edge_scores(
        raw_scores, "table", "image", "confidence"
    )

    assert model.confidence_scale("table", "image").item() > 0
    assert torch.equal(torch.argsort(raw_scores), torch.argsort(logits))
    assert torch.equal(torch.argsort(raw_scores), torch.argsort(confidence))
    torch.testing.assert_close(confidence, torch.sigmoid(logits))
    torch.testing.assert_close(
        model.transform_edge_scores(
            raw_scores, "table", "image", "raw_logit"
        ),
        raw_scores,
    )


def test_disabled_student_confidence_transform_is_checkpoint_compatible_identity():
    model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    raw_scores = torch.tensor([-1.0, 0.0, 2.0])

    assert model.confidence_parameters() == []
    torch.testing.assert_close(
        model.transform_edge_scores(
            raw_scores, "table", "text", "confidence_logit"
        ),
        raw_scores,
    )
    torch.testing.assert_close(
        model.transform_edge_scores(
            raw_scores, "table", "text", "confidence"
        ),
        torch.sigmoid(raw_scores),
    )


def test_lowrank_student_starts_at_identity_and_matches_explicit_residual():
    torch.manual_seed(5)
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=4,
        initialization="identity",
        freeze_projections=True,
        relation_param="lowrank",
        relation_rank=2,
    )
    source = torch.tensor([0.2, -0.3, 0.5, 0.7])
    destination = torch.tensor([-0.1, 0.4, 0.6, -0.2])
    key = "table_to_image"

    torch.testing.assert_close(
        model.score_embeddings(source, "table", destination, "image"),
        torch.dot(source, destination),
    )
    assert model.ann_dim == 6
    assert model.config()["relation_param"] == "lowrank"
    assert model.config()["relation_rank"] == 2
    assert stage1_training.student_relation_drift(model)[key] == 0.0
    torch.testing.assert_close(
        model.index_vector(destination, "image", source_type="table")[4:],
        torch.zeros(2),
    )

    with torch.no_grad():
        model.relation_as[key].copy_(
            torch.tensor(
                [[1.0, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, 1.0]]
            )
        )
        model.relation_bs[key].copy_(
            torch.tensor(
                [[0.1, 0.0], [0.0, -0.2], [0.3, 0.0], [0.0, 0.4]]
            )
        )

    relation = torch.eye(4) + model.relation_as[key] @ model.relation_bs[key].T
    expected = source @ relation @ destination
    score = model.score_embeddings(source, "table", destination, "image")
    query = model.relation_query(source, "table", "image")
    index = model.index_vector(destination, "image", source_type="table")
    torch.testing.assert_close(score, expected)
    torch.testing.assert_close(torch.dot(query, index), expected)
    torch.testing.assert_close(
        stage1_training.student_relation_drift(model)[key],
        torch.linalg.vector_norm(relation - torch.eye(4)).item(),
    )


def test_student_projection_drift_is_normalized_per_object_type():
    basis = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="pca",
        initialization_basis=basis,
        freeze_projections=True,
    )

    assert stage1_training.student_projection_drift(model) == {
        "table": 0.0,
        "text": 0.0,
        "image": 0.0,
    }

    with torch.no_grad():
        model.projections["table"].weight.add_(1.0)

    drift = stage1_training.student_projection_drift(model)
    assert drift["table"] == pytest.approx(1.0)
    assert drift["text"] == 0.0
    assert drift["image"] == 0.0


@pytest.mark.parametrize("relation_param", ["full", "lowrank"])
def test_student_embedding_matrix_matches_inline_relation_formula(relation_param):
    torch.manual_seed(19)
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        relation_param=relation_param,
        relation_rank=2,
    )
    sources = torch.randn(3, 4)
    destinations = torch.randn(5, 4)
    source_vectors = model.project(sources, "table")
    destination_vectors = model.project(destinations, "image")
    key = model.relation_key("table", "image")
    if relation_param == "full":
        expected = source_vectors @ model.relations[key] @ destination_vectors.T
    else:
        expected = source_vectors @ destination_vectors.T
        expected = expected + (
            (source_vectors @ model.relation_as[key])
            @ (destination_vectors @ model.relation_bs[key]).T
        )

    actual = model.score_embedding_matrix(
        sources,
        "table",
        destinations,
        "image",
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("relation_param", ["full", "lowrank"])
def test_student_pair_scoring_matches_embedding_scoring_gradients(relation_param):
    torch.manual_seed(23)
    embedding_model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        relation_param=relation_param,
        relation_rank=2,
    )
    pair_model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        relation_param=relation_param,
        relation_rank=2,
    )
    pair_model.load_state_dict(embedding_model.state_dict())
    source = feature("source", "table", 0.2)
    destination = feature("destination", "image", 0.7)

    embedding_score = embedding_model.score_embeddings(
        source.embedding,
        source.object_type,
        destination.embedding,
        destination.object_type,
    )
    pair_score = pair_model.score_pairs([source], [destination])[0]
    embedding_score.backward()
    pair_score.backward()

    torch.testing.assert_close(pair_score, embedding_score)
    for (embedding_name, embedding_parameter), (pair_name, pair_parameter) in zip(
        embedding_model.named_parameters(), pair_model.named_parameters()
    ):
        assert pair_name == embedding_name
        if embedding_parameter.grad is None:
            assert pair_parameter.grad is None
        else:
            torch.testing.assert_close(pair_parameter.grad, embedding_parameter.grad)


def test_lowrank_student_first_step_updates_the_zero_initialized_factor():
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=4,
        initialization="identity",
        relation_param="lowrank",
        relation_rank=2,
    )
    source = torch.tensor([0.2, -0.3, 0.5, 0.7])
    destination = torch.tensor([-0.1, 0.4, 0.6, -0.2])

    model.score_embeddings(source, "table", destination, "image").backward()

    assert model.relation_bs["table_to_image"].grad is not None
    assert model.relation_bs["table_to_image"].grad.abs().sum() > 0


def test_student_identity_noise_initialization_starts_near_raw_geometry():
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=4,
        initialization="identity_noise",
        initialization_noise_std=0.0,
    )

    for projection in model.projections.values():
        torch.testing.assert_close(projection.weight, torch.eye(4))
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=4,
        initialization="identity_noise",
        initialization_noise_std=0.01,
    )
    for projection in model.projections.values():
        assert torch.linalg.norm(projection.weight - torch.eye(4)) < 0.1


def test_student_identity_noise_initialization_requires_square_projection():
    with pytest.raises(ValueError, match="student_dim == input_dim"):
        StudentJoinabilityModel(
            input_dim=4,
            student_dim=3,
            initialization="identity_noise",
        )


def test_student_random_orthogonal_initialization_is_shared_low_rank_raw_geometry():
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="random_orthogonal",
    )
    projection = model.projections["table"].weight
    torch.testing.assert_close(projection @ projection.T, torch.eye(2))
    for object_type in ("text", "image"):
        torch.testing.assert_close(model.projections[object_type].weight, projection)
    for relation in model.relations.values():
        torch.testing.assert_close(relation, torch.eye(2))

    source = torch.tensor([1.0, 2.0, -1.0, 0.5])
    destination = torch.tensor([-0.5, 1.0, 3.0, 2.0])
    basis = projection.T
    expected = source @ basis @ basis.T @ destination
    actual = model.score_embeddings(source, "table", destination, "image")
    torch.testing.assert_close(actual, expected)


def test_student_pca_initialization_uses_shared_basis_and_identity_relations():
    projection = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="pca",
        initialization_basis=projection,
    )

    for object_type in ("table", "text", "image"):
        torch.testing.assert_close(model.projections[object_type].weight, projection)
    for relation in model.relations.values():
        torch.testing.assert_close(relation, torch.eye(2))


def test_frozen_pca_student_records_freeze_and_anchor_penalizes_only_relations():
    projection = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="pca",
        initialization_basis=projection,
        freeze_projections=True,
    )

    assert model.config()["freeze_projections"] is True
    assert all(
        not parameter.requires_grad for parameter in model.projections.parameters()
    )
    assert stage1_training.student_anchor_loss(model).item() == pytest.approx(0.0)

    with torch.no_grad():
        model.projections["table"].weight.add_(10.0)
        model.relations["table_to_table"][0, 0].add_(1.0)
    assert stage1_training.student_anchor_loss(model).item() == pytest.approx(0.25)


def test_evidence_anchor_weight_targets_only_table_evidence_relations():
    model = StudentJoinabilityModel(
        input_dim=2,
        student_dim=2,
        initialization="identity",
        freeze_projections=True,
    )
    with torch.no_grad():
        model.relations["table_to_text"][0, 0].add_(1.0)

    assert stage1_training.student_anchor_loss(model).item() == pytest.approx(0.25)
    assert stage1_training.student_evidence_anchor_loss(model).item() == pytest.approx(
        0.25
    )
    anchor, weighted = stage1_training._anchor_losses(model, 0.1, 1.0)
    assert anchor.item() == pytest.approx(0.25)
    assert weighted.item() == pytest.approx(0.25)
    assert stage1_training.student_relation_drift(model)["table_to_text"] == pytest.approx(
        1.0
    )

    with torch.no_grad():
        model.relations["table_to_table"][0, 0].add_(1.0)
    _anchor, weighted = stage1_training._anchor_losses(model, 0.1, 1.0)
    assert weighted.item() == pytest.approx(0.275)


def test_path_distillation_row_mask_excludes_unselected_dataset_rows():
    direct = stage1_training.ListScores(
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    evidence = stage1_training.ListScores(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        torch.ones(2, 2, dtype=torch.bool),
        torch.tensor([0, 0]),
    )
    student = stage1_training.TargetScores(direct, evidence)
    teacher = stage1_training.TargetScores(
        stage1_training.ListScores(
            torch.tensor([[0.0, 3.0], [3.0, 0.0]]),
            direct.candidate_mask,
            direct.positive_indices,
        ),
        stage1_training.ListScores(
            torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            evidence.candidate_mask,
            evidence.positive_indices,
        ),
    )

    selected = stage1_training._path_distillation_losses(
        student, teacher, 1.0, torch.tensor([False, True])
    )[0]
    expected = stage1_training._path_distillation_losses(
        stage1_training.TargetScores(
            stage1_training.ListScores(
                direct.logits[1:], direct.candidate_mask[1:], direct.positive_indices[1:]
            ),
            stage1_training.ListScores(
                evidence.logits[1:],
                evidence.candidate_mask[1:],
                evidence.positive_indices[1:],
            ),
        ),
        stage1_training.TargetScores(
            stage1_training.ListScores(
                teacher.direct.logits[1:],
                teacher.direct.candidate_mask[1:],
                teacher.direct.positive_indices[1:],
            ),
            stage1_training.ListScores(
                teacher.evidence.logits[1:],
                teacher.evidence.candidate_mask[1:],
                teacher.evidence.positive_indices[1:],
            ),
        ),
        1.0,
    )[0]

    assert selected.item() == pytest.approx(expected.item())


def test_student_in_batch_scoring_expands_lists_and_respects_maximum():
    store = FeatureStore(
        {
            object_id: feature(object_id, "table", value)
            for object_id, value in {
                "q1": 0.1,
                "q2": 0.2,
                "p1": 0.3,
                "p2": 0.4,
                "n1": 0.5,
                "n2": 0.6,
            }.items()
        }
    )
    model = StudentJoinabilityModel(4, 3)
    examples = [
        EdgeExample("q1", ("p1", "n1"), 0),
        EdgeExample("q2", ("p2", "n2"), 0),
    ]

    expanded = score_edge_batch_in_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        max_negatives=1,
        rng=random.Random(4),
    )
    unexpanded = score_edge_batch_in_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        max_negatives=0,
    )

    assert expanded.candidate_mask.sum(dim=1).tolist() == [3, 3]
    assert unexpanded.candidate_mask.sum(dim=1).tolist() == [2, 2]


def test_edge_scoring_preserves_all_ranking_positives():
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "p1": feature("p1", "table", 0.2),
            "p2": feature("p2", "table", 0.3),
            "n1": feature("n1", "table", 0.8),
        }
    )
    example = EdgeExample(
        "q",
        ("p1", "p2", "n1"),
        0,
        positive_ids=("p1", "p2"),
    )

    scores = score_edge_batch(
        StudentJoinabilityModel(4, 3),
        [example],
        store,
        torch.device("cpu"),
    )

    assert scores.positive_mask.tolist() == [[True, True, False]]


def test_relation_macro_edge_bce_excludes_unknown_candidates():
    logits = torch.tensor(
        [[0.0, 100.0, 0.0], [0.0, 0.0, -100.0]], requires_grad=True
    )
    labels = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    confirmed = torch.tensor(
        [[True, False, True], [True, True, False]]
    )

    loss = relation_macro_binary_cross_entropy_with_logits(
        logits,
        labels,
        confirmed,
        ["table_to_text", "table_to_image"],
    )
    loss.backward()

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert logits.grad[0, 1].item() == 0.0
    assert logits.grad[1, 2].item() == 0.0


def test_teacher_edge_bce_excludes_unknowns_and_macro_averages_relations():
    examples = [
        EdgeExample(
            "q1",
            ("p1", "n1"),
            0,
            source_type="table",
            destination_type="text",
            confirmed_labels=(1, None),
        ),
        EdgeExample(
            "q2",
            ("p2", "n2"),
            0,
            source_type="table",
            destination_type="text",
            confirmed_labels=(0, None),
        ),
        EdgeExample(
            "q3",
            ("p3", "n3"),
            0,
            source_type="table",
            destination_type="image",
            confirmed_labels=(1, None),
        ),
    ]
    candidate_mask = torch.ones((3, 2), dtype=torch.bool)
    positive_indices = torch.zeros(3, dtype=torch.long)
    positive_mask = torch.tensor([[True, False]] * 3)
    ranking = ListScores(
        torch.zeros((3, 2)), candidate_mask, positive_indices, positive_mask
    )
    confidence_logits = torch.tensor(
        [[0.0, 100.0], [2.0, -100.0], [-2.0, 100.0]],
        requires_grad=True,
    )
    confidence = ListScores(
        confidence_logits, candidate_mask, positive_indices, positive_mask
    )

    losses = stage1_training._teacher_edge_losses(
        examples, ranking, confidence, edge_bce_weight=1.0
    )
    losses["absolute_loss"].backward()

    log_two = torch.log(torch.tensor(2.0))
    positive_two = torch.nn.functional.softplus(torch.tensor(2.0))
    expected = ((log_two + positive_two) / 2 + positive_two) / 2
    assert losses["absolute_loss"].item() == pytest.approx(expected.item())
    assert confidence_logits.grad[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_edge_loader_preserves_positive_sets_and_ternary_confirmation(tmp_path):
    path = tmp_path / "edges.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "positive_id": "p1",
                "positive_ids": ["p1", "p2"],
                "candidate_ids": ["p1", "p2", "negative", "unknown"],
                "confirmed_positive_ids": ["p1", "p2"],
                "confirmed_negative_ids": ["negative"],
                "source_type": "table",
                "destination_type": "text",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    example = load_edge_examples(path)[0]

    assert example.positive_ids == ("p1", "p2")
    assert example.confirmed_labels == (1, 1, 0, None)


def test_student_edge_bce_uses_confirmed_labels_and_reports_participation():
    store = feature_store()
    student_model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        confidence_transform=True,
    )
    history = train_student_edges(
        student_model,
        [
            EdgeExample(
                "q",
                ("positive", "negative"),
                0,
                source_type="table",
                destination_type="table",
                confirmed_labels=(1, 0),
            )
        ],
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.0,
        edge_bce_weight=1.0,
    )

    assert history[0]["absolute_loss"] > 0
    assert history[0]["confirmed_labels"]["positive"] == 1
    assert history[0]["confirmed_labels"]["negative"] == 1


def test_lowrank_in_batch_scoring_matches_pair_scoring_without_expansion():
    store = FeatureStore(
        {
            object_id: feature(object_id, "table", value)
            for object_id, value in {
                "q1": 0.1,
                "q2": 0.2,
                "p1": 0.3,
                "p2": 0.4,
                "n1": 0.5,
                "n2": 0.6,
            }.items()
        }
    )
    model = StudentJoinabilityModel(
        4,
        3,
        relation_param="lowrank",
        relation_rank=2,
    )
    with torch.no_grad():
        model.relation_as["table_to_table"].normal_(std=0.2)
        model.relation_bs["table_to_table"].normal_(std=0.2)
    examples = [
        EdgeExample("q1", ("p1", "n1"), 0),
        EdgeExample("q2", ("p2", "n2"), 0),
    ]

    pair_scores = score_edge_batch(
        model, examples, store, torch.device("cpu")
    )
    in_batch_scores = score_edge_batch_in_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        max_negatives=0,
    )

    torch.testing.assert_close(in_batch_scores.logits, pair_scores.logits)


def test_target_in_batch_scoring_excludes_all_known_positive_targets():
    store = FeatureStore(
        {
            object_id: feature(object_id, "table", value)
            for object_id, value in {
                "q1": 0.1,
                "q2": 0.2,
                "p1": 0.3,
                "p2": 0.4,
                "n1": 0.5,
                "n2": 0.6,
            }.items()
        }
    )
    examples = [
        TargetExample(
            "q1",
            (TargetCandidate("p1", ()), TargetCandidate("n1", ())),
            0,
            0,
            positive_target_ids=("p1", "p2"),
        ),
        TargetExample(
            "q2",
            (TargetCandidate("p2", ()), TargetCandidate("n2", ())),
            0,
            0,
            positive_target_ids=("p2",),
        ),
    ]

    scores = score_target_direct_batch_in_batch(
        StudentJoinabilityModel(4, 3),
        examples,
        store,
        torch.device("cpu"),
    )

    assert scores.candidate_mask.sum(dim=1).tolist() == [3, 4]


def test_student_pca_checkpoint_loads_without_external_basis(tmp_path):
    projection = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=2,
        initialization="pca",
        initialization_basis=projection,
    )
    path = tmp_path / "student.pt"
    torch.save(checkpoint(model, "student-edge"), path)

    loaded = load_student(path, torch.device("cpu"))

    assert loaded.config()["initialization"] == "pca"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_lowrank_student_checkpoint_round_trips(tmp_path):
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        relation_param="lowrank",
        relation_rank=2,
    )
    path = tmp_path / "student_lowrank.pt"
    torch.save(checkpoint(model, "student-path"), path)

    loaded = load_student(path, torch.device("cpu"))

    assert loaded.config()["relation_param"] == "lowrank"
    assert loaded.config()["relation_rank"] == 2
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_student_confidence_transform_checkpoint_round_trips(tmp_path):
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        confidence_transform=True,
    )
    with torch.no_grad():
        model.confidence_biases["table_to_text"].fill_(-0.7)
    path = tmp_path / "student_confidence.pt"
    torch.save(checkpoint(model, "student-edge"), path)

    loaded = load_student(path, torch.device("cpu"))

    assert loaded.config()["confidence_transform"] is True
    assert loaded.config()["confidence_epsilon"] == pytest.approx(1e-6)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_pca_projection_finds_top_component_and_round_trips(tmp_path):
    embeddings = torch.tensor(
        [
            [-4.0, -0.2, 0.0],
            [-2.0, 0.1, 0.0],
            [2.0, -0.1, 0.0],
            [4.0, 0.2, 0.0],
        ]
    )
    projection, mean, explained = compute_pca_projection(
        embeddings,
        1,
        device=torch.device("cpu"),
        oversampling=2,
        iterations=2,
    )
    assert abs(projection[0, 0]) > 0.99
    assert explained > 0.99

    path = tmp_path / "pca.pt"
    torch.save(
        {
            "format_version": 1,
            "input_dim": 3,
            "student_dim": 1,
            "mean": mean,
            "projection": projection,
        },
        path,
    )
    torch.testing.assert_close(
        load_pca_projection(path, input_dim=3, student_dim=1), projection
    )

    spectrum_path = tmp_path / "spectrum.pt"
    torch.save(
        {
            "format_version": 1,
            "artifact_kind": "stage1_pca_spectrum",
            "input_dim": 3,
            "max_components": 2,
            "projection": torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            ),
        },
        spectrum_path,
    )
    torch.testing.assert_close(
        load_pca_projection(spectrum_path, input_dim=3, student_dim=1),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )


def test_pca_projection_reorthogonalizes_lowrank_components(monkeypatch):
    embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    def scaled_lowrank(values, *, q, center, niter):
        assert q == 2
        assert center
        components = torch.tensor(
            [
                [0.9996, 0.0],
                [0.0, 1.0004],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            device=values.device,
        )
        return (
            torch.zeros((values.shape[0], q), device=values.device),
            torch.tensor([2.0, 1.0], device=values.device),
            components,
        )

    monkeypatch.setattr(torch, "pca_lowrank", scaled_lowrank)

    projection, _mean, _explained = compute_pca_projection(
        embeddings,
        2,
        device=torch.device("cpu"),
        oversampling=0,
    )

    torch.testing.assert_close(projection @ projection.T, torch.eye(2))


def test_identity_student_uses_raw_inner_product_for_every_type_pair():
    model = IdentityStudentJoinabilityModel(4)
    query = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([0.5, -1.0, 2.0, 0.25])

    for source_type in ("table", "text", "image"):
        for destination_type in ("table", "text", "image"):
            relation_query = model.relation_query(
                query, source_type, destination_type
            )
            index_vector = model.index_vector(
                target, destination_type, source_type=source_type
            )

            assert relation_query is query
            assert index_vector is target
            assert torch.dot(relation_query, index_vector) == torch.dot(query, target)

    projected = ProjectedIdentityStudentJoinabilityModel(torch.eye(4))
    torch.testing.assert_close(
        projected.index_vector(target, "image", source_type="table"),
        target,
    )


def test_path_aggregator_accumulates_only_evidence_paths():
    aggregator = PathAggregator("logsumexp")
    query_evidence = torch.tensor([[[2.0, 0.0]]])
    evidence_target = torch.tensor([[[3.0, 1.0]]])
    mask = torch.tensor([[[True, True]]])

    score = aggregator(query_evidence, evidence_target, mask)

    assert score.item() == pytest.approx(torch.logsumexp(torch.tensor([5.0, 1.0]), dim=0).item())


def test_path_aggregator_all_mask_has_finite_gradients():
    query_evidence = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    evidence_target = torch.tensor([[[3.0, 1.0]]], requires_grad=True)
    mask = torch.zeros_like(query_evidence, dtype=torch.bool)

    with torch.autograd.set_detect_anomaly(True):
        score = PathAggregator("logsumexp")(query_evidence, evidence_target, mask)
        score.sum().backward()

    assert score.item() == 0.0
    assert torch.equal(query_evidence.grad, torch.zeros_like(query_evidence))
    assert torch.equal(evidence_target.grad, torch.zeros_like(evidence_target))


def test_target_scoring_and_listwise_loss_backpropagate_through_paths():
    torch.manual_seed(7)
    model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=1,
        )
    ]

    store = feature_store()
    scores = score_target_batch(model, examples, store, torch.device("cpu"), PathAggregator())
    with torch.no_grad():
        expected_direct = model.score_pairs(
            [store.get("q"), store.get("q")],
            [store.get("positive"), store.get("negative")],
        ).reshape(1, 2)
    direct_loss = listwise_cross_entropy(
        scores.direct.logits,
        scores.direct.positive_indices,
        scores.direct.candidate_mask,
    )
    evidence_loss = listwise_cross_entropy(
        scores.evidence.logits,
        scores.evidence.positive_indices,
        scores.evidence.candidate_mask,
    )
    loss = direct_loss + evidence_loss
    loss.backward()

    assert scores.direct.logits.shape == (1, 2)
    assert scores.evidence.logits.shape == (1, 2)
    assert scores.direct.positive_indices.tolist() == [0]
    assert scores.evidence.positive_indices.tolist() == [1]
    assert torch.allclose(scores.direct.logits, expected_direct)
    assert torch.equal(scores.evidence.candidate_mask, torch.tensor([[True, True]]))
    assert torch.isfinite(loss)
    assert model.relations["table_to_text"].grad is not None
    assert model.relations["text_to_table"].grad is not None


def test_listwise_cross_entropy_preserves_single_positive_behavior():
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.5, 1.5, 0.0]])
    candidate_mask = torch.tensor([[True, True, False], [True, True, True]])
    positive_indices = torch.tensor([0, 1])

    actual = listwise_cross_entropy(logits, positive_indices, candidate_mask)
    expected = torch.nn.functional.cross_entropy(
        logits.masked_fill(~candidate_mask, -torch.inf), positive_indices
    )

    torch.testing.assert_close(actual, expected)
    assert torch.equal(
        positive_indices_to_mask(positive_indices, candidate_mask),
        torch.tensor([[True, False, False], [False, True, False]]),
    )


def test_listwise_cross_entropy_uses_all_positive_probability_mass():
    logits = torch.tensor([[2.0, 1.0, -1.0]])
    candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    positive_indices = torch.tensor([0])
    positive_mask = torch.tensor([[True, True, False]])

    single_positive = listwise_cross_entropy(
        logits, positive_indices, candidate_mask
    )
    multi_positive = listwise_cross_entropy(
        logits, positive_indices, candidate_mask, positive_mask
    )
    expected = torch.logsumexp(logits, dim=-1) - torch.logsumexp(
        logits[:, :2], dim=-1
    )

    torch.testing.assert_close(multi_positive, expected.mean())
    assert multi_positive < single_positive


def test_listwise_cross_entropy_mean_log_probability_averages_positive_nll():
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 0.5, 1.5]])
    candidate_mask = torch.tensor(
        [[True, True, True], [True, True, False]]
    )
    positive_indices = torch.tensor([0, 0])
    positive_mask = torch.tensor(
        [[True, True, False], [True, False, False]]
    )

    actual = listwise_cross_entropy(
        logits,
        positive_indices,
        candidate_mask,
        positive_mask,
        positive_loss_mode="mean_log_probability",
    )
    log_probabilities = torch.log_softmax(
        logits.masked_fill(~candidate_mask, -torch.inf), dim=-1
    )
    expected = torch.stack(
        [
            -log_probabilities[0, :2].mean(),
            -log_probabilities[1, 0],
        ]
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_optional_listwise_cross_entropy_skips_rows_without_a_negative():
    logits = torch.tensor([[2.0, 1.0]], requires_grad=True)
    candidate_mask = torch.ones_like(logits, dtype=torch.bool)
    positive_mask = torch.ones_like(logits, dtype=torch.bool)

    loss = optional_listwise_cross_entropy(
        logits, torch.tensor([0]), candidate_mask, positive_mask
    )
    loss.backward()

    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_multi_positive_student_path_smoke_has_finite_losses():
    store = FeatureStore(
        {
            "q_multi": feature("q_multi", "table", 0.1),
            "positive_1": feature("positive_1", "table", 0.2),
            "positive_2": feature("positive_2", "table", 0.3),
            "negative_multi": feature("negative_multi", "table", 0.8),
            "evidence_multi": feature("evidence_multi", "text", 0.4),
        }
    )
    example = TargetExample(
        "q_multi",
        (
            TargetCandidate("positive_1", ()),
            TargetCandidate("positive_2", ("evidence_multi",)),
            TargetCandidate("negative_multi", ("evidence_multi",)),
        ),
        direct_positive_index=0,
        evidence_positive_index=1,
        dataset="multi",
        teacher_direct_logits=(2.0, 1.0, -1.0),
        teacher_evidence_logits=(1.5, 1.0, -0.5),
        positive_target_ids=("positive_1", "positive_2"),
    )
    student = StudentJoinabilityModel(input_dim=4, student_dim=3)

    scores = score_target_batch(
        student, [example], store, torch.device("cpu"), PathAggregator()
    )
    assert scores.direct.positive_mask.tolist() == [[True, True, False]]
    assert scores.evidence.positive_mask.tolist() == [[False, True, False]]

    history = train_student_paths(
        student,
        [example],
        store,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        PathAggregator(),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.5,
        positive_loss_mode="mean_log_probability",
        in_batch_negatives=True,
    )

    assert len(history) == 1
    assert all(
        torch.isfinite(torch.tensor(history[0][name]))
        for name in (
            "loss",
            "supervised_loss",
            "distillation_loss",
            "direct_supervised_loss",
            "evidence_supervised_loss",
        )
    )


def test_teacher_target_scoring_compresses_each_object_once(monkeypatch):
    model = teacher()
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=1,
        )
    ]
    compress_calls = Counter()
    original_compress_many = model.compress_many

    def counted_compress_many(features, compression_cache):
        compress_calls.update(item.object_id for item in features)
        return original_compress_many(features, compression_cache)

    monkeypatch.setattr(model, "compress_many", counted_compress_many)
    scores = score_target_batch(
        model,
        examples,
        feature_store(),
        torch.device("cpu"),
        PathAggregator(),
    )
    (scores.direct.logits.sum() + scores.evidence.logits.sum()).backward()

    assert compress_calls == {"q": 1, "positive": 1, "negative": 1, "evidence": 1}
    assert model.poolers["text"].queries.grad is not None


def test_teacher_target_scoring_deduplicates_repeated_directed_pairs(monkeypatch):
    model = teacher()
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=1,
        )
    ]
    scored_pairs = []
    original_score_pairs = model.score_pairs

    def counted_score_pairs(sources, destinations, *, compression_cache=None):
        scored_pairs.append(
            [(source.object_id, destination.object_id) for source, destination in zip(sources, destinations)]
        )
        return original_score_pairs(
            sources,
            destinations,
            compression_cache=compression_cache,
        )

    monkeypatch.setattr(model, "score_pairs", counted_score_pairs)
    score_target_batch(
        model,
        examples,
        feature_store(),
        torch.device("cpu"),
        PathAggregator(),
    )

    assert scored_pairs == [
        [("q", "positive"), ("q", "negative")],
        [("q", "evidence")],
        [("evidence", "positive"), ("evidence", "negative")],
    ]


def test_teacher_batched_compression_matches_individual_compression():
    model = teacher().eval()
    items = [
        feature("table", "table", 0.1),
        feature("text-short", "text", 0.2),
        ObjectFeatures(
            "text-long",
            "text",
            feature("unused", "text", 0.3).embedding,
            torch.cat(
                [
                    feature("unused", "text", 0.3).hidden_states,
                    feature("unused", "text", 0.3).hidden_states[:2],
                ]
            ),
        ),
        feature("image", "image", 0.4),
    ]
    expected = {item.object_id: model.compress(item) for item in items}
    actual = {}

    model.compress_many(items, actual)

    assert actual.keys() == expected.keys()
    for object_id, expected_tokens in expected.items():
        torch.testing.assert_close(actual[object_id], expected_tokens)


def test_list_scores_selects_every_row_field():
    scores = ListScores(
        logits=torch.arange(12).reshape(3, 4),
        candidate_mask=torch.tensor(
            [[True, True, True, True], [True, True, False, False], [True, False, False, False]]
        ),
        positive_indices=torch.tensor([1, 0, 0]),
        positive_mask=torch.tensor(
            [[False, True, False, True], [True, False, False, False], [True, False, False, False]]
        ),
    )

    selected = scores.select(torch.tensor([True, False, True]))

    torch.testing.assert_close(selected.logits, scores.logits[[0, 2]])
    torch.testing.assert_close(
        selected.candidate_mask, scores.candidate_mask[[0, 2]]
    )
    torch.testing.assert_close(
        selected.positive_indices, scores.positive_indices[[0, 2]]
    )
    torch.testing.assert_close(
        selected.positive_mask, scores.positive_mask[[0, 2]]
    )


def test_teacher_vectorized_pair_packing_matches_reference_assembly():
    model = teacher().eval()
    sources = [
        model.compress(feature("source-table", "table", 0.1)),
        model.compress(feature("source-text", "text", 0.2)),
    ]
    destinations = [
        model.compress(feature("destination-image", "image", 0.3)),
        model.compress(feature("destination-table", "table", 0.4)),
    ]
    source_types = ["table", "text"]
    destination_types = ["image", "table"]
    reference_sequences = []
    for source, source_type, destination, destination_type in zip(
        sources, source_types, destinations, destination_types
    ):
        source_id = TYPE_TO_ID[source_type]
        destination_id = TYPE_TO_ID[destination_type]
        pair_id = source_id * len(TYPE_TO_ID) + destination_id
        reference_sequences.append(
            torch.cat(
                [
                    (
                        model.rel_token
                        + model.type_pair_embeddings.weight[pair_id]
                    ).unsqueeze(0),
                    source
                    + model.modality_embeddings.weight[source_id]
                    + model.role_embeddings.weight[0],
                    model.sep_token.unsqueeze(0),
                    destination
                    + model.modality_embeddings.weight[destination_id]
                    + model.role_embeddings.weight[1],
                ]
            )
        )
    reference_inputs = torch.nn.utils.rnn.pad_sequence(
        reference_sequences, batch_first=True
    )
    reference_mask = torch.arange(reference_inputs.shape[1]).unsqueeze(0) >= torch.tensor(
        [sequence.shape[0] for sequence in reference_sequences]
    ).unsqueeze(1)
    reference = model.scoring_head(
        model.relation_transformer(
            reference_inputs,
            src_key_padding_mask=reference_mask,
        )[:, 0]
    ).squeeze(-1)

    actual = model.score_compressed_pairs(
        sources,
        source_types,
        destinations,
        destination_types,
    )

    torch.testing.assert_close(actual, reference)


def test_cross_modal_edge_warmup_backpropagates_through_all_path_relations():
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.2),
            "negative": feature("negative", "table", 0.8),
            "positive_text": feature("positive_text", "text", 0.3),
            "negative_text": feature("negative_text", "text", 0.7),
            "positive_image": feature("positive_image", "image", 0.4),
            "negative_image": feature("negative_image", "image", 0.6),
        }
    )
    examples = [
        EdgeExample(
            "q", ("positive_text", "negative_text"), 0,
            source_type="table", destination_type="text",
        ),
        EdgeExample(
            "positive_text", ("positive", "negative"), 0,
            source_type="text", destination_type="table",
        ),
        EdgeExample(
            "q", ("positive_image", "negative_image"), 0,
            source_type="table", destination_type="image",
        ),
        EdgeExample(
            "positive_image", ("positive", "negative"), 0,
            source_type="image", destination_type="table",
        ),
    ]
    device = torch.device("cpu")

    teacher_model = teacher()
    teacher_scores = score_edge_batch(teacher_model, examples, store, device)
    listwise_cross_entropy(
        teacher_scores.logits,
        teacher_scores.positive_indices,
        teacher_scores.candidate_mask,
    ).backward()
    type_count = len(TYPE_TO_ID)
    for source_type, destination_type in (
        ("table", "text"),
        ("text", "table"),
        ("table", "image"),
        ("image", "table"),
    ):
        index = TYPE_TO_ID[source_type] * type_count + TYPE_TO_ID[destination_type]
        gradient = teacher_model.type_pair_embeddings.weight.grad[index]
        assert torch.count_nonzero(gradient) > 0

    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    student_scores = score_edge_batch(student_model, examples, store, device)
    listwise_cross_entropy(
        student_scores.logits,
        student_scores.positive_indices,
        student_scores.candidate_mask,
    ).backward()
    for relation_key in (
        "table_to_text",
        "text_to_table",
        "table_to_image",
        "image_to_table",
    ):
        gradient = student_model.relations[relation_key].grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_edge_scoring_validates_declared_destination_type():
    example = EdgeExample(
        "q",
        ("positive", "negative"),
        0,
        source_type="table",
        destination_type="text",
    )

    with pytest.raises(ValueError, match="declared destination_type"):
        score_edge_batch(
            StudentJoinabilityModel(input_dim=4, student_dim=3),
            [example],
            feature_store(),
            torch.device("cpu"),
        )


def test_all_four_training_stages_run_on_synthetic_features():
    torch.manual_seed(11)
    store = feature_store()
    edge_examples = [
        EdgeExample(
            "q",
            ("positive", "negative"),
            positive_index=0,
            teacher_logits=(1.0, -1.0),
        )
    ]
    target_examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
            teacher_direct_logits=(1.0, -1.0),
            teacher_evidence_logits=(1.0, -1.0),
        )
    ]
    teacher_model = teacher()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    aggregator = PathAggregator()
    device = torch.device("cpu")

    teacher_edge_history = train_teacher_edges(
        teacher_model,
        edge_examples,
        store,
        torch.optim.AdamW(teacher_model.parameters(), lr=1e-3),
        device=device,
        epochs=1,
        batch_size=1,
        seed=13,
    )
    teacher_path_history = train_teacher_paths(
        teacher_model,
        target_examples,
        store,
        torch.optim.AdamW(teacher_model.parameters(), lr=1e-3),
        aggregator,
        device=device,
        epochs=1,
        batch_size=1,
        seed=13,
    )
    student_edge_history = train_student_edges(
        student_model,
        edge_examples,
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        device=device,
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
    )
    student_path_history = train_student_paths(
        student_model,
        target_examples,
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        aggregator,
        device=device,
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.5,
    )

    histories = [teacher_edge_history, teacher_path_history, student_edge_history, student_path_history]
    assert all(len(history) == 1 for history in histories)
    assert all(torch.isfinite(torch.tensor(history[0]["loss"])) for history in histories)
    assert all(history[0]["optimizer_updates"] == 1 for history in histories)
    assert all(history[0]["examples_seen"] == 1 for history in histories)


def test_training_loss_refreshes_every_hundred_steps_and_at_epoch_end():
    refreshes = [
        step
        for step in range(1, 238)
        if stage1_training._loss_refresh_due(step, 237)
    ]

    assert refreshes == [100, 200, 237]


def test_train_stage1_uses_stage_specific_feature_cache_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_stage1.py",
            "teacher-edge",
            "--features",
            "features",
            "--base-data",
            "train.jsonl",
            "--dev-data",
            "dev.jsonl",
            "--output",
            "teacher.pt",
        ],
    )

    teacher_args = train_stage1.parse_args()
    assert teacher_args.feature_cache_size == 8_000
    assert teacher_args.teacher_table_tokens_per_group is None

    monkeypatch.setattr(
        sys, "argv", [sys.argv[0], "student-edge", *sys.argv[2:]]
    )
    assert train_stage1.parse_args().feature_cache_size == 60_000


def test_train_stage1_accepts_multiple_teacher_table_tokens(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_stage1.py",
            "teacher-edge",
            "--features",
            "features",
            "--base-data",
            "train.jsonl",
            "--dev-data",
            "dev.jsonl",
            "--output",
            "teacher.pt",
            "--teacher-table-tokens-per-group",
            "4",
        ],
    )

    assert train_stage1.parse_args().teacher_table_tokens_per_group == 4


def test_fresh_teacher_defaults_to_one_table_token_per_group():
    assert train_stage1._fresh_teacher_table_tokens_per_group(None) == 1
    assert train_stage1._fresh_teacher_table_tokens_per_group(4) == 4


def test_lazy_feature_store_and_target_jsonl(tmp_path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    cached = feature("q", "table", 0.1)
    torch.save(
        {
            "embedding": cached.embedding,
            "hidden_states": cached.hidden_states,
            "token_groups": cached.token_groups,
            "row_embeddings": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        },
        feature_dir / "q.pt",
    )
    (feature_dir / "manifest.jsonl").write_text(
        json.dumps({"object_id": "q", "object_type": "table", "feature_path": "q.pt"}) + "\n",
        encoding="utf-8",
    )
    data_path = tmp_path / "targets.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "direct_positive_target_id": "positive",
                "evidence_positive_target_id": "positive",
                "candidates": [
                    {
                        "target_id": "positive",
                        "evidence_ids": [f"e{index}" for index in range(10)],
                    },
                    {"target_id": "another_positive", "evidence_ids": []},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "positive_target_ids": ["positive", "another_positive"],
                "teacher_direct_logits": [2.0, 1.0, -1.0],
                "teacher_evidence_logits": [1.5, 0.0, 0.0],
                "dataset": "2k",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = FeatureStore.from_path(feature_dir)
    examples = load_target_examples(data_path)

    assert store.get("q").object_type == "table"
    assert store.get("q").row_embeddings.shape == (1, 4)
    scoring_copy = store.get("q").for_scoring(
        torch.device("cpu"), include_hidden=True
    )
    assert scoring_copy.row_embeddings is None
    assert examples[0].direct_positive_index == 0
    assert examples[0].evidence_positive_index == 0
    assert examples[0].candidates[0].evidence_ids == tuple(
        f"e{index}" for index in range(10)
    )
    assert examples[0].dataset == "2k"
    assert examples[0].teacher_direct_logits == (2.0, 1.0, -1.0)
    assert examples[0].teacher_evidence_logits == (1.5, 0.0, 0.0)
    assert examples[0].positive_target_ids == ("positive", "another_positive")


def test_target_loader_rejects_positive_missing_from_candidates(tmp_path):
    path = tmp_path / "targets.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "direct_positive_target_id": "positive",
                "evidence_positive_target_id": "positive",
                "positive_target_ids": ["positive", "missing_positive"],
                "candidates": [
                    {"target_id": "positive", "evidence_ids": []},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing candidates: missing_positive"):
        load_target_examples(path)


def test_lazy_feature_store_loads_teacher_tier_only_when_requested(tmp_path):
    feature_dir = tmp_path / "features"
    (feature_dir / "objects").mkdir(parents=True)
    (feature_dir / "teacher_objects").mkdir()
    torch.save(
        {
            "embedding": torch.ones(4),
            "row_embeddings": torch.ones(1, 4),
        },
        feature_dir / "objects" / "q.pt",
    )
    torch.save(
        {
            "hidden_states": torch.ones(2, 4),
            "token_groups": torch.tensor([0, 1]),
            "embedding": torch.zeros(4),
            "row_embeddings": torch.zeros(1, 4),
        },
        feature_dir / "teacher_objects" / "q.pt",
    )
    (feature_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "object_id": "q",
                "object_type": "table",
                "feature_path": "objects/q.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feature_dir / "teacher_manifest.jsonl").write_text(
        json.dumps(
            {
                "object_id": "q",
                "object_type": "table",
                "teacher_feature_path": "teacher_objects/q.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = FeatureStore.from_path(feature_dir)
    base = store.get("q", include_hidden=False)
    teacher_features = store.get("q", include_hidden=True)

    assert base.hidden_states is None
    assert base.row_embeddings.shape == (1, 4)
    assert teacher_features.hidden_states.shape == (2, 4)
    assert teacher_features.token_groups.tolist() == [0, 1]
    assert torch.equal(teacher_features.embedding, torch.ones(4))
    assert torch.equal(teacher_features.row_embeddings, torch.ones(1, 4))
    assert store.embedding_dimension() == 4
    assert store.teacher_dimension() == 4

    torch.save(
        {"token_groups": torch.tensor([0, 1])},
        feature_dir / "teacher_objects" / "q.pt",
    )
    with pytest.raises(ValueError, match="has no hidden_states"):
        FeatureStore.from_path(feature_dir).get("q", include_hidden=True)


def test_lazy_feature_store_preserves_native_teacher_dtype(tmp_path):
    feature_dir = tmp_path / "features"
    (feature_dir / "objects").mkdir(parents=True)
    (feature_dir / "teacher_objects").mkdir()
    torch.save(
        {"embedding": torch.ones(4)},
        feature_dir / "objects" / "text.pt",
    )
    torch.save(
        {"hidden_states": torch.ones(3, 4, dtype=torch.bfloat16)},
        feature_dir / "teacher_objects" / "text.pt",
    )
    (feature_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "object_id": "text",
                "object_type": "text",
                "feature_path": "objects/text.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (feature_dir / "teacher_manifest.jsonl").write_text(
        json.dumps(
            {
                "object_id": "text",
                "object_type": "text",
                "teacher_feature_path": "teacher_objects/text.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cached = FeatureStore.from_path(feature_dir).get("text", include_hidden=True)
    fp32 = cached.for_scoring(torch.device("cpu"), include_hidden=True)
    bf16 = cached.for_scoring(
        torch.device("cpu"),
        include_hidden=True,
        hidden_dtype=torch.bfloat16,
    )

    assert cached.hidden_states.dtype == torch.bfloat16
    assert fp32.hidden_states.dtype == torch.float32
    assert bf16.hidden_states.dtype == torch.bfloat16


def test_lazy_feature_store_keeps_frequency_selected_hot_objects(monkeypatch, tmp_path):
    feature_dir = tmp_path / "features"
    (feature_dir / "objects").mkdir(parents=True)
    (feature_dir / "teacher_objects").mkdir()
    manifest = []
    teacher_manifest = []
    for object_id in ("hot", "cold-1", "cold-2"):
        torch.save(
            {"embedding": torch.ones(4)},
            feature_dir / "objects" / f"{object_id}.pt",
        )
        torch.save(
            {"hidden_states": torch.ones(2, 4)},
            feature_dir / "teacher_objects" / f"{object_id}.pt",
        )
        manifest.append(
            json.dumps(
                {
                    "object_id": object_id,
                    "object_type": "text",
                    "feature_path": f"objects/{object_id}.pt",
                }
            )
        )
        teacher_manifest.append(
            json.dumps(
                {
                    "object_id": object_id,
                    "object_type": "text",
                    "teacher_feature_path": f"teacher_objects/{object_id}.pt",
                }
            )
        )
    (feature_dir / "manifest.jsonl").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    (feature_dir / "teacher_manifest.jsonl").write_text(
        "\n".join(teacher_manifest) + "\n", encoding="utf-8"
    )

    loads = Counter()
    original_load = stage1_features._load_tensor_file

    def counted_load(path):
        loads[path.stem] += 1
        return original_load(path)

    monkeypatch.setattr(stage1_features, "_load_tensor_file", counted_load)
    store = FeatureStore.from_path(feature_dir, cache_size=1, cache_bytes=1)
    hot_bytes = store.estimated_feature_bytes("hot", include_hidden=True)
    plan = store.configure_hot_cache(
        {"hot": 100.0, "cold-1": 1.0, "cold-2": 1.0},
        byte_budget=hot_bytes,
        object_budget=1,
        include_hidden=True,
    )

    for object_id in ("hot", "cold-1", "cold-2", "hot"):
        store.get(object_id, include_hidden=True)

    assert plan["planned_objects"] == 1
    assert loads["hot"] == 2
    assert store.cache_info()["hot_objects"] == 1
    assert store.cache_info()["lru_objects"] == 0


def test_legacy_feature_cache_conversion_pools_tables_and_drops_unneeded_hidden(
    tmp_path,
):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    records = []
    legacy_table_hidden = None
    for object_id, object_type, cached_type in (
        ("q", "table", "table"),
        ("t", "table_fragment", "table"),
        ("e", "text_asset", "text"),
    ):
        cached = feature(object_id, cached_type, 0.2)
        payload = {
            "embedding": cached.embedding,
            "hidden_states": cached.hidden_states,
        }
        if object_id == "q":
            payload["token_groups"] = cached.token_groups
        if object_id == "t":
            legacy_table_hidden = cached.hidden_states
        torch.save(payload, legacy / f"{object_id}.pt")
        records.append(
            {
                "object_id": object_id,
                "object_type": object_type,
                "feature_path": f"{object_id}.pt",
            }
        )
    (legacy / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    teacher_data = tmp_path / "teacher.jsonl"
    teacher_data.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidate_ids": ["t"],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "compact"
    args = argparse.Namespace(
        input_dir=str(legacy),
        output_dir=str(output),
        teacher_data=[str(teacher_data)],
        teacher_split="train",
    )

    compact_cache.run(args)
    compact_cache.run(args)

    store = FeatureStore.from_path(output)
    assert store.get("q").hidden_states.shape == (2, 4)
    assert store.get("q").token_groups is None
    assert torch.equal(store.get("t").hidden_states, legacy_table_hidden)
    assert store.get("t").token_groups is None
    assert store.get("e").hidden_states is None
    assert len((output / "manifest.jsonl").read_text().splitlines()) == 3
    assert len((output / "teacher_manifest.jsonl").read_text().splitlines()) == 2

    first_record = json.loads(
        (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    (output / first_record["feature_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="missing feature file"):
        compact_cache.run(args)


def test_compact_resume_validates_existing_payload_content_and_path(tmp_path):
    output = tmp_path / "compact"
    (output / "objects").mkdir(parents=True)
    (output / "teacher_objects").mkdir()
    object_id = "q"
    name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".pt"
    source = {
        "object_id": object_id,
        "object_type": "table",
        "source_fingerprint": "source-v1",
    }
    base_record = {
        **source,
        "feature_path": f"objects/{name}",
    }
    base_path = output / base_record["feature_path"]

    def validate_base(record=base_record):
        compact_cache._validate_completed_record(
            output,
            record,
            source,
            path_field="feature_path",
            directory="objects",
            required_tensor="embedding",
        )

    base_path.write_bytes(b"")
    with pytest.raises(ValueError, match="payload is empty"):
        validate_base()

    torch.save(torch.ones(4), base_path)
    with pytest.raises(ValueError, match="expected a feature mapping"):
        validate_base()

    torch.save({}, base_path)
    with pytest.raises(ValueError, match="no tensor embedding"):
        validate_base()

    wrong_path_record = {**base_record, "feature_path": "objects/wrong.pt"}
    with pytest.raises(ValueError, match="cached feature_path must be"):
        validate_base(wrong_path_record)

    teacher_record = {
        **source,
        "teacher_feature_path": f"teacher_objects/{name}",
    }
    torch.save({}, output / teacher_record["teacher_feature_path"])
    with pytest.raises(ValueError, match="no tensor hidden_states"):
        compact_cache._validate_completed_record(
            output,
            teacher_record,
            source,
            path_field="teacher_feature_path",
            directory="teacher_objects",
            required_tensor="hidden_states",
        )


def test_target_loader_rejects_obsolete_merged_teacher_logits(tmp_path):
    path = tmp_path / "targets.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "direct_positive_target_id": "positive",
                "evidence_positive_target_id": "positive",
                "candidates": [
                    {"target_id": "positive", "evidence_ids": ["evidence"]},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "teacher_logits": [2.0, -1.0],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="merged target teacher_logits are obsolete"):
        load_target_examples(path)


def test_target_loader_rejects_shared_channel_positive(tmp_path):
    path = tmp_path / "targets.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "positive_target_id": "positive",
                "candidates": [
                    {"target_id": "positive", "evidence_ids": ["evidence"]},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require separate"):
        load_target_examples(path)


class FakeQwenModel:
    device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


class FakeQwenEmbedder:
    model = FakeQwenModel()
    max_length = 128

    class Tokenizer:
        def __call__(self, text, **kwargs):
            del kwargs
            matches = list(__import__("re").finditer(r"\S+", text))
            return {
                "input_ids": list(range(1, len(matches) + 1)),
                "offset_mapping": [match.span() for match in matches],
            }

    class Processor:
        tokenizer = None

        def __init__(self):
            self.tokenizer = FakeQwenEmbedder.Tokenizer()

        @staticmethod
        def apply_chat_template(conversation, **kwargs):
            del kwargs
            return f"system prompt user {conversation} assistant"

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.forward_calls = 0
        self.instructions = []
        self.processor = self.Processor()

    def format_model_input(self, *, text=None, image=None, instruction=None):
        del image
        self.instructions.append(instruction)
        return text or "image"

    def _preprocess_inputs(self, conversations):
        rendered = [self.processor.apply_chat_template(value) for value in conversations]
        lengths = [len(self.processor.tokenizer(value)["input_ids"]) for value in rendered]
        width = max(lengths)
        input_ids = torch.arange(1, width + 1).repeat(len(lengths), 1)
        attention_mask = torch.arange(width).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
        return {"input_ids": input_ids, "attention_mask": attention_mask.long()}

    def forward(self, inputs):
        self.forward_calls += 1
        hidden = inputs["input_ids"].float().unsqueeze(-1).repeat(1, 1, 4)
        return {"last_hidden_state": hidden, "attention_mask": inputs["attention_mask"]}

    @staticmethod
    def _pooling_last(hidden_states, attention_mask):
        assert attention_mask.dtype == torch.long
        indices = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0]), indices]


def test_qwen_cache_builder_structurally_pools_table_parts(tmp_path):
    embedder = FakeQwenEmbedder()
    payload = build_object_features(
        embedder,
        {
            "object_id": "q",
            "object_type": "table",
            "embedding_role": "target",
            "table_parts": ["schema player country", "row Messi Argentina"],
        },
        input_dir=tmp_path,
        instruction="represent",
        storage_dtype=torch.float16,
    )

    assert embedder.forward_calls == 1
    assert payload["embedding"].shape == (4,)
    assert payload["embedding"].norm().item() == pytest.approx(1.0)
    assert payload["hidden_states"].shape == (2, 4)
    assert payload["hidden_states"].dtype == torch.float32
    assert "token_groups" not in payload


def test_qwen_cache_encoder_skips_hidden_state_cpu_payload_when_not_requested():
    outputs = stage1_cache.encode_inputs(
        FakeQwenEmbedder(),
        [{"text": "base-only object", "instruction": "represent"}],
        include_hidden=False,
    )

    embedding, hidden_states, input_ids = outputs[0]
    assert embedding.shape == (4,)
    assert hidden_states is None
    assert input_ids is None


def test_qwen_cache_builder_batches_base_only_non_table_objects(tmp_path):
    embedder = FakeQwenEmbedder()
    payloads = stage1_cache.build_base_object_features_batch(
        embedder,
        [
            {"object_id": "a", "object_type": "text", "text": "short"},
            {"object_id": "b", "object_type": "text", "text": "longer text"},
        ],
        input_dir=tmp_path,
        instruction=None,
    )

    assert embedder.forward_calls == 1
    assert len(payloads) == 2
    assert all(set(payload) == {"embedding"} for payload in payloads)
    assert all(payload["embedding"].shape == (4,) for payload in payloads)


def test_qwen_cache_batch_cost_uses_capped_image_pixels(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (20, 10)).save(image_path)

    assert stage1_cache._base_object_batch_cost(
        {"object_id": "i", "object_type": "image", "image": str(image_path)},
        input_dir=tmp_path,
        max_image_pixels=150,
    ) == 150
    assert stage1_cache._base_object_batch_cost(
        {"object_id": "t", "object_type": "text", "text": "12345"},
        input_dir=tmp_path,
        max_image_pixels=150,
    ) == 5


def test_qwen_cache_builder_skips_table_teacher_features_for_base_only(
    tmp_path, monkeypatch
):
    embedder = FakeQwenEmbedder()
    monkeypatch.setattr(
        stage1_cache,
        "_table_token_groups",
        lambda *_args, **_kwargs: pytest.fail("base-only build grouped table tokens"),
    )
    monkeypatch.setattr(
        stage1_cache,
        "structural_table_pool_with_groups",
        lambda *_args, **_kwargs: pytest.fail("base-only build pooled table tokens"),
    )

    payload = build_object_features(
        embedder,
        {
            "object_id": "t",
            "object_type": "table",
            "embedding_role": "target",
            "table_parts": ["schema player country", "row Messi Argentina"],
        },
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
        include_hidden=False,
    )

    assert set(payload) == {"embedding"}
    assert embedder.forward_calls == 1


def test_qwen_cache_builder_skips_query_rows_for_teacher_only(tmp_path):
    embedder = FakeQwenEmbedder()

    payload = build_object_features(
        embedder,
        {
            "object_id": "q",
            "object_type": "table",
            "embedding_role": "query",
            "table_parts": ["schema player country", "row Messi Argentina"],
        },
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
        include_row_embeddings=False,
    )

    assert "hidden_states" in payload
    assert "row_embeddings" not in payload
    assert embedder.forward_calls == 1


def test_qwen_cache_builder_keeps_multiple_tokens_per_table_group(tmp_path):
    payload = build_object_features(
        FakeQwenEmbedder(),
        {
            "object_id": "t",
            "object_type": "table",
            "embedding_role": "target",
            "table_parts": ["schema player country", "row Messi Argentina"],
        },
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
        table_tokens_per_group=2,
    )

    assert payload["hidden_states"].shape == (4, 4)
    assert payload["token_groups"].tolist() == [0, 0, 1, 1]


def test_teacher_object_ids_collects_only_the_selected_split(tmp_path):
    edges = tmp_path / "edges.jsonl"
    targets = tmp_path / "targets.jsonl"
    edges.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q",
                        "candidate_ids": ["positive", "negative"],
                        "split": "train",
                    }
                ),
                json.dumps(
                    {
                        "query_id": "held_out",
                        "candidate_ids": ["test_target"],
                        "split": "test",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    targets.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidates": [
                    {"target_id": "positive", "evidence_ids": ["evidence"]}
                ],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert teacher_object_ids([edges, targets]) == {
        "q",
        "positive",
        "negative",
        "evidence",
    }
    selector = tmp_path / "selector.jsonl"
    selector.write_text(json.dumps({"object_id": "selected"}) + "\n", encoding="utf-8")
    assert teacher_object_ids([selector], split=None) == {"selected"}


def test_qwen_cache_builder_adds_query_row_routing_embeddings(tmp_path):
    embedder = FakeQwenEmbedder()
    payload = build_object_features(
        embedder,
        {
            "object_id": "q",
            "object_type": "table",
            "embedding_role": "query",
            "table_parts": [
                "schema player country",
                "row Messi Argentina",
                "row Mbappe France",
            ],
        },
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
    )

    assert embedder.forward_calls == 2
    assert payload["row_embeddings"].shape == (2, 4)
    assert torch.allclose(payload["row_embeddings"].norm(dim=-1), torch.ones(2))
    assert embedder.instructions == [
        EMBEDDING_INSTRUCTIONS[("query", "table")],
        EMBEDDING_INSTRUCTIONS[("query", "table")],
        EMBEDDING_INSTRUCTIONS[("query_row", "table")],
        EMBEDDING_INSTRUCTIONS[("query_row", "table")],
    ]


def test_qwen_cache_builder_batches_query_row_routing_embeddings(tmp_path):
    embedder = FakeQwenEmbedder()
    payload = build_object_features(
        embedder,
        {
            "object_id": "q",
            "object_type": "table",
            "embedding_role": "query",
            "table_parts": ["schema", *[f"row {index}" for index in range(5)]],
        },
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
        table_row_batch_size=2,
    )

    assert embedder.forward_calls == 4
    assert payload["row_embeddings"].shape == (5, 4)


def test_qwen_cache_builder_truncates_oversized_table_parts_and_pools_each_group(tmp_path):
    class TruncatingFakeQwenEmbedder(FakeQwenEmbedder):
        max_length = 80

        def _preprocess_inputs(self, conversations):
            inputs = super()._preprocess_inputs(conversations)
            return {
                name: tensor[:, : self.max_length]
                for name, tensor in inputs.items()
            }

        def forward(self, inputs):
            self.forward_calls += 1
            token_ids = inputs["input_ids"].float()
            hidden = torch.stack(
                [
                    token_ids,
                    token_ids.square(),
                    token_ids.remainder(7),
                    torch.ones_like(token_ids),
                ],
                dim=-1,
            )
            return {
                "last_hidden_state": hidden,
                "attention_mask": inputs["attention_mask"],
            }

        class Tokenizer(FakeQwenEmbedder.Tokenizer):
            def __call__(self, text, **kwargs):
                tokenized = super().__call__(text, **kwargs)
                max_length = kwargs.get("max_length")
                if kwargs.get("truncation") and max_length is not None:
                    tokenized = {
                        name: values[:max_length]
                        for name, values in tokenized.items()
                    }
                return tokenized

        class Processor(FakeQwenEmbedder.Processor):
            def __init__(self):
                self.tokenizer = TruncatingFakeQwenEmbedder.Tokenizer()

    record = {
        "object_id": "q",
        "object_type": "table",
        "embedding_role": "query",
        "table_parts": [
            "schema",
            "row " + "x " * 5000,
            "row retained",
        ],
    }
    payload = build_object_features(
        TruncatingFakeQwenEmbedder(),
        record,
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
    )
    base_payload = build_object_features(
        TruncatingFakeQwenEmbedder(),
        record,
        input_dir=tmp_path,
        instruction=None,
        storage_dtype=torch.float16,
        include_hidden=False,
    )

    assert payload["hidden_states"].shape == (3, 4)
    assert "token_groups" not in payload
    assert payload["row_embeddings"].shape == (2, 4)
    assert torch.equal(payload["embedding"], base_payload["embedding"])
    assert torch.equal(payload["row_embeddings"], base_payload["row_embeddings"])


def test_qwen_cache_run_writes_base_tier_and_incremental_teacher_tier(
    tmp_path, monkeypatch
):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "object_id": "q",
                    "object_type": "table",
                    "embedding_role": "query",
                    "table_parts": ["schema", "row q"],
                },
                {
                    "object_id": "t",
                    "object_type": "table",
                    "embedding_role": "target",
                    "table_parts": ["schema", "row t"],
                },
                {"object_id": "e", "object_type": "text", "text": "evidence"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    initial = tmp_path / "initial.jsonl"
    initial.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidate_ids": ["e"],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    hard = tmp_path / "hard.jsonl"
    hard.write_text(
        json.dumps(
            {
                "query_id": "q",
                "candidate_ids": ["t"],
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "features"
    args = argparse.Namespace(
        input_jsonl=str(objects),
        output_dir=str(output),
        model_dir=str(tmp_path / "model"),
        device="cpu",
        dtype="fp16",
        instruction=None,
        teacher_data=[str(initial)],
        teacher_split="train",
    )
    monkeypatch.setattr(stage1_cache, "_load_embedder_class", lambda _path: FakeQwenEmbedder)

    stage1_cache.run(args)
    args.teacher_data = [str(hard)]
    stage1_cache.run(args)

    assert len((output / "manifest.jsonl").read_text().splitlines()) == 3
    assert len((output / "teacher_manifest.jsonl").read_text().splitlines()) == 3
    store = FeatureStore.from_path(output)
    assert store.get("t", include_hidden=False).hidden_states is None
    teacher_features = store.get("t", include_hidden=True)
    assert teacher_features.hidden_states.shape == (2, 4)
    assert teacher_features.token_groups is None


def test_qwen_cache_run_batches_base_only_non_table_objects(tmp_path, monkeypatch):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "".join(
            json.dumps(
                {
                    "object_id": f"text_{index}",
                    "object_type": "text",
                    "text": f"evidence {index}",
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    instances = []

    class TrackingFakeQwenEmbedder(FakeQwenEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr(
        stage1_cache,
        "_load_embedder_class",
        lambda _path: TrackingFakeQwenEmbedder,
    )
    stage1_cache.run(
        argparse.Namespace(
            input_jsonl=str(objects),
            output_dir=str(tmp_path / "features"),
            model_dir=str(tmp_path / "model"),
            device="cpu",
            dtype="fp16",
            instruction=None,
            teacher_data=[],
            teacher_split="train",
            object_batch_size=2,
        )
    )

    assert instances[0].forward_calls == 2
    assert len((tmp_path / "features" / "manifest.jsonl").read_text().splitlines()) == 3


def test_qwen_cache_run_prefetches_base_objects_off_the_gpu_thread(
    tmp_path, monkeypatch
):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "".join(
            json.dumps(
                {
                    "object_id": f"text_{index}",
                    "object_type": "text",
                    "text": f"evidence {index}",
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    preprocess_threads = []
    forward_threads = []

    class TrackingFakeQwenEmbedder(FakeQwenEmbedder):
        def _preprocess_inputs(self, conversations):
            preprocess_threads.append(threading.current_thread().name)
            return super()._preprocess_inputs(conversations)

        def forward(self, inputs):
            forward_threads.append(threading.current_thread().name)
            return super().forward(inputs)

    monkeypatch.setattr(
        stage1_cache,
        "_load_embedder_class",
        lambda _path: TrackingFakeQwenEmbedder,
    )
    stage1_cache.run(
        argparse.Namespace(
            input_jsonl=str(objects),
            output_dir=str(tmp_path / "features"),
            model_dir=str(tmp_path / "model"),
            device="cpu",
            dtype="fp16",
            instruction=None,
            teacher_data=[],
            teacher_split="train",
            object_batch_size=1,
            object_batch_buffer_size=2,
            prefetch_base_objects=True,
        )
    )

    assert len(preprocess_threads) == 3
    assert all(name != "MainThread" for name in preprocess_threads)
    assert forward_threads == ["MainThread"] * 3


def test_qwen_cache_prefetched_batch_falls_back_without_dropping_neighbors(
    tmp_path, monkeypatch
):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "".join(
            json.dumps(
                {
                    "object_id": f"text_{index}",
                    "object_type": "text",
                    "text": f"evidence {index}",
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    instances = []

    class CollapsingFakeQwenEmbedder(FakeQwenEmbedder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def _preprocess_inputs(self, conversations):
            if len(conversations) > 1:
                conversations = conversations[:1]
            return super()._preprocess_inputs(conversations)

    monkeypatch.setattr(
        stage1_cache,
        "_load_embedder_class",
        lambda _path: CollapsingFakeQwenEmbedder,
    )
    output = tmp_path / "features"
    stage1_cache.run(
        argparse.Namespace(
            input_jsonl=str(objects),
            output_dir=str(output),
            model_dir=str(tmp_path / "model"),
            device="cpu",
            dtype="fp16",
            instruction=None,
            teacher_data=[],
            teacher_split="train",
            object_batch_size=2,
            object_batch_buffer_size=3,
            prefetch_base_objects=True,
            base_prefetch_workers=1,
        )
    )

    assert len((output / "manifest.jsonl").read_text().splitlines()) == 3
    assert instances[0].forward_calls == 4


def test_qwen_cache_run_can_write_features_off_the_gpu_thread(tmp_path, monkeypatch):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        "".join(
            json.dumps(
                {
                    "object_id": f"text_{index}",
                    "object_type": "text",
                    "text": f"evidence {index}",
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    save_threads = []
    original_save = stage1_cache.torch.save

    def tracking_save(*args, **kwargs):
        save_threads.append(threading.current_thread().name)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(stage1_cache, "_load_embedder_class", lambda _path: FakeQwenEmbedder)
    monkeypatch.setattr(stage1_cache.torch, "save", tracking_save)
    output = tmp_path / "features"
    stage1_cache.run(
        argparse.Namespace(
            input_jsonl=str(objects),
            output_dir=str(output),
            model_dir=str(tmp_path / "model"),
            device="cpu",
            dtype="fp16",
            instruction=None,
            teacher_data=[],
            teacher_split="train",
            object_batch_size=1,
            async_write_workers=1,
            async_write_queue_size=2,
        )
    )

    assert len((output / "manifest.jsonl").read_text().splitlines()) == 3
    assert len(save_threads) == 3
    assert all(name != "MainThread" for name in save_threads)


def test_qwen_cache_run_persists_multi_token_table_groups(tmp_path, monkeypatch):
    objects = tmp_path / "objects.jsonl"
    objects.write_text(
        json.dumps(
            {
                "object_id": "t",
                "object_type": "table",
                "embedding_role": "target",
                "table_parts": ["schema player country", "row Messi Argentina"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selected = tmp_path / "selected.jsonl"
    selected.write_text(json.dumps({"object_id": "t"}) + "\n", encoding="utf-8")
    output = tmp_path / "features"
    args = argparse.Namespace(
        input_jsonl=str(objects),
        output_dir=str(output),
        model_dir=str(tmp_path / "model"),
        device="cpu",
        dtype="fp16",
        instruction=None,
        teacher_data=[str(selected)],
        teacher_split="all",
        table_tokens_per_group=2,
    )
    monkeypatch.setattr(stage1_cache, "_load_embedder_class", lambda _path: FakeQwenEmbedder)

    stage1_cache.run(args)

    features = FeatureStore.from_path(output).get("t", include_hidden=True)
    assert features.hidden_states.shape == (4, 4)
    assert features.token_groups.tolist() == [0, 0, 1, 1]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["table_tokens_per_group"] == 2


def test_embedding_instructions_distinguish_role_modality_and_query_rows():
    with pytest.raises(ValueError, match="must declare embedding_role"):
        embedding_instructions({"object_id": "legacy"}, "table")

    query, query_row, role = embedding_instructions(
        {"object_id": "q", "embedding_role": "query"}, "table"
    )
    target, target_row, _ = embedding_instructions(
        {"object_id": "t", "embedding_role": "target"}, "table"
    )
    text, text_row, _ = embedding_instructions(
        {"object_id": "e1", "embedding_role": "evidence"}, "text"
    )
    image, image_row, _ = embedding_instructions(
        {"object_id": "e2", "embedding_role": "evidence"}, "image"
    )

    assert role == "query"
    assert query == EMBEDDING_INSTRUCTIONS[("query", "table")]
    assert query_row == EMBEDDING_INSTRUCTIONS[("query_row", "table")]
    assert target == EMBEDDING_INSTRUCTIONS[("target", "table")]
    assert text == EMBEDDING_INSTRUCTIONS[("evidence", "text")]
    assert image == EMBEDDING_INSTRUCTIONS[("evidence", "image")]
    assert target_row is text_row is image_row is None
    assert len({query, query_row, target, text, image}) == 5
    assert all("context" not in value.casefold() for value in (query, query_row, target, text, image))
    assert "label" not in image.casefold()


def test_student_ann_scores_and_zero_one_hop_retrieval(tmp_path):
    torch.manual_seed(17)
    store = feature_store()
    model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    ids_by_type = {
        "table": ["positive", "negative"],
        "text": ["evidence"],
        "image": [],
    }
    build_indices(
        model,
        store,
        ids_by_type,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic",
        batch_size=2,
        m=8,
        ef_construction=20,
        ef_search=20,
    )
    indices = StudentANNIndices(
        model,
        store,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic",
    )

    table_hits = indices.search("q", "table", 2)
    results = retrieve_zero_one_hop(
        "q",
        indices,
        direct_k=2,
        evidence_k=1,
        targets_per_evidence=2,
        result_k=2,
        evidence_types=("text",),
    )

    expected = {
        target_id: model.score_pairs([store.get("q")], [store.get(target_id)])[0].item()
        for target_id in ("positive", "negative")
    }
    assert {target_id for target_id, _ in table_hits} == set(expected)
    for target_id, score in table_hits:
        assert score == pytest.approx(expected[target_id], abs=1e-5)
    assert {result["target_id"] for result in results} == {"positive", "negative"}
    assert all({path["kind"] for path in result["paths"]} == {"direct", "evidence"} for result in results)


def test_student_ann_confidence_rescores_without_changing_relation_order(tmp_path):
    torch.manual_seed(17)
    store = feature_store()
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        confidence_transform=True,
    )
    with torch.no_grad():
        model.confidence_alphas["table_to_table"].fill_(-0.5)
        model.confidence_biases["table_to_table"].fill_(0.8)
    ids_by_type = {
        "table": ["positive", "negative"],
        "text": ["evidence"],
        "image": [],
    }
    build_indices(
        model,
        store,
        ids_by_type,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic-confidence",
        batch_size=2,
        m=8,
        ef_construction=20,
        ef_search=20,
    )
    raw_indices = StudentANNIndices(
        model,
        store,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic-confidence",
    )
    confidence_indices = StudentANNIndices(
        model,
        store,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic-confidence",
        score_space="confidence",
    )

    raw_hits = raw_indices.search("q", "table", 2)
    confidence_hits = confidence_indices.search("q", "table", 2)

    assert [target_id for target_id, _score in confidence_hits] == [
        target_id for target_id, _score in raw_hits
    ]
    expected = {
        target_id: model.confidence_pairs(
            [store.get("q")], [store.get(target_id)]
        )[0].item()
        for target_id in ("positive", "negative")
    }
    for target_id, score in confidence_hits:
        assert score == pytest.approx(expected[target_id], abs=1e-5)


def test_lowrank_student_ann_scores_are_exact(tmp_path, monkeypatch):
    torch.manual_seed(17)
    store = feature_store()
    model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        relation_param="lowrank",
        relation_rank=2,
    )
    with torch.no_grad():
        model.relation_as["table_to_table"].normal_(std=0.2)
        model.relation_bs["table_to_table"].normal_(std=0.2)
    ids_by_type = {
        "table": ["positive", "negative"],
        "text": ["evidence"],
        "image": [],
    }
    embedding_reads = Counter()
    embedding_features = store.embedding_features

    def count_embedding_reads(object_id):
        embedding_reads[object_id] += 1
        return embedding_features(object_id)

    monkeypatch.setattr(store, "embedding_features", count_embedding_reads)
    manifest = build_indices(
        model,
        store,
        ids_by_type,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic",
        batch_size=2,
        m=8,
        ef_construction=20,
        ef_search=20,
    )
    assert embedding_reads == Counter(
        {"positive": 1, "negative": 1, "evidence": 1}
    )
    indices = StudentANNIndices(
        model,
        store,
        tmp_path,
        device=torch.device("cpu"),
        checkpoint_sha256="synthetic",
    )

    hits = indices.search("q", "table", 2)
    expected = {
        target_id: model.score_pairs(
            [store.get("q")], [store.get(target_id)]
        )[0].item()
        for target_id in ("positive", "negative")
    }

    assert manifest["relation_param"] == "lowrank"
    assert manifest["relation_rank"] == 2
    assert manifest["ann_dim"] == 5
    assert "table_to_table" in manifest["types"]
    for target_id, score in hits:
        assert score == pytest.approx(expected[target_id], abs=1e-5)


def test_raw_embedding_ann_uses_frozen_vectors_without_student_head(tmp_path):
    store = feature_store()
    ids_by_type = {
        "table": ["positive", "negative"],
        "text": ["evidence"],
        "image": [],
    }
    build_raw_embedding_indices(
        store,
        ids_by_type,
        tmp_path,
        corpus_sha256="synthetic-corpus",
        batch_size=2,
        m=8,
        ef_construction=20,
        ef_search=20,
    )
    indices = RawEmbeddingANNIndices(
        store,
        tmp_path,
        corpus_sha256="synthetic-corpus",
    )

    hits = indices.search("q", "table", 2)
    expected = {
        target_id: torch.dot(
            store.get("q", include_hidden=False).embedding,
            store.get(target_id, include_hidden=False).embedding,
        ).item()
        for target_id in ("positive", "negative")
    }

    assert {target_id for target_id, _ in hits} == set(expected)
    for target_id, score in hits:
        assert score == pytest.approx(expected[target_id], abs=1e-5)


def test_raw_embedding_ann_rejects_id_count_mismatch(tmp_path):
    store = feature_store()
    build_raw_embedding_indices(
        store,
        {"table": ["positive", "negative"], "text": [], "image": []},
        tmp_path,
        corpus_sha256="synthetic-corpus",
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    ids_path = tmp_path / manifest["types"]["table"]["ids_path"]
    ids_path.write_text(json.dumps(["positive"]), encoding="utf-8")

    with pytest.raises(ValueError, match="object count does not match the manifest"):
        RawEmbeddingANNIndices(
            store,
            tmp_path,
            corpus_sha256="synthetic-corpus",
        )


def test_ann_search_raises_ef_to_the_requested_k():
    class FakeIndex:
        def __init__(self):
            self.ef_values = []

        def set_ef(self, value):
            self.ef_values.append(value)

        def knn_query(self, _queries, k):
            return [list(range(k))], [[0.0] * k]

    object_ids = [f"t{index}" for index in range(10)]
    raw_index = FakeIndex()
    raw = object.__new__(RawEmbeddingANNIndices)
    raw.store = feature_store()
    raw.embedding_dim = 4
    raw.ef_search = 5
    raw.indices = {"table": raw_index}
    raw.object_ids = {"table": object_ids}

    raw.search_many(["q"], "table", 8)

    student_index = FakeIndex()
    student = object.__new__(StudentANNIndices)
    student.store = feature_store()
    student.device = torch.device("cpu")
    student.ef_search = 5
    student.indices = {"table": student_index}
    student.object_ids = {"table": object_ids}
    student._relation_queries = {}

    class IdentityRelations:
        @staticmethod
        def relation_query(embeddings, _source_type, _destination_type):
            return embeddings

    student.model = IdentityRelations()
    student.search_many(["q"], "table", 8)

    assert raw_index.ef_values == [8]
    assert student_index.ef_values == [8]


def test_student_path_dev_record_includes_reused_raw_embedding_baseline(tmp_path):
    store = feature_store()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps({"object_id": object_id}) + "\n"
            for object_id in ("positive", "negative", "evidence")
        ),
        encoding="utf-8",
    )
    example = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("evidence",)),
            TargetCandidate("negative", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=0,
        split="dev",
        positive_target_ids=("positive",),
    )
    args = argparse.Namespace(
        index_batch_size=2,
        hnsw_m=8,
        ef_construction=20,
        ef_search=20,
        direct_k=2,
        evidence_k=1,
        targets_per_evidence=2,
        evidence_types=["text"],
        rrf_k=60,
    )
    controller = train_stage1._EpochController(
        output=tmp_path / "student.pt",
        stage="student-path",
        aggregator=PathAggregator(),
        primary_metric="recall@10",
        min_delta=0.0,
        patience=1,
        store=store,
        device=torch.device("cpu"),
        dev_examples=[example],
        corpus_path=corpus,
        index_root=tmp_path / "dev_indices",
        raw_index_root=tmp_path / "raw_index",
        args=args,
    )
    record = {"epoch": 0, "training_state": "initial"}

    controller(0, StudentJoinabilityModel(4, 3), record)

    assert set(record["projection_drift"]) == {"table", "text", "image"}
    assert all(value == 0.0 for value in record["projection_drift"].values())
    assert "relation_drift" in record
    assert record["dev_retrieval"]["raw_embedding"]["queries"] == 1
    assert record["gate"]["improved"]
    assert controller.gate.best_epoch == 0
    assert controller.best_metrics["raw_embedding"] == record["dev_retrieval"][
        "raw_embedding"
    ]
    assert (tmp_path / "raw_index" / "manifest.json").is_file()


def test_online_retrieval_uses_configured_aggregation_and_unique_modalities():
    calls = Counter()

    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            del k
            destination_type = normalize_object_type(destination_type)
            calls[(source_id, destination_type)] += 1
            values = {
                ("q", "table"): [("t", 1.0)],
                ("q", "text"): [("e1", 2.0), ("e2", 1.0)],
                ("e1", "table"): [("t", 2.0)],
                ("e2", "table"): [("t", 1.0)],
            }
            return values.get((source_id, destination_type), [])

    results = retrieve_zero_one_hop(
        "q",
        StaticIndices(),
        direct_k=1,
        evidence_k=2,
        targets_per_evidence=1,
        evidence_types=("text", "text_asset"),
        evidence_aggregation="topk_mean",
        evidence_top_k=1,
    )

    result = results[0]
    assert result["evidence_score"] == pytest.approx(4.0)
    assert result["score"] == pytest.approx(2.0 / 61.0)
    assert calls[("q", "text")] == 1
    assert "direct_score" not in result
    assert "direct_rank" not in result
    assert "evidence_rank" not in result


def test_online_retrieval_derives_pools_and_result_limit_from_k():
    calls = []

    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            calls.append((source_id, destination_type, k))
            if source_id == "q" and destination_type == "table":
                return [(f"t{index}", float(10 - index)) for index in range(6)]
            return []

    results = retrieve_zero_one_hop(
        "q",
        StaticIndices(),
        k=2,
        gamma=3,
        gamma_evidence=1,
        evidence_types=(),
    )

    assert [result["target_id"] for result in results] == ["t0", "t1"]
    assert calls == [("q", "table", 6)]


def test_weighted_rrf_can_preserve_direct_ranking_when_evidence_is_noisy():
    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            del k
            values = {
                ("q", "table"): [("direct_first", 2.0), ("direct_second", 1.0)],
                ("q", "text"): [("e", 2.0)],
                ("e", "table"): [("direct_second", 2.0), ("direct_first", 1.0)],
            }
            return values.get((source_id, normalize_object_type(destination_type)), [])

    rankings = retrieve_zero_one_hop_detailed(
        "q",
        StaticIndices(),
        evidence_types=("text",),
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.0,
    )

    assert [row["target_id"] for row in rankings["fused"]] == [
        "direct_first",
        "direct_second",
    ]

    text_disabled = retrieve_zero_one_hop_detailed(
        "q",
        StaticIndices(),
        evidence_types=("text",),
        fusion_mode="weighted_rrf",
        evidence_weight=0.0,
        evidence_modality_weights={"text": 0.0},
    )
    assert not text_disabled["evidence"]
    assert [row["target_id"] for row in text_disabled["fused"]] == [
        "direct_first",
        "direct_second",
    ]


def test_online_retrieval_scores_all_paths_before_compacting_stage2_detail():
    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            del k
            values = {
                ("q", "table"): [("t1", 1.0), ("t2", 0.5)],
                ("q", "text"): [("e1", 3.0), ("e2", 2.0), ("e3", 1.0)],
                ("e1", "table"): [("t1", 1.0)],
                ("e2", "table"): [("t1", 1.0)],
                ("e3", "table"): [("t1", 1.0)],
            }
            return values.get((source_id, destination_type), [])

    results = retrieve_zero_one_hop(
        "q",
        StaticIndices(),
        direct_k=2,
        evidence_k=3,
        targets_per_evidence=1,
        result_k=2,
        evidence_types=("text",),
        path_result_k=1,
        evidence_path_k=2,
    )

    assert results[0]["target_id"] == "t1"
    assert results[0]["evidence_score"] == pytest.approx(
        torch.logsumexp(torch.tensor([4.0, 3.0, 2.0]), dim=0).item()
    )
    assert results[0]["paths"] == [
        {"kind": "direct"},
        {"kind": "evidence", "evidence_id": "e1", "path_score": 4.0},
        {"kind": "evidence", "evidence_id": "e2", "path_score": 3.0},
    ]
    assert "paths" not in results[1]


def test_online_retrieval_rrf_fuses_route_ranks_without_changing_route_scores():
    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            del k
            values = {
                ("q", "table"): [("mixed", 0.9), ("direct", 0.8)],
                ("q", "text"): [("e", 0.0)],
                ("e", "table"): [("evidence", 0.8), ("mixed", 0.1)],
            }
            return values.get((source_id, destination_type), [])

    results = retrieve_zero_one_hop(
        "q",
        StaticIndices(),
        direct_k=2,
        evidence_k=1,
        targets_per_evidence=2,
        result_k=3,
        evidence_types=("text",),
        rrf_k=0,
    )

    assert [result["target_id"] for result in results] == ["mixed", "evidence", "direct"]
    by_target = {result["target_id"]: result for result in results}
    assert by_target["mixed"]["evidence_score"] == pytest.approx(0.1)
    assert by_target["mixed"]["score"] == pytest.approx(1.0 + 0.5)


def test_path_checkpoint_persists_online_aggregation_configuration(tmp_path):
    path = tmp_path / "student.pt"
    aggregator = PathAggregator(
        "fixed_power_mean",
        2,
        temperature=0.3,
        power=3.0,
        path_combination="min",
        threshold=0.25,
        target_temperature=0.1,
    )
    torch.save(
        checkpoint(
            StudentJoinabilityModel(input_dim=4, student_dim=3),
            "student-path",
            aggregator,
        ),
        path,
    )

    loaded = load_path_aggregator(path)
    assert loaded.evidence_aggregation == "fixed_power_mean"
    assert loaded.top_k == 2
    assert loaded.temperature == pytest.approx(0.3)
    assert loaded.power == pytest.approx(3.0)
    assert loaded.path_combination == "min"
    assert loaded.threshold == pytest.approx(0.25)
    assert loaded.target_temperature == pytest.approx(0.1)


def test_legacy_path_checkpoint_defaults_target_temperature_to_one(tmp_path):
    path = tmp_path / "legacy_student.pt"
    payload = checkpoint(
        StudentJoinabilityModel(input_dim=4, student_dim=3),
        "student-path",
        PathAggregator(),
    )
    payload["path_aggregation"].pop("evidence_target_temperature")
    torch.save(payload, path)

    assert load_path_aggregator(path).target_temperature == 1.0


def test_dataset_sampling_alpha_balances_or_preserves_natural_mass():
    examples = [
        EdgeExample(f"small_{index}", ("positive", "negative"), 0, dataset="2k")
        for index in range(2)
    ] + [
        EdgeExample(f"large_{index}", ("positive", "negative"), 0, dataset="20k")
        for index in range(8)
    ]

    balanced = sample_balanced_epoch(examples, random.Random(13), dataset_sampling_alpha=0.0)
    natural = sample_balanced_epoch(examples, random.Random(13), dataset_sampling_alpha=1.0)
    repeated = sample_balanced_epoch(examples, random.Random(13), dataset_sampling_alpha=0.0)

    assert Counter(example.dataset for example in balanced) == {"2k": 5, "20k": 5}
    assert Counter(example.dataset for example in natural) == {"2k": 2, "20k": 8}
    assert [example.query_id for example in balanced] == [example.query_id for example in repeated]


@pytest.mark.parametrize(
    ("dataset_sampling_alpha", "expected_ids"),
    [
        (0.0, ["a2", "b0", "a0", "a1", "b1", "b1"]),
        (0.5, ["b1", "a3", "a0", "a1", "b0", "a2"]),
        (1.0, ["a0", "b1", "a1", "a2", "a3", "b0"]),
    ],
)
def test_balanced_epoch_preserves_seeded_sampling_sequence(
    dataset_sampling_alpha, expected_ids
):
    examples = [
        EdgeExample(f"a{index}", ("positive", "negative"), 0, dataset="a")
        for index in range(4)
    ] + [
        EdgeExample(f"b{index}", ("positive", "negative"), 0, dataset="b")
        for index in range(2)
    ]

    sampled = sample_balanced_epoch(
        examples,
        random.Random(17),
        dataset_sampling_alpha=dataset_sampling_alpha,
    )

    assert [example.query_id for example in sampled] == expected_ids


def test_balanced_epoch_preserves_single_dataset_sampling_sequence():
    examples = [
        EdgeExample(f"a{index}", ("positive", "negative"), 0, dataset="a")
        for index in range(4)
    ]

    sampled = sample_balanced_epoch(
        examples,
        random.Random(17),
        dataset_sampling_alpha=0.5,
    )

    assert [example.query_id for example in sampled] == ["a0", "a2", "a1", "a3"]


def test_student_edge_distillation_reuses_cached_teacher_logits():
    store = feature_store()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    example = EdgeExample(
        "q",
        ("positive", "negative"),
        0,
        dataset="2k",
        teacher_logits=(3.0, -2.0),
    )

    history = train_student_edges(
        student_model,
        [example],
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
    )

    assert history[0]["dataset_samples"] == {"2k": 1}


def test_student_path_distillation_reuses_separate_cached_teacher_logits():
    store = feature_store()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    example = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("evidence",)),
            TargetCandidate("negative", ("evidence",)),
        ),
        0,
        1,
        dataset="2k",
        teacher_direct_logits=(3.0, -2.0),
        teacher_evidence_logits=(-1.0, 2.0),
    )

    history = train_student_paths(
        student_model,
        [example],
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        PathAggregator(),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.5,
    )

    assert history[0]["dataset_samples"] == {"2k": 1}
    assert history[0]["direct_distillation_loss"] > 0
    assert history[0]["evidence_distillation_loss"] > 0


def test_student_path_pairs_continuous_edges_without_extra_optimizer_updates(
    monkeypatch,
):
    store = feature_store()
    student_model = StudentJoinabilityModel(
        input_dim=4,
        student_dim=3,
        confidence_transform=True,
    )
    path_examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            0,
            0,
            dataset=dataset,
            teacher_direct_logits=(3.0, -2.0),
            teacher_evidence_logits=(2.0, -1.0),
        )
        for dataset in ("paths_a", "paths_b")
    ]
    edge_examples = [
        EdgeExample(
            "q",
            ("positive", "negative"),
            0,
            dataset="edges_a",
            source_type="table",
            destination_type="table",
            confirmed_labels=(1, 0),
            teacher_logits=(2.0, -1.0),
        ),
        EdgeExample(
            "evidence",
            ("positive", "negative", "q"),
            0,
            dataset="edges_b",
            source_type="text",
            destination_type="table",
            confirmed_labels=(1, 0, None),
            teacher_logits=(1.5, -0.5, -1.0),
        ),
    ]
    anchor_calls = 0
    original_anchor = stage1_training.student_anchor_loss

    def counted_anchor(model):
        nonlocal anchor_calls
        anchor_calls += 1
        return original_anchor(model)

    monkeypatch.setattr(stage1_training, "student_anchor_loss", counted_anchor)

    history = train_student_paths(
        student_model,
        path_examples,
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        PathAggregator(),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=2,
        seed=13,
        temperature=1.0,
        distillation_weight=0.3,
        anchor_weight=0.1,
        continuous_edge_examples=edge_examples,
        continuous_edge_weight=1.0,
        continuous_edge_bce_weight=1.0,
        continuous_edge_batch_size=2,
    )

    record = history[0]
    assert record["path_batches"] == 1
    assert record["continuous_edge_batches"] == 1
    assert record["optimizer_updates"] == 1
    assert record["additional_optimizer_updates"] == 0
    assert record["continuous_edge_examples_seen"] == 2
    assert record["continuous_edge_participation"]["positive"] == 2
    assert record["continuous_edge_participation"]["negative"] == 2
    assert record["continuous_edge_participation"]["unknown"] == 1
    assert set(record["continuous_edge_loss_by_relation"]) == {
        "table_to_table",
        "text_to_table",
    }
    assert anchor_calls == 1


def test_student_path_optimizer_update_budget_truncates_the_last_epoch():
    store = feature_store()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            0,
            0,
            dataset=dataset,
            teacher_direct_logits=(3.0, -2.0),
            teacher_evidence_logits=(2.0, -1.0),
        )
        for dataset in ("a", "b")
    ]

    history = train_student_paths(
        student_model,
        examples,
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        PathAggregator(),
        device=torch.device("cpu"),
        epochs=3,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.3,
        max_optimizer_updates=1,
    )

    assert len(history) == 1
    assert history[0]["optimizer_updates"] == 1
    assert history[0]["cumulative_optimizer_updates"] == 1
    assert history[0]["examples_seen"] == 1
    assert history[0]["optimizer_update_budget_exhausted"]


def test_hard_candidate_merge_excludes_gt_and_keeps_path_hard_evidence():
    original = TargetExample(
        "q",
        (
            TargetCandidate("direct_positive", ()),
            TargetCandidate("evidence_positive", ("positive_evidence",)),
            TargetCandidate("other_positive", ()),
            TargetCandidate("fallback", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=1,
        dataset="20k",
        split="train",
        positive_target_ids=("direct_positive", "evidence_positive", "other_positive"),
    )
    candidate_set = build_hard_candidate_set(
        original,
        ["other_positive", "hard_1", "hard_2"],
        [],
        [
            HardPath("e1", "hard_1", 5.0),
            HardPath("e2", "hard_1", 4.0),
            HardPath("e3", "hard_1", 3.0),
            HardPath("e2", "hard_1", 2.0),
        ],
        hard_targets_per_query=2,
    )
    target_example = candidate_set.target_example

    assert [candidate.target_id for candidate in target_example.candidates] == [
        "direct_positive",
        "evidence_positive",
        "other_positive",
        "hard_1",
        "hard_2",
    ]
    assert target_example.direct_positive_index == 0
    assert target_example.evidence_positive_index == 1
    assert target_example.candidates[3:] == (
        TargetCandidate("hard_1", ("e1", "e2", "e3")),
        TargetCandidate("hard_2", ()),
    )

    store = FeatureStore(
        {
            object_id: feature(object_id, object_type, 0.1)
            for object_id, object_type in (
                ("q", "table"),
                ("direct_positive", "table"),
                ("evidence_positive", "table"),
                ("other_positive", "table"),
                ("hard_1", "table"),
                ("hard_2", "table"),
                ("positive_evidence", "text"),
            )
        }
    )
    target_records, edge_records = hard_candidate_records([candidate_set], store)
    assert [
        candidate["target_id"] for candidate in target_records[0]["candidates"][:3]
    ] == ["direct_positive", "evidence_positive", "other_positive"]
    assert edge_records[0]["candidate_ids"][:3] == [
        "direct_positive",
        "evidence_positive",
        "other_positive",
    ]


def test_hard_negative_refresh_mines_four_independent_candidate_pools():
    class StaticIndices(_BatchedSearchMixin):
        def search(self, source_id, destination_type, k):
            values = {
                ("q", "table"): [
                    ("direct_positive", 10.0),
                    ("hard_target", 9.0),
                ],
                ("q", "text"): [
                    ("positive_evidence", 5.0),
                    ("evidence_only", 4.0),
                    ("lower_path_evidence", 3.0),
                ],
                ("positive_evidence", "table"): [
                    ("evidence_positive", 6.0),
                    ("path_target", 3.0),
                ],
                ("evidence_only", "table"): [("evidence_positive", 2.0)],
                ("lower_path_evidence", "table"): [("lower_path_target", 4.0)],
            }
            return values.get((source_id, destination_type), [])[:k]

    original = TargetExample(
        "q",
        (
            TargetCandidate("direct_positive", ()),
            TargetCandidate("evidence_positive", ("positive_evidence",)),
            TargetCandidate("fallback", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=1,
        dataset="20k",
        split="train",
        positive_target_ids=("direct_positive", "evidence_positive"),
    )

    mined = retrieve_hard_candidate_sets(
        [original],
        StaticIndices(),
        hard_targets_per_query=1,
        hard_evidence_per_type=1,
        hard_targets_per_positive_evidence=1,
        hard_paths_per_query=1,
        direct_k=2,
        evidence_k=3,
        targets_per_evidence=2,
        evidence_types=("text",),
    )[0]
    duplicate_modality = retrieve_hard_candidate_sets(
        [original],
        StaticIndices(),
        hard_targets_per_query=1,
        hard_evidence_per_type=1,
        hard_targets_per_positive_evidence=1,
        hard_paths_per_query=1,
        direct_k=2,
        evidence_k=3,
        targets_per_evidence=2,
        evidence_types=("text", "text"),
    )[0]

    assert mined.evidence_negative_ids == ("evidence_only",)
    assert mined.direct_target_negative_ids == ("hard_target",)
    assert mined.evidence_target_negative_ids == (
        ("positive_evidence", ("path_target",)),
    )
    assert duplicate_modality == mined
    assert mined.target_example.candidates == (
        TargetCandidate("direct_positive", ()),
        TargetCandidate("evidence_positive", ("positive_evidence",)),
        TargetCandidate("hard_target", ()),
        TargetCandidate("path_target", ("positive_evidence",)),
    )
    assert summarize_hard_candidate_sets([mined]) == {
        "direct_target_candidates": 1,
        "evidence_candidates": 1,
        "evidence_target_candidates": 1,
        "evidence_target_unique_targets": 1,
        "direct_evidence_target_overlap": 0,
        "evidence_target_path_overlap": 1,
        "path_target_candidates": 1,
        "fallback_target_candidates": 0,
        "direct_path_overlap": 0,
        "base_negative_overlap": 0,
        "merged_negative_targets": 2,
        "queries": 1,
        "raw_target_candidates": 3,
        "target_candidates_removed_by_dedup": 1,
        "target_pool_dedup_rate": pytest.approx(1 / 3),
        "merged_targets_overlapping_base_rate": 0.0,
    }

    store = FeatureStore(
        {
            object_id: feature(object_id, object_type, value)
            for object_id, object_type, value in (
                ("q", "table", 0.1),
                ("direct_positive", "table", 0.2),
                ("evidence_positive", "table", 0.3),
                ("hard_target", "table", 0.7),
                ("path_target", "table", 0.8),
                ("positive_evidence", "text", 0.4),
                ("evidence_only", "text", 0.6),
            )
        }
    )
    _target_records, edge_records = score_hard_candidate_sets(
        [mined],
        teacher(),
        store,
        PathAggregator(),
        device=torch.device("cpu"),
        batch_size=1,
    )
    query_evidence_edge = next(
        record
        for record in edge_records
        if record["source_type"] == "table" and record["destination_type"] == "text"
    )
    assert query_evidence_edge["candidate_ids"] == [
        "positive_evidence",
        "evidence_only",
    ]
    direct_edge = next(
        record
        for record in edge_records
        if record["source_type"] == "table" and record["destination_type"] == "table"
    )
    assert direct_edge["candidate_ids"] == [
        "direct_positive",
        "evidence_positive",
        "hard_target",
    ]
    evidence_target_edge = next(
        record
        for record in edge_records
        if record["source_type"] == "text" and record["destination_type"] == "table"
    )
    assert evidence_target_edge["candidate_ids"] == [
        "evidence_positive",
        "path_target",
    ]


def test_mine_only_requires_edge_output_before_loading_inputs():
    with pytest.raises(
        ValueError, match="--output-edge-lists is required with --mine-only"
    ):
        hard_negative_refresh.run(
            argparse.Namespace(mine_only=True, output_edge_lists=None)
        )


def test_hard_negative_refresh_caches_teacher_target_and_edge_scores(tmp_path):
    features = {
        "q": feature("q", "table", 0.1),
        "positive": feature("positive", "table", 0.2),
        "hard": feature("hard", "table", 0.8),
        "evidence": feature("evidence", "text", 0.4),
    }
    store = FeatureStore(features)
    original = TargetExample(
        "q",
        (TargetCandidate("positive", ()), TargetCandidate("hard", ())),
        direct_positive_index=0,
        evidence_positive_index=0,
        dataset="2k",
        split="train",
    )
    candidate_set = build_hard_candidate_set(
        original,
        ["hard"],
        [],
        [HardPath("evidence", "hard", 3.0)],
        hard_targets_per_query=1,
    )

    pending_targets, pending_edges = hard_candidate_records([candidate_set], store)
    assert "teacher_direct_logits" not in pending_targets[0]
    assert "teacher_logits" not in pending_edges[0]
    assert pending_targets[0]["candidates"][1] == {
        "target_id": "hard",
        "evidence_ids": ["evidence"],
    }

    target_records, edge_records = score_hard_candidate_sets(
        [candidate_set],
        teacher(),
        store,
        PathAggregator(),
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert len(target_records[0]["teacher_direct_logits"]) == 2
    assert len(target_records[0]["teacher_evidence_logits"]) == 2
    assert len(edge_records[0]["teacher_logits"]) == 2
    assert edge_records[0]["destination_type"] == "table"
    assert set(target_records[0]) == {
        "query_id",
        "direct_positive_target_id",
        "evidence_positive_target_id",
        "positive_target_ids",
        "candidates",
        "teacher_direct_logits",
        "teacher_evidence_logits",
        "teacher_score_config",
        "dataset",
        "split",
    }
    assert target_records[0]["candidates"][1] == {
        "target_id": "hard",
        "evidence_ids": ["evidence"],
    }
    assert "student_logits" not in edge_records[0]

    ensemble_targets, ensemble_edges = score_hard_candidate_sets(
        [candidate_set],
        teacher(),
        store,
        PathAggregator(),
        device=torch.device("cpu"),
        batch_size=1,
        ensemble_alpha=0.7,
    )
    assert ensemble_targets[0]["teacher_logit_mode"] == "ensemble"
    assert ensemble_targets[0]["teacher_ensemble_alpha"] == pytest.approx(0.7)
    assert ensemble_edges[0]["teacher_logit_mode"] == "ensemble"
    assert ensemble_edges[0]["teacher_ensemble_alpha"] == pytest.approx(0.7)
    assert ensemble_targets[0]["teacher_direct_logits"] != pytest.approx(
        target_records[0]["teacher_direct_logits"]
    )
    assert ensemble_edges[0]["teacher_logits"] != pytest.approx(
        edge_records[0]["teacher_logits"]
    )

    target_path = tmp_path / "hard_targets.jsonl"
    edge_path = tmp_path / "hard_edges.jsonl"
    target_path.write_text(json.dumps(target_records[0]) + "\n", encoding="utf-8")
    edge_path.write_text(json.dumps(edge_records[0]) + "\n", encoding="utf-8")
    metadata = {
        "teacher_checkpoint_sha256": "synthetic",
        "evidence_aggregation": "logsumexp",
        "evidence_top_k": 4,
    }
    for path in (target_path, edge_path):
        path.with_suffix(path.suffix + ".metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
    loaded_target = load_target_examples(target_path)[0]
    loaded_edge = load_edge_examples(edge_path)[0]

    assert loaded_target.teacher_direct_logits == pytest.approx(
        target_records[0]["teacher_direct_logits"]
    )
    assert loaded_target.teacher_evidence_logits == pytest.approx(
        target_records[0]["teacher_evidence_logits"]
    )
    assert loaded_target.teacher_score_config.evidence_aggregation == "logsumexp"
    assert loaded_target.teacher_checkpoint_sha256 == "synthetic"
    assert loaded_edge.teacher_logits == pytest.approx(edge_records[0]["teacher_logits"])
    assert loaded_edge.teacher_checkpoint_sha256 == "synthetic"


def test_hard_target_loader_preserves_ensemble_provenance(tmp_path):
    path = tmp_path / "hard_targets.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q",
                "direct_positive_target_id": "positive",
                "evidence_positive_target_id": "positive",
                "candidates": [
                    {"target_id": "positive", "evidence_ids": []},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "teacher_direct_logits": [1.0, 0.0],
                "teacher_evidence_logits": [1.0, 0.0],
                "dataset": "data",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    path.with_suffix(".jsonl.metadata.json").write_text(
        json.dumps(
            {
                "teacher_checkpoint_sha256": "teacher-sha",
                "evidence_aggregation": "logsumexp",
                "evidence_top_k": 4,
                "teacher_target_logit_mode": "ensemble",
                "teacher_target_ensemble_alpha": 0.7,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_target_examples(path)[0]

    assert loaded.teacher_logit_mode == "ensemble"
    assert loaded.teacher_ensemble_alpha == pytest.approx(0.7)


def test_hard_negative_refresh_scores_pending_candidates_without_remining(
    tmp_path, monkeypatch
):
    features = {
        "q": feature("q", "table", 0.1),
        "positive": feature("positive", "table", 0.2),
        "hard": feature("hard", "table", 0.8),
        "positive_evidence": feature("positive_evidence", "text", 0.4),
        "hard_evidence": feature("hard_evidence", "text", 0.7),
    }
    store = FeatureStore(features)
    original = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("positive_evidence",)),
            TargetCandidate("hard", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=0,
        dataset="2k",
        split="train",
    )
    candidate_set = build_hard_candidate_set(
        original,
        ["hard"],
        ["hard_evidence"],
        [HardPath("hard_evidence", "hard", 3.0)],
        hard_targets_per_query=1,
    )
    pending_target_records, pending_edge_records = hard_candidate_records(
        [candidate_set], store
    )

    features_path = tmp_path / "features.pt"
    torch.save(
        {
            "objects": {
                object_id: {
                    "object_type": value.object_type,
                    "embedding": value.embedding,
                    "hidden_states": value.hidden_states,
                    "token_groups": value.token_groups,
                }
                for object_id, value in features.items()
            }
        },
        features_path,
    )
    student_path = tmp_path / "student.pt"
    teacher_path = tmp_path / "teacher.pt"
    teacher_model = teacher()
    torch.save(
        checkpoint(
            StudentJoinabilityModel(input_dim=4, student_dim=3),
            "student-path",
            PathAggregator(),
        ),
        student_path,
    )
    torch.save(
        checkpoint(teacher_model, "teacher-path", PathAggregator()), teacher_path
    )
    training_aggregator = PathAggregator(
        "fixed_power_mean",
        4,
        power=2.0,
        path_combination="min",
        target_temperature=0.1,
    )
    expected_targets, expected_edges = score_hard_candidate_sets(
        [candidate_set],
        teacher_model,
        store,
        training_aggregator,
        device=torch.device("cpu"),
        batch_size=1,
        teacher_score_space="confidence",
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("{}\n", encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_manifest = index_dir / "manifest.json"
    index_manifest.write_text("{}\n", encoding="utf-8")

    pending_targets = tmp_path / "hard_targets.pending.jsonl"
    pending_edges = tmp_path / "hard_edges.pending.jsonl"
    pending_targets.write_text(
        "".join(json.dumps(record) + "\n" for record in pending_target_records),
        encoding="utf-8",
    )
    pending_edges.write_text(
        "".join(json.dumps(record) + "\n" for record in pending_edge_records),
        encoding="utf-8",
    )
    metadata = {
        "mining_round": 1,
        "student_checkpoint_sha256": checkpoint_fingerprint(student_path),
        "corpus_sha256": checkpoint_fingerprint(corpus_path),
        "index_manifest_sha256": checkpoint_fingerprint(index_manifest),
        "evidence_aggregation": "logsumexp",
        "evidence_top_k": 4,
        "teacher_target_channels": ["direct", "evidence"],
        "hard_negative_mining": {
            "hard_evidence": "query_to_evidence_ann",
            "hard_target": "query_to_target_ann",
            "path_hard": "raw_query_evidence_target_path_score",
        },
        "hard_targets_per_query": 1,
        "hard_evidence_per_type": 1,
        "hard_paths_per_query": 1,
        "direct_k": 2,
        "evidence_k": 2,
        "targets_per_evidence": 2,
        "evidence_types": ["text"],
        "teacher_scoring": "pending",
    }
    for path in (pending_targets, pending_edges):
        path.with_suffix(path.suffix + ".metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    monkeypatch.setattr(
        hard_negative_refresh,
        "retrieve_hard_candidate_sets",
        lambda *_args, **_kwargs: pytest.fail("pending scoring repeated ANN mining"),
    )
    output_targets = tmp_path / "hard_targets.jsonl"
    output_edges = tmp_path / "hard_edges.jsonl"
    hard_negative_refresh.run(
        argparse.Namespace(
            features=str(features_path),
            teacher_checkpoint=str(teacher_path),
            student_checkpoint=str(student_path),
            index_dir=str(index_dir),
            corpus=str(corpus_path),
            target_lists=[str(tmp_path / "unused.jsonl")],
            output_target_lists=str(output_targets),
            output_edge_lists=str(output_edges),
            pending_target_lists=str(pending_targets),
            pending_edge_lists=str(pending_edges),
            split="train",
            device="cpu",
            feature_cache_size=128,
            teacher_batch_size=1,
            teacher_score_space="confidence",
            training_student_score_space="confidence",
            mine_only=False,
            mining_round=1,
            hard_targets_per_query=1,
            hard_evidence_per_type=1,
            hard_paths_per_query=1,
            direct_k=2,
            evidence_k=2,
            targets_per_evidence=2,
            evidence_types=["text"],
            evidence_aggregation="fixed_power_mean",
            evidence_top_k=4,
            evidence_temperature=1.0,
            evidence_power=2.0,
            path_combination="min",
            evidence_threshold=0.0,
            evidence_target_temperature=0.1,
        )
    )

    target_record = json.loads(output_targets.read_text(encoding="utf-8"))
    edge_records = [
        json.loads(line)
        for line in output_edges.read_text(encoding="utf-8").splitlines()
    ]
    assert target_record["candidates"] == pending_target_records[0]["candidates"]
    assert len(target_record["teacher_direct_logits"]) == 2
    assert len(target_record["teacher_evidence_logits"]) == 2
    assert target_record["teacher_direct_logits"] == pytest.approx(
        expected_targets[0]["teacher_direct_logits"]
    )
    assert target_record["teacher_evidence_logits"] == pytest.approx(
        expected_targets[0]["teacher_evidence_logits"]
    )
    assert len(edge_records) == 3
    assert all(len(record["teacher_logits"]) == 2 for record in edge_records)
    for record, expected in zip(edge_records, expected_edges):
        assert record["teacher_logits"] == pytest.approx(expected["teacher_logits"])
    query_evidence_edge = next(
        record for record in edge_records if record["destination_type"] == "text"
    )
    assert query_evidence_edge["candidate_ids"] == [
        "positive_evidence",
        "hard_evidence",
    ]
    output_metadata = json.loads(
        output_targets.with_suffix(".jsonl.metadata.json").read_text(encoding="utf-8")
    )
    assert output_metadata["teacher_scoring"] == "complete"
    assert output_metadata["teacher_checkpoint_sha256"] == checkpoint_fingerprint(
        teacher_path
    )
    assert output_metadata["evidence_aggregation"] == "logsumexp"
    assert output_metadata.get("path_combination", "sum") == "sum"
    assert output_metadata["mining_selection_config"] == {
        "student_score_space": "raw_logit",
        "evidence_aggregation": "logsumexp",
        "evidence_top_k": 4,
        "evidence_temperature": 1.0,
        "evidence_power": 2.0,
        "path_combination": "sum",
        "evidence_threshold": 0.0,
        "evidence_target_temperature": 1.0,
    }
    assert output_metadata["training_score_config"] == {
        **training_aggregator.config(),
        "teacher_score_space": "confidence",
        "student_score_space": "confidence",
        "teacher_ensemble_alpha": None,
    }
    loaded_target = load_target_examples(output_targets)[0]
    assert loaded_target.teacher_score_config.evidence_aggregation == (
        "fixed_power_mean"
    )
    assert loaded_target.teacher_score_config.path_combination == "min"
    assert loaded_target.teacher_score_config.evidence_target_temperature == 0.1
    assert loaded_target.teacher_logit_mode == "teacher_confidence"
    assert train_stage1._validate_hard_provenance(
        [output_targets, output_edges],
        teacher_checkpoint=teacher_path,
        source_student_checkpoint=student_path,
        aggregator=training_aggregator,
        student_score_space="confidence",
        teacher_score_space="confidence",
    ) == 1
    with pytest.raises(ValueError, match="training aggregation"):
        train_stage1._validate_hard_provenance(
            [output_targets],
            teacher_checkpoint=teacher_path,
            source_student_checkpoint=student_path,
            aggregator=PathAggregator(),
            student_score_space="confidence",
            teacher_score_space="confidence",
        )


def test_hard_negative_refresh_rescores_cross_modal_edge_lists():
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.2),
            "hard": feature("hard", "table", 0.8),
            "positive_text": feature("positive_text", "text", 0.3),
            "hard_text": feature("hard_text", "text", 0.7),
            "positive_image": feature("positive_image", "image", 0.4),
            "hard_image": feature("hard_image", "image", 0.6),
        }
    )
    original = TargetExample(
        "q",
        (
            TargetCandidate("positive", ("positive_text", "positive_image")),
            TargetCandidate("hard", ()),
        ),
        direct_positive_index=0,
        evidence_positive_index=0,
        dataset="2k",
        split="train",
    )
    candidate_set = build_hard_candidate_set(
        original,
        ["hard"],
        ["hard_text", "hard_image"],
        [
            HardPath("hard_text", "hard", 4.0),
            HardPath("hard_image", "hard", 3.0),
        ],
        hard_targets_per_query=1,
    )

    _target_records, edge_records = score_hard_candidate_sets(
        [candidate_set],
        teacher(),
        store,
        PathAggregator(),
        device=torch.device("cpu"),
        batch_size=1,
    )

    edges = {
        (record["source_type"], record["destination_type"]): record
        for record in edge_records
    }
    assert set(edges) == {
        ("table", "table"),
        ("table", "text"),
        ("text", "table"),
        ("table", "image"),
        ("image", "table"),
    }
    assert edges[("table", "text")]["candidate_ids"] == [
        "positive_text",
        "hard_text",
    ]
    assert edges[("table", "image")]["candidate_ids"] == [
        "positive_image",
        "hard_image",
    ]
    assert edges[("text", "table")]["candidate_ids"] == ["positive", "hard"]
    assert edges[("image", "table")]["candidate_ids"] == ["positive", "hard"]
    assert all(len(record["teacher_logits"]) == 2 for record in edge_records)


def test_student_pair_scoring_projects_each_unique_object_once():
    store = feature_store()
    model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    sources = [store.get("q"), store.get("q"), store.get("evidence")]
    destinations = [
        store.get("positive"),
        store.get("negative"),
        store.get("positive"),
    ]
    expected = torch.stack(
        [
            model.score_embeddings(
                source.embedding,
                source.object_type,
                destination.embedding,
                destination.object_type,
            )
            for source, destination in zip(sources, destinations)
        ]
    )
    projected_batch_sizes = {}
    hooks = [
        projection.register_forward_hook(
            lambda _module, inputs, _output, object_type=object_type: (
                projected_batch_sizes.setdefault(object_type, []).append(
                    inputs[0].shape[0]
                )
            )
        )
        for object_type, projection in model.projections.items()
    ]

    actual = model.score_pairs(sources, destinations)
    for hook in hooks:
        hook.remove()

    assert torch.allclose(actual, expected)
    assert projected_batch_sizes == {"table": [3], "text": [1]}


def test_target_scoring_batches_path_aggregation_once():
    store = feature_store()
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        ),
        TargetExample(
            "q",
            (
                TargetCandidate("negative", ()),
                TargetCandidate("positive", ("evidence",)),
            ),
            direct_positive_index=1,
            evidence_positive_index=1,
        ),
    ]
    aggregator = PathAggregator()
    calls = []
    model = StudentJoinabilityModel(4, 3)
    projection_calls = Counter()
    hook = aggregator.register_forward_hook(
        lambda _module, inputs, _output: calls.append(inputs[0].shape)
    )
    projection_hooks = [
        projection.register_forward_hook(
            lambda _module, _inputs, _output, object_type=object_type: projection_calls.update(
                [object_type]
            )
        )
        for object_type, projection in model.projections.items()
    ]

    scores = score_target_batch(
        model,
        examples,
        store,
        torch.device("cpu"),
        aggregator,
    )
    hook.remove()
    for projection_hook in projection_hooks:
        projection_hook.remove()

    assert scores.evidence.logits.shape == (2, 2)
    assert calls == [torch.Size([1, 4, 1])]
    assert projection_calls == {"table": 1, "text": 1}


def test_student_training_rejects_missing_teacher_logits():
    store = feature_store()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    with pytest.raises(ValueError, match="requires cached Teacher logits"):
        train_student_edges(
            student_model,
            [EdgeExample("q", ("negative", "positive"), 1)],
            store,
            torch.optim.AdamW(student_model.parameters(), lr=1e-3),
            device=torch.device("cpu"),
            epochs=1,
            batch_size=1,
            seed=13,
            temperature=1.0,
        )


def test_zero_kd_student_training_needs_no_teacher_logits():
    store = feature_store()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    history = train_student_edges(
        student_model,
        [EdgeExample("q", ("positive", "negative"), 0)],
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
        distillation_weight=0.0,
        in_batch_negatives=True,
    )

    assert history[0]["distillation_loss"] == pytest.approx(0.0)
    assert history[0]["supervised_loss"] > 0


def test_teacher_logit_sidecars_support_teacher_free_student_training(tmp_path):
    store = feature_store()
    teacher_model = teacher()
    edge_examples = [EdgeExample("q", ("positive", "negative"), 0)]
    cached_edges, edge_path = score_and_cache_teacher_logits(
        edge_examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
    )
    loaded_edges, loaded_path, hit = load_teacher_logits(
        edge_examples, tmp_path, "teacher-sha"
    )

    assert hit
    assert loaded_path == edge_path
    assert loaded_edges == cached_edges
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    train_student_edges(
        student_model,
        loaded_edges,
        store,
        torch.optim.AdamW(student_model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        epochs=1,
        batch_size=1,
        seed=13,
        temperature=1.0,
    )

    target_examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        )
    ]
    aggregator = PathAggregator()
    cached_targets, _path = score_and_cache_teacher_logits(
        target_examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=aggregator,
    )
    loaded_targets, _path, hit = load_teacher_logits(
        target_examples, tmp_path, "teacher-sha", aggregator
    )

    assert hit
    assert loaded_targets == cached_targets
    assert not has_teacher_logits(
        loaded_targets,
        "teacher-sha",
        PathAggregator("topk_sum", 1),
    )


def test_teacher_logit_sidecars_separate_aggregation_parameters(tmp_path):
    store = feature_store()
    teacher_model = teacher()
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ("evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        )
    ]
    low_temperature = PathAggregator(
        "softmax_weighted_mean", temperature=0.1
    )
    high_temperature = PathAggregator(
        "softmax_weighted_mean", temperature=1.0
    )

    cached, low_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=low_temperature,
    )
    _other, high_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=high_temperature,
    )

    assert low_path != high_path
    assert has_teacher_logits(cached, "teacher-sha", low_temperature)
    assert not has_teacher_logits(cached, "teacher-sha", high_temperature)


def test_teacher_logit_sidecars_cache_path_ensemble_edges_separately(tmp_path):
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.25),
            "negative": feature("negative", "table", 0.9),
        }
    )
    teacher_model = teacher()
    examples = [EdgeExample("q", ("positive", "negative"), 0)]
    ordinary, ordinary_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
    )
    ensemble, ensemble_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        ensemble_alpha=0.0,
    )
    loaded, loaded_path, hit = load_teacher_logits(
        examples,
        tmp_path,
        "teacher-sha",
        ensemble_alpha=0.0,
    )

    assert ordinary_path != ensemble_path
    assert ordinary[0].teacher_logit_mode == "teacher"
    assert ensemble[0].teacher_logit_mode == "ensemble"
    assert ensemble[0].teacher_ensemble_alpha == 0.0
    assert has_teacher_logits(loaded, "teacher-sha", ensemble_alpha=0.0)
    assert not has_teacher_logits(loaded, "teacher-sha")
    assert hit
    assert loaded_path == ensemble_path
    raw = [
        torch.nn.functional.cosine_similarity(
            store.embedding_features("q").embedding,
            store.embedding_features(object_id).embedding,
            dim=0,
        ).item()
        for object_id in ("positive", "negative")
    ]
    assert loaded[0].teacher_logits == pytest.approx(z_scores(raw))


def test_teacher_logit_sidecars_cache_path_ensemble_targets_separately(tmp_path):
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.25),
            "negative": feature("negative", "table", 0.9),
            "positive_evidence": feature("positive_evidence", "text", 0.3),
            "negative_evidence": feature("negative_evidence", "text", 0.75),
        }
    )
    teacher_model = teacher()
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("positive_evidence",)),
                TargetCandidate("negative", ("negative_evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        )
    ]
    aggregator = PathAggregator()
    ordinary, ordinary_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=aggregator,
    )
    ensemble, ensemble_path = score_and_cache_teacher_logits(
        examples,
        teacher_model,
        store,
        tmp_path,
        "teacher-sha",
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=aggregator,
        ensemble_alpha=0.0,
    )
    loaded, loaded_path, hit = load_teacher_logits(
        examples,
        tmp_path,
        "teacher-sha",
        aggregator,
        ensemble_alpha=0.0,
    )

    assert ordinary_path != ensemble_path
    assert "ensemble-edge-v2" in ensemble_path.name
    assert ordinary[0].teacher_logit_mode == "teacher"
    assert ensemble[0].teacher_logit_mode == "ensemble"
    assert ensemble[0].teacher_ensemble_alpha == 0.0
    assert has_teacher_logits(loaded, "teacher-sha", aggregator, ensemble_alpha=0.0)
    assert not has_teacher_logits(loaded, "teacher-sha", aggregator)
    assert hit
    assert loaded_path == ensemble_path
    raw_direct = [
        torch.nn.functional.cosine_similarity(
            store.embedding_features("q").embedding,
            store.embedding_features(object_id).embedding,
            dim=0,
        ).item()
        for object_id in ("positive", "negative")
    ]
    assert loaded[0].teacher_direct_logits == pytest.approx(z_scores(raw_direct))
    raw_evidence_target = [
        torch.nn.functional.cosine_similarity(
            store.embedding_features(evidence_id).embedding,
            store.embedding_features(target_id).embedding,
            dim=0,
        ).item()
        for evidence_id, target_id in (
            ("positive_evidence", "positive"),
            ("negative_evidence", "negative"),
        )
    ]
    assert loaded[0].teacher_evidence_logits == pytest.approx(
        z_scores(raw_evidence_target)
    )


def test_pure_cosine_kd_cache_does_not_need_a_teacher(tmp_path):
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.25),
            "negative": feature("negative", "table", 0.9),
            "positive_evidence": feature("positive_evidence", "text", 0.3),
            "negative_evidence": feature("negative_evidence", "text", 0.75),
        }
    )
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("positive_evidence",)),
                TargetCandidate("negative", ("negative_evidence",)),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        )
    ]
    aggregator = PathAggregator()

    cached, path = score_and_cache_cosine_logits(
        examples,
        store,
        tmp_path,
        device=torch.device("cpu"),
        batch_size=1,
        aggregator=aggregator,
    )
    loaded, loaded_path, hit = load_teacher_logits(
        examples,
        tmp_path,
        FROZEN_COSINE_TARGET_SHA256,
        aggregator,
        ensemble_alpha=0.0,
    )

    assert hit
    assert loaded_path == path
    assert loaded == cached
    assert path.parent.name == FROZEN_COSINE_TARGET_SHA256
    assert loaded[0].teacher_logit_mode == "ensemble"
    assert loaded[0].teacher_ensemble_alpha == 0.0


def test_pure_cosine_kd_cache_pads_variable_candidate_counts(tmp_path):
    store = FeatureStore(
        {
            "q": feature("q", "table", 0.1),
            "positive": feature("positive", "table", 0.25),
            "negative": feature("negative", "table", 0.9),
            "evidence": feature("evidence", "text", 0.3),
        }
    )
    examples = [
        TargetExample(
            "q",
            (
                TargetCandidate("positive", ("evidence",)),
                TargetCandidate("negative", ()),
            ),
            direct_positive_index=0,
            evidence_positive_index=0,
        ),
        TargetExample(
            "q",
            (TargetCandidate("positive", ("evidence",)),),
            direct_positive_index=0,
            evidence_positive_index=0,
        ),
    ]
    aggregator = PathAggregator()

    cached, path = score_and_cache_cosine_logits(
        examples,
        store,
        tmp_path,
        device=torch.device("cpu"),
        batch_size=2,
        aggregator=aggregator,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    loaded, _path, hit = load_teacher_logits(
        examples,
        tmp_path,
        FROZEN_COSINE_TARGET_SHA256,
        aggregator,
        ensemble_alpha=0.0,
    )

    assert hit
    assert payload["teacher_direct_logits"].shape == (2, 2)
    assert payload["teacher_evidence_logits"].shape == (2, 2)
    assert payload["target_source"] == "frozen_embedding_cosine"
    assert loaded == cached
    assert len(loaded[1].teacher_direct_logits) == 1
    assert len(loaded[1].teacher_evidence_logits) == 1


def test_kd_target_teacher_alpha_is_the_canonical_cli_name(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_stage1.py",
            "student-path",
            "--features",
            "features",
            "--base-data",
            "train.jsonl",
            "--dev-data",
            "dev.jsonl",
            "--output",
            "student.pt",
            "--kd-target-teacher-alpha",
            "0.3",
        ],
    )

    args = train_stage1.parse_args()

    assert args.kd_target_teacher_alpha == pytest.approx(0.3)
    assert args.teacher_ensemble_alpha is None


def test_student_edge_entrypoint_reuses_ensemble_logits_without_teacher_hidden_tier(
    tmp_path, monkeypatch
):
    features_path = tmp_path / "features.pt"
    store = feature_store()
    features = {
        object_id: {
            "object_type": value.object_type,
            "embedding": value.embedding,
            "hidden_states": value.hidden_states,
            "token_groups": value.token_groups,
        }
        for object_id in store.object_ids()
        for value in [store.get(object_id)]
    }
    torch.save({"objects": features}, features_path)
    data_path = tmp_path / "edges.jsonl"
    data_path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": "q",
                    "candidate_ids": ["positive", "negative"],
                    "positive_id": "positive",
                    "dataset": "2k",
                    "split": split,
                }
            )
            + "\n"
            for split in ("train", "dev")
        ),
        encoding="utf-8",
    )
    teacher_path = tmp_path / "teacher.pt"
    torch.save(
        checkpoint(teacher(), "teacher-path", PathAggregator()), teacher_path
    )
    cache_dir = tmp_path / "teacher_logits"

    def arguments(output: Path):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_stage1.py",
                "student-edge",
                "--features",
                str(features_path),
                "--base-data",
                str(data_path),
                "--dev-data",
                str(data_path),
                "--teacher-checkpoint",
                str(teacher_path),
                "--teacher-logit-cache",
                str(cache_dir),
                "--kd-target-teacher-alpha",
                "1.0",
                "--output",
                str(output),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--batch-size",
                "1",
            ],
        )
        return train_stage1.parse_args()

    first = train_stage1.run(arguments(tmp_path / "student_first.pt"))
    assert first["teacher_cache_generated"]

    for payload in features.values():
        payload.pop("hidden_states")
        payload.pop("token_groups")
    torch.save({"objects": features}, features_path)
    monkeypatch.setattr(
        train_stage1,
        "load_teacher",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Teacher was loaded after the logit cache was complete")
        ),
    )

    second = train_stage1.run(arguments(tmp_path / "student_second.pt"))

    assert not second["teacher_cache_generated"]
    assert second["teacher_logit_cache_hits"] == 2


def test_student_path_entrypoint_uses_independent_continuous_edge_teacher_cache(
    tmp_path,
):
    features_path = tmp_path / "features.pt"
    store = feature_store()
    torch.save(
        {
            "objects": {
                object_id: {
                    "object_type": value.object_type,
                    "embedding": value.embedding,
                    "hidden_states": value.hidden_states,
                    "token_groups": value.token_groups,
                }
                for object_id in store.object_ids()
                for value in [store.get(object_id)]
            }
        },
        features_path,
    )
    target_path = tmp_path / "targets.jsonl"
    target_path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": "q",
                    "direct_positive_target_id": "positive",
                    "evidence_positive_target_id": "positive",
                    "candidates": [
                        {
                            "target_id": "positive",
                            "evidence_ids": ["evidence"],
                        },
                        {
                            "target_id": "negative",
                            "evidence_ids": ["evidence"],
                        },
                    ],
                    "dataset": "tiny",
                    "split": split,
                }
            )
            + "\n"
            for split in ("train", "dev")
        ),
        encoding="utf-8",
    )
    edge_path = tmp_path / "edges.jsonl"
    edge_path.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": "q",
                    "positive_id": "positive",
                    "candidate_ids": ["positive", "negative"],
                    "confirmed_labels": [1, 0],
                    "source_type": "table",
                    "destination_type": "table",
                    "dataset": "tiny_edges",
                    "split": split,
                }
            )
            + "\n"
            for split in ("train", "dev")
        ),
        encoding="utf-8",
    )
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(
            json.dumps({"object_id": object_id}) + "\n"
            for object_id in ("positive", "negative", "evidence")
        ),
        encoding="utf-8",
    )
    edge_teacher_path = tmp_path / "teacher_edge.pt"
    path_teacher_path = tmp_path / "teacher_path.pt"
    torch.save(checkpoint(teacher(), "teacher-edge"), edge_teacher_path)
    torch.save(
        checkpoint(teacher(), "teacher-path", PathAggregator()),
        path_teacher_path,
    )
    path_cache = tmp_path / "path_teacher_logits"
    edge_cache = tmp_path / "edge_teacher_logits"
    output = tmp_path / "student_path.pt"
    args = train_stage1._argument_parser().parse_args(
        [
            "student-path",
            "--features",
            str(features_path),
            "--base-data",
            str(target_path),
            "--dev-data",
            str(target_path),
            "--corpus",
            str(corpus_path),
            "--teacher-checkpoint",
            str(path_teacher_path),
            "--teacher-logit-cache",
            str(path_cache),
            "--continuous-edge-data",
            str(edge_path),
            "--continuous-edge-dev-data",
            str(edge_path),
            "--continuous-edge-teacher-checkpoint",
            str(edge_teacher_path),
            "--continuous-edge-teacher-logit-cache",
            str(edge_cache),
            "--student-confidence-transform",
            "--positive-loss-mode",
            "mean_log_probability",
            "--edge-bce-weight",
            "1",
            "--student-dim",
            "3",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--initialize-only",
            "--primary-metric",
            "recall@1",
            "--recall-ks",
            "1",
            "--output",
            str(output),
        ]
    )

    summary = train_stage1.run(args)
    history = json.loads(
        output.with_suffix(".pt.history.json").read_text(encoding="utf-8")
    )
    selection = json.loads(
        output.with_suffix(".pt.selection.json").read_text(encoding="utf-8")
    )
    continuous = history["continuous_edge_training"]

    assert summary["continuous_edge_training"]["enabled"]
    assert continuous == selection["continuous_edge_training"]
    assert continuous["train_examples"] == 1
    assert continuous["dev_examples"] == 1
    assert continuous["train_participation"]["positive"] == 1
    assert continuous["train_participation"]["negative"] == 1
    assert continuous["teacher_checkpoint_sha256"] == checkpoint_fingerprint(
        edge_teacher_path
    )
    assert continuous["teacher_target"] == "teacher"
    assert continuous["teacher_cache_generated"]
    assert len(continuous["teacher_logit_caches"]) == 2
    assert all(str(edge_cache.resolve()) in path for path in continuous["teacher_logit_caches"])
    assert all(str(path_cache.resolve()) not in path for path in continuous["teacher_logit_caches"])
    assert history["preloaded_embeddings"] == 4
    assert history["positive_loss_mode"] == "mean_log_probability"
    assert selection["positive_loss_mode"] == "mean_log_probability"


def test_programmatic_training_args_use_cli_defaults():
    cli = train_stage1._argument_parser().parse_args(
        [
            "student-edge",
            "--features",
            "features.pt",
            "--dev-data",
            "dev.jsonl",
            "--output",
            "student.pt",
        ]
    )
    programmatic = argparse.Namespace(
        stage="student-edge",
        features="features.pt",
        dev_data=["dev.jsonl"],
        output="student.pt",
    )

    train_stage1._apply_argument_defaults(programmatic)

    assert vars(programmatic) == vars(cli)


def test_feature_store_preloads_contiguous_training_embeddings():
    store = feature_store()
    count = store.preload_embeddings(["q", "positive", "q"])
    query = store.embedding_features("q").embedding
    positive = store.embedding_features("positive").embedding

    assert count == 2
    assert query.data_ptr() + query.numel() * query.element_size() == positive.data_ptr()
    store.get = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("preloaded embedding fell back to the object file")
    )
    assert torch.equal(store.embedding_features("q").embedding, query)


def test_feature_store_preload_preserves_query_row_embeddings():
    row_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    store = FeatureStore(
        {
            "q": ObjectFeatures(
                "q",
                "table",
                torch.ones(4),
                row_embeddings=row_embeddings,
            )
        }
    )

    store.preload_embeddings(["q"])

    assert torch.equal(
        store.embedding_features("q").row_embeddings,
        row_embeddings,
    )


def test_retrieval_batches_all_evidence_to_target_queries():
    class BatchIndices:
        def __init__(self):
            self.batch_calls = []

        def search(self, source_id, destination_type, k):
            del k
            values = {
                ("q", "table"): [("direct", 1.0)],
                ("q", "text"): [("e1", 2.0), ("e2", 1.0)],
            }
            return values.get((source_id, destination_type), [])

        def search_many(self, source_ids, destination_type, k):
            self.batch_calls.append((source_ids, destination_type, k))
            values = {
                ("q", "table"): [("direct", 1.0)],
                ("q", "text"): [("e1", 2.0), ("e2", 1.0)],
            }
            return [
                values.get((source_id, destination_type), [("target", 0.5)])
                for source_id in source_ids
            ]

    indices = BatchIndices()
    retrieve_zero_one_hop_detailed(
        "q",
        indices,
        direct_k=1,
        evidence_k=2,
        targets_per_evidence=3,
        evidence_types=("text",),
    )

    assert indices.batch_calls == [
        (["q"], "table", 1),
        (["q"], "text", 2),
        (["e1", "e2"], "table", 3),
    ]


def test_epoch_controller_retains_epoch_zero_best_and_latest_indices(tmp_path):
    paths = [tmp_path / f"epoch_{epoch:03d}" for epoch in range(4)]
    for path in paths:
        path.mkdir()
    controller = object.__new__(train_stage1._EpochController)
    controller.created_indices = paths
    controller.epoch_zero_fallback = (tmp_path / "epoch_000.pt", {}, paths[0])
    controller.best_index = paths[1]
    controller.latest_index = paths[3]

    controller.prune_indices()

    assert paths[0].is_dir()
    assert paths[1].is_dir()
    assert not paths[2].exists()
    assert paths[3].is_dir()

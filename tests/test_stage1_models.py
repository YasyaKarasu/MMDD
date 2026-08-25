from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache_stage1_features import (
    EMBEDDING_INSTRUCTIONS,
    build_object_features,
    embedding_instructions,
)
from mmdd_stage1.checkpoints import load_path_aggregation
from mmdd_stage1.data import (
    EdgeExample,
    TargetCandidate,
    TargetExample,
    load_edge_examples,
    load_target_examples,
)
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.mining import (
    HardPath,
    build_hard_candidate_set,
    retrieve_hard_candidate_sets,
    score_hard_candidate_sets,
)
from mmdd_stage1.models import (
    TYPE_TO_ID,
    StudentJoinabilityModel,
    TeacherJoinabilityModel,
    structural_table_pool,
)
from mmdd_stage1.objectives import PathAggregator, listwise_cross_entropy
from mmdd_stage1.retrieval import (
    StudentANNIndices,
    build_indices,
    retrieve_zero_one_hop,
)
from mmdd_stage1.scoring import score_edge_batch, score_target_batch
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


def test_structural_table_pool_returns_schema_and_row_tokens():
    hidden = torch.tensor([[1.0, 1.0], [3.0, 3.0], [6.0, 4.0]])
    groups = torch.tensor([0, 0, 1])

    pooled = structural_table_pool(hidden, groups)

    assert torch.equal(pooled, torch.tensor([[2.0, 2.0], [6.0, 4.0]]))


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
    assert torch.allclose(projected_after_pooling, projected_before_pooling)
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
    edge_examples = [EdgeExample("q", ("positive", "negative"), positive_index=0)]
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
        teacher_model,
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
        teacher_model,
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
        json.dumps({"object_id": "q", "object_type": "table_fragment", "feature_path": "q.pt"}) + "\n",
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
                    {"target_id": "positive", "evidence_ids": ["e1"]},
                    {"target_id": "negative", "evidence_ids": []},
                ],
                "positive_target_ids": ["positive", "another_positive"],
                "teacher_direct_logits": [2.0, -1.0],
                "teacher_evidence_logits": [1.5, 0.0],
                "dataset": "2k",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = FeatureStore.from_path(feature_dir)
    examples = load_target_examples(data_path, max_evidence=1)

    assert store.get("q").object_type == "table"
    assert store.get("q").row_embeddings.shape == (1, 4)
    assert examples[0].direct_positive_index == 0
    assert examples[0].evidence_positive_index == 0
    assert examples[0].candidates[0].evidence_ids == ("e1",)
    assert examples[0].dataset == "2k"
    assert examples[0].teacher_direct_logits == (2.0, -1.0)
    assert examples[0].teacher_evidence_logits == (1.5, 0.0)
    assert examples[0].positive_target_ids == ("positive", "another_positive")


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

    def __init__(self):
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
    assert payload["hidden_states"].shape == (6, 4)
    assert payload["hidden_states"].dtype == torch.float16
    assert payload["token_groups"].tolist() == [0, 0, 0, 1, 1, 1]


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


def test_online_retrieval_uses_configured_two_level_path_aggregation():
    class StaticIndices:
        def search(self, source_id, destination_type, k):
            del k
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
        evidence_types=("text",),
        evidence_aggregation="topk_mean",
        evidence_top_k=1,
    )

    result = results[0]
    assert result["evidence_score"] == pytest.approx(4.0)
    assert result["score"] == pytest.approx(2.0 / 61.0)
    assert "direct_score" not in result
    assert "direct_rank" not in result
    assert "evidence_rank" not in result


def test_online_retrieval_rrf_fuses_route_ranks_without_changing_route_scores():
    class StaticIndices:
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
    torch.save(
        checkpoint(
            StudentJoinabilityModel(input_dim=4, student_dim=3),
            "student-path",
            PathAggregator("topk_sum", 2),
        ),
        path,
    )

    assert load_path_aggregation(path) == ("topk_sum", 2)


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


def test_student_edge_distillation_reuses_cached_teacher_logits():
    store = feature_store()
    teacher_model = teacher()
    student_model = StudentJoinabilityModel(input_dim=4, student_dim=3)
    example = EdgeExample(
        "q",
        ("positive", "negative"),
        0,
        dataset="2k",
        teacher_logits=(3.0, -2.0),
    )

    def unexpected_teacher_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached Teacher logits were not used")

    teacher_model.score_pairs = unexpected_teacher_call
    history = train_student_edges(
        student_model,
        teacher_model,
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
    teacher_model = teacher()
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

    def unexpected_teacher_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cached Teacher channel logits were not used")

    teacher_model.score_pairs = unexpected_teacher_call
    history = train_student_paths(
        student_model,
        teacher_model,
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


def test_hard_candidate_merge_excludes_gt_and_keeps_path_hard_evidence():
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
        positive_target_ids=("direct_positive", "evidence_positive", "other_positive"),
    )
    candidate_set = build_hard_candidate_set(
        original,
        ["other_positive", "hard_1", "hard_2"],
        [],
        [HardPath("e1", "hard_1", 5.0), HardPath("e2", "hard_1", 4.0)],
        hard_targets_per_query=2,
        max_evidence_per_target=2,
    )
    target_example = candidate_set.target_example

    assert [candidate.target_id for candidate in target_example.candidates] == [
        "direct_positive",
        "evidence_positive",
        "hard_1",
        "hard_2",
    ]
    assert target_example.direct_positive_index == 0
    assert target_example.evidence_positive_index == 1
    assert target_example.candidates[2:] == (
        TargetCandidate("hard_1", ("e1", "e2")),
        TargetCandidate("hard_2", ()),
    )


def test_hard_negative_refresh_mines_three_independent_candidate_pools():
    class StaticIndices:
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
        hard_paths_per_query=1,
        max_evidence_per_target=2,
        direct_k=2,
        evidence_k=3,
        targets_per_evidence=2,
        evidence_types=("text",),
    )[0]

    assert mined.evidence_negative_ids == ("evidence_only",)
    assert mined.target_example.candidates == (
        TargetCandidate("direct_positive", ()),
        TargetCandidate("evidence_positive", ("positive_evidence",)),
        TargetCandidate("hard_target", ()),
        TargetCandidate("path_target", ("positive_evidence",)),
    )

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
        max_evidence_per_target=1,
    )

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
        "dataset",
        "split",
    }
    assert target_records[0]["candidates"][1] == {
        "target_id": "hard",
        "evidence_ids": ["evidence"],
    }
    assert "student_logits" not in edge_records[0]

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
    loaded_target = load_target_examples(target_path, max_evidence=1)[0]
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
        max_evidence_per_target=2,
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

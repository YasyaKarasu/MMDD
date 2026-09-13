from __future__ import annotations

import pytest
import torch

from mmdd_stage1.data import EdgeExample
from mmdd_stage1.features import ObjectFeatures
from mmdd_stage1.models import TeacherJoinabilityModel
from run_stage1_r18 import GlobalResidualTeacher, _base_state_load
from run_stage1_r19 import (
    R19GlobalResidualTeacher,
    backward_logical_batch,
    construct_nested_tt_lists,
    stable_candidate_stream,
    state_dict_content_hash,
    widen_global_branch,
)
from mmdd_stage1.features import FeatureStore


def _features(object_id: str, object_type: str) -> ObjectFeatures:
    return ObjectFeatures(
        object_id=object_id,
        object_type=object_type,
        embedding=torch.randn(8),
        hidden_states=torch.randn(3, 8),
    )


def _a1() -> GlobalResidualTeacher:
    base = TeacherJoinabilityModel(
        input_dim=8,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
    ).eval()
    model = GlobalResidualTeacher(**base.config()).eval()
    _base_state_load(
        model,
        base,
        ("global_adapters.", "global_norms.", "global_residual_in.", "global_residual_out."),
    )
    with torch.no_grad():
        model.global_residual_out.weight.normal_()
        model.global_residual_out.bias.normal_()
    return model


def test_widen_global_branch_preserves_function_and_breaks_weight_symmetry() -> None:
    torch.manual_seed(7)
    parent = _a1()
    # The production transfer is fixed at 512. Exercise its algebra on an equivalent
    # small model by temporarily constructing the widened blocks directly.
    model = R19GlobalResidualTeacher(
        **parent.config(), global_dim=16, global_hidden_dim=8
    ).eval()
    excluded = ("global_adapters.", "global_norms.", "global_residual_in.", "global_residual_out.")
    model.load_state_dict(
        {k: v for k, v in parent.state_dict().items() if not k.startswith(excluded)},
        strict=False,
    )
    with torch.no_grad():
        for kind in ("table", "text", "image"):
            model.global_adapters[kind].weight.copy_(
                torch.cat([parent.global_adapters[kind].weight] * 2)
            )
            model.global_adapters[kind].bias.copy_(
                torch.cat([parent.global_adapters[kind].bias] * 2)
            )
            model.global_norms[kind].weight.copy_(
                torch.cat([parent.global_norms[kind].weight] * 2)
            )
            model.global_norms[kind].bias.copy_(
                torch.cat([parent.global_norms[kind].bias] * 2)
            )
        old = parent.global_residual_in.weight
        blocks = []
        generator = torch.Generator().manual_seed(190911)
        for index in range(4):
            block = old[:, index * 8 : (index + 1) * 8]
            noise = torch.randn(block.shape, generator=generator) * (1e-3 * block.square().mean().sqrt())
            blocks.append(torch.cat([0.5 * block + noise, 0.5 * block - noise], dim=1))
        model.global_residual_in.weight.copy_(torch.cat([*blocks, old[:, 32:40]], dim=1))
        model.global_residual_in.bias.copy_(parent.global_residual_in.bias)
        model.global_residual_out.load_state_dict(parent.global_residual_out.state_dict())
    source = _features("q", "table")
    destinations = [_features("t", "table"), _features("e", "text"), _features("i", "image")]
    expected = parent.score_pairs([source] * 3, destinations)
    actual = model.score_pairs([source] * 3, destinations)
    assert torch.allclose(expected, actual, atol=1e-5, rtol=0.0)
    first = model.global_residual_in.weight[:, :8]
    second = model.global_residual_in.weight[:, 8:16]
    assert not torch.equal(first, second)


def test_global_teacher_rejects_compressed_only_scoring() -> None:
    model = R19GlobalResidualTeacher(
        input_dim=8,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
    )
    with pytest.raises(RuntimeError, match="requires object embeddings"):
        model.score_compressed_pairs([], [], [], [])


def test_candidate_stream_is_stable_tiered_unique_and_positive_safe() -> None:
    kwargs = dict(
        source_id="q",
        relation="table_to_table",
        list_id="7:abc",
        tiers=(("a", "p", "b", "a"), ("c", "b"), ("d",)),
        excluded={"p"},
    )
    first = stable_candidate_stream(**kwargs)
    assert first == stable_candidate_stream(**kwargs)
    assert {value for value, _tier in first} == {"a", "b", "c", "d"}
    assert [tier for _value, tier in first] == sorted(tier for _value, tier in first)


def test_nested_c2_c3_only_change_tt_and_preserve_positive_position() -> None:
    tt = EdgeExample(
        "q",
        ("n0", "p", "n1"),
        1,
        source_type="table",
        destination_type="table",
        positive_ids=("p",),
    )
    ti = EdgeExample(
        "q",
        ("i0", "ip"),
        1,
        source_type="table",
        destination_type="image",
        positive_ids=("ip",),
    )
    c2, c3, summary, _audit = construct_nested_tt_lists(
        [tt, ti],
        {"q": {"top50": [f"h{x}" for x in range(40)], "remaining_natural": []}},
        long_length=32,
    )
    assert len(c2[0].candidate_ids) == 3
    assert c2[0].candidate_ids[1] == "p"
    assert len(c3[0].candidate_ids) == 32
    assert c3[0].candidate_ids[:3] == c2[0].candidate_ids
    assert len(set(c3[0].candidate_ids)) == 32
    assert c2[1] == ti and c3[1] == ti
    assert summary["long32_complete_fraction"] == 1.0


def test_state_dict_content_hash_is_order_independent_and_content_sensitive() -> None:
    left = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    right = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert state_dict_content_hash(left) == state_dict_content_hash(right)
    assert state_dict_content_hash(left) != state_dict_content_hash(changed)


def test_complete_list_microbatch_has_equivalent_loss_and_gradients() -> None:
    torch.manual_seed(17)
    left = R19GlobalResidualTeacher(
        input_dim=8,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
    )
    right = R19GlobalResidualTeacher(**left.config())
    right.load_state_dict(left.state_dict())
    features = {
        object_id: _features(object_id, "table")
        for object_id in ("q0", "q1", "p0", "n0", "n1", "p1", "n2", "n3")
    }
    store = FeatureStore(eager_features=features)
    examples = [
        EdgeExample(
            "q0", ("p0", "n0", "n1"), 0,
            source_type="table", destination_type="table", positive_ids=("p0",),
        ),
        EdgeExample(
            "q1", ("n2", "p1", "n3"), 1,
            source_type="table", destination_type="table", positive_ids=("p1",),
        ),
    ]
    loss_full, _ = backward_logical_batch(
        left, examples, store, torch.device("cpu"), microbatch_lists=2
    )
    loss_split, _ = backward_logical_batch(
        right, examples, store, torch.device("cpu"), microbatch_lists=1
    )
    assert loss_full == pytest.approx(loss_split, abs=1e-6)
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
        left.named_parameters(), right.named_parameters(), strict=True
    ):
        assert left_name == right_name
        if left_parameter.grad is None or right_parameter.grad is None:
            assert left_parameter.grad is None and right_parameter.grad is None
        else:
            assert torch.allclose(
                left_parameter.grad, right_parameter.grad, atol=2e-6, rtol=1e-5
            )

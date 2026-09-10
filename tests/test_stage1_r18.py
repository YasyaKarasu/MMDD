from __future__ import annotations

import torch

from mmdd_stage1.features import ObjectFeatures
from mmdd_stage1.models import TeacherJoinabilityModel
from run_stage1_r18 import (
    GlobalResidualTeacher,
    RelationHeadsTeacher,
    _base_state_load,
    protect_known_positives,
)
from mmdd_stage1.data import EdgeExample


def _features(object_id: str, object_type: str) -> ObjectFeatures:
    return ObjectFeatures(
        object_id=object_id,
        object_type=object_type,
        embedding=torch.randn(8),
        hidden_states=torch.randn(3, 8),
    )


def _base() -> TeacherJoinabilityModel:
    return TeacherJoinabilityModel(
        input_dim=8,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        text_latents=2,
        image_latents=2,
        dropout=0.0,
    ).eval()


def test_global_residual_is_function_equivalent_at_step0() -> None:
    torch.manual_seed(7)
    base = _base()
    model = GlobalResidualTeacher(**base.config()).eval()
    _base_state_load(
        model,
        base,
        ("global_adapters.", "global_norms.", "global_residual_in.", "global_residual_out."),
    )
    source = _features("q", "table")
    destinations = [_features("t", "table"), _features("e", "text")]
    expected = base.score_pairs([source, source], destinations)
    actual = model.score_pairs([source, source], destinations)
    assert torch.equal(expected, actual)


def test_relation_heads_copy_shared_head_at_step0() -> None:
    torch.manual_seed(11)
    base = _base()
    model = RelationHeadsTeacher(**base.config()).eval()
    state = {
        name: value
        for name, value in base.state_dict().items()
        if not name.startswith("scoring_head.")
    }
    model.load_state_dict(state, strict=False)
    for head in model.relation_scoring_heads.values():
        head.load_state_dict(base.scoring_head.state_dict(), strict=True)
    source = _features("q", "table")
    destinations = [_features("t", "table"), _features("e", "image")]
    assert torch.allclose(
        base.score_pairs([source, source], destinations),
        model.score_pairs([source, source], destinations),
        atol=1e-6,
        rtol=0.0,
    )


def test_known_positive_protection_is_query_relation_local() -> None:
    examples = [
        EdgeExample(
            "q",
            ("a", "b", "c"),
            0,
            source_type="table",
            destination_type="table",
            positive_ids=("a",),
        ),
        EdgeExample(
            "q",
            ("a", "b", "d"),
            1,
            source_type="table",
            destination_type="table",
            positive_ids=("b",),
        ),
    ]
    protected, exclusions = protect_known_positives(examples)
    assert protected[0].positive_ids == ("a", "b")
    assert protected[1].positive_ids == ("a", "b")
    assert exclusions == 2

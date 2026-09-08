from __future__ import annotations

import math

import pytest
import torch

from mmdd_stage1.objectives import PathAggregator
from mmdd_stage1.retrieval import fuse_ranked_channels, rank_detailed_paths
from mmdd_stage1.data import TargetCandidate, TargetExample
from mmdd_stage1.features import FeatureStore, ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.protocol import validate_r6_readonly_invariants
from mmdd_stage1.scoring import score_target_batch


def test_r6_path_aggregator_variants_match_scalar_definitions():
    query_evidence = torch.tensor([[[2.0, 0.0]]])
    evidence_target = torch.tensor([[[3.0, 1.0]]])
    mask = torch.ones_like(query_evidence, dtype=torch.bool)

    expected = {
        "max": 5.0,
        "topk_mean": 3.0,
        "topk_sum": 6.0,
        "comb_mnz": 12.0,
        "power_mean": 1.0 + math.sqrt(8.0),
    }
    for name, value in expected.items():
        aggregator = PathAggregator(name, 2, power=2.0)
        assert aggregator(query_evidence, evidence_target, mask).item() == pytest.approx(
            value
        )

    softmax = PathAggregator("softmax_weighted_mean", temperature=1.0)
    weights = torch.softmax(torch.tensor([5.0, 1.0]), dim=0)
    expected_softmax = float((weights * torch.tensor([5.0, 1.0])).sum())
    assert softmax(query_evidence, evidence_target, mask).item() == pytest.approx(
        expected_softmax
    )


@pytest.mark.parametrize(
    "name",
    [
        "comb_mnz",
        "fixed_power_mean",
        "logmeanexp",
        "logsumexp",
        "max",
        "power_mean",
        "softmax_weighted_mean",
        "topk_logmeanexp",
        "topk_logsumexp",
        "topk_mean",
        "topk_sum",
    ],
)
def test_r6_path_aggregators_have_finite_all_mask_gradients(name):
    left = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    right = torch.tensor([[[3.0, 4.0]]], requires_grad=True)
    mask = torch.zeros_like(left, dtype=torch.bool)

    score = PathAggregator(name)(left, right, mask)
    score.sum().backward()

    assert score.item() == 0.0
    assert torch.isfinite(left.grad).all()
    assert torch.isfinite(right.grad).all()


def test_logmeanexp_uses_temperature_and_counts_only_unmasked_paths():
    left = torch.zeros((1, 2, 4))
    right = torch.tensor([[[1.5, 1.0, 0.0, 100.0], [0.0, 0.0, 7.0, 9.0]]])
    mask = torch.tensor([[[True, True, True, False], [True, True, False, False]]])

    actual = PathAggregator("logmeanexp", temperature=0.3)(left, right, mask)
    expected = torch.stack(
        [
            0.3 * torch.logsumexp(right[0, 0, :3] / 0.3, dim=0)
            - 0.3 * math.log(3),
            torch.tensor(0.0),
        ]
    ).unsqueeze(0)

    torch.testing.assert_close(actual, expected)


def test_logmeanexp_singleton_and_complete_replication_are_invariant():
    singleton = torch.tensor([[[1.25]]])
    replicated = singleton.repeat(1, 1, 8)

    aggregator = PathAggregator("logmeanexp", temperature=0.1)
    singleton_score = aggregator(
        torch.zeros_like(singleton),
        singleton,
        torch.ones_like(singleton, dtype=torch.bool),
    )
    replicated_score = aggregator(
        torch.zeros_like(replicated),
        replicated,
        torch.ones_like(replicated, dtype=torch.bool),
    )

    assert singleton_score.item() == pytest.approx(1.25)
    assert replicated_score.item() == pytest.approx(1.25)


def test_logmeanexp_and_logsumexp_have_equal_fixed_pool_gradients():
    lme_values = torch.tensor([[[1.5, 1.0, -0.25]]], requires_grad=True)
    lse_values = lme_values.detach().clone().requires_grad_()
    mask = torch.ones_like(lme_values, dtype=torch.bool)

    lme = PathAggregator("logmeanexp", temperature=0.3)(
        torch.zeros_like(lme_values), lme_values, mask
    )
    lse = PathAggregator("logsumexp", temperature=0.3)(
        torch.zeros_like(lse_values), lse_values, mask
    )
    lme.backward()
    lse.backward()

    torch.testing.assert_close(lme_values.grad, lse_values.grad)


def test_topk_log_aggregations_apply_a_real_path_limit():
    values = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    mask = torch.ones_like(values, dtype=torch.bool)

    full_lse = PathAggregator("logsumexp", 2)(
        torch.zeros_like(values), values, mask
    )
    top_lse = PathAggregator("topk_logsumexp", 2)(
        torch.zeros_like(values), values, mask
    )
    top_lme = PathAggregator("topk_logmeanexp", 2)(
        torch.zeros_like(values), values, mask
    )

    assert top_lse.item() == pytest.approx(
        torch.logsumexp(values[0, 0, :2], dim=0).item()
    )
    assert top_lme.item() == pytest.approx(top_lse.item() - math.log(2))
    assert top_lse.item() < full_lse.item()


def test_fixed_power_mean_uses_fixed_budget_and_threshold():
    values = torch.tensor([[[0.8, 0.3, 0.2, 0.1, 0.05]]])
    mask = torch.ones_like(values, dtype=torch.bool)

    g3 = PathAggregator("fixed_power_mean", 4, power=4)(
        torch.zeros_like(values), values, mask
    )
    g4 = PathAggregator(
        "fixed_power_mean", 4, power=4, threshold=0.5, path_combination="min"
    )(
        torch.ones_like(values), values, mask
    )

    assert g3.item() == pytest.approx(
        ((0.8**4 + 0.3**4 + 0.2**4 + 0.1**4) / 4) ** 0.25
    )
    assert g4.item() == pytest.approx((0.6**4 / 4) ** 0.25)


def test_fixed_power_mean_is_bounded_and_ignores_paths_after_budget():
    base = torch.tensor([[[0.8, 0.3, 0.2, 0.1]]])
    extended = torch.cat([base, torch.full((1, 1, 20), 0.09)], dim=-1)
    aggregator = PathAggregator(
        "fixed_power_mean", 4, power=2, path_combination="min"
    )

    base_score = aggregator(
        torch.ones_like(base), base, torch.ones_like(base, dtype=torch.bool)
    )
    extended_score = aggregator(
        torch.ones_like(extended),
        extended,
        torch.ones_like(extended, dtype=torch.bool),
    )

    torch.testing.assert_close(base_score, extended_score)
    assert 0.0 <= base_score.item() <= 1.0


def test_path_combination_sum_min_and_product_are_explicit():
    left = torch.tensor([[[0.8]]])
    right = torch.tensor([[[0.5]]])
    mask = torch.ones_like(left, dtype=torch.bool)

    assert PathAggregator("max", path_combination="sum")(left, right, mask).item() == pytest.approx(1.3)
    assert PathAggregator("max", path_combination="min")(left, right, mask).item() == pytest.approx(0.5)
    assert PathAggregator("max", path_combination="product")(left, right, mask).item() == pytest.approx(0.4)


def test_evidence_target_temperature_scales_only_aggregated_target_score():
    left = torch.tensor([[[0.8, 0.3]]])
    right = torch.tensor([[[0.5, 0.7]]])
    mask = torch.ones_like(left, dtype=torch.bool)

    base = PathAggregator("max", path_combination="min")(
        left, right, mask
    )
    sharpened = PathAggregator(
        "max", path_combination="min", target_temperature=0.1
    )(left, right, mask)

    torch.testing.assert_close(sharpened, base / 0.1)


def test_fixed_power_mean_scalar_and_tensor_implementations_match():
    left = torch.tensor([[[0.8, 0.3, 0.2]]])
    right = torch.tensor([[[0.9, 0.7, 0.1]]])
    aggregator = PathAggregator(
        "fixed_power_mean",
        4,
        power=4,
        threshold=0.25,
        path_combination="min",
    )
    tensor_score = aggregator(
        left, right, torch.ones_like(left, dtype=torch.bool)
    ).item()
    paths = [
        {
            "kind": "evidence",
            "evidence_id": f"e-{index}",
            "evidence_type": "text",
            "query_evidence_score": float(left[0, 0, index]),
            "evidence_target_score": float(right[0, 0, index]),
            "path_score": float(left[0, 0, index] + right[0, 0, index]),
        }
        for index in range(left.shape[-1])
    ]
    channels = rank_detailed_paths(
        {"target": paths},
        aggregator=aggregator,
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
        gated_evidence_min_paths=2,
        gated_evidence_quantile=0.75,
    )

    assert channels["evidence"][0]["evidence_score"] == pytest.approx(tensor_score)


@pytest.mark.parametrize(
    "name,top_k",
    [
        ("logmeanexp", 4),
        ("logsumexp", 4),
        ("topk_logmeanexp", 2),
        ("topk_logsumexp", 2),
    ],
)
def test_log_aggregation_matches_training_and_retrieval_paths(name, top_k):
    values = torch.tensor([[[3.0, 2.0, 1.0]]])
    aggregator = PathAggregator(name, top_k, temperature=0.3)
    tensor_score = aggregator(
        torch.zeros_like(values), values, torch.ones_like(values, dtype=torch.bool)
    ).item()
    channels = rank_detailed_paths(
        {
            "target": [
                {"kind": "evidence", "path_score": value}
                for value in values.flatten().tolist()
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

    assert channels["evidence"][0]["evidence_score"] == pytest.approx(tensor_score)


def test_r6_normalized_score_fusion_uses_channel_amplitudes():
    paths = [{"kind": "direct"}]
    direct = [
        {"target_id": "a", "direct_score": 2.0, "evidence_score": 1.0, "paths": paths},
        {"target_id": "b", "direct_score": 1.0, "evidence_score": 2.0, "paths": paths},
    ]
    evidence = list(reversed(direct))

    fused = fuse_ranked_channels(
        direct,
        evidence,
        fusion_mode="normalized_score",
        direct_weight=1.0,
        evidence_weight=2.0,
        score_normalization="minmax",
    )

    assert [row["target_id"] for row in fused] == ["b", "a"]
    assert [row["score"] for row in fused] == pytest.approx([2.0, 1.0])


def test_r6_path_edge_zscore_is_per_relation_and_preserves_pool():
    original = {
        "t1": [
            {"kind": "direct", "path_score": 3.0},
            {
                "kind": "evidence",
                "evidence_id": "text-1",
                "evidence_type": "text",
                "query_evidence_score": 10.0,
                "evidence_target_score": 1.0,
                "path_score": 11.0,
            },
        ],
        "t2": [
            {"kind": "direct", "path_score": 2.0},
            {
                "kind": "evidence",
                "evidence_id": "text-2",
                "evidence_type": "text",
                "query_evidence_score": 8.0,
                "evidence_target_score": 9.0,
                "path_score": 17.0,
            },
            {
                "kind": "evidence",
                "evidence_id": "image-1",
                "evidence_type": "image",
                "query_evidence_score": 100.0,
                "evidence_target_score": 50.0,
                "path_score": 150.0,
            },
        ],
    }

    ranked = rank_detailed_paths(
        original,
        aggregator=PathAggregator("max"),
        path_edge_normalization="zscore",
        rrf_k=60,
        fusion_mode="weighted_rrf",
        direct_weight=1.0,
        evidence_weight=0.05,
        gated_evidence_min_paths=2,
        gated_evidence_quantile=0.75,
    )

    assert {row["target_id"] for row in ranked["direct"]} == {"t1", "t2"}
    assert {row["target_id"] for row in ranked["evidence"]} == {"t1", "t2"}
    text_paths = {
        row["target_id"]: next(
            path
            for path in row["paths"]
            if path.get("evidence_type") == "text"
        )
        for row in ranked["evidence"]
    }
    assert text_paths["t1"]["normalized_query_evidence_score"] == pytest.approx(1.0)
    assert text_paths["t2"]["normalized_query_evidence_score"] == pytest.approx(-1.0)
    assert text_paths["t1"]["normalized_evidence_target_score"] == pytest.approx(-1.0)
    assert text_paths["t2"]["normalized_evidence_target_score"] == pytest.approx(1.0)
    assert "normalized_query_evidence_score" not in original["t1"][1]


def test_r6_relation_loss_weights_scale_image_gradients_without_changing_scores():
    store = FeatureStore(
        {
            "q": ObjectFeatures("q", "table", torch.tensor([1.0, 0.0])),
            "target": ObjectFeatures("target", "table", torch.tensor([0.5, 0.5])),
            "image": ObjectFeatures("image", "image", torch.tensor([0.0, 1.0])),
        }
    )
    examples = [
        TargetExample(
            "q",
            (TargetCandidate("target", ("image",)),),
            direct_positive_index=0,
            evidence_positive_index=0,
        )
    ]
    base = StudentJoinabilityModel(2, 2)
    weighted = StudentJoinabilityModel(2, 2)
    weighted.load_state_dict(base.state_dict())

    base_scores = score_target_batch(
        base, examples, store, torch.device("cpu"), PathAggregator("max")
    )
    weighted_scores = score_target_batch(
        weighted,
        examples,
        store,
        torch.device("cpu"),
        PathAggregator("max"),
        relation_loss_weights={"table_to_image": 2.0, "image_to_table": 2.0},
    )
    assert torch.equal(base_scores.evidence.logits, weighted_scores.evidence.logits)

    base_scores.evidence.logits.sum().backward()
    weighted_scores.evidence.logits.sum().backward()
    for key in ("table_to_image", "image_to_table"):
        assert torch.allclose(
            weighted.relations[key].grad,
            2.0 * base.relations[key].grad,
        )


def test_r6_readonly_protocol_requires_image_fixed_pool_and_two_teachers():
    valid = {
        "evidence_types": ("text", "image"),
        "candidate_pool_fingerprints": {"baseline": "pool", "variant": "pool"},
        "teacher_checkpoints": {"entitables": "teacher-e", "wdc": "teacher-w"},
    }
    validate_r6_readonly_invariants(**valid)

    with pytest.raises(ValueError, match="retain image"):
        validate_r6_readonly_invariants(**{**valid, "evidence_types": ("text",)})
    with pytest.raises(ValueError, match="fixed candidate"):
        validate_r6_readonly_invariants(
            **{
                **valid,
                "candidate_pool_fingerprints": {
                    "baseline": "pool-a",
                    "variant": "pool-b",
                },
            }
        )
    with pytest.raises(ValueError, match="distinct lake-local"):
        validate_r6_readonly_invariants(
            **{
                **valid,
                "teacher_checkpoints": {
                    "entitables": "teacher",
                    "wdc": "teacher",
                },
            }
        )

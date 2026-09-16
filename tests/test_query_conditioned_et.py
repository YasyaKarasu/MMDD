import torch

from mmdd_stage1.query_conditioned_et import (
    QueryConditionedETAdapter,
    grouped_paired_bootstrap,
    residual_geometry,
    sum_probability_listwise_loss,
)


def test_query_conditioned_adapter_has_exact_step0_parity():
    torch.manual_seed(3)
    adapter = QueryConditionedETAdapter(8, 4)
    query = torch.randn(5, 8)
    evidence = torch.randn(5, 8)
    base = torch.randn(5, 8)
    for arm in ("e_only", "qe"):
        assert torch.equal(adapter(query, evidence, arm), torch.zeros_like(base))
        assert torch.equal(adapter.conditioned_query(base, query, evidence, arm), base)


def test_e_only_is_query_invariant_but_qe_input_is_not():
    adapter = QueryConditionedETAdapter(3, 2)
    query = torch.tensor([[1.0, 2.0, 3.0]])
    shuffled = torch.tensor([[3.0, 1.0, 2.0]])
    evidence = torch.tensor([[0.5, 0.25, 0.75]])
    assert torch.equal(
        adapter.features(query, evidence, "e_only"),
        adapter.features(shuffled, evidence, "e_only"),
    )
    assert not torch.equal(
        adapter.features(query, evidence, "qe"),
        adapter.features(shuffled, evidence, "qe"),
    )


def test_sum_probability_loss_rewards_combined_positive_mass():
    mask = torch.tensor([[True, True, False]])
    better = sum_probability_listwise_loss(torch.tensor([[1.0, 1.0, 0.0]]), mask)
    worse = sum_probability_listwise_loss(torch.tensor([[0.0, 0.0, 1.0]]), mask)
    assert better < worse


def test_residual_geometry_and_grouped_bootstrap():
    geometry = residual_geometry(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.0]]))
    assert geometry["residual_ratio"].item() == 0.0
    assert geometry["base_conditioned_cosine"].item() == 1.0
    result = grouped_paired_bootstrap(
        {"q1": 1.0, "q2": 0.0},
        {"q1": 0.0, "q2": 0.0},
        {"q1": "g1", "q2": "g2"},
        samples=100,
        seed=7,
    )
    assert result["delta"] == 0.5
    assert result["queries"] == 2

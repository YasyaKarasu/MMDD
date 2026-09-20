"""CPU synthetic tests. Not a claim that the MMDD pipeline was executed."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from contracts import (
    BoundedAdapter, ContractError, bounded_query, build_masks,
    full_target_losses, gate_a, gate_b, path_lse, query_macro_weights,
    recall_at_k, round_robin_order, stable_rank, strict_cohort,
    sum_probability_losses, support_contrast, weighted_microbatch_loss,
)

torch.set_num_threads(1)


def test_zero_initialization_and_parity():
    torch.manual_seed(13)
    a = BoundedAdapter(5, 'QE')
    q, e, v = [torch.randn(4, 5) for _ in range(3)]
    assert torch.equal(a(q, e, v), v)
    assert torch.count_nonzero(a.output.weight) == 0
    assert torch.count_nonzero(a.output.bias) == 0


def test_identical_initial_states_across_arms():
    a = BoundedAdapter(5, 'QE')
    b = BoundedAdapter(5, 'EONLY')
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_first_step_reaches_output_layer():
    torch.manual_seed(13)
    a = BoundedAdapter(5, 'QE')
    q, e, v = [torch.randn(4, 5) for _ in range(3)]
    (a(q, e, v) * torch.randn(4, 5)).sum().backward()
    assert a.output.weight.grad.abs().sum() > 0
    assert a.hidden.weight.grad.abs().sum() == 0


def test_norm_cap_for_huge_residual():
    torch.manual_seed(9)
    b = torch.randn(100, 7)
    d = torch.randn(100, 7) * 1e5
    out = bounded_query(b, d)
    ratios = (out - b).norm(dim=1) / b.norm(dim=1)
    assert ratios.max() <= 0.500001
    assert torch.isfinite(out).all()


def test_zero_and_tiny_residual_do_not_renormalize():
    b = torch.tensor([[3.0, 4.0]])
    assert torch.equal(bounded_query(b, torch.zeros_like(b)), b)
    d = torch.tensor([[1e-6, -1e-6]])
    assert torch.equal(bounded_query(b, d), b + d)
    assert not torch.isclose(bounded_query(b, d).norm(), torch.tensor(1.0))


def test_zero_base_rejected():
    with pytest.raises(ContractError):
        bounded_query(torch.zeros(1, 3), torch.ones(1, 3))


def test_clip_factor_gradient_is_not_detached():
    b = torch.tensor([[2., 1., 0.]], dtype=torch.double)
    d = torch.tensor([[3., 5., 7.]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: bounded_query(b, x), (d,))


def test_eonly_is_q_invariant_after_nonzero_weights():
    torch.manual_seed(3)
    a = BoundedAdapter(5, 'EONLY')
    with torch.no_grad():
        a.output.weight.normal_(std=0.1)
    q, e, v = [torch.randn(4, 5) for _ in range(3)]
    assert torch.equal(a(q, e, v), a(q * 17 + 5, e, v))


def test_qe_can_depend_on_q_after_nonzero_weights():
    torch.manual_seed(3)
    a = BoundedAdapter(5, 'QE')
    with torch.no_grad():
        a.output.weight.normal_(std=0.1)
    q, e, v = [torch.randn(4, 5) for _ in range(3)]
    assert not torch.allclose(a(q, e, v), a(q * 17 + 5, e, v))


def test_non_symmetric_relation_orientation():
    e = torch.tensor([[1., 3.]])
    t = torch.tensor([[2., -1.]])
    r = torch.tensor([[1., 2.], [-3., 4.]])
    correct = (e @ r) @ t.T
    assert torch.allclose(correct, e @ r @ t.T)
    assert not torch.allclose(correct, (e @ r.T) @ t.T)


def test_masks_protect_other_known_targets_without_marking_positive():
    p, i, n = build_masks(['a', 'b', 'c', 'd'], {'a', 'b', 'c', 'd'},
                         {'a'}, {'a', 'b'}, {'a', 'c'})
    assert p.tolist() == [True, False, False, False]
    assert i.tolist() == [False, True, True, False]
    assert n.tolist() == [False, False, False, True]


def test_label_error_blocked():
    with pytest.raises(ContractError):
        build_masks(['a', 'b'], {'b'}, {'a'}, {'a'}, set())


def test_ignore_logits_and_gradients():
    x = torch.tensor([[0.4, 10000., -0.3]], requires_grad=True)
    p = torch.tensor([[True, False, False]])
    n = torch.tensor([[False, False, True]])
    loss = sum_probability_losses(x, p, n)
    other = sum_probability_losses(torch.tensor([[0.4, -10000., -0.3]]), p, n)
    assert torch.allclose(loss, other)
    loss.sum().backward()
    assert x.grad[0, 1] == 0


def test_new_global_competitor_has_gradient():
    x = torch.tensor([[2.0, -2.0, 9.0]], requires_grad=True)
    p = torch.tensor([[True, False, False]])
    n = ~p
    sum_probability_losses(x, p, n).sum().backward()
    assert x.grad[0, 2] > 0.9


def test_empty_positives_not_silently_defaulted():
    p = torch.zeros(1, 3, dtype=torch.bool)
    with pytest.raises(ContractError):
        sum_probability_losses(torch.ones(1, 3), p, ~p)
    packet = {'positive_ids': []}
    restored = json.loads(json.dumps(packet))
    assert restored['positive_ids'] == [] and 'positive_ids' in restored


def test_empty_competitors_rejected_before_loss():
    p = torch.tensor([[True, False]])
    with pytest.raises(ContractError):
        sum_probability_losses(torch.ones(1, 2), p, torch.zeros_like(p))


def test_multi_positive_sum_probability_formula():
    x = torch.tensor([[1., 2., 3.]], dtype=torch.double)
    p = torch.tensor([[True, True, False]])
    expected = -torch.log((x[0, :2].exp().sum()) / x[0].exp().sum())
    assert torch.allclose(sum_probability_losses(x, p, ~p)[0], expected)


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 8])
def test_global_chunk_loss_and_gradient_match_dense(chunk_size):
    torch.manual_seed(1)
    q = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
    t = torch.randn(9, 4, dtype=torch.double)
    p = torch.zeros(3, 9, dtype=torch.bool)
    p[0, [0, 8]] = True
    p[1, [3]] = True
    p[2, [2, 6]] = True
    n = ~p
    n[:, 5] = False  # ignored target; some chunks have no positives
    dense = full_target_losses(q, t, p, n)
    dg = torch.autograd.grad(dense.sum(), q)[0]
    qc = q.detach().clone().requires_grad_(True)
    chunked = full_target_losses(qc, t, p, n, chunk_size)
    cg = torch.autograd.grad(chunked.sum(), qc)[0]
    assert torch.allclose(dense, chunked, atol=1e-10, rtol=1e-10)
    assert torch.allclose(dg, cg, atol=1e-10, rtol=1e-10)


def test_chunked_extremely_low_positive_is_finite():
    q = torch.tensor([[1.]], requires_grad=True)
    t = torch.tensor([[-10000.], [10000.], [0.]])
    p = torch.tensor([[True, False, False]])
    n = ~p
    loss = full_target_losses(q, t, p, n, 1)
    assert torch.isfinite(loss).all() and torch.allclose(loss, torch.tensor([20000.]))
    loss.sum().backward()
    assert torch.isfinite(q.grad).all()


def test_weighted_microbatch_gradient_and_tail():
    x = torch.arange(1., 8., requires_grad=True)
    w = torch.tensor([1., 0.5, 2., 1., 2., 1., 0.5])
    dense = weighted_microbatch_loss(x.square(), w, 7)
    gd = torch.autograd.grad(dense, x)[0]
    z = x.detach().clone().requires_grad_(True)
    for start in range(0, 7, 3):
        weighted_microbatch_loss(z[start:start+3].square(), w[start:start+3], 7).backward()
    assert torch.allclose(gd, z.grad)


def test_query_macro_total_weight():
    qids = ['q1', 'q1', 'q1', 'q2']
    w = query_macro_weights(qids)
    assert torch.allclose(w[:3].sum(), w[3])
    assert torch.allclose(w.mean(), torch.tensor(1., dtype=torch.double))


def test_sampler_exact_coverage_and_determinism():
    pairs = [('q1', 'e1'), ('q2', 'e1'), ('q1', 'e2'), ('q3', 'e2'), ('q4', 'e3')]
    a = round_robin_order(pairs, 13, 1)
    assert sorted(a) == list(range(len(pairs)))
    assert a == round_robin_order(pairs, 13, 1)


def test_sampler_rejects_duplicate_pair():
    with pytest.raises(ContractError):
        round_robin_order([('q1', 'e1'), ('q1', 'e1')], 13, 1)


def test_path_lse_multiplicity_and_empty():
    z = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    e = torch.tensor([0.4, -1.2], dtype=torch.double, requires_grad=True)
    value = path_lse(z, e, [3, 2])
    expanded = torch.logsumexp(torch.stack([z, e[0], e[0], e[0], e[1], e[1]]), 0)
    assert torch.allclose(value, expanded)
    assert torch.equal(path_lse(z, e[:0], []), z)
    assert torch.allclose(torch.autograd.grad(value, e, retain_graph=True)[0],
                          torch.autograd.grad(expanded, e)[0])


def test_invalid_multiplicity_rejected():
    with pytest.raises(ContractError):
        path_lse(torch.tensor(0.), torch.tensor([1.]), [0])


def test_qt_support_same_scalar_has_zero_gradient():
    scalar = torch.tensor(3., requires_grad=True)
    loss = support_contrast(scalar, scalar)
    loss.backward()
    assert scalar.grad == 0
    assert math.isclose(loss.item(), math.log1p(math.e), rel_tol=1e-6)


def test_strict_double_exclusion_and_recall_not_hit():
    gold = {'a', 'b', 'c'}
    strict = strict_cohort(gold, {'a', 'b', 'c'}, {'a'}, {'b'})
    assert strict == {'c'}
    assert recall_at_k(gold, ['x', 'a', 'y'], 2) == 1/3


def test_id_tie_break():
    assert stable_rank(['b', 'a', 'c'], [1., 1., 0.]) == ['a', 'b', 'c']


def test_gate_a_all_checks_required():
    base = dict(et_r10=.30, u_implicit=.6, u_overall=.7, u_explicit=.8, c100=.58)
    e = dict(et_r10=.31)
    qe = dict(et_r10=.33, u_implicit=.62, u_overall=.705, u_explicit=.8, c100=.58)
    assert gate_a(base, e, qe).passed
    assert not gate_a(base, e, {**qe, 'c100': .55}).passed
    assert not gate_a(base, e, qe, complete=False).passed


def test_gate_b_qt_is_control_not_fallback():
    t0 = dict(overall=.47, implicit=.39, strict=.34)
    qt = dict(overall=.475, implicit=.40, strict=.35)
    path = dict(overall=.478, implicit=.42, strict=.38)
    assert gate_b(t0, qt, path, .462).passed
    assert not gate_b(t0, qt, {**path, 'strict': .35}, .462).passed


def test_protocol_matches_reference_constants():
    config = json.loads((Path(__file__).parents[1] / 'protocol.json').read_text())
    assert config['A']['rho'] == 0.5
    assert config['A']['logit_temperature'] == 1.0
    assert config['A']['logical_batch_size'] == 128
    assert config['A']['epochs'] == 3
    assert config['B']['epochs'] == 2
    assert config['B']['support_weight'] == .2
    assert config['max_formal_training_jobs'] == 8
    assert config['out_of_scope']['kd_training'] is True


def test_gate_rejects_nonfinite_or_out_of_range_metric():
    base = dict(et_r10=.3, u_implicit=.6, u_overall=.7, u_explicit=.8, c100=.58)
    for invalid in [float('nan'), float('inf'), -0.1, 1.1]:
        with pytest.raises(ContractError):
            gate_a(base, dict(et_r10=.3), {**base, 'et_r10': invalid})


def test_gate_equality_only_has_fixed_roundoff_tolerance():
    base = dict(et_r10=.3, u_implicit=.6, u_overall=.7, u_explicit=.8, c100=.58)
    qe = dict(et_r10=.31, u_implicit=.605, u_overall=.6975, u_explicit=.795, c100=.575)
    assert gate_a(base, dict(et_r10=.305), qe).passed
    assert not gate_a(base, dict(et_r10=.305), {**qe, 'et_r10': .30999}).passed


def test_gate_missing_metric_is_not_an_automatic_pass():
    with pytest.raises(ContractError):
        gate_b(dict(overall=.4, implicit=.3), dict(overall=.4, implicit=.3, strict=.2),
               dict(overall=.5, implicit=.4, strict=.3), .4)

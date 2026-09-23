from __future__ import annotations

import torch

from mmdd_stage2.r4c_fast_localizer import (
    agreement_scores,
    concentration,
    consensus_attribute_map,
    joint_map,
    prominence,
    raea_attribute_map,
    resolve_layer_indices,
    tight_crop_box,
)


class _Attention:
    def __init__(self, full: bool) -> None:
        if full:
            self.v_proj = object()


class _Layer:
    def __init__(self, full: bool) -> None:
        self.self_attn = _Attention(full)


def test_layer_resolver_only_selects_vproj_and_is_deterministic() -> None:
    layers = [_Layer(index in {3, 7, 11, 15, 19, 23, 27, 31}) for index in range(32)]
    first, audit = resolve_layer_indices(layers)
    second, _ = resolve_layer_indices(layers)
    assert first == second == (15, 23, 27)
    assert audit["valid_vproj_layers"] == [3, 7, 11, 15, 19, 23, 27, 31]


def test_entropy_concentration_delta_exceeds_uniform() -> None:
    uniform = torch.ones(4, 4)
    delta = torch.zeros(4, 4)
    delta[1, 2] = 1
    assert concentration(delta) > concentration(uniform)


def test_centroid_agreement_and_prominence_are_bounded() -> None:
    maps = torch.stack([torch.eye(4), torch.eye(4), torch.flip(torch.eye(4), [1])])
    agreement = agreement_scores(maps)
    assert torch.all((agreement >= 0) & (agreement <= 1))
    assert 0 <= prominence(torch.eye(4)) <= 1


def test_raea_weights_sum_one_and_joint_is_normalized() -> None:
    maps = torch.rand(6, 5, 7)
    result = raea_attribute_map(maps)
    assert torch.isclose(result.weights.sum(), torch.tensor(1.0))
    assert torch.isclose(result.heatmap.sum(), torch.tensor(1.0))
    joint = joint_map(result.heatmap, result.heatmap)
    assert torch.isclose(joint.sum(), torch.tensor(1.0))


def test_consensus_suppresses_a_single_isolated_peak() -> None:
    base = torch.zeros(5, 5)
    base[2, 2] = 1
    outlier = torch.zeros(5, 5)
    outlier[0, 0] = 1
    maps = torch.stack([base, base, outlier])
    raea = raea_attribute_map(maps)
    consensus = consensus_attribute_map(maps, raea.heatmap)
    assert consensus[2, 2] > consensus[0, 0]


def test_mass_box_contains_top_mass_and_stays_in_bounds() -> None:
    heatmap = torch.zeros(10, 10)
    heatmap[4:6, 4:6] = 1
    box, reason = tight_crop_box(heatmap, 1000, 500)
    assert reason is None
    assert box is not None
    x1, y1, x2, y2 = box
    assert 0 <= x1 < 400 < 600 < x2 <= 1000
    assert 0 <= y1 < 200 < 300 < y2 <= 500


def test_uniform_map_falls_back_before_degenerate_crop() -> None:
    box, reason = tight_crop_box(torch.ones(10, 10), 1000, 500)
    assert box is None
    assert reason == "DIFFUSE_HEATMAP"


def test_broad_concentrated_map_can_trigger_85_percent_fallback() -> None:
    heatmap = torch.full((10, 10), 1e-6)
    heatmap[0:9, 0:9] = 1
    box, reason = tight_crop_box(heatmap, 1000, 1000)
    assert box is None
    assert reason in {"DIFFUSE_HEATMAP", "DEGENERATE_FULL_IMAGE"}


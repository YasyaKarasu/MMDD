"""Unit contracts for the frozen feature recipe in ``cache_stage1_features``.

``structural_table_pool_with_groups`` defines the per schema/row table tokens that the
content store and the feature-provenance probe depend on; these pin its arithmetic.
"""
from __future__ import annotations

import torch

from cache_stage1_features import structural_table_pool_with_groups


def test_structural_table_pool_returns_schema_and_row_tokens():
    hidden = torch.tensor([[1.0, 1.0], [3.0, 3.0], [6.0, 4.0]])
    groups = torch.tensor([0, 0, 1])

    pooled, pooled_groups = structural_table_pool_with_groups(hidden, groups)

    assert torch.equal(pooled, torch.tensor([[2.0, 2.0], [6.0, 4.0]]))
    assert torch.equal(pooled_groups, torch.tensor([0, 1]))


def test_structural_table_pool_keeps_contiguous_segments_per_group():
    hidden = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    pooled, pooled_groups = structural_table_pool_with_groups(hidden, groups, tokens_per_group=2)

    expected = torch.stack(
        [hidden[0:2].mean(0), hidden[2:4].mean(0), hidden[4:6].mean(0), hidden[6:8].mean(0)]
    )
    torch.testing.assert_close(pooled, expected)
    assert torch.equal(pooled_groups, torch.tensor([0, 0, 1, 1]))


def test_structural_table_pool_without_groups_is_identity():
    hidden = torch.randn(5, 3)
    pooled, pooled_groups = structural_table_pool_with_groups(hidden, None)
    assert pooled is hidden and pooled_groups is None

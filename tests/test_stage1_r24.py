"""Small tensor-level regressions for the R24 path objectives."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mmdd_stage1.scoring import ListScores, TargetScores
from mmdd_stage1.objectives import listwise_cross_entropy
from run_stage1_r24 import _fused_loss


def _scores(evidence_mask: torch.Tensor) -> TargetScores:
    direct_logits = torch.tensor([[1.0, 0.0, -1.0], [0.5, -0.5, -2.0]], requires_grad=True)
    candidate_mask = torch.ones_like(direct_logits, dtype=torch.bool)
    positive_indices = torch.tensor([0, 1], dtype=torch.long)
    direct = ListScores(direct_logits, candidate_mask, positive_indices)
    evidence_logits = torch.tensor([[2.0, -3.0, -4.0], [7.0, 6.0, 5.0]], requires_grad=True)
    evidence = ListScores(evidence_logits, evidence_mask, positive_indices)
    return TargetScores(direct=direct, evidence=evidence)


def test_fused_loss_falls_back_to_direct_when_no_evidence_path_exists():
    scores = _scores(torch.zeros(2, 3, dtype=torch.bool))
    fused = _fused_loss(scores)
    direct = listwise_cross_entropy(
        scores.direct.logits,
        scores.direct.positive_indices,
        scores.direct.candidate_mask,
    )
    assert torch.isfinite(fused)
    assert torch.allclose(fused, direct)


def test_fused_loss_has_finite_gradient_with_masked_evidence():
    scores = _scores(torch.tensor([[True, False, False], [True, True, False]]))
    loss = _fused_loss(scores)
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.direct.logits.grad is not None
    assert torch.isfinite(scores.direct.logits.grad).all()
    assert scores.evidence.logits.grad is not None
    assert torch.isfinite(scores.evidence.logits.grad).all()

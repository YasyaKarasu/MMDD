"""Focused regression tests for the R23 masked Teacher-KD repair."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from mmdd_stage1.data import EdgeExample
from mmdd_stage1.models import StudentJoinabilityModel
from mmdd_stage1.scoring import ListScores
from mmdd_stage1.training import _student_edge_losses


def _examples() -> list[EdgeExample]:
    return [
        EdgeExample("q_tt", ("t0", "t1"), 0, source_type="table", destination_type="table"),
        EdgeExample("q_non", ("t0", "t1"), 0, source_type="text", destination_type="table"),
    ]


def test_partial_teacher_mask_excludes_non_tt_rows_from_kd_gradient():
    model = StudentJoinabilityModel(2, 2, initialization="identity", freeze_projections=True)
    logits = torch.stack([model.relations["table_to_table"][0, :2], model.relations["table_to_table"][1, :2]])
    student = ListScores(logits, torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
    teacher = ListScores(
        torch.zeros(2, 2),
        torch.tensor([[True, True], [False, False]]),
        torch.zeros(2, dtype=torch.long),
    )
    terms = _student_edge_losses(
        model, _examples(), student, teacher, None, None,
        ranking_weight=0.0, temperature=1.0, distillation_weight=1.0,
        edge_bce_weight=0.0, anchor_weight=0.0, anchor_weight_evidence=0.0,
    )
    assert torch.isfinite(terms["loss"])
    terms["loss"].backward()
    gradient = model.relations["table_to_table"].grad
    assert gradient is not None
    assert torch.linalg.vector_norm(gradient[0, :2]) > 0
    assert torch.count_nonzero(gradient[1, :2]) == 0


def test_all_false_teacher_mask_is_zero_and_finite():
    model = StudentJoinabilityModel(2, 2, initialization="identity", freeze_projections=True)
    student = ListScores(torch.zeros(2, 2, requires_grad=True), torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
    teacher = ListScores(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
    terms = _student_edge_losses(
        model, _examples(), student, teacher, None, None,
        ranking_weight=0.0, temperature=1.0, distillation_weight=1.0,
        edge_bce_weight=0.0, anchor_weight=0.0, anchor_weight_evidence=0.0,
    )
    assert terms["distillation_loss"].item() == 0.0
    assert torch.isfinite(terms["loss"])

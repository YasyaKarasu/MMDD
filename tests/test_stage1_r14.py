from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdd_stage1.checkpoints import load_student
from mmdd_stage1.features import ObjectFeatures
from mmdd_stage1.models import StudentJoinabilityModel, add_projection_residual
from mmdd_stage1.scoring import ListScores, TargetScores
from mmdd_stage1.training import _student_path_losses, checkpoint


def _feature(object_id: str, object_type: str, values: list[float]) -> ObjectFeatures:
    return ObjectFeatures(object_id, object_type, torch.tensor(values))


def _scores(direct: torch.Tensor, evidence: torch.Tensor) -> TargetScores:
    mask = torch.ones_like(direct, dtype=torch.bool)
    positives = torch.tensor([[True, False]])
    indices = torch.zeros(1, dtype=torch.long)
    return TargetScores(
        ListScores(direct, mask, indices, positives),
        ListScores(evidence, mask, indices, positives),
    )


def test_projection_residual_is_step_zero_equivalent_and_checkpoint_round_trips(
    tmp_path: Path,
):
    torch.manual_seed(13)
    base = StudentJoinabilityModel(4, 4, initialization="identity")
    query = _feature("q", "table", [1.0, 2.0, -1.0, 0.5])
    image = _feature("i", "image", [0.25, -0.5, 1.5, 2.0])

    torch.manual_seed(17)
    residual = add_projection_residual(
        base, "gelu", hidden_dim=3, scales={"table": 0.5, "image": 2.0}
    )
    torch.testing.assert_close(
        residual.score_pairs([query], [image]),
        base.score_pairs([query], [image]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        residual.relation_query(query.embedding, "table", "image")
        @ residual.index_vector(image.embedding, "image"),
        residual.score_pairs([query], [image])[0],
    )

    path = tmp_path / "residual.pt"
    torch.save(checkpoint(residual, "student-path"), path)
    loaded = load_student(path, torch.device("cpu"))
    assert loaded.projection_adapter == "gelu"
    assert loaded.projection_scales == {"table": 0.5, "text": 1.0, "image": 2.0}
    torch.testing.assert_close(
        loaded.score_pairs([query], [image]),
        residual.score_pairs([query], [image]),
    )


def test_zero_output_initialization_delays_residual_input_gradient():
    base = StudentJoinabilityModel(2, 2, initialization="identity")
    residual = add_projection_residual(base, "linear", hidden_dim=2)
    embedding = torch.tensor([1.0, -2.0])

    residual.project(embedding, "table").sum().backward()
    assert torch.count_nonzero(
        residual.projection_residual_inputs["table"].weight.grad
    ) == 0
    assert torch.count_nonzero(
        residual.projection_residual_outputs["table"].weight.grad
    ) > 0


def test_evidence_loss_off_preserves_forward_but_removes_evidence_gradient():
    model = StudentJoinabilityModel(2, 2, initialization="identity")
    direct = torch.tensor([[2.0, 0.0]], requires_grad=True)
    evidence = torch.tensor([[0.5, -0.5]], requires_grad=True)
    scores = _scores(direct, evidence)
    full = _student_path_losses(
        model,
        scores,
        None,
        None,
        temperature=1.0,
        distillation_weight=0.3,
        anchor_weight=0.0,
        anchor_weight_evidence=0.0,
        distillation_rows=None,
    )
    ablated = _student_path_losses(
        model,
        scores,
        None,
        None,
        temperature=1.0,
        distillation_weight=0.3,
        anchor_weight=0.0,
        anchor_weight_evidence=0.0,
        distillation_rows=None,
        evidence_loss_weight=0.0,
    )
    direct_loss = full["direct_supervised_loss"]
    evidence_loss = full["evidence_supervised_loss"]
    torch.testing.assert_close(full["loss"], direct_loss + evidence_loss)
    torch.testing.assert_close(ablated["loss"], direct_loss)
    ablated["loss"].backward()
    assert torch.count_nonzero(direct.grad) > 0
    assert torch.count_nonzero(evidence.grad) == 0
